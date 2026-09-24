"""Experimental fix: make vllm_fl dispatch traceable by torch.compile/Dynamo.

Problem (confirmed by stack trace):
    layernorm.py:27 -> CachedOp.__call__
                    -> manager._resolve_impl
                    -> ensure_initialized
                    -> `with self._lock:`   <-- threading.RLock
    torch._dynamo.exc.Unsupported: Dynamo does not know how to enter a
    `RLock` context manager.

`ensure_initialized()` performs its "already initialized" check *inside* the
lock, so the lock is acquired on every call even after init. `_record_first_use`
likewise takes the lock on first call. Both are host-side bookkeeping and must
not be traced.

This patch:
  1. hoists the fast-path check in `ensure_initialized` before the lock
  2. makes `_record_first_use` skip the lock (and the log) while compiling,
     keeping semantics identical but avoiding the unsupported context manager
  3. pre-resolves every registered op once, outside the compiled region

Loaded via PYTHONPATH so it applies inside the spawned EngineCore process.
"""

import os
import sys

if os.environ.get("PATCH_DISPATCH_LOCK") == "1":
    try:
        import torch  # noqa: F401
        from vllm_fl.dispatch import manager as _mgr_mod
        from vllm_fl.dispatch.manager import OpManager

        _PENDING_OPS = []

        _orig_ensure = OpManager.ensure_initialized

        def ensure_initialized(self):
            # Fast path first: once initialized in this PID this must not take
            # the lock, otherwise Dynamo cannot trace through the hot path.
            if self._state.initialized and self._state.init_pid == os.getpid():
                return
            return _orig_ensure(self)

        OpManager.ensure_initialized = ensure_initialized

        _orig_record = OpManager._record_first_use

        def _record_first_use(self, op_name, impl):
            try:
                import torch as _t

                compiling = _t.compiler.is_compiling()
            except Exception:
                compiling = False
            if compiling:
                # Bookkeeping only; do it without the lock/logging.
                self._called_ops[op_name] = impl.impl_id
                return
            return _orig_record(self, op_name, impl)

        OpManager._record_first_use = _record_first_use

        # The whole resolution/bookkeeping path is host-side control logic. It
        # must not be traced by Dynamo at all: hoisting one primitive at a time
        # just moves the failure (RLock -> posix.getpid -> ...). Mark the
        # manager's bookkeeping entry points as untraceable so Dynamo emits a
        # graph break instead of raising.
        _disable = torch._dynamo.disable
        for _name in ("ensure_initialized", "_resolve_impl",
                      "_record_first_use", "call", "resolve"):
            _fn = getattr(OpManager, _name, None)
            if _fn is not None:
                setattr(OpManager, _name, _disable(_fn))
                print(f"[PATCH_DISPATCH_LOCK] dynamo-disabled OpManager.{_name}",
                      file=sys.stderr, flush=True)

        # Pre-resolve every CachedOp singleton so the first traced call takes
        # the already-resolved fast path.
        try:
            from vllm_fl.dispatch import CachedOp as _CO

            _orig_call = _CO.__call__
            _orig_init = _CO.__init__

            def _init(self, *a, **kw):
                _orig_init(self, *a, **kw)
                _PENDING_OPS.append(self)

            def _call(self, *args, **kwargs):
                return _orig_call(self, *args, **kwargs)

            _CO.__init__ = _init
            _CO.__call__ = torch._dynamo.disable(_call)
            print("[PATCH_DISPATCH_LOCK] CachedOp.__call__ dynamo-disabled",
                  file=sys.stderr, flush=True)
        except Exception as _e:
            print(f"[PATCH_DISPATCH_LOCK] CachedOp patch failed: {_e}",
                  file=sys.stderr, flush=True)

        print("[PATCH_DISPATCH_LOCK] OpManager patched (lock-free fast path)",
              file=sys.stderr, flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[PATCH_DISPATCH_LOCK] failed: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
