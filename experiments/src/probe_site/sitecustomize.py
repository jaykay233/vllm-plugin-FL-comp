"""Installed via PYTHONPATH so it runs inside the spawned EngineCore process.

Logs, for every vLLM CustomOp instantiation, which forward method vLLM picked:
    forward_oot   -> plugin dispatch -> FlagGems   (what we want)
    forward_native-> inductor-compiled torch       (FlagGems bypassed)
"""

import os
import sys

# experimental dispatch-lock patch (see patch_dispatch.py)
if os.environ.get("PATCH_DISPATCH_LOCK") == "1":
    try:
        import patch_dispatch  # noqa: F401
    except Exception as _e:  # pragma: no cover
        print(f"[sitecustomize] patch_dispatch import failed: {_e}",
              file=sys.stderr, flush=True)

if os.environ.get("PROBE_DISPATCH_CHOICE") == "1":
    try:
        from vllm.model_executor.custom_op import CustomOp

        _orig = CustomOp.dispatch_forward

        def patched(self, *args, **kwargs):
            fn = _orig(self, *args, **kwargs)
            cls = type(self)
            name = getattr(cls, "name", None)
            try:
                enabled = self.enabled()
            except Exception as e:  # pragma: no cover
                enabled = f"err:{type(e).__name__}"
            has_name = hasattr(cls, "name")
            print(
                f"[PROBE_DISPATCH] cls={cls.__name__} name={name!r} has_name={has_name} "
                f"enabled={enabled} chosen={getattr(fn, '__name__', fn)}",
                file=sys.stderr,
                flush=True,
            )
            return fn

        CustomOp.dispatch_forward = patched

        from vllm.model_executor.custom_op import get_cached_compilation_config

        try:
            cc = get_cached_compilation_config()
            print(
                f"[PROBE_DISPATCH] compilation_config custom_ops={cc.custom_ops} "
                f"mode={cc.mode}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass

    except Exception as e:  # pragma: no cover
        print(f"[PROBE_DISPATCH] install failed: {e}", file=sys.stderr, flush=True)
