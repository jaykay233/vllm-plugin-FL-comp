# Copyright (c) 2025 BAAI. All rights reserved.
"""Guard vLLM's post-grad pass manager on out-of-tree platforms.

``PostGradPassManager.configure`` (``vllm/compilation/passes/pass_manager.py``)
appends a number of platform-specific fusion passes whose *imports* are gated on
``current_platform.is_cuda_alike()`` / ``is_xpu()`` / ``is_cuda()`` but whose
*use sites* are not gated at all.  On an out-of-tree platform such as
``vllm_fl.platform.PlatformFL`` (all three predicates are ``False``) those names
are therefore undefined, so a stock ``vllm serve`` that enables compilation
aborts during ``profile_run`` with, for example::

    NameError: name 'RMSNormQuantFusionPass' is not defined

This is reachable with no special flags at all.  FL registers ``vllm.ir``
providers, so ``kernel_config.ir_op_priority.rms_norm[0]`` is not ``"native"``,
which makes ``vllm.config.vllm.enable_norm_fusion()`` return true and thus
``pass_config.fuse_norm_quant`` default to ``True`` on optimization level O2+.
Other flags (``enable_sp``, ``fuse_act_quant``, ``fuse_rope_kvcache``, ...) reach
equally unguarded names, so the fix must be generic rather than one-name deep.

Re-importing the classes is not an option: the fusion modules touch
``torch.ops._C.*_fp8*`` and ``current_platform.fp8_dtype()`` at module scope, and
``+empty`` builds ship no ``vllm._C``.  Their kernels would also be the wrong
implementation for this backend.

Instead we install a no-op stand-in for every pass class this vLLM build did not
import.  Reporting ``is_applicable_for_range() -> False`` means the stand-in is
never executed, so the observable behaviour matches vLLM's own intent for a
platform where the class was conditionally not imported: the pass does not run.

The injected names are discovered from the bytecode of ``configure`` itself, so
this keeps working if vLLM adds or renames guarded passes.
"""

from __future__ import annotations

import logging
import types

logger = logging.getLogger(__name__)

# Set on PostGradPassManager once the guard is installed, so repeated plugin
# loads (the platform plugin is loaded in every spawned process) stay cheap.
_SENTINEL = "_vllm_fl_post_grad_pass_guard"


def _global_names(func) -> set[str]:
    """Collect every global name referenced by ``func``, including nested code."""
    names: set[str] = set()
    stack = [func.__code__]
    while stack:
        code = stack.pop()
        names.update(code.co_names)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)
    return names


def _make_standin_pass():
    """Build a never-applicable ``VllmInductorPass`` subclass.

    Subclassing the real base keeps ``isinstance`` checks, ``uuid()`` (used for
    the Inductor code-cache key) and the ``dump_prefix`` bookkeeping in
    ``PostGradPassManager.__call__`` working unchanged.
    """
    from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass

    class _FLUnavailablePass(VllmInductorPass):
        """Placeholder for a pass this vLLM build did not import.

        Never applies to any compile range, hence never runs.
        """

        def is_applicable_for_range(self, compile_range) -> bool:
            return False

        def __call__(self, graph) -> None:
            return None

    return _FLUnavailablePass


def patch_post_grad_pass_manager() -> bool:
    """Make unimported post-grad passes inert instead of fatal.

    Returns ``True`` when at least one stand-in was installed, ``False`` when
    there was nothing to do or the hook could not be applied.
    """
    try:
        from vllm.compilation.passes import pass_manager as pm
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("post-grad pass manager guard skipped: %s", exc)
        return False

    manager = getattr(pm, "PostGradPassManager", None)
    if manager is None or getattr(manager, _SENTINEL, False):
        return False

    configure = getattr(manager, "configure", None)
    if configure is None:  # pragma: no cover - defensive
        return False

    missing = sorted(
        name
        for name in _global_names(configure)
        if name.endswith("Pass") and not hasattr(pm, name)
    )
    if not missing:
        # This platform imported every guarded pass; nothing to guard.
        return False

    try:
        standin = _make_standin_pass()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("post-grad pass manager guard skipped: %s", exc)
        return False

    for name in missing:
        setattr(pm, name, standin)

    setattr(manager, _SENTINEL, True)
    logger.info(
        "vllm_fl: installed inert stand-ins for %d post-grad pass(es) this "
        "vLLM build did not import: %s",
        len(missing),
        ", ".join(missing),
    )
    return True
