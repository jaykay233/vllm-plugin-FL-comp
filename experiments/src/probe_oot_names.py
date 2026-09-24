#!/usr/bin/env python3
"""Why does `custom_ops=["+rms_norm", ...]` not switch the FL OOT ops on?

vLLM matches `custom_ops` entries against `CustomOp.name`:
    enabled = f"+{cls.name}" in custom_ops
The FL plugin registers OOT classes through
    CustomOp.register_oot(cls, name=registration_name)
which *overwrites* `cls.name` with the registration name.  This script prints
the resulting names and what `+op` spelling would actually be needed.

Run:  conda activate mx && python /root/src/probe_oot_names.py
"""

from __future__ import annotations

import torch  # noqa: F401


def main() -> None:
    from vllm.config.compilation import CompilationConfig

    cc = CompilationConfig(custom_ops=["+rms_norm", "+silu_and_mul", "+rotary_embedding"])
    # mirror what vllm/config/vllm.py does for inductor
    if all(s not in cc.custom_ops for s in ("all", "none")):
        cc.custom_ops.append("none")
    print("custom_ops after vllm defaulting :", cc.custom_ops)

    from vllm_fl.ops.custom_ops import OOT_OPS, register_oot_ops
    from vllm.model_executor.custom_op import op_registry_oot

    register_oot_ops()

    print()
    print(f"{'op_name (FL)':<22}{'reg_name':<22}{'cls.__name__':<18}{'cls.name':<18}"
          f"{'+cls.name matches?':<22}{'enabled()'}")
    print("-" * 118)
    for op_name, (cls, reg_name) in OOT_OPS.items():
        cls_name = getattr(cls, "name", None)
        matches = cc.is_custom_op_enabled(cls_name) if isinstance(cls_name, str) else None
        # what CustomOp.enabled() computes
        enabled = (cc.is_custom_op_enabled("all") if "all" in cc.custom_ops else False) or (
            f"+{cls_name}" in cc.custom_ops
        )
        print(f"{op_name:<22}{reg_name:<22}{cls.__name__:<18}{str(cls_name):<18}"
              f"{str(matches):<22}{enabled}")

    print()
    print("op_registry_oot keys :", sorted(op_registry_oot.keys()))
    print("would vLLM accept '+rms_norm'? ",
          cc.is_custom_op_enabled("rms_norm"),
          "   (op_registry_oot lookup key used by __new__ is the *class name*)")
    print()
    print("PROBE_OOT_NAMES_DONE", flush=True)


if __name__ == "__main__":
    main()
