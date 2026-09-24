#!/usr/bin/env python
"""Switch metax.yaml between 'whitelist' (ship default) and 'off' (no operator
selection at all).

  whitelist -> flagos_whitelist: [silu_and_mul, rms_norm, rotary_embedding]
  off       -> flagos_whitelist: []      (empty => the framework ignores it)

An empty list is falsy in vllm_fl.utils, so it is identical to having no
whitelist while keeping the YAML valid.
"""

import re
import sys

YAML = ("/workspace/vllm-plugin-FL/vllm_fl/dispatch/config/metax.yaml")
WL_OPS = ["silu_and_mul", "rms_norm", "rotary_embedding"]


def main() -> int:
    mode = sys.argv[1]
    src = open(YAML).read()

    # match the key (with an optional inline value) plus any indented list items
    pat = re.compile(r"^flagos_whitelist:(?![ \t]*\[\])[^\n]*\n(?:[ \t]+-[^\n]*\n)*"
                     r"|^flagos_whitelist:[ \t]*\[\][ \t]*\n",
                     re.M)
    if not pat.search(src):
        print(f"ERROR: flagos_whitelist block not found in {YAML}")
        return 1

    if mode == "off":
        repl = "flagos_whitelist: []\n"
    elif mode == "whitelist":
        items = "".join(f"  - {o}\n" for o in WL_OPS)
        repl = f"flagos_whitelist:\n{items}"
    else:
        print(f"ERROR: unknown mode {mode!r}")
        return 1

    pat.sub(repl, src, count=1)
    open(YAML, "w").write(pat.sub(repl, src, count=1))
    print(f"  metax.yaml -> {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
