"""Is the enable-oplist file a valid proxy for the whitelist?

The worker calls flag_gems.only_enable(include=<whitelist>, record=True,
path=<oplist>) and the resulting file listed 36 ops (including `linear`, `mul`,
`softmax`) even though use_flaggems_op() says those must NOT use FlagGems.

This script reproduces that exact call with a 3-op include and then invokes a
few ops that are NOT in the include list. If those still get logged, the oplist
records invocations rather than registrations and is therefore NOT a valid
whitelist proxy.
"""

import os

PATH = "/tmp/only_enable_probe.txt"
os.environ["FLAGGEMS_ENABLE_OPLIST_PATH"] = PATH

import torch  # noqa: E402

import flag_gems  # noqa: E402

INCLUDE = ["silu_and_mul", "rms_norm", "rotary_embedding"]

if os.path.exists(PATH):
    os.remove(PATH)

print("calling only_enable(include=3 ops, record=True)", flush=True)
flag_gems.only_enable(include=INCLUDE, record=True, once=True, path=PATH)

# Ops that are NOT in the include list. If the oplist is a registration proxy,
# none of these may show up.
a = torch.randn(8, 16, device="cuda")
b = torch.randn(16, 32, device="cuda")
torch.mul(a, 2.0)
torch.mm(a, b)
torch.softmax(a, dim=-1)
print("ran mul / mm / softmax", flush=True)

torch.cuda.synchronize()

lines = []
if os.path.exists(PATH):
    with open(PATH) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

print(f"\noplist lines: {len(lines)}")
names = sorted({ln.split(": ", 1)[-1] for ln in lines})
print("distinct ops logged:", names)

excluded_logged = [n for n in names if any(k in n for k in ("MUL", "MM", "SOFTMAX"))]
print(f"\nNOT-in-include ops that still got logged: {excluded_logged}")
if excluded_logged:
    print("  => oplist records *invocations*, NOT registrations; NOT a whitelist proxy")
else:
    print("  => oplist looks like a registration proxy")
