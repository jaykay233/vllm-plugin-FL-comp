"""Recursively SIGTERM a process and all its descendants.

Used to tear down a vllm serve tree (vllm + VLLM::EngineCore + workers) without
pattern-matching process names, so the killer can never match its own command
line.
"""

import os
import signal
import sys
import time


def children_of(ppid: int) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = open(f"/proc/{d}/stat").read()
            # comm may contain spaces/parens; state follows the last ') '
            state_ppid = stat.rsplit(") ", 1)[1].split()
            if int(state_ppid[1]) == ppid:
                out.append(int(d))
        except Exception:
            continue
    return out


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    try:
        root = int(sys.argv[1])
    except ValueError:
        return 2
    if root <= 1:
        return 0

    order, stack = [], [root]
    while stack:
        p = stack.pop()
        order.append(p)
        stack.extend(children_of(p))

    for p in reversed(order):
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    # vllm serve has been observed to ignore SIGTERM and linger (holding its
    # parent alive while the EngineCore becomes a zombie). Escalate.
    deadline = time.time() + 8
    while time.time() < deadline:
        if not os.path.exists(f"/proc/{root}"):
            return 0
        time.sleep(0.5)
    for p in reversed(order):
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
