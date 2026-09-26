#!/bin/bash
# Apply / revert / check the [stage-prof] EngineCore instrumentation patch.
#
# The patch edits site-packages/vllm/v1/engine/core.py IN PLACE. It is not part
# of any commit, so a vLLM reinstall or reinstall-into-site-packages silently
# wipes the probes away. That is exactly what happened once already: the driver
# scripts survived but the probes were gone, and nobody noticed until a run
# produced no window lines at all. So always run `status` first.
#
# Usage:
#   stage_prof_patch.sh status    # applied or not (run this first!)
#   stage_prof_patch.sh apply     # idempotent apply + py_compile check
#   stage_prof_patch.sh revert    # undo
#   stage_prof_patch.sh verify    # round-trip apply/revert on this install
set -uo pipefail

SITE="${SITE:-/opt/conda/envs/mx/lib/python3.12/site-packages}"
PY="${PY:-/opt/conda/envs/mx/bin/python}"
TARGET="$SITE/vllm/v1/engine/core.py"
PATCH="$(cd "$(dirname "$0")/../patches" && pwd)/vllm_iter_stage_profile.patch"
MARKER="_STAGE_PROFILE_ENABLED"

usage() { sed -n '2,12p' "$0"; exit 1; }
[ $# -ge 1 ] || usage
[ -f "$PATCH" ] || { echo "patch file missing: $PATCH"; exit 2; }

applied() { grep -q "$MARKER" "$TARGET" 2>/dev/null; }

case "$1" in
  status)
    [ -f "$TARGET" ] || { echo "MISSING: $TARGET"; exit 2; }
    if applied; then
      echo "APPLIED    : $TARGET"
      printf "             probes: %s\n" "$(grep -c '_sp\.add(' "$TARGET")"
      printf "             emit calls: %s\n" "$(grep -c '_stage_emit(' "$TARGET")"
      printf "             sink: %s\n" \
        "$(grep -o 'VLLM_ITER_STAGE_FILE[^)]*' "$TARGET" | head -1)"
    else
      echo "NOT APPLIED: $TARGET"
      echo "             run '$0 apply' before any VLLM_ITER_STAGE_PROFILE run,"
      echo "             otherwise the server will just log nothing and look fine."
      exit 1
    fi
    ;;
  apply)
    if applied; then echo "already applied"; exit 0; fi
    patch -p1 -d "$SITE" < "$PATCH" || exit 1
    if ! "$PY" -m py_compile "$TARGET"; then
      echo "!! py_compile failed, reverting"
      patch -R -p1 -d "$SITE" < "$PATCH"
      exit 1
    fi
    echo "applied; py_compile OK"
    ;;
  revert)
    if ! applied; then echo "not applied"; exit 0; fi
    patch -R -p1 -d "$SITE" < "$PATCH" && echo "reverted"
    ;;
  verify)
    tmp="$(mktemp -d)"
    cp "$TARGET" "$tmp/before.py"
    was_applied=0; applied && was_applied=1
    if [ "$was_applied" = 0 ]; then
      patch -p1 -d "$SITE" < "$PATCH" >/dev/null || { echo "apply failed"; rm -rf "$tmp"; exit 1; }
    fi
    cp "$TARGET" "$tmp/patched.py"
    patch -R -p1 -d "$SITE" < "$PATCH" >/dev/null || { echo "reverse failed"; rm -rf "$tmp"; exit 1; }
    cp "$TARGET" "$tmp/pristine.py"
    patch -p1 -d "$SITE" < "$PATCH" >/dev/null || { echo "re-apply failed"; rm -rf "$tmp"; exit 1; }
    if diff -q "$tmp/patched.py" "$TARGET" >/dev/null; then
      echo "round-trip OK ($(grep -c '_sp\.add(' "$TARGET") probes)"
    else
      echo "!! round-trip differs:"
      diff -u "$tmp/patched.py" "$TARGET" | head -30
    fi
    "$PY" -m py_compile "$TARGET" && echo "py_compile OK"
    [ "$was_applied" = 0 ] && { patch -R -p1 -d "$SITE" < "$PATCH"; echo "(restored to unpatched)"; }
    rm -rf "$tmp"
    ;;
  *) usage ;;
esac
