#!/usr/bin/env bash
set -euo pipefail

readonly CANONICAL_SCRATCH_ROOT="/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d"
readonly CODE_ROOT="/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D/Code"
readonly JULIA_BIN="/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia"
readonly BLOCKER_PATTERN='/mnt/ssd2T-2/xfliu/pgs_runs/[g]lobal_cover_run1'

scratch_root="${KAPPA3_SCRATCH_ROOT:-$CANONICAL_SCRATCH_ROOT}"
production_script="${KAPPA3_PRODUCTION_SCRIPT:-$CODE_ROOT/scripts/run_production.jl}"
max_load="${KAPPA3_MAX_LOAD1:-36}"
min_memory_gib="${KAPPA3_MIN_MEMORY_GIB:-64}"
min_disk_gib="${KAPPA3_MIN_DISK_GIB:-100}"
check_only=0

if [[ "${1:-}" == "--check-only" ]]; then
    check_only=1
    shift
fi

# ProductionRunner itself accepts exactly this campaign root.  Check the raw
# spelling before any mkdir so paths containing `..`, descendants that the
# Julia layer would reject, and symlink aliases cannot cause side effects.
if [[ "$scratch_root" != "$CANONICAL_SCRATCH_ROOT" ]]; then
    echo "REFUSED: scratch path must be exactly $CANONICAL_SCRATCH_ROOT" >&2
    exit 64
fi

if pgrep -u "$(id -u)" -af "$BLOCKER_PATTERN" >/dev/null 2>&1; then
    # Opt-in (2026-10-29, owner's decision): run this single niced process
    # on the cores global_cover_run1 leaves idle instead of waiting for it.
    if [[ "${KAPPA3_ALLOW_CONCURRENT_GLOBAL_COVER:-0}" != "1" ]]; then
        echo "REFUSED: unrelated global_cover_run1 processes are still active." >&2
        echo "Wait for /mnt/ssd2T-2/xfliu/pgs_runs/global_cover_run1 to finish." >&2
        exit 75
    fi
    echo "NOTE: global_cover_run1 is active; running concurrently at nice 19 by opt-in." >&2
fi

if [[ ! -x "$JULIA_BIN" ]]; then
    echo "REFUSED: pinned Julia executable is absent: $JULIA_BIN" >&2
    exit 69
fi
if [[ ! -f "$production_script" ]]; then
    echo "REFUSED: production entry point is absent: $production_script" >&2
    exit 66
fi
if [[ ! -d /mnt/ssd2T-2/xfliu ]]; then
    echo "REFUSED: hpc scratch mount is unavailable." >&2
    exit 69
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "REFUSED: flock is required for a crash-safe production lock." >&2
    exit 69
fi

load1="$(awk '{print $1}' /proc/loadavg)"
if ! awk -v load="$load1" -v limit="$max_load" 'BEGIN { exit !(load < limit) }'; then
    echo "REFUSED: one-minute load $load1 is not below the limit $max_load." >&2
    exit 75
fi

available_kib="$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo)"
required_kib="$((min_memory_gib * 1024 * 1024))"
if (( available_kib < required_kib )); then
    echo "REFUSED: less than ${min_memory_gib} GiB memory is available." >&2
    exit 75
fi

available_disk_kib="$(df -Pk /mnt/ssd2T-2 | awk 'NR == 2 {print $4}')"
required_disk_kib="$((min_disk_gib * 1024 * 1024))"
if (( available_disk_kib < required_disk_kib )); then
    echo "REFUSED: less than ${min_disk_gib} GiB hpc scratch space is available." >&2
    exit 75
fi

if (( check_only )); then
    echo "READY: blocker, load, memory, disk, Julia, and entry-point checks passed."
    exit 0
fi

mkdir -p "$scratch_root"
resolved_root="$(realpath "$scratch_root")"
if [[ "$resolved_root" != "$CANONICAL_SCRATCH_ROOT" ]]; then
    echo "REFUSED: canonical scratch root resolves to an unexpected location." >&2
    exit 64
fi

lock_file="$scratch_root/.production.lockfile"
if ! exec 9>"$lock_file"; then
    echo "REFUSED: cannot open production lock file: $lock_file" >&2
    exit 73
fi
if ! flock -n 9; then
    echo "REFUSED: another production runner holds $lock_file" >&2
    exit 73
fi
# The descriptor is inherited by the Julia child, so even a killed launcher
# cannot release the lock while its computation is still running.  The
# kernel releases it automatically after the final process exits; no stale
# directory needs manual cleanup after SIGKILL or a node failure.

threads="${KAPPA3_JULIA_THREADS:-1}"
if (( threads != 1 )); then
    echo "REFUSED: this proof runner is sequential; KAPPA3_JULIA_THREADS must be 1." >&2
    exit 64
fi

echo "Starting sequential production; Julia, BLAS, and OpenMP threads fixed at 1."
env JULIA_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    KAPPA3_LAUNCHER_GUARD=passed \
    KAPPA3_SCRATCH_ROOT="$scratch_root" \
    nice -n 19 "$JULIA_BIN" --startup-file=no --project="$CODE_ROOT" \
    "$production_script" "$@"
