#!/usr/bin/env bash
set -uo pipefail

# Conservative unattended driver for the resumable proof campaign.  This
# process only polls while the unrelated global computation is active; every
# numerical invocation still passes through launch_production.sh and its
# blocker/resource/provenance checks.

readonly CODE_ROOT="/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D/Code"
readonly LAUNCHER="$CODE_ROOT/scripts/launch_production.sh"
readonly JULIA_BIN="/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia"
readonly STATUS_SCRIPT="$CODE_ROOT/scripts/run_production.jl"
readonly QUEUE_LOCK="/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d.queue.lock"

poll_seconds="${KAPPA3_QUEUE_POLL_SECONDS:-300}"

usage() {
    echo "Usage: $0 [--check-only]" >&2
}

if [[ ! "$poll_seconds" =~ ^[0-9]+$ ]] ||
   (( poll_seconds < 30 || poll_seconds > 3600 )); then
    echo "REFUSED: KAPPA3_QUEUE_POLL_SECONDS must be an integer in 30:3600." >&2
    exit 64
fi

if (( $# > 1 )); then
    usage
    exit 64
fi
if (( $# == 1 )); then
    if [[ "$1" == "--check-only" ]]; then
        exec "$LAUNCHER" --check-only
    fi
    usage
    exit 64
fi

if [[ ! -x "$LAUNCHER" ]]; then
    echo "REFUSED: guarded production launcher is absent or not executable: $LAUNCHER" >&2
    exit 66
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "REFUSED: flock is required for a crash-safe queue lock." >&2
    exit 69
fi
if ! exec 8>"$QUEUE_LOCK"; then
    echo "REFUSED: cannot open queue lock: $QUEUE_LOCK" >&2
    exit 73
fi
if ! flock -n 8; then
    echo "REFUSED: another kappa3 queue driver holds $QUEUE_LOCK" >&2
    exit 73
fi

stamp() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

trap 'stamp "queue driver stopped by signal"; exit 130' INT TERM HUP

wait_until_ready() {
    local status
    while true; do
        "$LAUNCHER" --check-only
        status=$?
        case "$status" in
            0)
                stamp "hpc resource guard is ready"
                return 0
                ;;
            75)
                stamp "hpc is not ready; checking again in ${poll_seconds}s"
                sleep "$poll_seconds"
                ;;
            *)
                stamp "fatal readiness-check status $status"
                return "$status"
                ;;
        esac
    done
}

campaign_status() {
    env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        "$JULIA_BIN" --startup-file=no --project="$CODE_ROOT" \
        "$STATUS_SCRIPT" status
}

stamp "queue driver started; no computation will begin until the guard is ready"
wait_until_ready || exit $?

while true; do
    "$LAUNCHER" initialize
    status=$?
    case "$status" in
        0)
            stamp "campaign provenance initialized or verified"
            break
            ;;
        73)
            stamp "initialization lock is busy; retrying in ${poll_seconds}s"
            sleep "$poll_seconds"
            wait_until_ready || exit $?
            ;;
        75)
            stamp "initialization deferred by the resource guard"
            wait_until_ready || exit $?
            ;;
        *)
            stamp "fatal initialization status $status"
            exit "$status"
            ;;
    esac
done

previous_status="$(campaign_status)" || {
    stamp "fatal: could not inspect initialized campaign"
    exit 70
}
step=0
unchanged_steps=0
while true; do
    wait_until_ready || exit $?
    step=$((step + 1))
    stamp "starting bounded production step $step"
    "$LAUNCHER" run \
        --max-point-solves=32 \
        --max-search-nodes=200000 \
        --max-enrichments=8 \
        --max-cycles=2
    status=$?
    case "$status" in
        0)
            stamp "proof campaign completed and independent replay passed"
            exit 0
            ;;
        73)
            stamp "production lock is busy; retrying in ${poll_seconds}s"
            sleep "$poll_seconds"
            ;;
        75)
            # A completed bounded invocation uses 75 to mean "checkpointed,
            # not yet proved".  A launcher-side resource refusal uses the
            # same temporary status; the readiness loop distinguishes it and
            # sleeps only when necessary.
            stamp "bounded step $step is incomplete or temporarily deferred"
            current_status="$(campaign_status)" || {
                stamp "fatal: could not inspect resumable campaign"
                exit 70
            }
            if [[ "$current_status" == "$previous_status" ]]; then
                unchanged_steps=$((unchanged_steps + 1))
                stamp "campaign summary was unchanged ($unchanged_steps/3)"
                if (( unchanged_steps >= 3 )); then
                    stamp "fatal: three ready retries made no checkpoint progress"
                    exit 74
                fi
                sleep "$poll_seconds"
            else
                unchanged_steps=0
                previous_status="$current_status"
            fi
            ;;
        *)
            stamp "fatal production status $status; preserved checkpoints for diagnosis"
            exit "$status"
            ;;
    esac
done
