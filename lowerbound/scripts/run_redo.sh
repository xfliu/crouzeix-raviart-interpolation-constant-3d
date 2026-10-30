#!/usr/bin/env bash
# Start (or resume) the auxiliary recomputation jobs on hpc, each as a
# detached single-thread process at nice 19 holding its own flock.  Safe to
# rerun: a job whose lock is held is skipped, and every batch skips rows it
# has already written.
set -uo pipefail

readonly JULIA="/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia"
readonly LB="/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D/LowerBound"
readonly CODE="/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D/Code"
readonly CAMPAIGN="/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d"
readonly OUT="/mnt/ssd2T-2/xfliu/kappa3_redo"
mkdir -p "$OUT/lower" "$OUT/upper" "$OUT/logs" "$OUT/locks"

export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

start_job() {   # name, command...
    local name="$1"; shift
    local lock="$OUT/locks/$name.lock"
    if flock -n "$lock" true; then
        nohup setsid flock -n "$lock" nice -n 19 "$@" \
            >> "$OUT/logs/$name.log" 2>&1 < /dev/null &
        echo "started $name"
    else
        echo "skip $name (already running)"
    fi
}

start_job lower_daemon "$JULIA" --startup-file=no --project="$LB" \
    "$LB/scripts/lower_daemon.jl" "$CAMPAIGN" "$OUT/lower/campaign_points_M10.tsv" 10

start_job lower_batches bash -c "
  for pair in exp1_lower:exp1 small_lower:small exp2_samples:exp2 diag_lower:diag; do
    in=\${pair%%:*}; out=\${pair##*:}
    \"$JULIA\" --startup-file=no --project=\"$LB\" \"$LB/scripts/lower_batch.jl\" \
        \"$LB/lists/\$in.tsv\" \"$OUT/lower/\$out.tsv\" || echo \"lower batch \$in exited \$?\"
  done; echo lower_batches finished"

start_job upper_batches bash -c "
  for pair in exp1_upper:exp1 exp2_samples_upper:exp2 diag_upper:diag; do
    in=\${pair%%:*}; out=\${pair##*:}
    \"$JULIA\" --startup-file=no --project=\"$CODE\" \"$LB/scripts/upper_batch.jl\" \
        \"$LB/lists/\$in.tsv\" \"$OUT/upper/\$out.tsv\" || echo \"upper batch \$in exited \$?\"
  done; echo upper_batches finished"
