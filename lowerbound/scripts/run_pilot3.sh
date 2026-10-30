#!/usr/bin/env bash
# Pilot round 3: level-5 grid over the whole star domain, then the full
# [0,60]^3 star cover at 4/25 (detached; see RUNBOOK.md).
set -u
WS=/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D
JULIA=/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
OUT=/mnt/ssd2T-2/xfliu/kappa3_redo/pilot
LB=$WS/LowerBound
mkdir -p "$OUT"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

for s in 0 1 2 3 4 5 6 7; do
    nice -n 19 "$JULIA" --startup-file=no --project="$WS/Code" \
        "$LB/scripts/pilot_points.jl" "$LB/lists/pilot3_L5.tsv" \
        "$OUT/R3_shard$s.tsv" $s 8 >> "$OUT/points3.log" 2>&1 &
done
wait
echo "[$(date -Is)] round-3 points done" >> "$OUT/points3.log"

nice -n 19 "$JULIA" --startup-file=no --project="$WS/Code" \
    "$LB/scripts/pilot_cover.jl" 60 10 300000 \
    "$OUT"/L5_shard*.tsv "$OUT"/R2_shard*.tsv "$OUT"/R3_shard*.tsv \
    > "$OUT/cover3_0_60.log" 2>&1
echo "[$(date -Is)] round-3 pilot finished" >> "$OUT/points3.log"
