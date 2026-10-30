#!/usr/bin/env bash
# Pilot round 4: level-5 path references, then the full path cover at 0.1598
# (detached; see RUNBOOK.md).
set -u
WS=/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D
JULIA=/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
OUT=/mnt/ssd2T-2/xfliu/kappa3_redo/pilot
LB=$WS/LowerBound
mkdir -p "$OUT"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

for s in 0 1 2 3 4 5 6 7; do
    nice -n 19 "$JULIA" --startup-file=no --project="$WS/Code" \
        "$LB/scripts/pilot_points.jl" "$LB/lists/pilot4_path_L5.tsv" \
        "$OUT/R4_shard$s.tsv" $s 8 >> "$OUT/points4.log" 2>&1 &
done
wait
echo "[$(date -Is)] round-4 points done" >> "$OUT/points4.log"

for t in 1598/10000 4/25; do
    PILOT_FAMILY=path PILOT_TARGET=$t nice -n 19 "$JULIA" --startup-file=no \
        --project="$WS/Code" "$LB/scripts/pilot_cover.jl" 60 40 300000 \
        "$OUT"/R4_shard*.tsv > "$OUT/cover4_path_${t%%/*}.log" 2>&1
done
echo "[$(date -Is)] round-4 pilot finished" >> "$OUT/points4.log"
