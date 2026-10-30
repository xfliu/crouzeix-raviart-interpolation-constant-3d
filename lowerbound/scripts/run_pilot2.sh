#!/usr/bin/env bash
# Pilot round 2: references to 15 deg plus near-flat and thin references,
# then the [0,10]^3 star cover again (detached; see RUNBOOK.md).
set -u
WS=/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D
JULIA=/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
OUT=/mnt/ssd2T-2/xfliu/kappa3_redo/pilot
LB=$WS/LowerBound
mkdir -p "$OUT"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

for s in 0 1 2 3 4 5; do
    nice -n 19 "$JULIA" --startup-file=no --project="$WS/Code" \
        "$LB/scripts/pilot_points.jl" "$LB/lists/pilot2_L5.tsv" \
        "$OUT/R2_shard$s.tsv" $s 6 >> "$OUT/points2.log" 2>&1 &
done
wait
echo "[$(date -Is)] round-2 points done" >> "$OUT/points2.log"

nice -n 19 "$JULIA" --startup-file=no --project="$WS/Code" \
    "$LB/scripts/pilot_cover.jl" 10 10 300000 \
    "$OUT"/L5_shard*.tsv "$OUT"/R2_shard*.tsv \
    > "$OUT/cover2_0_10.log" 2>&1
echo "[$(date -Is)] round-2 pilot finished" >> "$OUT/points2.log"
