#!/usr/bin/env bash
# Small-angle star cover pilot (detached; see RUNBOOK.md).
set -u
WS=/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D
JULIA=/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia
OUT=/mnt/ssd2T-2/xfliu/kappa3_redo/pilot
LB=$WS/LowerBound
mkdir -p "$OUT"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

run_points() {  # list output shard nshards
    nice -n 19 "$JULIA" --startup-file=no --project="$WS/Code" \
        "$LB/scripts/pilot_points.jl" "$1" "$2" "$3" "$4" >> "$OUT/points.log" 2>&1
}

for s in 0 1 2 3; do
    run_points "$LB/lists/pilot_L5.tsv" "$OUT/L5_shard$s.tsv" $s 4 &
done
run_points "$LB/lists/pilot_L6.tsv" "$OUT/L6.tsv" 0 1 &
wait
echo "[$(date -Is)] points done" >> "$OUT/points.log"

refs=("$OUT"/L5_shard*.tsv "$OUT/L6.tsv")
for hi in 5 10; do
    nice -n 19 "$JULIA" --startup-file=no --project="$WS/Code" \
        "$LB/scripts/pilot_cover.jl" "$hi" 10 300000 "${refs[@]}" \
        > "$OUT/cover_0_$hi.log" 2>&1
done
echo "[$(date -Is)] pilot finished" >> "$OUT/points.log"
