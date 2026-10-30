# kappa_3 recomputation on hpc (started 2026-10-29)

Everything runs on `hpc` as single-thread processes at nice 19, detached
(`setsid nohup`), restarted at reboot by two `@reboot` crontab lines.
`global_cover_run1` keeps running; we use its idle cores by explicit opt-in.

## 1. Proof campaign (upper bound, target kappa_3 <= 4/25)
* Code: `/home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D/Code`
* Root: `/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d` (`points/`, `checkpoints/`, `proof/`)
* Log: `/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d.queue.log`
* Done when `proof-bundle.txt` exists and the log says
  "proof campaign completed and independent replay passed".
* Status: `julia --project=Code Code/scripts/run_production.jl status`
* The superseded, never-computed 2026-10-26 initialization is kept at
  `/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d.superseded-init-20261026`.
* First 2026-10-29 run (159 points) stopped entering the cover phase: the
  relabelled Gram T*G*T' had one-ulp (i,j)/(j,i) asymmetry rejected by
  `strict_pd3`.  Fixed in `ProofRecords._transform_gram` and
  `ReferencePlan.transformed_gram` (compute upper triangle, mirror) with a
  regression test; restarted 12:30 JST in a fresh root.  The old run is kept
  at `...kappa3_cr_constant_3d.superseded-symfix-20261029`.

## 2. Auxiliary jobs (`LowerBound/scripts/run_redo.sh`, rerun safely)
Outputs in `/mnt/ssd2T-2/xfliu/kappa3_redo/{lower,upper,logs}`:
* `lower_daemon`: certified degree-10 Bernstein lower bound for every
  campaign point -> `lower/campaign_points_M10.tsv`.
* `lower_batches`: `exp1` (11 shapes, M=2..14), `small` (tiny equal-angle
  stars, M=10,12,14), `exp2` (160 samples, M=10), `diag` (eps=0.25..60).
* `upper_batches`: CR certificates via `Kappa3Certificate.point_certificate`
  for `exp1` (levels 2..5), `exp2` (level 5), `diag` (level 5).

## 3. Small-angle cover pilot (2026-10-29 17:30, not a proof record)
The star search stalls at angles <= 20 deg (level-4 references exceed 0.16
there; level 5 is < 0.16 only from 2 deg up).  `scripts/run_pilot.sh`
computes 143 level-5 references on {2,2.5,3,4,...,10}^3 and 3 level-6 ones
(`lists/pilot_L*.tsv`), then runs `pilot_cover.jl` on [0,5]^3 and [0,10]^3
at target 4/25.  Output: `/mnt/ssd2T-2/xfliu/kappa3_redo/pilot/`
(`L5_shard*.tsv`, `L6.tsv`, `cover_0_5.log`, `cover_0_10.log`).
Not restarted at reboot; rerun the script (points resume, covers redo).
Round 1 result: [0,5]^3 closes with one box; [0,10]^3 stalls only at the
flat edge (10,10,0); level 6 gives ~0.1677 (worse than level 5).
Round 2 (`run_pilot2.sh`, 20:15): 494 more level-5 references
(`lists/pilot2_L5.tsv`: grid to 15 deg, near-flat c=a+b-d, thin a<=1),
then [0,10]^3 again -> `pilot/R2_shard*.tsv`, `pilot/cover2_0_10.log`.
Round 2 result: [0,10]^3 closes with 8 boxes of side 5 deg.
Round 3 (`run_pilot3.sh`, 21:50): 825 more level-5 references on
{2,...,10,11,...,15,17.5,...,30,35,...,60}^3 (`lists/pilot3_L5.tsv`), then
the full [0,60]^3 star cover -> `pilot/R3_shard*.tsv`, `pilot/cover3_0_60.log`.

Round 3 result: full star [0,60]^3 closes at 4/25 (147 boxes, 41 refs)
and at **0.1598** (274 boxes, 73 refs, 43 s); 0.1597 fails only at the
fully degenerate corner (level-5 refs near 2 deg give ~0.15974).
User decision 2026-10-29: final target **0.1598** = 799/5000.
Round 4 (`run_pilot4.sh`, 23:15): 1155 level-5 path references
(`lists/pilot4_path_L5.tsv`: theta1<=theta2 on {2.5,...,60}, phi = 11
fractions of the |AD|=1 cap; chain reversal added in the search), then the
full path cover at 0.1598 and 4/25 -> `pilot/cover4_path_*.log`.
`pilot_cover.jl` now takes PILOT_FAMILY and PILOT_TARGET.
Round 4 result: path closes at 0.1598 (340 boxes, 127 refs), 0.155, 0.153
and **0.1520** (4247 boxes, 337 refs; `cover4_path_t1520.log`).  The 0.1515
run was stopped on 2026-10-30 by user decision: P* <= 0.1520 suffices.

**v2 rerun 2026-10-30 10:11 JST (GitHub dependencies).** For the public
repository xfliu/crouzeix-raviart-interpolation-constant-3d, `Code/setup.jl`
now installs VFEM (xfliu/vfem @ ee15fc7) and Veigs (xfliu/veigs, VEIGS.jl @
ddd17a2) from GitHub; their sources are byte-identical to the local copies.
`ProofRecords.dependency_root` finds the installed roots. Since
`Manifest.toml` is fingerprinted, the campaign was rerun in a fresh root;
the v1 records are kept at
`/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d.v1-localdeps-20261030`.

**PROOF COMPLETE 2026-10-30 02:02 JST**: `proof-bundle.txt` written and the
independent replay passed (star 799/5000: 274 leaves / 2192 corners; path
31/200: 1062 leaves / 8496 corners; 319 replayed point proofs).

**Campaign restarted 2026-10-30 00:24 JST with the level-5 catalogue.**
User decision: star target 799/5000 (0.1598), path target 31/200 (0.155).
Code changes (backup of the old tree:
`/mnt/ssd2T-2/xfliu/code_backup_20261030/Code_before_level5_catalogue.tgz`):
`BoxCover` STAR_TARGET/PATH_TARGET/family_target; `ProductionRunner` and
`scripts/replay_proof.jl` use the per-family targets; `ReferencePlan`
catalogue = the 73 star + 137 path references used by the pilot covers
(`pilot/catalogue_{star,path}.tsv`), all level 5, enrichment level 5;
queue step budget 32 points.  All test suites pass.  Old root kept at
`...kappa3_cr_constant_3d.superseded-level4-catalogue-20261029`; the
`@reboot` line is re-enabled.  The lower-bound daemon was restarted with a
fresh `lower/campaign_points_M10.tsv` (point ids were reused).

**Campaign paused 2026-10-29 21:45** (user decision): step 21 killed, its
`@reboot` crontab line commented out ("# PAUSED"); backup
`/mnt/ssd2T-2/xfliu/crontab.backup.20261029b`.  Its level-4-heavy catalogue
cannot close the small-angle star region; the catalogue will be rebuilt from
the pilot's level-5 grid and the campaign restarted in a fresh root.

Check progress: `wc -l /mnt/ssd2T-2/xfliu/kappa3_redo/*/*.tsv` and
`tail /mnt/ssd2T-2/xfliu/kappa3_redo/logs/*.log`.

## Method of the lower bounds (`LowerBound/src/BernsteinLB.jl`)
Bernstein mass/stiffness kernels are exact rationals; the four face-mean
constraints are integer equations sum_{alpha_k=0} c_alpha = 0, solved
exactly for the vertex coefficients; the only interval quantity is the
gradient metric (256-bit). C(K) >= sqrt(mass/energy), rounded down.
Validated against the paper's independent values (S1, P6, star 1/5 deg,
path witness) to 1e-11.
