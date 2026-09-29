# M2 structured-v2 reference production after verified readiness 990

This is the active runbook. Do not use any previous M2 full-run launcher.

## Evidence and scope

At scientific engineering commit `6c80719266485c861fd8aa13f931eebcf5bfc8f8`, regression 988
accepted 64/64 inputs and fresh confirmation 989 accepted 32/32. All measured minimum retention
fractions were 1.0; maximum embedding lengths were 6,953 and 7,050. Readiness 990 recursively
verified those stages. Its `checked: 0` means it did not itself generate summaries. These are
engineering results, not M2 clinical performance or proof of full-cohort reliability.

This new production integration keeps that generation contract, the compact reversible renderer,
all fidelity thresholds, the same M1-retained source, Qwen instruction model and 0.6B encoder.
It freezes C1 train/validation rows and the original deterministic five C3 validation histories
per task for the 40-history fidelity review (five C1 plus five C3 per task). C3 review histories
are not a full C3 performance experiment. Test is closed; M1 and M3 are never rerun.

The separate namespace is `m2_structured_v2_reference_v1`. Engineering and execution commits are
recorded separately. Old caches and failed sentinels are untouched. Exact verified v2 engineering
responses with matching production requests can seed the new cache. All other legacy versions
are ineligible. Repeated identical inputs share generation work, not patient metadata or labels.

## Scheduling and safeguards

The planner reconstructs canonical M1 inputs, freezes the manual sample before generation, and
hashes task rows and deterministic 16-input work chunks. Up to two chunks run at once, with two
requests each: four total, as tested in jobs 988/989. Each chunk gets six hours. Two 600-second
attempts per input at concurrency two imply 2h40 of request time for 16 inputs, leaving roughly
a 2x contingency plus overhead. Network read timeouts are not an absolute wall-clock guarantee;
Slurm remains the final bound. Neither the shared serving process nor its capacity is changed.

Every received response and attempt is retained privately. Reuse revalidates source/request,
response, fidelity, token budget and hashes. A interrupted in-flight request consumes its attempt
rather than silently resetting the two-attempt lifetime budget. A completed rejected input sets
a global HALT; later shards fail before new generation. Do not delete HALT or accept/drop a failed
row. The next diagnostic/repair must account for that evidence. A killed job can leave useful
admitted rows without a COMPLETE sentinel: resume creates new job IDs and reuses those rows.

The graph is generation array -> coverage seal + review packet -> 0.6B embedding array.
`afterok` plus semantic checks gate each stage. Failed dependencies are canceled automatically.
The launcher writes each job ID immediately; never blindly retry a partial submission.
The final fit is deliberately NOT queued until human review is complete. All package-dependent
Python runs in the NVIDIA container on the verified node. No laptop generation or model execution.

## 1. Deploy from the Mac

The implementation is intentionally uncommitted. This explicit user-invoked helper validates,
stages only this change's named source/context files, commits and pushes; it leaves paper edits
and deliverables alone. It stops if unrelated changes are already staged.

```bash
cd /Users/williamdelery/code/temporal-ehr-ml4h
bash scripts/git/publish_m2_structured_production.sh
git --no-pager log -1 --oneline
```

## 2. Connect to VPN, then SSH and deploy

```bash
ssh wdelery@10.45.62.90
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
squeue -u "$USER"
git status --short --branch
```

If the queue is empty and the checkout is clean, continue:

```bash
eval "$(ssh-agent -s)"
ssh-add /home/wdelery/.ssh/id_ed25519_github_dgx
git pull --ff-only origin main
git --no-pager log -1 --oneline
bash scripts/slurm/submit_m2_structured_production.sh start
bash scripts/slurm/m2_production_status.sh
```

The start command first performs a containerized plan/gate check (up to one hour allocated), then
submits the graph. It sets the project/model paths itself. It does not redownload data or models.
Wait for printed job IDs and the queue check before leaving. Leave the source checkout unchanged
while any job is queued/running. After submission you may `exit` SSH and disconnect VPN; jobs do
not depend on your laptop. Reconnect only to inspect, review, or copy aggregates.

The optional browser explorer may be submitted after this graph is in the queue. Because this
launcher deliberately requires an otherwise empty user queue, cancel any still-running explorer
service before a later `resume` or `finalize`, then restart the explorer after submission.

## 3. Monitor and diagnose

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/m2_production_status.sh
source outputs/private/m2_production_job_ids.env
grep -h '"event": "m2_production_progress"' \
  slurm-tehr-m2-production-${M2_PRODUCTION_ARRAY}_*.out | tail -n 20
```

No events while pending is normal. The new worker reports after each wave of two inputs, not
after 100 rows. COMPLETE counts indicate execution status, not human/scientific completion.
Use the actual printed IDs, never an invented job number or an old `prompt4_reference` record.

```bash
cat reports/generated/final/m2_structured_v2_reference_v1/plan.json
```

Planning uses the observed engineering rate (~55 inputs/hour overall) for a rough generation-only
estimate before cache reuse. It is not representative enough to guarantee an ETA. Around 8,800
unique inputs would imply about 6–7 days, potentially longer with queuing/rejections. Short shards
prevent losing a long whole-task job; they do not make the instruction server generate faster.

For infrastructure interruption only, after reviewing errors and when the queue is completely
empty, the same unchanged checkout can resume with:

```bash
bash scripts/slurm/submit_m2_structured_production.sh resume
```

This rechecks the frozen plan and schedules new IDs; it never overwrites old status files. Valid
generation rows and completed embeddings are reused. It refuses a fidelity HALT. If HALT appears,
stop and return the safe status/log error categories for review, not raw source/summary contents.

## 4. Required human review on DGX

When the coverage/seal job is COMPLETE, its packet exists. Review can overlap the dependent GPU
embeddings; finalization must wait until all four embedding jobs are COMPLETE.

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
python3 scripts/review_m2_structured_packet.py
```

This shows original retained source and the actual compact encoder input. Read its field legend.
In the viewer, Space advances, `b` goes back, and `q` returns to the questions. Answer each
criterion `pass` or `fail` yourself. `quit` safely stops; rerun the command to continue where you
left off. After all 40 histories, type `COMPLETE`. Never mark pass without checking. Any failed
criterion blocks fitting. Do not use these held-out-for-fidelity validation histories to tune
the prompt. Do not copy the packet, decisions, source, or summaries off DGX.

## 5. Validation-only readout and independent verification

After review and embeddings, with an empty queue:

```bash
bash scripts/slurm/submit_m2_structured_production.sh finalize
bash scripts/slurm/m2_production_status.sh
```

When the fit job leaves the queue and its sentinel is COMPLETE:

```bash
bash scripts/slurm/verify_m2_production.sh
```

This rechecks hashes, patient coverage, prediction-derived metrics, equal-weight macro metrics,
manual review, and sentinels without fitting or encoding again. It publishes an aggregate-only
handoff. Required final outputs are `reference_validation.json`, `coverage.json`, `plan.json`,
four `embedding/*.json` files, and `reports/generated/handoff/m2_structured_v2_reference_handoff.json`.
Report AUROC, AUPRC, Brier score, log loss and calibration per task. Regularization uses the same
frozen grid and validation selection rule; these are exploratory validation results, not test.

Copy only reviewed public aggregates from a Mac terminal while connected to VPN:

```bash
PROJECT=/Users/williamdelery/code/temporal-ehr-ml4h
mkdir -p "$PROJECT/reports/generated/final/m2_structured_v2_reference_v1" "$PROJECT/reports/generated/handoff"
rsync -avP --checksum \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/final/m2_structured_v2_reference_v1/ \
  "$PROJECT/reports/generated/final/m2_structured_v2_reference_v1/"
rsync -avP --checksum \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/handoff/m2_structured_v2_reference_handoff.json \
  "$PROJECT/reports/generated/handoff/"
```

## Remainder of the project

Reference completion is not the entire temporal sensitivity matrix. Compare M2 C1 with the
already-completed M1/M3 validation evidence on aligned patients. Prospectively document any
post-M1 validation-only extension with Jeff: B1 latest-window, C0 order-only, C2 reversed earlier
gaps, and three C3 seeds are the original sensitivity candidates, not automatically authorized
new held-out experiments. Those additional M2 cells are NOT implemented by this reference runner.
The prior selection-to-test launcher is obsolete after the M1-only test amendment.

Then update task-specific analyses, figures, methods, limitations and the manuscript. Null temporal
contrasts do not establish Markovianity or equivalence; the structured-summary interface also
changes within-window cross-code interleaving. Preserve and report that limitation. New clinical
questions, cohorts, look-back ablations or testing beyond the original one-time evaluation require
a separate prospective protocol. Nothing here changes the encoder to 8B or reruns M1/M3/test.
