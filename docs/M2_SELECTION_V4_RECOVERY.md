# M2 selection-v4 recovery after canary 1551

Local validation on 2026-10-06: 339 synthetic tests passed, Ruff passed, all four new shell
scripts passed `bash -n`, durable-context checks passed, and the repository privacy pre-push
check passed. No real data or model ran locally. No DGX v4 success is claimed.

## What failed and what changes

Job 1551 checked 32 frozen engineering inputs and accepted 14. Its 23 concept-retention errors
are failed-attempt entries, not 23 patients. It ended in 219 seconds: scheduling was not the
root cause. Exact source copying already prevented invented measurements; unrestricted series
selection still let the model omit too many concepts. Preserve 1551, the v3 plan and every cache.

The v4 model returns **omission indices**, not rewritten clinical text. Each window may omit
at most floor(number_of_series/10) singleton code-only series. Numeric/text values, units,
recurrence and ordered measurement runs are protected. An empty omission list retains all
facts only when that is the actual model response. There is no failure fallback or automatic
repair. Independent fidelity and 8,192-token checks remain authoritative.

This is a methodological narrowing to constrained extractive selection with deterministic
compaction, not unrestricted abstractive summarization. Cross-code event interleaving is not
preserved by within-code grouping; within-series measurement order is preserved. Report the
actual amount omitted, including zero omissions, rather than claiming meaningful compression
from a successful API call. Keep clinical sources, tasks, Qwen instruction model, decoding,
0.6B encoder, logistic classifier and development-only boundary unchanged.

## Deployment

On the Mac, publish the scoped recovery commit (do not stage unrelated manuscript or private files):

```bash
cd /Users/williamdelery/code/temporal-ehr-ml4h
git push origin main
```

Connect to MedNet VPN, then open an SSH session. All remaining scientific commands run on DGX:

```bash
ssh wdelery@10.45.62.90
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
git status --short --branch
eval "$(ssh-agent -s)"
ssh-add /home/wdelery/.ssh/id_ed25519_github_dgx
git pull --ff-only origin main
git --no-pager log -1 --oneline
squeue -u "$USER"
bash scripts/slurm/submit_m2_selection_v4.sh start
```

Stop if pull fails or source is dirty. The helper refuses active jobs and sets both required
root paths. It runs a containerized preflight: validates prior failure/source hashes, reuses the
same 32 regression cases, proves the original independent 32 confirmation cases are unexposed,
and tests full compact-source feasibility on every planned input. It requests no new summaries
if this conservative 8,192-token feasibility check fails. No raw-data/cohort rebuild occurs.

Then it queues canary -> confirmation -> semantic readiness -> bounded generation shards ->
coverage seal and 40-history review packet -> four embedding tasks. All links require upstream
success; invalid dependencies are canceled. Production uses 16 inputs/shard, at most 2 active
shards and 2 requests/shard. Gate jobs allow4 hours, generation 6 hours, embedding 12 hours.
These are limits, not predicted completion times. Each generation has at most 2 fixed attempts.
No automatic retries of failed scientific gates, no cache mixing, no M1/M3/test rerun.

## Monitoring

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/m2_selection_v4_status.sh
source outputs/private/m2_selection_v4_job_ids.env
for phase_job in "$M2_SELECTION_CANARY_JOB" "$M2_SELECTION_CONFIRMATION_JOB" "$M2_SELECTION_READINESS_JOB"; do
  find "outputs/private/run_status/$phase_job" -maxdepth 1 -type f -name '*.json' -printf '%f\n' 2>/dev/null
done
```

Empty queue is not success. Check COMPLETE without FAILED, expected commit, gate report PASS,
and eventually the final semantic verifier. Read logs locally; do not paste clinical content.
If start fails before submission, inspect the preflight message and
`reports/generated/engineering/m2_selection_v4/source-feasibility.json` if present.
If a fidelity gate fails or HALT appears, preserve it and stop. Do not delete it or launch again.
For an infrastructure-only interruption with no failed fidelity gate/HALT, an idle same-commit
checkout may use `bash scripts/slurm/submit_m2_selection_v4.sh resume`; valid sealed rows are
reverified, not regenerated. Original job records remain preserved.

You may disconnect after successful submission. VPN/SSH are needed to monitor, not to keep
submitted batch jobs alive. The lab inference endpoint must remain available from DGX.

## Human review, fit and completion

Wait for the seal and all 4 embedding tasks to complete. Review is personal and cannot be
automatically marked correct:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
python3 scripts/review_m2_selection_v4_packet.py
```

Read all 40 source/output pairs, press `q` to leave each pager, then enter pass/fail for every
criterion. `quit` saves progress. Type `COMPLETE` only after actual review. Then:

```bash
bash scripts/slurm/submit_m2_selection_v4.sh finalize
bash scripts/slurm/m2_selection_v4_status.sh
```

After the fit completes:

```bash
bash scripts/slurm/verify_m2_selection_v4.sh
```

Only then is the **M2 reference validation** complete. The four-task C1 reference plus frozen
C3 fidelity sample is not the full M2 C0-C3/B1 sensitivity experiment. Review reference evidence
before separately authorizing that next stage. Keep the completed M1 held-out evaluation closed.
Copy only reviewed aggregate reports and
`reports/generated/handoff/m2_selection_v4_reference_handoff.json` to the Mac.

## Exact browser examples from all four tasks

The following builds a private HTML atlas from hash-verified saved M1 training inputs, with no
model call. It shows the full C0-C3/B1/B2 text for the same patient within each task. It rejects
test rows and selects examples by fixed hash among eligible instances, not by outcomes.

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data --container-workdir="$PWD" \
  env PYTHONPATH=src .venv/bin/python scripts/build_internal_example_atlas.py
python3 -m http.server 8766 --bind 127.0.0.1 \
  --directory "outputs/private/internal_example_atlas/$(git rev-parse HEAD)"
```

The last command stays running. In a separate **Mac** terminal, while on VPN:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8766:127.0.0.1:8766 wdelery@10.45.62.90
```

Open http://127.0.0.1:8766 in your browser. Keep content in the governed internal workflow.
These four illustrative examples do not replace the frozen 40-history M2 fidelity review.
Stop each server/tunnel with Ctrl-C when finished.
