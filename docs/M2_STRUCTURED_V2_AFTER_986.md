# M2 generation repair after confirmation 986

2026-09-25. Supersedes the render-v2 launch instructions. Readiness is not full M2 completion.

## What changes, and what does not

User-run DGX diagnostics semantically verified replay 985 (all 32 accepted; maximum 6,953
embedding tokens) and checked the immutable 986 evidence. Confirmation 986 accepted 29/32;
failed outputs included duplicate series, changed measurement content/counts/runs, and changed
series ordering. Categories overlap. It finished in about 40 minutes, not at the 12-hour limit.
The later SSH disconnect did not cause the recorded fidelity failure.

This repair is a prompt amendment, not a claim that prompt wording guarantees success. It adds
explicit instructions about complete code/source/unit identities, copying each retained fact once,
preserving every ordered run/count and null/string distinction, and never sorting or repairing
supplied time labels. The model still emits clinical facts, not IDs. The JSON object keys are
serialized canonically so requests and identities survive saving/loading; no clinical arrays sort.
The strict schema, validator, measured-retention floors, compact render-v2, 8,192-token cap, original
clinical source, model and decoding remain unchanged. Temperature is 0, top-p 1, seed 20260815,
thinking off, maximum generation tokens 16,384. The encoder remains 0.6B; no embeddings run here.

No answer is deduplicated, reordered, filled from source, merged with another answer, or silently
truncated. At most two fixed requests are made per input, with no rejection feedback or recursive
summarization. The first valid response wins. Repeated rejected content is detected using the
parsed document, not API request IDs. All attempts stay private. New prompt/request keys cannot
reuse historical answers as new-generation results. Original jobs 982/985/986 stay unchanged.

## Automated graph

1. **Regression, 8-hour allocation:** verify original 985/986 evidence and freeze both new-stage
   samples before any generation. Regress all 64 inspected 982-development/986-confirmation
   inputs, not just failures. Three synthetic cases precede clinical requests. Require every
   input to pass unchanged fidelity, reversible rendering and real tokenizer checks.
2. **Fresh confirmation, 4-hour allocation:** starts only after regression succeeds. Uses 32
   training inputs (four tasks, C1/C3, four source-length positions), excluding all 980/982/986
   engineering people globally across tasks. Selection is deterministic, outcome-free and frozen
   before regression, not chosen after seeing prompt performance. Require every input to pass.
3. **Readiness, 30-minute allocation:** recursively verifies both completed stages, original
   evidence, source selection, all manifests/responses, semantic fidelity and cache identities.
   Produces an aggregate handoff and stops. No new generation in this phase.

Four concurrent endpoint requests maximum, no DGX GPU, 10-minute socket timeout per request,
bounded waves, a full-failure-wave circuit breaker and deadline reserves. Limits are not ETAs;
network/server delays can still cause a failure. All Python runs through the NVIDIA container on
node dgit00038673. The local SSH client and VPN do not need to stay connected after submission.
No live weight digest is available from the endpoint: the existing served metadata identity is
verified before/after; this is not proof against a silent server-side weight replacement.

`afterok` and `--kill-on-invalid-dep=yes` block/cancel successors on failure. Jobs additionally
verify the parent's final sentinel and semantic/hash evidence. The launcher records each ID as
soon as it is returned and refuses duplicate submission at the same commit. If submission is
partial, inspect the record; do not erase it or repeat the launcher. No auto-retry loop hides
failures. Changing the source while any stage is queued/running is forbidden.

## 1. Mac: validate, commit only this repair, and push

Run outside SSH. This helper explicitly commits and pushes the repair files only; it does not
stage unrelated paper/deliverable edits. It refuses pre-existing staged changes. The assistant
has left these changes uncommitted. If any command fails, stop and return the error.

```bash
cd /Users/williamdelery/code/temporal-ehr-ml4h
bash scripts/git/publish_m2_structured_v2.sh
```

Save the printed full commit hash. If only the push failed after a successful commit, do not
rerun the publisher; resolve authentication and run `git push origin main` from this Mac repository.
No local real data, model, new download, or test performance is needed.

## 2. DGX: deploy and submit once

Connect to MedNet VPN, then from the Mac:

```bash
ssh wdelery@10.45.62.90
```

Paste this complete block inside the DGX SSH terminal. Parentheses keep failure from logging you
out and `set -e` stops the block before submission if an earlier check fails. Enter the Mac's
printed commit when prompted; the check prevents running an old or different repair.

```bash
(
  set -e
  cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
  test -z "$(squeue -h -u "$(id -un)")"
  test -z "$(git status --porcelain)"
  eval "$(ssh-agent -s)"
  ssh-add /home/wdelery/.ssh/id_ed25519_github_dgx
  git pull --ff-only origin main
  read -r -p 'Paste the full Mac repair commit: ' EXPECTED_M2_V2_COMMIT
  test "$(git rev-parse HEAD)" = "$EXPECTED_M2_V2_COMMIT"
  test -z "$(git status --porcelain)"
  bash scripts/slurm/submit_m2_structured_v2.sh
  bash scripts/slurm/m2_structured_v2_status.sh
)
```

The launcher sets project/model roots, so missing exports after reconnect cannot cause the old
failure. It requires 985 COMPLETE and 986 FAILED; the first job verifies their full evidence.
Write down the three returned IDs. Inspect initial queue/logs. Then `exit` and disconnect VPN if
desired; Slurm runs all eligible successors without another interactive command.

## 3. Monitor after reconnecting

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/m2_structured_v2_status.sh
```

For recent progress in the regression stage:

```bash
source outputs/private/m2_structured_v2_job_ids.env
tail -n 80 "slurm-tehr-m2-v2-${REGRESSION_JOB_ID}.out"
```

Clinical-stage progress is written every four checked inputs. Preparation/hash checks and the
synthetic probe may precede the first progress line. A COMPLETE regression may retain errors from
an earlier rejected attempt; admission still requires its eventual response to pass. The new
failure report counts error occurrences only for finally failed inputs; counts are not necessarily
independent patients. Small counts are suppressed. Do not paste any private requests/responses.

## 4. Verify all stages and return only aggregate reports

After all three stages show COMPLETE with no FAILED sentinel:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
bash scripts/slurm/verify_m2_structured_v2.sh
```

This recursively revalidates, without generation, embeddings or clinical performance access.
If a stage failed or has no final sentinel, instead return its aggregate report and safe error
lines; do not bypass the gate. A scheduler dependency cancellation may have no run sentinel.

From a separate **Mac terminal**, while on VPN:

```bash
mkdir -p /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/engineering/m2_structured_v2
rsync -avP --checksum --include='*/' --include='regression-*.json' \
  --include='confirmation-*.json' --include='readiness-*.json' --exclude='*' \
  wdelery@10.45.62.90:/data/shared73/projects/wdelery/temporal-ehr-ml4h/reports/generated/engineering/m2_structured_v2/ \
  /Users/williamdelery/code/temporal-ehr-ml4h/reports/generated/engineering/m2_structured_v2/
```

## What is still not automated or complete

This patch does not implement or launch the full-cohort structured-v2 production worker, cache
adapter, embeddings or fitting. Reviewed engineering evidence is needed to size and integrate that
worker safely, rather than connecting a new schema to a legacy validator. Next stages are bounded,
resumable train/validation generation, 0.6B embedding and the frozen 40-history review packet,
explicit human fidelity review, then validation-only logistic regression. Manual review cannot be
auto-approved. M1, canonical M3 and the one-time test are unchanged. Passing 96 engineering inputs
does not prove general fidelity, clinical utility, compression benefit or Markovianity.
