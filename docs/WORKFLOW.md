# Version-controlled execution workflow

The repository is the durable project memory. Chat transcripts are supporting context, not the
source of truth.

## Start every implementation session

1. Read `AGENTS.md` and every current context document it requires.
2. Run `.venv/bin/python scripts/check_project_context.py`.
3. Read only the one prompt under `prompts/next/`.
4. Verify that prompt's hard gates against `docs/RESULTS_REGISTRY.md`, Git, privacy-safe aggregates,
   and DGX sentinels. Never infer completion from an absent queue entry or file existence alone.

## End every implementation session

1. Run the full synthetic tests, Ruff, changed-shell syntax checks, and
   `bash scripts/git/pre_push_safety.sh`.
2. Update `PROJECT_STATE.md` with source and governed execution state.
3. Record decisions or deviations in `DECISION_LOG.md` or `PROTOCOL_DEVIATIONS.md`.
4. Record commits, DGX job IDs, aggregate paths, and gates in `RESULTS_REGISTRY.md`.
5. Update `NEXT_ACTION.md` to one explicit action and one active prompt.
6. Generate the privacy-safe handoff bundle.
7. After the phase's real jobs and semantic validators pass, move its prompt from `prompts/next/`
   to `prompts/completed/`, move exactly one successor from `prompts/planned/` to `prompts/next/`,
   and rerun `scripts/check_project_context.py`.

## Mac and DGX boundary

- Source implementation, synthetic validation, review, commits, and pushes occur in the Mac
  checkout. VPN state does not change scientific rules; GitHub access may depend on the current
  network configuration.
- Governed EHRSHOT work and model execution occur only through Slurm on the DGX.
- Private artifacts remain ignored on the DGX. Copy back only reviewed aggregate reports and the
  privacy-safe handoff.
- Each formal scientific run uses a clean commit, immutable configuration/hash identity,
  dependency-locked jobs, atomic sentinels, and semantic validators.

## Current Prompt 2 shortcut

After the DGX checkout is clean and current and the four required path variables are exported, run:

```bash
bash scripts/slurm/submit_prompt2_execution.sh
```

The helper validates prerequisites, submits the synthetic smoke gate, submits the pilot chain
behind it, and records all IDs in a versioned private submission record plus
`outputs/private/prompt2_job_ids.env` as a pointer to the newest record.
