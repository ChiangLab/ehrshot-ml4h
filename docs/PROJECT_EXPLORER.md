# Governed project explorer

The project explorer is a private browser interface for presentation preparation and scientific
review on the DGX. It is not a new analysis and does not change any experiment.

## Scope and architecture

- The original `data.parquet` remains the immutable EHRSHOT source of truth. The DuckDB file does
  not copy the full source table. It exposes only events belonging to the selected train or
  validation cohort and only when `event_time < prediction_time`.
- `clinical.original_events_pre_prediction` retains the original seven MEDS/EHRSHOT fields for
  those authorized histories. `clinical.deduplicated_events_pre_prediction` applies the project's
  exact seven-field deduplication and adds days-before-T plus the task-specific 30/90-day window.
- M1 serialized inputs and the new M2 validated summaries are copied into the private catalog for
  easy review. M2 stays empty until structured-v2 production rows exist.
- Every complete M1, structured-v2 M2, and M3 embedding array is cataloged. Every vector row is
  mapped to its private metadata. Full arrays remain immutable `.npy` files; the browser reads only
  a selected vector for statistics and a coordinate preview. This prevents a multi-gigabyte second
  copy while still indexing all embeddings.
- B1-B4 and M1-M3 are described explicitly. The governed protocol defines no M4, so the explorer
  says `M4: absent from protocol` rather than inventing an output.
- Development validation metrics and non-test run sentinels are indexed. The one-time held-out
  submission directory and test run sentinels are excluded. There is no arbitrary-SQL or download
  route.

The database and UI are governed patient-level artifacts. They stay under `outputs/private`, bind
to DGX localhost only, and must not be copied to the Mac. Browser access uses an SSH tunnel.

## Start on the DGX

Use this only from a clean checkout with the source commit deployed. M2 can run independently, but
do not rebuild the explorer while it is already serving. The job performs a one-time deep scan of
embedding hashes, finiteness, row maps, and L2 norms before starting the web server.

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_PROJECT_ROOT="$PWD"
export EHRSHOT_ASSET_ROOT="/data/shared73/projects/wdelery/ehrshot-assets"
export EHRSHOT_DATA_PATH="${EHRSHOT_ASSET_ROOT}/data.parquet"
export PROJECT_EXPLORER_PORT=8765

bash scripts/slurm/submit_project_explorer.sh
bash scripts/slurm/project_explorer_status.sh
```

The validation/build can take several minutes because every indexed embedding array is scanned.
Wait until the log contains `project_explorer_ready`.

Submit M2 production before the explorer. The M2 start/resume/finalize launcher intentionally
requires an otherwise empty user queue. If an explorer service is still active when M2 must be
resumed or finalized, stop the explorer first, perform the scientific submission, and then submit
a fresh explorer job.

## Open from the Mac

Remain connected to the VPN. Open a second ordinary Mac terminal and keep this command running:

```bash
ssh -N -L 8765:127.0.0.1:8765 wdelery@10.45.62.90
```

Then open <http://127.0.0.1:8765> in Chrome. Closing the tunnel does not stop the Slurm job; it only
closes browser access. The server job has a 12-hour time limit. Restarting it rebuilds the catalog
from the then-current verified artifacts, which is appropriate after M2 advances.

## Stop or refresh

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/project_explorer_job.env
scancel "$PROJECT_EXPLORER_JOB_ID"
```

After M2 generation, review, embedding, or fitting completes, submit a new explorer job to refresh
the M2 summary, embedding, metric, and run-status views. Never use the explorer to select a prompt,
model, or analysis from held-out performance.
