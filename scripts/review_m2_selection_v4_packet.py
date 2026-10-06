#!/usr/bin/env python3
"""Interactive governed review for M2 selection-v4; never auto-check a criterion."""

import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    root = Path("/data/shared73/projects/wdelery/temporal-ehr-ml4h")
    if Path.cwd().resolve() != root or not os.environ.get("SSH_CONNECTION"):
        raise SystemExit("Run interactively inside the DGX repository over SSH.")
    directory = root / "outputs/private/final/m2_selection_v4_reference_v1/review"
    if (directory / "ACCEPTED.json").exists():
        raise SystemExit("Review is already accepted and frozen; do not edit it.")
    packet = json.loads((directory / "packet.json").read_text())
    decisions = json.loads((directory / "decisions.json").read_text())
    template = json.loads((directory / "decision_template.json").read_text())
    if len(packet["entries"]) != 40 or len(decisions["decisions"]) != 40:
        raise SystemExit("Incomplete selection-v4 review packet.")
    criteria = [
        key
        for key in template["decisions"][0]
        if key not in {"review_id", "task", "condition"}
    ]

    def save() -> None:
        descriptor, name = tempfile.mkstemp(prefix=".review-", dir=directory)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(decisions, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, directory / "decisions.json")

    print("Governed review: do not copy records or summaries off DGX. Press q to leave less.")
    for index, (entry, decision) in enumerate(
        zip(packet["entries"], decisions["decisions"], strict=True)
    ):
        if any(entry[key] != decision[key] for key in ("review_id", "task", "condition")):
            raise SystemExit("Review identity mismatch.")
        if all(type(decision.get(key)) is bool for key in criteria):
            continue
        body = (
            f"ITEM {index + 1}/40 — {entry['task']} — {entry['condition']}\n\n"
            + "ORIGINAL M1-RETAINED SOURCE\n"
            + entry["source_text"]
            + "\n\nACTUAL ENCODER INPUT (read the field legend)\n"
            + entry["summary"]
        )
        subprocess.run(["less", "-R"], input=body, text=True, check=True)
        for criterion in criteria:
            if type(decision.get(criterion)) is bool:
                continue
            while True:
                answer = input(f"{criterion}: pass / fail / quit? ").strip().lower()
                if answer == "quit":
                    save()
                    return
                if answer in {"pass", "fail"}:
                    decision[criterion] = answer == "pass"
                    save()
                    break
    answer = input("All 40 reviewed. Type COMPLETE to finalize: ").strip()
    if answer == "COMPLETE":
        decisions["review_complete"] = True
        save()
        print("Decisions saved; the fit job independently enforces every criterion.")


if __name__ == "__main__":
    main()
