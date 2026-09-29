#!/usr/bin/env python3
"""Interactive human review on DGX. Never auto-check a clinical criterion."""
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    root = Path('/data/shared73/projects/wdelery/temporal-ehr-ml4h')
    if Path.cwd().resolve() != root or not os.environ.get('SSH_CONNECTION'):
        raise SystemExit('Run interactively inside the DGX repository over SSH, not on the Mac.')
    directory = root / 'outputs/private/final/m2_structured_v2_reference_v1/review'
    if (directory / 'ACCEPTED.json').exists():
        raise SystemExit('Review already accepted and frozen. Do not edit it.')
    packet = json.loads((directory / 'packet.json').read_text())
    decisions = json.loads((directory / 'decisions.json').read_text())
    template = json.loads((directory / 'decision_template.json').read_text())
    if len(packet['entries']) != 40 or len(decisions['decisions']) != 40:
        raise SystemExit('Incomplete review packet.')
    criteria = [k for k in template['decisions'][0] if k not in {'review_id', 'task', 'condition'}]

    def save():
        fd, name = tempfile.mkstemp(prefix='.review-', dir=directory)
        with os.fdopen(fd, 'w') as handle:
            json.dump(decisions, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, directory / 'decisions.json')

    print('Governed review: do not copy source records or summaries off DGX. Press q to leave the viewer.')
    for index, (entry, decision) in enumerate(zip(packet['entries'], decisions['decisions'], strict=True)):
        if any(entry[k] != decision[k] for k in ('review_id', 'task', 'condition')):
            raise SystemExit('Review identity mismatch.')
        if all(type(decision.get(k)) is bool for k in criteria):
            continue
        body = (f"ITEM {index+1}/40 — {entry['task']} — {entry['condition']}\n\n"
                + 'ORIGINAL M1-RETAINED SOURCE\n' + entry['source_text']
                + '\n\nACTUAL ENCODER INPUT (read the field legend)\n' + entry['summary'])
        subprocess.run(['less', '-R'], input=body, text=True, check=True)
        for criterion in criteria:
            if type(decision.get(criterion)) is bool:
                continue
            while True:
                answer = input(f'{criterion}: pass / fail / quit? ').strip().lower()
                if answer == 'quit':
                    save()
                    return
                if answer in {'pass', 'fail'}:
                    decision[criterion] = answer == 'pass'
                    save()
                    break
    answer = input('All 40 reviewed. Type COMPLETE to finalize the decisions: ').strip()
    if answer == 'COMPLETE':
        decisions['review_complete'] = True
        save()
        print('Decisions saved. The fit job will independently enforce every criterion; failures block fitting.')


if __name__ == '__main__':
    main()
