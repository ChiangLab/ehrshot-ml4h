#!/usr/bin/env python3
"""Build exact saved-input comparisons on DGX; never print clinical content."""
import os
import subprocess
from pathlib import Path

from temporal_ehr.example_atlas import render_atlas, select_example
from temporal_ehr.experiments import load_final_m1_authorization
from temporal_ehr.final_m1.throughput import read_final_task_texts


def main():
    root = Path('/data/shared73/projects/wdelery/temporal-ehr-ml4h')
    if Path.cwd().resolve() != root or not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run only in a DGX Slurm container from the repository root.')
    os.umask(0o077)
    spec = load_final_m1_authorization(root / 'configs/experiments/final_four_task_m1.yaml', repository_root=root)
    examples = {task: select_example(read_final_task_texts(spec, task)) for task in spec.tasks}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    output = root / 'outputs/private/internal_example_atlas' / commit / 'index.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    content = render_atlas(examples, commit)
    if output.exists():
        if output.read_text() != content:
            raise SystemExit('STOP: existing atlas differs; do not overwrite.')
    else:
        with output.open('x') as handle:
            handle.write(content)
    print('PASS: private four-task atlas built from hash-verified saved inputs.')
    print(f'Directory: {output.parent}')


if __name__ == '__main__':
    main()
