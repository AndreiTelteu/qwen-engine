#!/usr/bin/env python3
"""Print recorded experiments without treating incomplete probes as wins."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1] / 'artifacts/optimization-lab/results'
print('profile,status,ctx,tui_tps,code_tps,reason_tps,samples')
for directory in sorted(root.iterdir()):
    if not (directory / 'manifest.json').is_file():
        continue
    manifest = json.loads((directory / 'manifest.json').read_text())
    samples = json.loads((directory / 'samples.json').read_text()) if (directory / 'samples.json').exists() else []
    values = {}
    for key in ('tui', 'code', 'reason'):
        measured = [s['timings']['predicted_per_second'] for s in samples
                    if s['label'].endswith('-' + key) and s.get('timings', {}).get('predicted_per_second')]
        values[key] = f'{statistics.median(measured):.2f}' if measured else ''
    print(','.join([directory.name, manifest.get('status', 'running'), str(manifest['args']['ctx']),
                    values['tui'], values['code'], values['reason'], str(sum(s['label'] != 'warmup' for s in samples))]))
