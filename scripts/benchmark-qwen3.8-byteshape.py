#!/usr/bin/env python3
"""Serialized ByteShape launcher/decoder comparison with reasoning and GPU records."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import importlib.util
spec = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('optimization-probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
ROOT = Path(__file__).resolve().parents[1]


def gpu_snapshot():
    result = {}
    for device in sorted(Path('/sys/class/drm').glob('card*/device')):
        if not (device / 'mem_info_vram_used').exists():
            continue
        values = {'pci': device.resolve().name}
        for field in ('mem_info_vram_used', 'mem_info_vram_total', 'gpu_busy_percent'):
            path = device / field
            if path.exists():
                try:
                    values[field] = int(path.read_text())
                except OSError as error:
                    values[field] = {'unavailable': str(error)}
        result[device.parent.name] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=384)
    parser.add_argument('--ctx', type=int, default=131072)
    parser.add_argument('--prompts', default='code,reason', help='Prompt names from optimization-probe.py')
    parser.add_argument('--port', type=int, default=8093)
    parser.add_argument('--name', default=time.strftime('byteshape-%Y%m%d-%H%M%S'))
    parser.add_argument('--profiles', help='Comma-separated labels to run (default: full matrix)')
    args = parser.parse_args()
    prompt_keys = args.prompts.split(',')
    if set(prompt_keys) - set(probe.PROMPTS):
        parser.error('Unknown prompt name')
    if min(args.runs, args.tokens, args.ctx) < 1:
        parser.error('runs, tokens and ctx must be positive')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must be a simple label')
    output = ROOT / 'artifacts/byteshape' / args.name
    output.mkdir(parents=True, exist_ok=False)
    base = f'http://127.0.0.1:{args.port}'
    if subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True).returncode == 0:
        raise RuntimeError('An existing llama-server is running; stop it before benchmarking')
    try:
        probe.request(base, '/health', timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError('Benchmark port is occupied')
    draft_dir = ROOT / 'llama-hip/models/qwen3.8-27b-q4_0'
    model = ROOT / 'llama-hip/models/qwen3.8-27b-byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf'
    profiles = []
    for launcher in ('hip', 'fork', 'fork-dual'):
        for decoder, spec_type, depth, draft in (
            ('none', 'none', 0, None),
            ('mtp', 'draft-mtp', 3, draft_dir / 'MTP/mtp-Qwen3.8-27B-Q4_0.gguf'),
            ('mtp-embedded', 'draft-mtp', 3, 'embedded'),
            ('dflash3', 'draft-dflash', 3, draft_dir / 'DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf'),
            ('dflash7', 'draft-dflash', 7, draft_dir / 'DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf')):
            profiles.append((f'{launcher}-{decoder}', launcher, 'byteshape', spec_type, depth, draft, args.ctx, False))
    profiles += [('q4_0-fork-dflash3', 'fork', 'q4_0', 'draft-dflash', 3,
                  draft_dir / 'DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf', args.ctx, False)]
    # Validate each launcher's native context/parallel/cache defaults separately.
    for launcher in ('hip', 'fork', 'fork-dual'):
        profiles.append((f'{launcher}-defaults', launcher, 'byteshape', 'draft-dflash', 3,
                         draft_dir / 'DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf', None, True))
    if args.profiles:
        selected = set(args.profiles.split(','))
        known = {p[0] for p in profiles}
        if selected - known:
            parser.error(f'Unknown profiles: {selected - known}')
        profiles = [p for p in profiles if p[0] in selected]
    rows, failures = [], []
    (output / 'experiment.json').write_text(json.dumps({
        'args': vars(args), 'profiles': [[str(v) if isinstance(v, Path) else v for v in p] for p in profiles],
        'prompts': {key: probe.PROMPTS[key] for key in prompt_keys},
        'sampling': {'temperature': 0, 'seed': 3407, 'cache_prompt': False},
        'reasoning': 'auto/medium', 'kv': 'q8_0', 'draft_kv': 'f16',
        'note': 'Short prompts in allocated contexts; not a filled-context test. Default checks use native cache/parallel settings.',
        'source_revision': 'c9cc5b2ae2520ab77d1e5d3fae395b4140bb37e8',
        'model_sha256': '89434f23dc89c5f990894e3fe9fdad19d88c370f0d3638a176f29933f218b78b',
    }, indent=2))
    for label, launcher, quant, spec_type, depth, draft, ctx, smoke in profiles:
        directory = output / label
        directory.mkdir()
        env = os.environ.copy()
        env.update(MODEL_QUANT=quant, SPEC_TYPE=spec_type, SPEC_DRAFT_N_MAX=str(depth),
                   PORT=str(args.port), HOST='127.0.0.1', REASONING='auto', REASONING_EFFORT='medium')
        env.pop('MODEL', None)
        if not smoke:
            env.update(CTX_SIZE=str(ctx), PARALLEL='1', DRAFT_CACHE_TYPE='f16',
                       SPEC_DRAFT_P_MIN='0.20', UBATCH_SIZE='512', CACHE_TYPE_K='q8_0', CACHE_TYPE_V='q8_0')
        if draft:
            env['DRAFT'] = str(draft)
        command = [str(ROOT / f'start-llama-{launcher}.sh')]
        binary = ROOT / {'hip': 'llama-hip/build-hip/bin/llama-server',
                         'fork': 'llama-fork/build-rocm-gfx1100-portable/bin/llama-server',
                         'fork-dual': 'llama-fork/build-rocm-dual/bin/llama-server'}[launcher]
        manifest = {'command': command, 'env': {k: env[k] for k in ('MODEL_QUANT','SPEC_TYPE','SPEC_DRAFT_N_MAX','CTX_SIZE','PARALLEL','DRAFT_CACHE_TYPE','SPEC_DRAFT_P_MIN','DRAFT') if k in env},
                    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'gpu_before': gpu_snapshot()}
        child = None
        try:
            print(f'START {label}', flush=True)
            with (directory / 'server.log').open('w') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 600
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'server startup exit {child.returncode}')
                try:
                    probe.request(base, '/health', timeout=2)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError('startup timeout')
                    time.sleep(1)
            manifest['props'] = probe.request(base, '/props')
            manifest['gpu_loaded'] = gpu_snapshot()
            cases = [('warmup', 'code', 96)]
            cases += [(f'{run+1}-{key}', key, args.tokens if not smoke else 192)
                      for run in range(args.runs if not smoke else 1)
                      for key in prompt_keys]
            for name, key, count in cases:
                payload = {'messages': [{'role': 'user', 'content': probe.PROMPTS[key]}],
                           'temperature': 0, 'seed': 3407, 'max_tokens': count, 'cache_prompt': False}
                (directory / f'{name}-request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
                response = probe.request(base, '/v1/chat/completions', payload, timeout=max(300, count // 5))
                (directory / f'{name}.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
                manifest['gpu_after_request'] = gpu_snapshot()
                if name == 'warmup':
                    continue
                t = response['timings']
                drafted, accepted = t.get('draft_n', 0), t.get('draft_n_accepted', 0)
                message = response['choices'][0]['message']
                row = {'profile': label, 'scenario': key, 'run': name.split('-')[0],
                       'prompt_tok_s': t.get('prompt_per_second', 0), 'decode_tok_s': t.get('predicted_per_second', 0),
                       'prompt_tokens': response.get('usage', {}).get('prompt_tokens'),
                       'completion_tokens': response.get('usage', {}).get('completion_tokens'),
                       'accepted': accepted, 'drafted': drafted, 'acceptance': accepted / drafted if drafted else 0,
                       'finish_reason': response['choices'][0]['finish_reason'],
                       'output_sha256': hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()}
                rows.append(row)
                print(json.dumps(row), flush=True)
            manifest['status'] = 'passed'
        except Exception as error:
            manifest.update(status='failed', error=str(error))
            failures.append({'profile': label, 'error': str(error)})
            print(f'FAIL {label}: {error}', flush=True)
        finally:
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            time.sleep(2)
            manifest['gpu_after_stop'] = gpu_snapshot()
            (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            (output / 'samples.json').write_text(json.dumps(rows, indent=2))
            (output / 'failures.json').write_text(json.dumps(failures, indent=2))
    with (output / 'summary.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(('profile','scenario','runs','prompt_tok_s','decode_tok_s','acceptance'))
        for label, *_ in profiles:
            for key in prompt_keys:
                samples = [r for r in rows if r['profile'] == label and r['scenario'] == key]
                if samples:
                    writer.writerow((label,key,len(samples),*(round(statistics.median(r[f] for r in samples),4)
                                                            for f in ('prompt_tok_s','decode_tok_s','acceptance'))))
    print(f'RESULTS {output}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
