#!/usr/bin/env python3
"""Compare the ByteShape baseline with RDNA boosts DFlash2 and embedded MTP."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('optimization-probe.py'))
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
RELEASE_PROMPTS = ROOT / 'artifacts/rdna-boosts-v16-ebbb18522-r2/prompts'
PROMPTS = dict(PROBE.PROMPTS)
if RELEASE_PROMPTS.is_dir():
    PROMPTS.update({
        'code-long': (RELEASE_PROMPTS / 'code-python.txt').read_text(),
        'prose-long': (RELEASE_PROMPTS / 'prose-rdna-boosts.txt').read_text(),
    })


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


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git_value(repository, value):
    return subprocess.check_output(['git', '-C', repository, 'rev-parse', value], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=2000)
    parser.add_argument('--ctx', type=int, default=131072)
    parser.add_argument('--prompts', default='code-long,prose-long')
    parser.add_argument('--port', type=int, default=8094)
    parser.add_argument('--name', default=time.strftime('rdna-boosts-%Y%m%d-%H%M%S'))
    parser.add_argument('--profiles', help='Comma-separated profile labels')
    args = parser.parse_args()
    prompt_keys = args.prompts.split(',')
    if set(prompt_keys) - set(PROMPTS):
        parser.error('Unknown prompt name')
    if min(args.runs, args.tokens, args.ctx) < 1:
        parser.error('runs, tokens and ctx must be positive')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must be a simple label')

    draft = ROOT / 'llama-hip/models/qwen3.8-27b-q4_0/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf'
    model = ROOT / 'llama-hip/models/qwen3.8-27b-byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf'
    profiles = [
        {
            'label': 'fork-dflash3',
            'launcher': ROOT / 'start-llama-fork.sh',
            'binary': ROOT / 'llama-fork/build-rocm-gfx1100-portable/bin/llama-server',
            'repository': ROOT / 'llama-fork',
            'spec_type': 'draft-dflash',
            'depth': 3,
            'draft': str(draft),
        },
        {
            'label': 'rdna-dflash3',
            'launcher': ROOT / 'start-llama-rdna-boosts.sh',
            'binary': ROOT / 'llama-rdna-boosts/build-rocm-gfx1100/bin/llama-server',
            'repository': ROOT / 'llama-rdna-boosts',
            'spec_type': 'draft-dflash',
            'depth': 3,
            'draft': str(draft),
        },
        {
            'label': 'rdna-mtp3',
            'launcher': ROOT / 'start-llama-rdna-boosts.sh',
            'binary': ROOT / 'llama-rdna-boosts/build-rocm-gfx1100/bin/llama-server',
            'repository': ROOT / 'llama-rdna-boosts',
            'spec_type': 'draft-mtp',
            'depth': 3,
            'draft': 'embedded',
        },
        {
            'label': 'rdna-mtp-adaptive12',
            'launcher': ROOT / 'start-llama-rdna-boosts.sh',
            'binary': ROOT / 'llama-rdna-boosts/build-rocm-gfx1100/bin/llama-server',
            'repository': ROOT / 'llama-rdna-boosts',
            'spec_type': 'draft-mtp-adaptive',
            'depth': 12,
            'draft': 'embedded',
        },
    ]
    if args.profiles:
        selected = set(args.profiles.split(','))
        known = {profile['label'] for profile in profiles}
        if selected - known:
            parser.error(f'Unknown profiles: {selected - known}')
        profiles = [profile for profile in profiles if profile['label'] in selected]

    if not model.is_file() or not draft.is_file():
        raise FileNotFoundError('ByteShape target or DFlash2 draft is missing')
    if subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True).returncode == 0:
        raise RuntimeError('An existing llama-server is running; stop it before benchmarking')

    output = ROOT / 'artifacts/rdna-boosts' / args.name
    output.mkdir(parents=True, exist_ok=False)
    base = f'http://127.0.0.1:{args.port}'
    try:
        PROBE.request(base, '/health', timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError('Benchmark port is occupied')

    experiment = {
        'args': vars(args),
        'profiles': profiles,
        'prompts': {key: {'sha256': hashlib.sha256(PROMPTS[key].encode()).hexdigest(),
                          'bytes': len(PROMPTS[key].encode())} for key in prompt_keys},
        'sampling': {'temperature': 0, 'seed': 3407, 'cache_prompt': False},
        'server_defaults': {
            'reasoning': 'off/medium',
            'target_kv': 'q8_0',
            'draft_kv': 'f16',
            'batch': 2048,
            'ubatch': 512,
            'parallel': 1,
            'spec_draft_p_min': 0.20,
        },
        'model': str(model),
        'model_sha256': file_sha256(model),
        'draft': str(draft),
        'draft_sha256': file_sha256(draft),
        'rdna_release': 'v16-ebbb18522-r2',
        'rdna_tree': git_value(ROOT / 'llama-rdna-boosts', 'HEAD^{tree}'),
        'note': 'Short prompts in an allocated context, not a filled-context test. Two 2000-token axes meet the documented minimum length but are narrower than the upstream four-axis adaptive-MTP gate.',
    }
    (output / 'experiment.json').write_text(json.dumps(experiment, indent=2, default=str))

    rows = []
    failures = []
    for profile in profiles:
        label = profile['label']
        directory = output / label
        directory.mkdir()
        env = os.environ.copy()
        env.update(
            MODEL_QUANT='byteshape',
            SPEC_TYPE=profile['spec_type'],
            SPEC_DRAFT_N_MAX=str(profile['depth']),
            SPEC_DRAFT_P_MIN='0.20',
            DRAFT=profile['draft'],
            PORT=str(args.port),
            HOST='127.0.0.1',
            CTX_SIZE=str(args.ctx),
            PARALLEL='1',
            CACHE_TYPE_K='q8_0',
            CACHE_TYPE_V='q8_0',
            DRAFT_CACHE_TYPE='f16',
            BATCH_SIZE='2048',
            UBATCH_SIZE='512',
            REASONING='off',
            REASONING_EFFORT='medium',
        )
        env.pop('MODEL', None)
        command = [str(profile['launcher'])]
        binary = profile['binary']
        manifest = {
            'command': command,
            'env': {key: env[key] for key in (
                'MODEL_QUANT', 'SPEC_TYPE', 'SPEC_DRAFT_N_MAX', 'SPEC_DRAFT_P_MIN',
                'DRAFT', 'CTX_SIZE', 'PARALLEL', 'CACHE_TYPE_K', 'CACHE_TYPE_V',
                'DRAFT_CACHE_TYPE', 'BATCH_SIZE', 'UBATCH_SIZE', 'REASONING',
                'REASONING_EFFORT')},
            'binary_sha256': file_sha256(binary),
            'git_head': git_value(profile['repository'], 'HEAD'),
            'git_tree': git_value(profile['repository'], 'HEAD^{tree}'),
            'gpu_before': gpu_snapshot(),
        }
        child = None
        try:
            print(f'START {label}', flush=True)
            with (directory / 'server.log').open('w') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 900
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'server startup exit {child.returncode}')
                try:
                    PROBE.request(base, '/health', timeout=2)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError('startup timeout')
                    time.sleep(1)
            manifest['props'] = PROBE.request(base, '/props')
            manifest['gpu_loaded'] = gpu_snapshot()
            cases = [('warmup', 'code', 128)]
            cases += [(f'{run + 1}-{key}', key, args.tokens)
                      for run in range(args.runs) for key in prompt_keys]
            for name, key, count in cases:
                payload = {
                    'messages': [{'role': 'user', 'content': PROMPTS[key]}],
                    'temperature': 0,
                    'seed': 3407,
                    'max_tokens': count,
                    'cache_prompt': False,
                }
                (directory / f'{name}-request.json').write_text(json.dumps(payload, indent=2))
                response = PROBE.request(base, '/v1/chat/completions', payload, timeout=max(600, count // 2))
                (directory / f'{name}.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
                manifest['gpu_after_request'] = gpu_snapshot()
                if name == 'warmup':
                    continue
                timings = response['timings']
                drafted = timings.get('draft_n', 0)
                accepted = timings.get('draft_n_accepted', 0)
                message = response['choices'][0]['message']
                row = {
                    'profile': label,
                    'scenario': key,
                    'run': int(name.split('-')[0]),
                    'prompt_tok_s': timings.get('prompt_per_second', 0),
                    'decode_tok_s': timings.get('predicted_per_second', 0),
                    'prompt_tokens': response.get('usage', {}).get('prompt_tokens'),
                    'completion_tokens': response.get('usage', {}).get('completion_tokens'),
                    'accepted': accepted,
                    'drafted': drafted,
                    'acceptance': accepted / drafted if drafted else 0,
                    'finish_reason': response['choices'][0]['finish_reason'],
                    'output_sha256': hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest(),
                }
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
            time.sleep(3)
            manifest['gpu_after_stop'] = gpu_snapshot()
            (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            (output / 'samples.json').write_text(json.dumps(rows, indent=2))
            (output / 'failures.json').write_text(json.dumps(failures, indent=2))

    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('profile', 'scenario', 'runs', 'prompt_tok_s', 'decode_tok_s', 'acceptance'))
        for profile in profiles:
            for key in prompt_keys:
                samples = [row for row in rows if row['profile'] == profile['label'] and row['scenario'] == key]
                if samples:
                    writer.writerow((profile['label'], key, len(samples), *(
                        round(statistics.median(row[field] for row in samples), 4)
                        for field in ('prompt_tok_s', 'decode_tok_s', 'acceptance'))))
    print(f'RESULTS {output}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
