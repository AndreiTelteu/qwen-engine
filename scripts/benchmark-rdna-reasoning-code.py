#!/usr/bin/env python3
"""Screen llama-fork and RDNA boosts profiles on one long reasoning+code workload."""
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
PROMPT_PATH = ROOT / 'artifacts/rdna-boosts-v16-ebbb18522-r2/prompts/code-python.txt'
MODEL = ROOT / 'llama-hip/models/qwen3.8-27b-byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf'
DRAFT = ROOT / 'llama-hip/models/qwen3.8-27b-q4_0/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf'


def profile(label, engine='rdna', **overrides):
    values = {
        'label': label,
        'engine': engine,
        'SPEC_TYPE': 'draft-dflash',
        'SPEC_DRAFT_N_MAX': '3',
        'SPEC_DRAFT_P_MIN': '0.20',
        'DRAFT': str(DRAFT),
        'BATCH_SIZE': '2048',
        'UBATCH_SIZE': '512',
        'CACHE_TYPE_K': 'q8_0',
        'CACHE_TYPE_V': 'q8_0',
        'DRAFT_CACHE_TYPE': 'f16',
        'FLASH_ATTN': 'on',
        'LOAD_MODE': 'none',
        'GGML_CUDA_GDN_CHUNKED_BF16': '1',
    }
    values.update({key: str(value) for key, value in overrides.items()})
    return values


PROFILES = [
    profile('fork-reference', engine='fork'),
    profile('rdna-reference'),
    profile('rdna-ub256', UBATCH_SIZE=256),
    profile('rdna-ub1024', UBATCH_SIZE=1024),
    profile('rdna-ub2048', UBATCH_SIZE=2048),
    profile('rdna-b1024-ub512', BATCH_SIZE=1024),
    profile('rdna-dflash-n4', SPEC_DRAFT_N_MAX=4),
    profile('rdna-dflash-n5', SPEC_DRAFT_N_MAX=5),
    profile('rdna-dflash-n6', SPEC_DRAFT_N_MAX=6),
    profile('rdna-dflash-n7', SPEC_DRAFT_N_MAX=7),
    profile('rdna-n5-ub256', SPEC_DRAFT_N_MAX=5, UBATCH_SIZE=256),
    profile('rdna-n5-ub1024', SPEC_DRAFT_N_MAX=5, UBATCH_SIZE=1024),
    profile('rdna-n5-native-off', SPEC_DRAFT_N_MAX=5, GGML_CUDA_FA_KV_NATIVE=0),
    profile('rdna-n5-draft-kv-q8', SPEC_DRAFT_N_MAX=5, DRAFT_CACHE_TYPE='q8_0'),
    profile('rdna-pmin010', SPEC_DRAFT_P_MIN='0.10'),
    profile('rdna-kv-q4', CACHE_TYPE_K='q4_0', CACHE_TYPE_V='q4_0'),
    profile('rdna-kv-bf16', CACHE_TYPE_K='bf16', CACHE_TYPE_V='bf16', GGML_CUDA_FA_KV_NATIVE=1),
    profile('rdna-native-kv-off', GGML_CUDA_FA_KV_NATIVE=0),
    profile('rdna-lazy-off', LAZY_MODE='off'),
    profile('rdna-gdn-fp32', GGML_CUDA_GDN_CHUNKED_BF16=0),
    profile('rdna-draft-kv-q8', DRAFT_CACHE_TYPE='q8_0'),
    profile('rdna-mtp3', SPEC_TYPE='draft-mtp', SPEC_DRAFT_N_MAX=3, DRAFT='embedded'),
    profile('rdna-mtp-adaptive7', SPEC_TYPE='draft-mtp-adaptive', SPEC_DRAFT_N_MAX=7, DRAFT='embedded'),
]


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git_value(repository, value):
    return subprocess.check_output(['git', '-C', repository, 'rev-parse', value], text=True).strip()


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


def reasoning_text(message):
    for key in ('reasoning_content', 'reasoning'):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=768)
    parser.add_argument('--ctx', type=int, default=131072)
    parser.add_argument('--port', type=int, default=8095)
    parser.add_argument('--name', default=time.strftime('reasoning-code-%Y%m%d-%H%M%S'))
    parser.add_argument('--profiles', help='Comma-separated labels; default is the full screening matrix')
    args = parser.parse_args()
    if min(args.runs, args.tokens, args.ctx) < 1:
        parser.error('runs, tokens and ctx must be positive')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('name must be a simple label')
    if not PROMPT_PATH.is_file() or not MODEL.is_file() or not DRAFT.is_file():
        raise FileNotFoundError('Release prompt, ByteShape target, or DFlash2 draft is missing')

    profiles = list(PROFILES)
    if args.profiles:
        selected = args.profiles.split(',')
        known = {item['label']: item for item in profiles}
        unknown = set(selected) - set(known)
        if unknown:
            parser.error(f'Unknown profiles: {sorted(unknown)}')
        profiles = [known[label] for label in selected]
    if subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True).returncode == 0:
        raise RuntimeError('An existing llama-server is running; stop it before benchmarking')

    output = ROOT / 'artifacts/rdna-reasoning-code' / args.name
    output.mkdir(parents=True, exist_ok=False)
    base = f'http://127.0.0.1:{args.port}'
    try:
        PROBE.request(base, '/health', timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError('Benchmark port is occupied')

    prompt = PROMPT_PATH.read_text()
    engine_info = {
        'fork': {
            'launcher': ROOT / 'start-llama-fork.sh',
            'binary': ROOT / 'llama-fork/build-rocm-gfx1100-portable/bin/llama-server',
            'repository': ROOT / 'llama-fork',
        },
        'rdna': {
            'launcher': ROOT / 'start-llama-rdna-boosts.sh',
            'binary': ROOT / 'llama-rdna-boosts/build-rocm-gfx1100/bin/llama-server',
            'repository': ROOT / 'llama-rdna-boosts',
        },
    }
    experiment = {
        'args': vars(args),
        'profiles': profiles,
        'prompt': {'path': str(PROMPT_PATH), 'bytes': len(prompt.encode()),
                   'sha256': hashlib.sha256(prompt.encode()).hexdigest()},
        'sampling': {'temperature': 0, 'seed': 3407, 'cache_prompt': False},
        'common': {'model_quant': 'byteshape', 'reasoning': 'on', 'reasoning_effort': 'medium',
                   'parallel': 1, 'split_mode': 'none', 'fit': 'off'},
        'model': {'path': str(MODEL), 'sha256': file_sha256(MODEL)},
        'draft': {'path': str(DRAFT), 'sha256': file_sha256(DRAFT)},
        'engines': {},
    }
    for key, info in engine_info.items():
        experiment['engines'][key] = {
            'git_head': git_value(info['repository'], 'HEAD'),
            'git_tree': git_value(info['repository'], 'HEAD^{tree}'),
            'binary_sha256': file_sha256(info['binary']),
        }
    (output / 'experiment.json').write_text(json.dumps(experiment, indent=2, default=str))

    rows = []
    failures = []
    for run in range(1, args.runs + 1):
        ordered = profiles if run % 2 else list(reversed(profiles))
        for item in ordered:
            label = item['label']
            directory = output / f'run-{run}' / label
            directory.mkdir(parents=True)
            info = engine_info[item['engine']]
            env = os.environ.copy()
            env.update(
                MODEL_QUANT='byteshape',
                PORT=str(args.port),
                HOST='127.0.0.1',
                CTX_SIZE=str(args.ctx),
                PARALLEL='1',
                SPLIT_MODE='none',
                FIT='off',
                REASONING='on',
                REASONING_EFFORT='medium',
            )
            for key, value in item.items():
                if key not in ('label', 'engine'):
                    env[key] = value
            env.pop('MODEL', None)
            manifest = {
                'profile': item,
                'command': [str(info['launcher'])],
                'env': {key: env[key] for key in sorted(set(item) | {
                    'MODEL_QUANT', 'CTX_SIZE', 'PARALLEL', 'SPLIT_MODE', 'FIT',
                    'REASONING', 'REASONING_EFFORT'}) if key in env},
                'gpu_before': gpu_snapshot(),
            }
            child = None
            started = time.monotonic()
            try:
                print(f'START run={run} profile={label}', flush=True)
                with (directory / 'server.log').open('w') as log:
                    child = subprocess.Popen([str(info['launcher'])], env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
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
                manifest['startup_seconds'] = time.monotonic() - started
                manifest['props'] = PROBE.request(base, '/props')
                manifest['gpu_loaded'] = gpu_snapshot()

                warmup = {'messages': [{'role': 'user', 'content': PROBE.PROMPTS['code']}],
                          'temperature': 0, 'seed': 3407, 'max_tokens': 128, 'cache_prompt': False}
                PROBE.request(base, '/v1/chat/completions', warmup, timeout=600)
                payload = {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                           'seed': 3407, 'max_tokens': args.tokens, 'cache_prompt': False}
                (directory / 'request.json').write_text(json.dumps(payload, indent=2))
                response = PROBE.request(base, '/v1/chat/completions', payload,
                                         timeout=max(900, args.tokens))
                (directory / 'response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
                timings = response['timings']
                drafted = timings.get('draft_n', 0)
                accepted = timings.get('draft_n_accepted', 0)
                message = response['choices'][0]['message']
                thought = reasoning_text(message)
                content = message.get('content') or ''
                row = {
                    'run': run,
                    'profile': label,
                    'engine': item['engine'],
                    'prompt_tok_s': timings.get('prompt_per_second', 0),
                    'decode_tok_s': timings.get('predicted_per_second', 0),
                    'prompt_tokens': response.get('usage', {}).get('prompt_tokens'),
                    'completion_tokens': response.get('usage', {}).get('completion_tokens'),
                    'accepted': accepted,
                    'drafted': drafted,
                    'acceptance': accepted / drafted if drafted else 0,
                    'reasoning_chars': len(thought),
                    'content_chars': len(content),
                    'finish_reason': response['choices'][0]['finish_reason'],
                    'output_sha256': hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest(),
                    'loaded_vram_mib': next((values['mem_info_vram_used'] / 1048576
                                             for values in manifest['gpu_loaded'].values()
                                             if values['pci'] == '0000:03:00.0'), None),
                }
                if not thought:
                    raise RuntimeError('response contained no reasoning text')
                rows.append(row)
                manifest['status'] = 'passed'
                manifest['gpu_after_request'] = gpu_snapshot()
                print(json.dumps(row), flush=True)
            except Exception as error:
                manifest.update(status='failed', error=str(error))
                failures.append({'run': run, 'profile': label, 'error': str(error)})
                print(f'FAIL run={run} profile={label}: {error}', flush=True)
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
        writer.writerow(('profile', 'engine', 'runs', 'prompt_tok_s', 'decode_tok_s',
                         'acceptance', 'reasoning_chars', 'content_chars', 'loaded_vram_mib'))
        for item in profiles:
            samples = [row for row in rows if row['profile'] == item['label']]
            if samples:
                writer.writerow((item['label'], item['engine'], len(samples), *(
                    round(statistics.median(row[field] for row in samples), 4)
                    for field in ('prompt_tok_s', 'decode_tok_s', 'acceptance',
                                  'reasoning_chars', 'content_chars', 'loaded_vram_mib'))))
    print(f'RESULTS {output}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
