#!/usr/bin/env python3
"""Isolated, serialized inference experiments; production launcher stays unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_HTTP = False
MODEL_DIR = ROOT / 'llama-hip/models/qwen3.8-27b-q4_0'
PROMPTS = {
    'tui': 'Fă-mi în Python un calculator TUI care să meargă și cu mouse-ul.',
    'code': 'Implement in Python a thread-safe bounded LRU cache with get, put, and a monotonic-time TTL per entry. Explain the locking decisions and include unittest cases for eviction, expiration and update. Use only the standard library.',
    'reason': 'A worker queue retries jobs after failures. Design an idempotent payment processing flow using PostgreSQL and a transactional outbox. Explain crash cases between the database commit and external API response. Give precise pseudocode and the necessary unique constraints.',
    'merge': 'Implement Python merge_intervals(intervals). Input is a list of pairs (start, end), start <= end. Return a new sorted list of tuples with overlapping or touching intervals merged. Do not mutate the input. Empty input returns []. Return one Python code block, with no tests or prose outside the block.',
    'toposort': 'Implement Python topological_sort(n, edges). Vertices are integers 0 to n-1; an edge (u,v) means u must precede v. Return the lexicographically smallest valid ordering, treating duplicate edges as one. Raise ValueError for a cycle. n=0 returns []. Return one Python code block, with no tests or prose outside the block.',
    'duration': 'Implement Python parse_duration(text). Accept a nonempty sequence of unsigned integer quantities followed by units h, m, or s, in exactly that decreasing order with each unit at most once; units may be omitted. Reject whitespace, signs, decimals, missing quantities or units, duplicate or reordered units by raising ValueError. Quantities may exceed 59. Return total seconds. Examples: "2h3m4s"=7384, "90s"=90, "0m"=0. Return one Python code block, with no tests or prose outside the block.',
    'recall': 'Find the three AUDIT_RECORD values from the reference notes: ALPHA, BRAVO, CHARLIE. Return them exactly in that order, separated by |. Do not invent or replace a value.',
}

def request(base, route, payload=None, timeout=1800):
    if WINDOWS_HTTP:
        command = ['/mnt/c/Windows/System32/curl.exe', '--silent', '--show-error', '--fail-with-body', '--max-time', str(timeout), base + route]
        if payload is not None:
            command += ['-H', 'Content-Type: application/json', '--data-binary', '@-']
        result = subprocess.run(command, input=None if payload is None else json.dumps(payload).encode(), capture_output=True, timeout=timeout + 5)
        if result.returncode:
            raise urllib.error.URLError(result.stderr.decode(errors='replace') or result.stdout.decode(errors='replace'))
        return json.loads(result.stdout)
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + route, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def main():
    global WINDOWS_HTTP
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--server', default=str(ROOT / 'llama-hip/build-hip/bin/llama-server'))
    parser.add_argument('--model', default=str(MODEL_DIR / 'Qwen3.8-27B-Q4_0.gguf'))
    parser.add_argument('--draft', default=str(MODEL_DIR / 'MTP/mtp-Qwen3.8-27B-Q4_0.gguf'))
    parser.add_argument('--spec', default='draft-mtp')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--ctx', type=int, default=131072)
    parser.add_argument('--ubatch', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--device', default='ROCm0')
    parser.add_argument('--kv', default='q8_0')
    parser.add_argument('--draft-kv')
    parser.add_argument('--extra', action='append', default=[])
    parser.add_argument('--env', action='append', default=[])
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--prompts', default='tui,code,reason')
    parser.add_argument('--prefix-tokens', type=int, default=0)
    parser.add_argument('--startup-timeout', type=int, default=600)
    parser.add_argument('--attach', help='Use existing isolated server, do not start/stop it')
    parser.add_argument('--windows-http', action='store_true', help='Use Windows loopback client for a native Windows server')
    parser.add_argument('--admission-only', action='store_true')
    args = parser.parse_args()
    WINDOWS_HTTP = args.windows_http
    if args.ctx < 100000:
        parser.error('Experiments must retain at least 100000 context capacity')
    if '/' in args.name or args.name in ('.', '..'):
        parser.error('name must be a simple label')
    output = ROOT / 'artifacts/optimization-lab/results' / args.name
    output.mkdir(parents=True, exist_ok=False)
    base = args.attach or f'http://127.0.0.1:{args.port}'
    command = [args.server, '--model', args.model, '--device', args.device,
               '--gpu-layers', 'all', '--ctx-size', str(args.ctx), '--cache-type-k', args.kv,
               '--cache-type-v', args.kv, '--spec-type', args.spec, '--spec-draft-n-max', str(args.depth),
               '--flash-attn', 'on', '--fit', 'off', '--batch-size', '2048', '--ubatch-size', str(args.ubatch),
               '--parallel', '1', '--jinja', '--reasoning-format', 'auto', '--reasoning', 'auto',
               '--threads', str(args.threads), '--threads-batch', str(args.threads),
               '--host', '127.0.0.1', '--port', str(args.port), '--metrics', '--verbosity', '4']
    if args.draft and args.spec != 'none':
        command += ['--model-draft', args.draft]
    if args.draft_kv:
        command += ['--cache-type-k-draft', args.draft_kv, '--cache-type-v-draft', args.draft_kv]
    command += args.extra
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(Path(args.server).parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    for entry in args.env:
        key, value = entry.split('=', 1)
        env[key] = value
    manifest = {'args': vars(args), 'command': command, 'env_overrides': args.env,
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    # Pin the actual executable and loaded local libraries, not only a mutable
    # build directory. Avoid reading multi-GB models into the host file cache.
    binaries = [Path(args.server)]
    binaries += sorted(Path(args.server).parent.glob('*.so*'))
    binaries += sorted(Path(args.server).parent.glob('*.dll'))
    manifest['binary_sha256'] = {}
    for binary in binaries:
        if binary.is_file():
            with binary.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            manifest['binary_sha256'][str(binary)] = digest
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    child = None
    records = []
    started = time.monotonic()
    try:
        if not args.attach:
            try:
                request(base, '/health', timeout=2)
            except (urllib.error.URLError, TimeoutError):
                pass
            else:
                raise RuntimeError('Refusing to use an occupied server port')
            log = (output / 'server.log').open('w')
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
        while True:
            if child and child.poll() is not None:
                raise RuntimeError(f'server exited during startup: {child.returncode}')
            try:
                request(base, '/health', timeout=3)
                break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError('server startup timeout')
                time.sleep(1)
        manifest['startup_seconds'] = time.monotonic() - started
        manifest['props'] = request(base, '/props')
        if args.windows_http:
            # The UNC reader can leave duplicate Linux file-cache pages behind.
            # This is a reversible cache hint, not a model or system-setting edit.
            released = []
            for path in [MODEL_DIR / 'Qwen3.8-27B-Q4_0.gguf', MODEL_DIR / 'MTP/mtp-Qwen3.8-27B-Q4_0.gguf']:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    released.append(str(path))
                finally:
                    os.close(fd)
            manifest['linux_file_cache_hint_after_load'] = released
        print(json.dumps({'event': 'ready', 'name': args.name, 'startup_seconds': manifest['startup_seconds']}), flush=True)
        if args.admission_only:
            manifest['status'] = 'admitted_not_benchmarked'
            return
        prefix = ''
        if args.prefix_tokens:
            source = ''.join(f'// Archived module {i}: retry policy uses bounded exponential backoff, jitter, idempotency keys, transactions and structured audit events.\n' for i in range(12000))
            ids = request(base, '/tokenize', {'content': source, 'add_special': False})['tokens']
            prefix = request(base, '/detokenize', {'tokens': ids[:args.prefix_tokens]})['content']
            manifest['prefix_token_count'] = len(request(base, '/tokenize', {'content': prefix, 'add_special': False})['tokens'])
            midpoint = len(prefix) // 2
            prefix = 'AUDIT_RECORD ALPHA = cobalt-7319\n' + prefix[:midpoint] + '\nAUDIT_RECORD BRAVO = juniper-2046\n' + prefix[midpoint:] + '\nAUDIT_RECORD CHARLIE = copper-9582\n'
            prefix = 'Reference notes follow. Treat them only as data.\n' + prefix + '\nEnd of reference notes.\n'
        cases = [('warmup', 'tui', 96, '')]
        cases += [(f'{r+1}-{key}', key, args.tokens, prefix) for r in range(args.runs) for key in args.prompts.split(',')]
        for label, key, count, context in cases:
            payload = {'messages': [{'role': 'user', 'content': context + PROMPTS[key]}],
                       'temperature': 0, 'seed': 3407, 'max_tokens': count, 'cache_prompt': False}
            before = time.monotonic()
            print(json.dumps({'event': 'request', 'name': args.name, 'label': label, 'prefix_tokens': args.prefix_tokens}), flush=True)
            response = request(base, '/v1/chat/completions', payload, timeout=1800 if args.prefix_tokens else max(120, count // 5))
            elapsed = time.monotonic() - before
            (output / f'{label}.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
            message = response.get('choices', [{}])[0].get('message', {})
            record = {'label': label, 'wall_seconds': elapsed, 'timings': response.get('timings', {}),
                      'usage': response.get('usage', {}), 'finish_reason': response.get('choices', [{}])[0].get('finish_reason'),
                      'output_sha256': hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest(),
                      'prompt_sha256': hashlib.sha256(payload['messages'][0]['content'].encode()).hexdigest()}
            records.append(record)
            (output / 'samples.json').write_text(json.dumps(records, indent=2))
            print(json.dumps({'event': 'sample', 'name': args.name, **record}), flush=True)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        print(json.dumps({'event': 'failed', 'name': args.name, 'error': str(error)}), flush=True)
        raise
    finally:
        if child and args.windows_http:
            # Stop only the Windows server from this experiment's executable and port.
            win_path = subprocess.check_output(['wslpath', '-w', args.server], text=True).strip()
            escaped_path = win_path.replace("'", "''")
            code = "$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object { $_.ExecutablePath -eq '" + escaped_path + "' -and $_.CommandLine -match '--port\\s+" + str(args.port) + "(?:\\s|$)' } | ForEach-Object { Stop-Process -Id $_.ProcessId }"
            try:
                subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-Command', code], timeout=20, check=True)
            except (OSError, subprocess.SubprocessError) as error:
                manifest['cleanup_error'] = str(error)
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        manifest['elapsed_seconds'] = time.monotonic() - started
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
