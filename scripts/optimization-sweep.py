#!/usr/bin/env python3
"""Run chosen candidates one at a time and retain failures as experimental data."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'artifacts/optimization-lab'
PATCHED = str(LAB / 'rdna-patch/bin-narrow/llama-server')
FOLDED = str(LAB / 'rdna-patch/build-hip-rdna3-mmvq/bin/llama-server')
DFLASH = str(LAB / 'models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf')
ROCMFPX_SERVER = str(LAB / 'rocmfpx/build-rdna3/bin/llama-server')
ROCMFPX_MODEL = str(LAB / 'rocmfpx/models/Qwen3.8-27B-ROCmFP4-FAST.gguf')
WIN_SERVER = '/mnt/c/Users/Andrei/AppData/Local/qwen-engine-vulkan-b10793/llama-server.exe'
WIN_MODEL = r'\\wsl.localhost\Ubuntu\home\andrei\qwen-engine\llama-hip\models\qwen3.8-27b-q4_0\Qwen3.8-27B-Q4_0.gguf'
WIN_DRAFT = r'\\wsl.localhost\Ubuntu\home\andrei\qwen-engine\llama-hip\models\qwen3.8-27b-q4_0\MTP\mtp-Qwen3.8-27B-Q4_0.gguf'
VULKAN = ['--server', WIN_SERVER, '--model', WIN_MODEL, '--draft', WIN_DRAFT, '--device', 'Vulkan0', '--windows-http']
PROFILES = {
    'hip-baseline': [],
    'hip-upstream': ['--server', str(LAB / 'upstream/build-hip-gfx1100-release/bin/llama-server')],
    'hip-upstream-ub512': ['--server', str(LAB / 'upstream/build-hip-gfx1100-release/bin/llama-server'),
                           '--ubatch', '512'],
    'hip-mtp1': ['--depth', '1'],
    'hip-mtp3': ['--depth', '3'],
    'hip-mtp4': ['--depth', '4'],
    'hip-mtp5': ['--depth', '5'],
    'hip-embedded-mtp': ['--draft', ''],
    'hip-draft-q8': ['--draft-kv', 'q8_0'],
    'hip-ub512': ['--ubatch', '512'],
    'hip-threads4': ['--threads', '4'],
    'hip-ngram': ['--spec', 'draft-mtp,ngram-mod'],
    'hip-ar': ['--spec', 'none'],
    'hip-dflash-f16': ['--spec', 'draft-dflash', '--draft', DFLASH, '--depth', '7'],
    'hip-dflash-q8': ['--spec', 'draft-dflash', '--draft', DFLASH, '--depth', '7', '--draft-kv', 'q8_0'],
    'hip-dflash-q8-ub512': ['--spec', 'draft-dflash', '--draft', DFLASH, '--depth', '7', '--draft-kv', 'q8_0', '--ubatch', '512'],
    'hip-dflash-q8-ub512-b4': ['--spec', 'draft-dflash', '--draft', DFLASH, '--depth', '3', '--draft-kv', 'q8_0', '--ubatch', '512'],
    'hip-dflash-q8-100k': ['--spec', 'draft-dflash', '--draft', DFLASH, '--depth', '7', '--draft-kv', 'q8_0', '--ctx', '102400', '--ubatch', '512'],
    'hip-dflash-weightq8': ['--spec', 'draft-dflash', '--draft', str(LAB / 'models/Qwen3.8-27B-DFlash2-Q8_0.gguf'), '--depth', '7', '--draft-kv', 'q8_0', '--ctx', '102400', '--ubatch', '512'],
    'hip-patch2': ['--server', PATCHED],
    'hip-patch4': ['--server', PATCHED, '--depth', '4'],
    'hip-fold2': ['--server', FOLDED],
    'hip-fold2-ub512': ['--server', FOLDED, '--ubatch', '512'],
    'hip-fold4': ['--server', FOLDED, '--depth', '4'],
    'hip-fold-dflash': ['--server', FOLDED, '--spec', 'draft-dflash', '--draft', DFLASH, '--depth', '7', '--draft-kv', 'q8_0', '--ubatch', '512'],
    'hip-fold-dflash-b3': ['--server', FOLDED, '--spec', 'draft-dflash', '--draft', DFLASH,
                           '--depth', '3', '--draft-kv', 'q8_0', '--ubatch', '512'],
    'hip-patch-dflash': ['--server', PATCHED, '--spec', 'draft-dflash', '--draft', DFLASH, '--depth', '7', '--draft-kv', 'q8_0', '--ctx', '102400', '--ubatch', '512'],
    'hip-iq4nl': ['--model', str(LAB / 'models/Qwen3.8-27B-IQ4_NL.gguf')],
    'hip-iq4nl-ub512': ['--model', str(LAB / 'models/Qwen3.8-27B-IQ4_NL.gguf'),
                        '--ubatch', '512'],
    'rocmfpx-ar': ['--server', ROCMFPX_SERVER, '--model', ROCMFPX_MODEL,
                   '--spec', 'none', '--draft', '', '--ubatch', '512'],
    'rocmfpx-dflash': ['--server', ROCMFPX_SERVER, '--model', ROCMFPX_MODEL,
                       '--spec', 'draft-dflash', '--draft', DFLASH, '--depth', '3',
                       '--draft-kv', 'q8_0', '--ctx', '102400', '--ubatch', '512'],
    'vulkan-mtp2': VULKAN,
    'vulkan-mtp4': VULKAN + ['--depth', '4'],
    'vulkan-block4g': VULKAN + ['--env', 'GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296'],
    'vulkan-dflash': VULKAN + ['--spec', 'draft-dflash', '--draft', r'\\wsl.localhost\Ubuntu\home\andrei\qwen-engine\artifacts\optimization-lab\models\Qwen3.8-27B-DFlash2-Q4_K_M.gguf', '--depth', '7', '--draft-kv', 'q8_0', '--ctx', '102400', '--ubatch', '512'],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('profiles', nargs='+', choices=PROFILES)
    parser.add_argument('--suffix', default='screen')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--prompts', default='tui,code,reason')
    parser.add_argument('--prefix-tokens', type=int, default=0)
    args = parser.parse_args()
    for profile in args.profiles:
        command = [sys.executable, str(ROOT / 'scripts/optimization-probe.py'), '--name', profile + '-' + args.suffix,
                   '--runs', str(args.runs), '--tokens', str(args.tokens), '--prompts', args.prompts,
                   '--prefix-tokens', str(args.prefix_tokens)] + PROFILES[profile]
        result = subprocess.run(command, cwd=ROOT)
        print(f'PROFILE_FINISHED {profile} exit={result.returncode}', flush=True)

if __name__ == '__main__':
    main()
