"""Download the pinned benchmark tables and verify every file before installation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

RELEASE = Path(__file__).with_name('release.json')


def safe_target(root: Path, relative: str) -> Path:
    root = root.resolve()
    target = (root / relative).resolve()
    if Path(relative).is_absolute() or target == root or root not in target.parents:
        raise ValueError(f'Unsafe dataset path: {relative}')
    return target


def verify_file(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Checksum mismatch for {path.name}')


def download(dataset: str, root: Path, cache_dir=None) -> list[Path]:
    from huggingface_hub import hf_hub_download
    release = json.loads(RELEASE.read_text())['datasets'][dataset]
    if not release.get('revision'):
        raise RuntimeError('This checkout has no pinned dataset revision yet.')
    common = {'repo_id': release['repo_id'], 'repo_type': 'dataset', 'revision': release['revision'],
              'cache_dir': str(cache_dir) if cache_dir else None}
    manifest_path = Path(hf_hub_download(filename='release_manifest.json', **common))
    manifest = json.loads(manifest_path.read_text())
    installed = []
    for entry in manifest['files']:
        relative = entry.get('restore_path')
        if relative is None:
            continue
        target = safe_target(root, relative)
        if target.exists():
            verify_file(target, entry['sha256'])
        else:
            source = Path(hf_hub_download(filename=entry['path'], **common))
            verify_file(source, entry['sha256'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            verify_file(target, entry['sha256'])
        installed.append(target)
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['notes', 'audio', 'all'], default='all')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    for name in (['notes', 'audio'] if args.dataset == 'all' else [args.dataset]):
        files = download(name, args.root, args.cache_dir)
        print(f'{name}: verified {len(files)} benchmark files under {args.root.resolve()}')


if __name__ == '__main__':
    main()
