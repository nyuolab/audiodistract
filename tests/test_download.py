from pathlib import Path
import hashlib
import pytest
from llm_distract.download import safe_target, verify_file


def test_rejects_traversal_absolute_and_symlink_escape(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'outside').symlink_to(tmp_path, target_is_directory=True)
    for name in ['../escape', '/tmp/escape', 'outside/escape']:
        with pytest.raises(ValueError, match='Unsafe'):
            safe_target(root, name)
    assert safe_target(root, 'data/pairs.parquet') == root / 'data/pairs.parquet'


def test_corruption_is_rejected(tmp_path):
    path = tmp_path / 'file'
    path.write_bytes(b'benchmark')
    expected = hashlib.sha256(b'benchmark').hexdigest()
    verify_file(path, expected)
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Checksum'):
        verify_file(path, expected)
