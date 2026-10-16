"""Prepare an isolated, pinned upstream oracle for exact-search tests."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

destination = Path(sys.argv[1]).resolve() / 'mctx_ref'
if destination.exists():
    raise SystemExit(f'Refusing to replace {destination}')
with tempfile.TemporaryDirectory(prefix='nanoaz-mctx-') as temporary:
    subprocess.run(['git', 'clone', '-q', 'https://github.com/google-deepmind/mctx.git', temporary], check=True)
    subprocess.run(['git', '-C', temporary, 'checkout', '-q',
                    '428bfb7e1c931715cd3d74f9c65e9991afd86df8'], check=True)
    shutil.copytree(Path(temporary) / 'mctx', destination)
    for path in destination.rglob('*.py'):
        path.write_text(path.read_text().replace('mctx._src', 'mctx_ref._src')
                        .replace('from mctx import', 'from mctx_ref import'))
print(f'Use PYTHONPATH={destination.parent}')
