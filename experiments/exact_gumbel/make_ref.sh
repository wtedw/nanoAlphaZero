#!/usr/bin/env bash
# Vendor upstream google-deepmind/mctx (commit 428bfb7) as `mctx_ref`, next to this script.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf ref _mctx_upstream
git clone -q https://github.com/google-deepmind/mctx.git _mctx_upstream
git -C _mctx_upstream checkout -q 428bfb7
mkdir ref && cp -r _mctx_upstream/mctx ref/mctx_ref && rm -rf ref/mctx_ref/_src/tests _mctx_upstream
grep -rl "mctx\._src\|from mctx\b\|import mctx" ref/mctx_ref | xargs sed -i \
  's/\bmctx\._src/mctx_ref._src/g; s/^from mctx import/from mctx_ref import/; s/^import mctx$/import mctx_ref as mctx/'
