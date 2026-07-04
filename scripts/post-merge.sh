#!/bin/bash
# Post-merge setup for IDP Kit.
# Runs after a task agent's changes are merged into main. Must be idempotent
# and non-interactive — stdin is closed.
set -euo pipefail

# The project's tools (pymupdf, tokenizers, …) need libstdc++ from gcc's
# Nix package; mirror what the workflow does so pip installs that compile
# native extensions can find it.
export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")":"${LD_LIBRARY_PATH:-}"

echo "[post-merge] Installing Python dependencies from requirements.txt..."
python -m pip install --quiet --disable-pip-version-check -r requirements.txt

# DB schema is auto-applied by idpkit.db.session.init_db() on app startup,
# so no manual migration step is needed here.

echo "[post-merge] Done."
