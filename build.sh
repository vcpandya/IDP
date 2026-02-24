#!/bin/bash
set -e

pip install --no-cache-dir ".[postgres]"
pip install --no-cache-dir gunicorn uvicorn[standard]

echo "Verifying gunicorn..."
which gunicorn || python -m gunicorn --version || { echo "ERROR: gunicorn not found"; exit 1; }

mkdir -p .deploy-libs

GCC_LIB_DIR=$(gcc -print-file-name=libstdc++.so.6 2>/dev/null | xargs dirname 2>/dev/null || true)

for lib in libstdc++.so.6 libgcc_s.so.1; do
    src=""
    if [ -n "$GCC_LIB_DIR" ] && [ -f "$GCC_LIB_DIR/$lib" ]; then
        src="$GCC_LIB_DIR/$lib"
    fi
    if [ -z "$src" ] || [ ! -f "$src" ]; then
        src=$(ldconfig -p 2>/dev/null | grep -m1 "$lib" | awk '{print $NF}')
    fi
    if [ -n "$src" ] && [ -f "$src" ]; then
        cp -L "$src" .deploy-libs/
        echo "Copied $lib from $src"
    else
        echo "WARNING: $lib not found (non-fatal)"
    fi
done

echo "Checking .pythonlibs/lib/python3.10/site-packages..."
ls .pythonlibs/lib/python3.10/site-packages/idpkit/ > /dev/null
echo "idpkit package: OK"

rm -rf node_modules tests attached_assets cookbook tutorials logs pageindex \
       .cache .upm .replit_integration_files \
       .pytest_cache

rm -f replit.md idpkit.db package.json package-lock.json

find . -maxdepth 5 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -maxdepth 5 -type f -name "*.pyc" -delete 2>/dev/null || true

echo "Build complete"
