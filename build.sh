#!/bin/bash
set -e

# Install Python package and dependencies (into .pythonlibs via Replit's pip config)
pip install --no-cache-dir ".[postgres]"

# Copy system shared libraries needed at runtime (these exist in dev but not deployment)
mkdir -p .deploy-libs
for lib in libstdc++.so.6 libgcc_s.so.1; do
    src=$(find /lib /usr/lib /nix/store -name "$lib" -not -path "*/32/*" 2>/dev/null | head -1)
    if [ -n "$src" ]; then
        cp -L "$src" .deploy-libs/
        echo "Copied $lib from $src"
    else
        echo "WARNING: $lib not found"
    fi
done

# Verify the deployment-critical paths exist
echo "Checking .pythonlibs/lib/python3.10/site-packages..."
ls .pythonlibs/lib/python3.10/site-packages/idpkit/ > /dev/null
echo "idpkit package: OK"
ls .deploy-libs/libstdc++.so.6 > /dev/null
echo "libstdc++: OK"

# Remove development/test directories not needed for deployment
rm -rf node_modules tests attached_assets cookbook tutorials logs pageindex \
       .cache .upm .replit_integration_files \
       .pytest_cache

# Remove unnecessary files
rm -f replit.md idpkit.db package.json package-lock.json

# Clean Python bytecode caches throughout the project
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
