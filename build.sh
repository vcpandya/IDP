#!/bin/bash
# Remove development/test directories not needed for deployment
rm -rf node_modules tests attached_assets cookbook tutorials logs pageindex \
       .cache .upm .local .config .replit_integration_files \
       __pycache__ .pytest_cache

# Remove unnecessary files
rm -f replit.md idpkit.db package.json package-lock.json

# Clean Python bytecode caches throughout the project
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
