#!/bin/bash
# Quick installation test for ClayTertorch

echo "🧪 Testing ClayTertorch installation..."

# Test 1: Check if CLI is available
if ! command -v clayterractorch &> /dev/null && [ ! -f clayterractorch.py ]; then
    echo "❌ clayterractorch not found. Run from project directory or install."
    exit 1
fi

echo "✅ clayterractorch command accessible"

# Test 2: Check Python imports (will fail in fresh env - this is expected)
echo "🔍 Testing core Python imports (expected to fail in fresh environment)..."
python3 -c "
import sys
try:
    import torch
    import numpy as np
    import pandas
    import yaml
    import rasterio
    import rioxarray
    import odc_stac
    import pystac_client
    import planetary_computer
    import tqdm
    import pyproj
    import sklearn
    print('✅ Basic scientific imports OK - environment already prepared')
except ImportError as e:
    # This is expected in a fresh clone - user needs to run setup
    print('ℹ️  Dependencies not installed yet - this is expected in fresh environment')
    # Don't exit - this is normal
"

# Test optional imports (also expected to fail)
python3 -c "
import sys
try:
    import terratorch
    print('✅ Terratorch import OK')
except ImportError:
    print('ℹ️  Terratorch not installed yet - run setup to install deps')
"

python3 -c "
import sys
try:
    from claymodel.module import ClayMAEModule
    print('✅ CLAY import OK')
except ImportError:
    print('ℹ️  CLAY model not installed yet - run setup to install deps')
"

# Test 3: Check CLI help
echo "🔍 Testing CLI help..."
python3 clayterractorch.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ CLI help works"
else
    echo "❌ CLI help failed"
    exit 1
fi

# Test 4: Check directory structure creation
echo "🔍 Testing setup function..."
python3 clayterractorch.py setup > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Setup completed successfully"
else
    echo "❌ Setup failed"
    exit 1
fi

echo ""
echo "🎉 CLI and setup functions work! To complete installation:"
echo ""
echo "1. Install dependencies:"
echo "   micromamba create -f environment.yml"
echo "   micromamba activate clay_terratorch"
echo ""
echo "2. Then run the full test:"
echo "   ./INSTALL_TEST.sh"
echo ""
echo "3. After dependencies are installed:"
echo "   cp example_config.yaml configs/source_to_target.yaml"
echo "   # Edit config for your AOI"
echo "   clayterractorch gen-cubes --config configs/source_to_target.yaml"
echo ""