#!/bin/bash

# Build and install torch_ishmem_extension
# Usage: ./build.sh [clean]

set -e

# Check required environment variables
if [ -z "$ISHMEM_ROOT" ]; then
    echo "Error: ISHMEM_ROOT is not set"
    echo "Please set ISHMEM_ROOT to your ISHMEM installation directory"
    exit 1
fi

# Try to find SYCL compiler root
if [ -z "$CMPLR_ROOT" ] && [ -z "$SYCL_ROOT" ]; then
    echo "Warning: Neither CMPLR_ROOT nor SYCL_ROOT is set"
    echo "Will try to detect from icpx location..."
fi

# Clean if requested
if [ "$1" == "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -rf build dist *.egg-info
    find . -name "*.so" -delete
    find . -name "*.o" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "Clean complete"
    exit 0
fi

# Build
echo "Building torch_ishmem_extension..."
echo "  ISHMEM_ROOT: $ISHMEM_ROOT"
echo "  MPI_ROOT: ${MPI_ROOT:-<not set>}"

# Force use icpx as the compiler
export CC=icx
export CXX=icpx
export LDSHARED="icpx -shared"

# Build and install in development mode
python setup.py develop

echo ""
echo "Build complete!"
echo "You can now import torch_ishmem_extension in Python"
