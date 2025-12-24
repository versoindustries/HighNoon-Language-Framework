#!/bin/bash
#
# Copyright 2025 Verso Industries
#
# This script checks the compiled custom operator binaries for any lingering
# C-style memory allocation symbols (malloc, free, realloc). Its purpose is
# to help enforce RAII patterns and modern C++ memory management.
#
# Usage: ./scripts/check_allocations.sh

set -euo pipefail

echo "--- Checking for C-style allocation symbols in custom op .so files ---"

SYMBOLS=$(nm -D build/_fused_*.so 2>/dev/null | grep -E ' (malloc|free|realloc|calloc)\b' || true)

if [[ -n "$SYMBOLS" ]]; then
    echo "FAIL: Found C-style memory allocation symbols:"
    echo "$SYMBOLS"
    exit 1
else
    echo "PASS: No C-style memory allocation symbols found."
    exit 0
fi
