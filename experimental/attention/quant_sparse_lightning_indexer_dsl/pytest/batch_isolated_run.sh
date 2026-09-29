#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
ARTIFACT_DIR=${QSLI_ARTIFACT_DIR:-/tmp/ops_transformer_indexer/qsli}
mkdir -p "$ARTIFACT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
export TMPDIR="$ARTIFACT_DIR"
export CANNBOTDSL_CACHE_DIR="$ARTIFACT_DIR/cache"
PT_PATH=${1:-"$ARTIFACT_DIR/pt_path"}
DEVICE_ID=${2:-0}
shopt -s nullglob
files=("${PT_PATH}"/*.pt)
[ ${#files[@]} -gt 0 ] || { echo "No PT cases found: $PT_PATH" >&2; exit 1; }
failed=0
for file in "${files[@]}"; do
  echo "===== $(basename "${file}") ====="
  if ! QSLI_TESTCASE_PATH=${file} QSLI_TESTCASE_DIR=${PT_PATH} QSLI_DEVICE_ID=${DEVICE_ID} \
    "$PYTHON" -m pytest -rA -s "$SCRIPT_DIR/test_quant_sparse_lightning_indexer_batch.py" -v -m ci; then
    failed=$((failed + 1))
  fi
done
exit ${failed}
