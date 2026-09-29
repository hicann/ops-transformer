#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ARTIFACT_DIR=${QLIV2_ARTIFACT_DIR:-/tmp/ops_transformer_indexer/qli}
mkdir -p "$ARTIFACT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
export TMPDIR="$ARTIFACT_DIR"
export CANNBOTDSL_CACHE_DIR="$ARTIFACT_DIR/cache"
PYTHON=${PYTHON:-python3}
TESTCASE_DIR=${1:-"$ARTIFACT_DIR/pt_path"}
RESULT_XLSX=${2:-"$ARTIFACT_DIR/result.xlsx"}

mapfile -t CASE_FILES < <(find "$TESTCASE_DIR" -maxdepth 1 -name '*.pt' | sort)
[ ${#CASE_FILES[@]} -gt 0 ] || { echo "未找到PT用例: $TESTCASE_DIR"; exit 1; }
pass=0; fail=0
for case_file in "${CASE_FILES[@]}"; do
  echo "执行: $(basename "$case_file")"
  if QLIV2_TESTCASE_DIR="$TESTCASE_DIR" QLIV2_TESTCASE_PATH="$case_file" \
     QLIV2_RESULT_PATH="$RESULT_XLSX" "$PYTHON" -m pytest -q \
       "$SCRIPT_DIR/test_quant_lightning_indexer_v2_batch.py"; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
  fi
done
echo "总计=${#CASE_FILES[@]} 通过=$pass 失败=$fail"
[ "$fail" -eq 0 ]
