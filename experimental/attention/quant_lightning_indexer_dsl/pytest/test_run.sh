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
ARTIFACT_DIR=${QLIV2_ARTIFACT_DIR:-/tmp/ops_transformer_indexer/qli}
mkdir -p "$ARTIFACT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
export TMPDIR="$ARTIFACT_DIR"
export CANNBOTDSL_CACHE_DIR="$ARTIFACT_DIR/cache"
PYTHON=${PYTHON:-python3}
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${OPS_TRANSFORMER_TORCH_EXTENSION_DIR:-}" ]; then
  export PYTHONPATH="$OPS_TRANSFORMER_TORCH_EXTENSION_DIR:$PYTHONPATH"
fi

COMMAND=${1:-help}
shift || true
EXCEL=""; SHEET="Sheet1"; PT_PATH="$ARTIFACT_DIR/pt_path"
RESULT="$ARTIFACT_DIR/result.xlsx"; CASES=""; INDEXES=""; SAVE_PT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -E|--excel) EXCEL="$2"; shift 2 ;;
    -S|--sheet) SHEET="$2"; shift 2 ;;
    -P|--pt-path) PT_PATH="$2"; shift 2 ;;
    -O|--output) RESULT="$2"; shift 2 ;;
    -C|--cases) CASES="$2"; shift 2 ;;
    -I|--indexes) INDEXES="$2"; shift 2 ;;
    --save-pt) SAVE_PT="$2"; shift 2 ;;
    -M|--run-mode) [ "$2" = eager ] || { echo "DSL QLI仅支持eager"; exit 2; }; shift 2 ;;
    -h|--help) COMMAND=help; shift ;;
    *) echo "未知选项: $1"; exit 2 ;;
  esac
done

case "$COMMAND" in
  single)
    QLIV2_CASE_NAMES="$CASES" QLIV2_SINGLE_SAVE_PT_DIR="$SAVE_PT" \
    QLIV2_SINGLE_RESULT_PATH="$RESULT" "$PYTHON" -m pytest -rA -s \
      "$SCRIPT_DIR/test_quant_lightning_indexer_v2_single.py" -v -m ci ;;
  batch)
    if [ -n "$EXCEL" ]; then
      "$PYTHON" "$SCRIPT_DIR/batch/quant_lightning_indexer_v2_pt_save.py" \
        "$EXCEL" "$PT_PATH" --sheet "$SHEET" --indexes "$INDEXES"
    fi
    QLIV2_TESTCASE_DIR="$PT_PATH" QLIV2_CASE_NAMES="$CASES" \
    QLIV2_CASE_INDEXES="" QLIV2_RESULT_PATH="$RESULT" \
      "$PYTHON" -m pytest -rA -s "$SCRIPT_DIR/test_quant_lightning_indexer_v2_batch.py" -v -m ci ;;
  batch_exec)
    [ -n "$EXCEL" ] || { echo "batch_exec必须指定--excel"; exit 2; }
    FILES=$("$PYTHON" "$SCRIPT_DIR/batch/list_pt_from_excel.py" "$EXCEL" "$PT_PATH" --sheet "$SHEET")
    QLIV2_TESTCASE_DIR="$PT_PATH" QLIV2_PT_FILE_LIST="$FILES" QLIV2_RESULT_PATH="$RESULT" \
      "$PYTHON" -m pytest -rA -s "$SCRIPT_DIR/test_quant_lightning_indexer_v2_batch.py" -v -m ci ;;
  help)
    echo "用法: $0 single|batch|batch_exec [-E excel] [-S sheet] [-P pt_dir] [-O result.xlsx] [-C names] [-I indexes] [--save-pt dir]" ;;
  *) echo "未知命令: $COMMAND"; exit 2 ;;
esac
