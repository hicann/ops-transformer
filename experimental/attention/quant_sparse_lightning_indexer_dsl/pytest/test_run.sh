#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

PYTHON=${PYTHON:-python3}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ARTIFACT_DIR=${QSLI_ARTIFACT_DIR:-/tmp/ops_transformer_indexer/qsli}
mkdir -p "$ARTIFACT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
export TMPDIR="$ARTIFACT_DIR"
export CANNBOTDSL_CACHE_DIR="$ARTIFACT_DIR/cache"

MODE=${1:-single}
shift || true
EXCEL=""
PT_PATH="$ARTIFACT_DIR/pt_path"
SHEET="TestCases"
CASES=""
INDEXES=""
DEVICE_ID=0
while getopts "E:P:S:C:I:D:" opt; do
  case ${opt} in
    E) EXCEL=${OPTARG} ;;
    P) PT_PATH=${OPTARG} ;;
    S) SHEET=${OPTARG} ;;
    C) CASES=${OPTARG} ;;
    I) INDEXES=${OPTARG} ;;
    D) DEVICE_ID=${OPTARG} ;;
  esac
done
export QSLI_DEVICE_ID=${DEVICE_ID}
export QSLI_CASE_NAMES=${CASES}
export QSLI_CASE_INDEXES=${INDEXES}
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${OPS_TRANSFORMER_TORCH_EXTENSION_DIR:-}" ]; then
  export PYTHONPATH="$OPS_TRANSFORMER_TORCH_EXTENSION_DIR:$PYTHONPATH"
fi

case ${MODE} in
  single)
    "$PYTHON" -m pytest -rA -s "$SCRIPT_DIR/test_quant_sparse_lightning_indexer_single.py" -v -m ci
    ;;
  batch)
    if [[ -n ${EXCEL} ]]; then
      "$PYTHON" "$SCRIPT_DIR/batch/quant_sparse_lightning_indexer_pt_save.py" "${EXCEL}" "${PT_PATH}" --sheet "${SHEET}" --indexes "${INDEXES}"
    fi
    QSLI_CASE_INDEXES="" QSLI_TESTCASE_DIR=${PT_PATH} "$PYTHON" -m pytest -rA -s "$SCRIPT_DIR/test_quant_sparse_lightning_indexer_batch.py" -v -m ci
    ;;
  batch_exec)
    [[ -n ${EXCEL} ]] || { echo "batch_exec requires -E" >&2; exit 2; }
    QSLI_PT_FILE_LIST=$("$PYTHON" "$SCRIPT_DIR/batch/list_pt_from_excel.py" "${EXCEL}" "${PT_PATH}" --sheet "${SHEET}")
    export QSLI_PT_FILE_LIST
    QSLI_TESTCASE_DIR=${PT_PATH} "$PYTHON" -m pytest -rA -s "$SCRIPT_DIR/test_quant_sparse_lightning_indexer_batch.py" -v -m ci
    ;;
  *) echo "usage: bash test_run.sh {single|batch|batch_exec} [-E xlsx] [-P pt_dir] [-S sheet] [-C names] [-I indexes] [-D device]" >&2; exit 2 ;;
esac
