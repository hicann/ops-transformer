#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
    cat <<'EOF'
Usage: bash test_run.sh <command> [options] [-- pytest-options]
  single       Generate inputs/golden and run current paramset (or --excel).
  batch_save   Save CPU inputs/golden only; no NPU or installed wheel required.
  batch_exec   Replay saved .pt files without Excel, input generation or golden.
  batch        Save then replay each case; keep all saved data.
Options:
  --excel FILE       Workbook (batch_save/batch default: whitebox cases).
  --sheet NAME       Sheet name (default: decode).
  --pt-dir DIR       Data directory (default: tests/mqsmla_testcase).
  --device-id ID     NPU index (default: 0).
  --metadata-backend BACKEND
                    Metadata implementation: aicpu (default) or python.
  --verify-metadata-backends
                    Compare Python metadata with AICPU metadata before attention.
  --                 Forward remaining arguments to pytest, e.g. -k SHAPE_S1_03.
Load the CANN/Python environment first. Paths supplied by the caller are relative
to the caller's working directory. PYTHON selects the Python executable.
Saved data is kept; repeated saves replace the same case's file atomically.
batch_exec replays every .pt in its directory in filename order; use a separate
directory for each suite. DSL files use their own format, not mainline .pt format.
EOF
}

case "$command" in
    help|-h|--help) usage; exit 0 ;;
    single) data_mode=run ;;
    batch_save) data_mode=save ;;
    batch_exec) data_mode=replay ;;
    batch) data_mode=save-run ;;
    *) echo "Unknown command: $command" >&2; usage >&2; exit 2 ;;
esac

excel="${MQSMLA_EXCEL:-}"
sheet="${MQSMLA_SHEET:-decode}"
pt_dir="${MQSMLA_PT_DIR:-$TEST_DIR/mqsmla_testcase}"
device_id=0
metadata_backend="${MQSMLA_METADATA_BACKEND:-aicpu}"
verify_metadata_backends=false
pytest_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --excel|--sheet|--pt-dir|--device-id|--metadata-backend)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "Missing value for $1" >&2; exit 2
            fi
            case "$1" in
                --excel) excel="$2" ;;
                --sheet) sheet="$2" ;;
                --pt-dir) pt_dir="$2" ;;
                --device-id) device_id="$2" ;;
                --metadata-backend) metadata_backend="$2" ;;
            esac
            shift 2 ;;
        --verify-metadata-backends) verify_metadata_backends=true; shift ;;
        --) shift; pytest_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1; put pytest options after --" >&2; exit 2 ;;
    esac
done
if [[ ! "$device_id" =~ ^[0-9]+$ ]]; then
    echo '--device-id must be a non-negative integer' >&2; exit 2
fi
case "$metadata_backend" in
    aicpu|python) ;;
    *) echo '--metadata-backend must be aicpu or python' >&2; exit 2 ;;
esac
if [[ "$command" == batch_save || "$command" == batch ]]; then
    excel="${excel:-$TEST_DIR/excel/mqsmla_whitebox_cases.xlsx}"
fi
args=(-q -s --rootdir "$TEST_DIR" "$TEST_DIR/test_mqsmla.py" --data-mode "$data_mode"
      --pt-dir "$pt_dir" --device-id "$device_id")
args+=(--metadata-backend "$metadata_backend")
if [[ "$verify_metadata_backends" == true ]]; then
    args+=(--verify-metadata-backends)
fi
if [[ "$data_mode" != replay && -n "$excel" ]]; then
    args+=(--excel "$excel" --sheet "$sheet")
fi
exec "$PYTHON_BIN" -m pytest "${args[@]}" "${pytest_args[@]}"
