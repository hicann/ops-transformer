#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail
gmm_tools_dir=$(cd "$(dirname "$0")" && pwd)
gmm_build_dir=${1:?Usage: run_tests.sh BUILD_DIR PYTHON_SITE [PACKAGE]}
gmm_python_site=${2:?PYTHON_SITE is the parent of the installed package directory}
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh"
gmm_vendor="$gmm_build_dir/stage/packages/vendors/experimental_group_matmul"
test -d "$gmm_vendor" || { echo "Missing compiled package: $gmm_vendor" >&2; exit 1; }
export ASCEND_CUSTOM_OPP_PATH="$gmm_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$gmm_vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$gmm_python_site${PYTHONPATH:+:$PYTHONPATH}"
export CANN_OPS_TRANSFORMER_PACKAGE=${3:-cann_ops_transformer}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$gmm_build_dir/torch_extensions}
export ASCEND_WORK_PATH="$gmm_build_dir/runtime"
export MAX_JOBS=${MAX_JOBS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
mkdir -p "$ASCEND_WORK_PATH"
python3 -u "$gmm_tools_dir/../tests/test_group_matmul.py"
