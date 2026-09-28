#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail
lig_tools_dir=$(cd "$(dirname "$0")" && pwd)
lig_build=${1:?Usage: run_golden.sh GRAD_BUILD INDEXER_BUILD PYTHON_SITE [PACKAGE]}
li_build=${2:?Specify the migrated forward indexer build directory}
lig_python_site=${3:?Specify the wheel installation directory}
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh"
export ASCEND_CUSTOM_OPP_PATH="$lig_build/stage/packages/vendors/experimental_lightning_indexer_grad_kl_loss:$li_build/stage/packages/vendors/experimental_lightning_indexer"
export LD_LIBRARY_PATH="$lig_build/stage/packages/vendors/experimental_lightning_indexer_grad_kl_loss/op_api/lib:$li_build/stage/packages/vendors/experimental_lightning_indexer/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$lig_python_site:$PYTHONPATH"
export CANN_OPS_TRANSFORMER_PACKAGE=${4:-cann_ops_transformer}
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$lig_build/torch_extensions}"
export ASCEND_WORK_PATH="$lig_build/runtime"
export MAX_JOBS=4
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$ASCEND_WORK_PATH"
exec python3 -u "$lig_tools_dir/../tests/lightning_indexer_klloss_golden.py"
