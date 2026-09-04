#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Batch-invariance test runner for AllGatherMatmulV2 (MXFP8). See README.md.
# Launches bi_test_driver.py at WS=2/4/8. Requires idle NPU cards and a build/vendor with the
# blockSize gate fix (issue #2778 / PR #6137), else all cells report ERROR (EZ0002).
#
# Usage:  bash run_bi_test.sh [WS_LIST] [DEVS]
#   bash run_bi_test.sh "2 4 8"            # default; devices auto 0..WS-1
#   bash run_bi_test.sh "2 4" 0,1,2,3
# Env:  BI_ORIENT=both|notrans|trans   BI_DATA_MODE=random|mx
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$SCRIPT_DIR/bi_test_driver.py"
WS_LIST="${1:-2 4 8}"
DEVS_OVERRIDE="${2:-}"
PORT=29640

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-300}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-256}"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"

rc=0
for WS in $WS_LIST; do
    if [ -n "$DEVS_OVERRIDE" ]; then DEVS="$DEVS_OVERRIDE"; else
        DEVS=$(seq -s, 0 $((WS - 1)))
    fi
    echo "########## WS=$WS  devices=$DEVS ##########"
    ASCEND_RT_VISIBLE_DEVICES="$DEVS" python3 -m torch.distributed.run \
        --nproc_per_node="$WS" --master_port="$PORT" "$DRIVER" || rc=1
    PORT=$((PORT + 1))
done
exit $rc
