#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# matmul_reduce_scatter_v2 BI 测试启动器。
# 详见同目录 README.md。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="$SCRIPT_DIR/bi_test_driver.py"

# 默认环境变量。已设置则不覆盖。
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-256}"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"

# 用户可通过 BI_WS_LIST="2 4 8" 自定义。
BI_WS_LIST="${BI_WS_LIST:-2 4 8}"

EXIT_CODE=0
FIRST=1
for WS in $BI_WS_LIST; do
    case $WS in
        2) DEVS=0,1 ;;
        4) DEVS=0,1,2,3 ;;
        8) DEVS=0,1,2,3,4,5,6,7 ;;
        *) DEVS=$(seq -s, 0 $((WS-1))) ;;
    esac
    PORT=$((32700 + WS))
    if [ $FIRST -eq 0 ]; then
        # 给 HCCL/NPU 一点时间清理上一轮 worker 的 device 资源
        # 可通过 BI_SLEEP_BETWEEN_WS 环境变量调整（默认 15s）
        sleep "${BI_SLEEP_BETWEEN_WS:-15}"
    fi
    FIRST=0
    echo ""
    echo "##### Running BI test WS=$WS DEVS=$DEVS #####"
    ASCEND_RT_VISIBLE_DEVICES=$DEVS \
        python3 -m torch.distributed.run \
            --nproc-per-node=$WS \
            --master-port=$PORT \
            "$DRIVER" || EXIT_CODE=$?
done

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "BI test FAILED — see above output. Exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
