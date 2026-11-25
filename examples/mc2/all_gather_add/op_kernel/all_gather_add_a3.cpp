/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_add_a3.cpp
 * \brief arch22 (Atlas A2/A3) kernel entry for AllGatherAdd，AICPU 高层 Hccl 路径。
 *        注意：示例算子 kernel 源码需平铺放在 op_kernel/ 下，由 opFile.value 按文件名选择。
 */
#include "all_gather_add.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void all_gather_add(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM,
                                                     GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 设置kernel类型，AIC、AIV混合场景下，控制算子执行时仅启动AI Core上的Vector核（arch22 专用）
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    // 注册算子Tiling结构体
    REGISTER_TILING_DEFAULT(AllGatherAddTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    TPipe pipe;

    // A2/A3 走 AICPU 高层 Hccl
    AllGatherAdd<HcclServerType::HCCL_SERVER_TYPE_AICPU> allGatherAdd;
    allGatherAdd.Init(aGM, bGM, cGM, gatherGM, &tilingData, &pipe);
    allGatherAdd.Process();
}
