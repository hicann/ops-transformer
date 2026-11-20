/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if (__CCE_AICORE__ == 220)
#include "arch22/sparse_flash_mla_softmax_l1_norm_arch22.h"

using namespace AscendC;

namespace {
constexpr int32_t kTilingMagic = 0x53464D4C;

__aicore__ inline bool LoadTiling(GM_ADDR tiling, SmlaL1Norm::SmlaTilingPod& data)
{
    GlobalTensor<int32_t> words;
    words.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tiling));
    if (words.GetValue(0) != kTilingMagic) {
        return false;
    }
    data.b = words.GetValue(1);
    data.sq = words.GetValue(2);
    data.sk = words.GetValue(3);
    data.g = words.GetValue(4);
    data.d = words.GetValue(5);
    data.t1 = words.GetValue(6);
    data.t2 = words.GetValue(7);
    data.maxSeqlenK = words.GetValue(8);
    data.kLength = words.GetValue(9);
    data.cmpRatio = words.GetValue(10);
    data.totalRows = words.GetValue(11);
    data.numCores = words.GetValue(12);
    data.outRowLen = words.GetValue(13);
    data.wsSlotFloats = words.GetValue(14);
    data.maskMode = words.GetValue(15);
    data.isTnd = words.GetValue(16);
    data.isSparse = words.GetValue(17);
    data.dtype = words.GetValue(18);
    data.hasSequsedQ = words.GetValue(19);
    data.hasSequsedK = words.GetValue(20);
    data.hasTopkLength = words.GetValue(21);
    union {
        int32_t bits;
        float value;
    } scale;
    scale.bits = words.GetValue(22);
    data.softmaxScale = scale.value;
    data.packRows = words.GetValue(23);
    if (data.packRows <= 0) {
        data.packRows = 1;
    }
    data.gHalfFirst = (data.g + 1) / 2;
    data.initPerCoreNum = 0;
    data.initTotalNum = 0;
    const bool supportedD = data.d == SmlaL1Norm::kMinHeadDim || data.d == SmlaL1Norm::kMaxHeadDim;
    return data.numCores > 0 && data.g > 0 && supportedD && data.outRowLen >= 0;
}
} // namespace

extern "C" __global__ __aicore__ void sparse_flash_mla_softmax_l1_norm(
    GM_ADDR q, GM_ADDR k, GM_ADDR softmaxLse, GM_ADDR sparseIndices, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK,
    GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK, GM_ADDR topkLength, GM_ADDR metadata,
    GM_ADDR softmaxL1Norm, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (q == nullptr || k == nullptr || softmaxLse == nullptr || softmaxL1Norm == nullptr || workspace == nullptr ||
        tiling == nullptr) {
        return;
    }

    SmlaL1Norm::SmlaTilingPod tilingData;
    if (!LoadTiling(tiling, tilingData)) {
        return;
    }

    TPipe pipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (tilingData.dtype == 0) {
        SmlaL1Norm::SmlaL1NormKernel<half> op;
        op.Init(q, k, softmaxLse, sparseIndices, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, topkLength,
                metadata, softmaxL1Norm, userWorkspace, &pipe, &tilingData);
        op.Process();
    } else {
        SmlaL1Norm::SmlaL1NormKernel<bfloat16_t> op;
        op.Init(q, k, softmaxLse, sparseIndices, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, topkLength,
                metadata, softmaxL1Norm, userWorkspace, &pipe, &tilingData);
        op.Process();
    }
}
#endif
