/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_sinkhorn_backward_tiling.cpp
 * \brief mhc_sinkhorn_backward_tiling
 */

#include <vector>
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "mhc_sinkhorn_base_tiling.h"

using namespace AscendC;
namespace optiling {

constexpr int64_t BUFFER_NUM = 3;
constexpr int64_t DOUBLE_SIZE = 2;
constexpr int64_t VL_SIZE = 256;
constexpr int64_t MASK_BUFFER = 64;
constexpr int64_t MAX_BUFFER_BACKWARD = 256 * BUFFER_NUM;
constexpr int64_t MAX_BUFFER_FORWARD = 256 * BUFFER_NUM;


MhcSinkhornSplitCoreInfo MhcSinkhornBaseTiling::BaseSplitCores(int64_t tSize, int64_t nSize, int64_t xDtypeSize,
                                                               int64_t outFlag)
{
    MhcSinkhornSplitCoreInfo forwardInfo;
    MhcSinkhornSplitCoreInfo backwardInfo;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    int64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    BackwardSplitCore(ubSize, totalCoreNum, tSize, nSize, xDtypeSize, backwardInfo);
    ForwardSplitCore(ubSize, totalCoreNum, tSize, nSize, xDtypeSize, outFlag, forwardInfo);
    return forwardInfo.tUbFactor <= backwardInfo.tUbFactor ? forwardInfo : backwardInfo;
}

int64_t MhcSinkhornBaseTiling::BackwardCalOccupySize(int64_t ubFactor, int64_t nSize, int64_t xDtypeSize)
{
    // UB分配：[(2 t n n + 2 n t align_n + 2 t align_n) * 4B] + 256 + 64 < 可用UB大小/2
    // gradInQueue: t n n
    // gradOutQueue: t n n
    // normInQueue: 2 n t align_n
    // sumInQueue: 2 t align_n
    // maxBuff: 256B
    // maskBuff: 64B
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t dataBlockNum = Ops::Base::FloorDiv(ubBlock, xDtypeSize);
    int64_t nAlignSize = Ops::Base::CeilAlign(nSize, dataBlockNum);

    int64_t inQueSize = Ops::Base::CeilAlign(ubFactor * nSize * nSize, dataBlockNum);
    int64_t outQueSize = Ops::Base::CeilAlign(ubFactor * nSize * nSize, dataBlockNum);
    int64_t normOutSize = Ops::Base::CeilAlign(2 * ubFactor * nSize * nAlignSize, dataBlockNum);
    int64_t sumOutSize = Ops::Base::CeilAlign(2 * ubFactor * nAlignSize, dataBlockNum);
    int64_t occupySize = (inQueSize + outQueSize + normOutSize + sumOutSize) * xDtypeSize;
    return occupySize;
}

void MhcSinkhornBaseTiling::BackwardSplitCore(uint64_t ubSize, int64_t totalCoreNum, int64_t tSize, int64_t nSize,
                                              int64_t xDtypeSize, MhcSinkhornSplitCoreInfo &splitInfo)
{
    splitInfo.tNormCore = Ops::Base::CeilDiv(tSize, totalCoreNum);
    splitInfo.usedCoreNum = Ops::Base::CeilDiv(tSize, splitInfo.tNormCore);
    splitInfo.tTailCore = tSize - splitInfo.tNormCore * (splitInfo.usedCoreNum - 1);
    int64_t availableUbSize = ubSize - MASK_BUFFER - MAX_BUFFER_BACKWARD;
    // 所有buffer开启double buffer
    int64_t singleUbSize = availableUbSize / DOUBLE_SIZE;
    int64_t occupySize = BackwardCalOccupySize(splitInfo.tNormCore, nSize, xDtypeSize);
    if (occupySize <= singleUbSize) {
        splitInfo.tUbFactor = splitInfo.tNormCore;
    } else {
        int64_t onePiceSize = BackwardCalOccupySize(1, nSize, xDtypeSize);
        splitInfo.tUbFactor = singleUbSize / onePiceSize;
        // kernel约束: tUbFactor向下跟8对齐, 8=nAlignSize
        int64_t tVlFactor = Ops::Base::FloorDiv(VL_SIZE, (8 * xDtypeSize));
        splitInfo.tUbFactor = Ops::Base::FloorAlign(splitInfo.tUbFactor, tVlFactor);
    }
    splitInfo.tNormCoreLoop =
        splitInfo.tUbFactor == 0 ? 1 : Ops::Base::CeilDiv(splitInfo.tNormCore, splitInfo.tUbFactor);
    splitInfo.tTailCoreLoop =
        splitInfo.tUbFactor == 0 ? 1 : Ops::Base::CeilDiv(splitInfo.tTailCore, splitInfo.tUbFactor);

    splitInfo.tUbFactorTail = splitInfo.tNormCore - (splitInfo.tNormCoreLoop - 1) * splitInfo.tUbFactor;
    splitInfo.tUbTailTail = splitInfo.tTailCore - (splitInfo.tTailCoreLoop - 1) * splitInfo.tUbFactor;
}

int64_t MhcSinkhornBaseTiling::ForwardCalOccupySize(int64_t ubFactor, int64_t nSize, int64_t xDtypeSize,
                                                    int64_t outFlag)
{
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t dataBlockNum = ubBlock / xDtypeSize;

    int64_t nnSize = nSize * nSize;
    int64_t nAlign = Ops::Base::CeilAlign(nSize, dataBlockNum);

    int64_t inQueSize = Ops::Base::CeilAlign(ubFactor * nnSize, dataBlockNum);
    int64_t outQueSize = Ops::Base::CeilAlign(ubFactor * nnSize, dataBlockNum);
    int64_t normOutSize = Ops::Base::CeilAlign(nSize * ubFactor * nAlign, dataBlockNum);
    int64_t sumColSize = Ops::Base::CeilAlign(ubFactor * nAlign, dataBlockNum);
    int64_t sumRowSize = Ops::Base::CeilAlign(ubFactor * nAlign, dataBlockNum);

    int64_t occupySize = (inQueSize + outQueSize + normOutSize + sumColSize) * xDtypeSize;
    if (outFlag == 1) {
        occupySize = (inQueSize + outQueSize + normOutSize + sumColSize + sumRowSize) * xDtypeSize;
    }
    return occupySize;
}

void MhcSinkhornBaseTiling::ForwardSplitCore(uint64_t ubSize, int64_t totalCoreNum, int64_t tSize, int64_t nSize,
                                             int64_t xDtypeSize, int64_t outFlag, MhcSinkhornSplitCoreInfo &splitInfo)
{
    splitInfo.tNormCore = Ops::Base::CeilDiv(tSize, totalCoreNum);
    splitInfo.usedCoreNum = Ops::Base::CeilDiv(tSize, splitInfo.tNormCore);
    splitInfo.tTailCore = tSize - splitInfo.tNormCore * (splitInfo.usedCoreNum - 1);
    auto ubSizeUsed = ubSize - MASK_BUFFER - MAX_BUFFER_FORWARD;
    auto availableUbSize = ubSizeUsed / DOUBLE_SIZE;

    splitInfo.tUbFactor = splitInfo.tNormCore;
    int64_t occupySize = ForwardCalOccupySize(splitInfo.tUbFactor, nSize, xDtypeSize, outFlag);
    if (occupySize > availableUbSize) {
        auto onePiceSize = ForwardCalOccupySize(1, nSize, xDtypeSize, outFlag);
        splitInfo.tUbFactor = availableUbSize / onePiceSize;
        splitInfo.tUbFactor = Ops::Base::FloorAlign(splitInfo.tUbFactor, static_cast<int64_t>(8));
    }
    splitInfo.tUbFactor = splitInfo.tUbFactor == 0 ? 8 : splitInfo.tUbFactor;

    splitInfo.tNormCoreLoop = Ops::Base::CeilDiv(splitInfo.tNormCore, splitInfo.tUbFactor);
    splitInfo.tTailCoreLoop = Ops::Base::CeilDiv(splitInfo.tTailCore, splitInfo.tUbFactor);

    splitInfo.tUbFactorTail = splitInfo.tNormCore - (splitInfo.tNormCoreLoop - 1) * splitInfo.tUbFactor;
    splitInfo.tUbTailTail = splitInfo.tTailCore - (splitInfo.tTailCoreLoop - 1) * splitInfo.tUbFactor;
}

} // namespace optiling
