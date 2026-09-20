/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2_common_arch35.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_COMMON_ARCH35_H
#define LIGHTNING_INDEXER_V2_COMMON_ARCH35_H

#include "common/lightning_indexer_v2_base_arch35.h"
#include "common/lightning_indexer_v2_const_info_common.h"

using namespace AscendC;
namespace LIV2Common {

// 与tiling的layout保持一致
enum class LI_V2_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename OUT_T, typename QK_T, typename SCORE_T,
          const bool PAGE_ATTENTION = false, LI_V2_LAYOUT LAYOUT_T = LI_V2_LAYOUT::BSND,
          LI_V2_LAYOUT K_LAYOUT_T = LI_V2_LAYOUT::PA_BBND, bool DT_W_FLAG = false, typename... Args>
struct LIV2Type {
    static constexpr bool weightsTypeFlag = DT_W_FLAG; // weight的dtype是否为FP32
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    using queryKeyType = QK_T;
    using scoreType = SCORE_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_V2_LAYOUT layout = LAYOUT_T;
    static constexpr LI_V2_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start = 0;
    uint32_t s2LoopEnd = 0;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    uint32_t curCuSeqlensQ = 0;
    uint32_t curCuSeqlensK = 0;
    uint32_t curSequsedQ = 0;
    uint32_t curSequsedK = 0;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;
    uint64_t outputIdxCoreOffset;
    bool isOutputIdxOffsetValid;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
    bool isNeedLD = false;
    bool needTndPadding = false;
    uint32_t saveWorkSpaceIdx = 0;
};

struct ConstInfo : ConstInfoCommon<LI_V2_LAYOUT> {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t LI_SYNC_MODE4 = 4;
    // float负无穷
    static constexpr uint32_t NEG_INF_FLOAT = 0xFF800000;

    // 8字节字段
    uint64_t topk; // topK选取大小
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    int64_t cmpRatio = 1;
    int64_t cmpResidualK = 0;

    // 4字节字段
    uint32_t mBaseSizeAlign = 1U;

    // 1字节字段
    bool batchSupperFlag = false;     // Qactual_seq长度是否为B+1
    bool isSparseCountOver2K = false; // sparseCount小于等于2048为false
    bool returnValueFlag = false;
};

using SplitCoreInfo = LightningIndexerV2SplitCoreInfo;
using LdSplitCoreInfo = LightningIndexerV2LdSplitCoreInfo;
} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_COMMON_ARCH35_H
