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
 * \file quant_lightning_indexer_v2_common_arch35.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_COMMON_H
#define QUANT_LIGHTNING_INDEXER_V2_COMMON_H

#include "../../../lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_base_arch35.h"
#include "../../../lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_const_info_common.h"

using namespace AscendC;
namespace QLIV2Common {
using FP8E4M3 = fp8_e4m3fn_t;
using FP4E2M1 = fp4x2_e2m1_t;
using FP8E8M0 = fp8_e8m0_t;

constexpr uint32_t MX_SCALE_GROUP_SIZE = 32; // 每32个D维度元素一个scale
constexpr uint32_t FP8_TWO = 2;              // 2个FP8E8M0打包成1个BF16
constexpr uint32_t FP4_PACK_NUM = 2;         // 2个FP4E2M1打包成1个字节存储

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename QK_T, typename SCORE_T, typename OUT_T,
          const bool PAGE_ATTENTION = false, LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND,
          LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BBND, typename SCALE_T = float, typename WEIGHT_T = float,
          typename... Args>
struct QLIV2Type {
    static_assert((std::is_same_v<QK_T, float> &&
                   (std::is_same_v<SCORE_T, uint32_t> || std::is_same_v<SCORE_T, uint16_t>)) ||
                      (std::is_same_v<QK_T, bfloat16_t> && std::is_same_v<SCORE_T, uint16_t>) ||
                      (std::is_same_v<QK_T, int32_t> && std::is_same_v<SCORE_T, uint16_t>),
                  "Invalid combination of QK_T and SCORE_T");
    using rawQueryType = Q_T;
    using rawKeyType = K_T;

    static constexpr bool isMxFp8 = std::is_same_v<rawQueryType, FP8E4M3> && std::is_same_v<rawKeyType, FP8E4M3> &&
                                    std::is_same_v<SCALE_T, FP8E8M0>;
    static constexpr bool isMxFp4 = std::is_same_v<rawQueryType, FP4E2M1> && std::is_same_v<rawKeyType, FP4E2M1> &&
                                    std::is_same_v<SCALE_T, FP8E8M0>;
    static constexpr bool isMx = isMxFp8 || isMxFp4;

    using queryType = std::conditional_t<isMxFp4, uint8_t, rawQueryType>;
    using keyType = std::conditional_t<isMxFp4, uint8_t, rawKeyType>;
    using queryKeyType = QK_T;
    using scoreType = SCORE_T;
    using outputType = OUT_T;
    using scaleType = SCALE_T;
    using weightType = WEIGHT_T;

    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
    static constexpr bool isWeightFP16 = std::is_same_v<WEIGHT_T, half>;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start;
    uint32_t s2LoopEnd;
    uint32_t validS2Len;
    uint32_t qScaleLoop;
    uint32_t kScaleLoop;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    uint32_t curCuSeqlensQ;
    uint32_t curCuSeqlensK;
    uint32_t curSequsedQ;
    uint32_t curSequsedK;
    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorQScaleOffset; // MX场景qScale在Cube核使用的偏移
    uint64_t tensorKeyScaleOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;
    uint64_t outputIdxCoreOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
    bool isNeedLD = false;
    bool isOutputIdxOffsetValid = false;
    bool needTndPadding = false;
    uint32_t saveWorkSpaceIdx = 0;
};

struct ConstInfo : LIV2Common::ConstInfoCommon<LI_LAYOUT> {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t QLIV2_SYNC_MODE4 = 4;
    static constexpr uint16_t NEG_INF_BFLOAT = 0xFF80;

    // 8字节字段
    uint64_t sparseCount; // topK选取大小

    // 4字节字段
    uint32_t mBaseSizeMax = 1U;
    uint32_t cmpRatio = 1;
    uint32_t keyDequantScaleStride0 = 0;
    int32_t maxSeqlenQ = -1;
    uint32_t quantMode = 1;           // quant模式，默认为1
    uint32_t cmpResiduaKLenDims = 0U; // cmpResidualK的维度
};

using SplitCoreInfo = LIV2Common::LightningIndexerV2SplitCoreInfo;
using LdSplitCoreInfo = LIV2Common::LightningIndexerV2LdSplitCoreInfo;
using LIV2Common::Align;
using LIV2Common::CeilDiv;
using LIV2Common::Max;
using LIV2Common::Min;
} // namespace QLIV2Common

#endif // QUANT_LIGHTNING_INDEXER_V2_COMMON_H
