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
 * \file masked_causal_conv1d_tiling_arch35.cpp
 * \brief MaskedCausalConv1d tiling implementation for Arch35 (ascend950)
 */

#include "masked_causal_conv1d_tiling_arch35.h"

namespace optiling {

constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;

constexpr uint64_t INPUT_X_INDEX = 0;
constexpr uint64_t INPUT_WEIGHT_INDEX = 1;
constexpr uint64_t INPUT_MASK_INDEX = 2;

constexpr int64_t H_MIN = 192 * 2;                          // min H = 384 (aligned with backward)
constexpr int64_t H_MAX = 192 * 128;                        // max H = 24576 (aligned with backward)
constexpr int64_t BS_MAX = 512 * 1024;                      // max B*S = 524288
constexpr uint64_t CONV_WINDOW_SIZE = 3;                    // W: causal conv1d kernel size (fixed)
constexpr uint64_t PREFIX_ROW_COUNT = CONV_WINDOW_SIZE - 1; // prefix rows in ioQueue slot (W-1)
constexpr uint64_t MASK_ALIGN_ELEMENTS = 32;                // sUb alignment granularity for mask buffers
constexpr uint64_t DOUBLE_BUFFER_NUM = 2;                   // ioQueue / weightQueue double buffer
constexpr uint64_t MASK_BUF_COUNT = 2;                      // maskRowBuf + maskCastBuf
constexpr uint64_t X_EXPECTED_NDIM = 3;                     // input x is 3-D [S,B,H]
constexpr uint32_t WORKSPACE_NUM = 1;                       // number of workspace buffers
constexpr uint64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024;      // 8 KB
constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

constexpr uint64_t TILING_KEY_BF16 = 10000UL;
constexpr uint64_t TILING_KEY_FP16 = 10001UL;

// ---- IsCapable ----
bool MaskedCausalConv1dTilingArch35::IsCapable()
{
    return true;
}

ge::graphStatus MaskedCausalConv1dTilingArch35::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// ---- GetPlatformInfo ----
ge::graphStatus MaskedCausalConv1dTilingArch35::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platform info is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    if (coreNum_ == 0UL) {
        OP_LOGE(context_->GetNodeName(), "coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0UL) {
        OP_LOGE(context_->GetNodeName(), "ubSize is 0");
        return ge::GRAPH_FAILED;
    }
    if (ubSize <= SYSTEM_RESERVED_UB_SIZE) {
        OP_LOGE(context_->GetNodeName(), "ubSize %lu too small", ubSize);
        return ge::GRAPH_FAILED;
    }
    ubSize_ = ubSize - SYSTEM_RESERVED_UB_SIZE;
    return ge::GRAPH_SUCCESS;
}

// ---- GetInputShapes ----
ge::graphStatus MaskedCausalConv1dTilingArch35::GetInputShapes()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_X_INDEX));
    xShape_ = context_->GetInputShape(INPUT_X_INDEX)->GetOriginShape();
    if (xShape_.GetDimNum() != X_EXPECTED_NDIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(xShape_.GetDimNum()) + "D", "The shape of x must be 3D");
        return ge::GRAPH_FAILED;
    }
    S_ = static_cast<uint64_t>(xShape_.GetDim(DIM_0));
    B_ = static_cast<uint64_t>(xShape_.GetDim(DIM_1));
    H_ = static_cast<uint64_t>(xShape_.GetDim(DIM_2));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_WEIGHT_INDEX));
    weightShape_ = context_->GetInputShape(INPUT_WEIGHT_INDEX)->GetOriginShape();

    auto maskShapePtr = context_->GetOptionalInputShape(INPUT_MASK_INDEX);
    if (maskShapePtr != nullptr) {
        maskShape_ = maskShapePtr->GetOriginShape();
        isMaskNone_ = 0;
    } else {
        isMaskNone_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

// ---- GetInputDtypes ----
ge::graphStatus MaskedCausalConv1dTilingArch35::GetInputDtypes()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_X_INDEX));
    xType_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();
    if (xType_ != ge::DataType::DT_FLOAT16 && xType_ != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                              ge::TypeUtils::DataTypeToSerialString(xType_).c_str(),
                                              "The dtype of x must be within the range [float16, bfloat16]");
        return ge::GRAPH_FAILED;
    }
    xDtypeSize_ = GetSizeByDataType(xType_);

    // weight dtype must match x dtype
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_WEIGHT_INDEX));
    ge::DataType weightType = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();
    if (weightType != xType_) {
        std::string incorrectDtypes =
            ge::TypeUtils::DataTypeToSerialString(xType_) + " and " + ge::TypeUtils::DataTypeToSerialString(weightType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and weight", incorrectDtypes.c_str(),
                                               "The dtypes of x and weight must be the same");
        return ge::GRAPH_FAILED;
    }

    // mask dtype must be DT_BOOL (only when mask is present)
    if (isMaskNone_ == 0) {
        auto maskDesc = context_->GetOptionalInputDesc(INPUT_MASK_INDEX);
        if (maskDesc != nullptr) {
            ge::DataType maskType = maskDesc->GetDataType();
            if (maskType != ge::DataType::DT_BOOL) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "mask",
                                                      ge::TypeUtils::DataTypeToSerialString(maskType).c_str(),
                                                      "The dtype of mask must be BOOL");
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ---- CheckInputParams ----
ge::graphStatus MaskedCausalConv1dTilingArch35::CheckInputParams()
{
    if (S_ == 0 || B_ == 0 || H_ == 0) {
        std::string incorrectShape =
            "[" + std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", incorrectShape.c_str(),
                                              "All dimensions of this parameter must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (static_cast<int64_t>(H_) < H_MIN || static_cast<int64_t>(H_) > H_MAX) {
        std::string incorrectShape =
            "[" + std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", incorrectShape.c_str(),
                                              "Shape [2] of this parameter must be within the range [384, 24576]");
        return ge::GRAPH_FAILED;
    }
    if (static_cast<int64_t>(B_) * static_cast<int64_t>(S_) > BS_MAX) {
        std::string incorrectShape =
            "[" + std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", incorrectShape.c_str(),
                                              "The following constraint must be met: shape [0] * shape [1] <= 524288");
        return ge::GRAPH_FAILED;
    }
    if (H_ % hReg_ != 0) {
        std::string incorrectShape =
            "[" + std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", incorrectShape.c_str(),
                                              "Shape [2] of this parameter must be exactly divided by hReg (64)");
        return ge::GRAPH_FAILED;
    }

    // weight shape: [K, H], K must equal CONV_WINDOW_SIZE, H must equal x's H
    if (weightShape_.GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "weight",
                                                 std::to_string(weightShape_.GetDimNum()) + "D",
                                                 "The shape of weight must be 2D");
        return ge::GRAPH_FAILED;
    }
    uint64_t wK = static_cast<uint64_t>(weightShape_.GetDim(DIM_0));
    if (wK != CONV_WINDOW_SIZE) {
        std::string incorrectShape = "[" + std::to_string(wK) + ", " + std::to_string(weightShape_.GetDim(DIM_1)) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "weight", incorrectShape.c_str(),
                                              "Shape [0] of this parameter must be equal to 3");
        return ge::GRAPH_FAILED;
    }
    uint64_t wH = static_cast<uint64_t>(weightShape_.GetDim(DIM_1));
    if (wH != H_) {
        std::string incorrectShapes = "[" + std::to_string(wK) + ", " + std::to_string(wH) + "] and [" +
                                      std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "weight and x", incorrectShapes.c_str(),
                                               "Shape [1] of weight must be equal to shape [2] of x");
        return ge::GRAPH_FAILED;
    }

    // mask shape: [B, S] when present
    if (isMaskNone_ == 0) {
        if (maskShape_.GetDimNum() != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "mask",
                                                     std::to_string(maskShape_.GetDimNum()) + "D",
                                                     "The shape of mask must be 2D");
            return ge::GRAPH_FAILED;
        }
        uint64_t mB = static_cast<uint64_t>(maskShape_.GetDim(DIM_0));
        uint64_t mS = static_cast<uint64_t>(maskShape_.GetDim(DIM_1));
        if (mB != B_) {
            std::string incorrectShapes = "[" + std::to_string(mB) + ", " + std::to_string(mS) + "] and [" +
                                          std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) +
                                          "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "mask and x", incorrectShapes.c_str(),
                                                   "Shape [0] of mask must be equal to shape [1] of x");
            return ge::GRAPH_FAILED;
        }
        if (mS != S_) {
            std::string incorrectShapes = "[" + std::to_string(mB) + ", " + std::to_string(mS) + "] and [" +
                                          std::to_string(S_) + ", " + std::to_string(B_) + ", " + std::to_string(H_) +
                                          "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "mask and x", incorrectShapes.c_str(),
                                                   "Shape [1] of mask must be equal to shape [0] of x");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ---- GetShapeAttrsInfo ----
ge::graphStatus MaskedCausalConv1dTilingArch35::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("MaskedCausalConv1d", "context is null"), return ge::GRAPH_FAILED);
    if (GetInputShapes() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (GetInputDtypes() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    // Compute hReg_ = 128 / xDtypeSize_ (bf16/fp16→64)
    hReg_ = 128 / xDtypeSize_;

    if (CheckInputParams() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

// ---- CalcLoopParams ----
ge::graphStatus MaskedCausalConv1dTilingArch35::CalcLoopParams(int64_t len, int64_t factor, int64_t &loopNum,
                                                               int64_t &tailFactor)
{
    if (factor == 0) {
        OP_LOGE(context_->GetNodeName(), "CalcLoopParams: factor is 0");
        return ge::GRAPH_FAILED;
    }
    loopNum = (len + factor - 1) / factor;
    tailFactor = len - (loopNum - 1) * factor;
    return ge::GRAPH_SUCCESS;
}

// ---- SearchBestCoreSplit ----
// Greedy H > B > S 3D inter-core search.
// H granularity: hReg_ = 128/dtypeSize elements per tile.
// 关键约束：hcc * bcc * scc <= coreNum_（硬件核数）
ge::graphStatus MaskedCausalConv1dTilingArch35::SearchBestCoreSplit()
{
    // H 可拆分的最大份数（每个 tile 处理 hReg_ 个 H 元素）
    uint64_t hMax = H_ / hReg_;

    // 关键修正 1：hcc 起始值不能超过硬件核数
    // 当 H 特别大时，hMax 可能远超 coreNum_，从 hMax 开始搜索毫无意义
    uint64_t hccStart = std::min(hMax, coreNum_);

    uint64_t bestTotal = 0;
    uint64_t bestH = 1;
    uint64_t bestB = 1;
    uint64_t bestS = 1;
    bool searchDone = false;

    for (uint64_t hcc = hccStart; hcc >= 1 && !searchDone; --hcc) {
        // 在 hcc 确定后，B×S 方向最多可分配的核数
        uint64_t maxBSByH = coreNum_ / hcc;
        if (maxBSByH == 0) {
            // hcc 太大，剩余核数不足分配 1 个 B×S 组合，跳过
            continue;
        }

        // B 方向搜索上限：不能超过 maxBSByH，也不能超过 B_
        uint64_t bccStart = std::min(B_, maxBSByH);

        for (uint64_t bcc = bccStart; bcc >= 1 && !searchDone; --bcc) {
            // 在 hcc, bcc 确定后，S 方向最多可分配的核数
            uint64_t maxSByHB = maxBSByH / bcc;
            if (maxSByHB == 0)
                continue;

            // scc 取三者最小值：
            //   1. S_：S 维度实际大小
            //   2. maxSByHB：核数约束下的上限
            //   3. coreNum_ / (hcc * bcc)：核数约束（冗余但安全）
            uint64_t scc = std::min({S_, maxSByHB, coreNum_ / (hcc * bcc)});
            if (scc == 0)
                scc = 1;

            uint64_t total = hcc * bcc * scc;

            // 关键检查：total 不能超过硬件核数（应该已经由上面约束保证，这里再确认）
            if (total > coreNum_)
                continue;

            if (total > bestTotal) {
                bestTotal = total;
                bestH = hcc;
                bestB = bcc;
                bestS = scc;
            }
            // 满核时提前退出
            if (bestTotal == coreNum_)
                searchDone = true;
        }
    }
    realCoreNum_ = bestTotal;
    hCoreCnt_ = bestH;
    bCoreCnt_ = bestB;
    sCoreCnt_ = bestS;

    // H-dim split (unit = hReg_)
    {
        uint64_t hTileTotal = H_ / hReg_;
        uint64_t hTilesPerMain = (hTileTotal + hCoreCnt_ - 1) / hCoreCnt_;
        uint64_t rem = hTileTotal % hCoreCnt_;
        hMainCnt_ = (rem > 0) ? rem : 0;
        hBlockFactor_ = hTilesPerMain * hReg_;
        hBlockTailFactor_ = (rem > 0 ? hTilesPerMain - 1 : hTilesPerMain) * hReg_;
        if (hMainCnt_ == 0) {
            hBlockTailFactor_ = hBlockFactor_;
        }
    }

    // B-dim split
    {
        uint64_t base = B_ / bCoreCnt_;
        uint64_t rem = B_ % bCoreCnt_;
        bMainCnt_ = rem;
        bBlockFactor_ = (rem > 0) ? (base + 1) : base;
        bBlockTailFactor_ = base;
        if (rem == 0)
            bBlockTailFactor_ = bBlockFactor_;
    }

    // S-dim split
    {
        uint64_t base = S_ / sCoreCnt_;
        uint64_t rem = S_ % sCoreCnt_;
        sMainCnt_ = rem;
        sBlockFactor_ = (rem > 0) ? (base + 1) : base;
        sBlockTailFactor_ = base;
        if (rem == 0)
            sBlockTailFactor_ = sBlockFactor_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dTilingArch35::CalcUbTiling()
{
    const int64_t d = static_cast<int64_t>(xDtypeSize_);

    // sUbAlign = ⌈sUb/MASK_ALIGN_ELEMENTS⌉×MASK_ALIGN_ELEMENTS (align up)
    auto alignUp = [](uint64_t x, uint64_t align) -> uint64_t { return ((x + align - 1) / align) * align; };

    // Total UB usage:
    //   ioQueue:     DOUBLE_BUFFER_NUM * d * b * (s+PREFIX_ROW_COUNT) * h
    //   weightQueue: DOUBLE_BUFFER_NUM * CONV_WINDOW_SIZE * d * h
    //   maskRowBuf:  DOUBLE_BUFFER_NUM * d * b * sUbAlign
    //   maskCastBuf: DOUBLE_BUFFER_NUM * d * max(b, h) * sUbAlign
    auto calcUsedUb = [&](uint64_t h, uint64_t b, uint64_t s) -> uint64_t {
        uint64_t sUbAlign = alignUp(s, MASK_ALIGN_ELEMENTS);
        uint64_t M = std::max(b, h);
        return DOUBLE_BUFFER_NUM * static_cast<uint64_t>(d) * b * (s + PREFIX_ROW_COUNT) * h +
               DOUBLE_BUFFER_NUM * CONV_WINDOW_SIZE * static_cast<uint64_t>(d) * h +
               DOUBLE_BUFFER_NUM * static_cast<uint64_t>(d) * b * sUbAlign +
               DOUBLE_BUFFER_NUM * static_cast<uint64_t>(d) * M * sUbAlign;
    };

    // Find max sUb: estimate first, then search downward to satisfy exact constraint
    auto findSUbMax = [&](uint64_t h, uint64_t b) -> int64_t {
        uint64_t M = std::max(b, h);
        int64_t prefixOverhead = static_cast<int64_t>(DOUBLE_BUFFER_NUM * PREFIX_ROW_COUNT) * static_cast<int64_t>(b);
        int64_t weightOverhead = static_cast<int64_t>(DOUBLE_BUFFER_NUM * CONV_WINDOW_SIZE);
        int64_t num = static_cast<int64_t>(ubSize_) - static_cast<int64_t>(h) * d * (prefixOverhead + weightOverhead);
        int64_t den = static_cast<int64_t>(DOUBLE_BUFFER_NUM) * d *
                      (static_cast<int64_t>(b) * static_cast<int64_t>(h) + static_cast<int64_t>(b + M));
        if (num <= 0 || den <= 0)
            return 0;
        int64_t sUbEst = num / den;
        // Search downward: sUbAlign staircase may cause estimate to exceed budget
        for (int64_t sUb = sUbEst; sUb >= 1; --sUb) {
            if (calcUsedUb(h, b, static_cast<uint64_t>(sUb)) <= ubSize_) {
                return sUb;
            }
        }
        return 0;
    };

    ubFactorH_ = static_cast<int64_t>(hReg_);
    uint64_t bUb = bBlockFactor_;
    uint64_t sLen = sBlockFactor_;

    // Step1: try full load BS with hUb=hReg_=128/dtypeSize
    if (calcUsedUb(static_cast<uint64_t>(ubFactorH_), bUb, sLen) <= ubSize_) {
        // Situation A: B+S fully loaded → Step2 expand hUb
        uint64_t sUb = sLen;

        // Cap at min(hBlockFactor_, hBlockTailFactor_) so ubFactorH fits all cores
        uint64_t hLenMin = (hMainCnt_ > 0) ? std::min(hBlockFactor_, hBlockTailFactor_) : hBlockFactor_;

        for (uint64_t k = 2; k * hReg_ <= hLenMin; ++k) {
            uint64_t hUbTry = k * hReg_;
            if (calcUsedUb(hUbTry, bUb, sLen) <= ubSize_) {
                ubFactorH_ = static_cast<int64_t>(hUbTry);
            } else {
                break;
            }
        }

        ubFactorB_ = static_cast<int64_t>(bUb);
        ubFactorS_ = static_cast<int64_t>(sUb);
    } else {
        // Situation B: can't fully load → Step3 reduce B first
        bool found = false;
        while (bUb > 1) {
            --bUb;
            if (calcUsedUb(static_cast<uint64_t>(ubFactorH_), bUb, sLen) <= ubSize_) {
                // B reduced, S fully loaded
                ubFactorB_ = static_cast<int64_t>(bUb);
                ubFactorS_ = static_cast<int64_t>(sLen);
                found = true;
                break;
            }
        }

        if (!found) {
            // bUb=1 still can't fully load S → cut S
            bUb = 1;
            int64_t sUbMax = findSUbMax(static_cast<uint64_t>(ubFactorH_), bUb);
            if (sUbMax < 1) {
                OP_LOGE(context_->GetNodeName(), "UB tiling failed: no valid sUb found with bUb=1");
                return ge::GRAPH_FAILED;
            }
            ubFactorB_ = 1;
            ubFactorS_ = static_cast<int64_t>(std::min(static_cast<uint64_t>(sUbMax), sLen));
        }
    }

    if (ubFactorB_ == 0 || ubFactorS_ == 0) {
        OP_LOGE(context_->GetNodeName(), "UB tiling failed: ubFactorB=%ld ubFactorS=%ld", ubFactorB_, ubFactorS_);
        return ge::GRAPH_FAILED;
    }

    // Loop params — main core
    if (CalcLoopParams(static_cast<int64_t>(hBlockFactor_), ubFactorH_, loopNumH_, ubTailFactorH_) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CalcLoopParams(static_cast<int64_t>(bBlockFactor_), ubFactorB_, loopNumB_, ubTailFactorB_) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CalcLoopParams(static_cast<int64_t>(sBlockFactor_), ubFactorS_, loopNumS_, ubTailFactorS_) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    // Loop params — tail core
    if (CalcLoopParams(static_cast<int64_t>(hBlockTailFactor_), ubFactorH_, tailBlockLoopNumH_,
                       tailBlockUbTailFactorH_) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CalcLoopParams(static_cast<int64_t>(bBlockTailFactor_), ubFactorB_, tailBlockLoopNumB_,
                       tailBlockUbTailFactorB_) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CalcLoopParams(static_cast<int64_t>(sBlockTailFactor_), ubFactorS_, tailBlockLoopNumS_,
                       tailBlockUbTailFactorS_) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    return ge::GRAPH_SUCCESS;
}

// ---- DoOpTiling ----
ge::graphStatus MaskedCausalConv1dTilingArch35::DoOpTiling()
{
    if (SearchBestCoreSplit() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CalcUbTiling() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

// ---- GetTilingKey ----
uint64_t MaskedCausalConv1dTilingArch35::GetTilingKey() const
{
    if (xType_ == ge::DataType::DT_BF16)
        return TILING_KEY_BF16;
    return TILING_KEY_FP16;
}

// ---- GetWorkspaceSize ----
ge::graphStatus MaskedCausalConv1dTilingArch35::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    auto workspaces = context_->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

// ---- PostTiling ----
ge::graphStatus MaskedCausalConv1dTilingArch35::PostTiling()
{
    context_->SetBlockDim(static_cast<uint32_t>(realCoreNum_));

    MaskedCausalConv1dTilingData td;
    td.S = static_cast<int64_t>(S_);
    td.B = static_cast<int64_t>(B_);
    td.H = static_cast<int64_t>(H_);

    td.hCoreCnt = static_cast<int64_t>(hCoreCnt_);
    td.hMainCnt = static_cast<int64_t>(hMainCnt_);
    td.hBlockFactor = static_cast<int64_t>(hBlockFactor_);
    td.hBlockTailFactor = static_cast<int64_t>(hBlockTailFactor_);

    td.bCoreCnt = static_cast<int64_t>(bCoreCnt_);
    td.bMainCnt = static_cast<int64_t>(bMainCnt_);
    td.bBlockFactor = static_cast<int64_t>(bBlockFactor_);
    td.bBlockTailFactor = static_cast<int64_t>(bBlockTailFactor_);

    td.sCoreCnt = static_cast<int64_t>(sCoreCnt_);
    td.sMainCnt = static_cast<int64_t>(sMainCnt_);
    td.sBlockFactor = static_cast<int64_t>(sBlockFactor_);
    td.sBlockTailFactor = static_cast<int64_t>(sBlockTailFactor_);

    td.ubFactorH = ubFactorH_;
    td.ubFactorB = ubFactorB_;
    td.ubFactorS = ubFactorS_;

    td.loopNumH = loopNumH_;
    td.ubTailFactorH = ubTailFactorH_;
    td.loopNumB = loopNumB_;
    td.ubTailFactorB = ubTailFactorB_;
    td.loopNumS = loopNumS_;
    td.ubTailFactorS = ubTailFactorS_;

    td.tailBlockLoopNumH = tailBlockLoopNumH_;
    td.tailBlockUbTailFactorH = tailBlockUbTailFactorH_;
    td.tailBlockLoopNumB = tailBlockLoopNumB_;
    td.tailBlockUbTailFactorB = tailBlockUbTailFactorB_;
    td.tailBlockLoopNumS = tailBlockLoopNumS_;
    td.tailBlockUbTailFactorS = tailBlockUbTailFactorS_;

    td.realCoreNum = static_cast<int64_t>(realCoreNum_);
    td.isMaskNone = isMaskNone_;

    auto tilingDataSize = sizeof(MaskedCausalConv1dTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&td), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

// ---- DumpTilingInfo ----
void MaskedCausalConv1dTilingArch35::DumpTilingInfo()
{
    std::ostringstream info;
    info << "S=" << S_ << " B=" << B_ << " H=" << H_ << "\n";
    info << "hCoreCnt=" << hCoreCnt_ << " hMainCnt=" << hMainCnt_ << " hBlockFactor=" << hBlockFactor_
         << " hBlockTailFactor=" << hBlockTailFactor_ << "\n";
    info << "bCoreCnt=" << bCoreCnt_ << " bMainCnt=" << bMainCnt_ << " bBlockFactor=" << bBlockFactor_
         << " bBlockTailFactor=" << bBlockTailFactor_ << "\n";
    info << "sCoreCnt=" << sCoreCnt_ << " sMainCnt=" << sMainCnt_ << " sBlockFactor=" << sBlockFactor_
         << " sBlockTailFactor=" << sBlockTailFactor_ << "\n";
    info << "ubFactorH=" << ubFactorH_ << " ubFactorB=" << ubFactorB_ << " ubFactorS=" << ubFactorS_
         << " sUbAlign=" << (((ubFactorS_ + MASK_ALIGN_ELEMENTS - 1) / MASK_ALIGN_ELEMENTS) * MASK_ALIGN_ELEMENTS)
         << "\n";
    info << "loopNumH=" << loopNumH_ << " ubTailFactorH=" << ubTailFactorH_ << "\n";
    info << "loopNumB=" << loopNumB_ << " ubTailFactorB=" << ubTailFactorB_ << "\n";
    info << "loopNumS=" << loopNumS_ << " ubTailFactorS=" << ubTailFactorS_ << "\n";
    info << "realCoreNum=" << realCoreNum_ << "\n";
    info << "isMaskNone=" << isMaskNone_ << "\n";
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

} // namespace optiling
