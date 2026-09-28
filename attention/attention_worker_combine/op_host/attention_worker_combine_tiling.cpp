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
 * \file attention_worker_combine_tiling.cpp
 * \brief
 */
#include "attention_worker_combine_tiling_base.h"

namespace optiling {

class AttentionWorkerCombineTiling : public AttentionWorkerCombineTilingBase {
public:
    explicit AttentionWorkerCombineTiling(gert::TilingContext *context)
        : AttentionWorkerCombineTilingBase(context)
    {}
    ~AttentionWorkerCombineTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus DoGetPlatformInfo() override;
    ge::graphStatus DoGetShapeAttrsInfo() override;
    ge::graphStatus CalcOpTiling() override;
    ge::graphStatus CalcTilingKey() override;
    void DoPostTiling() override;

private:
    void DoOpTilingFullK(int64_t batchSize, int64_t bsCoreNum, int64_t hAlign, int64_t kIn);
    void SelectBestHCore(int64_t hAlign, int64_t hInSplitK, int64_t &bsCoreNum, int64_t &lastCoreNum,
                         int64_t &lastBestHCore);

    AttentionWorkerCombineTilingData tilingData_;
};

constexpr int64_t EXPERT_SCALES_INDEX = 1;
constexpr int64_t SHAPE_IDX_BS = 0;
constexpr int64_t SHAPE_IDX_K = 1;
constexpr int64_t HIDDEN_SIZE_INDEX = 0;
constexpr int64_t TOKEN_DTYPE_INDEX = 1;
constexpr int64_t NEED_SCHEDULE_INDEX = 2;
constexpr int64_t DIM_SIZE = 2;
constexpr int64_t H_ALIGN_SIZE = 512;
constexpr int64_t K_UPPER_BOUND = 64;

constexpr int64_t TILING_KEY_DIVIDE_BS_FP16 = 10000;
constexpr int64_t TILING_KEY_DIVIDE_BS_BF16 = 10001;
constexpr int64_t TILING_KEY_DIVIDE_H_FP16 = 10010;
constexpr int64_t TILING_KEY_DIVIDE_H_BF16 = 10011;
constexpr int64_t TILING_KEY_DIVIDE_K_FP16 = 10020;
constexpr int64_t TILING_KEY_DIVIDE_K_BF16 = 10021;

constexpr int64_t B32_DTYPE_BYTES = 4;
constexpr int64_t B16_DTYPE_BYTES = 2;

static std::tuple<int64_t, int64_t> GetShapeTuple(const gert::TilingContext *context, const int64_t index = 0)
{
    const gert::StorageShape *shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context->GetNodeName(), "Shape is nullptr."),
                return std::make_tuple(0, 0));
    OP_CHECK_IF(shapePtr->GetStorageShape().GetDimNum() != DIM_SIZE,
                OP_LOGE(context->GetNodeName(), "Shape must be (BS, K)."), return std::make_tuple(0, 0));
    return std::make_tuple(shapePtr->GetStorageShape().GetDim(SHAPE_IDX_BS),
                           shapePtr->GetStorageShape().GetDim(SHAPE_IDX_K));
}

ge::graphStatus AttentionWorkerCombineTiling::DoGetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const AttentionWorkerCombineCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr."),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize_ = ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionWorkerCombineTiling::DoGetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."),
                return ge::GRAPH_FAILED);

    auto expertScalesShapeTuple = GetShapeTuple(context_, EXPERT_SCALES_INDEX);
    int64_t batchSize = std::get<SHAPE_IDX_BS>(expertScalesShapeTuple);
    tilingData_.set_BS(batchSize);
    int64_t k = std::get<SHAPE_IDX_K>(expertScalesShapeTuple);
    tilingData_.set_K(k);

    OP_CHECK_IF(k > K_UPPER_BOUND, OP_LOGE(context_->GetNodeName(), "k should <= 64."), return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto hiddenSize = attrs->GetInt(HIDDEN_SIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hiddenSize);
    tilingData_.set_H(*hiddenSize);
    auto tokenDtype = attrs->GetInt(TOKEN_DTYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tokenDtype);
    tokenDtype_ = *tokenDtype;
    OP_CHECK_IF(tokenDtype_ < 0 || tokenDtype_ > 1,
                OP_LOGE(context_->GetNodeName(), "MXFP tokens are only supported on Ascend950."),
                return ge::GRAPH_FAILED);
    auto needSchedule = attrs->GetInt(NEED_SCHEDULE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, needSchedule);
    tilingData_.set_needSchedule(*needSchedule);
    return ge::GRAPH_SUCCESS;
}

void AttentionWorkerCombineTiling::DoOpTilingFullK(int64_t batchSize, int64_t bsCoreNum, int64_t hAlign, int64_t kIn)
{
    int64_t bsCoreFactor = 0;
    int64_t mainCoreNum = 0;
    int64_t tailCoreNum = 0;
    int64_t tailCoreBsLoopNum = 0;

    tilingData_.set_usedCoreNum(bsCoreNum);
    tilingData_.set_BsSplitFactor(1);
    tilingData_.set_BsSplitCoreNum(bsCoreNum);

    bsCoreFactor = CeilDiv(batchSize, bsCoreNum);
    tilingData_.set_mainCoreBsLoopNum(bsCoreFactor);
    if (bsCoreFactor == 0) {
        bsCoreFactor = 1;
    }
    mainCoreNum = batchSize / bsCoreFactor;
    tailCoreNum = bsCoreNum - mainCoreNum;
    if (tailCoreNum == 0) {
        tilingData_.set_tailCoreBsLoopNum(bsCoreFactor);
    } else {
        tailCoreBsLoopNum = (batchSize - mainCoreNum * bsCoreFactor) / tailCoreNum;
        tilingData_.set_tailCoreBsLoopNum(tailCoreBsLoopNum);
    }

    tilingData_.set_HSplitFactor(hAlign);
    tilingData_.set_HSplitCoreNum(1);
    tilingData_.set_mainCoreHLoopNum(1);
    tilingData_.set_tailCoreHLoopNum(1);

    tilingData_.set_KSplitFactor(kIn);
    tilingData_.set_KSplitTailFactor(kIn);
    tilingData_.set_KSplitLoopNum(1);

    tilingKey_ = TILING_KEY_DIVIDE_BS_FP16;
    context_->SetBlockDim(bsCoreNum);
}

void AttentionWorkerCombineTiling::SelectBestHCore(int64_t hAlign, int64_t hInSplitK, int64_t &bsCoreNum,
                                                   int64_t &lastCoreNum, int64_t &lastBestHCore)
{
    int64_t bsMaxCore = 0;
    int64_t batchSize = tilingData_.get_BS();
    int64_t maxCoreNum = static_cast<int64_t>(coreNum_ * 0.8);

    int64_t hOut = CeilDiv(hAlign, hInSplitK);
    int64_t hMaxCore = std::min(hOut, maxCoreNum);
    for (int x = 2; x < hMaxCore + 1; x++) {
        bsMaxCore = (static_cast<int64_t>(coreNum_) / static_cast<int64_t>(x));
        bsCoreNum = CeilDiv(batchSize, CeilDiv(batchSize, bsMaxCore));
        if (x * bsCoreNum > lastCoreNum) {
            lastCoreNum = x * bsCoreNum;
            lastBestHCore = x;
        }
        if (x * bsCoreNum >= maxCoreNum) {
            break;
        }
    }
}

ge::graphStatus AttentionWorkerCombineTiling::CalcOpTiling()
{
    int64_t k = tilingData_.get_K();
    int64_t hiddenSize = tilingData_.get_H();
    int64_t batchSize = tilingData_.get_BS();
    int64_t needSchedule = tilingData_.get_needSchedule();

    int64_t maxCoreNum = static_cast<int64_t>(coreNum_ * 0.8);
    int64_t kAddOne = k + 1;
    int64_t hInFullK = AlignDown(
        static_cast<int64_t>(ubSize_) / (2 * kAddOne * B16_DTYPE_BYTES + 2 * B16_DTYPE_BYTES + 2 * B32_DTYPE_BYTES),
        H_ALIGN_SIZE);
    int64_t bsCoreFactor = CeilDiv(batchSize, static_cast<int64_t>(coreNum_));
    int64_t bsCoreNum = CeilDiv(batchSize, bsCoreFactor);
    int64_t hInSplitK =
        AlignDown(static_cast<int64_t>(ubSize_) / (2 * B16_DTYPE_BYTES + 2 * B16_DTYPE_BYTES + 2 * B32_DTYPE_BYTES),
                  H_ALIGN_SIZE);
    int64_t kIn = 1;
    int64_t hAlign = AlignUp(hiddenSize, H_ALIGN_SIZE);
    int64_t lastCoreNum = 1;
    int64_t lastBestHCore = 1;
    int64_t blockDim = bsCoreNum;
    int64_t mainCoreNum = 0;
    int64_t tailCoreNum = 0;
    int64_t hTotalBlockNum = 0;
    int64_t tailCoreBsLoopNum = 1;
    int64_t hSplitFactor = hInSplitK;
    int64_t hSplitCoreNum = 1;
    int64_t hSplitTailFactor = 0;
    int64_t mainCoreHLoopNum = 0;
    int64_t tailCoreHLoopNum = 1;

    if (hAlign <= hInFullK) {
        DoOpTilingFullK(batchSize, bsCoreNum, hAlign, k);
        return ge::GRAPH_SUCCESS;
    }

    if (bsCoreNum <= maxCoreNum) {
        if (hAlign > hInSplitK) {
            if (needSchedule) {
                hSplitFactor = hInSplitK;
                tailCoreHLoopNum = CeilDiv(hAlign, hInSplitK);
            } else {
                SelectBestHCore(hAlign, hInSplitK, bsCoreNum, lastCoreNum, lastBestHCore);
                blockDim = lastCoreNum;
                hSplitFactor = hInSplitK;
                hSplitTailFactor = hInSplitK == 0 ? hInSplitK : hAlign % hInSplitK;

                hTotalBlockNum = CeilDiv(hAlign, hSplitFactor);
                mainCoreHLoopNum = CeilDiv(hTotalBlockNum, lastBestHCore);
                hSplitCoreNum = CeilDiv(hTotalBlockNum, mainCoreHLoopNum);
                tailCoreHLoopNum = mainCoreHLoopNum == 0 ? mainCoreHLoopNum : hTotalBlockNum % mainCoreHLoopNum;
            }
            tilingKey_ = TILING_KEY_DIVIDE_H_FP16;
        } else {
            hSplitFactor = hAlign;
            if (hAlign == 0) {
                hAlign = 1;
            }
            kIn = std::min(k, hInSplitK / hAlign);
            tilingKey_ = TILING_KEY_DIVIDE_K_FP16;
        }
    } else {
        hSplitCoreNum = 1;
        if (hAlign > hInSplitK) {
            hSplitFactor = hInSplitK;
            tailCoreHLoopNum = CeilDiv(hAlign, hInSplitK);
            tilingKey_ = TILING_KEY_DIVIDE_H_FP16;
        } else {
            hSplitFactor = hAlign;
            if (hAlign == 0) {
                hAlign = 1;
            }
            kIn = std::min(k, hInSplitK / hAlign);
            tilingKey_ = TILING_KEY_DIVIDE_K_FP16;
        }
    }
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_BsSplitFactor(1);
    tilingData_.set_BsSplitCoreNum(bsCoreNum);

    if (hSplitTailFactor == 0) {
        hSplitTailFactor = hInSplitK;
    }
    tilingData_.set_HSplitFactor(hSplitFactor);
    tilingData_.set_HSplitTailFactor(hAlign % hSplitFactor);
    tilingData_.set_HSplitCoreNum(hSplitCoreNum);

    tilingData_.set_KSplitFactor(kIn);
    if (kIn == 0) {
        kIn = 1;
    }
    tilingData_.set_KSplitTailFactor(k % kIn);
    tilingData_.set_KSplitLoopNum(k / kIn);

    bsCoreFactor = CeilDiv(batchSize, bsCoreNum);
    tilingData_.set_mainCoreBsLoopNum(bsCoreFactor);
    if (bsCoreFactor == 0) {
        bsCoreFactor = 1;
    }
    mainCoreNum = batchSize / bsCoreFactor;
    tailCoreNum = bsCoreNum - mainCoreNum;
    if (tailCoreNum == 0) {
        tilingData_.set_tailCoreBsLoopNum(bsCoreFactor);
    } else {
        tailCoreBsLoopNum = (batchSize - mainCoreNum * bsCoreFactor) / tailCoreNum;
        tilingData_.set_tailCoreBsLoopNum(tailCoreBsLoopNum);
    }

    if (tailCoreHLoopNum == 0) {
        tailCoreHLoopNum = mainCoreHLoopNum;
    }
    tilingData_.set_mainCoreHLoopNum(mainCoreHLoopNum);
    tilingData_.set_tailCoreHLoopNum(tailCoreHLoopNum);

    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionWorkerCombineTiling::CalcTilingKey()
{
    if (tokenDtype_ == 1) {
        tilingKey_ += 1;
    }
    return ge::GRAPH_SUCCESS;
}

void AttentionWorkerCombineTiling::DoPostTiling()
{
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

REGISTER_OPS_TILING_TEMPLATE(AttentionWorkerCombine, AttentionWorkerCombineTiling, 10000);

} // namespace optiling
