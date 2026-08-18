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
 * \file inplace_partial_rotary_mul_grad_regbase_tiling_a_and_b.cpp
 * \brief A (NO_BROADCAST) and B (BROADCAST_BSN) template tiling for InplacePartialRotaryMulGrad
 */
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"
namespace optiling {
constexpr uint64_t TILING_KEY_A = 205;
constexpr uint64_t TILING_KEY_B = 206;
constexpr uint64_t INPLACE_PARTIAL_ROTARY_MUL_GRAD_A_AND_B_TILING_PRIORITY = 40000;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t UB_COS_SIN_FACTOR = 2;

class InplacePartialRotaryMulGradRegbaseTilingAAndB : public InplacePartialRotaryMulGradRegbaseTiling {
public:
    explicit InplacePartialRotaryMulGradRegbaseTilingAAndB(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegbaseTiling(context)
    {}

protected:
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
            (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST ||
             layout_ == InplacePartialRotaryMulGradLayout::BROADCAST_BSN)) {
            return true;
        }
        return false;
    }

private:
    ge::graphStatus MergeDim();
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_ = 0;
    int64_t blockFactorB_ = 0;
    int64_t ubFactorB_ = 0;
    InplacePartialRotaryMulGradRegbaseTilingData tilingDataAAndB_;
};

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::DoOpTiling()
{
    if (isNoOp_) {
        usedCoreNum_ = 1;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(MergeDim() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "MergeDim failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SplitCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SplitCore failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "ComputeUbFactor failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::MergeDim()
{
    // Flatten B×N×S into single B dimension
    b_ = b_ * n_ * s_;
    n_ = 1;
    s_ = 1;
    if (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST) {
        cosb_ = b_;
    } else {
        cosb_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::SplitCore()
{
    int64_t coreNum = static_cast<int64_t>(aicoreParams_.numBlocks);
    blockFactorB_ = Ops::Base::CeilDiv(b_, coreNum);
    blockNumB_ = Ops::Base::CeilDiv(b_, blockFactorB_);
    usedCoreNum_ = blockNumB_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::ComputeUbFactor()
{
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);

    int64_t dSizeDy = Ops::Base::CeilAlign(sliceLength_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    int64_t dSizeCos = Ops::Base::CeilAlign(sliceLength_ * cosTypeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    int64_t effectiveCosOverhead = std::max(int64_t(1), Ops::Base::CeilDiv(dSizeCos, dSizeDy));
    int64_t baseBlockInUb =
        Ops::Base::FloorAlign(static_cast<int64_t>(aicoreParams_.ubSize / UB_FACTOR), blockSize_) / dSizeDy;
    if (baseBlockInUb < effectiveCosOverhead + 1) {
        OP_LOGE(context_->GetNodeName(), "ubSize can't load slice, sliceLength=%ld, dSizeDy=%ld, dSizeCos=%ld.",
                sliceLength_, dSizeDy, dSizeCos);
        return ge::GRAPH_FAILED;
    }

    if (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST) {
        // A: cos varies with B → UB = 4*ubFactorB*dSizeDy + 4*ubFactorB*dSizeCos
        //   = 4*ubFactorB*(dSizeDy + dSizeCos) = 4*ubFactorB*dSizeDy*(1 + overhead)
        // ubFactorB <= baseBlockInUb / (1 + effectiveCosOverhead)
        ubFactorB_ = std::min(blockFactorB_, Ops::Base::FloorDiv(baseBlockInUb, 1 + effectiveCosOverhead));
    } else {
        // B: cos broadcast → UB = 4*ubFactorB*dSizeDy + 2*dSizeCos
        //   dy+dx: 2 × double-buffer = 4*ubFactorB*dSizeDy
        //   cos+sin: 2 × single-buffer = 2*dSizeCos
        int64_t availableForDy = static_cast<int64_t>(aicoreParams_.ubSize) - UB_COS_SIN_FACTOR * dSizeCos;
        if (availableForDy <= 0) {
            OP_LOGE(context_->GetNodeName(), "ubSize can't fit cos/sin, dSizeCos=%ld.", dSizeCos);
            return ge::GRAPH_FAILED;
        }
        int64_t numOfDAvailable = Ops::Base::FloorDiv(availableForDy, UB_FACTOR * dSizeDy);
        OP_CHECK_IF(
            numOfDAvailable < 1,
            OP_LOGE(context_->GetNodeName(), "ubSize can't load slice, sliceLength=%ld, dSizeDy=%ld, dSizeCos=%ld.",
                    sliceLength_, dSizeDy, dSizeCos),
            return ge::GRAPH_FAILED);
        ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
    }
    ubFactorB_ = std::min(ubFactorB_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    if (ubFactorB_ <= 0) {
        ubFactorB_ = 1;
    }

    // Verify UB capacity
    int64_t dyBytes = UB_FACTOR * ubFactorB_ * dSizeDy;
    int64_t cosBytes;
    if (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST) {
        cosBytes = UB_FACTOR * ubFactorB_ * dSizeCos;
    } else {
        cosBytes = UB_COS_SIN_FACTOR * dSizeCos;
    }
    if (dyBytes + cosBytes > static_cast<int64_t>(aicoreParams_.ubSize)) {
        OP_LOGE(context_->GetNodeName(), "UB overflow: need %ld > %ld.", dyBytes + cosBytes, aicoreParams_.ubSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAAndB::PostTiling()
{
    tilingDataAAndB_.set_b(b_);
    tilingDataAAndB_.set_s(s_);
    tilingDataAAndB_.set_d(d_);
    tilingDataAAndB_.set_n(n_);
    tilingDataAAndB_.set_blockNumB(blockNumB_);
    tilingDataAAndB_.set_blockFactorB(blockFactorB_);
    tilingDataAAndB_.set_blockNumS(0);
    tilingDataAAndB_.set_blockFactorS(0);
    tilingDataAAndB_.set_ubFactorS(0);
    tilingDataAAndB_.set_ubFactorB(ubFactorB_);
    tilingDataAAndB_.set_ubLoopNumN(0);
    tilingDataAAndB_.set_ubFactorN(0);
    tilingDataAAndB_.set_ubTailFactorN(0);
    tilingDataAAndB_.set_usedCoreNum(usedCoreNum_);
    tilingDataAAndB_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingDataAAndB_.set_sliceStart(sliceStart_);
    tilingDataAAndB_.set_sliceEnd(sliceEnd_);
    tilingDataAAndB_.set_sliceLength(sliceLength_);
    tilingDataAAndB_.set_dSplitCoef(dSplitCoef_);

    size_t *userWorkspaceSize = context_->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = RESERVED_WORKSPACE;

    auto rawTilingDataPtr = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingDataPtr);
    if (tilingDataAAndB_.GetDataSize() > rawTilingDataPtr->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingDataAAndB_.SaveToBuffer(rawTilingDataPtr->GetData(), rawTilingDataPtr->GetCapacity());
    rawTilingDataPtr->SetDataSize(tilingDataAAndB_.GetDataSize());

    if (layout_ == InplacePartialRotaryMulGradLayout::NO_BROADCAST) {
        tilingKey_ = (isNoOp_ ? TILING_KEY_EMPTY : static_cast<uint64_t>(TILING_KEY_A));
    } else {
        tilingKey_ = (isNoOp_ ? TILING_KEY_EMPTY : static_cast<uint64_t>(TILING_KEY_B));
    }

    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);

    OP_LOGI(context_->GetNodeName(),
            "InplacePartialRotaryMulGradA&B tiling: B=%ld S=%ld D=%ld N=%ld cosb=%ld layout=%ld "
            "blockNumB=%ld blockFactorB=%ld blockNumS=0 blockFactorS=0 "
            "ubFactorS=0 ubFactorB=%ld ubLoopNumN=0 ubFactorN=0 ubTailFactorN=0 "
            "usedCoreNum=%ld rotaryMode=%ld dSplitCoef=%ld "
            "sliceStart=%ld sliceEnd=%ld sliceLength=%ld tilingKey=%lu",
            b_, s_, d_, n_, cosb_, static_cast<int64_t>(layout_), blockNumB_, blockFactorB_, ubFactorB_, usedCoreNum_,
            static_cast<int64_t>(rotaryMode_), dSplitCoef_, sliceStart_, sliceEnd_, sliceLength_, tilingKey_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingAAndB,
                             INPLACE_PARTIAL_ROTARY_MUL_GRAD_A_AND_B_TILING_PRIORITY);
} // namespace optiling
