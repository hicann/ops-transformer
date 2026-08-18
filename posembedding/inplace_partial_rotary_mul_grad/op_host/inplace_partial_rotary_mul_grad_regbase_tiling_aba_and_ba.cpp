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
 * \file inplace_partial_rotary_mul_grad_regbase_tiling_aba_and_ba.cpp
 * \brief ABA (cosb_!=1) and BA (cosb_==1) template tiling for InplacePartialRotaryMulGrad (BNSD layout)
 */
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"
namespace optiling {
constexpr uint64_t TILING_KEY_ABA = 201;
constexpr uint64_t TILING_KEY_BA = 202;
constexpr uint64_t INPLACE_PARTIAL_ROTARY_MUL_GRAD_ABA_AND_BA_TILING_PRIORITY = 10000;
constexpr int64_t UB_FACTOR = 4;

class InplacePartialRotaryMulGradRegbaseTilingAbaAndBa : public InplacePartialRotaryMulGradRegbaseTiling {
public:
    explicit InplacePartialRotaryMulGradRegbaseTilingAbaAndBa(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegbaseTiling(context)
    {}

protected:
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
            (layout_ == InplacePartialRotaryMulGradLayout::BNSD)) {
            return true;
        }
        return false;
    }

private:
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_ = 0;
    int64_t blockFactorB_ = 0;
    int64_t blockNumS_ = 0;
    int64_t blockFactorS_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorB_ = 0;
    int64_t ubLoopNumN_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t ubTailFactorN_ = 0;
    InplacePartialRotaryMulGradRegbaseTilingData tilingDataAbaAndBa_;
};

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::DoOpTiling()
{
    if (isNoOp_) {
        usedCoreNum_ = 1;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(SplitCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SplitCore failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "ComputeUbFactor failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::SplitCore()
{
    int64_t coreNum = static_cast<int64_t>(aicoreParams_.numBlocks);

    // B divisible by core count → split on B only
    if (b_ % coreNum == 0) {
        blockNumB_ = coreNum;
        blockFactorB_ = b_ / coreNum;
        blockNumS_ = 1;
        blockFactorS_ = s_;
        usedCoreNum_ = blockNumB_ * blockNumS_;
        return ge::GRAPH_SUCCESS;
    }

    // S divisible by core count → split on S only
    if (s_ % coreNum == 0) {
        blockNumS_ = coreNum;
        blockFactorS_ = s_ / coreNum;
        blockNumB_ = 1;
        blockFactorB_ = b_;
        usedCoreNum_ = blockNumB_ * blockNumS_;
        return ge::GRAPH_SUCCESS;
    }

    // Try B-priority split
    auto blockFactorB1 = Ops::Base::CeilDiv(b_, coreNum);
    auto blockNumB1 = Ops::Base::CeilDiv(b_, blockFactorB1);
    auto blockNumS1 = std::min(coreNum / blockNumB1, s_);
    auto blockFactorS1 = Ops::Base::CeilDiv(s_, blockNumS1);
    blockNumS1 = Ops::Base::CeilDiv(s_, blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    // Try S-priority split
    auto blockFactorS2 = Ops::Base::CeilDiv(s_, coreNum);
    auto blockNumS2 = Ops::Base::CeilDiv(s_, blockFactorS2);
    auto blockNumB2 = std::min(coreNum / blockNumS2, b_);
    auto blockFactorB2 = Ops::Base::CeilDiv(b_, blockNumB2);
    blockNumB2 = Ops::Base::CeilDiv(b_, blockFactorB2);
    auto usedCoreNum2 = blockNumB2 * blockNumS2;

    if (usedCoreNum1 >= usedCoreNum2) {
        blockNumB_ = blockNumB1;
        blockFactorB_ = blockFactorB1;
        blockNumS_ = blockNumS1;
        blockFactorS_ = blockFactorS1;
    } else {
        blockNumB_ = blockNumB2;
        blockFactorB_ = blockFactorB2;
        blockNumS_ = blockNumS2;
        blockFactorS_ = blockFactorS2;
    }
    usedCoreNum_ = blockNumB_ * blockNumS_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::ComputeUbFactor()
{
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);

    // Per-type aligned row sizes in bytes
    int64_t dSizeDy = Ops::Base::CeilAlign(sliceLength_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    int64_t dSizeCos = Ops::Base::CeilAlign(sliceLength_ * cosTypeSize / dSplitCoef_, blockSize_) * dSplitCoef_;

    // Kernel UB layout:
    //   dy/dx: UB_FACTOR * ubFactorB * ubFactorS * ubFactorN * dSizeDy
    //   cos/sin (Broadcast): UB_FACTOR * ubFactorS * dSizeCos
    //   cos/sin (!Broadcast): UB_FACTOR * ubFactorB * ubFactorS * dSizeCos
    //
    // Use effectiveCosOverhead like BAB/AB to handle mixed precision uniformly:
    //   Overhead = max(1, CeilDiv(dSizeCos, dSizeDy)) "dy-row equivalents" per cos row
    int64_t effectiveCosOverhead = std::max(int64_t(1), Ops::Base::CeilDiv(dSizeCos, dSizeDy));

    // baseBlockInUb: effective dy-rows in UB/4
    int64_t baseBlockInUb =
        Ops::Base::FloorAlign(static_cast<int64_t>(aicoreParams_.ubSize / UB_FACTOR), blockSize_) / dSizeDy;
    if (baseBlockInUb < effectiveCosOverhead + 1) {
        OP_LOGE(context_->GetNodeName(), "ubSize can't load slice, sliceLength=%ld, dSizeDy=%ld, dSizeCos=%ld.",
                sliceLength_, dSizeDy, dSizeCos);
        return ge::GRAPH_FAILED;
    }

    if (cosb_ == 1) {
        // BA: cos broadcasts on B → UB = 4*ubFactorB*ubFactorS*ubFactorN*dSizeDy + 4*ubFactorS*dSizeCos
        //   = 4*ubFactorS*dSizeDy * (ubFactorB*ubFactorN + effectiveCosOverhead)
        // Constraint: ubFactorS * (ubFactorB * ubFactorN + effectiveCosOverhead) <= baseBlockInUb

        // Step 1: ubFactorN = min(n_, baseBlockInUb - effectiveCosOverhead)
        //   (reserve overhead for cos row, assume ubFactorB>=1, ubFactorS>=1)
        ubFactorN_ = std::min(n_, baseBlockInUb - effectiveCosOverhead);
        ubFactorN_ = std::min(ubFactorN_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
        if (ubFactorN_ <= 0) {
            ubFactorN_ = 1;
        }

        // Step 2: ubFactorB * ubFactorS <= baseBlockInUb / (ubFactorN + effectiveCosOverhead/ubFactorB)
        // Approximate: remaining = baseBlockInUb / (ubFactorN + effectiveCosOverhead/ubFactorB)
        // For conservative estimate, use effectiveCosOverhead directly:
        int64_t afterN = Ops::Base::FloorDiv(baseBlockInUb, ubFactorN_ + effectiveCosOverhead);
        ubFactorB_ = std::min(blockFactorB_, afterN);
        if (ubFactorB_ <= 0) {
            ubFactorB_ = 1;
        }

        // Step 3: ubFactorS from remaining budget
        ubFactorS_ =
            std::min(blockFactorS_, Ops::Base::FloorDiv(baseBlockInUb, ubFactorB_ * ubFactorN_ + effectiveCosOverhead));
    } else {
        // ABA: cos varies with B → UB = 4*ubFactorB*ubFactorS*(ubFactorN*dSizeDy + dSizeCos)
        //   = 4*ubFactorB*ubFactorS*dSizeDy * (ubFactorN + CeilDiv(dSizeCos, dSizeDy))
        //   => ubFactorB * ubFactorS * (ubFactorN + effectiveCosOverhead) <= baseBlockInUb

        ubFactorN_ = std::min(n_, baseBlockInUb - effectiveCosOverhead);
        ubFactorN_ = std::min(ubFactorN_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
        if (ubFactorN_ <= 0) {
            ubFactorN_ = 1;
        }

        int64_t afterN = Ops::Base::FloorDiv(baseBlockInUb, ubFactorN_ + effectiveCosOverhead);
        ubFactorB_ = std::min(blockFactorB_, afterN);
        if (ubFactorB_ <= 0) {
            ubFactorB_ = 1;
        }

        ubFactorS_ = std::min(blockFactorS_,
                              Ops::Base::FloorDiv(baseBlockInUb, ubFactorB_ * (ubFactorN_ + effectiveCosOverhead)));
    }

    if (ubFactorS_ == 0) {
        ubFactorS_ = 1;
    }
    if (ubFactorB_ == 0) {
        ubFactorB_ = 1;
    }
    if (ubFactorN_ == 0) {
        ubFactorN_ = 1;
    }

    // Final UB verification
    int64_t dyBytes = UB_FACTOR * ubFactorB_ * ubFactorS_ * ubFactorN_ * dSizeDy;
    int64_t cosBytes;
    if (cosb_ == 1) {
        cosBytes = UB_FACTOR * ubFactorS_ * dSizeCos;
    } else {
        cosBytes = UB_FACTOR * ubFactorB_ * ubFactorS_ * dSizeCos;
    }
    if (dyBytes + cosBytes > static_cast<int64_t>(aicoreParams_.ubSize)) {
        OP_LOGE(context_->GetNodeName(), "UB overflow: need %ld > %ld.", dyBytes + cosBytes, aicoreParams_.ubSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAbaAndBa::PostTiling()
{
    tilingDataAbaAndBa_.set_b(b_);
    tilingDataAbaAndBa_.set_s(s_);
    tilingDataAbaAndBa_.set_d(d_);
    tilingDataAbaAndBa_.set_n(n_);
    tilingDataAbaAndBa_.set_blockNumB(blockNumB_);
    tilingDataAbaAndBa_.set_blockFactorB(blockFactorB_);
    tilingDataAbaAndBa_.set_blockNumS(blockNumS_);
    tilingDataAbaAndBa_.set_blockFactorS(blockFactorS_);
    tilingDataAbaAndBa_.set_ubFactorS(ubFactorS_);
    tilingDataAbaAndBa_.set_ubFactorB(ubFactorB_);
    tilingDataAbaAndBa_.set_ubLoopNumN(ubLoopNumN_);
    tilingDataAbaAndBa_.set_ubFactorN(ubFactorN_);
    tilingDataAbaAndBa_.set_ubTailFactorN(ubTailFactorN_);
    tilingDataAbaAndBa_.set_usedCoreNum(usedCoreNum_);
    tilingDataAbaAndBa_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingDataAbaAndBa_.set_sliceStart(sliceStart_);
    tilingDataAbaAndBa_.set_sliceEnd(sliceEnd_);
    tilingDataAbaAndBa_.set_sliceLength(sliceLength_);
    tilingDataAbaAndBa_.set_dSplitCoef(dSplitCoef_);

    size_t *userWorkspaceSize = context_->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = RESERVED_WORKSPACE;

    auto rawTilingDataPtr = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingDataPtr);
    if (tilingDataAbaAndBa_.GetDataSize() > rawTilingDataPtr->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingDataAbaAndBa_.SaveToBuffer(rawTilingDataPtr->GetData(), rawTilingDataPtr->GetCapacity());
    rawTilingDataPtr->SetDataSize(tilingDataAbaAndBa_.GetDataSize());

    if (cosb_ == 1) {
        tilingKey_ = (isNoOp_ ? TILING_KEY_EMPTY : static_cast<uint64_t>(TILING_KEY_BA));
    } else {
        tilingKey_ = (isNoOp_ ? TILING_KEY_EMPTY : static_cast<uint64_t>(TILING_KEY_ABA));
    }

    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);

    OP_LOGI(context_->GetNodeName(),
            "InplacePartialRotaryMulGradABA&BA tiling: B=%ld S=%ld D=%ld N=%ld cosb=%ld "
            "blockNumB=%ld blockFactorB=%ld blockNumS=%ld blockFactorS=%ld "
            "ubFactorS=%ld ubFactorB=%ld ubLoopNumN=%ld ubFactorN=%ld ubTailFactorN=%ld "
            "usedCoreNum=%ld rotaryMode=%ld dSplitCoef=%ld "
            "sliceStart=%ld sliceEnd=%ld sliceLength=%ld tilingKey=%lu",
            b_, s_, d_, n_, cosb_, blockNumB_, blockFactorB_, blockNumS_, blockFactorS_, ubFactorS_, ubFactorB_,
            ubLoopNumN_, ubFactorN_, ubTailFactorN_, usedCoreNum_, static_cast<int64_t>(rotaryMode_), dSplitCoef_,
            sliceStart_, sliceEnd_, sliceLength_, tilingKey_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingAbaAndBa,
                             INPLACE_PARTIAL_ROTARY_MUL_GRAD_ABA_AND_BA_TILING_PRIORITY);
} // namespace optiling
