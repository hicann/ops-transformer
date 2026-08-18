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
 * \file inplace_partial_rotary_mul_grad_regbase_tiling_ab.cpp
 * \brief AB template tiling for InplacePartialRotaryMulGrad (SBND layout, BS+N blocking)
 */
#include "inplace_partial_rotary_mul_grad_regbase_tiling.h"
namespace optiling {
constexpr uint64_t TILING_KEY_AB = 204;
constexpr uint64_t INPLACE_PARTIAL_ROTARY_MUL_GRAD_AB_TILING_PRIORITY = 25000;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t DB_FLAG = 2;

class InplacePartialRotaryMulGradRegbaseTilingAb : public InplacePartialRotaryMulGradRegbaseTiling {
public:
    explicit InplacePartialRotaryMulGradRegbaseTilingAb(gert::TilingContext *context)
        : InplacePartialRotaryMulGradRegbaseTiling(context)
    {}

protected:
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
            (layout_ == InplacePartialRotaryMulGradLayout::SBND)) {
            return true;
        }
        return false;
    }

private:
    int64_t blockNumBS_ = 0;
    int64_t blockFactorBS_ = 0;
    int64_t blockTailBS_ = 0;
    int64_t blockNumN_ = 0;
    int64_t blockFactorN_ = 0;
    int64_t blockTailN_ = 0;
    int64_t ubFactorBS_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t dAlign_ = 0;
    int64_t bn_ = 0;
    InplacePartialRotaryMulGradRegbaseTilingDataAb tilingDataAb_;
};

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAb::DoOpTiling()
{
    if (isNoOp_) {
        usedCoreNum_ = 1;
        return ge::GRAPH_SUCCESS;
    }
    int64_t bs = b_ * s_;
    bn_ = n_;
    if (cosb_ == 1) {
        bs = s_;
        bn_ = b_ * n_;
    }
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);

    // Compute per-type aligned row sizes (in elements), matching kernel's dAlignLenDy/dAlignLenCos
    int64_t dAlignDy = Ops::Base::CeilAlign(sliceLength_ / dSplitCoef_, blockSize_ / typeSize) * dSplitCoef_;
    int64_t dAlignCos = Ops::Base::CeilAlign(sliceLength_ / dSplitCoef_, blockSize_ / cosTypeSize) * dSplitCoef_;
    int64_t dSizeDy = dAlignDy * typeSize;      // bytes per dy/dx row
    int64_t dSizeCos = dAlignCos * cosTypeSize; // bytes per cos/sin row
    dAlign_ = dAlignDy;                         // store dy alignment in tiling data for logging

    int64_t coreNum = static_cast<int64_t>(aicoreParams_.numBlocks);
    blockFactorBS_ = Ops::Base::CeilDiv(bs, coreNum);
    blockNumBS_ = Ops::Base::CeilDiv(bs, blockFactorBS_);
    blockTailBS_ = bs - (blockNumBS_ - 1) * blockFactorBS_;

    if (bs <= coreNum / CONST_TWO) {
        if (blockNumBS_ == 0) {
            OP_LOGE(context_->GetNodeName(), "blockNumBS_ == 0");
            return ge::GRAPH_FAILED;
        }
        blockNumN_ = coreNum / blockNumBS_;
        blockFactorN_ = Ops::Base::CeilDiv(bn_, blockNumN_);
        blockNumN_ = Ops::Base::CeilDiv(bn_, blockFactorN_);
        blockTailN_ = bn_ - (blockNumN_ - 1) * blockFactorN_;
    } else {
        blockNumN_ = 1;
        blockFactorN_ = bn_;
        blockTailN_ = bn_;
    }
    usedCoreNum_ = blockNumBS_ * blockNumN_;

    // UB split: derivable rows from half UB with double-buffer headroom
    //   UB total = 4 * ubFactorBS * (ubFactorN * dSizeDy + dSizeCos) <= ubSize
    //   => ubFactorBS * (ubFactorN + CeilDiv(dSizeCos, dSizeDy)) <= ubSize/(4*dSizeDy)
    int64_t effectiveCosOverhead = std::max(int64_t(1), Ops::Base::CeilDiv(dSizeCos, dSizeDy));
    int64_t baseBlockInUb =
        Ops::Base::FloorAlign(static_cast<int64_t>(aicoreParams_.ubSize / CONST_TWO / DB_FLAG), blockSize_) / dSizeDy;
    if (baseBlockInUb < effectiveCosOverhead + 1) {
        OP_LOGE(context_->GetNodeName(),
                "ubSize can't load slice, sliceLength=%ld, dSizeDy=%ld, dSizeCos=%ld, baseBlockInUb=%ld.", sliceLength_,
                dSizeDy, dSizeCos, baseBlockInUb);
        return ge::GRAPH_FAILED;
    }

    ubFactorN_ = std::min(blockFactorN_, baseBlockInUb - effectiveCosOverhead);
    ubFactorN_ = std::min(ubFactorN_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    if (ubFactorN_ <= 0) {
        ubFactorN_ = 1;
    }

    ubFactorBS_ = std::min(Ops::Base::FloorDiv(baseBlockInUb, ubFactorN_ + effectiveCosOverhead), blockFactorBS_);
    ubFactorBS_ = (ubFactorBS_ == 0) ? 1 : ubFactorBS_;

    // Verify UB capacity (4 = CONST_TWO * DB_FLAG: 2cos+2sin or 2dy+2dx, double-buffered)
    int64_t dyBytes = CONST_TWO * DB_FLAG * ubFactorBS_ * ubFactorN_ * dSizeDy;
    int64_t cosBytes = CONST_TWO * DB_FLAG * ubFactorBS_ * dSizeCos;
    if (dyBytes + cosBytes > static_cast<int64_t>(aicoreParams_.ubSize)) {
        OP_LOGE(context_->GetNodeName(), "UB overflow: need %ld > %ld.", dyBytes + cosBytes, aicoreParams_.ubSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradRegbaseTilingAb::PostTiling()
{
    tilingDataAb_.set_b(b_);
    tilingDataAb_.set_s(s_);
    tilingDataAb_.set_d(d_);
    tilingDataAb_.set_n(bn_);
    tilingDataAb_.set_dAlign(dAlign_);
    tilingDataAb_.set_dSplitCoef(dSplitCoef_);
    tilingDataAb_.set_blockNumBS(blockNumBS_);
    tilingDataAb_.set_blockFactorBS(blockFactorBS_);
    tilingDataAb_.set_blockTailBS(blockTailBS_);
    tilingDataAb_.set_blockNumN(blockNumN_);
    tilingDataAb_.set_blockFactorN(blockFactorN_);
    tilingDataAb_.set_blockTailN(blockTailN_);
    tilingDataAb_.set_ubFactorBS(ubFactorBS_);
    tilingDataAb_.set_ubFactorN(ubFactorN_);
    tilingDataAb_.set_usedCoreNum(usedCoreNum_);
    tilingDataAb_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingDataAb_.set_sliceStart(sliceStart_);
    tilingDataAb_.set_sliceEnd(sliceEnd_);
    tilingDataAb_.set_sliceLength(sliceLength_);

    // set userWorkspace
    size_t *userWorkspaceSize = context_->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = RESERVED_WORKSPACE;

    auto rawTilingDataPtr = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingDataPtr);
    if (tilingDataAb_.GetDataSize() > rawTilingDataPtr->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingDataAb_.SaveToBuffer(rawTilingDataPtr->GetData(), rawTilingDataPtr->GetCapacity());
    rawTilingDataPtr->SetDataSize(tilingDataAb_.GetDataSize());

    tilingKey_ = (isNoOp_ ? TILING_KEY_EMPTY : static_cast<uint64_t>(TILING_KEY_AB));
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);

    OP_LOGI(context_->GetNodeName(),
            "InplacePartialRotaryMulGradAB tiling: B=%ld S=%ld D=%ld N=%ld "
            "dAlign=%ld dSplitCoef=%ld "
            "blockNumBS=%ld blockFactorBS=%ld blockTailBS=%ld "
            "blockNumN=%ld blockFactorN=%ld blockTailN=%ld "
            "ubFactorBS=%ld ubFactorN=%ld "
            "usedCoreNum=%ld rotaryMode=%ld "
            "sliceStart=%ld sliceEnd=%ld sliceLength=%ld "
            "tilingKey=%lu",
            b_, s_, d_, bn_, dAlign_, dSplitCoef_, blockNumBS_, blockFactorBS_, blockTailBS_, blockNumN_, blockFactorN_,
            blockTailN_, ubFactorBS_, ubFactorN_, usedCoreNum_, static_cast<int64_t>(rotaryMode_), sliceStart_,
            sliceEnd_, sliceLength_, tilingKey_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMulGrad, InplacePartialRotaryMulGradRegbaseTilingAb,
                             INPLACE_PARTIAL_ROTARY_MUL_GRAD_AB_TILING_PRIORITY);
} // namespace optiling
