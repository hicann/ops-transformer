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
 * \file apply_rotary_pos_emb_grad_tiling_a.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_grad_tiling.h"
#include <algorithm>

namespace optiling {
constexpr uint64_t APPLY_ROPE_GRAD_A_TILING_PRIORITY = 40000;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t HALF_COEF = 2;
constexpr int64_t STG1_BUF_NUM = 12;
constexpr int64_t STG3_INPUT_BUF_NUM = 12;
constexpr int64_t STG3_OUTPUT_BUF_NUM = 4;
constexpr int64_t STG3_TMP_BUF_NUM = 1;

class ApplyRotaryPosEmbGradRegbaseTilingClassA : public ApplyRotaryPosEmbGradRegbaseTilingClass {
public:
    explicit ApplyRotaryPosEmbGradRegbaseTilingClassA(gert::TilingContext *context)
        : ApplyRotaryPosEmbGradRegbaseTilingClass(context)
    {}

protected:
    bool IsCapable() override
    {
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) &&
               (layout_ == ApplyRopeGradLayout::NO_BROADCAST);
    }
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus MergeDim();
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_{0};
    int64_t blockFactorB_{0};
    int64_t ubFactorB_{0};
};

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassA::MergeDim()
{
    b_ = b_ * nQ_ * s_;
    nQ_ = 1;
    nK_ = 1;
    s_ = 1;
    cosb_ = b_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassA::SplitCore()
{
    blockFactorB_ = Ops::Base::CeilDiv(static_cast<int64_t>(b_), static_cast<int64_t>(aicoreParams_.numBlocks));
    blockNumB_ = Ops::Base::CeilDiv(static_cast<int64_t>(b_), blockFactorB_);
    usedCoreNum_ = blockNumB_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassA::ComputeUbFactor()
{
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t dBufBytes = Ops::Base::CeilAlign(d_ / HALF_COEF * typeSize, blockSize_) * HALF_COEF;
    int64_t ubMaxBytes = STG1_BUF_NUM * dBufBytes;
    if (dCosFlag_ == 1) {
        int64_t inputBytes = d_ * typeSize;
        if (typeSize != static_cast<int64_t>(sizeof(float))) {
            inputBytes = d_ * (typeSize + static_cast<int64_t>(sizeof(float)));
        }
        int64_t accBytes = d_ * static_cast<int64_t>(sizeof(float));
        int64_t stg3Bytes = STG3_INPUT_BUF_NUM * inputBytes + (STG3_OUTPUT_BUF_NUM + STG3_TMP_BUF_NUM) * accBytes;
        ubMaxBytes = std::max(ubMaxBytes, stg3Bytes);
    }
    int64_t numDAvail = Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), ubMaxBytes);
    OP_CHECK_IF(numDAvail < 1, OP_LOGE(context_, "D too big for UB"), return ge::GRAPH_FAILED);
    ubFactorB_ = std::min(blockFactorB_, numDAvail);
    bool dHalfAligned = ((d_ / HALF_COEF) % (blockSize_ / typeSize)) == 0;
    int64_t maxCopyB = dHalfAligned ? MAX_COPY_BLOCK_COUNT : MAX_COPY_BLOCK_COUNT / HALF_COEF;
    ubFactorB_ = std::min(ubFactorB_, maxCopyB);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassA::DoOpTiling()
{
    OP_CHECK_IF(MergeDim() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "MergeDim failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SplitCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "SplitCore failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ComputeUbFactor failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "InitTilingData failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassA::PostTiling()
{
    tilingData_->ropeGradParams.layout = static_cast<uint64_t>(layout_);
    tilingData_->ropeGradParams.b = b_;
    tilingData_->ropeGradParams.s = s_;
    tilingData_->ropeGradParams.d = d_;
    tilingData_->ropeGradParams.nQ = nQ_;
    tilingData_->ropeGradParams.nK = nK_;
    tilingData_->ropeGradParams.blockNumB = blockNumB_;
    tilingData_->ropeGradParams.blockFactorB = blockFactorB_;
    tilingData_->ropeGradParams.blockNumS = 1;
    tilingData_->ropeGradParams.blockFactorS = 1;
    tilingData_->ropeGradParams.ubFactorS = ubFactorB_;
    tilingData_->ropeGradParams.ubLoopNumN = 1;
    tilingData_->ropeGradParams.ubFactorN = 1;
    tilingData_->ropeGradParams.ubTailFactorN = 1;
    tilingData_->ropeGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->ropeGradParams.rotaryMode = static_cast<int64_t>(rotaryMode_);

    SetTilingKeyBlockDim(static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A));
    tilingData_->dCosFlag = dCosFlag_;
    tilingData_->layout = static_cast<uint32_t>(layout_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ApplyRotaryPosEmbGrad, ApplyRotaryPosEmbGradRegbaseTilingClassA,
                             APPLY_ROPE_GRAD_A_TILING_PRIORITY);
} // namespace optiling
