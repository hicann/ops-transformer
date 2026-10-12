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
 * \file rope_grad_regbase_tiling_aba_and_ba.cc
 * \brief
 */
#include "rotary_position_embedding_grad_tiling.h"

using namespace AscendC;

namespace optiling {
constexpr uint64_t ROPE_GRAD_ABA_AND_BA_TILING_PRIORITY = 10000;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t HALF_INTERLEAVE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;

class RopeGradRegBaseTilingClassABAAndBA : public RopeGradRegBaseTilingClass {
public:
    explicit RopeGradRegBaseTilingClassABAAndBA(gert::TilingContext* context)
        : RopeGradRegBaseTilingClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus PostTiling() override;
    void SetTilingData();
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;

private:
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t ubFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubFactorN_{0};
    int64_t ubLoopNumN_{0};
    int64_t ubTailFactorN_{0};
    int64_t blockNumB_{0};
    int64_t blockFactorB_{0};
    int64_t blockNumS_{0};
    int64_t blockFactorS_{0};
};

bool RopeGradRegBaseTilingClassABAAndBA::IsCapable()
{
    // BNSD对应11SD和B1SD两种brc模式
    return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) && (layout_ == RopeLayout::BNSD);
}

ge::graphStatus RopeGradRegBaseTilingClassABAAndBA::SplitCore()
{
    // B大于等于核数，且能被核数整除，则仅在B轴分核
    if (b_ % aicoreParams_.numBlocks == 0) {
        blockNumB_ = aicoreParams_.numBlocks;
        blockFactorB_ = b_ / aicoreParams_.numBlocks;
        blockNumS_ = 1;
        blockFactorS_ = s_;
        usedCoreNum_ = blockNumB_ * blockNumS_;
        return ge::GRAPH_SUCCESS;
    }

    // S大于等于核数，且能被核数整除，则仅在S轴分核
    if (s_ % aicoreParams_.numBlocks == 0) {
        blockNumS_ = aicoreParams_.numBlocks;
        blockFactorS_ = s_ / aicoreParams_.numBlocks;
        blockNumB_ = 1;
        blockFactorB_ = b_;
        usedCoreNum_ = blockNumB_ * blockNumS_;
        return ge::GRAPH_SUCCESS;
    }

    // 尝试优先对B分核，再尝试优先对S分核，比较二者切分后的总核数
    auto blockFactorB1 = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), aicoreParams_.numBlocks);
    auto blockNumB1 = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), blockFactorB1);
    auto blockNumS1 = std::min(static_cast<uint64_t>(s_), aicoreParams_.numBlocks / blockNumB1);
    auto blockFactorS1 = Ops::Base::CeilDiv(static_cast<uint64_t>(s_), blockNumS1);
    blockNumS1 = Ops::Base::CeilDiv(static_cast<uint64_t>(s_), blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    auto blockFactorS2 = Ops::Base::CeilDiv(static_cast<uint64_t>(s_), aicoreParams_.numBlocks);
    auto blockNumS2 = Ops::Base::CeilDiv(static_cast<uint64_t>(s_), blockFactorS2);
    auto blockNumB2 = std::min(static_cast<uint64_t>(b_), aicoreParams_.numBlocks / blockNumS2);
    auto blockFactorB2 = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), blockNumB2);
    blockNumB2 = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), blockFactorB2);
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

ge::graphStatus RopeGradRegBaseTilingClassABAAndBA::ComputeUbFactor()
{
    ubFactorS_ = 1;
    ubFactorB_ = 1;
    ubFactorN_ = 1;
    // UB占用大小：
    // 11sd：4 * (bn + 1) * s * dAlign
    // b1sd：4 * b * (n + 1) * s * dAlign
    int64_t dSplitCoef = 1;
    if (rotaryMode_ == RotaryPosEmbeddingMode::HALF || rotaryMode_ == RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        dSplitCoef = HALF_INTERLEAVE_COEF;
    } else if (rotaryMode_ == RotaryPosEmbeddingMode::QUARTER) {
        dSplitCoef = QUARTER_MODE_COEF;
    }
    int64_t dSize = Ops::Base::CeilAlign(d_ * GetSizeByDataType(dtype_) / dSplitCoef, this->blockSize_) * dSplitCoef;
    int64_t numOfDAvailable = Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize);
    OP_CHECK_IF(numOfDAvailable < ubFactorB_ + 1,
                OP_LOGE(context_, "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
                        static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize * (ubFactorB_ + 1)),
                return ge::GRAPH_FAILED);

    ubFactorS_ = std::min(blockFactorS_, Ops::Base::FloorDiv(numOfDAvailable, ubFactorB_ + 1));
    ubFactorS_ = std::min(ubFactorS_, MAX_COPY_BLOCK_COUNT / dSplitCoef);
    numOfDAvailable /= ubFactorS_;
    if (numOfDAvailable <= ubFactorB_ + 1) {
        return ge::GRAPH_SUCCESS;
    }

    if (cosb_ == 1) {
        numOfDAvailable -= 1;
        ubFactorN_ = std::min(n_, numOfDAvailable);
        numOfDAvailable /= ubFactorN_;
    } else {
        ubFactorN_ = std::min(n_, numOfDAvailable - 1);
        numOfDAvailable /= (ubFactorN_ + 1);
    }

    if (numOfDAvailable <= 1) {
        return ge::GRAPH_SUCCESS;
    }
    ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);

    return ge::GRAPH_SUCCESS;
}

void RopeGradRegBaseTilingClassABAAndBA::SetTilingData()
{
    tilingData_->ropeGradParams.b = b_;
    tilingData_->ropeGradParams.s = s_;
    tilingData_->ropeGradParams.d = d_;
    tilingData_->ropeGradParams.n = n_;
    tilingData_->ropeGradParams.blockNumB = blockNumB_;
    tilingData_->ropeGradParams.blockFactorB = blockFactorB_;
    tilingData_->ropeGradParams.blockNumS = blockNumS_;
    tilingData_->ropeGradParams.blockFactorS = blockFactorS_;
    tilingData_->ropeGradParams.ubFactorS = ubFactorS_;
    tilingData_->ropeGradParams.ubFactorB = ubFactorB_;
    tilingData_->ropeGradParams.ubLoopNumN = ubLoopNumN_;
    tilingData_->ropeGradParams.ubFactorN = ubFactorN_;
    tilingData_->ropeGradParams.ubTailFactorN = ubTailFactorN_;
    tilingData_->ropeGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->ropeGradParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->dCosFlag = dCosFlag_;

    OP_LOGI(context_->GetNodeName(),
            "RopeGradRegBaseTilingClassABAAndBA tilingData: "
            "B is %ld, S is %ld, D is %ld, N is %ld, blockNumB %ld, blockFactorB_ is %ld, "
            "blockNumS %ld, blockFactorS is %ld, ubFactorS is %ld, ubFactorB is %ld,"
            "ubLoopNumN is %ld, ubFactorN is %ld, ubTailFactorN is %ld, rotaryMode is %ld, tilingKey is %ld,"
            "usedCoreNum is %ld.",
            tilingData_->ropeGradParams.b, tilingData_->ropeGradParams.s, tilingData_->ropeGradParams.d,
            tilingData_->ropeGradParams.n, tilingData_->ropeGradParams.blockNumB,
            tilingData_->ropeGradParams.blockFactorB, tilingData_->ropeGradParams.blockNumS,
            tilingData_->ropeGradParams.blockFactorS, tilingData_->ropeGradParams.ubFactorS,
            tilingData_->ropeGradParams.ubFactorB, tilingData_->ropeGradParams.ubLoopNumN,
            tilingData_->ropeGradParams.ubFactorN, tilingData_->ropeGradParams.ubTailFactorN,
            tilingData_->ropeGradParams.rotaryMode, tilingKey_, tilingData_->ropeGradParams.usedCoreNum);
}

ge::graphStatus RopeGradRegBaseTilingClassABAAndBA::DoOpTiling()
{
    OP_CHECK_IF(SplitCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "failed to split core."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "failed to compute ub factor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "InitTilingData failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingReduce failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClassABAAndBA::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClassABAAndBA::PostTiling()
{
    SetTilingData();
    SetRotaryXTilingData();
    uint32_t tilingKey = (cosb_ == 1 ? static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_BA) :
                                       static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_ABA));
    SetTilingKeyBlockDim(tilingKey);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbeddingGrad, RopeGradRegBaseTilingClassABAAndBA,
                             ROPE_GRAD_ABA_AND_BA_TILING_PRIORITY);
} // namespace optiling
