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
 * \file apply_rotary_pos_emb_regbase_tiling_aba_and_ba_arch35.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"

// using namespace AscendC;

namespace optiling {

constexpr uint64_t ROPE_ABA_AND_BA_TILING_PRIORITY = 20000;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t MAX_COPY_BLOCK_COUNT = 4095;
constexpr uint64_t TILING_KEY_ABA = 20010;
constexpr uint64_t TILING_KEY_BA = 20011;
constexpr int64_t HALF_INTERLEAVE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;

class ApplyRotaryPosEmbTilingABAAndBA : public ApplyRotaryPosEmbRegbaseTilingBaseClass {
public:
    explicit ApplyRotaryPosEmbTilingABAAndBA(gert::TilingContext *context)
        : ApplyRotaryPosEmbRegbaseTilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void SetTilingData();

private:
    ge::graphStatus SplitCore();
    ge::graphStatus ComputeUbFactor();

    int64_t blockNumB_{0};
    int64_t blockFactorB_{0};
    int64_t blockNumS_{0};
    int64_t blockFactorS_{0};
    int64_t ubFactorB_{0};
    int64_t ubLoopNumB_{0};
    int64_t ubTailFactorB_{0};
    int64_t ubFactorS_{0};
    int64_t ubLoopNumS_{0};
    int64_t ubTailFactorS_{0};
    int64_t ubFactorQN_{0};
    int64_t ubLoopNumQN_{0};
    int64_t ubTailFactorQN_{0};
    int64_t ubFactorKN_{0};
    int64_t ubLoopNumKN_{0};
    int64_t ubTailFactorKN_{0};
    ApplyRotaryPosEmbRegbaseTilingData tilingData_;
};

bool ApplyRotaryPosEmbTilingABAAndBA::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    // 当前只支持bnsd格式下，11sd（ba）和b1sd（aba）两种boardcast，后续可支持全部的ba和aba boardcast
    return layout_ == ApplyRotaryPosEmbLayout::BNSD;
}

ge::graphStatus ApplyRotaryPosEmbTilingABAAndBA::SplitCore()
{
    // B大于等于核数，且能被核数整除，则仅在B轴分核
    if (b_ % aicoreParams_.numBlocks == 0) {
        blockNumB_ = aicoreParams_.numBlocks;
        blockFactorB_ = b_ / aicoreParams_.numBlocks;
        blockNumS_ = 1;
        blockFactorS_ = s_;
        return ge::GRAPH_SUCCESS;
    }

    // S大于等于核数，且能被核数整除，则仅在S轴分核
    if (s_ % aicoreParams_.numBlocks == 0) {
        blockNumS_ = aicoreParams_.numBlocks;
        blockFactorS_ = s_ / aicoreParams_.numBlocks;
        blockNumB_ = 1;
        blockFactorB_ = b_;
        return ge::GRAPH_SUCCESS;
    }

    // 尝试优先对B分核，再尝试优先对S分核，比较二者切分后的总核数
    auto blockFactorB1 = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), aicoreParams_.numBlocks);
    auto blockNumB1 = Ops::Base::CeilDiv(static_cast<uint64_t>(b_), blockFactorB1);
    if (blockNumB1 == 0) {
        OP_LOGE(context_->GetNodeName(), "blockNumB1 can't be 0.");
        return ge::GRAPH_FAILED;
    }
    auto blockNumS1 = std::min(static_cast<uint64_t>(s_), aicoreParams_.numBlocks / blockNumB1);
    auto blockFactorS1 = Ops::Base::CeilDiv(static_cast<uint64_t>(s_), blockNumS1);
    blockNumS1 = Ops::Base::CeilDiv(static_cast<uint64_t>(s_), blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    auto blockFactorS2 = Ops::Base::CeilDiv(static_cast<uint64_t>(s_), aicoreParams_.numBlocks);
    auto blockNumS2 = Ops::Base::CeilDiv(static_cast<uint64_t>(s_), blockFactorS2);
    if (blockNumS2 == 0) {
        OP_LOGE(context_->GetNodeName(), "blockNumS2 can't be 0.");
        return ge::GRAPH_FAILED;
    }
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
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbTilingABAAndBA::ComputeUbFactor()
{
    ubFactorS_ = 1;
    ubFactorB_ = 1;
    ubFactorQN_ = 1;
    ubFactorKN_ = 1;
    // UB占用大小：
    // 11sd：4 * (bn + 1) * s * dAlign
    // b1sd：4 * b * (n + 1) * s * dAlign
    int64_t dSize = Ops::Base::CeilAlign(d_ * GetSizeByDataType(dtype_) / dSplitCoef_, this->blockSize_) * dSplitCoef_;
    int64_t numOfDAvailable = Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize);
    OP_CHECK_IF(numOfDAvailable < ubFactorB_ + 1,
                OP_LOGE(context_, "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
                        static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize * (ubFactorB_ + 1)),
                return ge::GRAPH_FAILED);

    ubFactorS_ = std::min(blockFactorS_, Ops::Base::FloorDiv(numOfDAvailable, ubFactorB_ + 1));
    ubFactorS_ = std::min(ubFactorS_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    if (ubFactorS_ == 0) {
        OP_LOGE(context_->GetNodeName(), "ubFactorS_ can't be 0.");
        return ge::GRAPH_FAILED;
    }
    numOfDAvailable /= ubFactorS_;
    if (numOfDAvailable <= ubFactorB_ + 1) {
        return ge::GRAPH_SUCCESS;
    }

    if (cosb_ == 1) {
        numOfDAvailable -= 1;
        ubFactorQN_ = std::min(std::max(qn_, kn_), numOfDAvailable);
        if (ubFactorQN_ == 0) {
            OP_LOGE(context_->GetNodeName(), "ubFactorQN_ can't be 0.");
            return ge::GRAPH_FAILED;
        }
        ubFactorKN_ = ubFactorQN_;
        numOfDAvailable /= ubFactorQN_;
    } else {
        ubFactorQN_ = std::min(std::max(qn_, kn_), numOfDAvailable - 1);
        ubFactorKN_ = ubFactorQN_;
        numOfDAvailable /= (ubFactorQN_ + 1);
    }

    if (numOfDAvailable <= 1) {
        return ge::GRAPH_SUCCESS;
    }
    ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);

    return ge::GRAPH_SUCCESS;
}

void ApplyRotaryPosEmbTilingABAAndBA::SetTilingData()
{
    tilingData_.set_B(b_);
    tilingData_.set_CosB(cosb_);
    tilingData_.set_S(s_);
    tilingData_.set_D(d_);
    tilingData_.set_QN(qn_);
    tilingData_.set_KN(kn_);
    tilingData_.set_blockNumB(blockNumB_);
    tilingData_.set_blockFactorB(blockFactorB_);
    tilingData_.set_blockNumS(blockNumS_);
    tilingData_.set_blockFactorS(blockFactorS_);
    tilingData_.set_ubLoopNumS(ubLoopNumS_);
    tilingData_.set_ubFactorS(ubFactorS_);
    tilingData_.set_ubTailFactorS(ubTailFactorS_);
    tilingData_.set_ubLoopNumB(ubLoopNumB_);
    tilingData_.set_ubFactorB(ubFactorB_);
    tilingData_.set_ubTailFactorB(ubTailFactorB_);
    tilingData_.set_ubLoopNumQN(ubLoopNumQN_);
    tilingData_.set_ubFactorQN(ubFactorQN_);
    tilingData_.set_ubTailFactorQN(ubTailFactorQN_);
    tilingData_.set_ubLoopNumKN(ubLoopNumKN_);
    tilingData_.set_ubFactorKN(ubFactorKN_);
    tilingData_.set_ubTailFactorKN(ubTailFactorKN_);
    tilingData_.set_realDim(reald_);
    tilingData_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingData_.set_isPartialRope(isPartialRope_);

    OP_LOGI(context_->GetNodeName(),
            "ApplyRotaryPosEmbTilingABAAndBA tiling data: B[%ld] CosB[%ld] S[%ld] D[%ld] QN[%ld] KN[%ld] "
            "blockNumB[%ld] blockFactorB[%ld] blockNumS[%ld] blockFactorS[%ld] ubLoopNumS[%ld] ubFactorS[%ld] "
            "ubTailFactorS[%ld] "
            "ubLoopNumB[%ld] ubFactorB[%ld] ubTailFactorB[%ld] ubLoopNumQN[%ld] ubFactorQN[%ld] ubTailFactorQN[%ld] "
            "ubLoopNumKN[%ld] ubFactorKN[%ld] ubTailFactorKN[%ld] rotaryMode[%ld] realDim[%ld] isPartialRope[%d]",
            tilingData_.get_B(), tilingData_.get_CosB(), tilingData_.get_S(), tilingData_.get_D(), tilingData_.get_QN(),
            tilingData_.get_KN(), tilingData_.get_blockNumB(), tilingData_.get_blockFactorB(),
            tilingData_.get_blockNumS(), tilingData_.get_blockFactorS(), tilingData_.get_ubLoopNumS(),
            tilingData_.get_ubFactorS(), tilingData_.get_ubTailFactorS(), tilingData_.get_ubLoopNumB(),
            tilingData_.get_ubFactorB(), tilingData_.get_ubTailFactorB(), tilingData_.get_ubLoopNumQN(),
            tilingData_.get_ubFactorQN(), tilingData_.get_ubTailFactorQN(), tilingData_.get_ubLoopNumKN(),
            tilingData_.get_ubFactorKN(), tilingData_.get_ubTailFactorKN(), tilingData_.get_rotaryMode(),
            tilingData_.get_realDim(), tilingData_.get_isPartialRope());
    return;
}

ge::graphStatus ApplyRotaryPosEmbTilingABAAndBA::DoOpTiling()
{
    OP_CHECK_IF(SplitCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "failed to split core."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "failed to compute ub factor."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbTilingABAAndBA::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ApplyRotaryPosEmbTilingABAAndBA::GetTilingKey() const
{
    return cosb_ == 1 ? TILING_KEY_BA : TILING_KEY_ABA;
}

ge::graphStatus ApplyRotaryPosEmbTilingABAAndBA::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbTilingABAAndBA::PostTiling()
{
    SetTilingData();
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(blockNumB_ * blockNumS_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ApplyRotaryPosEmb, ApplyRotaryPosEmbTilingABAAndBA, ROPE_ABA_AND_BA_TILING_PRIORITY);
} // namespace optiling
