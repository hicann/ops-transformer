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
 * \file apply_rotary_pos_emb_regbase_tiling_ab_arch35.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_tiling.h"
#include "log/log.h"
#include "util/math_util.h"

namespace optiling {

constexpr int64_t MAX_COPY_BLOCK_COUNT = 4095;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t DB_FLAG = 2;
constexpr int64_t TILING_KEY_AB = 20030;
constexpr uint64_t ROPE_AB_TILING_PRIORITY = 10000;

class ApplyRotaryPosEmbTilingAB : public ApplyRotaryPosEmbRegbaseTilingBaseClass {
public:
    explicit ApplyRotaryPosEmbTilingAB(gert::TilingContext *context)
        : ApplyRotaryPosEmbRegbaseTilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    int64_t blockNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockTail_ = 0;
    int64_t ubLoop_ = 0;
    int64_t ubFactor_ = 0;
    int64_t ubFactorBS_ = 0;
    int64_t ubTail_ = 0;
    int64_t dAlign_ = 0;
    ApplyRotaryPosEmbRegbaseABTilingData tilingData_;
};

ge::graphStatus ApplyRotaryPosEmbTilingAB::DoOpTiling()
{
    int64_t bs = b_ * s_;
    if (layout_ == ApplyRotaryPosEmbLayout::SBND && cosb_ == 1) {
        bs = s_;
        qn_ = b_ * qn_;
        kn_ = b_ * kn_;
    }

    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    if (dSplitCoef_ == 0) {
        OP_LOGE(context_->GetNodeName(), "dSplitCoef_ can't be 0.");
        return ge::GRAPH_FAILED;
    }
    if (typeSize == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "query", Ops::Base::ToString(dtype_).c_str(),
                                              "The dtype size of input query cannot be 0");
        return ge::GRAPH_FAILED;
    }

    dAlign_ = Ops::Base::CeilAlign(reald_ / dSplitCoef_, blockSize_ / typeSize) * dSplitCoef_;
    blockFactor_ = Ops::Base::CeilDiv(bs, int64_t(aicoreParams_.numBlocks));
    blockNum_ = Ops::Base::CeilDiv(bs, blockFactor_);
    blockTail_ = bs - (blockNum_ - 1) * blockFactor_;

    int64_t total_n = qn_ + kn_;
    int64_t baseBufferSize = dAlign_ * typeSize;
    int64_t qkUbSize = aicoreParams_.ubSize / CONST_TWO / DB_FLAG;
    int64_t baseBlockInUb = Ops::Base::FloorAlign(qkUbSize, blockSize_) / baseBufferSize;
    OP_CHECK_IF(baseBlockInUb < 2, OP_LOGE(context_->GetNodeName(), "ubSize can't load 8 d size, d = %ld.", d_),
                return ge::GRAPH_FAILED);
    ubLoop_ = Ops::Base::CeilDiv(total_n, baseBlockInUb - 1);
    ubFactor_ = Ops::Base::CeilDiv(total_n, ubLoop_);
    ubFactor_ = std::min(ubFactor_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    ubTail_ = total_n - (ubLoop_ - 1) * ubFactor_;
    ubFactorBS_ = Ops::Base::FloorDiv(baseBlockInUb, total_n + 1);
    ubFactorBS_ = (ubFactorBS_ == 0) ? 1 : ubFactorBS_;
    ubFactorBS_ = std::min(ubFactorBS_, blockFactor_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbTilingAB::PostTiling()
{
    tilingData_.set_B(b_);
    tilingData_.set_CosB(cosb_);
    tilingData_.set_S(s_);
    tilingData_.set_D(d_);
    tilingData_.set_QN(qn_);
    tilingData_.set_KN(kn_);
    tilingData_.set_dAlign(dAlign_);
    tilingData_.set_dSplitCoef(dSplitCoef_);
    tilingData_.set_blockNumBS(blockNum_);
    tilingData_.set_blockFactorBS(blockFactor_);
    tilingData_.set_blockTailBS(blockTail_);
    tilingData_.set_ubFactorBS(ubFactorBS_);
    tilingData_.set_ubLoopN(ubLoop_);
    tilingData_.set_ubFactorN(ubFactor_);
    tilingData_.set_ubTailN(ubTail_);
    tilingData_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingData_.set_realDim(reald_);
    tilingData_.set_isPartialRope(isPartialRope_);

    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    OP_LOGI(context_->GetNodeName(),
            "ApplyRotaryPosEmbAB tilingData is B: %ld, CosB: %ld, S: %ld, D: %ld, QN: %ld, KN: %ld, kAlign: %ld, "
            "dSplitCoef: %ld, BlockNum: %ld, BlockFactor: %ld, BlockTail: %ld, ubFactorBS: %ld, UBLoop: %ld, UBFactor: "
            "%ld, "
            "UBTail: %ld, RotaryMode: %ld, TilingKey: %lu, isPartialRope: %ld.",
            tilingData_.get_B(), tilingData_.get_CosB(), tilingData_.get_S(), tilingData_.get_D(), tilingData_.get_QN(),
            tilingData_.get_KN(), tilingData_.get_dAlign(), tilingData_.get_dSplitCoef(), tilingData_.get_blockNumBS(),
            tilingData_.get_blockFactorBS(), tilingData_.get_blockTailBS(), tilingData_.get_ubFactorBS(),
            tilingData_.get_ubLoopN(), tilingData_.get_ubFactorN(), tilingData_.get_ubTailN(),
            tilingData_.get_rotaryMode(), GetTilingKey(), tilingData_.get_isPartialRope());

    return ge::GRAPH_SUCCESS;
}

uint64_t ApplyRotaryPosEmbTilingAB::GetTilingKey() const
{
    return TILING_KEY_AB;
}

bool ApplyRotaryPosEmbTilingAB::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }

    // 1. qk:bsnd, cos:bs1d 2. qk:sbnd, cos:sb1d 3. qk:sbnd, cos:s11d 4. qk:tnd, cos:t1d
    if ((layout_ == ApplyRotaryPosEmbLayout::BSND && cosb_ == b_) ||
        (layout_ == ApplyRotaryPosEmbLayout::SBND && cosb_ == b_) ||
        (layout_ == ApplyRotaryPosEmbLayout::SBND && cosb_ == 1) ||
        (layout_ == ApplyRotaryPosEmbLayout::TND && cosb_ == b_ && cosb_ == 1)) {
        return true;
    }
    return false;
}

REGISTER_OPS_TILING_TEMPLATE(ApplyRotaryPosEmb, ApplyRotaryPosEmbTilingAB, ROPE_AB_TILING_PRIORITY);

} // namespace optiling
