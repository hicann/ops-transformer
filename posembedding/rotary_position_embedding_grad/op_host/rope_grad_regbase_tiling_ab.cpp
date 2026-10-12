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
 * \file rope_grad_regbase_tiling_ab.cc
 * \brief
 */
#include "op_host/tiling_templates_registry.h"
#include "rotary_position_embedding_grad_tiling.h"

namespace optiling {
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t DB_FLAG = 2;

class RopeGradRegBaseTilingClassAB : public RopeGradRegBaseTilingClass {
public:
    explicit RopeGradRegBaseTilingClassAB(gert::TilingContext* context)
        : RopeGradRegBaseTilingClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

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
};

ge::graphStatus RopeGradRegBaseTilingClassAB::DoOpTiling()
{
    int64_t bs = b_ * s_;
    bn_ = n_;
    if (cosb_ == 1) {
        bs = s_;
        bn_ = b_ * n_;
    }
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    dAlign_ = Ops::Base::CeilAlign(d_ / dSplitCoef_, blockSize_ / typeSize) * dSplitCoef_;

    blockFactorBS_ = Ops::Base::CeilDiv(bs, int64_t(aicoreParams_.numBlocks));
    blockNumBS_ = Ops::Base::CeilDiv(bs, blockFactorBS_);
    blockTailBS_ = bs - (blockNumBS_ - 1) * blockFactorBS_;

    if (bs <= int64_t(aicoreParams_.numBlocks) / CONST_TWO) {
        blockNumN_ = aicoreParams_.numBlocks / blockNumBS_;
        blockFactorN_ = Ops::Base::CeilDiv(bn_, blockNumN_);
        blockNumN_ = Ops::Base::CeilDiv(bn_, blockFactorN_);
        blockTailN_ = bn_ - (blockNumN_ - 1) * blockFactorN_;
    } else {
        blockNumN_ = 1;
        blockFactorN_ = bn_;
        blockTailN_ = bn_;
    }
    usedCoreNum_ = blockNumBS_ * blockNumN_;
    int64_t baseBufferSize = dAlign_ * typeSize;
    int64_t baseBlockInUb =
        Ops::Base::FloorAlign(static_cast<int64_t>(aicoreParams_.ubSize / CONST_TWO / DB_FLAG), blockSize_) /
        baseBufferSize;
    OP_CHECK_IF(baseBlockInUb < 1, OP_LOGE(context_->GetNodeName(), "ubSize can't load 8 d size, d = %ld.", d_),
                return ge::GRAPH_FAILED);

    ubFactorN_ = std::min(blockFactorN_, baseBlockInUb - 1);
    ubFactorN_ = std::min(ubFactorN_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);

    ubFactorBS_ = std::min(Ops::Base::FloorDiv(baseBlockInUb, ubFactorN_ + 1), blockFactorBS_);
    ubFactorBS_ = (ubFactorBS_ == 0) ? 1 : ubFactorBS_;
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "InitTilingData failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingReduce failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeGradRegBaseTilingClassAB::PostTiling()
{
    tilingData_->ropeGradABParams.b = b_;
    tilingData_->ropeGradABParams.s = s_;
    tilingData_->ropeGradABParams.d = d_;
    tilingData_->ropeGradABParams.n = bn_;
    tilingData_->ropeGradABParams.dAlign = dAlign_;
    tilingData_->ropeGradABParams.dSplitCoef = dSplitCoef_;
    tilingData_->ropeGradABParams.blockNumBS = blockNumBS_;
    tilingData_->ropeGradABParams.blockFactorBS = blockFactorBS_;
    tilingData_->ropeGradABParams.blockTailBS = blockTailBS_;
    tilingData_->ropeGradABParams.blockNumN = blockNumN_;
    tilingData_->ropeGradABParams.blockFactorN = blockFactorN_;
    tilingData_->ropeGradABParams.blockTailN = blockTailN_;
    tilingData_->ropeGradABParams.ubFactorBS = ubFactorBS_;
    tilingData_->ropeGradABParams.ubFactorN = ubFactorN_;
    tilingData_->ropeGradABParams.usedCoreNum = usedCoreNum_;
    tilingData_->ropeGradABParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->dCosFlag = dCosFlag_;
    SetRotaryXTilingData();
    SetTilingKeyBlockDim(static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_AB));

    OP_LOGI(context_->GetNodeName(),
            "RopeGradRegBaseTilingClassAB tilingData is B: %ld, S: %ld, N: %ld, D: %ld, kAlign: %ld, "
            "dSplitCoef: %ld, BlockNumBS: %ld, BlockFactorBS: %ld, BlockTailBS: %ld, BlockNumN: %ld, "
            "BlockFactorN: %ld, blockTailN: %ld, ubFactorBS: %ld, ubFactorN: %ld, "
            "usedCoreNum_: %ld, RotaryMode: %ld, TilingKey: %ld.",
            tilingData_->ropeGradABParams.b, tilingData_->ropeGradABParams.s, tilingData_->ropeGradABParams.n,
            tilingData_->ropeGradABParams.d, tilingData_->ropeGradABParams.dAlign,
            tilingData_->ropeGradABParams.dSplitCoef, tilingData_->ropeGradABParams.blockNumBS,
            tilingData_->ropeGradABParams.blockFactorBS, tilingData_->ropeGradABParams.blockTailBS,
            tilingData_->ropeGradABParams.blockNumN, tilingData_->ropeGradABParams.blockFactorN,
            tilingData_->ropeGradABParams.blockTailN, tilingData_->ropeGradABParams.ubFactorBS,
            tilingData_->ropeGradABParams.ubFactorN, tilingData_->ropeGradABParams.usedCoreNum,
            tilingData_->ropeGradABParams.rotaryMode, tilingKey_);
    return ge::GRAPH_SUCCESS;
}

bool RopeGradRegBaseTilingClassAB::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }

    OP_LOGI(context_->GetNodeName(), "layout: %ld", static_cast<int64_t>(layout_));
    // 1. qk:bsnd, cos:bs1d 2. qk:sbnd, cos:sb1d 3. qk:sbnd, cos:s11d
    return layout_ == RopeLayout::SBND;
}

REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbeddingGrad, RopeGradRegBaseTilingClassAB, 25000);
} // namespace optiling
