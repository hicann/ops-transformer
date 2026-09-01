/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rope_grad_regbase_tiling_bab.cc
 * \brief
 */

#include "rotary_position_embedding_grad_tiling.h"

namespace optiling {
constexpr uint64_t ROPE_GRAD_BAB_TILING_PRIORITY = 20000;
constexpr uint32_t MIN_UB_LOAD_D_NUM = 4; // x, y或in, cos输入开doubleBuffer
constexpr uint32_t DOUBLE_BUFFER = 2;

class RopeGradRegBaseTilingClassBAB : public RopeGradRegBaseTilingClass {
public:
    explicit RopeGradRegBaseTilingClassBAB(gert::TilingContext *context_)
        : RopeGradRegBaseTilingClass(context_)
    {}
    ~RopeGradRegBaseTilingClassBAB() override = default;
    void Reset(gert::TilingContext *context_) override
    {
        RopeGradRegBaseTilingClass::Reset(context_);
    }

protected:
    // 计算数据切分
    ge::graphStatus DoOpTiling() override;
    // 设置Tiling数据
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        // BSND format, 1s1d模版，后续可扩展支持所有bab类型的boardcast
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) && (layout_ == RopeLayout::BSND) &&
            (cosb_ == 1)) {
            return true;
        }
        return false;
    }

private:
    int64_t coreNum_ = 0;
    int64_t blockNumB_ = 0;
    int64_t blockFactorB_ = 0;
    int64_t blockNumS_ = 0;
    int64_t blockFactorS_ = 0;
    int64_t ubFactorS_ = 1;
    int64_t ubFactorB_ = 1;
    int64_t ubLoopNumN_ = 0;    // 核内处理N循环了多少次
    int64_t ubFactorN_ = 1;     // 每次循环处理多少个N
    int64_t ubTailFactorN_ = 0; // 最后一次循环处理多少N
    int64_t ubSize_ = 0;

    void SplitCore();
    ge::graphStatus SplitUb();
    void PrintTilingData();
};

ge::graphStatus RopeGradRegBaseTilingClassBAB::DoOpTiling()
{
    ubSize_ = aicoreParams_.ubSize;
    coreNum_ = aicoreParams_.numBlocks;
    ge::graphStatus status = SplitUb();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SplitUb Failed.");
        return ge::GRAPH_FAILED;
    }
    SplitCore();
    if (blockNumB_ * blockNumS_ > coreNum_) {
        OP_LOGE(
            context_->GetNodeName(), "split coreNum [%ld] larger than coreNum [%ld]",
            blockNumB_ * blockNumS_, coreNum_);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "InitTilingData failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingReduce failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void RopeGradRegBaseTilingClassBAB::SplitCore()
{
    // 尝试先对B分核，再尝试优先对S分核，比较二者切分之后的总核数
    auto blockFactorB1 = Ops::Base::CeilDiv(b_, coreNum_);
    auto blockNumB1 = Ops::Base::CeilDiv(b_, blockFactorB1);
    auto blockNumS1 = std::min(coreNum_ / blockNumB1, s_);
    auto blockFactorS1 = Ops::Base::CeilDiv(s_, blockNumS1);
    blockNumS1 = Ops::Base::CeilDiv(s_, blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    auto blockFactorS2 = Ops::Base::CeilDiv(s_, coreNum_);
    auto blockNumS2 = Ops::Base::CeilDiv(s_, blockFactorS2);
    auto blockNumB2 = std::min(coreNum_ / blockNumS2, b_);
    auto blockFactorB2 = Ops::Base::CeilDiv(b_, blockNumB2);
    blockNumB2 = Ops::Base::CeilDiv(b_, blockFactorB2);
    auto usedCoreNum2 = blockNumB2 * blockNumS2;

    // ubFactorS 很小的时候，选择核数多的，ubFactorS大于blockFactorS时， 综合考虑分核和UB切分
    auto ubFactorS1 = std::min(ubFactorS_, blockFactorS1);
    auto ubFactorS2 = std::min(ubFactorS_, blockFactorS2);
    if (usedCoreNum1 * ubFactorS1 >= usedCoreNum2 * ubFactorS2) {
        blockNumB_ = blockNumB1;
        blockFactorB_ = blockFactorB1;
        blockNumS_ = blockNumS1;
        blockFactorS_ = blockFactorS1;
        usedCoreNum_ = usedCoreNum1;
        ubFactorS_ = ubFactorS1;
    } else {
        blockNumB_ = blockNumB2;
        blockFactorB_ = blockFactorB2;
        blockNumS_ = blockNumS2;
        blockFactorS_ = blockFactorS2;
        usedCoreNum_ = usedCoreNum2;
        ubFactorS_ = ubFactorS2;
    }
    return;
}

ge::graphStatus RopeGradRegBaseTilingClassBAB::SplitUb()
{
    uint32_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t dAlign = Ops::Base::CeilAlign(d_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    // interleave模式需要补pad, D对齐
    int64_t canLoadDNum = Ops::Base::FloorDiv(ubSize_, dAlign);
    if (canLoadDNum < MIN_UB_LOAD_D_NUM + MIN_UB_LOAD_D_NUM) {
        OP_LOGE(context_->GetNodeName(), "ubSize_ can't load 10 d_, d_ = %ld.", d_);
        return ge::GRAPH_FAILED;
    }
    canLoadDNum = canLoadDNum / MIN_UB_LOAD_D_NUM;
    int64_t ubLoopNum = Ops::Base::CeilDiv(n_, (canLoadDNum - 1)); // 这里减去sin, cos 开doubleBuffer
    ubFactorN_ = std::min(Ops::Base::CeilDiv(n_, ubLoopNum), MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    ubLoopNumN_ = Ops::Base::CeilDiv(n_, ubFactorN_);
    ubTailFactorN_ = (n_ % ubFactorN_ == 0) ? ubFactorN_ : n_ % ubFactorN_;
    int64_t ubFactorS = Ops::Base::FloorDiv(canLoadDNum, n_ + 1);
    ubFactorS_ = (ubFactorS == 0) ? 1 : ubFactorS;
    return ge::GRAPH_SUCCESS;
}

void RopeGradRegBaseTilingClassBAB::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "RopeGradRegBaseTilingBAB tilingData: useCoreNum is %ld,"
            "B is %ld, S is %ld, D is %ld, N is %ld, blockNumB %ld,"
            "blockFactorB_ is %ld, blockNumS %ld, blockFactorS is %ld, ubFactorS is %ld, ubFactorB is %ld,"
            "ubLoopNumN is %ld, ubFactorN is %ld, ubTailFactorN is %ld, rotaryMode is %ld, tilingKey is %ld",
            usedCoreNum_, tilingData_->ropeGradParams.b, tilingData_->ropeGradParams.s, tilingData_->ropeGradParams.d,
            tilingData_->ropeGradParams.n, tilingData_->ropeGradParams.blockNumB,
            tilingData_->ropeGradParams.blockFactorB, tilingData_->ropeGradParams.blockNumS,
            tilingData_->ropeGradParams.blockFactorS, tilingData_->ropeGradParams.ubFactorS,
            tilingData_->ropeGradParams.ubFactorB, tilingData_->ropeGradParams.ubLoopNumN,
            tilingData_->ropeGradParams.ubFactorN, tilingData_->ropeGradParams.ubTailFactorN,
            tilingData_->ropeGradParams.rotaryMode, tilingKey_);
    return;
}

ge::graphStatus RopeGradRegBaseTilingClassBAB::PostTiling()
{
    tilingData_->ropeGradParams.b = b_;
    tilingData_->ropeGradParams.s = s_;
    tilingData_->ropeGradParams.d = d_;
    tilingData_->ropeGradParams.n = n_;
    tilingData_->ropeGradParams.ubFactorS = ubFactorS_;
    tilingData_->ropeGradParams.ubFactorB = ubFactorB_;
    tilingData_->ropeGradParams.ubLoopNumN = ubLoopNumN_;
    tilingData_->ropeGradParams.ubFactorN = ubFactorN_;
    tilingData_->ropeGradParams.ubTailFactorN = ubTailFactorN_;
    tilingData_->ropeGradParams.blockNumB = blockNumB_;
    tilingData_->ropeGradParams.blockFactorB = blockFactorB_;
    tilingData_->ropeGradParams.blockNumS = blockNumS_;
    tilingData_->ropeGradParams.blockFactorS = blockFactorS_;
    tilingData_->ropeGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->ropeGradParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->dCosFlag = dCosFlag_;
    SetRotaryXTilingData();
    SetTilingKeyBlockDim(static_cast<uint32_t>(ropeGradTilingKey::TILING_KEY_BAB));
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbeddingGrad, RopeGradRegBaseTilingClassBAB, ROPE_GRAD_BAB_TILING_PRIORITY);
} // namespace optiling
