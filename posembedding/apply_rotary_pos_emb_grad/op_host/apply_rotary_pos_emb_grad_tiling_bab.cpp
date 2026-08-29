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
 * \file apply_rotary_pos_emb_grad_tiling_bab.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_grad_tiling.h"
#include <algorithm>

namespace optiling {
constexpr uint64_t APPLY_ROPE_GRAD_BAB_TILING_PRIORITY = 20000;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t MIN_UB_LOAD_D_NUM_DCOS0 = 8;
constexpr uint32_t MIN_UB_LOAD_D_NUM_DCOS1 = 16;
constexpr uint32_t MIN_UB_LOAD_D_NUM_DCOS1_FLOAT_PARTIAL = 20;

class ApplyRotaryPosEmbGradRegbaseTilingClassBAB : public ApplyRotaryPosEmbGradRegbaseTilingClass {
public:
    explicit ApplyRotaryPosEmbGradRegbaseTilingClassBAB(gert::TilingContext *context_)
        : ApplyRotaryPosEmbGradRegbaseTilingClass(context_)
    {}
    ~ApplyRotaryPosEmbGradRegbaseTilingClassBAB() override = default;
    void Reset(gert::TilingContext *context_) override
    {
        ApplyRotaryPosEmbGradRegbaseTilingClass::Reset(context_);
    }

protected:
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        // BAB 模板适用条件:
        // 1. Regbase soc (ascend950)
        // 2. BSND layout 且 cosb == 1 (1S1D广播模式)
        //    TND 已退化为 B=1 的 BSND: N 轴>1 时 layout=BSND 进入本模板;
        //    shape 完全一致 (B=1, N=1) 时 layout=NO_BROADCAST 走 A 模板, 不在此列
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) && layout_ == ApplyRopeGradLayout::BSND &&
               cosb_ == 1;
    }

private:
    int64_t coreNum_ = 0;
    int64_t blockNumB_ = 0;
    int64_t blockFactorB_ = 0;
    int64_t blockNumS_ = 0;
    int64_t blockFactorS_ = 0;
    int64_t ubFactorS_ = 1;
    int64_t ubLoopNumN_ = 0;    // N 轴 UB 循环次数
    int64_t ubFactorN_ = 1;     // 每次循环处理的 N 数
    int64_t ubTailFactorN_ = 0; // 最后一次循环处理的 N 数
    int64_t ubSize_ = 0;
    int64_t maxN_ = 0; // max(nQ, nK)

    void SplitCore();
    ge::graphStatus SplitUb();
    void PrintTilingData();
};

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassBAB::DoOpTiling()
{
    ubSize_ = aicoreParams_.ubSize;
    coreNum_ = aicoreParams_.numBlocks;
    maxN_ = std::max(nQ_, nK_);
    ge::graphStatus status = SplitUb();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SplitUb Failed.");
        return ge::GRAPH_FAILED;
    }
    SplitCore();
    if (blockNumB_ * blockNumS_ > coreNum_) {
        OP_LOGE(context_->GetNodeName(), "split coreNum [%ld] large than coreNum[%ld]", blockNumB_ * blockNumS_,
                coreNum_);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "InitTilingData failed."),
                return ge::GRAPH_FAILED);
    reduceInputFloat_ = true;
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "TilingReduce failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassBAB::SplitUb()
{
    uint32_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t dAlign = Ops::Base::CeilAlign(d_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    // dCosFlag=0: gradQueryEmb+gradKeyEmb+gradQ+gradK = 4路 tensor, double buffer → ×8
    // dCosFlag=1: +query+key+gradCos+gradSin = 8路 tensor, double buffer → ×16
    // 非 float 输入下 gradCos/gradSin partial 按 float 申请, 需要额外 UB 预算
    uint32_t minUbLoadDNum = MIN_UB_LOAD_D_NUM_DCOS0;
    if (dCosFlag_ == 1) {
        minUbLoadDNum = (typeSize == sizeof(float)) ? MIN_UB_LOAD_D_NUM_DCOS1 : MIN_UB_LOAD_D_NUM_DCOS1_FLOAT_PARTIAL;
    }
    int64_t canLoadDNum = Ops::Base::FloorDiv(ubSize_, dAlign);
    if (canLoadDNum < static_cast<int64_t>(minUbLoadDNum)) {
        OP_LOGE(context_->GetNodeName(),
                "ubSize_ can't load minimum D lines, d_ = %ld, dAlign = %ld, canLoadDNum = %ld, minUbLoadDNum = %u", d_,
                dAlign, canLoadDNum, minUbLoadDNum);
        return ge::GRAPH_FAILED;
    }
    canLoadDNum = canLoadDNum / minUbLoadDNum;

    // 假设 S=1 (最坏情况)，确定 N 维度的切分
    // canLoadDNum 表示 S × (N + 1) 的上限, N ≤ canLoadDNum - 1
    int64_t ubLoopNum = Ops::Base::CeilDiv(maxN_, (canLoadDNum - 1));
    ubFactorN_ = Ops::Base::CeilDiv(maxN_, ubLoopNum);
    ubLoopNumN_ = Ops::Base::CeilDiv(maxN_, ubFactorN_);
    ubTailFactorN_ = (maxN_ % ubFactorN_ == 0) ? ubFactorN_ : maxN_ % ubFactorN_;

    // 用完整的 N 保守反算 S 能放多少
    int64_t ubFactorS = Ops::Base::FloorDiv(canLoadDNum, maxN_ + 1);
    ubFactorS_ = (ubFactorS == 0) ? 1 : ubFactorS;
    return ge::GRAPH_SUCCESS;
}

void ApplyRotaryPosEmbGradRegbaseTilingClassBAB::SplitCore()
{
    // 方案1: B 优先铺满核
    auto blockFactorB1 = Ops::Base::CeilDiv(b_, coreNum_);
    auto blockNumB1 = Ops::Base::CeilDiv(b_, blockFactorB1);
    auto blockNumS1 = std::min(coreNum_ / blockNumB1, s_);
    auto blockFactorS1 = Ops::Base::CeilDiv(s_, blockNumS1);
    blockNumS1 = Ops::Base::CeilDiv(s_, blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    // 方案2: S 优先铺满核
    auto blockFactorS2 = Ops::Base::CeilDiv(s_, coreNum_);
    auto blockNumS2 = Ops::Base::CeilDiv(s_, blockFactorS2);
    auto blockNumB2 = std::min(coreNum_ / blockNumS2, b_);
    auto blockFactorB2 = Ops::Base::CeilDiv(b_, blockNumB2);
    blockNumB2 = Ops::Base::CeilDiv(b_, blockFactorB2);
    auto usedCoreNum2 = blockNumB2 * blockNumS2;

    // 选优: usedCoreNum × ubFactorS 最大者为优
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

void ApplyRotaryPosEmbGradRegbaseTilingClassBAB::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "ApplyRotaryPosEmbGrad BAB tilingData: useCoreNum is %ld, "
            "B is %ld, S is %ld, D is %ld, nQ is %ld, nK is %ld, maxN is %ld, "
            "blockNumB %ld, blockFactorB is %ld, blockNumS %ld, blockFactorS is %ld, "
            "ubFactorS is %ld, ubLoopNumN is %ld, ubFactorN is %ld, ubTailFactorN is %ld, "
            "rotaryMode is %ld, dCosFlag is %u, tilingKey is %lu",
            usedCoreNum_, tilingData_->ropeGradParams.b, tilingData_->ropeGradParams.s, tilingData_->ropeGradParams.d,
            tilingData_->ropeGradParams.nQ, tilingData_->ropeGradParams.nK, maxN_,
            tilingData_->ropeGradParams.blockNumB, tilingData_->ropeGradParams.blockFactorB,
            tilingData_->ropeGradParams.blockNumS, tilingData_->ropeGradParams.blockFactorS,
            tilingData_->ropeGradParams.ubFactorS, tilingData_->ropeGradParams.ubLoopNumN,
            tilingData_->ropeGradParams.ubFactorN, tilingData_->ropeGradParams.ubTailFactorN,
            tilingData_->ropeGradParams.rotaryMode, tilingData_->dCosFlag, tilingKey_);
    return;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassBAB::PostTiling()
{
    tilingData_->ropeGradParams.b = b_;
    tilingData_->ropeGradParams.s = s_;
    tilingData_->ropeGradParams.d = d_;
    tilingData_->ropeGradParams.nQ = nQ_;
    tilingData_->ropeGradParams.nK = nK_;
    tilingData_->ropeGradParams.ubFactorS = ubFactorS_;
    tilingData_->ropeGradParams.ubLoopNumN = ubLoopNumN_;
    tilingData_->ropeGradParams.ubFactorN = ubFactorN_;
    tilingData_->ropeGradParams.ubTailFactorN = ubTailFactorN_;
    tilingData_->ropeGradParams.blockNumB = blockNumB_;
    tilingData_->ropeGradParams.blockFactorB = blockFactorB_;
    tilingData_->ropeGradParams.blockNumS = blockNumS_;
    tilingData_->ropeGradParams.blockFactorS = blockFactorS_;
    tilingData_->ropeGradParams.usedCoreNum = usedCoreNum_;
    tilingData_->ropeGradParams.rotaryMode = static_cast<int64_t>(rotaryMode_);
    tilingData_->ropeGradParams.layout = static_cast<int64_t>(layout_);
    tilingData_->ropeGradParams.dCosFlag = dCosFlag_;
    tilingData_->dCosFlag = dCosFlag_;
    tilingData_->layout = static_cast<uint32_t>(layout_);
    SetTilingKeyBlockDim(static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_BAB));
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ApplyRotaryPosEmbGrad, ApplyRotaryPosEmbGradRegbaseTilingClassBAB,
                             APPLY_ROPE_GRAD_BAB_TILING_PRIORITY);
} // namespace optiling
