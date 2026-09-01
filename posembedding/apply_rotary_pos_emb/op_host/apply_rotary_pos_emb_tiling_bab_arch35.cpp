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
 * \file apply_rotary_pos_emb_regbase_tiling_bab.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_tiling.h"
#include "log/log.h"
#include "util/math_util.h"

namespace optiling {
constexpr uint64_t ROPE_BAB_TILING_PRIORITY = 30000;
constexpr int64_t MIN_UB_LOAD_D_NUM = 4; // q/k, qout/kout 或sin, cos开doublebuffer
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr int64_t MIN_COPY_BLOCK_COUNT = 4095;
constexpr size_t WORK_SPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;
constexpr int64_t TILING_KEY_BAB = 20020;

class ApplyRotaryPosEmbTilingBAB : public ApplyRotaryPosEmbRegbaseTilingBaseClass {
public:
    explicit ApplyRotaryPosEmbTilingBAB(gert::TilingContext *context)
        : ApplyRotaryPosEmbRegbaseTilingBaseClass(context)
    {}
    ~ApplyRotaryPosEmbTilingBAB() override = default;

    void Reset(gert::TilingContext *context) override
    {
        ApplyRotaryPosEmbRegbaseTilingBaseClass::Reset(context);
    }

protected:
    // 计算数据切分
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 设置Tiling数据
    ge::graphStatus PostTiling() override;

    bool IsCapable() override
    {
        if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
            return false;
        }
        // BSND format, 1s1d模版，后续可扩展支持所有bab类型的boardcast
        if ((layout_ == ApplyRotaryPosEmbLayout::BSND) && (cosb_ == 1)) {
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
    int64_t usedCoreNum_ = 0;
    int64_t ubLoopNumS_ = 0;
    int64_t ubFactorS_ = 1;
    int64_t ubTailFactorS_ = 0;
    int64_t ubLoopNumB_ = 0;
    int64_t ubFactorB_ = 1;
    int64_t ubTailFactorB_ = 0;
    int64_t ubLoopNumQN_ = 0;
    int64_t ubFactorQN_ = 1;
    int64_t ubTailFactorQN_ = 0;
    int64_t ubLoopNumKN_ = 0;    // 核内处理KN循环了多少次
    int64_t ubFactorKN_ = 1;     // 每次循环处理多少个KN
    int64_t ubTailFactorKN_ = 0; // 最后一次循环处理多少KN
    int64_t ubSize_ = 0;
    uint64_t tilingKey_ = 0;

    void SplitCore();
    ge::graphStatus SplitUb();
    void PrintTilingData();
    ApplyRotaryPosEmbRegbaseTilingData tilingData_;
};

ge::graphStatus ApplyRotaryPosEmbTilingBAB::DoOpTiling()
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
        OP_LOGE(context_->GetNodeName(), "split coreNum [%ld] larger than coreNum [%ld]", blockNumB_ * blockNumS_,
                coreNum_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void ApplyRotaryPosEmbTilingBAB::SplitCore()
{
    // 尝试先对B分核，再尝试优先对S分核，比较二者切分之后的总核数
    auto blockFactorB1 = Ops::Base::CeilDiv(b_, coreNum_);
    auto blockNumB1 = Ops::Base::CeilDiv(b_, blockFactorB1);
    if (blockNumB1 == 0) {
        OP_LOGE(context_->GetNodeName(), "blockNumB1 can't be 0.");
        return;
    }
    auto blockNumS1 = std::min(coreNum_ / blockNumB1, s_);
    auto blockFactorS1 = Ops::Base::CeilDiv(s_, blockNumS1);
    blockNumS1 = Ops::Base::CeilDiv(s_, blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    auto blockFactorS2 = Ops::Base::CeilDiv(s_, coreNum_);
    auto blockNumS2 = Ops::Base::CeilDiv(s_, blockFactorS2);
    if (blockNumS2 == 0) {
        OP_LOGE(context_->GetNodeName(), "blockNumS2 can't be 0.");
        return;
    }
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

ge::graphStatus ApplyRotaryPosEmbTilingBAB::SplitUb()
{
    uint32_t typeSize = ge::GetSizeByDataType(dtype_);
    int64_t dAlign = Ops::Base::CeilAlign(reald_ * typeSize / dSplitCoef_, blockSize_) * dSplitCoef_;
    // interleave模式需要补pad, D对齐
    int64_t canLoadDNum = Ops::Base::FloorDiv(ubSize_, dAlign);
    if (canLoadDNum < MIN_UB_LOAD_D_NUM + MIN_UB_LOAD_D_NUM) {
        OP_LOGE(context_->GetNodeName(), "ubSize_ can't load 10 d_, d_ = %ld.", d_);
        return ge::GRAPH_FAILED;
    }
    canLoadDNum = canLoadDNum / MIN_UB_LOAD_D_NUM; // 3输入1输出开doubleBuffer
    int64_t ubLoopNum = Ops::Base::CeilDiv(qn_, (canLoadDNum - 1));
    ubFactorQN_ = std::min(Ops::Base::CeilDiv(qn_, ubLoopNum), MIN_COPY_BLOCK_COUNT / dSplitCoef_);
    ubLoopNumQN_ = Ops::Base::CeilDiv(qn_, ubFactorQN_);
    if (ubFactorQN_ == 0) {
        OP_LOGE(context_->GetNodeName(), "ubFactorQN_ can't be 0.");
        return ge::GRAPH_FAILED;
    }
    ubTailFactorQN_ = (qn_ % ubFactorQN_ == 0) ? ubFactorQN_ : qn_ % ubFactorQN_;
    ubLoopNum = Ops::Base::CeilDiv(kn_, (canLoadDNum - 1));
    ubFactorKN_ = std::min(Ops::Base::CeilDiv(kn_, ubLoopNum), MIN_COPY_BLOCK_COUNT / dSplitCoef_);
    if (ubFactorKN_ == 0) {
        OP_LOGE(context_->GetNodeName(), "ubFactorKN_ can't be 0.");
        return ge::GRAPH_FAILED;
    }
    ubLoopNumKN_ = Ops::Base::CeilDiv(kn_, ubFactorKN_);
    ubTailFactorKN_ = (kn_ % ubFactorKN_ == 0) ? ubFactorKN_ : kn_ % ubFactorKN_;
    int64_t ubFactorS = Ops::Base::FloorDiv(canLoadDNum, (std::max(qn_, kn_) + 1));
    ubFactorS_ = (ubFactorS == 0) ? 1 : ubFactorS;
    return ge::GRAPH_SUCCESS;
}

void ApplyRotaryPosEmbTilingBAB::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "ApplyRotaryPosEmbTilingBAB tilingData: useCoreNum is %ld,"
            "B is %ld, S is %ld, D is %ld, QN is %ld, KN is %ld, blockNumB %ld,"
            "blockFactorB_ is %ld, blockNumS %ld, blockFactorS is %ld, ubLoopNumS is %ld,"
            "ubFactorS is %ld, ubTailFactorS %ld, ubLoopNumB is %ld, ubFactorB is %ld,"
            "ubTailFactorB is %ld, ubLoopNumQN is %ld, ubFactorQN is %ld, ubTailFactorQN is %ld,"
            "ubLoopNumKN is %ld, ubFactorKN is %ld, ubTailFactorKN is %ld, tilingKey is %lu, realDim is %ld,"
            "isPartialRope is %d",
            usedCoreNum_, tilingData_.get_B(), tilingData_.get_S(), tilingData_.get_D(), tilingData_.get_QN(),
            tilingData_.get_KN(), tilingData_.get_blockNumB(), tilingData_.get_blockFactorB(),
            tilingData_.get_blockNumS(), tilingData_.get_blockFactorS(), tilingData_.get_ubLoopNumS(),
            tilingData_.get_ubFactorS(), tilingData_.get_ubTailFactorS(), tilingData_.get_ubLoopNumB(),
            tilingData_.get_ubFactorB(), tilingData_.get_ubTailFactorB(), tilingData_.get_ubLoopNumQN(),
            tilingData_.get_ubFactorQN(), tilingData_.get_ubTailFactorQN(), tilingData_.get_ubLoopNumKN(),
            tilingData_.get_ubFactorKN(), tilingData_.get_ubTailFactorKN(), tilingKey_, tilingData_.get_realDim(),
            tilingData_.get_isPartialRope());
    return;
}

ge::graphStatus ApplyRotaryPosEmbTilingBAB::PostTiling()
{
    tilingData_.set_B(b_);
    tilingData_.set_CosB(0);
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
    tilingData_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingData_.set_realDim(reald_);
    tilingData_.set_isPartialRope(isPartialRope_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t ApplyRotaryPosEmbTilingBAB::GetTilingKey() const
{
    return TILING_KEY_BAB;
}

REGISTER_OPS_TILING_TEMPLATE(ApplyRotaryPosEmb, ApplyRotaryPosEmbTilingBAB, ROPE_BAB_TILING_PRIORITY);
} // namespace optiling
