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
 * \file apply_rotary_pos_emb_grad_tiling_ab.cpp
 * \brief
 *
 * 核间: BS 合并轴 × N 轴 二维分核
 * 核内: 外层 BS 循环搬 cos/sin(被 Q/K 双路复用), 内层 N 循环搬 grad、算 dx
 * cos/sin 的 B 轴可为 1(广播, B 合并进 N) 或与输入一致(每 (b,s) 独立)
 */
#include "apply_rotary_pos_emb_grad_tiling.h"
#include <algorithm>

namespace optiling {
constexpr uint64_t APPLY_ROPE_GRAD_AB_TILING_PRIORITY = 30000;
constexpr int64_t AB_DOUBLE_BUFFER = 2;
constexpr int64_t AB_CONST_TWO = 2;
constexpr int64_t AB_UB_DIVISOR_WITH_DCOS = 16;
constexpr int64_t AB_UB_DIVISOR_WITH_DCOS_FLOAT_PARTIAL = 20;
constexpr int64_t AB_UB_DIVISOR_WITHOUT_DCOS = 8;

class ApplyRotaryPosEmbGradRegbaseTilingClassAB : public ApplyRotaryPosEmbGradRegbaseTilingClass {
public:
    explicit ApplyRotaryPosEmbGradRegbaseTilingClassAB(gert::TilingContext *context)
        : ApplyRotaryPosEmbGradRegbaseTilingClass(context)
    {}

protected:
    bool IsCapable() override
    {
        // AB 模板适用条件:
        // 1. Regbase soc (ascend950)
        // 2. SBND 内部 layout — 覆盖 attr=1(BSND) 且 cosb>1 的场景, 以及 attr=2(SBND)
        //    (cos/sin 的 N 维为 1, 在 N 方向天然广播, 故 N 轴可独立切分并行)
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_) && (layout_ == ApplyRopeGradLayout::SBND);
    }
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus SplitCore();
    ge::graphStatus SplitUb();
    void PrintTilingData();

    int64_t coreNum_{0};
    int64_t ubSize_{0};
    int64_t dAlign_{0};
    int64_t bs_{0};      // 合并后的 BS 数 (cosb>1: b*s; cosb==1: s)
    int64_t bn_{0};      // 合并后的 N 数 = max(totalNQ, totalNK)
    int64_t totalNQ_{0}; // Q 路有效 N (cosb>1: nQ; cosb==1: b*nQ)
    int64_t totalNK_{0}; // K 路有效 N (cosb>1: nK; cosb==1: b*nK)
    int64_t blockNumBS_{0};
    int64_t blockFactorBS_{0};
    int64_t blockTailBS_{0};
    int64_t blockNumN_{0};
    int64_t blockFactorN_{0};
    int64_t blockTailN_{0};
    int64_t ubFactorBS_{0};
    int64_t ubFactorN_{0};
};

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassAB::SplitUb()
{
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    dAlign_ = Ops::Base::CeilAlign(d_ / dSplitCoef_, blockSize_ / typeSize) * dSplitCoef_;
    int64_t bufferSize = dAlign_ * typeSize; // 一行 dAlign 元素的字节数
    int64_t maxCopyN = MAX_COPY_BLOCK_COUNT / dSplitCoef_;

    int64_t divisor = AB_UB_DIVISOR_WITHOUT_DCOS;
    if (dCosFlag_ == 1) {
        divisor = (typeSize == static_cast<int64_t>(sizeof(float))) ? AB_UB_DIVISOR_WITH_DCOS :
                                                                      AB_UB_DIVISOR_WITH_DCOS_FLOAT_PARTIAL;
    }
    int64_t baseBlockInUb = Ops::Base::FloorAlign(ubSize_ / divisor, blockSize_) / bufferSize;
    OP_CHECK_IF(baseBlockInUb < 1,
                OP_LOGE(context_->GetNodeName(), "AB SplitUb baseBlockInUb < 1, d_=%ld, dAlign=%ld, ubSize=%ld", d_,
                        dAlign_, ubSize_),
                return ge::GRAPH_FAILED);

    // 先定 N: 留 1 行预算给 cos/sin
    ubFactorN_ = std::min(blockFactorN_, baseBlockInUb - 1);
    ubFactorN_ = std::min(ubFactorN_, maxCopyN); // DataCopy blockCount 上限
    OP_CHECK_IF(ubFactorN_ < 1,
                OP_LOGE(context_->GetNodeName(), "AB SplitUb ubFactorN < 1, baseBlockInUb=%ld", baseBlockInUb),
                return ge::GRAPH_FAILED);
    // 再定 BS: 约束 ubFactorBS * (ubFactorN + 1) <= baseBlockInUb (cos/sin 占 +1)
    ubFactorBS_ = std::min(Ops::Base::FloorDiv(baseBlockInUb, ubFactorN_ + 1), blockFactorBS_);
    ubFactorBS_ = (ubFactorBS_ == 0) ? 1 : ubFactorBS_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassAB::SplitCore()
{
    int64_t numBlocks = static_cast<int64_t>(aicoreParams_.numBlocks);
    // BS 维切分: 每核至少 blockFactorBS 个 bs
    blockFactorBS_ = Ops::Base::CeilDiv(bs_, numBlocks);
    blockNumBS_ = Ops::Base::CeilDiv(bs_, blockFactorBS_);
    blockTailBS_ = bs_ - (blockNumBS_ - 1) * blockFactorBS_;

    // N 维切分: 仅当 BS 较稀疏(<= numBlocks/2)时才把剩余核分给 N
    if (bs_ <= numBlocks / AB_CONST_TWO) {
        blockNumN_ = numBlocks / blockNumBS_;
        blockFactorN_ = Ops::Base::CeilDiv(bn_, blockNumN_);
        blockNumN_ = Ops::Base::CeilDiv(bn_, blockFactorN_);
        blockTailN_ = bn_ - (blockNumN_ - 1) * blockFactorN_;
    } else {
        blockNumN_ = 1;
        blockFactorN_ = bn_;
        blockTailN_ = bn_;
    }
    usedCoreNum_ = blockNumBS_ * blockNumN_;
    if (usedCoreNum_ > numBlocks) {
        OP_LOGE(context_->GetNodeName(), "AB SplitCore usedCoreNum [%ld] > numBlocks [%ld]", usedCoreNum_, numBlocks);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassAB::DoOpTiling()
{
    ubSize_ = aicoreParams_.ubSize;
    coreNum_ = static_cast<int64_t>(aicoreParams_.numBlocks);

    // 统一 BS / N 维度:
    //   cosb>1: cos 每 (b,s) 独立一行 → bs=b*s, N=nQ/nK
    //   cosb==1: cos 沿 B 广播 → bs=s, B 合并进 N → N=b*nQ/b*nK
    if (cosb_ == 1) {
        bs_ = s_;
        totalNQ_ = b_ * nQ_;
        totalNK_ = b_ * nK_;
    } else {
        bs_ = b_ * s_;
        totalNQ_ = nQ_;
        totalNK_ = nK_;
    }
    bn_ = std::max(totalNQ_, totalNK_);

    OP_CHECK_IF(SplitCore() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "AB SplitCore failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SplitUb() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "AB SplitUb failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "AB InitTilingData failed"),
                return ge::GRAPH_FAILED);
    reduceInputFloat_ = true;
    OP_CHECK_IF(TilingReduce() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "AB TilingReduce failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ApplyRotaryPosEmbGradRegbaseTilingClassAB::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "ApplyRotaryPosEmbGrad AB tilingData: usedCoreNum=%ld, b=%ld, s=%ld, d=%ld, nQ=%ld(raw), nK=%ld(raw), "
            "cosb=%ld, bs=%ld, bn=%ld, totalNQ=%ld, totalNK=%ld, dAlign=%ld, "
            "blockNumBS=%ld, blockFactorBS=%ld, blockTailBS=%ld, "
            "blockNumN=%ld, blockFactorN=%ld, blockTailN=%ld, "
            "ubFactorBS=%ld, ubFactorN=%ld, rotaryMode=%ld, dCosFlag=%u, tilingKey=%lu",
            usedCoreNum_, b_, s_, d_, nQ_, nK_, cosb_, bs_, bn_, totalNQ_, totalNK_, dAlign_, blockNumBS_,
            blockFactorBS_, blockTailBS_, blockNumN_, blockFactorN_, blockTailN_, ubFactorBS_, ubFactorN_,
            static_cast<int64_t>(rotaryMode_), dCosFlag_, tilingKey_);
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClassAB::PostTiling()
{
    tilingData_->ropeGradABParams.b = b_;
    tilingData_->ropeGradABParams.s = s_;
    tilingData_->ropeGradABParams.d = d_;
    tilingData_->ropeGradABParams.nQ = totalNQ_;
    tilingData_->ropeGradABParams.nK = totalNK_;
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

    tilingData_->ropeGradParams.b = b_;
    tilingData_->ropeGradParams.s = s_;
    tilingData_->ropeGradParams.d = d_;
    tilingData_->ropeGradParams.nQ = nQ_;
    tilingData_->ropeGradParams.nK = nK_;

    tilingData_->dCosFlag = dCosFlag_;
    tilingData_->layout = static_cast<uint32_t>(layout_);
    SetTilingKeyBlockDim(static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_AB));
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ApplyRotaryPosEmbGrad, ApplyRotaryPosEmbGradRegbaseTilingClassAB,
                             APPLY_ROPE_GRAD_AB_TILING_PRIORITY);
} // namespace optiling
