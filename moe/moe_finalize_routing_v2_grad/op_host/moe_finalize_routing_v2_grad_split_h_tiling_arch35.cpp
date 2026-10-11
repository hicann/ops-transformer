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
 * \file moe_finalize_routing_v2_grad_split_h_tiling_arch35.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_grad_tiling.h"
#include "op_host/tiling_util.h"

namespace optiling {
constexpr int64_t TILING_KEY_WITH_SCALE_CUT_H_WITHOUT_BIAS = 20012;
constexpr int64_t TILING_KEY_WITH_SCALE_CUT_H_WITH_BIAS = 20022;
constexpr int64_t TWO = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BINARY_ADD_CACHE_BUFFER_SIZE = 3 * 1024;
constexpr int64_t BINARY_ADD_BUFFER_SIZE = 1024;

class MoeFinalizeRoutingV2GradRegbaseSplitH : public MoeFinalizeRoutingV2GradTiling {
public:
    explicit MoeFinalizeRoutingV2GradRegbaseSplitH(gert::TilingContext* context)
        : MoeFinalizeRoutingV2GradTiling(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        MoeFinalizeRoutingV2GradTiling::Reset(context);
    }

    ~MoeFinalizeRoutingV2GradRegbaseSplitH() override = default;

protected:
    bool IsCapable() override
    {
        if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
            return false;
        }
        return true;
    }

    ge::graphStatus CalcTilingKey() override;

    ge::graphStatus PostTiling() override;

    ge::graphStatus CheckOptionalInputDtype() override;

private:
    int64_t mainFactor_;
    int64_t mainBlockNum_;
    int64_t foldFactor_;
    int64_t foldBlockNum_;
    int64_t foldTailLen_;
    int64_t foldTailBlockNum_;
    MoeFinalizeRoutingV2GradSplitHTilingData splitHTilingData_;
};

ge::graphStatus MoeFinalizeRoutingV2GradRegbaseSplitH::CheckOptionalInputDtype()
{
    if (CheckExpandedXAndRowIdxDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScalesDtypeForDavid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBiasExpertIdxDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradRegbaseSplitH::PostTiling()
{
    splitHTilingData_.baseParams.set_initOutNeedCoreNum(initOutNeedCoreNum_);
    splitHTilingData_.baseParams.set_initOutEachCoreBatchNum(initOutEachCoreBatchNum_);
    splitHTilingData_.baseParams.set_initOutModCoreNum(initOutModCoreNum_);
    splitHTilingData_.baseParams.set_computeNeedCoreNum(computeNeedCoreNum_);
    splitHTilingData_.baseParams.set_computeEachCoreBatchNum(computeEachCoreBatchNum_);
    splitHTilingData_.baseParams.set_computeModCoreNum(computeModCoreNum_);
    splitHTilingData_.baseParams.set_dropPadMode(dropPadMode_);
    splitHTilingData_.baseParams.set_topK(topK_);
    splitHTilingData_.baseParams.set_hidden(hidden_);
    splitHTilingData_.baseParams.set_expandedXDim0(expandedXDim0_);
    splitHTilingData_.set_mainFactor(mainFactor_);
    splitHTilingData_.set_mainBlockNum(mainBlockNum_);
    splitHTilingData_.set_foldFactor(foldFactor_);
    splitHTilingData_.set_foldBlockNum(foldBlockNum_);
    splitHTilingData_.set_foldTailLen(foldTailLen_);
    splitHTilingData_.set_foldTailBlockNum(foldTailBlockNum_);
    SetBinaryAddParams(splitHTilingData_.mainBinAddParams, mainFactor_);
    SetBinaryAddParams(splitHTilingData_.foldBinAddParams, foldFactor_);
    splitHTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                   context_->GetRawTilingData()->GetCapacity());

    context_->GetRawTilingData()->SetDataSize(splitHTilingData_.GetDataSize());
    context_->SetBlockDim(initOutNeedCoreNum_);
    if (computeNeedCoreNum_ > initOutNeedCoreNum_) {
        context_->SetBlockDim(computeNeedCoreNum_);
    }

    OP_LOGI(
        nodeName_,
        "MoeFinalizeRoutingV2Grad tilingData is initOutNeedCoreNum:%ld, initOutEachCoreBatchNum:%ld, "
        "initOutModCoreNum:%ld, computeNeedCoreNum:%ld, computeEachCoreBatchNum:%ld, computeModCoreNum:%ld, "
        "dropPadMode:%ld, topK:%ld, hidden:%ld, expandedXDim0:%ld, tilingKey:%ld, "
        "mainFactor:%ld, mainBlockNum:%ld, foldFactor:%ld, foldBlockNum:%ld, foldTailLen:%ld, foldTailBlockNum:%ld, "
        "mainBlockBinaryAddParams(binaryAddQuotient:%ld, binaryAddk:%ld, binaryAddLastNum:%ld), "
        "foldBlockBinaryAddParams(binaryAddQuotient:%ld, binaryAddk:%ld, binaryAddLastNum:%ld)",
        splitHTilingData_.baseParams.get_initOutNeedCoreNum(),
        splitHTilingData_.baseParams.get_initOutEachCoreBatchNum(),
        splitHTilingData_.baseParams.get_initOutModCoreNum(), splitHTilingData_.baseParams.get_computeNeedCoreNum(),
        splitHTilingData_.baseParams.get_computeEachCoreBatchNum(),
        splitHTilingData_.baseParams.get_computeModCoreNum(), splitHTilingData_.baseParams.get_dropPadMode(),
        splitHTilingData_.baseParams.get_topK(), splitHTilingData_.baseParams.get_hidden(),
        splitHTilingData_.baseParams.get_expandedXDim0(), tilingKey_, splitHTilingData_.get_mainFactor(),
        splitHTilingData_.get_mainBlockNum(), splitHTilingData_.get_foldFactor(), splitHTilingData_.get_foldBlockNum(),
        splitHTilingData_.get_foldTailLen(), splitHTilingData_.get_foldTailBlockNum(),
        splitHTilingData_.mainBinAddParams.get_binaryAddQuotient(), splitHTilingData_.mainBinAddParams.get_binaryAddk(),
        splitHTilingData_.mainBinAddParams.get_binaryAddLastNum(),
        splitHTilingData_.foldBinAddParams.get_binaryAddQuotient(), splitHTilingData_.foldBinAddParams.get_binaryAddk(),
        splitHTilingData_.foldBinAddParams.get_binaryAddLastNum());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradRegbaseSplitH::CalcTilingKey()
{
    int64_t dtypeSize = (gradYType_ == ge::DT_FLOAT) ? sizeof(float) : sizeof(int16_t);
    int64_t bufferNum = isBiasExist_ ? BUFFER_NUM_WITH_BIAS : BUFFER_NUM_WITHOUT_BIAS;
    // 预留1个block输出gradScale, UB间二分折叠预留3K, UB内的二分折叠预留1K空间
    int64_t reserverUbSize = BINARY_ADD_CACHE_BUFFER_SIZE + BINARY_ADD_BUFFER_SIZE + blockSize_;
    // foldFactor_ = mainFactor_  / 2, foldFactor_ 需要block对齐
    mainFactor_ = Ops::Base::FloorDiv((aicoreParams_.ubSize - reserverUbSize) / DOUBLE_BUFFER / bufferNum,
                                      static_cast<uint64_t>(blockSize_ * TWO)) *
                  (blockSize_ * TWO) / dtypeSize;
    int64_t blockNum = hidden_ / mainFactor_;
    mainBlockNum_ = blockNum <= 0 ? 1L : std::max(1L, (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(blockNum))));
    int64_t foldLen = hidden_ - mainFactor_ * mainBlockNum_;
    foldFactor_ = mainFactor_ / TWO;
    foldBlockNum_ = Ops::Base::FloorDiv(foldLen, mainFactor_) * TWO;
    foldTailLen_ = foldLen - foldBlockNum_ * foldFactor_;
    foldTailBlockNum_ = foldTailLen_ > foldFactor_ ? TWO : (foldTailLen_ > 0 ? 1 : 0);

    tilingKey_ = isBiasExist_ ? TILING_KEY_WITH_SCALE_CUT_H_WITH_BIAS : TILING_KEY_WITH_SCALE_CUT_H_WITHOUT_BIAS;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeFinalizeRoutingV2Grad, MoeFinalizeRoutingV2GradRegbaseSplitH, 4000);
} // namespace optiling
