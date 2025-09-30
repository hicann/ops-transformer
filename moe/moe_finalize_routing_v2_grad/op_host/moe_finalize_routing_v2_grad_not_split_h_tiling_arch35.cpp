/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_grad_not_split_h_tiling_arch35.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_grad_tiling.h"

namespace optiling {
constexpr int64_t TILING_KEY_WITH_SCALE_NOT_CUT_H_WITHOUT_BIAS = 20011;
constexpr int64_t TILING_KEY_WITH_SCALE_NOT_CUT_H_WITH_BIAS = 20021;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t FP32_SIZE = sizeof(float);

class MoeFinalizeRoutingV2GradNotSplitHRegbase : public MoeFinalizeRoutingV2GradTiling
{
public:
    explicit MoeFinalizeRoutingV2GradNotSplitHRegbase(gert::TilingContext* context)
        : MoeFinalizeRoutingV2GradTiling(context)
    {}
    ~MoeFinalizeRoutingV2GradNotSplitHRegbase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        MoeFinalizeRoutingV2GradTiling::Reset(context);
    }

protected:
    ge::graphStatus CalcTilingKey() override;
    ge::graphStatus PostTiling() override;
    bool IsCapable() override
    {
        if (socVersion_ != platform_ascendc::SocVersion::ASCEND910_95) {
            return false;
        }
        CalcHAlignAndBinaryAddParamInUb();
        CalcMaxHiddenInUb();
        if (hiddenPrePart_ <= 0 || hiddenPrePart_ < hAlign_) {
            return false;
        }
        return true;
    }
    ge::graphStatus CheckOptionalInputDtype() override;

private:
    void CalcHAlignAndBinaryAddParamInUb();
    void CalcMaxHiddenInUb();

private:
    MoeFinalizeRoutingV2GradNotSplitHTilingData tilingData_;
    MoeFinalizeRoutingV2GradBinaryAddTilingData binaryAddTilingData_;
    int64_t hAlign_ = 0;
};

ge::graphStatus MoeFinalizeRoutingV2GradNotSplitHRegbase::CheckOptionalInputDtype()
{
     OP_CHECK_IF(
        (expandedXType_ != gradYType_), OP_LOGE(nodeName_, "expanded_x and grad_y dtype must be same."),
        return ge::GRAPH_FAILED);
     OP_CHECK_IF(
        ((scalesType_ != ge::DT_FLOAT) && (scalesType_ != ge::DT_BF16) && (scalesType_ != ge::DT_FLOAT16)),
        OP_LOGE(nodeName_, "scales dtype must be FLOAT or FLOAT16 or BFLOAT16."), return ge::GRAPH_FAILED);
     OP_CHECK_IF(
        (expandedRowIdxType_ != ge::DataType::DT_INT32),
        OP_LOGE(nodeName_, "expanded_row_idx dtype only support int32."), return ge::GRAPH_FAILED);
    if (isBiasExist_) {
         OP_CHECK_IF(
            (expertIdxType_ != expandedRowIdxType_),
            OP_LOGE(nodeName_, "expert_idx and expanded_row_idx dtype must be same."), return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (biasType_ != gradYType_), OP_LOGE(nodeName_, "bias and grad_y dtype must be same."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradNotSplitHRegbase::CalcTilingKey()
{
    tilingKey_ =
        isBiasExist_ ? TILING_KEY_WITH_SCALE_NOT_CUT_H_WITH_BIAS : TILING_KEY_WITH_SCALE_NOT_CUT_H_WITHOUT_BIAS;
    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2GradNotSplitHRegbase::CalcHAlignAndBinaryAddParamInUb()
{
    hAlign_ = (hidden_ * gradYTypeByteSize_ + blockSize_ - 1) / blockSize_ * blockSize_ / gradYTypeByteSize_;
    SetBinaryAddParams(binaryAddTilingData_, hidden_);
}

void MoeFinalizeRoutingV2GradNotSplitHRegbase::CalcMaxHiddenInUb()
{
    int64_t gradYElementSize = gradYTypeByteSize_ * DOUBLE_BUFFER;
    int64_t expandedXElementSize = gradYTypeByteSize_ * DOUBLE_BUFFER;
    int64_t gradExpandedXElementSize = gradYTypeByteSize_ * DOUBLE_BUFFER;
    int64_t elementTotalSize = gradYElementSize + expandedXElementSize + gradExpandedXElementSize;
    if (isBiasExist_) {
        uint64_t biasTypePart = gradYElementSize;
        elementTotalSize = elementTotalSize + biasTypePart;
    }
    int64_t gradScalesUbSize = blockSize_ * DOUBLE_BUFFER;
    int64_t quotientVcaddNum = binaryAddTilingData_.get_binaryAddQuotient() / this->vlFp32_;
    int64_t binaryAddUbSize = ((quotientVcaddNum * FP32_SIZE + blockSize_ - 1) / blockSize_) * blockSize_;
    hiddenPrePart_ = (aicoreParams_.ubSize - binaryAddUbSize - gradScalesUbSize) / elementTotalSize *
                     gradYTypeByteSize_ / blockSize_ * blockSize_ / gradYTypeByteSize_;
}

ge::graphStatus MoeFinalizeRoutingV2GradNotSplitHRegbase::PostTiling()
{
    tilingData_.baseParams.set_dropPadMode(dropPadMode_);
    tilingData_.baseParams.set_topK(topK_);
    tilingData_.baseParams.set_hidden(hidden_);
    tilingData_.baseParams.set_expandedXDim0(expandedXDim0_);
    tilingData_.baseParams.set_initOutNeedCoreNum(initOutNeedCoreNum_);
    tilingData_.baseParams.set_initOutEachCoreBatchNum(initOutEachCoreBatchNum_);
    tilingData_.baseParams.set_initOutModCoreNum(initOutModCoreNum_);
    tilingData_.baseParams.set_computeNeedCoreNum(computeNeedCoreNum_);
    tilingData_.baseParams.set_computeEachCoreBatchNum(computeEachCoreBatchNum_);
    tilingData_.baseParams.set_computeModCoreNum(computeModCoreNum_);

    tilingData_.set_hAlign(hAlign_);

    tilingData_.binAddParams.set_binaryAddQuotient(binaryAddTilingData_.get_binaryAddQuotient());
    tilingData_.binAddParams.set_binaryAddk(binaryAddTilingData_.get_binaryAddk());
    tilingData_.binAddParams.set_binaryAddLastNum(binaryAddTilingData_.get_binaryAddLastNum());

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());

    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(std::max(computeNeedCoreNum_, initOutNeedCoreNum_));
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MoeFinalizeRoutingV2Grad", MoeFinalizeRoutingV2GradNotSplitHRegbase, 3000);

} // namespace optiling