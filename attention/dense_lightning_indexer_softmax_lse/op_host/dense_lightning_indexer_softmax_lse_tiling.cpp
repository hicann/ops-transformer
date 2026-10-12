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
 * \file dense_lightning_indexer_softmax_lse_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "dense_lightning_indexer_softmax_lse_tiling.h"
#include "../op_kernel/dense_lightning_indexer_softmax_lse_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {

struct DenseLightningIndexerSoftmaxLseCompileInfo {};

// --------------------------DenseLISoftmaxLseInfoParser类成员函数定义-------------------------------------
ge::graphStatus DenseLISoftmaxLseInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE(opName_, "Shape of tensor query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE(opName_, "Shape of tensor k is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE(opName_, "Desc of tensor k is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr, OP_LOGE(opName_, "Shape of tensor weights is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr, OP_LOGE(opName_, "Desc of tensor weights is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxMaxOut.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor output softmaxMax is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxMaxOut.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor output softmaxMax is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxSumOut.shape == nullptr,
                OP_LOGE(opName_, "Shape of tensor output softmaxSum is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxSumOut.desc == nullptr,
                OP_LOGE(opName_, "Desc of tensor output softmaxSum is nullptr"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layOut == nullptr, OP_LOGE(opName_, "attr layout is nullptr"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, OP_LOGE(opName_, "attr sparse_mode is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("DenseLightningIndexerSoftmaxLse", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93) &&
        (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_))) {
        OP_LOGE(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr, OP_LOGE(opName_, "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OP_LOGE(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void DenseLISoftmaxLseInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
}

void DenseLISoftmaxLseInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGHT_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGHT_INDEX);
    GetOptionalInputParaInfo();
}

void DenseLISoftmaxLseInfoParser::GetOutputParaInfo()
{
    opParamInfo_.softmaxMaxOut.desc = context_->GetOutputDesc(SOFTMAX_MAX_INDEX);
    opParamInfo_.softmaxMaxOut.shape = context_->GetOutputShape(SOFTMAX_MAX_INDEX);
    opParamInfo_.softmaxSumOut.desc = context_->GetOutputDesc(SOFTMAX_SUM_INDEX);
    opParamInfo_.softmaxSumOut.shape = context_->GetOutputShape(SOFTMAX_SUM_INDEX);
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.layOut = attrs->GetStr(ATTR_LAYOUT_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
    if (opParamInfo_.layOut != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout is:%s", opParamInfo_.layOut);
    }
    if (opParamInfo_.sparseMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "sparse mode is:%d", *opParamInfo_.sparseMode);
    }
    if (opParamInfo_.preTokens != nullptr) {
        OP_LOGI(context_->GetNodeName(), "pre tokens is:%d", *opParamInfo_.preTokens);
    }
    if (opParamInfo_.nextTokens != nullptr) {
        OP_LOGI(context_->GetNodeName(), "next tokens is:%d", *opParamInfo_.nextTokens);
    }
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");
    OP_CHECK_IF(*opParamInfo_.sparseMode != SPARSE_MODE_LOWER,
                OP_LOGE(opName_, "input attr sparse_mode only supported 3."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.preTokens != INT64_MAX,
                OP_LOGE(opName_, "input attr pre_tokens only supported INT64_MAX."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.nextTokens != INT64_MAX,
                OP_LOGE(opName_, "input attr nextTokens only supported INT64_MAX."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    softmaxMaxOutType_ = opParamInfo_.softmaxMaxOut.desc->GetDataType();
    softmaxSumOutType_ = opParamInfo_.softmaxSumOut.desc->GetDataType();

    bool inDTypeAllEqual = inputQType_ == inputKType_;
    OP_CHECK_IF(!inDTypeAllEqual, OP_LOGE(opName_, "The data types of the input query and key must be the same."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
                OP_LOGE(opName_, "The data types of the input query, key, and weights must be float16 or bfloat16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(softmaxMaxOutType_ != ge::DT_FLOAT,
                OP_LOGE(opName_, "The data types of the output softmax max must be float."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxSumOutType_ != softmaxMaxOutType_,
                OP_LOGE(opName_, "The data types of the output softmax sum must be same as softmax max datatype."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutMap = {{"BSND", DataLayout::BSND}, {"TND", DataLayout::TND}};

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        layout_ = it->second;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetAndCheckOptionalInput()
{
    if (layout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
                    OP_LOGE(opName_, "when layout_query is TND, input actual_seq_lengths_query must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                    OP_LOGE(opName_, "when layout_key is TND, input actual_seq_lengths_key must not be null"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
                    opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT64,
                OP_LOGE(opName_, "input actual_seq_lengths_query data type only support int64"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor != nullptr &&
                    opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT64,
                OP_LOGE(opName_, "input actual_seq_lengths_key data type only support int64"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::CheckShapeDim()
{
    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();

    uint32_t softmaxMaxOutShapeDim = opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDimNum();
    uint32_t softmaxSumOutShapeDim = opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDimNum();

    uint32_t qExpectShapeDim = DIM_NUM_FOUR;
    uint32_t kExpectShapeDim = DIM_NUM_FOUR;
    if (layout_ == DataLayout::TND) {
        qExpectShapeDim = DIM_NUM_THREE;
        kExpectShapeDim = DIM_NUM_THREE;
    }

    OP_CHECK_IF(kShapeDim != kExpectShapeDim,
                OP_LOGE(opName_, "the dim num of key's shape should be %u, but now is %u", kExpectShapeDim, kShapeDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        qShapeDim != qExpectShapeDim,
        OP_LOGE(opName_, "the dim num of query's shape should be %u, but now is %u", qExpectShapeDim, qShapeDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxMaxOutShapeDim != qExpectShapeDim - DIM_NUM_ONE,
                OP_LOGE(opName_, "the dim num of softmaxMax's shape should be %u, but now is %u",
                        qExpectShapeDim - DIM_NUM_ONE, softmaxMaxOutShapeDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxSumOutShapeDim != qExpectShapeDim - DIM_NUM_ONE,
                OP_LOGE(opName_, "the dim num of softmaxSum's shape should be %u, but now is %u",
                        qExpectShapeDim - DIM_NUM_ONE, softmaxSumOutShapeDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == qExpectShapeDim - DIM_NUM_ONE),
                OP_LOGE(opName_, "the dim num of weights's shape should be %u, but now is %u", qExpectShapeDim - 1,
                        weightsShapeDim),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetN1Size()
{
    if (layout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    }
    OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetActualSeqLenSize(uint32_t& size, const gert::Tensor* tensor,
                                                                 const std::string& actualSeqLenName)
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= DIM_NUM_ZERO) {
        OP_LOGE(opName_, "%s's shape size is %u, it should be greater than 0.", actualSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (layout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    OP_LOGI(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != N2_SIZE_LIMIT, OP_LOGE(opName_, "key shape[%u] is numhead, only support 1.", n2Index),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE(opName_, "input query's head_num %u can not be a multiple of key's head_num %u.", n1Size_, n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((layout_ == DataLayout::TND)) {
        return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "input actual_seq_lengths_query");
    } else { // BSND
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetHeadDim()
{
    // 以query的D维度为基准
    uint32_t dIndex = DIM_IDX_TWO;
    // 根据layout确定D维度在shape中的位置
    switch (layout_) {
        case DataLayout::TND:
            // TND格式: [Total, N, D] -> D是第2维(索引2)
            dIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            // BSND格式: [Batch, SeqLen, N, D] -> D是第3维(索引3)
            dIndex = DIM_IDX_THREE;
            break;
        default:
            OP_LOGE(opName_, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    OP_CHECK_IF(headDim_ != HEAD_DIM_LIMIT, OP_LOGE(opName_, "input query's last dim head_dim only support 128."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetS1Size()
{
    if (layout_ == DataLayout::TND) {
        return GetMaxSeqLenFromTensor(opParamInfo_.actualSeqLengthsQ.tensor, s1Size_, true);
    } else if (layout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetS2Size()
{
    if (layout_ == DataLayout::TND) {
        return GetMaxSeqLenFromTensor(opParamInfo_.actualSeqLengths.tensor, s2Size_, true);
    } else if (layout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::CheckSeqLenLimit()
{
    OP_CHECK_IF((s1Size_ > MAX_SEQ_LENGTH),
                OP_LOGE(opName_, "qSeqLen [%lu] exceeds max supported seqLen [%u].", s1Size_, MAX_SEQ_LENGTH),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((s2Size_ > MAX_SEQ_LENGTH),
                OP_LOGE(opName_, "kvSeqLen [%lu] exceeds max supported seqLen [%u].", s2Size_, MAX_SEQ_LENGTH),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetStorageSeqSize()
{
    if (layout_ == DataLayout::TND) {
        storageS1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
        storageS2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
    } else {
        storageS1Size_ = static_cast<uint64_t>(bSize_) * s1Size_;
        storageS2Size_ = static_cast<uint64_t>(bSize_) * s2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetMaxSeqLenFromTensor(const gert::Tensor* tensor, uint64_t& maxSeqLen,
                                                                    bool isAccumSeq)
{
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t* data = tensor->GetData<int64_t>();
    if (data == nullptr) {
        OP_LOGE(opName_, "actualSeqLengths tensor data is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (isAccumSeq) {
        maxSeqLen = data[0];
        for (int64_t i = 1; i < shapeSize; ++i) {
            int64_t diff = data[i] - data[i - 1];
            if (diff > static_cast<int64_t>(maxSeqLen)) {
                maxSeqLen = diff;
            }
        }
    } else {
        for (int64_t i = 0; i < shapeSize; ++i) {
            if (data[i] > static_cast<int64_t>(maxSeqLen)) {
                maxSeqLen = data[i];
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::ValidateInputShapesMatchTnd()
{
    // -----------------------check BatchSize-------------------
    // bSize_ 来源于act_seq_q
    OP_CHECK_IF((opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
                OP_LOGE(opName_,
                        "TND case input actual_seq_lengths_query, actual_seq_lengths_key are "
                        "%u, %ld respectively, they must be same.",
                        bSize_, opParamInfo_.actualSeqLengths.tensor->GetShapeSize()),
                return ge::GRAPH_FAILED);
    // -----------------------check T-------------------
    uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != qTsize) ||
                    (opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != qTsize) ||
                    (opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != qTsize),
                OP_LOGE(opName_,
                        "TND case input query, weights, output softmaxMax, "
                        "softmaxSum dim 1 are %u, %ld, %ld, %ld respectively, they must be same.",
                        qTsize, opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ZERO),
                        opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE),
                        opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::ValidateInputShapesMatchBsnd()
{
    // -----------------------check BatchSize-------------------
    // bSize_ 来源于query
    OP_CHECK_IF(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != bSize_,
                OP_LOGE(opName_, "BSND case input query, key dim 0 are %u, %ld respectively, they must be same.",
                        bSize_, opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ZERO)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.actualSeqLengths.tensor != nullptr) &&
                    (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
                OP_LOGE(opName_,
                        "BSND case input query, actual_seq_lengths_key dim 0 are "
                        "%u, %ld respectively, they must be same.",
                        bSize_, opParamInfo_.actualSeqLengths.tensor->GetShapeSize()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != bSize_) ||
                    (opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != bSize_) ||
                    (opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != bSize_),
                OP_LOGE(opName_,
                        "BSND case input query, weight, output softmaxMax, "
                        "softmaxSumOut dim 0 are %u, %ld, %ld, %ld respectively, they must be same.",
                        bSize_, opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ZERO),
                        opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_ZERO),
                        opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_ZERO)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.actualSeqLengthsQ.tensor != nullptr) &&
                    (opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize() != bSize_),
                OP_LOGE(opName_,
                        "BSND case input query, actual_seq_lengths_query dim 0 are %u, %ld "
                        "respectively, they must be same",
                        bSize_, opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize()),
                return ge::GRAPH_FAILED);
    // -----------------------check S1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != s1Size_) ||
                    (opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_TWO) != s1Size_) ||
                    (opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_TWO) != s1Size_),
                OP_LOGE(opName_,
                        "BSND case input query, weight, output softmaxMax, softmaxSum dim 2 "
                        "are %u, %ld, %ld, %ld, they must be same.",
                        s1Size_, opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ONE),
                        opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_TWO),
                        opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_TWO)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::ValidateInputShapesMatch()
{
    uint32_t queryWeightsN1Dim = DIM_IDX_ONE;
    if (layout_ == DataLayout::TND) {
        if (ValidateInputShapesMatchTnd() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else { // qLayout_ BSND
        if (ValidateInputShapesMatchBsnd() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        queryWeightsN1Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
                OP_LOGE(opName_, "input query, weight shape dim N1 must be same."), return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    uint32_t keyDDim = layout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    OP_CHECK_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) != headDim_),
                OP_LOGE(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void DenseLISoftmaxLseInfoParser::GenerateInfo(DenseLISoftmaxLseTilingInfo& denseLISoftmaxTilingInfo)
{
    denseLISoftmaxTilingInfo.opName = opName_;
    denseLISoftmaxTilingInfo.platformInfo = platformInfo_;
    denseLISoftmaxTilingInfo.opParamInfo = opParamInfo_;
    denseLISoftmaxTilingInfo.socVersion = socVersion_;

    denseLISoftmaxTilingInfo.bSize = bSize_;
    denseLISoftmaxTilingInfo.n1Size = n1Size_;
    denseLISoftmaxTilingInfo.n2Size = n2Size_;
    denseLISoftmaxTilingInfo.s1Size = s1Size_;
    denseLISoftmaxTilingInfo.s2Size = s2Size_;
    denseLISoftmaxTilingInfo.storageS1Size = storageS1Size_;
    denseLISoftmaxTilingInfo.storageS2Size = storageS2Size_;
    denseLISoftmaxTilingInfo.gSize = gSize_;

    denseLISoftmaxTilingInfo.inputQType = inputQType_;
    denseLISoftmaxTilingInfo.inputKType = inputKType_;
    denseLISoftmaxTilingInfo.softmaxMaxOutType = softmaxMaxOutType_;
    denseLISoftmaxTilingInfo.softmaxSumOutType = softmaxSumOutType_;
    denseLISoftmaxTilingInfo.sparseMode = *opParamInfo_.sparseMode;
    denseLISoftmaxTilingInfo.preTokens = *opParamInfo_.preTokens;
    denseLISoftmaxTilingInfo.nextTokens = *opParamInfo_.nextTokens;
    denseLISoftmaxTilingInfo.layout = layout_;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::ParseAndCheck(DenseLISoftmaxLseTilingInfo& denseLISoftmaxTilingInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeDim() || ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() || ge::GRAPH_SUCCESS != CheckSeqLenLimit() ||
        ge::GRAPH_SUCCESS != GetStorageSeqSize()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(denseLISoftmaxTilingInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------DenseLightningIndexerSoftmaxLseTiling类成员函数定义-----------------------
ge::graphStatus DenseLightningIndexerSoftmaxLseTiling::DoTiling(DenseLISoftmaxLseTilingInfo* tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    // -------------set workspacesize-----------------
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += M_BASE_SIZE * S2_BASE_SIZE * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    uint64_t s2AlignSize = (tilingInfo->s2Size + S2_BASE_SIZE - 1) / S2_BASE_SIZE * S2_BASE_SIZE;
    workspaceSize += aicNum * S1_BASE_SIZE * s2AlignSize * MM1_RES_ELEM_SIZE;
    size_t* workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_storageS1Size(tilingInfo->storageS1Size);
    tilingData_.set_storageS2Size(tilingInfo->storageS2Size);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_sparseMode(tilingInfo->sparseMode);
    tilingData_.set_preTokens(tilingInfo->preTokens);
    tilingData_.set_nextTokens(tilingInfo->nextTokens);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    uint32_t layout = static_cast<uint32_t>(tilingInfo->layout);
    uint32_t tilingKey = GET_TPL_TILING_KEY(layout);
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus DenseLightningIndexerSoftmaxLseTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("DenseLightningIndexerSoftmaxLse", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    DenseLISoftmaxLseTilingInfo denseLISoftmaxTilingInfo;
    DenseLISoftmaxLseInfoParser DenseLISoftmaxLseInfoParser(context);
    if (DenseLISoftmaxLseInfoParser.ParseAndCheck(denseLISoftmaxTilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DenseLightningIndexerSoftmaxLseTiling denseLightningIndexerSoftmaxLseTiling(context);
    return denseLightningIndexerSoftmaxLseTiling.DoTiling(&denseLISoftmaxTilingInfo);
}

static ge::graphStatus TilingParseForDenseLightningIndexerSoftmaxLse([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(DenseLightningIndexerSoftmaxLse)
    .Tiling(DenseLightningIndexerSoftmaxLseTilingFunc)
    .TilingParse<DenseLightningIndexerSoftmaxLseCompileInfo>(TilingParseForDenseLightningIndexerSoftmaxLse);
} // namespace optiling
