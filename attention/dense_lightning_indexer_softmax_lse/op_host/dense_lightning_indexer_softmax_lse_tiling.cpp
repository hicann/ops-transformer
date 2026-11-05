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

#include "util/shape_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"
#include "graph/utils/type_utils.h"
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
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "query_index"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "query_index"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "key_index"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "key_index"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "weight"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "weight"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.softmaxMaxOut.shape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{opName_, "Shape of tensor output softmaxMax is nullptr"})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.softmaxMaxOut.desc == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{opName_, "Desc of tensor output softmaxMax is nullptr"})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.softmaxSumOut.shape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{opName_, "Shape of tensor output softmaxSum is nullptr"})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.softmaxSumOut.desc == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{opName_, "Desc of tensor output softmaxSum is nullptr"})),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(
        opParamInfo_.layOut == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{opName_, "Tiling: registered optional attribute layout is unavailable; check "
                                               "context construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        opParamInfo_.sparseMode == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{opName_, "Tiling: registered optional attribute sparse_mode is unavailable; "
                                               "check context construction and default-attribute materialization."})),
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
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"DenseLightningIndexerSoftmaxLse", "opName got from TilingContext is nullptr"}));
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName_, "GetPlatformInfo is nullptr."})),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{opName_, (std::string("Platform core counts must be nonzero; AIC=") +
                                                        std::to_string(aicNum) + ", AIV=" + std::to_string(aivNum))
                                                           .c_str()})),
                return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93) &&
        (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_))) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ3002", (std::vector<const char*>{"optype", "opskernel", "reason"}),
            (std::vector<const char*>{
                "DenseLightningIndexerSoftmaxLse", "AiCore",
                (std::string("SOC Version[") + std::to_string((int32_t)socVersion_) + "] is not support.").c_str()}));
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName_, "workSpaceSize got from ge is nullptr"})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetRawTilingData() == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{opName_, "RawTilingData got from GE context is nullptr."})),
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
    OP_CHECK_IF(attrs == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{opName_, "attrs got from ge is nullptr"})),
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
                OP_LOGE_FOR_INVALID_VALUE(opName_, "sparse_mode", std::to_string(*opParamInfo_.sparseMode).c_str(),
                                          std::to_string(SPARSE_MODE_LOWER).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.preTokens != INT64_MAX,
                OP_LOGE_FOR_INVALID_VALUE(opName_, "pre_tokens", std::to_string(*opParamInfo_.preTokens).c_str(),
                                          std::to_string(INT64_MAX).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.nextTokens != INT64_MAX,
                OP_LOGE_FOR_INVALID_VALUE(opName_, "next_tokens", std::to_string(*opParamInfo_.nextTokens).c_str(),
                                          std::to_string(INT64_MAX).c_str()),
                return ge::GRAPH_FAILED);

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
    OP_CHECK_IF(!inDTypeAllEqual,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "query_index,key_index",
                                                       (ge::TypeUtils::DataTypeToSerialString(inputQType_) + "," +
                                                        ge::TypeUtils::DataTypeToSerialString(inputKType_))
                                                           .c_str(),
                                                       "query_index and key_index must have the same dtype."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "query_index",
                                                      (ge::TypeUtils::DataTypeToSerialString(inputQType_)).c_str(),
                                                      "Supported dtypes are FLOAT16 and BF16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(softmaxMaxOutType_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "softmax_max", (ge::TypeUtils::DataTypeToSerialString(softmaxMaxOutType_)).c_str(),
                    "The output dtype must be FLOAT."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxSumOutType_ != softmaxMaxOutType_,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "softmax_max,softmax_sum",
                                                       (ge::TypeUtils::DataTypeToSerialString(softmaxMaxOutType_) +
                                                        "," + ge::TypeUtils::DataTypeToSerialString(softmaxSumOutType_))
                                                           .c_str(),
                                                       "The two output dtypes must match."),
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
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "actual_seq_lengths_query",
                                                             "This input must be provided when layout is TND."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "actual_seq_lengths_key",
                                                             "This input must be provided when layout is TND."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
                    opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "actual_seq_lengths_query",
                    (ge::TypeUtils::DataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType())).c_str(),
                    "The dtype must be INT64."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor != nullptr &&
                    opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "actual_seq_lengths_key",
                    (ge::TypeUtils::DataTypeToSerialString(opParamInfo_.actualSeqLengths.desc->GetDataType())).c_str(),
                    "The dtype must be INT64."),
                return ge::GRAPH_FAILED);
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
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "key_index", std::to_string(kShapeDim).c_str(),
                                             std::to_string(kExpectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qShapeDim != qExpectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "query_index", std::to_string(qShapeDim).c_str(),
                                             std::to_string(qExpectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxMaxOutShapeDim != qExpectShapeDim - DIM_NUM_ONE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "softmax_max", std::to_string(softmaxMaxOutShapeDim).c_str(),
                                             std::to_string(qExpectShapeDim - DIM_NUM_ONE).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxSumOutShapeDim != qExpectShapeDim - DIM_NUM_ONE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "softmax_sum", std::to_string(softmaxSumOutShapeDim).c_str(),
                                             std::to_string(qExpectShapeDim - DIM_NUM_ONE).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == qExpectShapeDim - DIM_NUM_ONE),
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "weight", std::to_string(weightsShapeDim).c_str(),
                                             std::to_string(qExpectShapeDim - DIM_NUM_ONE).c_str()),
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
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, actualSeqLenName.c_str(), std::to_string(size).c_str(),
                                                  "The element count must be greater than zero.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (layout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    OP_LOGI(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != N2_SIZE_LIMIT,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "key_index", std::to_string(n2Size_).c_str(),
                                                          "The head count must equal 1."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName_, "query_index,key_index", (std::to_string(n1Size_) + "," + std::to_string(n2Size_)).c_str(),
            "The query_index head count must be a multiple of the key_index head count.");
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
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(opName_, "layout", opParamInfo_.layOut,
                                                   "Supported layouts are TND and BSND.");
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    OP_CHECK_IF(headDim_ != HEAD_DIM_LIMIT,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "query_index", std::to_string(headDim_).c_str(),
                                                          "The last dimension must equal 128."),
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
    OP_CHECK_IF(
        (s1Size_ > MAX_SEQ_LENGTH),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, "query_index", std::to_string(s1Size_).c_str(),
            (std::string("The sequence length must not exceed ") + std::to_string(MAX_SEQ_LENGTH) + ".").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (s2Size_ > MAX_SEQ_LENGTH),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, "key_index", std::to_string(s2Size_).c_str(),
            (std::string("The sequence length must not exceed ") + std::to_string(MAX_SEQ_LENGTH) + ".").c_str()),
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

ge::graphStatus DenseLISoftmaxLseInfoParser::CheckSeqLenRelation()
{
    if (layout_ == DataLayout::BSND) {
        // 约束说明: 当layout为BSND时, S1<=S2
        OP_CHECK_IF(s1Size_ > s2Size_,
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        opName_, "query_index,key_index",
                        (std::string("S1=") + std::to_string(s1Size_) + ",S2=" + std::to_string(s2Size_)).c_str(),
                        "When layout is BSND, the query sequence length must not exceed the key sequence length."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // TND: actual_seq_lengths_query/actual_seq_lengths_key为必传的累计值数组, 元素个数均为bSize_
    const gert::Tensor* querySeqTensor = opParamInfo_.actualSeqLengthsQ.tensor;
    const gert::Tensor* keySeqTensor = opParamInfo_.actualSeqLengths.tensor;
    OP_CHECK_IF(querySeqTensor == nullptr || keySeqTensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "actual_seq_lengths_query,actual_seq_lengths_key",
                                                         "This input must be provided when layout is TND."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        querySeqTensor->GetShapeSize() != keySeqTensor->GetShapeSize(),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName_, "actual_seq_lengths_query,actual_seq_lengths_key",
            (std::to_string(querySeqTensor->GetShapeSize()) + "," + std::to_string(keySeqTensor->GetShapeSize()))
                .c_str(),
            "The two sequence-length arrays must have the same element count."),
        return ge::GRAPH_FAILED);
    const int64_t* querySeqData = querySeqTensor->GetData<int64_t>();
    const int64_t* keySeqData = keySeqTensor->GetData<int64_t>();
    OP_CHECK_IF(querySeqData == nullptr || keySeqData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "actual_seq_lengths_query,actual_seq_lengths_key",
                                                         "The sequence-length tensor data must not be nullptr."),
                return ge::GRAPH_FAILED);
    // 逐索引校验: 累计值单调不减, 且相同索引位置S1<=S2(等价于相同索引位置累计值query<=key)
    int64_t prevQueryLen = 0;
    int64_t prevKeyLen = 0;
    for (uint32_t i = 0; i < bSize_; ++i) {
        const int64_t batchQueryLen = querySeqData[i] - prevQueryLen;
        const int64_t batchKeyLen = keySeqData[i] - prevKeyLen;
        OP_CHECK_IF(batchQueryLen < 0 || batchKeyLen < 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "actual_seq_lengths_query,actual_seq_lengths_key",
                        (std::to_string(querySeqData[i]) + "," + std::to_string(keySeqData[i])).c_str(),
                        "The accumulated sequence lengths must be non-decreasing."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(batchQueryLen > batchKeyLen,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "actual_seq_lengths_query,actual_seq_lengths_key",
                        (std::to_string(batchQueryLen) + "," + std::to_string(batchKeyLen)).c_str(),
                        (std::string("At index ") + std::to_string(i) +
                         ", the query sequence length must be less than or equal to the key sequence length.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        prevQueryLen = querySeqData[i];
        prevKeyLen = keySeqData[i];
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
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{opName_, "actualSeqLengths tensor data is nullptr."}));
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
    OP_CHECK_IF(
        (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName_, "actual_seq_lengths_query,actual_seq_lengths_key",
            (std::to_string(bSize_) + "," + std::to_string(opParamInfo_.actualSeqLengths.tensor->GetShapeSize()))
                .c_str(),
            "The sequence-length arrays must have the same element count for TND."),
        return ge::GRAPH_FAILED);
    // -----------------------check T-------------------
    uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != qTsize) ||
                    (opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != qTsize) ||
                    (opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != qTsize),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query_index,weight,softmax_max,softmax_sum",
                    (Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.softmaxMaxOut.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.softmaxSumOut.shape->GetStorageShape()))
                        .c_str(),
                    "For TND, query_index and weight dimension 0 must match dimension 1 of both softmax outputs."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLISoftmaxLseInfoParser::ValidateInputShapesMatchBsnd()
{
    // -----------------------check BatchSize-------------------
    // bSize_ 来源于query
    OP_CHECK_IF(
        opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != bSize_,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query_index,key_index",
                                               (Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + "," +
                                                Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()))
                                                   .c_str(),
                                               "For BSND, query_index and key_index batch dimensions must match."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (opParamInfo_.actualSeqLengths.tensor != nullptr) &&
            (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName_, "query_index,actual_seq_lengths_key",
            (std::to_string(bSize_) + "," + std::to_string(opParamInfo_.actualSeqLengths.tensor->GetShapeSize()))
                .c_str(),
            "For BSND, the query_index batch dimension must equal the sequence-length array element count."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != bSize_) ||
                    (opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != bSize_) ||
                    (opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) != bSize_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query_index,weight,softmax_max,softmax_sum",
                    (Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.softmaxMaxOut.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.softmaxSumOut.shape->GetStorageShape()))
                        .c_str(),
                    "For BSND, the batch dimensions of these tensors must match."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) &&
            (opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize() != bSize_),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName_, "query_index,actual_seq_lengths_query",
            (std::to_string(bSize_) + "," + std::to_string(opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize()))
                .c_str(),
            "For BSND, the query_index batch dimension must equal the sequence-length array element count."),
        return ge::GRAPH_FAILED);
    // -----------------------check S1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != s1Size_) ||
                    (opParamInfo_.softmaxMaxOut.shape->GetStorageShape().GetDim(DIM_IDX_TWO) != s1Size_) ||
                    (opParamInfo_.softmaxSumOut.shape->GetStorageShape().GetDim(DIM_IDX_TWO) != s1Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query_index,weight,softmax_max,softmax_sum",
                    (Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.softmaxMaxOut.shape->GetStorageShape()) + "," +
                     Ops::Base::ToString(opParamInfo_.softmaxSumOut.shape->GetStorageShape()))
                        .c_str(),
                    "For BSND, query_index and weight dimension 1 must match dimension 2 of both softmax outputs."),
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
    OP_CHECK_IF(
        (opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query_index,weight",
                                               (Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + "," +
                                                Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()))
                                                   .c_str(),
                                               "The query_index and weight head counts must match."),
        return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    uint32_t keyDDim = layout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    OP_CHECK_IF(
        (opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) != headDim_),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query_index,key_index",
                                               (Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + "," +
                                                Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()))
                                                   .c_str(),
                                               "The query_index and key_index last dimensions must match."),
        return ge::GRAPH_FAILED);
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
        ge::GRAPH_SUCCESS != CheckSeqLenRelation() || ge::GRAPH_SUCCESS != GetStorageSeqSize()) {
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
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"DenseLightningIndexerSoftmaxLse", "Tiling context is null."})),
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
