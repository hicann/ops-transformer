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
 * \file lightning_indexer_tiling.cpp
 * \brief
 */

#include "lightning_indexer_tiling.h"
#include "../op_kernel/lightning_indexer_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},   // fp8_e4m3 type
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},             // hifloat8 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

static std::string LIDataTypeToSerialString(ge::DataType type)
{
    const auto liIt = DATATYPE_TO_STRING_MAP.find(type);
    if (liIt != DATATYPE_TO_STRING_MAP.end()) {
        return liIt->second;
    } else {
        OP_LOGE("LightningIndexer", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static std::string ToStringRaw(const gert::Shape &shape)
{
    std::ostringstream oss;
    auto v = ToVector(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

ge::graphStatus LIInfoParser::CheckTensorShapes() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "The shape of query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key", "The shape of key is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "weights", "The shape of weights is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.attenOut.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The shape of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.valuesOut.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_values", "The shape of sparse_values is nullptr"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckTensorDescriptions() const
{
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "The desc of query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key", "The desc of key is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "weights", "The desc of weights is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.attenOut.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "attention_out", "The desc of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.valuesOut.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_values", "The desc of sparse_values is nullptr"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// --------------------------LIInfoParser类成员函数定义-------------------------------------
ge::graphStatus LIInfoParser::CheckRequiredInOutExistence() const
{
    ge::graphStatus status = CheckTensorShapes();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = CheckTensorDescriptions();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layOut == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_query", "Layout_query is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.layOutKey == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_key", "Layout_key is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseCount == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_count", "Sparse_count is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_mode", "Sparse_mode is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("LightningIndexer", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if ((npuArch_ != NpuArch::DAV_2201) && (npuArch_ != NpuArch::DAV_3510)) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr, OP_LOGE(opName_, "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OP_LOGE(opName_, "RawTilingData got from GE context is nullptr"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void LIInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void LIInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    GetOptionalInputParaInfo();
}

void LIInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(LIGHTNING_INDEXER);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(LIGHTNING_INDEXER);
    opParamInfo_.valuesOut.desc = context_->GetOutputDesc(LIGHTNING_VALUES);
    opParamInfo_.valuesOut.shape = context_->GetOutputShape(LIGHTNING_VALUES);
}

ge::graphStatus LIInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from GE is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
    opParamInfo_.returnValue = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);

    auto keyStrides = context_->GetDynamicInputStride(KEY_INDEX, 0);
    if (keyStrides != nullptr && keyStrides->GetDimNum() > 1) {
        keyStride0_ = keyStrides->GetStride(0);
        keyStride1_ = keyStrides->GetStride(1);
    }

    if (opParamInfo_.layOut != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_query is:%s", opParamInfo_.layOut);
    }
    if (opParamInfo_.layOutKey != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_key is:%s", opParamInfo_.layOutKey);
    }
    if (opParamInfo_.sparseCount != nullptr) {
        OP_LOGI(context_->GetNodeName(), "selscted count is:%d", *opParamInfo_.sparseCount);
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
    if (opParamInfo_.returnValue != nullptr) {
        OP_LOGI(context_->GetNodeName(), "return value is:%d", *opParamInfo_.returnValue);
    }
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");
    OP_CHECK_IF(((std::string(opParamInfo_.layOutKey) != "PA_BSND") &&
                 (std::string(opParamInfo_.layOut) != std::string(opParamInfo_.layOutKey))),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "layout_query and layout_key",
                    std::string(opParamInfo_.layOut) + " and " + std::string(opParamInfo_.layOutKey),
                    "When layout_key is non-PA, layout_query and layout_key must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BSND") && (std::string(opParamInfo_.layOutKey) != "BSND") &&
         (std::string(opParamInfo_.layOutKey) != "TND")),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_key", std::string(opParamInfo_.layOutKey).c_str(),
                                              "Layout_key only supports PA_BSND, BSND or TND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((std::string(opParamInfo_.layOut) != "BSND") && (std::string(opParamInfo_.layOut) != "TND")),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_query", std::string(opParamInfo_.layOut).c_str(),
                                                      "Layout_query only supports BSND or TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.sparseCount <= 0 || *opParamInfo_.sparseCount > static_cast<int32_t>(TOPK_MAX) ||
                    (*opParamInfo_.sparseCount > static_cast<int32_t>(SPARSE_2K) &&
                     *opParamInfo_.sparseCount % TOPK_MULTIPLE != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName_, "sparse_count", std::to_string(*opParamInfo_.sparseCount).c_str(),
                    "Sparse_count must > 0 and <= 8192."
                    " And when sparse_count > 2048, sparse_count must be an integer multiple of 1024"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!((*opParamInfo_.sparseMode == 0) || (*opParamInfo_.sparseMode == SPARSE_MODE_LOWER)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_mode", std::to_string(*opParamInfo_.sparseMode),
                                                      "Sparse_mode must be 0 or 3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.preTokens != INT64_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "pre_tokens", std::to_string(*opParamInfo_.preTokens),
                                                      "Pre_tokens must be INT64_MAX"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.nextTokens != INT64_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "next_tokens", std::to_string(*opParamInfo_.nextTokens),
                                                      "Next_tokens must be INT64_MAX"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.returnValue && std::string(opParamInfo_.layOutKey) == "PA_BSND",
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_key", std::string(opParamInfo_.layOutKey),
                                                      "When return_value is true, layout_key does not support PA_BSND"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    valuesOutType_ = opParamInfo_.valuesOut.desc->GetDataType();

    bool inDTypeAllEqual = (inputQType_ == inputKType_);
    OP_CHECK_IF(!inDTypeAllEqual,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "query and key",
                    LIDataTypeToSerialString(inputQType_) + " and " + LIDataTypeToSerialString(inputKType_),
                    "The dtype of query and key must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "query and key",
                    LIDataTypeToSerialString(inputQType_) + " and " + LIDataTypeToSerialString(inputKType_),
                    "The dtype of query and key must be float16 or bfloat16"),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF((inputQType_ != weightsType_),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "query, key and weights",
                                                           LIDataTypeToSerialString(inputQType_) + ", " +
                                                               LIDataTypeToSerialString(inputKType_) + " and " +
                                                               LIDataTypeToSerialString(inputKType_),
                                                           "The dtype of query, key and weights must be same"),
                    return ge::GRAPH_FAILED);
    } else {
        if (weightsType_ != ge::DT_FLOAT) {
            OP_CHECK_IF((inputQType_ != weightsType_),
                        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "query, key and weights",
                                                               LIDataTypeToSerialString(inputQType_) + ", " +
                                                                   LIDataTypeToSerialString(inputKType_) + " and " +
                                                                   LIDataTypeToSerialString(inputKType_),
                                                               "The dtype of query, key and weights must be same"),
                        return ge::GRAPH_FAILED);
        }
    }
    OP_CHECK_IF(
        outputType_ != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sparse_indices", LIDataTypeToSerialString(outputType_).c_str(),
                                              "The dtype of sparse_indices must be int32"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(valuesOutType_ != inputQType_,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "query and sparse_values",
                    LIDataTypeToSerialString(inputQType_) + " and " + LIDataTypeToSerialString(valuesOutType_),
                    "The dtype of query and sparse_values must be same"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutMap = {
        {"BSND", DataLayout::BSND}, {"TND", DataLayout::TND}, {"PA_BSND", DataLayout::BnBsND}};

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second;
    }

    std::string layoutKey(opParamInfo_.layOutKey);
    auto itKey = layoutMap.find(layoutKey);
    if (itKey != layoutMap.end()) {
        kLayout_ = itKey->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::BnBsND) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "block_table", "When layout_key is PA_BSND, block_table must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "actual_seq_lengths_key",
                        "When layout_key is PA_BSND, actual_seq_lengths_key must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "block_table", LIDataTypeToSerialString(opParamInfo_.blockTable.desc->GetDataType()).c_str(),
                "The dtype of block_table must be int32"),
            return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::TND) {
        OP_CHECK_IF(
            opParamInfo_.actualSeqLengths.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "actual_seq_lengths_key",
                                                     "When layout_key is TND, actual_seq_lengths_key must not be null"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor != nullptr &&
                    opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "actual_seq_lengths_key",
                    LIDataTypeToSerialString(opParamInfo_.actualSeqLengths.desc->GetDataType()).c_str(),
                    "The dtype of actual_seq_lengths_key must be int32"),
                return ge::GRAPH_FAILED);
    if (qLayout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "actual_seq_lengths_query",
                        "When layout_query is TND, actual_seq_lengths_query must not be null"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
                    opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "actual_seq_lengths_query",
                    LIDataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str(),
                    "The dtype of actual_seq_lengths_query must be int32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kLayout_ != DataLayout::BnBsND && opParamInfo_.blockTable.tensor != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "block_table",
                                                         "When layout_key is not PA_BSND, block_table must be null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckShapeDim()
{
    OP_CHECK_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                    (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "block_table",
                    std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum()).c_str(),
                    "The shape dim of block_table must be 2"),
                return ge::GRAPH_FAILED);

    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t valuesOutShapeDim = opParamInfo_.valuesOut.shape->GetStorageShape().GetDimNum();
    uint32_t qExpectShapeDim = DIM_NUM_FOUR;
    uint32_t kExpectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        qExpectShapeDim = DIM_NUM_THREE;
    }
    if (kLayout_ == DataLayout::TND) {
        kExpectShapeDim = DIM_NUM_THREE;
    }
    OP_CHECK_IF(
        kShapeDim != kExpectShapeDim,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "key", std::to_string(kShapeDim).c_str(),
                                                 "The shape dim of key must be " + std::to_string(kExpectShapeDim)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        qShapeDim != qExpectShapeDim,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "query", std::to_string(qShapeDim).c_str(),
                                                 "The shape dim of query must be " + std::to_string(qExpectShapeDim)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShapeDim != qExpectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "sparse_indices", std::to_string(outShapeDim).c_str(),
                    "The shape dim of sparse_indices must be " + std::to_string(qExpectShapeDim)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valuesOutShapeDim != qExpectShapeDim && (*opParamInfo_.returnValue),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "sparse_values", std::to_string(valuesOutShapeDim).c_str(),
                    "The shape dim of sparse_values must be " + std::to_string(qExpectShapeDim)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == qExpectShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "weights", std::to_string(weightsShapeDim).c_str(),
                    "The shape dim of weights must be " + std::to_string(qExpectShapeDim - 1)),
                return ge::GRAPH_FAILED);
    if (opParamInfo_.valuesOut.shape->GetStorageShape().GetShapeSize() != 0 && !(*opParamInfo_.returnValue)) {
        OP_LOGW(opName_, "when returnValue is false, valuesOut must be null.");
    }
    return ge::GRAPH_SUCCESS;
}

// key非连续校验：通过shape计算expected stride进行校验
// PA_BBND时，只允许0轴非连续，其余轴必须连续
// 非PA_BBND时，所有轴都必须连续
ge::graphStatus LIInfoParser::CheckKeyContiguous() const
{
    if (npuArch_ != NpuArch::DAV_3510) {
        return ge::GRAPH_SUCCESS; // AutoContiguous
    }

    bool keyNonContiguous = false;
    // PA_BBND: 0轴允许非连续，从1轴开始检查；非PA_BBND: 检查shape总size
    // PA_BBND: axis 0 allows non-contiguous, check starts from axis 1
    // Non-PA_BBND: check total shape size
    if (keyStride0_ != 0 && opParamInfo_.key.shape != nullptr) {
        auto &shape = opParamInfo_.key.shape->GetStorageShape();
        uint64_t totalElements = 0;
        uint64_t shapeSize = 0;

        if (kLayout_ == DataLayout::BnBsND) {
            totalElements = keyStride1_ * static_cast<uint64_t>(shape.GetDim(DIM_IDX_ONE));
            shapeSize = static_cast<uint64_t>(shape.GetDim(DIM_IDX_ONE)) *
                        static_cast<uint64_t>(shape.GetDim(DIM_IDX_TWO)) *
                        static_cast<uint64_t>(shape.GetDim(DIM_IDX_THREE));
        } else {
            totalElements = keyStride0_ * static_cast<uint64_t>(shape.GetDim(0));
            shapeSize = context_->GetOptionalInputTensor(KEY_INDEX)->GetShapeSize();
        }

        keyNonContiguous = (shapeSize != totalElements);
    }

    OP_CHECK_IF(keyNonContiguous,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "key", "key only support non-contiguous tensor on the 0-axis with PA_BSND layout"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    }
    OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(n1Size_ != QUERY_HEAD_NUM_LIMIT_950_64 && n1Size_ != QUERY_HEAD_NUM_LIMIT_950_32 &&
                        n1Size_ != QUERY_HEAD_NUM_LIMIT_950_24 && n1Size_ != QUERY_HEAD_NUM_LIMIT_950_16 &&
                        n1Size_ != QUERY_HEAD_NUM_LIMIT_950_8,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "query", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                        "The head num of query must equal 64, 32, 24, 16 or 8"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(n1Size_ > QUERY_HEAD_NUM_LIMIT,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "query", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                        "The head num of query must be no greater than " + std::to_string(QUERY_HEAD_NUM_LIMIT)),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LiGetActualSeqLenSize(uint32_t &liSize, const gert::Tensor *liTensor,
                                             const std::string &liActualSeqLenName, const char *liOpName)
{
    liSize = static_cast<uint32_t>(liTensor->GetShapeSize());
    if (liSize == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            liOpName, liActualSeqLenName.c_str(), std::to_string(liSize).c_str(),
            "The shape size of " + liActualSeqLenName + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetActualSeqLenSize(uint32_t &liSize, const gert::Tensor *liTensor,
                                                  const std::string &liSeqLenName) const
{
    return LiGetActualSeqLenSize(liSize, liTensor, liSeqLenName, opName_);
}

ge::graphStatus LIInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (kLayout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    OP_LOGI(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "key",
                                                      ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                                                      "The head num of key must be 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "query",
                                              ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                                              "The head num of query must be a multiple of the head num of key");
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((qLayout_ == DataLayout::TND)) {
        return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "actual_seq_lengths_query");
    } else { // BSND
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        return ge::GRAPH_SUCCESS;
    }
}

static ge::graphStatus LiGetHeadDim(const TilingRequiredParaInfo &liQuery, DataLayout liQLayout, const char *liOpName,
                                    uint32_t &liHeadDim)
{
    uint32_t liDIndex = DIM_IDX_TWO;
    switch (liQLayout) {
        case DataLayout::TND:
            liDIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            liDIndex = DIM_IDX_THREE;
            break;
        default:
            OP_LOGE(liOpName, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    liHeadDim = liQuery.shape->GetStorageShape().GetDim(liDIndex);
    OP_CHECK_IF(
        liHeadDim != HEAD_DIM_LIMIT,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(liOpName, "query", ToStringRaw(liQuery.shape->GetStorageShape()).c_str(),
                                              "The head num of query only supports 128"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetHeadDim()
{
    return LiGetHeadDim(opParamInfo_.query, qLayout_, opName_, headDim_);
}

ge::graphStatus LIInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(((blockSize_ % 16 != 0) || (blockSize_ == 0) || (blockSize_ > 1024)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetShape()).c_str(),
                    "The block_size of key must be a multiple of 16 and be within the range (0, 1024]"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OP_CHECK_IF(
        (blockCount_ == 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetShape()).c_str(),
                                              "The block_count of key cannot be 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(DIM_IDX_ONE);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
            maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::BnBsND) {
        return GetS2SizeForPageAttention();
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    } else if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::ValidateInputShapesMatchQtnd()
{
    // -----------------------check BatchSize-------------------
    // bSize_ 来源于act_seq_q
    if (kLayout_ == DataLayout::TND) {
        OP_CHECK_IF((opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query and actual_seq_lengths_key",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.actualSeqLengths.tensor->GetStorageShape()),
                        "TND case actual_seq_lengths_query, actual_seq_lengths_key are " + std::to_string(bSize_) +
                            ", " + std::to_string(opParamInfo_.actualSeqLengths.tensor->GetShapeSize()) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
    } else { // kLayout_ PA_BSND
        OP_CHECK_IF((opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_) ||
                        (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, actual_seq_lengths_key and block_table",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.actualSeqLengths.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()),
                        "TND case actual_seq_lengths_query, actual_seq_lengths_key, block_table dim 0 are " +
                            std::to_string(bSize_) + ", " +
                            std::to_string(opParamInfo_.actualSeqLengths.tensor->GetShapeSize()) + ", " +
                            std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
    }
    // -----------------------check T-------------------
    uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                    (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query, weights and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "TND case input query, weights and sparse_indices dim 0 are " + std::to_string(qTsize) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0) != qTsize && (*opParamInfo_.returnValue)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and sparse_values",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.valuesOut.shape->GetStorageShape()),
                    "TND case query and sparse_values dim 0 are " + std::to_string(qTsize) + ", " +
                        std::to_string(opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::ValidateInputShapesMatchQbsnd()
{
    // -----------------------check BatchSize-------------------
    // bSize_ 来源于query
    if (kLayout_ == DataLayout::BnBsND) {
        OP_CHECK_IF((opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                        (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, actual_seq_lengths_key and block_table",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.actualSeqLengths.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()),
                        "BSND case query, actual_seq_lengths_key, block_table dim 0 are " + std::to_string(bSize_) +
                            ", " + std::to_string(opParamInfo_.actualSeqLengths.tensor->GetShapeSize()) + ", " +
                            std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::BSND) {
        OP_CHECK_IF(opParamInfo_.key.shape->GetStorageShape().GetDim(0) != bSize_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query and key",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                        "BSND case query, key dim 0 are " + std::to_string(bSize_) + ", " +
                            std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((opParamInfo_.actualSeqLengths.tensor != nullptr) &&
                        (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query and actual_seq_lengths_key",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.actualSeqLengths.tensor->GetStorageShape()),
                        "BSND case query, actual_seq_lengths_key dim 0 are " + std::to_string(bSize_) + ", " +
                            std::to_string(opParamInfo_.actualSeqLengths.tensor->GetShapeSize()) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                    (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query, weights and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "BSND case query, weights and sparse_indices dim 0 are " + std::to_string(bSize_) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0) != bSize_ && (*opParamInfo_.returnValue)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and sparse_values",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.valuesOut.shape->GetStorageShape()),
                    "BSND case query, sparse_values dim 0 are " + std::to_string(bSize_) + ", " +
                        std::to_string(opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.actualSeqLengthsQ.tensor != nullptr) &&
                    (opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize() != bSize_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and actual_seq_lengths_query",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.actualSeqLengthsQ.tensor->GetStorageShape()),
                    "BSND case query, actual_seq_lengths_query dim 0 are " + std::to_string(bSize_) + ", " +
                        std::to_string(opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize()) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check S1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != s1Size_) ||
                    (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != s1Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query, weights and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "BSND case query, weights and sparse_indices dim 1 are " + std::to_string(s1Size_) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ONE)) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != s1Size_ && (*opParamInfo_.returnValue)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "query and sparse_values",
            Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                Ops::Base::ToString(opParamInfo_.valuesOut.shape->GetStorageShape()),
            "BSND case query and sparse_values dim 1 are " + std::to_string(s1Size_) + ", " +
                std::to_string(opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE)) +
                " respectively, they must be same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::ValidateInputShapesMatch()
{
    /*
    TND:
    query [T,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [T,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize],
    out [T,N2,topk]
    ----------------------
    BSND:
    query [BatchSize,S1,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [BatchSize,S1,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize] 可选
    out [BatchSize,S1,N2,topk]
    */
    uint32_t queryWeightsN1Dim = 1;
    uint32_t outN2Dim = 1;
    if (qLayout_ == DataLayout::TND) {
        if (ValidateInputShapesMatchQtnd() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else { // qLayout_ BSND
        if (ValidateInputShapesMatchQbsnd() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and weights",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()),
                    "The head num of query and weights must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    uint32_t keyDDim = kLayout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    OP_CHECK_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) != headDim_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and key",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                    "The last dim of query and key shape must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "key and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "The head num of key and sparse_indices are " + std::to_string(n2Size_) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_ && (*opParamInfo_.returnValue)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "key and sparse_values",
            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                Ops::Base::ToString(opParamInfo_.valuesOut.shape->GetStorageShape()),
            "The head num of key and sparse_values are " + std::to_string(n2Size_) + ", " +
                std::to_string(opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim)) +
                " respectively, they must be same"),
        return ge::GRAPH_FAILED);
    // -----------------------check sparse_count-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "sparse_indices", ToStringRaw(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "The last dim of sparse_indices and sparse_count are " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1)) + ", " +
                        std::to_string(*opParamInfo_.sparseCount) + " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount &&
                 (*opParamInfo_.returnValue)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "sparse_values", ToStringRaw(opParamInfo_.valuesOut.shape->GetStorageShape()),
                    "The last dim of sparse_values and sparse_count are " +
                        std::to_string(opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim + 1)) + ", " +
                        std::to_string(*opParamInfo_.sparseCount) + " respectively, they must be same"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void LIInfoParser::GenerateInfo(LITilingInfo &liInfo)
{
    liInfo.opName = opName_;
    liInfo.platformInfo = platformInfo_;
    liInfo.opParamInfo = opParamInfo_;
    liInfo.socVersion = socVersion_;
    liInfo.npuArch = npuArch_;

    liInfo.bSize = bSize_;
    liInfo.n1Size = n1Size_;
    liInfo.n2Size = n2Size_;
    liInfo.s1Size = s1Size_;
    liInfo.s2Size = s2Size_;
    liInfo.gSize = gSize_;

    liInfo.inputQType = inputQType_;
    liInfo.inputKType = inputKType_;
    liInfo.weightsType = weightsType_;
    liInfo.outputType = outputType_;

    liInfo.blockSize = blockSize_;
    liInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    std::string layOutKeyStr(opParamInfo_.layOutKey);
    liInfo.pageAttentionFlag = layOutKeyStr == "PA_BSND" ? true : false;
    liInfo.sparseMode = *opParamInfo_.sparseMode;
    liInfo.sparseCount = *opParamInfo_.sparseCount;
    liInfo.preTokens = *opParamInfo_.preTokens;
    liInfo.nextTokens = *opParamInfo_.nextTokens;
    liInfo.returnValue = *opParamInfo_.returnValue;

    liInfo.inputQLayout = qLayout_;
    liInfo.inputKLayout = kLayout_;

    if (npuArch_ != NpuArch::DAV_3510) {
        liInfo.keyStride0 = 0; // 910无需使用stride
    } else {
        if (keyStride0_ != 0) {
            liInfo.keyStride0 = keyStride0_;
        } else if (layOutKeyStr == "PA_BSND") {
            liInfo.keyStride0 = blockSize_ * n2Size_ * headDim_;
        } else {
            liInfo.keyStride0 = 0; // 非PA无需使用stride
        }
    }
}
ge::graphStatus LIInfoParser::CheckFeatureMlaNoQuantShape() const
{
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(bSize_ <= 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName_, "batch_size", std::to_string(bSize_).c_str(),
                                                               "Batch_size should be greater than 0"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(s1Size_ <= 0 && (qLayout_ == DataLayout::BSND),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName_, "query", std::to_string(s1Size_).c_str(),
                                                               "BSND case query dim 1 should be greater than 0"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(s2Size_ <= 0 && (kLayout_ == DataLayout::BSND),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName_, "key", std::to_string(s2Size_).c_str(),
                                                               "BSND case key dim 1 should be greater than 0"),
                    return ge::GRAPH_FAILED);

        uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF(qTsize <= 0 && (qLayout_ == DataLayout::TND),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName_, "query", std::to_string(qTsize).c_str(),
                                                               "TND case query dim 0 should be greater than 0"),
                    return ge::GRAPH_FAILED);

        uint32_t kTsize = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF(kTsize <= 0 && (kLayout_ == DataLayout::TND),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName_, "key", std::to_string(kTsize).c_str(),
                                                               "TND case key dim 0 should be greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIInfoParser::ParseAndCheck(LITilingInfo &liInfo)
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
        ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch() || ge::GRAPH_SUCCESS != CheckKeyContiguous()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(liInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForLightningIndexer(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------LightningIndexerTiling类成员函数定义-----------------------
static uint64_t LiCalcWorkspaceSize(const platform_ascendc::PlatformAscendC &liPlatform, int64_t liS2Size,
                                    uint32_t liAicNum)
{
    constexpr uint32_t liMm1ResElemSize = 4;
    constexpr uint32_t liDoubleBuffer = 2;
    constexpr uint32_t liMBaseSize = 512;
    constexpr uint32_t liS2BaseSize = 512;
    constexpr uint32_t liV1ResElemSize = 4;
    constexpr uint32_t liV1ResElemType = 2;
    constexpr uint32_t liV1DecodeParamElemSize = 8;
    constexpr uint32_t liV1DecodeParamNum = 16;
    constexpr uint32_t liV1DecodeDataNum = 2;
    constexpr uint32_t liS1BaseSize = 8;
    constexpr uint32_t liTopkMaxSize = 2048;

    uint64_t liWorkspaceSize = liPlatform.GetLibApiWorkSpaceSize();
    if (liPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t li3510S1Base = 4;
        constexpr uint32_t li3510S2Base = 128;
        liWorkspaceSize +=
            li3510S1Base * ((liS2Size + li3510S2Base - 1) / li3510S2Base) * li3510S2Base * sizeof(uint32_t) * liAicNum;
    } else {
        constexpr uint32_t liMm1ResSize = liMBaseSize * liS2BaseSize;
        liWorkspaceSize += liMm1ResSize * liMm1ResElemSize * liDoubleBuffer * liAicNum;
        liWorkspaceSize +=
            liV1DecodeDataNum * liS1BaseSize * liV1ResElemType * liTopkMaxSize * liV1ResElemSize * liAicNum;
        liWorkspaceSize += liV1DecodeDataNum * liS1BaseSize * liV1DecodeParamNum * liV1DecodeParamElemSize * liAicNum;
    }
    return liWorkspaceSize;
}

ge::graphStatus LightningIndexerTiling::DoTiling(LITilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    uint64_t workspaceSize = LiCalcWorkspaceSize(ascendcPlatform, tilingInfo->s2Size, aicNum);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_sparseCount(tilingInfo->sparseCount);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_sparseMode(tilingInfo->sparseMode);
    tilingData_.set_preTokens(tilingInfo->preTokens);
    tilingData_.set_nextTokens(tilingInfo->nextTokens);
    tilingData_.set_returnValue(tilingInfo->returnValue);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_keyStride0(tilingInfo->keyStride0);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // int DT_W_FLAG, DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t weightsType = static_cast<uint32_t>(tilingInfo->weightsType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t weightTypeFlag = (weightsType == ge::DT_FLOAT) ? 1 : 0;
    uint64_t tilingKey = GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout,
                                            inputKLayout, weightTypeFlag);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1); // 1: batchmode模式

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForLightningIndexer(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("LightningIndexer", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    LITilingInfo liInfo;
    LIInfoParser LIInfoParser(context);
    if (LIInfoParser.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    LightningIndexerTiling liTiling(context);
    return liTiling.DoTiling(&liInfo);
}

// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(LightningIndexer)
    .Tiling(TilingForLightningIndexer)
    .TilingParse<LICompileInfo>(TilingPrepareForLightningIndexer);

} // namespace optiling
