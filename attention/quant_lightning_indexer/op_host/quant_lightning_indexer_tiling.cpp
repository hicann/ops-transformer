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
 * \file quant_lightning_indexer_tiling.cpp
 * \brief
 */

#include "quant_lightning_indexer_tiling.h"

#include "../op_kernel/quant_lightning_indexer_template_tiling_key.h"

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

static std::string QLIDataTypeToSerialString(ge::DataType type)
{
    const auto qliIt = DATATYPE_TO_STRING_MAP.find(type);
    if (qliIt != DATATYPE_TO_STRING_MAP.end()) {
        return qliIt->second;
    } else {
        OP_LOGE("QuantLightningIndexer", "datatype %d not support", type);
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

ge::graphStatus QLIInfoParser::CheckTensorShapes() const
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
    OP_CHECK_IF(opParamInfo_.query_dequant_scale.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_dequant_scale",
                                                         "The shape of query_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key_dequant_scale.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key_dequant_scale",
                                                         "The shape of key_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices",
                    "The shape of sparse_indices is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckTensorDescriptions() const
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
    OP_CHECK_IF(opParamInfo_.query_dequant_scale.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_dequant_scale",
                                                         "The desc of query_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key_dequant_scale.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key_dequant_scale",
                                                         "The desc of key_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices",
                    "The desc of sparse_indices is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// --------------------------QLIInfoParser类成员函数定义-------------------------------------
ge::graphStatus QLIInfoParser::CheckRequiredInOutExistence() const
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

ge::graphStatus QLIInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layOutQuery == nullptr,
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
    OP_CHECK_IF(opParamInfo_.queryQuantMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_quant_mode", "Layout_query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.keyQuantMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key_quant_mode", "Key_quant_mode is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("QuantLightningIndexer", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr,
                OP_LOGE(opName_, "GetPlatformInfo is nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_2201 && npuArch_ != NpuArch::DAV_3510) {
        OP_LOGE(opName_, "Npu Arch Version[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(
        context_->GetWorkspaceSizes(1) == nullptr,
        OP_LOGE(opName_, "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OP_LOGE(opName_, "RawTilingData got from GE context is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QLIInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsK.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengthsK.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void QLIInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    opParamInfo_.query_dequant_scale.desc = context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX);
    opParamInfo_.query_dequant_scale.shape = context_->GetInputShape(QUERY_DEQUANT_SCALE_INDEX);
    opParamInfo_.key_dequant_scale.desc = context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX);
    opParamInfo_.key_dequant_scale.shape = context_->GetInputShape(KEY_DEQUANT_SCALE_INDEX);
    GetOptionalInputParaInfo();
}

void QLIInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(quant_lightning_indexer);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(quant_lightning_indexer);
}

ge::graphStatus QLIInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE(opName_, "attrs got from GE is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    opParamInfo_.layOutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);

    opParamInfo_.queryQuantMode = attrs->GetAttrPointer<int32_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    opParamInfo_.keyQuantMode = attrs->GetAttrPointer<int32_t>(ATTR_KEY_QUANT_MODE_INDEX);
    opParamInfo_.layOutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
    opParamInfo_.keyStride0 = *(attrs->GetAttrPointer<int64_t>(ATTR_KEY_STRIDE0_INDEX));
    opParamInfo_.keyDequantScaleStride0 = *(attrs->GetAttrPointer<int64_t>(ATTR_KEY_DEQUANT_SCALE_STRIDE0_INDEX));

    if (opParamInfo_.layOutQuery != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_query is:%s", opParamInfo_.layOutQuery);
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
        OP_LOGI(context_->GetNodeName(), "preTokens is:%d", *opParamInfo_.preTokens);
    }
    if (opParamInfo_.nextTokens != nullptr) {
        OP_LOGI(context_->GetNodeName(), "nextTokens is:%d", *opParamInfo_.nextTokens);
    }
    if (opParamInfo_.queryQuantMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "query_quant_mode mode is:%d", *opParamInfo_.queryQuantMode);
    }
    if (opParamInfo_.keyQuantMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "key_quant_mode mode is:%d", *opParamInfo_.keyQuantMode);
    }
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckAttrParaInfo()
{
    OP_CHECK_IF(
        ((std::string(opParamInfo_.layOutKey) == "BNSD") || (std::string(opParamInfo_.layOutKey) == "PA_BBND")),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_key", std::string(opParamInfo_.layOutKey).c_str(),
                                              "Layout_key only supported PA_BSND, PA_BBND, BSND or TND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((std::string(opParamInfo_.layOutQuery) != "BSND") && (std::string(opParamInfo_.layOutQuery) != "TND")),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_query", std::string(opParamInfo_.layOutQuery).c_str(),
                                              "Layout_query only supported BSND or TND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((std::string(opParamInfo_.layOutKey) != "PA_BSND") &&
                 (std::string(opParamInfo_.layOutQuery)) != (std::string(opParamInfo_.layOutKey))),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "layout_query and layout_key",
                    std::string(opParamInfo_.layOutQuery) + " and " + std::string(opParamInfo_.layOutKey),
                    "When layout_key is not PA_BSND, layout_query and layout_key must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !((*opParamInfo_.sparseCount > 0) && (*opParamInfo_.sparseCount <= SPARSE_LIMIT)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_count", std::to_string(*opParamInfo_.sparseCount),
                                              "sparse_count must > 0 and <= 2048"),
        return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        auto queryType = opParamInfo_.query.desc->GetDataType();
        OP_CHECK_IF((queryType == ge::DT_HIFLOAT8) && (*opParamInfo_.sparseCount != SPARSE_LIMIT),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_count",
                                                          std::to_string(*opParamInfo_.sparseCount).c_str(),
                                                          "When query type is hifloat8, sparse_count must be 2048"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        !((*opParamInfo_.sparseMode == 0) || (*opParamInfo_.sparseMode == SPARSE_MODE_LOWER)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_mode", std::to_string(*opParamInfo_.sparseMode).c_str(),
                                              "Sparse_mode must be 0 or 3"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *opParamInfo_.preTokens != 9223372036854775807,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "pre_tokens", std::to_string(*opParamInfo_.preTokens).c_str(),
                                              "Pre_tokens must be 9223372036854775807"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *opParamInfo_.nextTokens != 9223372036854775807,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "next_tokens", std::to_string(*opParamInfo_.nextTokens).c_str(),
                                              "Next_tokens must be 9223372036854775807"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((opParamInfo_.keyStride0 < 0) && (opParamInfo_.keyStride0 != -1)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName_, "key_stride0", std::to_string(opParamInfo_.keyStride0).c_str(), "key_stride0 must > 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((opParamInfo_.keyDequantScaleStride0 < 0) && (opParamInfo_.keyDequantScaleStride0 != -1)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "key_dequant_scale_stride0",
                                                      std::to_string(opParamInfo_.keyDequantScaleStride0).c_str(),
                                                      "Key_dequant_scale_stride0 must >= 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(*opParamInfo_.queryQuantMode != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "query_quant_mode",
                                                      std::to_string(*opParamInfo_.queryQuantMode).c_str(),
                                                      "Query_quant_mode must be 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *opParamInfo_.keyQuantMode != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "key_quant_mode", std::to_string(*opParamInfo_.keyQuantMode).c_str(), "Key_quant_mode must be 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    inputQueryScaleType_ = opParamInfo_.query_dequant_scale.desc->GetDataType();
    inputKeyScaleType_ = opParamInfo_.key_dequant_scale.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();

    OP_CHECK_IF(!(inputQType_ == inputKType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "query and key",
                    QLIDataTypeToSerialString(inputQType_) + " and " + QLIDataTypeToSerialString(inputKType_),
                    "The dtype of query and key must be same"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !(inputQueryScaleType_ == inputKeyScaleType_),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "query_dequant_scale and key_dequant_scale",
            QLIDataTypeToSerialString(inputQueryScaleType_) + " and " + QLIDataTypeToSerialString(inputKeyScaleType_),
            "The dtype of query_dequant_scale and key_dequant_scale must be same"),
        return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(inputQType_ != ge::DT_FLOAT8_E4M3FN && inputQType_ != ge::DT_HIFLOAT8 && inputQType_ != ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query and key",
                        QLIDataTypeToSerialString(inputQType_) + " and " + QLIDataTypeToSerialString(inputKType_),
                        "The dtype of query and key must be float8_e4m3, hifloat8 or int8"),
                    return ge::GRAPH_FAILED);
        // hifloat8只支持bf16的weights和fp32的scale
        if (inputQType_ == ge::DT_FLOAT8_E4M3FN) {
            OP_CHECK_IF(
                (weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT) &&
                    (weightsType_ != ge::DT_FLOAT16 || inputQueryScaleType_ != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "weights, query_dequant_scale and key_dequant_scale",
                    QLIDataTypeToSerialString(weightsType_) + ", " + QLIDataTypeToSerialString(inputQueryScaleType_) +
                        " and " + QLIDataTypeToSerialString(inputKeyScaleType_),
                    "When input query and key are float8_e4m3, "
                    "the dtype of weights, query_dequant_scale and key_dequant_scale must be "
                    "(bfloat16, float32, float32) or (float16, float16, float16)"),
                return ge::GRAPH_FAILED);
        } else if (inputKType_ == ge::DT_HIFLOAT8) {
            OP_CHECK_IF(
                weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "weights, query_dequant_scale and key_dequant_scale",
                                                       QLIDataTypeToSerialString(weightsType_) + ", " +
                                                           QLIDataTypeToSerialString(inputQueryScaleType_) + " and " +
                                                           QLIDataTypeToSerialString(inputKeyScaleType_),
                                                       "When input query and key are hifloat8, the dtype of weights, "
                                                       "queryScale and keyScale must be (bfloat16, float32, float32)"),
                return ge::GRAPH_FAILED);
        } else if (inputQType_ == ge::DT_INT8) {
            OP_CHECK_IF(
                weightsType_ != ge::DT_FLOAT16 || inputQueryScaleType_ != ge::DT_FLOAT16,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "weights, query_dequant_scale and key_dequant_scale",
                                                       QLIDataTypeToSerialString(weightsType_) + ", " +
                                                           QLIDataTypeToSerialString(inputQueryScaleType_) + " and " +
                                                           QLIDataTypeToSerialString(inputKeyScaleType_),
                                                       "When input query and key are int8, the dtype of weights, "
                                                       "queryScale and keyScale must be float16"),
                return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(inputQType_ != ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query and key",
                        QLIDataTypeToSerialString(inputQType_) + " and " + QLIDataTypeToSerialString(inputKType_),
                        "The dtype of query and key must be int8"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            weightsType_ != ge::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "weights", QLIDataTypeToSerialString(weightsType_).c_str(),
                                                  "The dtype of weights must be float16"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(inputQueryScaleType_ != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query_dequant_scale and key_dequant_scale",
                        QLIDataTypeToSerialString(inputQueryScaleType_) + " and " +
                            QLIDataTypeToSerialString(inputKeyScaleType_),
                        "The dtype of query_dequant_scale and key_dequant_scale must be float16"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        outputType_ != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sparse_indices", QLIDataTypeToSerialString(outputType_).c_str(),
                                              "The dtype of sparse_indices must be int32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutQueryMap = {{"BSND", DataLayout::BSND}, {"TND", DataLayout::TND}};

    std::string layout_query(opParamInfo_.layOutQuery);
    auto QLayout_ = layoutQueryMap.find(layout_query);
    if (QLayout_ != layoutQueryMap.end()) {
        qLayout_ = QLayout_->second;
    }

    const map<string, DataLayout> layoutKeyMap = {{"BSND", DataLayout::BSND},
                                                  {"TND", DataLayout::TND},
                                                  {"PA_BSND", DataLayout::PA_BSND},
                                                  {"PA_BBND", DataLayout::PA_BSND}};
    std::string layout_key(opParamInfo_.layOutKey);
    auto KLayout = layoutKeyMap.find(layout_key);
    if (KLayout != layoutKeyMap.end()) {
        kLayout_ = KLayout->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::PA_BSND) {
        OP_CHECK_IF(
            opParamInfo_.blockTable.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                opName_, "block_table", "Layout_key only supports PA_BSND, block_table must not be null"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "actual_seq_lengths_key",
                        "Layout_key only supports PA_BSND, actual_seq_lengths_key must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "block_table", QLIDataTypeToSerialString(opParamInfo_.blockTable.desc->GetDataType()).c_str(),
                "The dtype of block_table must be int32"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "block_table", "When Layout_key is not PA_BSND, block_table must be null"),
                    return ge::GRAPH_FAILED);
    }

    if (kLayout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "actual_seq_lengths_key",
                        "When layout_key is TND, actual_seq_lengths_key must not be null"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsK.tensor != nullptr &&
                    opParamInfo_.actualSeqLengthsK.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "actual_seq_lengths_key",
                    QLIDataTypeToSerialString(opParamInfo_.actualSeqLengthsK.desc->GetDataType()).c_str(),
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
                    QLIDataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str(),
                    "The dtype of actual_seq_lengths_query must be int32"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckShapeDim()
{
    OP_CHECK_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                    (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "block_table",
                    std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum()).c_str(),
                    "The shape dim of block_table must be 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((kLayout_ == DataLayout::PA_BSND) || (kLayout_ == DataLayout::BSND)) &&
                    (opParamInfo_.key.shape->GetShape().GetDimNum() != DIM_NUM_FOUR),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "key", std::to_string(opParamInfo_.key.shape->GetShape().GetDimNum()).c_str(),
                    "The shape dim of key must be 4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kLayout_ == DataLayout::TND) && (opParamInfo_.key.shape->GetShape().GetDimNum() != DIM_NUM_THREE),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "key", std::to_string(opParamInfo_.key.shape->GetShape().GetDimNum()).c_str(),
                    "The shape dim of key must be 3"),
                return ge::GRAPH_FAILED);

    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t expectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        expectShapeDim = DIM_NUM_THREE;
    }
    OP_CHECK_IF(
        qShapeDim != expectShapeDim,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "query", std::to_string(qShapeDim).c_str(),
                                                 "The shape dim of query must be " + std::to_string(expectShapeDim)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShapeDim != expectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "sparse_indices", std::to_string(outShapeDim).c_str(),
                    "The shape dim of sparse_indices must be " + std::to_string(expectShapeDim)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == expectShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "weights", std::to_string(weightsShapeDim).c_str(),
                    "The shape dim of weights must be " + std::to_string(expectShapeDim - 1)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    }
    OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QliGetActualSeqLenSize(uint32_t &qliSize, const gert::Tensor *qliTensor,
                                              const std::string &qliActualSeqLenName, const char *qliOpName)
{
    qliSize = static_cast<uint32_t>(qliTensor->GetShapeSize());
    if (qliSize == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            qliOpName, qliActualSeqLenName.c_str(), std::to_string(qliSize).c_str(),
            "The shape size of " + qliActualSeqLenName + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                   const std::string &actualSeqLenName) const
{
    return QliGetActualSeqLenSize(size, tensor, actualSeqLenName, opName_);
}

ge::graphStatus QLIInfoParser::GetAndCheckN2Size()
{
    // PA_BSND
    if (kLayout_ == DataLayout::TND) {
        n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetShape().GetDim(DIM_IDX_ONE));
    } else {
        n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetShape().GetDim(DIM_IDX_TWO));
    }
    OP_LOGI(context_->GetNodeName(), "N2 is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                    "The head num of key must be 1"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "query",
                                              ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                                              "The head num of query can not be a multiple of the head num of key");
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;

    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(gSize_ != G_SIZE_LIMIT_950 && gSize_ != G_SIZE_LIMIT && gSize_ != G_SIZE_LIMIT_32_950 &&
                        gSize_ != G_SIZE_LIMIT_16_950 && gSize_ != G_SIZE_LIMIT_8_950,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "query", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                        "The head num of query divided by the head num of key must equal 64, 32, 24, 16 or 8"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            gSize_ > G_SIZE_LIMIT,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "query", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                "The head num of query divided by the head num of key must <= " + std::to_string(G_SIZE_LIMIT)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == DataLayout::TND) {
        return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "actual_seq_lengths_query");
    } else { // BSND
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
        OP_LOGI(context_->GetNodeName(), "b: %d, s: %d, n: %d,d :%d",
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        return ge::GRAPH_SUCCESS;
    }
}

static ge::graphStatus QliGetHeadDim(const TilingRequiredParaInfo &qliQuery, DataLayout qliQLayout,
                                     const char *qliOpName, uint32_t &qliHeadDim)
{
    uint32_t qliDIndex = DIM_IDX_TWO;
    switch (qliQLayout) {
        case DataLayout::TND:
            qliDIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            qliDIndex = DIM_IDX_THREE;
            break;
        default:
            OP_LOGE(qliOpName, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    qliHeadDim = qliQuery.shape->GetStorageShape().GetDim(qliDIndex);
    OP_CHECK_IF(qliHeadDim != HEAD_DIM_LIMIT,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(qliOpName, "query",
                                                      ToStringRaw(qliQuery.shape->GetStorageShape()).c_str(),
                                                      "The last dim of query only support 128"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetHeadDim() { return QliGetHeadDim(opParamInfo_.query, qLayout_, opName_, headDim_); }

ge::graphStatus QLIInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetShape().GetDim(1));
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(((blockSize_ % BLOCK_SIZE_FACTOR != 0) || (blockSize_ == 0) || (blockSize_ > BLOCK_SIZE_LIMIT)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetShape()).c_str(),
                    "The block_size of key must be a multiple of 16 and be within the range (0, 1024]"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetShape().GetDim(0));
    OP_CHECK_IF((blockCount_ == 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "key",
                                                      ToStringRaw(opParamInfo_.key.shape->GetShape()).c_str(),
                                                      "The block_count of key cannot be 0"),
                return ge::GRAPH_FAILED);

    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
            maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetS2SizeForBatchContinuous()
{
    std::string layout_key(opParamInfo_.layOutKey);
    if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetShape().GetDim(DIM_IDX_ONE);
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetShape().GetDim(DIM_IDX_ZERO);
    }
    OP_CHECK_IF((kLayout_ != DataLayout::BSND) && (kLayout_ != DataLayout::TND),
                OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_key", layout_key.c_str(), "BSND or TND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::PA_BSND) {
        return GetS2SizeForPageAttention();
    }
    return GetS2SizeForBatchContinuous();
}

ge::graphStatus QLIInfoParser::ValidateInputShapesMatch()
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
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于act_seq_q
        OP_CHECK_IF((kLayout_ == DataLayout::PA_BSND) &&
                        ((opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                         (opParamInfo_.blockTable.tensor != nullptr &&
                          opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, actual_seq_lengths_key and block_table",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.actualSeqLengthsK.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()),
                        "TND case actual_seq_lengths_query, actual_seq_lengths_key, block_table dim 0 are " +
                            std::to_string(bSize_) + ", " +
                            std::to_string(opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize()) + ", " +
                            std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ((kLayout_ != DataLayout::PA_BSND) && opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "query and actual_seq_lengths_key",
                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                    Ops::Base::ToString(opParamInfo_.actualSeqLengthsK.tensor->GetStorageShape()),
                "TND case actual_seq_lengths_query, actual_seq_lengths_key are " + std::to_string(bSize_) + ", " +
                    std::to_string(opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize()) +
                    " respectively, they must be same"),
            return ge::GRAPH_FAILED);
        // -----------------------check T-------------------
        uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, weights and sparse_indices",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                        "TND case query, weights, sparse_indices dim 0 are " + std::to_string(qTsize) + ", " +
                            std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        tSize_ = qTsize;
    } else {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于query
        OP_CHECK_IF(
            (kLayout_ == DataLayout::PA_BSND) &&
                ((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                 (opParamInfo_.blockTable.tensor != nullptr &&
                  opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                 (opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                 (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "query, weights, actual_seq_lengths_key, block_table and sparse_indices",
                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                    Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + ", " +
                    Ops::Base::ToString(opParamInfo_.actualSeqLengthsK.tensor->GetStorageShape()) + ", " +
                    Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()) + " and " +
                    Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                "BSND case query, weights, actual_seq_lengths_key, block_table, sparse_indices dim 0 are " +
                    std::to_string(bSize_) + ", " +
                    std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                    std::to_string(opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize()) + ", " +
                    std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)) + ", " +
                    std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                    " respectively, they must be same"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (kLayout_ != DataLayout::PA_BSND) && ((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                                                  (opParamInfo_.actualSeqLengthsK.tensor != nullptr &&
                                                   opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                                                  (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "query, weights, actual_seq_lengths_key and sparse_indices",
                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                    Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + ", " +
                    Ops::Base::ToString(opParamInfo_.actualSeqLengthsK.tensor->GetStorageShape()) + " and " +
                    Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                "BSND case query, weights, actual_seq_lengths_key, sparse_indices dim 0 are " +
                    std::to_string(bSize_) + ", " +
                    std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                    std::to_string(opParamInfo_.actualSeqLengthsK.tensor->GetShapeSize()) + ", " +
                    std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
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
        OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, weights and sparse_indices",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                        "BSND case query, weights, sparse_indices dim 1 are " + std::to_string(s1Size_) + ", " +
                            std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(1)) + ", " +
                            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
        tSize_ = bSize_ * s1Size_;
    }
    // -----------------------check N1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and weights",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()),
                    "The head num of query, weights are " + std::to_string(n1Size_) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(1)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    OP_CHECK_IF(
        ((kLayout_ != DataLayout::TND && opParamInfo_.key.shape->GetShape().GetDim(DIM_IDX_THREE) != headDim_) ||
         (kLayout_ == DataLayout::TND && opParamInfo_.key.shape->GetShape().GetDim(DIM_IDX_TWO) != headDim_)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query and key",
                                               Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) +
                                                   " and " +
                                                   Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                                               "Query, key shape last dim must be same"),
        return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "key and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "The head num of key and output sparse_indices must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check sparse_count-------------------
    OP_CHECK_IF(
        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "sparse_indices and sparse_count",
                                               Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()) +
                                                   " and " + std::to_string(*opParamInfo_.sparseCount),
                                               "The last dim of sparse_indices and sparse_count must be same"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIInfoParser::CheckScaleShape()
{
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t kShapeDim = opParamInfo_.key.shape->GetShape().GetDimNum();
    uint32_t qDequantScaleShapeDim = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDimNum();
    uint32_t kDequantScaleShapeDim = opParamInfo_.key_dequant_scale.shape->GetShape().GetDimNum();
    OP_CHECK_IF(qDequantScaleShapeDim != (qShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "query_dequant_scale", std::to_string(qDequantScaleShapeDim),
                    "The shape dim of query_dequant_scale must be " + std::to_string(qShapeDim - 1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDequantScaleShapeDim != (kShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "key_dequant_scale", std::to_string(kDequantScaleShapeDim),
                    "The shape dim of key_dequant_scale must be " + std::to_string(kShapeDim - 1)),
                return ge::GRAPH_FAILED);
    // check q scale
    for (uint32_t i = 0; i < (qShapeDim - 1); i++) {
        uint32_t dimValueQueryScale = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(i);
        uint32_t dimValueQuery = opParamInfo_.query.shape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(
            dimValueQueryScale != dimValueQuery,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "query and query_dequant_scale",
                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                    Ops::Base::ToString(opParamInfo_.query_dequant_scale.shape->GetStorageShape()),
                "Query_dequant_scale's shape[" + std::to_string(i) + "] " + std::to_string(dimValueQueryScale) +
                    " and query's shape[" + std::to_string(i) + "] " + std::to_string(dimValueQuery) + " is not same"),
            return ge::GRAPH_FAILED);
    }
    // check k scale
    for (uint32_t i = 0; i < (kShapeDim - 1); i++) {
        uint32_t dimValueKeyScale = opParamInfo_.key_dequant_scale.shape->GetShape().GetDim(i);
        uint32_t dimValueKey = opParamInfo_.key.shape->GetShape().GetDim(i);
        OP_CHECK_IF(
            dimValueKeyScale != dimValueKey,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "key and key_dequant_scale",
                Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                    Ops::Base::ToString(opParamInfo_.key_dequant_scale.shape->GetStorageShape()),
                "Key_dequant_scale's shape[" + std::to_string(i) + "] " + std::to_string(dimValueKeyScale) +
                    " and key's shape[" + std::to_string(i) + "] " + std::to_string(dimValueKey) + " is not same"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// key非连续校验：通过shape计算expected stride进行校验
// PA_BSND时，只允许0轴非连续，其余轴必须连续
// 非PA_BSND时，所有轴都必须连续
ge::graphStatus QLIInfoParser::CheckContiguous()
{
    if (npuArch_ != NpuArch::DAV_3510) {
        return ge::GRAPH_SUCCESS;
    }

    std::vector<uint32_t> keyStridesVec_;
    std::vector<uint32_t> keyDequantScaleStridesVec_;

    auto keyStrides = context_->GetDynamicInputStride(KEY_INDEX, 0);
    auto keyDequantScaleStrides = context_->GetDynamicInputStride(KEY_DEQUANT_SCALE_INDEX, 0);
    if (keyStrides != nullptr && keyStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < keyStrides->GetDimNum(); i++) {
            keyStridesVec_.push_back(keyStrides->GetStride(i));
        }
    }
    if (keyDequantScaleStrides != nullptr && keyDequantScaleStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < keyDequantScaleStrides->GetDimNum(); i++) {
            keyDequantScaleStridesVec_.push_back(keyDequantScaleStrides->GetStride(i));
        }
    }

    bool keyNonContiguous = false;
    bool scaleNonContiguous = false;
    // PA_BSND: 0轴允许非连续，从1轴开始检查；非PA_BSND: 从0轴开始检查
    // PA_BSND: axis 0 allows non-contiguous, check starts from axis 1
    // Non-PA_BSND: check starts from axis 0
    size_t checkStartIdx = (kLayout_ == DataLayout::PA_BSND) ? 1 : 0;
    if (!keyStridesVec_.empty() && opParamInfo_.key.shape != nullptr) {
        auto &shape = opParamInfo_.key.shape->GetStorageShape();
        std::vector<uint32_t> expectedStrides;
        if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BSND) {
            expectedStrides = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3), shape.GetDim(2) * shape.GetDim(3),
                               shape.GetDim(3), 1};
        } else if (kLayout_ == DataLayout::TND) {
            expectedStrides = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
        }
        for (size_t i = checkStartIdx; i < expectedStrides.size(); ++i) {
            if (i < keyStridesVec_.size() && keyStridesVec_[i] != expectedStrides[i]) {
                keyNonContiguous = true;
                break;
            }
        }
    }
    if (!keyDequantScaleStridesVec_.empty() && opParamInfo_.key_dequant_scale.shape != nullptr) {
        auto &shape = opParamInfo_.key_dequant_scale.shape->GetStorageShape();
        std::vector<uint32_t> expectedStrides;
        if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BSND) {
            expectedStrides = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
        } else if (kLayout_ == DataLayout::TND) {
            expectedStrides = {shape.GetDim(1), 1};
        }
        for (size_t i = checkStartIdx; i < expectedStrides.size(); ++i) {
            if (i < keyDequantScaleStridesVec_.size() && keyDequantScaleStridesVec_[i] != expectedStrides[i]) {
                scaleNonContiguous = true;
                break;
            }
        }
    }
    OP_CHECK_IF(
        keyNonContiguous || scaleNonContiguous,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "key and key_dequant_scale",
            "Key and key_dequant_scale only supports non-contiguous tensor on the 0-axis in PA scenarios"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QLIInfoParser::GenerateInfo(QLITilingInfo &QLIInfo)
{
    QLIInfo.opName = opName_;
    QLIInfo.platformInfo = platformInfo_;
    QLIInfo.opParamInfo = opParamInfo_;
    QLIInfo.npuArch = npuArch_;

    QLIInfo.bSize = bSize_;
    QLIInfo.tSize = tSize_;
    QLIInfo.n1Size = n1Size_;
    QLIInfo.n2Size = n2Size_;
    QLIInfo.s1Size = s1Size_;
    QLIInfo.s2Size = s2Size_;
    QLIInfo.gSize = gSize_;

    QLIInfo.inputQType = inputQType_;
    QLIInfo.inputKType = inputKType_;
    QLIInfo.outputType = outputType_;

    QLIInfo.blockSize = blockSize_;
    QLIInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    QLIInfo.pageAttentionFlag = (kLayout_ == DataLayout::PA_BSND);
    QLIInfo.sparseMode = *opParamInfo_.sparseMode;
    QLIInfo.sparseCount = *opParamInfo_.sparseCount;
    QLIInfo.preTokens = *opParamInfo_.preTokens;
    QLIInfo.nextTokens = *opParamInfo_.nextTokens;

    if (opParamInfo_.keyStride0 != -1) {
        QLIInfo.keyStride0 = opParamInfo_.keyStride0;
    } else {
        QLIInfo.keyStride0 = blockSize_ * n2Size_ * headDim_;
    }
    if (opParamInfo_.keyDequantScaleStride0 != -1) {
        QLIInfo.keyDequantScaleStride0 = opParamInfo_.keyDequantScaleStride0;
    } else {
        QLIInfo.keyDequantScaleStride0 = blockSize_;
    }

    QLIInfo.inputQLayout = qLayout_;
    QLIInfo.inputKLayout = kLayout_;
}

ge::graphStatus QLIInfoParser::ParseAndCheck(QLITilingInfo &QLIInfo)
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
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch() || ge::GRAPH_SUCCESS != CheckScaleShape()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckContiguous()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(QLIInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForQuantLightningIndexer(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantLightningIndexerTiling类成员函数定义-----------------------
static uint64_t QliCalcWorkspaceSize(const platform_ascendc::PlatformAscendC &qliPlatform, int64_t qliS2Size,
                                     uint32_t qliAicNum, ge::DataType inputQType)
{
    constexpr uint32_t qliMm1ResElemSize = 4;
    constexpr uint32_t qliDoubleBuffer = 2;
    constexpr uint32_t qliMBaseSize = 512;
    constexpr uint32_t qliS2BaseSize = 512;
    constexpr uint32_t qliV1ResElemSize = 4;
    constexpr uint32_t qliV1ResElemType = 2;
    constexpr uint32_t qliV1DecodeParamElemSize = 8;
    constexpr uint32_t qliV1DecodeParamNum = 16;
    constexpr uint32_t qliV1DecodeDataNum = 2;
    constexpr uint32_t qliS1BaseSize = 8;
    constexpr uint32_t qliTopkMaxSize = 2048;

    uint64_t qliWorkspaceSize = qliPlatform.GetLibApiWorkSpaceSize();
    if (qliPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t qli3510S1Base = 4;
        constexpr uint32_t qli3510S2Base = 128;
        uint32_t scoreElemSize = (inputQType == ge::DT_INT8) ? sizeof(uint32_t) : sizeof(uint16_t);
        qliWorkspaceSize += qli3510S1Base * ((qliS2Size + qli3510S2Base - 1) / qli3510S2Base) * qli3510S2Base *
                            scoreElemSize * qliAicNum;
    } else {
        constexpr uint32_t qliMm1ResSize = qliMBaseSize * qliS2BaseSize;
        qliWorkspaceSize += qliMm1ResSize * qliMm1ResElemSize * qliDoubleBuffer * qliAicNum;
        qliWorkspaceSize +=
            qliV1DecodeDataNum * qliS1BaseSize * qliV1ResElemType * qliTopkMaxSize * qliV1ResElemSize * qliAicNum;
        qliWorkspaceSize +=
            qliV1DecodeDataNum * qliS1BaseSize * qliV1DecodeParamNum * qliV1DecodeParamElemSize * qliAicNum;
    }
    return qliWorkspaceSize;
}

ge::graphStatus QuantLightningIndexerTiling::DoTiling(QLITilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    uint64_t workspaceSize = QliCalcWorkspaceSize(ascendcPlatform, tilingInfo->s2Size, aicNum, tilingInfo->inputQType);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_tSize(tilingInfo->tSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_sparseCount(tilingInfo->sparseCount);
    tilingData_.set_keyStride0(tilingInfo->keyStride0);
    tilingData_.set_keyDequantScaleStride0(tilingInfo->keyDequantScaleStride0);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_sparseMode(tilingInfo->sparseMode);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1); // 1: batchmode模式

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForQuantLightningIndexer(gert::TilingContext *context)
{
    OP_CHECK_IF(
        context == nullptr,
        OP_LOGE("QuantLightningIndexer", "Tilingcontext is null"),
        return ge::GRAPH_FAILED);
    QLITilingInfo QLIInfo;
    QLIInfoParser QLIInfoParser(context);
    if (QLIInfoParser.ParseAndCheck(QLIInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantLightningIndexerTiling QLITiling(context);
    return QLITiling.DoTiling(&QLIInfo);
}

// --------------------------Tiling及函数TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantLightningIndexer)
    .Tiling(TilingForQuantLightningIndexer)
    .TilingParse<QLICompileInfo>(TilingPrepareForQuantLightningIndexer);

} // namespace optiling
