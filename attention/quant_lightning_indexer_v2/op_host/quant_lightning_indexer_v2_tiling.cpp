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
 * \file quant_lightning_indexer_v2_tiling.cpp
 * \brief
 */

#include "quant_lightning_indexer_v2_tiling.h"

#include "../op_kernel/quant_lightning_indexer_v2_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {

static const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
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
    {ge::DT_UINT2, "DT_UINT2"},                   // dt_variant type
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0"},
    {ge::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"}};

std::string QLIV2DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("QLIV2DataTypeToSerialString ", "datatype %d not support", type);
        return "UNDEFINED";
    }
}
// --------------------------QLIV2InfoParser类成员函数定义-------------------------------------
ge::graphStatus QLIV2InfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor key"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor key"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor weights"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor weights"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query_dequant_scale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor query_dequant_scale"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query_dequant_scale.desc == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor query_dequant_scale"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key_dequant_scale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor key_dequant_scale"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key_dequant_scale.desc == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor key_dequant_scale"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor output"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor output"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layOutQuery == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr layout_query"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.layOutKey == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr layout_key"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseCount == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr sparse_count"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr sparse_mode"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.quantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "query_quant_mode"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("QuantLightningIndexerV2", "opName got from TilingContext");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

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
                OP_LOGE(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QLIV2InfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.cuSeqLensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqLensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqLensK.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_K_INDEX);
    opParamInfo_.cuSeqLensK.desc = context_->GetOptionalInputDesc(CU_SEQLENS_K_INDEX);
    opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedK.tensor = context_->GetOptionalInputTensor(SEQUSED_K_INDEX);
    opParamInfo_.sequsedK.desc = context_->GetOptionalInputDesc(SEQUSED_K_INDEX);
    opParamInfo_.cmpResidualK.tensor = context_->GetOptionalInputTensor(CMP_RESIDUAL_K_INDEX);
    opParamInfo_.cmpResidualK.desc = context_->GetOptionalInputDesc(CMP_RESIDUAL_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.outputIdxOffset.tensor = context_->GetOptionalInputTensor(OUTPUT_IDX_OFFSET_INDEX);
    opParamInfo_.outputIdxOffset.desc = context_->GetOptionalInputDesc(OUTPUT_IDX_OFFSET_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void QLIV2InfoParser::GetInputParaInfo()
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

void QLIV2InfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(SPARSE_INDICES_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(SPARSE_INDICES_INDEX);
    opParamInfo_.sparseValues.desc = context_->GetOutputDesc(SPARSE_VALUES_INDEX);
    opParamInfo_.sparseValues.shape = context_->GetOutputShape(SPARSE_VALUES_INDEX);
}

ge::graphStatus QLIV2InfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");

    opParamInfo_.quantMode = attrs->GetAttrPointer<int32_t>(ATTR_QUANT_MODE_INDEX);
    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int32_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    opParamInfo_.layOutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_TOPK_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_MASK_MODE_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<int32_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.returnValue = attrs->GetAttrPointer<int32_t>(ATTR_RETURN_VALUE_INDEX);
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
    if (opParamInfo_.cmpRatio != nullptr) {
        OP_LOGI(context_->GetNodeName(), "cmpRatio is:%d", *opParamInfo_.cmpRatio);
    }
    if (opParamInfo_.returnValue != nullptr) {
        OP_LOGI(context_->GetNodeName(), "returnValue is:%s", *opParamInfo_.returnValue ? "true" : "false");
    }
    if (opParamInfo_.maxSeqlenQ != nullptr) {
        OP_LOGI(context_->GetNodeName(), "maxSeqlenQ  is:%d", *opParamInfo_.maxSeqlenQ);
    }
    if (opParamInfo_.quantMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "query_quant_mode mode is:%d", *opParamInfo_.quantMode);
    }
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckAttrParaInfo()
{
    std::string layout_key(opParamInfo_.layOutKey);
    std::string layout_query(opParamInfo_.layOutQuery);

    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF((std::string(opParamInfo_.layOutKey) != "PA_BBND"),
                    OP_LOGE(opName_,
                            "input attr layout_key only supported PA_BBND,"
                            "but now layout_key is %s.",
                            layout_key.c_str()),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(((std::string(opParamInfo_.layOutKey) != "PA_BBND") &&
                     (std::string(opParamInfo_.layOutKey) != "BSND") && (std::string(opParamInfo_.layOutKey) != "TND")),
                    OP_LOGE(opName_, "input attr layout_key only supported PA_BBND, BSND or TND"),
                    return ge::GRAPH_FAILED);
    }

    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(!((*opParamInfo_.sparseCount > 0) && (*opParamInfo_.sparseCount <= SPARSE_LIMIT)),
                    OP_LOGE(opName_, "input attr sparse_count must > 0 and <= %d, but now sparse_count is %d",
                            SPARSE_LIMIT, *opParamInfo_.sparseCount),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((*opParamInfo_.cmpRatio <= 0) || (*opParamInfo_.cmpRatio > 128) ||
                        ((*opParamInfo_.cmpRatio & (*opParamInfo_.cmpRatio - 1)) != 0),
                    OP_LOGE(opName_,
                            "input attr cmpRatio must > 0 and <= 128 and should be powers of 2,"
                            " but now cmpRatio is %ld.",
                            *opParamInfo_.cmpRatio),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(!((*opParamInfo_.sparseCount > 0) && (*opParamInfo_.sparseCount <= SPARSE_LIMIT)),
                    OP_LOGE(opName_, "input attr sparse_count must > 0 and <= %d, but now sparse_count is %d",
                            SPARSE_LIMIT, *opParamInfo_.sparseCount),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((*opParamInfo_.cmpRatio <= 0) || (*opParamInfo_.cmpRatio > 128),
                    OP_LOGE(opName_, "input attr cmpRatio must > 0 and <= 128, but now cmpRatio is %ld.",
                            *opParamInfo_.cmpRatio),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(((std::string(opParamInfo_.layOutQuery) != "BSND") &&
                (std::string(opParamInfo_.layOutQuery) != "TND")),
                OP_LOGE(opName_, "input attr layout_query only supported BSND or TND."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(((std::string(opParamInfo_.layOutKey) != "PA_BBND") &&
                 (std::string(opParamInfo_.layOutQuery)) != (std::string(opParamInfo_.layOutKey))),
                OP_LOGE(opName_,
                        "outside of PA, input attr layout_query and input attr layout_key must be the same,"
                        "but now layout_key is %s, layout_query is %s.",
                        layout_key.c_str(), layout_query.c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!((*opParamInfo_.sparseMode == 0) || (*opParamInfo_.sparseMode == SPARSE_MODE_LOWER)),
                OP_LOGE(opName_, "input attr sparse_mode only supported 0 or 3, but now sparseMode is %d.",
                        *opParamInfo_.sparseMode),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(*opParamInfo_.quantMode != 2, OP_LOGE(opName_, "input attr quant_mode only supported 2."),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF((*opParamInfo_.quantMode != QUANT_MODE_FP8) && (*opParamInfo_.quantMode != QUANT_MODE_INT8) &&
                    (*opParamInfo_.quantMode != QUANT_MODE_MXFP8) &&
                    (*opParamInfo_.quantMode != QUANT_MODE_HIFLOAT8) &&
                    (*opParamInfo_.quantMode != QUANT_MODE_MXFP4),
                    OP_LOGE(opName_, "input attr quant_mode only supported 1, 2, 3, 4 and 5."),
                    return ge::GRAPH_FAILED);
    }

    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(*opParamInfo_.returnValue, OP_LOGE(opName_, "input attr returnValue only supported False."),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF((*opParamInfo_.returnValue != 0) && (*opParamInfo_.returnValue != 1),
                    OP_LOGE(opName_,
                            "input attr return_value only support 0 or 1, "
                            "but now return_value is %ld.",
                            *opParamInfo_.returnValue),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((*opParamInfo_.maxSeqlenQ < -1),
                OP_LOGE(opName_,
                        "input attr max_seqlens_q must >= -1, "
                        "but now max_seqlen_q is %d.",
                        *opParamInfo_.maxSeqlenQ),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetOpParaInfo()
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

ge::graphStatus QLIV2InfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    inputQueryScaleType_ = opParamInfo_.query_dequant_scale.desc->GetDataType();
    inputKeyScaleType_ = opParamInfo_.key_dequant_scale.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    valuesOutType_ = opParamInfo_.sparseValues.desc->GetDataType();

    OP_CHECK_IF(!(inputQType_ == inputKType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "query and key",
                    QLIV2DataTypeToSerialString(inputQType_) + " and " + QLIV2DataTypeToSerialString(inputKType_),
                    "The data types of the input query and key must be the same"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!(inputQueryScaleType_ == inputKeyScaleType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "query_dequant_scale and key_dequant_scale",
                    QLIV2DataTypeToSerialString(inputQueryScaleType_) + " and " +
                        QLIV2DataTypeToSerialString(inputKeyScaleType_),
                    "The data types of the input query_dequant_scale and key_dequant_scale must be the same"),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(inputQType_ != ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query and key",
                        QLIV2DataTypeToSerialString(inputQType_) + " and " + QLIV2DataTypeToSerialString(inputKType_),
                        "The data types of the input query and key must be int8"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputQueryScaleType_ != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query_dequant_scale and key_dequant_scale",
                        QLIV2DataTypeToSerialString(inputQueryScaleType_) + " and " +
                            QLIV2DataTypeToSerialString(inputKeyScaleType_),
                        "The data types of the input query_dequant_scale and key_dequant_scale must be float16"),
                    return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        ge::DataType expectQType = ge::DT_FLOAT8_E4M3FN;
        ge::DataType expectScaleType = ge::DT_FLOAT;
        if (*opParamInfo_.quantMode == QUANT_MODE_MXFP8) {
            expectScaleType = ge::DT_FLOAT8_E8M0;
        } else if (*opParamInfo_.quantMode == QUANT_MODE_HIFLOAT8) {
            expectQType = ge::DT_HIFLOAT8;
        } else if (*opParamInfo_.quantMode == QUANT_MODE_MXFP4) {
            expectQType = ge::DT_FLOAT4_E2M1;
            expectScaleType = ge::DT_FLOAT8_E8M0;
        } else if (*opParamInfo_.quantMode == QUANT_MODE_INT8) {
            expectQType = ge::DT_INT8;
            expectScaleType = ge::DT_FLOAT16;
        }
        OP_CHECK_IF(inputQType_ != expectQType,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query and key",
                        QLIV2DataTypeToSerialString(inputQType_) + " and " + QLIV2DataTypeToSerialString(inputKType_),
                        "The data types of the input query and key must match quant_mode"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputQueryScaleType_ != expectScaleType,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query_dequant_scale and key_dequant_scale",
                        QLIV2DataTypeToSerialString(inputQueryScaleType_) + " and " +
                            QLIV2DataTypeToSerialString(inputKeyScaleType_),
                        "The data types of the input query_dequant_scale and key_dequant_scale must match quant_mode"),
                    return ge::GRAPH_FAILED);
    }

    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(
            weightsType_ != ge::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "weights", QLIV2DataTypeToSerialString(weightsType_).c_str(),
                                                  "The data types of the input weights must be float16"),
            return ge::GRAPH_FAILED);
    } else if (npuArch_ == NpuArch::DAV_3510) {
        if (inputQType_ == ge::DT_INT8) {
            OP_CHECK_IF(weightsType_ != ge::DT_FLOAT16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "weights",
                    QLIV2DataTypeToSerialString(weightsType_).c_str(),
                    "When data types of input query is int8, the data types of the input weights must be float16"),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(weightsType_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "weights",
                    QLIV2DataTypeToSerialString(weightsType_).c_str(),
                    "When data types of input query is not int8, "
                    "the data types of the input weights must be float"),
                return ge::GRAPH_FAILED);
        }
    }

    OP_CHECK_IF(outputType_ != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sparse_indices",
                                                      QLIV2DataTypeToSerialString(outputType_).c_str(),
                                                      "The data types of the output sparse_indices must be int32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valuesOutType_ != ge::DT_BF16,
                OP_LOGE(opName_, "The data types of the output sparse_values must be bfloat16."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetQueryKeyAndOutLayout()
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
                                                  {"PA_BSND", DataLayout::PA_BBND},
                                                  {"PA_BBND", DataLayout::PA_BBND}};
    std::string layout_key(opParamInfo_.layOutKey);
    auto KLayout = layoutKeyMap.find(layout_key);
    if (KLayout != layoutKeyMap.end()) {
        kLayout_ = KLayout->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetAndCheckOptionalInput()
{
    // =============== K 侧校验 ===============
    if (kLayout_ == DataLayout::PA_BBND) {
        // PA_BBND: block_table 必传, seqused_k 必传, cu_seqlens_k 不传
        OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "input block_table", "key layout is PA_BBND, input block_table must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.sequsedK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "input seqused_k",
                                                             "key layout is PA_BBND, input seqused_k must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.cuSeqLensK.tensor != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "input cu_seqlens_k",
                                                     "key layout is PA_BBND, input cu_seqlens_k must not be provided"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "block_table",
                        QLIV2DataTypeToSerialString(opParamInfo_.blockTable.desc->GetDataType()).c_str(),
                        "input block_table data type only support int32"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "seqused_k", QLIV2DataTypeToSerialString(opParamInfo_.sequsedK.desc->GetDataType()).c_str(),
                "input seqused_k data type only support int32"),
            return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::TND) {
        // TND: cu_seqlens_k 必传, seqused_k 可选, cu_seqlens_k 不传
        OP_CHECK_IF(opParamInfo_.cuSeqLensK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "input cu_seqlens_k", "when layout_key is TND, input cu_seqlens_k must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cuSeqLensK.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "cu_seqlens_k",
                        QLIV2DataTypeToSerialString(opParamInfo_.cuSeqLensK.desc->GetDataType()).c_str(),
                        "input cu_seqlens_k data type only support int32"),
                    return ge::GRAPH_FAILED);
        // seqused_k 可选 - 仅校验数据类型
        if (opParamInfo_.sequsedK.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "seqused_k",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedK.desc->GetDataType()).c_str(),
                            "input seqused_k data type only support int32"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // BSND: cu_seqlens_k 不传, seqused_k 可选
        OP_CHECK_IF(opParamInfo_.cuSeqLensK.tensor != nullptr,
                    OP_LOGE(opName_, "when layout_key is BSND, input cu_seqlens_k must not be provided"),
                    return ge::GRAPH_FAILED);
        if (opParamInfo_.sequsedK.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "seqused_k",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedK.desc->GetDataType()).c_str(),
                            "input seqused_k data type only support int32"),
                        return ge::GRAPH_FAILED);
        }
    }

    // block_table 非 PA 场景必须为空
    if (kLayout_ != DataLayout::PA_BBND) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "block_table", Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()),
                        "key layout is not PA_BBND, input block_table must be null"),
                    return ge::GRAPH_FAILED);
    }

    // =============== cmpResidualK 校验 ===============
    // cmpRatio 不等于 1 且 sparseMode 不等于 0 时 cmpResidualK 必传
    if (opParamInfo_.cmpRatio != nullptr && *opParamInfo_.cmpRatio != 1 && opParamInfo_.sparseMode != nullptr &&
        *opParamInfo_.sparseMode != 0) {
        OP_CHECK_IF(opParamInfo_.cmpResidualK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "input cmp_residual_k",
                        "cmp_ratio is not 1 and sparse_mode is not 0, input cmp_residual_k must not be null"),
                    return ge::GRAPH_FAILED);
        // cmpResidualK 传入时校验维度 & 数据类型
        if (qLayout_ == DataLayout::BSND) {
            OP_CHECK_IF(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) !=
                            opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize(),
                        OP_LOGE(opName_, "when layout_q is BSND, input cmp_residual_k'shape must be (B,)"),
                        return ge::GRAPH_FAILED);
        } else if (qLayout_ == DataLayout::TND) {
            OP_CHECK_IF(opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize() !=
                            opParamInfo_.cuSeqLensQ.tensor->GetStorageShape().GetShapeSize() - 1,
                        OP_LOGE(opName_, "when layout_q is TND, input cmp_residual_k's shape "
                                         "must equal input cu_seqlens_q's shape - 1"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(opParamInfo_.cmpResidualK.tensor != nullptr,
                    OP_LOGE(opName_, "cmp_ratio is 1 or sparse_mode is 0, "
                                     "input cmp_residual_k must be null"),
                    return ge::GRAPH_FAILED);
    }
    // cmpResidualK 传入时校验数据类型
    if (opParamInfo_.cmpResidualK.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpResidualK.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "cmp_residual_k",
                        QLIV2DataTypeToSerialString(opParamInfo_.cmpResidualK.desc->GetDataType()).c_str(),
                        "input cmp_residual_k data type only support int32"),
                    return ge::GRAPH_FAILED);
    }

    // =============== Q 侧校验 ===============
    if (qLayout_ == DataLayout::TND) {
        // TND: cu_seqlens_q 必传, seqused_q 可选
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "input cu_seqlens_q", "when layout_query is TND, input cu_seqlens_q must not be null"),
                    return ge::GRAPH_FAILED);
        if (kLayout_ == DataLayout::PA_BBND) {
            // k为PA_BBND必传sequsedK, 用sequsedK的维度校验
            OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape().GetShapeSize() !=
                            opParamInfo_.sequsedK.tensor->GetStorageShape().GetShapeSize() + 1,
                        OP_LOGE(opName_, "when layout_q is TND and layout_k is PA_BBND, "
                                         "input cu_seqlens_q's shape must equal seqused_k's shape + 1"),
                        return ge::GRAPH_FAILED);
        } else if (kLayout_ == DataLayout::TND) {
            // q、k都为TND, cuSeqlensQ与cuSeqlensK维度一致校验
            OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape().GetShapeSize() !=
                            opParamInfo_.cuSeqLensK.tensor->GetStorageShape().GetShapeSize(),
                        OP_LOGE(opName_, "when layout_q is TND and layout_k is TND, "
                                         "input cu_seqlens_q'shape must equal cu_seqlens_k's shape"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        opName_, "cu_seqlens_q",
                        QLIV2DataTypeToSerialString(opParamInfo_.cuSeqLensK.desc->GetDataType()).c_str(),
                        "input cu_seqlens_q data type only support int32"),
                    return ge::GRAPH_FAILED);
        // seqused_q 可选 - 仅校验数据类型
        if (opParamInfo_.sequsedQ.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.sequsedQ.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "seqused_q",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedQ.desc->GetDataType()).c_str(),
                            "input seqused_q data type only support int32"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // BSND: cu_seqlens_q 不传, seqused_q 可选
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor != nullptr,
                    OP_LOGE(opName_, "when layout_query is BSND, input cu_seqlens_q must not be provided"),
                    return ge::GRAPH_FAILED);
        if (opParamInfo_.sequsedQ.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.sequsedQ.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "seqused_q",
                            QLIV2DataTypeToSerialString(opParamInfo_.sequsedQ.desc->GetDataType()).c_str(),
                            "input seqused_q data type only support int32"),
                        return ge::GRAPH_FAILED);
        }
    }
    if (npuArch_ == NpuArch::DAV_3510) {
        if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.outputIdxOffset.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE(opName_, "input output_idx_offset data type only support int32"),
                        return ge::GRAPH_FAILED);
        }
    }
    // metadata 必传
    OP_CHECK_IF(opParamInfo_.metadata.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "input metadata", "input metadata must not be null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckShapeDim()
{
    OP_CHECK_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                    (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "block_table",
                    std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((kLayout_ == DataLayout::PA_BBND) || (kLayout_ == DataLayout::BSND)) &&
                    (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR),
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "key", std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDimNum()).c_str(), "4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (kLayout_ == DataLayout::TND) && (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName_, "key", std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDimNum()).c_str(), "3"),
        return ge::GRAPH_FAILED);

    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t expectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        expectShapeDim = DIM_NUM_THREE;
    }
    OP_CHECK_IF(qShapeDim != expectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "query", std::to_string(qShapeDim).c_str(),
                                             std::to_string(expectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShapeDim != expectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "sparse_indices", std::to_string(outShapeDim).c_str(),
                                             std::to_string(expectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
        uint32_t outputIdxOffsetShapeDim = opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDimNum();
        OP_CHECK_IF((outputIdxOffsetShapeDim != expectShapeDim - 1),
                    OP_LOGE(opName_, "the dim num of output_idx_offset's shape should be %u, but now is %u",
                            expectShapeDim - 1, outputIdxOffsetShapeDim),
                    return ge::GRAPH_FAILED);
    }
    if (npuArch_ == NpuArch::DAV_3510 && *opParamInfo_.returnValue == 1) {
        uint32_t sparseValuesShapeDim = opParamInfo_.sparseValues.shape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(sparseValuesShapeDim != expectShapeDim,
                    OP_LOGE(opName_, "the dim num of sparse_values's shape should be %u, but now is %u", expectShapeDim,
                            sparseValuesShapeDim),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!(weightsShapeDim == expectShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "weights", std::to_string(weightsShapeDim).c_str(),
                                             std::to_string(expectShapeDim - 1).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetN1Size()
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

ge::graphStatus QLIV2InfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                     const std::string &actualSeqLenName) const
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= 0) {
        OP_LOGE(opName_, "%s's shape size is %u, it should be greater than 0.", actualSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetAndCheckN2Size()
{
    // PA_BBND
    if (kLayout_ == DataLayout::TND) {
        n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    } else {
        n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    }
    OP_LOGI(context_->GetNodeName(), "N2 is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                    "N2 only support 1"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query and key",
                                               Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) +
                                                   " and " +
                                                   Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                                               "input query's head_num can not be a multiple of key's head_num");
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;

    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(gSize_ != G_SIZE_LIMIT && gSize_ != G_SIZE_LIMIT_32_950,
                    OP_LOGE(opName_, "input query's head_num divided by input key's head_num must equal 64 or 32"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(gSize_ != G_SIZE_LIMIT,
                    OP_LOGE(opName_, "N1 is %u, N2 is %u, N1 divided by N2 must equal 64.", n1Size_, n2Size_),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, 以cu_seqlens_q的shape[0]-1为B轴大小
    if (qLayout_ == DataLayout::BSND) {
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
        OP_LOGI(context_->GetNodeName(), "b: %d, s: %d, n: %d,d :%d",
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO),
                opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        return ge::GRAPH_SUCCESS;
    } else { // TND
        // cu_seqlens_q shape is [B+1], batch_size = shape[0] - 1
        uint32_t cuSeqLensQSize = 0;
        if (GetActualSeqLenSize(cuSeqLensQSize, opParamInfo_.cuSeqLensQ.tensor, "input cu_seqlens_q") !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(
            cuSeqLensQSize <= 1,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "cu_seqlens_q", std::to_string(cuSeqLensQSize).c_str(),
                                                      "cu_seqlens_q's shape size should be greater than  1 (B+1)"),
            return ge::GRAPH_FAILED);
        bSize_ = cuSeqLensQSize - 1;

        // Validate key side batch size consistency
        if (kLayout_ == DataLayout::TND) {
            uint32_t cuSeqLensKSize = 0;
            if (GetActualSeqLenSize(cuSeqLensKSize, opParamInfo_.cuSeqLensK.tensor, "input cu_seqlens_k") !=
                ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF((cuSeqLensKSize - 1) != bSize_,
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            opName_, "the batch sizes derived from cu_seqlens_q and cu_seqlens_k",
                            std::to_string(bSize_) + ", " + std::to_string(cuSeqLensKSize - 1),
                            "the batch sizes derived from cu_seqlens_q and cu_seqlens_k must be same"),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QLIV2InfoParser::GetHeadDim()
{
    // 以query的D维度为基准
    uint32_t dIndex = DIM_IDX_TWO;
    // 根据layout确定D维度在shape中的位置
    switch (qLayout_) {
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
    OP_CHECK_IF(headDim_ != HEAD_DIM_LIMIT,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "query",
                                                      Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()),
                                                      "input query's last dim head_dim only support 128"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(((blockSize_ % BLOCK_SIZE_FACTOR != 0) || (blockSize_ == 0) || (blockSize_ > BLOCK_SIZE_LIMIT)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                    "input key's block_size must be a multiple of 16 and belong to (0, 1024]"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OP_CHECK_IF((blockCount_ == 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "key",
                                                      Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                                                      "input key's block_count cannot be 0"),
                return ge::GRAPH_FAILED);

    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
            maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetS2SizeForBatchContinuous()
{
    std::string layout_key(opParamInfo_.layOutKey);
    if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE);
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
    }
    OP_CHECK_IF((kLayout_ != DataLayout::BSND) && (kLayout_ != DataLayout::TND),
                OP_LOGE_FOR_INVALID_FORMAT(opName_, "key", layout_key.c_str(), "BSND or TND"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::PA_BBND) {
        return GetS2SizeForPageAttention();
    }
    return GetS2SizeForBatchContinuous();
}

ge::graphStatus QLIV2InfoParser::ValidateInputShapesMatch()
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
        // bSize_ 来源于cu_seqlens_q (shape=[B+1], bSize_=B)
        OP_CHECK_IF((kLayout_ == DataLayout::PA_BBND) &&
                        ((opParamInfo_.sequsedK.tensor->GetShapeSize() != bSize_) ||
                         (opParamInfo_.blockTable.tensor != nullptr &&
                          opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "cu_seqlens_q, seqused_k and block_table dim 0",
                        std::to_string(bSize_) + ", " + std::to_string(opParamInfo_.sequsedK.tensor->GetShapeSize()) +
                            " and " + std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)),
                        "TND case input cu_seqlens_q, seqused_k, block_table dim 0 must be consistent"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (kLayout_ == DataLayout::TND) && (opParamInfo_.cuSeqLensK.tensor->GetShapeSize() != bSize_ + 1),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "cu_seqlens_q, cu_seqlens_k",
                std::to_string(bSize_ + 1) + ", " + std::to_string(opParamInfo_.cuSeqLensK.tensor->GetShapeSize()),
                "TND case input cu_seqlens_q, cu_seqlens_k shape size must be same"),
            return ge::GRAPH_FAILED);
        // -----------------------check T-------------------
        uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "query, weights, sparse_indices dim 0",
                        std::to_string(qTsize) + ", " +
                            std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + " and " +
                            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                        "TND case input query, weights, sparse_indices dim 0 must be same"),
                    return ge::GRAPH_FAILED);
        if (npuArch_ == NpuArch::DAV_3510) {
            if (*opParamInfo_.returnValue == 1 && opParamInfo_.sparseValues.shape != nullptr) {
                OP_CHECK_IF((opParamInfo_.sparseValues.shape->GetStorageShape().GetDim(0) != qTsize),
                            OP_LOGE(opName_,
                                    "TND case input query and sparse_values dim 0 are %u, %ld "
                                    "respectively, they must be same.",
                                    qTsize, opParamInfo_.sparseValues.shape->GetStorageShape().GetDim(0)),
                            return ge::GRAPH_FAILED);
            }
            if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
                OP_CHECK_IF((opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDim(0) != qTsize),
                            OP_LOGE(opName_,
                                    "TND case input query and output_idx_offset dim 0 are %u, %ld "
                                    "respectively, they must be same.",
                                    opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDim(0)),
                            return ge::GRAPH_FAILED);
            }
        }
    } else {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于query
        OP_CHECK_IF((kLayout_ == DataLayout::PA_BBND) &&
                        ((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                         (opParamInfo_.blockTable.tensor != nullptr &&
                          opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                         (opParamInfo_.sequsedK.tensor->GetShapeSize() != bSize_) ||
                         (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "query, weight, seqused_k, block_table, sparse_indices dim 0",
                        std::to_string(bSize_) + ", " +
                            std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + " , " +
                            std::to_string(opParamInfo_.sequsedK.tensor->GetShapeSize()) + " , " +
                            std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)) + " , " +
                            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                        "BSND case input query, weight, seqused_k, block_table, sparse_indices dim 0 must be same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (kLayout_ != DataLayout::PA_BBND) &&
                ((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                 (opParamInfo_.sequsedK.tensor != nullptr && opParamInfo_.sequsedK.tensor->GetShapeSize() != bSize_) ||
                 (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "query, weight, seqused_k, sparse_indices dim 0",
                std::to_string(bSize_) + ", " +
                    std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + " , " +
                    std::to_string(
                        opParamInfo_.sequsedK.tensor != nullptr ? opParamInfo_.sequsedK.tensor->GetShapeSize() : 0) +
                    " , " + std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                "BSND case input query, weight, seqused_k, sparse_indices dim 0 must be same"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (opParamInfo_.sequsedQ.tensor != nullptr) && (opParamInfo_.sequsedQ.tensor->GetShapeSize() != bSize_),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "query, seqused_q dim 0",
                std::to_string(bSize_) + ", " + std::to_string(opParamInfo_.sequsedQ.tensor->GetShapeSize()),
                "BSND case input query, seqused_q dim 0 must be same"),
            return ge::GRAPH_FAILED);
        // -----------------------check S1-------------------
        OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                        (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "query, weight and sparse_indices dim 1",
                        std::to_string(s1Size_) + ", " +
                            std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(1)) + " and " +
                            std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1)),
                        "BSND case input query, weight and sparse_indices dim 1 must be same"),
                    return ge::GRAPH_FAILED);
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "query, weight shape dim N1",
                    std::to_string(n1Size_) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim)),
                    "input query, weight shape dim N1 must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    OP_CHECK_IF(
        ((kLayout_ != DataLayout::TND && opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_THREE) != headDim_) ||
         (kLayout_ == DataLayout::TND && opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO) != headDim_)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "query, key shape last dim",
                                               std::to_string(headDim_) + ", " +
                                                   std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDim(
                                                       (kLayout_ == DataLayout::TND) ? DIM_IDX_TWO : DIM_IDX_THREE)),
                                               "input query, key shape last dim must be same"),
        return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "query and output sparse_indices shape n2 dim",
                    std::to_string(n2Size_) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim)),
                    "input query and output sparse_indices shape n2 dim must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check sparse_count-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "tosparse_countpk and sparse_indices shape last dim",
                    std::to_string(*opParamInfo_.sparseCount) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1)),
                    "output sparse_indices shape last dim must be same as attr sparse_count"),
                return ge::GRAPH_FAILED);
    // -----------------------check cmp_residual_k-------------------
    if (opParamInfo_.cmpResidualK.tensor != nullptr) {
        OP_CHECK_IF(
            (opParamInfo_.cmpResidualK.tensor->GetShapeSize() != bSize_),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "batch_size, cmp_residual_k's shape size",
                std::to_string(bSize_) + ", " + std::to_string(opParamInfo_.cmpResidualK.tensor->GetShapeSize()),
                "input cmp_residual_k's shape size must be equal to batch_size"),
            return ge::GRAPH_FAILED);
    }
    // -----------------------check sparse_values------------------
    if (npuArch_ == NpuArch::DAV_3510) {
        if (*opParamInfo_.returnValue == 1) {
            OP_CHECK_IF((opParamInfo_.sparseValues.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                        OP_LOGE(opName_, "input query and output sparse_values shape n2 dim must be same."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                (opParamInfo_.sparseValues.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.sparseCount),
                OP_LOGE(opName_, "input query and output sparse_values last dim must be same as attr topk."),
                return ge::GRAPH_FAILED);
        }
    }
    // -----------------------check metadata-------------------
    OP_CHECK_IF((opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_LIMIT),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName_, "metadata", std::to_string(opParamInfo_.metadata.tensor->GetShapeSize()).c_str(),
                    "input metadata dim0 must be " + std::to_string(METADATA_LIMIT)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QLIV2InfoParser::CheckScaleShape()
{
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qDequantScaleShapeDim = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDimNum();
    uint32_t kDequantScaleShapeDim = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDimNum();
    bool isMxQuantMode = (*opParamInfo_.quantMode == QUANT_MODE_MXFP8) || (*opParamInfo_.quantMode == QUANT_MODE_MXFP4);

    if (*opParamInfo_.quantMode == QUANT_MODE_HIFLOAT8) {
        OP_CHECK_IF(qDequantScaleShapeDim != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, "query_dequant_scale", std::to_string(qDequantScaleShapeDim).c_str(),
                        "when quant_mode is 4, query_dequant_scale shape dim should be 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDequantScaleShapeDim != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName_, "key_dequant_scale", std::to_string(kDequantScaleShapeDim).c_str(),
                        "when quant_mode is 4, key_dequant_scale shape dim should be 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(0) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "query_dequant_scale",
                        Ops::Base::ToString(opParamInfo_.query_dequant_scale.shape->GetStorageShape()).c_str(),
                        "when quant_mode is 4, query_dequant_scale's shape[0] should be 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(0) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "key_dequant_scale",
                        Ops::Base::ToString(opParamInfo_.key_dequant_scale.shape->GetStorageShape()).c_str(),
                        "when quant_mode is 4, key_dequant_scale's shape[0] should be 1"),
                    return ge::GRAPH_FAILED);
    } else if (isMxQuantMode) {
        OP_CHECK_IF(headDim_ % MX_SCALE_SHAPE_ALIGN != 0,
                    OP_LOGE(opName_, "when quant_mode is %d, head_dim should be a multiple of %u, but now is %u",
                            *opParamInfo_.quantMode, MX_SCALE_SHAPE_ALIGN, headDim_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(qDequantScaleShapeDim != (qShapeDim + 1),
                    OP_LOGE(opName_, "when quant_mode is %d, query_dequant_scale dim num should be %u, but now is %u",
                            *opParamInfo_.quantMode, qShapeDim + 1, qDequantScaleShapeDim),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDequantScaleShapeDim != (kShapeDim + 1),
                    OP_LOGE(opName_, "when quant_mode is %d, key_dequant_scale dim num should be %u, but now is %u",
                            *opParamInfo_.quantMode, kShapeDim + 1, kDequantScaleShapeDim),
                    return ge::GRAPH_FAILED);
        for (uint32_t i = 0; i < (qShapeDim - 1); i++) {
            uint32_t dimValueQueryScale = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(i);
            uint32_t dimValueQuery = opParamInfo_.query.shape->GetStorageShape().GetDim(i);
            OP_CHECK_IF(dimValueQueryScale != dimValueQuery,
                        OP_LOGE(opName_, "query_dequant_scale's shape[%u] %u and query's shape[%u] %u are not the same",
                                i, dimValueQueryScale, i, dimValueQuery),
                        return ge::GRAPH_FAILED);
        }
        for (uint32_t i = 0; i < (kShapeDim - 1); i++) {
            uint32_t dimValueKeyScale = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(i);
            uint32_t dimValueKey = opParamInfo_.key.shape->GetStorageShape().GetDim(i);
            OP_CHECK_IF(dimValueKeyScale != dimValueKey,
                        OP_LOGE(opName_, "key_dequant_scale's shape[%u] %u and key's shape[%u] %u are not the same", i,
                                dimValueKeyScale, i, dimValueKey),
                        return ge::GRAPH_FAILED);
        }
        uint32_t expectScaleD = headDim_ / MX_SCALE_SHAPE_ALIGN;
        OP_CHECK_IF(
            (opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(qShapeDim - 1) != expectScaleD) ||
                (opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(qShapeDim) != MX_E8M0_SCALE_PACK_NUM),
            OP_LOGE(opName_, "when quant_mode is %d, query_dequant_scale's last dims should be [%u, %u].",
                    *opParamInfo_.quantMode, expectScaleD, MX_E8M0_SCALE_PACK_NUM),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(kShapeDim - 1) != expectScaleD) ||
                (opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(kShapeDim) != MX_E8M0_SCALE_PACK_NUM),
            OP_LOGE(opName_, "when quant_mode is %d, key_dequant_scale's last dims should be [%u, %u].",
                    *opParamInfo_.quantMode, expectScaleD, MX_E8M0_SCALE_PACK_NUM),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            qDequantScaleShapeDim != (qShapeDim - 1),
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "query_dequant_scale", std::to_string(qDequantScaleShapeDim).c_str(),
                                         std::to_string(qShapeDim - 1).c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            kDequantScaleShapeDim != (kShapeDim - 1),
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "key_dequant_scale", std::to_string(kDequantScaleShapeDim).c_str(),
                                         std::to_string(kShapeDim - 1).c_str()),
            return ge::GRAPH_FAILED);
        // check q scale
        for (uint32_t i = 0; i < (qShapeDim - 1); i++) {
            uint32_t dimValueQueryScale = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(i);
            uint32_t dimValueQuery = opParamInfo_.query.shape->GetStorageShape().GetDim(i);
            OP_CHECK_IF(dimValueQueryScale != dimValueQuery,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "query_dequant_scale and query",
                            Ops::Base::ToString(opParamInfo_.query_dequant_scale.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()),
                            "query_dequant_scale's shape[" + std::to_string(i) + "] and query's shape[" +
                                std::to_string(i) + "] are not the same"),
                        return ge::GRAPH_FAILED);
        }
        // check k scale
        for (uint32_t i = 0; i < (kShapeDim - 1); i++) {
            uint32_t dimValueKeyScale = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDim(i);
            uint32_t dimValueKey = opParamInfo_.key.shape->GetStorageShape().GetDim(i);
            OP_CHECK_IF(dimValueKeyScale != dimValueKey,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "key_dequant_scale and key",
                            Ops::Base::ToString(opParamInfo_.key_dequant_scale.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                            "key_dequant_scale's shape[" + std::to_string(i) + "] and key's shape[" +
                                std::to_string(i) + "] are not the same"),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

// key非连续校验：通过shape计算expected stride进行校验
// PA_BBND时，只允许0轴非连续，其余轴必须连续
// 非PA_BBND时，所有轴都必须连续
ge::graphStatus QLIV2InfoParser::CheckKeyContiguous() const
{
    bool keyNonContiguous = false;
    bool scaleNonContiguous = false;
    // A5 PA_BBND: 0轴允许非连续，从1轴开始检查；非PA_BBND或A3: 从0轴开始检查
    // PA_BBND: axis 0 allows non-contiguous, check starts from axis 1
    // Non-PA_BBND: check starts from axis 0
    size_t checkStartIdx = (npuArch_ == NpuArch::DAV_3510 && kLayout_ == DataLayout::PA_BBND) ? 1 : 0;
    if (!keyStridesVec_.empty() && opParamInfo_.key.shape != nullptr) {
        auto &shape = opParamInfo_.key.shape->GetStorageShape();
        std::vector<uint32_t> expectedStrides;
        if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BBND) {
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
    bool isMxQuantMode = (*opParamInfo_.quantMode == QUANT_MODE_MXFP8) || (*opParamInfo_.quantMode == QUANT_MODE_MXFP4);
    if ((*opParamInfo_.quantMode != QUANT_MODE_HIFLOAT8) && !keyDequantScaleStridesVec_.empty() &&
        opParamInfo_.key_dequant_scale.shape != nullptr) {
        auto &shape = opParamInfo_.key_dequant_scale.shape->GetStorageShape();
        std::vector<uint32_t> expectedStrides;
        if (isMxQuantMode) {
            if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BBND) {
                expectedStrides = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3) * shape.GetDim(4),
                                   shape.GetDim(2) * shape.GetDim(3) * shape.GetDim(4),
                                   shape.GetDim(3) * shape.GetDim(4), shape.GetDim(4), 1};
            } else if (kLayout_ == DataLayout::TND) {
                expectedStrides = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3),
                                   shape.GetDim(2) * shape.GetDim(3), shape.GetDim(3), 1};
            }
        } else {
            if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BBND) {
                expectedStrides = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
            } else if (kLayout_ == DataLayout::TND) {
                expectedStrides = {shape.GetDim(1), 1};
            }
        }
        for (size_t i = checkStartIdx; i < expectedStrides.size(); ++i) {
            if (i < keyDequantScaleStridesVec_.size() && keyDequantScaleStridesVec_[i] != expectedStrides[i]) {
                scaleNonContiguous = true;
                break;
            }
        }
    }
    if (kLayout_ == DataLayout::PA_BBND) {
        if (!keyStridesVec_.empty()) {
            if (*opParamInfo_.quantMode == QUANT_MODE_MXFP4) {
                OP_CHECK_IF(
                    keyStridesVec_[0] <= 0 || keyStridesVec_[0] % MXFP4_PACK_NUM != 0,
                    OP_LOGE(opName_,
                            "when quant_mode is 5 and layout_k is PA_BBND, key stride0 must be positive and satisfy "
                            "2-element FP4 packing alignment, but got %u",
                            keyStridesVec_[0]),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(keyStridesVec_[0] <= 0,
                            OP_LOGE(opName_, "when layout_k is PA_BBND, key stride0 must be positive, but got %u",
                                    keyStridesVec_[0]),
                            return ge::GRAPH_FAILED);
            }
        }
        if (isMxQuantMode && !keyDequantScaleStridesVec_.empty()) {
            OP_CHECK_IF(
                keyDequantScaleStridesVec_[0] <= 0 || keyDequantScaleStridesVec_[0] % MX_E8M0_SCALE_PACK_NUM != 0,
                OP_LOGE(opName_,
                        "when quant_mode is 3 or 5 and layout_k is PA_BBND, key_dequant_scale stride0 must be positive "
                        "and satisfy 2-element E8M0 packing alignment, but got %u",
                        keyDequantScaleStridesVec_[0]),
                return ge::GRAPH_FAILED);
        }
    }
    OP_CHECK_IF(
        keyNonContiguous || scaleNonContiguous,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "keyNonContiguous and scaleNonContiguous",
                                              std::string(keyNonContiguous ? "true" : "false") + " and " +
                                                  std::string(scaleNonContiguous ? "true" : "false"),
                                              "key and keyscale only support non-continuous keying on the 0-axis"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QLIV2InfoParser::GenerateInfo(QLIV2TilingInfo &QLIV2Info)
{
    QLIV2Info.opName = opName_;
    QLIV2Info.platformInfo = platformInfo_;
    QLIV2Info.opParamInfo = opParamInfo_;
    QLIV2Info.socVersion = socVersion_;
    QLIV2Info.npuArch = npuArch_;

    QLIV2Info.bSize = bSize_;
    QLIV2Info.n1Size = n1Size_;
    QLIV2Info.n2Size = n2Size_;
    QLIV2Info.s1Size = s1Size_;
    QLIV2Info.s2Size = s2Size_;
    QLIV2Info.gSize = gSize_;

    QLIV2Info.inputQType = inputQType_;
    QLIV2Info.inputKType = inputKType_;
    QLIV2Info.outputType = outputType_;

    QLIV2Info.blockSize = blockSize_;
    QLIV2Info.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    QLIV2Info.pageAttentionFlag = (kLayout_ == DataLayout::PA_BBND);
    QLIV2Info.sparseMode = *opParamInfo_.sparseMode;
    QLIV2Info.sparseCount = *opParamInfo_.sparseCount;
    QLIV2Info.cmpRatio = *opParamInfo_.cmpRatio;
    QLIV2Info.returnValue = *opParamInfo_.returnValue;
    QLIV2Info.maxSeqlenQ = (opParamInfo_.maxSeqlenQ != nullptr) ? *opParamInfo_.maxSeqlenQ : -1;

    QLIV2Info.keyStridesVec = keyStridesVec_;
    QLIV2Info.keyDequantScaleStridesVec = keyDequantScaleStridesVec_;
    if (!keyStridesVec_.empty()) {
        uint32_t keyStride0 = static_cast<uint32_t>(keyStridesVec_[0]);
        if (*opParamInfo_.quantMode == QUANT_MODE_MXFP4) {
            // FP4 shape stride以逻辑元素计数，kernel侧以打包后的uint8为寻址单位。
            keyStride0 /= MXFP4_PACK_NUM;
        }
        QLIV2Info.keyStride0 = keyStride0;
    } else {
        QLIV2Info.keyStride0 = 0; // 非PA无需使用stride
    }
    if (!keyDequantScaleStridesVec_.empty()) {
        QLIV2Info.keyDequantScaleStride0 = static_cast<uint32_t>(keyDequantScaleStridesVec_[0]);
    } else if ((*opParamInfo_.quantMode == QUANT_MODE_MXFP8) || (*opParamInfo_.quantMode == QUANT_MODE_MXFP4)) {
        QLIV2Info.keyDequantScaleStride0 = static_cast<uint32_t>(blockSize_) * (headDim_ / MX_SCALE_GROUP_SIZE);
    } else {
        QLIV2Info.keyDequantScaleStride0 = 0;
    }

    QLIV2Info.inputQLayout = qLayout_;
    QLIV2Info.inputKLayout = kLayout_;
}

ge::graphStatus QLIV2InfoParser::ParseAndCheck(QLIV2TilingInfo &QLIV2Info)
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
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch() || ge::GRAPH_SUCCESS != CheckScaleShape() ||
        ge::GRAPH_SUCCESS != CheckKeyContiguous()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(QLIV2Info);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForQuantLightningIndexerV2(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantLightningIndexerV2Tiling类成员函数定义-----------------------
ge::graphStatus QuantLightningIndexerV2Tiling::DoTiling(QLIV2TilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 256;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 2048;           // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 4;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;          // TopK选取个数
    constexpr uint32_t TOPK_MAX_SIZE_950 = 8192;      // A5 TopK最大选取个数
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t S1_BASE_SIZE_950 = 4;
        constexpr uint32_t S2_BASE_SIZE_950 = 128;
        workspaceSize += S1_BASE_SIZE_950 * ((tilingInfo->s2Size + S2_BASE_SIZE_950 - 1) / S2_BASE_SIZE_950) *
                         S2_BASE_SIZE_950 * sizeof(uint16_t) * aicNum;
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE_950 * V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    } else {
        uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
        workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
        // Decode流程(LD)需要Workspace大小
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    }
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
    tilingData_.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.set_returnValue(tilingInfo->returnValue);
    tilingData_.set_maxSeqlenQ(tilingInfo->maxSeqlenQ);
    tilingData_.set_keyStride0(tilingInfo->keyStride0);
    tilingData_.set_keyDequantScaleStride0(tilingInfo->keyDequantScaleStride0);
    tilingData_.set_quantMode(*tilingInfo->opParamInfo.quantMode);
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
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForQuantLightningIndexerV2(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("QuantLightningIndexerV2", "Tiling context"),
                return ge::GRAPH_FAILED);
    QLIV2TilingInfo QLIV2Info;
    QLIV2InfoParser QLIV2InfoParser(context);
    if (QLIV2InfoParser.ParseAndCheck(QLIV2Info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantLightningIndexerV2Tiling QLIV2Tiling(context);
    return QLIV2Tiling.DoTiling(&QLIV2Info);
}

// --------------------------Tiling及函数TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantLightningIndexerV2)
    .Tiling(TilingForQuantLightningIndexerV2)
    .TilingParse<QLIV2CompileInfo>(TilingPrepareForQuantLightningIndexerV2);

} // namespace optiling
