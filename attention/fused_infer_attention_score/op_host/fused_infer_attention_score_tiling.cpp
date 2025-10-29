/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling.h"
#include "../../incre_flash_attention/op_kernel/incre_flash_attention_tiling.h"
#include "../../prompt_flash_attention/op_kernel/prompt_flash_attention_tiling_data.h"
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "fused_infer_attention_score_tiling_v3.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
static void ConvertDataTypePFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    contextKeyParams.inputDataType = context.GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context.GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context.GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
        context.GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
        context.GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.quantScale2Type = (context.GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.blockTableType = (context.GetOptionalInputDesc(BLOCK_TABLE_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(BLOCK_TABLE_INDEX)->GetDataType() : ge::DT_INT32;
    contextKeyParams.outputDataType = context.GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
    contextKeyParams.KeyAntiquantScaleType = (context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.valueAntiquantScaleType = (context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.KeyAntiquantOffsetType = (context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.valueAntiquantOffsetType = (context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX) != nullptr) ?
        context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX)->GetDataType() : contextKeyParams.inputDataType;
}

static void ConvertShapePFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    contextKeyParams.queryInputShape = context.GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context.GetInputShape(KEY_INDEX);
    contextKeyParams.valueInputShape = context.GetInputShape(VALUE_INDEX);
    contextKeyParams.pseShiftShape = context.GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context.GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.deqScale1Shape = context.GetOptionalInputShape(DEQUANT_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context.GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context.GetOptionalInputShape(DEQUANT_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context.GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context.GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context.GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context.GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.blockTableShape = context.GetOptionalInputShape(BLOCK_TABLE_INDEX);
    contextKeyParams.queryRope = context.GetOptionalInputShape(QUERY_ROPE_INDEX);
    contextKeyParams.keyRope = context.GetOptionalInputShape(KEY_ROPE_INDEX);
    contextKeyParams.outputShape = context.GetOutputShape(ATTENTION_OUT_INDEX);
    contextKeyParams.lseoutputShape = context.GetOutputShape(SOFTMAX_LSE_INDEX);

    contextKeyParams.KeyAntiquantScaleShape = context.GetOptionalInputShape(KEY_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.valueAntiquantScaleShape = context.GetOptionalInputShape(VALUE_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.KeyAntiquantOffsetShape = context.GetOptionalInputShape(KEY_ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.valueAntiquantOffsetShape = context.GetOptionalInputShape(VALUE_ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.learnableSinkShape = context.GetOptionalInputShape(LEARNABLE_SINK_INDEX);
}

static ge::graphStatus ConvertAttrsPFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Attributes returned from GetAttrs() is a nullptr"),
        return ge::GRAPH_FAILED);
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_INDEX);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    contextKeyParams.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.blockSize = attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE_INDEX);
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1U : 0U;
    contextKeyParams.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    contextKeyParams.isSoftMaxLseEnable =
        (contextKeyParams.softmaxLseFlag == nullptr) ? false : *contextKeyParams.softmaxLseFlag;
    contextKeyParams.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(KEY_ANTIQUANT_MODE_INDEX);
    contextKeyParams.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(VALUE_ANTIQUANT_MODE_INDEX);

    OP_CHECK_IF(context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX) != nullptr ||
        (attrs->GetAttrPointer<int64_t>(QUERY_QUANT_MODE_INDEX) != nullptr &&
        *attrs->GetAttrPointer<int64_t>(QUERY_QUANT_MODE_INDEX) != 0),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "PFA not support query dequant now"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetCumulativeKeyValueSInBSH(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams, int64_t &cumulativeKeyS, int64_t &cumulativeValueS,
    const int64_t validBatchOfK)
{
    // DIM_2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
    auto standardH = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(DIM_2);
    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_2) != standardH) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_2) != standardH)) {
            OP_LOGE(context.GetNodeName(), "D is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
            OP_LOGE(context.GetNodeName(), "S from Key and Value are not equal!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) == 0) {
            contextKeyParams.emptyTensor = 1U;
        }
        cumulativeKeyS += contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1);
        cumulativeValueS += contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1);
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            static_cast<uint32_t>(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetCumulativeKeyValueSInBNSD(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams, int64_t &cumulativeKeyS, int64_t &cumulativeValueS,
    const int64_t validBatchOfK)
{
    auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(1);
    auto standardD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(DIM_3);
    int64_t tmpNKv = (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads :
                                                                    *contextKeyParams.headsNumber;
    if (tmpNKv != standardN) {
        OP_LOGE(context.GetNodeName(), "kvN from tensorlist does NOT EQUAL kvN from attribute!");
        return ge::GRAPH_FAILED;
    }

    for (int64_t idx = 0; idx < validBatchOfK; ++idx) {
        if ((contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(1) != standardN) ||
            (contextKeyParams.vTensorList[idx]->GetStorageShape().GetDim(1) != standardN)) {
            OP_LOGE(context.GetNodeName(), "N is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if ((contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_3) != standardD) ||
            (contextKeyParams.vTensorList[idx]->GetStorageShape().GetDim(DIM_3) != standardD)) {
            OP_LOGE(context.GetNodeName(), "D is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_2) !=
            contextKeyParams.vTensorList[idx]->GetStorageShape().GetDim(DIM_2)) {
            OP_LOGE(context.GetNodeName(), "S from Key and Value does NOT equal but they should!");
            return ge::GRAPH_FAILED;
        }
        // DIM_2: Traverse the k list of the tiling key to check whether the second dimension of each tensor is 0.
        if (contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_2) == 0) {
            contextKeyParams.emptyTensor = 1U;
        }
        cumulativeKeyS += contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_2);
        cumulativeValueS += contextKeyParams.vTensorList[idx]->GetStorageShape().GetDim(DIM_2);
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            uint32_t(contextKeyParams.kTensorList[idx]->GetStorageShape().GetDim(DIM_2)));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetCumulativeKeyValueSInBSND(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams, int64_t &cumulativeKeyS, int64_t &cumulativeValueS,
    const int64_t validBatchOfK)
{
    auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(DIM_2);
    auto standardD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(DIM_3);
    int64_t tmpNKv = (*contextKeyParams.numKeyValueHeads != 0) ?
        *contextKeyParams.numKeyValueHeads :*contextKeyParams.headsNumber;
    if (tmpNKv != standardN) {
        OP_LOGE(context.GetNodeName(), "kvN from tensorlist does NOT EQUAL kvN from attribute!");
        return ge::GRAPH_FAILED;
    }

    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
         // DIM_2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_2) != standardN) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_2) != standardN)) {
            OP_LOGE(context.GetNodeName(), "N is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_3) != standardD) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(DIM_3) != standardD)) {
            OP_LOGE(context.GetNodeName(), "D is not the same across batch and Key Value under tensorlist mode!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
            OP_LOGE(context.GetNodeName(), "S from Key and Value does NOT equal but they should!");
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) == 0) {
            contextKeyParams.emptyTensor = 1U;
        }
        cumulativeKeyS += contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1);
        cumulativeValueS += contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1);
        contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs,
            uint32_t(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCumulativeKeyValue(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams, int64_t &cumulativeKeyS, int64_t &cumulativeValueS,
    const int64_t validBatchOfK)
{
    const string layoutStr = string(contextKeyParams.layout);
    if (layoutStr == "BSH") {
        // check all H across batches and KVs are the same under BSH layout
        OP_CHECK_IF(GetCumulativeKeyValueSInBSH(
            context, contextKeyParams, cumulativeKeyS, cumulativeValueS, validBatchOfK)!= ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "get cumulativeKeyS and cumulativeValueS in BSH failed"), return ge::GRAPH_FAILED);
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") {
        // check N and D, respectively, are the same across batches and KVs under BNSD/BNSD_BSND
        OP_CHECK_IF(GetCumulativeKeyValueSInBNSD(
            context, contextKeyParams, cumulativeKeyS, cumulativeValueS, validBatchOfK) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "get cumulativeKeyS and cumulativeValueS in BNSD/BNSD_BSND failed"), return ge::GRAPH_FAILED);
    } else {
        // check N and D, respectively, are the same across batches and KVs under BSND
        OP_CHECK_IF(GetCumulativeKeyValueSInBSND(
            context, contextKeyParams, cumulativeKeyS, cumulativeValueS, validBatchOfK) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "get cumulativeKeyS and cumulativeValueS in BSND failed"), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((contextKeyParams.emptyTensor == 1) && (cumulativeKeyS != 0) && (cumulativeValueS != 0),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Got empty tensor in key and value which is not continuous.!!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckKvPFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    const string layoutStr = string(contextKeyParams.layout);
    auto batchOfQ = 1;
    auto batchOfK = 1;
    if (layoutStr != "NSD") {
        batchOfQ = contextKeyParams.queryInputShape->GetStorageShape().GetDim(0);
        batchOfK = contextKeyParams.keyInputShape->GetStorageShape().GetDim(0);
    }

    int64_t validBatchOfK = 0; // Obtain the actual number of K input elements and determine whether they belong to the tensorlist scene
    while (context.GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
        validBatchOfK++;
        if (validBatchOfK > 1) { // If there are more than 1, break. When the input is large, it saves time. The tensorlist scene also needs to verify separately whether it is 1
            break;
        }
    }
    if ((batchOfQ != batchOfK) && (validBatchOfK > 1) && (contextKeyParams.blockTable == nullptr)) {
        validBatchOfK = 0;
        int64_t validBatchOfV = 0;
        int64_t cumulativeKeyS = 0;
        int64_t cumulativeValueS = 0;
        contextKeyParams.kTensorList.resize(batchOfQ);
        contextKeyParams.vTensorList.resize(batchOfQ);
        while (context.GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
            contextKeyParams.kTensorList[validBatchOfK] = context.GetDynamicInputShape(KEY_INDEX, validBatchOfK);
            OP_CHECK_IF(contextKeyParams.kTensorList[validBatchOfK]->GetStorageShape().GetDim(0) != 1,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                "Batch value of Key is NOT 1 but should be 1 under tensorlist mode!"),
                return ge::GRAPH_FAILED);
            validBatchOfK++;
        }

        while (context.GetDynamicInputShape(VALUE_INDEX, validBatchOfV) != nullptr) {
            contextKeyParams.vTensorList[validBatchOfV] = context.GetDynamicInputShape(VALUE_INDEX, validBatchOfV);
            OP_CHECK_IF(contextKeyParams.vTensorList[validBatchOfV]->GetStorageShape().GetDim(0) != 1,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                "Batch value of Value is NOT 1 but should be 1 under tensorlist mode!"),
                return ge::GRAPH_FAILED);
            validBatchOfV++;
        }

        OP_CHECK_IF((batchOfQ != validBatchOfK) || (validBatchOfK != validBatchOfV),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "Batch of Query, Key and Value do NOT equal but should equal under tensorlist mode!"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(CheckCumulativeKeyValue(
            context, contextKeyParams, cumulativeKeyS, cumulativeValueS, validBatchOfK) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "check cumulativeKeyS and cumulativeValueS failed"),
            return ge::GRAPH_FAILED);

        contextKeyParams.isKvContinuous = 0U;
    }
     return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParamsPFA(gert::TilingContext &context, ContextParamsForPFATiling &contextKeyParams)
{
    OP_CHECK_IF(CheckKvPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "check kv failed"), return ge::GRAPH_FAILED);

    const string layoutStr = string(contextKeyParams.layout);
    OP_CHECK_IF(((contextKeyParams.isKvContinuous == 0) && layoutStr == "TND"),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when layout is TND, tensorlist is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((contextKeyParams.queryPaddingSize != nullptr || contextKeyParams.kvPaddingSize != nullptr) && layoutStr == "TND"),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when layout is TND, left padding is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((contextKeyParams.isKvContinuous == 0) &&
         ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr))),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when tensorlist is used, left padding is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(((contextKeyParams.queryPaddingSize != nullptr) &&
        (contextKeyParams.queryPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
        contextKeyParams.queryPaddingSize->GetStorageShape().GetDimNum() != 1)),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query PaddingSize input is invalid!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((contextKeyParams.kvPaddingSize != nullptr) &&
        (contextKeyParams.kvPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
        contextKeyParams.kvPaddingSize->GetStorageShape().GetDimNum() != 1)),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "KV PaddingSize input is invalid!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(((contextKeyParams.blockTable != nullptr) &&
        ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr))),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "when page attention is used, left padding is not supported!"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(((contextKeyParams.queryPaddingSize != nullptr) && (contextKeyParams.actualSequenceLengthQ == nullptr)),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "if Query has leftpadding, the query's actual sequence lengths are required!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(((contextKeyParams.kvPaddingSize != nullptr) && (contextKeyParams.actualSequenceLengthKV == nullptr)),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "if KV has leftpadding, the key/value's actual sequence lengths are required!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsPFA(gert::TilingContext &context,
    ContextParamsForPFATiling &contextKeyParams)
{
    constexpr uint32_t FROM_FUSED_FLAG = 71;

    contextKeyParams.opName = context.GetNodeName();
    bool inputOutputIsNullPtr =
        (context.GetInputDesc(QUERY_INDEX) == nullptr) || (context.GetInputDesc(KEY_INDEX) == nullptr) ||
        (context.GetInputDesc(VALUE_INDEX) == nullptr) || (context.GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
        (context.GetInputShape(QUERY_INDEX) == nullptr) || (context.GetInputShape(KEY_INDEX) == nullptr) ||
        (context.GetInputShape(VALUE_INDEX) == nullptr) || (context.GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OP_CHECK_IF(inputOutputIsNullPtr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "q, k, v or attenOut is nullptr!"),
        return ge::GRAPH_FAILED);

    contextKeyParams.isKvContinuous = 1U;
    contextKeyParams.emptyTensor = 0U;
    contextKeyParams.fromTilingSink = 0U;
    contextKeyParams.fromFused = FROM_FUSED_FLAG;
    contextKeyParams.maxKVs = 0U;
    contextKeyParams.pseShift = context.GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context.GetOptionalInputTensor(ATTEN_MASK_INDEX);
    OP_CHECK_IF((contextKeyParams.attentionMask != nullptr) &&
        (context.GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_BOOL) &&
        (context.GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_INT8) &&
        (context.GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_UINT8),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Invalid attention mask datatype! Only support BOOL, INT8 and UINT8"), return ge::GRAPH_FAILED);
    contextKeyParams.actualSequenceLengthQ = context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSequenceLengthKV = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.queryPaddingSize = context.GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    contextKeyParams.kvPaddingSize = context.GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    contextKeyParams.blockTable = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    contextKeyParams.keySharedPrefix = context.GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    contextKeyParams.valueSharedPrefix = context.GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    contextKeyParams.actualSharedPrefixLen = context.GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);
    contextKeyParams.learnableSink = context.GetOptionalInputTensor(LEARNABLE_SINK_INDEX);
    contextKeyParams.hasKeyAntiquantScale =
        (context.GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX) == nullptr) ? false : true;
    contextKeyParams.hasValueAntiquantScale =
        (context.GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX) == nullptr) ? false : true;

    ConvertDataTypePFA(context, contextKeyParams);
    ConvertShapePFA(context, contextKeyParams);

    contextKeyParams.hasLearnableSink = ((contextKeyParams.learnableSink != nullptr) && (contextKeyParams.learnableSinkShape != nullptr) &&
                                        (contextKeyParams.learnableSinkShape->GetStorageShape().GetShapeSize() != 0) ) ? true : false;

    OP_CHECK_IF(ConvertAttrsPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "convert attrs failed"), return ge::GRAPH_FAILED);
    contextKeyParams.workspaceSize = context.GetWorkspaceSizes(1);

    OP_CHECK_IF(CheckParamsPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "check params failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void ConvertOptionalInputsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INDEX);
    ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INDEX);

    ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INDEX);
    ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INDEX);

    ifaContext.actualSeqLengthsQ.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
    ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE1_INDEX);
    ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
    ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE2_INDEX);
    ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
    ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE2_INDEX);
    ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX);
    ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
    ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    ifaContext.queryPaddingSize.tensor = context.GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    ifaContext.keyAntiquantScale.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyAntiquantScale.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyAntiquantOffset.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
    ifaContext.keyAntiquantOffset.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX);
    ifaContext.valueAntiquantScale.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    ifaContext.valueAntiquantScale.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX);
    ifaContext.valueAntiquantOffset.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
    ifaContext.valueAntiquantOffset.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX);
    ifaContext.keySharedPrefix.tensor = context.GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    ifaContext.keySharedPrefix.desc = context.GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX);
    ifaContext.valueSharedPrefix.tensor = context.GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    ifaContext.valueSharedPrefix.desc = context.GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX);
    ifaContext.actualSharedPrefixLen.tensor = context.GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);

    ifaContext.queryRope.tensor = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    ifaContext.queryRope.desc = context.GetOptionalInputDesc(QUERY_ROPE_INDEX);
    ifaContext.keyRope.tensor = context.GetOptionalInputTensor(KEY_ROPE_INDEX);
    ifaContext.keyRope.desc = context.GetOptionalInputDesc(KEY_ROPE_INDEX);
    ifaContext.keyRopeAntiquantScale.tensor = context.GetOptionalInputTensor(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    ifaContext.keyRopeAntiquantScale.desc = context.GetOptionalInputDesc(KEY_ROPE_ANTIQUANT_SCALE_INDEX);
    ifaContext.dequantScaleQuery.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    ifaContext.dequantScaleQuery.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX);
}

static ge::graphStatus ConvertAttrsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "attrs got from ge is nullptr"),
        return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    ifaContext.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    ifaContext.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
    ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX);
    ifaContext.antiquantMode = attrs->GetAttrPointer<int64_t>(ANTIQUANT_MODE_INDEX);
    ifaContext.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    ifaContext.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(KEY_ANTIQUANT_MODE_INDEX);
    ifaContext.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(VALUE_ANTIQUANT_MODE_INDEX);
    ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(ATTR_INNER_PRECISE_INDEX);
    ifaContext.sparseMode = attrs->GetAttrPointer<uint32_t>(ATTR_SPARSE_MODE_INDEX);
    ifaContext.queryQuantMode = attrs->GetAttrPointer<int64_t>(QUERY_QUANT_MODE_INDEX);
    ifaContext.windowSize = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsIFA(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ifaContext.opName = context.GetNodeName();
    ifaContext.platformInfo = context.GetPlatformInfo();
    ifaContext.query.desc = context.GetInputDesc(QUERY_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INDEX);
    OP_CHECK_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "shape of query of shape of key is null."),
        return ge::GRAPH_FAILED);
    auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(0);
    if (batchOfQuery != batchOfKey) {
        ifaContext.kCache.resize(batchOfQuery);
        ifaContext.vCache.resize(batchOfQuery);
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            ifaContext.kCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, size));
            ifaContext.vCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, size));
        }
    } else {
        ifaContext.kCache.resize(1);
        ifaContext.vCache.resize(1);
        ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, 0));
        ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, 0));
    }

    ifaContext.value.desc = context.GetInputDesc(VALUE_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INDEX);
    ifaContext.attenOut.desc = context.GetOutputDesc(ATTENTION_OUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(ATTENTION_OUT_INDEX);

    ConvertOptionalInputsIFA(context, ifaContext);

    OP_CHECK_IF(ConvertAttrsIFA(context, ifaContext) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "convert attrs failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    ifaContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDequantParams(gert::TilingContext &context, const int64_t s)
{
    OP_CHECK_IF((context.GetAttrs()->GetAttrPointer<uint64_t>(ANTIQUANT_MODE_INDEX) != nullptr) &&
        (*context.GetAttrs()->GetAttrPointer<uint64_t>(ANTIQUANT_MODE_INDEX) != 0),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "antiquant_mode is not supported!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(s > NUM_16 && (context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX) != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when s(%ld) > 16, not support dequantScaleQuery exist", s),
        return ge::GRAPH_FAILED);

    auto qRope = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    OP_CHECK_IF(qRope == nullptr && (context.GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX) != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "when qRope is null, not support dequantScaleQuery exist"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckLseShape(gert::TilingContext &context, bool lseFlag, const int64_t b,
    const int64_t s, const int64_t n)
{
    const string inputLayoutStr = string(context.GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    auto tempLse = context.GetOutputShape(SOFTMAX_LSE_INDEX);

    if (inputLayoutStr == "TND" || inputLayoutStr == "NTD_TND") {
        OP_CHECK_IF(((lseFlag != false) && (tempLse->GetStorageShape().GetDimNum() != 3)),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "Layout is %s SoftmaxLse shape dim should be 3, but got %zu!",
            inputLayoutStr.c_str(), tempLse->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

        auto tempQ = context.GetInputShape(QUERY_INDEX);
        OP_CHECK_IF(((lseFlag != false) &&
            ((tempLse->GetStorageShape().GetDim(DIM_0) != s) || (tempLse->GetStorageShape().GetDim(DIM_1) != n) ||
            (tempLse->GetStorageShape().GetDim(DIM_2) != 1))),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                "Layout is %s query Shape is [%ld, %ld, %ld], expect SoftmaxLse shape TN1 [%ld, %ld, 1], but got "
                "SoftmaxLse shape [%ld, %ld, %ld]!",
                inputLayoutStr.c_str(),
                tempQ->GetStorageShape().GetDim(DIM_0), tempQ->GetStorageShape().GetDim(DIM_1),
                tempQ->GetStorageShape().GetDim(DIM_2), s, n,
                tempLse->GetStorageShape().GetDim(DIM_0), tempLse->GetStorageShape().GetDim(DIM_1),
                tempLse->GetStorageShape().GetDim(DIM_2)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(((lseFlag != false) && (tempLse->GetStorageShape().GetDimNum() != 4)),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "SoftmaxLse shape dim should be 4!"),
            return ge::GRAPH_FAILED);

        uint32_t tempN = *context.GetAttrs()->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
        OP_CHECK_IF(((lseFlag != false) &&
            ((tempLse->GetStorageShape().GetDim(0) != b) || (tempLse->GetStorageShape().GetDim(1) != tempN) ||
            (tempLse->GetStorageShape().GetDim(DIM_2) != s) || (tempLse->GetStorageShape().GetDim(DIM_3) != 1))),
            OPS_REPORT_VECTOR_INNER_ERR(
                context.GetNodeName(),
                "SoftmaxLse shape size[%ld, %ld, %ld, %ld] does not match BNS1[%ld, %u, %ld, 1]!",
                tempLse->GetStorageShape().GetDim(0), tempLse->GetStorageShape().GetDim(1),
                tempLse->GetStorageShape().GetDim(DIM_2), tempLse->GetStorageShape().GetDim(DIM_3), b, tempN, s),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetPlatformInfo(gert::TilingContext &context, PromptFlashAttentionCompileInfo &compileInfoPtr)
{
    auto platformInfoPtr = context.GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr.l0BSize);
    compileInfoPtr.socShortName = ascendcPlatform.GetSocVersion();

    if (compileInfoPtr.socShortName == platform_ascendc::SocVersion::ASCEND310P) {
        // sys workspace size default value
        compileInfoPtr.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    } else {
        compileInfoPtr.defaultSysWorkspaceSize = 0U;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingProcess4PFA(gert::TilingContext *context, const uint32_t tempD, const int64_t b,
    const int64_t s, const int64_t n)
{
    constexpr uint64_t BENCHMARK_TILING_KEY = 1000000000000000000;
    constexpr int64_t D_ALIGN_32 = 32;
    constexpr int64_t D_ALIGN_16 = 16;

    PromptFlashAttentionTilingData* pfaTilingData = context->GetTilingData<PromptFlashAttentionTilingData>();
    PromptFlashAttentionTiling pfa_tiling(nullptr);
    ContextParamsForPFATiling contextParamsForPFATiling;
    PromptFlashAttentionCompileInfo tempCompileInfoPtr = {0, 0, 0, 0, 0, 0, 0, 0,
        platform_ascendc::SocVersion::ASCEND310P};

    auto ret = CheckDequantParams(*context, s);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    ret = SetPlatformInfo(*context, tempCompileInfoPtr);
    if (ret != ge::GRAPH_SUCCESS) return ret;

    contextParamsForPFATiling.compileInfoPtr = &tempCompileInfoPtr;
    ret = ConvertContextToParamsPFA(*context, contextParamsForPFATiling);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while convert tilingContext to PFA context");
        return ret;
    }

    bool lseFlag = *context->GetAttrs()->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    if (lseFlag != false) {
        if (pfa_tiling.CheckNonEmptyShapeExceptions(contextParamsForPFATiling,
                                                  contextParamsForPFATiling.lseoutputShape, "softmaxLse")) {
            return ge::GRAPH_FAILED;
        }
        ret = CheckLseShape(*context, lseFlag, b, s, n);
        if (ret != ge::GRAPH_SUCCESS) return ret;
    }

    const string inputLayout = string(contextParamsForPFATiling.layout);
    OP_CHECK_IF((((contextParamsForPFATiling.inputDataType == ge::DT_INT8) ||
        (contextParamsForPFATiling.kDataType == ge::DT_INT8) ||
        (contextParamsForPFATiling.outputDataType == ge::DT_INT8)) && (tempD % D_ALIGN_32 != 0)),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "D should be 32 elements aligned when int8 is involved!!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempD % D_ALIGN_16 != 0), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "D should be 16 elements aligned when with FP16/BF16 dtype!"), return ge::GRAPH_FAILED);
    uint64_t tilingKey = 7U;
    uint32_t blockDimToBeSet;
    pfa_tiling.fromPFA_ = false;
    ret = pfa_tiling.RunBigKernelTilingWithParams(contextParamsForPFATiling, tilingKey, blockDimToBeSet, pfaTilingData);
    // tilingKey += BENCHMARK_TILING_KEY;
    OP_LOGD(contextParamsForPFATiling.opName, "The final tiling key is: %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDimToBeSet);
    pfa_tiling.PromptFlashAttentionSetTilingData(context, pfaTilingData);

    return ret;
}

static bool IsUsingIFA(gert::TilingContext &context, const uint32_t tempD, const int64_t s)
{
    const string inputLayoutStr = string(context.GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    auto tempK = context.GetInputShape(KEY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    bool isPageAttention = context.GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr ? true : false;
    bool isAntiquantKv = (context.GetInputDesc(KEY_INDEX)->GetDataType() == ge::DT_INT8) &&
                         (context.GetInputDesc(QUERY_INDEX)->GetDataType() != ge::DT_INT8);
    bool isAntiquantParamFloat = (context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX) != nullptr) &&
                                 (context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX) != nullptr) &&
                                 (context.GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX)->GetDataType() == ge::DT_FLOAT) &&
                                 (context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX)->GetDataType() == ge::DT_FLOAT);

    bool usingIFA = false;
    auto qRope = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    if ((inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND")) {
        if (tempD == 512U) {
            usingIFA = true;
        } else if (!isPageAttention) {
            int64_t tempKD = tempK->GetStorageShape().GetDim(DIM_2);
            int64_t tempVD = tempV->GetStorageShape().GetDim(DIM_2);
            bool isPFADSize = (tempD == 192U && tempKD == 192 && tempVD == 192) ||
                (tempD == 192U && tempKD == 192 && tempVD == 128) ||
                (tempD == 128U && tempKD == 128 && tempVD == 128);
            if (isPFADSize) {
                usingIFA = false;
            }
        } else if (isPageAttention && isAntiquantKv && isAntiquantParamFloat) {
            // TND, antiquant,IFA支持MTP
            usingIFA = true;
        }
    } else {
        bool isMlaMtp = (qRope != nullptr) && (s > 1 && s <= 16) && (tempD == 512U); // mla mtp mode
        bool isSliding =  (*context.GetAttrs()->GetAttrPointer<uint32_t>(ATTR_SPARSE_MODE_INDEX) == 4) &&
                          (qRope != nullptr) && (tempD == 512U); // Sliding Attention
        bool isIFALayout = (inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND") ||
            (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") || (inputLayoutStr == "BSH_NBSD");
        uint32_t kDimNum = tempK->GetStorageShape().GetDimNum();
        bool isGQAMtp = (s > 1 && s <= 16) && (kDimNum == 5U);
        if (((s == 1 || isMlaMtp || isGQAMtp || isSliding) && isIFALayout)) {
            usingIFA = true;
        }
        OP_LOGI(context.GetNodeName(),
            "usingFIA is %d with inputLayoutStr[%s], Qblocksize[%ld], (qRope != nullptr)[%d]",
            usingIFA, inputLayoutStr.c_str(), s, qRope != nullptr);
    }
    return usingIFA;
}

static ge::graphStatus TilingProcess4IFA(gert::TilingContext *context)
{
    //IFA tiling path
    IncreFlashAttentionTilingDataV2* ifaTilingData = context->GetTilingData<IncreFlashAttentionTilingDataV2>();
    IncreFlashAttentionContext ifaContext {};
    auto ret = ConvertContextToParamsIFA(*context, ifaContext);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occored while convert tilingContext to ifa context");
        return ret;
    }

    if (RouteToFia(context, ifaContext)) {
        return TilingFusedInferAttentionScoreV3(context);
    }
    return TilingIncreFlashAttentionAdapter(context, ifaContext, ifaTilingData);
}

static ge::graphStatus CheckQKV(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempK = context.GetInputShape(KEY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_IF((tempQ == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempK == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Key input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempV == nullptr), OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Value input is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempOut == nullptr),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Attention_Out is null pointer!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == 0) && (tempOut->GetStorageShape().GetShapeSize() != 0),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
               "Query head should not be 0, or when attentionOut is not empty tensor, query input shoud not be empty tensor!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Query input dims are invalid!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempK->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Key input dims are invalid!"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempV->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Value input dims are invalid!"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void GetBNS(gert::TilingContext &context, int64_t &b, int64_t &n, int64_t &s)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    const string inputLayoutStr = string(context.GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "BNSD" || inputLayoutStr == "BNSD_BSND" || inputLayoutStr == "BNSD_NBSD") {
        // DIM_2: When inputLayoutStr is BNSD or BNSD_BSND, the second dimension of Q is s
        s = tempQ->GetStorageShape().GetDim(DIM_2);
    } else if (inputLayoutStr == "TND") {
        auto t = tempQ->GetStorageShape().GetDim(DIM_0);
        s = t;
        n = tempQ->GetStorageShape().GetDim(DIM_1);
    } else if(inputLayoutStr == "NTD_TND") {
        auto t = tempQ->GetStorageShape().GetDim(DIM_1);
        s = t;
        n = tempQ->GetStorageShape().GetDim(DIM_0);
    } else {
        s = tempQ->GetStorageShape().GetDim(DIM_1);
    }
    if (inputLayoutStr == "NSD") {
        b = 1;
    }
}

static ge::graphStatus CheckOutShapeInNSD(gert::TilingContext &context, uint32_t &tempD)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != 3), // 3: dim nsd
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Layout is NSD, queryDims must be 3! but actual value is %zu.",
        tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    tempD = tempQ->GetStorageShape().GetDim(DIM_2);
    OP_CHECK_IF((tempQ->GetStorageShape() != tempOut->GetStorageShape()),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                                    "Layout is NSD and Query shape size[%ld, %ld, %ld] does NOT match "
                                    "Attention Out shape size[%ld, %ld, %ld]!",
                                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                                    tempQ->GetStorageShape().GetDim(DIM_2), tempOut->GetStorageShape().GetDim(0),
                                    tempOut->GetStorageShape().GetDim(1),
                                    tempOut->GetStorageShape().GetDim(DIM_2)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInBSH(gert::TilingContext &context, const int64_t s)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    auto attrs = context.GetAttrs();
    int64_t preToken = *attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);

    // ifa page attention and sliding setting
    auto qRope = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    bool isPageAttention = context.GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr ? true : false;
    bool isUnequalKvDim = (s == 1) && (preToken > 0) && isPageAttention &&
        (tempV->GetStorageShape().GetDimNum() == DIM_BSH) && (qRope == nullptr);
    if (isUnequalKvDim) {
        uint32_t tempN = *attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
        uint32_t tempNKv = *attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
        if (tempNKv == 0U) {
            tempNKv = tempN;
        }
        uint32_t tempVD = tempV->GetStorageShape().GetDim(2) / tempNKv;
        int64_t expectThirdDim = static_cast<int64_t>(tempN * tempVD);
        bool isOutputShapeInvalid =
            (tempQ->GetStorageShape().GetDim(FIRST_DIM) != tempOut->GetStorageShape().GetDim(FIRST_DIM)) ||
            (tempQ->GetStorageShape().GetDim(SECOND_DIM) != tempOut->GetStorageShape().GetDim(SECOND_DIM)) ||
            (expectThirdDim != tempOut->GetStorageShape().GetDim(THIRD_DIM));
        OP_CHECK_IF(isOutputShapeInvalid,
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                                        "Layout is BSH expect Attention Out shape [%ld, %ld, %ld] but got "
                                        "Attention Out shape [%ld, %ld, %ld]!",
                                        tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                                        expectThirdDim, tempOut->GetStorageShape().GetDim(0),
                                        tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(2)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((tempQ->GetStorageShape() != tempOut->GetStorageShape()),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                                        "Layout is BSH and Query shape size[%ld, %ld, %ld] does NOT match "
                                        "Attention Out shape size[%ld, %ld, %ld]!",
                                        tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                                        tempQ->GetStorageShape().GetDim(2), tempOut->GetStorageShape().GetDim(0),
                                        tempOut->GetStorageShape().GetDim(1),
                                        tempOut->GetStorageShape().GetDim(2)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInBSHNBSD(gert::TilingContext &context, const uint32_t tempD, const uint32_t tempN)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != 4),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Layout is BSH_NBSD, output shape dim(%zu) should be 4!",
            tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

    bool inOutShapeFlag = (tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) || // B
                            (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(DIM_2)) || // S
                            (tempN != tempOut->GetStorageShape().GetDim(0)) || // N
                            (tempD != tempOut->GetStorageShape().GetDim(DIM_3));   // D
    OP_CHECK_IF(
        inOutShapeFlag, OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                                    "Layout is BSH_NBSD and Query shape size[%ld, %ld, %ld], the Query's heads num[%u],"
                                    "expect Attention Out shape size[%u, %ld, %ld, %u],"
                                    "but got Attention Out shape size[%ld, %ld, %ld, %ld]!",
                                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                                    tempQ->GetStorageShape().GetDim(DIM_2), tempN, tempN,
                                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),tempD,
                                    tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                                    tempOut->GetStorageShape().GetDim(DIM_2), tempOut->GetStorageShape().GetDim(DIM_3)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInTND(gert::TilingContext &context, const uint32_t tempD)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    auto qRope = context.GetOptionalInputTensor(QUERY_ROPE_INDEX);
    bool ifaWithoutPA = (tempD == 512 && qRope != nullptr);
    OP_CHECK_IF(ifaWithoutPA,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Layout is TND, MLA enabled, PA must be enabled when query's D dimension is 512."), return ge::GRAPH_FAILED);
    bool isOutputShapeInvalid =
        (tempQ->GetStorageShape().GetDim(FIRST_DIM) != tempOut->GetStorageShape().GetDim(FIRST_DIM)) ||
        (tempQ->GetStorageShape().GetDim(SECOND_DIM) != tempOut->GetStorageShape().GetDim(SECOND_DIM)) ||
        (tempV->GetStorageShape().GetDim(THIRD_DIM) != tempOut->GetStorageShape().GetDim(THIRD_DIM));
    OP_CHECK_IF(isOutputShapeInvalid,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                                    "Layout is TND expect Attention Out shape [%ld, %ld, %ld] but got "
                                    "Attention Out shape [%ld, %ld, %ld]!",
                                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                                    tempV->GetStorageShape().GetDim(DIM_2), tempOut->GetStorageShape().GetDim(0),
                                    tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(DIM_2)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInTNDNTD(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    bool inOutShapeFlag = (tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) ||
        (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(0)) ||
        (tempQ->GetStorageShape().GetDim(DIM_2) != tempOut->GetStorageShape().GetDim(DIM_2));
    OP_CHECK_IF(inOutShapeFlag, OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Layout is TND_NTD and Query shape size[%ld, %ld, %ld],"
        "expect Attention Out shape size[%ld, %ld, %ld],"
        "but got Attention Out shape size[%ld, %ld, %ld]!",
        tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(DIM_2),
        tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(DIM_2),
        tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),tempOut->GetStorageShape().GetDim(DIM_2)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInNTDTND(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    bool inOutShapeFlag = (tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) ||
        (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(0)) ||
        (tempV->GetStorageShape().GetDim(DIM_2) != tempOut->GetStorageShape().GetDim(DIM_2));
    OP_CHECK_IF(inOutShapeFlag, OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Layout is NTD_TND and Query shape size[%ld, %ld, %ld],"
        "expect Attention Out shape size[%ld, %ld, %ld],"
        "but got Attention Out shape size[%ld, %ld, %ld]!",
        tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1), tempV->GetStorageShape().GetDim(DIM_2),
        tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(DIM_2),
        tempOut->GetStorageShape().GetDim(0),tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(DIM_2)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInBNSDBSND(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    OP_CHECK_IF(
        ((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(0)) ||
        (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(DIM_2)) ||
        (tempQ->GetStorageShape().GetDim(DIM_2) != tempOut->GetStorageShape().GetDim(1)) ||
        (tempQ->GetStorageShape().GetDim(DIM_3) != tempOut->GetStorageShape().GetDim(DIM_3))),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                                    "Layout is BNSD_BSND and Query shape size[%ld, %ld, %ld, %ld],"
                                    "expect Attention Out shape size[%ld, %ld, %ld, %ld],"
                                    "but got Attention Out shape size[%ld, %ld, %ld, %ld]!",
                                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                                    tempQ->GetStorageShape().GetDim(DIM_2), tempQ->GetStorageShape().GetDim(DIM_3),
                                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(DIM_2),
                                    tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(DIM_3),
                                    tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                                    tempOut->GetStorageShape().GetDim(DIM_2), tempOut->GetStorageShape().GetDim(DIM_3)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInBNSDNBSD(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    bool inOutShapeFlag = (tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) ||
                            (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(0)) ||
                            (tempQ->GetStorageShape().GetDim(DIM_2) != tempOut->GetStorageShape().GetDim(DIM_2)) ||
                            (tempQ->GetStorageShape().GetDim(DIM_3) != tempOut->GetStorageShape().GetDim(DIM_3));
    OP_CHECK_IF(inOutShapeFlag,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                                    "Layout is BNSD_NBSD and Query shape size[%ld, %ld, %ld, %ld],"
                                    "expect Attention Out shape size[%ld, %ld, %ld, %ld],"
                                    "but got Attention Out shape size[%ld, %ld, %ld, %ld]!",
                                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                                    tempQ->GetStorageShape().GetDim(DIM_2), tempQ->GetStorageShape().GetDim(DIM_3),
                                    tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(0),
                                    tempQ->GetStorageShape().GetDim(DIM_2), tempQ->GetStorageShape().GetDim(DIM_3),
                                    tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                                    tempOut->GetStorageShape().GetDim(DIM_2), tempOut->GetStorageShape().GetDim(DIM_3)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInBSNDNBSD(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    bool inOutShapeFlag = (tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(1)) ||
                            (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(DIM_2)) ||
                            (tempQ->GetStorageShape().GetDim(DIM_2) != tempOut->GetStorageShape().GetDim(0)) ||
                            (tempQ->GetStorageShape().GetDim(DIM_3) != tempOut->GetStorageShape().GetDim(DIM_3));
    OP_CHECK_IF(inOutShapeFlag,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
                                    "Layout is BSND_NBSD and Query shape size[%ld, %ld, %ld, %ld],"
                                    "expect Attention Out shape size[%ld, %ld, %ld, %ld],"
                                    "but got Attention Out shape size[%ld, %ld, %ld, %ld]!",
                                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1),
                                    tempQ->GetStorageShape().GetDim(DIM_2), tempQ->GetStorageShape().GetDim(DIM_3),
                                    tempQ->GetStorageShape().GetDim(DIM_2), tempQ->GetStorageShape().GetDim(0),
                                    tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(DIM_3),
                                    tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1),
                                    tempOut->GetStorageShape().GetDim(DIM_2), tempOut->GetStorageShape().GetDim(DIM_3)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInBNSD(gert::TilingContext &context)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    bool inOutShapeFlag = false;
    int64_t outD = 0;
    bool isPageAttention = context.GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr ? true : false;
    if (isPageAttention) {
        inOutShapeFlag = tempQ->GetStorageShape() != tempOut->GetStorageShape();
        outD = tempQ->GetStorageShape().GetDim(DIM_3);
    } else {
        inOutShapeFlag = (tempQ->GetStorageShape().GetDim(DIM_0) != tempOut->GetStorageShape().GetDim(DIM_0)) ||
            (tempQ->GetStorageShape().GetDim(DIM_1) != tempOut->GetStorageShape().GetDim(DIM_1)) ||
            (tempQ->GetStorageShape().GetDim(DIM_2) != tempOut->GetStorageShape().GetDim(DIM_2)) ||
            (tempV->GetStorageShape().GetDim(DIM_3) != tempOut->GetStorageShape().GetDim(DIM_3));
        outD = tempV->GetStorageShape().GetDim(DIM_3);
    }
    OP_CHECK_IF(
        inOutShapeFlag,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "Layout is BNSD and Query shape [%ld, %ld, %ld, %ld] "
            "expect Attention Out Shape [%ld, %ld, %ld, %ld] but got "
            "Attention Out shape [%ld, %ld, %ld, %ld]!",
            tempQ->GetStorageShape().GetDim(DIM_0), tempQ->GetStorageShape().GetDim(DIM_1),
            tempQ->GetStorageShape().GetDim(DIM_2), tempQ->GetStorageShape().GetDim(DIM_3),
            tempQ->GetStorageShape().GetDim(DIM_0), tempQ->GetStorageShape().GetDim(DIM_1),
            tempQ->GetStorageShape().GetDim(DIM_2), outD,
            tempOut->GetStorageShape().GetDim(DIM_0), tempOut->GetStorageShape().GetDim(DIM_1),
            tempOut->GetStorageShape().GetDim(DIM_2), tempOut->GetStorageShape().GetDim(DIM_3)),
            return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInTND4PA(gert::TilingContext &context, const int64_t n)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    auto tempInVDim = tempV->GetStorageShape().GetDim(THIRD_DIM);
    int32_t tempNKvHeads = *context.GetAttrs()->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    int64_t tmpNKv = (tempNKvHeads != 0) ? tempNKvHeads : n;
    if (tempV->GetStorageShape().GetDimNum() == DIM_NUM_3) {
        tempInVDim = tempV->GetStorageShape().GetDim(THIRD_DIM) / tmpNKv;
    } else if (tempV->GetStorageShape().GetDimNum() == DIM_NUM_4) {
        tempInVDim = tempV->GetStorageShape().GetDim(FOURTH_DIM);
    } else {
        tempInVDim = tempV->GetStorageShape().GetDim(THIRD_DIM) * tempV->GetStorageShape().GetDim(FIFTH_DIM);
    }
    bool isOutputShapeInvalid =
        (tempQ->GetStorageShape().GetDim(FIRST_DIM) != tempOut->GetStorageShape().GetDim(FIRST_DIM)) ||
        (tempQ->GetStorageShape().GetDim(SECOND_DIM) != tempOut->GetStorageShape().GetDim(SECOND_DIM)) ||
        (tempInVDim != tempOut->GetStorageShape().GetDim(THIRD_DIM));
    OP_CHECK_IF(isOutputShapeInvalid,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Layout is TND expect Attention Out shape [%ld, %ld, %ld] but got "
        "Attention Out shape [%ld, %ld, %ld]!",
        tempQ->GetStorageShape().GetDim(FIRST_DIM), tempQ->GetStorageShape().GetDim(SECOND_DIM),
        tempInVDim, tempOut->GetStorageShape().GetDim(FIRST_DIM),
        tempOut->GetStorageShape().GetDim(SECOND_DIM), tempOut->GetStorageShape().GetDim(THIRD_DIM)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeInNTDTND4PA(gert::TilingContext &context, const int64_t n)
{
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);

    auto tempInVD = tempV->GetStorageShape().GetDim(THIRD_DIM);
    int32_t tempNKvHeads = *context.GetAttrs()->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    int64_t tmpNKv = (tempNKvHeads != 0) ? tempNKvHeads : n;
    if (tempV->GetStorageShape().GetDimNum() == DIM_NUM_3) {
        tempInVD = tempV->GetStorageShape().GetDim(THIRD_DIM) / tmpNKv;
    } else if (tempV->GetStorageShape().GetDimNum() == DIM_NUM_4) {
        tempInVD = tempV->GetStorageShape().GetDim(FOURTH_DIM);
    } else {
        tempInVD = tempV->GetStorageShape().GetDim(THIRD_DIM) * tempV->GetStorageShape().GetDim(FIFTH_DIM);
    }
    bool inOutShapeFlag =
        (tempQ->GetStorageShape().GetDim(FIRST_DIM) != tempOut->GetStorageShape().GetDim(SECOND_DIM)) ||
        (tempQ->GetStorageShape().GetDim(SECOND_DIM) != tempOut->GetStorageShape().GetDim(FIRST_DIM)) ||
        (tempInVD != tempOut->GetStorageShape().GetDim(THIRD_DIM));
    OP_CHECK_IF(inOutShapeFlag,
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
        "Layout is NTD_TND and Query shape size[%ld, %ld, %ld],"
        "expect Attention Out shape size[%ld, %ld, %ld],"
        "but got Attention Out shape size[%ld, %ld, %ld]!",
        tempQ->GetStorageShape().GetDim(FIRST_DIM), tempQ->GetStorageShape().GetDim(SECOND_DIM), tempQ->GetStorageShape().GetDim(THIRD_DIM),
        tempQ->GetStorageShape().GetDim(SECOND_DIM),tempQ->GetStorageShape().GetDim(FIRST_DIM), tempInVD,
        tempOut->GetStorageShape().GetDim(FIRST_DIM),tempOut->GetStorageShape().GetDim(SECOND_DIM), tempOut->GetStorageShape().GetDim(THIRD_DIM)),
        return ge::GRAPH_FAILED);
        
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutShapeTNDs(gert::TilingContext &context, uint32_t &tempD, const int64_t n)
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    auto tempQ = context.GetInputShape(QUERY_INDEX);
    auto tempK = context.GetInputShape(KEY_INDEX);
    auto tempV = context.GetInputShape(VALUE_INDEX);
    auto tempOut = context.GetOutputShape(ATTENTION_OUT_INDEX);
    const string inputLayoutStr = string(context.GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));

    bool isPageAttention = context.GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr ? true : false;
    if (!isPageAttention && (inputLayoutStr == "TND" || inputLayoutStr == "NTD_TND")) {
        OP_CHECK_IF(((tempQ->GetStorageShape().GetDimNum() != DIM_TND) || (tempK->GetStorageShape().GetDimNum() != DIM_TND) ||
            (tempV->GetStorageShape().GetDimNum() != DIM_TND)),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(),
            "When layout is %s, queryDim(%zu) keyDim(%zu) valueDim(%zu) must be 3!",
            inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDimNum(), tempK->GetStorageShape().GetDimNum(),
            tempV->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((tempOut->GetStorageShape().GetDimNum() != DIM_TND),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Layout is %s Attention out shape dim should be 3, but got %zu!",
        inputLayoutStr.c_str(), tempOut->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != DIM_TND),
        OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Layout is %s, queryDims must be 3! but actual value is %zu.",
        inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

    tempD = tempQ->GetStorageShape().GetDim(THIRD_DIM);
    if (inputLayoutStr == "TND") {
        // PFA has a scenario which qkv with different D，the output needs determined by valueD.
        ret = isPageAttention ? CheckOutShapeInTND4PA(context, n) : CheckOutShapeInTND(context, tempD);
    } else if (inputLayoutStr == "TND_NTD") {
        ret = CheckOutShapeInTNDNTD(context);
    } else if (inputLayoutStr == "NTD_TND") {
        ret = isPageAttention ? CheckOutShapeInNTDTND4PA(context, n) : CheckOutShapeInNTDTND(context);
    }

    return ret;
}

static ge::graphStatus CheckOutShape(gert::TilingContext &context, uint32_t &tempD, const int64_t n, const int64_t s)
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    uint32_t tempN = *context.GetAttrs()->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    OP_CHECK_IF(tempN == 0, OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Q numhead is 0!"), return ge::GRAPH_FAILED);

    auto tempQ = context.GetInputShape(QUERY_INDEX);
    const string inputLayoutStr = string(context.GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "NSD") {
        ret = CheckOutShapeInNSD(context, tempD);
    } else if (inputLayoutStr == "BSH" || inputLayoutStr == "BSH_NBSD") {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != DIM_BSH),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Layout is %s, queryDims must be 3! but actual value is %zu.",
            inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        tempD = tempQ->GetStorageShape().GetDim(DIM_2) / tempN; // 2: When inputLayoutStr is BSH, the second dimension of Q divided by N is D
        if (inputLayoutStr == "BSH") {
            ret = CheckOutShapeInBSH(context, s);
        } else if (inputLayoutStr == "BSH_NBSD") {
            ret = CheckOutShapeInBSHNBSD(context, tempD, tempN);
        }
    } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD" || inputLayoutStr == "NTD_TND") {
        ret = CheckOutShapeTNDs(context, tempD, n);
    } else {
        OP_CHECK_IF((tempQ->GetStorageShape().GetDimNum() != DIM_BNSD),
            OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Layout is %s, queryDims must be 4! but actual value is %zu.",
            inputLayoutStr.c_str(), tempQ->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
        tempD = tempQ->GetStorageShape().GetDim(DIM_3); // 3: In other cases, the third dimension of Q is D

        if (inputLayoutStr == "BNSD_BSND") {
            ret = CheckOutShapeInBNSDBSND(context);
        } else if (inputLayoutStr == "BNSD_NBSD") {
            ret = CheckOutShapeInBNSDNBSD(context);
        } else if (inputLayoutStr == "BSND_NBSD") {
            ret = CheckOutShapeInBSNDNBSD(context);
        } else if (inputLayoutStr == "BNSD") {
            ret = CheckOutShapeInBNSD(context);
        }
    }
    bool isLearnableSink = context.GetOptionalInputTensor(LEARNABLE_SINK_INDEX) != nullptr ? true : false;
    OP_CHECK_IF((isLearnableSink) && !(((inputLayoutStr == "TND") || (inputLayoutStr == "NTD_TND")) && ((tempD == 64) || (tempD == 128))),
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "Only layout TND/NTD_TND and valueD = 128/64 supported learnable_sink, but actual layout is %s and D = %u!",
               inputLayoutStr.c_str(), tempD),
               return ge::GRAPH_FAILED);

    return ret;
}

ge::graphStatus TilingFusedInferAttentionScore(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(CheckQKV(*context) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check query/key/value failed"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "Attributes returned from GetAttrs() is a nullptr"), return ge::GRAPH_FAILED);
    const string inputLayoutStr = string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    if (inputLayoutStr == "SH") {
        OP_LOGE(context->GetNodeName(), "SH layout is not supported!");
        return ge::GRAPH_FAILED;
    }

    // MLA support layout
    bool isIfaMlaLayout = (inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") ||
        (inputLayoutStr == "BSND") || (inputLayoutStr == "BNSD_NBSD") || (inputLayoutStr == "BSND_NBSD") ||
        (inputLayoutStr == "BSH_NBSD") || (inputLayoutStr == "TND_NTD") || (inputLayoutStr == "TND");
    OP_CHECK_IF(((context->GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr) && !isIfaMlaLayout && (inputLayoutStr != "NTD_TND")),
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "inputlayout(%s) only support {BSH, BNSD, BSND, TND, BNSD_NBSD, "
                                                            "BSND_NBSD, BSH_NBSD, TND_NTD} in mla.", inputLayoutStr.c_str()),
        return ge::GRAPH_FAILED);

    int64_t b = context->GetInputShape(QUERY_INDEX)->GetStorageShape().GetDim(0);
    int64_t n = 0;
    int64_t s = 0;
    GetBNS(*context, b, n, s);

    uint32_t tempD = 1U;
    OP_CHECK_IF(CheckOutShape(*context, tempD, n, s) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "check output shape failed"), return ge::GRAPH_FAILED);
    uint32_t maxDlimit = 512U;
    OP_CHECK_IF((tempD > maxDlimit), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "D should be less than or equal to 512 of Q/KV shape! but now D = %u. "
        "When layout is BNSD, D is the last dimension of Q/KV shape, and layout is BSH, D = h / n", tempD),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((s == 1) && (inputLayoutStr == "BNSD_BSND")), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
        "BNSD_BSND layout is not supported when S is 1!"), return ge::GRAPH_FAILED);

    bool usingIFA = IsUsingIFA(*context, tempD, s);
    if (usingIFA) {
        // IFA tiling process
        OP_CHECK_IF(TilingProcess4IFA(context) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling process fo ifa failed"),
            return ge::GRAPH_FAILED);
    } else {
        // PFA tiling process
        OP_CHECK_IF(TilingProcess4PFA(context, tempD, b, s, n) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling process fo ifa failed"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

FIA_EXTERN_C ge::graphStatus DoOpTilingFusedInferAttentionScore(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("FusedInferAttentionScore", "Tiling context is null."),
        return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto socShortName = ascendcPlatform.GetSocVersion();
    return TilingFusedInferAttentionScore(context);
}

extern "C" {
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context)
{
    return TilingIncreFlashAttention(context);
}
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingFusedInferAttentionScore(
    gert::TilingContext *context)
{
    return DoOpTilingFusedInferAttentionScore(context);
}
}

} // namespace optiling
