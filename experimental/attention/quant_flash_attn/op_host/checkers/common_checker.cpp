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
 * \file common_checker.cpp
 * \brief Common checker for layout, shape, dtype parameters (文档约束: 公共参数组)
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../quant_flash_attn_tiling_info.h"
#include "common_checker.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;

ge::graphStatus CommonChecker::CheckSingleParaLayout(const QuantFlashAttnTilingInfo &qfaInfo)
{
    const std::vector<FiaLayout> supportedQLayouts = {FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND};
    const std::vector<FiaLayout> supportedKvLayouts = {FiaLayout::BSND,  FiaLayout::BNSD,   FiaLayout::TND,
                                                       FiaLayout::BnBsH, FiaLayout::BnNBsD, FiaLayout::NZ};
    const std::vector<FiaLayout> supportedOutLayouts = {FiaLayout::BSND, FiaLayout::BNSD, FiaLayout::TND};

    if (std::find(supportedQLayouts.begin(), supportedQLayouts.end(), qfaInfo.layoutQ) == supportedQLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, LAYOUT_Q_NAME.c_str(),
                                              LayoutToSerialStr(qfaInfo.layoutQ).c_str(),
                                              "The value of layout_q must be in BSND/BNSD/TND");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), qfaInfo.layoutKV) == supportedKvLayouts.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, LAYOUT_KV_NAME.c_str(),
                                              LayoutToSerialStr(qfaInfo.layoutKV).c_str(),
                                              "The value of layout_kv can only be BSND/BNSD/TND/PA_BBND/PA_BNBD/PA_NZ"),
        return ge::GRAPH_FAILED);

    if (std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), qfaInfo.layoutOut) ==
        supportedOutLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, LAYOUT_OUT_NAME.c_str(),
                                              LayoutToSerialStr(qfaInfo.layoutOut).c_str(),
                                              "The value of layout_out must be in BSND/BNSD/TND");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSingleParaDtype(const QuantFlashAttnTilingInfo &qfaInfo)
{
    const gert::CompileTimeTensorDesc *queryDesc = qfaInfo.opParamInfo.query.desc;
    if (queryDesc != nullptr) {
        const std::vector<ge::DataType> supportedQkvDtypes = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1};
        OP_CHECK_IF(std::find(supportedQkvDtypes.begin(), supportedQkvDtypes.end(), queryDesc->GetDataType()) ==
                        supportedQkvDtypes.end(),
                    OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, QUERY_NAME.c_str(),
                                              DataTypeToSerialStr(queryDesc->GetDataType()).c_str(),
                                              "FLOAT8_E4M3FN/FLOAT4_E2M1"),
                    return ge::GRAPH_FAILED);
        if (CheckFormatSupport(queryDesc, QUERY_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::CompileTimeTensorDesc *keyDesc = qfaInfo.opParamInfo.key.desc;
    if (keyDesc != nullptr) {
        const std::vector<ge::DataType> supportedQkvDtypes = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1};
        OP_CHECK_IF(
            std::find(supportedQkvDtypes.begin(), supportedQkvDtypes.end(), keyDesc->GetDataType()) ==
                supportedQkvDtypes.end(),
            OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, KEY_NAME.c_str(),
                                      DataTypeToSerialStr(keyDesc->GetDataType()).c_str(), "FLOAT8_E4M3FN/FLOAT4_E2M1"),
            return ge::GRAPH_FAILED);
        if (CheckFormatSupport(keyDesc, KEY_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::CompileTimeTensorDesc *valueDesc = qfaInfo.opParamInfo.value.desc;
    if (valueDesc != nullptr) {
        const std::vector<ge::DataType> supportedQkvDtypes = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1};
        OP_CHECK_IF(std::find(supportedQkvDtypes.begin(), supportedQkvDtypes.end(), valueDesc->GetDataType()) ==
                        supportedQkvDtypes.end(),
                    OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, VALUE_NAME.c_str(),
                                              DataTypeToSerialStr(valueDesc->GetDataType()).c_str(),
                                              "FLOAT8_E4M3FN/FLOAT4_E2M1"),
                    return ge::GRAPH_FAILED);
        if (CheckFormatSupport(valueDesc, VALUE_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::CompileTimeTensorDesc *attnOutDesc = qfaInfo.opParamInfo.attnOut.desc;
    if (attnOutDesc != nullptr) {
        OP_CHECK_IF(attnOutDesc->GetDataType() != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, ATTEN_OUT_NAME.c_str(),
                                              DataTypeToSerialStr(attnOutDesc->GetDataType()).c_str(), "BFLOAT16"),
                    return ge::GRAPH_FAILED);
        if (CheckFormatSupport(attnOutDesc, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // metadata: tensor_type 仅支持 INT32
    const gert::CompileTimeTensorDesc *metadataDesc = qfaInfo.opParamInfo.metadata.desc;
    if (metadataDesc != nullptr) {
        OP_CHECK_IF(metadataDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, METADATA_NAME.c_str(),
                                              DataTypeToSerialStr(metadataDesc->GetDataType()).c_str(), "INT32"),
                    return ge::GRAPH_FAILED);
        if (CheckFormatSupport(metadataDesc, METADATA_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSingleParaShapeDim(const QuantFlashAttnTilingInfo &qfaInfo)
{
    const std::vector<uint32_t> supportedQDims = {DIM_NUM_3, DIM_NUM_4};
    const std::vector<uint32_t> supportedKvDims = {DIM_NUM_3, DIM_NUM_4, DIM_NUM_5};
    const std::vector<uint32_t> supportedOutDims = {DIM_NUM_3, DIM_NUM_4};

    const gert::StorageShape *queryShape = qfaInfo.opParamInfo.query.shape;
    if (queryShape != nullptr) {
        uint32_t queryDimNum = queryShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(std::find(supportedQDims.begin(), supportedQDims.end(), queryDimNum) == supportedQDims.end(),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, QUERY_NAME.c_str(),
                                                 (std::to_string(queryDimNum) + "D").c_str(), "3D/4D"),
                    return ge::GRAPH_FAILED);
    }

    const gert::StorageShape *keyShape = qfaInfo.opParamInfo.key.shape;
    if (keyShape != nullptr) {
        uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(std::find(supportedKvDims.begin(), supportedKvDims.end(), keyDimNum) == supportedKvDims.end(),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, KEY_NAME.c_str(),
                                                 (std::to_string(keyDimNum) + "D").c_str(), "3D/4D/5D"),
                    return ge::GRAPH_FAILED);
    }

    const gert::StorageShape *valueShape = qfaInfo.opParamInfo.value.shape;
    if (valueShape != nullptr) {
        uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(std::find(supportedKvDims.begin(), supportedKvDims.end(), valueDimNum) == supportedKvDims.end(),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, VALUE_NAME.c_str(),
                                                 (std::to_string(valueDimNum) + "D").c_str(), "3D/4D/5D"),
                    return ge::GRAPH_FAILED);
    }

    const gert::StorageShape *attnOutShape = qfaInfo.opParamInfo.attnOut.shape;
    if (attnOutShape != nullptr) {
        uint32_t attnOutDimNum = attnOutShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            std::find(supportedOutDims.begin(), supportedOutDims.end(), attnOutDimNum) == supportedOutDims.end(),
            OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, ATTEN_OUT_NAME.c_str(),
                                         (std::to_string(attnOutDimNum) + "D").c_str(), "3D/4D"),
            return ge::GRAPH_FAILED);
    }

    const gert::Tensor *metadataTensor = qfaInfo.opParamInfo.metadata.tensor;
    if (metadataTensor != nullptr) {
        uint32_t dimNum = metadataTensor->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != DIM_NUM_1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, METADATA_NAME.c_str(),
                                                 (std::to_string(dimNum) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);

        int64_t dim0 = metadataTensor->GetStorageShape().GetDim(0);
        OP_CHECK_IF(dim0 <= 0, OP_LOGE(qfaInfo.opName, "metadata shape dim0(%ld) must be greater than 0", dim0),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSingleParaSoftmaxScale(const QuantFlashAttnTilingInfo &qfaInfo)
{
    OP_CHECK_IF(qfaInfo.softmaxScale <= 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, SOFTMAX_SCALE_NAME.c_str(),
                                                      std::to_string(qfaInfo.softmaxScale).c_str(),
                                                      "softmax_scale must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckSingleParaLayout(qfaInfo) != ge::GRAPH_SUCCESS || CheckSingleParaDtype(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaShapeDim(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaSoftmaxScale(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMetadataExistence(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // metadata 为可选输入，但当前不支持不传入，desc 或 tensor 为空即视为未传入
    if (qfaInfo.opParamInfo.metadata.desc == nullptr || qfaInfo.opParamInfo.metadata.tensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, METADATA_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo)
{
    OP_CHECK_IF(qfaInfo.opParamInfo.query.desc == nullptr || qfaInfo.opParamInfo.query.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, QUERY_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.key.desc == nullptr || qfaInfo.opParamInfo.key.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, KEY_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.value.desc == nullptr || qfaInfo.opParamInfo.value.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, VALUE_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.attnOut.desc == nullptr || qfaInfo.opParamInfo.attnOut.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, ATTEN_OUT_NAME.c_str()), return ge::GRAPH_FAILED);

    if (CheckMetadataExistence(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDtypeConsistency(const QuantFlashAttnTilingInfo &qfaInfo)
{
    const gert::CompileTimeTensorDesc *queryDesc = qfaInfo.opParamInfo.query.desc;
    const gert::CompileTimeTensorDesc *keyDesc = qfaInfo.opParamInfo.key.desc;
    const gert::CompileTimeTensorDesc *valueDesc = qfaInfo.opParamInfo.value.desc;
    if (queryDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    ge::DataType queryDtype = queryDesc->GetDataType();

    OP_CHECK_IF(keyDesc != nullptr && keyDesc->GetDataType() != queryDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(qfaInfo.opName, KEY_NAME.c_str(),
                                                      DataTypeToSerialStr(keyDesc->GetDataType()).c_str(),
                                                      "The dtype of key must be the same as dtype of query"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(valueDesc != nullptr && valueDesc->GetDataType() != queryDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(qfaInfo.opName, VALUE_NAME.c_str(),
                                                      DataTypeToSerialStr(valueDesc->GetDataType()).c_str(),
                                                      "The dtype of value must be the same as dtype of query"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckDtypeConsistency(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAxis(const QuantFlashAttnTilingInfo &qfaInfo)
{
    //   65536 > B > 0；Q_S ≥ 0；KV_S ≥ 0；Q_T ≥ 0、KV_T ≥ 0；D 仅支持 64 或 128
    if (qfaInfo.bSize >= B_LIMIT || qfaInfo.bSize <= 0) {
        std::string reason = "The value of B must be within the range (0, " + std::to_string(B_LIMIT) + ")";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "axis B", std::to_string(qfaInfo.bSize).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if (qfaInfo.layoutQ == FiaLayout::TND) {
        OP_CHECK_IF(
            qfaInfo.queryTSize < 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(qfaInfo.opName, QUERY_NAME.c_str(),
                                                  ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                                                  "T of query must be greater than or equal to 0"),
            return ge::GRAPH_FAILED);
    }
    if (qfaInfo.layoutKV == FiaLayout::TND) {
        OP_CHECK_IF(
            qfaInfo.keyTSize < 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(qfaInfo.opName, KEY_NAME.c_str(),
                                                  ToString(qfaInfo.opParamInfo.key.shape->GetStorageShape()).c_str(),
                                                  "T of key/value must be greater than or equal to 0"),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        qfaInfo.s1Size < 0 && qfaInfo.layoutQ != FiaLayout::TND,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            qfaInfo.opName, QUERY_NAME.c_str(), ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
            "When layout of query is not TND, S of query must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.s2Size < 0,
                OP_LOGE(qfaInfo.opName, "The axis KV_S must be greater than or equal to 0, the current is %ld.",
                        qfaInfo.s2Size),
                return ge::GRAPH_FAILED);

    const std::vector<int64_t> supportedHeadDims = {64, 128};
    OP_CHECK_IF(CheckValueSupport(qfaInfo.qkHeadDim, supportedHeadDims) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(qfaInfo.opName, "axis D of query and key",
                                                       std::to_string(qfaInfo.qkHeadDim).c_str(),
                                                       "The value of axis D of query and key can only be 64/128"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckValueSupport(qfaInfo.vHeadDim, supportedHeadDims) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "axis D of value",
                                                      std::to_string(qfaInfo.vHeadDim).c_str(),
                                                      "The value of axis D of value can only be 64/128"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckHeadNum(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // Q_N % KV_N == 0 且 Q_N / KV_N > 0
    OP_CHECK_IF(
        qfaInfo.n1Size <= 0 || qfaInfo.n2Size <= 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(qfaInfo.opName, "query and key",
                                               (ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                                                ToString(qfaInfo.opParamInfo.key.shape->GetStorageShape()))
                                                   .c_str(),
                                               "N of query and key must be greater than 0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qfaInfo.n1Size < qfaInfo.n2Size,
                OP_LOGE(qfaInfo.opName, "numHeads(%ld) should be greater than or equal to numKeyValueHeads(%ld)!",
                        qfaInfo.n1Size, qfaInfo.n2Size),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        qfaInfo.n1Size % qfaInfo.n2Size != 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(qfaInfo.opName, "query and key",
                                               (ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                                                ToString(qfaInfo.opParamInfo.key.shape->GetStorageShape()))
                                                   .c_str(),
                                               "N of query must be an integer multiple of the same axis of key"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckAxis(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckHeadNum(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
