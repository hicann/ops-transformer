/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file post_quant_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "post_quant_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// CheckSingle
ge::graphStatus PostQuantChecker::CheckSingleDtype(const FiaTilingInfo &fiaInfo)
{
    // QuantScale2 and quantOffset2 only support bf16/fp32 data type.
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.quantScale2.desc, QUANT_SCALE2_NAME)) {
        OP_LOGE(fiaInfo.opName, "QuantScale2 only support bf16/fp32 data type!");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.quantOffset2.desc, QUANT_OFFSET2_NAME)) {
        OP_LOGE(fiaInfo.opName, "QuantOffset2 only support bf16/fp32 data type!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// CheckParaExistence
ge::graphStatus PostQuantChecker::CheckExistenceQuantScale2(const FiaTilingInfo &fiaInfo)
{
    // Post-quantization scenarios must include quantScale2.
    if (fiaInfo.isOutQuantEnable) {
        OP_CHECK_IF(fiaInfo.opParamInfo.quantScale2.tensor == nullptr,
                    OP_LOGE(fiaInfo.opName, "Quant_scale2 is nullptr in post quant scenario!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.opParamInfo.quantScale2.desc == nullptr,
                    OP_LOGE(fiaInfo.opName, "Desc of quant_scale2 is nullptr in post quant scenario!"),
                    return ge::GRAPH_FAILED);
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        OP_CHECK_IF(quantScale2ShapeSize <= 0,
                    OP_LOGE(fiaInfo.opName, "Shape size of quant_scale2 is nonpositive in post quant scenario!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckFeature
ge::graphStatus PostQuantChecker::CheckFeatureAttenOut(const FiaTilingInfo &fiaInfo)
{
    // post-quantization scenarios only support int8/fp8_e4m3fn/hifloat8 output data type
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(
            outputType != ge::DT_INT8 && outputType != ge::DT_FLOAT8_E4M3FN && outputType != ge::DT_HIFLOAT8,
            OP_LOGE(fiaInfo.opName, "The quantScale2 exists, output data type only supports int8/fp8_e4m3fn/hifloat8!"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureQueryDType(const FiaTilingInfo &fiaInfo)
{
    // Post-quantization scale dtype must be FP32. BF16 is allowed only if the query is BF16
    if (fiaInfo.isOutQuantEnable) {
        const ge::DataType quantScale2Type = fiaInfo.opParamInfo.quantScale2.tensor->GetDataType();
        OP_CHECK_IF(
            fiaInfo.inputQType != ge::DT_BF16 && quantScale2Type != ge::DT_FLOAT,
            OP_LOGE(fiaInfo.opName, "When query is not bf16, the post quant scale dtype only supports float32!"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureLayout(const FiaTilingInfo &fiaInfo)
{
    // For post quant per-tensor, quant scale/offset only support [1].
    // For post quant per-channel, quant scale/offset dim multiply result only support qN * vD.
    if (fiaInfo.isOutQuantEnable) {
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        size_t quantScale2Dim = fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape().GetDimNum();
        std::string layoutString = fiaInfo.opParamInfo.layOut;
        bool isSupportedLayout =
            layoutString == "BSH" || layoutString == "BSND" || layoutString == "BNSD" || layoutString == "BNSD_BSND";
        uint32_t numHeads = *(fiaInfo.opParamInfo.numHeads);
        uint64_t quantScale2ShapeSizePerChannel =
            static_cast<uint64_t>(numHeads) * static_cast<uint64_t>(fiaInfo.vHeadDim);
        // per-tensor or per-channel verification
        if (quantScale2Dim == 1) {
            if (static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel || !isSupportedLayout) {
                OP_CHECK_IF((static_cast<uint64_t>(quantScale2ShapeSize) != 1U),
                            OP_LOGE(fiaInfo.opName,
                                    "For post quant per-tensor, quant scale/offset only support [1], now is [%d]",
                                    quantScale2ShapeSize),
                            return ge::GRAPH_FAILED);
            }
        } else {
            if (isSupportedLayout) {
                OP_CHECK_IF((static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel),
                            OP_LOGE(fiaInfo.opName,
                                    "For post quant per-channel, when layout is %s, "
                                    "quantScale2/quantOffset2 dim multiply "
                                    "result only support qN * vD(%u * %u = %lu), now is (%ld).",
                                    layoutString.c_str(), numHeads, fiaInfo.vHeadDim, quantScale2ShapeSizePerChannel,
                                    quantScale2ShapeSize),
                            return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape() !=
                                gert::Shape({numHeads, fiaInfo.vHeadDim}),
                            OP_LOGE(fiaInfo.opName,
                                    "For post quant per-channel, when layout is %s, "
                                    "quantScale2/quantOffset2 expect shape is [%u, %u].",
                                    layoutString.c_str(), numHeads, fiaInfo.vHeadDim),
                            return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureOutputEqual(const FiaTilingInfo &fiaInfo)
{
    // In the Anti-quantization scenario,
    // only KV input and post-quantization outputs with the same data type are supported.
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(outputType != fiaInfo.inputKvType,
                    OP_LOGE(fiaInfo.opName,
                            "The quantScale2 exists, output data type only supports that matches the KV input type!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeaturePrefix(const FiaTilingInfo &fiaInfo)
{
    // When prefix exists, post-quantization scenarios only support int8 output data types.
    if (fiaInfo.isOutQuantEnable) {
        ge::DataType outputType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
        OP_CHECK_IF(fiaInfo.sysPrefixFlag && outputType != ge::DT_INT8,
                    OP_LOGE(fiaInfo.opName, "When prefix exists, the output data type only supports int8!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckFeatureRowValid(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.isOutQuantEnable) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.s1Size > 1) {
        bool checkPostQuantOffset =
            (fiaInfo.outputType == ge::DT_INT8) &&
            (fiaInfo.opParamInfo.quantOffset2.tensor != nullptr && fiaInfo.opParamInfo.quantOffset2.desc != nullptr) &&
            (fiaInfo.opParamInfo.quantOffset2.tensor->GetStorageShape().GetShapeSize() != 0);
        if (!fiaInfo.isMaxWorkspace) {
            std::vector<int64_t> actualSeqLengthsKV{};
            std::vector<int64_t> actualSeqLengths{};
            actualSeqLengthsKV.resize(fiaInfo.bSize);
            actualSeqLengths.resize(fiaInfo.bSize);

            const gert::Tensor *tempData = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
            const gert::Tensor *tempDataKV = fiaInfo.opParamInfo.actualSeqLengths.tensor;
            const int64_t preTokens = fiaInfo.opParamInfo.preToken == nullptr ? 0 : *fiaInfo.opParamInfo.preToken;
            const int64_t nextTokens = fiaInfo.opParamInfo.nextToken == nullptr ? 0 : *fiaInfo.opParamInfo.nextToken;
            uint32_t actualLenDims = (tempData != nullptr) ? tempData->GetShapeSize() : 0;
            uint32_t actualLenDimsKV = (tempDataKV != nullptr) ? tempDataKV->GetShapeSize() : 0;
            for (uint32_t i = 0; i < fiaInfo.bSize; i++) {
                if ((actualLenDims == 0) || (tempData == nullptr) || (tempData->GetData<int64_t>() == nullptr)) {
                    actualSeqLengths[i] = fiaInfo.s1Size;
                } else {
                    actualSeqLengths[i] = (actualLenDims > 1) ? static_cast<uint32_t>(tempData->GetData<int64_t>()[i]) :
                                                                static_cast<uint32_t>(tempData->GetData<int64_t>()[0]);
                }
                if ((actualLenDimsKV == 0) || (tempDataKV == nullptr) ||
                    (tempDataKV->GetData<int64_t>() == nullptr)) { // The user did not input act_seq_kv
                    if (fiaInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
                        actualSeqLengthsKV[i] = fiaInfo.s2Size;
                    } else {
                        actualSeqLengthsKV[i] = fiaInfo.kvListSeqLens[i];
                    }
                } else {
                    actualSeqLengthsKV[i] = (actualLenDimsKV > 1) ?
                                                static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[i]) :
                                                static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[0]);
                }
                int64_t preTokensPerbatch = 0;
                int64_t nextTokensPerbatch = 0;
                if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
                    preTokensPerbatch = static_cast<int64_t>(SPARSE_MODE_INT_MAX);
                    nextTokensPerbatch =
                        actualSeqLengthsKV[i] + static_cast<int64_t>(fiaInfo.systemPrefixLen) - actualSeqLengths[i];
                } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
                    preTokensPerbatch = fiaInfo.preToken - actualSeqLengthsKV[i] -
                                        static_cast<int64_t>(fiaInfo.systemPrefixLen) + actualSeqLengths[i];
                    nextTokensPerbatch = fiaInfo.nextToken + actualSeqLengthsKV[i] +
                                         static_cast<int64_t>(fiaInfo.systemPrefixLen) - actualSeqLengths[i];
                } else {
                    preTokensPerbatch = fiaInfo.preToken;
                    nextTokensPerbatch = fiaInfo.nextToken;
                }
                OP_CHECK_IF(
                    (checkPostQuantOffset && ((preTokensPerbatch + actualSeqLengthsKV[i] +
                                                   static_cast<int64_t>(fiaInfo.systemPrefixLen) - actualSeqLengths[i] <
                                               0) ||
                                              (nextTokensPerbatch < 0))),
                    OPS_REPORT_VECTOR_INNER_ERR(
                        fiaInfo.opName,
                        "When sparse mode = %d, output dtype is int8, the output's dequant offset "
                        "is not null or empty tensor, "
                        "preTokens = %ld and nextTokens = %ld, some rows of the matrix do not "
                        "participate in the calculation, "
                        "the accuracy of the final result will be incorrect. Please see the "
                        "documentation for more details.",
                        fiaInfo.sparseMode, preTokens, nextTokens),
                    return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckAntiquantNotSupport(const FiaTilingInfo &fiaInfo)
{
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        OP_CHECK_IF((fiaInfo.inputKvType == ge::DT_FLOAT8_E4M3FN &&
                     (fiaInfo.outputType != ge::DT_BF16 && fiaInfo.outputType != ge::DT_FLOAT16)),
                    OP_LOGE(fiaInfo.opName, "When keyAntiquantMode and valueAntiquantMode is 1, "
                                            "if data type of key/value is FLOAT8_E4M3FN, post quant is not supported."),
                    return ge::GRAPH_FAILED);
    }

    if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        OP_CHECK_IF((fiaInfo.inputKvType == ge::DT_FLOAT8_E4M3FN &&
                     (fiaInfo.outputType != ge::DT_BF16 && fiaInfo.outputType != ge::DT_FLOAT16)),
                    OP_LOGE(fiaInfo.opName, "When keyAntiquantMode and valueAntiquantMode is 4, "
                                            "if data type of key/value is FLOAT8_E4M3FN, post quant is not supported."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckCheckMultiPara
ge::graphStatus PostQuantChecker::CheckMultiParaQuantOffset2(const FiaTilingInfo &fiaInfo)
{
    // Scale2 and offset2 should have same dtype and shape
    if (fiaInfo.isOutQuantEnable && fiaInfo.opParamInfo.quantOffset2.tensor != nullptr &&
        fiaInfo.opParamInfo.quantOffset2.desc != nullptr) {
        const ge::DataType quantScale2Type = fiaInfo.opParamInfo.quantScale2.tensor->GetDataType();
        int64_t quantScale2ShapeSize = fiaInfo.opParamInfo.quantScale2.tensor->GetShapeSize();
        size_t quantScale2Dim = fiaInfo.opParamInfo.quantScale2.tensor->GetStorageShape().GetDimNum();

        const ge::DataType quantOffset2Datatype = fiaInfo.opParamInfo.quantOffset2.tensor->GetDataType();
        int64_t quantOffset2ShapeSize = fiaInfo.opParamInfo.quantOffset2.tensor->GetShapeSize();
        size_t quantOffset2Dim = fiaInfo.opParamInfo.quantOffset2.tensor->GetStorageShape().GetDimNum();

        OP_CHECK_IF(quantOffset2Datatype != quantScale2Type,
                    OP_LOGE(fiaInfo.opName, "QuantScale2 and quantOffset2 should have same dtype!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(quantOffset2Dim != quantScale2Dim,
                    OP_LOGE(fiaInfo.opName, "QuantScale2 and quantOffset2 should have same dim!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(quantOffset2ShapeSize != quantScale2ShapeSize,
                    OP_LOGE(fiaInfo.opName, "QuantScale2 and quantOffset2 should have same shape size!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSingleDtype(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceQuantScale2(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureAttenOut(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeatureQueryDType(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeaturePrefix(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
        if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
            if (ge::GRAPH_SUCCESS != CheckFeatureRowValid(fiaInfo)) {
                return ge::GRAPH_FAILED;
            }
        }
    } else if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureOutputEqual(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckAntiquantNotSupport(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PostQuantChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckMultiParaQuantOffset2(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
