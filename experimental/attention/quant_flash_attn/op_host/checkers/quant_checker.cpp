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
 * \file quant_checker.cpp
 * \brief Checker for quant_mode, q_descale, k_descale, v_descale, p_scale, layout_q_descale
 *        (文档约束: 全量化参数组)
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
#include "quant_checker.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;

ge::graphStatus QuantChecker::CheckSingleParaQuantMode(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // data_type 支持 INT32；当前仅实现 quant_mode=5（MxFP4）
    // quant_mode 为属性, parser 中以 const int64_t* 存储, 此处校验其原始值范围
    if (qfaInfo.opParamInfo.quantMode == nullptr) {
        return ge::GRAPH_SUCCESS; // 存在性校验负责
    }
    const std::vector<int64_t> supportedQuantModes = {
        QUANT_MODE_A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16,
    };
    int64_t quantModeVal = *qfaInfo.opParamInfo.quantMode;
    OP_CHECK_IF(
        std::find(supportedQuantModes.begin(), supportedQuantModes.end(), quantModeVal) == supportedQuantModes.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            qfaInfo.opName, QUANT_MODE_NAME.c_str(), std::to_string(quantModeVal).c_str(),
            "The value of quant_mode must be A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16 (5)"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaQDescale(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 支持 4、5
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.qDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS; // 存在性校验负责
    }

    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, Q_DESCALE_NAME.c_str(),
                                          DataTypeToSerialStr(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, Q_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const std::vector<uint32_t> supportedDims = {DIM_NUM_4, DIM_NUM_5};
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(std::find(supportedDims.begin(), supportedDims.end(), dimNum) == supportedDims.end(),
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, Q_DESCALE_NAME.c_str(),
                                             (std::to_string(dimNum) + "D").c_str(), "4D/5D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaKDescale(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 支持 4、5、6
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.kDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, K_DESCALE_NAME.c_str(),
                                          DataTypeToSerialStr(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, K_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const std::vector<uint32_t> supportedDims = {DIM_NUM_4, DIM_NUM_5, DIM_NUM_6};
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(std::find(supportedDims.begin(), supportedDims.end(), dimNum) == supportedDims.end(),
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, K_DESCALE_NAME.c_str(),
                                             (std::to_string(dimNum) + "D").c_str(), "4D/5D/6D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaVDescale(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 支持 4、5、6
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.vDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, V_DESCALE_NAME.c_str(),
                                          DataTypeToSerialStr(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, V_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const std::vector<uint32_t> supportedDims = {DIM_NUM_4, DIM_NUM_5, DIM_NUM_6};
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(std::find(supportedDims.begin(), supportedDims.end(), dimNum) == supportedDims.end(),
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, V_DESCALE_NAME.c_str(),
                                             (std::to_string(dimNum) + "D").c_str(), "4D/5D/6D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaPScale(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // tensor_type 仅支持 FLOAT32；shape 仅支持 (1,)
    // p_scale 为可选参数，未传入时跳过
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.pScale.desc;
    const gert::Tensor *tensor = qfaInfo.opParamInfo.pScale.tensor;
    if (desc == nullptr || tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, P_SCALE_NAME.c_str(),
                                          DataTypeToSerialStr(desc->GetDataType()).c_str(), "FLOAT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, P_SCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = tensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, P_SCALE_NAME.c_str(),
                                             (std::to_string(dimNum) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);

    int64_t dim0 = tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(dim0 != 1, OP_LOGE(qfaInfo.opName, "p_scale shape must be (1,), but got dim0=%ld", dim0),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaLayoutQDescale(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // data_type 支持 STRING；支持输入 BNSD/BSND/TND/N2TGD
    // FiaLayout 枚举未引入 N2TGD, parser 仅解析 BSND/BNSD/TND, checker 对齐 parser 实际值。
    const std::vector<FiaLayout> supportedLayouts = {FiaLayout::BNSD, FiaLayout::BSND, FiaLayout::TND};
    OP_CHECK_IF(
        std::find(supportedLayouts.begin(), supportedLayouts.end(), qfaInfo.layoutQDescale) == supportedLayouts.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, LAYOUT_Q_DESCALE_NAME.c_str(),
                                              LayoutToSerialStr(qfaInfo.layoutQDescale).c_str(),
                                              "The value of layout_q_descale must be in BNSD/BSND/TND"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckSingleParaQuantMode(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaQDescale(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaKDescale(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaVDescale(qfaInfo) != ge::GRAPH_SUCCESS || CheckSingleParaPScale(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaLayoutQDescale(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // quant_mode: 必选属性
    OP_CHECK_IF(qfaInfo.opParamInfo.quantMode == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, QUANT_MODE_NAME.c_str()), return ge::GRAPH_FAILED);

    // q_descale / k_descale / v_descale: 必须存在
    OP_CHECK_IF(qfaInfo.opParamInfo.qDescale.desc == nullptr || qfaInfo.opParamInfo.qDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, Q_DESCALE_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.kDescale.desc == nullptr || qfaInfo.opParamInfo.kDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, K_DESCALE_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.vDescale.desc == nullptr || qfaInfo.opParamInfo.vDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, V_DESCALE_NAME.c_str()), return ge::GRAPH_FAILED);

    // p_scale: 可选参数，默认值为 [1.0f]，不强制存在
    return ge::GRAPH_SUCCESS;
}

namespace {
struct QuantLayoutConstraintConfig {
    FiaLayout supportedQLayout;
    std::vector<FiaLayout> supportedKvLayouts;
    FiaLayout supportedOutLayout;
};

const std::map<int64_t, QuantLayoutConstraintConfig> QUANT_LAYOUT_CONSTRAINT_TABLE = {
    {QUANT_MODE_MXFP8, {FiaLayout::TND, {FiaLayout::TND, FiaLayout::BnNBsD, FiaLayout::NZ}, FiaLayout::TND}}, // MxFP8
    {QUANT_MODE_MXFP4, {FiaLayout::BNSD, {FiaLayout::BNSD}, FiaLayout::BNSD}},                                // MxFP4
};
} // namespace

ge::graphStatus QuantChecker::CheckLayoutConstraint(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    if (qfaInfo.opParamInfo.quantMode == nullptr) {
        return ge::GRAPH_SUCCESS; // 存在性校验负责
    }
    int64_t quantModeVal = *qfaInfo.opParamInfo.quantMode;
    auto it = QUANT_LAYOUT_CONSTRAINT_TABLE.find(quantModeVal);
    OP_CHECK_IF(it == QUANT_LAYOUT_CONSTRAINT_TABLE.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, QUANT_MODE_NAME.c_str(),
                                                      std::to_string(quantModeVal).c_str(),
                                                      "quant_mode is not supported in layout constraint table"),
                return ge::GRAPH_FAILED);

    const auto &config = it->second;

    OP_CHECK_IF(qfaInfo.layoutQ != config.supportedQLayout,
                OP_LOGE(qfaInfo.opName, "When quant_mode is %ld, layout_q must be %s, but got %s", quantModeVal,
                        LayoutToSerialStr(config.supportedQLayout).c_str(), LayoutToSerialStr(qfaInfo.layoutQ).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(config.supportedKvLayouts.begin(), config.supportedKvLayouts.end(), qfaInfo.layoutKV) ==
                    config.supportedKvLayouts.end(),
                OP_LOGE(qfaInfo.opName, "When quant_mode is %ld, layout_kv must be in supported list, but got %s",
                        quantModeVal, LayoutToSerialStr(qfaInfo.layoutKV).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        qfaInfo.layoutOut != config.supportedOutLayout,
        OP_LOGE(qfaInfo.opName, "When quant_mode is %ld, layout_out must be %s, but got %s", quantModeVal,
                LayoutToSerialStr(config.supportedOutLayout).c_str(), LayoutToSerialStr(qfaInfo.layoutOut).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckQueryShape(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    const gert::StorageShape *shape = qfaInfo.opParamInfo.query.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    // MxFP8 TND: (Q_T, Q_N, D); MxFP4 BNSD: (B, Q_N, Q_S, D)
    std::vector<int64_t> expected;
    if (qfaInfo.layoutQ == FiaLayout::TND) {
        expected = {qfaInfo.queryTSize, qfaInfo.n1Size, qfaInfo.qkHeadDim};
    } else if (qfaInfo.layoutQ == FiaLayout::BNSD) {
        expected = {qfaInfo.bSize, qfaInfo.n1Size, qfaInfo.s1Size, qfaInfo.qkHeadDim};
    } else {
        OP_LOGE(qfaInfo.opName, "query shape check: layout_q %s is unsupported.",
                LayoutToSerialStr(qfaInfo.layoutQ).c_str());
        return ge::GRAPH_FAILED;
    }
    return CheckShapeEqual(*shape, expected, QUERY_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckKVShape(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    const gert::StorageShape *keyShape = qfaInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = qfaInfo.opParamInfo.value.shape;
    if (keyShape == nullptr && valueShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // MxFP8 TND: (KV_T, KV_N, D); PA_BNBD: (Bn, KV_N, Bs, D); PA_NZ: (Bn, KV_N, D/32, Bs, 32)
    // MxFP4 BNSD: (B, KV_N, KV_S, D)
    std::vector<int64_t> expected;
    if (qfaInfo.layoutKV == FiaLayout::TND) {
        expected = {qfaInfo.keyTSize, qfaInfo.n2Size, qfaInfo.qkHeadDim};
    } else if (qfaInfo.layoutKV == FiaLayout::BnNBsD) {
        expected = {qfaInfo.maxBlockNumPerBatch, qfaInfo.n2Size, qfaInfo.blockSize, qfaInfo.qkHeadDim};
    } else if (qfaInfo.layoutKV == FiaLayout::NZ) {
        expected = {qfaInfo.maxBlockNumPerBatch, qfaInfo.n2Size, qfaInfo.qkHeadDim / 32, qfaInfo.blockSize, 32};
    } else if (qfaInfo.layoutKV == FiaLayout::BNSD) {
        expected = {qfaInfo.bSize, qfaInfo.n2Size, qfaInfo.s2Size, qfaInfo.qkHeadDim};
    } else {
        OP_LOGE(qfaInfo.opName, "k/v shape check: layout_kv %s is unsupported.",
                LayoutToSerialStr(qfaInfo.layoutKV).c_str());
        return ge::GRAPH_FAILED;
    }

    if (keyShape != nullptr && CheckShapeEqual(*keyShape, expected, KEY_NAME, qfaInfo.opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // v 的 D 用 vHeadDim（与 k 的 qkHeadDim 区分）
    if (qfaInfo.layoutKV == FiaLayout::NZ) {
        expected[2] = qfaInfo.vHeadDim / 32;
    } else if (qfaInfo.layoutKV == FiaLayout::TND || qfaInfo.layoutKV == FiaLayout::BnNBsD ||
               qfaInfo.layoutKV == FiaLayout::BNSD) {
        expected[expected.size() - 1] = qfaInfo.vHeadDim;
    }
    if (valueShape != nullptr &&
        CheckShapeEqual(*valueShape, expected, VALUE_NAME, qfaInfo.opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckAttnOutShape(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    const gert::StorageShape *shape = qfaInfo.opParamInfo.attnOut.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    // MxFP8 TND: (Q_T, Q_N, D); MxFP4 BNSD: (B, Q_N, Q_S, D)
    // attn_out 的 D 取 vHeadDim（反量化后输出 dtype 为 BF16）
    std::vector<int64_t> expected;
    if (qfaInfo.layoutOut == FiaLayout::TND) {
        expected = {qfaInfo.queryTSize, qfaInfo.n1Size, qfaInfo.vHeadDim};
    } else if (qfaInfo.layoutOut == FiaLayout::BNSD) {
        expected = {qfaInfo.bSize, qfaInfo.n1Size, qfaInfo.s1Size, qfaInfo.vHeadDim};
    } else {
        OP_LOGE(qfaInfo.opName, "attn_out shape check: layout_out %s is unsupported.",
                LayoutToSerialStr(qfaInfo.layoutOut).c_str());
        return ge::GRAPH_FAILED;
    }
    return CheckShapeEqual(*shape, expected, ATTEN_OUT_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckShapeMatch(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    if (CheckQueryShape(qfaInfo) != ge::GRAPH_SUCCESS || CheckKVShape(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckAttnOutShape(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMxFp4Constraint(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    if (qfaInfo.opParamInfo.quantMode == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t quantModeVal = *qfaInfo.opParamInfo.quantMode;
    if (quantModeVal != QUANT_MODE_MXFP4) { // 非 MxFP4 跳过
        return ge::GRAPH_SUCCESS;
    }

    // MxFP4: layout_q_descale 仅支持 BNSD
    OP_CHECK_IF(qfaInfo.layoutQDescale != FiaLayout::BNSD,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, LAYOUT_Q_DESCALE_NAME.c_str(),
                                                      LayoutToSerialStr(qfaInfo.layoutQDescale).c_str(),
                                                      "MxFP4 only supports layout_q_descale = BNSD"),
                return ge::GRAPH_FAILED);

    // MxFP4: 仅支持 Q_N == KV_N (即 G=1)
    OP_CHECK_IF(
        qfaInfo.n1Size != qfaInfo.n2Size,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(qfaInfo.opName, "query and key",
                                               (ToString(qfaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                                                ToString(qfaInfo.opParamInfo.key.shape->GetStorageShape()))
                                                   .c_str(),
                                               "MxFP4 only supports Q_N == KV_N (G=1, GQA is not supported)"),
        return ge::GRAPH_FAILED);

    // MxFP4: D 仅支持 128
    const std::vector<int64_t> supportedHeadDims = {128};
    OP_CHECK_IF(
        CheckValueSupport(qfaInfo.qkHeadDim, supportedHeadDims) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "axis D of query and key",
                                              std::to_string(qfaInfo.qkHeadDim).c_str(), "MxFP4 only supports D=128"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckValueSupport(qfaInfo.vHeadDim, supportedHeadDims) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "axis D of value",
                                              std::to_string(qfaInfo.vHeadDim).c_str(), "MxFP4 only supports D=128"),
        return ge::GRAPH_FAILED);

    if (CheckMxFp4QkvDtype(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMxFp4QkvDtype(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    const gert::CompileTimeTensorDesc *queryDesc = qfaInfo.opParamInfo.query.desc;
    const gert::CompileTimeTensorDesc *keyDesc = qfaInfo.opParamInfo.key.desc;
    const gert::CompileTimeTensorDesc *valueDesc = qfaInfo.opParamInfo.value.desc;
    OP_CHECK_IF(queryDesc != nullptr && queryDesc->GetDataType() != ge::DT_FLOAT4_E2M1,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, QUERY_NAME.c_str(),
                                          DataTypeToSerialStr(queryDesc->GetDataType()).c_str(), "FLOAT4_E2M1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDesc != nullptr && keyDesc->GetDataType() != ge::DT_FLOAT4_E2M1,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, KEY_NAME.c_str(),
                                          DataTypeToSerialStr(keyDesc->GetDataType()).c_str(), "FLOAT4_E2M1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valueDesc != nullptr && valueDesc->GetDataType() != ge::DT_FLOAT4_E2M1,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, VALUE_NAME.c_str(),
                                          DataTypeToSerialStr(valueDesc->GetDataType()).c_str(), "FLOAT4_E2M1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckLayoutConstraint(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeMatch(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMxFp4Constraint(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckShapeEqual(const gert::StorageShape &actual, const std::vector<int64_t> &expected,
                                              const std::string &paraName, const char *opName) const
{
    if (actual.GetStorageShape().GetDimNum() != expected.size()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, paraName.c_str(),
                                     (std::to_string(actual.GetStorageShape().GetDimNum()) + "D").c_str(),
                                     (std::to_string(expected.size()) + "D").c_str());
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < expected.size(); i++) {
        int64_t actDim = actual.GetStorageShape().GetDim(i);
        // expected[i] < 0 表示该维度不做数值校验(通配), 用于依赖 device 数据的维度
        if (expected[i] >= 0 && actDim != expected[i]) {
            OP_LOGE(opName, "%s shape dim[%zu] is %ld, expected %ld.", paraName.c_str(), i, actDim, expected[i]);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckQDescaleShape(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t D = qfaInfo.qkHeadDim;
    int64_t DPerGroup = D / 64; // block size = 64
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();

    std::vector<int64_t> expected;
    if (dimNum == DIM_NUM_4) {
        // MxFP8 prefill: (Q_T, Q_N, D/64, 2)
        expected = {qfaInfo.queryTSize, qfaInfo.n1Size, DPerGroup, 2};
    } else if (dimNum == DIM_NUM_5) {
        if (qfaInfo.layoutQ == FiaLayout::BNSD) {
            // MxFP4 BNSD: (B, Q_N, Q_S, D/64, 2)
            expected = {qfaInfo.bSize, qfaInfo.n1Size, qfaInfo.s1Size, DPerGroup, 2};
        } else {
            // MxFP8 decode: (KV_N, Q_T, G, D/64, 2)
            expected = {qfaInfo.n2Size, qfaInfo.queryTSize, qfaInfo.gSize, DPerGroup, 2};
        }
    } else {
        OP_LOGE(qfaInfo.opName, "q_descale shape dim %u is unsupported, expected 4D or 5D.", dimNum);
        return ge::GRAPH_FAILED;
    }

    return CheckShapeEqual(*shape, expected, Q_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckKDescaleShape(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t D = qfaInfo.qkHeadDim;
    int64_t DPerGroup = D / 64;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();

    std::vector<int64_t> expected;
    if (qfaInfo.layoutKV == FiaLayout::TND) {
        expected = {qfaInfo.keyTSize, qfaInfo.n2Size, DPerGroup, 2};
    } else if (qfaInfo.layoutKV == FiaLayout::BnNBsD) {
        expected = {qfaInfo.maxBlockNumPerBatch, qfaInfo.n2Size, qfaInfo.blockSize, DPerGroup, 2};
    } else if (qfaInfo.layoutKV == FiaLayout::NZ) {
        expected = {qfaInfo.maxBlockNumPerBatch, qfaInfo.n2Size, qfaInfo.blockSize / 16, DPerGroup, 16, 2};
    } else if (qfaInfo.layoutKV == FiaLayout::BNSD) {
        // MxFP4 BNSD: (B, KV_N, KV_S, D/64, 2)
        expected = {qfaInfo.bSize, qfaInfo.n2Size, qfaInfo.s2Size, DPerGroup, 2};
    } else {
        OP_LOGE(qfaInfo.opName, "k_descale shape check: kv_layout %s is unsupported.",
                LayoutToSerialStr(qfaInfo.layoutKV).c_str());
        return ge::GRAPH_FAILED;
    }

    if (expected.size() != dimNum) {
        OP_LOGE(qfaInfo.opName, "k_descale shape dim %u does not match layout %s, expected %zuD.", dimNum,
                LayoutToSerialStr(qfaInfo.layoutKV).c_str(), expected.size());
        return ge::GRAPH_FAILED;
    }
    return CheckShapeEqual(*shape, expected, K_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckDescaleShape(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // v_descale shape 不做校验, 仅校验 q_descale / k_descale
    if (CheckQDescaleShape(qfaInfo) != ge::GRAPH_SUCCESS || CheckKDescaleShape(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckDescaleDtype(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    //   MxFP8/MxFP4 场景下, q/k/v descale 的 tensor_type 仅支持 FLOAT8_E8M0
    if (qfaInfo.opParamInfo.quantMode == nullptr) {
        return ge::GRAPH_SUCCESS; // 存在性校验负责
    }
    int64_t quantModeVal = *qfaInfo.opParamInfo.quantMode;
    // MxFP8(1) 与 MxFP4(5) 场景下 descale dtype 限定为 FLOAT8_E8M0; 其他 mode 不限制
    if (quantModeVal != QUANT_MODE_MXFP8 && quantModeVal != QUANT_MODE_MXFP4) {
        return ge::GRAPH_SUCCESS;
    }

    const auto CheckDescaleDtypeFn = [&qfaInfo, this](const gert::CompileTimeTensorDesc *desc,
                                                      const std::string &paraName) -> ge::graphStatus {
        if (desc == nullptr) {
            return ge::GRAPH_SUCCESS; // 存在性校验负责
        }
        OP_CHECK_IF(desc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, paraName.c_str(),
                                              DataTypeToSerialStr(desc->GetDataType()).c_str(), "FLOAT8_E8M0"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    if (CheckDescaleDtypeFn(qfaInfo.opParamInfo.qDescale.desc, Q_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDescaleDtypeFn(qfaInfo.opParamInfo.kDescale.desc, K_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDescaleDtypeFn(qfaInfo.opParamInfo.vDescale.desc, V_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckDescaleShape(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDescaleDtype(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
