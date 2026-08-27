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
 * \file quant_checker.cpp
 * \brief Checker for quant_mode, q_descale, k_descale, v_descale, p_scale ( 全量化参数组)
 */

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../qfa_tiling_info.h"
#include "../quant_flash_attn_tiling_utils.h"
#include "quant_checker.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;

namespace {
// descale dtype 期望值表: 各量化场景下 q/k/v descale 的 tensor_type 要求
//   MxFP8: FLOAT8_E8M0; GQA_FP8_FULLQUANT: FLOAT32; HIF8: FLOAT32
const std::map<QfaQuantMode, std::pair<ge::DataType, std::string>> DESCALE_DTYPE_TABLE = {
    {QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32, {ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0"}},
    {QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32,
     {ge::DT_FLOAT, "FLOAT32"}},
    {QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32, {ge::DT_FLOAT, "FLOAT32"}},
};
} // namespace

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus QuantChecker::CheckSingleParaQuantMode(const QfaTilingInfo &qfaInfo)
{
    //  data_type 支持 INT32；当前支持 quant_mode = 1、0
    // quantMode 为属性，QfaTilingInfo 中存储为 QfaQuantMode 枚举
    const std::vector<uint32_t> supportedQuantModes = {1, 0};
    uint32_t quantModeVal = static_cast<uint32_t>(qfaInfo.quantMode);
    OP_CHECK_IF(
        std::find(supportedQuantModes.begin(), supportedQuantModes.end(), quantModeVal) == supportedQuantModes.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "quant_mode", std::to_string(quantModeVal).c_str(),
                                              "The value of quant_mode must be 1 or 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckQDescaleDimMxFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    const std::vector<uint32_t> supportedDims = {DIM_NUM_4, DIM_NUM_5};
    OP_CHECK_IF(std::find(supportedDims.begin(), supportedDims.end(), dimNum) == supportedDims.end(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    qfaInfo.opName, Q_DESCALE_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(),
                    "In MxFP8 scenario, the shape dim of q_descale must be 4D or 5D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckQDescaleDimGqaFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode !=
        QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    // GQA: q_descale 为 2D [N1, T]
    OP_CHECK_IF(dimNum != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    qfaInfo.opName, Q_DESCALE_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(),
                    "In GQA_FP8_FULLQUANT scenario, the shape dim of q_descale must be 2D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckQDescaleDimHif8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(qfaInfo.opName, Q_DESCALE_NAME.c_str(),
                                                         (std::to_string(dimNum) + "D").c_str(),
                                                         "In HIF8 scenario, the shape dim of q_descale must be 1D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaQDescale(const QfaTilingInfo &qfaInfo)
{
    //  tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 按场景区分
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.qDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS; // 存在性校验负责
    }

    // 公共: dtype / format 校验
    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, Q_DESCALE_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, Q_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 场景: shape dim 校验 (MxFP8/FP8/HIF8)
    if (CheckQDescaleDimMxFp8(qfaInfo) != ge::GRAPH_SUCCESS || CheckQDescaleDimGqaFp8(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckQDescaleDimHif8(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckKDescaleDimMxFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    const std::vector<uint32_t> supportedDims = {DIM_NUM_4, DIM_NUM_5, DIM_NUM_6};
    OP_CHECK_IF(std::find(supportedDims.begin(), supportedDims.end(), dimNum) == supportedDims.end(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    qfaInfo.opName, K_DESCALE_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(),
                    "In MxFP8 scenario, the shape dim of k_descale must be 4D/5D/6D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckKDescaleDimGqaFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode !=
        QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    // GQA: k_descale 为 3D [Bn, N2, Bs]
    OP_CHECK_IF(dimNum != DIM_NUM_3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    qfaInfo.opName, K_DESCALE_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(),
                    "In GQA_FP8_FULLQUANT scenario, the shape dim of k_descale must be 3D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckKDescaleDimHif8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(qfaInfo.opName, K_DESCALE_NAME.c_str(),
                                                         (std::to_string(dimNum) + "D").c_str(),
                                                         "In HIF8 scenario, the shape dim of k_descale must be 1D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaKDescale(const QfaTilingInfo &qfaInfo)
{
    //  tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 按场景区分
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.kDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // 公共: dtype / format 校验
    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, K_DESCALE_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, K_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 场景: shape dim 校验 (MxFP8/FP8/HIF8)
    if (CheckKDescaleDimMxFp8(qfaInfo) != ge::GRAPH_SUCCESS || CheckKDescaleDimGqaFp8(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckKDescaleDimHif8(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckVDescaleDimMxFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    const std::vector<uint32_t> supportedDims = {DIM_NUM_4, DIM_NUM_5, DIM_NUM_6};
    OP_CHECK_IF(std::find(supportedDims.begin(), supportedDims.end(), dimNum) == supportedDims.end(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    qfaInfo.opName, V_DESCALE_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(),
                    "In MxFP8 scenario, the shape dim of v_descale must be 4D/5D/6D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckVDescaleDimGqaFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode !=
        QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    // GQA: v_descale 为 1D [N2]
    OP_CHECK_IF(dimNum != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    qfaInfo.opName, V_DESCALE_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(),
                    "In GQA_FP8_FULLQUANT scenario, the shape dim of v_descale must be 1D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckVDescaleDimHif8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(qfaInfo.opName, V_DESCALE_NAME.c_str(),
                                                         (std::to_string(dimNum) + "D").c_str(),
                                                         "In HIF8 scenario, the shape dim of v_descale must be 1D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaVDescale(const QfaTilingInfo &qfaInfo)
{
    //  tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 按场景区分
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.vDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // 公共: dtype / format 校验
    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, V_DESCALE_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, V_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 场景: shape dim 校验 (MxFP8/FP8/HIF8)
    if (CheckVDescaleDimMxFp8(qfaInfo) != ge::GRAPH_SUCCESS || CheckVDescaleDimGqaFp8(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckVDescaleDimHif8(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaPScale(const QfaTilingInfo &qfaInfo)
{
    //  tensor_type 仅支持 FLOAT32；shape 仅支持 (1,)
    // p_scale 为可选参数，未传入时跳过
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.pScale.desc;
    const gert::Tensor *tensor = qfaInfo.opParamInfo.pScale.tensor;
    if (desc == nullptr || tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, P_SCALE_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT32"),
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
    OP_CHECK_IF(dim0 != 1,
                OP_LOGE_FOR_INVALID_SHAPE(qfaInfo.opName, P_SCALE_NAME.c_str(),
                                          ("[" + std::to_string(dim0) + "]").c_str(), "[1]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    if (CheckSingleParaQuantMode(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaQDescale(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaKDescale(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaVDescale(qfaInfo) != ge::GRAPH_SUCCESS || CheckSingleParaPScale(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence
// ============================================================================

ge::graphStatus QuantChecker::CheckParaExistenceGqaFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode !=
        QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // GQA_FP8_FULLQUANT: block_table 必选（GQA 强制 PA 场景）
    OP_CHECK_IF(qfaInfo.opParamInfo.blockTable.tensor == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, BLOCK_TABLE_NAME.c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckParaExistence(const QfaTilingInfo &qfaInfo)
{
    // 公共: quant_mode 必选属性
    OP_CHECK_IF(qfaInfo.opParamInfo.quantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, "quant_mode"),
                return ge::GRAPH_FAILED);

    // 公共: q_descale / k_descale / v_descale 必须存在
    OP_CHECK_IF(qfaInfo.opParamInfo.qDescale.desc == nullptr || qfaInfo.opParamInfo.qDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, Q_DESCALE_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.kDescale.desc == nullptr || qfaInfo.opParamInfo.kDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, K_DESCALE_NAME.c_str()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.vDescale.desc == nullptr || qfaInfo.opParamInfo.vDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, V_DESCALE_NAME.c_str()), return ge::GRAPH_FAILED);

    // 场景: GQA_FP8_FULLQUANT 下 block_table 必选
    if (CheckParaExistenceGqaFp8(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // p_scale: 可选参数，默认值为 [1.0f]，不强制存在
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// MultiPara — descale shape consistency (文档"一致性校验"列: descale_shape匹配关系表)
// ============================================================================

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
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, paraName.c_str(),
                                                  ("[" + std::to_string(actDim) + "]").c_str(),
                                                  ("The value of dim[" + std::to_string(i) + "] of " + paraName +
                                                   " shape must be " + std::to_string(expected[i]))
                                                      .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckQDescaleShapeMxFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // 文档(descale_shape匹配关系表): MxFP8, layout_q=TND
    //   4D: (Q_T, Q_N, D/64, 2)              prefill场景，layout_q_descale=TND
    //   5D: (KV_N, Q_T, G, D/64, 2)          decode场景，layout_q_descale=N2TGD
    //   其中 G = Q_N / KV_N
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    int64_t D = qfaInfo.qkHeadDim;
    int64_t dPerGroup = (D + 63) / 64; // MxFP8 block size = 64
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();

    std::vector<int64_t> expected;
    if (dimNum == DIM_NUM_4) {
        expected = {qfaInfo.qTSize, qfaInfo.n1Size, dPerGroup, 2};
    } else if (dimNum == DIM_NUM_5) {
        expected = {qfaInfo.n2Size, qfaInfo.qTSize, qfaInfo.gSize, dPerGroup, 2};
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(qfaInfo.opName, Q_DESCALE_NAME.c_str(),
                                                 (std::to_string(dimNum) + "D").c_str(),
                                                 "In MxFP8 scenario, the shape dim of q_descale must be 4D or 5D");
        return ge::GRAPH_FAILED;
    }

    // 约束: MxFP8场景下 q_descale 两种 shape 对应不同使用场景
    //   4D 为 prefill 场景，layout_q_descale 必须为 TND；5D 为 decode 场景，layout_q_descale 必须为 N2TGD
    std::string shapeStr = Ops::Base::ToString(shape->GetStorageShape());
    bool isDecode = (qfaInfo.layoutQDescale == QfaLayout::N2TGD);
    if (dimNum == DIM_NUM_4) {
        OP_CHECK_IF(isDecode,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, Q_DESCALE_NAME.c_str(),
                                                          QfaLayoutToSerialString(qfaInfo.layoutQDescale).c_str(),
                                                          ("q_descale shape " + shapeStr +
                                                           " is for prefill scene, layout_q_descale must be TND, "
                                                           "but got " +
                                                           QfaLayoutToSerialString(qfaInfo.layoutQDescale))
                                                              .c_str()),
                    return ge::GRAPH_FAILED);
    } else { // DIM_NUM_5, decode scene
        OP_CHECK_IF(!isDecode,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, Q_DESCALE_NAME.c_str(),
                                                          QfaLayoutToSerialString(qfaInfo.layoutQDescale).c_str(),
                                                          ("q_descale shape " + shapeStr +
                                                           " is for decode scene, layout_q_descale must be N2TGD, "
                                                           "but got " +
                                                           QfaLayoutToSerialString(qfaInfo.layoutQDescale))
                                                              .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return CheckShapeEqual(*shape, expected, Q_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckQDescaleShapeGqaFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode !=
        QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // GQA_FP8_FULLQUANT, layout_q_descale=NT: 2D (N1, T)
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    std::vector<int64_t> expected = {qfaInfo.n1Size, qfaInfo.qTSize};
    return CheckShapeEqual(*shape, expected, Q_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckQDescaleShape(const QfaTilingInfo &qfaInfo) const
{
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // HIF8 per-tensor: q_descale shape must be (1,), 1D
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        return CheckShapeEqual(*shape, {1}, Q_DESCALE_NAME, qfaInfo.opName);
    }

    if (CheckQDescaleShapeMxFp8(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckQDescaleShapeGqaFp8(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckKDescaleShapeMxFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // 文档(k_descale/v_descale shape匹配关系表): MxFP8
    //   TND:      (KV_T, KV_N, D/64, 2)              - 4D
    //   PA_BBND:  (Bn, Bs, KV_N, D/64, 2)            - 5D
    //   PA_BNBD:  (Bn, KV_N, Bs, D/64, 2)            - 5D
    //   PA_NZ:    (Bn, KV_N, Bs/16, D/64, 16, 2)     - 6D
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    int64_t D = qfaInfo.qkHeadDim;
    int64_t dPerGroup = (D + 63) / 64;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();

    std::vector<int64_t> expected;
    if (qfaInfo.kvLayout == QfaLayout::TND) {
        expected = {qfaInfo.kTSize, qfaInfo.n2Size, dPerGroup, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_BBND) {
        expected = {qfaInfo.totalBlockNum, qfaInfo.blockSize, qfaInfo.n2Size, dPerGroup, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_BNBD) {
        expected = {qfaInfo.totalBlockNum, qfaInfo.n2Size, qfaInfo.blockSize, dPerGroup, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_NZ) {
        expected = {qfaInfo.totalBlockNum, qfaInfo.n2Size, qfaInfo.blockSize / 16, dPerGroup, 16, 2};
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            qfaInfo.opName, "layout_kv", QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(),
            "In MxFP8 scenario, the kv_layout for k_descale shape check is unsupported");
        return ge::GRAPH_FAILED;
    }

    if (expected.size() != dimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            qfaInfo.opName, K_DESCALE_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(),
            ("In MxFP8 scenario, the shape dim of k_descale must be " + std::to_string(expected.size()) +
             "D when layout_kv is " + QfaLayoutToSerialString(qfaInfo.kvLayout))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return CheckShapeEqual(*shape, expected, K_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckKDescaleShapeGqaFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode !=
        QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // GQA_FP8_FULLQUANT (PA_BNBD 强制): 3D (Bn, N2, Bs)
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    OP_CHECK_IF(qfaInfo.kvLayout != QfaLayout::PA_BNBD,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "layout_kv",
                                                      QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(),
                                                      "In GQA_FP8_FULLQUANT scenario, layout_kv must be PA_BNBD"),
                return ge::GRAPH_FAILED);
    std::vector<int64_t> expected = {qfaInfo.totalBlockNum, qfaInfo.n2Size, qfaInfo.blockSize};
    return CheckShapeEqual(*shape, expected, K_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckKDescaleShape(const QfaTilingInfo &qfaInfo) const
{
    // 公共: 空指针跳过
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // HIF8 per-tensor: k_descale shape must be (1,), 1D
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        return CheckShapeEqual(*shape, {1}, K_DESCALE_NAME, qfaInfo.opName);
    }

    if (CheckKDescaleShapeMxFp8(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckKDescaleShapeGqaFp8(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckVDescaleShapeMxFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // 文档(k_descale/v_descale shape匹配关系表): MxFP8
    //   TND:      (KV_T/64, KV_N, D, 2)              - 4D
    //   PA_BBND:  (Bn, Bs/64, KV_N, D, 2)            - 5D
    //   PA_BNBD:  (Bn, KV_N, Bs/64, D, 2)            - 5D
    //   PA_NZ:    (Bn, KV_N, D/16, Bs/64, 16, 2)     - 6D
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    const int64_t mxfp8BlockSize = 64;
    int64_t D = qfaInfo.vHeadDim; // v_descale 的 D 用 vHeadDim
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();

    std::vector<int64_t> expected;
    if (qfaInfo.kvLayout == QfaLayout::TND) {
        // TND(非PA)场景: dim0 = Σ ceil(curLen, 64), 依赖 cu_seqlens_kv 的实际数值。
        // tiling 阶段 cu_seqlens_kv 为 device tensor, host 侧无法安全读取,
        // 因此 dim0 不做数值校验(用 -1 占位), 仅校验维度数为 4D 及 dim[1..3]。
        expected = {-1, qfaInfo.n2Size, D, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_BBND) {
        expected = {qfaInfo.totalBlockNum, CeilDivision(qfaInfo.blockSize, mxfp8BlockSize), qfaInfo.n2Size, D, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_BNBD) {
        expected = {qfaInfo.totalBlockNum, qfaInfo.n2Size, CeilDivision(qfaInfo.blockSize, mxfp8BlockSize), D, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_NZ) {
        expected = {
            qfaInfo.totalBlockNum, qfaInfo.n2Size, D / 16, CeilDivision(qfaInfo.blockSize, mxfp8BlockSize), 16, 2};
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            qfaInfo.opName, "layout_kv", QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(),
            "In MxFP8 scenario, the kv_layout for v_descale shape check is unsupported");
        return ge::GRAPH_FAILED;
    }

    if (expected.size() != dimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            qfaInfo.opName, V_DESCALE_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(),
            ("In MxFP8 scenario, the shape dim of v_descale must be " + std::to_string(expected.size()) +
             "D when layout_kv is " + QfaLayoutToSerialString(qfaInfo.kvLayout))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return CheckShapeEqual(*shape, expected, V_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckVDescaleShapeGqaFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode !=
        QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // GQA_FP8_FULLQUANT (per-head): 1D (N2)
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    std::vector<int64_t> expected = {qfaInfo.n2Size};
    return CheckShapeEqual(*shape, expected, V_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckVDescaleShape(const QfaTilingInfo &qfaInfo) const
{
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // HIF8 per-tensor: v_descale shape must be (1,), 1D
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        return CheckShapeEqual(*shape, {1}, V_DESCALE_NAME, qfaInfo.opName);
    }

    if (CheckVDescaleShapeMxFp8(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckVDescaleShapeGqaFp8(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

int64_t QuantChecker::CalcVDescaleTndDim0(const QfaTilingInfo &qfaInfo) const
{
    // TND 场景下 v_descale dim0 的精确计算依赖 cu_seqlens_kv 的 device 数据,
    // tiling(host)阶段无法安全读取, 该函数已不再被 CheckVDescaleShape 调用,
    // 保留仅供其他场景或后续 host 数据可用时使用。
    return (qfaInfo.kTSize + 64 - 1) / 64;
}

ge::graphStatus QuantChecker::CheckDescaleShape(const QfaTilingInfo &qfaInfo)
{
    if (CheckQDescaleShape(qfaInfo) != ge::GRAPH_SUCCESS || CheckKDescaleShape(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckVDescaleShape(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMultiPara(const QfaTilingInfo &qfaInfo)
{
    if (CheckDescaleShape(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDescaleDtype(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckDescaleDtype(const QfaTilingInfo &qfaInfo) const
{
    // 约束(一致性校验):
    //   MxFP8 场景下, q/k/v descale 的 tensor_type 仅支持 FLOAT8_E8M0
    //   GQA_FP8_FULLQUANT 场景下, q/k/v descale 的 tensor_type 仅支持 FLOAT32
    //   HIF8 场景下, q/k/v descale 的 tensor_type 仅支持 FLOAT32
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 &&
        qfaInfo.quantMode !=
            QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 &&
        qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }

    const auto it = DESCALE_DTYPE_TABLE.find(qfaInfo.quantMode);
    const ge::DataType expectedDtype = it->second.first;
    const std::string expectedDtypeStr = it->second.second;

    const auto CheckDescaleDtype = [&qfaInfo, this, expectedDtype, &expectedDtypeStr](
                                       const QfaRequiredParaInfo &slot,
                                       const std::string &paraName) -> ge::graphStatus {
        const gert::CompileTimeTensorDesc *desc = slot.desc;
        if (desc == nullptr) {
            return ge::GRAPH_SUCCESS; // 存在性校验负责
        }
        OP_CHECK_IF(desc->GetDataType() != expectedDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        qfaInfo.opName, paraName.c_str(), DataTypeToSerialString(desc->GetDataType()).c_str(),
                        ("The dtype of " + paraName + " must be " + expectedDtypeStr + " when quant_mode is " +
                         std::to_string(static_cast<uint32_t>(qfaInfo.quantMode)) + " (" +
                         QfaQuantModeToSerialString(qfaInfo.quantMode) + ")")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    if (CheckDescaleDtype(qfaInfo.opParamInfo.qDescale, Q_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDescaleDtype(qfaInfo.opParamInfo.kDescale, K_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDescaleDtype(qfaInfo.opParamInfo.vDescale, V_DESCALE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — Layout 匹配关系校验 (文档: layout匹配关系表)
// MxFP8: layout_q=TND, layout_kv∈{TND,PA_BNBD,PA_NZ}, layout_out=TND
// ============================================================================

namespace {
struct QfaLayoutConstraintConfig {
    std::vector<QfaLayout> supportedQLayouts;
    std::vector<QfaLayout> supportedKvLayouts;
    std::vector<QfaLayout> supportedOutLayouts;
    std::vector<QfaLayout> supportedQDescaleLayouts;
    bool requireLayoutConsistent;
};

const std::map<QfaQuantMode, QfaLayoutConstraintConfig> QFA_LAYOUT_CONSTRAINT_TABLE = {
    {QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32,
     {{QfaLayout::TND},
      {QfaLayout::TND, QfaLayout::PA_BBND, QfaLayout::PA_BNBD, QfaLayout::PA_NZ},
      {QfaLayout::TND},
      {QfaLayout::TND, QfaLayout::N2TGD},
      false}},
    {QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32,
     {{QfaLayout::TND}, {QfaLayout::PA_BNBD}, {QfaLayout::TND}, {QfaLayout::NT}, false}},
    {QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32,
     {{QfaLayout::TND, QfaLayout::BSND, QfaLayout::BNSD},
      {QfaLayout::TND, QfaLayout::BSND, QfaLayout::BNSD},
      {QfaLayout::TND, QfaLayout::BSND, QfaLayout::BNSD},
      {QfaLayout::BSND},
      true}},
};
} // namespace

ge::graphStatus QuantChecker::CheckLayoutConstraint(const QfaTilingInfo &qfaInfo) const
{
    auto it = QFA_LAYOUT_CONSTRAINT_TABLE.find(qfaInfo.quantMode);
    OP_CHECK_IF(it == QFA_LAYOUT_CONSTRAINT_TABLE.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "quant_mode",
                                                      std::to_string(static_cast<uint32_t>(qfaInfo.quantMode)).c_str(),
                                                      "The value of quant_mode must be 1 or 6 or 0"),
                return ge::GRAPH_FAILED);

    const auto &config = it->second;
    const std::string qLayoutStr = QfaLayoutToSerialString(qfaInfo.qLayout);
    const std::string quantModeStr = std::to_string(static_cast<uint32_t>(qfaInfo.quantMode)) + " (" +
                                     QfaQuantModeToSerialString(qfaInfo.quantMode) + ")";

    OP_CHECK_IF(std::find(config.supportedQLayouts.begin(), config.supportedQLayouts.end(), qfaInfo.qLayout) ==
                    config.supportedQLayouts.end(),
                OP_LOGE(qfaInfo.opName, "When quant_mode is %s, layout_q must be in supported list, but got %s",
                        quantModeStr.c_str(), qLayoutStr.c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(config.supportedKvLayouts.begin(), config.supportedKvLayouts.end(), qfaInfo.kvLayout) ==
                    config.supportedKvLayouts.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    qfaInfo.opName, "layout_kv", QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(),
                    ("When quant_mode is " + quantModeStr + " and layout_q is " + qLayoutStr +
                     ", layout_kv must be in {TND, PA_BBND, PA_BNBD, PA_NZ}")
                        .c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        std::find(config.supportedOutLayouts.begin(), config.supportedOutLayouts.end(), qfaInfo.outLayout) ==
            config.supportedOutLayouts.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            qfaInfo.opName, "layout_out", QfaLayoutToSerialString(qfaInfo.outLayout).c_str(),
            ("When quant_mode is " + quantModeStr + " and layout_q is " + qLayoutStr + ", layout_out must be TND")
                .c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(config.supportedQDescaleLayouts.begin(), config.supportedQDescaleLayouts.end(),
                          qfaInfo.layoutQDescale) == config.supportedQDescaleLayouts.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "layout_q_descale",
                                                      QfaLayoutToSerialString(qfaInfo.layoutQDescale).c_str(),
                                                      ("When quant_mode is " + quantModeStr + " and layout_q is " +
                                                       qLayoutStr + ", layout_q_descale must be in {TND, N2TGD}")
                                                          .c_str()),
                return ge::GRAPH_FAILED);

    if (config.requireLayoutConsistent) {
        OP_CHECK_IF(qfaInfo.qLayout != qfaInfo.kvLayout || qfaInfo.qLayout != qfaInfo.outLayout,
                    OP_LOGE(qfaInfo.opName,
                            "When quant_mode is %s, layout_q, layout_kv and layout_out must be the same, "
                            "but got layout_q=%s, layout_kv=%s, layout_out=%s",
                            quantModeStr.c_str(), qLayoutStr.c_str(), QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(),
                            QfaLayoutToSerialString(qfaInfo.outLayout).c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — q/k/v/attn_out shape 校验 (文档: q/k/v/attn_out shape匹配关系表)
// ============================================================================

void QuantChecker::SetQfaShapeCompare(const QfaTilingInfo &qfaInfo)
{
    queryShapeCmp_ = std::make_shared<QfaTilingShapeCompare>(qfaInfo.opParamInfo.query.shape->GetStorageShape(),
                                                             qfaInfo.qLayout, QUERY_NAME, qfaInfo.opName);
    keyShapeCmp_ = std::make_shared<QfaTilingShapeCompare>(qfaInfo.opParamInfo.key.shape->GetStorageShape(),
                                                           qfaInfo.kvLayout, KEY_NAME, qfaInfo.opName);
    valueShapeCmp_ = std::make_shared<QfaTilingShapeCompare>(qfaInfo.opParamInfo.value.shape->GetStorageShape(),
                                                             qfaInfo.kvLayout, VALUE_NAME, qfaInfo.opName);
    attnOutShapeCmp_ = std::make_shared<QfaTilingShapeCompare>(qfaInfo.opParamInfo.attnOut.shape->GetStorageShape(),
                                                               qfaInfo.outLayout, ATTN_OUT_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckQueryShape(const QfaTilingInfo &qfaInfo) const
{
    // q: TND -> (Q_T, Q_N, D)  BSND -> (B, S, Q_N, D)  BNSD -> (B, Q_N, S, D)
    QfaTilingShapeCompareParam shapeParams;
    shapeParams.N = qfaInfo.n1Size;
    shapeParams.D = qfaInfo.qkHeadDim;
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32 &&
        (qfaInfo.qLayout == QfaLayout::BSND || qfaInfo.qLayout == QfaLayout::BNSD)) {
        shapeParams.B = qfaInfo.bSize;
        shapeParams.S = qfaInfo.s1Size;
    } else {
        shapeParams.T = qfaInfo.qTSize;
    }
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus QuantChecker::CheckKVShape(const QfaTilingInfo &qfaInfo) const
{
    // k/v: TND -> (KV_T, KV_N, D)；PA_BBND -> (Bn, Bs, KV_N, D)；
    //      PA_BNBD -> (Bn, KV_N, Bs, D)；PA_NZ -> (Bn, KV_N, D/32, Bs, 32)
    QfaTilingShapeCompareParam shapeParams;
    shapeParams.N = qfaInfo.n2Size;
    shapeParams.D = qfaInfo.qkHeadDim;

    if (qfaInfo.kvLayout == QfaLayout::TND) {
        shapeParams.T = qfaInfo.kTSize;
    } else if (qfaInfo.kvLayout == QfaLayout::PA_BBND || qfaInfo.kvLayout == QfaLayout::PA_BNBD ||
               qfaInfo.kvLayout == QfaLayout::PA_NZ) {
        shapeParams.Bn = qfaInfo.totalBlockNum;
        shapeParams.Bs = qfaInfo.blockSize;
    } else if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32 &&
               (qfaInfo.kvLayout == QfaLayout::BSND || qfaInfo.kvLayout == QfaLayout::BNSD)) {
        shapeParams.B = qfaInfo.bSize;
        shapeParams.S = qfaInfo.s2Size;
    }
    // PA_NZ k/v: D0=32, shape 为 (Bn, KV_N, D/32, Bs, 32)
    if (qfaInfo.kvLayout == QfaLayout::PA_NZ) {
        shapeParams.D0 = 32;
    }

    if (keyShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // v 的 D 用 qkHeadDim
    shapeParams.D = qfaInfo.qkHeadDim;
    if (valueShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckAttnOutShape(const QfaTilingInfo &qfaInfo) const
{
    // attn_out: TND -> (Q_T, Q_N, D)  BSND -> (B, S, Q_N, D)  BNSD -> (B, Q_N, S, D)
    // D 取 vHeadDim（反量化后输出 dtype 为 BF16）
    QfaTilingShapeCompareParam shapeParams;
    shapeParams.N = qfaInfo.n1Size;
    shapeParams.D = qfaInfo.vHeadDim;
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32 &&
        (qfaInfo.outLayout == QfaLayout::BSND || qfaInfo.outLayout == QfaLayout::BNSD)) {
        shapeParams.B = qfaInfo.bSize;
        shapeParams.S = qfaInfo.s1Size;
    } else {
        shapeParams.T = qfaInfo.qTSize;
    }
    return attnOutShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus QuantChecker::CheckShapeMatch(const QfaTilingInfo &qfaInfo)
{
    SetQfaShapeCompare(qfaInfo);
    if (CheckQueryShape(qfaInfo) != ge::GRAPH_SUCCESS || CheckKVShape(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckAttnOutShape(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckFeature(const QfaTilingInfo &qfaInfo)
{
    // 文档"特性交叉校验"列(rowspan=5)包含:
    //   1. q/k/v dtype 与 quant_mode 精确匹配
    //   2. q/out ShapeDim 与 quant_mode 精确匹配
    //   3. 非连续 Tensor 支持
    //   4. Layout 校验(layout 匹配关系表, 含 layout_q_descale)
    //   5. q/k/v/attn_out shape 校验(shape 匹配关系表)
    if (CheckQkvDtype(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQkvShapeDim(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckLayoutConstraint(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        constexpr int64_t HIF8_HEAD_DIM = 128;
        OP_CHECK_IF(
            qfaInfo.qkHeadDim != HIF8_HEAD_DIM,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "head_dim", std::to_string(qfaInfo.qkHeadDim).c_str(),
                                                  "HIF8 only supports head_dim = 128"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(qfaInfo.vHeadDim != HIF8_HEAD_DIM,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "head_dim_v",
                                                          std::to_string(qfaInfo.vHeadDim).c_str(),
                                                          "HIF8 only supports head_dim_v = 128"),
                    return ge::GRAPH_FAILED);
    }
    if (CheckShapeMatch(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputAxisFullquant(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckN1SizeFullquant(const QfaTilingInfo &qfaInfo) const
{
    OP_CHECK_IF(
        (qfaInfo.n1Size > N1_LIMIT || qfaInfo.n1Size < 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "num_heads", std::to_string(qfaInfo.n1Size).c_str(),
                                              "The value of num_heads must be within the range [1, 256]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckN2SizeFullquant(const QfaTilingInfo &qfaInfo) const
{
    OP_CHECK_IF((qfaInfo.n2Size > N2_LIMIT || qfaInfo.n2Size < 1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    qfaInfo.opName, "num_key_value_heads", std::to_string(qfaInfo.n2Size).c_str(),
                    "The value of num_key_value_heads must be within the range [1, 256]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckGSizeFullquant(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.gSize < 1 || qfaInfo.gSize > G_LIMIT) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "axis G", std::to_string(qfaInfo.gSize).c_str(),
                                              "The value of axis G must be within the range [1, 64]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckInputAxisFullquant(const QfaTilingInfo &qfaInfo) const
{
    if (CheckN1SizeFullquant(qfaInfo) != ge::GRAPH_SUCCESS || CheckN2SizeFullquant(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckGSizeFullquant(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — q/k/v dtype 与 quant_mode 精确匹配校验 (跨参数组特性交叉校验)
//   MXFP8/FP8: q/k/v dtype 必须为 FLOAT8_E4M3FN
//   HIF8:      q/k/v dtype 必须为 HIFLOAT8
// ============================================================================

ge::graphStatus QuantChecker::CheckQkvDtype(const QfaTilingInfo &qfaInfo) const
{
    ge::DataType expectedDtype;
    std::string expectedDtypeStr;
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        expectedDtype = ge::DT_HIFLOAT8;
        expectedDtypeStr = "HIFLOAT8";
    } else {
        expectedDtype = ge::DT_FLOAT8_E4M3FN;
        expectedDtypeStr = "FLOAT8_E4M3FN";
    }

    const auto checkDtype = [&](const gert::CompileTimeTensorDesc *desc, const std::string &name) -> ge::graphStatus {
        if (desc == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        OP_CHECK_IF(desc->GetDataType() != expectedDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        qfaInfo.opName, name.c_str(), DataTypeToSerialString(desc->GetDataType()).c_str(),
                        ("The dtype of " + name + " must be " + expectedDtypeStr + " when quant_mode is " +
                         std::to_string(static_cast<uint32_t>(qfaInfo.quantMode)) + " (" +
                         QfaQuantModeToSerialString(qfaInfo.quantMode) + ")")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    if (checkDtype(qfaInfo.opParamInfo.query.desc, QUERY_NAME) != ge::GRAPH_SUCCESS ||
        checkDtype(qfaInfo.opParamInfo.key.desc, KEY_NAME) != ge::GRAPH_SUCCESS ||
        checkDtype(qfaInfo.opParamInfo.value.desc, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — q/out ShapeDim 与 quant_mode 精确匹配校验 (跨参数组特性交叉校验)
//   MXFP8/FP8: q/attn_out shape dim 仅支持 3D
//   HIF8:      q/attn_out shape dim 支持 3D/4D
// ============================================================================

ge::graphStatus QuantChecker::CheckQkvShapeDim(const QfaTilingInfo &qfaInfo) const
{
    std::vector<uint32_t> supportedDims;
    std::string dimStr;
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        supportedDims = {DIM_NUM_3, DIM_NUM_4};
        dimStr = "3D/4D";
    } else {
        supportedDims = {DIM_NUM_3};
        dimStr = "3D";
    }

    const auto checkDim = [&](const gert::StorageShape *shape, const std::string &name) -> ge::graphStatus {
        if (shape == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        uint32_t dimNum = shape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(std::find(supportedDims.begin(), supportedDims.end(), dimNum) == supportedDims.end(),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        qfaInfo.opName, name.c_str(), (std::to_string(dimNum) + "D").c_str(),
                        ("The shape dim of " + name + " must be " + dimStr + " when quant_mode is " +
                         std::to_string(static_cast<uint32_t>(qfaInfo.quantMode)) + " (" +
                         QfaQuantModeToSerialString(qfaInfo.quantMode) + ")")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    if (checkDim(qfaInfo.opParamInfo.query.shape, QUERY_NAME) != ge::GRAPH_SUCCESS ||
        checkDim(qfaInfo.opParamInfo.attnOut.shape, ATTN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
