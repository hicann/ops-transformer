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
 * \brief Checker for quant_mode, q_descale, k_descale, v_descale, p_scale (文档约束: 全量化参数组)
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

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus QuantChecker::CheckSingleParaQuantMode(const QfaTilingInfo &qfaInfo)
{
    // 文档约束: data_type 支持 INT32；当前仅支持 quant_mode = 1
    // quantMode 为属性，QfaTilingInfo 中存储为 QfaQuantMode 枚举
    const std::vector<uint32_t> supportedQuantModes = {1};
    uint32_t quantModeVal = static_cast<uint32_t>(qfaInfo.quantMode);
    OP_CHECK_IF(
        std::find(supportedQuantModes.begin(), supportedQuantModes.end(), quantModeVal) == supportedQuantModes.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "quant_mode",
                                              std::to_string(quantModeVal).c_str(),
                                              "The value of quant_mode must be 1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckSingleParaQDescale(const QfaTilingInfo &qfaInfo)
{
    // 文档约束: tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 支持 4、5
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.qDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS; // 存在性校验负责
    }

    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, Q_DESCALE_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
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

ge::graphStatus QuantChecker::CheckSingleParaKDescale(const QfaTilingInfo &qfaInfo)
{
    // 文档约束: tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 支持 4、5、6
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.kDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, K_DESCALE_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
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

ge::graphStatus QuantChecker::CheckSingleParaVDescale(const QfaTilingInfo &qfaInfo)
{
    // 文档约束: tensor_type 支持 FLOAT8_E8M0、FLOAT32；shape dim 支持 4、5、6
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.vDescale.desc;
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    if (desc == nullptr || shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), desc->GetDataType()) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, V_DESCALE_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT8_E8M0/FLOAT32"),
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

ge::graphStatus QuantChecker::CheckSingleParaPScale(const QfaTilingInfo &qfaInfo)
{
    // 文档约束: tensor_type 仅支持 FLOAT32；shape 仅支持 (1,)
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
    OP_CHECK_IF(dim0 != 1, OP_LOGE(qfaInfo.opName, "p_scale shape must be (1,), but got dim0=%ld", dim0),
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

ge::graphStatus QuantChecker::CheckParaExistence(const QfaTilingInfo &qfaInfo)
{
    // quant_mode: 必选属性
    OP_CHECK_IF(qfaInfo.opParamInfo.quantMode == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, "quant_mode"), return ge::GRAPH_FAILED);

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
            OP_LOGE(opName, "%s shape dim[%zu] is %ld, expected %ld.", paraName.c_str(), i, actDim, expected[i]);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckQDescaleShape(const QfaTilingInfo &qfaInfo) const
{
    // 文档(descale_shape匹配关系表): MxFP8, layout_q=TND
    //   4D: (Q_T, Q_N, D/64, 2)              prefill场景，layout_q_descale=TND
    //   5D: (KV_N, Q_T, G, D/64, 2)          decode场景，layout_q_descale=N2TGD
    //   其中 G = Q_N / KV_N
    const gert::StorageShape *shape = qfaInfo.opParamInfo.qDescale.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t D = qfaInfo.qkHeadDim;
    int64_t DPerGroup = D / 64; // MxFP8 block size = 64
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();

    std::vector<int64_t> expected;
    if (dimNum == DIM_NUM_4) {
        expected = {qfaInfo.qTSize, qfaInfo.n1Size, DPerGroup, 2};
    } else if (dimNum == DIM_NUM_5) {
        expected = {qfaInfo.n2Size, qfaInfo.qTSize, qfaInfo.gSize, DPerGroup, 2};
    } else {
        OP_LOGE(qfaInfo.opName, "q_descale shape dim %u is unsupported, expected 4D or 5D.", dimNum);
        return ge::GRAPH_FAILED;
    }

    // 约束: MxFP8场景下 q_descale 两种 shape 对应不同使用场景
    //   4D 为 prefill 场景，layout_q_descale 必须为 TND；5D 为 decode 场景，layout_q_descale 必须为 N2TGD
    if (qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
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
    }

    return CheckShapeEqual(*shape, expected, Q_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckKDescaleShape(const QfaTilingInfo &qfaInfo) const
{
    // 文档(k_descale/v_descale shape匹配关系表): MxFP8
    //   TND:      (KV_T, KV_N, D/64, 2)              - 4D
    //   PA_BNBD:  (Bn, KV_N, Bs, D/64, 2)            - 5D
    //   PA_NZ:    (Bn, KV_N, Bs/16, D/64, 16, 2)     - 6D
    const gert::StorageShape *shape = qfaInfo.opParamInfo.kDescale.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t D = qfaInfo.qkHeadDim;
    int64_t DPerGroup = D / 64;
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();

    std::vector<int64_t> expected;
    if (qfaInfo.kvLayout == QfaLayout::TND) {
        expected = {qfaInfo.kTSize, qfaInfo.n2Size, DPerGroup, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_BNBD) {
        expected = {qfaInfo.totalBlockNum, qfaInfo.n2Size, qfaInfo.blockSize, DPerGroup, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_NZ) {
        expected = {qfaInfo.totalBlockNum, qfaInfo.n2Size, qfaInfo.blockSize / 16, DPerGroup, 16, 2};
    } else {
        OP_LOGE(qfaInfo.opName, "k_descale shape check: kv_layout %s is unsupported.",
                QfaLayoutToSerialString(qfaInfo.kvLayout).c_str());
        return ge::GRAPH_FAILED;
    }

    if (expected.size() != dimNum) {
        OP_LOGE(qfaInfo.opName, "k_descale shape dim %u does not match layout %s, expected %zuD.", dimNum,
                QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(), expected.size());
        return ge::GRAPH_FAILED;
    }
    return CheckShapeEqual(*shape, expected, K_DESCALE_NAME, qfaInfo.opName);
}

ge::graphStatus QuantChecker::CheckVDescaleShape(const QfaTilingInfo &qfaInfo) const
{
    // 文档(k_descale/v_descale shape匹配关系表): MxFP8
    //   TND:      (KV_T/64, KV_N, D, 2)              - 4D
    //   PA_BNBD:  (Bn, KV_N, Bs/64, D, 2)            - 5D
    //   PA_NZ:    (Bn, KV_N, D/16, Bs/64, 16, 2)     - 6D
    const gert::StorageShape *shape = qfaInfo.opParamInfo.vDescale.shape;
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t D = qfaInfo.vHeadDim; // v_descale 的 D 用 vHeadDim
    uint32_t dimNum = shape->GetStorageShape().GetDimNum();

    std::vector<int64_t> expected;
    if (qfaInfo.kvLayout == QfaLayout::TND) {
        // TND(非PA)场景: dim0 = Σ ceil(curLen, 64), 依赖 cu_seqlens_kv 的实际数值。
        // tiling 阶段 cu_seqlens_kv 为 device tensor, host 侧无法安全读取,
        // 因此 dim0 不做数值校验(用 -1 占位), 仅校验维度数为 4D 及 dim[1..3]。
        expected = {-1, qfaInfo.n2Size, D, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_BNBD) {
        expected = {qfaInfo.totalBlockNum, qfaInfo.n2Size, qfaInfo.blockSize / 64, D, 2};
    } else if (qfaInfo.kvLayout == QfaLayout::PA_NZ) {
        expected = {qfaInfo.totalBlockNum, qfaInfo.n2Size, D / 16, qfaInfo.blockSize / 64, 16, 2};
    } else {
        OP_LOGE(qfaInfo.opName, "v_descale shape check: kv_layout %s is unsupported.",
                QfaLayoutToSerialString(qfaInfo.kvLayout).c_str());
        return ge::GRAPH_FAILED;
    }

    if (expected.size() != dimNum) {
        OP_LOGE(qfaInfo.opName, "v_descale shape dim %u does not match layout %s, expected %zuD.", dimNum,
                QfaLayoutToSerialString(qfaInfo.kvLayout).c_str(), expected.size());
        return ge::GRAPH_FAILED;
    }
    return CheckShapeEqual(*shape, expected, V_DESCALE_NAME, qfaInfo.opName);
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
    // 文档约束(一致性校验): MxFP8 场景下, q/k/v descale 的 tensor_type 仅支持 FLOAT8_E8M0
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }

    const auto CheckDescaleDtype = [&qfaInfo, this](const QfaRequiredParaInfo &slot,
                                                    const std::string &paraName) -> ge::graphStatus {
        const gert::CompileTimeTensorDesc *desc = slot.desc;
        if (desc == nullptr) {
            return ge::GRAPH_SUCCESS; // 存在性校验负责
        }
        OP_CHECK_IF(desc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, paraName.c_str(),
                                              DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT8_E8M0"),
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
// Feature — 非连续 Tensor 支持校验 (文档"特性交叉校验"列)
// 规则: 仅 PA 场景(layout_kv ∈ {PA_BNBD, PA_NZ})时，k/v/k_descale/v_descale
//       仅支持 0 轴和 1 轴非连续，其余轴必须连续；非 PA 场景均不支持非连续。
// ============================================================================

ge::graphStatus QuantChecker::CheckNonContiguousSupport(const QfaTilingInfo &qfaInfo) const
{
    bool isPaLayout = (qfaInfo.kvLayout == QfaLayout::PA_BNBD || qfaInfo.kvLayout == QfaLayout::PA_NZ);
    if (!isPaLayout) {
        // 非 PA 场景: k/v/k_descale/v_descale 均不支持非连续
        int32_t dimIndex = 0;
        OP_CHECK_IF((CheckTensorContiguous(qfaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum(),
                                           qfaInfo.opParamInfo.key.shape->GetStorageShape(), qfaInfo.keyStrides,
                                           dimIndex) != ge::GRAPH_SUCCESS),
                    OP_LOGE(qfaInfo.opName,
                            "In non-PA scenarios, key must be contiguous, but dim %d is non-contiguous.", dimIndex),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((CheckTensorContiguous(qfaInfo.opParamInfo.value.shape->GetStorageShape().GetDimNum(),
                                           qfaInfo.opParamInfo.value.shape->GetStorageShape(), qfaInfo.valueStrides,
                                           dimIndex) != ge::GRAPH_SUCCESS),
                    OP_LOGE(qfaInfo.opName,
                            "In non-PA scenarios, value must be contiguous, but dim %d is non-contiguous.", dimIndex),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (CheckTensorContiguous(qfaInfo.opParamInfo.kDescale.shape->GetStorageShape().GetDimNum(),
                                   qfaInfo.opParamInfo.kDescale.shape->GetStorageShape(), qfaInfo.kDescaleStrides,
                                   dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE(qfaInfo.opName, "In non-PA scenarios, k_descale must be contiguous, but dim %d is non-contiguous.",
                    dimIndex),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (CheckTensorContiguous(qfaInfo.opParamInfo.vDescale.shape->GetStorageShape().GetDimNum(),
                                   qfaInfo.opParamInfo.vDescale.shape->GetStorageShape(), qfaInfo.vDescaleStrides,
                                   dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE(qfaInfo.opName, "In non-PA scenarios, v_descale must be contiguous, but dim %d is non-contiguous.",
                    dimIndex),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // PA 场景(layout_kv ∈ {PA_BNBD, PA_NZ}): k/v/k_descale/v_descale 仅支持 0/1 轴非连续
    int32_t dimIndex = 0;
    OP_CHECK_IF(
        ((ge::GRAPH_SUCCESS != CheckTensorContiguous(qfaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum(),
                                                     qfaInfo.opParamInfo.key.shape->GetStorageShape(),
                                                     qfaInfo.keyStrides, dimIndex)) &&
         (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE(qfaInfo.opName,
                "In PA BnNBsD/NZ scenarios, key only supports non-contiguous tensors in dimensions 0 or 1, "
                "but the first non-contiguous dimension is index %d.",
                dimIndex),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((ge::GRAPH_SUCCESS != CheckTensorContiguous(qfaInfo.opParamInfo.value.shape->GetStorageShape().GetDimNum(),
                                                     qfaInfo.opParamInfo.value.shape->GetStorageShape(),
                                                     qfaInfo.valueStrides, dimIndex)) &&
         (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE(qfaInfo.opName,
                "In PA BnNBsD/NZ scenarios, value only supports non-contiguous tensors in dimensions 0 or 1, "
                "but the first non-contiguous dimension is index %d.",
                dimIndex),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((ge::GRAPH_SUCCESS != CheckTensorContiguous(qfaInfo.opParamInfo.kDescale.shape->GetStorageShape().GetDimNum(),
                                                     qfaInfo.opParamInfo.kDescale.shape->GetStorageShape(),
                                                     qfaInfo.kDescaleStrides, dimIndex)) &&
         (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE(qfaInfo.opName,
                "In PA BnNBsD/NZ scenarios, k_descale only supports non-contiguous tensors in dimensions 0 or "
                "1, but the first non-contiguous dimension is index %d.",
                dimIndex),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((ge::GRAPH_SUCCESS != CheckTensorContiguous(qfaInfo.opParamInfo.vDescale.shape->GetStorageShape().GetDimNum(),
                                                     qfaInfo.opParamInfo.vDescale.shape->GetStorageShape(),
                                                     qfaInfo.vDescaleStrides, dimIndex)) &&
         (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE(qfaInfo.opName,
                "In PA BnNBsD/NZ scenarios, v_descale only supports non-contiguous tensors in dimensions 0 or "
                "1, but the first non-contiguous dimension is index %d.",
                dimIndex),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — Layout 匹配关系校验 (文档: layout匹配关系表)
// MxFP8: layout_q=TND, layout_kv∈{TND,PA_BNBD,PA_NZ}, layout_out=TND
// ============================================================================

namespace {
struct QfaLayoutConstraintConfig {
    std::vector<QfaLayout> supportedKvLayouts;
    std::vector<QfaLayout> supportedOutLayouts;
    std::vector<QfaLayout> supportedQDescaleLayouts;
};

const std::map<QfaQuantMode, QfaLayoutConstraintConfig> QFA_LAYOUT_CONSTRAINT_TABLE = {
    {QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32,
     {{QfaLayout::TND, QfaLayout::PA_BNBD, QfaLayout::PA_NZ}, {QfaLayout::TND}, {QfaLayout::TND, QfaLayout::N2TGD}}},
};
} // namespace

ge::graphStatus QuantChecker::CheckLayoutConstraint(const QfaTilingInfo &qfaInfo) const
{
    auto it = QFA_LAYOUT_CONSTRAINT_TABLE.find(qfaInfo.quantMode);
    OP_CHECK_IF(it == QFA_LAYOUT_CONSTRAINT_TABLE.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "quant_mode",
                                                      std::to_string(static_cast<uint32_t>(qfaInfo.quantMode)).c_str(),
                                                      "quant_mode is not supported in layout constraint table"),
                return ge::GRAPH_FAILED);

    const auto &config = it->second;
    const std::string qLayoutStr = QfaLayoutToSerialString(qfaInfo.qLayout);

    OP_CHECK_IF(std::find(config.supportedKvLayouts.begin(), config.supportedKvLayouts.end(), qfaInfo.kvLayout) ==
                    config.supportedKvLayouts.end(),
                OP_LOGE(qfaInfo.opName,
                        "When quant_mode is MxFP8 and layout_q is %s, "
                        "layout_kv must be in {TND, PA_BNBD, PA_NZ}, but got %s",
                        qLayoutStr.c_str(), QfaLayoutToSerialString(qfaInfo.kvLayout).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        std::find(config.supportedOutLayouts.begin(), config.supportedOutLayouts.end(), qfaInfo.outLayout) ==
            config.supportedOutLayouts.end(),
        OP_LOGE(qfaInfo.opName, "When quant_mode is MxFP8 and layout_q is %s, layout_out must be TND, but got %s",
                qLayoutStr.c_str(), QfaLayoutToSerialString(qfaInfo.outLayout).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        std::find(config.supportedQDescaleLayouts.begin(), config.supportedQDescaleLayouts.end(),
                  qfaInfo.layoutQDescale) == config.supportedQDescaleLayouts.end(),
        OP_LOGE(qfaInfo.opName,
                "When quant_mode is MxFP8 and layout_q is %s, layout_q_descale must be in {TND, N2TGD}, but got %s",
                qLayoutStr.c_str(), QfaLayoutToSerialString(qfaInfo.layoutQDescale).c_str()),
        return ge::GRAPH_FAILED);

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
    // q: TND -> (Q_T, Q_N, D)
    QfaTilingShapeCompareParam shapeParams;
    shapeParams.T = qfaInfo.qTSize;
    shapeParams.N = qfaInfo.n1Size;
    shapeParams.D = qfaInfo.qkHeadDim;
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus QuantChecker::CheckKVShape(const QfaTilingInfo &qfaInfo) const
{
    // k/v: TND -> (KV_T, KV_N, D)；PA_BNBD -> (Bn, KV_N, Bs, D)；
    //      PA_NZ -> (Bn, KV_N, D/32, Bs, 32)
    QfaTilingShapeCompareParam shapeParams;
    shapeParams.N = qfaInfo.n2Size;
    shapeParams.D = qfaInfo.qkHeadDim;

    if (qfaInfo.kvLayout == QfaLayout::TND) {
        shapeParams.T = qfaInfo.kTSize;
    } else if (qfaInfo.kvLayout == QfaLayout::PA_BNBD || qfaInfo.kvLayout == QfaLayout::PA_NZ) {
        shapeParams.Bn = qfaInfo.totalBlockNum;
        shapeParams.Bs = qfaInfo.blockSize;
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
    // attn_out: TND -> (Q_T, Q_N, D)，D 取 vHeadDim（反量化后输出 dtype 为 BF16）
    QfaTilingShapeCompareParam shapeParams;
    shapeParams.T = qfaInfo.qTSize;
    shapeParams.N = qfaInfo.n1Size;
    shapeParams.D = qfaInfo.vHeadDim;
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
    // 文档"特性交叉校验"列(rowspan=5)包含三项:
    //   1. 非连续 Tensor 支持
    //   2. Layout 校验(layout 匹配关系表, 含 layout_q_descale)
    //   3. q/k/v/attn_out shape 校验(shape 匹配关系表)
    if (CheckNonContiguousSupport(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckLayoutConstraint(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeMatch(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
