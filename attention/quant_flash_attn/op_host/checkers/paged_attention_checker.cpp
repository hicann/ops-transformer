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
 * \file paged_attention_checker.cpp
 * \brief Checker for block_table ( Paged Attention参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../qfa_tiling_info.h"
#include "paged_attention_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;

bool PagedAttentionChecker::IsPageAttention(const QfaTilingInfo &qfaInfo) const
{
    // 仅通过 layout_kv 判断是否为 PA 场景 (PA_BBND/PA_BNBD/PA_NZ)
    return (qfaInfo.kvLayout == QfaLayout::PA_BBND || qfaInfo.kvLayout == QfaLayout::PA_BNBD ||
            qfaInfo.kvLayout == QfaLayout::PA_NZ);
}

ge::graphStatus PagedAttentionChecker::CheckSingleParaBlockTable(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验列):
    //   - tensor_type 仅支持 INT32
    //   - tensor_shape 为 (B, Bn)
    //   - 值只能为正整数 (属于值约束, tiling 阶段无法获取 tensor 数值, 不校验)
    // block_table 是否传入的校验属于特性交叉校验列, 此处为空时跳过 dtype/shape 校验
    const gert::Tensor *blockTableTensor = qfaInfo.opParamInfo.blockTable.tensor;
    if (blockTableTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *blockTableDesc = qfaInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, "TensorDesc of block_table"),
                return ge::GRAPH_FAILED);

    // dtype 校验
    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, BLOCK_TABLE_NAME.c_str(),
                                          DataTypeToSerialString(blockTableDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    // format 校验: 使用基类 CheckFormatSupport (基于 GetOriginFormat + FORMAT_SUPPORT_SET)
    if (ge::GRAPH_SUCCESS != CheckFormatSupport(blockTableDesc, BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    // shape dim 校验: 2D
    const gert::Shape &shape = blockTableTensor->GetStorageShape();
    size_t dimNum = shape.GetDimNum();
    OP_CHECK_IF(dimNum != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, BLOCK_TABLE_NAME.c_str(),
                                             (std::to_string(dimNum) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);

    // shape size 校验: (B, Bn)
    // B 由 qfaInfo.bSize 给出; Bn(每批次最大块数) 由 qfaInfo.maxBlockNumPerBatch 给出
    int64_t dim0 = static_cast<int64_t>(shape.GetDim(0));
    int64_t dim1 = static_cast<int64_t>(shape.GetDim(1));
    OP_CHECK_IF(
        dim0 != qfaInfo.bSize,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            qfaInfo.opName, BLOCK_TABLE_NAME.c_str(),
            ("[" + std::to_string(dim0) + ", " + std::to_string(dim1) + "]").c_str(),
            ("The value of dim0 of block_table shape must be " + std::to_string(qfaInfo.bSize) + " (B)").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dim1 != qfaInfo.maxBlockNumPerBatch,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            qfaInfo.opName, BLOCK_TABLE_NAME.c_str(),
            ("[" + std::to_string(dim0) + ", " + std::to_string(dim1) + "]").c_str(),
            ("The value of dim1 of block_table shape must be " + std::to_string(qfaInfo.maxBlockNumPerBatch) + " (Bn)")
                .c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验列): block_table 的 dtype/shape
    // 仅 PA 场景下校验 block_table, 通过 layout_kv 判断
    if (!IsPageAttention(qfaInfo)) {
        return ge::GRAPH_SUCCESS;
    }

    if (CheckSingleParaBlockTable(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockSizeMxFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode != QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // MxFP8 场景: blockSize 仅支持 64、128、256、512或1024
    OP_CHECK_IF(qfaInfo.blockSize != 64 && qfaInfo.blockSize != 128 && qfaInfo.blockSize != 256 &&
                    qfaInfo.blockSize != 512 && qfaInfo.blockSize != 1024,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    qfaInfo.opName, "block_size", std::to_string(qfaInfo.blockSize).c_str(),
                    "When quant_mode is MxFP8, block_size must be in [64, 128, 256, 512, 1024]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockSizeGqaFp8(const QfaTilingInfo &qfaInfo) const
{
    if (qfaInfo.quantMode !=
        QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        return ge::GRAPH_SUCCESS;
    }
    // GQA FP8 fullquant 场景: blockSize 固定 128
    OP_CHECK_IF(
        qfaInfo.blockSize != 128,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "block_size", std::to_string(qfaInfo.blockSize).c_str(),
                                              "When quant_mode is GQA_FP8_FULLQUANT, block_size must be 128"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeature(const QfaTilingInfo &qfaInfo)
{
    // 约束(特性交叉校验列):
    //   - PagedAttention 开启情况下, block_table 必须不为空
    //   - PagedAttention 开启情况下, 必须传入 seqused_kv
    //   - MxFP8 仅支持 Bs 为 64、128、256、512或1024
    //   - 非连续 Tensor 支持校验
    const gert::Tensor *blockTableTensor = qfaInfo.opParamInfo.blockTable.tensor;
    if (IsPageAttention(qfaInfo)) {
        // PA 场景: block_table 必须非空
        OP_CHECK_IF(blockTableTensor == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, BLOCK_TABLE_NAME.c_str(), "empty",
                        "When layout_kv is PA (PA_BBND/PA_BNBD/PA_NZ), block_table must not be empty"),
                    return ge::GRAPH_FAILED);

        // PA 场景: 必须传入 seqused_kv
        const gert::Tensor *sequsedKvTensor = qfaInfo.opParamInfo.sequsedKv.tensor;
        OP_CHECK_IF(
            sequsedKvTensor == nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, SEQUSED_KV_NAME.c_str(), "empty",
                                                  "When PagedAttention is enabled, seqused_kv must be provided"),
            return ge::GRAPH_FAILED);

        // 场景: blockSize 校验
        if (CheckBlockSizeMxFp8(qfaInfo) != ge::GRAPH_SUCCESS || CheckBlockSizeGqaFp8(qfaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // 非 PA 场景: block_table 不应传入
        OP_CHECK_IF(blockTableTensor != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, BLOCK_TABLE_NAME.c_str(), "provided",
                        "When layout_kv is not PA (PA_BBND/PA_BNBD/PA_NZ), block_table must not be provided"),
                    return ge::GRAPH_FAILED);
    }
    // 非连续 Tensor 支持校验
    if (CheckNonContiguousSupport(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — 非连续 Tensor 支持校验 (文档"特性交叉校验"列)
// 规则: 仅 PA 场景(layout_kv ∈ {PA_BNBD, PA_NZ})时，k/v/k_descale/v_descale
//       仅支持 0 轴和 1 轴非连续，其余轴必须连续；非 PA 场景均不支持非连续。
// ============================================================================

ge::graphStatus PagedAttentionChecker::CheckNonContiguousSupport(const QfaTilingInfo &qfaInfo) const
{
    if (!IsPageAttention(qfaInfo)) {
        // 非 PA 场景: k/v/k_descale/v_descale 均不支持非连续
        int32_t dimIndex = 0;
        OP_CHECK_IF((CheckTensorContiguous(qfaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum(),
                                           qfaInfo.opParamInfo.key.shape->GetStorageShape(), qfaInfo.keyStrides,
                                           dimIndex) != ge::GRAPH_SUCCESS),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(qfaInfo.opName, "key",
                                                             ("In non-PA scenarios, key must be contiguous, but dim " +
                                                              std::to_string(dimIndex) + " is non-contiguous")
                                                                 .c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (CheckTensorContiguous(qfaInfo.opParamInfo.value.shape->GetStorageShape().GetDimNum(),
                                   qfaInfo.opParamInfo.value.shape->GetStorageShape(), qfaInfo.valueStrides,
                                   dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(qfaInfo.opName, "value",
                                                     ("In non-PA scenarios, value must be contiguous, but dim " +
                                                      std::to_string(dimIndex) + " is non-contiguous")
                                                         .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (CheckTensorContiguous(qfaInfo.opParamInfo.kDescale.shape->GetStorageShape().GetDimNum(),
                                   qfaInfo.opParamInfo.kDescale.shape->GetStorageShape(), qfaInfo.kDescaleStrides,
                                   dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(qfaInfo.opName, "k_descale",
                                                     ("In non-PA scenarios, k_descale must be contiguous, but dim " +
                                                      std::to_string(dimIndex) + " is non-contiguous")
                                                         .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (CheckTensorContiguous(qfaInfo.opParamInfo.vDescale.shape->GetStorageShape().GetDimNum(),
                                   qfaInfo.opParamInfo.vDescale.shape->GetStorageShape(), qfaInfo.vDescaleStrides,
                                   dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(qfaInfo.opName, "v_descale",
                                                     ("In non-PA scenarios, v_descale must be contiguous, but dim " +
                                                      std::to_string(dimIndex) + " is non-contiguous")
                                                         .c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // PA 场景
    // (layout_kv ∈ {PA_BBND}): k/v/k_descale/v_descale 仅支持 0 轴非连续
    // (layout_kv ∈ {PA_BNBD, PA_NZ}): k/v/k_descale/v_descale 仅支持 0/1 轴非连续
    int32_t dimIndex = 0;
    if (qfaInfo.kvLayout == QfaLayout::PA_BBND) {
        OP_CHECK_IF(
            ((ge::GRAPH_SUCCESS != CheckTensorContiguous(qfaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum(),
                                                         qfaInfo.opParamInfo.key.shape->GetStorageShape(),
                                                         qfaInfo.keyStrides, dimIndex)) &&
             (dimIndex != 0)),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                qfaInfo.opName, "key",
                ("In PA BnBsND scenario, only 0th axis of key can be non-contiguous, the " + std::to_string(dimIndex) +
                 "th axis of key must be contiguous")
                    .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ((ge::GRAPH_SUCCESS != CheckTensorContiguous(qfaInfo.opParamInfo.value.shape->GetStorageShape().GetDimNum(),
                                                         qfaInfo.opParamInfo.value.shape->GetStorageShape(),
                                                         qfaInfo.valueStrides, dimIndex)) &&
             (dimIndex != 0)),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                qfaInfo.opName, "value",
                ("In PA BnBsND scenario, only 0th axis of value can be non-contiguous, the " +
                 std::to_string(dimIndex) + "th axis of value must be contiguous")
                    .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(((ge::GRAPH_SUCCESS !=
                      CheckTensorContiguous(qfaInfo.opParamInfo.kDescale.shape->GetStorageShape().GetDimNum(),
                                            qfaInfo.opParamInfo.kDescale.shape->GetStorageShape(),
                                            qfaInfo.kDescaleStrides, dimIndex)) &&
                     (dimIndex != 0)),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        qfaInfo.opName, "k_descale",
                        ("In PA BnBsND scenario, only 0th axis of k_descale can be non-contiguous, the " +
                         std::to_string(dimIndex) + "th axis of k_descale must be contiguous")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(((ge::GRAPH_SUCCESS !=
                      CheckTensorContiguous(qfaInfo.opParamInfo.vDescale.shape->GetStorageShape().GetDimNum(),
                                            qfaInfo.opParamInfo.vDescale.shape->GetStorageShape(),
                                            qfaInfo.vDescaleStrides, dimIndex)) &&
                     (dimIndex != 0)),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        qfaInfo.opName, "v_descale",
                        ("In PA BnBsND scenario, only 0th axis of v_descale can be non-contiguous, the " +
                         std::to_string(dimIndex) + "th axis of v_descale must be contiguous")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            ((ge::GRAPH_SUCCESS != CheckTensorContiguous(qfaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum(),
                                                         qfaInfo.opParamInfo.key.shape->GetStorageShape(),
                                                         qfaInfo.keyStrides, dimIndex)) &&
             (dimIndex != 0 && dimIndex != 1)),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                qfaInfo.opName, "key",
                ("In PA BnNBsD/NZ scenario, only 0th and 1st axis of key can be non-contiguous, the " +
                 std::to_string(dimIndex) + "th axis of key must be contiguous")
                    .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ((ge::GRAPH_SUCCESS != CheckTensorContiguous(qfaInfo.opParamInfo.value.shape->GetStorageShape().GetDimNum(),
                                                         qfaInfo.opParamInfo.value.shape->GetStorageShape(),
                                                         qfaInfo.valueStrides, dimIndex)) &&
             (dimIndex != 0 && dimIndex != 1)),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                qfaInfo.opName, "value",
                ("In PA BnNBsD/NZ scenario, only 0th and 1st axis of value can be non-contiguous, the " +
                 std::to_string(dimIndex) + "th axis of value must be contiguous")
                    .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(((ge::GRAPH_SUCCESS !=
                      CheckTensorContiguous(qfaInfo.opParamInfo.kDescale.shape->GetStorageShape().GetDimNum(),
                                            qfaInfo.opParamInfo.kDescale.shape->GetStorageShape(),
                                            qfaInfo.kDescaleStrides, dimIndex)) &&
                     (dimIndex != 0 && dimIndex != 1)),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        qfaInfo.opName, "k_descale",
                        ("In PA BnNBsD/NZ scenario, only 0th and 1st axis of k_descale can be non-contiguous, the " +
                         std::to_string(dimIndex) + "th axis of k_descale must be contiguous")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(((ge::GRAPH_SUCCESS !=
                      CheckTensorContiguous(qfaInfo.opParamInfo.vDescale.shape->GetStorageShape().GetDimNum(),
                                            qfaInfo.opParamInfo.vDescale.shape->GetStorageShape(),
                                            qfaInfo.vDescaleStrides, dimIndex)) &&
                     (dimIndex != 0 && dimIndex != 1)),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        qfaInfo.opName, "v_descale",
                        ("In PA BnNBsD/NZ scenario, only 0th and 1st axis of v_descale can be non-contiguous, the " +
                         std::to_string(dimIndex) + "th axis of v_descale must be contiguous")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const QfaTilingInfo &qfaInfo)
{
    // 约束(一致性校验列): 无
    // block_table shape (B, Bn) 的校验已在 CheckSingleParaBlockTable 中完成
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
