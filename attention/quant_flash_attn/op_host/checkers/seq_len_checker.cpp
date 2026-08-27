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
 * \file seq_len_checker.cpp
 * \brief Checker for cu_seqlens_q/kv, seqused_q/kv, max_seqlen_q/kv ( SeqLens参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../qfa_tiling_info.h"
#include "seq_len_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;

ge::graphStatus SeqLenChecker::CheckSingleParaSequsedQ(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验): dtype=INT32, format=ND, shape=(B,)
    // 值约束(非负、<=Q_S)需读 tensor 数值，tiling 阶段无法获取，不校验（见文档 line 214）
    const gert::Tensor *tensor = qfaInfo.opParamInfo.sequsedQ.tensor;
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.sequsedQ.desc;
    if (tensor == nullptr || desc == nullptr) {
        return ge::GRAPH_SUCCESS; // 可选参数，存在性由 CheckParaExistence 负责
    }
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, SEQUSED_Q_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, SEQUSED_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, SEQUSED_Q_NAME.c_str(),
                                             (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != qfaInfo.bSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    qfaInfo.opName, SEQUSED_Q_NAME.c_str(), ("[" + std::to_string(shape.GetDim(0)) + "]").c_str(),
                    ("The value of dim0 of " + SEQUSED_Q_NAME + " shape must be equal to B(" +
                     std::to_string(qfaInfo.bSize) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaSequsedKv(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验): dtype=INT32, format=ND, shape=(B,)
    // 值约束(非负、<=KV_S)需读 tensor 数值，tiling 阶段无法获取，不校验
    const gert::Tensor *tensor = qfaInfo.opParamInfo.sequsedKv.tensor;
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.sequsedKv.desc;
    if (tensor == nullptr || desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, SEQUSED_KV_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, SEQUSED_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, SEQUSED_KV_NAME.c_str(),
                                             (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != qfaInfo.bSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    qfaInfo.opName, SEQUSED_KV_NAME.c_str(), ("[" + std::to_string(shape.GetDim(0)) + "]").c_str(),
                    ("The value of dim0 of " + SEQUSED_KV_NAME + " shape must be equal to B(" +
                     std::to_string(qfaInfo.bSize) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaCuSeqlensQ(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验): dtype=INT32, format=ND, shape=(B+1,)
    // 值约束(非递减、首元素0、末元素=Q_T)需读 tensor 数值，不校验
    const gert::Tensor *tensor = qfaInfo.opParamInfo.cuSeqlensQ.tensor;
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.cuSeqlensQ.desc;
    if (tensor == nullptr || desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, CU_SEQLENS_Q_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, CU_SEQLENS_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, CU_SEQLENS_Q_NAME.c_str(),
                                             (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != qfaInfo.bSize + 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    qfaInfo.opName, CU_SEQLENS_Q_NAME.c_str(), ("[" + std::to_string(shape.GetDim(0)) + "]").c_str(),
                    ("The value of dim0 of " + CU_SEQLENS_Q_NAME + " shape must be equal to B+1(" +
                     std::to_string(qfaInfo.bSize + 1) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaCuSeqlensKv(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验): dtype=INT32, format=ND, shape=(B+1,)
    // 值约束(非递减、首元素0、末元素=KV_T)需读 tensor 数值，不校验
    const gert::Tensor *tensor = qfaInfo.opParamInfo.cuSeqlensKv.tensor;
    const gert::CompileTimeTensorDesc *desc = qfaInfo.opParamInfo.cuSeqlensKv.desc;
    if (tensor == nullptr || desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, CU_SEQLENS_KV_NAME.c_str(),
                                          DataTypeToSerialString(desc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(desc, CU_SEQLENS_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, CU_SEQLENS_KV_NAME.c_str(),
                                             (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(0) != qfaInfo.bSize + 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    qfaInfo.opName, CU_SEQLENS_KV_NAME.c_str(), ("[" + std::to_string(shape.GetDim(0)) + "]").c_str(),
                    ("The value of dim0 of " + CU_SEQLENS_KV_NAME + " shape must be equal to B+1(" +
                     std::to_string(qfaInfo.bSize + 1) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaMaxSeqlenQ(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验): data_type 支持 INT32，默认值为 -1
    // 数值约束已移至特性交叉校验(CheckFeature): 非TND时与seqused_q至少传1个
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaMaxSeqlenKv(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验): data_type 支持 INT32，默认值为 -1
    // 数值约束已移至特性交叉校验(CheckFeature): 非TND非PA时与seqused_kv至少传1个
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    if (CheckSingleParaSequsedQ(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaSequsedKv(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensQ(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensKv(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenQ(qfaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenKv(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckParaExistence(const QfaTilingInfo &qfaInfo)
{
    // 约束(存在性校验列):
    //   - seqused_q / seqused_kv: 可选参数
    //   - cu_seqlens_q / cu_seqlens_kv: 可选参数
    //   - max_seqlen_q / max_seqlen_kv: 可选属性，默认值为 -1
    // 各参数的存在性约束(如非TND时A/B至少传1个、PA时seqused_kv必传)属于特性交叉校验(CheckFeature)。
    // seqused[b] <= cu_seqlens[b+1] - cu_seqlens[b] 属于值约束，tiling 阶段无法获取 tensor 数值，不校验。
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckCuSeqlensLayoutConsistency(const QfaTilingInfo &qfaInfo)
{
    // 约束(特性交叉校验列):
    //   - cu_seqlens_q: 当 layout_q 为 TND 时，必须传入；当 layout_q 不为 TND 时，不支持传入
    //   - cu_seqlens_kv: 当 layout_kv 为 TND 时，必须传入；当 layout_kv 不为 TND 时，不支持传入
    bool cuSeqlensQExists =
        (qfaInfo.opParamInfo.cuSeqlensQ.tensor != nullptr && qfaInfo.opParamInfo.cuSeqlensQ.desc != nullptr);
    if (qfaInfo.qLayout == QfaLayout::TND || qfaInfo.qLayout == QfaLayout::NTD) {
        OP_CHECK_IF(!cuSeqlensQExists, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, CU_SEQLENS_Q_NAME.c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cuSeqlensQExists,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, CU_SEQLENS_Q_NAME.c_str(), "provided",
                        ("cu_seqlens_q should not be provided when layout_q is " +
                         QfaLayoutToSerialString(qfaInfo.qLayout) + ", only supported in TND and NTD layout")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }

    bool cuSeqlensKvExists =
        (qfaInfo.opParamInfo.cuSeqlensKv.tensor != nullptr && qfaInfo.opParamInfo.cuSeqlensKv.desc != nullptr);
    if (qfaInfo.kvLayout == QfaLayout::TND) {
        OP_CHECK_IF(!cuSeqlensKvExists, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, CU_SEQLENS_KV_NAME.c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cuSeqlensKvExists,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, CU_SEQLENS_KV_NAME.c_str(), "provided",
                        ("cu_seqlens_kv should not be provided when layout_kv is " +
                         QfaLayoutToSerialString(qfaInfo.kvLayout) + ", only supported in TND layout")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSequsedMaxSeqlenAtLeastOne(const QfaTilingInfo &qfaInfo)
{
    // 约束(特性交叉校验列):
    //   - seqused_q: 当 layout_q 不为 TND 时, seqused_q 与 max_seqlen_q 至少传入其中一个
    //   - seqused_kv: 当 layout_kv 不为 TND 且不为 PA 场景时, seqused_kv 与 max_seqlen_kv 至少传入其中一个
    //                (PA 场景下 seqused_kv 必传由 paged_attention_checker 负责)
    //   - max_seqlen_q / max_seqlen_kv: 同上
    // 注意: max_seqlen_q/kv 默认值为 -1, 表示未传入; seqused_q/kv 为 nullptr 表示未传入

    // q 侧: layout_q 不为 TND 时, seqused_q 与 max_seqlen_q 至少传1个
    if (qfaInfo.qLayout != QfaLayout::TND && qfaInfo.qLayout != QfaLayout::NTD) {
        bool sequsedQExists =
            (qfaInfo.opParamInfo.sequsedQ.tensor != nullptr && qfaInfo.opParamInfo.sequsedQ.desc != nullptr);
        bool maxSeqQProvided = (qfaInfo.maxSeqQ >= 0);
        OP_CHECK_IF(!sequsedQExists && !maxSeqQProvided,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, "seqused_q, max_seqlen_q", "empty",
                        ("When layout_q is " + QfaLayoutToSerialString(qfaInfo.qLayout) +
                         " (not TND or NTD), at least one of seqused_q or max_seqlen_q must be provided")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }

    // kv 侧: layout_kv 不为 TND 且不为 PA 场景时, seqused_kv 与 max_seqlen_kv 至少传1个
    bool isPaLayout = (qfaInfo.kvLayout == QfaLayout::PA_BBND || qfaInfo.kvLayout == QfaLayout::PA_BNBD ||
                       qfaInfo.kvLayout == QfaLayout::PA_NZ);
    if (qfaInfo.kvLayout != QfaLayout::TND && !isPaLayout) {
        bool sequsedKvExists =
            (qfaInfo.opParamInfo.sequsedKv.tensor != nullptr && qfaInfo.opParamInfo.sequsedKv.desc != nullptr);
        bool maxSeqKvProvided = (qfaInfo.maxSeqKv >= 0);
        OP_CHECK_IF(!sequsedKvExists && !maxSeqKvProvided,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, "seqused_kv, max_seqlen_kv", "empty",
                        ("When layout_kv is " + QfaLayoutToSerialString(qfaInfo.kvLayout) +
                         " (not TND and not PA), at least one of seqused_kv or max_seqlen_kv must be provided")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckFeature(const QfaTilingInfo &qfaInfo)
{
    // 约束(特性交叉校验列):
    //   - cu_seqlens_q / cu_seqlens_kv: 与 layout 的关系约束
    //   - seqused_q / max_seqlen_q: 非TND时至少传1个
    //   - seqused_kv / max_seqlen_kv: 非TND非PA时至少传1个
    // seqused[b] <= cu_seqlens[b+1] - cu_seqlens[b] 属于值约束，tiling 阶段无法获取 tensor 数值，不校验。
    if (CheckCuSeqlensLayoutConsistency(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckSequsedMaxSeqlenAtLeastOne(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckMultiPara(const QfaTilingInfo &qfaInfo)
{
    // 约束(一致性校验列): 全部标注为"无"
    //   - seqused_q / seqused_kv / cu_seqlens_q / cu_seqlens_kv / max_seqlen_q / max_seqlen_kv
    // 无一致性校验需要实现，使用基类默认实现(返回 GRAPH_SUCCESS)即可。
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
