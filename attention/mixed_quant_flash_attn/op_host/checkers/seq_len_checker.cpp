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
 * \file actual_seq_len_checker.cpp
 * \brief Checker for cu_seqlens_q/kv (B+1) and seqused_q/kv (B) parameters
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../mqfa_fa_tiling_info.h"
#include "mqfa_seq_len_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus ActualSeqLenChecker::CheckSingleParaSequsedQ(const FaTilingInfo& faInfo)
{
    auto& sequsedQTensor = faInfo.opParamInfo.seqUsedQ.tensor;
    if (sequsedQTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc* sequsedQDesc = faInfo.opParamInfo.seqUsedQ.desc;
    OP_CHECK_IF(sequsedQDesc != nullptr && sequsedQDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "seqused_q",
                                          DataTypeToSerialString(sequsedQDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(sequsedQDesc, SEQUSED_Q_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sequsedQDimNum = sequsedQTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        sequsedQDimNum != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "seqused_q", (std::to_string(sequsedQDimNum) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        faInfo.seqUsedQSize != faInfo.bSize,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            faInfo.opName, "seqused_q", std::to_string(faInfo.seqUsedQSize).c_str(),
            ("The shape size of seqused_q must be equal to batch_size (" + std::to_string(faInfo.bSize) + ")").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaSequsedKv(const FaTilingInfo& faInfo)
{
    auto& sequsedKvTensor = faInfo.opParamInfo.seqUsedKv.tensor;
    if (sequsedKvTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc* sequsedKvDesc = faInfo.opParamInfo.seqUsedKv.desc;
    OP_CHECK_IF(sequsedKvDesc != nullptr && sequsedKvDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "seqused_kv",
                                          DataTypeToSerialString(sequsedKvDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(sequsedKvDesc, SEQUSED_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sequsedKvDimNum = sequsedKvTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(sequsedKvDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "seqused_kv",
                                             (std::to_string(sequsedKvDimNum) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.seqUsedKvSize != faInfo.bSize,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    faInfo.opName, "seqused_kv", std::to_string(faInfo.seqUsedKvSize).c_str(),
                    ("The shape size of seqused_kv must be equal to batch_size (" + std::to_string(faInfo.bSize) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaCuSeqlensQ(const FaTilingInfo& faInfo)
{
    auto& cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (cuSeqlensQTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc* cuSeqlensQDesc = faInfo.opParamInfo.cuSeqlensQ.desc;
    OP_CHECK_IF(cuSeqlensQDesc != nullptr && cuSeqlensQDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "cu_seqlens_q",
                                          DataTypeToSerialString(cuSeqlensQDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(cuSeqlensQDesc, CU_SEQLENS_Q_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t cuSeqlensQDimNum = cuSeqlensQTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(cuSeqlensQDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "cu_seqlens_q",
                                             (std::to_string(cuSeqlensQDimNum) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        faInfo.cuSeqLensQSize != faInfo.bSize + 1,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            faInfo.opName, "cu_seqlens_q", std::to_string(faInfo.cuSeqLensQSize).c_str(),
            ("The shape size of cu_seqlens_q must be equal to batch_size (" + std::to_string(faInfo.bSize) + ") plus 1")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaMaxSeqlenQ(const FaTilingInfo& faInfo)
{
    if (faInfo.maxSeqQ < -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSingleParaMaxSeqlenKv(const FaTilingInfo& faInfo)
{
    if (faInfo.maxSeqKv < -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FaTilingInfo& faInfo)
{
    if (CheckSingleParaSequsedQ(faInfo) != ge::GRAPH_SUCCESS || CheckSingleParaSequsedKv(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenQ(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenKv(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckParaExistence(const FaTilingInfo& faInfo)
{
    if (faInfo.pageAttentionFlag) {
        auto& sequsedKvTensor = faInfo.opParamInfo.seqUsedKv.tensor;
        OP_CHECK_IF(sequsedKvTensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        faInfo.opName, "seqused_kv", "When PagedAttention is enabled, seqused_kv cannot be empty"),
                    return ge::GRAPH_FAILED);
    }

    auto& cuSeqlensQTensor = faInfo.opParamInfo.cuSeqlensQ.tensor;
    if (faInfo.qLayout == FaLayout::TND) {
        OP_CHECK_IF(cuSeqlensQTensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "cu_seqlens_q",
                                                             "When layout_q is TND, cu_seqlens_q cannot be empty"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cuSeqlensQTensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "cu_seqlens_q",
                                                             "When layout_q is not TND, cu_seqlens_q cannot be passed"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckFeature(const FaTilingInfo& faInfo)
{
    (void)faInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckMultiPara(const FaTilingInfo& faInfo)
{
    // 非TND场景：A组(seqused_q/seqused_kv)和B组(max_seqlen_q/max_seqlen_kv)至少传入一组
    if (faInfo.qLayout != FaLayout::TND) {
        auto& sequsedQTensor = faInfo.opParamInfo.seqUsedQ.tensor;
        auto& sequsedKvTensor = faInfo.opParamInfo.seqUsedKv.tensor;
        bool aGroupProvided = (sequsedQTensor != nullptr) || (sequsedKvTensor != nullptr);
        bool bGroupProvided = (faInfo.maxSeqQ != -1) || (faInfo.maxSeqKv != -1);
        OP_CHECK_IF(!aGroupProvided && !bGroupProvided,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        faInfo.opName, "seqused_q/seqused_kv/max_seqlen_q/max_seqlen_kv",
                        "When layout_q is not TND, at least one of seqused_q, seqused_kv, max_seqlen_q and "
                        "max_seqlen_kv must be provided"),
                    return ge::GRAPH_FAILED);

        // 非TND场景下，seqused_q未传入时，max_seqlen_q必须等于q的S轴大小
        OP_CHECK_IF(sequsedQTensor == nullptr && faInfo.maxSeqQ != static_cast<int64_t>(faInfo.s1Size),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        faInfo.opName, "max_seqlen_q", std::to_string(faInfo.maxSeqQ).c_str(),
                        ("When layout_q is BSND or BNSD and seqused_q is not passed, the value of max_seqlen_q "
                         "must be equal to the size (" +
                         std::to_string(faInfo.s1Size) + ") of the axis S of q")
                            .c_str()),
                    return ge::GRAPH_FAILED);

        // 非TND场景下，若只传了B组(max_seqlen_q、max_seqlen_kv)参数，则max_seqlen_q需等于Q_S，max_seqlen_kv需等于KV_S
        if (!aGroupProvided && bGroupProvided) {
            OP_CHECK_IF(faInfo.maxSeqQ != static_cast<int64_t>(faInfo.s1Size),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            faInfo.opName, "max_seqlen_q", std::to_string(faInfo.maxSeqQ).c_str(),
                            ("When layout_q is not TND and neither seqused_q nor seqused_kv is provided, the value of "
                             "max_seqlen_q must be equal to the sequence length of q (" +
                             std::to_string(faInfo.s1Size) + ")")
                                .c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(faInfo.maxSeqKv != static_cast<int64_t>(faInfo.s2Size),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            faInfo.opName, "max_seqlen_kv", std::to_string(faInfo.maxSeqKv).c_str(),
                            ("When layout_q is not TND and neither seqused_q nor seqused_kv is provided, the value "
                             "of max_seqlen_kv must be equal to the sequence length of k/v (" +
                             std::to_string(faInfo.s2Size) + ")")
                                .c_str()),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace mixed_quant_flash_attn
} // namespace optiling
