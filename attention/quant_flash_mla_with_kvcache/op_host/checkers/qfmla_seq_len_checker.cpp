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
 * \file qfmla_seq_len_checker.cpp
 * \brief Checker for cu_seqlens_q, seqused_q, max_seqlen_q/kv ( SeqLengths参数组)
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qmla_tiling_info.h"
#include "../qmla_tiling_info_parser.h"
#include "qfmla_seq_len_checker.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

using std::string;
using namespace ge;
using namespace Ops::Base;

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus SeqLenChecker::CheckSingleParaSequsedQ(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    if (!qmlaInfo.seqUsedQFlag) {
        return ge::GRAPH_SUCCESS; // 可选参数，存在性由 CheckFeature 负责
    }
    // dtype: int32
    OP_CHECK_IF(qmlaInfo.opParamInfo.sequsedQ.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, SEQUSED_Q_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qmlaInfo.opParamInfo.sequsedQ.desc->GetDataType()).c_str(),
                    "The dtype of seqused_q must be int32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qmlaInfo.opParamInfo.sequsedQ.desc, SEQUSED_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // shape: (B,)
    const auto& shape = qmlaInfo.opParamInfo.sequsedQ.shape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 1 || static_cast<int64_t>(shape.GetDim(0)) != qmlaInfo.bSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, SEQUSED_Q_NAME, ToString(shape).c_str(),
                                                      "The shape of seqused_q must be (B,)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaCuSeqlensQ(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    if (!qmlaInfo.cuSeqLenQFlag) {
        return ge::GRAPH_SUCCESS;
    }
    // dtype: int32
    OP_CHECK_IF(qmlaInfo.opParamInfo.cuSeqlensQ.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, CU_SEQLENS_Q_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qmlaInfo.opParamInfo.cuSeqlensQ.desc->GetDataType()).c_str(),
                    "The dtype of cu_seqlens_q must be int32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qmlaInfo.opParamInfo.cuSeqlensQ.desc, CU_SEQLENS_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // shape: (B+1,)
    const auto& shape = qmlaInfo.opParamInfo.cuSeqlensQ.shape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 1 || static_cast<int64_t>(shape.GetDim(0)) != qmlaInfo.bSize + 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, CU_SEQLENS_Q_NAME, ToString(shape).c_str(),
                                                      "The shape of cu_seqlens_q must be (B+1,)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaMaxSeqlenQ(const QmlaTilingInfo& qmlaInfo)
{
    // max_seqlen_q: -1 (未传入) 或 >= 1; 0 无意义需拦截
    OP_CHECK_IF(
        qmlaInfo.maxSeqQ < -1 || qmlaInfo.maxSeqQ == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qmlaInfo.opName, "max_seqlen_q", std::to_string(qmlaInfo.maxSeqQ).c_str(),
                                              "max_seqlen_q must be -1 (not provided) or >= 1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaMaxSeqlenKv(const QmlaTilingInfo& qmlaInfo)
{
    // max_seqlen_kv: >= -1
    OP_CHECK_IF(
        qmlaInfo.maxSeqKv < -1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qmlaInfo.opName, "max_seqlen_kv",
                                              std::to_string(qmlaInfo.maxSeqKv).c_str(), "max_seqlen_kv must be >= -1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSinglePara(const QmlaTilingInfo& qmlaInfo)
{
    if (CheckSingleParaSequsedQ(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensQ(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenQ(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenKv(qmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — cu_seqlens_q 与 layout 关系
// ============================================================================

ge::graphStatus SeqLenChecker::CheckCuSeqlensLayoutConsistency(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // layout_q=TND 时, cu_seqlens_q 必须传入; 非 TND 时不支持传入
    if (qmlaInfo.layoutQ == QmlaLayout::TND) {
        OP_CHECK_IF(!qmlaInfo.cuSeqLenQFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CU_SEQLENS_Q_NAME, "provided",
                                                          "When layout_q is TND, cu_seqlens_q must be provided"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(qmlaInfo.cuSeqLenQFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CU_SEQLENS_Q_NAME, "not provided",
                                                          "When layout_q is not TND, cu_seqlens_q is not supported"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — TND 时 seqused_q 与 max_seqlen_q 至少传1个
// ============================================================================

ge::graphStatus SeqLenChecker::CheckSequsedMaxSeqlenAtLeastOne(const QmlaTilingInfo& qmlaInfo)
{
    const char* opName = qmlaInfo.opName;
    // layout_q=TND 时, seqused_q 与 max_seqlen_q 至少传入其中一个
    if (qmlaInfo.layoutQ == QmlaLayout::TND) {
        OP_CHECK_IF(!qmlaInfo.seqUsedQFlag && qmlaInfo.maxSeqQ <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName, "seqused_q", "provided",
                        "When layout_q is TND, seqused_q and max_seqlen_q must be provided at least one of them"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckFeature(const QmlaTilingInfo& qmlaInfo)
{
    if (CheckCuSeqlensLayoutConsistency(qmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSequsedMaxSeqlenAtLeastOne(qmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
