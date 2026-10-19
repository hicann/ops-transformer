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
#include "../qfmla_tiling_info.h"
#include "../qfmla_tiling_info_parser.h"
#include "qfmla_seq_len_checker.h"

namespace optiling {
namespace quant_flash_mla {

using std::string;
using namespace ge;
using namespace Ops::Base;

// ============================================================================
// SinglePara
// ============================================================================

ge::graphStatus SeqLenChecker::CheckSingleParaSequsedQ(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    if (!qfmlaInfo.seqUsedQFlag) {
        return ge::GRAPH_SUCCESS; // 可选参数，存在性由 CheckFeature 负责
    }
    // dtype: int32
    OP_CHECK_IF(qfmlaInfo.opParamInfo.sequsedQ.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, SEQUSED_Q_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.sequsedQ.desc->GetDataType()).c_str(),
                    "The dtype of seqused_q must be int32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qfmlaInfo.opParamInfo.sequsedQ.desc, SEQUSED_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // shape: (B,)
    const auto& shape = qfmlaInfo.opParamInfo.sequsedQ.shape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 1 || static_cast<int64_t>(shape.GetDim(0)) != qfmlaInfo.bSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, SEQUSED_Q_NAME, ToString(shape).c_str(),
                                                      "The shape of seqused_q must be (B,)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaCuSeqlensQ(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    if (!qfmlaInfo.cuSeqLenQFlag) {
        return ge::GRAPH_SUCCESS;
    }
    // dtype: int32
    OP_CHECK_IF(qfmlaInfo.opParamInfo.cuSeqlensQ.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, CU_SEQLENS_Q_NAME,
                    ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.cuSeqlensQ.desc->GetDataType()).c_str(),
                    "The dtype of cu_seqlens_q must be int32"),
                return ge::GRAPH_FAILED);
    // format: ND
    if (CheckFormatSupport(qfmlaInfo.opParamInfo.cuSeqlensQ.desc, CU_SEQLENS_Q_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // shape: (B+1,)
    const auto& shape = qfmlaInfo.opParamInfo.cuSeqlensQ.shape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 1 || static_cast<int64_t>(shape.GetDim(0)) != qfmlaInfo.bSize + 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, CU_SEQLENS_Q_NAME, ToString(shape).c_str(),
                                                      "The shape of cu_seqlens_q must be (B+1,)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaMaxSeqlenQ(const QfmlaTilingInfo& qfmlaInfo)
{
    // MLA 当前单 batch Q 长度最大为 16；-1 表示未传入。
    OP_CHECK_IF(qfmlaInfo.maxSeqQ < -1 || qfmlaInfo.maxSeqQ == 0 || qfmlaInfo.maxSeqQ > 16,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfmlaInfo.opName, "max_seqlen_q",
                                                      std::to_string(qfmlaInfo.maxSeqQ).c_str(),
                                                      "max_seqlen_q must be -1 (not provided) or in [1, 16]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSingleParaMaxSeqlenKv(const QfmlaTilingInfo& qfmlaInfo)
{
    // max_seqlen_kv: -1 (未传入) 或 >= 1；0 无意义需拦截。
    OP_CHECK_IF(qfmlaInfo.maxSeqKv < -1 || qfmlaInfo.maxSeqKv == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfmlaInfo.opName, "max_seqlen_kv",
                                                      std::to_string(qfmlaInfo.maxSeqKv).c_str(),
                                                      "max_seqlen_kv must be -1 (not provided) or >= 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckSingleParaSequsedQ(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaCuSeqlensQ(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenQ(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaMaxSeqlenKv(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — cu_seqlens_q 与 layout 关系
// ============================================================================

ge::graphStatus SeqLenChecker::CheckCuSeqlensLayoutConsistency(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // layout_q=TND 时, cu_seqlens_q 必须传入; 非 TND 时不支持传入
    if (qfmlaInfo.layoutQ == QfmlaLayout::TND) {
        OP_CHECK_IF(!qfmlaInfo.cuSeqLenQFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CU_SEQLENS_Q_NAME, "provided",
                                                          "When layout_q is TND, cu_seqlens_q must be provided"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(qfmlaInfo.cuSeqLenQFlag,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CU_SEQLENS_Q_NAME, "not provided",
                                                          "When layout_q is not TND, cu_seqlens_q is not supported"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Feature — TND 时 seqused_q 与 max_seqlen_q 至少传1个
// ============================================================================

ge::graphStatus SeqLenChecker::CheckSequsedMaxSeqlenAtLeastOne(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // layout_q=TND 时, seqused_q 与 max_seqlen_q 至少传入其中一个
    if (qfmlaInfo.layoutQ == QfmlaLayout::TND) {
        OP_CHECK_IF(!qfmlaInfo.seqUsedQFlag && qfmlaInfo.maxSeqQ <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName, "seqused_q", "provided",
                        "When layout_q is TND, seqused_q and max_seqlen_q must be provided at least one of them"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckFeature(const QfmlaTilingInfo& qfmlaInfo)
{
    if (CheckCuSeqlensLayoutConsistency(qfmlaInfo) != ge::GRAPH_SUCCESS ||
        CheckSequsedMaxSeqlenAtLeastOne(qfmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling
