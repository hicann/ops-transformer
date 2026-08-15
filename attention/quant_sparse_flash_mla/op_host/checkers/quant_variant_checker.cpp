/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_variant_checker.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "QuantSparseFlashMla" : context.opName;
}
} // namespace

ge::graphStatus QuantVariantChecker::CheckDescale(const CheckContext &context, const TensorParam &param,
                                                  const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckTensorDesc(context, param, name, {ge::DT_FLOAT}) != ge::GRAPH_SUCCESS ||
        CheckShape(context, param, name, {1}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantVariantChecker::CheckSinglePara(const CheckContext &context) const
{
    OP_CHECK_IF(context.quantMode != 1,
                OP_LOGE_FOR_INVALID_VALUE(Op(context), "quant_mode", std::to_string(context.quantMode).c_str(), "1"),
                return ge::GRAPH_FAILED);
    if (CheckDescale(context, context.qDescale, "q_descale") != ge::GRAPH_SUCCESS ||
        CheckDescale(context, context.oriKvDescale, "ori_kv_descale") != ge::GRAPH_SUCCESS ||
        CheckDescale(context, context.cmpKvDescale, "cmp_kv_descale") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantVariantChecker::CheckParaExistence(const CheckContext &context) const
{
    OP_CHECK_IF(!context.qDescale.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "q_descale", "Q_descale is required when quant_mode=1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!context.oriKvDescale.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "ori_kv_descale", "Ori_kv_descale is required when quant_mode=1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.cmpKv.present != context.cmpKvDescale.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "cmp_kv_descale",
                    "Cmp_kv_descale must be present exactly when cmp_kv is present"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
