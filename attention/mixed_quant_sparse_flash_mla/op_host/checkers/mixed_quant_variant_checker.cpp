/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mixed_quant_variant_checker.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
constexpr int64_t TURBO_QUANT_MODE = 3;
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "MixedQuantSparseFlashMla" : context.opName;
}
} // namespace

ge::graphStatus MixedQuantVariantChecker::CheckSinglePara(const CheckContext &context) const
{
    OP_CHECK_IF(
        context.quantMode != 1 && context.quantMode != 2 && context.quantMode != TURBO_QUANT_MODE,
        OP_LOGE_FOR_INVALID_VALUE(Op(context), "quant_mode", std::to_string(context.quantMode).c_str(), "1, 2 or 3"),
        return ge::GRAPH_FAILED);
    const bool turboQuant = context.quantMode == TURBO_QUANT_MODE;
    OP_CHECK_IF(
        (turboQuant && context.npuArch != NpuArch::DAV_2201) || (!turboQuant && context.npuArch != NpuArch::DAV_3510),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            Op(context), "quant_mode", std::to_string(context.quantMode).c_str(),
            "quant_mode 3 is supported only on arch22; quant_mode 1/2 are supported only on arch35"),
        return ge::GRAPH_FAILED);
    // The RoPE part of the TurboQuant layout has a fixed 64-dimensional head.
    OP_CHECK_IF(
        context.ropeHeadDim != 64,
        OP_LOGE_FOR_INVALID_VALUE(Op(context), "rope_head_dim", std::to_string(context.ropeHeadDim).c_str(), "64"),
        return ge::GRAPH_FAILED);
    if (!turboQuant) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        context.qLayout != Layout::TND || context.kvLayout != Layout::PA_BBND,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "layout_q/layout_kv", "TurboQuant requires TND/PA_BBND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.qNumHeads % 4 != 0,
                OP_LOGE_FOR_INVALID_VALUE(Op(context), "q head count", std::to_string(context.qNumHeads).c_str(),
                                          "a multiple of 4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.cmpTopk != 512 && context.cmpTopk != 1024,
                OP_LOGE_FOR_INVALID_VALUE(Op(context), "cmp_sparse_indices K", std::to_string(context.cmpTopk).c_str(),
                                          "512 or 1024"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context.oriMaskMode != 4 || context.cmpMaskMode != 3 || context.oriWinRight != 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            Op(context), "mask attributes", "TurboQuant requires ori_mask_mode=4, cmp_mask_mode=3 and ori_win_right=0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context.oriWinLeft < 0 || (context.cmpRatio != 4 && context.cmpRatio != 128),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "TurboQuant attributes",
                                                 "ori_win_left must be non-negative and cmp_ratio must be 4 or 128"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.oriBlockSize <= 0 || context.oriBlockSize > 1024 || context.oriBlockSize % 16 != 0 ||
                    context.cmpBlockSize <= 0 || context.cmpBlockSize > 1024 || context.cmpBlockSize % 16 != 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "KV block size", "TurboQuant block sizes must be in [16, 1024] and aligned to 16"),
                return ge::GRAPH_FAILED);
    const uint64_t minOriKvStride =
        static_cast<uint64_t>(context.oriBlockSize) * static_cast<uint64_t>(context.kvNumHeads) * 512U;
    const uint64_t minCmpKvStride =
        static_cast<uint64_t>(context.cmpBlockSize) * static_cast<uint64_t>(context.kvNumHeads) * 258U;
    OP_CHECK_IF(context.oriKvStrides.empty() || context.cmpKvStrides.empty() || context.oriKvStrides[0] < 0 ||
                    context.cmpKvStrides[0] < 0 || static_cast<uint64_t>(context.oriKvStrides[0]) < minOriKvStride ||
                    static_cast<uint64_t>(context.cmpKvStrides[0]) < minCmpKvStride,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "KV stride", "The first-axis stride must cover one physical TurboQuant cache block"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
