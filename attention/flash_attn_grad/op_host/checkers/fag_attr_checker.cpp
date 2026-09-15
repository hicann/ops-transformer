/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fag_attr_checker.h"

#include "log/log.h"

namespace optiling {
namespace {

// 属性缺省时保留 ctx 里的默认值，不报错：这些属性在 flash_attn_grad_def.cpp
// 里都带默认值，图模式下可能整个 attrs 都拿不到。
void ReadInt64Attr(const gert::RuntimeAttrs *attrs, size_t index, int64_t &out)
{
    auto ptr = attrs->GetAttrPointer<int64_t>(index);
    if (ptr != nullptr) {
        out = *ptr;
    }
}

void ReadStrAttr(const gert::RuntimeAttrs *attrs, size_t index, std::string &out)
{
    auto ptr = attrs->GetAttrPointer<char>(index);
    if (ptr != nullptr) {
        out = std::string(ptr);
    }
}

// "必须是 -1 或非负" 的属性反复出现，收成一条。-1 表示"未设置"。
ge::graphStatus CheckMinusOneOrNonNegative(const char *opName, const char *attrName, int64_t value)
{
    OP_CHECK_IF(value < -1, OP_LOGE(opName, "%s must be -1 or >= 0, but got %ld.", attrName, value),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

bool IsSupportedLayout(const std::string &layout)
{
    return layout == "BSND" || layout == "TND" || layout == "BNSD";
}

} // namespace

ge::graphStatus FagAttrRangeChecker::Check(FagCheckCtx &ctx)
{
    const char *opName = ctx.opName;

    auto attrs = ctx.context->GetAttrs();
    if (attrs != nullptr) {
        ReadInt64Attr(attrs, ATTR_MASK_MODE_INDEX, ctx.maskMode);
        ReadInt64Attr(attrs, ATTR_WIN_LEFT_INDEX, ctx.winLeft);
        ReadInt64Attr(attrs, ATTR_WIN_RIGHT_INDEX, ctx.winRight);
        ReadInt64Attr(attrs, ATTR_MAX_SEQLEN_Q_INDEX, ctx.maxSeqlenQ);
        ReadInt64Attr(attrs, ATTR_MAX_SEQLEN_KV_INDEX, ctx.maxSeqlenKv);
        ReadStrAttr(attrs, ATTR_LAYOUT_Q_INDEX, ctx.layoutQ);
        ReadStrAttr(attrs, ATTR_LAYOUT_KV_INDEX, ctx.layoutKv);
        ReadStrAttr(attrs, ATTR_LAYOUT_OUT_INDEX, ctx.layoutOut);
    }

    OP_CHECK_IF(
        ctx.maskMode != MASK_MODE_NO_MASK && ctx.maskMode != MASK_MODE_CAUSAL && ctx.maskMode != MASK_MODE_WINDOW,
        OP_LOGE(opName, "mask_mode only supports 0, 3, 4, but got %ld.", ctx.maskMode), return ge::GRAPH_PARAM_INVALID);

    if (CheckMinusOneOrNonNegative(opName, "win_left", ctx.winLeft) != ge::GRAPH_SUCCESS ||
        CheckMinusOneOrNonNegative(opName, "win_right", ctx.winRight) != ge::GRAPH_SUCCESS ||
        CheckMinusOneOrNonNegative(opName, "max_seqlen_q", ctx.maxSeqlenQ) != ge::GRAPH_SUCCESS ||
        CheckMinusOneOrNonNegative(opName, "max_seqlen_kv", ctx.maxSeqlenKv) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FagMaskChecker::Check(FagCheckCtx &ctx)
{
    const char *opName = ctx.opName;
    const bool hasMask = IsOptionalTensorExist(ctx.context, ATTN_MASK_INDEX);

    // mask_mode 决定 attn_mask 该不该来、window 参数该不该给：
    //   0 无 mask，attn_mask 与 window 都不能出现；
    //   3 causal，需要 attn_mask，但不吃 window；
    //   4 window，需要 attn_mask，且 window 两侧都必须显式给。
    if (ctx.maskMode == MASK_MODE_NO_MASK) {
        OP_CHECK_IF(hasMask, OP_LOGE(opName, "attn_mask must not be provided when mask_mode is 0."),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(ctx.winLeft != -1,
                    OP_LOGE(opName, "win_left must be -1 when mask_mode is 0, but got %ld.", ctx.winLeft),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(ctx.winRight != -1,
                    OP_LOGE(opName, "win_right must be -1 when mask_mode is 0, but got %ld.", ctx.winRight),
                    return ge::GRAPH_PARAM_INVALID);
    } else if (ctx.maskMode == MASK_MODE_CAUSAL) {
        OP_CHECK_IF(!hasMask, OP_LOGE(opName, "attn_mask must be provided when mask_mode is 3."),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(ctx.winLeft != -1,
                    OP_LOGE(opName, "win_left must be -1 when mask_mode is 3, but got %ld.", ctx.winLeft),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(ctx.winRight != -1,
                    OP_LOGE(opName, "win_right must be -1 when mask_mode is 3, but got %ld.", ctx.winRight),
                    return ge::GRAPH_PARAM_INVALID);
    } else if (ctx.maskMode == MASK_MODE_WINDOW) {
        OP_CHECK_IF(!hasMask, OP_LOGE(opName, "attn_mask must be provided when mask_mode is 4."),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(ctx.winLeft < 0,
                    OP_LOGE(opName, "win_left must be >= 0 when mask_mode is 4, but got %ld.", ctx.winLeft),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(ctx.winRight < 0,
                    OP_LOGE(opName, "win_right must be >= 0 when mask_mode is 4, but got %ld.", ctx.winRight),
                    return ge::GRAPH_PARAM_INVALID);
    }

    if (hasMask) {
        auto &maskShape = ctx.context->GetOptionalInputShape(ATTN_MASK_INDEX)->GetStorageShape();
        OP_CHECK_IF(maskShape.GetDimNum() != 2U,
                    OP_LOGE(opName, "attn_mask must be 2D, but got %zu dims.", maskShape.GetDimNum()),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(maskShape.GetDim(0) != ATTN_MASK_DIM || maskShape.GetDim(1) != ATTN_MASK_DIM,
                    OP_LOGE(opName, "attn_mask shape must be [%ld, %ld], but got [%ld, %ld].", ATTN_MASK_DIM,
                            ATTN_MASK_DIM, maskShape.GetDim(0), maskShape.GetDim(1)),
                    return ge::GRAPH_PARAM_INVALID);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FagLayoutChecker::Check(FagCheckCtx &ctx)
{
    const char *opName = ctx.opName;

    const struct {
        const char *attrName;
        const std::string &value;
    } layouts[] = {
        {"layout_q", ctx.layoutQ},
        {"layout_kv", ctx.layoutKv},
        {"layout_out", ctx.layoutOut},
    };

    for (const auto &item : layouts) {
        OP_CHECK_IF(!IsSupportedLayout(item.value),
                    OP_LOGE(opName, "%s only supports BSND, TND, BNSD, but got %s.", item.attrName, item.value.c_str()),
                    return ge::GRAPH_PARAM_INVALID);
    }

    // 三者必须相同：kernel 只按一个 layout 推导视图参数，混用会静默读错内存。
    OP_CHECK_IF(ctx.layoutQ != ctx.layoutKv || ctx.layoutQ != ctx.layoutOut,
                OP_LOGE(opName, "layout_q(%s), layout_kv(%s), layout_out(%s) must be the same.", ctx.layoutQ.c_str(),
                        ctx.layoutKv.c_str(), ctx.layoutOut.c_str()),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
