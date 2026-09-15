/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fag_shape_checker.h"

#include "log/log.h"

namespace optiling {
namespace {

// 八个参与 shape 校验的张量在这里统一带上名字，好让下面的循环把错误信息里的
// 张量名打出来 —— 原先是每个张量各写一遍 OP_CHECK_IF，加一个张量要改八处。
struct TensorRef {
    const char *name;
    const gert::Shape *shape;
};

// 每个张量按 layout 应有的维度数：TND 是 3 维，BSND/BNSD 是 4 维。
ge::graphStatus CheckDimNum(const char *opName, const char *layout, size_t expectDims, const TensorRef *tensors,
                            size_t num)
{
    for (size_t i = 0; i < num; ++i) {
        OP_CHECK_IF(tensors[i].shape->GetDimNum() != expectDims,
                    OP_LOGE(opName, "%s must be %zuD when layout is %s, but got %zu dims.", tensors[i].name, expectDims,
                            layout, tensors[i].shape->GetDimNum()),
                    return ge::GRAPH_PARAM_INVALID);
    }
    return ge::GRAPH_SUCCESS;
}

// 一组张量在某个轴上必须与参考张量取值相同。dimTag 只用于错误信息。
ge::graphStatus CheckDimEqual(const char *opName, const char *dimTag, const char *refName, int64_t refVal,
                              size_t dimIdx, const TensorRef *tensors, size_t num)
{
    for (size_t i = 0; i < num; ++i) {
        const int64_t got = tensors[i].shape->GetDim(dimIdx);
        OP_CHECK_IF(got != refVal,
                    OP_LOGE(opName, "%s dim of %s must equal %s %s(%ld), but got %ld.", dimTag, tensors[i].name,
                            refName, dimTag, refVal, got),
                    return ge::GRAPH_PARAM_INVALID);
    }
    return ge::GRAPH_SUCCESS;
}

// 各轴在 storage shape 里的下标只取决于 layout，集中在这里换算，避免下面散落
// isTnd / isBnsd 的三元表达式。
struct AxisIndex {
    size_t d;
    size_t n;
    size_t s; // TND 下无意义，调用方以 isTnd 短路
};

AxisIndex GetAxisIndex(bool isTnd, bool isBnsd)
{
    if (isTnd) {
        return {2U, 1U, 0U}; // [T, N, D]
    }
    if (isBnsd) {
        return {3U, 1U, 2U}; // [B, N, S, D]
    }
    return {3U, 2U, 1U}; // [B, S, N, D]
}

ge::graphStatus CheckHeadDim(const char *opName, const AxisIndex &axis, const TensorRef &q, const TensorRef &v,
                             const TensorRef *qSide, size_t qSideNum, const TensorRef *vSide, size_t vSideNum)
{
    // Q 侧（q/dq/dk/k）共用 D，V 侧（v/dout/attn_out/dv）共用 Dv，两者可以不等。
    const int64_t qD = q.shape->GetDim(axis.d);
    const int64_t vD = v.shape->GetDim(axis.d);
    if (CheckDimEqual(opName, "D", "q", qD, axis.d, qSide, qSideNum) != ge::GRAPH_SUCCESS ||
        CheckDimEqual(opName, "D", "v", vD, axis.d, vSide, vSideNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_PARAM_INVALID;
    }

    // Dv <= D 不只是惯例：kernel 把每块片上 tile 都按 D 派生的宽度分配，V/dO 侧
    // 只填前 Dv 列（见 d_align tilingkey 字段）。Dv > D 会写越界。
    // 必须查：kernel 按 D 分配 tile，Dv > D 会写越界。
    OP_CHECK_IF(vD > qD, OP_LOGE(opName, "D dim of v (%ld) must not exceed D dim of q (%ld).", vD, qD),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(vD <= 0 || qD <= 0, OP_LOGE(opName, "D dims must be positive, but got q D=%ld, v D=%ld.", qD, vD),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(qD > MAX_HEAD_DIM || vD > MAX_HEAD_DIM,
                OP_LOGE(opName, "D dims must be in (0, %ld], but got q D=%ld, v D=%ld.", MAX_HEAD_DIM, qD, vD),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckCuSeqlens(const char *opName, gert::TilingContext *context, bool isTnd)
{
    const struct {
        const char *name;
        size_t index;
    } cuSeqlens[] = {
        {"cu_seqlens_q", CU_SEQLENS_Q_INDEX},
        {"cu_seqlens_kv", CU_SEQLENS_KV_INDEX},
    };

    for (const auto &item : cuSeqlens) {
        const bool exists = IsOptionalTensorExist(context, item.index);
        if (!isTnd) {
            OP_CHECK_IF(exists, OP_LOGE(opName, "%s must not be provided when layout is not TND.", item.name),
                        return ge::GRAPH_PARAM_INVALID);
            continue;
        }
        OP_CHECK_IF(!exists, OP_LOGE(opName, "%s must be provided when layout is TND.", item.name),
                    return ge::GRAPH_PARAM_INVALID);
        auto &shape = context->GetOptionalInputShape(item.index)->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1U,
                    OP_LOGE(opName, "%s must be 1D, but got %zu dims.", item.name, shape.GetDimNum()),
                    return ge::GRAPH_PARAM_INVALID);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus FagShapeChecker::Check(FagCheckCtx &ctx)
{
    gert::TilingContext *context = ctx.context;
    const char *opName = ctx.opName;
    const char *layout = ctx.layoutQ.c_str();
    const bool isTnd = (ctx.layoutQ == "TND");
    const bool isBnsd = (ctx.layoutQ == "BNSD");

    const TensorRef q = {"q", &context->GetInputShape(Q_INDEX)->GetStorageShape()};
    const TensorRef k = {"k", &context->GetInputShape(K_INDEX)->GetStorageShape()};
    const TensorRef v = {"v", &context->GetInputShape(V_INDEX)->GetStorageShape()};
    const TensorRef dout = {"dout", &context->GetInputShape(DOUT_INDEX)->GetStorageShape()};
    const TensorRef attnOut = {"attn_out", &context->GetInputShape(ATTN_OUT_INDEX)->GetStorageShape()};
    const TensorRef dq = {"dq", &context->GetOutputShape(DQ_OUT_INDEX)->GetStorageShape()};
    const TensorRef dk = {"dk", &context->GetOutputShape(DK_OUT_INDEX)->GetStorageShape()};
    const TensorRef dv = {"dv", &context->GetOutputShape(DV_OUT_INDEX)->GetStorageShape()};
    auto &softmaxLseSS = context->GetInputShape(SOFTMAX_LSE_INDEX)->GetStorageShape();

    // softmax_lse 的维度数与其他张量不同（TND 下是 2D 的 (N, T)），单独校验。
    const TensorRef mainTensors[] = {q, k, v, dout, attnOut, dq, dk, dv};
    const size_t expectDims = isTnd ? 3U : 4U;
    if (CheckDimNum(opName, layout, expectDims, mainTensors, sizeof(mainTensors) / sizeof(mainTensors[0])) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_PARAM_INVALID;
    }
    if (isTnd) {
        OP_CHECK_IF(softmaxLseSS.GetDimNum() != 2U,
                    OP_LOGE(opName, "softmax_lse must be 2D (N, T) when layout is TND, but got %zu dims.",
                            softmaxLseSS.GetDimNum()),
                    return ge::GRAPH_PARAM_INVALID);
    }

    const AxisIndex axis = GetAxisIndex(isTnd, isBnsd);

    const TensorRef dSideQ[] = {k, dq, dk};
    const TensorRef dSideV[] = {dout, dv, attnOut};
    if (CheckHeadDim(opName, axis, q, v, dSideQ, sizeof(dSideQ) / sizeof(dSideQ[0]), dSideV,
                     sizeof(dSideV) / sizeof(dSideV[0])) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_PARAM_INVALID;
    }

    // N：Q 侧共用 N1，KV 侧共用 N2，且 N1 必须是 N2 的整数倍（GQA 的 G）。
    const int64_t qN = q.shape->GetDim(axis.n);
    const int64_t kN = k.shape->GetDim(axis.n);
    const TensorRef qSide[] = {dout, dq, attnOut};
    const TensorRef kvSide[] = {v, dk, dv};
    if (CheckDimEqual(opName, "N", "q", qN, axis.n, qSide, sizeof(qSide) / sizeof(qSide[0])) != ge::GRAPH_SUCCESS ||
        CheckDimEqual(opName, "N", "k", kN, axis.n, kvSide, sizeof(kvSide) / sizeof(kvSide[0])) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_PARAM_INVALID;
    }
    OP_CHECK_IF(kN == 0 || qN % kN != 0,
                OP_LOGE(opName, "N dim of q(%ld) must be divisible by N dim of k(%ld).", qN, kN),
                return ge::GRAPH_PARAM_INVALID);

    // S 与 B 只在非 TND 下成轴；TND 把 B*S 压成 T，长度由 cu_seqlens 描述。
    if (!isTnd) {
        const int64_t qS = q.shape->GetDim(axis.s);
        const int64_t kS = k.shape->GetDim(axis.s);
        if (CheckDimEqual(opName, "S", "q", qS, axis.s, qSide, sizeof(qSide) / sizeof(qSide[0])) != ge::GRAPH_SUCCESS ||
            CheckDimEqual(opName, "S", "k", kS, axis.s, kvSide, sizeof(kvSide) / sizeof(kvSide[0])) !=
                ge::GRAPH_SUCCESS) {
            return ge::GRAPH_PARAM_INVALID;
        }

        const int64_t bDim = q.shape->GetDim(0);
        const TensorRef bTensors[] = {k, v, dout, dq, dk, dv, attnOut};
        if (CheckDimEqual(opName, "B", "q", bDim, 0U, bTensors, sizeof(bTensors) / sizeof(bTensors[0])) !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_PARAM_INVALID;
        }
    }

    return CheckCuSeqlens(opName, context, isTnd);
}

} // namespace optiling
