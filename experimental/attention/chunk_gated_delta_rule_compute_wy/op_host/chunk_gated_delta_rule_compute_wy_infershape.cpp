/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace gert;

namespace ops {
namespace {
constexpr size_t Q_INDEX = 0;
constexpr size_t V_INDEX = 2;
constexpr size_t G_INDEX = 3;

constexpr size_t Q_KERNEL_INDEX = 0;
constexpr size_t K_KERNEL_INDEX = 1;
constexpr size_t W_KERNEL_INDEX = 2;
constexpr size_t U_KERNEL_INDEX = 3;
constexpr size_t G_KERNEL_INDEX = 4;

constexpr size_t DIM_B = 0;
constexpr size_t DIM_T = 1;
constexpr size_t DIM_H = 2;
constexpr size_t DIM_D = 3;

constexpr size_t QKV_DIM_NUM = 4;

// [B, T, H, D] -> [B, H, T, D]
void SetBhtd(gert::Shape *out, int64_t b, int64_t h, int64_t t, int64_t d)
{
    out->SetDimNum(QKV_DIM_NUM);
    out->SetDim(DIM_B, b);
    out->SetDim(1, h);
    out->SetDim(2, t);
    out->SetDim(3, d);
}
} // namespace

static ge::graphStatus InferShapeChunkGatedDeltaRuleComputeWy(InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto qShape = context->GetInputShape(Q_INDEX);
    const auto vShape = context->GetInputShape(V_INDEX);
    auto qKernelShape = context->GetOutputShape(Q_KERNEL_INDEX);
    auto kKernelShape = context->GetOutputShape(K_KERNEL_INDEX);
    auto wKernelShape = context->GetOutputShape(W_KERNEL_INDEX);
    auto uKernelShape = context->GetOutputShape(U_KERNEL_INDEX);
    auto gKernelShape = context->GetOutputShape(G_KERNEL_INDEX);
    if (qShape == nullptr || vShape == nullptr || qKernelShape == nullptr || kKernelShape == nullptr ||
        wKernelShape == nullptr || uKernelShape == nullptr || gKernelShape == nullptr) {
        OP_LOGE("ChunkGatedDeltaRuleComputeWy",
                "infershape null shape: q=%p v=%p qKernel=%p kKernel=%p wKernel=%p "
                "uKernel=%p gKernel=%p",
                qShape, vShape, qKernelShape, kKernelShape, wKernelShape, uKernelShape, gKernelShape);
        return ge::GRAPH_FAILED;
    }
    if (qShape->GetDimNum() != QKV_DIM_NUM || vShape->GetDimNum() != QKV_DIM_NUM) {
        OP_LOGE("ChunkGatedDeltaRuleComputeWy", "infershape invalid dim num: q=%zu v=%zu, both must be 4",
                qShape->GetDimNum(), vShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t b = qShape->GetDim(DIM_B);
    const int64_t t = qShape->GetDim(DIM_T);
    const int64_t hk = qShape->GetDim(DIM_H);
    const int64_t kDim = qShape->GetDim(DIM_D);
    const int64_t hv = vShape->GetDim(DIM_H);
    const int64_t vDim = vShape->GetDim(DIM_D);

    // q/k: [B, T, Hk, K] -> [B, Hk, T, K]
    SetBhtd(qKernelShape, b, hk, t, kDim);
    SetBhtd(kKernelShape, b, hk, t, kDim);
    // w is expanded over the value heads and keeps the key head dim: [B, Hv, T, K]
    SetBhtd(wKernelShape, b, hv, t, kDim);
    // u carries the value head dim: [B, Hv, T, V]
    SetBhtd(uKernelShape, b, hv, t, vDim);
    // g holds the per-token intra-chunk cumsum: [B, Hv, T]
    gKernelShape->SetDimNum(3);
    gKernelShape->SetDim(0, b);
    gKernelShape->SetDim(1, hv);
    gKernelShape->SetDim(2, t);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkGatedDeltaRuleComputeWy(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto qDtype = context->GetInputDataType(Q_INDEX);
    const auto vDtype = context->GetInputDataType(V_INDEX);
    context->SetOutputDataType(Q_KERNEL_INDEX, qDtype);
    context->SetOutputDataType(K_KERNEL_INDEX, qDtype);
    context->SetOutputDataType(W_KERNEL_INDEX, qDtype);
    context->SetOutputDataType(U_KERNEL_INDEX, vDtype);
    // g_kernel is the fp32 cumsum of the fp32 gate input.
    context->SetOutputDataType(G_KERNEL_INDEX, context->GetInputDataType(G_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRuleComputeWy)
    .InferShape(InferShapeChunkGatedDeltaRuleComputeWy)
    .InferDataType(InferDataTypeChunkGatedDeltaRuleComputeWy);

} // namespace ops
