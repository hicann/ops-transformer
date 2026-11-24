// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "dual_level_grouped_matmul_tiling.h"

namespace ge {
using optiling::DualLevelAttrIndex;
using optiling::DualLevelInputIndex;
using DualLevelGroupedMatmul::WEIGHT_NK_AXIS;
static graphStatus InferDualLevelGroupedMatmulShape(gert::InferShapeContext* ctx)
{
    const auto* x = ctx->GetInputShape(DualLevelInputIndex::X);
    const auto* w = ctx->GetInputShape(DualLevelInputIndex::WEIGHT);
    auto* y = ctx->GetOutputShape(0);
    const auto* attrs = ctx->GetAttrs();
    if (x == nullptr || w == nullptr || y == nullptr || attrs == nullptr || x->GetDimNum() != optiling::MATRIX_RANK ||
        w->GetDimNum() != optiling::EXPERT_WEIGHT_RANK) {
        return GRAPH_FAILED;
    }
    const auto* trans = attrs->GetAttrPointer<bool>(DualLevelAttrIndex::TRANSPOSE_WEIGHT);
    if (trans == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t wk = w->GetDim(*trans ? WEIGHT_NK_AXIS : 1);
    const int64_t xk = x->GetDim(1);
    if (wk >= 0 && xk >= 0 && wk != xk) {
        return GRAPH_FAILED;
    }
    *y = gert::Shape({x->GetDim(0), w->GetDim(*trans ? 1 : WEIGHT_NK_AXIS)});
    return GRAPH_SUCCESS;
}
static graphStatus InferDualLevelGroupedMatmulType(gert::InferDataTypeContext* ctx)
{
    const auto* attrs = ctx->GetAttrs();
    const auto* dtype = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(DualLevelAttrIndex::OUTPUT_DTYPE);
    if (dtype == nullptr || (*dtype != DT_FLOAT16 && *dtype != DT_BF16)) {
        return GRAPH_FAILED;
    }
    ctx->SetOutputDataType(0, static_cast<DataType>(*dtype));
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DualLevelGroupedMatmul)
    .InferShape(InferDualLevelGroupedMatmulShape)
    .InferDataType(InferDualLevelGroupedMatmulType);
} // namespace ge
