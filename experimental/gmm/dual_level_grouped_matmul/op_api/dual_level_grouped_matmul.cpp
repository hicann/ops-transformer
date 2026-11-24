// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "dual_level_grouped_matmul.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "util/math_util.h"
using namespace op;
namespace {
constexpr int64_t MX_SCALE_PAIR = 2;
constexpr size_t WEIGHT_N_AXIS = 2;
// Only the two GMM views are accepted. Reinterpret their storage without moving device data.
aclTensor* Normalize(const aclTensor* tensor, const Shape& shape, const Strides& strides, aclOpExecutor* executor)
{
    auto* view = executor->CreateView(tensor, shape, 0);
    if (view != nullptr) {
        view->SetOriginalShape(shape);
        view->SetStorageShape(tensor->GetStorageShape());
        view->SetStorageFormat(tensor->GetStorageFormat());
        view->SetViewStrides(strides);
    }
    return view;
}
} // namespace
namespace l0op {
OP_TYPE_REGISTER(DualLevelGroupedMatmul);
aclnnStatus DualLevelGroupedMatmul(const aclTensorList* x, const aclTensorList* weight,
                                   const aclTensorList* biasOptional, const aclTensorList* scaleOptional,
                                   const aclTensorList* perTokenScaleOptional, const aclTensor* groupListOptional,
                                   const aclTensorList* xLevel0Scale, const aclTensorList* weightLevel0Scale,
                                   bool transposeWeight, int64_t groupListType, const aclIntArray* tuningConfigOptional,
                                   int64_t level0GroupSize, int64_t level1GroupSize, aclTensorList* out,
                                   aclOpExecutor* executor)
{
    L0_DFX(DualLevelGroupedMatmul, x, weight, biasOptional, scaleOptional, perTokenScaleOptional, groupListOptional,
           xLevel0Scale, weightLevel0Scale, transposeWeight, groupListType, level0GroupSize, level1GroupSize, out);
    const auto* a = (*x)[0];
    const auto* w = (*weight)[0];
    const auto* a0 = (*xLevel0Scale)[0];
    const auto* b0 = (*weightLevel0Scale)[0];
    const auto* a1 = (*perTokenScaleOptional)[0];
    const auto* b1 = (*scaleOptional)[0];
    const auto* bias = biasOptional == nullptr ? nullptr : (*biasOptional)[0];
    auto* y = (*out)[0];
    const auto dtype = y->GetDataType();
    const auto& shape = w->GetViewShape();
    const auto e = shape.GetDim(0);
    const auto k = shape.GetDim(1);
    const auto n = shape.GetDim(WEIGHT_N_AXIS);
    const auto mxScaleGroups = Ops::Base::CeilDiv(k, level1GroupSize * MX_SCALE_PAIR);
    const bool transpose = transposeWeight;
    const int64_t weightRows = transpose ? n : k;
    const int64_t weightCols = transpose ? k : n;
    auto* weightView =
        Normalize(w, Shape{e, weightRows, weightCols}, Strides{weightRows * weightCols, weightCols, 1}, executor);
    const auto scaleRows = transpose ? n : mxScaleGroups;
    const auto scaleCols = transpose ? mxScaleGroups : n;
    auto* scaleView =
        Normalize(b1, Shape{e, scaleRows, scaleCols, MX_SCALE_PAIR},
                  Strides{n * mxScaleGroups * MX_SCALE_PAIR, scaleCols * MX_SCALE_PAIR, MX_SCALE_PAIR, 1}, executor);
    CHECK_RET(weightView && scaleView, ACLNN_ERR_INNER_NULLPTR);
    const int64_t defaultTuning = 0;
    auto* config = tuningConfigOptional == nullptr ? executor->AllocIntArray(&defaultTuning, 1) : tuningConfigOptional;
    CHECK_RET(config != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ADD_TO_LAUNCHER_LIST_AICORE(
        DualLevelGroupedMatmul, OP_INPUT(a, a0, a1, weightView, b0, scaleView, groupListOptional, bias), OP_OUTPUT(y),
        OP_ATTR(transpose, groupListType, config, static_cast<int64_t>(dtype), level0GroupSize, level1GroupSize));
}
} // namespace l0op
