// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef LEVEL0_DUAL_LEVEL_GROUPED_MATMUL_H_
#define LEVEL0_DUAL_LEVEL_GROUPED_MATMUL_H_
#include "opdev/op_executor.h"
namespace l0op {
aclnnStatus DualLevelGroupedMatmul(const aclTensorList* x, const aclTensorList* weight,
                                   const aclTensorList* biasOptional, const aclTensorList* scaleOptional,
                                   const aclTensorList* perTokenScaleOptional, const aclTensor* groupListOptional,
                                   const aclTensorList* xLevel0Scale, const aclTensorList* weightLevel0Scale,
                                   bool transposeWeight, int64_t groupListType, const aclIntArray* tuningConfigOptional,
                                   int64_t level0GroupSize, int64_t level1GroupSize, aclTensorList* out,
                                   aclOpExecutor* executor);
} // namespace l0op
#endif
