// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef DUAL_LEVEL_GROUPED_MATMUL_CHECKER_H_
#define DUAL_LEVEL_GROUPED_MATMUL_CHECKER_H_
#include "aclnn_dual_level_grouped_matmul_weight_nz.h"
namespace dual_level_gmm {
inline constexpr int64_t LEVEL0_GROUP_SIZE = 512;
inline constexpr int64_t LEVEL1_GROUP_SIZE = 32;

struct DualLevelGroupedMatmulParams {
    const aclTensorList* x;
    const aclTensorList* weight;
    const aclTensorList* biasOptional;
    const aclTensorList* scaleOptional;
    const aclTensorList* perTokenScaleOptional;
    const aclTensor* groupListOptional;
    const aclTensorList* xLevel0Scale;
    const aclTensorList* weightLevel0Scale;
    int64_t level0GroupSize;
    int64_t level1GroupSize;
    int64_t splitItem;
    int64_t groupType;
    int64_t groupListType;
    const aclIntArray* tuningConfigOptional;
    aclTensorList* out;
};
aclnnStatus CheckParams(const DualLevelGroupedMatmulParams& params, bool& transposeWeight);
} // namespace dual_level_gmm
#endif
