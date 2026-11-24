// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef ACLNN_DUAL_LEVEL_GROUPED_MATMUL_WEIGHT_NZ_H_
#define ACLNN_DUAL_LEVEL_GROUPED_MATMUL_WEIGHT_NZ_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Prepare split-M MXFP4 E2M1 grouped matmul on Ascend 950.
 * @param x Required one-element TensorList: E2M1 ND (M,K), M>0 and even K>2.
 * @param weight Required one-element TensorList: E2M1 FRACTAL_NZ, logical view (E,K,N).
 *        Canonical contiguous or transposed strides select the physical layout; N>1.
 * @param biasOptional Optional one-element TensorList: FP32 ND (E,N); nullptr omits bias.
 * @param scaleOptional Required one-element TensorList: E8M0 logical (E,ceil(K/64),N,2).
 *        Its physical layout must follow weight's transpose mode.
 * @param perTokenScaleOptional Required one-element TensorList: E8M0 ND (M,ceil(K/64),2).
 * @param groupListOptional Required INT64 ND: (E,) for types 0/1, (E,2) for type 2.
 *        Type 0 holds cumulative end rows, type 1 row counts, type 2 [expert_id,row_count].
 *        Row counts sum to M. For type 2, nonzero groups precede zero groups, with ascending expert IDs within each.
 * @param xLevel0Scale Required one-element TensorList: FP32 ND (M,ceil(K/512)).
 * @param weightLevel0Scale Required one-element TensorList: FP32 ND (E,ceil(K/512),N).
 * @param level0GroupSize Number of K elements per FP32 scale group; must be 512.
 * @param level1GroupSize Number of K elements per E8M0 scale group; must be 32.
 * @param splitItem Must be 2 or 3; both use one input/output tensor in this interface.
 * @param groupType Split axis; must be 0 (M).
 * @param groupListType Group encoding: 0, 1 or 2.
 * @param tuningConfigOptional nullptr or the single-element array {0}.
 * @param out Required one-element TensorList: FP16/BF16 ND (M,N).
 * @param workspaceSize Non-null output for required workspace bytes.
 * @param executor Non-null output for the prepared execution plan.
 * @retval ACLNN_SUCCESS Preparation succeeded; device execution is performed by phase 2.
 * @retval ACLNN_ERR_PARAM_NULLPTR A required pointer or TensorList element is null.
 * @retval ACLNN_ERR_PARAM_INVALID An attribute, shape, dtype, format, stride or offset is unsupported.
 * @note Deterministic by default for identical inputs and attributes in the same environment.
 *       No additional determinism switch is required.
 * @note ND inputs/output must be dense with zero view offset. Weight and transposed scale
 *       also require zero view offset. Device grouplist content checks do not return
 *       synchronous ACLNN errors. Partial sums and level0 accumulation use FP32; bias is last.
 *       ND denotes the actual device storage layout, not only logical contiguous strides.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* biasOptional,
    const aclTensorList* scaleOptional, const aclTensorList* perTokenScaleOptional, const aclTensor* groupListOptional,
    const aclTensorList* xLevel0Scale, const aclTensorList* weightLevel0Scale, int64_t level0GroupSize,
    int64_t level1GroupSize, int64_t splitItem, int64_t groupType, int64_t groupListType,
    aclIntArray* tuningConfigOptional, aclTensorList* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Execute a plan prepared by aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize.
 * @param workspace Device workspace; may be nullptr when workspaceSize is zero.
 * @param workspaceSize Available workspace size in bytes.
 * @param executor Prepared plan returned by phase 1.
 * @param stream Target ACL runtime stream.
 * @return ACLNN execution status. Stream synchronization is the caller's responsibility.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDualLevelGroupedMatmulWeightNz(void* workspace,
                                                                                       uint64_t workspaceSize,
                                                                                       aclOpExecutor* executor,
                                                                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
