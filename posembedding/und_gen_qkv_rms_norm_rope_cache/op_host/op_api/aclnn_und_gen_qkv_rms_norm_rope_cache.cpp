/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_und_gen_qkv_rms_norm_rope_cache.cpp
 * \brief
 */

#include "aclnn_und_gen_qkv_rms_norm_rope_cache.h"
#include "und_gen_qkv_rms_norm_rope_cache.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace und_gen_qkv_rms_norm_rope_cache {

static inline bool CheckNotNull(const aclTensor *undQkv, const aclTensor *undWeightsQ, const aclTensor *undWeightsK,
                                const aclTensor *cosSinCache, aclTensor *kCacheRef, aclTensor *vCacheRef,
                                const aclTensor *slotMapping, const aclTensor *positions, aclTensor *qOut)
{
    OP_CHECK_NULL(undQkv, return false);
    OP_CHECK_NULL(undWeightsQ, return false);
    OP_CHECK_NULL(undWeightsK, return false);
    OP_CHECK_NULL(cosSinCache, return false);
    OP_CHECK_NULL(kCacheRef, return false);
    OP_CHECK_NULL(vCacheRef, return false);
    OP_CHECK_NULL(slotMapping, return false);
    OP_CHECK_NULL(positions, return false);
    OP_CHECK_NULL(qOut, return false);
    return true;
}

// k_cache/v_cache 是调用方预分配、算子原地写入的缓冲区，不能走 Contiguous：
// 非连续时 Contiguous 产出的是副本，kernel 写进副本后调用方的 cache 不会被更新且无任何报错。
// 本算子只支持连续 BBND 布局，这里直接把非连续输入拒掉。
static inline bool CheckCacheContiguous(const aclTensor *kCacheRef, const aclTensor *vCacheRef)
{
    if (!IsContiguous(kCacheRef) || !IsContiguous(vCacheRef)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kCacheRef/vCacheRef must be contiguous [Bn, Bs, N, D] tensors, "
                                         "non-contiguous KV Cache is not supported.");
        return false;
    }
    return true;
}

static inline bool CheckGenPaired(const aclTensor *genQkvOptional, const aclTensor *genWeightsQOptional,
                                  const aclTensor *genWeightsKOptional)
{
    if (genQkvOptional == nullptr) {
        return true;
    }
    if (genWeightsQOptional == nullptr || genWeightsKOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "genWeightsQOptional/genWeightsKOptional are required when genQkvOptional is provided.");
        return false;
    }
    return true;
}
} // namespace und_gen_qkv_rms_norm_rope_cache

aclnnStatus aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor *undQkv, const aclTensor *undWeightsQ, const aclTensor *undWeightsK, const aclTensor *cosSinCache,
    aclTensor *kCacheRef, aclTensor *vCacheRef, const aclTensor *slotMapping, const aclTensor *positions,
    const aclTensor *genQkvOptional, const aclTensor *genWeightsQOptional, const aclTensor *genWeightsKOptional,
    const aclTensor *catIndicesOptional, int64_t numHeadsQ, int64_t numHeadsK, int64_t numHeadsV, double normEps,
    const aclIntArray *mropeSection, aclTensor *qOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnUndGenQkvRmsNormRopeCache,
                   DFX_IN(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCacheRef, vCacheRef, slotMapping, positions,
                          genQkvOptional, genWeightsQOptional, genWeightsKOptional, catIndicesOptional, numHeadsQ,
                          numHeadsK, numHeadsV, normEps, mropeSection),
                   DFX_OUT(qOut, kCacheRef, vCacheRef));

    // 参数检查：L2 只做空指针与 KV Cache 连续性校验（后者 tiling 侧看不到 view stride，只能在这里拦），
    // dtype/shape/属性取值的完整校验统一在 tiling 侧
    // （aclnn 单算子路径与图模式都会走 tiling，避免两处重复实现导致漂移）
    auto notNull = und_gen_qkv_rms_norm_rope_cache::CheckNotNull(undQkv, undWeightsQ, undWeightsK, cosSinCache,
                                                                 kCacheRef, vCacheRef, slotMapping, positions, qOut);
    CHECK_RET(notNull, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(und_gen_qkv_rms_norm_rope_cache::CheckGenPaired(genQkvOptional, genWeightsQOptional, genWeightsKOptional),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(und_gen_qkv_rms_norm_rope_cache::CheckCacheContiguous(kCacheRef, vCacheRef), ACLNN_ERR_PARAM_INVALID);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 将输入转换成连续的tensor
    auto undQkvContiguous = l0op::Contiguous(undQkv, uniqueExecutor.get());
    CHECK_RET(undQkvContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto undWeightsQContiguous = l0op::Contiguous(undWeightsQ, uniqueExecutor.get());
    CHECK_RET(undWeightsQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto undWeightsKContiguous = l0op::Contiguous(undWeightsK, uniqueExecutor.get());
    CHECK_RET(undWeightsKContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cosSinCacheContiguous = l0op::Contiguous(cosSinCache, uniqueExecutor.get());
    CHECK_RET(cosSinCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto positionsContiguous = l0op::Contiguous(positions, uniqueExecutor.get());
    CHECK_RET(positionsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 可选输入：为空时直接透传 nullptr
    const aclTensor *genQkvContiguous = nullptr;
    const aclTensor *genWeightsQContiguous = nullptr;
    const aclTensor *genWeightsKContiguous = nullptr;
    const aclTensor *catIndicesContiguous = nullptr;
    if (genQkvOptional != nullptr) {
        genQkvContiguous = l0op::Contiguous(genQkvOptional, uniqueExecutor.get());
        CHECK_RET(genQkvContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        genWeightsQContiguous = l0op::Contiguous(genWeightsQOptional, uniqueExecutor.get());
        CHECK_RET(genWeightsQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        genWeightsKContiguous = l0op::Contiguous(genWeightsKOptional, uniqueExecutor.get());
        CHECK_RET(genWeightsKContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (catIndicesOptional != nullptr) {
        catIndicesContiguous = l0op::Contiguous(catIndicesOptional, uniqueExecutor.get());
        CHECK_RET(catIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // KV Cache 为原地更新的输入输出：必须直接用调用方传入的 tensor（上面已校验连续），
    // 一旦走 Contiguous 拿到的就是副本，kernel 的原地写会落到副本上而调用方看不到结果。

    // 调用l0接口进行计算
    auto l0Ret = l0op::UndGenQkvRmsNormRopeCache(
        undQkvContiguous, undWeightsQContiguous, undWeightsKContiguous, cosSinCacheContiguous, kCacheRef, vCacheRef,
        slotMappingContiguous, positionsContiguous, genQkvContiguous, genWeightsQContiguous, genWeightsKContiguous,
        catIndicesContiguous, numHeadsQ, numHeadsK, numHeadsV, normEps, mropeSection, qOut, uniqueExecutor.get());
    CHECK_RET(l0Ret == ACLNN_SUCCESS, l0Ret);

    // 获取workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUndGenQkvRmsNormRopeCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUndGenQkvRmsNormRopeCache);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
