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
 * \file aclnn_stem_oam_prep_paged_kv.cpp
 * \brief
 */

#include "aclnn_stem_oam_prep_paged_kv.h"
#include "stem_oam_prep_paged_kv.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> KV_CACHE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> KV_SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

constexpr int64_t KV_BLOCK_SIZE_64 = 64;
constexpr int64_t KV_BLOCK_SIZE_128 = 128;
constexpr int64_t STEM_BLOCK_SIZE_ALIGN = 32;
constexpr int64_t STEM_BLOCK_SIZE_MAX = 256;
constexpr int64_t STEM_STRIDE_ALIGN = 16;
constexpr int64_t STEM_STRIDE_MAX = 64;
constexpr int64_t CACHE_LAYOUT_A = 0;
constexpr int64_t CACHE_LAYOUT_B = 1;
constexpr size_t KV_CACHE_DIM_NUM = 4;
constexpr size_t SHAPE_DIM_0 = 0;
constexpr size_t SHAPE_DIM_1 = 1;
constexpr size_t SHAPE_DIM_2 = 2;

inline static bool CheckNull(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices,
                             const aclTensor *kScaleCache, const aclTensor *vScale, const aclTensor *kFlat,
                             const aclTensor *vBias)
{
    OP_CHECK_NULL(kCache, return false);
    OP_CHECK_NULL(vCache, return false);
    OP_CHECK_NULL(kvIndices, return false);
    OP_CHECK_NULL(kScaleCache, return false);
    OP_CHECK_NULL(vScale, return false);
    OP_CHECK_NULL(kFlat, return false);
    OP_CHECK_NULL(vBias, return false);
    return true;
}

inline static bool CheckEmpty(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices,
                              const aclTensor *kScaleCache, const aclTensor *vScale)
{
    if (kCache->IsEmpty() || vCache->IsEmpty() || kvIndices->IsEmpty() || kScaleCache->IsEmpty() || vScale->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnStemOamPrepPagedKv does not support empty tensor!");
        return false;
    }
    return true;
}

inline static bool CheckDtype(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kScaleCache,
                              const aclTensor *vScale)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(kCache, KV_CACHE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(vCache, KV_CACHE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(kScaleCache, KV_SCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(vScale, KV_SCALE_DTYPE_SUPPORT_LIST, return false);
    return true;
}

inline static bool CheckShape(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kScaleCache,
                              const aclTensor *vScale, int64_t cacheLayout, int64_t kvBlockSize)
{
    auto kCacheShape = kCache->GetViewShape();
    auto vCacheShape = vCache->GetViewShape();
    auto kScaleCacheShape = kScaleCache->GetViewShape();
    if (kCacheShape.GetDimNum() != KV_CACHE_DIM_NUM || vCacheShape.GetDimNum() != KV_CACHE_DIM_NUM ||
        kScaleCacheShape.GetDimNum() != KV_CACHE_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "kCache/vCache/kScaleCache must be 4D, got kCache dimNum=%zu, "
                "vCache dimNum=%zu, kScaleCache dimNum=%zu.",
                kCacheShape.GetDimNum(), vCacheShape.GetDimNum(), kScaleCacheShape.GetDimNum());
        return false;
    }
    if (cacheLayout == CACHE_LAYOUT_A &&
        (kScaleCacheShape.GetDim(SHAPE_DIM_1) != kvBlockSize ||
         kScaleCacheShape.GetDim(SHAPE_DIM_2) != vScale->GetViewShape().GetDim(SHAPE_DIM_0))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "cacheLayout=0 requires kScaleCacheShape[1]=kvBlockSize(%ld) and "
                "kScaleCacheShape[2]=H_kv(%ld), got kScaleCacheShape[1]=%ld, kScaleCacheShape[2]=%ld.",
                kvBlockSize, vScale->GetViewShape().GetDim(SHAPE_DIM_0), kScaleCacheShape.GetDim(SHAPE_DIM_1),
                kScaleCacheShape.GetDim(SHAPE_DIM_2));
        return false;
    } else if (cacheLayout == CACHE_LAYOUT_B &&
               (kScaleCacheShape.GetDim(SHAPE_DIM_2) != kvBlockSize ||
                kScaleCacheShape.GetDim(SHAPE_DIM_1) != vScale->GetViewShape().GetDim(SHAPE_DIM_0))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "cacheLayout=1 requires kScaleCacheShape[2]=kvBlockSize(%ld) and "
                "kScaleCacheShape[1]=H_kv(%ld), got kScaleCacheShape[1]=%ld, kScaleCacheShape[2]=%ld.",
                kvBlockSize, vScale->GetViewShape().GetDim(SHAPE_DIM_0), kScaleCacheShape.GetDim(SHAPE_DIM_1),
                kScaleCacheShape.GetDim(SHAPE_DIM_2));
        return false;
    }
    return true;
}

inline static bool CheckAttr(int64_t stemBlockSize, int64_t stemStride)
{
    if (stemBlockSize % STEM_BLOCK_SIZE_ALIGN != 0 || stemBlockSize > STEM_BLOCK_SIZE_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stemBlockSize must be multiple of 32 and <=256, got %ld.", stemBlockSize);
        return false;
    }
    if (stemStride % STEM_STRIDE_ALIGN != 0 || stemStride > STEM_STRIDE_MAX || stemStride > stemBlockSize ||
        stemBlockSize % stemStride != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "stemStride must be multiple of 16, <=64, <=stemBlockSize(%ld), "
                "and stemBlockSize must be a multiple of stemStride, got %ld.",
                stemBlockSize, stemStride);
        return false;
    }
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices,
                                      const aclTensor *kScaleCache, const aclTensor *vScale, int64_t cacheLayout,
                                      int64_t kvBlockSize, int64_t stemBlockSize, int64_t stemStride,
                                      const aclTensor *kFlat, const aclTensor *vBias)
{
    CHECK_RET(CheckNull(kCache, vCache, kvIndices, kScaleCache, vScale, kFlat, vBias), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckEmpty(kCache, vCache, kvIndices, kScaleCache, vScale), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(kCache, vCache, kScaleCache, vScale), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(kCache, vCache, kScaleCache, vScale, cacheLayout, kvBlockSize), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttr(stemBlockSize, stemStride), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStemOamPrepPagedKvGetWorkspaceSize(const aclTensor *kCache, const aclTensor *vCache,
                                                    const aclTensor *kvIndices, const aclIntArray *kvSeqLens,
                                                    const aclTensor *kScaleCache, const aclTensor *vScale,
                                                    double lambdaMag, int64_t cacheLayout, int64_t kvBlockSize,
                                                    int64_t stemBlockSize, int64_t stemStride, const aclTensor *kFlat,
                                                    const aclTensor *vBias, uint64_t *workspaceSize,
                                                    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnStemOamPrepPagedKv,
                   DFX_IN(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale, lambdaMag, cacheLayout,
                          kvBlockSize, stemBlockSize, stemStride),
                   DFX_OUT(kFlat, vBias));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(kCache, vCache, kvIndices, kScaleCache, vScale, cacheLayout, kvBlockSize, stemBlockSize,
                           stemStride, kFlat, vBias);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto kCacheView = IsContiguous(kCache) ?
                          kCache :
                          uniqueExecutor->CreateView(kCache, kCache->GetViewShape(), kCache->GetStorageShape(),
                                                     kCache->GetViewStrides(), kCache->GetViewOffset());
    auto vCacheView = IsContiguous(vCache) ?
                          vCache :
                          uniqueExecutor->CreateView(vCache, vCache->GetViewShape(), vCache->GetStorageShape(),
                                                     vCache->GetViewStrides(), vCache->GetViewOffset());
    auto kScaleCacheView =
        IsContiguous(kScaleCache) ?
            kScaleCache :
            uniqueExecutor->CreateView(kScaleCache, kScaleCache->GetViewShape(), kScaleCache->GetStorageShape(),
                                       kScaleCache->GetViewStrides(), kScaleCache->GetViewOffset());

    auto kvIndicesContiguous = l0op::Contiguous(kvIndices, uniqueExecutor.get());
    const aclTensor *kvSeqLensTensor = uniqueExecutor->ConvertToTensor(kvSeqLens, DataType::DT_INT32);
    OP_CHECK_NULL(kvSeqLensTensor, return ACLNN_ERR_PARAM_NULLPTR);
    auto vScaleContiguous = l0op::Contiguous(vScale, uniqueExecutor.get());
    float lambdaMagFloat = static_cast<float>(lambdaMag);

    auto result = l0op::StemOamPrepPagedKv(kCacheView, vCacheView, kvIndicesContiguous, kvSeqLensTensor,
                                           kScaleCacheView, vScaleContiguous, lambdaMagFloat, cacheLayout, kvBlockSize,
                                           stemBlockSize, stemStride, kFlat, vBias, uniqueExecutor.get());
    CHECK_RET(std::get<0>(result) != nullptr && std::get<1>(result) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStemOamPrepPagedKv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStemOamPrepPagedKv);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
