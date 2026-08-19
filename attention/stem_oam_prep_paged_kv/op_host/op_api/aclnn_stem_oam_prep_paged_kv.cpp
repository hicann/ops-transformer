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
constexpr int64_t KV_LAYOUT_BBND = 0;
constexpr int64_t KV_LAYOUT_BNBD = 1;
constexpr int64_t HKVMAX = 8;
constexpr int64_t BATCHMAX = 16;
constexpr int64_t KVBLOCKSIZEONE = 64;
constexpr int64_t KVBLOCKSIZETWO = 128;
constexpr size_t KV_CACHE_DIM_NUM = 4;
constexpr size_t SHAPE_DIM_0 = 0;
constexpr size_t SHAPE_DIM_1 = 1;
constexpr size_t SHAPE_DIM_2 = 2;
constexpr size_t SHAPE_DIM_3 = 3;

inline static bool CheckNull(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices,
                             const aclTensor *kScaleCacheOptional, const aclTensor *vScaleOptional,
                             const aclTensor *kFlat, const aclTensor *vBias)
{
    OP_CHECK_NULL(kCache, return false);
    OP_CHECK_NULL(vCache, return false);
    OP_CHECK_NULL(kvIndices, return false);
    OP_CHECK_NULL(kFlat, return false);
    OP_CHECK_NULL(vBias, return false);
    if (kCache->GetDataType() == DataType::DT_FLOAT8_E4M3FN) {
        OP_CHECK_NULL(kScaleCacheOptional, return false);
        OP_CHECK_NULL(vScaleOptional, return false);
    }
    return true;
}

inline static bool CheckEmpty(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices,
                              const aclTensor *kScaleCacheOptional, const aclTensor *vScaleOptional)
{
    if (kCache->IsEmpty() || vCache->IsEmpty() || kvIndices->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnStemOamPrepPagedKv does not support empty tensor!");
        return false;
    }
    if (kScaleCacheOptional != nullptr && kScaleCacheOptional->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kScaleCache does not support empty tensor!");
        return false;
    }
    if (vScaleOptional != nullptr && vScaleOptional->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "vScale does not support empty tensor!");
        return false;
    }
    return true;
}

inline static bool CheckCacheContinuous(const aclTensor *shape)
{
    auto strideBuf = shape->GetViewStrides();
    auto strideSize = strideBuf.size();
    if (strideBuf[strideSize - 1] != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The stride of dimension -1 of A must be 1!");
        return false;
    }

    return true;
}

inline static bool CheckDtype(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kScaleCacheOptional,
                              const aclTensor *vScaleOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(kCache, KV_CACHE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(vCache, KV_CACHE_DTYPE_SUPPORT_LIST, return false);
    if (kScaleCacheOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(kScaleCacheOptional, KV_SCALE_DTYPE_SUPPORT_LIST, return false);
    }
    if (vScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(vScaleOptional, KV_SCALE_DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

inline static bool CheckShape(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kScaleCacheOptional,
                              const aclTensor *vScaleOptional, const aclTensor *kvIndices, const aclIntArray *kvSeqLens,
                              int64_t kvLayout)
{
    auto kCacheShape = kCache->GetViewShape();
    auto vCacheShape = vCache->GetViewShape();
    auto kvIndicesShape = kvIndices->GetViewShape();
    auto kScaleCacheOptionalShape = kScaleCacheOptional->GetViewShape();

    if (!CheckCacheContinuous(kCache) || !CheckCacheContinuous(vCache)) {
        return false;
    }

    if (kCache->GetDataType() == DataType::DT_FLOAT8_E4M3FN) {
        if ((kCacheShape.GetDim(SHAPE_DIM_0) != kScaleCacheOptionalShape.GetDim(SHAPE_DIM_0)) ||
            (kCacheShape.GetDim(SHAPE_DIM_1) != kScaleCacheOptionalShape.GetDim(SHAPE_DIM_1)) ||
            (kCacheShape.GetDim(SHAPE_DIM_2) != kScaleCacheOptionalShape.GetDim(SHAPE_DIM_2))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In the DT_FLOAT8_E4M3FN scenario, the first three dimensions of kCache "
                                             "and kScaleCacheOptional must be the same.");
            return false;
        }
        if (!CheckCacheContinuous(kScaleCacheOptional)) {
            return false;
        }
    }

    if ((kCacheShape.GetDim(SHAPE_DIM_0) != vCacheShape.GetDim(SHAPE_DIM_0)) ||
        (kCacheShape.GetDim(SHAPE_DIM_1) != vCacheShape.GetDim(SHAPE_DIM_1)) ||
        (kCacheShape.GetDim(SHAPE_DIM_2) != vCacheShape.GetDim(SHAPE_DIM_2)) ||
        (kCacheShape.GetDim(SHAPE_DIM_3) != vCacheShape.GetDim(SHAPE_DIM_3))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimensions of kCache and vCache must be the same.");
        return false;
    }

    if (kCacheShape.GetDimNum() != KV_CACHE_DIM_NUM || vCacheShape.GetDimNum() != KV_CACHE_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kCache/vCache must be 4D, got kCache dimNum=%zu, vCache dimNum=%zu.",
                kCacheShape.GetDimNum(), vCacheShape.GetDimNum());
        return false;
    }
    if (kScaleCacheOptional == nullptr || vScaleOptional == nullptr) {
        return true;
    }
    auto kScaleCacheShape = kScaleCacheOptional->GetViewShape();
    if (kScaleCacheShape.GetDimNum() != KV_CACHE_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kScaleCache must be 4D, got dimNum=%zu.", kScaleCacheShape.GetDimNum());
        return false;
    }
    int64_t kvBlockSize =
        (kvLayout == KV_LAYOUT_BBND) ? kCacheShape.GetDim(SHAPE_DIM_1) : kCacheShape.GetDim(SHAPE_DIM_2);
    int64_t hkv = (kvLayout == KV_LAYOUT_BBND) ? kCacheShape.GetDim(SHAPE_DIM_2) : kCacheShape.GetDim(SHAPE_DIM_1);
    int64_t batch = kvIndicesShape.GetDim(SHAPE_DIM_0);

    if (batch != kvSeqLens->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvIndicesShape[0] and kvSeqLensShape[0] are not equal.");
        return false;
    }

    if ((hkv > HKVMAX) || (batch > BATCHMAX)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "requires hkv less than 8 and batch less than 16"
                "got hkv=%ld, batch=%ld.",
                hkv, batch);
        return false;
    }

    if ((kvBlockSize != KVBLOCKSIZEONE) && (kvBlockSize != KVBLOCKSIZETWO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvBlockSize must be 64 or 128.");
        return false;
    }

    if (kvLayout == KV_LAYOUT_BBND &&
        (kScaleCacheShape.GetDim(SHAPE_DIM_1) != kvBlockSize ||
         kScaleCacheShape.GetDim(SHAPE_DIM_2) != vScaleOptional->GetViewShape().GetDim(SHAPE_DIM_0))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "kvLayout=BBND requires kScaleCacheShape[1]=kvBlockSize(%ld) and "
                "kScaleCacheShape[2]=H_kv(%ld), got kScaleCacheShape[1]=%ld, kScaleCacheShape[2]=%ld.",
                kvBlockSize, vScaleOptional->GetViewShape().GetDim(SHAPE_DIM_0), kScaleCacheShape.GetDim(SHAPE_DIM_1),
                kScaleCacheShape.GetDim(SHAPE_DIM_2));
        return false;
    } else if (kvLayout == KV_LAYOUT_BNBD &&
               (kScaleCacheShape.GetDim(SHAPE_DIM_2) != kvBlockSize ||
                kScaleCacheShape.GetDim(SHAPE_DIM_1) != vScaleOptional->GetViewShape().GetDim(SHAPE_DIM_0))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "kvLayout=BNBD requires kScaleCacheShape[2]=kvBlockSize(%ld) and "
                "kScaleCacheShape[1]=H_kv(%ld), got kScaleCacheShape[1]=%ld, kScaleCacheShape[2]=%ld.",
                kvBlockSize, vScaleOptional->GetViewShape().GetDim(SHAPE_DIM_0), kScaleCacheShape.GetDim(SHAPE_DIM_1),
                kScaleCacheShape.GetDim(SHAPE_DIM_2));
        return false;
    }
    return true;
}

inline static bool CheckAttr(int64_t stemBlockSize, int64_t stemStride, double lambdaMag)
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
    if ((lambdaMag < 0) || (lambdaMag > 1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "lambdaMag must be greater than 0 ans less than or equal to 1, "
                "got lambdaMag=%f",
                lambdaMag);
        return false;
    }

    return true;
}

inline static aclnnStatus CheckParams(const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices,
                                      const aclTensor *kScaleCacheOptional, const aclTensor *vScaleOptional,
                                      const aclIntArray *kvSeqLens, int64_t kvLayout, int64_t stemBlockSize,
                                      int64_t stemStride, const aclTensor *kFlat, const aclTensor *vBias,
                                      double lambdaMag)
{
    CHECK_RET(CheckNull(kCache, vCache, kvIndices, kScaleCacheOptional, vScaleOptional, kFlat, vBias),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckEmpty(kCache, vCache, kvIndices, kScaleCacheOptional, vScaleOptional), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(kCache, vCache, kScaleCacheOptional, vScaleOptional), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(kCache, vCache, kScaleCacheOptional, vScaleOptional, kvIndices, kvSeqLens, kvLayout),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttr(stemBlockSize, stemStride, lambdaMag), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStemOamPrepPagedKvGetWorkspaceSize(const aclTensor *kCache, const aclTensor *vCache,
                                                    const aclTensor *kvIndices, const aclIntArray *kvSeqLens,
                                                    const aclTensor *kScaleCacheOptional,
                                                    const aclTensor *vScaleOptional, double lambdaMag, char *kvLayout,
                                                    int64_t stemBlockSize, int64_t stemStride, const aclTensor *kFlat,
                                                    const aclTensor *vBias, uint64_t *workspaceSize,
                                                    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnStemOamPrepPagedKv,
                   DFX_IN(kCache, vCache, kvIndices, kvSeqLens, kScaleCacheOptional, vScaleOptional, lambdaMag,
                          kvLayout, stemBlockSize, stemStride),
                   DFX_OUT(kFlat, vBias));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (kvLayout == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    std::string kvLayoutStr(kvLayout);
    int64_t kvLayoutVal = KV_LAYOUT_BBND;
    if (kvLayoutStr == "BNBD") {
        kvLayoutVal = KV_LAYOUT_BNBD;
    } else if (kvLayoutStr == "BBND") {
        kvLayoutVal = KV_LAYOUT_BBND;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvLayout must be BBND or BNBD, got %s.", kvLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto ret = CheckParams(kCache, vCache, kvIndices, kScaleCacheOptional, vScaleOptional, kvSeqLens, kvLayoutVal,
                           stemBlockSize, stemStride, kFlat, vBias, lambdaMag);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto kCacheView = IsContiguous(kCache) ?
                          kCache :
                          uniqueExecutor->CreateView(kCache, kCache->GetViewShape(), kCache->GetStorageShape(),
                                                     kCache->GetViewStrides(), kCache->GetViewOffset());
    auto vCacheView = IsContiguous(vCache) ?
                          vCache :
                          uniqueExecutor->CreateView(vCache, vCache->GetViewShape(), vCache->GetStorageShape(),
                                                     vCache->GetViewStrides(), vCache->GetViewOffset());
    const aclTensor *kScaleCacheView = nullptr;
    if (kScaleCacheOptional != nullptr) {
        kScaleCacheView =
            IsContiguous(kScaleCacheOptional) ?
                kScaleCacheOptional :
                uniqueExecutor->CreateView(kScaleCacheOptional, kScaleCacheOptional->GetViewShape(),
                                           kScaleCacheOptional->GetStorageShape(),
                                           kScaleCacheOptional->GetViewStrides(), kScaleCacheOptional->GetViewOffset());
    }
    const aclTensor *vScaleContiguous = nullptr;
    if (vScaleOptional != nullptr) {
        vScaleContiguous = l0op::Contiguous(vScaleOptional, uniqueExecutor.get());
    }

    auto kvIndicesContiguous = l0op::Contiguous(kvIndices, uniqueExecutor.get());
    const aclTensor *kvSeqLensTensor = uniqueExecutor->ConvertToTensor(kvSeqLens, DataType::DT_INT32);
    OP_CHECK_NULL(kvSeqLensTensor, return ACLNN_ERR_PARAM_NULLPTR);
    float lambdaMagFloat = static_cast<float>(lambdaMag);

    auto result = l0op::StemOamPrepPagedKv(kCacheView, vCacheView, kvIndicesContiguous, kvSeqLensTensor,
                                           kScaleCacheView, vScaleContiguous, lambdaMagFloat, kvLayoutStr,
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
