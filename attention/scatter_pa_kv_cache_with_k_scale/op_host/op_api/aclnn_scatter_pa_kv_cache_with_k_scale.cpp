/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_scatter_pa_kv_cache_with_k_scale.h"

#include <new>

#include "scatter_pa_kv_cache_with_k_scale.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
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

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<op::DataType> KEY_VALUE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E5M2,
                                                                                 DataType::DT_FLOAT8_E4M3FN};

static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static aclnnStatus CheckNullptr(const aclTensor* key, const aclTensor* value, const aclTensor* keyCacheRef,
                                const aclTensor* valueCacheRef, const aclTensor* slotMapping, const aclTensor* keyScale,
                                const aclTensor* keyScaleCacheRef, const uint64_t* workspaceSize,
                                aclOpExecutor** executor)
{
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyCacheRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(valueCacheRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyScaleCacheRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypeValid(const aclTensor* key, const aclTensor* value, const aclTensor* keyCacheRef,
                                   const aclTensor* valueCacheRef, const aclTensor* slotMapping,
                                   const aclTensor* keyScale, const aclTensor* keyScaleCacheRef)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(key, KEY_VALUE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(value, KEY_VALUE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(keyCacheRef, KEY_VALUE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(valueCacheRef, KEY_VALUE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(slotMapping, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(keyScale, SCALE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(keyScaleCacheRef, SCALE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    auto keyDtype = key->GetDataType();
    auto keyCacheDtype = keyCacheRef->GetDataType();
    if (keyDtype != keyCacheDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of key[%s] and keyCacheRef[%s] are not equal.",
                op::ToString(DataType(keyDtype)).GetString(), op::ToString(DataType(keyCacheDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto valueDtype = value->GetDataType();
    auto valueCacheDtype = valueCacheRef->GetDataType();
    if (valueDtype != valueCacheDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of value[%s] and valueCacheRef[%s] are not equal.",
                op::ToString(DataType(valueDtype)).GetString(), op::ToString(DataType(valueCacheDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (keyDtype != valueDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of key[%s] and value[%s] are not equal.",
                op::ToString(DataType(keyDtype)).GetString(), op::ToString(DataType(valueDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto slotMappingDtype = slotMapping->GetDataType();
    auto keyScaleDtype = keyScale->GetDataType();
    auto keyScaleCacheDtype = keyScaleCacheRef->GetDataType();
    if (keyScaleDtype != keyScaleCacheDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of keyScale[%s] and keyScaleCacheRef[%s] are not equal.",
                op::ToString(DataType(keyScaleDtype)).GetString(),
                op::ToString(DataType(keyScaleCacheDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const aclTensor* key, const aclTensor* value, const aclTensor* keyCacheRef,
                              const aclTensor* valueCacheRef, const aclTensor* slotMapping, const aclTensor* keyScale,
                              const aclTensor* keyScaleCacheRef)
{
    auto keyDimNum = key->GetViewShape().GetDimNum();
    auto valueDimNum = value->GetViewShape().GetDimNum();
    auto slotMappingDimNum = slotMapping->GetViewShape().GetDimNum();

    if (keyDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of key must be 3, but got %ld.", keyDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto keyLastDim = key->GetViewShape().GetDim(keyDimNum - 1);
    if (keyLastDim == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The head_size of key cannot be 0.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (valueDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of value must be 3, but got %ld.", valueDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto valueLastDim = value->GetViewShape().GetDim(valueDimNum - 1);
    if (valueLastDim == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The head_size of value cannot be 0.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (slotMappingDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of slotMapping must be 1, but got %ld.", slotMappingDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto keyCacheDimNum = keyCacheRef->GetViewShape().GetDimNum();
    if (keyCacheDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of keyCache must be 4, but got %ld.", keyCacheDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto keyCacheLastDim = keyCacheRef->GetViewShape().GetDim(keyCacheDimNum - 1);
    if (keyCacheLastDim == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The head_size of keyCache cannot be 0.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto valueCacheDimNum = valueCacheRef->GetViewShape().GetDimNum();
    if (valueCacheDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of valueCache must be 4, but got %ld.", valueCacheDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto valueCacheLastDim = valueCacheRef->GetViewShape().GetDim(valueCacheDimNum - 1);
    if (valueCacheLastDim == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The head_size of valueCache cannot be 0.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto keyScaleDimNum = keyScale->GetViewShape().GetDimNum();
    if (keyScaleDimNum != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of keyScale must be 2, but got %ld.", keyScaleDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto keyScaleCacheDimNum = keyScaleCacheRef->GetViewShape().GetDimNum();
    if (keyScaleCacheDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of keyScaleCache must be 4, but got %ld.", keyScaleCacheDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static bool IsLastAxisContiguous(const aclTensor* tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    auto viewShape = tensor->GetViewShape();
    auto viewStrides = tensor->GetViewStrides();
    int64_t dimNum = viewShape.GetDimNum();
    if (dimNum < 1) {
        return false;
    }

    return viewStrides[dimNum - 1] == 1;
}

static bool IsSupportNonContiguous(const aclTensor* key, const aclTensor* value, const aclTensor* keyCacheRef,
                                   const aclTensor* valueCacheRef, const aclTensor* keyScale,
                                   const aclTensor* keyScaleCacheRef)
{
    bool keyLastAxisContiguous = IsLastAxisContiguous(key);
    bool valueLastAxisContiguous = IsLastAxisContiguous(value);
    bool keyCacheLastAxisContiguous = IsLastAxisContiguous(keyCacheRef);
    bool valueCacheLastAxisContiguous = IsLastAxisContiguous(valueCacheRef);
    bool keyScaleCacheLastAxisContiguous = IsLastAxisContiguous(keyScaleCacheRef);

    bool allLastAxisContiguous = keyLastAxisContiguous && valueLastAxisContiguous && keyCacheLastAxisContiguous &&
                                 valueCacheLastAxisContiguous && keyScaleCacheLastAxisContiguous;

    bool anyNonContiguous = !IsContiguous(key) || !IsContiguous(value) || !IsContiguous(keyCacheRef) ||
                            !IsContiguous(valueCacheRef) || !IsContiguous(keyScale) || !IsContiguous(keyScaleCacheRef);

    return allLastAxisContiguous && anyNonContiguous;
}

static aclnnStatus ProcessNonContiguous(const aclTensor* key, const aclTensor* value, aclTensor* keyCacheRef,
                                        aclTensor* valueCacheRef, const aclTensor* slotMapping, aclTensor* keyScale,
                                        aclTensor* keyScaleCacheRef, char* cacheLayoutOptional, uint64_t* workspaceSize,
                                        aclOpExecutor** executor, UniqueExecutor& uniqueExecutor)
{
    const aclTensor* keyView = nullptr;
    if (IsLastAxisContiguous(key)) {
        keyView = uniqueExecutor->CreateView(key, key->GetViewShape(), key->GetStorageShape(), key->GetViewStrides(),
                                             key->GetViewOffset());
        CHECK_RET(keyView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        keyView = l0op::Contiguous(key, uniqueExecutor.get());
        CHECK_RET(keyView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* valueView = nullptr;
    if (IsLastAxisContiguous(value)) {
        valueView = uniqueExecutor->CreateView(value, value->GetViewShape(), value->GetStorageShape(),
                                               value->GetViewStrides(), value->GetViewOffset());
        CHECK_RET(valueView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        valueView = l0op::Contiguous(value, uniqueExecutor.get());
        CHECK_RET(valueView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto keyCacheView =
        uniqueExecutor->CreateView(keyCacheRef, keyCacheRef->GetViewShape(), keyCacheRef->GetStorageShape(),
                                   keyCacheRef->GetViewStrides(), keyCacheRef->GetViewOffset());
    CHECK_RET(keyCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueCacheView =
        uniqueExecutor->CreateView(valueCacheRef, valueCacheRef->GetViewShape(), valueCacheRef->GetStorageShape(),
                                   valueCacheRef->GetViewStrides(), valueCacheRef->GetViewOffset());
    CHECK_RET(valueCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyScaleView = uniqueExecutor->CreateView(keyScale, keyScale->GetViewShape(), keyScale->GetStorageShape(),
                                                   keyScale->GetViewStrides(), keyScale->GetViewOffset());
    CHECK_RET(keyScaleView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyScaleCacheView = uniqueExecutor->CreateView(
        keyScaleCacheRef, keyScaleCacheRef->GetViewShape(), keyScaleCacheRef->GetStorageShape(),
        keyScaleCacheRef->GetViewStrides(), keyScaleCacheRef->GetViewOffset());
    CHECK_RET(keyScaleCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scatterResult = l0op::ScatterPaKvCacheWithKScale(
        keyView, valueView, const_cast<aclTensor*>(keyCacheView), const_cast<aclTensor*>(valueCacheView),
        slotMappingContiguous, const_cast<aclTensor*>(keyScaleView), const_cast<aclTensor*>(keyScaleCacheView),
        cacheLayoutOptional, uniqueExecutor.get());
    CHECK_RET(std::get<0>(scatterResult) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessContiguous(const aclTensor* key, const aclTensor* value, aclTensor* keyCacheRef,
                                     aclTensor* valueCacheRef, const aclTensor* slotMapping, aclTensor* keyScale,
                                     aclTensor* keyScaleCacheRef, char* cacheLayoutOptional, uint64_t* workspaceSize,
                                     aclOpExecutor** executor, UniqueExecutor& uniqueExecutor)
{
    auto keyContiguous = l0op::Contiguous(key, uniqueExecutor.get());
    CHECK_RET(keyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueContiguous = l0op::Contiguous(value, uniqueExecutor.get());
    CHECK_RET(valueContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyCacheContiguous = const_cast<aclTensor*>(l0op::Contiguous(keyCacheRef, uniqueExecutor.get()));
    CHECK_RET(keyCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueCacheContiguous = const_cast<aclTensor*>(l0op::Contiguous(valueCacheRef, uniqueExecutor.get()));
    CHECK_RET(valueCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyScaleContiguous = const_cast<aclTensor*>(l0op::Contiguous(keyScale, uniqueExecutor.get()));
    CHECK_RET(keyScaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyScaleCacheContiguous = const_cast<aclTensor*>(l0op::Contiguous(keyScaleCacheRef, uniqueExecutor.get()));
    CHECK_RET(keyScaleCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scatterResult = l0op::ScatterPaKvCacheWithKScale(
        keyContiguous, valueContiguous, keyCacheContiguous, valueCacheContiguous, slotMappingContiguous,
        keyScaleContiguous, keyScaleCacheContiguous, cacheLayoutOptional, uniqueExecutor.get());
    CHECK_RET(std::get<0>(scatterResult) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyCacheViewCopy = l0op::ViewCopy(std::get<0>(scatterResult), keyCacheRef, uniqueExecutor.get());
    CHECK_RET(keyCacheViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueCacheViewCopy = l0op::ViewCopy(std::get<1>(scatterResult), valueCacheRef, uniqueExecutor.get());
    CHECK_RET(valueCacheViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyScaleCacheViewCopy = l0op::ViewCopy(std::get<2>(scatterResult), keyScaleCacheRef, uniqueExecutor.get());
    CHECK_RET(keyScaleCacheViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize(const aclTensor* key, const aclTensor* value,
                                                            aclTensor* keyCacheRef, aclTensor* valueCacheRef,
                                                            const aclTensor* slotMapping, aclTensor* keyScale,
                                                            aclTensor* keyScaleCacheRef, char* cacheLayoutOptional,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnScatterPaKvCacheWithKScale,
        DFX_IN(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayoutOptional),
        DFX_OUT(keyCacheRef, valueCacheRef, keyScaleCacheRef));

    aclnnStatus nullPtrRet = CheckNullptr(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale,
                                          keyScaleCacheRef, workspaceSize, executor);
    if (nullPtrRet != ACLNN_SUCCESS) {
        return nullPtrRet;
    }

    bool keyEmpty = key->GetViewShape().GetShapeSize() == 0 || keyCacheRef->GetViewShape().GetShapeSize() == 0;
    bool valueEmpty = value->GetViewShape().GetShapeSize() == 0 || valueCacheRef->GetViewShape().GetShapeSize() == 0;
    bool scaleEmpty =
        keyScale->GetViewShape().GetShapeSize() == 0 || keyScaleCacheRef->GetViewShape().GetShapeSize() == 0;
    bool slotMappingEmpty = slotMapping->GetViewShape().GetShapeSize() == 0;
    if (slotMappingEmpty || (keyEmpty && valueEmpty && scaleEmpty)) {
        *workspaceSize = 0;
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(
        CheckShape(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef) == ACLNN_SUCCESS,
        ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckDtypeValid(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    bool isNonContiguous = IsSupportNonContiguous(key, value, keyCacheRef, valueCacheRef, keyScale, keyScaleCacheRef);
    if (isNonContiguous) {
        return ProcessNonContiguous(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef,
                                    cacheLayoutOptional, workspaceSize, executor, uniqueExecutor);
    }

    return ProcessContiguous(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef,
                             cacheLayoutOptional, workspaceSize, executor, uniqueExecutor);
}

aclnnStatus aclnnScatterPaKvCacheWithKScale(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnScatterPaKvCacheWithKScale);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ScatterPaKvCacheWithKScale launch aicore failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
