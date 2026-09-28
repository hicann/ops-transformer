/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_scatter_pa_kv_cache.h"

#include <new>

#include "scatter_pa_kv_cache.h"

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
#include "log/log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char *kOpName = "aclnnScatterPaKvCache";

static const std::initializer_list<op::DataType> KEY_VALUE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16,     DataType::DT_FLOAT,      DataType::DT_BF16,        DataType::DT_INT8,
    DataType::DT_UINT8,       DataType::DT_INT16,      DataType::DT_UINT16,      DataType::DT_INT32,
    DataType::DT_UINT32,      DataType::DT_HIFLOAT8,   DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN,
    DataType::DT_FLOAT4_E1M2, DataType::DT_FLOAT4_E2M1};

static const std::initializer_list<op::DataType> KEY_VALUE_DTYPE_SUPPORT_LIST_910 = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static aclnnStatus CheckNullptr(const aclTensor *key, const aclTensor *keyCacheRef, const aclTensor *slotMapping,
                                const aclTensor *value, const aclTensor *valueCacheRef, const uint64_t *workspaceSize,
                                aclOpExecutor **executor)
{
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyCacheRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (value != nullptr) {
        CHECK_RET(valueCacheRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypeValid(const aclTensor *key, const aclTensor *keyCacheRef, const aclTensor *slotMapping,
                                   const aclTensor *value, const aclTensor *valueCacheRef,
                                   const aclTensor *compressLensOptional, const aclTensor *compressSeqOffsetOptional,
                                   const aclTensor *seqLensOptional, char *cacheModeOptional)
{
    auto &platformInfo = op::GetCurrentPlatformInfo();
    auto socVersion = platformInfo.GetCurNpuArch();
    bool isArch35 = (socVersion == NpuArch::DAV_3510);
    auto dtypeSupportList = isArch35 ? KEY_VALUE_DTYPE_SUPPORT_LIST : KEY_VALUE_DTYPE_SUPPORT_LIST_910;

    if (!CheckType(key->GetDataType(), dtypeSupportList)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "key", op::ToString(key->GetDataType()).GetString(),
            std::string("The dtype of key must be within the range ") + op::ToString(dtypeSupportList).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckType(keyCacheRef->GetDataType(), dtypeSupportList)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "keyCacheRef",
                                              op::ToString(keyCacheRef->GetDataType()).GetString(),
                                              std::string("The dtype of keyCacheRef must be within the range ") +
                                                  op::ToString(dtypeSupportList).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckType(slotMapping->GetDataType(), INDEX_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "slotMapping",
                                              op::ToString(slotMapping->GetDataType()).GetString(),
                                              std::string("The dtype of slotMapping must be within the range ") +
                                                  op::ToString(INDEX_DTYPE_SUPPORT_LIST).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto keyDtype = key->GetDataType();
    auto keyCacheDtype = keyCacheRef->GetDataType();
    if (keyDtype != keyCacheDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(kOpName, "key, keyCacheRef",
                                               (op::ToString(DataType(keyDtype)).GetString() + std::string(", ") +
                                                op::ToString(DataType(keyCacheDtype)).GetString())
                                                   .c_str(),
                                               "The data type of key and keyCacheRef must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (value != nullptr) {
        if (!CheckType(value->GetDataType(), dtypeSupportList)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "value", op::ToString(value->GetDataType()).GetString(),
                                                  std::string("The dtype of value must be within the range ") +
                                                      op::ToString(dtypeSupportList).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!CheckType(valueCacheRef->GetDataType(), dtypeSupportList)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, "valueCacheRef",
                                                  op::ToString(valueCacheRef->GetDataType()).GetString(),
                                                  std::string("The dtype of valueCacheRef must be within the range ") +
                                                      op::ToString(dtypeSupportList).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }

        auto valueDtype = value->GetDataType();
        auto valueCacheDtype = valueCacheRef->GetDataType();
        bool isPaNz = (cacheModeOptional != nullptr && strcmp(cacheModeOptional, "PA_NZ") == 0);

        if (valueDtype != valueCacheDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(kOpName, "value, valueCacheRef",
                                                   (op::ToString(DataType(valueDtype)).GetString() + std::string(", ") +
                                                    op::ToString(DataType(valueCacheDtype)).GetString())
                                                       .c_str(),
                                                   "The data type of value and valueCacheRef must be equal");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (!isPaNz && keyDtype != valueDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                kOpName, "key, value",
                (op::ToString(DataType(keyDtype)).GetString() + std::string(", ") +
                 op::ToString(DataType(valueDtype)).GetString())
                    .c_str(),
                "The data type of key and value must be equal when cacheMode is not PA_NZ");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    auto slotMappingDtype = slotMapping->GetDataType();
    if (compressLensOptional != nullptr) {
        if (slotMappingDtype != compressLensOptional->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                kOpName, "slotMapping, compressLensOptional",
                (op::ToString(DataType(slotMappingDtype)).GetString() + std::string(", ") +
                 op::ToString(DataType(compressLensOptional->GetDataType())).GetString())
                    .c_str(),
                "The data type of slotMapping and compressLensOptional must be equal");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (compressSeqOffsetOptional != nullptr) {
        if (slotMappingDtype != compressSeqOffsetOptional->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                kOpName, "slotMapping, compressSeqOffsetOptional",
                (op::ToString(DataType(slotMappingDtype)).GetString() + std::string(", ") +
                 op::ToString(DataType(compressSeqOffsetOptional->GetDataType())).GetString())
                    .c_str(),
                "The data type of slotMapping and compressSeqOffsetOptional must be equal");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (seqLensOptional != nullptr) {
        if (slotMappingDtype != seqLensOptional->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                kOpName, "slotMapping, seqLensOptional",
                (op::ToString(DataType(slotMappingDtype)).GetString() + std::string(", ") +
                 op::ToString(DataType(seqLensOptional->GetDataType())).GetString())
                    .c_str(),
                "The data type of slotMapping and seqLensOptional must be equal");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const aclTensor *key, const aclTensor *value)
{
    auto keyDimNum = key->GetViewShape().GetDimNum();
    auto valueDimNum = value != nullptr ? value->GetViewShape().GetDimNum() : 0;

    if (keyDimNum != 3 && keyDimNum != 4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "key", std::to_string(keyDimNum).c_str(),
                                                 "The dim num of key must be 3 or 4");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (value != nullptr && valueDimNum != 0 && valueDimNum != 3 && valueDimNum != 4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "value", std::to_string(valueDimNum).c_str(),
                                                 "The dim num of value must be 0, 3 or 4");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static bool IsAxesContiguous(const aclTensor *tensor, int64_t startAxis, int64_t endAxis)
{
    if (tensor == nullptr || startAxis < 0 || endAxis < 0 || startAxis >= endAxis) {
        return false;
    }

    auto viewShape = tensor->GetViewShape();
    auto viewStrides = tensor->GetViewStrides();
    int64_t dimNum = viewShape.GetDimNum();
    if (endAxis > dimNum) {
        return false;
    }

    int64_t validStride = 1;
    for (int64_t i = endAxis - 1; i >= startAxis; i--) {
        if (viewShape.GetDim(i) == 1) {
            continue;
        }
        if (viewStrides[i] != validStride) {
            return false;
        }
        validStride *= viewShape.GetDim(i);
    }
    return true;
}

static bool IsFirstAxisNonContiguous(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return false;
    }

    auto viewShape = tensor->GetViewShape();
    auto viewStrides = tensor->GetViewStrides();
    int64_t dimNum = viewShape.GetDimNum();
    if (dimNum < 2) {
        return false;
    }

    bool firstAxisContiguous = (viewStrides[0] == viewShape.GetDim(1) * viewStrides[1]);
    bool otherAxesContiguous = IsAxesContiguous(tensor, 1, dimNum);

    return (!firstAxisContiguous) && otherAxesContiguous;
}

static bool IsLastAxisContiguous(const aclTensor *tensor)
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

static bool IsSupportNonContiguousCache(const aclTensor *key, const aclTensor *keyCacheRef, const aclTensor *value,
                                        const aclTensor *valueCacheRef)
{
    bool keyContiguous = IsContiguous(key);
    bool valueContiguous = true;
    if (value != nullptr) {
        valueContiguous = IsContiguous(value);
    }

    bool keyCacheNonContiguous = IsFirstAxisNonContiguous(keyCacheRef);
    bool valueCacheNonContiguous = true;
    if (value != nullptr) {
        valueCacheNonContiguous = IsFirstAxisNonContiguous(valueCacheRef);
    }

    bool keyAndValueContiguous = keyContiguous && valueContiguous;
    bool cacheNonContiguous = keyCacheNonContiguous || valueCacheNonContiguous;

    return keyAndValueContiguous && cacheNonContiguous;
}

static bool IsSupportNonContiguousKeyAndCache(const aclTensor *key, const aclTensor *keyCacheRef,
                                              const aclTensor *value, const aclTensor *valueCacheRef)
{
    bool keyLastAxisContiguous = IsLastAxisContiguous(key);
    bool keyCacheLastAxisContiguous = IsLastAxisContiguous(keyCacheRef);
    bool valueLastAxisContiguous = true;
    bool valueCacheLastAxisContiguous = true;

    if (value != nullptr) {
        valueLastAxisContiguous = IsLastAxisContiguous(value);
        valueCacheLastAxisContiguous = IsLastAxisContiguous(valueCacheRef);
    }

    bool allLastAxisContiguous =
        keyLastAxisContiguous && keyCacheLastAxisContiguous && valueLastAxisContiguous && valueCacheLastAxisContiguous;

    bool anyNonContiguous = !IsContiguous(key) || !IsContiguous(keyCacheRef);
    if (value != nullptr) {
        anyNonContiguous = anyNonContiguous || !IsContiguous(value) || !IsContiguous(valueCacheRef);
    }

    return allLastAxisContiguous && anyNonContiguous;
}

static aclnnStatus ProcessNonContiguous(const aclTensor *key, aclTensor *keyCacheRef, const aclTensor *slotMapping,
                                        const aclTensor *value, aclTensor *valueCacheRef,
                                        const aclTensor *compressLensOptional,
                                        const aclTensor *compressSeqOffsetOptional, const aclTensor *seqLensOptional,
                                        char *cacheModeOptional, char *scatterModeOptional,
                                        const aclIntArray *stridesOptional, const aclIntArray *offsetsOptional,
                                        uint64_t *workspaceSize, aclOpExecutor **executor,
                                        UniqueExecutor &uniqueExecutor, bool isKeyAndCacheNonContiguous)
{
    const aclTensor *keyView = nullptr;
    if (isKeyAndCacheNonContiguous && IsLastAxisContiguous(key)) {
        keyView = uniqueExecutor->CreateView(key, key->GetViewShape(), key->GetStorageShape(), key->GetViewStrides(),
                                             key->GetViewOffset());
        CHECK_RET(keyView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        keyView = l0op::Contiguous(key, uniqueExecutor.get());
        CHECK_RET(keyView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto keyCacheView =
        uniqueExecutor->CreateView(keyCacheRef, keyCacheRef->GetViewShape(), keyCacheRef->GetStorageShape(),
                                   keyCacheRef->GetViewStrides(), keyCacheRef->GetViewOffset());
    CHECK_RET(keyCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *valueView = value;
    const aclTensor *valueCacheView = nullptr;
    if (value != nullptr) {
        if (isKeyAndCacheNonContiguous && IsLastAxisContiguous(value)) {
            valueView = uniqueExecutor->CreateView(value, value->GetViewShape(), value->GetStorageShape(),
                                                   value->GetViewStrides(), value->GetViewOffset());
            CHECK_RET(valueView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            valueView = l0op::Contiguous(value, uniqueExecutor.get());
            CHECK_RET(valueView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        valueCacheView =
            uniqueExecutor->CreateView(valueCacheRef, valueCacheRef->GetViewShape(), valueCacheRef->GetStorageShape(),
                                       valueCacheRef->GetViewStrides(), valueCacheRef->GetViewOffset());
        CHECK_RET(valueCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto compressLensContiguous = compressLensOptional;
    if (compressLensOptional != nullptr) {
        compressLensContiguous = l0op::Contiguous(compressLensOptional, uniqueExecutor.get());
        CHECK_RET(compressLensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto compressSeqOffsetContiguous = compressSeqOffsetOptional;
    if (compressSeqOffsetOptional != nullptr) {
        compressSeqOffsetContiguous = l0op::Contiguous(compressSeqOffsetOptional, uniqueExecutor.get());
        CHECK_RET(compressSeqOffsetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto seqLensContiguous = seqLensOptional;
    if (seqLensOptional != nullptr) {
        seqLensContiguous = l0op::Contiguous(seqLensOptional, uniqueExecutor.get());
        CHECK_RET(seqLensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto scatterResult = l0op::ScatterPaKvCache(
        keyView, const_cast<aclTensor *>(keyCacheView), slotMappingContiguous, valueView,
        const_cast<aclTensor *>(valueCacheView), compressLensContiguous, compressSeqOffsetContiguous, seqLensContiguous,
        cacheModeOptional, scatterModeOptional, stridesOptional, offsetsOptional, uniqueExecutor.get());
    CHECK_RET(std::get<0>(scatterResult) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessContiguous(const aclTensor *key, aclTensor *keyCacheRef, const aclTensor *slotMapping,
                                     const aclTensor *value, aclTensor *valueCacheRef,
                                     const aclTensor *compressLensOptional, const aclTensor *compressSeqOffsetOptional,
                                     const aclTensor *seqLensOptional, char *cacheModeOptional,
                                     char *scatterModeOptional, const aclIntArray *stridesOptional,
                                     const aclIntArray *offsetsOptional, uint64_t *workspaceSize,
                                     aclOpExecutor **executor, UniqueExecutor &uniqueExecutor)
{
    auto keyContiguous = l0op::Contiguous(key, uniqueExecutor.get());
    CHECK_RET(keyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyCacheContiguous = const_cast<aclTensor *>(l0op::Contiguous(keyCacheRef, uniqueExecutor.get()));
    CHECK_RET(keyCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueContiguous = value;
    auto valueCacheContiguous = valueCacheRef;
    if (value != nullptr) {
        valueContiguous = l0op::Contiguous(value, uniqueExecutor.get());
        CHECK_RET(valueContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        valueCacheContiguous = const_cast<aclTensor *>(l0op::Contiguous(valueCacheRef, uniqueExecutor.get()));
        CHECK_RET(valueCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto compressLensContiguous = compressLensOptional;
    if (compressLensOptional != nullptr) {
        compressLensContiguous = l0op::Contiguous(compressLensOptional, uniqueExecutor.get());
        CHECK_RET(compressLensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto compressSeqOffsetContiguous = compressSeqOffsetOptional;
    if (compressSeqOffsetOptional != nullptr) {
        compressSeqOffsetContiguous = l0op::Contiguous(compressSeqOffsetOptional, uniqueExecutor.get());
        CHECK_RET(compressSeqOffsetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto seqLensContiguous = seqLensOptional;
    if (seqLensOptional != nullptr) {
        seqLensContiguous = l0op::Contiguous(seqLensOptional, uniqueExecutor.get());
        CHECK_RET(seqLensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto scatterResult = l0op::ScatterPaKvCache(
        keyContiguous, keyCacheContiguous, slotMappingContiguous, valueContiguous, valueCacheContiguous,
        compressLensContiguous, compressSeqOffsetContiguous, seqLensContiguous, cacheModeOptional, scatterModeOptional,
        stridesOptional, offsetsOptional, uniqueExecutor.get());
    CHECK_RET(std::get<0>(scatterResult) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyCacheViewCopy = l0op::ViewCopy(std::get<0>(scatterResult), keyCacheRef, uniqueExecutor.get());
    CHECK_RET(keyCacheViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (value != nullptr) {
        auto valueCacheViewCopy = l0op::ViewCopy(std::get<1>(scatterResult), valueCacheRef, uniqueExecutor.get());
        CHECK_RET(valueCacheViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(const aclTensor *key, aclTensor *keyCacheRef,
                                                  const aclTensor *slotMapping, const aclTensor *value,
                                                  aclTensor *valueCacheRef, const aclTensor *compressLensOptional,
                                                  const aclTensor *compressSeqOffsetOptional,
                                                  const aclTensor *seqLensOptional, char *cacheModeOptional,
                                                  char *scatterModeOptional, const aclIntArray *stridesOptional,
                                                  const aclIntArray *offsetsOptional, uint64_t *workspaceSize,
                                                  aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnScatterPaKvCache,
        DFX_IN(key, keyCacheRef, slotMapping, value, valueCacheRef, compressLensOptional, compressSeqOffsetOptional,
               seqLensOptional, cacheModeOptional, scatterModeOptional, stridesOptional, offsetsOptional),
        DFX_OUT(keyCacheRef, valueCacheRef));

    CHECK_RET(
        CheckNullptr(key, keyCacheRef, slotMapping, value, valueCacheRef, workspaceSize, executor) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER_NULLPTR);

    bool keyEmpty = key->GetViewShape().GetShapeSize() == 0 || keyCacheRef->GetViewShape().GetShapeSize() == 0;
    bool valueEmpty = false;
    if (value != nullptr) {
        valueEmpty = value->GetViewShape().GetShapeSize() == 0 || valueCacheRef->GetViewShape().GetShapeSize() == 0;
    }

    if (keyEmpty && (value == nullptr || valueEmpty)) {
        *workspaceSize = 0;
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(CheckShape(key, value) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckDtypeValid(key, keyCacheRef, slotMapping, value, valueCacheRef, compressLensOptional,
                              compressSeqOffsetOptional, seqLensOptional, cacheModeOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    bool isKey3Dim = (key->GetViewShape().GetDimNum() == 3);
    bool isScatterModeEmpty = (scatterModeOptional == nullptr || strlen(scatterModeOptional) == 0 ||
                               strcmp(scatterModeOptional, "None") == 0 || strcmp(scatterModeOptional, "NHSD") == 0);
    if (isKey3Dim && isScatterModeEmpty) {
        bool isCacheNonContiguous = IsSupportNonContiguousCache(key, keyCacheRef, value, valueCacheRef);
        bool isKeyAndCacheNonContiguous = IsSupportNonContiguousKeyAndCache(key, keyCacheRef, value, valueCacheRef);
        if (isCacheNonContiguous || isKeyAndCacheNonContiguous) {
            return ProcessNonContiguous(key, keyCacheRef, slotMapping, value, valueCacheRef, compressLensOptional,
                                        compressSeqOffsetOptional, seqLensOptional, cacheModeOptional,
                                        scatterModeOptional, stridesOptional, offsetsOptional, workspaceSize, executor,
                                        uniqueExecutor, isKeyAndCacheNonContiguous);
        }
    }

    return ProcessContiguous(key, keyCacheRef, slotMapping, value, valueCacheRef, compressLensOptional,
                             compressSeqOffsetOptional, seqLensOptional, cacheModeOptional, scatterModeOptional,
                             stridesOptional, offsetsOptional, workspaceSize, executor, uniqueExecutor);
}

aclnnStatus aclnnScatterPaKvCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnScatterPaKvCache);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ScatterPaKvCache launch aicore failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
