/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gather_pa_kv_cache.h"

#include <new>

#include "gather_pa_kv_cache.h"

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

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT,  DataType::DT_BF16,  DataType::DT_INT8,  DataType::DT_UINT8,
    DataType::DT_INT16,   DataType::DT_UINT16, DataType::DT_INT32, DataType::DT_UINT32};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_A5 = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT,    DataType::DT_BF16,        DataType::DT_INT8,
    DataType::DT_UINT8,   DataType::DT_INT16,    DataType::DT_UINT16,      DataType::DT_INT32,
    DataType::DT_UINT32,  DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static aclnnStatus CheckNullptr(const aclTensor* keyCache, const aclTensor* valueCache, const aclTensor* blockTables,
                                const aclTensor* seqLens, const aclTensor* keyRef, const aclTensor* valueRef,
                                const uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_RET(keyCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(valueCache != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTables != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(seqLens != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(valueRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypeValid(const aclTensor* keyCache, const aclTensor* valueCache, const aclTensor* blockTables,
                                   const aclTensor* seqLens, const aclTensor* keyRef, const aclTensor* valueRef,
                                   const aclTensor* seqOffsetOptional, char* cacheModeOptional)
{
    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto socVersion = platformInfo.GetCurNpuArch();
    bool isArch35 = (socVersion == NpuArch::DAV_3510);
    const auto& kvDtypeList = isArch35 ? DTYPE_SUPPORT_LIST_A5 : DTYPE_SUPPORT_LIST_910B;

    OP_CHECK_DTYPE_NOT_SUPPORT(keyCache, kvDtypeList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(valueCache, kvDtypeList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(blockTables, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(seqLens, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(keyRef, kvDtypeList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(valueRef, kvDtypeList, return ACLNN_ERR_PARAM_INVALID);

    auto keyCacheDtype = keyCache->GetDataType();
    auto keyRefDtype = keyRef->GetDataType();
    if (keyCacheDtype != keyRefDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of keyCache[%s] and keyRef[%s] are not equal.",
                op::ToString(DataType(keyCacheDtype)).GetString(), op::ToString(DataType(keyRefDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto valueCacheDtype = valueCache->GetDataType();
    auto valueRefDtype = valueRef->GetDataType();
    if (valueCacheDtype != valueRefDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of valueCache[%s] and valueRef[%s] are not equal.",
                op::ToString(DataType(valueCacheDtype)).GetString(), op::ToString(DataType(valueRefDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto blockTablesDtype = blockTables->GetDataType();
    auto seqLensDtype = seqLens->GetDataType();
    if (blockTablesDtype != seqLensDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of blockTables[%s] and seqLens[%s] are not equal.",
                op::ToString(DataType(blockTablesDtype)).GetString(), op::ToString(DataType(seqLensDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (seqOffsetOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(seqOffsetOptional, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
        if (blockTablesDtype != seqOffsetOptional->GetDataType()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type of blockTables[%s] and seqOffsetOptional[%s] are not equal.",
                    op::ToString(DataType(blockTablesDtype)).GetString(),
                    op::ToString(DataType(seqOffsetOptional->GetDataType())).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const aclTensor* keyCache, const aclTensor* valueCache, const aclTensor* blockTables,
                              const aclTensor* seqLens, const aclTensor* keyRef, const aclTensor* valueRef)
{
    auto keyCacheDimNum = keyCache->GetViewShape().GetDimNum();
    if (keyCacheDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of keyCache must be 4, but got %ld.", keyCacheDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto valueCacheDimNum = valueCache->GetViewShape().GetDimNum();
    if (valueCacheDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of valueCache must be 4, but got %ld.", valueCacheDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto blockTablesDimNum = blockTables->GetViewShape().GetDimNum();
    if (blockTablesDimNum != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of blockTables must be 2, but got %ld.", blockTablesDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto seqLensDimNum = seqLens->GetViewShape().GetDimNum();
    if (seqLensDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqLens must be 1, but got %ld.", seqLensDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto keyRefDimNum = keyRef->GetViewShape().GetDimNum();
    if (keyRefDimNum != 2 && keyRefDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of keyRef must be 2 or 3, but got %ld.", keyRefDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto valueRefDimNum = valueRef->GetViewShape().GetDimNum();
    if (valueRefDimNum != 2 && valueRefDimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of valueRef must be 2 or 3, but got %ld.", valueRefDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static bool IsAxesContiguous(const aclTensor* tensor, int64_t startAxis, int64_t endAxis)
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

static bool IsFirstAxisNonContiguous(const aclTensor* tensor)
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

    bool lastAxisContiguous = (viewStrides[dimNum - 1] == 1);
    bool otherAxesNonContiguous = false;

    if (dimNum > 1) {
        otherAxesNonContiguous = !IsAxesContiguous(tensor, 0, dimNum - 1);
    }

    return lastAxisContiguous && (dimNum == 1 || otherAxesNonContiguous);
}

// 判断单个tensor是否需要走ProcessContiguous物理连续化兜底(kernel无法处理的非连续形态):
static bool NeedContiguousFallback(const aclTensor* tensor, bool isPaNz)
{
    if (tensor == nullptr) {
        return false;
    }
    auto viewShape = tensor->GetViewShape();
    int64_t dimNum = viewShape.GetDimNum();
    if (dimNum < 1) {
        return false;
    }
    if (isPaNz) {
        // dimNum<2时无"非dim0轴", 不可能非连续。
        return dimNum >= 2 && !IsAxesContiguous(tensor, 1, dimNum);
    }
    // ND: 仅判尾轴, size为1视为连续。
    if (viewShape.GetDim(dimNum - 1) == 1) {
        return false;
    }
    return tensor->GetViewStrides()[dimNum - 1] != 1;
}

static bool IsRefNonContiguous(const aclTensor* ref)
{
    if (ref == nullptr) {
        return false;
    }
    int64_t dimNum = ref->GetViewShape().GetDimNum();
    if (dimNum < 1) {
        return false;
    }
    return !IsAxesContiguous(ref, 0, dimNum);
}

static bool NeedContiguousFallbackAny(const aclTensor* keyCache, const aclTensor* valueCache, const aclTensor* keyRef,
                                      const aclTensor* valueRef, bool isPaNz)
{
    // ref与cache采用同一尾轴判定: 尾轴非连续是kernel无法散写的形态, 必须物理连续化兜底。
    // ref内部轴(ND dim0/dim1, NZ dim0)非连续但尾轴连续 -> kernel可零拷贝散写, 不进兜底。
    return NeedContiguousFallback(keyCache, isPaNz) || NeedContiguousFallback(valueCache, isPaNz) ||
           NeedContiguousFallback(keyRef, isPaNz) || NeedContiguousFallback(valueRef, isPaNz);
}

static bool IsSupportNonContiguousCache(const aclTensor* keyCache, const aclTensor* valueCache)
{
    bool keyCacheCase1 = IsFirstAxisNonContiguous(keyCache);
    bool valueCacheCase1 = IsFirstAxisNonContiguous(valueCache);

    return keyCacheCase1 || valueCacheCase1;
}

// ref(key/value)内部轴非连续(ND dim0/dim1, NZ dim0)、尾轴连续: kernel可零拷贝按stride散写。
// 尾轴连续性由调用侧mustContiguous=false保证(NeedContiguousFallback判尾轴), 故此处只判整体非连续。
static bool IsSupportNonContiguousRef(const aclTensor* keyRef, const aclTensor* valueRef)
{
    return IsRefNonContiguous(keyRef) || IsRefNonContiguous(valueRef);
}

// 仅ND(Norm)模式: cache内部轴(dim1 slot / dim2 head)非连续, ref连续。
// 此形态kernel可零拷贝(arch35 nd kernel Case B/C用stride1/stride2搬运), tiling从view stride检测。
// 尾轴连续性由调用侧mustContiguous=false保证(NeedContiguousFallback对ND判尾轴), 故此处只需判整体非连续。
// !isPaNz守卫: NZ kernel不支持内部轴非连续(tiling硬拦截GRAPH_FAILED), 必须走连续化兜底。
static bool IsSupportNonContiguousCacheNdInner(const aclTensor* keyCache, const aclTensor* valueCache, bool isPaNz)
{
    if (isPaNz) {
        return false;
    }
    return !IsContiguous(keyCache) || !IsContiguous(valueCache);
}

static aclnnStatus ProcessNonContiguous(const aclTensor* keyCache, const aclTensor* valueCache,
                                        const aclTensor* blockTables, const aclTensor* seqLens, aclTensor* keyRef,
                                        aclTensor* valueRef, const aclTensor* seqOffsetOptional, char* cacheMode,
                                        bool isSeqLensCumsum, uint64_t* workspaceSize, aclOpExecutor** executor,
                                        UniqueExecutor& uniqueExecutor)
{
    auto keyCacheView = uniqueExecutor->CreateView(keyCache, keyCache->GetViewShape(), keyCache->GetStorageShape(),
                                                   keyCache->GetViewStrides(), keyCache->GetViewOffset());
    CHECK_RET(keyCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueCacheView =
        uniqueExecutor->CreateView(valueCache, valueCache->GetViewShape(), valueCache->GetStorageShape(),
                                   valueCache->GetViewStrides(), valueCache->GetViewOffset());
    CHECK_RET(valueCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto blockTablesContiguous = l0op::Contiguous(blockTables, uniqueExecutor.get());
    CHECK_RET(blockTablesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto seqLensContiguous = l0op::Contiguous(seqLens, uniqueExecutor.get());
    CHECK_RET(seqLensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // ref非连续(尾轴连续, 由mustContiguous=false保证)走CreateView零拷贝: kernel按view stride直写原ref内存,
    // 无需ViewCopy回写。ref连续则Contiguous(no-op)。
    const aclTensor* keyRefView = nullptr;
    if (!IsContiguous(keyRef) && IsLastAxisContiguous(keyRef)) {
        keyRefView = uniqueExecutor->CreateView(keyRef, keyRef->GetViewShape(), keyRef->GetStorageShape(),
                                                keyRef->GetViewStrides(), keyRef->GetViewOffset());
        CHECK_RET(keyRefView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        keyRefView = l0op::Contiguous(keyRef, uniqueExecutor.get());
        CHECK_RET(keyRefView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* valueRefView = nullptr;
    if (!IsContiguous(valueRef) && IsLastAxisContiguous(valueRef)) {
        valueRefView = uniqueExecutor->CreateView(valueRef, valueRef->GetViewShape(), valueRef->GetStorageShape(),
                                                  valueRef->GetViewStrides(), valueRef->GetViewOffset());
        CHECK_RET(valueRefView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        valueRefView = l0op::Contiguous(valueRef, uniqueExecutor.get());
        CHECK_RET(valueRefView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto seqOffsetContiguous = seqOffsetOptional;
    if (seqOffsetOptional != nullptr) {
        seqOffsetContiguous = l0op::Contiguous(seqOffsetOptional, uniqueExecutor.get());
        CHECK_RET(seqOffsetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto gatherResult = l0op::GatherPaKvCache(keyCacheView, valueCacheView, blockTablesContiguous, seqLensContiguous,
                                              const_cast<aclTensor*>(keyRefView), const_cast<aclTensor*>(valueRefView),
                                              seqOffsetContiguous, cacheMode, isSeqLensCumsum, uniqueExecutor.get());
    CHECK_RET(std::get<0>(gatherResult) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static const aclTensor* ContiguousNzCacheAsNd(const aclTensor* cache, UniqueExecutor& uniqueExecutor)
{
    auto ndView = uniqueExecutor->CreateView(cache, cache->GetViewShape(), cache->GetStorageShape(),
                                             cache->GetViewStrides(), cache->GetViewOffset());
    if (ndView == nullptr) {
        return nullptr;
    }
    ndView->SetViewFormat(op::Format::FORMAT_ND);
    ndView->SetStorageFormat(op::Format::FORMAT_ND);

    auto ndContiguous = const_cast<aclTensor*>(l0op::Contiguous(ndView, uniqueExecutor.get()));
    if (ndContiguous == nullptr) {
        return nullptr;
    }

    // 连续化结果贴回NZ format给kernel。
    ndContiguous->SetViewFormat(op::Format::FORMAT_FRACTAL_NZ);
    ndContiguous->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    return ndContiguous;
}

static aclnnStatus ProcessContiguous(const aclTensor* keyCache, const aclTensor* valueCache,
                                     const aclTensor* blockTables, const aclTensor* seqLens, aclTensor* keyRef,
                                     aclTensor* valueRef, const aclTensor* seqOffsetOptional, char* cacheMode,
                                     bool isSeqLensCumsum, uint64_t* workspaceSize, aclOpExecutor** executor,
                                     UniqueExecutor& uniqueExecutor)
{
    bool isPaNz = (cacheMode != nullptr && strcmp(cacheMode, "PA_NZ") == 0);
    auto keyCacheContiguous = (isPaNz && NeedContiguousFallback(keyCache, isPaNz)) ?
                                  ContiguousNzCacheAsNd(keyCache, uniqueExecutor) :
                                  l0op::Contiguous(keyCache, uniqueExecutor.get());
    CHECK_RET(keyCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueCacheContiguous = (isPaNz && NeedContiguousFallback(valueCache, isPaNz)) ?
                                    ContiguousNzCacheAsNd(valueCache, uniqueExecutor) :
                                    l0op::Contiguous(valueCache, uniqueExecutor.get());
    CHECK_RET(valueCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto blockTablesContiguous = l0op::Contiguous(blockTables, uniqueExecutor.get());
    CHECK_RET(blockTablesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto seqLensContiguous = l0op::Contiguous(seqLens, uniqueExecutor.get());
    CHECK_RET(seqLensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyRefContiguous = const_cast<aclTensor*>(l0op::Contiguous(keyRef, uniqueExecutor.get()));
    CHECK_RET(keyRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueRefContiguous = const_cast<aclTensor*>(l0op::Contiguous(valueRef, uniqueExecutor.get()));
    CHECK_RET(valueRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto seqOffsetContiguous = seqOffsetOptional;
    if (seqOffsetOptional != nullptr) {
        seqOffsetContiguous = l0op::Contiguous(seqOffsetOptional, uniqueExecutor.get());
        CHECK_RET(seqOffsetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto gatherResult = l0op::GatherPaKvCache(keyCacheContiguous, valueCacheContiguous, blockTablesContiguous,
                                              seqLensContiguous, keyRefContiguous, valueRefContiguous,
                                              seqOffsetContiguous, cacheMode, isSeqLensCumsum, uniqueExecutor.get());
    CHECK_RET(std::get<0>(gatherResult) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyRefViewCopy = l0op::ViewCopy(std::get<0>(gatherResult), keyRef, uniqueExecutor.get());
    CHECK_RET(keyRefViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueRefViewCopy = l0op::ViewCopy(std::get<1>(gatherResult), valueRef, uniqueExecutor.get());
    CHECK_RET(valueRefViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnGatherPaKvCacheGetWorkspaceSize(const aclTensor* keyCache, const aclTensor* valueCache,
                                                 const aclTensor* blockTables, const aclTensor* seqLens,
                                                 aclTensor* keyRef, aclTensor* valueRef,
                                                 const aclTensor* seqOffsetOptional, char* cacheMode,
                                                 bool isSeqLensCumsum, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGatherPaKvCache,
                   DFX_IN(keyCache, valueCache, blockTables, seqLens, keyRef, valueRef, seqOffsetOptional, cacheMode,
                          isSeqLensCumsum),
                   DFX_OUT(keyRef, valueRef));

    CHECK_RET(CheckNullptr(keyCache, valueCache, blockTables, seqLens, keyRef, valueRef, workspaceSize, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    bool keyCacheEmpty = keyCache->GetViewShape().GetShapeSize() == 0;
    bool valueCacheEmpty = valueCache->GetViewShape().GetShapeSize() == 0;
    bool keyRefEmpty = keyRef->GetViewShape().GetShapeSize() == 0;
    bool valueRefEmpty = valueRef->GetViewShape().GetShapeSize() == 0;
    if (keyCacheEmpty || valueCacheEmpty || keyRefEmpty || valueRefEmpty) {
        *workspaceSize = 0;
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(CheckShape(keyCache, valueCache, blockTables, seqLens, keyRef, valueRef) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckDtypeValid(keyCache, valueCache, blockTables, seqLens, keyRef, valueRef, seqOffsetOptional,
                              cacheMode) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Norm和PA_NZ模式都支持非连续: PA_NZ非连续view若走ProcessContiguous,
    bool isPaNz = (cacheMode != nullptr && strcmp(cacheMode, "PA_NZ") == 0);
    // kernel无法处理的非连续形态(NZ非dim0 / ND尾轴)必须走ProcessContiguous物理连续化兜底,
    bool mustContiguous = NeedContiguousFallbackAny(keyCache, valueCache, keyRef, valueRef, isPaNz);

    bool isCacheNonContiguous = IsSupportNonContiguousCache(keyCache, valueCache);
    // ND模式cache内部轴(dim1/dim2)非连续、ref连续: kernel可零拷贝, 放行进ProcessNonContiguous。
    bool isNdCacheInner = IsSupportNonContiguousCacheNdInner(keyCache, valueCache, isPaNz);
    // ref(key/value)内部轴非连续(ND dim0/dim1, NZ dim0)、尾轴连续: kernel可零拷贝散写, 放行。
    bool isRefNonContig = IsSupportNonContiguousRef(keyRef, valueRef);
    if (!mustContiguous && (isCacheNonContiguous || isNdCacheInner || isRefNonContig)) {
        return ProcessNonContiguous(keyCache, valueCache, blockTables, seqLens, keyRef, valueRef, seqOffsetOptional,
                                    cacheMode, isSeqLensCumsum, workspaceSize, executor, uniqueExecutor);
    }
    return ProcessContiguous(keyCache, valueCache, blockTables, seqLens, keyRef, valueRef, seqOffsetOptional, cacheMode,
                             isSeqLensCumsum, workspaceSize, executor, uniqueExecutor);
}

aclnnStatus aclnnGatherPaKvCache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGatherPaKvCache);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "GatherPaKvCache launch aicore failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
