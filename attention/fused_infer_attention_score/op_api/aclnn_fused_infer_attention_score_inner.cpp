/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_infer_attention_score_inner.h"
#include "fused_infer_attention_score.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/fast_vector.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "acl/acl.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
const uint64_t INT4_NUMS_IN_INT32 = 8;
}

void TensorPreProcess(const aclTensorList *&tensorListKey, const aclTensorList *&tensorListValue)
{
    if (tensorListKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcess exit.");
        return;
    }
    if (tensorListValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcess exit.");
        return;
    }
    if ((*tensorListKey)[0]->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kv's from OriginalShape is completed.");
        return;
    }
    if ((*tensorListValue)[0]->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kv's from OriginalShape is completed.");
        return;
    }
    auto tempKey = const_cast<aclTensorList *>(tensorListKey);
    // 仅 INT4 输入场景（DT_INT32 打包存储）：将 viewShape/StorageShape/viewStrides 换算到 int4 口径（末维 ×8）
    // SetViewShape 会把 viewStrides 重算成连续值，需用保存的原始 stride 换算后覆盖以保留非连续信息
    for (uint64_t i = 0; i < tempKey->Size(); i++) {
        if ((*tempKey)[i] != nullptr) {
            op::Shape viewShape = (*tempKey)[i]->GetViewShape();
            auto viewShapeDim = viewShape.GetDimNum();
            auto origStrides = (*tempKey)[i]->GetViewStrides();
            if (viewShapeDim >= 1) {
                viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
            }
            (*tempKey)[i]->SetViewShape(viewShape);
            (*tempKey)[i]->SetDataType(DataType::DT_INT4);
            auto storageShape = (*tempKey)[i]->GetStorageShape();
            if (storageShape.GetDimNum() >= 1) {
                storageShape[storageShape.GetDimNum() - 1] =
                    storageShape[storageShape.GetDimNum() - 1] * INT4_NUMS_IN_INT32;
                (*tempKey)[i]->SetStorageShape(storageShape);
            }
            if (origStrides.size() > 0 && viewShapeDim >= 1) {
                for (uint64_t d = 0; d < origStrides.size() - 1; ++d) {
                    origStrides[d] = origStrides[d] * static_cast<int64_t>(INT4_NUMS_IN_INT32);
                }
                (*tempKey)[i]->SetViewStrides(origStrides);
            }
        }
    }

    auto tempValue = const_cast<aclTensorList *>(tensorListValue);
    for (uint64_t i = 0; i < tempValue->Size(); i++) {
        if ((*tempValue)[i] != nullptr) {
            op::Shape viewShape = (*tempValue)[i]->GetViewShape();
            auto viewShapeDim = viewShape.GetDimNum();
            auto origStrides = (*tempValue)[i]->GetViewStrides();
            if (viewShapeDim >= 1) {
                viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
            }
            (*tempValue)[i]->SetViewShape(viewShape);
            (*tempValue)[i]->SetDataType(DataType::DT_INT4);
            auto storageShapeV = (*tempValue)[i]->GetStorageShape();
            if (storageShapeV.GetDimNum() >= 1) {
                storageShapeV[storageShapeV.GetDimNum() - 1] =
                    storageShapeV[storageShapeV.GetDimNum() - 1] * INT4_NUMS_IN_INT32;
                (*tempValue)[i]->SetStorageShape(storageShapeV);
            }
            if (origStrides.size() > 0 && viewShapeDim >= 1) {
                for (uint64_t d = 0; d < origStrides.size() - 1; ++d) {
                    origStrides[d] = origStrides[d] * static_cast<int64_t>(INT4_NUMS_IN_INT32);
                }
                (*tempValue)[i]->SetViewStrides(origStrides);
            }
        }
    }

    OP_LOGD("The conversion of kv from int32 to int4 is completed.");
}

void PrefixTensorPreProcess(const aclTensor *&tensorKey, const aclTensor *&tensorValue)
{
    if (tensorKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcess exit.");
        return;
    }
    if (tensorValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcess exit..");
        return;
    }
    if (tensorKey->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kvPrefix's from OriginalShape is completed.");
        return;
    }
    if (tensorValue->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kvPrefix's from OriginalShape is completed.");
        return;
    }
    auto tempKey = const_cast<aclTensor *>(tensorKey);
    op::Shape viewKeyShape = tempKey->GetViewShape();
    auto viewKeyShapeDim = viewKeyShape.GetDimNum();
    auto keyOrigStrides = tempKey->GetViewStrides();
    viewKeyShape[viewKeyShapeDim - 1] = viewKeyShape[viewKeyShapeDim - 1] * INT4_NUMS_IN_INT32;
    tempKey->SetViewShape(viewKeyShape);
    tempKey->SetDataType(DataType::DT_INT4);
    auto keyStorageShape = tempKey->GetStorageShape();
    if (keyStorageShape.GetDimNum() >= 1) {
        keyStorageShape[keyStorageShape.GetDimNum() - 1] =
            keyStorageShape[keyStorageShape.GetDimNum() - 1] * INT4_NUMS_IN_INT32;
        tempKey->SetStorageShape(keyStorageShape);
    }
    if (keyOrigStrides.size() > 0 && viewKeyShapeDim >= 1) {
        for (uint64_t d = 0; d < keyOrigStrides.size() - 1; ++d) {
            keyOrigStrides[d] = keyOrigStrides[d] * static_cast<int64_t>(INT4_NUMS_IN_INT32);
        }
        tempKey->SetViewStrides(keyOrigStrides);
    }

    auto tempValue = const_cast<aclTensor *>(tensorValue);
    op::Shape viewValueShape = tempValue->GetViewShape();
    auto viewValueShapeDim = viewValueShape.GetDimNum();
    auto valueOrigStrides = tempValue->GetViewStrides();
    viewValueShape[viewValueShapeDim - 1] = viewValueShape[viewValueShapeDim - 1] * INT4_NUMS_IN_INT32;
    tempValue->SetViewShape(viewValueShape);
    tempValue->SetDataType(DataType::DT_INT4);
    auto valueStorageShape = tempValue->GetStorageShape();
    if (valueStorageShape.GetDimNum() >= 1) {
        valueStorageShape[valueStorageShape.GetDimNum() - 1] =
            valueStorageShape[valueStorageShape.GetDimNum() - 1] * INT4_NUMS_IN_INT32;
        tempValue->SetStorageShape(valueStorageShape);
    }
    if (valueOrigStrides.size() > 0 && viewValueShapeDim >= 1) {
        for (uint64_t d = 0; d < valueOrigStrides.size() - 1; ++d) {
            valueOrigStrides[d] = valueOrigStrides[d] * static_cast<int64_t>(INT4_NUMS_IN_INT32);
        }
        tempValue->SetViewStrides(valueOrigStrides);
    }

    OP_LOGD("The conversion of kvPrefix from int32 to int4 is completed.");
}

aclnnStatus FakeArray(const aclIntArray *inArray, aclTensor *&outTensor)
{
    OP_LOGD("start fake tensor");
    if (inArray != nullptr) {
        OP_LOGD("input array is not nullptr");
        int64_t size = static_cast<int64_t>(inArray->Size());
        std::vector<int64_t> shape = {size};
        outTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT64, nullptr, 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), nullptr);
        if (outTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc tensor failed");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    OP_LOGD("end fake tensor");
    return ACLNN_SUCCESS;
}

void FusedInferAttentionScoreProcessSoftmaxLse(bool softmaxLseFlag, const aclTensor *softmaxLse,
                                               const aclTensor *&tempTensor, const aclTensor *&placeHolder)
{
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void *>(&addr));
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
}

#ifdef __cplusplus
}
#endif

namespace {
enum class CacheStridePolicy {
    MAKE_CONTIGUOUS,
    KEEP_FAI_KV_DIM0,
    KEEP_ARCH22_MLA_KV_ROPE_DIM0,
};

bool IsCacheScene(const aclTensor *blockTableOptional)
{
    return blockTableOptional != nullptr && blockTableOptional->GetViewShape().GetShapeSize() != 0;
}

bool IsSupportedArch22MlaLayout(const std::string &inputLayout)
{
    return inputLayout == "BSH" || inputLayout == "BSND" || inputLayout == "BNSD" || inputLayout == "TND" ||
           inputLayout == "BSH_NBSD" || inputLayout == "BSND_NBSD" || inputLayout == "BNSD_NBSD" ||
           inputLayout == "TND_NTD";
}

bool HasQueryHeadDim(const aclTensor *tensor, int64_t numHeads, int64_t expectedHeadDim, bool hiddenLayout)
{
    if (tensor == nullptr || numHeads <= 0) {
        return false;
    }
    const auto shape = tensor->GetViewShape();
    const auto dimNum = shape.GetDimNum();
    if (dimNum < 3) {
        return false;
    }
    const int64_t lastDim = shape.GetDim(dimNum - 1);
    return hiddenLayout ? lastDim == numHeads * expectedHeadDim : lastDim == expectedHeadDim;
}

int64_t GetPageAttentionCacheHeadDim(const aclTensor *tensor, int64_t numKeyValueHeads)
{
    if (tensor == nullptr || numKeyValueHeads <= 0) {
        return -1;
    }
    const auto shape = tensor->GetViewShape();
    const auto dimNum = shape.GetDimNum();
    if (dimNum == 3) { // BnBsH
        const int64_t hiddenSize = shape.GetDim(2);
        return hiddenSize % numKeyValueHeads == 0 ? hiddenSize / numKeyValueHeads : -1;
    }
    if (dimNum == 4) { // BnNBsD
        return shape.GetDim(3);
    }
    if (dimNum == 5) { // NZ: Bn, N, D/16, Bs, 16
        return shape.GetDim(2) * shape.GetDim(4);
    }
    return -1;
}

bool IsArch22MlaD512RoutingCandidate(const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
                                     const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
                                     const char *inputLayout, int64_t numHeads, int64_t numKeyValueHeads,
                                     int64_t sparseMode, int64_t blockSize, bool hasUnsupportedFeature)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 || hasUnsupportedFeature ||
        query == nullptr || queryRopeOptional == nullptr || keyRopeOptional == nullptr || inputLayout == nullptr ||
        key == nullptr || value == nullptr || key->Size() != 1 || value->Size() != 1 || (*key)[0] == nullptr ||
        (*value)[0] == nullptr || numHeads <= 0 || numKeyValueHeads != 1 || numHeads % numKeyValueHeads != 0 ||
        blockSize <= 0) {
        return false;
    }

    const std::string inputLayoutStr(inputLayout);
    if (!IsSupportedArch22MlaLayout(inputLayoutStr) ||
        (sparseMode != 0 && sparseMode != 3 && sparseMode != 4 && sparseMode != 9)) {
        return false;
    }
    const bool hiddenLayout = inputLayoutStr == "BSH" || inputLayoutStr == "BSH_NBSD";

    const auto dtype = query->GetDataType();
    if ((dtype != DataType::DT_FLOAT16 && dtype != DataType::DT_BF16) || queryRopeOptional->GetDataType() != dtype ||
        (*key)[0]->GetDataType() != dtype || (*value)[0]->GetDataType() != dtype ||
        keyRopeOptional->GetDataType() != dtype) {
        return false;
    }

    return query->GetViewShape().GetShapeSize() > 0 && (*key)[0]->GetViewShape().GetShapeSize() > 0 &&
           (*value)[0]->GetViewShape().GetShapeSize() > 0 && queryRopeOptional->GetViewShape().GetShapeSize() > 0 &&
           keyRopeOptional->GetViewShape().GetShapeSize() > 0 && HasQueryHeadDim(query, numHeads, 512, hiddenLayout) &&
           HasQueryHeadDim(queryRopeOptional, numHeads, 64, hiddenLayout) &&
           GetPageAttentionCacheHeadDim((*key)[0], numKeyValueHeads) == 512 &&
           GetPageAttentionCacheHeadDim((*value)[0], numKeyValueHeads) == 512 &&
           GetPageAttentionCacheHeadDim(keyRopeOptional, numKeyValueHeads) == 64;
}

bool IsFAIRoutingCandidate(const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
                           const aclTensor *attenMaskOptional, const aclTensor *blockTableOptional,
                           const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
                           const aclTensor *learnableSinkOptional, const char *inputLayout, int64_t numHeads,
                           int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise)
{
    if (query == nullptr || key == nullptr || value == nullptr || key->Size() == 0 || value->Size() == 0 ||
        (*key)[0] == nullptr || (*value)[0] == nullptr || inputLayout == nullptr) {
        return false;
    }

    constexpr int64_t MAX_HEAD_DIM = 256;
    constexpr int64_t MAX_BLOCK_SIZE = 512;
    constexpr int64_t BLOCK_SIZE_ALIGNMENT = 16;
    constexpr int64_t NZ_INNER_BLOCK = 16;
    constexpr int64_t UNSUPPORTED_SINK_HEAD_DIM = 64;

    const std::string inputLayoutStr(inputLayout);
    const auto queryDtype = query->GetDataType();
    const auto queryShape = query->GetViewShape();
    const auto keyShape = (*key)[0]->GetViewShape();
    const auto valueShape = (*value)[0]->GetViewShape();
    if (queryShape.GetDimNum() <= 2) {
        return false;
    }
    const int64_t queryHeadDim = queryShape.GetDim(2);

    bool learnableSinkSupported = true;
    if (learnableSinkOptional != nullptr && inputLayoutStr == "TND" && queryHeadDim == UNSUPPORTED_SINK_HEAD_DIM &&
        learnableSinkOptional->GetDataType() == DataType::DT_BF16) {
        learnableSinkSupported = false;
    }
    const bool isRopeSplitMla = queryRopeOptional != nullptr && keyRopeOptional != nullptr;
    const bool sparseModeSupported = sparseMode == 0 || sparseMode == 3 || sparseMode == 4;
    const bool isMha = numKeyValueHeads == 0 || numHeads == numKeyValueHeads;
    const bool mhaConditions = isMha && !(queryDtype == DataType::DT_BF16 && innerPrecise == 1) &&
                               !(sparseMode == 0 && attenMaskOptional != nullptr);
    const bool nonMhaConditions = !isMha && innerPrecise == 0;
    if (inputLayoutStr != "TND" || !learnableSinkSupported || isRopeSplitMla || !sparseModeSupported ||
        (!mhaConditions && !nonMhaConditions)) {
        return false;
    }

    if (blockTableOptional == nullptr) {
        if (keyShape.GetDimNum() <= 2 || valueShape.GetDimNum() <= 2) {
            return false;
        }
        const int64_t keyHeadDim = keyShape.GetDim(2);
        const int64_t valueHeadDim = valueShape.GetDim(2);
        return queryHeadDim <= MAX_HEAD_DIM && keyHeadDim <= MAX_HEAD_DIM && valueHeadDim <= MAX_HEAD_DIM &&
               queryHeadDim == keyHeadDim && queryHeadDim == valueHeadDim;
    }
    if (keyShape.GetDimNum() == 3) {
        if (valueShape.GetDimNum() <= 2 || numKeyValueHeads == 0) {
            return false;
        }
        const int64_t blockSize = keyShape.GetDim(1);
        const int64_t keyHeadDim = keyShape.GetDim(2) / numKeyValueHeads;
        const int64_t valueHeadDim = valueShape.GetDim(2) / numKeyValueHeads;
        return queryHeadDim <= MAX_HEAD_DIM && keyHeadDim <= MAX_HEAD_DIM && valueHeadDim <= MAX_HEAD_DIM &&
               queryHeadDim == keyHeadDim && queryHeadDim == valueHeadDim && blockSize % BLOCK_SIZE_ALIGNMENT == 0 &&
               blockSize <= MAX_BLOCK_SIZE;
    }
    if (keyShape.GetDimNum() == 5) {
        if (valueShape.GetDimNum() <= 3) {
            return false;
        }
        const int64_t blockSize = keyShape.GetDim(3);
        const int64_t keyHeadDim = keyShape.GetDim(2) * NZ_INNER_BLOCK;
        const int64_t valueHeadDim = valueShape.GetDim(2) * NZ_INNER_BLOCK;
        const bool headDimSupported = queryHeadDim <= MAX_HEAD_DIM && keyHeadDim <= MAX_HEAD_DIM &&
                                      valueHeadDim <= MAX_HEAD_DIM && queryHeadDim == keyHeadDim &&
                                      queryHeadDim == valueHeadDim && queryHeadDim != 64 && queryHeadDim != 128;
        return headDimSupported && blockSize % BLOCK_SIZE_ALIGNMENT == 0 && blockSize <= MAX_BLOCK_SIZE;
    }
    return false;
}

bool GetAclTensorViewStrides(const aclTensor *tensor, int64_t *&stridesValue, uint64_t &stridesNum)
{
    stridesValue = nullptr;
    stridesNum = 0;
    auto retView = aclGetViewStrides(tensor, &stridesValue, &stridesNum);
    if (retView != ACL_SUCCESS || stridesValue == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclGetViewStrides failed.");
        delete[] stridesValue;
        stridesValue = nullptr;
        stridesNum = 0;
        return false;
    }
    return true;
}

bool IsFirstAxisOnlyNonContiguous(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr) {
        return true;
    }
    if (IsContiguous(tensor)) {
        return true;
    }

    auto viewShape = tensor->GetViewShape();
    int64_t dimNum = viewShape.GetDimNum();
    if (dimNum <= 1) {
        OP_LOGW("%s is non-contiguous with dim num[%ld] <= 1, it will be forced to be contiguous.", name, dimNum);
        return false;
    }

    int64_t *viewStrides = nullptr;
    uint64_t stridesNum = 0;
    if (!GetAclTensorViewStrides(tensor, viewStrides, stridesNum)) {
        OP_LOGW("Failed to get view strides for %s, it will be forced to be contiguous.", name);
        return false;
    }
    if (stridesNum < static_cast<uint64_t>(dimNum)) {
        OP_LOGW("%s view strides num[%lu] is less than view shape dim num[%ld], it will be forced to be contiguous.",
                name, stridesNum, dimNum);
        delete[] viewStrides;
        return false;
    }
    if (viewShape.GetDim(0) > 1 && viewStrides[0] <= 0) {
        OP_LOGW("%s has invalid dim0 stride[%ld] for dim0 size[%ld], it will be forced to be contiguous.", name,
                viewStrides[0], viewShape.GetDim(0));
        delete[] viewStrides;
        return false;
    }

    bool isFirstAxisOnlyNonContiguous = true;
    int64_t expectedStride = 1;
    for (int64_t dim = dimNum - 1; dim >= 1; --dim) {
        if (viewShape.GetDim(dim) != 1 && viewStrides[dim] != expectedStride) {
            OP_LOGW("%s is non-contiguous at axis[%ld] (0-based): actual stride[%ld], expected stride[%ld], "
                    "shape[%s]. It will be forced to be contiguous.",
                    name, dim, viewStrides[dim], expectedStride, op::ToString(viewShape).GetString());
            isFirstAxisOnlyNonContiguous = false;
            break;
        }
        expectedStride *= viewShape.GetDim(dim);
    }

    delete[] viewStrides;
    return isFirstAxisOnlyNonContiguous;
}

void SetTensorFormatToND(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    auto mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
}

aclnnStatus MakeTensorListContiguous(const aclTensorList *&tensorList, const char *name, aclOpExecutor *executor)
{
    if (tensorList == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensorList is nullptr.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    std::vector<const aclTensor *> tensors;
    tensors.reserve(tensorList->Size());
    for (uint64_t i = 0; i < tensorList->Size(); ++i) {
        auto tensor = (*tensorList)[i];
        if (tensor == nullptr) {
            tensors.emplace_back(nullptr);
            continue;
        }
        auto contiguousTensor = l0op::Contiguous(tensor, executor);
        if (contiguousTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try make %s[%lu] contiguous failed.", name, i);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        tensors.emplace_back(contiguousTensor);
    }

    auto contiguousList = executor->AllocTensorList(tensors.data(), tensors.size());
    if (contiguousList == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc contiguous %s tensorList failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    tensorList = contiguousList;

    for (uint64_t i = 0; i < tensorList->Size(); ++i) {
        auto tensor = (*tensorList)[i];
        if (tensor != nullptr) {
            SetTensorFormatToND(tensor);
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus MakeTensorContiguous(const aclTensor *&tensor, const char *name, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }

    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try make %s contiguous failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    SetTensorFormatToND(tensor);
    return ACLNN_SUCCESS;
}

const aclTensor *CreateStrideAwareView(const aclTensor *tensor, const char *name, aclOpExecutor *executor)
{
    if (tensor == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Tensor or executor is nullptr when creating stride view for %s.", name);
        return nullptr;
    }

    auto strideView = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                           tensor->GetViewStrides(), tensor->GetViewOffset());
    if (strideView == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try create stride view for %s failed.", name);
        return nullptr;
    }
    const_cast<aclTensor *>(strideView)->SetStorageShape(tensor->GetViewShape());
    return strideView;
}

aclnnStatus NormalizeDim0CacheTensorList(const aclTensorList *&tensorList, const char *name,
                                         bool completeStrideMetadata, aclOpExecutor *executor)
{
    if (tensorList == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensorList is nullptr.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    std::vector<const aclTensor *> tensors;
    tensors.reserve(tensorList->Size());
    for (uint64_t i = 0; i < tensorList->Size(); ++i) {
        auto tensor = (*tensorList)[i];
        if (tensor == nullptr) {
            tensors.emplace_back(nullptr);
            continue;
        }

        const aclTensor *normalizedTensor = nullptr;
        std::string itemName = std::string(name) + "[" + std::to_string(i) + "]";
        if (IsContiguous(tensor)) {
            auto contiguousTensor = l0op::Contiguous(tensor, executor);
            if (contiguousTensor == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try normalize contiguous %s failed.", itemName.c_str());
                return ACLNN_ERR_INNER_NULLPTR;
            }
            normalizedTensor = completeStrideMetadata ?
                                   CreateStrideAwareView(contiguousTensor, itemName.c_str(), executor) :
                                   contiguousTensor;
        } else if (IsFirstAxisOnlyNonContiguous(tensor, itemName.c_str())) {
            normalizedTensor = CreateStrideAwareView(tensor, itemName.c_str(), executor);
        } else {
            auto contiguousTensor = l0op::Contiguous(tensor, executor);
            if (contiguousTensor == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try make %s contiguous failed.", itemName.c_str());
                return ACLNN_ERR_INNER_NULLPTR;
            }
            normalizedTensor = completeStrideMetadata ?
                                   CreateStrideAwareView(contiguousTensor, itemName.c_str(), executor) :
                                   contiguousTensor;
        }
        if (normalizedTensor == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
        SetTensorFormatToND(normalizedTensor);
        tensors.emplace_back(normalizedTensor);
    }

    auto normalizedList = executor->AllocTensorList(tensors.data(), tensors.size());
    if (normalizedList == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc normalized %s tensorList failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    tensorList = normalizedList;
    return ACLNN_SUCCESS;
}

aclnnStatus NormalizeDim0CacheTensor(const aclTensor *&tensor, const char *name, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }

    const aclTensor *normalizedTensor = nullptr;
    if (IsContiguous(tensor)) {
        auto contiguousTensor = l0op::Contiguous(tensor, executor);
        if (contiguousTensor != nullptr) {
            normalizedTensor = CreateStrideAwareView(contiguousTensor, name, executor);
        }
    } else if (IsFirstAxisOnlyNonContiguous(tensor, name)) {
        normalizedTensor = CreateStrideAwareView(tensor, name, executor);
    } else {
        auto contiguousTensor = l0op::Contiguous(tensor, executor);
        if (contiguousTensor != nullptr) {
            normalizedTensor = CreateStrideAwareView(contiguousTensor, name, executor);
        }
    }
    if (normalizedTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try normalize %s failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    SetTensorFormatToND(normalizedTensor);
    tensor = normalizedTensor;
    return ACLNN_SUCCESS;
}

CacheStridePolicy DecideCacheStridePolicy(bool supportTensorV2, bool isCacheScene, bool isFAIRoutingCandidate,
                                          bool isArch22MlaRoutingCandidate)
{
    if (!supportTensorV2 || !isCacheScene) {
        return CacheStridePolicy::MAKE_CONTIGUOUS;
    }
    if (isArch22MlaRoutingCandidate) {
        return CacheStridePolicy::KEEP_ARCH22_MLA_KV_ROPE_DIM0;
    }
    if (isFAIRoutingCandidate) {
        return CacheStridePolicy::KEEP_FAI_KV_DIM0;
    }
    return CacheStridePolicy::MAKE_CONTIGUOUS;
}

aclnnStatus ProcessCacheForL0Input(const aclTensorList *&key, const aclTensorList *&value, const aclTensor *&keyRope,
                                   CacheStridePolicy policy, aclOpExecutor *executor)
{
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (key == nullptr || value == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key or value tensorList is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (key->Size() != value->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "key tensorList size[%lu] should be equal to value tensorList size[%lu].",
                key->Size(), value->Size());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (policy == CacheStridePolicy::KEEP_FAI_KV_DIM0 || policy == CacheStridePolicy::KEEP_ARCH22_MLA_KV_ROPE_DIM0) {
        const bool completeStrideMetadata = policy == CacheStridePolicy::KEEP_ARCH22_MLA_KV_ROPE_DIM0;
        CHECK_RET(NormalizeDim0CacheTensorList(key, "key", completeStrideMetadata, executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(NormalizeDim0CacheTensorList(value, "value", completeStrideMetadata, executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        if (policy == CacheStridePolicy::KEEP_ARCH22_MLA_KV_ROPE_DIM0) {
            CHECK_RET(NormalizeDim0CacheTensor(keyRope, "keyRope", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        } else {
            CHECK_RET(MakeTensorContiguous(keyRope, "keyRope", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
        return ACLNN_SUCCESS;
    }

    CHECK_RET(MakeTensorListContiguous(key, "key", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorListContiguous(value, "value", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(keyRope, "keyRope", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus InnerFusedInferAttentionScoreGetWorkspaceSizeImpl(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional, const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, int64_t numHeads, double scaleValue,
    int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode,
    int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode,
    int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType, int64_t outDtype,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor,
    bool enableArch22MlaDim0Stride)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(query, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(key, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(value, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(attentionOut, return ACLNN_ERR_PARAM_NULLPTR);
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (softmaxLseFlag) {
        OP_CHECK_NULL(softmaxLse, return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (inputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "inputLayout is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    const bool isFAIRoutingCandidate = IsFAIRoutingCandidate(
        query, key, value, attenMaskOptional, blockTableOptional, queryRopeOptional, keyRopeOptional,
        learnableSinkOptional, inputLayout, numHeads, numKeyValueHeads, sparseMode, innerPrecise);
    const bool hasUnsupportedMlaFeature =
        pseShiftOptional != nullptr || queryPaddingSizeOptional != nullptr || kvPaddingSizeOptional != nullptr ||
        keySharedPrefixOptional != nullptr || valueSharedPrefixOptional != nullptr ||
        learnableSinkOptional != nullptr || deqScale1Optional != nullptr || quantScale1Optional != nullptr ||
        deqScale2Optional != nullptr || quantScale2Optional != nullptr || quantOffset2Optional != nullptr ||
        antiquantScaleOptional != nullptr || antiquantOffsetOptional != nullptr ||
        keyAntiquantScaleOptional != nullptr || keyAntiquantOffsetOptional != nullptr ||
        valueAntiquantScaleOptional != nullptr || valueAntiquantOffsetOptional != nullptr ||
        keyRopeAntiquantScaleOptional != nullptr || dequantScaleQueryOptional != nullptr ||
        qStartIdxOptional != nullptr || kvStartIdxOptional != nullptr || antiquantMode != 0 || keyAntiquantMode != 0 ||
        valueAntiquantMode != 0 || queryQuantMode != 0;
    const bool isArch22MlaRoutingCandidate =
        enableArch22MlaDim0Stride &&
        IsArch22MlaD512RoutingCandidate(query, key, value, queryRopeOptional, keyRopeOptional, inputLayout, numHeads,
                                        numKeyValueHeads, sparseMode, blockSize, hasUnsupportedMlaFeature);
    const bool supportTensorV2 = NnopbaseSupportTensorV2 != nullptr;
    const auto cacheStridePolicy = DecideCacheStridePolicy(supportTensorV2, IsCacheScene(blockTableOptional),
                                                           isFAIRoutingCandidate, isArch22MlaRoutingCandidate);
    if (!supportTensorV2) {
        OP_LOGW("Current opbase does not support TensorV2, make key, value and keyRope contiguous.");
    }

    if (attentionOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor *processedQuery = query;
    const aclTensorList *processedKey = key;
    const aclTensorList *processedValue = value;
    const aclTensor *processedPseShift = pseShiftOptional;
    const aclTensor *processedAttenMask = attenMaskOptional;
    const aclTensor *processedBlockTable = blockTableOptional;
    const aclTensor *processedKeySharedPrefix = keySharedPrefixOptional;
    const aclTensor *processedValueSharedPrefix = valueSharedPrefixOptional;
    const aclTensor *processedQueryRope = queryRopeOptional;
    const aclTensor *processedKeyRope = keyRopeOptional;
    const aclTensor *processedDeqScale1 = deqScale1Optional;
    const aclTensor *processedQuantScale1 = quantScale1Optional;
    const aclTensor *processedDeqScale2 = deqScale2Optional;
    const aclTensor *processedQuantScale2 = quantScale2Optional;
    const aclTensor *processedQuantOffset2 = quantOffset2Optional;
    const aclTensor *processedAntiquantScale = antiquantScaleOptional;
    const aclTensor *processedAntiquantOffset = antiquantOffsetOptional;
    const aclTensor *processedQueryPaddingSize = queryPaddingSizeOptional;
    const aclTensor *processedKvPaddingSize = kvPaddingSizeOptional;
    const aclTensor *processedKeyAntiquantScale = keyAntiquantScaleOptional;
    const aclTensor *processedKeyAntiquantOffset = keyAntiquantOffsetOptional;
    const aclTensor *processedValueAntiquantScale = valueAntiquantScaleOptional;
    const aclTensor *processedValueAntiquantOffset = valueAntiquantOffsetOptional;
    const aclTensor *processedKeyRopeAntiquantScale = keyRopeAntiquantScaleOptional;
    const aclTensor *processedDequantScaleQuery = dequantScaleQueryOptional;
    const aclTensor *processedLearnableSink = learnableSinkOptional;

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(MakeTensorContiguous(processedQuery, "query", l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedPseShift, "pseShift", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedAttenMask, "attenMask", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedBlockTable, "blockTable", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKeySharedPrefix, "keySharedPrefix", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedValueSharedPrefix, "valueSharedPrefix", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQueryRope, "queryRope", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedDeqScale1, "deqScale1", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQuantScale1, "quantScale1", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedDeqScale2, "deqScale2", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQuantScale2, "quantScale2", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQuantOffset2, "quantOffset2", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedAntiquantScale, "antiquantScale", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedAntiquantOffset, "antiquantOffset", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQueryPaddingSize, "queryPaddingSize", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKvPaddingSize, "kvPaddingSize", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKeyAntiquantScale, "keyAntiquantScale", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKeyAntiquantOffset, "keyAntiquantOffset", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedValueAntiquantScale, "valueAntiquantScale", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedValueAntiquantOffset, "valueAntiquantOffset", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(
        MakeTensorContiguous(processedKeyRopeAntiquantScale, "keyRopeAntiquantScale", l0Executor) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedDequantScaleQuery, "dequantScaleQuery", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedLearnableSink, "learnableSink", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    auto cacheProcessRet =
        ProcessCacheForL0Input(processedKey, processedValue, processedKeyRope, cacheStridePolicy, l0Executor);
    CHECK_RET(cacheProcessRet == ACLNN_SUCCESS, cacheProcessRet);

    auto l0Out = l0op::FusedInferAttentionScore(
        processedQuery, processedKey, processedValue, processedPseShift, processedAttenMask, actualSeqLengthsOptional,
        actualSeqLengthsKvOptional, processedDeqScale1, processedQuantScale1, processedDeqScale2, processedQuantScale2,
        processedQuantOffset2, processedAntiquantScale, processedAntiquantOffset, processedBlockTable,
        processedQueryPaddingSize, processedKvPaddingSize, processedKeyAntiquantScale, processedKeyAntiquantOffset,
        processedValueAntiquantScale, processedValueAntiquantOffset, processedKeySharedPrefix,
        processedValueSharedPrefix, actualSharedPrefixLenOptional, processedQueryRope, processedKeyRope,
        processedKeyRopeAntiquantScale, processedDequantScaleQuery, processedLearnableSink, qStartIdxOptional,
        kvStartIdxOptional, numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode,
        innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryQuantMode,
        pseType, outDtype, attentionOut, softmaxLse, l0Executor);

    auto l0AttentionOut = std::get<0>(l0Out);
    auto l0SoftmaxLse = std::get<1>(l0Out);
    if (l0AttentionOut == nullptr || l0SoftmaxLse == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto attentionViewCopy = l0op::ViewCopy(l0AttentionOut, attentionOut, l0Executor);
    if (attentionViewCopy == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (softmaxLseFlag) {
        auto softmaxLseViewCopy = l0op::ViewCopy(l0SoftmaxLse, softmaxLse, l0Executor);
        CHECK_RET(softmaxLseViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus InnerFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional, const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, int64_t numHeads, double scaleValue,
    int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode,
    int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode,
    int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType, int64_t outDtype,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // V1-V3 retain the shared Inner behavior without the V4-only arch22 MLA stride extension.
    return InnerFusedInferAttentionScoreGetWorkspaceSizeImpl(
        query, key, value, pseShiftOptional, attenMaskOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
        deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional, quantOffset2Optional,
        antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional, queryPaddingSizeOptional,
        kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional, valueAntiquantScaleOptional,
        valueAntiquantOffsetOptional, keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenOptional,
        queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional,
        learnableSinkOptional, qStartIdxOptional, kvStartIdxOptional, numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType, outDtype, attentionOut, softmaxLse,
        workspaceSize, executor, false);
}

aclnnStatus InnerFusedInferAttentionScoreV4GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional, const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, int64_t numHeads, double scaleValue,
    int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode,
    int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode,
    int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType, int64_t outDtype,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // Only V4 opts in to preserving arch22 MLA D512 K/V/KeyRope dim0 strides.
    return InnerFusedInferAttentionScoreGetWorkspaceSizeImpl(
        query, key, value, pseShiftOptional, attenMaskOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
        deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional, quantOffset2Optional,
        antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional, queryPaddingSizeOptional,
        kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional, valueAntiquantScaleOptional,
        valueAntiquantOffsetOptional, keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenOptional,
        queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional,
        learnableSinkOptional, qStartIdxOptional, kvStartIdxOptional, numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType, outDtype, attentionOut, softmaxLse,
        workspaceSize, executor, true);
}

aclnnStatus InnerFusedInferAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          const aclrtStream stream)
{
    L2_DFX_PHASE_2(InnerFusedInferAttentionScore);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
