/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_minimax_sparse_attention_split_kv.h"

#include "minimax_sparse_attention_split_kv.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include <acl/acl.h>
#include <cstring>
#include <tuple>
#include <vector>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static bool CheckQkvDataType(const aclTensor *query, const aclTensor *key, const aclTensor *value)
{
    const DataType qDtype = query->GetDataType();
    const bool supported = qDtype == DataType::DT_FLOAT16 || qDtype == DataType::DT_BF16;
    if (!supported) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query dtype must be FLOAT16 or BF16, got %d.", static_cast<int>(qDtype));
        return false;
    }
    if (key->GetDataType() != qDtype || value->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query, key, and value must use the same dtype.");
        return false;
    }
    return true;
}

// 检测当前 NPU 是否为 A2 (ascend910b / ascend910_93)。
// A5 (ascend950) 返回 false。
// 用于区分 A2 专用的 aclnn 逻辑（CreateView stride 保留、连续 KV dummy blockTable）。
static bool IsA2Platform()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    return socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93;
}

static aclnnStatus MakeContiguous(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                  const aclTensor *&blockTable, const aclTensor *&k2qRowPtr,
                                  const aclTensor *&k2qQIndices, const aclTensor *&k2qSlotIndices,
                                  const aclTensor *&actualSeqLengths, const aclTensor *&actualSeqLengthsKv,
                                  aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (blockTable != nullptr) {
        blockTable = l0op::Contiguous(blockTable, executor);
        CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    k2qRowPtr = l0op::Contiguous(k2qRowPtr, executor);
    CHECK_RET(k2qRowPtr != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    k2qQIndices = l0op::Contiguous(k2qQIndices, executor);
    CHECK_RET(k2qQIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    k2qSlotIndices = l0op::Contiguous(k2qSlotIndices, executor);
    CHECK_RET(k2qSlotIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    actualSeqLengths = l0op::Contiguous(actualSeqLengths, executor);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    actualSeqLengthsKv = l0op::Contiguous(actualSeqLengthsKv, executor);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

} // namespace

// aclnnMinimaxSparseAttentionSplitKvGetWorkspaceSize: aclnn 两阶段 API 的第一阶段（规划阶段）。
// 功能：
//   1. 校验所有输入/输出指针非空；
//   2. 创建 executor；
//   3. 调用 MakeContiguous 保证输入连续；
//   4. 调用 L0 层 MinimaxSparseAttentionSplitKv 构建算子计算图；
//   5. 通过 ViewCopy 将内部输出拷贝到用户提供的 attentionOut；
//   6. 计算并返回 workspace 大小，同时输出 executor 供第二阶段使用。
__attribute__((visibility("default"))) aclnnStatus aclnnMinimaxSparseAttentionSplitKvGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockTable,
    const aclTensor *k2qRowPtr, const aclTensor *k2qQIndices, const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengths, const aclTensor *actualSeqLengthsKv, int64_t numKeyValueHeads, double scaleValue,
    int64_t blockSize, int64_t topK, int64_t innerPrecise, bool softmaxLseFlag, const char *inputLayout,
    aclTensor *attentionOut, aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (!CheckQkvDataType(query, key, value)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(k2qRowPtr != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qQIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qSlotIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // softmaxLse is optional (A2 only, nullptr when A5 or not requested)
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (softmaxLseFlag) {
        CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (innerPrecise != 0 && innerPrecise != 1 && innerPrecise != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "innerPrecise must be 0, 1 or 4, got %ld.", innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }
    DataType qDtype = query->GetDataType();
    DataType kDtype = key->GetDataType();
    DataType vDtype = value->GetDataType();
    auto isQkvOk = [](DataType d) { return d == DataType::DT_BF16 || d == DataType::DT_FLOAT8_E4M3FN; };
    if (!isQkvOk(qDtype) || !isQkvOk(kDtype) || !isQkvOk(vDtype)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MinimaxSparseAttentionSplitKv Q/K/V only support BF16 or FLOAT8_E4M3FN.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype != kDtype || qDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MinimaxSparseAttentionSplitKv Q/K/V dtype must be consistent.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (attentionOut->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MinimaxSparseAttentionSplitKv attentionOut must be BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype == DataType::DT_FLOAT8_E4M3FN && innerPrecise != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MinimaxSparseAttentionSplitKv fp8 path uses FP32 O_partial only "
                "(innerPrecise=4); got %ld. innerPrecise=0/1 are not implemented.",
                innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const char *layout = (inputLayout == nullptr || inputLayout[0] == '\0') ? "TND" : inputLayout;
    if (strcmp(layout, "TND") != 0 && strcmp(layout, "BNSD") != 0 && strcmp(layout, "BSND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputLayout must be TND, BNSD or BSND, got %s.", layout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    L2_DFX_PHASE_1(
        aclnnMinimaxSparseAttentionSplitKv,
        DFX_IN(query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths,
               actualSeqLengthsKv, numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise, softmaxLseFlag, layout),
        DFX_OUT(attentionOut, softmaxLse));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    aclnnStatus ret = MakeContiguous(query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                     actualSeqLengths, actualSeqLengthsKv, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // A2 专用: Torch ACL tensors 经 MakeContiguous 后 StorageShape 可能被压平,
    // 但 A2 tiling 需要通过 GetInputStride() 读取 stride 信息来计算
    // queryTokenStride / keyBlockStride 等。CreateView 以零拷贝方式重建
    // 逻辑视图, 保留 ViewShape + ViewStrides。
    // A5 不读取 stride, 直接使用原 tensor 即可 (GetOriginShape 返回正确逻辑形状)。
    const bool isA2 = IsA2Platform();
    const aclTensor *queryArg = query;
    const aclTensor *keyArg = key;
    const aclTensor *valueArg = value;
    if (isA2) {
        queryArg = executorImpl->CreateView(query, query->GetViewShape(), query->GetStorageShape(),
                                            query->GetViewStrides(), query->GetViewOffset());
        keyArg = executorImpl->CreateView(key, key->GetViewShape(), key->GetStorageShape(), key->GetViewStrides(),
                                          key->GetViewOffset());
        valueArg = executorImpl->CreateView(value, value->GetViewShape(), value->GetStorageShape(),
                                            value->GetViewStrides(), value->GetViewOffset());
        CHECK_RET(queryArg != nullptr && keyArg != nullptr && valueArg != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    const aclTensor *tempTensor = nullptr;
    const aclTensor *lsePlaceHolder = nullptr;
    int64_t dummyLseAddr = 0xff;
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void *>(&dummyLseAddr));
        lsePlaceHolder = tempTensor;
    } else {
        lsePlaceHolder = softmaxLse;
    }
    if (lsePlaceHolder == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "softmaxLse placeholder is nullptr.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // A2 专用: 连续 KV 模式 (blockSize==0) 下 blockTable 为 nullptr。
    // GE 会压缩缺失的可选输入导致后续输入位置错位, 用 actualSeqLengths 占位
    // 保持输入位置稳定 (kernel 不会读取此 dummy blockTable)。
    // A5 不支持连续 KV 模式, blockTable 必须非空。
    if (isA2 && blockTable == nullptr) {
        blockTable = actualSeqLengths;
    }

    auto outputs = l0op::MinimaxSparseAttentionSplitKv(
        queryArg, keyArg, valueArg, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths,
        actualSeqLengthsKv, numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise, softmaxLseFlag, layout,
        attentionOut, lsePlaceHolder, executorImpl);
    auto output = std::get<0>(outputs);
    auto lseOut = std::get<1>(outputs);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(const_cast<aclTensor *>(tempTensor));
    }
    if (output == nullptr || lseOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MinimaxSparseAttentionSplitKv returned nullptr output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(output, attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (softmaxLseFlag) {
        auto lseViewCopy = l0op::ViewCopy(lseOut, softmaxLse, executorImpl);
        CHECK_RET(lseViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// aclnnMinimaxSparseAttentionSplitKv: aclnn 两阶段 API 的第二阶段（执行阶段）。
// 功能：在指定的 aclrtStream 上启动算子执行，复用第一阶段生成的 executor 与 workspace。
__attribute__((visibility("default"))) aclnnStatus aclnnMinimaxSparseAttentionSplitKv(void *workspace,
                                                                                      uint64_t workspaceSize,
                                                                                      aclOpExecutor *executor,
                                                                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMinimaxSparseAttentionSplitKv);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
