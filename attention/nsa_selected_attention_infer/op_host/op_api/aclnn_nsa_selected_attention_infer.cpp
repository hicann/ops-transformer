/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_nsa_selected_attention_infer.cpp
 * \brief
 */

#include <memory>

#include "nsa_selected_attention_infer.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "base/err_msg.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "aclnn_nsa_selected_attention_infer.h"

using namespace op;
using namespace ge;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_0 = 0;
static const int64_t HEAD_DIM_MAX = 192;
static const int64_t HEAD_DIM_V_MAX = 128;
static const int64_t GROUP_LIMITS = 16;

void ReportInvalidParameter(const std::string& reason)
{
    const std::string message = "aclnnNsaSelectedAttentionInfer: " + reason;
    OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", message.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1001", (std::vector<const char*>{"message"}),
                              (std::vector<const char*>{message.c_str()}));
}

struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dk;
    int64_t dv;
    int64_t numtokens;
};

enum class InputLayout {
    BSND,
    BSH,
    TND
};

struct NSAShapeInfo {
    AxesInfo axes;
    InputLayout inputLayout;
    uint64_t dimNum = 0;
};

__attribute__((visibility("default"))) aclnnStatus aclnnNsaSelectedAttentionInferGetMaxWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* topkIndices,
    const aclTensor* attenMaskOptional, const aclTensor* blockTableOptional, const aclIntArray* actualQSeqLenOptional,
    const aclIntArray* actualKvSeqLenOptional, char* layoutOptional, int64_t numHeads, int64_t numKeyValueHeads,
    int64_t selectBlockSize, int64_t selectBlockCount, int64_t pageBlockSize, double scaleValue, int64_t sparseMode,
    aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor);

void AnalysisAxisForBSH(const Shape& qShape, const Shape& kShape, const Shape& vShape, NSAShapeInfo& shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSH;
    shapeInfo.axes.b = qShape[DIM_NUM_0];
    shapeInfo.axes.d = qShape[DIM_NUM_2] / shapeInfo.axes.n1;
    shapeInfo.axes.dk = kShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.dv = vShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.s1 = qShape[DIM_NUM_1];
    shapeInfo.axes.s2 = vShape[DIM_NUM_1] * vShape[DIM_NUM_0];
}

void AnalysisAxisForBSND(const Shape& qShape, const Shape& kShape, const Shape& vShape, NSAShapeInfo& shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSND;
    shapeInfo.axes.b = qShape[DIM_NUM_0];
    shapeInfo.axes.d = qShape[DIM_NUM_3];
    shapeInfo.axes.dk = kShape[DIM_NUM_3];
    shapeInfo.axes.dv = vShape[DIM_NUM_3];
    shapeInfo.axes.s1 = qShape[DIM_NUM_1];
    shapeInfo.axes.s2 = vShape[DIM_NUM_1] * vShape[DIM_NUM_0];
}

void AnalysisAxisForTND(const Shape& qShape, const Shape& kShape, const Shape& vShape, NSAShapeInfo& shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::TND;
    shapeInfo.axes.numtokens = qShape[DIM_NUM_0];
    shapeInfo.axes.d = qShape[DIM_NUM_2];
    shapeInfo.axes.dk = kShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.dv = vShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.s2 = vShape[DIM_NUM_1] * vShape[DIM_NUM_0];
}

aclnnStatus AnalysisAxisShape(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                              const char* inputLayout, NSAShapeInfo& shapeInfo)
{
    Shape qShape = query->GetViewShape();
    Shape kShape = key->GetViewShape();
    Shape vShape = value->GetViewShape();
    shapeInfo.dimNum = qShape.GetDimNum();

    // 记录轴的长度 b, n2, g, s1, s2, d
    // H1等于N1*D, H2等于N2*D
    // N1等于g*N2
    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "BSH") {
        AnalysisAxisForBSH(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_4 && inputLayoutStr == "BSND") {
        AnalysisAxisForBSND(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "TND") {
        AnalysisAxisForTND(qShape, kShape, vShape, shapeInfo);
    } else {
        ReportInvalidParameter("query rank " + std::to_string(shapeInfo.dimNum) + " is invalid for layout " +
                               inputLayoutStr + "; BSH/TND require rank 3 and BSND requires rank 4");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d != shapeInfo.axes.dk) {
        ReportInvalidParameter("query and key head dimensions must match; got query=" +
                               std::to_string(shapeInfo.axes.d) + ", key=" + std::to_string(shapeInfo.axes.dk));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d < shapeInfo.axes.dv) {
        ReportInvalidParameter("key head dimension must be at least the value head dimension; got key=" +
                               std::to_string(shapeInfo.axes.dk) + ", value=" + std::to_string(shapeInfo.axes.dv));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus DtypeCheck(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                       const aclTensor* topkIndices)
{
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    auto topkDtype = topkIndices->GetDataType();
    if (qDtype != kDtype || kDtype != vDtype) {
        ReportInvalidParameter(std::string("query, key and value must use the same dtype; got query=") +
                               op::ToString(qDtype).GetString() + ", key=" + op::ToString(kDtype).GetString() +
                               ", value=" + op::ToString(vDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype != ge::DT_FLOAT16 && qDtype != ge::DT_BF16) {
        ReportInvalidParameter(std::string("query, key and value must use FLOAT16 or BFLOAT16; got ") +
                               op::ToString(qDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (topkDtype != ge::DT_INT32) {
        ReportInvalidParameter(std::string("topkIndices dtype must be INT32; got ") +
                               op::ToString(topkDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// topkIndices: BSH/BSND为[B, S, numKeyValueHeads, selectBlockCount]，TND为[T, numKeyValueHeads, selectBlockCount]
aclnnStatus TopkIndicesShapeCheck(const aclTensor* topkIndices, const char* inputLayout)
{
    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    uint64_t dimNumExpected = (inputLayoutStr == "TND") ? DIM_NUM_3 : DIM_NUM_4;
    uint64_t dimNum = topkIndices->GetViewShape().GetDimNum();
    if (dimNum != dimNumExpected) {
        ReportInvalidParameter("topkIndices rank must be " + std::to_string(dimNumExpected) + " for layout " +
                               inputLayoutStr + "; got " + std::to_string(dimNum));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AnalysisInputNsa(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                             const aclTensor* topkIndices, char* inputLayout, NSAShapeInfo& shapeInfo,
                             const aclIntArray* actualQSeqLenOptional = nullptr,
                             const aclIntArray* actualSeqKvLenOptional = nullptr)
{
    CHECK_RET(DtypeCheck(query, key, value, topkIndices) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(TopkIndicesShapeCheck(topkIndices, inputLayout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(AnalysisAxisShape(query, key, value, inputLayout, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.d != HEAD_DIM_MAX) {
        ReportInvalidParameter("query and key head dimensions must be 192; got " + std::to_string(shapeInfo.axes.d));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.dv != HEAD_DIM_V_MAX) {
        ReportInvalidParameter("value head dimension must be 128; got " + std::to_string(shapeInfo.axes.dv));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if ((actualQSeqLenOptional != nullptr && actualSeqKvLenOptional != nullptr) &&
        (actualQSeqLenOptional->Size() > 3 * 1024 || actualSeqKvLenOptional->Size() > 3 * 1024)) {
        ReportInvalidParameter("actualQSeqLenOptional and actualKvSeqLenOptional sizes must not exceed 3072; got Q=" +
                               std::to_string(actualQSeqLenOptional->Size()) +
                               ", KV=" + std::to_string(actualSeqKvLenOptional->Size()));
        return ACLNN_ERR_PARAM_INVALID;
    }

    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    if (inputLayoutStr == "TND") {
        if (shapeInfo.axes.numtokens == 0 || shapeInfo.axes.s2 == 0) {
            ReportInvalidParameter("TND query token count and value cache sequence size must be nonzero; got " +
                                   std::to_string(shapeInfo.axes.numtokens) + ", " + std::to_string(shapeInfo.axes.s2));
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (shapeInfo.axes.b == 0 || shapeInfo.axes.s1 == 0 || shapeInfo.axes.s2 == 0) {
            ReportInvalidParameter("query batch size, query sequence size and value cache sequence size must be "
                                   "nonzero; got " +
                                   std::to_string(shapeInfo.axes.b) + ", " + std::to_string(shapeInfo.axes.s1) + ", " +
                                   std::to_string(shapeInfo.axes.s2));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ContiguousTensors(const aclTensor*& query, const aclTensor*& key, const aclTensor*& value,
                              const aclTensor*& topkIndices, const aclTensor*& blockTable,
                              const aclTensor*& attenMaskOptional, aclOpExecutor* executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    topkIndices = l0op::Contiguous(topkIndices, executor);
    CHECK_RET(topkIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    blockTable = l0op::Contiguous(blockTable, executor);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (attenMaskOptional) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNsaSelectParam(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                const aclTensor* topkIndices, const aclTensor* blockTableOptional,
                                const aclIntArray* actualQSeqLenOptional, const aclIntArray* actualKvSeqLenOptional,
                                const char* layoutOptional, const aclTensor* attentionOut,
                                const uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const std::pair<const void*, const char*> requiredParameters[] = {
        {query, "query"},
        {key, "key"},
        {value, "value"},
        {topkIndices, "topkIndices"},
        {blockTableOptional, "blockTableOptional"},
        {actualQSeqLenOptional, "actualQSeqLenOptional"},
        {actualKvSeqLenOptional, "actualKvSeqLenOptional"},
        {layoutOptional, "layoutOptional"},
        {executor, "executor"},
        {workspaceSize, "workspaceSize"},
        {attentionOut, "attentionOut"},
    };
    for (const auto& parameter : requiredParameters) {
        if (parameter.first == nullptr) {
            ReportInvalidParameter(std::string(parameter.second) + " must not be null");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

using OwnedIntArray = std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)>;

aclnnStatus MakeWorkspaceLengthArray(const aclIntArray* inArray, OwnedIntArray& outArray)
{
    OP_LOGD("start fake array");
    if (inArray != nullptr) {
        OP_LOGD("input array is not nullptr");
        uint64_t size = inArray->Size();
        std::vector<int64_t> fake_array(size, -1);
        // tiling侧认为有tensor但没有data就是计算最大workspace
        outArray.reset(aclCreateIntArray(fake_array.data(), size));
        if (outArray == nullptr) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "aclCreateIntArray returned null for %lu elements", size);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    OP_LOGD("end fake array");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaSelectedAttentionInferGetMaxWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* topkIndices,
    const aclTensor* attenMaskOptional, const aclTensor* blockTableOptional, const aclIntArray* actualQSeqLenOptional,
    const aclIntArray* actualKvSeqLenOptional, char* layoutOptional, int64_t numHeads, int64_t numKeyValueHeads,
    int64_t selectBlockSize, int64_t selectBlockCount, int64_t pageBlockSize, double scaleValue, int64_t sparseMode,
    aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OwnedIntArray fakeActualKvSeqLenOptional(nullptr, aclDestroyIntArray);
    OwnedIntArray fakeActualQSeqLenOptional(nullptr, aclDestroyIntArray);
    // nullptr不处理， nullptr是空指针，这样不会影响原来就不传入actual seq length为空的逻辑
    aclnnStatus ret = MakeWorkspaceLengthArray(actualKvSeqLenOptional, fakeActualKvSeqLenOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(ret, "MakeWorkspaceLengthArray failed for actualKvSeqLenOptional with status %d", ret);
        return ret;
    }

    ret = MakeWorkspaceLengthArray(actualQSeqLenOptional, fakeActualQSeqLenOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(ret, "MakeWorkspaceLengthArray failed for actualQSeqLenOptional with status %d", ret);
        return ret;
    }

    return aclnnNsaSelectedAttentionInferGetWorkspaceSize(
        query, key, value, topkIndices, attenMaskOptional, blockTableOptional, fakeActualQSeqLenOptional.get(),
        fakeActualKvSeqLenOptional.get(), layoutOptional, numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount,
        pageBlockSize, scaleValue, sparseMode, output, workspaceSize, executor);
}

aclnnStatus aclnnNsaSelectedAttentionInferGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* topkIndices,
    const aclTensor* attenMaskOptional, const aclTensor* blockTableOptional, const aclIntArray* actualQSeqLenOptional,
    const aclIntArray* actualKvSeqLenOptional, char* layoutOptional, int64_t numHeads, int64_t numKeyValueHeads,
    int64_t selectBlockSize, int64_t selectBlockCount, int64_t pageBlockSize, double scaleValue, int64_t sparseMode,
    aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_RET(
        CheckNsaSelectParam(query, key, value, topkIndices, blockTableOptional, actualQSeqLenOptional,
                            actualKvSeqLenOptional, layoutOptional, output, workspaceSize, executor) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnNsaSelectedAttentionInfer,
                   DFX_IN(query, key, value, topkIndices, attenMaskOptional, blockTableOptional, actualQSeqLenOptional,
                          actualKvSeqLenOptional, layoutOptional, numHeads, numKeyValueHeads, selectBlockSize,
                          selectBlockCount, pageBlockSize, scaleValue, sparseMode),
                   DFX_OUT(output));

    auto uniqueExecutor = CREATE_EXECUTOR();
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (output->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    std::string layoutStr = op::ToString(layoutOptional).GetString();
    if ((layoutStr != "BSND") && (layoutStr != "BSH") && (layoutStr != "TND")) {
        ReportInvalidParameter("layoutOptional must be BSND, BSH or TND; got " + layoutStr);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    NSAShapeInfo shapeInfo;
    shapeInfo.axes.n1 = numHeads;
    shapeInfo.axes.n2 = numKeyValueHeads;
    if (numHeads == 0 || numKeyValueHeads == 0 || numHeads % numKeyValueHeads != 0) {
        ReportInvalidParameter("numHeads and numKeyValueHeads must be nonzero and numHeads must be divisible by "
                               "numKeyValueHeads; got " +
                               std::to_string(numHeads) + ", " + std::to_string(numKeyValueHeads));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeads / numKeyValueHeads > GROUP_LIMITS) {
        ReportInvalidParameter("numHeads / numKeyValueHeads must not exceed " + std::to_string(GROUP_LIMITS) +
                               "; got " + std::to_string(numHeads / numKeyValueHeads));
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(AnalysisInputNsa(query, key, value, topkIndices, layoutOptional, shapeInfo, actualQSeqLenOptional,
                               actualKvSeqLenOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor* l0Executor = uniqueExecutor.get();
    CHECK_RET(ContiguousTensors(query, key, value, topkIndices, blockTableOptional, attenMaskOptional, l0Executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    auto l0attentionOut = l0op::NsaSelectedAttentionInfer(
        query, key, value, topkIndices, attenMaskOptional, blockTableOptional, actualQSeqLenOptional,
        actualKvSeqLenOptional, layoutOptional, numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount,
        pageBlockSize, scaleValue, sparseMode, l0Executor);
    if (l0attentionOut == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "NsaSelectedAttentionInfer returned a null attention output");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(l0attentionOut, output, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaSelectedAttentionInfer(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNsaSelectedAttentionInfer);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
