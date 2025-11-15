/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_nsa_selected_attention_infer.cpp
 * \brief
 */

#include "nsa_selected_attention_infer.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
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
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *value, 
    const aclTensor *topkIndices, 
    const aclTensor *attenMaskOptional,
    const aclTensor *blockTableOptional,
    const aclIntArray *actualQSeqLenOptional,
    const aclIntArray *actualKvSeqLenOptional,
    char *layoutOptional,
    int64_t numHeads,
    int64_t numKeyValueHeads,
    int64_t selectBlockSize,
    int64_t selectBlockCount,
    int64_t pageBlockSize,
    double scaleValue,
    int64_t sparseMode,
    aclTensor *output,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);

void AnalysisAxisForBSH(const Shape &qShape, const Shape &kShape, const Shape &vShape, NSAShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSH;
    shapeInfo.axes.b = qShape[DIM_NUM_0];
    shapeInfo.axes.d = qShape[DIM_NUM_2] / shapeInfo.axes.n1;
    shapeInfo.axes.dk = kShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.dv = vShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.s1 = qShape[DIM_NUM_1];
    shapeInfo.axes.s2 = vShape[DIM_NUM_1] * vShape[DIM_NUM_0];
}

void AnalysisAxisForBSND(const Shape &qShape, const Shape &kShape, const Shape &vShape, NSAShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSND;
    shapeInfo.axes.b = qShape[DIM_NUM_0];
    shapeInfo.axes.d = qShape[DIM_NUM_3];
    shapeInfo.axes.dk = kShape[DIM_NUM_3];
    shapeInfo.axes.dv = vShape[DIM_NUM_3];
    shapeInfo.axes.s1 = qShape[DIM_NUM_1];
    shapeInfo.axes.s2 = vShape[DIM_NUM_1] * vShape[DIM_NUM_0];
}

void AnalysisAxisForTND(const Shape &qShape, const Shape &kShape, const Shape &vShape, NSAShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::TND;
    shapeInfo.axes.numtokens = qShape[DIM_NUM_0];
    shapeInfo.axes.d = qShape[DIM_NUM_2];
    shapeInfo.axes.dk = kShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.dv = vShape[DIM_NUM_2] / shapeInfo.axes.n2;
    shapeInfo.axes.s2 = vShape[DIM_NUM_1] * vShape[DIM_NUM_0];
}

aclnnStatus AnalysisAxisShape(const aclTensor *query, const aclTensor *key, const aclTensor *value, 
                                const char *inputLayout, NSAShapeInfo &shapeInfo)
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
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support input_layout %s with dim_num %lu", inputLayout, shapeInfo.dimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d != shapeInfo.axes.dk) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qD and kD should be same, but got qD=%ld kD=%ld", shapeInfo.axes.d, 
            shapeInfo.axes.dk);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d < shapeInfo.axes.dv) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support kD >= vD, but got kD=%ld vD=%ld", shapeInfo.axes.d, 
            shapeInfo.axes.dv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus DtypeCheck(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *topkIndices)
{
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    auto topkDtype = topkIndices->GetDataType();
    if (qDtype != kDtype || kDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of query[%s], key[%s], value[%s] are not equal.",
                op::ToString(DataType(qDtype)).GetString(), op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype != ge::DT_FLOAT16 && qDtype != ge::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type does not meet requirements.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (topkDtype != ge::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "topk dtype does not match with required dtype int32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AnalysisInputNsa(const aclTensor *query, const aclTensor *key, const aclTensor *value, 
    const aclTensor *topkIndices, char *inputLayout, NSAShapeInfo &shapeInfo, 
    const aclIntArray *actualQSeqLenOptional = nullptr, const aclIntArray *actualSeqKvLenOptional = nullptr)
{
    CHECK_RET(DtypeCheck(query, key, value, topkIndices)  == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        AnalysisAxisShape(query, key, value, inputLayout, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.d != HEAD_DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QK Head dim must == 192, but got %ld", shapeInfo.axes.d);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.dv != HEAD_DIM_V_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "V Head dim must == 128, but got %ld", shapeInfo.axes.dv);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if ((actualQSeqLenOptional != nullptr && actualSeqKvLenOptional != nullptr) && 
                    (actualQSeqLenOptional->Size() > 3 * 1024 || actualSeqKvLenOptional->Size() > 3 * 1024)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
                "actualSeqQLen size and actualSeqKvLen size must <= 3072, but got %lu ", actualQSeqLenOptional->Size());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (shapeInfo.axes.n2 == 0 || shapeInfo.axes.d == 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    if (inputLayoutStr == "TND") {
        if (shapeInfo.axes.numtokens == 0 || shapeInfo.axes.s2 == 0) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (shapeInfo.axes.b == 0 || shapeInfo.axes.s1 == 0 || shapeInfo.axes.s2 == 0) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ContiguousTensors(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                    const aclTensor *&topkIndices, const aclTensor *&blockTable,
                    const aclTensor *&attenMaskOptional, aclOpExecutor *executor)
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

aclnnStatus CheckNsaSelectParam(const aclTensor *query, const aclTensor *key, const aclTensor *value, 
    const aclTensor *topkIndices, const aclTensor *blockTableOptional, const aclIntArray *actualQSeqLenOptional,
    const aclIntArray *actualKvSeqLenOptional, const char *layoutOptional, const aclTensor *attentionOut,
    const uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 必须的参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(topkIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualQSeqLenOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualKvSeqLenOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(layoutOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus FakeArray_sai(const aclIntArray *inArray, aclIntArray *&outArray) {
    OP_LOGD("start fake array");
    if (inArray != nullptr) {
        OP_LOGD("input array is not nullptr");
        uint64_t size = inArray->Size();
        std::vector<int64_t> fake_array(size, -1);
        // tiling侧认为有tensor但没有data就是计算最大workspace
        outArray = aclCreateIntArray(fake_array.data(), size);
        if (outArray == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc array failed");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    OP_LOGD("end fake array");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaSelectedAttentionInferGetMaxWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *value, 
    const aclTensor *topkIndices, 
    const aclTensor *attenMaskOptional,
    const aclTensor *blockTableOptional,
    const aclIntArray *actualQSeqLenOptional,
    const aclIntArray *actualKvSeqLenOptional,
    char *layoutOptional,
    int64_t numHeads,
    int64_t numKeyValueHeads,
    int64_t selectBlockSize,
    int64_t selectBlockCount,
    int64_t pageBlockSize,
    double scaleValue,
    int64_t sparseMode,
    aclTensor *output,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
){
        aclIntArray *fakeActualKvSeqLenOptional{nullptr};
        aclIntArray *fakeActualQSeqLenOptional{nullptr};
        // nullptr不处理， nullptr是空指针，这样不会影响原来就不传入actual seq length为空的逻辑
        aclnnStatus ret = FakeArray_sai(actualKvSeqLenOptional, fakeActualKvSeqLenOptional);
        CHECK_RET_CODE(ret, "Try alloc fake actualKvSeqLenOptional failed");

        ret = FakeArray_sai(actualQSeqLenOptional, fakeActualQSeqLenOptional);
        CHECK_RET_CODE(ret, "Try alloc fake actualQSeqLenOptional failed");

        ret = aclnnNsaSelectedAttentionInferGetWorkspaceSize(query, 
                                                             key, 
                                                             value, 
                                                             topkIndices, 
                                                             attenMaskOptional,
                                                             blockTableOptional,
                                                             fakeActualQSeqLenOptional,
                                                             fakeActualKvSeqLenOptional,
                                                             layoutOptional,
                                                             numHeads,
                                                             numKeyValueHeads,
                                                             selectBlockSize,
                                                             selectBlockCount,
                                                             pageBlockSize,
                                                             scaleValue,
                                                             sparseMode,
                                                             output,
                                                             workspaceSize,
                                                             executor
        );
        aclDestroyIntArray(fakeActualKvSeqLenOptional);
        aclDestroyIntArray(fakeActualQSeqLenOptional);
        return ret;
}

aclnnStatus aclnnNsaSelectedAttentionInferGetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *value, 
    const aclTensor *topkIndices, 
    const aclTensor *attenMaskOptional,
    const aclTensor *blockTableOptional,
    const aclIntArray *actualQSeqLenOptional,
    const aclIntArray *actualKvSeqLenOptional,
    char *layoutOptional,
    int64_t numHeads,
    int64_t numKeyValueHeads,
    int64_t selectBlockSize,
    int64_t selectBlockCount,
    int64_t pageBlockSize,
    double scaleValue,
    int64_t sparseMode,
    aclTensor *output,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(CheckNsaSelectParam(query, key, value, topkIndices, blockTableOptional, actualQSeqLenOptional, actualKvSeqLenOptional, 
    layoutOptional, output, workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnNsaSelectedAttentionInfer,
                DFX_IN(query, key, value, topkIndices, attenMaskOptional, blockTableOptional, 
                        actualQSeqLenOptional, actualKvSeqLenOptional, layoutOptional, numHeads, numKeyValueHeads,
                        selectBlockSize, selectBlockCount, pageBlockSize, scaleValue, sparseMode),
                DFX_OUT(output));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (output->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    std::string layoutStr = op::ToString(layoutOptional).GetString();
    if ((layoutStr !="BSND") && (layoutStr != "BSH") && (layoutStr != "TND")) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not BSND or BSH or TND, invalid shape, please check", layoutOptional);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    NSAShapeInfo shapeInfo;
    shapeInfo.axes.n1 = numHeads;
    shapeInfo.axes.n2 = numKeyValueHeads;
    CHECK_RET(!(numHeads == 0 || numKeyValueHeads == 0 || numHeads % numKeyValueHeads != 0), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(shapeInfo.axes.n1 / shapeInfo.axes.n2 <= GROUP_LIMITS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(AnalysisInputNsa(query, key, value, topkIndices, layoutOptional, 
        shapeInfo, actualQSeqLenOptional, actualKvSeqLenOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(ContiguousTensors(query, key, value, topkIndices, blockTableOptional, attenMaskOptional, l0Executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

    auto l0attentionOut = l0op::NsaSelectedAttentionInfer(
        query, key, value, topkIndices, attenMaskOptional, blockTableOptional, actualQSeqLenOptional, 
        actualKvSeqLenOptional, layoutOptional, numHeads, numKeyValueHeads, selectBlockSize,
        selectBlockCount, pageBlockSize, scaleValue, sparseMode, l0Executor);
    if (l0attentionOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0attentionOut is null");
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

aclnnStatus aclnnNsaSelectedAttentionInfer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNsaSelectedAttentionInfer);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // namespace

#ifdef __cplusplus
}
#endif
 