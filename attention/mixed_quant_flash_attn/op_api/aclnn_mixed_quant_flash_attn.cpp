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
 * \file aclnn_mixed_quant_flash_attn.cpp
 * \brief
 */

#include "aclnn_mixed_quant_flash_attn.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_mixed_quant_flash_attn_inner.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 新版本opbase存在TensorV2的新接口，用弱符号判断当前opbase是新版本还是旧版本，旧版本不支持传入非连续tensor
bool NnopbaseSupportTensorV2() __attribute__((weak));

// 逐参检查k/v/k_descale/v_descale的连续性，返回首个非连续tensor的参数名；
// 全部连续（或参数为空指针，空指针交由inner接口统一返回ACLNN_ERR_PARAM_NULLPTR）时返回空串
static std::string FindNonContiguousTensorName(const aclTensor* key, const aclTensor* value, const aclTensor* kDescale,
                                               const aclTensor* vDescale)
{
    if (key != nullptr && !IsContiguous(key)) {
        return "k";
    }
    if (value != nullptr && !IsContiguous(value)) {
        return "v";
    }
    if (kDescale != nullptr && !IsContiguous(kDescale)) {
        return "k_descale";
    }
    if (vDescale != nullptr && !IsContiguous(vDescale)) {
        return "v_descale";
    }
    return "";
}

// 第一段接口：计算workspace大小
aclnnStatus aclnnMixedQuantFlashAttnGetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* v, const aclTensor* kDescale, const aclTensor* vDescale,
    const aclTensor* blockTableOptional, const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional, const aclTensor* sinksOptional, const aclTensor* attnMaskOptional,
    const aclTensor* metadataOptional, int64_t quantComputeMode, double softmaxScale, int64_t maskMode, int64_t winLeft,
    int64_t winRight, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char* layoutQ, const char* layoutKv,
    const char* layoutAttnOut, bool returnSoftmaxLse, const aclTensor* attnOut, const aclTensor* softmaxLseOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("start aclnnMixedQuantFlashAttnGetWorkspaceSize");

    // sinks shape为{0}时置nullptr
    MixedQuantFlashAttnProcessSinks(sinksOptional);

    const aclTensor* placeHolder = nullptr;
    const aclTensor* tempTensor = nullptr;

    MixedQuantFlashAttnProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);

    std::string nonContiguousTensor = FindNonContiguousTensorName(k, v, kDescale, vDescale);
    // 旧版本opbase（弱符号未定义）不支持传入非连续tensor，直接报错；新版本opbase放行至tiling校验
    if (!nonContiguousTensor.empty() && NnopbaseSupportTensorV2 == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR,
                "MixedQuantFlashAttn only supports non-contiguous "
                "%s with new opbase, current opbase does not support TensorV2 stride info.",
                nonContiguousTensor.c_str());
        // 释放softmax_lse占位符
        if (returnSoftmaxLse == false) {
            aclDestroyTensor(tempTensor);
        }
        return ACLNN_ERR_INNER_TILING_ERROR;
    }

    aclnnStatus ret = aclnnInnerMixedQuantFlashAttnGetWorkspaceSize(
        q, k, v, kDescale, vDescale, blockTableOptional, cuSeqlensQOptional, sequsedQOptional, sequsedKvOptional,
        sinksOptional, attnMaskOptional, metadataOptional, quantComputeMode, softmaxScale, maskMode, winLeft, winRight,
        maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv, layoutAttnOut, returnSoftmaxLse, attnOut, placeHolder,
        workspaceSize, executor);

    // 销毁占位符
    if (returnSoftmaxLse == false) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

// 第二段接口：执行计算
aclnnStatus aclnnMixedQuantFlashAttn(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     const aclrtStream stream)
{
    return aclnnInnerMixedQuantFlashAttn(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
