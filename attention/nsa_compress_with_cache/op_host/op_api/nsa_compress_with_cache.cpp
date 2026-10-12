/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nsa_compress_with_cache.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(NsaCompressWithCache);

const aclTensor* NsaCompressWithCache(const aclTensor* input, const aclTensor* weight, const aclTensor* slotMapping,
                                      aclTensor* outputCacheRef, const aclIntArray* actSeqLenOptional,
                                      const aclTensor* blockTableOptional, char* layoutOptional,
                                      int64_t compressBlockSize, int64_t compressStride, int64_t actSeqLenType,
                                      int64_t pageBlockSize, aclOpExecutor* executor)
{
    // L0接口时延统计以及入参打印
    L0_DFX(NsaCompressWithCache, input, weight, slotMapping, outputCacheRef, actSeqLenOptional, blockTableOptional,
           layoutOptional, compressBlockSize, compressStride, actSeqLenType, pageBlockSize);

    // 处理optional变量，如果为null需要创建占位符
    if (blockTableOptional == nullptr) {
        blockTableOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    // 处理aclIntArray，转换成aclTensor
    const aclTensor* actSeqLen = nullptr;
    if (actSeqLenOptional) {
        actSeqLen = executor->ConvertToTensor(actSeqLenOptional, DataType::DT_INT64);
        const_cast<aclTensor*>(actSeqLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actSeqLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actSeqLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actSeqLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    // 调用inferShape
    auto ret = INFER_SHAPE(NsaCompressWithCache,
                           OP_INPUT(input, weight, slotMapping, outputCacheRef, actSeqLen, blockTableOptional),
                           OP_OUTPUT(outputCacheRef),
                           OP_ATTR(layoutOptional, compressBlockSize, compressStride, actSeqLenType, pageBlockSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    // 发起aicore任务
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        NsaCompressWithCache, OP_INPUT(input, weight, slotMapping, outputCacheRef, actSeqLen, blockTableOptional),
        OP_OUTPUT(outputCacheRef),
        OP_ATTR(layoutOptional, compressBlockSize, compressStride, actSeqLenType, pageBlockSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return outputCacheRef;
}

} // namespace l0op
