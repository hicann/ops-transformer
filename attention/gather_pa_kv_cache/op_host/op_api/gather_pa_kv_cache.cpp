/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_pa_kv_cache.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GatherPaKvCache);

std::tuple<const aclTensor*, const aclTensor*> GatherPaKvCache(const aclTensor* keyCache, const aclTensor* valueCache,
                                                               const aclTensor* blockTables, const aclTensor* seqLens,
                                                               aclTensor* keyRef, aclTensor* valueRef,
                                                               const aclTensor* seqOffsetOptional, char* cacheMode,
                                                               bool isSeqLensCumsum, aclOpExecutor* executor)
{
    L0_DFX(GatherPaKvCache, keyCache, valueCache, blockTables, seqLens, keyRef, valueRef, seqOffsetOptional, cacheMode,
           isSeqLensCumsum);

    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto socVersion = platformInfo.GetCurNpuArch();
    bool isArch35 = (socVersion == NpuArch::DAV_3510);

    const aclTensor* seqOffset = seqOffsetOptional;
    if (!isArch35 && seqOffset == nullptr) {
        seqOffset = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto ret =
        INFER_SHAPE(GatherPaKvCache, OP_INPUT(keyCache, valueCache, blockTables, seqLens, keyRef, valueRef, seqOffset),
                    OP_OUTPUT(keyRef, valueRef), OP_ATTR(cacheMode, isSeqLensCumsum));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(GatherPaKvCache,
                                      OP_INPUT(keyCache, valueCache, blockTables, seqLens, keyRef, valueRef, seqOffset),
                                      OP_OUTPUT(keyRef, valueRef), OP_ATTR(cacheMode, isSeqLensCumsum));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return std::tuple<const aclTensor*, const aclTensor*>(keyRef, valueRef);
}

} // namespace l0op
