/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_pa_kv_cache_with_k_scale.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ScatterPaKvCacheWithKScale);

std::tuple<const aclTensor*, const aclTensor*, const aclTensor*> ScatterPaKvCacheWithKScale(
    const aclTensor* key, const aclTensor* value, aclTensor* keyCacheRef, aclTensor* valueCacheRef,
    const aclTensor* slotMapping, aclTensor* keyScale, aclTensor* keyScaleCacheRef, char* cacheLayoutOptional,
    aclOpExecutor* executor)
{
    L0_DFX(ScatterPaKvCacheWithKScale, key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef,
           cacheLayoutOptional);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ScatterPaKvCacheWithKScale,
        OP_INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef),
        OP_OUTPUT(keyCacheRef, valueCacheRef, keyScaleCacheRef), OP_ATTR(cacheLayoutOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>(keyCacheRef, valueCacheRef,
                                                                            keyScaleCacheRef);
}

} // namespace l0op
