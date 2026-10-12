/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_pa_kv_cache.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ScatterPaKvCache);

std::tuple<const aclTensor*, const aclTensor*> ScatterPaKvCache(
    const aclTensor* key, aclTensor* keyCacheRef, const aclTensor* slotMapping, const aclTensor* value,
    aclTensor* valueCacheRef, const aclTensor* compressLensOptional, const aclTensor* compressSeqOffsetOptional,
    const aclTensor* seqLensOptional, char* cacheModeOptional, char* scatterModeOptional,
    const aclIntArray* stridesOptional, const aclIntArray* offsetsOptional, aclOpExecutor* executor)
{
    L0_DFX(ScatterPaKvCache, key, keyCacheRef, slotMapping, value, valueCacheRef, compressLensOptional,
           compressSeqOffsetOptional, seqLensOptional, cacheModeOptional, scatterModeOptional, stridesOptional,
           offsetsOptional);

    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto socVersion = platformInfo.GetCurNpuArch();
    bool isArch35 = (socVersion == NpuArch::DAV_3510);

    const aclTensor* compressLens = compressLensOptional;
    if (!isArch35 && compressLensOptional == nullptr) {
        compressLens = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor* compressSeqOffset = compressSeqOffsetOptional;
    if (!isArch35 && compressSeqOffsetOptional == nullptr) {
        compressSeqOffset = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor* seqLens = seqLensOptional;
    if (!isArch35 && seqLensOptional == nullptr) {
        seqLens = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ScatterPaKvCache,
        OP_INPUT(key, keyCacheRef, slotMapping, value, valueCacheRef, compressLens, compressSeqOffset, seqLens),
        OP_OUTPUT(keyCacheRef, valueCacheRef),
        OP_ATTR(cacheModeOptional, scatterModeOptional, stridesOptional, offsetsOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return std::tuple<const aclTensor*, const aclTensor*>(keyCacheRef, valueCacheRef);
}

} // namespace l0op
