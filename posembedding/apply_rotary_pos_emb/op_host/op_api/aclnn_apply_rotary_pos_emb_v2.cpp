/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>

#include "aclnn_apply_rotary_pos_emb_v2.h"
#include "aclnnInner_apply_rotary_pos_emb.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/op_log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static void CheckRotaryModeSupportAndWarn(const char* rotaryMode)
{
    if (Ops::Transformer::AclnnUtil::IsRegbase()) {
        return;
    }
    if (rotaryMode == nullptr || std::strcmp(rotaryMode, "half") == 0) {
        return;
    }
    OP_LOGW("ApplyRotaryPosEmb: rotaryMode=\"%s\" is not supported on the current platform "
            "(\"interleave\" and \"quarter\" are only supported on Ascend 950PR&950DT). "
            "The call still returns success, but the result is computed with \"half\" "
            "semantics and does not match the requested mode.",
            rotaryMode);
}

} // namespace

aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(aclTensor* queryRef, aclTensor* keyRef, const aclTensor* cos,
                                                     const aclTensor* sin, int64_t layout, char* rotaryMode,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CheckRotaryModeSupportAndWarn(rotaryMode);
    aclnnStatus ret = aclnnInnerApplyRotaryPosEmbGetWorkspaceSize(queryRef, keyRef, cos, sin, layout, rotaryMode,
                                                                  workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnApplyRotaryPosEmbV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerApplyRotaryPosEmb(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
