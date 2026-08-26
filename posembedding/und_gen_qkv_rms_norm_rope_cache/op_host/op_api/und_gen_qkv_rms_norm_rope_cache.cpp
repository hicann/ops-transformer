/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "und_gen_qkv_rms_norm_rope_cache.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UndGenQkvRmsNormRopeCache);

aclnnStatus UndGenQkvRmsNormRopeCache(const aclTensor *undQkv, const aclTensor *undWeightsQ,
                                      const aclTensor *undWeightsK, const aclTensor *cosSinCache, aclTensor *kCacheRef,
                                      aclTensor *vCacheRef, const aclTensor *slotMapping, const aclTensor *positions,
                                      const aclTensor *genQkv, const aclTensor *genWeightsQ,
                                      const aclTensor *genWeightsK, const aclTensor *catIndices, int64_t numHeadsQ,
                                      int64_t numHeadsK, int64_t numHeadsV, double normEps,
                                      const aclIntArray *mropeSection, aclTensor *qOut, aclOpExecutor *executor)
{
    L0_DFX(UndGenQkvRmsNormRopeCache, undQkv, undWeightsQ, undWeightsK, cosSinCache, kCacheRef, vCacheRef, slotMapping,
           positions, genQkv, genWeightsQ, genWeightsK, catIndices, numHeadsQ, numHeadsK, numHeadsV, normEps,
           mropeSection);

    // k_cache/v_cache 原地更新：同一个 tensor 既作为输入也作为输出下发，与仓内其他原地算子保持一致
    auto ret = INFER_SHAPE(UndGenQkvRmsNormRopeCache,
                           OP_INPUT(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCacheRef, vCacheRef, slotMapping,
                                    positions, genQkv, genWeightsQ, genWeightsK, catIndices),
                           OP_OUTPUT(qOut, kCacheRef, vCacheRef),
                           OP_ATTR(numHeadsQ, numHeadsK, numHeadsV, static_cast<float>(normEps), mropeSection));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return ret, "UndGenQkvRmsNormRopeCache InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        UndGenQkvRmsNormRopeCache,
        OP_INPUT(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCacheRef, vCacheRef, slotMapping, positions, genQkv,
                 genWeightsQ, genWeightsK, catIndices),
        OP_OUTPUT(qOut, kCacheRef, vCacheRef),
        OP_ATTR(numHeadsQ, numHeadsK, numHeadsV, static_cast<float>(normEps), mropeSection));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ret,
                                         "UndGenQkvRmsNormRopeCache ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return ACLNN_SUCCESS;
}
} // namespace l0op
