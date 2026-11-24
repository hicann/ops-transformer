/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_MEGA_MHC_H_
#define OP_API_INC_ACLNN_MEGA_MHC_H_
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
/** Build the seven-output contract; newPrevMixOut is required exactly when shiftedPrevMixOptional is supplied.
 * outputMode=0 requires yOut; yQuantOut and yScaleOut must be null.
 * outputMode=1 requires yQuantOut and yScaleOut; yOut must be null. Mode 2 is reserved and unsupported.
 * numSinkhornIters is the iteration count: 0 returns logits without Sinkhorn iterations;
 * 20 performs 20 Sinkhorn iterations and returns normalized Res coefficients. */
aclnnStatus aclnnMegaMhcGetWorkspaceSize(
    const aclTensor* x, const aclTensor* residual, const aclTensor* postMix, const aclTensor* fn,
    const aclTensor* mixScales, const aclTensor* mixBases, const aclTensor* rmsNorm1Weight,
    const aclTensor* combResMixOptional, const aclTensor* shiftedPrevMixOptional,
    const aclTensor* rmsNorm0WeightOptional, const aclTensor* rmsNormMidWeightOptional, double rmsNormEps,
    double hcNormEps, double hcPreEps, int64_t opImplMode, int64_t hcMult, double hcPostScale, double sinkhornEps,
    int64_t numSinkhornIters, double rmsNormScale, int64_t outputMode, aclTensor* newResidualOut,
    aclTensor* newPostMixOut, aclTensor* newCombResMixOut, aclTensor* newPrevMixOut, aclTensor* yOut,
    aclTensor* yQuantOut, aclTensor* yScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnMegaMhc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
