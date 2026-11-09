/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MHC_L0OP_H_
#define MEGA_MHC_L0OP_H_
#include <cstdint>
#include "opdev/op_executor.h"
namespace l0op {
/** Launch optional RMS0 -> Post -> optional Mid -> Pre -> RMS1 with contiguous output views. */
bool MegaMhc(const aclTensor* x, const aclTensor* residual, const aclTensor* postMix, const aclTensor* fn,
             const aclTensor* mixScales, const aclTensor* mixBases, const aclTensor* rmsNorm1Weight,
             const aclTensor* combResMixOptional, const aclTensor* shiftedPrevMixOptional,
             const aclTensor* rmsNorm0WeightOptional, const aclTensor* rmsNormMidWeightOptional, double rmsNormEps,
             double hcNormEps, double hcPreEps, int64_t opImplMode, int64_t hcMult, double hcPostScale,
             double sinkhornEps, int64_t numSinkhornIters, double rmsNormScale, int64_t outputMode,
             const aclTensor* newResidualOut, const aclTensor* newPostMixOut, const aclTensor* newCombResMixOut,
             const aclTensor* newPrevMixOut, const aclTensor* yOut, const aclTensor* yQuantOut,
             const aclTensor* yScaleOut, aclOpExecutor* executor);
} // namespace l0op
#endif
