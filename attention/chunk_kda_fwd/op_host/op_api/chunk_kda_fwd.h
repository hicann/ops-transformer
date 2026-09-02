/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_CHUNK_KDA_FWD_OP_H
#define OP_API_INC_LEVEL0_OP_CHUNK_KDA_FWD_OP_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
using KdaCoreOutputs = std::array<const aclTensor *, 11>;

KdaCoreOutputs KdaChunkForward(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
                               const aclTensor *beta, const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
                               const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
                               const aclIntArray *chunkIndicesOptional, double scale, int64_t chunkSize, bool safeGate,
                               bool inputSequenceMajor, bool useGateInKernel, double lowerBound,
                               const aclTensor *attnOut, const aclTensor *finalStateOut, const aclTensor *gkOut,
                               const aclTensor *aqkOut, const aclTensor *akkOut, const aclTensor *wOut,
                               const aclTensor *uOut, const aclTensor *qgOut, const aclTensor *kgOut,
                               const aclTensor *vNewOut, const aclTensor *hOut, aclOpExecutor *executor);
} // namespace l0op

#endif
