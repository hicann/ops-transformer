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
 * \file stem_oam_prep_paged_kv.h
 * \brief
 */
#ifndef L0_STEM_OAM_PREP_PAGED_KV_H
#define L0_STEM_OAM_PREP_PAGED_KV_H

#include <tuple>
#include <string>
#include "opdev/op_executor.h"

namespace l0op {
std::tuple<const aclTensor *, const aclTensor *> StemOamPrepPagedKv(
    const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices, const aclTensor *kvSeqLens,
    const aclTensor *kScaleCache, const aclTensor *vScale, float lambdaMag, const std::string &kvLayout,
    int64_t stemBlockSize, int64_t stemStride, const aclTensor *kFlat, const aclTensor *vBias, aclOpExecutor *executor);
}

#endif // L0_STEM_OAM_PREP_PAGED_KV_H
