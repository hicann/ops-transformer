/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mhc_sinkhorn_tiling.h
 * \brief mhc_sinkhorn_tiling
 */

#ifndef MHC_SINKHORN_BASE_TILING_H_
#define MHC_SINKHORN_BASE_TILING_H_
#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
using Ops::Transformer::OpTiling::TilingBaseClass;

struct MhcSinkhornSplitCoreInfo {
    int64_t tNormCore{0};
    int64_t usedCoreNum{0};
    int64_t tTailCore{0};
    int64_t tUbFactor{0};
    int64_t tNormCoreLoop{0};
    int64_t tTailCoreLoop{0};
    int64_t tUbFactorTail{0};
    int64_t tUbTailTail{0};
};

class MhcSinkhornBaseTiling : public TilingBaseClass {
public:
    explicit MhcSinkhornBaseTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

    MhcSinkhornSplitCoreInfo BaseSplitCores(int64_t tSize, int64_t nSize, int64_t xDtypeSize, int64_t outFlag = 0);

protected:
    void BackwardSplitCore(uint64_t ubSize, int64_t totalCoreNum, int64_t tSize, int64_t nSize, int64_t xDtypeSize,
                           MhcSinkhornSplitCoreInfo &splitInfo);
    void ForwardSplitCore(uint64_t ubSize, int64_t totalCoreNum, int64_t tSize, int64_t nSize, int64_t xDtypeSize,
                          int64_t outFlag, MhcSinkhornSplitCoreInfo &splitInfo);
    int64_t BackwardCalOccupySize(int64_t ubFactor, int64_t nSize, int64_t xDtypeSize);
    int64_t ForwardCalOccupySize(int64_t ubFactor, int64_t nSize, int64_t xDtypeSize, int64_t outFlag);
};
} // namespace optiling
#endif // MHC_SINKHORN_BASE_TILING_H_
