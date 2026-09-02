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
 * \file allto_allv_grouped_mat_mul_tiling_a5.h
 * \brief
 */
#ifndef ALLTO_ALLV_GROUPED_MATMUL_TILING_A5_H
#define ALLTO_ALLV_GROUPED_MATMUL_TILING_A5_H
#include <vector>
#include "../allto_allv_grouped_mat_mul_tiling.h"

namespace optiling {
class AlltoAllvGmmTilingA5 : public AlltoAllvGmmTiling {
public:
    explicit AlltoAllvGmmTilingA5(gert::TilingContext *context)
        : AlltoAllvGmmTiling(context)
    {}
    std::vector<int64_t> GetEpWorldSizeOptional() const override;
    bool NeedToCheckCounts() const override;
};

class AlltoAllvGmmTilingFuncA5 : public AlltoAllvGmmTilingCommon {
public:
    ge::graphStatus AlltoAllvGmmOpTilingFunc(gert::TilingContext *context) override;
};

class AlltoAllvGmmTilingStructA5 : public AlltoAllvGmmTilingStruct {
public:
    explicit AlltoAllvGmmTilingStructA5(gert::TilingContext *context)
        : AlltoAllvGmmTilingStruct(context) {};

protected:
    ge::graphStatus DoOpTiling() override;
};
} // namespace optiling

#endif
