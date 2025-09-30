
/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_tiling_a2a3.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TILING_A2A3_H
#define MOE_DISTRIBUTE_COMBINE_TILING_A2A3_H

#include "tiling/moe_tiling_base.h"
#include "moe_distribute_combine_tiling_helper.h"

namespace optiling {
class MoeDistributeCombineTilingA2A3 : public MoeTilingBase {
public:
    explicit MoeDistributeCombineTilingA2A3(gert::TilingContext *context) : MoeTilingBase(context) {};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};
} // namespace optiling

#endif // MOE_DISTRIBUTE_COMBINE_TILING_A2A3_H