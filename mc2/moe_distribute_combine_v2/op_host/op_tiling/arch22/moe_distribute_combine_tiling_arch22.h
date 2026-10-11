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
 * \file moe_distribute_combine_tiling_arch22.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TILING_ARCH22_H
#define MOE_DISTRIBUTE_COMBINE_TILING_ARCH22_H

#include "../moe_distribute_combine_v2_tiling_base.h"

namespace optiling {

class MoeDistributeCombineV2TilingFuncA2A3 : public MoeDistributeCombineV2TilingFuncBase {
public:
    ge::graphStatus MoeDistributeCombineTilingFuncImpl(gert::TilingContext* context,
                                                       const CombineV2Config& config) override;
};

} // namespace optiling

#endif
