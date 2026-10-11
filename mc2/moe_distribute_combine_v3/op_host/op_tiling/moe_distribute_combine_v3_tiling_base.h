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
 * \file moe_distribute_combine_v3_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V3_TILING_BASE_H
#define MOE_DISTRIBUTE_COMBINE_V3_TILING_BASE_H

#include "../../../moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.h"

namespace optiling {
class MoeDistributeCombineV3TilingFuncBase {
public:
    ge::graphStatus MoeDistributeCombineV3TilingFunc(gert::TilingContext* context);
    virtual ge::graphStatus MoeDistributeCombineV2TilingFuncNew(gert::TilingContext* context,
                                                                const CombineV2Config& config) = 0;
};
} // namespace optiling
#endif
