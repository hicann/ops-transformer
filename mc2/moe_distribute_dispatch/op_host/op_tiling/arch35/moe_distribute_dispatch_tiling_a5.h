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
 * \file moe_distribute_dispatch_tiling_a5.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_A5_H
#define MOE_DISTRIBUTE_DISPATCH_TILING_A5_H

#include "../moe_distribute_dispatch_tiling_base.h"

namespace optiling {
class MoeDistributeDispatchTilingA5 : public MoeDistributeDispatchTilingBase {
public:
    explicit MoeDistributeDispatchTilingA5(gert::TilingContext *context)
        : MoeDistributeDispatchTilingBase(context) {};
    ge::graphStatus MoeDistributeDispatchTilingFunc(gert::TilingContext *context) override;
    ge::graphStatus CheckEpWorldSizeAttrs(const char *nodeName, MoeDistributeDispatchTilingData &tilingData) override;
};

} // namespace optiling

#endif // MOE_DISTRIBUTE_DISPATCH_TILING_A5_H
