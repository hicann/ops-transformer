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
 * \file moe_re_routing_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_V2_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_V2_H_

#include "../../moe_re_routing/op_host/moe_re_routing_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(MoeReRoutingV2, MoeReRoutingTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRoutingV2_200000, MoeReRoutingReTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRoutingV2_200100, MoeReRoutingReTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRoutingV2_200200, MoeReRoutingReTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRoutingV2_210000, MoeReRoutingRTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRoutingV2_210100, MoeReRoutingRTilingData);
REGISTER_TILING_DATA_CLASS(MoeReRoutingV2_210200, MoeReRoutingRTilingData);
} // namespace optiling

#endif
