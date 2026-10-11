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
 * \file moe_distribute_dispatch_v3_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_V3_BASE
#define MOE_DISTRIBUTE_DISPATCH_TILING_V3_BASE

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_v2.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_exception_dump.h"

namespace optiling {
class MoeDistributeDispatchV3TilingFuncBase {
public:
    virtual ge::graphStatus MoeDistributeDispatchA3TilingFuncImplPublic(gert::TilingContext* context,
                                                                        DispatchV2Config& config) = 0;
    ge::graphStatus MoeDistributeDispatchV3TilingFunc(gert::TilingContext* context);
};
} // namespace optiling

#endif
