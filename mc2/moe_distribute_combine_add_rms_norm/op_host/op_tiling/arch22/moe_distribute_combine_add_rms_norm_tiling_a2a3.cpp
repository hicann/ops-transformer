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
 * \file moe_distribute_combine_add_rms_norm_tiling_a2a3.cpp
 * \brief
 */

#include "moe_distribute_combine_add_rms_norm_tiling_a2a3.h"
#include "../../../../moe_distribute_combine_v2/op_host/op_tiling/arch22/moe_distribute_combine_tiling_arch22.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_exception_dump.h"

namespace optiling {
ge::graphStatus MoeDistributeCombineAddRmsNormTilingFuncA2A3::MoeDistributeCombineV2TilingFunc(
    gert::TilingContext *context)
{
    MoeDistributeCombineV2TilingFuncA2A3 funcA2A3;
    return funcA2A3.MoeDistributeCombineV2TilingFunc(context);
}

ge::graphStatus MoeDistributeCombineAddRmsNormTilingFuncImplA2A3(gert::TilingContext *context)
{
    MoeDistributeCombineAddRmsNormTilingFuncA2A3 impl;
    return impl.MoeDistributeCombineAddRmsNormTilingFunc(context);
}

struct MoeDistributeCombineAddRmsNormCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineAddRmsNorm(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineAddRmsNorm)
    .Tiling(MoeDistributeCombineAddRmsNormTilingFuncImplA2A3)
    .TilingParse<MoeDistributeCombineAddRmsNormCompileInfo>(TilingParseForMoeDistributeCombineAddRmsNorm);
} // namespace optiling
