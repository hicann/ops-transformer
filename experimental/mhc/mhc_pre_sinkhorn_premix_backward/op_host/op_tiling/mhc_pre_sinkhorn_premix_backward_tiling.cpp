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
 * \file mhc_pre_sinkhorn_premix_backward_tiling.cpp
 * \brief MhcPreSinkhornPremixBackward tiling file
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "arch35/mhc_pre_sinkhorn_premix_backward_arch35_tiling_base.h"
#include "arch22/mhc_pre_sinkhorn_premix_backward_arch22_tiling.h"
#include "mhc_pre_sinkhorn_premix_backward_tiling.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"

namespace optiling {

static ge::graphStatus Tiling4MhcPreSinkhornPremixBackward(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "MhcPreSinkhornPremixBackward is running.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE("TilingForMhcPreSinkhornPremixBackward", "Tiling platformInfo is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context->GetNodeName(), "Using arch35 tiling for ASCEND950");
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }

    return TilingMhcPreSinkhornPremixBackwardArch22(context);
}

static ge::graphStatus TilingPrepare4MhcPreSinkhornPremixBackward(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the MhcPreSinkhornPremixBackward op.
IMPL_OP_OPTILING(MhcPreSinkhornPremixBackward)
    .Tiling(Tiling4MhcPreSinkhornPremixBackward)
    .TilingParse<MhcPreSinkhornPremixBackwardCompileInfo>(TilingPrepare4MhcPreSinkhornPremixBackward);
} // namespace optiling
