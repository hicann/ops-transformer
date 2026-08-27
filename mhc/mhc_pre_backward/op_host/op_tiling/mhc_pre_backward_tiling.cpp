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
 * \file mhc_pre_backward_tiling.cpp
 * \brief MhcPreBackward tiling dispatch entry - routes to arch22 (A2) or arch35 (A3/A5)
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "arch22/mhc_pre_backward_arch22_tiling.h"
#include "mhc_pre_backward_tiling.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "err/ops_err.h"

namespace optiling {

static ge::graphStatus Tiling4MhcPreBackward(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "MhcPreBackward is running.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("TilingForMhcPreBackward", "Tiling platformInfo is null"),
                return ge::GRAPH_FAILED);

    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context->GetNodeName(), "Using arch35 tiling for ASCEND950");
        return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }

    OP_LOGD(context->GetNodeName(), "Using arch22 tiling for ASCEND910B");
    return TilingMhcPreBackwardArch22(context);
}

static ge::graphStatus TilingPrepare4MhcPreBackward(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("[TilingPrepare4mHC]", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MhcPreBackwardCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);

    OP_LOGI(context->GetNodeName(), "parse compile info success l1Size:%lu, l2Size:%lu, coreNum:%lu",
            compileInfoPtr->l1Size, compileInfoPtr->l2Size, compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the MhcPreBackward op.
IMPL_OP_OPTILING(MhcPreBackward)
    .Tiling(Tiling4MhcPreBackward)
    .TilingParse<MhcPreBackwardCompileInfo>(TilingPrepare4MhcPreBackward);
} // namespace optiling
