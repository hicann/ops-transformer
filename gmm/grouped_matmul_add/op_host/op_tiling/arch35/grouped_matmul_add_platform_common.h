/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_add_platform_common.h
 * \brief
 */
#ifndef __GROUPED_MATMUL_ADD_PLATFORM_COMMON_H__
#define __GROUPED_MATMUL_ADD_PLATFORM_COMMON_H__

#include "tiling/platform/platform_ascendc.h"
#include <vector>
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "platform/soc_spec.h"
namespace optiling {
const std::vector<NpuArch> AdvancedNpuArch = {NpuArch::DAV_3510};

template <typename T>
inline typename std::enable_if<
    std::is_same<T, gert::TilingParseContext>::value || std::is_same<T, gert::TilingContext>::value, bool>::type
IsAdvancedNpuArch(T *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("GroupedMatmulAdd", "context is null"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    return std::find(AdvancedNpuArch.begin(), AdvancedNpuArch.end(), npuArch) != AdvancedNpuArch.end();
}
} // namespace optiling
#endif // __OP_HOST_GROUPED_MATMUL_ADD_PLATFORM_COMMON_H__
