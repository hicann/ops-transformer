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
 * \file compressor_grad_tiling_register.cpp
 * \brief CompressorGrad 算子 tiling 入口注册与 arch 分发（arch35(ascend950)/arch92(ascend960dt) = PyPTO
 *        共用同一 tiling 实现，arch22(ascend910b/ascend910_93) = 原生 AscendC）
 */

#include "register/op_def_registry.h"
#include <register/op_impl_registry.h>
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "compressor_grad_tiling.h"

namespace optiling {

#ifdef ASCENDC_OP_TEST
#define CMP_GRAD_EXTERN_C extern "C"
#else
#define CMP_GRAD_EXTERN_C
#endif

CMP_GRAD_EXTERN_C ge::graphStatus TilingCompressorGradArch22(gert::TilingContext* context);
#ifdef COMPRESSOR_GRAD_HAS_ARCH35
CMP_GRAD_EXTERN_C ge::graphStatus TilingCompressorGradArch35(gert::TilingContext* context);
#endif

CMP_GRAD_EXTERN_C ge::graphStatus TilingCompressorGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "context", "is nullptr"),
                return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
#ifdef COMPRESSOR_GRAD_HAS_ARCH35
    // ascend950(DAV_3510/a5) 与 ascend960dt(DAV_9201/9202) 均走 PyPTO tiling：
    // 实现（compressor_grad_tiling.cpp）按平台核数自适配 a5 2:1 / ascend960dt 1:1 CV 配比，
    // tiling 结构体由对应 arch 的 PyPTO codegen 生成头 force-include 提供
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510 || npuArch == NpuArch::DAV_9201 || npuArch == NpuArch::DAV_9202) {
        return TilingCompressorGradArch35(context);
    }
#endif
    return TilingCompressorGradArch22(context);
}

ge::graphStatus TilingPrepareForCompressorGrad(gert::TilingParseContext* const context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompressorGrad)
    .Tiling(TilingCompressorGrad)
    .TilingParse<CompressorGradCompileInfo>(TilingPrepareForCompressorGrad);

} // namespace optiling
