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
 * \file quant_compressor_tiling_register.cpp
 * \brief QuantCompressor 算子 tiling 入口注册与 arch 分发
 */

#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

namespace optiling {

#ifdef ASCENDC_OP_TEST
#define CMP_EXTERN_C extern "C"
#else
#define CMP_EXTERN_C
#endif

CMP_EXTERN_C ge::graphStatus TilingQuantCompressorArch35(gert::TilingContext *context);
CMP_EXTERN_C ge::graphStatus TilingQuantCompressorArch92(gert::TilingContext *context);

struct QuantCompressorCompileInfo {
    int64_t core_num;
};

CMP_EXTERN_C ge::graphStatus TilingQuantCompressor(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantCompressor", "context", "is nullptr"),
                return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantCompressor", "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        return TilingQuantCompressorArch35(context);
    } else if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_9201) {
        return TilingQuantCompressorArch92(context);
    }
}

ge::graphStatus TilingPrepareForQuantCompressor(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantCompressor)
    .Tiling(TilingQuantCompressor)
    .TilingParse<QuantCompressorCompileInfo>(TilingPrepareForQuantCompressor);

} // namespace optiling
