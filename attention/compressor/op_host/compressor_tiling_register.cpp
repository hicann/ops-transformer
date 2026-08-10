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
 * \file compressor_tiling_register.cpp
 * \brief Compressor 算子 tiling 入口注册与 arch 分发
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

CMP_EXTERN_C ge::graphStatus TilingCompressorArch22(gert::TilingContext *context);
CMP_EXTERN_C ge::graphStatus TilingCompressorArch35(gert::TilingContext *context);

struct CompressorCompileInfo {
    int64_t core_num;
};

CMP_EXTERN_C ge::graphStatus TilingCompressor(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("Compressor", "context", "is nullptr"),
                return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("Compressor", "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        return TilingCompressorArch35(context);
    } else {
        return TilingCompressorArch22(context);
    }
}

ge::graphStatus TilingPrepareForCompressor(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Compressor).Tiling(TilingCompressor).TilingParse<CompressorCompileInfo>(TilingPrepareForCompressor);

} // namespace optiling
