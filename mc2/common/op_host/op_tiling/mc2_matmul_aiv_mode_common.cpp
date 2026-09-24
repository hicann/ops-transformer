/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_matmul_aiv_mode_common.h"

#include "platform/platform_ascendc.h"

namespace mc2tiling {

AivPlatformInfo GetAivPlatformInfo(const gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    AivPlatformInfo info{platform.GetCoreNumAiv(), 0U};
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    return info;
}

AivScaleDtypeResult CheckAivMatmulScaleDtype(const gert::Tensor *scale, ge::DataType outputType)
{
    if (scale == nullptr) {
        return {false, false};
    }
    const auto scaleType = scale->GetDataType();
    const bool isInt64 = outputType == ge::DT_FLOAT16 && scaleType == ge::DT_INT64;
    return {scaleType == ge::DT_FLOAT || isInt64, isInt64};
}

} // namespace mc2tiling
