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
 * \file quant_flash_attn_tiling_common.h
 * \brief QuantFlashAttn编译期信息结构体，由TilingParse阶段填充并缓存
 */

#pragma once

#include <cstdint>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace quant_flash_attn {
struct QuantFlashAttnCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
};

struct QfaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t config = 0;
    uint64_t quantMode = 0;
    bool hasAttenMask = false;
    uint64_t kvLayoutType = 0;
    bool isFd = false;
};

struct QfaPlatFormInfo {
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};
} // namespace quant_flash_attn
} // namespace optiling
