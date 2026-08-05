/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_tiling_constants.h
 * \brief Tiling constants used by quant_flash_attn arch35 tiling implementation.
 *        Local to quant_flash_attn; only the symbols actually referenced by
 *        quant_flash_attn_tiling_mxfp8.cpp are retained.
 */

#ifndef QUANT_FLASH_ATTN_TILING_CONSTANTS_H
#define QUANT_FLASH_ATTN_TILING_CONSTANTS_H

#include <cstdint>

namespace optiling {
namespace quant_flash_attn {
namespace arch35QFA {

constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SOUTER_128 = 128;
constexpr uint32_t SINNER_512 = 512;
constexpr uint32_t DSIZE_64 = 64;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_256 = 256;
constexpr uint32_t DSIZE_512 = 512;
constexpr uint32_t DSIZE_576 = 576;

} // namespace arch35QFA
} // namespace quant_flash_attn
} // namespace optiling

#endif // QUANT_FLASH_ATTN_TILING_CONSTANTS_H
