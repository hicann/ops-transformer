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
 * \file base_defs.hpp
 * \brief
 */

#ifndef GBSAG_BASE_DEFS_HPP
#define GBSAG_BASE_DEFS_HPP

#include <cstdint>

#include <kernel_operator.h>

#include "../attn_infra/detail/gbsag_alignment.hpp"
#include "../attn_infra/detail/gbsag_dependent_false.hpp"
#include "../attn_infra/detail/gbsag_macros.hpp"

namespace NpuArch {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;

constexpr uint32_t STRIDE_LIMIT = 65536;

} // namespace NpuArch

#endif // GBSAG_BASE_DEFS_HPP
