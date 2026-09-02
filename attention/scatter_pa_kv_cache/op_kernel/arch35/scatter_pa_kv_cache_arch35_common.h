/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_pa_kv_cache_arch35_common.h
 * \brief
 */

#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

#ifndef SCATTER_PA_KV_CACHE_ARCH35_COMMON_H
#define SCATTER_PA_KV_CACHE_ARCH35_COMMON_H
namespace ScatterPaKvCache {
constexpr int32_t B8 = 1;
constexpr int32_t B16 = 2;
constexpr int32_t B32 = 4;
constexpr int64_t DUAL_IN_OUT = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
} // namespace ScatterPaKvCache
#endif // SCATTER_PA_KV_CACHE_ARCH35_COMMON_H
