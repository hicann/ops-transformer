/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_COMM_ALG_DEF_H
#define MOE_EP_COMM_ALG_DEF_H

#include <cstdint>

namespace Mc2Tiling {

constexpr int64_t MOE_EP_COMM_ALG_AUTO = 0;
// Resolved implementation modes. The public clos_only policy selects by EP size.
// Preserve the existing shared-Jetty value in device tiling data.
constexpr int64_t MOE_EP_COMM_ALG_SHARED_JETTY = 1;
constexpr int64_t MOE_EP_COMM_ALG_CLOS_ONLY = 2; // Legacy context/kernel, Clos channels only.
constexpr int64_t MOE_EP_COMM_ALG_FULLMESH = MOE_EP_COMM_ALG_AUTO;
constexpr int64_t MOE_EP_SHARED_JETTY_MIN_RANK_SIZE = 16;

} // namespace Mc2Tiling

#endif // MOE_EP_COMM_ALG_DEF_H
