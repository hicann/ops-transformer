/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_KERNEL_A2_INT8_HPP
#define MEGA_MOE_KERNEL_A2_INT8_HPP
#include "mega_moe_kernel_a2.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, bool BatchedDispatch = false>
using MegaMoeKernelA2Int8 = MegaMoeKernelA2<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_,
                                            BlockEpilogue2_, true, false, BatchedDispatch>;

} // namespace Catlass::Gemm::Kernel

#endif // MEGA_MOE_KERNEL_A2_INT8_HPP
