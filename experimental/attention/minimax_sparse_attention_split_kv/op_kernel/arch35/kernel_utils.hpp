/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MINIMAX_SA_SPLIT_KV_ARCH35_KERNEL_UTILS
#define MINIMAX_SA_SPLIT_KV_ARCH35_KERNEL_UTILS

#include "../attn_infra/base_defs.hpp"
#include "../attn_infra/arch/arch.hpp"
#include "../attn_infra/layout/layout.hpp"

#include "../attn_infra/gemm/block/block_mmad.hpp"
#include "../attn_infra/gemm/dispatch_policy.hpp"
#include "../attn_infra/gemm/gemm_type.hpp"

#include "../attn_infra/arch/cross_core_sync.hpp"
#include "../attn_infra/arch/resource.hpp"
#include "../attn_infra/epilogue/block/block_epilogue.hpp"
#include "../attn_infra/epilogue/dispatch_policy.hpp"
#include "../tla/tensor.hpp"
#include "../tla/layout.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MinimaxSaSplitKvKernelArch35 {

struct MinimaxSaSplitKvKernelParamsArch35 {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR blockTable;
    GM_ADDR k2qRowPtr;
    GM_ADDR k2qQIndices;
    GM_ADDR k2qSlotIndices;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR softmaxLse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    __aicore__ inline MinimaxSaSplitKvKernelParamsArch35() {}

    __aicore__ inline MinimaxSaSplitKvKernelParamsArch35(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR blockTable_,
                                                         GM_ADDR k2qRowPtr_, GM_ADDR k2qQIndices_,
                                                         GM_ADDR k2qSlotIndices_, GM_ADDR actualQseqlen_,
                                                         GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR softmaxLse_,
                                                         GM_ADDR workSpace_, GM_ADDR tiling_)
        : q(q_),
          k(k_),
          v(v_),
          blockTable(blockTable_),
          k2qRowPtr(k2qRowPtr_),
          k2qQIndices(k2qQIndices_),
          k2qSlotIndices(k2qSlotIndices_),
          actualQseqlen(actualQseqlen_),
          actualKvseqlen(actualKvseqlen_),
          o(o_),
          softmaxLse(softmaxLse_),
          workSpace(workSpace_),
          tiling(tiling_)
    {}
};

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t RoundUp(uint32_t a, uint32_t b)
{
    return CeilDiv(a, b) * b;
}

} // namespace MinimaxSaSplitKvKernelArch35

#endif // MINIMAX_SA_SPLIT_KV_ARCH35_KERNEL_UTILS
