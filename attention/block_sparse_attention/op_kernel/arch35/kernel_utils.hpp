/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BSA_ARCH35_KERNEL_UTILS
#define BSA_ARCH35_KERNEL_UTILS

#include "../attn_infra/bsa_base_defs.hpp"
#include "../attn_infra/arch/bsa_arch.hpp"
#include "../attn_infra/layout/bsa_layout.hpp"

#include "../attn_infra/gemm/block/bsa_block_mmad.hpp"
#include "../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../attn_infra/gemm/bsa_gemm_type.hpp"

#include "../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../attn_infra/arch/bsa_resource.hpp"
#include "../attn_infra/epilogue/block/bsa_block_epilogue.hpp"
#include "../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../tla/tensor_bsa.hpp"
#include "../tla/layout.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace BsaKernelArch35 {

enum class Format {
    TND = 0,
    BNSD = 1,
    BSND = 2
};

struct BsaKernelParamsArch35 {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR blockSparseMask;
    GM_ADDR o;
    GM_ADDR lse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    // Methods
    __aicore__ inline
    BsaKernelParamsArch35() {}
    __aicore__ inline
    BsaKernelParamsArch35(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
        GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR blockSparseMask_, GM_ADDR o_,
        GM_ADDR workSpace_, GM_ADDR lse_, GM_ADDR tiling_)
        : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
        actualKvseqlen(actualKvseqlen_), blockSparseMask(blockSparseMask_), o(o_),
        workSpace(workSpace_), lse(lse_), tiling(tiling_) {}
};

struct BsaFullQuantKernelParamsArch35 {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR blockSparseMask;
    GM_ADDR attenMask;
    GM_ADDR blockTable;
    GM_ADDR actualSeqLengths;
    GM_ADDR actualSeqLengthsKv;
    GM_ADDR qDequantScale;
    GM_ADDR kDequantScale;
    GM_ADDR vDequantScale;
    GM_ADDR attentionOut;
    GM_ADDR lse;
    GM_ADDR workSpace;
    GM_ADDR tiling;

    // Methods
    __aicore__ inline BsaFullQuantKernelParamsArch35() {}
    __aicore__ inline BsaFullQuantKernelParamsArch35(GM_ADDR query_, GM_ADDR key_, GM_ADDR value_,
                                                     GM_ADDR blockSparseMask_, GM_ADDR attenMask_, GM_ADDR blockTable_,
                                                     GM_ADDR actualSeqLengths_, GM_ADDR actualSeqLengthsKv_,
                                                     GM_ADDR qDequantScale_, GM_ADDR kDequantScale_,
                                                     GM_ADDR vDequantScale_, GM_ADDR attentionOut_,
                                                     GM_ADDR workSpace_, GM_ADDR lse_, GM_ADDR tiling_)
        : query(query_), key(key_), value(value_), blockSparseMask(blockSparseMask_), attenMask(attenMask_),
          blockTable(blockTable_), actualSeqLengths(actualSeqLengths_), actualSeqLengthsKv(actualSeqLengthsKv_),
          qDequantScale(qDequantScale_), kDequantScale(kDequantScale_), vDequantScale(vDequantScale_),
          attentionOut(attentionOut_), workSpace(workSpace_), lse(lse_), tiling(tiling_) {}
};

__aicore__ inline
uint32_t GetCurQSTileNum(int64_t curQSeqlen, uint32_t blockShapeX, uint32_t qBaseTile)
{
    uint32_t fullXBlockNum = curQSeqlen / blockShapeX;
    uint32_t tailXBlockSize = curQSeqlen % blockShapeX;
    uint32_t qSTileNumPerFullXBlock = (blockShapeX + qBaseTile - 1) / qBaseTile;
    uint32_t qSTileNumTailXBlock = (tailXBlockSize + qBaseTile - 1) / qBaseTile;
    uint32_t curQSTileNum = qSTileNumPerFullXBlock * fullXBlockNum + qSTileNumTailXBlock;
    return curQSTileNum;
}

}

#endif