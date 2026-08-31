/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/minimax_sparse_attention_split_kv_kernel_arch35.h"
#endif

#if (__CCE_AICORE__ == 310)

using namespace NpuArch;
using namespace MinimaxSaSplitKvKernelArch35;

template <class InDtype, class SMDtype, class REDtype>
__global__ __aicore__ void MinimaxSaSplitKvInferIntf(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR blockTable,
                                                     GM_ADDR k2qRowPtr, GM_ADDR k2qQIndices, GM_ADDR k2qSlotIndices,
                                                     GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR o,
                                                     GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
    using KernelBlockMmadQK = Gemm::Block::BlockMmadQKSplitKvArch35<InDtype, InDtype, SMDtype>;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using PType = Gemm::GemmType<InDtype, LayoutPDummy>;
    using SType = Gemm::GemmType<SMDtype, LayoutS>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa;
    using KernelEpilogueSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    using KernelBlockMmadPV = Gemm::Block::BlockMmadPVSplitKvArch35<InDtype, InDtype, REDtype>;
    using DispatchPolicyRescaleO = Epilogue::EpilogueRescaleOSplitKvArch35;
    using KernelEpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, bfloat16_t, REDtype, InDtype>;

    using MinimaxSaSplitKvKernel = MinimaxSparseAttentionSplitKvKernelArch35<KernelBlockMmadQK, KernelEpilogueSoftmax,
                                                                             KernelBlockMmadPV, KernelEpilogueRescaleO>;

    MinimaxSaSplitKvKernelParamsArch35 params{
        q, k,          v,         blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualQseqlen, actualKvseqlen,
        o, softmaxLse, workspace, tiling};
    MinimaxSaSplitKvKernel minimaxSaSplitKvKernel;
    minimaxSaSplitKvKernel(params);
}

#endif
