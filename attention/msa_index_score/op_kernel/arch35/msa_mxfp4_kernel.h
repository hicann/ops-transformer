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
 * \file msa_mxfp4_kernel.h
 * \brief Packed MXFP4 E2M1 x E2M1 with E8M0 group-32 scales on Ascend 950.
 */
#ifndef MSA_MXFP4_KERNEL_H
#define MSA_MXFP4_KERNEL_H

#include "msa_index_score_kernel_arch35.h"

namespace MsaIndexScoreNs {

template <bool ENABLE_TOPK = false>
class MsaMxFp4Kernel {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR blockTable, GM_ADDR queryScale, GM_ADDR keyScale,
                                GM_ADDR actualSeqQlen, GM_ADDR actualSeqKlen, GM_ADDR startLoc, GM_ADDR score,
                                GM_ADDR workspace, const MsaIndexScoreTilingData* __restrict tiling,
                                GM_ADDR sparseIndices = nullptr, GM_ADDR sparseValues = nullptr)
    {
        query_ = query;
        key_ = key;
        blockTable_ = blockTable;
        queryScale_ = queryScale;
        keyScale_ = keyScale;
        actualSeqQlen_ = actualSeqQlen;
        actualSeqKlen_ = actualSeqKlen;
        startLoc_ = startLoc;
        score_ = score;
        workspace_ = workspace;
        tiling_ = tiling;
        sparseIndices_ = sparseIndices;
        sparseValues_ = sparseValues;
    }

    // Separate block16/block128 tiling keys select the corresponding template instances.
    __aicore__ inline void ProcessFp4()
    {
        if (tiling_->mxFp4WideN != 0U) {
            Run<false, true>();
        } else {
            Run<false, false>();
        }
    }

    // A single short TND page has no grouping opportunity. Keep its Cube hot path compact.
    __aicore__ inline void ProcessLatency()
    {
        if (tiling_->mxFp4WideN != 0U) {
            Run<false, true, false, true>();
        } else {
            Run<false, false, false, true>();
        }
    }

    __aicore__ inline void ProcessPrefill()
    {
        if (tiling_->mxFp4WideN != 0U) {
            Run<false, true, true>();
        } else {
            Run<false, false, true>();
        }
    }

    __aicore__ inline void ProcessBlock16()
    {
        Run<true, false>();
    }

private:
    template <bool BLOCK16, bool WIDE_N, bool FUSED = false, bool LATENCY_TND = false>
    __aicore__ inline void Run()
    {
        MsaIndexScoreKernel<fp8_e4m3fn_t, false, BLOCK16, (MSA_A5_USE_C2UB != 0), true, WIDE_N, ENABLE_TOPK, FUSED,
                            LATENCY_TND>
            op;
        op.InitMx(query_, key_, queryScale_, keyScale_, blockTable_, actualSeqQlen_, actualSeqKlen_, startLoc_, score_,
                  workspace_, tiling_, sparseIndices_, sparseValues_);
        op.Process();
    }

    const MsaIndexScoreTilingData* __restrict tiling_ = nullptr;
    GM_ADDR query_ = nullptr;
    GM_ADDR key_ = nullptr;
    GM_ADDR blockTable_ = nullptr;
    GM_ADDR queryScale_ = nullptr;
    GM_ADDR keyScale_ = nullptr;
    GM_ADDR actualSeqQlen_ = nullptr;
    GM_ADDR actualSeqKlen_ = nullptr;
    GM_ADDR startLoc_ = nullptr;
    GM_ADDR score_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    GM_ADDR sparseIndices_ = nullptr;
    GM_ADDR sparseValues_ = nullptr;
};

} // namespace MsaIndexScoreNs
#endif // MSA_MXFP4_KERNEL_H
