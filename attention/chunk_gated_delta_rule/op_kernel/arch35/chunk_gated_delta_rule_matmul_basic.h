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
 * \file chunk_gated_delta_rule_matmul_basic.h
 * \brief CGDR basic API matmul: manually controls GM→L1→L0A/L0B→Mmad→L0C→GM pipeline.
 */

#ifndef CHUNK_GATED_DELTA_RULE_MATMUL_BASIC_H
#define CHUNK_GATED_DELTA_RULE_MATMUL_BASIC_H

#include "kernel_operator.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

constexpr uint32_t MM_BLOCK_CUBE = 16;
constexpr uint32_t MM_NZ_FRACTAL = 16;
constexpr uint32_t MM_MAX_DIM = 128;
constexpr uint32_t MM_L1_ELEM_CNT = MM_MAX_DIM * MM_MAX_DIM;  // 16384 BF16 elements = 32KB
constexpr uint32_t MM_L0_ELEM_CNT = MM_MAX_DIM * MM_MAX_DIM;  // 16384 elements
constexpr uint32_t MM_L0C_ELEM_CNT = MM_MAX_DIM * MM_MAX_DIM; // 16384 FP32 elements = 64KB

constexpr uint32_t MM_L1A_OFFSET = 0;                                            // L1 offset for A (bytes)
constexpr uint32_t MM_L1B_OFFSET = MM_MAX_DIM * MM_MAX_DIM * sizeof(bfloat16_t); // 32KB
constexpr uint32_t MM_L0A_ADDR = 0;
constexpr uint32_t MM_L0B_ADDR = 0;
constexpr uint32_t MM_L0C_ADDR = 0;

constexpr uint32_t FLAG_ZERO = 0;

enum class BSource {
    Copy,
    SameAsA,
    Reuse
};

__aicore__ inline uint32_t MmAlign(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

class CGDRMatmulBasic {
public:
    __aicore__ inline CGDRMatmulBasic() {}

    __aicore__ inline void Init()
    {
        l1aTensor_ = LocalTensor<bfloat16_t>(TPosition::A1, MM_L1A_OFFSET, MM_L1_ELEM_CNT);
        l1bTensor_ = LocalTensor<bfloat16_t>(TPosition::A1, MM_L1B_OFFSET, MM_L1_ELEM_CNT);
        l0aTensor_ = LocalTensor<bfloat16_t>(TPosition::A2, MM_L0A_ADDR, MM_L0_ELEM_CNT);
        l0bTensor_ = LocalTensor<bfloat16_t>(TPosition::B2, MM_L0B_ADDR, MM_L0_ELEM_CNT);
        l0cTensor_ = LocalTensor<float>(TPosition::CO1, MM_L0C_ADDR, MM_L0C_ELEM_CNT);
        SetFlag<HardEvent::FIX_MTE2>(FLAG_ZERO);
    }

    __aicore__ inline void End()
    {
        WaitFlag<HardEvent::FIX_MTE2>(FLAG_ZERO);
    }

    template <bool transA, bool transB, bool accum, typename dstType = bfloat16_t, BSource bSrc = BSource::Copy>
    __aicore__ inline void Execute(GlobalTensor<bfloat16_t> aGm, GlobalTensor<bfloat16_t> bGm,
                                   GlobalTensor<dstType> cGm, uint32_t m, uint32_t n, uint32_t k,
                                   uint32_t aGmRowStride = 0, uint32_t bGmRowStride = 0, uint32_t cGmRowStride = 0)
    {
        if ASCEND_IS_AIV {
            return;
        }

        uint32_t aRows = transA ? k : m;
        uint32_t aCols = transA ? m : k;
        uint32_t bRows = transB ? n : k;
        uint32_t bCols = transB ? k : n;
        uint32_t aSrcD = (aGmRowStride == 0) ? aCols : aGmRowStride;
        uint32_t bSrcD = (bGmRowStride == 0) ? bCols : bGmRowStride;
        uint32_t cDstStride = (cGmRowStride == 0) ? n : cGmRowStride;

        // (1) GM->L1
        WaitFlag<HardEvent::FIX_MTE2>(FLAG_ZERO);

        CopyGmToL1(aGm, aRows, aCols, aSrcD, l1aTensor_);
        if constexpr (bSrc == BSource::Copy) {
            CopyGmToL1(bGm, bRows, bCols, bSrcD, l1bTensor_);
        }

        SetFlag<HardEvent::MTE2_MTE1>(FLAG_ZERO);

        // (2) L1->L0
        WaitFlag<HardEvent::MTE2_MTE1>(FLAG_ZERO);

        LoadL1ToL0<transA>(l0aTensor_, l1aTensor_, aRows, aCols);
        if constexpr (bSrc == BSource::SameAsA) {
            LoadL1ToL0<!transB>(l0bTensor_, l1aTensor_, bRows, bCols);
        } else {
            LoadL1ToL0<!transB>(l0bTensor_, l1bTensor_, bRows, bCols);
        }

        SetFlag<HardEvent::MTE1_M>(FLAG_ZERO);

        // (3) Mmad
        WaitFlag<HardEvent::MTE1_M>(FLAG_ZERO);

        DoMmad(l0cTensor_, l0aTensor_, l0bTensor_, m, n, k);

        SetFlag<HardEvent::M_FIX>(FLAG_ZERO);

        // (4) L0C -> GM
        WaitFlag<HardEvent::M_FIX>(FLAG_ZERO);

        CopyL0CToGm<dstType, accum>(cGm, l0cTensor_, m, n, cDstStride);

        SetFlag<HardEvent::FIX_MTE2>(FLAG_ZERO);
    }

private:
    LocalTensor<bfloat16_t> l1aTensor_;
    LocalTensor<bfloat16_t> l1bTensor_;
    LocalTensor<bfloat16_t> l0aTensor_;
    LocalTensor<bfloat16_t> l0bTensor_;
    LocalTensor<float> l0cTensor_;

    __aicore__ inline void CopyGmToL1(GlobalTensor<bfloat16_t> &gm, uint32_t rows, uint32_t cols, uint32_t srcDStride,
                                      LocalTensor<bfloat16_t> &l1Tensor)
    {
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = rows;
        nd2nz.dValue = cols;
        nd2nz.srcDValue = srcDStride;
        nd2nz.dstNzC0Stride = MmAlign(rows, MM_BLOCK_CUBE);
        nd2nz.dstNzNStride = 1;
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(l1Tensor, gm, nd2nz);
    }

    template <bool ifTranspose>
    __aicore__ inline void LoadL1ToL0(LocalTensor<bfloat16_t> &l0Tensor, LocalTensor<bfloat16_t> &l1Tensor,
                                      uint32_t srcRows, uint32_t srcCols)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = MmAlign(srcRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.kStep = MmAlign(srcCols, MM_NZ_FRACTAL) / MM_NZ_FRACTAL;
        params.srcStride = params.mStep;
        params.dstStride = ifTranspose ? params.kStep : params.mStep;
        params.ifTranspose = ifTranspose;
        LoadData<bfloat16_t>(l0Tensor, l1Tensor, params);
    }

    __aicore__ inline void DoMmad(LocalTensor<float> &l0cTensor, LocalTensor<bfloat16_t> &l0aTensor,
                                  LocalTensor<bfloat16_t> &l0bTensor, uint32_t m, uint32_t n, uint32_t k)
    {
        MmadParams mmadParams;
        mmadParams.m = (m < MM_BLOCK_CUBE) ? MM_BLOCK_CUBE : m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        Mmad(l0cTensor, l0aTensor, l0bTensor, mmadParams);
    }

    template <typename dstType, bool accum>
    __aicore__ inline void CopyL0CToGm(GlobalTensor<dstType> &gm, LocalTensor<float> &l0cTensor, uint32_t m, uint32_t n,
                                       uint32_t dstStride)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixParams;
        fixParams.mSize = m;
        fixParams.nSize = n;
        fixParams.srcStride = MmAlign(m, MM_BLOCK_CUBE);
        fixParams.dstStride = dstStride;
        fixParams.unitFlag = 0;

        if constexpr (std::is_same_v<dstType, bfloat16_t>) {
            fixParams.quantPre = QuantMode_t::F322BF16;
        }

        if constexpr (accum) {
            SetAtomicAdd<dstType>();
        }
        Fixpipe<dstType, float, CFG_ROW_MAJOR>(gm, l0cTensor, fixParams);
        if constexpr (accum) {
            SetAtomicNone();
        }
    }
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_MATMUL_BASIC_H
