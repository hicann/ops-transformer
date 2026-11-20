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
 * \file mhc_pre_sinkhorn_backward_base_arch35.h
 * \brief MhcPre 与 Sinkhorn 反向融合的向量化 (VF) 计算函数、共享常量、缓冲区和 Workspace 定义
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_BASE_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_BASE_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"

namespace MhcPreSinkhornBackwardNs {

using namespace AscendC;

using MMConfig = MhcPreSinkhornBackwardMMConfig;

template <typename T>
__aicore__ auto CeilAlign(T a, T b) -> T
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ auto CeilDiv(T a, T b) -> T
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T lhs, T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint16_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

constexpr uint32_t NUMBER_TWO = 2U;
constexpr uint32_t KIBIBYTE = 1024U;
constexpr uint32_t DOUBLE_BUFFER = NUMBER_TWO;
constexpr uint32_t INOUT_QUEUE_SIZE = 32U * KIBIBYTE; // 32KB
constexpr uint32_t SMALL_QUEUE_SIZE = 1U * KIBIBYTE;  // 1KB
constexpr uint32_t UB_USABLE_SIZE = 248U * KIBIBYTE;
constexpr uint32_t MAIN_QUEUE_COUNT = 3U;
// FP32 计算缓冲区之外的小缓冲区，需同时为 VECOUT 双缓冲预留空间。
constexpr uint32_t EXTRA_SMALL_BUFS_SIZE = 1024U;
constexpr uint32_t FP32_BUF_SIZE =
    UB_USABLE_SIZE - MAIN_QUEUE_COUNT * DOUBLE_BUFFER * INOUT_QUEUE_SIZE - SMALL_QUEUE_SIZE - EXTRA_SMALL_BUFS_SIZE;
constexpr uint32_t PROCESS_V2_CHUNK_SIZE = 128U;
constexpr uint32_t BS_BLOCK_SIZE = 512U;
constexpr uint32_t ND_BLOCK_SIZE = 128U;
constexpr uint32_t ALPHA_GRAD_LAST_DIM_SIZE = 3;
constexpr uint32_t ALPHA_GRAD_PADDING = 24;
constexpr uint32_t ALPHA_GRAD_SHAPE_1_OFFSET = 0;
constexpr uint32_t ALPHA_GRAD_SHAPE_2_OFFSET = 8;
constexpr uint32_t ALPHA_GRAD_SHAPE_3_OFFSET = 16;

constexpr uint32_t LARGE_N_THRESHOLD = 6;
constexpr uint32_t VEC_CORE_VECIDX_MOD = NUMBER_TWO;
constexpr uint32_t MAX_D_LEN = 16384;
constexpr uint32_t VEC_DEAL_CHUNK_LARGE_N = 32;
constexpr uint32_t VEC_DEAL_CHUNK_SMALL_N = 64;
constexpr uint32_t N6_NONREUSE_MAX_ROWS = 32U;
constexpr uint32_t CEIL_ALIGN_DEFAULT = 32;
constexpr float NEG_HALF = -0.5f;
constexpr float ONE = 1.0f;
constexpr float ZERO = 0.0f;
constexpr uint32_t MATMUL_WRITE_OFFSET_M = 128;
constexpr uint32_t MATMUL_WRITE_OFFSET_N = 128;
constexpr uint32_t CROSS_CORE_FLAG_INDEX = NUMBER_TWO;
constexpr uint32_t CROSS_CORE_WAIT_FLAG_C0 = 0x9;
constexpr uint32_t CROSS_CORE_WAIT_FLAG_C1 = 0x8;
constexpr uint32_t C0_UB_BUFFER_FREE_FLAG_BASE = 0xA;
// queue0 必须首先分配并位于 UB 偏移 0；V0 结束后，两个队列缓冲区供 C0/V2 直传复用。
constexpr uint32_t C0_UB_BUFFER_OFFSET = 0;
static_assert(MATMUL_WRITE_OFFSET_M * (ND_BLOCK_SIZE / 2) * sizeof(float) <= INOUT_QUEUE_SIZE,
              "C0 output exceeds one UB buffer");
constexpr uint32_t VEC_DEAL_VECIDX_DIV = NUMBER_TWO;
// AIV02Process 对 Gather 布局执行整寄存器读写，每个临时区域预留 64 个通道。
constexpr uint32_t VECTOR_OVERWRITE_SLACK = 64U;

// 单独校验 V0/V1 共用的计算缓冲区，不包含 Sinkhorn 队列空间。
constexpr uint32_t MAX_SUPPORTED_N = 8U;
constexpr uint32_t FloatAlign(uint32_t elements)
{
    return (elements + 7U) / 8U * 8U;
}

constexpr bool V0V1BuffersFit()
{
    for (uint32_t n = 1; n <= MAX_SUPPORTED_N; ++n) {
        const uint32_t maxRows = n > LARGE_N_THRESHOLD ? VEC_DEAL_CHUNK_LARGE_N : VEC_DEAL_CHUNK_SMALL_N;
        const uint32_t fusionSize = n * n + 2U * n;
        const uint32_t fusionElements = maxRows * fusionSize;
        for (uint32_t rows = 1; rows <= maxRows; ++rows) {
            const uint32_t preElements = (rows * n + CEIL_ALIGN_DEFAULT - 1U) / CEIL_ALIGN_DEFAULT * CEIL_ALIGN_DEFAULT;
            const uint32_t scratchOffset = FloatAlign(fusionSize) + preElements + FloatAlign(rows * n) +
                                           3U * fusionElements + VECTOR_OVERWRITE_SLACK;
            const uint32_t scratchElements = fusionElements + VECTOR_OVERWRITE_SLACK;
            uint32_t power = 1U;
            while (power < rows) {
                power *= 2U;
            }
            // 非复用 RA 需要按 2 的幂补齐临时空间。N=6 且超过 32 行时复用源缓冲区。
            const uint32_t raScratch =
                n == 6U && rows > N6_NONREUSE_MAX_ROWS ? 0U : (power / 2U) * FloatAlign(fusionSize) * sizeof(float);
            const uint32_t arScratch = fusionSize > 64U ? rows * 64U * sizeof(float) : 0U;
            const uint32_t reduceScratch = raScratch > arScratch ? raScratch : arScratch;
            if ((scratchOffset + scratchElements) * sizeof(float) + reduceScratch > FP32_BUF_SIZE) {
                return false;
            }
        }
    }
    return true;
}
static_assert(V0V1BuffersFit(), "V0/V1 operands and reduction scratch exceed the FP32 UB buffer");

constexpr Reg::CastTrait ctHalf2Fp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::UNKNOWN};

struct InitParams {
    GM_ADDR x;
    GM_ADDR phi;
    GM_ADDR alpha;
    GM_ADDR grad_h_in;
    GM_ADDR grad_h_post;
    GM_ADDR grad_h_res;
    GM_ADDR inv_rms;
    GM_ADDR h_mix;
    GM_ADDR h_pre;
    GM_ADDR bias;
    GM_ADDR sum_out;
    GM_ADDR norm_out;
    GM_ADDR pre_mix;
    GM_ADDR grad_pre;
    GM_ADDR grad_x;
    GM_ADDR grad_phi;
    GM_ADDR grad_alpha;
    GM_ADDR grad_bias;
    GM_ADDR grad_pre_mix;
    GM_ADDR workspace;
    TPipe* tPipeIn;
    const MhcPreSinkhornBackwardArch35TilingData* tilingData;
};

struct OpRunInfo {
    uint64_t totalLength_ = 0; // 总行数 (B*S 或 T)
    uint64_t nD = 0;           // n * D
    uint64_t fusionSize = 0;   // 2N + N^2
};

template <class P>
struct V0V1Buffers {
    LocalTensor<P> hPreGradBuf;
    LocalTensor<P> hPreBufS1;
    LocalTensor<P> hPreBufS2;
    LocalTensor<P> hPostBufS1;
    LocalTensor<P> hPostBufS2;
    LocalTensor<P> hResBufS1;
    LocalTensor<P> hFusionBuf;
    LocalTensor<P> gatherFusionOutBuf;
    LocalTensor<P> invRmsBuf;
    LocalTensor<P> calcTmpBuf;
    LocalTensor<uint8_t> brcbTmpBuf;
    uint32_t stepLength;
    uint32_t gatherLength;
};

// Workspace 元素布局，M=totalLength，H=fusionSize：
// h_mix_grad[M,H]、alpha_grad[AIV,24]、bias_grad[AIV,H]、inv_rms_grad[M]；
// x_rs_grad_mm 和 x_rs 各占 [AIC,DOUBLE_BUFFER,BS_BLOCK_SIZE,ND_BLOCK_SIZE]。
template <class P>
struct WorkspaceBuffer {
    uint64_t totalLength; // 总行数（B*S 或 T）
    uint64_t fusionSize;  // 2N + N*N
    uint64_t coreNum;     // AIV 数量

    // 各区域偏移，单位为元素
    uint64_t hMixGradOffset;
    uint64_t alphaGradOffset;
    uint64_t biasGradOffset;
    uint64_t invRmsGradOffset;
    uint64_t xRsGradOffset;
    uint64_t xRsOffset; // V2 输出、C1 输入
    __aicore__ inline void Init(uint64_t bs, uint64_t fs, uint64_t vecCoreNum, uint64_t cubeCoreNum)
    {
        totalLength = bs;
        fusionSize = fs;
        coreNum = vecCoreNum;

        uint64_t alphaGradSize = ALPHA_GRAD_PADDING * vecCoreNum;
        uint64_t biasGradRows = vecCoreNum;

        hMixGradOffset = 0;
        alphaGradOffset = totalLength * fusionSize;
        biasGradOffset = alphaGradOffset + alphaGradSize;
        invRmsGradOffset = biasGradOffset + biasGradRows * fusionSize;
        xRsGradOffset = invRmsGradOffset + totalLength;
        xRsOffset = xRsGradOffset + BS_BLOCK_SIZE * ND_BLOCK_SIZE * DOUBLE_BUFFER * cubeCoreNum;
    }

    __aicore__ inline uint64_t GetHMixGradOffset(uint64_t bsIndex)
    {
        return hMixGradOffset + bsIndex * fusionSize;
    }

    // 每个 AIV 的三个 alpha 部分和分别占用一个 8 元素块。
    __aicore__ inline uint64_t GetAlphaGradOffset(uint32_t coreId)
    {
        return alphaGradOffset + coreId * ALPHA_GRAD_PADDING;
    }

    __aicore__ inline uint64_t GetBiasGradOffset(uint32_t coreId)
    {
        return biasGradOffset + coreId * fusionSize;
    }

    __aicore__ inline uint64_t GetInvRmsGradOffset(uint64_t bsIndex)
    {
        return invRmsGradOffset + bsIndex;
    }

    __aicore__ inline uint64_t GetXRsGradOffset(uint32_t coreId, uint32_t buffId)
    {
        return CeilAlign(xRsGradOffset, uint64_t(CEIL_ALIGN_DEFAULT)) +
               (coreId * DOUBLE_BUFFER + buffId) * BS_BLOCK_SIZE * ND_BLOCK_SIZE;
    }

    __aicore__ inline uint64_t GetXRsOffset(uint32_t coreId, uint32_t buffId)
    {
        return CeilAlign(xRsOffset, uint64_t(CEIL_ALIGN_DEFAULT)) +
               (coreId * DOUBLE_BUFFER + buffId) * BS_BLOCK_SIZE * ND_BLOCK_SIZE;
    }
};

// 每个向量寄存器可容纳的 P 类型元素数，与 Kernel 侧初始化逻辑一致。
template <class P>
__aicore__ inline uint16_t GetEleNumPerVf()
{
    return static_cast<uint16_t>(GetVRegSize() / sizeof(P));
}

// V0 预处理：x 与 grad_h_in 沿 D 维点积累加到 hPreGrad。
template <class T, class P, bool accumulate>
__aicore__ inline void VFDoPreProcessV0(__ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr, __ubuf__ T* gradHInB16InAddr,
                                        uint16_t curLenN, uint32_t lenD, uint64_t i, uint32_t j, uint64_t runBSStart,
                                        uint32_t n)
{
    uint16_t eleNumPerVf = 64;
    uint16_t dLoopCnt = (lenD + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg;
        uint32_t offset = static_cast<uint32_t>((i - runBSStart) * n + j);
        for (uint16_t offsetN = 0; offsetN < curLenN; offsetN++) { // x [curLenN, lenD],  hinGrad [1, lenD]
            Reg::Duplicate(sumReg, 0.0f);
            uint32_t curLenD = lenD;
            Reg::RegTensor<T> xInB16Reg, gradHInB16InReg;
            Reg::RegTensor<P> xFp32Reg, gradHInFp32Reg, mulReg, tmpSumPerVfReg;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < dLoopCnt; vfBlockIdx++) {
                Reg::MaskReg mask = Reg::UpdateMask<P>(curLenD);

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(
                    xInB16Reg, xB16InAddr + offsetN * lenD + vfBlockIdx * eleNumPerVf);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(gradHInB16InReg,
                                                                  gradHInB16InAddr + vfBlockIdx * eleNumPerVf);
                Reg::Cast<float, T, ctHalf2Fp32Zero>(gradHInFp32Reg, gradHInB16InReg, mask);

                Reg::Mul(mulReg, xFp32Reg, gradHInFp32Reg, mask);
                Reg::Reduce<Reg::ReduceType::SUM>(tmpSumPerVfReg, mulReg, mask);
                Reg::Add(sumReg, sumReg, tmpSumPerVfReg, mask); // 沿 D 维累加
            }
            uint32_t bufIdx = offset + offsetN;
            if constexpr (accumulate) {
                uint32_t one = 1;
                Reg::MaskReg scalarMask = Reg::UpdateMask<P>(one);
                Reg::RegTensor<P> previous;
                Reg::Load<P>(previous, hPreGradAddr + bufIdx);
                Reg::Add(sumReg, sumReg, previous, scalarMask);
            }
            Reg::Store(hPreGradAddr + bufIdx, sumReg, 1);
        }
    }
}

// V0：重算 sigmoid(z_pre) 门控并与上游梯度相乘。
template <class P>
__aicore__ inline void VFDoV0PreGrad(__ubuf__ P* gradPreOut, __ubuf__ P* hMixAddr, __ubuf__ P* invRmsAddr,
                                     __ubuf__ P* gradHPreAddr, __ubuf__ P* biasPreAddr, float alphaPre, uint32_t rows,
                                     uint32_t fusionSize, uint32_t n)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> alphaReg;
        Reg::RegTensor<P> oneReg;
        Reg::RegTensor<P> invReg;
        Reg::RegTensor<P> hMixReg;
        Reg::RegTensor<P> zReg;
        Reg::RegTensor<P> negReg;
        Reg::RegTensor<P> expReg;
        Reg::RegTensor<P> sigReg;
        Reg::RegTensor<P> gateReg;
        Reg::RegTensor<P> biasReg;
        Reg::RegTensor<P> upReg;
        Reg::RegTensor<P> outReg;
        Reg::MaskReg maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate(alphaReg, alphaPre, maskAll);
        Reg::Duplicate(oneReg, 1.0f, maskAll);
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
            uint32_t len = n;
            Reg::MaskReg mask = Reg::UpdateMask<P>(len);
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invReg, invRmsAddr + r);
            Reg::Load<P>(hMixReg, hMixAddr + r * fusionSize);
            Reg::Mul(zReg, hMixReg, invReg, mask);
            Reg::Mul(zReg, zReg, alphaReg, mask);
            Reg::Load<P>(biasReg, biasPreAddr);
            Reg::Add(zReg, zReg, biasReg, mask);
            Reg::Muls(negReg, zReg, -1.0f, mask);
            Reg::Exp(expReg, negReg, mask);
            Reg::Adds(expReg, expReg, 1.0f, mask);
            Reg::Div(sigReg, oneReg, expReg, mask);
            Reg::Exp(gateReg, zReg, mask);
            Reg::Adds(gateReg, gateReg, 1.0f, mask);
            Reg::Div(gateReg, oneReg, gateReg, mask);
            Reg::Mul(sigReg, sigReg, gateReg, mask);
            Reg::Load<P>(upReg, gradHPreAddr + r * n);
            Reg::Mul(outReg, sigReg, upReg, mask);
            Reg::Store<P>(gradPreOut + r * n, outReg, n);
        }
    }
}

// V0：重算 post 门控（2*sigmoid 门）并与上游梯度相乘。
template <class P>
__aicore__ inline void VFDoV0PostGrad(__ubuf__ P* gradPostOut, __ubuf__ P* hMixAddr, __ubuf__ P* invRmsAddr,
                                      __ubuf__ P* gradHPostAddr, __ubuf__ P* biasPostAddr, float alphaPost,
                                      uint32_t rows, uint32_t fusionSize, uint32_t n)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> alphaReg;
        Reg::RegTensor<P> oneReg;
        Reg::RegTensor<P> invReg;
        Reg::RegTensor<P> hMixReg;
        Reg::RegTensor<P> zReg;
        Reg::RegTensor<P> negReg;
        Reg::RegTensor<P> expReg;
        Reg::RegTensor<P> sigReg;
        Reg::RegTensor<P> gateReg;
        Reg::RegTensor<P> biasReg;
        Reg::RegTensor<P> upReg;
        Reg::RegTensor<P> outReg;
        Reg::MaskReg maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate(alphaReg, alphaPost, maskAll);
        Reg::Duplicate(oneReg, 1.0f, maskAll);
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
            uint32_t len = n;
            Reg::MaskReg mask = Reg::UpdateMask<P>(len);
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invReg, invRmsAddr + r);
            Reg::Load<P>(hMixReg, hMixAddr + r * fusionSize + n);
            Reg::Mul(zReg, hMixReg, invReg, mask);
            Reg::Mul(zReg, zReg, alphaReg, mask);
            Reg::Load<P>(biasReg, biasPostAddr);
            Reg::Add(zReg, zReg, biasReg, mask);
            Reg::Muls(negReg, zReg, -1.0f, mask);
            Reg::Exp(expReg, negReg, mask);
            Reg::Adds(expReg, expReg, 1.0f, mask);
            Reg::Div(sigReg, oneReg, expReg, mask);
            Reg::Exp(gateReg, zReg, mask);
            Reg::Adds(gateReg, gateReg, 1.0f, mask);
            Reg::Div(gateReg, oneReg, gateReg, mask);
            Reg::Mul(sigReg, sigReg, gateReg, mask);
            Reg::Muls(sigReg, sigReg, 2.0f, mask);
            Reg::Load<P>(upReg, gradHPostAddr + r * n);
            Reg::Mul(outReg, sigReg, upReg, mask);
            Reg::Store<P>(gradPostOut + r * n, outReg, n);
        }
    }
}

// 每寄存器按 8×8 通道处理 8 行，独立归约最大值、首个最大值下标及加权和。
template <class P, uint32_t N>
__aicore__ inline void VFDoV0PackedRes(__ubuf__ P* output, __ubuf__ P* hc, __ubuf__ P* inv, __ubuf__ P* upstream,
                                       __ubuf__ P* bias, float alpha, uint32_t rows)
{
    constexpr uint32_t H = N * N + 2U * N;
    const uint32_t rowCount = rows * N;
    __VEC_SCOPE__
    {
        Reg::MaskReg full = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint32_t> order, group, column, absoluteRow, batch, matrixRow, stride;
        Reg::RegTensor<uint32_t> gradientIndex, hcIndex, biasIndex, candidate, first, firstBroadcast, sentinel;
        Reg::Arange((Reg::RegTensor<int32_t>&)order, 0);
        Reg::ShiftRights(group, order, static_cast<int16_t>(3), full);
        Reg::Duplicate(stride, 7U);
        Reg::And(column, order, stride, full);
        Reg::Duplicate(sentinel, N);
        Reg::RegTensor<P> a, invReg, mix, b, z, maximum, maxBroadcast, expReg, weighted, total, totalBroadcast, zero;
        Reg::Duplicate(a, alpha);
        Reg::Duplicate(zero, 0.0f);
        for (uint16_t block = 0; block < static_cast<uint16_t>((rowCount + 7U) / 8U); ++block) {
            Reg::Adds(absoluteRow, group, static_cast<uint32_t>(block) * 8U, full);
            Reg::MaskReg validRows, validCols, active, equalMax, firstMask;
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(validRows, absoluteRow, rowCount, full);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(validCols, column, N, full);
            Reg::And(active, validRows, validCols, full);
            if constexpr (N == 8U) {
                Reg::ShiftRights(batch, absoluteRow, static_cast<int16_t>(3), full);
            } else {
                Reg::Duplicate(stride, N);
                Reg::Div(batch, absoluteRow, stride, full);
            }
            Reg::Duplicate(stride, N);
            Reg::Mul(matrixRow, batch, stride, full);
            Reg::Sub(matrixRow, absoluteRow, matrixRow, full);
            Reg::Mul(gradientIndex, absoluteRow, stride, full);
            Reg::Add(gradientIndex, gradientIndex, column, full);
            Reg::Mul(biasIndex, matrixRow, stride, full);
            Reg::Add(biasIndex, biasIndex, column, full);
            Reg::Duplicate(stride, H);
            Reg::Mul(hcIndex, batch, stride, full);
            Reg::Add(hcIndex, hcIndex, biasIndex, full);
            Reg::Adds(hcIndex, hcIndex, 2U * N, full);
            Reg::DataCopyGather(invReg, inv, batch, active);
            Reg::DataCopyGather(mix, hc, hcIndex, active);
            Reg::DataCopyGather(b, bias, biasIndex, active);
            Reg::Mul(z, mix, invReg, active);
            Reg::Mul(z, z, a, active);
            Reg::Add(z, z, b, active);
            Reg::ReduceMaxWithDataBlock(maximum, z, active);
            Reg::Gather(maxBroadcast, maximum, group);
            Reg::Compare<P, CMPMODE::EQ>(equalMax, z, maxBroadcast, active);
            Reg::Select(candidate, column, sentinel, equalMax);
            Reg::ReduceMinWithDataBlock(first, candidate, active);
            Reg::Gather(firstBroadcast, first, group);
            Reg::Compare<uint32_t, CMPMODE::EQ>(firstMask, column, firstBroadcast, active);
            Reg::Sub(expReg, z, maxBroadcast, active);
            Reg::Exp(expReg, expReg, active);
            Reg::DataCopyGather(weighted, upstream, gradientIndex, active);
            Reg::Mul(weighted, expReg, weighted, active);
            Reg::ReduceSumWithDataBlock(total, weighted, active);
            Reg::Gather(totalBroadcast, total, group);
            Reg::Select(totalBroadcast, totalBroadcast, zero, firstMask);
            Reg::Sub(weighted, weighted, totalBroadcast, active);
            Reg::DataCopyScatter(output, weighted, gradientIndex, active);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

// V0：Sinkhorn 反向完成后计算 exp/max 的梯度。
template <class P>
__aicore__ inline void VFDoV0ResGrad(__ubuf__ P* gradResOut, __ubuf__ P* hMixAddr, __ubuf__ P* invRmsAddr,
                                     __ubuf__ P* skAddr, __ubuf__ P* biasResAddr, float alphaRes, uint32_t rows,
                                     uint32_t fusionSize, uint32_t n)
{
    if (n == 6U) {
        VFDoV0PackedRes<P, 6U>(gradResOut, hMixAddr, invRmsAddr, skAddr, biasResAddr, alphaRes, rows);
        return;
    }
    if (n == 8U) {
        VFDoV0PackedRes<P, 8U>(gradResOut, hMixAddr, invRmsAddr, skAddr, biasResAddr, alphaRes, rows);
        return;
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<P> invReg;
        Reg::RegTensor<P> hMixReg;
        Reg::RegTensor<P> uReg;
        Reg::RegTensor<P> zReg;
        Reg::RegTensor<P> biasReg;
        Reg::RegTensor<P> alphaReg;
        Reg::RegTensor<P> maxReg;
        Reg::RegTensor<P> maxBroadcast;
        Reg::RegTensor<P> expReg;
        Reg::RegTensor<P> upstreamReg;
        Reg::RegTensor<P> weighted;
        Reg::RegTensor<P> weightedSum;
        Reg::RegTensor<P> weightedBroadcast;
        Reg::RegTensor<P> zeroReg;
        Reg::RegTensor<P> correction;
        Reg::RegTensor<P> gzReg;
        Reg::RegTensor<int32_t> order;
        Reg::RegTensor<int32_t> sentinel;
        Reg::RegTensor<int32_t> candidate;
        Reg::RegTensor<int32_t> firstIndex;
        Reg::RegTensor<int32_t> firstBroadcast;
        uint32_t sregN = n;
        Reg::MaskReg maskN = Reg::UpdateMask<P>(sregN);
        Reg::MaskReg equalMax;
        Reg::MaskReg firstMask;
        Reg::Duplicate(alphaReg, alphaRes, maskN);
        Reg::Duplicate(zeroReg, 0.0f, maskN);
        Reg::Arange(order, static_cast<int32_t>(0));
        Reg::Duplicate(sentinel, static_cast<int32_t>(n), maskN);
        uint32_t nn = n * n;
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invReg, invRmsAddr + r);
            for (uint16_t row = 0; row < static_cast<uint16_t>(n); ++row) {
                Reg::Load<P>(hMixReg, hMixAddr + r * fusionSize + 2 * n + row * n);
                Reg::Load<P>(biasReg, biasResAddr + row * n);
                Reg::Mul(uReg, hMixReg, invReg, maskN);
                Reg::Mul(zReg, uReg, alphaReg, maskN);
                Reg::Add(zReg, zReg, biasReg, maskN);
                Reg::Reduce<Reg::ReduceType::MAX>(maxReg, zReg, maskN);
                Reg::Duplicate(maxBroadcast, maxReg, maskN);
                Reg::Compare<P, CMPMODE::EQ>(equalMax, zReg, maxBroadcast, maskN);
                Reg::Select(candidate, order, sentinel, equalMax);
                Reg::Reduce<Reg::ReduceType::MIN>(firstIndex, candidate, maskN);
                Reg::Duplicate(firstBroadcast, firstIndex, maskN);
                Reg::Compare<int32_t, CMPMODE::EQ>(firstMask, order, firstBroadcast, maskN);
                Reg::Sub(expReg, zReg, maxBroadcast, maskN);
                Reg::Exp(expReg, expReg, maskN);
                Reg::Load<P>(upstreamReg, skAddr + r * nn + row * n);
                Reg::Mul(weighted, expReg, upstreamReg, maskN);
                Reg::Reduce<Reg::ReduceType::SUM>(weightedSum, weighted, maskN);
                Reg::Duplicate(weightedBroadcast, weightedSum, maskN);
                Reg::Select(correction, weightedBroadcast, zeroReg, firstMask);
                Reg::Sub(gzReg, weighted, correction, maskN);
                Reg::Store<P>(gradResOut + r * nn + row * n, gzReg, n);
            }
        }
    }
}

template <class P>
__aicore__ inline void VFDoV0ProcessGradHPost(__ubuf__ P* hPostIn, __ubuf__ P* gradHPostIn, __ubuf__ P* gradHPostOut,
                                              uint32_t stepLen)
{
    const uint16_t eleNumPerVf = GetEleNumPerVf<P>();
    uint16_t loopCnt = CeilDiv(stepLen, uint32_t(eleNumPerVf));
    uint32_t curLen = stepLen;
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<P> gradHPostReg, tmpReg;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            mask = Reg::UpdateMask<P>(curLen);
            Reg::LoadAlign(gradHPostReg, hPostIn + vfBlockIdx * eleNumPerVf);
            Reg::Muls(tmpReg, gradHPostReg, NEG_HALF, mask);
            Reg::Adds(tmpReg, tmpReg, ONE, mask);
            Reg::Mul(gradHPostReg, gradHPostReg, tmpReg, mask);
            Reg::LoadAlign(tmpReg, gradHPostIn + vfBlockIdx * eleNumPerVf);
            Reg::Mul(gradHPostReg, gradHPostReg, tmpReg, mask);
            Reg::StoreAlign(gradHPostOut + vfBlockIdx * eleNumPerVf, gradHPostReg, mask);
        }
    }
}

// AIV02Process 使用整寄存器写入拼接的 dL/dz。
template <class P>
__aicore__ inline void AIV02Process(V0V1Buffers<P>& buffers, LocalTensor<P>& hMixGradBuf, uint64_t currentDealBsNum,
                                    float alphaPre, float alphaPost, float alphaComb, uint32_t n)
{
    __ubuf__ P* hPreBufAddr = (__ubuf__ P*)buffers.hPreBufS1.GetPhyAddr();
    __ubuf__ P* hPostBufAddr = (__ubuf__ P*)buffers.hPostBufS1.GetPhyAddr();
    __ubuf__ P* hResBufAddr = (__ubuf__ P*)buffers.hResBufS1.GetPhyAddr();
    __ubuf__ P* gatherFusionOutBufAddr = (__ubuf__ P*)buffers.gatherFusionOutBuf.GetPhyAddr();
    __ubuf__ P* hMixGradBufAddr = (__ubuf__ P*)hMixGradBuf.GetPhyAddr();

    uint32_t blockLenPre = n;
    uint32_t blockLenPost = n;
    uint32_t blockLenComb = n * n;
    uint32_t totalBlockLen = blockLenPre + blockLenPost + blockLenComb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<P> gatherReg;
        Reg::RegTensor<P> hMixGradReg;
        Reg::RegTensor<P> alphaPreReg;
        Reg::RegTensor<P> alphaPostReg;
        Reg::RegTensor<P> alphaCombReg;
        Reg::MaskReg maskPre = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskPost = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskComb = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate<P>(alphaPreReg, alphaPre, maskPre);
        Reg::Duplicate<P>(alphaPostReg, alphaPost, maskPost);
        Reg::Duplicate<P>(alphaCombReg, alphaComb, maskComb);

        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(currentDealBsNum); bsIdx++) {
            uint32_t curLenPre = blockLenPre;
            uint32_t curLenPost = blockLenPost;
            uint32_t curLenComb = blockLenComb;
            uint32_t bsFusionOffset = bsIdx * totalBlockLen;

            uint32_t srcOffset = bsIdx * blockLenPre;
            uint32_t dstOffset = bsFusionOffset;
            maskPre = Reg::UpdateMask<P>(curLenPre);
            Reg::Load<P>(gatherReg, hPreBufAddr + srcOffset);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaPreReg, maskPre);
            Reg::Store<P>(hMixGradBufAddr + dstOffset, hMixGradReg);

            uint32_t srcOffset1 = bsIdx * blockLenPre;
            uint32_t dstOffset1 = bsFusionOffset;
            maskPost = Reg::UpdateMask<P>(curLenPost);
            Reg::Load<P>(gatherReg, hPostBufAddr + srcOffset1);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset1 + blockLenPre, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaPostReg, maskPost);
            Reg::Store<P>(hMixGradBufAddr + dstOffset1 + blockLenPre, hMixGradReg);

            uint32_t srcOffset2 = bsIdx * blockLenComb;
            uint32_t dstOffset2 = bsFusionOffset + blockLenPre + blockLenPost;
            maskComb = Reg::UpdateMask<P>(curLenComb);
            Reg::Load<P>(gatherReg, hResBufAddr + srcOffset2);
            Reg::Store<P>(gatherFusionOutBufAddr + dstOffset2, gatherReg);
            Reg::Mul(hMixGradReg, gatherReg, alphaCombReg, maskComb);
            Reg::Store<P>(hMixGradBufAddr + dstOffset2, hMixGradReg);
        }
    }
}

// V1：拼接 dL/dz 求和、dL/dh_mix 与 dL/d(alpha) 部分和。
template <class P>
__aicore__ inline void VFDoV1Process(__ubuf__ P* gradHMixOut, __ubuf__ P* hMixGradBuf, __ubuf__ P* gradAlphaOut,
                                     __ubuf__ P* gatherIn, __ubuf__ P* hMixIn, __ubuf__ P* invRmsIn, uint32_t curBSSize,
                                     uint32_t fusionSize)
{
    const uint16_t eleNumPerVf = GetEleNumPerVf<P>();
    uint32_t totalElem = curBSSize * fusionSize;
    uint16_t nLoopCnt = static_cast<uint16_t>(CeilDiv(totalElem, uint32_t(eleNumPerVf)));
    uint32_t curElemCnt = totalElem;

    __VEC_SCOPE__
    {
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf;
            Reg::MaskReg mask = Reg::UpdateMask<P>(curElemCnt);

            Reg::RegTensor<P> h1GradReg, invRmsReg, hMixReg, gatherReg;
            Reg::RegTensor<P> hMixGradReg, invRmsGradReg;
            Reg::RegTensor<P> hMul1Reg, hMul2Reg, hMul3Reg;

            Reg::LoadAlign(h1GradReg, hMixGradBuf + elemOffset);
            Reg::LoadAlign(invRmsReg, invRmsIn + elemOffset);

            Reg::Mul(hMixGradReg, h1GradReg, invRmsReg, mask);
            Reg::StoreAlign(gradHMixOut + elemOffset, hMixGradReg, mask);

            Reg::LoadAlign(hMixReg, hMixIn + elemOffset);
            Reg::Mul(hMul1Reg, h1GradReg, hMixReg, mask);
            Reg::StoreAlign(hMixGradBuf + elemOffset, hMul1Reg, mask);

            Reg::LoadAlign(gatherReg, gatherIn + elemOffset);
            Reg::Mul(hMul2Reg, invRmsReg, hMixReg, mask);
            Reg::Mul(hMul3Reg, hMul2Reg, gatherReg, mask);
            Reg::StoreAlign(gradAlphaOut + elemOffset, hMul3Reg, mask);
        }
    }
}

template <class P>
__aicore__ inline void VFDoV1ProcessBiasGradForN8(__ubuf__ P* outBufDst, __ubuf__ P* gatherFusion, uint32_t curBSSize,
                                                  uint32_t fusionSize)
{
    const uint16_t eleNumPerVf = GetEleNumPerVf<P>();
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg1, sumReg2;
        Reg::Duplicate(sumReg1, 0);
        Reg::Duplicate(sumReg2, 0);
        uint32_t dealMask1 = eleNumPerVf;
        uint32_t dealMask2 = fusionSize - eleNumPerVf;
        Reg::MaskReg mask1 = Reg::UpdateMask<P>(dealMask1);
        Reg::MaskReg mask2 = Reg::UpdateMask<P>(dealMask2);
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(curBSSize); ++bsIdx) {
            uint32_t elemOffset1 = bsIdx * fusionSize;
            uint32_t elemOffset2 = bsIdx * fusionSize + eleNumPerVf;
            Reg::RegTensor<P> gatherReg1, gatherReg2;

            Reg::LoadAlign(gatherReg1, gatherFusion + elemOffset1);
            Reg::LoadAlign(gatherReg2, gatherFusion + elemOffset2);

            Reg::Add(sumReg1, sumReg1, gatherReg1, mask1);
            Reg::Add(sumReg2, sumReg2, gatherReg2, mask2);
        }
        Reg::StoreAlign(outBufDst, sumReg1, mask1);
        Reg::StoreAlign(outBufDst + eleNumPerVf, sumReg2, mask2);
    }
}

template <class P, bool aligned>
__aicore__ inline void VFDoV1ProcessBiasGradForN4N6(__ubuf__ P* outBufDst, __ubuf__ P* gatherFusion, uint32_t curBSSize,
                                                    uint32_t fusionSize)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg;
        Reg::Duplicate(sumReg, 0);
        uint32_t dealMask = fusionSize;
        Reg::MaskReg mask = Reg::UpdateMask<P>(dealMask);
        for (uint16_t bsIdx = 0; bsIdx < static_cast<uint16_t>(curBSSize); ++bsIdx) {
            uint32_t elemOffset1 = bsIdx * fusionSize;
            Reg::RegTensor<P> gatherReg;

            if constexpr (aligned) {
                Reg::LoadAlign(gatherReg, gatherFusion + elemOffset1);
            } else {
                // fusionSize 为奇数时，相邻行的 UB 对齐位置不同。
                Reg::Load<P>(gatherReg, gatherFusion + elemOffset1);
            }

            Reg::Add(sumReg, sumReg, gatherReg, mask);
        }
        Reg::StoreAlign(outBufDst, sumReg, mask);
    }
}

template <class P>
__aicore__ inline void VFDoV1InvRmsCoeff(__ubuf__ P* invRmsGradAddr, __ubuf__ P* invRmsAddr, uint32_t count,
                                         float scaleMean)
{
    const uint16_t eleNumPerVf = GetEleNumPerVf<P>();
    uint16_t loopCnt = CeilDiv(count, uint32_t(eleNumPerVf));
    uint32_t curLen = count;
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> invReg, shMixReg, cubeReg, gradReg, outReg, scaleReg;
        Reg::MaskReg maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate(scaleReg, -scaleMean, maskAll);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf;
            Reg::MaskReg mask = Reg::UpdateMask<P>(curLen);
            Reg::LoadAlign(invReg, invRmsAddr + elemOffset);
            Reg::Mul(shMixReg, invReg, invReg, mask);
            Reg::Mul(cubeReg, shMixReg, invReg, mask);
            Reg::LoadAlign(gradReg, invRmsGradAddr + elemOffset);
            Reg::Mul(outReg, cubeReg, gradReg, mask);
            Reg::Mul(outReg, outReg, scaleReg, mask);
            Reg::StoreAlign(invRmsGradAddr + elemOffset, outReg, mask);
        }
    }
}

template <class P>
__aicore__ inline void AIV21Process(LocalTensor<P>& invRmsInBuf, LocalTensor<P>& invRmsUb, LocalTensor<P>& invRmsGradUb,
                                    uint32_t currentChunkSize, float scaleMean)
{
    const uint16_t eleNumPerVf = GetEleNumPerVf<P>();
    __ubuf__ P* invRmsInBufAddr = (__ubuf__ P*)invRmsInBuf.GetPhyAddr();
    __ubuf__ P* invRmsUbAddr = (__ubuf__ P*)invRmsUb.GetPhyAddr();
    __ubuf__ P* invRmsGradUbAddr = (__ubuf__ P*)invRmsGradUb.GetPhyAddr();

    uint32_t vfloopCnt = CeilDiv(currentChunkSize, uint32_t(eleNumPerVf));
    uint32_t curLen = currentChunkSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<P> invRmsInReg;
        Reg::RegTensor<P> invRmsTempReg;
        Reg::RegTensor<P> invRmsUbReg;
        Reg::RegTensor<P> invRmsGradUb;
        Reg::RegTensor<P> scaleMeanReg;
        Reg::MaskReg maskAll, mask;
        maskAll = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::Duplicate<P>(scaleMeanReg, (-scaleMean), maskAll);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < static_cast<uint16_t>(vfloopCnt); vfBlockIdx++) {
            uint32_t elemOffset = vfBlockIdx * eleNumPerVf;
            mask = Reg::UpdateMask<P>(curLen);
            Reg::LoadAlign(invRmsInReg, invRmsInBufAddr + elemOffset);
            Reg::Mul(invRmsTempReg, invRmsInReg, invRmsInReg, mask);
            Reg::Mul(invRmsUbReg, invRmsTempReg, invRmsInReg, mask);
            Reg::LoadAlign(invRmsGradUb, invRmsGradUbAddr + elemOffset);
            Reg::Mul(invRmsUbReg, invRmsUbReg, invRmsGradUb, mask);
            Reg::Mul(invRmsUbReg, invRmsUbReg, scaleMeanReg, mask);
            Reg::StoreAlign(invRmsUbAddr + elemOffset, invRmsUbReg, mask);
        }
    }
}

// V2：x 转 FP32，保留 FP32 副本供梯度计算并输出给 C1。
template <class T, class P>
__aicore__ inline void VFDoV2CastXToFp32(__ubuf__ P* xRsFp32Addr, __ubuf__ P* xRsOutAddr, __ubuf__ T* bf16InputAddr,
                                         uint32_t currentChunkSize, uint32_t copySizeND)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xInB16Reg;
        Reg::RegTensor<P> xFp32Reg;
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            // copySizeND <= 64，每次处理一行。
            uint32_t lenND = copySizeND;
            Reg::MaskReg mask = Reg::UpdateMask<P>(lenND);
            uint32_t offset = tIdx * copySizeND;

            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xInB16Reg, bf16InputAddr + offset);
            Reg::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

            Reg::StoreAlign(xRsFp32Addr + offset, xFp32Reg, mask); // 后续复用 xFp32

            Reg::StoreAlign(xRsOutAddr + offset, xFp32Reg, mask);
        }
    }
}

template <class P>
__aicore__ inline void VFDoV2AccumulateXGrad(__ubuf__ P* xGradVec3Addr, __ubuf__ P* xRsGradMmAddr,
                                             __ubuf__ P* xRsFp32Addr, __ubuf__ P* invRmsAddr, uint32_t currentChunkSize,
                                             uint32_t copySizeND)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> xRsFp32Reg, xRsGradMmReg, xRsGradReg, xGradReg;
        Reg::RegTensor<P> invRmsReg;
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            // 每行广播加载一个 FP32 invRms 标量
            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(invRmsReg, invRmsAddr + tIdx);
            // copySizeND <= 64，每次处理一行。
            Reg::MaskReg mask = Reg::UpdateMask<P>(lenND);
            uint32_t offset = tIdx * copySizeND;
            Reg::LoadAlign(xRsGradMmReg, xRsGradMmAddr + offset);
            Reg::LoadAlign(xGradReg, xGradVec3Addr + offset);
            Reg::LoadAlign(xRsFp32Reg, xRsFp32Addr + offset);

            Reg::Mul(xRsGradReg, xRsFp32Reg, invRmsReg, mask);
            Reg::Add(xGradReg, xGradReg, xRsGradReg, mask);

            Reg::Add(xGradReg, xGradReg, xRsGradMmReg, mask);
            Reg::StoreAlign(xGradVec3Addr + offset, xGradReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void VFDoV2HInMulHPre(__ubuf__ P* xGradVec3BufAddr, __ubuf__ T* hInGradInBufAddr,
                                        __ubuf__ P* hPreUbAddr, uint32_t currentChunkSize, uint32_t copySizeND,
                                        uint32_t currentN, uint32_t bsGlobalOffset, uint32_t n)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> hInReg;
        Reg::RegTensor<P> hInFP32Reg, hPreReg, hPreFullReg;
        Reg::MaskReg mask;
        uint32_t hPreOffset = bsGlobalOffset * n + currentN;
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            uint32_t hInOffset = tIdx * copySizeND;

            Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(hPreFullReg, hPreUbAddr + hPreOffset);
            // copySizeND <= 64，每次处理一行。
            mask = Reg::UpdateMask<P>(lenND);
            // hIn 转为 FP32
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(hInReg, hInGradInBufAddr + hInOffset);
            Reg::Cast<float, T, ctHalf2Fp32Zero>(hInFP32Reg, hInReg, mask);
            Reg::Mul(hInFP32Reg, hInFP32Reg, hPreFullReg, mask);
            Reg::StoreAlign(xGradVec3BufAddr + hInOffset, hInFP32Reg, mask);
            hPreOffset += n;
        }
    }
}

template <class T, class P>
__aicore__ inline void VFDoV2CastXGrad(__ubuf__ T* outputAddr, __ubuf__ P* xGradAddr, uint32_t totalElem)
{
    const uint16_t eleNumPerVf = GetEleNumPerVf<P>();
    uint16_t vfLoopCnt = CeilDiv(totalElem, static_cast<uint32_t>(eleNumPerVf));
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> xGradReg;
        Reg::RegTensor<T> outputReg;
        static constexpr Reg::CastTrait castFp32ToB16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        uint32_t len = totalElem;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
            Reg::MaskReg mask = Reg::UpdateMask<P>(len);
            uint32_t offset = vfBlockIdx * eleNumPerVf;
            Reg::LoadAlign(xGradReg, xGradAddr + offset);
            Reg::Cast<T, P, castFp32ToB16>(outputReg, xGradReg, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputAddr + offset, outputReg, mask);
        }
    }
}

// V3：单 AIV 时按三个有效标量输出 alpha 梯度。
template <class P>
__aicore__ inline void VFDoV3ProcessAlphaGradSingleCore(__ubuf__ P* alphaGradOut, __ubuf__ P* alphaGradIn)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg mask = Reg::CreateMask<P, Reg::MaskPattern::ALL>();
        Reg::RegTensor<P> zero, value, sum;
        Reg::Duplicate(zero, 0.0f, mask);
        // 保持 V3 中 0+partial 的累加方式，仅加载三个有效标量。
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(value, alphaGradIn);
        Reg::Add(sum, zero, value, mask);
        Reg::Store<P>(alphaGradOut, sum, 1);
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(value, alphaGradIn + ALPHA_GRAD_SHAPE_2_OFFSET);
        Reg::Add(sum, zero, value, mask);
        Reg::Store<P>(alphaGradOut + 1, sum, 1);
        Reg::DataCopy<P, Reg::LoadDist::DIST_BRC_B32>(value, alphaGradIn + ALPHA_GRAD_SHAPE_3_OFFSET);
        Reg::Add(sum, zero, value, mask);
        Reg::Store<P>(alphaGradOut + 2, sum, 1);
    }
}

// V3：多 AIV 时累加各核 alpha 部分和并按步长抽取三个标量。
template <class P>
__aicore__ inline void VFDoV3ProcessAlphaGrad(__ubuf__ P* alphaGradOut, __ubuf__ P* alphaGradIn,
                                              uint32_t usedVecCoreNum)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<P> sumReg;
        Reg::Duplicate(sumReg, 0);
        uint32_t dealMask = ALPHA_GRAD_PADDING;
        Reg::MaskReg mask = Reg::UpdateMask<P>(dealMask);
        for (uint16_t vcIdx = 0; vcIdx < static_cast<uint16_t>(usedVecCoreNum); ++vcIdx) {
            uint32_t elemOffset = vcIdx * ALPHA_GRAD_PADDING;
            Reg::RegTensor<P> alphaGradInReg;
            Reg::LoadAlign(alphaGradInReg, alphaGradIn + elemOffset);
            Reg::Add(sumReg, sumReg, alphaGradInReg, mask);
        }
        uint32_t outputCount = ALPHA_GRAD_LAST_DIM_SIZE;
        Reg::MaskReg outputMask = Reg::UpdateMask<P>(outputCount);
        Reg::RegTensor<uint32_t> index, zeroIndex;
        Reg::Arange((Reg::RegTensor<int32_t>&)index, static_cast<int32_t>(0));
        Reg::Muls(index, index, static_cast<uint32_t>(ALPHA_GRAD_SHAPE_2_OFFSET), mask);
        Reg::Duplicate(zeroIndex, 0U);
        Reg::Select(index, index, zeroIndex, outputMask);
        Reg::RegTensor<P> packed;
        Reg::Gather(packed, sumReg, index);
        Reg::StoreAlign(alphaGradOut, packed, outputMask);
    }
}

} // namespace MhcPreSinkhornBackwardNs

#endif // MHC_PRE_SINKHORN_BACKWARD_BASE_ARCH35_H
