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
 * \file mhc_pre_sinkhorn_backward_utils_arch35.h
 * \brief MhcPre 与 Sinkhorn 反向融合的常量、缓冲区和 Workspace 定义
 */

#ifndef ASCENDC_MHC_PRE_SINKHORN_BACKWARD_UTILS_H
#define ASCENDC_MHC_PRE_SINKHORN_BACKWARD_UTILS_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"

namespace MhcPreSinkhornBackwardUtils {

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

using namespace AscendC;

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
    GM_ADDR grad_x;
    GM_ADDR grad_phi;
    GM_ADDR grad_alpha;
    GM_ADDR grad_bias;
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
} // namespace MhcPreSinkhornBackwardUtils

#endif // ASCENDC_MHC_PRE_SINKHORN_BACKWARD_UTILS_H
