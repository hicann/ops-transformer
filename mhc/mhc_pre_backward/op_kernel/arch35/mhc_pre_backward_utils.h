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
 * \file mhc_pre_backward_utils.h
 * \brief
 */

#ifndef ASCENDC_MHC_PRE_BACKWARD_UTILS_H
#define ASCENDC_MHC_PRE_BACKWARD_UTILS_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace MhcPreBackwardUtils {

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

using namespace matmul;
using namespace AscendC;

constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t INOUT_QUEUE_SIZE = 32 * 1024;              // 32KB
constexpr uint32_t SMALL_QUEUE_SIZE = 1 * 1024;               // 1KB
constexpr uint32_t FP32_BUF_SIZE = (248 - 32 * 3 - 1) * 1024; // 151KB
constexpr uint32_t PROCESS_V2_CHUNK_SIZE = 128;               // ProcessV2函数使用的chunk大小
constexpr uint32_t SINGLE_M = 1024;
constexpr uint32_t ND_BLOCK_SIZE = 128;
constexpr uint32_t ALPHA_GRAD_LAST_DIM_SIZE = 3;
constexpr uint32_t ALPHA_GRAD_PADDING = 24;
constexpr uint32_t ALPHA_GRAD_SHAPE_1_OFFSET = 0;
constexpr uint32_t ALPHA_GRAD_SHAPE_2_OFFSET = 8;
constexpr uint32_t ALPHA_GRAD_SHAPE_3_OFFSET = 16;

constexpr uint32_t LARGE_N_THRESHOLD = 6;
constexpr uint32_t VEC_CORE_VECIDX_MOD = 2;
constexpr uint32_t MAX_D_LEN = 16384;
constexpr uint32_t VEC_DEAL_CHUNK_LARGE_N = 32;
constexpr uint32_t VEC_DEAL_CHUNK_SMALL_N = 64;
constexpr uint32_t CEIL_ALIGN_DEFAULT = 32;
constexpr uint32_t CEIL_ALIGN_16 = 16;
constexpr uint32_t CEIL_ALIGN_128 = 128;
constexpr float NEG_HALF = -0.5f;
constexpr float ONE = 1.0f;
constexpr float ZERO = 0.0f;
constexpr uint32_t MATMUL_WRITE_OFFSET_M = 256;
constexpr uint32_t MATMUL_WRITE_OFFSET_N = 128;
constexpr uint32_t CROSS_CORE_FLAG_INDEX = 0x2;
constexpr uint32_t CROSS_CORE_WAIT_FLAG_C0 = 0x9;
constexpr uint32_t CROSS_CORE_WAIT_FLAG_C1 = 0x8;
constexpr uint32_t VEC_DEAL_VECIDX_DIV = 2;

constexpr MicroAPI::CastTrait ctHalf2Fp32Zero = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

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
    GM_ADDR h_post;
    GM_ADDR gamma;
    GM_ADDR grad_x_post_optional;
    GM_ADDR grad_x;
    GM_ADDR grad_phi;
    GM_ADDR grad_alpha;
    GM_ADDR grad_bias;
    GM_ADDR grad_gamma;
    GM_ADDR workspace;
    TPipe *tPipeIn;
    const MhcPreBackwardTilingData *tilingData;
};

struct OpRunInfo {
    uint64_t totalLength_ = 0; // 总长度 (batch * sequence 或 T)
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

using aT_C0 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using bT_C0 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cT_C0 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MT_C0 = matmul::MatmulImpl<aT_C0, bT_C0, cT_C0>;

using aT_C1 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using bT_C1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cT_C1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MT_C1 = matmul::MatmulImpl<aT_C1, bT_C1, cT_C1>;

/**
 * @brief Workspace buffer管理结构体
 * 内存布局：
 * 1. h_mix_grad: [B, S, 2N + N*N] = [totalLength * fusionSize]
 * 2. alpha_grad: [ALPHA_GRAD_PADDING, vecCoreNum] = [24 * vecCoreNum]
 * 3. bias_grad: [vecCoreNum, 2N + N*N] = [vecCoreNum * fusionSize]
 * 4. inv_rms_grad: [B, S] = [totalLength]
 * 5. x_rs_grad_mm: [cubeCoreNum * DOUBLE_BUFFER, SINGLE_M * ND_BLOCK_SIZE]
 * 6. x_rs: [cubeCoreNum * DOUBLE_BUFFER, SINGLE_M * ND_BLOCK_SIZE]
 */
template <class P>
struct WorkspaceBuffer {
    uint64_t totalLength; // B * S
    uint64_t fusionSize;  // 2N + N*N
    uint64_t coreNum;     // vecCoreNum (AIV核数)

    // 各区域的起始偏移（以元素为单位）
    uint64_t hMixGradOffset;
    uint64_t alphaGradOffset;
    uint64_t biasGradOffset;
    uint64_t invRmsGradOffset;
    uint64_t xRsGradOffset;
    uint64_t xRsOffset; // V2输出，C1输入
    __aicore__ inline void Init(uint64_t bs, uint64_t fs, uint64_t vecCoreNum, uint64_t cubeCoreNum)
    {
        totalLength = bs;
        fusionSize = fs;
        coreNum = vecCoreNum;

        uint64_t alphaGradSize = ALPHA_GRAD_PADDING * vecCoreNum;
        uint64_t biasGradRows = vecCoreNum;

        // 计算各区域偏移
        hMixGradOffset = 0;
        alphaGradOffset = totalLength * fusionSize;
        biasGradOffset = alphaGradOffset + alphaGradSize;
        invRmsGradOffset = biasGradOffset + biasGradRows * fusionSize;
        xRsGradOffset = invRmsGradOffset + totalLength;
        xRsOffset = xRsGradOffset + SINGLE_M * ND_BLOCK_SIZE * DOUBLE_BUFFER * cubeCoreNum;
    }

    /**
     * @brief 获取h_mix_grad在指定bs索引位置的偏移
     * @param bsIndex batch*sequence的线性索引 (bsIndex = b * S + s)
     * @return uint64_t 偏移值（以元素为单位）
     */
    __aicore__ inline uint64_t GetHMixGradOffset(uint64_t bsIndex)
    {
        return hMixGradOffset + bsIndex * fusionSize;
    }

    /**
     * @brief 获取alpha_grad在指定(alphaIdx, coreId)位置的偏移
     * @param alphaIdx alpha索引 [0, 2]，对应3个alpha值
     * @param coreId 核心ID索引 [0, coreNum-1]
     * @return uint64_t 偏移值（以元素为单位）
     */
    __aicore__ inline uint64_t GetAlphaGradOffset(uint32_t coreId)
    {
        return alphaGradOffset + coreId * ALPHA_GRAD_PADDING;
    }

    /**
     * @brief 获取bias_grad在指定coreId的偏移
     * @param coreId 核心ID索引 [0, coreNum-1]
     * @return uint64_t 偏移值（以元素为单位）
     */
    __aicore__ inline uint64_t GetBiasGradOffset(uint32_t coreId)
    {
        return biasGradOffset + coreId * fusionSize;
    }

    /**
     * @brief 获取inv_rms_grad在指定bs索引位置的偏移
     * @param bsIndex batch*sequence的线性索引 (bsIndex = b * S + s)
     * @return uint64_t 偏移值（以元素为单位）
     */
    __aicore__ inline uint64_t GetInvRmsGradOffset(uint64_t bsIndex)
    {
        return invRmsGradOffset + bsIndex;
    }

    __aicore__ inline uint64_t GetXRsGradOffset(uint32_t coreId, uint32_t buffId)
    {
        return CeilAlign(xRsGradOffset, uint64_t(CEIL_ALIGN_DEFAULT)) +
               (coreId * DOUBLE_BUFFER + buffId) * SINGLE_M * ND_BLOCK_SIZE;
    }

    __aicore__ inline uint64_t GetXRsOffset(uint32_t coreId, uint32_t buffId)
    {
        return CeilAlign(xRsOffset, uint64_t(CEIL_ALIGN_DEFAULT)) +
               (coreId * DOUBLE_BUFFER + buffId) * SINGLE_M * ND_BLOCK_SIZE;
    }
};
} // namespace MhcPreBackwardUtils

#endif // ASCENDC_MHC_PRE_BACKWARD_UTILS_H
