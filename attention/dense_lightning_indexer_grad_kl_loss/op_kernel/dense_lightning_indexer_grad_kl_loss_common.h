/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_grad_kl_loss_common.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_COMMON_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig SFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

// V0 -> C1 P
constexpr uint8_t SYNC_V0_TO_C1_P_FLAG[2][2] = {{0, 1}, {2, 3}};
// V0 -> C1 SY
constexpr uint8_t SYNC_V0_TO_C1_SY_FLAG[2][2] = {{4, 5}, {6, 7}};
// C1 -> V1 P
constexpr uint8_t SYNC_C1_TO_V1_P_FLAG[2] = {8, 9};
// C1 -> V1 SY
constexpr uint8_t SYNC_C1_TO_V1_SY_FLAG[2] = {10, 11};
constexpr uint8_t SYNC_V1_TO_C2_DW_FLAG[2] = {12, 13};
constexpr uint8_t SYNC_C2_TO_V2_DW_FLAG[2] = {4, 5};
static constexpr uint8_t SYNC_MODE = 2;

static constexpr uint32_t N_WORKSPACE_SIZE = 1024; // n方向切分
static constexpr uint32_t ROPE_D_SIZE = 64;
static constexpr uint32_t S1_BASE_STEP = 128;
static constexpr uint32_t S2_BASE_STEP = 1024;
static constexpr uint32_t CUBE_BASE_BLOCK = 128;
static constexpr uint32_t C0_SIZE = 16;
static constexpr uint32_t VEC_ALIGN_SIZE = 32;
static constexpr uint32_t CUBE_MATRIX_SIZE = 256;
static constexpr uint32_t S1_VEC_SIZE_8 = 8;
static constexpr uint32_t DOUBLE_BUFFER = 2;
static constexpr uint32_t AIC_AIV_RATIO = 2;
static constexpr uint32_t FLOAT_DATA_BLOCK_NUM = 8;
static constexpr uint32_t FLOAT_REPEAT_NUM = 64;
static constexpr uint32_t S2_BASE_STEP_MASK_V3V4 = 128;
static constexpr uint32_t TEMP_VEC_SIZE_V3V4 = 24 * 1024;

// deter
static constexpr int64_t DETER_INVALID_RUNINFO_VALUE = -1;

enum class DLILayout {
    BSND = 0,
    TND = 1
};

enum class DLISparseMode {
    RightDown = 3
};

struct MMParam {
    uint32_t singleM = CUBE_BASE_BLOCK;
    uint32_t singleN = CUBE_BASE_BLOCK;
    uint32_t singleK = CUBE_BASE_BLOCK;
    uint32_t dstFixOffset = 0;
    uint32_t dstFixStride = 0;
    bool isLeftTranspose = false;
    bool isRightTranspose = false;
    bool needCopyRight = true; // 本次是否需要使用L0B
    bool isRightReuse = false; // 所有轮次是否涉及复用
    bool isL0CInit = true;     // 本次是否需要累加
    bool needAccumL0C = false; // 所有轮次是否涉及累加
    bool isOutKFisrt = true;
    bool isFixOut = false;
    bool isFixRelu = false;
    bool atomicFlag = true;
};

/** @name 模版类型定义
 *  @{
 */
template <typename InputQT, typename InputKT, typename InputWT, typename OutT, DLILayout LayoutQT = DLILayout::TND,
          DLILayout LayoutKT = DLILayout::TND, DLISparseMode SparseMode = DLISparseMode::RightDown,
          const bool HasRope = false, const bool Deterministic = false, typename... Args>
struct DLIType {
    using inputQT = InputQT;
    using inputKT = InputKT;
    using inputWT = InputWT;
    using outputT = OutT;
    static constexpr DLILayout inputQLayout = LayoutQT;
    static constexpr DLILayout inputKLayout = LayoutKT;
    static constexpr bool hasRope = HasRope;
    static constexpr bool deterministic = Deterministic;
};
/// @}

/** @name 核内数据结构定义
 *  @{
 */
struct DLIGradKLLossConstInfo {
    static constexpr uint32_t BUFFER_SIZE_BYTE_64 = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_128 = 128;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256 = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512 = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_9K = 9 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_11K = 11 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 64 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_192K = 192 * 1024;
    /** \brief 核信息 */
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint32_t subBlockIdx;
    uint32_t aivNum;
    uint32_t aicNum;

    /** \brief TilingData中的信息 */
    uint32_t bSize;
    uint32_t n1Size;              // 128/64/32
    uint32_t n2Size;              // 目前与n1Size相同
    uint32_t n1IndexSize;         // 64/32
    uint32_t n2IndexSize;         // 目前是1
    uint32_t gSizeQuery;          // 1
    uint32_t gSizeQueryIndex;     // 64/32
    uint32_t s1Size;              // 支持泛化
    uint32_t s2Size;              // 支持泛化
    uint32_t dSizeQuery;          // 默认不带Rope，固定等于128
    uint32_t dSizeQueryIndex;     // 默认不带Rope，固定等于128
    uint32_t dSizeQueryRope = 64; // Rope，固定等于64
    uint32_t kSize;               // 现阶段只支持2048
    uint32_t t1Size;
    DLISparseMode sparseMode; // 0或者3
    float scaleValue;
    uint32_t dSizeActual;
    uint32_t dLoopTimesCube;

    // 轴的乘积提取一些公共信息，减少scalar
    int32_t s2BaseSize;
    int32_t dSizeRope;
    int32_t gatherKeyDbNum;
    int32_t gatherKeyIndexDbNum;
    int32_t gatherKeySize;
    int32_t gatherKeyIndexSize;
    int32_t sparseBlockSize;
    int32_t s2BaseStep = S2_BASE_STEP;
    int32_t s2BaseBlk = CUBE_BASE_BLOCK;
    uint32_t softmaxHeadOffset = 0;

    uint32_t dKeySingleCoreSize = 0;
    uint32_t dKeyGmOffset = 0;

    int64_t dqPadOffset = 0;
    int64_t dqPadSize = 0;
    int64_t dkPadOffset = 0;
    int64_t dkPadSize = 0;
    int64_t dwPadOffset = 0;
    int64_t dwPadSize = 0;

    // 确定性参数
    uint32_t dKeyDeterGmOffset = 0;
    uint32_t dKeyDeterGmLength = 0;
    int64_t maxLoopSize = 0;
};

struct DLIGradKLLossRunInfo {
    uint32_t taskId;
    uint32_t taskIdMod2;
    uint32_t bIdx;
    uint32_t s1Idx;
    uint32_t kIdx;
    int64_t accumS1Idx; // 当前循环累加的T1
    int64_t accumS2Idx; // 当前循环累加的T2
    uint32_t actS1Size; // 当前batch的S1Size
    uint32_t actS2Size; // 当前batch的S2Size
    uint32_t kBaseSize; // k方向切分的基本块大小，在P和KLLoss阶段
    uint32_t kRealSize; // k切分之后的实际大小，在P和KLLoss阶段
    uint32_t kRealSizeAlign8;
    uint32_t kLoopTimes; // k方向的循环次数，在P和KLLoss阶段

    uint32_t curS1Size; // 当前轮次计算的S1Size
    uint32_t curS1SizeAlign16;
    uint32_t curS1SizeVec;  // 两个subBlock切分S1，当前aiv计算的S1大小
    uint32_t curS2StepSize; // 当前轮次计算的S2Size
    uint32_t curS2StepSizeAlign8;
    uint32_t curS2LoopTimes;
    uint32_t nBaseSizeP;
    uint32_t nRealSizeP;
    uint32_t nBaseSizeSY;
    uint32_t nRealSizeSY;
    uint32_t nIdxP;
    uint32_t nIdxSY;
    int32_t sparseMaskStartIdx; // 可能会出现负数

    // 存放一些offset，减少重复计算
    int32_t s2SparseLen;
    int32_t s2RealSize;
    int32_t s2LoopTimes;
    int32_t s2TailSize;
    int32_t s2Idx;

    // query相关gm的偏移量
    int64_t queryTensorOffset = 0;
    int64_t queryRopeTensorOffset = 0;
    int64_t queryIndexTensorOffset = 0;

    // key相关gm的偏移量
    int64_t keyTensorOffset = 0;
    int64_t keyRopeTensorOffset = 0;
    int64_t keyIndexTensorOffset = 0;

    // aiv相关gm的偏移量
    int64_t weightTensorOffset = 0;
    int64_t softmaxTensorOffset = 0;
    int64_t softmaxIndexTensorOffset = 0;

    uint32_t curS1InnerSizeV1V2;
    uint32_t pingPongFlagV1V2;
    uint32_t s1InnerIdxV1V2;

    bool lastS2 = false;
    bool isValid = false;
};
/// @}

// ================================Util functions==================================
template <typename T>
__aicore__ inline T SFAAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a < b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T>
__aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

template <typename T>
__aicore__ inline T AlignTo(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_COMMON_H
