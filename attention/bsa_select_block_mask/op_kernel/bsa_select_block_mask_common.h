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
 * \file bsa_select_block_mask_common.h
 * \brief
 */
#ifndef BSA_SELECT_BLOCK_MASK_COMMON_H
#define BSA_SELECT_BLOCK_MASK_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

constexpr uint8_t SYNC_V1_TO_C1_FLAG[2] = {0, 1};
constexpr uint8_t SYNC_C1_TO_V1_FLAG[9] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
constexpr uint32_t SYNC_C1_TO_V1_FLAG_NUMS = 9;
constexpr uint32_t CV_EXEC_RATIO = 4;
constexpr uint8_t SYNC_MODE = 2;
constexpr uint32_t BSA_FIXPIPE_SYNC_ANCHOR_M = 16;
constexpr uint32_t BSA_FIXPIPE_SYNC_ANCHOR_N = 16;
constexpr uint32_t BSA_FIXPIPE_SYNC_ANCHOR_ELEMS = BSA_FIXPIPE_SYNC_ANCHOR_M * BSA_FIXPIPE_SYNC_ANCHOR_N;

static constexpr uint32_t N_WORKSPACE_SIZE = 1024;
static constexpr uint32_t S1_BASE_STEP = 128;
static constexpr uint32_t S2_BASE_STEP = 1024;
static constexpr uint32_t CUBE_BASE_BLOCK = 128;
static constexpr uint32_t C0_SIZE = 16;
static constexpr uint32_t VEC_ALIGN_SIZE = 32;
static constexpr uint32_t CUBE_MATRIX_SIZE = 256;
static constexpr uint32_t AIC_AIV_RATIO = 2;
static constexpr uint32_t FLOAT_DATA_BLOCK_NUM = 8;
static constexpr uint32_t FP16_DATA_BLOCK_NUM = 16;
static constexpr uint32_t FLOAT_REPEAT_NUM = 64;
static constexpr uint32_t D_SIZE = 128;
static constexpr uint32_t Q_CHUNK_SIZE = 128;
static constexpr uint32_t K_CHUNK_SIZE = 128;
static constexpr uint32_t REPEAT_SIZE = 256;
static constexpr uint32_t BROC_BASE_NUM = 8;
static constexpr uint32_t BYTE_SIZE = 8;
static constexpr uint16_t XOR_OP_VALUE_16 = 0x8000;
static constexpr uint32_t HALF_SAFE_MAX = 2048;
static constexpr uint32_t RADIX_BITS_PER_ROUND = 2;
static constexpr uint32_t RADIX_NUM_BINS = 4;
static constexpr uint32_t RADIX_ROUNDS_FP16 = 8;
static constexpr float SOFTMAX_NEG_INF = -1e38f;
static constexpr uint32_t MAX_TILE_NUM_IN_UB = 6144;
static constexpr uint32_t MAX_TILE_NUM_IN_UB_BY2 = 3072;
static constexpr uint32_t MAX_TILE_NUM_IN_UB_BY3 = 2048;
static constexpr uint32_t FLOAT32_SAFE_INT = 16777216;
static constexpr uint32_t BLOCK_SIZE = 196608;
static constexpr uint32_t UB_BLOCK_SIZE = 32;

enum class BSALayout {
    TND = 0,
    BNSD = 1
};

enum class BSASparseMode {
    TopK = 0,
    BottomK = 1
};

template <typename InputT, typename OutputT, BSALayout LayoutQ = BSALayout::BNSD, BSALayout LayoutKV = BSALayout::BNSD,
          typename... Args>
struct BSAType {
    using inputT = InputT;
    using outputT = OutputT;
    static constexpr BSALayout layoutQ = LayoutQ;
    static constexpr BSALayout layoutKV = LayoutKV;
};

struct BSAConstInfo {
    static constexpr uint32_t BUFFER_SIZE_BYTE_64 = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_128 = 128;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256 = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512 = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_60K = 60 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 64 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_120K = 120 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_192K = 192 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_24K = 24 * 1024;

    uint32_t aicIdx;
    uint32_t aivIdx;
    uint32_t subBlockIdx;
    uint32_t aivNum;
    uint32_t aicNum;
    uint32_t activeYVecCoreNum;

    uint32_t batchSize;
    uint32_t numHeads;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t dSize;
    uint64_t blockShapeX;
    uint64_t blockShapeY;
    uint32_t xBlocks;
    uint32_t yBlocks;
    float scaleValue;
    float sparsity;
    uint64_t topKValue;
    BSASparseMode sparsityMode;

    uint32_t coreNum;
    uint32_t activeCoreNum;
    uint32_t rowsPerCore;
    uint32_t extraCores;
    uint32_t totalRows;
    uint32_t yBlocksPerCore;
    uint32_t extraYCores;
    uint32_t qChunkSize;
    uint32_t kChunkSize;

    uint64_t qCmpOffset;
    uint64_t kCmpOffset;
    uint64_t attnScoreOffset;
    uint64_t softmaxTmpOffset;
    uint64_t topkWorkspaceOffset;
    uint64_t pooledScoreOffset;

    uint8_t usePostBlockShape;
    uint16_t postBlockShapeX;
    uint16_t postBlockShapeY;
    uint32_t postXBlocks;
    uint32_t postYBlocks;
};

struct BSARunInfo {
    uint32_t taskId;
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t qBlockStart;
    uint32_t qBlockEnd;
    uint32_t curQBlocks;
    bool isValid;
};

template <typename T>
__aicore__ inline T BSACeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 BSAMax(T1 a, T2 b)
{
    return (a < b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 BSAMin(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T>
__aicore__ inline T BSAAlignTo(T n, T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

#endif // BSA_SELECT_BLOCK_MASK_COMMON_H
