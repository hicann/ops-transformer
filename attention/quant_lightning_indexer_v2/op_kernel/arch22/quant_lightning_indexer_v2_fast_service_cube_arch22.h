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
 * \file quant_lightning_indexer_v2_fast_service_cube_arch22.h
 * \brief Fast template cube service and shared layout constants on arch22 (int8, TND query).
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_CUBE_ARCH22_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_CUBE_ARCH22_H

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "quant_lightning_indexer_v2_common_arch22.h"
#include "quant_lightning_indexer_v2_vector.h"

namespace QLIV2Fast {
using namespace AscendC;
using namespace QLIV2Common;

static constexpr int D_SIZE = 128;
static constexpr int Q_ROWS = 384;
static constexpr int K_TILE_M = 128;
static constexpr int QK_CHUNK_K = 32;
static constexpr int QK_CHUNKS = 12;
static constexpr int QK_GROUP_CHUNKS = 4;
static constexpr int QK_GROUP_K = 128;
static constexpr int SCORE_ROWS = 512;
static constexpr int SCORE_STORAGE_COLS = 16;
static constexpr int SCORE_CHANNEL_COLS = 8;
static constexpr int TOP_K = 2048;
// TopK 状态队列按 value/index 交错存放，每个 topk 元素占 2 个 float 字
static constexpr int SCORE_INDEX_PAIR_WORDS = 2;

// fast 模板只支持 gSize 24/32 与 sparse_mode 3，取值与 host tiling 的
// GSIZE_FAST_24 / GSIZE_FAST_32 / SPARSE_MODE_LOWER 保持一致
static constexpr uint32_t GSIZE_FAST_24 = 24;
static constexpr uint32_t GSIZE_FAST_32 = 32;
static constexpr uint32_t SPARSE_MODE_LOWER = 3;
// gSize=24 的 weight 行 stride（元素个数）：24 个 head 经 Gather 补齐到 40
static constexpr int WEIGHT_ROW_STRIDE_N24 = 40;

static constexpr int SCORE_RING_DEPTH = 56;
static constexpr int SCORE_SYNC_PERIOD = 4;
static constexpr int SCORE_TILE_ELEMENTS = SCORE_ROWS * SCORE_STORAGE_COLS;
static constexpr int SCORE_RING_BYTES = SCORE_RING_DEPTH * SCORE_TILE_ELEMENTS * sizeof(float);

struct HardwareConstants {
    static constexpr int KILOBYTE_BYTES = 1024;
    static constexpr int VECTOR_BLOCK_BYTES = 32;
    static constexpr int VECTOR_BLOCKS_PER_REPEAT = QLIV2ServiceVec::B32_VEC_REPEAT_STRIDE;
    static constexpr int BITS_PER_BYTE = 8;
    static constexpr int CUBE_C0_ELEMENTS = 16;
    static constexpr int NZ_BLOCK_ELEMENTS = CUBE_C0_ELEMENTS * CUBE_C0_ELEMENTS;
    static constexpr int LOAD_DATA_BLOCK_BYTES = 512;
    static constexpr uint32_t QK_DEQUANT_SCALE_BITS = 0x3a800000U; // FP32 1/1024
    static constexpr uint32_t KEY_N2_INDEX = 0;                    // K 侧 head 索引，kHeadNum==1 时恒为 0
};

struct CrossCoreMessageId {
    static constexpr uint8_t MODE = ConstInfo::FIA_SYNC_MODE2; // 与慢模板 ConstInfo 相同的核间同步模式
    static constexpr int AIV_COUNT_PER_GROUP = 2;
    static constexpr int MAX_OUTSTANDING_MESSAGES = 14;
    static constexpr uint32_t SCORE_READY = 8;
    static constexpr uint32_t SCORE_FREE_OR_WEIGHT_READY = 9;
    static constexpr uint32_t VECTOR_DONE = 10;
    static constexpr uint32_t CUBE_RELEASE = 7;
};

struct KeyLoadContext {
    GlobalTensor<int8_t> key;
    GlobalTensor<int32_t> blockTable;
    uint64_t keyPrefix;
    uint64_t kCacheStride;
    uint32_t batch;
    uint32_t n2Index;
    uint32_t kHeadNum;
    uint32_t blockSize;
    uint32_t maxBlocksPerBatch;
    uint32_t actualS2;
    bool pageAttention;
};

struct VectorPostContext {
    GlobalTensor<half> keyScale;
    GlobalTensor<int32_t> blockTable;
    uint64_t scalePrefix;
    uint64_t kScaleStride;
    uint32_t batch;
    uint32_t n2Index;
    uint32_t kHeadNum;
    uint32_t blockSize;
    uint32_t maxBlocksPerBatch;
    uint32_t actualS1;
    uint32_t actualS2;
    uint32_t s1TileStart;
    uint32_t validSequences;
    uint32_t subblock;
    bool pageAttention;
    bool causal;
};

template <int N>
struct KernelConfig {
    static_assert(N == GSIZE_FAST_24 || N == GSIZE_FAST_32);
    static constexpr int QUERY_HEAD_NUM = N;
    static constexpr int S1_TILE = Q_ROWS / N;
    static constexpr int SEQUENCES_PER_AIV = S1_TILE / CrossCoreMessageId::AIV_COUNT_PER_GROUP;
    static constexpr int WEIGHT_ROW_STRIDE = N == GSIZE_FAST_24 ? WEIGHT_ROW_STRIDE_N24 : N;
    static constexpr int TOPK_WORKSPACE_BYTES_PER_AIV =
        SEQUENCES_PER_AIV * TOP_K * SCORE_INDEX_PAIR_WORDS * sizeof(float);
    static constexpr int GROUP_WORKSPACE_BYTES =
        SCORE_RING_BYTES + CrossCoreMessageId::AIV_COUNT_PER_GROUP * TOPK_WORKSPACE_BYTES_PER_AIV;
};

static_assert(KernelConfig<GSIZE_FAST_24>::S1_TILE == 16);
static_assert(KernelConfig<GSIZE_FAST_32>::S1_TILE == 12);
static_assert(KernelConfig<GSIZE_FAST_24>::SEQUENCES_PER_AIV == 8);
static_assert(KernelConfig<GSIZE_FAST_32>::SEQUENCES_PER_AIV == 6);
static_assert(SCORE_RING_DEPTH / SCORE_SYNC_PERIOD <= CrossCoreMessageId::MAX_OUTSTANDING_MESSAGES);

__aicore__ constexpr int KiB(int value)
{
    return value * HardwareConstants::KILOBYTE_BYTES;
}

__aicore__ constexpr event_t Event(int value)
{
    return static_cast<event_t>(value);
}

/* Mode 2 消息按 group 作用域生效，使用四位硬件计数器。 */
static __aicore__ inline void NotifyScoreReady()
{
    CrossCoreSetFlag<CrossCoreMessageId::MODE, PIPE_MTE3>(CrossCoreMessageId::SCORE_READY);
}

static __aicore__ inline void NotifyScoreFree()
{
    CrossCoreSetFlag<CrossCoreMessageId::MODE, PIPE_MTE2>(CrossCoreMessageId::SCORE_FREE_OR_WEIGHT_READY);
}

static __aicore__ inline void NotifyVectorDone()
{
    CrossCoreSetFlag<CrossCoreMessageId::MODE, PIPE_MTE3>(CrossCoreMessageId::VECTOR_DONE);
}

static __aicore__ inline void NotifyCubeRelease()
{
    CrossCoreSetFlag<CrossCoreMessageId::MODE, PIPE_MTE3>(CrossCoreMessageId::CUBE_RELEASE);
}

static __aicore__ inline void NotifyWeightReady()
{
    // 两个 AIV 各发一次；标志 9 需等 AIC 消费完这次一次性的 Weight 通知后，
    // 才复用为 Score FREE
    CrossCoreSetFlag<CrossCoreMessageId::MODE, PIPE_MTE3>(CrossCoreMessageId::SCORE_FREE_OR_WEIGHT_READY);
}

/* L1 布局：两块 96 KiB 的 QK tile、48 KiB Q、12 KiB W、四块 16 KiB K tile，
 * 以及一块紧凑的 score 暂存区。 */
static constexpr int QK_L1_CHUNK_BYTES = KiB(8);
static constexpr int QK_L1_TILE_BYTES = QK_CHUNKS * QK_L1_CHUNK_BYTES;
static constexpr int QK_L1_SLOT0 = 0;
static constexpr int Q_L1_BASE = QK_L1_TILE_BYTES;
static constexpr int Q_L1_CHUNK_BYTES = KiB(4);
static constexpr int Q_L1_BYTES = QK_CHUNKS * Q_L1_CHUNK_BYTES;
static constexpr int W_L1_BASE = Q_L1_BASE + Q_L1_BYTES;
static constexpr int W_L1_BYTES = KiB(12);
static constexpr int K_L1_SLOT_COUNT = 4;
static constexpr int K_L1_TILE_BYTES = KiB(16);
static constexpr int QK_L1_SLOT1 = KiB(288);
static constexpr int K_L1_BASE = QK_L1_SLOT1 - K_L1_SLOT_COUNT * K_L1_TILE_BYTES;
static constexpr int QK_L1_SLOT_STRIDE = QK_L1_SLOT1 - QK_L1_SLOT0;
static constexpr int OUTPUT_L1_BASE = QK_L1_SLOT1 + QK_L1_TILE_BYTES;
static constexpr int OUTPUT_L1_LOGICAL_BYTES = KiB(16);
static constexpr int OUTPUT_L1_GUARD_BYTES = 128;
static constexpr int OUTPUT_L1_BYTES = OUTPUT_L1_LOGICAL_BYTES + OUTPUT_L1_GUARD_BYTES;

static_assert(W_L1_BASE + W_L1_BYTES <= K_L1_BASE);
static_assert(K_L1_BASE + K_L1_SLOT_COUNT * K_L1_TILE_BYTES == QK_L1_SLOT1);
static_assert(QK_L1_SLOT1 + QK_L1_TILE_BYTES == OUTPUT_L1_BASE);

/* L0A：两块 K tile 与一组 QK group。 */
static constexpr int K_L0A_BASE = 0;
static constexpr int K_L0A_TILE_BYTES = KiB(16);
static constexpr int K_L0A_SLOT_COUNT = 2;
static constexpr int QK_L0A_BASE = K_L0A_SLOT_COUNT * K_L0A_TILE_BYTES;
static constexpr int QK_L0A_BYTES = KiB(32);

/* L0B：常驻的 Q^T 与 W^T。 */
static constexpr int Q_L0B_BASE = 0;
static constexpr int Q_L0B_CHUNK_BYTES = KiB(4);
static constexpr int Q_L0B_BYTES = KiB(48);
static constexpr int W_L0B_BASE = Q_L0B_BYTES;
static constexpr int W_L0B_GROUP_BYTES = KiB(4);
static constexpr int W_L0B_BYTES = KiB(12);

/* L0C：四块 INT32 QK chunk 与两块带保护行的 FP32 score 累加区。 */
static constexpr int QK_C_BYTES = KiB(16);
static constexpr int QK_C_SLOT_COUNT = 4;
static constexpr int OUTPUT_C_BASE = QK_C_SLOT_COUNT * QK_C_BYTES;
static constexpr int OUTPUT_C_SLOT_BYTES = KiB(17); // 16 KiB 逻辑区 + 1 KiB 保护行
static constexpr int OUTPUT_C_LOGICAL_BYTES = KiB(16);

struct CubePipelineConstants {
    static constexpr int PHYSICAL_KEY_SLOT_COUNT = K_L1_SLOT_COUNT;
    static constexpr int KEY_L0_SLOT_COUNT = K_L0A_SLOT_COUNT;
    static constexpr int QK_PHASE_COUNT = 2;
    static constexpr int QK_GROUP_COUNT = QK_CHUNKS / QK_GROUP_CHUNKS;
    static constexpr int QK_GROUP_0 = 0;
    static constexpr int QK_GROUP_1 = 1;
    static constexpr int QK_GROUP_2 = 2;
    static constexpr int QK_ACCUMULATOR_SLOT_COUNT = OUTPUT_C_BASE / QK_C_BYTES;
    static constexpr int SCORE_OUTPUT_TILE_COUNT = SCORE_ROWS / K_TILE_M;
    static constexpr int SCORE_CHANNEL_HALF_COUNT = 2;
    static constexpr int KEY_L0_EVENT_BASE = 2;
    static constexpr int SCORE_OUTPUT_EVENT_BASE = 4;
    static constexpr int QK_GROUP_EVENT = 4;
    static constexpr int FINAL_SCORE_EVENT = 6;
    static constexpr int SCORE_HALF_BYTES = OUTPUT_C_LOGICAL_BYTES / 2;
    static constexpr int INITIAL_KEY_TILE_COUNT = PHYSICAL_KEY_SLOT_COUNT;
    static constexpr int STEADY_STATE_TILES_PER_ROUND = PHYSICAL_KEY_SLOT_COUNT;
    static constexpr int PHASE_PING = 0;
    static constexpr int PHASE_PONG = 1;
    static constexpr int PHASE_TOGGLE = 1;
    static constexpr int KEY_L0_PING = 0;
    static constexpr int KEY_L0_PONG = 1;
    static constexpr int SCORE_SLOT_PING = 0;
    static constexpr int SCORE_SLOT_PONG = 1;
    static constexpr int SCORE_HALF_0 = 0;
    static constexpr int SCORE_HALF_1 = 1;
    static constexpr int SCORE_SEGMENT_0 = 0;
    static constexpr int SCORE_SEGMENT_1 = 1;
    static constexpr int SCORE_SEGMENT_2 = 2;
    static constexpr int SCORE_SEGMENT_3 = 3;
    static constexpr int PHYSICAL_KEY_SLOT_0 = 0;
    static constexpr int PHYSICAL_KEY_SLOT_1 = 1;
    static constexpr int PHYSICAL_KEY_SLOT_2 = 2;
    static constexpr int PHYSICAL_KEY_SLOT_3 = 3;
    // 初始装载阶段的逻辑 K tile 编号与首个 Q/W tile 编号
    static constexpr int KEY_TILE_0 = 0;
    static constexpr int KEY_TILE_1 = 1;
    static constexpr int KEY_TILE_2 = 2;
    static constexpr int KEY_TILE_3 = 3;
    // Q chunk 编号与 QK group 内 chunk 偏移编号（显式展开的模板实参，不使用递归展开）
    static constexpr int Q_CHUNK_0 = 0;
    static constexpr int Q_CHUNK_1 = 1;
    static constexpr int Q_CHUNK_2 = 2;
    static constexpr int Q_CHUNK_3 = 3;
    static constexpr int Q_CHUNK_4 = 4;
    static constexpr int Q_CHUNK_5 = 5;
    static constexpr int Q_CHUNK_6 = 6;
    static constexpr int Q_CHUNK_7 = 7;
    static constexpr int Q_CHUNK_8 = 8;
    static constexpr int Q_CHUNK_9 = 9;
    static constexpr int Q_CHUNK_10 = 10;
    static constexpr int Q_CHUNK_11 = 11;
    static constexpr int QK_GROUP_CHUNK_OFFSET_0 = 0;
    static constexpr int QK_GROUP_CHUNK_OFFSET_1 = 1;
    static constexpr int QK_GROUP_CHUNK_OFFSET_2 = 2;
    static constexpr int QK_GROUP_CHUNK_OFFSET_3 = 3;
    static constexpr int REFILL_AFTER_SEGMENT_0 = INITIAL_KEY_TILE_COUNT + 2;
    static constexpr int REFILL_AFTER_SEGMENT_1 = INITIAL_KEY_TILE_COUNT + 3;
    static constexpr int REFILL_AFTER_SEGMENT_2 = INITIAL_KEY_TILE_COUNT + 4;
    static constexpr int REFILL_AFTER_SEGMENT_3 = INITIAL_KEY_TILE_COUNT + 5;
};
static constexpr uint32_t Q_L1_EVENT = EVENT_ID4;
static constexpr uint32_t W_L1_EVENT = EVENT_ID5;

// {isSetFMatrix, isSetPadding}：按参数中的 l1H/l1W 设置 FMatrix；
// 跳过 padValue 配置（本路径整块搬运，不做 3D padding）
static constexpr IsResetLoad3dConfig LOAD_L1_TO_L0A_CONFIG = {true, false};

static __aicore__ inline uint32_t GetMixGroupIndex()
{
    if ASCEND_IS_AIC {
        return static_cast<uint32_t>(GetBlockIdx());
    }
    return static_cast<uint32_t>((GetBlockIdx() - GetSubBlockIdx()) / CrossCoreMessageId::AIV_COUNT_PER_GROUP);
}

static __aicore__ inline void InitCubeEvents()
{
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
    SetFlag<HardEvent::MTE1_MTE2>(Q_L1_EVENT);
    SetFlag<HardEvent::MTE1_MTE2>(W_L1_EVENT);

    SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
    asc_sync_notify(PIPE_MTE3, PIPE_FIX, EVENT_ID0);

    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);

    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    SetFlag<HardEvent::FIX_M>(EVENT_ID2);
    SetFlag<HardEvent::FIX_M>(EVENT_ID3);
    SetFlag<HardEvent::FIX_M>(EVENT_ID4);
    SetFlag<HardEvent::FIX_M>(EVENT_ID5);
}

static __aicore__ inline void FinalizeCubeEvents()
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::MTE1_MTE2>(Q_L1_EVENT);
    WaitFlag<HardEvent::MTE1_MTE2>(W_L1_EVENT);

    WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
    asc_sync_wait(PIPE_MTE3, PIPE_FIX, EVENT_ID0);

    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::M_MTE1>(Event(CubePipelineConstants::QK_GROUP_EVENT));

    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID4);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID5);
}

static __aicore__ inline void ZeroL1(const LocalTensor<half>& destination, uint32_t bytes)
{
    InitConstValueParams<half> fillParams;
    // 一次 repeat 覆盖整个区域：blockNum 以 32B 块计，块间无间隔
    fillParams.repeatTimes = 1;
    fillParams.blockNum = bytes / HardwareConstants::VECTOR_BLOCK_BYTES;
    fillParams.dstGap = 0;
    fillParams.initValue = static_cast<half>(0.0f);
    InitConstValue(destination, fillParams);
}

template <int N>
static __aicore__ inline void LoadQueryToL1(const GlobalTensor<int8_t>& queryGm, int s1Tile, int validS1)
{
    WaitFlag<HardEvent::MTE1_MTE2>(Q_L1_EVENT);
    LocalTensor<int8_t> destinationTensor(TPosition::A1, Q_L1_BASE, Q_L1_BYTES);
    const uint64_t sourceOffset = static_cast<uint64_t>(s1Tile) * Q_ROWS * D_SIZE;
    if (validS1 == KernelConfig<N>::S1_TILE) {
        // 整 tile 一次搬入：源端按 12 个 32×128 的 ND 矩阵连续读取（src 侧步长单位为元素），
        // 目的端按 NZ 分形排布（dstNz 侧步长单位为元素）
        Nd2NzParams copyParams;
        copyParams.ndNum = QK_CHUNKS;
        copyParams.nValue = QK_CHUNK_K;
        copyParams.dValue = D_SIZE;
        copyParams.srcNdMatrixStride = QK_CHUNK_K * D_SIZE;
        copyParams.srcDValue = D_SIZE;
        copyParams.dstNzC0Stride = QK_CHUNK_K;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = D_SIZE * QK_CHUNK_K;
        DataCopy(destinationTensor, queryGm[sourceOffset], copyParams);
    } else {
        const int validRows = validS1 * N;
        // 尾块：先整块清零，再逐 chunk 只搬有效行（源端行距 D_SIZE 个元素）
        for (int chunk = 0; chunk < QK_CHUNKS; ++chunk) {
            ZeroL1(destinationTensor[chunk * Q_L1_CHUNK_BYTES].template ReinterpretCast<half>(), Q_L1_CHUNK_BYTES);
        }
        PipeBarrier<PIPE_MTE2>();
        for (int chunk = 0; chunk < QK_CHUNKS; ++chunk) {
            const int rowOffset = chunk * QK_CHUNK_K;
            if (rowOffset >= validRows) {
                continue;
            }
            int rows = Min(validRows - rowOffset, QK_CHUNK_K);
            Nd2NzParams copyParams;
            copyParams.ndNum = 1;
            copyParams.nValue = static_cast<uint16_t>(rows);
            copyParams.dValue = D_SIZE;
            copyParams.srcNdMatrixStride = 0;
            copyParams.srcDValue = D_SIZE;
            copyParams.dstNzC0Stride = QK_CHUNK_K;
            copyParams.dstNzNStride = 1;
            copyParams.dstNzMatrixStride = 0;
            DataCopy(destinationTensor[chunk * Q_L1_CHUNK_BYTES], queryGm[sourceOffset + rowOffset * D_SIZE],
                     copyParams);
        }
    }
    SetFlag<HardEvent::MTE2_MTE1>(Q_L1_EVENT);
}

static __aicore__ inline void LoadWeightN24(const LocalTensor<half>& destination, const GlobalTensor<half>& source,
                                            uint64_t sourceOffset, int rows)
{
    constexpr int rowsPerPair = 2;
    constexpr int sourceRowElements = KernelConfig<GSIZE_FAST_24>::WEIGHT_ROW_STRIDE;
    constexpr int sourcePairElements = rowsPerPair * sourceRowElements;
    constexpr int firstSegmentElements = HardwareConstants::CUBE_C0_ELEMENTS;
    constexpr int middleSegmentElements = 2 * HardwareConstants::CUBE_C0_ELEMENTS;
    constexpr int pairDestinationStride = 3 * HardwareConstants::NZ_BLOCK_ELEMENTS + middleSegmentElements;
    constexpr int middleSegmentDestination = HardwareConstants::NZ_BLOCK_ELEMENTS;
    constexpr int lastSegmentDestination =
        2 * HardwareConstants::NZ_BLOCK_ELEMENTS + HardwareConstants::CUBE_C0_ELEMENTS;
    constexpr int middleSegmentSource = HardwareConstants::CUBE_C0_ELEMENTS;
    constexpr int lastSegmentSource = sourceRowElements + HardwareConstants::CUBE_C0_ELEMENTS / 2;

    const int pairCount = rows / rowsPerPair;
    if (pairCount > 0) {
        // 成对行搬入 NZ：源行 stride 为 40 个 half（srcDValue/srcNdMatrixStride 单位为元素），
        // 目的端每对占 3 个 16×16 分形加中段；三次 DataCopy 分别填充每对行的前/中/尾段
        Nd2NzParams copyParams;
        copyParams.ndNum = static_cast<uint16_t>(pairCount);
        copyParams.nValue = 1;
        copyParams.dValue = firstSegmentElements;
        copyParams.srcNdMatrixStride = sourcePairElements;
        copyParams.srcDValue = sourceRowElements;
        copyParams.dstNzC0Stride = HardwareConstants::CUBE_C0_ELEMENTS;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = pairDestinationStride;
        DataCopy(destination, source[sourceOffset], copyParams);

        copyParams.nValue = rowsPerPair;
        copyParams.srcDValue = HardwareConstants::CUBE_C0_ELEMENTS;
        DataCopy(destination[middleSegmentDestination], source[sourceOffset + middleSegmentSource], copyParams);

        copyParams.nValue = 1;
        copyParams.srcDValue = sourceRowElements;
        DataCopy(destination[lastSegmentDestination], source[sourceOffset + lastSegmentSource], copyParams);
    }
    if ((rows & 1) != 0) {
        constexpr int headNum = KernelConfig<GSIZE_FAST_24>::QUERY_HEAD_NUM;
        const int row = rows - 1;
        const int destinationOffset =
            row * HardwareConstants::CUBE_C0_ELEMENTS +
            ((row * headNum) / HardwareConstants::CUBE_C0_ELEMENTS) * HardwareConstants::NZ_BLOCK_ELEMENTS;
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = 1;
        copyParams.dValue = middleSegmentElements;
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue = sourceRowElements;
        copyParams.dstNzC0Stride = HardwareConstants::CUBE_C0_ELEMENTS;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(destination[destinationOffset], source[sourceOffset + row * sourceRowElements], copyParams);
    }
}

static __aicore__ inline void LoadWeightN32(const LocalTensor<half>& destination, const GlobalTensor<half>& source,
                                            uint64_t sourceOffset, int rows)
{
    constexpr int rowsPerHalf = KernelConfig<GSIZE_FAST_32>::S1_TILE / 2;
    constexpr int sourceRowElements = KernelConfig<GSIZE_FAST_32>::WEIGHT_ROW_STRIDE;
    constexpr int destinationRowStride = 2 * HardwareConstants::NZ_BLOCK_ELEMENTS + HardwareConstants::CUBE_C0_ELEMENTS;
    constexpr int secondHalfDestination = rowsPerHalf * destinationRowStride + sourceRowElements;

    const int first = Min(rows, rowsPerHalf);
    if (first > 0) {
        // N32 权重整行搬入 NZ：源行 stride 32 个 half（src 侧步长单位为元素），
        // 目的端每行占 2 个 16×16 分形加 1 个 C0 段；上下半 tile 分两次搬
        Nd2NzParams copyParams;
        copyParams.ndNum = static_cast<uint16_t>(first);
        copyParams.nValue = 1;
        copyParams.dValue = sourceRowElements;
        copyParams.srcNdMatrixStride = sourceRowElements;
        copyParams.srcDValue = sourceRowElements;
        copyParams.dstNzC0Stride = HardwareConstants::CUBE_C0_ELEMENTS;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = destinationRowStride;
        DataCopy(destination, source[sourceOffset], copyParams);
    }
    if (rows > rowsPerHalf) {
        Nd2NzParams copyParams;
        copyParams.ndNum = static_cast<uint16_t>(rows - rowsPerHalf);
        copyParams.nValue = 1;
        copyParams.dValue = sourceRowElements;
        copyParams.srcNdMatrixStride = sourceRowElements;
        copyParams.srcDValue = sourceRowElements;
        copyParams.dstNzC0Stride = HardwareConstants::CUBE_C0_ELEMENTS;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = destinationRowStride;
        DataCopy(destination[secondHalfDestination], source[sourceOffset + rowsPerHalf * sourceRowElements],
                 copyParams);
    }
}

template <int N>
static __aicore__ inline void LoadWeightToL1(const GlobalTensor<half>& weightGm, int s1Tile, int validS1,
                                             bool initialize)
{
    WaitFlag<HardEvent::MTE1_MTE2>(W_L1_EVENT);
    LocalTensor<half> destination(TPosition::B1, W_L1_BASE, W_L1_BYTES / sizeof(half));
    if (initialize || (validS1 != KernelConfig<N>::S1_TILE)) {
        ZeroL1(destination, W_L1_BYTES);
        PipeBarrier<PIPE_MTE2>();
    }
    const uint64_t sourceOffset =
        static_cast<uint64_t>(s1Tile) * KernelConfig<N>::S1_TILE * KernelConfig<N>::WEIGHT_ROW_STRIDE;
    if constexpr (N == GSIZE_FAST_24) {
        LoadWeightN24(destination, weightGm, sourceOffset, validS1);
    } else {
        LoadWeightN32(destination, weightGm, sourceOffset, validS1);
    }
    SetFlag<HardEvent::MTE2_MTE1>(W_L1_EVENT);
}

// 计算 [logicalRow, logicalRow + rows) 在 GM 上的起始偏移，rows 会截断到 PA 块边界
// key 加载与 kScale 加载共用（rowUnitElements 分别为 D_SIZE 和 1）
static __aicore__ inline uint64_t GetRowGmOffset(const GlobalTensor<int32_t>& blockTable, uint32_t batch,
                                                 uint32_t maxBlocksPerBatch, uint32_t blockSize, uint32_t kHeadNum,
                                                 uint32_t n2Index, uint64_t prefix, uint64_t blockStride,
                                                 uint32_t rowUnitElements, bool pageAttention, uint32_t logicalRow,
                                                 int& rows)
{
    if (pageAttention) {
        // 前置条件：blockSize 已由 host tiling 校验非零（PA 布局的块大小语义）
        const uint32_t logicalBlock = logicalRow / blockSize;
        const uint32_t rowInBlock = logicalRow % blockSize;
        const uint32_t physicalBlock =
            static_cast<uint32_t>(blockTable.GetValue(batch * maxBlocksPerBatch + logicalBlock));
        const uint32_t rowsInBlock = blockSize - rowInBlock;
        if (rows > static_cast<int>(rowsInBlock)) {
            rows = static_cast<int>(rowsInBlock);
        }
        return static_cast<uint64_t>(physicalBlock) * blockStride +
               (static_cast<uint64_t>(rowInBlock) * kHeadNum + n2Index) * rowUnitElements;
    }
    return ((prefix + logicalRow) * kHeadNum + n2Index) * rowUnitElements;
}

template <int PhysicalSlot>
static __aicore__ inline void LoadKeyToL1(const KeyLoadContext& context, int tile)
{
    static_assert(PhysicalSlot >= 0 && PhysicalSlot < CubePipelineConstants::PHYSICAL_KEY_SLOT_COUNT);
    WaitFlag<HardEvent::MTE1_MTE2>(Event(PhysicalSlot));
    LocalTensor<int8_t> destinationTensor(TPosition::A1, K_L1_BASE + PhysicalSlot * K_L1_TILE_BYTES, K_L1_TILE_BYTES);
    const int logicalStart = tile * K_TILE_M;
    int validRows = Max(0, Min(static_cast<int>(context.actualS2) - logicalStart, K_TILE_M));
    if (validRows < K_TILE_M) {
        ZeroL1(destinationTensor.template ReinterpretCast<half>(), K_L1_TILE_BYTES);
        if (validRows > 0) {
            PipeBarrier<PIPE_MTE2>();
        }
    }
    int copied = 0;
    while (copied < validRows) {
        const int logicalRow = logicalStart + copied;
        int rows = validRows - copied;
        const uint64_t sourceOffset = GetRowGmOffset(
            context.blockTable, context.batch, context.maxBlocksPerBatch, context.blockSize, context.kHeadNum,
            context.n2Index, context.keyPrefix, context.kCacheStride, D_SIZE, context.pageAttention, logicalRow, rows);
        // 逐行搬 K：srcDValue 为源端相邻行间隔（kHeadNum×D 个元素，跳过其余 head 的行），
        // 目的端 NZ 行步长为 K_TILE_M（元素单位）
        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = static_cast<uint16_t>(rows);
        nd2nz.dValue = D_SIZE;
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.srcDValue = context.kHeadNum * D_SIZE;
        nd2nz.dstNzC0Stride = K_TILE_M;
        nd2nz.dstNzNStride = 1;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(destinationTensor[copied * QK_CHUNK_K], context.key[sourceOffset], nd2nz);
        copied += rows;
    }
    SetFlag<HardEvent::MTE2_MTE1>(Event(PhysicalSlot));
}

template <int Chunk>
static __aicore__ inline void CopyQChunkToL0b()
{
    static_assert(Chunk >= 0 && Chunk < QK_CHUNKS);
    LocalTensor<int8_t> destination(TPosition::B2, Q_L0B_BASE + Chunk * Q_L0B_CHUNK_BYTES, Q_L0B_CHUNK_BYTES);
    LocalTensor<int8_t> source(TPosition::B1, Q_L1_BASE + Chunk * Q_L1_CHUNK_BYTES, Q_L1_CHUNK_BYTES);
    // L1 到 L0B 连续搬运：起始索引 0，repeat 间源步长 1 个 32B 块，无目的间隔，不转置
    LoadData2DParams copyParams;
    copyParams.startIndex = 0;
    copyParams.repeatTimes = Q_L0B_CHUNK_BYTES / HardwareConstants::LOAD_DATA_BLOCK_BYTES;
    copyParams.srcStride = 1;
    copyParams.dstGap = 0;
    copyParams.ifTranspose = false;
    LoadData(destination, source, copyParams);
}

static __aicore__ inline void ActivateQueryWeight()
{
    WaitFlag<HardEvent::MTE2_MTE1>(Q_L1_EVENT);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_0>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_1>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_2>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_3>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_4>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_5>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_6>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_7>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_8>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_9>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_10>();
    CopyQChunkToL0b<CubePipelineConstants::Q_CHUNK_11>();
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(Q_L1_EVENT);

    WaitFlag<HardEvent::MTE2_MTE1>(W_L1_EVENT);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    LocalTensor<half> weightL0b(TPosition::B2, W_L0B_BASE, W_L0B_BYTES / sizeof(half));
    LocalTensor<half> weightL1(TPosition::B1, W_L1_BASE, W_L1_BYTES / sizeof(half));
    // L1 到 L0B 连续搬运：起始索引 0，repeat 间源步长 1 个 32B 块，无目的间隔，不转置
    LoadData2DParams copyParams;
    copyParams.startIndex = 0;
    copyParams.repeatTimes = W_L0B_BYTES / HardwareConstants::LOAD_DATA_BLOCK_BYTES;
    copyParams.srcStride = 1;
    copyParams.dstGap = 0;
    copyParams.ifTranspose = false;
    LoadData(weightL0b, weightL1, copyParams);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(W_L1_EVENT);
}

static __aicore__ inline void ReleaseQueryWeight()
{
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
}

template <int PhysicalSlot, int L0ASlot>
static __aicore__ inline void PrefetchKeyToL0A()
{
    static_assert(PhysicalSlot >= 0 && PhysicalSlot < CubePipelineConstants::PHYSICAL_KEY_SLOT_COUNT);
    static_assert(L0ASlot >= 0 && L0ASlot < CubePipelineConstants::KEY_L0_SLOT_COUNT);
    WaitFlag<HardEvent::MTE2_MTE1>(Event(PhysicalSlot));
    WaitFlag<HardEvent::M_MTE1>(Event(L0ASlot + CubePipelineConstants::KEY_L0_EVENT_BASE));
    LocalTensor<int8_t> destination(TPosition::A2, K_L0A_BASE + L0ASlot * K_L0A_TILE_BYTES, K_L0A_TILE_BYTES);
    LocalTensor<int8_t> source(TPosition::A1, K_L1_BASE + PhysicalSlot * K_L1_TILE_BYTES, K_L1_TILE_BYTES);
    // 整块搬 K tile：L1 侧 1×128（l1H/l1W 单位元素），L0A 侧 m/k 扩展为 128×128，
    // 其余为 1×1 卷积常规配置，不转置
    LoadData3DParamsV2<int8_t> copyParams;
    copyParams.l1H = 1;
    copyParams.l1W = D_SIZE;
    copyParams.channelSize = D_SIZE;
    copyParams.kExtension = D_SIZE;
    copyParams.mExtension = K_TILE_M;
    copyParams.kStartPt = 0;
    copyParams.mStartPt = 0;
    copyParams.strideW = 1;
    copyParams.strideH = 1;
    copyParams.filterW = 1;
    copyParams.filterH = 1;
    copyParams.dilationFilterW = 1;
    copyParams.dilationFilterH = 1;
    copyParams.enTranspose = false;
    LoadData<int8_t, LOAD_L1_TO_L0A_CONFIG>(destination, source, copyParams);
    SetFlag<HardEvent::MTE1_M>(Event(L0ASlot + CubePipelineConstants::KEY_L0_EVENT_BASE));
    SetFlag<HardEvent::MTE1_MTE2>(Event(PhysicalSlot));
}

template <int Phase, int Chunk>
static __aicore__ inline constexpr int QkCSlot()
{
    static_assert(Phase >= 0 && Phase < CubePipelineConstants::QK_PHASE_COUNT);
    static_assert(Chunk >= 0 && Chunk < QK_CHUNKS);
    return (Phase * QK_CHUNKS + Chunk) & (CubePipelineConstants::QK_ACCUMULATOR_SLOT_COUNT - 1);
}

template <int Phase, int Chunk, int KSlot>
static __aicore__ inline void ProduceQkChunk()
{
    static_assert(KSlot >= 0 && KSlot < CubePipelineConstants::KEY_L0_SLOT_COUNT);
    constexpr int cSlot = QkCSlot<Phase, Chunk>();
    WaitFlag<HardEvent::FIX_M>(Event(cSlot));
    LocalTensor<int32_t> qkL0c(TPosition::CO1, cSlot * QK_C_BYTES, QK_C_BYTES / sizeof(int32_t));
    LocalTensor<int8_t> keyL0a(TPosition::A2, K_L0A_BASE + KSlot * K_L0A_TILE_BYTES, K_L0A_TILE_BYTES);
    LocalTensor<int8_t> queryL0b(TPosition::B2, Q_L0B_BASE + Chunk * Q_L0B_CHUNK_BYTES, Q_L0B_CHUNK_BYTES);
    MmadParams mmParams;
    mmParams.m = K_TILE_M;
    mmParams.n = QK_CHUNK_K;
    mmParams.k = D_SIZE;
    mmParams.cmatrixInitVal = true;
    mmParams.cmatrixSource = false;
    Mmad(qkL0c, keyL0a, queryL0b, mmParams);
    SetFlag<HardEvent::M_FIX>(Event(cSlot));
    WaitFlag<HardEvent::M_FIX>(Event(cSlot));
    LocalTensor<half> qkL1(TPosition::A1, Phase * QK_L1_SLOT_STRIDE + Chunk * QK_L1_CHUNK_BYTES,
                           QK_L1_CHUNK_BYTES / sizeof(half));
    // L0C→L1 搬运：32(n)×128(m) 的 INT32 QK 结果经 DEQF16 反量化为 half 写入 L1 的 NZ 区；
    // dstStride/srcStride 与 mSize 相同，表示 m 轴相邻行间距下源目两侧均连续
    DataCopyCO12DstParams copyParams;
    copyParams.nSize = QK_CHUNK_K;
    copyParams.mSize = K_TILE_M;
    copyParams.dstStride = K_TILE_M;
    copyParams.srcStride = K_TILE_M;
    copyParams.quantPre = QuantMode_t::DEQF16;
    // reluPre=1 是 DEQF16 反量化路径的配套取值（慢模板同路径一致；NoQuant 路径取 0）
    copyParams.reluPre = 1;
    copyParams.channelSplit = false;
    copyParams.nz2ndEn = false;
    DataCopy(qkL1, qkL0c, copyParams);
    SetFlag<HardEvent::FIX_M>(Event(cSlot));
}

template <int Phase, int Group, int KSlot>
static __aicore__ inline void ProduceQkGroup()
{
    ProduceQkChunk<Phase, Group * QK_GROUP_CHUNKS + CubePipelineConstants::QK_GROUP_CHUNK_OFFSET_0, KSlot>();
    ProduceQkChunk<Phase, Group * QK_GROUP_CHUNKS + CubePipelineConstants::QK_GROUP_CHUNK_OFFSET_1, KSlot>();
    ProduceQkChunk<Phase, Group * QK_GROUP_CHUNKS + CubePipelineConstants::QK_GROUP_CHUNK_OFFSET_2, KSlot>();
    ProduceQkChunk<Phase, Group * QK_GROUP_CHUNKS + CubePipelineConstants::QK_GROUP_CHUNK_OFFSET_3, KSlot>();
}

template <int Phase, int KSlot>
static __aicore__ inline void ProduceQkTile()
{
    static_assert(Phase >= 0 && Phase < CubePipelineConstants::QK_PHASE_COUNT);
    static_assert(KSlot >= 0 && KSlot < CubePipelineConstants::KEY_L0_SLOT_COUNT);
    WaitFlag<HardEvent::MTE1_M>(Event(KSlot + CubePipelineConstants::KEY_L0_EVENT_BASE));
    WaitFlag<HardEvent::MTE1_FIX>(Event(Phase));
    ProduceQkGroup<Phase, CubePipelineConstants::QK_GROUP_0, KSlot>();
    ProduceQkGroup<Phase, CubePipelineConstants::QK_GROUP_1, KSlot>();
    ProduceQkGroup<Phase, CubePipelineConstants::QK_GROUP_2, KSlot>();
    SetFlag<HardEvent::M_MTE1>(Event(KSlot + CubePipelineConstants::KEY_L0_EVENT_BASE));
    SetFlag<HardEvent::FIX_MTE1>(Event(Phase));
}

template <int Phase, int Group>
static __aicore__ inline void PrefetchQkGroup()
{
    static_assert(Phase >= 0 && Phase < CubePipelineConstants::QK_PHASE_COUNT);
    static_assert(Group >= 0 && Group < CubePipelineConstants::QK_GROUP_COUNT);
    WaitFlag<HardEvent::M_MTE1>(Event(CubePipelineConstants::QK_GROUP_EVENT));
    LocalTensor<half> destination(TPosition::A2, QK_L0A_BASE, QK_L0A_BYTES / sizeof(half));
    LocalTensor<half> source(TPosition::A1, Phase * QK_L1_SLOT_STRIDE + Group * QK_GROUP_CHUNKS * QK_L1_CHUNK_BYTES,
                             QK_L0A_BYTES / sizeof(half));
    // 整块搬 QK group：L1 侧 1×128（l1H/l1W 单位元素），L0A 侧 m/k 扩展为 128×128，不转置
    LoadData3DParamsV2<half> copyParams;
    copyParams.l1H = 1;
    copyParams.l1W = QK_GROUP_K;
    copyParams.channelSize = QK_GROUP_K;
    copyParams.kExtension = QK_GROUP_K;
    copyParams.mExtension = K_TILE_M;
    copyParams.kStartPt = 0;
    copyParams.mStartPt = 0;
    copyParams.strideW = 1;
    copyParams.strideH = 1;
    copyParams.filterW = 1;
    copyParams.filterH = 1;
    copyParams.dilationFilterW = 1;
    copyParams.dilationFilterH = 1;
    copyParams.enTranspose = false;
    LoadData<half, LOAD_L1_TO_L0A_CONFIG>(destination, source, copyParams);
    SetFlag<HardEvent::MTE1_M>(Event(CubePipelineConstants::QK_GROUP_EVENT));
}

template <int Group, int OutputSlot, int OutputTile>
static __aicore__ inline void ConsumeQkGroup()
{
    static_assert(Group >= 0 && Group < CubePipelineConstants::QK_GROUP_COUNT);
    static_assert(OutputSlot >= 0 && OutputSlot < CubePipelineConstants::SCORE_CHANNEL_HALF_COUNT);
    static_assert(OutputTile >= 0 && OutputTile < CubePipelineConstants::SCORE_OUTPUT_TILE_COUNT);
    constexpr int localTile = OutputTile & (CubePipelineConstants::SCORE_CHANNEL_HALF_COUNT - 1);
    WaitFlag<HardEvent::MTE1_M>(Event(CubePipelineConstants::QK_GROUP_EVENT));
    LocalTensor<float> scoreL0c(
        TPosition::CO1,
        OUTPUT_C_BASE + OutputSlot * OUTPUT_C_SLOT_BYTES + localTile * CubePipelineConstants::SCORE_HALF_BYTES,
        CubePipelineConstants::SCORE_HALF_BYTES / sizeof(float));
    LocalTensor<half> qkL0a(TPosition::A2, QK_L0A_BASE, QK_L0A_BYTES / sizeof(half));
    LocalTensor<half> weightL0b(TPosition::B2, W_L0B_BASE + Group * W_L0B_GROUP_BYTES,
                                W_L0B_GROUP_BYTES / sizeof(half));
    MmadParams mmParams;
    mmParams.m = K_TILE_M;
    mmParams.n = SCORE_STORAGE_COLS;
    mmParams.k = QK_GROUP_K;
    mmParams.cmatrixInitVal = (Group == CubePipelineConstants::QK_GROUP_0);
    mmParams.cmatrixSource = false;
    Mmad(scoreL0c, qkL0a, weightL0b, mmParams);
    SetFlag<HardEvent::M_MTE1>(Event(CubePipelineConstants::QK_GROUP_EVENT));
}

template <bool PrefetchFuture, int CurrentPhase, int OutputTile, int NextKSlot, int FuturePhysicalSlot, int FutureKSlot,
          int OutputSlot>
static __aicore__ inline void AdvanceQkPipeline()
{
    static_assert(CurrentPhase >= 0 && CurrentPhase < CubePipelineConstants::QK_PHASE_COUNT);
    static_assert(NextKSlot >= 0 && NextKSlot < CubePipelineConstants::KEY_L0_SLOT_COUNT);
    static_assert(FutureKSlot >= 0 && FutureKSlot < CubePipelineConstants::KEY_L0_SLOT_COUNT);
    constexpr int nextPhase = CurrentPhase ^ CubePipelineConstants::PHASE_TOGGLE;
    WaitFlag<HardEvent::FIX_MTE1>(Event(CurrentPhase));
    WaitFlag<HardEvent::MTE1_M>(Event(NextKSlot + CubePipelineConstants::KEY_L0_EVENT_BASE));
    WaitFlag<HardEvent::MTE1_FIX>(Event(nextPhase));

    PrefetchQkGroup<CurrentPhase, CubePipelineConstants::QK_GROUP_0>();
    ProduceQkGroup<nextPhase, CubePipelineConstants::QK_GROUP_0, NextKSlot>();
    ConsumeQkGroup<CubePipelineConstants::QK_GROUP_0, OutputSlot, OutputTile>();

    PrefetchQkGroup<CurrentPhase, CubePipelineConstants::QK_GROUP_1>();
    if constexpr (PrefetchFuture) {
        PrefetchKeyToL0A<FuturePhysicalSlot, FutureKSlot>();
    }
    ProduceQkGroup<nextPhase, CubePipelineConstants::QK_GROUP_1, NextKSlot>();
    ConsumeQkGroup<CubePipelineConstants::QK_GROUP_1, OutputSlot, OutputTile>();

    PrefetchQkGroup<CurrentPhase, CubePipelineConstants::QK_GROUP_2>();
    ProduceQkGroup<nextPhase, CubePipelineConstants::QK_GROUP_2, NextKSlot>();
    ConsumeQkGroup<CubePipelineConstants::QK_GROUP_2, OutputSlot, OutputTile>();

    SetFlag<HardEvent::M_MTE1>(Event(NextKSlot + CubePipelineConstants::KEY_L0_EVENT_BASE));
    SetFlag<HardEvent::FIX_MTE1>(Event(nextPhase));
    SetFlag<HardEvent::MTE1_FIX>(Event(CurrentPhase));
}

// 一个 Score tile 含四个 128 行 segment。以下包装保留全静态模板实例，
// 同时为各 segment 使用的阶段/槽位轮转命名
static __aicore__ inline void AdvanceScoreSegment0()
{
    AdvanceQkPipeline<true, CubePipelineConstants::PHASE_PING, CubePipelineConstants::SCORE_SEGMENT_0,
                      CubePipelineConstants::KEY_L0_PONG, CubePipelineConstants::PHYSICAL_KEY_SLOT_1,
                      CubePipelineConstants::KEY_L0_PING, CubePipelineConstants::SCORE_SLOT_PING>();
}

static __aicore__ inline void AdvanceScoreSegment1()
{
    AdvanceQkPipeline<true, CubePipelineConstants::PHASE_PONG, CubePipelineConstants::SCORE_SEGMENT_1,
                      CubePipelineConstants::KEY_L0_PING, CubePipelineConstants::PHYSICAL_KEY_SLOT_3,
                      CubePipelineConstants::KEY_L0_PONG, CubePipelineConstants::SCORE_SLOT_PING>();
}

template <bool PrefetchFuture>
static __aicore__ inline void AdvanceScoreSegment2()
{
    AdvanceQkPipeline<PrefetchFuture, CubePipelineConstants::PHASE_PING, CubePipelineConstants::SCORE_SEGMENT_2,
                      CubePipelineConstants::KEY_L0_PONG, CubePipelineConstants::PHYSICAL_KEY_SLOT_0,
                      CubePipelineConstants::KEY_L0_PING, CubePipelineConstants::SCORE_SLOT_PONG>();
}

static __aicore__ inline void AdvanceScoreSegment3()
{
    AdvanceQkPipeline<true, CubePipelineConstants::PHASE_PONG, CubePipelineConstants::SCORE_SEGMENT_3,
                      CubePipelineConstants::KEY_L0_PING, CubePipelineConstants::PHYSICAL_KEY_SLOT_2,
                      CubePipelineConstants::KEY_L0_PONG, CubePipelineConstants::SCORE_SLOT_PONG>();
}

template <int Phase, int OutputTile, int OutputSlot>
static __aicore__ inline void ConsumeFinalQkTile()
{
    WaitFlag<HardEvent::FIX_MTE1>(Event(Phase));
    PrefetchQkGroup<Phase, CubePipelineConstants::QK_GROUP_0>();
    ConsumeQkGroup<CubePipelineConstants::QK_GROUP_0, OutputSlot, OutputTile>();
    PrefetchQkGroup<Phase, CubePipelineConstants::QK_GROUP_1>();
    ConsumeQkGroup<CubePipelineConstants::QK_GROUP_1, OutputSlot, OutputTile>();
    PrefetchQkGroup<Phase, CubePipelineConstants::QK_GROUP_2>();
    ConsumeQkGroup<CubePipelineConstants::QK_GROUP_2, OutputSlot, OutputTile>();
    SetFlag<HardEvent::MTE1_FIX>(Event(Phase));
}

template <int Half, int OutputSlot>
static __aicore__ inline void StoreOutputHalf(const GlobalTensor<float>& scoreGm)
{
    static_assert(Half >= 0 && Half < CubePipelineConstants::SCORE_CHANNEL_HALF_COUNT);
    static_assert(OutputSlot >= 0 && OutputSlot < CubePipelineConstants::SCORE_CHANNEL_HALF_COUNT);
    SetFlag<HardEvent::M_FIX>(Event(OutputSlot + CubePipelineConstants::SCORE_OUTPUT_EVENT_BASE));
    WaitFlag<HardEvent::M_FIX>(Event(OutputSlot + CubePipelineConstants::SCORE_OUTPUT_EVENT_BASE));
    asc_sync_wait(PIPE_MTE3, PIPE_FIX, EVENT_ID0);
    LocalTensor<float> scoreL1(TPosition::A1, OUTPUT_L1_BASE, OUTPUT_L1_BYTES / sizeof(float));
    LocalTensor<float> scoreL0c(TPosition::CO1, OUTPUT_C_BASE + OutputSlot * OUTPUT_C_SLOT_BYTES,
                                OUTPUT_C_SLOT_BYTES / sizeof(float));
    constexpr int scoreRowsPerHalf = SCORE_ROWS / 2;
    constexpr int channelSplitGuardRows = 2;
    constexpr int channelSplitDestinationStride = scoreRowsPerHalf + channelSplitGuardRows;
    constexpr int channelSplitSourceStride = scoreRowsPerHalf;
    constexpr int channelSplitBurstCount = scoreRowsPerHalf;
    // 使用 void* 重载：只做通道拆分，L1 中的 FP32 结果保持原样；
    // 后三个 0 为 unit_flag_mode/quant_pre/relu_pre（不做单位转换与反量化），
    // true 启用通道拆分，false 不启用 nz2nd
    asc_copy_l0c2l1(reinterpret_cast<__cbuf__ void*>(scoreL1.GetPhyAddr()),
                    reinterpret_cast<__cc__ float*>(scoreL0c.GetPhyAddr()), static_cast<uint16_t>(SCORE_STORAGE_COLS),
                    static_cast<uint16_t>(channelSplitDestinationStride),
                    static_cast<uint32_t>(channelSplitSourceStride), static_cast<uint16_t>(channelSplitBurstCount), 0,
                    0, 0, true, false);
    SetFlag<HardEvent::FIX_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(Event(OutputSlot + CubePipelineConstants::SCORE_OUTPUT_EVENT_BASE));

    WaitFlag<HardEvent::FIX_MTE3>(EVENT_ID0);
    constexpr uint32_t halfScoreElements = scoreRowsPerHalf * SCORE_CHANNEL_COLS;
    DataCopy(scoreGm[Half * halfScoreElements], scoreL1, halfScoreElements);
    DataCopy(scoreGm[SCORE_ROWS * SCORE_CHANNEL_COLS + Half * halfScoreElements], scoreL1[halfScoreElements],
             halfScoreElements);
    asc_sync_notify(PIPE_MTE3, PIPE_FIX, EVENT_ID0);
}

static __aicore__ inline GlobalTensor<float> AcquireScoreSlot(const GlobalTensor<float>& base, int head, int& freeWaits)
{
    // 环形池绕回时，等待 AIV 释放旧数据
    if ((head >= SCORE_RING_DEPTH) && (head % SCORE_SYNC_PERIOD == 0)) {
        CrossCoreWaitFlag<CrossCoreMessageId::MODE, PIPE_S>(CrossCoreMessageId::SCORE_FREE_OR_WEIGHT_READY);
        ++freeWaits;
    }
    return base[(head % SCORE_RING_DEPTH) * SCORE_TILE_ELEMENTS];
}

static __aicore__ inline void CommitScoreSlot(int& head)
{
    ++head;
    // 每4个槽通知一次
    if (head % SCORE_SYNC_PERIOD == 0) {
        NotifyScoreReady();
    }
}

static __aicore__ inline void DrainScoreSlots(int head, int freeWaits)
{
    if (head % SCORE_SYNC_PERIOD != 0) {
        NotifyScoreReady();
    }
    const int releaseCount = CeilDiv(head, SCORE_SYNC_PERIOD);
    while (freeWaits < releaseCount) {
        CrossCoreWaitFlag<CrossCoreMessageId::MODE, PIPE_S>(CrossCoreMessageId::SCORE_FREE_OR_WEIGHT_READY);
        ++freeWaits;
    }
}

template <int PhysicalSlot, bool HasNextS1>
static __aicore__ inline void RefillK(const KeyLoadContext& context, int refillTile, int tileCount)
{
    if (refillTile < tileCount) {
        LoadKeyToL1<PhysicalSlot>(context, refillTile);
    } else if constexpr (HasNextS1) {
        LoadKeyToL1<PhysicalSlot>(context, refillTile - tileCount);
    }
}

template <bool HasNextS1>
static __aicore__ inline void ProcessS1Tile(const KeyLoadContext& keyContext, int s2,
                                            const GlobalTensor<float>& scoreRing, int& scoreHead, int& scoreFreeWaits)
{
    const int tileCount = s2 / K_TILE_M;

    // 初始装载：tile 0..3 已由 ProcessBatch 装入 L1；此处产出首个 QK tile 并回补 tile 4/5，
    // 维持 4 个物理槽的滚动窗口（窗口大小是 tileCount 的子集，不代表总计算量）
    ProduceQkTile<CubePipelineConstants::PHASE_PING, CubePipelineConstants::KEY_L0_PING>();
    RefillK<CubePipelineConstants::PHYSICAL_KEY_SLOT_0, HasNextS1>(
        keyContext, CubePipelineConstants::INITIAL_KEY_TILE_COUNT, tileCount);
    PrefetchKeyToL0A<CubePipelineConstants::PHYSICAL_KEY_SLOT_2, CubePipelineConstants::KEY_L0_PONG>();
    RefillK<CubePipelineConstants::PHYSICAL_KEY_SLOT_2, HasNextS1>(
        keyContext, CubePipelineConstants::INITIAL_KEY_TILE_COUNT + 1, tileCount);

    int tile = 0;
    // 稳态轮转：每轮产出 4 个 K tile 的 score（4 个 segment），同时按 REFILL_AFTER_SEGMENT_x
    // 回补后续 K tile，边算边补保持窗口满载
    for (; tile + CubePipelineConstants::STEADY_STATE_TILES_PER_ROUND < tileCount;
         tile += CubePipelineConstants::STEADY_STATE_TILES_PER_ROUND) {
        GlobalTensor<float> score = AcquireScoreSlot(scoreRing, scoreHead, scoreFreeWaits);

        WaitFlag<HardEvent::FIX_M>(Event(CubePipelineConstants::SCORE_OUTPUT_EVENT_BASE));
        AdvanceScoreSegment0();
        RefillK<CubePipelineConstants::PHYSICAL_KEY_SLOT_1, HasNextS1>(
            keyContext, tile + CubePipelineConstants::REFILL_AFTER_SEGMENT_0, tileCount);
        AdvanceScoreSegment1();
        RefillK<CubePipelineConstants::PHYSICAL_KEY_SLOT_3, HasNextS1>(
            keyContext, tile + CubePipelineConstants::REFILL_AFTER_SEGMENT_1, tileCount);
        StoreOutputHalf<CubePipelineConstants::SCORE_HALF_0, CubePipelineConstants::SCORE_SLOT_PING>(score);

        WaitFlag<HardEvent::FIX_M>(
            Event(CubePipelineConstants::SCORE_OUTPUT_EVENT_BASE + CubePipelineConstants::SCORE_SLOT_PONG));
        AdvanceScoreSegment2<true>();
        RefillK<CubePipelineConstants::PHYSICAL_KEY_SLOT_0, HasNextS1>(
            keyContext, tile + CubePipelineConstants::REFILL_AFTER_SEGMENT_2, tileCount);
        AdvanceScoreSegment3();
        RefillK<CubePipelineConstants::PHYSICAL_KEY_SLOT_2, HasNextS1>(
            keyContext, tile + CubePipelineConstants::REFILL_AFTER_SEGMENT_3, tileCount);
        StoreOutputHalf<CubePipelineConstants::SCORE_HALF_1, CubePipelineConstants::SCORE_SLOT_PONG>(score);
        CommitScoreSlot(scoreHead);
    }

    // 尾轮排空：处理不足一整轮（4 tile）的剩余部分；HasNextS1 决定回补是否跨 S1 预取，
    // 最后一个 segment 改用 ConsumeFinalQkTile 排空 QK 流水，不再为下一相位备料
    GlobalTensor<float> score = AcquireScoreSlot(scoreRing, scoreHead, scoreFreeWaits);
    WaitFlag<HardEvent::FIX_M>(Event(CubePipelineConstants::SCORE_OUTPUT_EVENT_BASE));
    AdvanceScoreSegment0();
    RefillK<CubePipelineConstants::PHYSICAL_KEY_SLOT_1, HasNextS1>(
        keyContext, tile + CubePipelineConstants::REFILL_AFTER_SEGMENT_0, tileCount);
    AdvanceScoreSegment1();
    RefillK<CubePipelineConstants::PHYSICAL_KEY_SLOT_3, HasNextS1>(
        keyContext, tile + CubePipelineConstants::REFILL_AFTER_SEGMENT_1, tileCount);
    StoreOutputHalf<CubePipelineConstants::SCORE_HALF_0, CubePipelineConstants::SCORE_SLOT_PING>(score);

    WaitFlag<HardEvent::FIX_M>(
        Event(CubePipelineConstants::SCORE_OUTPUT_EVENT_BASE + CubePipelineConstants::SCORE_SLOT_PONG));
    AdvanceScoreSegment2<HasNextS1>();
    ConsumeFinalQkTile<CubePipelineConstants::PHASE_PONG, CubePipelineConstants::SCORE_SEGMENT_3,
                       CubePipelineConstants::SCORE_SLOT_PONG>();
    StoreOutputHalf<CubePipelineConstants::SCORE_HALF_1, CubePipelineConstants::SCORE_SLOT_PONG>(score);
    CommitScoreSlot(scoreHead);

    SetFlag<HardEvent::M_FIX>(Event(CubePipelineConstants::FINAL_SCORE_EVENT));
    WaitFlag<HardEvent::M_FIX>(Event(CubePipelineConstants::FINAL_SCORE_EVENT));
}

} // namespace QLIV2Fast

#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_CUBE_ARCH22_H
