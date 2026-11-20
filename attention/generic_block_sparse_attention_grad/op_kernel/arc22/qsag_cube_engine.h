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
 * \file qsag_cube_engine.h
 * \brief AIC side of arch22 QSAG: L1/L0 buffers and the three gradient matmuls.
 */
#ifndef ATTENTION_QSAG_CUBE_ENGINE_ARCH22_H
#define ATTENTION_QSAG_CUBE_ENGINE_ARCH22_H

#include "qsag_common_arch22.h"

namespace QSAG {

template <typename T, typename Tiling>
class CubeEngine {
public:
    __aicore__ inline void Init(TPipe* pipe, const QsagParams& params)
    {
        // L1：Q/dO 每槽 [Gcap,Dp]，K 每槽 [Ktile,Dp]，V 每槽 [Ktile,Dcache]，
        // P/dS 各双槽 [Gcap,Ktile]。910B L1 512KiB：峰值约 448KiB（G<=128, D=256）。
        const uint32_t queryElements = params.groupCapacity * params.headDimAligned;
        const uint32_t keyElements = params.kvTileSize * params.headDimAligned;
        const uint32_t valueElements = params.kvTileSize * params.dCache;
        const uint32_t probabilityElements = params.groupCapacity * params.kvTileSize;
        const uint32_t l1Elements = 2 * params.queryBufferSlots * queryElements +
                                    SLOT_COUNT * (keyElements + valueElements + 2 * probabilityElements);
        pipe->InitBuffer(l1Buffer_, l1Elements * sizeof(T));
        uint32_t offset = 0;
        for (uint32_t slot = 0; slot < params.queryBufferSlots; ++slot) {
            queryL1_[slot] = TakeL1(queryElements, offset);
            dOutL1_[slot] = TakeL1(queryElements, offset);
        }
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            keyL1_[slot] = TakeL1(keyElements, offset);
            valueL1_[slot] = TakeL1(valueElements, offset);
            probabilityL1_[slot] = TakeL1(probabilityElements, offset);
            dsProbabilityL1_[slot] = TakeL1(probabilityElements, offset);
        }
        // Q/dO 有效行固定；padding 行只需初始化一次，后续 ND2NZ 不覆盖这些行。
        InitConstValueParams<T> zeroParams{1, static_cast<uint16_t>(queryElements * sizeof(T) / BLOCK_BYTES), 0,
                                           static_cast<T>(0)};
        for (uint32_t slot = 0; slot < params.queryBufferSlots; ++slot) {
            InitConstValue(queryL1_[slot], zeroParams);
            InitConstValue(dOutL1_[slot], zeroParams);
        }
        pipe->InitBuffer(l0aBuffer_, L0_OPERAND_BYTES);
        pipe->InitBuffer(l0bBuffer_, L0_OPERAND_BYTES);
        pipe->InitBuffer(l0cBuffer_, L0_RESULT_BYTES);
        l0a_ = l0aBuffer_.Get<T>();
        l0b_ = l0bBuffer_.Get<T>();
        l0c_ = l0cBuffer_.Get<float>();
        // 未被 TPipe::Init 预置（id0/1/2）的 M_MTE1 event id，供 Matmul 的
        // M→MTE1 fence 专用；与官方 GemmExecNmNopingpong 同一模式。
        mToMte1Event_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::M_MTE1));
    }

    __aicore__ inline void ProcessMm12(QsagContext<T, Tiling>& ctx, const TileTask& task, bool firstTile)
    {
        const QsagParams& params = ctx.params;
        const uint32_t slot = task.id % SLOT_COUNT;
        currentQueryL1_ = queryL1_[task.querySlot];
        currentDOutL1_ = dOutL1_[task.querySlot];
        AscendC::WaitEvent(GATHER_READY);
        if (firstTile) {
            LoadQueryToL1(ctx, task.token, task.head, params.groupRows);
        }
        // K 完整驻留至 MM3；V 仅缓存一个 D 分片。
        CopySelectedToL1(keyL1_[slot], ctx.selectedKey[slot], params.headDimAligned, params);
        for (uint32_t dStart = 0; dStart < params.headDimAligned; dStart += params.dTile) {
            const uint32_t width = Min(params.dTile, params.headDimAligned - dStart);
            // 上一个 V 分片已搬入 L0B，才允许覆盖当前 L1 槽。
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            CopySelectedToL1(valueL1_[slot], ctx.selectedValue[slot][dStart], width, params);
            Matmul(currentDOutL1_[dStart * params.groupCapacity], valueL1_[slot], params.groupCapacity, width,
                   params.kvTileSize, width, false, false, params.groupCapacity, params.kvTileSize, width, dStart == 0);
        }
        // Selected K/V in GM are no longer read after the last V slice reaches L1.
        CrossCoreSetFlag<2, PIPE_MTE2>(KV_FREE);
        StoreL0cToGm(ctx.gradGm[slot], AIV_PER_CUBE * params.rowsPerAiv, params.kvTileSize, params.groupCapacity,
                     params.kvTileSize, false);
        for (uint32_t dStart = 0; dStart < params.headDimAligned; dStart += params.dTile) {
            const uint32_t width = Min(params.dTile, params.headDimAligned - dStart);
            Matmul(currentQueryL1_[dStart * params.groupCapacity], keyL1_[slot][dStart * params.kvTileSize],
                   params.groupCapacity, width, params.kvTileSize, width, false, false, params.groupCapacity,
                   params.kvTileSize, width, dStart == 0);
        }
        StoreL0cToGm(ctx.scoreGm[slot], AIV_PER_CUBE * params.rowsPerAiv, params.kvTileSize, params.groupCapacity,
                     params.kvTileSize, false);
        CrossCoreSetFlag<2, PIPE_FIX>(SCORE_READY);
    }

    __aicore__ inline void ProcessMm345(QsagContext<T, Tiling>& ctx, const TileTask& task)
    {
        const QsagParams& params = ctx.params;
        const uint32_t slot = task.id % SLOT_COUNT;
        currentQueryL1_ = queryL1_[task.querySlot];
        currentDOutL1_ = dOutL1_[task.querySlot];
        AscendC::WaitEvent(PROBABILITY_READY);
        // AIV 已把 P/dS 写回 GM ND [gcap][ktile]；AIC Nd2Nz 读回 L1 NZ 供 MM4/MM5。
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = params.groupCapacity;
        copyParams.dValue = params.kvTileSize;
        copyParams.srcDValue = params.kvTileSize;
        copyParams.srcNdMatrixStride = 0;
        copyParams.dstNzC0Stride = params.groupCapacity;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(probabilityL1_[slot], ctx.probabilityGm[slot], copyParams);
        DataCopy(dsProbabilityL1_[slot], ctx.dsProbabilityGm[slot], copyParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        // dQ 的行是 query head，直接累加到 [T1,N1,Dp]；不借用 [Ktile,Dp] 的 MM4 区。
        const uint64_t dQueryOffset = Offset(ctx.tiling, task.token, task.head, true, params.headDimAligned);
        const uint64_t dQueryStride =
            ctx.tiling->inputLayout == 1 ? ctx.tiling->qSeqLen * params.headDimAligned : params.headDimAligned;
        for (uint32_t dStart = 0; dStart < params.headDimAligned; dStart += params.dTile) {
            const uint32_t width = Min(params.dTile, params.headDimAligned - dStart);
            Matmul(dsProbabilityL1_[slot], keyL1_[slot][dStart * params.kvTileSize], params.groupCapacity,
                   params.kvTileSize, params.kvTileSize, width, false, true, params.groupCapacity, width,
                   params.kvTileSize);
            StoreL0cToGm(ctx.dQueryWorkspace[dQueryOffset + dStart], params.groupRows, width, params.groupCapacity,
                         dQueryStride, true);
        }
        // MM4/MM5 的行是被选 KV token，等 Scatter 释放旧结果后再写入。
        AscendC::WaitEvent(RESULT_FREE);
        for (uint32_t dStart = 0; dStart < params.headDimAligned; dStart += params.dTile) {
            const uint32_t width = Min(params.dTile, params.headDimAligned - dStart);
            Matmul(dsProbabilityL1_[slot], currentQueryL1_[dStart * params.groupCapacity], params.groupCapacity,
                   params.kvTileSize, params.groupCapacity, width, true, true, params.kvTileSize, width,
                   params.groupCapacity);
            StoreL0cToGm(ctx.mm4[slot][dStart], params.kvTileSize, width, params.kvTileSize, params.headDimAligned,
                         false);
            Matmul(probabilityL1_[slot], currentDOutL1_[dStart * params.groupCapacity], params.groupCapacity,
                   params.kvTileSize, params.groupCapacity, width, true, true, params.kvTileSize, width,
                   params.groupCapacity);
            StoreL0cToGm(ctx.mm5[slot][dStart], params.kvTileSize, width, params.kvTileSize, params.headDimAligned,
                         false);
        }
        CrossCoreSetFlag<2, PIPE_FIX>(RESULT_READY);
    }

private:
    __aicore__ inline LocalTensor<T> TakeL1(uint32_t elements, uint32_t& offset)
    {
        auto tensor = l1Buffer_.GetWithOffset<T>(elements, offset);
        offset += elements * sizeof(T);
        return tensor;
    }

    // ND [rows, dp] -> NZ [dp/16][gcap/16][16][16]
    __aicore__ inline void CopySelectedToL1(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t columns,
                                            const QsagParams& params)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = params.kvTileSize;
        copyParams.dValue = columns;
        copyParams.srcDValue = params.headDimAligned;
        copyParams.srcNdMatrixStride = 0;
        copyParams.dstNzC0Stride = params.kvTileSize;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(dst, src, copyParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void LoadQueryToL1(QsagContext<T, Tiling>& ctx, uint64_t token, uint64_t head, uint32_t rows)
    {
        const QsagParams& params = ctx.params;
        // 调用前已等待首个 GatherReady，保证 Pre 的 Q/dO padding 已写完。
        // 上一 owner 的 L1->L0 读取完成后，才允许 MTE2 覆盖 Q/dO 槽。
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        const uint64_t offset = Offset(ctx.tiling, token, head, true, params.headDimAligned);
        const uint64_t stride =
            ctx.tiling->inputLayout == 1 ? ctx.tiling->qSeqLen * params.headDimAligned : params.headDimAligned;
        GlobalTensor<T> query = params.headDim == params.headDimAligned ? ctx.query : ctx.queryPadded;
        GlobalTensor<T> dOut = params.headDim == params.headDimAligned ? ctx.dOut : ctx.dOutPadded;
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = rows;
        copyParams.dValue = params.headDimAligned;
        copyParams.srcDValue = stride;
        copyParams.srcNdMatrixStride = 0;
        copyParams.dstNzC0Stride = params.groupCapacity;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        if (stride <= MAX_ND2NZ_SRC_STRIDE) {
            DataCopy(currentQueryL1_, query[offset], copyParams);
            DataCopy(currentDOutL1_, dOut[offset], copyParams);
        } else {
            // 长 BNSD 的 head 间距可能超限；逐行搬运仍由 AIC 直接完成 ND2NZ。
            copyParams.nValue = 1;
            copyParams.srcDValue = params.headDimAligned;
            for (uint32_t row = 0; row < rows; ++row) {
                DataCopy(currentQueryL1_[row * NZ_WIDTH], query[offset + row * stride], copyParams);
                DataCopy(currentDOutL1_[row * NZ_WIDTH], dOut[offset + row * stride], copyParams);
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    // L1(zN)→L0A，目标恒为 zZ[m][k]（dav_c220 的 V2 L1→L0 不可用，必须用 V1）。
    __aicore__ inline void LoadNzA(LocalTensor<T> dst, LocalTensor<T> src, uint32_t rows, uint32_t cols, bool transpose)
    {
        LoadData2DParams params;
        params.startIndex = 0;
        params.dstGap = 0;
        params.sid = 0;
        params.addrMode = 0;
        if (!transpose) {
            params.repeatTimes = static_cast<uint8_t>(cols / NZ_WIDTH);
            params.srcStride = static_cast<uint16_t>(rows / NZ_WIDTH);
            params.ifTranspose = false;
            for (uint32_t i = 0; i < rows / NZ_WIDTH; ++i) {
                LoadData(dst[i * cols * NZ_WIDTH], src[i * NZ_WIDTH * NZ_WIDTH], params);
            }
        } else {
            params.srcStride = 1;
            params.ifTranspose = true;
            for (uint32_t i = 0; i < cols / NZ_WIDTH; ++i) {
                params.startIndex = static_cast<uint16_t>(i * (rows / NZ_WIDTH));
                params.repeatTimes = static_cast<uint8_t>(rows / NZ_WIDTH);
                LoadData(dst[i * rows * NZ_WIDTH], src, params);
            }
        }
    }

    // L1(zN)→L0B，目标恒为 nZ[k][n]。
    __aicore__ inline void LoadNzB(LocalTensor<T> dst, LocalTensor<T> src, uint32_t rows, uint32_t cols, bool transpose)
    {
        LoadData2DParams params;
        params.startIndex = 0;
        params.dstGap = 0;
        params.sid = 0;
        params.addrMode = 0;
        if (transpose) {
            params.repeatTimes = static_cast<uint8_t>(cols / NZ_WIDTH);
            params.srcStride = static_cast<uint16_t>(rows / NZ_WIDTH);
            params.ifTranspose = true;
            for (uint32_t i = 0; i < rows / NZ_WIDTH; ++i) {
                LoadData(dst[i * cols * NZ_WIDTH], src[i * NZ_WIDTH * NZ_WIDTH], params);
            }
        } else {
            params.srcStride = 1;
            params.ifTranspose = false;
            for (uint32_t i = 0; i < cols / NZ_WIDTH; ++i) {
                params.startIndex = static_cast<uint16_t>(i * (rows / NZ_WIDTH));
                params.repeatTimes = static_cast<uint8_t>(rows / NZ_WIDTH);
                LoadData(dst[i * rows * NZ_WIDTH], src, params);
            }
        }
    }

    __aicore__ inline void Matmul(LocalTensor<T> left, LocalTensor<T> right, uint32_t leftRows, uint32_t leftCols,
                                  uint32_t rightRows, uint32_t rightCols, bool transposeLeft, bool transposeRight,
                                  uint32_t m, uint32_t n, uint32_t k, bool firstReduction = true)
    {
        // Complete previous consumers before reusing the sole L0 operands/result.
        SetFlag<HardEvent::M_MTE1>(mToMte1Event_);
        WaitFlag<HardEvent::M_MTE1>(mToMte1Event_);
        LoadNzA(l0a_, left, leftRows, leftCols, transposeLeft);
        LoadNzB(l0b_, right, rightRows, rightCols, transposeRight);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        MmadParams params;
        params.m = m;
        params.n = n;
        params.k = k;
        // Explicit M/FIX events own L0C lifetime; do not combine with the
        // implicit unitFlag locking protocol.
        params.cmatrixInitVal = firstReduction;
        params.unitFlag = 0;
        Mmad(l0c_, l0a_, l0b_, params);
    }

    __aicore__ inline void StoreL0cToGm(GlobalTensor<float> dst, uint32_t rows, uint32_t cols, uint32_t sourceRows,
                                        uint64_t stride, bool atomic)
    {
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 params;
        params.mSize = rows;
        params.nSize = cols;
        params.srcStride = sourceRows;
        params.unitFlag = 0;
        if (atomic) {
            // SetAtomicType 只设置数据类型，不会把搬运模式切到原子加；
            // dQ 的每个 tile 都写同一批行，必须显式使用 SetAtomicAdd 才能累加。
            SetAtomicAdd<float>();
        }
        if (stride <= MAX_FIXPIPE_STRIDE) {
            params.dstStride = static_cast<uint32_t>(stride);
            Fixpipe<float, float, CFG_ROW_MAJOR>(dst, l0c_, params);
        } else {
            // 超长 BNSD 的 head 步长不能截断到 32 位，改为逐行写出并保留 64 位 GM 偏移。
            params.mSize = 1;
            params.dstStride = cols;
            for (uint32_t row = 0; row < rows; ++row) {
                Fixpipe<float, float, CFG_ROW_MAJOR>(dst[row * stride], l0c_[row * NZ_WIDTH], params);
            }
        }
        if (atomic) {
            SetAtomicNone();
        }
    }

    TBuf<TPosition::A1> l1Buffer_;
    TBuf<TPosition::A2> l0aBuffer_;
    TBuf<TPosition::B2> l0bBuffer_;
    TBuf<TPosition::CO1> l0cBuffer_;
    LocalTensor<T> queryL1_[SLOT_COUNT];
    LocalTensor<T> dOutL1_[SLOT_COUNT];
    LocalTensor<T> keyL1_[SLOT_COUNT];
    LocalTensor<T> valueL1_[SLOT_COUNT];
    LocalTensor<T> probabilityL1_[SLOT_COUNT];
    LocalTensor<T> dsProbabilityL1_[SLOT_COUNT];
    LocalTensor<T> currentQueryL1_;
    LocalTensor<T> currentDOutL1_;
    LocalTensor<T> l0a_;
    LocalTensor<T> l0b_;
    LocalTensor<float> l0c_;
    event_t mToMte1Event_;
};

} // namespace QSAG
#endif // ATTENTION_QSAG_CUBE_ENGINE_ARCH22_H
