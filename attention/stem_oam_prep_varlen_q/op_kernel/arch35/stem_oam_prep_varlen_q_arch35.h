/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * stem_oam_prep_varlen_q kernel (arch35 regbase, __simd_vf__)
 *
 * GM 输入:
 *   q:           [total_tokens, H_q, D]       FP8_e4m3fn
 *   qscale:      [total_tokens, H_q]          FP32
 *   qSeqLens:    [batch]                       INT64
 *   cuSeqLensQ:  [batch+1]                     INT64
 *
 * GM 输出:
 *   qFlat:       [batch, H_q, max_Qb, S*D]    BF16
 *
 * 参数: B=stemBlockSize(128), S=stemStride(16), R=B/S(8), D=dim_qk(128)
 *
 * 计算流程 (每核处理 1 个 qb block):
 *
 *   1. CopyInQBlock:
 *      GM → UB: q[cu_off + qb*B : cu_off + qb*B + B, h, :]
 *      qBlockQue: [B, D] FP8, 超出 q_len 的行填零
 *
 *   2. CopyInScalesBulk:
 *      GM → UB: qscale[cu_off + qb*B : cu_off + qb*B + B, h]
 *      scalesQue: [B] FP32
 *
 *   3. CastBlockToFP32:
 *      UB: qBlockQue [B, D] FP8 → qBlockFP32Que [B, D] FP32
 *
 *   4. AccumulateBlock (Weighted Group Sum, in-place binary reduce):
 *      qBlockFP32Que 逻辑 reshape 为 [R, S, D]
 *      scalesQue 逻辑 reshape 为 [R, S]
 *
 *      Phase 1: qBlockFP32Que[(r*S+g)*D : (r*S+g+1)*D] *= scales[r*S+g]
 *                对每个 (r, g) 的原地乘 scale
 *      Phase 2: 沿 R 轴反向 reduce（stride 从 rMain 递减到 1）
 *                结果保留在 qBlockFP32Que 的前 S 行 (偏移 0..S*D-1)
 *
 *   5. FlattenAndCast:
 *      UB: qBlockFP32Que[0 : S*D] FP32 → Cast → outQueueQflat [S*D] BF16
 *
 *   6. CopyOutBlock:
 *      UB → GM: qFlat[b, h, qb, :] = outQueueQflat [S*D] BF16
 */

#ifndef STEM_OAM_PREP_VARLEN_Q_REGBASE_ARCH35_H
#define STEM_OAM_PREP_VARLEN_Q_REGBASE_ARCH35_H

#include "kernel_operator.h"
#include "stem_oam_prep_varlen_q_tiling_data.h"

using namespace AscendC;

constexpr uint32_t DIM_QK = 128;
constexpr uint32_t DOUBLE_BUFFER_DEPTH = 2;
constexpr uint32_t SCALE_PAD_SIZE = 8;

__simd_callee__ inline void MulsOnAllRows(__ubuf__ float* qFP32Ptr, __ubuf__ float* scalePtr, uint16_t R, uint16_t S,
                                          uint16_t D, uint16_t scaleStride)
{
    using namespace AscendC::Reg;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (D + VL_F - 1) / VL_F;

    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> regScale;
    RegTensor<float> regA;

    for (uint16_t r = 0; r < R; r++) {
        for (uint16_t g = 0; g < S; g++) {
            uint16_t rowIdx = static_cast<uint16_t>(r * S + g);
            uint32_t rowOff = static_cast<uint32_t>(rowIdx) * D;
            DataCopy<float, LoadDist::DIST_BRC_B32>(regScale, scalePtr + rowIdx * scaleStride);
            for (uint16_t vi = 0; vi < vlIter; vi++) {
                uint32_t off = rowOff + static_cast<uint32_t>(vi) * VL_F;
                DataCopy(regA, qFP32Ptr + off);
                Mul(regA, regA, regScale, maskAll);
                DataCopy(qFP32Ptr + off, regA, maskAll);
            }
        }
    }
}

__simd_vf__ inline void WeightedBinaryReduceFull(__ubuf__ float* qFP32Ptr, __ubuf__ float* scalePtr, uint16_t R,
                                                 uint16_t S, uint16_t D, uint16_t scaleStride)
{
    MulsOnAllRows(qFP32Ptr, scalePtr, R, S, D, scaleStride);

    using namespace AscendC::Reg;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (D + VL_F - 1) / VL_F;

    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> regA;
    RegTensor<float> regB;

    uint16_t rMain = 1;
    while ((rMain << 1) <= R) {
        rMain <<= 1;
    }
    uint16_t rTail = R - rMain;

    for (uint16_t stride = rMain; stride >= 1; stride >>= 1) {
        uint16_t ops = (stride == rMain) ? rTail : stride;
        for (uint16_t g = 0; g < S; g++) {
            for (uint16_t r = 0; r < ops; r++) {
                uint32_t offDst = static_cast<uint32_t>(r * S + g) * D;
                uint32_t offSrc = static_cast<uint32_t>((r + stride) * S + g) * D;
                for (uint16_t vi = 0; vi < vlIter; vi++) {
                    uint32_t d = static_cast<uint32_t>(vi) * VL_F;
                    DataCopy(regA, qFP32Ptr + offDst + d);
                    DataCopy(regB, qFP32Ptr + offSrc + d);
                    Add(regA, regA, regB, maskAll);
                    DataCopy(qFP32Ptr + offDst + d, regA, maskAll);
                }
            }
        }
    }
}

class StemOamPrepVarlenQ {
public:
    __aicore__ inline StemOamPrepVarlenQ() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR qscale, GM_ADDR cuSeqlensQ, GM_ADDR qflat,
                                const StemPrepQTilingData* tiling)
    {
        this->tiling = tiling;
        taskId = GetBlockIdx();
        if (static_cast<int64_t>(taskId) >= tiling->usedCoreNum) {
            return;
        }

        numQHeads = tiling->numQHeads;
        dimQk = tiling->dimQk;
        B = tiling->stemBlockSize;
        S = tiling->stemStride;
        R = tiling->rVal;
        kflatDim = tiling->kflatDim;
        totalTokens = tiling->totalTokens;
        ppNum = tiling->ubFactor > 1 ? DOUBLE_BUFFER_DEPTH : 1;

        mte2sEventId = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_S>());
        mte2sScaleEventId = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_S>());

        InitTaskRange(tiling->blocksPerCoreBase, tiling->blocksRemainder);
        InitGlobalBuffers(q, qscale, cuSeqlensQ, qflat);
        InitUBBuffers();
        LoadCuSeqLensQ();
    }

    __aicore__ inline void Process()
    {
        if (static_cast<int64_t>(taskId) >= tiling->usedCoreNum)
            return;

        // 等待 LoadCuSeqLensQ 完成（只执行一次）
        WaitFlag<HardEvent::MTE2_S>(mte2sEventId);

        for (int64_t taskIdx = qbStart; taskIdx < qbEnd; taskIdx++) {
            DecodeTaskId(taskIdx);
            ProcessOneBlock();
        }
    }

    __aicore__ inline void DecodeTaskId(int64_t taskIdx)
    {
        // 整数除法定位 batch（替代线性扫描，O(1) 复杂度）
        int64_t tasksPerBatch = tiling->maxQb * numQHeads;
        batchIdx = taskIdx / tasksPerBatch;
        int64_t inBatch = taskIdx % tasksPerBatch;
        headIdx = inBatch / tiling->maxQb;
        qbLocal = inBatch % tiling->maxQb;

        // 从 cuSeqLensQ 读取当前 batch 的信息
        LocalTensor<int64_t> cuSeqLensQ = cuSeqLensQBuf.Get<int64_t>();
        int64_t cuStart = cuSeqLensQ.GetValue(batchIdx);
        int64_t cuEnd = cuSeqLensQ.GetValue(batchIdx + 1);

        cuOff = cuStart;
        qLen = cuEnd - cuStart;
        numQb = (qLen + B - 1) / B;
    }

    __aicore__ inline void ProcessOneBlock()
    {
        if (totalTokens == 0 || qbLocal >= numQb) {
            WriteZeroBlock();
            return;
        }
        CopyInQBlock();
        CopyInScalesBulk();
        CastBlockToFP32();
        AccumulateBlock();
        FlattenAndCast();
        CopyOutBlock();
    }

    __aicore__ inline void CopyInQBlock()
    {
        LocalTensor<fp8_e4m3fn_t> qBlockLocal = qBlockQue.AllocTensor<fp8_e4m3fn_t>();
        Duplicate((LocalTensor<int8_t>&)qBlockLocal, (int8_t)0, B * dimQk);
        event_t vmte2Event = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vmte2Event);
        WaitFlag<HardEvent::V_MTE2>(vmte2Event);

        int64_t startRow = qbLocal * B;
        int64_t validRows = (startRow + B <= qLen) ? B : (qLen > startRow ? qLen - startRow : 0);

        if (validRows > 0) {
            int64_t qOff = (cuOff + startRow) * numQHeads * dimQk + headIdx * dimQk;
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(validRows);
            copyParams.blockLen = static_cast<uint32_t>(dimQk * sizeof(fp8_e4m3fn_t));
            copyParams.srcStride = (numQHeads - 1) * dimQk * sizeof(fp8_e4m3fn_t);
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            GlobalTensor<int8_t> qGmBytes = qGm.ReinterpretCast<int8_t>();
            DataCopyPad((LocalTensor<int8_t>&)qBlockLocal, qGmBytes[qOff], copyParams,
                        DataCopyPadExtParams<int8_t>{false, 0, 0, static_cast<int8_t>(0)});
        }
        qBlockQue.EnQue(qBlockLocal);
    }

    __aicore__ inline void CastBlockToFP32()
    {
        LocalTensor<fp8_e4m3fn_t> qBlockLocal = qBlockQue.DeQue<fp8_e4m3fn_t>();
        LocalTensor<float> qBlockFP32Local = qBlockFP32Que.AllocTensor<float>();
        Duplicate(qBlockFP32Local, 0.0f, B * dimQk);
        PipeBarrier<PIPE_V>();

        int64_t startRow = qbLocal * B;
        int64_t validRows = (startRow + B <= qLen) ? B : (qLen > startRow ? qLen - startRow : 0);
        int64_t castCount = validRows * dimQk;
        if (castCount > 0) {
            Cast<float, fp8_e4m3fn_t>(qBlockFP32Local, qBlockLocal, RoundMode::CAST_NONE,
                                      static_cast<int32_t>(castCount));
        }
        qBlockQue.FreeTensor(qBlockLocal);
        qBlockFP32Que.EnQue(qBlockFP32Local);
    }

    __aicore__ inline void CopyInScalesBulk()
    {
        int64_t startRow = qbLocal * B;
        int64_t validRows = (startRow + B <= qLen) ? B : (qLen > startRow ? qLen - startRow : 0);
        if (validRows > 0) {
            int64_t baseOffset = (cuOff + startRow) * numQHeads + headIdx;
            int64_t gmRange = validRows * numQHeads;
            qscaleGm.SetGlobalBuffer((__gm__ float*)qscaleBase + baseOffset, gmRange);
            LocalTensor<float> bulkBuf = scaleBulkBuf.Get<float>();
            if (validRows < B) {
                Duplicate(bulkBuf, 0.0f, B * SCALE_PAD_SIZE);
                event_t vsEvent = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(vsEvent);
                WaitFlag<HardEvent::V_MTE2>(vsEvent);
            }
            uint32_t copyCount = (numQHeads < SCALE_PAD_SIZE) ? numQHeads : SCALE_PAD_SIZE;
            LoopModeParams loopParams;
            loopParams.loop1Size = validRows;
            loopParams.loop1SrcStride = numQHeads * sizeof(float);
            loopParams.loop1DstStride = SCALE_PAD_SIZE * sizeof(float);
            loopParams.loop2Size = 1;
            loopParams.loop2SrcStride = 0;
            loopParams.loop2DstStride = 0;
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(copyCount * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams = {false, 0, 0, 0};
            DataCopyPad(bulkBuf, qscaleGm[0], copyParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
        SetFlag<HardEvent::MTE2_S>(mte2sScaleEventId);
    }

    __aicore__ inline void AccumulateBlock()
    {
        WaitFlag<HardEvent::MTE2_S>(mte2sScaleEventId);
        LocalTensor<float> fp32Buf = qBlockFP32Que.DeQue<float>();
        LocalTensor<float> scaleBuf = scaleBulkBuf.Get<float>();
        __ubuf__ float* qFP32Ptr = (__ubuf__ float*)fp32Buf.GetPhyAddr();
        __ubuf__ float* scalePtr = (__ubuf__ float*)scaleBuf.GetPhyAddr();

        VF_CALL<WeightedBinaryReduceFull>(qFP32Ptr, scalePtr, static_cast<uint16_t>(R), static_cast<uint16_t>(S),
                                          static_cast<uint16_t>(dimQk), static_cast<uint16_t>(SCALE_PAD_SIZE));

        qBlockFP32Que.EnQue(fp32Buf);
    }

    __aicore__ inline void FlattenAndCast()
    {
        LocalTensor<float> groupSumLocal = qBlockFP32Que.DeQue<float>();
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat.AllocTensor<bfloat16_t>();
        Cast<bfloat16_t, float>(qflatLocal, groupSumLocal, RoundMode::CAST_RINT, static_cast<int32_t>(kflatDim));
        qBlockFP32Que.FreeTensor(groupSumLocal);
        outQueueQflat.EnQue(qflatLocal);
    }

    __aicore__ inline void CopyOutBlock()
    {
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat.DeQue<bfloat16_t>();
        int64_t outOffset = batchIdx * (numQHeads * tiling->maxQb * kflatDim) + headIdx * (tiling->maxQb * kflatDim) +
                            qbLocal * kflatDim;
        DataCopyPad(qflatGm[outOffset], qflatLocal, {1, static_cast<uint16_t>(kflatDim * sizeof(bfloat16_t)), 0, 0});
        outQueueQflat.FreeTensor(qflatLocal);
    }

    __aicore__ inline void WriteZeroBlock()
    {
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat.AllocTensor<bfloat16_t>();
        Duplicate(qflatLocal, (bfloat16_t)0, kflatDim);
        outQueueQflat.EnQue<bfloat16_t>(qflatLocal);
        LocalTensor<bfloat16_t> qflatOut = outQueueQflat.DeQue<bfloat16_t>();
        int64_t outOffset = batchIdx * (numQHeads * tiling->maxQb * kflatDim) + headIdx * (tiling->maxQb * kflatDim) +
                            qbLocal * kflatDim;
        DataCopyPad(qflatGm[outOffset], qflatOut, {1, static_cast<uint16_t>(kflatDim * sizeof(bfloat16_t)), 0, 0});
        outQueueQflat.FreeTensor(qflatOut);
    }

private:
    __aicore__ inline void InitTaskRange(int64_t tasksPerCoreBase, int64_t tasksRemainder)
    {
        int64_t core = static_cast<int64_t>(taskId);
        if (core < tasksRemainder) {
            qbStart = core * (tasksPerCoreBase + 1);
            qbEnd = qbStart + tasksPerCoreBase + 1;
        } else {
            qbStart = tasksRemainder * (tasksPerCoreBase + 1) + (core - tasksRemainder) * tasksPerCoreBase;
            qbEnd = qbStart + tasksPerCoreBase;
        }
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR q, GM_ADDR qscale, GM_ADDR cuSeqlensQ, GM_ADDR qflat)
    {
        qGm.SetGlobalBuffer((__gm__ fp8_e4m3fn_t*)q);
        qscaleBase = qscale;
        cuSeqLensGm.SetGlobalBuffer((__gm__ int64_t*)cuSeqlensQ);
        qflatGm.SetGlobalBuffer((__gm__ bfloat16_t*)qflat);
    }

    __aicore__ inline void InitUBBuffers()
    {
        pipe.InitBuffer(qBlockQue, ppNum, B * dimQk * sizeof(fp8_e4m3fn_t));
        pipe.InitBuffer(qBlockFP32Que, ppNum, B * dimQk * sizeof(float));
        pipe.InitBuffer(outQueueQflat, ppNum, kflatDim * sizeof(bfloat16_t));
        pipe.InitBuffer(scaleBulkBuf, B * SCALE_PAD_SIZE * sizeof(float));
    }

    __aicore__ inline void LoadCuSeqLensQ()
    {
        int64_t cuSeqLensQCount = tiling->batchSize + 1;
        pipe.InitBuffer(cuSeqLensQBuf, static_cast<uint32_t>(cuSeqLensQCount * sizeof(int64_t)));
        LocalTensor<int64_t> cuSeqLensQLocal = cuSeqLensQBuf.Get<int64_t>();
        DataCopyPad(cuSeqLensQLocal, cuSeqLensGm, {1, static_cast<uint16_t>(cuSeqLensQCount * sizeof(int64_t)), 0, 0},
                    {0, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(mte2sEventId);
    }

    TPipe pipe;
    const StemPrepQTilingData* tiling = nullptr;

    uint32_t taskId = 0;
    int64_t batchIdx = 0;
    int64_t headIdx = 0;
    int64_t numQHeads = 0;
    int64_t dimQk = 0;
    int64_t qLen = 0;
    int64_t cuOff = 0;
    int64_t numQb = 0;
    int64_t B = 0;
    int64_t S = 0;
    int64_t R = 0;
    int64_t kflatDim = 0;
    int64_t totalTokens = 0;

    int64_t qbStart = 0;
    int64_t qbEnd = 0;
    int64_t qbLocal = 0;
    uint32_t ppNum = 2;

    event_t mte2sEventId;
    event_t mte2sScaleEventId;

    GM_ADDR qscaleBase = nullptr;

    GlobalTensor<fp8_e4m3fn_t> qGm;
    GlobalTensor<float> qscaleGm;
    GlobalTensor<int64_t> cuSeqLensGm;
    GlobalTensor<bfloat16_t> qflatGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> qBlockQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> qBlockFP32Que;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER_DEPTH> outQueueQflat;
    TBuf<QuePosition::VECCALC> cuSeqLensQBuf;
    TBuf<QuePosition::VECCALC> scaleBulkBuf;
};

#endif
