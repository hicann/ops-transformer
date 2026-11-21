/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_BACKWARD_SCHEDULE_FP32_WAVE_H
#define MHC_PRE_BACKWARD_SCHEDULE_FP32_WAVE_H

namespace MhcPreBackward {

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::PreProcessBsWave(LocalTensor<P>& output,
                                                                                           uint64_t begin, uint64_t end)
{
    // 该点积计算期间，V0 后续融合的临时数据尚未使用。
    // 将 X 暂存到所有 hPreGrad 行之后，并将 grad 保存在 input0 槽位。
    // C0 结果区（64~192 KiB）始终保持独立。
    const uint32_t rawXOffset =
        globalUbOffset_ +
        MhcPreBackwardUtils::CeilAlign(static_cast<uint32_t>(end - begin) * N_, uint32_t(CEIL_ALIGN_DEFAULT)) *
            sizeof(P);
    const uint32_t heads = Min(N_, (FP32_BUF_SIZE - rawXOffset) / uint32_t(sizeof(T)) / D_);
    auto x = fp32TBuf_.GetWithOffset<T>(heads * D_, rawXOffset);
    DataCopyParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadParams pad;
    pad.isPad = false;
    for (uint64_t row = begin; row < end; ++row) {
        LocalTensor<T> grad;
        vecInQueue0_.AllocTensor<T>(grad);
        params.blockLen = D_ * sizeof(T);
        DataCopyPad(grad, gradHInGm_[row * D_], params, pad);
        vecInQueue0_.EnQue(grad);
        vecInQueue0_.DeQue<T>(grad);
        for (uint32_t head = 0; head < N_; head += heads) {
            const uint32_t count = Min(heads, N_ - head);
            params.blockLen = count * D_ * sizeof(T);
            DataCopyPad(x, xGm_[row * nD_ + head * D_], params, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
            if (N_ == 4U && count == 4U) {
                VFDoPreProcessV0RowsN4((__ubuf__ P*)output.GetPhyAddr(), (__ubuf__ T*)x.GetPhyAddr(),
                                       (__ubuf__ T*)grad.GetPhyAddr(), 1U, D_, static_cast<uint32_t>(row - begin));
            } else {
                VFDoPreProcessV0((__ubuf__ P*)output.GetPhyAddr(), (__ubuf__ T*)x.GetPhyAddr(),
                                 (__ubuf__ T*)grad.GetPhyAddr(), count, D_, row, head, begin);
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        }
        vecInQueue0_.FreeTensor(grad);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::PublishBsWave(uint32_t)
{
    // 仅交接系数的 MTE3/MTE2 依赖，不阻塞无关的 Vector、M 或 FIX 流水。
    // 12/11/13 沿用 SDK 混合核屏障的就绪、全 AIC 收集、应答分工。
    CrossCoreSetFlag<4, PIPE_MTE3>(12U);
    CrossCoreWaitFlag<4, PIPE_S>(13U);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProduceBsWave(uint32_t wave)
{
    for (uint32_t row = 0U; row < BS_BLOCK_SIZE / vecCoreNum_; row += vecDealChunk_) {
        ProduceBsWaveRows(wave, row);
    }
    PublishBsWave(wave);
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProduceBsWaveRows(uint32_t wave, uint32_t row)
{
    PipeBarrier<PIPE_ALL>();
    const uint32_t rowsPerProducer = BS_BLOCK_SIZE / vecCoreNum_;
    prologueWaveIndex_ = wave * rowsPerProducer + row;
    globalUbOffset_ = 0;
    usedVecCoreNum_ = vecCoreNum_;
    dealStartBS_ = static_cast<uint64_t>(wave) * BS_BLOCK_SIZE + blockIdx_ * rowsPerProducer + row;
    dealEndBS_ = dealStartBS_ + vecDealChunk_;
    ProcessV0Main();
    if (wave == 0U && row == 0U && blockIdx_ < NUMBER_TWO * v1UsedCubeCoreNum_ && blockIdx_ % NUMBER_TWO == 0U) {
        // 每个输出只清零一次，之后再进行各 wave 的 gamma 原子累加。
        auto zero = vecOutQueueSmall_.AllocTensor<P>();
        Duplicate(zero, static_cast<P>(0), ND_BLOCK_SIZE);
        vecOutQueueSmall_.EnQue(zero);
        zero = vecOutQueueSmall_.DeQue<P>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = ND_BLOCK_SIZE * sizeof(P);
        params.srcStride = 0;
        params.dstStride = 0;
        const uint32_t startND = MhcPreBackwardUtils::CeilAlign(dealStartND_, uint32_t(ND_BLOCK_SIZE));
        for (uint32_t nd = startND; nd < dealEndND_; nd += ND_BLOCK_SIZE) {
            params.blockLen = Min(uint32_t(ND_BLOCK_SIZE), nD_ - nd) * sizeof(P);
            DataCopyPad(gammaGradGm_[nd], zero, params);
        }
        vecOutQueueSmall_.FreeTensor(zero);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessBsWave4FP32(uint64_t bsStart)
{
    const uint32_t nextWave = static_cast<uint32_t>(bsStart / BS_BLOCK_SIZE) + 1U;
    const bool produceNext = nextWave < totalLength_ / BS_BLOCK_SIZE;
    if (blockIdx_ >= NUMBER_TWO * v1UsedCubeCoreNum_) {
        if (produceNext) {
            ProduceBsWave(nextWave);
        }
        return;
    }
    globalUbOffset_ = PROCESS_V2_CHUNK_SIZE * sizeof(uint32_t);
    const uint32_t vecIdx = blockIdx_ % NUMBER_TWO;
    const bool splitTail = (dealEndND_ - dealStartND_) % ND_BLOCK_SIZE == ND_BLOCK_SIZE / NUMBER_TWO;
    uint32_t producedRows = 0U;
    const uint32_t rowsPerProducer = BS_BLOCK_SIZE / vecCoreNum_;
    const uint32_t tiles = CeilDiv(dealEndND_ - dealStartND_, uint32_t(ND_BLOCK_SIZE));
    const uint32_t rowsPerTick = CeilDiv(rowsPerProducer, tiles > 1U ? tiles - 1U : 1U);
    uint32_t buffId = 0;
    for (uint32_t ndBase = dealStartND_; ndBase < dealEndND_; ndBase += ND_BLOCK_SIZE) {
        const uint32_t ndSize = Min(ND_BLOCK_SIZE, dealEndND_ - ndBase);
        splitRowTail_ = splitTail && ndSize == ND_BLOCK_SIZE / NUMBER_TWO;
        rowPartitionCols_ = ndSize;
        rowPartitionTile_ = CanPartitionV2Rows(ndBase, ndSize);
        packedC0Read_ = packedC0Output_ && ndSize == ND_BLOCK_SIZE;
        vecColumnOffset_ = splitRowTail_ ? 0U : vecIdx * (ND_BLOCK_SIZE / NUMBER_TWO);
        const uint32_t nd = ndBase + vecColumnOffset_;
        directC0Chunk_ = true;
        auto sum = vecOutQueueSmall_.AllocTensor<P>();
        Duplicate(sum, static_cast<P>(0),
                  NUMBER_TWO * (rowPartitionTile_ ? rowPartitionCols_ : ND_BLOCK_SIZE / NUMBER_TWO));
        PipeBarrier<PIPE_V>();
        ProcessV2(nd, ND_BLOCK_SIZE / NUMBER_TWO, bsStart, bsStart + BS_BLOCK_SIZE, sum, buffId);
        if (!CanStreamC1(BS_BLOCK_SIZE)) {
            CrossCoreSetFlag<CROSS_CORE_FLAG_INDEX, PIPE_MTE3>(CROSS_CORE_WAIT_FLAG_C1);
        }
        vecOutQueueSmall_.EnQue(sum);
        sum = vecOutQueueSmall_.DeQue<P>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = (rowPartitionTile_ ? rowPartitionCols_ : ND_BLOCK_SIZE / NUMBER_TWO) * sizeof(P);
        params.srcStride = 0;
        params.dstStride = 0;
        if (rowPartitionTile_) {
            // 成对 AIV 的同列部分和先分开保存；同一 AIV 的各 wave 按固定顺序累加。
            auto partial = workSpaceGm_[workspaceBuf_.GetGammaPartialOffset(vecIdx, ndBase, nD_)];
            if (bsStart != 0U) {
                SetAtomicAdd<float>();
            }
            DataCopyPad(partial, sum, params);
            PipeBarrier<PIPE_MTE3>();
            if (bsStart != 0U) {
                SetAtomicNone();
            }
        } else {
            SetAtomicAdd<float>();
            DataCopyPad(gammaGradGm_[nd], sum, params);
            PipeBarrier<PIPE_MTE3>();
            SetAtomicNone();
        }
        vecOutQueueSmall_.FreeTensor(sum);
        ++buffId;
        if (buffId >= 2U && produceNext) {
            for (uint32_t row = 0U; row < rowsPerTick && producedRows < rowsPerProducer; row += vecDealChunk_) {
                ProduceBsWaveRows(nextWave, producedRows);
                producedRows += vecDealChunk_;
            }
            globalUbOffset_ = PROCESS_V2_CHUNK_SIZE * sizeof(uint32_t);
        }
    }
    if (produceNext) {
        while (producedRows < rowsPerProducer) {
            ProduceBsWaveRows(nextWave, producedRows);
            producedRows += vecDealChunk_;
        }
        PublishBsWave(nextWave);
    }
}

template <class T, class P, int8_t TILING_MODE, bool USE_HF32>
__aicore__ inline void MhcPreBackwardKernel<T, P, TILING_MODE, USE_HF32>::ProcessBsWave()
{
    const uint32_t waves = static_cast<uint32_t>(totalLength_ / BS_BLOCK_SIZE);
    if ASCEND_IS_AIV {
        waveAlphaPre_ = alphaGm_.GetValue(0);
        waveAlphaPost_ = alphaGm_.GetValue(1);
        waveAlphaComb_ = alphaGm_.GetValue(2);
        fusedInputPipeline_ = true;
        if (blockIdx_ < NUMBER_TWO * v1UsedCubeCoreNum_) {
            for (uint32_t slot = 0; slot < C0UbSlotCount(); ++slot) {
                CrossCoreSetFlag<4, PIPE_V>(C0UbFreeFlag(slot));
            }
        }
        ProduceBsWave(0U);
        for (uint32_t wave = 0; wave < waves; ++wave) {
            ProcessBsWave4FP32(static_cast<uint64_t>(wave) * BS_BLOCK_SIZE);
        }
        if (blockIdx_ < NUMBER_TWO * v1UsedCubeCoreNum_) {
            for (uint32_t slice = 0; slice < C1_L1_SLICE_COUNT; ++slice) {
                CrossCoreWaitFlag<4, PIPE_MTE3>(slice);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIC {
        const bool active = blockIdx_ < v1UsedCubeCoreNum_;
        int64_t oldHf = 0;
        int64_t oldTrans = 0;
        if constexpr (USE_HF32) {
            oldHf = GetCtrlSpr<HF32_MODE_BIT, HF32_MODE_BIT>();
            oldTrans = GetCtrlSpr<HF32_TRANS_MODE_BIT, HF32_TRANS_MODE_BIT>();
            SetHF32Mode(1);
            SetHF32TransMode(1);
        }
        if (active) {
            for (uint32_t slice = 0; slice < C1_L1_SLICE_COUNT; ++slice) {
                C1PublishSliceFree<PIPE_MTE1>(slice);
            }
        }
        for (uint32_t wave = 0; wave < waves; ++wave) {
            CrossCoreWaitFlag<4, PIPE_MTE2>(12U);
            CrossCoreWaitFlag<4, PIPE_MTE2>(12U + 16U);
            CrossCoreSetFlag<0, PIPE_MTE2>(11U);
            CrossCoreWaitFlag<0, PIPE_MTE2>(11U);
            // 将全 AIC 收集结果交给 Scalar，再向 AIV 发布全部系数写入的可见性。
            SetFlag<HardEvent::MTE2_S>(EVENT_ID7);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID7);
            CrossCoreSetFlag<4, PIPE_S>(13U);
            CrossCoreSetFlag<4, PIPE_S>(13U + 16U);
            if (!active) {
                continue;
            }
            const uint64_t bs = static_cast<uint64_t>(wave) * BS_BLOCK_SIZE;
            ProcessC0(dealStartND_, Min(ND_BLOCK_SIZE, dealEndND_ - dealStartND_), bs, BS_BLOCK_SIZE, 0U);
            uint32_t buffId = 0;
            for (uint32_t nd = dealStartND_; nd < dealEndND_; nd += ND_BLOCK_SIZE) {
                if (nd + ND_BLOCK_SIZE < dealEndND_) {
                    const uint32_t next = nd + ND_BLOCK_SIZE;
                    ProcessC0(next, Min(ND_BLOCK_SIZE, dealEndND_ - next), bs, BS_BLOCK_SIZE, buffId + 1U);
                }
                ProcessC1(nd, Min(ND_BLOCK_SIZE, dealEndND_ - nd), bs, BS_BLOCK_SIZE, buffId);
                ++buffId;
            }
        }
        if (active) {
            for (uint32_t slot = 0; slot < C0UbSlotCount(); ++slot) {
                C0WaitFreeSlot<PIPE_FIX>(slot);
            }
        }
        cubeCompute_.End();
        PipeBarrier<PIPE_ALL>();
        if constexpr (USE_HF32) {
            SetHF32TransMode(oldTrans);
            SetHF32Mode(oldHf);
        }
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        if constexpr (!USE_HF32) {
            if (UsePhiPartials()) {
                MergePhiPartials();
            }
        }
        if (rowPartitionV2_) {
            MergeGammaPartials();
        }
    }
    if ASCEND_IS_AIV {
        if (blockIdx_ == GetBlockNum() - 1U) {
            // V3 汇聚所有生产者后，恢复原有 scratch 大小。
            hFusionBufLen_ = (N_ > LARGE_N_THRESHOLD ? VEC_DEAL_CHUNK_LARGE_N : VEC_DEAL_CHUNK_SMALL_N) * fusionSize_;
            ProcessV3();
        }
    }
}

} // namespace MhcPreBackward

#endif // MHC_PRE_BACKWARD_SCHEDULE_FP32_WAVE_H
