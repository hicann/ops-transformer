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
 * \file causal_conv1d_update.h
 * \brief Decode mode — sequence-tiled incremental causal conv1d.
 *
 * Update / decode 模式下每个 batch 的 seqLen = 1（逐 token 增量推理）。
 * 调用链: Init() → Process() → ProcessImpl() → InitBlock() + ProcessBlock()。
 *
 * ProcessBlock 对每个 batch 组依次执行:
 *   1. InitRing  — 从 conv_states 加载历史状态（width-1 行）+ 当前 x 写入 ring buffer
 *   2. RunSeq     — 时间步循环: x 预取 / conv1d 计算 / y 写出 / state 写出
 *   3. WriteBackState — 将 ring 中存活的 state 写回 GM
 *
 * coBatch > 1 时 ring 布局从 [width+1][baseDim] 扩展为 [width+1][coBatch*baseDim]，
 * 在同一个 kernel invocation 内同时处理 coBatch 个 batch 的 token。
 */

#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "causal_conv1d.h"

namespace CausalConv1d {

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
class CausalConv1dUpdate : public CausalConv1d<T, inputModeKey, widthKey, hasBiasKey, activationKey> {
public:
    // ---- Init: 绑定 GM buffer + 分配 UB buffer --------------------------
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR bias, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR, GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut,
                                GM_ADDR y, GM_ADDR workspace, const CausalConv1dTilingData *tilingData)
    {
        (void)workspace;
        this->InitGlobalBuffers(x, weight, convStates, bias, queryStartLoc, cacheIndices, y, convStatesOut, tilingData);
        this->numAcceptedTokensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(numAcceptedTokens));
        this->InitBuffers(); // 分配 inBuf / outBuf / calcBuf，并初始化 dimBufferSize_
    }

    // ---- Process: 按 varlen / batch 模式分发 -----------------------------
    __aicore__ inline void Process()
    {
        if constexpr (IsVarlenInputModeKey(inputModeKey)) {
            this->template ProcessImpl<SEQ_PARTITION_MODE_VARLEN>();
        } else {
            this->template ProcessImpl<SEQ_PARTITION_MODE_BATCH>();
        }
    }

protected:
    // ---- ProcessImpl: Init + Loop 的入口 ---------------------------------
    template <int32_t kWindowMode>
    __aicore__ inline void ProcessImpl()
    {
        int64_t dimStart;
        int32_t curBaseDim;
        int64_t seqStart;
        int64_t seqEnd;
        if (!this->template InitBlock<kWindowMode>(dimStart, curBaseDim, seqStart, seqEnd)) {
            return; // 本 core 无有效任务
        }
        this->template ProcessBlock<kWindowMode>(dimStart, curBaseDim, seqStart, seqEnd);
    }

    // ---- InitBlock: tiling 读取 + core 分配 + weight/bias 加载 -------------
    //
    // 1. 根据 blockIdx 计算本 core 负责的 channel 段 [dimStart, dimStart+curBaseDim)
    // 2. 加载 weight/bias (DMA → calcBuf, bf16 → f32 cast)
    // 3. 计算本 core 负责的 batch 组范围 [seqStart, seqEnd)
    //
    // coBatch 语义: 每个 batch 组包含 coBatch 个连续 batch，
    // seqStart/seqEnd 以"组"为单位，batchIdx * coBatch 得到真实 batch 起始。
    template <int32_t kWindowMode>
    __aicore__ inline bool InitBlock(int64_t &dimStart, int32_t &curBaseDim, int64_t &seqStart, int64_t &seqEnd)
    {
        const int64_t dim = this->tilingData_->dim;
        const int64_t batch = this->tilingData_->batch;
        const int32_t baseDim = this->tilingData_->baseDim;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
        const uint32_t blockIdx = GetBlockIdx();
        const int64_t tokensPerBlock = this->tilingData_->tokensPerBlock;
        const int64_t tokenBlockCnt = this->tilingData_->tokenBlockCnt;
        const int64_t coreCnt = tokenBlockCnt * baseDimCnt;
        if (static_cast<int64_t>(blockIdx) >= coreCnt) {
            return false;
        }

        const int32_t tokBlock = blockIdx / baseDimCnt;
        const int32_t baseDimIdx = blockIdx % baseDimCnt;
        dimStart = baseDimIdx * baseDim;
        curBaseDim = (dimStart + baseDim <= dim) ? baseDim : (dim - dimStart);

        this->InitCalcBuf();
        this->LoadWeightAndBias(dimStart, curBaseDim);

        const int64_t tokensPerGroupBlock = tokensPerBlock / coBatch;
        const int64_t batchGroups = batch / coBatch;
        seqStart = tokBlock * tokensPerGroupBlock;
        seqEnd = (seqStart + tokensPerGroupBlock < batchGroups) ? (seqStart + tokensPerGroupBlock) : batchGroups;
        return true;
    }

    // ---- ProcessBlock: 逐个 batch 组处理 ----------------------------------
    //
    // 对 [seqStart, seqEnd) 范围内每个 batch 组:
    //   1. GetSeqWindow — 确定该组的 seq 窗口
    //   2. ResolveBatchCacheIndex — 查 cacheIndices 获取物理 cache slot
    //   3. InitRing — 填充 ring buffer (历史 state + 当前 x)
    //   4. RunSeq — 时间步循环 (compute + DMA 流水)
    //   5. WriteBackState — 存活 state 写回 GM
    //   6. SetEvent<MTE3_MTE2> — 确保 state 写完成后下一轮 state 读可以开始
    template <int32_t kWindowMode>
    __aicore__ inline void ProcessBlock(int64_t dimStart, int32_t curBaseDim, int64_t seqStart, int64_t seqEnd)
    {
        const int64_t dim = this->tilingData_->dim;
        const int64_t seqLen = this->tilingData_->seqLen;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int64_t kernelWidth = this->tilingData_->kernelWidth;

        for (int64_t batchIdx = seqStart; batchIdx < seqEnd; ++batchIdx) {
            int64_t curSeqStart = 0;
            int64_t curSeqEnd = 0;
            this->template GetSeqWindow<kWindowMode>(static_cast<int32_t>(batchIdx), seqLen, coBatch, curSeqStart,
                                                     curSeqEnd);
            int32_t len;
            if constexpr (kWindowMode == SEQ_PARTITION_MODE_BATCH) {
                len = static_cast<int32_t>(seqLen);
            } else {
                len = static_cast<int32_t>(curSeqEnd - curSeqStart);
            }

            const int32_t batchBase = static_cast<int32_t>(batchIdx * coBatch);
            int32_t cacheIdx;
            if (!this->ResolveBatchCacheIndex(batchBase, this->tilingData_->hasCacheIndices, cacheIdx)) {
                continue;
            }

            const int32_t stateTokenOffset = this->GetStateTokenOffset(batchBase);

            this->InitRing(cacheIdx, stateTokenOffset, curSeqStart, len, dimStart, curBaseDim, dim);

            // spec decode 时 conv_state 按 sliding window 更新:
            //   new_state = [old[offset+1:offset+width-1], x[0:len]]
            // 增量写出在 RunSeq 中逐步填充 [0..stateWriteStart-1]
            // WriteBackState 固定写末尾 lastWriteLen=width-1 行到 [stateWriteStart..]
            //   stateWriteStart = min(len-1, stateLen-lastWriteLen)
            //   - len <= stateLen: 增量写 stateWriteStart 行 + 回写 width-1 行，ring 中有所需值
            //   - len >  stateLen: 增量写满 [0..stateLen-1]，ring 中旧值已被覆盖，跳过回写
            bool doVarLenStateWrite = false;
            int32_t stateWriteStart = 0;
            const int64_t stateLen = this->tilingData_->stateLen;
            const int32_t lastWriteLen = static_cast<int32_t>(kernelWidth - 1);
            if (this->tilingData_->hasNumAcceptedTokens && len > 0) {
                doVarLenStateWrite = true;
                if (len - 1 < stateLen) {
                    stateWriteStart = (len - 1 < stateLen - lastWriteLen) ? (len - 1) : (stateLen - lastWriteLen);
                } else {
                    stateWriteStart = stateLen;
                }
            }

            this->RunSeq(curSeqStart, len, dimStart, curBaseDim, cacheIdx, doVarLenStateWrite, stateWriteStart);

            if (stateWriteStart < stateLen) {
                this->WriteBackState(cacheIdx, stateWriteStart + 1, dimStart, curBaseDim, dim, stateWriteStart);
            }

            // 保证 state 写出完成后再开始下一 batch 组的 InitRing state 读入
            SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
    }

    // ---- GetStateTokenOffset: 查询 numAcceptedTokens 得到 state 偏移 ----------
    __aicore__ inline int32_t GetStateTokenOffset(int32_t seq) const
    {
        if (!this->tilingData_->hasNumAcceptedTokens) {
            return 0;
        }
        int32_t offset = this->numAcceptedTokensGm_.GetValue(seq) - 1;
        return (offset >= 0) ? offset : 0;
    }

    // ---- InitRing: 初始化 ring buffer ------------------------------------
    //
    // DMA 流水线:
    //   MTE2: conv_states[stateBase] → ring[0..width-2]  (历史 state, width-1 行)
    //   MTE2: x[start]               → ring[slot0]        (当前 token)
    //   SetFlag<MTE2_V>                                   (通知 Vector 数据就绪)
    //
    // coBatch > 1 时:
    //   - SetLoopMode 使 DMA 自动遍历 coBatch 个 batch 的 state/x 数据
    //   - ring 布局: [width+1][dimBufferSize_] = [width+1][coBatch * baseDim]
    __aicore__ inline void InitRing(int32_t cacheIdx, int32_t stateTokenOffset, int64_t start, int32_t len,
                                    int64_t dimStart, int32_t baseDim, int64_t dim)
    {
        const int64_t stateLen = this->tilingData_->stateLen;
        const int64_t kernelWidth = this->tilingData_->kernelWidth;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int64_t seqLen = this->tilingData_->seqLen;
        LocalTensor<T> ring = this->inTensor_;

        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t srcGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);

        // ---- 1. 从 conv_states 加载历史 state (width-1 行) -----------------
        // state 在 GM 中的布局: [cacheIdx][stateLen][dim]，行间 stride = dim
        // ring 中的布局: [width+1][dimBufferSize_]，行间 stride = dimBufferSize_
        const int64_t stateBase =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(stateTokenOffset) * dim + dimStart;

        LoopModeParams loopParams;
        loopParams.loop1Size = static_cast<uint32_t>(coBatch);
        loopParams.loop2Size = 1;
        loopParams.loop1SrcStride = static_cast<uint64_t>(stateLen) * dim * sizeof(T);
        loopParams.loop1DstStride = static_cast<uint64_t>(baseDim) * sizeof(T);
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(kernelWidth - 1);
        copyParams.blockLen = blockBytes;
        copyParams.srcStride = srcGap;
        copyParams.dstStride = static_cast<uint32_t>(this->dimBufferSize_ - baseDim) * sizeof(T) / this->kUbBlockSize;
        copyParams.rsv = 0;
        DataCopyPad(ring[0], this->convStatesGm_[stateBase], copyParams, {false, 0, 0, 0});
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

        // ---- 2. 从 x 加载当前 token 到 ring 的 slot0 -----------------------
        // slot0 = (width-1) % (width+1) = width-1，即最后一个 slot
        const int32_t slot0 = CurrSlot(0, kernelWidth);
        const uint32_t srcGapBatch = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);

        DataCopyPad(ring[slot0 * this->dimBufferSize_], this->xGm_[start * dim + dimStart],
                    {static_cast<uint16_t>(coBatch), blockBytes, srcGapBatch, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    // ---- RunSeq: 时间步流水循环 -----------------------------------------
    //
    // 每个时间步 t 的流水线（4 级流水）:
    //   MTE2: 预取 x[t+1] 到 ring[slotNext]           (与当前步计算并行)
    //   Vec:  ComputeConv1d — ring 卷积 + bias + 激活  (核心计算)
    //   MTE3: 写出 y[t] 到 GM                          (输出)
    //   MTE3: 写出 state 到 GM (doVarLenStateWrite 时)  (增量 state 更新)
    //
    // 事件同步:
    //   MTE2_V:  x DMA → Vector 数据依赖
    //   MTE3_MTE2: MTE3 写完成 → 下一轮 MTE2 读可以开始
    //   V_MTE3:   Vector 计算完成 → MTE3 写出可以开始
    //   MTE3_V:   MTE3 读取 ring 完成 → Vector 可以复用 ring slot
    //
    // Update 模式 (seqLen=1): 循环体仅执行 t=0 一次，
    // x 预取分支 (t+1<len) 和 state 写出分支 (t+2<len) 均不执行。
    __aicore__ inline void RunSeq(int64_t start, int32_t len, int64_t dimStart, int32_t baseDim, int32_t cacheIdx,
                                  bool doVarLenStateWrite, int32_t stateWriteStart)
    {
        const int64_t dim = this->tilingData_->dim;
        const int64_t kernelWidth = this->tilingData_->kernelWidth;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int64_t seqLen = this->tilingData_->seqLen;
        const int64_t stateLen = this->tilingData_->stateLen;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        LocalTensor<T> ring = this->inTensor_;
        LocalTensor<T> outT = this->outTensor_;

        for (int32_t t = 0; t < len; ++t) {
            // ---- 等当前步的 x 数据就绪 (InitRing 或上一轮预取) --------------
            WaitFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);

            // ---- x 预取: t+1 步的 x 加载到 ring 的下一个 slot -------------
            if (t + 1 < len) {
                const int32_t slotNext = NextSlot(t, kernelWidth);
                if (t > 0) {
                    // 等上一轮 y/state 写出完成，ring slot 已释放
                    WaitFlag<HardEvent::MTE3_MTE2>((t & 1) ? EVENT_ID1 : EVENT_ID0);
                }
                const int64_t xGmBase = (start + t + 1) * dim + dimStart;
                const uint32_t srcGapX = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
                DataCopyPad(ring[slotNext * this->dimBufferSize_], this->xGm_[xGmBase],
                            {static_cast<uint16_t>(coBatch), blockBytes, srcGapX, 0, 0}, {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            }

            // ---- 等 MTE3 释放 ring slot (t >= 2 的 state 写出) ----------
            const int32_t outSlot = t & 1;
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            // ---- Vector 卷积计算 -------------------------------------------
            this->ComputeConv1d(ring, this->pool_.weight, this->pool_.bias, outT[outSlot * this->dimBufferSize_],
                                this->dimBufferSize_, t);

            // ---- y 写出: Vector 结果 → GM --------------------------------
            SetFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);

            const int64_t outGmBase = (start + t) * dim + dimStart;
            const uint32_t dstGapY = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);

            DataCopyPad(this->yGm_[outGmBase], outT[outSlot * this->dimBufferSize_],
                        {static_cast<uint16_t>(coBatch), blockBytes, 0, dstGapY, 0});
            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            // ---- varLen state 增量写出: ring 中已刷新的 state 行 → GM -----
            // 只写 [0..stateWriteStart-1]，末尾 width-1 行由 WriteBackState 统一回写
            if (doVarLenStateWrite && t >= 1 && (t - 1) < stateWriteStart) {
                const int32_t deadSlot = t % (kernelWidth + 1);
                const int64_t stateBase =
                    static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(t - 1) * dim + dimStart;
                const uint32_t dstGapState = static_cast<uint32_t>(stateLen * dim - baseDim) * sizeof(T);
                DataCopyPad(this->convStatesOutGm_[stateBase], ring[deadSlot * this->dimBufferSize_],
                            {static_cast<uint16_t>(coBatch), blockBytes, 0, dstGapState, 0});
            }

            // ---- 下一轮 MTE2 预取可以开始 ----------------------------------
            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_MTE2>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            }
        }
    }
};

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
__aicore__ inline void RunCausalConv1dUpdate(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR bias,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                             GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut, GM_ADDR y,
                                             GM_ADDR workspace, const CausalConv1dTilingData *tilingData)
{
    CausalConv1dUpdate<T, inputModeKey, widthKey, hasBiasKey, activationKey> op;
    op.Init(x, weight, convStates, bias, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
            convStatesOut, y, workspace, tilingData);
    op.Process();
}

} // namespace CausalConv1d

#endif // CAUSAL_CONV1D_UPDATE_H
