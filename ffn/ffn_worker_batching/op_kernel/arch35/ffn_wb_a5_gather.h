/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_a5_gather.h
 * \brief arch35(A5) 按 gather_idx 取行搬运:y / session_ids / micro_batch_ids / token_ids /
 *        expert_offsets / dynamic_scale。
 *
 * 逐行取 idx 到标量再算源地址、每行走 DataCopyPad 搬 H 字节 —— 这是 A5 上同类"按索引取行"
 * 的惯用形态(同仓 moe_v3_gather_out.h 的 arch35 实现同构):行数据是大块搬运,应交给 MTE;
 * 而源地址由数据决定,索引必须落到标量才能喂给搬运指令。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_GATHER_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_GATHER_H
#include "ffn_wb_a5_context.h"

namespace FfnWbBatchingArch35 {
using namespace AscendC;

// 索引换算的向量分道:每个中间量独占一道,不复用、不自我覆盖(PER_LOOP_ROWS=128,每道 512B)。
constexpr int64_t IDX_LANE_SRC = 0; // gatherIdx(fp32)
constexpr int64_t IDX_LANE_QA = 1;  // gatherIdx / bskProduct
constexpr int64_t IDX_LANE_AF = 2;  // float(aIdx)
constexpr int64_t IDX_LANE_REM = 3; // gatherIdx - aIdx*bskProduct
constexpr int64_t IDX_LANE_QB = 4;  // rem / K
constexpr int64_t IDX_LANE_BF = 5;  // float(bsIdx)
constexpr int64_t IDX_LANE_KF = 6;  // rem - bsIdx*K
constexpr int64_t IDX_LANE_AI = 0;  // aIdx(int32,独立缓冲)
constexpr int64_t IDX_LANE_BI = 1;  // bsIdx
constexpr int64_t IDX_LANE_KI = 2;  // kIdx
constexpr int64_t IDX_F_LANES = 7;  // srcF/qaF/aF/remF/qbF/bF/kF
constexpr int64_t IDX_I_LANES = 3;  // aIdx/bsIdx/kIdx

// ===================== step 2:gather 搬运(原 A2 ffn_wb_gather_out_all.h) =====================
template <bool isScanFlag = false>
class FfnWbA5Gather {
public:
    __aicore__ inline FfnWbA5Gather() {}
    __aicore__ inline void Init(GM_ADDR expertid_idx, GM_ADDR y, GM_ADDR session_ids, GM_ADDR micro_batch_ids,
                                GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale,
                                const ScheduleContextInfo *contextInfo, TPipe *pipe, uint32_t usedCoreNum)
    {
        contextInfo_ = contextInfo;
        curMicroBatchID = contextInfo_->curMicroBatchID;
        BsKPaddingCount = contextInfo_->BsKPaddingCount;
        int64_t useCore = contextInfo_->coreNum - usedCoreNum;

        tokenDtypeSize_ = (contextInfo_->tokenDtype == NUM_TWO) ? sizeof(int8_t) : sizeof(half); // attr 2: int8

        sessionNumBlockAlign_ = Align(contextInfo_->A, sizeof(int32_t));
        int64_t validGatherIdxLength = contextInfo_->validGatherIdxLength;

        // ⚠️ 逐项扣除本类**所有**会 InitBuffer 的缓冲。少扣一项会让 maxBlockSize_ 算大,
        // inQueueX_ 随即申请越界、分配失败,输出成为垃圾。新增缓冲必须同步加进这个式子。
        int64_t ubAvailable =
            contextInfo->ubSize -
            (BUFFER_NUM * PER_LOOP_ROWS * sizeof(int32_t) * VAR_NUM + BUFFER_NUM * PER_LOOP_ROWS * sizeof(int32_t) +
             sessionNumBlockAlign_ * sizeof(int32_t) * BUFFER_NUM + PER_LOOP_ROWS * BLOCK_SIZE * BUFFER_NUM +
             PER_LOOP_ROWS * (sizeof(float) * IDX_F_LANES + sizeof(int32_t) * IDX_I_LANES));

        int64_t maxTokenSize = ubAvailable / BUFFER_NUM;
        maxBlockSize_ = maxTokenSize - (contextInfo_->tokenDtype == TOKEN_KIND_TWO ? BLOCK_BYTES : 0);
        maxBlockSize_ = (maxBlockSize_ / BLOCK_BYTES * BLOCK_BYTES) / tokenDtypeSize_;
        hBlocks_ = (contextInfo_->H + maxBlockSize_ - 1) / maxBlockSize_;
        lastHBlockSize_ = contextInfo_->H - (hBlocks_ - 1) * maxBlockSize_;

        int64_t blockIdx = GetBlockIdx();
        int64_t perCoreRows = CeilDiv(validGatherIdxLength, useCore);
        needCoreNum_ = perCoreRows == 0 ? 0 : CeilDiv(validGatherIdxLength, perCoreRows);
        int64_t lastCoreRows = validGatherIdxLength - perCoreRows * (needCoreNum_ - 1);

        if (blockIdx == needCoreNum_ - 1) {
            lastLoopRows_ = lastCoreRows - (CeilDiv(lastCoreRows, PER_LOOP_ROWS) - 1) * PER_LOOP_ROWS;
            rowLoops_ = (lastCoreRows + PER_LOOP_ROWS - 1) / PER_LOOP_ROWS;
        } else {
            lastLoopRows_ = perCoreRows - (CeilDiv(perCoreRows, PER_LOOP_ROWS) - 1) * PER_LOOP_ROWS;
            rowLoops_ = (perCoreRows + PER_LOOP_ROWS - 1) / PER_LOOP_ROWS;
        }
        uint64_t SplitY = perCoreRows * contextInfo_->H * tokenDtypeSize_;

        GM_ADDR tokenDataBufAddr = reinterpret_cast<GM_ADDR>(contextInfo_->bufferPtr.tokenDataBuf);
        GM_ADDR sessionIdsBufAddr = reinterpret_cast<GM_ADDR>(contextInfo_->bufferPtr.sessionIdsBuf);
        GM_ADDR microBatchIdsBufAddr = reinterpret_cast<GM_ADDR>(contextInfo_->bufferPtr.microBatchIdsBuf);

        tokenDataBufGm_.SetGlobalBuffer((__gm__ int8_t *)tokenDataBufAddr);

        // 排序的后的  对应gather_index
        expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertid_idx + blockIdx * perCoreRows);

        sessionIdsInGm_.SetGlobalBuffer((__gm__ int32_t *)sessionIdsBufAddr, contextInfo_->A);
        microBatchIdsInGm_.SetGlobalBuffer((__gm__ int32_t *)microBatchIdsBufAddr, contextInfo_->A);

        // 输出空间
        yOutGm_.SetGlobalBuffer((__gm__ int8_t *)y + blockIdx * SplitY);

        sessionIdsOutGm_.SetGlobalBuffer((__gm__ int32_t *)session_ids + blockIdx * perCoreRows);
        microBatchIdsOutGm_.SetGlobalBuffer((__gm__ int32_t *)micro_batch_ids + blockIdx * perCoreRows);
        tokenIdsOutGm_.SetGlobalBuffer((__gm__ int32_t *)token_ids + blockIdx * perCoreRows);
        expertOffsetsOutGm_.SetGlobalBuffer((__gm__ int32_t *)expert_offsets + blockIdx * perCoreRows);
        if (contextInfo_->tokenDtype == TOKEN_KIND_TWO) {
            dynamicScaleOutGm_.SetGlobalBuffer((__gm__ float *)dynamic_scale + blockIdx * perCoreRows);
        }

        InitBuffers(pipe);
    }

    // 各缓冲的开辟:块大小已由上面按可用 UB 逐项扣除算好,这里只做分配。
    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        int64_t blockBufferSize =
            maxBlockSize_ * tokenDtypeSize_ + (contextInfo_->tokenDtype == TOKEN_KIND_TWO ? BLOCK_BYTES : 0);
        pipe->InitBuffer(inQueueX_, BUFFER_NUM, blockBufferSize);

        // PER_LOOP_ROWS 为长度 包含 额外5 + 1个输出;
        pipe->InitBuffer(outQueALL_, BUFFER_NUM,
                         PER_LOOP_ROWS * sizeof(int32_t) * VAR_NUM + PER_LOOP_ROWS * BLOCK_SIZE);

        // 将gm gather_idx 一段长度 放到 UB 空间的
        pipe->InitBuffer(expertIdxQue_, BUFFER_NUM, PER_LOOP_ROWS * sizeof(int32_t));
        // 索引换算的向量分道(已计入上面的 ubAvailable)
        pipe->InitBuffer(idxCalcBuf_, PER_LOOP_ROWS * sizeof(float) * IDX_F_LANES);
        pipe->InitBuffer(idxIntBuf_, PER_LOOP_ROWS * sizeof(int32_t) * IDX_I_LANES);

        // 将gm buf 放到 UB 空间的
        if constexpr (isScanFlag == false) {
            pipe->InitBuffer(tmpBuffer_, sessionNumBlockAlign_ * sizeof(int32_t) * BUFFER_NUM);
        }
    }

    __aicore__ inline void CopyInIds()
    {
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(contextInfo_->A * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams1{false, 0, 0, 0};
        sessionIdsLocal_ = tmpBuffer_.Get<int32_t>();
        microBatchIdsLocal_ = sessionIdsLocal_[sessionNumBlockAlign_];

        DataCopyPad(sessionIdsLocal_, sessionIdsInGm_, copyParams1, padParams1);
        DataCopyPad(microBatchIdsLocal_, microBatchIdsInGm_, copyParams1, padParams1);
    }

    __aicore__ inline void CopyInExpertIdx(int32_t expertIdxOffset, int32_t curRows)
    {
        LocalTensor<int32_t> expertIdxLocal = expertIdxQue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curRows * sizeof(int32_t)), 0, 0,
                                     0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(expertIdxLocal, expertIdxGm_[expertIdxOffset], copyParams, padParams);
        expertIdxQue_.EnQue(expertIdxLocal);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= needCoreNum_) {
            return;
        }

        int64_t bskProduct = contextInfo_->BS * contextInfo_->K;
        if constexpr (isScanFlag == false) {
            CopyInIds();
        } else {
            bskProduct = contextInfo_->BS * contextInfo_->K + BsKPaddingCount;
        }

        int64_t curLoopElements = PER_LOOP_ROWS;
        int64_t strideSession = contextInfo_->M * contextInfo_->BS * contextInfo_->K * contextInfo_->HS;
        int64_t strideMicroBatch = contextInfo_->BS * contextInfo_->K * contextInfo_->HS;
        int64_t strideBs = contextInfo_->K * contextInfo_->HS;
        int64_t strideK = contextInfo_->HS;

        for (int64_t i = 0; i < rowLoops_; i++) {
            int64_t currentOuterStart = i * PER_LOOP_ROWS;
            if (i == rowLoops_ - 1) {
                curLoopElements = lastLoopRows_;
            }

            CopyInExpertIdx(currentOuterStart, curLoopElements);
            LocalTensor<int32_t> expertIdxLocal = expertIdxQue_.DeQue<int32_t>();
            LocalTensor<int32_t> outAllLocal = outQueALL_.AllocTensor<int32_t>();
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            // 下标换算整块向量化,结果落在 aIdxAll / bsIdxAll / kIdxAll 三路分道
            DecomposeIndices(expertIdxLocal, bskProduct, curLoopElements);
            LocalTensor<int32_t> ints = idxIntBuf_.Get<int32_t>();
            LocalTensor<int32_t> aIdxAll = ints[PER_LOOP_ROWS * IDX_LANE_AI];
            LocalTensor<int32_t> bsIdxAll = ints[PER_LOOP_ROWS * IDX_LANE_BI];
            LocalTensor<int32_t> kIdxAll = ints[PER_LOOP_ROWS * IDX_LANE_KI];

            EmitTokens(outAllLocal, curLoopElements, currentOuterStart, strideSession, strideMicroBatch, strideBs,
                       strideK, aIdxAll, bsIdxAll, kIdxAll);

            outQueALL_.EnQue(outAllLocal);
            CopyAllLocalOut(currentOuterStart, curLoopElements);
            expertIdxQue_.FreeTensor(expertIdxLocal);
        }
    }

    // 逐 token:从三路分道取出 (a, bs, k),算出源地址搬 H 字节,并填四个 id 输出。
    __aicore__ inline void EmitTokens(const LocalTensor<int32_t> &outAllLocal, int64_t curLoopElements,
                                      int64_t currentOuterStart, int64_t strideSession, int64_t strideMicroBatch,
                                      int64_t strideBs, int64_t strideK, const LocalTensor<int32_t> &aIdxAll,
                                      const LocalTensor<int32_t> &bsIdxAll, const LocalTensor<int32_t> &kIdxAll)
    {
        for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
            int64_t aIndices = aIdxAll.GetValue(indicesIndex); // 三个下标均取自上面算好的向量分道
            int64_t bsIndices = bsIdxAll.GetValue(indicesIndex);
            int64_t kIndices = kIdxAll.GetValue(indicesIndex);
            int32_t sessionIndices = 0;
            int32_t microbatchIndices = 0;
            if constexpr (isScanFlag == false) {
                sessionIndices = sessionIdsLocal_.GetValue(aIndices);
                microbatchIndices = microBatchIdsLocal_.GetValue(aIndices);
            } else {
                sessionIndices = aIndices;
                microbatchIndices = curMicroBatchID;
            }

            outAllLocal.SetValue(indicesIndex, sessionIndices);
            outAllLocal.SetValue(PER_LOOP_ROWS * VAR_MICRO_BATCH_IDX + indicesIndex, microbatchIndices);
            outAllLocal.SetValue(PER_LOOP_ROWS * VAR_TOKEN_IDX + indicesIndex, bsIndices);
            outAllLocal.SetValue(PER_LOOP_ROWS * VAR_EXPERT_OFFSETS_IDX + indicesIndex, kIndices);

            for (int64_t hBlock = 0; hBlock < hBlocks_; hBlock++) {
                int64_t hStart = hBlock * maxBlockSize_;
                int64_t hSize = (hBlock == hBlocks_ - 1) ? lastHBlockSize_ : maxBlockSize_;
                int64_t globalXOffset = sessionIndices * strideSession + microbatchIndices * strideMicroBatch +
                                        bsIndices * strideBs + kIndices * strideK + hStart;

                bool isLastBlock = (hBlock == hBlocks_ - 1);
                CopyXIn(globalXOffset, hSize, indicesIndex, outAllLocal[PER_LOOP_ROWS * VAR_NUM], isLastBlock);
                int64_t outputOffset =
                    (indicesIndex + currentOuterStart) * contextInfo_->H * tokenDtypeSize_ + hStart * tokenDtypeSize_;
                CopyXOut(outputOffset, hSize);
            }
        }
    }

private:
    // 把扁平 gatherIdx 拆成 (aIdx, bsIdx, kIdx) 三路分道:整块向量运算,
    // 避免逐 token 做两次整数除法。三路结果供下方标量循环逐元素取用。
    __aicore__ inline void DecomposeIndices(const LocalTensor<int32_t> &expertIdxLocal, int64_t bskProduct,
                                            int64_t curLoopElements)
    {
        // ---- 索引换算(向量):把逐 token 的两次整数除法整块算完 ----
        // aIdx = gatherIdx / bskProduct;bsIdx = 余数 / K;kIdx = 余数 % K。
        // gatherIdx < Y <= 2^22,fp32 精确表示 2^24 内整数,商与余数无误差;商恒非负。
        LocalTensor<float> lanes = idxCalcBuf_.Get<float>();
        LocalTensor<float> srcF = lanes[PER_LOOP_ROWS * IDX_LANE_SRC];
        LocalTensor<float> qaF = lanes[PER_LOOP_ROWS * IDX_LANE_QA];
        LocalTensor<float> aF = lanes[PER_LOOP_ROWS * IDX_LANE_AF];
        LocalTensor<float> remF = lanes[PER_LOOP_ROWS * IDX_LANE_REM];
        LocalTensor<float> qbF = lanes[PER_LOOP_ROWS * IDX_LANE_QB];
        LocalTensor<float> bF = lanes[PER_LOOP_ROWS * IDX_LANE_BF];
        LocalTensor<float> kF = lanes[PER_LOOP_ROWS * IDX_LANE_KF];
        LocalTensor<int32_t> ints = idxIntBuf_.Get<int32_t>();
        LocalTensor<int32_t> aIdxAll = ints[PER_LOOP_ROWS * IDX_LANE_AI];
        LocalTensor<int32_t> bsIdxAll = ints[PER_LOOP_ROWS * IDX_LANE_BI];
        LocalTensor<int32_t> kIdxAll = ints[PER_LOOP_ROWS * IDX_LANE_KI];
        // 除法用倒数乘法(硬件无整数向量除法),但 1/n 与乘积各有一次舍入,恰好整除处
        // floor 会掉一档(实测 BS*K=189 时 a 偏 -1、bs 偏 +BS,k 因两处偏移抵消反而不变)。
        // 故每次除法后都按余数做一次 ±1 修正:余数是整数且落在 (-n, 2n),
        // 0/1 指示可以纯算术拿到,不需要比较掩码与 Select:
        //   hi = min(max(rem - n + 1, 0), 1) —— rem >= n 时为 1
        //   lo = min(max(-rem, 0), 1)        —— rem <  0 时为 1
        // 修正量 corr = hi - lo ∈ {-1, 0, 1}。全程整数值,fp32 精确(gatherIdx < 2^24)。
        const float bskF = static_cast<float>(bskProduct);
        const float kNumF = static_cast<float>(contextInfo_->K);
        Cast(srcF, expertIdxLocal, RoundMode::CAST_ROUND, curLoopElements);
        PipeBarrier<PIPE_V>();
        Muls(qaF, srcF, static_cast<float>(1.0f / bskF), curLoopElements);
        PipeBarrier<PIPE_V>();
        Cast(aIdxAll, qaF, RoundMode::CAST_FLOOR, curLoopElements);
        PipeBarrier<PIPE_V>();
        Cast(aF, aIdxAll, RoundMode::CAST_ROUND, curLoopElements);
        PipeBarrier<PIPE_V>();
        Muls(remF, aF, -bskF, curLoopElements);
        PipeBarrier<PIPE_V>();
        Add(remF, remF, srcF, curLoopElements); // rem = gatherIdx - aIdx*bskProduct
        PipeBarrier<PIPE_V>();
        CorrectQuotient(aF, remF, qaF, qbF, bskF, curLoopElements);
        Cast(aIdxAll, aF, RoundMode::CAST_RINT, curLoopElements);
        PipeBarrier<PIPE_V>();

        Muls(qbF, remF, static_cast<float>(1.0f / kNumF), curLoopElements);
        PipeBarrier<PIPE_V>();
        Cast(bsIdxAll, qbF, RoundMode::CAST_FLOOR, curLoopElements);
        PipeBarrier<PIPE_V>();
        Cast(bF, bsIdxAll, RoundMode::CAST_ROUND, curLoopElements);
        PipeBarrier<PIPE_V>();
        Muls(kF, bF, -kNumF, curLoopElements);
        PipeBarrier<PIPE_V>();
        Add(kF, remF, kF, curLoopElements); // kIdx = rem - bsIdx*K
        PipeBarrier<PIPE_V>();
        CorrectQuotient(bF, kF, qaF, qbF, kNumF, curLoopElements);
        Cast(bsIdxAll, bF, RoundMode::CAST_RINT, curLoopElements);
        PipeBarrier<PIPE_V>();
        Cast(kIdxAll, kF, RoundMode::CAST_RINT, curLoopElements);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }

    // 倒数乘法得到的商可能差 1,用余数把商与余数一起拉回正确值。
    // quot/rem 传入传出均为整数值的 fp32;t0/t1 是临时分道,调用后内容作废。
    __aicore__ inline void CorrectQuotient(const LocalTensor<float> &quot, const LocalTensor<float> &rem,
                                           const LocalTensor<float> &t0, const LocalTensor<float> &t1, float divisor,
                                           int64_t count)
    {
        Adds(t0, rem, 1.0f - divisor, count); // rem - divisor + 1
        PipeBarrier<PIPE_V>();
        Maxs(t0, t0, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Mins(t0, t0, 1.0f, count); // hi = (rem >= divisor)
        PipeBarrier<PIPE_V>();
        Muls(t1, rem, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Maxs(t1, t1, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Mins(t1, t1, 1.0f, count); // lo = (rem < 0)
        PipeBarrier<PIPE_V>();
        Sub(t0, t0, t1, count); // corr = hi - lo
        PipeBarrier<PIPE_V>();
        Add(quot, t0, quot, count);
        PipeBarrier<PIPE_V>();
        Muls(t1, t0, -divisor, count);
        PipeBarrier<PIPE_V>();
        Add(rem, t1, rem, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyAllLocalOut(int64_t allLocalOffset, int64_t copyLength)
    {
        LocalTensor<int32_t> outAllLocal = outQueALL_.DeQue<int32_t>();

        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(copyLength * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(sessionIdsOutGm_[allLocalOffset], outAllLocal, copyParams2);
        DataCopyPad(microBatchIdsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_MICRO_BATCH_IDX], copyParams2);
        DataCopyPad(tokenIdsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_TOKEN_IDX], copyParams2);
        DataCopyPad(expertOffsetsOutGm_[allLocalOffset], outAllLocal[PER_LOOP_ROWS * VAR_EXPERT_OFFSETS_IDX],
                    copyParams2);

        if (contextInfo_->tokenDtype == TOKEN_KIND_TWO) {
            LocalTensor<int32_t> srcOffsetLocal =
                outAllLocal[PER_LOOP_ROWS * VAR_DYNAMIC_SCALE].template ReinterpretCast<int32_t>();
            LocalTensor<float> dynamicScaleLocalFp32 =
                outAllLocal[PER_LOOP_ROWS * VAR_NUM].template ReinterpretCast<float>();
            ArithProgression<int32_t>(srcOffsetLocal, 0, BLOCK_SIZE, copyLength);
            PipeBarrier<PIPE_V>();
            Gather(dynamicScaleLocalFp32, dynamicScaleLocalFp32, srcOffsetLocal.template ReinterpretCast<uint32_t>(), 0,
                   copyLength);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(dynamicScaleOutGm_[allLocalOffset], dynamicScaleLocalFp32, copyParams2);
        }
        outQueALL_.FreeTensor(outAllLocal);
    }

    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols, int64_t indicesIndex,
                                   const LocalTensor<int32_t> &dynamicScaleLocal, bool isLastBlock)
    {
        LocalTensor<int8_t> xLocal = inQueueX_.AllocTensor<int8_t>();
        uint32_t copySize = curLoopCols * tokenDtypeSize_;
        DataCopyExtParams copyParams0{1, copySize, 0, 0, 0};
        DataCopyPadExtParams<int8_t> padParams0{false, 0, 0, 0};
        DataCopyPad(xLocal, tokenDataBufGm_[xSrcOffset], copyParams0, padParams0);

        if (isLastBlock && contextInfo_->tokenDtype == TOKEN_KIND_TWO) {
            DataCopyExtParams copyParams1{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            // 8 blocks
            LocalTensor<int8_t> dynScaleT = dynamicScaleLocal[indicesIndex * 8].template ReinterpretCast<int8_t>();
            DataCopyPad(dynScaleT, tokenDataBufGm_[xSrcOffset + copySize], copyParams1, padParams0);
        }

        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
    {
        LocalTensor<int8_t> xLocal = inQueueX_.DeQue<int8_t>();

        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(curLoopCols * tokenDtypeSize_), 0, 0, 0};
        DataCopyPad(yOutGm_[xDstOffset], xLocal, copyParams2);

        inQueueX_.FreeTensor(xLocal);
    }

private:
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr int32_t BUFFER_NUM = 2;      // tensor num for each queue
    static constexpr int32_t PER_LOOP_ROWS = 128; // tensor num for each queue
    static constexpr int64_t TOKEN_KIND_TWO = 2;

    static constexpr int32_t VAR_NUM = 5; // session_ids, micro_batch_ids, token_ids, expert_offsets
    static constexpr int32_t VAR_SESSION_IDX = 0;
    static constexpr int32_t VAR_MICRO_BATCH_IDX = 1;
    static constexpr int32_t VAR_TOKEN_IDX = 2;
    static constexpr int32_t VAR_EXPERT_OFFSETS_IDX = 3;
    static constexpr int32_t VAR_DYNAMIC_SCALE = 4;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> inQueueX_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueALL_;
    TQue<TPosition::VECIN, BUFFER_NUM> expertIdxQue_;
    TBuf<TPosition::VECIN> tmpBuffer_;
    TBuf<TPosition::VECCALC> idxCalcBuf_; // fp32 中转分道
    TBuf<TPosition::VECCALC> idxIntBuf_;  // int32 结果分道(类型独立,避免 ReinterpretCast)
    GlobalTensor<int8_t> tokenDataBufGm_; // 这里的token_data_buf 存储的是[A,M,BS,K,HS]
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> sessionIdsInGm_;
    GlobalTensor<int32_t> microBatchIdsInGm_;

    GlobalTensor<int8_t> yOutGm_;
    GlobalTensor<int32_t> sessionIdsOutGm_;
    GlobalTensor<int32_t> microBatchIdsOutGm_;
    GlobalTensor<int32_t> tokenIdsOutGm_;
    GlobalTensor<int32_t> expertOffsetsOutGm_;
    GlobalTensor<float> dynamicScaleOutGm_;

    LocalTensor<int32_t> sessionIdsLocal_;
    LocalTensor<int32_t> microBatchIdsLocal_;

    const ScheduleContextInfo *contextInfo_ = nullptr;

    int64_t needCoreNum_ = 0;
    int64_t lastLoopRows_ = 0;
    int64_t rowLoops_ = 0;
    int64_t tokenDtypeSize_ = 0;
    int64_t sessionNumBlockAlign_ = 0;

    int64_t maxBlockSize_ = 0;
    int64_t hBlocks_ = 0;
    int64_t lastHBlockSize_ = 0;
    uint32_t curMicroBatchID = 0;
    int64_t BsKPaddingCount = 0;
};

} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_A5_GATHER_H
