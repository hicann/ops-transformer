/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file ffn_wb_gather_out_all.h
 * \brief
 */

#ifndef OP_KERNEL_FFN_WB_GATHER_OUT_ALL_H
#define OP_KERNEL_FFN_WB_GATHER_OUT_ALL_H

#include "kernel_operator.h"
#include "ffn_wb_common.h"

#include "ffn_wb_get_schedule_context.h"

namespace FfnWbBatching {

using namespace AscendC;

template <bool isScanFlag = false>
class KernelFfnWBGatherOutAll {
public:
    __aicore__ inline KernelFfnWBGatherOutAll() {}
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

        int64_t ubAvailable =
            contextInfo->ubSize -
            (BUFFER_NUM * PER_LOOP_ROWS * sizeof(int32_t) * VAR_NUM + BUFFER_NUM * PER_LOOP_ROWS * sizeof(int32_t) +
             sessionNumBlockAlign_ * sizeof(int32_t) * BUFFER_NUM + PER_LOOP_ROWS * BLOCK_SIZE * BUFFER_NUM);

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

        int64_t blockBufferSize =
            maxBlockSize_ * tokenDtypeSize_ + (contextInfo_->tokenDtype == TOKEN_KIND_TWO ? BLOCK_BYTES : 0);
        pipe->InitBuffer(inQueueX_, BUFFER_NUM, blockBufferSize);

        // PER_LOOP_ROWS 为长度 包含 额外5 + 1个输出;
        pipe->InitBuffer(outQueALL_, BUFFER_NUM,
                         PER_LOOP_ROWS * sizeof(int32_t) * VAR_NUM + PER_LOOP_ROWS * BLOCK_SIZE);

        // 将gm gather_idx 一段长度 放到 UB 空间的
        pipe->InitBuffer(expertIdxQue_, BUFFER_NUM, PER_LOOP_ROWS * sizeof(int32_t));

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
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
                int64_t gatherIdx = expertIdxLocal.GetValue(indicesIndex);
                int64_t aIndices = gatherIdx / bskProduct;
                int64_t remainder = gatherIdx - aIndices * bskProduct;
                int64_t bsIndices = remainder / contextInfo_->K;
                int64_t kIndices = remainder - bsIndices * contextInfo_->K;
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
                    int64_t outputOffset = (indicesIndex + currentOuterStart) * contextInfo_->H * tokenDtypeSize_ +
                                           hStart * tokenDtypeSize_;
                    CopyXOut(outputOffset, hSize);
                }
            }

            outQueALL_.EnQue(outAllLocal);
            CopyAllLocalOut(currentOuterStart, curLoopElements);
            expertIdxQue_.FreeTensor(expertIdxLocal);
        }
    }

private:
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

} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_GATHER_OUT_ALL_H
