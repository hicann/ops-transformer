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
 * \file moe_token_permute_masked_select_v3.h
 * \brief
 */
#ifndef MOE_TOKEN_PERMUTE_MASKED_SELECT_V3_H
#define MOE_TOKEN_PERMUTE_MASKED_SELECT_V3_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "moe_sort_base_with_routing_map.h"
#include "moe_token_permute_with_routing_map_common.h"
using namespace AscendC;
#define IS_1_BYTES_TYPE is_same<T, int8_t>::value || is_same<T, uint8_t>::value
#define IS_2_BYTES_TYPE \
    is_same<T, int16_t>::value || is_same<T, uint16_t>::value || is_same<T, half>::value || \
        is_same<T, bfloat16_t>::value
#define IS_4_BYTES_TYPE is_same<T, int32_t>::value || is_same<T, uint32_t>::value || is_same<T, float>::value
#define IS_8_BYTES_TYPE is_same<T, int64_t>::value || is_same<T, uint64_t>::value || is_same<T, double>::value

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int64_t BUFFER_32B_CALNUM = 32 / sizeof(float);
constexpr int32_t HALf_INTERVAL = 2;
constexpr int32_t MASK_LEN = 64;
constexpr int32_t REPEAT_STRIDE_8 = 8;
constexpr int32_t HALF_INTERVAL_THRESHOLD = 128;
constexpr int64_t ONE_BLOCK_NUM = 8;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename, typename>
struct is_same : public false_type {};

template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

__aicore__ inline int32_t AlignUp(int32_t a, int32_t b)
{
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b)
{
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline void VToMTE2Sync()
{
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
}

__aicore__ inline void MTE2ToVSync()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

__aicore__ inline void VToMTE3Sync()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

// 二分累加到2 * offset以内，再累加成offset大小
__aicore__ inline void BinaryAddFunc(LocalTensor<float> tmpBuffer, int32_t hiddensizeLen, int32_t threshold,
                                     int32_t offset)
{
    int32_t totalLen = hiddensizeLen;
    int32_t halfLen = AlignUp(CeilDiv(totalLen, HALf_INTERVAL), BUFFER_32B_CALNUM);
    while (totalLen > threshold) {
        PipeBarrier<PIPE_V>();
        Add(tmpBuffer, tmpBuffer, tmpBuffer[halfLen], totalLen - halfLen);
        PipeBarrier<PIPE_V>();
        totalLen = halfLen;
        halfLen = AlignUp(CeilDiv(totalLen, HALf_INTERVAL), BUFFER_32B_CALNUM);
    }
    PipeBarrier<PIPE_V>();
    Add(tmpBuffer, tmpBuffer, tmpBuffer[offset], totalLen - offset);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumFunc(LocalTensor<float> dstBuffer, LocalTensor<float> tmpBuffer, int32_t hiddensizeLen)
{
    if (hiddensizeLen >= MASK_LEN) { // 二分累加到128以内，再累加成64大小，用blockreducesum
        PipeBarrier<PIPE_V>();
        BinaryAddFunc(tmpBuffer, hiddensizeLen, HALf_INTERVAL * MASK_LEN, MASK_LEN); // 累加成64大小
        PipeBarrier<PIPE_V>();
        // 用BlockReduceSum+WholeReduceSum 把64 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, 1, MASK_LEN, 1, 1, REPEAT_STRIDE_8); // 输出8
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstBuffer, tmpBuffer, REPEAT_STRIDE_8, 1, 1, 1, REPEAT_STRIDE_8); // 输出1
        PipeBarrier<PIPE_V>();
    } else { // 64个以内的元素直接WholeReduceSum成1个
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstBuffer, tmpBuffer, hiddensizeLen, 1, 1, 1, REPEAT_STRIDE_8);
        PipeBarrier<PIPE_V>();
    }
}

namespace AscendC {

constexpr uint32_t SHAPEOUT_SIZE = 2;
constexpr uint32_t BIT_NUM_PER_BYTE = 8;
constexpr uint32_t HEAD_BLOCK_SIZE = 64;
constexpr uint32_t OFFSET_SHIFT_BITS = 3;     // offset偏移量移位输，<<3 等价于 *8
constexpr uint32_t INT64_LENGTH_IN_INT32 = 2; // INT64 相当于 2个int32长
constexpr uint32_t GATHER_RESULT_STRIDE = 8;
constexpr uint32_t DATA_ALIGN = 256;

enum class MaskedSelectMode : int32_t {
    kFull = 0,
    kGatherExternalIndex = 1,
};

template <typename T>
class KernelMaskedSelectV3 {
public:
    __aicore__ inline KernelMaskedSelectV3() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR sortedIndices, GM_ADDR workspace,
                                const MaskedSelectRMTilingData *maskedSelectTilingData, int64_t topK, int64_t hasProb,
                                TPipe *tPipe, MaskedSelectMode mode = MaskedSelectMode::kFull,
                                GM_ADDR externalIndexGm = nullptr, int64_t valueStreamOffset = 0)
    {
        this->pipe = tPipe;
        this->processMode = mode;
        this->valueStreamOffset = valueStreamOffset;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        __gm__ T *globalWorkTensor = (__gm__ T *)((__gm__ uint64_t *)workspace);
        blockIdx = GetBlockIdx();
        this->topK = topK;
        InitForMaskedSelectTilingData(maskedSelectTilingData);

        this->hasProb = hasProb;
        if (blockIdx < this->formerNum) { // 分到大块核的处理
            this->tileLength = this->formertileLength / BUFFER_NUM;
            this->lasttileLength = this->formerlasttileLength / BUFFER_NUM;
            this->tileNum = this->formertileNum * BUFFER_NUM;
            if (hasProb) {
                xGlobal.SetGlobalBuffer((__gm__ T *)x + this->formerLength * blockIdx, this->formerLength);
            }
            maskGlobal.SetGlobalBuffer((__gm__ uint8_t *)mask + this->formerLength * blockIdx, this->formerLength);
            workGlobal.SetGlobalBuffer(globalWorkTensor + this->formerLength * blockIdx, this->formerLength);
        } else { // 分到小块核的处理，需要处理的数据量比大核少alignNum个
            this->tileLength = this->tailtileLength / BUFFER_NUM;
            this->lasttileLength = this->taillasttileLength / BUFFER_NUM;
            this->tileNum = this->tailtileNum * BUFFER_NUM;
            if (hasProb) {
                xGlobal.SetGlobalBuffer((__gm__ T *)x + this->formerLength * this->formerNum +
                                            this->tailLength * (blockIdx - this->formerNum),
                                        this->tailLength);
            }
            maskGlobal.SetGlobalBuffer((__gm__ uint8_t *)mask + this->formerLength * this->formerNum +
                                           this->tailLength * (blockIdx - this->formerNum),
                                       this->tailLength);
            workGlobal.SetGlobalBuffer(globalWorkTensor + this->formerLength * this->formerNum +
                                           this->tailLength * (blockIdx - this->formerNum),
                                       this->tailLength);
        }
        uint64_t alignNum = DATA_ALIGN / sizeof(half); // 256/<8>=32
        tileLengthAlign = (tileLength + alignNum - 1) / alignNum * alignNum;

        offsetGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace, numBlocks);
        indicesWorkspaceGm.SetGlobalBuffer((__gm__ int32_t *)workspace, tokenNum * topK);
        if (externalIndexGm != nullptr) {
            uint64_t totalMaskLen = this->formerNum * this->formerLength + this->tailNum * this->tailLength;
            externalIndexGlobal.SetGlobalBuffer((__gm__ int32_t *)externalIndexGm, totalMaskLen);
        }
        if (valueStreamOffset > 0) {
            valueStreamGm.SetGlobalBuffer((__gm__ int32_t *)workspace + valueStreamOffset, tokenNum * topK);
        }
        SetPipeBuffer(hasProb);
        indexLocal = indexBuf.Get<int32_t>();
        offsetLocal = offsetBuf.Get<int32_t>();
        offsetLocal.SetSize((numBlocks + ONE_BLOCK_NUM - 1) / ONE_BLOCK_NUM * ONE_BLOCK_NUM);
    }

    __aicore__ inline void Process(GM_ADDR y, GM_ADDR sortedIndices, int32_t &actualOutTokens)
    {
        if (processMode == MaskedSelectMode::kGatherExternalIndex) {
            ProcessGatherExternalIndex(y, sortedIndices, actualOutTokens);
            return;
        }

        if (this->blockIdx < needCoreNum) {
            int32_t loopCount = this->tileNum;
            PipeBarrier<PIPE_V>();
            Duplicate(offsetLocal, 0, ONE_BLOCK_NUM);

            PipeBarrier<PIPE_V>();

            maskHalfLocal = outQueueIndex.AllocTensor<half>();
            if (tokenNum == tileLength) {
                PipeBarrier<PIPE_V>();
                MoeTokenPermute::ArithProgressionSupportInt32<int32_t>(indexLocal, static_cast<int32_t>(0),
                                                                       static_cast<int32_t>(1), tokenNum);
                PipeBarrier<PIPE_V>();
            }
            for (int32_t i = 0; i < loopCount; ++i) {
                CopyInMask(i);
                ComputeOffset(i);
            }
            outQueueIndex.FreeTensor(maskHalfLocal);
            VToMTE3Sync();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(offsetGlobal[blockIdx], offsetLocal, copyParams);
        }
        SyncAll();
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numBlocks * sizeof(int32_t)), 0, 0, 0};
        uint64_t ind = 0;
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(offsetLocal, offsetGlobal, copyParams, padParams);
        PipeBarrier<PIPE_ALL>();

        actualOutTokens = 0;
        auto blockIdInNeedCore = blockIdx < needCoreNum ? blockIdx : needCoreNum;
        for (int32_t i = 0; i < blockIdInNeedCore; i++) {
            ind += offsetLocal.GetValue(i);
        }
        actualOutTokens = ind;
        for (int32_t i = blockIdInNeedCore; i < needCoreNum; i++) {
            actualOutTokens += offsetLocal.GetValue(i);
        }

        int32_t alignedTotalLength = static_cast<int32_t>(topK * tokenNum);
        if (actualOutTokens != alignedTotalLength) {
            return;
        }

        if (this->blockIdx < needCoreNum) {
            this->outOffset = 0;
            if (hasProb) {
                yGlobal.SetGlobalBuffer((__gm__ T *)y + ind);
            }

            indexGlobal.SetGlobalBuffer((__gm__ int32_t *)sortedIndices + ind);
            PipeBarrier<PIPE_ALL>();

            for (int32_t i = 0; i < tileNum; ++i) {
                CopyIn(i);
                Compute(i);
                CopyIndex2WorkSpace();
                if (hasProb) {
                    CopyOut2GM();
                }
                outOffset += rsvdCnt;
            }
        }
    }

    __aicore__ inline void ProcessGatherExternalIndex(GM_ADDR y, GM_ADDR sortedIndices, int32_t &actualOutTokens)
    {
        (void)sortedIndices;
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numBlocks * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(offsetLocal, offsetGlobal, copyParams, padParams);
        PipeBarrier<PIPE_ALL>();

        uint64_t ind = 0;
        auto blockIdInNeedCore = blockIdx < needCoreNum ? blockIdx : needCoreNum;
        for (int32_t i = 0; i < blockIdInNeedCore; i++) {
            ind += offsetLocal.GetValue(i);
        }
        actualOutTokens = static_cast<int32_t>(ind);
        for (int32_t i = blockIdInNeedCore; i < needCoreNum; i++) {
            actualOutTokens += static_cast<int32_t>(offsetLocal.GetValue(i));
        }

        if (this->blockIdx < needCoreNum) {
            this->outOffset = 0;
            if (hasProb) {
                yGlobal.SetGlobalBuffer((__gm__ T *)y + ind);
            }
            indexGlobal.SetGlobalBuffer((__gm__ int32_t *)valueStreamGm.GetPhyAddr() + ind);
            PipeBarrier<PIPE_ALL>();

            for (int32_t i = 0; i < tileNum; ++i) {
                CopyInExternalIndex(i);
                CopyIn(i);
                MTE2ToVSync();
                ComputeExternalIndex(i);
                CopyIndex2WorkSpace();
                if (hasProb) {
                    CopyOut2GM();
                }
                outOffset += rsvdCnt;
            }
        }
    }

private:
    __aicore__ inline void InitForMaskedSelectTilingData(const MaskedSelectRMTilingData *maskedSelectTilingData)
    {
        this->numBlocks = maskedSelectTilingData->needCoreNum;
        this->formerNum = maskedSelectTilingData->formerNum;
        this->formerLength = maskedSelectTilingData->formerLength;
        this->formertileNum = maskedSelectTilingData->formertileNum;
        this->formertileLength = maskedSelectTilingData->formertileLength;
        this->formerlasttileLength = maskedSelectTilingData->formerlasttileLength;
        this->needCoreNum = maskedSelectTilingData->needCoreNum;
        this->tailNum = maskedSelectTilingData->tailNum;
        this->tailLength = maskedSelectTilingData->tailLength;
        this->tailtileNum = maskedSelectTilingData->tailtileNum;
        this->tailtileLength = maskedSelectTilingData->tailtileLength;
        this->taillasttileLength = maskedSelectTilingData->taillasttileLength;
        this->tokenNum = maskedSelectTilingData->tokenNum;
    }

    __aicore__ inline void SetPipeBuffer(int64_t hasProb)
    {
        if (hasProb) {
            pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLengthAlign * sizeof(T));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLengthAlign * sizeof(T));
        }
        pipe->InitBuffer(inQueueMask, BUFFER_NUM, this->tileLengthAlign * sizeof(uint8_t));
        pipe->InitBuffer(outQueueIndex, BUFFER_NUM, this->tileLengthAlign * sizeof(int32_t));
        pipe->InitBuffer(offsetBuf, MASK_LEN);

        pipe->InitBuffer(sumBuf, BLOCK_SIZE);

        pipe->InitBuffer(maskCastBuf, this->tileLengthAlign * sizeof(float));
        pipe->InitBuffer(bitMaskBuf, this->tileLengthAlign);
        pipe->InitBuffer(indexBuf, this->tileLengthAlign * sizeof(int32_t));

        if constexpr (IS_1_BYTES_TYPE) {
            pipe->InitBuffer(xCastBuf, this->tileLengthAlign * sizeof(half));
            pipe->InitBuffer(yCastBuf, this->tileLengthAlign * sizeof(half));
        }
    }

    __aicore__ inline void CopyInExternalIndex(int32_t progress)
    {
        uint64_t ind = progress * this->tileLength;
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            length = this->lasttileLength;
        }
        uint64_t regionOffset = 0;
        if (blockIdx < this->formerNum) {
            regionOffset = this->formerLength * blockIdx + ind;
        } else {
            regionOffset = this->formerLength * this->formerNum + this->tailLength * (blockIdx - this->formerNum) + ind;
        }
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        VToMTE2Sync();
        DataCopyPad(indexLocal, externalIndexGlobal[regionOffset], copyParams, padParams);
    }

    __aicore__ inline void ComputeExternalIndex(int32_t progress)
    {
        LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();
        LocalTensor<uint8_t> bitMaskLocal = bitMaskBuf.Get<uint8_t>();
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            length = this->lasttileLength;
        }
        GenerateMask(maskLocal, bitMaskLocal, length);
        inQueueMask.FreeTensor(maskLocal);
        if (hasProb) {
            LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
            LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            GatherResult(yLocal, xLocal, bitMaskLocal, length);
            outQueueY.EnQue<T>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }

        LocalTensor<int32_t> indexOutLocal = outQueueIndex.AllocTensor<int32_t>();
        GatherIndexResult(indexOutLocal, indexLocal, bitMaskLocal, length);
        outQueueIndex.EnQue<int32_t>(indexOutLocal);
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        uint64_t ind = progress * this->tileLength;
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            // 最后一个block，最后一个tile
            length = this->lasttileLength;
        }
        if (hasProb) {
            LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGlobal[ind], copyParams, padParams);
            inQueueX.EnQue(xLocal);
        }

        LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(maskLocal, maskGlobal[ind], copyParams, padParams);
        inQueueMask.EnQue(maskLocal);
    }

    __aicore__ inline void CopyInMask(int32_t progress)
    {
        LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
        uint64_t ind = progress * this->tileLength;
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            // 最后一个block，最后一个tile
            length = this->lasttileLength;
        }

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(maskLocal, maskGlobal[ind], copyParams, padParams);

        inQueueMask.EnQue(maskLocal);
    }

    __aicore__ inline void GenerateMask(const LocalTensor<uint8_t> &mask, LocalTensor<uint8_t> &bitMask, uint32_t count)
    {
        LocalTensor<half> maskCastLocal = maskCastBuf.Get<half>();
        Cast(maskCastLocal, mask, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();

        if constexpr (IS_8_BYTES_TYPE) {
            LocalTensor<int16_t> maskCastInt16 = maskCastLocal.template ReinterpretCast<int16_t>();
            LocalTensor<int16_t> maskCastInt16Shift =
                maskCastLocal[this->tileLength].template ReinterpretCast<int16_t>();
            Cast(maskCastInt16, maskCastLocal, RoundMode::CAST_ROUND, this->tileLength);

            ShiftLeft(maskCastInt16Shift, maskCastInt16, static_cast<int16_t>(BIT_NUM_PER_BYTE), this->tileLengthAlign);
            Add(maskCastInt16Shift, maskCastInt16, maskCastInt16Shift, this->tileLength);

            Cast(maskCastLocal, maskCastInt16Shift.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE,
                 this->tileLengthAlign * INT64_LENGTH_IN_INT32);
            CompareScalar(bitMask, maskCastLocal, static_cast<half>(1.0), CMPMODE::EQ,
                          this->tileLength * INT64_LENGTH_IN_INT32);
        } else {
            CompareScalar(bitMask, maskCastLocal, static_cast<half>(1.0), CMPMODE::EQ, this->tileLengthAlign);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherResult(LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                        const LocalTensor<uint8_t> &bitMaskLocal, int32_t count)
    {
        GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = 1;
        params.src0RepeatStride = GATHER_RESULT_STRIDE;
        params.src1RepeatStride = 1;

        if constexpr (IS_8_BYTES_TYPE) {
            uint32_t mask = count * INT64_LENGTH_IN_INT32;
            LocalTensor<uint32_t> bitMask = bitMaskLocal.ReinterpretCast<uint32_t>();
            LocalTensor<int32_t> dstCastLocal = dstLocal.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> srcCastLocal = srcLocal.template ReinterpretCast<int32_t>();
            GatherMask(dstCastLocal, srcCastLocal, bitMask, true, mask, params, rsvdCnt);
        } else if constexpr (IS_4_BYTES_TYPE) {
            uint32_t mask = count;
            LocalTensor<uint32_t> bitMask = bitMaskLocal.ReinterpretCast<uint32_t>();
            GatherMask(dstLocal, srcLocal, bitMask, true, mask, params, rsvdCnt);
        } else if constexpr (IS_2_BYTES_TYPE) {
            uint32_t mask = count;
            LocalTensor<uint16_t> bitMask = bitMaskLocal.ReinterpretCast<uint16_t>();
            GatherMask(dstLocal, srcLocal, bitMask, true, mask, params, rsvdCnt); // rsvdCnt 最终有效元素个数
        } else {
            uint32_t mask = count;
            LocalTensor<half> xCastLocal = xCastBuf.Get<half>();
            LocalTensor<half> yCastLocal = yCastBuf.Get<half>();
            Duplicate(xCastLocal, static_cast<half>(0), static_cast<int32_t>(this->tileLength));
            Cast(xCastLocal, srcLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint16_t> bitMask = bitMaskLocal.ReinterpretCast<uint16_t>();
            GatherMask(yCastLocal, xCastLocal, bitMask, true, mask, params, rsvdCnt);
            Cast(dstLocal, yCastLocal, RoundMode::CAST_NONE, rsvdCnt);
        }
    }
    __aicore__ inline void GatherIndexResult(LocalTensor<int32_t> &dstLocal, const LocalTensor<int32_t> &srcLocal,
                                             const LocalTensor<uint8_t> &bitMaskLocal, int32_t count)
    {
        GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = 1;
        params.src0RepeatStride = GATHER_RESULT_STRIDE;
        params.src1RepeatStride = 1;
        uint32_t mask = count;
        LocalTensor<uint32_t> bitMask = bitMaskLocal.ReinterpretCast<uint32_t>();
        GatherMask(dstLocal, srcLocal, bitMask, true, mask, params, rsvdCnt);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        if (tokenNum != tileLength) {
            PipeBarrier<PIPE_V>();
            int32_t startId = ((progress + 1) * tileLength - 1) % tokenNum;
            int32_t startIndex = startId - tileLength + 1;
            MoeTokenPermute::ArithProgressionSupportInt32<int32_t>(indexLocal, static_cast<int32_t>(startIndex),
                                                                   static_cast<int32_t>(1), tileLength);
            int32_t zeroId = ((progress)*tileLength) % tokenNum;
            int32_t onceDiff = zeroId - startId;
            if (onceDiff != 1 - tileLength) {
                int32_t addLen = tokenNum - zeroId;
                PipeBarrier<PIPE_V>();
                Adds(indexLocal, indexLocal, static_cast<int32_t>(zeroId - startIndex), addLen);
            }
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();
        LocalTensor<uint8_t> bitMaskLocal = bitMaskBuf.Get<uint8_t>(); // GYW  DeQue 和 GET区别？
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            length = this->lasttileLength;
        }
        GenerateMask(maskLocal, bitMaskLocal, length);
        inQueueMask.FreeTensor(maskLocal);
        if (hasProb) {
            LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
            LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            GatherResult(yLocal, xLocal, bitMaskLocal, length);
            outQueueY.EnQue<T>(yLocal);
            inQueueX.FreeTensor(xLocal);
        }

        LocalTensor<int32_t> indexOutLocal = outQueueIndex.AllocTensor<int32_t>();
        GatherIndexResult(indexOutLocal, indexLocal, bitMaskLocal, length);
        outQueueIndex.EnQue<int32_t>(indexOutLocal);
    }

    __aicore__ inline void ComputeOffset(int32_t progress)
    {
        LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        LocalTensor<int32_t> sumInt32 = sumLocal.template ReinterpretCast<int32_t>();
        LocalTensor<float> maskCastLocal = maskCastBuf.Get<float>();

        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            length = this->lasttileLength;
        }
        PipeBarrier<PIPE_V>();
        Cast(maskHalfLocal, maskLocal, RoundMode::CAST_NONE, length);
        inQueueMask.FreeTensor(maskLocal);
        PipeBarrier<PIPE_V>();
        Cast(maskCastLocal, maskHalfLocal, RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();
        ReduceSumFunc(sumLocal, maskCastLocal, length);
        PipeBarrier<PIPE_V>();

        Cast(sumInt32, sumLocal, RoundMode::CAST_ROUND, 1);
        PipeBarrier<PIPE_V>();
        Add(offsetLocal, offsetLocal, sumInt32, 1);
    }
    __aicore__ inline void DataCopyPadDoubleWord(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
                                                 int64_t count)
    {
        GlobalTensor<int32_t> srcCastGlobal;
        srcCastGlobal.SetGlobalBuffer((__gm__ int32_t *)srcGlobal.GetPhyAddr(),
                                      count * INT64_LENGTH_IN_INT32); // 将GM 中 64 转成 32 * 2

        LocalTensor<int32_t> dstCastLocal = dstLocal.template ReinterpretCast<int32_t>(); // 将 ue转 int32

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * INT64_LENGTH_IN_INT32 * sizeof(int32_t)), 0, 0,
                                     0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(dstCastLocal, srcCastGlobal, copyParams, padParams);
    }

    __aicore__ inline void DataCopyPadDoubleWord(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
                                                 int64_t count)
    {
        GlobalTensor<int32_t> dstCastGlobal;
        dstCastGlobal.SetGlobalBuffer((__gm__ int32_t *)dstGlobal.GetPhyAddr(), count * INT64_LENGTH_IN_INT32);

        LocalTensor<int32_t> srcCastLocal = srcLocal.template ReinterpretCast<int32_t>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * INT64_LENGTH_IN_INT32 * sizeof(int32_t)), 0, 0,
                                     0};
        DataCopyPad(dstCastGlobal, srcCastLocal, copyParams);
    }

    __aicore__ inline void CopyOut2GM()
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(rsvdCnt * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGlobal[outOffset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyIndex2WorkSpace()
    {
        LocalTensor<int32_t> yLocal = outQueueIndex.DeQue<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(rsvdCnt * sizeof(uint32_t)), 0, 0, 0};
        DataCopyPad(indexGlobal[outOffset], yLocal, copyParams);
        outQueueIndex.FreeTensor(yLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueMask;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueIndex;
    TBuf<TPosition::VECCALC> maskCastBuf;
    TBuf<TPosition::VECCALC> bitMaskBuf;
    TBuf<TPosition::VECCALC> xCastBuf;
    TBuf<TPosition::VECCALC> yCastBuf;
    TBuf<TPosition::VECCALC> offsetBuf;
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> indexBuf;

    GlobalTensor<T> xGlobal;
    GlobalTensor<T> yGlobal;
    GlobalTensor<uint64_t> shapeoutGlobal;
    GlobalTensor<uint8_t> maskGlobal;
    GlobalTensor<T> workGlobal;
    GlobalTensor<int32_t> offsetGlobal;
    GlobalTensor<int32_t> indicesWorkspaceGm;
    GlobalTensor<int32_t> externalIndexGlobal;
    GlobalTensor<int32_t> valueStreamGm;
    LocalTensor<int32_t> offsetLocal;
    LocalTensor<half> maskHalfLocal;
    LocalTensor<int32_t> indexLocal;
    GlobalTensor<int32_t> indexGlobal;
    uint64_t needCoreNum;
    // 输入
    uint64_t numBlocks;
    uint64_t formerNum;
    uint64_t formerLength;
    uint64_t formertileNum;
    uint64_t formertileLength;
    uint64_t formerlasttileLength;
    uint64_t tailNum;
    uint64_t tailLength;
    uint64_t tailtileNum;
    uint64_t tailtileLength;
    uint64_t taillasttileLength;
    uint64_t tokenNum;
    int64_t topK;
    uint64_t tileLengthAlign;
    // 本block/核的
    uint64_t tileNum;
    uint64_t tileLength;
    uint64_t lasttileLength;
    int64_t hasProb;
    MaskedSelectMode processMode = MaskedSelectMode::kFull;
    int64_t valueStreamOffset = 0;
    uint64_t rsvdCnt = 0;
    uint64_t outOffset = 0;
    uint32_t blockIdx = 0;
};
} // namespace AscendC

#endif // MOE_TOKEN_PERMUTE_MASKED_SELECT_V3_H
