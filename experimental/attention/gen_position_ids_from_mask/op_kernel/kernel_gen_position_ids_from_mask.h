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
 * @file kernel_gen_position_ids_from_mask.h
 */

#ifndef GEN_POSITION_IDS_FROM_MASK_H_
#define GEN_POSITION_IDS_FROM_MASK_H_

#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

template <typename TIn>
class KernelGenPositionIdsFromMask {
public:
    __aicore__ inline KernelGenPositionIdsFromMask() {}

    __aicore__ inline void Init(GM_ADDR attentionMask, GM_ADDR positionIds,
                                const GenPositionIdsFromMaskTilingData *tiling)
    {
        b_ = tiling->b;
        s_ = tiling->s;
        fillValue_ = tiling->paddingFillValue;
        coreNum_ = tiling->coreNum;

        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t base = b_ / static_cast<int64_t>(coreNum_);
        const int64_t rem = b_ % static_cast<int64_t>(coreNum_);

        if (blockIdx < rem) {
            rowNum_ = base + 1;
            rowBegin_ = blockIdx * (base + 1);
        } else {
            rowNum_ = base;
            rowBegin_ = rem * (base + 1) + (blockIdx - rem) * base;
        }

        if (rowNum_ <= 0) {
            return;
        }

        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TIn *>(attentionMask), b_ * s_);
        posGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(positionIds), b_ * s_);

        // int64 输出按两个 int32 word 处理，相关缓冲区按 2 * TILE_SIZE 分配。
        pipe_.InitBuffer(inputQueue_, 1, TILE_SIZE * sizeof(TIn));
        pipe_.InitBuffer(outputQueue_, 1, TILE_SIZE * sizeof(int64_t));

        pipe_.InitBuffer(mask32Buf_, TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(scanABuf_, TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(scanBBuf_, TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(gatherBuf_, TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(shiftValidBuf_, TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(boolHalfBuf_, TILE_SIZE * sizeof(half));

        pipe_.InitBuffer(indexBuf_, WORD_TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(wordTmpABuf_, WORD_TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(wordTmpBBuf_, WORD_TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(expandedValidBuf_, WORD_TILE_SIZE * sizeof(int32_t));
        pipe_.InitBuffer(fillWordsBuf_, WORD_TILE_SIZE * sizeof(int32_t));

        // fillValueBuf_ 预留 32 字节以满足 UB 对齐要求。
        pipe_.InitBuffer(fillValueBuf_, BLOCK_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (rowNum_ <= 0) {
            return;
        }

        // padding_fill_value 写入 UB 后按两个 int32 word 使用。
        LocalTensor<int64_t> fillValueLocal = fillValueBuf_.Get<int64_t>();
        fillValueLocal.SetValue(0, fillValue_);
        PipeBarrier<PIPE_V>();

        for (int64_t r = 0; r < rowNum_; ++r) {
            ProcessRow(rowBegin_ + r, fillValueLocal);
        }
    }

private:
    __aicore__ inline void ProcessRow(int64_t row, const LocalTensor<int64_t> &fillValueLocal)
    {
        const int64_t rowOffset = row * s_;
        int32_t prefixCarry = 0;

        for (int64_t tileStart = 0; tileStart < s_; tileStart += TILE_SIZE) {
            int64_t remaining = s_ - tileStart;
            uint32_t validLen = static_cast<uint32_t>(remaining > TILE_SIZE ? TILE_SIZE : remaining);

            uint64_t gmOffset = static_cast<uint64_t>(rowOffset + tileStart);

            LocalTensor<TIn> inputLocal = CopyIn(gmOffset, validLen);

            LocalTensor<int32_t> mask32 = mask32Buf_.Get<int32_t>();
            NormalizeMask(mask32, inputLocal, validLen);

            inputQueue_.FreeTensor(inputLocal);

            LocalTensor<int32_t> scan = InclusiveScan(mask32, validLen);

            // 跨 tile 仅保留标量 carry，不进行逐元素 GM 标量访问。
            if (prefixCarry != 0) {
                Adds(scan, scan, prefixCarry, validLen);
                PipeBarrier<PIPE_V>();
            }

            // GetValue 前同步 PIPE_V -> PIPE_S，确保读取最终 scan 结果。
            TEventID eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            prefixCarry = scan.GetValue(validLen - 1);

            // 下一 tile 使用 prefixCarry 前同步 PIPE_S -> PIPE_V。
            TEventID eventIdSToV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(eventIdSToV);
            WaitFlag<HardEvent::S_V>(eventIdSToV);

            BuildAndCopyOutput(scan, mask32, fillValueLocal, gmOffset, validLen);
        }
    }

    __aicore__ inline LocalTensor<TIn> CopyIn(uint64_t gmOffset, uint32_t validLen)
    {
        LocalTensor<TIn> inputLocal = inputQueue_.AllocTensor<TIn>();

        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(validLen * sizeof(TIn)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

        DataCopyPadExtParams<TIn> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                            static_cast<TIn>(0)};

        DataCopyPad(inputLocal, maskGm_[gmOffset], copyParams, padParams);

        inputQueue_.EnQue<TIn>(inputLocal);
        return inputQueue_.DeQue<TIn>();
    }

    __aicore__ inline void NormalizeMask(LocalTensor<int32_t> dst, const LocalTensor<TIn> &src, uint32_t validLen)
    {
        if constexpr (std::is_same<TIn, int32_t>::value) {
            Adds(dst, src, static_cast<int32_t>(0), validLen);
            PipeBarrier<PIPE_V>();
        } else if constexpr (std::is_same<TIn, int64_t>::value) {
            Cast(dst, src, RoundMode::CAST_NONE, validLen);
            PipeBarrier<PIPE_V>();
        } else {
            // A2 不支持 int8_t 直接转 int32_t，BOOL 输入经 half 中转。
            LocalTensor<half> boolHalf = boolHalfBuf_.Get<half>();

            Cast(boolHalf, src, RoundMode::CAST_NONE, validLen);
            PipeBarrier<PIPE_V>();

            Cast(dst, boolHalf, RoundMode::CAST_RINT, validLen);
            PipeBarrier<PIPE_V>();
        }
    }

    // 使用 Hillis-Steele 实现 int32 包含式前缀和。
    __aicore__ inline LocalTensor<int32_t> InclusiveScan(const LocalTensor<int32_t> &mask32, uint32_t validLen)
    {
        LocalTensor<int32_t> scanA = scanABuf_.Get<int32_t>();
        LocalTensor<int32_t> scanB = scanBBuf_.Get<int32_t>();
        LocalTensor<int32_t> gathered = gatherBuf_.Get<int32_t>();
        LocalTensor<int32_t> index = indexBuf_.Get<int32_t>();
        LocalTensor<int32_t> shiftOffset = wordTmpABuf_.Get<int32_t>();
        LocalTensor<int32_t> shiftValid = shiftValidBuf_.Get<int32_t>();

        Adds(scanA, mask32, static_cast<int32_t>(0), validLen);
        PipeBarrier<PIPE_V>();

        CreateVecIndex(index, static_cast<int32_t>(0), validLen);
        PipeBarrier<PIPE_V>();

        for (uint32_t offset = 1; offset < validLen; offset <<= 1) {
            Adds(shiftOffset, index, -static_cast<int32_t>(offset), validLen);
            PipeBarrier<PIPE_V>();

            Maxs(shiftOffset, shiftOffset, static_cast<int32_t>(0), validLen);
            PipeBarrier<PIPE_V>();

            Muls(shiftOffset, shiftOffset, static_cast<int32_t>(sizeof(int32_t)), validLen);
            PipeBarrier<PIPE_V>();

            Gather(gathered, scanA, shiftOffset.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
                   validLen);
            PipeBarrier<PIPE_V>();

            // shiftValid 标记 i >= offset 的元素。
            Adds(shiftValid, index, static_cast<int32_t>(1) - static_cast<int32_t>(offset), validLen);
            PipeBarrier<PIPE_V>();

            Maxs(shiftValid, shiftValid, static_cast<int32_t>(0), validLen);
            PipeBarrier<PIPE_V>();

            Mins(shiftValid, shiftValid, static_cast<int32_t>(1), validLen);
            PipeBarrier<PIPE_V>();

            Mul(gathered, gathered, shiftValid, validLen);
            PipeBarrier<PIPE_V>();

            Add(scanB, scanA, gathered, validLen);
            PipeBarrier<PIPE_V>();

            LocalTensor<int32_t> tmp = scanA;
            scanA = scanB;
            scanB = tmp;
        }

        return scanA;
    }

    __aicore__ inline void BuildAndCopyOutput(const LocalTensor<int32_t> &inclusiveScan,
                                              const LocalTensor<int32_t> &mask32,
                                              const LocalTensor<int64_t> &fillValueLocal, uint64_t gmOffset,
                                              uint32_t validLen)
    {
        LocalTensor<int32_t> position32 = scanBBuf_.Get<int32_t>();

        Adds(position32, inclusiveScan, static_cast<int32_t>(-1), validLen);
        PipeBarrier<PIPE_V>();

        // 直接复用 NormalizeMask 的 0/1 mask，避免尾块的 CompareScalar 对齐限制。

        // 非负 int32 范围的 fill 在 Cast 前处理，其余值走 int64 回退路径。
        const bool useInt32FillFastPath =
            (fillValue_ >= static_cast<int64_t>(0)) && (fillValue_ <= static_cast<int64_t>(2147483647LL));

        if (useInt32FillFastPath) {
            const int32_t fill32 = static_cast<int32_t>(fillValue_);

            // InclusiveScan 后复用 gatherBuf_，避免额外申请 UB。
            LocalTensor<int32_t> fillTmp = gatherBuf_.Get<int32_t>();

            Adds(fillTmp, position32, -fill32, validLen);
            PipeBarrier<PIPE_V>();

            Mul(fillTmp, fillTmp, mask32, validLen);
            PipeBarrier<PIPE_V>();

            Adds(position32, fillTmp, fill32, validLen);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int64_t> outputLocal = outputQueue_.AllocTensor<int64_t>();

        Cast(outputLocal, position32, RoundMode::CAST_NONE, validLen);
        PipeBarrier<PIPE_V>();

        if (!useInt32FillFastPath) {
            SelectInt64Fill(outputLocal, mask32, fillValueLocal, validLen);
        }

        outputQueue_.EnQue<int64_t>(outputLocal);

        LocalTensor<int64_t> outputReady = outputQueue_.DeQue<int64_t>();

        DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(validLen * sizeof(int64_t)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

        DataCopyPad(posGm_[gmOffset], outputReady, copyParams);

        outputQueue_.FreeTensor(outputReady);
    }

    __aicore__ inline void SelectInt64Fill(LocalTensor<int64_t> output64, const LocalTensor<int32_t> &valid32,
                                           const LocalTensor<int64_t> &fillValueLocal, uint32_t validLen)
    {
        // A2 无所需的 int64 Select，按两个 int32 word 处理任意 int64 fill。
        const uint32_t wordCount = validLen * INT64_WORDS;

        LocalTensor<int32_t> outputWords = output64.template ReinterpretCast<int32_t>();

        LocalTensor<int32_t> wordIndex = indexBuf_.Get<int32_t>();

        LocalTensor<int32_t> tmpA = wordTmpABuf_.Get<int32_t>();

        LocalTensor<int32_t> tmpB = wordTmpBBuf_.Get<int32_t>();

        LocalTensor<int32_t> expandedValid = expandedValidBuf_.Get<int32_t>();

        LocalTensor<int32_t> fillWords = fillWordsBuf_.Get<int32_t>();

        LocalTensor<int32_t> fillWordSource = fillValueLocal.template ReinterpretCast<int32_t>();

        CreateVecIndex(wordIndex, static_cast<int32_t>(0), wordCount);
        PipeBarrier<PIPE_V>();

        ShiftRight(tmpA, wordIndex, static_cast<int32_t>(1), wordCount);
        PipeBarrier<PIPE_V>();

        Muls(tmpB, tmpA, static_cast<int32_t>(sizeof(int32_t)), wordCount);
        PipeBarrier<PIPE_V>();

        Gather(expandedValid, valid32, tmpB.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), wordCount);
        PipeBarrier<PIPE_V>();

        Muls(tmpA, tmpA, static_cast<int32_t>(INT64_WORDS), wordCount);
        PipeBarrier<PIPE_V>();

        Sub(tmpB, wordIndex, tmpA, wordCount);
        PipeBarrier<PIPE_V>();

        Muls(tmpB, tmpB, static_cast<int32_t>(sizeof(int32_t)), wordCount);
        PipeBarrier<PIPE_V>();

        Gather(fillWords, fillWordSource, tmpB.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
               wordCount);
        PipeBarrier<PIPE_V>();

        Mul(outputWords, outputWords, expandedValid, wordCount);
        PipeBarrier<PIPE_V>();

        Adds(tmpA, expandedValid, static_cast<int32_t>(-1), wordCount);
        PipeBarrier<PIPE_V>();

        Muls(tmpA, tmpA, static_cast<int32_t>(-1), wordCount);
        PipeBarrier<PIPE_V>();

        Mul(fillWords, fillWords, tmpA, wordCount);
        PipeBarrier<PIPE_V>();

        Add(outputWords, outputWords, fillWords, wordCount);
        PipeBarrier<PIPE_V>();
    }

private:
    static constexpr uint32_t BLOCK_BYTES = 32;
    static constexpr uint32_t TILE_SIZE = 1024;
    static constexpr uint32_t INT64_WORDS = 2;
    static constexpr uint32_t WORD_TILE_SIZE = TILE_SIZE * INT64_WORDS;

    TPipe pipe_;

    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;

    TBuf<QuePosition::VECCALC> mask32Buf_;
    TBuf<QuePosition::VECCALC> scanABuf_;
    TBuf<QuePosition::VECCALC> scanBBuf_;
    TBuf<QuePosition::VECCALC> gatherBuf_;
    TBuf<QuePosition::VECCALC> shiftValidBuf_;
    TBuf<QuePosition::VECCALC> boolHalfBuf_;

    TBuf<QuePosition::VECCALC> indexBuf_;
    TBuf<QuePosition::VECCALC> wordTmpABuf_;
    TBuf<QuePosition::VECCALC> wordTmpBBuf_;
    TBuf<QuePosition::VECCALC> expandedValidBuf_;
    TBuf<QuePosition::VECCALC> fillWordsBuf_;

    TBuf<QuePosition::VECCALC> fillValueBuf_;

    GlobalTensor<TIn> maskGm_;
    GlobalTensor<int64_t> posGm_;

    int64_t b_ = 0;
    int64_t s_ = 0;
    int64_t fillValue_ = 1;
    uint32_t coreNum_ = 1;
    int64_t rowNum_ = 0;
    int64_t rowBegin_ = 0;
};

#endif // GEN_POSITION_IDS_FROM_MASK_H_
