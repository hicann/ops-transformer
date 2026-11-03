/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

template <uint32_t BATCH>
class TqPrepare {
    static constexpr uint32_t ELEMENTS = BATCH * 128;

public:
    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR fp, GM_ADDR norms, uint32_t vectors, uint32_t blocks)
    {
        vectors_ = vectors;
        blocks_ = blocks;
        input_.SetGlobalBuffer((__gm__ bfloat16_t*)kv);
        fpOut_.SetGlobalBuffer((__gm__ float*)fp);
        normOut_.SetGlobalBuffer((__gm__ float*)norms);
        pipe_.InitBuffer(bfBuf_, ELEMENTS * sizeof(bfloat16_t));
        pipe_.InitBuffer(fpBuf_, (2 * ELEMENTS + 2 * BATCH) * sizeof(float));
        bf_ = bfBuf_.Get<bfloat16_t>();
        auto fpLocal = fpBuf_.Get<float>();
        x_ = fpLocal;
        work_ = fpLocal[ELEMENTS];
        reduced_ = fpLocal[2 * ELEMENTS];
        norm_ = fpLocal[2 * ELEMENTS + BATCH];
    }

    __aicore__ inline void Process()
    {
        uint32_t tiles = (vectors_ + BATCH - 1) / BATCH;
        for (uint32_t tile = GetBlockIdx(); tile < tiles; tile += blocks_) {
            ProcessTile(tile);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t tile)
    {
        uint32_t base = tile * BATCH;
        uint32_t valid = vectors_ - base < BATCH ? vectors_ - base : BATCH;
        if (valid < BATCH) {
            Duplicate(bf_.ReinterpretCast<uint16_t>(), uint16_t(0), ELEMENTS);
            Sync<HardEvent::V_MTE2>();
        }
        DataCopy(bf_, input_[static_cast<uint64_t>(base) * 128], valid * 128);
        Sync<HardEvent::MTE2_V>();
        Cast(x_, bf_, RoundMode::CAST_NONE, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(work_, x_, x_, ELEMENTS);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams params{1, 1, 1, 16, 16, 16};
        Add(work_, work_, work_[64], 64, BATCH, params);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduced_, work_, 64, BATCH, 1, 1, 16);
        Sqrt(reduced_, reduced_, BATCH);
        PipeBarrier<PIPE_V>();
        Adds(norm_, reduced_, 0.0f, BATCH);
        Maxs(reduced_, reduced_, 1.0e-30f, BATCH);
        PipeBarrier<PIPE_V>();
        Brcb(work_, reduced_, BATCH / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams normalize{1, 1, 0, 16, 16, 1};
        Div(x_, x_, work_, 64, BATCH, normalize);
        Div(x_[64], x_[64], work_, 64, BATCH, normalize);
        Sync<HardEvent::V_MTE3>();
        DataCopy(fpOut_[static_cast<uint64_t>(base) * 128], x_, ELEMENTS);
        DataCopy(normOut_[base], norm_, BATCH);
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_MTE2>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> bfBuf_, fpBuf_;
    LocalTensor<bfloat16_t> bf_;
    LocalTensor<float> x_, work_, reduced_, norm_;
    GlobalTensor<bfloat16_t> input_;
    GlobalTensor<float> fpOut_, normOut_;
    uint32_t vectors_, blocks_;
};

template <uint32_t BITS, uint32_t BATCH, typename Meta = bfloat16_t, bool HALF_OUT = false>
class TqQuantize {
    static constexpr uint32_t ELEMENTS = BATCH * 128;
    static constexpr uint32_t GROUPS = BATCH * 16;
    static constexpr uint32_t QUADS = GROUPS / 4;
    static constexpr uint32_t META_STRIDE = BATCH < 16 ? 16 : BATCH;

public:
    __aicore__ inline void Init(GM_ADDR rotated, GM_ADDR norms, GM_ADDR residual, GM_ADDR idx, GM_ADDR normOut,
                                GM_ADDR gammaOut, uint32_t vectors, uint32_t blocks, uint32_t idxStride)
    {
        vectors_ = vectors;
        blocks_ = blocks;
        idxStride_ = idxStride;
        rotated_.SetGlobalBuffer((__gm__ float*)rotated);
        normF_.SetGlobalBuffer((__gm__ float*)norms);
        residualOut_.SetGlobalBuffer((__gm__ float*)residual);
        idxOut_.SetGlobalBuffer((__gm__ uint8_t*)idx);
        normOut_.SetGlobalBuffer((__gm__ Meta*)normOut);
        gammaOut_.SetGlobalBuffer((__gm__ Meta*)gammaOut);
        pipe_.InitBuffer(fpBuf_, (4 * ELEMENTS + 2 * BATCH + 16) * sizeof(float));
        pipe_.InitBuffer(normBuf_, BATCH * sizeof(float));
        pipe_.InitBuffer(bfBuf_, 2 * META_STRIDE * sizeof(bfloat16_t));
        pipe_.InitBuffer(packedBuf_, BATCH * 64);
        pipe_.InitBuffer(maskBuf_, ELEMENTS / 8);
        pipe_.InitBuffer(packWorkBuf_, 2 * GROUPS * sizeof(int32_t));
        pipe_.InitBuffer(byteOffsetBuf_, GROUPS * sizeof(uint32_t));
        pipe_.InitBuffer(compactWorkBuf_, 3 * QUADS * sizeof(int32_t));
        pipe_.InitBuffer(compactOffsetBuf_, 7 * QUADS * sizeof(uint32_t));
        pipe_.InitBuffer(codebookBuf_, 16 * sizeof(float));
        pipe_.InitBuffer(packWeightBuf_, 64 * sizeof(float));
        pipe_.InitBuffer(oneBuf_, 64 * sizeof(float));
        auto fp = fpBuf_.Get<float>();
        y_ = fp;
        residual_ = fp[ELEMENTS];
        index_ = fp[2 * ELEMENTS];
        temp_ = fp[3 * ELEMENTS];
        work_ = index_;
        reduced_ = temp_;
        normLocal_ = normBuf_.Get<float>();
        auto bf = bfBuf_.Get<Meta>();
        normBf_ = bf;
        gammaBf_ = bf[META_STRIDE];
        idxPacked_ = packedBuf_.Get<uint8_t>();
        auto masks = maskBuf_.Get<uint8_t>();
        mask_ = masks;
        indices_ = index_.ReinterpretCast<int32_t>();
        auto packWork = packWorkBuf_.Get<int32_t>();
        codeOffsets_ = temp_.ReinterpretCast<int32_t>();
        threshold_ = residual_;
        laneValues_ = packWork;
        pack32_ = packWork[GROUPS];
        byteOffsets_ = byteOffsetBuf_.Get<uint32_t>();
        compactWork_ = compactWorkBuf_.Get<int32_t>();
        auto compactOffsets = compactOffsetBuf_.Get<uint32_t>();
        quadOffsets_ = compactOffsets;
        interleaveOffsets_ = compactOffsets[4 * QUADS];
        codebook_ = codebookBuf_.Get<float>();
        packWeights_ = packWeightBuf_.Get<float>();
        one_ = oneBuf_.Get<float>();
        Duplicate(one_, 1.0f, 64);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            packWeights_.SetValue(lane, float(1 << (BITS * (lane % (BITS == 4 ? 4 : 8)))));
        }
        for (uint32_t i = 0; i < (1U << BITS); ++i) {
            codebook_.SetValue(i, KvCacheTurboQuant<BITS>::Centroid(i));
        }
        if constexpr (BITS == 2) {
            CreateVecIndex(temp_, 0.0f, GROUPS);
            PipeBarrier<PIPE_V>();
            auto byteOffsets = byteOffsets_.ReinterpretCast<int32_t>();
            Cast(byteOffsets, temp_, RoundMode::CAST_RINT, GROUPS);
            PipeBarrier<PIPE_V>();
            ShiftLeft(byteOffsets, byteOffsets, static_cast<int32_t>(2), GROUPS);
        } else if constexpr (BITS == 3) {
            CreateVecIndex(temp_, 0.0f, QUADS);
            PipeBarrier<PIPE_V>();
            Cast(quadOffsets_.ReinterpretCast<int32_t>(), temp_, RoundMode::CAST_RINT, QUADS);
            PipeBarrier<PIPE_V>();
            ShiftLeft(quadOffsets_.ReinterpretCast<int32_t>(), quadOffsets_.ReinterpretCast<int32_t>(), 4, QUADS);
            PipeBarrier<PIPE_V>();
            for (uint32_t part = 1; part < 4; ++part) {
                Adds(quadOffsets_[part * QUADS].ReinterpretCast<int32_t>(), quadOffsets_.ReinterpretCast<int32_t>(),
                     static_cast<int32_t>(part * sizeof(int32_t)), QUADS);
            }
            constexpr uint32_t count = 3 * QUADS;
            constexpr float sourceStride = BATCH * 16.0f;
            constexpr float quadCorrection = 4.0f - BATCH * 48.0f;
            CreateVecIndex(temp_, 0.0f, count);
            PipeBarrier<PIPE_V>();
            Muls(threshold_, temp_, 1.0f / 3.0f, count);
            PipeBarrier<PIPE_V>();
            Floor(threshold_, threshold_, count);
            PipeBarrier<PIPE_V>();
            Muls(index_, temp_, sourceStride, count);
            Muls(threshold_, threshold_, quadCorrection, count);
            PipeBarrier<PIPE_V>();
            Add(threshold_, threshold_, index_, count);
            PipeBarrier<PIPE_V>();
            Cast(interleaveOffsets_.ReinterpretCast<int32_t>(), threshold_, RoundMode::CAST_RINT, count);
        }
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        uint32_t tiles = (vectors_ + BATCH - 1) / BATCH;
        for (uint32_t tile = GetBlockIdx(); tile < tiles; tile += blocks_) {
            Encode(tile * BATCH);
        }
    }

private:
    __aicore__ inline void SumSquares()
    {
        Mul(work_, residual_, residual_, ELEMENTS);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams params{1, 1, 1, 16, 16, 16};
        Add(work_, work_, work_[64], 64, BATCH, params);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduced_, work_, 64, BATCH, 1, 1, 16);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Encode(uint32_t base)
    {
        DataCopy(y_, rotated_[static_cast<uint64_t>(base) * 128], ELEMENTS);
        DataCopy(normLocal_, normF_[base], BATCH);
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::MTE2_V>();
        if constexpr (BITS != 4) {
            Duplicate(index_, 0.0f, ELEMENTS);
        }
        Abs(temp_, y_, ELEMENTS);
        PipeBarrier<PIPE_V>();
        if constexpr (BITS == 2) {
            AddMagnitudeBoundary(0.08676189929f);
        } else if constexpr (BITS == 3) {
            AddMagnitudeBoundary(0.04424276203f);
            AddMagnitudeBoundary(0.09280399233f);
            AddMagnitudeBoundary(0.1544964164f);
        } else {
            QuantizeFourBitMagnitude();
        }
        FinalizeSymmetricIndex();
        constexpr uint32_t idxBytes = 16 * BITS;
        // Accumulate at most 24 bits in FP32; split 4-bit words into two halves.
        BinaryRepeatParams weights{1, 1, 1, 8, 8, 0};
        Mul(temp_, index_, packWeights_, 64, ELEMENTS / 64, weights);
        PipeBarrier<PIPE_V>();
        if constexpr (BITS != 4) {
            BlockReduceSum(laneValues_.ReinterpretCast<float>(), temp_, ELEMENTS / 64, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            Cast(pack32_, laneValues_.ReinterpretCast<float>(), RoundMode::CAST_RINT, GROUPS);
            PipeBarrier<PIPE_V>();
        } else {
            uint64_t lowMask[2] = {0x0f0f0f0f0f0f0f0fULL, 0};
            uint64_t highMask[2] = {0xf0f0f0f0f0f0f0f0ULL, 0};
            BlockReduceSum(laneValues_.ReinterpretCast<float>(), temp_, ELEMENTS / 64, lowMask, 1, 1, 8);
            BlockReduceSum(threshold_, temp_, ELEMENTS / 64, highMask, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            Cast(pack32_, laneValues_.ReinterpretCast<float>(), RoundMode::CAST_RINT, GROUPS);
            Cast(codeOffsets_, threshold_, RoundMode::CAST_RINT, GROUPS);
            PipeBarrier<PIPE_V>();
            ShiftLeft(codeOffsets_, codeOffsets_, static_cast<int32_t>(16), GROUPS);
            PipeBarrier<PIPE_V>();
            Add(pack32_, pack32_, codeOffsets_, GROUPS);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (BITS == 2) {
            Gather(idxPacked_.ReinterpretCast<uint16_t>(), pack32_.ReinterpretCast<uint16_t>(), byteOffsets_, 0,
                   GROUPS);
        } else if constexpr (BITS == 3) {
            PackThreeBit();
        } else {
            Adds(idxPacked_.ReinterpretCast<int32_t>(), pack32_, static_cast<int32_t>(0), GROUPS);
        }
        PipeBarrier<PIPE_V>();
        // Packing is complete: reuse the float indices and scratch as integers.
        Cast(indices_, index_, RoundMode::CAST_RINT, ELEMENTS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(codeOffsets_, indices_, static_cast<int32_t>(2), ELEMENTS);
        PipeBarrier<PIPE_V>();
        Gather(residual_, codebook_, codeOffsets_.ReinterpretCast<uint32_t>(), 0, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Sub(residual_, y_, residual_, ELEMENTS);
        PipeBarrier<PIPE_V>();
        SumSquares();
        Sqrt(reduced_, reduced_, BATCH);
        PipeBarrier<PIPE_V>();
        Mul(reduced_, normLocal_, reduced_, BATCH);
        PipeBarrier<PIPE_V>();
        if constexpr (HALF_OUT) {
            Cast(normBf_, normLocal_, RoundMode::CAST_NONE, BATCH);
            Cast(gammaBf_, reduced_, RoundMode::CAST_NONE, BATCH);
        } else {
            Cast(normBf_, normLocal_, RoundMode::CAST_RINT, BATCH);
            Cast(gammaBf_, reduced_, RoundMode::CAST_RINT, BATCH);
        }
        Sync<HardEvent::V_MTE3>();
        DataCopy(residualOut_[static_cast<uint64_t>(base) * 128], residual_, ELEMENTS);
        uint32_t valid = vectors_ - base < BATCH ? vectors_ - base : BATCH;
        DataCopyPad(idxOut_[static_cast<uint64_t>(base) * idxStride_], idxPacked_,
                    DataCopyExtParams{1, valid * idxBytes, 0, 0, 0});
        if constexpr (BATCH == 32) {
            DataCopy(normOut_[base], normBf_, BATCH);
            DataCopy(gammaOut_[base], gammaBf_, BATCH);
        } else {
            DataCopyExtParams metaCopy{1, valid * uint32_t(sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(normOut_[base], normBf_, metaCopy);
            DataCopyPad(gammaOut_[base], gammaBf_, metaCopy);
        }
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void AddMagnitudeBoundary(float boundary)
    {
        CompareScalar(mask_, temp_, boundary, CMPMODE::GT, ELEMENTS);
        PipeBarrier<PIPE_V>();
        AddMaskToIndex();
    }

    __aicore__ inline void AddMaskToIndex()
    {
        BinaryRepeatParams select{1, 1, 0, 8, 0, 0};
        Select(threshold_, mask_, one_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, static_cast<uint8_t>(ELEMENTS / 64),
               select);
        PipeBarrier<PIPE_V>();
        Add(index_, index_, threshold_, ELEMENTS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void QuantizeFourBitMagnitude()
    {
        CompareScalar(mask_, temp_, 0.09712349623f, CMPMODE::GT, ELEMENTS);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams select{1, 1, 0, 8, 0, 0};
        Select(threshold_, mask_, one_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64, static_cast<uint8_t>(ELEMENTS / 64),
               select);
        PipeBarrier<PIPE_V>();
        Muls(index_, threshold_, 4.0f, ELEMENTS);
        Muls(residual_, threshold_, 0.162901029f - 0.04615073279f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Adds(residual_, residual_, 0.04615073279f, ELEMENTS);
        PipeBarrier<PIPE_V>();

        Compare(mask_, temp_, residual_, CMPMODE::GT, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Adds(threshold_, index_, 2.0f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Select(index_, mask_, threshold_, index_, SELMODE::VSEL_TENSOR_TENSOR_MODE, ELEMENTS);
        PipeBarrier<PIPE_V>();

        Muls(residual_, index_, 0.000423272412708f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Adds(residual_, residual_, -0.00147489784875f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(residual_, residual_, index_, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Adds(residual_, residual_, 0.0251697309717f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(residual_, residual_, index_, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Adds(residual_, residual_, 0.02281150408f, ELEMENTS);
        PipeBarrier<PIPE_V>();

        Compare(mask_, temp_, residual_, CMPMODE::GT, ELEMENTS);
        PipeBarrier<PIPE_V>();
        AddMaskToIndex();
    }

    __aicore__ inline void FinalizeSymmetricIndex()
    {
        constexpr float half = BITS == 2 ? 2.0f : (BITS == 3 ? 4.0f : 8.0f);
        Adds(threshold_, index_, half, ELEMENTS);
        Muls(temp_, index_, -1.0f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Adds(temp_, temp_, half - 1.0f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        auto signMask = mask_;
        CompareScalar(signMask, y_, 0.0f, CMPMODE::GT, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Select(index_, signMask, threshold_, temp_, SELMODE::VSEL_TENSOR_TENSOR_MODE, ELEMENTS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackThreeBit()
    {
        auto out0 = compactWork_;
        auto out1 = compactWork_[QUADS];
        auto out2 = compactWork_[2 * QUADS];
        Gather(out0, pack32_, quadOffsets_, 0, QUADS);
        Gather(laneValues_, pack32_, quadOffsets_[QUADS], 0, QUADS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(laneValues_, laneValues_, 24, QUADS);
        PipeBarrier<PIPE_V>();
        Add(out0, out0, laneValues_, QUADS);
        PipeBarrier<PIPE_V>();

        Gather(out1, pack32_, quadOffsets_[QUADS], 0, QUADS);
        Gather(laneValues_, pack32_, quadOffsets_[2 * QUADS], 0, QUADS);
        PipeBarrier<PIPE_V>();
        ShiftRight(out1, out1, 8, QUADS);
        ShiftLeft(laneValues_, laneValues_, 16, QUADS);
        PipeBarrier<PIPE_V>();
        Add(out1, out1, laneValues_, QUADS);
        PipeBarrier<PIPE_V>();

        Gather(out2, pack32_, quadOffsets_[2 * QUADS], 0, QUADS);
        Gather(laneValues_, pack32_, quadOffsets_[3 * QUADS], 0, QUADS);
        PipeBarrier<PIPE_V>();
        ShiftRight(out2, out2, 16, QUADS);
        ShiftLeft(laneValues_, laneValues_, 8, QUADS);
        PipeBarrier<PIPE_V>();
        Add(out2, out2, laneValues_, QUADS);
        PipeBarrier<PIPE_V>();
        Gather(idxPacked_.ReinterpretCast<int32_t>(), compactWork_, interleaveOffsets_, 0, 3 * QUADS);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> fpBuf_, normBuf_, bfBuf_, packedBuf_, maskBuf_, codebookBuf_;
    TBuf<TPosition::VECCALC> packWorkBuf_, byteOffsetBuf_;
    TBuf<TPosition::VECCALC> compactWorkBuf_, compactOffsetBuf_;
    TBuf<TPosition::VECCALC> packWeightBuf_, oneBuf_;
    LocalTensor<float> packWeights_, one_;
    LocalTensor<float> y_, residual_, index_, temp_;
    LocalTensor<float> work_, reduced_, normLocal_, codebook_, threshold_;
    LocalTensor<Meta> normBf_, gammaBf_;
    LocalTensor<uint8_t> idxPacked_, mask_;
    LocalTensor<int32_t> indices_, codeOffsets_, laneValues_, pack32_;
    LocalTensor<int32_t> compactWork_;
    LocalTensor<uint32_t> byteOffsets_, quadOffsets_, interleaveOffsets_;
    GlobalTensor<float> rotated_, normF_, residualOut_;
    GlobalTensor<Meta> normOut_, gammaOut_;
    GlobalTensor<uint8_t> idxOut_;
    uint32_t vectors_, blocks_, idxStride_;
};

template <uint32_t BATCH>
class TqSignPack {
    static constexpr uint32_t ELEMENTS = BATCH * 128;

public:
    __aicore__ inline void Init(GM_ADDR projected, GM_ADDR qjlOut, uint32_t vectors, uint32_t blocks)
    {
        vectors_ = vectors;
        blocks_ = blocks;
        projected_.SetGlobalBuffer((__gm__ float*)projected);
        qjlOut_.SetGlobalBuffer((__gm__ uint8_t*)qjlOut);
        pipe_.InitBuffer(fpBuf_, ELEMENTS * sizeof(float));
        pipe_.InitBuffer(packedBuf_, ELEMENTS / 8);
        fp_ = fpBuf_.Get<float>();
        packed_ = packedBuf_.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        uint32_t tiles = (vectors_ + BATCH - 1) / BATCH;
        for (uint32_t tile = GetBlockIdx(); tile < tiles; tile += blocks_) {
            ProcessTile(tile);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t tile)
    {
        uint32_t base = tile * BATCH;
        DataCopy(fp_, projected_[static_cast<uint64_t>(base) * 128], ELEMENTS);
        Sync<HardEvent::MTE2_V>();
        CompareScalar(packed_, fp_, 0.0f, CMPMODE::GE, ELEMENTS);
        Sync<HardEvent::V_MTE3>();
        uint32_t valid = vectors_ - base < BATCH ? vectors_ - base : BATCH;
        DataCopyPad(qjlOut_[static_cast<uint64_t>(base) * 16], packed_, DataCopyExtParams{1, valid * 16, 0, 0, 0});
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::MTE3_MTE2>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> fpBuf_, packedBuf_;
    LocalTensor<float> fp_;
    LocalTensor<uint8_t> packed_;
    GlobalTensor<float> projected_;
    GlobalTensor<uint8_t> qjlOut_;
    uint32_t vectors_, blocks_;
};
