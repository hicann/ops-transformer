/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "qsag_common.h"

namespace QSAG {
using namespace AscendC;

constexpr uint32_t VECTOR_BYTES = 256;
constexpr uint32_t FLOATS_PER_VECTOR = VECTOR_BYTES / sizeof(float);

constexpr AscendC::Reg::CastTrait NZ_CAST_EVEN = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                  AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait NZ_CAST_ODD = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                 AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__simd_vf__ inline void ProbabilityGradToNZ(__ubuf__ T* dstP, __ubuf__ T* dstDs, __ubuf__ float* score,
                                            __ubuf__ float* dp, __ubuf__ float* lse, __ubuf__ float* di, float scale,
                                            uint32_t width, uint32_t rows, uint32_t validRows, uint32_t count,
                                            uint32_t dstRows)
{
    using namespace AscendC::Reg;
    RegTensor<float> se, so, xe, xo, l, d;
    RegTensor<T> low, high, result, zero;
    MaskReg allF = CreateMask<float, MaskPattern::ALL>();
    MaskReg allT = CreateMask<T, MaskPattern::ALL>();
    Duplicate(zero, static_cast<T>(0));
    for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(l, lse + row * 8);
        DataCopy<float, LoadDist::DIST_BRC_B32>(d, di + row * 8);
        for (uint16_t col = 0; col < static_cast<uint16_t>(width); col += 128) {
            auto inputS = score + row * width + col;
            auto inputX = dp + row * width + col;
            DataCopy<float, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(se, so, inputS, 128);
            DataCopy<float, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(xe, xo, inputX, 128);
            Muls(se, se, scale, allF);
            Muls(so, so, scale, allF);
            Sub(se, se, l, allF);
            Sub(so, so, l, allF);
            Exp(se, se, allF);
            Exp(so, so, allF);
            Sub(xe, xe, d, allF);
            Sub(xo, xo, d, allF);
            Mul(xe, xe, se, allF);
            Mul(xo, xo, so, allF);
            uint32_t remaining = row < validRows && count > col ? count - col : 0;
            MaskReg valid = UpdateMask<T>(remaining);
            remaining = width - col;
            MaskReg store = UpdateMask<T>(remaining);
            const uint32_t offset = col / 16 * dstRows * 16 + row * 16;
            auto outputP = dstP + offset;
            auto outputDs = dstDs + offset;
            Cast<T, float, NZ_CAST_EVEN>(low, se, allT);
            Cast<T, float, NZ_CAST_ODD>(high, so, allT);
            Or((RegTensor<uint16_t>&)result, (RegTensor<uint16_t>&)low, (RegTensor<uint16_t>&)high, allT);
            Select(result, result, zero, valid);
            DataCopy<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(outputP, result, dstRows, 1,
                                                                                      store);
            Cast<T, float, NZ_CAST_EVEN>(low, xe, allT);
            Cast<T, float, NZ_CAST_ODD>(high, xo, allT);
            Or((RegTensor<uint16_t>&)result, (RegTensor<uint16_t>&)low, (RegTensor<uint16_t>&)high, allT);
            Select(result, result, zero, valid);
            DataCopy<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(outputDs, result, dstRows, 1,
                                                                                      store);
        }
    }
}

template <typename T>
__aicore__ inline void CastProbabilityToNZ(LocalTensor<T> dst, LocalTensor<float> src, uint32_t rows, uint32_t columns,
                                           uint32_t dstRows)
{
    using namespace AscendC::Reg;
    uint64_t inputBase = src.GetPhyAddr();
    uint64_t outputBase = dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> even, odd;
        RegTensor<T> low, high, result;
        MaskReg all = CreateMask<T, MaskPattern::ALL>();
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            for (uint16_t col = 0; col < static_cast<uint16_t>(columns); col += 128) {
                uint64_t input = inputBase + (row * columns + col) * sizeof(float);
                uint64_t output = outputBase + (col / 16 * dstRows * 16 + row * 16) * sizeof(T);
                uint32_t remaining = columns - col;
                MaskReg valid = UpdateMask<T>(remaining);
                DataCopy<float, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(
                    even, odd, ((__ubuf__ float*&)input), 128);
                Cast<T, float, NZ_CAST_EVEN>(low, even, all);
                Cast<T, float, NZ_CAST_ODD>(high, odd, all);
                Or((RegTensor<uint16_t>&)result, (RegTensor<uint16_t>&)low, (RegTensor<uint16_t>&)high, all);
                DataCopy<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(((__ubuf__ T*&)output),
                                                                                          result, dstRows, 1, valid);
            }
        }
    }
}

__simd_vf__ inline void ProbabilityGradRows(__ubuf__ float* score, __ubuf__ float* dp, __ubuf__ float* lse,
                                            __ubuf__ float* di, float scale, uint32_t width, uint32_t rows,
                                            uint32_t validRows, uint32_t count)
{
    using namespace AscendC::Reg;
    RegTensor<float> s, p, x, zero, l, d;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zero, 0.0f);
    for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(l, lse + r * 8);
        DataCopy<float, LoadDist::DIST_BRC_B32>(d, di + r * 8);
        uint32_t valid = r < validRows ? count : 0;
        for (uint16_t j = 0; j < width / FLOATS_PER_VECTOR; ++j) {
            uint32_t remaining = valid > j * FLOATS_PER_VECTOR ? valid - j * FLOATS_PER_VECTOR : 0;
            MaskReg mask = UpdateMask<float>(remaining);
            uint32_t off = r * width + j * FLOATS_PER_VECTOR;
            LoadAlign(s, score + off);
            LoadAlign(x, dp + off);
            Muls(s, s, scale, all);
            Sub(s, s, l, all);
            Exp(p, s, all);
            Sub(x, x, d, all);
            Mul(x, x, p, all);
            Select(p, p, zero, mask);
            Select(x, x, zero, mask);
            StoreAlign(score + off, p, all);
            StoreAlign(dp + off, x, all);
        }
    }
}

template <typename T>
__simd_vf__ inline void PackSelectedPairNZ(__ubuf__ T* dstK, __ubuf__ T* dstV, __ubuf__ T* srcK, __ubuf__ T* srcV,
                                           uint32_t rows, uint32_t rowStride, uint32_t dim, uint32_t nzStride)
{
    using namespace AscendC::Reg;
    RegTensor<T> key, value;
    constexpr uint32_t registerElements = 256 / sizeof(T);
    for (uint16_t column = 0; column < static_cast<uint16_t>(dim); column += registerElements) {
        uint32_t remaining = dim - column;
        MaskReg valid = UpdateMask<T>(remaining);
        auto outputK = dstK + column * nzStride;
        auto outputV = dstV + column * nzStride;
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            LoadAlign(key, srcK + row * rowStride + column);
            LoadAlign(value, srcV + row * rowStride + column);
            DataCopy<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(outputK, key, nzStride, 1, valid);
            DataCopy<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(outputV, value, nzStride, 1,
                                                                                      valid);
        }
    }
}

__simd_vf__ inline void DotRows(__ubuf__ float* dst, __ubuf__ float* a, __ubuf__ float* b, uint32_t dim, uint32_t rows)
{
    using namespace AscendC::Reg;
    RegTensor<float> x, y, sum, result, zero;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg one = CreateMask<float, MaskPattern::VL1>();
    Duplicate(zero, 0.0f);
    for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
        Duplicate(sum, 0.0f);
        for (uint16_t j = 0; j < (dim + FLOATS_PER_VECTOR - 1) / FLOATS_PER_VECTOR; ++j) {
            uint32_t remain = dim - j * FLOATS_PER_VECTOR;
            MaskReg mask = UpdateMask<float>(remain);
            LoadAlign(x, a + row * dim + j * FLOATS_PER_VECTOR);
            LoadAlign(y, b + row * dim + j * FLOATS_PER_VECTOR);
            Mul(x, x, y, mask);
            Select(x, x, zero, mask);
            Add(sum, sum, x, all);
        }
        ReduceSum(result, sum, all);
        StoreAlign(dst + row * FLOATS_PER_BLOCK, result, one);
    }
}

__simd_vf__ inline void ScalePairedK(__ubuf__ float* pairs, float scale, uint32_t rows, uint32_t dim)
{
    using namespace AscendC::Reg;
    RegTensor<float> value;
    for (uint16_t col = 0; col < static_cast<uint16_t>(dim); col += FLOATS_PER_VECTOR) {
        uint32_t remaining = dim - col;
        MaskReg valid = UpdateMask<float>(remaining);
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            LoadAlign(value, pairs + row * 2 * dim + col);
            Muls(value, value, scale, valid);
            StoreAlign(pairs + row * 2 * dim + col, value, valid);
        }
    }
}

template <typename T, typename Tiling>
class VectorOp {
    struct SparseCursor {
        uint32_t block, offset;
    };
    const ConstInfo<Tiling>* info_;
    SharedBuffers<T>* shared_;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<T> k_, v_, dout_, out_, dq_, dk_, dv_;
    GlobalTensor<float> lse_, di_, wq_, wk_, wv_;
    GlobalTensor<int32_t> indices_, rawIndices_, filteredCounts_;
    LocalTensor<float> lseCache_[SLOT_COUNT];
    LocalTensor<float> transfer0_, transfer1_;
    LocalTensor<int32_t> topkCache_, topkSlots_[PIPELINE_STAGES];
    LocalTensor<T> pNz_, dsNz_;
    TEventID nzReadyEvent_;
    bool nzReservedZero_;
    bool statsValid_ = false;
    uint64_t statsToken_ = 0, statsHead_ = 0;
    uint64_t pairSegmentElements_ = 0;
    bool regularPairedBlocks_ = false;
    bool fuseNz_ = false;
    uint32_t gatherRows_ = 64;

public:
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR out, GM_ADDR lse, GM_ADDR indices,
                                GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const ConstInfo<Tiling>& info,
                                SharedBuffers<T>& shared, TPipe* pipe)
    {
        info_ = &info;
        shared_ = &shared;
        // Wide-D direct results retain the original VF schedule after performance regression.
        fuseNz_ = info_->directResult && info_->dp <= 128;
        pairSegmentElements_ = info_->tk * info_->tiling->kvHeadNum * info_->d;
        gatherRows_ = GatherRowCapacity<T>(info_->dp, info_->transferElements, info_->n);
        regularPairedBlocks_ =
            info_->pairedKV && gatherRows_ % info_->blockSize == 0 && (info_->n / AIV_PER_CUBE) % info_->blockSize == 0;
        k_.SetGlobalBuffer((__gm__ T*)key);
        v_.SetGlobalBuffer((__gm__ T*)value);
        dout_.SetGlobalBuffer((__gm__ T*)dout);
        out_.SetGlobalBuffer((__gm__ T*)out);
        lse_.SetGlobalBuffer((__gm__ float*)lse);
        dq_.SetGlobalBuffer((__gm__ T*)dq);
        dk_.SetGlobalBuffer((__gm__ T*)dk);
        dv_.SetGlobalBuffer((__gm__ T*)dv);
        rawIndices_.SetGlobalBuffer((__gm__ int32_t*)indices);
        indices_.SetGlobalBuffer((__gm__ int32_t*)(workspace + info_->tiling->filteredIndicesOffset));
        filteredCounts_.SetGlobalBuffer((__gm__ int32_t*)(workspace + info_->tiling->filteredCountsOffset));

        di_.SetGlobalBuffer((__gm__ float*)(workspace + info_->tiling->diOffset));
        wq_.SetGlobalBuffer((__gm__ float*)(workspace + info_->tiling->dqOffset));
        wk_.SetGlobalBuffer((__gm__ float*)(workspace + info_->tiling->dkOffset));
        wv_.SetGlobalBuffer((__gm__ float*)(workspace + info_->tiling->dvOffset));

        // The same capacity calculation selects the path and sizes its allocation.
        const auto plan = info_->template PlanUb<T>(info_->directResult, info_->resultSlots, info_->cacheQueryIndices);
        const uint32_t matrixBytes = plan.matrixBytes, resultBytes = plan.resultBytes;
        const uint32_t lseBytes = plan.lseBytes;
        const uint32_t pNzBytes = plan.pNzBytes, dsNzBytes = plan.dsNzBytes, topkBytes = plan.topkBytes;
        const uint32_t transferBytes = info_->transferElements * sizeof(float);
        pipe->InitBuffer(ub_, plan.totalBytes);
        uint32_t ubOffset = 0;
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            shared_->score[slot] = TakeUb<float>(matrixBytes, ubOffset);
            shared_->grad[slot] = TakeUb<float>(matrixBytes, ubOffset);
        }
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            lseCache_[slot] = TakeUb<float>(lseBytes, ubOffset);
        }
        pNz_ = TakeUb<T>(pNzBytes, ubOffset);
        dsNz_ = TakeUb<T>(dsNzBytes, ubOffset);
        for (uint32_t slot = 0; slot < (info_->directResult ? PIPELINE_STAGES : 1); ++slot) {
            topkSlots_[slot] = TakeUb<int32_t>(topkBytes, ubOffset);
        }
        topkCache_ = topkSlots_[0];
        transfer0_ = TakeUb<float>(2 * transferBytes, ubOffset);
        transfer1_ = transfer0_[info_->transferElements];
        if (info_->directResult) {
            for (uint32_t slot = 0; slot < info_->resultSlots; ++slot) {
                shared_->resultK[slot] = TakeUb<float>(2 * resultBytes, ubOffset);
                shared_->resultV[slot] = shared_->resultK[slot][info_->dp];
            }
        }
    }

    // One AIV owns each row. Preserve order and multiplicity while removing masked blocks.
    __aicore__ inline void FilterQuery(uint64_t row, uint64_t visible, int32_t rawCount)
    {
        const uint64_t complete = visible / info_->blockSize;
        const uint32_t count = rawCount <= 0 ? 0 : Min(static_cast<uint64_t>(rawCount), info_->tiling->topK);
        auto input = transfer0_.template ReinterpretCast<int32_t>();
        auto output = transfer1_.template ReinterpretCast<int32_t>();
        uint32_t kept = 0;
        const uint64_t base = row * info_->tiling->topK;
        for (uint32_t first = 0; first < count; first += info_->transferElements) {
            uint32_t size = Min(info_->transferElements, count - first);
            DataCopyExtParams load{1, static_cast<uint32_t>(size * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(input, rawIndices_[base + first], load, DataCopyPadExtParams<int32_t>{});
            Fence<HardEvent::MTE2_S>();
            uint32_t valid = 0;
            for (uint32_t i = 0; i < size; ++i) {
                int32_t block = input.GetValue(i);
                if (block >= 0 && static_cast<uint64_t>(block) < complete) {
                    output.SetValue(valid++, block);
                }
            }
            if (valid != 0) {
                Fence<HardEvent::S_MTE3>();
                DataCopyExtParams store{1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(indices_[base + kept], output, store);
                Fence<HardEvent::MTE3_S>();
                Fence<HardEvent::MTE3_MTE2>();
            }
            kept += valid;
        }
        output.SetValue(0, kept);
        Fence<HardEvent::S_MTE3>();
        DataCopyExtParams store{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPad(filteredCounts_[row], output, store);
        Fence<HardEvent::MTE3_S>();
        Fence<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Begin(TPipe* pipe)
    {
        Pre();
        nzReadyEvent_ = pipe->AllocEventID<HardEvent::MTE3_V>();
        nzReservedZero_ = nzReadyEvent_ == EVENT_ID0;
        if (nzReservedZero_) {
            nzReadyEvent_ = pipe->AllocEventID<HardEvent::MTE3_V>();
        }
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(nzReadyEvent_));
    }

    __aicore__ inline void Finish(TPipe* pipe)
    {
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(nzReadyEvent_));
        pipe->ReleaseEventID<HardEvent::MTE3_V>(nzReadyEvent_);
        if (nzReservedZero_) {
            pipe->ReleaseEventID<HardEvent::MTE3_V>(EVENT_ID0);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        PostTensor(dq_, wq_, info_->tq * info_->tiling->qHeadNum, info_->tiling->softmaxScale);
        if (info_->pairedKV) {
            PostPairedKV(dk_, wk_);
            PostPairedKV(dv_, wk_[info_->dp]);
        } else if (info_->directResult) {
            PostPairedPhysical(dk_, wk_);
            PostPairedPhysical(dv_, wk_[info_->dp]);

        } else {
            PostTensor(dk_, wk_, info_->tk * info_->tiling->kvHeadNum, 1.0f);
            PostTensor(dv_, wv_, info_->tk * info_->tiling->kvHeadNum, 1.0f);
        }
    }

    __aicore__ inline void CacheQueryStatsVector(uint64_t token, uint64_t head, uint32_t rows)
    {
        uint32_t first = info_->sub * info_->halfRows;
        uint32_t valid = first < rows ? Min(info_->halfRows, rows - first) : 0;
        const uint64_t lseStride = info_->LseHeadStride();
        const uint64_t diStride = info_->tiling->inputLayout == 1 ? info_->tiling->qSeqLen : 1;
        // Check both DMA strides before narrowing, without overflowing uint64_t.
        const uint64_t maxStrideElements = MAX_FIXPIPE_STRIDE / sizeof(float) + 1;
        if (lseStride > maxStrideElements || diStride > maxStrideElements / FLOATS_PER_BLOCK) {
            Fence<HardEvent::V_S>();
            for (uint32_t r = 0; r < info_->halfRows; ++r) {
                float lse = 0.0f, di = 0.0f;
                if (r < valid) {
                    uint64_t off = info_->Offset(token, head + first + r, true, 1);
                    lse = lse_.GetValue(info_->LseOffset(token, head + first + r));
                    di = di_.GetValue(off * FLOATS_PER_BLOCK);
                }
                lseCache_[0].SetValue(r * FLOATS_PER_BLOCK, lse);
                lseCache_[1].SetValue(r * FLOATS_PER_BLOCK, di);
            }
            Fence<HardEvent::S_V>();
            return;
        }
        if (valid < info_->halfRows) {
            Duplicate(lseCache_[0][valid * FLOATS_PER_BLOCK], 0.0f, (info_->halfRows - valid) * FLOATS_PER_BLOCK);
            Duplicate(lseCache_[1][valid * FLOATS_PER_BLOCK], 0.0f, (info_->halfRows - valid) * FLOATS_PER_BLOCK);
        }
        Fence<HardEvent::V_MTE2>();
        if (valid != 0) {
            uint64_t off = info_->Offset(token, head + first, true, 1);
            const uint64_t lseOff = info_->LseOffset(token, head + first);
            DataCopyExtParams cp{static_cast<uint16_t>(valid), sizeof(float),
                                 static_cast<uint32_t>((lseStride - 1) * sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, FLOATS_PER_BLOCK - 1, 0.0f};
            DataCopyPad(lseCache_[0], lse_[lseOff], cp, pad);
            cp.srcStride = static_cast<uint32_t>((diStride * FLOATS_PER_BLOCK - 1) * sizeof(float));
            DataCopyPad(lseCache_[1], di_[off * FLOATS_PER_BLOCK], cp, pad);
        }
        Fence<HardEvent::MTE2_V>();
    }

    __aicore__ inline void Gather(const TileTask& task)
    {
        if (info_->directResult) {
            topkCache_ = topkSlots_[info_->cacheQueryIndices ? task.indexSlot : task.id % PIPELINE_STAGES];
        }
        if (info_->cacheQueryIndices && !task.tail) {
            if (task.first == 0) {
                LoadIndices(task.sparseBase, task.indexCount);
            }
            topkCache_ = topkCache_[task.first];
        }
        GatherRows(task.sparseBase, task.first, task.selected, task.tail, task.tailStart, task.kbase, task.kvHead,
                   task.id % SLOT_COUNT);
    }

    __aicore__ inline void Scatter(const TileTask& task)
    {
        if (info_->directResult) {
            topkCache_ = topkSlots_[info_->cacheQueryIndices ? task.indexSlot : task.id % PIPELINE_STAGES];
            if (info_->cacheQueryIndices && !task.tail) {
                topkCache_ = topkCache_[task.first];
            }
        }
        ScatterRows(task.sparseBase, task.first, task.selected, task.tail, task.tailStart, task.kbase, task.kvHead,
                    task.id % SLOT_COUNT);
    }

    __aicore__ inline void PrepareSoftmax(const TileTask& task)
    {
        if (info_->querySlots == SLOT_COUNT && (!statsValid_ || statsToken_ != task.token || statsHead_ != task.head)) {
            CacheQueryStatsVector(task.token, task.head, info_->groupRows);
            statsToken_ = task.token;
            statsHead_ = task.head;
            statsValid_ = true;
        }
    }

    __aicore__ inline void Softmax(uint32_t count, uint32_t slot)
    {
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(nzReadyEvent_));
        uint32_t first = info_->sub * info_->halfRows;
        uint32_t validRows = first < info_->groupRows ? Min(info_->halfRows, info_->groupRows - first) : 0;
        if (fuseNz_) {
            asc_vf_call<ProbabilityGradToNZ<T>>(
                (__ubuf__ T*)pNz_.GetPhyAddr(), (__ubuf__ T*)dsNz_.GetPhyAddr(),
                (__ubuf__ float*)shared_->score[slot].GetPhyAddr(), (__ubuf__ float*)shared_->grad[slot].GetPhyAddr(),
                (__ubuf__ float*)lseCache_[0].GetPhyAddr(), (__ubuf__ float*)lseCache_[1].GetPhyAddr(),
                info_->tiling->softmaxScale, info_->n, info_->halfRows, validRows, count, info_->nzRows);
        } else {
            asc_vf_call<ProbabilityGradRows>(
                (__ubuf__ float*)shared_->score[slot].GetPhyAddr(), (__ubuf__ float*)shared_->grad[slot].GetPhyAddr(),
                (__ubuf__ float*)lseCache_[0].GetPhyAddr(), (__ubuf__ float*)lseCache_[1].GetPhyAddr(),
                info_->tiling->softmaxScale, info_->n, info_->halfRows, validRows, count);
            PipeBarrier<PIPE_V>();
            // NZ 每列块增加一行间隔，容量与设计中的 (m+1)*NZRowBytes 对应。
            CastProbabilityToNZ<T>(pNz_, shared_->score[slot], info_->halfRows, info_->n, info_->nzRows);
            CastProbabilityToNZ<T>(dsNz_, shared_->grad[slot], info_->halfRows, info_->n, info_->nzRows);
        }
        Fence<HardEvent::V_MTE3>();
        CopyProbabilityToL1(shared_->pl[slot], pNz_);
        CopyProbabilityToL1(shared_->dsl[slot], dsNz_);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(nzReadyEvent_));
    }

private:
    template <typename Element>
    __aicore__ inline LocalTensor<Element> TakeUb(uint32_t bytes, uint32_t& offset)
    {
        auto tensor = ub_.GetWithOffset<Element>(bytes / sizeof(Element), offset);
        offset += AlignBytes(bytes);
        return tensor;
    }

    __aicore__ inline uint32_t NextSparseRun(SparseCursor& cursor, uint32_t remaining, uint32_t position, bool tail,
                                             uint64_t tailStart, uint64_t& token)
    {
        if (tail) {
            token = tailStart + position;
            return remaining;
        }
        token = static_cast<uint64_t>(topkCache_.GetValue(cursor.block)) * info_->blockSize + cursor.offset;
        uint32_t run = Min(remaining, info_->blockSize - cursor.offset);
        cursor.offset += run;
        if (cursor.offset == info_->blockSize) {
            ++cursor.block;
            cursor.offset = 0;
        }
        return run;
    }

    __aicore__ inline void ReadRow(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset, uint32_t rows)
    {
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams cp{static_cast<uint16_t>(rows), static_cast<uint32_t>(info_->d * sizeof(T)), 0, 0, 0};
        DataCopyPad(dst, src[offset], cp,
                    DataCopyPadExtParams<T>{true, 0, static_cast<uint8_t>(info_->dp - info_->d), static_cast<T>(0)});
        Fence<HardEvent::MTE2_V>();
    }

    __aicore__ inline void ZeroWorkspace(GlobalTensor<float> gm, uint64_t count)
    {
        const uint32_t zeroElements = info_->transferElements;
        Duplicate(transfer0_, 0.0f, zeroElements);
        Fence<HardEvent::V_MTE3>();
        for (uint64_t start = info_->vcore * zeroElements; start < count;
             start += info_->tiling->cubeCoreNum * AIV_PER_CUBE * zeroElements) {
            uint32_t len = Min(zeroElements, count - start);
            DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
            DataCopyPad(gm[start], transfer0_, cp);
        }
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void StorePairArena(uint64_t offset, LocalTensor<T> src, uint32_t elements)
    {
        if (offset >= pairSegmentElements_) {
            DataCopy(dv_[offset - pairSegmentElements_], src, elements);
        } else if (elements <= pairSegmentElements_ - offset) {
            DataCopy(dk_[offset], src, elements);
        } else {
            uint32_t firstElements = pairSegmentElements_ - offset;
            DataCopy(dk_[offset], src, firstElements);
            DataCopy(dv_, src[firstElements], elements - firstElements);
        }
    }

    __aicore__ inline void LoadPairArena(LocalTensor<T> dst, uint64_t offset, uint32_t elements)
    {
        if (offset >= pairSegmentElements_) {
            DataCopy(dst, dv_[offset - pairSegmentElements_], elements);
        } else if (elements <= pairSegmentElements_ - offset) {
            DataCopy(dst, dk_[offset], elements);
        } else {
            uint32_t firstElements = pairSegmentElements_ - offset;
            DataCopy(dst, dk_[offset], firstElements);
            DataCopy(dst[firstElements], dv_, elements - firstElements);
        }
    }

    __aicore__ inline uint32_t PhysicalKvRun(uint64_t row, uint32_t rows, uint64_t& offset, uint32_t& stride)
    {
        uint64_t head = row / info_->tk;
        uint64_t token = row % info_->tk;
        uint32_t run = Min(rows, info_->tk - token);
        if (info_->tiling->inputLayout == 1) {
            uint64_t seq = info_->tiling->kvSeqLen;
            run = Min(run, seq - token % seq);
            offset = ((token / seq * info_->tiling->kvHeadNum + head) * seq + token % seq) * info_->d;
            stride = 0;
        } else {
            offset = (token * info_->tiling->kvHeadNum + head) * info_->d;
            uint64_t wideStride = (info_->tiling->kvHeadNum - 1) * info_->d * sizeof(T);
            run = wideStride <= MAX_FIXPIPE_STRIDE ? run : 1;
            stride = wideStride <= MAX_FIXPIPE_STRIDE ? static_cast<uint32_t>(wideStride) : 0;
        }
        return run;
    }

    __aicore__ inline void PackPairedKV()
    {
        const uint32_t copyRows = info_->transferElements / info_->dp;
        const uint64_t totalRows = info_->tk * info_->tiling->kvHeadNum;
        auto local = transfer0_.template ReinterpretCast<T>();
        for (uint64_t row = info_->vcore * copyRows; row < totalRows;
             row += info_->tiling->cubeCoreNum * AIV_PER_CUBE * copyRows) {
            uint32_t rows = Min(copyRows, totalRows - row);
            for (uint32_t copied = 0; copied < rows;) {
                uint64_t off;
                uint32_t stride;
                uint32_t run = PhysicalKvRun(row + copied, rows - copied, off, stride);
                DataCopyExtParams cp{static_cast<uint16_t>(run), static_cast<uint32_t>(info_->d * sizeof(T)), stride,
                                     static_cast<uint32_t>(info_->d * sizeof(T) / BLOCK_BYTES), 0};
                DataCopyPad(local[copied * 2 * info_->d], k_[off], cp, DataCopyPadExtParams<T>{});
                DataCopyPad(local[(copied * 2 + 1) * info_->d], v_[off], cp, DataCopyPadExtParams<T>{});
                copied += run;
            }
            Fence<HardEvent::MTE2_MTE3>();
            StorePairArena(row * 2 * info_->d, local, rows * 2 * info_->d);
            Fence<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void PostPairedKV(GlobalTensor<T> output, GlobalTensor<float> src)
    {
        const uint32_t copyRows = info_->transferElements / info_->dp;
        const uint64_t totalRows = info_->tk * info_->tiling->kvHeadNum;
        auto localOutput = transfer1_.template ReinterpretCast<T>();
        for (uint64_t row = info_->vcore * copyRows; row < totalRows;
             row += info_->tiling->cubeCoreNum * AIV_PER_CUBE * copyRows) {
            uint32_t rows = Min(copyRows, totalRows - row);
            DataCopyExtParams load{static_cast<uint16_t>(rows), static_cast<uint32_t>(info_->dp * sizeof(float)),
                                   static_cast<uint32_t>(info_->dp * sizeof(float)), 0, 0};
            DataCopyPad(transfer0_, src[row * 2 * info_->dp], load, DataCopyPadExtParams<float>{});
            Fence<HardEvent::MTE2_V>();
            Cast(localOutput, transfer0_, RoundMode::CAST_RINT, rows * info_->dp);
            Fence<HardEvent::V_MTE3>();
            for (uint32_t copied = 0; copied < rows;) {
                uint64_t off;
                uint32_t stride;
                uint32_t run = PhysicalKvRun(row + copied, rows - copied, off, stride);
                DataCopyExtParams cp{static_cast<uint16_t>(run), static_cast<uint32_t>(info_->d * sizeof(T)), 0, stride,
                                     0};
                DataCopyPad(output[off], localOutput[copied * info_->d], cp);
                copied += run;
            }
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void PostPairedPhysical(GlobalTensor<T> output, GlobalTensor<float> src)
    {
        const uint32_t copyRows = info_->transferElements / info_->dp;
        const uint64_t totalRows = info_->tk * info_->tiling->kvHeadNum;
        auto localOutput = transfer1_.template ReinterpretCast<T>();
        for (uint64_t row = info_->vcore * copyRows; row < totalRows;
             row += info_->tiling->cubeCoreNum * AIV_PER_CUBE * copyRows) {
            uint32_t rows = Min(copyRows, totalRows - row);
            DataCopyExtParams load{static_cast<uint16_t>(rows), static_cast<uint32_t>(info_->dp * sizeof(float)),
                                   static_cast<uint32_t>(info_->dp * sizeof(float)), 0, 0};
            DataCopyPad(transfer0_, src[row * 2 * info_->dp], load, DataCopyPadExtParams<float>{});
            Fence<HardEvent::MTE2_V>();
            Cast(localOutput, transfer0_, RoundMode::CAST_RINT, rows * info_->dp);
            Fence<HardEvent::V_MTE3>();
            DataCopyExtParams store{static_cast<uint16_t>(rows), static_cast<uint32_t>(info_->d * sizeof(T)), 0, 0, 0};
            DataCopyPad(output[row * info_->d], localOutput, store);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void Pre()
    {
        ZeroWorkspace(wq_, info_->tq * info_->tiling->qHeadNum * info_->dp);
        const uint64_t kvElements = info_->tk * info_->tiling->kvHeadNum * info_->dp;
        if (info_->directResult) {
            ZeroWorkspace(wk_, 2 * kvElements);
        } else {
            ZeroWorkspace(wk_, kvElements);
            ZeroWorkspace(wv_, kvElements);
        }
        const uint64_t totalRows = info_->tq * info_->tiling->qHeadNum;
        const auto plan = info_->template PlanUb<T>(info_->directResult, info_->resultSlots);
        const uint32_t vectorDim = (info_->dp + FLOATS_PER_VECTOR - 1) / FLOATS_PER_VECTOR * FLOATS_PER_VECTOR;
        const uint32_t batchRows = Min(32, (plan.matrixBytes / sizeof(float) - (vectorDim - info_->dp)) / info_->dp);
        auto rawOut = shared_->score[0].template ReinterpretCast<T>();
        auto rawDout = shared_->score[1].template ReinterpretCast<T>();
        auto fpOut = shared_->grad[0];
        auto fpDout = shared_->grad[1];
        auto scalarResults = lseCache_[0];
        for (uint64_t row = info_->vcore * batchRows; row < totalRows;
             row += info_->tiling->cubeCoreNum * AIV_PER_CUBE * batchRows) {
            uint32_t rows = Min(batchRows, totalRows - row);
            // The flattened row and Di follow the actual Q layout; LSE uses its own layout.
            ReadRow(rawOut, out_, row * info_->d, rows);
            ReadRow(rawDout, dout_, row * info_->d, rows);
            Cast(fpOut, rawOut, RoundMode::CAST_NONE, rows * info_->dp);
            Cast(fpDout, rawDout, RoundMode::CAST_NONE, rows * info_->dp);
            PipeBarrier<PIPE_V>();
            Duplicate(scalarResults, 0.0f, rows * FLOATS_PER_BLOCK);
            PipeBarrier<PIPE_V>();
            asc_vf_call<DotRows>((__ubuf__ float*)scalarResults.GetPhyAddr(), (__ubuf__ float*)fpOut.GetPhyAddr(),
                                 (__ubuf__ float*)fpDout.GetPhyAddr(), info_->dp, rows);
            Fence<HardEvent::V_MTE3>();
            DataCopy(di_[row * FLOATS_PER_BLOCK], scalarResults, rows * FLOATS_PER_BLOCK);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
        if (info_->pairedKV) {
            PackPairedKV();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
    }

    __aicore__ inline void PostTensor(GlobalTensor<T> output, GlobalTensor<float> src, uint64_t rows, float scale)
    {
        const uint32_t copyRows = info_->transferElements / info_->dp;
        auto converted = transfer1_.template ReinterpretCast<T>();
        for (uint64_t row = info_->vcore * copyRows; row < rows;
             row += info_->tiling->cubeCoreNum * AIV_PER_CUBE * copyRows) {
            uint32_t batchRows = Min(copyRows, rows - row);
            DataCopy(transfer0_, src[row * info_->dp], batchRows * info_->dp);
            Fence<HardEvent::MTE2_V>();
            Muls(transfer0_, transfer0_, scale, batchRows * info_->dp);
            PipeBarrier<PIPE_V>();
            Cast(converted, transfer0_, RoundMode::CAST_RINT, batchRows * info_->dp);
            Fence<HardEvent::V_MTE3>();
            DataCopyExtParams cp{static_cast<uint16_t>(batchRows), static_cast<uint32_t>(info_->d * sizeof(T)), 0, 0,
                                 0};
            DataCopyPad(output[row * info_->d], converted, cp);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void PackSelectedNZ(LocalTensor<T> dst, LocalTensor<T> src, uint32_t rows, uint32_t rowStride)
    {
        using namespace AscendC::Reg;
        uint64_t input = src.GetPhyAddr();
        uint64_t outputBase = dst.GetPhyAddr();
        const uint32_t gatherStride = gatherRows_ + 1;
        __VEC_SCOPE__
        {
            RegTensor<T> value;
            constexpr uint32_t registerElements = 256 / sizeof(T);
            const uint32_t columnChunks = (info_->dp + registerElements - 1) / registerElements;
            for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(columnChunks); ++chunk) {
                const uint32_t column = chunk * registerElements;
                uint32_t remaining = info_->dp - column;
                MaskReg valid = UpdateMask<T>(remaining);
                uint64_t output = outputBase + column * gatherStride * sizeof(T);
                for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
                    LoadAlign(value, (__ubuf__ T*)input + row * rowStride + column);
                    DataCopy<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T*&)output), value, gatherStride, 1, valid);
                }
            }
        }
    }

    __aicore__ inline void GatherToL1(uint32_t count, bool tail, uint64_t tailStart, uint64_t kbase, uint64_t head,
                                      uint32_t slot)
    {
        const uint32_t halfTile = info_->n / AIV_PER_CUBE;
        const uint32_t halfBegin = info_->sub * halfTile;
        auto gatherK = transfer0_.template ReinterpretCast<T>();
        auto gatherV = transfer1_.template ReinterpretCast<T>();
        const uint32_t localStride = info_->pairedKV ? 2 * info_->dp : info_->dp;
        if (info_->pairedKV) {
            gatherV = gatherK[info_->dp];
        }
        GlobalTensor<T> sourceK = k_;
        GlobalTensor<T> sourceV = v_;
        const uint64_t rowStride =
            (info_->pairedKV || info_->tiling->inputLayout == 1) ? info_->d : info_->tiling->kvHeadNum * info_->d;
        const uint64_t base = info_->Offset(kbase, head, false, info_->d);
        const uint64_t stride = (rowStride - info_->d) * sizeof(T);
        const uint32_t maxRun = stride <= MAX_FIXPIPE_STRIDE ? info_->n : 1;
        const bool fullPairedBlocks = regularPairedBlocks_ && !tail && count == info_->n;
        const uint32_t pairBlockElements = info_->blockSize * 2 * info_->dp;
        SparseCursor cursor{halfBegin / info_->blockSize, halfBegin % info_->blockSize};
        for (uint32_t begin = halfBegin; begin < halfBegin + halfTile; begin += gatherRows_) {
            const uint32_t rows = Min(gatherRows_, halfBegin + halfTile - begin);
            const uint32_t end = Min(count, begin + rows);
            if (fullPairedBlocks) {
                const uint32_t lastBlock = end / info_->blockSize;
                uint32_t localOffset = 0;
                for (uint32_t block = begin / info_->blockSize; block < lastBlock; ++block) {
                    uint64_t off = 2 * base + static_cast<uint64_t>(topkCache_.GetValue(block)) * pairBlockElements;
                    LoadPairArena(gatherK[localOffset], off, pairBlockElements);
                    localOffset += pairBlockElements;
                }
            } else if (info_->pairedKV) {
                for (uint32_t r = begin; r < end;) {
                    uint64_t token;
                    uint32_t run = NextSparseRun(cursor, end - r, r, tail, tailStart, token);
                    uint64_t off = base + token * rowStride;
                    LoadPairArena(gatherK[(r - begin) * localStride], 2 * off, run * localStride);
                    r += run;
                }
            } else {
                for (uint32_t r = begin; r < end;) {
                    uint64_t token;
                    uint32_t run = NextSparseRun(cursor, Min(end - r, maxRun), r, tail, tailStart, token);
                    uint64_t off = base + token * rowStride;
                    DataCopyExtParams cp{static_cast<uint16_t>(run), static_cast<uint32_t>(info_->d * sizeof(T)),
                                         static_cast<uint32_t>(maxRun == 1 ? 0 : stride), 0, 0};
                    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(info_->dp - info_->d), static_cast<T>(0)};
                    if (stride == 0 && info_->d == info_->dp) {
                        cp.blockCount = 1;
                        cp.blockLen = run * info_->d * sizeof(T);
                    }
                    DataCopyPad(gatherK[(r - begin) * localStride], sourceK[off], cp, pad);
                    DataCopyPad(gatherV[(r - begin) * localStride], sourceV[off], cp, pad);
                    r += run;
                }
            }
            Fence<HardEvent::MTE2_V>();
            // P/dS NZ scratch may still be read by the preceding probability copy.
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(nzReadyEvent_));
            uint32_t validRows = end > begin ? end - begin : 0;
            if (validRows < rows) {
                Duplicate(gatherK[validRows * localStride], static_cast<T>(0), (rows - validRows) * localStride);
                if (!info_->pairedKV) {
                    Duplicate(gatherV[validRows * localStride], static_cast<T>(0), (rows - validRows) * localStride);
                }
                PipeBarrier<PIPE_V>();
            }
            if (fuseNz_) {
                asc_vf_call<PackSelectedPairNZ<T>>((__ubuf__ T*)pNz_.GetPhyAddr(), (__ubuf__ T*)dsNz_.GetPhyAddr(),
                                                   (__ubuf__ T*)gatherK.GetPhyAddr(), (__ubuf__ T*)gatherV.GetPhyAddr(),
                                                   rows, localStride, info_->dp, gatherRows_ + 1);
            } else {
                PackSelectedNZ(pNz_, gatherK, rows, localStride);
                PackSelectedNZ(dsNz_, gatherV, rows, localStride);
            }
            Fence<HardEvent::V_MTE3>();
            DataCopyParams cp;
            cp.blockCount = info_->dp / NZ_WIDTH;
            cp.blockLen = rows;
            cp.srcStride = gatherRows_ + 1 - rows;
            cp.dstStride = info_->n - rows;
            DataCopy(shared_->kl[slot][begin * NZ_WIDTH], pNz_, cp);
            DataCopy(shared_->vl[slot][begin * NZ_WIDTH], dsNz_, cp);
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(nzReadyEvent_));
            // The next phase may reuse the ND scratch with MTE2.
            Fence<HardEvent::V_MTE2>();
        }
    }

    __aicore__ inline void LoadIndices(uint64_t offset, uint32_t entries)
    {
        Fence<HardEvent::S_MTE2>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(entries * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(topkCache_, indices_[offset], cp, DataCopyPadExtParams<int32_t>{});
        Fence<HardEvent::MTE2_S>();
    }

    __aicore__ inline void GatherRows(uint64_t sparseBase, uint64_t first, uint32_t count, bool tail,
                                      uint64_t tailStart, uint64_t kbase, uint64_t head, uint32_t slot)
    {
        if (!tail && !info_->cacheQueryIndices) {
            LoadIndices(sparseBase + first, count / info_->blockSize);
        }
        if (info_->directResult) {
            GatherToL1(count, tail, tailStart, kbase, head, slot);
            return;
        }
        uint32_t begin = info_->sub * (info_->n / AIV_PER_CUBE);
        uint32_t end = begin + info_->n / AIV_PER_CUBE;
        const uint64_t rowStride = info_->tiling->inputLayout == 1 ? info_->d : info_->tiling->kvHeadNum * info_->d;
        const uint64_t base = info_->Offset(kbase, head, false, info_->d);
        const uint64_t stride = (rowStride - info_->d) * sizeof(T);
        const uint32_t maxRun = stride <= MAX_FIXPIPE_STRIDE ? info_->n : 1;
        SparseCursor cursor{begin / info_->blockSize, begin % info_->blockSize};
        auto gatherK = transfer0_.template ReinterpretCast<T>();
        auto gatherV = transfer1_.template ReinterpretCast<T>();
        for (uint32_t r = begin; r < end;) {
            uint32_t rows = Min(info_->transferElements / info_->dp, end - r);
            if (r < count) {
                rows = Min(rows, count - r);
                for (uint32_t copied = 0; copied < rows;) {
                    uint32_t pos = r + copied;
                    uint64_t token;
                    uint32_t run = NextSparseRun(cursor, Min(rows - copied, maxRun), pos, tail, tailStart, token);
                    uint64_t off = base + token * rowStride;
                    DataCopyExtParams cp{static_cast<uint16_t>(run), static_cast<uint32_t>(info_->d * sizeof(T)),
                                         static_cast<uint32_t>(maxRun == 1 ? 0 : stride), 0, 0};
                    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(info_->dp - info_->d), static_cast<T>(0)};
                    DataCopyPad(gatherK[copied * info_->dp], k_[off], cp, pad);
                    DataCopyPad(gatherV[copied * info_->dp], v_[off], cp, pad);
                    copied += run;
                }
                Fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(gatherK, static_cast<T>(0), rows * info_->dp);
                Duplicate(gatherV, static_cast<T>(0), rows * info_->dp);
                Fence<HardEvent::V_MTE3>();
            }
            DataCopy(shared_->selK[slot][r * info_->dp], gatherK, rows * info_->dp);
            DataCopy(shared_->selV[slot][r * info_->dp], gatherV, rows * info_->dp);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
            r += rows;
        }
    }

    __aicore__ inline void CopyProbabilityToL1(LocalTensor<T> dst, LocalTensor<T> src)
    {
        // C->V 按 ceil(G/2) 分行；两个 AIV 的有效行在 L1 中连续排列。
        DataCopyParams cp;
        cp.blockCount = info_->n / NZ_WIDTH;
        cp.blockLen = info_->halfRows;
        cp.srcStride = info_->nzRows - info_->halfRows;
        cp.dstStride = info_->groupCapacity - info_->halfRows;
        DataCopy(dst[info_->sub * info_->halfRows * NZ_WIDTH], src, cp);
    }

    __aicore__ inline void ScatterRows(uint64_t sparseBase, uint64_t first, uint32_t count, bool tail,
                                       uint64_t tailStart, uint64_t kbase, uint64_t head, uint32_t slot)
    {
        if (!tail && !info_->directResult) {
            LoadIndices(sparseBase + first, count / info_->blockSize);
        }
        uint32_t begin = info_->sub * (info_->n / AIV_PER_CUBE);
        uint32_t end = Min(count, begin + info_->n / AIV_PER_CUBE);
        const uint64_t rowStride = info_->KvGradRowStride() * info_->dp;
        const uint64_t base = info_->KvGradRow(kbase, head) * info_->dp;
        const uint64_t stride = (rowStride - info_->dp) * sizeof(float);
        const uint32_t maxRun = stride <= MAX_FIXPIPE_STRIDE ? info_->n : 1;
        SparseCursor cursor{begin / info_->blockSize, begin % info_->blockSize};
        if (info_->directResult) {
            auto localK = shared_->resultK[info_->ResultSlot(slot)];
            if (begin < end) {
                uint32_t vfRows = end - begin - (info_->dp == NZ_WIDTH ? 1 : 0);
                if (vfRows != 0) {
                    asc_vf_call<ScalePairedK>((__ubuf__ float*)localK.GetPhyAddr(), info_->tiling->softmaxScale, vfRows,
                                              info_->dp);
                }
                if (info_->dp == NZ_WIDTH) {
                    PipeBarrier<PIPE_V>();
                    Muls(localK[vfRows * 2 * info_->dp], localK[vfRows * 2 * info_->dp], info_->tiling->softmaxScale,
                         info_->dp);
                }
            }
            Fence<HardEvent::V_MTE3>();
            SetAtomicAdd<float>();
            if (regularPairedBlocks_ && !tail && count == info_->n) {
                const uint32_t pairBlockElements = info_->blockSize * 2 * info_->dp;
                const uint32_t lastBlock = end / info_->blockSize;
                uint32_t localOffset = 0;
                for (uint32_t block = begin / info_->blockSize; block < lastBlock; ++block) {
                    uint64_t off = 2 * base + static_cast<uint64_t>(topkCache_.GetValue(block)) * pairBlockElements;
                    DataCopy(wk_[off], localK[localOffset], pairBlockElements);
                    localOffset += pairBlockElements;
                }
            } else if (info_->pairedKV) {
                for (uint32_t r = begin; r < end;) {
                    uint64_t token;
                    uint32_t run = NextSparseRun(cursor, end - r, r, tail, tailStart, token);
                    uint64_t off = base + token * rowStride;
                    DataCopy(wk_[2 * off], localK[(r - begin) * 2 * info_->dp], run * 2 * info_->dp);
                    r += run;
                }
            } else {
                const uint64_t pairStride = 2 * stride;
                const uint32_t pairMaxRun = pairStride <= MAX_FIXPIPE_STRIDE ? info_->n : 1;
                for (uint32_t r = begin; r < end;) {
                    uint64_t token;
                    uint32_t run = NextSparseRun(cursor, Min(end - r, pairMaxRun), r, tail, tailStart, token);
                    uint64_t off = 2 * (base + token * rowStride);
                    DataCopyExtParams cp{static_cast<uint16_t>(run),
                                         static_cast<uint32_t>(2 * info_->dp * sizeof(float)), 0,
                                         static_cast<uint32_t>(pairMaxRun == 1 ? 0 : pairStride), 0};
                    if (pairStride == 0) {
                        cp.blockCount = 1;
                        cp.blockLen = run * 2 * info_->dp * sizeof(float);
                    }
                    DataCopyPad(wk_[off], localK[(r - begin) * 2 * info_->dp], cp);
                    r += run;
                }
            }
            SetAtomicNone();

            return;
        }
        for (uint32_t r = begin; r < end;) {
            uint32_t rows = Min(info_->transferElements / info_->dp, end - r);
            DataCopy(transfer0_, shared_->mm4[slot][r * info_->dp], rows * info_->dp);
            DataCopy(transfer1_, shared_->mm5[slot][r * info_->dp], rows * info_->dp);
            Fence<HardEvent::MTE2_V>();
            Muls(transfer0_, transfer0_, info_->tiling->softmaxScale, rows * info_->dp);
            Fence<HardEvent::V_MTE3>();
            SetAtomicAdd<float>();
            for (uint32_t copied = 0; copied < rows;) {
                uint32_t pos = r + copied;
                uint64_t token;
                uint32_t run = NextSparseRun(cursor, Min(rows - copied, maxRun), pos, tail, tailStart, token);
                uint64_t off = base + token * rowStride;
                DataCopyExtParams cp{static_cast<uint16_t>(run), static_cast<uint32_t>(info_->dp * sizeof(float)), 0,
                                     static_cast<uint32_t>(maxRun == 1 ? 0 : stride), 0};
                DataCopyPad(wk_[off], transfer0_[copied * info_->dp], cp);
                DataCopyPad(wv_[off], transfer1_[copied * info_->dp], cp);
                copied += run;
            }
            SetAtomicNone();
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
            r += rows;
        }
    }
};
} // namespace QSAG
