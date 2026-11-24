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
 * \file msa_index_score_topk_arch22.h
 * \brief A2/A3 阶段 B：在已写好的 fp32 score 行上做 TopK。
 *        入选规则与 950 相同：分数从高到低，同分保留更小列号；
 *        写出时列号小于 tailStart 的在前，尾块列号接在后面。
 *        UB 只有 192KB，长度表和行元数据放在 64KB/80KB。
 *        没有 950 的 RADIX_SELECT，k=16 走 Sort32。
 */

#ifndef MSA_INDEX_SCORE_TOPK_ARCH22_H
#define MSA_INDEX_SCORE_TOPK_ARCH22_H

#include "kernel_operator.h"
#include "../msa_index_score_common.h"

namespace MsaIndexScoreNs {

struct MsaTopkCand {
    float score;
    int32_t index;
};

template <bool NATIVE_A8C8 = false>
class MsaIndexScoreTopk {
public:
    __aicore__ inline void Run(const MsaIndexScoreTilingData* __restrict tiling,
                               const AscendC::GlobalTensor<float>& score, AscendC::GlobalTensor<int32_t>& indices,
                               AscendC::GlobalTensor<float>& values, const AscendC::GlobalTensor<int32_t>& qlen,
                               const AscendC::GlobalTensor<int32_t>& klen, bool writeValues,
                               const AscendC::LocalTensor<float>& ubTrunk)
    {
        ubTrunk_ = ubTrunk;
        if (tiling->totalQ == 0 || tiling->numQHeads == 0 || tiling->topk == 0) {
            return;
        }
        const uint32_t subIdx = AscendC::GetBlockIdx() % MSA_AIV_PER_AIC;
        const uint32_t coreIdx = AscendC::GetBlockIdx() / MSA_AIV_PER_AIC;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t vid = coreIdx * MSA_AIV_PER_AIC + subIdx;
        const uint32_t aivNum = coreNum * MSA_AIV_PER_AIC;
        CacheLens(tiling, qlen, klen);
        BuildRowMeta(tiling, qlen, klen);
        const uint32_t rows = tiling->numQHeads * tiling->totalQ;
        if (tiling->topk == MSA_TOPK_MAX && rowMetaReady_) {
            RunDense128(tiling, score, indices, values, qlen, klen, writeValues, vid, aivNum, rows);
            return;
        }
        for (uint32_t row = vid; row < rows; row += aivNum) {
            const uint32_t head = row / tiling->totalQ;
            const uint32_t token = row - head * tiling->totalQ;
            SelectRow(tiling, score, indices, values, qlen, klen, head, token, writeValues);
        }
    }

private:
    __aicore__ inline bool Better(const MsaTopkCand& a, const MsaTopkCand& b) const
    {
        if (a.score > b.score) {
            return true;
        }
        if (a.score < b.score) {
            return false;
        }
        return a.index < b.index;
    }

    __aicore__ inline void RecomputeWorst()
    {
        worst_ = 0;
        for (uint32_t i = 1; i < nsel_; ++i) {
            if (Better(sel_[worst_], sel_[i])) {
                worst_ = i;
            }
        }
    }

    // 拷贝长度对不齐 stride 时退回逐列扫描。集合未满时新元素若比当前最差更差，它就是新的最差。
    __aicore__ inline void Consider(float score, int32_t index, uint32_t limit)
    {
        MsaTopkCand cand{score, index};
        if (nsel_ < limit) {
            if (nsel_ == 0 || Better(sel_[worst_], cand)) {
                worst_ = nsel_;
            }
            sel_[nsel_] = cand;
            ++nsel_;
            return;
        }
        if (Better(cand, sel_[worst_])) {
            sel_[worst_] = cand;
            RecomputeWorst();
        }
    }

    // 长度前缀只搬一次。排序缓冲用 UB 低地址，这张表放在 128KB 之后。
    __aicore__ inline void CacheLens(const MsaIndexScoreTilingData* __restrict tiling,
                                     const AscendC::GlobalTensor<int32_t>& qlen,
                                     const AscendC::GlobalTensor<int32_t>& klen)
    {
        lensReady_ = false;
        const uint32_t nQ = tiling->batch + 1U;
        const bool pa = MsaIsPaKeyLayout(tiling->keyLayout);
        const uint32_t nK = pa ? tiling->batch : (tiling->batch + 1U);
        if (nQ < 2U || nQ > MSA_TOPK_LEN_CAP || nK > MSA_TOPK_LEN_CAP) {
            return;
        }
        qUb_ = ubTrunk_[MSA_TOPK_LEN_FLOAT_OFF_A2].ReinterpretCast<int32_t>();
        kUb_ = qUb_[MSA_TOPK_LEN_CAP];
        CopyLens(qUb_, qlen, nQ);
        CopyLens(kUb_, klen, nK);
        lensReady_ = true;
    }

    __aicore__ inline void CopyLens(AscendC::LocalTensor<int32_t>& dst, const AscendC::GlobalTensor<int32_t>& src,
                                    uint32_t n)
    {
        const uint32_t aligned = n & ~(MSA_FP32_PER_BLOCK - 1U);
        if (aligned > 0U) {
            AscendC::DataCopy(dst, src, aligned);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        }
        for (uint32_t i = aligned; i < n; ++i) {
            dst.SetValue(i, src.GetValue(i));
        }
    }

    // 每个 token 的请求长度只算一次。行循环里不再按 batch 重扫。
    __aicore__ inline void BuildRowMeta(const MsaIndexScoreTilingData* __restrict tiling,
                                        const AscendC::GlobalTensor<int32_t>& qlen,
                                        const AscendC::GlobalTensor<int32_t>& klen)
    {
        rowMetaReady_ = false;
        if (!lensReady_ || tiling->totalQ == 0U || tiling->totalQ > MSA_TOPK_META_CAP) {
            return;
        }
        qMeta_ = ubTrunk_[MSA_TOPK_META_FLOAT_OFF_A2].ReinterpretCast<int32_t>();
        kvMeta_ = qMeta_[MSA_TOPK_META_CAP];
        tOffMeta_ = qMeta_[MSA_TOPK_META_CAP * 2U];
        const bool pa = MsaIsPaKeyLayout(tiling->keyLayout);
        uint32_t b = 0;
        for (uint32_t token = 0; token < tiling->totalQ; ++token) {
            int32_t qLen = 0;
            int32_t kvLen = 0;
            int32_t tOff = 0;
            while (b < tiling->batch) {
                const int32_t qStart = LensAt(qUb_, qlen, b);
                int32_t qCur = LensAt(qUb_, qlen, b + 1U) - qStart;
                if (qCur < 0) {
                    qCur = 0;
                }
                if (tiling->qMaxSeq > 0U && static_cast<uint32_t>(qCur) > tiling->qMaxSeq) {
                    qCur = static_cast<int32_t>(tiling->qMaxSeq);
                }
                const int32_t qEnd = qStart + qCur;
                if (static_cast<int32_t>(token) < qStart) {
                    break;
                }
                if (static_cast<int32_t>(token) >= qEnd) {
                    ++b;
                    continue;
                }
                tOff = static_cast<int32_t>(token) - qStart;
                qLen = qCur;
                if (pa) {
                    kvLen = LensAt(kUb_, klen, b);
                } else {
                    kvLen = LensAt(kUb_, klen, b + 1U) - LensAt(kUb_, klen, b);
                    if (kvLen < 0) {
                        kvLen = 0;
                    }
                    if (tiling->kMaxSeq > 0U && static_cast<uint32_t>(kvLen) > tiling->kMaxSeq) {
                        kvLen = static_cast<int32_t>(tiling->kMaxSeq);
                    }
                }
                break;
            }
            qMeta_.SetValue(token, qLen);
            kvMeta_.SetValue(token, kvLen);
            tOffMeta_.SetValue(token, tOff);
        }
        rowMetaReady_ = true;
    }

    __aicore__ inline int32_t LensAt(const AscendC::LocalTensor<int32_t>& cached,
                                     const AscendC::GlobalTensor<int32_t>& gm, uint32_t index) const
    {
        if (lensReady_) {
            return cached.GetValue(index);
        }
        return gm.GetValue(index);
    }

    __aicore__ inline int32_t VisibleEnd(const MsaIndexScoreTilingData* __restrict tiling, int32_t kvLen, int32_t qLen,
                                         int32_t tOff) const
    {
        if (tiling->sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN) {
            int32_t end = kvLen - qLen + tOff + 1;
            if (end < 0) {
                end = 0;
            }
            const int32_t cap = (kvLen < 0) ? 0 : kvLen;
            if (end > cap) {
                end = cap;
            }
            return end;
        }
        return (kvLen < 0) ? 0 : kvLen;
    }

    __aicore__ inline void LoadRequest(const MsaIndexScoreTilingData* __restrict tiling,
                                       const AscendC::GlobalTensor<int32_t>& qlen,
                                       const AscendC::GlobalTensor<int32_t>& klen, uint32_t token, int32_t& qLen,
                                       int32_t& kvLen, int32_t& tOff) const
    {
        qLen = 0;
        kvLen = 0;
        tOff = 0;
        for (uint32_t b = 0; b < tiling->batch; ++b) {
            const int32_t qStart = LensAt(qUb_, qlen, b);
            int32_t qCur = LensAt(qUb_, qlen, b + 1) - qStart;
            if (qCur < 0) {
                qCur = 0;
            }
            if (tiling->qMaxSeq > 0U && static_cast<uint32_t>(qCur) > tiling->qMaxSeq) {
                qCur = static_cast<int32_t>(tiling->qMaxSeq);
            }
            const int32_t qEnd = qStart + qCur;
            if (static_cast<int32_t>(token) < qStart || static_cast<int32_t>(token) >= qEnd) {
                continue;
            }
            tOff = static_cast<int32_t>(token) - qStart;
            qLen = qCur;
            if (MsaIsPaKeyLayout(tiling->keyLayout)) {
                kvLen = LensAt(kUb_, klen, b);
            } else {
                kvLen = LensAt(kUb_, klen, b + 1) - LensAt(kUb_, klen, b);
                if (kvLen < 0) {
                    kvLen = 0;
                }
                if (tiling->kMaxSeq > 0U && static_cast<uint32_t>(kvLen) > tiling->kMaxSeq) {
                    kvLen = static_cast<int32_t>(tiling->kMaxSeq);
                }
            }
            return;
        }
    }

    __aicore__ inline uint32_t ValidBlocks(const MsaIndexScoreTilingData* __restrict tiling, int32_t kvLen,
                                           int32_t qLen, int32_t tOff) const
    {
        const int32_t visible = VisibleEnd(tiling, kvLen, qLen, tOff);
        uint32_t blocks = 0;
        if (visible > 0) {
            blocks = (static_cast<uint32_t>(visible) + MSA_BLOCK_SIZE - 1U) / MSA_BLOCK_SIZE;
        }
        uint32_t cap = tiling->maxBlocksPerBatch;
        if (MsaIsPaKeyLayout(tiling->keyLayout) && tiling->tableWidth > 0U && tiling->tableWidth < cap) {
            cap = tiling->tableWidth;
        }
        if (blocks > cap) {
            blocks = cap;
        }
        return blocks;
    }

    __aicore__ inline bool ColumnInRange(int32_t index, int32_t lo, int32_t hi) const
    {
        return index >= lo && index < hi;
    }

    // 同分时只留 room 个最小列号。out 里按列号升序，满了就丢掉更大的列号。
    __aicore__ inline void KeepSmallest(MsaTopkCand* out, uint32_t& n, uint32_t room, float score, int32_t index) const
    {
        if (n < room) {
            uint32_t pos = n;
            while (pos > 0 && out[pos - 1U].index > index) {
                out[pos] = out[pos - 1U];
                --pos;
            }
            out[pos] = {score, index};
            ++n;
            return;
        }
        if (index >= out[n - 1U].index) {
            return;
        }
        uint32_t pos = n - 1U;
        while (pos > 0 && out[pos - 1U].index > index) {
            out[pos] = out[pos - 1U];
            --pos;
        }
        out[pos] = {score, index};
    }

    // Sort32 只保证分数降序。同一分数的一段里再按列号取够剩余名额。
    __aicore__ inline void TakeTop(const AscendC::LocalTensor<float>& sorted, uint32_t nAlign, int32_t indexLo,
                                   int32_t indexHi, uint32_t limit, MsaTopkCand* out, uint32_t& nout) const
    {
        auto sortedBits = sorted.ReinterpretCast<uint32_t>();
        nout = 0;
        uint32_t i = 0;
        while (nout < limit && i < nAlign) {
            const float score = sorted.GetValue(i * MSA_TOPK_SORT_PAIR);
            uint32_t runEnd = i + 1U;
            while (runEnd < nAlign && sorted.GetValue(runEnd * MSA_TOPK_SORT_PAIR) == score) {
                ++runEnd;
            }
            uint32_t validInRun = 0;
            for (uint32_t j = i; j < runEnd; ++j) {
                const int32_t index = static_cast<int32_t>(sortedBits.GetValue(j * MSA_TOPK_SORT_PAIR + 1U));
                if (ColumnInRange(index, indexLo, indexHi)) {
                    ++validInRun;
                }
            }
            if (validInRun > 0U) {
                const uint32_t room = limit - nout;
                if (validInRun <= room) {
                    for (uint32_t j = i; j < runEnd; ++j) {
                        const int32_t index = static_cast<int32_t>(sortedBits.GetValue(j * MSA_TOPK_SORT_PAIR + 1U));
                        if (ColumnInRange(index, indexLo, indexHi)) {
                            out[nout] = {score, index};
                            ++nout;
                        }
                    }
                } else {
                    uint32_t kept = 0;
                    for (uint32_t j = i; j < runEnd; ++j) {
                        const int32_t index = static_cast<int32_t>(sortedBits.GetValue(j * MSA_TOPK_SORT_PAIR + 1U));
                        if (ColumnInRange(index, indexLo, indexHi)) {
                            KeepSmallest(out + nout, kept, room, score, index);
                        }
                    }
                    nout += kept;
                }
            }
            i = runEnd;
        }
    }

    // srcScore | srcIndex | sorted(2*n) | tmp(2*n)。nAlign 是 32 的倍数。
    __aicore__ inline void SortLoaded(uint32_t nAlign)
    {
        auto srcScore = ubTrunk_;
        auto srcIndex = ubTrunk_[nAlign].ReinterpretCast<uint32_t>();
        auto sorted = ubTrunk_[nAlign * MSA_TOPK_SORT_PAIR];
        auto tmp = ubTrunk_[nAlign * MSA_TOPK_SORT_PAIR * 2U];
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        AscendC::Sort<float, true>(sorted, srcScore, srcIndex, tmp, static_cast<int32_t>(nAlign / MSA_TOPK_SORT32_LEN));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    }

    __aicore__ inline void SortColumns(uint32_t begin, uint32_t count, const AscendC::GlobalTensor<float>& score,
                                       uint64_t rowBase)
    {
        uint32_t nAlign = MsaAlignUpPow2(count, MSA_TOPK_SORT32_LEN);
        if (nAlign < MSA_TOPK_SORT32_LEN) {
            nAlign = MSA_TOPK_SORT32_LEN;
        }
        auto srcScore = ubTrunk_;
        uint32_t copyCount = MsaAlignUpPow2(count, MSA_FP32_PER_BLOCK);
        if (copyCount > nAlign) {
            copyCount = nAlign;
        }
        AscendC::DataCopy(srcScore, score[rowBase + begin], copyCount);
        auto srcIndexI = ubTrunk_[nAlign].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(srcIndexI, static_cast<int32_t>(begin), nAlign);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        for (uint32_t i = count; i < nAlign; ++i) {
            srcScore.SetValue(i, MSA_FILL_VALUE);
            srcIndexI.SetValue(i, MSA_TOPK_PAD_INDEX);
        }
        SortLoaded(nAlign);
    }

    __aicore__ inline void SortCandidates(const MsaTopkCand* a, uint32_t na, const MsaTopkCand* b, uint32_t nb,
                                          uint32_t& nAlign)
    {
        const uint32_t n = na + nb;
        nAlign = MsaAlignUpPow2(n, MSA_TOPK_SORT32_LEN);
        if (nAlign < MSA_TOPK_SORT32_LEN) {
            nAlign = MSA_TOPK_SORT32_LEN;
        }
        auto srcScore = ubTrunk_;
        auto srcIndexI = ubTrunk_[nAlign].ReinterpretCast<int32_t>();
        for (uint32_t i = 0; i < na; ++i) {
            srcScore.SetValue(i, a[i].score);
            srcIndexI.SetValue(i, a[i].index);
        }
        for (uint32_t i = 0; i < nb; ++i) {
            srcScore.SetValue(na + i, b[i].score);
            srcIndexI.SetValue(na + i, b[i].index);
        }
        for (uint32_t i = n; i < nAlign; ++i) {
            srcScore.SetValue(i, MSA_FILL_VALUE);
            srcIndexI.SetValue(i, MSA_TOPK_PAD_INDEX);
        }
        SortLoaded(nAlign);
    }

    __aicore__ inline bool ChunkCopyFits(uint32_t begin, uint32_t count, uint32_t rowStride) const
    {
        const uint32_t copyCount = MsaAlignUpPow2(count, MSA_FP32_PER_BLOCK);
        const uint32_t room = (begin < rowStride) ? (rowStride - begin) : 0U;
        return copyCount >= count && copyCount <= room;
    }

    // 256 列、k=128。A2 上把多行拼进一次 Sort32/MrgSort 会把向量地址打出 192KB UB，
    // 所以这里只收集行，真正排序仍逐行走 EmitDense256。
    __aicore__ inline void RunDense128(const MsaIndexScoreTilingData* __restrict tiling,
                                       const AscendC::GlobalTensor<float>& score,
                                       AscendC::GlobalTensor<int32_t>& indices, AscendC::GlobalTensor<float>& values,
                                       const AscendC::GlobalTensor<int32_t>& qlen,
                                       const AscendC::GlobalTensor<int32_t>& klen, bool writeValues, uint32_t vid,
                                       uint32_t aivNum, uint32_t rows)
    {
        constexpr uint32_t kBatch = MSA_TOPK_DENSE_BATCH;
        uint32_t pending[kBatch];
        uint32_t nPending = 0;
        for (uint32_t row = vid; row < rows; row += aivNum) {
            const uint32_t token = row % tiling->totalQ;
            const int32_t qLen = qMeta_.GetValue(token);
            const int32_t kvLen = kvMeta_.GetValue(token);
            const int32_t tOff = tOffMeta_.GetValue(token);
            const uint32_t valid = ValidBlocks(tiling, kvLen, qLen, tOff);
            const bool dense = valid == MSA_TOPK_DENSE_COLS && ((valid / MSA_TOPK_TRUNK) * MSA_TOPK_TRUNK) == 0U &&
                               ChunkCopyFits(0U, MSA_TOPK_DENSE_COLS, tiling->strideOutToken);
            if (!dense) {
                FlushDense128(tiling, score, indices, values, qlen, klen, writeValues, pending, nPending);
                nPending = 0;
                const uint32_t head = row / tiling->totalQ;
                SelectRow(tiling, score, indices, values, qlen, klen, head, token, writeValues);
                continue;
            }
            pending[nPending] = row;
            ++nPending;
            if (nPending == kBatch) {
                FlushDense128(tiling, score, indices, values, qlen, klen, writeValues, pending, nPending);
                nPending = 0;
            }
        }
        FlushDense128(tiling, score, indices, values, qlen, klen, writeValues, pending, nPending);
    }

    __aicore__ inline void FlushDense128(const MsaIndexScoreTilingData* __restrict tiling,
                                         const AscendC::GlobalTensor<float>& score,
                                         AscendC::GlobalTensor<int32_t>& indices, AscendC::GlobalTensor<float>& values,
                                         const AscendC::GlobalTensor<int32_t>& qlen,
                                         const AscendC::GlobalTensor<int32_t>& klen, bool writeValues,
                                         const uint32_t* rows, uint32_t nRow)
    {
        for (uint32_t r = 0; r < nRow; ++r) {
            const uint32_t row = rows[r];
            const uint32_t head = row / tiling->totalQ;
            const uint32_t token = row - head * tiling->totalQ;
            const uint64_t rowBase = static_cast<uint64_t>(head) * tiling->strideOutHead +
                                     static_cast<uint64_t>(token) * tiling->strideOutToken;
            const uint64_t outBase = static_cast<uint64_t>(row) * tiling->topk;
            if (EmitDense256(score, indices, values, rowBase, outBase, tiling->strideOutToken, writeValues)) {
                continue;
            }
            SelectRow(tiling, score, indices, values, qlen, klen, head, token, writeValues);
        }
    }

    // 256 列、k=128：一次分数排序加一次列号排序，中间不再把向量流水排空。
    __aicore__ inline bool EmitDense256(const AscendC::GlobalTensor<float>& score,
                                        AscendC::GlobalTensor<int32_t>& indices, AscendC::GlobalTensor<float>& values,
                                        uint64_t rowBase, uint64_t outBase, uint32_t rowStride, bool writeValues)
    {
        constexpr uint32_t n = MSA_TOPK_DENSE_COLS;
        constexpr uint32_t k = MSA_TOPK_DENSE_K;
        if (!ChunkCopyFits(0U, n, rowStride)) {
            return false;
        }
        auto srcScore = ubTrunk_;
        auto srcIndex = ubTrunk_[n].ReinterpretCast<uint32_t>();
        auto sorted = ubTrunk_[n * MSA_TOPK_SORT_PAIR];
        auto sortTmp = ubTrunk_[n * MSA_TOPK_SORT_PAIR * 2U];
        AscendC::DataCopy(srcScore, score[rowBase], n);
        auto srcIndexI = srcIndex.ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(srcIndexI, static_cast<int32_t>(0), n);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Sort<float, true>(sorted, srcScore, srcIndex, sortTmp, static_cast<int32_t>(n / MSA_TOPK_SORT32_LEN));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        const float boundary = sorted.GetValue((k - 1U) * MSA_TOPK_SORT_PAIR);
        const float next = sorted.GetValue(k * MSA_TOPK_SORT_PAIR);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        if (boundary == next) {
            return false;
        }
        auto extScore = ubTrunk_[n * MSA_TOPK_SORT_PAIR * 3U];
        auto extIndex = ubTrunk_[n * MSA_TOPK_SORT_PAIR * 3U + n].ReinterpretCast<uint32_t>();
        AscendC::Extract(extScore, extIndex, sorted, static_cast<int32_t>(n / MSA_TOPK_SORT32_LEN));
        auto key = ubTrunk_;
        AscendC::Cast(key, extIndex.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_RINT, k);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(key, key, -1.0f, k);
        auto slotI = ubTrunk_[k].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(slotI, static_cast<int32_t>(0), k);
        auto sorted2 = ubTrunk_[n * MSA_TOPK_SORT_PAIR];
        auto tmp2 = ubTrunk_[n * MSA_TOPK_SORT_PAIR * 2U];
        AscendC::Sort<float, true>(sorted2, key, slotI.ReinterpretCast<uint32_t>(), tmp2,
                                   static_cast<int32_t>(k / MSA_TOPK_SORT32_LEN));
        auto slotOut = ubTrunk_.ReinterpretCast<uint32_t>();
        auto dummy = ubTrunk_[k];
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Extract(dummy, slotOut, sorted2, static_cast<int32_t>(k / MSA_TOPK_SORT32_LEN));
        AscendC::PipeBarrier<PIPE_V>();
        auto byteOff = ubTrunk_[n * MSA_TOPK_SORT_PAIR].ReinterpretCast<uint32_t>();
        AscendC::ShiftLeft(byteOff, slotOut, MSA_TOPK_GATHER_BYTE_SHIFT, static_cast<int32_t>(k));
        AscendC::PipeBarrier<PIPE_V>();
        auto outScore = ubTrunk_[n * MSA_TOPK_SORT_PAIR * 2U];
        auto outIndex = ubTrunk_[n * MSA_TOPK_SORT_PAIR * 2U + k].ReinterpretCast<int32_t>();
        AscendC::Gather(outScore, extScore, byteOff, 0U, k);
        AscendC::Gather(outIndex, extIndex.ReinterpretCast<int32_t>(), byteOff, 0U, k);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(indices[outBase], outIndex, k);
        if (writeValues) {
            AscendC::DataCopy(values[outBase], outScore, k);
        }
        if constexpr (NATIVE_A8C8) {
            // The next row can reuse this buffer through Scalar, Vector or MTE2.
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        return true;
    }

    // 长行 Top16 只需阈值选择，避免完整排序；候选仍由 WriteOutputs 按 prefix/tail 列号输出。
    __aicore__ inline bool EmitRadix16(const AscendC::GlobalTensor<float>& score,
                                       AscendC::GlobalTensor<int32_t>& indices, AscendC::GlobalTensor<float>& values,
                                       uint64_t rowBase, uint64_t outBase, uint32_t valid, uint32_t k,
                                       uint32_t rowStride, uint32_t tailStart, bool writeValues)
    {
        // A2/A3 没有 950 的 RADIX_SELECT。k=16 交给后面的 Sort32 路径，入选规则不变。
        (void)score;
        (void)indices;
        (void)values;
        (void)rowBase;
        (void)outBase;
        (void)valid;
        (void)k;
        (void)rowStride;
        (void)tailStart;
        (void)writeValues;
        return false;
    }

    // tailStart==0 时入选集合按列号升序即可。分数排序后用向量重排，不再逐个标量读。
    __aicore__ inline bool EmitIndexOrder(const AscendC::GlobalTensor<float>& score,
                                          AscendC::GlobalTensor<int32_t>& indices, AscendC::GlobalTensor<float>& values,
                                          uint64_t rowBase, uint64_t outBase, uint32_t valid, uint32_t limit,
                                          uint32_t k, uint32_t rowStride, bool writeValues)
    {
        uint32_t mAlign = MsaAlignUpPow2(limit, MSA_TOPK_SORT32_LEN);
        if (mAlign < MSA_TOPK_SORT32_LEN) {
            mAlign = MSA_TOPK_SORT32_LEN;
        }
        if (valid == 0U || valid > MSA_TOPK_SORT_CHUNK || k > mAlign || (k % MSA_FP32_PER_BLOCK) != 0U ||
            !ChunkCopyFits(0U, valid, rowStride)) {
            return false;
        }
        SortColumns(0U, valid, score, rowBase);
        uint32_t nAlign = MsaAlignUpPow2(valid, MSA_TOPK_SORT32_LEN);
        if (nAlign < MSA_TOPK_SORT32_LEN) {
            nAlign = MSA_TOPK_SORT32_LEN;
        }
        auto sorted = ubTrunk_[nAlign * MSA_TOPK_SORT_PAIR];
        // Sort padding can tie with a valid native score clamped to MSA_FILL_VALUE.
        if (limit < (NATIVE_A8C8 ? nAlign : valid)) {
            const float boundary = sorted.GetValue((limit - 1U) * MSA_TOPK_SORT_PAIR);
            const float next = sorted.GetValue(limit * MSA_TOPK_SORT_PAIR);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
            if (boundary == next) {
                return false;
            }
        }
        auto extScore = ubTrunk_;
        auto extIndex = ubTrunk_[nAlign].ReinterpretCast<uint32_t>();
        AscendC::Extract(extScore, extIndex, sorted, static_cast<int32_t>(nAlign / MSA_TOPK_SORT32_LEN));
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t keyBase = nAlign * MSA_TOPK_SORT_PAIR;
        const uint32_t slotBase = keyBase + mAlign;
        const uint32_t sort2 = slotBase + mAlign;
        const uint32_t tmp2 = sort2 + mAlign * MSA_TOPK_SORT_PAIR;
        auto key = ubTrunk_[keyBase];
        AscendC::Cast(key, extIndex.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_RINT, limit);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(key, key, -1.0f, limit);
        if (mAlign > limit) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            for (uint32_t i = limit; i < mAlign; ++i) {
                key.SetValue(i, MSA_FILL_VALUE);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2);
        }
        auto slotI = ubTrunk_[slotBase].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(slotI, static_cast<int32_t>(0), mAlign);
        auto sorted2 = ubTrunk_[sort2];
        auto tmp = ubTrunk_[tmp2];
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3);
        AscendC::Sort<float, true>(sorted2, key, slotI.ReinterpretCast<uint32_t>(), tmp,
                                   static_cast<int32_t>(mAlign / MSA_TOPK_SORT32_LEN));
        auto slotOut = ubTrunk_[keyBase].ReinterpretCast<uint32_t>();
        auto dummy = ubTrunk_[keyBase + mAlign];
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Extract(dummy, slotOut, sorted2, static_cast<int32_t>(mAlign / MSA_TOPK_SORT32_LEN));
        AscendC::PipeBarrier<PIPE_V>();
        auto byteOff = ubTrunk_[sort2].ReinterpretCast<uint32_t>();
        AscendC::ShiftLeft(byteOff, slotOut, MSA_TOPK_GATHER_BYTE_SHIFT, static_cast<int32_t>(limit));
        AscendC::PipeBarrier<PIPE_V>();
        auto outScore = ubTrunk_[tmp2];
        auto outIndex = ubTrunk_[tmp2 + k].ReinterpretCast<int32_t>();
        AscendC::Gather(outScore, extScore, byteOff, 0U, limit);
        AscendC::Gather(outIndex, extIndex.ReinterpretCast<int32_t>(), byteOff, 0U, limit);
        if (k > limit) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
            for (uint32_t j = limit; j < k; ++j) {
                outIndex.SetValue(j, MSA_TOPK_INVALID_INDEX);
                if (writeValues) {
                    outScore.SetValue(j, MSA_FILL_VALUE);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }
        AscendC::DataCopy(indices[outBase], outIndex, k);
        if (writeValues) {
            AscendC::DataCopy(values[outBase], outScore, k);
        }
        if constexpr (NATIVE_A8C8) {
            // The next row can reuse this buffer through Scalar, Vector or MTE2.
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        return true;
    }

    __aicore__ inline void SelectBySort(const AscendC::GlobalTensor<float>& score, uint64_t rowBase, uint32_t valid,
                                        uint32_t limit, uint32_t rowStride)
    {
        nsel_ = 0;
        for (uint32_t begin = 0; begin < valid; begin += MSA_TOPK_SORT_CHUNK) {
            uint32_t count = valid - begin;
            if (count > MSA_TOPK_SORT_CHUNK) {
                count = MSA_TOPK_SORT_CHUNK;
            }
            if (!ChunkCopyFits(begin, count, rowStride)) {
                nsel_ = 0;
                worst_ = 0;
                for (uint32_t col = 0; col < valid; ++col) {
                    Consider(score.GetValue(rowBase + col), static_cast<int32_t>(col), limit);
                }
                return;
            }
            SortColumns(begin, count, score, rowBase);
            uint32_t nAlign = MsaAlignUpPow2(count, MSA_TOPK_SORT32_LEN);
            if (nAlign < MSA_TOPK_SORT32_LEN) {
                nAlign = MSA_TOPK_SORT32_LEN;
            }
            auto sorted = ubTrunk_[nAlign * MSA_TOPK_SORT_PAIR];
            uint32_t nChunk = 0;
            TakeTop(sorted, nAlign, static_cast<int32_t>(begin), static_cast<int32_t>(begin + count), limit, cand_,
                    nChunk);
            if (nsel_ == 0U) {
                for (uint32_t i = 0; i < nChunk; ++i) {
                    sel_[i] = cand_[i];
                }
                nsel_ = nChunk;
                continue;
            }
            uint32_t mergedAlign = 0;
            SortCandidates(sel_, nsel_, cand_, nChunk, mergedAlign);
            TakeTop(ubTrunk_[mergedAlign * MSA_TOPK_SORT_PAIR], mergedAlign, 0, MSA_TOPK_PAD_INDEX, limit, sel_, nsel_);
        }
    }

    // 前缀列号升序在前，尾块列号升序在后。用排序键代替 O(k^2) 插入。
    __aicore__ inline void OrderSelected(uint32_t tailStart)
    {
        if (nsel_ <= 1U) {
            for (uint32_t i = 0; i < nsel_; ++i) {
                cand_[i] = sel_[i];
            }
            return;
        }
        uint32_t nAlign = MsaAlignUpPow2(nsel_, MSA_TOPK_SORT32_LEN);
        if (nAlign < MSA_TOPK_SORT32_LEN) {
            nAlign = MSA_TOPK_SORT32_LEN;
        }
        auto srcScore = ubTrunk_;
        auto srcIndexI = ubTrunk_[nAlign].ReinterpretCast<int32_t>();
        constexpr float kTailBias = MSA_TOPK_TAIL_KEY_BIAS;
        for (uint32_t i = 0; i < nsel_; ++i) {
            float key = static_cast<float>(sel_[i].index);
            if (sel_[i].index >= static_cast<int32_t>(tailStart)) {
                key += kTailBias;
            }
            srcScore.SetValue(i, -key);
            srcIndexI.SetValue(i, static_cast<int32_t>(i));
        }
        for (uint32_t i = nsel_; i < nAlign; ++i) {
            srcScore.SetValue(i, MSA_FILL_VALUE);
            srcIndexI.SetValue(i, MSA_TOPK_PAD_INDEX);
        }
        SortLoaded(nAlign);
        auto sortedBits = ubTrunk_[nAlign * MSA_TOPK_SORT_PAIR].ReinterpretCast<uint32_t>();
        for (uint32_t i = 0; i < nsel_; ++i) {
            const uint32_t slot = sortedBits.GetValue(i * MSA_TOPK_SORT_PAIR + 1U);
            cand_[i] = sel_[slot];
        }
    }

    __aicore__ inline void WriteOutputs(AscendC::GlobalTensor<int32_t>& indices, AscendC::GlobalTensor<float>& values,
                                        uint64_t outBase, uint32_t k, uint32_t tailStart, bool writeValues)
    {
        OrderSelected(tailStart);
        // 先在 UB 里排好。k 是 8 的倍数时按 32B DataCopy。
        // 否则按字节 DataCopyPad，避免 32B DataCopy 写进下一行，也不走刷不干净的 GM SetValue。
        const bool copyOut = (k % MSA_FP32_PER_BLOCK) == 0U;
        auto dstIndex = ubTrunk_.ReinterpretCast<int32_t>();
        const uint32_t valueOff = MsaAlignUpPow2(k, MSA_FP32_PER_BLOCK);
        auto dstValue = ubTrunk_[valueOff];
        for (uint32_t j = 0; j < k; ++j) {
            int32_t index = MSA_TOPK_INVALID_INDEX;
            float value = MSA_FILL_VALUE;
            if (j < nsel_) {
                index = cand_[j].index;
                value = cand_[j].score;
            }
            dstIndex.SetValue(j, index);
            if (writeValues) {
                dstValue.SetValue(j, value);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        if (copyOut) {
            AscendC::DataCopy(indices[outBase], dstIndex, k);
            if (writeValues) {
                AscendC::DataCopy(values[outBase], dstValue, k);
            }
        } else {
            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = k * static_cast<uint32_t>(sizeof(int32_t));
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopyPad(indices[outBase], dstIndex, params);
            if (writeValues) {
                params.blockLen = k * static_cast<uint32_t>(sizeof(float));
                AscendC::DataCopyPad(values[outBase], dstValue, params);
            }
        }
        if constexpr (NATIVE_A8C8) {
            // The next row can reuse this buffer through Scalar, Vector or MTE2.
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void SelectRow(const MsaIndexScoreTilingData* __restrict tiling,
                                     const AscendC::GlobalTensor<float>& score, AscendC::GlobalTensor<int32_t>& indices,
                                     AscendC::GlobalTensor<float>& values, const AscendC::GlobalTensor<int32_t>& qlen,
                                     const AscendC::GlobalTensor<int32_t>& klen, uint32_t head, uint32_t token,
                                     bool writeValues)
    {
        int32_t qLen = 0;
        int32_t kvLen = 0;
        int32_t tOff = 0;
        if (rowMetaReady_) {
            qLen = qMeta_.GetValue(token);
            kvLen = kvMeta_.GetValue(token);
            tOff = tOffMeta_.GetValue(token);
        } else {
            LoadRequest(tiling, qlen, klen, token, qLen, kvLen, tOff);
        }
        const uint32_t valid = ValidBlocks(tiling, kvLen, qLen, tOff);
        const uint32_t k = tiling->topk;
        const uint64_t rowBase =
            static_cast<uint64_t>(head) * tiling->strideOutHead + static_cast<uint64_t>(token) * tiling->strideOutToken;
        const uint64_t outBase = (static_cast<uint64_t>(head) * tiling->totalQ + token) * k;
        nsel_ = 0;
        worst_ = 0;
        const uint32_t limit = (valid < k) ? valid : k;
        const uint32_t tailStart = (valid / MSA_TOPK_TRUNK) * MSA_TOPK_TRUNK;
        if (limit == MSA_TOPK_DENSE_K && k == MSA_TOPK_DENSE_K && valid == MSA_TOPK_DENSE_COLS && tailStart == 0U &&
            EmitDense256(score, indices, values, rowBase, outBase, tiling->strideOutToken, writeValues)) {
            return;
        }
        if (limit == k && EmitRadix16(score, indices, values, rowBase, outBase, valid, k, tiling->strideOutToken,
                                      tailStart, writeValues)) {
            return;
        }
        if (limit > 0U && tailStart == 0U &&
            EmitIndexOrder(score, indices, values, rowBase, outBase, valid, limit, k, tiling->strideOutToken,
                           writeValues)) {
            return;
        }
        if (limit > 0U) {
            SelectBySort(score, rowBase, valid, limit, tiling->strideOutToken);
        }
        WriteOutputs(indices, values, outBase, k, tailStart, writeValues);
    }

    MsaTopkCand sel_[MSA_TOPK_MAX];
    MsaTopkCand cand_[MSA_TOPK_MAX];
    AscendC::LocalTensor<float> ubTrunk_;
    AscendC::LocalTensor<int32_t> qUb_;
    AscendC::LocalTensor<int32_t> kUb_;
    AscendC::LocalTensor<int32_t> qMeta_;
    AscendC::LocalTensor<int32_t> kvMeta_;
    AscendC::LocalTensor<int32_t> tOffMeta_;
    uint32_t nsel_ = 0;
    uint32_t worst_ = 0;
    bool lensReady_ = false;
    bool rowMetaReady_ = false;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_TOPK_ARCH22_H
