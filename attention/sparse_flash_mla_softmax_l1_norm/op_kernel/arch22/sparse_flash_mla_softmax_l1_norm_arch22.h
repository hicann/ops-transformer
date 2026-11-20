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
 * \file sparse_flash_mla_softmax_l1_norm_arch22.h
 * \brief A2/A3 (arch22 / dav-c220) implementation.
 *
 *  Structure follows attention/sparse_flash_mla op_kernel/arch22:
 *    - Cube: Nd2Nz GM->L1 (Q, K, K gathered row by row for the sparse case), LoadData L1->L0A/L0B,
 *            Mmad, Fixpipe L0C->GM workspace, CrossCoreSetFlag<2, PIPE_FIX> hand-off to vector.
 *    - Vector: DataCopy GM->UB, p = exp(qk * scale - lse), column reduce over G, /G,
 *              and write the fp32 output. The two AIV sub-blocks own one 64-column half each, so both sides
 *              consume the score workspace before returning the MIX 1:2 slot credit.
 */
#ifndef SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_ARCH22_H
#define SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_ARCH22_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../sparse_flash_mla_softmax_l1_norm_tiling_data.h"

namespace SmlaL1Norm {
using namespace AscendC;

// D is selected by tiling. The buffers are sized for the largest supported
// value so the same MIX kernel covers D=256 and D=512 without a second binary.
constexpr int32_t kMaxHeadDim = 512;
constexpr int32_t kMinHeadDim = 256;
constexpr int32_t kS2NarrowBlock = 128;
constexpr int32_t kS2WideBlock = 256;
constexpr int32_t kS2MaxVecBlock = kS2WideBlock / 2;
constexpr int32_t kK0Size = 128;     // K chunk of one Mmad
constexpr int32_t kMaxG = 128;       // max num_heads_q
constexpr int32_t kMaxGHalf = 64;    // one AIV handles at most 64 heads
constexpr int32_t kWideM = 64;       // packed score rows for the dense wide path
constexpr int32_t kMaxPackRows = 64; // G=1 wide-path upper bound
constexpr uint32_t kSlots = 2;       // GM score ping-pong slots
constexpr int32_t kGatherRows = 16;  // sparse K rows staged by one AIV transfer group
constexpr int32_t kVecSparseMinRows = kS2NarrowBlock;

// cross core flag ids (mode 2 = 1 AIC : 2 AIV)
constexpr uint8_t kFlagStart = 0;  // AIV -> AIC, flushes initial slot credits
constexpr uint8_t kFlagReady = 1;  // 1,2  AIC -> AIV
constexpr uint8_t kFlagFree = 3;   // 3,4  AIV -> AIC
constexpr uint8_t kFlagKReady = 5; // 5,6  AIV -> AIC
constexpr uint8_t kFlagKFree = 7;  // 7,8  AIC -> AIV
constexpr uint8_t kSyncMode = 2;
constexpr IsResetLoad3dConfig kLoad3dCfg = {true, true};

// intra core event ids
constexpr int32_t kEvL1Q = 0;
constexpr int32_t kEvL1K = 1; // +slot
constexpr int32_t kEvL0 = EVENT_ID3;
constexpr int32_t kEvFix = 3;
constexpr int32_t kEvVec = 0;
constexpr int32_t kEvQk = 0;     // V <-> MTE2 dependency for qkUb_
constexpr int32_t kEvLse = 1;    // V <-> MTE2 dependency for lseTmpUb_
constexpr int32_t kEvGather = 2; // MTE3 <-> MTE2 dependency for gatherUb_

// memory footprint
constexpr uint32_t kQBytes = kMaxG * kMaxHeadDim * 2;        // 128KB
constexpr uint32_t kKBytes = kS2WideBlock * kMaxHeadDim * 2; // 256KB
constexpr uint32_t kL1Bytes = kQBytes + kKBytes;             // 384KB
constexpr uint32_t kL0ABytes = kMaxG * kK0Size * 2;          // 32KB
constexpr uint32_t kL0BBytes = kK0Size * kS2WideBlock * 2;   // 64KB
// Wide dense tasks use M=64,N=256; the generic path uses M=128,N=128.
constexpr uint32_t kL0CBytes = kWideM * kS2WideBlock * 4;           // 64KB
constexpr uint32_t kUbTileBytes = kWideM * kS2MaxVecBlock * 4;      // 32KB
constexpr uint32_t kUbAccBytes = kMaxPackRows * kS2MaxVecBlock * 4; // 32KB
constexpr uint32_t kUbLseBrcBytes = kMaxG * 8 * 4;                  // one 32B block per score row
constexpr uint32_t kUbZeroBytes = 8 * 1024;                         // 2048 floats
constexpr uint32_t kUbGatherBytes = kGatherRows * kMaxHeadDim * 2;  // 16KB

__aicore__ inline int64_t AlignUp16(int64_t x)
{
    return (x + 15) / 16 * 16;
}

struct RowInfo {
    int64_t bIdx = 0;
    int64_t s1Idx = 0;
    int64_t t1Off = 0;
    int64_t t2Off = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t rowsInGroup = 1;
};

template <typename T>
class SmlaL1NormKernel {
public:
    __aicore__ inline SmlaL1NormKernel() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR softmaxLse, GM_ADDR sparseIndices, GM_ADDR cuSeqlensQ,
                                GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK,
                                GM_ADDR topkLength, GM_ADDR metadata, GM_ADDR out, GM_ADDR workspace, TPipe* pipe,
                                const SmlaTilingPod* tiling);
    __aicore__ inline void Process();

private:
    // ---------- common ----------
    __aicore__ inline void ResolveRow(int64_t taskIdx, RowInfo& row) const;
    __aicore__ inline int64_t CalcS2Real(const RowInfo& row) const;
    __aicore__ inline int64_t QOffset(const RowInfo& row) const;
    __aicore__ inline int64_t KOffset(const RowInfo& row, int64_t kvIdx) const;
    __aicore__ inline int64_t SparseIdxOffset(const RowInfo& row) const;
    __aicore__ inline int64_t LseOffset(const RowInfo& row) const;
    __aicore__ inline int64_t OutOffset(const RowInfo& row) const;

    // ---------- cube ----------
    __aicore__ inline void InitCubeBuffers();
    __aicore__ inline void CubeProcess();
    __aicore__ inline void LoadQTile(int64_t offset, int64_t gSize);
    __aicore__ inline void LoadKTileDense(int64_t offset, int64_t rows);
    __aicore__ inline void LoadKTileMerged(int64_t offset, int64_t rows);
    __aicore__ inline void LoadKTileSparse(const RowInfo& row, int64_t ki, int64_t rows);
    __aicore__ inline void MmadBlock(int64_t gAlign);
    __aicore__ inline void LoadDataA(const LocalTensor<T>& l0a, int64_t kChunk, int64_t mSize);
    __aicore__ inline void LoadDataB(const LocalTensor<T>& l0b, int64_t kChunk, int64_t nSize);

    // ---------- vector ----------
    __aicore__ inline void InitVecBuffers();
    __aicore__ inline void VecProcess();
    __aicore__ inline void LoadLse(const RowInfo& row);
    __aicore__ inline bool UseVecSparse(int64_t rows) const;
    __aicore__ inline void GatherKTileSparse(const RowInfo& row, int64_t ki, int64_t rows, int64_t slot);
    __aicore__ inline void PrefetchKTileSparse(const RowInfo& row, int64_t ki, int64_t rows, int64_t slot);
    __aicore__ inline void PrefetchNextKTileSparse(const RowInfo& row, int64_t s2Real, int64_t ki, int64_t slot);
    __aicore__ inline void ReduceRows(const LocalTensor<float>& qk, int64_t scoreRows, int64_t qRows, int64_t colCount);
    __aicore__ inline void StoreRows(const LocalTensor<float>& acc, int64_t rowOutOff, int64_t ki, int64_t outLen,
                                     int64_t qRows);
    __aicore__ inline void ZeroTail(int64_t rowOutOff, int64_t from, int64_t to);

private:
    TPipe* pipe_ = nullptr;
    const SmlaTilingPod* tiling_ = nullptr;

    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<float> lseGm_;
    GlobalTensor<int32_t> sparseGm_;
    GlobalTensor<int32_t> cuQGm_;
    GlobalTensor<int32_t> cuKGm_;
    GlobalTensor<int32_t> seqQGm_;
    GlobalTensor<int32_t> seqKGm_;
    GlobalTensor<int32_t> cmpResGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> metadataGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<float> wsGm_;
    GlobalTensor<T> kMergeGm_;

    // cube side
    TBuf<TPosition::A1> bufL1_;
    LocalTensor<T> qL1_;
    LocalTensor<T> kL1_;
    TBuf<TPosition::A2> bufL0A_;
    LocalTensor<T> l0a_;
    TBuf<TPosition::B2> bufL0B_;
    LocalTensor<T> l0b_;
    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    // vector side
    TBuf<TPosition::VECCALC> bufQk_;
    LocalTensor<float> qkUb_;
    TBuf<TPosition::VECCALC> bufWork_;
    LocalTensor<float> workUb_;
    TBuf<TPosition::VECCALC> bufLse_;
    LocalTensor<float> lseRepUb_;
    TBuf<TPosition::VECCALC> bufAcc_;
    LocalTensor<float> accUb_;
    TBuf<TPosition::VECCALC> bufZero_;
    LocalTensor<float> zeroUb_;
    TBuf<TPosition::VECCALC> bufLseTmp_;
    LocalTensor<float> lseTmpUb_;
    TBuf<TPosition::VECCALC> bufGather_;
    LocalTensor<T> gatherUb_;

    int64_t coreIdx_ = 0;
    int64_t subIdx_ = 0;
    int64_t aivIdx_ = 0;
    int64_t wsSlotFloats_ = 0;
    int64_t wsCoreBase_ = 0;
    int64_t kMergeCoreBase_ = 0;
    int64_t kMergeSlotElems_ = 0;
    int64_t tileCols_ = kS2NarrowBlock;
    int64_t vecCols_ = kS2NarrowBlock / 2;
    bool isAic_ = false;
    bool useVecSparse_ = false;
};

/* ============================== common ============================== */

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR softmaxLse, GM_ADDR sparseIndices,
                                                 GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ,
                                                 GM_ADDR sequsedK, GM_ADDR cmpResidualK, GM_ADDR topkLength,
                                                 GM_ADDR metadata, GM_ADDR out, GM_ADDR workspace, TPipe* pipe,
                                                 const SmlaTilingPod* tiling)
{
    tiling_ = tiling;
    pipe_ = pipe;
    isAic_ = (g_coreType == AscendC::AIC);

    qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q));
    kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k));
    lseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmaxLse));
    sparseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseIndices));
    cuQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensQ));
    cuKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensK));
    seqQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequsedQ));
    seqKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequsedK));
    cmpResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cmpResidualK));
    topkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topkLength));
    metadataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(metadata));
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
    wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

    if (isAic_) {
        coreIdx_ = static_cast<int64_t>(GetBlockIdx());
        InitCubeBuffers();
    } else {
        aivIdx_ = static_cast<int64_t>(GetBlockIdx());
        coreIdx_ = aivIdx_ / static_cast<int64_t>(GetSubBlockNum());
        subIdx_ = static_cast<int64_t>(GetSubBlockIdx());
        InitVecBuffers();
    }
    wsSlotFloats_ = tiling_->wsSlotFloats;
    wsCoreBase_ = coreIdx_ * static_cast<int64_t>(kSlots) * wsSlotFloats_;
    const int64_t scoreRows = AlignUp16(tiling_->g * tiling_->packRows);
    tileCols_ = wsSlotFloats_ / scoreRows;
    vecCols_ = tileCols_ / 2;
    useVecSparse_ = (tiling_->isSparse == 1);
    kMergeSlotElems_ = tileCols_ * tiling_->d;
    kMergeCoreBase_ = coreIdx_ * static_cast<int64_t>(kSlots) * kMergeSlotElems_;
    const int64_t scoreWorkspaceBytes =
        tiling_->numCores * static_cast<int64_t>(kSlots) * wsSlotFloats_ * static_cast<int64_t>(sizeof(float));
    kMergeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + scoreWorkspaceBytes / sizeof(T));
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::ResolveRow(int64_t taskIdx, RowInfo& row) const
{
    if (tiling_->isTnd == 0) {
        const int64_t groupsPerBatch = (tiling_->sq + tiling_->packRows - 1) / tiling_->packRows;
        row.bIdx = taskIdx / groupsPerBatch;
        row.s1Idx = (taskIdx % groupsPerBatch) * tiling_->packRows;
        row.s1Size = tiling_->sq;
        row.s2Size = tiling_->sk;
        row.rowsInGroup = tiling_->sq - row.s1Idx;
        if (row.rowsInGroup > tiling_->packRows) {
            row.rowsInGroup = tiling_->packRows;
        }
        return;
    }
    // TND: locate batch by scanning cu_seqlens_q
    int64_t b = 0;
    for (; b + 1 <= tiling_->b; ++b) {
        int64_t start = static_cast<int64_t>(cuQGm_.GetValue(static_cast<int64_t>(b)));
        int64_t end = static_cast<int64_t>(cuQGm_.GetValue(static_cast<int64_t>(b) + 1));
        if (taskIdx < end || b == tiling_->b - 1) {
            row.bIdx = b;
            row.s1Idx = taskIdx - start;
            row.t1Off = start;
            row.t2Off = static_cast<int64_t>(cuKGm_.GetValue(static_cast<int64_t>(b)));
            row.s1Size = end - start;
            row.s2Size = static_cast<int64_t>(cuKGm_.GetValue(static_cast<int64_t>(b) + 1)) - row.t2Off;
            row.rowsInGroup = 1;
            return;
        }
    }
}

template <typename T>
__aicore__ inline int64_t SmlaL1NormKernel<T>::CalcS2Real(const RowInfo& row) const
{
    int64_t curS1 = tiling_->hasSequsedQ ? static_cast<int64_t>(seqQGm_.GetValue(row.bIdx)) : row.s1Size;
    int64_t curS2 = tiling_->hasSequsedK ? static_cast<int64_t>(seqKGm_.GetValue(row.bIdx)) : row.s2Size;
    if (row.s1Idx >= curS1) {
        return 0;
    }
    int64_t s2Valid = curS2;
    if (tiling_->maskMode != 0) {
        int64_t oriS2 = curS2 * tiling_->cmpRatio;
        if (tiling_->cmpRatio > 1) {
            oriS2 += static_cast<int64_t>(cmpResGm_.GetValue(row.bIdx));
        }
        const int64_t qRef = (tiling_->isTnd == 1) ? row.s1Size : curS1;
        s2Valid = (oriS2 - qRef + row.s1Idx + 1) / tiling_->cmpRatio;
        if (s2Valid < 0) {
            s2Valid = 0;
        }
    }
    if (s2Valid > curS2) {
        s2Valid = curS2;
    }
    if (tiling_->isSparse == 0) {
        return s2Valid;
    }
    int64_t curKLen = tiling_->kLength;
    if (tiling_->hasTopkLength) {
        int64_t idx = (tiling_->isTnd == 0) ? (row.bIdx * tiling_->sq + row.s1Idx) : (row.t1Off + row.s1Idx);
        int64_t topkLen = static_cast<int64_t>(topkGm_.GetValue(idx));
        if (topkLen < curKLen) {
            curKLen = topkLen;
        }
    }
    return (curKLen < s2Valid) ? curKLen : s2Valid;
}

template <typename T>
__aicore__ inline int64_t SmlaL1NormKernel<T>::QOffset(const RowInfo& row) const
{
    if (tiling_->isTnd == 1) {
        return (row.t1Off + row.s1Idx) * tiling_->g * tiling_->d;
    }
    return (row.bIdx * tiling_->sq + row.s1Idx) * tiling_->g * tiling_->d;
}

template <typename T>
__aicore__ inline int64_t SmlaL1NormKernel<T>::KOffset(const RowInfo& row, int64_t kvIdx) const
{
    if (tiling_->isTnd == 1) {
        return (row.t2Off + kvIdx) * tiling_->d;
    }
    return (row.bIdx * tiling_->sk + kvIdx) * tiling_->d;
}

template <typename T>
__aicore__ inline int64_t SmlaL1NormKernel<T>::SparseIdxOffset(const RowInfo& row) const
{
    if (tiling_->isTnd == 1) {
        return (row.t1Off + row.s1Idx) * tiling_->kLength;
    }
    return (row.bIdx * tiling_->sq + row.s1Idx) * tiling_->kLength;
}

template <typename T>
__aicore__ inline int64_t SmlaL1NormKernel<T>::LseOffset(const RowInfo& row) const
{
    if (tiling_->isTnd == 1) {
        return (row.t1Off + row.s1Idx) * tiling_->g;
    }
    return (row.bIdx * tiling_->sq + row.s1Idx) * tiling_->g;
}

template <typename T>
__aicore__ inline int64_t SmlaL1NormKernel<T>::OutOffset(const RowInfo& row) const
{
    if (tiling_->isTnd == 1) {
        return (row.t1Off + row.s1Idx) * tiling_->outRowLen;
    }
    return (row.bIdx * tiling_->sq + row.s1Idx) * tiling_->outRowLen;
}

/* ============================== cube ============================== */

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::InitCubeBuffers()
{
    pipe_->InitBuffer(bufL1_, kL1Bytes);
    qL1_ = bufL1_.Get<T>();
    kL1_ = qL1_[kQBytes / sizeof(T)];
    pipe_->InitBuffer(bufL0A_, kL0ABytes);
    l0a_ = bufL0A_.Get<T>();
    pipe_->InitBuffer(bufL0B_, kL0BBytes);
    l0b_ = bufL0B_.Get<T>();
    pipe_->InitBuffer(bufL0C_, kL0CBytes);
    cL0_ = bufL0C_.Get<float>();

    // pre-arm the mte1->mte2 flags of the L1 buffers
    SetFlag<HardEvent::MTE1_MTE2>(kEvL1Q);
    SetFlag<HardEvent::MTE1_MTE2>(kEvL1K);
    SetFlag<HardEvent::M_MTE1>(kEvL0);
    SetFlag<HardEvent::FIX_M>(kEvFix);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::LoadQTile(int64_t offset, int64_t gSize)
{
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = static_cast<uint16_t>(gSize);
    p.dValue = static_cast<uint16_t>(tiling_->d);
    p.srcDValue = static_cast<uint16_t>(tiling_->d);
    p.dstNzC0Stride = static_cast<uint16_t>(AlignUp16(gSize));
    p.dstNzNStride = 1;
    p.srcNdMatrixStride = 0;
    p.dstNzMatrixStride = 0;
    DataCopy(qL1_, qGm_[offset], p);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::LoadKTileDense(int64_t offset, int64_t rows)
{
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = static_cast<uint16_t>(rows);
    p.dValue = static_cast<uint16_t>(tiling_->d);
    p.srcDValue = static_cast<uint16_t>(tiling_->d);
    p.dstNzC0Stride = static_cast<uint16_t>(tileCols_);
    p.dstNzNStride = 1;
    p.srcNdMatrixStride = 0;
    p.dstNzMatrixStride = 0;
    DataCopy(kL1_, kGm_[offset], p);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::LoadKTileMerged(int64_t offset, int64_t rows)
{
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = static_cast<uint16_t>(rows);
    p.dValue = static_cast<uint16_t>(tiling_->d);
    p.srcDValue = static_cast<uint16_t>(tiling_->d);
    p.dstNzC0Stride = static_cast<uint16_t>(tileCols_);
    p.dstNzNStride = 1;
    p.srcNdMatrixStride = 0;
    p.dstNzMatrixStride = 0;
    DataCopy(kL1_, kMergeGm_[offset], p);
}

// Sparse case: gather K rows one run of consecutive indices at a time. In NZ layout,
// row r starts at element r * 16, so a run [r, r+run) is one Nd2Nz copy.
template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::LoadKTileSparse(const RowInfo& row, int64_t ki, int64_t rows)
{
    const int64_t idxBase = SparseIdxOffset(row) + ki;
    const int64_t kBase = KOffset(row, 0);
    LocalTensor<T> dst = kL1_;
    int64_t r = 0;
    while (r < rows) {
        int32_t idx = sparseGm_.GetValue(idxBase + r);
        if (idx < 0) {
            break;
        }
        int64_t run = 1;
        while (r + run < rows) {
            int32_t nxt = sparseGm_.GetValue(idxBase + r + run);
            if (nxt < 0 || nxt != idx + static_cast<int32_t>(run)) {
                break;
            }
            ++run;
        }
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = static_cast<uint16_t>(run);
        p.dValue = static_cast<uint16_t>(tiling_->d);
        p.srcDValue = static_cast<uint16_t>(tiling_->d);
        p.dstNzC0Stride = static_cast<uint16_t>(tileCols_);
        p.dstNzNStride = 1;
        p.srcNdMatrixStride = 0;
        p.dstNzMatrixStride = 0;
        DataCopy(dst[r * 16], kGm_[kBase + static_cast<int64_t>(idx) * tiling_->d], p);
        r += run;
    }
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::LoadDataA(const LocalTensor<T>& l0a, int64_t kChunk, int64_t mSize)
{
    LocalTensor<T> src = qL1_[mSize * kK0Size * kChunk];
    LoadData3DParamsV2<T> params;
    params.l1H = static_cast<uint16_t>(mSize / 16);
    params.l1W = 16;
    params.padList[0] = 0;
    params.padList[1] = 0;
    params.padList[2] = 0;
    params.padList[3] = 255;
    params.mExtension = static_cast<uint16_t>(mSize);
    params.kExtension = static_cast<uint16_t>(kK0Size);
    params.mStartPt = 0;
    params.kStartPt = 0;
    params.strideW = 1;
    params.strideH = 1;
    params.filterW = 1;
    params.filterSizeW = 0;
    params.filterH = 1;
    params.filterSizeH = 0;
    params.dilationFilterW = 1;
    params.dilationFilterH = 1;
    params.enTranspose = 0;
    params.fMatrixCtrl = 0;
    params.channelSize = static_cast<uint16_t>(kK0Size);
    LoadData<T, kLoad3dCfg>(l0a, src, params);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::LoadDataB(const LocalTensor<T>& l0b, int64_t kChunk, int64_t nSize)
{
    // K was copied as an [N, K] matrix, so its L1 NZ image is the NK path.
    // Each K chunk is a contiguous run of (N/16)*(K/16) fractals and maps
    // directly to L0B ZN without a per-fractal transpose.
    LocalTensor<T> src = kL1_[(kChunk * kK0Size / 16) * (tileCols_ * 16)];
    LoadData2DParams params;
    params.startIndex = 0;
    params.repeatTimes = static_cast<uint16_t>((nSize / 16) * (kK0Size / 16));
    params.srcStride = 1;
    params.dstGap = 0;
    params.ifTranspose = false;
    LoadData(l0b, src, params);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::MmadBlock(int64_t gAlign)
{
    const int64_t kLoops = tiling_->d / kK0Size;
    for (int64_t kk = 0; kk < kLoops; ++kk) {
        LocalTensor<T> aL0 = l0a_;
        LocalTensor<T> bL0 = l0b_;
        WaitFlag<HardEvent::M_MTE1>(kEvL0);
        LoadDataA(aL0, kk, gAlign);

        LoadDataB(bL0, kk, tileCols_);

        SetFlag<HardEvent::MTE1_M>(kEvL0);
        WaitFlag<HardEvent::MTE1_M>(kEvL0);

        MmadParams mmadParams;
        mmadParams.m = static_cast<uint16_t>(gAlign);
        mmadParams.n = static_cast<uint16_t>(tileCols_);
        mmadParams.k = static_cast<uint16_t>(kK0Size);
        mmadParams.cmatrixInitVal = (kk == 0);
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0b10;
        Mmad(cL0_, aL0, bL0, mmadParams);
        // Retire the MMAD before returning the M -> MTE1 credit for the L0 buffers.
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(kEvL0);
    }
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::CubeProcess()
{
    if (useVecSparse_) {
        CrossCoreSetFlag<kSyncMode, PIPE_MTE2>(kFlagKFree);
        CrossCoreSetFlag<kSyncMode, PIPE_MTE2>(kFlagKFree + 1);
    }
    AscendC::WaitEvent(kFlagStart);
    const int64_t taskRounds = (tiling_->totalRows + tiling_->numCores - 1) / tiling_->numCores;
    const int64_t blockRounds = (tiling_->outRowLen + tileCols_ - 1) / tileCols_;
    for (int64_t round = 0; round < taskRounds; ++round) {
        const int64_t task = coreIdx_ + round * tiling_->numCores;
        RowInfo row;
        int64_t s2Real = 0;
        if (task < tiling_->totalRows) {
            ResolveRow(task, row);
            s2Real = CalcS2Real(row);
        }
        const int64_t scoreRows = row.rowsInGroup * tiling_->g;
        const int64_t gAlign = AlignUp16(scoreRows);

        if (s2Real > 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(kEvL1Q);
            LoadQTile(QOffset(row), scoreRows);
            SetFlag<HardEvent::MTE2_MTE1>(kEvL1Q);
            WaitFlag<HardEvent::MTE2_MTE1>(kEvL1Q);
        }
        for (int64_t block = 0; block < blockRounds; ++block) {
            const int64_t ki = block * tileCols_;
            const int64_t slot = block & 1;
            AscendC::WaitEvent(kFlagFree + slot);
            if (ki < s2Real) {
                int64_t rows = s2Real - ki;
                if (rows > tileCols_) {
                    rows = tileCols_;
                }
                const bool vecSparse = UseVecSparse(s2Real);
                if (vecSparse) {
                    AscendC::WaitEvent(kFlagKReady + slot);
                }
                WaitFlag<HardEvent::MTE1_MTE2>(kEvL1K);
                if (vecSparse) {
                    LoadKTileMerged(kMergeCoreBase_ + slot * kMergeSlotElems_, rows);
                } else if (tiling_->isSparse == 1) {
                    LoadKTileSparse(row, ki, rows);
                } else {
                    LoadKTileDense(KOffset(row, ki), rows);
                }
                SetFlag<HardEvent::MTE2_MTE1>(kEvL1K);
                if (vecSparse) {
                    CrossCoreSetFlag<kSyncMode, PIPE_MTE2>(kFlagKFree + slot);
                }
                WaitFlag<HardEvent::MTE2_MTE1>(kEvL1K);

                WaitFlag<HardEvent::FIX_M>(kEvFix);
                MmadBlock(gAlign);

                FixpipeParamsV220 fixParams;
                fixParams.nSize = static_cast<uint16_t>(tileCols_);
                fixParams.mSize = static_cast<uint16_t>(gAlign);
                fixParams.srcStride = static_cast<uint16_t>(gAlign);
                fixParams.dstStride = static_cast<uint16_t>(tileCols_);
                fixParams.unitFlag = 0b00;
                fixParams.ndNum = 1;
                SetFlag<HardEvent::M_FIX>(kEvFix);
                WaitFlag<HardEvent::M_FIX>(kEvFix);
                Fixpipe(wsGm_[wsCoreBase_ + slot * wsSlotFloats_], cL0_, fixParams);
                SetFlag<HardEvent::FIX_M>(kEvFix);
                SetFlag<HardEvent::MTE1_MTE2>(kEvL1K);
            }
            CrossCoreSetFlag<kSyncMode, PIPE_FIX>(kFlagReady + slot);
        }
        if (s2Real > 0) {
            SetFlag<HardEvent::MTE1_MTE2>(kEvL1Q);
        }
    }
    // drain the intra-core flags left armed by the last iteration so the next kernel starts clean
    WaitFlag<HardEvent::MTE1_MTE2>(kEvL1Q);
    WaitFlag<HardEvent::MTE1_MTE2>(kEvL1K);
    WaitFlag<HardEvent::M_MTE1>(kEvL0);
    WaitFlag<HardEvent::FIX_M>(kEvFix);
    PipeBarrier<PIPE_FIX>();
}

/* ============================== vector ============================== */

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::InitVecBuffers()
{
    pipe_->InitBuffer(bufQk_, kUbTileBytes);
    qkUb_ = bufQk_.Get<float>();
    pipe_->InitBuffer(bufWork_, kUbTileBytes);
    workUb_ = bufWork_.Get<float>();
    pipe_->InitBuffer(bufLse_, kUbLseBrcBytes);
    lseRepUb_ = bufLse_.Get<float>();
    pipe_->InitBuffer(bufAcc_, kUbAccBytes);
    accUb_ = bufAcc_.Get<float>();
    pipe_->InitBuffer(bufZero_, kUbZeroBytes);
    zeroUb_ = bufZero_.Get<float>();
    pipe_->InitBuffer(bufLseTmp_, 256 * sizeof(float));
    lseTmpUb_ = bufLseTmp_.Get<float>();
    pipe_->InitBuffer(bufGather_, kUbGatherBytes);
    gatherUb_ = bufGather_.Get<T>();
    Duplicate(zeroUb_, 0.0f, static_cast<int32_t>(kUbZeroBytes / sizeof(float)));
    SetFlag<HardEvent::V_MTE3>(kEvVec);
    WaitFlag<HardEvent::V_MTE3>(kEvVec);
    // Initial credits let the first MTE2 writes proceed. Each V consumer
    // returns its credit only after it has finished reading the same UB.
    SetFlag<HardEvent::V_MTE2>(kEvQk);
    SetFlag<HardEvent::V_MTE2>(kEvLse);
    SetFlag<HardEvent::MTE3_MTE2>(kEvGather);
}

template <typename T>
__aicore__ inline bool SmlaL1NormKernel<T>::UseVecSparse(int64_t rows) const
{
    return useVecSparse_ && rows >= kVecSparseMinRows;
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::GatherKTileSparse(const RowInfo& row, int64_t ki, int64_t rows,
                                                              int64_t slot)
{
    const int64_t split = (rows + 1) / 2;
    const int64_t begin = (subIdx_ == 0) ? 0 : split;
    const int64_t end = (subIdx_ == 0) ? split : rows;
    const int64_t kBase = KOffset(row, 0);
    for (int64_t first = begin; first < end; first += kGatherRows) {
        int64_t chunkRows = end - first;
        if (chunkRows > kGatherRows) {
            chunkRows = kGatherRows;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(kEvGather);
        const int64_t idxBase = SparseIdxOffset(row) + ki + first;
        int64_t localRow = 0;
        while (localRow < chunkRows) {
            int32_t idx = sparseGm_.GetValue(idxBase + localRow);
            if (idx < 0) {
                break;
            }
            int64_t run = 1;
            while (localRow + run < chunkRows) {
                int32_t next = sparseGm_.GetValue(idxBase + localRow + run);
                if (next < 0 || next != idx + static_cast<int32_t>(run)) {
                    break;
                }
                ++run;
            }
            DataCopyExtParams copyIn;
            copyIn.blockCount = 1;
            copyIn.blockLen = static_cast<uint32_t>(run * tiling_->d * sizeof(T));
            copyIn.srcStride = 0;
            copyIn.dstStride = 0;
            copyIn.rsv = 0;
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(gatherUb_[localRow * tiling_->d], kGm_[kBase + static_cast<int64_t>(idx) * tiling_->d], copyIn,
                        pad);
            localRow += run;
        }
        SetFlag<HardEvent::MTE2_MTE3>(kEvGather);
        WaitFlag<HardEvent::MTE2_MTE3>(kEvGather);
        DataCopyExtParams copyOut;
        copyOut.blockCount = 1;
        copyOut.blockLen = static_cast<uint32_t>(chunkRows * tiling_->d * sizeof(T));
        copyOut.srcStride = 0;
        copyOut.dstStride = 0;
        copyOut.rsv = 0;
        DataCopyPad(kMergeGm_[kMergeCoreBase_ + slot * kMergeSlotElems_ + first * tiling_->d], gatherUb_, copyOut);
        SetFlag<HardEvent::MTE3_MTE2>(kEvGather);
    }
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::PrefetchKTileSparse(const RowInfo& row, int64_t ki, int64_t rows,
                                                                int64_t slot)
{
    AscendC::WaitEvent(kFlagKFree + slot);
    GatherKTileSparse(row, ki, rows, slot);
    CrossCoreSetFlag<kSyncMode, PIPE_MTE3>(kFlagKReady + slot);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::PrefetchNextKTileSparse(const RowInfo& row, int64_t s2Real, int64_t ki,
                                                                    int64_t slot)
{
    const int64_t nextKi = ki + tileCols_;
    if (!UseVecSparse(s2Real) || nextKi >= s2Real) {
        return;
    }
    int64_t nextRows = s2Real - nextKi;
    if (nextRows > tileCols_) {
        nextRows = tileCols_;
    }
    PrefetchKTileSparse(row, nextKi, nextRows, slot ^ 1);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::LoadLse(const RowInfo& row)
{
    // Packed BSND tasks keep consecutive Q rows and their LSE values contiguous.
    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint32_t>(row.rowsInGroup * tiling_->g * sizeof(float));
    cp.srcStride = 0;
    cp.dstStride = 0;
    cp.rsv = 0;
    DataCopyPadExtParams<float> pad;
    pad.isPad = false;
    pad.leftPadding = 0;
    pad.rightPadding = 0;
    pad.paddingValue = 0;
    WaitFlag<HardEvent::V_MTE2>(kEvLse);
    DataCopyPad(lseTmpUb_, lseGm_[LseOffset(row)], cp, pad);
    SetFlag<HardEvent::MTE2_V>(kEvLse);
    WaitFlag<HardEvent::MTE2_V>(kEvLse);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::ReduceRows(const LocalTensor<float>& qk, int64_t scoreRows, int64_t qRows,
                                                       int64_t colCount)
{
    if (scoreRows <= 0 || qRows <= 0) {
        return;
    }
    const int32_t cnt = static_cast<int32_t>(scoreRows * colCount);
    Muls(workUb_, qk, tiling_->softmaxScale, cnt);
    // qk is no longer needed after Muls, so the next tile can be staged while
    // the remaining reductions continue in workUb_.
    SetFlag<HardEvent::V_MTE2>(kEvQk);
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams lseParams;
    lseParams.dstBlkStride = 1;
    lseParams.src0BlkStride = 1;
    lseParams.src1BlkStride = 0;
    lseParams.dstRepStride = static_cast<uint8_t>(colCount / 8);
    lseParams.src0RepStride = static_cast<uint8_t>(colCount / 8);
    lseParams.src1RepStride = 1;
    for (int64_t col = 0; col < colCount; col += 64) {
        Sub(workUb_[col], workUb_[col], lseRepUb_, 64, static_cast<uint8_t>(scoreRows), lseParams);
    }
    PipeBarrier<PIPE_V>();
    Exp(workUb_, workUb_, cnt);
    PipeBarrier<PIPE_V>();
    if (tiling_->g == 1) {
        Adds(accUb_, workUb_, 0.0f, static_cast<int32_t>(qRows * colCount));
        PipeBarrier<PIPE_V>();
        return;
    }

    // Reduce G as a tree. Independent pairs at one level can be issued together, so the
    // dependency depth is ceil(log2(G)) instead of G - 1. The strided form also handles
    // non-power-of-two group counts used by the generic path.
    const bool powerOfTwoG = (tiling_->g & (tiling_->g - 1)) == 0;
    for (int64_t stride = 1; stride < tiling_->g; stride <<= 1) {
        const bool finalLevel = (stride << 1) >= tiling_->g;
        const int64_t repeatStride = (stride << 1) * colCount / 8;
        if (powerOfTwoG && finalLevel && repeatStride <= 255) {
            // The final pair of every Q row lands in a dense output tile. Collapse the
            // per-row loop into one repeat command while gathering source rows at G stride.
            const uint8_t repeatTimes = static_cast<uint8_t>(qRows);
            BinaryRepeatParams params(1, 1, 1, static_cast<uint8_t>(colCount / 8), static_cast<uint8_t>(repeatStride),
                                      static_cast<uint8_t>(repeatStride));
            for (int64_t col = 0; col < colCount; col += 64) {
                Add(accUb_[col], workUb_[col], workUb_[stride * colCount + col], 64, repeatTimes, params);
            }
        } else if (powerOfTwoG && !finalLevel && repeatStride <= 255) {
            const uint8_t repeatTimes = static_cast<uint8_t>(qRows * tiling_->g / (stride << 1));
            BinaryRepeatParams params(1, 1, 1, static_cast<uint8_t>(repeatStride), static_cast<uint8_t>(repeatStride),
                                      static_cast<uint8_t>(repeatStride));
            for (int64_t col = 0; col < colCount; col += 64) {
                Add(workUb_[col], workUb_[col], workUb_[stride * colCount + col], 64, repeatTimes, params);
            }
        } else {
            for (int64_t qRow = 0; qRow < qRows; ++qRow) {
                const int64_t scoreBase = qRow * tiling_->g * colCount;
                for (int64_t g = 0; g + stride < tiling_->g; g += stride << 1) {
                    LocalTensor<float> dst = finalLevel ? accUb_[qRow * colCount] : workUb_[scoreBase + g * colCount];
                    Add(dst, workUb_[scoreBase + g * colCount], workUb_[scoreBase + (g + stride) * colCount],
                        static_cast<int32_t>(colCount));
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::StoreRows(const LocalTensor<float>& acc, int64_t rowOutOff, int64_t ki,
                                                      int64_t outLen, int64_t qRows)
{
    Muls(acc, acc, 1.0f / static_cast<float>(tiling_->g), static_cast<int32_t>(qRows * vecCols_));
    DataCopyExtParams cp;
    cp.blockLen = static_cast<uint32_t>(outLen * sizeof(float));
    cp.rsv = 0;
    SetFlag<HardEvent::V_MTE3>(kEvVec);
    WaitFlag<HardEvent::V_MTE3>(kEvVec);
    if (outLen == vecCols_) {
        cp.blockCount = static_cast<uint16_t>(qRows);
        cp.srcStride = 0;
        cp.dstStride = static_cast<uint32_t>((tiling_->outRowLen - outLen) * sizeof(float));
        DataCopyPad(outGm_[rowOutOff + ki], acc, cp);
    } else {
        // UB row gaps in a partial tile need not be 32-byte aligned, so issue the tail rows separately.
        cp.blockCount = 1;
        cp.srcStride = 0;
        cp.dstStride = 0;
        for (int64_t qRow = 0; qRow < qRows; ++qRow) {
            DataCopyPad(outGm_[rowOutOff + qRow * tiling_->outRowLen + ki], acc[qRow * vecCols_], cp);
        }
    }
    // The next score block rebuilds accUb_; wait until MTE3 has consumed it.
    SetFlag<HardEvent::MTE3_V>(kEvVec);
    WaitFlag<HardEvent::MTE3_V>(kEvVec);
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::ZeroTail(int64_t rowOutOff, int64_t from, int64_t to)
{
    constexpr int64_t kChunk = static_cast<int64_t>(kUbZeroBytes / sizeof(float));
    for (int64_t off = from; off < to; off += kChunk) {
        int64_t cnt = to - off;
        if (cnt > kChunk) {
            cnt = kChunk;
        }
        DataCopyExtParams cp;
        cp.blockCount = 1;
        cp.blockLen = static_cast<uint32_t>(cnt * sizeof(float));
        cp.srcStride = 0;
        cp.dstStride = 0;
        cp.rsv = 0;
        SetFlag<HardEvent::V_MTE3>(kEvVec);
        WaitFlag<HardEvent::V_MTE3>(kEvVec);
        DataCopyPad(outGm_[rowOutOff + off], zeroUb_, cp);
    }
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::VecProcess()
{
    // The first two score slots are initially available to the cube side.
    CrossCoreSetFlag<kSyncMode, PIPE_V>(kFlagFree);
    CrossCoreSetFlag<kSyncMode, PIPE_V>(kFlagFree + 1);
    CrossCoreSetFlag<kSyncMode, PIPE_V>(kFlagStart);

    const int64_t taskRounds = (tiling_->totalRows + tiling_->numCores - 1) / tiling_->numCores;
    const int64_t blockRounds = (tiling_->outRowLen + tileCols_ - 1) / tileCols_;
    for (int64_t round = 0; round < taskRounds; ++round) {
        const int64_t task = coreIdx_ + round * tiling_->numCores;
        RowInfo row;
        int64_t s2Real = 0;
        if (task < tiling_->totalRows) {
            ResolveRow(task, row);
            s2Real = CalcS2Real(row);
        }
        const int64_t rowOutOff = task < tiling_->totalRows ? OutOffset(row) : 0;
        const int64_t qRows = row.rowsInGroup;
        const int64_t scoreRows = qRows * tiling_->g;
        const bool vecSparse = UseVecSparse(s2Real);
        if (vecSparse) {
            int64_t firstRows = s2Real;
            if (firstRows > tileCols_) {
                firstRows = tileCols_;
            }
            PrefetchKTileSparse(row, 0, firstRows, 0);
        }
        if (s2Real > 0) {
            LoadLse(row);
            // One 32-byte block per score row; Sub repeats that block across all K columns.
            Brcb(lseRepUb_, lseTmpUb_, static_cast<uint8_t>((scoreRows + 7) / 8), {1, 8});
            SetFlag<HardEvent::V_MTE2>(kEvLse);
            PipeBarrier<PIPE_V>();
        }
        for (int64_t block = 0; block < blockRounds; ++block) {
            const int64_t ki = block * tileCols_;
            const int64_t slot = block & 1;
            const int64_t vecKi = ki + subIdx_ * vecCols_;
            int64_t vecEnd = vecKi + vecCols_;
            if (vecEnd > tiling_->outRowLen) {
                vecEnd = tiling_->outRowLen;
            }
            AscendC::WaitEvent(kFlagReady + slot);
            if (vecKi < s2Real && vecKi < vecEnd) {
                int64_t outLen = s2Real - vecKi;
                if (outLen > vecCols_) {
                    outLen = vecCols_;
                }
                LocalTensor<float> qk = qkUb_;
                DataCopyExtParams scoreCopy;
                scoreCopy.blockCount = static_cast<uint16_t>(scoreRows);
                scoreCopy.blockLen = static_cast<uint32_t>(vecCols_ * sizeof(float));
                scoreCopy.srcStride = static_cast<uint32_t>((tileCols_ - vecCols_) * sizeof(float));
                scoreCopy.dstStride = 0;
                scoreCopy.rsv = 0;
                DataCopyPadExtParams<float> scorePad;
                scorePad.isPad = false;
                scorePad.leftPadding = 0;
                scorePad.rightPadding = 0;
                scorePad.paddingValue = 0.0f;
                WaitFlag<HardEvent::V_MTE2>(kEvQk);
                DataCopyPad(qk, wsGm_[wsCoreBase_ + slot * wsSlotFloats_ + subIdx_ * vecCols_], scoreCopy, scorePad);
                SetFlag<HardEvent::MTE2_V>(kEvQk);
                WaitFlag<HardEvent::MTE2_V>(kEvQk);
                // The score tile is private in UB now, so Cube can reuse the GM slot while Vector computes.
                CrossCoreSetFlag<kSyncMode, PIPE_V>(kFlagFree + slot);
                PrefetchNextKTileSparse(row, s2Real, ki, slot);
                ReduceRows(qk, scoreRows, qRows, vecCols_);
                StoreRows(accUb_, rowOutOff, vecKi, outLen, qRows);
                for (int64_t qRow = 0; qRow < qRows; ++qRow) {
                    const int64_t qRowOutOff = rowOutOff + qRow * tiling_->outRowLen;
                    if (vecKi + outLen < vecEnd) {
                        ZeroTail(qRowOutOff, vecKi + outLen, vecEnd);
                    }
                }
            } else if (task < tiling_->totalRows && vecKi < vecEnd) {
                CrossCoreSetFlag<kSyncMode, PIPE_V>(kFlagFree + slot);
                PrefetchNextKTileSparse(row, s2Real, ki, slot);
                for (int64_t qRow = 0; qRow < qRows; ++qRow) {
                    ZeroTail(rowOutOff + qRow * tiling_->outRowLen, vecKi, vecEnd);
                }
            } else {
                CrossCoreSetFlag<kSyncMode, PIPE_V>(kFlagFree + slot);
                PrefetchNextKTileSparse(row, s2Real, ki, slot);
            }
        }
    }
    // Drain the final credits so no intra-core event remains armed at exit.
    WaitFlag<HardEvent::V_MTE2>(kEvQk);
    WaitFlag<HardEvent::V_MTE2>(kEvLse);
    WaitFlag<HardEvent::MTE3_MTE2>(kEvGather);
    if (useVecSparse_) {
        AscendC::WaitEvent(kFlagKFree);
        AscendC::WaitEvent(kFlagKFree + 1);
    }
    PipeBarrier<PIPE_MTE3>();
}

template <typename T>
__aicore__ inline void SmlaL1NormKernel<T>::Process()
{
    if (isAic_) {
        CubeProcess();
    } else {
        VecProcess();
    }
}

} // namespace SmlaL1Norm

#endif // SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_ARCH22_H
