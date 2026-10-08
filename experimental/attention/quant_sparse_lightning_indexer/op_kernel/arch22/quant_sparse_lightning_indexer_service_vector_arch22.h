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
 * \file quant_sparse_lightning_indexer_service_vector_arch22.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_sparse_lightning_indexer_common_arch22.h"
#include "quant_sparse_lightning_indexer_vector.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace QLIV2ServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr uint32_t ELE_NUM_32 = 32;
constexpr uint32_t ELE_NUM_128 = 128;
constexpr uint32_t ELE_NUM_512 = 512;

template <typename QLIV2T>
class QLIV2Vector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline QLIV2Vector(){};
    __aicore__ inline void ProcessVec0(const QLIV2Common::RunInfo& info);
    __aicore__ inline void ProcessVec1(const QLIV2Common::RunInfo& info);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitParams(const struct QLIV2Common::ConstInfo& constInfo,
                                      const struct QLIV2Common::LdSplitCoreInfo& ldInfo,
                                      const QLIV2TilingDataConsumer* __restrict tilingData);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm, GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                  GlobalTensor<float> vec1ResGm);
    __aicore__ inline void InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                  GlobalTensor<int32_t> outputIdxOffsetGm, bool outputIdxOffsetValid);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                              GlobalTensor<half> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline int32_t AlignS2(int32_t cuS2Len);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe* pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<half> vec0OutGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidateTopkIndexInGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================

private:
    __aicore__ inline void GetKeyScale(const QLIV2Common::RunInfo& runInfo, const LocalTensor<half>& resUb,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // candidate (two-level topk)
    __aicore__ inline int32_t CountGE(const LocalTensor<float>& sortedDesc, int32_t n, float x);
    __aicore__ inline void BuildCandidateMask(const QLIV2Common::RunInfo& info, int32_t cuS1Idx, int32_t cuBaseS2Idx,
                                              int32_t innerS1Idx);
    // candidate 新增向量 op 的 64 元素分块封装 (本版本 Level-2 count > 64 的 vmax 等指令会触发 aicore 异常)
    __aicore__ inline void VecMaxPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, int32_t n);
    __aicore__ inline void VecMinPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, int32_t n);
    __aicore__ inline void VecAbs(const LocalTensor<float>& dst, const LocalTensor<float>& src, int32_t n);
    __aicore__ inline void VecMinsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src, float v,
                                         int32_t n);
    __aicore__ inline void VecMaxsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src, float v,
                                         int32_t n);
    __aicore__ inline void VecSubPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, int32_t n);
    __aicore__ inline void VecAddsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src, float v,
                                         int32_t n);
    __aicore__ inline void VecMulPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, int32_t n);
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    // tmp buff for candidate (two-level topk)
    TBuf<TPosition::VECCALC> candBuf_;      // consumer: 排序后的候选块索引 [2048, 2] (value+idx)
    TBuf<TPosition::VECCALC> candConstBuf_; // consumer: negHuge 常量 [2048]

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> candIsOut_;   // consumer: position 级 0/1 (1=候选外), 仅用于分数降级 (R11 leak)
    LocalTensor<float> candNegHuge_; // -1e30 常量
    GlobalTensor<int32_t> outputIdxOffsetGm_; // A15: 每行输出索引偏移 (仅 sparse_indices)
    bool isOutputIdxOffsetValid_ = false;     // A15: offset 是否传入 (经 InitVecCandidateTensor 传递)

    int32_t blockId_ = -1;
    // P3: topk 自适应宽度 — 累加器/抽取/LD 归并按 sparseCount 收缩 (64 对齐),
    // 动态 topk (topk<2048) 时消除固定 2048 宽度的无效排序合并与搬运
    uint32_t topkWidth_ = BASE_TOPK; // Align(sparseCount, 64)
    uint32_t topkPairWidth_ = 4096;  // 2 * topkWidth_ (value+index 对)
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;

    struct QLIV2Common::ConstInfo constInfo_;
    struct QLIV2Common::LdSplitCoreInfo ldInfo_;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::GetKeyScale(const QLIV2Common::RunInfo& runInfo,
                                                        const LocalTensor<half>& resUb, int64_t batchId,
                                                        int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        // A11: k_scale 0 轴非连续 — 块基址 = 表值 x keyDequantScaleStride0 (真实 stride);
        // 0 时兜底原紧凑公式 (kCacheBlockSize_), 现网行为不变 (R8)
        // P13: 页跨度提升 int64 后再乘 — paged 混合缓存可超 2^31 个 scale 元素,
        // blockId * stride 的 int32 乘法在物理块号 >= ceil(2^31/stride) 时溢出
        // (stride=73856 时首个溢出块 29077), 负偏移符号扩展后越界/静默错读。
        // 必须在乘法前提升: static_cast<int64_t>(blockId * stride32) 已溢出无效。
        // 回归用例: qli_cand_test/test_qli_cand.py::test_qli_p13_kscale_int32_overflow
        int64_t kScaleBlkStride = constInfo_.keyDequantScaleStride0 != 0 ?
                                      static_cast<int64_t>(constInfo_.keyDequantScaleStride0) :
                                      static_cast<int64_t>(kCacheBlockSize_);
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        // P13 同型防御: batchId * maxBlockNumPerBatch_ 的 int32 乘法在 block_table
        // 总项超 2^31 (≈8.6GB 表) 时溢出 — 提前提升 int64
        int64_t blockTableBatchOffset = static_cast<int64_t>(batchId) * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(half);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb, kScaleGm[blockId * kScaleBlkStride + startBlockTableOffset], copyInParams,
                                 padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(half);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(half);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb[resUbBaseOffset + i * kCacheBlockSize_], kScaleGm[blockId * kScaleBlkStride],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(half);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(resUb, kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(inQueue_, 2, s2BaseSize_ * sizeof(float) * 2); // 32KB
    pipe->InitBuffer(outQueue_, 1, BASE_TOPK * sizeof(float));      // 8 KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));     // 8 KB
    pipe->InitBuffer(tmpBuf_, 64 * 1024);                           // 64KB
    // P3: 累加器按 topk 宽度分配 (topk<2048 时省 UB; 2048 时与原一致)
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_ * sizeof(float));

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    globalTopkNum_ = 0;

    // consumer 算子: 仅 mode=2 (candidate consumer) UB
    // R6 修复: candBuf 按行分区 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行, 行间 tile0 重排序
    // 会互相覆盖) — 2 行 x 2048 对 x 8B = 32KB; UB 预算 144+32+8=184KB <= 192KB
    pipe->InitBuffer(candBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2 * sizeof(float));
    pipe->InitBuffer(candConstBuf_, BASE_TOPK * sizeof(float)); // 8KB: negHuge
    candNegHuge_ = candConstBuf_.Get<float>();
    Duplicate(candNegHuge_.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_HUGE_F32, BASE_TOPK);
    PipeBarrier<PIPE_V>();

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitLDBuffers(TPipe* pipe)
{
    pipe->Reset();
    // P3: LD 归并缓冲按 topk 实际宽度收缩 (mrgListNum_ 列表 × 每列表 value+idx 对)
    pipe->InitBuffer(ldToBeMrgBuf_, topkPairWidth_ * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, topkPairWidth_ * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, topkWidth_ * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, topkWidth_ * sizeof(int32_t));
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitParams(const struct QLIV2Common::ConstInfo& constInfo,
                                                       const struct QLIV2Common::LdSplitCoreInfo& ldInfo,
                                                       const QLIV2TilingDataConsumer* __restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 2048
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    // P3: 按 sparseCount 计算实际 topk 工作宽度 (对齐 64; MergeSort/ExtractIndex 元素粒度需求)
    topkWidth_ = QLIV2Common::Align(static_cast<uint32_t>(constInfo.sparseCount), static_cast<uint32_t>(64));
    topkPairWidth_ = 2 * topkWidth_;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecInputTensor(GlobalTensor<half> weightsGm,
                                                               GlobalTensor<half> qScaleGm, GlobalTensor<half> kScaleGm,
                                                               GlobalTensor<int32_t> indiceOutGm,
                                                               GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm,
                                                                   GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                   GlobalTensor<float> vec1ResGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec0OutGm = vec0OutGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                                   GlobalTensor<int32_t> outputIdxOffsetGm,
                                                                   bool outputIdxOffsetValid)
{
    this->candidateTopkIndexInGm = candidateTopkIndexInGm;
    this->outputIdxOffsetGm_ = outputIdxOffsetGm; // A15
    this->isOutputIdxOffsetValid_ = outputIdxOffsetValid;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::AllocEventID()
{}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::FreeEventID()
{}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    QLIV2ServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
}

// 64 元素分块封装: 规避 Level-2 大 count 的 vmax/vmin/vabs 等指令异常
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                       const LocalTensor<float>& src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Max(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                       const LocalTensor<float>& src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Min(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAbs(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                   int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Abs(dst[off], src[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                          float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Mins(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                          float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Maxs(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecSubPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                       const LocalTensor<float>& src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Sub(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAddsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                          float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Adds(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMulPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                       const LocalTensor<float>& src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Mul(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

// 统计降序序列 (偶数 lane 为 value) 中 >= x 的元素个数, 二分实现
template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::CountGE(const LocalTensor<float>& sortedDesc, int32_t n, float x)
{
    int32_t lo = 0;
    int32_t hi = n;
    while (lo < hi) {
        int32_t mid = (lo + hi) / 2;
        if (sortedDesc.GetValue(2 * mid) >= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// consumer (use_candidate): 加载/排序候选行 (每行首个S2分片), 判定 tile 内块归属,
// 产出 position 级 isOut (fp32 0/1, 1=候选外) 与其 int32 形式
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::BuildCandidateMask(const QLIV2Common::RunInfo& info, int32_t cuS1Idx,
                                                               int32_t cuBaseS2Idx, int32_t innerS1Idx)
{
    int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t tileBlkNum = s2BaseSize_ / blockSize;
    int32_t tileBlockBase = cuBaseS2Idx / blockSize;
    // R6 修复: 同核多行 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行) 共享 candBuf 时,
    // 后一行的 tile0 重排序会覆盖前行候选 (s2 内层循环按 gS1 块整体推进) — candBuf 按行分区
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> candPairs = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    if (info.isFirstS2InnerLoop) {
        // 加载候选行到 tmp 尾部, 转 fp32 后与有序索引组成 [values | idx] 对并降序排序
        LocalTensor<int32_t> candInt = tmp[12288].template ReinterpretCast<int32_t>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = candBlocks * sizeof(int32_t);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        AscendC::DataCopyPad(candInt,
                             candidateTopkIndexInGm[info.candidateOutOffset + cuS1Idx * constInfo_.candidateTopkBlocks],
                             copyInParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        PipeBarrier<PIPE_V>();
        Cast(candPairs, candInt, RoundMode::CAST_NONE, candBlocks);
        PipeBarrier<PIPE_V>();
        DataCopy(candPairs[candBlocks].template ReinterpretCast<int32_t>(), globalTopkIndice_, candBlocks);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> candSortTmp = tmp[4096];
        QLIV2ServiceVec::SortAll(candPairs, candSortTmp, candBlocks);
        PipeBarrier<PIPE_V>();
        // V->S 围栏: 排序结果对后续标量 GetValue (CountGE 二分) 可见
        // (PipeBarrier<PIPE_V> 不足以保证标量读到 V 写数据; 参照 CANN topk_v200 的 SetFlag/WaitFlag<V_S> 模式)
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }
    // 候选与 tile 块号 [tileBlockBase, tileBlockBase+tileBlkNum) 的最小距离, 0 即命中
    // (candBuf 按行分区, R6: 同核行间覆盖已修复)
    LocalTensor<float> candPairs2 = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    int32_t lo = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase));
    int32_t hi = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase + tileBlkNum));
    LocalTensor<float> blkIdxF = tmp[4096]; // [tileBlkNum]
    LocalTensor<float> acc = tmp[4352];     // [tileBlkNum]
    LocalTensor<float> diff = tmp[4608];    // [tileBlkNum]
    Cast(blkIdxF, globalTopkIndice_, RoundMode::CAST_NONE, tileBlkNum);
    PipeBarrier<PIPE_V>();
    Adds(blkIdxF, blkIdxF, static_cast<float>(tileBlockBase), tileBlkNum);
    PipeBarrier<PIPE_V>();
    Duplicate(acc.ReinterpretCast<int32_t>(), QLIV2ServiceVec::POS_INF_F32, tileBlkNum);
    PipeBarrier<PIPE_V>();
    // 降序排序下: 值 >= base+tileBlkNum 占 [0, hi), 落在 tile 内的候选占 [hi, lo), < base (含 -1 pad) 占 [lo, ...)
    for (int32_t j = hi; j < lo; j++) {
        float v = candPairs2.GetValue(2 * j);
        Adds(diff, blkIdxF, -v, tileBlkNum);
        PipeBarrier<PIPE_V>();
        VecAbs(diff, diff, tileBlkNum);
        VecMinPair(acc, acc, diff, tileBlkNum);
    }
    // 距离 0 → 候选块; Brcb 展开到位置级; 距离 clamp 到 1 得 isOut (0/1)
    LocalTensor<float> posDist = tmp[12288]; // candInt 已释放, 可复用
    Brcb(posDist, acc, tileBlkNum / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    VecMinsScalar(posDist, posDist, 1.0f, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    candIsOut_ = posDist;
    // R11 (leak 语义, §1.6): isOut 仅用于分数降级 (pen 链), 索引保留真实位置号 —
    // 候选外可达位置作为 topk 填充泄漏成有效索引 (对齐模型 where(idxs < compress_lens) 语义),
    // 不再把候选外索引改 -1 (原 isOutI32/CAST_RINT 链已删)。
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec0(const QLIV2Common::RunInfo& info)
{
    // 只需要一个v核做
    if (blockId_ % 2 != 0) {
        return;
    }
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    // 计算输出w基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*64 + aic_offset
    int64_t vec0OutGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_ * BLOCK_CUBE));
    // 计算输入weight的地址偏移，qScale的地址偏移与weight相同
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * qHeadNum_;
    // 当前需要计算的S1行数，处理尾块场景
    int32_t cuS1ProcNum = cuBaseS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuProcRealNum = cuS1ProcNum * gSize_;
    int32_t cuProcEleNum = QLIV2Common::Align(cuProcRealNum, 32); // 32: UB对齐, 参照v1

    LocalTensor<half> inWeightsUb = inQueue_.AllocTensor<half>();
    LocalTensor<half> inQScaleUb = inWeightsUb[cuProcEleNum];
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = cuProcRealNum * sizeof(half);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    AscendC::DataCopyPad(inWeightsUb, weightsGm[weightGmOffset], copyInParams, padParams);
    AscendC::DataCopyPad(inQScaleUb, qScaleGm[weightGmOffset], copyInParams, padParams);

    inQueue_.EnQue<half>(inWeightsUb);
    inWeightsUb = inQueue_.DeQue<half>();
    AscendC::Mul(inWeightsUb, inWeightsUb, inQScaleUb, cuProcEleNum);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> resUb = outQueue_.AllocTensor<half>();
    AscendC::Brcb(resUb, inWeightsUb, static_cast<uint8_t>(cuProcEleNum / 8), {1, 8});
    inQueue_.FreeTensor(inWeightsUb);

    outQueue_.EnQue<half>(resUb);
    resUb = outQueue_.DeQue<half>();
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = cuProcRealNum * BLOCK_CUBE * sizeof(half);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    AscendC::DataCopyPad(vec0OutGm[vec0OutGmOffset], resUb, copyOutParams);
    outQueue_.FreeTensor(resUb);
}

template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::AlignS2(int32_t cuS2Len)
{
    // 限制：当前cuS2Len最大为2048，暂不考虑更长
    // 该函数目的是将cuS2Len对齐到形如 32*(4^n)*m 的形式 (m ∈ [1, 3])，方便后续sort/merge
    if (cuS2Len <= ELE_NUM_128) {
        return Align(cuS2Len, ELE_NUM_32);
    } else if (cuS2Len <= ELE_NUM_512) {
        return Align(cuS2Len, ELE_NUM_128);
    } else {
        return Align(cuS2Len, ELE_NUM_512);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec1(const QLIV2Common::RunInfo& info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*2048 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (s1BaseSize_ * s2BaseSize_);

    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);
    // 基本块基地址偏移奇数核加一个S1地址偏移
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * s2BaseSize_;
    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    int32_t cuRealAcSeqCount = 0;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + cuS1BeginIdxPerAiv;
    }
    int32_t cuRealAcSeqIni = cuRealAcSeq;

    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeqCount += 1;
            cuRealAcSeq = (cuRealAcSeqCount + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        // 当前vec1ResGm对应S1的位置
        // P10: LD 布局 = 每核 s1Base 行预留区 (起点 = saveWorkSpaceIdx * s1Base * pairW),
        // 行内用全局行号 cuS1Idx (0..s1Base-1) 作槽偏移 — 同一全局行在所有分段核的
        // 槽偏移一致, 头核归并时按 (核i*4 + 全局行) 寻址
        uint64_t wsOffset = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * topkPairWidth_ +
                            static_cast<uint64_t>(cuS1Idx) * topkPairWidth_;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = AlignS2(cuS2Len);
            LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
            LocalTensor<float> kScaleUb = mmInUb[cuS2LenVecAlign];
            LocalTensor<half> kScaleTUb = kScaleUb.template ReinterpretCast<half>()[cuS2LenVecAlign];
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> padTParams{false, 0, 0, 0};
            AscendC::DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = cuS2Len * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            copyInParams.rsv = 0;
            AscendC::DataCopyPad(mmInUb, mm1ResGm[mmGmOffset + innerS1Idx * s2BaseSize_], copyInParams, padParams);
            GetKeyScale(info, kScaleTUb, info.bIdx, cuBaseS2Idx, cuS2Len);
            inQueue_.EnQue<float>(mmInUb);
            mmInUb = inQueue_.DeQue<float>();
            AscendC::Cast(kScaleUb, kScaleTUb, RoundMode::CAST_NONE, cuS2Len);
            PipeBarrier<PIPE_V>();
            AscendC::Mul(mmInUb, mmInUb, kScaleUb, cuS2Len);
            PipeBarrier<PIPE_V>();
            // consumer (R11 leak): 候选块外 score 降级 NEG_HUGE (S4a, 纯算术无 NaN);
            // 仅降级排序资格、不取消入选资格 — 不足 topk 时候选外可达位置作为填充泄漏 (§1.6)
            // P11: 全候选快路径 — 若该批总块数 ≤ candidateTopkBlocks, 候选覆盖全部
            // 位置块, isOut 恒 0 (pen 恒 0, score 不降级), 整个 BuildCandidateMask+pen
            // 链可跳过 (生产 s2=16384: 2048 块 ≤ 2048 候选 → 恒命中)。逐块判定用
            // info.actS2SizeOrig (压缩前) — 候选块号域即压缩前位置域。
            // P11 撤销 (2026-09-17): 全候选快路径的前提"候选表覆盖全部块号"是数据属性,
            // 无法从 shape/attr 判定 (上游候选可含重复/-1, kernel 内严格判定需逐点读候选表,
            // 成本 ≈ 被跳过的 tile 内命中检测, 无净收益); randint 候选用例证明上游无覆盖契约。
            // 恢复无条件 BuildCandidateMask+pen 链 (正确性优先)。
            BuildCandidateMask(info, cuS1Idx, cuBaseS2Idx, innerS1Idx);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> pen = tmpBuf_.Get<float>()[4096]; // blkIdxF/acc/d 已释放, 复用
            Sub(pen, candNegHuge_, mmInUb, cuS2Len);
            PipeBarrier<PIPE_V>();
            Mul(pen, pen, candIsOut_, cuS2Len);
            PipeBarrier<PIPE_V>();
            Add(mmInUb, mmInUb, pen, cuS2Len);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
            LocalTensor<float> sortScoreUb = sortBuff;
            LocalTensor<float> sortIndiceUb = sortBuff[cuS2LenVecAlign];
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, mmInUb, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(mmInUb);
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();
            // R11 (leak 语义, §1.6): 候选块外索引不再置 -1 —
            // 分数已在 S4a 降级 NEG_HUGE (仍高于 -inf: 不足 topk 时作为填充入选, 排候选内之后),
            // 索引保留真实位置号, 与模型 where(idxs < compress_lens, idxs + offset, -1) 等价:
            // 仅不可达位置 (score -inf 沉底 + 尾部 -1 填充) 输出 -1。
            LocalTensor<float> tmpSortBuf = sortBuff[2 * cuS2LenVecAlign];
            QLIV2ServiceVec::SortAll(sortBuff, tmpSortBuf, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            QLIV2ServiceVec::MergeSort(globalTopkUb_[innerS1Idx * topkPairWidth_], topkWidth_, sortBuff,
                                       cuS2LenVecAlign, tmpSortBuf); // P3
            PipeBarrier<PIPE_V>();
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            // 中间结果保存
            if (needCopyOutGm && !info.isNeedLD) {
                LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
                // P3: 只抽取前 topkWidth_ 对 (降序排序下 2048 名之后永不入选, 截断语义等价)
                ExtractIndex(idxULocal, globalTopkUb_[innerS1Idx * topkPairWidth_].template ReinterpretCast<uint32_t>(),
                             topkWidth_);
                PipeBarrier<PIPE_V>();
                // A15: output_idx_offset — 输出拷 GM 前逐元素加行偏移 (对齐 arch35 IndicesAddOffset;
                // 零偏移零开销; GM 标量读无 V_S 竞态; int32 整数域 Adds 位精确, -1 槽 +0 不变。
                // candidate_topk_index 不加 (相对块号契约, §11.6))
                if (isOutputIdxOffsetValid_) {
                    int32_t rowOff = outputIdxOffsetGm_.GetValue(info.outputIdxOffsetCoreOffset + cuS1Idx * kHeadNum_);
                    if (rowOff != 0) {
                        LocalTensor<int32_t> idxI32 = idxULocal.template ReinterpretCast<int32_t>();
                        for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                            Adds(idxI32[off], idxI32[off], rowOff, 64);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                outQueue_.EnQue<uint32_t>(idxULocal);
                idxULocal = outQueue_.DeQue<uint32_t>();
                QLIV2ServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                         idxULocal.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
                outQueue_.FreeTensor(idxULocal);
            }
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                AscendC::DataCopyExtParams copyWsParams;
                copyWsParams.blockLen = topkPairWidth_ * sizeof(float); // P3: 按实际 topk 宽度
                copyWsParams.srcStride = 0;
                copyWsParams.dstStride = 0;
                copyWsParams.blockCount = 1;
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                AscendC::DataCopyPad(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_], copyWsParams);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                PipeBarrier<PIPE_V>();
            }
        } else if (cuRealAcSeq <= 0) {
            // 无效长度处理
            if (info.isNeedLD && info.isLastS2InnerLoop) {
                PipeBarrier<PIPE_V>();
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_],
                                         topkPairWidth_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            } else {
                CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
            }
        } else if (cuS2Len <= 0) {
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_],
                                         topkPairWidth_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // BNSD场景无效S1 输出-1
    if (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2SizeOrig;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessLD()
{
    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
    LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<float> indexValuePadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams indexValueParams;
    // P3: 每列表搬运/间距按实际 topk 对宽 (vec1ResGm 内相邻列表间距 = 3 对宽)
    indexValueParams.blockLen = topkPairWidth_ * sizeof(int32_t); // bytes
    // P10: 分段槽按核分布 (核 i 的区起点 = i*s1Base 槽), 跨 3 列表的间距 = s1Base 槽。
    // DataCopyExtParams.srcStride 单位 = 32B 块: (s1Base*pairW*4B)/32
    indexValueParams.srcStride = s1BaseSize_ * topkPairWidth_ * sizeof(int32_t) / 32U;
    indexValueParams.dstStride = 0;

    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    uint32_t ldProcessLen = 4; // 4: 4块归约任务做一次Merge
    uint32_t ldProcessNum = (ldProWorkspaceNum - 1) / (ldProcessLen - 1);
    uint32_t ldTailLen = ldProWorkspaceNum - (ldProcessNum * (ldProcessLen - 1) + 1); // 尾块长度

    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        // 拷贝第一块归约任务
        indexValueParams.blockCount = 1;
        uint64_t wsOffsetIni = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                               static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
        AscendC::DataCopyPad(curValueIdxUb, vec1ResGm[wsOffsetIni], indexValueParams, indexValuePadParams);
        uint64_t valueOffset = topkPairWidth_;
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // 处理等于4的部分
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // LD处理偏移
            uint64_t wsOffset = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                                static_cast<uint64_t>(i * (ldProcessLen - 1) + 1) * s1BaseSize_ * topkPairWidth_ +
                                static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
            indexValueParams.blockCount = 1; // P10: 逐段拷贝 (见上)

            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            // P10: 绕开 srcStride 语义不确定性 — 3 个分段槽逐个显式拷贝
            for (uint32_t k = 0; k < ldProcessLen - 1U; k++) {
                uint64_t segSlot = wsOffset + static_cast<uint64_t>(k) * s1BaseSize_ * topkPairWidth_;
                AscendC::DataCopyPad(curValueIdxUb[valueOffset + k * topkPairWidth_], vec1ResGm[segSlot],
                                     indexValueParams, indexValuePadParams);
            }
            // merge参数 (P3: elementLengths 按实际 topk 宽度)
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = topkWidth_;
            params.elementLengths[1] = topkWidth_;
            params.elementLengths[2] = topkWidth_;
            params.elementLengths[3] = topkWidth_;
            params.ifExhaustedSuspension = true;
            params.validBit = 0b1111;
            params.repeatTimes = 1;

            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[topkPairWidth_];
            srcList.src3 = curValueIdxUb[2 * topkPairWidth_];
            srcList.src4 = curValueIdxUb[3 * topkPairWidth_];
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, topkPairWidth_);
            PipeBarrier<PIPE_V>();
        }

        // 处理不等于4的部分
        if (ldTailLen != 0) {
            // 搬运尾块
            uint64_t wsOffsetTail =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                static_cast<uint64_t>(ldProcessNum * (ldProcessLen - 1) + 1) * s1BaseSize_ * topkPairWidth_ +
                static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
            indexValueParams.blockCount = ldTailLen;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            AscendC::DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffsetTail], indexValueParams,
                                 indexValuePadParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = topkWidth_;
            params.elementLengths[1] = topkWidth_;
            params.elementLengths[2] = topkWidth_;
            params.elementLengths[3] = topkWidth_;
            params.ifExhaustedSuspension = true;
            if (ldTailLen == 1) {
                params.validBit = 0b0011;
            } else if (ldTailLen == 2) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[topkPairWidth_];
            srcList.src3 = curValueIdxUb[2 * topkPairWidth_];
            srcList.src4 = curValueIdxUb[3 * topkPairWidth_];
            PipeBarrier<PIPE_V>();
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, topkPairWidth_);
            PipeBarrier<PIPE_V>();
        }

        // 搬出 (P3: Extract repeat = topkWidth_/32, 每 repeat 消费 32 对)
        Extract(outValueUb, outIdxUb, curValueIdxUb, (topkWidth_ / 32));
        PipeBarrier<PIPE_V>();
        InitSortOutBuf(curValueIdxUb, topkPairWidth_);
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        // A15: LD(decode) 路径同样加 output_idx_offset (行前缀 = indiceOutCoreOffset/sparseCount, 即
        // batch 前缀 x kHeadNum + n2; 与 ProcessVec1 消费点同语义)
        if (isOutputIdxOffsetValid_) {
            uint32_t rowGlobal = ldInfo_.mStart + j;
            int32_t rowOff = outputIdxOffsetGm_.GetValue(ldInfo_.indiceOutCoreOffset / constInfo_.sparseCount +
                                                         rowGlobal * constInfo_.kHeadNum);
            if (rowOff != 0) {
                for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                    Adds(idxULocal1[off], idxULocal1[off], rowOff, 64);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        uint64_t outOffset =
            ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) * constInfo_.kHeadNum * constInfo_.sparseCount;
        AscendC::DataCopyPad(indiceOutGm[outOffset], idxULocal1, copyOutParams);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
