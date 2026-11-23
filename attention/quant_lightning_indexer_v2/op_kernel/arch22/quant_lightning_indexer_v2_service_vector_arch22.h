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
 * \file quant_lightning_indexer_v2_service_vector_arch22.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"
#include "quant_lightning_indexer_v2_vector.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace QLIV2ServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr uint32_t ELE_NUM_32 = 32;
constexpr uint32_t ELE_NUM_128 = 128;
constexpr uint32_t ELE_NUM_512 = 512;

// ============ tmpBuf_(64KB) 内分区表, 单位: float 元素下标 ============
// 布局与 docs/qli_v2_two_level_topk_design.md §3.3 的表格一一对应
constexpr int64_t TMP_BUF_BYTES = 64 * 1024;
constexpr int64_t TMP_FLOAT_NUM = TMP_BUF_BYTES / sizeof(float); // 16384
constexpr int64_t TMP_BLK_SCORE_OFF = 6144;                      // 块级 score[blockNumPad], blkIdx 紧随其后
constexpr int64_t TMP_BLK_SORT_TMP_OFF = 11776;                  // MrgSort 临时区
constexpr int64_t TMP_BLK_SORT_TMP_SIZE = 4608; // 11776 + 4608 == TMP_FLOAT_NUM, 越界即 aicore 异常
// candidate 块化 scratch: 6 段 × 256 floats
// (256 = blockNumPad 上界 = s2BaseSize(2048) / candidateBlockSize(8))
constexpr int64_t TMP_CAND_SCRATCH_OFF = 14336;
constexpr int64_t TMP_CAND_SEG_NUM = 256;
// isPad 链与 pin 链先后复用同一片 6 段, 段内语义随阶段变化, 见各段注释
constexpr int64_t TMP_SEG_IDX_SRC_OFF = TMP_CAND_SCRATCH_OFF + 0 * TMP_CAND_SEG_NUM; // isPad: 等差源 / pin: score-PINF
constexpr int64_t TMP_SEG_THR_OFF = TMP_CAND_SCRATCH_OFF + 1 * TMP_CAND_SEG_NUM;     // isPad: 阈值 / pin: pA
constexpr int64_t TMP_SEG_IS_PAD_OFF = TMP_CAND_SCRATCH_OFF + 2 * TMP_CAND_SEG_NUM; // isPad: isPad 标记 / pin: pB
constexpr int64_t TMP_SEG_AUX_OFF = TMP_CAND_SCRATCH_OFF + 3 * TMP_CAND_SEG_NUM;    // isPad: (idx+1)*isPad / pin: pFlag
constexpr int64_t TMP_SEG_PIN_RAW_OFF = TMP_CAND_SCRATCH_OFF + 4 * TMP_CAND_SEG_NUM; // isPad: -isPad / pin: score-PINF
constexpr int64_t TMP_SEG_PIN_NEG_OFF =
    TMP_CAND_SCRATCH_OFF + 5 * TMP_CAND_SEG_NUM; // isPad: (score-NINF)*(-isPad) / pin: 0 基线

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
                                      const QLIV2TilingData* __restrict tilingData);
    __aicore__ inline void ProcessLD();
    // candidate(两级TopK 第一级): s2 被切分的行, 归并各分段核落工作区的块级候选后直出
    __aicore__ inline void ProcessLDCand();
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm, GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                  GlobalTensor<float> vec1ResGm, GlobalTensor<float> candidateResGm);
    __aicore__ inline void InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexOutGm,
                                                  GlobalTensor<int32_t> candidateBlockLengthGm,
                                                  GlobalTensor<int32_t> outputIdxOffsetGm, bool outputIdxOffsetValid);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                              GlobalTensor<half> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm, GlobalTensor<bfloat16_t> valueOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline int32_t AlignS2(int32_t cuS2Len);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe* pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<float> candidateResGm; // candidate 块级候选的跨核归约工作区
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<half> vec0OutGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<bfloat16_t> valueOutGm; // return_value: BF16 score 输出（与 arch35 对齐）
    GlobalTensor<int32_t> candidateTopkIndexOutGm;
    GlobalTensor<int32_t> candidateBlockLengthGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================

private:
    __aicore__ inline void GetKeyScale(const QLIV2Common::RunInfo& runInfo, const LocalTensor<half>& resUb,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // candidate (two-level topk)
    __aicore__ inline int32_t CountGE(const LocalTensor<float>& sortedDesc, int32_t n, float x);
    // candidate（两级 TopK source 模式）：每行有效候选块数 = min(ceil(压缩后 K 长度 / 块大小), 容量)
    __aicore__ inline int32_t GetCandidateLength(uint32_t actS2Size) const;
    __aicore__ inline void ProcessCandBlockTopk(const QLIV2Common::RunInfo& info, int32_t cuS1Idx, int32_t cuS2Len,
                                                int32_t cuS2LenVecAlign, int32_t cuRealAcSeq, int32_t innerS1Idx);
    __aicore__ inline void CopyOutCandTopkIndex(const QLIV2Common::RunInfo& info, int32_t cuS1Idx, int32_t innerS1Idx);
    // candidate 向量区间封装: 按 B32_VEC_ELM_NUM 分块 (本版本 Level-2 大 count 的 vmax 等指令会触发 aicore 异常)
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
    TBuf<TPosition::VECCALC> valueOutBuf_; // return_value: BF16 score 输出中转

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;
    TBuf<> ldOutBf16Buf_; // DCP LD: 归并后 BF16 sparse_values 输出中转

    // tmp buff for candidate (two-level topk)
    TBuf<TPosition::VECCALC> blockSortOutBuf_; // candidate 开启: 块级 topk 累加器 [CeilDiv(s1BaseSize,2), 2048, 2]
    TBuf<TPosition::VECCALC> candidateLenBuf_; // candidate_block_length 单元素搬出槽（与候选索引 UB 分离）

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> globalBlockTopkUb_;
    GlobalTensor<int32_t> outputIdxOffsetGm_; // A15: 每行输出索引偏移 (仅 sparse_indices)
    bool isOutputIdxOffsetValid_ = false;     // A15: offset 是否传入 (经 InitVecCandidateTensor 传递)

    int32_t blockId_ = -1;
    // P3: topk 自适应宽度 — 累加器/抽取/LD 归并按 sparseCount 收缩 (64 对齐),
    // 动态 topk (topk<2048) 时消除固定 2048 宽度的无效排序合并与搬运
    uint32_t topkWidth_ = BASE_TOPK;                           // Align(sparseCount, B32_VEC_ELM_NUM)
    uint32_t topkPairWidth_ = VALUE_AND_INDEX_NUM * BASE_TOPK; // 2 * topkWidth_ (value+index 对)
    // candidate(两级TopK 第一级) 块级候选宽度: 关闭态为 0 (LD 归并缓冲退回位置级宽度, 零开销)
    uint32_t candWidth_ = 0;     // candidateTopkBlocks
    uint32_t candPairWidth_ = 0; // 2 * candWidth_ (value+index 对)
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
    pipe->InitBuffer(tmpBuf_, TMP_BUF_BYTES);
    // P3: 累加器按 topk 宽度分配 (topk<2048 时省 UB; 2048 时与原一致)
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_ * sizeof(float));
    // return_value: FP32 分数抽取中转（写出前再 Cast 成 BF16；仅 return_value=1 时分配）
    if (constInfo_.returnValue != 0U) {
        pipe->InitBuffer(valueOutBuf_, topkWidth_ * sizeof(float));
    }

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    globalTopkNum_ = 0;

    // candidate (two-level topk) 第一级(source)分配: 关闭态不占用额外 UB
    // UB 预算 (192KB): 基础 144KB + blockSortOutBuf 32KB = 176KB
    // (拆分前的 consumer 形态 candBuf 16KB + candConstBuf 8KB 已迁出)
    if (constInfo_.candidateOn) {
        pipe->InitBuffer(blockSortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE * sizeof(float)); // 32KB
        pipe->InitBuffer(candidateLenBuf_, 32); // 32B: candidate_block_length 单元素独立搬出槽
        globalBlockTopkUb_ = blockSortOutBuf_.Get<float>();
        InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
    }

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
    // P3: LD 归并缓冲按 topk 实际宽度收缩 (mrgListNum_ 列表 × 每列表 value+idx 对);
    // candidate 块级归并与位置级归并复用同一组缓冲 (串行两趟), 宽度取两者较大值。
    uint32_t mrgPairWidth = topkPairWidth_ > candPairWidth_ ? topkPairWidth_ : candPairWidth_;
    uint32_t mrgWidth = topkWidth_ > candWidth_ ? topkWidth_ : candWidth_;
    pipe->InitBuffer(ldToBeMrgBuf_, mrgPairWidth * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, mrgPairWidth * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, mrgWidth * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, mrgWidth * sizeof(int32_t));
    if (constInfo_.returnValue != 0U) {
        pipe->InitBuffer(ldOutBf16Buf_, mrgWidth * sizeof(bfloat16_t));
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitParams(const struct QLIV2Common::ConstInfo& constInfo,
                                                       const struct QLIV2Common::LdSplitCoreInfo& ldInfo,
                                                       const QLIV2TilingData* __restrict tilingData)
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
    topkWidth_ =
        QLIV2Common::Align(static_cast<uint32_t>(constInfo.sparseCount), static_cast<uint32_t>(B32_VEC_ELM_NUM));
    topkPairWidth_ = 2 * topkWidth_;
    // candidate(两级TopK 第一级): 块级候选的归并宽度 (关闭态 0, 不占 UB)
    candWidth_ = constInfo.candidateOn ? constInfo.candidateTopkBlocks : 0U;
    candPairWidth_ = 2U * candWidth_;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecInputTensor(GlobalTensor<half> weightsGm,
                                                               GlobalTensor<half> qScaleGm, GlobalTensor<half> kScaleGm,
                                                               GlobalTensor<int32_t> indiceOutGm,
                                                               GlobalTensor<int32_t> blockTableGm,
                                                               GlobalTensor<bfloat16_t> valueOutGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm,
                                                                   GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                   GlobalTensor<float> vec1ResGm,
                                                                   GlobalTensor<float> candidateResGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->candidateResGm = candidateResGm;
    this->vec0OutGm = vec0OutGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexOutGm,
                                                                   GlobalTensor<int32_t> candidateBlockLengthGm,
                                                                   GlobalTensor<int32_t> outputIdxOffsetGm,
                                                                   bool outputIdxOffsetValid)
{
    this->candidateTopkIndexOutGm = candidateTopkIndexOutGm;
    this->candidateBlockLengthGm = candidateBlockLengthGm;
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
    // return_value: 无效行同步输出 BF16 -inf（与 arch35 对齐）
    if (constInfo_.returnValue != 0U) {
        LocalTensor<float> valF32 = valueOutBuf_.Get<float>();
        Duplicate(valF32.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_INF,
                  static_cast<int32_t>(constInfo_.sparseCount));
        LocalTensor<bfloat16_t> valBf16 = outQueue_.AllocTensor<bfloat16_t>();
        AscendC::Cast(valBf16, valF32, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(constInfo_.sparseCount));
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        QLIV2ServiceVec::CopyOut(valueOutGm[invalidS1offset], valBf16, static_cast<int64_t>(constInfo_.sparseCount));
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        outQueue_.FreeTensor(valBf16);
    }
    // candidate 开启: candidate_topk_index 同步填 -1 (行偏移 = sparse 偏移 / sparseCount * candidateTopkBlocks)
    if (constInfo_.candidateOn) {
        uint64_t candOffset =
            static_cast<uint64_t>(invalidS1offset) / constInfo_.sparseCount * constInfo_.candidateTopkBlocks;
        LocalTensor<float> candULocal = outQueue_.AllocTensor<float>();
        LocalTensor<int32_t> candIdxLocal = candULocal.template ReinterpretCast<int32_t>();
        Duplicate(candIdxLocal, constInfo_.INVALID_IDX, constInfo_.candidateTopkBlocks);
        outQueue_.EnQue<float>(candULocal);
        candULocal = outQueue_.DeQue<float>();
        QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[candOffset], candIdxLocal, constInfo_.candidateTopkBlocks);
        outQueue_.FreeTensor(candULocal);
        // 无效行没有有效候选块
        candidateBlockLengthGm.SetValue(candOffset / constInfo_.candidateTopkBlocks, 0);
    }
}

// 向量区间封装: 按 B32_VEC_ELM_NUM(一个 repeat 的 fp32 元素数)分块下发。
// 本版本 Level-2 count > 64 的 vmax/vmin/vabs 等指令会触发 aicore 异常, 故一律走分块。
// 前置条件: n 为 B32_VEC_ELM_NUM 的整数倍 (调用点均已按该粒度对齐)。
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                       const LocalTensor<float>& src1, int32_t n)
{
    {
        const BinaryRepeatParams repeatParams = QLIV2ServiceVec::B32BinaryRepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QLIV2ServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Max(dst[off], src0[off], src1[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                       const LocalTensor<float>& src1, int32_t n)
{
    {
        const BinaryRepeatParams repeatParams = QLIV2ServiceVec::B32BinaryRepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QLIV2ServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Min(dst[off], src0[off], src1[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAbs(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                   int32_t n)
{
    {
        const UnaryRepeatParams repeatParams = QLIV2ServiceVec::B32RepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QLIV2ServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Abs(dst[off], src[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                          float v, int32_t n)
{
    {
        const UnaryRepeatParams repeatParams = QLIV2ServiceVec::B32RepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QLIV2ServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Mins(dst[off], src[off], v, mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                          float v, int32_t n)
{
    {
        const UnaryRepeatParams repeatParams = QLIV2ServiceVec::B32RepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QLIV2ServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Maxs(dst[off], src[off], v, mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecSubPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                       const LocalTensor<float>& src1, int32_t n)
{
    {
        const BinaryRepeatParams repeatParams = QLIV2ServiceVec::B32BinaryRepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QLIV2ServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Sub(dst[off], src0[off], src1[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAddsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                          float v, int32_t n)
{
    {
        const UnaryRepeatParams repeatParams = QLIV2ServiceVec::B32RepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QLIV2ServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Adds(dst[off], src[off], v, mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMulPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                       const LocalTensor<float>& src1, int32_t n)
{
    {
        const BinaryRepeatParams repeatParams = QLIV2ServiceVec::B32BinaryRepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QLIV2ServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Mul(dst[off], src0[off], src1[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
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

template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::GetCandidateLength(uint32_t actS2Size) const
{
    if (actS2Size == 0U || constInfo_.candidateBlockSize == 0U) {
        return 0;
    }
    uint32_t blockNum = (actS2Size + constInfo_.candidateBlockSize - 1U) / constInfo_.candidateBlockSize;
    if (blockNum > constInfo_.candidateTopkBlocks) {
        blockNum = constInfo_.candidateTopkBlocks;
    }
    return static_cast<int32_t>(blockNum);
}

// candidate 开启: 块内 amax (log2(blockSize) 轮 Max 树) + pin 尾块 + 块级排序/归并
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessCandBlockTopk(const QLIV2Common::RunInfo& info, int32_t cuS1Idx,
                                                                 int32_t cuS2Len, int32_t cuS2LenVecAlign,
                                                                 int32_t cuRealAcSeq, int32_t innerS1Idx)
{
    int32_t blkLen = cuS2LenVecAlign;
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t blockNum = blkLen / blockSize;                        // 含尾块 (pad -inf 后按对齐长度计)
    int32_t realBlockNum = (cuS2Len + blockSize - 1) / blockSize; // 含有效位置的块数
    // 所有向量 count 按 64 对齐 (非对齐 count 的向量指令在本架构触发 aicore 异常);
    // 且块数必须过 AlignS2 (32*(4^n)*m, m<=3): SortAll 的 MrgSort 循环按 mrgGroups/4 缩减,
    // 非 4 幂组数 (如 192 块=6 组) 会整组丢失 (实测 s2=1536 丢块 128..191 段)
    int32_t blockNumPad = (blockNum < 64) ? 64 : AlignS2(blockNum);
    int32_t tileBlockBase = info.s2Idx * s2BaseSize_ / blockSize;
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    // 块内归约: vcgmax 每 32B 块 (8 fp32) 出 1 个 max, 紧凑输出 [blockNum]
    LocalTensor<float> blkScore = tmp[6144];
    // R10: blkLen=96 是 AlignS2 输出中唯一非 64 倍数的值 (≤128 段对齐到 32 的倍数),
    // 96/64 整除截断为 1 只归约 [0,64) — 块 8..11 残留上一 tile/行的 stale 分数
    // (实测 big128k_b2_varlen b1 行 73 tile 56: 块 14344 拿到 tile 55 块 14088 的
    // 3.5568, 虚高挤掉 2048 名边界块 2760; 小 shape 总块数≤2048 集合不变故未暴露)。
    // 改 CeilDiv 覆盖全部块; 多归约的 [96,128) stale 只落在 pad 块槽位, 被 -inf
    // 位型链位精确覆盖, 无害。
    int32_t brmRepeat = CeilDiv(blkLen, 64);
    BlockReduceMax(blkScore, tmp[0], brmRepeat, 64, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    int32_t lastBlk = (cuRealAcSeq - 1) / blockSize;
    int32_t pinLocal = lastBlk - tileBlockBase;
    // 块级 [scores | idx] 对: idx = tileBlockBase + j; j >= realBlockNum 的 pad 块置 -1
    // (禁止标量 SetValue: 标量写不受 PipeBarrier<PIPE_V> 围栏, 与后续 V 管道读存在确定性竞态
    //  (实测 innerS1Idx=1 行的填充被 SortAll 抢跑覆盖); 禁止 s32->f32 Cast (v220 无此组合)。
    //  改为纯 int32 向量算术: idx' = idx - (idx+1)*isPad, isPad = clamp(idx - thr, 0, 1),
    //  thr = base + realBlockNum - 1; 所有指令 count=blockNumPad (64 对齐), 按 64 分块避免 mask 寄存器限制。
    //  scratch 用 source 独占区 [14336,15360) (拆分后候选掩码相关缓冲已整体迁出至 consumer 算子),
    //  避开主路径 tmpSortBuf [4096,12288), 消除跨迭代残留读的隐患)
    LocalTensor<int32_t> idxScr = tmp[14336].template ReinterpretCast<int32_t>();              // 等差源
    LocalTensor<int32_t> thrI = tmp[14592].template ReinterpretCast<int32_t>();                // 阈值
    LocalTensor<int32_t> offI = tmp[14848].template ReinterpretCast<int32_t>();                // idx-thr -> isPad
    LocalTensor<int32_t> tI = tmp[15104].template ReinterpretCast<int32_t>();                  // (idx+1)*isPad
    LocalTensor<int32_t> blkIdx = tmp[6144 + blockNumPad].template ReinterpretCast<int32_t>(); // 最终 idx
    ArithProgression<int32_t>(idxScr, static_cast<int32_t>(tileBlockBase), 1, blockNumPad);
    Duplicate(thrI, tileBlockBase + realBlockNum - 1, blockNumPad);
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Sub(offI[off], idxScr[off], thrI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Maxs(offI[off], offI[off], 0, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Mins(offI[off], offI[off], 1, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Adds(tI[off], idxScr[off], 1, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Mul(tI[off], tI[off], offI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Sub(blkIdx[off], idxScr[off], tI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    // pad 块 score 归一为 -inf 位型: 覆盖 [realBlockNum, blockNumPad), 含 stale 区 [blockNum, blockNumPad)
    // (尾 tile cuS2Len 非整块时 BlockReduceMax 只写 [0, blockNum), 其后是脏数据 —
    //  脏 score 的 (-inf,-1) 不同构对会挤进 topk 挤掉真实块, 实测 128K 场景 -1 槽超标)
    // score' = score + (score - NINF_bits)*(-isPad), isPad 于 pad 位为 1; 全 int32 向量
    if (blockNumPad > realBlockNum) {
        LocalTensor<int32_t> pRaw = tmp[15360].template ReinterpretCast<int32_t>(); // -isPad
        LocalTensor<int32_t> pNeg = tmp[15616].template ReinterpretCast<int32_t>(); // (score-NINF)*(-isPad)
        LocalTensor<int32_t> scoreI = blkScore.template ReinterpretCast<int32_t>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Duplicate(pRaw[off], -1, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pRaw[off], pRaw[off], offI[off], 64); // -isPad
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pNeg[off], scoreI[off], 8388608, 64); // score - NINF_bits (NINF=0xFF800000=-8388608)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pNeg[off], pNeg[off], pRaw[off], 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(scoreI[off], scoreI[off], pNeg[off], 64); // pad 块 -> -inf 位型
        }
        PipeBarrier<PIPE_V>();
    }
    // pin 最新 token 所在块 (+inf 位型注入该块 score): 纯 int32 向量实现, 在 isPad 链之后
    // (复用其 scratch; 标量 SetValue 写不受 PipeBarrier<PIPE_V> fence, 与 SortAll 多轮读存在竞态, 已弃用)。
    // flag = -1 于 pin 位, 0 于其余: score' = score + (score - PINF_bits) * flag,
    // pin 位得 PINF_bits (0x7F800000 = +inf), 其余不变 (flag=0 使 pad 位 -inf 的中间溢出无害)。
    if (pinLocal >= 0 && pinLocal < realBlockNum) {
        LocalTensor<int32_t> pRaw = tmp[15360].template ReinterpretCast<int32_t>(); // [15360,15616)
        LocalTensor<int32_t> pNeg = tmp[15616].template ReinterpretCast<int32_t>(); // [15616,15872)
        LocalTensor<int32_t> pA = thrI;                                             // 复用 isPad 链已释放 scratch
        LocalTensor<int32_t> pB = offI;
        LocalTensor<int32_t> pFlag = tI;
        LocalTensor<int32_t> scoreI = blkScore.template ReinterpretCast<int32_t>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            // idxScr 为全局块号, pin 也必须用全局 id (tileBlockBase + pinLocal) 比较
            Adds(pRaw[off], idxScr[off], -(tileBlockBase + pinLocal), 64); // 0 于 pin 块
        }
        PipeBarrier<PIPE_V>();
        Duplicate(pNeg, 0, blockNumPad);
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Sub(pNeg[off], pNeg[off], pRaw[off], 64); // -(idx-pinLocal)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Maxs(pA[off], pRaw[off], 0, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mins(pA[off], pA[off], 1, 64); // 1 若 idx > pin
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Maxs(pB[off], pNeg[off], 0, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mins(pB[off], pB[off], 1, 64); // 1 若 idx < pin
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(pFlag[off], pA[off], pB[off], 64); // isNotPin (0/1, 互斥)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pFlag[off], pFlag[off], -1, 64); // -1 于 pin, 0 其余
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pRaw[off], scoreI[off], -2139095040, 64); // score - 0x7F800000 (复用 pRaw)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pRaw[off], pRaw[off], pFlag[off], 64); // (score-PINF)*flag
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(scoreI[off], scoreI[off], pRaw[off], 64); // pin -> +inf 位型
        }
        PipeBarrier<PIPE_V>();
    }
    // 块级排序 + 归并到块级累加器
    LocalTensor<float> blkPairs = tmp[6144]; // [scores blockNumPad | idx blockNumPad]
    // MrgSort tmp 需容纳 mrgDst+mrgSrc = 2048*2 + blockNumPad*2 <= 4608 floats;
    // 11776 + 4608 = 16384 恰为 tmpBuf_(64KB) 末尾, 12288 起会越界 2KB (aicore 异常)
    LocalTensor<float> blkSortTmp = tmp[11776]; // 与 blkPairs 不重叠 (MrgSort src/tmp 禁止重叠)
    QLIV2ServiceVec::SortAll(blkPairs, blkSortTmp, blockNumPad);
    PipeBarrier<PIPE_V>();
    // 候选专用合并: 纯 V 回拷 (MergeSort 尾部 UB->UB DataCopy 不受 PipeBarrier<PIPE_V> fence,
    // 多 tile 下一次 MrgSort 读 acc 存在调度敏感竞态, 实测 tile1 读到 stale 数据)
    QLIV2ServiceVec::MergeSortVecCopy(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE],
                                      constInfo_.candidateTopkBlocks, blkPairs, blockNumPad, blkSortTmp);
    PipeBarrier<PIPE_V>();
}

// mode=1: 行末直出 candidate_topk_index 并复位块级累加器
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CopyOutCandTopkIndex(const QLIV2Common::RunInfo& info, int32_t cuS1Idx,
                                                                 int32_t innerS1Idx)
{
    LocalTensor<uint32_t> candIdxULocal = outQueue_.AllocTensor<uint32_t>();
    ExtractIndex(candIdxULocal,
                 globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE].template ReinterpretCast<uint32_t>(),
                 constInfo_.candidateTopkBlocks);
    PipeBarrier<PIPE_V>();
    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
    outQueue_.EnQue<uint32_t>(candIdxULocal);
    candIdxULocal = outQueue_.DeQue<uint32_t>();
    QLIV2ServiceVec::CopyOut(
        candidateTopkIndexOutGm[info.candidateOutOffset + cuS1Idx * constInfo_.candidateTopkBlocks],
        candIdxULocal.template ReinterpretCast<int32_t>(), constInfo_.candidateTopkBlocks);
    // 每行有效候选数：candidate 输出行布局为 [..., N2]，行偏移 = 候选块偏移 / 候选容量。
    // 必须使用独立 UB 槽：复用候选索引 UB 会在 MTE3 尚未搬完时改写首元素，
    // 导致 candidate_block_indices[0] 被长度值覆盖（表现为首块 = 块数）。
    LocalTensor<int32_t> lenUb = candidateLenBuf_.Get<int32_t>();
    lenUb.SetValue(0, GetCandidateLength(info.actS2Size));
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    uint64_t lengthRowOffset =
        info.candidateOutOffset / constInfo_.candidateTopkBlocks + static_cast<uint64_t>(cuS1Idx) * constInfo_.kHeadNum;
    AscendC::DataCopyParams lengthParams;
    lengthParams.blockCount = 1;
    lengthParams.blockLen = sizeof(int32_t);
    lengthParams.srcStride = 0;
    lengthParams.dstStride = 0;
    AscendC::DataCopyPad(candidateBlockLengthGm[lengthRowOffset], lenUb, lengthParams);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(candIdxULocal);
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
        if (constInfo_.candidateOn) {
            InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
        }
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
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (constInfo_.dcpEnable) {
            // DCP: 全局可见上界 u = C - Q_len + row + 1, 再映射成本地可见前缀
            int64_t uSigned = static_cast<int64_t>(info.causalGlobalLen) - static_cast<int64_t>(info.actS1Size) +
                              static_cast<int64_t>(cuS1Idx) + 1;
            uint32_t u = (uSigned <= 0) ? 0U : static_cast<uint32_t>(uSigned);
            cuRealAcSeq = static_cast<int32_t>(QLIV2Common::CalcDcpVisibleLen(u, static_cast<uint32_t>(info.actS2Size),
                                                                              constInfo_.cpWorld, constInfo_.cpRank,
                                                                              constInfo_.cpInterleaveSize));
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        // 当前vec1ResGm对应S1的位置
        // LD 布局 = 每核 s1Base 行预留区 (起点 = saveWorkSpaceIdx * s1Base * pairW)。
        // 槽内行号必须用**块内相对行号** blockRowIdx (= cuS1Idx - cuBaseS1Idx)：
        // 归约侧 ProcessLD 按 metadata 的 fdMStart（块内 0 起）+ j 读行槽；若用 batch
        // 全局行号 cuS1Idx，gS1Idx≥1 时会整体错位 gS1Idx×s1Base——溢写落到下一个分片
        // 槽的行区、归约读到陈旧数据（910B 开 LD 后表现为相邻行交换/整行错误）。
        int32_t blockRowIdx = cuS1Idx - cuBaseS1Idx;
        uint64_t wsOffset = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * topkPairWidth_ +
                            static_cast<uint64_t>(blockRowIdx) * topkPairWidth_;
        // candidate 归约槽: 同一工作槽与块内相对行, 间距换成候选对宽
        uint64_t wsOffsetCand = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * candPairWidth_ +
                                static_cast<uint64_t>(blockRowIdx) * candPairWidth_;
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
            // A15: output_idx_offset 在索引成形阶段施加（对齐 arch35: 偏移先于 -1 填充）——
            // 累加器的 (-inf,-1) 垫底对不经过本加法, 输出 padding 恒为 -1;
            // candidate_topk_index 走独立路径 (相对块号), 不受影响。
            int32_t rowOff = 0;
            if (isOutputIdxOffsetValid_) {
                rowOff = outputIdxOffsetGm_.GetValue(info.outputIdxOffsetCoreOffset + cuS1Idx * kHeadNum_);
            }
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx) + rowOff, cuS2Len);
            PipeBarrier<PIPE_V>();
            // R11 (leak 语义, §1.6): 候选块外索引不再置 -1 —
            // 分数已在 S4a 降级 NEG_HUGE (仍高于 -inf: 不足 topk 时作为填充入选, 排候选内之后),
            // 索引保留真实位置号, 与模型 where(idxs < compress_lens, idxs + offset, -1) 等价:
            // 仅不可达位置 (score -inf 沉底 + 尾部 -1 填充) 输出 -1。
            // candidate 开启: 块化 amax + pin + 块级排序归并 (S5a/S6a)
            if (constInfo_.candidateOn) {
                ProcessCandBlockTopk(info, cuS1Idx, cuS2Len, cuS2LenVecAlign, cuRealAcSeq, innerS1Idx);
            }
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
                // return_value: 先从排序缓冲抽取偶数 lane 的分数（此时缓冲未被清空）
                LocalTensor<float> valueUb;
                if (constInfo_.returnValue != 0U) {
                    valueUb = valueOutBuf_.Get<float>();
                    ExtractValue(valueUb.template ReinterpretCast<uint32_t>(),
                                 globalTopkUb_[innerS1Idx * topkPairWidth_].template ReinterpretCast<uint32_t>(),
                                 topkWidth_);
                }
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                outQueue_.EnQue<uint32_t>(idxULocal);
                idxULocal = outQueue_.DeQue<uint32_t>();
                QLIV2ServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                         idxULocal.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
                outQueue_.FreeTensor(idxULocal);
                // return_value: FP32 分数 Cast 成 BF16 写出（与 arch35 输出 dtype 对齐）
                if (constInfo_.returnValue != 0U) {
                    LocalTensor<bfloat16_t> valBf16 = outQueue_.AllocTensor<bfloat16_t>();
                    AscendC::Cast(valBf16, valueUb, AscendC::RoundMode::CAST_RINT,
                                  static_cast<int32_t>(constInfo_.sparseCount));
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                    QLIV2ServiceVec::CopyOut(valueOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                             valBf16, static_cast<int64_t>(constInfo_.sparseCount));
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                    outQueue_.FreeTensor(valBf16);
                }
                // candidate 开启: 块级 topk 直出 candidate_topk_index
                if (constInfo_.candidateOn) {
                    CopyOutCandTopkIndex(info, cuS1Idx, innerS1Idx);
                }
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
                if (constInfo_.candidateOn) {
                    // 块级候选同样落工作区 (s2 切分行没有核能独占整行, 由 LD 核归并直出)
                    AscendC::DataCopyExtParams candWsParams;
                    candWsParams.blockLen = candPairWidth_ * sizeof(float);
                    candWsParams.srcStride = 0;
                    candWsParams.dstStride = 0;
                    candWsParams.blockCount = 1;
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                    AscendC::DataCopyPad(candidateResGm[wsOffsetCand],
                                         globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], candWsParams);
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
                    PipeBarrier<PIPE_V>();
                }
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
                if (constInfo_.candidateOn) {
                    // 本行无有效位置 → 候选槽同样落一份"无候选"(-inf/-1) 供归并
                    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
                    PipeBarrier<PIPE_V>();
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                    QLIV2ServiceVec::CopyOut(candidateResGm[wsOffsetCand],
                                             globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], candPairWidth_);
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                }
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
                if (constInfo_.candidateOn) {
                    QLIV2ServiceVec::CopyOut(candidateResGm[wsOffsetCand],
                                             globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], candPairWidth_);
                    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
                    PipeBarrier<PIPE_V>();
                }
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
    // P10: 分段槽按核分布 (核 i 的区起点 = i*s1Base 槽), 跨 3 列表的间距 = (s1Base-1) 个槽。
    // DataCopyExtParams.srcStride 单位 = 32B 块: (s1Base-1)*pairW*4B/32。
    // 注意: P10 当前逐槽显式拷贝 (blockCount=1), 该值仅在改回多块单次拷贝时生效。
    indexValueParams.srcStride = (s1BaseSize_ - 1U) * topkPairWidth_ * sizeof(int32_t) / 32U;
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
            params.validBit = MRG_VALID_BIT_4;
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
            indexValueParams.blockCount = 1;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            // P10 同型问题: 尾段逐槽显式拷贝 (多块单次拷贝依赖 srcStride 语义, 实测会漏掉最后一槽)
            for (uint32_t k = 0; k < ldTailLen; k++) {
                uint64_t segSlot = wsOffsetTail + static_cast<uint64_t>(k) * s1BaseSize_ * topkPairWidth_;
                AscendC::DataCopyPad(curValueIdxUb[valueOffset + k * topkPairWidth_], vec1ResGm[segSlot],
                                     indexValueParams, indexValuePadParams);
            }
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = topkWidth_;
            params.elementLengths[1] = topkWidth_;
            params.elementLengths[2] = topkWidth_;
            params.elementLengths[3] = topkWidth_;
            params.ifExhaustedSuspension = true;
            if (ldTailLen == 1) {
                params.validBit = MRG_VALID_BIT_2;
            } else if (ldTailLen == 2) {
                params.validBit = MRG_VALID_BIT_3;
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
        if (constInfo_.returnValue != 0U) {
            // DCP: LD 归并后的分数同步转 BF16，与 ProcessVec1 直出路径保持同一 dtype。
            LocalTensor<bfloat16_t> outValBf16 = ldOutBf16Buf_.Get<bfloat16_t>();
            AscendC::Cast(outValBf16, outValueUb, AscendC::RoundMode::CAST_RINT,
                          static_cast<int32_t>(constInfo_.sparseCount));
            PipeBarrier<PIPE_V>();
        }
        InitSortOutBuf(curValueIdxUb, topkPairWidth_);
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        // A15: output_idx_offset 已在 ProcessVec 的索引成形阶段施加, LD 归并只做 (value,index) 合并,
        // -1 垫底槽不经过偏移（对齐 arch35: 偏移先于 -1 填充）。
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        uint64_t outOffset =
            ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) * constInfo_.kHeadNum * constInfo_.sparseCount;
        AscendC::DataCopyPad(indiceOutGm[outOffset], idxULocal1, copyOutParams);
        if (constInfo_.returnValue != 0U) {
            LocalTensor<bfloat16_t> outValBf16 = ldOutBf16Buf_.Get<bfloat16_t>();
            QLIV2ServiceVec::CopyOut(valueOutGm[outOffset], outValBf16, static_cast<int64_t>(constInfo_.sparseCount));
        }
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

// candidate(两级TopK 第一级): s2 被 metadata 切给多核的行, 归并各分段核落工作区的块级候选并直出。
// 与 ProcessLD 同构 (元素宽度换成 candidateTopkBlocks 对), 由 LD 核在 SyncAll 之后调用;
// 输出为相对块号, 不加 output_idx_offset (与 ProcessVec1 的直出契约一致)。
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessLDCand()
{
    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();
    LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = constInfo_.candidateTopkBlocks * sizeof(int32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<float> indexValuePadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams indexValueParams;
    indexValueParams.blockLen = candPairWidth_ * sizeof(int32_t); // bytes (value+idx 对)
    // 同 ProcessLD: 跨槽间距为 (s1Base-1) 个槽；当前逐槽拷贝 (blockCount=1)，仅多块拷贝时生效
    indexValueParams.srcStride = (s1BaseSize_ - 1U) * candPairWidth_ * sizeof(int32_t) / 32U;
    indexValueParams.dstStride = 0;

    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    uint32_t ldProcessLen = 4; // 4: 4块归约任务做一次Merge
    uint32_t ldProcessNum = (ldProWorkspaceNum - 1) / (ldProcessLen - 1);
    uint32_t ldTailLen = ldProWorkspaceNum - (ldProcessNum * (ldProcessLen - 1) + 1); // 尾块长度

    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        indexValueParams.blockCount = 1;
        uint64_t wsOffsetIni = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * candPairWidth_ +
                               static_cast<uint64_t>(ldInfo_.mStart + j) * candPairWidth_;
        AscendC::DataCopyPad(curValueIdxUb, candidateResGm[wsOffsetIni], indexValueParams, indexValuePadParams);
        uint64_t valueOffset = candPairWidth_;
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // 处理等于4的部分
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            uint64_t wsOffset = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * candPairWidth_ +
                                static_cast<uint64_t>(i * (ldProcessLen - 1) + 1) * s1BaseSize_ * candPairWidth_ +
                                static_cast<uint64_t>(ldInfo_.mStart + j) * candPairWidth_;
            indexValueParams.blockCount = 1;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            for (uint32_t k = 0; k < ldProcessLen - 1U; k++) {
                uint64_t segSlot = wsOffset + static_cast<uint64_t>(k) * s1BaseSize_ * candPairWidth_;
                AscendC::DataCopyPad(curValueIdxUb[valueOffset + k * candPairWidth_], candidateResGm[segSlot],
                                     indexValueParams, indexValuePadParams);
            }
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = constInfo_.candidateTopkBlocks;
            params.elementLengths[1] = constInfo_.candidateTopkBlocks;
            params.elementLengths[2] = constInfo_.candidateTopkBlocks;
            params.elementLengths[3] = constInfo_.candidateTopkBlocks;
            params.ifExhaustedSuspension = true;
            params.validBit = MRG_VALID_BIT_4;
            params.repeatTimes = 1;

            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[candPairWidth_];
            srcList.src3 = curValueIdxUb[2 * candPairWidth_];
            srcList.src4 = curValueIdxUb[3 * candPairWidth_];
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, candPairWidth_);
            PipeBarrier<PIPE_V>();
        }

        // 处理不等于4的部分
        if (ldTailLen != 0) {
            uint64_t wsOffsetTail =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * candPairWidth_ +
                static_cast<uint64_t>(ldProcessNum * (ldProcessLen - 1) + 1) * s1BaseSize_ * candPairWidth_ +
                static_cast<uint64_t>(ldInfo_.mStart + j) * candPairWidth_;
            indexValueParams.blockCount = 1;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            // P10: 尾段同样逐槽显式拷贝 (多块单次拷贝依赖 srcStride 语义, 实测会漏掉最后一槽)
            for (uint32_t k = 0; k < ldTailLen; k++) {
                uint64_t segSlot = wsOffsetTail + static_cast<uint64_t>(k) * s1BaseSize_ * candPairWidth_;
                AscendC::DataCopyPad(curValueIdxUb[valueOffset + k * candPairWidth_], candidateResGm[segSlot],
                                     indexValueParams, indexValuePadParams);
            }
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = constInfo_.candidateTopkBlocks;
            params.elementLengths[1] = constInfo_.candidateTopkBlocks;
            params.elementLengths[2] = constInfo_.candidateTopkBlocks;
            params.elementLengths[3] = constInfo_.candidateTopkBlocks;
            params.ifExhaustedSuspension = true;
            if (ldTailLen == 1) {
                params.validBit = MRG_VALID_BIT_2;
            } else if (ldTailLen == 2) {
                params.validBit = MRG_VALID_BIT_3;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[candPairWidth_];
            srcList.src3 = curValueIdxUb[2 * candPairWidth_];
            srcList.src4 = curValueIdxUb[3 * candPairWidth_];
            PipeBarrier<PIPE_V>();
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, candPairWidth_);
            PipeBarrier<PIPE_V>();
        }

        // 搬出块号 (相对块号契约: 不加 output_idx_offset)
        ExtractIndex(outIdxUb, curValueIdxUb.template ReinterpretCast<uint32_t>(), constInfo_.candidateTopkBlocks);
        PipeBarrier<PIPE_V>();
        InitSortOutBuf(curValueIdxUb, candPairWidth_);
        LocalTensor<int32_t> idxULocal = outIdxUb.template ReinterpretCast<int32_t>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        uint64_t outOffset = ldInfo_.candidateOutCoreOffset + static_cast<uint64_t>(ldInfo_.mStart + j) *
                                                                  constInfo_.kHeadNum * constInfo_.candidateTopkBlocks;
        AscendC::DataCopyPad(candidateTopkIndexOutGm[outOffset], idxULocal, copyOutParams);
        uint64_t lengthRowOffset = ldInfo_.candidateOutCoreOffset / constInfo_.candidateTopkBlocks +
                                   static_cast<uint64_t>(ldInfo_.mStart + j) * constInfo_.kHeadNum;
        LocalTensor<int32_t> lenUb = candidateLenBuf_.Get<int32_t>();
        lenUb.SetValue(0, GetCandidateLength(ldInfo_.actS2Size));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        AscendC::DataCopyParams lengthParams;
        lengthParams.blockCount = 1;
        lengthParams.blockLen = sizeof(int32_t);
        lengthParams.srcStride = 0;
        lengthParams.dstStride = 0;
        AscendC::DataCopyPad(candidateBlockLengthGm[lengthRowOffset], lenUb, lengthParams);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
