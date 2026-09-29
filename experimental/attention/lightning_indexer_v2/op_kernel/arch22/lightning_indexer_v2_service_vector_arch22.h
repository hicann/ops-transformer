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
 * \file lightning_indexer_v2_service_vector_arch22.h
 * \brief Vector核服务类：加权ReLU求和(DoScale/DoReduce)、TopK排序(SortAll/MergeSort)、LD归约(ProcessLD)、输出写回
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_VECTOR_ARCH22_H
#define LIGHTNING_INDEXER_V2_SERVICE_VECTOR_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_arch22.h"
#include "lightning_indexer_v2_vector.h"

namespace LIV2Kernel {
using namespace LIV2Common;
using namespace LIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;

// 相邻配对解交织（candidate 块内 amax 折叠树用）：dst[i] = src[2i]（even）或 src[2i+1]（odd）。
// GatherMask pattern 1/2 为 32B 块内偶/奇索引筛选（8 选 4 紧凑输出），与 ExtractIndex 同族用法。
__aicore__ inline void GatherMaskDeinterleave(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                              int32_t srcNum, bool even)
{
    AscendC::GatherMaskParams params;
    params.repeatTimes = (srcNum * static_cast<int32_t>(sizeof(float)) + 255) / 256; // 256B per repeat
    params.src0BlockStride = 1;
    params.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    params.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;
    uint8_t pattern = even ? 1 : 2;
    AscendC::GatherMask(dst, src, pattern, false, static_cast<uint32_t>(0), params, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename LIT>
class LightningIndexerV2ServiceVector {
public:
    // ================================ 类型定义区 ================================
    // 中间计算数据类型为float，高精度模式
    using K_T = typename LIT::keyType;
    using OUT_V_T = float;
    using W_T = float;
    static constexpr LI_V2_LAYOUT LAYOUT_T = LIT::layout;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline LightningIndexerV2ServiceVector() {}
    __aicore__ inline void ProcessVec(const LIV2Common::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                      const LIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<OUT_V_T> valueOutGm);
    // candidate (two-level topk)：source 模式输出 GM 绑定（输入 GM 本算子恒空，签名保留供 sparse 克隆对齐）
    __aicore__ inline void InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                  GlobalTensor<int32_t> candidateTopkIndexOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);
    // candidate (two-level topk) source 链
    __aicore__ inline void ProcessCandBlockTopk(const LIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t cuS2Len,
                                                int32_t cuRealAcSeq, int32_t innerS1Idx,
                                                const LocalTensor<float> &sortScoreUb);
    __aicore__ inline void CopyOutCandTopkIndex(const LIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t innerS1Idx);
    // candidate source 全选快速路径行末直出（可达块数 ≤ K：输出恒为 [0..N) + (-1) 补齐）
    __aicore__ inline void CopyOutCandTopkIndexFullSel(const LIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                       int32_t realBlockNumTotal);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<OUT_V_T> valueOutGm;
    GlobalTensor<int32_t> candidateTopkIndexInGm;  // 本算子恒空（consumer 链归 sparse 算子）
    GlobalTensor<int32_t> candidateTopkIndexOutGm; // source 输出
    // ================================ 常量区 ================================

private:
    // ================================ Local Buffer区 ================================
    // queue
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    // tmp buff for candidate (two-level topk) source：块级 topk 累加器 [CeilDiv(s1BaseSize,2), 2048, 2]
    TBuf<TPosition::VECCALC> blockSortOutBuf_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;
    LocalTensor<float> globalBlockTopkUb_; // candidate source 块级累加器（mode=1 分配）

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);
    // candidate source 块级累加器每行容量（value+idx 对）：BASE_TOPK 对
    constexpr static uint32_t CAND_BLOCK_ACC_PAIR_SIZE = BASE_TOPK * 2;

    struct LIV2Common::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.sparseCount : BASE_TOPK;
    pipe->InitBuffer(outQueue_, 1, outNeedBufSize); // 32KB  extract
    // 68KB: cube结果和weight搬运db(2x34KB), mrgsort临时UB
    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float)); // 64KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                           // 2KB
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 3 * sizeof(float));                     // 6KB
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));

    //
    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    // candidate on（s1Base=4）时 sortOutBuf 仅 8192 floats，SortedBasicBlock_ 偏移越界；
    // D6 保证该缓存粗排路径不再执行，on 时不绑定该视图
    if (constInfo_.candidateMode != CANDIDATE_MODE_SOURCE) {
        SortedBasicBlock_ = globalTopkUb_[virTopK * 2 * 2];
    }
    globalTopkNum_ = 0;

    // candidate (two-level topk) source：块级累加器分配与初始化（off 不占 UB）
    // UB 预算：off 基础 ≈141.6KB + 32KB = ≈173.6KB ≤ 192KB（s1Base=4 置换 sortOutBuf 64→32KB）
    if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
        pipe->InitBuffer(blockSortOutBuf_, CeilDiv(s1BaseSize_, 2) * CAND_BLOCK_ACC_PAIR_SIZE * sizeof(float)); // 32KB
        globalBlockTopkUb_ = blockSortOutBuf_.Get<float>();
        InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * CAND_BLOCK_ACC_PAIR_SIZE);
    }

    // 基本块执行前初始化UB和GM
    // ArithProgression 豁免边界（R1 检视答复）：count 恒 s2BaseSize_=512（>64，不进 (8,64] 标量分支），
    // 且位于 InitBuffers（流水开始前，无在飞 V 指令）——流水中段禁用（见 ProcessCandBlockTopk 禁则注释）
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // globalTopkUb_ 布局 [CeilDiv(s1BaseSize_, 2), virTopK, 2]，填充交错的 -inf/-1（value/index 对）
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);

    // vec1ParamGm 整区初始化为 -1：LD 轮询标志 needFd=-1 表示尚无待处理数据，vector 侧写 WS 后置 1
    // vec1ResIn32Gm = [aic, 2, s1BaseSize_, 16] int32
    // ws清零 [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, ......]
    LocalTensor<float> tmpfBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpfBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +      // 2个AIV共同地址偏移
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_; // 每个AIV的地址偏移，S1方向
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpfBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(tmpfBuff);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                                                        const LIV2TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;

    // group ub 切分因子当前按照UB空间强制为16
    groupInner_ = 16;

    blockId_ = GetBlockIdx();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm, GlobalTensor<int64_t> vec1ParamGm,
    GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm, GlobalTensor<OUT_V_T> valueOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitVecCandidateTensor(
    GlobalTensor<int32_t> candidateTopkIndexInGm, GlobalTensor<int32_t> candidateTopkIndexOutGm)
{
    this->candidateTopkIndexInGm = candidateTopkIndexInGm;
    this->candidateTopkIndexOutGm = candidateTopkIndexOutGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output

    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    LIServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);

    if (constInfo_.returnValue) {
        LocalTensor<uint32_t> valueULocal = outQueue_.AllocTensor<uint32_t>();
        Duplicate(valueULocal, constInfo_.NEG_INF_FLOAT, constInfo_.sparseCount);
        outQueue_.EnQue<uint32_t>(valueULocal);
        valueULocal = outQueue_.DeQue<uint32_t>();

        GlobalTensor<uint32_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t *)valueOutGm.GetPhyAddr());
        LIServiceVec::CopyOut(valueOutGmTmp[invalidS1offset], valueULocal, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }
    // candidate (two-level topk) source：同步对 candidate_topk_indices 填 -1
    // （行偏移 = sparse 行偏移 / sparseCount * candidateTopkBlocks，调用侧保证 offset 为 sparseCount 整数倍）
    if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE && candidateTopkIndexOutGm.GetPhyAddr() != 0) {
        uint64_t candOffset =
            static_cast<uint64_t>(invalidS1offset) / constInfo_.sparseCount * constInfo_.candidateTopkBlocks;
        LocalTensor<float> candULocal = outQueue_.AllocTensor<float>();
        LocalTensor<int32_t> candIdxLocal = candULocal.template ReinterpretCast<int32_t>();
        Duplicate(candIdxLocal, constInfo_.INVALID_IDX, constInfo_.candidateTopkBlocks);
        outQueue_.EnQue<float>(candULocal);
        candULocal = outQueue_.DeQue<float>();
        LIServiceVec::CopyOut(candidateTopkIndexOutGm[candOffset], candIdxLocal, constInfo_.candidateTopkBlocks);
        outQueue_.FreeTensor(candULocal);
    }
}

// candidate (two-level topk) source 链 S5a/S6a：块化 amax + pad(-inf) + pin(+inf) + 块级排序/归并。
// 输入 sortScoreUb：当前 tile 的 score 行 [0, s2BaseSize)（已 -inf pad，cuS2Len 之后为 -inf）。
// 自 QLI v2 ProcessCandBlockTopk 移植，两处适配：
// 1) 块内 amax 按 blockSize（[2,64] 2 的幂）通用化：bs>=8 走 BlockReduceMax(32B 块 8:1) +
//    log2(bs/8) 轮折叠；bs<8 时 BlockReduceMax 粒度不足，改走 log2(bs) 轮 GatherMask 相邻配对 Max 树；
// 2) scratch 不用 tmpBuf_：LIV2 的 CopyIn 是 ping/pong 双缓冲 MTE2 交叠流水，下一 tile 的 MTE2 写在
//    WaitFlag<V_MTE2>(ping)（本 tile DoScale(ping) 末置位）后即可启动，早于本函数的 V 读——tmpBuf_
//    全域均为 CopyIn 目标，S5a 期间使用会被踩（QLI 为 SetWaitFlag 全排空模式无此问题）。故 scratch
//    全部取自 outQueue_ 窗口（sort 阶段空闲；先于位置级 tmpSortBuf 分配，串行不重叠）。
template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessCandBlockTopk(const LIV2Common::RunInfo &info,
                                                                                  int32_t cuS1Idx, int32_t cuS2Len,
                                                                                  int32_t cuRealAcSeq,
                                                                                  int32_t innerS1Idx,
                                                                                  const LocalTensor<float> &sortScoreUb)
{
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t blkLen = s2BaseSize_;          // LIV2: cuS2LenVecAlign 恒 512（-inf pad 后的对齐长度）
    int32_t blockNum = blkLen / blockSize; // 含尾块（pad -inf 后按对齐长度计）
    int32_t realBlockNum = (cuS2Len + blockSize - 1) / blockSize; // 含有效位置的块数
    // 向量 count 按 64 对齐（块数不足 64 补齐，stale 区由 pad 链归一 -inf 覆盖）
    int32_t blockNumPad = (blockNum < 64) ? 64 : blockNum;
    int32_t tileBlockBase = info.s2Idx * s2BaseSize_ / blockSize;

    // ---- scratch 偏移地图（outQueue_ 窗口 ≥8256 floats）----
    // 固定区 [0,1024)：treePing[0,256) treePong[256,512) gatherEven[512,768) gatherOdd[768,1024)
    // 动态区 P=1024 起（blockNumPad 对齐）：blkPairs[scores|idx] + idxScr/thrI/offI/tI/pRaw/pNeg 各
    // blockNumPad + blkSortTmp(2*(candidateTopkBlocks+blockNumPad))；
    // 最坏 bs=2/candBlocks=2048：1024 + 8*256 + 2*(2048+256) = 7680 ≤ 8256 ✓
    // 【白盒检视 §2.4 加固 2026-09-23】"需随 outQueue 尺寸变化复审计"由下述编译期断言机制化保障：
    // 尾偏移 = 1024 + 10*blockNumPad + 2*candidateTopkBlocks，blockNumPad 在 bs=2 取最大 256
    constexpr int32_t CAND_SCRATCH_BLK_PAD_MAX = 256;   // s2BaseSize(512) / candidate_block_size 最小值(2)
    constexpr int32_t CAND_SCRATCH_TOPK_BLK_MAX = 2048; // host 硬校验上界（CANDIDATE_TOPK_BLOCKS_FIX）
    constexpr int32_t CAND_SCRATCH_WORST_FLOATS = 1024 + 10 * CAND_SCRATCH_BLK_PAD_MAX + 2 * CAND_SCRATCH_TOPK_BLK_MAX;
    // 与 InitBuffers outNeedBufSize 同源：max(BASE_TOPK*2*2*4B, 256B + groupInner(16)*s2Base(512)*4B)/4B
    constexpr int32_t OUT_QUEUE_CAPACITY_FLOATS = (256 + 16 * 512 * 4) / 4;
    static_assert(CAND_SCRATCH_WORST_FLOATS <= OUT_QUEUE_CAPACITY_FLOATS,
                  "cand block-level scratch worst case exceeds outQueue single-slot capacity");
    LocalTensor<float> candScratch = outQueue_.AllocTensor<float>();
    LocalTensor<float> treePing = candScratch[0];
    LocalTensor<float> treePong = candScratch[256];
    LocalTensor<float> gatherEven = candScratch[512];
    LocalTensor<float> gatherOdd = candScratch[768];
    int32_t p = 1024;
    LocalTensor<float> blkPairs = candScratch[p];
    LocalTensor<int32_t> idxScr = candScratch[p + 2 * blockNumPad].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> thrI = candScratch[p + 3 * blockNumPad].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> offI = candScratch[p + 4 * blockNumPad].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> tI = candScratch[p + 5 * blockNumPad].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> pRaw = candScratch[p + 6 * blockNumPad].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> pNeg = candScratch[p + 7 * blockNumPad].template ReinterpretCast<int32_t>();
    LocalTensor<float> blkSortTmp = candScratch[p + 8 * blockNumPad];

    // ---- (a) S5a 块内 amax（结果落于 treePing/treePong [0, blockNum)）----
    int32_t fold = (blockSize >= 8) ? (blockSize / 8) : blockSize; // 需折叠的因子
    int32_t rounds = 0;
    for (int32_t v = fold; v > 1; v >>= 1) {
        rounds++;
    }
    int32_t curSrcLen = (blockSize >= 8) ? (blkLen / 8) : blkLen;
    LocalTensor<float> prev;
    if (blockSize >= 8) {
        // BlockReduceMax：每 32B 块（8 fp32）出 1 个 max，紧凑输出 [0, blkLen/8)
        BlockReduceMax(treePing, sortScoreUb, CeilDiv(blkLen, 64), 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        prev = treePing;
    } else {
        prev = sortScoreUb; // 只读源
    }
    for (int32_t r = 0; r < rounds; r++) {
        LocalTensor<float> dst = ((r % 2) == 0) ? treePing : treePong;
        // bs>=8 且 rounds>=1 时首轮 dst 与 prev 同为 treePing：两路 GatherMask 先完成全量读
        // （V 管道内有序 + PipeBarrier），其后 Max 再写 [0, curSrcLen/2) 无冲突
        GatherMaskDeinterleave(gatherEven, prev, curSrcLen, true);
        GatherMaskDeinterleave(gatherOdd, prev, curSrcLen, false);
        Max(dst, gatherEven, gatherOdd, curSrcLen / 2);
        PipeBarrier<PIPE_V>();
        prev = dst;
        curSrcLen /= 2;
    }
    // amax 结果拷入 blkPairs 值半区（count=blockNumPad，[blockNum, blockNumPad) stale 由 pad 链归一）
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Adds(blkPairs[off], prev[off], 0.0f, 64);
    }
    PipeBarrier<PIPE_V>();

    // ---- (b) 块级 [scores | idx] 对构造：idx = tileBlockBase + j；j >= realBlockNum 的 pad 块置 -1 ----
    // 全 int32 向量算术链（禁标量 SetValue：不受 PipeBarrier fence，与后续 V 读存在竞态；
    // 禁 s32->f32 Cast：v220 无此组合）。idx' = idx - (idx+1)*isPad，isPad = clamp(idx - thr, 0, 1)，
    // thr = tileBlockBase + realBlockNum - 1
    LocalTensor<int32_t> blkIdx = blkPairs[blockNumPad].template ReinterpretCast<int32_t>();
    // 禁 ArithProgression：其 (8,64] count 分支内部走标量 SetValue 写 UB（不受 PipeBarrier fence，
    // S_V 时序敏感）且将全局向量掩码泄漏为 8 lane（该分支无 ResetMask）——流水中间调用会污染后续
    // 位置级排序/归并（实测 sparse_indices 出现"单元素插入+右移"型随机错误）。
    // 改为复用 InitBuffers 已构建的常量索引 globalTopkIndice_ [0,512) + 一次纯 V 的 count-based Adds。
    Adds(idxScr, globalTopkIndice_, tileBlockBase, blockNumPad);
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
        Mins(offI[off], offI[off], 1, 64); // isPad
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

    // ---- (c) pad 块 score 归一为 -inf 位型：覆盖 [realBlockNum, blockNumPad)，含 stale 区 ----
    // score' = score + (score - NINF_bits)*(-isPad)，isPad 于 pad 位为 1；全 int32 向量
    // （脏 score 的 (-inf,-1) 不同构对会挤进 topk 挤掉真实块）
    if (blockNumPad > realBlockNum) {
        LocalTensor<int32_t> scoreI = blkPairs.template ReinterpretCast<int32_t>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Duplicate(pRaw[off], -1, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pRaw[off], pRaw[off], offI[off], 64); // -isPad
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pNeg[off], scoreI[off], 8388608, 64); // score - NINF_bits（NINF=0xFF800000=-8388608）
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

    // ---- (d) pin 最新 token 所在块（+inf 位型注入）：纯 int32 向量，复用 isPad 链已释放 scratch ----
    // flag = -1 于 pin 位、0 于其余：score' = score + (score - PINF_bits) * flag，
    // pin 位得 PINF_bits（0x7F800000 = +inf），其余不变（flag=0 使 pad 位 -inf 的中间溢出无害）
    int32_t lastBlk = (cuRealAcSeq - 1) / blockSize;
    int32_t pinLocal = lastBlk - tileBlockBase;
    if (pinLocal >= 0 && pinLocal < realBlockNum) {
        LocalTensor<int32_t> pA = thrI; // 复用 (b) 链 scratch
        LocalTensor<int32_t> pB = offI;
        LocalTensor<int32_t> pFlag = tI;
        LocalTensor<int32_t> scoreI = blkPairs.template ReinterpretCast<int32_t>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            // idxScr 为全局块号，pin 必须用全局 id（tileBlockBase + pinLocal）比较
            Adds(pRaw[off], idxScr[off], -(tileBlockBase + pinLocal), 64); // 0 于 pin 块
        }
        PipeBarrier<PIPE_V>();
        Duplicate(pNeg, 0, blockNumPad);
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Sub(pNeg[off], pNeg[off], pRaw[off], 64); // -(idx-pin)
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
            Add(pFlag[off], pA[off], pB[off], 64); // isNotPin（0/1 互斥）
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pFlag[off], pFlag[off], -1, 64); // -1 于 pin，0 其余
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pRaw[off], scoreI[off], -2139095040, 64); // score - 0x7F800000（复用 pRaw）
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

    // ---- (e) S6a 块级排序 + 归并到块级累加器 ----
    // blkSortTmp 需容纳 SortAll tmp（2*blockNumPad）与 MrgSort tmp（(candidateTopkBlocks+blockNumPad)*2）
    LIServiceVec::SortAll(blkPairs, blkSortTmp, blockNumPad);
    PipeBarrier<PIPE_V>();
    // 候选专用合并（P3）：纯 V 回拷，防跨 tile 累加器 stale 读
    LIServiceVec::MergeSortVecCopy(globalBlockTopkUb_[innerS1Idx * CAND_BLOCK_ACC_PAIR_SIZE],
                                   static_cast<int32_t>(constInfo_.candidateTopkBlocks), blkPairs, blockNumPad,
                                   blkSortTmp);
    PipeBarrier<PIPE_V>();
    outQueue_.FreeTensor(candScratch);
}

// candidate source：行末直出 candidate_topk_indices（前 candidate_topk_blocks 个块号）并复位该行块级累加器
template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::CopyOutCandTopkIndex(const LIV2Common::RunInfo &info,
                                                                                  int32_t cuS1Idx, int32_t innerS1Idx)
{
    LocalTensor<uint32_t> candIdxULocal = outQueue_.AllocTensor<uint32_t>();
    ExtractIndex(candIdxULocal,
                 globalBlockTopkUb_[innerS1Idx * CAND_BLOCK_ACC_PAIR_SIZE].template ReinterpretCast<uint32_t>(),
                 static_cast<int64_t>(constInfo_.candidateTopkBlocks));
    PipeBarrier<PIPE_V>();
    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * CAND_BLOCK_ACC_PAIR_SIZE], CAND_BLOCK_ACC_PAIR_SIZE);
    outQueue_.EnQue<uint32_t>(candIdxULocal);
    candIdxULocal = outQueue_.DeQue<uint32_t>();
    LIServiceVec::CopyOut(candidateTopkIndexOutGm[info.candidateOutOffset + cuS1Idx * constInfo_.candidateTopkBlocks],
                          candIdxULocal.template ReinterpretCast<int32_t>(), constInfo_.candidateTopkBlocks);
    outQueue_.FreeTensor(candIdxULocal);
}

// candidate source 全选快速路径行末直出：块号 [0..N) + (-1) 补齐（比对契约为无序集合）。
// 保持 inline 可内联——实测对本函数施加 noinline 会在巨型内联的 ProcessVec 调用点触发 CCE
// 调用边界状态物化（mix_aiv 每 LIT 实例 +13KB，off 模式每 launch 固定 +µs）。
template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::CopyOutCandTopkIndexFullSel(
    const LIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t realBlockNumTotal)
{
    LocalTensor<uint32_t> candIdxULocal = outQueue_.AllocTensor<uint32_t>();
    LocalTensor<int32_t> candIdxI = candIdxULocal.template ReinterpretCast<int32_t>();
    Duplicate(candIdxI, constInfo_.INVALID_IDX, constInfo_.candidateTopkBlocks);
    PipeBarrier<PIPE_V>();
    // [0..N) 块号：复用 InitBuffers 常量索引 globalTopkIndice_ [0,512)，512 分块 count-based
    // Adds（自含尾块掩码；scalar=off 复用同一常量索引源）
    int32_t off = 0;
    while (off < realBlockNumTotal) {
        int32_t chunk = realBlockNumTotal - off;
        chunk = (chunk > s2BaseSize_) ? s2BaseSize_ : chunk;
        Adds(candIdxI[off], globalTopkIndice_, off, chunk);
        off += chunk;
    }
    PipeBarrier<PIPE_V>();
    outQueue_.EnQue<uint32_t>(candIdxULocal);
    candIdxULocal = outQueue_.DeQue<uint32_t>();
    LIServiceVec::CopyOut(candidateTopkIndexOutGm[info.candidateOutOffset + cuS1Idx * constInfo_.candidateTopkBlocks],
                          candIdxI, constInfo_.candidateTopkBlocks);
    outQueue_.FreeTensor(candIdxULocal);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessVec(const LIV2Common::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> mBaseSizeAlign*s2BaseSize + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    // (B,S1,N1,1);(T,N1,1) -> (B,S1,N2,G,1) 当前只切分到S1轴
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
        // candidate source：块级累加器与位置级累加器成对重置（UT 不变量）
        if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
            InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * CAND_BLOCK_ACC_PAIR_SIZE);
        }
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
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();

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
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
            if (constInfo_.isSparseCountOver2K) {
                WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
                int32_t pingpong = outerGidx % 2;
                LocalTensor<float> dbTmpUb = tmpUb_[pingpong * (groupInner_ * s2BaseSize_ + s2BaseSize_)];
                LocalTensor<float> weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                WaitFlag<HardEvent::V_MTE2>(pingpong);
                LocalTensor<W_T> weightsInTUb = weightsInUb.template ReinterpretCast<W_T>();
                if constexpr (!IsSameType<W_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }
                LIServiceVec::CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm,
                                     mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                         outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign,
                                     weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_, procGnum,
                                     info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                SetFlag<HardEvent::MTE2_V>(pingpong);
                WaitFlag<HardEvent::MTE2_V>(pingpong);
                weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, weightsInUb, weightsInTUb,
                                      brcBuf, procGnum, s2BaseSize_, outerGidx);
                // confused reduceOp in DoScale
                // neednot use LIServiceVec::doReduce(mmInUb, reduceOutInner, procGnum, (s2BaseSize_+8));
                SetFlag<HardEvent::V_MTE2>(pingpong);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;

            LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();
            // candidate (two-level topk) source：S5a/S6a 块化 amax + pin + 块级排序/归并
            // （score 行已成形、idx 已填充；块级计算不得影响位置级结果——两套独立累加器）
            // 方案1 全选快速路径（按行判定，tile 间稳定——cuRealAcSeq 不随 tile 变化）：
            // 可达块数 ≤ K 时输出与 score 无关（"全选"为 golden/CPU 自检断言的不变量，pin 块
            // 天然含于全选集合，-1 槽数 = K − N），整行跳过块级 amax/排序/归并流水，行末直出。
            // 判定值短作用域（不跨排序区存活）——长存活局部会加剧 ProcessVec 寄存器压力，
            // 实测缓存粗排路径（off/decode）回退 +µs
            if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                int32_t candBs = static_cast<int32_t>(constInfo_.candidateBlockSize);
                int32_t candRealBlkTotal = (cuRealAcSeq + candBs - 1) / candBs;
                if (candRealBlkTotal > static_cast<int32_t>(constInfo_.candidateTopkBlocks)) {
                    ProcessCandBlockTopk(info, cuS1Idx, cuS2Len, cuRealAcSeq, innerS1Idx, sortScoreUb);
                }
            }
            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            // D6: candidate 恒走 SortAll+MergeSort（s1Base=4 下 SortedBasicBlock_ 视图越界，缓存粗排禁用）
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K ||
                constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                LIServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                      cuS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
                PipeBarrier<PIPE_V>();
                LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                LIServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                        cuS2LenVecAlign, UbTmpSort);
            } else {
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;

                Sort<float, true>(
                    SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2],
                    reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                    cuS2LenVecAlign / 32);
                AscendC::PipeBarrier<PIPE_V>();
                // 缓存4块512或者S2结束, 需要进行精排
                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2];

                    // 前4块直接精排覆盖到globalTopkUb_
                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], tt,
                                      static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                    } else { // 后面缓存在 SortedBasicBlock_, 先精排, 再merge到globalTopkUb_
                        if (globalTopkUbCacheIdx > 0) {
                            MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                            PipeBarrier<PIPE_V>();
                            DataCopy(SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf,
                                     (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                        }
                        PipeBarrier<PIPE_V>();
                        SparseTopK(globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                   SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf, BASE_TOPK,
                                   s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                    }
                }
            }
            if (constInfo_.isSparseCountOver2K) {
                SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;

            // 中间结果保存
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;

            if (needCopyOutGm) {
                int64_t offset = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? virTopK : constInfo_.sparseCount / 2;
                int64_t copyLen =
                    (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? constInfo_.sparseCount : constInfo_.sparseCount / 2;
                int64_t copyNum = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? 1 : 2;
                for (int64_t i = 0; i < copyNum; i++) {
                    LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                    LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                    Extract(outValueUb, outIdxUb, globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset],
                            (offset / 32));
                    PipeBarrier<PIPE_V>();

                    LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                    outQueue_.EnQue<float>(outValueUb);
                    outValueUb = outQueue_.DeQue<float>();

                    LIServiceVec::CopyOut(
                        indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset], idxULocal1,
                        copyLen);
                    if (constInfo_.returnValue) {
                        LIServiceVec::CopyOut(
                            valueOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset],
                            outValueUb, copyLen);
                    }
                    outQueue_.FreeTensor(outValueUb);
                }
                // candidate (two-level topk) source：行末直出 candidate_topk_indices
                // （直出块数 = candidate_topk_blocks，与位置级 topk 无关；相对块号，不加 output_idx_offset）
                if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                    // 行末重算（短作用域），不依赖跨排序区的长存活局部
                    int32_t candBs = static_cast<int32_t>(constInfo_.candidateBlockSize);
                    int32_t candRealBlkTotal = (cuRealAcSeq + candBs - 1) / candBs;
                    if (candRealBlkTotal <= static_cast<int32_t>(constInfo_.candidateTopkBlocks)) {
                        CopyOutCandTopkIndexFullSel(info, cuS1Idx, candRealBlkTotal);
                    } else {
                        CopyOutCandTopkIndex(info, cuS1Idx, innerS1Idx);
                    }
                }
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, topkOut_] float32
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64
                //     16 = [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, S1ProcNum, ......]

                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK + // 每个AIV的地址偏移，S1方向
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ + // 2个AIV共同地址偏移
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ + // 每个AIV的地址偏移，S1方向
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount));
                // 写入头尾判断
                // [head, tail]
                // head: 与前面规约，与前后规约
                // tail: 与后面规约
                bool isTailReduce = blockS2StartIdx_ == 0; // 一定是isLastTile
                // WS偏移规则 blockS2StartIdx_ != 0
                // 跟前面块做规约 写到0偏移 不用做计算 blockS2StartIdx_ == 0 and !isS2End
                // 跟后面块做规约 写到1偏移  需要 + s1BaseSize_, BASE_TOPK*2
                if (isTailReduce) { // S2不是最后结束的数据就需要往后做规约，放入第二块ws
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                LIServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }

    // BNSD场景无效S1 输出-1
    if (LAYOUT_T == LI_V2_LAYOUT::BSND) {
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

        int32_t invalidS1Num2 = static_cast<int32_t>(info.actS1Size - info.actS2SizeOrig) / constInfo_.cmpRatio;
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

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t bn2Idx;
    int64_t s1Idx;
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息
    // 开始必然没有头规约，因此从尾规约开始处理，while循环读取下一个核的头规约
    // 存满4个list或者遇到S2结尾，则做merge，直到做完S2
    // 每个核都忽略自己的头规约，因为必然由前面的核做完
    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    // S1逐行计算
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 重置偏移
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        // 搬入数据
        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;

        // 获取下一个核规约信息
        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);

        while (needFd == 1) {
            // 搬入头规约数据
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                       innerS1Idx * 2 * 2 * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * BASE_TOPK;
            acc_list_num++;

            // 每满4个list，聚合  前2K为mrg结果
            if (acc_list_num == mrgListNum_) {
                // MrgSort 四条2048的队列，Mrg成一条
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }

            // reduce到S2末尾，则跳出
            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        }

        // mrg不足4个list的数据
        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }

        // 搬出
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        if (!constInfo_.returnValue) {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        } else {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();

            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
            DataCopyPad(valueOutGm[outOffset], outValueUb,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(OUT_V_T)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}
} // namespace LIV2Kernel
#endif // LIGHTNING_INDEXER_V2_SERVICE_VECTOR_ARCH22_H
