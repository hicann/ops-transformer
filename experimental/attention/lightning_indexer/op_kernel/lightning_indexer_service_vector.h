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
 * \file lightning_indexer_service_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_common.h"
#include "lightning_indexer_vector.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
constexpr uint32_t LD_PARAM_NUM = 16;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t MAX_LOGITS_TMP = 0x7F7FFFFF;
constexpr uint32_t LOGITS_SCALE = 4;
template <typename LIT>
class LIVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using K_T = typename LIT::keyType;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline LIVector(){};
    __aicore__ inline void ProcessVec(const LICommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LICommon::ConstInfo &constInfo,
                                      const LITilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<float> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

    // QS 辅助：拼接数据 + 执行 QS + 写回结果
    __aicore__ inline void ExecuteQS(LocalTensor<float> &globScores, LocalTensor<int32_t> &globIndices,
                                     LocalTensor<float> &ovfScores, LocalTensor<int32_t> &ovfIndices,
                                     int32_t cacheScoreBase, int32_t cacheIdxBase, int32_t cachedLen,
                                     LocalTensor<float> &extraScores, LocalTensor<int32_t> &extraIndices,
                                     int32_t extraLen, float sugMax, int32_t innerS1Idx,
                                     LocalTensor<float> &tmpSortBuf);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<float> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> sortedBlockBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TQue<QuePosition::VECIN, 2> inQueue_;  // 输入双缓冲(depth2)：输入搬运与 DoScale 计算自动流水重叠
    TBuf<TPosition::VECCALC> overflowBuf_; // QS 容差窗候选独立 buffer
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;
    LocalTensor<float> overflowUb_; // QS overflow 独立存储
    // globalTopkUb_ per-token 布局: [scores(virTopK) | indices(virTopK)]
    // overflowUb_  per-token 布局: [scores(QS_OVF) | indices(QS_OVF)]，共 activeTokens_ 份
    //   QS 选出 top-virTopK 后，落在容差窗内、暂未截断的候选存这里；末轮 SortAll 再精确截断。
    //   按实际活跃 token 数 activeTokens_ 分配（每 AIV 仅处理 CeilDiv(s1BaseSize_,2) 个 token）。
    // QS_OVF — QS 收敛窗宽度：count(score≥pivot) 落入 [virTopK, virTopK+QS_OVF] 即判定收敛。
    //   窗越宽，QS 平均迭代轮数越少；OVF 段是宽容窗、不参与最终截断，不影响精度。
    //   约束：须为 64 的倍数（CompareScalar 对齐），上限由 QS 工作区容量决定（见下方 static_assert）。
    static constexpr int32_t QS_OVF = 384;
    static constexpr int32_t MAX_TOKENS_PER_AIV = 4; // per-token 标量状态数组上界（与 buffer 分配解耦）

    // ── QS 工作区容量校核（编译期 guard）──────────────────────────────────────
    // QS 在 tmpUb_ 内切三段 [srcScores(cap) | srcIndices(cap) | dstScores]。前两段满额各 cap；
    // 第三段仅作 firstQS 的 ReduceMin/ReduceMax work 区（二者复用同段），水位上限 2*BASE_TOPK，
    // 故真实写入水位 = 2*cap + 2*BASE_TOPK，tmpUb_ 据此精确分配。
    // 下面用最坏配置（blockLen=1: blockS2BaseSize=S2_BASE_SIZE、numCacheSlots=QS_NUM_CACHE_SLOTS）取最大 cap 做断言：
    // 调大 QS_OVF / numCacheSlots 一旦越界即编译失败，避免运行时静默 UB 越界。
    static constexpr int32_t QS_NUM_CACHE_SLOTS = 2; // 攒批槽数（cap ≤ 4096 约束下取上限）
    // 最坏新数据量：numCacheSlots 槽 + 当前块；blockLen=1 时 blockS2BaseSize=S2_BASE_SIZE（blockLen≥1 故 ≤
    // S2_BASE_SIZE）。 ⚠️ 此处 512 须与 kernel.h 的 S2_BASE_SIZE 保持一致（架构常量，极少变动）；若 S2_BASE_SIZE
    // 调大，需同步修改并重核 cap≤4096。
    static constexpr int32_t QS_NEWDATA_MAX = (QS_NUM_CACHE_SLOTS + 1) * 512;
    static constexpr int32_t QS_CAP_MAX = BASE_TOPK + QS_OVF + QS_NEWDATA_MAX; // 最坏 cap
    static constexpr int32_t QS_DST_USED = 2 * BASE_TOPK;                  // 第三段实际占用（Reduce work 区）
    static constexpr int32_t QS_TMPUB_NEED = 2 * QS_CAP_MAX + QS_DST_USED; // QS 真实写入水位
    static constexpr int32_t QS_TMPUB_FLOATS = QS_TMPUB_NEED;              // tmpBuf_ 按真实水位精确分配
    static_assert(QS_TMPUB_NEED <= QS_TMPUB_FLOATS, "QS_OVF/numCacheSlots 过大：QS 工作区超出 tmpUb_。"
                                                    "可收缩第三段 dstScores（仅需 2*BASE_TOPK）或减 numCacheSlots。");
    // 兜底约束：QS 未收敛时退化成 SortAll(4096) 全排截 topk，cap 必须 ≤ 4096 才装得下。
    //   QS_NUM_CACHE_SLOTS=2 + S2_BASE_SIZE=512 → cap=3968 < 4096。
    //   若调 numCacheSlots 或 QS_OVF 使 cap > 4096，编译失败，防止运行时兜底装不下。
    static_assert(QS_CAP_MAX <= 4096, "QS cap 超过 4096：未收敛兜底 SortAll(4096) 装不下。"
                                      "需减 numCacheSlots 或 QS_OVF 使 cap ≤ 4096。");
    // ──────────────────────────────────────────────────────────────────────────
    int32_t tokenStride_ = 0; // = virTopK * 2
    int32_t overflowCnt_[MAX_TOKENS_PER_AIV] = {};
    float lastThreshold_[MAX_TOKENS_PER_AIV] = {};
    int32_t cacheCnt_[MAX_TOKENS_PER_AIV] = {};
    // firstQS_：该 token 是否尚未做过 QS。首次需用 ReduceMin/滤哨兵 ReduceMax 现算二分上下界初值；
    //   之后下界沿用上轮收敛阈值 lastThreshold_（单调递增）、上界沿用 carry，省去每轮 Reduce 开销。
    //   初值置 1：首个基本块（info.loop==0）不经过下方 per-token 重置路径。
    int32_t firstQS_[MAX_TOKENS_PER_AIV] = {1, 1, 1, 1};
    // sugMaxCarry_：该 token 上轮 QS 实测的最紧合法上界 pivot，作下次 QS 的 curMax 起点（carry）。
    //   初值 +inf 表示"尚无 carry"，由首次 QS 内部自行求上界。新 S1 batch 时重置。
    float sugMaxCarry_[MAX_TOKENS_PER_AIV] = {3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f};

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t numCacheSlots_ = 0;
    int32_t activeTokens_ = 0; // = CeilDiv(s1BaseSize_,2)，每 AIV 实际活跃 token 数（overflow buffer 分配口径）
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    // max score
    const float MAX_LOGITS = *((float *)&MAX_LOGITS_TMP);
    const float SECOND_MAX_LOGITS = MAX_LOGITS / 2;
    const float CLIP_MAX_LOGITS = MAX_LOGITS / LOGITS_SCALE;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct LICommon::ConstInfo constInfo_;
    event_t eventIdVToMte2A;
    event_t eventIdVToMte2B;
    event_t eventIdMTE2ToV;
};

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;

    virTopK = constInfo_.sparseCountFlag ? constInfo_.sparseCount : BASE_TOPK;
    tokenStride_ = virTopK * 2;
    activeTokens_ = CeilDiv(s1BaseSize_, 2); // 每 AIV 实际活跃 token 数，overflow buffer 按此分配

    uint32_t blockS2BaseSize = s2BaseSize_ / constInfo_.blockLen;
    uint32_t sortOutBufSize = CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float);
    // 攒批 QS 策略：每 token 先把基本块结果暂存进缓存，存满 numCacheSlots_ 个槽
    //   （或遇 isS2End/isAllLoopEnd）才触发一次 QS，摊薄 QS 固定开销。
    //   numCacheSlots_=3 是 UB 预算下能容纳的槽数上限；槽越多，攒批收益越足。
    numCacheSlots_ = QS_NUM_CACHE_SLOTS;
    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);
    uint32_t tmpBufFloats =
        constInfo_.sparseCountFlag ? (virTopK + s2BaseSize_) * 2 : static_cast<uint32_t>(QS_TMPUB_NEED);
    pipe->InitBuffer(tmpBuf_, tmpBufFloats * sizeof(float));
    pipe->InitBuffer(inQueue_, 2, (groupInner_ * s2BaseSize_ + s2BaseSize_) * sizeof(float));
    pipe->InitBuffer(sortOutBuf_, sortOutBufSize);
    if (!constInfo_.sparseCountFlag) {
        pipe->InitBuffer(sortedBlockBuf_,
                         CeilDiv(s1BaseSize_, 2) * numCacheSlots_ * blockS2BaseSize * 2 * sizeof(float));
    }
    pipe->InitBuffer(overflowBuf_, activeTokens_ * QS_OVF * 2 * sizeof(float));
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));

    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    if (!constInfo_.sparseCountFlag) {
        SortedBasicBlock_ = sortedBlockBuf_.Get<float>();
    }
    overflowUb_ = overflowBuf_.Get<float>();

    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    if (constInfo_.sparseCountFlag) {
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
    } else {
        InitSortOutBufSeparated(globalTopkUb_, CeilDiv(s1BaseSize_, 2), tokenStride_, virTopK);
        Duplicate(overflowUb_.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, activeTokens_ * QS_OVF);
        Duplicate(overflowUb_[activeTokens_ * QS_OVF].template ReinterpretCast<int32_t>(), static_cast<int32_t>(-1),
                  activeTokens_ * QS_OVF);
    }
    PipeBarrier<PIPE_V>();

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
__aicore__ inline void LIVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitParams(const struct LICommon::ConstInfo &constInfo,
                                                 const LITilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;
    groupInner_ = 8;

    blockId_ = GetBlockIdx();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm, GlobalTensor<int64_t> vec1ParamGm,
    GlobalTensor<float> weightsGm, GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::AllocEventID()
{
    eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> indiceULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = indiceULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(indiceULocal);
    indiceULocal = outQueue_.DeQue<float>();
    LIServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(indiceULocal);

    if (constInfo_.returnValue) {
        K_T invalidValue = 0;
        LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
        LocalTensor<K_T> valULocal1 = valueULocal.template ReinterpretCast<K_T>();
        Duplicate(valULocal1, invalidValue, constInfo_.sparseCount);
        outQueue_.EnQue<float>(valueULocal);
        valueULocal = outQueue_.DeQue<float>();
        LIServiceVec::CopyOut(valueOutGm[invalidS1offset], valULocal1, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessVec(const LICommon::RunInfo &info)
{
    /// ---- 预处理 ---- ///
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 512*512 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_) * s2BaseSize_);
    // (B,S1,N1,1);(T,N1,1) -> (B,S1,N2,G,1) 当前只切分到S1轴
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        if (constInfo_.sparseCountFlag) {
            // 大 topk：交织格式重置
            InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
            PipeBarrier<PIPE_V>();
        } else {
            // QS：分离格式 + overflowUb_ 重置
            InitSortOutBufSeparated(globalTopkUb_, CeilDiv(s1BaseSize_, 2), tokenStride_, virTopK);
            Duplicate(overflowUb_.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, activeTokens_ * QS_OVF);
            Duplicate(overflowUb_[activeTokens_ * QS_OVF].template ReinterpretCast<int32_t>(), static_cast<int32_t>(-1),
                      activeTokens_ * QS_OVF);
            PipeBarrier<PIPE_V>();
            // 重置 per-token QS 状态（新 S1 batch 不能复用旧 threshold/overflow）
            for (int32_t t = 0; t < MAX_TOKENS_PER_AIV; t++) {
                lastThreshold_[t] = 0.0f;
                overflowCnt_[t] = 0;
                cacheCnt_[t] = 0;
                firstQS_[t] = 1;
                sugMaxCarry_[t] = 3.4e38f;
            }
        }
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;

    /// ---- main loop ---- ///
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeq += 1;
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();

            /// ---- 处理gemm结果 ---- ///
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
                // 从 inQueue_ 取双缓冲 tensor：TQue 自动处理 MTE2/V 同步，输入搬运与排序物理分离
                LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
                LocalTensor<float> weightsInUb = mmInUb[procGnum * s2BaseSize_];
                LIServiceVec::CopyIn(mmInUb, weightsInUb, mm1ResGm, weightsGm,
                                     mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                         outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign,
                                     weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_, procGnum,
                                     info.actualSingleProcessSInnerSizeAlign, mmUbStride);
                inQueue_.EnQue<float>(mmInUb);
                mmInUb = inQueue_.DeQue<float>();
                weightsInUb = mmInUb[procGnum * s2BaseSize_];
                LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, brcBuf, procGnum,
                                      s2BaseSize_, outerGidx);
                inQueue_.FreeTensor(mmInUb);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            /// kv block indexer处理 ///
            int32_t blockS2LenVecAlign = cuS2LenVecAlign / constInfo_.blockLen;
            int32_t blockS2Len = (cuS2Len + constInfo_.blockLen - 1) / constInfo_.blockLen;
            int32_t blockS2BaseSize = s2BaseSize_ / constInfo_.blockLen;
            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[blockS2LenVecAlign];
            if (constInfo_.blockLen == 2 || constInfo_.blockLen == 4) {
                // 512 -> 512 * 8
                AscendC::Brcb(tmpSortBuf, reduceOutInner, cuS2LenVecAlign / 8, {1, 8});
                PipeBarrier<PIPE_V>();
                // block reduce: blockLen->1, here blockLen*8->1
                // 用高阶api Sum对s2维度reduce(不支持源操作数与目的操作数地址重叠)
                AscendC::SumParams params;
                params.outter = cuS2LenVecAlign * 8 / (constInfo_.blockLen * 8);
                params.n = (constInfo_.blockLen * 8);
                params.inner = (constInfo_.blockLen * 8);
                // 得到 cuS2LenVecAlign / BLOCK_LEN个结果
                AscendC::Sum(reduceOutInner, tmpSortBuf, tmpUb_.template ReinterpretCast<uint8_t>(), params);
                PipeBarrier<PIPE_V>();
                AscendC::Muls(reduceOutInner, reduceOutInner, static_cast<float>(1.0f / (8 * constInfo_.blockLen)),
                              blockS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            } else if (constInfo_.blockLen > 4) { // blockLen is 8 or 16
                // 用高阶api Sum对s2维度reduce(不支持源操作数与目的操作数地址重叠)
                AscendC::SumParams params;
                params.outter = blockS2LenVecAlign;
                params.n = constInfo_.blockLen;
                params.inner = constInfo_.blockLen;
                // 得到 cuS2LenVecAlign / BLOCK_LEN个结果
                AscendC::Sum(tmpSortBuf, reduceOutInner, tmpUb_.template ReinterpretCast<uint8_t>(), params);
                PipeBarrier<PIPE_V>();
                AscendC::Muls(reduceOutInner, tmpSortBuf, static_cast<float>(1.0f / constInfo_.blockLen),
                              blockS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            // 对logits做clip，避免冲撞SECOND_MAX_LOGITS
            AscendC::Mins(reduceOutInner, reduceOutInner, CLIP_MAX_LOGITS, blockS2LenVecAlign);
            PipeBarrier<PIPE_V>();

            /// ---- sink & swa处理 ---- ///
            // 1. 指定head区域全选，赋予第二高分数
            if (info.s2Idx == 0 && constInfo_.initNum > 0) {
                uint32_t blockInitNum = constInfo_.initNum / constInfo_.blockLen;
                Duplicate(reduceOutInner, SECOND_MAX_LOGITS, blockInitNum);
                PipeBarrier<PIPE_V>();
            }
            // 2. 指定tail区域全选，赋予第二高分数
            // 当前处理块踩到了local 部分
            int32_t localStart = cuRealAcSeq - constInfo_.localNum;
            bool hasLocal = cuBaseS2Idx > localStart || cuBaseS2Idx + cuS2Len > localStart;
            if (constInfo_.localNum > 0 && hasLocal) {
                // 部分落在local 区间
                if (cuBaseS2Idx < localStart) {
                    int32_t nonLocalBlockNum =
                        (localStart - cuBaseS2Idx + constInfo_.blockLen - 1) / constInfo_.blockLen;
                    int32_t localBlockNum = blockS2Len - nonLocalBlockNum;
                    if (nonLocalBlockNum % 8 == 0) {
                        Duplicate(reduceOutInner[nonLocalBlockNum], SECOND_MAX_LOGITS, localBlockNum);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Duplicate(tmpSortBuf, SECOND_MAX_LOGITS, blockS2Len);
                        PipeBarrier<PIPE_V>();
                        Adds(tmpSortBuf, reduceOutInner, 0.0f, nonLocalBlockNum);
                        PipeBarrier<PIPE_V>();
                        Adds(reduceOutInner, tmpSortBuf, 0.0f, blockS2Len);
                        PipeBarrier<PIPE_V>();
                    }

                } else { // 当前block 全落在local 区间
                    Duplicate(reduceOutInner, SECOND_MAX_LOGITS, blockS2Len);
                    PipeBarrier<PIPE_V>();
                }
            }
            // 3. 最后一个block必选，赋予最高分数
            if (isS2End) {
                reduceOutInner.SetValue(blockS2Len - 1, MAX_LOGITS);
                SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            }

            /// ---- 整理score & indice ---- ///
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, blockS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, blockS2Len);
            PipeBarrier<PIPE_V>();

            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            if (blockS2LenVecAlign != blockS2Len) {
                Duplicate(sortIndiceUbInt, -1, blockS2LenVecAlign); // 无效数据索引填充为-1
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx / constInfo_.blockLen),
                 blockS2Len);
            PipeBarrier<PIPE_V>();

            /// ---- QuickSelect (攒批触发) / 大 topk MergeSort ---- ///
            int32_t tokenBase = innerS1Idx * tokenStride_;
            int32_t ovfScoreOff = innerS1Idx * QS_OVF;
            int32_t ovfIdxOff = activeTokens_ * QS_OVF + ovfScoreOff;
            LocalTensor<float> globScores = globalTopkUb_[tokenBase];
            LocalTensor<int32_t> globIndices = globalTopkUb_[tokenBase + virTopK].template ReinterpretCast<int32_t>();
            LocalTensor<float> ovfScores = overflowUb_[ovfScoreOff];
            LocalTensor<int32_t> ovfIndices = overflowUb_[ovfIdxOff].template ReinterpretCast<int32_t>();

            // SortedBasicBlock_ 缓存布局（分离格式，per token），提到 if/else 前以便末块 SortAll 兜底拼接
            int32_t cacheStride = numCacheSlots_ * blockS2BaseSize * 2;
            int32_t cacheScoreBase = innerS1Idx * cacheStride;
            int32_t cacheIdxBase = cacheScoreBase + numCacheSlots_ * blockS2BaseSize;

            if (constInfo_.sparseCountFlag) {
                // 大 topk 路径（sparseCount > 2048）：每轮全量 SortAll + MergeSort（交织格式）
                LIServiceVec::SortAll(reduceOutBuff, tmpSortBuf, blockS2LenVecAlign);
                PipeBarrier<PIPE_V>();
                LIServiceVec::MergeSort(globalTopkUb_[tokenBase], virTopK, reduceOutBuff, blockS2LenVecAlign, tmpUb_);
            } else {
                int32_t relativeS2 = info.s2Idx - static_cast<int32_t>(blockS2StartIdx_);
                int32_t appendRounds = virTopK / blockS2LenVecAlign;
                // cacheScoreBase/cacheIdxBase 已在 if/else 前声明，末块 SortAll 兜底复用

                if (relativeS2 < appendRounds) {
                    // 起始阶段：候选还没填满 virTopK，本块直接追加到 globalTopkUb_，无需 QS
                    int32_t appendOff = relativeS2 * blockS2LenVecAlign;
                    PipeBarrier<PIPE_V>();
                    Adds(globScores[appendOff], sortScoreUb, 0.0f, blockS2LenVecAlign);
                    Adds(globIndices[appendOff], sortIndiceUbInt, static_cast<int32_t>(0), blockS2LenVecAlign);
                    PipeBarrier<PIPE_V>();
                } else {
                    // 候选已满：本块进缓存攒批，存满 numCacheSlots_ 槽再触发一次 QS
                    if (cacheCnt_[innerS1Idx] < numCacheSlots_) {
                        // 缓存未满：暂存本块
                        int32_t slot = cacheCnt_[innerS1Idx];
                        PipeBarrier<PIPE_V>();
                        Adds(SortedBasicBlock_[cacheScoreBase + slot * blockS2BaseSize], sortScoreUb, 0.0f,
                             blockS2LenVecAlign);
                        Adds(SortedBasicBlock_[cacheIdxBase + slot * blockS2BaseSize]
                                 .template ReinterpretCast<int32_t>(),
                             sortIndiceUbInt, static_cast<int32_t>(0), blockS2LenVecAlign);
                        PipeBarrier<PIPE_V>();
                        cacheCnt_[innerS1Idx]++;
                    } else {
                        // 缓存已满：连同本块（作 extra）一起拼入 QS，重选 top-virTopK
                        ExecuteQS(globScores, globIndices, ovfScores, ovfIndices, cacheScoreBase, cacheIdxBase,
                                  cacheCnt_[innerS1Idx] * blockS2BaseSize, sortScoreUb, sortIndiceUbInt,
                                  blockS2LenVecAlign, 128.0f, innerS1Idx, tmpSortBuf);
                        cacheCnt_[innerS1Idx] = 0;
                    }

                    // 末块：不再走 QS（避免未收敛 tie 死锁时丢弃哨兵块）。
                    //   末块 cache（含本块 553 暂存的）原样保留，由下方末块 SortAll 兜底全排截 topk。
                    //   哨兵块（对角/local 末尾）在 cache 里，进 SortAll 即不丢。
                    //   （末块 cap = glob+ovf+cache ≤ 2944 < SORT_LEN=4096，全排不爆）
                }
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);
            /// ---- 末块精排 & 搬出 ---- ///
            // QS 只保证候选集正确（top-virTopK + 容差窗），不保证内部有序；
            // 末块用 SortAll 对 候选(virTopK) + 容差窗(QS_OVF) 整体排序，精确截出有序 top-virTopK。
            // 大 topk 路径的 globalTopkUb_ 已由每轮 MergeSort 维护成有序交织格式，无需此步。
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;
            if ((needCopyOutGm || needCopyWsGm) && !constInfo_.sparseCountFlag) {
                constexpr int32_t SORT_LEN = 4096; // SortAll 定长工作区（scores | indices 各一段）
                // 工作区填充哨兵：scores=-inf、indices=-1，余量不参与排序
                Duplicate(tmpUb_.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, SORT_LEN);
                Duplicate(tmpUb_[SORT_LEN].template ReinterpretCast<int32_t>(), static_cast<int32_t>(-1), SORT_LEN);
                PipeBarrier<PIPE_V>();
                // 搬入待排数据：候选段 globScores(virTopK) + 容差窗(QS_OVF) + 末块保留的 cache，indices 同步
                Adds(tmpUb_, globScores, 0.0f, virTopK);
                Adds(tmpUb_[virTopK], ovfScores, 0.0f, QS_OVF);
                Adds(tmpUb_[SORT_LEN].template ReinterpretCast<int32_t>(), globIndices, static_cast<int32_t>(0),
                     virTopK);
                Adds(tmpUb_[SORT_LEN + virTopK].template ReinterpretCast<int32_t>(), ovfIndices,
                     static_cast<int32_t>(0), QS_OVF);
                // 末块 cache 兜底：末块不再走 QS，cache（含哨兵块）原样并入 SortAll 全排，绝不丢块。
                //   末块 cacheCnt < numCacheSlots（满槽已在 565 触发 QS），cache ≤ (numSlots-1)*blockS2BaseSize，
                //   glob+ovf+cache ≤ 2432 + 2*512 = 3456 < SORT_LEN=4096，安全。
                int32_t cacheLen = cacheCnt_[innerS1Idx] * blockS2BaseSize;
                if (cacheLen > 0) {
                    Adds(tmpUb_[virTopK + QS_OVF], SortedBasicBlock_[cacheScoreBase], 0.0f, cacheLen);
                    Adds(tmpUb_[SORT_LEN + virTopK + QS_OVF].template ReinterpretCast<int32_t>(),
                         SortedBasicBlock_[cacheIdxBase].template ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                         cacheLen);
                    PipeBarrier<PIPE_V>();
                    cacheCnt_[innerS1Idx] = 0; // 末块 cache 已并入 SortAll，清空
                }
                PipeBarrier<PIPE_V>();
                LocalTensor<float> tmpSortBuf2 = outQueue_.AllocTensor<float>();
                LIServiceVec::SortAll(tmpUb_, tmpSortBuf2, SORT_LEN);
                PipeBarrier<PIPE_V>();
                outQueue_.FreeTensor(tmpSortBuf2);
                // 排序结果（score/index 交织）在 tmpUb_[0:virTopK*2]，搬回 globalTopkUb_
                Adds(globalTopkUb_[tokenBase].template ReinterpretCast<int32_t>(),
                     tmpUb_.template ReinterpretCast<int32_t>(), static_cast<int32_t>(0), virTopK * 2);
                PipeBarrier<PIPE_V>();
            }
            if (needCopyOutGm) {
                uint32_t validS2Block = (cuRealAcSeq + constInfo_.blockLen - 1) / constInfo_.blockLen;
                validS2Block = validS2Block < constInfo_.sparseCount ? validS2Block : constInfo_.sparseCount;
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                float lhsValue = globalTopkUb_[tokenBase].GetValue(0);
                int32_t lhsIdx = (globalTopkUb_[tokenBase].template ReinterpretCast<int32_t>()).GetValue(1);
                float rhsValue = globalTopkUb_[tokenBase].GetValue(2 * validS2Block - 2);
                int32_t rhsIdx =
                    (globalTopkUb_[tokenBase].template ReinterpretCast<int32_t>()).GetValue(2 * validS2Block - 1);
                globalTopkUb_[tokenBase].SetValue(0, rhsValue);
                (globalTopkUb_[tokenBase].template ReinterpretCast<int32_t>()).SetValue(1, rhsIdx);
                globalTopkUb_[tokenBase].SetValue(2 * validS2Block - 2, lhsValue);
                (globalTopkUb_[tokenBase].template ReinterpretCast<int32_t>()).SetValue(2 * validS2Block - 1, lhsIdx);
                SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

                // 结果提取 & 搬出
                int64_t offset = (constInfo_.sparseCount <= 4096) ? virTopK : constInfo_.sparseCount / 2;
                int64_t copyLen =
                    (constInfo_.sparseCount <= 4096) ? constInfo_.sparseCount : constInfo_.sparseCount / 2;
                int64_t copyNum = (constInfo_.sparseCount <= 4096) ? 1 : 2;
                for (int64_t i = 0; i < copyNum; i++) {
                    LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                    LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                    Extract(outValueUb, outIdxUb, globalTopkUb_[tokenBase + 2 * i * offset], (offset / 32));

                    LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
                    if (constInfo_.returnValue) {
                        PipeBarrier<PIPE_V>();
                        Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, copyLen);
                        PipeBarrier<PIPE_V>();
                    }

                    LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                    outQueue_.EnQue<float>(outValueUb);
                    outValueUb = outQueue_.DeQue<float>();

                    LIServiceVec::CopyOut(
                        indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset], idxULocal1,
                        copyLen);
                    if (constInfo_.returnValue) {
                        LIServiceVec::CopyOut(
                            valueOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount + i * offset],
                            valueULocal1, copyLen);
                    }
                    outQueue_.FreeTensor(outValueUb);
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
                // tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                // tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                // tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                // tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
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
                LIServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[tokenBase], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }

    // BNSD场景无效S1 输出-1
    if (LAYOUT_T == LI_LAYOUT::BSND) {
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

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
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
__aicore__ inline void LIVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    // int64_t s2Start;
    // int64_t s2End;
    int64_t isS2End;
    // int64_t bn2Idx;
    // int64_t s1Idx;
    uint32_t acc_list_num = 0;
    // int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    // int64_t nextneedFd;
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
        // s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
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

        // 对角线block当前在队首位置，我们把它放回到对角线上
        s2ActSeq = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_ + 1);
        uint32_t validS2Block = (s2ActSeq + constInfo_.blockLen - 1) / constInfo_.blockLen;
        validS2Block = validS2Block < constInfo_.sparseCount ? validS2Block : constInfo_.sparseCount;
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        float lhsValue = curValueIdxUb.GetValue(0);
        int32_t lhsIdx = (curValueIdxUb.template ReinterpretCast<int32_t>()).GetValue(1);
        float rhsValue = curValueIdxUb.GetValue(2 * validS2Block - 2);
        int32_t rhsIdx = (curValueIdxUb.template ReinterpretCast<int32_t>()).GetValue(2 * validS2Block - 1);
        curValueIdxUb.SetValue(0, rhsValue);
        (curValueIdxUb.template ReinterpretCast<int32_t>()).SetValue(1, rhsIdx);
        curValueIdxUb.SetValue(2 * validS2Block - 2, lhsValue);
        (curValueIdxUb.template ReinterpretCast<int32_t>()).SetValue(2 * validS2Block - 1, lhsIdx);
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

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
            LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
            Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, constInfo_.sparseCount);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
            DataCopyPad(valueOutGm[outOffset], valueULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(K_T)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}

// ExecuteQS：把当前候选(globScores/ovfScores) + 缓存块 + extra 块拼成 QS 输入，
//   调 QuickSelectPartition 重选 top-virTopK；收敛则把新候选写回 glob/ovf，未收敛则保持候选不变。
//   per-token 维护 lastThreshold_（下界）、sugMaxCarry_（上界）跨轮加速二分收敛。
template <typename LIT>
__aicore__ inline void LIVector<LIT>::ExecuteQS(LocalTensor<float> &globScores, LocalTensor<int32_t> &globIndices,
                                                LocalTensor<float> &ovfScores, LocalTensor<int32_t> &ovfIndices,
                                                int32_t cacheScoreBase, int32_t cacheIdxBase, int32_t cachedLen,
                                                LocalTensor<float> &extraScores, LocalTensor<int32_t> &extraIndices,
                                                int32_t extraLen, float sugMax, int32_t innerS1Idx,
                                                LocalTensor<float> &tmpSortBuf)
{
    int32_t mergedLen = cachedLen + extraLen;
    int32_t maxNewData = numCacheSlots_ * (s2BaseSize_ / constInfo_.blockLen);
    int32_t carrySafeCnt = virTopK + QS_OVF - maxNewData;
    float effSugMax = (sugMaxCarry_[innerS1Idx] < 3.0e38f) ? sugMaxCarry_[innerS1Idx] : sugMax;
    LIServiceVec::QSParams qsParams = {
        virTopK, QS_OVF, mergedLen, 32, lastThreshold_[innerS1Idx], effSugMax, firstQS_[innerS1Idx], carrySafeCnt};
    int32_t slotSize = qsParams.newDataOffset();
    LocalTensor<float> srcScores = tmpUb_;
    LocalTensor<int32_t> srcIndices = tmpUb_[qsParams.srcIndicesOff()].template ReinterpretCast<int32_t>();

    PipeBarrier<PIPE_V>();
    Adds(srcScores, globScores, 0.0f, virTopK);
    Adds(srcIndices, globIndices, static_cast<int32_t>(0), virTopK);
    Adds(srcScores[virTopK], ovfScores, 0.0f, QS_OVF);
    Adds(srcIndices[virTopK], ovfIndices, static_cast<int32_t>(0), QS_OVF);
    Adds(srcScores[slotSize], SortedBasicBlock_[cacheScoreBase], 0.0f, cachedLen);
    Adds(srcIndices[slotSize], SortedBasicBlock_[cacheIdxBase].template ReinterpretCast<int32_t>(),
         static_cast<int32_t>(0), cachedLen);
    if (extraLen > 0) {
        Adds(srcScores[slotSize + cachedLen], extraScores, 0.0f, extraLen);
        Adds(srcIndices[slotSize + cachedLen], extraIndices, static_cast<int32_t>(0), extraLen);
    }
    PipeBarrier<PIPE_V>();

    float newThreshold = lastThreshold_[innerS1Idx];
    float carryMax = 3.4e38f;
    int32_t totalAbove = LIServiceVec::QuickSelectPartition(tmpUb_, tmpSortBuf, qsParams, newThreshold, carryMax);
    sugMaxCarry_[innerS1Idx] = carryMax; // 携带给该 token 下一次 QS
    // totalAbove < virTopK 表示未收敛：放弃本轮写回，候选集保持不变，留待末块 SortAll 兜底
    if (totalAbove >= virTopK) {
        lastThreshold_[innerS1Idx] = newThreshold; // 收敛阈值作为下轮二分下界（单调递增）
        firstQS_[innerS1Idx] = 0;
        overflowCnt_[innerS1Idx] = totalAbove - virTopK;
        LocalTensor<float> dstScores = tmpUb_[qsParams.dstScoresOff()];
        LocalTensor<int32_t> dstIndices = tmpSortBuf.template ReinterpretCast<int32_t>();
        // 写回新候选：前 virTopK 个进 glob，余下容差窗进 ovf
        PipeBarrier<PIPE_V>();
        Adds(globScores, dstScores, 0.0f, virTopK);
        Adds(globIndices, dstIndices, static_cast<int32_t>(0), virTopK);
        Adds(ovfScores, dstScores[virTopK], 0.0f, QS_OVF);
        Adds(ovfIndices, dstIndices[virTopK], static_cast<int32_t>(0), QS_OVF);
        PipeBarrier<PIPE_V>();
    } else {
        // ── 未收敛兜底：退化成 SortAll(4096) 全排截 topk，绝不丢块 ──
        //   cap ≤ QS_CAP_MAX=3968 < 4096，全排不爆。哨兵块（高分）必入选，real 边界元素不漏。
        //   结果写回 glob/ovf（分离格式），后续 QS/末块 SortAll 可正常读。
        //   tmpSortBuf(8192 floats) 先作 SortAll 的 tmp，SortAll 后结果落回 tmpUb_，tmpSortBuf 复用作 Extract 输出。
        constexpr int32_t FB_SORT_LEN = 4096;
        // 重新拼到 SortAll 重载1 要求的 [values(4096) | indices(4096)] 布局
        Duplicate(tmpUb_.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, FB_SORT_LEN);
        Duplicate(tmpUb_[FB_SORT_LEN].template ReinterpretCast<int32_t>(), static_cast<int32_t>(-1), FB_SORT_LEN);
        PipeBarrier<PIPE_V>();
        Adds(tmpUb_, globScores, 0.0f, virTopK);
        Adds(tmpUb_[virTopK], ovfScores, 0.0f, QS_OVF);
        Adds(tmpUb_[virTopK + QS_OVF], SortedBasicBlock_[cacheScoreBase], 0.0f, cachedLen);
        Adds(tmpUb_[FB_SORT_LEN].template ReinterpretCast<int32_t>(), globIndices, static_cast<int32_t>(0), virTopK);
        Adds(tmpUb_[FB_SORT_LEN + virTopK].template ReinterpretCast<int32_t>(), ovfIndices, static_cast<int32_t>(0),
             QS_OVF);
        Adds(tmpUb_[FB_SORT_LEN + virTopK + QS_OVF].template ReinterpretCast<int32_t>(),
             SortedBasicBlock_[cacheIdxBase].template ReinterpretCast<int32_t>(), static_cast<int32_t>(0), cachedLen);
        if (extraLen > 0) {
            Adds(tmpUb_[virTopK + QS_OVF + cachedLen], extraScores, 0.0f, extraLen);
            Adds(tmpUb_[FB_SORT_LEN + virTopK + QS_OVF + cachedLen].template ReinterpretCast<int32_t>(), extraIndices,
                 static_cast<int32_t>(0), extraLen);
        }
        PipeBarrier<PIPE_V>();
        LIServiceVec::SortAll(tmpUb_, tmpSortBuf, FB_SORT_LEN); // 结果落回 tmpUb_[0:8192] 交织
        PipeBarrier<PIPE_V>();
        // de-interleave：tmpSortBuf 复用作 Extract 输出（scores+indices 各 extLen）
        int32_t extLen = virTopK + QS_OVF; // 2432，32 的倍数
        LocalTensor<float> outValueUb = tmpSortBuf;
        LocalTensor<uint32_t> outIdxUb = tmpSortBuf[extLen].template ReinterpretCast<uint32_t>();
        Extract(outValueUb, outIdxUb, tmpUb_, (extLen / 32));
        PipeBarrier<PIPE_V>();
        Adds(globScores, outValueUb, 0.0f, virTopK);
        Adds(globIndices, outIdxUb.template ReinterpretCast<int32_t>(), static_cast<int32_t>(0), virTopK);
        Adds(ovfScores, outValueUb[virTopK], 0.0f, QS_OVF);
        Adds(ovfIndices, outIdxUb[virTopK].template ReinterpretCast<int32_t>(), static_cast<int32_t>(0), QS_OVF);
        PipeBarrier<PIPE_V>();
        // 标记收敛状态：glob 已更新，后续 QS/末块正常处理
        lastThreshold_[innerS1Idx] = newThreshold;
        firstQS_[innerS1Idx] = 0;
        overflowCnt_[innerS1Idx] = QS_OVF;
    }
}

} // namespace LIKernel
#endif
