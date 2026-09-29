/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_service_vector_arch22.h
 * \brief cloned from lightning_indexer_v2 @Phase1, consumer-mode specialization.
 * Vector核服务类：加权ReLU求和(DoScale/DoReduce)、S4a 候选掩码构建+leak pen 链(BuildCandidateMask)、
 * TopK排序(SortAll/MergeSort)、LD归约(ProcessLD，noS2Split 下休眠)、输出写回。
 * 相对克隆基线的差异（设计 §4.3）：
 *  - InitBuffers：candBuf_(32KB 按行分区)/candConstBuf_(negHuge|isOut|pen 6KB) 恒分配，无 source 块级累加器；
 *    SortedBasicBlock_ 视图不绑定（s1Base=4 下越界，D6 恒走 SortAll+MergeSort）
 *  - ProcessVec：DoReduce 后、SortAll 前插入 S4a（BuildCandidateMask + leak pen 链，R11）
 *  - 排序路径 D6 常量化（恒 SortAll+MergeSort，短行缓存粗排分支删除）
 *  - 无 S5a/S6a/CopyOutCandTopkIndex/块级累加器及其重置点
 */

#ifndef SPARSE_LIGHTNING_INDEXER_SERVICE_VECTOR_ARCH22_H
#define SPARSE_LIGHTNING_INDEXER_SERVICE_VECTOR_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_lightning_indexer_common_arch22.h"
#include "sparse_lightning_indexer_vector.h"

namespace SparseLIKernel {
using namespace SparseLICommon;
using namespace LIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;
// 候选行 GM→UB 一次性装载的 MTE2→V fence 事件（MTE2_V 方向 id 2 空闲：ping/pong 0/1 为
// CopyIn 专用，V_MTE2 方向 id 2 在 over2K 恒 false 下无使用方；SetWaitFlag 自平衡，无悬挂 flag）
constexpr uint32_t EVENTID_MTE2_TO_V_CAND = 2;

template <typename LIT>
class SparseLightningIndexerServiceVector {
public:
    // ================================ 类型定义区 ================================
    // 中间计算数据类型为float，高精度模式
    using K_T = typename LIT::keyType;
    using OUT_V_T = float;
    using W_T = float;
    static constexpr SLI_LAYOUT LAYOUT_T = LIT::layout;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline SparseLightningIndexerServiceVector() {}
    __aicore__ inline void ProcessVec(const SparseLICommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct SparseLICommon::ConstInfo &constInfo,
                                      const SparseLITilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<OUT_V_T> valueOutGm);
    // candidate (two-level topk)：consumer 候选输入 GM 绑定（REQUIRED 输入；本算子无候选输出）
    __aicore__ inline void InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);
    // candidate (two-level topk) consumer 链 S4a：候选掩码构建（产出 candIsOutUb_，leak pen 链在 ProcessVec）
    __aicore__ inline void BuildCandidateMask(const SparseLICommon::RunInfo &info, int32_t cuS1Idx, int32_t cuBaseS2Idx,
                                              int32_t innerS1Idx);
    // S4a：块级 isOut 表构建（每行一次，覆盖 [0, blkCnt)）
    __aicore__ inline void BuildCandBlockTable(const SparseLICommon::RunInfo &info, int32_t cuS1Idx, int32_t blkCnt,
                                               int32_t innerS1Idx);
    // S4a：块级 isOut 表 → 位置级 candIsOutUb_ 展开（Brcb 向量展开 / 标量展开）
    __aicore__ inline void ExpandCandMaskToPosition(int32_t tileBlockBase, int32_t tileBlkNum, int32_t blockSize,
                                                    int32_t innerS1Idx);
    // S4a 回退：blkCnt 超块级表容量时走 2026-09-21 已验收的纯标量逐 tile 实现
    __aicore__ inline void BuildCandidateMaskLegacyTile(const SparseLICommon::RunInfo &info, int32_t cuS1Idx,
                                                        int32_t cuBaseS2Idx, int32_t innerS1Idx);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<OUT_V_T> valueOutGm;
    GlobalTensor<int32_t> candidateTopkIndexInGm; // consumer 候选输入（REQUIRED）
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

    // candidate (two-level topk) consumer：S4a 专用 UB（32KB，布局见下方常量区；按行分区 R6）
    TBuf<TPosition::VECCALC> candBuf_;
    // candidate 常量/掩码缓冲：[0,512) NEG_HUGE 常量 | [512,1024) isOut 掩码 | [1024,1536) pen
    TBuf<TPosition::VECCALC> candConstBuf_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> candBufUb_;
    LocalTensor<float> candNegHugeUb_; // [0,512)：Duplicate NEG_HUGE 常量（InitBuffers 一次性填充）
    LocalTensor<float> candIsOutUb_;   // [512,1024)：当前 tile 位置级 0/1 掩码（BuildCandidateMask 产出）
    LocalTensor<float> candPenUb_;     // [1024,1536)：pen 链临时（(NEG_HUGE−score)×isOut）

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
    // ---------------------------------------------------------------------------
    // candBuf_ 布局（S4a 专用，32KB = 8192 floats，与优化前分配大小完全一致，UB 预算零增长）
    //   [0, CAND_TBL_CAP)                       行 0（innerS1Idx=0）块级 isOut 表
    //   [CAND_TBL_CAP, 2*CAND_TBL_CAP)          行 1（innerS1Idx=1）块级 isOut 表（R6 行分区）
    //   [CAND_STAGE_OFFSET, +CAND_BRCB_FANOUT²) blockSize>8 时的 Brcb 预展开暂存（64 floats）
    //   [CAND_LOAD_OFFSET, +CAND_LOAD_LEN)      候选行 GM→UB int32 装载区（两行串行复用）
    //   [CAND_LOAD_OFFSET+CAND_LOAD_LEN, 8192)  DataCopyPad tailPadding 守护区（576 floats）
    // ---------------------------------------------------------------------------
    // 每行块级表容量（floats，64 对齐）：blkCnt ≤ CAP 走快路径；blkCnt > CAP 回退标量逐 tile 实现
    constexpr static uint32_t CAND_TBL_CAP = 2752;
    // Brcb：每 repeat 取 8 个 b32 元素、各自广播成 1 个 32B 块（float 即 8 份）→ 每 repeat 出 64 位置
    constexpr static uint32_t CAND_BRCB_FANOUT = 8;
    constexpr static uint32_t CAND_STAGE_OFFSET = 2 * CAND_TBL_CAP;                  // 5504
    constexpr static uint32_t CAND_STAGE_LEN = CAND_BRCB_FANOUT * CAND_BRCB_FANOUT;  // 64
    constexpr static uint32_t CAND_LOAD_OFFSET = CAND_STAGE_OFFSET + CAND_STAGE_LEN; // 5568
    constexpr static uint32_t CAND_LOAD_LEN = BASE_TOPK;                             // 2048 int32 = candBlocks 上限
    constexpr static uint32_t CAND_BUF_TOTAL = 8192;                                 // floats（32KB）
    // 候选行标量扫描的展开度（先集中发 8 条 UB 读再统一置位，令读延迟重叠、消除逐元素分支串行化）
    constexpr static int32_t CAND_SCAN_UNROLL = 8;
    // candConstBuf_ 内三个 512-float 区的固定偏移
    constexpr static uint32_t CAND_NEG_HUGE_OFFSET = 0;
    constexpr static uint32_t CAND_IS_OUT_OFFSET = 512;
    constexpr static uint32_t CAND_PEN_OFFSET = 1024;

    struct SparseLICommon::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.sparseCount : BASE_TOPK;
    pipe->InitBuffer(outQueue_, 1, outNeedBufSize); // 33KB: extract + reduceCache
    // 68KB: cube结果和weight搬运db(2x34KB), mrgsort临时UB
    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float)); // 32KB (s1Base=4)
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                           // 2KB
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 3 * sizeof(float));                     // 6KB
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));

    //
    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    // D6: consumer 恒走 SortAll+MergeSort——s1Base=4 下 sortOutBuf 仅 8192 floats，
    // SortedBasicBlock_（globalTopkUb_[8192] 起）偏移越界，该视图不绑定、缓存粗排路径不执行
    globalTopkNum_ = 0;

    // candidate (two-level topk) consumer：S4a 专用 UB（恒分配无 mode 分支，布局见常量区注释）
    // UB 预算：off 基础（s1Base=4 化后）≈141.6KB + candBuf 32KB + candConst 6KB = ≈179.6KB ≤ 192KB
    pipe->InitBuffer(candBuf_, CAND_BUF_TOTAL * sizeof(float)); // 32KB（与优化前同尺寸，UB 零增长）
    candBufUb_ = candBuf_.Get<float>();
    pipe->InitBuffer(candConstBuf_, 3 * CAND_IS_OUT_OFFSET * sizeof(float)); // 6KB: negHuge|isOut|pen
    LocalTensor<float> candConstUb = candConstBuf_.Get<float>();
    candNegHugeUb_ = candConstUb[CAND_NEG_HUGE_OFFSET];
    candIsOutUb_ = candConstUb[CAND_IS_OUT_OFFSET];
    candPenUb_ = candConstUb[CAND_PEN_OFFSET];
    // candNegHugeUb_ 不再是常量：pen 链每 tile 重建为 "NEG_HUGE − 全局位置×2^76"（降级值内嵌
    // 索引序，见 ProcessVec），此处不做初始化

    // 基本块执行前初始化UB和GM
    // ArithProgression 豁免边界（R1 检视答复）：count 恒 s2BaseSize_=512（>64，不进 (8,64] 标量分支），
    // 且位于 InitBuffers（流水开始前，无在飞 V 指令、无后续掩码敏感消费者先于下次 InitBuffers）——
    // 流水中段禁用 ArithProgression（见 ProcessVec pen 链 R1 修复注释）
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
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::InitParams(
    const struct SparseLICommon::ConstInfo &constInfo, const SparseLITilingData *__restrict tilingData)
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
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::InitVec1GlobalTensor(
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
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::InitVecCandidateTensor(
    GlobalTensor<int32_t> candidateTopkIndexInGm)
{
    this->candidateTopkIndexInGm = candidateTopkIndexInGm;
}

template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
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
    // consumer 无 candidate 输出，无同步清理项
}

// ===========================================================================================
// candidate (two-level topk) consumer 链 S4a：候选掩码构建
// ===========================================================================================
// 输入：candidateTopkIndexInGm 行（info.candidateOutOffset + cuS1Idx × candBlocks 起 candBlocks 个 int32）。
// 输出：candIsOutUb_[0, s2BaseSize_) 位置级 0/1 掩码（1 = 该位置所在块不在候选集内，即"可达但非候选"）。
//
// 【2026-09-22 性能重写（P1）】原实现（2026-09-21 缺陷修复轮）每 tile 每行做
//   512 次 SetValue 初始化 + candBlocks 次标量 GetValue 全量扫描 + hits×blockSize 次 SetValue 置位
//   + 一对 S_V fence；候选集合对整行所有 tile 不变，扫描结果本质重复。实测（mask_sens.py）该路径占
//   consumer 运行时 ~95%：模型主场景 S1/S2c=4096、Kc=2048、bs=8 下 30.1ms vs off 2.0ms（×15.0），
//   且耗时与 candBlocks、S1 均线性相关、与 N1 无关。
// 现实现按白盒检视报告 §6 修复方向重写为「块级表每行一次构建 + Brcb 向量展开」：
//   (1) 块级 isOut 表【每行仅构建一次】（BuildCandBlockTable）：
//       GM→UB 一次性装载 candBlocks 个 int32 → 标量 8 路展开扫描（先集中发 8 条 UB 读令读延迟重叠，
//       再统一置位，消除逐元素分支串行化）→ 命中 [0, blkCnt) 的候选块在块级表置 0.0，其余保持 1.0；
//   (2) 位置级展开【每 tile】：candIsOutUb_[b*bs+e] = tbl[tileBlockBase+b]——bs==8 走「Level-2 Adds
//       向量整块搬到固定 staging + 单条 Brcb(8 repeat)」（零标量、零 per-tile fence）；bs>8 走
//       「标量预展开 64 元素 staging + 单条 Brcb」；bs<8 走标量展开（Brcb 32B 粒度不足）
//       ——见 ExpandCandMaskToPosition；
//   (3) 竞态面收敛（沿用 2026-09-21 修复轮结论）：
//       - 块级表在行内【仅由 S pipe 写】（初始化 + 置位同管道 → 无 V/S 跨管道 WAW，根因 #4 不复现）；
//       - 位置级掩码 candIsOutUb_ 【仅由 V pipe 写】（Brcb），pen 链为纯 V 读；
//       - 候选装载 GM→UB 走 MTE2，用 MTE2_S 事件 fence 排空后才标量读（实测再加
//         PipeBarrier<PIPE_MTE2> 会打断 CopyIn ping/pong 的 V_MTE2 事件流水，禁用）；
//       - 表 S 写 → 展开 V 读用 S_V 事件 fence（同 pool_key_indexer arch22 BuildExpandGatherTpl 模式）；
//       - 掩码路径无任何排序依赖（根因 #1 的交错对误读结构性不存在）。
//   脏数据防御（P6f）不变：-1/越界块号（∉[0,blkCnt)）按"非候选"处理，重复候选幂等。
//   语义等价性：原实现逐 tile 命中窗口 [tileBlockBase, tileBlockBase+tileBlkNum) 的并集 = [0, blkCnt)，
//   与本实现整行一次构建的表值域一致；blkCnt 之外两侧均按非候选处理。
//   tileBlockBase = cuBaseS2Idx/blockSize = s2Idx×tileBlkNum，512 = 2⁹ 整除任意 [2,64] 幂次
//   blockSize，无跨 tile 块（brmRepeat 截断类 R10 问题结构性不存在）。
//
// 容量与回退：块级表容量 CAND_TBL_CAP（每行分区），blkCnt ≤ CAP 时走上述快路径；
//   blkCnt > CAP（极小 blockSize × 超长 S2，如 bs=2 且 S2>14336）回退 BuildCandidateMaskLegacyTile
//   —— 即 2026-09-21 已验收的纯标量逐 tile 实现（语义等价，仅该形态无性能收益）。
// 编译期开关：SLI_CAND_MASK_LEGACY=1 强制全形态走标量实现（A/B 归因与应急回滚用）。
#ifndef SLI_CAND_MASK_LEGACY
#define SLI_CAND_MASK_LEGACY 0
#endif
// S4a：块级 isOut 表 → 位置级 candIsOutUb_ 展开（每 tile 一次）。
// 目标：candIsOutUb_[b*blockSize + e] = tbl[tileBlockBase + b]（float 位型 0x00000000 / 0x3F800000），
//       b ∈ [0, tileBlkNum)、e ∈ [0, blockSize)，共 s2BaseSize_(=512) 个位置。
// 三条路径（按 blockSize 分派，均为「块级读一次 + 向量/标量写 512 位置」，不再重扫 candBlocks）：
//   bs == 8：Level-2 Adds(+0.0f) 把表当前 tile 切片（64 floats）整块搬到【固定偏移】staging，
//            再单条 Brcb（8 repeat，每 repeat 取 8 个 b32 各广播成 32B=8 float）→ 512 位置；
//            全程 V 管道，每 tile 零标量、零 S_V fence（模型主规格走此路径）；
//   bs > 8 ：每块值需复制 bs/8 份，Adds 无法做扇出 → 标量预展开到 64 元素 staging
//            （tileBlkNum×(bs/8) ≡ 64），再单条 Brcb；staging 为 S 写，需 per-tile S_V fence；
//   bs < 8 ：Brcb 粒度（32B=8 float）不足以做 2/4 倍展开，走标量展开（每块一次 GetValue +
//            blockSize 次 SetValue，共 tileBlkNum 次读 + 512 次写），需 per-tile S_V fence。
// 说明（实测约束，arch22/910B3）：曾评估 vgather（Gather API）与「直接以表子张量为 Brcb src」两种
//   纯向量展开——两者在 src 为「随行分区变化的子张量」（innerS1Idx=1，基址 +11008B）时均读出错位
//   数据（表值全 0 或全 1 时不可见，混合 0/1 时显形，约半数行错），而同一地址的标量 GetValue 与
//   向量写（Cast）均正确；改为「固定偏移 staging + Brcb」后全形态通过。c220 亦未提供 Scatter，
//   故候选置位仍为标量。
template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::ExpandCandMaskToPosition(int32_t tileBlockBase,
                                                                                          int32_t tileBlkNum,
                                                                                          int32_t blockSize,
                                                                                          int32_t innerS1Idx)
{
    LocalTensor<float> tblRow = candBufUb_[static_cast<uint32_t>(innerS1Idx) * CAND_TBL_CAP];
    LocalTensor<int32_t> tblI = tblRow.template ReinterpretCast<int32_t>();
    if (blockSize >= static_cast<int32_t>(CAND_BRCB_FANOUT)) {
        // bs∈{8,16,32,64}：标量预展开到 64 元素 staging（每块值复制 bs/8 份，tileBlkNum×(bs/8) ≡ 64），
        // 再单条 Brcb（8 repeat）→ 512 位置。
        // 注：staging 取 candBuf_ 内【固定偏移】——实测以「随行分区变化的子张量」
        // （tblRow[tileBlockBase]，innerS1Idx=1 时基址 +11008B）直接作 Brcb/vgather 的 src 时，
        // arch22 上会读出错位数据（标量 GetValue 读同一地址正确，vector 写亦正确），
        // 故一律先标量搬到固定 staging 再广播。
        const int32_t fan = blockSize / static_cast<int32_t>(CAND_BRCB_FANOUT);
        if (fan == 1) {
            // bs==8（模型主规格）：staging 即块级表当前 tile 切片的等长拷贝，走【普通 Level-2 向量指令】
            // （Adds +0.0f，表值仅 0.0f/1.0f → 位精确）——常规向量指令对子张量偏移无 Brcb/vgather 的
            // 上述限制，故每 tile 零标量、零 S_V fence（表在行首 tile 已由 S_V fence 交棒 V 管道）
            LocalTensor<float> stageF = candBufUb_[CAND_STAGE_OFFSET];
            Adds(stageF, tblRow[tileBlockBase], 0.0f, static_cast<uint32_t>(tileBlkNum));
            PipeBarrier<PIPE_V>();
        } else {
            // bs∈{16,32,64}：每块值需复制 fan=bs/8 份 → 标量预展开（tileBlkNum×fan ≡ 64）
            // 【R3 修复 2026-09-23】覆写 staging 前排空上一 tile 的 V 读（Brcb 读同一固定 staging，
            // PipeBarrier<PIPE_V> 只约束 V-V，不约束后续 S 写 → 跨 tile WAR 面）
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            LocalTensor<int32_t> stage = candBufUb_[CAND_STAGE_OFFSET].template ReinterpretCast<int32_t>();
            for (int32_t b = 0; b < tileBlkNum; b++) {
                int32_t v = tblI.GetValue(static_cast<uint32_t>(tileBlockBase + b));
                for (int32_t j = 0; j < fan; j++) {
                    stage.SetValue(static_cast<uint32_t>(b * fan + j), v);
                }
            }
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIdSToV);
            WaitFlag<HardEvent::S_V>(eventIdSToV);
        }
        Brcb(candIsOutUb_, candBufUb_[CAND_STAGE_OFFSET], static_cast<uint8_t>(CAND_BRCB_FANOUT),
             {1, B32_VEC_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        return;
    }
    // bs∈{2,4}：标量展开（每块一次 GetValue + blockSize 次 SetValue）
    // 【R3 修复 2026-09-23】直写 candIsOutUb_ 前排空上一 tile 的 V 读（pen 链 Mul 读本掩码，
    // 其后的 PipeBarrier<PIPE_V> 只约束 V-V，不约束本 tile 的 S 覆写 → 跨 tile WAR 面）
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    LocalTensor<int32_t> isOutI = candIsOutUb_.template ReinterpretCast<int32_t>();
    for (int32_t b = 0; b < tileBlkNum; b++) {
        int32_t v = tblI.GetValue(static_cast<uint32_t>(tileBlockBase + b));
        for (int32_t e = 0; e < blockSize; e++) {
            isOutI.SetValue(static_cast<uint32_t>(b * blockSize + e), v);
        }
    }
    event_t eventIdSToV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV2);
    WaitFlag<HardEvent::S_V>(eventIdSToV2);
}

// S4a：块级 isOut 表构建（每行一次，tblBase 恒 0，覆盖 [0, blkCnt)）。
// 表内容：tbl[b] = 1.0f 位型（非候选）/ 0.0f 位型（候选），b ∈ [0, tblLen)。
// 全程仅 S pipe 写（初始化 + 命中置位同管道 → 无跨管道 WAW）；候选装载区独立于表区（candBuf_ 尾部）。
template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::BuildCandBlockTable(
    const SparseLICommon::RunInfo &info, int32_t cuS1Idx, int32_t blkCnt, int32_t innerS1Idx)
{
    const int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    LocalTensor<int32_t> tbl =
        candBufUb_[static_cast<uint32_t>(innerS1Idx) * CAND_TBL_CAP].template ReinterpretCast<int32_t>();
    const int32_t tblLen = static_cast<int32_t>(SparseLICommon::Align(blkCnt, static_cast<int32_t>(64)));

    // 【R3 修复 2026-09-23】表区按 innerS1Idx 分区，本行初始化（S 写）与上一轮同分区表数据
    // 的 V 读（展开路径 Adds/Brcb/pen 链）构成跨轮 WAR——覆写前排空 V 管道（每行一次，开销可忽略）
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // (1) 块级表初始化全 1.0（纯标量写 float 1.0f 位型 0x3F800000，与命中置位同管道）
    constexpr int32_t CAND_IS_OUT_ONE = 0x3F800000; // float 1.0f 位型
    for (int32_t i = 0; i < tblLen; i++) {
        tbl.SetValue(static_cast<uint32_t>(i), CAND_IS_OUT_ONE);
    }

    // (2) 候选行 GM→UB 一次性装载（int32 直存，无 Cast：块号仅作整数索引使用，P4 索引恒 int32 域）
    LocalTensor<int32_t> candInt = candBufUb_[CAND_LOAD_OFFSET].template ReinterpretCast<int32_t>();
    // 【R3 修复 2026-09-23 · racecheck 实证 2048 条/例 WAR】装载区（CAND_LOAD_OFFSET）跨行复用：
    // 上一行 (3) 的标量扫描 GetValue（PIPE_S Read）与本行 DataCopyPad（PIPE_MTE2 Write）覆写之间
    // 无排序事件（既有 MTE2_S 仅约束本行"写→读"方向）——MTE2 覆写前排空 S 读
    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
    DataCopyPad(candInt, candidateTopkIndexInGm[info.candidateOutOffset + static_cast<uint64_t>(cuS1Idx) * candBlocks],
                {1, static_cast<uint16_t>(candBlocks * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
    // MTE2 写 → 标量 GetValue 读可见（P2；MTE2_S 为 AIV 合法事件，参照 flash_attention_score_grad arch22）
    // 注：实测在此追加 PipeBarrier<PIPE_MTE2> 会破坏 CopyIn ping/pong 的 V_MTE2 事件流水（快路径全形态
    // 非确定性），故仅用事件 fence
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    // (3) 标量扫描（8 路展开：先集中发 8 条 UB 读令延迟重叠，再统一置位消除逐元素分支串行化）
    //     [0, blkCnt) 外的块号（-1 脏数据 / 越界）不写入 → 脏数据免疫；重复候选幂等
    int32_t w = 0;
    for (; w + CAND_SCAN_UNROLL <= candBlocks; w += CAND_SCAN_UNROLL) {
        int32_t b0 = candInt.GetValue(static_cast<uint32_t>(w));
        int32_t b1 = candInt.GetValue(static_cast<uint32_t>(w + 1));
        int32_t b2 = candInt.GetValue(static_cast<uint32_t>(w + 2));
        int32_t b3 = candInt.GetValue(static_cast<uint32_t>(w + 3));
        int32_t b4 = candInt.GetValue(static_cast<uint32_t>(w + 4));
        int32_t b5 = candInt.GetValue(static_cast<uint32_t>(w + 5));
        int32_t b6 = candInt.GetValue(static_cast<uint32_t>(w + 6));
        int32_t b7 = candInt.GetValue(static_cast<uint32_t>(w + 7));
        LIServiceVec::CandTblMark(tbl, b0, 0, blkCnt);
        LIServiceVec::CandTblMark(tbl, b1, 0, blkCnt);
        LIServiceVec::CandTblMark(tbl, b2, 0, blkCnt);
        LIServiceVec::CandTblMark(tbl, b3, 0, blkCnt);
        LIServiceVec::CandTblMark(tbl, b4, 0, blkCnt);
        LIServiceVec::CandTblMark(tbl, b5, 0, blkCnt);
        LIServiceVec::CandTblMark(tbl, b6, 0, blkCnt);
        LIServiceVec::CandTblMark(tbl, b7, 0, blkCnt);
    }
    for (; w < candBlocks; w++) {
        LIServiceVec::CandTblMark(tbl, candInt.GetValue(static_cast<uint32_t>(w)), 0, blkCnt);
    }

    // (4) 标量写 → 向量读（展开路径）可见；行内后续 tile 无标量写，该 fence 一次覆盖该行全部 tile
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

// S4a 回退路径：2026-09-21 已验收的纯标量逐 tile 掩码实现（blkCnt > CAND_TBL_CAP 时启用；
// SLI_CAND_MASK_LEGACY=1 时全形态启用）。语义与快路径一致，仅无性能收益。
template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::BuildCandidateMaskLegacyTile(
    const SparseLICommon::RunInfo &info, int32_t cuS1Idx, int32_t cuBaseS2Idx, int32_t innerS1Idx)
{
    int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t tileBlkNum = s2BaseSize_ / blockSize;
    int32_t tileBlockBase = cuBaseS2Idx / blockSize;
    LocalTensor<int32_t> candInt = candBufUb_[CAND_LOAD_OFFSET].template ReinterpretCast<int32_t>();
    LocalTensor<float> candRow = candBufUb_[static_cast<uint32_t>(innerS1Idx) * CAND_TBL_CAP];

    if (info.s2Idx == blockS2StartIdx_) {
        // 【R3 修复 2026-09-23】装载区跨行复用：上一行 Cast（PIPE_V 读 candInt）与本行
        // DataCopyPad（PIPE_MTE2 写）之间无排序事件（MTE2_V 只约束本行"写→读"方向）——覆写前排空 V 读。
        // 注：不可用 SetWaitFlag(V_MTE2)——FetchEventID 在空占用位图上返回 id0，与本流水中
        // CopyIn ping/pong 手工占用的 V_MTE2 id{0,1,2} 冲突；显式取空闲 id3
        constexpr uint32_t EVENTID_V_TO_MTE2_CAND = 3;
        SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_CAND);
        WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_CAND);
        DataCopyPad(candInt,
                    candidateTopkIndexInGm[info.candidateOutOffset + static_cast<uint64_t>(cuS1Idx) * candBlocks],
                    {1, static_cast<uint16_t>(candBlocks * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(EVENTID_MTE2_TO_V_CAND);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(candRow, candInt, RoundMode::CAST_NONE, candBlocks);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }
    constexpr int32_t CAND_IS_OUT_ONE = 0x3F800000; // float 1.0f 位型
    LocalTensor<int32_t> candIsOutInt = candIsOutUb_.template ReinterpretCast<int32_t>();
    // 【R3 修复 2026-09-23】candIsOutUb_ 标量覆写前排空上一 tile pen 链的 V 读：
    // 上方 :517 的 V_S 仅在行首 tile 执行且只覆盖 candRow 读序，不覆盖本循环的跨 tile WAR
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    for (int32_t p = 0; p < s2BaseSize_; p++) {
        candIsOutInt.SetValue(p, CAND_IS_OUT_ONE);
    }
    for (int32_t w = 0; w < candBlocks; w++) {
        int32_t candBlk = static_cast<int32_t>(candRow.GetValue(w));
        if (candBlk >= tileBlockBase && candBlk < tileBlockBase + tileBlkNum) {
            int32_t blkOff = (candBlk - tileBlockBase) * blockSize;
            for (int32_t e = 0; e < blockSize; e++) {
                candIsOutInt.SetValue(blkOff + e, 0);
            }
        }
    }
    event_t eventSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSToV);
    WaitFlag<HardEvent::S_V>(eventSToV);
}

template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::BuildCandidateMask(const SparseLICommon::RunInfo &info,
                                                                                    int32_t cuS1Idx,
                                                                                    int32_t cuBaseS2Idx,
                                                                                    int32_t innerS1Idx)
{
    const int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    const int32_t tileBlkNum = s2BaseSize_ / blockSize;    // 512/blockSize ∈ [8,256]
    const int32_t tileBlockBase = cuBaseS2Idx / blockSize; // = s2Idx × tileBlkNum
    // 本行块号上界（含 -inf pad 尾块）：numTiles × tileBlkNum，numTiles = ceil(actS2Size/512)
    // —— 与原实现逐 tile 窗口的并集一致，作为表值域与脏数据越界判定基准
    const int32_t blkCnt =
        static_cast<int32_t>(CeilDiv(info.actS2Size, static_cast<uint32_t>(s2BaseSize_))) * tileBlkNum;

#if !SLI_CAND_MASK_LEGACY
    if (blkCnt <= static_cast<int32_t>(CAND_TBL_CAP)) {
        // 行首 tile 构建块级表（候选随行走，每行一次）
        if (info.s2Idx == blockS2StartIdx_) {
            BuildCandBlockTable(info, cuS1Idx, blkCnt, innerS1Idx);
        }
        // 位置级展开：块级表 → 512 位置（bs==8 单条 Brcb；bs>8 staging+Brcb；bs<8 标量）
        // 读上界 = tileBlockBase + tileBlkNum - 1 ≤ blkCnt - 1 < tblLen ✓
        ExpandCandMaskToPosition(tileBlockBase, tileBlkNum, blockSize, innerS1Idx);
        return;
    }
#endif
    // 回退：blkCnt 超块级表容量（极小 blockSize × 超长 S2），或 SLI_CAND_MASK_LEGACY=1
    BuildCandidateMaskLegacyTile(info, cuS1Idx, cuBaseS2Idx, innerS1Idx);
}

template <typename LIT>
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::ProcessVec(const SparseLICommon::RunInfo &info)
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
        // globalTopkUb_ value,index=-inf,-1（consumer 无块级累加器，无成对重置点）
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
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

            // candidate (two-level topk) consumer 链 S4a（本算子唯一 candidate 链，恒执行——
            // P11 全候选快路径已撤销，禁止重新引入；score 行已成形、idx 已填充）
            BuildCandidateMask(info, cuS1Idx, cuBaseS2Idx, innerS1Idx);
            // leak pen 链（R11）：count = cuS2Len（当前 tile 可达前缀）——该区间 score 全为有限值
            // （不可达位置由 -inf pad 兜底，不进入 pen 链），杜绝 ±inf×0 / -inf+inf 的 NaN 路径。
            // pen = (NEG_HUGE' − score) × isOut；score += pen。候选只降级排序（候选外可达位置降级，
            // 仍可作填充入选），不可达位置保持 -inf；不做索引置 -1（QLI 原 isOutI32/CAST_RINT 链不移植）
            // 降级值内嵌全局索引序（2026-09-21 regime 2 语义修复）：golden 对并列降级位按索引升序
            // 填充，原实现所有降级位同为 -1e30 精确并列，topk 选取次序随硬件归并次序漂移（实测
            // 选到高索引块）。现降级值 = NEG_HUGE − 全局位置×2^76：fp32 在 1e30 量级 ulp=2^76，
            // 1e30 尾数 ≈1.32e7 < 2^24，全局位置 ≤ 2^24 内逐位精确、严格递减 → 降级位全序恒等
            // 索引升序，跨 tile 归并无并列；真实 score（|·|≪ulp）在加减中全被吸收，不影响在选位
            //
            // 【R1 修复 2026-09-23】禁用 ArithProgression（P3 铁律，与 LIV2 ProcessCandBlockTopk 禁则同源）：
            // 其 (8,64] count 分支内部走标量 SetValue 写 UB（PipeBarrier<PIPE_V> 不构成 fence）且将全局
            // 向量掩码泄漏为 8 lane（无 ResetMask）；cuS2Len 为尾 tile 余数/causal 短前缀（如 actS2=1033
            // → 尾 tile cuS2Len=9），恒可落入 (8,64]，紧随其后的 SortAll/MergeSort 正是污染敏感下游
            // （LIV2 侧同型调用曾实测出"单元素插入+右移"型随机错误）。
            // 改为位域等价构造（纯 V、count-based、无标量写、无 Cast）：
            //   NEG_HUGE − p×2^76 == bits(NEG_HUGE) + p（int32 模加，p = cuBaseS2Idx + i）
            // 推导：1e30 量级 fp32 数均为 ulp=2^76 的整数倍，bits(−1e30f)=0xF149F2CA（=−13234889×2^76
            // 的位型）；p ≤ 3.5M（尾数余量 2^24−1.32e7）内与原 float 链【逐位一致】（host 侧 numpy 全域
            // 对拍验证）；p 更大时整数加法进位跨指数量级仍严格单调——原 float 链在 p>2^24 反而出现
            // ulp 并列退化，位域构造的全序性更优，排序契约（降级位按索引升序）不变。
            // 回绕上界：bits 加至 -inf 位型(0xFF800000)需 p > 238,423,350（0xFF800000−0xF149F2CA），
            // S2 物理上限（PA: 65535×1024≈67M）的 3.5 倍，不可达；全域无 NaN/回绕风险。
            LocalTensor<int32_t> candNegHugeI32 = candNegHugeUb_.template ReinterpretCast<int32_t>();
            // p = globalTopkIndice_[i] + cuBaseS2Idx（复用 InitBuffers 常量索引 [0,512)，count ≤ 512 ✓）
            Adds(candNegHugeI32, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();
            // bits(NEG_HUGE − p×2^76) = NEG_HUGE_F32 + p（负 float 位型随 |value| 整数单调递增）
            Adds(candNegHugeI32, candNegHugeI32, LIServiceVec::NEG_HUGE_F32, cuS2Len);
            PipeBarrier<PIPE_V>();
            Sub(candPenUb_, candNegHugeUb_, sortScoreUb, cuS2Len);
            PipeBarrier<PIPE_V>();
            Mul(candPenUb_, candPenUb_, candIsOutUb_, cuS2Len);
            PipeBarrier<PIPE_V>();
            Add(sortScoreUb, sortScoreUb, candPenUb_, cuS2Len);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            // D6: consumer 恒走 SortAll+MergeSort（s1Base=4 下 SortedBasicBlock_ 视图越界，
            // 缓存粗排分支删除；isSparseCountOver2K 恒 false 仅为克隆表达式保留）
            // P3（2026-09-21 缺陷修复）：SortAll/MergeSort 尾部 UB→UB 回拷若走 MTE DataCopy，
            // 其后的 PipeBarrier<PIPE_V> 不构成 fence——跨 tile 对累加器 globalTopkUb_ /
            // 行排序结果 reduceOutBuff 的下一次 MrgSort 读存在调度敏感竞态（多 tile 实测
            // topk 边界元素错乱）。改用纯 V 管道位精确回拷变体 SortAllVecCopy/MergeSortVecCopy
            // （克隆自带但未接线），全程受 PipeBarrier 约束。
            {
                LIServiceVec::SortAllVecCopy(reduceOutBuff, tmpSortBuf,
                                             cuS2LenVecAlign); // cuS2LenVecAlign <= s2BaseSize_, fill -inf
                PipeBarrier<PIPE_V>();
                LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                // mrgDstNum = virTopK = 2048 ≤ 3072 恒走 MergeSortVecCopy 单次 MrgSort 分支；
                // elementLengths = [cuS2LenVecAlign ≤ 512, 2048]（实测合法）
                LIServiceVec::MergeSortVecCopy(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                               cuS2LenVecAlign, UbTmpSort);
            }
            if (constInfo_.isSparseCountOver2K) {
                SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            // noS2Split 恒 true（D4）→ blockS2StartIdx_ ≡ 0，行末直出条件恒等于 isS2End
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;

            // 中间结果保存（LD 分支为死路：noS2Split 下 needCopyOutGm 覆盖 needCopyWsGm，N7 死路审计）
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
                // consumer 无 candidate 直出
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
    if (LAYOUT_T == SLI_LAYOUT::BSND) {
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
__aicore__ inline void SparseLightningIndexerServiceVector<LIT>::ProcessLD()
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
} // namespace SparseLIKernel
#endif // SPARSE_LIGHTNING_INDEXER_SERVICE_VECTOR_ARCH22_H
