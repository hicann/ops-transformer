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
 * \file compressor_block_vec.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_H
#define COMPRESSOR_BLOCK_VEC_H

#include "compressor_comm_arch92.h"
#include "compressor_tools_arch92.h"
#include "vf/vf_softmax_arch92.h"
#include "vf/vf_add_arch92.h"
#include "vf/vf_mul_arch92.h"
#include "limits"
#include <cstddef>

using namespace AscendC;

namespace Compressor {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

/* ===================== 静态 tensor（UB）视图 =====================
 * vec 侧 buffer 地址取 struct 的成员偏移（编译期常量），合计 368K。*/
inline constexpr uint32_t COMP_UB_MM1_RES_BYTES =
    BUFFER_SIZE_BYTE_64K; // NORMAL 顺序复用 64K；批量预载/FULL_LOAD 各 32K
inline constexpr uint32_t COMP_UB_PARTIAL_STAGE_BYTES = BUFFER_SIZE_BYTE_32K; // 切 K partial 的搬入/累加暂存
inline constexpr uint32_t COMP_UB_OUT_STAGE_BYTES = BUFFER_SIZE_BYTE_32K;     // 输出暂存
inline constexpr uint32_t COMP_UB_VEC1_RES_BYTES = BUFFER_SIZE_BYTE_16K;      // Vec 归约结果
inline constexpr uint32_t COMP_UB_SCORE_BYTES = BUFFER_SIZE_BYTE_64K;         // 压缩窗 score
inline constexpr uint32_t COMP_UB_KV_BYTES = BUFFER_SIZE_BYTE_64K;            // 压缩窗 KV
inline constexpr uint32_t COMP_UB_APE_BYTES = BUFFER_SIZE_BYTE_64K;           // APE

struct CompressorVecUbView {
    uint8_t mm1Res[COMP_UB_MM1_RES_BYTES];
    uint8_t partialStage0[COMP_UB_PARTIAL_STAGE_BYTES];
    uint8_t partialStage1[COMP_UB_PARTIAL_STAGE_BYTES];
    uint8_t outStage[COMP_UB_OUT_STAGE_BYTES];
    uint8_t vec1Res[COMP_UB_VEC1_RES_BYTES];
    uint8_t scoreLocal[COMP_UB_SCORE_BYTES];
    uint8_t kvLocal[COMP_UB_KV_BYTES];
    uint8_t ape[COMP_UB_APE_BYTES];
};
inline constexpr uint32_t COMP_UB_TOTAL_BYTES = sizeof(CompressorVecUbView); // 368K

/* ===================== bufid（Mutex）同步 ID =====================
 * 不再手写 SetFlag/WaitFlag：每个 buffer 一个 mutex id，生产者 pipe Lock/Unlock 后
 * 交给消费者 pipe Lock/Unlock，跨轮复用也由 mutex 自身保证。
 * 本算子 vec 侧占用 0~8（arch92 event/mutex 共享同一池 0~15，AIC 的 SetFlag/WaitFlag
 * 用另一组 per-core 事件，不冲突）。*/
inline constexpr uint32_t COMP_MTX_MM1 = 0U;            // mm1 结果：MTE2 生产 -> V 消费
inline constexpr uint32_t COMP_MTX_PARTIAL_STAGE0 = 1U; // GM->UB staging 0：MTE2 -> V
inline constexpr uint32_t COMP_MTX_PARTIAL_STAGE1 = 2U; // GM->UB staging 1：MTE2 -> V
inline constexpr uint32_t COMP_MTX_OUT_STAGE = 3U;      // UB->GM staging：V -> MTE3
inline constexpr uint32_t COMP_MTX_APE_IN = 4U;         // FULL_LOAD APE staging：MTE2 -> V
/* GM 侧资源读写共用 id：同一块 GM 资源的「写(MTE3)」与「读(MTE2)」绑定同一个 mutex id，
 * 让「写 GM -> 读 GM」这条跨 pipe 依赖由 mutex 自身排到一条序上，不再只靠 V 流水串行兜底。
 * 三类资源各自独立（不同资源组不得共用一个 id）。同一 id 上的 Lock/Unlock 由
 * CopyInWithSync / CopyOutWithSync 统一成「V 会合 -> 生产者搬运 -> V 会合」三对，
 * 严格交替，跨调用不留悬空 credit。*/
inline constexpr uint32_t COMP_MTX_IN_APE = 5U;      // apeUb：只读，无 GM 回写
inline constexpr uint32_t COMP_MTX_STATE = 6U;       // stateCache：写(SaveState) + 读(ReadState) 共用
inline constexpr uint32_t COMP_MTX_CACHE_SCORE = 7U; // score tail cache：写 + 读共用
inline constexpr uint32_t COMP_MTX_CACHE_KV = 8U;    // kv tail cache：写 + 读共用

template <typename COMP>
class CompressorBlockVector {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = BUFFER_SIZE_BYTE_32K; // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float SOFTMAX_MIN_NUM = -std::numeric_limits<float>::infinity();
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorBlockVector(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo& constInfo, const CompressorTools<COMP>& tools,
                                      const uint32_t& cvRatio);
    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* wKv, __gm__ uint8_t* wGate,
                                __gm__ uint8_t* stateCache, __gm__ uint8_t* ape, __gm__ uint8_t* stateBlockTable,
                                __gm__ uint8_t* cuSeqlens, __gm__ uint8_t* seqUsed, __gm__ uint8_t* startPos,
                                __gm__ uint8_t* cmpKvOut, __gm__ uint8_t* softmaxScoreOut, __gm__ uint8_t* kvOut);
    // =================================资源管理=================================
    // 静态 tensor 初始化
    __aicore__ inline void InitStaticTensors();
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
                                                GlobalTensor<T> kvCacheTcGm, GlobalTensor<T> scoreCacheTcGm);

protected:
    GlobalTensor<T> scoreMm1ResGm_;
    GlobalTensor<T> kvMm1ResGm_;
    GlobalTensor<T> kvCacheTcGm_;
    GlobalTensor<T> scoreCacheTcGm_;

protected:
    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline LoopInfo GetLoopInfo(const Vec1SplitInfo& splitInfo);
    template <typename DST_T, typename SRC_T, typename O>
    __aicore__ inline void DataCopyAlign(const DST_T& dst, const SRC_T& src, uint32_t copyRowCount,
                                         uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToUb(const LocalTensor<O>& dstLocal, const LocalTensor<O>& srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignGmToUb(const LocalTensor<O>& dstLocal, const GlobalTensor<O>& srcGm,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToGm(const GlobalTensor<O>& dstGm, const LocalTensor<O>& srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void CopyOutWithSync(const GlobalTensor<O>& dstGm, const LocalTensor<O>& srcLocal,
                                           uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                           uint32_t dstSingleRowCount, uint32_t bufferId = COMP_MTX_OUT_STAGE);
    template <typename O>
    __aicore__ inline void CopyInWithSync(const LocalTensor<O>& dstLocal, const GlobalTensor<O>& srcGm,
                                          uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                          uint32_t dstSingleRowCount, uint32_t bufferId);
    template <typename O>
    __aicore__ inline void LoadAndSumPartials(const LocalTensor<O>& dstLocal, const GlobalTensor<O>& srcGm,
                                              uint32_t dealRowCount, uint32_t dealColCount, uint32_t srcSingleRowCount,
                                              uint32_t dstSingleRowCount, uint32_t repeatTimes, uint64_t offset);
    __aicore__ inline void LoadRawFromWorkspace(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGm,
                                                uint32_t preDealSeqCnt, uint32_t dealSeqCnt, uint32_t dStartIdx,
                                                uint32_t dDealSize);
    template <bool IS_READ>
    __aicore__ inline void AccessCacheState(const LocalTensor<T>& local, const GlobalTensor<T>& state,
                                            const GlobalTensor<int32_t>& blockTableGm, uint32_t batchIdx,
                                            uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx,
                                            uint32_t dDealSize, uint32_t stateIdx);
    template <bool IS_READ>
    __aicore__ inline void AccessCacheStateLinear(const LocalTensor<T>& local, const GlobalTensor<T>& state,
                                                  const GlobalTensor<int32_t>& blockTableGm, uint32_t batchIdx,
                                                  uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx,
                                                  uint32_t dDealSize, uint32_t stateIdx);
    template <bool IS_READ>
    __aicore__ inline void AccessCacheStateRing(const LocalTensor<T>& local, const GlobalTensor<T>& state,
                                                const GlobalTensor<int32_t>& blockTableGm, uint32_t batchIdx,
                                                uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx,
                                                uint32_t dDealSize, uint32_t stateIdx);
    __aicore__ inline void SaveToWorkSpace(const LocalTensor<T>& srcLocal, const GlobalTensor<T>& cacheTcGm,
                                           const Vec1SliceInfo& sliceInfo, const LoopInfo& loopInfo, uint32_t dStartIdx,
                                           uint32_t dDealSize, uint32_t bufferId);
    __aicore__ inline void LoadFromWorkSpace(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& cacheTcGm,
                                             const GlobalTensor<T>& srcGm, const LocalTensor<T>& srcLocal,
                                             const Vec1SliceInfo& sliceInfo, const LoopInfo& loopInfo,
                                             uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize,
                                             uint32_t bufferId);
    __aicore__ inline void SoftmaxDN(const LocalTensor<T>& scoreLocal, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void PadAlign(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                    const Vec1SliceInfo& sliceInfo, uint32_t dBaseOffset, uint32_t dDealSize,
                                    uint32_t dBaseSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T>& kvLocal, const LocalTensor<T>& scoreLocal,
                                            const LocalTensor<T>& dstLocal, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void CopyOutMidResToOutput(const LocalTensor<T>& kvLocal, const LocalTensor<T>& scoreLocal,
                                                 const Vec1SliceInfo& sliceInfo, uint32_t compressTcSize,
                                                 uint32_t dStartIdx, uint32_t dDealSize, uint32_t compressedCnt);
    __aicore__ inline void CopyOutVec1ResToOutput(const LocalTensor<T>& comperssoredUb, const Vec1SliceInfo& sliceInfo,
                                                  uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo& splitInfo);
    __aicore__ inline void SaveState(const LocalTensor<T>& srcLocal, const GlobalTensor<T>& stateGm,
                                     const GlobalTensor<int32_t>& blockTableGm, const Vec1SliceInfo& sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize, uint32_t stateIdx);
    template <bool IS_SCORE>
    __aicore__ inline void DuplicateFirstBlock(const LocalTensor<T>& dstLocal, uint32_t duplicateRowCount,
                                               uint32_t duplicateColCount, uint32_t singleRowCount);
    template <bool IS_SCORE>
    __aicore__ inline void ReadState(const LocalTensor<T>& srcLocal, const GlobalTensor<T>& stateGm,
                                     const GlobalTensor<int32_t>& blockTableGm, const Vec1SliceInfo& sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t compressedCnt_ = 0;
    bool isExistSeqUsed_ = false;
    bool isExistStartPos_ = false;
    uint32_t cvRatio_ = 2U;
    CompressorTools<COMP> tools_;
    ConstInfo constInfo_ = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> stateBlockTableGm_;
    GlobalTensor<T> stateCacheGm_;
    GlobalTensor<T> apeGm_;
    GlobalTensor<X_T> cmpKvOutGm_;
    GlobalTensor<T> softmaxScoreOutGm_;
    GlobalTensor<T> kvOutGm_;

    // ================================Local Buffer区（静态 tensor）====================================
    // 地址为编译期常量（CompressorVecUbView 成员偏移），不走 TPipe 分配；
    LocalTensor<T> mm1ResUb;                // 顺序复用 64K；批量预载时拆成下列两个视图
    LocalTensor<T> scoreUb;                 // mm1ResUb 前半
    LocalTensor<T> kvUb;                    // mm1ResUb 后半
    LocalTensor<uint8_t> partialStageUb[2]; // 切 K 归并的暂存
    LocalTensor<uint8_t> outStageUb;        // UB->GM staging
    LocalTensor<T> vec1ResUb;               // Vec 归约结果
    LocalTensor<T> scoreLocalUb;            // 压缩窗 score
    LocalTensor<T> kvLocalUb;               // 压缩窗 KV
    LocalTensor<T> apeUb;
    LocalTensor<T> apeInUb; // FULL_LOAD APE 分段载入 staging
};

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitParams(const ConstInfo& constInfo,
                                                               const CompressorTools<COMP>& tools,
                                                               const uint32_t& cvRatio)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    coff_ = static_cast<uint32_t>(COMP::coff);
    cmpRatio_ = constInfo.cmpRatio;
    cvRatio_ = cvRatio;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::Init(__gm__ uint8_t* x, __gm__ uint8_t* wKv, __gm__ uint8_t* wGate,
                                                         __gm__ uint8_t* stateCache, __gm__ uint8_t* ape,
                                                         __gm__ uint8_t* stateBlockTable, __gm__ uint8_t* cuSeqlens,
                                                         __gm__ uint8_t* seqUsed, __gm__ uint8_t* startPos,
                                                         __gm__ uint8_t* cmpKvOut, __gm__ uint8_t* softmaxScoreOut,
                                                         __gm__ uint8_t* kvOut)
{
    stateBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)stateBlockTable);
    stateCacheGm_.SetGlobalBuffer((__gm__ T*)stateCache);
    apeGm_.SetGlobalBuffer((__gm__ T*)ape);
    cmpKvOutGm_.SetGlobalBuffer((__gm__ X_T*)cmpKvOut);
    softmaxScoreOutGm_.SetGlobalBuffer((__gm__ T*)softmaxScoreOut);
    kvOutGm_.SetGlobalBuffer((__gm__ T*)kvOut);
    isExistSeqUsed_ = (seqUsed != nullptr);
    isExistStartPos_ = (startPos != nullptr);
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t*)cuSeqlens);
    }
    if (isExistSeqUsed_) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t*)seqUsed);
    }
    if (isExistStartPos_) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t*)startPos);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitStaticTensors()
{
    // UB 静态布局总量必须落在 arch92 的 368K 预算内（尺寸与改造前一致，不做复用压缩）
    static_assert(COMP_UB_TOTAL_BYTES <= 368U * 1024U, "compress vec UB static layout exceeds 368K budget");
    // 静态 tensor：UB 地址取 struct 成员偏移
    mm1ResUb = LocalTensor<uint8_t>(TPosition::VECIN, offsetof(CompressorVecUbView, mm1Res), COMP_UB_MM1_RES_BYTES)
                   .template ReinterpretCast<T>();
    scoreUb = mm1ResUb;
    kvUb = mm1ResUb[BUFFER_SIZE_BYTE_32K / sizeof(T)];
    partialStageUb[0] = LocalTensor<uint8_t>(TPosition::VECIN, offsetof(CompressorVecUbView, partialStage0),
                                             COMP_UB_PARTIAL_STAGE_BYTES);
    partialStageUb[1] = LocalTensor<uint8_t>(TPosition::VECIN, offsetof(CompressorVecUbView, partialStage1),
                                             COMP_UB_PARTIAL_STAGE_BYTES);
    outStageUb =
        LocalTensor<uint8_t>(TPosition::VECOUT, offsetof(CompressorVecUbView, outStage), COMP_UB_OUT_STAGE_BYTES);
    vec1ResUb = LocalTensor<uint8_t>(TPosition::VECOUT, offsetof(CompressorVecUbView, vec1Res), COMP_UB_VEC1_RES_BYTES)
                    .template ReinterpretCast<T>();
    scoreLocalUb =
        LocalTensor<uint8_t>(TPosition::VECCALC, offsetof(CompressorVecUbView, scoreLocal), COMP_UB_SCORE_BYTES)
            .template ReinterpretCast<T>();
    kvLocalUb = LocalTensor<uint8_t>(TPosition::VECCALC, offsetof(CompressorVecUbView, kvLocal), COMP_UB_KV_BYTES)
                    .template ReinterpretCast<T>();
    apeUb = LocalTensor<uint8_t>(TPosition::VECCALC, offsetof(CompressorVecUbView, ape), COMP_UB_APE_BYTES)
                .template ReinterpretCast<T>();
    // APE 后半 32K 兼作 FULL_LOAD 的 APE 分段载入 staging
    apeInUb = apeUb[BUFFER_SIZE_BYTE_32K / sizeof(T)];
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::AllocEventID()
{}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::FreeEventID()
{
    // 所有 Lock/Unlock 都在 CopyInWithSync / CopyOutWithSync 内部成对闭合，无悬空 credit 需要补偿。
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm,
                                                                         GlobalTensor<T> scoreMm1ResGm,
                                                                         GlobalTensor<T> kvCacheTcGm,
                                                                         GlobalTensor<T> scoreCacheTcGm)
{
    this->kvMm1ResGm_ = kvMm1ResGm;
    this->scoreMm1ResGm_ = scoreMm1ResGm;
    this->kvCacheTcGm_ = kvCacheTcGm;
    this->scoreCacheTcGm_ = scoreCacheTcGm;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed_) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    } else {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            return (uint32_t)(cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx));
        } else {
            return constInfo_.sSize;
        }
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline LoopInfo CompressorBlockVector<COMP>::GetLoopInfo(const Vec1SplitInfo& splitInfo)
{
    LoopInfo loopInfo;
    loopInfo.groupSize = splitInfo.vec1GroupSize;
    loopInfo.groupNum = splitInfo.vec1GroupNum;
    loopInfo.coreRowIdx = GetBlockIdx() / splitInfo.vec1GroupSize;
    loopInfo.coreColIdx = GetBlockIdx() % splitInfo.vec1GroupSize;
    loopInfo.isCoreRowLast = loopInfo.coreRowIdx == splitInfo.vec1GroupNum - 1;
    loopInfo.isCoreRowFirst = loopInfo.coreRowIdx == 0;
    return loopInfo;
}

template <typename COMP>
template <typename DST_T, typename SRC_T, typename O>
__aicore__ inline void CompressorBlockVector<COMP>::DataCopyAlign(const DST_T& dst, const SRC_T& src,
                                                                  uint32_t copyRowCount, uint32_t copyColCount,
                                                                  uint32_t srcSingleRowCount,
                                                                  uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / BlockElementNum<O>();
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / BlockElementNum<O>();
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / BlockElementNum<O>();
    DataCopy(dst, src, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVector<COMP>::DataCopyAlignUbToUb(const LocalTensor<O>& dstLocal,
                                                                        const LocalTensor<O>& srcLocal,
                                                                        uint32_t copyRowCount, uint32_t copyColCount,
                                                                        uint32_t srcSingleRowCount,
                                                                        uint32_t dstSingleRowCount)
{
    DataCopyAlign<LocalTensor<O>, LocalTensor<O>, O>(dstLocal, srcLocal, copyRowCount, copyColCount, srcSingleRowCount,
                                                     dstSingleRowCount);
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVector<COMP>::DataCopyAlignGmToUb(const LocalTensor<O>& dstLocal,
                                                                        const GlobalTensor<O>& srcGm,
                                                                        uint32_t copyRowCount, uint32_t copyColCount,
                                                                        uint32_t srcSingleRowCount,
                                                                        uint32_t dstSingleRowCount)
{
    DataCopyAlign<LocalTensor<O>, GlobalTensor<O>, O>(dstLocal, srcGm, copyRowCount, copyColCount, srcSingleRowCount,
                                                      dstSingleRowCount);
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVector<COMP>::DataCopyAlignUbToGm(const GlobalTensor<O>& dstGm,
                                                                        const LocalTensor<O>& srcLocal,
                                                                        uint32_t copyRowCount, uint32_t copyColCount,
                                                                        uint32_t srcSingleRowCount,
                                                                        uint32_t dstSingleRowCount)
{
    DataCopyAlign<GlobalTensor<O>, LocalTensor<O>, O>(dstGm, srcLocal, copyRowCount, copyColCount, srcSingleRowCount,
                                                      dstSingleRowCount);
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVector<COMP>::CopyOutWithSync(const GlobalTensor<O>& dstGm,
                                                                    const LocalTensor<O>& srcLocal,
                                                                    uint32_t copyRowCount, uint32_t copyColCount,
                                                                    uint32_t srcSingleRowCount,
                                                                    uint32_t dstSingleRowCount, uint32_t bufferId)
{
    if (copyRowCount == 0) {
        return;
    }
    // stateCache / tail cache 的写与读共用一个 bufferId，直搬快道直接挂到该 id 上；
    // 其余（最终输出等）仍用 COMP_MTX_OUT_STAGE。
    const bool isSharedGmRes = bufferId != COMP_MTX_OUT_STAGE;
    if (srcSingleRowCount == dstSingleRowCount) {
        // 快道：源/目标行跨距相同 -> 一次 strided DataCopy 从 srcLocal 直搬 GM，省掉 staging 的 UB->UB 重排。
        // 同步全部走 bufid（占同一个 id），与 CopyInWithSync 同构的三对 Lock/Unlock：
        //   1) V 侧先过一对锁：把「srcLocal 已写完」排到 MTE3 之前（RAW，源是 V 算出来的），
        //      同时等同一 GM 资源上一次读写（MTE2 读 / MTE3 写）结束
        //   2) MTE3 搬出
        //   3) V 侧再等一次：MTE3 读完 srcLocal 后 V 才能继续（WAR，V_MTE3 等待，
        //      也保证下一轮 V 覆写 srcLocal 安全）
        Mutex::Lock<PIPE_V>(bufferId);
        Mutex::Unlock<PIPE_V>(bufferId);
        Mutex::Lock<PIPE_MTE3>(bufferId);
        DataCopyAlignUbToGm(dstGm, srcLocal, copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
        Mutex::Unlock<PIPE_MTE3>(bufferId);
        Mutex::Lock<PIPE_V>(bufferId);
        Mutex::Unlock<PIPE_V>(bufferId);
        return;
    }
    // staging 路径：staging UB 本身仍由 COMP_MTX_OUT_STAGE 管；GM 侧资源额外挂一次排空，
    // 保证同一 GM 资源的上一次读（MTE2）已结束、本次写完成后再允许下一次读。
    if (isSharedGmRes) {
        Mutex::Lock<PIPE_V>(bufferId);
        Mutex::Unlock<PIPE_V>(bufferId);
    }
    uint32_t singleCopyRowCount = BUFFER_SIZE_BYTE_32K / (copyColCount * sizeof(O));
    for (uint32_t rowCount = 0; rowCount < copyRowCount; rowCount += singleCopyRowCount) {
        LocalTensor<O> outputUb = outStageUb.template ReinterpretCast<O>();

        uint64_t srcOffset = rowCount * srcSingleRowCount;
        uint64_t dstOffset = rowCount * dstSingleRowCount;
        uint32_t curCopyRowCount = min(singleCopyRowCount, copyRowCount - rowCount);

        // bufid 同步：V 写 staging -> MTE3 读 staging（mutex 同时保证跨轮复用）
        Mutex::Lock<PIPE_V>(COMP_MTX_OUT_STAGE);
        DataCopyAlignUbToUb(outputUb, srcLocal[srcOffset], curCopyRowCount, copyColCount, srcSingleRowCount,
                            copyColCount);
        Mutex::Unlock<PIPE_V>(COMP_MTX_OUT_STAGE);

        Mutex::Lock<PIPE_MTE3>(COMP_MTX_OUT_STAGE);
        DataCopyAlignUbToGm(dstGm[dstOffset], outputUb, curCopyRowCount, copyColCount, copyColCount, dstSingleRowCount);
        Mutex::Unlock<PIPE_MTE3>(COMP_MTX_OUT_STAGE);
    }
    if (isSharedGmRes) {
        // MTE3 是 in-order：这里的 Unlock 在全部 GM 写完成之后才生效，随后 V 等它 = 写 GM 已完成
        Mutex::Lock<PIPE_MTE3>(bufferId);
        Mutex::Unlock<PIPE_MTE3>(bufferId);
        Mutex::Lock<PIPE_V>(bufferId);
        Mutex::Unlock<PIPE_V>(bufferId);
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVector<COMP>::CopyInWithSync(const LocalTensor<O>& dstLocal,
                                                                   const GlobalTensor<O>& srcGm, uint32_t copyRowCount,
                                                                   uint32_t copyColCount, uint32_t srcSingleRowCount,
                                                                   uint32_t dstSingleRowCount, uint32_t bufferId)
{
    if (copyRowCount == 0) {
        return;
    }
    // 调用方保证列宽与两侧行跨距均为 32B 对齐；源/目标跨距可独立设置。
    // 与 CopyOutWithSync 同构的三对 Lock/Unlock（V 会合 -> MTE2 搬入 -> V 会合）：
    //   1) V 先过一对锁：把「上一轮 V 读」和「同一 GM 资源上一次写」排到 MTE2 之前（WAR）
    //   2) MTE2 直搬 GM -> dstLocal
    //   3) V 再等一次：MTE2 搬完 V 才能读 dstLocal（RAW），也保证下一次覆写安全
    // 三对锁在单次调用内闭合，同一 id 上的 Lock/Unlock 跨调用严格交替，不依赖跨调用 credit。
    Mutex::Lock<PIPE_V>(bufferId);
    Mutex::Unlock<PIPE_V>(bufferId);
    Mutex::Lock<PIPE_MTE2>(bufferId);
    DataCopyAlignGmToUb(dstLocal, srcGm, copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
    Mutex::Unlock<PIPE_MTE2>(bufferId);
    Mutex::Lock<PIPE_V>(bufferId);
    Mutex::Unlock<PIPE_V>(bufferId);
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVector<COMP>::LoadAndSumPartials(
    const LocalTensor<O>& dstLocal, const GlobalTensor<O>& srcGm, uint32_t dealRowCount, uint32_t dealColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount, uint32_t repeatTimes, uint64_t offset)
{
    uint32_t cnt = dealRowCount * dstSingleRowCount;
    if (cnt == 0) {
        return;
    }
    constexpr uint32_t bufElem = BUFFER_SIZE_BYTE_32K / sizeof(O);
    uint32_t rowsPerSplit = bufElem / dstSingleRowCount;
    uint32_t splitNum = CeilDivT(dealRowCount, rowsPerSplit);
    for (uint32_t splitIdx = 0; splitIdx < splitNum; splitIdx++) {
        uint32_t rowStart = splitIdx * rowsPerSplit;
        uint32_t rowCnt = min(rowsPerSplit, dealRowCount - rowStart);
        uint32_t splitCnt = rowCnt * dstSingleRowCount;
        uint32_t groupSize = bufElem / splitCnt;
        uint32_t loopTimes = CeilDivT(repeatTimes, groupSize);
        uint64_t srcGmOffset = static_cast<uint64_t>(rowStart) * srcSingleRowCount;
        for (uint32_t idx = 0; idx < loopTimes; idx++) {
            uint32_t stageId = (splitIdx * loopTimes + idx) % 2;
            LocalTensor<O> splitLocal = partialStageUb[stageId].template ReinterpretCast<O>();
            uint32_t stageMtx = COMP_MTX_PARTIAL_STAGE0 + stageId;
            uint32_t curGroupSize = min(groupSize, (repeatTimes - groupSize * idx));
            // bufid 同步：MTE2 填 staging -> V 做 MultiAdd
            Mutex::Lock<PIPE_MTE2>(stageMtx);
            if (srcSingleRowCount == dealColCount && dstSingleRowCount == dealColCount) {
                for (uint32_t groupIdx = 0; groupIdx < curGroupSize; groupIdx++) {
                    DataCopy(splitLocal[groupIdx * splitCnt], srcGm[srcGmOffset], splitCnt);
                    srcGmOffset += offset;
                }
            } else {
                for (uint32_t groupIdx = 0; groupIdx < curGroupSize; groupIdx++) {
                    DataCopyAlignGmToUb(splitLocal[groupIdx * splitCnt], srcGm[srcGmOffset], rowCnt, dealColCount,
                                        srcSingleRowCount, dstSingleRowCount);
                    srcGmOffset += offset;
                }
            }
            Mutex::Unlock<PIPE_MTE2>(stageMtx);

            Mutex::Lock<PIPE_V>(stageMtx);
            PipeBarrier<PIPE_V>();
            if (idx == 0) {
                MultiAddVF<true>(dstLocal[rowStart * dstSingleRowCount], splitLocal, rowCnt, dealColCount,
                                 dstSingleRowCount, curGroupSize, splitCnt);
            } else {
                MultiAddVF<false>(dstLocal[rowStart * dstSingleRowCount], splitLocal, rowCnt, dealColCount,
                                  dstSingleRowCount, curGroupSize, splitCnt);
            }
            Mutex::Unlock<PIPE_V>(stageMtx);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::LoadRawFromWorkspace(const LocalTensor<T>& dstLocal,
                                                                         const GlobalTensor<T>& srcGm,
                                                                         uint32_t preDealSeqCnt, uint32_t dealSeqCnt,
                                                                         uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t srcSingleRowElemNum = constInfo_.headDim;
    uint32_t copyRowCount = dealSeqCnt * coff_;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize;
    uint64_t srcGmOffset = (uint64_t)preDealSeqCnt * srcSingleRowElemNum * coff_ + dStartIdx;
    if (constInfo_.kBaseNum == 1) {
        DataCopyAlignGmToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
    } else {
        LoadAndSumPartials(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount,
                           dstSingleRowCount, constInfo_.kBaseNum, constInfo_.mm1KvResSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::SaveToWorkSpace(const LocalTensor<T>& srcLocal,
                                                                    const GlobalTensor<T>& cacheTcGm,
                                                                    const Vec1SliceInfo& sliceInfo,
                                                                    const LoopInfo& loopInfo, uint32_t dStartIdx,
                                                                    uint32_t dDealSize, uint32_t bufferId)
{
    uint64_t curSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt;
    uint64_t totalSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.bSeqUsed;
    if (!loopInfo.isCoreRowLast || !loopInfo.isCoreLoopLast || !sliceInfo.isLast || totalSeqLen < cmpRatio_ ||
        curSeqLen > Trunc(totalSeqLen, (uint64_t)cmpRatio_) - cmpRatio_) {
        return;
    }
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint64_t srcLocalOffset =
        (sliceInfo.dealedSeqCnt + sliceInfo.validSeqCnt - min(sliceInfo.validSeqCnt, cmpRatio_)) * srcSingleRowElemNum;
    CopyOutWithSync(cacheTcGm[dStartIdx], srcLocal[srcLocalOffset],
                    curSeqLen - max(curSeqLen - cmpRatio_, sliceInfo.bStartPos), dDealSize, coff_ * dDealSize,
                    constInfo_.headDim, bufferId);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::LoadFromWorkSpace(
    const LocalTensor<T>& dstLocal, const GlobalTensor<T>& cacheTcGm, const GlobalTensor<T>& srcGm,
    const LocalTensor<T>& srcLocal, const Vec1SliceInfo& sliceInfo, const LoopInfo& loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize, uint32_t bufferId)
{
    if (sliceInfo.sIdx == 0) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, cmpRatio_);
    uint64_t dstLocalOffset =
        (sliceInfo.compressoredScCnt * cmpRatio_ + cmpRatio_ - copyRowCount) * dstSingleRowElemNum;
    if (loopInfo.isCoreRowFirst && loopInfo.isCoreLoopFirst && sliceInfo.isFirst) { // 从cacheGm获取
        uint32_t srcSingleRowElemNum = constInfo_.headDim;
        uint64_t srcLocalOffset = dStartIdx;

        CopyInWithSync(dstLocal[dstLocalOffset], cacheTcGm[srcLocalOffset], copyRowCount, dDealSize,
                       srcSingleRowElemNum, coff_ * dDealSize, bufferId);
    } else if (sliceInfo.isFirst) { // 从存放MatMul结果的WorkSpace中获取
        uint32_t srcSingleRowElemNum = constInfo_.headDim * coff_;
        uint64_t srcLocalOffset =
            (globalSeqIdx + sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum + dStartIdx;

        if (constInfo_.kBaseNum == 1) {
            CopyInWithSync(dstLocal[dstLocalOffset], srcGm[srcLocalOffset], copyRowCount, dDealSize,
                           srcSingleRowElemNum, coff_ * dDealSize, bufferId);
        } else {
            LoadAndSumPartials(dstLocal[dstLocalOffset], srcGm[srcLocalOffset], copyRowCount, dDealSize,
                               srcSingleRowElemNum, coff_ * dDealSize, constInfo_.kBaseNum, constInfo_.mm1KvResSize);
        }
    } else { // 从UB中获取
        uint32_t srcSingleRowElemNum = dDealSize * coff_;
        uint64_t srcLocalOffset = (sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum;
        DataCopyAlignUbToUb(dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], copyRowCount, dDealSize,
                            srcSingleRowElemNum, coff_ * dDealSize);
    }
}

template <typename COMP>
template <bool IS_READ>
__aicore__ inline void CompressorBlockVector<COMP>::AccessCacheStateLinear(const LocalTensor<T>& local,
                                                                           const GlobalTensor<T>& state,
                                                                           const GlobalTensor<int32_t>& blockTableGm,
                                                                           uint32_t batchIdx, uint32_t startSeqIdx,
                                                                           uint32_t endSeqIdx, uint32_t dStartIdx,
                                                                           uint32_t dDealSize, uint32_t stateIdx)
{
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCount > seqCnt) {
            copyRowCount = seqCnt - copyFinishRowCnt;
        }
        if constexpr (IS_READ) {
            uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                   remainRowCnt * STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;
            CopyInWithSync(local[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount, dDealSize,
                           coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR, coff_ * dDealSize, COMP_MTX_STATE);
        } else if (idInBlockTable != 0) { // 32
            uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                   remainRowCnt * STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;
            CopyOutWithSync(state[stateOffset], local[copyFinishRowCnt * coff_ * dDealSize], copyRowCount, dDealSize,
                            coff_ * dDealSize, coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR, COMP_MTX_STATE);
        }
        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
template <bool IS_READ>
__aicore__ inline void CompressorBlockVector<COMP>::AccessCacheStateRing(const LocalTensor<T>& local,
                                                                         const GlobalTensor<T>& state,
                                                                         const GlobalTensor<int32_t>& blockTableGm,
                                                                         uint32_t batchIdx, uint32_t startSeqIdx,
                                                                         uint32_t endSeqIdx, uint32_t dStartIdx,
                                                                         uint32_t dDealSize, uint32_t stateIdx)
{
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    uint64_t idInBlockTable = blockTableGm.GetValue(batchIdx);
    while (copyFinishRowCnt < seqCnt) {
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCount > seqCnt) {
            copyRowCount = seqCnt - copyFinishRowCnt;
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                               remainRowCnt * STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim +
                               stateIdx * coff_ * constInfo_.headDim + dStartIdx;
        if constexpr (IS_READ) {
            CopyInWithSync(local[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount, dDealSize,
                           coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR, coff_ * dDealSize, COMP_MTX_STATE);
        } else {
            CopyOutWithSync(state[stateOffset], local[copyFinishRowCnt * coff_ * dDealSize], copyRowCount, dDealSize,
                            coff_ * dDealSize, coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR, COMP_MTX_STATE);
        }
        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
template <bool IS_READ>
__aicore__ inline void CompressorBlockVector<COMP>::AccessCacheState(const LocalTensor<T>& local,
                                                                     const GlobalTensor<T>& state,
                                                                     const GlobalTensor<int32_t>& blockTableGm,
                                                                     uint32_t batchIdx, uint32_t startSeqIdx,
                                                                     uint32_t endSeqIdx, uint32_t dStartIdx,
                                                                     uint32_t dDealSize, uint32_t stateIdx)
{
    if constexpr (COMP::cacheMode == CACHE_MODE::LINEAR_BUFFER) {
        AccessCacheStateLinear<IS_READ>(local, state, blockTableGm, batchIdx, startSeqIdx, endSeqIdx, dStartIdx,
                                        dDealSize, stateIdx);
    } else {
        AccessCacheStateRing<IS_READ>(local, state, blockTableGm, batchIdx, startSeqIdx, endSeqIdx, dStartIdx,
                                      dDealSize, stateIdx);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::SaveState(const LocalTensor<T>& srcLocal,
                                                              const GlobalTensor<T>& stateGm,
                                                              const GlobalTensor<int32_t>& blockTableGm,
                                                              const Vec1SliceInfo& sliceInfo, uint32_t dStartIdx,
                                                              uint32_t dDealSize, uint32_t dBaseSize, uint32_t stateIdx)
{
    uint64_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dBaseSize;

    if constexpr (COMP::cacheMode == CACHE_MODE::RING_BUFFER) {
        uint64_t compressSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.bSeqUsed, (uint64_t)cmpRatio_);
        uint32_t writeSeqStartIdx =
            compressSeqIdx > (coff_ - 1) * cmpRatio_ ? compressSeqIdx - (coff_ - 1) * cmpRatio_ : 0;
        if (endSeqIdx <= writeSeqStartIdx) {
            return;
        }
        srcBaseOffset += (max(startSeqIdx, (uint64_t)writeSeqStartIdx) - startSeqIdx) * coff_ * dBaseSize;
        startSeqIdx = max(startSeqIdx, (uint64_t)writeSeqStartIdx);
    }

    if constexpr (COMP::coff == COFF::OVERLAP) {
        AccessCacheState<false>(srcLocal[srcBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                                dStartIdx, dDealSize, stateIdx);
        srcBaseOffset += dBaseSize;
        dStartIdx += constInfo_.headDim;
    }

    AccessCacheState<false>(srcLocal[srcBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                            dStartIdx, dDealSize, stateIdx);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVector<COMP>::DuplicateFirstBlock(const LocalTensor<T>& dstLocal,
                                                                        uint32_t duplicateRowCount,
                                                                        uint32_t duplicateColCount,
                                                                        uint32_t singleRowCount)
{
    for (uint32_t offset = 0; offset < duplicateColCount; offset += FP32_REPEAT_ELEMENT_NUM) {
        uint32_t curDuplicateColCount = min(duplicateColCount - offset, FP32_REPEAT_ELEMENT_NUM);
        if constexpr (IS_SCORE) {
            Duplicate(dstLocal[offset], SOFTMAX_MIN_NUM, curDuplicateColCount, duplicateRowCount, 1,
                      singleRowCount / REPEAT_STRIDE_NUM);
        } else {
            Duplicate(dstLocal[offset], FLOAT_ZERO, curDuplicateColCount, duplicateRowCount, 1,
                      singleRowCount / REPEAT_STRIDE_NUM);
        }
    }
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVector<COMP>::ReadState(const LocalTensor<T>& dstLocal,
                                                              const GlobalTensor<T>& stateGm,
                                                              const GlobalTensor<int32_t>& blockTableGm,
                                                              const Vec1SliceInfo& sliceInfo, uint32_t dStartIdx,
                                                              uint32_t dDealSize, uint32_t stateIdx)
{
    // 没有需要压缩的块时, 不需要读state的信息
    if (sliceInfo.compressTcSize == 0) {
        return;
    }
    // 填充右边
    if (sliceInfo.headHolderSeqCnt > 0) {
        // 整个batch的第一块
        uint64_t startSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, (uint64_t)cmpRatio_);
        uint32_t endSeqIdx = sliceInfo.bStartPos;
        uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * cmpRatio_ * coff_ * dDealSize;
        if constexpr (COMP::coff == Compressor::COFF::OVERLAP) {
            dstBaseOffset += (coff_ - 1) * dDealSize;
        }
        AccessCacheState<true>(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                               dStartIdx + (coff_ - 1) * constInfo_.headDim, dDealSize, stateIdx);
    }

    // 填充左边
    if constexpr (COMP::coff == Compressor::COFF::OVERLAP) {
        bool isFirst = sliceInfo.bStartPos + sliceInfo.sIdx < cmpRatio_;
        if (isFirst) {
            // 无历史数据
            // dDealSize必须为64
            uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * cmpRatio_ * coff_ * dDealSize;
            DuplicateFirstBlock<IS_SCORE>(dstLocal[dstBaseOffset], cmpRatio_, dDealSize, coff_ * dDealSize);
        }
        if (sliceInfo.sIdx < cmpRatio_ && (!isFirst || sliceInfo.compressTcSize > 1)) {
            uint32_t startSeqIdx = sliceInfo.bStartPos < cmpRatio_ ?
                                       0 :
                                       Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, (uint64_t)cmpRatio_) - cmpRatio_;
            uint32_t endSeqIdx = min(
                Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, (uint64_t)cmpRatio_) - cmpRatio_,
                sliceInfo.bStartPos);
            uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * cmpRatio_ * coff_ * dDealSize;
            if (isFirst) {
                dstBaseOffset += cmpRatio_ * coff_ * dDealSize;
            }
            AccessCacheState<true>(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx,
                                   endSeqIdx, dStartIdx, dDealSize, stateIdx);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::SoftmaxDN(const LocalTensor<T>& scoreLocal, uint32_t tcDealSize,
                                                              uint32_t dDealSize)
{
    uint32_t ReduceSize = coff_ * cmpRatio_;
    FaVectorApi::SoftmaxDnVF<T>(scoreLocal, scoreLocal, dDealSize, ReduceSize, tcDealSize, SOFTMAX_MIN_NUM, dDealSize);
}

template <typename COMP>

__aicore__ inline void CompressorBlockVector<COMP>::PadAlign(const LocalTensor<T>& dstLocal,
                                                             const LocalTensor<T>& srcLocal,
                                                             const Vec1SliceInfo& sliceInfo, uint32_t dBaseOffset,
                                                             uint32_t dDealSize, uint32_t dBaseSize)
{
    // Ub data layout after overlap when r = 4 and coff = 2:
    //  Tc0_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq04: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq04: |--- --D_L--- -|------D_R-----|
    uint32_t srcSingleRowElemNum = dBaseSize * this->coff_;
    uint32_t copyRowCount = sliceInfo.compressTcSize * this->cmpRatio_ - sliceInfo.headHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize * this->coff_; // left和right在seq方向是交错存储的
    uint64_t srcLocalOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum + dBaseOffset;

    uint64_t dstUbOffset = sliceInfo.compressoredScCnt * this->cmpRatio_ * dstSingleRowCount;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 左侧
        uint64_t preSrcLocalOffset = srcLocalOffset;
        uint64_t preDstUbOffset = dstUbOffset + (sliceInfo.headHolderSeqCnt + this->cmpRatio_) * dstSingleRowCount;
        this->DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
                                  copyRowCount - min(copyRowCount, this->cmpRatio_), copyColCount, srcSingleRowCount,
                                  dstSingleRowCount);
        dstUbOffset += dDealSize;
        srcLocalOffset += dBaseSize;
    }
    // 右侧
    dstUbOffset += sliceInfo.headHolderSeqCnt * dstSingleRowCount;
    this->DataCopyAlignUbToUb(dstLocal[dstUbOffset], srcLocal[srcLocalOffset], copyRowCount, copyColCount,
                              srcSingleRowCount, dstSingleRowCount);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::KvMulReduceScore(const LocalTensor<T>& kvLocal,
                                                                     const LocalTensor<T>& scoreLocal,
                                                                     const LocalTensor<T>& dstLocal,
                                                                     uint32_t tcDealSize, uint32_t dDealSize)
{
    MulReduceSumbaseVF(kvLocal, scoreLocal, dstLocal, coff_, cmpRatio_, dDealSize, tcDealSize);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CopyOutMidResToOutput(const LocalTensor<T>& kvLocal,
                                                                          const LocalTensor<T>& scoreLocal,
                                                                          const Vec1SliceInfo& sliceInfo,
                                                                          uint32_t compressTcSize, uint32_t dStartIdx,
                                                                          uint32_t dDealSize, uint32_t compressedCnt)
{
    if constexpr (COMP::xLayout == X_LAYOUT::BSH) {
        uint32_t bOutputScLen = CeilDivT(GetSeqLength(sliceInfo.bIdx), cmpRatio_);
        uint32_t bIdx = sliceInfo.bIdx;
        uint32_t sIdx = sliceInfo.sIdx;
        uint64_t ubOffset = 0;
        while (compressTcSize > 0) {
            uint32_t bStartPos = GetStartPos(bIdx);
            uint32_t preScSize = (sIdx + bStartPos) / cmpRatio_;
            uint32_t totalScSize = (GetSeqUsed(bIdx) + bStartPos) / cmpRatio_;
            if (preScSize < totalScSize) {
                uint32_t curScSize = min(compressTcSize, (totalScSize - preScSize));
                uint64_t padScIdx = bIdx * bOutputScLen + preScSize - (bStartPos / cmpRatio_);
                uint64_t outGmOffset = padScIdx * coff_ * cmpRatio_ * constInfo_.headDim + dStartIdx;
                // ubOffset：跨 batch 时（while 循环多次迭代）scoreLocal/kvLocal 的内容是
                // 按压缩块连续排布的（PadAlign 的 dstUbOffset = compressoredScCnt×cmpRatio 行），
                // 第二次迭代必须从 scoreLocal[ubOffset] 拷贝，否则会把第一个 slice 的内容
                // 重复输出（batch8 尾块内容串写到 batch9 块0 槽位，且末尾块丢失）。
                // 注意：与 CopyOutVec1ResToOutput 不同（那是 reduce 后每块 1 行），
                // 这里每块占 coff*cmpRatio 行，ubOffset 推进单位必须乘上行数。
                CopyOutWithSync(softmaxScoreOutGm_[outGmOffset], scoreLocal[ubOffset], curScSize * coff_ * cmpRatio_,
                                dDealSize, dDealSize, constInfo_.headDim);
                CopyOutWithSync(kvOutGm_[outGmOffset], kvLocal[ubOffset], curScSize * coff_ * cmpRatio_, dDealSize,
                                dDealSize, constInfo_.headDim);
                compressTcSize -= curScSize;
                ubOffset += curScSize * coff_ * cmpRatio_ * dDealSize;
            }
            bIdx++;
            sIdx = 0;
        }
    } else {
        uint64_t outGmOffset = compressedCnt * coff_ * cmpRatio_ * constInfo_.headDim + dStartIdx;
        CopyOutWithSync(softmaxScoreOutGm_[outGmOffset], scoreLocal, compressTcSize * coff_ * cmpRatio_, dDealSize,
                        dDealSize, constInfo_.headDim);
        CopyOutWithSync(kvOutGm_[outGmOffset], kvLocal, compressTcSize * coff_ * cmpRatio_, dDealSize, dDealSize,
                        constInfo_.headDim);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CopyOutVec1ResToOutput(const LocalTensor<T>& comperssoredUb,
                                                                           const Vec1SliceInfo& sliceInfo,
                                                                           uint32_t compressTcSize, uint32_t dStartIdx,
                                                                           uint32_t dDealSize)
{
    // bufid 同步：V 把结果 Cast 成输出类型写 staging，MTE3 再把 staging 搬到 GM
    Mutex::Lock<PIPE_V>(COMP_MTX_OUT_STAGE);
    LocalTensor<X_T> outputUb = outStageUb.template ReinterpretCast<X_T>();
    Cast(outputUb, comperssoredUb, RoundMode::CAST_ROUND, compressTcSize * dDealSize);
    Mutex::Unlock<PIPE_V>(COMP_MTX_OUT_STAGE);

    Mutex::Lock<PIPE_MTE3>(COMP_MTX_OUT_STAGE);
    if constexpr (COMP::xLayout == X_LAYOUT::BSH) {
        uint32_t bOutputScLen = CeilDivT(GetSeqLength(sliceInfo.bIdx), cmpRatio_);
        uint32_t bIdx = sliceInfo.bIdx;
        uint32_t sIdx = sliceInfo.sIdx;
        uint64_t ubOffset = 0;
        while (compressTcSize > 0) {
            uint32_t bStartPos = GetStartPos(bIdx);
            uint32_t preScSize = (sIdx + bStartPos) / cmpRatio_;
            uint32_t totalScSize = (GetSeqUsed(bIdx) + bStartPos) / cmpRatio_;
            if (preScSize < totalScSize) {
                uint32_t curScSize = min(compressTcSize, (totalScSize - preScSize));
                uint64_t padScIdx = bIdx * bOutputScLen + preScSize - (bStartPos / cmpRatio_);
                uint64_t outGmOffset = padScIdx * constInfo_.headDim + dStartIdx;
                DataCopyAlignUbToGm(cmpKvOutGm_[outGmOffset], outputUb[ubOffset], curScSize, dDealSize, dDealSize,
                                    constInfo_.headDim);
                compressTcSize -= curScSize;
                ubOffset += curScSize * dDealSize;
            }
            bIdx++;
            sIdx = 0;
        }
    } else {
        uint64_t outGmOffset = compressedCnt_ * constInfo_.headDim + dStartIdx;
        DataCopyAlignUbToGm(cmpKvOutGm_[outGmOffset], outputUb, compressTcSize, dDealSize, dDealSize,
                            constInfo_.headDim);
    }
    Mutex::Unlock<PIPE_MTE3>(COMP_MTX_OUT_STAGE);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcTilingStrategy(Vec1SplitInfo& splitInfo)
{
    // 计算headDim和Tc方向切分大小
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_64K / (cmpRatio_ * coff_ * sizeof(T));

    // 切块逻辑
    if (maxDealColNum < splitInfo.dBaseSize) {
        splitInfo.tcSplitSize = 1;
        splitInfo.dLoopCount = CeilDivT(splitInfo.dBaseSize, maxDealColNum);
        splitInfo.dSplitSize = splitInfo.dBaseSize / splitInfo.dLoopCount;
    } else {
        splitInfo.dSplitSize = splitInfo.dBaseSize;
        splitInfo.dLoopCount = splitInfo.dBaseSize / splitInfo.dSplitSize; // 此处常等于1，保留原逻辑
        splitInfo.tcSplitSize = maxDealColNum / splitInfo.dBaseSize;
    }
}

} // namespace Compressor
#endif // COMPRESSOR_BLOCK_VEC_H
