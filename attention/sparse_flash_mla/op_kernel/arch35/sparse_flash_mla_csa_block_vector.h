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
 * \file sparse_flash_mla_csa_block_vector.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
#define SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H

#include "util_regbase.h"
#include "sparse_flash_mla_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using AscendC::Reg::StoreDist;

#include "common/flash_decode.h"

#if __has_include("../../common/op_kernel/arch35/vf/vf_flash_decode.h")
#include "../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#else
#include "../common/arch35/vf/vf_flash_decode.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h")
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#else
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_flashupdate_new.h")
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#else
#include "../../common/arch35/vf/vf_flashupdate_new.h"
#endif

#if __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../common/buffers_policy.h"
#endif
#if __has_include("../../common/op_kernel/buffer_manager.h")
#include "../../common/op_kernel/buffer_manager.h"
#else
#include "../common/buffer_manager.h"
#endif
#if __has_include("../../common/op_kernel/buffer.h")
#include "../../common/op_kernel/buffer.h"
#else
#include "../common/buffer.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;
using AttentionCommon::FdRunInfo;

namespace SMLAKernel {
TEMPLATES_DEF
class CSABlockVec {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr float R0 = 1.0f;

    // ==================== Functions ======================
    __aicore__ inline CSABlockVec(){};
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
            }
            if (cuSeqlensCmpKv != nullptr) {
                cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            }
            if (seqUsedOriKV != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedOriKV);
            }
            if (seqUsedCmpKV != nullptr) {
                actualSeqLengthsCmpKVGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedCmpKV);
            }
            if (cmpResidualKV != nullptr) {
                cmpResidualKVGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKV);
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
        RunInfo &runInfo,
        ConstInfo &constInfo);
    __aicore__ inline void InitS2SplitStaging(
        Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &fdStaging)
    {
        fdStagingBase = fdStaging.template GetTensor<uint8_t>().GetPhyAddr(0);
        stagingOutGm = fdStaging.template GetTensor<float>();
    }
    __aicore__ inline void InitS2SplitStaging(
        Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &intraCoreCombine,
        Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &crossCoreCombine)
    {
        intraCoreCombineBase = intraCoreCombine.template GetTensor<uint8_t>().GetPhyAddr(0);
        intraCoreCombineGm = intraCoreCombine.template GetTensor<float>();
        crossCoreCombineBase = crossCoreCombine.template GetTensor<uint8_t>().GetPhyAddr(0);
        crossCoreCombineGm = crossCoreCombine.template GetTensor<float>();
        fdStagingBase = crossCoreCombineBase;
        stagingOutGm = crossCoreCombineGm;
    }
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo);
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo);

private:
    template <bool UPDATE>
    __aicore__ inline void ComputeVec1Softmax(LocalTensor<Q_T> &stage1CastTensor, LocalTensor<T> &mmRes,
        LocalTensor<float> &sumUb, LocalTensor<float> &maxUb, LocalTensor<T> &apiTmpBuffer,
        RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void InitVec1SoftmaxFromSinks(
        LocalTensor<float> &sumUb, LocalTensor<float> &maxUb, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyVec1ResultToL1(
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        LocalTensor<Q_T> &stage1CastTensor, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void StageCrossCoreVec1Lse(
        LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void StageBatchConsistencyVec1Lse(
        LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void StageLegacyVec1Lse(
        LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutVec1Lse(
        LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline uint32_t GetStagingSlotNum(bool isInner = false) const
    {
        if constexpr (IS_BATCH_CONSISTENCY) {
            if (isInner) {
                if constexpr (IS_SPLIT_G) {
                    return GetBlockNum();
                }
                return GetBlockNum() << 1U;
            }
            if constexpr (IS_SPLIT_G) {
                return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * (GetBlockNum() >> 1U);
            }
            return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * GetBlockNum();
        }
        if constexpr (IS_SPLIT_G) {
            return AttentionCommon::FD_MAX_S2_SPLIT_NUM * (GetBlockNum() >> 1U);
        } else {
            return AttentionCommon::FD_MAX_S2_SPLIT_NUM * GetBlockNum();
        }
    }

    __aicore__ inline uint32_t GetIntraCoreWorkspaceIdx(
        const RunInfo &runInfo, const ConstInfo &constInfo) const
    {
        uint32_t coreIdx;
        if constexpr (IS_SPLIT_G) {
            coreIdx = static_cast<uint32_t>(constInfo.aivIdx >> 2U);
        } else {
            coreIdx = static_cast<uint32_t>(constInfo.aivIdx >> 1U);
        }
        return (coreIdx << 1U) + runInfo.multiCoreIdxMod2;
    }

    __aicore__ inline uint32_t GetCrossCoreWorkspaceIdx(const RunInfo &runInfo) const
    {
        return static_cast<uint32_t>(runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx);
    }

    __aicore__ inline int64_t GetFaStagingMOffset(const RunInfo &runInfo, const ConstInfo &constInfo) const
    {
        int64_t stagingMOffset = (constInfo.subBlockIdx == 1) ? static_cast<int64_t>(runInfo.firstHalfMRealSize) : 0L;
        if constexpr (IS_SPLIT_G) {
            stagingMOffset += static_cast<int64_t>(runInfo.goIdx);
        }
        return stagingMOffset;
    }

    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                           Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
                                           const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
                                              int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyToOutUb(LocalTensor<Q_T> kvNzUb, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                       ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                          LocalTensor<Q_T> kvNzOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                          ConstInfo &constInfo);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                           int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                               LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &constInfo);
    __aicore__ inline void ReduceIntraBlockAndStage(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<T> &vec2ResUb, LocalTensor<T> &partialTmpUb);

    TPipe *tPipe;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> oriSparseIndicesGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKVGm;
    GlobalTensor<int32_t> cmpResidualKVGm;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TQue<QuePosition::VECIN, 2> stage0InQue;      // for v0 input, 2份表示可能存在pingpong
    TQue<QuePosition::VECOUT, 1> stage0OutQue;    // for v0 output, 2份表示可能存在pingpong
    TBuf<> stage0OutBuf[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2];     // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2];     // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TEventID mte3ToVAttnOutId; // 存放MTE3_V的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID vToMte3AttnOutId; // 存放V_MTE3的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID stageMte3ToVId;   // staging 专用 MTE3_V event
    TEventID mte3ToVLseOutId;  // 存放MTE3_V的eventId, 用于V1 LSE拷出阶段的同步
    TEventID vToMte3LseOutId;  // 存放V_MTE3的eventId, 用于V1 LSE拷出阶段的同步
    TEventID mte2ToMte3[2];
    TEventID mte3ToMte2[2];
    TEventID intraLseMte3ToMte2Id[2];
    TEventID intraAttnOutMte3ToMte2Id[2];
    TEventID intraPartialOVToMte2Id;
    TEventID reduceMaxSumVToMte2Id;
    TEventID reduceMte2ToVId;
    // Flash-decode reduction uses dedicated events because it may process
    // multiple bounded chunks and must not alias the Vec0 pipeline events.
    TEventID fdVToMte2Id[2];
    TEventID fdMte2ToVId;
    TEventID fdMte3ToVId;
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxFinalMaxBuf[2];
    TBuf<> softmaxFinalSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> batchReduceTmpBuf;
    TBuf<> outLseBuf[2];
    TBuf<> vselrIndexesBuf[2];
    AttentionCommon::FdBuffers<TBuf<>> fdBuffers;
    __gm__ uint8_t *fdStagingBase = nullptr;
    GlobalTensor<float> stagingOutGm;
    __gm__ uint8_t *intraCoreCombineBase = nullptr;
    GlobalTensor<float> intraCoreCombineGm;
    __gm__ uint8_t *crossCoreCombineBase = nullptr;
    GlobalTensor<float> crossCoreCombineGm;
    TEventID mte2ToVV0Id[2];
    TEventID vToMte2V0Id[2];

    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;

    TEventID initOutputEventId; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
                                                                   int64_t s2IdxInBase, const RunInfo &runInfo,
                                                                   ConstInfo &constInfo)
{
    int64_t sparseBlockCount = 0;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    // CSA、ORI_SPARSE、ORI_CMP_SPARSE均可通过runInfo.isCmp判断
    if (runInfo.isCmp) {
        sparseBlockCount = constInfo.cmpSparseBlockCount;
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
    } else {
        sparseBlockCount = constInfo.oriSparseBlockCount;
    }

    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx +=
            runInfo.boIdx * constInfo.s1Size * sparseBlockCount + runInfo.s1oIdx * sparseBlockCount; // B, S1, N2(1), K
    }

    int64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    if (unlikely(topkKIdx >= sparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx + runInfo.s2StartIdx);
    }
    topkKIdx += 1;
    if (unlikely((topkKIdx >= sparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        token1Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx + runInfo.s2StartIdx);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t CSABlockVec<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo,
                                                                   ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKeyStride0 : constInfo.oriKeyStride0;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * paBlockStride +
                        blkTableOffset * constInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        if (runInfo.isCmp) {
            realkeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSize +
                            runInfo.n2oIdx * constInfo.cmpS2Size * constInfo.dSize + s2Idx * constInfo.dSize; // BSN(1)D
        } else {
            realkeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.s2Size * constInfo.dSize +
                            runInfo.n2oIdx * constInfo.s2Size * constInfo.dSize + s2Idx * constInfo.dSize; // BSN(1)D
        }
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        realkeyOffset = (cuSeqlensKvGm.GetValue(runInfo.boIdx) + s2Idx) * constInfo.n2Size * constInfo.dSize +
                        runInfo.n2oIdx * constInfo.dSize; // TN(1)D
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                  int64_t keyOffset, ConstInfo &constInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = constInfo.dSize * sizeof(KV_T);

    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding =
        (CeilAlign(constInfo.dSize * sizeof(KV_T), BUFFER_SIZE_BYTE_32B) - constInfo.dSize * sizeof(KV_T)) /
        sizeof(KV_T);
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * constInfo.dSize], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t CSABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                      int64_t token0Idx, int64_t token1Idx,
                                                                      const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    int64_t combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
    int64_t keySrcStride =
        (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) * sizeof(KV_T) - combineBytes;
    if (unlikely(keyOffset1 < 0)) {
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
    } else if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, constInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = combineBytes;
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) - combineBytes) / sizeof(KV_T);
        padParams.paddingValue = 0;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }
        DataCopyPad(kvInUb[startRow * constInfo.dSize], keyGm[keyOffset], intriParams, padParams);
    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyToOutUb(LocalTensor<Q_T> kvOutUb, LocalTensor<KV_T> srcTensor,
                                                               int64_t dealRow, ConstInfo &constInfo)
{
    LocalTensor<Q_T> kvNdUb = srcTensor.template ReinterpretCast<Q_T>();
    DataCopy(kvOutUb, kvNdUb, dealRow * constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                           LocalTensor<Q_T> kvNzOutUb, int64_t dealRow, int64_t s2StartIdx,
                                           const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * 16], kvNzOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                           const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * constInfo.dSize], kvOutUb, dealRow * constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        int32_t vecCnt = (aicIdx % 2U == 0) ? (GetSubBlockIdx() == 0 ? 0 : 1) : (GetSubBlockIdx() == 0 ? 2 : 3);
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = 0;
            } else {
                sparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2);
            }
        }
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        sparseCalSize = GetSubBlockIdx() == 0 ? v0S2SizeFirstCore : runInfo.s2RealSize - v0S2SizeFirstCore;
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : v0S2SizeFirstCore;
        sparseS2End = sparseS2Start + sparseCalSize;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                        const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) {
            if constexpr (IS_SPLIT_G) {
                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
            }
            return;
        }
        keyGm = cmpKVGm;
        cuSeqlensKvGm = cuSeqlensCmpKvGm;
        sparseIndicesGm = cmpSparseIndicesGm;
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            blockTableGm = cmpBlockTableGm;
            blockSize = constInfo.cmpBlockSize;
            maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        }
        CalSparseCalSize(runInfo, constInfo);
        ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo, startPos);
        if constexpr (IS_SPLIT_G) {
            CrossCoreSetFlag<0, PIPE_MTE3>(15);
            CrossCoreWaitFlag<0, PIPE_MTE3>(15);
        }
        outputL1.SetCrossCore();
        v0ResGm.SetCrossCore();
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!runInfo.isCmp) {
            keyGm = oriKVGm;
            cuSeqlensKvGm = cuSeqlensOriKvGm;
            sparseIndicesGm = oriSparseIndicesGm;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                blockTableGm = oriBlockTableGm;
                blockSize = constInfo.oriBlockSize;
                maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
            }
        } else {
            keyGm = cmpKVGm;
            cuSeqlensKvGm = cuSeqlensCmpKvGm;
            sparseIndicesGm = cmpSparseIndicesGm;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                blockTableGm = cmpBlockTableGm;
                blockSize = constInfo.cmpBlockSize;
                maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
            }
        }
        CalSparseCalSize(runInfo, constInfo);
        ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo, startPos);
        if constexpr (IS_SPLIT_G) {
            CrossCoreSetFlag<0, PIPE_MTE3>(15);
            CrossCoreWaitFlag<0, PIPE_MTE3>(15);
        }
        outputL1.SetCrossCore();
        v0ResGm.SetCrossCore();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                            Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                            const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (sparseCalSize == 0) {
            return;
        }
        bool meetEnd = false;
        int64_t s2Start = sparseS2Start;
        int64_t s2 = sparseS2Start;
        int64_t token0Idx;
        int64_t token1Idx;
        uint32_t pingPong = 0;
        while ((s2 < sparseS2End) && !meetEnd) {
            int64_t dealRow = 0;
            LocalTensor<Q_T> stage0OutUb = this->stage0OutBuf[pingPong].template Get<Q_T>();
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            while (dealRow < Min(16, sparseCalSize) && s2 < sparseS2End) {
                GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
                s2 += 2;
                if (token0Idx == -1 && token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
                dealRow += CopyInKvSparse(stage0OutUb, dealRow, token0Idx, token1Idx, runInfo, constInfo);
                if (token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
            }
            if (dealRow == 0) {
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
                pingPong ^= 1;
                return;
            }
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
            CopyOutKvUb2Gm(v0ResGm, stage0OutUb, dealRow, s2Start, runInfo, constInfo);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            s2Start += dealRow;
            pingPong ^= 1;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool UPDATE>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ComputeVec1Softmax(
    LocalTensor<Q_T> &stage1CastTensor, LocalTensor<T> &mmRes, LocalTensor<float> &sumUb,
    LocalTensor<float> &maxUb, LocalTensor<T> &apiTmpBuffer, RunInfo &runInfo, ConstInfo &constInfo)
{
    if (likely(runInfo.s2RealSize == 128 && runInfo.s2RealSizeUpdate == 128)) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf,
            runInfo.halfMRealSize, runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale),
            negativeFloatScalar);
    } else if (runInfo.s2RealSize <= 64) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize,
            FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf,
            runInfo.halfMRealSize, runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale),
            negativeFloatScalar);
    } else if (runInfo.s2RealSize < 128 || runInfo.s2RealSizeUpdate < 128) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize,
            FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf,
            runInfo.halfMRealSize, runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale),
            negativeFloatScalar);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitVec1SoftmaxFromSinks(
    LocalTensor<float> &sumUb, LocalTensor<float> &maxUb, RunInfo &runInfo, ConstInfo &constInfo)
{
    bool includeSink = (!runInfo.isCrossCoreSplit) || runInfo.isFirstS2SplitCore;
    if constexpr (IS_BATCH_CONSISTENCY) {
        includeSink = includeSink && (runInfo.reduceBlockId == 0);
    }
    if (!includeSink) {
        Duplicate(maxUb, this->negativeFloatScalar, runInfo.halfMRealSize);
        Duplicate(sumUb, static_cast<T>(0), runInfo.halfMRealSize);
        return;
    }
    int64_t sinksOffset = 0;
    if constexpr (!IS_SPLIT_G) {
        sinksOffset = GetBlockIdx() % 2 == 0 ? 0 : runInfo.firstHalfMRealSize;
    } else {
        sinksOffset = runInfo.goIdx;
        if (constInfo.subBlockIdx == 1) {
            sinksOffset += runInfo.firstHalfMRealSize;
        }
    }
    LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
    InitSoftmaxFromSinks<T>(sumUb, maxUb, sinksUb, sinksOffset, R0, runInfo.halfMRealSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyVec1ResultToL1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    LocalTensor<Q_T> &stage1CastTensor, RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t stage1Offset = runInfo.taskIdMod2;
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
            (runInfo.mRealSize - runInfo.halfMRealSize)], stage1CastTensor,
            {s2BaseSize / 16, static_cast<uint16_t>(runInfo.halfMRealSize),
            static_cast<uint16_t>(vec1Srcstride - runInfo.halfMRealSize),
            static_cast<uint16_t>(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::StageCrossCoreVec1Lse(
    LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo, ConstInfo &constInfo)
{
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize, dTemplateAlign64,
        GetStagingSlotNum(false), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    LocalTensor<float> tmpUb = this->batchReduceTmpBuf.template Get<float>();
    AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase, GetCrossCoreWorkspaceIdx(runInfo),
        GetFaStagingMOffset(runInfo, constInfo), runInfo.halfMRealSize, maxUb, sumUb, tmpUb,
        vToMte3AttnOutId, stageMte3ToVId);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::StageBatchConsistencyVec1Lse(
    LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo, ConstInfo &constInfo)
{
    if (!runInfo.isLastBase) {
        return;
    }
    if (runInfo.halfMRealSize > 0) {
        LocalTensor<float> finalMaxUb =
            this->softmaxFinalMaxBuf[runInfo.taskIdMod2].template Get<float>();
        LocalTensor<float> finalSumUb =
            this->softmaxFinalSumBuf[runInfo.taskIdMod2].template Get<float>();
        uint64_t snapshotElems = Align8Func(runInfo.halfMRealSize);
        DataCopy(finalMaxUb, maxUb, snapshotElems);
        DataCopy(finalSumUb, sumUb, snapshotElems);
    }
    if (runInfo.isCrossCoreSplit && !runInfo.isFirstS2SplitCore) {
        StageCrossCoreVec1Lse(maxUb, sumUb, runInfo, constInfo);
    } else if (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
        runInfo.s2LoopCount < runInfo.s2LoopLimit) {
        AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize, dTemplateAlign64,
            GetStagingSlotNum(true), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
            AttentionCommon::FD_REDUCE_CHUNK_ROWS};
        LocalTensor<float> tmpUb = this->batchReduceTmpBuf.template Get<float>();
        AttentionCommon::StageVec1Lse(stagingLayout, intraCoreCombineBase,
            GetIntraCoreWorkspaceIdx(runInfo, constInfo), GetFaStagingMOffset(runInfo, constInfo),
            runInfo.halfMRealSize, maxUb, sumUb, tmpUb, vToMte3AttnOutId, stageMte3ToVId);
        SetFlag<HardEvent::MTE3_MTE2>(intraLseMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
    } else if (runInfo.isCrossCoreSplit && runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0) {
        StageCrossCoreVec1Lse(maxUb, sumUb, runInfo, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::StageLegacyVec1Lse(
    LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo, ConstInfo &constInfo)
{
    if (!runInfo.isCrossCoreSplit || runInfo.halfMRealSize <= 0 ||
        runInfo.s2LoopCount != runInfo.s2LoopLimit) {
        return;
    }
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize, dTemplateAlign64,
        GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    LocalTensor<float> tmpUb = this->stage2OutBuf.template Get<float>();
    AttentionCommon::StageVec1Lse(stagingLayout, fdStagingBase, GetCrossCoreWorkspaceIdx(runInfo),
        GetFaStagingMOffset(runInfo, constInfo), static_cast<uint32_t>(runInfo.halfMRealSize),
        maxUb, sumUb, tmpUb, vToMte3AttnOutId, stageMte3ToVId);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutVec1Lse(
    LocalTensor<float> &maxUb, LocalTensor<float> &sumUb, RunInfo &runInfo, ConstInfo &constInfo)
{
    bool copyOutLse = constInfo.returnSoftmaxLse && runInfo.halfMRealSize > 0 &&
        runInfo.s2LoopCount == runInfo.s2LoopLimit;
    if constexpr (IS_BATCH_CONSISTENCY) {
        copyOutLse = copyOutLse && !runInfo.isCrossCoreSplit && !runInfo.needReduce;
    }
    if (!copyOutLse) {
        return;
    }
    LocalTensor<float> outLse = this->outLseBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = sizeof(float) * runInfo.halfMRealSize;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    ComputeLse<float>(outLse, sumUb, maxUb, runInfo.halfMRealSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], outLse, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                        RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    runInfo.s2RealSizeUpdate = runInfo.s2RealSize;

    bool isFirstSoftmaxBase = runInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        isFirstSoftmaxBase = runInfo.isFirstBase;
    }
    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (isFirstSoftmaxBase && !isSinks) {
        ComputeVec1Softmax<false>(stage1CastTensor, mmRes, sumUb, maxUb, apiTmpBuffer, runInfo, constInfo);
    } else {
        if (isFirstSoftmaxBase && isSinks) {
            InitVec1SoftmaxFromSinks(sumUb, maxUb, runInfo, constInfo);
        }
        ComputeVec1Softmax<true>(stage1CastTensor, mmRes, sumUb, maxUb, apiTmpBuffer, runInfo, constInfo);
    }
    bmm1ResBuf.SetCrossCore();
    CopyVec1ResultToL1(outputBuf, stage1CastTensor, runInfo, constInfo);
    if (!isFirstSoftmaxBase || isSinks) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if constexpr (IS_BATCH_CONSISTENCY) {
        StageBatchConsistencyVec1Lse(maxUb, sumUb, runInfo, constInfo);
    } else {
        StageLegacyVec1Lse(maxUb, sumUb, runInfo, constInfo);
    }
    CopyOutVec1Lse(maxUb, sumUb, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ReduceIntraBlockAndStage(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<T> &vec2ResUb, LocalTensor<T> &partialTmpUb)
{
    AttentionCommon::S2SplitFdStagingLayout intraLayout = {constInfo.gSize, dTemplateAlign64,
        GetStagingSlotNum(true), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    AttentionCommon::S2SplitFdStagingLayout crossLayout = {constInfo.gSize, dTemplateAlign64,
        GetStagingSlotNum(false), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    uint32_t intraWorkspaceIdx = GetIntraCoreWorkspaceIdx(runInfo, constInfo);
    uint32_t crossWorkspaceIdx = static_cast<uint32_t>(
        runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx - runInfo.reduceBlockId);
    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
    LocalTensor<float> tmpUb = this->batchReduceTmpBuf.template Get<float>();
    LocalTensor<float> blockMaxUb = tmpUb;
    LocalTensor<float> blockSumUb = tmpUb[256];
    LocalTensor<float> lseBroadcastUb = tmpUb[512];
    LocalTensor<float> sumBroadcastUb = tmpUb[640];
    LocalTensor<float> maxUb =
        this->softmaxFinalMaxBuf[runInfo.taskIdMod2].template Get<float>();
    LocalTensor<float> sumUb =
        this->softmaxFinalSumBuf[runInfo.taskIdMod2].template Get<float>();
    bool copyOutMergedLse = constInfo.returnSoftmaxLse && !runInfo.isCrossCoreSplit &&
        runInfo.s2LoopCount == runInfo.s2LoopLimit;

    WaitFlag<HardEvent::MTE3_MTE2>(intraLseMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
    WaitFlag<HardEvent::MTE3_MTE2>(intraAttnOutMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
    int64_t startRow = 0;
    while (startRow < runInfo.vec2MRealSize) {
        int64_t dealRowCount = intraLayout.chunkRows;
        if (startRow + dealRowCount > runInfo.vec2MRealSize) {
            dealRowCount = runInfo.vec2MRealSize - startRow;
        }
        LocalTensor<T> chunkCurrent = vec2ResUb[startRow * dTemplateAlign64];
        LocalTensor<float> chunkMaxUb = maxUb[startRow];
        LocalTensor<float> chunkSumUb = sumUb[startRow];
        if (copyOutMergedLse) {
            WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        }
        AttentionCommon::MergeStagedAndCurrentChunk<T, dTemplateAlign64>(intraLayout,
            intraCoreCombineBase, intraWorkspaceIdx, stagingMOffset + startRow,
            dealRowCount, static_cast<int64_t>(constInfo.dSizeV), chunkMaxUb, chunkSumUb,
            chunkCurrent, blockMaxUb, blockSumUb, partialTmpUb, lseBroadcastUb, sumBroadcastUb,
            reduceMaxSumVToMte2Id, intraPartialOVToMte2Id, reduceMte2ToVId);

        AttentionCommon::StageBroadcastMaxSum(intraLayout, intraCoreCombineBase, intraWorkspaceIdx,
            stagingMOffset + startRow, dealRowCount, lseBroadcastUb, sumBroadcastUb,
            vToMte3AttnOutId, stageMte3ToVId);
        if (copyOutMergedLse) {
            DataCopyExtParams lseParams;
            lseParams.blockCount = static_cast<uint16_t>(dealRowCount);
            lseParams.blockLen = sizeof(float);
            lseParams.srcStride = 0;
            lseParams.dstStride = 0;
            SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
            WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
            DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset + startRow],
                lseBroadcastUb, lseParams);
            SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        }
        if (runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
            AttentionCommon::StageBroadcastMaxSum(crossLayout, crossCoreCombineBase, crossWorkspaceIdx,
                stagingMOffset + startRow, dealRowCount, lseBroadcastUb, sumBroadcastUb,
                vToMte3AttnOutId, stageMte3ToVId);
        }
        startRow += intraLayout.chunkRows;
    }

    AttentionCommon::StageVec2PartialOAndWait<T>(intraLayout, intraCoreCombineGm, intraWorkspaceIdx,
        stagingMOffset, runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb,
        vToMte3AttnOutId, stageMte3ToVId);
    if (runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        AttentionCommon::StageVec2PartialOAndWait<T>(crossLayout, crossCoreCombineGm, crossWorkspaceIdx,
            stagingMOffset, runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb,
            vToMte3AttnOutId, stageMte3ToVId);
    }
    if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE3_MTE2>(intraLseMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
        SetFlag<HardEvent::MTE3_MTE2>(intraAttnOutMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    bool needIntraBlockReduce = false;
    if constexpr (IS_BATCH_CONSISTENCY) {
        needIntraBlockReduce =
            runInfo.isLastBase && runInfo.isFirstS2SplitCore && runInfo.reduceBlockId > 0;
        if (needIntraBlockReduce) {
            WaitFlag<HardEvent::V_MTE2>(intraPartialOVToMte2Id);
            WaitFlag<HardEvent::V_MTE2>(reduceMaxSumVToMte2Id);
        }
    }
    bool isFirstVec2Base = runInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        isFirstVec2Base = runInfo.isFirstBase;
    }
    if (unlikely(isFirstVec2Base)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        if (runInfo.s2RealSizeUpdate > 0) {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
            bool isLastVec2Base = (runInfo.s2LoopCount == runInfo.s2LoopLimit);
            if constexpr (IS_BATCH_CONSISTENCY) {
                isLastVec2Base = runInfo.isLastBase;
            }
            if (isLastVec2Base) {
                LocalTensor<float> sumUb;
                if constexpr (IS_BATCH_CONSISTENCY) {
                    sumUb = this->softmaxFinalSumBuf[runInfo.taskIdMod2].template Get<float>();
                } else {
                    sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                }
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb,
                    runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb,
                    runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            }
        } else {
            bool isLastVec2Base = runInfo.s2LoopCount >= runInfo.s2LoopLimit;
            if constexpr (IS_BATCH_CONSISTENCY) {
                isLastVec2Base = runInfo.isLastBase;
            }
            if (isLastVec2Base) {
                LocalTensor<float> sumUb;
                if constexpr (IS_BATCH_CONSISTENCY) {
                    sumUb = this->softmaxFinalSumBuf[runInfo.taskIdMod2].template Get<float>();
                } else {
                    sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                }
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
        }
    }

    if constexpr (IS_BATCH_CONSISTENCY) {
        if (runInfo.isLastBase) {
            if (unlikely(isFirstVec2Base)) {
                LocalTensor<float> sumUb =
                    this->softmaxFinalSumBuf[runInfo.taskIdMod2].template Get<float>();
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
            if (needIntraBlockReduce) {
                SetFlag<HardEvent::V_MTE2>(intraPartialOVToMte2Id);
                SetFlag<HardEvent::V_MTE2>(reduceMaxSumVToMte2Id);
                ReduceIntraBlockAndStage(runInfo, constInfo, vec2ResUb, mmRes);
                if (!runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
                }
            } else {
                if (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                    runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
                        dTemplateAlign64, GetStagingSlotNum(true), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(stagingLayout, intraCoreCombineGm,
                        GetIntraCoreWorkspaceIdx(runInfo, constInfo), stagingMOffset, runInfo.vec2MRealSize,
                        static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb,
                        vToMte3AttnOutId, stageMte3ToVId);
                    SetFlag<HardEvent::MTE3_MTE2>(intraAttnOutMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
                }
                if (runInfo.isCrossCoreSplit && (!runInfo.isFirstS2SplitCore ||
                    (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                    runInfo.s2LoopCount == runInfo.s2LoopLimit))) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
                        dTemplateAlign64, GetStagingSlotNum(false), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(stagingLayout, crossCoreCombineGm,
                        workspaceIdx, stagingMOffset, runInfo.vec2MRealSize,
                        static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb,
                        vToMte3AttnOutId, stageMte3ToVId);
                } else if (!runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
                }
            }
        }
    } else if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize,
                                                                  dTemplateAlign64, 1.0);
        }
        if (runInfo.isCrossCoreSplit) {
            AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                AttentionCommon::FD_REDUCE_CHUNK_ROWS};
            uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
            int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
            AttentionCommon::StageVec2PartialO<T>(
                stagingLayout, stagingOutGm, workspaceIdx, stagingMOffset, static_cast<uint32_t>(runInfo.vec2MRealSize),
                static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, vToMte3AttnOutId, stageMte3ToVId);
        } else {
            this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
        }
    }
    bmm2ResBuf.SetCrossCore();
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitFDBuffers(FdRunInfo &fdRunInfo)
{
    FdRunInfo fdBufferInfo = fdRunInfo;
    if (fdBufferInfo.mNum > AttentionCommon::FD_REDUCE_CHUNK_ROWS) {
        fdBufferInfo.mNum = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
    }
    AttentionCommon::InitFDBuffers<T, dTemplateAlign64>(fdBufferInfo, this->tPipe, fdBuffers);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo)
{
    InitFDBuffers(fdRunInfo);
    int64_t seqOffset = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        seqOffset = this->cuSeqlensQGm.GetValue(fdRunInfo.bn2Idx);
    } else {
        seqOffset = fdRunInfo.bn2Idx * constInfo.s1Size;
    }
    int64_t attentionOutOffset = seqOffset * constInfo.n2GDv +
        fdRunInfo.mIdx * constInfo.n2GDv +
        fdRunInfo.mStartIdx * constInfo.dSizeV;
    int64_t softmaxLseOffset = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        softmaxLseOffset = (seqOffset + fdRunInfo.mIdx) * constInfo.gSize + fdRunInfo.mStartIdx;
    } else {
        softmaxLseOffset = (fdRunInfo.bn2Idx * constInfo.s1Size + fdRunInfo.mIdx) *
            constInfo.gSize + fdRunInfo.mStartIdx;
    }
    LocalTensor<T> accumulatedO = this->fdBuffers.accumOut.template Get<T>();
    LocalTensor<float> lseExpUb = this->fdBuffers.lseExp.template Get<float>();
    LocalTensor<float> blockMaxUb = this->fdBuffers.blockMax.template Get<float>();
    LocalTensor<float> blockSumUb = this->fdBuffers.blockSum.template Get<float>();
    LocalTensor<T> partialOFp32 = this->fdBuffers.partialO.template Get<T>();
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
        dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    int64_t attentionOutRowStride = static_cast<int64_t>(constInfo.dSizeV) +
        static_cast<int64_t>(constInfo.attentionOutStride) / sizeof(OUTPUT_T);
    int64_t startRow = 0;
    while (startRow < fdRunInfo.mNum) {
        int64_t dealRowCount = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
        if (startRow + dealRowCount > fdRunInfo.mNum) {
            dealRowCount = fdRunInfo.mNum - startRow;
        }
        WaitFlag<HardEvent::MTE3_V>(fdMte3ToVId);
        AttentionCommon::ReduceWithLse<T, dTemplateAlign64>(stagingLayout, fdStagingBase,
            fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
            static_cast<uint32_t>(fdRunInfo.mStartIdx + startRow), dealRowCount,
            static_cast<uint32_t>(constInfo.dSizeV),
            accumulatedO, lseExpUb, blockMaxUb, blockSumUb, partialOFp32,
            constInfo.returnSoftmaxLse, softmaxLseGm, softmaxLseOffset + startRow,
            fdVToMte2Id[0], fdVToMte2Id[1], fdMte2ToVId,
            vToMte3LseOutId, mte3ToVLseOutId);
        RunInfo runInfo;
        runInfo.vec2MRealSize = dealRowCount;
        runInfo.attentionOutOffset = attentionOutOffset + startRow * attentionOutRowStride;
        int64_t vec2CalcSize = dealRowCount * dTemplateAlign64;
        this->CopyOutAttentionOut(runInfo, constInfo, accumulatedO, 0, vec2CalcSize);
        SetFlag<HardEvent::MTE3_V>(fdMte3ToVId);
        startRow += dealRowCount;
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                   LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                   int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                       LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                       int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    if (constInfo.returnSoftmaxLse) {
        uint64_t totalReturnSoftmaxSize = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
            totalReturnSoftmaxSize = constInfo.bSize * constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            totalReturnSoftmaxSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize; // (N2,T1,G)
        }
        if (coreNum != 0 && totalReturnSoftmaxSize > 0) {
            uint64_t singleCoreSoftmaxSize = (totalReturnSoftmaxSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
            uint64_t tailSoftmaxSize = totalReturnSoftmaxSize - constInfo.aivIdx * singleCoreSoftmaxSize;
            uint64_t singleInitSoftmaxSize =
                tailSoftmaxSize < singleCoreSoftmaxSize ? tailSoftmaxSize : singleCoreSoftmaxSize;
            if (constInfo.aivIdx * singleCoreSoftmaxSize < totalReturnSoftmaxSize && singleInitSoftmaxSize > 0) {
                WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
                matmul::InitOutput<float>(this->softmaxLseGm[constInfo.aivIdx * singleCoreSoftmaxSize],
                                          singleInitSoftmaxSize, 0);
                SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            }
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                               ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxLseGm.SetGlobalBuffer((__gm__ T *)softmaxLse);
        if (constInfo.needInit == 1) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore(constInfo);
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
        }
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)oriSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    if constexpr (IS_BATCH_CONSISTENCY) {
        tPipe->InitBuffer(softmaxFinalSumBuf[0], softmaxBufSize);
        tPipe->InitBuffer(softmaxFinalSumBuf[1], softmaxBufSize);
        tPipe->InitBuffer(softmaxFinalMaxBuf[0], softmaxBufSize);
        tPipe->InitBuffer(softmaxFinalMaxBuf[1], softmaxBufSize);
        tPipe->InitBuffer(batchReduceTmpBuf, 768 * sizeof(float));
    }
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo &constInfo)
{
    LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
    TEventID mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    // ub buffer
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B
    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        tPipe->InitBuffer(stage0OutBuf[0], dVTemplateType * 16 * sizeof(KV_T));
        tPipe->InitBuffer(stage0OutBuf[1], dVTemplateType * 16 * sizeof(KV_T));
    }
    if (constInfo.returnSoftmaxLse) {
        tPipe->InitBuffer(outLseBuf[0], 256);
        tPipe->InitBuffer(outLseBuf[1], 256);
    }

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVAttnOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    stageMte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    vToMte3AttnOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    mte3ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    mte3ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
    mte2ToMte3[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToMte3[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    fdVToMte2Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    fdVToMte2Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    fdMte2ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    fdMte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::V_MTE2>(fdVToMte2Id[0]);
    SetFlag<HardEvent::V_MTE2>(fdVToMte2Id[1]);
    SetFlag<HardEvent::MTE3_V>(fdMte3ToVId);
    if constexpr (IS_BATCH_CONSISTENCY) {
        for (uint32_t eventIdx = 0; eventIdx < 2; ++eventIdx) {
            intraLseMte3ToMte2Id[eventIdx] =
                GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            intraAttnOutMte3ToMte2Id[eventIdx] =
                GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        }
        intraPartialOVToMte2Id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        reduceMaxSumVToMte2Id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        reduceMte2ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::V_MTE2>(intraPartialOVToMte2Id);
        SetFlag<HardEvent::V_MTE2>(reduceMaxSumVToMte2Id);
    }
    mte2ToVV0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte2ToVV0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    vToMte2V0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    vToMte2V0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[0]);
    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[1]);
    initOutputEventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetExtremeValue(T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF
class CSABlockVecDummy {
public:
    __aicore__ inline CSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo)
    {
    }
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
        __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
        __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
        __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ,
        __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV) {};
    __aicore__ inline void InitS2SplitStaging(
        Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &fdStaging) {}
    __aicore__ inline void InitS2SplitStaging(
        Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &intraCoreCombine,
        Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &crossCoreCombine) {}
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo) {}
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo) {}
};
} // namespace SMLAKernel
#endif // SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
