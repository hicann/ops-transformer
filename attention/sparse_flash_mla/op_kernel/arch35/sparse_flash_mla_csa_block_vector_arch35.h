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
 * \file sparse_flash_mla_csa_block_vector_arch35.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH35_H
#define SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH35_H

#include "util_regbase.h"
#include "sparse_flash_mla_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using AscendC::Reg::StoreDist;

#include "common/flash_decode.h"
#include "common/get_kv_phy_addr_vf.h"
#include "common/smla_vector_common_arch35.h"

#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/buffers_policy.h"
#include "../../../common/op_kernel/attn_buffer_manager.h"
#include "../../../common/op_kernel/attn_buffer.h"
#include "../../../common/op_kernel/init_output.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;
using namespace fa_base_matmul;
using AttentionCommon::FdRunInfo;

namespace SMLAKernel {

// 统一窗口公式
struct PhyAddrValidInfo {
    static constexpr int64_t BIAS_UNBOUND = 0x7FFFFFFF; // INT32_MAX
    int64_t oriLeftBias = BIAS_UNBOUND;
    int64_t oriRightBias = BIAS_UNBOUND;
    int32_t oriS2Act = 0;
    bool oriTopkMode = false; // oriMaskMode==0: 走topkLength语义(保持原行为)
    bool cmpTopkMode = true;  // cmpMaskMode==0: 走topkLength语义(保持原行为)
    int64_t cmpBase = 0;      // restoredSize - actualS1Size + 1
};

TEMPLATES_DEF
class SmlaCsaBlockVector {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    // StageVec1Lse uses two 1 KiB broadcast blocks for max and sum.
    static constexpr uint32_t FD_VEC1_MAX_ROWS = 32U;
    static constexpr uint32_t FD_VEC1_BROADCAST_BLOCK_ELEMS =
        FD_VEC1_MAX_ROWS * AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW;
    static constexpr uint32_t FD_VEC1_LSE_TMP_ELEMS = 2U * FD_VEC1_BROADCAST_BLOCK_ELEMS;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr float R0 = 1.0f;
    static constexpr uint32_t initOutputEventId =
        INNERCORE_INITOUT_MTE3_V; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
    // Sparse KV搬入/拷出块大小：每次搬入8行，每16行做一次拷出
    static constexpr int64_t KV_COPYIN_UNIT = 8;   // 每次搬入8行
    static constexpr int64_t KV_PROCESS_UNIT = 16; // 每次拷出16行

    // ==================== Functions ======================
    __aicore__ inline SmlaCsaBlockVector(){};
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV)
    {
        if ASCEND_IS_AIV {
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
    __aicore__ inline void InitLocalBuffer(ConstInfo &smla35VecConstInfo, uint32_t ubBaseAddr);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                       ConstInfo &smla35VecConstInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &smla35VecConstInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void ProcessVec1(StaticBuffer<Q_T> &outputBuf, StaticBuffer<T> &bmm1ResBuf,
                                       RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &fdStaging)
    {
        fdStagingBase = fdStaging.template GetTensor<uint8_t>().GetPhyAddr(0);
        stagingOutGm = fdStaging.template GetTensor<float>();
    }
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &intraCoreCombine,
                                              Buffer<BufferType::GM, SyncType::NO_SYNC> &crossCoreCombine)
    {
        intraCoreCombineBase = intraCoreCombine.template GetTensor<uint8_t>().GetPhyAddr(0);
        intraCoreCombineGm = intraCoreCombine.template GetTensor<float>();
        crossCoreCombineBase = crossCoreCombine.template GetTensor<uint8_t>().GetPhyAddr(0);
        crossCoreCombineGm = crossCoreCombine.template GetTensor<float>();
        fdStagingBase = crossCoreCombineBase;
        stagingOutGm = crossCoreCombineGm;
    }
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo);
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void ProcessVec2(StaticBuffer<T> &bmm2ResBuf, RunInfo &smla35VecRunInfo,
                                       ConstInfo &smla35VecConstInfo);
    __aicore__ inline void GetKVPhyAddr(uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx,
                                        uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
                                        bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
                                        GlobalTensor<int32_t> actualSeqOriKvlenGm,
                                        GlobalTensor<int32_t> cuSeqlensOriKvGm, GlobalTensor<int32_t> oriTopkLengthGm,
                                        bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
                                        GlobalTensor<int32_t> actualSeqCmpKvlenGm,
                                        GlobalTensor<int32_t> cuSeqlensCmpKvGm, GlobalTensor<int32_t> cmpTopkLengthGm,
                                        GlobalTensor<int32_t> cmpResidualKvGm, GlobalTensor<int32_t> actualSeqQlenGm,
                                        GlobalTensor<int32_t> cuSeqlensQGm, __gm__ uint8_t *workspace,
                                        ConstInfo &smla35VecConstInfo);
    __aicore__ inline void FreeEvent(ConstInfo &smla35VecConstInfo);

private:
    template <bool UPDATE>
    __aicore__ inline void ComputeVec1Softmax(LocalTensor<Q_T> &stage1CastTensor, LocalTensor<T> &mmRes,
                                              LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
                                              LocalTensor<T> &apiTmpBuffer, RunInfo &smla35VecRunInfo,
                                              ConstInfo &smla35VecConstInfo);
    __aicore__ inline void InitVec1SoftmaxFromSinks(LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
                                                    RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void CopyVec1ResultToL1(StaticBuffer<Q_T> &outputBuf, LocalTensor<Q_T> &stage1CastTensor,
                                              RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void StageCrossCoreVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
                                                 RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void StageBatchConsistencyVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
                                                        RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void StageLegacyVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
                                              RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void CopyOutVec1Lse(LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
                                          RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);

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

    __aicore__ inline uint32_t GetIntraCoreWorkspaceIdx(const RunInfo &smla35VecRunInfo,
                                                        const ConstInfo &smla35VecConstInfo) const
    {
        uint32_t coreIdx;
        if constexpr (IS_SPLIT_G) {
            coreIdx = static_cast<uint32_t>(smla35VecConstInfo.aivIdx >> 2U);
        } else {
            coreIdx = static_cast<uint32_t>(smla35VecConstInfo.aivIdx >> 1U);
        }
        return (coreIdx << 1U) + smla35VecRunInfo.multiCoreIdxMod2;
    }

    __aicore__ inline uint32_t GetCrossCoreWorkspaceIdx(const RunInfo &smla35VecRunInfo) const
    {
        return static_cast<uint32_t>(smla35VecRunInfo.firstFdDataWorkspaceIdx + smla35VecRunInfo.s2SplitIdx);
    }

    __aicore__ inline int64_t GetFaStagingMOffset(const RunInfo &smla35VecRunInfo,
                                                  const ConstInfo &smla35VecConstInfo) const
    {
        int64_t stagingMOffset =
            (smla35VecConstInfo.subBlockIdx == 1) ? static_cast<int64_t>(smla35VecRunInfo.firstHalfMRealSize) : 0L;
        if constexpr (IS_SPLIT_G) {
            stagingMOffset += static_cast<int64_t>(smla35VecRunInfo.goIdx);
        }
        return stagingMOffset;
    }

    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           const RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &smla35VecRunInfo,
                                           ConstInfo &smla35VecConstInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase, const RunInfo &smla35VecRunInfo,
                                           ConstInfo &smla35VecConstInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t *tokenData,
                                          const RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void CopyIn8Block(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t &s2,
                                        const RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void CopyToOutUb(LocalTensor<Q_T> kvNzUb, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                       ConstInfo &smla35VecConstInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                          ConstInfo &smla35VecConstInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase, const RunInfo &smla35VecRunInfo,
                                         ConstInfo &smla35VecConstInfo);
    __aicore__ inline void GetKVPhyAddrForKvType(
        uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
        bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> &actualSeqQlenGm,
        GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &actualSeqKvlenGm,
        GlobalTensor<int32_t> &cuSeqlensKvGm, GlobalTensor<int32_t> &topkLengthGm,
        GlobalTensor<int32_t> &cmpResidualKvGm, ConstInfo &smla35VecConstInfo, GlobalTensor<int32_t> &blockTableGm,
        GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm, uint32_t kvStride,
        uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount, uint32_t alignedSparseBlockCount,
        bool isOriKv);
    __aicore__ inline int32_t GetSmlaSeqLen(int32_t batchIndex, bool useExplicitLength, bool useCumulativeLength,
                                            GlobalTensor<int32_t> &explicitLengthGm,
                                            GlobalTensor<int32_t> &cumulativeLengthGm, int64_t fallbackLength);
    __aicore__ inline PhyAddrValidInfo CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size, int32_t actualOriS2Size,
                                                            int64_t restoredSize, ConstInfo &smla35VecConstInfo);
    __aicore__ inline int32_t CalcCurValidS2(uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv,
                                             GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &topkLengthGm,
                                             ConstInfo &smla35VecConstInfo, int32_t sparseBlockCount,
                                             const PhyAddrValidInfo &smla35ValidWindow);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo,
                                           LocalTensor<VEC2_RES_T> &smla35Vec2ResultUb, int64_t vec2S1Idx,
                                           int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo,
                                               LocalTensor<VEC2_RES_T> &smla35Vec2ResultUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer(uint32_t &ubAddr);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &smla35VecConstInfo);
    __aicore__ inline void ReduceIntraBlockAndStage(RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo,
                                                    LocalTensor<T> &smla35Vec2ResultUb, LocalTensor<T> &partialTmpUb);

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
    GlobalTensor<uint32_t> oriKvPhyAddrGm;
    GlobalTensor<uint32_t> cmpKvPhyAddrGm;

    StaticBuffer<float> fdLseTmpUb;
    StaticBuffer<T> commonUb;
    StaticBuffer<T> sinksUb;
    StaticBuffer<Q_T> stage1OutBufs[2];
    StaticBuffer<T> stage2OutBufs;
    StaticBuffer<Q_T> stage0OutBufs[2];
    StaticBuffer<float> softmaxMaxBufs[2];
    StaticBuffer<float> softmaxSumBufs[2];
    StaticBuffer<float> softmaxFinalMaxBufs[2];
    StaticBuffer<float> softmaxFinalSumBufs[2];
    StaticBuffer<T> softmaxExpBufs[2];
    StaticBuffer<float> batchReduceTmpUb;
    StaticBuffer<float> outLseUbs[2];
    TBuf<> vselrIndexesBuf[2];
    AttentionCommon::FdBuffers<StaticBuffer<uint8_t>> fdBuffers;
    uint32_t pingPongV0 = 0;
    __gm__ uint8_t *fdStagingBase = nullptr;
    GlobalTensor<float> stagingOutGm;
    __gm__ uint8_t *intraCoreCombineBase = nullptr;
    GlobalTensor<float> intraCoreCombineGm;
    __gm__ uint8_t *crossCoreCombineBase = nullptr;
    GlobalTensor<float> crossCoreCombineGm;

    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
};

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase,
                                                                          const RunInfo &smla35VecRunInfo,
                                                                          ConstInfo &smla35VecConstInfo)
{
    int64_t curSparseS2End = this->sparseS2End;
    int64_t sparseBlockCount = 0;
    int64_t curS2LoopCnt = smla35VecRunInfo.s2LoopCount;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        sparseBlockCount = smla35VecConstInfo.cmpSparseBlockCount;
        curS2LoopCnt -= smla35VecRunInfo.oriKvLoopEndIdx;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        sparseBlockCount = smla35VecConstInfo.oriSparseBlockCount;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (smla35VecRunInfo.isCmp) {
            sparseBlockCount = smla35VecConstInfo.cmpSparseBlockCount;
            curS2LoopCnt -= smla35VecRunInfo.oriKvLoopEndIdx;
        } else {
            sparseBlockCount = smla35VecConstInfo.oriSparseBlockCount;
        }
    }
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(smla35VecRunInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + smla35VecRunInfo.s1oIdx) * sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += smla35VecRunInfo.boIdx * smla35VecConstInfo.s1Size * sparseBlockCount +
                      smla35VecRunInfo.s1oIdx * sparseBlockCount; // B, S1, N2(1), K
    }

    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * smla35VecConstInfo.s2BaseSize;
    for (uint64_t i = 0; i < KV_COPYIN_UNIT; ++i) { // 每次处理8个数据块
        uint64_t idx = topkBS1Idx + smla35VecRunInfo.s2StartIdx + topkKIdx + i;
        if constexpr (!IS_FULL) {
            // 尾块：保留边界判断，防止越界读取
            if (likely(s2IdxInBase + i < curSparseS2End)) {
                tokenData[i] = sparseIndicesGm.GetValue(idx);
            } else {
                break;
            }
        } else {
            // 非尾块：8行均有效，直接读取
            tokenData[i] = sparseIndicesGm.GetValue(idx);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase,
                                                                        const RunInfo &smla35VecRunInfo,
                                                                        ConstInfo &smla35VecConstInfo)
{
    int64_t curSparseS2End = this->sparseS2End;
    uint32_t smlaAlignedSparseBlockCount = smla35VecRunInfo.isCmp ? smla35VecConstInfo.alignedCmpSparseBlockCount :
                                                                    smla35VecConstInfo.alignedOriSparseBlockCount;
    int64_t smlaCurS2LoopCnt = smla35VecRunInfo.s2LoopCount;
    GlobalTensor<int64_t> smlaPhyAddrGm64;
    if (smla35VecRunInfo.isCmp) {
        smlaCurS2LoopCnt -= smla35VecRunInfo.oriKvLoopEndIdx;
        smlaPhyAddrGm64 = cmpKvPhyAddrGm.template ReinterpretCast<int64_t>();
    } else {
        smlaPhyAddrGm64 = oriKvPhyAddrGm.template ReinterpretCast<int64_t>();
    }

    uint64_t smlaTopkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(smla35VecRunInfo.boIdx);
        smlaTopkBS1Idx += (actualSeqQPrefixSum + smla35VecRunInfo.s1oIdx) * smlaAlignedSparseBlockCount;
    } else {
        smlaTopkBS1Idx += smla35VecRunInfo.boIdx * smla35VecConstInfo.s1Size * smlaAlignedSparseBlockCount +
                          smla35VecRunInfo.s1oIdx * smlaAlignedSparseBlockCount;
    }
    uint64_t topkKIdx = s2IdxInBase + smlaCurS2LoopCnt * smla35VecConstInfo.s2BaseSize;
    for (uint64_t i = 0; i < KV_COPYIN_UNIT; ++i) { // 每次处理8个数据块
        uint64_t idx = smlaTopkBS1Idx + smla35VecRunInfo.s2StartIdx + topkKIdx + i;
        if constexpr (!IS_FULL) {
            // 尾块：保留边界判断，防止越界读取
            if (likely(s2IdxInBase + i < curSparseS2End)) {
                tokenData[i] = smlaPhyAddrGm64.GetValue(idx);
            } else {
                break;
            }
        } else {
            // 非尾块：8行均有效，直接读取
            tokenData[i] = smlaPhyAddrGm64.GetValue(idx);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SmlaCsaBlockVector<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx,
                                                                          const RunInfo &smla35VecRunInfo,
                                                                          ConstInfo &smla35VecConstInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t smlaRealkeyOffset = 0;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride =
            smla35VecRunInfo.isCmp ? smla35VecConstInfo.cmpKvStride : smla35VecConstInfo.oriKvStride;
        smlaRealkeyOffset =
            blockTableGm.GetValue(smla35VecRunInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * paBlockStride +
            blkTableOffset * smla35VecConstInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        if (smla35VecRunInfo.isCmp) {
            smlaRealkeyOffset = smla35VecRunInfo.boIdx * smla35VecConstInfo.n2Size * smla35VecConstInfo.cmpS2Size *
                                    smla35VecConstInfo.dSize +
                                smla35VecRunInfo.n2oIdx * smla35VecConstInfo.cmpS2Size * smla35VecConstInfo.dSize +
                                s2Idx * smla35VecConstInfo.dSize; // BSN(1)D
        } else {
            smlaRealkeyOffset = smla35VecRunInfo.boIdx * smla35VecConstInfo.n2Size * smla35VecConstInfo.s2Size *
                                    smla35VecConstInfo.dSize +
                                smla35VecRunInfo.n2oIdx * smla35VecConstInfo.s2Size * smla35VecConstInfo.dSize +
                                s2Idx * smla35VecConstInfo.dSize; // BSN(1)D
        }
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        smlaRealkeyOffset = (cuSeqlensKvGm.GetValue(smla35VecRunInfo.boIdx) + s2Idx) * smla35VecConstInfo.n2Size *
                                smla35VecConstInfo.dSize +
                            smla35VecRunInfo.n2oIdx * smla35VecConstInfo.dSize; // TN(1)D
    }
    return smlaRealkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                         int64_t keyOffset,
                                                                         ConstInfo &smla35VecConstInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = smla35VecConstInfo.dSize * sizeof(KV_T);

    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = (CeilAlign(smla35VecConstInfo.dSize * sizeof(KV_T), BUFFER_SIZE_BYTE_32B) -
                              smla35VecConstInfo.dSize * sizeof(KV_T)) /
                             sizeof(KV_T);
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * smla35VecConstInfo.dSize], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                         int64_t *tokenData,
                                                                         const RunInfo &smla35VecRunInfo,
                                                                         ConstInfo &smla35VecConstInfo)
{
    for (uint32_t i = 0; i < 8; i += 2) { // 遍历8个元素的数组/缓冲区，每次处理2个元素
        int64_t smlaKeyOffset0;
        int64_t smlaKeyOffset1;
        if constexpr (IS_VEC_S2PHYADDR) {
            smlaKeyOffset0 = tokenData[i];
            smlaKeyOffset1 = tokenData[i + 1];
        } else {
            smlaKeyOffset0 = GetkeyOffset(tokenData[i], smla35VecRunInfo, smla35VecConstInfo);
            smlaKeyOffset1 = GetkeyOffset(tokenData[i + 1], smla35VecRunInfo, smla35VecConstInfo);
        }
        if constexpr (!IS_FULL) {
            // 尾块：提前返回判断
            if (unlikely(smlaKeyOffset0 < 0 && smlaKeyOffset1 < 0)) {
                return;
            }
        }
        int64_t combineBytes = smla35VecConstInfo.dSizeVInput * sizeof(KV_T);
        int64_t keySrcStride;
        if constexpr (IS_BATCH_CONSISTENCY) {
            // batch一致性场景，token读取顺序只与逻辑顺序有关，为保证确定性不可交换读取顺序
            keySrcStride = (smlaKeyOffset1 - smlaKeyOffset0) * sizeof(KV_T) - combineBytes;
        } else {
            keySrcStride = (smlaKeyOffset0 > smlaKeyOffset1 ? (smlaKeyOffset0 - smlaKeyOffset1) :
                                                              (smlaKeyOffset1 - smlaKeyOffset0)) *
                               sizeof(KV_T) -
                           combineBytes;
        }
        if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || smla35VecConstInfo.sparseBlockSize > 1) {
            // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
            CopyInSingleKv(kvInUb, startRow, smlaKeyOffset0, smla35VecConstInfo);
            CopyInSingleKv(kvInUb, startRow + 1, smlaKeyOffset1, smla35VecConstInfo);
        } else {
            DataCopyExtParams intriParams;
            if constexpr (!IS_FULL) {
                // 尾块：根据实际有效条目数设置blockCount,且此处仅有可能存在keyOffset1为-1的情况
                intriParams.blockCount = 1 + (smlaKeyOffset1 >= 0);
            } else {
                // 非尾块：两条均有效，blockCount恒为2
                intriParams.blockCount = 2;
            }
            intriParams.blockLen = combineBytes;
            intriParams.dstStride = 0;
            intriParams.srcStride = keySrcStride;
            DataCopyPadExtParams<KV_T> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = (CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) - combineBytes) / sizeof(KV_T);
            padParams.paddingValue = 0;

            int64_t keyOffset;
            if constexpr (!IS_FULL) {
                keyOffset = (smlaKeyOffset1 > -1 && smlaKeyOffset1 < smlaKeyOffset0) ? smlaKeyOffset1 : smlaKeyOffset0;
            } else {
                // 非尾块：两条均有效，取较小地址作为起始
                keyOffset = smlaKeyOffset0 < smlaKeyOffset1 ? smlaKeyOffset0 : smlaKeyOffset1;
            }
            DataCopyPad(kvInUb[startRow * smla35VecConstInfo.dSize], keyGm[keyOffset], intriParams, padParams);
        }
        startRow += 2; // 每次迭代处理2个输入元素
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CopyToOutUb(LocalTensor<Q_T> kvOutUb,
                                                                      LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                                                      ConstInfo &smla35VecConstInfo)
{
    LocalTensor<Q_T> kvNdUb = srcTensor.template ReinterpretCast<Q_T>();
    DataCopy(kvOutUb, kvNdUb, dealRow * smla35VecConstInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> kvOutUb, int64_t dealRow,
    int64_t s2StartIdx, const RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * smla35VecConstInfo.dSize], kvOutUb, dealRow * smla35VecConstInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &smla35VecRunInfo,
                                                                           ConstInfo &smla35VecConstInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = smla35VecConstInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(smla35VecRunInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = smla35VecRunInfo.s2RealSize - v0S2SizeFirstCore;
        int32_t vecCnt = (aicIdx % 2U == 0) ?
                             (GetSubBlockIdx() == 0 ? 0 : 1) :
                             (GetSubBlockIdx() == 0 ? 2 : 3); // 2，3：根据核心索引和子块索引设置处理参数
        if (aicIdx % 2 == 0) { // 2：根据aicIdx的奇偶性来区分不同的处理逻辑
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeFirstCore, 2); // 2：处理大小为v0S2SizeFirstCore的一半
                sparseS2Start = 0;
            } else {
                sparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2); // 2：处理剩余部分
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2); // 2：起始位置为v0S2SizeFirstCore的一半
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeSecondCore, 2); // 2：处理大小为v0S2SizeSecondCore的一半
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2); // 2：处理剩余部分
                sparseS2Start =
                    v0S2SizeFirstCore +
                    CeilDiv(v0S2SizeSecondCore, 2); // 2：起始位置为v0S2SizeFirstCore加上v0S2SizeSecondCore的一半
            }
        }
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        uint32_t v0S2SizeFirstCore = CeilDiv(smla35VecRunInfo.s2RealSize, 2); // 2：平均分配给两个子块
        sparseCalSize = GetSubBlockIdx() == 0 ? v0S2SizeFirstCore : smla35VecRunInfo.s2RealSize - v0S2SizeFirstCore;
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : v0S2SizeFirstCore;
        sparseS2End = sparseS2Start + sparseCalSize;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &smla35VecRunInfo,
    ConstInfo &smla35VecConstInfo)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (smla35VecRunInfo.s2LoopCount < smla35VecRunInfo.oriKvLoopEndIdx) {
            return;
        }
        keyGm = cmpKVGm;
        cuSeqlensKvGm = cuSeqlensCmpKvGm;
        sparseIndicesGm = cmpSparseIndicesGm;
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            blockTableGm = cmpBlockTableGm;
            blockSize = smla35VecConstInfo.cmpBlockSize;
            maxBlockNumPerBatch = smla35VecConstInfo.cmpMaxBlockNumPerBatch;
        }
        CalSparseCalSize(smla35VecRunInfo, smla35VecConstInfo);
        ProcessSparseKv(v0ResGm, smla35VecRunInfo, smla35VecConstInfo);
        v0ResGm.SetCrossCore();
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!smla35VecRunInfo.isCmp) {
            keyGm = oriKVGm;
            cuSeqlensKvGm = cuSeqlensOriKvGm;
            sparseIndicesGm = oriSparseIndicesGm;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                blockTableGm = oriBlockTableGm;
                blockSize = smla35VecConstInfo.oriBlockSize;
                maxBlockNumPerBatch = smla35VecConstInfo.oriMaxBlockNumPerBatch;
            }
        } else {
            keyGm = cmpKVGm;
            cuSeqlensKvGm = cuSeqlensCmpKvGm;
            sparseIndicesGm = cmpSparseIndicesGm;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                blockTableGm = cmpBlockTableGm;
                blockSize = smla35VecConstInfo.cmpBlockSize;
                maxBlockNumPerBatch = smla35VecConstInfo.cmpMaxBlockNumPerBatch;
            }
        }
        CalSparseCalSize(smla35VecRunInfo, smla35VecConstInfo);
        ProcessSparseKv(v0ResGm, smla35VecRunInfo, smla35VecConstInfo);
        v0ResGm.SetCrossCore();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CopyIn8Block(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                       int64_t &s2, const RunInfo &smla35VecRunInfo,
                                                                       ConstInfo &smla35VecConstInfo)
{
    // tokenData元素为-1表示无效token，尾块场景下GetReal*仅填充有效区间，其余保持-1
    int64_t tokenData[KV_COPYIN_UNIT] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if constexpr (IS_VEC_S2PHYADDR) {
        GetRealS2Addr<IS_FULL>(tokenData, s2, smla35VecRunInfo, smla35VecConstInfo);
    } else {
        GetRealCmpS2Idx<IS_FULL>(tokenData, s2, smla35VecRunInfo, smla35VecConstInfo);
    }
    s2 += KV_COPYIN_UNIT;
    CopyInKvSparse<IS_FULL>(kvInUb, startRow, tokenData, smla35VecRunInfo, smla35VecConstInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &smla35VecRunInfo,
    ConstInfo &smla35VecConstInfo)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t curSparseCalSize = this->sparseCalSize;
        if (curSparseCalSize == 0) {
            return;
        }

        // 前置计算：16行拷出循环数、尾块行数、尾块内8行搬入次数及剩余行数
        int64_t process16LoopCnt = curSparseCalSize / KV_PROCESS_UNIT;
        int64_t tail16Rows = curSparseCalSize % KV_PROCESS_UNIT;
        int64_t tail8FullCnt = tail16Rows / KV_COPYIN_UNIT;
        int64_t tail8Remain = tail16Rows % KV_COPYIN_UNIT;
        int64_t s2 = sparseS2Start;

        // 阶段1：完整16行块（非尾块，8行均有效，无需判断）
        for (int64_t i = 0; i < process16LoopCnt; i++) {
            int64_t s2StartIdx = sparseS2Start + i * KV_PROCESS_UNIT;
            // 1、copy kv in, gm -> ub
            LocalTensor<Q_T> stage0OutUb = this->stage0OutBufs[pingPongV0].tensor;
            WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(pingPongV0));
            CopyIn8Block<true>(stage0OutUb, 0, s2, smla35VecRunInfo, smla35VecConstInfo);
            CopyIn8Block<true>(stage0OutUb, KV_COPYIN_UNIT, s2, smla35VecRunInfo, smla35VecConstInfo);
            // 2、copy kv out, ub -> l1
            SetFlag<HardEvent::MTE2_MTE3>(INNERCORE_STAGE0OUT_MTE2_MTE3(pingPongV0));
            WaitFlag<HardEvent::MTE2_MTE3>(INNERCORE_STAGE0OUT_MTE2_MTE3(pingPongV0));
            CopyOutKvUb2Gm(v0ResGm, stage0OutUb, KV_PROCESS_UNIT, s2StartIdx, smla35VecRunInfo, smla35VecConstInfo);
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(pingPongV0));
            pingPongV0 ^= 1;
        }

        // 阶段2：尾块（不足16行，保留判断逻辑）
        if (tail16Rows > 0) {
            int64_t s2StartIdx = sparseS2Start + process16LoopCnt * KV_PROCESS_UNIT;
            int64_t dealRow = 0;
            // 1、copy kv in, gm -> ub
            LocalTensor<Q_T> stage0OutUb = this->stage0OutBufs[pingPongV0].tensor;
            WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(pingPongV0));
            // 尾块内满8行搬入（8行均有效，无需判断）
            for (int64_t j = 0; j < tail8FullCnt; j++) {
                CopyIn8Block<true>(stage0OutUb, dealRow, s2, smla35VecRunInfo, smla35VecConstInfo);
                dealRow += KV_COPYIN_UNIT;
            }
            // 尾块内不足8行（需判断边界，与当前逻辑一致）
            if (tail8Remain > 0) {
                CopyIn8Block<false>(stage0OutUb, dealRow, s2, smla35VecRunInfo, smla35VecConstInfo);
                dealRow += tail8Remain;
            }
            // 2、copy kv out, ub -> l1
            SetFlag<HardEvent::MTE2_MTE3>(INNERCORE_STAGE0OUT_MTE2_MTE3(pingPongV0));
            WaitFlag<HardEvent::MTE2_MTE3>(INNERCORE_STAGE0OUT_MTE2_MTE3(pingPongV0));
            CopyOutKvUb2Gm(v0ResGm, stage0OutUb, tail16Rows, s2StartIdx, smla35VecRunInfo, smla35VecConstInfo);
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(pingPongV0));
            pingPongV0 ^= 1;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool UPDATE>
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::ComputeVec1Softmax(
    LocalTensor<Q_T> &stage1CastTensor, LocalTensor<T> &mmRes, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb,
    LocalTensor<T> &apiTmpBuffer, RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo)
{
    if (likely(smla35VecRunInfo.s2RealSize == 128 && smla35VecRunInfo.s2RealSizeUpdate == 128)) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, smla35VecRunInfo.halfMRealSize,
            smla35VecRunInfo.s2RealSizeUpdate, static_cast<T>(smla35VecConstInfo.softmaxScale), negativeFloatScalar);
    } else if (smla35VecRunInfo.s2RealSize <= 64) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, smla35VecRunInfo.halfMRealSize,
            smla35VecRunInfo.s2RealSizeUpdate, static_cast<T>(smla35VecConstInfo.softmaxScale), negativeFloatScalar);
    } else if (smla35VecRunInfo.s2RealSize < 128 || smla35VecRunInfo.s2RealSizeUpdate < 128) {
        ProcessVec1Vf<T, Q_T, UPDATE, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
            stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, smla35VecRunInfo.halfMRealSize,
            smla35VecRunInfo.s2RealSizeUpdate, static_cast<T>(smla35VecConstInfo.softmaxScale), negativeFloatScalar);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::InitVec1SoftmaxFromSinks(LocalTensor<float> &sumUb,
                                                                                   LocalTensor<float> &maxUb,
                                                                                   RunInfo &smla35VecRunInfo,
                                                                                   ConstInfo &smla35VecConstInfo)
{
    bool includeSink = (!smla35VecRunInfo.isCrossCoreSplit) || smla35VecRunInfo.isFirstS2SplitCore;
    if constexpr (IS_BATCH_CONSISTENCY) {
        includeSink = includeSink && (smla35VecRunInfo.reduceBlockId == 0);
    }
    if (!includeSink) {
        Duplicate(maxUb, this->negativeFloatScalar, smla35VecRunInfo.halfMRealSize);
        Duplicate(sumUb, static_cast<T>(0), smla35VecRunInfo.halfMRealSize);
        return;
    }
    int64_t sinksOffset = 0;
    if constexpr (!IS_SPLIT_G) {
        sinksOffset = GetBlockIdx() % 2 == 0 ? 0 : smla35VecRunInfo.firstHalfMRealSize;
    } else {
        sinksOffset = smla35VecRunInfo.goIdx;
        if (smla35VecConstInfo.subBlockIdx == 1) {
            sinksOffset += smla35VecRunInfo.firstHalfMRealSize;
        }
    }
    LocalTensor<T> sinksUb = this->sinksUb.tensor;
    InitSoftmaxFromSinks<T>(sumUb, maxUb, sinksUb, sinksOffset, R0, smla35VecRunInfo.halfMRealSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CopyVec1ResultToL1(StaticBuffer<Q_T> &outputBuf,
                                                                             LocalTensor<Q_T> &stage1CastTensor,
                                                                             RunInfo &smla35VecRunInfo,
                                                                             ConstInfo &smla35VecConstInfo)
{
    int64_t stage1Offset = smla35VecRunInfo.taskIdMod2;
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE1(stage1Offset));
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE1(stage1Offset));
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.tensor;
    if (likely(smla35VecRunInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[smla35VecConstInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (smla35VecRunInfo.mRealSize - smla35VecRunInfo.halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 16, static_cast<uint16_t>(smla35VecRunInfo.halfMRealSize),
                  static_cast<uint16_t>(vec1Srcstride - smla35VecRunInfo.halfMRealSize),
                  static_cast<uint16_t>(Align16Func(smla35VecRunInfo.mRealSize) - smla35VecRunInfo.halfMRealSize)});
    }
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(stage1Offset));
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CROSSCORE_L1P(outputBuf.idx));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::StageCrossCoreVec1Lse(LocalTensor<float> &maxUb,
                                                                                LocalTensor<float> &sumUb,
                                                                                RunInfo &smla35VecRunInfo,
                                                                                ConstInfo &smla35VecConstInfo)
{
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
        smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false),
        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
    AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase, GetCrossCoreWorkspaceIdx(smla35VecRunInfo),
                                  GetFaStagingMOffset(smla35VecRunInfo, smla35VecConstInfo),
                                  smla35VecRunInfo.halfMRealSize, maxUb, sumUb, tmpUb, INNERCORE_STAGE2,
                                  INNERCORE_STAGE_FD_MTE3_V);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::StageBatchConsistencyVec1Lse(LocalTensor<float> &maxUb,
                                                                                       LocalTensor<float> &sumUb,
                                                                                       RunInfo &smla35VecRunInfo,
                                                                                       ConstInfo &smla35VecConstInfo)
{
    if (!smla35VecRunInfo.isLastBase) {
        return;
    }
    if (smla35VecRunInfo.halfMRealSize > 0) {
        LocalTensor<float> finalMaxUb = this->softmaxFinalMaxBufs[smla35VecRunInfo.taskIdMod2].tensor;
        LocalTensor<float> finalSumUb = this->softmaxFinalSumBufs[smla35VecRunInfo.taskIdMod2].tensor;
        uint64_t snapshotElems = Align8Func(smla35VecRunInfo.halfMRealSize);
        DataCopy(finalMaxUb, maxUb, snapshotElems);
        DataCopy(finalSumUb, sumUb, snapshotElems);
    }
    if (smla35VecRunInfo.isCrossCoreSplit && !smla35VecRunInfo.isFirstS2SplitCore) {
        StageCrossCoreVec1Lse(maxUb, sumUb, smla35VecRunInfo, smla35VecConstInfo);
    } else if (smla35VecRunInfo.isFirstS2SplitCore && smla35VecRunInfo.reduceBlockId == 0 &&
               smla35VecRunInfo.s2LoopCount < smla35VecRunInfo.s2LoopLimit) {
        AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
            smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true),
            AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
        LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
        AttentionCommon::StageVec1Lse(
            stagingLayout, intraCoreCombineBase, GetIntraCoreWorkspaceIdx(smla35VecRunInfo, smla35VecConstInfo),
            GetFaStagingMOffset(smla35VecRunInfo, smla35VecConstInfo), smla35VecRunInfo.halfMRealSize, maxUb, sumUb,
            tmpUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        // 零行 AIV 的 Vec2 会提前返回，不会等待此事件，因此不发送信号。
        if (smla35VecRunInfo.halfMRealSize > 0) {
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(smla35VecRunInfo.multiCoreIdxMod2));
        }
    } else if (smla35VecRunInfo.isCrossCoreSplit && smla35VecRunInfo.isFirstS2SplitCore &&
               smla35VecRunInfo.reduceBlockId == 0) {
        StageCrossCoreVec1Lse(maxUb, sumUb, smla35VecRunInfo, smla35VecConstInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::StageLegacyVec1Lse(LocalTensor<float> &maxUb,
                                                                             LocalTensor<float> &sumUb,
                                                                             RunInfo &smla35VecRunInfo,
                                                                             ConstInfo &smla35VecConstInfo)
{
    if (!smla35VecRunInfo.isCrossCoreSplit || smla35VecRunInfo.halfMRealSize <= 0 ||
        smla35VecRunInfo.s2LoopCount != smla35VecRunInfo.s2LoopLimit) {
        return;
    }
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
        smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    LocalTensor<float> tmpUb = this->fdLseTmpUb.tensor;
    AttentionCommon::StageVec1Lse(stagingLayout, fdStagingBase, GetCrossCoreWorkspaceIdx(smla35VecRunInfo),
                                  GetFaStagingMOffset(smla35VecRunInfo, smla35VecConstInfo),
                                  static_cast<uint32_t>(smla35VecRunInfo.halfMRealSize), maxUb, sumUb, tmpUb,
                                  INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CopyOutVec1Lse(LocalTensor<float> &maxUb,
                                                                         LocalTensor<float> &sumUb,
                                                                         RunInfo &smla35VecRunInfo,
                                                                         ConstInfo &smla35VecConstInfo)
{
    // 零行 AIV 的 Vec2 会提前返回，不会等待此事件，因此不发送信号。
    bool copyOutLse = smla35VecConstInfo.isSoftmaxLseEnable && smla35VecRunInfo.halfMRealSize > 0 &&
                      smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit;
    if constexpr (IS_BATCH_CONSISTENCY) {
        copyOutLse = copyOutLse && !smla35VecRunInfo.isCrossCoreSplit && !smla35VecRunInfo.needReduce;
    }
    if (!copyOutLse) {
        return;
    }
    LocalTensor<float> outLse = this->outLseUbs[smla35VecRunInfo.multiCoreIdxMod2].tensor;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = sizeof(float) * smla35VecRunInfo.halfMRealSize;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
    ComputeLse<float>(outLse, sumUb, maxUb, smla35VecRunInfo.halfMRealSize);
    SetFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
    DataCopyPad(this->softmaxLseGm[smla35VecRunInfo.softmaxLseOffset], outLse, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec1(StaticBuffer<Q_T> &outputBuf,
                                                                      StaticBuffer<T> &bmm1ResBuf,
                                                                      RunInfo &smla35VecRunInfo,
                                                                      ConstInfo &smla35VecConstInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1ResBuf.idx));

    LocalTensor<float> smlaSumUb = this->softmaxSumBufs[smla35VecRunInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> smlaMaxUb = this->softmaxMaxBufs[smla35VecRunInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> smlaExpUb = this->softmaxExpBufs[smla35VecRunInfo.taskIdMod2].tensor;
    int64_t smlaStage1Offset = smla35VecRunInfo.taskIdMod2;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(smlaStage1Offset));
    LocalTensor<Q_T> smlaStage1CastTensor = this->stage1OutBufs[smlaStage1Offset].tensor;

    LocalTensor<T> smlaApiTmpBuffer = this->commonUb.tensor;
    LocalTensor<T> smlaMmRes = bmm1ResBuf.tensor;

    smla35VecRunInfo.s2RealSizeUpdate = smla35VecRunInfo.s2RealSize;

    bool isFirstSoftmaxBase = smla35VecRunInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        isFirstSoftmaxBase = smla35VecRunInfo.isFirstBase;
    }
    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (isFirstSoftmaxBase && !isSinks) {
        ComputeVec1Softmax<false>(smlaStage1CastTensor, smlaMmRes, smlaSumUb, smlaMaxUb, smlaApiTmpBuffer,
                                  smla35VecRunInfo, smla35VecConstInfo);
    } else {
        if (isFirstSoftmaxBase && isSinks) {
            InitVec1SoftmaxFromSinks(smlaSumUb, smlaMaxUb, smla35VecRunInfo, smla35VecConstInfo);
        }
        ComputeVec1Softmax<true>(smlaStage1CastTensor, smlaMmRes, smlaSumUb, smlaMaxUb, smlaApiTmpBuffer,
                                 smla35VecRunInfo, smla35VecConstInfo);
    }
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1ResBuf.idx));
    CopyVec1ResultToL1(outputBuf, smlaStage1CastTensor, smla35VecRunInfo, smla35VecConstInfo);
    if (!isFirstSoftmaxBase || isSinks) {
        SFAUpdateExpSumAndExpMax<T>(smlaSumUb, smlaMaxUb, smlaExpUb, smlaSumUb, smlaMaxUb, smlaApiTmpBuffer,
                                    smla35VecRunInfo.halfMRealSize);
    }
    if constexpr (IS_BATCH_CONSISTENCY) {
        StageBatchConsistencyVec1Lse(smlaMaxUb, smlaSumUb, smla35VecRunInfo, smla35VecConstInfo);
    } else {
        StageLegacyVec1Lse(smlaMaxUb, smlaSumUb, smla35VecRunInfo, smla35VecConstInfo);
    }
    CopyOutVec1Lse(smlaMaxUb, smlaSumUb, smla35VecRunInfo, smla35VecConstInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::ReduceIntraBlockAndStage(RunInfo &smla35VecRunInfo,
                                                                                   ConstInfo &smla35VecConstInfo,
                                                                                   LocalTensor<T> &smla35Vec2ResultUb,
                                                                                   LocalTensor<T> &partialTmpUb)
{
    AttentionCommon::S2SplitFdStagingLayout intraLayout = {
        smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true),
        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    AttentionCommon::S2SplitFdStagingLayout crossLayout = {
        smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false),
        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    uint32_t intraWorkspaceIdx = GetIntraCoreWorkspaceIdx(smla35VecRunInfo, smla35VecConstInfo);
    uint32_t crossWorkspaceIdx = static_cast<uint32_t>(smla35VecRunInfo.firstFdDataWorkspaceIdx +
                                                       smla35VecRunInfo.s2SplitIdx - smla35VecRunInfo.reduceBlockId);
    int64_t stagingMOffset = GetFaStagingMOffset(smla35VecRunInfo, smla35VecConstInfo);
    LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
    LocalTensor<float> blockMaxUb = tmpUb;
    LocalTensor<float> blockSumUb = tmpUb[256];
    LocalTensor<float> lseBroadcastUb = tmpUb[512];
    LocalTensor<float> sumBroadcastUb = tmpUb[640];
    LocalTensor<float> maxUb = this->softmaxFinalMaxBufs[smla35VecRunInfo.taskIdMod2].tensor;
    LocalTensor<float> sumUb = this->softmaxFinalSumBufs[smla35VecRunInfo.taskIdMod2].tensor;
    bool copyOutMergedLse = smla35VecConstInfo.isSoftmaxLseEnable && !smla35VecRunInfo.isCrossCoreSplit &&
                            smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit;

    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(smla35VecRunInfo.multiCoreIdxMod2));
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(smla35VecRunInfo.multiCoreIdxMod2));
    LocalTensor<T> sinkUb;
    int64_t startRow = 0;
    while (startRow < smla35VecRunInfo.vec2MRealSize) {
        int64_t smla35DealRows = intraLayout.chunkRows;
        if (startRow + smla35DealRows > smla35VecRunInfo.vec2MRealSize) {
            smla35DealRows = smla35VecRunInfo.vec2MRealSize - startRow;
        }
        LocalTensor<T> chunkCurrent = smla35Vec2ResultUb[startRow * dTemplateAlign64];
        LocalTensor<float> chunkMaxUb = maxUb[startRow];
        LocalTensor<float> chunkSumUb = sumUb[startRow];
        if (copyOutMergedLse) {
            WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
        AttentionCommon::MergeStagedAndCurrentChunk<T, dTemplateAlign64>(
            intraLayout, intraCoreCombineBase, intraWorkspaceIdx, stagingMOffset + startRow, smla35DealRows,
            static_cast<int64_t>(smla35VecConstInfo.dSizeV), chunkMaxUb, chunkSumUb, chunkCurrent, blockMaxUb,
            blockSumUb, partialTmpUb, lseBroadcastUb, sumBroadcastUb, sinkUb, INNERCORE_REDUCE_MAXSUM_V_MTE2,
            INNERCORE_INTRAPARTIALO_V_MTE2, INNERCORE_REDUCE_MTE2_V);

        AttentionCommon::StageBroadcastMaxSum(intraLayout, intraCoreCombineBase, intraWorkspaceIdx,
                                              stagingMOffset + startRow, smla35DealRows, lseBroadcastUb, sumBroadcastUb,
                                              INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        if (copyOutMergedLse) {
            DataCopyExtParams lseParams;
            lseParams.blockCount = static_cast<uint16_t>(smla35DealRows);
            lseParams.blockLen = sizeof(float);
            lseParams.srcStride = 0;
            lseParams.dstStride = 0;
            SetFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
            DataCopyPad(this->softmaxLseGm[smla35VecRunInfo.softmaxLseOffset + startRow], lseBroadcastUb, lseParams);
            SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
        if (smla35VecRunInfo.isCrossCoreSplit && smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit) {
            AttentionCommon::StageBroadcastMaxSum(crossLayout, crossCoreCombineBase, crossWorkspaceIdx,
                                                  stagingMOffset + startRow, smla35DealRows, lseBroadcastUb,
                                                  sumBroadcastUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        }
        startRow += intraLayout.chunkRows;
    }

    AttentionCommon::StageVec2PartialOAndWait<T>(intraLayout, intraCoreCombineGm, intraWorkspaceIdx, stagingMOffset,
                                                 smla35VecRunInfo.vec2MRealSize,
                                                 static_cast<uint32_t>(smla35VecConstInfo.dSizeV), smla35Vec2ResultUb,
                                                 INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
    if (smla35VecRunInfo.isCrossCoreSplit && smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit) {
        AttentionCommon::StageVec2PartialOAndWait<T>(crossLayout, crossCoreCombineGm, crossWorkspaceIdx, stagingMOffset,
                                                     smla35VecRunInfo.vec2MRealSize,
                                                     static_cast<uint32_t>(smla35VecConstInfo.dSizeV),
                                                     smla35Vec2ResultUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
    }
    if (smla35VecRunInfo.s2LoopCount < smla35VecRunInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(smla35VecRunInfo.multiCoreIdxMod2));
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(smla35VecRunInfo.multiCoreIdxMod2));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec2(StaticBuffer<T> &bmm2ResBuf,
                                                                      RunInfo &smla35VecRunInfo,
                                                                      ConstInfo &smla35VecConstInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    if (unlikely(smla35VecRunInfo.vec2MBaseSize == 0)) {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
        return;
    }

    smla35VecRunInfo.vec2MRealSize = smla35VecRunInfo.vec2MBaseSize;
    int64_t smlaVec2CalcSize = smla35VecRunInfo.vec2MRealSize * dTemplateAlign64;
    LocalTensor<T> smlaVec2ResUb = this->stage2OutBufs.tensor;
    LocalTensor<T> smlaMmRes = bmm2ResBuf.tensor;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    bool smlaNeedIntraBlockReduce = false;
    if constexpr (IS_BATCH_CONSISTENCY) {
        smlaNeedIntraBlockReduce =
            smla35VecRunInfo.isLastBase && smla35VecRunInfo.isFirstS2SplitCore && smla35VecRunInfo.reduceBlockId > 0;
        if (smlaNeedIntraBlockReduce) {
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
        }
    }
    bool smlaIsFirstVec2Base = smla35VecRunInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        smlaIsFirstVec2Base = smla35VecRunInfo.isFirstBase;
    }
    if (unlikely(smlaIsFirstVec2Base)) {
        DataCopy(smlaVec2ResUb, smlaMmRes, smlaVec2CalcSize);
    } else {
        if (smla35VecRunInfo.s2RealSizeUpdate > 0) {
            LocalTensor<T> expUb = softmaxExpBufs[smla35VecRunInfo.taskIdMod2].tensor;
            bool isLastVec2Base = (smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit);
            if constexpr (IS_BATCH_CONSISTENCY) {
                isLastVec2Base = smla35VecRunInfo.isLastBase;
            }
            if (isLastVec2Base) {
                LocalTensor<float> smlaSumUb;
                if constexpr (IS_BATCH_CONSISTENCY) {
                    smlaSumUb = this->softmaxFinalSumBufs[smla35VecRunInfo.taskIdMod2].tensor;
                } else {
                    smlaSumUb = this->softmaxSumBufs[smla35VecRunInfo.multiCoreIdxMod2].tensor;
                }
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    smlaVec2ResUb, smlaMmRes, smlaVec2ResUb, expUb, expUb, smlaSumUb, smla35VecRunInfo.vec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    smlaVec2ResUb, smlaMmRes, smlaVec2ResUb, expUb, expUb, smla35VecRunInfo.vec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
            }
        } else {
            bool isLastVec2Base = smla35VecRunInfo.s2LoopCount >= smla35VecRunInfo.s2LoopLimit;
            if constexpr (IS_BATCH_CONSISTENCY) {
                isLastVec2Base = smla35VecRunInfo.isLastBase;
            }
            if (isLastVec2Base) {
                LocalTensor<float> smlaSumUb;
                if constexpr (IS_BATCH_CONSISTENCY) {
                    smlaSumUb = this->softmaxFinalSumBufs[smla35VecRunInfo.taskIdMod2].tensor;
                } else {
                    smlaSumUb = this->softmaxSumBufs[smla35VecRunInfo.multiCoreIdxMod2].tensor;
                }
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    smlaVec2ResUb, smlaVec2ResUb, smlaSumUb, smla35VecRunInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
        }
    }

    if constexpr (IS_BATCH_CONSISTENCY) {
        if (smla35VecRunInfo.isLastBase) {
            if (unlikely(smlaIsFirstVec2Base)) {
                LocalTensor<float> smlaSumUb = this->softmaxFinalSumBufs[smla35VecRunInfo.taskIdMod2].tensor;
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    smlaVec2ResUb, smlaVec2ResUb, smlaSumUb, smla35VecRunInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
            if (smlaNeedIntraBlockReduce) {
                SetFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
                SetFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
                ReduceIntraBlockAndStage(smla35VecRunInfo, smla35VecConstInfo, smlaVec2ResUb, smlaMmRes);
                if (!smla35VecRunInfo.isCrossCoreSplit &&
                    smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(smla35VecRunInfo, smla35VecConstInfo, smlaVec2ResUb, 0, smlaVec2CalcSize);
                }
            } else {
                if (smla35VecRunInfo.isFirstS2SplitCore && smla35VecRunInfo.reduceBlockId == 0 &&
                    smla35VecRunInfo.s2LoopCount < smla35VecRunInfo.s2LoopLimit) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                        smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(true),
                        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    int64_t smlaStagingMOffset = GetFaStagingMOffset(smla35VecRunInfo, smla35VecConstInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(
                        stagingLayout, intraCoreCombineGm,
                        GetIntraCoreWorkspaceIdx(smla35VecRunInfo, smla35VecConstInfo), smlaStagingMOffset,
                        smla35VecRunInfo.vec2MRealSize, static_cast<uint32_t>(smla35VecConstInfo.dSizeV), smlaVec2ResUb,
                        INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
                    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(smla35VecRunInfo.multiCoreIdxMod2));
                }
                if (smla35VecRunInfo.isCrossCoreSplit &&
                    (!smla35VecRunInfo.isFirstS2SplitCore ||
                     (smla35VecRunInfo.isFirstS2SplitCore && smla35VecRunInfo.reduceBlockId == 0 &&
                      smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit))) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                        smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(false),
                        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    uint32_t smlaWorkspaceIdx = GetCrossCoreWorkspaceIdx(smla35VecRunInfo);
                    int64_t smlaStagingMOffset = GetFaStagingMOffset(smla35VecRunInfo, smla35VecConstInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(
                        stagingLayout, crossCoreCombineGm, smlaWorkspaceIdx, smlaStagingMOffset,
                        smla35VecRunInfo.vec2MRealSize, static_cast<uint32_t>(smla35VecConstInfo.dSizeV), smlaVec2ResUb,
                        INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
                } else if (!smla35VecRunInfo.isCrossCoreSplit &&
                           smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(smla35VecRunInfo, smla35VecConstInfo, smlaVec2ResUb, 0, smlaVec2CalcSize);
                }
            }
        }
    } else if (smla35VecRunInfo.s2LoopCount == smla35VecRunInfo.s2LoopLimit) {
        if (unlikely(smla35VecRunInfo.s2LoopCount == 0)) {
            LocalTensor<float> smlaSumUb = this->softmaxSumBufs[smla35VecRunInfo.multiCoreIdxMod2].tensor;
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                smlaVec2ResUb, smlaVec2ResUb, smlaSumUb, smla35VecRunInfo.vec2MRealSize, dTemplateAlign64, 1.0);
        }
        if (smla35VecRunInfo.isCrossCoreSplit) {
            AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(),
                AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
            uint32_t smlaWorkspaceIdx = GetCrossCoreWorkspaceIdx(smla35VecRunInfo);
            int64_t smlaStagingMOffset = GetFaStagingMOffset(smla35VecRunInfo, smla35VecConstInfo);
            AttentionCommon::StageVec2PartialOAndWait<T>(
                stagingLayout, stagingOutGm, smlaWorkspaceIdx, smlaStagingMOffset,
                static_cast<uint32_t>(smla35VecRunInfo.vec2MRealSize), static_cast<uint32_t>(smla35VecConstInfo.dSizeV),
                smlaVec2ResUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        } else {
            this->CopyOutAttentionOut(smla35VecRunInfo, smla35VecConstInfo, smlaVec2ResUb, 0, smlaVec2CalcSize);
        }
    }
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::InitFDBuffers(FdRunInfo &fdRunInfo)
{
    FdRunInfo smlaFdBufferInfo = fdRunInfo;
    if (smlaFdBufferInfo.mNum > AttentionCommon::FD_REDUCE_CHUNK_ROWS) {
        smlaFdBufferInfo.mNum = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
    }
    AttentionCommon::InitFDBuffersStatic<T, dTemplateAlign64>(smlaFdBufferInfo, 0, fdBuffers);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessFlashDecode(FdRunInfo &fdRunInfo,
                                                                             ConstInfo &smla35VecConstInfo)
{
    InitFDBuffers(fdRunInfo);
    int64_t seqOffset = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        seqOffset = this->cuSeqlensQGm.GetValue(fdRunInfo.bn2Idx);
    } else {
        seqOffset = fdRunInfo.bn2Idx * smla35VecConstInfo.s1Size;
    }
    int64_t attentionOutOffset = seqOffset * smla35VecConstInfo.n2GDv + fdRunInfo.mIdx * smla35VecConstInfo.n2GDv +
                                 fdRunInfo.mStartIdx * smla35VecConstInfo.dSizeV;
    int64_t softmaxLseOffset = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        softmaxLseOffset = (seqOffset + fdRunInfo.mIdx) * smla35VecConstInfo.gSize + fdRunInfo.mStartIdx;
    } else {
        softmaxLseOffset = (fdRunInfo.bn2Idx * smla35VecConstInfo.s1Size + fdRunInfo.mIdx) * smla35VecConstInfo.gSize +
                           fdRunInfo.mStartIdx;
    }
    LocalTensor<T> smlaAccumulatedO = this->fdBuffers.accumOut.tensor.template ReinterpretCast<T>();
    LocalTensor<float> smlaLseExpUb = this->fdBuffers.lseExp.tensor.template ReinterpretCast<float>();
    LocalTensor<float> smlaBlockMaxUb = this->fdBuffers.blockMax.tensor.template ReinterpretCast<float>();
    LocalTensor<float> smlaBlockSumUb = this->fdBuffers.blockSum.tensor.template ReinterpretCast<float>();
    LocalTensor<T> smlaPartialOFp32 = this->fdBuffers.partialO.tensor.template ReinterpretCast<T>();
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
        smla35VecConstInfo.gSize, dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    int64_t attentionOutRowStride = static_cast<int64_t>(smla35VecConstInfo.dSizeV) +
                                    static_cast<int64_t>(smla35VecConstInfo.attentionOutStride) / sizeof(OUTPUT_T);
    int64_t startRow = 0;
    while (startRow < fdRunInfo.mNum) {
        int64_t smlaDealRowCount = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
        if (startRow + smlaDealRowCount > fdRunInfo.mNum) {
            smlaDealRowCount = fdRunInfo.mNum - startRow;
        }
        WaitFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
        if constexpr (IS_BATCH_CONSISTENCY) {
            WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
            AttentionCommon::ReducePairwiseWithLse<T, dTemplateAlign64>(
                stagingLayout, fdStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
                static_cast<uint32_t>(fdRunInfo.mStartIdx + startRow), smlaDealRowCount,
                static_cast<uint32_t>(smla35VecConstInfo.dSizeV), smlaAccumulatedO, smlaLseExpUb, smlaBlockMaxUb,
                smlaBlockSumUb, smlaPartialOFp32, smla35VecConstInfo.isSoftmaxLseEnable, softmaxLseGm,
                softmaxLseOffset + startRow, INNERCORE_FD_V_MTE2(0), INNERCORE_FD_V_MTE2(1), INNERCORE_FD_MTE2_V,
                INNERCORE_LSE_V_MTE3, INNERCORE_LSE_MTE3_V);
        } else {
            AttentionCommon::ReduceWithLse<T, dTemplateAlign64>(
                stagingLayout, fdStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
                static_cast<uint32_t>(fdRunInfo.mStartIdx + startRow), smlaDealRowCount,
                static_cast<uint32_t>(smla35VecConstInfo.dSizeV), smlaAccumulatedO, smlaLseExpUb, smlaBlockMaxUb,
                smlaBlockSumUb, smlaPartialOFp32, smla35VecConstInfo.isSoftmaxLseEnable, softmaxLseGm,
                softmaxLseOffset + startRow, INNERCORE_FD_V_MTE2(0), INNERCORE_FD_V_MTE2(1), INNERCORE_FD_MTE2_V,
                INNERCORE_LSE_V_MTE3, INNERCORE_LSE_MTE3_V);
        }
        RunInfo smla35VecRunInfo;
        smla35VecRunInfo.vec2MRealSize = smlaDealRowCount;
        smla35VecRunInfo.attentionOutOffset = attentionOutOffset + startRow * attentionOutRowStride;
        int64_t smlaVec2CalcSize = smlaDealRowCount * dTemplateAlign64;
        this->CopyOutAttentionOut(smla35VecRunInfo, smla35VecConstInfo, smlaAccumulatedO, 0, smlaVec2CalcSize);
        if constexpr (IS_BATCH_CONSISTENCY) {
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
        }
        SetFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
        startRow += smlaDealRowCount;
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo &smla35VecRunInfo,
                                                                          ConstInfo &smla35VecConstInfo,
                                                                          LocalTensor<VEC2_RES_T> &smla35Vec2ResultUb,
                                                                          int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> smlaAttenOut;
    int64_t smlaDSizeAligned64 = (int64_t)dTemplateAlign64;

    smlaAttenOut.SetAddr(smla35Vec2ResultUb.address_);
    Cast(smlaAttenOut, smla35Vec2ResultUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE2);
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE2);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = smla35VecConstInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride =
        (smlaDSizeAligned64 - smla35VecConstInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = smla35VecConstInfo.attentionOutStride;
    dataCopyParams.blockCount = smla35VecRunInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[smla35VecRunInfo.attentionOutOffset], smlaAttenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo, LocalTensor<VEC2_RES_T> &smla35Vec2ResultUb,
    int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(smla35VecRunInfo, smla35VecConstInfo, smla35Vec2ResultUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &smla35VecConstInfo)
{
    uint32_t smlaCoreNum = GetBlockNum();
    uint32_t vecCoreNum = CV_RATIO * smlaCoreNum;
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        totalOutputSize =
            smla35VecConstInfo.bSize * smla35VecConstInfo.gSize * smla35VecConstInfo.s1Size * smla35VecConstInfo.dSizeV;
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        totalOutputSize = smla35VecConstInfo.s1Size * smla35VecConstInfo.gSize * smla35VecConstInfo.dSizeV;
    }

    static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 184U * 1024U;
    static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = (32U * 1024U) / sizeof(OUTPUT_T);
    if (smlaCoreNum != 0 && totalOutputSize > 0) {
        AttentionCommon::InitOutput<OUTPUT_T, initOutputEventId, ATTEN_OUT_POP_BUF_START_ADDR,
                                    ATTEN_OUT_POP_BUF_ELE_SIZE, false>(this->attentionOutGm, totalOutputSize,
                                                                       vecCoreNum, static_cast<OUTPUT_T>(0));
    }
    if (smla35VecConstInfo.isSoftmaxLseEnable) {
        uint64_t totalReturnSoftmaxSize = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
            totalReturnSoftmaxSize = smla35VecConstInfo.bSize * smla35VecConstInfo.n2Size * smla35VecConstInfo.s1Size *
                                     smla35VecConstInfo.gSize;
        } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            totalReturnSoftmaxSize =
                smla35VecConstInfo.n2Size * smla35VecConstInfo.s1Size * smla35VecConstInfo.gSize; // (N2,T1,G)
        }
        static constexpr uint32_t LSE_POP_BUF_START_ADDR = 216U * 1024U;
        static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = (32U * 1024U) / sizeof(float);
        if (smlaCoreNum != 0 && totalReturnSoftmaxSize > 0) {
            AttentionCommon::InitOutput<float, initOutputEventId, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, false>(
                this->softmaxLseGm, totalReturnSoftmaxSize, vecCoreNum, static_cast<float>(0));
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut,
                                                                      __gm__ uint8_t *softmaxLse,
                                                                      ConstInfo &smla35VecConstInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxLseGm.SetGlobalBuffer((__gm__ T *)softmaxLse);
        if (smla35VecConstInfo.needInit == 1) {
            InitOutputSingleCore(smla35VecConstInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::InitGlobalBuffer(
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
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::SoftmaxInitBuffer(uint32_t &ubAddr)
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    constexpr uint32_t softmaxElems = softmaxBufSize / sizeof(float);
    softmaxSumBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 0};
    ubAddr += softmaxBufSize;
    softmaxSumBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 1};
    ubAddr += softmaxBufSize;
    softmaxMaxBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 0};
    ubAddr += softmaxBufSize;
    softmaxMaxBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 1};
    ubAddr += softmaxBufSize;
    if constexpr (IS_BATCH_CONSISTENCY) {
        softmaxFinalSumBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 0};
        ubAddr += softmaxBufSize;
        softmaxFinalSumBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 1};
        ubAddr += softmaxBufSize;
        softmaxFinalMaxBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 0};
        ubAddr += softmaxBufSize;
        softmaxFinalMaxBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, softmaxElems), 1};
        ubAddr += softmaxBufSize;
        batchReduceTmpUb = {LocalTensor<float>(TPosition::VECIN, ubAddr, 768),
                            0}; // 768：batchReduceTmpUb申请内存大小为768个float
        ubAddr += 768U * sizeof(float);
    } else {
        // 普通 FD 即使不向调用方返回 softmax LSE，也需要暂存 max 和 sum。
        fdLseTmpUb = {LocalTensor<float>(TPosition::VECIN, ubAddr, FD_VEC1_LSE_TMP_ELEMS), 0};
        ubAddr += FD_VEC1_LSE_TMP_ELEMS * sizeof(float);
    }
    softmaxExpBufs[0] = {LocalTensor<T>(TPosition::VECIN, ubAddr, softmaxBufSize / sizeof(T)), 0};
    ubAddr += softmaxBufSize;
    softmaxExpBufs[1] = {LocalTensor<T>(TPosition::VECIN, ubAddr, softmaxBufSize / sizeof(T)), 1};
    ubAddr += softmaxBufSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo &smla35VecConstInfo)
{
    LocalTensor<T> smlaSinksUb = this->sinksUb.tensor;
    const uint32_t smlaMaxN = smla35VecConstInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = smlaMaxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(smlaSinksUb, this->sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_SINKS_SYNC);
    WaitFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_SINKS_SYNC);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::InitLocalBuffer(ConstInfo &smla35VecConstInfo,
                                                                          uint32_t ubBaseAddr)
{
    uint32_t ubAddr = ubBaseAddr;

    SoftmaxInitBuffer(ubAddr);

    commonUb = {LocalTensor<T>(TPosition::VECIN, ubAddr, 512 / sizeof(T)), 0}; // 512 for common ub size
    ubAddr += 512;                                                             // 512 for common ub offset
    sinksUb = {LocalTensor<T>(TPosition::VECIN, ubAddr, 512 / sizeof(T)), 0};  // 512 for sinks ub size
    ubAddr += 512;                                                             // 512 for sinks ub offset
    if (this->isSinks) {
        InitSinksBuffer(smla35VecConstInfo);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        stage0OutBufs[0] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, dVTemplateType * 16U),
                            0}; // 输出缓冲区处理16个seq
        ubAddr += dVTemplateType * 16U * sizeof(KV_T);
        stage0OutBufs[1] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, dVTemplateType * 16U),
                            1}; // 输出缓冲区处理16个seq
        ubAddr += dVTemplateType * 16U * sizeof(KV_T);
    }
    if (smla35VecConstInfo.isSoftmaxLseEnable) {
        outLseUbs[0] = {LocalTensor<float>(TPosition::VECIN, ubAddr, 256 / sizeof(float)),
                        0}; // outLseBuf[0]内存申请256B
        ubAddr += 256U;
        outLseUbs[1] = {LocalTensor<float>(TPosition::VECIN, ubAddr, 256 / sizeof(float)),
                        1}; // outLseBuf[1]内存申请256B
        ubAddr += 256U;
    }

    stage1OutBufs[0] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, vec1Srcstride * s2BaseSize), 0};
    ubAddr += vec1Srcstride * s2BaseSize * sizeof(Q_T);
    stage1OutBufs[1] = {LocalTensor<Q_T>(TPosition::VECIN, ubAddr, vec1Srcstride * s2BaseSize), 1};
    ubAddr += vec1Srcstride * s2BaseSize * sizeof(Q_T);

    stage2OutBufs = {LocalTensor<T>(TPosition::VECIN, ubAddr, (s1BaseSize / CV_RATIO) * dTemplateAlign64), 0};

    // 显式 flag 初始化 (替代 AllocEventID + 初始 SetFlag)
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    if constexpr (IS_BATCH_CONSISTENCY) {
        SetFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
        SetFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
    }
    if (smla35VecConstInfo.isSoftmaxLseEnable) {
        SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
    }
    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(0));
    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(1));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(0));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(1));
    SetFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(0));
    SetFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(1));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
    if constexpr (IS_BATCH_CONSISTENCY) {
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::FreeEvent(ConstInfo &smla35VecConstInfo)
{
    if constexpr (IS_BATCH_CONSISTENCY) {
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
    }
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    if (smla35VecConstInfo.isSoftmaxLseEnable) {
        WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(0));
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_STAGE0OUT_MTE3_MTE2(1));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(0));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(1));
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(0));
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_FD_V_MTE2(1));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
    if constexpr (IS_BATCH_CONSISTENCY) {
        WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::GetExtremeValue(T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t SmlaCsaBlockVector<TEMPLATE_ARGS>::GetSmlaSeqLen(int32_t batchIndex, bool useExplicitLength,
                                                                           bool useCumulativeLength,
                                                                           GlobalTensor<int32_t> &explicitLengthGm,
                                                                           GlobalTensor<int32_t> &cumulativeLengthGm,
                                                                           int64_t fallbackLength)
{
    if (useExplicitLength) {
        return explicitLengthGm.GetValue(batchIndex);
    } else if (useCumulativeLength) {
        return cumulativeLengthGm.GetValue(batchIndex + 1) - cumulativeLengthGm.GetValue(batchIndex);
    } else {
        return fallbackLength;
    }
}
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline PhyAddrValidInfo SmlaCsaBlockVector<TEMPLATE_ARGS>::CalcPhyAddrValidInfo(
    bool isOriKv, int32_t actualS1Size, int32_t actualOriS2Size, int64_t restoredSize, ConstInfo &smla35VecConstInfo)
{
    // per-batch执行一次,  per-s1循环内不再判断maskmode
    PhyAddrValidInfo smla35ValidWindow;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        smla35ValidWindow.oriS2Act = actualOriS2Size;
        if (isOriKv) {
            if (smla35VecConstInfo.oriMaskMode == 0U) {
                smla35ValidWindow.oriTopkMode = true;
            } else if (smla35VecConstInfo.oriMaskMode == 3U) {
                smla35ValidWindow.oriRightBias = 0;
            } else {
                smla35ValidWindow.oriLeftBias = (smla35VecConstInfo.oriWinLeft == -1) ?
                                                    PhyAddrValidInfo::BIAS_UNBOUND :
                                                    smla35VecConstInfo.oriWinLeft + 1;
                smla35ValidWindow.oriRightBias = (smla35VecConstInfo.oriWinRight == -1) ?
                                                     PhyAddrValidInfo::BIAS_UNBOUND :
                                                     smla35VecConstInfo.oriWinRight;
            }
        }
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            smla35ValidWindow.cmpTopkMode = (smla35VecConstInfo.cmpMaskMode == 0U);
            smla35ValidWindow.cmpBase = restoredSize - actualS1Size + 1;
        }
    }
    return smla35ValidWindow;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t SmlaCsaBlockVector<TEMPLATE_ARGS>::CalcCurValidS2(
    uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv, GlobalTensor<int32_t> &cuSeqlensQGm,
    GlobalTensor<int32_t> &topkLengthGm, ConstInfo &smla35VecConstInfo, int32_t sparseBlockCount,
    const PhyAddrValidInfo &smla35ValidWindow)
{
    bool smlaTopkMode = false;
    bool smlaHasTopk = false;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            smlaTopkMode = smla35ValidWindow.oriTopkMode;
            smlaHasTopk = smla35VecConstInfo.hasOriTopkLength;
        }
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            smlaTopkMode = smla35ValidWindow.cmpTopkMode;
            smlaHasTopk = smla35VecConstInfo.hasCmpTopkLength;
        }
    }
    if (smlaTopkMode) {
        uint64_t topkIdx = (LAYOUT_T == SMLA_LAYOUT::TND) ? (cuSeqlensQGm.GetValue(bIdx) + s1Idx) :
                                                            (bIdx * smla35VecConstInfo.s1Size + s1Idx);
        int32_t topkLen = smlaHasTopk ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
        return Min(topkLen, sparseBlockCount);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            int64_t thr = smla35ValidWindow.oriS2Act - actualS1Size + 1 + s1Idx;
            int64_t leftBound = Max(thr - smla35ValidWindow.oriLeftBias, 0);
            int64_t rightBound =
                Min(thr + smla35ValidWindow.oriRightBias, static_cast<int64_t>(smla35ValidWindow.oriS2Act));
            return Min(static_cast<int32_t>(Max(0, rightBound - leftBound)), sparseBlockCount);
        }
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t numerator = Max(smla35ValidWindow.cmpBase + s1Idx, 0);
        return Min(sparseBlockCount,
                   static_cast<int32_t>(numerator / static_cast<int32_t>(smla35VecConstInfo.cmpRatio)));
    }
    return 0;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::GetKVPhyAddrForKvType(
    uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
    bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> &actualSeqQlenGm,
    GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &actualSeqKvlenGm, GlobalTensor<int32_t> &cuSeqlensKvGm,
    GlobalTensor<int32_t> &topkLengthGm, GlobalTensor<int32_t> &cmpResidualKvGm, ConstInfo &smla35VecConstInfo,
    GlobalTensor<int32_t> &blockTableGm, GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm,
    uint32_t kvStride, uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount,
    uint32_t alignedSparseBlockCount, bool isOriKv)
{
    static constexpr uint16_t s2NumPerLoop = 128;
    static constexpr uint32_t vecCoreNum = IS_SPLIT_G ? 4 : 2;
    uint32_t smlaVecCoreIdx = IS_SPLIT_G ? smla35VecConstInfo.aivIdx % 4 : smla35VecConstInfo.aivIdx % 2;
    uint32_t phyAddrUb = 0;
    int16_t shiftRightNum = 0;
    LocalTensor<int32_t> blkTableUb;

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        int32_t blkSize = static_cast<int32_t>(blockSize);
        while (blkSize > 1) {
            blkSize >>= 1;
            shiftRightNum++;
        }
        blkTableUb = LocalTensor<int32_t>(TPosition::VECIN, phyAddrUb, maxBlockNumPerBatch);
        phyAddrUb = CeilAlign(phyAddrUb + maxBlockNumPerBatch * sizeof(int32_t), BUFFER_SIZE_BYTE_32B);
    }
    LocalTensor<int32_t> sparseIdxUb(TPosition::VECIN, phyAddrUb, alignedSparseBlockCount);
    phyAddrUb += alignedSparseBlockCount * sizeof(int32_t);
    LocalTensor<uint32_t> kvPhyAddrUb(TPosition::VECIN, phyAddrUb, alignedSparseBlockCount * 2); // 2 for ori/cmp kv

    // 第一遍: 统计totalValidS1
    int64_t smlaTotalValidS1 = 0;
    uint32_t smlaTmpGS1Start = gS1StartIdx;
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx; ++bIdx) {
        bool smlaLastBN = (bIdx == bN2EndIdx - 1);
        int32_t smlaActualS1Size = GetSmlaSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm,
                                                 smla35VecConstInfo.s1Size);
        int32_t smlaS1End = smlaActualS1Size;
        if (smlaLastBN && nextGs1Idx != 0) {
            smlaS1End = nextGs1Idx;
        }

        int64_t smlaBS1IdxBase = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            smlaBS1IdxBase = hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : smla35VecConstInfo.s1Size * bIdx;
        } else {
            smlaBS1IdxBase = smla35VecConstInfo.s1Size * bIdx;
        }

        int64_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && smla35VecConstInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSmlaSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                     cuSeqlensKvGm, smla35VecConstInfo.cmpS2Size);
                int64_t residual = (smla35VecConstInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
                restoredSize =
                    static_cast<int64_t>(actualKvSize) * static_cast<int64_t>(smla35VecConstInfo.cmpRatio) + residual;
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && smla35VecConstInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSmlaSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                cuSeqlensKvGm, smla35VecConstInfo.s2Size);
            }
        }
        PhyAddrValidInfo smla35ValidWindow =
            CalcPhyAddrValidInfo(isOriKv, smlaActualS1Size, actualOriS2Size, restoredSize, smla35VecConstInfo);

        for (int32_t s1Idx = smlaTmpGS1Start; s1Idx < smlaS1End; ++s1Idx) {
            int32_t smlaCurValidS2 =
                CalcCurValidS2(bIdx, s1Idx, smlaActualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm, smla35VecConstInfo,
                               static_cast<int32_t>(sparseBlockCount), smla35ValidWindow);
            if (smlaCurValidS2 > 0) {
                smlaTotalValidS1++;
            }
        }
        smlaTmpGS1Start = 0;
    }

    int64_t smlaS1PerVecCore = smlaTotalValidS1 / vecCoreNum;
    int64_t smlaS1Tail = smlaTotalValidS1 % vecCoreNum;
    int64_t smlaCurStart = smlaS1PerVecCore * smlaVecCoreIdx + Min((int64_t)smlaVecCoreIdx, smlaS1Tail);
    int64_t smlaCurCount = smlaS1PerVecCore + (smlaVecCoreIdx < (uint32_t)smlaS1Tail ? 1 : 0);

    if (smlaCurCount == 0) {
        return;
    }

    // 第二遍: 实际计算
    int64_t smlaValidCounter = 0;
    int64_t smlaProcessedCount = 0;
    smlaTmpGS1Start = gS1StartIdx;
    bool smlaDone = false;

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        SetFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
    }
    SetFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
    SetFlag<AscendC::HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx && !smlaDone; ++bIdx) {
        bool smlaLastBN = (bIdx == bN2EndIdx - 1);
        int32_t smlaActualS1Size = GetSmlaSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm,
                                                 smla35VecConstInfo.s1Size);
        int64_t bS1Idx = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            bS1Idx = hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : smla35VecConstInfo.s1Size * bIdx;
        } else {
            bS1Idx = smla35VecConstInfo.s1Size * bIdx;
        }

        int32_t smlaS1End = smlaActualS1Size;
        if (smlaLastBN && nextGs1Idx != 0) {
            smlaS1End = nextGs1Idx;
        }

        // per-batch 参数预计算
        uint32_t kvPrefix = 0;
        uint32_t bS2BaseLow = 0;
        uint32_t bS2BaseHigh = 0;
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
            kvPrefix = static_cast<uint32_t>(cuSeqlensKvGm.GetValue(bIdx));
        } else {
            uint32_t s2Size = isOriKv ? static_cast<uint32_t>(smla35VecConstInfo.s2Size) :
                                        static_cast<uint32_t>(smla35VecConstInfo.cmpS2Size);
            uint64_t bS2Base = static_cast<uint64_t>(bIdx) * s2Size * static_cast<uint64_t>(smla35VecConstInfo.dSize);
            bS2BaseLow = static_cast<uint32_t>(bS2Base);
            bS2BaseHigh = static_cast<uint32_t>(bS2Base >> 32U);
        }

        int64_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && smla35VecConstInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSmlaSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                     cuSeqlensKvGm, smla35VecConstInfo.cmpS2Size);
                int64_t residual = (smla35VecConstInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
                restoredSize =
                    static_cast<int64_t>(actualKvSize) * static_cast<int64_t>(smla35VecConstInfo.cmpRatio) + residual;
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && smla35VecConstInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSmlaSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                cuSeqlensKvGm, smla35VecConstInfo.s2Size);
            }
        }
        PhyAddrValidInfo smla35ValidWindow =
            CalcPhyAddrValidInfo(isOriKv, smlaActualS1Size, actualOriS2Size, restoredSize, smla35VecConstInfo);

        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
            AttentionCommon::CopyPaTableToUb(blkTableUb, bIdx, blockTableGm, maxBlockNumPerBatch);
            SetFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
            WaitFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
        }

        for (int32_t s1Idx = smlaTmpGS1Start; s1Idx < smlaS1End; ++s1Idx) {
            int32_t smlaCurValidS2 =
                CalcCurValidS2(bIdx, s1Idx, smlaActualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm, smla35VecConstInfo,
                               static_cast<int32_t>(sparseBlockCount), smla35ValidWindow);
            if (smlaCurValidS2 <= 0) {
                continue;
            }

            if (smlaValidCounter < smlaCurStart || smlaValidCounter >= smlaCurStart + smlaCurCount) {
                smlaValidCounter++;
                continue;
            }
            smlaValidCounter++;

            uint16_t smlaS2Loop = (smlaCurValidS2 + s2NumPerLoop - 1) / s2NumPerLoop;
            int32_t smlaS2Tail = smlaCurValidS2 - (smlaS2Loop - 1) * s2NumPerLoop;
            WaitFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
            AttentionCommon::CopySparseIdxToUb(sparseIdxUb, bS1Idx, s1Idx, smlaCurValidS2, sparseIndicesGm,
                                               sparseBlockCount);
            SetFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_PHYADDR_SPARSEIDX_READY);

            WaitFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_PHYADDR_SPARSEIDX_READY);
            WaitFlag<AscendC::HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                AttentionCommon::GetKVPhyAddrVFPa<uint32_t>(
                    kvPhyAddrUb, sparseIdxUb, blkTableUb, smlaS2Loop, smlaS2Tail, blockSize, shiftRightNum,
                    smla35VecConstInfo.sparseBlockSize, static_cast<uint32_t>(smla35VecConstInfo.dSize), kvStride);
            } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
                AttentionCommon::GetKVPhyAddrVFTnd<uint32_t>(kvPhyAddrUb, sparseIdxUb, smlaS2Loop, smlaS2Tail,
                                                             smla35VecConstInfo.sparseBlockSize,
                                                             static_cast<uint32_t>(smla35VecConstInfo.dSize), kvPrefix);
            } else {
                AttentionCommon::GetKVPhyAddrVFBsnd<uint32_t>(
                    kvPhyAddrUb, sparseIdxUb, smlaS2Loop, smlaS2Tail, smla35VecConstInfo.sparseBlockSize,
                    static_cast<uint32_t>(smla35VecConstInfo.dSize), bS2BaseLow, bS2BaseHigh);
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
            SetFlag<AscendC::HardEvent::V_MTE3>(INNERCORE_PHYADDR_KVADDR_READY);
            WaitFlag<AscendC::HardEvent::V_MTE3>(INNERCORE_PHYADDR_KVADDR_READY);
            AttentionCommon::CopyPhyAddrToGm(kvPhyAddrUb, bS1Idx, s1Idx, smlaCurValidS2, s2NumPerLoop, phyAddrGm,
                                             alignedSparseBlockCount);
            SetFlag<AscendC::HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);

            smlaProcessedCount++;
            if (smlaProcessedCount >= smlaCurCount) {
                smlaDone = true;
                break;
            }
        }
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            SetFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
        }
        smlaTmpGS1Start = 0;
    }
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        WaitFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
    WaitFlag<AscendC::HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SmlaCsaBlockVector<TEMPLATE_ARGS>::GetKVPhyAddr(
    uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx,
    bool hasActualSeqQlen, bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
    GlobalTensor<int32_t> actualSeqOriKvlenGm, GlobalTensor<int32_t> cuSeqlensOriKvGm,
    GlobalTensor<int32_t> oriTopkLengthGm, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
    GlobalTensor<int32_t> actualSeqCmpKvlenGm, GlobalTensor<int32_t> cuSeqlensCmpKvGm,
    GlobalTensor<int32_t> cmpTopkLengthGm, GlobalTensor<int32_t> cmpResidualKvGm, GlobalTensor<int32_t> actualSeqQlenGm,
    GlobalTensor<int32_t> cuSeqlensQGm, __gm__ uint8_t *workspace, ConstInfo &smla35VecConstInfo)
{
    if (hasLoad == 0) {
        return;
    }

    // GM分配: ori在前, cmp在后
    int64_t smlaV0TotalOffset = 0;
    uint32_t smlaV0ResSize = smla35VecConstInfo.s2BaseSize * smla35VecConstInfo.dSize * sizeof(Q_T);
    if constexpr (IS_SPLIT_G) {
        smlaV0TotalOffset = smlaV0ResSize * 3 * (GetBlockNum() >> 1U);
    } else {
        smlaV0TotalOffset = smlaV0ResSize * 3 * GetBlockNum();
    }

    // SMLA特有: 加上s2RealBuf大小
    constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
    constexpr uint32_t S2_REAL_BUF_LEN = 128;
    smlaV0TotalOffset += TRIPLE_BUFFER_NUM * S2_REAL_BUF_LEN * sizeof(int32_t) * GetBlockNum();

    uint32_t totalBS1 = (LAYOUT_T == SMLA_LAYOUT::TND) ? smla35VecConstInfo.s1Size :
                                                         (smla35VecConstInfo.bSize * smla35VecConstInfo.s1Size);

    uint64_t oriPhyAddrSize = 0;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriPhyAddrSize =
            static_cast<uint64_t>(totalBS1) * smla35VecConstInfo.alignedOriSparseBlockCount * sizeof(int64_t);
        this->oriKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + smlaV0TotalOffset));
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        uint64_t cmpPhyAddrSize =
            static_cast<uint64_t>(totalBS1) * smla35VecConstInfo.alignedCmpSparseBlockCount * sizeof(int64_t);
        this->cmpKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + smlaV0TotalOffset + oriPhyAddrSize));
    }

    // ori部分 (先计算)
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        GetKVPhyAddrForKvType(
            bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ, hasActualSeqOriKvlen,
            hasCuSeqlensOriKv, actualSeqQlenGm, cuSeqlensQGm, actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm,
            cmpResidualKvGm, smla35VecConstInfo, oriBlockTableGm, oriSparseIndicesGm, oriKvPhyAddrGm,
            smla35VecConstInfo.oriKvStride, smla35VecConstInfo.oriBlockSize, smla35VecConstInfo.oriMaxBlockNumPerBatch,
            smla35VecConstInfo.oriSparseBlockCount, smla35VecConstInfo.alignedOriSparseBlockCount, true);
    }

    // cmp部分 (后计算)
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        GetKVPhyAddrForKvType(
            bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ, hasActualSeqCmpKvlen,
            hasCuSeqlensCmpKv, actualSeqQlenGm, cuSeqlensQGm, actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm,
            cmpResidualKvGm, smla35VecConstInfo, cmpBlockTableGm, cmpSparseIndicesGm, cmpKvPhyAddrGm,
            smla35VecConstInfo.cmpKvStride, smla35VecConstInfo.cmpBlockSize, smla35VecConstInfo.cmpMaxBlockNumPerBatch,
            smla35VecConstInfo.cmpSparseBlockCount, smla35VecConstInfo.alignedCmpSparseBlockCount, false);
    }
}

TEMPLATES_DEF
class CSABlockVecDummy {
public:
    __aicore__ inline CSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                       ConstInfo &smla35VecConstInfo)
    {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv)
    {}
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV) {};
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &fdStaging) {}
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC> &intraCoreCombine,
                                              Buffer<BufferType::GM, SyncType::NO_SYNC> &crossCoreCombine)
    {}
    __aicore__ inline void InitLocalBuffer(ConstInfo &smla35VecConstInfo, uint32_t ubBaseAddr) {}
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo) {}
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &smla35VecConstInfo) {}
    __aicore__ inline void ProcessVec1(StaticBuffer<Q_T> &outputBuf, StaticBuffer<T> &bmm1ResBuf,
                                       RunInfo &smla35VecRunInfo, ConstInfo &smla35VecConstInfo)
    {}
    __aicore__ inline void ProcessVec2(StaticBuffer<T> &bmm2ResBuf, RunInfo &smla35VecRunInfo,
                                       ConstInfo &smla35VecConstInfo)
    {}
    __aicore__ inline void FreeEvent(ConstInfo &smla35VecConstInfo) {}
};
} // namespace SMLAKernel
#endif // SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH35_H
