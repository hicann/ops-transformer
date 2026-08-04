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
 * \file mixed_quant_sparse_flash_mla_csa_block_vector.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "util_regbase.h"
#include "mixed_quant_sparse_flash_mla_common_arch35.h"
using AscendC::Reg::StoreDist;
#if __has_include("../../../sparse_flash_mla/op_kernel/arch35/common/flash_decode.h")
#include "../../../sparse_flash_mla/op_kernel/arch35/common/flash_decode.h"
#else
#include "../../sparse_flash_mla/arch35/common/flash_decode.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_flash_decode.h")
#include "../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#else
#include "../common/arch35/vf/vf_flash_decode.h"
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

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace optiling;
using namespace optiling::detail;
using namespace regbaseutil;
using namespace matmul;
using AttentionCommon::FdRunInfo;

namespace BaseApi {
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
    static constexpr uint32_t dVTemplateTypeInput = 608; // Dsize
    static constexpr uint32_t dCombineBytes = 576;       // rope(64 * 2) + nope(448)
    static constexpr uint32_t scaleBytes = 8;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr uint32_t uint64Touint8 = sizeof(uint64_t) / sizeof(uint8_t);
    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
    bool isSoftmaxLseGmValid = false;                 // 标记 softmaxLseGm 是否已有效 SetGlobalBuffer
    // ==================== Functions ======================
    __aicore__ inline CSABlockVec(){};
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedOriKv,
                                        __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                          TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
                if (cuSeqlensCmpKv != nullptr) {
                    cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
                }
            }
            if (sequsedOriKv != nullptr) {
                actualSeqOriKvGm.SetGlobalBuffer((__gm__ int32_t *)sequsedOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                          TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
                if (sequsedCmpKv != nullptr) {
                    actualSeqCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)sequsedCmpKv);
                }
                if (cmpResidualKv != nullptr) {
                    cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKv);
                }
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &fdStaging)
    {
        fdStagingBase = fdStaging.template GetTensor<uint8_t>().GetPhyAddr(0);
        stagingOutGm = fdStaging.template GetTensor<float>();
    }
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &intraCoreCombine,
        Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &crossCoreCombine)
    {
        intraCoreCombineBase = intraCoreCombine.template GetTensor<uint8_t>().GetPhyAddr(0);
        intraCoreCombineGm = intraCoreCombine.template GetTensor<float>();
        crossCoreCombineBase = crossCoreCombine.template GetTensor<uint8_t>().GetPhyAddr(0);
        crossCoreCombineGm = crossCoreCombine.template GetTensor<float>();
        fdStagingBase = crossCoreCombineBase;
        stagingOutGm = crossCoreCombineGm;
    }
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo);

private:
    __aicore__ inline uint32_t GetStagingSlotNum(bool isInner = false) const
    {
        if constexpr (!IS_BATCH_CONSISTENCY) {
            if constexpr (IS_SPLIT_G) {
                return AttentionCommon::FD_MAX_S2_SPLIT_NUM * (GetBlockNum() >> 1U);
            } else {
                return AttentionCommon::FD_MAX_S2_SPLIT_NUM * GetBlockNum();
            }
        } else {
            if constexpr (IS_SPLIT_G) {
                if (isInner) {  // 核内
                    return GetBlockNum();
                }
                return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * (GetBlockNum() >> 1U);  // 核间
            } else {
                if (isInner) {  // 核内
                    return GetBlockNum() << 1U;
                }
                return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * GetBlockNum();  // 核间
            }
        }
    }

    __aicore__ inline uint32_t GetCrossCoreWorkspaceIdx(const RunInfo &runInfo) const
    {
        uint32_t workspaceIdx = static_cast<uint32_t>(runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx);
        return workspaceIdx;
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
                                           const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessNotSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                              Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                              const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetKeyOffset(int64_t s2Idx, int64_t &realKeyOffset, int64_t &realScaleOffset,
                                        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int32_t *tokenData, int64_t s2IdxInBase, const RunInfo &runInfo,
                                           ConstInfo &constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
                                             int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int32_t *tokenData,
                                              const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void DequantKv(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                     int64_t s2ProcessBaseSize, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                          LocalTensor<Q_T> antiKvTensorAsB16, int64_t v0Loop, int64_t dealRow,
                                          int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutMrgeResult(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                             int64_t mte2Size, int64_t mte3Size, int64_t s2keyOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                          int64_t scaleOffset);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
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
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriSparseIndicesGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqOriKvGm;
    GlobalTensor<int32_t> actualSeqCmpKvGm;
    GlobalTensor<int32_t> cmpResidualKvGm;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId;
    TEventID stageMte3ToVId;
    TEventID v0PairMte3ToVId;
    TEventID vToMte3Id;
    TEventID intraLseMte3ToMte2Id[2];
    TEventID intraAttnOutMte3ToMte2Id[2];
    TEventID mte3ToVLseOutId;
    TEventID vToMte3LseOutId;
    TEventID mte2ToVV0Id[2];
    TEventID vToMte2V0Id[2];
    TEventID intraPartialOVToMte2Id;
    TEventID vToMte3V0Id[2];
    TEventID mte3ToVV0Id[2];
    TEventID reduceMaxSumVToMte2Id;
    TEventID reduceMte2ToVId;
    TEventID fdVToMte2Id[2];
    TEventID fdMte2ToVId;
    TEventID fdMte3ToVId;
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxFinalMaxBuf[2];
    TBuf<> softmaxFinalSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> batchReduceTmpBuf;
    TBuf<> dequantScaleBuff;
    TBuf<> outLseBuf[2];
    TBuf<> stage0InBuf[2];
    TBuf<> stage0OutBuf[2];
    TBuf<> vselrIndexesBuf[2];
    AttentionCommon::FdBuffers<TBuf<>> fdBuffers;
    uint32_t pingPongV0 = 0;

    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
    __gm__ uint8_t *fdStagingBase = nullptr;
    GlobalTensor<float> stagingOutGm;
    __gm__ uint8_t *intraCoreCombineBase = nullptr;
    GlobalTensor<float> intraCoreCombineGm;
    __gm__ uint8_t *crossCoreCombineBase = nullptr;
    GlobalTensor<float> crossCoreCombineGm;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int32_t *tokenData, int64_t s2IdxInBase,
                                                                   const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t sparseBlockCount = 0;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        sparseBlockCount = constInfo.cmpSparseBlockCount;
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        sparseBlockCount = constInfo.oriSparseBlockCount;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (runInfo.isCmp) {
            sparseBlockCount = constInfo.cmpSparseBlockCount;
            curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
        } else {
            sparseBlockCount = constInfo.oriSparseBlockCount;
        }
    }
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * sparseBlockCount;
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * sparseBlockCount + runInfo.s1oIdx * sparseBlockCount;
    }

    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    for (uint64_t i = 0; i < 8; ++i) {
        uint64_t idx = topkBS1Idx + runInfo.s2StartIdx + topkKIdx + i;
        if (likely((topkKIdx + i < sparseBlockCount) && (s2IdxInBase + i < sparseS2End))) {
            tokenData[i] = sparseIndicesGm.GetValue(idx);
        } else {
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKeyOffset(int64_t s2Idx, int64_t &realKeyOffset,
                                                                int64_t &realScaleOffset, const RunInfo &runInfo,
                                                                ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return;
    }
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        int32_t physBlockId = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx);
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
            realKeyOffset = physBlockId * paBlockStride + blkTableOffset * constInfo.dSizeVInput;
        } else {
            realKeyOffset = physBlockId * paBlockStride + blkTableOffset * constInfo.n2Size * dCombineBytes +
                            (uint64_t)(runInfo.n2oIdx * dCombineBytes); // BlockNum, BlockSize, N(1), D(ROPE+NOPE = 576)
            realScaleOffset = physBlockId * paBlockStride +
                              static_cast<int64_t>(blockSize) * constInfo.n2Size * dCombineBytes +
                              blkTableOffset * scaleBytes;
        }
    } else if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        int64_t tPrefix =
            runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx);
        realKeyOffset =
            (tPrefix + s2Idx) * constInfo.n2Size * constInfo.dSizeVInput + runInfo.n2oIdx * constInfo.dSizeVInput;
    } else {
        if (runInfo.isCmp) {
            realKeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.cmpS2Size * constInfo.dSizeVInput +
                            s2Idx * constInfo.dSizeVInput; // BSN(1)D
        } else {
            realKeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.s2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.s2Size * constInfo.dSizeVInput +
                            s2Idx * constInfo.dSizeVInput; // BSN(1)D
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                  int64_t keyOffset, int64_t scaleOffset)
{
    if (keyOffset < 0) {
        return;
    }
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.dstStride = 0;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<KV_T> padParams;
        // 当前仅支持COMBINE模式
        uint32_t combineBytes = dVTemplateTypeInput * sizeof(KV_T);
        intriParams.blockLen = combineBytes;
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
    } else {
        // 当前仅支持COMBINE模式
        uint32_t combineDim = dVTemplateTypeInput / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(dVTemplateTypeInput, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        DataCopyExtParams intriParams;
        intriParams.blockLen = dCombineBytes;
        intriParams.blockCount = 1;
        intriParams.dstStride = 0;
        intriParams.srcStride = 0;

        DataCopyPadExtParams<KV_T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);

        DataCopyExtParams scaleParams;
        DataCopyPadExtParams<KV_T> scalePadParams;
        scaleParams.blockCount = 1;
        scaleParams.blockLen = scaleBytes;
        scaleParams.srcStride = 0;
        scaleParams.dstStride = 0;
        scalePadParams.isPad = false;
        scalePadParams.leftPadding = 0;
        scalePadParams.rightPadding = 0;
        scalePadParams.paddingValue = 0;
        if (scaleOffset >= 0) {
            DataCopyPad(kvInUb[startRow * combineDimAlign + dCombineBytes], keyGm[scaleOffset], scaleParams,
                        scalePadParams);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t CSABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                      int32_t *tokenData, const RunInfo &runInfo,
                                                                      ConstInfo &constInfo)
{
    int64_t s2IdLimit = runInfo.s2RealSize;
    s2IdLimit = (runInfo.s2RealSize - runInfo.actualS1Size + runInfo.s1oIdx + 1) / constInfo.cmpRatio;
    uint32_t dealRow = 0;
    for (uint32_t i = 0; i < 8; i += 2) {
        int64_t keyOffset0 = -1;
        int64_t keyOffset1 = -1;
        int64_t scaleOffset0 = -1;
        int64_t scaleOffset1 = -1;
        GetKeyOffset(tokenData[i], keyOffset0, scaleOffset0, runInfo, constInfo);
        GetKeyOffset(tokenData[i + 1], keyOffset1, scaleOffset1, runInfo, constInfo);
        if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
            return dealRow;
        }
        uint32_t combineBytes;
        int64_t keySrcStride;
        int64_t scaleSrcStride;
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
            combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
            keySrcStride =
                (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - combineBytes;
        } else {
            keySrcStride =
                (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - dCombineBytes;
            scaleSrcStride =
                (scaleOffset0 > scaleOffset1 ? (scaleOffset0 - scaleOffset1) : (scaleOffset1 - scaleOffset0)) -
                scaleBytes;
        }
        if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || constInfo.sparseBlockSize > 1) {
            // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
            CopyInSingleKv(kvInUb, startRow, keyOffset0, scaleOffset0);
            CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, scaleOffset1);
        } else {
            DataCopyExtParams intriParams;
            intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
                intriParams.blockLen = combineBytes;
                intriParams.dstStride = 0;
            } else {
                intriParams.blockLen = dCombineBytes;
                intriParams.dstStride = (dVTemplateTypeInput - dCombineBytes) / DATABLOCK_BYTES;
            }
            intriParams.srcStride = keySrcStride;
            DataCopyPadExtParams<KV_T> padParams;

            int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
            if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
                keyOffset = keyOffset1;
            }

            // 当前仅支持COMBINE模式
            uint32_t combineDim;
            uint32_t combineDimAlign;
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
                combineDim = combineBytes / sizeof(KV_T);
                combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            } else {
                combineDim = dVTemplateTypeInput / sizeof(KV_T);
                combineDimAlign = CeilAlign(dVTemplateTypeInput, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            }
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::NONCONTIGUOUS) {
                DataCopyExtParams scaleParams;
                DataCopyPadExtParams<KV_T> scalePadParams;
                scaleParams.blockCount = 1;
                scaleParams.blockLen = scaleBytes;
                scaleParams.srcStride = 0;
                scaleParams.dstStride = 0;
                scalePadParams.isPad = false;
                scalePadParams.leftPadding = 0;
                scalePadParams.rightPadding = 0;
                scalePadParams.paddingValue = 0;

                if (scaleOffset0 >= 0) {
                    DataCopyPad(kvInUb[startRow * combineDimAlign + dCombineBytes], keyGm[scaleOffset0], scaleParams,
                                scalePadParams);
                }
                if (scaleOffset1 >= 0) {
                    DataCopyPad(kvInUb[(startRow + 1) * combineDimAlign + dCombineBytes], keyGm[scaleOffset1],
                                scaleParams, scalePadParams);
                }
            }
        }
        dealRow += (keyOffset0 >= 0) + (keyOffset1 >= 0);
        startRow += 2;
    }
    return dealRow;
}

// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float *ubDstAddr, __ubuf__ int8_t *ubSrcAddr, uint32_t dealRowCount)
{
    MicroAPI::RegTensor<fp8_e8m0_t> vScale0;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res0;
    MicroAPI::RegTensor<float> vScalefp32Res0;
    __ubuf__ int8_t *ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float *ubDstAddrTmp = ubDstAddr;
    MicroAPI::MaskReg bf16TypeMaskAll = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t> &)vScale0, ubScaleSrcAddrTemp, 608);

        MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        MicroAPI::StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, vScalefp32Res0, 64,
                                                                             bf16TypeMaskAll);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void CastScale(LocalTensor<float> &outputUb, LocalTensor<KV_T> &inputUb, uint32_t dealRowCount)
{
    __ubuf__ float *ubDstAddr = (__ubuf__ float *)(outputUb.GetPhyAddr());
    __ubuf__ int8_t *ubScaleAddr = (__ubuf__ int8_t *)(inputUb[448 + 64 * 2].GetPhyAddr());
    CastScaleImpl<Q_T, KV_T>(ubDstAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ Q_T *ubKRopeNzAddr, __ubuf__ int8_t *ubSrcAddr,
                                        __ubuf__ Q_T *ubDstAddr, __ubuf__ Q_T *ubScaleSrcAddr,
                                        __ubuf__ int8_t *ubKRopeAddr, uint32_t dealRowCount)
{
    uint32_t combineDim = 608; // Dsize，32对齐
    MicroAPI::RegTensor<KV_T> vKvData0;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<Q_T> vScale0Bf16;
    MicroAPI::RegTensor<Q_T> vScale1Bf16;
    MicroAPI::RegTensor<half> vKvDataHalf0;
    MicroAPI::RegTensor<half> vKvDataHalf1;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<Q_T> vMulRes0;
    MicroAPI::RegTensor<Q_T> vMulRes1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vCastResPack0;
    MicroAPI::RegTensor<Q_T> vCastResPack1;
    MicroAPI::RegTensor<int8_t> vKvRope;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskHalf = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp16MaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    uint32_t blockStride = 17; // +1 to solve bank confict
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;

    // tilesize is 64, deal 128 b8 kv, deal 2 fp32 scale
    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ Q_T *ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            // load scale
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t> &)vKvData0, ubSrcTemp, tileSize);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t> &)vKvData1, ubSrcTemp, combineDim - tileSize);

            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(
                (MicroAPI::RegTensor<bfloat16_t> &)vScale0Bf16, ubScaleSrcAddrTemp + i * (combineDim / 2));
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(
                (MicroAPI::RegTensor<bfloat16_t> &)vScale1Bf16, ubScaleSrcAddrTemp + 1 + i * (combineDim / 2));

            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res1, vKvData1, fp32MaskAll);

            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, fp16MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vCastFp32Res1, fp16MaskAll);

            MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastRes0, vScale0Bf16, fp16MaskAll);
            MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes1, vCastRes1, vScale1Bf16, fp16MaskAll);

            MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vMulRes0, vMulRes1);

            MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }

    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop; // 偏移已经处理的循环次数
    __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + lastLoopOffset * kvNumPerLoop;
    __ubuf__ Q_T *ubScaleSrcAddrTemp = ubScaleSrcAddr + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + lastLoopOffset * kvNumPerLoop * blockStride;
    MicroAPI::Duplicate(vCastRes1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            (MicroAPI::RegTensor<int8_t> &)vKvRope, ubKRopeAddr, combineDim);
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t> &)vKvData0, ubSrcTemp, combineDim);
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(
            (MicroAPI::RegTensor<bfloat16_t> &)vScale0Bf16, ubScaleSrcAddrTemp + i * (combineDim / 2));
        MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
        MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, fp16MaskAll);
        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastRes0, vScale0Bf16, fp16MaskAll);
        MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vMulRes0, vCastRes1);

        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubKRopeNzAddr, (MicroAPI::RegTensor<Q_T> &)vKvRope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T> &kRopeUbNz, LocalTensor<Q_T> &outputUb,
                                          LocalTensor<KV_T> &inputUb, LocalTensor<Q_T> &scaleUb,
                                          LocalTensor<int8_t> &kRopeUb, uint32_t dealRowCount)
{
    const uint32_t nopeDim = 448;
    const uint32_t ropeDim = 64;
    __ubuf__ int8_t *ubSrcAddr = (__ubuf__ int8_t *)(inputUb[64 * sizeof(Q_T)].GetPhyAddr());
    // sizeof(bf16) / sizeof(fp8) = 2
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)(scaleUb[ropeDim + nopeDim / 2].GetPhyAddr());
    __ubuf__ int8_t *ubKRopeAddr = (__ubuf__ int8_t *)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(outputUb.GetPhyAddr());
    __ubuf__ Q_T *ubKRopeNzAddr = (__ubuf__ Q_T *)(kRopeUbNz.GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr, dealRowCount);
}

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448_FloatScale(__ubuf__ Q_T *ubKRopeNzAddr, __ubuf__ int8_t *ubSrcAddr,
                                                   __ubuf__ Q_T *ubDstAddr, __ubuf__ float *ubScaleSrcAddr,
                                                   __ubuf__ int8_t *ubKRopeAddr, uint32_t dealRowCount)
{
    uint32_t combineDim = 608;
    MicroAPI::RegTensor<KV_T> vKvData0;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<float> vScale0;
    MicroAPI::RegTensor<float> vScale1;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<float> vMulRes0;
    MicroAPI::RegTensor<float> vMulRes1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vCastResPack0;
    MicroAPI::RegTensor<Q_T> vCastResPack1;
    MicroAPI::RegTensor<int8_t> vKvRope;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskHalf = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t blockStride = 17;
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;

    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ float *ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t> &)vKvData0, ubSrcTemp, tileSize);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t> &)vKvData1, ubSrcTemp, combineDim - tileSize);

            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<float> &)vScale0, ubScaleSrcAddrTemp, 1);
            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<float> &)vScale1, ubScaleSrcAddrTemp, tileSize - 1);

            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res1, vKvData1, fp32MaskAll);

            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes1, vCastFp32Res1, vScale1, fp32MaskAll);

            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vMulRes0, fp32MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vMulRes1, fp32MaskAll);

            MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

            MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }

    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop;
    __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + lastLoopOffset * kvNumPerLoop;
    __ubuf__ float *ubScaleSrcAddrTemp = ubScaleSrcAddr + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + lastLoopOffset * kvNumPerLoop * blockStride;
    MicroAPI::Duplicate(vCastRes1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            (MicroAPI::RegTensor<int8_t> &)vKvRope, ubKRopeAddr, combineDim);
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t> &)vKvData0, ubSrcTemp, combineDim);
        MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            (MicroAPI::RegTensor<float> &)vScale0, ubScaleSrcAddrTemp, tileSize);
        MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
        MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vMulRes0, fp32MaskAll);
        MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubKRopeNzAddr, (MicroAPI::RegTensor<Q_T> &)vKvRope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448_FloatScale(LocalTensor<Q_T> &kRopeUbNz, LocalTensor<Q_T> &outputUb,
                                                     LocalTensor<KV_T> &inputUb, LocalTensor<float> &scaleUb,
                                                     LocalTensor<int8_t> &kRopeUb, uint32_t dealRowCount)
{
    __ubuf__ int8_t *ubSrcAddr = (__ubuf__ int8_t *)(inputUb.GetPhyAddr());
    __ubuf__ int8_t *ubKRopeAddr = (__ubuf__ int8_t *)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(outputUb.GetPhyAddr());
    __ubuf__ Q_T *ubKRopeNzAddr = (__ubuf__ Q_T *)(kRopeUbNz.GetPhyAddr());
    __ubuf__ float *ubScaleAddr = (__ubuf__ float *)(scaleUb.GetPhyAddr());

    AntiquantVFImplFp8D448_FloatScale<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr,
                                                 dealRowCount);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
                                                             LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                                             int64_t s2ProcessBaseSize, ConstInfo &constInfo)
{
    LocalTensor<int8_t> kRopeUb;
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (16 + 1)];
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        kRopeUb = srcTensor.template ReinterpretCast<int8_t>();
        LocalTensor<Q_T> scaleUb = srcTensor.template ReinterpretCast<bfloat16_t>();
        AntiquantVFFp8D448<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, scaleUb, kRopeUb, dealRow);
    } else {
        LocalTensor<float> floatScale = dequantScaleBuff.Get<float>();
        kRopeUb = srcTensor[448].template ReinterpretCast<int8_t>();
        CastScale<Q_T, KV_T>(floatScale, srcTensor, dealRow);
        AntiquantVFFp8D448_FloatScale<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, floatScale, kRopeUb, dealRow);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, LocalTensor<Q_T> antiKvTensorAsB16,
    int64_t v0Loop, int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> antiKvTensorAsB16,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;
    DataCopy(v0ResGmTensor[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessNotSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t s2ProcessBaseSize = 32;
    int64_t s2ProcessSize = s2ProcessBaseSize;
    int64_t s2V0LoopTimes = (runInfo.s2RealSize + s2ProcessBaseSize - 1) / s2ProcessBaseSize;
    int64_t s2Tail = runInfo.s2RealSize - (s2V0LoopTimes - 1) * s2ProcessBaseSize;
    for (uint32_t i = 0; i < s2V0LoopTimes; i++) {
        if (i == s2V0LoopTimes - 1) {
            s2ProcessSize = s2Tail;
        }
        int64_t dealRow =
            GetSubBlockIdx() == 0 ? CeilDiv(s2ProcessSize, 2L) : s2ProcessSize - CeilDiv(s2ProcessSize, 2L);
        if (dealRow == 0) {
            continue;
        }
        int64_t s2StartIdx = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2ProcessSize, 2L);
        s2StartIdx += i * s2ProcessBaseSize;
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
        CopyInKvNotSparse(kvInUb, i, dealRow, s2StartIdx, runInfo, constInfo);
        SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);

        // 2、dequant by vf
        WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        LocalTensor<Q_T> kvDequantOutUb = stage0OutBuf[pingPongV0].Get<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, dealRow, s2ProcessBaseSize, constInfo);
        SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);

        // 3、copy kv out, ub -> l1
        SetFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        if constexpr (IS_SPLIT_G) {
            CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRow, s2StartIdx, runInfo, constInfo);
        } else {
            CopyOutKvUb2L1(outputL1, kvDequantOutUb, i, dealRow, s2StartIdx, runInfo, constInfo);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop,
                                                                     int64_t dealRow, int64_t s2StartOffset,
                                                                     const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t s2LoopCount = (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) ?
                              (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) :
                              runInfo.s2LoopCount;
    int64_t s2Idx = s2StartOffset + s2LoopCount * constInfo.s2BaseSize + runInfo.s2StartIdx;
    uint32_t combineBytes;
    uint32_t combineDim;
    uint32_t combineDimAlign;
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
        combineDim = combineBytes / sizeof(KV_T);
        combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    } else {
        combineDim = dVTemplateTypeInput / sizeof(KV_T);
        combineDimAlign = CeilAlign(dVTemplateTypeInput, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    }
    DataCopyExtParams intriParams;
    intriParams.blockCount = dealRow;
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        intriParams.blockLen = combineBytes;
        intriParams.dstStride = 0;
    } else {
        intriParams.blockLen = dCombineBytes;
        intriParams.dstStride = (dVTemplateTypeInput - dCombineBytes) / DATABLOCK_BYTES;
    }
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    if constexpr (isPa) {
        uint64_t blockTableBaseOffset = runInfo.boIdx * maxBlockNumPerBatch;
        uint64_t dstOffset = 0;
        uint32_t copyFinishElmenCnt = 0;
        uint32_t curSequence = s2Idx;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        while (copyFinishElmenCnt < dealRow) {
            uint64_t blockIdOffset = curSequence / blockSize;
            uint64_t remainElmenCnt = curSequence % blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
            uint32_t copyElmenCnt = blockSize - remainElmenCnt;
            if (copyElmenCnt + copyFinishElmenCnt > dealRow) {
                copyElmenCnt = dealRow - copyFinishElmenCnt;
            }
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
                uint64_t srcOffset = idInBlockTable * paBlockStride + remainElmenCnt * constInfo.n2Size * combineBytes +
                                     (uint64_t)(runInfo.n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
                intriParams.blockCount = copyElmenCnt;                          // base s2 size
                DataCopyPad(kvMergUb[dstOffset * combineDimAlign], keyGm[srcOffset], intriParams, padParams);
            } else {
                uint64_t keyOffset = idInBlockTable * paBlockStride +
                                     remainElmenCnt * constInfo.n2Size * dCombineBytes +
                                     (uint64_t)(runInfo.n2oIdx * dCombineBytes); // BlockNum, BlockSize, N, D
                uint64_t scaleOffset = idInBlockTable * paBlockStride + blockSize * constInfo.n2Size * dCombineBytes +
                                       remainElmenCnt * scaleBytes;
                intriParams.blockCount = copyElmenCnt; // base s2 size
                DataCopyPad(kvMergUb[dstOffset * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
                // 获取scale部分并将scale写入ub对应位置
                DataCopyExtParams scaleParams;
                DataCopyPadExtParams<KV_T> scalePadParams;
                scaleParams.blockCount = copyElmenCnt;
                scaleParams.blockLen = scaleBytes;
                scaleParams.srcStride = 0;
                scaleParams.dstStride = dCombineBytes / DATABLOCK_BYTES;
                scalePadParams.isPad = true;
                scalePadParams.leftPadding = 0;
                scalePadParams.rightPadding = DATABLOCK_BYTES - scaleBytes;
                scalePadParams.paddingValue = 0;
                DataCopyPad(kvMergUb[dstOffset * combineDimAlign + dCombineBytes], keyGm[scaleOffset], scaleParams,
                            scalePadParams);
            }
            dstOffset += copyElmenCnt;
            copyFinishElmenCnt += copyElmenCnt;
            curSequence += copyElmenCnt;
        }
    } else if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        int64_t tPrefix =
            runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx);
        DataCopyPad(kvMergUb, keyGm[(tPrefix + s2Idx) * constInfo.n2Size * combineDim + runInfo.n2oIdx * combineDim],
                    intriParams, padParams);
    } else {
        int64_t realKeyOffset;
        if (runInfo.isCmp) {
            realKeyOffset = runInfo.boIdx * constInfo.cmpS2Size * constInfo.n2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.dSizeVInput + s2Idx * constInfo.dSizeVInput;
        } else {
            realKeyOffset = runInfo.boIdx * constInfo.s2Size * constInfo.n2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.dSizeVInput + s2Idx * constInfo.dSizeVInput;
        }
        DataCopyPad(kvMergUb, keyGm[realKeyOffset], intriParams, padParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        sparseIndicesGm = runInfo.isCmp ? cmpSparseIndicesGm : oriSparseIndicesGm;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        sparseIndicesGm = oriSparseIndicesGm;
    } else {
        sparseIndicesGm = cmpSparseIndicesGm;
    }
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
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize);
        sparseS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
        sparseCalSize = sparseS2End - sparseS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    bool isCmp = runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx;
    if (isCmp) {
        keyGm = cmpKVGm;
        if constexpr (isPa) {
            blockTableGm = cmpBlockTableGm;
            blockSize = constInfo.cmpBlockSize;
            maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        }
    } else {
        keyGm = oriKVGm;
        if constexpr (isPa) {
            blockTableGm = oriBlockTableGm;
            blockSize = constInfo.oriBlockSize;
            maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
        }
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (isCmp) {
            CalSparseCalSize(runInfo, constInfo);
            ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo);
        } else {
            ProcessNotSparseKv(outputL1, v0ResGm, runInfo, constInfo);
        }
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        CalSparseCalSize(runInfo, constInfo);
        ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo);
    } else {
        ProcessNotSparseKv(outputL1, v0ResGm, runInfo, constInfo);
    }
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE3>(15);
        CrossCoreWaitFlag<0, PIPE_MTE3>(15);
    }

    if constexpr (IS_SPLIT_G) {
        v0ResGm.SetCrossCore();
    } else {
        outputL1.SetCrossCore(); // 核间同步
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (sparseCalSize == 0) {
        return;
    }
    // Left-closed, right-open interval
    // 4x = 2x + 2x
    // 4x + 1 = (2x + 2) + (2x - 1)
    // 4x + 2 = (2x + 2) + (2x)
    // 4x + 3 = (2x + 2) + (2x + 1)
    int64_t s2ProcessBaseSize = 32;
    bool meetEnd = false;
    int64_t s2Start = sparseS2Start;
    int64_t s2 = sparseS2Start;
    // 处理一个s2的base块
    while ((s2 < sparseS2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
        while (dealRow < Min(16, sparseCalSize) && s2 < sparseS2End) { // 拷贝满16行或者遇到-1
            int32_t tokenData[8] = {-1, -1, -1, -1, -1, -1, -1, -1};   // 拷贝进入的8个token的index
            GetRealCmpS2Idx(tokenData, s2, runInfo, constInfo);
            s2 += 8; // 每次搬运8行
            if (tokenData[0] == -1 && tokenData[1] == -1 && tokenData[2] == -1 && tokenData[3] == -1 &&
                tokenData[4] == -1 && tokenData[5] == -1 && tokenData[6] == -1 && tokenData[7] == -1) {
                meetEnd = true;
                break;
            }
            dealRow += CopyInKvSparse(kvInUb, dealRow, tokenData, runInfo, constInfo);
            if (tokenData[7] == -1) {
                meetEnd = true;
                break;
            }
        }
        if (dealRow == 0) {
            SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
            return;
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        // 2、dequant by vf
        WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        LocalTensor<Q_T> kvDequantOutUb = stage0OutBuf[pingPongV0].Get<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, dealRow, s2ProcessBaseSize, constInfo);
        SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        // 3、copy kv out, ub -> l1
        SetFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        if constexpr (IS_SPLIT_G) {
            CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRow, s2Start, runInfo, constInfo);
        } else {
            CopyOutKvUb2L1(outputL1, kvDequantOutUb, 0, dealRow, s2Start, runInfo, constInfo);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        s2Start += dealRow;
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    bool isFirstSoftmaxBase = runInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        isFirstSoftmaxBase = runInfo.isFirstBase;
    }
    // The first base block of every reduce block starts a new online-softmax state.
    // With sinks, use update mode after initializing max/sum from the sink state.
    if (isFirstSoftmaxBase && !isSinks) {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (isFirstSoftmaxBase && isSinks) {
            bool includeSink = (!runInfo.isCrossCoreSplit) || runInfo.isFirstS2SplitCore;
            if constexpr (IS_BATCH_CONSISTENCY) {
                includeSink = includeSink && runInfo.reduceBlockId == 0;
            }
            if (includeSink) {
                // s1切1,vec0: 0 ~ halfMRealSize - 1, vec1: gSize - halfMRealSize ~ gSize
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
            } else {
                // Later reduce blocks and non-first FD splits start without the sink state.
                Duplicate(maxUb, this->negativeFloatScalar, runInfo.halfMRealSize);
                Duplicate(sumUb, static_cast<T>(0), runInfo.halfMRealSize);
            }
        }
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    outputBuf.WaitCrossCore();
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (runInfo.mRealSize - runInfo.halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize, (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
                  (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    // ======================================================
    if (!isFirstSoftmaxBase || isSinks) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }

    if constexpr (IS_BATCH_CONSISTENCY) {
        if (runInfo.isLastBase) {
            if (runInfo.halfMRealSize > 0) {
                LocalTensor<float> finalMaxUb =
                    this->softmaxFinalMaxBuf[runInfo.taskIdMod2].template Get<float>();
                LocalTensor<float> finalSumUb =
                    this->softmaxFinalSumBuf[runInfo.taskIdMod2].template Get<float>();
                // FP32 UB-to-UB DataCopy must cover complete 32-byte blocks.
                uint64_t snapshotElems = Align8Func(runInfo.halfMRealSize);
                DataCopy(finalMaxUb, maxUb, snapshotElems);
                DataCopy(finalSumUb, sumUb, snapshotElems);
            }
            if (runInfo.isCrossCoreSplit && !runInfo.isFirstS2SplitCore) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
                    dTemplateAlign64, GetStagingSlotNum(false),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpBuf.template Get<float>();
                AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase,
                    workspaceIdx, stagingMOffset, runInfo.halfMRealSize,
                    maxUb, sumUb, tmpUb, vToMte3Id, stageMte3ToVId);
            } else if (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
                    dTemplateAlign64, GetStagingSlotNum(true), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                    AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpBuf.template Get<float>();
                AttentionCommon::StageVec1Lse(stagingLayout, intraCoreCombineBase,
                    GetIntraCoreWorkspaceIdx(runInfo, constInfo), stagingMOffset, runInfo.halfMRealSize,
                    maxUb, sumUb, tmpUb, vToMte3Id, stageMte3ToVId);
                SetFlag<HardEvent::MTE3_MTE2>(
                    intraLseMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
            } else if (runInfo.isCrossCoreSplit && runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
                    dTemplateAlign64, GetStagingSlotNum(false),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpBuf.template Get<float>();
                AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase,
                    workspaceIdx, stagingMOffset, runInfo.halfMRealSize,
                    maxUb, sumUb, tmpUb, vToMte3Id, stageMte3ToVId);
            }
        }
    } else {
        // 常规FD
        if (runInfo.isLastBase && runInfo.isCrossCoreSplit) {
            AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
                dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                AttentionCommon::FD_REDUCE_CHUNK_ROWS};
            uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
            int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
            LocalTensor<float> tmpUb = this->dequantScaleBuff.template Get<float>();
            AttentionCommon::StageVec1Lse(stagingLayout, fdStagingBase,
                workspaceIdx, stagingMOffset, runInfo.halfMRealSize,
                maxUb, sumUb, tmpUb, vToMte3Id, stageMte3ToVId);
        }
    }
    bool copyOutLse = constInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid &&
        runInfo.halfMRealSize > 0 && runInfo.s2LoopCount == runInfo.s2LoopLimit;
    if constexpr (IS_BATCH_CONSISTENCY) {
        copyOutLse = copyOutLse && !runInfo.isCrossCoreSplit && !runInfo.needReduce;
    }
    if (copyOutLse) {
        LocalTensor<float> outLse = this->outLseBuf[runInfo.multiCoreIdxMod2].template Get<float>();
        DataCopyExtParams lseParams;
        lseParams.blockCount = 1;
        lseParams.blockLen = static_cast<uint32_t>(runInfo.halfMRealSize * sizeof(float));
        lseParams.srcStride = 0;
        lseParams.dstStride = 0;
        WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        ComputeLse<float>(outLse, sumUb, maxUb, static_cast<uint32_t>(runInfo.halfMRealSize));
        SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
        WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
        DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], outLse, lseParams);
        SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    }
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
    // s2SplitIdx advances across all reduce blocks handled by this core. Move back
    // to the first slot of the current intra-core reduce group before staging it.
    uint32_t crossWorkspaceIdx = static_cast<uint32_t>(
        runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx - runInfo.reduceBlockId);
    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
    LocalTensor<float> tmpUb = this->batchReduceTmpBuf.template Get<float>();
    LocalTensor<float> blockMaxUb = tmpUb;
    LocalTensor<float> blockSumUb = tmpUb[256];
    LocalTensor<float> lseBroadcastUb = tmpUb[512];
    LocalTensor<float> sumBroadcastUb = tmpUb[640];
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    if constexpr (IS_BATCH_CONSISTENCY) {
        maxUb = this->softmaxFinalMaxBuf[runInfo.taskIdMod2].template Get<float>();
        sumUb = this->softmaxFinalSumBuf[runInfo.taskIdMod2].template Get<float>();
    }
    bool copyOutMergedLse = constInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid &&
        !runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit;

    WaitFlag<HardEvent::MTE3_MTE2>(
        intraLseMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
    WaitFlag<HardEvent::MTE3_MTE2>(
        intraAttnOutMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
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
            vToMte3Id, stageMte3ToVId);
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
                vToMte3Id, stageMte3ToVId);
        }
        startRow += intraLayout.chunkRows;
    }

    AttentionCommon::StageVec2PartialOAndWait<T>(intraLayout, intraCoreCombineGm, intraWorkspaceIdx,
        stagingMOffset, runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb,
        vToMte3Id, stageMte3ToVId);
    if (runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        AttentionCommon::StageVec2PartialOAndWait<T>(crossLayout, crossCoreCombineGm, crossWorkspaceIdx,
            stagingMOffset, runInfo.vec2MRealSize, static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb,
            vToMte3Id, stageMte3ToVId);
    }
    if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE3_MTE2>(
            intraLseMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
        SetFlag<HardEvent::MTE3_MTE2>(
            intraAttnOutMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
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
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if constexpr (IS_BATCH_CONSISTENCY) {
            if (runInfo.isLastBase) {
                LocalTensor<float> sumUb =
                    this->softmaxFinalSumBuf[runInfo.taskIdMod2].template Get<float>();
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb,
                    runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb,
                    runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            }
        } else {
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb,
                    runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb,
                    runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
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
                        static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, vToMte3Id, stageMte3ToVId);
                    SetFlag<HardEvent::MTE3_MTE2>(
                        intraAttnOutMte3ToMte2Id[runInfo.multiCoreIdxMod2]);
                }
                if (runInfo.isCrossCoreSplit &&
                    (!runInfo.isFirstS2SplitCore ||
                    (runInfo.isFirstS2SplitCore && runInfo.reduceBlockId == 0 &&
                    runInfo.s2LoopCount == runInfo.s2LoopLimit))) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
                        dTemplateAlign64, GetStagingSlotNum(false), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                    int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(stagingLayout, crossCoreCombineGm,
                        workspaceIdx, stagingMOffset, runInfo.vec2MRealSize,
                        static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, vToMte3Id, stageMte3ToVId);
                } else if (!runInfo.isCrossCoreSplit && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
                }
            }
        }
    } else {
        if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
            if (runInfo.isCrossCoreSplit) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize,
                    dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                    AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(runInfo);
                int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
                AttentionCommon::StageVec2PartialOAndWait<T>(stagingLayout, stagingOutGm,
                    workspaceIdx, stagingMOffset, runInfo.vec2MRealSize,
                    static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, vToMte3Id, stageMte3ToVId);
            } else {
                this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
            }
        }
    }
    bmm2ResBuf.SetCrossCore();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo)
{
    InitFDBuffers(fdRunInfo);
    int64_t seqOffset = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        seqOffset = this->cuSeqlensQGm.GetValue(fdRunInfo.bn2Idx);
    } else {
        seqOffset = fdRunInfo.bn2Idx * constInfo.s1Size;
    }
    int64_t attentionOutOffset = seqOffset * constInfo.n2GDv +
        fdRunInfo.mIdx * constInfo.n2GDv +
        fdRunInfo.mStartIdx * constInfo.dSizeV;
    int64_t softmaxLseOffset = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
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
            static_cast<int64_t>(fdRunInfo.mStartIdx) + startRow, dealRowCount,
            static_cast<int64_t>(constInfo.dSizeV),
            accumulatedO, lseExpUb, blockMaxUb, blockSumUb, partialOFp32,
            constInfo.isSoftmaxLseEnable, softmaxLseGm, softmaxLseOffset + startRow,
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
    SetFlag<HardEvent::V_MTE3>(vToMte3Id);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id);
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
    if (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if (LAYOUT_T == QSMLA_LAYOUT::TND) {
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
    if (constInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid) {
        uint64_t totalLseSize = 0;
        if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
            totalLseSize = constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s1Size;
        } else if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
            totalLseSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        }
        if (coreNum != 0 && totalLseSize > 0) {
            uint64_t singleCoreLse = (totalLseSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
            uint64_t tailLse = totalLseSize - constInfo.aivIdx * singleCoreLse;
            uint64_t singleInitLse = tailLse < singleCoreLse ? tailLse : singleCoreLse;
            if (constInfo.aivIdx * singleCoreLse < totalLseSize && singleInitLse > 0) {
                WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
                matmul::InitOutput<float>(this->softmaxLseGm[constInfo.aivIdx * singleCoreLse], singleInitLse, 0);
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
        if (constInfo.isSoftmaxLseEnable && softmaxLse != nullptr) {
            this->softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
            this->isSoftmaxLseGmValid = true;
        }
        if (constInfo.needInit == 1) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
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
    if (oriBlockTable != nullptr) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
    }

    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        if (cmpBlockTable != nullptr) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
        }
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)oriSparseIndices);
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
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
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    // ub buffer
    pipe->InitBuffer(dequantScaleBuff, 64 * 16 * 2 * sizeof(float)); // v0阶段每次处理16行，每行64个元素，开2 buffer

    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B
    if (constInfo.isSoftmaxLseEnable) {
        tPipe->InitBuffer(outLseBuf[0], 256);
        tPipe->InitBuffer(outLseBuf[1], 256);
    }

    tPipe->InitBuffer(stage0InBuf[0], dVTemplateTypeInput * 16 * sizeof(KV_T)); // V0阶段每次处理16个seq, 开2 buffer
    tPipe->InitBuffer(stage0InBuf[1], dVTemplateTypeInput * 16 * sizeof(KV_T));
    // kv输入D轴608, V0阶段每次处理16个seq, 开2 buffer
    tPipe->InitBuffer(stage0OutBuf[0], dVTemplateTypeInput * (16 + 1) * sizeof(Q_T));
    tPipe->InitBuffer(stage0OutBuf[1], dVTemplateTypeInput * (16 + 1) * sizeof(Q_T));

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    stageMte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    if constexpr (IS_BATCH_CONSISTENCY) {
        v0PairMte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    }
    vToMte3Id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    if constexpr (IS_BATCH_CONSISTENCY) {
        for (uint32_t eventIdx = 0; eventIdx < 2; ++eventIdx) {
            intraLseMte3ToMte2Id[eventIdx] =
                GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            intraAttnOutMte3ToMte2Id[eventIdx] =
                GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        }
        intraPartialOVToMte2Id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(intraPartialOVToMte2Id);
        reduceMaxSumVToMte2Id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        reduceMte2ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::V_MTE2>(reduceMaxSumVToMte2Id);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
    if (constInfo.isSoftmaxLseEnable) {
        mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    }
    mte2ToVV0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte2ToVV0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    vToMte2V0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    vToMte2V0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[0]);
    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[1]);
    vToMte3V0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3V0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    mte3ToVV0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVV0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[1]);

    fdVToMte2Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    fdVToMte2Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    fdMte2ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    fdMte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::V_MTE2>(fdVToMte2Id[0]);
    SetFlag<HardEvent::V_MTE2>(fdVToMte2Id[1]);
    SetFlag<HardEvent::MTE3_V>(fdMte3ToVId);

    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }
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
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetExtremeValue(T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF
class CSABlockVecDummy {
public:
    __aicore__ inline CSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo)
    {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv)
    {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedOriKv,
                                        __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo)
    {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &fdStaging) {}
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &intraCoreCombine,
        Buffer<BufferType::GM, SyncType::INNER_CORE_SYNC> &crossCoreCombine) {}
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo) {}
};
} // namespace BaseApi
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
