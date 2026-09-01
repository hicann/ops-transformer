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
 * \file quant_sparse_flash_mla_csa_block_vector.h
 * \brief
 */
#ifndef QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
#define QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H

#include "util_regbase.h"
#include "quant_sparse_flash_mla_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using AscendC::Reg::StoreDist;
#if __has_include("../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h")
#include "../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#else
#include "../common/arch35/vf/vf_flash_decode_arch35.h"
#endif

#if __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../common/buffers_policy.h"
#endif
#if __has_include("../../common/op_kernel/attn_buffer_manager.h")
#include "../../common/op_kernel/attn_buffer_manager.h"
#else
#include "../common/attn_buffer_manager.h"
#endif
#if __has_include("../../common/op_kernel/attn_buffer.h")
#include "../../common/op_kernel/attn_buffer.h"
#else
#include "../common/attn_buffer.h"
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

namespace BaseApi {

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
    static constexpr uint32_t dVTemplateTypeInput = 512;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint32_t vec1ScmBlockFp8 = s1BaseSize * 16;
    static constexpr float hifp8ScaleValue = 16.0f;
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
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE) {
                if (cuSeqlensCmpKv != nullptr) {
                    cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
                }
            }
            if (sequsedOriKv != nullptr) {
                actualSeqOriKvGm.SetGlobalBuffer((__gm__ int32_t *)sequsedOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE) {
                if (sequsedCmpKv != nullptr) {
                    actualSeqCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)sequsedCmpKv);
                }
                cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKv);
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *qDescale,
                                            __gm__ uint8_t *oriKvDescale, __gm__ uint8_t *cmpKvDescale,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo);
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
                                        ConstInfo &constInfo);
    __aicore__ inline void FreeEvent(ConstInfo &constInfo);

private:
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                           Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessNotSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                              Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                              const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalProcessSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase, const RunInfo &runInfo,
                                           ConstInfo &constInfo);
    __aicore__ inline void GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase, const RunInfo &runInfo,
                                         ConstInfo &constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0ProcessSize, int64_t s2StartIdx,
                                             const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t *tokenData,
                                              const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CastNd2Nz(LocalTensor<Q_T> dstTensor, LocalTensor<KV_T> srcTensor, int64_t dealRow);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                          LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutMrgeResult(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                             int64_t mte2Size, int64_t mte3Size, int64_t s2keyOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset);
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
    __aicore__ inline void CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx, int64_t s1Idx,
                                           int64_t validS2, int64_t alignNum, GlobalTensor<uint32_t> &phyAddrGm,
                                           uint32_t alignedSparseBlockCount);
    __aicore__ inline void CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                           GlobalTensor<int32_t> &blockTableGm, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx, int64_t s1Idx,
                                             int64_t validS2, GlobalTensor<int32_t> &sparseIndicesGm,
                                             uint32_t sparseBlockCount);
    __aicore__ inline void GetKVPhyAddrForKvType(
        uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
        bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> actualSeqQlenGm,
        GlobalTensor<int32_t> cuSeqlensQGm, GlobalTensor<int32_t> actualSeqKvlenGm, GlobalTensor<int32_t> cuSeqlensKvGm,
        GlobalTensor<int32_t> topkLengthGm, GlobalTensor<int32_t> cmpResidualKvGm, ConstInfo &constInfo,
        GlobalTensor<int32_t> &blockTableGm, GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm,
        uint32_t kvStride, uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount,
        uint32_t alignedSparseBlockCount, bool isOriKv);
    __aicore__ inline int32_t GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                        GlobalTensor<int32_t> &actualSeqGm, GlobalTensor<int32_t> &cuSeqlensGm,
                                        int64_t defaultSize);
    __aicore__ inline PhyAddrValidInfo CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size, int32_t actualOriS2Size,
                                                            int32_t restoredSize, ConstInfo &constInfo);
    __aicore__ inline int32_t CalcCurValidS2(uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv,
                                             GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &topkLengthGm,
                                             ConstInfo &constInfo, int32_t sparseBlockCount,
                                             const PhyAddrValidInfo &validInfo);

    TPipe *tPipe;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<float> qDescaleGm;
    GlobalTensor<float> oriKvDescaleGm;
    GlobalTensor<float> cmpKvDescaleGm;
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
    GlobalTensor<uint32_t> oriKvPhyAddrGm;
    GlobalTensor<uint32_t> cmpKvPhyAddrGm;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId;
    TEventID vToMte3Id;
    TEventID mte3ToVLseOutId;
    TEventID vToMte3LseOutId;
    TEventID mte2ToVV0Id[2];
    TEventID vToMte2V0Id[2];
    TEventID vToMte3V0Id[2];
    TEventID mte3ToVV0Id[2];
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> stage0InBuf[2];
    TBuf<> stage0OutBuf[2];
    TBuf<> outLseBuf[2];
    TBuf<> vselrIndexesBuf[2];
    uint32_t pingPongV0 = 0;

    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t processSize;
    int64_t processS2Start;
    int64_t processS2End;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase,
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
    for (uint64_t i = 0; i < 8; ++i) { // 每次处理8个数据块
        uint64_t idx = topkBS1Idx + runInfo.s2StartIdx + topkKIdx + i;
        if (likely((topkKIdx + i < sparseBlockCount) && (s2IdxInBase + i < processS2End))) {
            tokenData[i] = sparseIndicesGm.GetValue(idx);
        } else {
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase,
                                                                 const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint32_t sparseBlockCount = runInfo.isCmp ? constInfo.cmpSparseBlockCount : constInfo.oriSparseBlockCount;
    uint32_t alignedSparseBlockCount =
        runInfo.isCmp ? constInfo.alignedCmpSparseBlockCount : constInfo.alignedOriSparseBlockCount;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    GlobalTensor<int64_t> phyAddrGm64;
    if (runInfo.isCmp) {
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
        phyAddrGm64 = cmpKvPhyAddrGm.template ReinterpretCast<int64_t>();
    } else {
        phyAddrGm64 = oriKvPhyAddrGm.template ReinterpretCast<int64_t>();
    }

    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * alignedSparseBlockCount;
    } else {
        topkBS1Idx +=
            runInfo.boIdx * constInfo.s1Size * alignedSparseBlockCount + runInfo.s1oIdx * alignedSparseBlockCount;
    }
    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    for (uint64_t i = 0; i < 8; ++i) { // 每次处理8个数据块
        uint64_t idx = topkBS1Idx + runInfo.s2StartIdx + topkKIdx + i;
        if (likely((topkKIdx + i < sparseBlockCount) && (s2IdxInBase + i < processS2End))) {
            tokenData[i] = phyAddrGm64.GetValue(idx);
        } else {
            break;
        }
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
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * paBlockStride +
                        blkTableOffset * constInfo.dSizeVInput;
    } else if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        int64_t tPrefix =
            runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) : cuSeqlensOriKvGm.GetValue(runInfo.boIdx);
        realkeyOffset =
            (tPrefix + s2Idx) * constInfo.n2Size * constInfo.dSizeVInput + runInfo.n2oIdx * constInfo.dSizeVInput;
    } else {
        if (runInfo.isCmp) {
            realkeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSizeVInput +
                            runInfo.n2oIdx * constInfo.cmpS2Size * constInfo.dSizeVInput +
                            s2Idx * constInfo.dSizeVInput; // BSN(1)D
        } else {
            realkeyOffset = runInfo.boIdx * constInfo.n2S2Dv + runInfo.n2oIdx * constInfo.s2Dv +
                            s2Idx * constInfo.dSizeVInput; // BSN(1)D
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                  int64_t keyOffset)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;

    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    uint32_t combineBytes = 512;
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t CSABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                      int64_t *tokenData, const RunInfo &runInfo,
                                                                      ConstInfo &constInfo)
{
    int64_t s2IdLimit = runInfo.s2RealSize;
    s2IdLimit = (runInfo.s2RealSize - runInfo.actualS1Size + runInfo.s1oIdx + 1) / constInfo.cmpRatio;
    uint32_t dealRow = 0;
    for (uint32_t i = 0; i < 8; i += 2) {
        int64_t keyOffset0;
        int64_t keyOffset1;
        if constexpr (IS_VEC_S2PHYADDR) {
            keyOffset0 = tokenData[i];
            keyOffset1 = tokenData[i + 1];
        } else {
            keyOffset0 = GetkeyOffset(tokenData[i], runInfo, constInfo);
            keyOffset1 = GetkeyOffset(tokenData[i + 1], runInfo, constInfo);
        }
        if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
            return dealRow;
        }
        uint32_t combineBytes = constInfo.dSizeVInput;
        int64_t keySrcStride =
            (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - combineBytes;
        if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || constInfo.sparseBlockSize > 1) {
            // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
            CopyInSingleKv(kvInUb, startRow, keyOffset0);
            CopyInSingleKv(kvInUb, startRow + 1, keyOffset1);
        } else {
            DataCopyExtParams intriParams;
            intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
            intriParams.blockLen = combineBytes;
            intriParams.dstStride = 0;
            intriParams.srcStride = keySrcStride;
            DataCopyPadExtParams<KV_T> padParams;

            int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
            if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
                keyOffset = keyOffset1;
            }

            // 当前仅支持COMBINE模式
            uint32_t combineDim = combineBytes / sizeof(KV_T);
            uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
        }
        dealRow += (keyOffset0 >= 0) + (keyOffset1 >= 0);
        startRow += 2;
    }
    return dealRow;
}

__simd_vf__ void CastB8Nd2NzVFImpl(__ubuf__ int8_t *ubDstAddr, __ubuf__ int8_t *ubSrcAddr, const uint32_t dealRowCount,
                                   const uint32_t blockStride)
{
    MicroAPI::RegTensor<int8_t> kv_data_0;
    MicroAPI::RegTensor<int8_t> kv_data_1;

    MicroAPI::MaskReg b8_mask_all = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();
    const uint32_t repeat_stride = 1;
    const uint32_t combine_dim = 512; // nope(448) + rope(64)
    const uint32_t element_num_per_vl = 256;

    __ubuf__ int8_t *ub_src_addr_tmp = ubSrcAddr + element_num_per_vl;
    __ubuf__ int8_t *ub_dst_addr_tmp = ubDstAddr + element_num_per_vl * blockStride;

    // fp8 copy in 512 element (448 nope + 64 rope) in each loop
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load nope + rope
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            kv_data_0, ubSrcAddr, combine_dim);
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            kv_data_1, ub_src_addr_tmp, combine_dim);
        MicroAPI::StoreAlign<int8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddr, kv_data_0, blockStride, repeat_stride, b8_mask_all);
        MicroAPI::StoreAlign<int8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ub_dst_addr_tmp, kv_data_1, blockStride, repeat_stride, b8_mask_all);
    }
}

/**
 * @brief 将16x512的Nd排布转换为Nz排布，数据类型为8位数据类型
 */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CastNd2Nz(LocalTensor<Q_T> dstTensor, LocalTensor<KV_T> srcTensor,
                                                             int64_t dealRow)
{
    __ubuf__ int8_t *ubDstAddr = (__ubuf__ int8_t *)(dstTensor.GetPhyAddr());
    __ubuf__ int8_t *ubSrcAddr = (__ubuf__ int8_t *)(srcTensor.GetPhyAddr());
    const uint32_t blockStride = dealRow | 0x1; // odd row num to avoid bank confict
    CastB8Nd2NzVFImpl(ubDstAddr, ubSrcAddr, dealRow, blockStride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, LocalTensor<Q_T> kvOutUb, int64_t dealRow,
    int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    uint64_t blockElementNum = 32;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = (dealRow | 0x1) - dealRow;
    dataCopyParams.dstGap = ((runInfo.s2RealSize + 31) >> 5 << 5) - dealRow;
    DataCopy(dst[s2StartIdx * blockElementNum], kvOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> kvOutUb, int64_t dealRow,
    int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    uint64_t blockElementNum = 32;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = (dealRow | 0x1) - dealRow;
    dataCopyParams.dstGap = ((runInfo.s2RealSize + 31) >> 5 << 5) - dealRow;
    DataCopy(v0ResGmTensor[s2StartIdx * blockElementNum], kvOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessNotSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (processSize == 0) {
        return;
    }
    int64_t v0ProcessBase = 32;
    int64_t v0ProcessSize = v0ProcessBase;
    int64_t loopTimes = (processSize + v0ProcessBase - 1) / v0ProcessBase;
    for (uint32_t i = 0; i < loopTimes; i++) {
        int64_t s2StartIdx = processS2Start + i * v0ProcessBase;
        if (i == loopTimes - 1) {
            v0ProcessSize = processSize - i * v0ProcessBase;
        }

        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
        CopyInKvNotSparse(kvInUb, v0ProcessSize, s2StartIdx, runInfo, constInfo);
        SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);

        // 2、cast nd2nz
        WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        LocalTensor<Q_T> kvOutUb = stage0OutBuf[pingPongV0].Get<Q_T>();
        CastNd2Nz(kvOutUb, kvInUb, v0ProcessSize);
        SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);

        // 3、copy kv out, ub -> l1
        SetFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        CopyOutKvUb2Gm(v0ResGm, kvOutUb, v0ProcessSize, s2StartIdx, runInfo, constInfo);

        SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0ProcessSize,
                                                                     int64_t s2StartOffset, const RunInfo &runInfo,
                                                                     ConstInfo &constInfo)
{
    int64_t s2LoopCount = (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) ?
                              (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) :
                              runInfo.s2LoopCount;
    int64_t s2Idx = s2StartOffset + s2LoopCount * constInfo.s2BaseSize + runInfo.s2StartIdx;
    uint32_t combineBytes = constInfo.dSizeVInput;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    DataCopyExtParams intriParams;
    intriParams.blockCount = v0ProcessSize;
    intriParams.blockLen = combineBytes;
    intriParams.dstStride = 0;
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
        while (copyFinishElmenCnt < v0ProcessSize) {
            uint64_t blockIdOffset = curSequence / blockSize;
            uint64_t remainElmenCnt = curSequence % blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
            uint32_t copyElmenCnt = blockSize - remainElmenCnt;
            if (copyElmenCnt + copyFinishElmenCnt > v0ProcessSize) {
                copyElmenCnt = v0ProcessSize - copyFinishElmenCnt;
            }
            uint64_t srcOffset = idInBlockTable * paBlockStride + remainElmenCnt * constInfo.n2Size * combineBytes +
                                 (uint64_t)(runInfo.n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
            intriParams.blockCount = copyElmenCnt;                          // base s2 size
            DataCopyPad(kvMergUb[dstOffset * combineDimAlign], keyGm[srcOffset], intriParams, padParams);
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
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CalProcessSize(const RunInfo &runInfo, ConstInfo &constInfo)
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
                processSize = CeilDiv(v0S2SizeFirstCore, 2);
                processS2Start = 0;
            } else {
                processSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                processS2Start = CeilDiv(v0S2SizeFirstCore, 2);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                processSize = CeilDiv(v0S2SizeSecondCore, 2);
                processS2Start = v0S2SizeFirstCore;
            } else {
                processSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2);
                processS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2);
            }
        }
        processS2End = processS2Start + processSize;
    } else {
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
        processS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize);
        processS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
        processSize = processS2End - processS2Start;
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
    CalProcessSize(runInfo, constInfo);
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (isCmp) {
            ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo);
        } else {
            ProcessNotSparseKv(outputL1, v0ResGm, runInfo, constInfo);
        }
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo);
    } else {
        ProcessNotSparseKv(outputL1, v0ResGm, runInfo, constInfo);
    }
    v0ResGm.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (processSize == 0) {
        return;
    }

    bool meetEnd = false;
    int64_t s2Start = processS2Start;
    int64_t s2 = processS2Start;
    // 处理一个s2的base块
    while ((s2 < processS2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
        while (dealRow < Min(32, processSize) && s2 < processS2End) { // 拷贝满32行或者遇到-1
            int64_t tokenData[8] = {-1, -1, -1, -1, -1, -1, -1, -1};  // 拷贝进入的8个token的index
            if constexpr (IS_VEC_S2PHYADDR) {
                GetRealS2Addr(tokenData, s2, runInfo, constInfo);
            } else {
                GetRealCmpS2Idx(tokenData, s2, runInfo, constInfo);
            }
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
        // 2、cast nd2nz by vf
        WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        LocalTensor<Q_T> kvOutUb = stage0OutBuf[pingPongV0].Get<Q_T>();
        CastNd2Nz(kvOutUb, kvInUb, dealRow);
        SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        // 3、copy kv out, ub -> l1
        SetFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        CopyOutKvUb2Gm(v0ResGm, kvOutUb, dealRow, s2Start, runInfo, constInfo);
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
    float softmaxScale =
        runInfo.isCmp ? static_cast<T>(constInfo.softmaxScale * qDescaleGm.GetValue(0) * cmpKvDescaleGm.GetValue(0)) :
                        static_cast<T>(constInfo.softmaxScale * qDescaleGm.GetValue(0) * oriKvDescaleGm.GetValue(0));

    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (runInfo.s2LoopCount == 0 && !isSinks) {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        }
    } else {
        if (runInfo.s2LoopCount == 0 && isSinks) {
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
        }
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSize, softmaxScale, negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (runInfo.mRealSize - runInfo.halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 32, (uint16_t)runInfo.halfMRealSize, (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
                  (uint16_t)(Align32Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0 || (runInfo.s2LoopCount == 0 && isSinks)) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if (constInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid && runInfo.halfMRealSize > 0 &&
        runInfo.s2LoopCount == runInfo.s2LoopLimit) {
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
    bool lastLoopIsOri = runInfo.s2LoopCount <= runInfo.oriKvLoopEndIdx;
    float vDescale = runInfo.isCmp ? cmpKvDescaleGm.GetValue(0) : oriKvDescaleGm.GetValue(0);
    float vDescalePre = lastLoopIsOri ? oriKvDescaleGm.GetValue(0) : cmpKvDescaleGm.GetValue(0);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            if (unlikely(runInfo.s2LoopCount == 1)) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, true, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, vDescale,
                    vDescalePre);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, vDescale, 1.0);
            }
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            if (unlikely(runInfo.s2LoopCount == 1)) {
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, true, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, vDescale,
                    vDescalePre);
            } else {
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, vDescale,
                    1.0);
            }
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize,
                                                                  dTemplateAlign64, vDescale);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                   LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                   int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    constexpr float hifp8ScaleValueRec = 1 / 16.0;
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;
    // 除以缩放系数
    if constexpr (IsSameType<Q_T, hifloat8_t>::value) {
        Muls(vec2ResUb, vec2ResUb, hifp8ScaleValueRec, vec2CalcSize);
    }
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
__aicore__ inline int32_t CSABlockVec<TEMPLATE_ARGS>::GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                                                GlobalTensor<int32_t> &actualSeqGm,
                                                                GlobalTensor<int32_t> &cuSeqlensGm, int64_t defaultSize)
{
    if (hasActualSeq) {
        return actualSeqGm.GetValue(bIdx);
    } else if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(bIdx + 1) - cuSeqlensGm.GetValue(bIdx);
    } else {
        return defaultSize;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline PhyAddrValidInfo CSABlockVec<TEMPLATE_ARGS>::CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size,
                                                                                    int32_t actualOriS2Size,
                                                                                    int32_t restoredSize,
                                                                                    ConstInfo &constInfo)
{
    // per-batch执行一次,  per-s1循环内不再判断maskmode
    PhyAddrValidInfo validInfo;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        validInfo.oriS2Act = actualOriS2Size;
        if (isOriKv) {
            if (constInfo.oriMaskMode == 0U) {
                validInfo.oriTopkMode = true;
            } else if (constInfo.oriMaskMode == 3U) {
                validInfo.oriRightBias = 0;
            } else {
                validInfo.oriLeftBias =
                    (constInfo.oriWinLeft == -1) ? PhyAddrValidInfo::BIAS_UNBOUND : constInfo.oriWinLeft + 1;
                validInfo.oriRightBias =
                    (constInfo.oriWinRight == -1) ? PhyAddrValidInfo::BIAS_UNBOUND : constInfo.oriWinRight;
            }
        }
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            validInfo.cmpTopkMode = (constInfo.cmpMaskMode == 0U);
            validInfo.cmpBase = restoredSize - actualS1Size + 1;
        }
    }
    return validInfo;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t CSABlockVec<TEMPLATE_ARGS>::CalcCurValidS2(uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size,
                                                                     bool isOriKv, GlobalTensor<int32_t> &cuSeqlensQGm,
                                                                     GlobalTensor<int32_t> &topkLengthGm,
                                                                     ConstInfo &constInfo, int32_t sparseBlockCount,
                                                                     const PhyAddrValidInfo &validInfo)
{
    bool topkMode = false;
    bool hasTopk = false;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            topkMode = validInfo.oriTopkMode;
            hasTopk = constInfo.hasOriTopkLength;
        }
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            topkMode = validInfo.cmpTopkMode;
            hasTopk = constInfo.hasCmpTopkLength;
        }
    }
    if (topkMode) {
        uint64_t topkIdx =
            (LAYOUT_T == QSMLA_LAYOUT::TND) ? (cuSeqlensQGm.GetValue(bIdx) + s1Idx) : (bIdx * constInfo.s1Size + s1Idx);
        int32_t topkLen = hasTopk ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
        return Min(topkLen, sparseBlockCount);
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            int64_t thr = validInfo.oriS2Act - actualS1Size + 1 + s1Idx;
            int64_t leftBound = Max(thr - validInfo.oriLeftBias, 0);
            int64_t rightBound = Min(thr + validInfo.oriRightBias, static_cast<int64_t>(validInfo.oriS2Act));
            return Min(static_cast<int32_t>(Max(0, rightBound - leftBound)), sparseBlockCount);
        }
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t numerator = Max(validInfo.cmpBase + s1Idx, 0);
        return Min(sparseBlockCount, static_cast<int32_t>(numerator / static_cast<int32_t>(constInfo.cmpRatio)));
    }
    return 0;
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                    __ubuf__ int32_t *blkTableUb, const uint16_t s2Loop, uint32_t s2Tail,
                                    const uint32_t blockSize, const int16_t shiftRightNum,
                                    const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    static const uint16_t s2_num_per_loop = 128;
    static const uint16_t s2_num_per_reg = 64;
    static const uint16_t out_offset_per_loop = 256;
    static const uint16_t out_offset_per_reg = 128;
    static const uint32_t invalid_value = 0xFFFFFFFF;
    MicroAPI::MaskReg preg_all_b32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg add_carry_l_1;
    MicroAPI::MaskReg add_carry_h_1;
    MicroAPI::MaskReg add_carry_l_2;
    MicroAPI::MaskReg add_carry_h_2;
    MicroAPI::MaskReg preg_tail_neg_1_b32;
    MicroAPI::MaskReg preg_tail_neg_2_b32;

    MicroAPI::RegTensor<uint32_t> vreg_kv_stride;
    MicroAPI::RegTensor<uint32_t> vreg_sparse_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_sparse_idx_2;
    MicroAPI::RegTensor<uint32_t> vreg_block_size;
    MicroAPI::RegTensor<uint32_t> vreg_shift_rights_num;
    MicroAPI::RegTensor<uint32_t> vreg_pa_blk_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_blk_idx_2;
    MicroAPI::RegTensor<uint32_t> vreg_pa_tmp_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_tmp_2;
    MicroAPI::RegTensor<uint32_t> vreg_pa_offset_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_offset_2;
    MicroAPI::RegTensor<uint32_t> vreg_phy_offset_1;
    MicroAPI::RegTensor<uint32_t> vreg_phy_offset_2;
    MicroAPI::RegTensor<uint32_t> vreg_phy_blk_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_phy_blk_idx_2;

    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_h_1;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_h_1;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_l_1;
    MicroAPI::RegTensor<uint32_t> vreg_mul_overflow_l_1;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_l_1;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_h_1;

    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_h_2;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_h_2;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_l_2;
    MicroAPI::RegTensor<uint32_t> vreg_mul_overflow_l_2;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_l_2;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_h_2;

    MicroAPI::RegTensor<uint32_t> vreg_zero;
    MicroAPI::Duplicate(vreg_zero, 0);
    MicroAPI::Duplicate(vreg_kv_stride, kvStride);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<int32_t> &)vreg_sparse_idx_1, sparseIdxUb + i * s2_num_per_loop);
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<int32_t> &)vreg_sparse_idx_2, sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
            // * sparseBlockSize
            MicroAPI::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            MicroAPI::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // 计算右移位数
            // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
            MicroAPI::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_all_b32);
            MicroAPI::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_all_b32);

            MicroAPI::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_all_b32);
            MicroAPI::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_all_b32);
            // offset
            MicroAPI::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_all_b32);
            MicroAPI::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_all_b32);
            // 物理页内offset
            MicroAPI::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_all_b32);
            MicroAPI::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_all_b32);

            // int32 paBlockId -> 物理id
            DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_all_b32);
            DataCopyGather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_all_b32);

            // 分高低32位计算int64物理地址 -- 乘 stride
            // 低位乘 带进位
            MicroAPI::Mull(vreg_blk_id_mul_stride_l_1, vreg_mul_overflow_l_1, vreg_phy_blk_idx_1, vreg_kv_stride,
                           preg_all_b32);
            MicroAPI::Mull(vreg_blk_id_mul_stride_l_2, vreg_mul_overflow_l_2, vreg_phy_blk_idx_2, vreg_kv_stride,
                           preg_all_b32);

            // 分高低32位计算int64物理地址 -- 加 offset
            MicroAPI::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_blk_id_mul_stride_l_1, vreg_phy_offset_1,
                          preg_all_b32);
            MicroAPI::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_blk_id_mul_stride_l_2, vreg_phy_offset_2,
                          preg_all_b32);

            MicroAPI::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_zero, add_carry_l_1,
                           preg_all_b32);
            MicroAPI::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_zero, add_carry_l_2,
                           preg_all_b32);

            // 搬出 由于拆分为了int32类型，元素个数翻倍
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2,
                vreg_total_offset_h_2, preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        MicroAPI::MaskReg preg_tail_1_b32 = MicroAPI::UpdateMask<int32_t>(s2Tail);
        MicroAPI::MaskReg preg_tail_2_b32 = MicroAPI::UpdateMask<int32_t>(s2Tail);
        MicroAPI::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        MicroAPI::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                                    sparseIdxUb + i * s2_num_per_loop);
        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                                    sparseIdxUb + s2_num_per_reg + i * s2_num_per_loop);
        // * sparseBlockSize
        MicroAPI::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        MicroAPI::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // 计算右移位数
        // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
        MicroAPI::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_tail_1_b32);
        MicroAPI::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_tail_2_b32);

        MicroAPI::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_tail_1_b32);
        MicroAPI::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_tail_2_b32);
        // offset
        MicroAPI::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_tail_1_b32);
        MicroAPI::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_tail_2_b32);
        // 物理页内offset
        MicroAPI::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_tail_1_b32);
        MicroAPI::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_tail_2_b32);

        // int32 paBlockId -> 物理id
        DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_tail_1_b32);
        DataCopyGather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 乘 stride
        // 低位乘 带进位
        MicroAPI::Mull(vreg_blk_id_mul_stride_l_1, vreg_mul_overflow_l_1, vreg_phy_blk_idx_1, vreg_kv_stride,
                       preg_tail_1_b32);
        MicroAPI::Mull(vreg_blk_id_mul_stride_l_2, vreg_mul_overflow_l_2, vreg_phy_blk_idx_2, vreg_kv_stride,
                       preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 加 offset
        MicroAPI::Add(add_carry_l_1, vreg_total_offset_l_1, vreg_blk_id_mul_stride_l_1, vreg_phy_offset_1,
                      preg_tail_1_b32);
        MicroAPI::Add(add_carry_l_2, vreg_total_offset_l_2, vreg_blk_id_mul_stride_l_2, vreg_phy_offset_2,
                      preg_tail_2_b32);

        MicroAPI::AddC(add_carry_h_1, vreg_total_offset_h_1, vreg_mul_overflow_l_1, vreg_zero, add_carry_l_1,
                       preg_tail_1_b32);
        MicroAPI::AddC(add_carry_h_2, vreg_total_offset_h_2, vreg_mul_overflow_l_2, vreg_zero, add_carry_l_2,
                       preg_tail_2_b32);

        // 无效值填充-1(0xFFFFFFFF)
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_l_1, invalid_value,
                                                                        preg_tail_neg_1_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_h_1, invalid_value,
                                                                        preg_tail_neg_1_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_l_2, invalid_value,
                                                                        preg_tail_neg_2_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_h_2, invalid_value,
                                                                        preg_tail_neg_2_b32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * out_offset_per_loop, vreg_total_offset_l_1, vreg_total_offset_h_1, preg_all_b32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + out_offset_per_reg + i * out_offset_per_loop, vreg_total_offset_l_2, vreg_total_offset_h_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVF(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                      LocalTensor<int32_t> blkTableTensor, const uint16_t s2Loop, const uint32_t s2Tail,
                                      const uint32_t blockSize, const int16_t shiftRightNum,
                                      const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    __ubuf__ int32_t *blk_table_ub = (__ubuf__ int32_t *)(blkTableTensor.GetPhyAddr());
    GetKVPhyAddrVFImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, blk_table_ub, s2Loop, s2Tail, blockSize, shiftRightNum,
                                 sparseBlockSize, kvDim, kvStride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx,
                                                                   int64_t s1Idx, int64_t validS2, int64_t alignNum,
                                                                   GlobalTensor<uint32_t> &phyAddrGm,
                                                                   uint32_t alignedSparseBlockCount)
{
    constexpr int64_t numPerBlock = 32;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = ((validS2 + alignNum - 1) / alignNum * alignNum) * sizeof(int64_t) / numPerBlock;
    dataCopyParams.srcGap = 0U;
    dataCopyParams.dstGap = 0U;
    DataCopy(phyAddrGm[(bS1Idx + s1Idx) * alignedSparseBlockCount * 2], kvPhyAddrUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                                                   GlobalTensor<int32_t> &blockTableGm,
                                                                   uint32_t maxBlockNumPerBatch)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxBlockNumPerBatch * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(blkTableUb, blockTableGm[bIdx * maxBlockNumPerBatch], dataCopyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx,
                                                                     int64_t s1Idx, int64_t validS2,
                                                                     GlobalTensor<int32_t> &sparseIndicesGm,
                                                                     uint32_t sparseBlockCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = validS2 * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(sparseIdxUb, sparseIndicesGm[(bS1Idx + s1Idx) * sparseBlockCount], dataCopyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKVPhyAddrForKvType(
    uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
    bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> actualSeqQlenGm,
    GlobalTensor<int32_t> cuSeqlensQGm, GlobalTensor<int32_t> actualSeqKvlenGm, GlobalTensor<int32_t> cuSeqlensKvGm,
    GlobalTensor<int32_t> topkLengthGm, GlobalTensor<int32_t> cmpResidualKvGm, ConstInfo &constInfo,
    GlobalTensor<int32_t> &blockTableGm, GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm,
    uint32_t kvStride, uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount,
    uint32_t alignedSparseBlockCount, bool isOriKv)
{
    static constexpr uint16_t s2NumPerLoop = 128;
    static constexpr uint32_t vecCoreNum = IS_SPLIT_G ? 4 : 2;
    uint32_t vecCoreIdx = IS_SPLIT_G ? constInfo.aivIdx % 4 : constInfo.aivIdx % 2;

    int32_t blkSize = static_cast<int32_t>(blockSize);
    int16_t shiftRightNum = 0;
    while (blkSize > 1) {
        blkSize >>= 1;
        shiftRightNum++;
    }

    TBuf<> blkTableBuf;
    TBuf<> sparseIdxBuf;
    TBuf<> kvPhyAddrBuf;
    tPipe->InitBuffer(blkTableBuf, maxBlockNumPerBatch * sizeof(int32_t));
    tPipe->InitBuffer(sparseIdxBuf, alignedSparseBlockCount * sizeof(int32_t));
    tPipe->InitBuffer(kvPhyAddrBuf, alignedSparseBlockCount * sizeof(int64_t));
    LocalTensor<int32_t> blkTableUb = blkTableBuf.template Get<int32_t>();
    LocalTensor<int32_t> sparseIdxUb = sparseIdxBuf.template Get<int32_t>();
    LocalTensor<uint32_t> kvPhyAddrUb = kvPhyAddrBuf.template Get<uint32_t>();

    // 第一遍: 统计totalValidS1
    int64_t totalValidS1 = 0;
    uint32_t tmpGS1Start = gS1StartIdx;
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx; ++bIdx) {
        bool lastBN = (bIdx == bN2EndIdx - 1);
        int32_t actualS1Size =
            GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        int32_t s1End = actualS1Size;
        if (lastBN && nextGs1Idx != 0) {
            s1End = nextGs1Idx;
        }

        int64_t bS1IdxBase = 0;
        if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
            bS1IdxBase = hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : constInfo.s1Size * bIdx;
        } else {
            bS1IdxBase = constInfo.s1Size * bIdx;
        }

        int32_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && constInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                 cuSeqlensKvGm, constInfo.cmpS2Size);
                restoredSize = actualKvSize * static_cast<int32_t>(constInfo.cmpRatio) + cmpResidualKvGm.GetValue(bIdx);
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && constInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm, cuSeqlensKvGm,
                                            constInfo.s2Size);
            }
        }
        PhyAddrValidInfo validInfo =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, constInfo);

        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t curValidS2 = CalcCurValidS2(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                constInfo, static_cast<int32_t>(sparseBlockCount), validInfo);
            if (curValidS2 > 0) {
                totalValidS1++;
            }
        }
        tmpGS1Start = 0;
    }

    int64_t s1PerVecCore = totalValidS1 / vecCoreNum;
    int64_t s1Tail = totalValidS1 % vecCoreNum;
    int64_t curStart = s1PerVecCore * vecCoreIdx + Min((int64_t)vecCoreIdx, s1Tail);
    int64_t curCount = s1PerVecCore + (vecCoreIdx < (uint32_t)s1Tail ? 1 : 0);

    if (curCount == 0) {
        return;
    }

    // 第二遍: 实际计算
    int64_t validCounter = 0;
    int64_t processedCount = 0;
    tmpGS1Start = gS1StartIdx;
    bool done = false;

    SetFlag<AscendC::HardEvent::V_MTE2>(3);
    SetFlag<AscendC::HardEvent::V_MTE2>(4);
    SetFlag<AscendC::HardEvent::MTE3_V>(7);

    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx && !done; ++bIdx) {
        bool lastBN = (bIdx == bN2EndIdx - 1);
        int32_t actualS1Size =
            GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        int64_t bS1Idx = 0;
        if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
            bS1Idx = hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : constInfo.s1Size * bIdx;
        } else {
            bS1Idx = constInfo.s1Size * bIdx;
        }

        int32_t s1End = actualS1Size;
        if (lastBN && nextGs1Idx != 0) {
            s1End = nextGs1Idx;
        }

        int32_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && constInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                 cuSeqlensKvGm, constInfo.cmpS2Size);
                restoredSize = actualKvSize * static_cast<int32_t>(constInfo.cmpRatio) + cmpResidualKvGm.GetValue(bIdx);
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && constInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm, cuSeqlensKvGm,
                                            constInfo.s2Size);
            }
        }
        PhyAddrValidInfo validInfo =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, constInfo);

        WaitFlag<AscendC::HardEvent::V_MTE2>(3);
        CopyPaTableToUb(blkTableUb, bIdx, blockTableGm, maxBlockNumPerBatch);
        SetFlag<AscendC::HardEvent::MTE2_V>(8);
        WaitFlag<AscendC::HardEvent::MTE2_V>(8);

        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t curValidS2 = CalcCurValidS2(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                constInfo, static_cast<int32_t>(sparseBlockCount), validInfo);
            if (curValidS2 <= 0) {
                continue;
            }

            if (validCounter < curStart || validCounter >= curStart + curCount) {
                validCounter++;
                continue;
            }
            validCounter++;

            uint16_t s2Loop = (curValidS2 + s2NumPerLoop - 1) / s2NumPerLoop;
            int32_t s2Tail = curValidS2 - (s2Loop - 1) * s2NumPerLoop;
            WaitFlag<AscendC::HardEvent::V_MTE2>(4);
            CopySparseIdxToUb(sparseIdxUb, bS1Idx, s1Idx, curValidS2, sparseIndicesGm, sparseBlockCount);
            SetFlag<AscendC::HardEvent::MTE2_V>(6);

            WaitFlag<AscendC::HardEvent::MTE2_V>(6);
            WaitFlag<AscendC::HardEvent::MTE3_V>(7);
            GetKVPhyAddrVF<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, blockSize, shiftRightNum,
                                     constInfo.sparseBlockSize, dVTemplateTypeInput, kvStride);
            SetFlag<AscendC::HardEvent::V_MTE2>(4);
            SetFlag<AscendC::HardEvent::V_MTE3>(5);
            WaitFlag<AscendC::HardEvent::V_MTE3>(5);
            CopyPhyAddrToGm(kvPhyAddrUb, bS1Idx, s1Idx, curValidS2, s2NumPerLoop, phyAddrGm, alignedSparseBlockCount);
            SetFlag<AscendC::HardEvent::MTE3_V>(7);

            processedCount++;
            if (processedCount >= curCount) {
                done = true;
                break;
            }
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(3);
        tmpGS1Start = 0;
    }

    WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    WaitFlag<AscendC::HardEvent::V_MTE2>(4);
    WaitFlag<AscendC::HardEvent::MTE3_V>(7);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKVPhyAddr(
    uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx,
    bool hasActualSeqQlen, bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
    GlobalTensor<int32_t> actualSeqOriKvlenGm, GlobalTensor<int32_t> cuSeqlensOriKvGm,
    GlobalTensor<int32_t> oriTopkLengthGm, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
    GlobalTensor<int32_t> actualSeqCmpKvlenGm, GlobalTensor<int32_t> cuSeqlensCmpKvGm,
    GlobalTensor<int32_t> cmpTopkLengthGm, GlobalTensor<int32_t> cmpResidualKvGm, GlobalTensor<int32_t> actualSeqQlenGm,
    GlobalTensor<int32_t> cuSeqlensQGm, __gm__ uint8_t *workspace, ConstInfo &constInfo)
{
    if (hasLoad == 0) {
        SyncAll();
        tPipe->Reset();
        return;
    }

    // GM分配: ori在前, cmp在后
    int64_t v0TotalOffset = 0;
    uint32_t v0ResSize = constInfo.s2BaseSize * constInfo.dSize * sizeof(Q_T);
    if constexpr (IS_SPLIT_G) {
        v0TotalOffset = v0ResSize * 3 * (GetBlockNum() >> 1U);
    } else {
        v0TotalOffset = v0ResSize * 3 * GetBlockNum();
    }

    uint32_t totalBS1 = (LAYOUT_T == QSMLA_LAYOUT::TND) ? constInfo.s1Size : (constInfo.bSize * constInfo.s1Size);

    uint64_t oriPhyAddrSize = 0;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriPhyAddrSize = static_cast<uint64_t>(totalBS1) * constInfo.alignedOriSparseBlockCount * sizeof(int64_t);
        this->oriKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + v0TotalOffset));
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        uint64_t cmpPhyAddrSize =
            static_cast<uint64_t>(totalBS1) * constInfo.alignedCmpSparseBlockCount * sizeof(int64_t);
        this->cmpKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + v0TotalOffset + oriPhyAddrSize));
    }

    // ori部分 (先计算)
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        GetKVPhyAddrForKvType(bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                              hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqQlenGm, cuSeqlensQGm,
                              actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm, cmpResidualKvGm, constInfo,
                              oriBlockTableGm, oriSparseIndicesGm, oriKvPhyAddrGm, constInfo.oriKvStride,
                              constInfo.oriBlockSize, constInfo.oriMaxBlockNumPerBatch, constInfo.oriSparseBlockCount,
                              constInfo.alignedOriSparseBlockCount, true);
    }

    // cmp部分 (后计算)
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        GetKVPhyAddrForKvType(bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                              hasActualSeqCmpKvlen, hasCuSeqlensCmpKv, actualSeqQlenGm, cuSeqlensQGm,
                              actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm, cmpResidualKvGm, constInfo,
                              cmpBlockTableGm, cmpSparseIndicesGm, cmpKvPhyAddrGm, constInfo.cmpKvStride,
                              constInfo.cmpBlockSize, constInfo.cmpMaxBlockNumPerBatch, constInfo.cmpSparseBlockCount,
                              constInfo.alignedCmpSparseBlockCount, false);
    }

    SyncAll();
    tPipe->Reset();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *qDescale, __gm__ uint8_t *oriKvDescale,
    __gm__ uint8_t *cmpKvDescale, __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    qDescaleGm.SetGlobalBuffer((__gm__ float *)qDescale);
    if (oriKV != nullptr) {
        oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
        oriKvDescaleGm.SetGlobalBuffer((__gm__ float *)oriKvDescale);
    }

    if (oriBlockTable != nullptr) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
    }

    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        cmpKvDescaleGm.SetGlobalBuffer((__gm__ float *)cmpKvDescale);
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
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B
    if (constInfo.isSoftmaxLseEnable) {
        tPipe->InitBuffer(outLseBuf[0], 256);
        tPipe->InitBuffer(outLseBuf[1], 256);
    }
    tPipe->InitBuffer(stage0InBuf[0], dVTemplateTypeInput * 32 * sizeof(KV_T)); // V0阶段每次处理16个seq, 开2 buffer
    tPipe->InitBuffer(stage0InBuf[1], dVTemplateTypeInput * 32 * sizeof(KV_T));
    // kv输入D轴512, V0阶段每次处理16个seq, 开2 buffer
    tPipe->InitBuffer(stage0OutBuf[0], dVTemplateTypeInput * (32 + 1) * sizeof(Q_T));
    tPipe->InitBuffer(stage0OutBuf[1], dVTemplateTypeInput * (32 + 1) * sizeof(Q_T));

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    vToMte3Id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
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

    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }

    // vselrIndexesBuf
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)], 128);
    tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], 64);

    LocalTensor<uint8_t> vselrIndexesTensor =
        vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
    for (int i = 0; i < 128; i++) {
        vselrIndexesTensor.SetValue(i, i * 2);
    }
    vselrIndexesTensor =
        vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
    for (int i = 0; i < 64; i++) {
        vselrIndexesTensor.SetValue(i, i * 4);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::FreeEvent(ConstInfo &constInfo)
{
    if (constInfo.isSoftmaxLseEnable) {
        WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    }
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
    WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[0]);
    WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[1]);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[0]);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[1]);
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
    {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *qDescale,
                                            __gm__ uint8_t *oriKvDescale, __gm__ uint8_t *cmpKvDescale,
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
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo)
    {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo) {}
};
} // namespace BaseApi
#endif // QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
