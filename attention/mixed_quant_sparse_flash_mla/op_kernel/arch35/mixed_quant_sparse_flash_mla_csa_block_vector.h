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
#include "../../../common/op_kernel/init_output.h"
#include "util_regbase.h"
#include "mixed_quant_sparse_flash_mla_common_arch35.h"
using AscendC::Reg::StoreDist;
#include "../../../sparse_flash_mla/op_kernel/arch35/common/flash_decode.h"
#include "../../../sparse_flash_mla/op_kernel/arch35/common/get_kv_phy_addr_vf.h"
#include "../../../sparse_flash_mla/op_kernel/arch35/common/smla_vector_common_arch35.h"
#include "../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../common/op_kernel/buffers_policy.h"
#include "../../common/op_kernel/attn_buffer_manager.h"
#include "../../common/op_kernel/attn_buffer.h"
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace optiling;
using namespace optiling::detail;
using namespace regbaseutil;
using namespace matmul;
using namespace fa_base_matmul;
using AttentionCommon::FdRunInfo;

namespace BaseApi {

// 统一窗口公式
struct Mq35PhyAddrValidInfo {
    static constexpr int64_t BIAS_UNBOUND = 0x7FFFFFFF; // INT32_MAX
    int64_t oriLeftBias = BIAS_UNBOUND;
    int64_t oriRightBias = BIAS_UNBOUND;
    int32_t oriS2Act = 0;
    bool oriTopkMode = false; // oriMaskMode==0: 走topkLength语义(保持原行为)
    bool cmpTopkMode = true;  // cmpMaskMode==0: 走topkLength语义(保持原行为)
    int64_t cmpBase = 0;      // restoredSize - actualS1Size + 1
};

TEMPLATES_DEF
class MqsmlaCsaBlockVector {
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
    static constexpr uint32_t v0BufferDSize = 640;
    static constexpr uint32_t dCombineBytes = 576; // rope(64 * 2) + nope(448)
    static constexpr uint32_t scaleBytes = 8;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr uint32_t uint64Touint8 = sizeof(uint64_t) / sizeof(uint8_t);
    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
    // Sparse KV搬入/Dequant块大小：每次搬入8行，每16行做一次dequant
    static constexpr int64_t KV_COPYIN_UNIT = 8;   // 每次搬入8行
    static constexpr int64_t KV_DEQUANT_UNIT = 16; // 每次dequant 16行
    bool isSoftmaxLseGmValid = false;              // 标记 softmaxLseGm 是否已有效 SetGlobalBuffer
    // ==================== Functions ======================
    __aicore__ inline MqsmlaCsaBlockVector(){};
    __aicore__ inline void InitVecBlock(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                        __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedOriKv,
                                        __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
    {
        if ASCEND_IS_AIV {
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                          TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
                if (cuSeqlensCmpKv != nullptr) {
                    cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensCmpKv);
                }
            }
            if (sequsedOriKv != nullptr) {
                actualSeqOriKvGm.SetGlobalBuffer((__gm__ int32_t*)sequsedOriKv);
            }
            if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                          TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
                if (sequsedCmpKv != nullptr) {
                    actualSeqCmpKvGm.SetGlobalBuffer((__gm__ int32_t*)sequsedCmpKv);
                }
                if (cmpResidualKv != nullptr) {
                    cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t*)cmpResidualKv);
                }
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(ConstInfo<HIGH_PERF>& mq35VecConstInfo, uint32_t ubBaseAddr);
    __aicore__ inline void InitSinks(ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void InitFDBuffers(FdRunInfo& fdRunInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
                                       ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void FreeEvent(ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                            __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                            __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
                                            __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv,
                                            __gm__ uint8_t* cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC>& fdStaging)
    {
        fdStagingBase = fdStaging.template GetTensor<uint8_t>().GetPhyAddr(0);
        stagingOutGm = fdStaging.template GetTensor<float>();
    }
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC>& mqsmlaIntraCoreBuffer,
                                              Buffer<BufferType::GM, SyncType::NO_SYNC>& mqsmlaCrossCoreBuffer)
    {
        intraCoreCombineBase = mqsmlaIntraCoreBuffer.template GetTensor<uint8_t>().GetPhyAddr(0);
        intraCoreCombineGm = mqsmlaIntraCoreBuffer.template GetTensor<float>();
        crossCoreCombineBase = mqsmlaCrossCoreBuffer.template GetTensor<uint8_t>().GetPhyAddr(0);
        crossCoreCombineGm = mqsmlaCrossCoreBuffer.template GetTensor<float>();
        fdStagingBase = crossCoreCombineBase;
        stagingOutGm = crossCoreCombineGm;
    }
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessFlashDecode(FdRunInfo& fdRunInfo, ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm,
                                       const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                       ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void ProcessVec1(StaticBuffer<Q_T>& outputBuf, StaticBuffer<T>& bmm1ResBuf,
                                       RunInfo<HIGH_PERF>& mq35VecRunInfo, ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void ProcessVec2(StaticBuffer<T>& bmm2ResBuf, RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                       ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void GetKVPhyAddr(uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx,
                                        uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
                                        bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
                                        GlobalTensor<int32_t>& actualSeqOriKvlenGm,
                                        GlobalTensor<int32_t>& cuSeqlensOriKvGm, GlobalTensor<int32_t>& oriTopkLengthGm,
                                        bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
                                        GlobalTensor<int32_t>& actualSeqCmpKvlenGm,
                                        GlobalTensor<int32_t>& cuSeqlensCmpKvGm, GlobalTensor<int32_t>& cmpTopkLengthGm,
                                        GlobalTensor<int32_t>& cmpResidualKvGm, GlobalTensor<int32_t>& actualSeqQlenGm,
                                        GlobalTensor<int32_t>& cuSeqlensQGm, __gm__ uint8_t* workspace,
                                        ConstInfo<HIGH_PERF>& mq35VecConstInfo);

private:
    __aicore__ inline uint32_t GetMq35StagingSlotNum(bool isInner = false) const
    {
        if constexpr (!IS_BATCH_CONSISTENCY) {
            if constexpr (IS_SPLIT_G) {
                return AttentionCommon::FD_MAX_S2_SPLIT_NUM * (GetBlockNum() >> 1U);
            } else {
                return AttentionCommon::FD_MAX_S2_SPLIT_NUM * GetBlockNum();
            }
        } else {
            if constexpr (IS_SPLIT_G) {
                if (isInner) { // 核内
                    return GetBlockNum();
                }
                return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * (GetBlockNum() >> 1U); // 核间
            } else {
                if (isInner) { // 核内
                    return GetBlockNum() << 1U;
                }
                return BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * GetBlockNum(); // 核间
            }
        }
    }

    __aicore__ inline uint32_t GetCrossCoreWorkspaceIdx(const RunInfo<HIGH_PERF>& mq35VecRunInfo) const
    {
        uint32_t workspaceIdx =
            static_cast<uint32_t>(mq35VecRunInfo.firstFdDataWorkspaceIdx + mq35VecRunInfo.s2SplitIdx);
        return workspaceIdx;
    }

    __aicore__ inline uint32_t GetIntraCoreWorkspaceIdx(const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                        const ConstInfo<HIGH_PERF>& mq35VecConstInfo) const
    {
        uint32_t mqsmlaCoreIdx;
        if constexpr (IS_SPLIT_G) {
            mqsmlaCoreIdx = static_cast<uint32_t>(mq35VecConstInfo.aivIdx >> 2U);
        } else {
            mqsmlaCoreIdx = static_cast<uint32_t>(mq35VecConstInfo.aivIdx >> 1U);
        }
        return (mqsmlaCoreIdx << 1U) + mq35VecRunInfo.multiCoreIdxMod2;
    }

    __aicore__ inline int64_t GetFaStagingMOffset(const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                  const ConstInfo<HIGH_PERF>& mq35VecConstInfo) const
    {
        int64_t stagingMOffset =
            (mq35VecConstInfo.subBlockIdx == 1) ? static_cast<int64_t>(mq35VecRunInfo.firstHalfMRealSize) : 0L;
        if constexpr (IS_SPLIT_G) {
            stagingMOffset += static_cast<int64_t>(mq35VecRunInfo.goIdx);
        }
        return stagingMOffset;
    }

    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm,
                                           const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                           ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void ProcessNotSparseKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm,
                                              const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                              ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void CalProcessSize(const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                          ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void GetKeyOffset(int64_t s2Idx, int64_t& realKeyOffset, int64_t& realScaleOffset,
                                        const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                        ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void GetRealCmpS2Idx(int64_t* tokenData, int64_t s2IdxInBase,
                                           const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                           ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void GetRealS2Addr(int64_t* tokenData, int64_t s2IdxInBase,
                                         const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                         ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
                                             int64_t s2StartIdx, const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                             ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t* tokenData,
                                          const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                          ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    template <bool IS_FULL = false>
    __aicore__ inline void CopyIn8Block(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t& s2,
                                        const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                        ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void DequantAndCopyOutKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm,
                                               LocalTensor<KV_T> kvInUb, int64_t dealRow, int64_t s2StartIdx,
                                               const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                               ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void DequantKv(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                     ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm,
                                          LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                          ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                          int64_t scaleOffset);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<HIGH_PERF>& mq35VecRunInfo, ConstInfo<HIGH_PERF>& mq35VecConstInfo,
                                           LocalTensor<VEC2_RES_T>& mq35Vec2ResultUb, int64_t vec2S1Idx,
                                           int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                               ConstInfo<HIGH_PERF>& mq35VecConstInfo,
                                               LocalTensor<VEC2_RES_T>& mq35Vec2ResultUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer(uint32_t& ubAddr);
    __aicore__ inline void GetExtremeValue(T& negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline int32_t GetSeqLenForPhyAddr(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                                  GlobalTensor<int32_t>& actualSeqGm,
                                                  GlobalTensor<int32_t>& cuSeqlensGm, int64_t defaultSize);
    __aicore__ inline Mq35PhyAddrValidInfo CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size,
                                                                int32_t actualOriS2Size, int64_t restoredSize,
                                                                ConstInfo<HIGH_PERF>& mq35VecConstInfo);
    __aicore__ inline int32_t CalcCurValidS2ForPhyAddr(uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv,
                                                       GlobalTensor<int32_t>& cuSeqlensQGm,
                                                       GlobalTensor<int32_t>& topkLengthGm,
                                                       ConstInfo<HIGH_PERF>& mq35VecConstInfo, int32_t sparseBlockCount,
                                                       const Mq35PhyAddrValidInfo& mq35ValidWindow);
    __aicore__ inline void GetKVPhyAddrForKvType(
        uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
        bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t>& actualSeqQlenGm,
        GlobalTensor<int32_t>& cuSeqlensQGm, GlobalTensor<int32_t>& actualSeqKvlenGm,
        GlobalTensor<int32_t>& cuSeqlensKvGm, GlobalTensor<int32_t>& topkLengthGm,
        GlobalTensor<int32_t>& cmpResidualKvGm, ConstInfo<HIGH_PERF>& mq35VecConstInfo,
        GlobalTensor<int32_t>& mqActiveBlockTableGm, GlobalTensor<int32_t>& mqActiveSparseIndicesGm,
        GlobalTensor<uint32_t>& phyAddrGm, uint32_t kvStride, uint32_t mqActiveBlockSize,
        uint32_t mqActiveMaxBlocksPerBatch, uint32_t sparseBlockCount, uint32_t alignedSparseBlockCount, bool isOriKv);
    __aicore__ inline void ReduceIntraBlockAndStage(RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                    ConstInfo<HIGH_PERF>& mq35VecConstInfo,
                                                    LocalTensor<T>& mq35Vec2ResultUb, LocalTensor<T>& partialTmpUb);

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> mqActiveKvGm;
    GlobalTensor<int32_t> mqsmlaCmpSparseIndicesGm;
    GlobalTensor<int32_t> mqsmlaOriSparseIndicesGm;
    GlobalTensor<int32_t> mqActiveSparseIndicesGm;
    GlobalTensor<int32_t> mqsmlaOriBlockTableGm;
    GlobalTensor<int32_t> mqsmlaCmpBlockTableGm;
    GlobalTensor<int32_t> mqActiveBlockTableGm;
    GlobalTensor<T> mqsmlaSinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqOriKvGm;
    GlobalTensor<int32_t> actualSeqCmpKvGm;
    GlobalTensor<int32_t> cmpResidualKvGm;
    GlobalTensor<uint32_t> oriKvPhyAddrGm;
    GlobalTensor<uint32_t> cmpKvPhyAddrGm;

    StaticBuffer<T> commonUb;
    StaticBuffer<T> sinksUb;
    StaticBuffer<Q_T> stage1OutBufs[2];
    StaticBuffer<T> stage2OutBufs;
    StaticBuffer<float> softmaxMaxBufs[2];
    StaticBuffer<float> softmaxSumBufs[2];
    StaticBuffer<float> softmaxFinalMaxBufs[2];
    StaticBuffer<float> softmaxFinalSumBufs[2];
    StaticBuffer<T> softmaxExpBufs[2];
    StaticBuffer<float> batchReduceTmpUb;
    StaticBuffer<float> dequantScaleUb;
    StaticBuffer<float> outLseUbs[2];
    StaticBuffer<KV_T> stage0InBufs[2];
    StaticBuffer<Q_T> stage0OutBufs[2];
    TBuf<> vselrIndexesBuf[2];
    AttentionCommon::FdBuffers<StaticBuffer<uint8_t>> fdBuffers;
    uint32_t pingPongV0 = 0;

    T negativeFloatScalar;
    bool mqsmlaHasSinks = false;
    uint32_t mqActiveMaxBlocksPerBatch;
    uint32_t mqActiveBlockSize;
    int64_t processSize;
    int64_t processS2Start;
    int64_t processS2End;
    __gm__ uint8_t* fdStagingBase = nullptr;
    GlobalTensor<float> stagingOutGm;
    __gm__ uint8_t* intraCoreCombineBase = nullptr;
    GlobalTensor<float> intraCoreCombineGm;
    __gm__ uint8_t* crossCoreCombineBase = nullptr;
    GlobalTensor<float> crossCoreCombineGm;
};

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t* tokenData, int64_t s2IdxInBase,
                                                                            const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                            ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    uint32_t curProcessS2End = this->processS2End;
    int64_t sparseBlockCount = 0;
    int64_t curS2LoopCnt = mq35VecRunInfo.s2LoopCount;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        sparseBlockCount = mq35VecConstInfo.cmpSparseBlockCount;
        curS2LoopCnt -= mq35VecRunInfo.oriKvLoopEndIdx;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        sparseBlockCount = mq35VecConstInfo.oriSparseBlockCount;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (mq35VecRunInfo.isCmp) {
            sparseBlockCount = mq35VecConstInfo.cmpSparseBlockCount;
            curS2LoopCnt -= mq35VecRunInfo.oriKvLoopEndIdx;
        } else {
            sparseBlockCount = mq35VecConstInfo.oriSparseBlockCount;
        }
    }
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(mq35VecRunInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + mq35VecRunInfo.s1oIdx) * sparseBlockCount;
    } else {
        topkBS1Idx += mq35VecRunInfo.boIdx * mq35VecConstInfo.s1Size * sparseBlockCount +
                      mq35VecRunInfo.s1oIdx * sparseBlockCount;
    }

    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * mq35VecConstInfo.s2BaseSize;
    uint64_t idxBase = topkBS1Idx + mq35VecRunInfo.s2StartIdx + topkKIdx;
    for (uint64_t i = 0; i < KV_COPYIN_UNIT; ++i) {
        uint64_t mqsmlaIdx = idxBase + i;
        if constexpr (!IS_FULL) {
            // 尾块：保留边界判断，防止越界读取
            if (likely(s2IdxInBase + i < curProcessS2End)) {
                tokenData[i] = mqActiveSparseIndicesGm.GetValue(mqsmlaIdx);
            } else {
                break;
            }
        } else {
            // 非尾块：8行均有效，直接读取
            tokenData[i] = mqActiveSparseIndicesGm.GetValue(mqsmlaIdx);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetRealS2Addr(int64_t* tokenData, int64_t s2IdxInBase,
                                                                          const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                          ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    uint32_t curProcessS2End = this->processS2End;
    uint32_t alignedSparseBlockCount = 0;
    int64_t curS2LoopCnt = mq35VecRunInfo.s2LoopCount;
    GlobalTensor<int64_t> phyAddrGm;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        alignedSparseBlockCount = mq35VecConstInfo.alignedCmpSparseBlockCount;
        curS2LoopCnt -= mq35VecRunInfo.oriKvLoopEndIdx;
        phyAddrGm = cmpKvPhyAddrGm.template ReinterpretCast<int64_t>();
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        alignedSparseBlockCount = mq35VecConstInfo.alignedOriSparseBlockCount;
        phyAddrGm = oriKvPhyAddrGm.template ReinterpretCast<int64_t>();
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (mq35VecRunInfo.isCmp) {
            alignedSparseBlockCount = mq35VecConstInfo.alignedCmpSparseBlockCount;
            curS2LoopCnt -= mq35VecRunInfo.oriKvLoopEndIdx;
            phyAddrGm = cmpKvPhyAddrGm.template ReinterpretCast<int64_t>();
        } else {
            alignedSparseBlockCount = mq35VecConstInfo.alignedOriSparseBlockCount;
            phyAddrGm = oriKvPhyAddrGm.template ReinterpretCast<int64_t>();
        }
    }
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        topkBS1Idx = (cuSeqlensQGm.GetValue(mq35VecRunInfo.boIdx) + mq35VecRunInfo.s1oIdx) * alignedSparseBlockCount;
    } else {
        topkBS1Idx = (mq35VecRunInfo.boIdx * mq35VecConstInfo.s1Size + mq35VecRunInfo.s1oIdx) * alignedSparseBlockCount;
    }
    uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * mq35VecConstInfo.s2BaseSize;
    uint64_t idxBase = topkBS1Idx + mq35VecRunInfo.s2StartIdx + topkKIdx;
    for (uint64_t i = 0; i < KV_COPYIN_UNIT; ++i) {
        uint64_t idx = idxBase + i;
        if constexpr (!IS_FULL) {
            // 尾块：保留边界判断，防止越界读取
            if (likely(s2IdxInBase + i < curProcessS2End)) {
                tokenData[i] = phyAddrGm.GetValue(idx);
            } else {
                break;
            }
        } else {
            // 非尾块：8行均有效，直接读取
            tokenData[i] = phyAddrGm.GetValue(idx);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetKeyOffset(int64_t s2Idx, int64_t& realKeyOffset,
                                                                         int64_t& realScaleOffset,
                                                                         const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                         ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    if (s2Idx < 0) {
        return;
    }
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / mqActiveBlockSize;
        int64_t blkTableOffset = s2Idx % mqActiveBlockSize;
        int64_t paBlockStride = mq35VecRunInfo.isCmp ? mq35VecConstInfo.cmpKvStride : mq35VecConstInfo.oriKvStride;
        int32_t physBlockId = blkTableIdx;
        if constexpr (TOPK_VALUE_MODE == TopkValueMode::TOPK_INDEX_MODE) {
            physBlockId = mqActiveBlockTableGm.GetValue(mq35VecRunInfo.boIdx * mqActiveMaxBlocksPerBatch + blkTableIdx);
        }
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
            realKeyOffset = physBlockId * paBlockStride + blkTableOffset * mq35VecConstInfo.dSizeVInput;
        } else {
            realKeyOffset =
                physBlockId * paBlockStride + blkTableOffset * mq35VecConstInfo.n2Size * dCombineBytes +
                (uint64_t)(mq35VecRunInfo.n2oIdx * dCombineBytes); // BlockNum, BlockSize, N(1), D(ROPE+NOPE = 576)
            realScaleOffset = physBlockId * paBlockStride +
                              static_cast<int64_t>(mqActiveBlockSize) * mq35VecConstInfo.n2Size * dCombineBytes +
                              blkTableOffset * scaleBytes;
        }
    } else if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        int64_t tPrefix = mq35VecRunInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(mq35VecRunInfo.boIdx) :
                                                 cuSeqlensOriKvGm.GetValue(mq35VecRunInfo.boIdx);
        realKeyOffset = (tPrefix + s2Idx) * mq35VecConstInfo.n2Size * mq35VecConstInfo.dSizeVInput +
                        mq35VecRunInfo.n2oIdx * mq35VecConstInfo.dSizeVInput;
    } else {
        if (mq35VecRunInfo.isCmp) {
            realKeyOffset = mq35VecRunInfo.boIdx * mq35VecConstInfo.n2Size * mq35VecConstInfo.cmpS2Size *
                                mq35VecConstInfo.dSizeVInput +
                            mq35VecRunInfo.n2oIdx * mq35VecConstInfo.cmpS2Size * mq35VecConstInfo.dSizeVInput +
                            s2Idx * mq35VecConstInfo.dSizeVInput; // BSN(1)D
        } else {
            realKeyOffset = mq35VecRunInfo.boIdx * mq35VecConstInfo.n2Size * mq35VecConstInfo.s2Size *
                                mq35VecConstInfo.dSizeVInput +
                            mq35VecRunInfo.n2oIdx * mq35VecConstInfo.s2Size * mq35VecConstInfo.dSizeVInput +
                            s2Idx * mq35VecConstInfo.dSizeVInput; // BSN(1)D
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
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
        DataCopyPad(kvInUb[startRow * combineDimAlign], mqActiveKvGm[keyOffset], intriParams, padParams);
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
        DataCopyPad(kvInUb[startRow * combineDimAlign], mqActiveKvGm[keyOffset], intriParams, padParams);

        DataCopyExtParams mqsmlaScaleCopyParams;
        DataCopyPadExtParams<KV_T> mqsmlaScalePadParams;
        mqsmlaScaleCopyParams.blockCount = 1;
        mqsmlaScaleCopyParams.blockLen = scaleBytes;
        mqsmlaScaleCopyParams.srcStride = 0;
        mqsmlaScaleCopyParams.dstStride = 0;
        mqsmlaScalePadParams.isPad = false;
        mqsmlaScalePadParams.leftPadding = 0;
        mqsmlaScalePadParams.rightPadding = 0;
        mqsmlaScalePadParams.paddingValue = 0;
        if (scaleOffset >= 0) {
            DataCopyPad(kvInUb[startRow * combineDimAlign + dCombineBytes], mqActiveKvGm[scaleOffset],
                        mqsmlaScaleCopyParams, mqsmlaScalePadParams);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                           int64_t* tokenData,
                                                                           const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                           ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    int64_t s2IdLimit = mq35VecRunInfo.s2RealSize;
    s2IdLimit = (mq35VecRunInfo.s2RealSize - mq35VecRunInfo.actualS1Size + mq35VecRunInfo.s1oIdx + 1) /
                mq35VecConstInfo.cmpRatio;
    for (uint32_t i = 0; i < 8; i += 2) { // 遍历8个元素的数组/缓冲区，每次处理2个元素
        int64_t keyOffset0 = -1;
        int64_t keyOffset1 = -1;
        int64_t scaleOffset0 = -1;
        int64_t scaleOffset1 = -1;
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS && IS_VEC_S2PHYADDR) {
            keyOffset0 = tokenData[i];
            keyOffset1 = tokenData[i + 1];
        } else {
            GetKeyOffset(tokenData[i], keyOffset0, scaleOffset0, mq35VecRunInfo, mq35VecConstInfo);
            GetKeyOffset(tokenData[i + 1], keyOffset1, scaleOffset1, mq35VecRunInfo, mq35VecConstInfo);
        }
        if constexpr (!IS_FULL) {
            // 尾块：提前返回判断
            if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
                return;
            }
        }
        uint32_t combineBytes;
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
            combineBytes = mq35VecConstInfo.dSizeVInput * sizeof(KV_T);
        } else {
            combineBytes = dCombineBytes;
        }
        int64_t keySrcStride;
        if constexpr (IS_BATCH_CONSISTENCY) {
            // batch一致性场景，token读取顺序只与逻辑顺序有关，为保证确定性不可交换读取顺序
            keySrcStride = (keyOffset1 - keyOffset0) * sizeof(KV_T) - combineBytes;
        } else {
            keySrcStride =
                (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) * sizeof(KV_T) -
                combineBytes;
        }
        if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || mq35VecConstInfo.sparseBlockSize > 1) {
            // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
            CopyInSingleKv(kvInUb, startRow, keyOffset0, scaleOffset0);
            CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, scaleOffset1);
        } else {
            DataCopyExtParams intriParams;
            if constexpr (!IS_FULL) {
                // 尾块：根据实际有效条目数设置blockCount,且此处仅有可能存在keyOffset1为-1的情况
                intriParams.blockCount = 1 + (keyOffset1 >= 0);
            } else {
                // 非尾块：两条均有效，blockCount恒为2
                intriParams.blockCount = 2;
            }
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
                intriParams.blockLen = combineBytes;
                intriParams.dstStride = 0;
            } else {
                intriParams.blockLen = dCombineBytes;
                intriParams.dstStride = (dVTemplateTypeInput - dCombineBytes) / DATABLOCK_BYTES;
            }
            intriParams.srcStride = keySrcStride;
            DataCopyPadExtParams<KV_T> padParams;

            int64_t keyOffset;
            if constexpr (!IS_FULL) {
                keyOffset = (keyOffset1 > -1 && keyOffset1 < keyOffset0) ? keyOffset1 : keyOffset0;
            } else {
                // 非尾块：两条均有效，取较小地址作为起始
                keyOffset = keyOffset0 < keyOffset1 ? keyOffset0 : keyOffset1;
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
            DataCopyPad(kvInUb[startRow * combineDimAlign], mqActiveKvGm[keyOffset], intriParams, padParams);
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::NONCONTIGUOUS) {
                DataCopyExtParams mqsmlaScaleCopyParams;
                DataCopyPadExtParams<KV_T> mqsmlaScalePadParams;
                mqsmlaScaleCopyParams.blockCount = 1;
                mqsmlaScaleCopyParams.blockLen = scaleBytes;
                mqsmlaScaleCopyParams.srcStride = 0;
                mqsmlaScaleCopyParams.dstStride = 0;
                mqsmlaScalePadParams.isPad = false;
                mqsmlaScalePadParams.leftPadding = 0;
                mqsmlaScalePadParams.rightPadding = 0;
                mqsmlaScalePadParams.paddingValue = 0;

                // batch一致性场景下，feature和scale均保持token逻辑顺序
                // 非batch一致性场景下，feature从较小地址开始搬运，scale需同步交换
                // 尾块中keyOffset1可能为-1（仅1条有效），此时不发生交换
                bool swapOrder = false;
                if constexpr (!IS_BATCH_CONSISTENCY) {
                    swapOrder = (keyOffset0 > -1 && keyOffset1 > -1 && keyOffset0 > keyOffset1);
                }
                if (scaleOffset0 >= 0) {
                    uint32_t dstRow = swapOrder ? (startRow + 1) : startRow;
                    DataCopyPad(kvInUb[dstRow * combineDimAlign + dCombineBytes], mqActiveKvGm[scaleOffset0],
                                mqsmlaScaleCopyParams, mqsmlaScalePadParams);
                }
                if (scaleOffset1 >= 0) {
                    uint32_t dstRow = swapOrder ? startRow : (startRow + 1);
                    DataCopyPad(kvInUb[dstRow * combineDimAlign + dCombineBytes], mqActiveKvGm[scaleOffset1],
                                mqsmlaScaleCopyParams, mqsmlaScalePadParams);
                }
            }
        }
        startRow += 2U;
    }
}

// fp8->fp32
// fp8->fp32
// fp32->fp16
// fp32->fp16
static constexpr Reg::CastTrait castTraitFp8EvenToFp32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait castTraitFp8OddToFp32 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait castTraitFp32ToFp16Even = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTraitFp32ToFp16Odd = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float* ubDstAddr, __ubuf__ int8_t* ubSrcAddr, uint32_t mq35DealRows)
{
    Reg::RegTensor<fp8_e8m0_t> vScale0;
    Reg::RegTensor<bfloat16_t> vScalebf16Res0;
    Reg::RegTensor<float> vScalefp32Res0;
    __ubuf__ int8_t* ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float* ubDstAddrTmp = ubDstAddr;
    Reg::MaskReg bf16TypeMaskAll = Reg::CreateMask<bfloat16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg fp32MaskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(mq35DealRows); i++) {
        // load scale
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<int8_t>&)vScale0, ubScaleSrcAddrTemp, 608);

        Reg::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8EvenToFp32>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        Reg::Cast<float, bfloat16_t, castTraitFp8EvenToFp32>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, vScalefp32Res0, 64, bf16TypeMaskAll);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void CastScale(LocalTensor<float>& outputUb, LocalTensor<KV_T>& inputUb, uint32_t mq35DealRows)
{
    __ubuf__ float* ubDstAddr = (__ubuf__ float*)(outputUb.GetPhyAddr());
    __ubuf__ int8_t* ubScaleAddr = (__ubuf__ int8_t*)(inputUb[448 + 64 * 2].GetPhyAddr());
    CastScaleImpl<Q_T, KV_T>(ubDstAddr, ubScaleAddr, mq35DealRows);
}

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ Q_T* ubKRopeNzAddr, __ubuf__ int8_t* ubSrcAddr,
                                        __ubuf__ Q_T* ubDstAddr, __ubuf__ Q_T* ubScaleSrcAddr,
                                        __ubuf__ int8_t* ubKRopeAddr, uint32_t mq35DealRows)
{
    uint32_t combineDim = 608; // Dsize，32对齐
    Reg::RegTensor<KV_T> vKvData0;
    Reg::RegTensor<KV_T> vKvData1;
    Reg::RegTensor<Q_T> vScale0Bf16;
    Reg::RegTensor<Q_T> vScale1Bf16;
    Reg::RegTensor<half> vKvDataHalf0;
    Reg::RegTensor<half> vKvDataHalf1;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<Q_T> vMulRes0;
    Reg::RegTensor<Q_T> vMulRes1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vCastResPack0;
    Reg::RegTensor<Q_T> vCastResPack1;
    Reg::RegTensor<int8_t> vKvRope;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg kvRopeTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg kvRopeTypeMaskHalf = Reg::CreateMask<Q_T, Reg::MaskPattern::H>();
    Reg::MaskReg fp32MaskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg fp16MaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();
    uint32_t blockStride = 17; // +1 to solve bank confict
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;

    // tilesize is 64, deal 128 b8 kv, deal 2 fp32 scale
    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ Q_T* ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(mq35DealRows); i++) {
            // load scale
            Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<int8_t>&)vKvData0, ubSrcTemp, tileSize);
            Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<int8_t>&)vKvData1, ubSrcTemp, combineDim - tileSize);

            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>((Reg::RegTensor<bfloat16_t>&)vScale0Bf16,
                                                                    ubScaleSrcAddrTemp + i * (combineDim / 2));
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>((Reg::RegTensor<bfloat16_t>&)vScale1Bf16,
                                                                    ubScaleSrcAddrTemp + 1 + i * (combineDim / 2));

            Reg::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res0, vKvData0, fp32MaskAll);
            Reg::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res1, vKvData1, fp32MaskAll);

            Reg::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes0, vCastFp32Res0, fp16MaskAll);
            Reg::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes1, vCastFp32Res1, fp16MaskAll);

            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vMulRes0, vCastRes0, vScale0Bf16, fp16MaskAll);
            Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vMulRes1, vCastRes1, vScale1Bf16, fp16MaskAll);

            Reg::DeInterleave(vCastResPack0, vCastResPack1, vMulRes0, vMulRes1);

            Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }

    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop; // 偏移已经处理的循环次数
    __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + lastLoopOffset * kvNumPerLoop;
    __ubuf__ Q_T* ubScaleSrcAddrTemp = ubScaleSrcAddr + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + lastLoopOffset * kvNumPerLoop * blockStride;
    Reg::Duplicate(vCastRes1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(mq35DealRows); i++) {
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
            (Reg::RegTensor<int8_t>&)vKvRope, ubKRopeAddr, combineDim);
        // load scale
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<int8_t>&)vKvData0, ubSrcTemp, combineDim);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>((Reg::RegTensor<bfloat16_t>&)vScale0Bf16,
                                                                ubScaleSrcAddrTemp + i * (combineDim / 2));
        Reg::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res0, vKvData0, fp32MaskAll);
        Reg::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes0, vCastFp32Res0, fp16MaskAll);
        Reg::Mul<Q_T, Reg::MaskMergeMode::ZEROING>(vMulRes0, vCastRes0, vScale0Bf16, fp16MaskAll);
        Reg::DeInterleave(vCastResPack0, vCastResPack1, vMulRes0, vCastRes1);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubKRopeNzAddr, (Reg::RegTensor<Q_T>&)vKvRope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& kRopeUbNz, LocalTensor<Q_T>& outputUb,
                                          LocalTensor<KV_T>& inputUb, LocalTensor<Q_T>& scaleUb,
                                          LocalTensor<int8_t>& kRopeUb, uint32_t mq35DealRows)
{
    const uint32_t nopeDim = 448;
    const uint32_t ropeDim = 64;
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb[64 * sizeof(Q_T)].GetPhyAddr());
    // sizeof(bf16) / sizeof(fp8) = 2
    __ubuf__ Q_T* ubScaleAddr = (__ubuf__ Q_T*)(scaleUb[ropeDim + nopeDim / 2].GetPhyAddr());
    __ubuf__ int8_t* ubKRopeAddr = (__ubuf__ int8_t*)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ Q_T* ubKRopeNzAddr = (__ubuf__ Q_T*)(kRopeUbNz.GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr, mq35DealRows);
}

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448_FloatScale(__ubuf__ Q_T* ubKRopeNzAddr, __ubuf__ int8_t* ubSrcAddr,
                                                   __ubuf__ Q_T* ubDstAddr, __ubuf__ float* ubScaleSrcAddr,
                                                   __ubuf__ int8_t* ubKRopeAddr, uint32_t mq35DealRows)
{
    uint32_t combineDim = 608;
    Reg::RegTensor<KV_T> vKvData0;
    Reg::RegTensor<KV_T> vKvData1;
    Reg::RegTensor<float> vScale0;
    Reg::RegTensor<float> vScale1;
    Reg::RegTensor<float> vCastFp32Res0;
    Reg::RegTensor<float> vCastFp32Res1;
    Reg::RegTensor<float> vMulRes0;
    Reg::RegTensor<float> vMulRes1;
    Reg::RegTensor<Q_T> vCastRes0;
    Reg::RegTensor<Q_T> vCastRes1;
    Reg::RegTensor<Q_T> vCastResPack0;
    Reg::RegTensor<Q_T> vCastResPack1;
    Reg::RegTensor<int8_t> vKvRope;

    Reg::MaskReg kvTypeMaskAll = Reg::CreateMask<KV_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg kvRopeTypeMaskAll = Reg::CreateMask<Q_T, Reg::MaskPattern::ALL>();
    Reg::MaskReg kvRopeTypeMaskHalf = Reg::CreateMask<Q_T, Reg::MaskPattern::H>();
    Reg::MaskReg fp32MaskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t blockStride = 17;
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;

    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ float* ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(mq35DealRows); i++) {
            Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<int8_t>&)vKvData0, ubSrcTemp, tileSize);
            Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<int8_t>&)vKvData1, ubSrcTemp, combineDim - tileSize);

            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_BRC_B32>(
                (Reg::RegTensor<float>&)vScale0, ubScaleSrcAddrTemp, 1);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_BRC_B32>(
                (Reg::RegTensor<float>&)vScale1, ubScaleSrcAddrTemp, tileSize - 1);

            Reg::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res0, vKvData0, fp32MaskAll);
            Reg::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res1, vKvData1, fp32MaskAll);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vMulRes1, vCastFp32Res1, vScale1, fp32MaskAll);

            Reg::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes0, vMulRes0, fp32MaskAll);
            Reg::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes1, vMulRes1, fp32MaskAll);

            Reg::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

            Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }

    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop;
    __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + lastLoopOffset * kvNumPerLoop;
    __ubuf__ float* ubScaleSrcAddrTemp = ubScaleSrcAddr + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + lastLoopOffset * kvNumPerLoop * blockStride;
    Reg::Duplicate(vCastRes1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(mq35DealRows); i++) {
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
            (Reg::RegTensor<int8_t>&)vKvRope, ubKRopeAddr, combineDim);
        Reg::LoadAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<int8_t>&)vKvData0, ubSrcTemp, combineDim);
        Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_BRC_B32>(
            (Reg::RegTensor<float>&)vScale0, ubScaleSrcAddrTemp, tileSize);
        Reg::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res0, vKvData0, fp32MaskAll);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
        Reg::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes0, vMulRes0, fp32MaskAll);
        Reg::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        Reg::StoreAlign<Q_T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ubKRopeNzAddr, (Reg::RegTensor<Q_T>&)vKvRope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448_FloatScale(LocalTensor<Q_T>& kRopeUbNz, LocalTensor<Q_T>& outputUb,
                                                     LocalTensor<KV_T>& inputUb, LocalTensor<float>& scaleUb,
                                                     LocalTensor<int8_t>& kRopeUb, uint32_t mq35DealRows)
{
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb.GetPhyAddr());
    __ubuf__ int8_t* ubKRopeAddr = (__ubuf__ int8_t*)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ Q_T* ubKRopeNzAddr = (__ubuf__ Q_T*)(kRopeUbNz.GetPhyAddr());
    __ubuf__ float* ubScaleAddr = (__ubuf__ float*)(scaleUb.GetPhyAddr());

    AntiquantVFImplFp8D448_FloatScale<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr,
                                                 mq35DealRows);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
                                                                      LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                                                      ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    LocalTensor<int8_t> kRopeUb;
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[mq35VecConstInfo.dSizeNope * (16 + 1)]; // V0单次处理16行数据
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        kRopeUb = srcTensor.template ReinterpretCast<int8_t>();
        LocalTensor<Q_T> scaleUb = srcTensor.template ReinterpretCast<bfloat16_t>();
        AntiquantVFFp8D448<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, scaleUb, kRopeUb, dealRow);
    } else {
        LocalTensor<float> floatScale = dequantScaleUb.tensor;
        kRopeUb = srcTensor[448].template ReinterpretCast<int8_t>(); // 448：每个分组包含448个元素
        CastScale<Q_T, KV_T>(floatScale, srcTensor, dealRow);
        AntiquantVFFp8D448_FloatScale<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, floatScale, kRopeUb, dealRow);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm, LocalTensor<Q_T> antiKvTensorAsB16,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo<HIGH_PERF>& mq35VecRunInfo,
    ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (mq35VecConstInfo.dSizeNope + mq35VecConstInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(mq35VecRunInfo.s2RealSize) - dealRow;
    DataCopy(v0ResGmTensor[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessNotSparseKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm, const RunInfo<HIGH_PERF>& mq35VecRunInfo,
    ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    if (processSize == 0) {
        return;
    }
    constexpr int64_t v0ProcessBase = 16;
    int64_t v0ProcessSize = v0ProcessBase;
    int64_t loopTimes = (processSize + v0ProcessBase - 1) / v0ProcessBase;
    for (uint32_t i = 0; i < loopTimes; i++) {
        int64_t s2StartIdx = processS2Start + i * v0ProcessBase;
        if (i == loopTimes - 1) {
            v0ProcessSize = processSize - i * v0ProcessBase;
        }
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
        LocalTensor<KV_T> kvInUb = stage0InBufs[pingPongV0].tensor;
        CopyInKvNotSparse(kvInUb, i, v0ProcessSize, s2StartIdx, mq35VecRunInfo, mq35VecConstInfo);
        SetFlag<HardEvent::MTE2_V>(INNERCORE_STAGE0_IN(pingPongV0));
        WaitFlag<HardEvent::MTE2_V>(INNERCORE_STAGE0_IN(pingPongV0));

        // 2、dequant by vf
        WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(pingPongV0));
        LocalTensor<Q_T> kvDequantOutUb = stage0OutBufs[pingPongV0].tensor;
        DequantKv(kvDequantOutUb, kvInUb, v0ProcessSize, mq35VecConstInfo);
        SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));

        // 3、copy kv out, ub -> l1
        SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE0_OUT(pingPongV0));
        WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE0_OUT(pingPongV0));
        CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, v0ProcessSize, s2StartIdx, mq35VecRunInfo, mq35VecConstInfo);
        SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(pingPongV0));
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb,
                                                                              int64_t v0Loop, int64_t dealRow,
                                                                              int64_t s2StartOffset,
                                                                              const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                              ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    int64_t s2LoopCount = (mq35VecRunInfo.s2LoopCount >= mq35VecRunInfo.oriKvLoopEndIdx) ?
                              (mq35VecRunInfo.s2LoopCount - mq35VecRunInfo.oriKvLoopEndIdx) :
                              mq35VecRunInfo.s2LoopCount;
    int64_t s2Idx = s2StartOffset + s2LoopCount * mq35VecConstInfo.s2BaseSize + mq35VecRunInfo.s2StartIdx;
    uint32_t combineBytes;
    uint32_t combineDim;
    uint32_t combineDimAlign;
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        combineBytes = mq35VecConstInfo.dSizeVInput * sizeof(KV_T);
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
        uint64_t mqsmlaBlockTableBaseOffset = mq35VecRunInfo.boIdx * mqActiveMaxBlocksPerBatch;
        uint64_t mqsmlaDstOffset = 0;
        uint32_t mqsmlaCopyFinishElmenCnt = 0;
        uint32_t mqsmlaCurSequence = s2Idx;
        int64_t mqsmlaPaBlockStride =
            mq35VecRunInfo.isCmp ? mq35VecConstInfo.cmpKvStride : mq35VecConstInfo.oriKvStride;
        while (mqsmlaCopyFinishElmenCnt < dealRow) {
            uint64_t blockIdOffset = mqsmlaCurSequence / mqActiveBlockSize;
            uint64_t remainElmenCnt = mqsmlaCurSequence % mqActiveBlockSize;
            uint64_t idInBlockTable = mqActiveBlockTableGm.GetValue(mqsmlaBlockTableBaseOffset + blockIdOffset);
            uint32_t copyElmenCnt = mqActiveBlockSize - remainElmenCnt;
            if (copyElmenCnt + mqsmlaCopyFinishElmenCnt > dealRow) {
                copyElmenCnt = dealRow - mqsmlaCopyFinishElmenCnt;
            }
            if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
                uint64_t srcOffset = idInBlockTable * mqsmlaPaBlockStride +
                                     remainElmenCnt * mq35VecConstInfo.n2Size * combineBytes +
                                     (uint64_t)(mq35VecRunInfo.n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
                intriParams.blockCount = copyElmenCnt;
                DataCopyPad(kvMergUb[mqsmlaDstOffset * combineDimAlign], mqActiveKvGm[srcOffset], intriParams,
                            padParams);
            } else {
                uint64_t keyOffset = idInBlockTable * mqsmlaPaBlockStride +
                                     remainElmenCnt * mq35VecConstInfo.n2Size * dCombineBytes +
                                     (uint64_t)(mq35VecRunInfo.n2oIdx * dCombineBytes); // BlockNum, BlockSize, N, D
                uint64_t scaleOffset = idInBlockTable * mqsmlaPaBlockStride +
                                       mqActiveBlockSize * mq35VecConstInfo.n2Size * dCombineBytes +
                                       remainElmenCnt * scaleBytes;
                intriParams.blockCount = copyElmenCnt;
                DataCopyPad(kvMergUb[mqsmlaDstOffset * combineDimAlign], mqActiveKvGm[keyOffset], intriParams,
                            padParams);
                // 获取scale部分并将scale写入ub对应位置
                DataCopyExtParams mqsmlaScaleCopyParams;
                DataCopyPadExtParams<KV_T> mqsmlaScalePadParams;
                mqsmlaScaleCopyParams.blockCount = copyElmenCnt;
                mqsmlaScaleCopyParams.blockLen = scaleBytes;
                mqsmlaScaleCopyParams.srcStride = 0;
                mqsmlaScaleCopyParams.dstStride = dCombineBytes / DATABLOCK_BYTES;
                mqsmlaScalePadParams.isPad = true;
                mqsmlaScalePadParams.leftPadding = 0;
                mqsmlaScalePadParams.rightPadding = DATABLOCK_BYTES - scaleBytes;
                mqsmlaScalePadParams.paddingValue = 0;
                DataCopyPad(kvMergUb[mqsmlaDstOffset * combineDimAlign + dCombineBytes], mqActiveKvGm[scaleOffset],
                            mqsmlaScaleCopyParams, mqsmlaScalePadParams);
            }
            mqsmlaDstOffset += copyElmenCnt;
            mqsmlaCopyFinishElmenCnt += copyElmenCnt;
            mqsmlaCurSequence += copyElmenCnt;
        }
    } else if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        int64_t tPrefix = mq35VecRunInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(mq35VecRunInfo.boIdx) :
                                                 cuSeqlensOriKvGm.GetValue(mq35VecRunInfo.boIdx);
        DataCopyPad(
            kvMergUb,
            mqActiveKvGm[(tPrefix + s2Idx) * mq35VecConstInfo.n2Size * combineDim + mq35VecRunInfo.n2oIdx * combineDim],
            intriParams, padParams);
    } else {
        int64_t realKeyOffset;
        if (mq35VecRunInfo.isCmp) {
            realKeyOffset = mq35VecRunInfo.boIdx * mq35VecConstInfo.cmpS2Size * mq35VecConstInfo.n2Size *
                                mq35VecConstInfo.dSizeVInput +
                            mq35VecRunInfo.n2oIdx * mq35VecConstInfo.dSizeVInput + s2Idx * mq35VecConstInfo.dSizeVInput;
        } else {
            realKeyOffset = mq35VecRunInfo.boIdx * mq35VecConstInfo.s2Size * mq35VecConstInfo.n2Size *
                                mq35VecConstInfo.dSizeVInput +
                            mq35VecRunInfo.n2oIdx * mq35VecConstInfo.dSizeVInput + s2Idx * mq35VecConstInfo.dSizeVInput;
        }
        DataCopyPad(kvMergUb, mqActiveKvGm[realKeyOffset], intriParams, padParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CalProcessSize(const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                           ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        mqActiveSparseIndicesGm = mq35VecRunInfo.isCmp ? mqsmlaCmpSparseIndicesGm : mqsmlaOriSparseIndicesGm;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        mqActiveSparseIndicesGm = mqsmlaOriSparseIndicesGm;
    } else {
        mqActiveSparseIndicesGm = mqsmlaCmpSparseIndicesGm;
    }
    if constexpr (IS_SPLIT_G) {
        uint32_t mqsmlaAicIdx = mq35VecConstInfo.aivIdx >> 1U;
        uint32_t mqsmlaV0S2SizeFirstCore = CeilDiv(mq35VecRunInfo.s2RealSize, 2);
        uint32_t mqsmlaV0S2SizeSecondCore = mq35VecRunInfo.s2RealSize - mqsmlaV0S2SizeFirstCore;
        int32_t vecCnt = (mqsmlaAicIdx % 2U == 0) ? (GetSubBlockIdx() == 0 ? 0 : 1) : (GetSubBlockIdx() == 0 ? 2 : 3);
        if (mqsmlaAicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                processSize = CeilDiv(mqsmlaV0S2SizeFirstCore, 2U);
                processS2Start = 0;
            } else {
                processSize = mqsmlaV0S2SizeFirstCore - CeilDiv(mqsmlaV0S2SizeFirstCore, 2U);
                processS2Start = CeilDiv(mqsmlaV0S2SizeFirstCore, 2U);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                processSize = CeilDiv(mqsmlaV0S2SizeSecondCore, 2U);
                processS2Start = mqsmlaV0S2SizeFirstCore;
            } else {
                processSize = mqsmlaV0S2SizeSecondCore - CeilDiv(mqsmlaV0S2SizeSecondCore, 2U);
                processS2Start = mqsmlaV0S2SizeFirstCore + CeilDiv(mqsmlaV0S2SizeSecondCore, 2U);
            }
        }
        processS2End = processS2Start + processSize;
    } else {
        int64_t mqsmlaS2PerVecLoop = 2LL;
        int64_t mqsmlaVecNum = 2LL;
        int64_t mqsmlaS2Loops = CeilDiv(CeilDiv(mq35VecRunInfo.s2RealSize, mqsmlaVecNum), mqsmlaS2PerVecLoop);
        processS2Start = GetSubBlockIdx() == 0 ? 0 : Min(mqsmlaS2Loops * mqsmlaS2PerVecLoop, mq35VecRunInfo.s2RealSize);
        processS2End = GetSubBlockIdx() == 0 ? Min(mqsmlaS2Loops * mqsmlaS2PerVecLoop, mq35VecRunInfo.s2RealSize) :
                                               mq35VecRunInfo.s2RealSize;
        processSize = processS2End - processS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm, const RunInfo<HIGH_PERF>& mq35VecRunInfo,
    ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    bool mqsmlaIsCmp = mq35VecRunInfo.s2LoopCount >= mq35VecRunInfo.oriKvLoopEndIdx;
    if (mqsmlaIsCmp) {
        mqActiveKvGm = cmpKVGm;
        if constexpr (isPa) {
            mqActiveBlockTableGm = mqsmlaCmpBlockTableGm;
            mqActiveBlockSize = mq35VecConstInfo.cmpBlockSize;
            mqActiveMaxBlocksPerBatch = mq35VecConstInfo.cmpMaxBlockNumPerBatch;
        }
    } else {
        mqActiveKvGm = oriKVGm;
        if constexpr (isPa) {
            mqActiveBlockTableGm = mqsmlaOriBlockTableGm;
            mqActiveBlockSize = mq35VecConstInfo.oriBlockSize;
            mqActiveMaxBlocksPerBatch = mq35VecConstInfo.oriMaxBlockNumPerBatch;
        }
    }

    CalProcessSize(mq35VecRunInfo, mq35VecConstInfo);
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (mqsmlaIsCmp) {
            ProcessSparseKv(v0ResGm, mq35VecRunInfo, mq35VecConstInfo);
        } else {
            ProcessNotSparseKv(v0ResGm, mq35VecRunInfo, mq35VecConstInfo);
        }
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        ProcessSparseKv(v0ResGm, mq35VecRunInfo, mq35VecConstInfo);
    } else {
        ProcessNotSparseKv(v0ResGm, mq35VecRunInfo, mq35VecConstInfo);
    }
    v0ResGm.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
template <bool IS_FULL>
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyIn8Block(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                         int64_t& s2,
                                                                         const RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                         ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    // tokenData元素为-1表示无效token，尾块场景下GetReal*仅填充有效区间，其余保持-1
    int64_t tokenData[KV_COPYIN_UNIT] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS && IS_VEC_S2PHYADDR) {
        GetRealS2Addr<IS_FULL>(tokenData, s2, mq35VecRunInfo, mq35VecConstInfo);
    } else {
        GetRealCmpS2Idx<IS_FULL>(tokenData, s2, mq35VecRunInfo, mq35VecConstInfo);
    }
    s2 += KV_COPYIN_UNIT;
    CopyInKvSparse<IS_FULL>(kvInUb, startRow, tokenData, mq35VecRunInfo, mq35VecConstInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::DequantAndCopyOutKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm, LocalTensor<KV_T> kvInUb, int64_t dealRow,
    int64_t s2StartIdx, const RunInfo<HIGH_PERF>& mq35VecRunInfo, ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    SetFlag<HardEvent::MTE2_V>(INNERCORE_STAGE0_IN(pingPongV0));
    WaitFlag<HardEvent::MTE2_V>(INNERCORE_STAGE0_IN(pingPongV0));
    // 2、dequant by vf
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(pingPongV0));
    LocalTensor<Q_T> kvDequantOutUb = stage0OutBufs[pingPongV0].tensor;
    DequantKv(kvDequantOutUb, kvInUb, dealRow, mq35VecConstInfo);
    SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
    // 3、copy kv out, ub -> l1
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE0_OUT(pingPongV0));
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE0_OUT(pingPongV0));
    CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRow, s2StartIdx, mq35VecRunInfo, mq35VecConstInfo);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(pingPongV0));
    pingPongV0 ^= 1;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>& v0ResGm, const RunInfo<HIGH_PERF>& mq35VecRunInfo,
    ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    int64_t curProcessSize = this->processSize;
    if (curProcessSize == 0) {
        return;
    }

    // 前置计算：16行dequant循环数、尾块行数、尾块内8行搬入次数及剩余行数
    int64_t dequant16LoopCnt = curProcessSize / KV_DEQUANT_UNIT;
    int64_t tail16Rows = curProcessSize % KV_DEQUANT_UNIT;
    int64_t tail8FullCnt = tail16Rows / KV_COPYIN_UNIT;
    int64_t tail8Remain = tail16Rows % KV_COPYIN_UNIT;
    int64_t s2 = processS2Start;

    // 阶段1：完整16行块（非尾块，8行均有效，无需判断）
    for (int64_t i = 0; i < dequant16LoopCnt; i++) {
        int64_t s2StartIdx = processS2Start + i * KV_DEQUANT_UNIT;
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
        LocalTensor<KV_T> kvInUb = stage0InBufs[pingPongV0].tensor;
        CopyIn8Block<true>(kvInUb, 0, s2, mq35VecRunInfo, mq35VecConstInfo);
        CopyIn8Block<true>(kvInUb, KV_COPYIN_UNIT, s2, mq35VecRunInfo, mq35VecConstInfo);
        // 2、dequant 3、copy kv out, ub -> l1
        DequantAndCopyOutKv(v0ResGm, kvInUb, KV_DEQUANT_UNIT, s2StartIdx, mq35VecRunInfo, mq35VecConstInfo);
    }

    // 阶段2：尾块（不足16行，保留判断逻辑）
    if (tail16Rows > 0) {
        int64_t s2StartIdx = processS2Start + dequant16LoopCnt * KV_DEQUANT_UNIT;
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(pingPongV0));
        LocalTensor<KV_T> kvInUb = stage0InBufs[pingPongV0].tensor;
        // 尾块内满8行搬入（8行均有效，无需判断）
        for (int64_t j = 0; j < tail8FullCnt; j++) {
            CopyIn8Block<true>(kvInUb, dealRow, s2, mq35VecRunInfo, mq35VecConstInfo);
            dealRow += KV_COPYIN_UNIT;
        }
        // 尾块内不足8行（需判断边界，与当前逻辑一致）
        if (tail8Remain > 0) {
            CopyIn8Block<false>(kvInUb, dealRow, s2, mq35VecRunInfo, mq35VecConstInfo);
            dealRow += tail8Remain;
        }
        // 2、dequant 3、copy kv out, ub -> l1
        DequantAndCopyOutKv(v0ResGm, kvInUb, tail16Rows, s2StartIdx, mq35VecRunInfo, mq35VecConstInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec1(StaticBuffer<Q_T>& outputBuf,
                                                                        StaticBuffer<T>& bmm1ResBuf,
                                                                        RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                        ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1ResBuf.idx));

    LocalTensor<float> mqsmlaSumUb = this->softmaxSumBufs[mq35VecRunInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> mqsmlaMaxUb = this->softmaxMaxBufs[mq35VecRunInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> mqsmlaExpUb = this->softmaxExpBufs[mq35VecRunInfo.taskIdMod2].tensor;
    int64_t mqsmlaStage1Offset = mq35VecRunInfo.taskIdMod2;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(mqsmlaStage1Offset));
    LocalTensor<Q_T> mqsmlaStage1CastTensor = this->stage1OutBufs[mqsmlaStage1Offset].tensor;

    LocalTensor<T> mqsmlaApiTmpBuffer = this->commonUb.tensor;
    LocalTensor<T> mqsmlaMmRes = bmm1ResBuf.tensor;

    bool isFirstSoftmaxBase = mq35VecRunInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        isFirstSoftmaxBase = mq35VecRunInfo.isFirstBase;
    }
    // The first base block of every reduce block starts a new online-softmax state.
    // With sinks, use update mode after initializing max/sum from the sink state.
    if (isFirstSoftmaxBase && !mqsmlaHasSinks) {
        if (likely(mq35VecRunInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, mq35VecRunInfo.halfMRealSize, mq35VecRunInfo.s2RealSize,
                static_cast<T>(mq35VecConstInfo.softmaxScale), negativeFloatScalar);
        } else if (mq35VecRunInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, mq35VecRunInfo.halfMRealSize, mq35VecRunInfo.s2RealSize,
                static_cast<T>(mq35VecConstInfo.softmaxScale), negativeFloatScalar);
        } else if (mq35VecRunInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, mq35VecRunInfo.halfMRealSize, mq35VecRunInfo.s2RealSize,
                static_cast<T>(mq35VecConstInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (isFirstSoftmaxBase && mqsmlaHasSinks) {
            bool includeSink = (!mq35VecRunInfo.isCrossCoreSplit) || mq35VecRunInfo.isFirstS2SplitCore;
            if constexpr (IS_BATCH_CONSISTENCY) {
                includeSink = includeSink && mq35VecRunInfo.reduceBlockId == 0;
            }
            if (includeSink) {
                // s1切1,vec0: 0 ~ halfMRealSize - 1, vec1: gSize - halfMRealSize ~ gSize
                int64_t sinksOffset = 0;
                if constexpr (!IS_SPLIT_G) {
                    sinksOffset = GetBlockIdx() % 2U == 0 ? 0 : mq35VecRunInfo.firstHalfMRealSize;
                } else {
                    sinksOffset = mq35VecRunInfo.goIdx;
                    if (mq35VecConstInfo.subBlockIdx == 1) {
                        sinksOffset += mq35VecRunInfo.firstHalfMRealSize;
                    }
                }
                LocalTensor<T> sinksUb = this->sinksUb.tensor;
                InitSoftmaxFromSinks<T>(mqsmlaSumUb, mqsmlaMaxUb, sinksUb, sinksOffset, R0,
                                        mq35VecRunInfo.halfMRealSize);
            } else {
                // Later reduce blocks and non-first FD splits start without the sink state.
                Duplicate(mqsmlaMaxUb, this->negativeFloatScalar, mq35VecRunInfo.halfMRealSize);
                Duplicate(mqsmlaSumUb, static_cast<T>(0), mq35VecRunInfo.halfMRealSize);
            }
        }
        if (likely(mq35VecRunInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, mq35VecRunInfo.halfMRealSize, mq35VecRunInfo.s2RealSize,
                static_cast<T>(mq35VecConstInfo.softmaxScale), negativeFloatScalar);
        } else if (mq35VecRunInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, mq35VecRunInfo.halfMRealSize, mq35VecRunInfo.s2RealSize,
                static_cast<T>(mq35VecConstInfo.softmaxScale), negativeFloatScalar);
        } else if (mq35VecRunInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                mqsmlaStage1CastTensor, mqsmlaMmRes, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                vselrIndexesBuf, mq35VecRunInfo.halfMRealSize, mq35VecRunInfo.s2RealSize,
                static_cast<T>(mq35VecConstInfo.softmaxScale), negativeFloatScalar);
        }
    }
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1ResBuf.idx));

    // ===================DataCopy to L1 ====================
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE1(mqsmlaStage1Offset));
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE1(mqsmlaStage1Offset));

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.tensor;
    if (likely(mq35VecRunInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[mq35VecConstInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (mq35VecRunInfo.mRealSize - mq35VecRunInfo.halfMRealSize)],
                 mqsmlaStage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)mq35VecRunInfo.halfMRealSize,
                  (uint16_t)(vec1Srcstride - mq35VecRunInfo.halfMRealSize),
                  (uint16_t)(Align16Func(mq35VecRunInfo.mRealSize) - mq35VecRunInfo.halfMRealSize)});
    }

    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE1(mqsmlaStage1Offset));

    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CROSSCORE_L1P(outputBuf.idx));
    // ======================================================
    if (!isFirstSoftmaxBase || mqsmlaHasSinks) {
        SFAUpdateExpSumAndExpMax<T>(mqsmlaSumUb, mqsmlaMaxUb, mqsmlaExpUb, mqsmlaSumUb, mqsmlaMaxUb, mqsmlaApiTmpBuffer,
                                    mq35VecRunInfo.halfMRealSize);
    }

    if constexpr (IS_BATCH_CONSISTENCY) {
        if (mq35VecRunInfo.isLastBase) {
            if (mq35VecRunInfo.halfMRealSize > 0) {
                LocalTensor<float> finalMaxUb = this->softmaxFinalMaxBufs[mq35VecRunInfo.taskIdMod2].tensor;
                LocalTensor<float> finalSumUb = this->softmaxFinalSumBufs[mq35VecRunInfo.taskIdMod2].tensor;
                // FP32 UB-to-UB DataCopy must cover complete 32-byte blocks.
                uint64_t snapshotElems = Align8Func(mq35VecRunInfo.halfMRealSize);
                DataCopy(finalMaxUb, mqsmlaMaxUb, snapshotElems);
                DataCopy(finalSumUb, mqsmlaSumUb, snapshotElems);
            }
            if (mq35VecRunInfo.isCrossCoreSplit && !mq35VecRunInfo.isFirstS2SplitCore) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                    mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(false),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(mq35VecRunInfo);
                int64_t stagingMOffset = GetFaStagingMOffset(mq35VecRunInfo, mq35VecConstInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
                AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase, workspaceIdx, stagingMOffset,
                                              mq35VecRunInfo.halfMRealSize, mqsmlaMaxUb, mqsmlaSumUb, tmpUb,
                                              INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
            } else if (mq35VecRunInfo.isFirstS2SplitCore && mq35VecRunInfo.reduceBlockId == 0 &&
                       mq35VecRunInfo.s2LoopCount < mq35VecRunInfo.s2LoopLimit) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                    mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(true),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                int64_t stagingMOffset = GetFaStagingMOffset(mq35VecRunInfo, mq35VecConstInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
                AttentionCommon::StageVec1Lse(stagingLayout, intraCoreCombineBase,
                                              GetIntraCoreWorkspaceIdx(mq35VecRunInfo, mq35VecConstInfo),
                                              stagingMOffset, mq35VecRunInfo.halfMRealSize, mqsmlaMaxUb, mqsmlaSumUb,
                                              tmpUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
                // 空行 AIV 的 Vec2 会提前返回，因此这里不能留下无人等待的同步事件。
                if (mq35VecRunInfo.halfMRealSize > 0) {
                    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(mq35VecRunInfo.multiCoreIdxMod2));
                }
            } else if (mq35VecRunInfo.isCrossCoreSplit && mq35VecRunInfo.isFirstS2SplitCore &&
                       mq35VecRunInfo.reduceBlockId == 0) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                    mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(false),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(mq35VecRunInfo);
                int64_t stagingMOffset = GetFaStagingMOffset(mq35VecRunInfo, mq35VecConstInfo);
                LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
                AttentionCommon::StageVec1Lse(stagingLayout, crossCoreCombineBase, workspaceIdx, stagingMOffset,
                                              mq35VecRunInfo.halfMRealSize, mqsmlaMaxUb, mqsmlaSumUb, tmpUb,
                                              INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
            }
        }
    } else {
        // 常规FD
        if (mq35VecRunInfo.isLastBase && mq35VecRunInfo.isCrossCoreSplit) {
            AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(),
                AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
            uint32_t workspaceIdx = GetCrossCoreWorkspaceIdx(mq35VecRunInfo);
            int64_t stagingMOffset = GetFaStagingMOffset(mq35VecRunInfo, mq35VecConstInfo);
            LocalTensor<float> tmpUb = this->dequantScaleUb.tensor;
            AttentionCommon::StageVec1Lse(stagingLayout, fdStagingBase, workspaceIdx, stagingMOffset,
                                          mq35VecRunInfo.halfMRealSize, mqsmlaMaxUb, mqsmlaSumUb, tmpUb,
                                          INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        }
    }
    if constexpr (!HIGH_PERF) {
        bool copyOutLse = mq35VecConstInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid &&
                          mq35VecRunInfo.halfMRealSize > 0 && mq35VecRunInfo.s2LoopCount == mq35VecRunInfo.s2LoopLimit;
        if constexpr (IS_BATCH_CONSISTENCY) {
            copyOutLse = copyOutLse && !mq35VecRunInfo.isCrossCoreSplit && !mq35VecRunInfo.needReduce;
        }
        if (copyOutLse) {
            LocalTensor<float> outLse = this->outLseUbs[mq35VecRunInfo.multiCoreIdxMod2].tensor;
            DataCopyExtParams lseParams;
            lseParams.blockCount = 1;
            lseParams.blockLen = static_cast<uint32_t>(mq35VecRunInfo.halfMRealSize * sizeof(float));
            lseParams.srcStride = 0;
            lseParams.dstStride = 0;
            WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
            ComputeLse<float>(outLse, mqsmlaSumUb, mqsmlaMaxUb, static_cast<uint32_t>(mq35VecRunInfo.halfMRealSize));
            SetFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
            DataCopyPad(this->softmaxLseGm[mq35VecRunInfo.softmaxLseOffset], outLse, lseParams);
            SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ReduceIntraBlockAndStage(
    RunInfo<HIGH_PERF>& mq35VecRunInfo, ConstInfo<HIGH_PERF>& mq35VecConstInfo, LocalTensor<T>& mq35Vec2ResultUb,
    LocalTensor<T>& partialTmpUb)
{
    AttentionCommon::S2SplitFdStagingLayout mqsmlaIntraLayout = {
        mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(true),
        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    AttentionCommon::S2SplitFdStagingLayout mqsmlaCrossLayout = {
        mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(false),
        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    uint32_t intraWorkspaceIdx = GetIntraCoreWorkspaceIdx(mq35VecRunInfo, mq35VecConstInfo);
    // s2SplitIdx advances across all reduce blocks handled by this core. Move back
    // to the first slot of the current intra-core reduce group before staging it.
    uint32_t crossWorkspaceIdx = static_cast<uint32_t>(mq35VecRunInfo.firstFdDataWorkspaceIdx +
                                                       mq35VecRunInfo.s2SplitIdx - mq35VecRunInfo.reduceBlockId);
    int64_t stagingMOffset = GetFaStagingMOffset(mq35VecRunInfo, mq35VecConstInfo);
    LocalTensor<float> tmpUb = this->batchReduceTmpUb.tensor;
    LocalTensor<float> blockMaxUb = tmpUb;
    LocalTensor<float> blockSumUb = tmpUb[256];
    LocalTensor<float> lseBroadcastUb = tmpUb[512];
    LocalTensor<float> sumBroadcastUb = tmpUb[640];
    LocalTensor<float> maxUb = this->softmaxMaxBufs[mq35VecRunInfo.multiCoreIdxMod2].tensor;
    LocalTensor<float> sumUb = this->softmaxSumBufs[mq35VecRunInfo.multiCoreIdxMod2].tensor;
    if constexpr (IS_BATCH_CONSISTENCY) {
        maxUb = this->softmaxFinalMaxBufs[mq35VecRunInfo.taskIdMod2].tensor;
        sumUb = this->softmaxFinalSumBufs[mq35VecRunInfo.taskIdMod2].tensor;
    }
    bool copyOutMergedLse = false;
    if constexpr (!HIGH_PERF) {
        copyOutMergedLse = mq35VecConstInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid &&
                           !mq35VecRunInfo.isCrossCoreSplit && mq35VecRunInfo.s2LoopCount == mq35VecRunInfo.s2LoopLimit;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(mq35VecRunInfo.multiCoreIdxMod2));
    WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(mq35VecRunInfo.multiCoreIdxMod2));
    LocalTensor<T> sinkUb;
    int64_t startRow = 0;
    while (startRow < mq35VecRunInfo.vec2MRealSize) {
        int64_t mq35DealRows = mqsmlaIntraLayout.chunkRows;
        if (startRow + mq35DealRows > mq35VecRunInfo.vec2MRealSize) {
            mq35DealRows = mq35VecRunInfo.vec2MRealSize - startRow;
        }
        LocalTensor<T> chunkCurrent = mq35Vec2ResultUb[startRow * dTemplateAlign64];
        LocalTensor<float> chunkMaxUb = maxUb[startRow];
        LocalTensor<float> chunkSumUb = sumUb[startRow];
        if constexpr (!HIGH_PERF) {
            if (copyOutMergedLse) {
                WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
            }
        }
        AttentionCommon::MergeStagedAndCurrentChunk<T, dTemplateAlign64>(
            mqsmlaIntraLayout, intraCoreCombineBase, intraWorkspaceIdx, stagingMOffset + startRow, mq35DealRows,
            static_cast<int64_t>(mq35VecConstInfo.dSizeV), chunkMaxUb, chunkSumUb, chunkCurrent, blockMaxUb, blockSumUb,
            partialTmpUb, lseBroadcastUb, sumBroadcastUb, sinkUb, INNERCORE_REDUCE_MAXSUM_V_MTE2,
            INNERCORE_INTRAPARTIALO_V_MTE2, INNERCORE_REDUCE_MTE2_V);

        AttentionCommon::StageBroadcastMaxSum(mqsmlaIntraLayout, intraCoreCombineBase, intraWorkspaceIdx,
                                              stagingMOffset + startRow, mq35DealRows, lseBroadcastUb, sumBroadcastUb,
                                              INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        if constexpr (!HIGH_PERF) {
            if (copyOutMergedLse) {
                DataCopyExtParams lseParams;
                lseParams.blockCount = static_cast<uint16_t>(mq35DealRows);
                lseParams.blockLen = sizeof(float);
                lseParams.srcStride = 0;
                lseParams.dstStride = 0;
                SetFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
                WaitFlag<HardEvent::V_MTE3>(INNERCORE_LSE_V_MTE3);
                DataCopyPad(this->softmaxLseGm[mq35VecRunInfo.softmaxLseOffset + startRow], lseBroadcastUb, lseParams);
                SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
            }
        }
        if (mq35VecRunInfo.isCrossCoreSplit && mq35VecRunInfo.s2LoopCount == mq35VecRunInfo.s2LoopLimit) {
            AttentionCommon::StageBroadcastMaxSum(mqsmlaCrossLayout, crossCoreCombineBase, crossWorkspaceIdx,
                                                  stagingMOffset + startRow, mq35DealRows, lseBroadcastUb,
                                                  sumBroadcastUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
        }
        startRow += mqsmlaIntraLayout.chunkRows;
    }

    AttentionCommon::StageVec2PartialOAndWait<T>(
        mqsmlaIntraLayout, intraCoreCombineGm, intraWorkspaceIdx, stagingMOffset, mq35VecRunInfo.vec2MRealSize,
        static_cast<uint32_t>(mq35VecConstInfo.dSizeV), mq35Vec2ResultUb, INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
    if (mq35VecRunInfo.isCrossCoreSplit && mq35VecRunInfo.s2LoopCount == mq35VecRunInfo.s2LoopLimit) {
        AttentionCommon::StageVec2PartialOAndWait<T>(mqsmlaCrossLayout, crossCoreCombineGm, crossWorkspaceIdx,
                                                     stagingMOffset, mq35VecRunInfo.vec2MRealSize,
                                                     static_cast<uint32_t>(mq35VecConstInfo.dSizeV), mq35Vec2ResultUb,
                                                     INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
    }
    if (mq35VecRunInfo.s2LoopCount < mq35VecRunInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRALSE_MTE3_MTE2(mq35VecRunInfo.multiCoreIdxMod2));
        SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(mq35VecRunInfo.multiCoreIdxMod2));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec2(StaticBuffer<T>& bmm2ResBuf,
                                                                        RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                        ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    if (unlikely(mq35VecRunInfo.vec2MBaseSize == 0)) {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
        return;
    }

    mq35VecRunInfo.vec2S1RealSize = mq35VecRunInfo.vec2S1BaseSize;
    mq35VecRunInfo.vec2MRealSize = mq35VecRunInfo.vec2MBaseSize;
    int64_t mqsmlaVec2CalcSize = mq35VecRunInfo.vec2MRealSize * dTemplateAlign64;
    LocalTensor<T> mqsmlaVec2ResUb = this->stage2OutBufs.tensor;
    LocalTensor<T> mqsmlaMmRes = bmm2ResBuf.tensor;
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    bool mqsmlaNeedIntraBlockReduce = false;
    if constexpr (IS_BATCH_CONSISTENCY) {
        mqsmlaNeedIntraBlockReduce =
            mq35VecRunInfo.isLastBase && mq35VecRunInfo.isFirstS2SplitCore && mq35VecRunInfo.reduceBlockId > 0;
        if (mqsmlaNeedIntraBlockReduce) {
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
        }
    }
    bool mqsmlaIsFirstVec2Base = mq35VecRunInfo.s2LoopCount == 0;
    if constexpr (IS_BATCH_CONSISTENCY) {
        mqsmlaIsFirstVec2Base = mq35VecRunInfo.isFirstBase;
    }
    if (unlikely(mqsmlaIsFirstVec2Base)) {
        DataCopy(mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBufs[mq35VecRunInfo.taskIdMod2].tensor;
        if constexpr (IS_BATCH_CONSISTENCY) {
            if (mq35VecRunInfo.isLastBase) {
                LocalTensor<float> mqsmlaSumUb = this->softmaxFinalSumBufs[mq35VecRunInfo.taskIdMod2].tensor;
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2ResUb, expUb, expUb, mqsmlaSumUb,
                    mq35VecRunInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2ResUb, expUb, expUb, mq35VecRunInfo.vec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
            }
        } else {
            if (mq35VecRunInfo.s2LoopCount < mq35VecRunInfo.s2LoopLimit) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2ResUb, expUb, expUb, mq35VecRunInfo.vec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> mqsmlaSumUb = this->softmaxSumBufs[mq35VecRunInfo.multiCoreIdxMod2].tensor;
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    mqsmlaVec2ResUb, mqsmlaMmRes, mqsmlaVec2ResUb, expUb, expUb, mqsmlaSumUb,
                    mq35VecRunInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            }
        }
    }

    if constexpr (IS_BATCH_CONSISTENCY) {
        if (mq35VecRunInfo.isLastBase) {
            if (unlikely(mqsmlaIsFirstVec2Base)) {
                LocalTensor<float> mqsmlaSumUb = this->softmaxFinalSumBufs[mq35VecRunInfo.taskIdMod2].tensor;
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    mqsmlaVec2ResUb, mqsmlaVec2ResUb, mqsmlaSumUb, mq35VecRunInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
            if (mqsmlaNeedIntraBlockReduce) {
                SetFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
                SetFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
                ReduceIntraBlockAndStage(mq35VecRunInfo, mq35VecConstInfo, mqsmlaVec2ResUb, mqsmlaMmRes);
                if (!mq35VecRunInfo.isCrossCoreSplit && mq35VecRunInfo.s2LoopCount == mq35VecRunInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(mq35VecRunInfo, mq35VecConstInfo, mqsmlaVec2ResUb, 0, mqsmlaVec2CalcSize);
                }
            } else {
                if (mq35VecRunInfo.isFirstS2SplitCore && mq35VecRunInfo.reduceBlockId == 0 &&
                    mq35VecRunInfo.s2LoopCount < mq35VecRunInfo.s2LoopLimit) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                        mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(true),
                        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    int64_t mqsmlaStagingMOffset = GetFaStagingMOffset(mq35VecRunInfo, mq35VecConstInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(
                        stagingLayout, intraCoreCombineGm, GetIntraCoreWorkspaceIdx(mq35VecRunInfo, mq35VecConstInfo),
                        mqsmlaStagingMOffset, mq35VecRunInfo.vec2MRealSize,
                        static_cast<uint32_t>(mq35VecConstInfo.dSizeV), mqsmlaVec2ResUb, INNERCORE_STAGE2,
                        INNERCORE_STAGE_FD_MTE3_V);
                    SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_INTRAATTN_MTE3_MTE2(mq35VecRunInfo.multiCoreIdxMod2));
                }
                if (mq35VecRunInfo.isCrossCoreSplit &&
                    (!mq35VecRunInfo.isFirstS2SplitCore ||
                     (mq35VecRunInfo.isFirstS2SplitCore && mq35VecRunInfo.reduceBlockId == 0 &&
                      mq35VecRunInfo.s2LoopCount == mq35VecRunInfo.s2LoopLimit))) {
                    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                        mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(false),
                        AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                    uint32_t mqsmlaWorkspaceIdx = GetCrossCoreWorkspaceIdx(mq35VecRunInfo);
                    int64_t mqsmlaStagingMOffset = GetFaStagingMOffset(mq35VecRunInfo, mq35VecConstInfo);
                    AttentionCommon::StageVec2PartialOAndWait<T>(
                        stagingLayout, crossCoreCombineGm, mqsmlaWorkspaceIdx, mqsmlaStagingMOffset,
                        mq35VecRunInfo.vec2MRealSize, static_cast<uint32_t>(mq35VecConstInfo.dSizeV), mqsmlaVec2ResUb,
                        INNERCORE_STAGE2, INNERCORE_STAGE_FD_MTE3_V);
                } else if (!mq35VecRunInfo.isCrossCoreSplit &&
                           mq35VecRunInfo.s2LoopCount == mq35VecRunInfo.s2LoopLimit) {
                    this->CopyOutAttentionOut(mq35VecRunInfo, mq35VecConstInfo, mqsmlaVec2ResUb, 0, mqsmlaVec2CalcSize);
                }
            }
        }
    } else {
        if (mq35VecRunInfo.s2LoopCount == mq35VecRunInfo.s2LoopLimit) {
            if (unlikely(mq35VecRunInfo.s2LoopCount == 0)) {
                LocalTensor<float> mqsmlaSumUb = this->softmaxSumBufs[mq35VecRunInfo.multiCoreIdxMod2].tensor;
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                    mqsmlaVec2ResUb, mqsmlaVec2ResUb, mqsmlaSumUb, mq35VecRunInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
            if (mq35VecRunInfo.isCrossCoreSplit) {
                AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                    mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(),
                    AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW, AttentionCommon::FD_REDUCE_CHUNK_ROWS};
                uint32_t mqsmlaWorkspaceIdx = GetCrossCoreWorkspaceIdx(mq35VecRunInfo);
                int64_t mqsmlaStagingMOffset = GetFaStagingMOffset(mq35VecRunInfo, mq35VecConstInfo);
                AttentionCommon::StageVec2PartialOAndWait<T>(
                    stagingLayout, stagingOutGm, mqsmlaWorkspaceIdx, mqsmlaStagingMOffset, mq35VecRunInfo.vec2MRealSize,
                    static_cast<uint32_t>(mq35VecConstInfo.dSizeV), mqsmlaVec2ResUb, INNERCORE_STAGE2,
                    INNERCORE_STAGE_FD_MTE3_V);
            } else {
                this->CopyOutAttentionOut(mq35VecRunInfo, mq35VecConstInfo, mqsmlaVec2ResUb, 0, mqsmlaVec2CalcSize);
            }
        }
    }
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessFlashDecode(FdRunInfo& fdRunInfo,
                                                                               ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    InitFDBuffers(fdRunInfo);
    int64_t seqOffset = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        seqOffset = this->cuSeqlensQGm.GetValue(fdRunInfo.bn2Idx);
    } else {
        seqOffset = fdRunInfo.bn2Idx * mq35VecConstInfo.s1Size;
    }
    int64_t attentionOutOffset = seqOffset * mq35VecConstInfo.n2GDv + fdRunInfo.mIdx * mq35VecConstInfo.n2GDv +
                                 fdRunInfo.mStartIdx * mq35VecConstInfo.dSizeV;
    int64_t softmaxLseOffset = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        softmaxLseOffset = (seqOffset + fdRunInfo.mIdx) * mq35VecConstInfo.gSize + fdRunInfo.mStartIdx;
    } else {
        softmaxLseOffset = (fdRunInfo.bn2Idx * mq35VecConstInfo.s1Size + fdRunInfo.mIdx) * mq35VecConstInfo.gSize +
                           fdRunInfo.mStartIdx;
    }
    LocalTensor<T> mqsmlaAccumulatedO = this->fdBuffers.accumOut.tensor.template ReinterpretCast<T>();
    LocalTensor<float> mqsmlaLseExpUb = this->fdBuffers.lseExp.tensor.template ReinterpretCast<float>();
    LocalTensor<float> mqsmlaBlockMaxUb = this->fdBuffers.blockMax.tensor.template ReinterpretCast<float>();
    LocalTensor<float> mqsmlaBlockSumUb = this->fdBuffers.blockSum.tensor.template ReinterpretCast<float>();
    LocalTensor<T> mqsmlaPartialOFp32 = this->fdBuffers.partialO.tensor.template ReinterpretCast<T>();

    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
        mq35VecConstInfo.gSize, dTemplateAlign64, GetMq35StagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
        AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    int64_t attentionOutRowStride = static_cast<int64_t>(mq35VecConstInfo.dSizeV) +
                                    static_cast<int64_t>(mq35VecConstInfo.attentionOutStride) / sizeof(OUTPUT_T);
    int64_t startRow = 0;
    bool softmaxLseFlag = false;
    if constexpr (!HIGH_PERF) {
        softmaxLseFlag = mq35VecConstInfo.isSoftmaxLseEnable;
    }
    while (startRow < fdRunInfo.mNum) {
        int64_t mqsmlaDealRowCount = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
        if (startRow + mqsmlaDealRowCount > fdRunInfo.mNum) {
            mqsmlaDealRowCount = fdRunInfo.mNum - startRow;
        }
        WaitFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
        if constexpr (IS_BATCH_CONSISTENCY) {
            WaitFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
            AttentionCommon::ReducePairwiseWithLse<T, dTemplateAlign64>(
                stagingLayout, fdStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
                static_cast<int64_t>(fdRunInfo.mStartIdx) + startRow, mqsmlaDealRowCount,
                static_cast<int64_t>(mq35VecConstInfo.dSizeV), mqsmlaAccumulatedO, mqsmlaLseExpUb, mqsmlaBlockMaxUb,
                mqsmlaBlockSumUb, mqsmlaPartialOFp32, softmaxLseFlag, softmaxLseGm, softmaxLseOffset + startRow,
                INNERCORE_FD_V_MTE2(0), INNERCORE_FD_V_MTE2(1), INNERCORE_FD_MTE2_V, INNERCORE_LSE_V_MTE3,
                INNERCORE_LSE_MTE3_V);
        } else {
            AttentionCommon::ReduceWithLse<T, dTemplateAlign64>(
                stagingLayout, fdStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
                static_cast<int64_t>(fdRunInfo.mStartIdx) + startRow, mqsmlaDealRowCount,
                static_cast<int64_t>(mq35VecConstInfo.dSizeV), mqsmlaAccumulatedO, mqsmlaLseExpUb, mqsmlaBlockMaxUb,
                mqsmlaBlockSumUb, mqsmlaPartialOFp32, softmaxLseFlag, softmaxLseGm, softmaxLseOffset + startRow,
                INNERCORE_FD_V_MTE2(0), INNERCORE_FD_V_MTE2(1), INNERCORE_FD_MTE2_V, INNERCORE_LSE_V_MTE3,
                INNERCORE_LSE_MTE3_V);
        }

        RunInfo<HIGH_PERF> mq35VecRunInfo;
        mq35VecRunInfo.vec2MRealSize = mqsmlaDealRowCount;
        mq35VecRunInfo.attentionOutOffset = attentionOutOffset + startRow * attentionOutRowStride;
        int64_t mqsmlaVec2CalcSize = mqsmlaDealRowCount * dTemplateAlign64;
        this->CopyOutAttentionOut(mq35VecRunInfo, mq35VecConstInfo, mqsmlaAccumulatedO, 0, mqsmlaVec2CalcSize);
        if constexpr (IS_BATCH_CONSISTENCY) {
            SetFlag<HardEvent::MTE3_MTE2>(INNERCORE_FD_MTE3_MTE2);
        }
        SetFlag<HardEvent::MTE3_V>(INNERCORE_FD_MTE3_V);
        startRow += mqsmlaDealRowCount;
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                                                            ConstInfo<HIGH_PERF>& mq35VecConstInfo,
                                                                            LocalTensor<VEC2_RES_T>& mq35Vec2ResultUb,
                                                                            int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> mqsmlaAttenOut;
    int64_t mqsmlaDSizeAligned64 = (int64_t)dTemplateAlign64;

    mqsmlaAttenOut.SetAddr(mq35Vec2ResultUb.address_);
    Cast(mqsmlaAttenOut, mq35Vec2ResultUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(INNERCORE_STAGE2);
    WaitFlag<HardEvent::V_MTE3>(INNERCORE_STAGE2);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = mq35VecConstInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride =
        (mqsmlaDSizeAligned64 - mq35VecConstInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = mq35VecConstInfo.attentionOutStride;
    dataCopyParams.blockCount = mq35VecRunInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[mq35VecRunInfo.attentionOutOffset], mqsmlaAttenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<HIGH_PERF>& mq35VecRunInfo, ConstInfo<HIGH_PERF>& mq35VecConstInfo,
    LocalTensor<VEC2_RES_T>& mq35Vec2ResultUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(mq35VecRunInfo, mq35VecConstInfo, mq35Vec2ResultUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint32_t vecCoreNum = CV_RATIO * coreNum;
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        totalOutputSize =
            mq35VecConstInfo.bSize * mq35VecConstInfo.gSize * mq35VecConstInfo.s1Size * mq35VecConstInfo.dSizeV;
    } else if (LAYOUT_T == QSMLA_LAYOUT::TND) {
        totalOutputSize = mq35VecConstInfo.s1Size * mq35VecConstInfo.gSize * mq35VecConstInfo.dSizeV;
    }

    static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 184U * 1024U;
    static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = (32U * 1024U) / sizeof(OUTPUT_T);
    if (coreNum != 0 && totalOutputSize > 0) {
        AttentionCommon::InitOutput<OUTPUT_T, initOutputEventId, ATTEN_OUT_POP_BUF_START_ADDR,
                                    ATTEN_OUT_POP_BUF_ELE_SIZE, false>(this->attentionOutGm, totalOutputSize,
                                                                       vecCoreNum, static_cast<OUTPUT_T>(0));
    }
    if constexpr (!HIGH_PERF) {
        if (mq35VecConstInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid) {
            uint64_t totalLseSize = 0;
            if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
                totalLseSize =
                    mq35VecConstInfo.bSize * mq35VecConstInfo.n2Size * mq35VecConstInfo.gSize * mq35VecConstInfo.s1Size;
            } else if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
                totalLseSize = mq35VecConstInfo.n2Size * mq35VecConstInfo.s1Size * mq35VecConstInfo.gSize;
            }
            static constexpr uint32_t LSE_POP_BUF_START_ADDR = 216U * 1024U;
            static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = (32U * 1024U) / sizeof(float);
            if (coreNum != 0 && totalLseSize > 0) {
                AttentionCommon::InitOutput<float, initOutputEventId, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE,
                                            false>(this->softmaxLseGm, totalLseSize, vecCoreNum, static_cast<float>(0));
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t* attentionOut,
                                                                        __gm__ uint8_t* softmaxLse,
                                                                        ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if constexpr (!HIGH_PERF) {
            if (mq35VecConstInfo.isSoftmaxLseEnable && softmaxLse != nullptr) {
                this->softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
                this->isSoftmaxLseGmValid = true;
            }
        }
        if (mq35VecConstInfo.needInit == 1) {
            InitOutputSingleCore(mq35VecConstInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetSeqLenForPhyAddr(int32_t bIdx, bool hasActualSeq,
                                                                                   bool hasCuSeqlens,
                                                                                   GlobalTensor<int32_t>& actualSeqGm,
                                                                                   GlobalTensor<int32_t>& cuSeqlensGm,
                                                                                   int64_t defaultSize)
{
    if (hasActualSeq) {
        return actualSeqGm.GetValue(bIdx);
    }
    if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(bIdx + 1) - cuSeqlensGm.GetValue(bIdx);
    }
    return defaultSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline Mq35PhyAddrValidInfo MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CalcPhyAddrValidInfo(
    bool isOriKv, int32_t actualS1Size, int32_t actualOriS2Size, int64_t restoredSize,
    ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    // per-batch执行一次,  per-s1循环内不再判断maskmode
    Mq35PhyAddrValidInfo mq35ValidWindow;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        mq35ValidWindow.oriS2Act = actualOriS2Size;
        if constexpr (HIGH_PERF) {
            mq35ValidWindow.oriLeftBias = (mq35VecConstInfo.oriWinLeft == -1) ? Mq35PhyAddrValidInfo::BIAS_UNBOUND :
                                                                                mq35VecConstInfo.oriWinLeft + 1;
            mq35ValidWindow.oriRightBias = (mq35VecConstInfo.oriWinRight == -1) ? Mq35PhyAddrValidInfo::BIAS_UNBOUND :
                                                                                  mq35VecConstInfo.oriWinRight;
        } else {
            if (isOriKv) {
                if (mq35VecConstInfo.oriMaskMode == 0U) {
                    mq35ValidWindow.oriTopkMode = true;
                } else if (mq35VecConstInfo.oriMaskMode == 3U) {
                    mq35ValidWindow.oriRightBias = 0;
                } else {
                    mq35ValidWindow.oriLeftBias = (mq35VecConstInfo.oriWinLeft == -1) ?
                                                      Mq35PhyAddrValidInfo::BIAS_UNBOUND :
                                                      mq35VecConstInfo.oriWinLeft + 1;
                    mq35ValidWindow.oriRightBias = (mq35VecConstInfo.oriWinRight == -1) ?
                                                       Mq35PhyAddrValidInfo::BIAS_UNBOUND :
                                                       mq35VecConstInfo.oriWinRight;
                }
            }
        }
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            mq35ValidWindow.cmpTopkMode = (mq35VecConstInfo.cmpMaskMode == 0U);
            mq35ValidWindow.cmpBase = restoredSize - actualS1Size + 1;
        }
    }
    return mq35ValidWindow;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CalcCurValidS2ForPhyAddr(
    uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv, GlobalTensor<int32_t>& cuSeqlensQGm,
    GlobalTensor<int32_t>& topkLengthGm, ConstInfo<HIGH_PERF>& mq35VecConstInfo, int32_t sparseBlockCount,
    const Mq35PhyAddrValidInfo& mq35ValidWindow)
{
    bool topkMode = false;
    bool hasTopk = false;
    if constexpr (!HIGH_PERF) {
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv) {
                topkMode = mq35ValidWindow.oriTopkMode;
                hasTopk = mq35VecConstInfo.hasOriTopkLength;
            }
        }
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv) {
                topkMode = mq35ValidWindow.cmpTopkMode;
                hasTopk = mq35VecConstInfo.hasCmpTopkLength;
            }
        }
    }
    if (topkMode) {
        uint64_t topkIdx = (LAYOUT_T == QSMLA_LAYOUT::TND) ? (cuSeqlensQGm.GetValue(bIdx) + s1Idx) :
                                                             (bIdx * mq35VecConstInfo.s1Size + s1Idx);
        int32_t topkLen = hasTopk ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
        return Min(topkLen, sparseBlockCount);
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            int64_t thr = mq35ValidWindow.oriS2Act - actualS1Size + 1 + s1Idx;
            int64_t leftBound = Max(thr - mq35ValidWindow.oriLeftBias, 0);
            int64_t rightBound =
                Min(thr + mq35ValidWindow.oriRightBias, static_cast<int64_t>(mq35ValidWindow.oriS2Act));
            return Min(static_cast<int32_t>(Max(0, rightBound - leftBound)), sparseBlockCount);
        }
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t numerator = Max(mq35ValidWindow.cmpBase + s1Idx, 0);
        return Min(sparseBlockCount, static_cast<int32_t>(numerator / static_cast<int32_t>(mq35VecConstInfo.cmpRatio)));
    }
    return 0;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetKVPhyAddrForKvType(
    uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
    bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t>& actualSeqQlenGm,
    GlobalTensor<int32_t>& cuSeqlensQGm, GlobalTensor<int32_t>& actualSeqKvlenGm, GlobalTensor<int32_t>& cuSeqlensKvGm,
    GlobalTensor<int32_t>& topkLengthGm, GlobalTensor<int32_t>& cmpResidualKvGm, ConstInfo<HIGH_PERF>& mq35VecConstInfo,
    GlobalTensor<int32_t>& mqActiveBlockTableGm, GlobalTensor<int32_t>& mqActiveSparseIndicesGm,
    GlobalTensor<uint32_t>& phyAddrGm, uint32_t kvStride, uint32_t mqActiveBlockSize,
    uint32_t mqActiveMaxBlocksPerBatch, uint32_t sparseBlockCount, uint32_t alignedSparseBlockCount, bool isOriKv)
{
    constexpr uint16_t s2NumPerLoop = 128;
    constexpr uint32_t vecCoreNum = IS_SPLIT_G ? 4 : 2;
    uint32_t vecCoreIdx = IS_SPLIT_G ? mq35VecConstInfo.aivIdx % 4 : mq35VecConstInfo.aivIdx % 2;
    int16_t shiftRightNum = 0;
    for (int32_t size = static_cast<int32_t>(mqActiveBlockSize); size > 1; size >>= 1) {
        ++shiftRightNum;
    }

    uint32_t phyAddrUb = 0;
    LocalTensor<int32_t> blkTableUb(TPosition::VECIN, phyAddrUb, mqActiveMaxBlocksPerBatch);
    phyAddrUb = CeilAlign(phyAddrUb + mqActiveMaxBlocksPerBatch * sizeof(int32_t), BUFFER_SIZE_BYTE_32B);
    LocalTensor<int32_t> sparseIdxUb(TPosition::VECIN, phyAddrUb, alignedSparseBlockCount);
    phyAddrUb += alignedSparseBlockCount * sizeof(int32_t);
    LocalTensor<uint32_t> kvPhyAddrUb(TPosition::VECIN, phyAddrUb,
                                      alignedSparseBlockCount * 2); // 2：每个稀疏块需要存储两个物理地址

    int64_t totalValidS1 = 0;
    uint32_t tmpGS1Start = gS1StartIdx;
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx; ++bIdx) {
        bool lastBN = bIdx == bN2EndIdx - 1;
        int32_t actualS1Size = GetSeqLenForPhyAddr(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm,
                                                   mq35VecConstInfo.s1Size);
        int32_t s1End = (lastBN && nextGs1Idx != 0) ? nextGs1Idx : actualS1Size;
        int64_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && mq35VecConstInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                           cuSeqlensKvGm, mq35VecConstInfo.cmpS2Size);
                int64_t residual = (mq35VecConstInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
                restoredSize =
                    static_cast<int64_t>(actualKvSize) * static_cast<int64_t>(mq35VecConstInfo.cmpRatio) + residual;
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && mq35VecConstInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                      cuSeqlensKvGm, mq35VecConstInfo.s2Size);
            }
        }

        Mq35PhyAddrValidInfo mq35ValidWindow =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, mq35VecConstInfo);
        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t validS2 = CalcCurValidS2ForPhyAddr(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                       mq35VecConstInfo, sparseBlockCount, mq35ValidWindow);
            totalValidS1 += validS2 > 0;
        }
        tmpGS1Start = 0;
    }
    int64_t rowsPerCore = totalValidS1 / vecCoreNum;
    int64_t tailRows = totalValidS1 % vecCoreNum;
    int64_t rowStart = rowsPerCore * vecCoreIdx + Min(static_cast<int64_t>(vecCoreIdx), tailRows);
    int64_t rowCount = rowsPerCore + (vecCoreIdx < static_cast<uint32_t>(tailRows) ? 1 : 0);
    if (rowCount == 0) {
        return;
    }

    int64_t validCounter = 0;
    int64_t processedCount = 0;
    tmpGS1Start = gS1StartIdx;
    bool done = false;
    SetFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
    SetFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
    SetFlag<HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx && !done; ++bIdx) {
        bool lastBN = bIdx == bN2EndIdx - 1;
        int32_t actualS1Size = GetSeqLenForPhyAddr(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm,
                                                   mq35VecConstInfo.s1Size);
        int64_t bS1Idx = (LAYOUT_T == QSMLA_LAYOUT::TND) ?
                             (hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : mq35VecConstInfo.s1Size * bIdx) :
                             mq35VecConstInfo.s1Size * bIdx;
        int32_t s1End = (lastBN && nextGs1Idx != 0) ? nextGs1Idx : actualS1Size;
        int64_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && mq35VecConstInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                           cuSeqlensKvGm, mq35VecConstInfo.cmpS2Size);
                int64_t residual = (mq35VecConstInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
                restoredSize =
                    static_cast<int64_t>(actualKvSize) * static_cast<int64_t>(mq35VecConstInfo.cmpRatio) + residual;
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && mq35VecConstInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                      cuSeqlensKvGm, mq35VecConstInfo.s2Size);
            }
        }

        Mq35PhyAddrValidInfo mq35ValidWindow =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, mq35VecConstInfo);
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
        AttentionCommon::CopyPaTableToUb(blkTableUb, bIdx, mqActiveBlockTableGm, mqActiveMaxBlocksPerBatch);
        SetFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
        WaitFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t validS2 = CalcCurValidS2ForPhyAddr(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                       mq35VecConstInfo, sparseBlockCount, mq35ValidWindow);
            if (validS2 <= 0) {
                continue;
            }
            if (validCounter < rowStart || validCounter >= rowStart + rowCount) {
                ++validCounter;
                continue;
            }
            ++validCounter;
            uint16_t s2Loop = (validS2 + s2NumPerLoop - 1) / s2NumPerLoop;
            int32_t s2Tail = validS2 - (s2Loop - 1) * s2NumPerLoop;
            WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
            AttentionCommon::CopySparseIdxToUb(sparseIdxUb, bS1Idx, s1Idx, validS2, mqActiveSparseIndicesGm,
                                               sparseBlockCount);
            SetFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_SPARSEIDX_READY);
            WaitFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_SPARSEIDX_READY);
            WaitFlag<HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
            AttentionCommon::GetKVPhyAddrVFPa<uint32_t>(
                kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, mqActiveBlockSize, shiftRightNum,
                mq35VecConstInfo.sparseBlockSize, mq35VecConstInfo.dSizeVInput, kvStride);
            SetFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
            SetFlag<HardEvent::V_MTE3>(INNERCORE_PHYADDR_KVADDR_READY);
            WaitFlag<HardEvent::V_MTE3>(INNERCORE_PHYADDR_KVADDR_READY);
            AttentionCommon::CopyPhyAddrToGm(kvPhyAddrUb, bS1Idx, s1Idx, validS2, s2NumPerLoop, phyAddrGm,
                                             alignedSparseBlockCount);
            SetFlag<HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
            if (++processedCount >= rowCount) {
                done = true;
                break;
            }
        }
        SetFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
        tmpGS1Start = 0;
    }
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_SPARSEIDX_FREE);
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_PHYADDR_KVADDR_FREE);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetKVPhyAddr(
    uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx,
    bool hasActualSeqQlen, bool hasCuSeqlensQ, bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv,
    GlobalTensor<int32_t>& actualSeqOriKvlenGm, GlobalTensor<int32_t>& cuSeqlensOriKvGm,
    GlobalTensor<int32_t>& oriTopkLengthGm, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv,
    GlobalTensor<int32_t>& actualSeqCmpKvlenGm, GlobalTensor<int32_t>& cuSeqlensCmpKvGm,
    GlobalTensor<int32_t>& cmpTopkLengthGm, GlobalTensor<int32_t>& cmpResidualKvGm,
    GlobalTensor<int32_t>& actualSeqQlenGm, GlobalTensor<int32_t>& cuSeqlensQGm, __gm__ uint8_t* workspace,
    ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    if (hasLoad == 0) {
        return;
    }
    uint64_t v0ResSize = static_cast<uint64_t>(mq35VecConstInfo.s2BaseSize) * mq35VecConstInfo.dSize * sizeof(Q_T);
    uint64_t v0RegionSize = v0ResSize * 3 * (IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum());
    uint64_t totalBS1 = (LAYOUT_T == QSMLA_LAYOUT::TND) ?
                            mq35VecConstInfo.s1Size :
                            static_cast<uint64_t>(mq35VecConstInfo.bSize) * mq35VecConstInfo.s1Size;
    uint64_t oriPhyAddrSize = 0;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriPhyAddrSize = totalBS1 * mq35VecConstInfo.alignedOriSparseBlockCount * sizeof(int64_t);
        oriKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t*)(workspace + v0RegionSize));
        GetKVPhyAddrForKvType(
            bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ, hasActualSeqOriKvlen,
            hasCuSeqlensOriKv, actualSeqQlenGm, cuSeqlensQGm, actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm,
            cmpResidualKvGm, mq35VecConstInfo, mqsmlaOriBlockTableGm, mqsmlaOriSparseIndicesGm, oriKvPhyAddrGm,
            mq35VecConstInfo.oriKvStride, mq35VecConstInfo.oriBlockSize, mq35VecConstInfo.oriMaxBlockNumPerBatch,
            mq35VecConstInfo.oriSparseBlockCount, mq35VecConstInfo.alignedOriSparseBlockCount, true);
    }
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t*)(workspace + v0RegionSize + oriPhyAddrSize));
        GetKVPhyAddrForKvType(
            bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ, hasActualSeqCmpKvlen,
            hasCuSeqlensCmpKv, actualSeqQlenGm, cuSeqlensQGm, actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm,
            cmpResidualKvGm, mq35VecConstInfo, mqsmlaCmpBlockTableGm, mqsmlaCmpSparseIndicesGm, cmpKvPhyAddrGm,
            mq35VecConstInfo.cmpKvStride, mq35VecConstInfo.cmpBlockSize, mq35VecConstInfo.cmpMaxBlockNumPerBatch,
            mq35VecConstInfo.cmpSparseBlockCount, mq35VecConstInfo.alignedCmpSparseBlockCount, false);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
    __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
    __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T*)(oriKV));
    if (oriBlockTable != nullptr) {
        mqsmlaOriBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)oriBlockTable);
    }

    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T*)cmpKV);
        if (cmpBlockTable != nullptr) {
            mqsmlaCmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)cmpBlockTable);
        }
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        mqsmlaOriSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)oriSparseIndices);
    }

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        mqsmlaCmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)cmpSparseIndices);
    }

    if (sinks != nullptr) {
        mqsmlaSinksGm.SetGlobalBuffer((__gm__ T*)sinks);
        this->mqsmlaHasSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::SoftmaxInitBuffer(uint32_t& ubBaseAddr)
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    constexpr uint32_t softmaxElems = softmaxBufSize / sizeof(float);
    softmaxSumBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 0};
    ubBaseAddr += softmaxBufSize;
    softmaxSumBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 1};
    ubBaseAddr += softmaxBufSize;
    softmaxMaxBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 0};
    ubBaseAddr += softmaxBufSize;
    softmaxMaxBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 1};
    ubBaseAddr += softmaxBufSize;
    if constexpr (IS_BATCH_CONSISTENCY) {
        softmaxFinalSumBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 0};
        ubBaseAddr += softmaxBufSize;
        softmaxFinalSumBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 1};
        ubBaseAddr += softmaxBufSize;
        softmaxFinalMaxBufs[0] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 0};
        ubBaseAddr += softmaxBufSize;
        softmaxFinalMaxBufs[1] = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, softmaxElems), 1};
        ubBaseAddr += softmaxBufSize;
        batchReduceTmpUb = {LocalTensor<float>(TPosition::VECIN, ubBaseAddr, 768), 0};
        ubBaseAddr += 768U * sizeof(float);
    }
    softmaxExpBufs[0] = {LocalTensor<T>(TPosition::VECIN, ubBaseAddr, softmaxBufSize / sizeof(T)), 0};
    ubBaseAddr += softmaxBufSize;
    softmaxExpBufs[1] = {LocalTensor<T>(TPosition::VECIN, ubBaseAddr, softmaxBufSize / sizeof(T)), 1};
    ubBaseAddr += softmaxBufSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    LocalTensor<T> mqsmlaSinksUb = this->sinksUb.tensor;
    const uint32_t mqsmlaMaxN = mq35VecConstInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = mqsmlaMaxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(mqsmlaSinksUb, this->mqsmlaSinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_SINKS_SYNC);
    WaitFlag<AscendC::HardEvent::MTE2_V>(INNERCORE_SINKS_SYNC);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitLocalBuffer(ConstInfo<HIGH_PERF>& mq35VecConstInfo,
                                                                            uint32_t ubBaseAddr)
{
    uint32_t mqsmlaUbAddr = ubBaseAddr;

    // {64, 16, 2}：每个向量操作单元处理的元素个数为64，每个处理块包含的向量数量为16，需要处理两个不同的数据流或阶段
    dequantScaleUb = {LocalTensor<float>(TPosition::VECIN, mqsmlaUbAddr, 64 * 16 * 2), 0};
    mqsmlaUbAddr += 64 * 16 * 2 * sizeof(float); // {64, 16, 2}：同上

    SoftmaxInitBuffer(mqsmlaUbAddr);

    commonUb = {LocalTensor<T>(TPosition::VECIN, mqsmlaUbAddr, 512 / sizeof(T)), 0}; // 512：缓冲区大小，单位是字节
    mqsmlaUbAddr += 512;                                                             // 512：同上
    sinksUb = {LocalTensor<T>(TPosition::VECIN, mqsmlaUbAddr, 512 / sizeof(T)), 0}; // 512：同上
    mqsmlaUbAddr += 512;                                                            // 512：同上
    if constexpr (!HIGH_PERF) {
        if (mq35VecConstInfo.isSoftmaxLseEnable) {
            outLseUbs[0] = {LocalTensor<float>(TPosition::VECIN, mqsmlaUbAddr, 256 / sizeof(float)),
                            0}; // 256：缓冲区大小，单位是字节
            mqsmlaUbAddr += 256U;
            outLseUbs[1] = {LocalTensor<float>(TPosition::VECIN, mqsmlaUbAddr, 256 / sizeof(float)), 1}; // 256：同上
            mqsmlaUbAddr += 256U;
        }
    }

    stage0InBufs[0] = {LocalTensor<KV_T>(TPosition::VECIN, mqsmlaUbAddr, v0BufferDSize * 16),
                       0};                             // 16：每个v0缓冲区块中存储的向量数量
    mqsmlaUbAddr += v0BufferDSize * 16 * sizeof(KV_T); // 16：同上
    stage0InBufs[1] = {LocalTensor<KV_T>(TPosition::VECIN, mqsmlaUbAddr, v0BufferDSize * 16), 1}; // 16：同上
    mqsmlaUbAddr += v0BufferDSize * 16 * sizeof(KV_T);                                            // 16：同上
    stage0OutBufs[0] = {LocalTensor<Q_T>(TPosition::VECIN, mqsmlaUbAddr, v0BufferDSize * (16U + 1)), 0};
    mqsmlaUbAddr += v0BufferDSize * (16U + 1) * sizeof(Q_T);
    stage0OutBufs[1] = {LocalTensor<Q_T>(TPosition::VECIN, mqsmlaUbAddr, v0BufferDSize * (16U + 1)), 1};
    mqsmlaUbAddr += v0BufferDSize * (16U + 1) * sizeof(Q_T);

    stage1OutBufs[0] = {LocalTensor<Q_T>(TPosition::VECIN, mqsmlaUbAddr, vec1Srcstride * s2BaseSize), 0};
    mqsmlaUbAddr += vec1Srcstride * s2BaseSize * sizeof(Q_T);
    stage1OutBufs[1] = {LocalTensor<Q_T>(TPosition::VECIN, mqsmlaUbAddr, vec1Srcstride * s2BaseSize), 1};
    mqsmlaUbAddr += vec1Srcstride * s2BaseSize * sizeof(Q_T);

    stage2OutBufs = {LocalTensor<T>(TPosition::VECIN, mqsmlaUbAddr, (s1BaseSize / CV_RATIO) * dTemplateAlign64), 0};

    // 显式 flag 初始化 (替代 AllocEventID + 初始 SetFlag)
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    if constexpr (IS_BATCH_CONSISTENCY) {
        SetFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
        SetFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
    }
    if constexpr (!HIGH_PERF) {
        if (mq35VecConstInfo.isSoftmaxLseEnable) {
            SetFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
    }
    SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(0));
    SetFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(1));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(0));
    SetFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(1));
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
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitSinks(ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    if ASCEND_IS_AIV {
        if (this->mqsmlaHasSinks) {
            InitSinksBuffer(mq35VecConstInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::FreeEvent(ConstInfo<HIGH_PERF>& mq35VecConstInfo)
{
    if constexpr (IS_BATCH_CONSISTENCY) {
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_INTRAPARTIALO_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_REDUCE_MAXSUM_V_MTE2);
    }
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2);
    if constexpr (!HIGH_PERF) {
        if (mq35VecConstInfo.isSoftmaxLseEnable) {
            WaitFlag<HardEvent::MTE3_V>(INNERCORE_LSE_MTE3_V);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(0));
    WaitFlag<HardEvent::V_MTE2>(INNERCORE_STAGE0_IN(1));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(0));
    WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE0_OUT(1));
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
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitFDBuffers(FdRunInfo& fdRunInfo)
{
    FdRunInfo mqsmlaFdBufferInfo = fdRunInfo;
    if (mqsmlaFdBufferInfo.mNum > AttentionCommon::FD_REDUCE_CHUNK_ROWS) {
        mqsmlaFdBufferInfo.mNum = AttentionCommon::FD_REDUCE_CHUNK_ROWS;
    }
    AttentionCommon::InitFDBuffersStatic<T, dTemplateAlign64>(mqsmlaFdBufferInfo, 0, fdBuffers);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetExtremeValue(T& negativeScalar)
{
    uint32_t mqsmlaTmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float*)&mqsmlaTmp1);
}

TEMPLATES_DEF
class CSABlockVecDummy {
public:
    __aicore__ inline CSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
                                       ConstInfo<HIGH_PERF>& mq35VecConstInfo)
    {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                            __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                            __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
                                            __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv,
                                            __gm__ uint8_t* cmpResidualKv)
    {}
    __aicore__ inline void InitVecBlock(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                        __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedOriKv,
                                        __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv) {};
    __aicore__ inline void InitLocalBuffer(ConstInfo<HIGH_PERF>& mq35VecConstInfo, uint32_t ubBaseAddr) {}
    __aicore__ inline void InitSinks(ConstInfo<HIGH_PERF>& mq35VecConstInfo) {}
    __aicore__ inline void InitFDBuffers(FdRunInfo& fdRunInfo) {}
    __aicore__ inline void ProcessVec1(StaticBuffer<Q_T>& outputBuf, StaticBuffer<T>& bmm1ResBuf,
                                       RunInfo<HIGH_PERF>& mq35VecRunInfo, ConstInfo<HIGH_PERF>& mq35VecConstInfo)
    {}

    __aicore__ inline void ProcessVec2(StaticBuffer<T>& bmm2ResBuf, RunInfo<HIGH_PERF>& mq35VecRunInfo,
                                       ConstInfo<HIGH_PERF>& mq35VecConstInfo)
    {}
    __aicore__ inline void FreeEvent(ConstInfo<HIGH_PERF>& mq35VecConstInfo) {}
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC>& fdStaging) {}
    __aicore__ inline void InitS2SplitStaging(Buffer<BufferType::GM, SyncType::NO_SYNC>& mqsmlaIntraCoreBuffer,
                                              Buffer<BufferType::GM, SyncType::NO_SYNC>& mqsmlaCrossCoreBuffer)
    {}
    __aicore__ inline void ProcessFlashDecode(FdRunInfo& fdRunInfo, ConstInfo<HIGH_PERF>& mq35VecConstInfo) {}
};
} // namespace BaseApi
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
