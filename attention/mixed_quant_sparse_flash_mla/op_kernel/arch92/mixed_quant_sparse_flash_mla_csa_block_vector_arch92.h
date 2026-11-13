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
 * \file mixed_quant_sparse_flash_mla_csa_block_vector_arch92.h
 * \brief
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H

#include "util_regbase_arch92.h"
#include "mixed_quant_sparse_flash_mla_common_arch92.h"
#include "../../common/op_kernel/buffers_policy.h"
#include "../../common/op_kernel/attn_buffer_manager.h"
#include "../../common/op_kernel/attn_buffer.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#include "../../common/op_kernel/arch92/vf/vf_flashupdate_new_sfa.h"
#include "../../../sparse_flash_mla/op_kernel/arch35/common/get_kv_phy_addr_vf.h"
#include "../../../sparse_flash_mla/op_kernel/arch35/common/smla_vector_common_arch35.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace optiling;
using namespace optiling::detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {

using mmResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
using outputl1Pos = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
using gmPos = Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD>;

struct Mq92PhyAddrValidInfo {
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
    static constexpr uint32_t vec1Srcstride = s1BaseSize + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr uint32_t dVTemplateTypeInput = 608; // 608: rope(64 * 2) + nope(448) + scale(7 * 2) + pad(18)
    static constexpr uint32_t dCombineBytes = 576;       // rope(64 * 2) + nope(448)
    static constexpr uint32_t scaleBytes = 8;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint32_t dstGap = 384;
    static constexpr uint32_t NUM_6 = 6;
    static constexpr uint32_t NUM_128 = 128;
    static constexpr uint32_t NUM_256 = 256;
    static constexpr uint32_t NUM_384 = 384;

    // ==================== Functions ======================
    __aicore__ inline MqsmlaCsaBlockVector(){};
    __aicore__ inline void InitVecBlock(TPipe* pipe, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                        __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedOriKv,
                                        __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensOriKv);
            }
            if constexpr (TEMPLATE_MODE != MQSMLATemplateMode::SWA_TEMPLATE_MODE &&
                          TEMPLATE_MODE != MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
                if (cuSeqlensCmpKv != nullptr) {
                    cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensCmpKv);
                }
            }
            if (sequsedOriKv != nullptr) {
                actualSeqOriKvGm.SetGlobalBuffer((__gm__ int32_t*)sequsedOriKv);
            }
            if constexpr (TEMPLATE_MODE != MQSMLATemplateMode::SWA_TEMPLATE_MODE &&
                          TEMPLATE_MODE != MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
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
    __aicore__ inline void InitLocalBuffer(TPipe* pipe, ConstInfo& constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, ConstInfo& constInfo);
    __aicore__ inline void InitOutput(ConstInfo& constInfo, RunParamStr& runParam, int32_t sIdx, int32_t gS1Cur);
    __aicore__ inline void FreeEvent();
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                            __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                            __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
                                            __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv,
                                            __gm__ uint8_t* cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo& constInfo);
    __aicore__ inline void ProcessVec0(outputl1Pos& outputL1, gmPos& v0ResGm, const RunInfo& runInfo,
                                       ConstInfo& constInfo);
    __aicore__ inline void ProcessVec1(outputl1Pos& outputBuf, mmResPos& bmm1ResBuf, RunInfo& runInfo,
                                       ConstInfo& constInfo);
    __aicore__ inline void ProcessVec2(mmResPos& bmm2ResBuf, mmResPos& bmm2ResBuf1, RunInfo& runInfo,
                                       ConstInfo& constInfo);
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
                                        ConstInfo& constInfo);

private:
    __aicore__ inline void ProcessSparseKv(outputl1Pos& outputL1, gmPos& v0ResGm, const RunInfo& runInfo,
                                           ConstInfo& constInfo);
    __aicore__ inline void ProcessNotSparseKv(outputl1Pos& outputL1, gmPos& v0ResGm, const RunInfo& runInfo,
                                              ConstInfo& constInfo);
    __aicore__ inline void CalProSize(const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void CopyInMode2Rows(LocalTensor<KV_T> kvInUb, int64_t dealRows, int64_t firstToken, bool sparse,
                                           const RunInfo& runInfo, const ConstInfo& constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t dealRow, int64_t s2StartIdx,
                                             const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void DequantKv(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                     int64_t s2ProcessBaseSize, ConstInfo& constInfo);
    __aicore__ inline void CopyOutKvUb2L1(outputl1Pos& outputL1, LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow,
                                          int64_t s2StartIdx, const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(gmPos& v0ResGm, LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow,
                                          int64_t s2StartIdx, const RunInfo& runInfo, ConstInfo& constInfo);
    __aicore__ inline void CopyOutMrgeResult(outputl1Pos& outputL1, int64_t mte2Size, int64_t mte3Size,
                                             int64_t s2keyOffset, int64_t mergeMte3Idx, const RunInfo& runInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo& runInfo, ConstInfo& constInfo, LocalTensor<VEC2_RES_T>& vec2ResUb,
                                           int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo& runInfo, ConstInfo& constInfo,
                                               LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void GetExtremeValue(T& negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo& constInfo);
    __aicore__ inline void CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx, int64_t s1Idx,
                                           int64_t validS2, int64_t alignNum, ConstInfo& constInfo);
    __aicore__ inline void CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx, ConstInfo& constInfo);
    __aicore__ inline void CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx, int64_t s1Idx,
                                             int64_t validS2, ConstInfo& constInfo);
    __aicore__ inline int32_t GetActualS1Size(uint32_t bIdx, __gm__ int32_t* actualSeqQlenAddr,
                                              __gm__ int32_t* cuSeqlensQAddr, const ConstInfo& constInfo);
    __aicore__ inline int32_t GetActualS2Size(uint32_t bIdx, __gm__ int32_t* actualSeqKvlenAddr,
                                              const ConstInfo& constInfo);
    __aicore__ inline int32_t GetSeqLenForPhyAddr(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                                  GlobalTensor<int32_t>& actualSeqGm,
                                                  GlobalTensor<int32_t>& cuSeqlensGm, int64_t defaultSize);
    __aicore__ inline Mq92PhyAddrValidInfo CalcPhyAddrValidInfo(bool isOriKv, int32_t actualS1Size,
                                                                int32_t actualOriS2Size, int64_t restoredSize,
                                                                ConstInfo& constInfo);
    __aicore__ inline int32_t CalcCurValidS2ForPhyAddr(uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv,
                                                       GlobalTensor<int32_t>& cuSeqlensQGm,
                                                       GlobalTensor<int32_t>& topkLengthGm, ConstInfo& constInfo,
                                                       int32_t sparseBlockCount,
                                                       const Mq92PhyAddrValidInfo& mq92ValidWindow);
    __aicore__ inline void GetKVPhyAddrForKvType(
        uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
        bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t>& actualSeqQlenGm,
        GlobalTensor<int32_t>& cuSeqlensQGm, GlobalTensor<int32_t>& actualSeqKvlenGm,
        GlobalTensor<int32_t>& cuSeqlensKvGm, GlobalTensor<int32_t>& topkLengthGm,
        GlobalTensor<int32_t>& cmpResidualKvGm, ConstInfo& constInfo, GlobalTensor<int32_t>& mqActiveBlockTableGm,
        GlobalTensor<int32_t>& mqActiveSparseIndicesGm, GlobalTensor<uint32_t>& phyAddrGm, uint32_t kvStride,
        uint32_t mqActiveBlockSize, uint32_t mqActiveMaxBlocksPerBatch, uint32_t sparseBlockCount,
        uint32_t alignedSparseBlockCount, bool isOriKv);

    TPipe* tPipe;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm_;
    bool isSoftmaxLseGmValid_ = false;
    TBuf<> outLseBuf_[2];
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<uint8_t> oriKVGmBytes;
    GlobalTensor<uint8_t> cmpKVGmBytes;
    GlobalTensor<uint8_t> keyGmBytes;
    GlobalTensor<int32_t> oriSparseIndicesGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqOriKvGm;
    GlobalTensor<int32_t> actualSeqCmpKvGm;
    GlobalTensor<int32_t> cmpResidualKvGm;
    GlobalTensor<uint32_t> kvPhyAddrGm;
    GlobalTensor<uint32_t> oriKvPhyAddrGm;
    GlobalTensor<uint32_t> cmpKvPhyAddrGm;

    TBuf<> dequantScaleMode2Buf_;
    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue; // 2份表示可能存在pingpong
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId;
    TEventID mte2ToSOffsetsId;
    TEventID sToMte2OffsetsId;
    TEventID vToMte3Id;
    TEventID mte3ToVLseOutId_;
    TEventID vToMte3LseOutId_;
    TEventID mte2ToVV0Id[2];
    TEventID vToMte3V0Id[2];
    TEventID mte3ToVV0Id[2];
    TEventID vToMte2V0Id[2];
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> stage0InBuf[2];
    TBuf<> stage0OutBuf[2];
    TBuf<> vselrIndexesBuf[2];
    TBuf<> sparseIndiceBuf;
    TBuf<> paBlockIdxOffsetBuf;
    TBuf<> tailBlockOffsetBuf;
    TBuf<> paBlockIdBuf;
    TBuf<> totalOffsetBuf;

    LocalTensor<int32_t> sparseIndiceTensor;
    LocalTensor<int32_t> paBlockIdxOffsetTensor;
    LocalTensor<uint32_t> tailBlockOffsetTensor;
    LocalTensor<int32_t> paBlockIdTensor;
    LocalTensor<int64_t> totalOffsetTensor;
    uint32_t pingPongV0 = 0;
    uint32_t v0BaseSize = 32;
    uint32_t NUM2 = 2;
    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t procSize;
    int64_t procS2Start;
    int64_t procS2End;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyInMode2Rows(LocalTensor<KV_T> kvInUb, int64_t dealRows,
                                                                            int64_t firstToken, bool sparse,
                                                                            const RunInfo& runInfo,
                                                                            const ConstInfo& constInfo)
{
    // A5 mode 2: [all 576-byte token records][all 8-byte E8M0 scales] per page.
    auto zeroRows = kvInUb.template ReinterpretCast<int16_t>();
    Duplicate(zeroRows, static_cast<int16_t>(0), static_cast<uint32_t>(dealRows * dVTemplateTypeInput / NUM_2));
    const auto zeroReady = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(zeroReady);
    WaitFlag<HardEvent::V_MTE2>(zeroReady);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(zeroReady);
    const int64_t pageStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
    for (int64_t row = 0; row < dealRows; ++row) {
        const int64_t token = sparse ? sparseIndicesGm.GetValue(firstToken + row) : firstToken + row;
        if (token < 0) {
            continue;
        }
        const int64_t page = token / blockSize;
        const int64_t inPage = token % blockSize;
        int64_t physicalPage = page;
        // Non-sparse ORI/CMP traversal always uses logical token positions.
        if (!sparse || TOPK_VALUE_MODE == TopkValueMode::TOPK_INDEX_MODE) {
            physicalPage = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + page);
        }
        if (physicalPage < 0) {
            continue;
        }
        const int64_t dataOffset =
            physicalPage * pageStride + inPage * constInfo.n2Size * dCombineBytes + runInfo.n2oIdx * dCombineBytes;
        const int64_t scaleOffset = physicalPage * pageStride +
                                    static_cast<int64_t>(blockSize) * constInfo.n2Size * dCombineBytes +
                                    inPage * scaleBytes;
        DataCopyExtParams dataParams;
        dataParams.blockCount = 1;
        dataParams.blockLen = dCombineBytes;
        dataParams.srcStride = 0;
        dataParams.dstStride = 0;
        DataCopyPadExtParams<KV_T> padding;
        padding.isPad = false;
        padding.leftPadding = 0;
        padding.rightPadding = 0;
        padding.paddingValue = 0;
        DataCopyPad(kvInUb[row * dVTemplateTypeInput], keyGm[dataOffset], dataParams, padding);
        dataParams.blockLen = scaleBytes;
        DataCopyPad(kvInUb[row * dVTemplateTypeInput + dCombineBytes], keyGm[scaleOffset], dataParams, padding);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo& runInfo,
                                                                            ConstInfo& constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realKeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        if constexpr (TOPK_VALUE_MODE == TopkValueMode::TOPK_OFFSET_MODE) {
            realKeyOffset = blkTableIdx * paBlockStride + blkTableOffset * constInfo.dSizeVInput;
        } else {
            realKeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * paBlockStride +
                            blkTableOffset * constInfo.dSizeVInput;
        } // BlockNum, BlockSize, N(1), D
    } else if constexpr (KV_LAYOUT_T == MQSMLA_LAYOUT::TND) {
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
    return realKeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
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
    uint32_t combineBytes = dVTemplateTypeInput;
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
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
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ Q_T* kRopeNzUb, __ubuf__ int8_t* srcUb, __ubuf__ Q_T* dstUb,
                                        __ubuf__ bfloat16_t* scaleSrcUb, __ubuf__ int8_t* kRopeUb,
                                        uint32_t dealRowCount)
{
    uint32_t combineDim = 608; // 32B对齐
    MicroAPI::RegTensor<KV_T> vreg_kv_0;
    MicroAPI::RegTensor<KV_T> vreg_kv_1;
    MicroAPI::RegTensor<Q_T> vreg_scale0_bf16;
    MicroAPI::RegTensor<Q_T> vreg_scale1_bf16;
    MicroAPI::RegTensor<float> vreg_cast_fp_res_0;
    MicroAPI::RegTensor<float> vreg_cast_fp_res_1;
    MicroAPI::RegTensor<Q_T> vreg_mul_res_0;
    MicroAPI::RegTensor<Q_T> vreg_mul_res_1;
    MicroAPI::RegTensor<float> vreg_scale_0;
    MicroAPI::RegTensor<float> vreg_scale_1;
    MicroAPI::RegTensor<Q_T> vreg_cast_res_0;
    MicroAPI::RegTensor<Q_T> vreg_cast_res_1;
    MicroAPI::RegTensor<Q_T> vreg_cast_res_pack_0;
    MicroAPI::RegTensor<Q_T> vreg_cast_res_pack_1;
    MicroAPI::RegTensor<int8_t> vreg_kv_rope;
    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskHalf = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp16MaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    uint32_t blockStride = 33; // +1 to solve bank confict
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;
    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t* srcTemp = srcUb + j * kvNumPerLoop;
        __ubuf__ bfloat16_t* scaleSrcUbTemp = scaleSrcUb + j * scaleNumPerLoop;
        __ubuf__ Q_T* dstUbTmp = dstUb + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vreg_kv_0, srcTemp, tileSize);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vreg_kv_1, srcTemp, combineDim - tileSize);
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(
                (MicroAPI::RegTensor<bfloat16_t>&)vreg_scale0_bf16, scaleSrcUbTemp + i * 304);
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(
                (MicroAPI::RegTensor<bfloat16_t>&)vreg_scale1_bf16, scaleSrcUbTemp + 1 + i * 304);
            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vreg_cast_fp_res_0, vreg_kv_0, fp32MaskAll);
            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vreg_cast_fp_res_1, vreg_kv_1, fp32MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vreg_cast_res_0, vreg_cast_fp_res_0, fp16MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vreg_cast_res_1, vreg_cast_fp_res_1, fp16MaskAll);
            MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vreg_mul_res_0, vreg_cast_res_0, vreg_scale0_bf16,
                                                                 fp16MaskAll);
            MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vreg_mul_res_1, vreg_cast_res_1, vreg_scale1_bf16,
                                                                 fp16MaskAll);
            MicroAPI::DeInterleave(vreg_cast_res_pack_0, vreg_cast_res_pack_1, vreg_mul_res_0, vreg_mul_res_1);
            MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstUbTmp, vreg_cast_res_pack_0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }
    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop; // 偏移已经处理的循环次数
    __ubuf__ int8_t* srcTemp = srcUb + lastLoopOffset * kvNumPerLoop;
    __ubuf__ bfloat16_t* scaleSrcUbTemp = scaleSrcUb + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T* dstUbTmp = dstUb + lastLoopOffset * kvNumPerLoop * blockStride;
    MicroAPI::Duplicate(vreg_cast_res_1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            (MicroAPI::RegTensor<int8_t>&)vreg_kv_rope, kRopeUb, combineDim);
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t>&)vreg_kv_0, srcTemp, combineDim);
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(
            (MicroAPI::RegTensor<bfloat16_t>&)vreg_scale0_bf16, scaleSrcUbTemp + i * 304);
        MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vreg_cast_fp_res_0, vreg_kv_0, fp32MaskAll);
        MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vreg_cast_res_0, vreg_cast_fp_res_0, fp16MaskAll);
        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vreg_mul_res_0, vreg_cast_res_0, vreg_scale0_bf16,
                                                             fp16MaskAll);
        MicroAPI::DeInterleave(vreg_cast_res_pack_0, vreg_cast_res_pack_1, vreg_mul_res_0, vreg_cast_res_1);
        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dstUbTmp, vreg_cast_res_pack_0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            kRopeNzUb, (MicroAPI::RegTensor<Q_T>&)vreg_kv_rope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& kRopeUbNz, LocalTensor<Q_T>& outputUb,
                                          LocalTensor<KV_T>& inputUb, LocalTensor<bfloat16_t>& scaleUb,
                                          LocalTensor<int8_t>& kRopeUb, uint32_t dealRowCount)
{
    const uint32_t nopeDim = 448;
    const uint32_t ropeDim = 64;
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb[ropeDim * sizeof(Q_T)].GetPhyAddr());
    __ubuf__ bfloat16_t* ubScaleAddr =
        (__ubuf__ bfloat16_t*)(scaleUb[ropeDim + nopeDim / sizeof(bfloat16_t)].GetPhyAddr());
    __ubuf__ int8_t* ubKRopeAddr = (__ubuf__ int8_t*)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ Q_T* ubKRopeNzAddr = (__ubuf__ Q_T*)(kRopeUbNz.GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr, dealRowCount);
}

static constexpr MicroAPI::CastTrait castTraitFp8EvenToFp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait castTraitFp8OddToFp32 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait castTraitFp32ToFp16Even = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr MicroAPI::CastTrait castTraitFp32ToFp16Odd = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float* ubDstAddr, __ubuf__ int8_t* ubSrcAddr, uint32_t mq92DealRows)
{
    const uint32_t combineDim = 608;
    const uint32_t scaleNumPerRow = 64;
    MicroAPI::RegTensor<fp8_e8m0_t> vScale0;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res0;
    MicroAPI::RegTensor<float> vScalefp32Res0;
    __ubuf__ int8_t* ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float* ubDstAddrTmp = ubDstAddr;
    MicroAPI::MaskReg bf16TypeMaskAll = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(mq92DealRows); i++) {
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t>&)vScale0, ubScaleSrcAddrTemp, combineDim);

        MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8EvenToFp32>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        MicroAPI::Cast<float, bfloat16_t, castTraitFp8EvenToFp32>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        MicroAPI::StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, vScalefp32Res0,
                                                                             scaleNumPerRow, bf16TypeMaskAll);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void CastScale(LocalTensor<float>& outputUb, LocalTensor<KV_T>& inputUb, uint32_t mq92DealRows)
{
    const uint32_t nopeDim = 448;
    const uint32_t ropeDim = 64;
    __ubuf__ float* ubDstAddr = (__ubuf__ float*)(outputUb.GetPhyAddr());
    __ubuf__ int8_t* ubScaleAddr = (__ubuf__ int8_t*)(inputUb[nopeDim + ropeDim * sizeof(Q_T)].GetPhyAddr());
    CastScaleImpl<Q_T, KV_T>(ubDstAddr, ubScaleAddr, mq92DealRows);
}

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448FloatScale(__ubuf__ Q_T* ubKRopeNzAddr, __ubuf__ int8_t* ubSrcAddr,
                                                  __ubuf__ Q_T* ubDstAddr, __ubuf__ float* ubScaleSrcAddr,
                                                  __ubuf__ int8_t* ubKRopeAddr, uint32_t mq92DealRows)
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
    uint32_t blockStride = 33; // A6 processes 32 rows per V0 chunk, plus one bank-conflict row.
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;

    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ float* ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(mq92DealRows); i++) {
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData0, ubSrcTemp, tileSize);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData1, ubSrcTemp, combineDim - tileSize);

            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<float>&)vScale0, ubScaleSrcAddrTemp, 1);
            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<float>&)vScale1, ubScaleSrcAddrTemp, tileSize - 1);

            MicroAPI::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res0, vKvData0, fp32MaskAll);
            MicroAPI::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res1, vKvData1, fp32MaskAll);

            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes1, vCastFp32Res1, vScale1, fp32MaskAll);

            MicroAPI::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes0, vMulRes0, fp32MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes1, vMulRes1, fp32MaskAll);

            MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

            MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }

    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop;
    __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + lastLoopOffset * kvNumPerLoop;
    __ubuf__ float* ubScaleSrcAddrTemp = ubScaleSrcAddr + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + lastLoopOffset * kvNumPerLoop * blockStride;
    MicroAPI::Duplicate(vCastRes1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(mq92DealRows); i++) {
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            (MicroAPI::RegTensor<int8_t>&)vKvRope, ubKRopeAddr, combineDim);
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t>&)vKvData0, ubSrcTemp, combineDim);
        MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            (MicroAPI::RegTensor<float>&)vScale0, ubScaleSrcAddrTemp, tileSize);
        MicroAPI::Cast<float, KV_T, castTraitFp8EvenToFp32>(vCastFp32Res0, vKvData0, fp32MaskAll);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
        MicroAPI::Cast<Q_T, float, castTraitFp32ToFp16Even>(vCastRes0, vMulRes0, fp32MaskAll);
        MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);

        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubKRopeNzAddr, (MicroAPI::RegTensor<Q_T>&)vKvRope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448FloatScale(LocalTensor<Q_T>& kRopeUbNz, LocalTensor<Q_T>& outputUb,
                                                    LocalTensor<KV_T>& inputUb, LocalTensor<float>& scaleUb,
                                                    LocalTensor<int8_t>& kRopeUb, uint32_t mq92DealRows)
{
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb.GetPhyAddr());
    __ubuf__ int8_t* ubKRopeAddr = (__ubuf__ int8_t*)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ Q_T* ubKRopeNzAddr = (__ubuf__ Q_T*)(kRopeUbNz.GetPhyAddr());
    __ubuf__ float* ubScaleAddr = (__ubuf__ float*)(scaleUb.GetPhyAddr());

    AntiquantVFImplFp8D448FloatScale<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr,
                                                mq92DealRows);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
                                                                      LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                                                      int64_t s2ProcessBaseSize, ConstInfo& constInfo)
{
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::NONCONTIGUOUS) {
        LocalTensor<float> scaleUb = dequantScaleMode2Buf_.template Get<float>();
        CastScale<Q_T, KV_T>(scaleUb, srcTensor, static_cast<uint32_t>(dealRow));
        const uint32_t nopeDim = 448;
        LocalTensor<int8_t> kRopeUb = srcTensor[nopeDim].template ReinterpretCast<int8_t>();
        LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (v0BaseSize + 1)];
        AntiquantVFFp8D448FloatScale<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, scaleUb, kRopeUb,
                                                static_cast<uint32_t>(dealRow));
    } else {
        // srcTensor是rope(448) + nope(64) + scale + pad, dstTensor是nope(448) + rope(64)
        LocalTensor<int8_t> kRopeUb = srcTensor.template ReinterpretCast<int8_t>();
        LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (v0BaseSize + 1)]; // V0单次处理16行数据
        LocalTensor<bfloat16_t> scaleUb = srcTensor.template ReinterpretCast<bfloat16_t>();
        AntiquantVFFp8D448<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, scaleUb, kRopeUb, dealRow);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyOutKvUb2L1(outputl1Pos& outputL1,
                                                                           LocalTensor<Q_T> antiKvTensorAsB16,
                                                                           int64_t dealRow, int64_t s2StartIdx,
                                                                           const RunInfo& runInfo, ConstInfo& constInfo)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = v0BaseSize + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyOutKvUb2Gm(gmPos& v0ResGm,
                                                                           LocalTensor<Q_T> antiKvTensorAsB16,
                                                                           int64_t dealRow, int64_t s2StartIdx,
                                                                           const RunInfo& runInfo, ConstInfo& constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = v0BaseSize + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;
    DataCopy(v0ResGmTensor[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessNotSparseKv(outputl1Pos& outputL1, gmPos& v0ResGm,
                                                                               const RunInfo& runInfo,
                                                                               ConstInfo& constInfo)
{
    if (procSize == 0) {
        return;
    }
    int64_t s2ProcessBaseSize = v0BaseSize;
    int64_t s2ProcessSize = s2ProcessBaseSize;
    int64_t s2V0LoopTimes = (procSize + s2ProcessBaseSize - 1) / s2ProcessBaseSize;
    int64_t s2Tail = procSize - (s2V0LoopTimes - 1) * s2ProcessBaseSize;
    for (uint32_t i = 0; i < s2V0LoopTimes; i++) {
        int64_t s2StartIdx = procS2Start + i * s2ProcessBaseSize;
        if (i == s2V0LoopTimes - 1) {
            s2ProcessSize = s2Tail;
        }
        int64_t dealRows = s2ProcessSize;

        if (dealRows == 0) {
            continue;
        }
        WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
        CopyInKvNotSparse(kvInUb, dealRows, s2StartIdx, runInfo, constInfo);
        SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        LocalTensor<Q_T> kvDequantOutUb = stage0OutBuf[pingPongV0].Get<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, dealRows, s2ProcessBaseSize, constInfo);

        SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);

        SetFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);

        if constexpr (IS_SPLIT_G) {
            CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRows, s2StartIdx, runInfo, constInfo);
        } else {
            CopyOutKvUb2L1(outputL1, kvDequantOutUb, dealRows, s2StartIdx, runInfo, constInfo);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb,
                                                                              int64_t dealRow, int64_t s2StartOffset,
                                                                              const RunInfo& runInfo,
                                                                              ConstInfo& constInfo)
{
    int64_t s2LoopCount = (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) ?
                              (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) :
                              runInfo.s2LoopCount;
    int64_t s2Idx = s2StartOffset + s2LoopCount * constInfo.s2BaseSize + runInfo.s2StartIdx;
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::NONCONTIGUOUS) {
        CopyInMode2Rows(kvMergUb, dealRow, s2Idx, false, runInfo, constInfo);
        return;
    }
    uint32_t combineBytes = constInfo.dSizeVInput;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    DataCopyExtParams intriParams;
    intriParams.blockCount = dealRow;
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
        while (copyFinishElmenCnt < dealRow) {
            uint64_t blockIdOffset = curSequence / blockSize;
            uint64_t remainElmenCnt = curSequence % blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
            uint32_t copyElmenCnt = blockSize - remainElmenCnt;
            if (copyElmenCnt + copyFinishElmenCnt > dealRow) {
                copyElmenCnt = dealRow - copyFinishElmenCnt;
            }
            uint64_t srcOffset = idInBlockTable * paBlockStride + remainElmenCnt * constInfo.n2Size * combineBytes +
                                 (uint64_t)(runInfo.n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
            intriParams.blockCount = copyElmenCnt;                          // base s2 size
            DataCopyPad(kvMergUb[dstOffset * combineDimAlign], keyGm[srcOffset], intriParams, padParams);
            dstOffset += copyElmenCnt;
            copyFinishElmenCnt += copyElmenCnt;
            curSequence += copyElmenCnt;
        }
    } else {
        uint64_t srcOffset = 0;
        if constexpr (KV_LAYOUT_T == MQSMLA_LAYOUT::TND) {
            uint64_t kvS2Idx = s2Idx + (runInfo.isCmp ? cuSeqlensCmpKvGm.GetValue(runInfo.boIdx) :
                                                        cuSeqlensOriKvGm.GetValue(runInfo.boIdx));
            srcOffset = (kvS2Idx * constInfo.n2Size + runInfo.n2oIdx) * combineDim;
        } else {
            const int64_t kvS2Size = runInfo.isCmp ? constInfo.cmpS2Size : constInfo.s2Size;
            srcOffset = ((runInfo.boIdx * kvS2Size + s2Idx) * constInfo.n2Size + runInfo.n2oIdx) * combineDim;
        }
        DataCopyPad(kvMergUb, keyGm[srcOffset], intriParams, padParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CalProSize(const RunInfo& runInfo, ConstInfo& constInfo)
{
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        sparseIndicesGm = runInfo.isCmp ? cmpSparseIndicesGm : oriSparseIndicesGm;
    } else if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        sparseIndicesGm = oriSparseIndicesGm;
    } else {
        sparseIndicesGm = cmpSparseIndicesGm;
    }
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = GetBlockIdx();
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        if (aicIdx % 2U == 0) {
            procSize = v0S2SizeFirstCore;
            procS2Start = 0;
        } else {
            procSize = v0S2SizeSecondCore;
            procS2Start = v0S2SizeFirstCore;
        }
        procS2End = procS2Start + procSize;
    } else {
        procS2Start = 0;
        procS2End = runInfo.s2RealSize;
        procSize = procS2End - procS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec0(outputl1Pos& outputL1, gmPos& v0ResGm,
                                                                        const RunInfo& runInfo, ConstInfo& constInfo)
{
    bool isCmp = runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx;
    if (isCmp) {
        keyGm = cmpKVGm;
        keyGmBytes = cmpKVGmBytes;
        kvPhyAddrGm = cmpKvPhyAddrGm;
        if constexpr (isPa) {
            blockTableGm = cmpBlockTableGm;
            blockSize = constInfo.cmpBlockSize;
            maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        }
    } else {
        keyGm = oriKVGm;
        keyGmBytes = oriKVGmBytes;
        kvPhyAddrGm = oriKvPhyAddrGm;
        if constexpr (isPa) {
            blockTableGm = oriBlockTableGm;
            blockSize = constInfo.oriBlockSize;
            maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
        }
    }

    CalProSize(runInfo, constInfo);
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (isCmp) {
            ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo);
        } else {
            ProcessNotSparseKv(outputL1, v0ResGm, runInfo, constInfo);
        }
    } else if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo);
    } else {
        ProcessNotSparseKv(outputL1, v0ResGm, runInfo, constInfo);
    }

    if constexpr (IS_SPLIT_G) {
        v0ResGm.SetCrossCore();
    } else {
        outputL1.SetCrossCore(); // 核间同步
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessSparseKv(outputl1Pos& outputL1, gmPos& v0ResGm,
                                                                            const RunInfo& runInfo,
                                                                            ConstInfo& constInfo)
{
    if (procSize == 0) {
        return;
    }
    int64_t s2ProcessBaseSize = v0BaseSize;
    int64_t s2RealSize = procSize; // procSize = 64
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == MQSMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        if (runInfo.isCmp) {
            if constexpr (!IS_VEC_S2PHYADDR) {
                topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.cmpSparseBlockCount; // T, N2(1), K
            } else {
                topkBS1Idx +=
                    (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.alignedCmpSparseBlockCount; // T, N2(1), K
            }
        } else {
            if constexpr (!IS_VEC_S2PHYADDR) {
                topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.oriSparseBlockCount; // T, N2(1), K
            } else {
                topkBS1Idx +=
                    (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.alignedOriSparseBlockCount; // T, N2(1), K
            }
        }
    } else {
        if (runInfo.isCmp) {
            if constexpr (!IS_VEC_S2PHYADDR) {
                topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.cmpSparseBlockCount +
                              runInfo.s1oIdx * constInfo.cmpSparseBlockCount; // B, S1, N2(1), K
            } else {
                topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.alignedCmpSparseBlockCount +
                              runInfo.s1oIdx * constInfo.alignedCmpSparseBlockCount; // B, S1, N2(1), K
            }
        } else {
            if constexpr (!IS_VEC_S2PHYADDR) {
                topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.oriSparseBlockCount +
                              runInfo.s1oIdx * constInfo.oriSparseBlockCount; // B, S1, N2(1), K
            } else {
                topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.alignedOriSparseBlockCount +
                              runInfo.s1oIdx * constInfo.alignedOriSparseBlockCount; // B, S1, N2(1), K
            }
        }
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount;
    if (runInfo.isCmp) {
        cmpS2LoopCnt = runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx;
    }
    int64_t topkKIdx = procS2Start + cmpS2LoopCnt * constInfo.s2BaseSize;
    LocalTensor<int64_t> totalOffset64Tensor = totalOffsetTensor.ReinterpretCast<int64_t>();
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::CONTIGUOUS) {
        GlobalTensor<int64_t> kvPhyAddrGm64 = kvPhyAddrGm.template ReinterpretCast<int64_t>();
        if constexpr (TOPK_VALUE_MODE == TopkValueMode::TOPK_OFFSET_MODE || (!IS_VEC_S2PHYADDR)) {
            SetFlag<HardEvent::MTE2_S>(mte2ToSOffsetsId);
            WaitFlag<HardEvent::MTE2_S>(mte2ToSOffsetsId);
            const int64_t sparseBase = topkBS1Idx + runInfo.s2StartIdx + topkKIdx;
            for (int64_t row = 0; row < s2RealSize; ++row) {
                const int64_t token = sparseIndicesGm.GetValue(sparseBase + row);
                const int64_t offset = GetkeyOffset(token, runInfo, constInfo);
                totalOffset64Tensor.SetValue(row, offset < 0 ? -1 : offset * static_cast<int64_t>(sizeof(KV_T)));
            }
            SetFlag<HardEvent::S_MTE2>(sToMte2OffsetsId);
            WaitFlag<HardEvent::S_MTE2>(sToMte2OffsetsId);
        } else {
            static constexpr int64_t numPerBlock = 32;
            // static constexpr int64_t numPerBlock = 8;
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = 1U; // 每次处理1行
            dataCopyParams.blockLen = (s2RealSize * sizeof(int64_t) + numPerBlock - 1) / numPerBlock;
            dataCopyParams.srcGap = 0U;
            dataCopyParams.dstGap = 0U;
            DataCopy(totalOffset64Tensor, kvPhyAddrGm64[topkBS1Idx + runInfo.s2StartIdx + topkKIdx], dataCopyParams);
            pipe_barrier(PIPE_MTE2);
        }
    }
    int64_t s2ProcessBaseLoop = CeilDiv(s2RealSize, s2ProcessBaseSize);
    int64_t s2Start = procS2Start;
    for (uint16_t i = 0; i < s2ProcessBaseLoop; ++i) {
        int64_t remain = s2RealSize - i * s2ProcessBaseSize;
        int64_t dealRows;
        if (s2RealSize < s2ProcessBaseSize) {
            dealRows = s2RealSize;
        } else {
            dealRows = (s2ProcessBaseSize < remain) ? s2ProcessBaseSize : remain;
        }
        // 1、gather kv in gm->ub
        WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        LocalTensor<KV_T> kvInUb = stage0InBuf[pingPongV0].Get<KV_T>();
        LocalTensor<uint8_t> kvInUbBytes = stage0InBuf[pingPongV0].Get<uint8_t>();
        LocalTensor<Q_T> kvDequantOutUb = stage0OutBuf[pingPongV0].Get<Q_T>();
        if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::NONCONTIGUOUS) {
            const int64_t sparseBase = topkBS1Idx + runInfo.s2StartIdx + topkKIdx + s2ProcessBaseSize * i;
            CopyInMode2Rows(kvInUb, dealRows, sparseBase, true, runInfo, constInfo);
        } else {
            GatherParams gatherParams2{
                static_cast<uint16_t>(dealRows),
                static_cast<uint32_t>(constInfo.dSizeVInput),
                static_cast<uint32_t>(constInfo.dSizeVInput),
            };
            GatherPadParams<uint8_t> gatherPadparams2{GatherPaddingMode::CONSTANT_PADDING_MODE,
                                                      static_cast<uint8_t>(2)};
            GatherDataCopy(kvInUbBytes, keyGmBytes, totalOffset64Tensor[s2ProcessBaseSize * i], gatherParams2,
                           gatherPadparams2);
        }

        SetFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVV0Id[pingPongV0]);
        // 2、 dequant by vf
        WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        DequantKv(kvDequantOutUb, kvInUb, dealRows, s2ProcessBaseSize, constInfo);
        SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[pingPongV0]);
        // 3、 copy kv out, ub->l1
        SetFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3V0Id[pingPongV0]);
        if constexpr (IS_SPLIT_G) {
            CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRows, s2Start, runInfo, constInfo);
        } else {
            CopyOutKvUb2L1(outputL1, kvDequantOutUb, dealRows, s2Start, runInfo, constInfo);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVV0Id[pingPongV0]);
        s2Start += dealRows;
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec1(outputl1Pos& outputBuf, mmResPos& bmm1ResBuf,
                                                                        RunInfo& runInfo, ConstInfo& constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    auto stage1CastTensor = this->stage1OutQue.template AllocTensor<Q_T>();
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    if (runInfo.s2LoopCount == 0 && !isSinks) {
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
        if (runInfo.s2LoopCount == 0 && isSinks) {
            int64_t sinksOffset = 0;
            if constexpr (!IS_SPLIT_G) {
                sinksOffset = GetBlockIdx() % NUM2 == 0 ? 0 : constInfo.gSize - runInfo.halfMRealSize;
            } else {
                sinksOffset = runInfo.goIdx;
            }
            LocalTensor<float> sinksUb = this->sinksBuf.template Get<float>();
            InitSoftmaxFromSinks<T>(sumUb, maxUb, sinksUb, sinksOffset, R0, runInfo.halfMRealSize);
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
    this->stage1OutQue.template EnQue(stage1CastTensor);
    this->stage1OutQue.template DeQue<Q_T>();
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (runInfo.mRealSize - runInfo.halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)(runInfo.halfMRealSize), (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
                  (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }
    this->stage1OutQue.template FreeTensor(stage1CastTensor);
    outputBuf.SetCrossCore();
    if (runInfo.s2LoopCount != 0 || (runInfo.s2LoopCount == 0 && isSinks)) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    const bool copyOutLse = constInfo.isSoftmaxLseEnable && this->isSoftmaxLseGmValid_ && runInfo.halfMRealSize > 0 &&
                            runInfo.s2LoopCount == runInfo.s2LoopLimit;
    if (copyOutLse) {
        LocalTensor<float> outLse = outLseBuf_[runInfo.multiCoreIdxMod2].template Get<float>();
        DataCopyExtParams lseParams;
        lseParams.blockCount = 1;
        lseParams.blockLen = static_cast<uint32_t>(runInfo.halfMRealSize * sizeof(float));
        lseParams.srcStride = 0;
        lseParams.dstStride = 0;
        WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId_);
        ComputeLse<float>(outLse, sumUb, maxUb, static_cast<uint32_t>(runInfo.halfMRealSize));
        SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId_);
        DataCopyPad(this->softmaxLseGm_[runInfo.softmaxLseOffset], outLse, lseParams);
        SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId_);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::ProcessVec2(mmResPos& bmm2ResBuf, mmResPos& bmm2ResBuf1,
                                                                        RunInfo& runInfo, ConstInfo& constInfo)
{
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.WaitCrossCore();
        bmm2ResBuf.SetCrossCore();
        bmm2ResBuf1.WaitCrossCore();
        bmm2ResBuf1.SetCrossCore();
        bmm2ResBuf.WaitCrossCore();
        bmm2ResBuf.SetCrossCore();
        bmm2ResBuf1.WaitCrossCore();
        bmm2ResBuf1.SetCrossCore();
        return;
    }

    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    LocalTensor<T> mmRes1 = bmm2ResBuf1.template GetTensor<T>();

    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopyParams intriParams;
        intriParams.blockCount = runInfo.vec2MRealSize; // 连续数据块个数为1
        intriParams.blockLen = constInfo.dSizeV / NUM_4 * sizeof(float) /
                               BUFFER_SIZE_BYTE_32B; // 连续数据块长度，单位为DataBlock，此处长度为512个half元素
        intriParams.srcGap = 0;                      // 源操作数做连续搬运
        intriParams.dstGap = dstGap * sizeof(float) / BUFFER_SIZE_BYTE_32B; // 目的操作数连续排布
        bmm2ResBuf.WaitCrossCore();
        DataCopy(vec2ResUb[0], mmRes, intriParams); // todo
        bmm2ResBuf.SetCrossCore();
        bmm2ResBuf1.WaitCrossCore();
        DataCopy(vec2ResUb[NUM_128], mmRes1, intriParams); // todo
        bmm2ResBuf1.SetCrossCore();
        bmm2ResBuf.WaitCrossCore();
        DataCopy(vec2ResUb[NUM_256], mmRes, intriParams); // todo
        bmm2ResBuf.SetCrossCore();
        bmm2ResBuf1.WaitCrossCore();
        DataCopy(vec2ResUb[NUM_384], mmRes1, intriParams); // todo
        bmm2ResBuf1.SetCrossCore();
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            bmm2ResBuf.WaitCrossCore();
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb,
                                                               runInfo.vec2MRealSize, 0);
            bmm2ResBuf.SetCrossCore();
            bmm2ResBuf1.WaitCrossCore();
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes1, vec2ResUb, expUb,
                                                               runInfo.vec2MRealSize, NUM_2);
            bmm2ResBuf1.SetCrossCore();
            bmm2ResBuf.WaitCrossCore();
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb,
                                                               runInfo.vec2MRealSize, NUM_4);
            bmm2ResBuf.SetCrossCore();
            bmm2ResBuf1.WaitCrossCore();
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes1, vec2ResUb, expUb,
                                                               runInfo.vec2MRealSize, NUM_6);
            bmm2ResBuf1.SetCrossCore();
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            bmm2ResBuf.WaitCrossCore();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb, sumUb,
                                                                   runInfo.vec2MRealSize, 0);
            bmm2ResBuf.SetCrossCore();
            bmm2ResBuf1.WaitCrossCore();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes1, vec2ResUb, expUb, sumUb,
                                                                   runInfo.vec2MRealSize, NUM_2);
            bmm2ResBuf1.SetCrossCore();
            bmm2ResBuf.WaitCrossCore();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb, sumUb,
                                                                   runInfo.vec2MRealSize, NUM_4);
            bmm2ResBuf.SetCrossCore();
            bmm2ResBuf1.WaitCrossCore();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes1, vec2ResUb, expUb, sumUb,
                                                                   runInfo.vec2MRealSize, NUM_6);
            bmm2ResBuf1.SetCrossCore();
        }
    }
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize);
        }
        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo& runInfo, ConstInfo& constInfo,
                                                                            LocalTensor<VEC2_RES_T>& vec2ResUb,
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
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo& runInfo, ConstInfo& constInfo,
                                                                                LocalTensor<VEC2_RES_T>& vec2ResUb,
                                                                                int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo& constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if (LAYOUT_T == MQSMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if (LAYOUT_T == MQSMLA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t* attentionOut,
                                                                        __gm__ uint8_t* softmaxLse,
                                                                        ConstInfo& constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        isSoftmaxLseGmValid_ = constInfo.isSoftmaxLseEnable && softmaxLse != nullptr;
        if (isSoftmaxLseGmValid_) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitOutput(ConstInfo& constInfo, RunParamStr& runParam,
                                                                       int32_t sIdx, int32_t gS1Cur)
{
    if ASCEND_IS_AIV {
        int64_t seqOffset = 0;
        if constexpr (LAYOUT_T == MQSMLA_LAYOUT ::TND) {
            seqOffset = cuSeqlensQGm.GetValue(sIdx);
        } else {
            seqOffset = sIdx * constInfo.s1Size;
        }

        int64_t outputGStart = 0;
        int64_t outputGSize = constInfo.gSize;
        if constexpr (IS_SPLIT_G) {
            const int64_t halfG = (constInfo.gSize + 1) / 2;
            const bool firstHalf = constInfo.aivIdx % NUM2 == 0;
            outputGStart = firstHalf ? 0 : halfG;
            outputGSize = firstHalf ? halfG : constInfo.gSize - halfG;
        }
        const int64_t attentionOutSeqOffset = (seqOffset + gS1Cur) * constInfo.n2GDv + outputGStart * constInfo.dSizeV;
        matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[attentionOutSeqOffset], outputGSize * constInfo.dSizeV, 0);
        if (constInfo.isSoftmaxLseEnable && isSoftmaxLseGmValid_) {
            const int64_t lseOffset = (seqOffset + gS1Cur) * constInfo.gSize + outputGStart;
            matmul::InitOutput<float>(softmaxLseGm_[lseOffset], outputGSize, 0);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetActualS1Size(uint32_t bIdx,
                                                                               __gm__ int32_t* actualSeqQlenAddr,
                                                                               __gm__ int32_t* cuSeqlensQAddr,
                                                                               const ConstInfo& constInfo)
{
    int32_t actualS1Size = 0;
    if constexpr (LAYOUT_T == MQSMLA_LAYOUT::TND) {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? (cuSeqlensQAddr[bIdx + 1] - cuSeqlensQAddr[bIdx]) :
                                                        actualSeqQlenAddr[bIdx];
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size : actualSeqQlenAddr[bIdx];
    }
    return actualS1Size;
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
__aicore__ inline Mq92PhyAddrValidInfo MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CalcPhyAddrValidInfo(
    bool isOriKv, int32_t actualS1Size, int32_t actualOriS2Size, int64_t restoredSize, ConstInfo& constInfo)
{
    // per-batch执行一次,  per-s1循环内不再判断maskmode
    Mq92PhyAddrValidInfo mq92ValidWindow;
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        mq92ValidWindow.oriS2Act = actualOriS2Size;
        if (isOriKv) {
            if (constInfo.oriMaskMode == 0U) {
                mq92ValidWindow.oriTopkMode = true;
            } else if (constInfo.oriMaskMode == 3U) {
                mq92ValidWindow.oriRightBias = 0;
            } else {
                mq92ValidWindow.oriLeftBias =
                    (constInfo.oriWinLeft == -1) ? Mq92PhyAddrValidInfo::BIAS_UNBOUND : constInfo.oriWinLeft + 1;
                mq92ValidWindow.oriRightBias =
                    (constInfo.oriWinRight == -1) ? Mq92PhyAddrValidInfo::BIAS_UNBOUND : constInfo.oriWinRight;
            }
        }
    }
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            mq92ValidWindow.cmpTopkMode = (constInfo.cmpMaskMode == 0U);
            mq92ValidWindow.cmpBase = restoredSize - actualS1Size + 1;
        }
    }
    return mq92ValidWindow;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CalcCurValidS2ForPhyAddr(
    uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv, GlobalTensor<int32_t>& cuSeqlensQGm,
    GlobalTensor<int32_t>& topkLengthGm, ConstInfo& constInfo, int32_t sparseBlockCount,
    const Mq92PhyAddrValidInfo& mq92ValidWindow)
{
    bool topkMode = false;
    bool hasTopk = false;

    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            topkMode = mq92ValidWindow.oriTopkMode;
            hasTopk = constInfo.hasOriTopkLength;
        }
    }
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!isOriKv) {
            topkMode = mq92ValidWindow.cmpTopkMode;
            hasTopk = constInfo.hasCmpTopkLength;
        }
    }
    if (topkMode) {
        uint64_t topkIdx = (LAYOUT_T == MQSMLA_LAYOUT::TND) ? (cuSeqlensQGm.GetValue(bIdx) + s1Idx) :
                                                              (bIdx * constInfo.s1Size + s1Idx);
        int32_t topkLen = hasTopk ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
        return Min(topkLen, sparseBlockCount);
    }

    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (isOriKv) {
            int64_t thr = mq92ValidWindow.oriS2Act - actualS1Size + 1 + s1Idx;
            int64_t leftBound = Max(thr - mq92ValidWindow.oriLeftBias, 0);
            int64_t rightBound =
                Min(thr + mq92ValidWindow.oriRightBias, static_cast<int64_t>(mq92ValidWindow.oriS2Act));
            return Min(static_cast<int32_t>(Max(0, rightBound - leftBound)), sparseBlockCount);
        }
    }
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t numerator = Max(mq92ValidWindow.cmpBase + s1Idx, 0);
        return Min(sparseBlockCount, static_cast<int32_t>(numerator / static_cast<int32_t>(constInfo.cmpRatio)));
    }
    return 0;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetKVPhyAddrForKvType(
    uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t nextGs1Idx, bool hasActualSeqQlen,
    bool hasCuSeqlensQ, bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t>& actualSeqQlenGm,
    GlobalTensor<int32_t>& cuSeqlensQGm, GlobalTensor<int32_t>& actualSeqKvlenGm, GlobalTensor<int32_t>& cuSeqlensKvGm,
    GlobalTensor<int32_t>& topkLengthGm, GlobalTensor<int32_t>& cmpResidualKvGm, ConstInfo& constInfo,
    GlobalTensor<int32_t>& mqActiveBlockTableGm, GlobalTensor<int32_t>& mqActiveSparseIndicesGm,
    GlobalTensor<uint32_t>& phyAddrGm, uint32_t kvStride, uint32_t mqActiveBlockSize,
    uint32_t mqActiveMaxBlocksPerBatch, uint32_t sparseBlockCount, uint32_t alignedSparseBlockCount, bool isOriKv)
{
    constexpr uint16_t s2NumPerLoop = 128;
    constexpr uint32_t vecCoreNum = IS_SPLIT_G ? 2 : 1;
    uint32_t vecCoreIdx = IS_SPLIT_G ? constInfo.aivIdx % 2 : constInfo.aivIdx % 1;
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
        int32_t actualS1Size =
            GetSeqLenForPhyAddr(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        int32_t s1End = (lastBN && nextGs1Idx != 0) ? nextGs1Idx : actualS1Size;
        int64_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && constInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                           cuSeqlensKvGm, constInfo.cmpS2Size);
                int64_t residual = (constInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
                restoredSize = static_cast<int64_t>(actualKvSize) * static_cast<int64_t>(constInfo.cmpRatio) + residual;
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && constInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                      cuSeqlensKvGm, constInfo.s2Size);
            }
        }

        Mq92PhyAddrValidInfo mq92ValidWindow =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, constInfo);
        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t validS2 = CalcCurValidS2ForPhyAddr(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                       constInfo, sparseBlockCount, mq92ValidWindow);
            totalValidS1 += validS2 > 0;
        }
        tmpGS1Start = 0;
    }
    int64_t rowsPerCore = totalValidS1 / vecCoreNum;
    int64_t tailRows = totalValidS1 % vecCoreNum;
    int64_t rowStart = rowsPerCore * vecCoreIdx + Min(static_cast<int64_t>(vecCoreIdx), tailRows);
    int64_t rowCount = rowsPerCore + (vecCoreIdx < static_cast<uint32_t>(tailRows) ? 1 : 0);
    if (rowCount == 0) {
        SyncAll();
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
        int32_t actualS1Size =
            GetSeqLenForPhyAddr(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        int64_t bS1Idx = (LAYOUT_T == MQSMLA_LAYOUT::TND) ?
                             (hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : constInfo.s1Size * bIdx) :
                             constInfo.s1Size * bIdx;
        int32_t s1End = (lastBN && nextGs1Idx != 0) ? nextGs1Idx : actualS1Size;
        int64_t restoredSize = 0;
        if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (!isOriKv && constInfo.cmpMaskMode != 0) {
                int32_t actualKvSize = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                           cuSeqlensKvGm, constInfo.cmpS2Size);
                int64_t residual = (constInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
                restoredSize = static_cast<int64_t>(actualKvSize) * static_cast<int64_t>(constInfo.cmpRatio) + residual;
            }
        }
        int32_t actualOriS2Size = 0;
        if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            if (isOriKv && constInfo.oriMaskMode != 0) {
                actualOriS2Size = GetSeqLenForPhyAddr(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                      cuSeqlensKvGm, constInfo.s2Size);
            }
        }

        Mq92PhyAddrValidInfo mq92ValidWindow =
            CalcPhyAddrValidInfo(isOriKv, actualS1Size, actualOriS2Size, restoredSize, constInfo);
        WaitFlag<HardEvent::V_MTE2>(INNERCORE_PHYADDR_BLKTABLE_FREE);
        AttentionCommon::CopyPaTableToUb(blkTableUb, bIdx, mqActiveBlockTableGm, mqActiveMaxBlocksPerBatch);
        SetFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
        WaitFlag<HardEvent::MTE2_V>(INNERCORE_PHYADDR_BLKTABLE_READY);
        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t validS2 = CalcCurValidS2ForPhyAddr(bIdx, s1Idx, actualS1Size, isOriKv, cuSeqlensQGm, topkLengthGm,
                                                       constInfo, sparseBlockCount, mq92ValidWindow);
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
            AttentionCommon::GetKVPhyAddrVFPa<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail,
                                                        mqActiveBlockSize, shiftRightNum, constInfo.sparseBlockSize,
                                                        constInfo.dSizeVInput, kvStride);
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
    SyncAll();
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
    ConstInfo& constInfo)
{
    uint64_t v0ResSize = static_cast<uint64_t>(constInfo.s2BaseSize) * constInfo.dSize * sizeof(Q_T);
    uint64_t v0RegionSize = v0ResSize * 3 * (IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum());
    uint64_t totalBS1 =
        (LAYOUT_T == MQSMLA_LAYOUT::TND) ? constInfo.s1Size : static_cast<uint64_t>(constInfo.bSize) * constInfo.s1Size;
    uint64_t oriPhyAddrSize = 0;
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (hasLoad == 0) {
            SyncAll();
        } else {
            oriPhyAddrSize = totalBS1 * constInfo.alignedOriSparseBlockCount * sizeof(int64_t);
            oriKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t*)(workspace + v0RegionSize));
            kvPhyAddrGm = oriKvPhyAddrGm;
            GetKVPhyAddrForKvType(bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                                  hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqQlenGm, cuSeqlensQGm,
                                  actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm, cmpResidualKvGm, constInfo,
                                  oriBlockTableGm, oriSparseIndicesGm, oriKvPhyAddrGm, constInfo.oriKvStride,
                                  constInfo.oriBlockSize, constInfo.oriMaxBlockNumPerBatch,
                                  constInfo.oriSparseBlockCount, constInfo.alignedOriSparseBlockCount, true);
        }
    }

    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (hasLoad == 0) {
            SyncAll();
        } else {
            cmpKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t*)(workspace + v0RegionSize + oriPhyAddrSize));
            kvPhyAddrGm = cmpKvPhyAddrGm;
            GetKVPhyAddrForKvType(bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                                  hasActualSeqCmpKvlen, hasCuSeqlensCmpKv, actualSeqQlenGm, cuSeqlensQGm,
                                  actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm, cmpResidualKvGm, constInfo,
                                  cmpBlockTableGm, cmpSparseIndicesGm, cmpKvPhyAddrGm, constInfo.cmpKvStride,
                                  constInfo.cmpBlockSize, constInfo.cmpMaxBlockNumPerBatch,
                                  constInfo.cmpSparseBlockCount, constInfo.alignedCmpSparseBlockCount, false);
        }
    }

    if (hasLoad == 0) {
        tPipe->Reset();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetActualS2Size(uint32_t bIdx,
                                                                               __gm__ int32_t* actualSeqKvlenAddr,
                                                                               const ConstInfo& constInfo)
{
    int32_t actualS2Size = 0;
    if (constInfo.isActualLenDimsKVNull) {
        actualS2Size = constInfo.s2Size;
    } else {
        actualS2Size = actualSeqKvlenAddr[bIdx];
    }
    return actualS2Size;
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFImpl(__ubuf__ uint32_t* kvPhyAddrUb, __ubuf__ int32_t* sparseIdxUb,
                                    __ubuf__ int32_t* blkTableUb, const uint16_t s2Loop, uint32_t s2Tail,
                                    const uint32_t blockSize, const int16_t shiftRightNum,
                                    const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    static const uint16_t s2NumPerLoop = 128;
    static const uint16_t s2NumPerReg = 64;
    static const uint16_t outOffsetPerLoop = 256;
    static const uint16_t outOffsetPerReg = 128;
    static const uint32_t inValidValue = 0xFFFFFFFF;
    MicroAPI::MaskReg preg_all_b32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg add_carry_l_1;
    MicroAPI::MaskReg add_carry_h_1;
    MicroAPI::MaskReg add_carry_l_2;
    MicroAPI::MaskReg add_carry_h_2;
    MicroAPI::MaskReg preg_tail_neg_1_b32;
    MicroAPI::MaskReg preg_tail_neg_2_b32;

    MicroAPI::RegTensor<uint32_t> vreg_kvStride;
    MicroAPI::RegTensor<uint32_t> vreg_sparse_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_sparse_idx_2;
    MicroAPI::RegTensor<uint32_t> vreg_blockSize;
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

    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_H_1;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_H_1;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_mul_overflow_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_H_1;

    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_H_2;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_H_2;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_mul_overflow_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_H_2;

    MicroAPI::RegTensor<uint32_t> vreg_zero;
    MicroAPI::Duplicate(vreg_zero, 0);
    MicroAPI::Duplicate(vreg_kvStride, kvStride);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<int32_t>&)vreg_sparse_idx_1, sparseIdxUb + i * s2NumPerLoop);
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<int32_t>&)vreg_sparse_idx_2, sparseIdxUb + s2NumPerReg + i * s2NumPerLoop);
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
            MicroAPI::Mull(vreg_blk_id_mul_stride_L_1, vreg_mul_overflow_L_1, vreg_phy_blk_idx_1, vreg_kvStride,
                           preg_all_b32);
            MicroAPI::Mull(vreg_blk_id_mul_stride_L_2, vreg_mul_overflow_L_2, vreg_phy_blk_idx_2, vreg_kvStride,
                           preg_all_b32);

            // 分高低32位计算int64物理地址 -- 加 offset
            MicroAPI::Add(add_carry_l_1, vreg_total_offset_L_1, vreg_blk_id_mul_stride_L_1, vreg_phy_offset_1,
                          preg_all_b32);
            MicroAPI::Add(add_carry_l_2, vreg_total_offset_L_2, vreg_blk_id_mul_stride_L_2, vreg_phy_offset_2,
                          preg_all_b32);

            MicroAPI::AddC(add_carry_h_1, vreg_total_offset_H_1, vreg_mul_overflow_L_1, vreg_zero, add_carry_l_1,
                           preg_all_b32);
            MicroAPI::AddC(add_carry_h_2, vreg_total_offset_H_2, vreg_mul_overflow_L_2, vreg_zero, add_carry_l_2,
                           preg_all_b32);

            // 搬出 由于拆分为了int32类型，元素个数翻倍
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * outOffsetPerLoop, vreg_total_offset_L_1, vreg_total_offset_H_1, preg_all_b32);
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + outOffsetPerReg + i * outOffsetPerLoop, vreg_total_offset_L_2, vreg_total_offset_H_2,
                preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        MicroAPI::MaskReg preg_tail_1_b32 = MicroAPI::UpdateMask<int32_t>(s2Tail);
        MicroAPI::MaskReg preg_tail_2_b32 = MicroAPI::UpdateMask<int32_t>(s2Tail);
        MicroAPI::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        MicroAPI::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)vreg_sparse_idx_1,
                                                                    sparseIdxUb + i * s2NumPerLoop);
        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t>&)vreg_sparse_idx_2,
                                                                    sparseIdxUb + s2NumPerReg + i * s2NumPerLoop);
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
        MicroAPI::Mull(vreg_blk_id_mul_stride_L_1, vreg_mul_overflow_L_1, vreg_phy_blk_idx_1, vreg_kvStride,
                       preg_tail_1_b32);
        MicroAPI::Mull(vreg_blk_id_mul_stride_L_2, vreg_mul_overflow_L_2, vreg_phy_blk_idx_2, vreg_kvStride,
                       preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 加 offset
        MicroAPI::Add(add_carry_l_1, vreg_total_offset_L_1, vreg_blk_id_mul_stride_L_1, vreg_phy_offset_1,
                      preg_tail_1_b32);
        MicroAPI::Add(add_carry_l_2, vreg_total_offset_L_2, vreg_blk_id_mul_stride_L_2, vreg_phy_offset_2,
                      preg_tail_2_b32);

        MicroAPI::AddC(add_carry_h_1, vreg_total_offset_H_1, vreg_mul_overflow_L_1, vreg_zero, add_carry_l_1,
                       preg_tail_1_b32);
        MicroAPI::AddC(add_carry_h_2, vreg_total_offset_H_2, vreg_mul_overflow_L_2, vreg_zero, add_carry_l_2,
                       preg_tail_2_b32);

        // 无效值填充-1(0xFFFFFFFF)
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_1, inValidValue,
                                                                        preg_tail_neg_1_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_1, inValidValue,
                                                                        preg_tail_neg_1_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_2, inValidValue,
                                                                        preg_tail_neg_2_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_2, inValidValue,
                                                                        preg_tail_neg_2_b32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * outOffsetPerLoop, vreg_total_offset_L_1, vreg_total_offset_H_1, preg_all_b32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + outOffsetPerReg + i * outOffsetPerLoop, vreg_total_offset_L_2, vreg_total_offset_H_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVF(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                      LocalTensor<int32_t> blkTableTensor, const uint16_t s2Loop, const uint32_t s2Tail,
                                      const uint32_t blockSize, const int16_t shiftRightNum,
                                      const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    __ubuf__ uint32_t* kvPhyAddrUb = (__ubuf__ uint32_t*)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t* sparseIdxUb = (__ubuf__ int32_t*)(sparseIdxTensor.GetPhyAddr());
    __ubuf__ int32_t* blkTableUb = (__ubuf__ int32_t*)(blkTableTensor.GetPhyAddr());
    GetKVPhyAddrVFImpl<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, blockSize, shiftRightNum,
                                 sparseBlockSize, kvDim, kvStride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb,
                                                                            int64_t bS1Idx, int64_t s1Idx,
                                                                            int64_t validS2, int64_t alignNum,
                                                                            ConstInfo& constInfo)
{
    static constexpr int64_t numPerBlock = 32;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1U; // 每次处理1行
    dataCopyParams.blockLen = ((validS2 + alignNum - 1) / alignNum * alignNum) * sizeof(int64_t) / numPerBlock;
    dataCopyParams.srcGap = 0U;
    dataCopyParams.dstGap = 0U;
    // gm数据类型int32，实际元素类型int64，字节长度为2倍
    DataCopy(this->kvPhyAddrGm[(bS1Idx + s1Idx) * constInfo.sparseBlockCount * 2], kvPhyAddrUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopyPaTableToUb(LocalTensor<int32_t> blkTableUb,
                                                                            int64_t bIdx, ConstInfo& constInfo)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U; // 每次处理1行
    dataCopyParams.blockLen = constInfo.cmpMaxBlockNumPerBatch * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(blkTableUb, this->cmpBlockTableGm[bIdx * constInfo.cmpMaxBlockNumPerBatch], dataCopyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb,
                                                                              int64_t bS1Idx, int64_t s1Idx,
                                                                              int64_t validS2, ConstInfo& constInfo)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = validS2 * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(sparseIdxUb, this->cmpSparseIndicesGm[(bS1Idx + s1Idx) * constInfo.sparseBlockCount], dataCopyParams,
                padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
    __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
    __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
{
    if (oriKV != nullptr) {
        oriKVGm.SetGlobalBuffer((__gm__ KV_T*)(oriKV));
        oriKVGmBytes.SetGlobalBuffer((__gm__ uint8_t*)(oriKV));
    }

    if constexpr (isPa) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)oriBlockTable);
    }

    if constexpr (TEMPLATE_MODE != MQSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T*)cmpKV);
        cmpKVGmBytes.SetGlobalBuffer((__gm__ uint8_t*)cmpKV);
        if constexpr (isPa) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)cmpBlockTable);
        }
    }
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)oriSparseIndices);
    }

    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == MQSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)cmpSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T*)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::SoftmaxInitBuffer()
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
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo& constInfo)
{
    LocalTensor<float> sinksUb = this->sinksBuf.template Get<float>();
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(float);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<float> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::InitLocalBuffer(TPipe* pipe, ConstInfo& constInfo)
{
    SoftmaxInitBuffer();
    mte2ToSOffsetsId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
    sToMte2OffsetsId = GetTPipePtr()->AllocEventID<HardEvent::S_MTE2>();
    if (constInfo.isSoftmaxLseEnable) {
        tPipe->InitBuffer(outLseBuf_[0], 256U);
        tPipe->InitBuffer(outLseBuf_[1], 256U);
    }
    if constexpr (QUANT_MODE == SCALE_CONTIGUOUS_MODE::NONCONTIGUOUS) {
        const uint32_t scaleNumPerRow = 64;
        tPipe->InitBuffer(dequantScaleMode2Buf_, v0BaseSize * scaleNumPerRow * sizeof(float));
    }

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B

    tPipe->InitBuffer(stage0InBuf[0], dVTemplateTypeInput * v0BaseSize * sizeof(KV_T));
    tPipe->InitBuffer(stage0InBuf[1], dVTemplateTypeInput * v0BaseSize * sizeof(KV_T));
    tPipe->InitBuffer(stage0OutBuf[0], dVTemplateTypeInput * (v0BaseSize + 1) * sizeof(Q_T));
    tPipe->InitBuffer(stage0OutBuf[1], dVTemplateTypeInput * (v0BaseSize + 1) * sizeof(Q_T));

    tPipe->InitBuffer(stage1OutQue, 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));

    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    tPipe->InitBuffer(sparseIndiceBuf, s2BaseSize * sizeof(int32_t));
    tPipe->InitBuffer(paBlockIdxOffsetBuf, s2BaseSize * sizeof(int32_t));
    tPipe->InitBuffer(tailBlockOffsetBuf, s2BaseSize * sizeof(uint32_t));
    tPipe->InitBuffer(paBlockIdBuf, s2BaseSize * sizeof(int32_t));
    tPipe->InitBuffer(totalOffsetBuf, s2BaseSize * sizeof(int64_t));

    sparseIndiceTensor = sparseIndiceBuf.Get<int32_t>();
    paBlockIdxOffsetTensor = paBlockIdxOffsetBuf.Get<int32_t>();
    tailBlockOffsetTensor = tailBlockOffsetBuf.Get<uint32_t>();
    paBlockIdTensor = paBlockIdBuf.Get<int32_t>();
    totalOffsetTensor = totalOffsetBuf.Get<int64_t>();

    mte3ToVId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    vToMte3Id = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
    mte3ToVLseOutId_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    vToMte3LseOutId_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId_);
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
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::FreeEvent()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(mte2ToSOffsetsId);
    GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE2>(sToMte2OffsetsId);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVLseOutId_);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3LseOutId_);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
    WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[0]);
    WaitFlag<HardEvent::V_MTE2>(vToMte2V0Id[1]);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[0]);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVV0Id[1]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void MqsmlaCsaBlockVector<TEMPLATE_ARGS>::GetExtremeValue(T& negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float*)&tmp1);
}

TEMPLATES_DEF
class MqsmlaCsaBlockVectorDummy {
public:
    __aicore__ inline MqsmlaCsaBlockVectorDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, ConstInfo& constInfo)
    {}
    __aicore__ inline void InitOutput(ConstInfo& constInfo, RunParamStr& runParam, int32_t sIdx, int32_t gS1Cur) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                            __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                            __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sinks,
                                            __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv,
                                            __gm__ uint8_t* cmpResidualKv)
    {}
    __aicore__ inline void InitVecBlock(TPipe* pipe, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                        __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedOriKv,
                                        __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv) {};
    __aicore__ inline void InitLocalBuffer(TPipe* pipe, ConstInfo& constInfo) {}
    __aicore__ inline void ProcessVec1(outputl1Pos& outputBuf, mmResPos& bmm1ResBuf, RunInfo& runInfo,
                                       ConstInfo& constInfo)
    {}
    __aicore__ inline void ProcessVec2(mmResPos& bmm2ResBuf, mmResPos& bmm2ResBuf1, RunInfo& runInfo,
                                       ConstInfo& constInfo)
    {}
};
} // namespace BaseApi
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_ARCH92_H
