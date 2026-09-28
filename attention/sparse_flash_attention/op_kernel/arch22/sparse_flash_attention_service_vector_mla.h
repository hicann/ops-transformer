/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SFAT>
class SFAVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename SFAT::kvType;
    using OUT_T = typename SFAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline SFAVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &sfaVecInfo);
    __aicore__ inline void ProcessVec2L(const RunInfo &sfaVecInfo);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &sfaVecConstInfo,
                                      const SparseFlashAttentionTilingDataMla *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                const GlobalTensor<KV_T> &kvMergeGm,
                                                const GlobalTensor<KV_T> &keyRopeGm, const GlobalTensor<KV_T> &keyGm,
                                                const GlobalTensor<int32_t> &blkTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm,
                                                GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t sfaDealRows, uint32_t sfaColumns, uint32_t sfaActualColumns);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t sfaDealRows, uint32_t sfaColumns, uint32_t sfaActualColumns);
    // ================================Vector0==========================================
    __aicore__ inline void MergeKv(const RunInfo &sfaVecRunInfo);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const RunInfo &sfaVecRunInfo, int64_t s2IdLimit);
    __aicore__ inline int64_t GetKeyRopeGmOffset(int64_t realS2Idx, const RunInfo &sfaVecRunInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &sfaVecRunInfo);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &sfaVecRunInfo);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &sfaVecRunInfo);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &sfaVecRunInfo);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t sfaDealRows, uint32_t sfaColumns,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t sfaDealRows, uint32_t sfaColumns,
                                     uint64_t startId, uint64_t endId);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo,
                                                uint32_t startRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                 uint32_t sfaActualColumns);
    __aicore__ inline void AmlaVecCompute(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo,
                                          LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                          uint32_t startRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                          uint32_t sfaActualColumns);
    __aicore__ inline void ElewiseCompute(const RunInfo &sfaVecInfo, const LocalTensor<T> &mmResUb,
                                          uint32_t sfaDealRows, uint32_t sfaColumns);
    __aicore__ inline void ProcessAmlaNupdate(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    __aicore__ inline void CopyFALseToGm(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo,
                                         LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    __aicore__ inline void SetBmm2FirstSInnerBias(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo,
                                                uint32_t startRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                uint32_t sfaActualColumns);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &sfaVecInfo, const MSplitInfo &sfaVecSplitInfo,
                                            uint32_t mStartRow, uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &sfaVecInfo, LocalTensor<OUT_T> &attenOutUb,
                                                uint32_t sfaWorkspaceRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                uint32_t sfaActualColumns);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &sfaVecInfo, LocalTensor<T> &sfaBmm2ResultUb,
                                          uint32_t sfaWorkspaceRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                          uint32_t sfaActualColumns);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &sfaVecInfo, LocalTensor<T> &sfaBmm2ResultUb,
                                              uint32_t sfaWorkspaceRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                              uint32_t sfaActualColumns);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &sfaVecInfo, LocalTensor<T> &sfaBmm2ResultUb,
                                             uint32_t sfaWorkspaceRow, uint32_t sfaDealRows, uint32_t sfaColumns,
                                             uint32_t sfaActualColumns);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = SFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAT::templateMode;
    static constexpr bool FLASH_DECODE = SFAT::flashDecode;
    static constexpr SFA_LAYOUT LAYOUT_T = SFAT::layout;
    static constexpr SFA_LAYOUT KV_LAYOUT_T = SFAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint64_t SYNC_INPUT_V0BUF_FLAG = 6;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::SFA_BUFFER_BYTES_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::SFA_BUFFER_BYTES_1K;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::SFA_BUFFER_BYTES_32K / sizeof(T); // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                               // 32/4=8
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;

    const SparseFlashAttentionTilingDataMla *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo sfaVecConstInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K
    TBuf<> inputBuff2;  // 16K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff1;        // 32K
    TBuf<> v0ValidSizeBuff; // 8K

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<KV_T> ropeMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
};

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::SFA_BUFFER_BYTES_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::SFA_BUFFER_BYTES_8K * 2);  // 2:pingpong
    pipe->InitBuffer(outputBuff1, ConstInfo::SFA_BUFFER_BYTES_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::SFA_BUFFER_BYTES_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::SFA_BUFFER_BYTES_32K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::SFA_BUFFER_BYTES_8K);

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer
    pipe->InitBuffer(nValueBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);
    pipe->InitBuffer(cofValueBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);
    pipe->InitBuffer(aMlaSumBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::SFA_BUFFER_BYTES_1K * sfaVecConstInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::SFA_BUFFER_BYTES_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::SFA_BUFFER_BYTES_1K);

    nValueUb = nValueBuff.Get<T>();
    cofValueUb = cofValueBuff.Get<T>();
    aMlaSumUb = aMlaSumBuff.Get<T>();

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();
    ropeMergUb_ = inputBuff2.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitParams(
    const struct ConstInfo &sfaVecConstInfo, const SparseFlashAttentionTilingDataMla *__restrict tilingData)
{
    this->sfaVecConstInfo = sfaVecConstInfo;
    this->tilingData = tilingData;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                                    const GlobalTensor<KV_T> &kvMergeGm,
                                                                    const GlobalTensor<KV_T> &keyRopeGm,
                                                                    const GlobalTensor<KV_T> &keyGm,
                                                                    const GlobalTensor<int32_t> &blkTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm, GlobalTensor<int32_t> actualSeqLengthsQGm,
    GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm,
    GlobalTensor<int32_t> topKGm, GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
    this->softmaxMaxGm = softmaxMaxGm;
    this->softmaxSumGm = softmaxSumGm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                    GlobalTensor<UPDATE_T> vec2ResGm,
                                                                    GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                    GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyFALseToGm(const RunInfo &sfaVecInfo,
                                                             const MSplitInfo &sfaVecSplitInfo,
                                                             LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb)

{
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2;
    size_t size = sfaVecSplitInfo.vecDealM;

    int64_t offset = 0;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) { // lse layout为N2 T G
        uint64_t actualSeqQTotal =
            (sfaVecInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(sfaVecConstInfo.batchSize - 1);
        uint64_t actualSeqQPrefixSum = (sfaVecInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(sfaVecInfo.bIdx - 1);
        offset += sfaVecInfo.n2Idx * actualSeqQTotal * sfaVecConstInfo.gSize +
                  (actualSeqQPrefixSum + sfaVecInfo.gS1Idx / sfaVecConstInfo.gSize) * sfaVecConstInfo.gSize +
                  sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM;
    } else {
        offset += sfaVecInfo.bIdx * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.qSeqSize * sfaVecConstInfo.gSize +
                  sfaVecInfo.n2Idx * sfaVecConstInfo.qSeqSize * sfaVecConstInfo.gSize +
                  sfaVecInfo.gS1Idx / sfaVecConstInfo.gSize * sfaVecConstInfo.gSize + sfaVecSplitInfo.nBufferStartM +
                  sfaVecSplitInfo.vecStartM;
    }

    if (sfaVecInfo.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(T) * size;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        size_t alignedSize = (sizeof(T) * size + 31) / 32 * 32 / sizeof(T);
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxMaxUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxMaxGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxSumUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxSumGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(softmaxSumGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(softmaxMaxGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &sfaVecInfo,
                                                                           const MSplitInfo &sfaVecSplitInfo,
                                                                           LocalTensor<T> &softmaxSumUb,
                                                                           LocalTensor<T> &softmaxMaxUb)
{
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2;
    size_t size = sfaVecSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t sfaAccumOutputCount = CalcAccumOffset(sfaVecInfo.bIdx, sfaVecInfo.gS1Idx);
    uint64_t offset =
        (sfaAccumOutputCount * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.mBaseSize +               // taskoffset
         sfaVecInfo.tndCoreStartKVSplitPos * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.mBaseSize + // 份数offset
         sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM) *
        FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (sfaVecInfo.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxSumUb[baseOffset], (sfaVecSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxMaxUb[baseOffset], (sfaVecSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ElewiseCompute(const RunInfo &sfaVecInfo, const LocalTensor<T> &mmResUb,
                                                              uint32_t sfaDealRows, uint32_t sfaColumns)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), sfaDealRows * sfaColumns);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // v0的无效值判断
        uint64_t s2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + sfaVecInfo.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t s2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + sfaVecInfo.loop % MERGE_CACHE_GM_BUF_NUM);

        int64_t s2ProcessSize = sfaVecInfo.actualSingleProcessSInnerSize;
        int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * sfaVecConstInfo.sparseBlockSize);
        int64_t s2Mid = CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize;
        if (s2Mid > s2ProcessSize) {
            s2Mid = s2ProcessSize;
        }
        if (unlikely(s2ValidSizeFirstPart < s2Mid)) {
            int64_t s2StartCeilAlign = CeilAlign(s2ValidSizeFirstPart, 8);
            int64_t s2MidFloorAlign = s2Mid / 8 * 8;
            // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
            // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
            // 场景二 s2Mid <= s2ValidSizeFirstPart + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2Mid 第一阶段取到mid
            // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
            SetInfInBlk(mmResUb, sfaDealRows, sfaColumns, s2ValidSizeFirstPart,
                        s2StartCeilAlign >= s2Mid ? s2Mid : s2StartCeilAlign);
            SetMidInf(mmResUb, sfaDealRows, sfaColumns, s2StartCeilAlign, s2MidFloorAlign);
            SetInfInBlk(mmResUb, sfaDealRows, sfaColumns,
                        s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign, s2Mid);
        }
        if (unlikely(s2ValidSizeSecondPart < s2ProcessSize - s2Mid)) {
            // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
            // 场景二 s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
            // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
            int64_t s2StartCeilAlign = CeilAlign(s2Mid + s2ValidSizeSecondPart, 8);
            int64_t s2EndFloorAlign = s2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, sfaDealRows, sfaColumns, s2Mid + s2ValidSizeSecondPart,
                        s2StartCeilAlign >= s2ProcessSize ? s2ProcessSize : s2StartCeilAlign);
            SetMidInf(mmResUb, sfaDealRows, sfaColumns, s2StartCeilAlign, s2EndFloorAlign);
            SetInfInBlk(mmResUb, sfaDealRows, sfaColumns,
                        s2StartCeilAlign <= s2EndFloorAlign ? s2EndFloorAlign : s2StartCeilAlign, s2ProcessSize);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t sfaDealRows,
                                                           uint32_t sfaColumns, uint64_t startId, uint64_t endId)
{
    //       startId     endId
    // x x x   0      0   0     x x x
    // 从startId到endId部分置-inf, endId、startId为endId一个blk内部的下标
    if (startId >= endId) {
        return;
    }

    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = (1 << (startId - startFloorAlignSize)) - 1;
    uint64_t notComputePostMaskOneBlk = ~((1 << (endId - startFloorAlignSize)) - 1);
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;

    uint64_t maskOneBlk = ~notComputeMaskOneBlk;
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    for (uint64_t rowId = 0; rowId < sfaDealRows; rowId += 8) {
        Duplicate(mmResUb[rowId * sfaColumns + startFloorAlignSize], SOFTMAX_MIN_NUM, mask, 1, CeilDiv(sfaColumns, 8),
                  0);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SetMidInf(const LocalTensor<T> &mmResUb, uint32_t sfaDealRows,
                                                         uint32_t sfaColumns, uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分置-inf, startId、endId为32B对齐的下标
    for (uint64_t rowId = 0; rowId < sfaDealRows; rowId++) {
        Duplicate(mmResUb[rowId * sfaColumns + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SoftmaxFlashV2Compute(const RunInfo &sfaVecInfo,
                                                                     const MSplitInfo &sfaVecSplitInfo,
                                                                     LocalTensor<T> &mmResUb,
                                                                     LocalTensor<uint8_t> &softmaxTmpUb,
                                                                     uint32_t startRow, uint32_t sfaDealRows,
                                                                     uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = sfaVecInfo.loop % (sfaVecConstInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (sfaVecInfo.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (sfaVecInfo.loop - 1) % (sfaVecConstInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    if (sfaActualColumns != 0) {
        SoftMaxShapeInfo srcShape{sfaDealRows, sfaColumns, sfaDealRows, sfaActualColumns};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, SFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        uint32_t dealRowCountAlign = SFAAlign(sfaDealRows, FP32_BLOCK_ELEMENT_NUM);
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCountAlign);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCountAlign);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::AmlaVecCompute(const RunInfo &sfaVecInfo,
                                                              const MSplitInfo &sfaVecSplitInfo,
                                                              LocalTensor<T> &mmResUb,
                                                              LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow,
                                                              uint32_t sfaDealRows, uint32_t sfaColumns,
                                                              uint32_t sfaActualColumns)
{
    uint32_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t calCount = sfaDealRows;
    uint32_t outIdx = sfaVecInfo.loop % (sfaVecConstInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // compute n(i)
    LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
    LocalTensor<T> nUpdateTmp = nTmp[SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);

    PipeBarrier<PIPE_V>();
    Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
    PipeBarrier<PIPE_V>();

    uint32_t prOutIdx = (sfaVecInfo.loop - 1) % (sfaVecConstInfo.preLoadNum);
    uint32_t PreSoftmaxOutOffset = prOutIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // n(i) - n(i-1)
    if (sfaVecInfo.isFirstSInnerLoop) {
        Duplicate(nUpdateTmp, ConstInfo::FLOAT_ZERO, calCount); // n1=n0
    } else {
        Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
    }
    PipeBarrier<PIPE_V>();
    // update n(i), DataCopy not support when calCount is not align 32B, so use Adds
    Adds(nValueUb[softmaxOutOffset], nTmp, ConstInfo::FLOAT_ZERO, calCount);
    PipeBarrier<PIPE_V>();

    // update softmax res
    LocalTensor<T> nUpdateTmp2 = nTmp[2 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<KV_T> nTmp_KvT = nTmp[3 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)].template ReinterpretCast<KV_T>();
    LocalTensor<T> tmpCofUb = nTmp[4 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<T> epsUb = nTmp[5 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
    PipeBarrier<PIPE_V>();
    Add(nTmp, nUpdateTmp2, nTmp, calCount);
    PipeBarrier<PIPE_V>();
    Muls(nTmp, nTmp, LN2, calCount);
    PipeBarrier<PIPE_V>();
    Exp(nTmp, nTmp, calCount);
    PipeBarrier<PIPE_V>();
    Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount); // fp32->fp16/bf16
    PipeBarrier<PIPE_V>();
    Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount); // fp16/bf16->fp32
    PipeBarrier<PIPE_V>();
    if (sfaVecInfo.s2Idx + 1 == sfaVecInfo.curSInnerLoopTimes) {
        Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
    }
    if (sfaActualColumns == 0) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        return;
    }
    LocalTensor<T> nTmp3 = nTmp[6 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Brcb(nTmp3, nUpdateTmp2, (sfaDealRows + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    RowMuls(mmResUb, mmResUb, nTmp3, sfaDealRows, sfaColumns, sfaActualColumns);

    Div(tmpCofUb, nTmp, nUpdateTmp2, calCount); // cof(i)=tmpS32/tmpS16
    if (sfaVecInfo.isFirstSInnerLoop) {
        Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount); // cof_0=1
        PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount); // 1 / cof(i)
    } else {
        PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount); // cof(i - 1) / cof(i)
    }
    PipeBarrier<PIPE_V>();

    Adds(cofValueUb[softmaxOutOffset], tmpCofUb, ConstInfo::FLOAT_ZERO, calCount); // store cof(i)
    Adds(epsUb, epsUb, (T)(-1.0), calCount);                                       // cof(i - 1) / cof(i) - 1
    PipeBarrier<PIPE_V>();
    Muls(epsUb, epsUb, (T)1.5, calCount); // (cof(i - 1) - cof(i)) / cof(i) * 1.5

    Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount); // N = max(n(i) - n(i-1), -30)
    PipeBarrier<PIPE_V>();
    Adds(epsUb, epsUb, (T)(0.000001), calCount);
    PipeBarrier<PIPE_V>();
    Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
    PipeBarrier<PIPE_V>();
    Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount); // N = N * pow(2, 23)
    PipeBarrier<PIPE_V>();

    // nUpdate int32 out
    LocalTensor<int32_t> tmQue = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    LocalTensor<int32_t> nInt32Out = tmQue[startRow]; // 缓存nUpdate

    Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, sfaDealRows);
    PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::DealBmm1ResBaseBlock(const RunInfo &sfaVecInfo,
                                                                    const MSplitInfo &sfaVecSplitInfo,
                                                                    uint32_t startRow, uint32_t sfaDealRows,
                                                                    uint32_t sfaColumns, uint32_t loopId)
{
    uint32_t computeSize = sfaDealRows * sfaColumns;
    uint64_t inOutGmOffset = (sfaVecInfo.loop % sfaVecConstInfo.preLoadNum) * sfaVecConstInfo.mmResUbSize +
                             (sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM + startRow) * sfaColumns;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(sfaVecInfo, mmResUb, sfaDealRows, sfaColumns);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(sfaVecInfo, sfaVecSplitInfo, mmResUb, softmaxTmpUb, startRow, sfaDealRows, sfaColumns,
                          sfaVecInfo.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    AmlaVecCompute(sfaVecInfo, sfaVecSplitInfo, mmResUb, softmaxTmpUb, startRow, sfaDealRows, sfaColumns,
                   sfaVecInfo.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SetBmm2FirstSInnerBias(const RunInfo &sfaVecInfo,
                                                                      const MSplitInfo &sfaVecSplitInfo)
{
    uint32_t mSplitSize = 16U;
    uint64_t baseoffset = (sfaVecInfo.bn2IdxInCurCore % sfaVecConstInfo.preLoadNum) * sfaVecConstInfo.bmm2ResUbSize +
                          (sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM) * sfaVecConstInfo.headDim;
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    LocalTensor<int32_t> tmpTensor = outputBuff1.Get<int32_t>();
    Duplicate(tmpTensor, static_cast<int32_t>(394264576),
              mSplitSize * sfaVecConstInfo.headDim); // 394264576 : fp32下2^(-80)的二进制表示对应的int32数值
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint32_t loopCount = (sfaVecSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    for (uint32_t loop = 0; loop < loopCount; loop++) {
        DataCopy(mm2ResInt32Gm[baseoffset + loop * mSplitSize * sfaVecConstInfo.headDim], tmpTensor,
                 mSplitSize * sfaVecConstInfo.headDim);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessAmlaNupdate(const RunInfo &sfaVecInfo,
                                                                  const MSplitInfo &sfaVecSplitInfo)
{
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }
    if (sfaVecInfo.isFirstSInnerLoop) {
        SetBmm2FirstSInnerBias(sfaVecInfo, sfaVecSplitInfo);
        return;
    }

    LocalTensor<int32_t> nUpdateTensor = outputBuff2.Get<int32_t>(); // shape:1/2*s1*g
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    constexpr uint32_t dGroupSize = 128U;
    constexpr uint32_t mSplitSize =
        64U; // tmpQue size 32KB，一次只能处理64个N，最大保存的数据大小：64*128*sizeof(int32)
    constexpr uint32_t ONE_BLOCK_SIZE = 32U; // 32B

    uint32_t subMSize = SFAAlign(sfaVecSplitInfo.vecDealM, 16U);
    uint16_t elementPerBlock = ONE_BLOCK_SIZE / sizeof(int32_t); // 单个datablock的元素数，int32_t类型的为32/4=8
    uint32_t loopCount = (subMSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = subMSize - (loopCount - 1) * mSplitSize; // 尾块

    for (uint32_t loop = 0, processMSize = mSplitSize; loop < loopCount; loop++) {
        if (loop == (loopCount - 1)) {
            processMSize = tailSplitSize;
        }
        LocalTensor<int32_t> tmpQue = outputBuff1.Get<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        // (m,1)单次brcb扩充成(m,8), 重复16次, 扩充为(m,128)
        for (uint32_t i = 0; i < dGroupSize / elementPerBlock; i++) {
            Brcb(tmpQue[i * elementPerBlock], nUpdateTensor[loop * mSplitSize],
                 static_cast<uint8_t>((processMSize + elementPerBlock - 1) / elementPerBlock),
                 {static_cast<uint16_t>(
                      dGroupSize / elementPerBlock), // 单次迭代内，目的操作数不同datablock间地址步长,单位为datablock
                  static_cast<uint16_t>(dGroupSize)}); // 相邻迭代间，目的操作数相同datablock地址步长
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t baseoffset =
            (sfaVecInfo.bn2IdxInCurCore % sfaVecConstInfo.preLoadNum) * sfaVecConstInfo.bmm2ResUbSize +
            (sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM + loop * mSplitSize) * sfaVecConstInfo.headDim;

        SetAtomicAdd<int32_t>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(processMSize);
        dataCopyParams.blockLen = dGroupSize * sizeof(int32_t) / ONE_BLOCK_SIZE; // 每个block是128个元素，单位为32B
        dataCopyParams.srcStride = 0; // 前面一个数据块的尾与后面数据块的头的间隔
        dataCopyParams.dstStride = static_cast<uint16_t>((sfaVecConstInfo.headDim - dGroupSize) * sizeof(int32_t) /
                                                         ONE_BLOCK_SIZE);     // 单位为32B
        for (uint32_t i = 0; i < sfaVecConstInfo.headDim / dGroupSize; i++) { // 4=512/128
            DataCopy(mm2ResInt32Gm[baseoffset + i * dGroupSize], tmpQue, dataCopyParams);
        }
        SetAtomicNone();
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec1SingleBuf(const RunInfo &sfaVecInfo,
                                                                    const MSplitInfo &sfaVecSplitInfo)
{
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = sfaVecInfo.actualSingleProcessSInnerSize == 0 ?
                              16 :
                              BASE_BLOCK_MAX_ELEMENT_NUM / sfaVecInfo.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > sfaVecSplitInfo.vecDealM) {
        mSplitSize = sfaVecSplitInfo.vecDealM;
    }
    uint32_t loopCount = (sfaVecSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = sfaVecSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 256 * sizeof(int32_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> padParams;
        // 额外偏移128个元素，避免不同loop下v0和v1互相影响
        DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[sfaVecInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
                    dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(loopCount == 0)) {
            // scalar同步影响较大，挪到循环内部进行
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(sfaVecInfo, sfaVecSplitInfo, i * mSplitSize, dealSize,
                             sfaVecInfo.actualSingleProcessSInnerSizeAlign, i);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                            int64_t topkGmBaseOffset, const RunInfo &sfaVecRunInfo)
{
    int64_t topkGmIdx =
        (s2GmOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize) / sfaVecConstInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= sfaVecConstInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(sfaVecConstInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + sfaVecRunInfo.s2Idx * sfaVecConstInfo.s2BaseSize) %
                                     sfaVecConstInfo.sparseBlockSize);
}

template <typename SFAT>
__aicore__ inline int64_t SFAVectorService<SFAT>::GetKeyGmOffset(int64_t realS2Idx, const RunInfo &sfaVecRunInfo,
                                                                 int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / sfaVecConstInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % sfaVecConstInfo.kvCacheBlockSize;
        realKeyGmOffset = blkTableGm_.GetValue(sfaVecRunInfo.bIdx * sfaVecConstInfo.maxBlockNumPerBatch + blkTableIdx) *
                              static_cast<int64_t>(sfaVecConstInfo.kvCacheBlockSize) *
                              static_cast<int64_t>(sfaVecConstInfo.kvHeadNum) +
                          blkTableOffset;
    } else {
        realKeyGmOffset =
            (sfaVecRunInfo.tensorBOffset + realS2Idx * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.headDim) /
            sfaVecConstInfo.headDim;
    }
    return realKeyGmOffset;
}

template <typename SFAT>
__aicore__ inline int64_t SFAVectorService<SFAT>::GetKeyRopeGmOffset(int64_t realS2Idx, const RunInfo &sfaVecRunInfo,
                                                                     int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    if constexpr (!SFAT::hasRope) {
        return -1;
    }
    int64_t realKeyRopeGmOffset = 0;
    realKeyRopeGmOffset =
        (sfaVecRunInfo.tensorBRopeOffset + realS2Idx * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.headDimRope) /
        sfaVecConstInfo.headDimRope;
    return realKeyRopeGmOffset;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                              int64_t realS2Idx, int64_t keyBNBOffset,
                                                              int64_t s2IdLimit, const RunInfo &sfaVecRunInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count = (realS2Idx + sfaVecConstInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx :
                                                                                      sfaVecConstInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * sfaVecConstInfo.headDim * sizeof(KV_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * sfaVecConstInfo.headDim],
                keyGm_[keyBNBOffset * sfaVecConstInfo.headDim], intriParams, padParams);
    if constexpr (!SFAT::hasRope) {
        mte2Size += validS2Count;
        return;
    }
    intriParams.blockLen = validS2Count * sfaVecConstInfo.headDimRope * sizeof(KV_T);

    DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * sfaVecConstInfo.headDimRope],
                keyRopeGm_[keyBNBOffset * sfaVecConstInfo.headDimRope], intriParams, padParams);
    mte2Size += validS2Count;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                        int64_t realS2Idx1, int64_t realS2Idx2,
                                                        const RunInfo &sfaVecRunInfo)
{
    int64_t s2IdLimit = sfaVecRunInfo.curActualSeqLenOri;
    if (sfaVecConstInfo.sparseMode == 3) {
        s2IdLimit = sfaVecRunInfo.curActualSeqLenOri - sfaVecRunInfo.actS1Size +
                    sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize + 1;
    }

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, sfaVecRunInfo, s2IdLimit);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, sfaVecRunInfo, s2IdLimit);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableSrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                                     sfaVecConstInfo.sparseBlockSize);
        keySrcStride = blkTableSrcStride * sfaVecConstInfo.headDim * sizeof(KV_T);
        keyRopeSrcStride = blkTableSrcStride * sfaVecConstInfo.headDimRope * sizeof(KV_T);
    } else {
        int64_t keyRopeOffset1 = GetKeyRopeGmOffset(realS2Idx1, sfaVecRunInfo, s2IdLimit);
        int64_t keyRopeOffset2 = GetKeyRopeGmOffset(realS2Idx2, sfaVecRunInfo, s2IdLimit);
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                        sfaVecConstInfo.sparseBlockSize) *
                       sfaVecConstInfo.headDim * sizeof(KV_T);
        keyRopeSrcStride =
            ((keyRopeOffset1 > keyRopeOffset2 ? (keyRopeOffset1 - keyRopeOffset2) : (keyRopeOffset2 - keyRopeOffset1)) -
             sfaVecConstInfo.sparseBlockSize) *
            sfaVecConstInfo.headDimRope * sizeof(KV_T);
    }

    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
                 (!PAGE_ATTENTION && (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0)) ||
                 realS2Idx1 + sfaVecConstInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + sfaVecConstInfo.sparseBlockSize >= s2IdLimit)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, sfaVecRunInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, sfaVecRunInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = sfaVecConstInfo.sparseBlockSize * sfaVecConstInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * sfaVecConstInfo.headDim],
                    keyGm_[startGmOffset * sfaVecConstInfo.headDim], intriParams, padParams);

        if constexpr (SFAT::hasRope) {
            intriParams.blockLen = sfaVecConstInfo.sparseBlockSize * sfaVecConstInfo.headDimRope * sizeof(KV_T);
            intriParams.dstStride = 0;
            intriParams.srcStride = keyRopeSrcStride;
            DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * sfaVecConstInfo.headDimRope],
                        keyRopeGm_[startGmOffset * sfaVecConstInfo.headDimRope], intriParams, padParams);
        }
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * sfaVecConstInfo.sparseBlockSize;
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                 int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                 const RunInfo &sfaVecRunInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = sfaVecConstInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % 4 * 512 * 576 + (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * 32 * 512], dataCopyParams);

    if constexpr (SFAT::hasRope) {
        dataCopyParams.blockLen = sfaVecConstInfo.headDimRope * sizeof(KV_T);
        DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % 4 * 512 * 576 + 512 * 512 +
                               (s2GmStartOffset + mte3Size) * sfaVecConstInfo.headDimRope],
                    ropeMergUb_[mergeMte3Idx % 2 * 32 * 64], dataCopyParams);
    }
}

// b s1 k
template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::MergeKv(const RunInfo &sfaVecRunInfo)
{
    int64_t s2ProcessSize = sfaVecRunInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * sfaVecConstInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum =
            (sfaVecRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(sfaVecRunInfo.bIdx - 1);
        topkGmBaseOffset += (actualSeqQPrefixSum + sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize) *
                                sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.sparseBlockCount +
                            sfaVecRunInfo.n2Idx * sfaVecConstInfo.sparseBlockCount;
    } else {
        topkGmBaseOffset += sfaVecRunInfo.bIdx * sfaVecConstInfo.qSeqSize * sfaVecConstInfo.sparseBlockCount +
                            sfaVecRunInfo.gS1Idx / sfaVecConstInfo.gSize * sfaVecConstInfo.sparseBlockCount;
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize;
    int64_t s2GmLimit =
        GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * sfaVecConstInfo.sparseBlockSize : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * sfaVecConstInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, sfaVecRunInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + sfaVecConstInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, sfaVecRunInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, sfaVecRunInfo);
        if ((mte2Size - mte3Size + 2 * sfaVecConstInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * sfaVecConstInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, sfaVecRunInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }

    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), sfaVecConstInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sfaVecConstInfo.headDim * sizeof(KV_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                   s2GmOffset * sfaVecConstInfo.headDim],
                        kvMergUb_, dataCopyParams);
        }
        if constexpr (SFAT::hasRope) {
            dataCopyParams.blockLen = sfaVecConstInfo.headDimRope * sizeof(KV_T);
            for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
                DataCopyPad(kvMergeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                       512 * sfaVecConstInfo.headDim + s2GmOffset * sfaVecConstInfo.headDimRope],
                            kvMergUb_, dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[sfaVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_S>(SYNC_INPUT_V0BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_S>(SYNC_INPUT_V0BUF_FLAG);
    return;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec1L(const RunInfo &sfaVecInfo)
{
    uint32_t nBufferLoopTimes =
        (sfaVecInfo.actMBaseSize + sfaVecConstInfo.nBufferMBaseSize - 1) / sfaVecConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail = sfaVecInfo.actMBaseSize - (nBufferLoopTimes - 1) * sfaVecConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo sfaVecSplitInfo;
        sfaVecSplitInfo.nBufferIdx = i;
        sfaVecSplitInfo.nBufferStartM = i * sfaVecConstInfo.nBufferMBaseSize;
        sfaVecSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? sfaVecConstInfo.nBufferMBaseSize : nBufferTail;

        sfaVecSplitInfo.vecDealM = (sfaVecSplitInfo.nBufferDealM <= 16) ?
                                       sfaVecSplitInfo.nBufferDealM :
                                       (((sfaVecSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        sfaVecSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            sfaVecSplitInfo.vecStartM = sfaVecSplitInfo.vecDealM;
            sfaVecSplitInfo.vecDealM = sfaVecSplitInfo.nBufferDealM - sfaVecSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(sfaVecConstInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(sfaVecInfo, sfaVecSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(sfaVecConstInfo.syncV1C2);
        CrossCoreWaitFlag(sfaVecConstInfo.syncC2V1);
        // add nUpdate to mm2ResGm
        if (sfaVecInfo.actualSingleProcessSInnerSize != 0) {
            ProcessAmlaNupdate(sfaVecInfo, sfaVecSplitInfo);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(sfaVecConstInfo.syncV1NupdateC2);
        }
        // move lse for flash decode or FA
        if (sfaVecInfo.s2Idx == sfaVecInfo.curSInnerLoopTimes - 1 &&
            (sfaVecConstInfo.returnSoftmaxLse || sfaVecInfo.tndIsS2SplitCore)) {
            uint32_t outIdx = sfaVecInfo.loop % (sfaVecConstInfo.preLoadNum);
            auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
            auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
            if (sfaVecConstInfo.returnSoftmaxLse) {
                CopyFALseToGm(sfaVecInfo, sfaVecSplitInfo, sumTensor, maxTensor);
            }
            if (sfaVecInfo.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    ComputeLogSumExpAndCopyToGm(sfaVecInfo, sfaVecSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename SFAT>
__aicore__ inline uint64_t SFAVectorService<SFAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec2SingleBuf(const RunInfo &sfaVecInfo,
                                                                    const MSplitInfo &sfaVecSplitInfo)
{
    if (sfaVecInfo.s2Idx + 1 != sfaVecInfo.curSInnerLoopTimes) {
        return;
    }
    if (sfaVecSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(sfaVecInfo, sfaVecSplitInfo, 0, sfaVecSplitInfo.vecDealM);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec2L(const RunInfo &sfaVecInfo)
{
    uint32_t nBufferLoopTimes =
        (sfaVecInfo.actMBaseSize + sfaVecConstInfo.nBufferMBaseSize - 1) / sfaVecConstInfo.nBufferMBaseSize;
    uint32_t nBufferTail = sfaVecInfo.actMBaseSize - (nBufferLoopTimes - 1) * sfaVecConstInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo sfaVecSplitInfo;
        sfaVecSplitInfo.nBufferIdx = i;
        sfaVecSplitInfo.nBufferStartM = i * sfaVecConstInfo.nBufferMBaseSize;
        sfaVecSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? sfaVecConstInfo.nBufferMBaseSize : nBufferTail;

        sfaVecSplitInfo.vecStartM = 0;
        sfaVecSplitInfo.vecDealM = (sfaVecSplitInfo.nBufferDealM <= 16) ?
                                       sfaVecSplitInfo.nBufferDealM :
                                       (((sfaVecSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        if (GetBlockIdx() % 2 == 1) {
            sfaVecSplitInfo.vecStartM = sfaVecSplitInfo.vecDealM;
            sfaVecSplitInfo.vecDealM = sfaVecSplitInfo.nBufferDealM - sfaVecSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(sfaVecConstInfo.syncC2V2);
        ProcessVec2SingleBuf(sfaVecInfo, sfaVecSplitInfo);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec2Inner(const RunInfo &sfaVecInfo,
                                                                const MSplitInfo &sfaVecSplitInfo, uint32_t mStartRow,
                                                                uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / sfaVecConstInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(sfaVecInfo, sfaVecSplitInfo, i * mSplitSize + mStartRow, dealSize, sfaVecConstInfo.headDim,
                             sfaVecConstInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::GetConfusionTransposeTiling(int64_t numR, int64_t numC,
                                                                           const uint32_t stackBufferSize,
                                                                           const uint32_t typeSize,
                                                                           ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::Bmm2FDDataCopyOut(const RunInfo &sfaVecInfo,
                                                                 LocalTensor<T> &sfaBmm2ResultUb,
                                                                 uint32_t sfaWorkspaceRow, uint32_t sfaDealRows,
                                                                 uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, sfaBmm2ResultUb, sfaColumns * sfaDealRows);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t sfaAccumOutputCount = CalcAccumOffset(sfaVecInfo.bIdx, sfaVecInfo.gS1Idx);
    uint64_t offset = sfaAccumOutputCount * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.mBaseSize *
                          sfaVecConstInfo.headDim + // taskoffset
                      sfaVecInfo.tndCoreStartKVSplitPos * sfaVecConstInfo.kvHeadNum * sfaVecConstInfo.mBaseSize *
                          sfaVecConstInfo.headDim +       // 份数offset
                      sfaWorkspaceRow * sfaActualColumns; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (sfaVecInfo.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = sfaDealRows;
        dataCopyParams.blockLen = sfaActualColumns * sizeof(T);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = (sfaColumns - sfaActualColumns) / (BYTE_BLOCK / sizeof(T));
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, sfaDealRows * sfaActualColumns, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::Bmm2DataCopyOutTrans(const RunInfo &sfaVecInfo,
                                                                    LocalTensor<OUT_T> &attenOutUb,
                                                                    uint32_t sfaWorkspaceRow, uint32_t sfaDealRows,
                                                                    uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = sfaDealRows;
    dataCopyParams.blockLen = sfaActualColumns * sizeof(OUT_T);
    dataCopyParams.srcStride = (sfaColumns - sfaActualColumns) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[sfaVecInfo.attenOutOffset + sfaWorkspaceRow * sfaActualColumns], attenOutUb,
                dataCopyParams);
    return;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::Bmm2CastAndCopyOut(const RunInfo &sfaVecInfo,
                                                                  LocalTensor<T> &sfaBmm2ResultUb,
                                                                  uint32_t sfaWorkspaceRow, uint32_t sfaDealRows,
                                                                  uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, sfaBmm2ResultUb, AscendC::RoundMode::CAST_RINT, sfaDealRows * sfaColumns);
    } else {
        Cast(tmpBmm2ResCastTensor, sfaBmm2ResultUb, AscendC::RoundMode::CAST_ROUND, sfaDealRows * sfaColumns);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(sfaVecInfo, tmpBmm2ResCastTensor, sfaWorkspaceRow, sfaDealRows, sfaColumns, sfaActualColumns);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::Bmm2ResCopyOut(const RunInfo &sfaVecInfo,
                                                              LocalTensor<T> &sfaBmm2ResultUb, uint32_t sfaWorkspaceRow,
                                                              uint32_t sfaDealRows, uint32_t sfaColumns,
                                                              uint32_t sfaActualColumns)
{
    if constexpr (FLASH_DECODE) {
        if (sfaVecInfo.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(sfaVecInfo, sfaBmm2ResultUb, sfaWorkspaceRow, sfaDealRows, sfaColumns, sfaActualColumns);
        } else {
            Bmm2CastAndCopyOut(sfaVecInfo, sfaBmm2ResultUb, sfaWorkspaceRow, sfaDealRows, sfaColumns, sfaActualColumns);
        }
    } else {
        Bmm2CastAndCopyOut(sfaVecInfo, sfaBmm2ResultUb, sfaWorkspaceRow, sfaDealRows, sfaColumns, sfaActualColumns);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::DealBmm2ResBaseBlock(const RunInfo &sfaVecInfo,
                                                                    const MSplitInfo &sfaVecSplitInfo,
                                                                    uint32_t startRow, uint32_t sfaDealRows,
                                                                    uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    uint32_t vec2ComputeSize = sfaDealRows * sfaColumns;
    uint32_t mStart = sfaVecSplitInfo.nBufferStartM + sfaVecSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset =
        (sfaVecInfo.bn2IdxInCurCore % sfaVecConstInfo.preLoadNum) * sfaVecConstInfo.bmm2ResUbSize + mStart * sfaColumns;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    // 将绝对值大于1e10的数置为0
    LocalTensor<T> sfaBmm2ResultUb = tmpBuff1.Get<T>();
    sfaBmm2ResultUb.SetSize(vec2ComputeSize);
    LocalTensor<T> absBmm2ResUb = sfaBmm2ResultUb.template ReinterpretCast<T>();
    Abs(absBmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    Select(tmpBmm2ResUb, cmpMaskUb, tmpBmm2ResUb, ConstInfo::FLOAT_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           vec2ComputeSize);
    PipeBarrier<PIPE_V>();
    uint32_t baseOffset = sfaVecSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t idx = sfaVecInfo.loop % (sfaVecConstInfo.preLoadNum);
    LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[384]; // sumUb用临时内存 16 * 32B  = 512B
    Brcb(tmpSumUb, aMlaSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (sfaDealRows + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    RowDivs(sfaBmm2ResultUb, tmpBmm2ResUb, tmpSumUb, sfaDealRows, sfaColumns, sfaActualColumns);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    Bmm2ResCopyOut(sfaVecInfo, sfaBmm2ResultUb, mStart, sfaDealRows, sfaColumns, sfaActualColumns);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub,
                                                       LocalTensor<float> src1Ub, uint32_t sfaDealRows,
                                                       uint32_t sfaColumns, uint32_t sfaActualColumns)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = sfaActualColumns / dtypeMask;
    uint32_t dRemain = sfaActualColumns % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = sfaColumns / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = sfaColumns / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= sfaDealRows) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, sfaDealRows, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < sfaDealRows; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += sfaColumns;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, sfaDealRows, repeatParamsDiv);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub,
                                                       LocalTensor<T> src1Ub, uint32_t sfaDealRows, uint32_t sfaColumns,
                                                       uint32_t sfaActualColumns)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dRemain = sfaActualColumns % repeatElementNum;
    uint32_t dLoop = sfaActualColumns / repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (sfaColumns < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = sfaColumns / blockElementNum;
        repeatParams.src0RepStride = sfaColumns / blockElementNum;
        repeatParams.src1RepStride = 1;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= sfaDealRows) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, sfaDealRows, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            for (uint32_t i = 0; i < sfaDealRows; i++) {
                Mul(dstUb[i * sfaColumns], src0Ub[i * sfaColumns], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                    columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, sfaDealRows,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.dstRepStride = 8;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < sfaDealRows; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * sfaColumns], src0Ub[i * sfaColumns], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * sfaColumns + dLoop * repeatElementNum], src0Ub[i * sfaColumns + dLoop * repeatElementNum],
                    src1Ub[i * blockElementNum], dRemain, 1, repeatParams);
            }
        }
    }
}

#endif // SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
