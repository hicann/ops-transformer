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
 * \file flash_attn_block_vec_dn.h
 * \brief FANoQuantGqaBlockVecDn —— Dn 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTN_BLOCK_VEC_DN_H_
#define FLASH_ATTN_BLOCK_VEC_DN_H_

#include <limits>
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/const_def.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/init_output.h"
#include "memory_copy_arch35.h"
#include "../utils/attn_sink_gs1.h"

using namespace AscendC::Impl::Detail;
using namespace AscendC;
using namespace FaVectorApi;

namespace FlashAttnKernel {

template <typename FA_T>
class FANoQuantGqaBlockVecDn {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;

    using T = float;
    static constexpr uint32_t dTemplateAlign64 = BaseApi::Align64Func((uint16_t)FA_T::dVBaseSize);

    // DN alternates its two vector result slots.
    static constexpr uint32_t DN_VEC_SLOTS = 2;
    // 索引使用 loop & (DN_VEC_SLOTS - 1) 代替 loop % DN_VEC_SLOTS，要求 DN_VEC_SLOTS 必须是2的幂，否则位掩码结果错误
    static_assert(DN_VEC_SLOTS > 0 && (DN_VEC_SLOTS & (DN_VEC_SLOTS - 1)) == 0,
                  "DN_VEC_SLOTS must be a power of two for bitmask indexing");

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t CC_MM_0 = 0U;
    static constexpr uint32_t CC_MM_1 = 1U;
    static constexpr uint32_t CC_MM_2 = 2U;
    static constexpr uint32_t CC_MM_3 = 3U;
    static constexpr uint32_t CC_L1P_0 = 5U;
    static constexpr uint32_t CC_L1P_1 = 6U;
    static constexpr uint32_t CC_L1P_2 = 7U;

    // 核内同步ID
    // MTE3<->V, 输出buffer
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = 3;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = 5;

    // L1
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> l1PBuffersDn_;

    // UB
    static constexpr uint32_t UB_MM_RES_BUFCNT = (dBaseSize > 128) ? 2U : 4U;
    static constexpr uint32_t UB_MM_RES_BUF_BYTES =
        mBaseSize / CV_RATIO * (s2BaseSize > dVBaseSize ? s2BaseSize : dVBaseSize) * sizeof(T);
    LocalTensor<uint8_t> ubMmResBuffersDn_;
    uint32_t mmResBufIdDn_ = 0;

    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    LocalTensor<T> ubVec2ResDn_; // 存放vec2阶段VEC的中间处理结果, 并且作为attn_out的输出buffer, 需配对的MTE3和V的同步ID

    static constexpr uint32_t UB_VEC1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = (mBaseSize / CV_RATIO + 1U) * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> ubVec1ResBuffersDn_;
    uint32_t vec1ResUbBufIdDn_ = 0;

    static constexpr uint32_t UB_SOFTMAX_MAX_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_MAX_BUF_BYTES = 256U;
    LocalTensor<T> softmaxSumBufDn_;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUF_BYTES = 256U;
    LocalTensor<T> softmaxMaxBufDn_;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUF_BYTES = 256U;
    LocalTensor<T> softmaxExpBufDn_;

    static constexpr uint32_t UB_LSE_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES = 2048U;
    LocalTensor<uint8_t> ubLseOutBuffersDn_;
    uint32_t lseOutUbBufIdDn_ = 0;

    const ConstInfo_t &constInfoDn_;

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensToolDn_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensToolDn_;

    // GM
    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensorDn_;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGmDn_;
    GlobalTensor<OUTPUT_T> attentionOutGmDn_;
    GlobalTensor<float> softmaxLseGmDn_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxFDSumGm_;
    GlobalTensor<float> softmaxFDMaxGm_;
    GlobalTensor<float> sinkGmDn_;

    T negativeFloatScalarDn_;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecDn(ConstInfo_t &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                             SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfoDn_(constInfo),
          qSeqLensToolDn_(qSeqLensTool),
          kvSeqLensToolDn_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                     __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalarDn_ = *((T *)&tmp1);

        this->attentionOutGmDn_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        InitAttenOutBufferDn(constInfoDn_.bSize, constInfoDn_.n2Size, constInfoDn_.gSize, constInfoDn_.s1Size,
                             constInfoDn_.dSizeV, outGmTensorDn_, attentionOut);

        if (constInfoDn_.isSoftmaxLseEnable) {
            softmaxLseGmDn_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if (constInfoDn_.enableFlashDecode) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfoDn_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfoDn_.accumOutSize +
                                            constInfoDn_.logSumExpSize);
        }
        if (constInfoDn_.learnableSinkFlag) {
            sinkGmDn_.SetGlobalBuffer((__gm__ float *)learnableSink);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // l1P 三缓冲
        uint32_t addrL1 = 0;
        l1PBuffersDn_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMmResBuffersDn_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                                 UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES); // CV通信BUF
        addrUb = UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES;
        ubVec2ResDn_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_VEC2_RES_BUF_BYTES)
                           .template ReinterpretCast<T>(); // 输出BUF: attn_out拷出
        addrUb += UB_VEC2_RES_BUF_BYTES;
        ubVec1ResBuffersDn_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES); // 2 * 32.25K = 64.5K, 输出BUF: softmax结果拷贝至L1
        addrUb += UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES;

        // softmaxSum×3 + softmaxMax×3 + softmaxExp×3，各 256 bytes
        softmaxSumBufDn_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                                UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES)
                               .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES;
        softmaxMaxBufDn_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                                UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES)
                               .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES;
        softmaxExpBufDn_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                                UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES)
                               .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES;

        ubLseOutBuffersDn_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_LSE_OUT_BUFCNT *
                UB_LSE_OUT_BUF_BYTES); // 2 * 2K = 4K, 输出BUF: FD中间结果SUM和MAX拷出至GM，或者LSE结果拷出
        addrUb += UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES;
    }

    __aicore__ inline void ResetSoftmaxBufferDn(uint32_t slotIdx, const RunInfo &dnRunInfo)
    {
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T);
        LocalTensor<T> sumUb = softmaxSumBufDn_[slotIdx * softmaxBufElementCount];
        LocalTensor<T> maxUb = softmaxMaxBufDn_[slotIdx * softmaxBufElementCount];
        if (constInfoDn_.learnableSinkFlag && dnRunInfo.isFirstFdBlock) {
            Duplicate<T>(sumUb, static_cast<T>(1), softmaxBufElementCount);

            uint32_t gs1Start = dnRunInfo.gS1Idx + dnRunInfo.vecMbaseIdx;

            Mutex::Lock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufIdDn_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffersDn_[vec1ResUbBufIdDn_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();
                if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
                    AttentionCommon::SinkCopyInS1G(sinkTmpUb, sinkGmDn_, gs1Start, dnRunInfo.actVecMSize,
                                                   dnRunInfo.actS1Size, dnRunInfo.n2Idx, constInfoDn_.gSize);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    AttentionCommon::SinkCopyInGS1(sinkTmpUb, sinkGmDn_, gs1Start, dnRunInfo.actVecMSize,
                                                   dnRunInfo.actS1Size, dnRunInfo.n2Idx, constInfoDn_.gSize);
                }
            }
            Mutex::Unlock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufIdDn_);

            Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufIdDn_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffersDn_[vec1ResUbBufIdDn_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();

                if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    AttentionCommon::SinkExpandMaxVf<T, float, true>(maxUb, sinkTmpUb, gs1Start, dnRunInfo.actVecMSize,
                                                                     dnRunInfo.actS1Size, constInfoDn_.gSize);
                } else {
                    AttentionCommon::SinkExpandMaxVf<T, float, false>(maxUb, sinkTmpUb, gs1Start, dnRunInfo.actVecMSize,
                                                                      dnRunInfo.actS1Size, constInfoDn_.gSize);
                }
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufIdDn_);
        } else {
            Duplicate<T>(sumUb, static_cast<T>(0), softmaxBufElementCount);
            Duplicate<T>(maxUb, static_cast<T>(-std::numeric_limits<float>::infinity()), softmaxBufElementCount);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_1);
        if constexpr (dBaseSize <= 128) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_2);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_3);
        }
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline void ProcessVec1(RunInfo dnRunInfo)
    {
        uint32_t mmResUbBufId = mmResBufIdDn_;
        mmResBufIdDn_ = (mmResBufIdDn_ + 1) % UB_MM_RES_BUFCNT;
        uint32_t pL1BufId = dnRunInfo.loop % L1_P_BUFCNT;
        uint32_t mmSyncIdx = CC_MM_0 + mmResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffersDn_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMmResBuffersDn_[mmResUbBufId * UB_MM_RES_BUF_BYTES].template ReinterpretCast<T>();

        if (unlikely(dnRunInfo.isFirstS2Loop)) {
            ResetSoftmaxBufferDn(dnRunInfo.mloop % UB_SOFTMAX_SUM_BUFCNT, dnRunInfo);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        ProcessVec1Dn(pL1Tensor, mm1ResUbTensor, dnRunInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 通知BMM2: Vec1已读完mmRes, 可覆写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx);
        Vec1PostProcessDn(dnRunInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        if (constInfoDn_.needInitOutput) {
            uint32_t vecCoreNum = 2 * constInfoDn_.coreNum;
            uint64_t tSize = constInfoDn_.bSize * constInfoDn_.s1Size;
            if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                tSize = qSeqLensToolDn_.cuSeqLensParser.GetTSize();
            }
            uint64_t attenOutTotalSize = tSize * constInfoDn_.n2Size * constInfoDn_.gSize * constInfoDn_.dSizeV;

            static constexpr OUTPUT_T ATTEN_OUT_INIT_VAL = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(OUTPUT_T);
            AttentionCommon::InitOutput<OUTPUT_T, EVENT_ID0, ATTEN_OUT_POP_BUF_START_ADDR, ATTEN_OUT_POP_BUF_ELE_SIZE,
                                        true>(attentionOutGmDn_, attenOutTotalSize, vecCoreNum, ATTEN_OUT_INIT_VAL);

            if (constInfoDn_.isSoftmaxLseEnable) {
                uint64_t lseTotalSize = tSize * constInfoDn_.n2Size * constInfoDn_.gSize;

                static constexpr float LSE_INIT_VAL = 3e+99;
                static constexpr uint32_t LSE_POP_BUF_START_ADDR = BUFFER_SIZE_BYTE_32K;
                static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(float);
                AttentionCommon::InitOutput<float, EVENT_ID1, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, true>(
                    softmaxLseGmDn_, lseTotalSize, vecCoreNum, LSE_INIT_VAL);
            }

            SyncAll();
        }
    }

    __aicore__ inline void InitAttenOutBufferDn(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                                uint32_t headDim, FaGmTensorOut &outGmTensor, __gm__ uint8_t *gm)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUTPUT_T *)gm);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim,
                                              qSeqLensToolDn_.seqUsedParser);
        } else {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensToolDn_.cuSeqLensParser);
        }
    }

    __aicore__ inline void SoftmaxDataCopyOutDn(RunInfo dnRunInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (constInfoDn_.enableFlashDecode) {
            if (dnRunInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGmDn(dnRunInfo, sumUb, maxUb);
            }
        }

        if (constInfoDn_.enableFlashDecode) {
            if (!dnRunInfo.isS2SplitCore && constInfoDn_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOutDn(sumUb, maxUb, dnRunInfo);
            }
        } else {
            if (constInfoDn_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOutDn(sumUb, maxUb, dnRunInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOutDn(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                               RunInfo &dnRunInfo)
    {
        if (unlikely(dnRunInfo.actVecMSize == 0)) {
            return;
        }

        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        uint32_t vecMIdx = dnRunInfo.gS1Idx + dnRunInfo.vecMbaseIdx;
        LocalTensor<float> lseUb =
            ubLseOutBuffersDn_[lseOutUbBufIdDn_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, dnRunInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
            uint32_t prefixBS1 = qSeqLensToolDn_.cuSeqLensParser.GetTBase(dnRunInfo.bIdx);
            uint64_t bN2Offset = dnRunInfo.n2Idx * constInfoDn_.gSize * constInfoDn_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfo_t>(softmaxLseGmDn_, lseUb, bN2Offset, vecMIdx,
                                                            dnRunInfo.actVecMSize, constInfoDn_);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
            uint64_t bN2Offset = dnRunInfo.bIdx * constInfoDn_.n2Size * constInfoDn_.gSize * constInfoDn_.s1Size +
                                 dnRunInfo.n2Idx * constInfoDn_.gSize * constInfoDn_.s1Size;
            uint64_t qActSeqLens = qSeqLensToolDn_.seqUsedParser.GetActualSeqLength(dnRunInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfo_t>(softmaxLseGmDn_, lseUb, bN2Offset, vecMIdx,
                                                         dnRunInfo.actVecMSize, constInfoDn_);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
            uint64_t bN2Offset = dnRunInfo.bIdx * constInfoDn_.n2Size * constInfoDn_.gSize * constInfoDn_.s1Size +
                                 dnRunInfo.n2Idx * constInfoDn_.gSize * constInfoDn_.s1Size;
            uint64_t qActSeqLens = qSeqLensToolDn_.seqUsedParser.GetActualSeqLength(dnRunInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfo_t>(softmaxLseGmDn_, lseUb, bN2Offset, vecMIdx,
                                                         dnRunInfo.actVecMSize, constInfoDn_, qActSeqLens);
        }
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        lseOutUbBufIdDn_ = (lseOutUbBufIdDn_ + 1) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ProcessVec1Dn(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor,
                                         RunInfo dnRunInfo)
    {
        if (unlikely(dnRunInfo.actVecMSize == 0)) {
            return;
        }

        static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
        static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
        static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
        static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
        static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;

        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<T> sumUb =
            softmaxSumBufDn_[(dnRunInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBufDn_[(dnRunInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb =
            softmaxExpBufDn_[(dnRunInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufIdDn_);

        float descaleQK = 1.0;

        LocalTensor<INPUT_T> stage1CastTensor =
            ubVec1ResBuffersDn_[vec1ResUbBufIdDn_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, nullptr, attenMaskUb, dnRunInfo.actMSizeAlign32 >> 1,
            dnRunInfo.actSingleLoopS2SizeAlign, dnRunInfo.actSingleLoopS2Size, static_cast<T>(constInfoDn_.scaleValue),
            descaleQK, negativeFloatScalarDn_, 0.0F, false);

        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufIdDn_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufIdDn_);
        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Tensor;

        if (dnRunInfo.actSingleLoopS2Size > vec1S2CopyLenDn) {
            DataCopy(mm2AL1Tensor[constInfoDn_.subBlockIdx * vec1HalfS1BaseSize * dnRunInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, vec1S2CopyLenDn, 1,
                      static_cast<uint16_t>(dnRunInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn)});
            DataCopy(mm2AL1Tensor[constInfoDn_.subBlockIdx * vec1HalfS1BaseSize * dnRunInfo.actSingleLoopS2SizeAlign +
                                  vec1S2strideDn],
                     stage1CastTensor[vec1ResOffsetDn],
                     {vec1S2CopyCountDn, static_cast<uint16_t>(dnRunInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn),
                      static_cast<uint16_t>(s2BaseSize - dnRunInfo.actSingleLoopS2SizeAlign + 1), vec1S2CopyLenDn});
        } else {
            DataCopy(mm2AL1Tensor[constInfoDn_.subBlockIdx * vec1HalfS1BaseSize * dnRunInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, static_cast<uint16_t>(dnRunInfo.actSingleLoopS2SizeAlign),
                      static_cast<uint16_t>(vec1S2CopyLenDn - dnRunInfo.actSingleLoopS2SizeAlign + 1), 0});
        }

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufIdDn_);
        vec1ResUbBufIdDn_ = (vec1ResUbBufIdDn_ + 1U) % UB_VEC1_RES_BUFCNT;
    }

    __aicore__ inline void Vec1PostProcessDn(RunInfo dnRunInfo)
    {
        LocalTensor<T> sumUb =
            softmaxSumBufDn_[(dnRunInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBufDn_[(dnRunInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];

        if (unlikely(dnRunInfo.isLastS2Loop)) {
            SoftmaxDataCopyOutDn(dnRunInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTransDn(const RunInfo &dnInfo, LocalTensor<OUTPUT_T> &attenOutUb,
                                                  uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoord gmCoord{.bIdx = dnInfo.bIdx,
                        .n2Idx = dnInfo.n2Idx,
                        .gS1Idx = dnInfo.gS1Idx + dnInfo.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfoDn_.dSizeV};
        copyAttenOutUbToGmDn_(outGmTensorDn_, ubTensor, gmCoord);
    }

    __aicore__ inline void BroadCastAndCopyOutDn(const RunInfo &dnRunInfo, LocalTensor<float> &sumUb,
                                                 LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        LocalTensor<float> sumBrdcstBuf =
            ubLseOutBuffersDn_[lseOutUbBufIdDn_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        FaVectorApi::BroadcastMaxSum(sumBrdcstBuf, sumUb, dnRunInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        DataCopy(softmaxFDSumGm_[gmOffset], sumBrdcstBuf, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        lseOutUbBufIdDn_ = (lseOutUbBufIdDn_ + 1U) % UB_LSE_OUT_BUFCNT;

        LocalTensor<float> maxBrdcstBuf =
            ubLseOutBuffersDn_[lseOutUbBufIdDn_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        FaVectorApi::BroadcastMaxSum(maxBrdcstBuf, maxUb, dnRunInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxBrdcstBuf, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufIdDn_);
        lseOutUbBufIdDn_ = (lseOutUbBufIdDn_ + 1U) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGmDn(const RunInfo &dnRunInfo, LocalTensor<float> &sumUb,
                                                         LocalTensor<float> &maxUb)
    {
        if (unlikely(dnRunInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = dnRunInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = dnRunInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + dnRunInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOutDn(dnRunInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOutDn(const RunInfo &dnRunInfo, LocalTensor<T> &ubVec2Res,
                                                 uint32_t mStartVec, uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVBaseSize;
        uint64_t gmOffset = dnRunInfo.faTmpOutWsPos * mBaseSize * constInfoDn_.dSizeV +
                            (dnRunInfo.vecMbaseIdx + mStartVec) * constInfoDn_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfoDn_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfoDn_.dSizeV) / (AttentionCommon::BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm_[gmOffset], ubVec2Res, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(RunInfo dnRunInfo)
    {
        uint32_t mmResUbBufId = mmResBufIdDn_;
        mmResBufIdDn_ = (mmResBufIdDn_ + 1) % UB_MM_RES_BUFCNT;
        uint32_t mmSyncIdx = CC_MM_0 + mmResUbBufId;
        if (unlikely(dnRunInfo.actVecMSize == 0)) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            return;
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        {
            Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            LocalTensor<T> mm2ResUbTensor =
                ubMmResBuffersDn_[mmResUbBufId * UB_MM_RES_BUF_BYTES].template ReinterpretCast<T>();
            if (unlikely(dnRunInfo.isFirstS2Loop)) {
                uint32_t vec2CalcSize = dnRunInfo.actVecMSize * dTemplateAlign64;
                DataCopy(ubVec2ResDn_, mm2ResUbTensor, vec2CalcSize);
            } else {
                LocalTensor<T> expUb =
                    softmaxExpBufDn_[(dnRunInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
                LocalTensor<T> pScaleUb;

                float deSCalePreVValue = 1.0f;
                if (!dnRunInfo.isLastS2Loop) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        ubVec2ResDn_, mm2ResUbTensor, ubVec2ResDn_, expUb, pScaleUb, dnRunInfo.actVecMSize,
                        dTemplateAlign64, 1.0, 1.0);
                } else {
                    LocalTensor<float> sumUb = softmaxSumBufDn_[(dnRunInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) *
                                                                (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        ubVec2ResDn_, mm2ResUbTensor, ubVec2ResDn_, expUb, pScaleUb, sumUb, dnRunInfo.actVecMSize,
                        dTemplateAlign64, 1.0, 1.0);
                }
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 通知下个BMM1: Vec2已读完mmRes, slot空闲

        if (dnRunInfo.isLastS2Loop) {
            if (unlikely(dnRunInfo.isFirstS2Loop)) {
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                LocalTensor<float> sumUb = softmaxSumBufDn_[(dnRunInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) *
                                                            (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    ubVec2ResDn_, ubVec2ResDn_, sumUb, dnRunInfo.actVecMSize, (uint16_t)dTemplateAlign64, 0.0F);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            }
            uint32_t mStartVec = 0;
            uint32_t mDealSize = dnRunInfo.actVecMSize;
            if (constInfoDn_.enableFlashDecode && dnRunInfo.isS2SplitCore) {
                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2ResForFDCopyOutDn(dnRunInfo, ubVec2ResDn_, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            } else {
                LocalTensor<OUTPUT_T> attenOut;
                int64_t dSizeAligned64 = (int64_t)dVBaseSize;

                attenOut.SetAddr(ubVec2ResDn_.address_);
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                Cast(attenOut, ubVec2ResDn_, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);

                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2DataCopyOutTransDn(dnRunInfo, attenOut, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            }
        }
    }
};

// AIC/AIV 分编译占位（Mix kernel 在 AIC 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockVecDummyDn {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;

    __aicore__ inline FANoQuantGqaBlockVecDummyDn(ConstInfo_t &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                  SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace FlashAttnKernel
#endif // FLASH_ATTN_BLOCK_VEC_DN_H_
