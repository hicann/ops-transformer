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
 * \file flash_attention_antiquant_block_vec_base.h
 * \brief
 */
#ifndef FLASH_ATTENTION_ANTIQUANT_GQA_BLOCK_VEC_H_
#define FLASH_ATTENTION_ANTIQUANT_GQA_BLOCK_VEC_H_

#include "kernel_operator.h"
#include "vf/vf_flashupdate.h"
#include "vf/vf_mul_nd2nz.h"
#include "vf/vf_sel_softmaxflashv2_nd.h"
#include "vf/vf_mul_sel_softmaxflashv2_antiquant_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../utils/mixed_quant_flash_attn_utils.h"
#include "../utils/mqfa_attenmask_gs1.h"
#include "../../../common/op_kernel/vector_common.h"
#include "memory_copy_arch35_mixed_quant_flash_attn.h"
#include "mqfa_init_output.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_vector_gs1;

namespace BaseApi {
template <typename MQFA_T>
class VecBlockBase {
public:
    using ConstInfoX = ConstInfo_t<FiaKernelType::ANTI_QUANT>;
    using Q_T = typename MQFA_T::qType;
    using OUT_T = typename MQFA_T::outType;
    using KV_SCALE_T = typename MQFA_T::kvScaleType;
    using T = float;
    using MM_OUT_T = T;
    static constexpr LayOutTypeEnum outLayout = MQFA_T::layoutOut;
    static constexpr uint32_t mBaseSize = (uint32_t)MQFA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)MQFA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)MQFA_T::dBaseSize;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dBaseSize);
    static constexpr LayOutTypeEnum layout = MQFA_T::layoutQ;
    static constexpr uint8_t QUANT_COMPUTE_MODE = MQFA_T::quantComputeMode;
    static constexpr bool PAGE_ATTENTION = MQFA_T::pageAttention;
    static constexpr bool HAS_MASK = MQFA_T::hasMask;
    static constexpr uint8_t LAYOUT_KV = MQFA_T::LAYOUT_KV;
    static constexpr bool USE_DN = MQFA_T::useDN;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, LAYOUT_KV>(); // TODO AntiquantCommon::

    static constexpr uint32_t DB = 2U;
    static constexpr uint32_t PRELOAD_N = 2U;

    static constexpr fa_base_vector_gs1::LAYOUT_Q MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH) ? fa_base_vector_gs1::LAYOUT_Q::SG : fa_base_vector_gs1::LAYOUT_Q::GS;

    static constexpr uint32_t VEC1_RES_BYTE = (mBaseSize + 1) * s2BaseSize * sizeof(Q_T); // +1解bank冲突
    static constexpr uint32_t VEC2_RES_BYTE = mBaseSize * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t UB_MM1RES_BUF_NUM =
        (mBaseSize == 48 || mBaseSize == 32) ? 2 : 1;                        // 48x512 2块 32x512 2块 64x512 1块
    static constexpr uint32_t UB_MM2RES_BUF_NUM = (mBaseSize == 48) ? 1 : 2; // 48x512 1块 32/64x512 2块
    static constexpr uint32_t UB_MM2RES_BYTE = mBaseSize * dBaseSize * sizeof(MM_OUT_T); // TODO: mBaseSize+1
    static constexpr uint32_t UB_MM1RES_BYTE = mBaseSize * s2BaseSize * sizeof(MM_OUT_T);

    uint32_t ubBaseAddr = UB_MM1RES_BYTE * UB_MM1RES_BUF_NUM + UB_MM2RES_BYTE * UB_MM2RES_BUF_NUM;

    __aicore__ inline VecBlockBase(ConstInfoX& constInfo){};
};

template <typename MQFA_T>
class FAAntiQuantGqaBlockVec {
public:
    /* =================编译期常量的基本块信息================= */
    using ConstInfoX = ConstInfo_t<FiaKernelType::ANTI_QUANT>;
    using Q_T = typename MQFA_T::qType;
    using OUT_T = typename MQFA_T::outType;
    using KV_SCALE_T = typename MQFA_T::kvScaleType;
    using T = float;
    using MM_OUT_T = T;
    static constexpr LayOutTypeEnum outLayout = MQFA_T::layoutOut;
    static constexpr uint32_t mBaseSize = (uint32_t)MQFA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)MQFA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)MQFA_T::dBaseSize;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dBaseSize);
    static constexpr LayOutTypeEnum layout = MQFA_T::layoutQ;
    static constexpr uint8_t QUANT_COMPUTE_MODE = MQFA_T::quantComputeMode;
    static constexpr bool PAGE_ATTENTION = MQFA_T::pageAttention;
    static constexpr bool HAS_MASK = MQFA_T::hasMask;
    static constexpr uint8_t LAYOUT_KV = MQFA_T::LAYOUT_KV;
    static constexpr bool USE_DN = MQFA_T::useDN;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, LAYOUT_KV>(); // TODO AntiquantCommon::

    static constexpr uint32_t DB = 2U;
    static constexpr uint32_t PRELOAD_N = 2U;
    static constexpr fa_base_vector_gs1::LAYOUT_Q MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH) ? fa_base_vector_gs1::LAYOUT_Q::SG : fa_base_vector_gs1::LAYOUT_Q::GS;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    uint32_t firstS2LoopCount = 0U;
    uint32_t lastS2LoopCount = 0U;

    static constexpr float descaleQK = 1.0f;

    using attenMaskGmType = typename std::conditional<HAS_MASK, GlobalTensor<uint8_t>, int8_t>::type;
    using flashdecodeGmType = GlobalTensor<float>;

    // gm
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<Q_T> sinkGm;

    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qActSeqLensParser; // TODO parser待修改

    attenMaskGmType attenMaskGmInt;
    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    FaGmTensor<Q_T, Q_FORMAT> queryGm;
    FaGmTensor<KV_SCALE_T, GmFormat::BNSD> keyScaleGm;
    FaGmTensor<KV_SCALE_T, GmFormat::BNSD> valueScaleGm;

    CopyQueryGmToUb<Q_T, Q_FORMAT> copyQueryGmToUb;
    CopyAntiquantGmToUb<KV_SCALE_T, GmFormat::BNSD> copyAntiquantGmToUb;
    // ub
    LocalTensor<uint8_t> softmaxApiBuf;
    LocalTensor<uint8_t> vec1ResBuf;
    LocalTensor<uint8_t> attenMaskInBuf;
    LocalTensor<uint8_t> keyScaleBuf;
    LocalTensor<uint8_t> valueScaleBuf;
    LocalTensor<uint8_t> vec2ResBuf;
    LocalTensor<uint8_t> softmaxMaxBuf;
    LocalTensor<uint8_t> softmaxSumBuf;
    LocalTensor<uint8_t> softmaxExpBuf;
    LocalTensor<uint8_t> mm2InBuf;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    LocalTensor<uint8_t> maxBrdcstBuf;
    LocalTensor<uint8_t> sumBrdcstBuf;
    LocalTensor<uint8_t> lseTmpBuff;
    LocalTensor<uint8_t> softmaxLseBuf;
    LocalTensor<uint8_t> FDResOutputQue;
    LocalTensor<uint8_t> accumOutInputQue;
    LocalTensor<uint8_t> sinkBuf; // AttentionSink

    static constexpr TEventID V_MTE3_QTMP_0 = static_cast<TEventID>(0);
    static constexpr TEventID V_MTE3_QTMP_1 = static_cast<TEventID>(1);
    static constexpr TEventID V_MTE3_P_0 = static_cast<TEventID>(0);
    static constexpr TEventID V_MTE3_P_1 = static_cast<TEventID>(1);
    static constexpr TEventID V_MTE3_LSE_0 = static_cast<TEventID>(2);
    static constexpr TEventID V_MTE3_LSE_1 = static_cast<TEventID>(3);
    static constexpr TEventID V_MTE3_MM2RES_0 = static_cast<TEventID>(4);
    static constexpr TEventID V_MTE3_MM2RES_1 = static_cast<TEventID>(5);
    static constexpr TEventID V_MTE3_INIT = static_cast<TEventID>(5);
    static constexpr TEventID MTE3_V_LSE = static_cast<TEventID>(0);
    static constexpr TEventID MTE3_V_VEC2RES = static_cast<TEventID>(1);
    static constexpr TEventID MTE3_V_VEC1RES_0 = static_cast<TEventID>(3);
    static constexpr TEventID MTE3_V_VEC1RES_1 = static_cast<TEventID>(4);
    static constexpr TEventID MTE3_V_INIT = static_cast<TEventID>(5);
    static constexpr TEventID MTE2_V_KSCALE = static_cast<TEventID>(0);
    static constexpr TEventID MTE2_V_VSCALE = static_cast<TEventID>(1);
    static constexpr TEventID MTE2_V_MASK = static_cast<TEventID>(2);
    static constexpr TEventID V_MTE2_MASK = static_cast<TEventID>(2);
    static constexpr TEventID V_2_S = static_cast<TEventID>(0);

    static constexpr uint32_t SOFTMAX_BUF_BYTES = 64 * sizeof(T);
    static constexpr uint32_t BRDCST_SIZE = 8;
    static constexpr uint32_t VEC1_RES_BUF_NUM =
        (mBaseSize == 48 || mBaseSize == 32) ? 2 : 1; // 48x512/32x512 2块 64x512 1块
    static constexpr uint32_t UB_MM1RES_BUF_NUM =
        (mBaseSize == 48 || mBaseSize == 32) ? 2 : 1;                        // 48x512/32x512 2块 64x512 1块
    static constexpr uint32_t UB_MM2RES_BUF_NUM = (mBaseSize == 48) ? 1 : 2; // 48x512 1块 32/64x512 2块
    static constexpr uint32_t MASK_BUF_NUM = (mBaseSize == 48) ? 1 : 2;      // 48x512 1块 32/64x512 2块
    static constexpr uint32_t UB_MM2RES_BYTE = mBaseSize * dBaseSize * sizeof(MM_OUT_T); // TODO: mBaseSize+1
    static constexpr uint32_t UB_MM1RES_BYTE = mBaseSize * s2BaseSize * sizeof(MM_OUT_T);
    static constexpr uint32_t SINK_BUF_BYTE = 256;
    static constexpr uint32_t INIT_BUF_BYTE = 8192;
    static constexpr uint32_t VEC1_RES_BYTE = (mBaseSize + 1) * s2BaseSize * sizeof(Q_T); // +1解bank冲突
    static constexpr uint32_t VEC2_RES_BYTE = mBaseSize * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t ATTNMASK_BUF_BYTE = mBaseSize * s2BaseSize;

    uint32_t ubBaseAddr = UB_MM1RES_BYTE * UB_MM1RES_BUF_NUM + UB_MM2RES_BYTE * UB_MM2RES_BUF_NUM;

    const ConstInfoX& constInfo;
    T negativeFloatScalar;
    T positiveFloatScalar;
    uint32_t vec1ResId = 0;
    static constexpr SyncAllConfig SYNC_ALL_CONFIG = {PIPE_MTE3, PIPE_MTE3};
    // ==================== Functions ======================
    __aicore__ inline FAAntiQuantGqaBlockVec(ConstInfoX& constInfo)
        : constInfo(constInfo){};

    __aicore__ inline void InitVecBlock()
    {
        GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
        InitLocalBuffer();
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t* query, __gm__ uint8_t* seqUsedQAddr,
                                        __gm__ uint8_t* seqUsedKvAddr, __gm__ uint8_t* attenMask,
                                        __gm__ uint8_t* learnableSink, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut,
                                        __gm__ uint8_t* workspace) // TODO seqUsedKvAddr没有使用到，是否优化？
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUT_T*)attentionOut);
        softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);

        if (constInfo.seqUsedQSize != 0U) {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t*)seqUsedQAddr, constInfo.seqUsedQSize);
        }

        if constexpr (HAS_MASK) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
        }

        if (learnableSink != nullptr) {
            sinkGm.SetGlobalBuffer((__gm__ Q_T*)learnableSink);
        }

        if (constInfo.isFd) {
            accumOutGm.SetGlobalBuffer((__gm__ float*)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float*)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float*)workspace + constInfo.accumOutSize + constInfo.logSumExpSize);
        }

        if constexpr (QUANT_COMPUTE_MODE == AntiquantMode_FP8_PC) {
            queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T*)query);
            queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                          constInfo.dSize, seqUsedGmQ, constInfo.seqUsedQSize);
        }
        if (constInfo.isSoftmaxLseEnable) {
            qActSeqLensParser.Init(seqUsedGmQ, constInfo.seqUsedQSize, constInfo.s1Size);
        }
    }

    __aicore__ inline void FreeVec()
    {
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_VEC2RES);
        WaitFlag<HardEvent::MTE3_V>(V_MTE3_LSE_0);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_VEC1RES_0);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_VEC1RES_1);
        WaitFlag<HardEvent::V_MTE2>(V_MTE2_MASK);
    }

    __aicore__ inline void InitQuant(__gm__ uint8_t* keyScale, __gm__ uint8_t* valueScale)
    {
        keyScaleGm.gmTensor.SetGlobalBuffer((__gm__ KV_SCALE_T*)keyScale);
        keyScaleGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, 1, constInfo.dSize, seqUsedGmQ,
                                         constInfo.seqUsedQSize);

        valueScaleGm.gmTensor.SetGlobalBuffer((__gm__ KV_SCALE_T*)valueScale);
        valueScaleGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, 1, constInfo.dSize, seqUsedGmQ,
                                           constInfo.seqUsedQSize);
    }

    __aicore__ inline void ClearOutput()
    {
        uint32_t vecCoreNum = constInfo.coreNum;
        uint64_t tSize = constInfo.bSize * constInfo.s1Size;
        // if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        //     tSize = qSeqLensTool.cuSeqLensParser.GetTSize();
        // }
        uint64_t attenOutTotalSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSize;

        static constexpr OUT_T ATTEN_OUT_INIT_VAL = 0;
        static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0;
        static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = INIT_BUF_BYTE / sizeof(OUT_T);
        AttentionCommon::InitOutput<OUT_T, V_MTE3_INIT, ATTEN_OUT_POP_BUF_START_ADDR, ATTEN_OUT_POP_BUF_ELE_SIZE>(
            attentionOutGm, attenOutTotalSize, vecCoreNum, ATTEN_OUT_INIT_VAL);
        if (constInfo.isSoftmaxLseEnable) {
            uint64_t lseTotalSize = tSize * constInfo.n2Size * constInfo.gSize;

            static constexpr float LSE_INIT_VAL = 3e+99;
            static constexpr uint32_t LSE_POP_BUF_START_ADDR = BUFFER_SIZE_BYTE_8K;
            static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_8K / sizeof(float);
            AttentionCommon::InitOutput<float, EVENT_ID1, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, true>(
                softmaxLseGm, lseTotalSize, vecCoreNum, LSE_INIT_VAL);
        }
        // SyncAll<true, SYNC_ALL_CONFIG>();
        SyncAll();
    }

    __aicore__ inline void ProcessVec0(LocalTensor<Q_T>& outputBuf, uint32_t l1QBufId, LocalTensor<Q_T>& qTmpBuf,
                                       RunInfoX runInfo)
    {
        if (runInfo.isFirstS2Loop) {
            uint32_t dbId = MOD2(runInfo.mloop);

            // copyIn query
            FaUbTensor<Q_T> qTmpUbTensor{
                .tensor = qTmpBuf, .rowCount = runInfo.actMSize, .colCount = static_cast<uint32_t>(dBaseSize)};
            GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                    .n2Idx = runInfo.n2Idx,
                                    .gS1Idx = runInfo.gS1Idx,
                                    .dIdx = 0U,
                                    .gS1DealSize = runInfo.actMSize,
                                    .dDealSize = static_cast<uint32_t>(constInfo.dSize)};
            copyQueryGmToUb(qTmpUbTensor, queryGm, gmCoord);

            // copyIn keyScale
            LocalTensor<KV_SCALE_T> keyScaleUb = keyScaleBuf.template ReinterpretCast<KV_SCALE_T>();
            FaUbTensor<KV_SCALE_T> keyScaleUbTensor{.tensor = keyScaleUb,
                                                    .rowCount = static_cast<uint32_t>(constInfo.dSize), // 此配置无效
                                                    .colCount = 1U};
            AntiqGmCoord antiqGmGoord{
                .bIdx = 0U,
                .n2Idx = runInfo.n2Idx,
                .s2Idx = 0U,
                .s2DealSize = runInfo.actSingleLoopS2Size // 在perchannel场景下无效
            };
            copyAntiquantGmToUb(keyScaleUbTensor, keyScaleGm, antiqGmGoord);
            SetFlag<HardEvent::MTE2_V>(MTE2_V_KSCALE + dbId);
            WaitFlag<HardEvent::MTE2_V>(MTE2_V_KSCALE + dbId);

            // query mul keyScale
            uint32_t dSizeAlign = Align((uint32_t)constInfo.dSize, (uint32_t)FA_BYTE_BLOCK);
            uint32_t qDstOffset = mBaseSize * dTemplateAlign64;
            // copyout L1
            MulND2NZ<Q_T, 0>(qTmpBuf[qDstOffset], qTmpBuf, keyScaleUb, mBaseSize, runInfo.actVecMSize,
                             dTemplateAlign64);

            SetFlag<HardEvent::V_MTE3>(V_MTE3_QTMP_0 + dbId);
            WaitFlag<HardEvent::V_MTE3>(V_MTE3_QTMP_0 + dbId);

            DataCopy(outputBuf, qTmpBuf[qDstOffset],
                     {((dTemplateAlign64 * sizeof(Q_T)) >> 5), static_cast<uint16_t>(runInfo.actMSize),
                      static_cast<uint16_t>(mBaseSize + 1U - runInfo.actMSize),
                      static_cast<uint16_t>(((runInfo.actMSize + 15U) >> 4 << 4) - runInfo.actMSize)});

            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1Q_0 + l1QBufId);
        }
    }

    __aicore__ inline void ProcessVec1(LocalTensor<Q_T>& outputBuf, uint32_t l1PBufId,
                                       LocalTensor<MM_OUT_T>& bmm1ResBuf, uint32_t bmm1BufId, RunInfoX runInfo)
    {
        if constexpr (USE_DN) {
            ProcessVec1Dn(outputBuf, l1PBufId, bmm1ResBuf, bmm1BufId, runInfo);
        } else {
            ProcessVec1Nd(outputBuf, l1PBufId, bmm1ResBuf, bmm1BufId, runInfo);
        }
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(LocalTensor<Q_T>& outputBuf, uint32_t l1PBufId,
                                         LocalTensor<MM_OUT_T>& bmm1ResBuf, uint32_t bmm1BufId, RunInfoX runInfo)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(
            CC_BMM1_0 + bmm1BufId); // TODO 后续优化，会阻塞到V流水上，尽量贴近具体使用的地方
        LocalTensor<uint8_t> attenMaskUb = attenMaskInBuf[(runInfo.loop % MASK_BUF_NUM) * ATTNMASK_BUF_BYTE];
        if constexpr (HAS_MASK) {
            AttenMaskCopyIn(attenMaskUb, 0U, runInfo.actVecMSize, MTE2_V_MASK, runInfo);
        }

        uint32_t softmaxBufOffset = (runInfo.mloop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES;
        LocalTensor<T> sumUb = this->softmaxSumBuf[softmaxBufOffset].template ReinterpretCast<T>();
        LocalTensor<T> maxUb = this->softmaxMaxBuf[softmaxBufOffset].template ReinterpretCast<T>();
        LocalTensor<T> expUb =
            this->softmaxExpBuf[(runInfo.loop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES].template ReinterpretCast<T>();
        // todo vec2res单buffer同步id适配
        //  uint32_t stage1Id = MOD2(runInfo.loop);
        uint32_t stage1Id = 0;
        auto vec1ResUb = this->vec1ResBuf[stage1Id * VEC1_RES_BYTE].template ReinterpretCast<Q_T>();

        if (unlikely(runInfo.isFirstS2Loop)) {
            ProcessAntiquantVec1VfDn<T, Q_T, false, HAS_MASK, mBaseSize, s2BaseSize, false>(
                vec1ResUb, sumUb, maxUb, bmm1ResBuf, expUb, nullptr, attenMaskUb, runInfo.actMSizeAlign32,
                runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.scaleValue),
                descaleQK, negativeFloatScalar, 0.0F, HAS_MASK);
        } else {
            ProcessAntiquantVec1VfDn<T, Q_T, true, HAS_MASK, mBaseSize, s2BaseSize, false>(
                vec1ResUb, sumUb, maxUb, bmm1ResBuf, expUb, nullptr, attenMaskUb, runInfo.actMSizeAlign32,
                runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.scaleValue),
                descaleQK, negativeFloatScalar, 0.0F, HAS_MASK);
        }

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_0 + bmm1BufId);
        // TODO 用set/wait替代enque/deque
        //  SetFlag<HardEvent::V_MTE3>(V_MTE3_P_0 + stage1Id);
        //  WaitFlag<HardEvent::V_MTE3>(V_MTE3_P_0 + stage1Id);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + l1PBufId);
        Mm2CopyInAToL1(outputBuf, vec1ResUb, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + l1PBufId);

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void Mm2CopyInAToL1(LocalTensor<Q_T> mm2AL1Tensor, LocalTensor<Q_T> vec1ResUb, RunInfoX runInfo)
    {
        if constexpr (USE_DN) {
            uint16_t blockCount = 128 / mBaseSize; // 多少行拼成一行，128 * sizeof(half) = 256B一个寄存器大小
            uint16_t elementSize = 32 / sizeof(Q_T);      // 一个block元素个数
            uint16_t copyCount = mBaseSize / elementSize; // 一次搬运的数据块
            uint16_t actCopyCount = (runInfo.actVecMSize + elementSize - 1U) / elementSize;

            if constexpr (mBaseSize == 32) {
                actCopyCount = actCopyCount << 1;
                for (uint16_t i = 0; i < actCopyCount; i++) {
                    struct DataCopyParams Bmm2DataCopyParamsForP;
                    Bmm2DataCopyParamsForP.blockCount = (s2BaseSize >> 2); // 1为解bank冲突的
                    Bmm2DataCopyParamsForP.blockLen = 1;
                    Bmm2DataCopyParamsForP.srcStride = 0U;
                    Bmm2DataCopyParamsForP.dstStride = 3U; // 中间空3行，因为4行拼成1行了
                    uint64_t l1Offset = elementSize * i;
                    uint64_t ubOffset = (s2BaseSize / blockCount + 1) * 2 * elementSize * i; // 两个datablock
                    DataCopy(mm2AL1Tensor[l1Offset], vec1ResUb[ubOffset], Bmm2DataCopyParamsForP);
                }
                for (uint16_t i = 0; i < actCopyCount; i++) {
                    struct DataCopyParams Bmm2DataCopyParamsForP;
                    Bmm2DataCopyParamsForP.blockCount = s2BaseSize >> 2;
                    Bmm2DataCopyParamsForP.blockLen = 1;
                    Bmm2DataCopyParamsForP.srcStride = 0U;
                    Bmm2DataCopyParamsForP.dstStride = 3U; // 中间空3行，因为4行拼成1行了
                    uint64_t l1Offset = s2BaseSize * elementSize + elementSize * i;
                    uint64_t ubOffset = (s2BaseSize / blockCount + 1) * elementSize +
                                        (s2BaseSize / blockCount + 1) * 2 * elementSize * i; // 两个datablock
                    DataCopy(mm2AL1Tensor[l1Offset], vec1ResUb[ubOffset], Bmm2DataCopyParamsForP);
                }
            } else {
                struct DataCopyParams Bmm2DataCopyParamsForP;
                Bmm2DataCopyParamsForP.blockCount = blockCount;
                Bmm2DataCopyParamsForP.blockLen = (runInfo.actSingleLoopS2Size + blockCount - 1) / blockCount;
                Bmm2DataCopyParamsForP.srcStride = (s2BaseSize / blockCount) * ((mBaseSize * sizeof(Q_T) >> 5) - 1) +
                                                   (mBaseSize * sizeof(Q_T) >> 5) + s2BaseSize / blockCount -
                                                   Bmm2DataCopyParamsForP.blockLen;
                Bmm2DataCopyParamsForP.dstStride = 0U;
                for (uint16_t i = 0; i < actCopyCount; ++i) {
                    DataCopy(mm2AL1Tensor[runInfo.actSingleLoopS2Size * elementSize * i],
                             vec1ResUb[(s2BaseSize / blockCount + 1) * elementSize * i], Bmm2DataCopyParamsForP);
                }
            }
        } else {
            uint16_t blockCount = ((runInfo.actSingleLoopS2Size * sizeof(Q_T) + 31U) >> 5);
            uint16_t blockLen = runInfo.actVecMSize;
            uint16_t srcStride = (mBaseSize + 1) - runInfo.actVecMSize; // 解bank冲突
            uint16_t dstStride =
                ((runInfo.actVecMSize + 15U) >> 4 << 4) - runInfo.actVecMSize; // mmad需满足16*16故需16对齐
            DataCopy(mm2AL1Tensor, vec1ResUb, {blockCount, blockLen, srcStride, dstStride});
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float>& sumUb, LocalTensor<float>& maxUb)
    {
        if (runInfo.isS2SplitCore) {
            if (constInfo.isFd) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        } else {
            if (constInfo.learnableSinkFlag) {
                this->Vec1SinkComputeGSFused(runInfo, sumUb, maxUb);
            }
            SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
                                             RunInfoX& runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0U)) {
            return;
        }

        if (!constInfo.isSoftmaxLseEnable) {
            return;
        }
        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        WaitFlag<HardEvent::MTE3_V>(V_MTE3_LSE_0);
        LocalTensor<float> lseUb = this->softmaxLseBuf.template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_LSE_0);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_LSE_0);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo, 0U);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo, qActSeqLens, 0U);
        }

        SetFlag<HardEvent::MTE3_V>(V_MTE3_LSE_0);
    }

    __aicore__ inline void Vec1SinkCompute(RunInfoX& runInfo, LocalTensor<float>& sumUb, LocalTensor<float>& maxUb)
    {
        int64_t sinkOffset = runInfo.n2Idx * constInfo.gSize + runInfo.gIdx;
        auto sinkRaw = this->sinkGm.GetValue(sinkOffset);
        float sinkValue;
        if constexpr (IsSameType<decltype(sinkRaw), half>::value) {
            sinkValue = static_cast<float>(sinkRaw);
        } else {
            sinkValue = ToFloat(sinkRaw);
        }
        SinkSubExpAddVF<float>(sumUb, maxUb, sinkValue, runInfo.actVecMSize);
    }

    __aicore__ inline void Vec1SinkComputeGSFused(RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                                  LocalTensor<float>& maxUb)
    {
        // TODO  适配GS1 合轴
        CopySinkIn(runInfo);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_LSE);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_LSE);
        LocalTensor<MM_OUT_T> sinkUb = sinkBuf.template ReinterpretCast<MM_OUT_T>();
        // SinkSubExpAddGSFusedVF<float, Q_T>(sinkUb, sumUb, maxUb, runInfo.actVecMSize);
    }

    __aicore__ inline void CopySinkIn(RunInfoX& runInfo)
    {
        LocalTensor<Q_T> sinkUbBf16 = sinkBuf.template ReinterpretCast<Q_T>();
        int64_t sinkOffset =
            runInfo.n2Idx * constInfo.gSize + constInfo.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize);
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1U;                              // 进行一次连续拷贝
        sinkCopyParams.blockLen = runInfo.actVecMSize * sizeof(Q_T); // 实际需要拷贝的字节数
        sinkCopyParams.srcStride = 0U;                               // 源地址连续
        sinkCopyParams.dstStride = 0U;                               // 目的地址连续

        DataCopyPadExtParams<Q_T> sinkCopyPadParams{};
        DataCopyPad(sinkUbBf16, this->sinkGm[sinkOffset], sinkCopyParams, sinkCopyPadParams);
    }

    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T>& maxUb, uint32_t negativeIntScalar,
                                                   SoftMaxShapeInfo& softmaxShapeInfo)
    {
        SetFlag<HardEvent::V_S>(V_2_S);
        WaitFlag<HardEvent::V_S>(V_2_S);
        bool isUpdateNeedCheck = false;
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
        for (uint32_t i = 0; i < softmaxShapeInfo.srcM; i++) {
            T maxValue = maxUb.GetValue(i);
            uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
            if (checkValue == negativeIntScalar) {
                isUpdateNeedCheck = true;
                break;
            }
        }
        SetMaskNorm();
        ResetMask();
        return isUpdateNeedCheck;
    }

    __aicore__ inline void InvalidLineProcess(RunInfoX runInfo, LocalTensor<T>& sumUb, LocalTensor<T>& maxUb)
    {
        if (constInfo.softMaxCheckRes) {
            SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(runInfo.actVecMSize), static_cast<uint32_t>(1),
                                              static_cast<uint32_t>(runInfo.actVecMSize), static_cast<uint32_t>(1)};
            bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VALUE_FP32, softmaxShapeInfo);
            if (unlikely(runInfo.isLastS2Loop && res)) {
                SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.actVecMSize, this->negativeFloatScalar,
                                    this->positiveFloatScalar);
            }
        }
    }

    __aicore__ inline void ProcessVec1Nd(LocalTensor<Q_T>& outputBuf, uint32_t l1PBufId,
                                         LocalTensor<MM_OUT_T>& bmm1ResBuf, uint32_t bmm1BufId, RunInfoX runInfo)
    {
        bool needMask = false;
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (HAS_MASK) {
            // TODO: 多开同步ID进行mask同步
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_MASK);
            attenMaskUb = attenMaskInBuf[(runInfo.loop % MASK_BUF_NUM) * ATTNMASK_BUF_BYTE];
            needMask = AttenMaskCopyIn(attenMaskUb, 0U, runInfo.actVecMSize, MTE2_V_MASK, runInfo);
        }
        uint32_t softmaxBufOffset = (runInfo.mloop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES;
        LocalTensor<T> sumUb = this->softmaxSumBuf[softmaxBufOffset].template ReinterpretCast<T>();
        LocalTensor<T> maxUb = this->softmaxMaxBuf[softmaxBufOffset].template ReinterpretCast<T>();
        LocalTensor<T> expUb =
            this->softmaxExpBuf[(runInfo.loop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES].template ReinterpretCast<T>();

        LocalTensor<Q_T> pseUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> dropMaskUb;
        LocalTensor<T> pScaleUb;

        constexpr float descaleQK = 1.0f;
        constexpr float deSCaleKValue = 1.0f;
        constexpr float slopes = 0.0f;
        constexpr float posShift = 0.0f;
        constexpr uint32_t pseStride = 0;
        constexpr uint32_t doubleMBaseSize = mBaseSize;

        auto vec1ResUb = vec1ResBuf[vec1ResId * VEC1_RES_BYTE].template ReinterpretCast<Q_T>();

        WaitFlag<HardEvent::MTE3_V>(MTE3_V_VEC1RES_0 + vec1ResId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_0 + bmm1BufId);

        if (runInfo.isFirstS2Loop) {
            if (runInfo.actSingleLoopS2Size == 512 && needMask) {
                ProcessVec1Vf<T, Q_T, Q_T, false, doubleMBaseSize, s2BaseSize, EQ_512, true, PseTypeEnum::PSE_NONE_TYPE,
                              false, false, false>(vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb,
                                                   attenMaskUb, pseUb, dropMaskUb, softmaxApiBuf, pScaleUb,
                                                   runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                                                   posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                                                   negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size == 512 && !needMask) {
                ProcessVec1Vf<T, Q_T, Q_T, false, doubleMBaseSize, s2BaseSize, EQ_512, false,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, queryScaleUb,
                    deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size == 128) {
                ProcessVec1Vf<T, Q_T, Q_T, false, doubleMBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, queryScaleUb,
                    deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, Q_T, Q_T, false, doubleMBaseSize, s2BaseSize, GT_0_AND_LTE_64, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, queryScaleUb,
                    deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, Q_T, Q_T, false, doubleMBaseSize, s2BaseSize, GT_64_AND_LTE_128, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, queryScaleUb,
                    deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 256) {
                ProcessVec1Vf<T, Q_T, Q_T, false, doubleMBaseSize, s2BaseSize, GT_128_AND_LTE_256, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F);
            } else if (runInfo.actSingleLoopS2Size <= 512) {
                ProcessVec1Vf<T, Q_T, Q_T, false, doubleMBaseSize, s2BaseSize, GT_256_AND_LTE_512, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F);
            }
        } else {
            if (runInfo.actSingleLoopS2Size == 512 && needMask) {
                ProcessVec1Vf<T, Q_T, Q_T, true, doubleMBaseSize, s2BaseSize, EQ_512, true, PseTypeEnum::PSE_NONE_TYPE,
                              false, false, false>(vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb,
                                                   attenMaskUb, pseUb, dropMaskUb, softmaxApiBuf, pScaleUb,
                                                   runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                                                   posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                                                   negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size == 512 && !needMask) {
                ProcessVec1Vf<T, Q_T, Q_T, true, doubleMBaseSize, s2BaseSize, EQ_512, false, PseTypeEnum::PSE_NONE_TYPE,
                              false, false, false>(vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb,
                                                   attenMaskUb, pseUb, dropMaskUb, softmaxApiBuf, pScaleUb,
                                                   runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                                                   posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                                                   negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size == 128) {
                ProcessVec1Vf<T, Q_T, Q_T, true, doubleMBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, queryScaleUb,
                    deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, Q_T, Q_T, true, doubleMBaseSize, s2BaseSize, GT_0_AND_LTE_64, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, queryScaleUb,
                    deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, Q_T, Q_T, true, doubleMBaseSize, s2BaseSize, GT_64_AND_LTE_128, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, queryScaleUb,
                    deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 256) {
                ProcessVec1Vf<T, Q_T, Q_T, true, doubleMBaseSize, s2BaseSize, GT_128_AND_LTE_256, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F);
            } else if (runInfo.actSingleLoopS2Size <= 512) {
                ProcessVec1Vf<T, Q_T, Q_T, true, doubleMBaseSize, s2BaseSize, GT_256_AND_LTE_512, HAS_MASK,
                              PseTypeEnum::PSE_NONE_TYPE, false>(
                    vec1ResUb, nullptr, sumUb, maxUb, bmm1ResBuf, expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb,
                    softmaxApiBuf, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F);
            }
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_0 + bmm1BufId);

        if constexpr (HAS_MASK) {
            SetFlag<HardEvent::V_MTE2>(V_MTE2_MASK);
        }

        SetFlag<HardEvent::V_MTE3>(V_MTE3_P_0 + vec1ResId);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_P_0 + vec1ResId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + l1PBufId);
        Mm2CopyInAToL1(outputBuf, vec1ResUb, runInfo);

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + l1PBufId);
        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T, mBaseSize>(sumUb, maxUb, expUb, sumUb, maxUb, softmaxApiBuf, runInfo.actVecMSize);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }

        SetFlag<HardEvent::MTE3_V>(MTE3_V_VEC1RES_0 + vec1ResId);
        vec1ResId = (vec1ResId + 1) % VEC1_RES_BUF_NUM;
    }

    __aicore__ inline void ProcessVec2OnUb(LocalTensor<MM_OUT_T> bmm2ResBuf, uint32_t bmm2BufId, RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0U)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_0 + bmm2BufId);
            return;
        }
        // 只有perchannel时需要用到，perchannel时KV_SCALE_T等于INPUT_T，若修改VF需同步修改
        LocalTensor<Q_T> valueScaleUb;
        if constexpr (QUANT_COMPUTE_MODE == AntiquantMode_FP8_PC) {
            uint32_t dbId = MOD2(runInfo.mloop);
            valueScaleUb = valueScaleBuf.template ReinterpretCast<KV_SCALE_T>();
            // TODO 增加valueScaleUb同步
            FaUbTensor<KV_SCALE_T> valueScaleUbTensor{
                .tensor = valueScaleUb, .rowCount = static_cast<uint32_t>(constInfo.dSize), .colCount = 1U};
            AntiqGmCoord antiqGmGoord{
                .bIdx = 0U, .n2Idx = runInfo.n2Idx, .s2Idx = 0U, .s2DealSize = runInfo.actSingleLoopS2Size};
            copyAntiquantGmToUb(valueScaleUbTensor, valueScaleGm, antiqGmGoord);
            SetFlag<HardEvent::MTE2_V>(MTE2_V_VSCALE + dbId);
            WaitFlag<HardEvent::MTE2_V>(MTE2_V_VSCALE + dbId);
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
        float deSCaleVValue;

        WaitFlag<HardEvent::MTE3_V>(MTE3_V_VEC2RES);

        LocalTensor<T> vec2ResUb = this->vec2ResBuf.template ReinterpretCast<T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, bmm2ResBuf, vec2CalcSize);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<T> expUb =
                softmaxExpBuf[(runInfo.loop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES].template ReinterpretCast<T>();

            constexpr float deSCalePreVValue = 1.0f;
            if (!runInfo.isLastS2Loop) {
                FlashUpdate<T, T, Q_T, T, dTemplateAlign64, QUANT_COMPUTE_MODE == 0>(
                    vec2ResUb, bmm2ResBuf, vec2ResUb, expUb, valueScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                    deSCalePreVValue);
            } else {
                // todo QUANT_COMPUTE_MODE == 0 适用pc场景
                LocalTensor<T> sumUb = this->softmaxSumBuf[(runInfo.mloop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES]
                                           .template ReinterpretCast<T>();
                FlashUpdateLast<T, T, Q_T, T, dTemplateAlign64, QUANT_COMPUTE_MODE == 0>(
                    vec2ResUb, bmm2ResBuf, vec2ResUb, expUb, sumUb, valueScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                    deSCalePreVValue);
            }
        }

        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<T> sumUb = this->softmaxSumBuf[(runInfo.mloop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES]
                                           .template ReinterpretCast<T>();
                FlashUpdateDiv<T, T, Q_T, T, dTemplateAlign64, QUANT_COMPUTE_MODE == 0>(
                    vec2ResUb, bmm2ResBuf, sumUb, valueScaleUb, runInfo.actVecMSize, dTemplateAlign64, deSCaleVValue);
            }
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_0 + bmm2BufId);

        if (runInfo.isLastS2Loop) {
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_V_VEC2RES);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2CastAndCopyOut(RunInfoX& runInfo, LocalTensor<VEC2_RES_T>& vec2ResUb, uint32_t mStartVec,
                                              uint32_t mDealSize)
    {
        LocalTensor<OUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

        attenOut.SetAddr(vec2ResUb.address_);
        int64_t vec2MaxBufOffset = mStartVec;
        LocalTensor<float> maxTensor = softmaxMaxBuf[(runInfo.mloop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES]
                                           .template ReinterpretCast<float>()[vec2MaxBufOffset];
        InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, mDealSize, dSizeAligned64,
                                               this->negativeFloatScalar, 0.0f);

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_MM2RES_0);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_MM2RES_0);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    // mStartVec： [0, runInfo.vecMsize)
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<VEC2_RES_T>& vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if (constInfo.isFd) {
            if (runInfo.isS2SplitCore) {
                Bmm2FDDataCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2CastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2CastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX& runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo.gSize;
            s1EndTdx = vecMEndIdx / constInfo.gSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            // GS1 layout
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                // 只跨1个G
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
                // 跨多个G
                ret = (s1FirstValidToken > 0) || (s1LastValidToken < (runInfo.actS1Size - 1));
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX& runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool needRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (needRowInvalid) {
                LocalTensor<float> maxTensor = softmaxMaxBuf[(runInfo.mloop % (PRELOAD_N + 1)) * SOFTMAX_BUF_BYTES]
                                                   .template ReinterpretCast<float>()[mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX& info, LocalTensor<OUT_T>& attenOutUb, uint32_t vecMIdx,
                                                uint32_t dealRowCount)
    {
        FaUbTensor<OUT_T> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = (uint32_t)(dTemplateAlign64)};
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.n2Idx,
                                .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                                .dIdx = 0U,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo.dSizeV};

        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUT_T>& ubTensor, GmCoordGs1Merge& gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, seqUsedGmQ, constInfo.seqUsedQSize);
            CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, seqUsedGmQ, constInfo.seqUsedQSize);
            CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                               LocalTensor<float>& maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcstBuf.template ReinterpretCast<float>();
        FaVectorApi::BroadcastMaxSumAntiquant(sumOutTensor, sumUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_LSE_0);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_LSE_0);
        DataCopy(softmaxFDSumGm[gmOffset], sumOutTensor, calculateSize);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcstBuf.template ReinterpretCast<float>();
        FaVectorApi::BroadcastMaxSumAntiquant(maxOutTensor, maxUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_LSE_1);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_LSE_1);
        DataCopy(softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                                       LocalTensor<float>& maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0U)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;

        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfoX& runInfo, LocalTensor<MM_OUT_T>& vec2ResUb,
                                             uint32_t mStartVec, uint32_t mDealSize)
    {
        LocalTensor<T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;
        SetFlag<HardEvent::V_MTE3>(V_MTE3_MM2RES_0);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_MM2RES_0);
        attenOut = vec2ResUb;
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfo.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0U;

        DataCopyPad(accumOutGm[gmOffset], attenOut, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(LocalTensor<MM_OUT_T> bmm2ResBuf, uint32_t bmm2BufId, RunInfoX runInfo)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_0 + bmm2BufId);
        ProcessVec2OnUb(bmm2ResBuf, bmm2BufId, runInfo);
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        // TODO TPosition类型待确定
        softmaxApiBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, 512);
        ubBaseAddr += 512;

        constexpr uint32_t SOFTMAX_BUFS_BYTES = (PRELOAD_N + 1) * SOFTMAX_BUF_BYTES;
        softmaxSumBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, SOFTMAX_BUFS_BYTES);
        ubBaseAddr += SOFTMAX_BUFS_BYTES;

        softmaxMaxBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, SOFTMAX_BUFS_BYTES);
        ubBaseAddr += SOFTMAX_BUFS_BYTES;

        softmaxExpBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, SOFTMAX_BUFS_BYTES);
        ubBaseAddr += SOFTMAX_BUFS_BYTES;

        maxBrdcstBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BRDCST_SIZE * SOFTMAX_BUF_BYTES);
        ubBaseAddr += BRDCST_SIZE * SOFTMAX_BUF_BYTES;
        sumBrdcstBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BRDCST_SIZE * SOFTMAX_BUF_BYTES);
        ubBaseAddr += BRDCST_SIZE * SOFTMAX_BUF_BYTES;
    }

    __aicore__ inline void InitLocalBuffer()
    {
        if constexpr (QUANT_COMPUTE_MODE == AntiquantMode_FP8_PC) {
            keyScaleBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, dTemplateAlign64 * sizeof(T));
            ubBaseAddr += dTemplateAlign64 * sizeof(T);
            valueScaleBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, dTemplateAlign64 * sizeof(T));
            ubBaseAddr += dTemplateAlign64 * sizeof(T);
        }

        SoftmaxInitBuffer();

        if constexpr (HAS_MASK) {
            attenMaskInBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, MASK_BUF_NUM * ATTNMASK_BUF_BYTE);
            ubBaseAddr += MASK_BUF_NUM * ATTNMASK_BUF_BYTE;
        }
        vec1ResBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, VEC1_RES_BUF_NUM * VEC1_RES_BYTE);
        ubBaseAddr += VEC1_RES_BUF_NUM * VEC1_RES_BYTE;
        // TODO 老模板待修改
        vec2ResBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, VEC2_RES_BYTE);
        ubBaseAddr += VEC2_RES_BYTE;

        if (constInfo.isSoftmaxLseEnable) {
            // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
            softmaxLseBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, mBaseSize * sizeof(float) * BRDCST_SIZE);
            ubBaseAddr += mBaseSize * sizeof(float) * BRDCST_SIZE;
        }

        if (constInfo.learnableSinkFlag) {
            sinkBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, SINK_BUF_BYTE);
            ubBaseAddr += SINK_BUF_BYTE;
        }

        SetFlag<HardEvent::MTE3_V>(MTE3_V_VEC2RES);
        SetFlag<HardEvent::MTE3_V>(V_MTE3_LSE_0);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_VEC1RES_0);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_VEC1RES_1);
        SetFlag<HardEvent::V_MTE2>(V_MTE2_MASK);
    }

    __aicore__ inline void GetExtremeValue(T& negativeScalar, T& positiveScalar)
    {
        uint16_t tmp1 = NEGATIVE_MIN_VALUE_FP16;
        negativeScalar = *((half*)&tmp1);
    }

    __aicore__ inline bool AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           TEventID MTE2_V_MASK, RunInfoX& runInfo)
    {
        fa_base_vector_gs1::MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<fa_base_vector_gs1::SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1U) ? 0U : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - Align(maskInfo.s2dealNum, 32U)) >> 5;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.layout = MASK_LAYOUT;
        maskInfo.attenMaskType = fa_base_vector_gs1::MaskDataType::MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);
        if (IsSkipMask && IsSkipMaskForPre) {
            if (constInfo.sparseMode == fa_base_vector_gs1::SparseMode::RIGHT_DOWN_CAUSAL &&
                runInfo.actSingleLoopS2Size == 512) {
            } else {
                Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
                PipeBarrier<PIPE_V>();
            }
            return false;
        }

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUb, attenMaskGmInt, maskInfo,
                                                                        MTE2_V_MASK);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            PipeBarrier<PIPE_V>();
        }

        if (!IsSkipMaskForPre) { // TODO SOFTMAX_BUF_BYTES ??
            LocalTensor<uint8_t> attenMaskUbPre =
                this->attenMaskInBuf[(1 - MOD2(runInfo.loop)) * SOFTMAX_BUF_BYTES].template ReinterpretCast<uint8_t>();
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUbPre, attenMaskGmInt, maskInfo,
                                                                        MTE2_V_MASK, true);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
        }
        return true;
    }
};
} // namespace BaseApi
#endif // FLASH_ATTENTION_ANTIQUANT_GQA_BLOCK_VEC_H_
