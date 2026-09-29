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
 * \file quant_flash_attn_block_vec_mxfp8_nd_arch92.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_ND_ARCH92_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_ND_ARCH92_H_

#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch92/vf/vf_mul_sel_softmaxflashv2_cast_nz_mxfp8_arch92.h"
#include "../../../common/op_kernel/arch92/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn_mxfp8_arch92.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../arch35/memory_copy_arch35_quant_flash_attn.h"
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "quant_flash_attn_common_def_arch92.h"
#include "../../../common/op_kernel/init_output.h"

#include "kernel_operator.h"
#include "adv_api/activation/softmax.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

// /* ============确定bmm2ResBuffer的类型============= */

namespace BaseApi {

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false, bool isDAligned = true>
class QuantFlashAttnBlockVecMxfp8Nd {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize;
    static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = mBaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = mBaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = mBaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = mBaseSize + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2

    static constexpr uint32_t s2SplitSize = s2BaseSize >> 1;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr MaskFormat MASK_LAYOUT = MaskFormat::SG;

    static constexpr UbFormat OUT_UB_FORMAT = GetOutUbFormat<layout>();
    using OutGmCoord = GmCoordGs1Merge;

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar_ = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using attenMaskGmType = typename std::conditional<HAS_MASK, GlobalTensor<uint8_t>, int8_t>::type;
    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;

    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;
    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<half> attentionOutInitGm_;
    GlobalTensor<float> softmaxLseGm_;

    GlobalTensor<int32_t> actualSeqLengthsGmQ_;
    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND), ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;
    QSeqParserType* qActSeqLensParser_ = nullptr;

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType& qParser)
    {
        this->qActSeqLensParser_ = &qParser;
    }

    attenMaskGmType attenMaskGmInt_;

    postQuantGmType postQuantScaleGm_;
    postQuantGmType postQuantOffsetGm_;
    postQuantBf16GmType postQuantScaleBf16Gm_;
    postQuantBf16GmType postQuantOffsetBf16Gm_;
    quantGmType pScaleGm_;

    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    // ub
    static constexpr uint32_t mm1ResultSize = mBaseSize * s2SplitSize;
    static constexpr uint32_t mm2ResultSize = mBaseSize * dTemplateAlign64;
    static constexpr uint32_t stage1OutSize = (mBaseSize + 1) * s2SplitSize;
    static constexpr uint32_t stage2OutSize = (mBaseSize + 1) * dTemplateAlign64;
    static constexpr uint32_t maskSize = mBaseSize * s2BaseSize;

    LocalTensor<fp8_e4m3fn_t> stage1OutQue;
    LocalTensor<T> stage2OutBuf;
    LocalTensor<fp8_e8m0_t> pScaleSubLoop0Que;
    LocalTensor<uint8_t> attenMaskUB;
    LocalTensor<float> softmaxSumUB;
    LocalTensor<float> softmaxMaxUB;
    LocalTensor<float> softmaxExpUB;
    LocalTensor<float> preLoopMaxUB;
    LocalTensor<float> preLoopSumUB;
    // /* 用来做Broadcast[S1,1]->[S1,8]的临时UB区域 */
    LocalTensor<float> maxBrdcstUB;
    LocalTensor<float> sumBrdcstUB;

    LocalTensor<float> softmaxLseUB;

    LocalTensor<float> firstLoopSumUB;
    LocalTensor<uint8_t> vselrIndexesUB;
    LocalTensor<uint8_t> commonUB;

    // eventID
    static constexpr uint32_t MTE3_V_EVENT0 = EVENT_ID0; // mm2Res
    static constexpr uint32_t MTE3_V_EVENT1 = EVENT_ID1; // lseOut
    static constexpr uint32_t MTE3_V_EVENT2 = EVENT_ID2; // stage1OutQue
    static constexpr uint32_t MTE3_V_EVENT3 = EVENT_ID3; // stage1OutQue
    static constexpr uint32_t MTE3_V_EVENT4 = EVENT_ID4; // pScale
    static constexpr uint32_t MTE3_V_EVENT5 = EVENT_ID5; // sumOut
    static constexpr uint32_t MTE3_V_EVENT6 = EVENT_ID6; // maxOut

    static constexpr uint32_t V_MTE3_EVENT0 = EVENT_ID0; // stage2OutBuf
    static constexpr uint32_t V_MTE3_EVENT1 = EVENT_ID1; // lseOut
    static constexpr uint32_t V_MTE3_EVENT2 = EVENT_ID2; // stage1OutQue
    static constexpr uint32_t V_MTE3_EVENT3 = EVENT_ID3; // stage1OutQue
    static constexpr uint32_t V_MTE3_EVENT4 = EVENT_ID4; // pScale
    static constexpr uint32_t V_MTE3_EVENT5 = EVENT_ID5; // sumOut
    static constexpr uint32_t V_MTE3_EVENT6 = EVENT_ID6; // maxOut

    static constexpr uint32_t V_MTE2_EVENT0 = EVENT_ID0; // attenmask

    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {7, 9};
    static constexpr uint16_t CROSS_CORE_SYNC_C1_V1[2] = {6, 8};
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_P_C2[3] = {1, 2, 3};

    const ConstInfoX& constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP32_LN2};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecMxfp8Nd(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* pScale, __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32_LN2;
        this->negativeFloatScalar_ = *((T*)&tmp1);
        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, attenMask, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* pScale, __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }

        uint64_t actualLenQSize =
            (layout == LayOutTypeEnum::LAYOUT_TND) ? constInfo_.cuSeqLensQSize : constInfo_.seqUsedQSize;
        actualSeqLengthsGmQ_.SetGlobalBuffer((__gm__ int32_t*)actualSeqQlenAddr, actualLenQSize);

        if constexpr (HAS_MASK) {
            attenMaskGmInt_.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
        }
        if constexpr (isFp8) {
            if (pScale != nullptr) {
                pScaleGm_.SetGlobalBuffer((__gm__ float*)pScale);
                pScaleValue_ = this->pScaleGm_.GetValue(0);
            }
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm_.SetGlobalBuffer((__gm__ float*)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
    }

    __aicore__ inline void ProcessVec1(LocalTensor<fp8_e4m3fn_t> pTensor, LocalTensor<fp8_e8m0_t> pScaleTensor,
                                       LocalTensor<T> mm1Res, RunInfoX runInfo, uint32_t subLoop)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        ProcessVec1Nd(pTensor, pScaleTensor, mm1Res, runInfo, subLoop);
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            // SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore();
            if (constInfo_.isSoftmaxLseEnable) {
                SyncAll();
                InitLseOutputSingleCore();
            }
            // WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo_.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.dSizeV;
        if constexpr (POST_QUANT) {
            totalOutputSize /= 2;
        }

        static constexpr OUT_T ATTEN_OUT_INIT_VAL = 0;
        static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0;
        static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(OUTPUT_T);

        AttentionCommon::InitOutput<OUT_T, EVENT_ID0, ATTEN_OUT_POP_BUF_START_ADDR, ATTEN_OUT_POP_BUF_ELE_SIZE, true>(
            attentionOutGm_, totalOutputSize, constInfo_.coreNum, ATTEN_OUT_INIT_VAL);
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;

        static constexpr float LSE_INIT_VAL = 3e+99;
        static constexpr uint32_t LSE_POP_BUF_START_ADDR = BUFFER_SIZE_BYTE_32K;
        static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(float);
        AttentionCommon::InitOutput<float, EVENT_ID1, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, true>(
            softmaxLseGm_, totalOutputSize, constInfo_.coreNum, LSE_INIT_VAL);
    }

    // =================================Private Functions=================================

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float>& sumUb, LocalTensor<float>& maxUb)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if constexpr (FLASH_DECODE) {
            if (!runInfo.isS2SplitCore && constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
                                             RunInfoX& runInfo)
    {
        // vec 核按 16 行对齐处理（受 pscale update 影响），往 GM 搬运时满足真实行数即可
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMSize = runInfo.actMSizeAlign16;
        uint32_t gmDealRowCount;
        gmDealRowCount = runInfo.actVecMSize;
        if (gmDealRowCount == 0) {
            return;
        }
        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx; // 121的runInfo.vecMbaseIdx一直都是0
        LocalTensor<float> lseUb = softmaxLseUB;                 // 单buffer
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT1);              // softmaxLseUB AllocTensor

        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT1);  // softmaxLseUB EnQue
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT1); // softmaxLseUB DeQue

        // MXFP8 目前只支持 TND
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                           constInfo_);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
            uint32_t s1Size = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * constInfo_.n2Size * constInfo_.gSize + runInfo.n2Idx * constInfo_.gSize;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                       constInfo_, s1Size);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, s1LeftPaddingSize);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, qActSeqLens, s1LeftPaddingSize);
        }

        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT1); // softmaxLseUB FreeTensor
    }

    __aicore__ inline void ProcessVec1Nd(LocalTensor<fp8_e4m3fn_t> mm2AL1Tensor,
                                         LocalTensor<fp8_e8m0_t> mm2AScaleL1Tensor, LocalTensor<T> mm1Res,
                                         RunInfoX runInfo, uint32_t subLoop)
    {
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;
        uint32_t actVecMSizeAlign =
            runInfo.actMSizeAlign16; // 差异点，S1方向需要向16向上对齐（满足最小分形16*2），原来是actVecMSize

        uint32_t mxfp8s2RealSize = runInfo.actSingleLoopS2Size;
        if (likely(runInfo.actSingleLoopS2Size > s2SplitSize)) {
            mxfp8s2RealSize = ((int32_t)s2SplitSize < (int32_t)(runInfo.actSingleLoopS2Size - subLoop * s2SplitSize)) ?
                                  (int32_t)s2SplitSize :
                                  (int32_t)(runInfo.actSingleLoopS2Size - subLoop * s2SplitSize);
        }

        // TODO:mask修改
        LocalTensor<uint8_t> attenMaskUb = attenMaskUB;
        uint32_t s2VecCalcSize = mxfp8s2RealSize;
        if constexpr (HAS_MASK) {
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0);                          // attemMaskUb AllocTensor
            AttenMaskCopyIn(attenMaskUb, 0, actVecMSizeAlign, runInfo, subLoop); // 全量拷贝
        }

        int64_t stage1Offset = 0;
        stage1Offset = subLoop % DB;

        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + stage1Offset); // stage1OutQue AllocTensor
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT4);                // pScale AllocTensor

        LocalTensor<float> sumUb = softmaxSumUB[runInfo.mloop % (PRELOAD_N + 1) * 64];
        LocalTensor<float> maxUb = softmaxMaxUB[runInfo.mloop % (PRELOAD_N + 1) * 64];
        LocalTensor<float> expUb = softmaxExpUB[runInfo.loop % (PRELOAD_N + 1) * 64];
        LocalTensor<float> preLoopMaxUb = preLoopMaxUB;
        LocalTensor<float> preLoopSumUb = preLoopSumUB;
        LocalTensor<float> firstLoopSumUb = firstLoopSumUB;
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer = commonUB;

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mmRes = mm1Res;
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutQue[stage1Offset * stage1OutSize];
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor = pScaleSubLoop0Que;

        if (unlikely(runInfo.isFirstS2Loop)) {
            // mxfp8 路径下 oriNRange 模板参数不参与实现选择（统一走 GeneralImpl256），
            // 尾块大小由运行期参数 s2VecCalcSize 在 VF 内部处理，与原版保持单分支结构
            ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, false, mBaseSize, s2SplitSize, GT_128_AND_LTE_256, HAS_MASK,
                               PseTypeEnum::PSE_NONE_TYPE, false, false>(
                stage1CastTensor, vselrIndexesUB, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb, pseUb,
                dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                subLoop, actVecMSizeAlign, s2VecCalcSize, pseStride, slopes, posShift,
                static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                deSCaleKValue, pScaleValue_);

            // s2方向，除了第一个块，均需要进行update操作
        } else {
            ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, true, mBaseSize, s2SplitSize, GT_128_AND_LTE_256, HAS_MASK,
                               PseTypeEnum::PSE_NONE_TYPE, false, false>(
                stage1CastTensor, vselrIndexesUB, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb, pseUb,
                dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                subLoop, actVecMSizeAlign, s2VecCalcSize, pseStride, slopes, posShift,
                static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                deSCaleKValue, pScaleValue_);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSS_CORE_SYNC_V1_C1[subLoop % 2]);
        if constexpr (HAS_MASK) {
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0); // attenMaskUb FreeTensor
        }

        // pScale
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT4);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT4);

        // ===================DataCopy to L1 ====================
        uint64_t pScaleDataLen = actVecMSizeAlign * s2SplitSize / MXFP_GROUP_SIZE;
        uint16_t pScaleDstStride = static_cast<uint16_t>(s2SplitSize / MXFP_GROUP_SIZE / 2 - 1);
        uint16_t copyCount = actVecMSizeAlign / 16;
        uint64_t vecOffset = constInfo_.subBlockIdx * pScaleDataLen;
        constexpr uint32_t copyTimes = s2SplitSize / MXFP_GROUP_SIZE / 2;
        uint32_t subLoopCnt = (runInfo.actSingleLoopS2Size + s2SplitSize - 1) / s2SplitSize;
        if (unlikely(runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            for (uint16_t i = 0; i < copyTimes;
                 i++) { // PScale在s2方向的block块大小为32，L1上需满足16x2分形，故重复拷贝 copyTimes 次
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor,
                         {copyCount, 1, 0, pScaleDstStride});
            }
        } else if (unlikely(subLoop % 2 == 1)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                DataCopy(mm2AScaleL1Tensor[(subLoop - 1) * pScaleDataLen + vecOffset + i * 32], pScaleSubLoop0Tensor,
                         {copyCount, 1, 0, pScaleDstStride});
                DataCopy(mm2AScaleL1Tensor[subLoop * pScaleDataLen + vecOffset + i * 32], pScaleSubLoop0Tensor[128],
                         {copyCount, 1, 0, pScaleDstStride});
            }
        } else if (unlikely(subLoop == subLoopCnt - 1)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                DataCopy(mm2AScaleL1Tensor[subLoop * pScaleDataLen + vecOffset + i * 32], pScaleSubLoop0Tensor[128],
                         {copyCount, 1, 0, pScaleDstStride});
            }
        }

        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT4);                 // pScale FreeTensor
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + stage1Offset);  // stage1OutQue EnQue
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + stage1Offset); // stage1OutQue EnQue

        if (likely(runInfo.actVecMSize != 0)) {
            int64_t dstOffset = s2SplitSize * mBaseSize * subLoop;
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor,
                     {s2SplitSize / 32, (uint16_t)actVecMSizeAlign, (uint16_t)(vec1Srcstride - actVecMSizeAlign),
                      (uint16_t)(mBaseSize - actVecMSizeAlign)});
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + stage1Offset); // stage1OutQue FreeTensor subloop

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(LocalTensor<T> bmm2ResBuf, RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
            return;
        }

        uint32_t vecMSize =
            runInfo.actVecMSize; // 这里要传入真实的s1方向的值，用作后续往GM上搬运，不需要对齐，只需要真实的S1方向
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;
        float deSCaleVValue = 1.0f;

        LocalTensor<T> mmRes = bmm2ResBuf;
        LocalTensor<T> vec2ResUb = stage2OutBuf;
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpUB[(runInfo.loop % 3) * 64];
            LocalTensor<T> pScaleUb;
            float deSCalePreVValue = 1.0f;
            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = softmaxSumUB[(runInfo.mloop % 3) * 64];
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, 1.0, 1.0);
            }
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = softmaxSumUB[(runInfo.mloop % 3) * 64];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, vecMSize,
                                                                          (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            uint32_t DealRowCount = runInfo.actVecMSize;
            if (DealRowCount == 0) {
                SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
                return;
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize, DealRowCount);
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX& runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);
        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        } else {
            PostQuant(runInfo, attenOut, vec2ResUb, mStartVec, mDealSize, dSizeAligned64);
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        }
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT0);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT0);
        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize, gmDealRowCount);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfoX& runInfo, LocalTensor<OUTPUT_T>& attenOut,
                                     LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                     int64_t dSizeAligned64)
    {
        return;
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize, gmDealRowCount);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize, gmDealRowCount);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX& runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo_.realGSize;
            s1EndTdx = vecMEndIdx / constInfo_.realGSize;
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
                ret = (s1StartTdx < s1FirstValidToken);
                ret = ret || (s1EndTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
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
            bool batchNeedRowInvalid = ((constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor = softmaxMaxUB[runInfo.mloop % (PRELOAD_N + 1) * 64][mStartVec];
                if constexpr (!POST_QUANT) {
                    RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo_.dSizeV,
                                              static_cast<uint32_t>(dSizeAligned64), minValue_);
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo_.dSizeV, sizeof(float)); // w8后量化后的处理长度
                    RowInvalidUpdateVF<float>(*((LocalTensor<float>*)&vec2ResUb), maxTensor, mDealSize, dSize, dStride,
                                              minValue_);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX& info, LocalTensor<OUTPUT_T>& attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        // mxfp8 colCount 只能为64或者128，与dDealSize相等；dim 非对齐（如 72）时 UB 侧需按非对齐布局拷出
        FaUbTensor<OUTPUT_T, !isDAligned> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        OutGmCoord gmCoord{.bIdx = info.bIdx,
                           .n2Idx = info.realN2Idx,
                           .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                           .dIdx = 0,
                           .gS1DealSize = gmDealRowCount,
                           .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, !isDAligned>& ubTensor, OutGmCoord& gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    // FD场景使用, 暂时未用到
    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                               LocalTensor<float>& maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcstUB;
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5); // sumBrdcstUB AllocTensor
        BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT5);  // sumBrdcstUB EnQue
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT5); // sumBrdcstUB DeQue
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5); // sumBrdcstUB FreeTensor

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcstUB;
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6); // sumBrdcstUB AllocTensor
        BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT6);  // sumBrdcstUB EnQue
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT6); // sumBrdcstUB DeQue
        DataCopy(softmaxFDMaxGm_[gmOffset], maxOutTensor, calculateSize);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6); // sumBrdcstUB FreeTensor
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                                       LocalTensor<float>& maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX& runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT0);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT0);
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (AttentionCommon::BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(LocalTensor<float>& bmm2ResBuf, RunInfoX runInfo)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
        ProcessVec2OnUb(bmm2ResBuf, runInfo);
        return;
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT1);

        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT3);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT4);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6);

        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT1);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT3);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT4);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6);

        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0);
    }

    __aicore__ inline void ReleaseTensors()
    {
        FreeEventID();
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t addrUBStart = (mm1ResultSize * 2 + mm2ResultSize) * sizeof(float);
        stage1OutQue = LocalTensor<fp8_e4m3fn_t>(TPosition::VECCALC, addrUBStart, stage1OutSize * 2);

        addrUBStart += stage1OutSize * 2 * sizeof(fp8_e4m3fn_t);
        softmaxSumUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128 * 3);

        addrUBStart += 128 * 3 * sizeof(float);
        softmaxMaxUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128 * 3);

        addrUBStart += 128 * 3 * sizeof(float);
        softmaxExpUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128 * 3);

        addrUBStart += 128 * 3 * sizeof(float);
        preLoopMaxUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128);

        addrUBStart += 128 * sizeof(float);
        preLoopSumUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128);

        addrUBStart += 128 * sizeof(float);
        firstLoopSumUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128);

        addrUBStart += 128 * sizeof(float);
        commonUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBStart, 1024);

        addrUBStart += 1024 * sizeof(uint8_t);
        stage2OutBuf = LocalTensor<float>(TPosition::VECCALC, addrUBStart, stage2OutSize);

        addrUBStart += stage2OutSize * sizeof(float);
        pScaleSubLoop0Que = LocalTensor<fp8_e8m0_t>(TPosition::VECCALC, addrUBStart, 512);

        addrUBStart += 512 * sizeof(fp8_e8m0_t);
        vselrIndexesUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBStart, 192);

        addrUBStart += 192 * sizeof(uint8_t);
        if constexpr (HAS_MASK) {
            attenMaskUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBStart, maskSize);
            addrUBStart += maskSize * sizeof(uint8_t);
        }

        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128 * 8); // 8个重复lse元素（32B对齐）
            addrUBStart += 128 * 8 * sizeof(float);
        }

        for (int i = 0; i < 128; i++) {
            vselrIndexesUB.SetValue(i, i * 2);
        }

        for (int i = 0; i < 64; i++) {
            vselrIndexesUB[128].SetValue(i, i * 4);
        }

        if constexpr (FLASH_DECODE) {
            maxBrdcstUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128 * 8);
            addrUBStart += 128 * 8 * sizeof(float);
            sumBrdcstUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, 128 * 8);
        }
    }

    template <typename T1>
    __aicore__ inline void AttentionmaskDataCopy_DT(LocalTensor<T1>& attenMaskUb, GlobalTensor<T1>& srcGmAddr,
                                                    MaskInfo& info, uint32_t s1StartIdx, uint32_t s1EndIdx,
                                                    bool isPre = false)
    {
        uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
        uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, isPre);
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
        dataCopyParams.blockLen = s2SplitSize / 32;
        dataCopyParams.srcStride = (info.attenMaskS1Stride - s2SplitSize) / 32;
        dataCopyParams.dstStride = 0;
        DataCopy(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams);
    }

    __aicore__ inline uint64_t ComputeAttenMaskOffsetCompress_DT(MaskInfo& info, uint32_t s1StartIdx)
    {
        int64_t nextToken = 0; // sparse2 本身原点就是左上角
        if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
            nextToken =
                static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
        } else if (info.sparseMode == BAND) {                                          // 4
            nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
        }
        uint64_t offset = 0;
        int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
        uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
        if (delta < 0) {
            offset =
                (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
        } else {
            offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                     info.attenMaskS1Stride; // min(delta, s2inner)
        }
        return offset;
    }

    template <typename T1, uint32_t s2BaseSize>
    __aicore__ inline void AttentionmaskCopyInForSgLayout_DT(LocalTensor<T1>& attenMaskUb, GlobalTensor<T1>& srcGmAddr,
                                                             MaskInfo& info, bool isPre = false)
    {
        uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
        uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
        uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
        uint32_t headGSize = info.gs1dealNum;
        if (s1Count > 1) {
            headGSize = (info.gs1StartIdx % info.gSize == 0) ? 0 : (info.gSize - info.gs1StartIdx % info.gSize);
        }
        uint32_t remainRowCount = info.gs1dealNum - headGSize;
        uint32_t midS1Count = remainRowCount / info.gSize;
        uint32_t tailGSize = remainRowCount % info.gSize;
        uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
        uint32_t attenMaskS2Stride = attenMaskSizeAlign + 32 * info.attenMaskDstStride;

        // ub-head
        if (headGSize > 0) {
            AttentionmaskDataCopy_DT(attenMaskUb, srcGmAddr, info, s1StartIdx, s1StartIdx + 1, isPre);
            s1StartIdx++;
        }

        if (remainRowCount > 0) {
            uint64_t maskOffset = ComputeAttenMaskOffsetCompress_DT(info, s1StartIdx);
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = midS1Count + (tailGSize > 0);
            dataCopyParams.blockLen = s2SplitSize / 32;
            dataCopyParams.srcStride = (info.attenMaskS1Stride - s2SplitSize) / 32;
            dataCopyParams.dstStride = (info.gSize - 1) * attenMaskS2Stride / 32;
            DataCopy(attenMaskUb[headGSize * s2SplitSize], srcGmAddr[maskOffset], dataCopyParams);
        }

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        MaskUbCopyS1G<T1, s2BaseSize>(attenMaskUb, headGSize, info.gSize, tailGSize, midS1Count);
    }

    __aicore__ inline bool IsSkipAttentionmask_MX(MaskInfo& info)
    {
        int32_t s1StartIdx = info.gs1StartIdx / info.gSize;

        int64_t nextToken = 0; // sparse2 本身原点就在左上角
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算Token
        if (static_cast<int64_t>(info.s2StartIdx + s2SplitSize) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            return true;
        }
        return false;
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        if (likely(runInfo.actSingleLoopS2Size > s2SplitSize)) {
            s2RealSize = (s2SplitSize < runInfo.actSingleLoopS2Size - subLoop * s2SplitSize) ?
                             s2SplitSize :
                             runInfo.actSingleLoopS2Size - subLoop * s2SplitSize;
        }

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx + subLoop * s2SplitSize;
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.preToken = constInfo_.preTokens;
        maskInfo.nextToken = constInfo_.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = (constInfo_.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2SplitSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar_;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask_MX(maskInfo);
        if (unlikely(!IsSkipMask)) {
            AttentionmaskCopyInForSgLayout_DT<uint8_t, s2SplitSize>(attenMaskUb, attenMaskGmInt_, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2SplitSize);
        }
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecMxfp8NdDummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecMxfp8NdDummy(ConstInfoX& constInfo){};
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_ND_ARCH92_H_
