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
 * \file quant_flash_attn_block_vec_fullquant_fp8.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_FP8_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_FP8_H_
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "memory_copy_arch35_quant_flash_attn.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecGqaFp8 {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;

    static constexpr uint32_t initOutputEventId = 0U;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

    static constexpr bool USE_DN = useDn;
    static constexpr bool IS_PER_TOEKN_HEAD = true;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr GmFormat Q_SCALE_FORMAT = GmFormat::NGT;
    static constexpr GmFormat K_SCALE_FORMAT = GmFormat::PA_BnNBs;
    using pseShiftType = half;

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;
    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // gm
    TPipe *tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<half> attentionOutInitGm_;
    GlobalTensor<float> softmaxLseGm_;

    GlobalTensor<int32_t> actualSeqLengthsGmQ_;
    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD),
                                  ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;
    QSeqParserType *qActSeqLensParser_ = nullptr;

    GlobalTensor<int32_t> blockTableGm_;

    postQuantGmType postQuantScaleGm_;
    postQuantGmType postQuantOffsetGm_;
    postQuantBf16GmType postQuantScaleBf16Gm_;
    postQuantBf16GmType postQuantOffsetBf16Gm_;
    quantGmType pScaleGm_;
    static constexpr bool Q_SCALE_NEEDS_WZH = (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_NT);
    using FaGmTensorQScale = FaGmTensor<float, Q_SCALE_FORMAT, int32_t, Q_SCALE_NEEDS_WZH>;
    using FaGmTensorKScale = FaGmTensor<float, K_SCALE_FORMAT, int32_t>;
    FaGmTensorQScale queryScaleGm_;
    FaGmTensorKScale keyScaleGm_;
    quantGmType deScaleVGm_; // GQA: 1D [N2]的V descale

    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    CopyQueryScaleGmToUb<float, Q_SCALE_FORMAT> copyQueryScaleGmToUb_;
    // ub
    TBuf<> commonTBuf_;
    TQue<QuePosition::VECOUT, 1> stage1OutQue_[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue_[2];
    TBuf<> stage2OutBuf_;
    TEventID mte3ToVId_[2];
    TEventID vToMte3Id_[2];
    TBuf<> softmaxMaxBuf_[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf_[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf_[PRELOAD_N + 1];
    TBuf<> vselrIndexesBuf_[4];
    TQue<QuePosition::VECOUT, 1> maxBrdcst_;
    TQue<QuePosition::VECOUT, 1> sumBrdcst_;

    TBuf<> lseTmpBuff_;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue_;
    TBuf<> queryAntiqScaleInputQue_;
    TBuf<> keyAntiqScaleInputQue_[2]; // 2:DB
    const ConstInfoX &constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    bool isSkipMask_{false};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecGqaFp8(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *pScale,
                                        __gm__ uint8_t *blockTable, __gm__ uint8_t *deqScaleQ,
                                        __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace, QSeqParserType &qParser)
    {
        tPipe_ = pipe;
        this->qActSeqLensParser_ = &qParser;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, blockTable, deqScaleQ, deqScaleK, deqScaleV,
                     softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *pScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *deqScaleQ,
                                        __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        uint64_t actualLenQSize = (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) ?
                                      constInfo_.cuSeqLensQSize :
                                      constInfo_.seqUsedQSize;
        actualSeqLengthsGmQ_.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr, actualLenQSize);

        if (pScale != nullptr) {
            pScaleGm_.SetGlobalBuffer((__gm__ float *)pScale);
            pScaleValue_ = this->pScaleGm_.GetValue(0);
        }
        if (deqScaleV != nullptr) {
            deScaleVGm_.SetGlobalBuffer((__gm__ float *)deqScaleV);
        }
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size,
                         queryScaleGm_, deqScaleQ);
        InitKScaleBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, keyScaleGm_,
                         deqScaleK, constInfo_.kDescaleStrides.bnStride, constInfo_.kDescaleStrides.n2Stride);
        if constexpr (FLASH_DECODE) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            FaGmTensorQScale &qScaleGmTensor, __gm__ uint8_t *gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float *)gm);
        if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_NT) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, *this->qActSeqLensParser_);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, FaGmTensorKScale &kScaleGmTensor,
                                            __gm__ uint8_t *gm, uint64_t bnStrides, uint64_t n2Strides)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float *)gm);
        kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, blockTableGm_, constInfo_.maxBlockNumPerBatch,
                                             bnStrides, n2Strides);
    }

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfoX runInfo)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            outputBuf.SetCrossCore();
            return;
        }

        if constexpr (USE_DN) {
            ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            InitOutputSingleCore();
            if (constInfo_.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
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
        int64_t singleCoreSize = (totalOutputSize + (2 * constInfo_.coreNum) - 1) / (2 * constInfo_.coreNum);
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if constexpr (IsSameType<OUT_T, int8_t>::value) {
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOutGm_.GetPhyAddr(0)));
                matmul::InitOutput<half>(attentionOutTmpGm[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
            } else {
                matmul::InitOutput<OUT_T>(attentionOutGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;
        int64_t singleCoreSize = (totalOutputSize + (2 * constInfo_.coreNum) - 1) / (2 * constInfo_.coreNum);
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 3e+99);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<float> &dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX &runInfo)
    {
        FaUbTensor<float> ubTensor{
            .tensor = dstTensor,
            .rowCount = runInfo.actVecMSize,
            .colCount = dRealSize,
        };

        GmCoord gmCoord{
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.realN2Idx,
            .gS1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx,
            .dIdx = dOffset,
            .gS1DealSize = runInfo.actVecMSize,
        };
        copyQueryScaleGmToUb_(ubTensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<float> &dstTensor, RunInfoX &runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, 1, runInfo);
    }

    __aicore__ inline void CopyKeyScaleSlice(const LocalTensor<float> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                             RunInfoX &runInfo, bool isPreload)
    {
        FaUbTensor<float> ubTensor{.tensor = dstTensor, .rowCount = runInfo.actSingleLoopS2Size, .colCount = 1};

        uint32_t s2Idx = runInfo.s2Idx;
        uint32_t s2DealSize = runInfo.actSingleLoopS2Size;
        if (likely(isPreload)) {
            s2Idx = runInfo.s2Idx + s2BaseSize;
            if (s2Idx + s2BaseSize > runInfo.actS2Size) {
                s2DealSize = runInfo.actS2Size - s2Idx;
            }
        }

        AntiqGmCoord antiqGmCoord{
            .bIdx = runInfo.bIdx, .n2Idx = runInfo.n2Idx, .s2Idx = s2Idx, .s2DealSize = s2DealSize};
        ProcessAntiqPA(ubTensor, keyScaleGm_, antiqGmCoord);
    }

    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<float> &dstTensor, RunInfoX &runInfo, bool isPreload)
    {
        CopyKeyScaleSlice(dstTensor, 0, 1, runInfo, isPreload);
    }

    __aicore__ inline void ProcessAntiqPA(FaUbTensor<T> &dstTensor, FaGmTensorKScale &srcTensor,
                                          AntiqGmCoord &antiqGmCoord)
    {
        auto &offsetCal = srcTensor.offsetCalculator;

        uint64_t dstOffset = 0;
        uint32_t copyFinishElmeCnt = 0;
        uint32_t curS2Idx = antiqGmCoord.s2Idx;

        while (copyFinishElmeCnt < antiqGmCoord.s2DealSize) {
            uint32_t copyElemCnt =
                offsetCal.GetDimBlockSize() - curS2Idx % offsetCal.GetDimBlockSize(); // 一次只能处理一个block
            if (copyFinishElmeCnt + copyElemCnt > antiqGmCoord.s2DealSize) {
                copyElemCnt = antiqGmCoord.s2DealSize - copyFinishElmeCnt; // 一个block未拷满
            }

            uint64_t srcOffset = offsetCal.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, curS2Idx);
            CopyKeyScaleNDToND(dstTensor.tensor[dstOffset], srcTensor.gmTensor[srcOffset], copyElemCnt);

            dstOffset += copyElemCnt;
            copyFinishElmeCnt += copyElemCnt;
            curS2Idx += copyElemCnt;
        }
    }

    __aicore__ inline void CopyKeyScaleNDToND(LocalTensor<T> ubTensor, const GlobalTensor<T> gmTensor,
                                              uint32_t actDataLen)
    {
        uint32_t blockElemNum = FA_BYTE_BLOCK / sizeof(T);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(4); // 4: 4份，用于解bank冲突
        dataCopyParams.blockLen = actDataLen * sizeof(T);
        dataCopyParams.srcStride = -static_cast<int64_t>(dataCopyParams.blockLen);
        dataCopyParams.dstStride = (s2BaseSize + blockElemNum - actDataLen) * sizeof(T) / FA_BYTE_BLOCK;

        DataCopyPadExtParams<T> dataCopyPadParams;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, dataCopyPadParams);
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo)
    {
        int64_t maskLine = 0;
        if constexpr (HAS_MASK) {
            maskLine = ComputeMaskLineDN(runInfo, 0);
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];

        auto expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<T>()[0];
        int64_t stage1Offset = runInfo.loop % DB;
        int64_t kscaleOffset = (runInfo.s2Idx >> 8) & 1;

        float descaleQK = 1.0;

        // 加载qScale/kScale
        LocalTensor<float> qScaleUbTensor;
        LocalTensor<float> kScaleUbTensor;
        if (unlikely(runInfo.isFirstS2Loop)) {
            qScaleUbTensor = queryAntiqScaleInputQue_.template Get<float>();
            CopyQueryScaleTile(qScaleUbTensor, runInfo);
            kScaleUbTensor = keyAntiqScaleInputQue_[kscaleOffset].template Get<float>();
            CopyKeyScaleTile(kScaleUbTensor, runInfo, false);
        } else {
            kScaleUbTensor = keyAntiqScaleInputQue_[kscaleOffset].template Get<float>();
        }

        event_t mte2VEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2VEvtID);
        WaitFlag<HardEvent::MTE2_V>(mte2VEvtID);

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();
        if (unlikely(runInfo.isLastS2Loop && !runInfo.isFirstS2Loop)) {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, false, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        } else if (unlikely(runInfo.isFirstS2Loop)) {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, hasAtten, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, false, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        } else {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, false, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf_, qScaleUbTensor,
                    kScaleUbTensor, ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        }

        if (likely(!runInfo.isLastS2Loop)) {
            LocalTensor<float> kScaleUbNextTensor = keyAntiqScaleInputQue_[1 - kscaleOffset].template Get<float>();
            CopyKeyScaleTile(kScaleUbNextTensor, runInfo, true);
        }

        bmm1ResBuf.SetCrossCore();

        this->stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        //-------------------------Data copy to l1-------------------------
        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

        // 按照64对齐搬运
        uint32_t singleProcessSOuterSize = mBaseSize >> 1;
        uint32_t s2RealSizeAlign = (((runInfo.actSingleLoopS2Size + 63) >> 6) << 6);
        uint32_t mm2aBase = constInfo_.subBlockIdx * singleProcessSOuterSize * s2RealSizeAlign;
        DataCopy(mm2AL1Tensor[mm2aBase], stage1CastTensor,
                 {static_cast<uint16_t>((runInfo.actSingleLoopS2Size + 63) >> 6), 64, 66, 0});
        DataCopy(mm2AL1Tensor[mm2aBase + s2RealSizeAlign * 32], stage1CastTensor[65 << 5],
                 {static_cast<uint16_t>((runInfo.actSingleLoopS2Size + 63) >> 6), 64, 66, 0});

        //-----------------------------------------------------------------
        this->stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        outputBuf.SetCrossCore();

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
        return;
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
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

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = this->softmaxLseQueue_.template AllocTensor<float>();

        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        softmaxLseQueue_.template EnQue(lseUb);
        softmaxLseQueue_.DeQue<float>();

        uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
        uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
        DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                 runInfo.actVecMSize, constInfo_);

        softmaxLseQueue_.FreeTensor(lseUb);
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
        int32_t deScaleVOffset = runInfo.n2Idx;
        float deScaleVValue = this->deScaleVGm_.GetValue(deScaleVOffset); // GQA: per-head V descale

        LocalTensor<T> vec2ResUb = this->stage2OutBuf_.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;

            if (!runInfo.isLastS2Loop) {
                if (unlikely(runInfo.s2Idx == s2BaseSize)) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                }
            } else {
                if (unlikely(runInfo.s2Idx == s2BaseSize)) {
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                } else {
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                }
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, deScaleVValue);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
    }

    __aicore__ inline int64_t ComputeMaskLineDN(RunInfoX &runInfo, uint32_t vecMIdx)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;

        int64_t nextToken = static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
        uint32_t s1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        uint32_t s2StartIdx = runInfo.s2Idx;
        if (static_cast<int64_t>(s2StartIdx + s2RealSize) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            isSkipMask_ = true;
        } else {
            isSkipMask_ = false;
        }
        int64_t maskLine = nextToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(s2StartIdx);
        return maskLine;
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
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
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfoX &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                     LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                     int64_t dSizeAligned64)
    {
        return;
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            s1StartTdx = vecMStartIdx / constInfo_.realGSize;
            s1EndTdx = vecMEndIdx / constInfo_.realGSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
                ret = (s1StartTdx < s1FirstValidToken);
                ret = ret || (s1EndTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = ((constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) && hasValidRow);
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
                if constexpr (!POST_QUANT) {
                    RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo_.dSizeV,
                                              static_cast<uint32_t>(dSizeAligned64));
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo_.dSizeV, sizeof(float));
                    RowInvalidUpdateVF<float>(*((LocalTensor<float> *)&vec2ResUb), maxTensor, mDealSize, dSize,
                                              dStride);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.realN2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcst_.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        sumBrdcst_.template EnQue(sumOutTensor);
        sumBrdcst_.template DeQue<float>();
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst_.template FreeTensor(sumOutTensor);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcst_.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        maxBrdcst_.template EnQue(maxOutTensor);
        maxBrdcst_.template DeQue<float>();
        DataCopy(softmaxFDMaxGm_[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst_.template FreeTensor(maxOutTensor);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        // 是否要改成halfMRealSize
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // flashDecodeS2Idx?nBufferStartM?
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[runInfo.loop % DB]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[runInfo.loop % DB]);
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        ProcessVec2OnUb(bmm2ResBuf, runInfo);
        return;
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        tPipe_->InitBuffer(softmaxSumBuf_[0], 256);
        tPipe_->InitBuffer(softmaxSumBuf_[1], 256);
        tPipe_->InitBuffer(softmaxSumBuf_[2], 256);

        tPipe_->InitBuffer(softmaxMaxBuf_[0], 256);
        tPipe_->InitBuffer(softmaxMaxBuf_[1], 256);
        tPipe_->InitBuffer(softmaxMaxBuf_[2], 256);

        tPipe_->InitBuffer(softmaxExpBuf_[0], 256);
        tPipe_->InitBuffer(softmaxExpBuf_[1], 256);
        tPipe_->InitBuffer(softmaxExpBuf_[2], 256);
    }

    __aicore__ inline void InitBuffers()
    {
        SoftmaxInitBuffer();
        tPipe_->InitBuffer(queryAntiqScaleInputQue_, (mBaseSize >> 1U) * sizeof(float));
        // 4: 4份解bank冲突
        tPipe_->InitBuffer(keyAntiqScaleInputQue_[0], (s2BaseSize + (FA_BYTE_BLOCK / sizeof(T))) * sizeof(float) * 4);
        tPipe_->InitBuffer(keyAntiqScaleInputQue_[1], (s2BaseSize + (FA_BYTE_BLOCK / sizeof(T))) * sizeof(float) * 4);
        tPipe_->InitBuffer(stage2OutBuf_, 64 * dTemplateAlign64 * sizeof(T));
        tPipe_->InitBuffer(stage1OutQue_[0], 1, 16640);
        tPipe_->InitBuffer(stage1OutQue_[1], 1, 16640);
        tPipe_->InitBuffer(commonTBuf_, 512);
        if constexpr (HAS_MASK) {
            tPipe_->InitBuffer(attenMaskInQue_[0], 1, 16384);
        }

        if (constInfo_.isSoftmaxLseEnable) {
            this->tPipe_->InitBuffer(softmaxLseQueue_, 1, (mBaseSize >> 1U) * sizeof(float) * 8);
        }
        if constexpr (isFp8) {
            if constexpr (USE_DN) {
                tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], 256);

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < (256 >> 2); j++) {
                        vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
                    }
                }
            } else {
                tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)], 128);
                tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], 64);

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
                vselrIndexesTensor.SetValue(0, 0x7f);
                for (int i = 0; i < 128; i++) {
                    vselrIndexesTensor.SetValue(i, i * 2);
                }
                vselrIndexesTensor =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
                for (int i = 0; i < 64; i++) {
                    vselrIndexesTensor.SetValue(i, i * 4);
                }
            }
        }
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[1]);
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecGqaFp8Dummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecGqaFp8Dummy(ConstInfoX &constInfo){};
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_FULLQUANT_FP8_H_
