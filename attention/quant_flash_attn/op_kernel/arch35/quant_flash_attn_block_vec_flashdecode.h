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
 * \file quant_flash_attn_block_vec_flashdecode.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
#define QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_flash_attn_common_def.h"
#if __has_include("../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h")
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#else
#include "../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#endif

namespace BaseApi {

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr fa_base_vector::UbInputFormat GeInputUbFormat()
{
    static_assert((LAYOUT == LayOutTypeEnum::LAYOUT_BSH) || (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) ||
                      (LAYOUT == LayOutTypeEnum::LAYOUT_TND) || (LAYOUT == LayOutTypeEnum::LAYOUT_NTD),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return fa_base_vector::UbInputFormat::S1G;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return fa_base_vector::UbInputFormat::GS1;
    }
}

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool useDn = false>
class QuantFlashAttnBlockVecFlashDecode {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using SINK_T = INPUT_T;
    struct TaskInfo {
        uint32_t bIdx;
        uint32_t n2Idx;
        uint32_t gS1Idx;
        uint32_t actualCombineLoopSize;
    };

private:
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 0;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 1;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 3;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 4;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 5;
    static constexpr uint64_t SYNC_SINK_BUF1_FLAG = 6;
    static constexpr uint64_t SYNC_SINK_BUF2_FLAG = 7;

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum_ = 2U;
    uint32_t dSizeV_Align_;
    static constexpr bool attenMaskFlag = hasAtten;
    using ConstInfoX = ConstInfo_t;
    // 基本块大小
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;

protected:
    GlobalTensor<float> lseSumFdGm_;
    GlobalTensor<float> lseMaxFdGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<SINK_T> sinkGm_;

    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<layout>();
    static constexpr bool isS1G = (UB_FORMAT == UbFormat::S1G);
    static constexpr bool isPa = KvLayoutType > 0;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, isPa>();
    __gm__ uint8_t *keyPtr_ = nullptr;

    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD),
                                  ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;

    using KvSeqParserType = typename std::conditional<
        (!isPa && (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD)),
        ActualSeqLensParser<KV_MODE, int32_t, true>, ActualSeqLensParser<KV_MODE, int32_t>>::type;

    QSeqParserType *qActSeqLensParser_ = nullptr;
    KvSeqParserType *kvActSeqLensParser_ = nullptr;

    int64_t preTokensPerBatch_ = 0;
    int64_t nextTokensPerBatch_ = 0;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar_ = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    bool learnableSinkFlag_ = false;

    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    // ================================类成员变量====================================
    // 结构体
    const ConstInfoX &constInfo_;
    TaskInfo taskInfo_{};

private:
    // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1_;    // 1.5k: 16*24*4
    TBuf<> fdSumBuf2_;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf1_;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf2_;    // 1.5k: 16*24*4
    TBuf<> fdLseExpBuf_;  // 1.5k: 16*24*4
    TBuf<> fdMm2ResBuf1_; // 32k: 16*512*4
    TBuf<> fdMm2ResBuf2_; // 32k: 16*512*4
    TBuf<> fdReduceBuf_;  // 32k: 16*512*4
    TBuf<> fdOutputBuf_;  // 32k: 16*512*4
    TBuf<> fdSinkCopyInBuf_;
    TBuf<> fdSinkValueBuf_;
    TBuf<> fdSinkExpBuf_;
    TBuf<> fdSinkTmpBuf_;

    TBuf<> fdLseMaxUbBuf1_;
    TBuf<> fdLseMaxUbBuf2_;
    TBuf<> fdLseUbBuf_;

public:
    __aicore__ inline QuantFlashAttnBlockVecFlashDecode(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
                                            __gm__ uint8_t *key)
    {
        this->lseMaxFdGm_ = lseMaxFdGm;
        this->lseSumFdGm_ = lseSumFdGm;
        this->accumOutGm_ = accumOutGm;
        this->attentionOutGm_ = attentionOutGm;
        this->keyPtr_ = key;
    }

    __aicore__ inline void SetCuSeqLensParsers(QSeqParserType &qParser, KvSeqParserType &kvParser)
    {
        this->qActSeqLensParser_ = &qParser;
        this->kvActSeqLensParser_ = &kvParser;
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm_ = softmaxLseGm;
    }
    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink)
    {
        learnableSinkFlag_ = true;
        this->sinkGm_ = learnableSink;
    }
    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align_ = this->Align(constInfo_.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }
    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        if ASCEND_IS_AIV {
            pipe->Reset();
            // InQue, DB, SYNC_LSE_MAX_SUM_BUF1_FLAG SYNC_LSE_MAX_SUM_BUF2_FLAG
            pipe->InitBuffer(fdSumBuf1_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);

            pipe->InitBuffer(fdSumBuf2_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(fdMaxBuf1_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(fdMaxBuf2_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            // TmpBuf
            pipe->InitBuffer(fdLseExpBuf_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            // InQue, DB, SYNC_MM2RES_BUF1_FLAG SYNC_MM2RES_BUF2_FLAG
            pipe->InitBuffer(fdMm2ResBuf1_, BUFFER_SIZE_BYTE_16K);
            pipe->InitBuffer(fdMm2ResBuf2_, BUFFER_SIZE_BYTE_16K);
            // TmpBuf
            pipe->InitBuffer(fdReduceBuf_, BUFFER_SIZE_BYTE_16K);
            // OutQue, SYNC_FDOUTPUT_BUF_FLAG
            pipe->InitBuffer(fdOutputBuf_, BUFFER_SIZE_BYTE_16K);
            pipe->InitBuffer(fdLseMaxUbBuf1_, BUFFER_SIZE_BYTE_256B);
            pipe->InitBuffer(fdLseMaxUbBuf2_, BUFFER_SIZE_BYTE_256B);
            // OutQue, SYNC_LSEOUTPUT_BUF_FLAG
            pipe->InitBuffer(fdLseUbBuf_, BUFFER_SIZE_BYTE_256B);

            // TmpBuf
            if (unlikely(learnableSinkFlag_)) {
                // InQue, DB, SYNC_SINK_BUF1_FLAG SYNC_SINK_BUF2_FLAG
                pipe->InitBuffer(fdSinkCopyInBuf_, BUFFER_SIZE_BYTE_2K);
                // TmpBuf
                pipe->InitBuffer(fdSinkValueBuf_, BUFFER_SIZE_BYTE_2K);
                pipe->InitBuffer(fdSinkTmpBuf_, BUFFER_SIZE_BYTE_2K);
            }
            // 后面要取地址，放到外面
            pipe->InitBuffer(fdSinkExpBuf_, BUFFER_SIZE_BYTE_256B);
        }
    }
    __aicore__ inline void AllocEventID()
    {
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF2_FLAG);
    }
    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF2_FLAG);
    }

protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount)
    {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> copyInPadParams;
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = constInfo_.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align_ - constInfo_.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align_ - constInfo_.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset = startRow * constInfo_.dSizeV +                 // taskoffset + g轴offset
                                         splitKVIndex * s1BaseSize * constInfo_.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm_[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_.Get<T>() : fdSumBuf2_.Get<T>();
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_.Get<T>() : fdMaxBuf2_.Get<T>();

        uint64_t combineLseOffset = (baseOffset + startRow) * BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = s1BaseSize * BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm_[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm_[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM, uint32_t startRow)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_.Get<T>() : fdSumBuf2_.Get<T>();
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_.Get<T>() : fdMaxBuf2_.Get<T>();
        if (unlikely(learnableSinkFlag_)) {
            SinkMax(startRow, dealRowCount);
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_.Get<T>() : fdLseMaxUbBuf2_.Get<T>();

        LocalTensor<T> sinkExpBuf = fdSinkExpBuf_.Get<T>();
        LocalTensor<T> maxLseUb = fdLseUbBuf_.Get<T>();
        ComputeScaleValue_VF_FD(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount,
                                actualCombineLoopSize, constInfo_.isSoftmaxLseEnable, learnableSinkFlag_);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoord gmCoord{.bIdx = taskInfo_.bIdx,
                        .n2Idx = taskInfo_.n2Idx,
                        .gS1Idx = taskInfo_.gS1Idx + startRow,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo_.dSizeV};

        if (constInfo_.outputLayout == FA_LAYOUT::BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size,
                                              constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo_.outputLayout == FA_LAYOUT::BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size,
                                              constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo_.outputLayout == FA_LAYOUT::TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.n2Size, constInfo_.gSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo_.outputLayout == FA_LAYOUT::NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.n2Size, constInfo_.gSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align_;
        ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf_.Get<OUTPUT_T>();
        AscendC::PipeBarrier<PIPE_V>();
        DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align_);
        DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align_, cntM);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo_.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT,
                 dealRowCount * this->dSizeV_Align_);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                 dealRowCount * this->dSizeV_Align_);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align_);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    __aicore__ inline void CalcPreNextTokens()
    {
        actSeqLensQ_ = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
        if (constInfo_.cuSeqLensKVSize == 0 && constInfo_.seqUsedKvSize == 0 && !constInfo_.isKvContinuous) {
            actSeqLensKv_ = SeqLenFromTensorList<layout>(keyPtr_, taskInfo_.bIdx);
        } else {
            actSeqLensKv_ = kvActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
        }
        int64_t safePreToken = constInfo_.preTokens;
        int64_t safeNextToken = constInfo_.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ_, actSeqLensKv_, safePreToken, safeNextToken,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == BAND) {
            preTokensPerBatch_ = safePreToken;
            nextTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safeNextToken;
        } else if ((constInfo_.sparseMode == DEFAULT_MASK) && attenMaskFlag) {
            nextTokensPerBatch_ = safeNextToken;
            preTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safePreToken;
        } else {
            nextTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_;
            preTokensPerBatch_ = 0;
        }
    }
    __aicore__ inline void CopySinkIn(uint32_t cntM)
    {
        LocalTensor<SINK_T> sinkCopyInBuf =
            fdSinkCopyInBuf_.GetWithOffset<SINK_T>(BUFFER_SIZE_BYTE_1K, (cntM & 1) * BUFFER_SIZE_BYTE_1K);

        uint64_t sinkGmOffset = taskInfo_.n2Idx * constInfo_.gSize;
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = constInfo_.gSize * sizeof(SINK_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<SINK_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<SINK_T>(0);

        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + (cntM & 1));
        DataCopyPad(sinkCopyInBuf, sinkGm_[sinkGmOffset], sinkCopyParams, sinkPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINK_BUF1_FLAG + (cntM & 1));
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINK_BUF1_FLAG + (cntM & 1));

        LocalTensor<T> tmpSinkCastBuf = fdSinkTmpBuf_.Get<T>();
        Cast(tmpSinkCastBuf, sinkCopyInBuf, AscendC::RoundMode::CAST_NONE, constInfo_.gSize);
        AscendC::PipeBarrier<PIPE_V>();

        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + (cntM & 1));

        LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf_.Get<T>();
        Brcb(sinkBrcbBuf, tmpSinkCastBuf, (constInfo_.gSize + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
             {1, BLOCK_ELEMENT_NUM});
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void SinkMax(uint32_t startRow, uint32_t dealRowCount)
    {
        constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
        int64_t gIdx = 0;
        LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf_.Get<T>();
        LocalTensor<T> sinkExpBuf = fdSinkExpBuf_.Get<T>();

        for (int64_t row = 0; row < dealRowCount; ++row) {
            if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) { // 内存按照S1G排布
                gIdx = (taskInfo_.gS1Idx + startRow + row) % constInfo_.gSize;
            } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) { // 内存按照GS1排布
                int64_t actS1Size = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
                gIdx = (taskInfo_.gS1Idx + startRow + row) / actS1Size;
            }
            DataCopy(sinkExpBuf[row * BLOCK_ELEMENT_NUM], sinkBrcbBuf[gIdx * BLOCK_ELEMENT_NUM], BLOCK_ELEMENT_NUM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if (!attenMaskFlag) {
            return;
        }

        if (constInfo_.sparseMode == ALL_MASK || constInfo_.sparseMode == LEFT_UP_CAUSAL) {
            return;
        }

        fa_base_vector::InvalidRowParams params{
            .actS1Size = actSeqLensQ_,
            .gSize = static_cast<uint64_t>(constInfo_.gSize),
            .gS1Idx = taskInfo_.gS1Idx + startRow,
            .dealRowCount = dealRowCount,
            .columnCount = columnCount,
            .preTokensPerBatch = preTokensPerBatch_,
            .nextTokensPerBatch = nextTokensPerBatch_,
        };

        fa_base_vector::InvalidRows<UBOUT_T, GeInputUbFormat<layout>()> invalidRows;
        invalidRows(attenOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if (!attenMaskFlag) {
            return;
        }
        if (constInfo_.sparseMode != DEFAULT_MASK && constInfo_.sparseMode != ALL_MASK) {
            return;
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_.Get<T>() : fdLseMaxUbBuf2_.Get<T>();

        // 这里要找到lseMaxUb 最大值为-inf 与 attenOutUb的对应位置之间的关系
        // 由于到这里的lseMaxUb 和 attenOutUb都是经过偏移后的，所以offset = 0
        // 同时，这里的lseMaxUb是经过brcb后的，所以填写true

        fa_base_vector::InvalidMaskRows<UBOUT_T, T, true>(0, dealRowCount, columnCount, lseMaxUb, negativeIntScalar_,
                                                          attenOutUb);
    }

public:
    __aicore__ inline void FlashDecode(FDparamsX &fd)
    {
        uint32_t fdBalanceMBaseSize = 8U;
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        taskInfo_.bIdx = fd.fdBN2Idx / constInfo_.n2Size;
        taskInfo_.n2Idx = fd.fdBN2Idx % constInfo_.n2Size;
        taskInfo_.gS1Idx = fd.fdMIdx * s1BaseSize;
        taskInfo_.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * s1BaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             ++fdS1gOuterMIdx) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf_.Get<T>();
            LocalTensor<T> reduceOut = fdReduceBuf_.Get<T>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            if (unlikely(learnableSinkFlag_)) {
                CopySinkIn(reduceMLoop);
            }
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum_; ++preLoadIdx) {
                LocalTensor<T> mm2Res =
                    ((reduceGlobaLoop + preLoadIdx) & 1) == 0 ? fdMm2ResBuf1_.Get<T>() : fdMm2ResBuf2_.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo_.actualCombineLoopSize, reduceMLoop, startRow);
            CalcPreNextTokens();
            if (constInfo_.isSoftmaxLseEnable) {
                // lse行无效在ComputeScaleValue的VF计算时已经进行了赋值inf处理
                LocalTensor<T> maxLseUb = fdLseUbBuf_.Get<T>();
                // 判断是否行无效
                SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo_.gS1Idx + startRow;
                if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                    // LSE 输出改为 N-major 排布 [N2*G, T]: N 在外, T 在内
                    uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(taskInfo_.bIdx);
                    uint64_t bN2Offset = taskInfo_.n2Idx * constInfo_.gSize * constInfo_.t1Size + prefixBS1;
                    DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset,
                                                                             mOffset, actualGSplitSize, constInfo_);
                } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
                    uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(taskInfo_.bIdx);
                    uint32_t s1Size = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
                    uint64_t bN2Offset =
                        prefixBS1 * constInfo_.gSize * constInfo_.n2Size + taskInfo_.n2Idx * constInfo_.gSize;
                    DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                               actualGSplitSize, constInfo_, s1Size);
                } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
                    uint64_t bN2Offset = taskInfo_.bIdx * constInfo_.gSize * constInfo_.n2Size * constInfo_.s1Size +
                                         taskInfo_.n2Idx * constInfo_.gSize * constInfo_.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
                    uint64_t s1LeftPaddingSize = 0;
                    DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo_, s1LeftPaddingSize);
                } else { // BNSD
                    uint64_t bN2Offset = taskInfo_.bIdx * constInfo_.gSize * constInfo_.n2Size * constInfo_.s1Size +
                                         taskInfo_.n2Idx * constInfo_.gSize * constInfo_.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(taskInfo_.bIdx);
                    uint64_t s1LeftPaddingSize = 0;
                    DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo_, qActSeqLens,
                                                                s1LeftPaddingSize);
                }
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));

            for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1_.Get<T>() : fdMm2ResBuf2_.Get<T>();
                if (i >= preLoadNum_) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool useDn = false>
class QuantFlashAttnBlockVecFlashDecodeDummy {
public:
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecFlashDecodeDummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
