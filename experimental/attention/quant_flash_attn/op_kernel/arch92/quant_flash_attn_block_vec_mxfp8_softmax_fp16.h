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
 * \file quant_flash_attn_block_vec_mxfp8_softmax_fp16.h
 * \brief quant_mode=3 (QFA_MXFP8_SOFTMAX_FP16): MXFP8量化 + softmax FP16 的 VECTOR 侧实现
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_SOFTMAX_FP16_H_

#include "kernel_operator.h"
#include "utils/memory_copy_arch35.h"
#include "adv_api/activation/softmax.h"
#include "quant_flash_attn_common_def.h"
#include "vf/vf_flashupdate_trans_fp16.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn_mxfp8_v2.h"
#include "../../common/op_kernel/arch35/util_regbase.h"

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
          bool isFd = false, bool useDn = false, bool isDAligned = true>
class QuantFlashAttnBlockVecMxfp8SoftmaxFp16 {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2;      // C1 C1 C2
    static constexpr uint32_t s2SplitSize = 256U; // 仅支持d=128, 无需按d=256缩小split
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;

    static constexpr bool FLASH_DECODE = isFd;

    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse刷无效行会用到剩余ub，需加同步

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using attenMaskGmType = int8_t; // 不支持mask, 仅占位保持类型接口
    using quantGmType = GlobalTensor<float>;

    using ConstInfoX = ConstInfo_t;

    using MM1_OUT_T = half; // C1 FIXPIPE随路cast FP16
    using MM2_OUT_T = T;    // MM2输出仍为float(O累加与row_sum除法需要FP32精度)
    using OUT_T = OUTPUT_T;

    TPipe* tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<int32_t> actualSeqLengthsGmQ_;
    using QSeqParserType = ActualSeqLensParser<Q_MODE, int32_t>;
    QSeqParserType* qActSeqLensParser_ = nullptr;

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType& qParser)
    {
        this->qActSeqLensParser_ = &qParser;
    }

    quantGmType pScaleGm_;

    TQue<QuePosition::VECOUT, 1> stage1OutQue_[2];
    TBuf<> stage2OutBuf_;
    TEventID mte3ToVId_[2]; // MTE3_V事件: [0]=fixpipe完→V2读, [1]预留
    TEventID vToMte3Id_[2]; // V_MTE3事件: [0]=VfDivAndCast写outputT完→MTE3搬GM, [1]预留
    TBuf<> softmaxMaxBuf_[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf_[PRELOAD_N + 1];
    TBuf<> preLoopMaxBuf_;
    TBuf<> vselrIndexesBuf_[4];
    TBuf<> rowSumAccBuf_;
    TBuf<> expF32Buf_;
    TBuf<> lseTmpBuf_;

    const ConstInfoX& constInfo_;
    half negativeHalfScalar_; // HF16_MIN: FP16域的padding/初始max值
    float pScaleValue_{1.0f};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP16_LN2}; // LSE无效行检测的min check value(half域LN2对齐)

    __aicore__ inline QuantFlashAttnBlockVecMxfp8SoftmaxFp16(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe* pipe, __gm__ uint8_t* actualSeqQlenAddr,
                                        __gm__ uint8_t* actualSeqKvlenAddr, __gm__ uint8_t* pScale,
                                        __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        tPipe_ = pipe;
        // 初始minValue: FP16域无效行max初值(-65504)转float后的位表示
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP16_LN2;
        this->negativeHalfScalar_ = *((half*)&tmp1);
        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, softmaxLse, attentionOut);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* pScale, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }
        actualSeqLengthsGmQ_.SetGlobalBuffer((__gm__ int32_t*)actualSeqQlenAddr, constInfo_.seqUsedQSize);
        if (pScale != nullptr) {
            pScaleGm_.SetGlobalBuffer((__gm__ float*)pScale);
            pScaleValue_ = this->pScaleGm_.GetValue(0);
        }
    }

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf,
                                       RunInfoX runInfo, uint32_t subLoop)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
                outputBuf.SetCrossCore();
            }
            return;
        }

        ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo, subLoop);
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
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
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.dSizeV;
        int64_t singleCoreSize = (totalOutputSize + constInfo_.coreNum - 1) / constInfo_.coreNum;
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUT_T>(attentionOutGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;
        int64_t singleCoreSize = (totalOutputSize + constInfo_.coreNum - 1) / constInfo_.coreNum;
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // =================================Private Functions=================================
    __aicore__ inline void WriteLseNegInf(uint32_t bIdx, uint32_t realN2Idx, uint32_t rowCount)
    {
        static constexpr int32_t NEG_INF_BITS = static_cast<int32_t>(0xFF800000U); // fp32 -inf 位模式
        const float negInfValue = *reinterpret_cast<const float*>(&NEG_INF_BITS);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[1]);
        LocalTensor<float> lseInfUb = lseTmpBuf_.template Get<float>();
        Duplicate(lseInfUb, negInfValue, 2 * mBaseSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Id_[1]);
        WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Id_[1]);

        uint64_t base = ((uint64_t)bIdx * constInfo_.realN2Size + realN2Idx) * constInfo_.s1Size;
        uint32_t offset = 0U;
        while (offset < rowCount) {
            uint32_t chunk = AttentionCommon::Min(rowCount - offset, 2 * mBaseSize);
            SafeStrideCopy<float>(softmaxLseGm_[base + offset], lseInfUb, 1U, chunk * sizeof(float), 0U, 0U);
            offset += chunk;
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[1]);
    }

    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm1ResBuf,
                                         RunInfoX runInfo, uint32_t subLoop)
    {
        uint32_t softmaxBufIdx = runInfo.mloop % (PRELOAD_N + 1);
        uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1);
        int64_t stage1Offset = subLoop % DB;

        // FP16 softmax状态tensor: half, 每行m=128个元素; 无sum tensor(row_sum由C2阶段随路得出)
        LocalTensor<half> maxUb = this->softmaxMaxBuf_[softmaxBufIdx].template Get<half>();
        LocalTensor<half> preLoopMaxUb = this->preLoopMaxBuf_.template Get<half>();
        LocalTensor<half> expUb = this->softmaxExpBuf_[expBufIdx].template Get<half>();

        // C1 FIXPIPE随路cast FP16: bmm1结果直接为half, softmaxscale已随路乘完
        LocalTensor<half> mmRes = bmm1ResBuf.template GetTensor<half>();
        auto stage1CastTensor = this->stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();
        constexpr int32_t pScaleByteOffset = 2 * 16640; // 2*16640: P区域(2-way打包)的softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        static constexpr int32_t s2BaseSizeCur = s2BaseSize >> 1;
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2CalcSize = subLoop == 0 ? s2BaseSizeCur : runInfo.actSingleLoopS2Size - s2BaseSizeCur;
        }

        constexpr uint32_t vfM = mBaseSize;
        if (unlikely(runInfo.isFirstS2Loop)) {
            // NoUpdate: 首个S2 loop无历史max
            FaVectorApi::ProcessVec1VfDnMxfp8V2<half, INPUT_T, false, s2BaseSizeCur>(
                stage1CastTensor, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor, vfM, s2CalcSize,
                pScaleValue_, negativeHalfScalar_, preLoopMaxUb, subLoop);
        } else {
            // Update: 与历史max合并, expUb输出O的重缩放因子exp(旧max-新max)(half)
            FaVectorApi::ProcessVec1VfDnMxfp8V2<half, INPUT_T, true, s2BaseSizeCur>(
                stage1CastTensor, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor, vfM, s2CalcSize,
                pScaleValue_, negativeHalfScalar_, preLoopMaxUb, subLoop);
        }
        bmm1ResBuf.SetCrossCore();

        this->stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        uint64_t pScaleL1Offset = mBaseSize * s2BaseSize; // PScale在L1的偏移量（单位：元素）
        LocalTensor<fp8_e8m0_t> mm2AScaleL1Tensor = outputBuf.GetTensor<fp8_e8m0_t>(pScaleL1Offset);
        constexpr uint16_t pScaleDstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 4 - 1;
        constexpr uint32_t copyTimes = s2BaseSizeCur / MXFP_GROUP_SIZE / 4;
        constexpr uint16_t pScaleBlockCount = mBaseSize / 16;
        if ((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) {
            // VF写128个per-row E8M0, 8个k-group共享; 两次DataCopy各4iter从同一源复制到L1 1024B
            for (uint16_t i = 0; i < copyTimes; i++) {
                DataCopy(mm2AScaleL1Tensor[i * 32], pScaleSubLoop0Tensor, {pScaleBlockCount, 1, 0, pScaleDstStride});
            }
        }

        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        // 一个DataCopy搬整个P矩阵(256×128)到L1: subLoop0/1分别写L1偏移256*128*subLoop;
        // 源为2-way打包(行距2080B), srcStride=65(2080B)与VF blockStride对齐
        int64_t subLoopOffset = s2BaseSizeCur * mBaseSize * subLoop;
        constexpr uint16_t stage1RowStride32B = (s2BaseSizeCur >> 2) | 1U; // 65
        constexpr uint16_t stage1SrcStride32B = stage1RowStride32B;        // 65, 与VF blockStride对齐
        constexpr uint16_t stage1BlockCount = 1;
        DataCopy(mm2AL1Tensor[subLoopOffset], stage1CastTensor, {stage1BlockCount, 256 * 128 / 32, 0, 0});
        this->stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
            (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            outputBuf.SetCrossCore();
        }
    }

    // CastExpToF32: half exp(half) -> float, softmaxExpBuf[loop%(PRELOAD_N+1)] -> expF32Buf
    __aicore__ inline void CastExpToF32(LocalTensor<float>& expF32Ub, RunInfoX& runInfo)
    {
        LocalTensor<half> expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<half>();
        Cast(expF32Ub, expUb, RoundMode::CAST_NONE, mBaseSize);
    }

    __aicore__ inline void SoftmaxLseCopyOut(RunInfoX& runInfo, uint32_t vecMSize)
    {
        if (unlikely(vecMSize == 0)) {
            return;
        }
        // lseTmp布局: [0, mBaseSize) = maxF32(half->float), [mBaseSize, 2*mBaseSize) = lse结果
        LocalTensor<float> lseTmp = this->lseTmpBuf_.template Get<float>();
        LocalTensor<float> maxF32Ub = lseTmp[0];
        LocalTensor<float> lseUb = lseTmp[mBaseSize];
        LocalTensor<half> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<half>();
        Cast(maxF32Ub, maxUb, RoundMode::CAST_NONE, mBaseSize); // half->float无损

        LocalTensor<T> rowSumAccUb = this->rowSumAccBuf_.template Get<T>();
        float pScaleRec = 1.0f / pScaleValue_;   // pScale默认1.0, lse = max + ln(rs/pScale)
        float minValueF = *((float*)&minValue_); // half域LN2对齐后的无效行max(float域)
        FaVectorApi::ComputeLseTrans<float>(lseUb, rowSumAccUb, maxF32Ub, vecMSize, pScaleRec, minValueF);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]); // V写lse -> MTE3读
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

        uint32_t s1Start = runInfo.gS1Idx;
        uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
        uint32_t dealCount = vecMSize;
        if (qActSeqLens < (uint64_t)s1Start + vecMSize) {
            dealCount = qActSeqLens > s1Start ? (uint32_t)(qActSeqLens - s1Start) : 0;
        }
        if (dealCount == 0) {
            return;
        }
        uint64_t gmOffset =
            ((uint64_t)runInfo.bIdx * constInfo_.realN2Size + runInfo.realN2Idx) * constInfo_.s1Size + s1Start;
        DataCopyExtParams lseCopyParams;
        lseCopyParams.blockCount = 1;
        lseCopyParams.blockLen = dealCount * sizeof(float);
        lseCopyParams.srcStride = 0;
        lseCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm_[gmOffset], lseUb, lseCopyParams);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos& bmm2ResBuf, RunInfoX runInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }
        uint32_t vecMSize = runInfo.actVecMSize;
        LocalTensor<float> mmRes = bmm2ResBuf.template GetTensor<float>(); // (d+32)x128 float
        LocalTensor<float> stage2OutUb = this->stage2OutBuf_.template Get<float>();
        LocalTensor<float> accRowsumUb = this->rowSumAccBuf_.template Get<float>();
        LocalTensor<float> expF32Ub = this->expF32Buf_.template Get<float>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        // 跨S2分块累积: 首块纯拷贝起步(无需exp), 后续乘加
        if (unlikely(runInfo.isFirstS2Loop)) {
            FaVectorApi::VfFlashUpdateAcc<false>(stage2OutUb, accRowsumUb, mmRes, expF32Ub);
        } else {
            CastExpToF32(expF32Ub, runInfo);
            FaVectorApi::VfFlashUpdateAcc<true>(stage2OutUb, accRowsumUb, mmRes, expF32Ub);
        }

        // 末块: 除cast转置带pad输出(mm2Res别名区) + GM搬出
        if (unlikely(runInfo.isLastS2Loop)) {
            PipeBarrier<PIPE_V>();
            LocalTensor<bfloat16_t> outputT = mmRes.template ReinterpretCast<bfloat16_t>();
            FaVectorApi::VfDivAndCast(outputT, stage2OutUb, accRowsumUb);
            SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

            FaUbTensor<OUTPUT_T, !isDAligned> ubTensor{.tensor = outputT, .rowCount = vecMSize, .colCount = 144U};
            GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                                  .n2Idx = runInfo.realN2Idx,
                                  .gIdx = 0,
                                  .s1Idx = runInfo.gS1Idx,
                                  .dIdx = 0,
                                  .s1DealSize = vecMSize,
                                  .dDealSize = (uint32_t)constInfo_.dSizeV};
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, UbFormat::S1_ONLY> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);

            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(runInfo, vecMSize);
            }

            SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        bmm2ResBuf.SetCrossCore();
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        // FP16 softmax状态tensor: 每行m=128个half(256B), 无sum相关buffer
        tPipe_->InitBuffer(softmaxMaxBuf_[0], 256);
        tPipe_->InitBuffer(softmaxMaxBuf_[1], 256);
        tPipe_->InitBuffer(softmaxMaxBuf_[2], 256);

        tPipe_->InitBuffer(softmaxExpBuf_[0], 256);
        tPipe_->InitBuffer(softmaxExpBuf_[1], 256);
        tPipe_->InitBuffer(softmaxExpBuf_[2], 256);

        tPipe_->InitBuffer(preLoopMaxBuf_, 256);
    }

    __aicore__ inline void InitBuffers()
    {
        SoftmaxInitBuffer();
        tPipe_->InitBuffer(stage2OutBuf_, mBaseSize * dTemplateAlign64 * sizeof(T));
        // P(FP8 2-way打包)区域33280B + pScale(E8M0 x128)256B对齐
        tPipe_->InitBuffer(stage1OutQue_[0], 1, 2 * 16640 + 256);
        tPipe_->InitBuffer(stage1OutQue_[1], 1, 2 * 16640 + 256);
        tPipe_->InitBuffer(rowSumAccBuf_, mBaseSize * sizeof(float));
        tPipe_->InitBuffer(expF32Buf_, mBaseSize * sizeof(float));
        tPipe_->InitBuffer(lseTmpBuf_, 2 * mBaseSize * sizeof(float));

        // DN->NZ Gather索引表: 按VF V2的2-way打包重排, 位置128i+j取值i+2j,
        // 即输出位置p取源(p>>7)+2*(p&127), 与VF V2的Gather语义一致
        tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], 256);
        LocalTensor<uint8_t> vselrIndexesTensor =
            vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 128; j++) {
                vselrIndexesTensor.SetValue(i * 128 + j, i + (j << 1));
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
class QuantFlashAttnBlockVecMxfp8SoftmaxFp16Dummy {
public:
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecMxfp8SoftmaxFp16Dummy(ConstInfoX& constInfo){};
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_SOFTMAX_FP16_H_
