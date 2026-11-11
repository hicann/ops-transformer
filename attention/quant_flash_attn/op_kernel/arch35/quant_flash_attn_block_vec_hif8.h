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
 * \file quant_flash_attn_block_vec_hif8.h
 * \brief HIF8 per-tensor quantization vec block. No pScale, v_scale applied in vec2.
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_HIF8_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_HIF8_H_

#include "kernel_operator.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "adv_api/activation/softmax.h"
#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "../../../common/op_kernel/init_output.h"

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
class QuantFlashAttnBlockVecHif8 {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = mBaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = mBaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = mBaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

    static constexpr bool USE_DN = useDn;

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar_ = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    // UB 静态布局尺寸
    static constexpr uint32_t UB_SOFTMAX_BUFCNT = PRELOAD_N + 1;
    static constexpr uint32_t UB_SOFTMAX_BUF_BYTES = 256U;
    static constexpr uint32_t UB_BRDCST_BUFCNT = 2U;
    static constexpr uint32_t UB_BRDCST_BUF_BYTES = 2048U;
    static constexpr uint32_t UB_LSE_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float) * 8U;
    static constexpr uint32_t UB_VSELR_INDEX_BYTES = 256U;
    static constexpr uint32_t UB_STAGE1_OUT_BYTES = 16640U;
    static constexpr uint32_t UB_STAGE1_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_STAGE2_OUT_ELEM_CNT = 64U * dTemplateAlign64;

    // 跨核 bmm 区尺寸(与 cube/kernel 侧一致)
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = (mBaseSize / CV_RATIO) * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = (mBaseSize / CV_RATIO) * (uint32_t)dVTemplateType * sizeof(T);

    // 跨核同步 id(参考 flash_attn arch35): AIV 只用基址
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CC_MM1_READY_0 = 0U;
    static constexpr uint32_t CC_MM1_READY_1 = 1U;
    static constexpr uint32_t CC_MM2_READY = 2U;
    static constexpr uint32_t CC_MM1_FREE_0 = 8U;
    static constexpr uint32_t CC_MM1_FREE_1 = 9U;
    static constexpr uint32_t CC_MM2_FREE = 10U;
    static constexpr uint32_t CC_L1P_0 = 5U;
    static_assert(CC_L1P_0 + L1_P_BUFCNT <= CC_MM1_FREE_0,
                  "cross-core free flag ids must not overlap the L1 P flag id range");

    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm_;
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

    quantGmType pScaleGm_;
    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    // 跨核 buffer(裸 LocalTensor): bmm2 结果 / bmm1 结果 / L1 P, 由本 block 自行分配
    LocalTensor<uint8_t> ubBmm2ResBuf_;
    LocalTensor<uint8_t> ubBmm1ResBuffers_;
    LocalTensor<uint8_t> l1PBuffers_;
    uint32_t bmm1ResBufId_ = 0U;
    uint32_t l1PBufId_ = 0U;

    // ub (静态Tensor: 显式UB地址, 不再使用TBuf/TQue)
    LocalTensor<INPUT_T> stage1OutBuf_[UB_STAGE1_OUT_BUFCNT];
    LocalTensor<T> stage2OutBuf_;
    LocalTensor<float> softmaxMaxBuf_[UB_SOFTMAX_BUFCNT];
    LocalTensor<float> softmaxSumBuf_[PRELOAD_N + 1];
    LocalTensor<float> softmaxExpBuf_[PRELOAD_N + 1];
    LocalTensor<uint8_t> vselrIndexesTensor_;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域(仅FLASH_DECODE) */
    LocalTensor<float> maxBrdcst_;
    LocalTensor<float> sumBrdcst_;

    LocalTensor<float> softmaxLseBuf_;

    // ClearOutput 的 InitOutput 使用事件同步(ENABLE_LOCK=false)
    static constexpr uint32_t UB_OUT_VEC2_RES_ID = 0U; // stage2Out / bmm2 输出
    static constexpr uint32_t UB_OUT_LSE_OUT_ID = 4U;  // lseOut

    // 同核 V<->MTE3 使用编译期 eventId 的 SetFlag/WaitFlag, 避免 Mutex 信号量交接延迟
    static constexpr uint32_t MTE3_V_EVENT0 = EVENT_ID0; // stage2Out
    static constexpr uint32_t MTE3_V_EVENT1 = EVENT_ID1; // lseOut
    static constexpr uint32_t MTE3_V_EVENT2 = EVENT_ID2; // stage1Out buf0
    static constexpr uint32_t MTE3_V_EVENT3 = EVENT_ID3; // stage1Out buf1
    static constexpr uint32_t MTE3_V_EVENT5 = EVENT_ID5; // sumBrdcst
    static constexpr uint32_t MTE3_V_EVENT6 = EVENT_ID6; // maxBrdcst
    static constexpr uint32_t V_MTE3_EVENT0 = EVENT_ID0;
    static constexpr uint32_t V_MTE3_EVENT1 = EVENT_ID1;
    static constexpr uint32_t V_MTE3_EVENT2 = EVENT_ID2;
    static constexpr uint32_t V_MTE3_EVENT3 = EVENT_ID3;
    static constexpr uint32_t V_MTE3_EVENT4 = EVENT_ID4; // FD s2-split 输出
    static constexpr uint32_t V_MTE3_EVENT5 = EVENT_ID5;
    static constexpr uint32_t V_MTE3_EVENT6 = EVENT_ID6;

    const ConstInfoX& constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    float descaleQKValue_{1.0f};
    float vScaleValue_{1.0f};
    bool isSkipMask_{false};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP32};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecHif8(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void SetDescaleQK(float qScale, float kScale)
    {
        descaleQKValue_ = qScale * kScale;
    }

    __aicore__ inline void SetVScale(float vScale)
    {
        vScaleValue_ = vScale;
    }

    __aicore__ inline void InitVecBlock(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                        __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* pScale,
                                        __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T*)&tmp1);

        float qScale = 1.0f;
        float kScale = 1.0f;
        float vScale = 1.0f;
        if (dequantScaleQuery != nullptr) {
            qScale = *((__gm__ float*)dequantScaleQuery);
        }
        if (dequantScaleKey != nullptr) {
            kScale = *((__gm__ float*)dequantScaleKey);
        }
        if (dequantScaleValue != nullptr) {
            vScale = *((__gm__ float*)dequantScaleValue);
        }
        SetDescaleQK(qScale, kScale);
        SetVScale(vScale);
        UpdateMinCheckValue();

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

    __aicore__ inline void InitCrossCoreSync()
    {
        // AIV 先释放所有 UB slot, 供 AIC 首次写入。
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM2_FREE);
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void ProcessVec1(RunInfoX runInfo)
    {
        uint32_t bmm1ResBufId = bmm1ResBufId_;
        bmm1ResBufId_ = (bmm1ResBufId_ + 1U) % UB_MM1_RES_BUFCNT;
        uint32_t mmReadySyncIdx = CC_MM1_READY_0 + bmm1ResBufId;
        uint32_t mmFreeSyncIdx = CC_MM1_FREE_0 + bmm1ResBufId;
        uint32_t pL1BufId = l1PBufId_;
        l1PBufId_ = (l1PBufId_ + 1U) % L1_P_BUFCNT;

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmReadySyncIdx);
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmFreeSyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + pL1BufId);
            return;
        }

        if constexpr (USE_DN) {
            LocalTensor<T> mmRes = ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
            LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
            ProcessVec1Dn(pL1Tensor, mmRes, runInfo, mmFreeSyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + pL1BufId);
            if (unlikely(runInfo.isLastS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
            }
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (constInfo_.needInitOutput) {
            InitOutputSingleCore();
            if (constInfo_.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            SyncAll();
        }
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
            totalOutputSize /= 2; // 2 : post-quant output is 1 byte, init as half (2 bytes)
        }
        uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum; // 2 means c:v = 1:2
        // FA 计算前可复用跨核 BMM 区清零输出
        static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0U;
        static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = 32768U / sizeof(OUT_T);
        if constexpr (IsSameType<OUT_T, int8_t>::value) {
            GlobalTensor<half> attentionOutTmpGm;
            attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(attentionOutGm_.GetPhyAddr(0)));
            AttentionCommon::InitOutput<half, UB_OUT_VEC2_RES_ID, ATTEN_OUT_POP_BUF_START_ADDR, 32768U / sizeof(half),
                                        false>(attentionOutTmpGm, totalOutputSize, vecCoreNum, 0);
        } else {
            AttentionCommon::InitOutput<OUT_T, UB_OUT_VEC2_RES_ID, ATTEN_OUT_POP_BUF_START_ADDR,
                                        ATTEN_OUT_POP_BUF_ELE_SIZE, false>(attentionOutGm_, totalOutputSize, vecCoreNum,
                                                                           0);
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
        uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum; // 2 means c:v = 1:2
        static constexpr float LSE_INIT_VAL = 3e+99;
        static constexpr uint32_t LSE_POP_BUF_START_ADDR = 32768U;
        static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = 32768U / sizeof(float);
        AttentionCommon::InitOutput<float, UB_OUT_LSE_OUT_ID, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, false>(
            softmaxLseGm_, totalOutputSize, vecCoreNum, LSE_INIT_VAL);
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(LocalTensor<INPUT_T>& mm2AL1Tensor, LocalTensor<T>& mmRes, RunInfoX runInfo,
                                         uint32_t mmFreeSyncIdx)
    {
        int64_t maskLine = 0;
        if constexpr (HAS_MASK) {
            maskLine = ComputeMaskLineDN(runInfo);
        }

        uint32_t softmaxBufIdx = runInfo.mloop % (PRELOAD_N + 1);
        uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1);
        int64_t stage1Offset = runInfo.loop % DB;

        LocalTensor<float> sumUb = this->softmaxSumBuf_[softmaxBufIdx];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[softmaxBufIdx];
        LocalTensor<float> expUb = this->softmaxExpBuf_[expBufIdx];

        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));
        auto stage1CastTensor = this->stage1OutBuf_[stage1Offset];
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;

        if (unlikely(runInfo.isFirstS2Loop)) {
            if (unlikely(!isSkipMask_)) {
                FaVectorApi::ProcessVec1VfDnHif8<T, INPUT_T, false, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQKValue_, pScaleValue_, negativeFloatScalar_, 0.0F,
                    maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnHif8<T, INPUT_T, false, false, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQKValue_, pScaleValue_, negativeFloatScalar_, 0.0F,
                    maskLine);
            }
        } else {
            if (unlikely(!isSkipMask_)) {
                FaVectorApi::ProcessVec1VfDnHif8<T, INPUT_T, true, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQKValue_, pScaleValue_, negativeFloatScalar_, 0.0F,
                    maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnHif8<T, INPUT_T, true, false, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQKValue_, pScaleValue_, negativeFloatScalar_, 0.0F,
                    maskLine);
            }
        }
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));

        // BMM1 结果已消费, 立即释放跨核 slot
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmFreeSyncIdx);

        uint32_t s2RealSizeAlign = ((runInfo.actSingleLoopS2Size + 63) >> 6) << 6;
        constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
        DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * s2RealSizeAlign], stage1CastTensor,
                 {static_cast<uint16_t>(s2RealSizeAlign >> 6), 64, 66, 0});
        DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * s2RealSizeAlign + s2RealSizeAlign * 32],
                 stage1CastTensor[65 << 5], {static_cast<uint16_t>(s2RealSizeAlign >> 6), 64, 66, 0});
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));
    }

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

    __aicore__ inline void UpdateMinCheckValue()
    {
        float min = *((float*)&minValue_);
        if constexpr (USE_DN) {
            min *= constInfo_.scaleValue * descaleQKValue_;
        }
        minValue_ = static_cast<uint32_t>(*reinterpret_cast<int32_t*>(&min));
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
                                             RunInfoX& runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMSize = runInfo.actVecMSize;
        uint32_t gmDealRowCount = runInfo.actVecMSize;
        if (gmDealRowCount == 0) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = this->softmaxLseBuf_;
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT1);
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT1);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT1);

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                     gmDealRowCount, constInfo_);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.s1Size +
                                 runInfo.realN2Idx * constInfo_.realGSize * constInfo_.s1Size;
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, s1LeftPaddingSize);
        } else {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.s1Size +
                                 runInfo.realN2Idx * constInfo_.realGSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, qActSeqLens, s1LeftPaddingSize);
        }

        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT1);
    }

    __aicore__ inline void ProcessVec2OnUb(LocalTensor<T>& mmRes, RunInfoX runInfo)
    {
        uint32_t vecMSize = runInfo.actVecMSize;
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf_;
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
        if (unlikely(runInfo.isFirstS2Loop)) {
            Muls(vec2ResUb, mmRes, vScaleValue_, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
            LocalTensor<T> pScaleUb;

            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, vScaleValue_, 1.0f);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, vScaleValue_,
                    1.0f);
            }
        }
        if (likely(!runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
        }
    }

    __aicore__ inline void PostQuant(RunInfoX& runInfo, LocalTensor<OUTPUT_T>& attenOut, LocalTensor<T>& vec2ResUb,
                                     int64_t mStartVec, int64_t mDealSize, int64_t dSizeAligned64)
    {
        return;
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
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
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
            bool batchNeedRowInvalid = ((constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) && hasValidRow);
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor = softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)][mStartVec];
                if constexpr (!POST_QUANT) {
                    RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo_.dSizeV,
                                              static_cast<uint32_t>(dSizeAligned64), minValue_);
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo_.dSizeV, sizeof(float));
                    RowInvalidUpdateVF<float>(*((LocalTensor<float>*)&vec2ResUb), maxTensor, mDealSize, dSize, dStride,
                                              minValue_);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX& info, LocalTensor<OUTPUT_T>& attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        FaUbTensor<OUTPUT_T, false> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.realN2Idx,
                                .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                                .dIdx = 0,
                                .gS1DealSize = gmDealRowCount,
                                .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, false>& ubTensor, GmCoordGs1Merge& gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
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

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                               LocalTensor<float>& maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5);
        LocalTensor<float> sumOutTensor = sumBrdcst_;
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT5);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT5);
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5);

        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6);
        LocalTensor<float> maxOutTensor = maxBrdcst_;
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT6);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT6);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxOutTensor, calculateSize);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6);
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
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
    }

    __aicore__ inline void ProcessVec2(RunInfoX runInfo)
    {
        uint32_t mmReadySyncIdx = CC_MM2_READY;
        uint32_t mmFreeSyncIdx = CC_MM2_FREE;
        LocalTensor<T> mmRes = ubBmm2ResBuf_.template ReinterpretCast<T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmReadySyncIdx);
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmFreeSyncIdx);
            return;
        }

        ProcessVec2OnUb(mmRes, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmFreeSyncIdx);

        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    stage2OutBuf_, stage2OutBuf_, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, 1.0f);
            }
            CopyOutAttentionOut(runInfo, stage2OutBuf_, 0, runInfo.actVecMSize, runInfo.actVecMSize);
        }
    }

    // 跨核bmm区字节数(kernel侧InitMMResBuf公式一致), 作为FA vec区起始偏移
    // 注意: hif8 每 task 不做 s2 subLoop 切分, 故 mm1ResultSize 不含 mxfp8 那个 "/2";
    //       且本类模板参数名为 dVTemplateType(无 dVBaseSize), 与 kernel 侧 dVBaseSize 同值
    static __aicore__ inline constexpr uint32_t GetMmUbTotalSize()
    {
        constexpr uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T);
        return mm1ResultSize * 2 + mm2ResultSize;
    }

    // FA保留区字节数(紧跟kernel侧ubBufferManager_的跨核bmm区之后):
    // softmax状态(9*256) + brdcst(FLASH_DECODE时2*2048) + softmaxLseBuf(2048, 常开) + vselr索引表(256)
    static __aicore__ inline constexpr uint32_t GetPersistUbSize()
    {
        constexpr uint32_t softmaxSize = 9U * 256U;
        constexpr uint32_t brdcstSize = FLASH_DECODE ? (2U * 2048U) : 0U;
        constexpr uint32_t lseSize = (mBaseSize >> 1U) * sizeof(float) * 8U;
        constexpr uint32_t vselrSize = isFp8 ? 256U : 0U;
        return softmaxSize + brdcstSize + lseSize + vselrSize;
    }

    // FA瞬态区字节数: stage2Out + stage1Out*2, FD静态Tensor业务区(见FD block GetFdTotalUbSize)
    // 须完全落在该区间内, 供 kernel 侧 static_assert 校验
    static __aicore__ inline constexpr uint32_t GetTransientUbSize()
    {
        constexpr uint32_t stage2Size = 64U * dTemplateAlign64 * sizeof(T);
        constexpr uint32_t stage1Size = 2U * 16640U;
        return stage2Size + stage1Size;
    }

    __aicore__ inline void InitBuffers()
    {
        // 静态Tensor布局: 跨核区(bmm2 + bmm1) 之后依序为持久区、瞬态区
        // (与 GetPersistUbSize / GetTransientUbSize 口径严格一致)
        // 注: 原动态实现里的 commonTBuf_(512B) 在 hif8 路径从未被读写, 静态化后不再分配;
        //     softmaxLseBuf_ 改为常开分配(原按运行期 isSoftmaxLseEnable 条件分配),
        //     否则后续 buffer 的静态偏移无法编译期确定
        uint32_t addrUb = 0U;
        ubBmm2ResBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUF_BYTES);
        addrUb += UB_MM2_RES_BUF_BYTES;
        ubBmm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);
        addrUb += UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_P_BUFCNT * L1_P_BUF_BYTES);

        uint32_t addr = GetMmUbTotalSize();
        constexpr uint32_t softmaxBufBytes = 256U;
        for (uint32_t i = 0; i < PRELOAD_N + 1; ++i) {
            softmaxSumBuf_[i] = LocalTensor<float>(TPosition::VECCALC, addr, softmaxBufBytes / sizeof(float));
            addr += softmaxBufBytes;
        }
        for (uint32_t i = 0; i < PRELOAD_N + 1; ++i) {
            softmaxMaxBuf_[i] = LocalTensor<float>(TPosition::VECCALC, addr, softmaxBufBytes / sizeof(float));
            addr += softmaxBufBytes;
        }
        for (uint32_t i = 0; i < PRELOAD_N + 1; ++i) {
            softmaxExpBuf_[i] = LocalTensor<float>(TPosition::VECCALC, addr, softmaxBufBytes / sizeof(float));
            addr += softmaxBufBytes;
        }

        if constexpr (FLASH_DECODE) {
            maxBrdcst_ = LocalTensor<float>(TPosition::VECCALC, addr, 2048U / sizeof(float)); // [64, 8]
            addr += 2048U;
            sumBrdcst_ = LocalTensor<float>(TPosition::VECCALC, addr, 2048U / sizeof(float)); // [64, 8]
            addr += 2048U;
        }

        constexpr uint32_t lseBytes = (mBaseSize >> 1U) * sizeof(float) * 8U;
        softmaxLseBuf_ = LocalTensor<float>(TPosition::VECCALC, addr, lseBytes / sizeof(float));
        addr += lseBytes;

        if constexpr (isFp8) {
            // 仅 DN 路径使用 vselr 索引表(256B), 未使用分支不再建表
            if constexpr (USE_DN) {
                vselrIndexesTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, addr, 256U);
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < (256 >> 2); j++) {
                        vselrIndexesTensor_.SetValue(i * (256 >> 2) + j, i + (j << 2));
                    }
                }
            }
            // 与 GetPersistUbSize 的 vselrSize(isFp8 ? 256U : 0U) 口径一致
            addr += 256U;
        }

        // ===== 瞬态区: FD静态Tensor覆盖区, 分配顺序与GetTransientUbSize保持一致 =====
        stage2OutBuf_ = LocalTensor<T>(TPosition::VECCALC, addr, 64U * dTemplateAlign64);
        addr += 64U * dTemplateAlign64 * sizeof(T);
        // (32 + 1) * (256 / 32) * 64
        stage1OutBuf_[0] = LocalTensor<uint8_t>(TPosition::VECCALC, addr, 16640U).template ReinterpretCast<INPUT_T>();
        addr += 16640U;
        stage1OutBuf_[1] = LocalTensor<uint8_t>(TPosition::VECCALC, addr, 16640U).template ReinterpretCast<INPUT_T>();
        addr += 16640U;
    }

    __aicore__ inline void AllocEventID()
    {
        // 预置 MTE3->V 方向的事件额度, 与 FreeEventID 的 drain 一一镜像
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0); // stage2Out
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT1); // lseOut
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2); // stage1Out buf0
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT3); // stage1Out buf1
        if constexpr (FLASH_DECODE) {
            SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5); // sumBrdcst
            SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6); // maxBrdcst
        }
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT1);
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT2);
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT3);
        if constexpr (FLASH_DECODE) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT5);
            WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT6);
        }
    }

    __aicore__ inline int64_t ComputeMaskLineDN(RunInfoX& runInfo)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        int64_t nextToken = static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
        uint32_t s1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        uint32_t s2StartIdx = runInfo.s2Idx;
        if (static_cast<int64_t>(s2StartIdx + s2RealSize) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            isSkipMask_ = true;
        } else {
            isSkipMask_ = false;
        }
        int64_t maskLine = nextToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(s2StartIdx);
        return maskLine;
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecHif8Dummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecHif8Dummy(ConstInfoX& constInfo){};
    __aicore__ inline void InitCrossCoreSync() {}
    __aicore__ inline void UnInitCrossCoreSync() {}
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_VEC_HIF8_H_
