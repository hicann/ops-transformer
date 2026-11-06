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
#include "../../../common/op_kernel/init_output.h"
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

    // ==================== UB 布局常量(按地址顺序) ====================
    static constexpr uint32_t UB_SOFTMAX_BUFCNT = PRELOAD_N + 1;
    static constexpr uint32_t UB_SOFTMAX_BUF_BYTES = 256U;
    static constexpr uint32_t UB_BRDCST_BUFCNT = 2U;
    static constexpr uint32_t UB_BRDCST_BUF_BYTES = 2048U;
    static constexpr uint32_t UB_Q_SCALE_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float);
    // 4: 4份解bank冲突
    static constexpr uint32_t UB_K_SCALE_BUF_BYTES = (s2BaseSize + (FA_BYTE_BLOCK / sizeof(T))) * sizeof(float) * 4U;
    static constexpr uint32_t UB_K_SCALE_BUFCNT = 2U; // 2: 双缓冲
    static constexpr uint32_t UB_Q_SCALE_BUFCNT = 2U; // 2: 双缓冲(按 mloop%2 轮转)
    static constexpr uint32_t UB_LSE_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float) * 8U;
    static constexpr uint32_t UB_VSELR_INDEX_BYTES = 256U;
    // (32 + 1) * (256 / 32) * 64
    static constexpr uint32_t UB_STAGE1_OUT_BYTES = 16640U;
    static constexpr uint32_t UB_STAGE1_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_STAGE2_OUT_ELEM_CNT = 64U * dTemplateAlign64;
    static constexpr uint32_t UB_VSELR_INDEX_REPEAT = 4U;
    static constexpr uint32_t UB_VSELR_INDEX_ELEM_PER_REPEAT = UB_VSELR_INDEX_BYTES >> 2;

    // AIC 同步 base 和 base + AIV0_AIV1_OFFSET，AIV 仅使用 base
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CC_MM1_READY_0 = 0U; // AIC Fixpipe -> AIV, bmm1 slot0
    static constexpr uint32_t CC_MM1_READY_1 = 1U; // AIC Fixpipe -> AIV, bmm1 slot1
    static constexpr uint32_t CC_MM2_READY = 2U;   // AIC Fixpipe -> AIV, bmm2 slot
    static constexpr uint32_t CC_MM1_FREE_0 = 8U;  // AIV -> AIC, bmm1 slot0
    static constexpr uint32_t CC_MM1_FREE_1 = 9U;  // AIV -> AIC, bmm1 slot1
    static constexpr uint32_t CC_MM2_FREE = 10U;   // AIV -> AIC, bmm2 slot
    static constexpr uint32_t CC_L1P_0 = 5U;       // L1 P slot0
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static_assert(CC_L1P_0 + L1_P_BUFCNT <= CC_MM1_FREE_0,
                  "cross-core free flag ids must not overlap the L1 P flag id range");
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = (mBaseSize / CV_RATIO) * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES =
        (mBaseSize / CV_RATIO) * static_cast<uint32_t>(dVTemplateType) * sizeof(T);

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

    static constexpr bool USE_DN = useDn;
    static constexpr UbFormat OUT_UB_FORMAT = USE_DN ? UbFormat::S1_ONLY : GetOutUbFormat<layout>();
    using OutGmCoord = std::conditional_t<USE_DN, GmCoordS1Only, GmCoordGs1Merge>;
    static constexpr bool IS_PER_TOEKN_HEAD = true;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr GmFormat Q_SCALE_FORMAT = GmFormat::NGT;
    static constexpr GmFormat K_SCALE_FORMAT = GmFormat::PA_BnNBs;
    using pseShiftType = half;

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
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD),
                                  ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;
    QSeqParserType* qActSeqLensParser_ = nullptr;

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
    // 跨核 BMM 结果和 L1 P buffer
    LocalTensor<uint8_t> ubBmm2ResBuf_;
    LocalTensor<uint8_t> ubBmm1ResBuffers_;
    LocalTensor<uint8_t> l1PBuffers_;
    uint32_t bmm1ResBufId_ = 0U;
    uint32_t l1PBufId_ = 0U;
    // UB 静态 buffer
    LocalTensor<INPUT_T> stage1OutBuf_[UB_STAGE1_OUT_BUFCNT];
    LocalTensor<T> stage2OutBuf_;
    LocalTensor<float> softmaxMaxBuf_[UB_SOFTMAX_BUFCNT];
    LocalTensor<float> softmaxSumBuf_[UB_SOFTMAX_BUFCNT];
    LocalTensor<float> softmaxExpBuf_[UB_SOFTMAX_BUFCNT];
    LocalTensor<uint8_t> vselrIndexesTensor_;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域(仅FLASH_DECODE) */
    LocalTensor<float> maxBrdcst_;
    LocalTensor<float> sumBrdcst_;
    LocalTensor<float> softmaxLseBuf_;
    LocalTensor<float> queryAntiqScaleInputBuf_[UB_Q_SCALE_BUFCNT];
    LocalTensor<float> keyAntiqScaleInputBuf_[UB_K_SCALE_BUFCNT];

    // 核内 buffer 均使用 Mutex 同步，ID 按用途分段
    static constexpr uint32_t UB_OUT_VEC2_RES_ID = 0U;
    static constexpr uint32_t UB_OUT_VEC1_RES_ID0 = 2U;
    static constexpr uint32_t UB_OUT_VEC1_RES_ID1 = 3U;
    static constexpr uint32_t UB_OUT_LSE_OUT_ID = 4U;
    static constexpr uint32_t UB_OUT_BRDCST_SUM_ID = 6U;
    static constexpr uint32_t UB_OUT_BRDCST_MAX_ID = 7U;
    static constexpr uint32_t QBUF_SCALE_ID = 8U;
    static constexpr uint32_t KBUF_SCALE_ID = 10U;
    const ConstInfoX& constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    bool isSkipMask_{false};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecGqaFp8(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* pScale, __gm__ uint8_t* blockTable, __gm__ uint8_t* deqScaleQ,
                                        __gm__ uint8_t* deqScaleK, __gm__ uint8_t* deqScaleV,
                                        __gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut,
                                        __gm__ uint8_t* workspace, QSeqParserType& qParser)
    {
        this->qActSeqLensParser_ = &qParser;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T*)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, blockTable, deqScaleQ, deqScaleK, deqScaleV,
                     softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* pScale, __gm__ uint8_t* blockTable, __gm__ uint8_t* deqScaleQ,
                                        __gm__ uint8_t* deqScaleK, __gm__ uint8_t* deqScaleV,
                                        __gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut,
                                        __gm__ uint8_t* workspace)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }

        uint64_t actualLenQSize = (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) ?
                                      constInfo_.cuSeqLensQSize :
                                      constInfo_.seqUsedQSize;
        actualSeqLengthsGmQ_.SetGlobalBuffer((__gm__ int32_t*)actualSeqQlenAddr, actualLenQSize);

        if (pScale != nullptr) {
            pScaleGm_.SetGlobalBuffer((__gm__ float*)pScale);
            pScaleValue_ = this->pScaleGm_.GetValue(0);
        }
        if (deqScaleV != nullptr) {
            deScaleVGm_.SetGlobalBuffer((__gm__ float*)deqScaleV);
        }
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }

        InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size,
                         queryScaleGm_, deqScaleQ);
        InitKScaleBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, keyScaleGm_,
                         deqScaleK, constInfo_.kDescaleStrides.bnStride, constInfo_.kDescaleStrides.n2Stride);
        if constexpr (FLASH_DECODE) {
            accumOutGm_.SetGlobalBuffer((__gm__ float*)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            FaGmTensorQScale& qScaleGmTensor, __gm__ uint8_t* gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float*)gm);
        if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_NT) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, *this->qActSeqLensParser_);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, FaGmTensorKScale& kScaleGmTensor,
                                            __gm__ uint8_t* gm, uint64_t bnStrides, uint64_t n2Strides)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float*)gm);
        kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, blockTableGm_, constInfo_.maxBlockNumPerBatch,
                                             bnStrides, n2Strides);
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        // AIV 先释放所有 UB slot, 供 AIC 首次写入
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
            totalOutputSize /= 2;
        }
        uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum; // 2 means c:v = 1:2
        // FA 计算前可复用跨核 BMM 区清零输出
        static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0U;
        static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = 32768U / sizeof(OUT_T);
        if constexpr (IsSameType<OUT_T, int8_t>::value) {
            GlobalTensor<half> attentionOutTmpGm;
            attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(attentionOutGm_.GetPhyAddr(0)));
            AttentionCommon::InitOutput<half, UB_OUT_VEC2_RES_ID, ATTEN_OUT_POP_BUF_START_ADDR, 32768U / sizeof(half),
                                        true>(attentionOutTmpGm, totalOutputSize, vecCoreNum, 0);
        } else {
            AttentionCommon::InitOutput<OUT_T, UB_OUT_VEC2_RES_ID, ATTEN_OUT_POP_BUF_START_ADDR,
                                        ATTEN_OUT_POP_BUF_ELE_SIZE, true>(attentionOutGm_, totalOutputSize, vecCoreNum,
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
        AttentionCommon::InitOutput<float, UB_OUT_LSE_OUT_ID, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, true>(
            softmaxLseGm_, totalOutputSize, vecCoreNum, LSE_INIT_VAL);
    }

    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<float>& dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX& runInfo)
    {
        FaUbTensor<float> ubTensor{
            .tensor = dstTensor,
            .rowCount = runInfo.actVecMSize,
            .colCount = dRealSize,
        };

        GmCoordGs1Merge gmCoord{
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.realN2Idx,
            .gS1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx,
            .dIdx = dOffset,
            .gS1DealSize = runInfo.actVecMSize,
        };
        copyQueryScaleGmToUb_(ubTensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<float>& dstTensor, RunInfoX& runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, 1, runInfo);
    }

    __aicore__ inline void CopyKeyScaleSlice(const LocalTensor<float>& dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                             RunInfoX& runInfo, bool isPreload)
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

    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<float>& dstTensor, RunInfoX& runInfo, bool isPreload)
    {
        CopyKeyScaleSlice(dstTensor, 0, 1, runInfo, isPreload);
    }

    __aicore__ inline void ProcessAntiqPA(FaUbTensor<T>& dstTensor, FaGmTensorKScale& srcTensor,
                                          AntiqGmCoord& antiqGmCoord)
    {
        auto& offsetCal = srcTensor.offsetCalculator;

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

    __aicore__ inline void ProcessVec1Dn(LocalTensor<INPUT_T>& mm2AL1Tensor, LocalTensor<T>& mmRes, RunInfoX runInfo,
                                         uint32_t mmFreeSyncIdx)
    {
        int64_t maskLine = 0;
        if constexpr (HAS_MASK) {
            maskLine = ComputeMaskLineDN(runInfo, 0);
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)];

        LocalTensor<T> expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
        int64_t stage1Offset = runInfo.loop % DB;
        int64_t qscaleOffset = runInfo.mloop % DB;

        float descaleQK = 1.0;

        // q/k scale 双缓冲分别按 mloop、loop 轮转
        LocalTensor<float> qScaleUbTensor = queryAntiqScaleInputBuf_[qscaleOffset];
        LocalTensor<float> kScaleUbTensor = keyAntiqScaleInputBuf_[stage1Offset];
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(QBUF_SCALE_ID + static_cast<uint32_t>(qscaleOffset));
            CopyQueryScaleTile(qScaleUbTensor, runInfo);
            Mutex::Unlock<PIPE_MTE2>(QBUF_SCALE_ID + static_cast<uint32_t>(qscaleOffset));
            Mutex::Lock<PIPE_MTE2>(KBUF_SCALE_ID + static_cast<uint32_t>(stage1Offset));
            CopyKeyScaleTile(kScaleUbTensor, runInfo, false);
            Mutex::Unlock<PIPE_MTE2>(KBUF_SCALE_ID + static_cast<uint32_t>(stage1Offset));
        }

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_ID0 + static_cast<uint32_t>(stage1Offset));
        // qScale 仅在首个 S2 loop 加载，kScale 每轮读取
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_V>(QBUF_SCALE_ID + static_cast<uint32_t>(qscaleOffset));
        }
        Mutex::Lock<PIPE_V>(KBUF_SCALE_ID + static_cast<uint32_t>(stage1Offset));
        auto stage1CastTensor = this->stage1OutBuf_[stage1Offset];
        if (unlikely(runInfo.isLastS2Loop && !runInfo.isFirstS2Loop)) {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, false, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        } else if (unlikely(runInfo.isFirstS2Loop)) {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, hasAtten, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, false, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        } else {
            if (!isSkipMask_) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, false, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesTensor_, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_,
                    negativeFloatScalar_, 0.0F, maskLine);
            }
        }

        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_ID0 + static_cast<uint32_t>(stage1Offset));

        // BMM1 结果已消费，立即释放跨核 slot
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmFreeSyncIdx);

        Mutex::Unlock<PIPE_V>(KBUF_SCALE_ID + static_cast<uint32_t>(stage1Offset));
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Unlock<PIPE_V>(QBUF_SCALE_ID + static_cast<uint32_t>(qscaleOffset));
        }

        if (likely(!runInfo.isLastS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(KBUF_SCALE_ID + static_cast<uint32_t>(1 - stage1Offset));
            CopyKeyScaleTile(keyAntiqScaleInputBuf_[1 - stage1Offset], runInfo, true);
            Mutex::Unlock<PIPE_MTE2>(KBUF_SCALE_ID + static_cast<uint32_t>(1 - stage1Offset));
        }

        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_ID0 + static_cast<uint32_t>(stage1Offset));
        // 按 64 对齐搬运到 L1
        uint32_t singleProcessSOuterSize = mBaseSize >> 1;
        uint32_t s2RealSizeAlign = (((runInfo.actSingleLoopS2Size + 63) >> 6) << 6);
        uint32_t mm2aBase = constInfo_.subBlockIdx * singleProcessSOuterSize * s2RealSizeAlign;
        DataCopy(mm2AL1Tensor[mm2aBase], stage1CastTensor,
                 {static_cast<uint16_t>((runInfo.actSingleLoopS2Size + 63) >> 6), 64, 66, 0});
        DataCopy(mm2AL1Tensor[mm2aBase + s2RealSizeAlign * 32], stage1CastTensor[65 << 5],
                 {static_cast<uint16_t>((runInfo.actSingleLoopS2Size + 63) >> 6), 64, 66, 0});

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_ID0 + static_cast<uint32_t>(stage1Offset));
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

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
                                             RunInfoX& runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_ID);
        ComputeLseOutputVF(softmaxLseBuf_, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_ID);

        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_ID);
        uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
        uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
        DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, softmaxLseBuf_, bN2Offset, vecMIdx,
                                                                 runInfo.actVecMSize, constInfo_);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_ID);
    }

    __aicore__ inline void ProcessVec2OnUb(LocalTensor<T>& mmRes, float deScaleVValue, RunInfoX runInfo)
    {
        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;

        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_ID);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(stage2OutBuf_, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
            LocalTensor<T> pScaleUb;

            if (!runInfo.isLastS2Loop) {
                if (unlikely(runInfo.s2Idx == s2BaseSize)) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                        stage2OutBuf_, mmRes, stage2OutBuf_, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        stage2OutBuf_, mmRes, stage2OutBuf_, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                }
            } else {
                if (unlikely(runInfo.s2Idx == s2BaseSize)) {
                    LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                        stage2OutBuf_, mmRes, stage2OutBuf_, expUb, pScaleUb, sumUb, runInfo.actVecMSize,
                        dTemplateAlign64, deScaleVValue, deScaleVValue);
                } else {
                    LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        stage2OutBuf_, mmRes, stage2OutBuf_, expUb, pScaleUb, sumUb, runInfo.actVecMSize,
                        dTemplateAlign64, deScaleVValue, deScaleVValue);
                }
            }
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_ID);
    }

    __aicore__ inline int64_t ComputeMaskLineDN(RunInfoX& runInfo, uint32_t vecMIdx)
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

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX& runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);

        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_ID);
        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        } else {
            PostQuant(runInfo, attenOut, vec2ResUb, mStartVec, mDealSize, dSizeAligned64);
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_ID);

        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_ID);
        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_ID);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfoX& runInfo, LocalTensor<OUTPUT_T>& attenOut,
                                     LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                     int64_t dSizeAligned64)
    {
        return;
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
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
                                              static_cast<uint32_t>(dSizeAligned64));
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo_.dSizeV, sizeof(float));
                    RowInvalidUpdateVF<float>(*((LocalTensor<float>*)&vec2ResUb), maxTensor, mDealSize, dSize, dStride);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX& info, LocalTensor<OUTPUT_T>& attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        if constexpr (USE_DN) {
            // DN分支下realGSize=1，gS1Idx即s1Idx，gIdx恒为0
            OutGmCoord gmCoord{.bIdx = info.bIdx,
                               .n2Idx = info.realN2Idx,
                               .gIdx = 0,
                               .s1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                               .dIdx = 0,
                               .s1DealSize = dealRowCount,
                               .dDealSize = (uint32_t)constInfo_.dSizeV};
            CopyAttentionOut(ubTensor, gmCoord);
        } else {
            OutGmCoord gmCoord{.bIdx = info.bIdx,
                               .n2Idx = info.realN2Idx,
                               .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                               .dIdx = 0,
                               .gS1DealSize = dealRowCount,
                               .dDealSize = (uint32_t)constInfo_.dSizeV};
            CopyAttentionOut(ubTensor, gmCoord);
        }
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T>& ubTensor, OutGmCoord& gmCoord)
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

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                               LocalTensor<float>& maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        Mutex::Lock<PIPE_V>(UB_OUT_BRDCST_SUM_ID);
        FaVectorApi::BroadcastMaxSum(sumBrdcst_, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_BRDCST_SUM_ID);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_BRDCST_SUM_ID);
        DataCopy(softmaxFDSumGm_[gmOffset], sumBrdcst_, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_BRDCST_SUM_ID);

        Mutex::Lock<PIPE_V>(UB_OUT_BRDCST_MAX_ID);
        FaVectorApi::BroadcastMaxSum(maxBrdcst_, maxUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_BRDCST_MAX_ID);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_BRDCST_MAX_ID);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxBrdcst_, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_BRDCST_MAX_ID);
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
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_ID);
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_ID);
    }

    __aicore__ inline void ProcessVec2(RunInfoX runInfo)
    {
        uint32_t mmReadySyncIdx = CC_MM2_READY;
        uint32_t mmFreeSyncIdx = CC_MM2_FREE;
        LocalTensor<T> mmRes = ubBmm2ResBuf_.template ReinterpretCast<T>();
        float deScaleVValue = this->deScaleVGm_.GetValue(runInfo.n2Idx); // GQA: per-head V descale

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmReadySyncIdx);
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmFreeSyncIdx);
            return;
        }

        ProcessVec2OnUb(mmRes, deScaleVValue, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmFreeSyncIdx);

        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_ID);
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(stage2OutBuf_, stage2OutBuf_, sumUb,
                                                                          runInfo.actVecMSize,
                                                                          (uint16_t)dTemplateAlign64, deScaleVValue);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_ID);
            }
            CopyOutAttentionOut(runInfo, stage2OutBuf_, 0, runInfo.actVecMSize);
        }
    }

    // 跨核 BMM 区大小
    static __aicore__ inline constexpr uint32_t GetMmUbTotalSize()
    {
        constexpr uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T);
        return mm1ResultSize * 2 + mm2ResultSize;
    }

    // FA 持久区：softmax、scale、LSE 和 vselr
    static __aicore__ inline constexpr uint32_t GetPersistUbSize()
    {
        constexpr uint32_t softmaxSize = 3U * UB_SOFTMAX_BUFCNT * UB_SOFTMAX_BUF_BYTES;
        constexpr uint32_t brdcstSize = FLASH_DECODE ? (UB_BRDCST_BUFCNT * UB_BRDCST_BUF_BYTES) : 0U;
        constexpr uint32_t antiqSize =
            UB_Q_SCALE_BUFCNT * UB_Q_SCALE_BUF_BYTES + UB_K_SCALE_BUFCNT * UB_K_SCALE_BUF_BYTES;
        constexpr uint32_t lseSize = UB_LSE_BUF_BYTES;
        constexpr uint32_t vselrSize = isFp8 ? UB_VSELR_INDEX_BYTES : 0U;
        return softmaxSize + brdcstSize + antiqSize + lseSize + vselrSize;
    }

    // FA 与 FD 分时复用的瞬态区大小
    static __aicore__ inline constexpr uint32_t GetTransientUbSize()
    {
        constexpr uint32_t stage2Size = UB_STAGE2_OUT_ELEM_CNT * sizeof(T);
        constexpr uint32_t stage1Size = UB_STAGE1_OUT_BUFCNT * UB_STAGE1_OUT_BYTES;
        return stage2Size + stage1Size;
    }

    __aicore__ inline void InitBuffers()
    {
        // UB 按跨核区、持久区、瞬态区连续布局
        uint32_t addrUb = 0U;
        ubBmm2ResBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUF_BYTES);
        addrUb += UB_MM2_RES_BUF_BYTES;
        ubBmm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);
        addrUb += UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_P_BUFCNT * L1_P_BUF_BYTES);

        for (uint32_t i = 0; i < UB_SOFTMAX_BUFCNT; ++i) {
            softmaxSumBuf_[i] = LocalTensor<float>(TPosition::VECCALC, addrUb, UB_SOFTMAX_BUF_BYTES / sizeof(float));
            addrUb += UB_SOFTMAX_BUF_BYTES;
        }
        for (uint32_t i = 0; i < UB_SOFTMAX_BUFCNT; ++i) {
            softmaxMaxBuf_[i] = LocalTensor<float>(TPosition::VECCALC, addrUb, UB_SOFTMAX_BUF_BYTES / sizeof(float));
            addrUb += UB_SOFTMAX_BUF_BYTES;
        }
        for (uint32_t i = 0; i < UB_SOFTMAX_BUFCNT; ++i) {
            softmaxExpBuf_[i] = LocalTensor<float>(TPosition::VECCALC, addrUb, UB_SOFTMAX_BUF_BYTES / sizeof(float));
            addrUb += UB_SOFTMAX_BUF_BYTES;
        }

        if constexpr (FLASH_DECODE) {
            maxBrdcst_ = LocalTensor<float>(TPosition::VECCALC, addrUb, UB_BRDCST_BUF_BYTES / sizeof(float));
            addrUb += UB_BRDCST_BUF_BYTES;
            sumBrdcst_ = LocalTensor<float>(TPosition::VECCALC, addrUb, UB_BRDCST_BUF_BYTES / sizeof(float));
            addrUb += UB_BRDCST_BUF_BYTES;
        }

        // q/k per-token descale(GM→UB); q 双缓冲按 mloop%2 轮转, k 双缓冲按 loop%2 轮转
        for (uint32_t i = 0; i < UB_Q_SCALE_BUFCNT; ++i) {
            queryAntiqScaleInputBuf_[i] =
                LocalTensor<float>(TPosition::VECCALC, addrUb, UB_Q_SCALE_BUF_BYTES / sizeof(float));
            addrUb += UB_Q_SCALE_BUF_BYTES;
        }
        for (uint32_t i = 0; i < UB_K_SCALE_BUFCNT; ++i) {
            keyAntiqScaleInputBuf_[i] =
                LocalTensor<float>(TPosition::VECCALC, addrUb, UB_K_SCALE_BUF_BYTES / sizeof(float));
            addrUb += UB_K_SCALE_BUF_BYTES;
        }

        // 常开分配以保持静态布局偏移固定
        softmaxLseBuf_ = LocalTensor<float>(TPosition::VECCALC, addrUb, UB_LSE_BUF_BYTES / sizeof(float));
        addrUb += UB_LSE_BUF_BYTES;

        if constexpr (isFp8) {
            if constexpr (USE_DN) {
                vselrIndexesTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, addrUb, UB_VSELR_INDEX_BYTES);
                for (uint32_t i = 0; i < UB_VSELR_INDEX_REPEAT; i++) {
                    for (uint32_t j = 0; j < UB_VSELR_INDEX_ELEM_PER_REPEAT; j++) {
                        vselrIndexesTensor_.SetValue(i * UB_VSELR_INDEX_ELEM_PER_REPEAT + j, i + (j << 2));
                    }
                }
            }
            addrUb += UB_VSELR_INDEX_BYTES;
        }

        stage2OutBuf_ = LocalTensor<T>(TPosition::VECCALC, addrUb, UB_STAGE2_OUT_ELEM_CNT);
        addrUb += UB_STAGE2_OUT_ELEM_CNT * sizeof(T);
        for (uint32_t i = 0; i < UB_STAGE1_OUT_BUFCNT; ++i) {
            stage1OutBuf_[i] = LocalTensor<uint8_t>(TPosition::VECCALC, addrUb, UB_STAGE1_OUT_BYTES)
                                   .template ReinterpretCast<INPUT_T>();
            addrUb += UB_STAGE1_OUT_BYTES;
        }
    }

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}
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
    __aicore__ inline QuantFlashAttnBlockVecGqaFp8Dummy(ConstInfoX& constInfo){};
    __aicore__ inline void InitCrossCoreSync() {}
    __aicore__ inline void UnInitCrossCoreSync() {}
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_FULLQUANT_FP8_H_
