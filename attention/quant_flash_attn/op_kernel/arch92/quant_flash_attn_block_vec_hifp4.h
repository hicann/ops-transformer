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
 * \file quant_flash_attn_block_vec_hifp4.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_HIFP4_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_HIFP4_H_

#include "kernel_operator.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch92/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn_hif4.h"
#include "../../../common/op_kernel/arch92/vf/vf_flashupdate_hif4.h"
#include "quant_flash_attn_common_def_arch92.h"
#include "../../../common/op_kernel/vector_common.h"
#include "memory_copy_arch92_quant_flash_attn.h"

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
class QuantFlashAttnBlockVecHifp4 {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t BASE_K_BMM2 = 256;
    static constexpr uint32_t C2V2_1_C1V1_8 = 8;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t BASE_S2_C1_SIZE = 128U; // 子轮S2SIZE
    static constexpr uint32_t BANK_STRIDE_B8 = 1024U;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    float LN2 = static_cast<float>(0.6931471806f);
    float INV_LN2 = static_cast<float>(1.4426950409F);

    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

    static constexpr bool USE_DN = useDn;

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    // 核内同步id
    static constexpr int32_t EVENT_ID_P_MTE3_V_PING = 0; // 生产者 wait 消费者
    static constexpr int32_t EVENT_ID_P_MTE3_V_PONG = 1;
    static constexpr int32_t EVENT_ID_PSCALE_MTE3_V_PING = 2;
    static constexpr int32_t EVENT_ID_PSCALE_MTE3_V_PONG = 3;
    static constexpr int32_t EVENT_ID_V2_MTE3_V = 4;
    static constexpr int32_t EVENT_ID_P_V_MTE3_PING = 0; // 消费者 wait 生产者
    static constexpr int32_t EVENT_ID_P_V_MTE3_PONG = 1;
    static constexpr int32_t EVENT_ID_PSCALE_V_MTE3_PING = 2;
    static constexpr int32_t EVENT_ID_PSCALE_V_MTE3_PONG = 3;
    static constexpr int32_t EVENT_ID_V2_ATTEN_OUT_V_MTE3 = 4;
    static constexpr int32_t EVENT_ID_V2_LSE_V_MTE3 = 5;
    // 核间同步id
    static constexpr uint16_t FLAG_ID_V1_C1_BUFFER0 = 0;
    static constexpr uint16_t FLAG_ID_V1_C1_BUFFER1 = 1;
    static constexpr uint16_t FLAG_ID_V1_C1_BUFFER2 = 2;
    static constexpr uint16_t FLAG_ID_V1_C1_BUFFER3 = 3;
    static constexpr uint16_t FLAG_ID_C1_V1 = 4;
    static constexpr uint16_t FLAG_ID_C2_V1_PING = 5;
    static constexpr uint16_t FLAG_ID_C2_V1_PONG = 6;
    static constexpr uint16_t FLAG_ID_V1_C2_PING = 7;
    static constexpr uint16_t FLAG_ID_V1_C2_PONG = 8;
    static constexpr uint16_t FLAG_ID_V2_C2 = 9;
    static constexpr uint16_t FLAG_ID_C2_V2 = 10;

    static constexpr uint32_t BUFFER_CNT2 = 2;
    static constexpr uint32_t BUFFER_CNT4 = 4;
    static constexpr uint32_t BUFFER_CNT16 = 16;
    static constexpr uint32_t EIGHT = 8;

    static constexpr uint32_t P_ELEMENT_SIZE = mBaseSize * s2BaseSize; // 128 * 128 = 16k
    static constexpr uint32_t Pscale_L1_ELEMENT_SIZE =
        mBaseSize * s2BaseSize / MXFP_GROUP_SIZE * C2V2_1_C1V1_8;                            // 128 * 128 / 32 * 8= 4k
    static constexpr uint32_t Pscale_UB_ELEMENT_SIZE = mBaseSize * 2 * C2V2_1_C1V1_8;        // 128 * 2 * 8 = 2k
    static constexpr uint32_t Mm1Res_UB_ELEMENT_SIZE = mBaseSize * s2BaseSize * BUFFER_CNT4; // 128 * 128 * 4 = 64k
    static constexpr uint32_t Mm2Res_UB_ELEMENT_SIZE = mBaseSize * dTemplateAlign64;         // 128 * 128 = 16k
    static constexpr uint32_t SumOld_UB_ELEMENT_SIZE = mBaseSize;                            // 128
    static constexpr uint32_t SumNew_UB_ELEMENT_SIZE = mBaseSize * EIGHT;                    // 128 * 8 = 1k
    static constexpr uint32_t V2Res_UB_ELEMENT_SIZE = mBaseSize * dTemplateAlign64;          // 128 * 128 = 16k
    static constexpr uint32_t VselrIdx_SIZE = 256;

    using attenMaskGmType = typename std::conditional<HAS_MASK, GlobalTensor<uint8_t>, int8_t>::type;
    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;

    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<int32_t> actualSeqLengthsGmQ;
    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND), ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;
    QSeqParserType* qActSeqLensParser = nullptr;

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType& qParser)
    {
        this->qActSeqLensParser = &qParser;
    }

    attenMaskGmType attenMaskGmInt;

    postQuantGmType postQuantScaleGm;
    postQuantGmType postQuantOffsetGm;
    postQuantBf16GmType postQuantScaleBf16Gm;
    postQuantBf16GmType postQuantOffsetBf16Gm;
    GlobalTensor<float> pScaleGm;

    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    /* =====================静态LocalTensor成员====================*/
    // L1 静态Tensor (与cube侧共享物理L1地址, vec写P/Pscale, cube读)
    LocalTensor<uint8_t> PTensor[BUFFER_CNT16];
    LocalTensor<uint8_t> PscaleTensor[BUFFER_CNT2];

    // UB 静态Tensor
    // cube侧fixpipe输出 (vec读), 与cube侧共享物理UB地址
    LocalTensor<half> mm1ResUB;     // bmm1结果(cube fixpipe写, vec读)
    LocalTensor<float> mm2ResUB;    // bmm2结果(cube fixpipe写, vec读)
    LocalTensor<float> rowSumOldUB; // rowSumOld(vec自身使用)
    LocalTensor<float> rowSumNewUB; // rowSumNew(cube fixpipe写, vec读)

    // vec自身UB Tensor
    LocalTensor<uint8_t> v1PRes[BUFFER_CNT2];
    LocalTensor<uint8_t> v1PscaleRes[BUFFER_CNT2];
    LocalTensor<float> v2Res;
    LocalTensor<uint8_t> vselrIndexesTensor;
    LocalTensor<float> softmaxLseTensor;

    // VF接口所需的UB裸地址 (静态分配后与LocalTensor地址一致)
    uint64_t v1PResAddr[BUFFER_CNT2];
    uint64_t v1PscaleResAddr[BUFFER_CNT2];
    uint64_t softmaxMaxTensorAddr;
    uint64_t softmaxGlobalMaxTensorAddr[BUFFER_CNT2];
    uint64_t softmaxExpMaxTensorAddr[BUFFER_CNT2];
    uint64_t v2ResAddr;

    const ConstInfoX& constInfo;
    T negativeFloatScalar;
    float pScaleValue{1.0f};
    bool isSkipMask{false};
    uint32_t minValue{NEGATIVE_MIN_VALUE_FP32};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecHifp4(ConstInfoX& constInfo)
        : constInfo(constInfo){};

    __aicore__ inline void InitInput(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                     __gm__ uint8_t* pScale, __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxLse,
                                     __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP16;
        this->negativeFloatScalar = *((T*)&tmp1);
        UpdateMinCheckValue();

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, attenMask, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* pScale, __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }

        uint64_t actualLenQSize =
            (layout == LayOutTypeEnum::LAYOUT_TND) ? constInfo.cuSeqLensQSize : constInfo.seqUsedQSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int32_t*)actualSeqQlenAddr, actualLenQSize);

        if constexpr (HAS_MASK) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
        }
        if (pScale != nullptr) {
            pScaleGm.SetGlobalBuffer((__gm__ float*)pScale);
            pScaleValue = this->pScaleGm.GetValue(0);
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm.SetGlobalBuffer((__gm__ float*)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float*)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float*)workspace + constInfo.accumOutSize + constInfo.logSumExpSize);
        }
    }

    __aicore__ inline void ProcessVec1(RunInfoX runInfo, uint32_t subLoop)
    {
        if constexpr (USE_DN) {
            ProcessVec1Dn(runInfo, subLoop);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore();
            if (constInfo.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.realN2Size * constInfo.realGSize * constInfo.dSizeV;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if constexpr (IsSameType<OUT_T, int8_t>::value) {
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(attentionOutGm.GetPhyAddr(0)));
                matmul::InitOutput<half>(attentionOutTmpGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            } else {
                matmul::InitOutput<OUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.realN2Size * constInfo.realGSize;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                      3e+99); // 3e+99: set the value of invalid batch to inf
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(RunInfoX runInfo, uint32_t subLoop)
    {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_ID_C1_V1); // C1.Fix trigger V1.calc

        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_P_MTE3_V_PING + subLoop % DB);
        uint32_t c1v1Loop = CeilDiv(runInfo.actSingleLoopS2Size, BASE_S2_C1_SIZE) - 1;
        if (subLoop == c1v1Loop) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID_PSCALE_MTE3_V_PING + runInfo.loop % DB);
        }
        uint32_t s2CalcSize = BASE_S2_C1_SIZE;
        if (unlikely(subLoop == c1v1Loop)) {
            s2CalcSize = runInfo.actSingleLoopS2Size - c1v1Loop * BASE_S2_C1_SIZE;
        }

        // 内部静态Tensor索引 (与kernel原调用逻辑一致)
        uint32_t pTensorIdx = (runInfo.loop % DB == 0) ? (subLoop % 8) : (subLoop % 8 + 8); // 8: 一个bank上的p块数
        uint32_t pScaleIdx = runInfo.loop % DB;
        LocalTensor<uint8_t>& outputBuf = PTensor[pTensorIdx];
        LocalTensor<uint8_t>& outputScaleBuf = PscaleTensor[pScaleIdx];
        LocalTensor<half> bmm1ResBuf =
            mm1ResUB[subLoop % BUFFER_CNT4 * 128]; // 128: 解bank冲突，每个buffer起始地址相差256B

        if (unlikely(runInfo.isFirstS2Loop)) {
            ProcessVec1VfDnHif4<T, INPUT_T, false, false, BASE_S2_C1_SIZE>(
                v1PResAddr[subLoop % DB], softmaxMaxTensorAddr, softmaxGlobalMaxTensorAddr[runInfo.mloop % DB],
                bmm1ResBuf, softmaxExpMaxTensorAddr[runInfo.loop % DB], vselrIndexesTensor,
                v1PscaleResAddr[runInfo.loop % DB], mBaseSize, BASE_S2_C1_SIZE, s2CalcSize, pScaleValue,
                negativeFloatScalar, subLoop, c1v1Loop);
        } else {
            ProcessVec1VfDnHif4<T, INPUT_T, true, false, BASE_S2_C1_SIZE>(
                v1PResAddr[subLoop % DB], softmaxMaxTensorAddr, softmaxGlobalMaxTensorAddr[runInfo.mloop % DB],
                bmm1ResBuf, softmaxExpMaxTensorAddr[runInfo.loop % DB], vselrIndexesTensor,
                v1PscaleResAddr[runInfo.loop % DB], mBaseSize, BASE_S2_C1_SIZE, s2CalcSize, pScaleValue,
                negativeFloatScalar, subLoop, c1v1Loop);
        }
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_ID_V1_C1_BUFFER0 + subLoop % BUFFER_CNT4); // V1.calc trigger C1.Fix

        //-------------------------Data copy to l1-------------------------
        if (subLoop == c1v1Loop) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(FLAG_ID_C2_V1_PING + runInfo.loop % DB); // C2.MTE1 trigger V1.MTE3
        }
        // MTE3 P
        SetFlag<HardEvent::V_MTE3>(EVENT_ID_P_V_MTE3_PING + subLoop % DB);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID_P_V_MTE3_PING + subLoop % DB);
        DataCopyParams DataCopyParamsP;
        DataCopyParamsP.blockCount = 16; // 16: P_BLOCK_COUNT
        DataCopyParamsP.blockLen = 8;    // 8: P_BLOCK_LEN
        DataCopyParamsP.srcGap = 24;     // 24: P_SRC_GAP
        DataCopyParamsP.dstGap = 0;
        static constexpr uint32_t NZ_DST_STRIDE = 32 * 2;
        static constexpr uint32_t NZ_SRC_STRIDE = 16;
        DataCopy(outputBuf, v1PRes[subLoop % DB], DataCopyParamsP);
        DataCopy(outputBuf[BASE_S2_C1_SIZE * NZ_DST_STRIDE], v1PRes[subLoop % DB][NZ_SRC_STRIDE * BANK_STRIDE_B8],
                 DataCopyParamsP); //  32:NZ datablock 16:NZ偏移
        DataCopy(outputBuf[BASE_S2_C1_SIZE * NZ_DST_STRIDE * 2],
                 v1PRes[subLoop % DB][NZ_SRC_STRIDE * BANK_STRIDE_B8 * 2], DataCopyParamsP); // 2: datablock索引
        DataCopy(outputBuf[BASE_S2_C1_SIZE * NZ_DST_STRIDE * 3],
                 v1PRes[subLoop % DB][NZ_SRC_STRIDE * BANK_STRIDE_B8 * 3], DataCopyParamsP); // 3: datablock索引
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_P_MTE3_V_PING + subLoop % DB);
        // MTE3 PScale
        if (subLoop == c1v1Loop) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID_PSCALE_V_MTE3_PING + runInfo.loop % DB);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID_PSCALE_V_MTE3_PING + runInfo.loop % DB);
            DataCopyParams DataCopyParamsPscale;
            DataCopyParamsPscale.blockCount = 8; // 8: P_BLOCK_COUNT
            DataCopyParamsPscale.blockLen = 1;
            DataCopyParamsPscale.srcGap = 0;
            DataCopyParamsPscale.dstGap = 3; // 3: P_DST_GAP
            for (uint16_t i = 0; i < C2V2_1_C1V1_8; i++) {
                // 1024:一个C2基本块pscale=1KB 64:2个NZ datablock 256:一个基本块pscale大小的一半256B
                static constexpr uint32_t PSCALE_L1_SIZE =
                    BASE_K_BMM2 * dTemplateAlign64 / MXFP_GROUP_SIZE; // 一个C2基本块pscale，256*128/32*1=1KB
                static constexpr uint32_t TWO = 2;
                static constexpr uint32_t NZ_BLOCK_SIZE = 32;
                DataCopy(outputScaleBuf[PSCALE_L1_SIZE * (i / TWO) + NZ_BLOCK_SIZE * TWO * (i % TWO)],
                         v1PscaleRes[runInfo.loop % DB][mBaseSize * TWO * i], DataCopyParamsPscale);
                DataCopy(outputScaleBuf[PSCALE_L1_SIZE * (i / TWO) + NZ_BLOCK_SIZE * TWO * (i % TWO) + NZ_BLOCK_SIZE],
                         v1PscaleRes[runInfo.loop % DB][mBaseSize * TWO * i], DataCopyParamsPscale);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID_PSCALE_MTE3_V_PING + runInfo.loop % DB);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_ID_V1_C2_PING + runInfo.loop % DB); // V1.MTE3 trigger C2.MTE1
        }
        return;
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, const uint64_t& maxUb, LocalTensor<float>& sumUb)
    {
        if constexpr (FLASH_DECODE) {
            if (!runInfo.isS2SplitCore && constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(maxUb, sumUb, runInfo);
            }
        } else {
            if (constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(maxUb, sumUb, runInfo);
            }
        }
    }

    __aicore__ inline void UpdateMinCheckValue()
    {
        float min = *((float*)&minValue);
        if constexpr (USE_DN) {
            min *= constInfo.scaleValue;
        }
        float tmp = min * INV_LN2;
        // int32_t 范围 [-2147483648, 2147483647]，下限约 -2.1e9
        // 若 tmp 小于 int32 最小值，float 精度丢失，直接原样计算
        if (tmp > -2147483648.0f) {
            min = static_cast<int32_t>(tmp) * LN2;
        } else {
            min = tmp * LN2;
        }
        minValue = static_cast<uint32_t>(*reinterpret_cast<int32_t*>(&min));
    }

    __aicore__ inline void SoftmaxLseCopyOut(const uint64_t& softmaxMaxTmp, LocalTensor<float>& softmaxSumTmp,
                                             RunInfoX& runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMSize = USE_DN ? runInfo.actVecMSize : (runInfo.actMSizeAlign32 / 2);
        uint32_t gmDealRowCount;
        if constexpr (USE_DN) {
            gmDealRowCount = runInfo.actMSize;
        } else {
            uint32_t groupsOf32 = (runInfo.actMSize + 31) / 32; // 31: 向上对齐32
            if (constInfo.subBlockIdx == 0) {
                gmDealRowCount = groupsOf32 * 16 > runInfo.actMSize ? runInfo.actMSize : groupsOf32 * 16; // 16: 16对齐
            } else {
                int32_t vec1RemainRows = runInfo.actMSize - 16 * groupsOf32; // 16: 16对齐
                gmDealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
            }
        }
        if (gmDealRowCount == 0) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        ComputeLseOutputVF(softmaxLseTensor, softmaxMaxTmp, softmaxSumTmp, vecMSize, minValue);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID_V2_LSE_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V2_LSE_V_MTE3);

        GmCoordGs1Merge lseGmCoord{.bIdx = runInfo.bIdx,
                                   .n2Idx = runInfo.realN2Idx,
                                   .gS1Idx = vecMIdx,
                                   .dIdx = 0,
                                   .gS1DealSize = gmDealRowCount,
                                   .dDealSize = 0};

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat LSE_FORMAT = GmFormat::LSE_N2GT;
            FaGmTensor<float, LSE_FORMAT, int32_t, true> lseGmTensor_;
            lseGmTensor_.gmTensor = softmaxLseGm;
            lseGmTensor_.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, *qActSeqLensParser);
            CopyLseUbToGm<float, LSE_FORMAT, GetOutUbFormat<layout>()> copyLseUbToGm_;
            copyLseUbToGm_(lseGmTensor_, softmaxLseTensor, lseGmCoord);
        } else { // BSND BSH BNSD
            constexpr GmFormat LSE_FORMAT = GmFormat::LSE_BN2GS1;
            FaGmTensor<float, LSE_FORMAT, int32_t> lseGmTensor_;
            lseGmTensor_.gmTensor = softmaxLseGm;
            lseGmTensor_.offsetCalculator.Init(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize,
                                               constInfo.s1Size, *qActSeqLensParser);
            CopyLseUbToGm<float, LSE_FORMAT, GetOutUbFormat<layout>()> copyLseUbToGm_;
            copyLseUbToGm_(lseGmTensor_, softmaxLseTensor, lseGmCoord);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(RunInfoX runInfo)
    {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_ID_C2_V2); // C2.Fix trigger V2.calc
        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;

        if (unlikely(runInfo.isFirstS2Loop)) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID_V2_MTE3_V);
            DataCopy(v2Res, mm2ResUB, vec2CalcSize);
            DataCopy(rowSumOldUB, rowSumNewUB, mBaseSize);
        } else {
            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>( //  SinnerCnt=1
                    v2ResAddr, mm2ResUB, v2ResAddr, softmaxExpMaxTensorAddr[runInfo.loop % DB], rowSumOldUB,
                    rowSumNewUB, mBaseSize, dTemplateAlign64);
            } else {
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(
                    v2ResAddr, mm2ResUB, v2ResAddr, softmaxExpMaxTensorAddr[runInfo.loop % DB], rowSumOldUB,
                    rowSumNewUB, mBaseSize, dTemplateAlign64);
            }
        }
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_ID_V2_C2); // V2.calc tigger C2.Fix
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(v2ResAddr, v2ResAddr, rowSumOldUB,
                                                                   runInfo.actVecMSize, dTemplateAlign64);
            }
            CopyOutAttentionOut(runInfo, v2Res, 0, runInfo.actVecMSize, runInfo.actMSize); // MTE3
            if (constInfo.isSoftmaxLseEnable) {
                SoftmaxDataCopyOut(runInfo, softmaxGlobalMaxTensorAddr[runInfo.mloop % DB], rowSumOldUB);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID_V2_MTE3_V);
        }
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX& runInfo, LocalTensor<float>& vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID_V2_ATTEN_OUT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V2_ATTEN_OUT_V_MTE3);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize, gmDealRowCount);
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<float>& vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize, gmDealRowCount);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX& info, LocalTensor<OUTPUT_T>& attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        // mxfp8 colCount 只能为64或者128，与dDealSize相等
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.realN2Idx,
                                .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                                .dIdx = 0,
                                .gS1DealSize = gmDealRowCount,
                                .dDealSize = (uint32_t)constInfo.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T>& ubTensor, GmCoordGs1Merge& gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, constInfo.dSizeV,
                                              *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, constInfo.dSizeV,
                                              *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize,
                                              constInfo.s1Size, constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize,
                                              constInfo.s1Size, constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void ProcessVec2(RunInfoX runInfo)
    {
        ProcessVec2OnUb(runInfo);
        return;
    }

    __aicore__ inline void InitTensors()
    {
        AllocEventID();

        // L1 静态分配 (与cube侧共享物理L1地址, 布局与cube InitTensors一致)
        // bank0
        static constexpr uint32_t BUFFER_NUM0 = 0;
        static constexpr uint32_t BUFFER_NUM1 = 1;
        static constexpr uint32_t BUFFER_NUM2 = 2;
        static constexpr uint32_t BUFFER_NUM3 = 3;
        static constexpr uint32_t BUFFER_NUM4 = 4;
        static constexpr uint32_t BUFFER_NUM5 = 5;
        static constexpr uint32_t BUFFER_NUM6 = 6;
        static constexpr uint32_t BUFFER_NUM7 = 7;
        static constexpr uint32_t BUFFER_NUM8 = 8;
        static constexpr uint32_t BUFFER_NUM9 = 9;
        static constexpr uint32_t BUFFER_NUM10 = 10;
        static constexpr uint32_t BUFFER_NUM11 = 11;
        static constexpr uint32_t BUFFER_NUM12 = 12;
        static constexpr uint32_t BUFFER_NUM13 = 13;
        static constexpr uint32_t BUFFER_NUM14 = 14;
        static constexpr uint32_t BUFFER_NUM15 = 15;

        static constexpr uint32_t KB_UNIT = 1024U;
        static constexpr uint32_t VSELR_IDX_NZ_GROUP = 8;
        static constexpr uint32_t STRIDE_256 = 256;
        static constexpr uint32_t SHIFT_3 = 3;

        PTensor[BUFFER_NUM0] =
            LocalTensor<uint8_t>(TPosition::A1, 117 * KB_UNIT, P_ELEMENT_SIZE); // 117: buffer address
        PTensor[BUFFER_NUM1] =
            LocalTensor<uint8_t>(TPosition::A1, 121 * KB_UNIT, P_ELEMENT_SIZE); // 121: buffer address
        PTensor[BUFFER_NUM2] =
            LocalTensor<uint8_t>(TPosition::A1, 149 * KB_UNIT, P_ELEMENT_SIZE); // 149: buffer address
        PTensor[BUFFER_NUM3] =
            LocalTensor<uint8_t>(TPosition::A1, 153 * KB_UNIT, P_ELEMENT_SIZE); // 153: buffer address
        PTensor[BUFFER_NUM4] =
            LocalTensor<uint8_t>(TPosition::A1, 181 * KB_UNIT, P_ELEMENT_SIZE); // 181: buffer address
        PTensor[BUFFER_NUM5] =
            LocalTensor<uint8_t>(TPosition::A1, 185 * KB_UNIT, P_ELEMENT_SIZE); // 185: buffer address
        PTensor[BUFFER_NUM6] =
            LocalTensor<uint8_t>(TPosition::A1, 213 * KB_UNIT, P_ELEMENT_SIZE); // 213: buffer address
        PTensor[BUFFER_NUM7] =
            LocalTensor<uint8_t>(TPosition::A1, 217 * KB_UNIT, P_ELEMENT_SIZE); // 217: buffer address
        PscaleTensor[BUFFER_NUM0] =
            LocalTensor<uint8_t>(TPosition::A1, 245 * KB_UNIT, Pscale_L1_ELEMENT_SIZE); // 245: buffer address
        // bank1 (偏移256K,即bank1起始)
        constexpr uint32_t BANK1_OFFSET = 256 * KB_UNIT; // 256: buffer address
        PTensor[BUFFER_NUM8] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 117 * KB_UNIT, P_ELEMENT_SIZE); // 117: buffer address
        PTensor[BUFFER_NUM9] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 121 * KB_UNIT, P_ELEMENT_SIZE); // 121: buffer address
        PTensor[BUFFER_NUM10] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 149 * KB_UNIT, P_ELEMENT_SIZE); // 149: buffer address
        PTensor[BUFFER_NUM11] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 153 * KB_UNIT, P_ELEMENT_SIZE); // 153: buffer address
        PTensor[BUFFER_NUM12] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 181 * KB_UNIT, P_ELEMENT_SIZE); // 181: buffer address
        PTensor[BUFFER_NUM13] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 185 * KB_UNIT, P_ELEMENT_SIZE); // 185: buffer address
        PTensor[BUFFER_NUM14] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 213 * KB_UNIT, P_ELEMENT_SIZE); // 213: buffer address
        PTensor[BUFFER_NUM15] =
            LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 217 * KB_UNIT, P_ELEMENT_SIZE); // 217: buffer address
        PscaleTensor[BUFFER_NUM1] = LocalTensor<uint8_t>(TPosition::A1, BANK1_OFFSET + 245 * KB_UNIT,
                                                         Pscale_L1_ELEMENT_SIZE); // 245: buffer address

        // UB 静态分配 (与cube侧共享物理UB地址, 布局与原InitMMResBuf一致)
        // cube fixpipe输出 (vec读)
        mm1ResUB = LocalTensor<half>(TPosition::VECCALC, 0, Mm1Res_UB_ELEMENT_SIZE);              // 0: buffer address
        mm2ResUB = LocalTensor<float>(TPosition::VECCALC, 192 * KB_UNIT, Mm2Res_UB_ELEMENT_SIZE); // 192: buffer address
        rowSumOldUB =
            LocalTensor<float>(TPosition::VECCALC, 327 * KB_UNIT, SumOld_UB_ELEMENT_SIZE); // 327: buffer address
        rowSumNewUB =
            LocalTensor<float>(TPosition::VECCALC, 328 * KB_UNIT, SumNew_UB_ELEMENT_SIZE); // 328: buffer address

        // vec自身UB
        // P (v1PRes)
        v1PRes[BUFFER_NUM1] =
            LocalTensor<uint8_t>(TPosition::VECCALC, 128 * KB_UNIT, P_ELEMENT_SIZE); // 128: buffer address
        v1PRes[BUFFER_NUM0] =
            LocalTensor<uint8_t>(TPosition::VECCALC, 128 * KB_UNIT + STRIDE_256, P_ELEMENT_SIZE); // 128: buffer address
        v1PResAddr[BUFFER_NUM1] = 128 * KB_UNIT;                                                  // 128: buffer address
        v1PResAddr[BUFFER_NUM0] = 128 * KB_UNIT + STRIDE_256;                                     // 128: buffer address
        // Pscale
        v1PscaleRes[BUFFER_NUM0] =
            LocalTensor<uint8_t>(TPosition::VECCALC, 322 * KB_UNIT, Pscale_UB_ELEMENT_SIZE); // 322: buffer address
        v1PscaleRes[BUFFER_NUM1] =
            LocalTensor<uint8_t>(TPosition::VECCALC, 324 * KB_UNIT, Pscale_UB_ELEMENT_SIZE); // 324: buffer address
        v1PscaleResAddr[BUFFER_NUM0] = 322 * KB_UNIT;                                        // 322: buffer address
        v1PscaleResAddr[BUFFER_NUM1] = 324 * KB_UNIT;                                        // 324: buffer address
        // max
        softmaxMaxTensorAddr = 320 * KB_UNIT; // 320: buffer address
        // globalmax
        softmaxGlobalMaxTensorAddr[BUFFER_NUM0] = 333 * KB_UNIT;              // 333: buffer address
        softmaxGlobalMaxTensorAddr[BUFFER_NUM1] = 333 * KB_UNIT + STRIDE_256; // 333: buffer address
        // exp max
        softmaxExpMaxTensorAddr[BUFFER_NUM0] = 326 * KB_UNIT;              // 326: buffer address
        softmaxExpMaxTensorAddr[BUFFER_NUM1] = 326 * KB_UNIT + STRIDE_256; // 326: buffer address
        // v2Res
        v2Res = LocalTensor<float>(TPosition::VECCALC, 256 * KB_UNIT, V2Res_UB_ELEMENT_SIZE); // 256: buffer address
        v2ResAddr = 256 * KB_UNIT;                                                            // 256: buffer address
        // vselrIndexes
        vselrIndexesTensor =
            LocalTensor<uint8_t>(TPosition::VECCALC, 332 * KB_UNIT, VselrIdx_SIZE); // 332: buffer address

        for (int i = 0; i < VSELR_IDX_NZ_GROUP; i++) {
            for (int j = 0; j < (VselrIdx_SIZE >> SHIFT_3); j++) {
                vselrIndexesTensor.SetValue(i * (VselrIdx_SIZE >> SHIFT_3) + j, i + (j << SHIFT_3));
            }
        }
        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseTensor = LocalTensor<float>(TPosition::VECCALC, 349 * KB_UNIT,
                                                  (mBaseSize >> 1U) * sizeof(float) * EIGHT); // 349: buffer address
        }
    }

    __aicore__ inline void ReleaseTensors()
    {
        FreeEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_ID_V1_C1_BUFFER0);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_ID_V1_C1_BUFFER1);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_ID_V1_C1_BUFFER2);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_ID_V1_C1_BUFFER3);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_ID_V2_C2); // V2.calc trigger C2.Fix

        SetFlag<HardEvent::MTE3_V>(EVENT_ID_P_MTE3_V_PING);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_P_MTE3_V_PONG);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_PSCALE_MTE3_V_PING);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_PSCALE_MTE3_V_PONG);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_V2_MTE3_V);
    }

    __aicore__ inline void FreeEventID()
    {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(FLAG_ID_C2_V1_PING); // C2.MTE1 trigger V1.MTE3
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(FLAG_ID_C2_V1_PONG);

        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_P_MTE3_V_PING);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_P_MTE3_V_PONG);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_PSCALE_MTE3_V_PING);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_PSCALE_MTE3_V_PONG);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_V2_MTE3_V);
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecHifp4Dummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecHifp4Dummy(ConstInfoX& constInfo){};
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_HIFP4_H_
