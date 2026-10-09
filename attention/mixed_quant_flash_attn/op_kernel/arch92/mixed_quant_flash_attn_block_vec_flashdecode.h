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
 * \file fia_block_vec_flashdecode.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_FLASHDECODE_H
#define FIA_BLOCK_VEC_FLASHDECODE_H

// #include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
// #include "kernel_tiling/kernel_tiling.h"
// #include "lib/matmul_intf.h"
// #include "lib/matrix/matmul/tiling.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "vf/vf_flash_decode.h"
#include "../utils/mixed_quant_flash_attn_utils.h"
#include "memory_copy_arch35_mixed_quant_flash_attn.h"

using namespace FaVectorApiAntiQuant;
// using namespace MQFA;
namespace BaseApi {
struct TaskInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

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

template <typename MQFA_T>
class FiaBlockVecFlashDecodeBase {
public:
    using ConstInfoX = ConstInfo_t<FiaKernelType::ANTI_QUANT>;

    __aicore__ inline FiaBlockVecFlashDecodeBase(ConstInfoX& constInfo){};
};

template <typename MQFA_T>
class FiaBlockVecFlashDecode {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    static constexpr GmFormat PostQuant_FORMAT = GmFormat::NGD;
    using SINK_T = typename MQFA_T::qType;
    using OUTPUT_T = typename MQFA_T::outType;
    using T = float;
    static constexpr LayOutTypeEnum layout = MQFA_T::layoutQ;
    static constexpr LayOutTypeEnum outLayout = MQFA_T::layoutOut;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr bool hasMask = MQFA_T::hasMask;
    static constexpr uint32_t mBaseSize = (uint32_t)MQFA_T::mBaseSize;
    using ConstInfoX = ConstInfo_t<FiaKernelType::ANTI_QUANT>;

private:
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;

    // TODO: 待整改为自动获取eventId
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 0;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 1;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 3;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 4;  // 此处不能给0，可能与syncall的eventId重复
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 5; // 此处不能给1，可能与syncall的eventId重复
    static constexpr uint64_t SYNC_SINK_BUF1_FLAG = 6;     // TODO 6.7不可用，暂未用到sink
    static constexpr uint64_t SYNC_SINK_BUF2_FLAG = 7;

    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_6K = 6144;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum = 2U;
    uint32_t dSizeV_Align = 0U;

protected:
    GlobalTensor<float> lseSumFdGm;
    GlobalTensor<float> lseMaxFdGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<int32_t> cuSeqLensGmQ; // TODO 未使用到，看是否需要适配
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<SINK_T> sinkGm;

    // postquant
    FaGmTensor<T, PostQuant_FORMAT> quantScale2GmTensor;
    FaGmTensor<T, PostQuant_FORMAT> quantOffset2GmTensor;
    FaGmTensor<bfloat16_t, PostQuant_FORMAT> quantScale2Bf16GmTensor;
    FaGmTensor<bfloat16_t, PostQuant_FORMAT> quantOffset2Bf16GmTensor;
    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<layout>();
    static constexpr bool isS1G = (UB_FORMAT == UbFormat::S1G);
    static constexpr bool isPa = MQFA_T::pageAttention;
    T scale2Value = 0;
    T offset2Value = 0;
    bool isQuantOffset2Exit = false;
    bool isQuant2PerChn = false;
    bool isQuant2Bf16 = false;
    float postQuantScaleValue = 0;
    float postQuantOffsetValue = 0;

    // =======================获取实际Act_S===========================
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, isPa>();
    // tensorlist
    __gm__ uint8_t* keyPtr = nullptr;
    ActualSeqLensParser<Q_MODE, int32_t> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE, int32_t> kvActSeqLensParser;

    int64_t preTokensPerBatch = 0; // TODO 改
    int64_t nextTokensPerBatch = 0;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    bool learnableSinkFlag = false;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t blockIdx = 0U;
    const ConstInfoX& constInfo;
    TaskInfo taskInfo{};

private:
    // ================================FD Local Buffer区====================================
    LocalTensor<uint8_t> fdSumBuf1;    // 1.5k: 16*24*4
    LocalTensor<uint8_t> fdSumBuf2;    // 1.5k: 16*24*4
    LocalTensor<uint8_t> fdMaxBuf1;    // 1.5k: 16*24*4
    LocalTensor<uint8_t> fdMaxBuf2;    // 1.5k: 16*24*4
    LocalTensor<uint8_t> fdLseExpBuf;  // 1.5k: 16*24*4
    LocalTensor<uint8_t> fdMm2ResBuf1; // 32k: 16*512*4
    LocalTensor<uint8_t> fdMm2ResBuf2; // 32k: 16*512*4
    LocalTensor<uint8_t> fdReduceBuf;  // 32k: 16*512*4
    LocalTensor<uint8_t> fdOutputBuf;  // 32k: 16*512*4
    LocalTensor<uint8_t> fdSinkCopyInBuf;
    LocalTensor<uint8_t> fdSinkValueBuf;
    LocalTensor<uint8_t> fdSinkExpBuf;
    LocalTensor<uint8_t> fdSinkTmpBuf;

    LocalTensor<uint8_t> fdLseMaxUbBuf1;
    LocalTensor<uint8_t> fdLseMaxUbBuf2;
    LocalTensor<uint8_t> quant2TmpBuf1;
    LocalTensor<uint8_t> quant2TmpBuf2;
    LocalTensor<uint8_t> fdLseUbBuf;

public:
    __aicore__ inline FiaBlockVecFlashDecode(ConstInfoX& constInfo)
        : constInfo(constInfo){};

    template <typename U> // 避免重名用U
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
                                            GlobalTensor<int32_t> seqUsedGmQ, GlobalTensor<int32_t> seqUsedGmKv,
                                            __gm__ uint8_t* key)
    {
        this->lseMaxFdGm = lseMaxFdGm;
        this->lseSumFdGm = lseSumFdGm;
        this->accumOutGm = accumOutGm;
        this->attentionOutGm = attentionOutGm;
        this->seqUsedGmQ = seqUsedGmQ;
        this->seqUsedGmKv = seqUsedGmKv;

        this->keyPtr = key;

        qActSeqLensParser.Init(this->seqUsedGmQ, constInfo.seqUsedQSize, constInfo.s1Size); // TODO parser待改
        kvActSeqLensParser.Init(this->seqUsedGmKv, constInfo.seqUsedKvSize, constInfo.s2Size);
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm = softmaxLseGm;
    }
    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink)
    {
        learnableSinkFlag = true;
        this->sinkGm = learnableSink;
    }
    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align = this->Align(static_cast<uint32_t>(constInfo.dSizeV), FP32_REPEAT_ELEMENT_NUM);
    }
    __aicore__ inline void InitDecodeParams()
    {
        this->blockIdx = GetBlockIdx();
    }
    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            uint32_t ubBaseAddr = 0;
            // InQue, DB, SYNC_LSE_MAX_SUM_BUF1_FLAG SYNC_LSE_MAX_SUM_BUF2_FLAG
            fdSumBuf1 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_6K); // 0
            ubBaseAddr += BUFFER_SIZE_BYTE_6K;
            fdSumBuf2 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_6K); // 6K
            ubBaseAddr += BUFFER_SIZE_BYTE_6K;
            fdMaxBuf1 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_6K); // 12K
            ubBaseAddr += BUFFER_SIZE_BYTE_6K;
            fdMaxBuf2 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_6K); // 18K
            ubBaseAddr += BUFFER_SIZE_BYTE_6K;
            fdLseExpBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_6K); // 24K
            ubBaseAddr += BUFFER_SIZE_BYTE_6K;
            fdMm2ResBuf1 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_16K); // 30K
            ubBaseAddr += BUFFER_SIZE_BYTE_16K;
            fdMm2ResBuf2 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_16K); // 46K
            ubBaseAddr += BUFFER_SIZE_BYTE_16K;
            fdReduceBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_16K); // 62K
            ubBaseAddr += BUFFER_SIZE_BYTE_16K;
            fdOutputBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_16K); // 78K
            ubBaseAddr += BUFFER_SIZE_BYTE_16K;
            fdLseMaxUbBuf1 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_256B); // 96K
            ubBaseAddr += BUFFER_SIZE_BYTE_256B;
            fdLseMaxUbBuf2 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_256B); // 96K + 256
            ubBaseAddr += BUFFER_SIZE_BYTE_256B;
            fdLseUbBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_256B); // 96K + 512
            ubBaseAddr += BUFFER_SIZE_BYTE_256B;
            quant2TmpBuf1 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_16K); // 96K + 768
            ubBaseAddr += BUFFER_SIZE_BYTE_16K;
            quant2TmpBuf2 = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_8K); // 112K + 768
            ubBaseAddr += BUFFER_SIZE_BYTE_8K;

            if (unlikely(learnableSinkFlag)) {
                fdSinkCopyInBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_2K); // 120K + 768
                ubBaseAddr += BUFFER_SIZE_BYTE_2K;
                // TmpBuf
                fdSinkValueBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_2K); // 122K + 768
                ubBaseAddr += BUFFER_SIZE_BYTE_2K;
                fdSinkTmpBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_2K); // 124K + 768
                ubBaseAddr += BUFFER_SIZE_BYTE_2K;

                fdSinkExpBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_256B); // 126K+768
                ubBaseAddr += BUFFER_SIZE_BYTE_256B;
            } else {
                fdSinkExpBuf = LocalTensor<uint8_t>(TPosition::VECIN, ubBaseAddr, BUFFER_SIZE_BYTE_256B); // 120K + 768
                ubBaseAddr += BUFFER_SIZE_BYTE_256B;
            }
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
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount)
    {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> copyInPadParams;
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align - constInfo.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align - constInfo.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset = startRow * constInfo.dSizeV +                // taskoffset + g轴offset
                                         splitKVIndex * mBaseSize * constInfo.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.ReinterpretCast<T>() : fdSumBuf2.ReinterpretCast<T>();
        LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.ReinterpretCast<T>() : fdMaxBuf2.ReinterpretCast<T>();

        uint64_t combineLseOffset = (baseOffset + startRow) * FP32_BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = mBaseSize * FP32_BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * FP32_BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }
    __aicore__ inline void ComputeScaleValue(LocalTensor<T>& lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM, uint32_t startRow)
    {
        LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.ReinterpretCast<T>() : fdSumBuf2.ReinterpretCast<T>();
        LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.ReinterpretCast<T>() : fdMaxBuf2.ReinterpretCast<T>();
        if (unlikely(learnableSinkFlag)) {
            SinkMax(startRow, dealRowCount);
        }
        LocalTensor<T> lseMaxUb =
            cntM % 2 == 0 ? fdLseMaxUbBuf1.ReinterpretCast<T>() : fdLseMaxUbBuf2.ReinterpretCast<T>();

        LocalTensor<T> sinkExpBuf = fdSinkExpBuf.ReinterpretCast<T>();
        LocalTensor<T> maxLseUb = fdLseUbBuf.ReinterpretCast<T>();
        ComputeScaleValue_VF_FD_(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount,
                                 actualCombineLoopSize, constInfo.isSoftmaxLseEnable, learnableSinkFlag);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T>& attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoordGs1Merge gmCoord{.bIdx = taskInfo.bIdx,
                                .n2Idx = taskInfo.n2Idx,
                                .gS1Idx = taskInfo.gS1Idx + startRow,
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo.dSizeV};

        if (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, seqUsedGmQ, constInfo.seqUsedQSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, seqUsedGmQ, constInfo.seqUsedQSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, seqUsedGmQ,
                                              constInfo.seqUsedQSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, seqUsedGmQ,
                                              constInfo.seqUsedQSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T>& reduceOut, LocalTensor<T>& mm2Res, LocalTensor<T>& lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align;
        ReduceFinalRes_VF_<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM)
    {
        if constexpr (POST_QUANT) {
            if (isQuant2PerChn) {
                DealPostQuantOutPerChn(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align, cntM);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                DealPostQuantOutPerTensor(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf.ReinterpretCast<OUTPUT_T>();
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (POST_QUANT) {
            DealInvalidRows(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
            DealInvalidMaskRows(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align, cntM);
        } else {
            DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align);
            DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align, cntM);
        }
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        if constexpr (!POST_QUANT) {
            uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.dSizeV};
            tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
            if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
                Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT,
                     dealRowCount * this->dSizeV_Align);
            } else {
                Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                     dealRowCount * this->dSizeV_Align);
            }
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    __aicore__ inline void CalcPreNextTokens()
    {
        actSeqLensQ = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
        if (constInfo.seqUsedKvSize == 0 && !constInfo.isKvContinuous) {
            actSeqLensKv = SeqLenFromTensorList<layout>(keyPtr, taskInfo.bIdx);
        } else {
            actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
        }
        int64_t safePreToken = constInfo.preTokens;
        int64_t safeNextToken = constInfo.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken, constInfo.sparseMode);

        if (constInfo.sparseMode == fa_base_vector_gs1::BAND) {
            preTokensPerBatch = safePreToken;
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ + safeNextToken;
        } else if ((constInfo.sparseMode == fa_base_vector_gs1::DEFAULT_MASK) && hasMask) {
            nextTokensPerBatch = safeNextToken;
            preTokensPerBatch = actSeqLensKv - actSeqLensQ + safePreToken;
        } else {
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ;
            preTokensPerBatch = 0;
        }
    }
    __aicore__ inline void CopySinkIn(uint32_t cntM)
    {
        LocalTensor<SINK_T> sinkCopyInBuf = fdSinkCopyInBuf[(cntM % 2) * BUFFER_SIZE_BYTE_1K].ReinterpretCast<SINK_T>();
        uint32_t copySize = Align(
            static_cast<uint32_t>(constInfo.gSize),
            static_cast<uint32_t>(
                BYTE_BLOCK / sizeof(SINK_T))); // DataCopy函数搬运量要求是32字节整数倍，不对齐时，将向下取整，因此该处32
                                               // byte对齐
        uint64_t sinkGmOffset = taskInfo.n2Idx * constInfo.gSize;

        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + cntM % 2);
        DataCopy(sinkCopyInBuf, sinkGm[sinkGmOffset], copySize);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINK_BUF1_FLAG + cntM % 2);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINK_BUF1_FLAG + cntM % 2);

        LocalTensor<T> tmpSinkCastBuf = fdSinkTmpBuf.ReinterpretCast<T>();
        Cast(tmpSinkCastBuf, sinkCopyInBuf, AscendC::RoundMode::CAST_NONE, constInfo.gSize);
        AscendC::PipeBarrier<PIPE_V>();

        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + cntM % 2);

        LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf.ReinterpretCast<T>();
        Brcb(sinkBrcbBuf, tmpSinkCastBuf, (constInfo.gSize + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
             {1, BLOCK_ELEMENT_NUM});
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void SinkMax(uint32_t startRow, uint32_t dealRowCount)
    {
        constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
        int64_t gIdx = 0;
        LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf.ReinterpretCast<T>();
        LocalTensor<T> sinkExpBuf = fdSinkExpBuf.ReinterpretCast<T>();

        for (int64_t row = 0; row < dealRowCount; ++row) {
            if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) { // 内存按照S1G排布
                gIdx = (taskInfo.gS1Idx + startRow + row) % constInfo.gSize;
            } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) { // 内存按照GS1排布
                int64_t actS1Size = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
                gIdx = (taskInfo.gS1Idx + startRow + row) / actS1Size;
            }
            DataCopy(sinkExpBuf[row * BLOCK_ELEMENT_NUM], sinkBrcbBuf[gIdx * BLOCK_ELEMENT_NUM], BLOCK_ELEMENT_NUM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T>& attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if (!hasMask) {
            return;
        }

        if (constInfo.sparseMode == fa_base_vector_gs1::ALL_MASK ||
            constInfo.sparseMode == fa_base_vector_gs1::LEFT_UP_CAUSAL) {
            return;
        }

        fa_base_vector::InvalidRowParams params{
            .actS1Size = actSeqLensQ,
            .gSize = static_cast<uint64_t>(constInfo.gSize),
            .gS1Idx = taskInfo.gS1Idx + startRow,
            .dealRowCount = dealRowCount,
            .columnCount = columnCount,
            .preTokensPerBatch = preTokensPerBatch,
            .nextTokensPerBatch = nextTokensPerBatch,
        };

        fa_base_vector::InvalidRows<UBOUT_T, GeInputUbFormat<layout>()> invalidRows;
        invalidRows(attenOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T>& attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if (!constInfo.isRowInvalidOpen || !hasMask) {
            return;
        }
        if (constInfo.sparseMode != fa_base_vector_gs1::DEFAULT_MASK &&
            constInfo.sparseMode != fa_base_vector_gs1::ALL_MASK) {
            return;
        }
        LocalTensor<T> lseMaxUb =
            cntM % 2 == 0 ? fdLseMaxUbBuf1.ReinterpretCast<T>() : fdLseMaxUbBuf2.ReinterpretCast<T>();

        // 这里要找到lseMaxUb 最大值为-inf 与 attenOutUb的对应位置之间的关系
        // 由于到这里的lseMaxUb 和 attenOutUb都是经过偏移后的，所以offset = 0
        // 同时，这里的lseMaxUb是经过brcb后的，所以填写true

        fa_base_vector::InvalidMaskRows<UBOUT_T, T, true>(0, dealRowCount, columnCount, lseMaxUb, negativeIntScalar,
                                                          attenOutUb);
    }

    // __aicore__ inline void InitPostQuant(__gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset)
    // {
    //     isQuant2PerChn = constInfo.isPostQuantPerChnl;
    //     isQuant2Bf16 = constInfo.isPostQuantBF16;
    //     if constexpr (POST_QUANT) {
    //         if (!isQuant2PerChn && !isQuant2Bf16) {
    //             if (postQuantScale != nullptr) {
    //                 GlobalTensor<T> postQuantScaleGm;
    //                 postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
    //                 postQuantScaleValue = postQuantScaleGm.GetValue(0);
    //             }
    //             if (postQuantOffset != nullptr) {
    //                 isQuantOffset2Exit = true;
    //                 GlobalTensor<T> postQuantOffsetGm;
    //                 postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
    //                 postQuantOffsetValue = postQuantOffsetGm.GetValue(0);
    //             } else {
    //                 postQuantOffsetValue = 0.0;
    //             }
    //         }

    //         if (!isQuant2PerChn && isQuant2Bf16) {
    //             if (postQuantScale != nullptr) {
    //                 GlobalTensor<bfloat16_t> postQuantScaleBf16Gm;
    //                 postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
    //                 postQuantScaleValue = ToFloat(postQuantScaleBf16Gm.GetValue(0));
    //             }
    //             if (postQuantOffset != nullptr) {
    //                 isQuantOffset2Exit = true;
    //                 GlobalTensor<bfloat16_t> postQuantOffsetBf16Gm;
    //                 postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
    //                 postQuantOffsetValue = ToFloat(postQuantOffsetBf16Gm.GetValue(0));
    //             } else {
    //                 postQuantOffsetValue = 0.0;
    //             }
    //         }

    //         if (isQuant2PerChn && !isQuant2Bf16) {
    //             if (postQuantScale != nullptr) {
    //                 GlobalTensor<T> postQuantScaleGm;
    //                 postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
    //                 quantScale2GmTensor.gmTensor = postQuantScaleGm;
    //                 quantScale2GmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV);
    //             }
    //             if (postQuantOffset != nullptr) {
    //                 isQuantOffset2Exit = true;
    //                 GlobalTensor<T> postQuantOffsetGm;
    //                 postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
    //                 quantOffset2GmTensor.gmTensor = postQuantOffsetGm;
    //                 quantOffset2GmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV);
    //             }
    //         }

    //         if (isQuant2PerChn && isQuant2Bf16) {
    //             if (postQuantScale != nullptr) {
    //                 GlobalTensor<bfloat16_t> postQuantScaleBf16Gm;
    //                 postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
    //                 quantScale2Bf16GmTensor.gmTensor = postQuantScaleBf16Gm;
    //                 quantScale2Bf16GmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize,
    //                 constInfo.dSizeV);
    //             }
    //             if (postQuantOffset != nullptr) {
    //                 isQuantOffset2Exit = true;
    //                 GlobalTensor<bfloat16_t> postQuantOffsetBf16Gm;
    //                 postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
    //                 quantOffset2Bf16GmTensor.gmTensor = postQuantOffsetBf16Gm;
    //                 quantOffset2Bf16GmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize,
    //                 constInfo.dSizeV);
    //             }
    //         }
    //     }
    // }

public:
    __aicore__ inline void FlashDecode(FDparamsX& fd)
    {
        if (!fd.fdCoreEnable) {
            return;
        }

        uint32_t fdBalanceMBaseSize = 8U;
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        taskInfo.bIdx = fd.fdBN2Idx / constInfo.n2Size;
        taskInfo.n2Idx = fd.fdBN2Idx % constInfo.n2Size;
        taskInfo.gS1Idx = fd.fdMIdx * mBaseSize;
        taskInfo.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * mBaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf.ReinterpretCast<T>();
            LocalTensor<T> reduceOut = fdReduceBuf.ReinterpretCast<T>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            if (unlikely(learnableSinkFlag)) {
                CopySinkIn(reduceMLoop);
            }
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; preLoadIdx++) {
                LocalTensor<T> mm2Res = ((reduceGlobaLoop + preLoadIdx) % 2) == 0 ? fdMm2ResBuf1.ReinterpretCast<T>() :
                                                                                    fdMm2ResBuf2.ReinterpretCast<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
            }
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo.actualCombineLoopSize, reduceMLoop, startRow);
            CalcPreNextTokens();
            if (constInfo.isSoftmaxLseEnable) {
                LocalTensor<T> maxLseUb = fdLseUbBuf.ReinterpretCast<T>();
                SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo.gS1Idx + startRow;
                if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
                    DataCopySoftmaxLseBSNDArch35(softmaxLseGm, maxLseUb, bN2Offset, mOffset, actualGSplitSize,
                                                 constInfo);
                } else { // BNSD
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
                    DataCopySoftmaxLseBNSDArch35(softmaxLseGm, maxLseUb, bN2Offset, mOffset, actualGSplitSize,
                                                 constInfo, qActSeqLens);
                }
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                LocalTensor<T> mm2Res =
                    reduceGlobaLoop % 2 == 0 ? fdMm2ResBuf1.ReinterpretCast<T>() : fdMm2ResBuf2.ReinterpretCast<T>();
                if (i >= preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                }
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

} // namespace BaseApi
#endif
