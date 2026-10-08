/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_block_vec_flashdecode_fp8.h
 * \brief QuantFlashMlaWithKvcache FD阶段vec block（自FIA MLA/QFA FD裁剪适配，
 *        规约workspace中间结果到输出，适配int32序列解析与多section metadata调度）
 */

#ifndef QUANT_FLASH_MLA_BLOCK_VEC_FLASHDECODE_FP8_H_
#define QUANT_FLASH_MLA_BLOCK_VEC_FLASHDECODE_FP8_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#if __has_include("../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h")
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#else
#include "../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#endif
#include "../../../common/op_kernel/vector_common.h"
#include "memory_copy_arch35_quant_flash_mla_with_kvcache.h"
#include "quant_flash_mla_with_kvcache_public_def.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AttentionCommon;

namespace BaseApi {

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
    LayOutTypeEnum outLayout = LayOutTypeEnum::LAYOUT_TND, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned576,
    DTemplateType dVTemplateType = DTemplateType::Aligned512>
class QuantFlashMlaBlockVecFlashDecodeFp8 {
public:
    // =================================类型定义区=================================
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
    // Mutex ID（核内跨流水互斥，对齐flash_attn FD双buffer轮转方案，不占用EventID池，
    // 避免与FA vec block的AllocEventID分配冲突；qfmla kernel内无其他Mutex用户）
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 2;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 4;

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum = 2U;
    uint32_t dSizeV_Align;
    using ConstInfoX = QmlaConstInfo;
    // 基本块大小（与metadata分核mBaseSize=64一致）
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2FdBaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dVFdBaseSize = (uint32_t)dVTemplateType;

    // ================================UB绝对偏移布局区=================================
    // 跨核bmm通信区位于UB起始（kernel InitMMResBuf最先经ubBufferManager分配）:
    // bmm1/bmm2均双buffer = (s1BaseSize/2) * (2*dVBaseSize + 2*s2BaseSize) * sizeof(T) = 160K
    // FD工作区紧随其后，覆盖FA vec block死区（同核上FA/FD阶段串行，FD前后SyncAll保证流水排空）
    static constexpr uint32_t mm1ResSize = (s1BaseSize / 2U) * s2FdBaseSize * sizeof(T); // 16K
    static constexpr uint32_t mm2ResSize = (s1BaseSize / 2U) * dVFdBaseSize * sizeof(T); // 64K
    static constexpr uint32_t FD_BASE = mm2ResSize * 2U + mm1ResSize;                    // 160K
    // FD终点 = FD_BASE + 5*6144 + 4*16K + 4*256 = 261120B < 262144B(256K UB)
    static constexpr uint32_t LSE_STRIDE = BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K; // 6144
    static constexpr uint32_t OFF_FD_SUM1 = FD_BASE;
    static constexpr uint32_t OFF_FD_SUM2 = FD_BASE + LSE_STRIDE;
    static constexpr uint32_t OFF_FD_MAX1 = FD_BASE + 2U * LSE_STRIDE;
    static constexpr uint32_t OFF_FD_MAX2 = FD_BASE + 3U * LSE_STRIDE;
    static constexpr uint32_t OFF_FD_LSEEXP = FD_BASE + 4U * LSE_STRIDE;
    static constexpr uint32_t OFF_FD_MM2RES1 = FD_BASE + 5U * LSE_STRIDE;
    static constexpr uint32_t OFF_FD_MM2RES2 = OFF_FD_MM2RES1 + BUFFER_SIZE_BYTE_16K;
    static constexpr uint32_t OFF_FD_REDUCE = OFF_FD_MM2RES2 + BUFFER_SIZE_BYTE_16K;
    static constexpr uint32_t OFF_FD_OUTPUT = OFF_FD_REDUCE + BUFFER_SIZE_BYTE_16K;
    static constexpr uint32_t OFF_FD_LSEMAXUB1 = OFF_FD_OUTPUT + BUFFER_SIZE_BYTE_16K;
    static constexpr uint32_t OFF_FD_LSEMAXUB2 = OFF_FD_LSEMAXUB1 + BUFFER_SIZE_BYTE_256B;
    static constexpr uint32_t OFF_FD_LSEUB = OFF_FD_LSEMAXUB2 + BUFFER_SIZE_BYTE_256B;
    static constexpr uint32_t OFF_FD_SINKEXP = OFF_FD_LSEUB + BUFFER_SIZE_BYTE_256B;

protected:
    GlobalTensor<float> lseSumFdGm_;
    GlobalTensor<float> lseMaxFdGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;

    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<layout>();

    // Q: TND, cu_seq(含前导0)+seq_used, int32（与FA vec block保持一致的解析器）
    using QSeqParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>;
    QSeqParserType* qActSeqLensParser_ = nullptr;

    // ================================类成员变量====================================
    // 结构体
    const ConstInfoX& constInfo_;
    TaskInfo taskInfo_{};

private:
    // ================================FD Local Buffer区====================================
    // 绝对偏移LocalTensor（不走TPipe分配器，见上方UB绝对偏移布局区）
    LocalTensor<T> fdSumBuf1_;          // 1.5k: 16*24*4
    LocalTensor<T> fdSumBuf2_;          // 1.5k: 16*24*4
    LocalTensor<T> fdMaxBuf1_;          // 1.5k: 16*24*4
    LocalTensor<T> fdMaxBuf2_;          // 1.5k: 16*24*4
    LocalTensor<T> fdLseExpBuf_;        // 1.5k: 16*24*4
    LocalTensor<T> fdMm2ResBuf1_;       // 16k: 8*512*4
    LocalTensor<T> fdMm2ResBuf2_;       // 16k: 8*512*4
    LocalTensor<T> fdReduceBuf_;        // 16k: 8*512*4
    LocalTensor<OUTPUT_T> fdOutputBuf_; // 16k: 8*512*4

    LocalTensor<T> fdLseMaxUbBuf1_;
    LocalTensor<T> fdLseMaxUbBuf2_;
    LocalTensor<T> fdLseUbBuf_;
    LocalTensor<T> fdSinkExpBuf_;

public:
    __aicore__ inline QuantFlashMlaBlockVecFlashDecodeFp8(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm)
    {
        this->lseMaxFdGm_ = lseMaxFdGm;
        this->lseSumFdGm_ = lseSumFdGm;
        this->accumOutGm_ = accumOutGm;
        this->attentionOutGm_ = attentionOutGm;
    }

    __aicore__ inline void SetCuSeqLensParsers(QSeqParserType& qParser)
    {
        this->qActSeqLensParser_ = &qParser;
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm_ = softmaxLseGm;
    }

    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align = AttentionCommon::Align(constInfo_.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // 与FA block共享UB布局：跨核bmm区在前（必须保留，AIC可能在FD阶段访问下一section），
            // FD业务缓冲区紧随其后覆盖FA vec死区。使用LocalTensor构造函数直接指定绝对字节偏移，
            // 完全自主管理内存，不走TPipe分配器。
            // 严禁pipe->Reset()/pipe->InitBuffer：会摧毁FA阶段注册的跨核TQue表项与flag记账，
            // 导致AIC死等已销毁的响应flag（死锁）及UB跨核区覆写竞争（aicore error）。
            fdSumBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_SUM1, LSE_STRIDE).template ReinterpretCast<T>();
            fdSumBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_SUM2, LSE_STRIDE).template ReinterpretCast<T>();
            fdMaxBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_MAX1, LSE_STRIDE).template ReinterpretCast<T>();
            fdMaxBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_MAX2, LSE_STRIDE).template ReinterpretCast<T>();
            fdLseExpBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_LSEEXP, LSE_STRIDE).template ReinterpretCast<T>();
            fdMm2ResBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_MM2RES1, BUFFER_SIZE_BYTE_16K)
                                .template ReinterpretCast<T>();
            fdMm2ResBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_MM2RES2, BUFFER_SIZE_BYTE_16K)
                                .template ReinterpretCast<T>();
            fdReduceBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_REDUCE, BUFFER_SIZE_BYTE_16K)
                               .template ReinterpretCast<T>();
            fdOutputBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_OUTPUT, BUFFER_SIZE_BYTE_16K)
                               .template ReinterpretCast<OUTPUT_T>();
            fdLseMaxUbBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_LSEMAXUB1, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseMaxUbBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_LSEMAXUB2, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseUbBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_LSEUB, BUFFER_SIZE_BYTE_256B)
                              .template ReinterpretCast<T>();
            fdSinkExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_FD_SINKEXP, BUFFER_SIZE_BYTE_256B)
                                .template ReinterpretCast<T>();
        }
    }

protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount)
    {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> copyInPadParams;
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = constInfo_.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align - constInfo_.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align - constInfo_.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset = startRow * constInfo_.dSizeV +                 // taskoffset + g轴offset
                                         splitKVIndex * s1BaseSize * constInfo_.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm_[combineAccumOutOffset], copyInParams, copyInPadParams);
    }

    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_ : fdSumBuf2_;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_ : fdMaxBuf2_;

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

    __aicore__ inline void ComputeScaleValue(LocalTensor<T>& lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_ : fdSumBuf2_;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_ : fdMaxBuf2_;
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_ : fdLseMaxUbBuf2_;

        LocalTensor<T> sinkExpBuf = fdSinkExpBuf_;
        LocalTensor<T> maxLseUb = fdLseUbBuf_;
        ComputeScaleValue_VF_FD(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount,
                                actualCombineLoopSize, constInfo_.isSoftmaxLseEnable, false);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T>& attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoordGs1Merge gmCoord{.bIdx = taskInfo_.bIdx,
                                .n2Idx = taskInfo_.n2Idx,
                                .gS1Idx = taskInfo_.gS1Idx + startRow,
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T>& ubTensor, GmCoordGs1Merge& gmCoord)
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
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else { // BNSD
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void ReduceFinalRes(LocalTensor<T>& reduceOut, LocalTensor<T>& mm2Res, LocalTensor<T>& lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align;
        ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
    }

    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf_;
        AscendC::PipeBarrier<PIPE_V>();
        Mutex::Lock<PIPE_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo_.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * this->dSizeV_Align);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                 dealRowCount * this->dSizeV_Align);
        }
        Mutex::Unlock<PIPE_V>(SYNC_FDOUTPUT_BUF_FLAG);
        Mutex::Lock<PIPE_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
        Mutex::Unlock<PIPE_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    }

    __aicore__ inline void CopySoftmaxLseOutput(uint32_t actualGSplitSize, uint32_t startRow)
    {
        if (constInfo_.isSoftmaxLseEnable) {
            // lse行无效在ComputeScaleValue的VF计算时已经进行了赋值inf处理
            LocalTensor<T> maxLseUb = fdLseUbBuf_;
            Mutex::Lock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
            uint32_t mOffset = taskInfo_.gS1Idx + startRow;
            // 新接口LSE输出统一为NT格式[N2*G, T]，与FA vec block一致
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(taskInfo_.bIdx);
            uint64_t bN2Offset = taskInfo_.n2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                     actualGSplitSize, constInfo_);
            Mutex::Unlock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
        }
    }

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
        // MLA: realN2Size = n2Size = 1, bIdx即fdBN2Idx
        taskInfo_.bIdx = fd.fdBN2Idx / constInfo_.realN2Size;
        taskInfo_.n2Idx = fd.fdBN2Idx % constInfo_.realN2Size;
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

            LocalTensor<T> lseExp = fdLseExpBuf_;
            LocalTensor<T> reduceOut = fdReduceBuf_;
            Mutex::Lock<PIPE_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            Mutex::Unlock<PIPE_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; ++preLoadIdx) {
                LocalTensor<T> mm2Res = ((reduceGlobaLoop + preLoadIdx) & 1) == 0 ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            Mutex::Lock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            Mutex::Lock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo_.actualCombineLoopSize, reduceMLoop);
            Mutex::Unlock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            Mutex::Unlock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopySoftmaxLseOutput(actualGSplitSize, startRow);

            for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                if (i >= preLoadNum) {
                    Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                Mutex::Lock<PIPE_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                Mutex::Unlock<PIPE_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize);
            reduceMLoop += 1;
        }
    }
};

// AIC侧编译占位（避免cube编译单元实例化FD vec逻辑）
template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
    LayOutTypeEnum outLayout = LayOutTypeEnum::LAYOUT_TND, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned576,
    DTemplateType dVTemplateType = DTemplateType::Aligned512>
class QuantFlashMlaBlockVecFlashDecodeFp8Dummy {
public:
    using ConstInfoX = QmlaConstInfo;
    __aicore__ inline QuantFlashMlaBlockVecFlashDecodeFp8Dummy(ConstInfoX& constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_MLA_BLOCK_VEC_FLASHDECODE_FP8_H_
