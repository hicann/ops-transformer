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
 * \file flash_mla_block_vec_flashdecode.h
 * \brief
 */
#ifndef FLASH_MLA_BLOCK_VEC_FLASHDECODE_H
#define FLASH_MLA_BLOCK_VEC_FLASHDECODE_H

#include "../utils/flash_mla_attenmask_gs1.h"

#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"

#include "../utils/flash_mla_memory_copy_arch35.h"
#include "flash_mla_public_define_arch35.h"
#include "../utils/flash_mla_config.h"
#include "../utils/flash_mla_type.h"

namespace FlashMlaKernel {
using ConstInfoX = AttentionCommon::ConstInfo_t<FlashMlaKernelType::NO_QUANT>;
using ConstInfo_t = ConstInfoX;

struct TaskInfoMla {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

template <typename FA_T>
class FlashMlaBlockVecFlashDecodeMla {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FLASH_MLA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FLASH_MLA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;
    using T = float;

private:
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;
    // Mutex ID（核内静态旗标）：需与 vecFaBlock_（同 AIV 核、同 PIPE）所用 0..8 错开，故从 9 起编
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 9ULL;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 11ULL;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 13ULL;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 14ULL;

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384U;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    // Each 6 KiB FD state slot holds 192 rows across all KV splits (8 FP32 values per row).
    static constexpr uint32_t FD_STATE_ROWS = 192U;
    static constexpr uint32_t FD_MAX_BATCH_ROWS = 8U;

    static __aicore__ inline constexpr uint32_t GetFdBatchRows(uint32_t splits)
    {
        if (splits == 0U) {
            return 0U;
        }
        // The FD VF loads 256 B even when fewer than eight rows are active.
        // For r rows, the final load ends at ((splits-1)*r+8)*32 bytes.
        const uint32_t rows = splits == 1U ? FD_MAX_BATCH_ROWS : (FD_STATE_ROWS - FD_MAX_BATCH_ROWS) / (splits - 1U);
        return rows < FD_MAX_BATCH_ROWS ? rows : FD_MAX_BATCH_ROWS;
    }

    static constexpr uint32_t PRE_LOAD_NUM = 2U;
    uint32_t dSizeV_Align_;

protected:
    GlobalTensor<float> lseSumFdGm_;
    GlobalTensor<float> lseMaxFdGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxLseGm_;

    int64_t winLeftPerBatch_ = 0;
    int64_t winRightPerBatch_ = 0;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar_ = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    const ConstInfo_t& constInfo_;
    TaskInfoMla taskInfo_{};

    using SEQLEN_T = uint32_t;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>();
    using SeqLensToolType = FlashMlaSeqLensTool<Q_MODE, KV_MODE>;
    SeqLensToolType& seqLensTool_;

    static constexpr GmFormat OUT_FORMAT = GetQueryGmFormat<LAYOUT_OUT>();
    // WITH_ZERO_HEAD 由输入布局决定，TND→NTD 输出转置不能改变序列解析方式。
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, (LAYOUT_T == FLASH_MLA_LAYOUT::TND)>;
    FaGmTensorOut outGmTensor_;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm_;

private:
    LocalTensor<T> fdSumBuf1_;          // 6 KiB；实际占用 parts*dealRowCount*8*sizeof(float)
    LocalTensor<T> fdSumBuf2_;          // 6 KiB；实际占用 parts*dealRowCount*8*sizeof(float)
    LocalTensor<T> fdMaxBuf1_;          // 6 KiB；实际占用 parts*dealRowCount*8*sizeof(float)
    LocalTensor<T> fdMaxBuf2_;          // 6 KiB；实际占用 parts*dealRowCount*8*sizeof(float)
    LocalTensor<T> fdLseExpBuf_;        // 6 KiB；实际占用 parts*dealRowCount*8*sizeof(float)
    LocalTensor<T> fdMm2ResBuf1_;       // 16 KiB：8*512*sizeof(float)
    LocalTensor<T> fdMm2ResBuf2_;       // 16 KiB：8*512*sizeof(float)
    LocalTensor<T> fdReduceBuf_;        // 16 KiB：8*512*sizeof(float)
    LocalTensor<OUTPUT_T> fdOutputBuf_; // 8 KiB：8*512*sizeof(OUTPUT_T)，输出元素 2 字节

    LocalTensor<T> fdLseMaxUbBuf1_;
    LocalTensor<T> fdLseMaxUbBuf2_;
    LocalTensor<T> fdLseUbBuf_;

public:
    __aicore__ inline FlashMlaBlockVecFlashDecodeMla(ConstInfo_t& constInfo, SeqLensToolType& seqLensTool)
        : constInfo_(constInfo),
          seqLensTool_(seqLensTool){};

    template <typename U> // 避免重名用U
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitBlock(__gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut)
    {
        this->dSizeV_Align_ = this->Align(constInfo_.dSizeV, FP32_REPEAT_ELEMENT_NUM);

        InitAttenOutBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSizeV,
                           outGmTensor_, attentionOut);

        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }
    }

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm)
    {
        this->lseMaxFdGm_ = lseMaxFdGm;
        this->lseSumFdGm_ = lseSumFdGm;
        this->accumOutGm_ = accumOutGm;
    }

    template <uint32_t UB_FD_BASE_OFFSET>
    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // 跳过 bmm1/bmm2 CV 区后放置 FD 业务区，空间上避免跨 section 时 AIC 的 FA 写 mm2Res 与 FD 使用冲突。
            constexpr uint32_t BASE = UB_FD_BASE_OFFSET;
            constexpr uint32_t STRIDE_6K = BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K; // 6144
            fdSumBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE, STRIDE_6K).template ReinterpretCast<T>();
            fdSumBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, BASE + STRIDE_6K, STRIDE_6K).template ReinterpretCast<T>();
            fdMaxBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, BASE + 2U * STRIDE_6K, STRIDE_6K).template ReinterpretCast<T>();
            fdMaxBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, BASE + 3U * STRIDE_6K, STRIDE_6K).template ReinterpretCast<T>();
            fdLseExpBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, BASE + 4U * STRIDE_6K, STRIDE_6K).template ReinterpretCast<T>();
            constexpr uint32_t LSE_SUM_MAX_BYTES = BASE + 5U * STRIDE_6K;
            fdMm2ResBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, LSE_SUM_MAX_BYTES, BUFFER_SIZE_BYTE_16K)
                                .template ReinterpretCast<T>();
            fdMm2ResBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, LSE_SUM_MAX_BYTES + BUFFER_SIZE_BYTE_16K, BUFFER_SIZE_BYTE_16K)
                    .template ReinterpretCast<T>();
            fdReduceBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, LSE_SUM_MAX_BYTES + 2U * BUFFER_SIZE_BYTE_16K,
                                                BUFFER_SIZE_BYTE_16K)
                               .template ReinterpretCast<T>();
            fdOutputBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, LSE_SUM_MAX_BYTES + 3U * BUFFER_SIZE_BYTE_16K,
                                                BUFFER_SIZE_BYTE_8K)
                               .template ReinterpretCast<OUTPUT_T>();
            constexpr uint32_t LSE_SUM_MAX_MM2_BYTES =
                LSE_SUM_MAX_BYTES + 3U * BUFFER_SIZE_BYTE_16K + BUFFER_SIZE_BYTE_8K;
            fdLseMaxUbBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, LSE_SUM_MAX_MM2_BYTES, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseMaxUbBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, LSE_SUM_MAX_MM2_BYTES + BUFFER_SIZE_BYTE_256B,
                                                   BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseUbBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, LSE_SUM_MAX_MM2_BYTES + 2U * BUFFER_SIZE_BYTE_256B,
                                               BUFFER_SIZE_BYTE_256B)
                              .template ReinterpretCast<T>();
#if __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202
            static_assert(LSE_SUM_MAX_MM2_BYTES + 3U * BUFFER_SIZE_BYTE_256B <= FlashMlaConfig::ARCH92.ubBytes,
                          "FlashDecode exceeds arch92 UB budget");
#else
            static_assert(LSE_SUM_MAX_MM2_BYTES + 3U * BUFFER_SIZE_BYTE_256B <= FlashMlaConfig::ARCH35.ubBytes,
                          "FlashDecode exceeds arch35 UB budget");
#endif
        }
    }

protected:
    __aicore__ inline void InitAttenOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                              uint32_t headDim, FaGmTensorOut& outGmTensor, __gm__ uint8_t* gm)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUTPUT_T*)gm);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim,
                                              seqLensTool_.qActSeqLensParser_);
        } else {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, seqLensTool_.qActSeqLensParser_);
        }
    }

    __aicore__ inline void CopyAccumOutIn(LocalTensor<T>& accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
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
        uint64_t combineAccumOutOffset = startRow * constInfo_.dSizeV +                // taskoffset + g轴offset
                                         splitKVIndex * mBaseSize * constInfo_.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm_[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_ : fdSumBuf2_;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_ : fdMaxBuf2_;

        uint64_t combineLseOffset = (baseOffset + startRow) * FP32_BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = mBaseSize * FP32_BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * FP32_BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; i++) {
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

        LocalTensor<T> sinkExpBuf;
        ComputeScaleValue_VF_FD<T, T, SinkInjectStage::AT_FA_PROLOGUE>(
            sinkExpBuf, lseMax, lseSum, lseExp, fdLseUbBuf_, lseMaxUb, dealRowCount, actualCombineLoopSize,
            constInfo_.isSoftmaxLseEnable, false, -FLT_MAX_NEW);
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
        copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T>& reduceOut, LocalTensor<T>& mm2Res, LocalTensor<T>& lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, static_cast<uint64_t>(dSizeV_Align_), cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf_;
        AscendC::PipeBarrier<PIPE_V>();
        DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align_);
        DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align_, cntM);
        Mutex::Lock<PIPE_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo_.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT,
                 dealRowCount * this->dSizeV_Align_);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                 dealRowCount * this->dSizeV_Align_);
        }
        Mutex::Unlock<PIPE_V>(SYNC_FDOUTPUT_BUF_FLAG);
        Mutex::Lock<PIPE_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align_);
        Mutex::Unlock<PIPE_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    __aicore__ inline void CalcWindowPerBatch()
    {
        actSeqLensQ_ = seqLensTool_.qActSeqLensParser_.GetActualSeqLength(taskInfo_.bIdx);
        actSeqLensKv_ = seqLensTool_.kvActSeqLensParser_.GetActualSeqLength(taskInfo_.bIdx);

        int64_t safeWinLeft = constInfo_.winLeft;
        int64_t safeWinRight = constInfo_.winRight;

        fa_base_vector::GetSafeActToken(actSeqLensQ_, actSeqLensKv_, safeWinLeft, safeWinRight, constInfo_.sparseMode);

        if (constInfo_.sparseMode == BAND) {
            winLeftPerBatch_ = safeWinLeft;
            winRightPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safeWinRight;
        } else if ((constInfo_.sparseMode == DEFAULT_MASK) && HAS_MASK) {
            winRightPerBatch_ = safeWinRight;
            winLeftPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safeWinLeft;
        } else {
            winRightPerBatch_ = actSeqLensKv_ - actSeqLensQ_;
            winLeftPerBatch_ = 0;
        }
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T>& attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if constexpr (!HAS_MASK) {
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
            .preTokensPerBatch = winLeftPerBatch_,
            .nextTokensPerBatch = winRightPerBatch_,
        };

        fa_base_vector::InvalidRows<UBOUT_T, AttentionCommon::GeInputUbFormat<LAYOUT_T>()> invalidRows;
        invalidRows(attenOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T>& attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if constexpr (!HAS_MASK) {
            return;
        }
        if (constInfo_.sparseMode != DEFAULT_MASK && constInfo_.sparseMode != ALL_MASK) {
            return;
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_ : fdLseMaxUbBuf2_;

        fa_base_vector::InvalidMaskRows<UBOUT_T, T, true>(0, dealRowCount, columnCount, lseMaxUb, negativeIntScalar_,
                                                          attenOutUb);
    }

public:
    __aicore__ inline void FlashDecode(FlashMlaFdParamsX& fd)
    {
        if (!fd.fdCoreEnable) {
            return;
        }
        uint32_t fdBalanceMBaseSize = GetFdBatchRows(fd.fdS2SplitNum);
        // Bound both packed states and the last unmasked 256-byte VF read in each 6 KiB slot.
        if (fd.fdS2SplitNum == 0U || fdBalanceMBaseSize == 0U || fd.mLen == 0U) {
            return;
        }
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        taskInfo_.bIdx = fd.fdBN2Idx / constInfo_.n2Size;
        taskInfo_.n2Idx = fd.fdBN2Idx % constInfo_.n2Size;
        taskInfo_.gS1Idx = fd.fdMIdx * mBaseSize;
        taskInfo_.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t taskOffset = static_cast<uint64_t>(fd.fdWorkspaceIdx) * mBaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭
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
            for (uint32_t preLoadIdx = 0; preLoadIdx < PRE_LOAD_NUM; preLoadIdx++) {
                LocalTensor<T> mm2Res = (((reduceGlobaLoop + preLoadIdx) & 1) == 0) ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            Mutex::Lock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            Mutex::Lock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo_.actualCombineLoopSize, reduceMLoop);
            Mutex::Unlock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            Mutex::Unlock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CalcWindowPerBatch();
            if (constInfo_.isSoftmaxLseEnable) {
                LocalTensor<T> maxLseUb = fdLseUbBuf_;
                Mutex::Lock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo_.gS1Idx + startRow;
                if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::TND) {
                    uint32_t prefixBS1 = seqLensTool_.qActSeqLensParser_.GetTBase(taskInfo_.bIdx);
                    uint64_t bN2Offset = taskInfo_.n2Idx * constInfo_.gSize * constInfo_.t1Size;
                    DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfo_t>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                    actualGSplitSize, prefixBS1, constInfo_);
                } else if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::BSH) {
                    uint64_t bN2Offset = taskInfo_.bIdx * constInfo_.gSize * constInfo_.n2Size * constInfo_.s1Size +
                                         taskInfo_.n2Idx * constInfo_.gSize * constInfo_.s1Size;
                    DataCopySoftmaxLseBSNDArch35<T, ConstInfo_t>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                 actualGSplitSize, constInfo_);
                } else if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::BNSD) {
                    uint64_t bN2Offset = taskInfo_.bIdx * constInfo_.gSize * constInfo_.n2Size * constInfo_.s1Size +
                                         taskInfo_.n2Idx * constInfo_.gSize * constInfo_.s1Size;
                    uint64_t qActSeqLens = seqLensTool_.qActSeqLensParser_.GetActualSeqLength(taskInfo_.bIdx);
                    DataCopySoftmaxLseBNSDArch35<T, ConstInfo_t>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                 actualGSplitSize, constInfo_, qActSeqLens);
                }
                Mutex::Unlock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
            }

            for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; i++) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                if (i >= PRE_LOAD_NUM) {
                    Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                Mutex::Lock<PIPE_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                Mutex::Unlock<PIPE_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

} // namespace FlashMlaKernel
#endif
