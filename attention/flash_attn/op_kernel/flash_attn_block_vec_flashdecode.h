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
 * \file flash_attn_block_vec_flashdecode.h
 * \brief
 */
#ifndef FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
#define FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H

#include "utils/flash_attn_common_def.h"
#if __has_include("../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h")
#include "../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#else
#include "../../common/arch35/vf/vf_flash_decode_arch35.h"
#endif
#include "memory_copy_arch35.h"
#include "../../common/op_kernel/vector_common.h"

namespace FlashAttnKernel {
struct TaskInfo {
    uint32_t bIdx;
    uint32_t n1Idx; // 不合轴: 当前 query 头(DN_LAYOUT 使用)
    uint32_t n2Idx;
    uint32_t gIdx;   // 不合轴: 头内 G 下标(DN_LAYOUT 使用)
    uint32_t s1Idx;  // 纯 s1 坐标(DN_LAYOUT 使用)
    uint32_t gS1Idx; // 合轴 gs1 坐标(非 DN_LAYOUT 使用)
    uint32_t actualCombineLoopSize;
};

template <FA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr fa_base_vector::UbInputFormat GeInputUbFormat()
{
    static_assert((LAYOUT_T == FA_LAYOUT::BSND) || (LAYOUT_T == FA_LAYOUT::BNSD) || (LAYOUT_T == FA_LAYOUT::TND),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FA_LAYOUT::TND || LAYOUT_T == FA_LAYOUT::BSND) {
        return fa_base_vector::UbInputFormat::S1G;
    } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
        return fa_base_vector::UbInputFormat::GS1;
    }
}

template <typename FA_T, bool DN_LAYOUT = false>
class FiaBlockVecFlashDecode {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;
    // =================================类型定义区=================================
    using T = float;
    using SINK_T = INPUT_T;

private:
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 2;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 4;

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);

    // FD 每轮规约的行数（与 ReduceFd 里的 fdBalanceMBaseSize 同源）
    static constexpr uint32_t FD_BALANCE_M_BASE = 8U;
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum_ = 2U;
    uint32_t dSizeV_Align_;

protected:
    GlobalTensor<float> lseSumFdGm_;
    GlobalTensor<float> lseMaxFdGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxLseGm_;

    static constexpr UbFormat UB_FORMAT = DN_LAYOUT ? UbFormat::S1_ONLY : GetOutUbFormat<LAYOUT_T>();
    int64_t preTokensPerBatch_ = 0;
    int64_t nextTokensPerBatch_ = 0;

    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    // ================================类成员变量====================================
    const ConstInfo_t& constInfo_;
    TaskInfo taskInfo_{};

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool_;

    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensor_;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, UB_FORMAT> copyAttenOutUbToGm_;

private:
    // ================================FD Local Buffer区====================================
    LocalTensor<T> fdSumBuf1_;          // FD_LSE_BUF_BYTES（24 splits × 8 行 × 8 float）
    LocalTensor<T> fdSumBuf2_;          // 同上（cntM 奇偶双缓冲）
    LocalTensor<T> fdMaxBuf1_;          // 同上
    LocalTensor<T> fdMaxBuf2_;          // 同上
    LocalTensor<T> fdLseExpBuf_;        // 同上
    LocalTensor<T> fdMm2ResBuf1_;       // 8 行 × Align(dV,64) × 4B
    LocalTensor<T> fdMm2ResBuf2_;       // 同上（preLoadNum_ 预取双缓冲）
    LocalTensor<T> fdReduceBuf_;        // 同上
    LocalTensor<OUTPUT_T> fdOutputBuf_; // 8 行 × Align(dV,64) × sizeof(OUTPUT_T)

    LocalTensor<T> fdLseMaxUbBuf1_;
    LocalTensor<T> fdLseMaxUbBuf2_;
    LocalTensor<T> fdLseUbBuf_;

public:
    __aicore__ inline FiaBlockVecFlashDecode(ConstInfo_t& constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool,
                                             SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    template <typename U> // 避免重名用U
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitBlock(__gm__ uint8_t* learnableSink, __gm__ uint8_t* softmaxLse,
                                     __gm__ uint8_t* attentionOut)
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
            // FD 业务区从 matmul 输出区之后开始，区内布局与尺寸由 FD 自管；不得回写 matmul 输出区
            // （AIC 不参与 SyncAll，FD 运行期间 AIC 可能正在写该区）。
            // 各块用绝对字节偏移直接构造 LocalTensor，不依赖线性分配器。
            constexpr uint32_t BASE = UB_FD_BASE_OFFSET;

            // lse 槽沿用原 6,144B：每个 split = FD_BALANCE_M_BASE 行 × FP32_BLOCK_ELEMENT_NUM 个 float，
            // 故 FD_SPLIT_MAX = 6,144 / 256 = 24
            constexpr uint32_t FD_DV_ALIGN64 = ((uint32_t)dVBaseSize + 63U) / 64U * 64U;
            constexpr uint32_t FD_SPLIT_MAX = 24U;
            constexpr uint32_t FD_SPLIT_LSE_BYTES = FD_BALANCE_M_BASE * FP32_BLOCK_ELEMENT_NUM * sizeof(T);
            constexpr uint32_t FD_LSE_BUF_BYTES = FD_SPLIT_MAX * FD_SPLIT_LSE_BYTES;
            constexpr uint32_t FD_MM2_BUF_BYTES = FD_BALANCE_M_BASE * FD_DV_ALIGN64 * sizeof(T);
            constexpr uint32_t FD_OUT_BUF_BYTES = FD_BALANCE_M_BASE * FD_DV_ALIGN64 * sizeof(OUTPUT_T);

            // 区内偏移
            constexpr uint32_t OFF_LSEMAX1 = BASE;
            constexpr uint32_t OFF_LSEMAX2 = OFF_LSEMAX1 + BUFFER_SIZE_BYTE_256B;
            constexpr uint32_t OFF_LSEUB = OFF_LSEMAX2 + BUFFER_SIZE_BYTE_256B;
            constexpr uint32_t OFF_SUM1 = OFF_LSEUB + BUFFER_SIZE_BYTE_256B;
            constexpr uint32_t OFF_SUM2 = OFF_SUM1 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MAX1 = OFF_SUM2 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MAX2 = OFF_MAX1 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_LSEEXP = OFF_MAX2 + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MM2RES1 = OFF_LSEEXP + FD_LSE_BUF_BYTES;
            constexpr uint32_t OFF_MM2RES2 = OFF_MM2RES1 + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_REDUCE = OFF_MM2RES2 + FD_MM2_BUF_BYTES;
            constexpr uint32_t OFF_OUTPUT = OFF_REDUCE + FD_MM2_BUF_BYTES;

            fdLseMaxUbBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEMAX1, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseMaxUbBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEMAX2, BUFFER_SIZE_BYTE_256B)
                                  .template ReinterpretCast<T>();
            fdLseUbBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEUB, BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();

            fdSumBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SUM1, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdSumBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_SUM2, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdMaxBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MAX1, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdMaxBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MAX2, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();
            fdLseExpBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSEEXP, FD_LSE_BUF_BYTES).template ReinterpretCast<T>();

            fdMm2ResBuf1_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MM2RES1, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdMm2ResBuf2_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_MM2RES2, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdReduceBuf_ =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_REDUCE, FD_MM2_BUF_BYTES).template ReinterpretCast<T>();
            fdOutputBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFF_OUTPUT, FD_OUT_BUF_BYTES)
                               .template ReinterpretCast<OUTPUT_T>();
        }
    }

protected:
    __aicore__ inline void InitAttenOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                              uint32_t headDim, FaGmTensorOut& outGmTensor, __gm__ uint8_t* gm)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUTPUT_T*)gm);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool_.seqUsedParser);
        } else {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool_.cuSeqLensParser);
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
                                             uint32_t actualCombineLoopSize, uint32_t cntM, uint32_t startRow)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_ : fdSumBuf2_;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_ : fdMaxBuf2_;
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_ : fdLseMaxUbBuf2_;

        LocalTensor<T> sinkExpBuf;
        LocalTensor<T> maxLseUb = fdLseUbBuf_;
        ComputeScaleValue_VF_FD<T, T, SinkInjectStage::AT_FA_PROLOGUE>(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb,
                                                                       lseMaxUb, dealRowCount, actualCombineLoopSize,
                                                                       constInfo_.isSoftmaxLseEnable, false);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T>& attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        if constexpr (DN_LAYOUT) {
            // 不合轴 DN 使用 S1_ONLY，拷贝不按 ROW_PAD 分支，保留默认张量类型。
            FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = columnCount};
            GmCoordS1Only gmCoord{.bIdx = taskInfo_.bIdx,
                                  .n2Idx = taskInfo_.n2Idx,
                                  .gIdx = taskInfo_.gIdx,
                                  .s1Idx = taskInfo_.s1Idx + startRow,
                                  .dIdx = 0,
                                  .s1DealSize = dealRowCount,
                                  .dDealSize = (uint32_t)constInfo_.dSizeV};
            copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
        } else {
            // 合轴 S1G 输出拷贝需要 ROW_PAD 区分是否可合并中间多行。
            constexpr bool rowPad = (static_cast<uint32_t>(FA_T::originalDV) % 16 != 0);
            FaUbTensor<OUTPUT_T, rowPad> ubTensor{
                .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = columnCount};
            GmCoordGs1Merge gmCoord{.bIdx = taskInfo_.bIdx,
                                    .n2Idx = taskInfo_.n2Idx,
                                    .gS1Idx = taskInfo_.gS1Idx + startRow,
                                    .dIdx = 0,
                                    .gS1DealSize = dealRowCount,
                                    .dDealSize = (uint32_t)constInfo_.dSizeV};
            copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
        }
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T>& reduceOut, LocalTensor<T>& mm2Res, LocalTensor<T>& lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align_ = (uint64_t)this->dSizeV_Align_;
        ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align_, cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf_;
        AscendC::PipeBarrier<PIPE_V>();
        DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align_);
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
    __aicore__ inline void CalcPreNextTokens()
    {
        actSeqLensQ_ = qSeqLensTool_.GetActualSeqLength(taskInfo_.bIdx);
        actSeqLensKv_ = kvSeqLensTool_.GetActualSeqLength(taskInfo_.bIdx);

        int64_t safePreToken = constInfo_.preTokens;
        int64_t safeNextToken = constInfo_.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ_, actSeqLensKv_, safePreToken, safeNextToken, constInfo_.maskMode);

        if (static_cast<FAMaskMode>(constInfo_.maskMode) == FAMaskMode::BAND) {
            preTokensPerBatch_ = safePreToken;
            nextTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safeNextToken;
        } else {
            nextTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_;
            preTokensPerBatch_ = 0;
        }
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T>& attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if constexpr (!HAS_MASK) {
            return;
        }

        fa_base_vector::InvalidRowParams params{
            .actS1Size = actSeqLensQ_,
            .gSize = static_cast<uint64_t>(constInfo_.gSize),
            .gS1Idx = (DN_LAYOUT ? taskInfo_.s1Idx : taskInfo_.gS1Idx) + startRow,
            .dealRowCount = dealRowCount,
            .columnCount = columnCount,
            .preTokensPerBatch = preTokensPerBatch_,
            .nextTokensPerBatch = nextTokensPerBatch_,
        };

        fa_base_vector::InvalidRows<UBOUT_T,
                                    DN_LAYOUT ? fa_base_vector::UbInputFormat::GS1 : GeInputUbFormat<LAYOUT_T>()>
            invalidRows;
        invalidRows(attenOutUb, params);
    }

public:
    __aicore__ inline void FlashDecode(FDparams& fd)
    {
        if (!fd.fdCoreEnable) {
            return;
        }
        constexpr uint32_t fdBalanceMBaseSize = FD_BALANCE_M_BASE;
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        if constexpr (DN_LAYOUT) {
            // 不合轴(BN1_S1): fdBN1Idx 语义为 bn1 = b*n1, K/V 按 n2 = n1/gSize 共享
            taskInfo_.bIdx = fd.fdBN1Idx / constInfo_.n1Size;
            taskInfo_.n1Idx = fd.fdBN1Idx % constInfo_.n1Size;
            taskInfo_.n2Idx = taskInfo_.n1Idx / constInfo_.gSize;
            taskInfo_.gIdx = taskInfo_.n1Idx % constInfo_.gSize;
            taskInfo_.s1Idx = fd.fdMIdx * mBaseSize;
        } else {
            taskInfo_.bIdx = fd.fdBN2Idx / constInfo_.n2Size;
            taskInfo_.n2Idx = fd.fdBN2Idx % constInfo_.n2Size;
            taskInfo_.gS1Idx = fd.fdMIdx * mBaseSize;
        }
        taskInfo_.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * mBaseSize;

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
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum_; preLoadIdx++) {
                LocalTensor<T> mm2Res = (((reduceGlobaLoop + preLoadIdx) & 1) == 0) ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            Mutex::Lock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            Mutex::Lock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo_.actualCombineLoopSize, reduceMLoop, startRow);
            Mutex::Unlock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            Mutex::Unlock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CalcPreNextTokens();
            if (constInfo_.isSoftmaxLseEnable) {
                LocalTensor<T> maxLseUb = fdLseUbBuf_;
                Mutex::Lock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo_.gS1Idx + startRow;
                if constexpr (DN_LAYOUT) {
                    DataCopySoftmaxLseS1Only<LAYOUT_T>(softmaxLseGm_, maxLseUb, taskInfo_.bIdx, taskInfo_.n2Idx,
                                                       taskInfo_.gIdx, taskInfo_.s1Idx + startRow, actualGSplitSize,
                                                       constInfo_, qSeqLensTool_);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                    uint32_t prefixBS1 = qSeqLensTool_.cuSeqLensParser.GetTBase(taskInfo_.bIdx);
                    uint64_t bN2Offset = taskInfo_.n2Idx * constInfo_.gSize * constInfo_.t1Size + prefixBS1;
                    DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfo_t>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                    actualGSplitSize, constInfo_);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
                    uint64_t bN2Offset = taskInfo_.bIdx * constInfo_.gSize * constInfo_.n2Size * constInfo_.s1Size +
                                         taskInfo_.n2Idx * constInfo_.gSize * constInfo_.s1Size;
                    uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(taskInfo_.bIdx);
                    DataCopySoftmaxLseBSNDArch35<T, ConstInfo_t>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                 actualGSplitSize, constInfo_);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    uint64_t bN2Offset = taskInfo_.bIdx * constInfo_.gSize * constInfo_.n2Size * constInfo_.s1Size +
                                         taskInfo_.n2Idx * constInfo_.gSize * constInfo_.s1Size;
                    uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(taskInfo_.bIdx);
                    DataCopySoftmaxLseBNSDArch35<T, ConstInfo_t>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                 actualGSplitSize, constInfo_, qActSeqLens);
                }
                Mutex::Unlock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
            }

            for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; i++) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1_ : fdMm2ResBuf2_;
                if (i >= preLoadNum_) {
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

} // namespace FlashAttnKernel
#endif
