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
 * \file flash_attn_block_vec_flashdecode_arch92.h
 * \brief
 */
#ifndef FLASH_ATTN_BLOCK_VEC_FLASHDECODE_ARCH92_H
#define FLASH_ATTN_BLOCK_VEC_FLASHDECODE_ARCH92_H

#include "../utils/attenmask_gs1.h"

#if __has_include("../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h")
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#else
#include "../../common/arch35/vf/vf_flash_decode_arch35.h"
#endif

#include "memory_copy_arch92.h"

#if __has_include("../../../common/op_kernel/arch36/arch_info_arch36.h")
#include "../../../common/op_kernel/arch36/arch_info_arch36.h"
#else
#include "../../common/op_kernel/arch36/arch_info_arch36.h"
#endif

namespace BaseApi {
template <FaInnerMLayout M_LAYOUT>
struct TaskInfo;

template <>
struct TaskInfo<FaInnerMLayout::GS1_MERGE_LAYOUT> {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

template <>
struct TaskInfo<FaInnerMLayout::S1_ONLY_LAYOUT> {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gIdx;
    uint32_t s1Idx;
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

template <typename FD_T>
class FDNoQuantGqaBlockVec {
public:
    static constexpr FaInnerMLayout M_LAYOUT = FD_T::mLayout;
    using OUTPUT_T = typename FD_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FD_T::mBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FD_T::dVBaseSize;
    static constexpr uint32_t fdBalanceMBaseSize = 8U;
    static constexpr FA_LAYOUT LAYOUT_T = FD_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FD_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FD_T::attnOutLayout;
    static constexpr bool HAS_MASK = FD_T::hasMask;
    // =================================类型定义区=================================
    using T = float;

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

    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum_ = 2U;
    uint32_t dSizeV_Align_;
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    // ================================UB Buffer 常量区====================================
    // FD 业务缓冲区大小
    // LSE max/sum 缓冲区：每个 256B，存储 fdBalanceMBaseSize 行的 LSE 值
    static constexpr uint32_t FD_LSE_MAX_UB_BUF_BYTES = fdBalanceMBaseSize * BYTE_BLOCK;
    // Sum/Max/Exp 缓冲区，存储 combineLoop 维度的 softmax 中间结果
    static constexpr uint32_t FD_SUM_MAX_BUF_BYTES = fdBalanceMBaseSize * BYTE_BLOCK * ArchInfo::NPU_MAX_CUBE_NUM;
    // MM2 结果缓冲区：16K，存储 dealRowCount 行的 accumOut 数据
    static constexpr uint32_t FD_MM2_RES_BUF_BYTES = fdBalanceMBaseSize * dVBaseSize * sizeof(T);
    // Reduce/Output 缓冲区：16K，存储规约结果和最终输出
    static constexpr uint32_t FD_REDUCE_BUF_BYTES = fdBalanceMBaseSize * dVBaseSize * sizeof(T);
    static constexpr uint32_t FD_OUTPUT_BUF_BYTES = fdBalanceMBaseSize * dVBaseSize * sizeof(OUTPUT_T);

    // UB 总容量限制
    static constexpr uint32_t UB_SIZE_LIMIT = (ArchInfo::CV_RATIO == 1) ? 376 * 1024 : 248 * 1024;

protected:
    GlobalTensor<float> lseSumFdGm_;
    GlobalTensor<float> lseMaxFdGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxLseGm_;

    static constexpr UbFormat UB_FORMAT =
        (M_LAYOUT == FaInnerMLayout::GS1_MERGE_LAYOUT) ? GetOutUbFormat<LAYOUT_T>() : UbFormat::S1_ONLY;

    int64_t preTokensPerBatch_ = 0;
    int64_t nextTokensPerBatch_ = 0;

    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    // ================================类成员变量====================================
    const ConstInfoX& constInfo_;
    TaskInfo<M_LAYOUT> taskInfo_{};

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool_;

    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensor_;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, UB_FORMAT> copyAttenOutUbToGm_;

private:
    // ================================FD Local Buffer区====================================
    LocalTensor<T> fdSumBuf1_;          // 1.5k: 16*24*4
    LocalTensor<T> fdSumBuf2_;          // 1.5k: 16*24*4
    LocalTensor<T> fdMaxBuf1_;          // 1.5k: 16*24*4
    LocalTensor<T> fdMaxBuf2_;          // 1.5k: 16*24*4
    LocalTensor<T> fdLseExpBuf_;        // 1.5k: 16*24*4
    LocalTensor<T> fdMm2ResBuf1_;       // 32k: 16*512*4
    LocalTensor<T> fdMm2ResBuf2_;       // 32k: 16*512*4
    LocalTensor<T> fdReduceBuf_;        // 32k: 16*512*4
    LocalTensor<OUTPUT_T> fdOutputBuf_; // 32k: 16*512*4

    LocalTensor<T> fdLseMaxUbBuf1_;
    LocalTensor<T> fdLseMaxUbBuf2_;
    LocalTensor<T> fdLseUbBuf_;

public:
    __aicore__ inline FDNoQuantGqaBlockVec(ConstInfoX& constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T>& qSeqLensTool,
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
            // 与 FA block 共享UB布局：bmm1/bmm2 区域在前，FD 业务缓冲区紧随其后。
            // 使用 LocalTensor 构造函数直接指定绝对字节偏移，实现内存完全自主管理，
            // 无需 LocalMemAllocator 线性分配器。
            // FD 业务区起始字节偏移（跳过 FA 侧 bmm1/bmm2 区域），
            // 以模板参数传入（取值为 FA vec block 的 GetFdBaseOffset()），避免两侧重复推导漂移
            constexpr uint32_t BASE = UB_FD_BASE_OFFSET;

            struct FdUbLayout {
                uint8_t fdLseMaxBuf1[FD_LSE_MAX_UB_BUF_BYTES];
                uint8_t fdLseMaxBuf2[FD_LSE_MAX_UB_BUF_BYTES];
                uint8_t fdLseBuf[FD_LSE_MAX_UB_BUF_BYTES];
                uint8_t fdSumBuf1[FD_SUM_MAX_BUF_BYTES];
                uint8_t fdSumBuf2[FD_SUM_MAX_BUF_BYTES];
                uint8_t fdMaxBuf1[FD_SUM_MAX_BUF_BYTES];
                uint8_t fdMaxBuf2[FD_SUM_MAX_BUF_BYTES];
                uint8_t fdLseExpBuf[FD_SUM_MAX_BUF_BYTES];
                uint8_t fdMm2ResBuf1[FD_MM2_RES_BUF_BYTES];
                uint8_t fdMm2ResBuf2[FD_MM2_RES_BUF_BYTES];
                uint8_t fdReduceBuf[FD_REDUCE_BUF_BYTES];
                uint8_t fdOutputBuf[FD_OUTPUT_BUF_BYTES];
            };

            static_assert(BASE + sizeof(FdUbLayout) <= UB_SIZE_LIMIT, "FD UB buffer too large");

            fdLseMaxUbBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdLseMaxBuf1),
                                                   SIZE_OF_MEMBER(FdUbLayout, fdLseMaxBuf1))
                                  .template ReinterpretCast<T>();
            fdLseMaxUbBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdLseMaxBuf2),
                                                   SIZE_OF_MEMBER(FdUbLayout, fdLseMaxBuf2))
                                  .template ReinterpretCast<T>();
            fdLseUbBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdLseBuf),
                                               SIZE_OF_MEMBER(FdUbLayout, fdLseBuf))
                              .template ReinterpretCast<T>();

            fdSumBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdSumBuf1),
                                              SIZE_OF_MEMBER(FdUbLayout, fdSumBuf1))
                             .template ReinterpretCast<T>();
            fdSumBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdSumBuf2),
                                              SIZE_OF_MEMBER(FdUbLayout, fdSumBuf2))
                             .template ReinterpretCast<T>();
            fdMaxBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdMaxBuf1),
                                              SIZE_OF_MEMBER(FdUbLayout, fdMaxBuf1))
                             .template ReinterpretCast<T>();
            fdMaxBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdMaxBuf2),
                                              SIZE_OF_MEMBER(FdUbLayout, fdMaxBuf2))
                             .template ReinterpretCast<T>();
            fdLseExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdLseExpBuf),
                                                SIZE_OF_MEMBER(FdUbLayout, fdLseExpBuf))
                               .template ReinterpretCast<T>();

            fdMm2ResBuf1_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdMm2ResBuf1),
                                                 SIZE_OF_MEMBER(FdUbLayout, fdMm2ResBuf1))
                                .template ReinterpretCast<T>();
            fdMm2ResBuf2_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdMm2ResBuf2),
                                                 SIZE_OF_MEMBER(FdUbLayout, fdMm2ResBuf2))
                                .template ReinterpretCast<T>();

            fdReduceBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdReduceBuf),
                                                SIZE_OF_MEMBER(FdUbLayout, fdReduceBuf))
                               .template ReinterpretCast<T>();
            fdOutputBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, BASE + OFFSET_OF_MEMBER(FdUbLayout, fdOutputBuf),
                                                SIZE_OF_MEMBER(FdUbLayout, fdOutputBuf))
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
        bool learnableSinkFlag = false;
        ComputeScaleValue_VF_FD(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount,
                                actualCombineLoopSize, constInfo_.isSoftmaxLseEnable, learnableSinkFlag);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T>& attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        if constexpr (M_LAYOUT == FaInnerMLayout::GS1_MERGE_LAYOUT) {
            GmCoordGs1Merge gmCoord{.bIdx = taskInfo_.bIdx,
                                    .n2Idx = taskInfo_.n2Idx,
                                    .gS1Idx = taskInfo_.gS1Idx + startRow,
                                    .dIdx = 0,
                                    .gS1DealSize = dealRowCount,
                                    .dDealSize = (uint32_t)constInfo_.dSizeV};
            copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
        } else {
            GmCoordS1Only gmCoord{.bIdx = taskInfo_.bIdx,
                                  .n2Idx = taskInfo_.n2Idx,
                                  .gIdx = taskInfo_.gIdx,
                                  .s1Idx = taskInfo_.s1Idx + startRow,
                                  .dIdx = 0,
                                  .s1DealSize = dealRowCount,
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
    __aicore__ inline void CopyFinalResOut(LocalTensor<T>& accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM)
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

        fa_base_vector::GetSafeActToken(actSeqLensQ_, actSeqLensKv_, safePreToken, safeNextToken,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == BAND) {
            preTokensPerBatch_ = safePreToken;
            nextTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safeNextToken;
        } else if ((constInfo_.sparseMode == DEFAULT_MASK) && HAS_MASK) {
            nextTokensPerBatch_ = safeNextToken;
            preTokensPerBatch_ = actSeqLensKv_ - actSeqLensQ_ + safePreToken;
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

        if (constInfo_.sparseMode == ALL_MASK || constInfo_.sparseMode == LEFT_UP_CAUSAL) {
            return;
        }

        if constexpr (M_LAYOUT == FaInnerMLayout::GS1_MERGE_LAYOUT) {
            fa_base_vector::InvalidRowParams params{
                .actS1Size = actSeqLensQ_,
                .gSize = static_cast<uint64_t>(constInfo_.gSize),
                .gS1Idx = taskInfo_.gS1Idx + startRow,
                .dealRowCount = dealRowCount,
                .columnCount = columnCount,
                .preTokensPerBatch = preTokensPerBatch_,
                .nextTokensPerBatch = nextTokensPerBatch_,
            };

            fa_base_vector::InvalidRows<UBOUT_T, GeInputUbFormat<LAYOUT_T>(),
                                        fa_base_vector::InnerRowLayout::GS1_MERGE_LAYOUT>
                invalidRows;
            invalidRows(attenOutUb, params);
        } else {
            fa_base_vector::InvalidRowParamsT<fa_base_vector::InnerRowLayout::S1_ONLY_LAYOUT> params{
                .actS1Size = actSeqLensQ_,
                .s1Idx = taskInfo_.s1Idx + startRow,
                .dealRowCount = dealRowCount,
                .columnCount = columnCount,
                .preTokensPerBatch = preTokensPerBatch_,
                .nextTokensPerBatch = nextTokensPerBatch_,
            };

            fa_base_vector::InvalidRows<UBOUT_T, GeInputUbFormat<LAYOUT_T>(),
                                        fa_base_vector::InnerRowLayout::S1_ONLY_LAYOUT>
                invalidRows;
            invalidRows(attenOutUb, params);
        }
    }

public:
    __aicore__ inline void FlashDecode(FDparamsX& fd)
    {
        if (!fd.fdCoreEnable) {
            return;
        }
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;

        if constexpr (M_LAYOUT == FaInnerMLayout::GS1_MERGE_LAYOUT) {
            taskInfo_.bIdx = fd.fdBNIdx / constInfo_.n2Size;
            taskInfo_.n2Idx = fd.fdBNIdx % constInfo_.n2Size;
            taskInfo_.gS1Idx = fd.fdMIdx * mBaseSize;
        } else {
            taskInfo_.bIdx = fd.fdBNIdx / (constInfo_.n2Size * constInfo_.gSize);
            uint32_t n1Idx = fd.fdBNIdx % (constInfo_.n2Size * constInfo_.gSize);
            taskInfo_.n2Idx = n1Idx / constInfo_.gSize;
            taskInfo_.gIdx = n1Idx % constInfo_.gSize;
            taskInfo_.s1Idx = fd.fdMIdx * mBaseSize;
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
                uint32_t mOffset;
                if constexpr (M_LAYOUT == FaInnerMLayout::GS1_MERGE_LAYOUT) {
                    mOffset = taskInfo_.gS1Idx + startRow;
                    if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                        uint32_t prefixBS1 = qSeqLensTool_.cuSeqLensParser.GetTBase(taskInfo_.bIdx);
                        uint64_t bN2Offset = taskInfo_.n2Idx * constInfo_.gSize * constInfo_.t1Size + prefixBS1;
                        DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                       actualGSplitSize, constInfo_);
                    } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
                        uint64_t bN2Offset = taskInfo_.bIdx * constInfo_.gSize * constInfo_.n2Size * constInfo_.s1Size +
                                             taskInfo_.n2Idx * constInfo_.gSize * constInfo_.s1Size;
                        DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                    actualGSplitSize, constInfo_);
                    } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                        uint64_t bN2Offset = taskInfo_.bIdx * constInfo_.gSize * constInfo_.n2Size * constInfo_.s1Size +
                                             taskInfo_.n2Idx * constInfo_.gSize * constInfo_.s1Size;
                        uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(taskInfo_.bIdx);
                        DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                    actualGSplitSize, constInfo_, qActSeqLens);
                    }
                } else {
                    mOffset = taskInfo_.s1Idx + startRow;
                    uint32_t n1Idx = taskInfo_.n2Idx * constInfo_.gSize + taskInfo_.gIdx;
                    uint32_t n1Size = constInfo_.n2Size * constInfo_.gSize;
                    if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                        uint32_t prefixBS1 = qSeqLensTool_.cuSeqLensParser.GetTBase(taskInfo_.bIdx);
                        uint64_t gmBaseOffset = static_cast<uint64_t>(n1Idx) * constInfo_.t1Size + prefixBS1;
                        DataCopySoftmaxLseS1OnlyTNDtoNTArch35<T>(softmaxLseGm_, maxLseUb, gmBaseOffset, mOffset,
                                                                 actualGSplitSize);
                    } else {
                        uint64_t gmBaseOffset = static_cast<uint64_t>(taskInfo_.bIdx) * n1Size * constInfo_.s1Size +
                                                static_cast<uint64_t>(n1Idx) * constInfo_.s1Size;
                        uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(taskInfo_.bIdx);
                        DataCopySoftmaxLseS1OnlyBXXDArch35<T>(softmaxLseGm_, maxLseUb, gmBaseOffset, mOffset,
                                                              actualGSplitSize, qActSeqLens);
                    }
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
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

} // namespace BaseApi
#endif
