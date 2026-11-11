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
 * \file flash_mla_block_vec_arch92.h
 * \brief arch92 flash_mla MLA vector；
 *        静态 tensor UbLayout + Mutex/cross-core 显式同步 + FA 协同清零
 *        （FD 业务 UB 从 160 KiB 起，避开 Cube/Vector 共用的 MM1/MM2 区）
 */
#ifndef FLASH_MLA_BLOCK_VEC_ARCH92_H_
#define FLASH_MLA_BLOCK_VEC_ARCH92_H_

#include "../utils/flash_mla_attn_sink_gs1.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include <limits>

#include "../utils/flash_mla_attenmask_gs1.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/init_output.h"
#include "../../../common/op_kernel/const_def.h"
#include "../arch35/flash_mla_public_define_arch35.h"
#include "../utils/flash_mla_config.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../utils/flash_mla_memory_copy_arch35.h"
#include "../utils/flash_mla_type.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AttentionCommon;

namespace FlashMlaKernel {
template <bool first, bool last>
__simd_vf__ inline void MlaStreamUpdate(__ubuf__ float* dst, __ubuf__ float* cur, __ubuf__ float* exp,
                                        __ubuf__ float* sum, uint16_t rows)
{
    using namespace AscendC::Reg;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> e0, s0, x00, x01, x02, x03, o00, o01, o02, o03;
    for (uint16_t i = 0; i < rows; ++i) {
        if constexpr (!first) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(e0, exp + i);
        }
        if constexpr (last) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s0, sum + i);
        }
        LoadAlign(x00, cur + (i) * 256U + 0U);
        if constexpr (!first) {
            LoadAlign(o00, dst + (i) * 512U + 0U);
        }
        LoadAlign(x01, cur + (i) * 256U + 64U);
        if constexpr (!first) {
            LoadAlign(o01, dst + (i) * 512U + 64U);
        }
        LoadAlign(x02, cur + (i) * 256U + 128U);
        if constexpr (!first) {
            LoadAlign(o02, dst + (i) * 512U + 128U);
        }
        LoadAlign(x03, cur + (i) * 256U + 192U);
        if constexpr (!first) {
            LoadAlign(o03, dst + (i) * 512U + 192U);
        }
        if constexpr (!first) {
            Mul(o00, o00, e0, mask);
        }
        if constexpr (!first) {
            Mul(o01, o01, e0, mask);
        }
        if constexpr (!first) {
            Mul(o02, o02, e0, mask);
        }
        if constexpr (!first) {
            Mul(o03, o03, e0, mask);
        }
        if constexpr (!first) {
            Add(x00, o00, x00, mask);
        }
        if constexpr (!first) {
            Add(x01, o01, x01, mask);
        }
        if constexpr (!first) {
            Add(x02, o02, x02, mask);
        }
        if constexpr (!first) {
            Add(x03, o03, x03, mask);
        }
        if constexpr (last) {
            Div(x00, x00, s0, mask);
        }
        if constexpr (last) {
            Div(x01, x01, s0, mask);
        }
        if constexpr (last) {
            Div(x02, x02, s0, mask);
        }
        if constexpr (last) {
            Div(x03, x03, s0, mask);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + 0U, x00, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + 64U, x01, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + 128U, x02, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dst + (i) * 512U + 192U, x03, mask);
    }
}

// A sink must not turn a fully masked query row into a valid row.
template <bool GS1>
__simd_vf__ inline void MaskInvalidSinkRows(__ubuf__ float* maxUb, __ubuf__ float* sumUb, uint32_t start, uint32_t rows,
                                            uint32_t s1Size, uint32_t gSize, int32_t firstValid, int32_t endValid)
{
    RegTensor<int32_t> index, divisor, quotient, position;
    RegTensor<float> maximum, total, invalidMax, zero;
    MaskReg all = CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Arange<int32_t>(index, static_cast<int32_t>(start));
    Reg::Duplicate<int32_t>(divisor, static_cast<int32_t>(GS1 ? s1Size : gSize), all);
    Reg::Div<int32_t>(quotient, index, divisor, all);
    if constexpr (GS1) {
        Reg::Mul<int32_t>(position, quotient, divisor, all);
        Reg::Sub<int32_t>(position, index, position, all);
    } else {
        Reg::Adds<int32_t>(position, quotient, 0, all);
    }
    LoadAlign<float, Reg::LoadDist::DIST_NORM>(maximum, maxUb);
    Reg::Duplicate<float>(total, 1.0f, all);
    Reg::Duplicate<float>(invalidMax, -std::numeric_limits<float>::infinity(), all);
    Reg::Duplicate<float>(zero, 0.0f, all);
    MaskReg valid;
    Reg::Compares<int32_t, CMPMODE::GE>(valid, position, firstValid, all);
    Reg::Select<float>(maximum, maximum, invalidMax, valid);
    Reg::Select<float>(total, total, zero, valid);
    Reg::Compares<int32_t, CMPMODE::LT>(valid, position, endValid, all);
    Reg::Select<float>(maximum, maximum, invalidMax, valid);
    Reg::Select<float>(total, total, zero, valid);
    MaskReg tail = UpdateMask<float>(rows);
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(maxUb, maximum, tail);
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(sumUb, total, tail);
}

template <typename FA_T>
class FlashMlaNoQuantMlaBlockVec {
public:
    using INPUT_T = typename FA_T::inputType;
    using T = typename FA_T::mmType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr FLASH_MLA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FLASH_MLA_LAYOUT outLayout = FA_T::attnOutLayout;
    static constexpr bool hasAtten = FA_T::hasMask;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize / CV_RATIO;
    static constexpr uint32_t vec1Srcstride = (mBaseSize / CV_RATIO) + 1; // 解bank冲突，需要加1行
    static constexpr uint32_t dVTemplateAlign64 = BaseApi::Align64Func((uint16_t)FA_T::dVBaseSize);

    static constexpr uint32_t DB = 2U;
    static constexpr uint32_t PRELOAD_N = 2U; // C1 C1 C2 C2
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = FA_T::flashDecode;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, (FA_T::kvLayoutType > 0)>();
    using SeqLensToolType = FlashMlaSeqLensTool<Q_MODE, KV_MODE>;
    static constexpr MaskFormat MASK_LAYOUT =
        (LAYOUT_T == FLASH_MLA_LAYOUT::BSH || LAYOUT_T == FLASH_MLA_LAYOUT::TND) ? MaskFormat::SG : MaskFormat::GS;

    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    using ConstInfoNoQuant = ConstInfo_t<FlashMlaKernelType::NO_QUANT>;
    using OUT_T = OUTPUT_T;

    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4ULL;
    static constexpr uint32_t CROSSCORE_BMM1_0 = 0U;
    static constexpr uint32_t CROSSCORE_BMM2_0 = 2U;
    static constexpr uint32_t CROSSCORE_BMM2_1 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;

    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0U;  // vec2Res（96K@120K）V/MTE3
    static constexpr uint32_t EVENT_ID0 = 1U;               // InitOutput atten-out pop 同步（PIPE_V/PIPE_MTE3）
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2U;  // stage1/vec1Res（12.25K@216K）V/MTE3
    static constexpr uint32_t EVENT_ID1 = 3U;               // InitOutput LSE pop 同步（PIPE_V/PIPE_MTE3）
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4U;   // LSE 1.5 KiB 输出（V/MTE3）
    static constexpr uint32_t UB_BRDCST_MAX_SUM_EVENT = 5U; // max/sumBrdcst 2 KiB（V/MTE3）
    // 掩码槽复用前需等待 Vector 消费结束，防止覆盖未读数据。
    static constexpr uint32_t UB_IN_MASK_EVENT0 = 6U;

    static_assert(mBaseSize == 64 && s2BaseSize == 128 && CV_RATIO == 1, "arch92 requires M64/S2=128/CV=1");
    // Full-block layout: 327.5 KiB, including two 8 KiB mask slots and sink staging.
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * 256U * sizeof(T);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 1U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * 128U * sizeof(T);
    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVTemplateAlign64 * sizeof(T);
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = (mBaseSize / CV_RATIO + 1) * 128U * sizeof(INPUT_T);
    static constexpr uint32_t UB_MASK_BUFCNT = DB;
    static constexpr uint32_t UB_MASK_BUF_BYTES = mBaseSize / CV_RATIO * 128U;
    static constexpr uint32_t UB_SOFTMAX_BUFCNT = PRELOAD_N + 1;
    static constexpr uint32_t UB_SOFTMAX_BUF_BYTES = 256U; // SOFTMAX VF max/sum/exp 按 256B 对齐
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES = mBaseSize / CV_RATIO * sizeof(float) * 8;
    static constexpr uint32_t UB_BRDCST_BUF_BYTES = 2048U; // max/sumBrdcst
    static constexpr uint32_t UB_TMP_BUF_BYTES = 512U;     // commonTBuf 512B
    static_assert(UB_SOFTMAX_BUF_BYTES >= ((mBaseSize / CV_RATIO + 63U) / 64U) * 256U,
                  "Each softmax state must allow a full FP32 vector access");
    static_assert(UB_TMP_BUF_BYTES >= 2U * UB_SOFTMAX_BUF_BYTES,
                  "Softmax scratch contains separate full-vector sum and max regions");
    static_assert(UB_BRDCST_BUF_BYTES >= ((mBaseSize / CV_RATIO + 7U) / 8U) * 256U,
                  "FD broadcast output rounds query rows up to groups of eight");
    static_assert(UB_LSE_OUT_BUF_BYTES >= ((mBaseSize / CV_RATIO + 7U) / 8U) * 256U,
                  "LSE staging must hold the broadcast FP32 row blocks");
    static_assert(UB_MASK_BUF_BYTES >= ((mBaseSize / CV_RATIO + 1U) / 2U) * 256U,
                  "Mask merge accesses pairs of 128-byte rows");

    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = s2BaseSize * 576 * sizeof(INPUT_T);
    static constexpr uint32_t L1_Q_PREFIX_BYTES = mBaseSize * 576 * sizeof(INPUT_T);

    static __aicore__ inline constexpr uint32_t GetFdBaseOffset()
    {
        return UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES + UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES;
    }

    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<float> sinkGm_;
    LocalTensor<uint8_t> sinkTmpUb_;
    SeqLensToolType& seqLensTool_;

    attenMaskGmType attenMaskGmInt_;

    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxFDSumGm_;
    GlobalTensor<float> softmaxFDMaxGm_;

    static constexpr GmFormat OUT_FORMAT = GetQueryGmFormat<outLayout>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t,
                                     (outLayout == FLASH_MLA_LAYOUT::TND || outLayout == FLASH_MLA_LAYOUT::NTD)>;
    FaGmTensorOut outGmTensor_;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm_;

    LocalTensor<uint8_t> ubMm2ResBuffers_;
    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubVec2Res_;
    LocalTensor<uint8_t> ubVec1ResBuffers_;
    LocalTensor<uint8_t> ubMaskBuffers_;
    LocalTensor<float> softmaxSumBuf_;
    LocalTensor<float> softmaxMaxBuf_;
    LocalTensor<float> softmaxExpBuf_;
    LocalTensor<uint8_t> ubLseOutBuf_;
    LocalTensor<uint8_t> ubSumBrdcstBuf_;
    LocalTensor<uint8_t> ubMaxBrdcstBuf_;
    LocalTensor<uint8_t> vec1ApiTmpBuf_;
    LocalTensor<uint8_t> l1PBuffers_;

    const ConstInfoNoQuant& constInfo_;

    __aicore__ inline FlashMlaNoQuantMlaBlockVec(ConstInfoNoQuant& constInfo, SeqLensToolType& seqLensTool)
        : constInfo_(constInfo),
          seqLensTool_(seqLensTool){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t* sinks, __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        InitVecInput(sinks, attenMask, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t* sinks, __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxLse,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);

        if (unlikely(constInfo_.isSoftmaxLseEnable)) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }

        if constexpr (hasAtten) {
            attenMaskGmInt_.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm_.SetGlobalBuffer((__gm__ float*)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
        if (constInfo_.learnableSinkFlag) {
            sinkGm_.SetGlobalBuffer((__gm__ float*)sinks);
        }
        InitAttenOutBuffer();
    }

    __aicore__ inline void InitAttenOutBuffer()
    {
        outGmTensor_.gmTensor = attentionOutGm_;
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor_.offsetCalculator.Init(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size,
                                               constInfo_.dSizeV, seqLensTool_.qActSeqLensParser_);
        } else {
            outGmTensor_.offsetCalculator.Init(constInfo_.n2Size, constInfo_.gSize, constInfo_.dSizeV,
                                               seqLensTool_.qActSeqLensParser_);
        }
    }

    __aicore__ inline void ProcessVec1(FlashMlaRunInfoX runInfo)
    {
        if (unlikely(runInfo.loop == 0U)) {
            AscendC::ICachePreLoad(7);
        }
        uint32_t mm1ResUbBufId = runInfo.loop % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t c1v1CrossCoreSyncIdx = CROSSCORE_BMM1_0 + mm1ResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        if (unlikely(runInfo.isFirstS2Loop)) {
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_BUFCNT, runInfo);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Prepare the current mask while QK is running; consume it only after QK is ready.
        bool useMask = false;
        if constexpr (hasAtten) {
            if (runInfo.actVecMSize != 0) {
                auto maskUb = ubMaskBuffers_[(runInfo.loop % DB) * UB_MASK_BUF_BYTES];
                useMask = AttenMaskCopyIn(maskUb, 0, runInfo.actVecMSize, runInfo);
            }
        }
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx); // 反堵 c1v1：AIC 可覆写本槽
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx); // 反堵 v1c2：本 AIV 无 P 行
            return;
        }

        ProcessVec1Nd(pL1Tensor, mm1ResUbTensor, runInfo, useMask);

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(c1v1CrossCoreSyncIdx);    // C1 收到后可启动 FIXPIPE 写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx); // P 已写 L1，C2 可读
        Vec1PostProcess(runInfo);
    }

    // 初始化借用 CV 缓冲区，SyncAll 后才能发布 BMM1/BMM2 可用标志。
    __aicore__ inline void ClearOutput(bool needInit)
    {
        if (needInit) {
            uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum;
            int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
            if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::TND || LAYOUT_T == FLASH_MLA_LAYOUT::NTD ||
                          LAYOUT_T == FLASH_MLA_LAYOUT::NTD_TND) {
                tSize = seqLensTool_.qActSeqLensParser_.GetTSize();
            }
            int64_t totalOutputSize = tSize * constInfo_.n2Size * constInfo_.gSize * constInfo_.dSizeV;

            static constexpr uint32_t UB_ATTEN_POP_BUF_ELE = BUFFER_SIZE_BYTE_32K / sizeof(OUT_T);
            AttentionCommon::InitOutput<OUT_T, EVENT_ID0, 0U, UB_ATTEN_POP_BUF_ELE, true>(
                attentionOutGm_, static_cast<uint64_t>(totalOutputSize), vecCoreNum, static_cast<OUT_T>(0));

            if (unlikely(constInfo_.isSoftmaxLseEnable)) {
                int64_t lseTotalSize = tSize * constInfo_.n2Size * constInfo_.gSize;
                static constexpr uint32_t UB_LSE_POP_BUF_ELE = BUFFER_SIZE_BYTE_32K / sizeof(float);
                AttentionCommon::InitOutput<float, EVENT_ID1, BUFFER_SIZE_BYTE_32K, UB_LSE_POP_BUF_ELE, true>(
                    softmaxLseGm_, static_cast<uint64_t>(lseTotalSize), vecCoreNum,
                    3e+99); // 3e+99: set the value of invalid batch to inf
            }
            SyncAll();
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(FlashMlaRunInfoX runInfo, LocalTensor<float>& sumUb,
                                              LocalTensor<float>& maxUb)
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
            if (unlikely(constInfo_.isSoftmaxLseEnable)) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
                                             FlashMlaRunInfoX& runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0);
        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = ubLseOutBuf_.template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0);

        // runInfo 按 token 排列，LSE 写出时需转为 head-major。
        if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::NTD) {
            uint32_t prefixBS1 = seqLensTool_.qActSeqLensParser_.GetTBase(runInfo.bIdx);
            uint32_t s1Size = seqLensTool_.qActSeqLensParser_.GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo_.n2Size * constInfo_.gSize * constInfo_.t1Size;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoNoQuant>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                             runInfo.actVecMSize, constInfo_, s1Size);
        } else if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::TND) {
            uint32_t prefixBS1 = seqLensTool_.qActSeqLensParser_.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo_.n2Size * constInfo_.gSize * constInfo_.t1Size;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoNoQuant>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                 runInfo.actVecMSize, prefixBS1, constInfo_);
        } else if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = seqLensTool_.qActSeqLensParser_.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoNoQuant>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                              runInfo.actVecMSize, constInfo_, 0);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = seqLensTool_.qActSeqLensParser_.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoNoQuant>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                              runInfo.actVecMSize, constInfo_, qActSeqLens, 0);
        }

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0);
    }

    template <bool APPLY_MASK>
    __aicore__ inline void ProcessSoftmax(LocalTensor<INPUT_T>& stage1CastTensor, LocalTensor<T>& mmRes,
                                          LocalTensor<uint8_t>& attenMaskUb, const FlashMlaRunInfoX& runInfo)
    {
        using pseShiftType = INPUT_T;
        constexpr bool HAS_DROP = false;
        constexpr PseTypeEnum PSE_MODE = PseTypeEnum::PSE_NONE_TYPE;
        const T negativeFloatScalar = *((const T*)&NEGATIVE_MIN_VALUE_FP32);
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;

        LocalTensor<float> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<float> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<float> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<T> pScaleUb;
        if (likely(runInfo.actSingleLoopS2Size == 128)) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, 128U, EQ_128, APPLY_MASK, PSE_MODE, HAS_DROP,
                          false, false>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                                        attenMaskUb, pseUb, dropMaskUb, vec1ApiTmpBuf_, pScaleUb, runInfo.actVecMSize,
                                        runInfo.actSingleLoopS2Size, 0U, 0.0F, 0.0F, constInfo_.scaleValue, 1.0F,
                                        negativeFloatScalar, 0.0F);
        } else if (runInfo.actSingleLoopS2Size <= 64) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, 128U, GT_0_AND_LTE_64, APPLY_MASK, PSE_MODE,
                          HAS_DROP, false, false>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                                                  attenMaskUb, pseUb, dropMaskUb, vec1ApiTmpBuf_, pScaleUb,
                                                  runInfo.actVecMSize, runInfo.actSingleLoopS2Size, 0U, 0.0F, 0.0F,
                                                  constInfo_.scaleValue, 1.0F, negativeFloatScalar, 0.0F);
        } else if (runInfo.actSingleLoopS2Size < 128) {
            ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, 128U, GT_64_AND_LTE_128, APPLY_MASK, PSE_MODE,
                          HAS_DROP, false, false>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                                                  attenMaskUb, pseUb, dropMaskUb, vec1ApiTmpBuf_, pScaleUb,
                                                  runInfo.actVecMSize, runInfo.actSingleLoopS2Size, 0U, 0.0F, 0.0F,
                                                  constInfo_.scaleValue, 1.0F, negativeFloatScalar, 0.0F);
        } else {
            if constexpr (s2BaseSize == 256) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, 128U, GT_128_AND_LTE_256, APPLY_MASK, PSE_MODE,
                              HAS_DROP>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                                        attenMaskUb, pseUb, dropMaskUb, vec1ApiTmpBuf_, expUb, runInfo.actVecMSize,
                                        runInfo.actSingleLoopS2Size, 0U, 0.0F, 0.0F, constInfo_.scaleValue, 1.0F,
                                        negativeFloatScalar, 0.0F);
            }
        }
    }

    __aicore__ inline void ProcessVec1Nd(LocalTensor<INPUT_T>& pL1Tensor, LocalTensor<T>& mm1ResUbTensor,
                                         FlashMlaRunInfoX runInfo, bool useMask)
    {
        LocalTensor<uint8_t> attenMaskUb;
        const uint32_t maskBufId = runInfo.loop % DB;
        if constexpr (hasAtten) {
            attenMaskUb = ubMaskBuffers_[maskBufId * UB_MASK_BUF_BYTES];
            if (useMask) {
                Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            }
        }

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0);
        LocalTensor<INPUT_T> stage1CastTensor = ubVec1ResBuffers_.template ReinterpretCast<INPUT_T>();
        if constexpr (hasAtten) {
            if (useMask) {
                ProcessSoftmax<true>(stage1CastTensor, mm1ResUbTensor, attenMaskUb, runInfo);
            } else {
                ProcessSoftmax<false>(stage1CastTensor, mm1ResUbTensor, attenMaskUb, runInfo);
            }
        } else {
            ProcessSoftmax<false>(stage1CastTensor, mm1ResUbTensor, attenMaskUb, runInfo);
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0);
        if constexpr (hasAtten) {
            if (useMask) {
                Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            }
        }

        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0);

        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(pL1Tensor[s2BaseSize * dVBaseSize +
                               runInfo.vecMbaseIdx * (AttentionCommon::BYTE_BLOCK / sizeof(INPUT_T))],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0);
    }

    __aicore__ inline void Vec1PostProcess(FlashMlaRunInfoX runInfo)
    {
        LocalTensor<float> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<float> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        LocalTensor<float> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];

        // 折叠软最大值：首轮时 mloop 槽 sum/max 已由 ProcessVec1 的 ResetSoftmaxBuffer
        // 置 0/-inf，Update 恒执行（消除 isFirstS2Loop 区分）
        UpdateExpSumAndExpMax<T, mBaseSize>(sumUb, maxUb, expUb, sumUb, maxUb, vec1ApiTmpBuf_, runInfo.actVecMSize);

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2(FlashMlaRunInfoX runInfo)
    {
        auto dst = ubVec2Res_.template ReinterpretCast<T>();
        auto exp = softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        auto sum = softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float))];
        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        for (uint32_t n = 0; n < 2; ++n) {
            auto cur = ubMm2ResBuffers_[(n & 1U) * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<T>();
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_0 + (n & 1U));
            if (runInfo.actVecMSize != 0) {
                if (runInfo.isFirstS2Loop) {
                    if (runInfo.isLastS2Loop) {
                        MlaStreamUpdate<true, true>(
                            (__ubuf__ float*)dst.GetPhyAddr() + n * 256U, (__ubuf__ float*)cur.GetPhyAddr(),
                            (__ubuf__ float*)exp.GetPhyAddr(), (__ubuf__ float*)sum.GetPhyAddr(), runInfo.actVecMSize);
                    } else {
                        MlaStreamUpdate<true, false>(
                            (__ubuf__ float*)dst.GetPhyAddr() + n * 256U, (__ubuf__ float*)cur.GetPhyAddr(),
                            (__ubuf__ float*)exp.GetPhyAddr(), (__ubuf__ float*)sum.GetPhyAddr(), runInfo.actVecMSize);
                    }
                } else {
                    if (runInfo.isLastS2Loop) {
                        MlaStreamUpdate<false, true>(
                            (__ubuf__ float*)dst.GetPhyAddr() + n * 256U, (__ubuf__ float*)cur.GetPhyAddr(),
                            (__ubuf__ float*)exp.GetPhyAddr(), (__ubuf__ float*)sum.GetPhyAddr(), runInfo.actVecMSize);
                    } else {
                        MlaStreamUpdate<false, false>(
                            (__ubuf__ float*)dst.GetPhyAddr() + n * 256U, (__ubuf__ float*)cur.GetPhyAddr(),
                            (__ubuf__ float*)exp.GetPhyAddr(), (__ubuf__ float*)sum.GetPhyAddr(), runInfo.actVecMSize);
                    }
                }
            }
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_0 + (n & 1U));
        }
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        if (runInfo.isLastS2Loop && runInfo.actVecMSize != 0) {
            CopyOutAttentionOut(runInfo, dst, 0, runInfo.actVecMSize);
        }
    }

    __aicore__ inline void CopyOutAttentionOut(FlashMlaRunInfoX runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
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

    __aicore__ inline void Bmm2ResCastAndCopyOut(FlashMlaRunInfoX& runInfo, LocalTensor<T>& vec2ResUb,
                                                 uint32_t mStartVec, uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        attenOut.SetAddr(vec2ResUb.address_);

        int64_t dSizeAligned64 = static_cast<int64_t>(FA_T::dVBaseSize);

        Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(FlashMlaRunInfoX& runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::BSH || LAYOUT_T == FLASH_MLA_LAYOUT::TND) {
            s1StartTdx = vecMStartIdx / constInfo_.gSize;
            s1EndTdx = vecMEndIdx / constInfo_.gSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx != gEndIdx) { // 跨多个G
                ret = (s1FirstValidToken > 0) || (s1LastValidToken < (runInfo.actS1Size - 1));
            } else { // 只跨1个G
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      FlashMlaRunInfoX& runInfo, int64_t dSizeAligned64)
    {
        if constexpr (hasAtten) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.winRightFromTopLeft, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.winLeftFromTopLeft + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = (constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) && hasValidRow;
            if (!batchNeedRowInvalid) {
                return;
            }
            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_BUFCNT) * (UB_SOFTMAX_BUF_BYTES / sizeof(float)) +
                                   mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo_.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const FlashMlaRunInfoX& info, LocalTensor<OUTPUT_T>& attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.n2Idx,
                                .gS1Idx = (info.gS1Idx + info.vecMbaseIdx + vecMIdx),
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo_.dSizeV};
        FaUbTensor<OUTPUT_T, false> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = (uint32_t)(dVTemplateAlign64)};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, false>& ubTensor, GmCoordGs1Merge& gmCoord)
    {
        copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
    }

    __aicore__ inline void BroadCastAndCopyOut(const FlashMlaRunInfoX& runInfo, LocalTensor<float>& sumUb,
                                               LocalTensor<float>& maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        Mutex::Lock<PIPE_V>(UB_BRDCST_MAX_SUM_EVENT);
        LocalTensor<float> sumOutTensor = ubSumBrdcstBuf_.template ReinterpretCast<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_BRDCST_MAX_SUM_EVENT);
        Mutex::Lock<PIPE_MTE3>(UB_BRDCST_MAX_SUM_EVENT);
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_BRDCST_MAX_SUM_EVENT);

        Mutex::Lock<PIPE_V>(UB_BRDCST_MAX_SUM_EVENT);
        LocalTensor<float> maxOutTensor = ubMaxBrdcstBuf_.template ReinterpretCast<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_BRDCST_MAX_SUM_EVENT);
        Mutex::Lock<PIPE_MTE3>(UB_BRDCST_MAX_SUM_EVENT);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxOutTensor, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_BRDCST_MAX_SUM_EVENT);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const FlashMlaRunInfoX& runInfo, LocalTensor<float>& sumUb,
                                                       LocalTensor<float>& maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const FlashMlaRunInfoX& runInfo, LocalTensor<T>& vec2ResUb,
                                               uint32_t mStartVec, uint32_t mDealSize)
    {
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
        int64_t dSizeAligned64 = (int64_t)FA_T::dVBaseSize;
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (BaseApi::FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, L1_Q_PREFIX_BYTES, L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        struct UbLayout {
            uint8_t mm2ResBuffers[UB_MM2_RES_BUFCNT][UB_MM2_RES_BUF_BYTES];
            uint8_t mm1ResBuffers[UB_MM1_RES_BUFCNT][UB_MM1_RES_BUF_BYTES];
            uint8_t vec2Res[UB_VEC2_RES_BUF_BYTES];
            uint8_t vec1Res[UB_VEC1_RES_BUF_BYTES];
            uint8_t maskBuffers[UB_MASK_BUFCNT][UB_MASK_BUF_BYTES];
            uint8_t softmaxSumBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 3*256B，sum 常驻缓冲
            uint8_t softmaxMaxBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 3*256B，max 常驻缓冲
            uint8_t softmaxExpBuf[UB_SOFTMAX_BUFCNT][UB_SOFTMAX_BUF_BYTES]; // 3*256B，exp 常驻缓冲
            uint8_t lseOutBuf[UB_LSE_OUT_BUF_BYTES];
            uint8_t sumBrdcstBuf[UB_BRDCST_BUF_BYTES]; // 2K，FD sum/max staging
            // FD max/sum 共用 2K staging，并使用同一个 V/MTE3 Mutex。
            uint8_t tmpBuf[UB_TMP_BUF_BYTES]; // 0.5K，softmax 中间结果缓存
            uint8_t sinkTmp[512U];            // Preserve the fixed Cube communication buffer offsets.
        };
        static_assert(sizeof(UbLayout) == 327U * 1024U + 512U, "arch92 UB layout must match the tile model");
        static_assert(sizeof(UbLayout) <= FlashMlaConfig::ARCH92.ubBytes, "UB buffer too large");
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, mm2ResBuffers),
                                                SIZE_OF_MEMBER(UbLayout, mm2ResBuffers));
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, mm1ResBuffers),
                                                SIZE_OF_MEMBER(UbLayout, mm1ResBuffers));
        ubVec2Res_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, vec2Res),
                                          SIZE_OF_MEMBER(UbLayout, vec2Res));
        ubVec1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, vec1Res),
                                                 SIZE_OF_MEMBER(UbLayout, vec1Res));
        ubMaskBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, maskBuffers),
                                              SIZE_OF_MEMBER(UbLayout, maskBuffers));
        softmaxSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxSumBuf),
                                              SIZE_OF_MEMBER(UbLayout, softmaxSumBuf))
                             .template ReinterpretCast<float>();
        softmaxMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxMaxBuf),
                                              SIZE_OF_MEMBER(UbLayout, softmaxMaxBuf))
                             .template ReinterpretCast<float>();
        softmaxExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, softmaxExpBuf),
                                              SIZE_OF_MEMBER(UbLayout, softmaxExpBuf))
                             .template ReinterpretCast<float>();
        ubLseOutBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, lseOutBuf),
                                            SIZE_OF_MEMBER(UbLayout, lseOutBuf));
        sinkTmpUb_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, sinkTmp),
                                          SIZE_OF_MEMBER(UbLayout, sinkTmp));
        ubSumBrdcstBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, sumBrdcstBuf),
                                               SIZE_OF_MEMBER(UbLayout, sumBrdcstBuf));
        ubMaxBrdcstBuf_ = ubSumBrdcstBuf_;
        vec1ApiTmpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, tmpBuf),
                                              SIZE_OF_MEMBER(UbLayout, tmpBuf));
    }

    __aicore__ inline void ResetSoftmaxBuffer(uint32_t slotIdx, const FlashMlaRunInfoX& runInfo)
    {
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_BUF_BYTES / sizeof(float);
        LocalTensor<float> sumUb = softmaxSumBuf_[slotIdx * softmaxBufElementCount];
        LocalTensor<float> maxUb = softmaxMaxBuf_[slotIdx * softmaxBufElementCount];
        if (constInfo_.learnableSinkFlag && runInfo.isFirstFdBlock && runInfo.actVecMSize > 0) {
            Duplicate<float>(sumUb, 1.0f, softmaxBufElementCount);
            const uint32_t gs1Start = runInfo.gS1Idx + runInfo.vecMbaseIdx;
            LocalTensor<float> sinkTmp = sinkTmpUb_.template ReinterpretCast<float>();
            Mutex::Lock<PIPE_MTE2>(EVENT_ID0);
            if constexpr (LAYOUT_T == FLASH_MLA_LAYOUT::BNSD) {
                AttentionCommon::SinkCopyInGS1(sinkTmp, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                               runInfo.n2Idx, constInfo_.gSize);
            } else {
                AttentionCommon::SinkCopyInS1G(sinkTmp, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                               runInfo.n2Idx, constInfo_.gSize);
            }
            Mutex::Unlock<PIPE_MTE2>(EVENT_ID0);
            Mutex::Lock<PIPE_V>(EVENT_ID0);
            AttentionCommon::SinkExpandMaxVf<float, float, LAYOUT_T == FLASH_MLA_LAYOUT::BNSD>(
                maxUb, sinkTmp, gs1Start, runInfo.actVecMSize, runInfo.actS1Size, constInfo_.gSize);
            if constexpr (hasAtten) {
                int64_t firstValid =
                    AttentionCommon::Min(AttentionCommon::Max(-runInfo.winRightFromTopLeft, 0), runInfo.actS1Size);
                int64_t endValid = AttentionCommon::Min(
                    AttentionCommon::Max(runInfo.winLeftFromTopLeft + runInfo.actS2Size, 0), runInfo.actS1Size);
                if (firstValid > 0 || endValid < runInfo.actS1Size) {
                    AscendC::PipeBarrier<PIPE_V>();
                    MaskInvalidSinkRows<LAYOUT_T == FLASH_MLA_LAYOUT::BNSD>(
                        (__ubuf__ float*)maxUb.GetPhyAddr(), (__ubuf__ float*)sumUb.GetPhyAddr(), gs1Start,
                        runInfo.actVecMSize, runInfo.actS1Size, constInfo_.gSize, static_cast<int32_t>(firstValid),
                        static_cast<int32_t>(endValid));
                }
            }
            Mutex::Unlock<PIPE_V>(EVENT_ID0);
        } else {
            Duplicate<float>(sumUb, 0.0f, softmaxBufElementCount);
            Duplicate<float>(maxUb, -std::numeric_limits<float>::infinity(), softmaxBufElementCount);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        // 预置 BMM1/BMM2 可用标志，避免 AIC 首次 Wait 阻塞。
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2_1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1_0);
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline bool AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           FlashMlaRunInfoX& runInfo)
    {
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.winLeft = constInfo_.winLeft;
        maskInfo.winRight = constInfo_.winRight;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = (constInfo_.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (128U - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskFormat = MASK_LAYOUT;

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);
        const uint32_t maskBufId = runInfo.loop % DB;
        if (IsSkipMask && IsSkipMaskForPre) {
            // Fully visible tiles use the unmasked VF and do not access the mask slots.
            return false;
        }

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, 128U>(attenMaskUb, attenMaskGmInt_, maskInfo, false,
                                                                  UB_IN_MASK_EVENT0 + maskBufId);
        } else {
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * 128U);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        if (!IsSkipMaskForPre) {
            const uint32_t preBufId = maskBufId ^ 1U;
            LocalTensor<uint8_t> attenMaskUbPre = ubMaskBuffers_[preBufId * UB_MASK_BUF_BYTES];
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, 128U>(attenMaskUbPre, attenMaskGmInt_, maskInfo, true,
                                                                  UB_IN_MASK_EVENT0 + preBufId);
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + preBufId);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, 128U);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + preBufId);
        }
        return true;
    }
};

template <typename FA_T>
class FlashMlaNoQuantMlaBlockVecDummy {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr bool HAS_MASK = FA_T::hasMask;
    static constexpr bool FLASH_DECODE = FA_T::flashDecode;
    using OUT_T = OUTPUT_T;
    using ConstInfoNoQuant = ConstInfo_t<FlashMlaKernelType::NO_QUANT>;
    template <typename FlashMlaSeqLensToolT>
    __aicore__ inline FlashMlaNoQuantMlaBlockVecDummy(ConstInfoNoQuant& constInfo, FlashMlaSeqLensToolT& seqLensTool){};
};

} // namespace FlashMlaKernel

#endif // FLASH_MLA_BLOCK_VEC_ARCH92_H_
