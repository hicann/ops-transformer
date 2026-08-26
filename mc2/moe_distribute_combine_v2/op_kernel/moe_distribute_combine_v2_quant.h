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
 * \file moe_distribute_v2_quant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_QUANT_H
#define MOE_DISTRIBUTE_COMBINE_V2_QUANT_H

#if __has_include("../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#endif
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if __has_include("../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_common.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_v2_common.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_common.h"
#endif
#if __has_include("../common/quantize_functions.h")
#include "../common/quantize_functions.h"
#else
#include "../../common/op_kernel/quantize_functions.h"
#endif
#endif

namespace Mc2Kernel {
using namespace AscendC;
using namespace MoeDistributeV2Base;

template <typename ExpandXType, typename XType, typename ExpandIdxType, uint8_t QuantMode, bool HasAddRmsNorm>
class MoeDistributeCombineQuant {
public:
    float scaleValFloat_;
    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> floatLocalTemp_;
    uint32_t axisH_{0};
    uint32_t mask_{0};
    uint32_t repeatNum_{0};
    uint32_t hAlign32Size_{0};
    uint32_t quantScaleNum_{0};
    LocalTensor<int8_t> castLocalTensor_;
    LocalTensor<XType> scaleDivTensor_;

    __aicore__ inline MoeDistributeCombineQuant() = default;

    __aicore__ inline void SetQuantInitParams(LocalTensor<float> winTpSendCountFloatTensor,
                                              LocalTensor<half> fp16CastTensor, LocalTensor<float> absFloatTensor,
                                              LocalTensor<float> reduceMaxFloatTensor,
                                              LocalTensor<float> scaleDupLocalTensor)
    {
        winTpSendCountFloatTensor_ = winTpSendCountFloatTensor;
        floatLocalTemp_ = winTpSendCountFloatTensor;
        fp16CastTensor_ = fp16CastTensor;
        absFloatTensor_ = absFloatTensor;
        reduceMaxFloatTensor_ = reduceMaxFloatTensor;
        scaleDupLocalTensor_ = scaleDupLocalTensor;
    }

    __aicore__ inline void SetDeQuantInitParams(LocalTensor<half> fp16CastTensor, LocalTensor<float> absFloatTensor,
                                                LocalTensor<float> scaleDupLocalTensor,
                                                LocalTensor<float> scaleDivFloatTensor)
    {
        fp16CastTensor_ = fp16CastTensor;
        absFloatTensor_ = absFloatTensor;
        scaleDupLocalTensor_ = scaleDupLocalTensor;
        scaleDivFloatTensor_ = scaleDivFloatTensor;
    }

    __aicore__ inline void QuantInit(uint32_t &scaleNum_, uint32_t &hExpandXAlign32Size_, uint32_t &hExpandXAlignSize_,
                                     uint32_t &scaleNumAlignSize_, uint32_t &hFloatAlign256Size_,
                                     uint32_t &tokenScaleCnt_, uint32_t axisH)
    {
        axisH_ = axisH;
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
            scaleValFloat_ = static_cast<float>(1.0f / SCALE_PARAM);
            uint32_t scaleGranu =
                static_cast<uint32_t>(UB_ALIGN / sizeof(float)); // 计算每个block得到的reducemax结果数量
            quantScaleNum_ = (hExpandXAlign32Size_ / sizeof(ExpandXType)) / scaleGranu; // 得到有效scale的个数
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            // A5 INT8融合反量化按512B覆盖尾段，scale数量需与Brcb的repeat覆盖范围一致
            quantScaleNum_ = (hFloatAlign256Size_ / sizeof(float)) / scaleGranu;
#endif
            scaleNum_ = quantScaleNum_;
            hExpandXAlignSize_ = hExpandXAlign32Size_;
            scaleNumAlignSize_ = Ceil(scaleNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
            repeatNum_ = static_cast<uint32_t>(hFloatAlign256Size_ / ALIGNED_LEN); // 每次256b参与计算
            mask_ = static_cast<uint32_t>(ALIGNED_LEN / sizeof(float));
            tokenScaleCnt_ = hAlign32Size_ / sizeof(ExpandXType) + quantScaleNum_; // int8_align + scale有效个数
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
            hExpandXAlignSize_ = Align128(axisH) * sizeof(ExpandXType);
            quantScaleNum_ = Align2(Ceil32(axisH));
            scaleNum_ = quantScaleNum_;
            scaleNumAlignSize_ = Align128(scaleNum_) * sizeof(ExpandXType) * BUFFER_NUM; // 双搬
            tokenScaleCnt_ = Align256(axisH) / sizeof(ExpandXType) + scaleNum_;
        }
#endif
    }

    __aicore__ inline void Int8QuantProcess(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal)
    {
        castLocalTensor_ = outLocal.template ReinterpretCast<int8_t>(); // 长度为int8H_Align + scaleNum
        scaleDivTensor_ = castLocalTensor_[hAlign32Size_].template ReinterpretCast<ExpandXType>(); // 偏移前面的int8

        Cast(winTpSendCountFloatTensor_, inLocal, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Abs(absFloatTensor_, winTpSendCountFloatTensor_,
            axisH_); // absFloatTensor_ align到256并写0，支持ReduceMax与Brcb
        PipeBarrier<PIPE_V>();
        BlockReduceMax(reduceMaxFloatTensor_, absFloatTensor_, repeatNum_, mask_, 1, 1, BLOCK_NUM); // 32->1 256->8
        PipeBarrier<PIPE_V>();
        Muls(reduceMaxFloatTensor_, reduceMaxFloatTensor_, scaleValFloat_, quantScaleNum_); // 有效个数
        PipeBarrier<PIPE_V>();
        Cast(scaleDivTensor_, reduceMaxFloatTensor_, RoundMode::CAST_RINT, quantScaleNum_); // 有效个数
        PipeBarrier<PIPE_V>();
        Brcb(scaleDupLocalTensor_, reduceMaxFloatTensor_, repeatNum_, {1, BLOCK_NUM}); // 一次256
        PipeBarrier<PIPE_V>();
        Div(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, scaleDupLocalTensor_, axisH_); // 有效个数
        PipeBarrier<PIPE_V>();
        Cast(fp16CastTensor_, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(castLocalTensor_, fp16CastTensor_, RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
    }

    __aicore__ inline void Int8DequantProcess(LocalTensor<XType> &inLocal, LocalTensor<XType> &outLocal)
    {
        castLocalTensor_ = inLocal.template ReinterpretCast<int8_t>();
        scaleDivTensor_ = inLocal[hAlign32Size_ / INT8_DIVIVE];
        SyncFunc<AscendC::HardEvent::S_V>();
        Cast(scaleDivFloatTensor_, scaleDivTensor_, RoundMode::CAST_NONE, quantScaleNum_);
        Cast(fp16CastTensor_, castLocalTensor_, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(absFloatTensor_, fp16CastTensor_, RoundMode::CAST_NONE, axisH_);
        Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_, repeatNum_, {1, BLOCK_NUM});
        PipeBarrier<PIPE_V>();
        Mul(absFloatTensor_, absFloatTensor_, scaleDupLocalTensor_, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, absFloatTensor_, RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    template <bool ApplyExpertScale = true>
    __aicore__ inline void Int8DequantProcessA5(LocalTensor<XType> &inLocal, LocalTensor<float> &sumFinalTensor,
                                                float scaleVal)
    {
        LocalTensor<int8_t> tokenInt8Tensor = inLocal.template ReinterpretCast<int8_t>();
        scaleDivTensor_ = inLocal[hAlign32Size_ / INT8_DIVIVE];
        SyncFunc<AscendC::HardEvent::S_V>();
        Cast(scaleDivFloatTensor_, scaleDivTensor_, RoundMode::CAST_NONE, quantScaleNum_);
        PipeBarrier<PIPE_V>();
        Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_, repeatNum_, {1, BLOCK_NUM});
        PipeBarrier<PIPE_V>();

        __ubuf__ int8_t *tokenPtr0 = (__ubuf__ int8_t *)tokenInt8Tensor.GetPhyAddr();
        __ubuf__ float *dyScaleFp32Ptr = (__ubuf__ float *)scaleDupLocalTensor_.GetPhyAddr();
        __ubuf__ float *sumFinalDstPtr = (__ubuf__ float *)sumFinalTensor.GetPhyAddr();

        uint32_t fp32RepeatSize = Quant::GetVRegSizeDispatch() / sizeof(float);
        uint32_t elementsPerRepeat = fp32RepeatSize * INT8_DIVIVE;
        uint16_t fp32RepeatTimes = Ceil(axisH_, elementsPerRepeat);
        uint32_t tailElements = axisH_ % elementsPerRepeat;
        bool hasTail = tailElements != 0U;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int8_t> tokenSrcReg;
            AscendC::MicroAPI::RegTensor<half> tokenHalfReg;
            AscendC::MicroAPI::RegTensor<float> tokFp32_1;
            AscendC::MicroAPI::RegTensor<float> tokFp32_2;
            AscendC::MicroAPI::RegTensor<float> dyScaleFp32_1;
            AscendC::MicroAPI::RegTensor<float> dyScaleFp32_2;
            AscendC::MicroAPI::RegTensor<float> deqFp32_1;
            AscendC::MicroAPI::RegTensor<float> deqFp32_2;
            AscendC::MicroAPI::RegTensor<XType> deqXType_1;
            AscendC::MicroAPI::RegTensor<XType> deqXType_2;
            AscendC::MicroAPI::RegTensor<float> sumLocal_1;
            AscendC::MicroAPI::RegTensor<float> sumLocal_2;
            AscendC::MicroAPI::RegTensor<float> sumFinal_1;
            AscendC::MicroAPI::RegTensor<float> sumFinal_2;

            static constexpr AscendC::MicroAPI::CastTrait castZero = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castOne = {
                AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castRint = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

            AscendC::MicroAPI::MaskReg maskF32All =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskF16All =
                AscendC::MicroAPI::CreateMask<half, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskF32Tail;
            AscendC::MicroAPI::MaskReg maskF16Tail;
            if (hasTail) {
                uint32_t tailF32Elements = Ceil(tailElements, INT8_DIVIVE);
                maskF32Tail = AscendC::MicroAPI::UpdateMask<float>(tailF32Elements);
                maskF16Tail = AscendC::MicroAPI::UpdateMask<half>(tailElements);
            }
            AscendC::MicroAPI::MaskReg maskF32;
            AscendC::MicroAPI::MaskReg maskF16;

            for (uint16_t i = 0; i < fp32RepeatTimes; ++i) {
                bool isTailRepeat = hasTail && (i == fp32RepeatTimes - 1U);
                if (isTailRepeat) {
                    maskF32 = maskF32Tail;
                    maskF16 = maskF16Tail;
                } else {
                    maskF32 = maskF32All;
                    maskF16 = maskF16All;
                }

                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    dyScaleFp32_1, dyScaleFp32_2, dyScaleFp32Ptr + INT8_DIVIVE * i * fp32RepeatSize);
                MicroAPI::DataCopy<int8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                    tokenSrcReg, tokenPtr0 + INT8_DIVIVE * i * fp32RepeatSize);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    sumFinal_1, sumFinal_2, sumFinalDstPtr + INT8_DIVIVE * i * fp32RepeatSize);

                MicroAPI::Cast<half, int8_t, castZero>(tokenHalfReg, tokenSrcReg, maskF16);
                MicroAPI::Cast<float, half, castZero>(tokFp32_1, tokenHalfReg, maskF16);
                MicroAPI::Cast<float, half, castOne>(tokFp32_2, tokenHalfReg, maskF16);

                MicroAPI::Mul(deqFp32_1, dyScaleFp32_1, tokFp32_1, maskF32);
                MicroAPI::Mul(deqFp32_2, dyScaleFp32_2, tokFp32_2, maskF32);

                MicroAPI::Cast<XType, float, castRint>(deqXType_1, deqFp32_1, maskF32);
                MicroAPI::Cast<XType, float, castRint>(deqXType_2, deqFp32_2, maskF32);
                MicroAPI::Cast<float, XType, castZero>(deqFp32_1, deqXType_1, maskF32);
                MicroAPI::Cast<float, XType, castZero>(deqFp32_2, deqXType_2, maskF32);

                if constexpr (ApplyExpertScale) {
                    MicroAPI::Muls(sumLocal_1, deqFp32_1, scaleVal, maskF32);
                    MicroAPI::Muls(sumLocal_2, deqFp32_2, scaleVal, maskF32);
                    MicroAPI::Add(sumFinal_1, sumFinal_1, sumLocal_1, maskF32);
                    MicroAPI::Add(sumFinal_2, sumFinal_2, sumLocal_2, maskF32);
                } else {
                    MicroAPI::Add(sumFinal_1, sumFinal_1, deqFp32_1, maskF32);
                    MicroAPI::Add(sumFinal_2, sumFinal_2, deqFp32_2, maskF32);
                }

                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
                    sumFinalDstPtr + i * fp32RepeatSize * INT8_DIVIVE, sumFinal_1, sumFinal_2, maskF32);
            }
        }
    }
    __aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal)
    {
        uint32_t mxScaleNum = Align2(Ceil32(axisH_));
        __ubuf__ ExpandXType *srcAddr = (__ubuf__ ExpandXType *)inLocal.GetPhyAddr();
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)floatLocalTemp_.GetPhyAddr();
        __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)floatLocalTemp_[Align32(mxScaleNum)].GetPhyAddr();
        __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
        __ubuf__ uint16_t *mxScaleLocalAddr =
            (__ubuf__ uint16_t *)outLocal[Align256<uint32_t>(axisH_) / INT8_DIVIVE].GetPhyAddr();
        Quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_); // 计算最大Exp
        if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT) {
            // 计算scales并填充
            Quant::ComputeScale<fp8_e5m2_t>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
            Quant::ComputeFp8Data<ExpandXType, fp8_e5m2_t, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleLocalAddr, outLocalAddr,
                                                                 axisH_); // 计算量化后的expandx并填充
        } else if constexpr (QuantMode == MXFP8_E4M3_COMM_QUANT) {
            // 计算scales并填充
            Quant::ComputeScale<fp8_e4m3fn_t>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
            Quant::ComputeFp8Data<ExpandXType, fp8_e4m3fn_t, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleLocalAddr, outLocalAddr,
                                                                 axisH_); // 计算量化后的expandx并填充
        }
    }

    template <typename T, bool ApplyExpertScale = true>
    __aicore__ inline void DeQuantMxFp8(LocalTensor<XType> &inLocal, LocalTensor<float> &sumTensor,
                                        LocalTensor<float> &sumFinalTensor, float scaleVal)
    {
        LocalTensor<T> castFp8LocalTensor_ = inLocal.template ReinterpretCast<T>();
        // bf16/fp16量化为mxfp8后，字节差2倍
        LocalTensor<fp8_e8m0_t> scaleDivFp8Tensor_ =
            inLocal[Align256<uint32_t>(axisH_) / INT8_DIVIVE].template ReinterpretCast<fp8_e8m0_t>();
        __ubuf__ float *dyScaleFp32Ptr = (__ubuf__ float *)scaleDupLocalTensor_.GetPhyAddr(); // 大小是h*2字节
        __ubuf__ fp8_e8m0_t *srcPtr0 = (__ubuf__ fp8_e8m0_t *)scaleDivFp8Tensor_.GetPhyAddr();
        __ubuf__ T *tokenPtr0 = (__ubuf__ T *)castFp8LocalTensor_.GetPhyAddr();
        __ubuf__ float *sumFinalDstPtr = (__ubuf__ float *)sumFinalTensor.GetPhyAddr();
        uint32_t fp32RepeatSize = Quant::GetVRegSizeDispatch() / sizeof(float); // 64
        uint16_t repeatTimes = Ceil(quantScaleNum_, fp32RepeatSize);
        uint16_t fp32RepeatTimes = Ceil(axisH_, fp32RepeatSize * INT8_DIVIVE);
        uint32_t axisHx4 = 4 * axisH_; // 4为fp32与fp8字节数比例，用于换算fp8 mask粒度
        const int16_t expShift = 23;   // fp32指数字段起始位，e8m0指数左移到该字段
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<fp8_e8m0_t> vSrcReg;
            AscendC::MicroAPI::RegTensor<T> tokenSrcReg;
            AscendC::MicroAPI::RegTensor<float> tokenFp32SrcReg_1;
            AscendC::MicroAPI::RegTensor<float> tokenFp32SrcReg_2;
            AscendC::MicroAPI::RegTensor<float> dyScaleFp32Reg;
            AscendC::MicroAPI::RegTensor<float> sumLocalDstReg_1;
            AscendC::MicroAPI::RegTensor<float> sumLocalDstReg_2;
            AscendC::MicroAPI::RegTensor<float> sumFinalDstReg_1;
            AscendC::MicroAPI::RegTensor<float> sumFinalDstReg_2;
            static constexpr AscendC::MicroAPI::CastTrait castTrait0 = {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
            static constexpr AscendC::MicroAPI::CastTrait castTrait2 = {
                AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::UNKNOWN,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::MaskReg maskReg2;
            AscendC::MicroAPI::MaskReg maskReg3;
            // read all the scales, cast all the e8m0 scale to FP32, store them
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                maskReg = AscendC::MicroAPI::UpdateMask<float>(quantScaleNum_);
                MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vSrcReg,
                                                                                    srcPtr0 + i * fp32RepeatSize);
                // cast e8m0 to FP32
                MicroAPI::ShiftLefts((AscendC::MicroAPI::RegTensor<uint32_t> &)dyScaleFp32Reg,
                                     (AscendC::MicroAPI::RegTensor<uint32_t> &)vSrcReg, expShift, maskReg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
                    dyScaleFp32Ptr + i * fp32RepeatSize * INT8_DIVIVE, dyScaleFp32Reg, dyScaleFp32Reg, maskReg);
            }

            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < fp32RepeatTimes; ++i) {
                maskReg2 = AscendC::MicroAPI::UpdateMask<float>(axisH_);
                maskReg3 = AscendC::MicroAPI::UpdateMask<T>(axisHx4);
                // 广播4B->32B，8倍
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
                // 接收到的token float8_e5m2_t
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B8>(tokenSrcReg,
                                                                          tokenPtr0 + INT8_DIVIVE * i * fp32RepeatSize);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    sumFinalDstReg_1, sumFinalDstReg_2, sumFinalDstPtr + INT8_DIVIVE * i * fp32RepeatSize);
                MicroAPI::Cast<float, T, castTrait0>(tokenFp32SrcReg_1, tokenSrcReg, maskReg3);
                MicroAPI::Cast<float, T, castTrait2>(tokenFp32SrcReg_2, tokenSrcReg, maskReg3);
                // token与量化参数相乘
                MicroAPI::Mul(sumLocalDstReg_1, dyScaleFp32Reg, tokenFp32SrcReg_1, maskReg2);
                MicroAPI::Mul(sumLocalDstReg_2, dyScaleFp32Reg, tokenFp32SrcReg_2, maskReg2);
                if constexpr (ApplyExpertScale) {
                    MicroAPI::Muls(sumLocalDstReg_1, sumLocalDstReg_1, scaleVal, maskReg2);
                    MicroAPI::Muls(sumLocalDstReg_2, sumLocalDstReg_2, scaleVal, maskReg2);
                }
                MicroAPI::Add(sumFinalDstReg_1, sumFinalDstReg_1, sumLocalDstReg_1, maskReg2); // combine累加
                MicroAPI::Add(sumFinalDstReg_2, sumFinalDstReg_2, sumLocalDstReg_2, maskReg2); // combine累加
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
                    sumFinalDstPtr + i * fp32RepeatSize * INT8_DIVIVE, sumFinalDstReg_1, sumFinalDstReg_2,
                    maskReg2); // 最后搬出 float类型
            }
        }
    }
#endif
    __aicore__ inline void QuantProcess(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal)
    {
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            Int8QuantProcess(outLocal, inLocal);
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT || QuantMode == MXFP8_E4M3_COMM_QUANT) {
            QuantMxFp8(outLocal, inLocal);
        }
#endif
    }
    __aicore__ inline void DeQuantProcess(LocalTensor<XType> &inLocal, LocalTensor<XType> &outLocal,
                                          LocalTensor<float> &sumTensor, LocalTensor<float> &sumFinalTensor,
                                          float scaleVal)
    {
        if constexpr (QuantMode == INT8_COMM_QUANT) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            // A5 走反量化及加权求和融合
            Int8DequantProcessA5(inLocal, sumFinalTensor, scaleVal);
#else
            Int8DequantProcess(inLocal, outLocal);
#endif
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT) {
            DeQuantMxFp8<fp8_e5m2_t>(inLocal, sumTensor, sumFinalTensor, scaleVal);
        } else if constexpr (QuantMode == MXFP8_E4M3_COMM_QUANT) {
            DeQuantMxFp8<fp8_e4m3fn_t>(inLocal, sumTensor, sumFinalTensor, scaleVal);
        }
#endif
    }

    __aicore__ inline void DeQuantProcessWithoutExpertScale(LocalTensor<XType> &inLocal, LocalTensor<XType> &outLocal,
                                                            LocalTensor<float> &sumTensor,
                                                            LocalTensor<float> &sumFinalTensor)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            Int8DequantProcessA5<false>(inLocal, sumFinalTensor, 0.0f);
        } else if constexpr (QuantMode == MXFP8_E5M2_COMM_QUANT) {
            DeQuantMxFp8<fp8_e5m2_t, false>(inLocal, sumTensor, sumFinalTensor, 0.0f);
        } else if constexpr (QuantMode == MXFP8_E4M3_COMM_QUANT) {
            DeQuantMxFp8<fp8_e4m3fn_t, false>(inLocal, sumTensor, sumFinalTensor, 0.0f);
        }
#else
        if constexpr (QuantMode == INT8_COMM_QUANT) {
            Int8DequantProcess(inLocal, outLocal);
        }
#endif
    }
};
} // namespace Mc2Kernel
#endif // MOE_DISTRIBUTE_V2_QUANT_H
