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
 * \file weight_quant_vec_compute.h
 * \brief
 */
#ifndef GMM_FR_WEIGHT_QUANT_VEC_COMPUTE_H
#define GMM_FR_WEIGHT_QUANT_VEC_COMPUTE_H

#include "../common/basic_block_config.h"
#include "../common/basic_block_vf_mx.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "weight_quant_tool.h"
#include "gmm_fr_mx_a8w4_vf.h"

using AscendC::BLOCK_CUBE;
using AscendC::CacheMode;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::GetRound;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::TypeGet;

namespace WeightQuantBatchMatmulV2::Arch35 {
struct GmmFrUbBufferInfo {
    uint64_t weightHighBitBufferNum;
    uint64_t weightHighBitTotalSize;
    uint64_t weightHighBitSingleBufferSize;
    uint64_t weightLowbitTotalSize;
    uint64_t weightLowBitSingleBufferSize;
    uint64_t biasTotalSize;
    uint64_t biasSingleBufferSize;
    uint64_t biasRescaleTotalSize;
    uint64_t biasRescaleSingleBufferSize;
    uint64_t logitsTotalSize;
    uint64_t logitsSingleBufferSize;
    uint64_t rowIndexTotalSize;
    uint64_t rowIndexSingleBufferSize;
};

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr GmmFrUbBufferInfo GetGmmFRMxA8W4BufferInfo()
{
    return {.weightHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
            .weightHighBitTotalSize = 160 * GetKBUnit<int8_t>(), // 160KB
            .weightHighBitSingleBufferSize = 160 * GetKBUnit<int8_t>() / QUADRUPLE_BUFFER_NUM,
            .weightLowbitTotalSize = 80 * GetKBUnit<int8_t>(), // 80KB
            .weightLowBitSingleBufferSize = 80 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .biasTotalSize = 1 * GetKBUnit<half>(),                                    // 1KB
            .biasSingleBufferSize = 1 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum, // 2KB
            .biasRescaleTotalSize = 1 * GetKBUnit<half>(),                             // 1KB
            .biasRescaleSingleBufferSize = 1 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .logitsTotalSize = 1 * GetKBUnit<float>(), // 1KB
            .logitsSingleBufferSize = 1 * GetKBUnit<float>() / DOUBLE_BUFFER_NUM,
            .rowIndexTotalSize = 2 * GetKBUnit<int64_t>(), // 2KB
            .rowIndexSingleBufferSize = 2 * GetKBUnit<int64_t>() / DOUBLE_BUFFER_NUM};
}

#define GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM \
    template <typename xType, typename wType, typename biasType, typename yType, typename sharedInputDType, \
              typename logitsType, typename rowIndexType, bool hasBias, const WqmmConfig &wqmmConfig, \
              const VecAntiQuantConfig &vecConfig>

#define GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS \
    GMMFRWeightQuantVecCompute<xType, wType, biasType, yType, sharedInputDType, logitsType, rowIndexType, hasBias, \
                               wqmmConfig, vecConfig>

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
class GMMFRWeightQuantVecCompute {
public:
    __aicore__ inline GMMFRWeightQuantVecCompute(){};

    __aicore__ inline void UpdateGlobalAddr(__gm__ wType *weight, __gm__ biasType *bias, __gm__ logitsType *logits,
                                            __gm__ rowIndexType *rowIndex, const bool weightL2Cacheable);
    __aicore__ inline void Init(float sharedInputWeight, __gm__ yType *yFp32Addr);
    __aicore__ inline void WaitVToMTE2();
    __aicore__ inline void SetVToMTE2();
    __aicore__ inline void CopyShareInputGmToUb(uint64_t sharedInputGmOffset, uint64_t initSharedInputRealSize,
                                                __gm__ sharedInputDType *shareInputAddr);
    __aicore__ inline void CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset, uint64_t kL1Offset,
                                      uint64_t biasGmOffset, uint64_t biasRealSize,
                                      const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyShareInputUbToGm(uint64_t sharedInputGmOffset, uint64_t initSharedInputRealSize);
    __aicore__ inline void WeightAntiQuantComputeNzNk(uint64_t kRealSize, uint64_t biasRealSize, uint64_t kGmOffset,
                                                      const LocalTensor<xType> &weightHighBitL1,
                                                      const LocalTensor<biasType> &biasL1,
                                                      const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void MulLogits(const LocalTensor<float> &ubOutputF32Buffer, uint64_t mRealSize,
                                     const BasicBlockOffsetParam &offsetParam, uint64_t rlLoopIdx);
    __aicore__ inline void WaitSToMTE2(uint64_t rlLoopIdx);
    __aicore__ inline void SetSToMTE2(uint64_t rlLoopIdx);
    __aicore__ inline void WaitFrVToMTE2(uint64_t rlLoopIdx);
    __aicore__ inline void SetFrToMTE2(uint64_t rlLoopIdx);
    __aicore__ inline void SetMte2ToS(uint64_t rlLoopIdx);
    __aicore__ inline void WaitMte2ToS(uint64_t rlLoopIdx);
    __aicore__ inline void End();
    __aicore__ inline void RoutingYToGm(uint64_t mRealSize, const LocalTensor<float> &ubOutputF32Buffer,
                                        const BasicBlockOffsetParam &offsetParam, uint64_t rlLoopIdx);
    __aicore__ inline void CopyRowIndexLogitsGmToUb(uint64_t mGmOffset, uint64_t mRealSize, uint64_t rlLoopIdx);

    __aicore__ inline void InitGmToZero(uint64_t yGmOffset, uint64_t initZeroSize);

private:
    __aicore__ inline void AntiQuantProcessNzMxA8W4(uint64_t biasRealSize, uint64_t ubMte2KSize, uint64_t kGmOffset,
                                                    const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyWeightHighBitForAligned(uint64_t antiQuantRealN, uint64_t antiQuantRealK,
                                                       const LocalTensor<xType> &weightHighBitL1);
    constexpr static GmmFrUbBufferInfo UB_BUFFER_INFO = GetGmmFRMxA8W4BufferInfo<vecConfig>();
    // mte2搬运计数，用于控制weight输入的buffer和 mte2&&V间同步控制
    uint64_t weightMte2LoopIdx_ = 0;
    // vf中标准计算单元(vfNStandardLen, vfKStandardLen)的计数，用于控制weight反量化后输出的buffer和V&&mte3间同步控制
    uint64_t ubComputeLoopIdx_ = 0;

    static constexpr uint32_t EVENT_ID_V_TO_MTE2 = 0;
    static constexpr uint32_t EVENT_ID_FR_V_TO_MTE2 = EVENT_ID_V_TO_MTE2 + QUADRUPLE_BUFFER_NUM;

    static constexpr uint32_t EVENT_ID_MTE2_TO_V = 0;
    static constexpr uint32_t EVENT_ID_RL_MTE2_TO_V = 1;

    static constexpr uint32_t EVENT_ID_MTE3_TO_V = 0;
    static constexpr uint32_t EVENT_ID_BIAS_FR_MTE3_TO_V = EVENT_ID_MTE3_TO_V + QUADRUPLE_BUFFER_NUM;

    static constexpr uint32_t EVENT_ID_V_TO_MTE3 = 0;

    static constexpr uint32_t EVENT_ID_S_TO_MTE2 = 0;

    static constexpr uint32_t EVENT_ID_MTE2_TO_S = 0;

    static constexpr uint32_t QUADRUPLE_BUFFER_MASK = QUADRUPLE_BUFFER_NUM - 1;

    float sharedInputWeight_ = 0.0f;

    GlobalTensor<wType> wGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yFp32Global_;
    GlobalTensor<rowIndexType> rowIndexGlobal_;
    GlobalTensor<logitsType> logitsGlobal_;

    LocalTensor<int8_t> weightLowBit_;
    LocalTensor<xType> weightHighBit_;
    LocalTensor<biasType> bias_;
    LocalTensor<biasType> biasRescale_;
    LocalTensor<rowIndexType> rowIndex_;
    LocalTensor<logitsType> logits_;

    constexpr static uint32_t C0_SIZE =
        (IsSameType<xType, int8_t>::value || IsSameType<xType, fp8_e4m3fn_t>::value) ? C0_SIZE_B8 : BLOCK_CUBE;
    constexpr static uint64_t VEC_REG_ELEM =
        IsMxA8W4<xType, wqmmConfig.antiQuantType>() ? VECTOR_REG_WIDTH : VEC_MAX_ELEM_B16;

    constexpr static uint64_t UB_AVAILABLE_SIZE = 248 * GetKBUnit<int8_t>();

    constexpr static VfConfig VF_CONFIG = GetVfConfig<xType, wqmmConfig, vecConfig>();

    constexpr static uint64_t ANTIQUANT_Y_STANDARD_N_SIZE = VECTOR_REG_WIDTH / sizeof(int32_t);
};

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::UpdateGlobalAddr(
    __gm__ wType *weight, __gm__ biasType *bias, __gm__ logitsType *logits, __gm__ rowIndexType *rowIndex,
    const bool weightL2Cacheable)
{
    wGlobal_.SetGlobalBuffer(weight);
    rowIndexGlobal_.SetGlobalBuffer(rowIndex);
    logitsGlobal_.SetGlobalBuffer(logits);
    if (!weightL2Cacheable) {
        wGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    if constexpr (hasBias) {
        biasGlobal_.SetGlobalBuffer(bias);
    }
}

/*
 * 初始化buffer和同步所需的EventID
 */
GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::Init(float sharedInputWeight,
                                                                               __gm__ yType *yFp32Addr)
{
    sharedInputWeight_ = sharedInputWeight;
    yFp32Global_.SetGlobalBuffer(yFp32Addr);
    // MxA8W4
    weightLowBit_ = LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightLowbitTotalSize);
    uint64_t ubOffset = UB_BUFFER_INFO.weightLowbitTotalSize;
    weightHighBit_ = LocalTensor<xType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.weightHighBitTotalSize);
    ubOffset += UB_BUFFER_INFO.weightHighBitTotalSize;

    bias_ = LocalTensor<biasType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.biasTotalSize);
    ubOffset += UB_BUFFER_INFO.biasTotalSize * sizeof(biasType);
    biasRescale_ = LocalTensor<biasType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.biasRescaleTotalSize);
    ubOffset += UB_BUFFER_INFO.biasRescaleTotalSize * sizeof(biasType);
    rowIndex_ = LocalTensor<rowIndexType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.rowIndexTotalSize);
    ubOffset += UB_BUFFER_INFO.rowIndexTotalSize * sizeof(rowIndexType);
    logits_ = LocalTensor<logitsType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.logitsTotalSize);

    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < DOUBLE_BUFFER_NUM; idx++) {
        SetFlag<HardEvent::S_MTE2>(EVENT_ID_S_TO_MTE2 + idx);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_FR_V_TO_MTE2 + idx);
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::InitGmToZero(uint64_t yGmOffset,
                                                                                       uint64_t initZeroSize)
{
    uint64_t bufId = ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + bufId);

    InitZeroVf((__ubuf__ float *)weightHighBit_.GetPhyAddr(bufId * VECTOR_REG_WIDTH),
               CeilDivide(initZeroSize, VEC_MAX_ELEM_B32));

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    // 判断是否存在尾块，若存在则单独搬运
    uint64_t mte3TailSize = initZeroSize % VEC_MAX_ELEM_B32;
    uint64_t mte3MainCount = initZeroSize / VEC_MAX_ELEM_B32;
    // 主块的搬运逻辑
    if (mte3MainCount > 0) {
        DataCopyPad2D(yFp32Global_[yGmOffset],
                      weightHighBit_[bufId * VECTOR_REG_WIDTH].template ReinterpretCast<float>(), mte3MainCount,
                      VEC_MAX_ELEM_B32, VEC_MAX_ELEM_B32 * QUADRUPLE_BUFFER_NUM, VEC_MAX_ELEM_B32);
    }
    // 尾块的搬运逻辑
    if (mte3TailSize > 0) {
        DataCopyPad2D(yFp32Global_[yGmOffset + mte3MainCount * VEC_MAX_ELEM_B32],
                      weightHighBit_[bufId * VECTOR_REG_WIDTH].template ReinterpretCast<float>(), 1, mte3TailSize,
                      VEC_MAX_ELEM_B32, VEC_MAX_ELEM_B32);
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + bufId);
    ubComputeLoopIdx_++;
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyShareInputGmToUb(
    uint64_t sharedInputGmOffset, uint64_t initSharedInputRealSize, __gm__ sharedInputDType *shareInputAddr)
{
    if (initSharedInputRealSize == 0) {
        return;
    }
    GlobalTensor<sharedInputDType> sharedInputGlobal;
    sharedInputGlobal.SetGlobalBuffer(shareInputAddr);
    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize]
            .template ReinterpretCast<sharedInputDType>(),
        sharedInputGlobal[sharedInputGmOffset], 1, initSharedInputRealSize, initSharedInputRealSize,
        initSharedInputRealSize);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyShareInputUbToGm(
    uint64_t sharedInputGmOffset, uint64_t initSharedInputRealSize)
{
    uint64_t vfBufId = ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + vfBufId);
    CastAndMulWithSharedWeightVf<sharedInputDType>(
        (__ubuf__ float *)weightHighBit_.GetPhyAddr(vfBufId * VECTOR_REG_WIDTH),
        (__ubuf__ sharedInputDType *)weightLowBit_.GetPhyAddr((weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                              UB_BUFFER_INFO.weightLowBitSingleBufferSize),
        CeilDivide(initSharedInputRealSize, VEC_MAX_ELEM_B32), sharedInputWeight_);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    uint64_t mte3TailSize = initSharedInputRealSize % VEC_MAX_ELEM_B32;
    uint64_t mte3MainCount = initSharedInputRealSize / VEC_MAX_ELEM_B32;
    // 主块搬运
    if (mte3MainCount > 0) {
        DataCopyPad2D(yFp32Global_[sharedInputGmOffset],
                      weightHighBit_[vfBufId * VECTOR_REG_WIDTH].template ReinterpretCast<float>(), mte3MainCount,
                      VEC_MAX_ELEM_B32, VEC_MAX_ELEM_B32 * QUADRUPLE_BUFFER_NUM, VEC_MAX_ELEM_B32);
    }
    // 尾块搬运
    if (mte3TailSize > 0) {
        DataCopyPad2D(
            yFp32Global_[sharedInputGmOffset + mte3MainCount * VEC_MAX_ELEM_B32],
            weightHighBit_[vfBufId * VECTOR_REG_WIDTH + mte3MainCount * VECTOR_REG_WIDTH * QUADRUPLE_BUFFER_NUM]
                .template ReinterpretCast<float>(),
            1, mte3TailSize, VEC_MAX_ELEM_B32, VEC_MAX_ELEM_B32);
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + vfBufId);
    ubComputeLoopIdx_++;
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WaitVToMTE2()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)));
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::SetVToMTE2()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)));
    weightMte2LoopIdx_++;
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WaitSToMTE2(uint64_t rlLoopIdx)
{
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID_S_TO_MTE2 + (rlLoopIdx & 1));
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::SetSToMTE2(uint64_t rlLoopIdx)
{
    SetFlag<HardEvent::S_MTE2>(EVENT_ID_S_TO_MTE2 + (rlLoopIdx & 1));
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WaitFrVToMTE2(uint64_t rlLoopIdx)
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID_FR_V_TO_MTE2 + (rlLoopIdx & 1));
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::SetFrToMTE2(uint64_t rlLoopIdx)
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID_FR_V_TO_MTE2 + (rlLoopIdx & 1));
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::SetMte2ToS(uint64_t rlLoopIdx)
{
    SetFlag<HardEvent::MTE2_S>(EVENT_ID_MTE2_TO_S + (rlLoopIdx & 1));
    SetFlag<HardEvent::MTE2_V>(EVENT_ID_RL_MTE2_TO_V + (rlLoopIdx & 1));
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WaitMte2ToS(uint64_t rlLoopIdx)
{
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID_MTE2_TO_S + (rlLoopIdx & 1));
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_RL_MTE2_TO_V + (rlLoopIdx & 1));
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyGmToUb(
    uint64_t ubMte2KSize, uint64_t kGmOffset, uint64_t kL1Offset, uint64_t biasGmOffset, uint64_t biasRealSize,
    const BasicBlockOffsetParam &offsetParam)
{
    // Inline CopyBiasGmToUb logic
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            DataCopyPad2D(bias_[(weightMte2LoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasSingleBufferSize],
                          biasGlobal_[biasGmOffset], 1, biasRealSize, biasRealSize, biasRealSize);

            SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        }
    }

    // ubMte2NSize和ubMte2KSize为实际MTE2搬运到UB的有效数据，
    // 其按照ubMte2InnerSize进行跳写，垃圾数据无需操作，搬出的时搬运有效数据即可。
    if (offsetParam.nL1Size == 0 || ubMte2KSize == 0) {
        return;
    }

    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize]
            .template ReinterpretCast<wType>(),
        wGlobal_[(kGmOffset + kL1Offset) * offsetParam.nAlign + offsetParam.nOffset * static_cast<uint64_t>(C0_SIZE)],
        CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
        CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE, offsetParam.nAlign * C0_SIZE);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WeightAntiQuantComputeNzNk(
    uint64_t kRealSize, uint64_t biasRealSize, uint64_t kGmOffset, const LocalTensor<xType> &weightHighBitL1,
    const LocalTensor<biasType> &biasL1, const BasicBlockOffsetParam &offsetParam)
{
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));

    AntiQuantProcessNzMxA8W4(biasRealSize, kRealSize, kGmOffset, offsetParam);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    if (likely(kRealSize > 0)) {
        CopyWeightHighBitForAligned(offsetParam.nL1Size, kRealSize, weightHighBitL1);
    }
    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            DataCopy(
                biasL1,
                biasRescale_[(ubComputeLoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasRescaleSingleBufferSize],
                biasRealSize);
        }
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));
    ubComputeLoopIdx_++;
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantProcessNzMxA8W4(
    uint64_t biasRealSize, uint64_t ubMte2KSize, uint64_t kGmOffset, const BasicBlockOffsetParam &offsetParam)
{
    MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams;
    uint64_t ubMte2BufferIdx = weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1);
    mxA8W4NzParams.nRealSizeAlign = CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ wType *)weightLowBit_[ubMte2BufferIdx * UB_BUFFER_INFO.weightLowBitSingleBufferSize].GetPhyAddr();

    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ xType *)
            weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VECTOR_REG_WIDTH]
                .GetPhyAddr();
    mxA8W4NzParams.loopKNum = CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE));
    mxA8W4NzParams.innerLoopNum =
        CeilDivide(CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                   static_cast<uint64_t>(VECTOR_REG_WIDTH));
    // 跳写UB避免bank冲突
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.weightHighBitBufferNum;
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;

    if constexpr (hasBias) {
        if (kGmOffset == 0 && biasRealSize != 0) {
            mxA8W4NzParams.biasInUbAddr = (__ubuf__ biasType *)bias_.GetPhyAddr(
                (weightMte2LoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasSingleBufferSize);
            mxA8W4NzParams.biasOutUbAddr = (__ubuf__ biasType *)biasRescale_.GetPhyAddr(
                (ubComputeLoopIdx_ & QUADRUPLE_BUFFER_MASK) * UB_BUFFER_INFO.biasRescaleSingleBufferSize);
            mxA8W4NzParams.biasLoopNum = CeilDivide(biasRealSize, VEC_MAX_ELEM_B16);
            FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, true>(mxA8W4NzParams);
        } else {
            FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, false>(mxA8W4NzParams);
        }
    } else {
        FrAntiQuantMxA8W4NzNkVf<xType, wType, biasType, false>(mxA8W4NzParams);
    }
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyWeightHighBitForAligned(
    uint64_t antiQuantRealN, uint64_t antiQuantRealK, const LocalTensor<xType> &weightHighBitL1)
{
    DataCopyParams params;
    params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(C0_SIZE)) *
                        CeilAlign(antiQuantRealN, static_cast<uint64_t>(BLOCK_CUBE)) / VEC_REG_ELEM;

    params.blockLen = VEC_REG_ELEM / ONE_BLK_SIZE;
    params.srcStride = (UB_BUFFER_INFO.weightHighBitBufferNum - 1) * params.blockLen;
    params.dstStride = 0; // dst地址连续
    DataCopy(weightHighBitL1,
             weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VEC_REG_ELEM], params);
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::MulLogits(
    const LocalTensor<float> &ubOutputF32Buffer, uint64_t mRealSize, const BasicBlockOffsetParam &offsetParam,
    uint64_t rlLoopIdx)
{
    if (mRealSize == 0) {
        return;
    }
    FrMulLogitsVf(static_cast<uint16_t>(mRealSize), CeilDivide(offsetParam.nL1Size, VEC_MAX_ELEM_B32),
                  (__ubuf__ float *)logits_.GetPhyAddr((rlLoopIdx & 1) * UB_BUFFER_INFO.logitsSingleBufferSize),
                  (__ubuf__ float *)ubOutputF32Buffer.GetPhyAddr());
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::RoutingYToGm(
    uint64_t mRealSize, const LocalTensor<float> &ubOutputF32Buffer, const BasicBlockOffsetParam &offsetParam,
    uint64_t rlLoopIdx)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    AscendC::SetAtomicAdd<float>();
    for (uint32_t mIdx = 0; mIdx < mRealSize; mIdx++) {
        int64_t mGmOffset = rowIndex_[(rlLoopIdx & 1) * UB_BUFFER_INFO.rowIndexSingleBufferSize].GetValue(mIdx);
        if (mGmOffset < 0) {
            continue;
        }
        DataCopyPad2D(yFp32Global_[mGmOffset * offsetParam.nSize + offsetParam.nOffset], ubOutputF32Buffer[mIdx * 256],
                      1, offsetParam.nL1Size, offsetParam.nL1Size, offsetParam.nL1Size);
    }

    AscendC::SetAtomicNone();
    SetFlag<HardEvent::MTE3_V>(EVENT_ID_BIAS_FR_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_BIAS_FR_MTE3_TO_V);
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::End()
{
    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}

GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyRowIndexLogitsGmToUb(uint64_t mGmOffset,
                                                                                                   uint64_t mRealSize,
                                                                                                   uint64_t rlLoopIdx)
{
    if (mRealSize == 0) {
        return;
    }
    DataCopyPad2D(rowIndex_[(rlLoopIdx & 1) * UB_BUFFER_INFO.rowIndexSingleBufferSize], rowIndexGlobal_[mGmOffset], 1,
                  mRealSize, mRealSize, mRealSize);
    DataCopyPad2D(logits_[(rlLoopIdx & 1) * UB_BUFFER_INFO.logitsSingleBufferSize], logitsGlobal_[mGmOffset], 1,
                  mRealSize, mRealSize, mRealSize);
}
} // namespace WeightQuantBatchMatmulV2::Arch35

#endif // GROUPED_MATMUL_WEIGHT_QUANT_VEC_COMPUTE_H
