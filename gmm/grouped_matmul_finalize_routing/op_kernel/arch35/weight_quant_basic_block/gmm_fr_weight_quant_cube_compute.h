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
 * \file weight_quant_cube_compute.h
 * \brief
 */
#ifndef GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_H
#define GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_H

#include "../common/basic_block_config.h"
#include "gmm_fr_weight_quant_cube_compute_tools.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_tool.h"

using AscendC::Dn2NzParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::PipeBarrier;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPosition;
using AscendC::WaitFlag;

namespace WeightQuantBatchMatmulV2::Arch35 {

#define GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM \
    template <typename xType, typename biasType, typename antiQuantScaleType, typename perTokenScaleType, \
              typename yType, bool hasBias, const WqmmConfig &wqmmConfig>

#define GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS \
    GMMFRWeightQuantCubeCompute<xType, biasType, antiQuantScaleType, perTokenScaleType, yType, hasBias, wqmmConfig>

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
class GMMFRWeightQuantCubeCompute {
public:
    __aicore__ inline GMMFRWeightQuantCubeCompute(){};
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ yType *y, __gm__ biasType *bias,
                                            __gm__ antiQuantScaleType *antiquantScale, __gm__ uint64_t *quantScale,
                                            __gm__ perTokenScaleType *perTokenScale);
    __aicore__ inline void MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize, uint64_t mxBiasL1DbOffset,
                                      const LocalTensor<biasType> &biasL1);
    __aicore__ inline void LaunchMatmul(const LocalTensor<xType> &weightL1, int64_t kbOffset, uint64_t kbL1RealSize,
                                        uint64_t cvLoopIdx, const BasicBlockOffsetParam &param);
    __aicore__ inline void WaitMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SetMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WaitWeightMTE1ToMTE2();
    __aicore__ inline void SetWeightMTE1ToMTE2();
    __aicore__ inline void WaitScaleMTE1ToMTE2(uint64_t kbGmOffset);
    __aicore__ inline void SetScaleMTE1ToMTE2(uint64_t kbGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param, int64_t kaGmOffset,
                                              uint64_t cvLoopIdx);
    __aicore__ inline void CopyMxScaleGmToL1(const BasicBlockOffsetParam &param, uint64_t kbL1Offset);
    __aicore__ inline void GetTensorC(const BasicBlockOffsetParam &param);
    __aicore__ inline void GetTensorC(LocalTensor<yType> &yUb, const BasicBlockOffsetParam &param);
    __aicore__ inline void EndSync();
    __aicore__ inline void DoMmadCompute(uint64_t loopId, uint64_t cvLoopIdx,
                                         const L0CopyAndCalcParams &l0CopyAndCalcParams);
    __aicore__ inline void PadL1IfNotAlign(const LocalTensor<xType> &l1Buf, uint64_t dbOffset, uint64_t dbIdx,
                                           uint64_t realSize, uint64_t dimSize);

private:
    __aicore__ inline void FillL1WithZero(const LocalTensor<uint32_t> &l1Buf, uint64_t blockCount);
    __aicore__ inline void ConfigScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmSize, uint64_t scaleKL1Stride,
                                                  uint64_t scaleKL1RealSize, Dn2NzParams &dn2NzParams);
    using L0DataType = typename AscendC::GetL0DataType<xType, true>::Type;

    static constexpr uint64_t L0_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_OFFSET_B8 = 32 * 1024;
    static constexpr uint64_t BIAS_TABLE_OFFSET_B32 = 512;
    static constexpr uint64_t L0C_BUF_SIZE = 256 * 256;

    int8_t aL1DbNum_;
    static constexpr uint32_t KB_UNIT = GetKBUnit<xType>();
    static constexpr uint64_t MX_SCALE_L1_SIZE = 32 * GetKBUnit<xType>() * sizeof(xType); // scaleA/B单块分配空间
    static constexpr uint32_t EVENT_ID_WEIGHT_MTE1_TO_MTE2 = 1;
    static constexpr uint32_t EVENT_ID_MTE1_TO_MTE2 = 3;
    static constexpr uint32_t EVENT_ID_SCALE_MTE1_TO_MTE2 = 5;
    static constexpr uint32_t EVENT_ID_M_TO_MTE1 = 3;
    static constexpr uint32_t EVENT_ID_MTE1_TO_M = 3;
    static constexpr uint32_t EVENT_ID_MTE2_TO_MTE1 = 3;
    uint64_t aL1Count_ = 0;
    uint64_t aL1MaxHalfCount_ = 0;
    uint64_t mxScaleBufIdx_ = 0;
    uint64_t aL1BufIdx_ = 0;
    uint64_t weightL1BufIdx_ = 0;

    AscendC::TEventID cubeEventIdsMxScaleMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];
    GlobalTensor<xType> xGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<fp8_e8m0_t> mxScaleAGlobal_;
    GlobalTensor<fp8_e8m0_t> mxScaleBGlobal_;
    GlobalTensor<yType> yGlobal_;

    uint64_t l0LoopIdx_ = 0;

    LocalTensor<L0DataType> l0a_{TPosition::A2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};
    LocalTensor<L0DataType> l0b_{TPosition::B2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};
    LocalTensor<float> l0c_{TPosition::CO1, 0, 256 * GetKBUnit<int8_t>() / sizeof(float)};
    LocalTensor<float> biasTable_{TPosition::C2, 0, 4 * GetKBUnit<int8_t>() / sizeof(float)};

    LocalTensor<xType> aL1_;
    uint64_t aL1DbOffset_;

    LocalTensor<biasType> biasL1_;
    uint64_t biasL1DbOffset_;

    LocalTensor<fp8_e8m0_t> mxScaleAL1_;
    uint64_t mxScaleAL1DbOffset_;

    LocalTensor<fp8_e8m0_t> mxScaleBL1_;
    uint64_t mxScaleBL1DbOffset_;

    uint64_t l0cOffset_ = 0;
};

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize,
                                                                          uint64_t mxBiasL1DbOffset,
                                                                          const LocalTensor<biasType> &biasL1)
{
    //  MxA8W4场景空间分配: 其中 weight\Bias为全局分配，此处不感知
    //  (1) 有bias场景
    //  L1 (0~512KB): WeightL1_P0(64KB) |    Bias_P0(4KB)   | ScaleAL1_P0(32KB) | ScaleBL1_P0(32KB) | AL1_P0(124KB) |
    //              | AL1_P1(124KB)     | ScaleBL1_P1(32KB) | ScaleAL1_P1(32KB) | Bias_P1(4KB)      | WeightL1_P1(64KB)
    //  (2) 无bias场景
    //  L1 (0~512KB): WeightL1_P0(64KB) | ScaleAL1_P0(32KB) | ScaleBL1_P0(32KB) | AL1_P0(128KB) |
    //              | AL1_P1(128KB)     | ScaleBL1_P1(32KB) | ScaleAL1_P1(32KB) | WeightL1_P1(64KB)
    biasL1_ = biasL1; // 预计大小4Kb
    biasL1DbOffset_ = mxBiasL1DbOffset;
    aL1Count_ = 0;        // GMM 动态场景暂时关闭A分段全载策略
    aL1MaxHalfCount_ = 0; // GMM 动态场景暂时关闭A分段全载策略
    aL1DbNum_ = DOUBLE_BUFFER_NUM;

    mxScaleAL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(fp8_e8m0_t));
    mxScaleAL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    mxScaleBL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(fp8_e8m0_t));
    mxScaleBL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    aL1_ = LocalTensor<xType>(TPosition::TSCM, l1StartSize, l1RemainSize);
    aL1DbOffset_ = l1RemainSize >> 1; // 最后剩余空间全部给AL1开DB

    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_WEIGHT_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
    }
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::UpdateGlobalAddr(
    __gm__ xType *x, __gm__ yType *y, __gm__ biasType *bias, __gm__ antiQuantScaleType *antiquantScale,
    __gm__ uint64_t *quantScale, __gm__ perTokenScaleType *perTokenScale)
{
    xGlobal_.SetGlobalBuffer(x);
    yGlobal_.SetGlobalBuffer(y);
    if constexpr (hasBias) {
        biasGlobal_.SetGlobalBuffer(bias);
    }

    mxScaleAGlobal_.SetGlobalBuffer(perTokenScale);
    mxScaleBGlobal_.SetGlobalBuffer(antiquantScale);
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::WaitScaleMTE1ToMTE2(uint64_t kbGmOffset)
{
    if (kbGmOffset % MX_SCALE_K_L1_SIZE == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + (mxScaleBufIdx_ & 1));
    }
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::SetScaleMTE1ToMTE2(
    uint64_t kbGmOffset, const BasicBlockOffsetParam &offsetParam)
{
    if ((kbGmOffset + offsetParam.kbL1Size) % MX_SCALE_K_L1_SIZE == 0 ||
        kbGmOffset + offsetParam.kbL1Size >= offsetParam.kSize) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + (mxScaleBufIdx_ & 1));
        mxScaleBufIdx_++;
    }
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::WaitMTE1ToMTE2(uint64_t kaGmOffset,
                                                                              const BasicBlockOffsetParam &offsetParam)
{
    if (aL1DbNum_ > SINGLE_BUFFER_NUM && kaGmOffset % offsetParam.kaL1Size == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + (aL1BufIdx_ & 1));
    }
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::WaitWeightMTE1ToMTE2()
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_WEIGHT_MTE1_TO_MTE2 + (weightL1BufIdx_ & 1));
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::SetWeightMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_WEIGHT_MTE1_TO_MTE2 + (weightL1BufIdx_ & 1));
    weightL1BufIdx_++;
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::SetMTE1ToMTE2(uint64_t kaGmOffset,
                                                                             const BasicBlockOffsetParam &offsetParam)
{
    if ((kaGmOffset + offsetParam.kbL1Size) % offsetParam.kaL1Size == 0 ||
        kaGmOffset + offsetParam.kbL1Size >= offsetParam.kSize) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + (aL1BufIdx_ & 1));
        aL1BufIdx_++;
    }
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::CopyMxScaleGmToL1(const BasicBlockOffsetParam &param,
                                                                                 uint64_t kbL1Offset)
{
    if (kbL1Offset % MX_SCALE_K_L1_SIZE != 0) {
        return;
    }
    uint64_t kSizeAligned = CeilAlign(param.kSize, K_ALIGNMENT64);
    uint64_t scaleKGmSize = kSizeAligned / MX_GROUPSIZE;
    // 当前scaleFactor为1，暂不考虑scaleFactor相关计算
    uint64_t scaleKL1StandardLen = MX_SCALE_K_L1_SIZE / MX_GROUPSIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > kSizeAligned ?
                                    (kSizeAligned - kbL1Offset) / MX_GROUPSIZE :
                                    scaleKL1StandardLen;

    // copy mxScaleA
    Dn2NzParams scaleAdn2NzParams;
    ConfigScaleDn2NzParams(param.mL1Size, scaleKGmSize, scaleKL1StandardLen, scaleKL1RealSize, scaleAdn2NzParams);

    int64_t scaleAGmOffset = param.mOffset * scaleKGmSize + kbL1Offset / MX_GROUPSIZE;
    GlobalTensor<half> f16ScaleAGlobal;
    f16ScaleAGlobal.SetGlobalBuffer((__gm__ half *)mxScaleAGlobal_[scaleAGmOffset].GetPhyAddr(),
                                    (param.mL1Size * scaleKL1RealSize) >> 1);
    auto f16ScaleALocal = mxScaleAL1_[(mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_].template ReinterpretCast<half>();

    DataCopy(f16ScaleALocal, f16ScaleAGlobal, scaleAdn2NzParams);

    // copy mxScaleB
    Dn2NzParams scaleBdn2NzParams;
    ConfigScaleDn2NzParams(param.nL1Size, scaleKGmSize, scaleKL1StandardLen, scaleKL1RealSize, scaleBdn2NzParams);

    int64_t scaleBGmOffset = param.nOffset * scaleKGmSize + kbL1Offset / MX_GROUPSIZE;
    GlobalTensor<half> f16ScaleBGlobal;
    f16ScaleBGlobal.SetGlobalBuffer((__gm__ half *)mxScaleBGlobal_[scaleBGmOffset].GetPhyAddr(),
                                    (param.nL1Size * scaleKL1RealSize) >> 1);
    auto f16ScaleBLocal = mxScaleBL1_[(mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_].template ReinterpretCast<half>();

    DataCopy(f16ScaleBLocal, f16ScaleBGlobal, scaleBdn2NzParams);
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param,
                                                                                  int64_t kaGmOffset,
                                                                                  uint64_t cvLoopIdx)
{
    if (kaGmOffset % param.kaL1Size != 0) {
        return;
    }
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;

    nd2nzParams.nValue = param.mL1Size;
    nd2nzParams.dValue = kaL1RealSize;
    nd2nzParams.srcDValue = param.kSize;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(aL1_[(aL1BufIdx_ & 1) * aL1DbOffset_], xGlobal_[kaGmOffset + param.mOffset * param.kSize], nd2nzParams);

    PadL1IfNotAlign(aL1_, aL1DbOffset_, aL1BufIdx_, kaL1RealSize, param.mL1Size);
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::LaunchMatmul(const LocalTensor<xType> &weightL1,
                                                                            int64_t kbOffset, uint64_t kbL1RealSize,
                                                                            uint64_t cvLoopIdx,
                                                                            const BasicBlockOffsetParam &param)
{
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    uint64_t aL1Offset = 0;

    aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_;

    L0CopyAndCalcParams l0CopyAndCalcParams;
    l0CopyAndCalcParams.mL0Size = param.mL1Size;
    l0CopyAndCalcParams.mL1Size = param.mL1Size;
    uint64_t kL0Size = (param.mL1Size <= 128 && param.nL1Size <= 128) ? 256 : 128;
    l0CopyAndCalcParams.kL1Size = param.kaL1Size;
    l0CopyAndCalcParams.nL0Size = param.nL1Size;
    l0CopyAndCalcParams.nL1Size = param.nL1Size;
    l0CopyAndCalcParams.scaleKL1Size = MX_SCALE_K_L1_SIZE;
    uint64_t mL1AlignSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t nL1AlignSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    if (kbOffset == 0) {
        uint64_t nextL0cBufSize = CeilAlign(l0CopyAndCalcParams.mL0Size, static_cast<int64_t>(BLOCK_CUBE)) *
                                  CeilAlign(l0CopyAndCalcParams.nL0Size, static_cast<int64_t>(BLOCK_CUBE));
        if (l0cOffset_ + nextL0cBufSize > L0C_BUF_SIZE) {
            l0cOffset_ = 0;
        }
    }
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += kL0Size) {
        bool isLastL1K = l1KOffset + kL0Size >= kbL1RealSize;
        l0CopyAndCalcParams.isFirstKLoop = (l1KOffset + kbOffset) == 0;
        l0CopyAndCalcParams.isLastKLoop = l1KOffset + kbOffset + kL0Size >= param.kSize;

        uint64_t realL0k = isLastL1K ? kbL1RealSize - l1KOffset : kL0Size;
        l0CopyAndCalcParams.kL0Size = realL0k;
        uint64_t loopId = l0LoopIdx_ % L0_BUF_NUM; // 等价于 l0LoopIdx_ % L0_BUF_NUM，减少scalar
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId);
        LoadAAndScaleL1ToL0(l0a_[loopId * L0_BUF_OFFSET_B8],
                            aL1_[aL1Offset + mL1AlignSize * ((l1KOffset + kbOffset) % param.kaL1Size)],
                            mxScaleAL1_[(mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_ +
                                        BLOCK_CUBE * ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE / MX_GROUPSIZE)],
                            l0CopyAndCalcParams);
        LoadBAndScaleL1ToL0(l0b_[loopId * L0_BUF_OFFSET_B8], weightL1[nL1AlignSize * l1KOffset],
                            mxScaleBL1_[(mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_ +
                                        BLOCK_CUBE * ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE / MX_GROUPSIZE)],
                            l0CopyAndCalcParams);

        DoMmadCompute(loopId, cvLoopIdx, l0CopyAndCalcParams);
    }
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::DoMmadCompute(
    uint64_t loopId, uint64_t cvLoopIdx, const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    if constexpr (hasBias) {
        if (l0CopyAndCalcParams.isFirstKLoop) {
            LoadBiasToBt(biasTable_[loopId * BIAS_TABLE_OFFSET_B32], biasL1_[(cvLoopIdx & 1) * biasL1DbOffset_],
                         l0CopyAndCalcParams);
        }
    }
    SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
    if constexpr (hasBias) {
        if (l0CopyAndCalcParams.isFirstKLoop) {
            MmadCompute(l0c_[l0cOffset_], l0a_[loopId * L0_BUF_OFFSET_B8], l0b_[loopId * L0_BUF_OFFSET_B8],
                        biasTable_[loopId * BIAS_TABLE_OFFSET_B32], l0CopyAndCalcParams);
        } else {
            MmadCompute(l0c_[l0cOffset_], l0a_[loopId * L0_BUF_OFFSET_B8], l0b_[loopId * L0_BUF_OFFSET_B8],
                        l0CopyAndCalcParams);
        }
    } else {
        MmadCompute(l0c_[l0cOffset_], l0a_[loopId * L0_BUF_OFFSET_B8], l0b_[loopId * L0_BUF_OFFSET_B8],
                    l0CopyAndCalcParams);
    }
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId);
    l0LoopIdx_++;
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::ConfigScaleDn2NzParams(uint64_t rowNum,
                                                                                      uint64_t scaleKGmSize,
                                                                                      uint64_t scaleKL1Stride,
                                                                                      uint64_t scaleKL1RealSize,
                                                                                      Dn2NzParams &dn2NzParams)
{
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = rowNum; // 矩阵的行数，即待搬运的mxScaleA的m或mxScaleB的n
    dn2NzParams.nValue =
        CeilDivide(scaleKL1RealSize, SCALE_COPY_GROUP_SIZE); // 矩阵的列数，使用B16搬B8需要除以2向上取整
    dn2NzParams.srcDnMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
    dn2NzParams.srcDValue = CeilDivide(scaleKGmSize, SCALE_COPY_GROUP_SIZE); // 源矩阵一行所含B16元素个数
    // 目标矩阵行方向两个相邻分形起始地址之间的间隔，单位32B
    dn2NzParams.dstNzC0Stride = CeilDivide(MX_SCALE_K_L1_SIZE, MX_GROUPSIZE * SCALE_COPY_GROUP_SIZE);
    // 目标矩阵列方向两个相邻分形起始地址之间的间隔，单位32B
    dn2NzParams.dstNzNStride = SCALE_COPY_DEFAULT_N_STRIDE;
    dn2NzParams.dstNzMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::EndSync()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_WEIGHT_MTE1_TO_MTE2 + i);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + i);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + i);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
    }
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::PadL1IfNotAlign(const LocalTensor<xType> &l1Buf,
                                                                               uint64_t dbOffset, uint64_t dbIdx,
                                                                               uint64_t realSize, uint64_t dimSize)
{
    if (realSize % K_ALIGNMENT64 != 0) {
        uint64_t alignDimSize = CeilAlign(dimSize, static_cast<uint64_t>(BLOCK_CUBE));
        LocalTensor<uint32_t> fillTensor =
            l1Buf[(dbIdx & 1) * dbOffset + alignDimSize * realSize].template ReinterpretCast<uint32_t>();
        FillL1WithZero(fillTensor, alignDimSize);
    }
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::FillL1WithZero(const LocalTensor<uint32_t> &l1Buf,
                                                                              uint64_t blockCount)
{
    AscendC::InitConstValueParams<uint32_t> initParams;
    initParams.repeatTimes = 1;
    initParams.dstGap = 0;
    initParams.initValue = 0;
    initParams.blockNum = blockCount;
    AscendC::Fill(l1Buf, initParams);
}

GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_CUBE_COMPUTE_CLASS::GetTensorC(LocalTensor<yType> &yUb,
                                                                          const BasicBlockOffsetParam &param)
{
    FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = param.mL1Size; // M方向当前在L0上实际计算的大小
    fixL0CToDstParams.nL0Size = param.nL1Size; // N方向当前在L0上实际计算的大小
    fixL0CToDstParams.outNSize = 256;          // 输出N方向的总大小，用于dstStride
    fixL0CToDstParams.splitNSize = 0;          // fixp输出多个矩阵时，N方向切分的大小
    fixL0CToDstParams.dstNdStride = 0; // fixp输出多个矩阵时，目的相邻DN矩阵起始地址间的偏移
    FixL0CToDst(yUb, l0c_[l0cOffset_], fixL0CToDstParams);
    l0cOffset_ += Ops::Base::CeilAlign(fixL0CToDstParams.mL0Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                  CeilAlign(fixL0CToDstParams.nL0Size, static_cast<int64_t>(BLOCK_CUBE));
}
} // namespace WeightQuantBatchMatmulV2::Arch35
#endif
