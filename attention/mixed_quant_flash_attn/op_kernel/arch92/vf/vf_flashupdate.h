/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file vf_flashupdate.h
 * \brief
 */
#ifndef VF_FLASH_UPDATE_92_H
#define VF_FLASH_UPDATE_92_H

#include "kernel_tensor.h"

namespace AscendC {
/* **************************************************************************************************
 * FlashUpdate
 * [s1, k] = [64, 128], fp32
 * ************************************************************************************************* */
static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->Q_T
static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp16 -> bf16
static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr static MicroAPI::CastTrait castTrait16232_0 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr static MicroAPI::CastTrait castTrait16232_1 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

/*
 * @ingroup FlashUpdateTail_VF
 * @brief compute, dstTensor = (preTensor + curTensor) * expMaxTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 64 aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
                                          const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                          const uint16_t m, const uint16_t d, float dequantScale2)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vDstRegAdd;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrcCur;

        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegTailD = MicroAPI::UpdateMask<T>(pltTailD);

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegAdd,
                                                                                 maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur,
                                                                                  curUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                                 vDstRegAdd, maskRegTailD);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                         preUb + i * dSize + j * floatRepSize);
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                            vregSrcCur, curUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                               maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur,
                                                                             curUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                               maskRegAll);
                    }
                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * dSize + j * floatRepSize, vDstRegAdd, maskRegAll);
                }
                // 尾块处理
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                     preUb + i * dSize + dLoops * floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                        vregSrcCur, curUb + i * dSize + dLoops * floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur,
                                                                         curUb + i * dSize + dLoops * floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * dSize + dLoops * floatRepSize, vDstRegAdd, maskRegTailD);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateLastTail_VF
 * @brief compute, dstTensor = (preTensor  + curTensor ) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
                                              const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                              const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d,
                                              float dequantScale2)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vSrcRegAdd;
        MicroAPI::RegTensor<T> vDstRegDiv;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrcCur;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegTailD = MicroAPI::UpdateMask<T>(pltTailD);

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv,
                                                                                 maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur,
                                                                                  curUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegTailD);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                                 vDstRegDiv, maskRegTailD);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                         preUb + i * dSize + j * floatRepSize);
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                            vregSrcCur, curUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                               maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur,
                                                                             curUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                               maskRegAll);
                    }
                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * dSize + j * floatRepSize, vDstRegDiv, maskRegAll);
                }
                // 尾块处理
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                     preUb + i * dSize + dLoops * floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                        vregSrcCur, curUb + i * dSize + dLoops * floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur,
                                                                         curUb + i * dSize + dLoops * floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegTailD);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegTailD);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * dSize + dLoops * floatRepSize, vDstRegDiv, maskRegTailD);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateDivTail_VF
 * @brief compute, dstTensor = preTensor / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& preTensor,
                                             const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d,
                                             float dequantScale2)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* preUb = (__ubuf__ MMOUTPUT_T*)preTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vDstRegDiv;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrc;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegTailD = MicroAPI::UpdateMask<T>(pltTailD);

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv,
                                                                                 maskRegAll);

                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc,
                                                                                  preUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                                 vDstRegDiv, maskRegTailD);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                            vregSrc, preUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                               maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                             preUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                               maskRegAll);
                    }
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * dSize + j * floatRepSize, vDstRegDiv, maskRegAll);
                }
                // 尾块处理
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                        vregSrc, preUb + i * dSize + dLoops * floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                         preUb + i * dSize + dLoops * floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegTailD);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    dstUb + i * dSize + dLoops * floatRepSize, vDstRegDiv, maskRegTailD);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateNoTail_VF
 * @brief compute, dstTensor = (preTensor + curTensor) * expMaxTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 64 aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateNoTail_VF(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
                                            const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                            const uint16_t m, const uint16_t d, float dequantScale2)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vDstRegAdd;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrcCur;

        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegAdd,
                                                                                 maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur,
                                                                                  curUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                                 vDstRegAdd, maskRegAll);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                         preUb + i * dSize + j * floatRepSize);
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                            vregSrcCur, curUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                               maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur,
                                                                             curUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                               maskRegAll);
                    }
                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * dSize + j * floatRepSize, vDstRegAdd, maskRegAll);
                }
            }
        }
    }
}

/*
 * @ingroup FlashUpdateNoTailAntiQuant
 * @brief compute, dstTensor = (preTensor + curTensor) * expMaxTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateNoTailAntiQuant(const LocalTensor<T>& dstTensor,
                                                  const LocalTensor<MMOUTPUT_T>& curTensor,
                                                  const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                                  const LocalTensor<INPUT_T>& antiQuantResUb, const uint16_t m,
                                                  const uint16_t d)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ INPUT_T* antiQuantUb = (__ubuf__ INPUT_T*)antiQuantResUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vDstRegAdd;
        MicroAPI::RegTensor<T> vSrcRegPre1;
        MicroAPI::RegTensor<T> vSrcRegCur1;
        MicroAPI::RegTensor<T> vSrcRegMul1;
        MicroAPI::RegTensor<T> vDstRegAdd1;
        MicroAPI::RegTensor<T> vSrcRegPre2;
        MicroAPI::RegTensor<T> vSrcRegCur2;
        MicroAPI::RegTensor<T> vSrcRegMul2;
        MicroAPI::RegTensor<T> vDstRegAdd2;
        MicroAPI::RegTensor<T> vSrcRegPre3;
        MicroAPI::RegTensor<T> vSrcRegCur3;
        MicroAPI::RegTensor<T> vSrcRegMul3;
        MicroAPI::RegTensor<T> vDstRegAdd3;
        MicroAPI::RegTensor<T> vSrcRegPre4;
        MicroAPI::RegTensor<T> vSrcRegCur4;
        MicroAPI::RegTensor<T> vSrcRegMul4;
        MicroAPI::RegTensor<T> vDstRegAdd4;
        MicroAPI::RegTensor<T> vSrcRegPre5;
        MicroAPI::RegTensor<T> vSrcRegCur5;
        MicroAPI::RegTensor<T> vSrcRegMul5;
        MicroAPI::RegTensor<T> vDstRegAdd5;
        MicroAPI::RegTensor<T> vSrcRegPre6;
        MicroAPI::RegTensor<T> vSrcRegCur6;
        MicroAPI::RegTensor<T> vSrcRegMul6;
        MicroAPI::RegTensor<T> vDstRegAdd6;
        MicroAPI::RegTensor<T> vSrcRegPre7;
        MicroAPI::RegTensor<T> vSrcRegCur7;
        MicroAPI::RegTensor<T> vSrcRegMul7;
        MicroAPI::RegTensor<T> vDstRegAdd7;

        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant;
        MicroAPI::RegTensor<T> vSrcRegQuantEven;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant1;
        MicroAPI::RegTensor<T> vSrcRegQuantEven1;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd1;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant2;
        MicroAPI::RegTensor<T> vSrcRegQuantEven2;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd2;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant3;
        MicroAPI::RegTensor<T> vSrcRegQuantEven3;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd3;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant4;
        MicroAPI::RegTensor<T> vSrcRegQuantEven4;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd4;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant5;
        MicroAPI::RegTensor<T> vSrcRegQuantEven5;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd5;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant6;
        MicroAPI::RegTensor<T> vSrcRegQuantEven6;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd6;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant7;
        MicroAPI::RegTensor<T> vSrcRegQuantEven7;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd7;
        MicroAPI::MaskReg pRegAllB16 = MicroAPI::CreateMask<INPUT_T, MicroAPI::MaskPattern::ALL>();

        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        // 手动unroll，性能达到最优
        MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant, antiQuantUb);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven, vSrcRegQuant, pRegAllB16);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd, vSrcRegQuant, pRegAllB16);
        MicroAPI::Interleave<T>(vSrcRegQuantEven, vSrcRegQuantOdd, vSrcRegQuantEven, vSrcRegQuantOdd);
        MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant1, antiQuantUb + floatRepSize);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven1, vSrcRegQuant1, pRegAllB16);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd1, vSrcRegQuant1, pRegAllB16);
        MicroAPI::Interleave<T>(vSrcRegQuantEven1, vSrcRegQuantOdd1, vSrcRegQuantEven1, vSrcRegQuantOdd1);
        if constexpr (dSize > 128) {
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant2, antiQuantUb + 2 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven2, vSrcRegQuant2, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd2, vSrcRegQuant2, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven2, vSrcRegQuantOdd2, vSrcRegQuantEven2, vSrcRegQuantOdd2);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant3, antiQuantUb + 3 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven3, vSrcRegQuant3, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd3, vSrcRegQuant3, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven3, vSrcRegQuantOdd3, vSrcRegQuantEven3, vSrcRegQuantOdd3);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant4, antiQuantUb + 4 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven4, vSrcRegQuant4, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd4, vSrcRegQuant4, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven4, vSrcRegQuantOdd4, vSrcRegQuantEven4, vSrcRegQuantOdd4);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant5, antiQuantUb + 5 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven5, vSrcRegQuant5, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd5, vSrcRegQuant5, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven5, vSrcRegQuantOdd5, vSrcRegQuantEven5, vSrcRegQuantOdd5);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant6, antiQuantUb + 6 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven6, vSrcRegQuant6, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd6, vSrcRegQuant6, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven6, vSrcRegQuantOdd6, vSrcRegQuantEven6, vSrcRegQuantOdd6);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant7, antiQuantUb + 7 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven7, vSrcRegQuant7, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd7, vSrcRegQuant7, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven7, vSrcRegQuantOdd7, vSrcRegQuantEven7, vSrcRegQuantOdd7);
        }

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);

            // 手动unroll，性能达到最优
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
            // MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, vSrcRegQuantEven, maskRegAll);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
            MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegAdd, maskRegAll);

            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre1, preUb + i * dSize + floatRepSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur1, curUb + i * dSize + floatRepSize);
            // MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur1, vSrcRegCur1, vSrcRegQuantEven1,
            // maskRegAll);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul1, vSrcRegMax, vSrcRegPre1, maskRegAll);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd1, vSrcRegMul1, vSrcRegCur1, maskRegAll);
            MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                             vDstRegAdd1, maskRegAll);

            if constexpr (dSize > 128) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre2, preUb + i * dSize + 2 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur2, curUb + i * dSize + 2 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur2, vSrcRegCur2, vSrcRegQuantEven2,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul2, vSrcRegMax, vSrcRegPre2, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd2, vSrcRegMul2, vSrcRegCur2, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 2 * floatRepSize,
                                                                                 vDstRegAdd2, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre3, preUb + i * dSize + 3 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur3, curUb + i * dSize + 3 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur3, vSrcRegCur3, vSrcRegQuantEven3,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul3, vSrcRegMax, vSrcRegPre3, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd3, vSrcRegMul3, vSrcRegCur3, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 3 * floatRepSize,
                                                                                 vDstRegAdd3, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre4, preUb + i * dSize + 4 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur4, curUb + i * dSize + 4 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur4, vSrcRegCur4, vSrcRegQuantEven4,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul4, vSrcRegMax, vSrcRegPre4, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd4, vSrcRegMul4, vSrcRegCur4, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 4 * floatRepSize,
                                                                                 vDstRegAdd4, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre5, preUb + i * dSize + 5 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur5, curUb + i * dSize + 5 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur5, vSrcRegCur5, vSrcRegQuantEven5,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul5, vSrcRegMax, vSrcRegPre5, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd5, vSrcRegMul5, vSrcRegCur5, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 5 * floatRepSize,
                                                                                 vDstRegAdd5, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre6, preUb + i * dSize + 6 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur6, curUb + i * dSize + 6 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur6, vSrcRegCur6, vSrcRegQuantEven6,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul6, vSrcRegMax, vSrcRegPre6, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd6, vSrcRegMul6, vSrcRegCur6, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 6 * floatRepSize,
                                                                                 vDstRegAdd6, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre7, preUb + i * dSize + 7 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur7, curUb + i * dSize + 7 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur7, vSrcRegCur7, vSrcRegQuantEven7,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul7, vSrcRegMax, vSrcRegPre7, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vDstRegAdd7, vSrcRegMul7, vSrcRegCur7, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 7 * floatRepSize,
                                                                                 vDstRegAdd7, maskRegAll);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateLastNoTail_VF
 * @brief compute, dstTensor = (preTensor  + curTensor ) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastNoTail_VF(const LocalTensor<T>& dstTensor,
                                                const LocalTensor<MMOUTPUT_T>& curTensor,
                                                const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                                const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d,
                                                float dequantScale2)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vSrcRegAdd;
        MicroAPI::RegTensor<T> vDstRegDiv;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrcCur;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur, curUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv,
                                                                                 maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrcCur,
                                                                                  curUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                                 vDstRegDiv, maskRegAll);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                         preUb + i * dSize + j * floatRepSize);
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                            vregSrcCur, curUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegCur, vregSrcCur, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                               maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur,
                                                                             curUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegCur, dequantScale2,
                                                                               maskRegAll);
                    }
                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
                    MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegAdd, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * dSize + j * floatRepSize, vDstRegDiv, maskRegAll);
                }
            }
        }
    }
}

/*
 * @ingroup FlashUpdateLastNoTailAntiQuant
 * @brief compute, dstTensor = (preTensor  + curTensor ) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateLastNoTailAntiQuant(
    const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<INPUT_T>& antiQuantResUb,
    const uint16_t m, const uint16_t d)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* curUb = (__ubuf__ MMOUTPUT_T*)curTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ INPUT_T* antiQuantUb = (__ubuf__ INPUT_T*)antiQuantResUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegCur;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vSrcRegAdd;
        MicroAPI::RegTensor<T> vDstRegDiv;
        MicroAPI::RegTensor<T> vSrcRegPre1;
        MicroAPI::RegTensor<T> vSrcRegCur1;
        MicroAPI::RegTensor<T> vSrcRegMul1;
        MicroAPI::RegTensor<T> vSrcRegAdd1;
        MicroAPI::RegTensor<T> vDstRegDiv1;
        MicroAPI::RegTensor<T> vSrcRegPre2;
        MicroAPI::RegTensor<T> vSrcRegCur2;
        MicroAPI::RegTensor<T> vSrcRegMul2;
        MicroAPI::RegTensor<T> vSrcRegAdd2;
        MicroAPI::RegTensor<T> vDstRegDiv2;
        MicroAPI::RegTensor<T> vSrcRegPre3;
        MicroAPI::RegTensor<T> vSrcRegCur3;
        MicroAPI::RegTensor<T> vSrcRegMul3;
        MicroAPI::RegTensor<T> vSrcRegAdd3;
        MicroAPI::RegTensor<T> vDstRegDiv3;
        MicroAPI::RegTensor<T> vSrcRegPre4;
        MicroAPI::RegTensor<T> vSrcRegCur4;
        MicroAPI::RegTensor<T> vSrcRegMul4;
        MicroAPI::RegTensor<T> vSrcRegAdd4;
        MicroAPI::RegTensor<T> vDstRegDiv4;
        MicroAPI::RegTensor<T> vSrcRegPre5;
        MicroAPI::RegTensor<T> vSrcRegCur5;
        MicroAPI::RegTensor<T> vSrcRegMul5;
        MicroAPI::RegTensor<T> vSrcRegAdd5;
        MicroAPI::RegTensor<T> vDstRegDiv5;
        MicroAPI::RegTensor<T> vSrcRegPre6;
        MicroAPI::RegTensor<T> vSrcRegCur6;
        MicroAPI::RegTensor<T> vSrcRegMul6;
        MicroAPI::RegTensor<T> vSrcRegAdd6;
        MicroAPI::RegTensor<T> vDstRegDiv6;
        MicroAPI::RegTensor<T> vSrcRegPre7;
        MicroAPI::RegTensor<T> vSrcRegCur7;
        MicroAPI::RegTensor<T> vSrcRegMul7;
        MicroAPI::RegTensor<T> vSrcRegAdd7;
        MicroAPI::RegTensor<T> vDstRegDiv7;

        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant;
        MicroAPI::RegTensor<T> vSrcRegQuantEven;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant1;
        MicroAPI::RegTensor<T> vSrcRegQuantEven1;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd1;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant2;
        MicroAPI::RegTensor<T> vSrcRegQuantEven2;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd2;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant3;
        MicroAPI::RegTensor<T> vSrcRegQuantEven3;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd3;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant4;
        MicroAPI::RegTensor<T> vSrcRegQuantEven4;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd4;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant5;
        MicroAPI::RegTensor<T> vSrcRegQuantEven5;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd5;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant6;
        MicroAPI::RegTensor<T> vSrcRegQuantEven6;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd6;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant7;
        MicroAPI::RegTensor<T> vSrcRegQuantEven7;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd7;
        MicroAPI::MaskReg pRegAllB16 = MicroAPI::CreateMask<INPUT_T, MicroAPI::MaskPattern::ALL>();

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant, antiQuantUb);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven, vSrcRegQuant, pRegAllB16);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd, vSrcRegQuant, pRegAllB16);
        MicroAPI::Interleave<T>(vSrcRegQuantEven, vSrcRegQuantOdd, vSrcRegQuantEven, vSrcRegQuantOdd);
        MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant1, antiQuantUb + floatRepSize);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven1, vSrcRegQuant1, pRegAllB16);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd1, vSrcRegQuant1, pRegAllB16);
        MicroAPI::Interleave<T>(vSrcRegQuantEven1, vSrcRegQuantOdd1, vSrcRegQuantEven1, vSrcRegQuantOdd1);
        if constexpr (dSize > 128) {
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant2, antiQuantUb + 2 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven2, vSrcRegQuant2, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd2, vSrcRegQuant2, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven2, vSrcRegQuantOdd2, vSrcRegQuantEven2, vSrcRegQuantOdd2);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant3, antiQuantUb + 3 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven3, vSrcRegQuant3, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd3, vSrcRegQuant3, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven3, vSrcRegQuantOdd3, vSrcRegQuantEven3, vSrcRegQuantOdd3);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant4, antiQuantUb + 4 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven4, vSrcRegQuant4, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd4, vSrcRegQuant4, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven4, vSrcRegQuantOdd4, vSrcRegQuantEven4, vSrcRegQuantOdd4);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant5, antiQuantUb + 5 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven5, vSrcRegQuant5, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd5, vSrcRegQuant5, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven5, vSrcRegQuantOdd5, vSrcRegQuantEven5, vSrcRegQuantOdd5);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant6, antiQuantUb + 6 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven6, vSrcRegQuant6, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd6, vSrcRegQuant6, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven6, vSrcRegQuantOdd6, vSrcRegQuantEven6, vSrcRegQuantOdd6);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant7, antiQuantUb + 7 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven7, vSrcRegQuant7, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd7, vSrcRegQuant7, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven7, vSrcRegQuantOdd7, vSrcRegQuantEven7, vSrcRegQuantOdd7);
        }

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegMax, expMaxUb + i * reduceSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);

            // 手动unroll 性能最优
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur, curUb + i * dSize);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul, vSrcRegMax, vSrcRegPre, maskRegAll);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd, vSrcRegMul, vSrcRegCur, maskRegAll);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur, vSrcRegAdd, vSrcRegQuantEven, maskRegAll);
            MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegCur, vSrcRegSum, maskRegAll);
            MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv, maskRegAll);

            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre1, preUb + i * dSize + floatRepSize);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur1, curUb + i * dSize + floatRepSize);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul1, vSrcRegMax, vSrcRegPre1, maskRegAll);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd1, vSrcRegMul1, vSrcRegCur1, maskRegAll);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur1, vSrcRegAdd1, vSrcRegQuantEven1, maskRegAll);
            MicroAPI::Div<T, &mode>(vDstRegDiv1, vSrcRegCur1, vSrcRegSum, maskRegAll);
            MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                             vDstRegDiv1, maskRegAll);

            if constexpr (dSize > 128) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre2, preUb + i * dSize + 2 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur2, curUb + i * dSize + 2 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur2, vSrcRegCur2, vSrcRegQuantEven2,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul2, vSrcRegMax, vSrcRegPre2, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd2, vSrcRegMul2, vSrcRegCur2, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv2, vSrcRegAdd2, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 2 * floatRepSize,
                                                                                 vDstRegDiv2, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre3, preUb + i * dSize + 3 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur3, curUb + i * dSize + 3 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur3, vSrcRegCur3, vSrcRegQuantEven3,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul3, vSrcRegMax, vSrcRegPre3, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd3, vSrcRegMul3, vSrcRegCur3, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv3, vSrcRegAdd3, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 3 * floatRepSize,
                                                                                 vDstRegDiv3, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre4, preUb + i * dSize + 4 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur4, curUb + i * dSize + 4 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur4, vSrcRegCur4, vSrcRegQuantEven4,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul4, vSrcRegMax, vSrcRegPre4, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd4, vSrcRegMul4, vSrcRegCur4, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv4, vSrcRegAdd4, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 4 * floatRepSize,
                                                                                 vDstRegDiv4, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre5, preUb + i * dSize + 5 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur5, curUb + i * dSize + 5 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur5, vSrcRegCur5, vSrcRegQuantEven5,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul5, vSrcRegMax, vSrcRegPre5, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd5, vSrcRegMul5, vSrcRegCur5, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv5, vSrcRegAdd5, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 5 * floatRepSize,
                                                                                 vDstRegDiv5, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre6, preUb + i * dSize + 6 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur6, curUb + i * dSize + 6 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur6, vSrcRegCur6, vSrcRegQuantEven6,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul6, vSrcRegMax, vSrcRegPre6, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd6, vSrcRegMul6, vSrcRegCur6, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv6, vSrcRegAdd6, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 6 * floatRepSize,
                                                                                 vDstRegDiv6, maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre7, preUb + i * dSize + 7 * floatRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegCur7, curUb + i * dSize + 7 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegCur7, vSrcRegCur7, vSrcRegQuantEven7,
                                                                   maskRegAll);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegMul7, vSrcRegMax, vSrcRegPre7, maskRegAll);
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegAdd7, vSrcRegMul7, vSrcRegCur7, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv7, vSrcRegAdd7, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + 7 * floatRepSize,
                                                                                 vDstRegDiv7, maskRegAll);
            }
        }
    }
}

/*
 * @ingroup FlashUpdateDivNoTail_VF
 * @brief compute, dstTensor = preTensor / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input colums, should be 32 bytes aligned
 */
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivNoTail_VF(const LocalTensor<T>& dstTensor,
                                               const LocalTensor<MMOUTPUT_T>& preTensor,
                                               const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d,
                                               float dequantScale2)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* preUb = (__ubuf__ MMOUTPUT_T*)preTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vDstRegDiv;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrc;

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv,
                                                                                 maskRegAll);

                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc,
                                                                                  preUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                }
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                }
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                                 vDstRegDiv, maskRegAll);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                            vregSrc, preUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                               maskRegAll);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                             preUb + i * dSize + j * floatRepSize);
                    }
                    if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value || IsSameType<INPUT_T, hifloat8_t>::value) {
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                               maskRegAll);
                    }
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * dSize + j * floatRepSize, vDstRegDiv, maskRegAll);
                }
            }
        }
    }
}

template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
__aicore__ inline void FlashUpdateDivNoTailAntiQuant(const LocalTensor<T>& dstTensor,
                                                     const LocalTensor<MMOUTPUT_T>& preTensor,
                                                     const LocalTensor<T>& expSumTensor,
                                                     const LocalTensor<INPUT_T>& antiQuantResUb, const uint16_t m,
                                                     const uint16_t d, float dequantScale2)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ MMOUTPUT_T* preUb = (__ubuf__ MMOUTPUT_T*)preTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ INPUT_T* antiQuantUb = (__ubuf__ INPUT_T*)antiQuantResUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t reduceSize = 1;
    const uint16_t dLoops = d / floatRepSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegSum;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vDstRegDiv;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant;
        MicroAPI::RegTensor<T> vSrcRegQuantEven;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd;

        MicroAPI::RegTensor<MMOUTPUT_T> vregSrc;
        MicroAPI::MaskReg pRegAllB16 = MicroAPI::CreateMask<INPUT_T, MicroAPI::MaskPattern::ALL>();

        // false: normal mode; true: higher precision mode
        static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, false};
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant, antiQuantUb);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven, vSrcRegQuant, pRegAllB16);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd, vSrcRegQuant, pRegAllB16);
        MicroAPI::Interleave<T>(vSrcRegQuantEven, vSrcRegQuantOdd, vSrcRegQuantEven, vSrcRegQuantOdd);
        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vSrcRegSum, expSumUb + i * reduceSize);
            static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (dSize == 128) { // 真实d向上对齐至128的情况
                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc, preUb + i * dSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize);
                }
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, vSrcRegQuantEven,
                                                                   maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize, vDstRegDiv,
                                                                                 maskRegAll);

                if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                    MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(vregSrc,
                                                                                  preUb + i * dSize + floatRepSize);
                    MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                    MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                           maskRegAll);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * dSize + floatRepSize);
                }

                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, vSrcRegQuantOdd, maskRegAll);
                MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dSize + floatRepSize,
                                                                                 vDstRegDiv, maskRegAll);
            } else {
                for (uint16_t j = 0; j < dLoops; ++j) {
                    if constexpr (IsSameType<MMOUTPUT_T, int32_t>::value) {
                        MicroAPI::DataCopy<MMOUTPUT_T, MicroAPI::LoadDist::DIST_NORM>(
                            vregSrc, preUb + i * dSize + j * floatRepSize);
                        MicroAPI::Cast<T, MMOUTPUT_T, castTrait>(vSrcRegPre, vregSrc, maskRegAll);
                        MicroAPI::Muls<T, T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, dequantScale2,
                                                                               maskRegAll);

                        MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant,
                                                                                   antiQuantUb + j * floatRepSize);
                        MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven, vSrcRegQuant, pRegAllB16);
                    } else {
                        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre,
                                                                             preUb + i * dSize + j * floatRepSize);
                    }

                    MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, vSrcRegQuantEven,
                                                                       maskRegAll);
                    MicroAPI::Div<T, &mode>(vDstRegDiv, vSrcRegPre, vSrcRegSum, maskRegAll);
                    MicroAPI::DataCopy<OUTPUT_T, MicroAPI::StoreDist::DIST_NORM_B32>(
                        dstUb + i * dSize + j * floatRepSize, vDstRegDiv, maskRegAll);
                }
            }
        }
    }
}
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0,
          bool isPseudoQuant = false>
__aicore__ inline void FlashUpdate(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
                                   const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                   const LocalTensor<INPUT_T>& antiQuantResUb, const uint16_t m, const uint16_t d,
                                   float dequantScale2)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdate, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if constexpr (isPseudoQuant) {
        FlashUpdateNoTailAntiQuant<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor,
                                                                            expMaxTensor, antiQuantResUb, m, d);
    } else {
        if (d % floatRepSize == 0) {
            FlashUpdateNoTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor, expMaxTensor,
                                                                          m, d, dequantScale2);
        } else {
            FlashUpdateTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, curTensor, preTensor, expMaxTensor,
                                                                        m, d, dequantScale2);
        }
    }
}

template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0,
          bool isPseudoQuant = false>
__aicore__ inline void FlashUpdateLast(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& curTensor,
                                       const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor,
                                       const LocalTensor<T>& expSumTensor, const LocalTensor<INPUT_T>& antiQuantResUb,
                                       const uint16_t m, const uint16_t d, float dequantScale2)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if constexpr (isPseudoQuant) {
        FlashUpdateLastNoTailAntiQuant<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(
            dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, antiQuantResUb, m, d);
    } else {
        if (d % floatRepSize == 0) {
            FlashUpdateLastNoTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(
                dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, dequantScale2);
        } else {
            FlashUpdateLastTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(
                dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, dequantScale2);
        }
    }
}

// template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0>
// __aicore__ inline void FlashUpdateDiv(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& preTensor,
//     const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d, float dequantScale2)
// {
//     static_assert(IsSameType<T, float>::value, "VF FlashUpdateDiv, T must be float");

//     constexpr uint16_t floatRepSize = 64;
//     if (d % floatRepSize == 0) {
//         FlashUpdateDivNoTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, preTensor, expSumTensor, m, d,
//         dequantScale2);
//     }
// }
template <typename T, typename OUTPUT_T, typename INPUT_T, typename MMOUTPUT_T, uint32_t dSize = 0,
          bool isPseudoQuant = false>
__aicore__ inline void FlashUpdateDiv(const LocalTensor<T>& dstTensor, const LocalTensor<MMOUTPUT_T>& preTensor,
                                      const LocalTensor<T>& expSumTensor, const LocalTensor<INPUT_T>& antiQuantResUb,
                                      const uint16_t m, const uint16_t d, float dequantScale2)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateDiv, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if constexpr (isPseudoQuant) {
        FlashUpdateDivNoTailAntiQuant<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, preTensor, expSumTensor,
                                                                               antiQuantResUb, m, d, dequantScale2);
    } else if (d % floatRepSize == 0) {
        FlashUpdateDivNoTail_VF<T, OUTPUT_T, INPUT_T, MMOUTPUT_T, dSize>(dstTensor, preTensor, expSumTensor, m, d,
                                                                         dequantScale2);
    }
}

// template <typename T>
// __aicore__ inline void ComputeLseOutput_VF(const LocalTensor<T>& dstTensor, const LocalTensor<T>& softmaxSumTensor,
//     const LocalTensor<T>& softmaxMaxTensor, uint32_t dealCount)
// {
//     __ubuf__ T * srcSumUb = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
//     __ubuf__ T * srcMaxUb = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();
//     __ubuf__ T * dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();

//     __VEC_SCOPE__
//     {
//         MicroAPI::RegTensor<T> vregSum;
//         MicroAPI::RegTensor<T> vregMax;
//         MicroAPI::RegTensor<T> vregRes;
//         const uint32_t dealRows = 8;
//         const uint32_t  floatRepSize = 64; // 64: 一个寄存器存64个float
//         uint16_t updateLoops = ((dealCount) + (dealRows - 1)) / dealRows;
//         MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

//         for (uint16_t i = 0; i < updateLoops; ++i) {
//             MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregSum, srcSumUb + (i * dealRows));
//             MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregMax, srcMaxUb + (i * dealRows));

//             MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSum, pregAll);
//             MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregAll);

//             MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + (i * floatRepSize), vregRes, pregAll);
//         }
//     }
// }

template <typename T, typename INPUT_T, uint32_t dSize = 0>
__aicore__ inline void ComputeFirstInQuantVecNoTail(const LocalTensor<T>& dstTensor, const LocalTensor<T>& preTensor,
                                                    const LocalTensor<INPUT_T>& antiQuantResUb, const uint16_t m,
                                                    const uint16_t d)
{
    __ubuf__ float* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float* preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ INPUT_T* antiQuantUb = (__ubuf__ INPUT_T*)antiQuantResUb.GetPhyAddr();
    constexpr uint16_t floatRepSize = 64;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vSrcRegMax;
        MicroAPI::RegTensor<T> vSrcRegPre;
        MicroAPI::RegTensor<T> vSrcRegMul;
        MicroAPI::RegTensor<T> vDstRegAdd;
        MicroAPI::RegTensor<T> vSrcRegPre1;
        MicroAPI::RegTensor<T> vSrcRegMul1;
        MicroAPI::RegTensor<T> vDstRegAdd1;
        MicroAPI::RegTensor<T> vSrcRegPre2;
        MicroAPI::RegTensor<T> vSrcRegMul2;
        MicroAPI::RegTensor<T> vDstRegAdd2;
        MicroAPI::RegTensor<T> vSrcRegPre3;
        MicroAPI::RegTensor<T> vSrcRegMul3;
        MicroAPI::RegTensor<T> vDstRegAdd3;
        MicroAPI::RegTensor<T> vSrcRegPre4;
        MicroAPI::RegTensor<T> vSrcRegMul4;
        MicroAPI::RegTensor<T> vDstRegAdd4;
        MicroAPI::RegTensor<T> vSrcRegPre5;
        MicroAPI::RegTensor<T> vSrcRegMul5;
        MicroAPI::RegTensor<T> vDstRegAdd5;
        MicroAPI::RegTensor<T> vSrcRegPre6;
        MicroAPI::RegTensor<T> vSrcRegMul6;
        MicroAPI::RegTensor<T> vDstRegAdd6;
        MicroAPI::RegTensor<T> vSrcRegPre7;
        MicroAPI::RegTensor<T> vSrcRegMul7;
        MicroAPI::RegTensor<T> vDstRegAdd7;

        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant;
        MicroAPI::RegTensor<T> vSrcRegQuantEven;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant1;
        MicroAPI::RegTensor<T> vSrcRegQuantEven1;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd1;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant2;
        MicroAPI::RegTensor<T> vSrcRegQuantEven2;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd2;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant3;
        MicroAPI::RegTensor<T> vSrcRegQuantEven3;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd3;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant4;
        MicroAPI::RegTensor<T> vSrcRegQuantEven4;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd4;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant5;
        MicroAPI::RegTensor<T> vSrcRegQuantEven5;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd5;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant6;
        MicroAPI::RegTensor<T> vSrcRegQuantEven6;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd6;
        MicroAPI::RegTensor<INPUT_T> vSrcRegQuant7;
        MicroAPI::RegTensor<T> vSrcRegQuantEven7;
        MicroAPI::RegTensor<T> vSrcRegQuantOdd7;
        MicroAPI::MaskReg pRegAllB16 = MicroAPI::CreateMask<INPUT_T, MicroAPI::MaskPattern::ALL>();

        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        // 手动unroll，性能达到最优
        MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant, antiQuantUb);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven, vSrcRegQuant, pRegAllB16);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd, vSrcRegQuant, pRegAllB16);
        MicroAPI::Interleave<T>(vSrcRegQuantEven, vSrcRegQuantOdd, vSrcRegQuantEven, vSrcRegQuantOdd);
        MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant1, antiQuantUb + floatRepSize);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven1, vSrcRegQuant1, pRegAllB16);
        MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd1, vSrcRegQuant1, pRegAllB16);
        MicroAPI::Interleave<T>(vSrcRegQuantEven1, vSrcRegQuantOdd1, vSrcRegQuantEven1, vSrcRegQuantOdd1);
        if constexpr (dSize > 128) {
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant2, antiQuantUb + 2 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven2, vSrcRegQuant2, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd2, vSrcRegQuant2, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven2, vSrcRegQuantOdd2, vSrcRegQuantEven2, vSrcRegQuantOdd2);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant3, antiQuantUb + 3 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven3, vSrcRegQuant3, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd3, vSrcRegQuant3, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven3, vSrcRegQuantOdd3, vSrcRegQuantEven3, vSrcRegQuantOdd3);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant4, antiQuantUb + 4 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven4, vSrcRegQuant4, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd4, vSrcRegQuant4, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven4, vSrcRegQuantOdd4, vSrcRegQuantEven4, vSrcRegQuantOdd4);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant5, antiQuantUb + 5 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven5, vSrcRegQuant5, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd5, vSrcRegQuant5, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven5, vSrcRegQuantOdd5, vSrcRegQuantEven5, vSrcRegQuantOdd5);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant6, antiQuantUb + 6 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven6, vSrcRegQuant6, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd6, vSrcRegQuant6, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven6, vSrcRegQuantOdd6, vSrcRegQuantEven6, vSrcRegQuantOdd6);
            MicroAPI::DataCopy<INPUT_T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegQuant7, antiQuantUb + 7 * floatRepSize);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_0>(vSrcRegQuantEven7, vSrcRegQuant7, pRegAllB16);
            MicroAPI::Cast<T, INPUT_T, castTrait16232_1>(vSrcRegQuantOdd7, vSrcRegQuant7, pRegAllB16);
            MicroAPI::Interleave<T>(vSrcRegQuantEven7, vSrcRegQuantOdd7, vSrcRegQuantEven7, vSrcRegQuantOdd7);
        }

        for (uint16_t i = 0; i < m; ++i) {
            // 手动unroll，性能达到最优
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre, preUb + i * d);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre, vSrcRegPre, vSrcRegQuantEven, maskRegAll);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * d, vSrcRegPre, maskRegAll);

            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre1, preUb + i * d + floatRepSize);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre1, vSrcRegPre1, vSrcRegQuantEven1, maskRegAll);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * d + floatRepSize, vSrcRegPre1,
                                                                      maskRegAll);

            if constexpr (dSize > 128) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre2, preUb + i * d + 2 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre2, vSrcRegPre2, vSrcRegQuantEven2,
                                                                   maskRegAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * d + 2 * floatRepSize, vSrcRegPre2,
                                                                          maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre3, preUb + i * d + 3 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre3, vSrcRegPre3, vSrcRegQuantEven3,
                                                                   maskRegAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * d + 3 * floatRepSize, vSrcRegPre3,
                                                                          maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre4, preUb + i * d + 4 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre4, vSrcRegPre4, vSrcRegQuantEven4,
                                                                   maskRegAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * d + 4 * floatRepSize, vSrcRegPre4,
                                                                          maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre5, preUb + i * d + 5 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre5, vSrcRegPre5, vSrcRegQuantEven5,
                                                                   maskRegAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * d + 5 * floatRepSize, vSrcRegPre5,
                                                                          maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre6, preUb + i * d + 6 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre6, vSrcRegPre6, vSrcRegQuantEven6,
                                                                   maskRegAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * d + 6 * floatRepSize, vSrcRegPre6,
                                                                          maskRegAll);

                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vSrcRegPre7, preUb + i * d + 7 * floatRepSize);
                MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vSrcRegPre7, vSrcRegPre7, vSrcRegQuantEven7,
                                                                   maskRegAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * d + 7 * floatRepSize, vSrcRegPre7,
                                                                          maskRegAll);
            }
        }
    }
}

} // namespace AscendC

#endif // VF_FLASH_UPDATE_H
