/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. */

/*!
 * \file quant_grouped_matmul_dequant_gemv.h
 * \brief
 */
#ifndef _ASCENDC_QUANT_GROUPED_MATMUL_DEQUANT_GEMV_STAGED_H_
#define _ASCENDC_QUANT_GROUPED_MATMUL_DEQUANT_GEMV_STAGED_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_grouped_matmul_dequant_staged_base.h"
using namespace AscendC;

namespace AscendC {
namespace qgmmd_staged {
class QuantMatmulDequantGemv : public QuantMatmulDequantBase {
public:
    __aicore__ inline QuantMatmulDequantGemv() {}

    __aicore__ inline void Process()
    {
        uint32_t realsingleCoreFracN =
            tilingData->singleCoreFracN - (block_id < tilingData->singleCoreFracNTail ? 0 : 1);
        if (tilingData->dynamicQuant) {
            ProcessXScale();
            MySyncAll();
        }
        if (realsingleCoreFracN == 0) {
            return;
        }
        // x quantize and ub->L1->L0A
        ProcessX();

        uint32_t iterNL0C = (realsingleCoreFracN + baseFracNL0C - 1) / baseFracNL0C;
        uint32_t baseFracNL0CTail = (realsingleCoreFracN - 1) % baseFracNL0C + 1;
        uint32_t offsetFracN =
            tilingData->singleCoreFracN * block_id -
            (block_id > tilingData->singleCoreFracNTail ? (block_id - tilingData->singleCoreFracNTail) : 0);
        for (int32_t i = 0; i < iterNL0C; i++) {
            uint32_t realBaseFracNL0C = (i != (iterNL0C - 1)) ? baseFracNL0C : baseFracNL0CTail;
            // mm
            ProcessMM(realBaseFracNL0C, offsetFracN);
            // y L0C->ub and dequantize
            ProcessY(realBaseFracNL0C, offsetFracN);

            offsetFracN += realBaseFracNL0C;
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR bias,
                                GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y,
                                GM_ADDR usrWorkspace, const QuantGroupedMatmulDequantTilingData *__restrict qmmTiling)
    {
        // block_id
        block_id = GetBlockIdx();
        tilingData = qmmTiling;

        TilingInKernel();
        // alloc eventID
        InitEventId();

        InitGlobalTensors(x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, y, usrWorkspace);

        InitLocalTensors();

        if (tilingData->dynamicQuant) {
            InitSyncWs();
        }
    }

protected:
    __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR bias,
                                             GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y,
                                             GM_ADDR usrWorkspace)
    {
        InitCommonGlobalTensors(x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, y,
                                usrWorkspace);
    }

    __aicore__ inline void InitLocalTensors()
    {
        pipe.InitBuffer(UbBuf, UB_SIZE);
        LocalTensor<uint8_t> tmp = UbBuf.Get<uint8_t>();

        InitCommonLocalTensors(tmp);

        ubXHalfGemv = tmp.ReinterpretCast<half>();
        ubXFloatGemv = ubXHalfGemv[ubBaseK * realM].ReinterpretCast<float>();
        ubXInt8Gemv = ubXFloatGemv[ubBaseK].ReinterpretCast<int8_t>();
        if (tilingData->smoothScale) {
            ubSmoothScaleGemv = ubXInt8Gemv[ubBaseK * realM].ReinterpretCast<half>();
            ubSmoothScaleFloatGemv = ubSmoothScaleGemv[ubBaseK].ReinterpretCast<float>();
            if (tilingData->perToken) {
                ubXScaleFloatGemv = ubSmoothScaleFloatGemv[ubBaseK].ReinterpretCast<float>();
            }
        } else {
            if (tilingData->perToken) {
                ubXScaleFloatGemv = ubXInt8Gemv[ubBaseK * realM].ReinterpretCast<float>();
            }
        }

        ubYInt32Gemv = tmp.ReinterpretCast<int32_t>();
        ubYFloatGemv = tmp.ReinterpretCast<float>();
        ubWScaleGemv = tmp[UB_SIZE / NUMBER_2].ReinterpretCast<float>();
        ubYHalfGemv = tmp.ReinterpretCast<half>();

        pipe.InitBuffer(L1Buf, L1_SIZE);
        l1WGemv[0] = L1Buf.Get<int8_t>();
        l1WGemv[1] = l1WGemv[0][L0A_SIZE];
        l1XGemv = l1WGemv[1][L0A_SIZE];
        pipe.InitBuffer(L0ABuf, L0A_SIZE);
        l0aXGemv = L0ABuf.Get<int8_t>();
        pipe.InitBuffer(L0BBuf, L0B_SIZE);
        l0bWGemv[0] = L0BBuf.Get<int8_t>();
        l0bWGemv[1] = l0bWGemv[0][L0A_SIZE / NUMBER_2];
        pipe.InitBuffer(L0CBuf, L0C_SIZE);
        l0cYGemv = L0CBuf.Get<int32_t>();
    }

    __aicore__ inline void TilingInKernel()
    {
        realM = tilingData->originM;
        baseFracNL0C = tilingData->baseFracNL0C;
        ubBaseK = tilingData->ubBaseK;
        ubIterK = tilingData->ubIterK;
        ubBaseKTail = tilingData->ubBaseKTail;
    }

    __aicore__ inline void ProcessX()
    {
        DataCopyParams repeatParamsHalf;
        DataCopyParams repeatParamsFloat;
        DataCopyParams repeatParamsInt8;
        UnaryRepeatParams unaryParams;
        UnaryRepeatParams unaryParamsH2I8;
        UnaryRepeatParams unaryParamsH2F;
        UnaryRepeatParams unaryParamsF2I;
        BinaryRepeatParams binaryParams;
        uint64_t mask[NUMBER_2] = {tilingData->ubKMask, 0};
        uint32_t baseOffsetK = 0;
        unaryParamsH2I8.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
        unaryParamsH2F.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
        if (tilingData->perToken) {
            if (tilingData->dynamicQuant) {
                repeatParamsFloat.blockLen = realM;
            } else {
                repeatParamsFloat.blockLen = (realM + FLOAT_PERBLOCK - 1) / FLOAT_PERBLOCK;
            }
            DataCopy<float>(ubXScaleFloatGemv, xScaleGm, repeatParamsFloat);
            SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS[0]);
        }
        for (int32_t i = 0; i < ubIterK; i++) {
            uint32_t realUbBaseK = (i != (ubIterK - 1)) ? ubBaseK : ubBaseKTail;
            uint32_t xGm_offset = baseOffsetK;
            uint32_t ubX_offset = 0;
            uint32_t l1X_offset = baseOffsetK;
            repeatParamsHalf.blockLen = (realUbBaseK + HALF_PERBLOCK - 1) / HALF_PERBLOCK;
            repeatParamsInt8.blockLen = (realUbBaseK + INT8_PERBLOCK - 1) / INT8_PERBLOCK;
            uint32_t realUbBaseKAligned32 = repeatParamsInt8.blockLen * INT8_PERBLOCK;
            SetMaskCount();
            SetVectorMask<half, MaskMode::COUNTER>(realUbBaseKAligned32);
            if (tilingData->smoothScale) {
                DataCopy<half>(ubSmoothScaleGemv, smoothScaleGm[xGm_offset], repeatParamsHalf);
                SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
                Cast<float, half, false>(ubSmoothScaleFloatGemv, ubSmoothScaleGemv, RoundMode::CAST_NONE,
                                         MASK_PLACEHOLDER, 1, unaryParamsH2F);
            }
            for (int32_t j = 0; j < realM; j++) {
                DataCopy<half>(ubXHalfGemv[ubX_offset], xGm[xGm_offset], repeatParamsHalf);
                SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
                Cast<float, half, false>(ubXFloatGemv, ubXHalfGemv[ubX_offset], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                                         1, unaryParamsH2F);

                if (tilingData->smoothScale) {
                    PipeBarrier<PIPE_V>();
                    Mul<float, false>(ubXFloatGemv, ubXFloatGemv, ubSmoothScaleFloatGemv, MASK_PLACEHOLDER, 1,
                                      binaryParams);
                }

                PipeBarrier<PIPE_V>();
                if (tilingData->perToken) {
                    if (i == 0 && j == 0)
                        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS[0]);
                    if (i == 0) {
                        if (tilingData->dynamicQuant) {
                            pertokenScaleDequant[j] = ubXScaleFloatGemv.GetValue(j * FLOAT_PERBLOCK);
                        } else {
                            pertokenScaleDequant[j] = ubXScaleFloatGemv.GetValue(j);
                        }
                    }
                    Muls<float, false>(ubXFloatGemv, ubXFloatGemv, FLOAT_1 / pertokenScaleDequant[j], MASK_PLACEHOLDER,
                                       1, unaryParams);
                } else {
                    Muls<float, false>(ubXFloatGemv, ubXFloatGemv, x_scale_quant, MASK_PLACEHOLDER, 1, unaryParams);
                }

                PipeBarrier<PIPE_V>();
                Cast<int32_t, float, false>(ubXFloatGemv.ReinterpretCast<int32_t>(), ubXFloatGemv, RoundMode::CAST_RINT,
                                            MASK_PLACEHOLDER, 1, unaryParamsF2I);
                PipeBarrier<PIPE_V>();
                SetDeqScale((half)FLOAT_1);
                Cast<half, int32_t, false>(ubXHalfGemv[ubX_offset], ubXFloatGemv.ReinterpretCast<int32_t>(),
                                           RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsH2I8);
                if (realUbBaseK != realUbBaseKAligned32) {
                    PipeBarrier<PIPE_V>();
                    SetMaskNorm();
                    Duplicate<half>(ubXHalfGemv[ubX_offset + realUbBaseKAligned32 - K_FRACTAL_INT8], FLOAT_0, mask, 1,
                                    DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
                    SetMaskCount();
                    SetVectorMask<half, MaskMode::COUNTER>(realUbBaseKAligned32);
                }
                PipeBarrier<PIPE_V>();
                Cast<int8_t, half, false>(ubXInt8Gemv[ubX_offset], ubXHalfGemv[ubX_offset], RoundMode::CAST_ROUND,
                                          MASK_PLACEHOLDER, 1, unaryParamsH2I8);
                SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);

                WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
                DataCopy<int8_t>(l1XGemv[l1X_offset], ubXInt8Gemv[ubX_offset], repeatParamsInt8);

                xGm_offset += tilingData->originK;
                ubX_offset += realUbBaseKAligned32;
                l1X_offset += tilingData->originKAligned512;
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
            baseOffsetK += realUbBaseK;
        }
        SetFlag<HardEvent::MTE3_MTE1>(eventIdMTE3ToMTE1[0]);
        WaitFlag<HardEvent::MTE3_MTE1>(eventIdMTE3ToMTE1[0]);
    }

    __aicore__ inline void ProcessMM(uint32_t realBaseFracNL0C, uint32_t offsetFracN)
    {
        uint32_t iterK = (tilingData->fracK + tilingData->baseFracK - 1) / tilingData->baseFracK;
        uint32_t iterN = (realBaseFracNL0C + tilingData->baseFracN - 1) / tilingData->baseFracN;
        uint32_t baseFracNTail = (realBaseFracNL0C - 1) % tilingData->baseFracN + 1;
        uint32_t baseFracKTail = (tilingData->fracK - 1) % tilingData->baseFracK + 1;
        uint32_t offsetFracK = 0;
        DataCopyParams repeatParamsInt8;
        LoadData2dParams loadData2DA;
        LoadData2dParams loadData2DB;
        MmadParams mmadParams;
        mmadParams.m = 1;
        mmadParams.cmatrixInitVal = true;
        loadData2DB.srcStride = 1;
        loadData2DB.ifTranspose = false;
        loadData2DA.repeatTimes = realM;
        loadData2DA.srcStride = tilingData->originKAligned512 / L0_ADDR_ALIGN;
        loadData2DA.ifTranspose = false;
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[0]);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[1]);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[NUMBER_2]);
        for (int32_t i = 0; i < iterK; i++) {
            uint32_t realBaseFracK = (i != (iterK - 1)) ? tilingData->baseFracK : baseFracKTail;
            mmadParams.k = realBaseFracK * K_FRACTAL_INT8;
            uint32_t offsetFracN_ = offsetFracN;
            WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[NUMBER_2]);
            LoadData<int8_t>(l0aXGemv, l1XGemv[offsetFracK * K_FRACTAL_INT8], loadData2DA);
            SetFlag<HardEvent::MTE1_M>(eventIdMTE1ToM[NUMBER_2]);
            WaitFlag<HardEvent::MTE1_M>(eventIdMTE1ToM[NUMBER_2]);
            for (int32_t j = 0; j < iterN; j++) {
                uint32_t pingpong = j % NUMBER_2;
                uint32_t realBaseFracN = (j != (iterN - 1)) ? tilingData->baseFracN : baseFracNTail;
                mmadParams.n = realBaseFracN * NM_FRACTAL_INT8;
                repeatParamsInt8.blockCount = realBaseFracK;
                repeatParamsInt8.blockLen = realBaseFracN * L0_ADDR_ALIGN / INT8_PERBLOCK;
                repeatParamsInt8.srcStride = (tilingData->fracN - realBaseFracN) * L0_ADDR_ALIGN / INT8_PERBLOCK;
                repeatParamsInt8.dstStride = DEFAULT_DATA_COPY_STRIDE;
                WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[pingpong]);
                DataCopy<int8_t>(l1WGemv[pingpong],
                                 quantizedWeightGm[(offsetFracK * tilingData->fracN + offsetFracN_) * NM_FRACTAL_INT8 *
                                                   K_FRACTAL_INT8],
                                 repeatParamsInt8);
                SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[pingpong]);

                WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[pingpong]);
                WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[pingpong]);
                loadData2DB.repeatTimes = realBaseFracK * realBaseFracN;
                LoadData<int8_t>(l0bWGemv[pingpong], l1WGemv[pingpong], loadData2DB);
                SetFlag<HardEvent::MTE1_M>(eventIdMTE1ToM[pingpong]);
                SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[pingpong]);

                WaitFlag<HardEvent::MTE1_M>(eventIdMTE1ToM[pingpong]);
                for (int32_t k = 0; k < realM; k++) {
                    Mmad<int32_t, int8_t, int8_t>(
                        l0cYGemv[(k * realBaseFracNL0C + offsetFracN_ - offsetFracN) * L0C_FRACTAL * L0C_FRACTAL],
                        l0aXGemv[k * K_FRACTAL_INT8 * NM_FRACTAL_INT8], l0bWGemv[pingpong], mmadParams);
                }
                SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[pingpong]);

                offsetFracN_ += realBaseFracN;
            }
            SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[NUMBER_2]);
            if (i == 0)
                mmadParams.cmatrixInitVal = false;
            offsetFracK += realBaseFracK;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[0]);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[1]);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[NUMBER_2]);
        SetFlag<HardEvent::M_V>(eventIdMToV[0]);
        WaitFlag<HardEvent::M_V>(eventIdMToV[0]);
    }

    __aicore__ inline void ProcessY(uint32_t realBaseFracNL0C, uint32_t offsetFracN)
    {
        DataCopyEnhancedParams enhancedParams;
        DataCopyParams repeatParams;
        UnaryRepeatParams unaryParams;
        UnaryRepeatParams unaryParamsI2F;
        UnaryRepeatParams unaryParamsF2H;
        BinaryRepeatParams binaryParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;
        unaryParamsF2H.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;

        PipeBarrier<PIPE_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);

        WeightScaleCopyInAndCast(ubWScaleGemv, offsetFracN * L0C_FRACTAL, repeatParams, realBaseFracNL0C * L0C_FRACTAL);

        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        repeatParams.blockLen = realM * realBaseFracNL0C;
        DataCopy<int32_t>(ubYInt32Gemv, l0cYGemv, repeatParams, enhancedParams);
        SetFlag<HardEvent::V_M>(eventIdVToM[0]);
        WaitFlag<HardEvent::V_M>(eventIdVToM[0]);

        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(realM * realBaseFracNL0C * L0C_FRACTAL);
        Cast<float, int32_t, false>(ubYFloatGemv, ubYInt32Gemv, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                                    unaryParamsI2F);

        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(realBaseFracNL0C * L0C_FRACTAL);
        for (int32_t i = 0; i < realM; i++) {
            uint32_t offset = i * realBaseFracNL0C * L0C_FRACTAL;
            Mul<float, false>(ubYFloatGemv[offset], ubYFloatGemv[offset], ubWScaleGemv, MASK_PLACEHOLDER, 1,
                              binaryParams);
        }

        if (!isWScaleInt64) {
            PipeBarrier<PIPE_V>();
            if (tilingData->perToken) {
                for (int32_t i = 0; i < realM; i++) {
                    uint32_t offset = i * realBaseFracNL0C * L0C_FRACTAL;
                    Muls<float, false>(ubYFloatGemv[offset], ubYFloatGemv[offset], pertokenScaleDequant[i],
                                       MASK_PLACEHOLDER, 1, unaryParams);
                }
            } else {
                SetVectorMask<float, MaskMode::COUNTER>(realM * realBaseFracNL0C * L0C_FRACTAL);
                Muls<float, false>(ubYFloatGemv, ubYFloatGemv, x_scale_dequant, MASK_PLACEHOLDER, 1, unaryParams);
            }
        }

        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(realM * realBaseFracNL0C * L0C_FRACTAL);
        Cast<half, float, false>(ubYHalfGemv, ubYFloatGemv, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsF2H);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);

        repeatParams.blockCount = realM;
        repeatParams.blockLen = realBaseFracNL0C * L0C_FRACTAL / HALF_PERBLOCK;
        repeatParams.dstStride = (tilingData->originN - realBaseFracNL0C * L0C_FRACTAL) / HALF_PERBLOCK;
        DataCopy<half>(yGm[offsetFracN * L0C_FRACTAL], ubYHalfGemv, repeatParams);
    }

    LocalTensor<half> ubXHalfGemv;
    LocalTensor<float> ubXFloatGemv;
    LocalTensor<int8_t> ubXInt8Gemv;
    LocalTensor<half> ubSmoothScaleGemv;
    LocalTensor<float> ubXScaleFloatGemv;
    LocalTensor<float> ubSmoothScaleFloatGemv;

    LocalTensor<int8_t> l1XGemv;
    LocalTensor<int8_t> l1WGemv[NUMBER_2];
    LocalTensor<int8_t> l0aXGemv;
    LocalTensor<int8_t> l0bWGemv[NUMBER_2];
    LocalTensor<int32_t> l0cYGemv;

    LocalTensor<int32_t> ubYInt32Gemv;
    LocalTensor<float> ubYFloatGemv;
    LocalTensor<half> ubYHalfGemv;
    LocalTensor<float> ubWScaleGemv;

    float pertokenScaleDequant[GEMV_THRESHOLD];
    uint32_t baseFracNL0C;
    uint32_t ubBaseK;
    uint32_t ubIterK;
    uint32_t ubBaseKTail;
};
} // namespace qgmmd_staged
} // namespace AscendC
#endif
