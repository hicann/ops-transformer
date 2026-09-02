/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. */

/*!
 * \file quant_matmul_dequant_grouped.h
 * \brief
 */
#ifndef _ASCENDC_QUANT_MATMUL_DEQUANT_GROUPED_STAGED_H_
#define _ASCENDC_QUANT_MATMUL_DEQUANT_GROUPED_STAGED_H_

#include <cmath>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_grouped_matmul_dequant_staged_normal.h"
using namespace AscendC;

namespace AscendC {
namespace qgmmd_staged {
constexpr uint32_t SPECIAL_INIT_VAL_GMM = 123456;
constexpr uint32_t MYSYNCALL_SWIFT_THRES = 16;
constexpr uint32_t TILING2_THRESHOLD = 96;

class QuantMatmulDequantGrouped : public QuantMatmulDequantNormal {
public:
    __aicore__ inline QuantMatmulDequantGrouped() {}

    __aicore__ inline void Process()
    {
        uint32_t realsingleCoreFracN =
            tilingData->singleCoreFracN - (block_id < tilingData->singleCoreFracNTail ? 0 : 1);
        uint32_t startM = 0;
        uint32_t endM;
        fracM_ = 0;
        GlobalTensor<half> xGm0 = xGm;
        GlobalTensor<int8_t> quantizedWeightGm0 = quantizedWeightGm;
        GlobalTensor<float> wScaleGm0 = wScaleGm;
        GlobalTensor<int64_t> wScaleGmInt64_0 = wScaleGmInt64;
        GlobalTensor<float> xScaleGm0 = xScaleGm;
        GlobalTensor<half> yGm0 = yGm;
        FirstTiling();
        for (int32_t i = 0; i < tilingData->originE; i++) {
            if (startM >= tilingData->originM)
                return;
            endM = groupGM.GetValue(i);
            if (endM > tilingData->originM)
                endM = tilingData->originM;
            realM = endM - startM;
            if (realM == 0) {
                continue;
            }

            xGm = xGm0[(uint64_t)startM * (tilingData->originK)];
            quantizedWeightGm = quantizedWeightGm0[(uint64_t)i * (tilingData->fracK) * (tilingData->fracN) *
                                                   NM_FRACTAL_INT8 * K_FRACTAL_INT8];
            wScaleGm = wScaleGm0[(uint64_t)i * tilingData->originN];
            wScaleGmInt64 = wScaleGmInt64_0[(uint64_t)i * tilingData->originN];
            yGm = yGm0[(uint64_t)startM * (tilingData->originN)];

            if (tilingData->dynamicQuant) {
                ProcessXScale();
                MySyncAllSwift();
            } else if (tilingData->perToken) {
                xScaleGm = xScaleGm0[(uint64_t)startM];
            }

            startM = endM;
            if (realM > gemv_threshold || (tilingData->originKAligned512 * realM) > (L1_SIZE / NUMBER_2)) {
                if (!isSwift) {
                    TilingInKernelNormal();
                } else {
                    TilingInKernelSwift();
                }
                QuantMatmulDequantNormal::ProcessX();
                MySyncAllSwift();

                QuantMatmulDequantNormal::ProcessMM();
                MySyncAllSwift();
            } else {
                if (realsingleCoreFracN == 0) {
                    if (tilingData->dynamicQuant)
                        MySyncAllSwift();
                    continue;
                }
                TilingInKernelGemv();

                QuantMatmulDequantGemv::ProcessX();

                uint32_t iterNL0C = (realsingleCoreFracN + baseFracNL0C - 1) / baseFracNL0C;
                uint32_t baseFracNL0CTail = (realsingleCoreFracN - 1) % baseFracNL0C + 1;
                uint32_t offsetFracN =
                    tilingData->singleCoreFracN * block_id -
                    (block_id > tilingData->singleCoreFracNTail ? (block_id - tilingData->singleCoreFracNTail) : 0);
                for (int32_t j = 0; j < iterNL0C; j++) {
                    uint32_t realBaseFracNL0C = (j != (iterNL0C - 1)) ? baseFracNL0C : baseFracNL0CTail;
                    QuantMatmulDequantGemv::ProcessMM(realBaseFracNL0C, offsetFracN);
                    QuantMatmulDequantGemv::ProcessY(realBaseFracNL0C, offsetFracN);
                    offsetFracN += realBaseFracNL0C;
                }
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
                if (tilingData->dynamicQuant)
                    MySyncAllSwift();
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list,
                                GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y,
                                GM_ADDR usrWorkspace, const QuantGroupedMatmulDequantTilingData *__restrict qmmTiling)
    {
        // block_id
        block_id = GetBlockIdx();
        tilingData = qmmTiling;

        // alloc eventID
        InitEventId();

        InitGlobalTensors(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, y,
                          usrWorkspace);

        InitLocalTensors();

        InitSyncWsSwift();

        InitTransLocalLists();
    }
    __aicore__ inline void SetSwift()
    {
        isSwift = true;
    }

protected:
    __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale,
                                             GM_ADDR group_list, GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset,
                                             GM_ADDR smooth_scale, GM_ADDR y, GM_ADDR usrWorkspace)
    {
        groupGM.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(group_list), tilingData->originE);
        xZNGM.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(usrWorkspace),
                              (tilingData->originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 * NM_FRACTAL_INT8 *
                                  tilingData->originKAligned32);
        usrWorkspace += (tilingData->originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 * NM_FRACTAL_INT8 *
                        tilingData->originKAligned32;
        InitCommonGlobalTensors(x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, y,
                                usrWorkspace);
        quantizedWeightGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t *>(quantized_weight),
            tilingData->originE * tilingData->fracK * tilingData->fracN * NM_FRACTAL_INT8 * K_FRACTAL_INT8);
        if (isWScaleInt64) {
            wScaleGmInt64.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(weight_scale),
                                          tilingData->originE * tilingData->originN);
        } else {
            wScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_scale),
                                     tilingData->originE * tilingData->originN);
        }
    }

    __aicore__ inline void InitLocalTensors()
    {
        pipe.InitBuffer(UbBuf, UB_SIZE);
        LocalTensor<uint8_t> tmp = UbBuf.Get<uint8_t>();
        InitCommonLocalTensors(tmp);
        ubXHalfND = tmp.ReinterpretCast<half>();
        ubXHalfZN = ubXHalfND[tilingData->processXKBaseNMax * NM_FRACTAL_INT8];
        ubXFloatND = ubXHalfZN[tilingData->processXKBaseNMax * NM_FRACTAL_INT8].ReinterpretCast<float>();
        ubXInt8ZN = ubXHalfND[tilingData->processXKBaseNMax * NM_FRACTAL_INT8].ReinterpretCast<int8_t>();
        if (tilingData->smoothScale) {
            ubSmoothScale = ubXFloatND[tilingData->processXKBaseNMax * NM_FRACTAL_INT8].ReinterpretCast<half>();
            ubSmoothScaleFloat = ubSmoothScale[tilingData->processXKBaseNMax].ReinterpretCast<float>();
            if (tilingData->perToken)
                ubXScaleFloat = ubSmoothScaleFloat[tilingData->processXKBaseNMax].ReinterpretCast<float>();
        } else {
            if (tilingData->perToken)
                ubXScaleFloat = ubXFloatND[tilingData->processXKBaseNMax * NM_FRACTAL_INT8].ReinterpretCast<float>();
        }
        if (tilingData->perToken) {
            ubPertokenScale = tmp.ReinterpretCast<float>();
            if (tilingData->dynamicQuant)
                ubPertokenScaleRaw_ = ubPertokenScale.ReinterpretCast<float>();
        } else {
            ubYInt32 = tmp.ReinterpretCast<int32_t>();
        }

        ubXHalfGemv = tmp.ReinterpretCast<half>();
        ubYInt32Gemv = tmp.ReinterpretCast<int32_t>();
        ubYFloatGemv = tmp.ReinterpretCast<float>();
        ubWScaleGemv = tmp[UB_SIZE / NUMBER_2].ReinterpretCast<float>();
        ubYHalfGemv = tmp.ReinterpretCast<half>();

        pipe.InitBuffer(L1Buf, L1_SIZE);
        l1WGemv[0] = L1Buf.Get<int8_t>();
        l1WGemv[1] = l1WGemv[0][L0B_SIZE];
        l1XGemv = l1WGemv[1][L0B_SIZE];
        l1W_DB[0] = L1Buf.Get<int8_t>();
        l1W_DB[1] = l1W_DB[0][L0B_SIZE / NUMBER_2];
        l1X_DB[0] = l1W_DB[1][L0B_SIZE / NUMBER_2];
        l1X_DB[1] = l1X_DB[0][L0A_SIZE / NUMBER_2];

        pipe.InitBuffer(L0ABuf, L0A_SIZE);
        l0aXGemv = L0ABuf.Get<int8_t>();
        l0aX_DB[0] = L0ABuf.Get<int8_t>();
        l0aX_DB[1] = l0aX_DB[0][L0A_SIZE / NUMBER_2];

        pipe.InitBuffer(L0BBuf, L0B_SIZE);
        l0bWGemv[0] = L0BBuf.Get<int8_t>();
        l0bWGemv[1] = l0bWGemv[0][L0B_SIZE / NUMBER_2];
        l0bW_DB[0] = L0BBuf.Get<int8_t>();
        l0bW_DB[1] = l0bW_DB[0][L0B_SIZE / NUMBER_2];

        pipe.InitBuffer(L0CBuf, L0C_SIZE);
        l0cY = L0CBuf.Get<int32_t>();
        l0cYGemv = l0cY;
    }

    __aicore__ inline void InitTransLocalLists()
    {
        NDLocalList[0] = reinterpret_cast<uint64_t>(ubXHalfND.GetPhyAddr());
        ZNLocalList[0] = reinterpret_cast<uint64_t>(ubXHalfZN.GetPhyAddr());
        for (int32_t i = 1; i < NUMBER_16; i++) {
            ZNLocalList[i] = ZNLocalList[i - 1] + BLOCK_SIZE;
        }

        if (tilingData->perToken) {
            pertokenLocalList[0] = reinterpret_cast<uint64_t>(ubPertokenScale.GetPhyAddr());
            for (int32_t i = 1; i < NUMBER_16; i++) {
                pertokenLocalList[i] = pertokenLocalList[i - 1] + BLOCK_SIZE;
            }
            if (tilingData->dynamicQuant) {
                pertokenDynamicRawLocalList[0] = reinterpret_cast<uint64_t>(ubPertokenScaleRaw_.GetPhyAddr());
                for (int32_t i = 1; i < NUMBER_16; i++) {
                    pertokenDynamicRawLocalList[i] = pertokenDynamicRawLocalList[i - 1] + BLOCK_SIZE;
                }
            }
        }
    }

    __aicore__ inline void TilingInKernelNormal()
    {
        // x:quantize and ND->ZN
        fracM = (realM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        tailM = (realM - 1) % NM_FRACTAL_INT8 + 1;
        processXKloopPerfracM =
            (tilingData->originKAligned32 + tilingData->processXKBaseNMax - 1) / tilingData->processXKBaseNMax;
        processXKloop = (fracM * processXKloopPerfracM + tilingData->CoreNum - 1) / tilingData->CoreNum;
        processXKloopPerfracM = processXKloop * tilingData->CoreNum / fracM;
        processXKBaseN = (tilingData->fracK + processXKloopPerfracM - 1) / processXKloopPerfracM * K_FRACTAL_INT8;
        processXKTailN = tilingData->fracK % processXKloopPerfracM;
        processXKTailN = processXKTailN == 0 ? processXKloopPerfracM : processXKTailN;
        // mm
        // singleCoreM, singleCoreN
        int32_t chosen = 0;
        int32_t mte2Min = (tilingData->fracN << NUMBER_3) + (fracM << 0);
        for (int32_t i = 1; i < NUMBER_4; i++) {
            int32_t mte2Now = (tilingData->fracN << (NUMBER_3 - i)) + (fracM << i);
            if (mte2Now < mte2Min) {
                chosen = i;
                mte2Min = mte2Now;
            }
        }
        MCoreNum = 1 << (NUMBER_3 - chosen);
        NCoreNum = 1 << chosen;
        int32_t singleCoreM = (fracM + MCoreNum - 1) / MCoreNum;
        int32_t singleCoreN = (tilingData->fracN + NCoreNum - 1) / NCoreNum;
        int32_t singleCoreMTail = (fracM - 1) % MCoreNum + 1;
        int32_t singleCoreNTail = (tilingData->fracN - 1) % NCoreNum + 1;
        int32_t tmpM = ((int32_t)block_id / NCoreNum) - singleCoreMTail;
        int32_t tmpN = ((int32_t)block_id % NCoreNum) - singleCoreNTail;
        realSingleCoreM = singleCoreM - (tmpM < 0 ? 0 : 1);
        realSingleCoreN = singleCoreN - (tmpN < 0 ? 0 : 1);
        globalOffsetM = singleCoreM * ((int32_t)block_id / NCoreNum) - (tmpM > 0 ? tmpM : 0);
        globalOffsetN = singleCoreN * ((int32_t)block_id % NCoreNum) - (tmpN > 0 ? tmpN : 0);

        // baseMNum, baseNNum
        int32_t l0CMNFractal = NUMBER_256; // = 256 * 1024 / 2 / 16 / 16 / 4
        int32_t oriBaseMN = ScalarCast<float, int32_t, RoundMode::CAST_FLOOR>(std::sqrt((float)l0CMNFractal));
        int32_t extraN = tilingData->perToken ? 1 : 0;
        int32_t singleCoreMNFractal = (singleCoreM + 1) * (singleCoreN + extraN);
        baseMNum = 1;
        baseNNum = 1;
        if (singleCoreMNFractal > l0CMNFractal) {
            baseMNum = (singleCoreM + (oriBaseMN - 1) - 1) / (oriBaseMN - 1);
            baseNNum = (singleCoreN + (oriBaseMN - extraN) - 1) / (oriBaseMN - extraN);
            if (singleCoreM > singleCoreN) {
                uint32_t BaseMFractalN_ = (singleCoreM + baseMNum - 1) / baseMNum + 1;
                while (baseNNum > 1 &&
                       BaseMFractalN_ * ((singleCoreN + baseNNum - NUMBER_2) / (baseNNum - 1) + extraN) <=
                           l0CMNFractal) {
                    baseNNum -= 1;
                }
                uint32_t BaseNFractalN_ = (singleCoreN + baseNNum - 1) / baseNNum + extraN;
                while (baseMNum > 1 &&
                       BaseNFractalN_ * ((singleCoreM + baseMNum - NUMBER_2) / (baseMNum - 1) + 1) <= l0CMNFractal) {
                    baseMNum -= 1;
                }
            } else {
                uint32_t BaseNFractalN_ = (singleCoreN + baseNNum - 1) / baseNNum + extraN;
                while (baseMNum > 1 &&
                       BaseNFractalN_ * ((singleCoreM + baseMNum - NUMBER_2) / (baseMNum - 1) + 1) <= l0CMNFractal) {
                    baseMNum -= 1;
                }
                uint32_t BaseMFractalN_ = (singleCoreM + baseMNum - 1) / baseMNum + 1;
                while (baseNNum > 1 &&
                       BaseMFractalN_ * ((singleCoreN + baseNNum - NUMBER_2) / (baseNNum - 1) + extraN) <=
                           l0CMNFractal) {
                    baseNNum -= 1;
                }
            }
        }

        int32_t BaseMNumL0AB = (singleCoreM + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        int32_t BaseNNumL0AB = (singleCoreN + NUMBER_128 / NUMBER_2 - 1) / (NUMBER_128 / NUMBER_2);
        if (baseMNum < BaseMNumL0AB)
            baseMNum = BaseMNumL0AB;
        if (baseNNum < BaseNNumL0AB)
            baseNNum = BaseNNumL0AB;

        baseM = (realSingleCoreM + baseMNum - 1) / baseMNum;
        baseN = (realSingleCoreN + baseNNum - 1) / baseNNum;
        baseMTail = (realSingleCoreM - 1) % baseMNum + 1;
        baseNTail = (realSingleCoreN - 1) % baseNNum + 1;
        // baseKNum
        int32_t l0AMKFractal = NUMBER_128 / NUMBER_2 / baseM; // = 64 * 1024 / 16 / 32
        int32_t l0BNKFractal = NUMBER_128 / NUMBER_2 / baseN;
        int32_t MKPieces = (tilingData->fracK + l0AMKFractal - 1) / l0AMKFractal;
        int32_t NKPieces = (tilingData->fracK + l0BNKFractal - 1) / l0BNKFractal;
        baseKNum = MKPieces > NKPieces ? MKPieces : NKPieces;
        baseK = (tilingData->fracK + baseKNum - 1) / baseKNum;
        baseKTail = (tilingData->fracK - 1) % baseKNum + 1;

        if (tilingData->perToken) {
            ubPertokenScaleRaw = ubPertokenScale[baseM * L0C_FRACTAL * (L0C_FRACTAL - 1)].ReinterpretCast<float>();
            ubYInt32 = ubPertokenScale[baseM * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<int32_t>();
        }
        ubYFloat = ubYInt32.ReinterpretCast<float>();
        ubYHalfNZ = ubYInt32.ReinterpretCast<half>();
        ubYHalfND = ubYHalfNZ[baseM * baseN * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<half>();
        ubWScale = ubYHalfND[baseM * baseN * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<float>();

        for (int32_t i = 1; i < NUMBER_16; i++) {
            NDLocalList[i] = NDLocalList[i - 1] + processXKBaseN * sizeof(half);
        }

        if (tilingData->perToken) {
            pertokenRawLocalList[0] = reinterpret_cast<uint64_t>(ubPertokenScaleRaw.GetPhyAddr());
            for (int32_t i = 1; i < NUMBER_16; i++) {
                pertokenRawLocalList[i] = pertokenRawLocalList[i - 1];
            }
            if (tilingData->dynamicQuant) {
                pertokenDynamicLocalList[0] = reinterpret_cast<uint64_t>(ubPertokenScaleRaw.GetPhyAddr());
                pertokenDynamicLocalList[1] = pertokenDynamicLocalList[0] + BLOCK_SIZE;
                pertokenDynamicLocalList[NUMBER_2] =
                    reinterpret_cast<uint64_t>(ubPertokenScaleRaw_[baseM * NUMBER_128].GetPhyAddr());
                for (int32_t i = NUMBER_3; i < NUMBER_16; i++) {
                    pertokenDynamicLocalList[i] = pertokenDynamicLocalList[i - 1] + BLOCK_SIZE;
                }
            }
        }
    }

    __aicore__ inline void TilingInKernelGemv()
    {
        baseFracNL0C = UB_SIZE / sizeof(int32_t) / L0C_FRACTAL / L0C_FRACTAL / realM;
        ubBaseK = (UB_SIZE - (tilingData->perToken ? realM * BLOCK_SIZE : 0)) /
                  (realM * (sizeof(int8_t) + sizeof(half)) + sizeof(float) +
                   (tilingData->smoothScale ? (sizeof(half) + sizeof(float)) : 0)) /
                  NUMBER_256 * NUMBER_256;
        ubIterK = (tilingData->originK + ubBaseK - 1) / ubBaseK;
        ubBaseKTail = (tilingData->originK - 1) % ubBaseK + 1;

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
    }
    __aicore__ inline void FirstTiling()
    {
        if (isSwift) {
            realM = NUMBER_16;
            isTiling1 = false;
            MCoreNum = tilingData->MCoreNum;
            NCoreNum = tilingData->NCoreNum;
            TilingInKernelSwift();
            gemv_threshold = tilingData->swiftGEMVThreshold;
        } else {
            gemv_threshold = GEMV_THRESHOLD;
        }
    }
    __aicore__ inline void TilingInKernelSwift()
    {
        fracM = (realM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8;
        tailM = (realM - 1) % NM_FRACTAL_INT8 + 1;
        if (fracM != fracM_) {
            ChooseTiling1or2();
            processXKloop = (fracM * tilingData->processXKloopPerfracM + tilingData->CoreNum - 1) / tilingData->CoreNum;
            processXKloopPerfracM = processXKloop * tilingData->CoreNum / fracM;
            processXKBaseN = (tilingData->fracK + processXKloopPerfracM - 1) / processXKloopPerfracM * K_FRACTAL_INT8;
            processXKTailN = tilingData->fracK % processXKloopPerfracM;
            processXKTailN = processXKTailN == 0 ? processXKloopPerfracM : processXKTailN;

            int32_t singleCoreM = fracM;
            int32_t singleCoreMTail = fracM;
            int32_t tmpM = ((int32_t)block_id / tilingData->NCoreNum) - singleCoreMTail;
            if (tmpM < 0) {
                realSingleCoreM = singleCoreM;
                globalOffsetM = singleCoreM * ((int32_t)block_id / tilingData->NCoreNum);
            } else {
                realSingleCoreM = singleCoreM - 1;
                globalOffsetM = singleCoreM * ((int32_t)block_id / tilingData->NCoreNum) - tmpM;
            }
            baseM = (realSingleCoreM + tilingData->baseMNum - 1) / tilingData->baseMNum;
            baseMNum = tilingData->baseMNum;
            baseMTail = (realSingleCoreM - 1) % tilingData->baseMNum + 1;
            for (int32_t i = 1; i < NUMBER_16; i++) {
                NDLocalList[i] = NDLocalList[i - 1] + processXKBaseN * sizeof(half);
            }
            ubYFloat = ubYInt32.ReinterpretCast<float>();
            ubYHalfNZ = ubYInt32.ReinterpretCast<half>();
            ubYHalfND = ubYHalfNZ[baseM * baseN * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<half>();
            ubWScale = ubYHalfND[baseM * baseN * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<float>();
            fracM_ = fracM;
        }
    }
    __aicore__ inline void ChooseTiling1or2()
    {
        if (realM > TILING2_THRESHOLD && isTiling1) {
            isTiling1 = false;
            int32_t tmpN = ((int32_t)block_id % tilingData->NCoreNum) - tilingData->singleCoreNTail;
            if (tmpN < 0) {
                realSingleCoreN = tilingData->singleCoreN;
                globalOffsetN = tilingData->singleCoreN * ((int32_t)block_id % tilingData->NCoreNum);
            } else {
                realSingleCoreN = tilingData->singleCoreN - 1;
                globalOffsetN = tilingData->singleCoreN * ((int32_t)block_id % tilingData->NCoreNum) - tmpN;
            }
            baseNNum = tilingData->baseNNum_2;
            baseKNum = tilingData->baseKNum_2;
            baseK = tilingData->baseK_2;
            baseKTail = tilingData->baseKTail_2;
            baseN = (realSingleCoreN + baseNNum - 1) / baseNNum;
            baseNTail = (realSingleCoreN - 1) % baseNNum + 1;
        } else if (realM <= TILING2_THRESHOLD && !isTiling1) {
            isTiling1 = false;
            int32_t tmpN = ((int32_t)block_id % tilingData->NCoreNum) - tilingData->singleCoreNTail;
            if (tmpN < 0) {
                realSingleCoreN = tilingData->singleCoreN;
                globalOffsetN = tilingData->singleCoreN * ((int32_t)block_id % tilingData->NCoreNum);
            } else {
                realSingleCoreN = tilingData->singleCoreN - 1;
                globalOffsetN = tilingData->singleCoreN * ((int32_t)block_id % tilingData->NCoreNum) - tmpN;
            }
            baseNNum = tilingData->baseNNum;
            baseKNum = tilingData->baseKNum;
            baseK = tilingData->baseK;
            baseKTail = tilingData->baseKTail;
            baseN = (realSingleCoreN + baseNNum - 1) / baseNNum;
            baseNTail = (realSingleCoreN - 1) % baseNNum + 1;
        }
    }
    __aicore__ inline void InitSyncWsSwift()
    {
        InitSyncWs();
    }
    __aicore__ inline void MySyncAllSwift()
    {
        MySyncAll();
    }
    GlobalTensor<int64_t> groupGM;
    // swift
    bool isSwift = false;
    bool unTiling = true;
    bool isTiling1;
    uint32_t fracM_;
    uint32_t gemv_threshold;
};
} // namespace qgmmd_staged
} // namespace AscendC
#endif
