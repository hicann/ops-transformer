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
 * \file quant_grouped_matmul_dequant_base.h
 * \brief
 */
#ifndef _ASCENDC_QUANT_GROUPED_MATMUL_DEQUANT_BASE_H_
#define _ASCENDC_QUANT_GROUPED_MATMUL_DEQUANT_BASE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

namespace AscendC {
constexpr uint32_t UB_SIZE = 256 * 1024;
constexpr uint32_t L1_SIZE = 1024 * 1024;
constexpr uint32_t L0A_SIZE = 64 * 1024;
constexpr uint32_t L0B_SIZE = 64 * 1024;
constexpr uint32_t L0C_SIZE = 256 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SPECIAL_INIT_VAL = 0;

constexpr uint32_t INT8_PERBLOCK = 32;
constexpr uint32_t HALF_PERBLOCK = 16;
constexpr uint32_t FLOAT_PERBLOCK = 8;
constexpr uint32_t INT64_PERBLOCK = 4;
constexpr uint32_t HALF_PERREPEAT = 128;
constexpr uint32_t K_FRACTAL_INT8 = 32;
constexpr uint32_t NM_FRACTAL_INT8 = 16;
constexpr uint32_t L0C_FRACTAL = 16;
constexpr uint32_t L0_ADDR_ALIGN = 512;

constexpr uint32_t NUMBER_2 = 2;
constexpr uint32_t NUMBER_3 = 3;
constexpr uint32_t NUMBER_4 = 4;
constexpr uint32_t NUMBER_16 = 16;
constexpr uint32_t NUMBER_128 = 128;
constexpr uint32_t NUMBER_256 = 256;
constexpr uint32_t NUMBER_1024 = 1024;

constexpr float FLOAT_0 = 0.0f;
constexpr float FLOAT_1 = 1.0f;
constexpr float FLOAT_127 = 127.0f;

constexpr uint32_t GEMV_THRESHOLD = 8;

class QuantMatmulDequantBase {
public:
    TPipe pipe;
    __aicore__ inline QuantMatmulDequantBase() {}

protected:
    __aicore__ inline void InitSyncWs()
    {
        Duplicate<int32_t>(syncUbWs, SPECIAL_INIT_VAL, tilingData->CoreNum * FLOAT_PERBLOCK);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
        PipeBarrier<PIPE_ALL>();
        DataCopy<int32_t>(syncGmWs, syncUbWs, tilingData->CoreNum * FLOAT_PERBLOCK);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void MySyncAll()
    {
        AscendC::SyncAll<false>(syncGmWs, syncUbWs, tilingData->CoreNum);
    }
    __aicore__ inline void ProcessXScale()
    {
        uint32_t Mloop = realM / tilingData->CoreNum;
        uint32_t MTailCoreNum = realM % tilingData->CoreNum;
        uint32_t Moffset = (Mloop * block_id + (block_id < MTailCoreNum ? block_id : MTailCoreNum));
        Mloop += static_cast<uint32_t>(block_id < MTailCoreNum);

        DataCopyParams repeatParamsHalf;
        DataCopyParams repeatParamsFloat;
        UnaryRepeatParams unaryParamsH2F;
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;
        unaryParamsH2F.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
        repeatParamsFloat.blockLen = 1;

        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[1]);
        for (int32_t i = Moffset; i < (Moffset + Mloop); i++) {
            uint32_t pingpong = i % NUMBER_2;
            uint32_t mOffset = i * tilingData->originK;
            uint32_t kOffset = 0;
            for (int32_t j = 0; j < tilingData->dynamicIterK; j++) {
                uint32_t realBaseK =
                    (j == (tilingData->dynamicIterK - 1)) ? tilingData->dynamicBaseKTail : tilingData->dynamicBaseK;
                uint32_t realBaseKAligned256 = (realBaseK + NUMBER_256 - 1) / NUMBER_256 * NUMBER_256;
                repeatParamsHalf.blockLen = realBaseK / HALF_PERBLOCK;

                WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[pingpong]);
                if (tilingData->smoothScale && (i == Moffset || tilingData->dynamicIterK != 1)) {
                    DataCopy<half>(dynamicSmoothScale[pingpong], smoothScaleGm[kOffset], repeatParamsHalf);
                }
                DataCopy<half>(dynamicX[pingpong], xGm[kOffset + mOffset], repeatParamsHalf);
                SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
                SetMaskCount();
                SetVectorMask<half, MaskMode::COUNTER>(realBaseKAligned256);
                if (tilingData->smoothScale) {
                    Mul<half, false>(dynamicX[pingpong], dynamicX[pingpong], dynamicSmoothScale[pingpong],
                                     MASK_PLACEHOLDER, 1, binaryParams);
                    PipeBarrier<PIPE_V>();
                }
                Abs<half, false>(dynamicX[pingpong], dynamicX[pingpong], MASK_PLACEHOLDER, 1, unaryParams);

                if (realBaseK != realBaseKAligned256) {
                    PipeBarrier<PIPE_V>();
                    SetVectorMask<half, MaskMode::COUNTER>(realBaseKAligned256 - realBaseK);
                    Duplicate<half, false>(dynamicX[pingpong][realBaseK], FLOAT_0, MASK_PLACEHOLDER, 1,
                                           DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
                }

                PipeBarrier<PIPE_V>();
                SetVectorMask<half, MaskMode::COUNTER>(realBaseKAligned256);
                BlockReduceMax<half, false>(dynamicX[pingpong], dynamicX[pingpong], 1, MASK_PLACEHOLDER, 1, 1,
                                            HALF_PERBLOCK / NUMBER_2);

                uint32_t dynamicKTmp = realBaseKAligned256 / HALF_PERBLOCK;
                if (dynamicKTmp > HALF_PERREPEAT) {
                    SetVectorMask<half, MaskMode::COUNTER>(dynamicKTmp);
                    PipeBarrier<PIPE_V>();
                    BlockReduceMax<half, false>(dynamicX[pingpong], dynamicX[pingpong], 1, MASK_PLACEHOLDER, 1, 1,
                                                HALF_PERBLOCK / NUMBER_2);
                    dynamicKTmp /= HALF_PERBLOCK;
                }

                PipeBarrier<PIPE_V>();
                SetMaskNorm();
                WholeReduceMax<half>(dynamicScale[pingpong][HALF_PERBLOCK], dynamicX[pingpong], dynamicKTmp, 1, 1, 1, 0,
                                     ReduceOrder::ORDER_VALUE_INDEX);
                SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[pingpong]);
                SetMaskCount();
                SetVectorMask<half, MaskMode::COUNTER>(HALF_PERBLOCK);
                if (j == 0) {
                    PipeBarrier<PIPE_V>();
                    Duplicate<half, false>(dynamicScale[pingpong], FLOAT_0, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                                           DEFAULT_REPEAT_STRIDE);
                }

                PipeBarrier<PIPE_V>();
                Max<half, false>(dynamicScale[pingpong], dynamicScale[pingpong], dynamicScale[pingpong][HALF_PERBLOCK],
                                 MASK_PLACEHOLDER, 1, binaryParams);

                kOffset += realBaseK;
            }

            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[pingpong]);
            Cast<float, half, false>(dynamicScaleFloat[pingpong], dynamicScale[pingpong], RoundMode::CAST_NONE,
                                     MASK_PLACEHOLDER, 1, unaryParamsH2F);

            SetVectorMask<half, MaskMode::COUNTER>(FLOAT_PERBLOCK);
            PipeBarrier<PIPE_V>();
            Duplicate<float, false>(dynamicScaleFloat[pingpong][FLOAT_PERBLOCK], FLOAT_127, MASK_PLACEHOLDER, 1,
                                    DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);

            PipeBarrier<PIPE_V>();
            Div<float, false>(dynamicScaleFloat[pingpong], dynamicScaleFloat[pingpong],
                              dynamicScaleFloat[pingpong][FLOAT_PERBLOCK], MASK_PLACEHOLDER, 1, binaryParams);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);

            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
            DataCopy<float>(xScaleGm[i * FLOAT_PERBLOCK], dynamicScaleFloat[pingpong], repeatParamsFloat);
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[pingpong]);
        }
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[1]);
    }
    __aicore__ inline void InitCommonGlobalTensors(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale,
                                                   GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset,
                                                   GM_ADDR smooth_scale, GM_ADDR y, GM_ADDR usrWorkspace)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), tilingData->originM * tilingData->originK);
        quantizedWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(quantized_weight),
                                          tilingData->fracK * tilingData->fracN * K_FRACTAL_INT8 * NM_FRACTAL_INT8);
        if (tilingData->smoothScale) {
            smoothScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(smooth_scale), tilingData->originK);
        }
        isWScaleInt64 = (ORIG_DTYPE_WEIGHT_SCALE == DT_INT64);
        if (isWScaleInt64) {
            wScaleGmInt64.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(weight_scale), tilingData->originN);
        } else {
            wScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_scale), tilingData->originN);
        }
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), tilingData->originM * tilingData->originN);

        if (tilingData->dynamicQuant) {
            xScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(usrWorkspace),
                                     tilingData->originM * FLOAT_PERBLOCK);
            usrWorkspace += tilingData->originM * BLOCK_SIZE;
        } else {
            if (tilingData->perToken) {
                xScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x_scale), tilingData->originM);
            } else {
                isXScaleHalf = (tilingData->isXScaleHalf == 1);
                if (isXScaleHalf) {
                    xScaleGmHalf.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x_scale), 1);
                    x_scale_dequant = static_cast<float>(xScaleGmHalf.GetValue(0));
                    x_scale_quant = FLOAT_1 / x_scale_dequant;
                } else {
                    xScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x_scale), 1);
                    x_scale_dequant = xScaleGm.GetValue(0);
                    x_scale_quant = x_scale_dequant == 0 ? FLOAT_1 : FLOAT_1 / x_scale_dequant;
                }
            }
        }
        syncGmWs.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(usrWorkspace),
                                 tilingData->CoreNum * FLOAT_PERBLOCK);
        syncGmWsSelf.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(usrWorkspace + block_id * BLOCK_SIZE),
                                     FLOAT_PERBLOCK);
    }
    __aicore__ inline void InitCommonLocalTensors(LocalTensor<uint8_t> &ub)
    {
        syncUbWs = ub.ReinterpretCast<int32_t>();
        syncUbWsSelf = syncUbWs[block_id * FLOAT_PERBLOCK];

        dynamicX[0] = ub.ReinterpretCast<half>();
        dynamicX[1] = dynamicX[0][tilingData->dynamicBaseK];
        if (tilingData->smoothScale) {
            dynamicSmoothScale[0] = dynamicX[1][tilingData->dynamicBaseK];
            if (tilingData->dynamicIterK != 1) {
                dynamicSmoothScale[1] = dynamicSmoothScale[0][tilingData->dynamicBaseK];
            } else {
                dynamicSmoothScale[1] = dynamicSmoothScale[0];
            }
            dynamicScale[0] = dynamicSmoothScale[1][tilingData->dynamicBaseK];
        } else {
            dynamicScale[0] = dynamicX[1][tilingData->dynamicBaseK];
        }
        dynamicScale[1] = dynamicScale[0][NUMBER_2 * HALF_PERBLOCK];
        dynamicScaleFloat[0] = dynamicScale[1][NUMBER_2 * HALF_PERBLOCK].ReinterpretCast<float>();
        dynamicScaleFloat[1] = dynamicScaleFloat[0][NUMBER_2 * FLOAT_PERBLOCK];
    }
    __aicore__ inline void InitEventId()
    {
        eventIdMTE2ToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        eventIdMTE2ToS[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
        eventIdVToMTE3[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        eventIdMTE3ToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        eventIdMTE3ToV[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        eventIdMTE3ToMTE1[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE1>());
        eventIdMTE3ToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        eventIdMTE2ToMTE1[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        eventIdMTE2ToMTE1[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        eventIdMTE2ToMTE1[NUMBER_2] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        eventIdMTE2ToMTE1[NUMBER_3] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        eventIdMTE1ToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdMTE1ToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdMTE1ToMTE2[NUMBER_2] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdMTE1ToMTE2[NUMBER_3] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdMTE1ToM[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        eventIdMTE1ToM[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        eventIdMTE1ToM[NUMBER_2] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        eventIdMToMTE1[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        eventIdMToMTE1[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        eventIdMToMTE1[NUMBER_2] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        eventIdMToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_V>());
        eventIdVToM[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_M>());
        eventIdSToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE2>());
        eventIdSToMTE3[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE3>());
    }
    __aicore__ inline void WeightScaleCopyInAndCast(LocalTensor<float> ubWScale, uint64_t gmOffset,
                                                    DataCopyParams repeatParams, uint64_t dataCount)
    {
        if (!isWScaleInt64) {
            repeatParams.blockLen = dataCount / FLOAT_PERBLOCK;
            DataCopy<float>(ubWScale, wScaleGm[gmOffset], repeatParams);

            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
            return;
        }

        // 输入的weightScale类型为int64, 需要适配
        LocalTensor<int64_t> ubWScaleInt64 = ubWScale.ReinterpretCast<int64_t>();
        LocalTensor<int32_t> ubWScaleInt32 = ubWScale.ReinterpretCast<int32_t>();

        repeatParams.blockLen = dataCount / INT64_PERBLOCK;
        DataCopy<int64_t>(ubWScaleInt64, wScaleGmInt64[gmOffset], repeatParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

        GatherMaskParams gatherMaskParams;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.repeatTimes = (dataCount * sizeof(int64_t) + NUMBER_256 - 1) / NUMBER_256;
        gatherMaskParams.src0RepeatStride = NUMBER_256 / BLOCK_SIZE;
        gatherMaskParams.src1RepeatStride = 0;
        AscendC::GatherMask(ubWScaleInt32, ubWScaleInt32, 1, false, 0, gatherMaskParams, dataCount);

        PipeBarrier<PIPE_V>();
    }
    // GlobalTensor
    GlobalTensor<half> xGm;
    GlobalTensor<int8_t> xZNGM;
    GlobalTensor<int8_t> quantizedWeightGm;
    GlobalTensor<float> xScaleGm;
    GlobalTensor<half> xScaleGmHalf;
    GlobalTensor<float> wScaleGm;
    GlobalTensor<int64_t> wScaleGmInt64;
    GlobalTensor<half> smoothScaleGm;
    GlobalTensor<half> yGm;

    GlobalTensor<int32_t> syncGmWs;
    GlobalTensor<int32_t> syncGmWsSelf;
    // LocalTensor
    TBuf<TPosition::VECCALC> UbBuf;
    TBuf<TPosition::TSCM> L1Buf;
    TBuf<TPosition::A2> L0ABuf;
    TBuf<TPosition::B2> L0BBuf;
    TBuf<TPosition::CO1> L0CBuf;

    LocalTensor<int32_t> syncUbWs;
    LocalTensor<int32_t> syncUbWsSelf;

    LocalTensor<half> dynamicX[NUMBER_2];
    LocalTensor<half> dynamicSmoothScale[NUMBER_2];
    LocalTensor<half> dynamicScale[NUMBER_2];
    LocalTensor<float> dynamicScaleFloat[NUMBER_2];

    const QuantMatmulDequantTilingData *__restrict tilingData;

    event_t eventIdMTE3ToV[NUMBER_2];
    event_t eventIdMTE2ToV[1];
    event_t eventIdMToV[1];
    event_t eventIdVToM[1];
    event_t eventIdVToMTE3[1];
    event_t eventIdVToMTE2[NUMBER_2];
    event_t eventIdMTE3ToMTE1[1];
    event_t eventIdMTE3ToMTE2[1];
    event_t eventIdMTE1ToMTE2[NUMBER_4];
    event_t eventIdMTE2ToMTE1[NUMBER_4];
    event_t eventIdMTE1ToM[NUMBER_3];
    event_t eventIdMToMTE1[NUMBER_3];
    event_t eventIdMTE2ToS[1];
    event_t eventIdSToMTE2[1];
    event_t eventIdSToMTE3[1];

    uint32_t block_id;
    uint32_t realM;
    uint32_t fracM;
    uint32_t tailM;
    float x_scale_dequant;
    float x_scale_quant;

    bool isXScaleHalf = false;
    bool isWScaleInt64 = false;

    uint32_t curValue_ = 0;
};
} // namespace AscendC
#endif
