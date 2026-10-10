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
 * \file grouped_matmul_quant_w4a16.cpp
 * \brief
 */

#include "grouped_matmul_quant_w4a16.h"
#include <cmath>

using namespace AscendC;

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::Init(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale,
                                                        GM_ADDR weight_offset, GM_ADDR group_list, GM_ADDR y,
                                                        GM_ADDR usrWorkspace,
                                                        const GroupedMatmulQuantTilingData* __restrict gmmTiling)
{
    block_id = get_block_idx();
    subblock_id = get_subblockid();
    tilingData = gmmTiling;

    InitEventId();

    InitGlobalTensors(x, quantized_weight, weight_scale, weight_offset, group_list, y);

    InitWorkspaceTensors(usrWorkspace);

    InitLocalTensors();

    TilingInKernel();
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::InitEventId()
{
    if ASCEND_IS_AIV {
        eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        eventIdMTE3ToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        eventIdMTE3ToV[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMTE2[2] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        eventIdMTE2ToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        eventIdMTE2ToV[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        eventIdMTE2ToV[2] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    } else {
        eventIdMTE1ToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdMTE1ToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdMTE2ToMTE1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        eventIdMTE1ToM = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        eventIdMToMTE1[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        eventIdMToMTE1[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        eventIdMToFIX = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_FIX>());
        eventIdFIXToM = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::FIX_M>());
    }
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::InitWorkspaceTensors(GM_ADDR usrWorkspace)
{
    uint32_t workspaceBaseN = tilingData->L0ASize / 2 / sizeof(T);
    workspaceWGm[0][0].SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(usrWorkspace + tilingData->L0ASize / 2 * block_id * 2 * MY_BUFFER_NUM),
        workspaceBaseN);
    workspaceWGm[1][0].SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(usrWorkspace + tilingData->L0ASize / 2 * (block_id * 2 + 1) * MY_BUFFER_NUM),
        workspaceBaseN);
    for (int32_t i = 1; i < MY_BUFFER_NUM; i++) {
        workspaceWGm[0][i] = workspaceWGm[0][i - 1][workspaceBaseN];
        workspaceWGm[1][i] = workspaceWGm[1][i - 1][workspaceBaseN];
    }
    workspaceCGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(usrWorkspace + tilingData->L0ASize * tilingData->CoreNum * MY_BUFFER_NUM),
        tilingData->originM * tilingData->originN);
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::InitGlobalTensors(GM_ADDR x, GM_ADDR quantized_weight,
                                                                     GM_ADDR weight_scale, GM_ADDR weight_offset,
                                                                     GM_ADDR group_list, GM_ADDR y)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), tilingData->originM * tilingData->originK);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), tilingData->originM * tilingData->originN);
    wGm0.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(quantized_weight),
                         tilingData->originE * tilingData->originK * tilingData->originN / 8);
    wScaleGm0.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight_scale),
                              tilingData->originE * tilingData->scaleK * tilingData->originN);
    wOffsetGm0.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight_offset),
                               tilingData->originE * tilingData->scaleK * tilingData->originN);
    groupListGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list), tilingData->originE);
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::InitLocalTensors()
{
    if ASCEND_IS_AIV {
        int32_t baseNumber = tilingData->L0ASize / 2 / sizeof(bfloat16_t);
        int32_t baseNumberCast = tilingData->UBSize / 6 / sizeof(bfloat16_t);
        pipe.InitBuffer(UbBuf, tilingData->UBSize);
        clearUb = UbBuf.Get<float>();
        castUbFloat[0] = clearUb.ReinterpretCast<float>();
        castUbFloat[1] = castUbFloat[0][baseNumberCast];
        castUbBF16[0] = castUbFloat[1][baseNumberCast].ReinterpretCast<T>();
        castUbBF16[1] = castUbBF16[0][baseNumberCast];
        ubWInt4 = clearUb.ReinterpretCast<int4b_t>();
        ubWHalf = ubWInt4[baseNumber].ReinterpretCast<half>();
        ubWFloat = ubWHalf[baseNumber].ReinterpretCast<float>();
        ubWOffsetHalf = ubWFloat[baseNumber].ReinterpretCast<half>();
        ubWScaleFloat = ubWOffsetHalf[baseNumber / 16].ReinterpretCast<float>();
        ubWScaleT = ubWScaleFloat[baseNumber / 16].ReinterpretCast<T>();
        ubWT = ubWScaleT[baseNumber / 16];
    } else {
        int32_t baseNumber = tilingData->L0ASize / 2 / sizeof(bfloat16_t);
        pipe.InitBuffer(L1Buf, tilingData->L1Size);
        L1MatB[0] = L1Buf.Get<T>();
        L1MatB[1] = L1MatB[0][baseNumber];
        L1MatA[0] = L1MatB[1][baseNumber];
        L1MatA[1] = L1MatA[0][baseNumber];
        pipe.InitBuffer(L0ABuf, tilingData->L0ASize);
        L0MatA[0] = L0ABuf.Get<T>();
        L0MatA[1] = L0MatA[0][baseNumber];
        pipe.InitBuffer(L0BBuf, tilingData->L0BSize);
        L0MatB[0] = L0BBuf.Get<T>();
        L0MatB[1] = L0MatB[0][baseNumber];
        pipe.InitBuffer(L0CBuf, tilingData->L0CSize);
        L0MatC = L0CBuf.Get<float>();
    }
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::TilingInKernel()
{
    int32_t startM = 0, endM;
    int32_t totalTaskNumMNK = 0;
    int32_t totalTaskNumMN = 0;
    // 64 * 1024 / (16 * 16 * 2) / 2 = 64
    // 128 * 1024 / (16 * 16 * 4) = 128
    // baseN * baseM <= 128
    baseKNum = (tilingData->fracK + 4 - 1) / 4;
    baseNNum = (tilingData->fracN + 16 - 1) / 16;
    if (tilingData->noGroup) {
        int32_t fracM = (tilingData->originM + 16 - 1) / 16;
        baseMNum[0] = (fracM + 8 - 1) / 8;
        realM[0] = tilingData->originM;
        totalTaskNumMNK = (baseMNum[0] * baseKNum * baseNNum);
        totalTaskNumMN = (baseMNum[0] * baseNNum);
    } else {
        for (int32_t i = 0; i < tilingData->originE; i++) {
            endM = groupListGm.GetValue(i);
            if (endM > tilingData->originM)
                endM = tilingData->originM;
            if (endM <= startM) {
                realM[i] = 0;
                continue;
            }
            int32_t realM_ = endM - startM;
            int32_t fracM = (realM_ + 16 - 1) / 16;
            baseMNum[i] = (fracM + 8 - 1) / 8;

            realM[i] = realM_;
            totalTaskNumMNK += (baseMNum[i] * baseKNum * baseNNum);
            totalTaskNumMN += (baseMNum[i] * baseNNum);
            startM = endM;
        }
    }

    splitK = (tilingData->splitK != 0);
    if (splitK) {
        taskNum = totalTaskNumMNK / tilingData->CoreNum + (block_id < (totalTaskNumMNK % tilingData->CoreNum) ? 1 : 0);
        taskOffset =
            totalTaskNumMNK / tilingData->CoreNum * block_id +
            (block_id < (totalTaskNumMNK % tilingData->CoreNum) ? block_id : (totalTaskNumMNK % tilingData->CoreNum));
    } else {
        taskNum = totalTaskNumMN / tilingData->CoreNum + (block_id < (totalTaskNumMN % tilingData->CoreNum) ? 1 : 0);
        taskOffset =
            totalTaskNumMN / tilingData->CoreNum * block_id +
            (block_id < (totalTaskNumMN % tilingData->CoreNum) ? block_id : (totalTaskNumMN % tilingData->CoreNum));
        taskNum *= baseKNum;
        taskOffset *= baseKNum;
    }
}

template <>
__aicore__ inline void GroupedMatmulQuantW4A16<half>::AntiQuantWeight(uint32_t task_id, uint32_t ping_pong,
                                                                      bool endKFlag, uint32_t baseN, uint32_t baseK,
                                                                      uint32_t offsetN, uint32_t offsetK)
{
    uint32_t scaleBegin = offsetK * 16 / tilingData->scaleGroupSize;
    uint32_t scaleEnd = ((offsetK + baseK) * 16 - 1) / tilingData->scaleGroupSize;

    UnaryRepeatParams unaryParams16To32, unaryParams32To16, unaryParams4To16;
    unaryParams16To32.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams32To16.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams4To16.srcRepStride = ONE_FOURTH_DEFAULT_REPEAT_STRIDE;
    DataCopyParams repeatParamsInt4Weight;
    repeatParamsInt4Weight.blockCount = baseK;
    repeatParamsInt4Weight.blockLen = baseN * 4;
    repeatParamsInt4Weight.srcStride = (tilingData->fracN - baseN) * 4;
    repeatParamsInt4Weight.dstStride = 0;
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    DataCopy<int32_t>(ubWInt4.ReinterpretCast<int32_t>(), wGm[((offsetK * tilingData->fracN) + offsetN) * 16 * 2],
                      repeatParamsInt4Weight);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    DataCopyParams repeatParamsScale;
    repeatParamsScale.blockCount = (-scaleBegin + scaleEnd + 1);
    repeatParamsScale.blockLen = baseN;
    repeatParamsScale.srcStride = (tilingData->fracN - baseN);
    repeatParamsScale.dstStride = 0;
    DataCopy<half>(ubWOffsetHalf, wOffsetGm[scaleBegin * tilingData->originN + offsetN * 16], repeatParamsScale);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
    DataCopy<half>(ubWScaleT, wScaleGm[scaleBegin * tilingData->originN + offsetN * 16], repeatParamsScale);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[2]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(baseN * baseK * 16 * 16);
    Cast<half, int4b_t, false>(ubWHalf, ubWInt4, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams4To16);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();

    BinaryRepeatParams binaryParams;
    binaryParams.dstBlkStride = 1;
    binaryParams.src0BlkStride = 1;
    binaryParams.src1BlkStride = 0;
    binaryParams.dstRepStride = 16;
    binaryParams.src0RepStride = 16;
    binaryParams.src1RepStride = 1;
    uint32_t mulOffsetA = 0;
    uint32_t mulOffsetB = 0;
    for (int32_t i = 0; i < baseK; i++) {
        Add<half, false>(ubWHalf[mulOffsetB], ubWHalf[mulOffsetB], ubWOffsetHalf[mulOffsetA], MASK_PLACEHOLDER, baseN,
                         binaryParams);
        Add<half, false>(ubWHalf[mulOffsetB + 16 * 8], ubWHalf[mulOffsetB + 16 * 8], ubWOffsetHalf[mulOffsetA],
                         MASK_PLACEHOLDER, baseN, binaryParams);
        bool tmp = ((offsetK + i + 1) * 16) % tilingData->scaleGroupSize == 0;
        mulOffsetA += baseN * 16 * (tmp ? 1 : 0);
        mulOffsetB += 16 * 16 * baseN;
    }
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);

    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[2]);
    PipeBarrier<PIPE_V>();
    mulOffsetA = 0;
    mulOffsetB = 0;
    for (int32_t i = 0; i < baseK; i++) {
        Mul<half, false>(ubWT[mulOffsetB], ubWHalf[mulOffsetB], ubWScaleT[mulOffsetA], MASK_PLACEHOLDER, baseN,
                         binaryParams);
        Mul<half, false>(ubWT[mulOffsetB + 16 * 8], ubWHalf[mulOffsetB + 16 * 8], ubWScaleT[mulOffsetA],
                         MASK_PLACEHOLDER, baseN, binaryParams);
        bool tmp = ((offsetK + i + 1) * 16) % tilingData->scaleGroupSize == 0;
        mulOffsetA += baseN * 16 * (tmp ? 1 : 0);
        mulOffsetB += 16 * 16 * baseN;
    }
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    if ((task_id / 2) == 0)
        wait_flag_dev(ping_pong ? SYNC_PING_AIC_AIV_FLAG : SYNC_PONG_AIC_AIV_FLAG);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    DataCopyParams repeatParamsBF16Weight;
    repeatParamsBF16Weight.blockLen = baseK * baseN * 16;
    DataCopy<half>(workspaceWGm[ping_pong][task_id], ubWT, repeatParamsBF16Weight);

    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    if (((task_id / 2 + 1) == MY_BUFFER_NUM / 2) || endKFlag)
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, ping_pong ? SYNC_PING_AIV_AIC_FLAG : SYNC_PONG_AIV_AIC_FLAG));
}

template <>
__aicore__ inline void GroupedMatmulQuantW4A16<bfloat16_t>::AntiQuantWeight(uint32_t task_id, uint32_t ping_pong,
                                                                            bool endKFlag, uint32_t baseN,
                                                                            uint32_t baseK, uint32_t offsetN,
                                                                            uint32_t offsetK)
{
    uint32_t scaleBegin = offsetK * 16 / tilingData->scaleGroupSize;
    uint32_t scaleEnd = ((offsetK + baseK) * 16 - 1) / tilingData->scaleGroupSize;

    UnaryRepeatParams unaryParams16To32, unaryParams32To16, unaryParams4To16;
    unaryParams16To32.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams32To16.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams4To16.srcRepStride = ONE_FOURTH_DEFAULT_REPEAT_STRIDE;
    DataCopyParams repeatParamsInt4Weight;
    repeatParamsInt4Weight.blockCount = baseK;
    repeatParamsInt4Weight.blockLen = baseN * 4;
    repeatParamsInt4Weight.srcStride = (tilingData->fracN - baseN) * 4;
    repeatParamsInt4Weight.dstStride = 0;
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    DataCopy<int32_t>(ubWInt4.ReinterpretCast<int32_t>(), wGm[((offsetK * tilingData->fracN) + offsetN) * 16 * 2],
                      repeatParamsInt4Weight);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    DataCopyParams repeatParamsScale;
    repeatParamsScale.blockCount = (-scaleBegin + scaleEnd + 1);
    repeatParamsScale.blockLen = baseN;
    repeatParamsScale.srcStride = (tilingData->fracN - baseN);
    repeatParamsScale.dstStride = 0;
    DataCopy<half>(ubWOffsetHalf, wOffsetGm[scaleBegin * tilingData->originN + offsetN * 16], repeatParamsScale);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
    DataCopy<bfloat16_t>(ubWScaleT, wScaleGm[scaleBegin * tilingData->originN + offsetN * 16], repeatParamsScale);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[2]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(baseN * baseK * 16 * 16);
    Cast<half, int4b_t, false>(ubWHalf, ubWInt4, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams4To16);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();

    BinaryRepeatParams binaryParams;
    binaryParams.dstBlkStride = 1;
    binaryParams.src0BlkStride = 1;
    binaryParams.src1BlkStride = 0;
    binaryParams.dstRepStride = 16;
    binaryParams.src0RepStride = 16;
    binaryParams.src1RepStride = 1;
    uint32_t mulOffsetA = 0;
    uint32_t mulOffsetB = 0;
    for (int32_t i = 0; i < baseK; i++) {
        Add<half, false>(ubWHalf[mulOffsetB], ubWHalf[mulOffsetB], ubWOffsetHalf[mulOffsetA], MASK_PLACEHOLDER, baseN,
                         binaryParams);
        Add<half, false>(ubWHalf[mulOffsetB + 16 * 8], ubWHalf[mulOffsetB + 16 * 8], ubWOffsetHalf[mulOffsetA],
                         MASK_PLACEHOLDER, baseN, binaryParams);
        bool tmp = ((offsetK + i + 1) * 16) % tilingData->scaleGroupSize == 0;
        mulOffsetA += baseN * 16 * (tmp ? 1 : 0);
        mulOffsetB += 16 * 16 * baseN;
    }
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);

    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(baseN * baseK * 16 * 16);
    Cast<float, half, false>(ubWFloat, ubWHalf, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams16To32);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[2]);
    SetVectorMask<half, MaskMode::COUNTER>((-scaleBegin + scaleEnd + 1) * baseN * 16);
    Cast<float, bfloat16_t, false>(ubWScaleFloat, ubWScaleT, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                                   unaryParams16To32);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);

    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    binaryParams.dstBlkStride = 2;
    binaryParams.src0BlkStride = 2;
    binaryParams.src1BlkStride = 0;
    binaryParams.dstRepStride = 32;
    binaryParams.src0RepStride = 32;
    binaryParams.src1RepStride = 2;
    mulOffsetA = 0;
    mulOffsetB = 0;
    for (int32_t i = 0; i < baseK; i++) {
        Mul<float, false>(ubWFloat[mulOffsetB], ubWFloat[mulOffsetB], ubWScaleFloat[mulOffsetA], MASK_PLACEHOLDER,
                          baseN, binaryParams);
        Mul<float, false>(ubWFloat[mulOffsetB + 16 * 8], ubWFloat[mulOffsetB + 16 * 8], ubWScaleFloat[mulOffsetA],
                          MASK_PLACEHOLDER, baseN, binaryParams);
        Mul<float, false>(ubWFloat[mulOffsetB + 8], ubWFloat[mulOffsetB + 8], ubWScaleFloat[mulOffsetA + 8],
                          MASK_PLACEHOLDER, baseN, binaryParams);
        Mul<float, false>(ubWFloat[mulOffsetB + 16 * 8 + 8], ubWFloat[mulOffsetB + 16 * 8 + 8],
                          ubWScaleFloat[mulOffsetA + 8], MASK_PLACEHOLDER, baseN, binaryParams);
        bool tmp = ((offsetK + i + 1) * 16) % tilingData->scaleGroupSize == 0;
        mulOffsetA += baseN * 16 * (tmp ? 1 : 0);
        mulOffsetB += 16 * 16 * baseN;
    }

    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(baseN * baseK * 16 * 16);
    Cast<bfloat16_t, float, false>(ubWT, ubWFloat, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, unaryParams32To16);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    SetMaskNorm();
    ResetMask();

    if ((task_id / 2) == 0)
        wait_flag_dev(ping_pong ? SYNC_PING_AIC_AIV_FLAG : SYNC_PONG_AIC_AIV_FLAG);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    DataCopyParams repeatParamsBF16Weight;
    repeatParamsBF16Weight.blockLen = baseK * baseN * 16;
    DataCopy<bfloat16_t>(workspaceWGm[ping_pong][task_id], ubWT, repeatParamsBF16Weight);

    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    if (((task_id / 2 + 1) == MY_BUFFER_NUM / 2) || endKFlag)
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, ping_pong ? SYNC_PING_AIV_AIC_FLAG : SYNC_PONG_AIV_AIC_FLAG));
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::ComputeMatMul(uint32_t task_id, uint32_t ping_pong, bool startKFlag,
                                                                 bool endKFlag, uint32_t realM, uint32_t baseM,
                                                                 uint32_t baseN, uint32_t baseK, uint32_t offsetM,
                                                                 uint32_t offsetN, uint32_t offsetK)
{
    MmadParams mmadParams;
    LoadData2dParams loadDataA, loadDataB;
    DataCopyParams repeatParamsB;
    Nd2NzParams repeatParamsA;
    uint32_t pingpong = task_id % 2;
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[pingpong]);
    repeatParamsA.ndNum = 1;
    repeatParamsA.nValue = ((baseM + offsetM) * 16) < realM ? (baseM * 16) : (realM - offsetM * 16);
    repeatParamsA.dValue = baseK * 16;
    repeatParamsA.srcNdMatrixStride = 0;
    repeatParamsA.srcDValue = tilingData->originK;
    repeatParamsA.dstNzC0Stride = baseM * 16;
    repeatParamsA.dstNzNStride = 1;
    repeatParamsA.dstNzMatrixStride = 1;
    DataCopy<T>(L1MatA[pingpong], xGm[offsetM * 16 * tilingData->originK + offsetK * 16], repeatParamsA);

    if (task_id == 0)
        wait_flag_dev(ping_pong ? SYNC_PING_AIV_AIC_FLAG : SYNC_PONG_AIV_AIC_FLAG);
    repeatParamsB.blockLen = baseK * baseN * 16;
    DataCopy<T>(L1MatB[pingpong], workspaceWGm[ping_pong][task_id], repeatParamsB);
    SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1);
    if ((task_id + 1) == MY_BUFFER_NUM)
        ffts_cross_core_sync(PIPE_MTE2, GetffstMsg(0x02, ping_pong ? SYNC_PING_AIC_AIV_FLAG : SYNC_PONG_AIC_AIV_FLAG));

    WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[pingpong]);
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1);
    loadDataA.srcStride = baseM;
    loadDataA.ifTranspose = false;
    loadDataA.repeatTimes = baseK;
    for (int32_t i = 0; i < baseM; i++) {
        LoadData<T>(L0MatA[pingpong][i * baseK * 16 * 16], L1MatA[pingpong][i * 16 * 16], loadDataA);
    }

    loadDataB.srcStride = 1;
    loadDataB.ifTranspose = true;
    loadDataB.repeatTimes = baseK * baseN;
    LoadData<T>(L0MatB[pingpong], L1MatB[pingpong], loadDataB);
    SetFlag<HardEvent::MTE1_M>(eventIdMTE1ToM);
    SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[pingpong]);

    WaitFlag<HardEvent::MTE1_M>(eventIdMTE1ToM);
    mmadParams.m = baseM * 16;
    mmadParams.n = baseN * 16;
    mmadParams.k = baseK * 16;
    mmadParams.cmatrixInitVal = startKFlag ? true : false;
    Mmad<float, T, T>(L0MatC, L0MatA[pingpong], L0MatB[pingpong], mmadParams);
    SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[pingpong]);

    if (endKFlag) {
        SetFlag<HardEvent::M_FIX>(eventIdMToFIX);
        WaitFlag<HardEvent::M_FIX>(eventIdMToFIX);
        if (!splitK) {
            DataCopyCO12DstParams repeatParamsC;
            repeatParamsC.nSize = baseN * 16;
            repeatParamsC.mSize = ((baseM + offsetM) * 16) < realM ? (baseM * 16) : (realM - offsetM * 16);
            repeatParamsC.dstStride = tilingData->originN;
            repeatParamsC.srcStride = baseM * 16;
            if (tilingData->dataType == 1) {
                repeatParamsC.quantPre = QuantMode_t::F322F16;
            } else if (tilingData->dataType == 27) {
                repeatParamsC.quantPre = QuantMode_t::F322BF16;
            }
            repeatParamsC.nz2ndEn = true;
            SetFixpipePreQuantFlag(static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&deqScalar)));
            SetFixpipeNz2ndFlag(1, 0, 0);
            DataCopy<T, float>(yGm[offsetM * 16 * tilingData->originN + offsetN * 16], L0MatC, repeatParamsC);
        } else {
            DataCopyCO12DstParams repeatParamsC;
            repeatParamsC.nSize = baseN * 16;
            repeatParamsC.mSize = ((baseM + offsetM) * 16) < realM ? (baseM * 16) : (realM - offsetM * 16);
            repeatParamsC.dstStride = tilingData->originN;
            repeatParamsC.srcStride = baseM * 16;
            repeatParamsC.quantPre = QuantMode_t::NoQuant;
            repeatParamsC.nz2ndEn = true;
            SetFixpipeNz2ndFlag(1, 0, 0);
            SetAtomicAdd<float>();
            DataCopy<float, float>(workspaceCGm[offsetM * 16 * tilingData->originN + offsetN * 16], L0MatC,
                                   repeatParamsC);
            AscendC::SetAtomicNone();
        }

        SetFlag<HardEvent::FIX_M>(eventIdFIXToM);
        WaitFlag<HardEvent::FIX_M>(eventIdFIXToM);
    }
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::ClearOutTensor(uint32_t loop, uint32_t baseN, uint32_t tailN,
                                                                  uint32_t tailCoreNum)
{
    uint32_t block_id_ = block_id * 2 + subblock_id;
    uint32_t offset = (loop * baseN + tailN) * block_id_ + (block_id_ < tailCoreNum ? block_id_ : tailCoreNum) * 128;
    uint32_t tail = tailN + (block_id_ < tailCoreNum ? 128 : 0);

    SetMaskCount();
    if (loop > 0) {
        SetVectorMask<float, MaskMode::COUNTER>(0, baseN);
    } else if (tail > 0) {
        SetVectorMask<float, MaskMode::COUNTER>(0, tail);
    } else {
        SetMaskNorm();
        ResetMask();
        return;
    }
    Duplicate<float, false>(clearUb, static_cast<float>(0), MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                            DEFAULT_REPEAT_STRIDE);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    SetMaskNorm();
    ResetMask();

    for (int i = 0; i < loop; i++) {
        DataCopy<float>(workspaceCGm0[offset], clearUb, baseN);
        offset += baseN;
    }
    if (tail > 0) {
        DataCopy<float>(workspaceCGm0[offset], clearUb, tail);
    }

    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::CastOutTensor(uint32_t loop, uint32_t baseN, uint32_t tailN,
                                                                 uint32_t tailCoreNum)
{
    uint32_t block_id_ = block_id * 2 + subblock_id;
    uint32_t offset = (loop * baseN + tailN) * block_id_ + (block_id_ < tailCoreNum ? block_id_ : tailCoreNum) * 256;
    uint32_t tail = tailN + (block_id_ < tailCoreNum ? 256 : 0);
    uint32_t ping_pong = 0;
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, baseN);

    DataCopyParams repeatParamsFloat, repeatParamsBF16;
    UnaryRepeatParams unaryParams32To16;
    unaryParams32To16.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    repeatParamsFloat.blockLen = baseN / 8;
    repeatParamsBF16.blockLen = baseN / 16;

    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[1]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    for (int i = 0; i < loop; i++) {
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[ping_pong]);
        DataCopy<float>(castUbFloat[ping_pong], workspaceCGm0[offset], repeatParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[ping_pong]);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        Cast<T, float, false>(castUbBF16[ping_pong], castUbFloat[ping_pong], RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1,
                              unaryParams32To16);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[ping_pong]);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopy<T>(yGm0[offset], castUbBF16[ping_pong], repeatParamsBF16);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[ping_pong]);

        offset += baseN;
        ping_pong = !ping_pong;
    }
    if (tail > 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, tail);
        repeatParamsFloat.blockLen = tail / 8;
        repeatParamsBF16.blockLen = tail / 16;
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[ping_pong]);
        DataCopy<float>(castUbFloat[ping_pong], workspaceCGm0[offset], repeatParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[ping_pong]);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        Cast<T, float, false>(castUbBF16[ping_pong], castUbFloat[ping_pong], RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1,
                              unaryParams32To16);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[ping_pong]);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopy<T>(yGm0[offset], castUbBF16[ping_pong], repeatParamsBF16);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[ping_pong]);
    }
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[1]);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void GroupedMatmulQuantW4A16<T>::Process()
{
    if (splitK) {
        if ASCEND_IS_AIV {
            yGm0 = yGm;
            workspaceCGm0 = workspaceCGm;
            ClearOutTensor(tilingData->clearOutLoop, tilingData->clearBaseN, tilingData->clearOutTailN,
                           tilingData->clearOutTailCoreNum);
            ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_CLEAR_AIV_FLAG));
            wait_flag_dev(SYNC_CLEAR_AIV_FLAG);
            ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_CLEAR_AIV_AIC_FLAG));
        } else {
            wait_flag_dev(SYNC_CLEAR_AIV_AIC_FLAG);
        }
    }
    if ASCEND_IS_AIC {
        ffts_cross_core_sync(PIPE_MTE2, GetffstMsg(0x02, SYNC_PING_AIC_AIV_FLAG));
        ffts_cross_core_sync(PIPE_MTE2, GetffstMsg(0x02, SYNC_PONG_AIC_AIV_FLAG));
    }
    if ASCEND_IS_AIV {
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    } else {
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[0]);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[1]);
    }

    uint32_t ping_pong = 0;
    uint32_t baseN = tilingData->fracN / baseNNum;
    uint32_t tailN = tilingData->fracN % baseNNum;

    uint32_t baseK = tilingData->fracK / baseKNum;
    uint32_t tailK = tilingData->fracK % baseKNum;

    uint32_t task_id = 0;
    for (int32_t i = 0; i < tilingData->originE; i++) {
        if (realM[i] == 0)
            continue;

        if ((task_id + (baseMNum[i] * baseKNum * baseNNum)) <= taskOffset || task_id >= (taskOffset + taskNum)) {
            xGm = xGm[realM[i] * tilingData->originK];
            yGm = yGm[realM[i] * tilingData->originN];
            workspaceCGm = workspaceCGm[realM[i] * tilingData->originN];
            task_id += (baseMNum[i] * baseKNum * baseNNum);
            continue;
        }
        wGm = wGm0[i * tilingData->originK * tilingData->originN / 8];
        wScaleGm = wScaleGm0[i * tilingData->scaleK * tilingData->originN];
        wOffsetGm = wOffsetGm0[i * tilingData->scaleK * tilingData->originN];

        uint32_t fracM = (realM[i] + 15) / 16;
        uint32_t baseM = fracM / baseMNum[i];
        uint32_t tailM = fracM % baseMNum[i];

        uint32_t offsetM = 0;
        for (int32_t j = 0; j < baseMNum[i]; j++) {
            uint32_t realbaseM = baseM + (j < tailM ? 1 : 0);
            uint32_t offsetN = 0;
            for (int32_t k = 0; k < baseNNum; k++) {
                uint32_t realbaseN = baseN + (k < tailN ? 1 : 0);
                uint32_t offsetK = 0;
                for (int32_t l = 0; l < baseKNum; l++) {
                    uint32_t realbaseK = baseK + (l < tailK ? 1 : 0);
                    if (task_id < taskOffset || task_id >= (taskOffset + taskNum)) {
                        task_id++;
                        offsetK += realbaseK;
                        continue;
                    }
                    uint32_t inner_task_id = (task_id - taskOffset) % MY_BUFFER_NUM;
                    if ASCEND_IS_AIV {
                        if (subblock_id == (inner_task_id % 2)) {
                            bool endK =
                                (task_id == (taskOffset + taskNum - 1)) || (task_id == (taskOffset + taskNum - 2));
                            AntiQuantWeight(inner_task_id, ping_pong, endK, realbaseN, realbaseK, offsetN, offsetK);
                        } else if (inner_task_id == 0 && (task_id == (taskOffset + taskNum - 1))) {
                            ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, ping_pong ? SYNC_PING_AIV_AIC_FLAG :
                                                                                         SYNC_PONG_AIV_AIC_FLAG));
                        }
                    } else {
                        bool startK = (task_id == taskOffset) || (l == 0);
                        bool endK = (task_id == (taskOffset + taskNum - 1)) || (l == (baseKNum - 1));
                        ComputeMatMul(inner_task_id, ping_pong, startK, endK, realM[i], realbaseM, realbaseN, realbaseK,
                                      offsetM, offsetN, offsetK);
                    }
                    task_id++;
                    offsetK += realbaseK;
                    if ((inner_task_id + 1) % MY_BUFFER_NUM == 0)
                        ping_pong = !ping_pong;
                }
                offsetN += realbaseN;
            }
            offsetM += realbaseM;
        }
        xGm = xGm[realM[i] * tilingData->originK];
        yGm = yGm[realM[i] * tilingData->originN];
        workspaceCGm = workspaceCGm[realM[i] * tilingData->originN];
    }
    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    } else {
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[0]);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[1]);
    }

    if (splitK) {
        if ASCEND_IS_AIV {
            wait_flag_dev(SYNC_CLEAR_AIC_AIV_FLAG);
            CastOutTensor(tilingData->castOutLoop, tilingData->castBaseN, tilingData->castOutTailN,
                          tilingData->castOutTailCoreNum);
        } else {
            ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_CLEAR_AIC_FLAG));
            wait_flag_dev(SYNC_CLEAR_AIC_FLAG);
            ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x02, SYNC_CLEAR_AIC_AIV_FLAG));
        }
    }
    PipeBarrier<PIPE_ALL>();
}
