/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// quant member definitions for AlltoAllMatmul.
#pragma once

#include "allto_all_matmul.h"

namespace Mc2Kernel {

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantPerToken(
    LocalTensor<float> copyTensor, LocalTensor<float> smoothScaleTensor, LocalTensor<float> absTensor,
    LocalTensor<float> reduceMaxTensor, LocalTensor<float> quantScaleTensor, int32_t actualMoveSize,
    int32_t actualMoveToken, int32_t tokenPerMove, int32_t moveIdx, event_t eventId)
{
    float quantMaxValue = std::is_same_v<BType, int8_t> ? MAX_INT8 : MAX_INT4;
    for (int32_t tokenIdx = 0; tokenIdx < actualMoveToken; tokenIdx++) {
        PipeBarrier<PIPE_V>();
        uint32_t tokenOffset = tokenIdx * tokenSize;
        if (isSmoothQuant) {
            Mul(copyTensor[tokenOffset], copyTensor[tokenOffset], smoothScaleTensor, tokenSize);
            PipeBarrier<PIPE_V>();
        }
        Abs(absTensor, copyTensor[tokenOffset], tokenSize);
        PipeBarrier<PIPE_V>();

        ReduceMax<float>(reduceMaxTensor, absTensor, absTensor, tokenSize);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        float maxValue = reduceMaxTensor.GetValue(0);
        float quantScale = maxValue / quantMaxValue;
        float quantScaleReciproal = quantMaxValue / maxValue;
        quantScaleTensor.SetValue(moveIdx * tokenPerMove + tokenIdx, quantScale);
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);

        Muls(copyTensor[tokenOffset], copyTensor[tokenOffset], quantScaleReciproal, tokenSize);
        PipeBarrier<PIPE_V>();

        Cast(copyTensor.ReinterpretCast<int32_t>()[tokenOffset], copyTensor[tokenOffset], RoundMode::CAST_RINT,
             tokenSize);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<half>()[tokenOffset],
             copyTensor.ReinterpretCast<int32_t>()[tokenIdx * tokenSize], RoundMode::CAST_ROUND, tokenSize);
    }
    PipeBarrier<PIPE_V>();
    Cast(copyTensor.ReinterpretCast<BType>(), copyTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC,
         actualMoveSize);
    PipeBarrier<PIPE_V>();
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantToken(__gm__ AType *dataSrc, int64_t dataOffset,
                                                                     int32_t coreTokenOffset, int32_t dataLen,
                                                                     int32_t commIdx)
{
    int32_t tokenNum = dataLen / tokenSize;
    LocalTensor<float> ubTensor = uBuf_.Get<float>();
    /* 用于存储计算完成的量化系数 */
    LocalTensor<float> quantScaleTensor = ubTensor;

    /* 在smoothQuant场景，用于存储smoothScales系数 */
    int32_t smoothScaleOffset = Block32B<float>::AlignUp(tokenNum);
    LocalTensor<float> smoothScaleTensor = quantScaleTensor[smoothScaleOffset];

    // 用于存储allToAll后的A矩阵
    uint32_t copyTensorOffset = isSmoothQuant ? Block32B<float>::AlignUp(tokenSize) : 0;
    uint32_t smoothScaleCastOffset = Block32B<AType>::AlignUp(copyTensorOffset);
    uint32_t midElementCnt = (USED_UB_SIZE / sizeof(float) - smoothScaleOffset - copyTensorOffset) / BUFFER_NUM;
    uint32_t ub_offset = Block32B<float>::AlignUp(midElementCnt);
    LocalTensor<float> copyTensor0 = smoothScaleTensor[copyTensorOffset];
    LocalTensor<float> copyTensor1 = copyTensor0[ub_offset];

    int32_t copyTensorRemainUbSize =
        midElementCnt - Block32B<float>::AlignUp(tokenSize) - BLOCK_ALIGN_BYTES / sizeof(float);
    int32_t ubTokenAlignedPingPongSize = copyTensorRemainUbSize / tokenSize * tokenSize;
    int32_t pingPongMoveCount = (dataLen + ubTokenAlignedPingPongSize - 1) / ubTokenAlignedPingPongSize;
    int32_t actualMoveSize = ubTokenAlignedPingPongSize;
    int32_t tokenPerMove = tokenNum < actualMoveSize / tokenSize ? tokenNum : actualMoveSize / tokenSize;
    int32_t actualMoveToken = tokenPerMove; /* ub_ping_pong_size已经与tokenSize对齐，因此必然每次搬运整数倍token */
    // 动态量化为INT8场景，每个元素1字节；量化为INT4场景，每两个元素1字节
    uint32_t actualMoveBytes = std::is_same_v<BType, int8_t> ? actualMoveSize : actualMoveSize / 2;
    uint32_t sizeScale = std::is_same_v<BType, int8_t> ? 1 : 2;

    /* 用于存储计算quantScale的token取abs的结果 */
    int32_t absOffset = Block32B<float>::AlignUp(
        ubTokenAlignedPingPongSize); /* 从GM拷贝的数据用abs_offset_a大小空间，case为float后用abs_offset大小的空间 */
    int32_t castOffset = Block32B<AType>::AlignUp(absOffset); // 换成AType可能不一定32B对齐
    LocalTensor<float> absTensor0 = copyTensor0[absOffset];
    LocalTensor<float> absTensor1 = copyTensor1[absOffset];

    /* 用于存储取abs后，取出token中最大元素的值 */
    int32_t reduceMaxOffset = Block32B<float>::AlignUp(tokenSize);
    LocalTensor<float> reduceMaxTensor0 = absTensor0[reduceMaxOffset];
    LocalTensor<float> reduceMaxTensor1 = absTensor1[reduceMaxOffset];

    if (isSmoothQuant) {
        // smoothQuant场景需要将x1先乘上smoothQuantScale，MUL不支持bf16，因此需要先转换成float类型
        CopyGmToUbufAlignB16(smoothScaleTensor.ReinterpretCast<AType>()[smoothScaleCastOffset],
                             reinterpret_cast<__gm__ AType *>(x1ScaleGM_), 1, tokenSize * sizeof(AType), 0, 0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(smoothScaleTensor, smoothScaleTensor.ReinterpretCast<AType>()[smoothScaleCastOffset], RoundMode::CAST_NONE,
             tokenSize);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t moveIdx = 0; moveIdx < pingPongMoveCount; ++moveIdx) {
        if (moveIdx == pingPongMoveCount - 1) {
            actualMoveSize = dataLen - moveIdx * ubTokenAlignedPingPongSize;
            actualMoveToken = actualMoveSize / tokenSize;
            actualMoveBytes = std::is_same_v<BType, int8_t> ? actualMoveSize : actualMoveSize / 2;
        }
        auto eventId = (moveIdx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<float> copyTensor = (moveIdx & 1) ? copyTensor0 : copyTensor1;
        LocalTensor<float> absTensor = (moveIdx & 1) ? absTensor0 : absTensor1;
        LocalTensor<float> reduceMaxTensor = (moveIdx & 1) ? reduceMaxTensor0 : reduceMaxTensor1;

        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        CopyGmToUbufAlignB16(copyTensor.ReinterpretCast<AType>()[castOffset],
                             reinterpret_cast<__gm__ AType *>(dataSrc) + dataOffset, 1, actualMoveSize * sizeof(AType),
                             0, 0);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Cast(copyTensor, copyTensor.ReinterpretCast<AType>()[castOffset], RoundMode::CAST_NONE, actualMoveSize);
        QuantPerToken(copyTensor, smoothScaleTensor, absTensor, reduceMaxTensor, quantScaleTensor, actualMoveSize,
                      actualMoveToken, tokenPerMove, moveIdx, eventId);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        /* 搬运到GM上时，与peerMem上的相对位置保持不变，后续数据offset可以复用 */
        CopyUbufToGmAlignB16(reinterpret_cast<__gm__ int8_t *>(quantAGM_) + dataOffset / sizeScale,
                             copyTensor.ReinterpretCast<int8_t>(), 1, actualMoveBytes, 0, 0);
        dataOffset += actualMoveSize;
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    CopyUbufToGmAlignB16(reinterpret_cast<__gm__ float *>(quantScaleGM_) + coreTokenOffset + commIdx * mPerLoop,
                         quantScaleTensor, 1, tokenNum * sizeof(float), 0, 0);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::SmoothQuantProc(event_t eventId, int32_t dataSegmentOffset,
                                                                          int32_t smoothScaleCastOffset,
                                                                          int32_t actualMoveSize,
                                                                          LocalTensor<float> copyTensor,
                                                                          LocalTensor<float> smoothScaleTensor)
{
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    CopyGmToUbufAlignB16(smoothScaleTensor.ReinterpretCast<AType>()[smoothScaleCastOffset],
                         reinterpret_cast<__gm__ AType *>(x1ScaleGM_) + dataSegmentOffset, 1,
                         actualMoveSize * sizeof(AType), 0, 0);
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Cast(smoothScaleTensor, smoothScaleTensor.ReinterpretCast<AType>()[smoothScaleCastOffset], RoundMode::CAST_NONE,
         actualMoveSize);
    PipeBarrier<PIPE_V>();
    Mul(copyTensor, copyTensor, smoothScaleTensor, actualMoveSize);
    PipeBarrier<PIPE_V>();
}

// 公共片段：搬运一段token到UB并转float，随后按需执行smooth量化
template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::LoadCastAndSmoothToken(
    event_t eventId, LocalTensor<float> copyTensor, LocalTensor<float> smoothScaleTensor, int32_t castOffset,
    __gm__ AType *dataSrc, int64_t dataTokenOffset, int32_t dataSegmentOffset, int32_t smoothScaleCastOffset,
    int32_t actualMoveSize)
{
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    /* 下一步需要将copyTensor转换成float类型，目标地址复用copyTensor。为防止踩踏，将AType类型数据内存放在后半段 */
    CopyGmToUbufAlignB16(copyTensor.ReinterpretCast<AType>()[castOffset],
                         reinterpret_cast<__gm__ AType *>(dataSrc) + dataTokenOffset + dataSegmentOffset, 1,
                         actualMoveSize * sizeof(AType), 0, 0);
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Cast(copyTensor, copyTensor.ReinterpretCast<AType>()[castOffset], RoundMode::CAST_NONE, actualMoveSize);
    PipeBarrier<PIPE_V>();
    if (isSmoothQuant) {
        SmoothQuantProc(eventId, dataSegmentOffset, smoothScaleCastOffset, actualMoveSize, copyTensor,
                        smoothScaleTensor);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CalcTokenMaxValue(
    LocalTensor<float> copyTensor0, LocalTensor<float> copyTensor1, LocalTensor<float> absTensor0,
    LocalTensor<float> absTensor1, LocalTensor<float> smoothScaleTensor0, LocalTensor<float> smoothScaleTensor1,
    int32_t castOffset, __gm__ AType *dataSrc, int64_t dataTokenOffset, int32_t smoothScaleCastOffset,
    int32_t sizeScale, LocalTensor<float> reduceMaxTensor)
{
    int32_t actualMoveSize = copyTensorSize;
    int32_t dataSegmentOffset = 0;
    /* 获取当前token的max_abs_value */
    for (int32_t tokenSegmentLoop = 0; tokenSegmentLoop < copyTimes; tokenSegmentLoop++) {
        if (tokenSegmentLoop == copyTimes - 1) {
            actualMoveSize = tokenSize - tokenSegmentLoop * copyTensorSize;
        }
        auto eventId = (tokenSegmentLoop & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<float> copyTensor = (tokenSegmentLoop & 1) ? copyTensor0 : copyTensor1;
        LocalTensor<float> absTensor = (tokenSegmentLoop & 1) ? absTensor0 : absTensor1;
        LocalTensor<float> smoothScaleTensor = (tokenSegmentLoop & 1) ? smoothScaleTensor0 : smoothScaleTensor1;
        LoadCastAndSmoothToken(eventId, copyTensor, smoothScaleTensor, castOffset, dataSrc, dataTokenOffset,
                               dataSegmentOffset, smoothScaleCastOffset, actualMoveSize);
        Abs(absTensor, copyTensor, actualMoveSize);
        PipeBarrier<PIPE_V>();
        ReduceMax<float>(copyTensor, absTensor, absTensor, actualMoveSize);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        float currentMaxValue = copyTensor.GetValue(0);
        float lastMaxValue = reduceMaxTensor.GetValue(0);
        float maxValue = currentMaxValue > lastMaxValue ? currentMaxValue : lastMaxValue;
        reduceMaxTensor.SetValue(0, maxValue);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        dataSegmentOffset += actualMoveSize;
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantPerSegment(
    LocalTensor<float> copyTensor0, LocalTensor<float> copyTensor1, LocalTensor<float> absTensor0,
    LocalTensor<float> absTensor1, LocalTensor<float> smoothScaleTensor0, LocalTensor<float> smoothScaleTensor1,
    int32_t castOffset, __gm__ AType *dataSrc, int64_t dataTokenOffset, int32_t smoothScaleCastOffset,
    int32_t sizeScale, float quantScaleReciproal)
{
    int32_t actualMoveSize = copyTensorSize;
    int32_t actualMoveBytes = std::is_same_v<BType, int8_t> ? actualMoveSize : actualMoveSize / 2;
    int32_t dataSegmentOffset = 0;
    /* 量化当前token */
    for (int32_t tokenSegmentLoop = 0; tokenSegmentLoop < copyTimes; tokenSegmentLoop++) {
        if (tokenSegmentLoop == copyTimes - 1) {
            actualMoveSize = tokenSize - tokenSegmentLoop * copyTensorSize;
            actualMoveBytes = std::is_same_v<BType, int8_t> ? actualMoveSize : actualMoveSize / 2;
        }
        auto eventId = (tokenSegmentLoop & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<float> copyTensor = (tokenSegmentLoop & 1) ? copyTensor0 : copyTensor1;
        LocalTensor<float> absTensor = (tokenSegmentLoop & 1) ? absTensor0 : absTensor1;
        LocalTensor<float> smoothScaleTensor = (tokenSegmentLoop & 1) ? smoothScaleTensor0 : smoothScaleTensor1;
        LoadCastAndSmoothToken(eventId, copyTensor, smoothScaleTensor, castOffset, dataSrc, dataTokenOffset,
                               dataSegmentOffset, smoothScaleCastOffset, actualMoveSize);
        Muls(copyTensor, copyTensor, quantScaleReciproal, actualMoveSize);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<int32_t>(), copyTensor, RoundMode::CAST_RINT, actualMoveSize);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<half>(), copyTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND,
             actualMoveSize);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<BType>(), copyTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC,
             actualMoveSize);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        CopyUbufToGmAlignB16(
            reinterpret_cast<__gm__ int8_t *>(quantAGM_) + (dataTokenOffset + dataSegmentOffset) / sizeScale,
            copyTensor.ReinterpretCast<int8_t>(), 1, actualMoveBytes, 0, 0);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        dataSegmentOffset += actualMoveSize;
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantTokenSegment(__gm__ AType *dataSrc, int64_t dataOffset,
                                                                            int32_t coreTokenOffset, int32_t dataLen,
                                                                            int32_t commIdx)
{
    int32_t tokenNum = dataLen / tokenSize; /* 当前核实际处理的token数 */
    LocalTensor<float> ubTensor = uBuf_.Get<float>();
    LocalTensor<float> quantScaleTensor = ubTensor;

    int32_t reduceMaxOffset = Block32B<float>::AlignUp(tokenNum);
    LocalTensor<float> reduceMaxTensor = quantScaleTensor[reduceMaxOffset];

    // 用于存储allToAll后的A矩阵
    uint32_t copyTensorOffset = BLOCK_ALIGN_BYTES / sizeof(float);
    uint32_t midElementCnt = (USED_UB_SIZE / sizeof(float) - reduceMaxOffset - copyTensorOffset) / BUFFER_NUM;
    uint32_t ub_offset = Block32B<float>::AlignUp(midElementCnt);
    LocalTensor<float> copyTensor0 = reduceMaxTensor[copyTensorOffset];
    LocalTensor<float> copyTensor1 = copyTensor0[ub_offset];

    int32_t absOffset = Block32B<float>::AlignUp(
        copyTensorSize); /* 从GM拷贝的数据用abs_offset_a大小空间，case为float后用abs_offset大小的空间 */
    int32_t castOffset = Block32B<AType>::AlignUp(absOffset); // 换成AType可能不一定32B对齐
    LocalTensor<float> absTensor0 = copyTensor0[absOffset];
    LocalTensor<float> absTensor1 = copyTensor1[absOffset];

    /* 在smoothQuant场景，用于存储smoothScales系数 */
    int32_t smoothScaleOffset = Block32B<float>::AlignUp(copyTensorSize);
    int32_t smoothScaleCastOffset = Block32B<AType>::AlignDown(smoothScaleOffset);
    LocalTensor<float> smoothScaleTensor0 = absTensor0[smoothScaleOffset];
    LocalTensor<float> smoothScaleTensor1 = absTensor1[smoothScaleOffset];

    uint32_t sizeScale = std::is_same_v<BType, int8_t> ? 1 : 2;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (int32_t tokenLoop = 0; tokenLoop < tokenNum; tokenLoop++) {
        int64_t dataTokenOffset = dataOffset + static_cast<int64_t>(tokenLoop) * tokenSize;
        Duplicate<float>(reduceMaxTensor, static_cast<float>(0), 1);
        PipeBarrier<PIPE_V>();
        /* 获取当前token的max_abs_value */
        CalcTokenMaxValue(copyTensor0, copyTensor1, absTensor0, absTensor1, smoothScaleTensor0, smoothScaleTensor1,
                          castOffset, dataSrc, dataTokenOffset, smoothScaleCastOffset, sizeScale, reduceMaxTensor);
        float tokenMaxValue = reduceMaxTensor.GetValue(0);
        float quantMaxValue = std::is_same_v<BType, int8_t> ? MAX_INT8 : MAX_INT4;
        float quantScale = tokenMaxValue / quantMaxValue;
        float quantScaleReciproal = quantMaxValue / tokenMaxValue;
        quantScaleTensor.SetValue(tokenLoop, quantScale);
        /* 量化当前token */
        QuantPerSegment(copyTensor0, copyTensor1, absTensor0, absTensor1, smoothScaleTensor0, smoothScaleTensor1,
                        castOffset, dataSrc, dataTokenOffset, smoothScaleCastOffset, sizeScale, quantScaleReciproal);
    }
    CopyUbufToGmAlignB16(reinterpret_cast<__gm__ float *>(quantScaleGM_) + coreTokenOffset + commIdx * mPerLoop,
                         quantScaleTensor, 1, tokenNum * sizeof(float), 0, 0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Quant(uint64_t flagIdx, int32_t commIdx)
{
    __gm__ AType *dataSrc = (__gm__ AType *)buff[rank];
    int32_t totalDataSize =
        x1DataSize < allToAllSizeAllRanksPerLoop ? x1DataSize : allToAllSizeAllRanksPerLoop; // 实际需要处理的数据量
    int32_t quantSizePerCore = (totalDataSize / quantCoreNum) / tokenSize * tokenSize;       // 每核均分的数据量
    int32_t remainTokenNum = (totalDataSize - quantSizePerCore * quantCoreNum) /
                             tokenSize; // 每个核均分quantSizePerCore之后，剩余的token由前remainTokenNum个核各多分担一个
    uint32_t globalAivIdx = aicIdx * 2 + aivIdx;
    int32_t dataSrcCoreOffset = 0;
    int32_t dataLen = 0;
    int32_t coreTokenOffset = 0;
    int32_t tokenPercore = quantSizePerCore / tokenSize;
    if (globalAivIdx < remainTokenNum) {
        dataSrcCoreOffset = (globalAivIdx % quantCoreNum) * (quantSizePerCore + tokenSize);
        dataLen = dataSrcCoreOffset + (quantSizePerCore + tokenSize) > totalDataSize ?
                      totalDataSize - dataSrcCoreOffset :
                      quantSizePerCore + tokenSize;
        coreTokenOffset = (globalAivIdx % quantCoreNum) * (tokenPercore + 1);
    } else {
        dataSrcCoreOffset = remainTokenNum * (quantSizePerCore + tokenSize) +
                            (globalAivIdx % quantCoreNum - remainTokenNum) * (quantSizePerCore);
        dataLen =
            dataSrcCoreOffset + quantSizePerCore > totalDataSize ? totalDataSize - dataSrcCoreOffset : quantSizePerCore;
        coreTokenOffset =
            remainTokenNum * (tokenPercore + 1) + (globalAivIdx % quantCoreNum - remainTokenNum) * tokenPercore;
    }
    if (dataLen <= 0) {
        return;
    }
    int64_t dataSrcOffset = static_cast<int64_t>(flagIdx) * pingPongBlockSize;
    int64_t dataOffset = dataSrcOffset + dataSrcCoreOffset;
    if (isSegmentK) {
        // token过大，分段量化
        QuantTokenSegment(dataSrc, dataOffset, coreTokenOffset, dataLen, commIdx);
    } else {
        // token较小，一次量化多个
        QuantToken(dataSrc, dataOffset, coreTokenOffset, dataLen, commIdx);
    }
}

} // namespace Mc2Kernel
