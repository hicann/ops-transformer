/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_A2_A3_COPY_HELPERS_HPP
#define MEGA_MOE_A2_A3_COPY_HELPERS_HPP
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "utils/const_args.hpp"

namespace Catlass::Gemm::Kernel::Detail {
using namespace AscendC;

CATLASS_DEVICE uint32_t DispatchWorkerCount(bool fastPath, uint32_t coreNum, uint32_t tasks, uint32_t maxWorkers)
{
    return fastPath ? 1U : max(1U, min(maxWorkers, coreNum / tasks));
}

template <class Params>
CATLASS_DEVICE bool HasDispatchBatchRows(const Params& params, uint32_t first, uint32_t groups,
                                         const LocalTensor<int32_t>& sendCounts, const LocalTensor<int32_t>& recvCounts,
                                         uint32_t recvStride)
{
    // All AIVs must agree before skipping the batch barrier.
    for (uint32_t peer = 0; peer < params.EP; ++peer) {
        for (uint32_t group = first; group < first + groups; ++group) {
            if (sendCounts.GetValue(peer * params.expertPerRank + group) != 0 ||
                recvCounts.GetValue(peer * recvStride + group) != 0) {
                return true;
            }
        }
    }
    return false;
}

template <bool BatchedDispatch>
CATLASS_DEVICE bool CanSkipCombineExpertSync(uint32_t ep, uint32_t coreNum, uint32_t peerWorkers)
{
    return BatchedDispatch && ep <= SERVER_RANK_SIZE_A2 && ep * peerWorkers <= coreNum;
}

// Contiguous slices preserve row order for payload and per-token scales.
CATLASS_DEVICE uint32_t PartitionDispatchRows(uint32_t rows, uint32_t workers, uint32_t worker, uint32_t& first)
{
    const uint32_t capacity = (rows + workers - 1) / workers;
    first = worker * capacity;
    return first < rows ? min(capacity, rows - first) : 0;
}

template <uint32_t BatchRows, typename T, typename CopyIn, typename CopyOut>
CATLASS_DEVICE void CopyDispatchRows(AscendC::GlobalTensor<T>& dst, AscendC::GlobalTensor<T>& src,
                                     AscendC::LocalTensor<T>& buffer, int32_t rows, uint32_t width, CopyIn& copyIn,
                                     CopyOut& copyOut)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int32_t row = 0; row < rows; row += BatchRows) {
        uint32_t count = 1;
        if constexpr (BatchRows > 1) {
            count = min(static_cast<uint32_t>(rows - row), BatchRows);
        }
        const auto offset = row * width;
        const layout::RowMajor tile{count, width};
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        copyIn(buffer, src[offset], tile, tile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        copyOut(dst[offset], buffer, tile, tile);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void CopyGMToGMShared(Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<T> dst,
                                     AscendC::GlobalTensor<T> src, int32_t elemNum, int32_t ubMoveNum)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    using TType = Gemm::GemmType<T, layout::RowMajor>;
    using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
    using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
    CopyGmToUb copyGmToUb;
    CopyUbToGm copyUbToGm;
    constexpr int32_t BufferNum = 2;
    int tmpBufferSize = 32 * 1024 / sizeof(T); // 32 KB
    AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
    tmpBuffer1.SetSize(tmpBufferSize);
    int tmpBufferOffset = 96 * 1024; // half of UB
    AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
    tmpBuffer2.SetSize(tmpBufferSize);

    int pingpongId = 0;
    auto processCount = CeilDiv(elemNum, ubMoveNum);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        uint32_t curProcessNum =
            (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) : ubMoveNum;
        AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        auto processOffset = processIndex * ubMoveNum;

        auto inputOffset = processOffset;
        auto outputOffset = processOffset;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        copyGmToUb(buf, src[inputOffset], layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        copyUbToGm(dst[outputOffset], buf, layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        pingpongId = (pingpongId + 1) % BufferNum;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

CATLASS_DEVICE void CalculateTaskInfoEachCoreShared(uint32_t& curCoreTaskNum, uint32_t& curCoreStartOffset,
                                                    uint32_t totalM, uint32_t coreNum)
{
    int64_t eachCoreTaskNum = (totalM + coreNum - 1) / coreNum;
    int64_t taskNumPertailCore = eachCoreTaskNum - 1;
    int64_t usedCoreNum = totalM >= coreNum ? coreNum : totalM;
    uint32_t tailCoreIdx = totalM - (eachCoreTaskNum - 1) * usedCoreNum;
    uint32_t curCoreId = GetBlockIdx();
    curCoreTaskNum = curCoreId < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    curCoreStartOffset =
        curCoreId < tailCoreIdx ? eachCoreTaskNum * curCoreId : ((eachCoreTaskNum - 1) * curCoreId + tailCoreIdx);
}

template <class ArchTag>
CATLASS_DEVICE void LoadDispatchMsdBatch(AscendC::LocalTensor<int8_t>& input, AscendC::LocalTensor<int8_t>& scaleSlab,
                                         AscendC::GlobalTensor<int8_t>& src, uint32_t first, uint32_t count,
                                         uint32_t hiddenSize, uint32_t inputStride, AscendC::TEventID event)
{
    using Type = Gemm::GemmType<int8_t, layout::RowMajor>;
    Epilogue::Tile::CopyGm2Ub<ArchTag, Type> copyIn;
    WaitFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    copyIn(input, src[static_cast<uint64_t>(first) * inputStride], layout::RowMajor{count, hiddenSize},
           layout::RowMajor{count, hiddenSize, inputStride});
    copyIn(scaleSlab, src[static_cast<uint64_t>(first) * inputStride + hiddenSize], layout::RowMajor{count, 32},
           layout::RowMajor{count, 32, inputStride});
    SetFlag<HardEvent::MTE2_V>(event);
}

template <class ArchTag>
CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4BatchedShared(Arch::Resource<ArchTag>& resource,
                                                              AscendC::GlobalTensor<int8_t> gmOutput,
                                                              uint64_t gmOffsetA, AscendC::GlobalTensor<float> dstScale,
                                                              AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                              int32_t hiddenSize, uint32_t outputStride,
                                                              uint32_t inputPadding)
{
    if (rows <= 0)
        return;
    constexpr uint32_t BATCH_ROWS = 4;
    const uint32_t capacity = BATCH_ROWS * hiddenSize;
    const uint32_t inputStride = hiddenSize + inputPadding;
    AscendC::LocalTensor<int8_t> input[2], scaleSlab[2];
    AscendC::LocalTensor<float> compactScale[2];
    AscendC::LocalTensor<int4b_t> high[2], low[2];
    size_t offset = 0;
    for (uint32_t stage = 0; stage < 2; ++stage) {
        input[stage] = resource.ubBuf.template GetBufferByByte<int8_t>(offset);
        offset += capacity;
        scaleSlab[stage] = resource.ubBuf.template GetBufferByByte<int8_t>(offset);
        offset += BATCH_ROWS * 32;
        compactScale[stage] = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += 32;
        high[stage] = resource.ubBuf.template GetBufferByByte<int4b_t>(offset);
        offset += capacity / 2;
        low[stage] = resource.ubBuf.template GetBufferByByte<int4b_t>(offset);
        offset += capacity / 2;
        SetFlag<HardEvent::V_MTE2>(stage == 0 ? EVENT_ID6 : EVENT_ID7);
        SetFlag<HardEvent::MTE3_MTE2>(stage == 0 ? EVENT_ID6 : EVENT_ID7);
        SetFlag<HardEvent::MTE3_V>(stage == 0 ? EVENT_ID6 : EVENT_ID7);
    }
    auto highHalf = resource.ubBuf.template GetBufferByByte<half>(offset);
    offset += capacity * sizeof(half);
    auto lowHalf = resource.ubBuf.template GetBufferByByte<half>(offset);
    offset += capacity * sizeof(half);
    auto lowHalf2 = resource.ubBuf.template GetBufferByByte<half>(offset);
    offset += capacity * sizeof(half);
    auto mask = resource.ubBuf.template GetBufferByByte<int16_t>(offset);
    Duplicate(mask, static_cast<int16_t>(0x0F0F), 128);
    PipeBarrier<PIPE_V>();
    LoadDispatchMsdBatch<ArchTag>(input[0], scaleSlab[0], src, 0, min(BATCH_ROWS, static_cast<uint32_t>(rows)),
                                  hiddenSize, inputStride, EVENT_ID6);
    uint32_t stage = 0;
    for (uint32_t first = 0; first < static_cast<uint32_t>(rows); first += BATCH_ROWS) {
        const uint32_t count = min(BATCH_ROWS, static_cast<uint32_t>(rows) - first);
        const uint32_t elements = count * hiddenSize;
        const auto event = stage == 0 ? EVENT_ID6 : EVENT_ID7;
        if (first + count < static_cast<uint32_t>(rows)) {
            const uint32_t nextStage = 1 - stage;
            LoadDispatchMsdBatch<ArchTag>(input[nextStage], scaleSlab[nextStage], src, first + count,
                                          min(BATCH_ROWS, static_cast<uint32_t>(rows) - first - count), hiddenSize,
                                          inputStride, nextStage == 0 ? EVENT_ID6 : EVENT_ID7);
        }
        WaitFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
        SetFlag<HardEvent::MTE2_S>(event);
        WaitFlag<HardEvent::MTE2_S>(event);
        auto scaleValues = scaleSlab[stage].template ReinterpretCast<float>();
        for (uint32_t row = 0; row < count; ++row) {
            compactScale[stage].SetValue(row, scaleValues.GetValue(row * 8));
        }
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);
        DataCopyPad(dstScale[first], compactScale[stage], {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
        Cast(highHalf, input[stage], RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Muls(highHalf, highHalf, static_cast<half>(0.0625f), elements);
        PipeBarrier<PIPE_V>();
        Cast(high[stage], highHalf, RoundMode::CAST_FLOOR, elements);
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopyExtParams outputCopy{static_cast<uint16_t>(count), static_cast<uint32_t>(hiddenSize / 2), 0,
                                     static_cast<uint32_t>(outputStride - hiddenSize / 2), 0};
        const uint64_t out = gmOffsetA + static_cast<uint64_t>(first) * outputStride;
        DataCopyPad(gmOutput[out], high[stage].template ReinterpretCast<int8_t>(), outputCopy);
        And(lowHalf.template ReinterpretCast<int16_t>(), input[stage].template ReinterpretCast<int16_t>(), mask, 128,
            elements / 256, {1, 1, 1, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(event);
        Cast(lowHalf2, lowHalf.template ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Adds(highHalf, lowHalf2, static_cast<half>(-8), elements);
        PipeBarrier<PIPE_V>();
        Cast(low[stage], highHalf, RoundMode::CAST_NONE, elements);
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopyPad(gmOutput[out + outputStride / 2], low[stage].template ReinterpretCast<int8_t>(), outputCopy);
        SetFlag<HardEvent::MTE3_V>(event);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        stage = 1 - stage;
    }
    for (uint32_t stage = 0; stage < 2; ++stage) {
        const auto event = stage == 0 ? EVENT_ID6 : EVENT_ID7;
        WaitFlag<HardEvent::V_MTE2>(event);
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
    }
}

template <class ArchTag>
CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4Shared(Arch::Resource<ArchTag>& resource,
                                                       AscendC::GlobalTensor<int8_t> gmA1I4_I8, uint64_t gmOffsetA,
                                                       AscendC::GlobalTensor<float> dstScale,
                                                       AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                       int32_t hiddenSize, uint32_t outputStride, uint32_t coreNum,
                                                       uint32_t inputPadding = ALIGN_512)
{
    uint32_t tmpBufferOffset = 0;

    // double buffer 定义： tmpBuffer 包含int8_t token + float scale, xTensor 只包含int8_t token
    AscendC::LocalTensor<int8_t> xTensor0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    AscendC::LocalTensor<int8_t> tmpBuffer0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    xTensor0.SetSize(hiddenSize);
    tmpBuffer0.SetSize(hiddenSize + inputPadding);
    tmpBufferOffset += (hiddenSize + inputPadding) * sizeof(int8_t);

    AscendC::LocalTensor<int8_t> xTensor1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    AscendC::LocalTensor<int8_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    xTensor1.SetSize(hiddenSize);
    tmpBuffer1.SetSize(hiddenSize + inputPadding);
    tmpBufferOffset += (hiddenSize + inputPadding) * sizeof(int8_t);
    // double buffer 定义结束

    AscendC::LocalTensor<int4b_t> xHighI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
    xHighI4Tensor.SetSize(hiddenSize);
    tmpBufferOffset += hiddenSize / INT4_NUM_PER_BYTE;

    AscendC::LocalTensor<int4b_t> xLowI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
    xLowI4Tensor.SetSize(hiddenSize);
    tmpBufferOffset += hiddenSize / INT4_NUM_PER_BYTE;

    AscendC::LocalTensor<half> xHighHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    xHighHalfTensor.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    AscendC::LocalTensor<half> xLowHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    xLowHalfTensor.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    AscendC::LocalTensor<half> xLowHalfTensor2 = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    xLowHalfTensor2.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    AscendC::LocalTensor<int16_t> xLowI16Tensor = resource.ubBuf.template GetBufferByByte<int16_t>(tmpBufferOffset);
    constexpr int32_t MASK = 128; // int4 解包掩码宽度（int16 单位，bit 为 0x0F0F）
    xLowI16Tensor.SetSize(MASK * sizeof(int16_t));

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);

    Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK);
    PipeBarrier<PIPE_V>();
    const size_t LEN_VK = (hiddenSize / INT4_NUM_PER_BYTE) / MASK;
    const size_t LAST_LEN_VK = (hiddenSize % 256) / 2;
    const half ONE_SIXTEENTH = static_cast<half>(0.0625f);

    uint32_t totalM = rows;

    SetFlag<HardEvent::S_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID6);
    uint32_t curCoreTaskNum;
    uint32_t curCoreStartOffset;
    CalculateTaskInfoEachCoreShared(curCoreTaskNum, curCoreStartOffset, totalM, coreNum);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);

    constexpr size_t LEN_128 = 128;

    constexpr int32_t BufferNum = 2;
    uint32_t copyInNum = hiddenSize + inputPadding;

    int pingpongId = 0;
    for (uint32_t processIndex = 0; processIndex < rows; ++processIndex) {
        const uint64_t relStartAddr = processIndex * outputStride;
        uint64_t absStartAddr = gmOffsetA + relStartAddr;
        AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID6 : EVENT_ID7;
        AscendC::LocalTensor<int8_t> buf = pingpongId == 0 ? tmpBuffer0 : tmpBuffer1;
        AscendC::LocalTensor<int8_t> xTensor = pingpongId == 0 ? xTensor0 : xTensor1;
        AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
        auto inputOffset = processIndex * copyInNum;
        auto outputOffset = processIndex * hiddenSize;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID);
        AscendC::DataCopy(buf, src[inputOffset], copyInNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        // 远端 scale 拷贝到GM
        AscendC::DataCopyPad(dstScale[processIndex], bufScale, {1, 4, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID);
        // 高四位处理开始
        Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, hiddenSize);
        PipeBarrier<PIPE_V>();
        Muls(xHighHalfTensor, xHighHalfTensor, ONE_SIXTEENTH, hiddenSize);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
        Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, hiddenSize);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
        DataCopy(gmA1I4_I8[absStartAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), hiddenSize / INT4_NUM_PER_BYTE);
        // 高四位处理结束

        // 低四位处理开始
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
        And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), xLowI16Tensor, LEN_128,
            LEN_VK, {1, 1, 1, 8, 8, 0});
        if (LAST_LEN_VK > 0) {
            And(xLowHalfTensor[LEN_VK * LEN_128].ReinterpretCast<int16_t>(),
                xTensor[LEN_VK * LEN_128 * TWO].ReinterpretCast<int16_t>(), xLowI16Tensor, LAST_LEN_VK, 1,
                {1, 1, 1, 8, 8, 0});
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID);
        Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(),
             AscendC::RoundMode::CAST_NONE, hiddenSize);
        PipeBarrier<PIPE_V>();
        const half MINUS_EIGHT = static_cast<half>(-8);
        Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, hiddenSize);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
        Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, hiddenSize);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
        DataCopy(gmA1I4_I8[absStartAddr + outputStride / INT4_NUM_PER_BYTE], xLowI4Tensor.ReinterpretCast<int8_t>(),
                 hiddenSize / INT4_NUM_PER_BYTE);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
        // 低四位处理结束
        pingpongId = (pingpongId + 1) % BufferNum;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
}

} // namespace Catlass::Gemm::Kernel::Detail

#endif // MEGA_MOE_A2_A3_COPY_HELPERS_HPP
