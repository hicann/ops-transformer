/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MHC_POST_BACKWARD_CUBE_H
#define MHC_POST_BACKWARD_CUBE_H

#include "kernel_operator.h"

namespace MhcPostBackward {
using namespace AscendC;

// Mode 4 maps AIV1's flag IDs to the AIC's second flag range (+16).
constexpr uint16_t GRAD_Y_MTE1_TO_MTE3_FLAG_BASE = 6;
constexpr uint16_t GRAD_Y_MTE3_TO_MTE1_FLAG_BASE = 8;
constexpr uint16_t AIV_SUB_BLOCK_FLAG_OFFSET = 16;
constexpr uint16_t GRAD_H_RES_FIX_TO_MTE2_FLAG_BASE = 0;
constexpr uint16_t GRAD_H_RES_MTE2_TO_FIX_FLAG_BASE = 2;

struct MhcPostBackwardCubeParams {
    uint32_t headCount, hiddenSize, mergeBatch, l1TileK, l0TileK;
    uint32_t inputL1Offsets[2], gradYL1Offsets[2];
    uint32_t kPartitionSize{0};
};

template <typename T>
class MhcPostBackwardCube {
public:
    __aicore__ inline MhcPostBackwardCube()
    {
        if ASCEND_IS_AIC {
            TPipe* pipe = GetTPipePtr();
            for (uint32_t bufferIndex = 0; bufferIndex < 2; ++bufferIndex) {
                eventIdMte1ToMte2_[bufferIndex] = pipe->AllocEventID<HardEvent::MTE1_MTE2>();
                eventIdMte2ToMte1_[bufferIndex] = pipe->AllocEventID<HardEvent::MTE2_MTE1>();
                eventIdMToMte1_[bufferIndex] = pipe->AllocEventID<HardEvent::M_MTE1>();
                eventIdMte1ToM_[bufferIndex] = pipe->AllocEventID<HardEvent::MTE1_M>();
                eventIdFixToM_[bufferIndex] = pipe->AllocEventID<HardEvent::FIX_M>();
                eventIdMToFix_[bufferIndex] = pipe->AllocEventID<HardEvent::M_FIX>();

                SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_[bufferIndex]);
                SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_[bufferIndex]);
                SetFlag<HardEvent::FIX_M>(eventIdFixToM_[bufferIndex]);
            }
        }
    }

    __aicore__ inline ~MhcPostBackwardCube()
    {
        if ASCEND_IS_AIC {
            // Init can be skipped on a core with no tasks; its initial events still need releasing.
            TPipe* pipe = pipe_ != nullptr ? pipe_ : GetTPipePtr();
            for (uint32_t bufferIndex = 0; bufferIndex < 2; ++bufferIndex) {
                WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_[bufferIndex]);
                WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_[bufferIndex]);
                WaitFlag<HardEvent::FIX_M>(eventIdFixToM_[bufferIndex]);
                pipe->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_[bufferIndex]);
                pipe->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_[bufferIndex]);
                pipe->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1_[bufferIndex]);
                pipe->ReleaseEventID<HardEvent::MTE1_M>(eventIdMte1ToM_[bufferIndex]);
                pipe->ReleaseEventID<HardEvent::FIX_M>(eventIdFixToM_[bufferIndex]);
                pipe->ReleaseEventID<HardEvent::M_FIX>(eventIdMToFix_[bufferIndex]);
            }
        }
    }

    __aicore__ inline void Init(TPipe* pipe, const MhcPostBackwardCubeParams& params)
    {
        pipe_ = pipe;
        cubeParams_ = params;
        l1Buffer_ = LocalTensor<T>{TPosition::A1, 0, TOTAL_L1_SIZE / sizeof(T)};
        l0aBuffer_ = LocalTensor<T>{TPosition::A2, 0, TOTAL_L0A_SIZE / sizeof(T)};
        l0bBuffer_ = LocalTensor<T>{TPosition::B2, 0, TOTAL_L0B_SIZE / sizeof(T)};
        l0cBuffer_ = LocalTensor<float>{TPosition::CO1, 0, TOTAL_L0C_SIZE / sizeof(float)};
        processedGroupCount_ = 0;
        l1TransferCount_ = 0;
        mmadCount_ = 0;
    }

    MhcPostBackwardCube(const MhcPostBackwardCube&) = delete;

    MhcPostBackwardCube& operator=(const MhcPostBackwardCube&) = delete;

    template <bool SEGMENTED, bool USE_RESULT_WORKSPACE>
    __aicore__ inline void operator()(const GlobalTensor<T>& xGlobal, const GlobalTensor<float>& gradHResGlobal,
                                      uint32_t first, uint32_t count, uint32_t nextFirst, uint32_t nextCount,
                                      const GlobalTensor<float>& partialResultsGlobal)
    {
        const uint32_t l0cPingPongIndex = processedGroupCount_ % 2;
        auto l0cResult = l0cBuffer_[l0cPingPongIndex * (TOTAL_L0C_SIZE / 2 / sizeof(float))];
        constexpr bool segmented = SEGMENTED;
        if (USE_RESULT_WORKSPACE && processedGroupCount_ >= 2) {
            CrossCoreWaitFlag<4, PIPE_FIX>(GRAD_H_RES_MTE2_TO_FIX_FLAG_BASE + l0cPingPongIndex);
            CrossCoreWaitFlag<4, PIPE_FIX>(GRAD_H_RES_MTE2_TO_FIX_FLAG_BASE + AIV_SUB_BLOCK_FLAG_OFFSET +
                                           l0cPingPongIndex);
        }
        WaitFlag<HardEvent::FIX_M>(eventIdFixToM_[l0cPingPongIndex]);
        uint32_t partStart = 0;
        uint32_t partEnd = GetKPartitionEnd(0);

        const uint64_t offset = static_cast<uint64_t>(first) * cubeParams_.headCount * cubeParams_.hiddenSize;
        uint32_t l1PingPongIndex = l1TransferCount_ % 2;
        if (processedGroupCount_ == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_[l1PingPongIndex]);
            DataCopyA(xGlobal[offset], 0, count, l1PingPongIndex);
            SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_[l1PingPongIndex]);
        }

        for (uint32_t k = 0; k < cubeParams_.hiddenSize; k += cubeParams_.l1TileK, l1PingPongIndex ^= 1) {
            const uint32_t realK = Min(cubeParams_.l1TileK, cubeParams_.hiddenSize - k);
            WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_[l1PingPongIndex]);
            if (k + cubeParams_.l1TileK < cubeParams_.hiddenSize) {
                const uint32_t nextL1PingPongIndex = l1PingPongIndex ^ 1;
                WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_[nextL1PingPongIndex]);
                DataCopyA(xGlobal[offset], k + cubeParams_.l1TileK, count, nextL1PingPongIndex);
                SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_[nextL1PingPongIndex]);
            }

            CrossCoreWaitFlag<4, PIPE_MTE1>(GRAD_Y_MTE3_TO_MTE1_FLAG_BASE + l1PingPongIndex);
            CrossCoreWaitFlag<4, PIPE_MTE1>(GRAD_Y_MTE3_TO_MTE1_FLAG_BASE + AIV_SUB_BLOCK_FLAG_OFFSET +
                                            l1PingPongIndex);

            const uint32_t stepK =
                SEGMENTED ? Min(cubeParams_.l0TileK, cubeParams_.kPartitionSize) : cubeParams_.l0TileK;
            for (uint32_t b = 0; b < realK; b += stepK) {
                const uint32_t validK = Min(stepK, realK - b);
                const uint32_t size = AlignCube(validK);
                const uint32_t l0PingPongIndex = mmadCount_++ % 2;
                auto aL0 = l0aBuffer_[l0PingPongIndex * (TOTAL_L0A_SIZE / 2 / sizeof(T))];
                auto bL0 = l0bBuffer_[l0PingPongIndex * (TOTAL_L0B_SIZE / 2 / sizeof(T))];

                WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_[l0PingPongIndex]);

                LoadDataA(aL0, l1Buffer_[cubeParams_.inputL1Offsets[l1PingPongIndex] + b * 16], size);
                LoadDataB(bL0, l1Buffer_[cubeParams_.gradYL1Offsets[l1PingPongIndex] + b * 16], size);

                if (b + stepK >= realK) {
                    SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_[l1PingPongIndex]);
                    CrossCoreSetFlag<4, PIPE_MTE1>(GRAD_Y_MTE1_TO_MTE3_FLAG_BASE + l1PingPongIndex);
                    CrossCoreSetFlag<4, PIPE_MTE1>(GRAD_Y_MTE1_TO_MTE3_FLAG_BASE + AIV_SUB_BLOCK_FLAG_OFFSET +
                                                   l1PingPongIndex);
                }

                SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM_[l0PingPongIndex]);
                WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM_[l0PingPongIndex]);

                const bool initializeC = SEGMENTED ? k + b == partStart : k == 0 && b == 0;
                Compute(l0cResult, aL0, bL0, validK, initializeC);
                SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_[l0PingPongIndex]);
                if (segmented && k + b + validK == partEnd) {
                    SetFlag<HardEvent::M_FIX>(eventIdMToFix_[l0cPingPongIndex]);
                    WaitFlag<HardEvent::M_FIX>(eventIdMToFix_[l0cPingPongIndex]);
                    PipeBarrier<PIPE_FIX>();
                    if (partStart != 0) {
                        SetAtomicAdd<float>();
                    }
                    if constexpr (USE_RESULT_WORKSPACE) {
                        if (count == 1) {
                            DataCopySingleResult(gradHResGlobal, l0cResult, first);
                        } else {
                            DataCopyPartialResult(partialResultsGlobal[l0cPingPongIndex * CUBE * CUBE], l0cResult);
                        }
                    } else {
                        DataCopySingleResult(gradHResGlobal, l0cResult, first);
                    }
                    DisableDmaAtomic();
                    SetFlag<HardEvent::FIX_M>(eventIdFixToM_[l0cPingPongIndex]);
                    if (partEnd < cubeParams_.hiddenSize) {
                        WaitFlag<HardEvent::FIX_M>(eventIdFixToM_[l0cPingPongIndex]);
                    }
                    partStart = partEnd;
                    partEnd = GetKPartitionEnd(partEnd);
                }
            }
            ++l1TransferCount_;
        }

        if (nextCount != 0) {
            const uint64_t nextOffset =
                static_cast<uint64_t>(nextFirst) * cubeParams_.headCount * cubeParams_.hiddenSize;
            const uint32_t nextL1PingPongIndex = l1TransferCount_ % 2;
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_[nextL1PingPongIndex]);
            DataCopyA(xGlobal[nextOffset], 0, nextCount, nextL1PingPongIndex);
            SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_[nextL1PingPongIndex]);
        }

        if constexpr (USE_RESULT_WORKSPACE) {
            CrossCoreSetFlag<4, PIPE_FIX>(GRAD_H_RES_FIX_TO_MTE2_FLAG_BASE + l0cPingPongIndex);
            CrossCoreSetFlag<4, PIPE_FIX>(GRAD_H_RES_FIX_TO_MTE2_FLAG_BASE + AIV_SUB_BLOCK_FLAG_OFFSET +
                                          l0cPingPongIndex);
        } else if constexpr (!SEGMENTED) {
            SetFlag<HardEvent::M_FIX>(eventIdMToFix_[l0cPingPongIndex]);
            WaitFlag<HardEvent::M_FIX>(eventIdMToFix_[l0cPingPongIndex]);
            DataCopyOut(gradHResGlobal, l0cResult, first, count, CUBE);
            SetFlag<HardEvent::FIX_M>(eventIdFixToM_[l0cPingPongIndex]);
        }

        ++processedGroupCount_;
    }

private:
    static constexpr uint32_t CUBE = 16;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline uint32_t AlignCube(uint32_t a)
    {
        return (a + 15) / 16 * 16;
    }

    __aicore__ inline void DataCopyA(GlobalTensor<T> sourceTensor, uint32_t k, uint32_t count, uint32_t l1PingPongIndex)
    {
        Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = count * cubeParams_.headCount;
        copy.dValue = Min(cubeParams_.l1TileK, cubeParams_.hiddenSize - k);
        copy.srcDValue = cubeParams_.hiddenSize;
        copy.dstNzC0Stride = 16;
        copy.dstNzNStride = 1;

        DataCopy(l1Buffer_[cubeParams_.inputL1Offsets[l1PingPongIndex]], sourceTensor[k], copy);
    }

    __aicore__ inline void LoadDataA(const LocalTensor<T>& aL0, const LocalTensor<T>& aL1, uint32_t size)
    {
        LoadData2DParamsV2 load{};
        load.mStep = 1;
        load.kStep = size / 16;
        load.srcStride = 1;
        load.dstStride = 1;
        LoadData(aL0, aL1, load);
    }

    __aicore__ inline void LoadDataB(const LocalTensor<T>& bL0, const LocalTensor<T>& bL1, uint32_t size)
    {
        LoadData2DParamsV2 load{};
        load.mStep = 1;
        load.kStep = size / 16;
        load.srcStride = 1;
        load.dstStride = 1;
        LoadData(bL0, bL1, load);
    }

    __aicore__ inline uint32_t GetKPartitionEnd(uint32_t start)
    {
        const uint32_t end = Min(start + cubeParams_.kPartitionSize, cubeParams_.hiddenSize);
        if (cubeParams_.hiddenSize - end < 64) {
            return cubeParams_.hiddenSize;
        }
        return end;
    }

    __aicore__ inline void Compute(const LocalTensor<float>& l0cResult, const LocalTensor<T>& aL0,
                                   const LocalTensor<T>& bL0, uint32_t size, bool initializeC)
    {
        MmadParams params{};
        params.m = 16;
        params.n = 16;
        params.k = size;
        params.disableGemv = true;
        params.cmatrixInitVal = initializeC;
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
        params.fixShiftVal = 42;
#endif
        Mmad(l0cResult, aL0, bL0, params);
    }

    __aicore__ inline void DataCopySingleResult(const GlobalTensor<float>& destination,
                                                const LocalTensor<float>& result, uint32_t first)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> copy{};
        copy.mSize = cubeParams_.headCount;
        copy.nSize = cubeParams_.headCount;
        copy.srcStride = CUBE;
        copy.dstStride = cubeParams_.headCount;
        copy.params.ndNum = 1;
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
        copy.fixShiftVal = 16;
#endif
        static constexpr FixpipeConfig CONFIG = {CO2Layout::ROW_MAJOR, false};
        const uint64_t offset = static_cast<uint64_t>(first) * cubeParams_.headCount * cubeParams_.headCount;
        Fixpipe<float, float, CONFIG>(destination[offset], result, copy);
    }

    __aicore__ inline void DataCopyPartialResult(const GlobalTensor<float>& destination,
                                                 const LocalTensor<float>& result)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> copy{};
        copy.mSize = CUBE;
        copy.nSize = CUBE;
        copy.srcStride = CUBE;
        copy.dstStride = CUBE;
        copy.params.ndNum = 1;
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
        copy.fixShiftVal = 16;
#endif
        static constexpr FixpipeConfig CONFIG = {CO2Layout::ROW_MAJOR, false};
        Fixpipe<float, float, CONFIG>(destination, result, copy);
    }

    __aicore__ inline void DataCopyOut(const GlobalTensor<float>& gradHResGlobal,
                                       const LocalTensor<float>& sourceTensor, uint32_t first, uint32_t count,
                                       uint64_t rows)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams{};
        fixpipeParams.srcStride = rows;
        fixpipeParams.dstStride = cubeParams_.headCount;
        fixpipeParams.params = {1, static_cast<uint16_t>(cubeParams_.headCount),
                                static_cast<uint16_t>(cubeParams_.headCount)};

#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
        fixpipeParams.fixShiftVal = 16;
#endif
        static constexpr FixpipeConfig CONFIG = {CO2Layout::ROW_MAJOR, false};

        const uint64_t outputBaseOffset = static_cast<uint64_t>(first) * cubeParams_.headCount * cubeParams_.headCount;
        if (CUBE % cubeParams_.headCount == 0) {
            const uint64_t period = CUBE / cubeParams_.headCount;
            const uint64_t phases = Min(count, period);
            fixpipeParams.params.srcNdStride = rows + CUBE;
            fixpipeParams.params.dstNdStride = period * cubeParams_.headCount * cubeParams_.headCount;

            for (uint64_t phase = phases; phase-- > 0;) {
                const uint64_t prefix = phase * cubeParams_.headCount;
                fixpipeParams.params.ndNum = (count - 1 - phase) / period + 1;
                fixpipeParams.nSize = prefix + cubeParams_.headCount;

                if (phase == 0) {
                    fixpipeParams.mSize = cubeParams_.headCount;

                    Fixpipe<float, float, CONFIG>(gradHResGlobal[outputBaseOffset], sourceTensor, fixpipeParams);
                    continue;
                }

                fixpipeParams.mSize = 1;
                for (uint64_t row = cubeParams_.headCount; row-- > 0;) {
                    const uint64_t sourceOffset = (phase * cubeParams_.headCount + row) * CUBE;
                    const uint64_t destinationOffset = outputBaseOffset +
                                                       phase * cubeParams_.headCount * cubeParams_.headCount +
                                                       row * cubeParams_.headCount - prefix;
                    Fixpipe<float, float, CONFIG>(gradHResGlobal[destinationOffset], sourceTensor[sourceOffset],
                                                  fixpipeParams);
                }
            }
        } else {
            fixpipeParams.params.ndNum = 1;

            fixpipeParams.mSize = 1;
            for (uint64_t batch = count; batch-- > 0;) {
                const uint64_t columnOffset = batch * cubeParams_.headCount;
                const uint64_t prefix = columnOffset % CUBE;
                fixpipeParams.nSize = prefix + cubeParams_.headCount;
                for (uint64_t row = cubeParams_.headCount; row-- > 0;) {
                    const uint64_t sourceOffset = (columnOffset / CUBE * rows + columnOffset + row) * CUBE;
                    const uint64_t destinationOffset = outputBaseOffset +
                                                       batch * cubeParams_.headCount * cubeParams_.headCount +
                                                       row * cubeParams_.headCount - prefix;
                    Fixpipe<float, float, CONFIG>(gradHResGlobal[destinationOffset], sourceTensor[sourceOffset],
                                                  fixpipeParams);
                }
            }
        }
    }

    TPipe* pipe_{nullptr};
    MhcPostBackwardCubeParams cubeParams_;
    TEventID eventIdMte1ToMte2_[2];
    TEventID eventIdMte2ToMte1_[2];
    TEventID eventIdMToMte1_[2];
    TEventID eventIdMte1ToM_[2];
    TEventID eventIdFixToM_[2];
    TEventID eventIdMToFix_[2];
    uint32_t mmadCount_, processedGroupCount_, l1TransferCount_;
    LocalTensor<T> l1Buffer_;
    LocalTensor<T> l0aBuffer_;
    LocalTensor<T> l0bBuffer_;
    LocalTensor<float> l0cBuffer_;
};

} // namespace MhcPostBackward

#endif
