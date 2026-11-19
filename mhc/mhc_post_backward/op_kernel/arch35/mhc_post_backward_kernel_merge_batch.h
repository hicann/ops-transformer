/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MHC_POST_BACKWARD_KERNEL_MERGE_BATCH_H
#define MHC_POST_BACKWARD_KERNEL_MERGE_BATCH_H

#include <type_traits>
#include "kernel_operator.h"
#include "mhc_post_backward_tiling_data_arch35.h"
#include "mhc_post_backward_cube.h"

namespace MhcPostBackward {
using namespace AscendC;

template <typename T, bool IS_HRES, bool IS_SEGMENTED, bool USE_RESULT_WORKSPACE>
class MhcPostBackwardKernelOptimized {
    // Retain 64 sums and their 64 corrections across the complete hidden dimension.
    static constexpr uint32_t GRAD_H_POST_ACCUMULATOR_ELEMENTS = 2 * 64;
    static constexpr uint32_t GRAD_H_POST_BUFFER_NUM = 2;
    // Split FP32 values into high significand bits and an explicitly retained residual.
    static constexpr uint32_t GRAD_H_POST_HIGH_BITS_MASK = 0xffff0000U;

    static constexpr uint32_t FLOAT_BLOCK_ELEMENTS = 32 / sizeof(float);

    static constexpr Reg::CastTrait UP = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                          RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait DOWN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::CAST_RINT};

public:
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR gradX,
                                GM_ADDR gradHRes, GM_ADDR gradHOut, GM_ADDR gradHPost, GM_ADDR workspace,
                                const MhcPostBackwardOptimizedTilingDataArch35* tiling, TPipe* pipe)
    {
        tilingData_ = tiling;
        cubeParams_ = {tiling->n, tiling->D, tiling->cubeBatchGroup, tiling->cubeL1K, tiling->cubeBaseK};
        cubeParams_.kPartitionSize = tiling->cubeKPartitionSize;

        const uint32_t l1TileElements = 16 * cubeParams_.l1TileK;
        const uint32_t l1Elements = TOTAL_L1_SIZE / sizeof(T);

        // Place B0/A0 at the beginning and A1/B1 at the end of L1, leaving unused space in between.
        const uint32_t gradYL1Elements = l1TileElements;
        cubeParams_.inputL1Offsets[0] = gradYL1Elements;
        cubeParams_.inputL1Offsets[1] = l1Elements - gradYL1Elements - l1TileElements;
        cubeParams_.gradYL1Offsets[0] = 0;
        cubeParams_.gradYL1Offsets[1] = l1Elements - gradYL1Elements;

        batchGroupCount_ = DivUp(tilingData_->totalItems, cubeParams_.mergeBatch);
        coreIndex_ = GetBlockIdx();

        if ASCEND_IS_AIV {
            coreIndex_ /= 2;
        }
        const uint32_t groupsPerCore = batchGroupCount_ / tilingData_->usedAic;
        const uint32_t remainingGroups = batchGroupCount_ % tilingData_->usedAic;
        taskCount_ = groupsPerCore + (coreIndex_ < remainingGroups ? 1 : 0);
        firstGroup_ = coreIndex_ * groupsPerCore + Min(coreIndex_, remainingGroups);
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        gradYGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradY));
        hOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hOut));
        hPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hPost));
        if constexpr (IS_HRES) {
            hResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(hRes));
            hResGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradX));
        gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradHRes));
        gradHOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradHOut));
        gradHPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradHPost));

        xGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        gradYGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        hOutGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        hPostGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        if (taskCount_ == 0) {
            return;
        }

        pipe_ = pipe;
        if constexpr (USE_RESULT_WORKSPACE) {
            partialResultsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) +
                                                  coreIndex_ * 2 * 16 * 16);
        }

        if ASCEND_IS_AIC {
            cube_.Init(pipe_, cubeParams_);
        }

        if ASCEND_IS_AIV {
            InitVec();
        }
    }

    __aicore__ inline void Process()
    {
        if (taskCount_ == 0) {
            return;
        }

        if ASCEND_IS_AIC {
            ProcessCube();
            if constexpr (USE_RESULT_WORKSPACE) {
                for (uint32_t bufferIndex = 0; bufferIndex < Min(2, taskCount_); ++bufferIndex) {
                    CrossCoreWaitFlag<4, PIPE_FIX>(GRAD_H_RES_MTE2_TO_FIX_FLAG_BASE + bufferIndex);
                    CrossCoreWaitFlag<4, PIPE_FIX>(GRAD_H_RES_MTE2_TO_FIX_FLAG_BASE + AIV_SUB_BLOCK_FLAG_OFFSET +
                                                   bufferIndex);
                }
            }
        }

        if ASCEND_IS_AIV {
            ProcessVecByHeads();

            for (uint32_t gradYPingPongIndex = 0; gradYPingPongIndex < Min(2, taskCount_ * tileKCount_);
                 ++gradYPingPongIndex) {
                CrossCoreWaitFlag<4, PIPE_MTE3>(GRAD_Y_MTE1_TO_MTE3_FLAG_BASE + gradYPingPongIndex);
            }
        }
    }

private:
    __aicore__ inline void ProcessCube()
    {
        for (uint32_t task = 0; task < taskCount_; ++task) {
            const uint32_t first = (firstGroup_ + task) * cubeParams_.mergeBatch;
            const uint32_t count = Min(cubeParams_.mergeBatch, tilingData_->totalItems - first);
            const uint32_t nextFirst = first + cubeParams_.mergeBatch;
            const uint32_t nextCount =
                task + 1 < taskCount_ ? Min(cubeParams_.mergeBatch, tilingData_->totalItems - nextFirst) : 0;
            cube_.template operator()<IS_SEGMENTED, USE_RESULT_WORKSPACE>(xGlobal_, gradHResGlobal_, first, count,
                                                                          nextFirst, nextCount, partialResultsGlobal_);
        }
    }

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline uint32_t DivUp(uint32_t a, uint32_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline uint32_t Align(uint32_t a, uint32_t b)
    {
        return DivUp(a, b) * b;
    }

    __aicore__ inline void ProcessVecByHeads()
    {
        if (cubeParams_.headCount == 4) {
            ProcessVec<4>();
        } else if (cubeParams_.headCount == 6) {
            ProcessVec<6>();
        } else {
            ProcessVec<8>();
        }
    }

    __aicore__ inline void InitVec()
    {
        batchPerAiv_ = cubeParams_.mergeBatch / 2;
        tileKCount_ = DivUp(cubeParams_.hiddenSize, cubeParams_.l1TileK);
        vecTaskCount_ = taskCount_ * batchPerAiv_;
        subBlockIndex_ = GetBlockIdx() % 2;
        vecStartToken_ = firstGroup_ * cubeParams_.mergeBatch + subBlockIndex_ * batchPerAiv_;

        gradYL1_ = LocalTensor<T>{TPosition::A1, 0, TOTAL_L1_SIZE / sizeof(T)};
        const uint32_t gradYBytes = batchPerAiv_ * cubeParams_.headCount * tilingData_->tileD * sizeof(T);
        const uint32_t hOutBytes = batchPerAiv_ * tilingData_->tileD * sizeof(T);
        pipe_->InitBuffer(gradYQueue_, 2, gradYBytes);
        pipe_->InitBuffer(hOutQueue_, 2, hOutBytes);
        if constexpr (IS_HRES) {
            pipe_->InitBuffer(gradXQueue_, 2, gradYBytes);
        } else {
            eventIdGradYMte3ToMte2_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        }
        pipe_->InitBuffer(gradHOutQueue_, 2, hOutBytes);
        if constexpr (IS_HRES) {
            pipe_->InitBuffer(hResQueue_, 2,
                              batchPerAiv_ *
                                  Align(cubeParams_.headCount * cubeParams_.headCount, FLOAT_BLOCK_ELEMENTS) *
                                  sizeof(float));
        }
        pipe_->InitBuffer(hPostQueue_, 2,
                          batchPerAiv_ * Align(cubeParams_.headCount, FLOAT_BLOCK_ELEMENTS) * sizeof(float));
        hasPendingOutput_ = false;

        pipe_->InitBuffer(gradHPostAccumulatorBuffer_,
                          batchPerAiv_ * cubeParams_.headCount * GRAD_H_POST_ACCUMULATOR_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(gradHPostQueue_, GRAD_H_POST_BUFFER_NUM,
                          Align(cubeParams_.headCount, FLOAT_BLOCK_ELEMENTS) * sizeof(float));
        pipe_->InitBuffer(packedGradYQueue_, 2, cubeParams_.headCount * cubeParams_.l1TileK * sizeof(T));
        if constexpr (USE_RESULT_WORKSPACE) {
            pipe_->InitBuffer(gradHResBuffer_, cubeParams_.headCount * 32);
        }
    }

    __aicore__ inline void DataCopyInWeights(uint32_t token, uint32_t count)
    {
        if constexpr (IS_HRES) {
            auto hRes = hResQueue_.template AllocTensor<float>();
            DataCopyPad(hRes, hResGlobal_[static_cast<uint64_t>(token) * cubeParams_.headCount * cubeParams_.headCount],
                        {static_cast<uint16_t>(count),
                         static_cast<uint16_t>(cubeParams_.headCount * cubeParams_.headCount * sizeof(float)), 0, 0},
                        {true, 0, 0, 0});
            hResQueue_.EnQue(hRes);
        }
        auto hPost = hPostQueue_.template AllocTensor<float>();
        DataCopyPad(hPost, hPostGlobal_[static_cast<uint64_t>(token) * cubeParams_.headCount],
                    {static_cast<uint16_t>(count), static_cast<uint16_t>(cubeParams_.headCount * sizeof(float)), 0, 0},
                    {true, 0, 0, 0});
        hPostQueue_.EnQue(hPost);
    }

    __aicore__ inline void DataCopyIn(uint32_t token, uint32_t start, uint32_t count)
    {
        auto gradY = gradYQueue_.template AllocTensor<T>();
        auto hOut = hOutQueue_.template AllocTensor<T>();
        uint32_t size = Min(tilingData_->tileD, cubeParams_.hiddenSize - start);
        constexpr uint32_t INPUT_BLOCK_ELEMENTS = 32 / sizeof(T);
        const uint64_t gradYOffset = static_cast<uint64_t>(token) * cubeParams_.headCount * cubeParams_.hiddenSize;
        const uint64_t hOutOffset = static_cast<uint64_t>(token) * cubeParams_.hiddenSize;
        DataCopyExtParams copy{static_cast<uint16_t>(count * cubeParams_.headCount),
                               static_cast<uint32_t>(size * sizeof(T)),
                               static_cast<uint32_t>((cubeParams_.hiddenSize - size) * sizeof(T)),
                               (tilingData_->tileD - Align(size, INPUT_BLOCK_ELEMENTS)) / INPUT_BLOCK_ELEMENTS, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(Align(size, INPUT_BLOCK_ELEMENTS) - size), 0};
        DataCopyPad(gradY, gradYGlobal_[gradYOffset + start], copy, pad);
        copy.blockCount = count;
        DataCopyPad(hOut, hOutGlobal_[hOutOffset + start], copy, pad);
        gradYQueue_.EnQue(gradY);
        hOutQueue_.EnQue(hOut);
    }

    __aicore__ inline void DataCopyGradXToGM(const LocalTensor<T>& source, uint32_t token, uint32_t start,
                                             uint32_t size, uint32_t count)
    {
        constexpr uint32_t INPUT_BLOCK_ELEMENTS = 32 / sizeof(T);
        const uint64_t offset = static_cast<uint64_t>(token) * cubeParams_.headCount * cubeParams_.hiddenSize;
        DataCopyExtParams copy{static_cast<uint16_t>(count * cubeParams_.headCount),
                               static_cast<uint32_t>(size * sizeof(T)),
                               (tilingData_->tileD - Align(size, INPUT_BLOCK_ELEMENTS)) / INPUT_BLOCK_ELEMENTS,
                               static_cast<uint32_t>((cubeParams_.hiddenSize - size) * sizeof(T)), 0};
        DataCopyPad(gradXGlobal_[offset + start], source, copy);
    }

    __aicore__ inline void DataCopyOut(LocalTensor<T> gradX, LocalTensor<T> gradHOut, uint32_t token, uint32_t start,
                                       uint32_t size, uint32_t count)
    {
        constexpr uint32_t INPUT_BLOCK_ELEMENTS = 32 / sizeof(T);
        const uint64_t gradHOutOffset = static_cast<uint64_t>(token) * cubeParams_.hiddenSize;
        if constexpr (IS_HRES) {
            DataCopyGradXToGM(gradX, token, start, size, count);
        }
        DataCopyExtParams copy{static_cast<uint16_t>(count), static_cast<uint32_t>(size * sizeof(T)),
                               (tilingData_->tileD - Align(size, INPUT_BLOCK_ELEMENTS)) / INPUT_BLOCK_ELEMENTS,
                               static_cast<uint32_t>((cubeParams_.hiddenSize - size) * sizeof(T)), 0};
        DataCopyPad(gradHOutGlobal_[gradHOutOffset + start], gradHOut, copy);
    }

    template <uint32_t HEADS>
    __aicore__ inline void ProcessVec()
    {
        uint32_t token = vecStartToken_;
        uint32_t transferIndex = 0;
        uint32_t count = token < tilingData_->totalItems ? Min(batchPerAiv_, tilingData_->totalItems - token) : 0;
        if (count > 0) {
            DataCopyIn(token, 0, count);
            DataCopyInWeights(token, count);
        }

        for (uint32_t task = 0; task < taskCount_; ++task, token += cubeParams_.mergeBatch) {
            count = token < tilingData_->totalItems ? Min(batchPerAiv_, tilingData_->totalItems - token) : 0;
            LocalTensor<float> hResWeights, hPostWeights;
            auto accumulators = gradHPostAccumulatorBuffer_.template Get<float>();
            if (count > 0) {
                if constexpr (IS_HRES) {
                    hResWeights = hResQueue_.template DeQue<float>();
                }
                hPostWeights = hPostQueue_.template DeQue<float>();
                Duplicate(accumulators, 0.0f, batchPerAiv_ * HEADS * GRAD_H_POST_ACCUMULATOR_ELEMENTS);
            }

            uint32_t kStep = tilingData_->tileD;
            for (uint32_t start = 0; start < cubeParams_.hiddenSize; start += kStep, ++transferIndex) {
                uint32_t size = Min(tilingData_->tileD, cubeParams_.hiddenSize - start);
                const uint32_t gradYPingPongIndex = transferIndex % 2;
                LocalTensor<T> gradY, hOut, gradX, gradHOut;
                if (count > 0) {
                    gradY = gradYQueue_.template DeQue<T>();
                    hOut = hOutQueue_.template DeQue<T>();
                    if constexpr (IS_HRES) {
                        gradX = gradXQueue_.template AllocTensor<T>();
                    } else {
                        gradX = gradY;
                        const auto eventIdMte2ToMte3 = pipe_->FetchEventID<HardEvent::MTE2_MTE3>();
                        SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                        DataCopyGradXToGM(gradY, token, start, size, count);
                        SetFlag<HardEvent::MTE3_MTE2>(eventIdGradYMte3ToMte2_);
                    }
                    if (start + size < cubeParams_.hiddenSize) {
                        DataCopyIn(token, start + size, count);
                    } else {
                        const uint32_t nextToken = token + cubeParams_.mergeBatch;
                        if (task + 1 < taskCount_ && nextToken < tilingData_->totalItems) {
                            const uint32_t nextCount = Min(batchPerAiv_, tilingData_->totalItems - nextToken);
                            DataCopyIn(nextToken, 0, nextCount);
                            DataCopyInWeights(nextToken, nextCount);
                        }
                    }
                    gradHOut = gradHOutQueue_.template AllocTensor<T>();
                }
                if (transferIndex >= 2) {
                    CrossCoreWaitFlag<4, PIPE_MTE3>(GRAD_Y_MTE1_TO_MTE3_FLAG_BASE + gradYPingPongIndex);
                }

                for (uint32_t batch = 0; batch < count; ++batch) {
                    auto packedGradY = packedGradYQueue_.template AllocTensor<T>();
                    auto hResBatch = hResWeights;
                    if constexpr (IS_HRES) {
                        hResBatch = hResWeights[batch * Align(HEADS * HEADS, FLOAT_BLOCK_ELEMENTS)];
                    }
                    ComputeVec<HEADS>(gradY[batch * HEADS * tilingData_->tileD], hOut[batch * tilingData_->tileD],
                                      hResBatch, hPostWeights[batch * Align(HEADS, FLOAT_BLOCK_ELEMENTS)],
                                      gradX[batch * HEADS * tilingData_->tileD], gradHOut[batch * tilingData_->tileD],
                                      accumulators[batch * HEADS * GRAD_H_POST_ACCUMULATOR_ELEMENTS], packedGradY,
                                      size);
                    DataCopyPendingOutput();
                    packedGradYQueue_.EnQue(packedGradY);
                    packedGradY = packedGradYQueue_.template DeQue<T>();
                    DataCopyGradYToL1(packedGradY, size, subBlockIndex_ * batchPerAiv_ + batch, gradYPingPongIndex);
                    packedGradYQueue_.FreeTensor(packedGradY);
                }
                CrossCoreSetFlag<4, PIPE_MTE3>(GRAD_Y_MTE3_TO_MTE1_FLAG_BASE + gradYPingPongIndex);

                if (count > 0) {
                    if constexpr (!IS_HRES) {
                        WaitFlag<HardEvent::MTE3_MTE2>(eventIdGradYMte3ToMte2_);
                    }
                    gradYQueue_.FreeTensor(gradY);
                    hOutQueue_.FreeTensor(hOut);
                    if constexpr (IS_HRES) {
                        gradXQueue_.EnQue(gradX);
                    }
                    gradHOutQueue_.EnQue(gradHOut);
                    pendingOutputToken_ = token;
                    pendingOutputStart_ = start;
                    pendingOutputSize_ = size;
                    pendingOutputCount_ = count;
                    hasPendingOutput_ = true;
                }
            }
            if constexpr (USE_RESULT_WORKSPACE) {
                DataCopyGradHRes(task, token, count);
            }
            if (count > 0) {
                if constexpr (IS_HRES) {
                    hResQueue_.FreeTensor(hResWeights);
                }
                hPostQueue_.FreeTensor(hPostWeights);
                for (uint32_t batch = 0; batch < count; ++batch) {
                    ReduceGradHPost(accumulators[batch * HEADS * GRAD_H_POST_ACCUMULATOR_ELEMENTS], token + batch);
                }
            }
        }
        DataCopyPendingOutput();
        if constexpr (!IS_HRES) {
            pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdGradYMte3ToMte2_);
        }
    }

    __aicore__ inline void DataCopyGradYToL1(const LocalTensor<T>& packedGradY, uint32_t size, uint32_t batch,
                                             uint32_t gradYPingPongIndex)
    {
        DataCopyParams copy{static_cast<uint16_t>(DivUp(size, 16)), static_cast<uint16_t>(cubeParams_.headCount), 0,
                            static_cast<uint16_t>(16 - cubeParams_.headCount)};
        uint32_t batchOffset = batch * cubeParams_.headCount * 16;
        DataCopy(gradYL1_[cubeParams_.gradYL1Offsets[gradYPingPongIndex] + batchOffset], packedGradY, copy);
    }

    __aicore__ inline void DataCopyGradHRes(uint32_t task, uint32_t token, uint32_t count)
    {
        const uint32_t resultPingPongIndex = task % 2;
        CrossCoreWaitFlag<4, PIPE_MTE2>(GRAD_H_RES_FIX_TO_MTE2_FLAG_BASE + resultPingPongIndex);
        auto result = gradHResBuffer_.template Get<float>();
        const auto eventIdMte2ToMte3 = pipe_->FetchEventID<HardEvent::MTE2_MTE3>();
        const auto eventIdMte3ToMte2 = pipe_->FetchEventID<HardEvent::MTE3_MTE2>();
        const uint32_t heads = cubeParams_.headCount;
        const uint32_t groupFirst = token - subBlockIndex_ * batchPerAiv_;
        const uint32_t groupCount = Min(cubeParams_.mergeBatch, tilingData_->totalItems - groupFirst);
        const uint32_t copyCount = groupCount > 1 ? count : 0;
        for (uint32_t batch = 0; batch < copyCount; ++batch) {
            const uint32_t diagonal = (subBlockIndex_ * batchPerAiv_ + batch) * heads;
            const uint32_t offset = resultPingPongIndex * 16 * 16 + diagonal * (16 + 1);
            DataCopyExtParams inputCopy{static_cast<uint16_t>(heads), static_cast<uint32_t>(heads * sizeof(float)),
                                        static_cast<uint32_t>((16 - heads) * sizeof(float)), 0, 0};
            DataCopyPad(result, partialResultsGlobal_[offset], inputCopy, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            DataCopyExtParams outputCopy{static_cast<uint16_t>(heads), static_cast<uint32_t>(heads * sizeof(float)), 0,
                                         0, 0};
            DataCopyPad(gradHResGlobal_[static_cast<uint64_t>(token + batch) * heads * heads], result, outputCopy);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        CrossCoreSetFlag<4, PIPE_MTE2>(GRAD_H_RES_MTE2_TO_FIX_FLAG_BASE + resultPingPongIndex);
    }

    __aicore__ inline void DataCopyPendingOutput()
    {
        if (!hasPendingOutput_) {
            return;
        }

        LocalTensor<T> gradX;
        if constexpr (IS_HRES) {
            gradX = gradXQueue_.template DeQue<T>();
        }
        auto gradHOut = gradHOutQueue_.template DeQue<T>();
        DataCopyOut(gradX, gradHOut, pendingOutputToken_, pendingOutputStart_, pendingOutputSize_, pendingOutputCount_);
        if constexpr (IS_HRES) {
            gradXQueue_.FreeTensor(gradX);
        }
        gradHOutQueue_.FreeTensor(gradHOut);
        hasPendingOutput_ = false;
    }

    __simd_vf__ static inline void ReduceGradHPostVF(__ubuf__ float* sourceAddress, __ubuf__ float* destinationAddress,
                                                     uint32_t heads)
    {
        Reg::RegTensor<float> sum, correction, highPart;
        Reg::RegTensor<uint32_t> highBitsMask;
        uint32_t length = 64;
        auto mask = Reg::UpdateMask<float>(length);
        Reg::Duplicate(highBitsMask, GRAD_H_POST_HIGH_BITS_MASK, mask);
        uint32_t scalarLength = 1;
        auto scalarMask = Reg::UpdateMask<float>(scalarLength);
        for (uint16_t row = 0; row < heads; ++row) {
            Reg::LoadAlign(sum, sourceAddress + row * GRAD_H_POST_ACCUMULATOR_ELEMENTS);
            Reg::LoadAlign(correction, sourceAddress + row * GRAD_H_POST_ACCUMULATOR_ELEMENTS + 64);
            Reg::And(reinterpret_cast<Reg::RegTensor<uint32_t>&>(highPart),
                     reinterpret_cast<Reg::RegTensor<uint32_t>&>(sum), highBitsMask, mask);
            Reg::Sub(sum, sum, highPart, mask);
            Reg::Add(sum, sum, correction, mask);
            Reg::Reduce<Reg::ReduceType::SUM>(highPart, highPart, mask);
            Reg::Reduce<Reg::ReduceType::SUM>(sum, sum, mask);
            Reg::Add(sum, sum, highPart, scalarMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(destinationAddress + row, sum, scalarMask);
        }
    }

    __aicore__ inline void ReduceGradHPost(LocalTensor<float> gradHPostAccumulators, uint32_t token)
    {
        auto gradHPostLocal = gradHPostQueue_.template AllocTensor<float>();
        auto sourceAddress = (__ubuf__ float*)gradHPostAccumulators.GetPhyAddr();
        auto destinationAddress = (__ubuf__ float*)gradHPostLocal.GetPhyAddr();

        PipeBarrier<PIPE_V>();
        const uint32_t heads = cubeParams_.headCount;

        asc_vf_call<ReduceGradHPostVF>(sourceAddress, destinationAddress, heads);

        gradHPostQueue_.EnQue(gradHPostLocal);
        gradHPostLocal = gradHPostQueue_.template DeQue<float>();
        DataCopyPad(gradHPostGlobal_[static_cast<uint64_t>(token) * cubeParams_.headCount], gradHPostLocal,
                    {1, static_cast<uint16_t>(cubeParams_.headCount * 4), 0, 0});
        gradHPostQueue_.FreeTensor(gradHPostLocal);
    }

    template <uint32_t HEADS>
    __aicore__ inline void ComputeVec(LocalTensor<T> gradY, LocalTensor<T> hOut, LocalTensor<float> hResWeights,
                                      LocalTensor<float> hPostWeights, LocalTensor<T> gradX, LocalTensor<T> gradHOut,
                                      LocalTensor<float> gradHPostAccumulators, LocalTensor<T> packedGradY,
                                      uint32_t size);

    template <uint32_t HEADS>
    __simd_vf__ static inline void ComputeVecVF(__ubuf__ T* gradYAddress, __ubuf__ T* hOutAddress,
                                                __ubuf__ float* hResAddress, __ubuf__ float* hPostAddress,
                                                __ubuf__ T* gradXAddress, __ubuf__ T* gradHOutAddress,
                                                __ubuf__ float* accumulatorAddress,
                                                __ubuf__ uint16_t* packedGradYAddress, uint32_t stride, uint32_t size,
                                                uint16_t blockCount);

    const MhcPostBackwardOptimizedTilingDataArch35* tilingData_{nullptr};
    MhcPostBackwardCubeParams cubeParams_{};
    uint32_t batchGroupCount_{0}, coreIndex_{0}, taskCount_{0}, firstGroup_{0};
    TPipe* pipe_{nullptr};
    MhcPostBackwardCube<T> cube_;
    GlobalTensor<float> partialResultsGlobal_;
    TBuf<TPosition::VECCALC> gradHResBuffer_;
    uint32_t batchPerAiv_{0}, tileKCount_{0}, vecTaskCount_{0};
    uint32_t subBlockIndex_{0}, vecStartToken_{0};
    LocalTensor<T> gradYL1_;
    GlobalTensor<T> xGlobal_, gradYGlobal_, hOutGlobal_, gradXGlobal_, gradHOutGlobal_;
    GlobalTensor<float> gradHResGlobal_, hResGlobal_, hPostGlobal_, gradHPostGlobal_;

    TQue<QuePosition::VECIN, 1> gradYQueue_, hOutQueue_, hResQueue_, hPostQueue_;
    TQue<QuePosition::VECOUT, 1> gradXQueue_, gradHOutQueue_;

    TEventID eventIdGradYMte3ToMte2_{0};
    bool hasPendingOutput_{false};
    uint32_t pendingOutputToken_{0};
    uint32_t pendingOutputStart_{0}, pendingOutputSize_{0}, pendingOutputCount_{0};

    TBuf<TPosition::VECCALC> gradHPostAccumulatorBuffer_;
    TQue<QuePosition::VECOUT, 1> gradHPostQueue_;
    TQue<QuePosition::VECOUT, 1> packedGradYQueue_;
};

template <typename T, bool IS_HRES, bool IS_SEGMENTED, bool USE_RESULT_WORKSPACE>
template <uint32_t HEADS>
__aicore__ inline void MhcPostBackwardKernelOptimized<T, IS_HRES, IS_SEGMENTED, USE_RESULT_WORKSPACE>::ComputeVec(
    LocalTensor<T> gradY, LocalTensor<T> hOut, LocalTensor<float> hResWeights, LocalTensor<float> hPostWeights,
    LocalTensor<T> gradX, LocalTensor<T> gradHOut, LocalTensor<float> gradHPostAccumulators, LocalTensor<T> packedGradY,
    uint32_t size)
{
    auto gradYAddress = (__ubuf__ T*)gradY.GetPhyAddr();
    auto hOutAddress = (__ubuf__ T*)hOut.GetPhyAddr();
    auto gradXAddress = (__ubuf__ T*)gradX.GetPhyAddr();
    auto gradHOutAddress = (__ubuf__ T*)gradHOut.GetPhyAddr();
    auto hPostAddress = (__ubuf__ float*)hPostWeights.GetPhyAddr();
    __ubuf__ float* hResAddress = nullptr;
    if constexpr (IS_HRES) {
        hResAddress = (__ubuf__ float*)hResWeights.GetPhyAddr();
    }
    auto accumulatorAddress = (__ubuf__ float*)gradHPostAccumulators.GetPhyAddr();
    auto packedGradYAddress = (__ubuf__ uint16_t*)packedGradY.GetPhyAddr();
    const uint32_t stride = tilingData_->tileD;

    PipeBarrier<PIPE_V>();

    const uint16_t blockCount = static_cast<uint16_t>((size + 63) / 64);
    asc_vf_call<ComputeVecVF<HEADS>>(gradYAddress, hOutAddress, hResAddress, hPostAddress, gradXAddress,
                                     gradHOutAddress, accumulatorAddress, packedGradYAddress, stride, size, blockCount);
}

template <typename T, bool IS_HRES, bool IS_SEGMENTED, bool USE_RESULT_WORKSPACE>
template <uint32_t HEADS>
__simd_vf__ inline void MhcPostBackwardKernelOptimized<T, IS_HRES, IS_SEGMENTED, USE_RESULT_WORKSPACE>::ComputeVecVF(
    __ubuf__ T* gradYAddress, __ubuf__ T* hOutAddress, __ubuf__ float* hResAddress, __ubuf__ float* hPostAddress,
    __ubuf__ T* gradXAddress, __ubuf__ T* gradHOutAddress, __ubuf__ float* accumulatorAddress,
    __ubuf__ uint16_t* packedGradYAddress, uint32_t stride, uint32_t size, uint16_t blockCount)
{
    Reg::RegTensor<T> valueB16Reg;
    Reg::RegTensor<uint16_t> gradYBitsReg;
    Reg::RegTensor<float> gradYReg[HEADS];
    Reg::RegTensor<float> gradHPostAccumulator[HEADS];
    Reg::RegTensor<float> gradHPostCorrection[HEADS];
    Reg::RegTensor<float> sumDifference, nextGradHPostSum;
    Reg::RegTensor<float> hPostWeight[HEADS];

    Reg::RegTensor<float> hOutReg, gradHOutReg, gradXReg, hResWeightReg;
    Reg::RegTensor<float> gradXProduct;
    uint32_t bitsLength = 64;
    auto bitsMask = Reg::UpdateMask<uint16_t>(bitsLength);
    uint32_t accumulatorLength = 64;
    auto accumulatorMask = Reg::UpdateMask<float>(accumulatorLength);
#pragma unroll
    for (uint16_t head = 0; head < HEADS; ++head) {
        Reg::LoadAlign(gradHPostAccumulator[head], accumulatorAddress + head * GRAD_H_POST_ACCUMULATOR_ELEMENTS);
        Reg::LoadAlign(gradHPostCorrection[head], accumulatorAddress + head * GRAD_H_POST_ACCUMULATOR_ELEMENTS + 64);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(hPostWeight[head], hPostAddress + head);
    }

    for (uint16_t block = 0; block < blockCount; ++block) {
        uint32_t length = size - block * 64;
        auto mask = Reg::UpdateMask<float>(length);
#pragma unroll
        for (uint16_t head = 0; head < HEADS; ++head) {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(valueB16Reg, gradYAddress + head * stride + block * 64);
            Reg::Cast<float, T, UP>(gradYReg[head], valueB16Reg, mask);
            Reg::Pack<uint16_t, uint32_t>(gradYBitsReg, reinterpret_cast<Reg::RegTensor<uint32_t>&>(valueB16Reg));
            Reg::StoreAlign<uint16_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(
                packedGradYAddress + block * HEADS * 64 + head * 16, gradYBitsReg, HEADS, bitsMask);
        }

        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(valueB16Reg, hOutAddress + block * 64);
        Reg::Cast<float, T, UP>(hOutReg, valueB16Reg, mask);

        Reg::Mul(gradHOutReg, gradYReg[0], hPostWeight[0], mask);
#pragma unroll
        for (uint16_t head = 1; head < HEADS; ++head) {
            Reg::MulAddDst(gradHOutReg, gradYReg[head], hPostWeight[head], mask);
        }
        Reg::Cast<T, float, DOWN>(valueB16Reg, gradHOutReg, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(gradHOutAddress + block * 64, valueB16Reg, mask);

#pragma unroll
        for (uint16_t row = 0; row < HEADS; ++row) {
            Reg::MulAddDst(gradHPostCorrection[row], gradYReg[row], hOutReg, accumulatorMask);
            Reg::Add(nextGradHPostSum, gradHPostAccumulator[row], gradHPostCorrection[row], accumulatorMask);
            Reg::Sub(sumDifference, gradHPostAccumulator[row], nextGradHPostSum, accumulatorMask);
            Reg::Add(gradHPostCorrection[row], sumDifference, gradHPostCorrection[row], accumulatorMask);
            Reg::Move(gradHPostAccumulator[row], nextGradHPostSum);

            if constexpr (IS_HRES) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(hResWeightReg, hResAddress + row * HEADS);
                Reg::Mul(gradXReg, gradYReg[0], hResWeightReg, mask);
#pragma unroll
                for (uint16_t col = 1; col < HEADS; ++col) {
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(hResWeightReg, hResAddress + row * HEADS + col);
                    if constexpr (std::is_same<T, half>::value) {
                        Reg::Mul(gradXProduct, gradYReg[col], hResWeightReg, mask);
                        Reg::Add(gradXReg, gradXReg, gradXProduct, mask);
                    } else {
                        Reg::MulAddDst(gradXReg, gradYReg[col], hResWeightReg, mask);
                    }
                }

                Reg::Cast<T, float, DOWN>(valueB16Reg, gradXReg, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(gradXAddress + row * stride + block * 64, valueB16Reg,
                                                                  mask);
            }
        }
    }

#pragma unroll
    for (uint16_t head = 0; head < HEADS; ++head) {
        Reg::StoreAlign(accumulatorAddress + head * GRAD_H_POST_ACCUMULATOR_ELEMENTS, gradHPostAccumulator[head],
                        accumulatorMask);
        Reg::StoreAlign(accumulatorAddress + head * GRAD_H_POST_ACCUMULATOR_ELEMENTS + 64, gradHPostCorrection[head],
                        accumulatorMask);
    }
}

} // namespace MhcPostBackward

#endif
