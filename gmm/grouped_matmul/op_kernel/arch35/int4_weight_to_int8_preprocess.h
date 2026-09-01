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
 * \file int4_weight_to_int8_preprocess.h
 * \brief Expand packed signed int4 grouped weights to int8 workspace.
 *
 * ND weights are expanded linearly. The A5 normal KN-NZ INT4Pack path already
 * uses C0=32. Normal KN-NZ is expanded linearly because its block order already
 * matches the INT8 workspace; special ENK-NZ additionally transposes the C0=32
 * blocks into KN-NZ. Legacy C0=64 inputs retain their original conversions.
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include <kernel_basic_intf.h>
#else
#include <kernel_operator.h>
#include <kernel_operator_intf.h>
#endif
#include <blaze/gemm/utils/common_utils.h>
#include <blaze/gemm/utils/layout_utils.h>

namespace GROUPED_MATMUL::INT4_PREPROCESS {

using AscendC::int4b_t;
using Blaze::Gemm::IsTrans;
using Blaze::Gemm::IsWeightNz;

template <class LayoutB_>
class Int4WeightToInt8Preprocess {
public:
    using LayoutB = LayoutB_;

    struct Params {
        GM_ADDR weightGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t groupNum{0};
        uint64_t n{0};
        uint64_t k{0};
        // The V5 special format stores logical [E, N, K] as INT4 NZ. The
        // expanded workspace is always the standard INT8 [E, K, N] NZ layout.
        bool inputTransposedNz{false};
        // A5 npu_format_cast(INT8 NZ) + npu_convert_weight_to_int4pack produces
        // an INT32 carrier with C0=4. ACLNN reinterprets it as logical INT4
        // C0=32 without moving bytes. This applies to both normal KN-NZ and
        // special ENK-NZ inputs.
        bool inputNzC032{false};
    };

    __aicore__ inline Int4WeightToInt8Preprocess() {}
    __aicore__ inline ~Int4WeightToInt8Preprocess() {}

    __aicore__ inline void Init(const Params &params, AscendC::TPipe *pipe)
    {
        groupNum_ = params.groupNum;
        n_ = params.n;
        k_ = params.k;
        inputTransposedNz_ = params.inputTransposedNz;
        inputNzC032_ = params.inputNzC032;
        if constexpr (WEIGHT_NZ) {
            nzInputInnerAxis_ = inputNzC032_ ? NZ_INT8_INNER_AXIS : NZ_INT4_INNER_AXIS;
            nzOutputWideBlockCount_ = CeilDiv(n_, static_cast<uint64_t>(NZ_INT8_INNER_AXIS));
            nzOutputNarrowBlockCount_ = CeilDiv(k_, static_cast<uint64_t>(NZ_NARROW_AXIS));
            if (inputTransposedNz_) {
                nzInputWideBlockCount_ = CeilDiv(k_, static_cast<uint64_t>(nzInputInnerAxis_));
                nzInputNarrowBlockCount_ = CeilDiv(n_, static_cast<uint64_t>(NZ_NARROW_AXIS));
                workItemCount_ = groupNum_ * nzInputWideBlockCount_ * nzOutputWideBlockCount_;
            } else if (inputNzC032_) {
                elementCount_ =
                    groupNum_ * nzOutputWideBlockCount_ * nzOutputNarrowBlockCount_ * NZ_OUTPUT_TILE_ELEMENTS;
                workItemCount_ = CeilDiv(elementCount_, static_cast<uint64_t>(NZ_LINEAR_TILE_ELEMENTS));
            } else {
                nzInputWideBlockCount_ = CeilDiv(n_, static_cast<uint64_t>(NZ_INT4_INNER_AXIS));
                nzInputNarrowBlockCount_ = CeilDiv(k_, static_cast<uint64_t>(NZ_NARROW_AXIS));
                workItemCount_ = groupNum_ * nzInputWideBlockCount_ * nzInputNarrowBlockCount_;
            }
        } else {
            elementCount_ = groupNum_ * n_ * k_;
            workItemCount_ = CeilDiv(elementCount_, static_cast<uint64_t>(LINEAR_TILE_ELEMENTS));
        }
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(params.weightGmAddr));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(params.workspaceGmAddr));
        if ASCEND_IS_AIV {
            pipe->InitBuffer(weightQueue_, 1, LOCAL_TILE_ELEMENTS / INT4_ELEMENTS_PER_BYTE);
            pipe->InitBuffer(outputQueue_, 1, LOCAL_TILE_ELEMENTS * sizeof(int8_t));
            pipe->InitBuffer(castBuffer_, LOCAL_TILE_ELEMENTS * sizeof(half));
            if constexpr (WEIGHT_NZ) {
                pipe->InitBuffer(nzTransposeBuffer_, NZ_OUTPUT_TILE_ELEMENTS * sizeof(int8_t));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (workItemCount_ == 0) {
            return;
        }

        const uint64_t taskIdx = AscendC::GetBlockIdx();
        const uint64_t taskCount = static_cast<uint64_t>(AscendC::GetBlockNum()) * AscendC::GetTaskRation();
        if (taskCount == 0) {
            return;
        }
        for (uint64_t workItemIdx = taskIdx; workItemIdx < workItemCount_; workItemIdx += taskCount) {
            if constexpr (WEIGHT_NZ) {
                if (inputTransposedNz_) {
                    ConvertTransposedNzTile(workItemIdx);
                } else if (inputNzC032_) {
                    const uint64_t elementOffset = workItemIdx * NZ_LINEAR_TILE_ELEMENTS;
                    const uint32_t actualElements = static_cast<uint32_t>(
                        Min(elementCount_ - elementOffset, static_cast<uint64_t>(NZ_LINEAR_TILE_ELEMENTS)));
                    ConvertLinearTile(elementOffset, actualElements);
                } else {
                    ConvertNzTile(workItemIdx);
                }
            } else {
                const uint64_t elementOffset = workItemIdx * LINEAR_TILE_ELEMENTS;
                const uint32_t actualElements = static_cast<uint32_t>(
                    Min(elementCount_ - elementOffset, static_cast<uint64_t>(LINEAR_TILE_ELEMENTS)));
                ConvertLinearTile(elementOffset, actualElements);
            }
        }
    }

    __aicore__ inline void operator()(const Params &params, AscendC::TPipe *pipe)
    {
        Init(params, pipe);
        Process();
    }

private:
    static constexpr uint32_t DATA_BLOCK_BYTES = 32;
    static constexpr uint32_t INT4_ELEMENTS_PER_BYTE = 2;
    static constexpr uint32_t INT4_ELEMENTS_PER_BLOCK = DATA_BLOCK_BYTES * INT4_ELEMENTS_PER_BYTE;
    static constexpr uint32_t LINEAR_TILE_ELEMENTS = 24 * 1024;
    static constexpr uint32_t NZ_NARROW_AXIS = 16;
    static constexpr uint32_t NZ_INT4_INNER_AXIS = 64;
    static constexpr uint32_t NZ_INT8_INNER_AXIS = 32;
    static constexpr uint32_t NZ_INPUT_TILE_ELEMENTS = NZ_NARROW_AXIS * NZ_INT4_INNER_AXIS;
    static constexpr uint32_t NZ_OUTPUT_TILE_ELEMENTS = NZ_NARROW_AXIS * NZ_INT8_INNER_AXIS;
    static constexpr uint32_t NZ_LINEAR_TILE_ELEMENTS = 2 * NZ_INPUT_TILE_ELEMENTS;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr uint32_t LOCAL_TILE_ELEMENTS = WEIGHT_NZ ? NZ_LINEAR_TILE_ELEMENTS : LINEAR_TILE_ELEMENTS;

    template <class T>
    __aicore__ inline static T Min(T lhs, T rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    template <class T>
    __aicore__ inline static T CeilDiv(T value, T divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    __aicore__ inline void ConvertLinearTile(uint64_t elementOffset, uint32_t actualElements)
    {
        const uint32_t alignedElements =
            static_cast<uint32_t>(CeilDiv(actualElements, INT4_ELEMENTS_PER_BLOCK) * INT4_ELEMENTS_PER_BLOCK);
        auto weightLocal = weightQueue_.AllocTensor<int4b_t>();
        const AscendC::DataCopyExtParams copyInParams{
            1, static_cast<uint32_t>(CeilDiv(actualElements, INT4_ELEMENTS_PER_BYTE)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<int4b_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(weightLocal, weightGm_[elementOffset], copyInParams, padParams);
        weightQueue_.EnQue(weightLocal);

        weightLocal = weightQueue_.DeQue<int4b_t>();
        auto outputLocal = outputQueue_.AllocTensor<int8_t>();
        auto castLocal = castBuffer_.Get<half>();
        AscendC::Cast(castLocal, weightLocal, AscendC::RoundMode::CAST_NONE, alignedElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outputLocal, castLocal, AscendC::RoundMode::CAST_NONE, alignedElements);
        outputQueue_.EnQue(outputLocal);
        weightQueue_.FreeTensor(weightLocal);

        outputLocal = outputQueue_.DeQue<int8_t>();
        const AscendC::DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(actualElements * sizeof(int8_t)), 0, 0,
                                                       0};
        AscendC::DataCopyPad(workspaceGm_[elementOffset], outputLocal, copyOutParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void ConvertNzTile(uint64_t workItemIdx)
    {
        const uint64_t blocksPerGroup = nzInputWideBlockCount_ * nzInputNarrowBlockCount_;
        const uint64_t groupIdx = workItemIdx / blocksPerGroup;
        const uint64_t blockInGroup = workItemIdx % blocksPerGroup;
        const uint64_t wideBlockIdx = blockInGroup / nzInputNarrowBlockCount_;
        const uint64_t narrowBlockIdx = blockInGroup % nzInputNarrowBlockCount_;
        const uint64_t inputElementOffset =
            ((groupIdx * nzInputWideBlockCount_ + wideBlockIdx) * nzInputNarrowBlockCount_ + narrowBlockIdx) *
            NZ_INPUT_TILE_ELEMENTS;

        auto weightLocal = weightQueue_.AllocTensor<int4b_t>();
        const AscendC::DataCopyExtParams copyInParams{1, NZ_INPUT_TILE_ELEMENTS / INT4_ELEMENTS_PER_BYTE, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<int4b_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(weightLocal, weightGm_[inputElementOffset], copyInParams, padParams);
        weightQueue_.EnQue(weightLocal);

        weightLocal = weightQueue_.DeQue<int4b_t>();
        auto outputLocal = outputQueue_.AllocTensor<int8_t>();
        auto castLocal = castBuffer_.Get<half>();
        AscendC::Cast(castLocal, weightLocal, AscendC::RoundMode::CAST_NONE, NZ_INPUT_TILE_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outputLocal, castLocal, AscendC::RoundMode::CAST_NONE, NZ_INPUT_TILE_ELEMENTS);
        outputQueue_.EnQue(outputLocal);
        weightQueue_.FreeTensor(weightLocal);

        outputLocal = outputQueue_.DeQue<int8_t>();
        const uint64_t firstOutputWideBlock = wideBlockIdx * 2;
        CopyNzHalfToGm(outputLocal, groupIdx, narrowBlockIdx, firstOutputWideBlock, 0);
        if (firstOutputWideBlock + 1 < nzOutputWideBlockCount_) {
            CopyNzHalfToGm(outputLocal, groupIdx, narrowBlockIdx, firstOutputWideBlock + 1, NZ_INT8_INNER_AXIS);
        }
        outputQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void ConvertTransposedNzTile(uint64_t workItemIdx)
    {
        const uint64_t blocksPerGroup = nzInputWideBlockCount_ * nzOutputWideBlockCount_;
        const uint64_t groupIdx = workItemIdx / blocksPerGroup;
        const uint64_t blockInGroup = workItemIdx % blocksPerGroup;
        const uint64_t inputKBlockIdx = blockInGroup / nzOutputWideBlockCount_;
        const uint64_t outputN32BlockIdx = blockInGroup % nzOutputWideBlockCount_;
        const uint64_t firstInputN16BlockIdx = outputN32BlockIdx * 2;
        const uint32_t validNHalves =
            static_cast<uint32_t>(Min(nzInputNarrowBlockCount_ - firstInputN16BlockIdx, static_cast<uint64_t>(2)));

        auto weightLocal = weightQueue_.AllocTensor<int4b_t>();
        const uint32_t inputTileElements = NZ_NARROW_AXIS * nzInputInnerAxis_;
        const AscendC::DataCopyExtParams copyInParams{1, inputTileElements / INT4_ELEMENTS_PER_BYTE, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<int4b_t> padParams{false, 0, 0, 0};
        for (uint32_t halfIdx = 0; halfIdx < validNHalves; ++halfIdx) {
            const uint64_t inputElementOffset =
                ((groupIdx * nzInputWideBlockCount_ + inputKBlockIdx) * nzInputNarrowBlockCount_ +
                 firstInputN16BlockIdx + halfIdx) *
                inputTileElements;
            AscendC::DataCopyPad(weightLocal[halfIdx * inputTileElements], weightGm_[inputElementOffset], copyInParams,
                                 padParams);
        }
        weightQueue_.EnQue(weightLocal);

        weightLocal = weightQueue_.DeQue<int4b_t>();
        auto sourceLocal = outputQueue_.AllocTensor<int8_t>();
        auto castLocal = castBuffer_.Get<half>();
        const uint32_t castElements = validNHalves * inputTileElements;
        AscendC::Cast(castLocal, weightLocal, AscendC::RoundMode::CAST_NONE, castElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(sourceLocal, castLocal, AscendC::RoundMode::CAST_NONE, castElements);
        outputQueue_.EnQue(sourceLocal);
        weightQueue_.FreeTensor(weightLocal);

        sourceLocal = outputQueue_.DeQue<int8_t>();
        auto outputLocal = nzTransposeBuffer_.Get<int8_t>();
        const uint64_t firstK = inputKBlockIdx * nzInputInnerAxis_;
        const uint32_t kPartCount = nzInputInnerAxis_ / NZ_NARROW_AXIS;
        for (uint32_t kPart = 0; kPart < kPartCount; ++kPart) {
            const uint64_t outputK16BlockIdx = inputKBlockIdx * kPartCount + kPart;
            if (outputK16BlockIdx >= nzOutputNarrowBlockCount_) {
                break;
            }
            AscendC::Duplicate(outputLocal, static_cast<int8_t>(0), NZ_OUTPUT_TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            const uint32_t validK =
                static_cast<uint32_t>(Min(k_ - firstK - kPart * NZ_NARROW_AXIS, static_cast<uint64_t>(NZ_NARROW_AXIS)));
            const uint64_t firstN = outputN32BlockIdx * NZ_INT8_INNER_AXIS;
            const uint32_t validN = static_cast<uint32_t>(Min(n_ - firstN, static_cast<uint64_t>(NZ_INT8_INNER_AXIS)));
            for (uint32_t kIdx = 0; kIdx < validK; ++kIdx) {
                for (uint32_t nIdx = 0; nIdx < validN; ++nIdx) {
                    const uint32_t halfIdx = nIdx / NZ_NARROW_AXIS;
                    const uint32_t nInHalf = nIdx % NZ_NARROW_AXIS;
                    const uint32_t sourceOffset =
                        halfIdx * inputTileElements + nInHalf * nzInputInnerAxis_ + kPart * NZ_NARROW_AXIS + kIdx;
                    outputLocal.SetValue(kIdx * NZ_INT8_INNER_AXIS + nIdx, sourceLocal.GetValue(sourceOffset));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            const uint64_t outputElementOffset =
                ((groupIdx * nzOutputWideBlockCount_ + outputN32BlockIdx) * nzOutputNarrowBlockCount_ +
                 outputK16BlockIdx) *
                NZ_OUTPUT_TILE_ELEMENTS;
            const AscendC::DataCopyExtParams copyOutParams{1, NZ_OUTPUT_TILE_ELEMENTS * sizeof(int8_t), 0, 0, 0};
            AscendC::DataCopyPad(workspaceGm_[outputElementOffset], outputLocal, copyOutParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
        outputQueue_.FreeTensor(sourceLocal);
    }

    __aicore__ inline void CopyNzHalfToGm(AscendC::LocalTensor<int8_t> outputLocal, uint64_t groupIdx,
                                          uint64_t narrowBlockIdx, uint64_t outputWideBlockIdx,
                                          uint32_t localElementOffset)
    {
        const uint64_t outputElementOffset =
            ((groupIdx * nzOutputWideBlockCount_ + outputWideBlockIdx) * nzOutputNarrowBlockCount_ + narrowBlockIdx) *
            NZ_OUTPUT_TILE_ELEMENTS;
        // Each source row contains 64 int8 values. Copy one 32-value half
        // from all 16 rows into one contiguous int8 NZ block.
        const AscendC::DataCopyExtParams copyOutParams{NZ_NARROW_AXIS, NZ_INT8_INNER_AXIS * sizeof(int8_t), 1, 0, 0};
        AscendC::DataCopyPad(workspaceGm_[outputElementOffset], outputLocal[localElementOffset], copyOutParams);
    }

    AscendC::GlobalTensor<int4b_t> weightGm_;
    AscendC::GlobalTensor<int8_t> workspaceGm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> weightQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> nzTransposeBuffer_;
    uint64_t groupNum_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t elementCount_{0};
    uint64_t workItemCount_{0};
    uint64_t nzInputWideBlockCount_{0};
    uint64_t nzInputNarrowBlockCount_{0};
    uint64_t nzOutputWideBlockCount_{0};
    uint64_t nzOutputNarrowBlockCount_{0};
    uint32_t nzInputInnerAxis_{NZ_INT4_INNER_AXIS};
    bool inputTransposedNz_{false};
    bool inputNzC032_{false};
};

} // namespace GROUPED_MATMUL::INT4_PREPROCESS
