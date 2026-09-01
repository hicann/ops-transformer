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
 * \file int4_tensor_to_int8_preprocess.h
 * \brief Expand packed signed int4 grouped tensors to int8 workspace.
 *
 * This kernel is layout-driven and semantically neutral: it handles any
 * grouped [G, outerDim, innerDim] int4 tensor (weight or activation).
 * ND tensors are expanded linearly. NZ tensors require an additional
 * physical-layout conversion because the packed int4 inner axis contains 64
 * elements while the int8 workspace inner axis contains 32 elements.
 * The transposed-NZ branch is only used by the V5 weight special format
 * ([E, N, K] int4 NZ -> [E, K, N] int8 NZ); other callers keep it disabled.
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

template <class Layout_>
class Int4TensorToInt8Preprocess {
public:
    using Layout = Layout_;

    struct Params {
        GM_ADDR srcInt4GmAddr{nullptr};
        GM_ADDR dstInt8GmAddr{nullptr};
        uint64_t groupNum{0};
        uint64_t outerDim{0};
        uint64_t innerDim{0};
        // Only meaningful for the V5 weight special format, which stores
        // logical [E, N, K] as INT4 NZ and needs to be expanded into the
        // standard INT8 [E, K, N] NZ layout. Non-weight callers pass false.
        bool inputTransposedNz{false};
    };

    __aicore__ inline Int4TensorToInt8Preprocess() {}
    __aicore__ inline ~Int4TensorToInt8Preprocess() {}

    __aicore__ inline void Init(const Params &params, AscendC::TPipe *pipe)
    {
        groupNum_ = params.groupNum;
        outerDim_ = params.outerDim;
        innerDim_ = params.innerDim;
        inputTransposedNz_ = params.inputTransposedNz;
        if constexpr (SRC_NZ) {
            nzOutputWideBlockCount_ = CeilDiv(outerDim_, static_cast<uint64_t>(NZ_INT8_INNER_AXIS));
            nzOutputNarrowBlockCount_ = CeilDiv(innerDim_, static_cast<uint64_t>(NZ_NARROW_AXIS));
            if (inputTransposedNz_) {
                nzInputWideBlockCount_ = CeilDiv(innerDim_, static_cast<uint64_t>(NZ_INT4_INNER_AXIS));
                nzInputNarrowBlockCount_ = CeilDiv(outerDim_, static_cast<uint64_t>(NZ_NARROW_AXIS));
                workItemCount_ = groupNum_ * nzInputWideBlockCount_ * nzOutputWideBlockCount_;
            } else {
                nzInputWideBlockCount_ = CeilDiv(outerDim_, static_cast<uint64_t>(NZ_INT4_INNER_AXIS));
                nzInputNarrowBlockCount_ = CeilDiv(innerDim_, static_cast<uint64_t>(NZ_NARROW_AXIS));
                workItemCount_ = groupNum_ * nzInputWideBlockCount_ * nzInputNarrowBlockCount_;
            }
        } else {
            elementCount_ = groupNum_ * outerDim_ * innerDim_;
            workItemCount_ = CeilDiv(elementCount_, static_cast<uint64_t>(LINEAR_TILE_ELEMENTS));
        }
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(params.srcInt4GmAddr));
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(params.dstInt8GmAddr));
        if ASCEND_IS_AIV {
            pipe->InitBuffer(srcQueue_, 1, LOCAL_TILE_ELEMENTS / INT4_ELEMENTS_PER_BYTE);
            pipe->InitBuffer(outputQueue_, 1, LOCAL_TILE_ELEMENTS * sizeof(int8_t));
            pipe->InitBuffer(castBuffer_, LOCAL_TILE_ELEMENTS * sizeof(half));
            if constexpr (SRC_NZ) {
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
            if constexpr (SRC_NZ) {
                if (inputTransposedNz_) {
                    ConvertTransposedNzTile(workItemIdx);
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
    static constexpr bool SRC_NZ = IsWeightNz<Layout>::value;
    static constexpr bool TRANS_SRC = IsTrans<Layout>::value;
    static constexpr uint32_t LOCAL_TILE_ELEMENTS = SRC_NZ ? 2 * NZ_INPUT_TILE_ELEMENTS : LINEAR_TILE_ELEMENTS;

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
        auto srcLocal = srcQueue_.AllocTensor<int4b_t>();
        const AscendC::DataCopyExtParams copyInParams{
            1, static_cast<uint32_t>(CeilDiv(actualElements, INT4_ELEMENTS_PER_BYTE)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<int4b_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(srcLocal, srcGm_[elementOffset], copyInParams, padParams);
        srcQueue_.EnQue(srcLocal);

        srcLocal = srcQueue_.DeQue<int4b_t>();
        auto outputLocal = outputQueue_.AllocTensor<int8_t>();
        auto castLocal = castBuffer_.Get<half>();
        AscendC::Cast(castLocal, srcLocal, AscendC::RoundMode::CAST_NONE, alignedElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outputLocal, castLocal, AscendC::RoundMode::CAST_NONE, alignedElements);
        outputQueue_.EnQue(outputLocal);
        srcQueue_.FreeTensor(srcLocal);

        outputLocal = outputQueue_.DeQue<int8_t>();
        const AscendC::DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(actualElements * sizeof(int8_t)), 0, 0,
                                                       0};
        AscendC::DataCopyPad(dstGm_[elementOffset], outputLocal, copyOutParams);
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

        auto srcLocal = srcQueue_.AllocTensor<int4b_t>();
        const AscendC::DataCopyExtParams copyInParams{1, NZ_INPUT_TILE_ELEMENTS / INT4_ELEMENTS_PER_BYTE, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<int4b_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(srcLocal, srcGm_[inputElementOffset], copyInParams, padParams);
        srcQueue_.EnQue(srcLocal);

        srcLocal = srcQueue_.DeQue<int4b_t>();
        auto outputLocal = outputQueue_.AllocTensor<int8_t>();
        auto castLocal = castBuffer_.Get<half>();
        AscendC::Cast(castLocal, srcLocal, AscendC::RoundMode::CAST_NONE, NZ_INPUT_TILE_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outputLocal, castLocal, AscendC::RoundMode::CAST_NONE, NZ_INPUT_TILE_ELEMENTS);
        outputQueue_.EnQue(outputLocal);
        srcQueue_.FreeTensor(srcLocal);

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
        const uint64_t inputInner64BlockIdx = blockInGroup / nzOutputWideBlockCount_;
        const uint64_t outputOuter32BlockIdx = blockInGroup % nzOutputWideBlockCount_;
        const uint64_t firstInputOuter16BlockIdx = outputOuter32BlockIdx * 2;
        const uint32_t validOuterHalves =
            static_cast<uint32_t>(Min(nzInputNarrowBlockCount_ - firstInputOuter16BlockIdx, static_cast<uint64_t>(2)));

        auto srcLocal = srcQueue_.AllocTensor<int4b_t>();
        const AscendC::DataCopyExtParams copyInParams{1, NZ_INPUT_TILE_ELEMENTS / INT4_ELEMENTS_PER_BYTE, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<int4b_t> padParams{false, 0, 0, 0};
        for (uint32_t halfIdx = 0; halfIdx < validOuterHalves; ++halfIdx) {
            const uint64_t inputElementOffset =
                ((groupIdx * nzInputWideBlockCount_ + inputInner64BlockIdx) * nzInputNarrowBlockCount_ +
                 firstInputOuter16BlockIdx + halfIdx) *
                NZ_INPUT_TILE_ELEMENTS;
            AscendC::DataCopyPad(srcLocal[halfIdx * NZ_INPUT_TILE_ELEMENTS], srcGm_[inputElementOffset], copyInParams,
                                 padParams);
        }
        srcQueue_.EnQue(srcLocal);

        srcLocal = srcQueue_.DeQue<int4b_t>();
        auto sourceLocal = outputQueue_.AllocTensor<int8_t>();
        auto castLocal = castBuffer_.Get<half>();
        const uint32_t castElements = validOuterHalves * NZ_INPUT_TILE_ELEMENTS;
        AscendC::Cast(castLocal, srcLocal, AscendC::RoundMode::CAST_NONE, castElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(sourceLocal, castLocal, AscendC::RoundMode::CAST_NONE, castElements);
        outputQueue_.EnQue(sourceLocal);
        srcQueue_.FreeTensor(srcLocal);

        sourceLocal = outputQueue_.DeQue<int8_t>();
        auto outputLocal = nzTransposeBuffer_.Get<int8_t>();
        const uint64_t firstInner = inputInner64BlockIdx * NZ_INT4_INNER_AXIS;
        for (uint32_t innerPart = 0; innerPart < NZ_INT4_INNER_AXIS / NZ_NARROW_AXIS; ++innerPart) {
            const uint64_t outputInner16BlockIdx = inputInner64BlockIdx * 4 + innerPart;
            if (outputInner16BlockIdx >= nzOutputNarrowBlockCount_) {
                break;
            }
            AscendC::Duplicate(outputLocal, static_cast<int8_t>(0), NZ_OUTPUT_TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            const uint32_t validInner = static_cast<uint32_t>(
                Min(innerDim_ - firstInner - innerPart * NZ_NARROW_AXIS, static_cast<uint64_t>(NZ_NARROW_AXIS)));
            const uint64_t firstOuter = outputOuter32BlockIdx * NZ_INT8_INNER_AXIS;
            const uint32_t validOuter =
                static_cast<uint32_t>(Min(outerDim_ - firstOuter, static_cast<uint64_t>(NZ_INT8_INNER_AXIS)));
            for (uint32_t innerIdx = 0; innerIdx < validInner; ++innerIdx) {
                for (uint32_t outerIdx = 0; outerIdx < validOuter; ++outerIdx) {
                    const uint32_t halfIdx = outerIdx / NZ_NARROW_AXIS;
                    const uint32_t outerInHalf = outerIdx % NZ_NARROW_AXIS;
                    const uint32_t sourceOffset = halfIdx * NZ_INPUT_TILE_ELEMENTS + outerInHalf * NZ_INT4_INNER_AXIS +
                                                  innerPart * NZ_NARROW_AXIS + innerIdx;
                    outputLocal.SetValue(innerIdx * NZ_INT8_INNER_AXIS + outerIdx, sourceLocal.GetValue(sourceOffset));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            const uint64_t outputElementOffset =
                ((groupIdx * nzOutputWideBlockCount_ + outputOuter32BlockIdx) * nzOutputNarrowBlockCount_ +
                 outputInner16BlockIdx) *
                NZ_OUTPUT_TILE_ELEMENTS;
            const AscendC::DataCopyExtParams copyOutParams{1, NZ_OUTPUT_TILE_ELEMENTS * sizeof(int8_t), 0, 0, 0};
            AscendC::DataCopyPad(dstGm_[outputElementOffset], outputLocal, copyOutParams);
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
        AscendC::DataCopyPad(dstGm_[outputElementOffset], outputLocal[localElementOffset], copyOutParams);
    }

    AscendC::GlobalTensor<int4b_t> srcGm_;
    AscendC::GlobalTensor<int8_t> dstGm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> srcQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> nzTransposeBuffer_;
    uint64_t groupNum_{0};
    uint64_t outerDim_{0};
    uint64_t innerDim_{0};
    uint64_t elementCount_{0};
    uint64_t workItemCount_{0};
    uint64_t nzInputWideBlockCount_{0};
    uint64_t nzInputNarrowBlockCount_{0};
    uint64_t nzOutputWideBlockCount_{0};
    uint64_t nzOutputNarrowBlockCount_{0};
    bool inputTransposedNz_{false};
};

} // namespace GROUPED_MATMUL::INT4_PREPROCESS
