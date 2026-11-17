/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP4_FD_COMBINE_ARCH35_H
#define GBSA_MXFP4_FD_COMBINE_ARCH35_H

#include "gbsa_layout_utils.hpp"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "../attn_infra/epilogue/block/mxfp4_vf/vf_fd_combine_mxfp4.h"

namespace GbsaKernelArch35 {

template <typename OutputT, Format QFormat>
class GbsaMxfp4FdCombineArch35 {
public:
    uint32_t D;
    using Resource = NpuArch::Arch::Resource<NpuArch::Arch::AtlasA5>;
    static constexpr uint32_t SPLITS = GbsaMetadata::MAX_FD_PARTIAL_PER_BASE_TASK;
    static constexpr uint32_t STAT_BYTES = SPLITS * 8U * sizeof(float);
    __aicore__ inline explicit GbsaMxfp4FdCombineArch35(Resource& resource, uint32_t dim)
        : D(dim)
    {
        maximum = resource.ubBuf.template GetBufferByByte<float>(0U);
        sum = resource.ubBuf.template GetBufferByByte<float>(STAT_BYTES);
        numerator = resource.ubBuf.template GetBufferByByte<float>(2U * STAT_BYTES);
        output = resource.ubBuf.template GetBufferByByte<OutputT>(2U * STAT_BYTES);
        lse = resource.ubBuf.template GetBufferByByte<float>(2U * STAT_BYTES + SPLITS * D * sizeof(float));
    }

    __aicore__ inline void operator()(const Mxfp4::KernelParams& params,
                                      const GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData& tiling,
                                      __gm__ const GbsaMetadata::Metadata* meta)
    {
        const bool outputLse = tiling.returnSoftmaxLse != 0U;
        AscendC::GlobalTensor<float> partialN, partialStats, globalLse;
        AscendC::GlobalTensor<OutputT> globalOutput;
        AscendC::GlobalTensor<int64_t> cuQ;
        AscendC::GlobalTensor<int32_t> usedQ;
        partialN.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace + tiling.fdPartialOOffset));
        partialStats.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace + tiling.fdPartialLseOffset));
        globalOutput.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(params.o));
        globalLse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.softmaxLse));
        cuQ.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.cuSeqLengths));
        usedQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.sequsedQ));
        const uint32_t aiv = AscendC::GetBlockIdx();
        const uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        const uint32_t group = tiling.groupSize;
        const uint32_t rows0 = (group + 1U) / 2U;
        const uint32_t statsStride = 4U * tiling.fdLseSubStride;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        for (uint32_t work = aiv; work < static_cast<uint32_t>(meta->combineTaskNum) * group; work += aivNum) {
            const auto& schedule = meta->combineSchedules[work / group];
            const uint32_t row = work % group;
            const uint32_t bank = row < rows0 ? 0U : 1U;
            const uint32_t localRow = row - bank * rows0;
            const uint32_t first = schedule.partialStart;
            const uint16_t count = schedule.partialCount;
            const uint64_t statsOffset =
                static_cast<uint64_t>(first) * statsStride + bank * tiling.fdLseSubStride + localRow;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::DataCopyExtParams copy = {count, sizeof(float),
                                               static_cast<int64_t>((statsStride - 1U) * sizeof(float)), 0, 0};
            AscendC::DataCopyPadExtParams<float> padding = {false, 0, 0, 0};
            AscendC::DataCopyPad(maximum, partialStats[statsOffset], copy, padding);
            AscendC::DataCopyPad(sum, partialStats[statsOffset + 2U * tiling.fdLseSubStride], copy, padding);
            copy.blockLen = D * sizeof(float);
            copy.srcStride = (group - 1U) * D * sizeof(float);
            AscendC::DataCopyPad(numerator, partialN[(static_cast<uint64_t>(first) * group + row) * D], copy, padding);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            if (outputLse) {
                NpuArch::Epilogue::Block::GbsaMxfp4VF::CombinePartialCallVF<OutputT, true>(
                    output, lse, numerator, maximum, sum, count, tiling.log2CxCeil, D);
            } else {
                NpuArch::Epilogue::Block::GbsaMxfp4VF::CombinePartialCallVF<OutputT, false>(
                    output, lse, numerator, maximum, sum, count, tiling.log2CxCeil, D);
            }

            uint32_t batch = 0U, token = 0U, storage = 0U, kvHead = 0U;
            BatchSeqInfo qSeq{};
            if constexpr (QFormat == Format::TND) {
                GsaFd::DecodeTaskStorage(schedule.baseTask, tiling.kvHeads, tiling.batch, cuQ, usedQ,
                                         params.sequsedQ != nullptr, storage, token, batch, kvHead);
                ReadBatchSeqInfo<QFormat>(cuQ, usedQ, params.sequsedQ != nullptr, batch, qSeq);
            } else {
                kvHead = schedule.baseTask % tiling.kvHeads;
                DecodeBatchedQToken(schedule.baseTask / tiling.kvHeads, tiling.batch, tiling.qSeqCapacity, usedQ, batch,
                                    token, qSeq);
            }
            const uint32_t head = kvHead * group + row;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            copy = {1, D * sizeof(OutputT), 0, 0, 0};
            AscendC::DataCopyPad(globalOutput[GetTensorOffset<QFormat>(qSeq, token, head, tiling.numHeads, D)], output,
                                 copy);
            if (outputLse) {
                copy.blockLen = sizeof(float);
                AscendC::DataCopyPad(globalLse[GetLseOffset<QFormat>(qSeq, token, head, tiling.numHeads)], lse, copy);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::LocalTensor<float> maximum, sum, numerator, lse;
    AscendC::LocalTensor<OutputT> output;
};

} // namespace GbsaKernelArch35
#endif
