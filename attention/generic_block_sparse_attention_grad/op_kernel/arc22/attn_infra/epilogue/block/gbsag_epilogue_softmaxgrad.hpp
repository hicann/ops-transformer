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
 * \file gbsag_epilogue_softmaxgrad.hpp
 * \brief Packet-local SoftmaxGrad implementation for GBSAG K_OUT.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_GBSAG_EPILOGUE_SOFTAXGRAD_HPP
#define CATLASS_EPILOGUE_BLOCK_GBSAG_EPILOGUE_SOFTAXGRAD_HPP

#include "../../../attn_infra/arch/gbsag_resource.hpp"
#include "../../../attn_infra/epilogue/gbsag_epilogue_dispatch_policy.hpp"
#include "gbsag_epilogue_packet_ub_layout.hpp"
#include "kernel_operator.h"

using namespace AscendC;

namespace NpuArch::Epilogue::Block {

template <typename InputDType, typename OutputDtype, uint32_t INPUT_LAYOUT>
class SoftmaxGrad {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    static constexpr uint64_t BLOCK_BYTE_SIZE = 32;
    static constexpr uint64_t BLOCK_SIZE = BLOCK_BYTE_SIZE / sizeof(float);
    static constexpr uint64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    static constexpr uint64_t SFMG_HIGH_PERF_D_FACTOR = 64;
    static constexpr uint64_t STAGES = 1;
    static constexpr uint64_t INPUT_NUM = PacketSoftmaxUbLayout::INPUT_COUNT;
    static constexpr uint64_t INPUT_BUFFER_LEN = PacketSoftmaxUbLayout::INPUT_BUFFER_BYTES;
    static constexpr uint64_t CAST_BUFFER_LEN = PacketSoftmaxUbLayout::CAST_BUFFER_BYTES;
    static constexpr uint64_t OUTPUT_BUFFER_LEN = PacketSoftmaxUbLayout::D_BYTES;

    __aicore__ inline explicit SoftmaxGrad(NpuArch::Arch::Resource<ArchTag> &resource_)
        : resource(resource_)
    {
        // prepD 整体迁入 [PREPD_BASE, 192K) 私有区，与
        // softmax 的 [0,104K) 区域彻底分离（布局见 PacketSoftmaxUbLayout）。
        // 输入按 headDim=128 全量定容（host 侧 EZ1001 锁死 128）；
        // CAST 按半行块（32 行）定容，ProcessPacket 分块处理。
        constexpr uint64_t inputBufferLenEachStage = INPUT_BUFFER_LEN / STAGES;
        constexpr uint64_t castBufferLenEachStage = CAST_BUFFER_LEN / STAGES;
        constexpr uint64_t outputBufferLenEachStage = OUTPUT_BUFFER_LEN / STAGES;
        for (uint64_t i = 0; i < STAGES; ++i) {
            doutTensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(PacketSoftmaxUbLayout::PREPD_BASE +
                                                                                inputBufferLenEachStage * i);
            outTensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(
                PacketSoftmaxUbLayout::PREPD_BASE + INPUT_BUFFER_LEN + inputBufferLenEachStage * i);
            doutFp32Tensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                PacketSoftmaxUbLayout::PREPD_BASE + INPUT_BUFFER_LEN * INPUT_NUM + castBufferLenEachStage * i);
            outFp32Tensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                PacketSoftmaxUbLayout::PREPD_BASE + INPUT_BUFFER_LEN * INPUT_NUM + CAST_BUFFER_LEN +
                castBufferLenEachStage * i);
            softmaxGradTensor[i] = resource.ubBuf.template GetBufferByByte<float>(PacketSoftmaxUbLayout::D_OFFSET +
                                                                                  outputBufferLenEachStage * i);
        }
        tempBuffer = resource.ubBuf.template GetBufferByByte<uint8_t>(PacketSoftmaxUbLayout::TEMP_OFFSET);
    }

    __aicore__ inline void ProcessPacket(GlobalTensor<InputDType> doutPack, GlobalTensor<InputDType> outPack,
                                         uint32_t rows, uint32_t headDim, GM_ADDR tilingDataAddr)
    {
        if (rows == 0) {
            return;
        }

        GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttentionGradTilingData, tilingData, tilingDataAddr);
        constexpr uint32_t ping = 0;
        auto eventId = EVENT_ID2;
        const uint64_t calcSize = static_cast<uint64_t>(rows) * headDim;

        // prepD 私有区（[PREPD_BASE,192K)，见 PacketSoftmaxUbLayout）。
        // 此处 MTE3→MTE2 等待是 pack GM 的 RAW 闸（GatherOut 刚写完 pack），
        // 与 UB 分区无关，必须保留。
        set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);

        DataCopy(doutTensor[ping], doutPack, calcSize);
        DataCopy(outTensor[ping], outPack, calcSize);

        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);

        // 192K UB 窗口放不下全量 CAST（2×32K），CAST 区按半行块
        // （CAST_BUFFER_BYTES=16K → 32 行）定容，64 行拆块逐块 cast + front。
        // 逐行规约（D=Σ_dout·out）行独立，分块结果与全量逐位一致。
        constexpr uint32_t CAST_CHUNK_ROWS = static_cast<uint32_t>(CAST_BUFFER_LEN / (128 * sizeof(float))); // 32
        uint32_t rowBegin = 0;
        while (rowBegin < rows) {
            uint32_t chunk = rows - rowBegin;
            if (chunk > CAST_CHUNK_ROWS) {
                // 非末块按 8 行对齐，尽量保住 front 的 basicBlock 高性能路径。
                chunk = CAST_CHUNK_ROWS / SFMG_HIGH_PERF_N_FACTOR * SFMG_HIGH_PERF_N_FACTOR;
            }
            const uint32_t calcChunk = chunk * headDim;
            const uint32_t outChunk = chunk * BLOCK_SIZE;

            Cast(doutFp32Tensor[ping], doutTensor[ping][rowBegin * headDim], RoundMode::CAST_NONE, calcChunk);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(outFp32Tensor[ping], outTensor[ping][rowBegin * headDim], RoundMode::CAST_NONE, calcChunk);
            AscendC::PipeBarrier<PIPE_V>();

            LocalTensor<float> dChunk = softmaxGradTensor[ping][rowBegin * BLOCK_SIZE];
            Duplicate<float>(dChunk, 0.0f, outChunk);
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t inputShape[] = {chunk, headDim};
            doutFp32Tensor[ping].SetShapeInfo(ShapeInfo(2, inputShape, DataFormat::ND));
            outFp32Tensor[ping].SetShapeInfo(ShapeInfo(2, inputShape, DataFormat::ND));
            uint32_t outputShape[] = {chunk, static_cast<uint32_t>(BLOCK_SIZE)};
            dChunk.SetShapeInfo(ShapeInfo(2, outputShape, DataFormat::ND));

            const bool isBasicBlock = chunk % SFMG_HIGH_PERF_N_FACTOR == 0 && headDim % SFMG_HIGH_PERF_D_FACTOR == 0;
            if (likely(isBasicBlock)) {
                SoftmaxGradFront<float, true>(dChunk, doutFp32Tensor[ping], outFp32Tensor[ping], tempBuffer,
                                              tilingData.softmaxGradTilingData);
            } else {
                SoftmaxGradFront<float, false>(dChunk, doutFp32Tensor[ping], outFp32Tensor[ping], tempBuffer,
                                               tilingData.softmaxGradTilingData);
            }
            rowBegin += chunk;
        }
        // D 交接：front 写 D（V）→ CalDs 读 D（V），V 按序流水天然安全，无需屏障。
        // 后续 softmax 的 MTE2 载入与本私有区无任何 RAW/WAR（分区分离）。
    }

private:
    NpuArch::Arch::Resource<ArchTag> &resource;
    LocalTensor<InputDType> doutTensor[STAGES];
    LocalTensor<InputDType> outTensor[STAGES];
    LocalTensor<float> doutFp32Tensor[STAGES];
    LocalTensor<float> outFp32Tensor[STAGES];
    LocalTensor<float> softmaxGradTensor[STAGES];
    LocalTensor<uint8_t> tempBuffer;
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_GBSAG_EPILOGUE_SOFTAXGRAD_HPP
