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
 * \file mega_moe_kernel_a3_mask.hpp
 * \brief MegaMoeKernel 的 Mask 专属成员函数类外定义。唯一包含点：
 *        mega_moe_kernel_a3.hpp 末尾，不可独立包含。
 */
#ifndef MEGA_MOE_KERNEL_A3_MASK_HPP
#define MEGA_MOE_KERNEL_A3_MASK_HPP

namespace Catlass::Gemm::Kernel {

using namespace AscendC;
using namespace Mc2Tiling;

// 位图、计数与输入量化完成后，向各目标卡发布本卡输入已就绪的信号。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_, bool BatchedDispatch>
CATLASS_DEVICE void MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_,
                                  BlockEpilogue3_, SharedBf16Mmad_, SharedBf16Epilogue_, SharedGmm2Bf16_,
                                  SharedBlockMmad_, SharedBlockEpilogue1_, SharedBlockEpilogue2_, BlockEpilogueDedup_,
                                  DedupOn_, BatchedDispatch>::PublishMaskDataReady(Params const& p)
{
    auto flag = resource.ubBuf.template GetBufferByByte<int32_t>(0);
    constexpr uint32_t flagElements = BYTE_PER_BLK / sizeof(int32_t);
    AscendC::Duplicate(flag, 1, flagElements);
    SyncEvent<AscendC::HardEvent::V_MTE3>();
    for (uint32_t dst = coreIdx; dst < p.EP; dst += coreNum) {
        if (!IsRankMasked(dst)) {
            AscendC::GlobalTensor<int32_t> remote;
            remote.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                shmem(peermemInfo.offsetDataReady + RuntimeRank(p) * ALIGN_512, dst)));
            AscendC::DataCopy(remote, flag, flagElements);
        }
    }
    SyncEvent<AscendC::HardEvent::MTE3_S>();
}

// 位图展开后、拉取特征前，分核等待各来源卡输入就绪，再汇合本卡各核。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_, bool BatchedDispatch>
CATLASS_DEVICE void
MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_, BlockEpilogue3_,
              SharedBf16Mmad_, SharedBf16Epilogue_, SharedGmm2Bf16_, SharedBlockMmad_, SharedBlockEpilogue1_,
              SharedBlockEpilogue2_, BlockEpilogueDedup_, DedupOn_, BatchedDispatch>::WaitMaskDataReady(Params const& p)
{
    for (uint32_t src = coreIdx; src < p.EP; src += coreNum) {
        if (!IsRankMasked(src)) {
            gm_signal_wait_until_ne(
                reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetDataReady + src * ALIGN_512), 0);
        }
    }
    AscendC::SyncAll<true>();
}

// 当前 chunk 的最后一轮完成后，清零 ready 槽位，供下一 chunk 复用。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_, bool BatchedDispatch>
CATLASS_DEVICE void MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_,
                                  BlockEpilogue3_, SharedBf16Mmad_, SharedBf16Epilogue_, SharedGmm2Bf16_,
                                  SharedBlockMmad_, SharedBlockEpilogue1_, SharedBlockEpilogue2_, BlockEpilogueDedup_,
                                  DedupOn_, BatchedDispatch>::ResetMaskDataReady(Params const& p)
{
    if (coreIdx >= p.EP) {
        return;
    }
    auto zero = resource.ubBuf.template GetBufferByByte<int32_t>(0);
    constexpr uint32_t flagElements = BYTE_PER_BLK / sizeof(int32_t);
    AscendC::Duplicate(zero, 0, flagElements);
    SyncEvent<AscendC::HardEvent::V_MTE3>();
    for (uint32_t src = coreIdx; src < p.EP; src += coreNum) {
        AscendC::GlobalTensor<int32_t> ready;
        ready.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetDataReady + src * ALIGN_512));
        AscendC::DataCopy(ready, zero, flagElements);
    }
    SyncEvent<AscendC::HardEvent::MTE3_S>();
}

// 位图展开后，按索引从来源卡 CCL 窗口拉取特征与 scale，写入本卡 GMM 工作区并保存返回槽位。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_, bool BatchedDispatch>
CATLASS_DEVICE void MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_,
                                  BlockEpilogue3_, SharedBf16Mmad_, SharedBf16Epilogue_, SharedGmm2Bf16_,
                                  SharedBlockMmad_, SharedBlockEpilogue1_, SharedBlockEpilogue2_, BlockEpilogueDedup_,
                                  DedupOn_, BatchedDispatch>::CopyMaskRows(Params const& p, uint32_t srcRank,
                                                                           uint32_t expert, uint32_t rows,
                                                                           uint32_t savedRow, uint64_t gmOffsetA,
                                                                           int32_t& pingpong)
{
    if (rows == 0) {
        return;
    }
    using MaskRoute = Epilogue::Block::MaskIndexedRouteTagT<BLOCKED_OUTPUT_>;
    constexpr uint32_t indexBatch = MoePermute::MaskUbLayout::COMBINE_INDICES_BYTES / sizeof(int32_t);
    static_assert(MoePermute::MaskUbLayout::MAX_SLOTS <= (1U << MaskRoute::RANK_SHIFT));
    constexpr uint32_t rowBatch = 4;
    constexpr int32_t bufferCount = 2;
    const uint32_t hidden = p.problemShape.k();
    const uint32_t stride = hidden + ALIGN_512;
    auto indices = resource.ubBuf.template GetBufferByByte<int32_t>(MoePermute::MaskUbLayout::COMBINE_INDICES_OFFSET);
    // 展开索引表按本地专家和来源卡分段，sourcePrefix 是该段的起始元素下标。
    uint32_t sourcePrefix = (expert * p.EP + srcRank) * this->slotStride_;
    if (this->maskMultipleRounds_) {
        // blockBegin 是该组在整个 chunk 接收顺序中的起始行，用于计算已处理的行数。
        int64_t blockBegin = srcRank == 0 ? 0 : chunkCumsumMM.GetValue((srcRank - 1) * p.expertPerRank + expert);
        for (uint32_t e = 0; e < expert; ++e) {
            blockBegin += chunkCumsumMM.GetValue((p.EP - 1) * p.expertPerRank + e);
        }
        int64_t consumed = this->maskRoundIdx_ * static_cast<int64_t>(p.maxOutputSize) - blockBegin;
        if (consumed > 0) {
            sourcePrefix += static_cast<uint32_t>(consumed);
        }
    }
    AscendC::GlobalTensor<int32_t> maskIndicesGm, routeIndexGm;
    maskIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrMaskIndices));
    routeIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrRouteIndex));
    AscendC::GlobalTensor<int8_t> remoteX;
    remoteX.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(shmem(peermemInfo.offsetA, srcRank)));
    for (uint32_t base = 0; base < rows; base += indexBatch) {
        uint32_t length = rows - base < indexBatch ? rows - base : indexBatch;
        AscendC::DataCopyPad(indices, maskIndicesGm[sourcePrefix + base],
                             {1, static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0}, {});
        SyncEvent<AscendC::HardEvent::MTE2_S, EVENT_ID2>();
        for (uint32_t off = 0; off < length; off += rowBatch) {
            uint32_t n = length - off < rowBatch ? length - off : rowBatch;
            pingpong = (pingpong + 1) % bufferCount;
            auto event = static_cast<AscendC::TEventID>(pingpong);
            auto buffer = resource.ubBuf.template GetBufferByByte<int8_t>(pingpong * rowBatch * stride);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            for (uint32_t row = 0; row < n; ++row) {
                uint32_t slot = indices.GetValue(off + row);
                uint32_t token = slot / p.topK;
                // 将来源卡号和原路由槽位一起保存，供 Combine 定位结果写回位置。
                indices.SetValue(off + row, slot | (srcRank << MaskRoute::RANK_SHIFT));
                AscendC::DataCopy(buffer[row * stride], remoteX[static_cast<uint64_t>(token) * stride],
                                  hidden + BYTE_PER_BLK);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            AscendC::DataCopyPad(gmA[gmOffsetA + static_cast<uint64_t>(base + off) * hidden], buffer,
                                 {static_cast<uint16_t>(n), hidden, ALIGN_512 / BYTE_PER_BLK, 0, 0});
            auto scales = buffer[hidden].template ReinterpretCast<float>();
            AscendC::DataCopyPad(gmPerTokenScale1[savedRow + base + off], scales,
                                 {static_cast<uint16_t>(n), sizeof(float), stride / BYTE_PER_BLK - 1, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        }
        SyncEvent<AscendC::HardEvent::S_MTE3, EVENT_ID2>();
        AscendC::DataCopyPad(routeIndexGm[savedRow + base], indices,
                             {1, static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0});
        SyncEvent<AscendC::HardEvent::MTE3_S, EVENT_ID2>();
    }
}

// 专家计数 AllGather 完成后，计算各来源卡的接收行前缀和，并更新专家累计 token 数。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_, bool BatchedDispatch>
CATLASS_DEVICE void
MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_, BlockEpilogue3_,
              SharedBf16Mmad_, SharedBf16Epilogue_, SharedGmm2Bf16_, SharedBlockMmad_, SharedBlockEpilogue1_,
              SharedBlockEpilogue2_, BlockEpilogueDedup_, DedupOn_,
              BatchedDispatch>::GetMaskCumsumForMMAIV(AscendC::GlobalTensor<int32_t>& tokenPerExpert,
                                                      AscendC::GlobalTensor<int32_t>& result, uint32_t expertPerRank,
                                                      uint32_t rankId, uint32_t EP, GM_ADDR expertTokenNums)
{
    const uint32_t stride = AlignUp(expertPerRank, 8);
    const uint32_t elements = EP * stride;
    auto prefix = resource.ubBuf.template GetBufferByByte<int32_t>(0);
    auto next = resource.ubBuf.template GetBufferByByte<int32_t>(elements * sizeof(int32_t));
    auto accumulated = resource.ubBuf.template GetBufferByByte<int32_t>(2 * elements * sizeof(int32_t));
    AscendC::GlobalTensor<int32_t> gmAccumulated;
    gmAccumulated.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertTokenNums));
    LoadExpertCountsForCumsum(prefix, tokenPerExpert, expertPerRank, rankId, EP);
    AscendC::DataCopyPad(accumulated, gmAccumulated, {1, U16(expertPerRank * sizeof(int32_t)), 0, 0}, {});
    SyncEvent<AscendC::HardEvent::MTE2_V>();
    for (uint32_t distance = 1; distance < EP; distance <<= 1) {
        const uint32_t offset = distance * stride;
        AscendC::Adds(next, prefix, static_cast<int32_t>(0), offset);
        AscendC::Add(next[offset], prefix[offset], prefix, elements - offset);
        AscendC::PipeBarrier<PIPE_V>();
        auto previous = prefix;
        prefix = next;
        next = previous;
    }
    AscendC::Add(accumulated, accumulated, prefix[(EP - 1) * stride], expertPerRank);
    SyncEvent<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopyPad(result, prefix, {U16(EP), U16(expertPerRank * sizeof(int32_t)), 0, 0});
    AscendC::DataCopyPad(gmAccumulated, accumulated, {1, U16(expertPerRank * sizeof(int32_t)), 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
}

// 发送侧生成位图和计数，将计数通信与单份输入量化并行执行，最后发布 ready 信号。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_, bool BatchedDispatch>
CATLASS_DEVICE void MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_,
                                  BlockEpilogue3_, SharedBf16Mmad_, SharedBf16Epilogue_, SharedGmm2Bf16_,
                                  SharedBlockMmad_, SharedBlockEpilogue1_, SharedBlockEpilogue2_, BlockEpilogueDedup_,
                                  DedupOn_, BatchedDispatch>::MaskPrologue(Params const& p, int64_t tokenBase,
                                                                           uint32_t tokens, int64_t countOffset)
{
    AscendC::GlobalTensor<int32_t> ids, counts, rowIndex;
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(p.expertIdx) + tokenBase * p.topK);
    counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem() + countOffset));
    rowIndex.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.expandedRowIdx));
    GM_ADDR active = p.ptrXActiveMask == nullptr ? nullptr : p.ptrXActiveMask + tokenBase;
    MaskRouting routing(resource, tokens, p.topK, p.EP * p.expertPerRank, coreIdx, coreNum);
    routing.Generate(ids, counts, shmem, peermemInfo.offsetMask, p.expertPerRank, rankMaskLocal,
                     maskBufferAddr != nullptr, active);
    AscendC::SyncAll<true>();
    const uint32_t metadataCores = p.EP < coreNum / 2 ? p.EP : coreNum / 2;
    if (coreIdx < metadataCores) {
        AllGatherTokenPerExpert(p, countOffset, metadataCores);
    } else {
        using Quant = MoePermute::MoePermutePrologue<PROLOGUE_UB_STAGES, MoePermute::MoePermuteProloguePolicy,
                                                     bfloat16_t, int8_t, true>;
        typename Quant::Params qp(tokens, p.problemShape.k(), p.topK, p.EP * p.expertPerRank);
        qp.reservedTokenCores = metadataCores;
        Quant quant(resource, qp);
        AscendC::GlobalTensor<bfloat16_t> x;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(p.ptrA) + tokenBase * p.problemShape.k());
        AscendC::GlobalTensor<int8_t> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(shmem() + peermemInfo.offsetA));
        quant(x, ids, payload, rowIndex, counts, workspaceInfo.ptrPermuteCounts, active);
    }
    AscendC::SyncAll<true>();
    PublishMaskDataReady(p);
}

// 在公共 Workspace 布局之后定位返回索引和位图展开索引，保持现有各段偏移。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_, bool BatchedDispatch>
CATLASS_DEVICE MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_,
                             BlockEpilogue3_, SharedBf16Mmad_, SharedBf16Epilogue_, SharedGmm2Bf16_, SharedBlockMmad_,
                             SharedBlockEpilogue1_, SharedBlockEpilogue2_, BlockEpilogueDedup_, DedupOn_,
                             BatchedDispatch>::MaskWorkspaceInfo::MaskWorkspaceInfo(const Params& params)
    : WorkspaceInfo(params)
{
    uint64_t workspaceOffset = reinterpret_cast<GM_ADDR>(this->ptrSoftFlagBase) - params.ptrWorkspace +
                               params.EP * sizeof(int32_t) * FLAGSTRIDE;
    workspaceOffset = RoundUp<ALIGN_512>(workspaceOffset);
    ptrRouteIndex = params.ptrWorkspace + workspaceOffset;
    workspaceOffset += RoundUp<ALIGN_512>(params.maxOutputSize * sizeof(int32_t));
    ptrMaskIndices = params.ptrWorkspace + workspaceOffset;
}

// 在公共通信布局上确定 Mask 的 D 区对齐、位图和 ready 槽位位置。
template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_, class BlockEpilogue1_,
          class BlockEpilogue2_, class BlockEpilogue3_, class SharedBf16Mmad_, class SharedBf16Epilogue_,
          bool SharedGmm2Bf16_, class SharedBlockMmad_, class SharedBlockEpilogue1_, class SharedBlockEpilogue2_,
          class BlockEpilogueDedup_, bool DedupOn_, bool BatchedDispatch>
CATLASS_DEVICE MegaMoeKernel<BlockMmad_, BlockScheduler_, ElementGroupList_, BlockEpilogue1_, BlockEpilogue2_,
                             BlockEpilogue3_, SharedBf16Mmad_, SharedBf16Epilogue_, SharedGmm2Bf16_, SharedBlockMmad_,
                             SharedBlockEpilogue1_, SharedBlockEpilogue2_, BlockEpilogueDedup_, DedupOn_,
                             BatchedDispatch>::MaskPeermemInfo::MaskPeermemInfo(const Params& params,
                                                                                const HcclShmem<false>& shmem)
    : PeermemInfo(params, shmem)
{
    if constexpr (BLOCKED_OUTPUT_) {
        this->offsetD = RoundUp<ALIGN_512>(this->offsetD);
    } else {
        if (params.problemShape.m() >= L1TileShape::M && params.problemShape.k() % L1TileShape::N == 0) {
            this->offsetD = RoundUp<ALIGN_512>(this->offsetD);
        }
    }
    const int64_t bs = params.problemShape.m();
    const int64_t chunkTokens = bs < PERMUTE_CHUNK ? bs : static_cast<int64_t>(PERMUTE_CHUNK);
    const int64_t dSize = chunkTokens * params.topK * params.problemShape.k() * sizeof(int16_t);
    offsetMask = AlignUp(this->offsetD + dSize, static_cast<int64_t>(ALIGN_512));
    offsetDataReady = this->offsetPeerTokenPerExpert - params.EP * ALIGN_512;
}

} // namespace Catlass::Gemm::Kernel

#endif // MEGA_MOE_KERNEL_A3_MASK_HPP
