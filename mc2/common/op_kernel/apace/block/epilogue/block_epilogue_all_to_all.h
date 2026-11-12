/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_all_to_all.h
 * \brief AllToAll epilogue on AIV: consume the L0C->UB fixpipe tile (FP32, DualDst SplitM),
 *        cast it to BF16 and dispatch rows to peer ranks' communication windows.
 *
 * Per tile (one invocation per AIV sub-core):
 *   AIC fixpipe writes the even-row-split M halves into each sub-core's UB view (offset 0).
 *   AIV: WaitFlag(MTE3_V) -> cast FP32->BF16 row by row (cast_reg_loop)
 *        -> SetFlag/WaitFlag(V_MTE3) -> Sync::NotifyCube ack (INTRA per-sub-block domain,
 *        FP32 slot released early) -> rank-batched MTE3 UB->remote-window copies
 *        -> SetFlag(MTE3_V)
 */

#pragma once

#include "adv_api/hccl/hccl.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/utils/apace_constant.h"
#include "apace/utils/asc_tensor_api_vector_compute.h"
#include "blaze/gemm/utils/sync.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

static constexpr uint64_t UB_ALIGN_BYTES = 32UL;
// FP32 槽 N 维元素对齐（与 kernel 侧 ubC 布局一致的 32 元素对齐）
static constexpr int64_t N_ELEM_ALIGN = 32;
static constexpr int64_t N_ELEM_ALIGN_MASK = N_ELEM_ALIGN - 1;
// DualDst SplitM：一个 AIC 的 L0C 结果按 M 对半分给 2 个 AIV 子核
static constexpr int64_t AIV_SUB_BLOCK_NUM = 2;
// bf16 发送缓冲跨 tile 复用的 V<->MTE3 事件
static constexpr uint16_t BF16_BUF_EVENT_FLAG = 0;

template <typename TypeC_, typename LayoutC_>
class BlockEpilogueAlltoAll {
public:
    using TypeC = TypeC_;
    using LayoutC = LayoutC_;
    using MakeLayoutUB = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<TypeC>>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        uint64_t m{0};
        uint64_t n{0};
        uint64_t ubBaseOffset{0}; // fixpipe FP32 槽基址，DualDst 固定写入 offset 0
        GM_ADDR ubmemContext{nullptr};
    };

    __aicore__ inline BlockEpilogueAlltoAll()
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(BF16_BUF_EVENT_FLAG);
    }

    __aicore__ inline ~BlockEpilogueAlltoAll()
    {
        if ASCEND_IS_AIC {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BF16_BUF_EVENT_FLAG);
    }

    __aicore__ inline void Init(const Params& params)
    {
        m_ = params.m;
        n_ = params.n;
        ubBaseOffset_ = params.ubBaseOffset;

        winContext_ = (__gm__ Apace::AivComm::CommUbmemContext*)params.ubmemContext;
        rankId_ = winContext_->rankId;
        tpWorldSize_ = winContext_->rankSize;
        if (tpWorldSize_ > 0) {
            tpSizeM_ = m_ / tpWorldSize_;
        }
    }

    __aicore__ inline void operator()(int64_t mPos, int64_t nPos, const BlockShape& singleShape)
    {
        CommProcess(mPos, nPos, singleShape);
    }

public:
    static __aicore__ uint64_t GetRequiredUBSize(const BlockShape& singleShape)
    {
        int64_t singleM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        int64_t singleN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        const int64_t halfRows = (singleM + AIV_SUB_BLOCK_NUM - 1) / AIV_SUB_BLOCK_NUM;
        const int64_t alignNF32 = (singleN + N_ELEM_ALIGN_MASK) & ~N_ELEM_ALIGN_MASK;
        const uint64_t padBytesPerRowBf16 =
            Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN) * sizeof(TypeC), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
        return static_cast<uint64_t>(halfRows) * alignNF32 * sizeof(float) +
               static_cast<uint64_t>(halfRows) * padBytesPerRowBf16;
    }

private:
    __aicore__ inline void CommProcess(int64_t mPos, int64_t nPos, const BlockShape& singleShape);

private:
    uint64_t m_{0};
    uint64_t n_{0};
    uint32_t tpWorldSize_{0};
    uint64_t tpSizeM_{0};
    uint32_t rankId_{0};
    uint64_t ubBaseOffset_{0};

    __gm__ Apace::AivComm::CommUbmemContext* winContext_{nullptr};

    decltype(asc::te::make_copy(asc::te::copy_ub_to_gm{})) copyUB2GM_;
};

template <typename TypeC_, typename LayoutC_>
__aicore__ inline void BlockEpilogueAlltoAll<TypeC_, LayoutC_>::CommProcess(int64_t mPos, int64_t nPos,
                                                                            const BlockShape& singleShape)
{
    int64_t singleM = asc::te::get<IDX_M_TILEIDX>(singleShape);
    int64_t singleN = asc::te::get<IDX_N_TILEIDX>(singleShape);
    if (singleM <= 0 || singleN <= 0) {
        return;
    }

    // 与 L0C->UB fixpipe（DualDst SplitM）的拆分镜像：偶数行对半，前半->vec0，后半->vec1
    const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    const int64_t evenRows = (singleM + AIV_SUB_BLOCK_NUM - 1) & ~(AIV_SUB_BLOCK_NUM - 1);
    const int64_t halfRows = evenRows / AIV_SUB_BLOCK_NUM;
    const int64_t rowCount = (subBlockIdx == AIV_SUB_BLOCK_NUM - 1) ? (singleM - halfRows) : halfRows;
    const int64_t rowBase = subBlockIdx * halfRows;
    if (rowCount <= 0) {
        Sync::NotifyCube();
        return;
    }

    const int64_t alignNF32 =
        (singleN + N_ELEM_ALIGN_MASK) & ~N_ELEM_ALIGN_MASK; // FP32 槽行宽（元素），与 kernel 侧 ubC 布局一致
    const uint64_t fp32SlotBytes = static_cast<uint64_t>(halfRows) * alignNF32 * sizeof(float);
    const uint64_t rowPaddedBytes =
        Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN) * sizeof(TypeC), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    const uint64_t padN = rowPaddedBytes / sizeof(TypeC); // bf16 缓冲行宽（元素）
    const uint64_t bf16BufOffset = fp32SlotBytes;         // alignNF32*4 为 128B 倍数，天然 32B 对齐

    // 等 bf16 缓冲空闲（上一 tile 的 MTE3 发送完成）
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BF16_BUF_EVENT_FLAG);

    for (int64_t r = 0; r < rowCount; ++r) {
        auto srcRow = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, float>(static_cast<uint64_t>(r) * alignNF32 * sizeof(float)),
            asc::te::make_layout(asc::te::make_shape(static_cast<uint64_t>(singleN))));
        auto dstRow = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, TypeC>(
                                               bf16BufOffset + static_cast<uint64_t>(r) * rowPaddedBytes),
                                           asc::te::make_layout(asc::te::make_shape(padN)));
        cast_reg_loop(srcRow, dstRow, static_cast<uint32_t>(singleN));
    }

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(BF16_BUF_EVENT_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(BF16_BUF_EVENT_FLAG);

    Sync::NotifyCube();

    auto layoutTensorRemote = asc::te::frame_layout_format<LayoutC, TypeC>{}(m_, static_cast<int64_t>(n_));
    auto ubBufTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, TypeC>(bf16BufOffset),
                                            MakeLayoutUB{}(rowCount, padN));
    int64_t processed = 0;
    while (processed < rowCount) {
        int64_t globalRow = mPos + rowBase + processed;
        if (tpSizeM_ == 0) {
            break;
        }
        int32_t dstRankId = static_cast<int32_t>(globalRow / tpSizeM_);
        if (dstRankId >= tpWorldSize_) {
            break;
        }
        int64_t tpBoundary = (dstRankId + 1) * tpSizeM_;
        int64_t rowsInBatch = Min(rowCount - processed, tpBoundary - globalRow);
        if (rowsInBatch <= 0) {
            break;
        }

        GM_ADDR remoteWinAddr = (GM_ADDR)(winContext_->commBufferAddrs[dstRankId] + APACE_MTE_STATE_WIN_SIZE +
                                          static_cast<uint64_t>(dstRankId) * APACE_EP_RANK_OFFSET_STEP);
        auto gmRemote = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ TypeC*>(remoteWinAddr)),
            layoutTensorRemote);
        auto remoteGMTensor =
            gmRemote.slice(asc::te::make_coord(static_cast<uint64_t>(rankId_) * tpSizeM_ + globalRow % tpSizeM_, nPos),
                           asc::te::make_shape(rowsInBatch, singleN));
        auto ubProcessTensor =
            ubBufTensor.slice(asc::te::make_coord(processed, 0), asc::te::make_shape(rowsInBatch, padN));
        asc::te::copy(copyUB2GM_, remoteGMTensor, ubProcessTensor);

        processed += rowsInBatch;
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(BF16_BUF_EVENT_FLAG);
}

} // namespace Block
} // namespace Gemm
} // namespace Blaze
