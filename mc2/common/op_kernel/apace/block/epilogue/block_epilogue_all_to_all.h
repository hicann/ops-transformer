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
 * \file block_epilogue_all_to_all.h
 * \brief
 */

#pragma once

#include "adv_api/hccl/hccl.h"
#include "apace/utils/comm_resource_builder.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

static constexpr uint64_t UB_ALIGN_BYTES = 32UL;
static constexpr int64_t AIV_UB_TILE_M = 32;
static constexpr int64_t AIV_AIC_RATIO = 2;
static constexpr uint16_t MTE2_MTE3_FLAG = 0;
static constexpr uint16_t MTE3_MTE2_FLAG = 1;

template <typename TypeC_, typename LayoutC_>
class BlockEpilogueAlltoAll {
public:
    using TypeC = TypeC_;
    using LayoutC = LayoutC_;
    using MakeLayoutUB =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<TypeC>>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR cGmAddr{nullptr};
        uint64_t m{0};
        uint64_t n{0};
        uint64_t ubBaseOffset{0};
        GM_ADDR hcclContext{nullptr};
    };

    __aicore__ inline BlockEpilogueAlltoAll() {}
    __aicore__ inline ~BlockEpilogueAlltoAll() {}

    __aicore__ inline void Init(const Params &params)
    {
        m_ = params.m;
        n_ = params.n;
        ubBaseOffset_ = params.ubBaseOffset;
        cGmAddr_ = reinterpret_cast<__gm__ TypeC *>(params.cGmAddr);

        winContext_ = (__gm__ Apace::HcclOpParam *)params.hcclContext;
        rankId_ = Apace::GetRankId(winContext_);
        tpWorldSize_ = Apace::GetRankDim(winContext_);

        if (tpWorldSize_ > 0) {
            tpSizeM_ = m_ / tpWorldSize_;
        }
    }

    __aicore__ inline void operator()(int64_t mPos, int64_t nPos, const BlockShape &singleShape)
    {
        CommProcess(mPos, nPos, singleShape);
    }

public:
    static __aicore__ uint64_t GetRequiredUBSize(const BlockShape &singleShape)
    {
        // 计算当前单次tile alltoall所需UB 空间
        int64_t singleN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        uint64_t padBytesPerRow =
            Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN * sizeof(TypeC)), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
        return (static_cast<uint64_t>(AIV_UB_TILE_M) * padBytesPerRow);
    }

private:
    __aicore__ inline void CommProcess(int64_t mPos, int64_t nPos, const BlockShape &singleShape);

private:
    uint64_t m_{0};
    uint64_t n_{0};
    uint32_t tpWorldSize_{0};
    uint64_t tpSizeM_{0};
    uint32_t rankId_{0};
    uint64_t ubBaseOffset_{0};

    __gm__ TypeC *cGmAddr_{nullptr};

    __gm__ Apace::HcclOpParam *winContext_{nullptr};

    decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{})) copyGM2UB_;
    decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{})) copyUB2GM_;
};

template <typename TypeC_, typename LayoutC_>
__aicore__ inline void BlockEpilogueAlltoAll<TypeC_, LayoutC_>::CommProcess(int64_t mPos, int64_t nPos,
                                                                            const BlockShape &singleShape)
{
    int64_t singleM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
    int64_t singleN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
    if (singleM <= 0 || singleN <= 0) {
        return;
    }

    int64_t aivIdx = AscendC::GetBlockIdx() % AIV_AIC_RATIO; // 两个V核交替搬运
    int64_t subTileCnt = Blaze::Gemm::CeilDiv(singleM, AIV_UB_TILE_M);

    auto layoutTensorC = AscendC::Te::FrameLayoutFormat<LayoutC, TypeC>{}(m_, n_);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutTensorC);

    uint64_t rowPaddedBytes =
        Blaze::Gemm::CeilDiv(static_cast<uint64_t>(singleN * sizeof(TypeC)), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    uint64_t paddingN = rowPaddedBytes / sizeof(TypeC);

    for (int64_t st = aivIdx; st < subTileCnt; st += AIV_AIC_RATIO) {
        int64_t subTileM = Min(AIV_UB_TILE_M, singleM - st * AIV_UB_TILE_M);
        if (subTileM <= 0) {
            continue;
        }
        int64_t globalBaseM = mPos + st * AIV_UB_TILE_M;

        auto layoutPaddingUB = MakeLayoutUB{}(subTileM, paddingN);
        auto layoutUB = MakeLayoutUB{}(subTileM, singleN);
        auto ubTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, TypeC>(ubBaseOffset_), layoutPaddingUB);
        auto gmTensor = gmC.Slice(AscendC::Te::MakeCoord(globalBaseM, nPos), AscendC::Te::MakeShape(subTileM, singleN));
        AscendC::Te::Copy(copyGM2UB_, ubTensor, gmTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(MTE2_MTE3_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(MTE2_MTE3_FLAG);

        int64_t processed = 0;
        while (processed < subTileM) {
            int64_t globalRow = globalBaseM + processed;
            if (tpSizeM_ == 0) {
                break;
            }
            int32_t dstRankId = static_cast<int32_t>(globalRow / tpSizeM_);
            if (dstRankId >= tpWorldSize_) {
                break;
            }
            int64_t tpBoundary = (dstRankId + 1) * tpSizeM_;
            int64_t rowsInBatch = Min(subTileM - processed, tpBoundary - globalRow);
            if (rowsInBatch <= 0) {
                break;
            }

            GM_ADDR remoteWinAddr = Apace::GetBaseWindAddrByRankId(winContext_, dstRankId);
            __gm__ TypeC *remoteGmAddr = reinterpret_cast<__gm__ TypeC *>(remoteWinAddr);

            auto layoutTensorRemote = AscendC::Te::FrameLayoutFormat<LayoutC, TypeC>{}(m_, static_cast<int64_t>(n_));
            auto gmRemote = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(remoteGmAddr),
                                                    layoutTensorRemote);
            auto remoteGMTensor = gmRemote.Slice(
                AscendC::Te::MakeCoord(static_cast<uint64_t>(rankId_) * tpSizeM_ + globalRow % tpSizeM_, nPos),
                AscendC::Te::MakeShape(rowsInBatch, singleN));
            auto ubProcessTensor =
                ubTensor.Slice(AscendC::Te::MakeCoord(processed, 0), AscendC::Te::MakeShape(rowsInBatch, paddingN));
            AscendC::Te::Copy(copyUB2GM_, remoteGMTensor, ubProcessTensor);

            processed += rowsInBatch;
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(MTE3_MTE2_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(MTE3_MTE2_FLAG);
    }
}

} // namespace Block
} // namespace Gemm
} // namespace Blaze
