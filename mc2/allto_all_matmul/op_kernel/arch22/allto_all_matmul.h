/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_matmul.h
 * \brief
 */

#ifndef ALL_TO_ALL_MATMUL_H
#define ALL_TO_ALL_MATMUL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "allto_all_matmul_tiling.h"
#include "allto_all_matmul_util.h"
#include "../../../common/op_kernel/mc2_block_epilogue_per_token_dequant.h"
#include "allto_all_matmul_tile_broadcast_add.hpp"

#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_arch.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_dispatch_policy.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_gm_to_ub.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tla_epilogue_dispatch_policy.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_broadcast_mul.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_broadcast_one_blk.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_swizzle.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_copy.hpp"

#include "allto_all_matmul.hpp"

using namespace AscendC;
using namespace Catlass;

namespace Mc2Kernel {

// A2AMM : AlltoAllMatmul
#define TemplateA2AMMClass \
    typename AType, typename BType, typename BiasType, typename PerTokenScaleType, typename ScaleType, typename CType, \
        typename AllToAllResultType, bool hasBias, bool transB, int32_t QuantType
#define TemplateA2AMMFunc \
    AType, BType, BiasType, PerTokenScaleType, ScaleType, CType, AllToAllResultType, hasBias, transB, QuantType

using namespace AscendC;
template <TemplateA2AMMClass>
class AlltoAllMatmul : public CommBase {
public:
    __aicore__ inline AlltoAllMatmul(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR scaleGM,
                                GM_ADDR cGM, GM_ADDR allToAllResult, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void CatlassMatmul();
    // 按 m0 选择 TileShape M/N 维的实现体，供 CatlassMatmul 分发
    template <uint32_t TileM, uint32_t TileN>
    __aicore__ inline void CatlassMatmulImpl();
    __aicore__ inline void LoadCastAndSmoothToken(event_t eventId, LocalTensor<float> copyTensor,
                                                  LocalTensor<float> smoothScaleTensor, int32_t castOffset,
                                                  __gm__ AType *dataSrc, int64_t dataTokenOffset,
                                                  int32_t dataSegmentOffset, int32_t smoothScaleCastOffset,
                                                  int32_t actualMoveSize);
    __aicore__ inline void AlltoAll();
    __aicore__ inline void Dequant();
    template <typename EpilogueTileShape>
    __aicore__ inline void DequantImpl();
    __aicore__ inline void Quant(uint64_t flagIdx, int32_t commIdx);
    __aicore__ inline void QuantPerToken(LocalTensor<float> copyTensor, LocalTensor<float> smoothScaleTensor,
                                         LocalTensor<float> absTensor, LocalTensor<float> reduceMaxTensor,
                                         LocalTensor<float> quantScaleTensor, int32_t actualMoveSize,
                                         int32_t actualMoveToken, int32_t tokenPerMove, int32_t moveIdx,
                                         event_t eventId);
    __aicore__ inline void QuantToken(__gm__ AType *dataSrc, int64_t dataOffset, int32_t coreTokenOffset,
                                      int32_t dataLen, int32_t commIdx);
    __aicore__ inline void QuantTokenSegment(__gm__ AType *dataSrc, int64_t dataOffset, int32_t coreTokenOffset,
                                             int32_t dataLen, int32_t commIdx);
    __aicore__ inline void SmoothQuantProc(event_t eventId, int32_t dataSegmentOffset, int32_t smoothScaleCastOffset,
                                           int32_t actualMoveSize, LocalTensor<float> copyTensor,
                                           LocalTensor<float> smoothScaleTensor);
    __aicore__ inline void CalcTokenMaxValue(LocalTensor<float> copyTensor0, LocalTensor<float> copyTensor1,
                                             LocalTensor<float> absTensor0, LocalTensor<float> absTensor1,
                                             LocalTensor<float> smoothScaleTensor0,
                                             LocalTensor<float> smoothScaleTensor1, int32_t castOffset,
                                             __gm__ AType *dataSrc, int64_t dataTokenOffset,
                                             int32_t smoothScaleCastOffset, int32_t sizeScale,
                                             LocalTensor<float> reduceMaxTensor);
    __aicore__ inline void QuantPerSegment(LocalTensor<float> copyTensor0, LocalTensor<float> copyTensor1,
                                           LocalTensor<float> absTensor0, LocalTensor<float> absTensor1,
                                           LocalTensor<float> smoothScaleTensor0, LocalTensor<float> smoothScaleTensor1,
                                           int32_t castOffset, __gm__ AType *dataSrc, int64_t dataTokenOffset,
                                           int32_t smoothScaleCastOffset, int32_t sizeScale, float quantScaleReciproal);

private:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR scaleGM_;
    GM_ADDR x1ScaleGM_;
    GM_ADDR allToAllResultGM_;
    GM_ADDR workspaceGM_;

    __gm__ AType *gmPeerMem_;
    __gm__ int8_t *quantAGM_;
    __gm__ int32_t *dequantCGM_;
    GM_ADDR quantScaleGM_;

    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
};

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                               GM_ADDR x1ScaleGM, GM_ADDR scaleGM, GM_ADDR cGM,
                                                               GM_ADDR allToAllResultGM, GM_ADDR workspaceGM,
                                                               GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    auto tiling = (__gm__ AlltoAllMatmulTilingData *)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclCombineOpParam *)contextGM;
    rank = winContext_->rankId;
    rankSize = tilingData.allToAllMatmulInfo.rankSize;

    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    x1ScaleGM_ = x1ScaleGM;
    scaleGM_ = scaleGM;
    allToAllResultGM_ = allToAllResultGM;
    workspaceGM_ = GetUserWorkspace(workspaceGM);

    CommBase::SetArgs<AType>(rank, rankSize, tilingData);
    this->ub_offset = Catlass::BytesToBits(UB_OFFSET) / Catlass::SizeOfBits<int8_t>::value;

    if constexpr (QuantType == MC2_DYNAMIC_QUANT) {
        quantAGM_ = reinterpret_cast<__gm__ int8_t *>(workspaceGM_);
        dequantCGM_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + quantSize);
        quantScaleGM_ = reinterpret_cast<GM_ADDR>(workspaceGM_ + quantSize + dequantSize);
    }
    if constexpr (QuantType == MC2_STATIC_QUANT) {
        x1ScaleGM_ += rank * (m / rankSize) * sizeof(PerTokenScaleType); // 全量化场景，对x1Scale进行偏移
        dequantCGM_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_);
    }
    kBytes = Catlass::BitsToBytes(k * Catlass::SizeOfBits<AType>::value);
    tokenBytes = Catlass::BitsToBytes(tokenSize * Catlass::SizeOfBits<AType>::value);

    AlltoAllMatmul<TemplateA2AMMFunc>::AICInit();
    AlltoAllMatmul<TemplateA2AMMFunc>::AIVInit();
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gmPeerMem_ = reinterpret_cast<__gm__ AType *>(buff[rank]);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNorm();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);
    }
}

// A16W4的tiling
template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AlltoAll()
{
    if ASCEND_IS_AIV {
        ResetIpcFlags(BUFFER_NUM);
        PipeBarrier<PIPE_ALL>();
        int64_t src_offset = 0;
        uint32_t elemBytes = sizeof(AType); // 将int4归入该文件时，需要修改
        for (int32_t commIdx = 0; commIdx <= commCount; ++commIdx) {
            uint64_t flagIdx = commIdx % MAX_BLOCK_COUNT;

            if (commIdx == commCount - 1) {
                allToAllSizePerRankPerLoop = allToAllSizePerRank - src_offset;
            }

            if (commIdx >= MAX_BLOCK_COUNT && commIdx < commCount) {
                WaitEvent(flagIdx);
            }

            SetAndWaitAivSync(flagIdx);
            if (commIdx < commCount) {
                CrossRankSyncV1(FLAG_ZERO_IDX, commIdx + 1);
            }
            SetAndWaitAivSync(flagIdx);

            if (aivIdx == 0 && commIdx < commCount && aicIdx < allToAllSendCoreNum) {
                int32_t dstRank = aicIdx / coreNumPerRank;
                int32_t dstLoc = aicIdx % coreNumPerRank;
                int32_t coreOffset = dstLoc * allToAllSizePerCore;
                int32_t dataLen = coreOffset + allToAllSizePerCore > allToAllSizePerRankPerLoop ?
                                      allToAllSizePerRankPerLoop - coreOffset :
                                      allToAllSizePerCore;
                uint64_t dataSrc = dstRank * allToAllSizePerRank + src_offset + coreOffset;
                int64_t dataDst = static_cast<int64_t>(flagIdx) * pingPongBlockSize +
                                  static_cast<int64_t>(coreOffset) * rankSize + static_cast<int64_t>(rank) * k;
                uint32_t copyBytes = Catlass::BitsToBytes(dataLen * Catlass::SizeOfBits<AType>::value);

                if (dataLen > 0) {
                    CopyTokensFromGMToGM(reinterpret_cast<__gm__ int8_t *>(aGM_) + ElemNumToBytes<AType>(dataSrc),
                                         (__gm__ int8_t *)buff[dstRank] + ElemNumToBytes<AType>(dataDst), copyBytes,
                                         kBytes, tokenBytes);
                }
                src_offset += allToAllSizePerRankPerLoop;
            } else if (isAlltoallOut && aivIdx == 1 && commIdx > 0 && aicIdx >= allToAllSendCoreNum &&
                       aicIdx < usedCoreNum) {
                int64_t blockDst = static_cast<int64_t>((commIdx - 1) % MAX_BLOCK_COUNT) * pingPongBlockSize;
                int32_t mThisLoop = commIdx == commCount ? m / rankSize - (commIdx - 1) * mPerLoop : mPerLoop;
                int32_t mThisLoopPerCore = DivCeil(mThisLoop, allToAllRecvCoreNum);
                int32_t mSt = (aicIdx - allToAllSendCoreNum) * mThisLoopPerCore;
                int32_t mThisCoreThisLoop = mSt + mThisLoopPerCore > mThisLoop ? mThisLoop - mSt : mThisLoopPerCore;
                int64_t srcSt = blockDst + static_cast<int64_t>(mSt) * tokenSize;
                int64_t dstSt = (static_cast<int64_t>(commIdx - 1) * mPerLoop + mSt) * tokenSize;
                if (mThisCoreThisLoop > 0) {
                    CopyTokensFromGMToGM(
                        (__gm__ int8_t *)buff[rank] + ElemNumToBytes<AType>(srcSt),
                        reinterpret_cast<__gm__ int8_t *>(allToAllResultGM_) + ElemNumToBytes<AType>(dstSt),
                        mThisCoreThisLoop * tokenBytes, tokenBytes, tokenBytes);
                }
            }

            SetAndWaitAivSync(flagIdx);
            if (commIdx < commCount) {
                CrossRankSyncV1(FLAG_ONE_IDX, commIdx + 1);
            }
            SetAndWaitAivSync(flagIdx);

            if constexpr (QuantType == MC2_DYNAMIC_QUANT) { // 动态量化场景，拷贝完成后，对左矩阵进行quant
                if (commIdx < commCount) {
                    Quant(flagIdx, commIdx);
                    SetAndWaitAivSync(flagIdx);
                }
            }

            if (commIdx < commCount) {
                SetAicSync(flagIdx);
            }
        }

        WaitEvent(FLAG_ZERO_IDX);
        if (commCount % 2 == 0) { // 若AIC计算次数为偶数，则多等一次
            WaitEvent(FLAG_ONE_IDX);
        }

        if constexpr (QuantType != MC2_NON_QUANT) {
            SetAndWaitAivSync(FLAG_ONE_IDX);
            Dequant();
        }

        PipeBarrier<PIPE_ALL>();
        ResetIpcFlags(1);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Process()
{
    AlltoAll();
    CatlassMatmul();
    SyncAll<false>();
}

} // namespace Mc2Kernel
#include "allto_all_matmul_quant_impl.h"
#include "allto_all_matmul_matmul_impl.h"

#endif // ALL_TO_ALL_MATMUL_H
