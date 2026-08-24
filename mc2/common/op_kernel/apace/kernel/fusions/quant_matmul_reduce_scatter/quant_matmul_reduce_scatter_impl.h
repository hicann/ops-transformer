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
 * \file quant_matmul_reduce_scatter_impl.h
 * \brief ReduceScatter post-processing for QuantMatmulReduceScatter.
 */

#pragma once

#include "apace/utils/comm_resource_builder.h"
#include "kernel_qbmm_mx_mix.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;

constexpr static uint64_t UB_ALIGN_BYTES = 32UL;
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;
constexpr int32_t FLAG_ID_NUM_ONE = 1;
constexpr float STATUS_FLAG_VAL = -1.0f;
constexpr float STATUS_FLAG_THREDSHOLD = 0.5f;

template <typename RoundMode, typename SatMode, typename IndexPos>
struct CastTrait {
    using roundMode = RoundMode;
    using satMode = SatMode;
    using indexPos = IndexPos;
};

using CastTraitF322Bf16 =
    CastTrait<AscendC::Te::CastRoundMode::RN, AscendC::Te::CastSatMode::NoSat, AscendC::Te::CastIndexPos::Even>;

template <typename ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class QuantMatmulReduceScatterImpl {
public:
    using KernelImpl =
        Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
                              AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                                  KernelMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>>;

    using CType = typename BlockMmad::CType;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutStatus = AscendC::Te::NDExtLayoutPtn;

    using MakeLayoutUB =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<CType>>;
    using CopyMakerGM2UB = decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{}));
    using CopyMakerUB2GM = decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{}));

    struct Params {
        typename KernelImpl::Params matmulKernelParams;
        GM_ADDR yGM{nullptr};
        GM_ADDR hcclContext{nullptr};
    };

    __aicore__ inline QuantMatmulReduceScatterImpl() {}
    __aicore__ inline ~QuantMatmulReduceScatterImpl() {}

    __aicore__ inline void operator()(Params &params)
    {
        KernelImpl kernel;
        kernel(params.matmulKernelParams);

        if ASCEND_IS_AIV {
            ReduceAddInit(params);
            ReduceAddProcess();
        }
    }

private:
    __aicore__ inline void ReduceAddInit(const Params &params);
    __aicore__ inline void ReduceAddProcess();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void ReadRemoteDataAdd();
    __aicore__ inline void SplitToCore(const uint32_t curSendCnt, const uint32_t curUseAivNum, const uint32_t coreId,
                                       uint32_t &startId, uint32_t &endId, uint32_t &sendNum);

private:
    __gm__ Apace::HcclOpParam *winContext_{nullptr};
    uint32_t rankId_{0};
    uint32_t coreVid_{0};
    uint32_t tpWorldSize_{0};
    uint64_t m_{0};
    uint64_t n_{0};
    GM_ADDR yGM_{0};

    CopyMakerGM2UB copyGM2UB_ = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
    CopyMakerUB2GM copyUB2GM_ = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
};

template <typename ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue,
                                                    BlockScheduler>::ReduceAddInit(const Params &params)
{
    winContext_ = (__gm__ Apace::HcclOpParam *)params.hcclContext;
    rankId_ = Apace::GetRankId(winContext_);
    coreVid_ = AscendC::GetBlockIdx();
    tpWorldSize_ = Apace::GetRankDim(winContext_);
    m_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.matmulKernelParams.problemShape));
    n_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(params.matmulKernelParams.problemShape));
    yGM_ = params.yGM;
}

template <typename ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ReduceAddProcess()
{
    SyncAll<true>();
    WriteStatusToWin();
    ReadStatus();
    SyncAll<true>();

    ReadRemoteDataAdd();
}

template <typename ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::SplitToCore(
    const uint32_t curSendCnt, const uint32_t curUseCoreNum, const uint32_t coreId, uint32_t &startId, uint32_t &endId,
    uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseCoreNum;
    uint32_t remainderNum = curSendCnt % curUseCoreNum;
    startId = sendNum * coreId;
    if (coreId < remainderNum) {
        sendNum += 1;
        startId += coreId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <typename ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ReadRemoteDataAdd()
{
    uint32_t startRowId = 0;
    uint32_t endRowId = 0;
    uint32_t rowNum = 0;
    uint32_t tokenIndex = 0;
    uint64_t tpRankM = tpWorldSize_ == 0 ? m_ : (m_ / tpWorldSize_);
    SplitToCore(tpRankM, GetBlockNum() * 2, coreVid_, startRowId, endRowId, rowNum);

    auto layoutTensorC = AscendC::Te::FrameLayoutFormat<LayoutC, CType>{}(m_, n_);
    auto layoutTensorY = AscendC::Te::FrameLayoutFormat<LayoutC, CType>{}(tpRankM, n_);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                           (__gm__ CType *)Apace::GetBaseWindAddrByRankId(winContext_, rankId_)),
                                       layoutTensorC);
    auto gmY = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ bfloat16_t *)yGM_),
                                       layoutTensorY);

    uint64_t rowPaddedBytes =
        Blaze::Gemm::CeilDiv(static_cast<uint64_t>(n_ * sizeof(CType)), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    uint64_t paddingN = rowPaddedBytes / sizeof(CType);

    uint64_t localbf16Offset = paddingN * sizeof(float);
    uint64_t localfp32Offset = localbf16Offset + paddingN * sizeof(CType);
    uint64_t sumbf16Offset = localfp32Offset + paddingN * sizeof(float);

    auto sumfp32UbTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(0),
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>{}(
            1, paddingN));
    for (uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
        AscendC::Te::Transform<AscendC::Te::Inst::MulScalar>(sumfp32UbTensor, sumfp32UbTensor, 0.0f);
        tokenIndex = startRowId + tileIdx;
        for (int tpIndex = 0; tpIndex < tpWorldSize_; ++tpIndex) {
            uint32_t mPos = tokenIndex + tpIndex * (tpRankM);
            uint32_t nPos = 0;

            auto layoutPaddingUB = MakeLayoutUB{}(1, paddingN);
            auto ubTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, CType>(localbf16Offset), layoutPaddingUB);
            auto gmTensor = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(1, n_));
            AscendC::Te::Copy(copyGM2UB_, ubTensor, gmTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

            auto fp32UbTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(localfp32Offset),
                AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>{}(
                    1, paddingN));
            AscendC::Te::Transform<AscendC::Te::Inst::Cast>(fp32UbTensor, ubTensor);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Te::Transform<AscendC::Te::Inst::Add>(sumfp32UbTensor, sumfp32UbTensor, fp32UbTensor);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
        }
        auto sumbf16UbTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, CType>(sumbf16Offset), MakeLayoutUB{}(1, paddingN));
        auto gmOutTensor = gmY.Slice(AscendC::Te::MakeCoord(tokenIndex, 0), AscendC::Te::MakeShape(1, n_));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Te::Transform<AscendC::Te::Inst::Cast, CastTraitF322Bf16>(sumbf16UbTensor, sumfp32UbTensor);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
        AscendC::Te::Copy(copyUB2GM_, gmOutTensor, sumbf16UbTensor);
    }
}

template <typename ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::WriteStatusToWin()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    uint32_t curOffset = rankId_ * FLOAT_UB_ALIGN_NUM * sizeof(float);
    auto layoutStatus = AscendC::Te::FrameLayoutFormat<LayoutStatus, float>{}(1, FLOAT_UB_ALIGN_NUM);
    auto ubStatusTensor =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(0), layoutStatus);
    auto gmStatusTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
            (__gm__ float *)(Apace::GetBaseWindStateAddrByRankId(winContext_, coreVid_) + curOffset)),
        layoutStatus);

    ubStatusTensor[AscendC::Te::MakeCoord(0, 0)] = (float)1;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(FLAG_ID_NUM_ONE);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(FLAG_ID_NUM_ONE);
    AscendC::Te::Copy(copyUB2GM_, gmStatusTensor, ubStatusTensor);
}

template <typename ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void
QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ReadStatus()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }

    uint32_t ubStatusOffset = UB_ALIGN_BYTES * 1;
    uint32_t ubResetOffset = ubStatusOffset + UB_ALIGN_BYTES;
    uint32_t gmStatusoffset = coreVid_ * FLOAT_UB_ALIGN_NUM * sizeof(float);
    auto layoutStatus = AscendC::Te::FrameLayoutFormat<LayoutStatus, float>{}(1, FLOAT_UB_ALIGN_NUM);
    auto ubStatusTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubStatusOffset), layoutStatus);
    auto ubResetTensor =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubResetOffset), layoutStatus);
    auto gmStatusTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
            (__gm__ float *)(Apace::GetBaseWindStateAddrByRankId(winContext_, rankId_) + gmStatusoffset)),
        layoutStatus);

    float flag = STATUS_FLAG_VAL;
    float minTarget = 1.0f - STATUS_FLAG_THREDSHOLD;
    float maxTarget = 1.0f + STATUS_FLAG_THREDSHOLD;
    while ((flag < minTarget) || (flag > maxTarget)) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(FLAG_ID_NUM_ONE);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(FLAG_ID_NUM_ONE);
        AscendC::Te::Copy(copyGM2UB_, ubStatusTensor, gmStatusTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(FLAG_ID_NUM_ONE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(FLAG_ID_NUM_ONE);
        flag = ubStatusTensor[AscendC::Te::MakeCoord(0, 0)];
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(FLAG_ID_NUM_ONE);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(FLAG_ID_NUM_ONE);
    ubResetTensor[AscendC::Te::MakeCoord(0, 0)] = 0.0f;
    AscendC::Te::Copy(copyUB2GM_, gmStatusTensor, ubResetTensor);
}

} // namespace Apace
