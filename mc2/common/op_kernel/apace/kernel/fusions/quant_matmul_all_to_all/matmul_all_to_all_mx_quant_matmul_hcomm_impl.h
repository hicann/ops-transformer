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
 * \file matmul_all_to_all_mx_quant_matmul_hcomm_impl.h
 * \brief Apace Blaze computation stage for the QuantMatmulAllToAll Hcomm pipeline.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/matmul_all_to_all_qbmm_mx_kernel.h"
#include "../../../../mc2_templates/communication/communicator/hccl_communicator.h"
#include "../../../../mc2_templates/communication/primitives/hccl_primitives.h"
#include "../../../tiling/quant_matmul_tiling_data.h"

namespace Apace {

template <typename T>
struct MatmulAllToAllMxPackFactor {
    static constexpr uint64_t VALUE = 1UL;
};

template <>
struct MatmulAllToAllMxPackFactor<fp4x2_e2m1_t> {
    static constexpr uint64_t VALUE = 2UL;
};

template <typename T>
__aicore__ inline uint64_t MatmulAllToAllMxByteOffset(uint64_t elementCount)
{
    return elementCount * sizeof(T) / MatmulAllToAllMxPackFactor<T>::VALUE;
}

template <typename T>
__aicore__ inline GM_ADDR MatmulAllToAllMxGmAddr(GM_ADDR baseAddr, uint64_t elementCount)
{
    return baseAddr + MatmulAllToAllMxByteOffset<T>(elementCount);
}

/**
 * Dedicated APACE implementation for MatmulAllToAll MXFP4/MXFP8.
 *
 * The complete APACE path stays here: tiling interpretation, Blaze output
 * layout, main/tail address calculation and Hcomm submission. The legacy
 * MxQuantMatmulAlltoAllArch35 wrapper is intentionally not reused, matching
 * the isolation pattern of AllToAllMxQuantMatmulHcommImpl.
 */
template <typename X1Type, typename X2Type, typename YType, typename TilingDataType, AscendC::HcclServerType ServerType>
class MatmulAllToAllMxQuantMatmulHcommImpl {
public:
    explicit __aicore__ inline MatmulAllToAllMxQuantMatmulHcommImpl(TilingDataType* tilingData)
        : tilingData_(tilingData),
          communication_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                GM_ADDR workspaceGM)
    {
        x1_ = x1;
        x2_ = x2;
        bias_ = bias;
        y_ = y;
        x1Scale_ = x1Scale;
        x2Scale_ = x2Scale;
        workspaceGM_ = workspaceGM;

        communication_.Init();
    }

    __aicore__ inline void Run()
    {
        const auto& tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        if (tilingInfo.tileCnt > 0) {
            ProcessBatch(0, tilingInfo.tileM, tilingInfo.tileCnt, tilingData_->tileQbmmTilingData);
        }
        if (tilingInfo.tailCnt > 0) {
            const uint64_t mainRows = static_cast<uint64_t>(tilingInfo.tileM) * tilingInfo.tileCnt;
            ProcessBatch(mainRows, tilingInfo.tailM, tilingInfo.tailCnt, tilingData_->tailQbmmTilingData);
        }
        communication_.End();
    }

private:
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC,
                                                    BiasType, LayoutBias>;
    using KernelImpl = Blaze::Gemm::Kernel::MatmulAllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename KernelImpl::Params;
    using SliceParams = typename KernelImpl::SliceParams;
    using BlockMmadParams = typename KernelImpl::BlockMmadParams;
    using L1Params = typename KernelImpl::L1Params;
    using BlockSchedulerParams = typename KernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename KernelImpl::QBMMTiling;
    using CommunicationType =
        MC2KernelTemplate::HcclCommunicator<true, ServerType, MC2KernelTemplate::MC2AlltoAllContext, TilingDataType,
                                            MC2KernelTemplate::MC2AlltoAllPrimitives, 1, 0>;

    // Keep the QBMM kernel independent of the concrete HCCL communicator.
    struct HcommIssuePolicy {
        CommunicationType* communication;

        __aicore__ inline void IssueSlice(uint32_t sliceIndex)
        {
            communication->Process(sliceIndex);
        }
    };

    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_SCALE_FACTOR = 2UL;

    __aicore__ inline void SetupParams(Params& params, uint64_t rowOffset, uint32_t taskM, uint32_t taskCnt,
                                       const QuantMatmulTilingData& mmTile)
    {
        const auto& tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        const uint64_t rankNPerPeer = static_cast<uint64_t>(tilingInfo.rankN) / tilingInfo.rankDim;
        const uint64_t scaleK =
            Blaze::Gemm::CeilDiv(static_cast<uint64_t>(tilingInfo.rankK), MXFP_GROUP_SIZE) * MXFP_SCALE_FACTOR;

        ProblemShape problemShape{mmTile.m, mmTile.n, mmTile.k, 1UL};
        BlockMmadParams mmParams;
        mmParams.aGmAddr = MatmulAllToAllMxGmAddr<X1Type>(x1_, rowOffset * tilingInfo.rankK);
        mmParams.scaleAGmAddr = MatmulAllToAllMxGmAddr<AscendC::fp8_e8m0_t>(x1Scale_, rowOffset * scaleK);
        mmParams.bGmAddr = x2_;
        mmParams.scaleBGmAddr = x2Scale_;
        mmParams.cGmAddr = MatmulAllToAllMxGmAddr<YType>(workspaceGM_, rowOffset * rankNPerPeer);
        mmParams.biasGmAddr = bias_;

        L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
        BlockSchedulerParams schedulerParams{mmTile.baseM,
                                             mmTile.baseN,
                                             mmTile.mTailTile,
                                             mmTile.nTailTile,
                                             mmTile.mBaseTailSplitCnt,
                                             mmTile.nBaseTailSplitCnt,
                                             mmTile.mTailMain,
                                             mmTile.nTailMain};
        QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c, bias_ != nullptr};
        SliceParams sliceParams{MatmulAllToAllMxByteOffset<X1Type>(static_cast<uint64_t>(taskM) * tilingInfo.rankK),
                                MatmulAllToAllMxByteOffset<AscendC::fp8_e8m0_t>(static_cast<uint64_t>(taskM) * scaleK),
                                MatmulAllToAllMxByteOffset<YType>(static_cast<uint64_t>(taskM) * rankNPerPeer),
                                taskCnt};
        params = {problemShape,       mmParams,   l1Params,
                  schedulerParams,    qbmmParams, static_cast<uint64_t>(tilingInfo.rankM) * rankNPerPeer,
                  tilingInfo.rankDim, sliceParams};
    }

    __aicore__ inline void MatmulProcess(uint64_t startRow, uint32_t taskM, uint32_t taskCnt,
                                         const QuantMatmulTilingData& mmTile)
    {
        Params params;
        SetupParams(params, startRow, taskM, taskCnt, mmTile);
        HcommIssuePolicy commPolicy{&communication_};
        kernel_(params, commPolicy);
    }

    /**
     * Blaze writes one M tile in rank-major layout. After every AIC finishes
     * writing GM, block 0 submits that tile to Hcomm.
     */
    __aicore__ inline void ProcessBatch(uint64_t startRow, uint32_t taskM, uint32_t taskCnt,
                                        const QuantMatmulTilingData& mmTile)
    {
        const auto& tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        const uint64_t rankNPerPeer = static_cast<uint64_t>(tilingInfo.rankN) / tilingInfo.rankDim;
        auto* communicationContext = communication_.GetContextPtr();
        communicationContext->sendBuffer = MatmulAllToAllMxGmAddr<YType>(workspaceGM_, startRow * rankNPerPeer);
        communicationContext->recvBuffer = MatmulAllToAllMxGmAddr<YType>(y_, startRow * rankNPerPeer);
        communicationContext->sendOffset = static_cast<uint64_t>(taskM) * rankNPerPeer * sizeof(YType);
        communicationContext->recvOffset = communicationContext->sendOffset;
        communicationContext->sendCount = static_cast<uint64_t>(taskM) * rankNPerPeer;
        communicationContext->strideCount = static_cast<uint64_t>(tilingInfo.rankM) * rankNPerPeer;
        communicationContext->hcclDataType = tilingInfo.hcclDataType;

        communication_.PrepareAll(taskCnt);
        MatmulProcess(startRow, taskM, taskCnt, mmTile);
    }

    TilingDataType* tilingData_;
    KernelImpl kernel_{};
    CommunicationType communication_;
    GM_ADDR x1_{nullptr};
    GM_ADDR x2_{nullptr};
    GM_ADDR bias_{nullptr};
    GM_ADDR y_{nullptr};
    GM_ADDR x1Scale_{nullptr};
    GM_ADDR x2Scale_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
};

} // namespace Apace
