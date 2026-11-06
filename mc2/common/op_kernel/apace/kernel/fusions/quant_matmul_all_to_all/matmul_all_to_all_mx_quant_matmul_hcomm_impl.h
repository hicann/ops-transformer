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
#include "../../matmul/quant_batch_matmul/qbmm_kernel_slice_n.h"
#include "../../../../mc2_templates/communication/communicator/hccl_communicator.h"
#include "../../../../mc2_templates/communication/primitives/hccl_primitives.h"
#include "../../../tiling/matmul_tiling/quant_matmul_tiling_data.h"

namespace Apace {

template <typename T>
struct MatmulAllToAllMxPackFactor {
    static constexpr uint64_t VALUE = 1UL;
};

template <>
struct MatmulAllToAllMxPackFactor<fp4x2_e2m1_t> {
    static constexpr uint64_t VALUE = 2UL;
};

// HiF4: hifloat4x2_t 与 fp4x2_e2m1_t 同为 2 个 4-bit 元素打包 1 字节
template <>
struct MatmulAllToAllMxPackFactor<hifloat4x2_t> {
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
template <typename X1Type, typename X2Type, typename YType, typename TilingDataType, AscendC::HcclServerType ServerType,
          bool TransB>
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
    using LayoutA = AscendC::Te::NDExtLayoutPtn; // A 始终非转置
    // B 矩阵布局: TransB=true 为转置 (N, K), false 为非转置 (K, N);
    // MX 场景恒为转置, HiF4 场景转置/非转置均支持
    using LayoutB = AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC,
                                                    BiasType, LayoutBias>;
    using KernelImpl = Blaze::Gemm::Kernel::QuantBatchMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>;
    // Scale 元素类型由 kernel 侧 BlockMmad 按 X1Type 键控导出(ops-tensor 公共 MxScaleType):
    // MX 量化 → fp8_e8m0_t, HiF4 量化(hifloat4x2_t) → hif4_scale; 编排层不显式传 scale 类型
    using ScaleType = typename BlockMmad::ScaleType;
    using Params = typename KernelImpl::Params;
    using LocalParams = typename KernelImpl::LocalParams;
    using BlockMmadParams = typename KernelImpl::BlockMmadParams;
    using L1Params = typename KernelImpl::L1Params;
    using BlockSchedulerParams = typename KernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename KernelImpl::QBMMTiling;
    using CommunicationType =
        MC2KernelTemplate::HcclCommunicator<true, ServerType, MC2KernelTemplate::MC2AlltoAllContext, TilingDataType,
                                            MC2KernelTemplate::MC2AlltoAllPrimitives, 1, 0>;

    static constexpr uint16_t AIC_BARRIER_FLAG_ID = 8U;

    __aicore__ inline void SetupParams(Params& params, uint64_t rowOffset, const QuantMatmulTilingData& mmTile,
                                       uint64_t rankK, uint64_t rankNChunk, uint64_t scaleK)
    {
        const auto& tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        ProblemShape problemShape{mmTile.m, mmTile.n, mmTile.k, 1UL};
        BlockMmadParams mmParams;
        mmParams.aGmAddr = MatmulAllToAllMxGmAddr<X1Type>(x1_, rowOffset * rankK);
        mmParams.scaleAGmAddr = MatmulAllToAllMxGmAddr<ScaleType>(x1Scale_, rowOffset * scaleK);
        mmParams.bGmAddr = x2_;
        mmParams.scaleBGmAddr = x2Scale_;
        mmParams.cGmAddr = MatmulAllToAllMxGmAddr<YType>(workspaceGM_, rowOffset * rankNChunk);
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
        QBMMTiling qbmmParams{tilingInfo.rankDim,
                              mmTile.baseM,
                              mmTile.baseN,
                              mmTile.baseK,
                              bias_ != nullptr ? QBMMTiling::BIAS_ENABLED : QBMMTiling::BIAS_DISABLED,
                              mmTile.dbL0c,
                              false};
        // 模版中通过 selfBatchIdx 传入，将其对应的 batch 计算得到的数据单独放在 cGmSelfAddr 中
        // hcomm 无 self batch 直写: 全部 rank 结果写 Win 区供 Hcomm 提交
        // 因此 selfBatchIdx 传 rankDim 作哨兵，不会单独处理任何 batch
        LocalParams localParams{static_cast<uint64_t>(tilingInfo.rankM), tilingInfo.rankDim, nullptr, false};
        params = {problemShape, mmParams, l1Params, schedulerParams, qbmmParams, localParams};
    }

    __aicore__ inline void MatmulProcess(uint64_t startRow, uint32_t taskM, uint32_t taskCnt,
                                         const QuantMatmulTilingData& mmTile, uint64_t rankK, uint64_t rankNChunk)
    {
        // scale K 向单元数(按 scale 元素计), 由 ops-tensor 公共 MxScaleKLen 按数据类型(X1Type)导出:
        // MX 为 E8M0 单元(K/64*2), HiF4 为 hif4_scale 单元(K/64)
        const uint64_t scaleK = Blaze::Gemm::MxScaleKLen<X1Type>(rankK);
        Params params;
        SetupParams(params, startRow, mmTile, rankK, rankNChunk, scaleK);
        for (uint32_t sliceIdx = 0; sliceIdx < taskCnt; ++sliceIdx) {
            kernel_(params);
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_BARRIER_FLAG_ID);
            AscendC::CrossCoreWaitFlag<0, PIPE_S>(AIC_BARRIER_FLAG_ID);
            communication_.Process(sliceIdx);
            if (sliceIdx + 1 < taskCnt) {
                params.mmadParams.aGmAddr += MatmulAllToAllMxByteOffset<X1Type>(static_cast<uint64_t>(taskM) * rankK);
                params.mmadParams.scaleAGmAddr +=
                    MatmulAllToAllMxByteOffset<ScaleType>(static_cast<uint64_t>(taskM) * scaleK);
                params.mmadParams.cGmAddr +=
                    MatmulAllToAllMxByteOffset<YType>(static_cast<uint64_t>(taskM) * rankNChunk);
            }
        }
    }

    /**
     * Blaze writes one M tile in rank-major layout. After every AIC finishes
     * writing GM, block 0 submits that tile to Hcomm.
     */
    __aicore__ inline void ProcessBatch(uint64_t startRow, uint32_t taskM, uint32_t taskCnt,
                                        const QuantMatmulTilingData& mmTile)
    {
        const auto& tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        const uint64_t rankNChunk = static_cast<uint64_t>(tilingInfo.rankN) / tilingInfo.rankDim;
        const uint64_t rankK = static_cast<uint64_t>(tilingInfo.rankK);
        auto* communicationContext = communication_.GetContextPtr();
        communicationContext->sendBuffer = MatmulAllToAllMxGmAddr<YType>(workspaceGM_, startRow * rankNChunk);
        communicationContext->recvBuffer = MatmulAllToAllMxGmAddr<YType>(y_, startRow * rankNChunk);
        communicationContext->sendOffset = static_cast<uint64_t>(taskM) * rankNChunk * sizeof(YType);
        communicationContext->recvOffset = communicationContext->sendOffset;
        communicationContext->sendCount = static_cast<uint64_t>(taskM) * rankNChunk;
        communicationContext->strideCount = static_cast<uint64_t>(tilingInfo.rankM) * rankNChunk;
        communicationContext->hcclDataType = tilingInfo.hcclDataType;

        communication_.PrepareAll(taskCnt);
        MatmulProcess(startRow, taskM, taskCnt, mmTile, rankK, rankNChunk);
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
