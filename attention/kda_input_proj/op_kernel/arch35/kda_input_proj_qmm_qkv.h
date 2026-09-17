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
 * \file kda_input_proj_qmm_qkv.h
 * \brief Stage2 AIC: QuantMatmul(qkv)
 */

#ifndef KDA_INPUT_PROJ_QMM_QKV_H
#define KDA_INPUT_PROJ_QMM_QKV_H

#include <cstdint>
#include "kernel_operator.h"
#include "kda_input_proj_common.h"
#include "../kda_input_proj_tiling_data.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"

namespace KdaInputProj {

template <typename TypePack>
class KdaInputProjQmmQkv {
public:
    __aicore__ inline KdaInputProjQmmQkv() {}

    __aicore__ inline void Init(__gm__ uint8_t *quantX, __gm__ uint8_t *weightQkv, __gm__ uint8_t *xScale,
                                __gm__ uint8_t *weightQkvScale, __gm__ uint8_t *qkv,
                                const optiling::KdaInputProjTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    // Quantized activation / weight for MX QMM (Stage1 DynamicMxQuant output).
    using AType = float8_e4m3_t;
    using BType = float8_e4m3_t;
    using OutType = typename TypePack::DtypeQkv;
    using BiasType = float;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Std::conditional_t<TypePack::TRANS_WEIGHT_QKV, AscendC::Te::DNExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;

    static constexpr uint64_t FULL_LOAD_MODE = 0;
    static constexpr uint32_t NO_BIAS = 0U;
    static constexpr int64_t NO_BATCH = 1L;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, LayoutA, LayoutB, AType>;
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleMx<FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutC>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    __aicore__ inline Params BuildBlazeParams() const;

    const optiling::KdaInputProjTilingData *tiling_{nullptr};
    __gm__ uint8_t *quantX_{nullptr};
    __gm__ uint8_t *weightQkv_{nullptr};
    __gm__ uint8_t *xScale_{nullptr};
    __gm__ uint8_t *weightQkvScale_{nullptr};
    __gm__ uint8_t *qkv_{nullptr};
};

template <typename TypePack>
__aicore__ inline void KdaInputProjQmmQkv<TypePack>::Init(__gm__ uint8_t *quantX, __gm__ uint8_t *weightQkv,
                                                          __gm__ uint8_t *xScale, __gm__ uint8_t *weightQkvScale,
                                                          __gm__ uint8_t *qkv,
                                                          const optiling::KdaInputProjTilingData *__restrict tiling)
{
    quantX_ = quantX;
    weightQkv_ = weightQkv;
    xScale_ = xScale;
    weightQkvScale_ = weightQkvScale;
    qkv_ = qkv;
    tiling_ = tiling;
}

template <typename TypePack>
__aicore__ inline typename KdaInputProjQmmQkv<TypePack>::Params KdaInputProjQmmQkv<TypePack>::BuildBlazeParams() const
{
    const auto &base = tiling_->baseParams;
    const auto &td = tiling_->qmmQkvParams;
    Params params{};
    params.problemShape = ProblemShape{static_cast<int64_t>(base.tSize), static_cast<int64_t>(base.qkvSize),
                                       static_cast<int64_t>(base.hiddenSize), NO_BATCH};
    params.mmadParams.aGmAddr = quantX_;
    params.mmadParams.bGmAddr = weightQkv_;
    params.mmadParams.cGmAddr = qkv_;
    params.mmadParams.biasGmAddr = nullptr;
    params.mmadParams.scaleAGmAddr = xScale_;
    params.mmadParams.scaleBGmAddr = weightQkvScale_;
    params.l1Params.kL1 = td.kL1;
    params.l1Params.scaleKL1 = td.scaleKL1;
    params.l1Params.l1BufNum = td.nBufferNum;
    params.schParams.baseM = td.baseM;
    params.schParams.baseN = td.baseN;
    params.schParams.mTailTile = td.mTailTile;
    params.schParams.nTailTile = td.nTailTile;
    params.schParams.mBaseTailSplitCnt = td.mBaseTailSplitCnt;
    params.schParams.nBaseTailSplitCnt = td.nBaseTailSplitCnt;
    params.schParams.mTailMain = td.mTailMain;
    params.schParams.nTailMain = td.nTailMain;
    params.qbmmParams.baseM = td.baseM;
    params.qbmmParams.baseN = td.baseN;
    params.qbmmParams.baseK = td.baseK;
    params.qbmmParams.isBias = NO_BIAS;
    params.qbmmParams.dbL0C = td.dbL0C;
    params.qbmmParams.bMustHitL2 = td.bMustHitL2;
    return params;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjQmmQkv<TypePack>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    MatmulKernel{}(BuildBlazeParams());
}
} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_QMM_QKV_H
