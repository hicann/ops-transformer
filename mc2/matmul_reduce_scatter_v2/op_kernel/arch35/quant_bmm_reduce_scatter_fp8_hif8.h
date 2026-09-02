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
 * \file quant_bmm_reduce_scatter_fp8_hif8.h
 * \brief
 */

#ifndef QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H
#define QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H

#include "lib/hccl/hccl.h"
#include "../common_def.h"
#include "../../../common/op_kernel/mc2_common_def.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_perblock.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"
#include "../../../common/op_kernel/mc2_quant_batch_matmul.h"
#include "../../../common/op_kernel/qbmm_mix_perblock_noncontiguous.h"
#include "../matmul_reduce_scatter_v2_c_tiling.h"

#define TEMPLATE_CLASS_PARAMS \
    template <typename AType, typename BType, typename CType, typename ScaleType, class MMClass, bool IsPerBlock, \
              bool ATrans, bool BTrans, int TPL_COMM_MODE>
#define TEMPLATE_FUNC_PARAMS AType, BType, CType, ScaleType, MMClass, IsPerBlock, ATrans, BTrans, TPL_COMM_MODE

namespace MatmulReduceScatterV2Impl {
using namespace AscendC;

TEMPLATE_CLASS_PARAMS
class QuantBMMReduceScatter {
public:
    __aicore__ inline QuantBMMReduceScatter() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM,
                                Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData *tilingData, TPipe *tpipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess();
    __aicore__ inline void MatMulReduceScatterSerial();
    __aicore__ inline void MatMulComputReduceScatter(GM_ADDR aGM, GM_ADDR cGM, GM_ADDR x1ScaleGM,
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBMmtiling,
                                                     uint32_t tileCnt, GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void MatMulComputReduceScatterPertensor(
        GM_ADDR aGM, GM_ADDR cGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBMmtiling, uint32_t tileCnt,
        GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void MatMulComputReduceScatterPerblock(
        GM_ADDR aGM, GM_ADDR cGM, GM_ADDR x1ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBMmtiling,
        uint32_t tileCnt, GM_ADDR gmToFloat, bool isTail);
    __aicore__ inline void PostProcess(); // 计算后处理，等待通信结束，并终止hcclserver

private:
    Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData *tilingData_;
    TPipe *tPipe_{nullptr};
    GM_ADDR aGM_{nullptr};
    GM_ADDR bGM_{nullptr};
    GM_ADDR cGM_{nullptr};
    GM_ADDR biasGM_{nullptr};
    GM_ADDR x1ScaleGM_{nullptr};
    GM_ADDR x2ScaleGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR gmToFloat_{nullptr};
    __gm__ HcclCombinOpParam *context_{nullptr};
    uint32_t rankId_{0};
    AscendC::HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
    uint8_t debugMode_{0};
    typename HcclTypeSelector<TPL_COMM_MODE>::type hccl_;
    AscendC::HcclHandle handles_[MAX_HANDLE]; // 最大支持64个handleId
    uint64_t preCoreNum_ = 0;
    uint32_t batchWeight_[MAX_HANDLE] = {0};
};

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM, GM_ADDR cGM, GM_ADDR contextGM,
    GM_ADDR workspaceGM, Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    const void *hcclInitTilingV2 = &(tilingData_->mc2InitTiling);
    uint64_t hcclCcTilingOffset = offsetof(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData, mc2CcTiling);
    hccl_.InitV2(contextGM, hcclInitTilingV2);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);
    context_ = (__gm__ HcclCombinOpParam *)(contextGM);
    tPipe_ = tPipe;
    debugMode_ = tilingData_->debugMode;
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    x1ScaleGM_ = x1ScaleGM;
    x2ScaleGM_ = x2ScaleGM;
    rankId_ = context_->rankId;
    auto &&cfg = tilingData_->param;
    for (uint32_t j = 0; j < cfg.rankDim; j++) {
        batchWeight_[j] = j;
    }
    // 划分workspace
    gmToFloat_ = workspaceGM;
    workspaceGM_ = gmToFloat_ + cfg.cToFloatLen;
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::PostProcess()
{
    auto &&cfg = tilingData_->param;
    // 等待reducescatter执行完成
    if ((GetBlockIdx() == 0) && (g_coreType == AIC)) {
        for (uint32_t i = 0; i < cfg.tileCnt + cfg.tailCnt; i++) {
            hccl_.Wait(handles_[i]);
        }
        // 终止hcclserver
        hccl_.Finalize();
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::Process()
{
    InnerProcess();
    PostProcess();
}

// perblock量化场景当rankM不满足128*ranksize，走计算通信串行
TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::MatMulReduceScatterSerial()
{
    auto &&qBMmtiling = tilingData_->quantBmmV3TileTiling;
    auto &&tiling = qBMmtiling.matmulTiling;
    auto &&cfg = tilingData_->param;
    auto cWork = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM_ : gmToFloat_;
    auto recvBuffer = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? gmToFloat_ : cGM_;

    uint64_t recvCount = static_cast<uint64_t>(tiling.M / cfg.rankDim) * static_cast<uint64_t>(tiling.N);
    this->tPipe_->Destroy();
    this->tPipe_->Init();
    MMClass op;
    uint32_t strideCount = cfg.rankM / cfg.rankDim;
    op.Init(aGM_, bGM_, biasGM_, x2ScaleGM_, x1ScaleGM_, cWork, workspaceGM_, &qBMmtiling, tPipe_, batchWeight_,
            strideCount, false);
    op.Process();
    SyncAll<false>();
    uint64_t stride = 0;
    uint8_t repeat = 1;
    if ASCEND_IS_AIC {
        recvCount = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? 1 : recvCount;
        handles_[0] = hccl_.template ReduceScatter<true>(cWork, recvBuffer, recvCount, dataType_,
                                                         HcclReduceOp::HCCL_REDUCE_SUM, stride, repeat);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::InnerProcess()
{
    auto &&tiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
    auto &&cfg = tilingData_->param;
    uint64_t aSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka);
    uint64_t cSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);

    // perblock场景下，未进行公式化切分，通算串行
    if constexpr (IsPerBlock) {
        if (tiling.M == cfg.rankM) {
            MatMulReduceScatterSerial();
            return;
        }
    }

    // fullmesh算法
    // 计算主块
    MatMulComputReduceScatter(aGM_, cGM_, x1ScaleGM_, tilingData_->quantBmmV3TileTiling, cfg.tileCnt, gmToFloat_,
                              (cfg.tailM ? false : true), false);
    // 计算尾块
    if (cfg.tailM) {
        if constexpr (!IsPerBlock) {
            cSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
            auto aGMTail = aGM_;
            uint64_t tileOffset = static_cast<uint64_t>(cfg.tileCnt) * cSize * sizeof(CType);
            auto cGMTail = cGM_ + tileOffset;
            auto gmToFloatTail = gmToFloat_ + tileOffset;
            MatMulComputReduceScatter(aGMTail, cGMTail, nullptr, tilingData_->quantBmmV3TailTiling, cfg.tailCnt,
                                      gmToFloatTail, true, true);
            return;
        }

        auto aGMTail = aGM_ + aSize * sizeof(AType) * static_cast<uint64_t>(cfg.tileCnt);
        auto cGMTail = cGM_ + cSize * sizeof(CType) * static_cast<uint64_t>(cfg.tileCnt);
        auto gmToFloatTail = gmToFloat_ + static_cast<uint64_t>(cfg.tileCnt) * static_cast<uint64_t>(cfg.rankDim) *
                                              cSize * sizeof(CType);
        auto x1ScaleTail = x1ScaleGM_ + static_cast<uint64_t>(tiling.M) / BLOCK_SIZE *
                                            CeilDiv(static_cast<uint64_t>(tiling.Ka), BLOCK_SIZE) * sizeof(ScaleType) *
                                            static_cast<uint64_t>(cfg.tileCnt);
        MatMulComputReduceScatter(aGMTail, cGMTail, x1ScaleTail, tilingData_->quantBmmV3TailTiling, cfg.tailCnt,
                                  gmToFloatTail, true, true);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatterPertensor(
    GM_ADDR aGM, GM_ADDR cGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBMmtiling, uint32_t tileCnt,
    GM_ADDR gmToFloat, bool isLast, bool isTail)
{
    if ASCEND_IS_AIV {
        return;
    }
    auto &&cfg = tilingData_->param;
    auto &&tiling = qBMmtiling.matmulTiling;
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        for (uint32_t i = 0; i < tileCnt; i++) {
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
            AscendC::CrossCoreWaitFlag(3);
        }
        return;
    }
    // 如果尾块的当前核数大于主块使用核数，计算尾块当前核的preCoreNum_
    if (isTail && (GetBlockIdx() >= tilingData_->quantBmmV3TileTiling.matmulTiling.usedCoreNum)) {
        auto &&tileTiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
        uint64_t headSliceM = (cfg.rankM / cfg.rankDim - cfg.tailM * cfg.tailCnt) / cfg.tileCnt;
        uint64_t mCnt = DequantBmm::CeilDiv(headSliceM, tileTiling.baseM) * cfg.rankDim;
        uint64_t nCnt = DequantBmm::CeilDiv(tileTiling.N, tileTiling.baseN);
        preCoreNum_ = (mCnt * nCnt * cfg.tileCnt) % tileTiling.usedCoreNum;
    }
    auto recvCount = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
    auto cOffset = recvCount * sizeof(CType);
    // 归一化Matmul计算类，负责MC2的Matmul计算
    auto cWork = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat;
    auto recvBuffer = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? gmToFloat : cGM;
    auto shift = isTail ? cfg.tileCnt : 0;
    uint64_t stride = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    uint8_t repeat = 1;
    tPipe_->Reset();
    Mc2MatmulV3::Mc2QuantBatchMatmulASWKernel<AType, BType, ScaleType, float, CType, CubeFormat::ND, CubeFormat::ND,
                                              CubeFormat::ND, ATrans, BTrans>
        mmv3;
    auto tempGM = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat_;
    mmv3.Init(aGM_, bGM_, biasGM_, x2ScaleGM_, x1ScaleGM_, tempGM, workspaceGM_, &qBMmtiling, GetTPipePtr(), cfg,
              isTail, false, preCoreNum_);
    for (uint32_t i = 0; i < tileCnt; i++) {
        mmv3.UpdateSlice(i, isTail);
        mmv3.Process(isLast && (i == (tileCnt - 1)));
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
        AscendC::CrossCoreWaitFlag(3);
        recvCount = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? 1 : recvCount;
        handles_[i + shift] = hccl_.template ReduceScatter<true>(cWork, recvBuffer, recvCount, dataType_,
                                                                 HcclReduceOp::HCCL_REDUCE_SUM, stride, repeat);
        cWork += cOffset;
        recvBuffer += cOffset;
    }
    preCoreNum_ = mmv3.GetPreCoreNum();
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatterPerblock(
    GM_ADDR aGM, GM_ADDR cGM, GM_ADDR x1ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBMmtiling,
    uint32_t tileCnt, GM_ADDR gmToFloat, bool isTail)
{
    auto &&cfg = tilingData_->param;
    auto &&tiling = qBMmtiling.matmulTiling;
    auto recvCount = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
    auto aOffset = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka) * sizeof(AType);
    auto cOffset = recvCount * sizeof(CType);
    auto raOffset = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankK) / cfg.rankDim * sizeof(AType);
    // 归一化Matmul计算类，负责MC2的Matmul计算

    auto aAddr = aGM;
    auto x1ScaleAddr = x1ScaleGM;
    auto cWork = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat;
    auto recvBuffer = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? gmToFloat : cGM;
    auto shift = isTail ? cfg.tileCnt : 0;
    uint64_t stride = 0;
    uint8_t repeat = 1;
    uint32_t strideCount = cfg.rankM / cfg.rankDim;

    // 卡内偏移
    uint64_t x1ScaleOffset = static_cast<uint64_t>(tiling.M / BLOCK_SIZE) *
                             CeilDiv(static_cast<uint64_t>(tiling.Ka), BLOCK_SIZE) * sizeof(ScaleType);
    for (uint32_t i = 0; i < tileCnt; i++) {
        this->tPipe_->Destroy();
        this->tPipe_->Init();
        MMClass op;
        op.Init(aAddr, bGM_, biasGM_, x2ScaleGM_, x1ScaleAddr, cWork, workspaceGM_, &qBMmtiling, tPipe_, batchWeight_,
                strideCount, false);
        op.Process();
        PipeBarrier<PIPE_V>();
        SyncAll<false>();
        if ASCEND_IS_AIC {
            recvCount = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? 1 : recvCount;
            handles_[i + shift] = hccl_.template ReduceScatter<true>(cWork, recvBuffer, recvCount, dataType_,
                                                                     HcclReduceOp::HCCL_REDUCE_SUM, stride, repeat);
        }
        aAddr += aOffset;
        cWork += cfg.rankDim * cOffset;
        x1ScaleAddr += x1ScaleOffset;
        recvBuffer += cOffset;
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBMMReduceScatter<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatter(
    GM_ADDR aGM, GM_ADDR cGM, GM_ADDR x1ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBMmtiling,
    uint32_t tileCnt, GM_ADDR gmToFloat, bool isLast, bool isTail)
{
    if constexpr (IsPerBlock) {
        MatMulComputReduceScatterPerblock(aGM, cGM, x1ScaleGM, qBMmtiling, tileCnt, gmToFloat, isTail);
    } else {
        MatMulComputReduceScatterPertensor(aGM, cGM, qBMmtiling, tileCnt, gmToFloat, isLast, isTail);
    }
}
} // namespace MatmulReduceScatterV2Impl

#endif // QUANT_BMM_MATMUL_REDUCE_SCATTER_FP8_HIF8_H
