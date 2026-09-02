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
 * \file matmul_reduce_scatter_fp16_bf16.h
 * \brief
 */

#ifndef MATMUL_REDUCE_SCATTER_FP16_BF16_H
#define MATMUL_REDUCE_SCATTER_FP16_BF16_H

#include "lib/hccl/hccl.h"
#include "../common_def.h"
#include "../../../common/op_kernel/mc2_common_def.h"
#include "../../../common/op_kernel/mc2_mat_mul_asw_kernel.h"
#include "../../../common/op_kernel/mc2_mat_mul_asw_block.h"
#include "../../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#include "../../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"
#include "../../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "../matmul_reduce_scatter_v2_c_tiling.h"

namespace MatmulReduceScatterV2Impl {
using namespace AscendC;

template <typename AType, typename BType, typename BiasType, typename CType, int TPL_COMM_MODE>
class MatmulReduceScatterFP16BF16 {
public:
    __aicore__ inline MatmulReduceScatterFP16BF16() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM,
                                GM_ADDR workspaceGM, Mc2Tiling::MatmulReduceScatterV2TilingData *tilingData,
                                TPipe *tpipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess();
    __aicore__ inline void Compute(GM_ADDR cGM, Mc2MatMulV3TilingData &tiling, uint32_t count, GM_ADDR gmToFloat,
                                   bool isLast, bool isTail);
    __aicore__ inline void MatMulV3Compute(GM_ADDR cGM, Mc2MatMulV3TilingData &tiling, uint32_t count,
                                           GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void PostProcess(); // 计算后处理，等待通信结束，并终止hcclserver

private:
    Mc2Tiling::MatmulReduceScatterV2TilingData *tilingData_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR gmToFloat_;
    __gm__ HcclCombinOpParam *context_;
    uint32_t rankId_;
    AscendC::HcclDataType dataType_;
    uint8_t debugMode_;
    typename HcclTypeSelector<TPL_COMM_MODE>::type hccl_;
    AscendC::HcclHandle handles_[MAX_HANDLE]; // 最大支持64个handleId
};

template <typename AType, typename BType, typename BiasType, typename CType, int TPL_COMM_MODE>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType, TPL_COMM_MODE>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM,
    Mc2Tiling::MatmulReduceScatterV2TilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    auto &&cfg = tilingData_->param;
    const void *hcclInitTilingV2 = &(tilingData_->mc2InitTiling);
    uint64_t hcclCcTilingOffset = offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, mc2CcTiling);
    hccl_.InitV2(contextGM, hcclInitTilingV2);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);
    context_ = (__gm__ HcclCombinOpParam *)(contextGM);
    tPipe_ = tPipe;
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    debugMode_ = tilingData_->debugMode;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    rankId_ = context_->rankId;

    // 划分workspace
    gmToFloat_ = workspaceGM;
}

template <typename AType, typename BType, typename BiasType, typename CType, int TPL_COMM_MODE>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType, TPL_COMM_MODE>::PostProcess()
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

template <typename AType, typename BType, typename BiasType, typename CType, int TPL_COMM_MODE>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType, TPL_COMM_MODE>::Process()
{
    InnerProcess();
    PostProcess();
}

template <typename AType, typename BType, typename BiasType, typename CType, int TPL_COMM_MODE>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType, TPL_COMM_MODE>::InnerProcess()
{
    auto &&tiling = tilingData_->mC2Mmv3TileTilingData.tCubeTiling;
    auto &&cfg = tilingData_->param;

    // fullmesh算法
    // 计算主块
    Compute(cGM_, tilingData_->mC2Mmv3TileTilingData, cfg.tileCnt, gmToFloat_, cfg.tailM ? false : true, false);
    // 计算尾块
    if (cfg.tailM) {
        uint64_t tileSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
        uint64_t tileOffset = tileSize * static_cast<uint64_t>(cfg.tileCnt) * sizeof(C_DTYPE);
        auto cGMTail = cGM_ + tileOffset;
        auto gmToFloatTail = gmToFloat_ + tileOffset;
        Compute(cGMTail, tilingData_->mC2Mmv3TailTilingData, cfg.tailCnt, gmToFloatTail, true, true);
    }
}

template <typename AType, typename BType, typename BiasType, typename CType, int TPL_COMM_MODE>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType, TPL_COMM_MODE>::Compute(
    GM_ADDR cGM, Mc2MatMulV3TilingData &tiling, uint32_t count, GM_ADDR gmToFloat, bool isLast, bool isTail)
{
    if ASCEND_IS_AIV {
        return;
    }

    if (block_idx >= tiling.tCubeTiling.usedCoreNum) {
        for (uint32_t i = 0; i < count; i++) {
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
            AscendC::CrossCoreWaitFlag(3);
        }
        return;
    }
    // matmul的一次计算流程
    MatMulV3Compute(cGM, tiling, count, gmToFloat, isLast, isTail);
}

template <typename AType, typename BType, typename BiasType, typename CType, int TPL_COMM_MODE>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType, TPL_COMM_MODE>::MatMulV3Compute(
    GM_ADDR cGM, Mc2MatMulV3TilingData &tiling, uint32_t count, GM_ADDR gmToFloat, bool isLast, bool isTail)
{
    using cDataType = typename CType::T;
    auto &&cfg = tilingData_->param;
    cfg.rankID = rankId_;
    MC2MatmulV3::MC2MatmulAswKernelDerive<AType, BType, CType, BiasType, MC2MatmulV3::MC2MatmulAswBlockDerive> mmv3;
    auto tempGM = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat_;
    mmv3.Init(aGM_, bGM_, tempGM, biasGM_, nullptr, nullptr, &tiling, GetTPipePtr(), cfg, isTail, false);
    uint64_t sliceM = static_cast<uint64_t>(tiling.tCubeTiling.M) / cfg.rankDim;
    auto recvCount = sliceM * static_cast<uint64_t>(tiling.tCubeTiling.N);
    auto cOffset = recvCount * sizeof(cDataType);
    auto cWork = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat;
    auto recvBuffer = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? gmToFloat : cGM;
    auto shift = isTail ? cfg.tileCnt : 0;
    uint64_t stride = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    uint8_t repeat = 1;
    for (uint32_t i = 0; i < count; i++) {
        mmv3.UpdateSlice(i, isTail);
        mmv3.Process(isLast && (i == (count - 1)));
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
        AscendC::CrossCoreWaitFlag(3);
        recvCount = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? 1 : recvCount;
        handles_[i + shift] = hccl_.template ReduceScatter<true>(cWork, recvBuffer, recvCount, dataType_,
                                                                 HcclReduceOp::HCCL_REDUCE_SUM, stride, repeat);
        cWork += cOffset;
        recvBuffer += cOffset;
    }
    mmv3.End();
}
} // namespace MatmulReduceScatterV2Impl

#endif // MATMUL_REDUCE_SCATTER_FP16_BF16_H
