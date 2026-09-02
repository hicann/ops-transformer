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
 * \file all_gather_matmul_fp16_bf16.h
 * \brief
 */

#ifndef ALL_GATHER_MATMUL_FP16_BF16_H
#define ALL_GATHER_MATMUL_FP16_BF16_H

#include "lib/hccl/hccl.h"
#include "../../../common/op_kernel/mc2_mat_mul_asw_kernel.h"
#include "../../../common/op_kernel/mc2_mat_mul_asw_block.h"
#include "../../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#include "../../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"
#include "../all_gather_matmul_tiling_data.h"
#include "../agm_v2_common.h"

namespace AllGatherMatmulImpl {
using namespace AscendC;

template <uint8_t commMode>
struct HcclTypeSelector {
    static_assert(commMode == HCCL_COMM_MODE_CCU || commMode == HCCL_COMM_MODE_AICPU,
                  "commMode must be 0 (CCU) or 1 (AICPU)");
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;
};

template <>
struct HcclTypeSelector<HCCL_COMM_MODE_AICPU> {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode = HCCL_COMM_MODE_CCU>

class AllGatherMatmulFP16BF16 {
public:
    __aicore__ inline AllGatherMatmulFP16BF16() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM,
                                GM_ADDR workspaceGM, GM_ADDR gatherOut,
                                Mc2Tiling::AllGatherMatmulTilingDataV2 *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void StartNotify();          // 开始通信
    __aicore__ inline void EndNotify();            // 结束通信
    __aicore__ inline void InnerProcessGatherMM(); // 远端
    __aicore__ inline void InnerProcess();         // 计算本卡、远端数据
    __aicore__ inline void InnerProcessLocalMM();
    __aicore__ inline void MatmulKernelComputeGather(GM_ADDR aGM, GM_ADDR cGM, Mc2MatMulV3TilingData &tiling,
                                                     uint32_t count, bool isLast, bool isTail);

private:
    Mc2Tiling::AllGatherMatmulTilingDataV2 *tilingData_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR gatherAddr_;
    GM_ADDR context_;
    uint32_t rankId_;
    AscendC::HcclDataType dataType_;
    uint8_t debugMode_;
    bool notifyFlag_{false};
    typename HcclTypeSelector<ServerMode>::type hccl_;
    AscendC::HcclHandle hHandles_[MAX_HANDLE]; // 最大支持63个handleID
};

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode>
__aicore__ inline void AllGatherMatmulFP16BF16<AType, BType, BiasType, CType, ServerMode>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM, GM_ADDR gatherOut,
    Mc2Tiling::AllGatherMatmulTilingDataV2 *tilingData, __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling,
    TPipe *tPipe)
{
    // 获取tilingdata数据
    tilingData_ = tilingData;
    auto &&cfg = tilingData_->param;
    // 获取 context
    context_ = contextGM;
    // 初始化Hccl类
    hccl_.InitV2(context_, &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::AllGatherMatmulTilingDataV2, mc2CcTiling));
    // 管道初始化
    tPipe_ = tPipe;

    // 其它
    debugMode_ = tilingData_->debugMode;
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    rankId_ = hccl_.GetRankId();
    workspaceGM_ = workspaceGM;
    gatherAddr_ = gatherOut;

    if ((cfg.gatherLen != 0) || (!gatherOut)) {
        gatherAddr_ = workspaceGM_;
        workspaceGM_ += cfg.gatherLen;
    }

    // 全核通信
    if ASCEND_IS_AIC {
        if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
            notifyFlag_ = true;
        }
    }
}

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode>
__aicore__ inline void AllGatherMatmulFP16BF16<AType, BType, BiasType, CType, ServerMode>::StartNotify()
{
    if (notifyFlag_) {
        GM_ADDR sendBuff;
        GM_ADDR recvBuffer;
        auto &&cfg = tilingData_->param;
        using A_T = typename AType::T;
        uint32_t tileCnt = cfg.tileCnt;
        uint32_t tailCnt = cfg.tailCnt;
        uint32_t tilingM = tilingData_->mc2MmV3TileTilingData.tCubeTiling.M / (cfg.rankDim - 1);
        uint32_t tilingka = tilingData_->mc2MmV3TileTilingData.tCubeTiling.Ka;
        uint32_t tailM = tilingData_->mc2MmV3TailTilingData.tCubeTiling.M / (cfg.rankDim - 1);
        // 当x2为空tensor时只做通信
        if (cfg.rankN == 0) {
            tilingka = cfg.rankK;
        }
        uint64_t tileSendCount = static_cast<uint64_t>(tilingM) * static_cast<uint64_t>(tilingka);
        uint64_t tailSendCount = static_cast<uint64_t>(tailM) * static_cast<uint64_t>(tilingka);
        uint64_t stride =
            (tileSendCount * static_cast<uint64_t>(tileCnt) + tailSendCount * static_cast<uint64_t>(tailCnt));
        uint8_t repeat = 1;

        // 处理主块
        for (uint32_t i = 0; i < tileCnt; i++) {
            // 计算sendbuff，recvbufBuff偏移
            sendBuff = aGM_ + static_cast<uint64_t>(i) * tileSendCount * sizeof(A_T);
            recvBuffer = gatherAddr_ + static_cast<uint64_t>(i) * tileSendCount * sizeof(A_T);
            hHandles_[i] =
                hccl_.template AllGather<true>(sendBuff, recvBuffer, tileSendCount, dataType_, stride, repeat);
        }
        // 存在尾块
        for (uint32_t i = tileCnt; i < tileCnt + tailCnt; i++) {
            // 计算sendbuff，recvbufBuff偏移
            sendBuff = aGM_ + static_cast<uint64_t>(tileCnt) * tileSendCount * sizeof(A_T) +
                       static_cast<uint64_t>(i - tileCnt) * tailSendCount * sizeof(A_T);
            recvBuffer = gatherAddr_ + static_cast<uint64_t>(tileCnt) * tileSendCount * sizeof(A_T) +
                         static_cast<uint64_t>(i - tileCnt) * tailSendCount * sizeof(A_T);
            hHandles_[i] =
                hccl_.template AllGather<true>(sendBuff, recvBuffer, tailSendCount, dataType_, stride, repeat);
        }
    }
}

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode>
__aicore__ inline void AllGatherMatmulFP16BF16<AType, BType, BiasType, CType, ServerMode>::EndNotify()
{
    // 在最后一次计算完成后单核清除状态位置，避免核间读写依赖
    if ASCEND_IS_AIC {
        // 防止block_idx0执行过快清空RcvCnt,增加全核同步
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
        AscendC::CrossCoreWaitFlag(3);
        if (notifyFlag_ && (AscendC::GetBlockIdx() == 0)) {
            hccl_.Finalize();
        }
    }
}

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode>
__aicore__ inline void AllGatherMatmulFP16BF16<AType, BType, BiasType, CType, ServerMode>::Process()
{
    if ASCEND_IS_AIC {
        // 先发出通信
        StartNotify();

        // 计算本卡、远端数据
        InnerProcess();

        // 结束通信
        EndNotify();
    }
}

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode>
__aicore__ inline void AllGatherMatmulFP16BF16<AType, BType, BiasType, CType, ServerMode>::InnerProcess()
{
    // 计算本卡数据
    InnerProcessLocalMM();

    // 获取远端数据计算
    InnerProcessGatherMM();
}

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode>
__aicore__ inline void AllGatherMatmulFP16BF16<AType, BType, BiasType, CType, ServerMode>::InnerProcessLocalMM()
{
    auto &&tiling = tilingData_->mc2MmV3LocalTilingData;
    if (GetBlockIdx() >= tiling.tCubeTiling.usedCoreNum) {
        return;
    }

    auto &&cfg = tilingData_->param;
    auto aLocalGM = aGM_;
    auto cLocalGM = cGM_;

    using C_T = typename CType::T;
    cLocalGM += (uint64_t)rankId_ * (uint64_t)cfg.rankM * (uint64_t)tiling.tCubeTiling.N * sizeof(C_T);

    if (cfg.rankN != 0) {
        Mc2MatmulV3Advanced::Mc2MatmulAswKernel<AType, BType, CType, BiasType> mmv3;
        mmv3.Init(aLocalGM, bGM_, cLocalGM, biasGM_, nullptr, nullptr, &tiling, GetTPipePtr());
        mmv3.Process();
        mmv3.End();
    }
}

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode>
__aicore__ inline void AllGatherMatmulFP16BF16<AType, BType, BiasType, CType, ServerMode>::InnerProcessGatherMM()
{
    auto &&cfg = tilingData_->param;
    // 处理gather主块
    if (debugMode_ == MC2_DEBUG_ONLY_CUBE) {
        MatmulKernelComputeGather(aGM_, cGM_, tilingData_->mc2MmV3TileTilingData, cfg.tileCnt, cfg.tailM ? false : true,
                                  false);
    } else {
        MatmulKernelComputeGather(gatherAddr_, cGM_, tilingData_->mc2MmV3TileTilingData, cfg.tileCnt,
                                  cfg.tailM ? false : true, false);
    }

    if (cfg.tailM) {
        if (debugMode_ == MC2_DEBUG_ONLY_CUBE) {
            Mc2MatMulV3TilingData tileTilingData = tilingData_->mc2MmV3TailTilingData;
            uint64_t tileM = tileTilingData.tCubeTiling.M;
            uint64_t tileN = tileTilingData.tCubeTiling.N;
            uint64_t tileK = tileTilingData.tCubeTiling.Ka;
            using A_T = typename AType::T;
            using C_T = typename CType::T;
            auto tailAGM = aGM_ + tileM * tileK * sizeof(A_T) * (uint64_t)(cfg.tileCnt);
            auto tailCGM = cGM_ + tileM * tileN * sizeof(C_T) * (uint64_t)(cfg.tileCnt);
            MatmulKernelComputeGather(tailAGM, tailCGM, tilingData_->mc2MmV3TailTilingData, cfg.tailCnt, true, true);
        } else {
            MatmulKernelComputeGather(gatherAddr_, cGM_, tilingData_->mc2MmV3TailTilingData, cfg.tailCnt, true, true);
        }
    }
}

template <typename AType, typename BType, typename BiasType, typename CType, uint8_t ServerMode>
__aicore__ inline void AllGatherMatmulFP16BF16<AType, BType, BiasType, CType, ServerMode>::MatmulKernelComputeGather(
    GM_ADDR aGM, GM_ADDR cGM, Mc2MatMulV3TilingData &tiling, uint32_t count, bool isLast, bool isTail)
{
    auto &&cfg = tilingData_->param;
    cfg.rankID = rankId_;
    uint32_t shift = isTail ? cfg.tileCnt : 0;
    if ((GetBlockIdx() >= tiling.tCubeTiling.usedCoreNum) || (cfg.rankN == 0)) {
        for (uint32_t i = 0; i < count; i++) {
            if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
                hccl_.Wait(hHandles_[i + shift]);
            }
        }
        return;
    }

    MC2MatmulV3::MC2MatmulAswKernelDerive<AType, BType, CType, BiasType, MC2MatmulV3::MC2MatmulAswBlockDerive> mmv3;
    mmv3.Init(aGM, bGM_, cGM, biasGM_, nullptr, nullptr, &tiling, GetTPipePtr(), cfg, isTail, true);
    for (uint32_t i = 0; i < count; i++) {
        if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
            hccl_.Wait(hHandles_[i + shift]);
        }
        mmv3.UpdateSlice(i, isTail);
        mmv3.Process(isLast && (i == (count - 1)));
    }
    mmv3.End();
}

}; // namespace AllGatherMatmulImpl

#endif // ALL_GATHER_MATMUL_FP16_BF16_H
