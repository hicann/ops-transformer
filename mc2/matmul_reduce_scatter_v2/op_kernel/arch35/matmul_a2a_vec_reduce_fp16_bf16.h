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
 * \file matmul_a2a_vec_reduce_fp16_bf16.h
 * \brief Matmul计算，之后通过 All2All 通信 + vec ReducSum 实现 ReduceScatter
 */

#ifndef MATMUL_A2A_VEC_REDUCE_FP16_BF16_H
#define MATMUL_A2A_VEC_REDUCE_FP16_BF16_H

#include "lib/hccl/hccl.h"
#include "../common_def.h"
#include "../../../common/op_kernel/mc2_common_def.h"
#include "../../../common/op_kernel/mc2_mat_mul_asw_kernel.h"
#include "../../../common/op_kernel/mc2_mat_mul_asw_block.h"
#include "../../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#include "../../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"
#include "../../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "../matmul_reduce_scatter_v2_c_tiling.h"
#include "../../../common/op_kernel/reduce_sum_cast_fp32.h"

#define TEMPLATE_CLASS_PARAMS \
    template <typename AType, typename BType, typename BiasType, typename CType, int TPL_COMM_MODE>
#define TEMPLATE_FUNC_PARAMS AType, BType, BiasType, CType, TPL_COMM_MODE

namespace MatmulReduceScatterV2Impl {
using namespace AscendC;
using namespace AiVReduceSumCastFp32Impl;

static constexpr uint16_t SYNC_AIC_ONLY_ALL_DET_FLAG = 4; // 用于 AIC 核间同步的 flagId
static constexpr uint16_t SYNC_AIC_AIV_DET_FLAG = 8;      // 用于 AIC 与 AIV 核间同步的 flagId
static constexpr uint64_t SYNC_MODE0 = 0;                 // 核间同步模式 0
static constexpr uint64_t SYNC_MODE2 = 2;                 // 核间同步模式 2

TEMPLATE_CLASS_PARAMS
class MatmulA2AVecReduceFP16BF16 {
public:
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM,
                                GM_ADDR workspaceGM, Mc2Tiling::MatmulReduceScatterV2TilingData *tilingData,
                                TPipe *tpipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess();
    __aicore__ inline void Compute(GM_ADDR recvGMAddr, Mc2MatMulV3TilingData &tiling, const uint32_t count,
                                   GM_ADDR sendGMAddr, const bool isLast, const bool isTail);
    __aicore__ inline void MatMulV3Compute(GM_ADDR recvGMAddr, Mc2MatMulV3TilingData &tiling, const uint32_t count,
                                           GM_ADDR sendGMAddr, const bool isLast, const bool isTail);
    __aicore__ inline void PostProcess();
    __aicore__ inline void ExecuteAicMatMulPipeline(Mc2MatMulV3TilingData &tiling, const uint32_t count,
                                                    const bool isLast, const bool isTail);
    __aicore__ inline void ExecuteAivCommReducePipeline(GM_ADDR recvGMAddr, GM_ADDR sendGMAddr,
                                                        Mc2MatMulV3TilingData &tiling, const uint32_t count,
                                                        const bool isTail);
    __aicore__ inline void CubeNotifyVector();
    __aicore__ inline void VecWaitCube();

private:
    ReduceSumForAlltoAll<C_DTYPE> reduceSum_; // AIV ReduceSum 相关实现

    Mc2Tiling::MatmulReduceScatterV2TilingData *tilingData_;
    TPipe *tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    __gm__ HcclCombinOpParam *context_;
    AscendC::HcclDataType dataType_;
    uint8_t debugMode_;
    typename HcclTypeSelector<TPL_COMM_MODE>::type hccl_;
    AscendC::HcclHandle handles_[MAX_HANDLE]; // 最大支持64个handleId
    GM_ADDR sendBuf_;                         // 存放 MatMul 输出（All2All send buffer）
    GM_ADDR recvBuf_;                         // 存放 All2All 接收的 slices（内容为
                                              // [slice_r_from_rank0][slice_r_from_rank1]...[slice_r_from_rankR-1]）
    uint32_t rankId_{0};
    uint64_t aivNum_{0};
    uint64_t fullMN_{0};
    uint64_t tileOffset_{0};
};

TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM,
    Mc2Tiling::MatmulReduceScatterV2TilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    auto &&cfg = tilingData_->param;

    // 初始化 HCCL
    const void *hcclInitTilingV2 = &(tilingData_->mc2InitTiling);
    uint64_t hcclCcTilingOffset = offsetof(Mc2Tiling::MatmulReduceScatterV2TilingData, mc2CcTiling);
    hccl_.InitV2(contextGM, hcclInitTilingV2);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);

    // 读取上下文和配置
    context_ = (__gm__ HcclCombinOpParam *)(contextGM);
    tPipe_ = tPipe;
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    debugMode_ = tilingData_->debugMode;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    rankId_ = context_->rankId;
    aivNum_ = cfg.aicCoreNum * GetTaskRation(); // 启用的AIV 数量, 此模板会全启用

    // all2all 通信相关参数, 划分workspace
    fullMN_ = static_cast<uint64_t>(cfg.rankM) * static_cast<uint64_t>(cfg.rankN); // M * N
    sendBuf_ = workspaceGM;                                                        // [0, fullMN)
    recvBuf_ = sendBuf_ + fullMN_ * sizeof(C_DTYPE);                               // [fullMN, 2*fullMN)
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::PostProcess()
{
    // 等待执行完成后，最后终止hcclserver
    if ((GetBlockIdx() == 0) && (g_coreType == AIV)) {
        hccl_.Finalize();
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::Process()
{
    InnerProcess(); // 核心计算+通信
    PostProcess();  // 等待计算与通信完成, 终止hcclserver
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::InnerProcess()
{
    auto &&tiling = tilingData_->mC2Mmv3TileTilingData.tCubeTiling;
    auto &&cfg = tilingData_->param;

    // fullmesh算法
    // 计算主块（整除部分）
    Compute(recvBuf_, tilingData_->mC2Mmv3TileTilingData, cfg.tileCnt, sendBuf_, cfg.tailM ? false : true, false);
    // 计算尾块（非整除部分）
    if (cfg.tailM) {
        uint64_t tileSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
        tileOffset_ = tileSize * static_cast<uint64_t>(cfg.tileCnt) * sizeof(C_DTYPE);
        auto recvGMTail = recvBuf_ + tileOffset_;
        auto sendBufTail = sendBuf_ + tileOffset_;
        Compute(recvGMTail, tilingData_->mC2Mmv3TailTilingData, cfg.tailCnt, sendBufTail, true, true);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::Compute(GM_ADDR recvGMAddr,
                                                                                 Mc2MatMulV3TilingData &tiling,
                                                                                 const uint32_t count,
                                                                                 GM_ADDR sendGMAddr, const bool isLast,
                                                                                 const bool isTail)
{
    // Cube 核执行 MatMul, Vector 核执行 all2all + reduceSum
    MatMulV3Compute(recvGMAddr, tiling, count, sendGMAddr, isLast, isTail);
}

/**
 * @brief [AIC] 执行 MatMul 计算流水线
 */
TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::ExecuteAicMatMulPipeline(
    Mc2MatMulV3TilingData &tiling, const uint32_t count, const bool isLast, const bool isTail)
{
    auto &&cfg = tilingData_->param;
    cfg.rankID = rankId_;

    // 初始化Matmul
    MC2MatmulV3::MC2MatmulAswKernelDerive<AType, BType, CType, BiasType, MC2MatmulV3::MC2MatmulAswBlockDerive> mmv3;
    mmv3.Init(aGM_, bGM_, sendBuf_, biasGM_, nullptr, nullptr, &tiling, GetTPipePtr(), cfg, isTail, false);

    for (uint32_t i = 0; i < count; i++) {
        mmv3.UpdateSlice(i, isTail);                // 更新 slice 偏移
        mmv3.Process(isLast && (i == (count - 1))); // 执行 MatMul

        // AIC 侧做完 Matmul 计算后通知 AIV 进行后处理
        CubeNotifyVector();
    }

    mmv3.End();
}

/**
 * @brief [AIV] 执行 All2All 通信 + ReduceSum 归约流水线 (双发模式)
 */
TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::ExecuteAivCommReducePipeline(
    GM_ADDR recvGMAddr, GM_ADDR sendGMAddr, Mc2MatMulV3TilingData &tiling, const uint32_t count, const bool isTail)
{
    auto &&cfg = tilingData_->param;

    // 每个 rank 应得的 M 维度大小
    uint64_t sliceM = static_cast<uint64_t>(tiling.tCubeTiling.M) / cfg.rankDim;
    // 每个 rank 分片的元素数量（M_per_rank * N）
    uint64_t rankSliceElems = sliceM * static_cast<uint64_t>(tiling.tCubeTiling.N);
    // 每个 rank 分片的字节数
    uint64_t rankSliceBytes = rankSliceElems * sizeof(C_DTYPE);

    // All2All 的 stride（单位：元素数），即每张卡数据在全局中的间隔
    uint64_t stride = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    // 通信重复次数, 1次
    uint8_t repeat = 1;
    // 若是尾块，通信 handleId 存放的起始偏移为 tileCnt
    uint32_t handleShift = isTail ? cfg.tileCnt : 0;

    // --- 指针初始化 ---
    GM_ADDR currSendPtr = sendGMAddr;                        // 当前发送缓冲区起始地址
    GM_ADDR currRecvPtr = recvGMAddr;                        // 当前接收缓冲区起始地址
    GM_ADDR currOutPtr = isTail ? cGM_ + tileOffset_ : cGM_; // 当前 reduceSum 输出的起始地址

    // --- Prologue: 启动第 0 轮通信 ---
    VecWaitCube(); // 确保依赖的 MatMul 已完成
    handles_[0 + handleShift] =
        hccl_.template AlltoAll<true>(currSendPtr, currRecvPtr, rankSliceElems, dataType_, stride, repeat);

    // 移动指针准备下一轮
    currSendPtr += rankSliceBytes;
    currRecvPtr += rankSliceBytes;
    currOutPtr += rankSliceBytes;

    // --- Loop: 重叠执行 (等待上一轮 + 归约上一轮 + 启动下一轮) ---
    for (uint32_t i = 0; i < count - 1; i++) {
        // 1. 等待上一轮 (i) 通信结束
        if (GetBlockIdx() == 0) {
            hccl_.Wait(handles_[i + handleShift]);
        }
        SyncAll<true>();

        // 2. 启动下一轮 (i+1) 通信
        VecWaitCube();
        handles_[i + 1 + handleShift] =
            hccl_.template AlltoAll<true>(currSendPtr, currRecvPtr, rankSliceElems, dataType_, stride, repeat);

        // 3. 执行上一轮 (i) 数据的 ReduceSum
        // 计算地址 = 当前指针 - 一步偏移
        GM_ADDR calcRecvPtr = currRecvPtr - rankSliceBytes;
        GM_ADDR calcOutPtr = currOutPtr - rankSliceBytes;

        tPipe_->Reset();
        reduceSum_.Init(rankSliceElems, stride, cfg.rankDim, aivNum_, calcRecvPtr, calcOutPtr, tPipe_);
        reduceSum_.ExecuteReduceSum();

        // 指针前移
        currSendPtr += rankSliceBytes;
        currRecvPtr += rankSliceBytes;
        currOutPtr += rankSliceBytes;
    }

    // --- Epilogue: 处理最后一轮 ---
    uint32_t lastIdx = count - 1;

    // 等待最后一轮通信结束
    if (GetBlockIdx() == 0) {
        hccl_.Wait(handles_[lastIdx + handleShift]);
    }
    SyncAll<true>();

    // 执行最后一轮数据的 reduceSum
    // 此时的计算地址同样是 "当前指针 - 偏移量"
    GM_ADDR calcRecvPtr = currRecvPtr - rankSliceBytes;
    GM_ADDR calcOutPtr = currOutPtr - rankSliceBytes;

    tPipe_->Reset();
    reduceSum_.Init(rankSliceElems, stride, cfg.rankDim, aivNum_, calcRecvPtr, calcOutPtr, tPipe_);
    reduceSum_.ExecuteReduceSum();
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::MatMulV3Compute(
    GM_ADDR recvGMAddr, Mc2MatMulV3TilingData &tiling, const uint32_t count, GM_ADDR sendGMAddr, const bool isLast,
    const bool isTail)
{
    // [AIC 阶段] 执行 MatMul 计算流水线
    if ASCEND_IS_AIC {
        ExecuteAicMatMulPipeline(tiling, count, isLast, isTail);
    }

    // [AIV 阶段] 执行 All2All 通信 + ReduceSum 归约流水线
    if ASCEND_IS_AIV {
        ExecuteAivCommReducePipeline(recvGMAddr, sendGMAddr, tiling, count, isTail);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::CubeNotifyVector()
{
    // 先全 AIC 同步一次
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_DET_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_DET_FLAG);
    // 通知 AIV
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_DET_FLAG);
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatmulA2AVecReduceFP16BF16<TEMPLATE_FUNC_PARAMS>::VecWaitCube()
{
    // 等待 AIC 完成
    // 修复同步问题，适配AICPU接口（等待管线由 PIPE_MTE2 改为 PIPE_S）
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_S>(SYNC_AIC_AIV_DET_FLAG);
}

} // namespace MatmulReduceScatterV2Impl

#endif // MATMUL_A2A_VEC_REDUCE_FP16_BF16_H
