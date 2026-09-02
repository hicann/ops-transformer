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
 * \file quant_bmm_a2a_vec_reduce_fp8_hif8.h
 * \brief Quantized BMM 计算，之后通过 All2All 通信 + vec ReducSum 实现 ReduceScatter
 */

#ifndef QUANT_BMM_A2A_VEC_REDUCE_FP8_HIF8_H
#define QUANT_BMM_A2A_VEC_REDUCE_FP8_HIF8_H

#include "lib/hccl/hccl.h"
#include "../common_def.h"
#include "../../../common/op_kernel/mc2_common_def.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_perblock.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"
#include "../../../common/op_kernel/mc2_quant_batch_matmul.h"
#include "../../../common/op_kernel/qbmm_mix_perblock_noncontiguous.h"
#include "../matmul_reduce_scatter_v2_c_tiling.h"
#include "../../../common/op_kernel/reduce_sum_cast_fp32.h"

#define TEMPLATE_CLASS_PARAMS \
    template <typename AType, typename BType, typename CType, typename ScaleType, class MMClass, bool IsPerBlock, \
              bool ATrans, bool BTrans, int TPL_COMM_MODE>
#define TEMPLATE_FUNC_PARAMS AType, BType, CType, ScaleType, MMClass, IsPerBlock, ATrans, BTrans, TPL_COMM_MODE

namespace MatmulReduceScatterV2Impl {
using namespace AscendC;
using namespace AiVReduceSumCastFp32Impl;

static constexpr uint16_t SYNC_AIC_ONLY_ALL_DET_FLAG = 4; // 用于 AIC 核间同步的 flagId
static constexpr uint16_t SYNC_AIC_AIV_DET_FLAG = 8;      // 用于 AIC 与 AIV 核间同步的 flagId
static constexpr uint64_t SYNC_MODE0 = 0;                 // 核间同步模式 0
static constexpr uint64_t SYNC_MODE2 = 2;                 // 核间同步模式 2

TEMPLATE_CLASS_PARAMS
class QuantBmmA2AVecReduceFP8HiF8 {
public:
    __aicore__ inline QuantBmmA2AVecReduceFP8HiF8() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM,
                                Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData *tilingData, TPipe *tpipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess();
    __aicore__ inline void MatMulReduceScatterSerial();
    __aicore__ inline void MatMulComputReduceScatter(GM_ADDR aGM, GM_ADDR recvGM, GM_ADDR x1ScaleGM,
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling,
                                                     const uint32_t count, GM_ADDR sendGM, const bool isLast,
                                                     const bool isTail);
    __aicore__ inline void MatMulComputReduceScatterPertensor(
        GM_ADDR recvGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling, const uint32_t count,
        GM_ADDR sendGM, const bool isLast, const bool isTail);
    __aicore__ inline void MatMulComputReduceScatterPerblock(
        GM_ADDR aGM, GM_ADDR recvGM, GM_ADDR x1ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling,
        const uint32_t count, GM_ADDR sendGM, const bool isTail);
    __aicore__ inline void PostProcess();
    __aicore__ inline void PrepareTailConfig(DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling,
                                             const bool isTail);
    __aicore__ inline void ExecuteAicMatMulPipeline(DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling,
                                                    const uint32_t count, const bool isLast, const bool isTail);
    __aicore__ inline void ExecuteAivCommReducePipeline(GM_ADDR recvGM, GM_ADDR sendGM, const uint32_t count,
                                                        const bool isTail,
                                                        DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling);
    __aicore__ inline void CubeNotifyVector();
    __aicore__ inline void VecWaitCube();

private:
    ReduceSumForAlltoAll<CType> reduceSum_; // AIV ReduceSum 相关实现

    Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData *tilingData_;
    TPipe *tPipe_{nullptr};
    GM_ADDR aGM_{nullptr};
    GM_ADDR bGM_{nullptr};
    GM_ADDR cGM_{nullptr};
    GM_ADDR biasGM_{nullptr};
    GM_ADDR x1ScaleGM_{nullptr};
    GM_ADDR x2ScaleGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR sendBuf_{nullptr}; // 存放 MatMul 输出（All2All send buffer）
    GM_ADDR recvBuf_{nullptr}; // 存放 All2All 接收的 slices（内容为
                               // [slice_r_from_rank0][slice_r_from_rank1]...[slice_r_from_rankR-1]）
    __gm__ HcclCombinOpParam *context_{nullptr};
    uint32_t rankId_{0};
    AscendC::HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
    uint8_t debugMode_{0};
    typename HcclTypeSelector<TPL_COMM_MODE>::type hccl_;
    AscendC::HcclHandle handles_[MAX_HANDLE]; // 最大支持64个handleId
    uint64_t preCoreNum_ = 0;
    uint32_t batchWeight_[MAX_HANDLE] = {0};
    uint64_t aivNum_{0};
    uint64_t tileMN_{0};
    uint64_t tailMN_{0};
    uint64_t tileOffset_{0};
};

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM, GM_ADDR cGM, GM_ADDR contextGM,
    GM_ADDR workspaceGM, Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    auto &&cfg = tilingData_->param;
    auto &&tiling = tilingData_->quantBmmV3TileTiling.matmulTiling;

    // 初始化 HCCL
    const void *hcclInitTilingV2 = &(tilingData_->mc2InitTiling);
    uint64_t hcclCcTilingOffset = offsetof(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData, mc2CcTiling);
    hccl_.InitV2(contextGM, hcclInitTilingV2);
    hccl_.SetCcTilingV2(hcclCcTilingOffset);

    // 读取上下文和配置
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
    aivNum_ = tiling.usedCoreNum * GetTaskRation(); // 启用的AIV 数量
    for (uint32_t j = 0; j < cfg.rankDim; j++) {
        batchWeight_[j] = j;
    }

    // all2all 通信相关参数, 划分workspace
    sendBuf_ = workspaceGM;                    // [0, fullMN)
    recvBuf_ = sendBuf_ + cfg.cToFloatLen;     // [fullMN, 2*fullMN)
    workspaceGM_ = recvBuf_ + cfg.cToFloatLen; // [2*fullMN, 3*fullMN)
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::PostProcess()
{
    // 后处理为纯 Vector 操作
    if ASCEND_IS_AIC {
        return;
    }

    // PerBlock 场景：ReduceSum 作为尾处理
    if constexpr (IsPerBlock) {
        auto &&cfg = tilingData_->param;

        // 等待 All2All 通信执行完成
        if (GetBlockIdx() == 0) {
            for (uint32_t i = 0; i < cfg.tileCnt + cfg.tailCnt; i++) {
                hccl_.Wait(handles_[i]);
            }
        }

        // V同步, 确保数据到达
        SyncAll<true>();

        // 准备 ReduceSum 参数
        auto &&tiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
        GM_ADDR curSrcPtr = recvBuf_; // reduceSum 要处理的输入地址, 即 All2All 接收缓冲区
        GM_ADDR curOutPtr = cGM_;     // reduceSum 输出地址, 即 output 地址

        // 主轮每轮数据量与偏移
        const bool isSerial = (tiling.M == cfg.rankM);
        const uint64_t tileElemCount =
            isSerial ? static_cast<uint64_t>(tiling.M / cfg.rankDim) * static_cast<uint64_t>(tiling.N) :
                       static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);

        const uint64_t tileByteSize = tileElemCount * sizeof(CType);
        const uint64_t tileStrideBytes = tileByteSize * cfg.rankDim; // 输入指针步长 (连续，rankDim 份数据)

        // 尾轮每轮数据量与偏移
        const uint64_t tailElemCount = static_cast<uint64_t>(cfg.tailM) * static_cast<uint64_t>(tiling.N);
        const uint64_t tailByteSize = tailElemCount * sizeof(CType);
        const uint64_t tailStrideBytes = tailByteSize * cfg.rankDim;

        // 处理主轮的数据
        for (uint32_t i = 0; i < cfg.tileCnt; i++) {
            tPipe_->Reset();
            reduceSum_.Init(tileElemCount, 0, cfg.rankDim, aivNum_, curSrcPtr, curOutPtr, tPipe_);
            reduceSum_.ExecuteReduceSum(); // AIV执行 reduce_sum

            // SrcInput 指针偏移
            curSrcPtr += tileStrideBytes;
            // outPut 指针偏移
            curOutPtr += tileByteSize;
        }

        // 处理尾轮的数据
        for (uint32_t i = 0; i < cfg.tailCnt; i++) {
            tPipe_->Reset();
            reduceSum_.Init(tailElemCount, 0, cfg.rankDim, aivNum_, curSrcPtr, curOutPtr, tPipe_);
            reduceSum_.ExecuteReduceSum(); // AIV执行 reduce_sum

            // SrcInput 指针偏移
            curSrcPtr += tailStrideBytes;
            // outPut 指针偏移
            curOutPtr += tailByteSize;
        }
    }

    // 等待执行完成后，最后终止hcclserver
    if (GetBlockIdx() == 0) {
        hccl_.Finalize();
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::Process()
{
    InnerProcess(); // 核心计算+通信
    PostProcess();  // 等待计算与通信完成, 终止hcclserver
}

// perblock量化场景当rankM不满足128*ranksize，走计算通信串行
TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::MatMulReduceScatterSerial()
{
    auto &&qBmmTiling = tilingData_->quantBmmV3TileTiling;
    auto &&tiling = qBmmTiling.matmulTiling;
    auto &&cfg = tilingData_->param;

    auto sendBuffer = sendBuf_; // 既作为 MatMul 的输出地址，也作为 AlltoAll 的发送缓冲地址
    auto recvBuffer = recvBuf_; // AlltoAll 的接收缓冲地址

    uint64_t recvCount = static_cast<uint64_t>(tiling.M / cfg.rankDim) * static_cast<uint64_t>(tiling.N);
    uint32_t strideCount = cfg.rankM / cfg.rankDim;
    uint64_t stride = 0;
    uint8_t repeat = 1;

    this->tPipe_->Destroy();
    this->tPipe_->Init();

    // 执行 MatMul 计算
    MMClass op;
    op.Init(aGM_, bGM_, biasGM_, x2ScaleGM_, x1ScaleGM_, sendBuffer, workspaceGM_, &qBmmTiling, tPipe_, batchWeight_,
            strideCount, false);
    op.Process();

    // 同步保证 MatMul 计算完成
    SyncAll<false>();

    // V核发起 All2All 通信
    if ASCEND_IS_AIV {
        handles_[0] = hccl_.template AlltoAll<true>(sendBuffer, recvBuffer, recvCount, dataType_, stride, repeat);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::InnerProcess()
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
    MatMulComputReduceScatter(aGM_, recvBuf_, x1ScaleGM_, tilingData_->quantBmmV3TileTiling, cfg.tileCnt, sendBuf_,
                              (cfg.tailM ? false : true), false);
    // 计算尾块
    if (cfg.tailM) {
        // 非 PerBlock, 非连续场景
        if constexpr (!IsPerBlock) {
            cSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) /
                    cfg.rankDim; // tiling不同,需手动除rankDim
            auto aGMTail = aGM_;
            tileOffset_ = static_cast<uint64_t>(cfg.tileCnt) * cSize * sizeof(CType);
            auto recvGMTail = recvBuf_ + tileOffset_;
            auto sendGMTail = sendBuf_ + tileOffset_;
            MatMulComputReduceScatter(aGMTail, recvGMTail, nullptr, tilingData_->quantBmmV3TailTiling, cfg.tailCnt,
                                      sendGMTail, true, true);
            return;
        }

        // PerBlock 连续场景
        auto aGMTail = aGM_ + aSize * sizeof(AType) * static_cast<uint64_t>(cfg.tileCnt);
        tileOffset_ = static_cast<uint64_t>(cfg.tileCnt) * cSize * sizeof(CType) * cfg.rankDim; // 连续场景, 故乘rankDim
        auto sendGMTail = sendBuf_ + tileOffset_;
        auto recvGMTail = recvBuf_ + tileOffset_;
        auto x1ScaleTail = x1ScaleGM_ + static_cast<uint64_t>(tiling.M) / BLOCK_SIZE *
                                            CeilDiv(static_cast<uint64_t>(tiling.Ka), BLOCK_SIZE) * sizeof(ScaleType) *
                                            static_cast<uint64_t>(cfg.tileCnt);
        MatMulComputReduceScatter(aGMTail, recvGMTail, x1ScaleTail, tilingData_->quantBmmV3TailTiling, cfg.tailCnt,
                                  sendGMTail, true, true);
    }
}

/**
 * @brief 处理尾块场景下的 preCoreNum_ 重计算逻辑
 */
TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::PrepareTailConfig(
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling, const bool isTail)
{
    if (!isTail) {
        return;
    }

    auto &&cfg = tilingData_->param;
    uint32_t currentCoreIdx = GetBlockIdx();
    uint32_t mainCoreNum = tilingData_->quantBmmV3TileTiling.matmulTiling.usedCoreNum;

    // 只有当当前核索引超出主块使用的核数时，才需要重新计算
    if (currentCoreIdx >= mainCoreNum) {
        auto &&tileTiling = tilingData_->quantBmmV3TileTiling.matmulTiling;
        uint64_t headSliceM = (cfg.rankM / cfg.rankDim - cfg.tailM * cfg.tailCnt) / cfg.tileCnt;
        uint64_t mCnt = DequantBmm::CeilDiv(headSliceM, tileTiling.baseM) * cfg.rankDim;
        uint64_t nCnt = DequantBmm::CeilDiv(tileTiling.N, tileTiling.baseN);
        preCoreNum_ = (mCnt * nCnt * cfg.tileCnt) % tileTiling.usedCoreNum;
    }
}

/**
 * @brief [AIC] 执行 MatMul 计算流水线
 */
TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::ExecuteAicMatMulPipeline(
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling, const uint32_t count, const bool isLast,
    const bool isTail)
{
    auto &&cfg = tilingData_->param;

    Mc2MatmulV3::Mc2QuantBatchMatmulASWKernel<AType, BType, ScaleType, float, CType, CubeFormat::ND, CubeFormat::ND,
                                              CubeFormat::ND, ATrans, BTrans>
        mmv3;

    // 初始化Matmul
    mmv3.Init(aGM_, bGM_, biasGM_, x2ScaleGM_, x1ScaleGM_, sendBuf_, workspaceGM_, &qBmmTiling, GetTPipePtr(), cfg,
              isTail, false, preCoreNum_);

    for (uint32_t i = 0; i < count; i++) {
        mmv3.UpdateSlice(i, isTail);                // 更新 slice 偏移
        mmv3.Process(isLast && (i == (count - 1))); // 执行 MatMul

        // AIC 侧做完 Matmul 计算后通知 AIV 进行后处理
        CubeNotifyVector();
    }

    // 更新 preCoreNum_ 供后续可能得逻辑使用
    preCoreNum_ = mmv3.GetPreCoreNum();
}

/**
 * @brief [AIV] 执行 All2All 通信 + ReduceSum 归约流水线 (双发模式)
 */
TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::ExecuteAivCommReducePipeline(
    GM_ADDR recvGM, GM_ADDR sendGM, const uint32_t count, const bool isTail,
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling)
{
    auto &&cfg = tilingData_->param;
    auto &&tiling = qBmmTiling.matmulTiling;

    // 预计算通信常量
    // 单个 Rank 接收的元素数量 (M * N / rankDim)
    const uint64_t rankSliceElems = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
    // 单个 Rank 接收的字节偏移
    const uint64_t rankSliceBytes = rankSliceElems * sizeof(CType);
    // All2All 步长 (元素数): (rankM / rankDim) * rankN
    const uint64_t stride = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    // 通信重复次数, 1次
    const uint8_t repeat = 1;
    // 若是尾块，通信 handleId 存放的起始偏移为 tileCnt
    const uint32_t handleShift = isTail ? cfg.tileCnt : 0;

    // 指针初始化
    GM_ADDR currentSendPtr = sendGM;                            // 当前发送缓冲区起始地址
    GM_ADDR currentRecvPtr = recvGM;                            // 当前接收缓冲区起始地址
    GM_ADDR currentOutPtr = isTail ? cGM_ + tileOffset_ : cGM_; // 当前 reduceSum 输出的起始地址

    // --- Prologue: 启动第 0 轮通信 ---
    VecWaitCube(); // 确保依赖的 MatMul 已完成
    handles_[0 + handleShift] =
        hccl_.template AlltoAll<true>(currentSendPtr, currentRecvPtr, rankSliceElems, dataType_, stride, repeat);

    // 移动指针准备下一轮
    currentSendPtr += rankSliceBytes;
    currentRecvPtr += rankSliceBytes;
    currentOutPtr += rankSliceBytes;

    // --- Loop: 重叠执行 (等待上一轮 + 归约上一轮 + 启动下一轮) ---
    for (uint32_t i = 0; i < count - 1; i++) {
        // 1. 等待上一轮 (i) 通信结束
        if (GetBlockIdx() == 0) {
            hccl_.Wait(handles_[i + handleShift]);
        }
        SyncAll<true>(); // V 同步确保数据到达

        // 2. 启动下一轮 (i+1) 通信
        VecWaitCube();
        handles_[i + 1 + handleShift] =
            hccl_.template AlltoAll<true>(currentSendPtr, currentRecvPtr, rankSliceElems, dataType_, stride, repeat);

        // 3. 执行上一轮 (i) 数据的 ReduceSum
        // 计算地址 = 当前指针 - 一步偏移
        GM_ADDR calculateRecvPtr = currentRecvPtr - rankSliceBytes;
        GM_ADDR calculateOutPtr = currentOutPtr - rankSliceBytes;

        tPipe_->Reset();
        reduceSum_.Init(rankSliceElems, stride, cfg.rankDim, aivNum_, calculateRecvPtr, calculateOutPtr, tPipe_);
        reduceSum_.ExecuteReduceSum();

        // 指针继续前移
        currentSendPtr += rankSliceBytes;
        currentRecvPtr += rankSliceBytes;
        currentOutPtr += rankSliceBytes;
    }

    // --- Epilogue: 处理最后一轮 (count-1) ---
    uint32_t lastIndex = count - 1;

    // 等待最后一轮通信结束
    if (GetBlockIdx() == 0) {
        hccl_.Wait(handles_[lastIndex + handleShift]);
    }
    SyncAll<true>();

    // 执行最后一轮数据的 reduceSum
    // 此时的计算地址同样是 "当前指针 - 偏移量"
    GM_ADDR calculateRecvPtr = currentRecvPtr - rankSliceBytes;
    GM_ADDR calculateOutPtr = currentOutPtr - rankSliceBytes;

    tPipe_->Reset();
    reduceSum_.Init(rankSliceElems, stride, cfg.rankDim, aivNum_, calculateRecvPtr, calculateOutPtr, tPipe_);
    reduceSum_.ExecuteReduceSum();
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatterPertensor(
    GM_ADDR recvGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling, const uint32_t count, GM_ADDR sendGM,
    const bool isLast, const bool isTail)
{
    // [配置阶段] 处理尾块特殊的 preCoreNum_ 逻辑
    PrepareTailConfig(qBmmTiling, isTail);

    // [AIC 阶段] 执行 MatMul 计算流水线
    if ASCEND_IS_AIC {
        ExecuteAicMatMulPipeline(qBmmTiling, count, isLast, isTail);
    }

    // [AIV 阶段] 执行 All2All 通信 + ReduceSum 归约流水线
    if ASCEND_IS_AIV {
        ExecuteAivCommReducePipeline(recvGM, sendGM, count, isTail, qBmmTiling);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatterPerblock(
    GM_ADDR aGM, GM_ADDR recvGM, GM_ADDR x1ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling,
    const uint32_t count, GM_ADDR sendGM, const bool isTail)
{
    // 获取配置与 Tiling 数据
    auto &&cfg = tilingData_->param;
    auto &&tiling = qBmmTiling.matmulTiling;

    // 单次通信的元素数量 (M * N)
    const uint64_t recvCount = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
    // A 矩阵单个 Tile 的字节偏移 (M * Ka * sizeof(AType))
    const uint64_t aTileBytes = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka) * sizeof(AType);
    // Scale 矩阵单个 Tile 的字节偏移
    const uint64_t scaleBlockM = static_cast<uint64_t>(tiling.M / BLOCK_SIZE);
    const uint64_t scaleBlockK = CeilDiv(static_cast<uint64_t>(tiling.Ka), BLOCK_SIZE);
    const uint64_t scaleTileBytes = scaleBlockM * scaleBlockK * sizeof(ScaleType);
    // C 矩阵单个 Tile 的字节偏移 (M * N * sizeof(CType) * rankDim), 连续场景
    const uint64_t cTileBytes = recvCount * sizeof(CType) * cfg.rankDim;

    // 初始化指针
    GM_ADDR currAPtr = aGM;
    GM_ADDR currScalePtr = x1ScaleGM;
    GM_ADDR currSendPtr = sendGM; // MatMul 输出缓冲区, All2All 发送缓冲区
    GM_ADDR currRecvPtr = recvGM; // All2All 接收缓冲区

    // 通信参数配置
    const uint32_t handleShift = isTail ? cfg.tileCnt : 0; // 存放 handleId 的偏移
    const uint64_t stride = 0;
    const uint8_t repeat = 1;                             // 通信重复次数
    const uint32_t strideCount = cfg.rankM / cfg.rankDim; // 传递给 MMClass 的步长参数

    // 按切分轮次循环. 先 MM + All2All
    for (uint32_t i = 0; i < count; i++) {
        tPipe_->Destroy();
        tPipe_->Init();

        // 执行 MatMul 计算
        MMClass op;
        op.Init(currAPtr, bGM_, biasGM_, x2ScaleGM_, currScalePtr, currSendPtr, workspaceGM_, &qBmmTiling, tPipe_,
                batchWeight_, strideCount, false);
        op.Process();

        // 同步保证 MatMul 计算完成
        PipeBarrier<PIPE_V>();
        SyncAll<false>();

        // All2All 通信, AIV 发起
        if ASCEND_IS_AIV {
            handles_[i + handleShift] = hccl_.template AlltoAll<true>(currSendPtr, // 发送缓冲区
                                                                      currRecvPtr, // 接收缓冲区
                                                                      recvCount,   // 元素数量
                                                                      dataType_,   // 数据类型
                                                                      stride,      // 步长
                                                                      repeat       // 重复次数
            );
        }

        // A 矩阵指针偏移
        currAPtr += aTileBytes;
        // Scale 矩阵指针偏移
        currScalePtr += scaleTileBytes;
        // Send 缓冲区指针偏移
        currSendPtr += cTileBytes;
        // Recv 缓冲区指针偏移
        currRecvPtr += cTileBytes;
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::MatMulComputReduceScatter(
    GM_ADDR aGM, GM_ADDR recvGM, GM_ADDR x1ScaleGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &qBmmTiling,
    const uint32_t count, GM_ADDR sendGM, const bool isLast, const bool isTail)
{
    if constexpr (IsPerBlock) {
        MatMulComputReduceScatterPerblock(aGM, recvGM, x1ScaleGM, qBmmTiling, count, sendGM, isTail);
    } else {
        MatMulComputReduceScatterPertensor(recvGM, qBmmTiling, count, sendGM, isLast, isTail);
    }
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::CubeNotifyVector()
{
    // 先全 AIC 同步一次
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_DET_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_DET_FLAG);
    // 通知 AIV
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_DET_FLAG);
}

TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBmmA2AVecReduceFP8HiF8<TEMPLATE_FUNC_PARAMS>::VecWaitCube()
{
    // 等待 AIC 完成
    // 修复同步问题，适配AICPU接口（等待管线由 PIPE_MTE2 改为 PIPE_S）
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_S>(SYNC_AIC_AIV_DET_FLAG);
}

} // namespace MatmulReduceScatterV2Impl

#endif // QUANT_BMM_A2A_VEC_REDUCE_FP8_HIF8_H
