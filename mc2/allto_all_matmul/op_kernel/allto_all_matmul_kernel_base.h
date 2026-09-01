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
 * \file allto_all_matmul_kernel_base.h
 * \brief
 */

#ifndef ALLTO_ALL_MATMUL_KERNEL_BASE_H
#define ALLTO_ALL_MATMUL_KERNEL_BASE_H

namespace Mc2Kernel {
// SEP_BIAS: 是否将 bias 从 matmul 中独立出来由 add_bias 节点完成 (arch35 FP 场景)
template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType,
          bool SEP_BIAS = false>
class AlltoAllMatmulKernelBase {
public:
    __aicore__ inline AlltoAllMatmulKernelBase(SchedulerType *pipeLine)
        : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                GM_ADDR workspaceGM, AlltoAllMatmulTilingDataType *tilingData, AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    SchedulerType *pipeLine_;
    SchedulerContextType pipeLineContext_;
    AlltoAllMatmulTilingDataType *tilingData_;
    AscendC::TPipe *tPipe_;
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR workspaceGM_;
    GM_ADDR commOutGM_;
    GM_ADDR transOutGM_;

private:
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
};

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool SEP_BIAS>
__aicore__ inline void AlltoAllMatmulKernelBase<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType,
                                                SEP_BIAS>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
                                                                GM_ADDR all2all_out, GM_ADDR workspaceGM,
                                                                AlltoAllMatmulTilingDataType *tilingData,
                                                                AscendC::TPipe *tPipe)
{
    // 获取tilingdata数据
    tilingData_ = tilingData;
    auto &&mc2Tiling_ = tilingData_->alltoAllMatmulTilingInfo;
    // 管道初始化
    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    workspaceGM_ = workspaceGM;
    commOutGM_ = workspaceGM;
    transOutGM_ = all2all_out;
    if (all2all_out == nullptr) {
        transOutGM_ = (GM_ADDR)((uint64_t)commOutGM_ + mc2Tiling_.commLen);
    }
    // 初始化流水线
    pipeLine_->Init();
    pipeLine_->GetContext(&pipeLineContext_);

    // 独立加 bias 节点: bias 每轮共享; matmulAddr/M/N/Offset 按 tile 在 ProcessTile/ProcessTail 填充
    if constexpr (SEP_BIAS) {
        pipeLineContext_.addBiasContext->biasAddr = bias_;
    }
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool SEP_BIAS>
__aicore__ inline void
AlltoAllMatmulKernelBase<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType, SEP_BIAS>::Process()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllMatmulTilingInfo;
    // 启动主块流水
    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 启动尾块流水
    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    // 结束流水线
    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool SEP_BIAS>
__aicore__ inline void AlltoAllMatmulKernelBase<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType,
                                                SEP_BIAS>::ProcessTile(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllMatmulTilingInfo;
    // 复用的变量
    uint64_t tileMMultiRankK = (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankK;

    // matmul计算相关地址和偏移
    pipeLineContext_.computationContext->baseData.aGM = transOutGM_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = y_;
    // SEP_BIAS 场景: bias 由独立 add_bias 节点处理, matmul 不再接收 bias
    pipeLineContext_.computationContext->baseData.biasGM = SEP_BIAS ? nullptr : bias_;
    // add_bias: 每轮处理一个主块 tile [tileM, rankN], 偏移随 task 索引递增 matmulOffset
    if constexpr (SEP_BIAS) {
        pipeLineContext_.addBiasContext->matmulAddr = y_;
        pipeLineContext_.addBiasContext->matmulOffset = (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
        pipeLineContext_.addBiasContext->biasOffset = 0;
        pipeLineContext_.addBiasContext->M = mc2Tiling_.tileM;
        pipeLineContext_.addBiasContext->N = mc2Tiling_.rankN;
    }
    pipeLineContext_.computationContext->baseData.aOffset =
        tileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bOffset = (uint64_t)0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        (uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2MmV3TileTilingData);

    // 转置相关地址和偏移
    pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_;
    pipeLineContext_.transposeContext->transposeDstAddr = transOutGM_;
    pipeLineContext_.transposeContext->transposeSrcOffset = tileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstOffset = tileMMultiRankK * mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->nextSrcBlockOffset =
        (uint64_t)mc2Tiling_.rankM * mc2Tiling_.rankK / mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->nextDstBlockOffset = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transposeContext->rankCnt = (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->innerAxis = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tileM;
    pipeLineContext_.transposeContext->innerOffsetIn = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transposeContext->innerOffsetOut = (uint64_t)mc2Tiling_.rankK * mc2Tiling_.rankDim;

    // 通信相关地址和偏移
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_;
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_;
    pipeLineContext_.communicationContext->sendOffset = tileMMultiRankK * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tileMMultiRankK;
    pipeLineContext_.communicationContext->strideCount = pipeLineContext_.transposeContext->nextSrcBlockOffset;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    pipeLine_->Process(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool SEP_BIAS>
__aicore__ inline void AlltoAllMatmulKernelBase<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType,
                                                SEP_BIAS>::ProcessTail(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllMatmulTilingInfo;
    // 复用的变量
    uint64_t tailMMultiRankK = (uint64_t)mc2Tiling_.tailM * (uint64_t)mc2Tiling_.rankK;
    uint64_t tileCntMultitileMMultiRankK =
        (uint64_t)mc2Tiling_.tileCnt * (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankK;

    pipeLineContext_.computationContext->baseData.aGM =
        transOutGM_ + tileCntMultitileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM =
        y_ + (uint64_t)mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->baseData.aOffset =
        tailMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bOffset = (uint64_t)0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2MmV3TailTilingData);

    // add_bias: 每轮处理一个尾块 tile [tailM, rankN], 起点在主块之后, 偏移随 task 索引递增 matmulOffset
    if constexpr (SEP_BIAS) {
        pipeLineContext_.addBiasContext->matmulAddr =
            y_ + (uint64_t)mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
        pipeLineContext_.addBiasContext->matmulOffset = (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
        pipeLineContext_.addBiasContext->biasOffset = 0;
        pipeLineContext_.addBiasContext->M = mc2Tiling_.tailM;
        pipeLineContext_.addBiasContext->N = mc2Tiling_.rankN;
    }

    pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstAddr =
        transOutGM_ + tileCntMultitileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeSrcOffset = tailMMultiRankK * (uint64_t)sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstOffset =
        tailMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tailM;

    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->sendOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tailMMultiRankK;

    pipeLine_->Process(taskCnt);
}
}; // namespace Mc2Kernel
#endif
