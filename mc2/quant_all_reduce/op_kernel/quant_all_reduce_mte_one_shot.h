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
 * \file quant_all_reduce_mte_one_shot.h
 * \brief quant_all_reduce mte_one_shot方式通信，通过1步allgather的kernel代码逻辑
 */

#ifndef QUANT_ALL_REDUCE_MTE_ONE_SHOT_H
#define QUANT_ALL_REDUCE_MTE_ONE_SHOT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "quant_all_reduce_tiling_data.h"
#include "../../quant_reduce_scatter/op_kernel/quant_reduce_scatter_utils.h"
#include "../../quant_reduce_scatter/op_kernel/mte_comm.h"
#include "../../quant_reduce_scatter/op_kernel/vec_comp.h"

namespace QuantAllReduceImpl {

using namespace QuantMTECommImpl;
using namespace VectorComputeImpl;
using namespace AscendC;

// 常量定义
constexpr static uint32_t MX_SIZE = 32U;         // MX量化时，32个x数据共有一个scale
constexpr static uint32_t PER_GROUP_SIZE = 128U; // KG量化时，128个x数据共有一个scale

#define TemplateTypeClass typename XType, typename ScalesType, typename OutputType
#define TemplateType XType, ScalesType, OutputType

template <TemplateTypeClass>
class QuantAllReduceMteOneShot {
public:
    __aicore__ inline QuantAllReduceMteOneShot(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output, TPipe *pipe,
                                const QuantAllReduceTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearSumTensor();
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t currXOffset, uint64_t currScaleOffset, uint32_t xNum,
                                                  uint32_t scaleNum);
    __aicore__ inline void ExecuteAllReduce();
    __aicore__ inline void ParseTilingInfo(const QuantAllReduceTilingData *tilingData);
    __aicore__ inline void ComputeXPerBlock(const QuantAllReduceTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void ComputeBlockDistribution(uint64_t aivNum);
    __aicore__ inline void InitSubModules(GM_ADDR x, GM_ADDR scales, GM_ADDR output, TPipe *tPipe, uint64_t aivNum);

    // 数据参数
    uint64_t xSize_{0};
    uint64_t xNums_{0};
    uint64_t totalWinSize_{0};
    uint32_t xPerBlock_{0};         // 动态计算的每块数据元素数（最大化UB利用）
    uint32_t scaleNumsPerBlock_{0}; // 每块对应的scale数量

    // 动态分块参数（基于xPerBlock_）
    uint32_t totalBlockNums_{0};    // 总块数（基于xPerBlock_）
    uint32_t assignedBlockNums_{0}; // 当前核分配的块数
    uint32_t round_{0};             // 基础轮次数
    uint32_t tailBlockNums_{0};     // 尾块数（分配给前几个核）
    uint32_t lastAivId_{0};         // 最后一个核的ID
    uint64_t xOffset_{0};           // 当前核的起始x偏移
    uint64_t scaleOffset_{0};       // 当前核的起始scale偏移
    uint32_t tailXNums_{0};         // 尾块的x数量

    MTECommunication<TemplateType> mteComm_; // MTE 通信相关实现
    VectorCompute<TemplateType> vecComp_;    // vector 计算相关实现

    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    LocalTensor<float> sumTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue_; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_;                                     // 用于Reduce_sum 求和
};

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                                                    TPipe *tPipe,
                                                                    const QuantAllReduceTilingData *tilingData)
{
    mteComm_.InitHcclContext();
    ParseTilingInfo(tilingData);
    tPipe->Reset();
    ComputeXPerBlock(tilingData, tPipe);
    uint64_t aivNum = tilingData->quantAllReduceTilingInfo.aivNum;
    ComputeBlockDistribution(aivNum);
    InitSubModules(x, scales, output, tPipe, aivNum);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ParseTilingInfo(
    const QuantAllReduceTilingData *tilingData)
{
    auto &&info = tilingData->quantAllReduceTilingInfo;
    totalWinSize_ = info.totalWinSize;
    xNums_ = info.bs * info.hiddenSize;
    xSize_ = xNums_ * sizeof(XType);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ComputeXPerBlock(
    const QuantAllReduceTilingData *tilingData, TPipe *tPipe)
{
    auto &&info = tilingData->quantAllReduceTilingInfo;

    // 固定开销：不随 xPerBlock_ 变化的 buffer
    uint64_t commFixedSpace = BUFFER_NUM * X_BLOCK_BYTES +                      // scaleQueue_
                              UB_ALIGN_BYTES +                                  // winFlagsBuf_
                              UB_ALIGN_BYTES +                                  // writeStateBuf_
                              mteComm_.hcclContext_->rankDim * UB_ALIGN_BYTES + // readStateBuf_
                              mteComm_.hcclContext_->rankDim * UB_ALIGN_BYTES;  // stateResetBuf_

    // 动态开销：每增加 1 个 x 需要的 UB 字节（整数部分，分数部分见下方比例校正）
    uint64_t dynamicBaseSize = BUFFER_NUM * sizeof(OutputType) + // xOutQueue_
                               BUFFER_NUM * sizeof(XType) +      // xQueue_
                               sizeof(float) +                   // brcbBuf_
                               sizeof(float) +                   // xCastBuf_
                               sizeof(bfloat16_t) +              // xCastBf16Buf_
                               sizeof(float16_t) +               // xCastfp16Buf_
                               sizeof(float) +                   // sumBuf_
                               BUFFER_NUM * sizeof(XType);       // xInQueue_

    uint64_t remainingSpace = TOTAL_UB_SIZE - commFixedSpace;
    uint64_t maxXPerBlock;
    // scale buffer 的开销以 FracDiv 精确计入 UB 预算（避免 < 1 B per X 被整数截断）
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        maxXPerBlock = FracDiv(remainingSpace, dynamicBaseSize, 4U, 1U); // + 1/4 B per X
    } else {
        maxXPerBlock = FracDiv(remainingSpace, dynamicBaseSize, 32U, 3U); // + 3/32 B per X
    }

    // host 推荐值（TARGET_ITER 公式）作为上限，kernel UB 约束作为 safety net
    uint32_t hostRecommended = info.xPerBlock;
    if (hostRecommended != 0U && maxXPerBlock > hostRecommended) {
        maxXPerBlock = hostRecommended;
    }

    // 数据量上限约束
    xPerBlock_ = static_cast<uint32_t>(maxXPerBlock < xNums_ ? maxXPerBlock : xNums_);

    // 对齐到 alignBlock
    uint32_t alignBlock = info.alignBlock;
    xPerBlock_ = FloorAlign(xPerBlock_, alignBlock);
    if (xPerBlock_ == 0) {
        xPerBlock_ = alignBlock;
    }

    // 每块 scale 数量（real 值，GM 偏移用；UB 分配按 32B 对齐）
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleNumsPerBlock_ = xPerBlock_ / MX_SIZE;
    } else {
        scaleNumsPerBlock_ = xPerBlock_ / PER_GROUP_SIZE;
    }

    uint32_t scaleAlignNum = UB_ALIGN_BYTES / sizeof(ScalesType);
    uint32_t scaleAllocCount = CeilAlignU32(scaleNumsPerBlock_, scaleAlignNum);
    uint64_t scaleInQueSize = scaleAllocCount * sizeof(ScalesType);

    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xPerBlock_ * sizeof(XType));
    tPipe->InitBuffer(scaleInQue_, BUFFER_NUM, scaleInQueSize);
    tPipe->InitBuffer(sumBuf_, xPerBlock_ * sizeof(float));
    sumTensor_ = sumBuf_.Get<float>();
}

// 计算多核分块参数：每核分配块数、尾块归属、起始偏移
template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ComputeBlockDistribution(uint64_t aivNum)
{
    uint64_t aivId = GetBlockIdx();
    totalBlockNums_ = CeilDiv(xNums_, xPerBlock_);
    round_ = totalBlockNums_ / aivNum;
    tailBlockNums_ = totalBlockNums_ % aivNum;
    assignedBlockNums_ = aivId < tailBlockNums_ ? round_ + 1 : round_;
    lastAivId_ = (round_ == 0) ? tailBlockNums_ - 1 : aivNum - 1;

    uint64_t blockIdx = aivId * round_ + (aivId < tailBlockNums_ ? aivId : tailBlockNums_);
    xOffset_ = blockIdx * xPerBlock_;
    scaleOffset_ = blockIdx * scaleNumsPerBlock_;
    tailXNums_ = BlockAlignMod(xNums_, xPerBlock_);
}

// 初始化 vecComp、mteComm 子模块参数并分配其 UB buffer，绑定 GM Tensor
template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::InitSubModules(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                                                              TPipe *tPipe, uint64_t aivNum)
{
    vecComp_.SetBlockSize(xPerBlock_);
    vecComp_.SetScaleNums(scaleNumsPerBlock_);
    mteComm_.round_ = round_;
    mteComm_.tailBlockNums_ = tailBlockNums_;
    mteComm_.ComputeTailAivId(aivNum);
    mteComm_.SetBlockSize(xPerBlock_, aivNum, tailXNums_, scaleNumsPerBlock_);
    mteComm_.InitParams();
    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);
    mteComm_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ReadDataBlockReduceSum(uint64_t currXOffset,
                                                                                      uint64_t currScaleOffset,
                                                                                      uint32_t xNum, uint32_t scaleNum)
{
    /* 读取 x 从 Win -> UB */
    LocalTensor<XType> xTempTensor = xInQueue_.AllocTensor<XType>();
    DataCopy(xTempTensor, remoteWinXTensor_[currXOffset], xNum);
    xInQueue_.EnQue(xTempTensor);
    xTempTensor = xInQueue_.DeQue<XType>();

    /* 读取 scale 从 Win -> UB（DataCopyPad 处理 sub-32B 对齐） */
    LocalTensor<ScalesType> scaleTempTensor = scaleInQue_.AllocTensor<ScalesType>();
    DataCopyParams scaleCopyParams;
    scaleCopyParams.blockLen = scaleNum * sizeof(ScalesType);
    scaleCopyParams.blockCount = 1;
    DataCopyPadParams scalePadParams;
    DataCopyPad(scaleTempTensor, remoteWinScaleTensor_[currScaleOffset], scaleCopyParams, scalePadParams);
    scaleInQue_.EnQue(scaleTempTensor);
    scaleTempTensor = scaleInQue_.DeQue<ScalesType>();

    /* 反量化计算与ReduceSum求和（累加到sumTensor_） */
    vecComp_.DequantReduceSum(xTempTensor, scaleTempTensor, sumTensor_);
    xInQueue_.FreeTensor(xTempTensor);
    scaleInQue_.FreeTensor(scaleTempTensor);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ClearSumTensor()
{
    Duplicate<float>(sumTensor_, (float)0.0, xPerBlock_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::ExecuteAllReduce()
{
    // 读状态位，软同步
    mteComm_.ReadStatus();

    uint64_t aivId = GetBlockIdx();

    // 遍历需要处理的数据块（基于动态xPerBlock_分块）
    for (uint64_t iterBlock = 0; iterBlock < assignedBlockNums_; ++iterBlock) {
        uint64_t currXOffset = xOffset_ + iterBlock * xPerBlock_;
        uint64_t currScaleOffset = scaleOffset_ + iterBlock * scaleNumsPerBlock_;

        // 计算当前块的x数量（尾块处理）
        uint32_t currXNum = xPerBlock_;
        uint32_t currScaleNum = scaleNumsPerBlock_;
        if ((aivId == lastAivId_) && (iterBlock == assignedBlockNums_ - 1)) {
            currXNum = tailXNums_;
            // 尾块的scale数量：real 值（DataCopyPad 处理 sub-32B 拷贝，不再需要硬对齐）
            if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
                currScaleNum = CeilDivU32(currXNum, MX_SIZE);
            } else {
                currScaleNum = CeilDivU32(currXNum, PER_GROUP_SIZE);
            }
        }

        ClearSumTensor();

        // 遍历每张卡，读取其Win区的数据
        uint32_t startRankId = mteComm_.hcclContext_->rankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
            uint32_t remoteRankId = (startRankId + i) % mteComm_.hcclContext_->rankDim;

            // 获取对端Win区中数据区相关的地址
            GM_ADDR remoteDataSpaceGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);
            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType *)remoteDataSpaceGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType *)(remoteDataSpaceGm + xSize_));

            // 读取对端对应地址的 x 和 scale数据，进行反量化和求和
            ReadDataBlockReduceSum(currXOffset, currScaleOffset, currXNum, currScaleNum);
        }

        // 将计算好的数据拷贝到输出tensor
        mteComm_.CopyResultToOutput(currXOffset, sumTensor_, currXNum);
    }
}

template <TemplateTypeClass>
__aicore__ inline void QuantAllReduceMteOneShot<TemplateType>::Process()
{
    // 纯AIV过程
    if ASCEND_IS_AIC {
        return;
    }

    // 一次性拷贝完所有数据到本地卡win区
    mteComm_.CopyDataToWin();
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
    // 执行AllReduce过程：等待状态区同步，读取数据并进行反量化ReduceSum
    ExecuteAllReduce();
}

} // namespace QuantAllReduceImpl
#endif // QUANT_ALL_REDUCE_MTE_ONE_SHOT_H
