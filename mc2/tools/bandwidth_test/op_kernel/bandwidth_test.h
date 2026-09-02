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
 * \file bandwidth_test.h
 * \brief
 */

#ifndef BANDWIDTH_TEST_H
#define BANDWIDTH_TEST_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "bandwidth_test_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"

#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"

namespace BandwidthTestImpl {

using namespace AscendC;
using namespace Mc2Kernel;
using namespace Mc2Aclnn;

#define TemplateBandwidthTestTypeClass typename XType, typename YType
#define TemplateBandwidthTestTypeFunc XType, YType

template <TemplateBandwidthTestTypeClass>
class BandwidthTest {
public:
    __aicore__ inline BandwidthTest(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR dstRankId, GM_ADDR y, GM_ADDR receiveCnt,
                                GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BufferInit();
    __aicore__ inline void ProcessToken(GlobalTensor<XType> &outTokenGT, uint32_t tokenIndex);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum);

    // 获取指定 rank 的通信窗口数据区地址
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffset_;
    }

    // 获取指定 rank 的通信窗口状态区地址
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)context_->epHcclBuffer_[rankId] + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;                // 输入数据 x
    GlobalTensor<int32_t> dstRankIdGMTensor_;      // 专家ID (用于确定目标rank)
    GlobalTensor<YType> yGMTensor_;                // 输出数据 y
    GlobalTensor<int32_t> receiveCntGMTensor_;     // 接收到的token计数
    GlobalTensor<float> windowInStatusFp32Tensor_; // 窗口状态区(用于同步)
    LocalTensor<XType> xTmpTensor_;                // 临时存储token数据
    LocalTensor<YType> yTmpTensor_;                // 临时存储输出数据
    LocalTensor<int32_t> dstRankIdsTensor_;        // 本地缓存的dst_rank_id数组
    LocalTensor<int32_t> statusTensor_;            // 状态数据
    LocalTensor<float> statusFp32Tensor_;          // 状态数据(FP32视图)
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;
    TBuf<> dstRankIdsBuf_;    // dst_rank_id 缓冲区
    TBuf<> statusBuf_;        // 状态缓冲区
    TBuf<> waitStatusBuf_;    // 等待状态缓冲区
    TBuf<> gatherMaskOutBuf_; // 中间计算缓冲区
    TBuf<> scalarBuf_;        // 标量缓冲区
    TBuf<> tokenNumBuf_;      // token数量缓冲区
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列

    // ==================== 全局内存地址 ====================
    GM_ADDR yGM_;           // 输出y的GM地址
    GM_ADDR receiveCntGM_;  // 接收计数的GM地址
    GM_ADDR statusSpaceGm_; // 状态空间的GM地址
    GM_ADDR windowGM_;      // 本地窗口的GM地址

    // ==================== HCCL通信上下文 ====================
    __gm__ Mc2Kernel::HcclOpParam *winContext_{nullptr}; // HCCL窗口上下文
    __gm__ Mc2MoeContext *context_{nullptr};

    uint32_t axisBS_{0};              // batch size (token数量)
    uint32_t axisH_{0};               // hidden size (特征维度)
    uint32_t aivNum_{0};              // AIV核总数
    uint32_t epWorldSize_{0};         // rank总数
    int32_t epRankId_{0};             // 当前rank ID
    int32_t mode_{0};                 // 模式: 0=完整流程, 1=仅发送
    uint32_t aivId_{0};               // 当前AIV核ID
    uint32_t globalBS_{0};            // 全局最大batch size
    uint32_t hOutSize_{0};            // 输出hidden size (字节)
    uint32_t hAlignWinSize_{0};       // 窗口对齐后的hidden size
    uint32_t hAlignWinCnt_{0};        // 窗口对齐后的元素个数
    uint32_t hOutAlignUbSize_{0};     // UB对齐后的hidden size
    uint32_t hAlignSize_{0};          // UB对齐后的hidden size
    uint32_t startRankId_{0};         // 当前核处理的起始expert ID
    uint32_t endRankId_{0};           // 当前核处理的结束expert ID
    uint32_t sendExpertNum_{0};       // 当前核处理的expert数量
    uint32_t totalCnt_{0};            // 总计处理的token数
    uint32_t dataState_{0};           // 数据状态 (用于双缓冲)
    uint32_t totalUsedUB_{0};         // 已使用的UB大小
    uint64_t winDataSizeOffset_{0};   // 窗口数据区偏移
    uint64_t expertPerSizeOnWin_{0};  // 每个expert在窗口中占用的空间
    uint32_t stateOffset_{0};         // 状态偏移 (32字节)
    uint32_t recStatusNumPerCore_{0}; // 每个核负责等待的rank数量
    uint32_t rscvStatusNum_{0};       // 需要等待的rank总数
    uint32_t remainderRankNum_{0};    // 分配后剩余的rank数量
    uint32_t startStatusIndex_{0};    // 起始状态索引
    uint32_t bufferNum_{0};           // 缓冲区数量
    uint32_t totalUbSize_{0};         // 总UB大小
    uint64_t totalWinSizeEp_{0};      // 总窗口大小
    float sumTarget_;                 // 状态同步目标值 (1.0)
};

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR dstRankId,
                                                                          GM_ADDR y, GM_ADDR receiveCnt,
                                                                          GM_ADDR workspaceGM, TPipe *pipe,
                                                                          const BandwidthTestTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx(); // 获取当前核ID

    // ============ HCCL窗口初始化 ============
    totalWinSizeEp_ = tilingData->totalWinSize;
    context_ = (__gm__ Mc2MoeContext *)context;

    // ============ 从tiling获取配置参数 ============
    totalUbSize_ = static_cast<uint32_t>(tilingData->totalUbSize);
    axisBS_ = tilingData->maxBs;   // token数量
    axisH_ = tilingData->dataSize; // hidden size
    epWorldSize_ = tilingData->worldSize;
    globalBS_ = tilingData->maxBs * epWorldSize_;
    aivNum_ = tilingData->aivNum;
    mode_ = tilingData->mode;

    // ============ 初始化窗口状态 ============
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    uint32_t epRankIdHccl{0};
    uint32_t epWorldSizeHccl{0};
    epRankIdHccl = context_->epRankId;
    statusSpaceGm_ = (GM_ADDR)(context_->epHcclBuffer_[epRankIdHccl]);
    epWorldSizeHccl = epWorldSize_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor;
    epRankId_ = epRankIdHccl;
    selfDataStatusGMTensor.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    dataState_ = MoeDistributeV2Base::InitWinState(selfDataStatusGMTensor, epRankIdHccl, epWorldSizeHccl, epRankId_,
                                                   epWorldSize_, epWorldSize_, globalBS_, dataStateBuf);

    // ============ 设置全局张量 ============
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    dstRankIdGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstRankId);
    yGMTensor_.SetGlobalBuffer((__gm__ YType *)y);
    receiveCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)receiveCnt);

    yGM_ = y;
    receiveCntGM_ = receiveCnt;

    // ============ 计算对齐参数 ============
    hOutSize_ = axisH_ * sizeof(XType);                                  // 原始hidden size
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;     // UB 32字节对齐
    hAlignWinSize_ = Ceil(hAlignSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // 窗口512字节对齐
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(XType);
    hOutAlignUbSize_ = hAlignSize_;

    // 每个rank在窗口中的空间大小 = global_bs/world_size * 对齐后的hidden size
    uint32_t axisMaxBS = globalBS_ / epWorldSize_;
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS) * hAlignWinSize_;

    // ============ 计算状态等待的任务分发 ============
    // 将 world_size 个 rank 的状态等待任务分发到各 AIV 核
    stateOffset_ = 32UL; // 32字节
    rscvStatusNum_ = epWorldSize_;
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_;
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_;
    if (aivId_ < remainderRankNum_) {
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }

    // ============ 初始化状态缓冲区 ============
    uint32_t statusBufCntAlign = Ceil(Ceil(epWorldSize_, aivNum_), 8) * 8;
    uint32_t statusBufSize = statusBufCntAlign * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
    statusTensor_ = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign * 8);
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankId_);

    // 初始化状态为 1.0f (0x3F800000)，用于状态同步
    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = {0x101010101010101, 0};
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign / 8, 1, 8);

    // ============ 计算窗口数据区偏移 ============
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffset_ = dataState_ * (totalWinSizeEp_ / 2) + axisMaxBS * hSizeAlignCombine;
    windowGM_ = GetWindAddrByRankId(epRankId_);
    windowInStatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));

    // ============ 初始化其他缓冲区 ============
    uint32_t dstRankIdsSize = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t dstRankIdsBufSize = dstRankIdsSize > bsAlign256 ? dstRankIdsSize : bsAlign256;
    tpipe_->InitBuffer(dstRankIdsBuf_, dstRankIdsBufSize);
    totalUsedUB_ += dstRankIdsBufSize;
    dstRankIdsTensor_ = dstRankIdsBuf_.Get<int32_t>();

    uint32_t hFp32Size = axisH_ * sizeof(float);
    uint32_t maxSize = hFp32Size > dstRankIdsSize ? hFp32Size : dstRankIdsSize;
    tpipe_->InitBuffer(gatherMaskOutBuf_, maxSize);
    totalUsedUB_ += maxSize;

    workLocalTensor_ = gatherMaskOutBuf_.Get<float>();

    tpipe_->InitBuffer(dstExpBuf_, maxSize);
    totalUsedUB_ += maxSize;
    tpipe_->InitBuffer(subExpBuf_, maxSize);
    totalUsedUB_ += maxSize;

    // ============ 初始化数据队列 ============
    uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
    bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
    tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_);

    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId = aivId_;
    startTokenId = sendTokenNum * newAivId; // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {     // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::ProcessToken(GlobalTensor<XType> &outTokenGT,
                                                                                  uint32_t tokenIndex)
{
    // 处理单个token：
    // 1. 从输入GM读取token数据到UB
    // 2. 将token数据写入目标rank的窗口

    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams hCommuCopyOutParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};

    // 从GM读取token到UB
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams, padParams);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 写入目标rank的窗口
    DataCopyPad(outTokenGT, xTmpTensor_, hCommuCopyOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    xQueue_.FreeTensor<XType>(xTmpTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::CalTokenSendExpertCnt(uint32_t dstExpertId,
                                                                                           int32_t calCnt,
                                                                                           int32_t &curExpertCnt)
{
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor_, dstRankIdsTensor_, dstExpIdTensor_, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor_, dstExpIdTensor_, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor_(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SendToMoeExpert()
{
    // 发送token到目标rank的窗口
    // 根据 dst_rank_id 确定目标 rank，将 token 写入该 rank 的通信窗口

    // ============ 计算当前核处理的token范围 ============
    uint32_t startTokenId, endTokenId, sendTokenNum; // 每个aiv发送时的起始rankid
    SplitToCore(axisBS_, aivNum_, startTokenId, endTokenId, sendTokenNum);

    if (startTokenId >= axisBS_) {
        return;
    }

    GlobalTensor<XType> dstWinGMTensor;

    // ============ 遍历token并发送到目标rank ============
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t toDstRankId = dstRankIdsTensor_(tokenIndex);
        int32_t curExpertCnt = 0;

        if (tokenIndex > 0) {
            SyncFunc<AscendC::HardEvent::S_V>();
            CalTokenSendExpertCnt(toDstRankId, tokenIndex, curExpertCnt);
        }

        SyncFunc<AscendC::HardEvent::S_V>();

        // 计算目标窗口地址:
        // 目标rank窗口基地址 + 当前rank的偏移(每个rank发来的数据独立存储) + 当前token偏移
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(toDstRankId) + epRankId_ * expertPerSizeOnWin_ +
                                            curExpertCnt * hAlignWinSize_);
        dstWinGMTensor.SetGlobalBuffer((__gm__ XType *)rankGM);
        GlobalTensor<XType> tempTensor = dstWinGMTensor[0];
        ProcessToken(tempTensor, tokenIndex);

        SyncFunc<AscendC::HardEvent::V_S>();
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SetStatus()
{
    // 设置状态：统计发送到各rank的token数量，并写入状态区通知对端
    // 状态区结构: [flag(1.0f), count, rank_id, world_size, ...]
    // ============ 计算当前核负责的rank范围 ============
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, sendExpertNum_);

    // ============ 统计发送到各rank的token数量 ============
    for (uint32_t curRankId = startRankId_; curRankId < endRankId_; ++curRankId) {
        int32_t curRankCnt = 0;
        int32_t cntPosIndex = (curRankId - startRankId_) * 8 + 1;
        SyncFunc<AscendC::HardEvent::S_V>();

        CalTokenSendExpertCnt(curRankId, axisBS_, curRankCnt);
        statusTensor_(cntPosIndex) = curRankCnt;
    }

    PipeBarrier<PIPE_ALL>();
    SyncAll<true>(); // 等待所有核完成统计
    if (startRankId_ >= epWorldSize_) {
        return;
    }

    // ============ 将状态写入目标rank的状态区 ============
    // 写入位置: 目标rank状态区 + 当前rank的偏移
    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t rankIndex = startRankId_; rankIndex < endRankId_; ++rankIndex) {
        uint32_t toDstRankId = rankIndex;
        uint32_t offset = stateOffset_ * epRankId_; // 当前rank在目标状态区的偏移
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(toDstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(rankIndex - startRankId_) * 8], 8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::BufferInit()
{
    // 重新初始化缓冲区，用于WaitDispatch阶段
    tpipe_->Reset();
    totalUsedUB_ = 0U;
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256) ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    totalUsedUB_ += waitStatusBufSize;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * sizeof(float);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t gatherMaskOutSize =
        (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);
    totalUsedUB_ += gatherMaskOutSize;

    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2);
    totalUsedUB_ += UB_ALIGN * 2;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::WaitDispatch()
{
    // 等待所有rank完成数据发送
    // 通过轮询状态区，等待所有rank的状态flag变为1.0f

    BufferInit();
    startRankId_ = startStatusIndex_;
    endRankId_ = startRankId_ + recStatusNumPerCore_;
    sendExpertNum_ = recStatusNumPerCore_;

    // 如果当前核没有需要等待的rank，直接返回
    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        SyncAll<true>();
        return;
    }

    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    float compareTarget = sumTarget_ * recStatusNumPerCore_; // 目标值 = 1.0 * 等待的rank数量
    float sumOfFlag = static_cast<float>(-1.0);

    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};

    // ============ 轮询等待状态 ============
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        // 读取状态区的flag值
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(statusFp32Tensor_, windowInStatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        // 计算所有flag的和
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutBuf_.Get<float>(), 1, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    // ============ 清零状态区，为下一次通信做准备 ============
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInStatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    SyncAll<true>(); // 等待所有核完成
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::LocalWindowCopy()
{
    // 从本地窗口读取接收到的数据，拷贝到输出tensor
    // 输出格式: y[rank_id * max_bs * hidden : (rank_id+1) * max_bs * hidden] = 来自该rank的数据

    if (startRankId_ >= rscvStatusNum_) {
        return;
    }

    // ============ 读取状态数据 ============
    TBuf<> localStatusBuf;
    // 状态区包含 world_size 个状态块，每个块 8 个 int32 (32B)
    const uint32_t statusCopySize = Ceil(epWorldSize_ * 8 * sizeof(int32_t), UB_ALIGN);
    tpipe_->InitBuffer(localStatusBuf, statusCopySize);
    // 获取状态张量（int32 视图）
    LocalTensor<int32_t> localStatusTensor = localStatusBuf.Get<int32_t>();

    // 从窗口状态区读取所有状态
    GlobalTensor<int32_t> srcStatus = windowInStatusFp32Tensor_.ReinterpretCast<int32_t>();
    DataCopyParams statusCopyParams{static_cast<uint16_t>(epWorldSize_), 8, 0, 0};
    DataCopy(localStatusTensor, srcStatus, statusCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 计算前缀和：当前核之前所有核收到的token总数
    uint32_t preCnt = 0;
    for (uint32_t i = 0; i < startRankId_; i++) {
        uint32_t value = localStatusTensor(i * 8 + 1);
        preCnt += value; // token数量存储在位置1
    }

    for (uint32_t idx = 0; idx < epWorldSize_; idx++) {
        // 只取 token 计数（索引 1）
        uint32_t receiveCount = localStatusTensor(idx * 8 + 1);
        receiveCntGMTensor_.SetValue(idx, receiveCount);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    uint32_t hOutElemCount = hOutSize_ / sizeof(XType);
    DataCopyPadParams padParams = {false, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};

    // ============ 初始化数据拷贝队列 ============
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tripleQueue;
    uint32_t leftUBSize = ((totalUbSize_ - totalUsedUB_) / UB_ALIGN) * UB_ALIGN;
    uint32_t dataSize = leftUBSize / BUFFER_NUM;
    tpipe_->InitBuffer(tripleQueue, BUFFER_NUM, dataSize);

    uint32_t countPerDataSize = dataSize / hOutAlignUbSize_; // 每次拷贝的token数量
    if (countPerDataSize == 0) {
        countPerDataSize = 1;
    }
    uint32_t beginIdx = preCnt;

    // ============ 遍历每个rank，从窗口拷贝数据到输出 ============
    for (uint32_t index = startRankId_; index < endRankId_; index++) {
        uint32_t i = index - startRankId_;
        uint32_t count = localStatusTensor(index * 8 + 1); // 从该rank收到的token数量

        // 计算窗口中该rank数据的起始地址
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + index * expertPerSizeOnWin_;
        GlobalTensor<YType> tokGlobal, yOutGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        // 分批拷贝数据
        uint32_t loopTimes = Ceil(count, countPerDataSize);
        uint32_t lastTimeCount = count - countPerDataSize * (loopTimes - 1);
        for (uint32_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
            uint32_t tokenCopiedNumThisExpert = countPerDataSize * loopIdx;
            tokGlobal.SetGlobalBuffer((__gm__ YType *)(wAddr + tokenCopiedNumThisExpert * hAlignWinSize_));

            uint16_t blockCount = static_cast<uint16_t>(((loopIdx < loopTimes - 1) ? countPerDataSize : lastTimeCount));
            uint16_t copyBlockLen = static_cast<uint16_t>(hOutAlignUbSize_ / UB_ALIGN);
            uint16_t srcStrideTokGlobal = static_cast<uint16_t>((hAlignWinSize_ - hOutAlignUbSize_) / UB_ALIGN);
            DataCopyParams dataCopyParamsTokGlobal{blockCount, copyBlockLen, srcStrideTokGlobal, 0U};

            // 从窗口读取到UB
            LocalTensor<YType> tripleLocal = tripleQueue.AllocTensor<YType>();
            DataCopy(tripleLocal, tokGlobal, dataCopyParamsTokGlobal);
            tripleQueue.EnQue(tripleLocal);
            tripleLocal = tripleQueue.DeQue<YType>();

            // 从UB写入输出tensor
            yOutGlobal.SetGlobalBuffer((__gm__ YType *)(yGM_ + (beginIdx + tokenCopiedNumThisExpert) * hOutSize_));
            uint32_t srcStrideY = (hOutAlignUbSize_ - hOutSize_) / UB_ALIGN;
            DataCopyExtParams dataCopyParamsY{blockCount, hOutSize_, srcStrideY, 0U, 0U};
            DataCopyPad(yOutGlobal, tripleLocal, dataCopyParamsY);
            tripleQueue.FreeTensor(tripleLocal);
        }
        beginIdx += count;
    }
    totalCnt_ = beginIdx;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Process()
{
    // 主处理函数
    // mode=0: 完整流程 (发送 + 同步 + 接收)
    // mode=1: 仅发送模式 (用于单向带宽测试)
    if ASCEND_IS_AIV {
        // 所有核统一填充dstRankIds数据，确保后续SetStatus统计时数据有效
        DataCopyExtParams dstRankIdsParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> dstRankIdsCopyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(dstRankIdsTensor_, dstRankIdGMTensor_, dstRankIdsParams, dstRankIdsCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        if (mode_ == 0) {
            SendToMoeExpert(); // 步骤1: 发送数据到各rank的窗口
            SetStatus();       // 步骤2: 设置状态通知对端
            WaitDispatch();    // 步骤3: 等待所有rank完成发送
            LocalWindowCopy(); // 步骤4: 从窗口读取数据到输出
        } else if (mode_ == 1) {
            SendToMoeExpert(); // 步骤1: 发送数据到各rank的窗口
            SetStatus();
            WaitDispatch();
        }
    }
}

} // namespace BandwidthTestImpl
#endif // BANDWIDTH_TEST_H
