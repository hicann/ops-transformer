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
 * \file moe_ep_dispatch.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_H
#define MOE_EP_DISPATCH_H

#include <cstddef>

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif
#include "moe_ep_dispatch_tiling.h"
#include "moe_ep_dispatch_base.h"

#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#include "../common/mc2_moe_context.h"
#include "../common/moe_ep_exception_dump_writer.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"
#endif

namespace MoeEpDispatchImpl {

#define TemplateMoeEpDispatchTypeClass \
    typename XType, typename ScalesType, bool DoCpuSync, bool IsCached, bool IsTopkWeights, uint8_t NetworkMode
#define TemplateMoeEpDispatchTypeFunc XType, ScalesType, DoCpuSync, IsCached, IsTopkWeights, NetworkMode

#if defined(ENABLE_MOE_EP_KERNEL)

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;  // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;       // UB_ALIGN/sizeof(int32_t)=8
constexpr uint32_t INT64_UB_STRIDE = 4U; // UB_ALIGN/sizeof(int64_t)=4
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint32_t STATE_STRIDE = 15U; // (WIN_ADDR_ALIGN-UB_ALIGN)/UB_ALIGN=15
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_SIZE = 40 * 1024U; // 计算count 40KB per group
constexpr uint32_t EXPERT_NUM_PER_GROUP = 256U; // 直方图每次计算0-255
constexpr uint32_t DATA_BLOCK_NUM = 8U;         // 256B/32B=8
constexpr uint8_t BUFFER_NUM = 2;
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};

template <TemplateMoeEpDispatchTypeClass>
class MoeEpDispatch {
public:
    __aicore__ inline MoeEpDispatch(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BufferInit();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void CalSendCntPerRank(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t tmpOffset);
    __aicore__ inline void CalSendCntPerExpert(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                               uint32_t tmpOffset);
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void Communication();
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline void DedupAndSendDirect(uint32_t hitStride);
    __aicore__ inline void WriteSlotToLocal(uint32_t dstRankId, uint32_t slot);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void SendPhase();
    __aicore__ inline void GetSlotStartNum();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,
                                       uint32_t &sendNum);

    TPipe *tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_;
    GlobalTensor<int32_t> numRecvPerRankGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_;
    GlobalTensor<int32_t> scaleupCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntPerRankGMTensor_;
    GlobalTensor<int32_t> sendCntPerExpertGMTensor_;
    GlobalTensor<XType> slotGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

    LocalTensor<XType> xLocalTensor_;
    LocalTensor<XType> tokenSlotTensor_;
    LocalTensor<int32_t> topkIdxTensor_;
    LocalTensor<int32_t> dstSlotIdxTensor_;
    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<int64_t> numRecvPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerRankTensor_;
    LocalTensor<int32_t> slotIdxPerRankTensor_;
    LocalTensor<int32_t> hitPerRankTensor_;
    LocalTensor<uint8_t> hcommTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> perSlotQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> dstSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> sendCntRankBuf_;
    TBuf<> sendCntExpertBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR localCntStateWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    GM_ADDR localWinAddr_{nullptr};
    GM_ADDR payloadStashWinAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t scalesBytes_{0};
    uint32_t perSlotBytes_{0};
    uint32_t moeExpertNum_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t perGroupTokenNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t rankNumPerCore_{0};
    uint32_t hAlignSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t counterAlign512_{0};
    uint32_t perGroupSizeAlign_{0};
    uint32_t moeNumPerRankSize_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
    uint32_t moeExpertNumAlign512_{0};
    uint32_t cntFlagWinSize_{0};
    uint32_t dispatchNotifyCount_{1};
    uint32_t metaOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t payloadStashWinOffset_{0};
    uint64_t dstRankWinOffset_{0};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyParams sendPerExpertParams_;
    DataCopyParams slotCopyParams_;
    DataCopyPadParams padParams_;
};

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM,
    TPipe *pipe, const MoeEpDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    epRankId_ = mc2Context_->epRankId;
    constexpr size_t metadataOffset =
        offsetof(MoeEpDispatchTilingData, moeEpDispatchInfo) + offsetof(MoeEpDispatchInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_DISPATCH, tpipe_);

    const auto &info = tilingData->moeEpDispatchInfo;
    axisBS_ = info.cfg.numTokens;
    axisH_ = info.cfg.hidden;
    axisK_ = info.cfg.topK;
    epWorldSize_ = info.cfg.epWorldSize;
    moeExpertNum_ = info.cfg.numExperts;
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    scalesBytes_ = info.scalesBytes;
    perSlotBytes_ = info.perSlotBytes;
    aivNum_ = info.aivNum;
    cntWinStateOffset_ = info.window.cntWinStateOffset;
    slotWinStateOffset_ = info.window.slotWinStateOffset;
    winDataOffset_ = info.window.winDataOffset;
    payloadStashWinOffset_ = info.window.payloadStashWinOffset;
    dispatchNotifyCount_ = info.dispatchNotifyCount;
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);

    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    metaOffset_ = hAlignSize_;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_; // token num per group
    perGroupSizeAlign_ = Ceil(perGroupTokenNum_ * axisK_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    counterAlign512_ = epWorldSizeAlign512_ / sizeof(int32_t);
    moeNumPerRankSize_ = moeExpertNumPerRank_ * sizeof(int32_t);
    moeNumPerRankAlign_ = Ceil(moeNumPerRankSize_, UB_ALIGN) * UB_ALIGN;
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeNumPerRankAlign512_ = Ceil(moeNumPerRankSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeExpertNumAlign512_ = Ceil(moeExpertNum_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    cntFlagWinSize_ = epWorldSize_ * WIN_ADDR_ALIGN;
    dstRankWinOffset_ = static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_; // 目标窗口地址偏移

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)scales);
        metaOffset_ += Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedSlotIdx);
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstBufferSlotIdx);

    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, static_cast<uint16_t>(STATE_STRIDE), 0U};
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U, static_cast<uint16_t>(STATE_STRIDE)};
    sendPerExpertParams_ = {1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
    slotCopyParams_ = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    padParams_ = {true, 0, 0, 0};

    scaleupCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = scaleupCounterAddr_ + aivNum_ * epWorldSizeAlign512_;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_);
    payloadStashWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadStashWinOffset_);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleupCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)localCntStateWinAddr_);
    sendCntPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sendCntWorkspaceAddr_);
    sendCntPerExpertGMTensor_.SetGlobalBuffer(
        (__gm__ int32_t *)(sendCntWorkspaceAddr_ + epWorldSize_ * WIN_ADDR_ALIGN));
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SplitToCore(uint32_t curSendCnt,
                                                                                 uint32_t curUseAivNum,
                                                                                 uint32_t &startId, uint32_t &endId,
                                                                                 uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的数量
    uint32_t remainderNum = curSendCnt % curUseAivNum; // 余数
    startId = sendNum * aivId_;                        // 每个aiv发送时的起始rankid
    if (aivId_ < remainderNum) {                       // 前remainderRankNum个aiv需要多发1个卡的数据
        sendNum++;
        startId += aivId_;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BufferInit()
{
    uint32_t sendCntRankSizeAlign = Ceil(epWorldSize_ * UB_ALIGN, ALIGNED_LEN_256) * ALIGNED_LEN_256;

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    tpipe_->InitBuffer(dstSlotQueue_, BUFFER_NUM, kAlignSize_);
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(dstExpBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(sendCntRankBuf_, sendCntRankSizeAlign);
    tpipe_->InitBuffer(sendCntExpertBuf_, moeExpertNumAlign_);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, 2 * moeNumPerRankAlign_);
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();
    sendCntPerRankTensor_ = sendCntRankBuf_.Get<int32_t>();
    sendCntPerExpertTensor_ = sendCntExpertBuf_.Get<int32_t>();
    Duplicate<int32_t>(sendCntPerRankTensor_, 0, epWorldSize_ * UB_STRIDE);
    Duplicate<int32_t>(sendCntPerExpertTensor_, 0, moeExpertNumAlign_ / sizeof(int32_t));
    if (aivId_ == 0) {                             // 状态位仅做1次累加
        uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是8个datablock，每8个数将首个设置为1
        Duplicate<int32_t>(sendCntPerRankTensor_, 1, mask, Ceil(epWorldSize_, DATA_BLOCK_NUM), 1, DATA_BLOCK_NUM);
    }
    ResetCounters();
    SyncAll<true>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ResetCounters()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
    DataCopy(sendCntPerRankGMTensor_, sendCntPerRankTensor_, clearStatusCopyParams_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerRank(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t tmpOffset)
{
    uint32_t tokenCnt = calCnt / axisK_;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    uint32_t tokenCntAlign = Ceil(tokenCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);
    uint32_t mask = tokenCnt;
    uint32_t shape[2] = {tokenCnt, axisK_};
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = maskTensorInt16.template ReinterpretCast<uint8_t>();

    Duplicate<int16_t>(dstTensorInt16, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    Div(tempTensorInt16, expertIdsTensor, dstTensorInt16, calCnt);
    // 筛选无效expert id 消除影响
    CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    Select(dstTensorInt16, gatherMaskTensorInt8, tempTensorInt16, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt);

    for (uint32_t dstRankId = 0; dstRankId < epWorldSize_; dstRankId++) {
        // 筛选出发送到目标卡的token
        uint64_t rsvdCnt = 0;
        uint32_t offset = dstRankId * UB_STRIDE + 1;
        Subs(expertIdsTensor, dstTensorInt16, static_cast<int16_t>(dstRankId), calCnt);
        Abs(tempTensorInt16, expertIdsTensor, calCnt);
        ReduceMin<int16_t, Pattern::Reduce::AR, true>(expertIdsTensor, tempTensorInt16, gatherMaskTensorInt8, shape,
                                                      false);   // 0为目标
        Duplicate<uint16_t>(maskTensorInt16, 0, tokenCntAlign); // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::EQ,
                      tokenCntAlign);
        GatherMask(tempTensorInt16, expertIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        if (rsvdCnt == 0) {
            continue;
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t curRankCnt = sendCntPerRankTensor_.GetValue(offset);
        sendCntPerRankTensor_.SetValue(offset, curRankCnt + static_cast<int32_t>(rsvdCnt));
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t tmpOffset)
{
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    uint32_t groupNum = Ceil(moeExpertNum_, EXPERT_NUM_PER_GROUP);
    uint32_t maskU16Cnt = Ceil(calCntAlign, BITS_PER_BYTE * sizeof(uint16_t));
    LocalTensor<int16_t> tempTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<uint8_t> dstTensorU8 = dstExpBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> dstTensorU32 = dstExpBuf_.Get<uint32_t>();
    LocalTensor<uint16_t> gatherMaskTensorU16 = topkIdsBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> gatherMaskU16GE = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> gatherMaskU16LT = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskU8GE = gatherMaskU16GE.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> gatherMaskU8LT = gatherMaskU16LT.ReinterpretCast<uint8_t>();

    for (uint32_t group = 0; group < groupNum; group++) {
        uint32_t baseExpertId = group * EXPERT_NUM_PER_GROUP;
        uint32_t curGroupSize = (group == groupNum - 1) ? (moeExpertNum_ - baseExpertId) : EXPERT_NUM_PER_GROUP;
        uint64_t rsvdCnt = 0;
        Subs(dstTensorInt16, expertIdsTensor, static_cast<int16_t>(baseExpertId), calCnt);
        Duplicate<uint16_t>(gatherMaskTensorU16, 0, calCntAlign * 2);
        CompareScalar(gatherMaskU8GE, dstTensorInt16, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CompareScalar(gatherMaskU8LT, dstTensorInt16, static_cast<int16_t>(curGroupSize - 1), AscendC::CMPMODE::LE,
                      calCntAlign);
        And(gatherMaskU16GE, gatherMaskU16GE, gatherMaskU16LT, maskU16Cnt);
        GatherMask(tempTensor, dstTensorInt16, gatherMaskU16GE, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        if (rsvdCnt == 0) {
            continue;
        }

        Cast(dstTensorU8, tempTensor, RoundMode::CAST_NONE, rsvdCnt);
        GetExpertFreq(gatherMaskTensorU16, dstTensorU8, rsvdCnt);
        Cast(dstTensorU32, gatherMaskTensorU16, RoundMode::CAST_NONE, curGroupSize);
        Add(sendCntPerExpertTensor_[baseExpertId], sendCntPerExpertTensor_[baseExpertId],
            dstTensorU32.ReinterpretCast<int32_t>(), static_cast<int32_t>(curGroupSize));
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_); // 按token数量分核
    if (startTokenId_ >= axisBS_) {
        return;
    }

    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    uint32_t calCnt = perGroupTokenNum_ * axisK_;
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    DataCopyParams sendPerRankParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(UB_ALIGN), 0U,
                                        static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN)};
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            calCnt = (sendTokenNum_ - group * perGroupTokenNum_) * axisK_;
        }

        uint32_t topkIdxOffset = (startTokenId_ + group * perGroupTokenNum_) * axisK_;
        uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
        DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        if (group > 0) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParams,
                    topkIdsCntCopyPadParams); // copy topkId
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, calCnt);

        // perExpert 和 perRank count 计算
        CalSendCntPerExpert(tempTensorInt16, calCnt, tmpOffset);
        CalSendCntPerRank(tempTensorInt16, calCnt, tmpOffset);
    }

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SetAtomicAdd<int32_t>();
    DataCopyPad(sendCntPerRankGMTensor_, sendCntPerRankTensor_, sendPerRankParams);
    DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
    SetAtomicNone();

    if constexpr (!IsCached) { // 非cache模式下计算每个核slot起始值
        LocalTensor<uint32_t> gatherTmpTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
        gatherTmpTensor.SetValue(0, 2); // 设置掩码，取源操作数每个datablock中的第2个元素
        uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
        uint64_t rsvdCnt = 0;
        GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(numRecvPerRankTensor_, sendCntPerRankTensor_, gatherTmpTensor, true, mask, recvMaskParams, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(scaleupCounterGMTensor_[aivId_ * counterAlign512_], numRecvPerRankTensor_, counterCnt_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_); // 按卡分核
    if (startRankId_ >= epWorldSize_) {                                            // 空闲核，直接返回
        return;
    }

    GlobalTensor<uint64_t> numSendGMTensorU64;
    numSendGMTensorU64.SetGlobalBuffer((__gm__ uint64_t *)(sendCntWorkspaceAddr_ + startRankId_ * WIN_ADDR_ALIGN));
    LocalTensor<uint64_t> sendCntPerRankU64 = sendCntRankBuf_.Get<uint64_t>();
    DataCopyParams cntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), 1U, static_cast<uint16_t>(STATE_STRIDE),
                                    0U};
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopy(sendCntPerRankU64, numSendGMTensorU64, cntCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint32_t notifyStride = (dstRankId - startRankId_) * INT64_UB_STRIDE;
        uint32_t srcOffset = dstRankId * moeExpertNumPerRank_;
        uint64_t notifyVal = sendCntPerRankU64.GetValue(notifyStride);
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
        GM_ADDR remoteCountAddr = remoteStateAddr + cntFlagWinSize_ + epRankId_ * moeNumPerRankAlign512_;
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + cntFlagWinSize_ + srcOffset * sizeof(int32_t);

        if (dstRankId != epRankId_) { // 远端 使用URMA发送 count + state
            uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId);
            hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DEFAULT_WQE_CONFIG>(
                commHandle, remoteCountAddr, srcWorkspaceAddr, moeNumPerRankSize_, notifyAddr, notifyVal);
            continue;
        }

        // 本端
        GlobalTensor<int32_t> countGMTensor;
        GlobalTensor<uint64_t> notifyGMTensor;
        countGMTensor.SetGlobalBuffer((__gm__ int32_t *)remoteCountAddr);
        notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t *)notifyAddr);
        LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.Get<int32_t>();
        DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeNumPerRankSize_), 0U, 0U};
        DataCopyPad(cntPerExpertTensor, sendCntPerExpertGMTensor_[srcOffset], expertCntCopyParams, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(countGMTensor, cntPerExpertTensor, expertCntCopyParams);
        PipeBarrier<PIPE_MTE3>(); // perExpert 写完再写notifyVal
        DataCopy(notifyGMTensor, sendCntPerRankU64[notifyStride], INT64_UB_STRIDE);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetRecvCount()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    // 最后一个核处理recvCount
    uint32_t mask = 1;
    int32_t sumOfFlag = -1;
    int32_t commpareFlag = static_cast<int32_t>(epWorldSize_);
    LocalTensor<int32_t> recvCounterTensor = topkIdsBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> tempFp32 = topkIdsBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, epWorldSize_ * UB_ALIGN);
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // buf 复用
    while (sumOfFlag != commpareFlag) {     // 状态位check
        DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
    }
    SetRecvNumPerExpert();                // 计算本卡上各专家接收的token总数
    SetRecvNumPerRank(recvCounterTensor); // 计算本端接收来自各卡的token总数
    // status clear
    Duplicate<int32_t>(recvCounterTensor, 0, epWorldSize_ * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_COUNT_READY);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerExpert()
{
    uint32_t recvExpertAlign = moeNumPerRankAlign_ / sizeof(int32_t);
    GlobalTensor<int32_t> localExpertRecvGMTensor;
    localExpertRecvGMTensor.SetGlobalBuffer((__gm__ int32_t *)(localCntStateWinAddr_ + cntFlagWinSize_));
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int64_t> recvCntTensor = tempBuf_.GetWithOffset<int64_t>(INT64_UB_STRIDE, 0);
    LocalTensor<int32_t> recvTensorInt32 = tempBuf_.GetWithOffset<int32_t>(epWorldSize_ * recvExpertAlign, UB_ALIGN);
    LocalTensor<int64_t> tempRecvTensorInt64 = dstExpBuf_.Get<int64_t>();
    LocalTensor<int64_t> sharedTmp = recvTensorInt32.template ReinterpretCast<int64_t>();
    LocalTensor<uint8_t> sharedTmpInt8 = recvTensorInt32.template ReinterpretCast<uint8_t>();

    const uint32_t shape[] = {epWorldSize_, recvExpertAlign};
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(moeNumPerRankAlign_),
                                      static_cast<uint16_t>(moeNumPerRankAlign512_ - moeNumPerRankAlign_), 0U};
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    DataCopyPad(recvTensorInt32, localExpertRecvGMTensor, inRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(tempRecvTensorInt64, recvTensorInt32, RoundMode::CAST_NONE, epWorldSize_ * recvExpertAlign);
    ReduceSum<int64_t, AscendC::Pattern::Reduce::RA, true>(numRecvPerExpertTensor_, tempRecvTensorInt64, sharedTmpInt8,
                                                           shape, true);

    if constexpr (DoCpuSync) { // 计算actualA，并写入host pin
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t *)hostPinnedCounterAddrGM_);
        ReduceSum(recvCntTensor, numRecvPerExpertTensor_, sharedTmp, moeExpertNumPerRank_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(hostPinnedCounterTensor, recvCntTensor, INT64_UB_STRIDE);
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerExpertGMTensor_, numRecvPerExpertTensor_, recvPerExpertParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerRank(
    LocalTensor<int32_t> recvTmpTensor)
{
    LocalTensor<uint32_t> gatherTmpTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    gatherTmpTensor.SetValue(0, 2); // 设置掩码，取源操作数每个datablock中的第2个元素
    uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
    uint64_t rsvdCnt = 0;
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherTmpTensor, true, mask, recvMaskParams, rsvdCnt);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DedupAndSendDirect(uint32_t hitStride)
{
    for (uint32_t k = 0; k < axisK_; k++) {
        int32_t expertId = topkIdxTensor_.GetValue(k);
        uint32_t dstRankId = expertId / moeExpertNumPerRank_;
        int32_t hit = (expertId < 0) ? -1 : hitPerRankTensor_.GetValue(hitStride + dstRankId);
        if ((hit >= 0) || (expertId < 0)) {
            dstSlotIdxTensor_.SetValue(k, hit);
            continue;
        }

        int32_t slot = slotIdxPerRankTensor_.GetValue(dstRankId);
        dstSlotIdxTensor_.SetValue(k, slot);
        hitPerRankTensor_.SetValue(hitStride + dstRankId, slot);
        slotIdxPerRankTensor_.SetValue(dstRankId, slot + 1);

        // 本卡 token 直接写 receive window，远端 token 写本卡 send window。
        WriteSlotToLocal(dstRankId, slot);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteSlotToLocal(uint32_t dstRankId, uint32_t slot)
{
    GM_ADDR rankGM = (dstRankId == epRankId_) ? localWinAddr_ : payloadStashWinAddr_;
    GM_ADDR slotAddr = rankGM + (static_cast<uint64_t>(dstRankId) * axisMaxBS_ + slot) * perSlotBytes_;
    slotGMTensor_.SetGlobalBuffer((__gm__ XType *)slotAddr);
    DataCopyPad(slotGMTensor_, tokenSlotTensor_, slotCopyParams_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    if (startRankId_ >= epWorldSize_) { // 空闲核，直接返回
        return;
    }

    LocalTensor<int32_t> sendNumPerRankTensor = sendCntRankBuf_.Get<int32_t>();
    DataCopyParams sendCntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), 1U, static_cast<uint16_t>(STATE_STRIDE),
                                        0U};
    DataCopy(sendNumPerRankTensor, sendCntPerRankGMTensor_[startRankId_ * WIN_ADDR_ALIGN / sizeof(int32_t)],
             sendCntCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint32_t sendTokenNum = sendNumPerRankTensor.GetValue((dstRankId - startRankId_) * UB_STRIDE + 1);
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
                             static_cast<uint64_t>(epRankId_) * dispatchNotifyCount_ * WIN_ADDR_ALIGN;
        if (unlikely(dstRankId == epRankId_)) {
            uint32_t elemCnt = dispatchNotifyCount_ * UB_STRIDE;
            LocalTensor<int32_t> statusTensor = tempBuf_.GetWithOffset<int32_t>(elemCnt, 0);
            GlobalTensor<int32_t> statusGMTensor;
            statusGMTensor.SetGlobalBuffer((__gm__ int32_t *)(notifyAddr));
            Duplicate<int32_t>(statusTensor, 1, elemCnt);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopyParams copyParams = {static_cast<uint16_t>(dispatchNotifyCount_), 1U, 0U,
                                         static_cast<uint16_t>(STATE_STRIDE)};
            DataCopy(statusGMTensor, statusTensor, copyParams);
            continue; //  本端slot已经写入win
        }

        uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId);
        uint32_t slotCntPerNotify = sendTokenNum / dispatchNotifyCount_;
        uint32_t remainder = sendTokenNum % dispatchNotifyCount_;
        GM_ADDR remoteWinAddr = GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset_;
        GM_ADDR localSendWinAddr = payloadStashWinAddr_ + static_cast<uint64_t>(dstRankId) * axisMaxBS_ * perSlotBytes_;
        for (uint32_t index = 0; index < dispatchNotifyCount_; index++) {
            uint32_t slotCnt = (index < remainder) ? slotCntPerNotify + 1 : slotCntPerNotify;
            uint64_t dataSize = (slotCnt > 0) ? static_cast<uint64_t>(perSlotBytes_) * slotCnt : UB_ALIGN;
            hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DEFAULT_WQE_CONFIG>(
                commHandle, remoteWinAddr, localSendWinAddr, dataSize, notifyAddr, 1);
            remoteWinAddr += dataSize;
            localSendWinAddr += dataSize;
            notifyAddr += WIN_ADDR_ALIGN;
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetSlotStartNum()
{
    slotIdxPerRankTensor_ = sendCntExpertBuf_.GetWithOffset<int32_t>(epWorldSizeAlign_ / sizeof(int32_t), 0);
    Duplicate<int32_t>(slotIdxPerRankTensor_, 0, epWorldSize_);
    if (aivId_ == 0) { // 0核起始id 均为0
        return;
    }
    uint32_t groupCnt = Ceil(aivId_ * epWorldSizeAlign_, perGroupSizeAlign_ * 2);
    uint32_t copyNumPerGroup = perGroupSizeAlign_ * 2 / epWorldSizeAlign_;
    LocalTensor<int32_t> counterTmpTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> hitTmpTensor = dstExpBuf_.GetWithOffset<int32_t>(epWorldSizeAlign_ / sizeof(int32_t), 0);
    LocalTensor<uint8_t> sharedTmpTensor = tempBuf_.Get<uint8_t>();
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    for (uint32_t i = 0; i < groupCnt; i++) {
        uint32_t copyNum = (i == groupCnt - 1) ? (aivId_ - copyNumPerGroup * i) : copyNumPerGroup;
        uint32_t gmOffset = i * copyNumPerGroup * counterAlign512_;
        DataCopyParams counterCopyParams = {static_cast<uint16_t>(copyNum), static_cast<uint16_t>(epWorldSizeAlign_),
                                            static_cast<uint16_t>(epWorldSizeAlign512_ - epWorldSizeAlign_), 0U};
        DataCopyPad(counterTmpTensor, scaleupCounterGMTensor_[gmOffset], counterCopyParams, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        const uint32_t shape[] = {copyNum, static_cast<uint32_t>(counterCnt_)};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(hitTmpTensor, counterTmpTensor, sharedTmpTensor, shape,
                                                               false);
        Add(slotIdxPerRankTensor_, slotIdxPerRankTensor_, hitTmpTensor, epWorldSize_);
        if (i + 1 < groupCnt) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }

    uint32_t groupCnt = perGroupSizeAlign_ / epWorldSizeAlign_;
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams topkCopyParams = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    DataCopyParams scalesCopyParams = {1U, static_cast<uint16_t>(scalesBytes_), 0U, 0U};
    hitPerRankTensor_ = dstExpBuf_.Get<int32_t>();

    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
        uint32_t topkOffset = tokenId * axisK_;
        uint32_t groupIdx = (tokenId - startTokenId_) % groupCnt;
        uint32_t hitStride = groupIdx * epWorldSizeAlign_ / sizeof(int32_t);
        if (groupIdx == 0) {
            SyncFunc<AscendC::HardEvent::S_V>();
            Duplicate<int32_t>(hitPerRankTensor_, -1, perGroupSizeAlign_ / sizeof(int32_t));
            SyncFunc<AscendC::HardEvent::V_S>();
        }
        xLocalTensor_ = perSlotQueue_.AllocTensor<XType>();
        metaLocalTensor_ = xLocalTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        DataCopyPad(xLocalTensor_, xGMTensor_[tokenId * axisH_], xCopyParams, padParams_);
        if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
            DataCopyPad(xLocalTensor_[hAlignSize_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                        scalesGMTensor_[tokenId * scalesBytes_ / sizeof(ScalesType)], scalesCopyParams, padParams_);
        }
        DataCopyPad(metaLocalTensor_, topkIdxGMTensor_[topkOffset], topkCopyParams, padParams_);
        if constexpr (IsTopkWeights) {
            DataCopyPad(metaLocalTensor_[axisKAlign_].template ReinterpretCast<float>(),
                        topkWeightsGMTensor_[topkOffset], topkCopyParams, padParams_);
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        metaLocalTensor_.SetValue(2 * axisKAlign_, epRankId_);
        metaLocalTensor_.SetValue(2 * axisKAlign_ + 1, tokenId);
        perSlotQueue_.EnQue(xLocalTensor_);
        tokenSlotTensor_ = perSlotQueue_.DeQue<XType>();
        topkIdxTensor_ = tokenSlotTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        dstSlotIdxTensor_ = dstSlotQueue_.AllocTensor<int32_t>();

        if constexpr (!IsCached) {
            DedupAndSendDirect(hitStride);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
        } else {
            DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkOffset], topkCopyParams, padParams_);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            for (uint32_t k = 0; k < axisK_; k++) {
                int32_t slot = dstSlotIdxTensor_.GetValue(k);
                if (slot == -1) {
                    continue;
                }
                int32_t expertId = topkIdxTensor_.GetValue(k);
                uint32_t dstRankId = expertId / moeExpertNumPerRank_;
                if (hitPerRankTensor_.GetValue(hitStride + dstRankId) == slot) {
                    continue;
                }
                WriteSlotToLocal(dstRankId, slot);
                hitPerRankTensor_.SetValue(hitStride + dstRankId, slot);
            }
        }
        DataCopyPad(dstSlotIdxGMTensor_[topkOffset], dstSlotIdxTensor_, topkCopyParams);
        perSlotQueue_.FreeTensor<XType>(tokenSlotTensor_);
        dstSlotQueue_.FreeTensor<int32_t>(dstSlotIdxTensor_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        BufferInit();
        CalSendCnt();
        SyncAll<true>();
        Communication();
        GetRecvCount();
        if constexpr (!IsCached) { // 计算起始slot id
            GetSlotStartNum();
        }
        SendPhase();
        SyncAll<true>();
        WriteToRemoteWindow();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
    }
}

#endif

} // namespace MoeEpDispatchImpl

#endif // MOE_EP_DISPATCH_H
