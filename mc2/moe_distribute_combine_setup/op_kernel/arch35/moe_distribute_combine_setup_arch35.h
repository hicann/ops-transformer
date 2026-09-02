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
 * \file moe_distribute_combine_setup_arch35.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
#define MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H

#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_combine_setup_base.h"
#include "../moe_distribute_combine_setup_tiling_data.h"

namespace MoeDistributeCombineSetupImpl {

struct WriteWithNotifySQEInfoParams {
    uint64_t dataSrcAddr;
    uint64_t dataDstAddr;
    uint32_t length;
    uint64_t notifyAddr;
    uint64_t notifyData;
    uint8_t cqe;
};

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeCombineSetup {
    constexpr static uint8_t BUFFER_NUM = 2;              // 多buf
    constexpr static uint64_t STATE_OFFSET = 512U;        // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024U * 1024U; // 1M
    constexpr static uint32_t UB_ALIGN = 32U;             // UB按32字节对齐
    constexpr static uint64_t STATE_SIZE_PER_CORE = 512U; // 数据和状态的0/1区标识占用空间
    constexpr static uint64_t COMBINE_STATE_OFFSET = 0U; // 本卡状态空间偏移地址，前面的地址给dispatch用
    constexpr static uint32_t STATE_COUNT_THRESHOLD = 512U; // moeExpertNumPerRank*epWorldSize状态数阈值

public:
    __aicore__ inline MoeDistributeCombineSetup(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX,
                                GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitCoreCalForExpert();
    __aicore__ inline void SplitCoreCal(); // 按卡分核，用于敲doorbell
    __aicore__ inline void CurRankComm(const WriteWithNotifySQEInfoParams &params);
    __aicore__ inline void Communication();
    __aicore__ inline void CallJfsDoorbell();
    __aicore__ inline void BuffInit();
    __aicore__ inline void AssistInfoLocalCopy();

    __aicore__ GM_ADDR GetWinAddrByRankId(const uint32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]) + winDataSizeOffset_ +
               expertPerSizeOnWin_ * static_cast<uint64_t>(expertLocalId);
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(uint32_t rankId)
    {
        return (GM_ADDR)(hcclContext_->windowsOut[rankId]) + COMBINE_STATE_OFFSET +
               WIN_STATE_OFFSET * static_cast<uint64_t>(dataState_);
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> assistInfoForCombineGlobal_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGM_;
    GM_ADDR expandXGM_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uint32_t进行处理
    const MoeDistributeCombineSetupInfo *moeDistributeCombineSetupInfo_{nullptr};
    uint32_t axisMaxBS_{0};
    uint32_t coreIdx_{0};    // aiv id
    uint32_t moeSendNum_{0}; // moeExpertPerRankNum * epWorldSize
    uint32_t curRankExpertNum_{0};
    uint64_t epDataOffsetOnWin_{0};
    uint64_t epStateOffsetOnWin_{0};
    uint64_t axisHExpandXTypeSize_{0};
    uint32_t startExpertId_{0}; // 当前核处理的起始专家序号（组装SQE阶段）
    uint32_t endExpertId_{0};   // 当前核处理的结束专家序号（组装SQE阶段）
    uint32_t sendExpertNum_{0};
    uint32_t startRankId_{0}; // 当前核负责敲doorbell的起始卡号
    uint32_t endRankId_{0};   // 当前核负责敲doorbell的结束卡号
    uint32_t sendRankNum_{0}; // 当前核负责敲doorbell的卡数
    uint32_t dataState_{0};
    uint64_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t remain_ub_space{0};
    bool isShardExpert_{false};

    TQue<QuePosition::VECIN, 1> assistInfoQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1>
        expertTokenTmpQueue_; // URMA本卡回环通信拷贝数据Local Memory暂存

    // URMA通信新增变量、Buf
    TBuf<> urmaSqInfoBuf_;  // 80B
    TBuf<> urmaCqInfoBuf_;  // 48B
    TBuf<> cqeBuf_;         // 32B * CQ_DEPTH_256
    TBuf<> templateSqeBuf_; // 96B
    TBuf<> piCiBuf_;        // 32B，用于 GetPICI 的 MTE 搬运

    __gm__ HcclCombinOpParam *hcclContext_{nullptr};
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR /*expertIds*/, GM_ADDR assistInfoForCombine, GM_ADDR /*quantExpandX*/,
    GM_ADDR /*commCmdInfoOut*/, GM_ADDR /*workspaceGM*/, TPipe *pipe,
    const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    moeDistributeCombineSetupInfo_ = &(tilingData->moeDistributeCombineSetupInfo);
    hcclContext_ = (__gm__ HcclCombinOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    // 获取win状态区地址，并保证数据一致
    // 在1M中选择512K偏移后的1.5k空间记录本卡历史状态
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)hcclContext_->windowsOut[moeDistributeCombineSetupInfo_->epRankId];
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET +
                                                            STATE_SIZE_PER_CORE * static_cast<uint64_t>(coreIdx_)));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);

    dataState_ = selfDataStatusTensor(0); // win状态区的0/1标识
    expandXGM_ = expandX;
    assistInfoForCombineGlobal_.SetGlobalBuffer((__gm__ int32_t *)assistInfoForCombine);

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    axisMaxBS_ = moeDistributeCombineSetupInfo_->globalBs / moeDistributeCombineSetupInfo_->epWorldSize;
    moeSendNum_ = moeDistributeCombineSetupInfo_->epWorldSize * moeDistributeCombineSetupInfo_->moeExpertPerRankNum;
    isShardExpert_ = (moeDistributeCombineSetupInfo_->epRankId <
                      moeDistributeCombineSetupInfo_->sharedExpertRankNum); // 当前rank是否为共享专家
    curRankExpertNum_ = (isShardExpert_) ? 1U : moeDistributeCombineSetupInfo_->moeExpertPerRankNum; // 本卡专家数

    axisHExpandXTypeSize_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->h) *
                            static_cast<uint64_t>(sizeof(ExpandXType));              // 一个token占用内存
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS_) * axisHExpandXTypeSize_; // 单个专家可能占用的最大空间

    winDataSizeOffset_ =
        static_cast<uint64_t>(dataState_) * (static_cast<uint64_t>(moeDistributeCombineSetupInfo_->totalWinSize) >> 1);
    stateOffset_ = (moeSendNum_ > STATE_COUNT_THRESHOLD) ? (STATE_OFFSET >> 1) : STATE_OFFSET;
    epStateOffsetOnWin_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->epRankId) * stateOffset_;
    epDataOffsetOnWin_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->epRankId) *
                         static_cast<uint64_t>(moeDistributeCombineSetupInfo_->moeExpertPerRankNum) *
                         expertPerSizeOnWin_; // 前面rank数据区占用内存地址偏移

    epWindowGM_ = GetWinAddrByRankId(moeDistributeCombineSetupInfo_->epRankId);
    epStatusSpaceGM_ = GetWinStateAddrByRankId(moeDistributeCombineSetupInfo_->epRankId);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXType>((__gm__ ExpandXType *)(epWindowGM_), moeDistributeCombineSetupInfo_->totalWinSize);
    OOMCheckAddrRange<float>((__gm__ float *)(epStatusSpaceGM_), STATE_SIZE);
#endif

    // 分核计算
    SplitCoreCalForExpert();
    SplitCoreCal();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SplitCoreCalForExpert()
{
    // 按epWorldSize*本卡专家数分核
    uint32_t coreIdxNew =
        (coreIdx_ + moeDistributeCombineSetupInfo_->epRankId) % moeDistributeCombineSetupInfo_->aivNum;
    sendExpertNum_ =
        moeDistributeCombineSetupInfo_->epWorldSize * curRankExpertNum_ / moeDistributeCombineSetupInfo_->aivNum;
    uint32_t remainderExpertNum =
        moeDistributeCombineSetupInfo_->epWorldSize * curRankExpertNum_ % moeDistributeCombineSetupInfo_->aivNum;
    startExpertId_ = sendExpertNum_ * coreIdxNew;
    if (coreIdxNew < remainderExpertNum) {
        ++sendExpertNum_;
        startExpertId_ += coreIdxNew;
    } else {
        startExpertId_ += remainderExpertNum;
    }
    endExpertId_ = startExpertId_ + sendExpertNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 按卡数分核，用于敲doorbell阶段
    uint32_t coreIdxNew =
        (coreIdx_ + moeDistributeCombineSetupInfo_->epRankId) % moeDistributeCombineSetupInfo_->aivNum;
    sendRankNum_ = moeDistributeCombineSetupInfo_->epWorldSize / moeDistributeCombineSetupInfo_->aivNum;
    uint32_t remainderRankNum = moeDistributeCombineSetupInfo_->epWorldSize % moeDistributeCombineSetupInfo_->aivNum;
    startRankId_ = sendRankNum_ * coreIdxNew;
    if (coreIdxNew < remainderRankNum) {
        ++sendRankNum_;
        startRankId_ += coreIdxNew;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::CallJfsDoorbell()
{
    // 按卡分核，区分本卡通信和其他卡通信
    LocalTensor<uint8_t> sqInfoU8 = urmaSqInfoBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> cqInfoU8 = urmaCqInfoBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> cqeTensorU8 = cqeBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> piCiU32 = piCiBuf_.Get<uint32_t>();

    for (uint32_t rankId = startRankId_; rankId < endRankId_; ++rankId) {
        if (unlikely(rankId == moeDistributeCombineSetupInfo_->epRankId)) {
            // 本卡回环通信：写状态
            GlobalTensor<int32_t> selfStatusTensor;
            selfStatusTensor.SetGlobalBuffer(
                (__gm__ int32_t *)(GetWinStateAddrByRankId(moeDistributeCombineSetupInfo_->epRankId) +
                                   epStateOffsetOnWin_));

            selfStatusTensor.SetValue(0, 0x3f800000);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusTensor);
        } else {
            // 其他卡通信：PollCQ -> UpdatePICI -> 敲JFS DoorBell
            GetURMASqInfoTensor(sqInfoU8, (GM_ADDR)hcclContext_, rankId);
            GetURMACqInfoTensor(cqInfoU8, (GM_ADDR)hcclContext_, rankId);

            // 获取PI CI
            GetPICI(piCiU32, (GM_ADDR)hcclContext_, moeDistributeCombineSetupInfo_->epRankId, rankId);
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
            uint32_t sqPi = piCiU32.GetValue(WIN_SQPI_OFFSET);
            uint32_t sqCi = piCiU32.GetValue(WIN_SQCI_OFFSET);
            uint32_t cqPi = piCiU32.GetValue(WIN_CQPI_OFFSET);
            uint32_t cqCi = piCiU32.GetValue(WIN_CQCI_OFFSET);
            uint32_t sqPiLinear = piCiU32.GetValue(WIN_SQPILINEAR_OFFSET);
            uint32_t cqCiLinear = piCiU32.GetValue(WIN_CQCILINEAR_OFFSET);
            bool isFirstInitCqe = (piCiU32.GetValue(WIN_FIRST_TIME_CREATE_WIN_FLAG_OFFSET) > 0);

            // 首次进入通信域后初始化一次CQE
            if (unlikely(!isFirstInitCqe)) {
                InvalidateCqeStatus(cqInfoU8, cqeTensorU8);
                AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 等cqGlobalTensor的Local->GM，后续GM->Local
            }

            // 更新本地的sqPi
            LocalTensor<uint32_t> sqInfoU32 = sqInfoU8.ReinterpretCast<uint32_t>();
            uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);
            sqPiLinear += (curRankExpertNum_ << 1);
            sqPi = sqPiLinear % sqDepth;

            // PollCQ更新sqCi和cqCi
            PollCommCQUpdateSQCI(sqInfoU8, cqInfoU8, cqeTensorU8, sqCi, cqCi, cqCiLinear);

            // 敲JFS DoorBell
            SendJFSDoorBell(sqInfoU8, sqPiLinear);

            // 更新PI CI
            piCiU32.SetValue(WIN_SQPI_OFFSET, sqPi);
            piCiU32.SetValue(WIN_SQCI_OFFSET, sqCi);
            piCiU32.SetValue(WIN_CQPI_OFFSET, cqPi);
            piCiU32.SetValue(WIN_CQCI_OFFSET, cqCi);
            piCiU32.SetValue(WIN_SQPILINEAR_OFFSET, sqPiLinear);
            piCiU32.SetValue(WIN_CQCILINEAR_OFFSET, cqCiLinear);
            piCiU32.SetValue(WIN_FIRST_TIME_CREATE_WIN_FLAG_OFFSET, 1U);
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
            UpdatePICI(piCiU32, (GM_ADDR)hcclContext_, moeDistributeCombineSetupInfo_->epRankId, rankId);
            AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 防止piCiU32被覆盖
            AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();    // 防止sqInfoU8被覆盖
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::BuffInit()
{
    uint32_t assistInfoQueueSize = Ceil(moeSendNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t urmaSqInfoBufSize = Ceil(static_cast<uint32_t>(sizeof(HcclAiRMAWQ)), UB_ALIGN) * UB_ALIGN;
    uint32_t urmaCqInfoBufSize = Ceil(static_cast<uint32_t>(sizeof(HcclAiRMACQ)), UB_ALIGN) * UB_ALIGN;
    uint32_t cqeBufSize = UB_ALIGN * CQ_DEPTH_256;
    uint32_t templateSqeBufSize = Ceil(WRITE_WITH_NOTIFY_SQE_SIZE, UB_ALIGN) * UB_ALIGN;
    uint32_t piCiBufSize = UB_ALIGN;
    remain_ub_space = moeDistributeCombineSetupInfo_->totalUbSize - assistInfoQueueSize - urmaSqInfoBufSize -
                      urmaCqInfoBufSize - cqeBufSize - templateSqeBufSize - piCiBufSize;

    tpipe_->Reset();
    tpipe_->InitBuffer(assistInfoQueue_, 1, assistInfoQueueSize); // epWorldSize * moeExpertPerRankNum * 4

    // 初始化urma相关buf
    tpipe_->InitBuffer(urmaSqInfoBuf_, urmaSqInfoBufSize);
    tpipe_->InitBuffer(urmaCqInfoBuf_, urmaCqInfoBufSize);
    tpipe_->InitBuffer(cqeBuf_, cqeBufSize);
    tpipe_->InitBuffer(templateSqeBuf_, templateSqeBufSize);
    tpipe_->InitBuffer(piCiBuf_, piCiBufSize);

    tpipe_->InitBuffer(expertTokenTmpQueue_, 1, static_cast<uint32_t>(remain_ub_space));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::AssistInfoLocalCopy()
{
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.AllocTensor<int32_t>();

    DataCopyExtParams epSendCntParams;
    if (isShardExpert_) {
        // 对于共享专家来说assistInfoForCombine输入维度为epWorldSize个
        epSendCntParams = {1U, moeDistributeCombineSetupInfo_->epWorldSize * static_cast<uint32_t>(sizeof(uint32_t)),
                           0U, 0U, 0U};
    } else {
        epSendCntParams = {1U, moeSendNum_ * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    }
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(assistInfoForCombineLocal, assistInfoForCombineGlobal_, epSendCntParams, copyPadParams);
    assistInfoQueue_.EnQue(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Communication()
{
    AssistInfoLocalCopy();
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.DeQue<int32_t>();
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>(); // 等assistInfoQueue_的GM->Local，后续标量读

    LocalTensor<uint8_t> sqInfoU8 = urmaSqInfoBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> templateSqeU8 = templateSqeBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> piCiU32 = piCiBuf_.Get<uint32_t>();

    for (uint32_t expertIdx = startExpertId_; expertIdx < endExpertId_; ++expertIdx) {
        // 计算当前处理的token块应发送到哪个卡的哪个专家
        uint32_t sendRankId = expertIdx % moeDistributeCombineSetupInfo_->epWorldSize;
        uint32_t sendExpertId = expertIdx / moeDistributeCombineSetupInfo_->epWorldSize; // 每卡逻辑专家序号

        uint32_t preCount = 0U;
        uint32_t assistInfoIdx = expertIdx; // 按epWorldSize*curRankExpertNum_分核等价于对assistInfo分核
        if (likely(assistInfoIdx > 0U)) {
            // 计算其他卡或专家已经发了多少token
            preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
        }
        // 当前要发送的token数量
        uint32_t curTokenNum = assistInfoForCombineLocal.GetValue(assistInfoIdx) - preCount;

        WriteWithNotifySQEInfoParams notifySqeInfo = {
            (uint64_t)expandXGM_ +
                (likely(curTokenNum > 0U) ? static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_ : 0U),
            (uint64_t)(GetWinAddrByRankId(sendRankId, sendExpertId) + epDataOffsetOnWin_),
            static_cast<uint32_t>(axisHExpandXTypeSize_) * max(curTokenNum, 1U),
            (uint64_t)(GetWinStateAddrByRankId(sendRankId) + epStateOffsetOnWin_ + sizeof(uint64_t)),
            0U,
            0U};

        if (unlikely(sendExpertId == curRankExpertNum_ - 1U)) {
            notifySqeInfo.notifyAddr -= sizeof(uint64_t);
            notifySqeInfo.notifyData = 0x3f800000;
            notifySqeInfo.cqe = 1U;
        }

        if (unlikely(sendRankId == moeDistributeCombineSetupInfo_->epRankId)) { // 本卡通信：仅搬运数据
            if (likely(curTokenNum > 0U)) {
                CurRankComm(notifySqeInfo);
            }
            continue;
        }

        // URMA 加载SQ CQ
        GetURMASqInfoTensor(sqInfoU8, (GM_ADDR)hcclContext_, sendRankId);

        // 获取PI CI
        GetPICI(piCiU32, (GM_ADDR)hcclContext_, moeDistributeCombineSetupInfo_->epRankId, sendRankId);
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t sqPi = piCiU32.GetValue(WIN_SQPI_OFFSET);

        // 根据当前处理卡号更新WQE模板
        UpdateCommWriteWithNotifySQE(templateSqeU8, sqInfoU8);

        // 组装WQE，并拷贝到对应sq队列的指定位置
        SetCommWriteWithNotifySQE(templateSqeU8, notifySqeInfo.dataSrcAddr, notifySqeInfo.dataDstAddr,
                                  notifySqeInfo.length, notifySqeInfo.notifyAddr, notifySqeInfo.notifyData,
                                  notifySqeInfo.cqe);
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等templateSqeU8标量写，后续Local->GM
        PutCommNotifySQE(sqInfoU8, templateSqeU8, sendExpertId, sqPi);
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>(); // 等sqInfoU8的Scalar读，防止被下一轮MTE2覆盖
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>(); // 等templateSqeU8的Local->GM，防止被下一轮覆盖
    }

    assistInfoQueue_.FreeTensor<int32_t>(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::CurRankComm(
    const WriteWithNotifySQEInfoParams &params)
{
    // expandX在GM上，winIn也属于GM，数据需要GM -> Local -> winIn
    GlobalTensor<uint8_t> selfDataSrcTensor;
    GlobalTensor<uint8_t> selfDataDstTensor;
    selfDataSrcTensor.SetGlobalBuffer((__gm__ uint8_t *)params.dataSrcAddr);
    selfDataDstTensor.SetGlobalBuffer((__gm__ uint8_t *)params.dataDstAddr);
    LocalTensor<uint8_t> expertTokenTmpU8 = expertTokenTmpQueue_.AllocTensor<uint8_t>();

    uint64_t sendBytes = static_cast<uint64_t>(params.length);
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(remain_ub_space), 0U, 0U, 0U};
    DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};

    uint64_t offset = 0U;
    for (; sendBytes > remain_ub_space; sendBytes -= remain_ub_space) {
        DataCopyPad(expertTokenTmpU8, selfDataSrcTensor[offset], copyParams, padParams);
        expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
        expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
        DataCopyPad(selfDataDstTensor[offset], expertTokenTmpU8, copyParams);
        offset += remain_ub_space;
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    copyParams.blockLen = static_cast<uint32_t>(sendBytes);
    DataCopyPad(expertTokenTmpU8, selfDataSrcTensor[offset], copyParams, padParams);
    expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
    expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
    DataCopyPad(selfDataDstTensor[offset], expertTokenTmpU8, copyParams);
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

    expertTokenTmpQueue_.FreeTensor<uint8_t>(expertTokenTmpU8);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        // URMA 通信相关Buffer初始化
        BuffInit();

        // URMA 生成WQE模板
        LocalTensor<uint8_t> templateSqeU8 = templateSqeBuf_.Get<uint8_t>();
        GenerateCommWriteWithNotifySQE(templateSqeU8);

        // Phase1: 组装SQE（按卡数*本卡专家数分核）
        Communication();

        // 全核同步，确保所有SQE组装完成
        SyncAll<true>();

        // Phase2: 按卡分核，敲doorbell
        CallJfsDoorbell();
    }
}
} // namespace MoeDistributeCombineSetupImpl

#endif // MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
