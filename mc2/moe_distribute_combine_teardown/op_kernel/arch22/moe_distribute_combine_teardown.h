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
 * \file moe_distribute_combine_teardown.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_TEARDOWN_H
#define MOE_DISTRIBUTE_COMBINE_TEARDOWN_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../moe_distribute_combine_setup/moe_distribute_combine_setup_base.h")
#include "../../moe_distribute_combine_setup/moe_distribute_combine_setup_base.h"
#else
#include "../../../moe_distribute_combine_setup/op_kernel/moe_distribute_combine_setup_base.h"
#endif
#include "moe_distribute_combine_teardown_tiling.h"

namespace MoeDistributeCombineTeardownImpl {

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeCombineTeardown {
    constexpr static uint8_t BUFFER_NUM = 2;             // 多buf
    constexpr static uint32_t STATE_OFFSET = 512U;       // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024U * 1024; // 1M
    constexpr static uint32_t UB_ALIGN = 32U;            // UB按32字节对齐
    constexpr static uint64_t WIN_STATE_OFFSET = 350UL * 1024;
    constexpr static uint64_t STATE_WIN_OFFSET = 950UL * 1024;
    constexpr static uint32_t COMBINE_STATE_OFFSET = 0U; // 本卡状态空间偏移地址，前面的地址给dispatch用

public:
    __aicore__ inline MoeDistributeCombineTeardown(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR quantExpandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                GM_ADDR expertScales, GM_ADDR commCmdInfo, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
                                GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineTeardownTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitStatusTargetSum();
    __aicore__ inline void AlltoAllBuffInit();
    __aicore__ inline void ReduceScatterTrans();
    __aicore__ inline void SetWaitTpStatusAndDisPatch();
    __aicore__ inline void CustomAdd(LocalTensor<ExpandXType> &dst, LocalTensor<ExpandXType> &src0,
                                     LocalTensor<ExpandXType> &src1, uint32_t dataCnt);
    __aicore__ inline void ExpertAlltoAllDispatchInnerCopyAdd(uint32_t tokenNumLoop, uint32_t srcStartTokenIdx,
                                                              uint32_t ep, uint32_t expertIdx);
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)((epRankId_ == rankId) ?
                             epWinContext_->localWindowsIn :
                             ((HcclRankRelationResV2 *)(epWinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) +
               winDataSizeOffset_ + expertLocalId * expertPerSizeOnWin_;
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)((epRankId_ == rankId) ?
                             epWinContext_->localWindowsExp :
                             ((HcclRankRelationResV2 *)(epWinContext_->remoteRes[rankId].nextDevicePtr))->windowsExp) +
               COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<ExpandIdxType> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<float> expandScalesGM_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<float> epStatusSpaceGlobalTensor_; // win区状态位置拷入相关参数
    GlobalTensor<ExpandXType> rowTmpGlobal_;
    GM_ADDR workspaceGM_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGm_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t coreIdx_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0}; // 共享专家卡数
    uint32_t moeExpertNum_{0};        // moe专家数
    uint32_t moeExpertPerRankNum_{0}; // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};          // moeExpertPerRankNum_ * epWorldSize_
    __gm__ HcclOpResParam *epWinContext_{nullptr};
    uint32_t epDataOffsetOnWin_{0};
    uint32_t epStateOffsetOnWin_{0};
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t bsKNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t totalWinSize_{0};
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TBuf<> expertIdsBuf_;
    TBuf<> expandScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> tokenBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> gatherTmpBuf_;
    TBuf<> statusSumOutBuf_;
    float sumTarget_{0.0};
    bool isShardExpert_{false};
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR quantExpandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR expertScales,
    GM_ADDR commCmdInfo, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeCombineTeardownTilingData *tilingData)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    epRankId_ = tilingData->moeDistributeCombineTeardownInfo.epRankId;
    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpResParam *)contextGM;
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)epWinContext_->localWindowsExp;
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET + coreIdx_ * 512));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0);
    if (dataState_ == 0U) {
        selfDataStatusTensor(0) = 1U;
    } else {
        selfDataStatusTensor(0) = 0U;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    pipe_barrier(PIPE_ALL);

    workspaceGM_ = workspaceGM;
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    expandScalesGM_.SetGlobalBuffer((__gm__ float *)expertScales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    axisBS_ = tilingData->moeDistributeCombineTeardownInfo.bs;
    axisH_ = tilingData->moeDistributeCombineTeardownInfo.h;
    axisK_ = tilingData->moeDistributeCombineTeardownInfo.k;
    sharedExpertNum_ = tilingData->moeDistributeCombineTeardownInfo.sharedExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeCombineTeardownInfo.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeCombineTeardownInfo.moeExpertNum;
    epWorldSize_ = tilingData->moeDistributeCombineTeardownInfo.epWorldSize;
    axisMaxBS_ = tilingData->moeDistributeCombineTeardownInfo.globalBs / epWorldSize_;
    aivNum_ = tilingData->moeDistributeCombineTeardownInfo.aivNum;
    moeExpertPerRankNum_ = 1U;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    totalWinSize_ = tilingData->moeDistributeCombineTeardownInfo.totalWinSize;
    stateOffset_ = (moeSendNum_ > 512U) ? static_cast<uint32_t>(STATE_OFFSET / 2) : static_cast<uint32_t>(STATE_OFFSET);
    expertPerSizeOnWin_ =
        static_cast<uint64_t>(axisMaxBS_) * static_cast<uint64_t>(axisH_) * static_cast<uint64_t>(sizeof(ExpandXType));
    winDataSizeOffset_ =
        static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineTeardownInfo.totalWinSize / 2UL);
    epWindowGM_ = GetWinAddrByRankId(epRankId_);
    epStatusSpaceGm_ = GetWinStateAddrByRankId(epRankId_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXType>((__gm__ ExpandXType *)(epWindowGM_), totalWinSize_);
    OOMCheckAddrRange<float>((__gm__ float *)(epStatusSpaceGm_), STATE_SIZE);
#endif
    epStatusSpaceGlobalTensor_.SetGlobalBuffer((__gm__ float *)epStatusSpaceGm_);
    epDataOffsetOnWin_ = epRankId_ * moeExpertPerRankNum_ * static_cast<uint32_t>(expertPerSizeOnWin_);
    epStateOffsetOnWin_ = epRankId_ * stateOffset_;
    isShardExpert_ = (epRankId_ < sharedExpertRankNum_);
    axisHFloatSize_ = axisH_ * static_cast<uint32_t>(sizeof(float));
    axisHExpandXTypeSize_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));
    bsKNum_ = axisBS_ * axisK_;

    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, axisHExpandXTypeSize_); // 7168 * 2 * 2 = 28672
    sumTarget_ = static_cast<float>(1.0);
    SplitCoreCal();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    sendRankNum_ = epWorldSize_ / aivNum_;
    uint32_t remainderRankNum = epWorldSize_ % aivNum_;
    startRankId_ = sendRankNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        sendRankNum_++;
        startRankId_ += coreIdx_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::AlltoAllBuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(statusBuf_, sendRankNum_ * UB_ALIGN);                 // 288 * 32 = 9216
    tpipe_->InitBuffer(expertIdsBuf_, axisBS_ * axisK_ * sizeof(int32_t));   // 32 * 8 * 4 = 1024
    tpipe_->InitBuffer(expandScalesBuf_, axisBS_ * axisK_ * sizeof(float));  // 32 * 8 * 4 = 1024
    tpipe_->InitBuffer(tokenBuf_, axisH_ * sizeof(ExpandXType));             // 7168 * 2 = 14336
    tpipe_->InitBuffer(rowTmpFloatBuf_, axisHFloatSize_);                    // 7168 * 4 = 28672
    tpipe_->InitBuffer(mulBuf_, axisHFloatSize_);                            // 7168 * 4 = 28672
    tpipe_->InitBuffer(sumFloatBuf_, axisHFloatSize_);                       // 7168 * 4 = 28672
    tpipe_->InitBuffer(indexCountsBuf_, axisBS_ * axisK_ * sizeof(int32_t)); // 32 * 8 * 4 = 1024
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, axisHExpandXTypeSize_);     // 7168 * 2 * 2 = 28672
    tpipe_->InitBuffer(gatherMaskOutBuf_, epWorldSize_ * sizeof(float));     // 288 * 4 = 1152
    tpipe_->InitBuffer(gatherTmpBuf_, sizeof(uint32_t));                     // 4
    tpipe_->InitBuffer(statusSumOutBuf_, sizeof(float));                     // 4
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::WaitDispatch()
{
    if (startRankId_ >= epWorldSize_) {
        SyncAll<true>();
        return;
    }
    LocalTensor<float> statusTensor = statusBuf_.Get<float>();
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = gatherTmpBuf_.Get<uint32_t>();
    LocalTensor<float> statusSumOutTensor = statusSumOutBuf_.Get<float>();

    gatherTmpTensor.SetValue(0, 1);
    uint32_t mask = 1; // gatherMask + sum 相关参数
    uint64_t rsvdCnt = 0;
    DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
                               static_cast<uint16_t>((moeSendNum_ > 512) ? 7 : 15), 0}; // srcStride为15个block
    DataCopyParams clearParams{static_cast<uint16_t>(sendRankNum_), 1, 0,
                               static_cast<uint16_t>((moeSendNum_ > 512) ? 7 : 15)}; // srcStride为15个block
    float sumOfFlag = static_cast<float>(-1.0);
    float minTarget = (sumTarget_ * sendRankNum_) - static_cast<float>(0.5);
    float maxTarget = (sumTarget_ * sendRankNum_) + static_cast<float>(0.5);
    SumParams sumParams{1, sendRankNum_, sendRankNum_};
    SyncFunc<AscendC::HardEvent::S_V>();
    while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
        DataCopy<float>(statusTensor, epStatusSpaceGlobalTensor_[startRankId_ * stateOffset_ / sizeof(float)],
                        intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(gatherMaskOutTensor, statusTensor, gatherTmpTensor, true, mask, {1, (uint16_t)sendRankNum_, 1, 0},
                   rsvdCnt);
        PipeBarrier<PIPE_V>();
        Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(statusTensor, 0, sendRankNum_ * 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy<float>(epStatusSpaceGlobalTensor_[startRankId_ * stateOffset_ / sizeof(float)], statusTensor, clearParams);
    pipe_barrier(PIPE_ALL);
    SyncAll<true>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::LocalWindowCopy()
{
    uint32_t beginIndex = 0;
    uint32_t endIndex = 0;
    uint32_t processLen = 0;
    uint32_t tokenOffset = 0;

    uint32_t tokenPerAivNum = axisBS_ / aivNum_;
    uint32_t remainderToken = axisBS_ % aivNum_;
    beginIndex = tokenPerAivNum * coreIdx_;
    if (coreIdx_ < remainderToken) {
        tokenPerAivNum++;
        beginIndex = tokenPerAivNum * coreIdx_;
    } else {
        beginIndex += remainderToken;
    }
    endIndex = beginIndex + tokenPerAivNum;
    processLen = axisH_;

    LocalTensor<ExpandIdxType> expertIdsLocal = expertIdsBuf_.Get<ExpandIdxType>();
    LocalTensor<float> expandScalesLocal = expandScalesBuf_.Get<float>();

    LocalTensor<float> rowTmpFloatLocal = rowTmpFloatBuf_.Get<float>();
    LocalTensor<float> mulBufLocal = mulBuf_.Get<float>();
    LocalTensor<float> sumFloatBufLocal = sumFloatBuf_.Get<float>();

    LocalTensor<ExpandIdxType> indexCountsLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams = {1U, static_cast<uint32_t>(bsKNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};

    DataCopyPad(indexCountsLocal, expandIdxGM_, bskParams, copyPadParams);
    DataCopyPad(expertIdsLocal, expertIdsGM_, bskParams, copyPadParams);
    DataCopyPad(expandScalesLocal, expandScalesGM_, bskParams, copyPadFloatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIndex = beginIndex; tokenIndex < endIndex; tokenIndex++) {
        uint32_t index = tokenIndex * axisK_;
        int32_t moeExpert = 0;
        float scaleVal = 0.0;
        GM_ADDR wAddr;
        SyncFunc<AscendC::HardEvent::MTE3_V>(); // 与结果搬出datacopy同tensor
        Duplicate(sumFloatBufLocal, (float)0, axisH_);
        LocalTensor<ExpandXType> tmpUb;
        for (uint32_t i = 0; i < axisK_; i++) {
            moeExpert = expertIdsLocal.GetValue(index);
            scaleVal = expandScalesLocal.GetValue(index);
            wAddr = (__gm__ uint8_t *)(epWindowGM_) +
                    expertPerSizeOnWin_ * moeExpertPerRankNum_ * sharedExpertRankNum_ +
                    expertPerSizeOnWin_ * moeExpert + indexCountsLocal.GetValue(index) * axisHExpandXTypeSize_ +
                    tokenOffset * sizeof(ExpandXType);
            rowTmpGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)wAddr);
            tmpUb = moeSumQueue_.AllocTensor<ExpandXType>();
            DataCopy(tmpUb, rowTmpGlobal_, processLen);
            moeSumQueue_.EnQue(tmpUb);
            tmpUb = moeSumQueue_.DeQue<ExpandXType>();
            Cast(rowTmpFloatLocal, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
            pipe_barrier(PIPE_V);
            AscendC::Muls(mulBufLocal, rowTmpFloatLocal, scaleVal, processLen);
            pipe_barrier(PIPE_V);
            AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, mulBufLocal, processLen);
            index++;
            moeSumQueue_.FreeTensor<ExpandXType>(tmpUb);
        }
        LocalTensor<ExpandXType> rowTmpLocal = tokenBuf_.Get<ExpandXType>();
        if (sharedExpertRankNum_ > 0U) {
            // 累加共享专家，当前bs范围，一个核处理一个token，根据当前tokenId反推对应的共享专家
            uint32_t temp = (epRankId_ * axisBS_) / sharedExpertRankNum_;
            uint32_t moeOnShareRank = Ceil((tokenIndex + 1U + temp) * sharedExpertRankNum_, axisBS_) - 1U - epRankId_;
            uint32_t preCnt = (moeOnShareRank + epRankId_) * axisBS_ / sharedExpertRankNum_ -
                              epRankId_ * axisBS_ / sharedExpertRankNum_;
            __gm__ ExpandXType *shareAddr =
                (__gm__ ExpandXType *)(epWindowGM_ + moeOnShareRank * expertPerSizeOnWin_ * moeExpertPerRankNum_) +
                (tokenIndex - preCnt) * axisH_ + tokenOffset;
            GlobalTensor<ExpandXType> shareTokGlobal;
            shareTokGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(shareAddr));
            SyncFunc<AscendC::HardEvent::V_MTE2>(); // 与结果搬出Cast同地址
            DataCopy(rowTmpLocal, shareTokGlobal, processLen);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(rowTmpFloatLocal, rowTmpLocal, AscendC::RoundMode::CAST_NONE, processLen);
            pipe_barrier(PIPE_V);
            AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, rowTmpFloatLocal, processLen);
        }
        // 结果搬出
        pipe_barrier(PIPE_V);
        LocalTensor<ExpandXType> sumBufLocal = tokenBuf_.Get<ExpandXType>();
        Cast(sumBufLocal, sumFloatBufLocal, AscendC::RoundMode::CAST_RINT, processLen);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[tokenIndex * axisH_ + tokenOffset], sumBufLocal, processLen);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineTeardown<TemplateMC2TypeFunc>::Process()
{
    AlltoAllBuffInit();
    WaitDispatch();
    LocalWindowCopy();
}
} // namespace MoeDistributeCombineTeardownImpl
#endif // MOE_DISTRIBUTE_COMBINE_TEARDOWN_H
