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
 * \file moe_distribute_dispatch.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_H
#define MOE_DISTRIBUTE_DISPATCH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_dispatch_tiling.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_tiling.h"
#endif

namespace MoeDistributeDispatchImpl {
constexpr uint8_t BUFFER_NUM = 2;            // 多buf
constexpr uint32_t STATE_OFFSET = 512;       // 状态空间偏移地址
constexpr uint32_t STATE_SIZE = 1024 * 1024; // 1M
constexpr uint32_t UB_ALIGN = 32;            // UB按32字节对齐
constexpr uint32_t STATE_ALIGN = 64;
constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint8_t COMM_NUM = 1; // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint32_t GATHER_NUM_PER_TIME = 6;
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;
constexpr uint64_t STATUS_DATA_OFFSET = 900 * 1024;

#define TemplateDispatchTypeClass \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateDispatchTypeFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
template <TemplateDispatchTypeClass>
class MoeDistributeDispatch {
public:
    __aicore__ inline MoeDistributeDispatch(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendToSharedExpert();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, int32_t totalCount);
    __aicore__ inline void CreateZeroTensor(LocalTensor<uint32_t> &outTensor);
    __aicore__ inline void ResetStatus();
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void UpdateMultiMoeTokenNumsOut();
    __aicore__ inline void UpdateTokenNumsOut();
    __aicore__ inline void InitBufferWait();
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return Mc2Kernel::GetBaseWindAddrByRankId(winContext_[COMM_EP_IDX], rankId, epRankId_) + winDataSizeOffsetEp_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_[COMM_EP_IDX], rankId, epRankId_) +
               dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    __aicore__ inline int32_t ReduceSumWorkNeedSize(int32_t calCnt)
    {
        int typeSize = sizeof(int32_t);
        int32_t elementsPerBlock = 32 / typeSize; // 32是1一个datablock的大小，此处计算一个datablock存放的元素
        int32_t iter1OutputCount = calCnt;
        int32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<ExpandXOutType> windowInQuantTensor_;
    GlobalTensor<int32_t> windowInstatusTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> receivestatusTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> dynamicScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;
    TBuf<> dynamicScalesBuf_;
    TBuf<> expertCountBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> getTotalBuf_;      // 计算totalCnt
    TBuf<> scalarBuf_;        // 辅助gather tensor定义
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> gatherTmpBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;                        // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                      // 量化使用，量化后的输出

    GM_ADDR expandXOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR expertTokenNumsOutGM_; // 这个输出没有使用
    GM_ADDR sendCountsOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;

    DataCopyExtParams dataCopyParamsFloat_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};               // aiv id
    uint32_t sharedExpertRankNum_{0}; // 共享专家卡数
    uint32_t moeExpertRankNum_{0};    // moe专家卡数，等于worldSize_ - 共享专家卡数
    uint32_t moeExpertNumPerRank_{0};
    uint32_t moeExpertNum_{0};
    uint32_t totalExpertNum_{0};
    uint32_t bufferSizePerRank_{0};
    uint32_t recvWinBlockNum_{0};
    uint32_t hSize_{0};
    uint32_t hOutSize_{0};
    uint32_t hCommuSize_{0};
    uint32_t scaleParamPad_{0};
    uint32_t axisHCommu_{0};
    uint32_t startExpertId_;
    uint32_t endExpertId_;
    uint32_t sendExpertNum_;
    uint32_t localCopyCoreNum_;
    uint32_t totalCnt_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint32_t recStatusNumPerCore_{0};
    int32_t reduceSumWorkNeedSize_{0};
    int32_t expertIdsCnt_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t windyquantOffset_;
    bool isShareExpertRank_ = false;
    bool isQuant_ = false;
    float sumTarget_;
    uint64_t totalWinSizeEp_{0};
    uint32_t expertTokenNumsType_{1};
    __gm__ Mc2Kernel::HcclOpParam *winContext_[COMM_NUM]{nullptr};
};

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    epRankId_ = tilingData->moeDistributeDispatchInfo.epRankId;
    winContext_[COMM_EP_IDX] = (__gm__ Mc2Kernel::HcclOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = Mc2Kernel::GetStatusDataSpaceGm(winContext_[COMM_EP_IDX]);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATUS_DATA_OFFSET));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[aivId_ * STATE_ALIGN]);
    dataState_ = selfDataStatusTensor(aivId_ * STATE_ALIGN);
    if (dataState_ == 0) {
        selfDataStatusTensor(aivId_ * STATE_ALIGN) = 1;
    } else {
        selfDataStatusTensor(aivId_ * STATE_ALIGN) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[aivId_ * STATE_ALIGN]);
    PipeBarrier<PIPE_ALL>();
    axisBS_ = tilingData->moeDistributeDispatchInfo.bs;
    axisH_ = tilingData->moeDistributeDispatchInfo.h;
    epWorldSize_ = tilingData->moeDistributeDispatchInfo.epWorldSize;
    axisMaxBS_ = tilingData->moeDistributeDispatchInfo.globalBs / epWorldSize_;
    moeExpertNum_ = tilingData->moeDistributeDispatchInfo.moeExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchInfo.sharedExpertRankNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchInfo.expertTokenNumsType;
    totalWinSizeEp_ = tilingData->moeDistributeDispatchInfo.totalWinSizeEp;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    expertPerSizeOnWin_ = axisMaxBS_ * axisH_ * sizeof(XType);
    winDataSizeOffsetEp_ = dataState_ * epWorldSize_ * expertPerSizeOnWin_ * moeExpertNumPerRank_;
    windowGM_ = GetWindAddrByRankId(epRankId_);
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankId_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(windowGM_), totalWinSizeEp_);
    OOMCheckAddrRange<int32_t>((__gm__ int32_t *)(statusSpaceGm_), STATE_SIZE);
#endif
    axisK_ = tilingData->moeDistributeDispatchInfo.k;
    aivNum_ = tilingData->moeDistributeDispatchInfo.aivNum;
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)expandXOut);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    windowInQuantTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)windowGM_);
    windowInstatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_));
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
    expandXOutGM_ = expandXOut;
    expandIdxOutGM_ = expandIdxOut;   // 无GlobalTensor
    sendCountsOutGM_ = sendCountsOut; // 无GlobalTensor
    isQuant_ = StaticQuant | DynamicQuant;
    hSize_ = axisH_ * sizeof(XType);
    hOutSize_ = axisH_ * sizeof(ExpandXOutType); // 如有量化，需要量化后通信
    scaleParamPad_ = (isQuant_ ? 128 : 0);       // 预留128B给量化参数，实际只使用了4B(fp32)
    hCommuSize_ = hOutSize_ + scaleParamPad_;
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    if (sharedExpertRankNum_ != 0) {                // 后面的卡才需要发给共享专家发数据
        sharedUsedAivNum_ = aivNum_ / (axisK_ + 1); // 均等分，取整
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    bufferSizePerRank_ = 32 * hSize_;
    expertIdsCnt_ = axisBS_ * axisK_;
    recvWinBlockNum_ = epWorldSize_ * moeExpertNumPerRank_;
    isShareExpertRank_ = (epRankId_ < sharedExpertRankNum_) ? true : false;
    windyquantOffset_ = epWorldSize_ * axisMaxBS_ * hOutSize_;
    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfStatusTensor[aivId_ * STATE_ALIGN]);
    int32_t state = selfStatusTensor(aivId_ * STATE_ALIGN);
    stateOffset_ = (recvWinBlockNum_ > 512) ? (STATE_OFFSET / 2) : STATE_OFFSET;
    tpipe_->InitBuffer(statusBuf_, Ceil(recvWinBlockNum_, 8) * 8 * UB_ALIGN); // Ceil(expertNum, 8) * 8 * 32B
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, recvWinBlockNum_ * 8); // 8 = UB_ALIGN / sizeof(int32_t)
    if (state == 0) {
        sumTarget_ = (float)1.0;
        // sumTarget_ = epWorldSize_ - (float)0.0;
        selfStatusTensor(aivId_ * STATE_ALIGN) = 0x3F800000;
        uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
        Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, Ceil(recvWinBlockNum_, 8), 1, 8); // 0x3F800000是float的1
    } else {
        sumTarget_ = 0.0;
        selfStatusTensor(aivId_ * STATE_ALIGN) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfStatusTensor[aivId_ * STATE_ALIGN]);
    if (isQuant_) {
        QuantInit(scales);
    } else {
        tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hCommuSize_); // 14k *2
        // 非量化模式下申请，量化模式下复用receiveDataCastFloatBuf_ 和 smoothScalesBuf_
        uint32_t expertIdxSize = expertIdsCnt_ * sizeof(int32_t);
        tpipe_->InitBuffer(dstExpBuf_, expertIdxSize); // BS * K * 4
        tpipe_->InitBuffer(subExpBuf_, expertIdxSize); // BS * K * 4
        dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
        subExpIdTensor_ = subExpBuf_.Get<int32_t>();
    }

    uint32_t expertIdsSize = axisBS_ * axisK_ * sizeof(int32_t); // 约束32对齐
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize);            // BS * K * 4
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    tpipe_->InitBuffer(expertCountBuf_, expertIdsSize); // BS * K * 4
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();

    // reduceSum计算所需的Tensor空间，取最大统一前面申请
    int32_t reduceSumCalMaxCnt = (epWorldSize_ > expertIdsCnt_) ? epWorldSize_ : expertIdsCnt_;
    reduceSumWorkNeedSize_ = ReduceSumWorkNeedSize(reduceSumCalMaxCnt);
    tpipe_->InitBuffer(workLocalBuf_, reduceSumWorkNeedSize_ * sizeof(int32_t));
    workLocalTensor_ = workLocalBuf_.Get<float>();

    dataCopyParamsFloat_ = {1U, sizeof(float), 0U, 0U, 0U};
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::QuantInit(GM_ADDR scales)
{
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_);       // 14K *2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_); // 7K *2
    scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    uint32_t hFp32Size = axisH_ * sizeof(float);
    if constexpr (DynamicQuant) {
        tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN); // 32B
    }
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, 1 * hFp32Size);  // 28KB
    tpipe_->InitBuffer(smoothScalesBuf_, axisH_ * sizeof(float)); // 28KB
    smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
    tpipe_->InitBuffer(dynamicScalesBuf_, axisBS_ * sizeof(float)); // 32 * 4
    dynamicScalesTensor_ = dynamicScalesBuf_.Get<float>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::SendToSharedExpert()
{
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(axisBS_, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= axisBS_) {
        return;
    }
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t temp = (epRankId_ * axisBS_) / sharedExpertRankNum_;
        uint32_t moeOnShareRank =
            Ceil((tokenIndex + 1 + temp) * sharedExpertRankNum_, axisBS_) - 1 - epRankId_; // 当前token发给哪个共享专家
        uint32_t preCnt = (moeOnShareRank + epRankId_) * axisBS_ / sharedExpertRankNum_ -
                          epRankId_ * axisBS_ / sharedExpertRankNum_; // 发给该共享专家已经有多少token数据
        GlobalTensor<ExpandXOutType> dstWinGMTensor;
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(GetWindAddrByRankId(moeOnShareRank) + expertPerSizeOnWin_ * epRankId_));
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(GetWindAddrByRankId(moeOnShareRank)),
                                          totalWinSizeEp_);
#endif
        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopy(xInTensor_, xGMTensor_[tokenIndex * axisH_], axisH_); // 约束对齐
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
            QuantProcess(0);
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
            if (isShareExpertRank_) {
                xOutFp32Tensor_ = xOutTensor_.template ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGMTensor_[tokenIndex], xOutFp32Tensor_[axisH_ / sizeof(float)],
                            dataCopyParamsFloat_);
                DataCopy(expandXOutGMTensor_[tokenIndex * axisH_], xOutTensor_, axisH_); // 约束对齐
            } else {
                DataCopy(dstWinGMTensor[(tokenIndex - preCnt) * axisHCommu_], xOutTensor_, axisHCommu_); // 约束对齐
            }
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopy(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], axisH_); // 约束对齐
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            if (isShareExpertRank_) {
                DataCopy(expandXOutGMTensor_[tokenIndex * axisHCommu_], xTmpTensor_, axisHCommu_);
            } else {
                DataCopy(dstWinGMTensor[(tokenIndex - preCnt) * axisHCommu_], xTmpTensor_, axisHCommu_); // 约束对齐
            }
            xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        }
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::CalTokenSendExpertCnt(uint32_t dstExpertId,
                                                                                              int32_t calCnt,
                                                                                              int32_t &curExpertCnt)
{
    if (isQuant_) { // 量化模式下buffer复用
        dstExpIdTensor_ = receiveDataCastFloatBuf_.Get<int32_t>();
        subExpIdTensor_ = smoothScalesBuf_.Get<int32_t>();
    }
    Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, calCnt);
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
    if (calCnt > curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - (aivNum_ - curUseAivNum);
    }
    startTokenId = sendTokenNum * newAivId; // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {     // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::SendToMoeExpert()
{
    uint32_t startTokenId = 0;
    uint32_t endTokenId = 0;
    uint32_t sendTokenNum = 0;
    SplitToCore(expertIdsCnt_, moeUsedAivNum_, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= expertIdsCnt_) {
        return;
    }
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    for (int32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t dstExpertId = expertIdsTensor_(tokenIndex);
        int32_t curExpertCnt = 0;
        if (tokenIndex > 0) {
            CalTokenSendExpertCnt(dstExpertId, tokenIndex, curExpertCnt);
        }
        expertCountTensor_(tokenIndex - startTokenId) = curExpertCnt;
        uint32_t tempRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(tempRankId) +
                                            (expertPerSizeOnWin_ *
                                             (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                            hCommuSize_ * curExpertCnt); // 计算地址偏移
        dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType *)rankGM);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(GetWindAddrByRankId(tempRankId)), totalWinSizeEp_);
#endif
        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopy(xInTensor_, xGMTensor_[tokenIndex / axisK_ * axisH_], axisH_); // 约束对齐
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
            uint32_t expertIndex = sharedExpertRankNum_ != 0 ? (dstExpertId + 1) : dstExpertId;
            QuantProcess(expertIndex);
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
            DataCopy(dstWinGMTensor, xOutTensor_, axisHCommu_); // 约束对齐
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopy(xTmpTensor_, xGMTensor_[tokenIndex / axisK_ * axisH_], axisH_); // 约束对齐
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            DataCopy(dstWinGMTensor, xTmpTensor_, axisHCommu_); // 约束对齐
            xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        }
    }
    GlobalTensor<int32_t> expandIdxGMTensor;
    expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOutGM_ + startTokenId * sizeof(int32_t)));
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPad(expandIdxGMTensor, expertCountTensor_, expertIdsCntParams);
}

/*
共享专家卡：所有核用于给moe专家发送数据
moe专家卡：部分核用于给共享专家发送数据，部分核用于给moe专家发送数据
*/
template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::AlltoAllDispatch()
{
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, copyPadParams);
    // expertIdsTensor_(tokenIndex)操作需要等前面的DataCopy MTE2
    AscendC::TQueSync<PIPE_MTE2, PIPE_S> expertCntLocalSync;
    expertCntLocalSync.SetFlag(0);
    expertCntLocalSync.WaitFlag(0);
    // statusTensor_取值需要等待Init中的Duplicate
    SyncFunc<AscendC::HardEvent::V_S>();

    if (!isShareExpertRank_) {
        for (uint32_t curStatusExpId = 0; curStatusExpId < sharedExpertRankNum_; ++curStatusExpId) {
            int32_t curExpertCnt = (curStatusExpId + 1 + epRankId_) * axisBS_ / sharedExpertRankNum_ -
                                   (curStatusExpId + epRankId_) * axisBS_ / sharedExpertRankNum_;
            statusTensor_((curStatusExpId) * 8 + 1) = curExpertCnt;
        }
    }
    if ((sharedExpertRankNum_ != 0) && (aivId_ >= moeUsedAivNum_)) { // 后面的核进行发给共享专家
        SendToSharedExpert();
        return;
    }
    SendToMoeExpert();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::SetStatus()
{
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    SplitToCore(totalExpertNum_, aivNum_, startExpertId_, endExpertId_, sendExpertNum_);
    for (uint32_t curExpertId = startExpertId_; curExpertId < endExpertId_; ++curExpertId) {
        if (curExpertId < sharedExpertRankNum_) {
            continue;
        }
        int32_t curExpertCnt = 0;
        int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
        CalTokenSendExpertCnt(curMoeExpertId, expertIdsCnt_, curExpertCnt);
        int32_t cntPosIndex = curExpertId * 8 + 1; // 8的含义为一个专家占32字节
        statusTensor_(cntPosIndex) = curExpertCnt;
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    if (startExpertId_ >= totalExpertNum_) {
        return;
    }
    GlobalTensor<int32_t> rankGMTensor;
    uint32_t offset = stateOffset_ * epRankId_;
    for (uint32_t rankIndex = startExpertId_; rankIndex < endExpertId_; ++rankIndex) {
        uint32_t dstRankId = rankIndex;
        if (moeExpertNumPerRank_ > 1 && (rankIndex >= sharedExpertRankNum_)) {
            dstRankId = ((rankIndex - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset =
                (epRankId_ + (rankIndex - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_) * stateOffset_;
        }
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset); // 计算地址偏移
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        OOMCheckAddrRange<int32_t>((__gm__ int32_t *)(GetWindStateAddrByRankId(dstRankId)), STATE_SIZE);
#endif
        DataCopy<int32_t>(rankGMTensor, statusTensor_[rankIndex * 8], 8UL); // 8时数据大小，按32对齐拷贝
    }
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::QuantProcess(uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();

    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    xInQueue_.FreeTensor<XType>(xInTensor_);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        if constexpr (DynamicQuant) {
            SyncFunc<AscendC::HardEvent::V_MTE2>(); // ub复用，循环同步
        }
        DataCopy(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], axisH_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        rowMaxTensor_ = rowMaxBuf_.Get<float>();

        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMax(rowMaxTensor_, floatLocalAbsTemp, floatLocalAbsTemp, axisH_, false);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / rowMaxTensor_.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(axisH_ / sizeof(float), float(1.0) / dynamicScale); // int8->float32
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::WaitDispatch()
{
    uint32_t startStatusIndex = 0;
    uint32_t endStatusIndex = 0;
    uint32_t rscvStatusNum = isShareExpertRank_ ? epWorldSize_ : recvWinBlockNum_;
    SplitToCore(rscvStatusNum, aivNum_, startStatusIndex, endStatusIndex, recStatusNumPerCore_);
    InitBufferWait();
    if (startStatusIndex >= rscvStatusNum) {
        SyncAll<true>();
        return;
    }

    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherTmpTensor.SetValue(0, 1);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
    uint32_t mask = 1; // gatherMask + sum 相关参数
    uint64_t rsvdCnt = 0;
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    float sumOfFlag = static_cast<float>(-1.0);
    float minTarget = (sumTarget_ * recStatusNumPerCore_) - (float)0.5;
    float maxTarget = (sumTarget_ * recStatusNumPerCore_) + (float)0.5;
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1,
                               static_cast<uint16_t>((recvWinBlockNum_ > 512) ? 7 : 15), 0}; // srcStride为15个block
    SyncFunc<AscendC::HardEvent::S_V>();
    while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
                   {1, (uint16_t)recStatusNumPerCore_, 1, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    SyncAll<true>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::GetCumSum(LocalTensor<int32_t> &outLocal,
                                                                                  int32_t totalCount)
{
    DataCopyParams intriParams{static_cast<uint16_t>(recvWinBlockNum_), 1,
                               static_cast<uint16_t>((recvWinBlockNum_ > 512) ? 7 : 15), 0}; // srcStride为15个block
    DataCopy(statusTensor_, windowInstatusTensor_, intriParams);

    if (isShareExpertRank_) {
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t curStatusExpId = 0; curStatusExpId < sharedExpertRankNum_; ++curStatusExpId) {
            int32_t curExpertCnt = (curStatusExpId + 1 + epRankId_) * axisBS_ / sharedExpertRankNum_ -
                                   (curStatusExpId + epRankId_) * axisBS_ / sharedExpertRankNum_;
            statusTensor_((curStatusExpId) * 8 + 1) = curExpertCnt;
        }
        SyncFunc<AscendC::HardEvent::S_V>();
    } else {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    outLocal = gatherMaskOutBuf_.Get<int32_t>(); // 内存复用
    LocalTensor<float> getTotalLocal = getTotalBuf_.Get<float>();
    // gather mask在一起

    LocalTensor<uint32_t> gatherTmpTensor = gatherTmpBuf_.Get<uint32_t>();
    Duplicate(gatherTmpTensor, (uint32_t)33686018,
              Ceil(recvWinBlockNum_, 4)); // 0000 0010 0000 0010 0000 0010 0000 0010
    PipeBarrier<PIPE_V>();
    uint32_t mask = recvWinBlockNum_ * 8; // 512 / 32
    uint64_t rsvdCnt = 0;
    GatherMask(outLocal, statusTensor_, gatherTmpTensor, true, mask, {1, 1, 0, 0}, rsvdCnt);
    LocalTensor<float> tmpFp32 = outLocal.ReinterpretCast<float>();
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(getTotalLocal, tmpFp32, workLocalTensor_, epWorldSize_);
    totalCnt_ = getTotalLocal.ReinterpretCast<int32_t>().GetValue(0);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpFp32, tmpFp32, workLocalTensor_, totalCount);
    PipeBarrier<PIPE_V>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::CreateZeroTensor(
    LocalTensor<uint32_t> &outLocal)
{
    TBuf<> outBuf;
    tpipe_->InitBuffer(outBuf, UB_ALIGN);
    outLocal = outBuf.Get<uint32_t>();
    for (uint32_t i = 0; i < 2; i++) {
        outLocal.SetValue(i, 0);
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::LocalWindowCopy()
{
    uint32_t totalMoeExpert = 0;
    LocalTensor<int32_t> outCountLocal;
    if (isShareExpertRank_) {
        totalMoeExpert = epWorldSize_;
    } else {
        // 处理MOE专家，获取本卡状态空间中的索引及token数目
        totalMoeExpert = epWorldSize_ * moeExpertNumPerRank_;
    }
    sendExpertNum_ = totalMoeExpert / aivNum_; // 每个aiv需要处理的专家数
    uint32_t remainderRankNum = totalMoeExpert % aivNum_;
    startExpertId_ = sendExpertNum_ * aivId_; // + sharedExpertRankNum_, 每个aiv发送的起始rankid
    if (aivId_ < remainderRankNum) {          // 前remainderRankNum个aiv需要多发1个卡的数据
        sendExpertNum_ += 1;
        startExpertId_ += aivId_;
    } else {
        startExpertId_ += remainderRankNum;
    }
    endExpertId_ = startExpertId_ + sendExpertNum_;
    if (startExpertId_ >= totalMoeExpert) { // 多余的核return
        return;
    }
    GetCumSum(outCountLocal, startExpertId_ + 1);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t index = 0;
    uint32_t beginIdx = 0;
    for (uint32_t index = startExpertId_; index < endExpertId_; index++) {
        uint32_t i = index - startExpertId_;
        if (i > 0) {
            outCountLocal.SetValue(i, outCountLocal.GetValue(i - 1) + outCountLocal.GetValue(index));
        }
        uint32_t count = statusTensor_.GetValue(index * 8 + 1);
        beginIdx = outCountLocal.GetValue(i) - count;
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        if (isShareExpertRank_) {
            if (index < sharedExpertRankNum_) { // 共享专家前面排布的是本卡数据，只需要统计epRecvCnt，不需要去搬出
                beginIdx += count;
                continue;
            }
        }
        uint32_t winOffset = index;
        if (!isShareExpertRank_) {
            if (moeExpertNumPerRank_ > 1) {
                winOffset =
                    index % epWorldSize_ * moeExpertNumPerRank_ + index / epWorldSize_; // 转换成数据区的排布偏移
            }
        }
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + winOffset * expertPerSizeOnWin_;
        GlobalTensor<ExpandXOutType> tokGlobal;
        GlobalTensor<ExpandXOutType> expandXOutGlobal;
#if !(defined(ASCENDC_OOM) && ASCENDC_OOM == 1)
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        for (uint32_t j = 0; j < count; j++) {
            tokGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr + j * hCommuSize_));
            // 将数据从Window拷贝到UB
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopy(xTmpTensor_, tokGlobal, axisHCommu_);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            if constexpr (DynamicQuant || StaticQuant) {
                xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGMTensor_[beginIdx + j], xOutFp32Tensor_[axisH_ / sizeof(float)],
                            dataCopyParamsFloat_);
            }
            expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (beginIdx + j) * axisH_,
                                             axisH_);
            DataCopy(expandXOutGlobal, xTmpTensor_, axisH_);
            xQueue_.FreeTensor(xTmpTensor_);
        }
        beginIdx += count;
    }
    totalCnt_ = beginIdx;
    lastCore_ = MIN(totalMoeExpert, aivNum_) - 1;

    DataCopyExtParams dataCopyOutParams = {1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    GlobalTensor<int32_t> sendCountsGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    DataCopyPad(sendCountsGlobal[startExpertId_], outCountLocal, dataCopyOutParams);
    PipeBarrier<PIPE_MTE3>();
}

// 更新多专家卡上的tokenNumsOut tensor
template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::UpdateMultiMoeTokenNumsOut()
{
    uint32_t tokenSums = 0;
    GlobalTensor<int32_t> sendCountsGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    for (uint32_t localMoeIndex = 0; localMoeIndex < moeExpertNumPerRank_; ++localMoeIndex) {
        if (localMoeIndex == 0) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                sendCountsGlobal[epWorldSize_ - 1]);
            uint32_t firstMoeCnt = sendCountsGlobal.GetValue(epWorldSize_ - 1);
            tokenSums = firstMoeCnt;
            expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenSums);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                expertTokenNumsOutGMTensor_[localMoeIndex]);
        } else {
            uint32_t preIndex = epWorldSize_ * (localMoeIndex - 1) + epWorldSize_ - 1;
            uint32_t curIndex = epWorldSize_ * localMoeIndex + epWorldSize_ - 1;
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                sendCountsGlobal[preIndex]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                sendCountsGlobal[curIndex]);
            uint32_t preMoeIndexCnt = sendCountsGlobal.GetValue(preIndex);
            uint32_t curMoeIndexCnt = sendCountsGlobal.GetValue(curIndex);
            tokenSums = ((expertTokenNumsType_ == 0) ? tokenSums : 0) + (curMoeIndexCnt - preMoeIndexCnt);
            expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenSums);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                expertTokenNumsOutGMTensor_[localMoeIndex]);
        }
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::InitBufferWait()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(statusBuf_, recvWinBlockNum_ * UB_ALIGN); // expertNum * 32B
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hCommuSize_);
    tpipe_->InitBuffer(workLocalBuf_, reduceSumWorkNeedSize_ * sizeof(int32_t));
    // 内存复用，取大
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = recvWinBlockNum_ * sizeof(float);
    uint64_t gatherMaskOutSize =
        (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);
    // worldsize * 单卡moe专家数 * 4B
    tpipe_->InitBuffer(getTotalBuf_, epWorldSize_ * moeExpertNumPerRank_ * sizeof(int32_t));
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2); // 这里的2 gatherTmpTensor使用UB_ALIGN，statusSumOutTensor使用UB_ALIGN
    tpipe_->InitBuffer(gatherTmpBuf_, Ceil(recvWinBlockNum_, 4) * 4); // 4B对齐防止epWorldSize为2时空间不足
    statusTensor_ = statusBuf_.Get<int32_t>();
    workLocalTensor_ = workLocalBuf_.Get<float>();
}

// 更新tokenNumsOut tensor
template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::UpdateTokenNumsOut()
{
    // 最后一个核做更新，Moe专家只有最后一个核有计算出所有 sendCountsGlobal
    if (!isShareExpertRank_ && moeExpertNumPerRank_ > 1) {
        SyncAll<true>();
        if (aivId_ != lastCore_)
            return;

        SyncFunc<AscendC::HardEvent::MTE3_S>();
        UpdateMultiMoeTokenNumsOut();
    } else {
        if (aivId_ != lastCore_)
            return;

        uint32_t tokenNum = 0;
        // Moe专家token总数在Cumsum内计算得出
        tokenNum = totalCnt_;
        expertTokenNumsOutGMTensor_.SetValue(0, tokenNum);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertTokenNumsOutGMTensor_);
    }
}

// 流水
//          24 win->ub ub->gm
// 48 alltoall                                 syncAll 48 AllgatherOut
//                              1 setStatus
//          24 win->ub ub->win
//                              1 waitStatus

template <TemplateDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatch<TemplateDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        AlltoAllDispatch();
        SetStatus();
        WaitDispatch();
        LocalWindowCopy();
        UpdateTokenNumsOut();
    }
}

} // namespace MoeDistributeDispatchImpl
#endif // MOE_DISTRIBUTE_DISPATCH_H
