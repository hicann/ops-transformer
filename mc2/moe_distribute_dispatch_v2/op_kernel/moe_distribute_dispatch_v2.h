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
 * \file moe_distribute_dispatch_v2.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_H
#define MOE_DISTRIBUTE_DISPATCH_V2_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "adv_api/math/cumsum.h"
#include "adv_api/reduce/reduce.h"
#endif
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_dispatch_v2_quant.h"
#include "moe_distribute_elastic.h"

#include "check_winsize.h"
#if __has_include("../common/mc2_moe_context.h")
#include "../common/mc2_moe_context.h"
#include "../common/moe_distribute_base.h"
#else
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#endif
#include "hccl_context_holder.h"
#include "mc2_moe_context_holder.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace Mc2Kernel {
#define TemplateDispatchV2TypeClass \
    typename ContextHolder, typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define TemplateDispatchV2TypeFunc ContextHolder, XType, ExpandXOutType, QuantMode, IsSmoothScaleExist

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace Mc2Aclnn;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
constexpr CumSumConfig SET_EXPERT_TOKEN_NUMS_CUMSUM_CONFIG{true, false, false,
                                                           CumSumAlgorithm::CUMSUM_ALGORITHM_LINEBYLINE};
#endif
template <TemplateDispatchV2TypeClass>
class MoeDistributeDispatchV2 {
public:
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    using XInType = typename std::conditional<
        (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value), uint8_t, XType>::type;
    using XOutType = typename std::conditional<(Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) ||
                                                   (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
                                               uint8_t, ExpandXOutType>::type;
#else
    using XInType = XType;
    using XOutType = ExpandXOutType;
#endif
    __aicore__ inline MoeDistributeDispatchV2(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                GM_ADDR expertScales, GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expandScalesOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessToken(GlobalTensor<XOutType> &outTokenGT, uint32_t tokenIndex, uint32_t topKIndex,
                                        DataCopyPadParams &padParams, DataCopyParams &scaleInParams,
                                        uint32_t expertIndex);
    __aicore__ inline void SendToSharedExpert();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void LocalWindowCopy();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void SetExpertTokenNumsA5();
    __aicore__ inline void SetExpertPrefixSumA5(uint32_t localExpertNum, bool useCumSum);
#else
    __aicore__ inline void SetExpertTokenNums();
#endif
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void CalValidBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalValidExpIdx(LocalTensor<bool> maskInputTensor);
    __aicore__ inline void GenerateGatherMaskTensor(uint32_t maskCnt);
    __aicore__ inline void MaskZeroComputeExpert(uint32_t maskCnt);
    __aicore__ inline void ZeroComputeExpertMaskCal();
    __aicore__ inline void SetStatus();
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t totalCount);
    __aicore__ inline void GetCumSumA5(LocalTensor<int32_t> &outLocal);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void FillTriple(LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void SyncCntOnCore(LocalTensor<float> &gatherMaskOutTensor,
                                         LocalTensor<uint32_t> &gatherTmpTensor,
                                         LocalTensor<float> &statusSumOutTensor);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void WaitStatusA5(LocalTensor<float> &statusSumOutTensor);
#else
    __aicore__ inline void WaitStatus(LocalTensor<float> &gatherMaskOutTensor, LocalTensor<float> &statusSumOutTensor);
#endif
    __aicore__ inline void PerformanceInfoPerRank(uint64_t performanceTimeStart, bool isCompactStatus);
    __aicore__ inline void PerformanceInfoCopyOut();
    __aicore__ inline void RunPosRecord();
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindAddrByRankId(rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindStateAddrByRankId(rankId, epRankIdOriginal_) + dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    __aicore__ inline int32_t ReduceSumWorkNeedSize(int32_t calCnt)
    {
        int typeSize = static_cast<int>(sizeof(int32_t));
        int32_t elementsPerBlock = 32 / typeSize;
        int32_t iter1OutputCount = calCnt;
        int32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XInType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<float> expertScalesGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<int32_t> performanceInfoGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    GlobalTensor<uint32_t> selfhcclDataStatusTensor_;
    GlobalTensor<int32_t> sendCountsGlobal_;

    LocalTensor<XOutType> xTmpTensor_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<XInType> xInTensor_;
    LocalTensor<XOutType> xOutTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> expertScalesTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIndexTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> performanceTmpTensor_;
    LocalTensor<int32_t> firstRecordTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    LocalTensor<float> expertCountTensor_;
    LocalTensor<float> expertTokenNumTensor_;
    LocalTensor<int64_t> expertTokenNumsLocalTensor_;
#endif

    TBuf<> expertIdsBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_; // 辅助gather tensor定义
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> waitStatusBuf_;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
    TBuf<> workLocalBuf_;
#endif
    TBuf<> maskBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> performanceTmpBuf_;
    TBuf<> firstRecordBuf_;
    TBuf<> gatherMaskTBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;                        // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                      // 量化使用，量化后的输出

    GM_ADDR expandXOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR expandScalesOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    uint32_t epRankId_{0};
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};    // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0}; // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t globalBS_{0};
    uint32_t moeExpertRankNum_{0}; // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t hAlignSize_{0};
    uint32_t hScaleIdxSize_{0};
    uint32_t expertScaleAlign_{0};
    uint32_t scaleInBytes_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t scalesCount_{0};
    uint32_t startExpertId_;
    uint32_t endExpertId_;
    uint32_t sendExpertNum_;
    uint32_t totalCnt_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t totalUsedUB_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_; // 接收Win区块数
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isPerformanceFlag_ = false;
    uint32_t cumsumTmpMinSize_ = 0;
    bool isScalingDownFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool hasExpertScalesFlag_ = false;
    float sumTarget_;
    uint64_t totalWinSizeEp_{0};
    uint32_t totalUbSize_{0};
    uint32_t expertTokenNumsType_{1};
    uint32_t preCnt_{0};
    uint32_t stateOffset_{0};
    uint32_t recStatusNumPerCore_{0};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t remainderRankNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t sendToSharedExpTokenCnt_{0};
    uint32_t maxSize_{0};
    uint32_t bufferNum_{0};
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};

    // 统一上下文持有器：v2 走 HcclOpParam，v3 走 Mc2MoeContext
    ContextHolder ctx_;

    DataCopyParams expandXCopyParams_;
    DataCopyParams xCopyParams_;
    DataCopyParams hCommuCopyOutParams_;
    DataCopyExtParams scaleOutParams_;

    MoeDistributeDispatchV2Quant<XInType, ExpandXOutType, XOutType, QuantMode, IsSmoothScaleExist> quantInst_;
    MoeDistributeElastic elasticInst_;
};

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expandScalesOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchV2TilingData *tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) // A3不支持MX量化，无需使能饱和模式
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;

    // 检查hcclwinsize是否越界
    totalWinSizeEp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    ctx_.InitAndCheck(mc2Context, tilingData->moeDistributeDispatchV2Info.epWorldSize, totalWinSizeEp_, tpipe_,
                      expandXOut);

    totalUbSize_ = static_cast<uint32_t>(tilingData->moeDistributeDispatchV2Info.totalUbSize);

    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    isPerformanceFlag_ = tilingData->moeDistributeDispatchV2Info.isPerformance;
    cumsumTmpMinSize_ = tilingData->moeDistributeDispatchV2Info.cumsumTmpMinSize;
    hasExpertScalesFlag_ = tilingData->moeDistributeDispatchV2Info.hasExpertScales;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    globalBS_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    scaleInBytes_ =
        tilingData->moeDistributeDispatchV2Info.scalesCol * tilingData->moeDistributeDispatchV2Info.scalesTypeSize;
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount;

    copyInAxisH_ = axisH_;
    copyOutAxisH_ = axisH_;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value ||
                  Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value) {
        copyOutAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
    if constexpr (Std::IsSame<XType, fp4x2_e2m1_t>::value || Std::IsSame<XType, fp4x2_e1m2_t>::value) {
        copyInAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
#endif

    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    statusDataSpaceGm_ = ctx_.GetStatusDataSpaceGm();
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempEpRankId = 0; tempEpRankId < epWorldSize_; tempEpRankId++) {
        OOMCheckAddrRange<XOutType>((__gm__ XOutType *)(GetWindAddrByRankId(tempEpRankId)), totalWinSizeEp_);
        OOMCheckAddrRange<float>((__gm__ float *)(GetWindStateAddrByRankId(tempEpRankId)), STATE_SIZE);
    }
#endif
    uint32_t epRankIdHccl = ctx_.GetEpRankId();
    uint32_t epWorldSizeHccl = ctx_.GetEpWorldSize();
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(elasticInfo));
    performanceInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(performanceInfo));
    elasticInst_.SetElasticInitParams(tpipe_, elasticInfoGMTensor_);
    if (hasElasticInfoFlag_) {
        totalUsedUB_ += elasticInst_.GetElasticInfoSizeAlign(epWorldSizeOriginal_);
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
        isScalingDownFlag_ = elasticInfoTensor_.GetValue(0);
        elasticInst_.InitElasticInfo(isScalingDownFlag_, epWorldSize_, sharedExpertRankNum_, moeExpertNum_, epRankId_,
                                     moeExpertNumPerRank_);
    }

    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }

    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;

    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    xGMTensor_.SetGlobalBuffer((__gm__ XInType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    sendCountsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOut));
    if constexpr (IsSmoothScaleExist) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    }
    expandXOutGM_ = expandXOut;
    recvCntWorkspaceGM_ = workspaceGM;

    hOutSize_ = copyOutAxisH_ * sizeof(XOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hScaleIdxSize_, scaleOutBytes_,
                         axisH_);
    if (hasExpertScalesFlag_) {
        expertScaleAlign_ = tokenQuantAlign_ + EXPAND_IDX_INFO;
        hScaleIdxSize_ += sizeof(float);
    }
    hAlignWinSize_ = Ceil(hScaleIdxSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(XOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    hOutAlignUbSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivNum_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    expertIdsCnt_ = axisBS_ * axisK_;
    recvWinBlockNum_ = epWorldSize_ * moeExpertNumPerRank_;
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    dealRankPerCore_ = (recvWinBlockNum_ + aivNum_ - 1) / aivNum_;
    stateOffset_ = STATE_OFFSET;
    PipeBarrier<PIPE_ALL>();
    if (isShareExpertRankFlag_) {
        rscvStatusNum_ = epWorldSize_;
    } else {
        rscvStatusNum_ = recvWinBlockNum_;
    }
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_; // 每个aiv需要处理的专家数
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_; // + sharedExpertRankNum_, 每个aiv发送的
    if (aivId_ < remainderRankNum_) {                  // 前remainderRankNum个aiv需要多发1个卡的数据
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }

    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    uint32_t statusBufCntAlign = Ceil(Ceil(totalExpertNum_, aivNum_), 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    uint32_t statusBufSize = statusBufCntAlign * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign * 8); // 8 = UB_ALIGN / sizeof(int32_t)
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign / 8, 1, 8); // 0x3F800000是float的1

    // 当前tpWin区划分为前后两半区，连续两次dispatch，切换半区, combine 数据区使用前面，
    // 即axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine, dispatch使用后面
    // DataAsFlag场景每512B有效数据为480B
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    uint64_t hSizeAlignCombine = Ceil(axisH_ * COMBINE_IN_DATA_SIZE, 480) * WIN_ADDR_ALIGN;
#else
    uint64_t hSizeAlignCombine = Ceil(axisH_ * COMBINE_IN_DATA_SIZE, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
#endif
    winDataSizeOffsetEp_ =
        dataState_ * (totalWinSizeEp_ / 2) + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;

    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<XOutType> winDouble;
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    winDouble.SetGlobalBuffer((__gm__ XOutType *)(windowGM_));
    OOMCheckAddrRange<XOutType>((__gm__ XOutType *)(winDouble.GetPhyAddr()), totalWinSizeEp_);
#endif
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));

    expertIdsCnt_ = axisBS_ * axisK_;
    uint32_t hFp32Size = axisH_ * sizeof(float);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr ((QuantMode == PERGROUP_DYNAMIC_QUANT) && IsSmoothScaleExist) {
        hFp32Size = Align128(axisH_) * sizeof(float);
    }
#endif
    uint32_t expertIdsSize = expertIdsCnt_ * sizeof(int32_t);
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t bsKAlign256 = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t expertIdsBufSize = expertIdsSize > bsAlign256 ? expertIdsSize : bsAlign256;
    expertIdsSize = Ceil(expertIdsSize, UB_ALIGN) * UB_ALIGN;
    maxSize_ = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSize_ = maxSize_ > xActivateMaskSize ? maxSize_ : xActivateMaskSize;
    maxSize_ = maxSize_ > bsKAlign256 ? maxSize_ : bsKAlign256;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize); // BS * K * 4 = 32K
    totalUsedUB_ += expertIdsSize;
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    tpipe_->InitBuffer(gatherMaskTBuf_, maxSize_);
    totalUsedUB_ += maxSize_;
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        uint32_t axisBSAlign = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(validBsIndexTBuf_, axisBSAlign);
        totalUsedUB_ += axisBSAlign;
        uint32_t validBufferSize = expertIdsSize > xActivateMaskSize ? expertIdsSize : xActivateMaskSize;
        tpipe_->InitBuffer(validExpertIndexBuf_, validBufferSize);
        totalUsedUB_ += expertIdsSize;
        validExpertIndexTensor_ = validExpertIndexBuf_.Get<int32_t>();
        validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
    }

    if (hasExpertScalesFlag_) {
        expandScalesOutGM_ = expandScalesOut;
        expertScalesGMTensor_.SetGlobalBuffer((__gm__ float *)expertScales);
        tpipe_->InitBuffer(expertScalesBuf_, expertIdsBufSize);
        totalUsedUB_ += expertIdsBufSize;
    }

    if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
        tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize_); // max{28K, BS * K * 4B}
        totalUsedUB_ += maxSize_;
        floatLocalTemp_ = receiveDataCastFloatBuf_.Get<float>();
        tpipe_->InitBuffer(smoothScalesBuf_, maxSize_); // max{28K, hAlign * 4B}
        totalUsedUB_ += maxSize_;
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN); // 32B
            rowMaxTensor_ = rowMaxBuf_.Get<float>();
        }
        uint32_t tmpTotalUB = totalUsedUB_ + BUFFER_NUM * hAlignSize_ + hOutAlignUbSize_ * BUFFER_NUM;
        bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
        tpipe_->InitBuffer(xInQueue_, bufferNum_, hAlignSize_);       // 14K * 2
        tpipe_->InitBuffer(xOutQueue_, bufferNum_, hOutAlignUbSize_); // 7K * 2 + 32 + 6
        dstExpBuf_ = receiveDataCastFloatBuf_;                        // 内存复用
        subExpBuf_ = smoothScalesBuf_;                                // 内存复用
    } else {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_); // BS * K * 4 = 32K
        totalUsedUB_ += maxSize_;
        tpipe_->InitBuffer(subExpBuf_, maxSize_); // BS * K * 4 = 32K
        totalUsedUB_ += maxSize_;
        uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
        bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
        tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_); // 7k*2 + 32 + 12
    }

    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    uint32_t axisHCommu = hScaleIdxSize_ / sizeof(XOutType); // 有效搬运长度
    xCopyParams_ = {1U, static_cast<uint16_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U};
    hCommuCopyOutParams_ = {1U, static_cast<uint16_t>(axisHCommu * sizeof(XOutType)), 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint16_t>(hOutSize_), 0U, 0U};
    scaleOutParams_ = {1U, static_cast<uint16_t>(scaleOutBytes_), 0U, 0U, 0U};

    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf_, dynamicScalesOutGMTensor_);
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_; // 由于是后面的核作为发送的共享专家，因此需要换算
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

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::FillTriple(
    LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;
    xOutTint32(tokenQuantAlign_ + 2) = k;
    LocalTensor<float> xOutTfloat = xOutTensor.template ReinterpretCast<float>();
    if ((k < axisK_) && (hasExpertScalesFlag_)) {
        xOutTfloat(expertScaleAlign_) = expertScalesTensor_.GetValue(tokenIndex * axisK_ + k);
    }
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::ProcessToken(
    GlobalTensor<XOutType> &outTokenGT, uint32_t tokenIndex, uint32_t topKIndex, DataCopyPadParams &padParams,
    DataCopyParams &scaleInParams, uint32_t expertIndex)
{
    if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
        xInTensor_ = xInQueue_.AllocTensor<XInType>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        LocalTensor<uint8_t> singleByteTok = xInTensor_.template ReinterpretCast<uint8_t>();
        if constexpr ((QuantMode == MX_QUANT) || (QuantMode == MX_QUANT_CLIP) ||
                      (QuantMode == PERGROUP_DYNAMIC_QUANT)) { // 提前填0，避免脏数据
            Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XInType));
        }
#endif
        SyncFunc<HardEvent::V_MTE2>();
        DataCopyPad(xInTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, padParams);
        xInQueue_.EnQue(xInTensor_);
        xInTensor_ = xInQueue_.DeQue<XInType>();
        xOutTensor_ = xOutQueue_.AllocTensor<XOutType>();
        if constexpr (QuantMode > UNQUANT) {
            quantInst_.QuantProcess(xOutTensor_, xInTensor_, expertIndex, scalesCount_, scalesGMTensor_); // 量化
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else { // 非量化直转hifp8
            Cast(floatLocalTemp_, xInTensor_, RoundMode::CAST_NONE, axisH_);
            Cast(xOutTensor_, floatLocalTemp_, RoundMode::CAST_ROUND, axisH_);
        }
#endif
        SyncFunc<AscendC::HardEvent::V_S>();
        FillTriple(xOutTensor_, tokenIndex, topKIndex);
        xOutQueue_.EnQue(xOutTensor_);
        xInQueue_.FreeTensor<XInType>(xInTensor_);
        xOutTensor_ = xOutQueue_.DeQue<XOutType>();
        DataCopyPad(outTokenGT, xOutTensor_, hCommuCopyOutParams_);
        xOutQueue_.FreeTensor<XOutType>(xOutTensor_);
    } else {
        xTmpTensor_ = xQueue_.AllocTensor<XOutType>();
        if constexpr (!IsSmoothScaleExist) {
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, padParams);
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else {
            auto tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * copyInAxisH_], xCopyParams_, padParams);
            DataCopyPad(xTmpTensor_[Align32(copyInAxisH_)].template ReinterpretCast<uint8_t>(),
                        tmp[tokenIndex * scaleInBytes_], scaleInParams, padParams);
        }
#endif
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        FillTriple(xTmpTensor_, tokenIndex, topKIndex);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XOutType>();
        DataCopyPad(outTokenGT, xTmpTensor_, hCommuCopyOutParams_);
        xQueue_.FreeTensor<XOutType>(xTmpTensor_);
    }
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SendToSharedExpert()
{
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_;
    uint32_t startTokenId, endTokenId, sendTokenNum; // 每个aiv发送时的起始rankid
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {
        return;
    }

    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_; // 计算目的共享专家卡在其所在共享专家组的id
    GlobalTensor<XOutType> dstWinGMTensor;
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t tokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        int32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ XOutType *)(GetWindAddrByRankId(toRankId) + expertPerSizeOnWin_ * epRankId_));
        uint32_t srcTokenIndex = tokenIndex;
        if (isExpertMaskFlag_) {
            srcTokenIndex = validBsIndexTensor_.GetValue(tokenIndex);
        }
        GlobalTensor<XOutType> tempTensor = dstWinGMTensor[tokenIndex * hAlignWinCnt_];
        ProcessToken(tempTensor, srcTokenIndex, axisK_ + toSharedExpertIndex, padParams, scaleInParams,
                     toSharedExpertIndex);
    }
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::CalTokenSendExpertCnt(uint32_t dstExpertId,
                                                                                                  int32_t calCnt,
                                                                                                  int32_t &curExpertCnt)
{
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    Abs(tmpoutFp32, tmpFp32, calCnt);
#else
    Abs(dstExpIdTensor_, subExpIdTensor_, calCnt);
#endif
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

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SendToMoeExpert()
{
    uint32_t startTokenId, endTokenId, sendTokenNum; // 每个aiv发送时的起始rankid
    SplitToCore(sendToMoeExpTokenCnt_, moeUsedAivNum_, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= sendToMoeExpTokenCnt_) {
        return;
    }
    GlobalTensor<XOutType> dstWinGMTensor;
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
    int32_t tokenIndex = 0;
    int32_t topKIndex = 0;
    uint32_t dstExpertId = 0;
    int32_t expertIdx = 0;
    if (isExpertMaskFlag_) {
        SyncFunc<AscendC::HardEvent::V_S>();
    }
    for (int32_t index = startTokenId; index < endTokenId; ++index) {
        if (isExpertMaskFlag_) {
            expertIdx = validExpertIndexTensor_(index);
            tokenIndex = expertIdx / axisK_;
        } else {
            expertIdx = index;
            tokenIndex = index / axisK_;
        }
        int32_t curExpertCnt = 0;
        topKIndex = expertIdx % axisK_;
        dstExpertId = expertIdsTensor_(index);
        if ((tokenIndex > 0) && (index > 0)) {
            SyncFunc<AscendC::HardEvent::S_V>();
            CalTokenSendExpertCnt(dstExpertId, index, curExpertCnt);
        }
        int32_t toRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;

        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }

        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(toRankId) +
                                            (expertPerSizeOnWin_ *
                                             (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                            hAlignWinSize_ * curExpertCnt); // 计算地址偏移
        dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType *)rankGM);
        ProcessToken(dstWinGMTensor, tokenIndex, topKIndex, padParams, scaleInParams, dstExpertId + sharedExpertNum_);
    }
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask, 当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskTmpTensor = subExpBuf_.Get<half>();
    sumOutTensor = gatherMaskTBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(maskTmpTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::CalValidExpIdx(
    LocalTensor<bool> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    LocalTensor<half> tempTensor = subExpBuf_.Get<half>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor = expertIdsBuf_.Get<int32_t>();

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0,
                        Ceil(expertIdsCnt_, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::CalValidBSCnt(
    LocalTensor<bool> maskStrideTensor)
{
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    LocalTensor<half> tempTensor = validExpertIndexBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = expertIdsBuf_.Get<half>();
    LocalTensor<half> tokenTargetTensor = validBsIndexTBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = gatherMaskTBuf_.Get<uint32_t>();
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize);
    PipeBarrier<PIPE_V>();
    Sum(tokenTargetTensor, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::ExpertActiveMaskCal()
{
    // 计算当前有效bs数量, stride搬入xActiveMask进行sum计算, 用于moe专家发送
    LocalTensor<bool> maskStrideTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskStrideCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskStrideParams{static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U,
                                       0U, 0U};
    DataCopyPad(maskStrideTensor, xActiveMaskGMTensor_, maskStrideParams, maskStrideCopyPadParams);
    CalValidBSCnt(maskStrideTensor);
    // 计算validExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送
    LocalTensor<bool> maskInputTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalValidExpIdx(maskInputTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::MaskZeroComputeExpert(uint32_t maskCnt)
{
    LocalTensor<int32_t> expertsIndexTensor = dstExpBuf_.Get<int32_t>();
    int32_t maskTensorInt16Cnt = Ceil(expertIdsCnt_, UB_ALIGN / 2);
    LocalTensor<uint8_t> maskTensorInt8 = validExpertIndexBuf_.Get<uint8_t>();    // bs*k*1
    LocalTensor<uint32_t> maskTensorInt32 = validExpertIndexBuf_.Get<uint32_t>(); // bs*k*1
    LocalTensor<half> expertIdsTensorCast = subExpBuf_.Get<half>();               // bs*k*4
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertNum_);

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    // half类型最大可精准表示的整数是2048，当前dispatch最大可支持专家数为1024，大于该值的情况均为特殊专家，dispatch均不做发送即可，combine再分专家类型进行对应处理；
    Cast(expertIdsTensorCast, expertIdsTensor_, RoundMode::CAST_NONE, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(maskTensorInt32, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    uint32_t calcCnt = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    // 逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。筛掉零计算量专家
    CompareScalar(maskTensorInt8, expertIdsTensorCast, static_cast<half>(moeExpertNumInt32), AscendC::CMPMODE::LT,
                  calcCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint16_t> maskTensorInt16 = validExpertIndexBuf_.Get<uint16_t>();  // 空间bs*k*1
    LocalTensor<uint16_t> gatherMaskTensorint16 = gatherMaskTBuf_.Get<uint16_t>(); // 空间bs*k*4
    /* 特殊专家的maskTensorInt16和之前的gatherMaskTensor_结果按位相与，AND 支持uint16，
     * gatherMaskTensor_和gatherMaskTensorint16是同一个地址 */
    And(gatherMaskTensorint16, gatherMaskTensorint16, maskTensorInt16, maskTensorInt16Cnt);
    PipeBarrier<PIPE_V>();
    // 再筛一次
    CreateVecIndex(expertsIndexTensor, 0, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, maskCnt, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::GenerateGatherMaskTensor(uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::ZeroComputeExpertMaskCal()
{
    uint32_t maskCnt = expertIdsCnt_;
    if (isTokenMaskFlag_) { // 一维
        maskCnt = activeMaskBsCnt_ * axisK_;
    }

    if (!isExpertMaskFlag_) { // 非二维要生成gatherMaskTensor_
        GenerateGatherMaskTensor(maskCnt);
    }

    // 零计算量专家剪枝
    MaskZeroComputeExpert(maskCnt);
}

/*
共享专家卡：所有核用于给moe专家发送数据
moe专家卡：部分核用于给共享专家发送数据，部分核用于给moe专家发送数据
*/
template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::AlltoAllDispatch()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }
    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }

    if (activeMaskBsCnt_ == 0) {
        return;
    }

    if (zeroComputeExpertNum_ != 0) {
        ZeroComputeExpertMaskCal();
    }
    // 后面的核向共享专家发数据
    bool isSendShared = (aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0);
    if (isSendShared) {
        SendToSharedExpert();
    }

    if (zeroComputeExpertNum_ != 0) {
        isExpertMaskFlag_ = true;
    }
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    if (isExpertMaskFlag_) {
        uint64_t rsvdCnt = 0;
        uint32_t mask = expertIdsCnt_;
        LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
        DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(expertIdsTensor_, tmpExpertIdsTensor, gatherMaskTensor_, true, mask, {1, 1, 1, 0}, rsvdCnt);
    } else {
        DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
    if (hasExpertScalesFlag_) {
        expertScalesTensor_ = expertScalesBuf_.Get<float>();
        DataCopyExtParams expertScalesParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> expertScalesPadParams{false, 0U, 0U, 0U};
        DataCopyPad(expertScalesTensor_, expertScalesGMTensor_, expertScalesParams, expertScalesPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
    if (isSendShared) { // 用于send共享专家数据的核，也需要搬运expertIds，后续会重新分核写状态位置，该核可能用于写moe专家flag
        return;
    }
    SendToMoeExpert();
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SetStatus()
{
    SplitToCore(totalExpertNum_, aivNum_, startExpertId_, endExpertId_, sendExpertNum_);

    for (uint32_t curExpertId = startExpertId_; curExpertId < endExpertId_; ++curExpertId) {
        int32_t curExpertCnt = 0;
        // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        int32_t cntPosIndex = (curExpertId - startExpertId_) * 8 + 1;
        if ((curExpertId < sharedExpertRankNum_) &&
            (activeMaskBsCnt_ > 0)) { // 当前处理专家为共享专家 shared:Cnt -> win -> LocalCopy(moe+share)
            if (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                curExpertCnt = activeMaskBsCnt_;
            }
        } else if (sendToMoeExpTokenCnt_ > 0) { // 当前处理卡为moe专家卡
            int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
            CalTokenSendExpertCnt(curMoeExpertId, sendToMoeExpTokenCnt_, curExpertCnt);
        }
        statusTensor_(cntPosIndex) = curExpertCnt;
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if (startExpertId_ >= totalExpertNum_) {
        return;
    }
    GlobalTensor<int32_t> rankGMTensor;

    for (uint32_t expertIndex = startExpertId_; expertIndex < endExpertId_; ++expertIndex) {
        uint32_t dstRankId = expertIndex;
        uint32_t offset = stateOffset_ * epRankId_;
        if (expertIndex >= sharedExpertRankNum_) {
            dstRankId = ((expertIndex - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((expertIndex - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * stateOffset_);
        }
        if (isScalingDownFlag_) {
            dstRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
        }
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset); // 计算地址偏移
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        // 按32对齐拷贝，8是32字节包含的元素个数, 本卡数据需要去掉起始index偏移
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(expertIndex - startExpertId_) * 8], 8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SyncCntOnCore(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, STATUS_BLOCK_COUNT_SEL_PATTERN); // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2; // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家recv cnt收集到gatherMaskOutTensor
    uint64_t rsvdCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
               {1, (uint16_t)recStatusNumPerCore_, 1, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) // 对inner要求32对齐
                                        * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);

    // 把当前核的所有专家的recv cnt之和写到workspace
    uint32_t coreOffset = WORKSPACE_ELEMENT_OFFSET * aivNum_;
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_) + coreOffset * aivId_; // 写workspace需要按照512字节对齐
    GlobalTensor<int32_t> sumTensor;
    sumTensor.SetGlobalBuffer((__gm__ int32_t *)wAddr);
    uint16_t workCoreNum = MIN(recvWinBlockNum_, aivNum_);
    // 每个核把sumOfRecvCnt重复写workCoreNum份
    LocalTensor<int32_t> sumCoreTensor = sumCoreBuf_.Get<int32_t>();
    // 仅处理每个datablock的首元素（对应maskArray[0]的bit0）。操作数为32bit情况下，maskArray只有第0个元素有效
    // 每个元素占4字节，每个32字节处理8份，mask中每8个bit的填充第1位
    uint64_t maskArray[2] = {0x0101010101010101, 0};
    // 每个核一个datablock，总共需要处理workCoreNum个核。每个repeat总共256字节，可以处理8个datablock
    uint8_t repeatTimes = (workCoreNum + 7) / 8;
    // 1代表单个repeat内不同的datablock连续，没有跳过
    // 8代表不同repeat的首元素间隔8个datablock
    Duplicate<int32_t>(sumCoreTensor, sumOfRecvCnt, maskArray, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(workCoreNum), 1, 0, 15};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(sumTensor, sumCoreTensor, sumIntriParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::BufferInit()
{
    tpipe_->Reset();
    totalUsedUB_ = 0U;
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256) ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize); // 1024/24 * 32B = 43 * 32B
    totalUsedUB_ += waitStatusBufSize;
    // 内存复用，取大
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = recvWinBlockNum_ * sizeof(float);
    uint64_t gatherMaskOutSize =
        (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    uint32_t setExpertTokenNumsLocalExpertNum = isShareExpertRankFlag_ ? 1U : moeExpertNumPerRank_;
    // SetExpertTokenNums中expert count按[expert][rank]紧凑存放。
    uint64_t setExpertTokenNumsStatusSpace =
        static_cast<uint64_t>(setExpertTokenNumsLocalExpertNum) * epWorldSize_ * sizeof(float);
    setExpertTokenNumsStatusSpace = Ceil(setExpertTokenNumsStatusSpace, UB_ALIGN) * UB_ALIGN;
    gatherMaskOutSize =
        (gatherMaskOutSize > setExpertTokenNumsStatusSpace) ? gatherMaskOutSize : setExpertTokenNumsStatusSpace;
#endif
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize); // recStatusNumPerCore_32对齐后大小  * 32B
    totalUsedUB_ += gatherMaskOutSize;
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN); // 48 * 32B
    totalUsedUB_ += aivNum_ * UB_ALIGN;
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN); // 48 * 32B
    totalUsedUB_ += aivNum_ * UB_ALIGN;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // reduceSum计算所需的Tensor空间
    int32_t reduceSumWorkNeedSize = ReduceSumWorkNeedSize(moeExpertNumPerRank_ * epWorldSize_);
    tpipe_->InitBuffer(sumContinueBuf_, reduceSumWorkNeedSize * sizeof(float));
    totalUsedUB_ += reduceSumWorkNeedSize * sizeof(float);
#else
    tpipe_->InitBuffer(sumContinueBuf_, aivNum_ * sizeof(float)); // 48 * 4B
    totalUsedUB_ += aivNum_ * sizeof(float);
#endif
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3); // 96B
    totalUsedUB_ += UB_ALIGN * 3;
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_); // 7k*2 + 32 + 12
    totalUsedUB_ += BUFFER_NUM * hOutAlignUbSize_;
    if (isPerformanceFlag_) {
        uint32_t performanceTmpSize = JUMP_WRITE * epWorldSizeOriginal_ * sizeof(int32_t);
        uint32_t performanceTmpAlign = Ceil(performanceTmpSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(performanceTmpBuf_, performanceTmpAlign);
        totalUsedUB_ += performanceTmpAlign;
        performanceTmpTensor_ = performanceTmpBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceTmpTensor_, 0, performanceTmpAlign / sizeof(int32_t));
        uint32_t firstRecordSize = recStatusNumPerCore_ * sizeof(int32_t);
        uint32_t firstRecordSizeAlign = Ceil(firstRecordSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(firstRecordBuf_, firstRecordSizeAlign);
        totalUsedUB_ += firstRecordSizeAlign;
        firstRecordTensor_ = firstRecordBuf_.Get<int32_t>();
        Duplicate<int32_t>(firstRecordTensor_, 0, firstRecordSizeAlign / sizeof(int32_t));
    }
    tpipe_->InitBuffer(tokenNumBuf_, Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN);
    totalUsedUB_ += Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // A5 statusBuf 兼作 CumSum tmpBuffer，超 UB 则不扩展
    uint32_t statusBufOrigin = moeExpertNumPerRank_ * epWorldSize_ * UB_ALIGN;
    uint32_t statusBufSize = (statusBufOrigin > cumsumTmpMinSize_) ? statusBufOrigin : cumsumTmpMinSize_;
    if (totalUsedUB_ + statusBufSize > totalUbSize_) {
        statusBufSize = statusBufOrigin;
    }
#else
    uint32_t statusBufSize = moeExpertNumPerRank_ * epWorldSize_ * UB_ALIGN;
#endif
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
    // A3基础ReduceSum需要workLocalBuf_；A5复用已有UB buffer，不单独申请
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
    uint32_t workLocalBufSize = Ceil(epWorldSize_, UB_ALIGN / sizeof(float)) * UB_ALIGN;
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
    totalUsedUB_ += workLocalBufSize;
#endif
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::PerformanceInfoPerRank(
    uint64_t performanceTimeStart, bool isCompactStatus)
{
    uint64_t performanceTimecheck = static_cast<uint64_t>(GetSystemCycle());
    int32_t performanceTimeWait = static_cast<int32_t>((performanceTimecheck - performanceTimeStart) / CYCLES_PER_US);
    SyncFunc<AscendC::HardEvent::V_S>();
    for (uint32_t i = 0; i < recStatusNumPerCore_; i++) {
        uint32_t flagIndex = isCompactStatus ? i : i * FLAG_OFFSET;
        if (statusFp32Tensor_.GetValue(flagIndex) > float(0.5) && firstRecordTensor_.GetValue(i) == 0) {
            uint32_t fromRankId;
            uint32_t index = startExpertId_ + i;
            if (isScalingDownFlag_) {
                fromRankId =
                    elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + index % epWorldSize_);
            } else {
                fromRankId = index % epWorldSize_;
            }
            if (performanceTimeWait > performanceTmpTensor_.GetValue(JUMP_WRITE * fromRankId)) {
                performanceTmpTensor_.SetValue(JUMP_WRITE * fromRankId, performanceTimeWait);
                firstRecordTensor_.SetValue(i, 1);
            }
        }
    }
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::PerformanceInfoCopyOut()
{
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SetAtomicMax<int32_t>();
    DataCopyExtParams performanceInfoCopyParams{
        1U, JUMP_WRITE * epWorldSizeOriginal_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(performanceInfoGMTensor_, performanceTmpTensor_, performanceInfoCopyParams);
    SetAtomicNone();
}

// A5用Compact只轮询flag字段；A3不支持compact搬运，回退普通DataCopy+ReduceSum
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::WaitStatusA5(
    LocalTensor<float> &statusSumOutTensor)
{
    float compareTarget = sumTarget_ * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyExtParams flagCopyParams{static_cast<uint16_t>(recStatusNumPerCore_), sizeof(float),
                                     STATE_OFFSET - sizeof(float), 0U, 0U};
    DataCopyPadExtParams<float> flagCopyPadParams{false, 0U, 0U, 0.0f};
    uint32_t flagSumInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams flagSumParams{1, flagSumInner, recStatusNumPerCore_};
    uint64_t performanceTimeStart = static_cast<uint64_t>(GetSystemCycle());
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopyPad<float, PaddingMode::Compact>(statusFp32Tensor_,
                                                 windowInstatusFp32Tensor_[startStatusIndex_ * FLAG_OFFSET],
                                                 flagCopyParams, flagCopyPadParams);
        if (isPerformanceFlag_) {
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            PerformanceInfoPerRank(performanceTimeStart, true);
        }
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, flagSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    if (isPerformanceFlag_) {
        PerformanceInfoCopyOut();
    }
    // 清状态仍按完整32B block回写，需要先补搬完整status以保留count字段。
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * FLAG_OFFSET], intriParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
}
#else
template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::WaitStatus(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<float> &statusSumOutTensor)
{
    float compareTarget = sumTarget_ * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    uint32_t mask = 1; // gatherMask + sum 相关参数
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t performanceTimeStart = static_cast<uint64_t>(GetSystemCycle());
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        if (isPerformanceFlag_) {
            PerformanceInfoPerRank(performanceTimeStart, false);
        }
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    if (isPerformanceFlag_) {
        PerformanceInfoCopyOut();
    }
}
#endif

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::WaitDispatch()
{
    BufferInit();
    startExpertId_ = startStatusIndex_; // 后面LocalWinCopy分核与此处保持一致
    endExpertId_ = startExpertId_ + recStatusNumPerCore_;
    sendExpertNum_ = recStatusNumPerCore_;
    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        SyncAll<true>();
        return;
    }
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherTmpTensor.SetValue(0, 1);
#endif
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    if (isScalingDownFlag_) {
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    WaitStatusA5(statusSumOutTensor);
#else
    WaitStatus(gatherMaskOutTensor, statusSumOutTensor);
#endif
    // 清状态
    WaitDispatchClearStatus();

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // A5不需要考虑避免同地址访问, 跳过SyncCntOnCore
#else
    // 核间同步token cnt
    SyncCntOnCore(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
#endif
    SyncAll<true>();
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::GetCumSumA5(LocalTensor<int32_t> &outLocal)
{
    // 不支持allgather场景，只需要拷贝totalCount个核的recv cnt
    uint16_t copySumNum = startStatusIndex_;

    // copySumNum为0表示不执行搬运，该接口被视为NOP(空操作)
    DataCopyParams statusCopyParams{static_cast<uint16_t>(copySumNum), 1, 0, 0};
    LocalTensor<float> statusTensorFp32 = statusBuf_.Get<float>();
    DataCopy(statusTensorFp32, windowInstatusFp32Tensor_, statusCopyParams);
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    LocalTensor<float> sharedTmpTensor = sumContinueBuf_.Get<float>();
    uint32_t mask = 2;
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    ReduceSum(recvCntSumOutTensor, statusTensorFp32, sharedTmpTensor, mask, startStatusIndex_, 1);
    SyncFunc<AscendC::HardEvent::V_S>();
    // 最终输出outLocal第0个元素是当前核前面所有核recv cnt总和
    outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::GetCumSum(LocalTensor<int32_t> &outLocal,
                                                                                      uint32_t totalCount)
{
    // 获取workspace中每个核的recvcnt
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_) + WORKSPACE_ELEMENT_OFFSET * aivId_;
    GlobalTensor<int32_t> sumTensor;
    sumTensor.SetGlobalBuffer((__gm__ int32_t *)wAddr);

    // 不支持allgather场景，只需要拷贝totalCount个核的recv cnt
    uint16_t copySumNum = totalCount;
    uint16_t copyStride = 16 * aivNum_ - 1;
    DataCopyParams sumIntriParams{static_cast<uint16_t>(copySumNum), 1, copyStride, 0};
    LocalTensor<int32_t> sumLocalTensor = sumLocalBuf_.Get<int32_t>();
    DataCopy(sumLocalTensor, sumTensor, sumIntriParams);
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherSumPattern.SetValue(0, 1);
    uint32_t mask = 1;
    uint64_t rsvdCnt = 0;
    LocalTensor<int32_t> sumContinueTensor = sumContinueBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, copySumNum, 1, 0}, rsvdCnt);
    // height, width(按照32字节对齐padding后总元素个数), nNum，结果矩阵第一列为对应行的求和结果
    uint32_t innerSumParams = (copySumNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN / sizeof(float);
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = sumContinueTensor.ReinterpretCast<float>();
    SumParams sumParams{1, innerSumParams, totalCount};
    Sum(recvCntSumOutTensor, tmpFp32, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    // 最终输出outLocal第0个元素是当前核前面所有核recv cnt总和
    outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::RunPosRecord()
{
    TBuf<> runPosBuf;
    tpipe_->InitBuffer(runPosBuf, UB_ALIGN);
    DataCopyParams dataStateParams{1U, sizeof(uint32_t), 0U, 0U};
    dataStateLocalTensor_ = runPosBuf.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, FLAG_AFTER_WAIT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams);
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::LocalWindowCopy()
{
    RunPosRecord(); // 维测打点
    LocalTensor<int32_t> outCountLocal, xTmpTensorInt;
    LocalTensor<float> xTmpTensorFloat;
    if (startExpertId_ >= rscvStatusNum_) {
        return;
    } // 分核已与前面的waitDispatch里保持一致return;

    outCountLocal = gatherMaskOutBuf_.Get<int32_t>();
    if (aivId_ != 0) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        GetCumSumA5(outCountLocal);
#else
        GetCumSum(outCountLocal, aivId_);
#endif
    } else {
        outCountLocal.SetValue(0, 0);
    }

    // hOutElemCount为expandXOutGlobal申请每个token的GM Buffer空间大小
    uint32_t index = 0, beginIdx = outCountLocal.GetValue(0), hOutElemCount = hOutSize_ / sizeof(XOutType);
    preCnt_ = beginIdx, statusTensor_ = waitStatusBuf_.Get<int32_t>();
    DataCopyPadParams padParams = {false, 0U, 0U, 0U};
    DataCopyExtParams dataCopyExpandIdxParams{1U, sizeof(int32_t) * EXPAND_IDX_INFO, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tripleQueue; // 用于搬运接收到的数据
    uint32_t leftUBSize = ((totalUbSize_ - totalUsedUB_) / UB_ALIGN) * UB_ALIGN;
    uint32_t dataSize = leftUBSize / BUFFER_NUM;
    tpipe_->InitBuffer(tripleQueue, BUFFER_NUM, dataSize);

    uint32_t countPerDataSize = dataSize / hOutAlignUbSize_;
    for (uint32_t index = startExpertId_; index < endExpertId_; index++) {
        uint32_t i = (index - startExpertId_), winOffset = index;
        uint32_t count = statusTensor_.GetValue(i * 8 + 1);
        outCountLocal.SetValue(i, beginIdx + count);
        if ((!isShareExpertRankFlag_) &&
            (moeExpertNumPerRank_ > 1)) { // moe专家卡且一卡多专家场景 转换成数据区的排布偏移
            winOffset = index % epWorldSize_ * moeExpertNumPerRank_ + index / epWorldSize_;
        }
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + winOffset * expertPerSizeOnWin_;
        GlobalTensor<XOutType> tokGlobal, expandXOutGlobal;
        GlobalTensor<float> expertScaleInGlobal, expandScalesOutGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        uint32_t loopTimes = Ceil(count, countPerDataSize);
        uint32_t lastTimeCount = count - countPerDataSize * (loopTimes - 1);
        for (uint32_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
            uint32_t tokenCopiedNumThisExpert = countPerDataSize * loopIdx;
            tokGlobal.SetGlobalBuffer((__gm__ XOutType *)(wAddr + tokenCopiedNumThisExpert * hAlignWinSize_));

            uint16_t blockCount = static_cast<uint16_t>(((loopIdx < loopTimes - 1) ? countPerDataSize : lastTimeCount));
            uint16_t copyBlockLen = static_cast<uint16_t>(hOutAlignUbSize_ / UB_ALIGN);
            uint16_t srcStrideTokGlobal = static_cast<uint16_t>((hAlignWinSize_ - hOutAlignUbSize_) / UB_ALIGN);
            DataCopyParams dataCopyParamsTokGlobal{blockCount, copyBlockLen, srcStrideTokGlobal, 0U};
            LocalTensor<XOutType> tripleLocal = tripleQueue.AllocTensor<XOutType>();
            DataCopy(tripleLocal, tokGlobal, dataCopyParamsTokGlobal);
            tripleQueue.EnQue(tripleLocal);
            tripleLocal = tripleQueue.DeQue<XOutType>();

            expandXOutGlobal.SetGlobalBuffer(
                (__gm__ XOutType *)(expandXOutGM_ + (beginIdx + tokenCopiedNumThisExpert) * hOutSize_));
            uint32_t srcStrideExpandX = (hOutAlignUbSize_ - hOutSize_) / UB_ALIGN;
            DataCopyExtParams dataCopyParamsExpandX{blockCount, hOutSize_, srcStrideExpandX, 0U, 0U};
            DataCopyPad(expandXOutGlobal, tripleLocal, dataCopyParamsExpandX);

            uint32_t srcStrideScales = (hOutAlignUbSize_ - scaleOutBytes_) / UB_ALIGN;
            DataCopyExtParams dataCopyParamsScales{blockCount, scaleOutBytes_, srcStrideScales, 0U, 0U};
            quantInst_.CopyScalesToOut(beginIdx + tokenCopiedNumThisExpert, scaleOutBytes_, tripleLocal,
                                       dataCopyParamsScales);

            LocalTensor<int32_t> tripleLocalInt32 = tripleLocal.template ReinterpretCast<int32_t>();
            uint32_t expandIdxSize = EXPAND_IDX_INFO * static_cast<uint32_t>(sizeof(int32_t));
            uint32_t srcStrideExpandIdx = (hOutAlignUbSize_ - expandIdxSize) / UB_ALIGN;
            DataCopyExtParams dataCopyParamsExpandIdx{blockCount, expandIdxSize, srcStrideExpandIdx, 0U, 0U};
            DataCopyPad(expandIdxGMTensor_[(beginIdx + tokenCopiedNumThisExpert) * EXPAND_IDX_INFO],
                        tripleLocalInt32[tokenQuantAlign_], dataCopyParamsExpandIdx);

            if (hasExpertScalesFlag_) {
                xTmpTensorFloat = tripleLocal.template ReinterpretCast<float>();
                expertScaleInGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                    wAddr + tokenCopiedNumThisExpert * hAlignWinSize_ + expertScaleAlign_ * sizeof(float)));
                expandScalesOutGlobal.SetGlobalBuffer((__gm__ float *)(expandScalesOutGM_) + beginIdx +
                                                      tokenCopiedNumThisExpert);
                DataCopyExtParams expertScaleCopyInParams{blockCount, static_cast<uint32_t>(sizeof(float)),
                                                          static_cast<uint32_t>(hAlignWinSize_ - sizeof(float)), 0U,
                                                          0U};
                DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0.0f};
                DataCopyExtParams expertScaleCopyOutParams{blockCount, static_cast<uint32_t>(sizeof(float)), 0U, 0U,
                                                           0U};
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                DataCopyPad(xTmpTensorFloat, expertScaleInGlobal, expertScaleCopyInParams, expertScalePadParams);
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                DataCopyPad(expandScalesOutGlobal, xTmpTensorFloat, expertScaleCopyOutParams);
            }
            tripleQueue.FreeTensor(tripleLocal);
        }
        beginIdx += count;
    }
    totalCnt_ = beginIdx;
    lastCore_ = MIN(rscvStatusNum_, aivNum_) - 1;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(sendCountsGlobal_[startExpertId_], outCountLocal, dataCopyOutParams);
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SetExpertPrefixSumA5(
    uint32_t localExpertNum, bool useCumSum)
{
    DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)), 0U, 0U,
                                                0U};
    LocalTensor<int32_t> expertTokenNumInt32Tensor = expertTokenNumTensor_.ReinterpretCast<int32_t>();
    if (!useCumSum) {
        SyncFunc<AscendC::HardEvent::V_S>();
        int64_t expertTokenNumCumsum = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
            int64_t expertTokenNum = static_cast<int64_t>(expertTokenNumInt32Tensor.GetValue(localExpertIdx));
            expertTokenNumCumsum += expertTokenNum;
            expertTokenNumsLocalTensor_.SetValue(localExpertIdx, expertTokenNumCumsum);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor_, expertTokenNumsCopyParams);
        return;
    }

    uint32_t cumsumInner = Ceil(localExpertNum * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    CumSumInfo cumsumInfo{1U, cumsumInner};
    LocalTensor<float> expertTokenNumCumsumTensor = expertCountTensor_;
    LocalTensor<float> cumsumLastRowTensor = scalarBuf_.Get<float>();
    LocalTensor<uint8_t> cumsumTmpTensor = statusBuf_.Get<uint8_t>();
    PipeBarrier<PIPE_V>();
    CumSum<float, SET_EXPERT_TOKEN_NUMS_CUMSUM_CONFIG>(expertTokenNumCumsumTensor, cumsumLastRowTensor,
                                                       expertTokenNumTensor_, cumsumTmpTensor, cumsumInfo);
    PipeBarrier<PIPE_V>();
    Cast(expertTokenNumsLocalTensor_, expertTokenNumCumsumTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
         localExpertNum);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor_, expertTokenNumsCopyParams);
}

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SetExpertTokenNumsA5()
{
    if (aivId_ != FIRST_CORE) {
        return;
    }

    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1U : moeExpertNumPerRank_;
    uint32_t localStatusSectionNum = localExpertNum * epWorldSize_;
    expertCountTensor_ = gatherMaskOutBuf_.Get<float>();
    DataCopyPadExtParams<float> countCopyPadParams{false, 0U, 0U, 0.0f};
    DataCopyExtParams countCopyParams{static_cast<uint16_t>(localStatusSectionNum), sizeof(float),
                                      STATE_OFFSET - sizeof(float), 0U, 0U};
    DataCopyPad<float, PaddingMode::Compact>(expertCountTensor_,
                                             windowInstatusFp32Tensor_[STATUS_BLOCK_COUNT_FIELD_OFFSET],
                                             countCopyParams, countCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    expertTokenNumTensor_ = statusBuf_.Get<float>();
    uint32_t statusBufSize = statusBuf_.Get<uint8_t>().GetSize();
    bool useCumSum = (expertTokenNumsType_ == 0) && (cumsumTmpMinSize_ > 0U) && (statusBufSize >= cumsumTmpMinSize_);
    if (useCumSum) {
        expertTokenNumTensor_ = tokenNumBuf_.Get<float>();
    }
    uint32_t expertCountShape[] = {localExpertNum, epWorldSize_};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(expertTokenNumTensor_, expertCountTensor_, expertCountShape,
                                                         false);

    expertTokenNumsLocalTensor_ = tokenNumBuf_.Get<int64_t>();
    DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)), 0U, 0U,
                                                0U};
    LocalTensor<int32_t> expertTokenNumInt32Tensor = expertTokenNumTensor_.ReinterpretCast<int32_t>();
    if (expertTokenNumsType_ != 0) {
        PipeBarrier<PIPE_V>();
        Cast(expertTokenNumsLocalTensor_, expertTokenNumInt32Tensor, RoundMode::CAST_NONE, localExpertNum);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor_, expertTokenNumsCopyParams);
        return;
    }

    SetExpertPrefixSumA5(localExpertNum, useCumSum);
}
#else
// 更新tokenNumsOut tensor
template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SetExpertTokenNums()
{
    if (aivId_ != FIRST_CORE) {
        return;
    }

    uint32_t localExpertNum;
    if (isShareExpertRankFlag_) {
        localExpertNum = 1U;
    } else {
        localExpertNum = moeExpertNumPerRank_;
    }
    uint16_t localStatusSectionNum = static_cast<uint16_t>(localExpertNum * epWorldSize_);
    DataCopyParams allStatusCopyParams{localStatusSectionNum, 1,
                                       static_cast<uint16_t>((stateOffset_ - UB_ALIGN) / UB_ALIGN), 0};
    LocalTensor<float> allStatusTensorFp32 = statusBuf_.Get<float>();
    DataCopy(allStatusTensorFp32, windowInstatusFp32Tensor_, allStatusCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    int64_t expertTokenNumCumsum = 0;
    LocalTensor<int64_t> expertTokenNumsLocalTensor = tokenNumBuf_.Get<int64_t>();
    LocalTensor<float> expertTokenNumTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), 0);
    LocalTensor<float> workLocalTensor = workLocalBuf_.Get<float>();

    for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
        LocalTensor<float> expertStatusTensor = statusBuf_.GetWithOffset<float>(
            epWorldSize_ * UB_ALIGN / static_cast<uint32_t>(sizeof(float)), localExpertIdx * epWorldSize_ * UB_ALIGN);
        uint32_t mask = 2;
        SyncFunc<AscendC::HardEvent::S_V>();
        ReduceSum(expertTokenNumTensor, expertStatusTensor, workLocalTensor, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();

        int64_t expertTokenNum = static_cast<int64_t>(expertTokenNumTensor.ReinterpretCast<int32_t>().GetValue(0));
        expertTokenNumCumsum += expertTokenNum;
        if (expertTokenNumsType_ == 0) {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNumCumsum);
        } else {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNum);
        }
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)), 0U, 0U,
                                                0U};
    DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor, expertTokenNumsCopyParams);
}
#endif

template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        AlltoAllDispatch();
        SetStatus();
        WaitDispatch();
        LocalWindowCopy();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        SetExpertTokenNumsA5();
#else
        SetExpertTokenNums();
#endif
    }
}
} // namespace Mc2Kernel
#endif // MOE_DISTRIBUTE_DISPATCH_V2_H
