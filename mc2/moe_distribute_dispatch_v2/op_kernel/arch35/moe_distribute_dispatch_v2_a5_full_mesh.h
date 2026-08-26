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
 * \file moe_distribute_dispatch_v2_a5_full_mesh.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_A5_FULL_MESH_H
#define MOE_DISTRIBUTE_DISPATCH_V2_A5_FULL_MESH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/math/cumsum.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_dispatch_v2_tiling.h"
#include "../moe_distribute_dispatch_v2_quant.h"
#if __has_include("../../common/mc2_moe_context.h")
#include "../../common/mc2_moe_context.h"
#else
#include "../../../common/op_kernel/mc2_moe_context.h"
#endif
#include "../moe_distribute_v2_base.h"
#include "../check_winsize.h"
#if __has_include("../../common/moe_distribute_base.h")
#include "../../common/moe_distribute_base.h"
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif
#include "../hccl_context_holder.h"
#include "../mc2_moe_context_holder.h"
#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace MoeDistributeDispatchV2A5FullMeshImpl {
using namespace Mc2Kernel;
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL; // 384 * 2，0/1标识区偏移
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint64_t SPLIT_BLOCK_COUNT = 128UL; // 128 = SPLIT_BLOCK_SIZE / sizeof(float)
constexpr int32_t FULL_MESH_MAX_UB_SIZE = 190 * 1024;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_DATA_COUNT = 120U; // 120 = SPLIT_BLOCK_DATA_SIZE / sizeof(float)
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t CUMSUM_MAX_CORE_NUM = 8U;
// 按32核上限预留前缀和数据区（32*32*32=32KB）
constexpr uint64_t CUMSUM_WS_FLAG_OFFSET = 32UL * 1024UL;
constexpr uint32_t RUNPOS_CALCUMSUM = 2U;
constexpr uint32_t RUNPOS_CUMSUMFLAG = 3U;
constexpr uint32_t RUNPOS_ARRIVECNT = 4U;
constexpr uint8_t VALID_EVENT_FLAG_NUM = 8U;
constexpr uint8_t LOCAL_COPY_BUFFER_NUM = 2U;
constexpr uint32_t DATA_COPY_MAX_BLOCK_COUNT = 4095U;
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U; // 8 = UB_ALIGN / sizeof(float) = UB_ALIGN / sizeof(int32_t)
constexpr uint32_t DURATION_OFFSET = sizeof(int64_t) / sizeof(int32_t);

#define TemplateMC2A5FullMeshTypeClass \
    typename ContextHolder, typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define TemplateMC2A5FullMeshTypeFunc ContextHolder, XType, ExpandXOutType, QuantMode, IsSmoothScaleExist

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace Mc2Aclnn;
template <TemplateMC2A5FullMeshTypeClass>
class MoeDistributeDispatchV2A5FullMesh {
public:
    using XInType = typename std::conditional<
        (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value), uint8_t, XType>::type;
    using XOutType = typename std::conditional<(Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) ||
                                                   (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
                                               uint8_t, ExpandXOutType>::type;
    __aicore__ inline MoeDistributeDispatchV2A5FullMesh(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                GM_ADDR expertScales, GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expandScalesOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ExpIdsCopyAndMaskCal();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void InitElasticInfo();
    __aicore__ inline void SetDataStatus();
    __aicore__ inline void SetTilingData(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void CalValidBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalValidExpIdx(LocalTensor<bool> maskInputTensor);
    __aicore__ inline void GenerateGatherMaskTensor(uint32_t maskCnt);
    __aicore__ inline void MaskZeroComputeExpert(uint32_t maskCnt);
    __aicore__ inline void ZeroComputeExpertMaskCal();
    __aicore__ inline void SetTilingDataAndCal(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void ExpertActiveMaskInit();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void CalcSendTokenBufNum(TBuf<> &outBuf);
    __aicore__ inline void AllToAllDispatchA5(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void AllToAllDispatch();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline int32_t CalExpertCnt(uint32_t expertId, uint32_t maskCnt);
    __aicore__ inline void CalAndSendCntByExp();
    __aicore__ inline void CalAndSendCntByRankAndExpert();
    __aicore__ inline void SendExpertCntRange(uint32_t startExpertId, uint32_t endExpertId);
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor,
                                            LocalTensor<uint32_t> &gatherTmpTensor,
                                            LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId);
    __aicore__ inline void RunPosRecord(const uint32_t runPos);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlag(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyInAndOut(LocalTensor<XOutType> xTmpTensor, GM_ADDR wAddr, uint32_t index,
                                        uint32_t dstPosition, uint32_t arriveCount, uint32_t dataBufferId,
                                        bool isDataBufferReused);
    __aicore__ inline void WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum);
    __aicore__ inline uint32_t GetLocalWindowSrcDataBlockIdx(uint32_t srcExpertId, uint32_t localExpertNum);
    __aicore__ inline void ClearLocalWindowDataFlags(TBuf<> tBuf, uint32_t validNum, uint32_t localExpertNum);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void FillTriple(LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void TokenToExpertInQuant(GlobalTensor<XOutType> dstWinGMTensor,
                                                TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex,
                                                uint32_t toExpertId, uint32_t toExpertIndex);
    __aicore__ inline void TokenToExpert(GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                         uint32_t srcTokenIndex, uint32_t toExpertIndex);
    __aicore__ inline void RecordRankCommDuration(LocalTensor<int32_t> &performanceInfoTensor, uint64_t startTime);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindAddrByRankId(rankId, epRankIdOriginal_) + winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindStateAddrByRankId(rankId, epRankIdOriginal_) + dataState_ * WIN_STATE_OFFSET;
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
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<float> cumsumWsGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    LocalTensor<int32_t> statusTensor_;
    LocalTensor<int32_t> waitStatusTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIdsTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> performanceInfoTensor_;
    LocalTensor<int32_t> performanceFlagTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> expertScalesTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<XOutType> outTensor_;
    LocalTensor<XOutType> tempTensor_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;

    LocalTensor<float> flagReduceWorkTensor_;
    LocalTensor<float> flagRecvTensor_;

    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> elasticInfoBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> performanceInfoBuf_;
    TBuf<> performanceFlagBuf_;
    TBuf<> calBeginBuf_;
    TBuf<> calEndBuf_;
    GM_ADDR expandXOutGM_;
    GM_ADDR expandScalesOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR statusSpaceGM_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGM_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t syncFlagId_{0};
    uint8_t sendTokenBufNum_{0};
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    int32_t epRankId_{0};
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};    // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0}; // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0}; // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t hOutSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hAlignSize_{0};
    uint32_t scaleInBytes_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t scalesCount_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t startId_;
    uint32_t endId_;
    uint32_t sendNum_;
    uint32_t statusCntAlign_;
    uint32_t dataState_{0};
    uint32_t tBufRealSize_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool hasExpertScalesFlag_ = false;
    bool isPerformanceFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool isScalingDownFlag_ = false;
    uint64_t totalWinSize_{0};
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    uint32_t expertScaleAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t axisHCommu_{0};
    uint32_t hCommuSize_{0};
    uint32_t expertIdsBufSize_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedAllToAll_{0};
    uint32_t maxSize_{0};
    uint32_t expertIdsSize_{0};
    uint32_t globalBS_{0};
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};
    uint64_t totalUbSize_{0};

    // 统一上下文持有器：v2 走 HcclOpParam，v3 走 Mc2MoeContext
    ContextHolder ctx_;

    DataCopyParams hCopyParams_;
    DataCopyParams dataStateParams_{1U, sizeof(uint32_t), 0U, 0U};

    MoeDistributeDispatchV2Quant<XInType, ExpandXOutType, XOutType, QuantMode, IsSmoothScaleExist,
                                 QuantMode == PERGROUP_DYNAMIC_QUANT>
        quantInst_;
};

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::InitElasticInfo()
{
    uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
    uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
    elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
    DataCopyExtParams elasticInfoParams = {
        1U, static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)), 0U,
        0U, 0U};
    DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    isScalingDownFlag_ = elasticInfoTensor_.GetValue(0);
    if (isScalingDownFlag_) {
        epWorldSize_ = elasticInfoTensor_.GetValue(EP_WORLD_SIZE_IDX);
        sharedExpertRankNum_ = elasticInfoTensor_.GetValue(SHARE_RANK_NUM_IDX);
        moeExpertNum_ = elasticInfoTensor_.GetValue(MOE_NUM_IDX);
        epRankId_ = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epRankId_);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SetTilingData(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    hasExpertScalesFlag_ = tilingData->moeDistributeDispatchV2Info.hasExpertScales;
    isPerformanceFlag_ = tilingData->moeDistributeDispatchV2Info.isPerformance;
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    globalBS_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    scaleInBytes_ =
        tilingData->moeDistributeDispatchV2Info.scalesCol * tilingData->moeDistributeDispatchV2Info.scalesTypeSize;
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount;
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    maxSize_ = tilingData->moeDistributeDispatchV2Info.maxSizeForUbBuffer;
    totalUbSize_ = tilingData->moeDistributeDispatchV2Info.totalUbSize;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SetTilingDataAndCal(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    SetTilingData(tilingData);
    copyInAxisH_ = axisH_;
    copyOutAxisH_ = axisH_;
    if constexpr (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value ||
                  Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value) {
        copyOutAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
    if constexpr (Std::IsSame<XType, fp4x2_e2m1_t>::value || Std::IsSame<XType, fp4x2_e1m2_t>::value) {
        copyInAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
    if (hasElasticInfoFlag_) {
        InitElasticInfo();
    }
    isShareExpertRankFlag_ = (epRankId_ < sharedExpertRankNum_);
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    expertIdsCnt_ = axisBS_ * axisK_;
    hOutSize_ = copyOutAxisH_ * sizeof(XOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hOutSizeAlign_, scaleOutBytes_,
                         axisH_);
    // 因为与V2中预留给三元组的大小不一致，需要重新计算
    hOutSizeAlign_ = tokenQuantAlign_ * sizeof(int32_t) + UB_ALIGN;
    if (hasExpertScalesFlag_) {
        expertScaleAlign_ = tokenQuantAlign_ + EXPAND_IDX_INFO;
    }
    blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
    hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    axisHCommu_ = hCommuSize_ / sizeof(XOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hCommuSize_;
    rscvStatusNum_ = isShareExpertRankFlag_ ? epWorldSize_ : (epWorldSize_ * moeExpertNumPerRank_);
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    statusCntAlign_ = Ceil(totalExpertNum_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;
    aivUsedCumSum_ = totalExpertNum_ / 16; // 单核处理16个专家cnt发送
    aivUsedCumSum_ = (aivUsedCumSum_ == 0) ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= (aivNum_ / 2)) ? (aivNum_ / 2) : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= CUMSUM_MAX_CORE_NUM) ? CUMSUM_MAX_CORE_NUM : aivUsedCumSum_;
    // 确保每个核至少处理一个状态
    aivUsedCumSum_ = (aivUsedCumSum_ >= rscvStatusNum_) ? rscvStatusNum_ : aivUsedCumSum_;
    aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivUsedAllToAll_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivUsedAllToAll_ - sharedUsedAivNum_;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SetDataStatus()
{
    statusDataSpaceGM_ = ctx_.GetStatusDataSpaceGm();
    uint32_t epRankIdHccl = ctx_.GetEpRankId();
    uint32_t epWorldSizeHccl = ctx_.GetEpWorldSize();
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGM_ + FLAG_FIELD_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
    uint64_t hSizeAlignCombine = Ceil(axisH_ * COMBINE_IN_DATA_SIZE, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_SIZE;
    winDataSizeOffset_ =
        dataState_ * (totalWinSize_ / BUFFER_NUM) + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expandScalesOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_ = pipe;
    tpipe_->InitBuffer(calBeginBuf_, UB_ALIGN);
    aivId_ = GetBlockIdx();
    totalWinSize_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    ctx_.InitAndCheck(mc2Context, tilingData->moeDistributeDispatchV2Info.epWorldSize, totalWinSize_, tpipe_,
                      expandXOut);
    xGMTensor_.SetGlobalBuffer((__gm__ XInType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(elasticInfo));
    scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    SetTilingDataAndCal(tilingData);
    if (hasExpertScalesFlag_) {
        expertScalesGMTensor_.SetGlobalBuffer((__gm__ float *)expertScales);
        expandScalesOutGM_ = expandScalesOut;
    }
    if (isPerformanceFlag_) {
        performanceInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(performanceInfo));
    }
    SetDataStatus();
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCntWorkspaceGM_ = AscendC::GetUserWorkspace(workspaceGM);
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGM_));
    selfRankWinInGMTensor_.SetGlobalBuffer((__gm__ float *)(statusDataSpaceGM_));
    uint64_t cumsumWsBaseOffset = static_cast<uint64_t>(WORKSPACE_ELEMENT_OFFSET) * aivNum_ * aivNum_;
    cumsumWsGMTensor_.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)(recvCntWorkspaceGM_) + cumsumWsBaseOffset));
    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
    hCopyParams_ = {1U, static_cast<uint32_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U};
    dataStateParams_ = {1U, sizeof(uint32_t), 0U, 0U};
    expertIdsSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::FillTriple(
    LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;      // 0:epRankId index
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex; // 1:token index
    xOutTint32(tokenQuantAlign_ + 2) = k;          // 2:topK value index
    LocalTensor<float> xOutTfloat = xOutTensor.template ReinterpretCast<float>();
    if ((k < axisK_) && hasExpertScalesFlag_) {
        xOutTfloat(expertScaleAlign_) = expertScalesTensor_.GetValue(tokenIndex * axisK_ + k);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::TokenToExpertInQuant(
    GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex,
    uint32_t fillExpertIdx, uint32_t quantExpertIdx)
{
    DataCopyPadParams copyPadParams{true, 0U, 0U, 0U};
    LocalTensor<XInType> xInTensor = inQueue.AllocTensor<XInType>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XInType>();
    if constexpr (QuantMode > UNQUANT) {
        quantInst_.QuantProcess(tempTensor_, xInTensor, quantExpertIdx, scalesCount_, scalesGMTensor_);
    } else { // 非量化直转hifp8
        Cast(floatLocalTemp_, xInTensor, RoundMode::CAST_NONE, axisH_);
        Cast(tempTensor_, floatLocalTemp_, RoundMode::CAST_ROUND, axisH_);
    }
    inQueue.FreeTensor<XInType>(xInTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
    FillTriple(tempTensor_, srcTokenIndex, fillExpertIdx);
    SyncFunc<AscendC::HardEvent::S_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    LocalTensor<int32_t> tempTensorInt32 = tempTensor_.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 =
        (outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_]).template ReinterpretCast<int32_t>();
    // 64 = 256 / sizeof(int32_t) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32, tempTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    // 64：偏移前一次拷贝的256字节；56 = (480 - 256) / sizeof(int32_t)
    // 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[64], tempTensorInt32[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    DataCopy(dstWinGMTensor, outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_], axisHCommu_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    syncFlagId_++;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::TokenToExpert(
    GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex,
    uint32_t toExpertIndex)
{
    DataCopyPadParams copyPadParams{false, 0U, 0U, 0U};
    LocalTensor<XInType> xInTensor = inQueue.AllocTensor<XInType>();
    if constexpr (!IsSmoothScaleExist) {
        DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
    } else {
        DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
        DataCopyPadParams padParams = {true, 0, 0, 0};
        auto tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
        DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * copyInAxisH_], hCopyParams_, copyPadParams);
        DataCopyPad(xInTensor[Align32(copyInAxisH_)].template ReinterpretCast<uint8_t>(),
                    tmp[srcTokenIndex * scaleInBytes_], scaleInParams, padParams);
    }
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XInType>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    FillTriple(xInTensor, srcTokenIndex, toExpertIndex);
    SyncFunc<AscendC::HardEvent::S_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    LocalTensor<int32_t> xInTensorInt32 = xInTensor.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 =
        (outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_]).template ReinterpretCast<int32_t>();
    // 64 = 256 / sizeof(int32_t) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32, xInTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    // 64：偏移前一次拷贝的256字节；56 = (480 - 256) / sizeof(int32_t)
    // 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[64], xInTensorInt32[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    inQueue.FreeTensor<XInType>(xInTensor);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    DataCopy(dstWinGMTensor, outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_], axisHCommu_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    syncFlagId_++;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else if (aivId_ >= aivUsedAllToAll_) { // aiv中后面aivUsedCumSum_个核给cusum计算使用
        newAivId = aivId_ - aivUsedAllToAll_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_; // aivUsedAllToAll_中后面的核发送给共享专家
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

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SendToSharedExpert(
    TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * sendTokenBufNum_ / sizeof(float));
    PipeBarrier<PIPE_V>();
    // 分核
    uint32_t startTokenId, endTokenId, sendTokenNum;
    // 参数 activeMaskBsCnt_、sharedExpertNum_、sharedUsedAivNum_
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_;
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {
        return;
    }
    // 发送
    GlobalTensor<XOutType> dstWinGMTensor;
    // 计算目的共享专家卡在其所在共享专家组的id
    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_;
    syncFlagId_ = 0;
    for (int i = 0; i < sendTokenBufNum_; i++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t sendTokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        int32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }
        dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType *)(uint64_t(GetWindAddrByRankId(toRankId)) +
                                                           expertPerSizeOnWin_ * epRankId_ +
                                                           sendTokenIndex * hCommuSize_));
        uint32_t srcTokenIndex = sendTokenIndex;
        if (isExpertMaskFlag_) {
            srcTokenIndex = validBsIndexTensor_.GetValue(sendTokenIndex);
        }
        if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
            uint32_t fillExpertIdx = axisK_ + toSharedExpertIndex;
            uint32_t quantExpertIdx = toSharedExpertIndex;
            TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, fillExpertIdx, quantExpertIdx);
        } else {
            TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, axisK_ + toSharedExpertIndex);
        }
    }
    for (int i = 0; i < sendTokenBufNum_; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SendToMoeExpert(
    TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    // 按Token分核
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * sendTokenBufNum_ / sizeof(float));
    uint32_t validTokenNum = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    GlobalTensor<XOutType> dstWinGMTensor;

    int32_t dstTokenIdx = 0;
    syncFlagId_ = 0;

    for (int i = 0; i < sendTokenBufNum_; i++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }

    for (int32_t index = aivId_; index < validTokenNum; index += moeUsedAivNum_) {
        int32_t tokenId = index / axisK_;
        int32_t topKId = index % axisK_;
        int32_t expertId = validExpertIdsTensor_(index);
        if (expertId >= moeExpertNum_ || expertId < 0)
            continue;
        int32_t toRankId = expertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }

        CalTokenSendExpertCnt(expertId, index, dstTokenIdx);

        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ XOutType *)(uint64_t(GetWindAddrByRankId(toRankId)) +
                                expertPerSizeOnWin_ * ((epRankId_ + toRankId) % epWorldSize_ * moeExpertNumPerRank_ +
                                                       expertId % moeExpertNumPerRank_) +
                                dstTokenIdx * hCommuSize_));
        if (hasElasticInfoFlag_) {
            dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType *)(uint64_t(GetWindAddrByRankId(toRankId)) +
                                                               expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ +
                                                                                      expertId % moeExpertNumPerRank_) +
                                                               dstTokenIdx * hCommuSize_));
        }
        if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
            uint32_t quantExpertIdx = expertId + sharedExpertNum_;
            TokenToExpertInQuant(dstWinGMTensor, inQueue, tokenId, topKId, quantExpertIdx);
        } else {
            TokenToExpert(dstWinGMTensor, inQueue, tokenId, topKId);
        }
    }

    for (int i = 0; i < sendTokenBufNum_; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalcSendTokenBufNum(
    TBuf<> &outBuf)
{
    tpipe_->InitBuffer(calEndBuf_, UB_ALIGN);
    uint64_t beiginUbAddr = (calBeginBuf_.Get<uint8_t>()).GetPhyAddr();
    uint64_t endUbAddr = (calEndBuf_.Get<uint8_t>()).GetPhyAddr();
    uint64_t remainUbSize = totalUbSize_ - (endUbAddr - beiginUbAddr + UB_ALIGN);

    sendTokenBufNum_ = remainUbSize / hCommuSize_;
    if (sendTokenBufNum_ > VALID_EVENT_FLAG_NUM)
        sendTokenBufNum_ = VALID_EVENT_FLAG_NUM;
    if (sendTokenBufNum_ == 0) {
        return;
    }

    tpipe_->InitBuffer(outBuf, hCommuSize_ * sendTokenBufNum_);
    outTensor_ = outBuf.Get<XOutType>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::AllToAllDispatch()
{
    // 使用的全局参数
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<> tempBuf, outBuf, inQueueCleanBuf, calTempBuf;
    TBuf<> smoothScalesBuf, receiveDataCastFloatBuf;
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    AscendC::TBufPool<AscendC::TPosition::VECIN> tbufPool0, tbufPool1;
    tpipe_->InitBufPool(tbufPool0, BUFFER_NUM * hAlignSize_);
    tpipe_->InitBufPool(tbufPool1, BUFFER_NUM * hAlignSize_, tbufPool0);

    tbufPool0.InitBuffer(inQueue, BUFFER_NUM, hAlignSize_);
    tbufPool1.InitBuffer(inQueueCleanBuf, BUFFER_NUM * hAlignSize_);
    // MX以及PERGROUP量化在计算scales时每次搬入256字节数据，
    // 所以在token搬入前需要对空间填0，避免引入脏数据
    if constexpr ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT) || (QuantMode == MX_QUANT_CLIP)) {
        LocalTensor<uint8_t> inTensor8 = inQueueCleanBuf.Get<uint8_t>();
        Duplicate(inTensor8, QUANT_PADDING_VALUE, BUFFER_NUM * hAlignSize_);
    }
    uint32_t calTokenIdxBuffSize = Ceil(axisBS_ * axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    if (hasExpertScalesFlag_) {
        tpipe_->InitBuffer(expertScalesBuf_, expertIdsBufSize_);
    }
    bool needMaskCalFlag = (isTokenMaskFlag_ || isExpertMaskFlag_ || zeroComputeExpertNum_ != 0);
    if (needMaskCalFlag) {
        tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);
    }
    if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
        hOutAlignUbSize_ = Ceil(hOutSizeAlign_, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(tempBuf, hOutAlignUbSize_);
        tpipe_->InitBuffer(receiveDataCastFloatBuf, maxSize_);
        tpipe_->InitBuffer(smoothScalesBuf, maxSize_);
        tempTensor_ = tempBuf.Get<XOutType>();
        floatLocalTemp_ = receiveDataCastFloatBuf.Get<float>();
        smoothScalesTensor_ = smoothScalesBuf.Get<float>();
        dstExpBuf_ = receiveDataCastFloatBuf; // 内存复用
        subExpBuf_ = smoothScalesBuf;         // 内存复用
    } else if (needMaskCalFlag) {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_);
        tpipe_->InitBuffer(subExpBuf_, maxSize_);
    } else {
        tpipe_->InitBuffer(dstExpBuf_, calTokenIdxBuffSize);
        tpipe_->InitBuffer(subExpBuf_, calTokenIdxBuffSize);
    }
    tpipe_->InitBuffer(calTempBuf, calTokenIdxBuffSize);
    workLocalTensor_ = calTempBuf.Get<float>();
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf, dynamicScalesOutGMTensor_);
    ExpIdsCopyAndMaskCal();
    if (activeMaskBsCnt_ == 0) {
        return;
    }
    AllToAllDispatchA5(inQueue, outBuf);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::AllToAllDispatchA5(
    TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    CalcSendTokenBufNum(outBuf);
    if ((aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0)) {
        SendToSharedExpert(inQueue, outBuf);
    } else {
        SendToMoeExpert(inQueue, outBuf);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalTokenSendExpertCnt(
    uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    if (calCnt < axisK_) {
        curExpertCnt = 0;
        return;
    }
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline int32_t MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalExpertCnt(
    uint32_t expertId, uint32_t maskCnt)
{
    int32_t expertCnt = 0;
    if ((expertId < sharedExpertRankNum_) && (activeMaskBsCnt_ > 0)) { // 当前处理专家id为共享专家
        if (expertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
            expertCnt = activeMaskBsCnt_;
        }
    } else if (sendToMoeExpTokenCnt_ > 0) { // 当前处理卡为moe专家卡
        int32_t moeExpertId = expertId - sharedExpertRankNum_;
        CalTokenSendExpertCnt(moeExpertId, maskCnt, expertCnt);
    }
    return expertCnt;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalAndSendCntByExp()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    uint32_t maskCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    SplitToCore(totalExpertNum_, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum, false);
    if (startExpertId >= totalExpertNum_) {
        return;
    }
    // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    uint64_t mask[2] = {0x101010101010101, 0};
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1,
                       8); // 0x3F800000为float的1 8为一次操作8个block
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
        int32_t curExpertCnt = CalExpertCnt(curExpertId, maskCnt);
        // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        int32_t cntPosIndex = (curExpertId - startExpertId) * 8 + 1;
        statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstRankId = curExpertId;
        uint32_t offset = STATE_OFFSET * epRankId_;
        if (curExpertId >= sharedExpertRankNum_) {
            dstRankId = ((curExpertId - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((curExpertId - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * STATE_OFFSET);
        }
        if (isScalingDownFlag_) {
            dstRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
        }
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(curExpertId - startExpertId) * UB_ALIGN_DATA_COUNT],
                          UB_ALIGN_DATA_COUNT);
    }
    // reset操作前需确保前面操作完成
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SendExpertCntRange(
    uint32_t startExpertId, uint32_t endExpertId)
{
    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t expertId = startExpertId; expertId < endExpertId;) {
        bool isSharedExpert = expertId < sharedExpertRankNum_;
        uint32_t moeExpertId = isSharedExpert ? 0U : expertId - sharedExpertRankNum_;
        uint32_t dstRankId = isSharedExpert ? expertId : moeExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        uint32_t localExpertId = isSharedExpert ? 0U : moeExpertId % moeExpertNumPerRank_;
        uint32_t sendExpertNum = 1U;
        if (!isSharedExpert) {
            sendExpertNum = endExpertId - expertId;
            uint32_t expertNumLeftInRank = moeExpertNumPerRank_ - localExpertId;
            sendExpertNum = sendExpertNum < expertNumLeftInRank ? sendExpertNum : expertNumLeftInRank;
        }

        uint32_t targetRankId = dstRankId;
        if (isScalingDownFlag_) {
            targetRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
        }
        uint32_t offset = STATE_OFFSET * epRankId_ + localExpertId * epWorldSize_ * STATE_OFFSET;
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(targetRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopyParams cntCopyParams = {static_cast<uint16_t>(sendExpertNum), 1U, 0U,
                                        static_cast<uint16_t>(epWorldSize_ - 1)};
        DataCopy<int32_t>(rankGMTensor, statusTensor_[expertId * UB_ALIGN_DATA_COUNT], cntCopyParams);
        expertId += sendExpertNum;
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalAndSendCntByRankAndExpert()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    // 将(rank, localExpert)展平为expertId后分核，使同一rank可由多个核并行处理。
    SplitToCore(totalExpertNum_, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum, false);
    if (sendExpertNum == 0) {
        return;
    }

    uint32_t maskCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    uint64_t mask[2] = {0x101010101010101, 0};
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1,
                       8); // 0x3F800000为float的1 8为一次操作8个block
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t expertId = startExpertId; expertId < endExpertId; ++expertId) {
        int32_t expertCnt = CalExpertCnt(expertId, maskCnt);
        // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        int32_t cntPosIndex = expertId * UB_ALIGN_DATA_COUNT + 1;
        statusTensor_.SetValue(cntPosIndex, expertCnt);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SendExpertCntRange(startExpertId, endExpertId);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::BufferInit()
{
    uint32_t waitStatusBufSize = Ceil((recStatusNumPerCore_ * UB_ALIGN), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * moeExpertNumPerRank_ * sizeof(float);
    uint64_t gatherMaskOutSize =
        (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    uint64_t sumContinueAlignSize = Ceil((aivNum_ * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);  // recStatusNumPerCore_32对齐后大小  * 32B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);       // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);      // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, sumContinueAlignSize); // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);              // 96 B
    if (isPerformanceFlag_) {
        uint32_t performanceFlagSize = recStatusNumPerCore_ * sizeof(int32_t);
        uint32_t performanceFlagSizeAlign = Ceil(performanceFlagSize, UB_ALIGN) * UB_ALIGN;
        uint32_t performanceInfoSize = epWorldSizeOriginal_ * sizeof(int64_t);
        uint32_t performanceInfoSizeAlign = Ceil(performanceInfoSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(performanceInfoBuf_, performanceInfoSizeAlign);
        performanceInfoTensor_ = performanceInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceInfoTensor_, 0, performanceInfoSizeAlign / sizeof(int32_t));
        tpipe_->InitBuffer(performanceFlagBuf_, performanceFlagSizeAlign);
        performanceFlagTensor_ = performanceFlagBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceFlagTensor_, 0, performanceFlagSizeAlign / sizeof(int32_t));
        if (isScalingDownFlag_) {
            uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
            uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
            elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
            DataCopyExtParams elasticInfoParams = {
                1U,
                static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)),
                0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
            DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
        }
    }
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次操作256字节，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8); // 8 = 256 / 32
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::GatherSumRecvCnt(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2); // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家 recvCnt 收集到gatherMaskOutTensor
    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
               {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和，对inner要求32对齐
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // 把当前核的所有专家recv cnt之和写到状态区
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // 每个核把sumOfRecvCnt重复写 aivUsedCumSum_ 份
    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8); // 8 = 256 / 32
    // 每次处理256字节，8个datablock，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(cumsumWsGMTensor_[(newAivId * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], sumCoreFP32Tensor,
             sumIntriParams);
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::GetCumSum(
    LocalTensor<int32_t> &outLocal, uint32_t newAivId)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1),
                                  0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, aivUsedCumSum_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();

    // 获取状态区中每个核的recvCnt
    while (true) {
        DataCopy(sumLocalTensor, cumsumWsGMTensor_[(newAivId * UB_ALIGN) / sizeof(float)], sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask,
                   {1, static_cast<uint16_t>(aivUsedCumSum_), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));
        if (cumSumFlag == aivUsedCumSum_) {
            break;
        }
    }

    // 0核前面所有核recv cnt总和是0
    if (newAivId == 0) {
        outLocal.SetValue(0, 0);
    } else {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask,
                   {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }
    // 清除 flag 用于下次aivUsedCumSum_软同步
    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();
    // 一次处理256字节，8个datablock
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);
    // 64 = 256 / sizeof(float) 一次操作字节数，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(cumsumWsGMTensor_[(newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor, cleanParams);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::WaitDispatch()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1;
    gatherTmpTensor.SetValue(0, 1);
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    uint64_t performanceTimeStart = static_cast<uint64_t>(GetSystemCycle());
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
                 intriParams);
        if (isPerformanceFlag_) {
            RecordRankCommDuration(performanceInfoTensor_, performanceTimeStart);
        }
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    RunPosRecord(RUNPOS_CALCUMSUM); // 维测打点
    if (isPerformanceFlag_) {
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        SetAtomicMax<int32_t>();
        DataCopyExtParams performanceInfoCopyParams{1U, static_cast<uint32_t>(epWorldSizeOriginal_ * sizeof(int64_t)),
                                                    0U, 0U, 0U};
        DataCopyPad(performanceInfoGMTensor_, performanceInfoTensor_, performanceInfoCopyParams);
        SetAtomicNone();
    }
    // 清状态
    WaitDispatchClearStatus();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalRecvAndSetFlag()
{
    // check flag 用于 aivUsedCumSum_ 软同步并计算 aivUsedCumSum_ 个核各自的recvCount
    LocalTensor<int32_t> outCountLocal;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    GetCumSum(outCountLocal, newAivId);
    // 计算epRecvCnt
    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    waitStatusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = waitStatusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + 1);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer((__gm__ int32_t *)wAddr);
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);
    // 复制aivNum_份
    for (uint32_t index = 0; index < aivNum_; index++) {
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
    }
    uint8_t repeatTimes = Ceil(aivNum_, 8); // 一次处理256字节，8个datablock
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivNum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();
    // 每次处理256字节，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_MTE3>();
    DataCopy(cumsumWsGMTensor_[(CUMSUM_WS_FLAG_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor,
             sumIntriParams); // 软同步
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::RecordRankCommDuration(
    LocalTensor<int32_t> &performanceInfoTensor, uint64_t startTime)
{
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t endTime = static_cast<uint64_t>(GetSystemCycle());
    int32_t duration = static_cast<int32_t>((endTime - startTime) / CYCLES_PER_US);
    for (uint32_t i = 0; i < recStatusNumPerCore_; i++) {
        float statusFp32 = statusFp32Tensor_.GetValue(i * FLAG_OFFSET);
        int32_t performanceFlag = performanceFlagTensor_.GetValue(i);
        if (statusFp32 > float(0.5) && performanceFlag == 0) {
            performanceFlagTensor_.SetValue(i, 1);
            uint32_t fromLocalRankId = (startStatusIndex_ + i) % epWorldSize_;
            uint32_t fromRankId =
                isScalingDownFlag_ ?
                    elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + fromLocalRankId) :
                    fromLocalRankId;
            int32_t savedTime = performanceInfoTensor.GetValue(fromRankId * DURATION_OFFSET);
            int32_t newValue = (duration > savedTime) ? duration : savedTime;
            if (newValue != savedTime) {
                // 乘2 使用int32_t是因为atomicAdd不支持int64_t类型，这里只赋值到int64_t的低32位。
                performanceInfoTensor.SetValue(fromRankId * DURATION_OFFSET, duration);
            }
        }
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalCumSum()
{
    // 当前核计算并发送所负责专家的token总数
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);                                                   // BS * K * 4
    tpipe_->InitBuffer(subExpBuf_, maxSize_);                                                   // BS * K * 4
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);                                     // BS * K * 4
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    tpipe_->InitBuffer(statusBuf_, statusCntAlign_ * UB_ALIGN);
    if (hasExpertScalesFlag_) {
        tpipe_->InitBuffer(expertScalesBuf_, expertIdsBufSize_);
    }
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    statusTensor_ = statusBuf_.Get<int32_t>();
    ExpIdsCopyAndMaskCal();
    if (hasElasticInfoFlag_) {
        CalAndSendCntByExp();
    } else {
        CalAndSendCntByRankAndExpert();
    }
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);
    BufferInit();
    WaitDispatch();
    CalRecvAndSetFlag();
    // 使用newAivId为0的核进行计算
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::WaitCumSumFlag()
{
    // Check cumsum is finished
    int32_t cumSumFlag = 0;
    int32_t targetFlag = aivUsedCumSum_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (CUMSUM_WS_FLAG_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = aivUsedCumSum_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    while (true) {
        DataCopy(statusFp32Tensor_, cumsumWsGMTensor_[cumSumFlagOffset], aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    RunPosRecord(RUNPOS_CUMSUMFLAG); // 维测打点
    // Clean flag for next round
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(cumsumWsGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SetValidExpertInfo(
    uint32_t expInfoSize, uint32_t &validNum)
{
    // 获取cumSum
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();

    if (aivId_ == 0) {
        uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
        int64_t lastVal = 0;
        uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
        LocalTensor<int64_t> expertTokenNumsLocalTensor = tokenNumBuf_.Get<int64_t>();
        for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
            if (expertTokenNumsType_ == 0) {
                expertTokenNumsLocalTensor(localExpertIdx) =
                    int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ + epWorldSize_ - 1));
            } else {
                expertTokenNumsLocalTensor(localExpertIdx) =
                    int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ + epWorldSize_ - 1)) - lastVal;
                lastVal = int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ + epWorldSize_ - 1));
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)), 0U, 0U,
                                                    0U};
        DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor, expertTokenNumsCopyParams);
    }

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    // 从sendCnt中挑选当前有发送过来的卡的token数量
    for (uint32_t index = startId_; index < endId_; index++) {
        expertMapTensor_(validNum) = index;
        if (index == 0) {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index);
        } else {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index) - sendCntTensor_(index - 1);
        }
        if (expertLeftNumTensor_(validNum) != 0) {
            validNum += 1;
        }
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CheckDataArriveWithFlag(
    uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt)
{
    uint32_t flagNum = blockCntPerToken_ * uint32_t(copyCnt);
    DataCopyParams expFlagCopyParams{static_cast<uint16_t>(flagNum), 1U,
                                     static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE / UB_ALIGN), 0U};
    GlobalTensor<float> dataFlagGlobal;
    GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + srcExpDataIdx * expertPerSizeOnWin_ + beginIdx * hCommuSize_ +
                    SPLIT_BLOCK_DATA_SIZE;
    dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
    DataCopy(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<float> flagSumOutTensor = scalarBuf_.Get<float>();
    ReduceSum(flagSumOutTensor, flagRecvTensor_, flagReduceWorkTensor_, 1U, flagNum, 1U);
    SyncFunc<AscendC::HardEvent::V_S>();
    // aicore禁止uint->float的显式转换，借助浮点乘法隐式转换得到期望和
    float expectFlagSum = 1.0F * flagNum;
    return flagSumOutTensor.GetValue(0) == expectFlagSum ? static_cast<uint32_t>(copyCnt) : 0U;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CopyInAndOut(
    LocalTensor<XOutType> xTmpTensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount,
    uint32_t dataBufferId, bool isDataBufferReused)
{
    uint32_t hOutElemCount = hOutSize_ / sizeof(XOutType); // expandXOutGlobal申请每个token的GM Buffer空间大小
    GlobalTensor<XOutType> dataFlagGlobal, expandXOutGlobal;
    GlobalTensor<float> expertScaleInGlobal, expandScalesOutGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ XOutType *)(wAddr));
    expandXOutGlobal.SetGlobalBuffer((__gm__ XOutType *)(expandXOutGM_) + (dstPosition)*hOutElemCount);
    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * arriveCount),
                                      static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint16_t>(UB_ALIGN), 0};
    DataCopyExtParams scalesCopyParams{
        uint16_t(arriveCount), static_cast<uint32_t>(scaleOutBytes_),
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams tokenCopyParams{
        uint16_t(arriveCount), hOutSize_,
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{
        uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor.template ReinterpretCast<int32_t>();

    if (isDataBufferReused) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dataBufferId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(dataBufferId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(dataBufferId);
    }
    DataCopyPad(xTmpTensor, dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(XOutType)],
                srcTokenCopyParams, srcTokenPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(dataBufferId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(dataBufferId);
    quantInst_.CopyScalesToOut(dstPosition, scaleOutBytes_, xTmpTensor, scalesCopyParams);
    DataCopyPad(expandXOutGlobal, xTmpTensor, tokenCopyParams);
    DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO], xOutInt32Tensor[tokenQuantAlign_],
                expandIdxCopyParams);
    if (hasExpertScalesFlag_) {
        LocalTensor<float> xOutFloatTensor = xTmpTensor.template ReinterpretCast<float>();
        uint32_t expertScaleOffset = expertScaleAlign_ * sizeof(float);
        uint32_t expertScaleWinOffset =
            expertScaleOffset / SPLIT_BLOCK_DATA_SIZE * SPLIT_BLOCK_SIZE + expertScaleOffset % SPLIT_BLOCK_DATA_SIZE;
        expertScaleInGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            wAddr + expertFinishNumTensor_(index) * hCommuSize_ + expertScaleWinOffset));
        expandScalesOutGlobal.SetGlobalBuffer((__gm__ float *)(expandScalesOutGM_) + dstPosition);
        DataCopyExtParams expertScaleCopyInParams{uint16_t(arriveCount), static_cast<uint32_t>(sizeof(float)),
                                                  static_cast<uint32_t>(hCommuSize_ - sizeof(float)), 0U, 0U};
        DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0.0f};
        DataCopyExtParams expertScaleCopyOutParams{uint16_t(arriveCount), static_cast<uint32_t>(sizeof(float)), 0U, 0U,
                                                   0U};
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(dataBufferId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(dataBufferId);
        DataCopyPad(xOutFloatTensor, expertScaleInGlobal, expertScaleCopyInParams, expertScalePadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(dataBufferId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(dataBufferId);
        DataCopyPad(expandScalesOutGlobal, xOutFloatTensor, expertScaleCopyOutParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(dataBufferId);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline uint32_t
MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::GetLocalWindowSrcDataBlockIdx(uint32_t srcExpertId,
                                                                                                uint32_t localExpertNum)
{
    uint32_t srcDataBlockIdx = srcExpertId % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_;
    if (!(isShareExpertRankFlag_ || hasElasticInfoFlag_)) {
        srcDataBlockIdx = (srcExpertId + epRankId_) % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_;
    }
    return srcDataBlockIdx;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ClearLocalWindowDataFlags(
    TBuf<> tBuf, uint32_t validNum, uint32_t localExpertNum)
{
    uint32_t maxCleanRecordNum = 0;
    for (uint32_t index = 0; index < validNum; ++index) {
        uint32_t cleanRecordNum = expertFinishNumTensor_(index) * blockCntPerToken_;
        maxCleanRecordNum = cleanRecordNum > maxCleanRecordNum ? cleanRecordNum : maxCleanRecordNum;
    }

    uint32_t cleanRecordCapacity = tBufRealSize_ / UB_ALIGN;
    cleanRecordCapacity =
        cleanRecordCapacity > DATA_COPY_MAX_BLOCK_COUNT ? DATA_COPY_MAX_BLOCK_COUNT : cleanRecordCapacity;
    uint32_t cleanRecordNum = maxCleanRecordNum > cleanRecordCapacity ? cleanRecordCapacity : maxCleanRecordNum;
    if (cleanRecordNum == 0) {
        return;
    }
    LocalTensor<float> cleanUpTensor = tBuf.Get<float>();
    Duplicate<float>(cleanUpTensor, float(0), cleanRecordNum * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::V_MTE3>();

    GlobalTensor<float> cleanGlobal;
    DataCopyParams cleanUpParams{1U, 1U, 0U, SPLIT_BLOCK_DATA_SIZE / UB_ALIGN};
    for (uint32_t index = 0; index < validNum; ++index) {
        uint32_t srcExpertId = expertMapTensor_(index);
        uint32_t srcDataBlockIdx = GetLocalWindowSrcDataBlockIdx(srcExpertId, localExpertNum);
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
        cleanGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));

        uint32_t leftRecordNum = expertFinishNumTensor_(index) * blockCntPerToken_;
        uint32_t recordOffset = 0;
        while (leftRecordNum > 0) {
            uint32_t curCleanRecordNum = leftRecordNum > cleanRecordNum ? cleanRecordNum : leftRecordNum;
            cleanUpParams.blockCount = static_cast<uint16_t>(curCleanRecordNum);
            uint32_t flagIndex = recordOffset * SPLIT_BLOCK_COUNT + SPLIT_BLOCK_DATA_COUNT;
            DataCopy(cleanGlobal[flagIndex], cleanUpTensor, cleanUpParams);
            recordOffset += curCleanRecordNum;
            leftRecordNum -= curCleanRecordNum;
        }
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::WaitAndFormatOutput(
    TBuf<> tBuf, uint32_t validNum)
{
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpertId, dstPosition, arriveCount, copyCnt, srcDataBlockIdx;
    uint64_t packedTokenSize = static_cast<uint64_t>(blockCntPerToken_) * SPLIT_BLOCK_DATA_SIZE;
    uint64_t minUbSizePerToken =
        LOCAL_COPY_BUFFER_NUM * packedTokenSize + static_cast<uint64_t>(blockCntPerToken_) * (UB_ALIGN + sizeof(float));
    uint32_t maxCopyTokenCnt = static_cast<uint32_t>(tBufRealSize_ / minUbSizePerToken);
    uint32_t dataBufferSize = 0;
    uint32_t reduceWorkSize = 0;
    uint32_t flagRecvSize = 0;
    while (maxCopyTokenCnt > 0) {
        dataBufferSize = static_cast<uint32_t>(packedTokenSize * maxCopyTokenCnt);
        reduceWorkSize = Ceil(blockCntPerToken_ * maxCopyTokenCnt * sizeof(float), SIZE_ALIGN_256) * SIZE_ALIGN_256;
        flagRecvSize = blockCntPerToken_ * maxCopyTokenCnt * UB_ALIGN;
        uint64_t totalBufferSize =
            static_cast<uint64_t>(LOCAL_COPY_BUFFER_NUM) * dataBufferSize + reduceWorkSize + flagRecvSize;
        if (totalBufferSize <= tBufRealSize_) {
            break;
        }
        --maxCopyTokenCnt;
    }

    LocalTensor<XOutType> xTmpPingTensor = tBuf.GetWithOffset<XOutType>(dataBufferSize / sizeof(XOutType), 0);
    LocalTensor<XOutType> xTmpPongTensor =
        tBuf.GetWithOffset<XOutType>(dataBufferSize / sizeof(XOutType), dataBufferSize);
    uint32_t flagWorkOffset = LOCAL_COPY_BUFFER_NUM * dataBufferSize;
    flagReduceWorkTensor_ = tBuf.GetWithOffset<float>(reduceWorkSize / sizeof(float), flagWorkOffset);
    flagRecvTensor_ = tBuf.GetWithOffset<float>(flagRecvSize / sizeof(float), flagWorkOffset + reduceWorkSize);

    uint32_t copyBatchId = 0;
    while (true) {
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum;     // 轮询查询每个有效的index
            continue;
        }
        srcExpertId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ?
                      maxCopyTokenCnt :
                      expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx = GetLocalWindowSrcDataBlockIdx(srcExpertId, localExpertNum);
        arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, expertFinishNumTensor_(index), copyCnt);
        if (arriveCount == copyCnt) {
            dstPosition = srcExpertId != 0 ? sendCntTensor_(srcExpertId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
            uint32_t dataBufferId = copyBatchId % LOCAL_COPY_BUFFER_NUM;
            LocalTensor<XOutType> xTmpTensor = xTmpPingTensor;
            if (dataBufferId != 0) {
                xTmpTensor = xTmpPongTensor;
            }
            CopyInAndOut(xTmpTensor, wAddr, index, dstPosition, arriveCount, dataBufferId,
                         copyBatchId >= LOCAL_COPY_BUFFER_NUM);
            ++copyBatchId;
            // Update progress here; data flags are cleared in batches after all outputs are issued.
            expertFinishNumTensor_(index) += arriveCount;
            expertLeftNumTensor_(index) -= arriveCount;
            if (expertLeftNumTensor_(index) == 0) {
                finishNum++;
            }
        } else {
            index = (index + 1) % validNum;
        }
        if (validNum == finishNum) {
            break;
        }
    }

    uint32_t usedDataBufferNum = copyBatchId > LOCAL_COPY_BUFFER_NUM ? LOCAL_COPY_BUFFER_NUM : copyBatchId;
    for (uint32_t dataBufferId = 0; dataBufferId < usedDataBufferNum; ++dataBufferId) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dataBufferId);
    }
    ClearLocalWindowDataFlags(tBuf, validNum, localExpertNum);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::RunPosRecord(
    const uint32_t runPos)
{
    TBuf<> runPosBuf;
    tpipe_->InitBuffer(runPosBuf, UB_ALIGN);
    dataStateLocalTensor_ = runPosBuf.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, runPos);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams_); // 维测打点
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::LocalWindowCopy()
{
    // 分核负责源专家数量
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    // 软同步
    WaitCumSumFlag();
    if (sendNum_ == 0) {
        return;
    }
    // 连续化
    TBuf<> expertMapBuf, expertFinishBuf, expertLeftBuf, tBuf;
    uint32_t validNum = 0;
    uint32_t expInfoSize = Ceil(sendNum_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tBufRealSize_ = FULL_MESH_MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) -
                    (expInfoSize * 3);       // 3为expInfoSize大小buffer申请个数
    tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    SetValidExpertInfo(expInfoSize, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    SyncFunc<AscendC::HardEvent::V_S>();
    WaitAndFormatOutput(tBuf, validNum);
    RunPosRecord(RUNPOS_ARRIVECNT); // 维测打点
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask，当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    uint32_t axisBsAlignSize = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
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
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalValidBSCnt(
    LocalTensor<bool> maskStrideTensor)
{
    uint32_t mask = axisBS_;
    uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
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

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalValidExpIdx(
    LocalTensor<bool> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
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
                        Ceil(expertIdsCnt_, SIZE_ALIGN_256) * SIZE_ALIGN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ExpertActiveMaskInit()
{
    uint32_t axisBSAlign = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    tpipe_->InitBuffer(validBsIndexTBuf_, axisBSAlign);
    uint32_t validBufferSize = expertIdsSize_ > xActivateMaskSize ? expertIdsSize_ : xActivateMaskSize;
    tpipe_->InitBuffer(validExpertIndexBuf_, validBufferSize);
    validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ExpertActiveMaskCal()
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

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::MaskZeroComputeExpert(
    uint32_t maskCnt)
{
    LocalTensor<int32_t> expertsIndexTensor = dstExpBuf_.Get<int32_t>();
    int32_t maskTensorInt16Cnt = Ceil(expertIdsCnt_, UB_ALIGN / 2);
    LocalTensor<uint8_t> maskTensorInt8 = validExpertIndexBuf_.Get<uint8_t>();    // bs*k*1
    LocalTensor<uint32_t> maskTensorInt32 = validExpertIndexBuf_.Get<uint32_t>(); // bs*k*1
    LocalTensor<half> expertIdsTensorCast = subExpBuf_.Get<half>();               // bs*k*4
    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertNum_);

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(expertIdsTensorCast, validExpertIdsTensor_, RoundMode::CAST_NONE, expertIdsCnt_);
    Duplicate<uint32_t>(maskTensorInt32, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    uint32_t calcCnt = Ceil(expertIdsCnt_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    // 逐元素比较tensor中的元素和Scalar，比较结果为真时，输出对应比特位为1，
    // 否则为0，以筛掉零计算量专家
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
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, maskCnt, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::GenerateGatherMaskTensor(
    uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ZeroComputeExpertMaskCal()
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

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ExpIdsCopyAndMaskCal()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        ExpertActiveMaskInit();
    }

    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
    if (activeMaskBsCnt_ == 0) { // DataCopyPad不能拷贝0个
        return;
    }
    if (zeroComputeExpertNum_ != 0) {
        ZeroComputeExpertMaskCal();
    }

    Duplicate<int32_t>(validExpertIdsTensor_, -1, int32_t(expertIdsBufSize_ / sizeof(int32_t)));
    // 拷贝bs*k个专家id 到local  补齐到32 字节对齐  bs*k = 3*3 = 9 expertIdsAlignCnt= 16 补7个 填充-1
    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
        LocalTensor<float> tmpExpertIdsTensorFloat = subExpBuf_.Get<float>();
        LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTensor_.ReinterpretCast<uint8_t>();
        DataCopyExtParams expertIdsMaskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> expertIdsMaskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsMaskParams, expertIdsMaskCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        LocalTensor<float> validExpertIdsFloat = validExpertIdsTensor_.ReinterpretCast<float>();
        Select(validExpertIdsFloat, gatherMaskTensorInt8, tmpExpertIdsTensorFloat, static_cast<float>(-1),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, expertIdsCnt_);
        SyncFunc<AscendC::HardEvent::V_S>();
    } else {
        uint32_t expertIdsMask = activeMaskBsCnt_ * axisK_;
        uint32_t expertIdsAlignCnt = Ceil(expertIdsMask, BITS_PER_BYTE) * BITS_PER_BYTE;
        uint32_t rightPadding = expertIdsAlignCnt - expertIdsMask;
        // rightPadding字节数不能超过32，不能超过8个u32
        DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{true, 0U, uint8_t(rightPadding), -1};
        DataCopyExtParams expertIdsCntParams{1U, static_cast<uint32_t>(expertIdsMask * sizeof(uint32_t)), 0U, 0U,
                                             0U}; // 第二个参数blockLen 范围[1, 2097151] 不能为0
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
    if (hasExpertScalesFlag_) {
        expertScalesTensor_ = expertScalesBuf_.Get<float>();
        DataCopyExtParams expertScalesParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> expertScalesPadParams{false, 0U, 0U, 0U};
        DataCopyPad(expertScalesTensor_, expertScalesGMTensor_, expertScalesParams, expertScalesPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        if (aivId_ < aivUsedAllToAll_) {
            AllToAllDispatch(); // 前面核all2all发送
        } else {
            CalCumSum(); // 后面核计算前缀和，输出epRecvCnt/exportTokenNums
        }
        // localWindowCopy中包含reset操作，需确保前面操作完成
        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy(); // 本卡上专家数据连续化，输出expandX/scales/expandIdx
    }
}

} // namespace MoeDistributeDispatchV2A5FullMeshImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_A5_FULL_MESH_H
