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
 * \file moe_distribute_combine_v2_a5_mte.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_A5_MTE_H
#define MOE_DISTRIBUTE_COMBINE_V2_A5_MTE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "../moe_distribute_combine_v2_tiling.h"
#include "../moe_distribute_combine_v2_quant.h"
#if __has_include("../../common/mc2_moe_context.h")
#include "../../common/mc2_moe_context.h"
#include "../../common/moe_distribute_base.h"
#include "../../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h"
#include "../../moe_distribute_dispatch_v2/check_winsize.h"
#include "../../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#include "../../moe_distribute_dispatch_v2/moe_distribute_elastic.h"
#else
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/check_winsize.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_elastic.h"
#endif

#define FLOAT_OVERFLOW_MODE_CTRL 60
#define A5_MTE_FLOAT_OVERFLOW_MODE_CTRL 60
namespace MoeDistributeCombineV2A5MteImpl {
using namespace MoeDistributeV2Base;
using namespace Mc2Kernel;
using namespace Mc2Aclnn;
#define A5MteCombineTypeClass \
    typename ExpandXType, typename XType, typename ExpandIdxType, uint8_t QuantMode, bool HasAddRmsNorm
#define A5MteCombineTypeFunc ExpandXType, XType, ExpandIdxType, QuantMode, HasAddRmsNorm

constexpr uint32_t SPLIT_BLOCK_SIZE = 512U;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_FLAG_SIZE = 32U;
constexpr uint32_t SPLIT_BLOCK_FLAG_COUNT = SPLIT_BLOCK_FLAG_SIZE / sizeof(float);

using namespace AscendC;
template <A5MteCombineTypeClass>
class MoeDistributeCombineV2A5Mte {
public:
    __aicore__ inline MoeDistributeCombineV2A5Mte(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                GM_ADDR epSendCount, GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expertScales,
                                GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
                                GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
                                GM_ADDR performanceInfo, GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitInputAndOutput(GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expandX, GM_ADDR expertIds,
                                              GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales,
                                              GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
                                              GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                              GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR yOut,
                                              GM_ADDR rstdOut, GM_ADDR XOut);
    __aicore__ inline void InitCommContext(GM_ADDR mc2Context, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitAttrs(GM_ADDR mc2Context, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitTilingAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void AlltoAllBuffInitAndMaskCal();
    __aicore__ inline void AlltoAllCommBuffInit();
    __aicore__ inline void TokenMaskCalCnt();
    __aicore__ inline void ExpertMaskCalCnt();
    __aicore__ inline void SetWaitTpStatusAndDisPatch();
    __aicore__ inline void ExpertAlltoAllDispatchInnerCopyAdd(uint32_t toRankId, uint32_t tokenId, uint32_t topkId,
                                                              uint32_t tkIndex);
    __aicore__ inline uint32_t ExpertAlltoAllDispatchBatchCopyAdd(uint32_t tokenOffset, uint32_t currentTokenNum,
                                                                  uint32_t batchId);
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void ProcessConstantExpert(uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal);
    __aicore__ inline void AddConstantExpert(uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal,
                                             float alpha1Float, float alpha2Float);
    __aicore__ inline void ProcessCopyExpert(uint32_t tokenIndex, float scaleVal);
    __aicore__ inline void ProcessMoeExpert(uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal);
    __aicore__ inline uint32_t GetMoeExpertSlotCount(uint32_t tokenIndex, uint32_t topkId);
    __aicore__ inline uint32_t ProcessMoeExpertSlots(uint32_t tokenIndex, uint32_t beginSlotIdx, uint32_t slotCount,
                                                     uint32_t &index, uint32_t tokenLocalIdx);
    __aicore__ inline void ProcessSpecialExpert(uint32_t tokenIndex, uint32_t expertId, uint32_t &index);
    __aicore__ inline bool ProcessExpert(uint32_t tokenIndex, uint32_t processLen, uint32_t tokenLocalIdx,
                                         uint32_t &topkId, uint32_t &index);
    __aicore__ inline bool ProcessMoeExpertsLoop(uint32_t tokenIndex, uint32_t &topkId, uint32_t &index,
                                                 uint32_t tokenLocalIdx);
    __aicore__ inline bool ProcessSharedExpertsLoop(uint32_t tokenIndex, uint32_t tokenIndexOffset, uint32_t processLen,
                                                    uint32_t &topkId, uint32_t tokenLocalIdx);
    __aicore__ inline void AddSharedExpertX(uint32_t tokenIndex, uint32_t processLen);
    __aicore__ inline void ExpertScaleCopy(const uint32_t beginIndex, const uint32_t tokenPerAivNum,
                                           const uint32_t receiveAivNum);
    __aicore__ inline void CalConstExpertAlpha(GlobalTensor<ExpandXType> constExpertAlphaGM, uint32_t const_expert_idx,
                                               float &alphaFloat);
    __aicore__ inline void LocalWindowInit();
    __aicore__ inline void TokenInit(uint32_t bufferIndex, uint32_t &tokenLocalIdx, uint32_t &topkId, uint32_t &index);
    __aicore__ inline void ProcessToken(uint32_t bufferIndex, uint32_t &tokenLocalIdx, uint32_t &topkId,
                                        uint32_t &index);
    __aicore__ inline bool LocalWindowSplitCoreCal();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void DispatchBufferInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline uint32_t WaitDispatch(uint32_t tokenIndex, uint32_t slotIdx, uint32_t slotCount,
                                            uint32_t tokenLocalIdx);
    __aicore__ inline void PerformanceInfoPerToken(uint32_t tokenIndex, uint32_t slotIdx, uint32_t tokenLocalIdx);
    __aicore__ inline void ClearPackedTokenFlags(uint32_t tokenIndex);
    __aicore__ inline uint32_t CheckPackedFlagRangeArriveInner(GM_ADDR flagBaseAddr, uint16_t blockCount,
                                                               uint32_t flagFloatNum, uint32_t srcStrideBytes);
    __aicore__ inline uint32_t CheckPackedFlagRangeArrive(GM_ADDR flagBaseAddr, uint16_t blockCount,
                                                          uint32_t flagFloatNum, uint32_t srcStrideBytes);
    __aicore__ inline uint32_t CheckPackedTokenArrive(GM_ADDR rankGM, uint32_t slotCount);
    __aicore__ inline void AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol,
                                                LocalTensor<float> &x1TmpFloatLocal,
                                                LocalTensor<float> &x2TmpFloatLocal,
                                                LocalTensor<float> &addOutTmpFloatLocal,
                                                const DataCopyExtParams &copyExtParams,
                                                const DataCopyPadExtParams<XType> &copyPadExtParams);
    __aicore__ inline void AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol,
                                                    LocalTensor<float> &xFp32, LocalTensor<float> &sqx,
                                                    LocalTensor<ExpandXType> &gammaLocal,
                                                    const DataCopyExtParams &copyExtParams);
    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (isMc2Context_) {
            return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffsetEp_;
        }
        return Mc2Kernel::GetBaseWindAddrByRankId(epWinContext_, rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (isMc2Context_) {
            return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + winStatusOffset_;
        }
        return Mc2Kernel::GetBaseWindStateAddrByRankId(epWinContext_, rankId, epRankIdOriginal_) + winStatusOffset_;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<ExpandIdxType> elasticInfoGM_;
    GlobalTensor<int32_t> performanceInfoGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> residualXGM_;
    GlobalTensor<XType> gammaGM_;
    GlobalTensor<XType> yOutGlobal_;
    GlobalTensor<float> rstdOutGlobal_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> rowTmpGlobal_;
    GlobalTensor<ExpandXType> oriXGM_;
    GlobalTensor<ExpandXType> constExpertAlpha1GM_;
    GlobalTensor<ExpandXType> constExpertAlpha2GM_;
    GlobalTensor<ExpandXType> constExpertVGM_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    GM_ADDR epWindowGM_;
    GM_ADDR maskCalcWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;

    __gm__ Mc2MoeContext *mc2Context_{nullptr};

    LocalTensor<ExpandXType> expandXInTensor_;
    LocalTensor<XType> outTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> performanceInfoTensor_;
    LocalTensor<int32_t> firstRecordTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<XType> gammaLocal_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sendAivNum_{0};
    uint32_t recvAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    uint32_t epRankId_{0};
    uint32_t epRankIdOriginal_{0};
    uint32_t coreIdx_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t rankNumPerShareExpert_{0};
    uint32_t moeExpertPerRankNum_{0}; // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};          // moeExpertPerRankNum_ * epWorldSize_
    uint32_t bufferNum_{0};
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertOriginalNum_{0};
    uint32_t globalBS_{0};
    __gm__ Mc2Kernel::HcclOpParam *epWinContext_{nullptr};
    uint32_t bsKNum_{0};
    uint32_t startTokenId_{0};
    uint32_t sendCntNum_{0};
    uint32_t maxTokenNumInUB_{0};
    uint32_t ubSize_{0};
    uint32_t dataState_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winStatusOffset_{0};
    uint64_t totalWinSizeEp_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t performanceTimeStart_{0};
    uint32_t selfSendCnt_{0};
    uint32_t activeMaskAlignSize_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hExpandXAlignSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t tokenScaleCnt_{0};
    uint32_t commDataBytes_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t scaleNumAlignSize_{0};
    uint32_t flagRcvCount_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t performanceInfoSizeAlign_{0};
    uint32_t tokenNumPerCoreAlign_{0};
    uint32_t beginIndex_{0};
    uint32_t newAivId_{0};
    uint32_t tokenPerAivNum_{0};
    uint32_t receiveAivNum_{0};
    uint32_t tokenNumCompleted_{0};
    uint32_t statePos_{0};
    bool outputCopyPending_{false};
    uint32_t nextTokenLocalIdx_{0};
    uint32_t sumFloatBufOffset_{0};
    float armAvgFactor_{0.0};
    float epsilon_{0.0};

    TQue<QuePosition::VECIN, 1> expandXInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutPackageQueue_;
    TBuf<> quantResultBuf_;
    TQue<QuePosition::VECIN, 1> moeMainSumQueue_;
    TBuf<> expertScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> winTpSendCountFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> gammaBuf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;
    TBuf<> xActMaskTBuf_;
    TBuf<> xActMaskCastTBuf_;
    TBuf<> tokenTargetTBuf_;
    TBuf<> xActMaskSumTBuf_;
    TBuf<> expertMaskBuf_;
    TBuf<> performanceInfoBuf_;
    TBuf<> firstRecordBuf_;
    TBuf<> packedClearFlagBuf_;
    TBuf<> calBeginBuf_;
    TBuf<> calEndBuf_;
    TBuf<> opPosDfxBuf_;
    bool isInputTokenMaskFlag_ = false;
    bool isInputExpertMaskFlag_ = false;
    bool hasSharedExpertX_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isPerformanceFlag_ = false;
    bool hasExpertScalesFlag_ = false;
    bool isScalingDownFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool enableSpecialExpert_ = false;
    bool isMc2Context_ = false;

    // int8量化
    TBuf<> xAbsBuf_;
    TBuf<> xMaxBuf_;
    TBuf<> xScaleMulBuf_;

    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<XType> sendLocalTensor_;
    LocalTensor<half> tokenTargetTensor_;
    LocalTensor<bool> expertMaskTensor_;
    LocalTensor<float> expertScalesLocal_;
    uint32_t dispatchBufferNum_{1};
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;

    MoeDistributeCombineQuant<A5MteCombineTypeFunc> quantInst_;
    MoeDistributeElastic elasticInst_;
};

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::TokenMaskCalCnt()
{
    // 一维mask, 计算得到有效bs数量
    LocalTensor<bool> xActiveMaskTensor = xActMaskTBuf_.Get<bool>();
    LocalTensor<half> tempTensor = xActMaskCastTBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = xActMaskSumTBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams{1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertMaskCalCnt()
{
    // 二维mask, 挑选有效token
    LocalTensor<bool> maskStrideTensor = tokenBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = sumFloatBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams{static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)),
                                        0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN) * ALIGNED_LEN / sizeof(half);

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    DataCopyPad(maskStrideTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor_, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams activeMaskSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};
    Sum(tokenTargetTensor_, maskTempTensor, activeMaskSumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(tokenTargetTensor_.GetValue(0));
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::InitInputAndOutput(
    GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
    GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR yOut,
    GM_ADDR rstdOut, GM_ADDR XOut)
{
    if constexpr (HasAddRmsNorm) {
        residualXGM_.SetGlobalBuffer((__gm__ XType *)residualX);
        gammaGM_.SetGlobalBuffer((__gm__ XType *)gamma);
        yOutGlobal_.SetGlobalBuffer((__gm__ XType *)yOut);
        rstdOutGlobal_.SetGlobalBuffer((__gm__ float *)rstdOut);
    }
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t *)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float *)expertScales);
    xActiveMaskGM_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    sharedExpertXGM_.SetGlobalBuffer((__gm__ XType *)sharedExpertX);
    elasticInfoGM_.SetGlobalBuffer((__gm__ int32_t *)elasticInfo);
    oriXGM_.SetGlobalBuffer((__gm__ ExpandXType *)oriX);
    constExpertAlpha1GM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertAlpha1);
    constExpertAlpha2GM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertAlpha2);
    constExpertVGM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertV);
    performanceInfoGM_.SetGlobalBuffer((__gm__ int32_t *)performanceInfo);

    expandOutGlobal_.SetGlobalBuffer((__gm__ XType *)XOut);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::InitTilingAttrs(
    const MoeDistributeCombineV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    ubSize_ = tilingData->moeDistributeCombineV2Info.totalUbSize;
    globalBS_ = tilingData->moeDistributeCombineV2Info.globalBs;
    hasElasticInfoFlag_ = tilingData->moeDistributeCombineV2Info.hasElasticInfo;
    isPerformanceFlag_ = tilingData->moeDistributeCombineV2Info.isPerformance;
    hasExpertScalesFlag_ = tilingData->moeDistributeCombineV2Info.hasExpertScales;
    epWorldSizeOriginal_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    epRankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epWorldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    moeExpertPerRankNum_ = tilingData->moeDistributeCombineV2Info.moeExpertPerRankNum;
    totalWinSizeEp_ = tilingData->moeDistributeCombineV2Info.totalWinSizeEp;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineV2Info.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineV2Info.isExpertMask;
    hasSharedExpertX_ = tilingData->moeDistributeCombineV2Info.hasSharedExpertX;
    bufferNum_ = tilingData->moeDistributeCombineV2Info.bufferNum;
    zeroExpertNum_ = tilingData->moeDistributeCombineV2Info.zeroExpertNum;
    copyExpertNum_ = tilingData->moeDistributeCombineV2Info.copyExpertNum;
    constExpertNum_ = tilingData->moeDistributeCombineV2Info.constExpertNum;
    moeExpertNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    moeExpertOriginalNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertRankNum;
    enableSpecialExpert_ = (constExpertNum_ + zeroExpertNum_ + copyExpertNum_ > 0U);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::InitCommContext(
    GM_ADDR mc2Context, const MoeDistributeCombineV2TilingData *tilingData)
{
    uint32_t epRankIdHccl{0};
    uint32_t epWorldSizeHccl{0};
    if (isMc2Context_) {
        // Using Mc2Context instead of hccl context
        mc2Context_ = (__gm__ Mc2MoeContext *)mc2Context;
        epRankIdHccl = mc2Context_->epRankId;
        epWorldSizeHccl = tilingData->moeDistributeCombineV2Info.epWorldSize;
        statusDataSpaceGm_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankIdHccl]);
    } else {
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        epWinContext_ = (__gm__ Mc2Kernel::HcclOpParam *)contextGM0;
        statusDataSpaceGm_ = Mc2Kernel::GetStatusDataSpaceGm(epWinContext_);
        epRankIdHccl = Mc2Kernel::GetRankId(epWinContext_);
        epWorldSizeHccl = Mc2Kernel::GetRankDim(epWinContext_);
    }
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempEpRankId = 0; tempEpRankId < epWorldSize_; tempEpRankId++) {
        OOMCheckAddrRange<XType>((__gm__ XType *)(GetWinAddrByRankId(tempEpRankId, EP_DOMAIN)), totalWinSizeEp_);
        OOMCheckAddrRange<float>((__gm__ float *)(GetWinStateAddrByRankId(tempEpRankId, EP_DOMAIN)), STATE_SIZE);
    }
#endif
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGm_ + COMBINE_STATE_WIN_OFFSET + coreIdx_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::InitAttrs(
    GM_ADDR mc2Context, const MoeDistributeCombineV2TilingData *tilingData)
{
    InitTilingAttrs(tilingData);
    InitCommContext(mc2Context, tilingData);
    if (hasElasticInfoFlag_) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(elasticInfoGM_);
        isScalingDownFlag_ = elasticInfoGM_.GetValue(0);
        elasticInst_.SetElasticInitParams(tpipe_, elasticInfoGM_);
        elasticInst_.InitElasticInfo(isScalingDownFlag_, epWorldSize_, sharedExpertRankNum_, moeExpertNum_, epRankId_,
                                     moeExpertPerRankNum_);
    }
    sharedExpertNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }

    if (sharedExpertNum_ != 0U) { // 除零保护
        rankNumPerShareExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }

    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hFloatAlign256Size_ = Ceil(hFloatSize, ALIGNED_LEN) * ALIGNED_LEN;
    if constexpr ((QuantMode == INT8_COMM_QUANT) || (QuantMode == MXFP8_E5M2_COMM_QUANT) ||
                  (QuantMode == MXFP8_E4M3_COMM_QUANT)) {
        constexpr uint32_t kA5FusedQuantAlign = ALIGNED_LEN * INT8_DIVIVE;
        uint32_t hFloatA5FusedQuantSize = Ceil(hFloatSize, kA5FusedQuantAlign) * kA5FusedQuantAlign;
        hFloatAlign32Size_ = hFloatA5FusedQuantSize;
        hFloatAlign256Size_ = hFloatA5FusedQuantSize;
    }
    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
    commDataBytes_ = hExpandXTypeSize_;
    if constexpr (QuantMode > UNQUANT) {
        uint32_t scaleNum = 0U;
        quantInst_.QuantInit(scaleNum, hExpandXAlign32Size_, hExpandXAlignSize_, scaleNumAlignSize_,
                             hFloatAlign256Size_, tokenScaleCnt_, axisH_);
        commDataBytes_ = tokenScaleCnt_ * sizeof(ExpandXType);
    }
    blockCntPerToken_ = Ceil(commDataBytes_, SPLIT_BLOCK_DATA_SIZE);
    hAlignWinSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    bsKNum_ = axisBS_ * axisK_;
    if constexpr (HasAddRmsNorm) {
        armAvgFactor_ = tilingData->moeDistributeCombineV2Info.armAvgFactor;
        epsilon_ = tilingData->moeDistributeCombineV2Info.epsilon;
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR yOut,
    GM_ADDR rstdOut, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    maskCalcWorkspaceGM_ = workspaceGM + coreIdx_ * MASK_CALC_NEED_WORKSPACE;
    InitInputAndOutput(residualX, gamma, expandX, expertIds, expandIdx, epSendCount, expertScales, xActiveMask,
                       sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                       performanceInfo, yOut, rstdOut, XOut);
    if (tilingData->moeDistributeCombineV2Info.isMc2Context) {
        isMc2Context_ = true;
        mc2Context_ = (__gm__ Mc2MoeContext *)mc2Context;
    } else {
        auto realWinSize = Mc2Kernel::GetWinSize(epWinContext_);
        CheckWindowSize(totalWinSizeEp_, realWinSize, tpipe_, XOut);
    }
    InitAttrs(mc2Context, tilingData);

    PipeBarrier<PIPE_ALL>();
    // 当前win区划分为前后两半区，连续两次dispatch，切换半区
    winDataSizeOffsetEp_ =
        static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineV2Info.totalWinSizeEp / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET; // 前面的预留给dispatch使用
    epWindowGM_ = GetWinAddrByRankId(epRankIdOriginal_, EP_DOMAIN);
    if (isShareExpertRankFlag_) {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            epSendCountGM_[epWorldSize_ - 1]);
        selfSendCnt_ = epSendCountGM_(epWorldSize_ - 1);
    } else {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            epSendCountGM_[moeSendNum_ - 1]);
        selfSendCnt_ = epSendCountGM_(moeSendNum_ - 1);
    }
    SplitCoreCal();
    flagRcvCount_ = axisK_ + sharedExpertNum_;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::BuffInit()
{
    tpipe_->Reset(); // reset后 ctrl寄存器会复位为默认值
    // 单指令饱和模式
    AscendC::SetCtrlSpr<A5_MTE_FLOAT_OVERFLOW_MODE_CTRL, A5_MTE_FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_->InitBuffer(calBeginBuf_, UB_ALIGN);
    uint32_t tokenScaleAlign32Size = Ceil(tokenScaleCnt_ * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN;
    if constexpr (QuantMode > UNQUANT) {
        uint32_t packedDataBytes = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE;
        tokenScaleAlign32Size = tokenScaleAlign32Size > packedDataBytes ? tokenScaleAlign32Size : packedDataBytes;
        tpipe_->InitBuffer(xAbsBuf_, hFloatAlign256Size_);
        uint32_t hFloatAlign256Cnt = hFloatAlign256Size_ / sizeof(float);
        tpipe_->InitBuffer(xMaxBuf_, (hFloatAlign256Cnt / REDUCE_NUM) * sizeof(float));
        tpipe_->InitBuffer(xScaleMulBuf_, hFloatAlign256Size_);
        tpipe_->InitBuffer(winTpSendCountFloatBuf_, hFloatAlign32Size_);
        winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
        absFloatTensor_ = xAbsBuf_.Get<float>();
        reduceMaxFloatTensor_ = xMaxBuf_.Get<float>();
        scaleDupLocalTensor_ = xScaleMulBuf_.Get<float>();
        fp16CastTensor_ = xAbsBuf_.Get<half>();
        Duplicate(absFloatTensor_, float(0), hFloatAlign256Cnt);
        quantInst_.SetQuantInitParams(winTpSendCountFloatTensor_, fp16CastTensor_, absFloatTensor_,
                                      reduceMaxFloatTensor_, scaleDupLocalTensor_);
        tpipe_->InitBuffer(quantResultBuf_, tokenScaleAlign32Size);
    }
    if (isScalingDownFlag_) {
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
    }
    DispatchBufferInit();
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::DispatchBufferInit()
{
    uint32_t expandXInSize = hExpandXAlignSize_;
    uint32_t perDispatchBufBytes = 0U;
    if constexpr (QuantMode > UNQUANT) {
        perDispatchBufBytes = hExpandXAlignSize_ + hAlignWinSize_;
    } else {
        uint32_t packedDataBytes = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE;
        expandXInSize = hExpandXAlign32Size_ > packedDataBytes ? hExpandXAlign32Size_ : packedDataBytes;
        perDispatchBufBytes = expandXInSize + hAlignWinSize_;
    }

    // 按实际剩余UB计算每次可处理的token数，常见场景仍一次处理完，极端偏斜场景分片处理
    TBuf<> fixedEndBuf;
    tpipe_->InitBuffer(fixedEndBuf, UB_ALIGN);
    uint64_t beginUbAddr = calBeginBuf_.Get<uint8_t>().GetPhyAddr();
    uint64_t fixedEndUbAddr = fixedEndBuf.Get<uint8_t>().GetPhyAddr();
    uint64_t fixedUbSize = fixedEndUbAddr - beginUbAddr + UB_ALIGN;
    uint64_t reservedUbSize = perDispatchBufBytes + UB_ALIGN;
    uint64_t remainUbSize = (ubSize_ > fixedUbSize + reservedUbSize) ? (ubSize_ - fixedUbSize - reservedUbSize) : 0U;
    remainUbSize = remainUbSize / UB_ALIGN * UB_ALIGN;
    uint32_t tokenBufSize = EXPAND_IDX_INFO * sizeof(ExpandIdxType);
    maxTokenNumInUB_ = MIN(sendCntNum_, static_cast<uint32_t>(remainUbSize / tokenBufSize));
    if (maxTokenNumInUB_ == 0U) {
        maxTokenNumInUB_ = 1U;
    }
    uint32_t indexCountsBufSize = Ceil(maxTokenNumInUB_ * EXPAND_IDX_INFO * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indexCountsBuf_, indexCountsBufSize);

    tpipe_->InitBuffer(calEndBuf_, UB_ALIGN);
    uint64_t endUbAddr = calEndBuf_.Get<uint8_t>().GetPhyAddr();
    uint64_t usedUbSize = endUbAddr - beginUbAddr + UB_ALIGN;
    remainUbSize = (ubSize_ > usedUbSize) ? (ubSize_ - usedUbSize) : 0U;
    dispatchBufferNum_ = static_cast<uint32_t>(remainUbSize / perDispatchBufBytes);
    if (dispatchBufferNum_ == 0U) {
        dispatchBufferNum_ = 1U;
    }
    if (dispatchBufferNum_ > 8U) {
        dispatchBufferNum_ = 8U;
    }
    tpipe_->InitBuffer(expandXInQueue_, dispatchBufferNum_, expandXInSize);
    tpipe_->InitBuffer(xOutPackageQueue_, dispatchBufferNum_, hAlignWinSize_);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AlltoAllCommBuffInit()
{
    activeMaskBsCnt_ = axisBS_;
    activeMaskAlignSize_ = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size_;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size_;
    uint32_t bsKFloatAlign = Ceil(bsKNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size_ > bsKFloatAlign ? hFloatAlign256Size_ : bsKFloatAlign;
    if (isInputExpertMaskFlag_) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize_ * sizeof(half);
        maxSizeTokenBuf = (activeMaskAlignSize_ > hExpandXAlign32Size_ ? activeMaskAlignSize_ : hExpandXAlign32Size_);
        maxSizeRowTmpFloatBuf =
            (activeMaskAlignHalfSize > hFloatAlign32Size_ ? activeMaskAlignHalfSize : hFloatAlign32Size_);
    }
    // InitBuffer需要在tiling中计算ub总量
    tpipe_->InitBuffer(tokenBuf_, maxSizeTokenBuf);             // 16K 用于搬入输入token
    tpipe_->InitBuffer(rowTmpFloatBuf_, maxSizeRowTmpFloatBuf); // 32K 用于存储cast之后的fp32 token数据
    tpipe_->InitBuffer(mulBuf_, mulBufSize); // 32K buffer复用， 最大用于存储Brcb之后的token，需要256对齐
    tpipe_->InitBuffer(sumFloatBuf_, hFloatAlign32Size_ * bufferNum_); // 单buf按序处理，双buf交错处理
    uint32_t packedDataBytes = blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE;
    uint32_t rawPackedDataBytes = Ceil(hExpandXTypeSize_, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_DATA_SIZE;
    uint32_t moeQueueBytes = Ceil(packedDataBytes, UB_ALIGN) * UB_ALIGN;
    moeQueueBytes = (moeQueueBytes > rawPackedDataBytes) ? moeQueueBytes : rawPackedDataBytes;
    moeQueueBytes = (moeQueueBytes > hExpandXAlign32Size_) ? moeQueueBytes : hExpandXAlign32Size_;
    // 普通、拷贝和常量专家串行处理，复用同一个队列
    tpipe_->InitBuffer(moeMainSumQueue_, bufferNum_, moeQueueBytes);
    if constexpr (HasAddRmsNorm) {
        tpipe_->InitBuffer(gammaBuf_, hExpandXAlign32Size_);
        tpipe_->InitBuffer(reduceFp32Buf_, NUM_PER_REP_FP32 * sizeof(float) * 2);
        // H取最大值时，根据ReduceSum接口公式计算所需空间至少为64 * 2 = 128个元素
    }
    uint32_t clearFlagCount = flagRcvCount_ * blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
    uint32_t clearFlagBufSize = Ceil(clearFlagCount * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(packedClearFlagBuf_, clearFlagBufSize);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AlltoAllBuffInitAndMaskCal()
{
    tpipe_->Reset();
    AlltoAllCommBuffInit();
    if constexpr (QuantMode > UNQUANT) {
        tpipe_->InitBuffer(xAbsBuf_, scaleNumAlignSize_);
        fp16CastTensor_ = mulBuf_.Get<half>();
        absFloatTensor_ = rowTmpFloatBuf_.Get<float>();
        scaleDupLocalTensor_ = mulBuf_.Get<float>();
        scaleDivFloatTensor_ = xAbsBuf_.Get<float>();
        quantInst_.SetDeQuantInitParams(fp16CastTensor_, absFloatTensor_, scaleDupLocalTensor_, scaleDivFloatTensor_);
    }
    if (isInputTokenMaskFlag_) {
        axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xActMaskTBuf_, axisBsAlignSize_);
        tpipe_->InitBuffer(xActMaskCastTBuf_, axisBsAlignSize_ * sizeof(half));
        tpipe_->InitBuffer(xActMaskSumTBuf_, axisBsAlignSize_ * sizeof(half));
        TokenMaskCalCnt(); // 计算一维mask
    }
    if (isInputExpertMaskFlag_) {
        tpipe_->InitBuffer(tokenTargetTBuf_, Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(expertMaskBuf_, Ceil(axisBS_ * axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
        tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();
        ExpertMaskCalCnt(); // 计算二维mask
        expertMaskTensor_ = expertMaskBuf_.Get<bool>();
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskTensor_, xActiveMaskGM_, maskParams, maskCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        SyncFunc<AscendC::HardEvent::V_S>();
    }
    if (isPerformanceFlag_) {
        uint32_t performanceInfoSize = JUMP_WRITE * epWorldSizeOriginal_ * sizeof(int32_t);
        performanceInfoSizeAlign_ = Ceil(performanceInfoSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(performanceInfoBuf_, performanceInfoSizeAlign_);
        performanceInfoTensor_ = performanceInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceInfoTensor_, 0, JUMP_WRITE * epWorldSizeOriginal_);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::SplitCoreCal()
{
    uint32_t maxBs = globalBS_ / epWorldSizeOriginal_;
    recvAivNum_ = MIN(maxBs, MIN(32U, aivNum_ / 2U));
    if (recvAivNum_ == 0U) {
        recvAivNum_ = 1U;
    }
    sendAivNum_ = aivNum_ - recvAivNum_;
    if (coreIdx_ >= sendAivNum_) {
        sendCntNum_ = 0U;
        startTokenId_ = 0U;
        return;
    }
    // 对需要发送的token数平均分核，得到每个核上处理的卡的数量
    sendCntNum_ = selfSendCnt_ / sendAivNum_;
    uint32_t remainderRankNum = selfSendCnt_ % sendAivNum_;
    startTokenId_ = sendCntNum_ * coreIdx_;

    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::SetWaitTpStatusAndDisPatch()
{
    PipeBarrier<PIPE_ALL>();
    if (coreIdx_ >= selfSendCnt_) {
        return;
    }
    ExpertAlltoAllDispatchCopyAdd();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <A5MteCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertAlltoAllDispatchBatchCopyAdd(
    uint32_t tokenOffset, uint32_t currentTokenNum, uint32_t batchId)
{
    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    uint32_t batchBegin = batchId * recvAivNum_;
    uint32_t batchEnd = batchBegin + recvAivNum_;
    uint32_t batchSendCount = 0U;
    uint32_t permStride = 1U;
    if (likely(currentTokenNum > 2U)) {
        permStride = currentTokenNum / 2U + 1U;
        if (((currentTokenNum & 1U) == 0U) && ((permStride & 1U) == 0U)) {
            permStride++;
        }
    }
    uint32_t localIdx = ((epRankId_ * currentTokenNum) / epWorldSize_) % currentTokenNum;
    for (uint32_t loop = 0U; loop < currentTokenNum; loop++) {
        uint32_t baseOffset = localIdx * EXPAND_IDX_INFO;
        uint32_t tokenId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 1U));
        if ((tokenId >= batchBegin) && (tokenId < batchEnd)) {
            uint32_t rankIdExpandIdx = static_cast<uint32_t>(expandIdxLocal(baseOffset));
            uint32_t toRankId = rankIdExpandIdx;
            if (isScalingDownFlag_) {
                toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + rankIdExpandIdx);
            }
            uint32_t topkId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 2U));
            uint32_t tkIndex = startTokenId_ + tokenOffset + localIdx;
            ExpertAlltoAllDispatchInnerCopyAdd(toRankId, tokenId, topkId, tkIndex);
            batchSendCount++;
        }
        localIdx += permStride;
        if (localIdx >= currentTokenNum) {
            localIdx -= currentTokenNum;
        }
    }
    return batchSendCount;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertAlltoAllDispatchCopyAdd()
{
    if (sendCntNum_ == 0U) { // 空闲核，直接返回
        return;
    }

    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    uint32_t tokenLoopNum = Ceil(sendCntNum_, maxTokenNumInUB_);
    if (tokenLoopNum == 1U) {
        const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(sendCntNum_ * EXPAND_IDX_INFO * sizeof(uint32_t)),
                                          0U, 0U, 0U};
        const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId_ * EXPAND_IDX_INFO], bskParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }

    uint32_t maxBs = globalBS_ / epWorldSizeOriginal_;
    uint32_t batchTokenNum = Ceil(maxBs, recvAivNum_);
    uint32_t batchSendCount = 0U;
    for (uint32_t batchId = 0U; batchId < batchTokenNum && batchSendCount < sendCntNum_; batchId++) {
        uint32_t firstLoopId = (epRankId_ + batchId) % tokenLoopNum;
        for (uint32_t tokenLoopId = 0U; tokenLoopId < tokenLoopNum; tokenLoopId++) {
            uint32_t currentLoopId = firstLoopId + tokenLoopId;
            if (currentLoopId >= tokenLoopNum) {
                currentLoopId -= tokenLoopNum;
            }
            uint32_t tokenOffset = currentLoopId * maxTokenNumInUB_;
            uint32_t currentTokenNum = MIN(maxTokenNumInUB_, sendCntNum_ - tokenOffset);
            if (sendCntNum_ > maxTokenNumInUB_) {
                uint32_t copySize = currentTokenNum * EXPAND_IDX_INFO * sizeof(uint32_t);
                const DataCopyExtParams bskParams{1U, copySize, 0U, 0U, 0U};
                const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                DataCopyPad(expandIdxLocal, expandIdxGM_[(startTokenId_ + tokenOffset) * EXPAND_IDX_INFO], bskParams,
                            copyPadParams);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
            }
            batchSendCount += ExpertAlltoAllDispatchBatchCopyAdd(tokenOffset, currentTokenNum, batchId);
        }
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertAlltoAllDispatchInnerCopyAdd(
    uint32_t toRankId, uint32_t tokenId, uint32_t topkId, uint32_t tkIndex)
{
    uint32_t epOffset = tokenId * (axisK_ + sharedExpertNum_) + topkId;
    uint32_t tokenGMOffset = tkIndex * axisH_;
    GM_ADDR rankGM = GetWinAddrByRankId(toRankId, EP_DOMAIN) + epOffset * hAlignWinSize_;
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{true, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyExtParams xScaleCopyParams{1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U, 0U};
    GlobalTensor<float> dstPackedGlobal;
    dstPackedGlobal.SetGlobalBuffer((__gm__ float *)(rankGM));
    if constexpr (QuantMode > UNQUANT) {
        expandXInTensor_ = expandXInQueue_.AllocTensor<ExpandXType>();
        LocalTensor<uint8_t> singleByteTok = expandXInTensor_.template ReinterpretCast<uint8_t>();
        if constexpr ((QuantMode == MXFP8_E5M2_COMM_QUANT) || (QuantMode == MXFP8_E4M3_COMM_QUANT)) {
            Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(ExpandXType));
        }
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(expandXInTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
        expandXInQueue_.EnQue(expandXInTensor_);
        expandXInTensor_ = expandXInQueue_.DeQue<ExpandXType>();
        LocalTensor<XType> quantResultLT_ = quantResultBuf_.Get<XType>();
        quantInst_.QuantProcess(quantResultLT_, expandXInTensor_);
        expandXInQueue_.FreeTensor<ExpandXType>(expandXInTensor_);
        PipeBarrier<PIPE_V>();
        sendLocalTensor_ = xOutPackageQueue_.AllocTensor<ExpandXType>();
        LocalTensor<float> srcDataTensor = quantResultLT_.template ReinterpretCast<float>();
        LocalTensor<float> padFlagFloatTensor = sendLocalTensor_.template ReinterpretCast<float>();
        Duplicate(padFlagFloatTensor, float(1.0), hAlignWinSize_ / sizeof(float));
        PipeBarrier<PIPE_V>();
        Copy(padFlagFloatTensor, srcDataTensor, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
        Copy(padFlagFloatTensor[64], srcDataTensor[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
        xOutPackageQueue_.EnQue(sendLocalTensor_);
        sendLocalTensor_ = xOutPackageQueue_.DeQue<ExpandXType>();
        DataCopy(dstPackedGlobal, padFlagFloatTensor, blockCntPerToken_ * SPLIT_BLOCK_SIZE / sizeof(float));
        xOutPackageQueue_.FreeTensor<ExpandXType>(sendLocalTensor_);
    } else {
        expandXInTensor_ = expandXInQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(expandXInTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
        expandXInQueue_.EnQue(expandXInTensor_);
        expandXInTensor_ = expandXInQueue_.DeQue<ExpandXType>();
        outTensor_ = xOutPackageQueue_.AllocTensor<XType>();
        LocalTensor<float> srcfloat = expandXInTensor_.template ReinterpretCast<float>();
        LocalTensor<float> packedfloat = outTensor_.template ReinterpretCast<float>();
        Duplicate(packedfloat, float(1.0), hAlignWinSize_ / sizeof(float));
        PipeBarrier<PIPE_V>();
        Copy(packedfloat, srcfloat, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
        Copy(packedfloat[64], srcfloat[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
        xOutPackageQueue_.EnQue(outTensor_);
        outTensor_ = xOutPackageQueue_.DeQue<ExpandXType>();
        DataCopy(dstPackedGlobal, packedfloat, blockCntPerToken_ * SPLIT_BLOCK_SIZE / sizeof(float));
        xOutPackageQueue_.FreeTensor<XType>(outTensor_);
        expandXInQueue_.FreeTensor<ExpandXType>(expandXInTensor_);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CheckPackedFlagRangeArriveInner(
    GM_ADDR flagBaseAddr, uint16_t blockCount, uint32_t flagFloatNum, uint32_t srcStrideBytes)
{
    LocalTensor<float> flagTensor = rowTmpFloatBuf_.Get<float>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    uint32_t processedBlockCount = 0U;
    uint32_t processedFlagNum = 0U;
    while (processedBlockCount < blockCount) {
        GlobalTensor<float> dataFlagGlobal;
        dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(flagBaseAddr + processedBlockCount * SPLIT_BLOCK_SIZE));
        DataCopyExtParams expFlagCopyParams{1U, SPLIT_BLOCK_FLAG_SIZE, srcStrideBytes, 0U, 0U};
        DataCopyPadExtParams<float> expFlagPadParams{false, 0U, 0U, 0U};
        DataCopyPad(flagTensor, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < SPLIT_BLOCK_FLAG_COUNT; ++i) {
            if (flagTensor.GetValue(i) != float(1)) {
                return MIN(processedFlagNum + i, flagFloatNum);
            }
        }
        processedBlockCount++;
        processedFlagNum += SPLIT_BLOCK_FLAG_COUNT;
    }
    return MIN(processedFlagNum, flagFloatNum);
}

template <A5MteCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CheckPackedFlagRangeArrive(
    GM_ADDR flagBaseAddr, uint16_t blockCount, uint32_t flagFloatNum, uint32_t srcStrideBytes)
{
    if (flagFloatNum == 0U) {
        return 0U;
    }
    // 等待阶段复用计算scratch，避免额外申请flag检查buffer
    LocalTensor<float> flagTensor = rowTmpFloatBuf_.Get<float>();
    LocalTensor<uint8_t> compResultTensor = mulBuf_.Get<uint8_t>();
    LocalTensor<uint64_t> compResultU64Tensor = mulBuf_.Get<uint64_t>();
    const uint32_t flagPerBlock = SPLIT_BLOCK_FLAG_COUNT;
    const uint32_t maxFlagFloatNum = hFloatAlign32Size_ / sizeof(float);
    // CompareScalar按64个元素对齐，分段大小也按64个flag对齐，避免尾段补齐写出scratch
    const uint32_t maxBlockCount = (maxFlagFloatNum / 64U) * (64U / flagPerBlock);
    if (maxBlockCount == 0U) {
        return CheckPackedFlagRangeArriveInner(flagBaseAddr, blockCount, flagFloatNum, srcStrideBytes);
    }
    uint32_t processedBlockCount = 0U;
    uint32_t processedFlagNum = 0U;
    while (processedBlockCount < blockCount) {
        uint32_t currentBlockCount = MIN(maxBlockCount, static_cast<uint32_t>(blockCount) - processedBlockCount);
        uint32_t currentFlagFloatNum = currentBlockCount * flagPerBlock;
        uint32_t compareCount = Ceil(currentFlagFloatNum, 64U) * 64U;
        uint32_t compResultU64Num = Ceil(currentFlagFloatNum, 64U);
        Duplicate<float>(flagTensor, float(0), compareCount);
        PipeBarrier<PIPE_V>();
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(currentBlockCount), SPLIT_BLOCK_FLAG_SIZE,
                                            srcStrideBytes, 0U, 0U};
        DataCopyPadExtParams<float> expFlagPadParams{false, 0U, 0U, 0U};
        GlobalTensor<float> dataFlagGlobal;
        dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(flagBaseAddr + processedBlockCount * SPLIT_BLOCK_SIZE));
        DataCopyPad(flagTensor, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        CompareScalar(compResultTensor, flagTensor, float(1), AscendC::CMPMODE::EQ, compareCount);
        SyncFunc<AscendC::HardEvent::V_S>();
        for (uint32_t i = 0U; i < compResultU64Num; ++i) {
            uint64_t flagCompMask = compResultU64Tensor.GetValue(i);
            int64_t firstInvalidIdx = ScalarGetSFFValue<0>(flagCompMask);
            if (firstInvalidIdx == -1) {
                continue;
            }
            uint32_t arriveFlagNum = processedFlagNum + i * 64U + static_cast<uint32_t>(firstInvalidIdx);
            return MIN(arriveFlagNum, flagFloatNum);
        }
        processedBlockCount += currentBlockCount;
        processedFlagNum += currentFlagFloatNum;
    }
    return MIN(processedFlagNum, flagFloatNum);
}

template <A5MteCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CheckPackedTokenArrive(GM_ADDR rankGM,
                                                                                                     uint32_t slotCount)
{
    if ((blockCntPerToken_ == 0U) || (slotCount == 0U)) {
        return slotCount;
    }
    const uint32_t flagCountPerSlot = blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT;
    const uint32_t blockCount = slotCount * blockCntPerToken_;
    const uint32_t flagFloatNum = slotCount * flagCountPerSlot;
    uint32_t arriveFlagNum = CheckPackedFlagRangeArrive(
        rankGM + SPLIT_BLOCK_DATA_SIZE, static_cast<uint16_t>(blockCount), flagFloatNum, SPLIT_BLOCK_DATA_SIZE);
    return arriveFlagNum / flagCountPerSlot;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ClearPackedTokenFlags(uint32_t tokenIndex)
{
    if (blockCntPerToken_ == 0U) {
        return;
    }

    LocalTensor<float> flagTensor = packedClearFlagBuf_.Get<float>();
    uint32_t clearBlockCount = flagRcvCount_ * blockCntPerToken_;
    const DataCopyExtParams clearFlagParams{static_cast<uint16_t>(clearBlockCount), SPLIT_BLOCK_FLAG_SIZE, 0U,
                                            SPLIT_BLOCK_DATA_SIZE, 0U};
    GM_ADDR wAddr = (__gm__ uint8_t *)(epWindowGM_) + tokenIndex * flagRcvCount_ * hAlignWinSize_;
    GlobalTensor<float> dstFlagGlobal;
    dstFlagGlobal.SetGlobalBuffer((__gm__ float *)(wAddr + SPLIT_BLOCK_DATA_SIZE));
    DataCopyPad(dstFlagGlobal, flagTensor, clearFlagParams);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::PerformanceInfoPerToken(
    uint32_t tokenIndex, uint32_t slotIdx, uint32_t tokenLocalIdx)
{
    uint32_t fromRankId;
    if (slotIdx < axisK_) {
        uint32_t moeExpertId = expertIdsGM_.GetValue(tokenIndex * axisK_ + slotIdx);
        fromRankId = moeExpertId / moeExpertPerRankNum_ + sharedExpertRankNum_;
    } else {
        fromRankId = (slotIdx - axisK_) * rankNumPerShareExpert_ + epRankId_ % rankNumPerShareExpert_;
    }
    if (isScalingDownFlag_) {
        fromRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + fromRankId);
    }
    if (firstRecordTensor_.GetValue(tokenLocalIdx * flagRcvCount_ + slotIdx) == 0) {
        uint64_t performanceTimeCheck = static_cast<uint64_t>(GetSystemCycle());
        int32_t performanceTimeWait =
            static_cast<int32_t>((performanceTimeCheck - performanceTimeStart_) / CYCLES_PER_US);
        uint32_t fromRankIdTime = performanceInfoTensor_.GetValue(JUMP_WRITE * fromRankId);
        uint32_t maxTimeValue = (fromRankIdTime < performanceTimeWait) ? performanceTimeWait : fromRankIdTime;
        performanceInfoTensor_.SetValue(JUMP_WRITE * fromRankId, maxTimeValue);
        firstRecordTensor_.SetValue(tokenLocalIdx * flagRcvCount_ + slotIdx, 1);
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::WaitDispatch(uint32_t tokenIndex,
                                                                                           uint32_t slotIdx,
                                                                                           uint32_t slotCount,
                                                                                           uint32_t tokenLocalIdx)
{
    GM_ADDR wAddr = (__gm__ uint8_t *)(epWindowGM_) + (tokenIndex * flagRcvCount_ + slotIdx) * hAlignWinSize_;
    uint32_t arriveSlotCount = CheckPackedTokenArrive(wAddr, slotCount);
    if (arriveSlotCount == 0U) {
        return 0U;
    }
    if (isPerformanceFlag_) {
        for (uint32_t slotOffset = 0U; slotOffset < arriveSlotCount; slotOffset++) {
            PerformanceInfoPerToken(tokenIndex, slotIdx + slotOffset, tokenLocalIdx);
        }
    }

    return arriveSlotCount;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AddRmsNormAddCompute(
    uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol, LocalTensor<float> &x1TmpFloatLocal,
    LocalTensor<float> &x2TmpFloatLocal, LocalTensor<float> &addOutTmpFloatLocal,
    const DataCopyExtParams &copyExtParams, const DataCopyPadExtParams<XType> &copyPadExtParams)
{
    // 计算x + residual_x
    LocalTensor<XType> x2 = tokenBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(x2, residualXGM_[tokenIndex * axisH_ + tokenOffset], copyExtParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(x2TmpFloatLocal, x2, AscendC::RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    AscendC::Add(addOutTmpFloatLocal, x1TmpFloatLocal, x2TmpFloatLocal, numCol);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AddRmsNormRmsNormCompute(
    uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol, LocalTensor<float> &xFp32, LocalTensor<float> &sqx,
    LocalTensor<ExpandXType> &gammaLocal, const DataCopyExtParams &copyExtParams)
{
    // 计算rstd
    LocalTensor<float> reduceBufLocal = reduceFp32Buf_.Get<float>();
    Mul(sqx, xFp32, xFp32, numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqx, sqx, armAvgFactor_, numCol);
    PipeBarrier<PIPE_V>();
    ReduceSum(sqx, sqx, reduceBufLocal, numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon_, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqx, sqx, 1);
    Duplicate(reduceBufLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(reduceBufLocal, reduceBufLocal, sqx, 1);

    // rstd结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyParams copyParams;
    copyParams.blockLen = sizeof(uint32_t);
    copyParams.blockCount = 1;
    DataCopyPad(rstdOutGlobal_[tokenIndex * 1 + tokenOffset], reduceBufLocal, copyParams);

    // 计算y
    SyncFunc<AscendC::HardEvent::V_S>();
    float rstdValue = reduceBufLocal.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xFp32, xFp32, rstdValue, numCol);
    PipeBarrier<PIPE_V>();
    LocalTensor<XType> yLocal = rowTmpFloatBuf_.Get<XType>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
    PipeBarrier<PIPE_V>();
    Cast(xFp32, yLocal, RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol); // gamma_fp32 reuse sqx
    PipeBarrier<PIPE_V>();
    Mul(xFp32, xFp32, sqx, numCol);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);

    // y结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(yOutGlobal_[tokenIndex * axisH_ + tokenOffset], yLocal, copyExtParams);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::CalConstExpertAlpha(
    GlobalTensor<ExpandXType> constExpertAlphaGM, uint32_t const_expert_idx, float &alphaFloat)
{
    LocalTensor<ExpandXType> weightLocal = moeMainSumQueue_.AllocTensor<ExpandXType>();
    LocalTensor<float> weightFloatLocal = mulBuf_.Get<float>();
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};

    // 使用moeMainSumQueue_分配缓冲区来存储alpha1对应的权重矩阵Wc
    DataCopyPad(weightLocal, constExpertAlphaGM[const_expert_idx * axisH_], expandXCopyParams, copyPadExtParams);
    moeMainSumQueue_.EnQue(weightLocal);
    weightLocal = moeMainSumQueue_.DeQue<ExpandXType>();
    Cast(weightFloatLocal, weightLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    // 计算Wc * x
    Mul(weightFloatLocal, weightFloatLocal, rowTmpFloatLocal_, axisH_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams params{1, innerAlign, axisH_};
    Sum(weightFloatLocal, weightFloatLocal, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    alphaFloat = weightFloatLocal.GetValue(0);
    moeMainSumQueue_.FreeTensor<ExpandXType>(weightLocal);
}

// 处理常量专家
template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessConstantExpert(
    uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal)
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<ExpandXType> rowTmpLocal = tokenBuf_.Get<ExpandXType>();
    LocalTensor<float> alphaFloatLocal = tokenBuf_.Get<float>();
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    float alpha1Float = static_cast<float>(0.0);
    float alpha2Float = static_cast<float>(0.0);

    // 读取输入token
    DataCopyPad(rowTmpLocal, oriXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(rowTmpFloatLocal_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    // 计算Wc * x
    CalConstExpertAlpha(constExpertAlpha1GM_, const_expert_idx, alpha1Float);
    CalConstExpertAlpha(constExpertAlpha2GM_, const_expert_idx, alpha2Float);

    // 计算softmax(Wc * x)
    float maxAlphaFloat = (alpha1Float > alpha2Float) ? alpha1Float : alpha2Float;
    alphaFloatLocal.SetValue(0, alpha1Float - maxAlphaFloat);
    alphaFloatLocal.SetValue(1, alpha2Float - maxAlphaFloat);
    SyncFunc<AscendC::HardEvent::S_V>();
    Exp(alphaFloatLocal, alphaFloatLocal, 2);
    SyncFunc<AscendC::HardEvent::V_S>();
    float alphaSumFloat = alphaFloatLocal.GetValue(0) + alphaFloatLocal.GetValue(1);
    alpha1Float = alphaFloatLocal.GetValue(0) / alphaSumFloat;
    alpha2Float = alphaFloatLocal.GetValue(1) / alphaSumFloat;
    AddConstantExpert(tokenIndex, const_expert_idx, scaleVal, alpha1Float, alpha2Float);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AddConstantExpert(
    uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal, float alpha1Float, float alpha2Float)
{
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    // 使用moeMainSumQueue_分配缓冲区来存储常量专家向量v
    LocalTensor<float> constVFloatLocal = mulBuf_.Get<float>();
    LocalTensor<ExpandXType> const_v_ub = moeMainSumQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(const_v_ub, constExpertVGM_[const_expert_idx * axisH_], expandXCopyParams, copyPadExtParams);
    moeMainSumQueue_.EnQue(const_v_ub);
    const_v_ub = moeMainSumQueue_.DeQue<ExpandXType>();

    Cast(constVFloatLocal, const_v_ub, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeMainSumQueue_.FreeTensor<ExpandXType>(const_v_ub);

    // 计算 alpha1 * x + alpha2 * v
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(rowTmpFloatLocal_, rowTmpFloatLocal_, alpha1Float, axisH_);
    Muls(constVFloatLocal, constVFloatLocal, alpha2Float, axisH_);
    PipeBarrier<PIPE_V>();
    Add(rowTmpFloatLocal_, rowTmpFloatLocal_, constVFloatLocal, axisH_);
    PipeBarrier<PIPE_V>();

    if (hasExpertScalesFlag_) {
        Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
        PipeBarrier<PIPE_V>();
        Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, axisH_);
    } else {
        Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, axisH_);
    }
    PipeBarrier<PIPE_V>();
}

// 处理拷贝专家
template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessCopyExpert(uint32_t tokenIndex,
                                                                                            float scaleVal)
{
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    LocalTensor<ExpandXType> tmpUb = moeMainSumQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(tmpUb, oriXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    moeMainSumQueue_.EnQue(tmpUb);
    tmpUb = moeMainSumQueue_.DeQue<ExpandXType>();

    Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeMainSumQueue_.FreeTensor<ExpandXType>(tmpUb);

    if (hasExpertScalesFlag_) {
        Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
        PipeBarrier<PIPE_V>();
        Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, axisH_);
    } else {
        Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, axisH_);
    }
    PipeBarrier<PIPE_V>();
}

// 处理Moe专家
template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessMoeExpert(uint32_t tokenIndexOffset,
                                                                                           uint32_t topkId,
                                                                                           float scaleVal)
{
    uint32_t processLen = axisH_;
    const DataCopyExtParams xScaleCopyParams{static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE,
                                             SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE,
                                              SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};

    GM_ADDR wAddr = (__gm__ uint8_t *)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
    rowTmpGlobal_.SetGlobalBuffer((__gm__ XType *)wAddr);
    LocalTensor<XType> tmpUb = moeMainSumQueue_.AllocTensor<XType>();
    if constexpr (QuantMode > UNQUANT) {
        DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
    } else {
        DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
    }
    moeMainSumQueue_.EnQue(tmpUb);
    tmpUb = moeMainSumQueue_.DeQue<XType>();
    LocalTensor<XType> outLocalTensor = fp16CastTensor_.template ReinterpretCast<XType>();
    if constexpr (QuantMode > UNQUANT) {
        if (hasExpertScalesFlag_) {
            quantInst_.DeQuantProcess(tmpUb, outLocalTensor, rowTmpFloatLocal_, sumFloatBufLocal_, scaleVal);
        } else {
            quantInst_.DeQuantProcessWithoutExpertScale(tmpUb, outLocalTensor, rowTmpFloatLocal_, sumFloatBufLocal_);
        }
    } else {
        Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
        PipeBarrier<PIPE_V>();
        if (hasExpertScalesFlag_) {
            AscendC::Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, processLen);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, processLen);
        } else {
            Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, processLen);
        }
    }
    moeMainSumQueue_.FreeTensor<XType>(tmpUb);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ExpertScaleCopy(const uint32_t beginIndex,
                                                                                          const uint32_t tokenPerAivNum,
                                                                                          const uint32_t receiveAivNum)
{
    uint32_t expertScaleCntPerCore = tokenPerAivNum * axisK_;
    if (hasExpertScalesFlag_) {
        tpipe_->InitBuffer(expertScalesBuf_, Ceil(expertScaleCntPerCore * sizeof(float), UB_ALIGN) * UB_ALIGN);
        expertScalesLocal_ = expertScalesBuf_.Get<float>();
        const DataCopyExtParams tokenScaleParams{
            static_cast<uint16_t>(tokenPerAivNum), static_cast<uint32_t>(axisK_ * sizeof(float)),
            static_cast<uint32_t>((receiveAivNum - 1U) * axisK_ * sizeof(float)), 0U, 0U};
        const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
        DataCopyPad<float, PaddingMode::Compact>(expertScalesLocal_, expertScalesGM_[beginIndex * axisK_],
                                                 tokenScaleParams, copyPadFloatParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::GetMoeExpertSlotCount(uint32_t tokenIndex,
                                                                                                    uint32_t topkId)
{
    uint32_t slotCount = axisK_ - topkId;
    if (!isInputExpertMaskFlag_ && !enableSpecialExpert_) {
        return slotCount;
    }
    slotCount = 1U;
    while (topkId + slotCount < axisK_) {
        if (isInputExpertMaskFlag_ && !expertMaskTensor_.GetValue(tokenIndex * axisK_ + topkId + slotCount)) {
            break;
        }
        if (enableSpecialExpert_) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                expertIdsGM_[tokenIndex * axisK_ + topkId + slotCount]);
            uint32_t nextExpertId = expertIdsGM_.GetValue(tokenIndex * axisK_ + topkId + slotCount);
            if (nextExpertId >= moeExpertOriginalNum_) {
                break;
            }
        }
        slotCount++;
    }
    return slotCount;
}

template <A5MteCombineTypeClass>
__aicore__ inline uint32_t MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessMoeExpertSlots(
    uint32_t tokenIndex, uint32_t beginSlotIdx, uint32_t slotCount, uint32_t &index, uint32_t tokenLocalIdx)
{
    uint32_t tokenIndexOffset = tokenIndex * (axisK_ + sharedExpertNum_);
    uint32_t slotIdx = beginSlotIdx;
    uint32_t arriveSlotCount = WaitDispatch(tokenIndex, slotIdx, slotCount, tokenLocalIdx);
    uint32_t endSlotIdx = slotIdx + arriveSlotCount;
    for (; slotIdx < endSlotIdx; slotIdx++) {
        float scaleVal = 0.0;
        if (hasExpertScalesFlag_) {
            scaleVal = expertScalesLocal_.GetValue(index);
        }
        ProcessMoeExpert(tokenIndexOffset, slotIdx, scaleVal);
        index++;
    }
    return arriveSlotCount;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessSpecialExpert(uint32_t tokenIndex,
                                                                                               uint32_t expertId,
                                                                                               uint32_t &index)
{
    float scaleVal = 0.0;
    if (hasExpertScalesFlag_) {
        scaleVal = expertScalesLocal_.GetValue(index);
    }
    if (expertId < moeExpertOriginalNum_ + zeroExpertNum_) {
        // 零专家不需要任何操作
        index++;
    } else if (expertId < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_) {
        ProcessCopyExpert(tokenIndex, scaleVal);
        index++;
    } else if (expertId < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_ + constExpertNum_) {
        uint32_t const_expert_idx = expertId - (moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_);
        ProcessConstantExpert(tokenIndex, const_expert_idx, scaleVal);
        index++;
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessMoeExpertsLoop(uint32_t tokenIndex,
                                                                                                uint32_t &topkId,
                                                                                                uint32_t &index,
                                                                                                uint32_t tokenLocalIdx)
{
    while (topkId < axisK_) {
        if (isInputExpertMaskFlag_ && !expertMaskTensor_.GetValue(tokenIndex * axisK_ + topkId)) {
            index++;
            topkId++;
            continue;
        }

        uint32_t expert_id = 0U;
        if (enableSpecialExpert_) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                expertIdsGM_[tokenIndex * axisK_ + topkId]);
            expert_id = expertIdsGM_.GetValue(tokenIndex * axisK_ + topkId);
        }
        if (!enableSpecialExpert_ || (expert_id < moeExpertOriginalNum_)) {
            uint32_t slotCount = GetMoeExpertSlotCount(tokenIndex, topkId);
            uint32_t arriveSlotCount = ProcessMoeExpertSlots(tokenIndex, topkId, slotCount, index, tokenLocalIdx);
            topkId += arriveSlotCount;
            if (arriveSlotCount < slotCount) {
                return false;
            }
            continue;
        }

        ProcessSpecialExpert(tokenIndex, expert_id, index);
        topkId++;
    }
    return true;
}

template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessSharedExpertsLoop(
    uint32_t tokenIndex, uint32_t tokenIndexOffset, uint32_t processLen, uint32_t &topkId, uint32_t tokenLocalIdx)
{
    GM_ADDR wAddr;
    const DataCopyExtParams xScaleCopyParams{static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE,
                                             SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{static_cast<uint16_t>(blockCntPerToken_), SPLIT_BLOCK_DATA_SIZE,
                                              SPLIT_BLOCK_FLAG_SIZE, 0U, 0U};
    LocalTensor<XType> tmpUb;

    uint32_t endSlotIdx = axisK_ + sharedExpertNum_;
    if (topkId >= endSlotIdx) {
        return true;
    }
    uint32_t arriveSlotCount = WaitDispatch(tokenIndex, topkId, endSlotIdx - topkId, tokenLocalIdx);
    uint32_t readyEndSlotIdx = topkId + arriveSlotCount;
    for (; topkId < readyEndSlotIdx; topkId++) {
        wAddr = (__gm__ uint8_t *)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
        rowTmpGlobal_.SetGlobalBuffer((__gm__ XType *)wAddr);
        tmpUb = moeMainSumQueue_.AllocTensor<XType>();

        if constexpr (QuantMode > UNQUANT) {
            DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
        } else {
            DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
        }
        moeMainSumQueue_.EnQue(tmpUb);
        tmpUb = moeMainSumQueue_.DeQue<XType>();

        LocalTensor<XType> outLocalTensor = fp16CastTensor_.template ReinterpretCast<XType>();
        if constexpr (QuantMode > UNQUANT) {
            quantInst_.DeQuantProcess(tmpUb, outLocalTensor, rowTmpFloatLocal_, sumFloatBufLocal_, 1.0f);
        } else {
            Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, processLen);
            PipeBarrier<PIPE_V>();
        }
        moeMainSumQueue_.FreeTensor<XType>(tmpUb);
    }
    return topkId == endSlotIdx;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::AddSharedExpertX(uint32_t tokenIndex,
                                                                                           uint32_t processLen)
{
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    LocalTensor<XType> rowTmpLocal = tokenBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(rowTmpLocal, sharedExpertXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(rowTmpFloatLocal_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, processLen);
}

template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessExpert(
    uint32_t tokenIndex, uint32_t processLen, uint32_t tokenLocalIdx, uint32_t &topkId, uint32_t &index)
{
    uint32_t tokenIndexOffset = tokenIndex * (axisK_ + sharedExpertNum_);

    // 按原有专家顺序处理，保证累加顺序不变
    if (!ProcessMoeExpertsLoop(tokenIndex, topkId, index, tokenLocalIdx)) {
        return false;
    }
    if (!ProcessSharedExpertsLoop(tokenIndex, tokenIndexOffset, processLen, topkId, tokenLocalIdx)) {
        return false;
    }

    if (hasSharedExpertX_) {
        AddSharedExpertX(tokenIndex, processLen);
    }
    return true;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::LocalWindowInit()
{
    tpipe_->InitBuffer(opPosDfxBuf_, UB_ALIGN);
    dataStateLocalTensor_ = opPosDfxBuf_.Get<uint32_t>();
    rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    sumFloatBufLocal_ = sumFloatBuf_.Get<float>();
    if constexpr (HasAddRmsNorm) {
        const DataCopyPadExtParams<XType> copyPadXTypeParams{false, 0U, 0U, 0U};
        const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
        gammaLocal_ = gammaBuf_.Get<XType>();
        DataCopyPad(gammaLocal_, gammaGM_, expandXCopyParams, copyPadXTypeParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }
    ExpertScaleCopy(beginIndex_ + newAivId_, tokenPerAivNum_, receiveAivNum_);
    if (isScalingDownFlag_) {
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
    }
    if (isPerformanceFlag_) {
        uint32_t tokenNumPerCore = tokenPerAivNum_ * flagRcvCount_ * sizeof(int32_t);
        tokenNumPerCoreAlign_ = Ceil(tokenNumPerCore, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(firstRecordBuf_, tokenNumPerCoreAlign_);
        firstRecordTensor_ = firstRecordBuf_.Get<int32_t>();
        Duplicate<int32_t>(firstRecordTensor_, static_cast<int32_t>(0), tokenPerAivNum_ * flagRcvCount_);
    }
    LocalTensor<float> flagTensor = packedClearFlagBuf_.Get<float>();
    Duplicate<float>(flagTensor, float(0), flagRcvCount_ * blockCntPerToken_ * SPLIT_BLOCK_FLAG_COUNT);
    SyncFunc<AscendC::HardEvent::V_S>();
    performanceTimeStart_ = static_cast<uint64_t>(GetSystemCycle());
    outputCopyPending_ = false;
    nextTokenLocalIdx_ = 0U;
    sumFloatBufOffset_ = hFloatAlign32Size_ / sizeof(float);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::TokenInit(uint32_t bufferIndex,
                                                                                    uint32_t &tokenLocalIdx,
                                                                                    uint32_t &topkId, uint32_t &index)
{
    if (tokenLocalIdx < tokenPerAivNum_) {
        return;
    }
    uint32_t tokenIndex = 0U;
    while (nextTokenLocalIdx_ < tokenPerAivNum_) {
        tokenLocalIdx = nextTokenLocalIdx_++;
        tokenIndex = beginIndex_ + newAivId_ + tokenLocalIdx * receiveAivNum_;
        if (!isInputExpertMaskFlag_) {
            break;
        }
        bool validToken = false;
        for (uint32_t slotIdx = 0U; slotIdx < axisK_; slotIdx++) {
            if (expertMaskTensor_.GetValue(tokenIndex * axisK_ + slotIdx)) {
                validToken = true;
                break;
            }
        }
        if (validToken) {
            break;
        }
        tokenNumCompleted_++;
        tokenLocalIdx = tokenPerAivNum_;
    }
    if (tokenLocalIdx >= tokenPerAivNum_) {
        return;
    }
    topkId = 0U;
    index = tokenLocalIdx * axisK_;
    if (outputCopyPending_) {
        // 上一个token搬出完成后，V流水才能复用tokenBuf_
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        outputCopyPending_ = false;
    }
    sumFloatBufLocal_ = sumFloatBuf_.Get<float>()[bufferIndex * sumFloatBufOffset_];
    Duplicate(sumFloatBufLocal_, static_cast<float>(0), axisH_);
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::ProcessToken(uint32_t bufferIndex,
                                                                                       uint32_t &tokenLocalIdx,
                                                                                       uint32_t &topkId,
                                                                                       uint32_t &index)
{
    if (tokenLocalIdx >= tokenPerAivNum_) {
        return;
    }
    uint32_t tokenIndex = beginIndex_ + newAivId_ + tokenLocalIdx * receiveAivNum_;
    uint32_t processLen = axisH_;
    const DataCopyPadExtParams<XType> copyPadXTypeParams{false, 0U, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    if (outputCopyPending_) {
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        outputCopyPending_ = false;
    }
    sumFloatBufLocal_ = sumFloatBuf_.Get<float>()[bufferIndex * sumFloatBufOffset_];
    if (!ProcessExpert(tokenIndex, processLen, tokenLocalIdx, topkId, index)) {
        return;
    }
    // token的所有slot到达并完成处理后再更新维测进度
    statePos_++;
    DataCopyParams dataStateParams{1U, sizeof(uint32_t), 0U, 0U};
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    dataStateLocalTensor_.SetValue(0, statePos_);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams);
    ClearPackedTokenFlags(tokenIndex);
    if constexpr (HasAddRmsNorm) {
        AddRmsNormAddCompute(tokenIndex, 0U, processLen, sumFloatBufLocal_, rowTmpFloatLocal_, sumFloatBufLocal_,
                             expandXCopyParams, copyPadXTypeParams);
    }
    // 结果搬出
    PipeBarrier<PIPE_V>();
    LocalTensor<XType> sumBufLocal = tokenBuf_.Get<XType>();
    Cast(sumBufLocal, sumFloatBufLocal_, AscendC::RoundMode::CAST_RINT, processLen);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(expandOutGlobal_[tokenIndex * axisH_], sumBufLocal, expandXCopyParams);
    if constexpr (HasAddRmsNorm) {
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        AddRmsNormRmsNormCompute(tokenIndex, 0U, processLen, sumFloatBufLocal_, mulBufLocal_, gammaLocal_,
                                 expandXCopyParams);
    }
    outputCopyPending_ = true;
    tokenLocalIdx = tokenPerAivNum_;
    tokenNumCompleted_++;
}

template <A5MteCombineTypeClass>
__aicore__ inline bool MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::LocalWindowSplitCoreCal()
{
    if (activeMaskBsCnt_ == 0U) {
        return false;
    }
    uint32_t tokenNum = isInputExpertMaskFlag_ ? axisBS_ : activeMaskBsCnt_;
    uint32_t earlyTokenNum = tokenNum;
    if (tokenNum > recvAivNum_) {
        // 发送核完成后参与接收，提前接收核按整轮分配
        uint32_t numerator = tokenNum * aivNum_ * recvAivNum_;
        uint32_t denominator = aivNum_ * recvAivNum_ + sendAivNum_ * sendAivNum_;
        earlyTokenNum = Ceil(numerator, denominator);
        earlyTokenNum = MIN(tokenNum, Ceil(earlyTokenNum, recvAivNum_) * recvAivNum_);
    }
    bool isRecvAiv = coreIdx_ >= sendAivNum_;
    receiveAivNum_ = isRecvAiv ? recvAivNum_ : sendAivNum_;
    uint32_t receiveTokenNum = isRecvAiv ? earlyTokenNum : tokenNum - earlyTokenNum;
    beginIndex_ = isRecvAiv ? 0U : earlyTokenNum;
    newAivId_ = isRecvAiv ? coreIdx_ - sendAivNum_ : coreIdx_;
    tokenPerAivNum_ = receiveTokenNum / receiveAivNum_;
    uint32_t remainderToken = receiveTokenNum % receiveAivNum_;
    if (newAivId_ < remainderToken) {
        tokenPerAivNum_++;
    }
    if (tokenPerAivNum_ == 0U) {
        return false;
    }
    return true;
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::LocalWindowCopy()
{
    if (!LocalWindowSplitCoreCal()) {
        return;
    }
    tokenNumCompleted_ = 0U;
    statePos_ = 1U;
    LocalWindowInit();
    uint32_t tokenLocalIdx0 = tokenPerAivNum_;
    uint32_t topkId0 = 0U;
    uint32_t index0 = 0U;
    uint32_t tokenLocalIdx1 = tokenPerAivNum_;
    uint32_t topkId1 = 0U;
    uint32_t index1 = 0U;
    // token内按slot顺序处理，两个token之间轮转
    while (tokenNumCompleted_ < tokenPerAivNum_) {
        if (bufferNum_ == 1U) {
            TokenInit(0U, tokenLocalIdx0, topkId0, index0);
            ProcessToken(0U, tokenLocalIdx0, topkId0, index0);
        } else {
            TokenInit(0U, tokenLocalIdx0, topkId0, index0);
            TokenInit(1U, tokenLocalIdx1, topkId1, index1);
            ProcessToken(0U, tokenLocalIdx0, topkId0, index0);
            ProcessToken(1U, tokenLocalIdx1, topkId1, index1);
        }
    }
    if (isPerformanceFlag_) {
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        SetAtomicMax<int32_t>();
        DataCopyExtParams performanceInfoCopyParams{
            1U, static_cast<uint32_t>(JUMP_WRITE * epWorldSizeOriginal_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(performanceInfoGM_, performanceInfoTensor_, performanceInfoCopyParams);
        SetAtomicNone();
    }
}

template <A5MteCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5Mte<A5MteCombineTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (coreIdx_ < sendAivNum_) {
            BuffInit();
            SetWaitTpStatusAndDisPatch();
            PipeBarrier<PIPE_ALL>();
        }
        AlltoAllBuffInitAndMaskCal();
        LocalWindowCopy();
    }
}

} // namespace MoeDistributeCombineV2A5MteImpl
#endif // MOE_DISTRIBUTE_COMBINE_V2_A5_MTE_H
