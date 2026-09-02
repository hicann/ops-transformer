/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 **/

/*!
 * \file moe_distribute_dispatch_v2_host_kfc.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_HOST_KFC_H
#define MOE_DISTRIBUTE_DISPATCH_V2_HOST_KFC_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/reduce.h"
#include "adv_api/math/xor.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_v2_constant.h"
#include "moe_distribute_dispatch_v2_quant.h"
#include "moe_distribute_elastic.h"

#include "moe_distribute_v2_base.h"
#include "check_winsize.h"
#if __has_include("../../common/moe_distribute_base.h")
#include "../../common/moe_distribute_base.h"
#else
#include "../../../common/op_kernel/moe_distribute_base.h"
#endif

namespace MoeDistributeDispatchV2HostKfcImpl {
constexpr uint32_t STATE_SIZE = 2048 * 1024; // 2M
constexpr uint64_t TIMEOUT_OFFSET = 1024UL * 1024UL;
constexpr uint8_t BUFFER_NUM = 2; // 多buf
constexpr uint8_t BUFFER_SINGLE = 1;
constexpr uint32_t STATE_OFFSET = 32U; // 状态空间偏移地址
constexpr uint8_t COMM_NUM = 1;        // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL;
constexpr uint64_t DISPATCH_STATE_WIN_OFFSET = 768UL * 1024UL;
constexpr uint64_t CUMSUM_CAL_OFFSET = 868UL * 1024UL;  // 768 + 100
constexpr uint64_t CUMSUM_FLAG_OFFSET = 876UL * 1024UL; // 868 + 8
constexpr uint64_t TIMEOUT_DETECTION_THRESHOLD = 50000UL;
constexpr uint64_t CYCLES_PER_US = 50UL;
constexpr uint64_t TIMEOUT_DETECTION_TX_UNITS = 8UL;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t EXPAND_IDX_INFO = 3U; // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr uint32_t FLAG_AFTER_WAIT = 2U;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t MAX_UB_SIZE = 240U * 1024U;
constexpr uint32_t BW_ITEM_SIZE = 32; // batchWriteItemSize
constexpr uint32_t B64_PER_BLOCK = 4;
constexpr uint64_t SERVER_STATE_ALIGN = 512UL;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t COMPARE_COUNT_PER_BLOCK = 256U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t TOKEN_RECV_CNT_MAX = 512U;
constexpr uint32_t SERVER_CNT_OFFSET = 0U;
constexpr uint32_t INSERT_DATA_STRIDE = 1U;
constexpr uint32_t COMM_COPY_BLOCK = 15U;       // 480 / 32 =15
constexpr uint32_t FLAG_COPY_BLOCK = 1U;        // 32 / 32 =1
constexpr uint32_t INSERT_FLAG_STRIDE = 15U;    // 480 / 32 =15
constexpr uint32_t COMM_FLAG_COPY_STRIDE = 508; // WIN_ADDR_ALIGN - sizeof(float32_t)copy flag的间隔
constexpr uint32_t EXPERT_CNT_FLAG_SPACE = 0;   // 专家结果flag全部顺序存放
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U;     // 8 = UB_ALIGN / sizeof(float) = UB_ALIGN / sizeof(int32_t)

enum class CommMode {
    INTRA_NODE = 0, // server内
    INTER_NODE = 1  // server间
};

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace Mc2Kernel;

#define TemplateDispatchKFCTypeClass typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define TemplateDispatchKFCTypeFunc XType, ExpandXOutType, QuantMode, IsSmoothScaleExist

template <TemplateDispatchKFCTypeClass>
class MoeDistributeDispatchV2HostKfc {
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
#define TemplateDispatchKFCQuantArgs XInType, ExpandXOutType, XOutType, QuantMode, IsSmoothScaleExist
    __aicore__ inline MoeDistributeDispatchV2HostKfc(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
                                GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR expandScalesOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t startAivId, uint32_t curUseAivNum,
                                       uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum);
    __aicore__ inline void InitRecieveTilingContext(GM_ADDR expandXOut, TPipe *pipe,
                                                    const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void InitCommInfo();
    __aicore__ inline void InitComputeInfo();
    __aicore__ inline void InitSetWindows(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void InitExtraInfo();
    __aicore__ inline void InitMaskInfo();
    __aicore__ inline void InitDispatchBetweenServerInfo();
    __aicore__ inline void InitMaskReSource();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void CalTokenMask();
    template <bool isStoreOffset>
    __aicore__ inline void CalServerSendCnt();
    __aicore__ inline void CalExpertTotalSendCnt();
    __aicore__ inline void SendExpertCnt();
    __aicore__ inline void InitStatusTensor();
    template <typename T>
    __aicore__ inline void ReadDataFromWindows(GM_ADDR dataAddr, uint32_t blockCnt, LocalTensor<T> &xInTensor);
    template <bool isContinueWait>
    __aicore__ inline bool WaitFlag(GM_ADDR flagAddr, LocalTensor<float32_t> flagInLTTensor, uint32_t stride,
                                    uint32_t waitNum);
    __aicore__ inline void SetStatus();
    __aicore__ inline void InitCalCumSumBuffer();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor,
                                            LocalTensor<uint32_t> &gatherTmpTensor,
                                            LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void WaitCntFrRemote();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t totalCount);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WriteExpertOffset();
    __aicore__ inline void CalTokenSendExpertOffset();
    __aicore__ inline void SetServerFlag();
    template <CommMode mode>
    __aicore__ inline void InitQuantBuffer();
    __aicore__ inline void InitScatterBuffer(uint32_t serverId);
    __aicore__ inline void CalServerSplitCore(uint32_t serverNum, uint32_t startAivId, uint32_t useAivNum,
                                              uint32_t &startServerId, uint32_t &endServerId, uint32_t &useAivPerServer,
                                              uint32_t &firstAivId);
    __aicore__ inline void CalExpertOffset(uint32_t startTokenId, uint32_t endTokenId);
    template <typename T>
    __aicore__ inline void WriteDataToWinOut(LocalTensor<T> &xOutTensor, GM_ADDR rankGM, uint32_t blockCnt);
    template <typename T>
    __aicore__ inline void WriteDataToWinOut(LocalTensor<T> &xOutTensor, LocalTensor<T> &xTempTensor, GM_ADDR rankGM,
                                             uint32_t blockCnt);
    __aicore__ inline void FillTriple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t srcRankIndex,
                                      uint32_t tokenIndex, uint32_t k, float expertScale);
    __aicore__ inline void QuantCleanTensor();
    __aicore__ inline bool IsInSameServer(uint32_t targetRankId);
    __aicore__ inline void ProcessToken(GM_ADDR rankGM, uint32_t tokenIndex, uint32_t topKIndex, uint32_t expertIndex,
                                        uint32_t srcRankIndex, float expertScale);
    __aicore__ inline void SendTokenToLocalServer(uint32_t firstAivId, uint32_t usedAivNum);
    __aicore__ inline void FillQuadruple(LocalTensor<XType> &xOutTensor, uint32_t startTokenIndex, uint32_t tokenIndex);
    __aicore__ inline void SingleTokenProcess(uint32_t startTokenIndex, uint32_t tokenIndex, uint32_t dstServerId,
                                              uint32_t cnt);
    __aicore__ inline void ScatterToken(uint32_t startIndex, uint32_t endIndex, uint32_t serverId);
    __aicore__ inline void SendTokenToRemoteServer(uint32_t serverIndex, uint32_t firstAivId, uint32_t usedAivNum);
    __aicore__ inline void ScatterTokenToRemote();
    __aicore__ inline void InitGatherTokenBuffer();
    __aicore__ inline void WaitStatusFlag(uint32_t serverIdx, TBuf<> &serverFlagBuf, uint32_t &rcvTokenCnt);
    __aicore__ inline void SendToExpert(uint32_t rcvCnt);
    __aicore__ inline void WaitToken(uint32_t tokenCnt, uint32_t serverIdx, uint32_t startTokenIdx);
    __aicore__ inline void CleanTokenFlag(uint32_t serverId, uint32_t startTokenId, uint32_t cleanTokenNum);
    __aicore__ inline void GatherTokenFrRemote();
    __aicore__ inline void InitLocalCopyBuffer(uint32_t processNum);
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void SetValidExpertInfo(uint32_t startProcessId, uint32_t endProcessId, uint32_t processNum,
                                              uint32_t &validNum);
    __aicore__ inline void CopyOut(LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index,
                                   uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void WaitAndFormatOutput(uint32_t validNum);
    __aicore__ inline void LocalWindowCopy();

    __aicore__ inline GM_ADDR GetSendAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        return GetBaseWindAddrByRankId(winContext_[COMM_EP_IDX], serverId, epRankIdOriginal_);
    }

    __aicore__ inline GM_ADDR GetReceiveAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        return GetBaseWindAddrByRankId(winContext_[COMM_EP_IDX], serverId, epRankIdOriginal_);
    }

    __aicore__ inline GM_ADDR GetSendStatusAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        return GetBaseWindStateAddrByRankId(winContext_[COMM_EP_IDX], serverId, epRankIdOriginal_);
    }

    __aicore__ inline GM_ADDR GetReceiveStatusAddrBetweenServer(uint8_t ctxIdx, const int32_t serverId)
    {
        return GetBaseWindStateAddrByRankId(winContext_[COMM_EP_IDX], serverId, epRankIdOriginal_);
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        return GetBaseWindAddrByRankId(winContext_[COMM_EP_IDX], rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        return GetBaseWindStateAddrByRankId(winContext_[COMM_EP_IDX], rankId, epRankIdOriginal_) +
               dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<float> expertScalesGMTensor_;
    GlobalTensor<float> expandScalesOutGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<int32_t> sendCountGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    GlobalTensor<uint32_t> selfhcclDataStatusTensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<float> selfRankWinInGMTensor_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<XType> recvTmpTensor_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<ExpandXOutType> insertFlagTensor_;
    LocalTensor<XType> tokenData_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<uint32_t> serverCountTensor_;
    LocalTensor<int32_t> tokenSendMap_;
    LocalTensor<float32_t> flagRecvTensor_;
    LocalTensor<uint32_t> tokenRcvStatus_;
    LocalTensor<int16_t> expertOffsetCntTensor_;
    LocalTensor<bool> expertMaskInputTensor_;
    LocalTensor<uint32_t> commFlagTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    LocalTensor<float> expertScalesTensor_;
    LocalTensor<int32_t> cleanTensor_;

    TBuf<> expertScalesBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_; // 辅助gather tensor定义
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> expertMaskInputBuf_;
    TBuf<> subExpBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> serverCountBuf_;
    TBuf<> serverMapBuf_;
    TBuf<> expertOffsetCntBuf_;
    TBuf<> xSendBuf_;
    TBuf<> flagBuf_;

    TQue<QuePosition::VECIN, 1> tokenInQueue_; // token输入队列
    TQue<QuePosition::VECIN, 1> xInQueue_;     // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;   // 量化使用，量化后的输出
    TQue<QuePosition::VECOUT, 1> xTempQueue_;  // 用于填充flag

    GM_ADDR expandXOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;
    GM_ADDR bufferChoseAddr_;

    GM_ADDR gmTemp;
    GM_ADDR gmMoeCntFlg;
    GM_ADDR gmExpertOffsetCnt;
    GM_ADDR gmSendTokenFlg;
    GM_ADDR workspaceGM_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t bufferId_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    int32_t epRankId_{0};
    int32_t serverId_{0};          // 当前卡所在server编号
    int32_t rankSizePerServer_{0}; // 一个server内的rank数 epWorldSize_ -> rankSizePerServer_
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};    // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0}; // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t globalBS_{0};
    uint32_t moeExpertRankNum_{0}; // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t startMoeExpertId_{0};     // 当前server的起始moe专家id 全局编号
    uint32_t startRankId_{0};          // 当前server的起始卡号 全局编号
    uint32_t shareRankNumInServer_{0}; // 当前server中共享专家卡数
    uint32_t expertNumInServer_{0}; // 当前server中共享专家副本数 + moe专家数 totalExpertNum_ -> expertNumInServer_
    uint32_t hOutSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t hAlignSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hCommuSize_{0};
    uint32_t axisHCommu_{0};
    uint32_t moeListAlign_{0};
    uint32_t moeCntAlign_{0};
    uint32_t moeExpertScalesAlign_{0};
    uint32_t statusBufCntAlign_{0};
    uint32_t dataState_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t totalUsedUB_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_{0}; // 接收Win区块数
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    uint64_t totalWinSizeEp_{0};
    uint32_t expertTokenNumsType_{1};
    uint32_t stateOffset_{0};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    uint32_t sendXTypeElemAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t maxSize_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t blockCntExpertPerServer_{0};
    uint32_t blockCntPerTokenToExpert_{0};
    uint32_t maxBlockCnt_{0};
    uint32_t sendTokenLengthAlign_{0};
    uint32_t sendTokenLength_{0};
    uint32_t serverNum_{0};
    uint32_t hScaleIdxSize_{0};
    uint32_t expertScaleAlign_{0};
    uint32_t scaleInBytes_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t scalesCount_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedScatter_{0};
    uint32_t aivUsedRemotComm_{0}; // 用于专门框间通信的aiv数
    uint32_t aivUsedGather_{0};
    uint32_t tBufRealSize_{0};
    uint32_t perServerMapLength_{0};
    uint32_t perServerMapCnt_{0};
    uint32_t moeCntRowAlign_{0};

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpParam *winContext_[COMM_NUM]{nullptr};

    DataCopyParams xCopyParams_;
    DataCopyExtParams hCommuCopyOutParams_;
    DataCopyParams doWindowCopyOutParams_;

    MoeDistributeDispatchV2Quant<TemplateDispatchKFCQuantArgs> quantInst_;
};

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t startAivId, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId,
    uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t relativeAivId = aivId_ - startAivId;
    if (relativeAivId < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId = relativeAivId * sendTokenNum;
    } else {
        startTokenId = sendTokenNum * relativeAivId + remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitRecieveTilingContext(
    GM_ADDR expandXOut, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
    gmMoeCntFlg = workspaceGM_ + 1UL * 1024 * 1024 * 1024;
    gmExpertOffsetCnt = gmMoeCntFlg + 32;
    gmSendTokenFlg = gmExpertOffsetCnt + 20 * 1024 * 1024;
    gmTemp = gmSendTokenFlg + 10 * 1024 * 1024;
    GM_ADDR bufferChoseAddr_ = workspaceGM_ + 3UL * 1024 * 1024 * 1024 + 1000UL * 1024 * 1024;

    tpipe_ = pipe;
    aivId_ = GetBlockIdx();

    // 检查hcclwinsize是否越界
    totalWinSizeEp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);

    serverNum_ = 1;
    rankSizePerServer_ = 2;
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    globalBS_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    scaleInBytes_ = tilingData->moeDistributeDispatchV2Info.scalesCol *
                    tilingData->moeDistributeDispatchV2Info.scalesTypeSize; // 量化使用
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount;     // 量化使用
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;

    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;

    aivUsedCumSum_ = 2;
    aivUsedRemotComm_ = serverNum_; // 每个核负责一个核发送
    aivUsedScatter_ = (aivNum_ - aivUsedCumSum_ - aivUsedRemotComm_) / 2;
    aivUsedGather_ = aivNum_ - aivUsedScatter_ - aivUsedRemotComm_ - aivUsedCumSum_;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitCommInfo()
{
    winContext_[COMM_EP_IDX] = reinterpret_cast<__gm__ HcclOpParam *>(AscendC::GetHcclContext<HCCL_GROUP_ID_0>());
    bufferChosenGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(bufferChoseAddr_));
    bufferId_ = bufferChosenGlobal_(0);

    statusDataSpaceGm_ = GetStatusDataSpaceGm(winContext_[COMM_EP_IDX]);
    selfRankWinInGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(statusDataSpaceGm_));
    selfDataStatusGMTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t *>(statusDataSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    uint32_t epRankIdHccl = Mc2Kernel::GetRankId(winContext_[COMM_EP_IDX]);
    uint32_t epWorldSizeHccl = Mc2Kernel::GetRankDim(winContext_[COMM_EP_IDX]);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
    statusSpaceGm_ = GetWindStateAddrByRankId(COMM_EP_IDX, epRankIdOriginal_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitComputeInfo()
{
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;

    serverId_ = epRankId_ / rankSizePerServer_;
    startRankId_ = serverId_ * rankSizePerServer_;
    if (startRankId_ + rankSizePerServer_ <= sharedExpertRankNum_) { // 只有共享专家
        shareRankNumInServer_ = rankSizePerServer_;
        startMoeExpertId_ = 0;
    } else if (startRankId_ < sharedExpertRankNum_) { // 有共享专家也有moe专家
        shareRankNumInServer_ = sharedExpertRankNum_ - startRankId_;
        startMoeExpertId_ = 0;
    } else { // 只有moe专家
        shareRankNumInServer_ = 0;
        startMoeExpertId_ = (startRankId_ - sharedExpertRankNum_) * moeExpertNumPerRank_;
    }
    expertNumInServer_ = shareRankNumInServer_ + (rankSizePerServer_ - shareRankNumInServer_) * moeExpertNumPerRank_;
    recvCntWorkspaceGM_ = workspaceGM_ + epWorldSize_ * expertNumInServer_ * UB_ALIGN;
    expertIdsCnt_ = axisBS_ * axisK_;
    recvWinBlockNum_ = epWorldSize_ * moeExpertNumPerRank_;
    stateOffset_ = STATE_OFFSET;
    if (isShareExpertRankFlag_) { // 当前卡是共享专家卡
        rscvStatusNum_ = epWorldSize_;
    } else { // 当前卡是moe专家卡
        rscvStatusNum_ = recvWinBlockNum_;
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitSetWindows(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<ExpandXOutType>(
            reinterpret_cast<__gm__ ExpandXOutType *>(GetWindAddrByRankId(COMM_EP_IDX, tempepRankId)), totalWinSizeEp_);
        OOMCheckAddrRange<float>(reinterpret_cast<__gm__ float *>(GetWindStateAddrByRankId(COMM_EP_IDX, tempepRankId)),
                                 STATE_SIZE);
    }
#endif

    // epWin区划分为前后两半区，连续两次dispatch，切换半区, combine 数据区使用前面，
    // 即axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine, dispatch使用后面
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffsetEp_ =
        dataState_ * (totalWinSizeEp_ / 2) +
        axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine; // 就是分成两块，去掉combine前面的
    windowGM_ = GetWindAddrByRankId(COMM_EP_IDX, epRankIdOriginal_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<ExpandXOutType> winDouble;
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    winDouble.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXOutType *>(windowGM_));
    OOMCheckAddrRange<ExpandXOutType>(reinterpret_cast<__gm__ ExpandXOutType *>(winDouble.GetPhyAddr()),
                                      totalWinSizeEp_);
#endif
    windowInstatusFp32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(statusSpaceGm_));
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitExtraInfo()
{
    // 量化使用
    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hScaleIdxSize_, scaleOutBytes_,
                         axisH_);

    hScaleIdxSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN;
    expertScaleAlign_ = hScaleIdxSize_ / sizeof(uint32_t);
    hScaleIdxSize_ = hScaleIdxSize_ + UB_ALIGN; // 用于存放expertscale 16.06KB

    // 框间使用
    sendXTypeElemAlign_ = hAlignSize_ / sizeof(uint32_t);
    uint32_t moeListSizeAlign = hAlignSize_ + UB_ALIGN; // moeExpertIdList起始偏移
    moeListAlign_ = moeListSizeAlign / sizeof(int32_t);

    uint32_t moeCntSizeAlign = moeListSizeAlign + Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // moeCntList偏移
    moeCntAlign_ = moeCntSizeAlign / sizeof(int16_t); // 注意这里的offsetCnt是int16_t

    uint32_t moeExpertScaleSizeAlign =
        moeCntSizeAlign + Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN; // moe expertscales
    moeExpertScalesAlign_ = moeExpertScaleSizeAlign / sizeof(uint32_t);

    uint32_t moeNumPerServer = moeExpertNumPerRank_ * rankSizePerServer_;
    sendTokenLength_ = moeExpertScaleSizeAlign + Ceil(axisK_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    blockCntPerToken_ = Ceil(sendTokenLength_, SPLIT_BLOCK_DATA_SIZE);
    blockCntExpertPerServer_ = Ceil(moeNumPerServer * sizeof(int32_t), SPLIT_BLOCK_DATA_SIZE);
    blockCntPerTokenToExpert_ = Ceil(hScaleIdxSize_, SPLIT_BLOCK_DATA_SIZE); // 框内块数
    sendTokenLengthAlign_ = blockCntPerToken_ * SERVER_STATE_ALIGN;          // 17K

    hCommuSize_ = blockCntPerTokenToExpert_ * SERVER_STATE_ALIGN; // 拼接flag后的字节数
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    hAlignWinSize_ = Ceil(hCommuSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    hOutAlignUbSize_ = Ceil(hScaleIdxSize_, UB_ALIGN) * UB_ALIGN;
    maxBlockCnt_ = max(max(blockCntPerToken_, blockCntExpertPerServer_), blockCntPerTokenToExpert_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitMaskInfo()
{
    activeMaskBsCnt_ = axisBS_;
    expertIdsCnt_ = axisBS_ * axisK_;

    uint32_t hFp32Size = axisH_ * sizeof(float);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr ((QuantMode == PERGROUP_DYNAMIC_QUANT) && IsSmoothScaleExist) {
        hFp32Size = Align128(axisH_) * sizeof(float);
    }
#endif
    uint32_t expertIdsSize = expertIdsCnt_ * sizeof(int32_t);
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    uint32_t expertCntAlign = Ceil(moeExpertNum_ * sizeof(int32_t), SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_DATA_SIZE;
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t bsKAlign256 = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t expertIdsBufSize = expertIdsSize > bsAlign256 ? expertIdsSize : bsAlign256;
    uint32_t expertOffsetCntSize = axisBS_ * Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint32_t exprtflagCntSize = Ceil((aivNum_ - aivUsedCumSum_) * sizeof(float32_t), UB_ALIGN) * UB_ALIGN;
    moeCntRowAlign_ = Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN / sizeof(int16_t);
    expertIdsSize = Ceil(expertIdsSize, UB_ALIGN) * UB_ALIGN;
    maxSize_ = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSize_ = maxSize_ > xActivateMaskSize ? maxSize_ : xActivateMaskSize;
    maxSize_ = maxSize_ > bsKAlign256 ? maxSize_ : bsKAlign256;
    maxSize_ = maxSize_ > expertCntAlign ? maxSize_ : expertCntAlign;
    maxSize_ = maxSize_ > expertOffsetCntSize ? maxSize_ : expertOffsetCntSize;
    maxSize_ =
        maxSize_ > exprtflagCntSize ? maxSize_ : exprtflagCntSize; // 因为计算flag时复用了buffer因此需要保证32B对齐

    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize); // BS * K * 4 = 32K
    tpipe_->InitBuffer(gatherMaskTBuf_, maxSize_);    // max:32K
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);         // max:32K
    tpipe_->InitBuffer(subExpBuf_, maxSize_);         // max:32K

    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    totalUsedUB_ += maxSize_;
    totalUsedUB_ += maxSize_;
    totalUsedUB_ += expertIdsSize;
    totalUsedUB_ += maxSize_;

    if (isExpertMaskFlag_) {
        tpipe_->InitBuffer(expertMaskInputBuf_, expertIdsCnt_ * sizeof(bool)); // 16k
        totalUsedUB_ += expertIdsCnt_ * sizeof(bool);
        expertMaskInputTensor_ = expertMaskInputBuf_.Get<bool>();
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitDispatchBetweenServerInfo()
{
    xCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U};
    perServerMapLength_ = Ceil(axisMaxBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    perServerMapCnt_ = perServerMapLength_ / sizeof(int32_t);
    uint32_t serverBuferLength = serverNum_;
    uint32_t serverMapLength = serverNum_ * perServerMapLength_;
    uint32_t expertOffsetCntSize = axisBS_ * Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint8_t commRepeatTime = static_cast<uint8_t>(Ceil(maxBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256));
    uint32_t commFlagSize = commRepeatTime * SIZE_ALIGN_256;
    uint64_t mask[1] = {0x0101010101010101};

    tpipe_->InitBuffer(serverCountBuf_, serverBuferLength * sizeof(uint32_t));
    tpipe_->InitBuffer(serverMapBuf_, serverMapLength);           // max server = 8, per server = 2K
    tpipe_->InitBuffer(expertOffsetCntBuf_, expertOffsetCntSize); // 16K
    tpipe_->InitBuffer(flagBuf_, commFlagSize);

    serverCountTensor_ = serverCountBuf_.Get<uint32_t>();
    tokenSendMap_ = serverMapBuf_.Get<int32_t>();
    expertOffsetCntTensor_ = expertOffsetCntBuf_.Get<int16_t>();
    commFlagTensor_ = flagBuf_.Get<uint32_t>();

    Duplicate<uint32_t>(serverCountTensor_, uint32_t(0), serverBuferLength);
    Duplicate<int32_t>(tokenSendMap_, int32_t(-1), serverMapLength);
    Duplicate<uint32_t>(commFlagTensor_, 0x3F800000, mask, commRepeatTime, uint16_t(1),
                        uint8_t(8)); // 0x3F800000是float的1
    Duplicate<int16_t>(expertOffsetCntTensor_, static_cast<int16_t>(0), expertIdsCnt_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR sendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    elasticInfoGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(elasticInfo));
    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(x));
    xActiveMaskGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(xActiveMask));
    expertScalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expertScales));
    expandScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expandScalesOut));
    expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertIds));
    dynamicScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dynamicScalesOut));
    expertTokenNumsOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOut));
    expandIdxGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expandIdxOut));
    workspaceGM_ = workspaceGM;
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut; // 无GlobalTensor
    sendCountGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspaceGM_));
    if constexpr (IsSmoothScaleExist) {
        scalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scales));
    }
    InitRecieveTilingContext(expandXOut, pipe, tilingData);
    InitCommInfo();
    InitComputeInfo();
    InitSetWindows(tilingData);
    InitExtraInfo();
    InitMaskInfo();
    InitDispatchBetweenServerInfo();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitMaskReSource()
{
    if (isExpertMaskFlag_) { // 当二维mask为true时，需要初始化专家mask输入张量
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskInputTensor_, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    }

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams); // copy expertid
    SyncFunc<AscendC::HardEvent::MTE2_V>(); // 后续全部都需要依赖此数据，在这里全部一起等待
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::TokenActiveMaskCal()
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
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ExpertActiveMaskCal()
{
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = expertMaskInputTensor_.ReinterpretCast<int8_t>();
    LocalTensor<half> maskStrideFp16Tensor = subExpBuf_.Get<half>();
    LocalTensor<int32_t> maskStrideInt32Tensor = gatherMaskTBuf_.Get<int32_t>();
    Adds(expertIdsTensor_, expertIdsTensor_, moeExpertNum_, expertIdsCnt_);
    Cast(maskStrideFp16Tensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, expertIdsCnt_);
    Cast(maskStrideInt32Tensor, maskStrideFp16Tensor, RoundMode::CAST_RINT, expertIdsCnt_);
    Mul(expertIdsTensor_, expertIdsTensor_, maskStrideInt32Tensor, expertIdsCnt_);
    Adds(expertIdsTensor_, expertIdsTensor_, -moeExpertNum_, expertIdsCnt_);
    Abs(expertIdsTensor_, expertIdsTensor_, expertIdsCnt_);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalTokenMask()
{
    activeMaskBsCnt_ = axisBS_;
    InitMaskReSource();
    if (isTokenMaskFlag_) { // 1维
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) { // 2维
        ExpertActiveMaskCal();
    }
}

template <TemplateDispatchKFCTypeClass>
template <bool isStoreOffset>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalServerSendCnt()
{
    uint32_t calCnt = isTokenMaskFlag_ ? activeMaskBsCnt_ * axisK_ : expertIdsCnt_;
    uint32_t activeMaskBsCnt = static_cast<uint32_t>(activeMaskBsCnt_);
    uint32_t compareCount =
        Ceil(activeMaskBsCnt * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    uint32_t maskSize = Ceil(compareCount / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN; // 为了匹配CompareScalar接口1bit一共8位
    uint64_t sendTokenCnt = 0;
    TBuf<> sharedTmpBuf;
    tpipe_->InitBuffer(sharedTmpBuf, SIZE_ALIGN_256);
    LocalTensor<float32_t> dstServerIdFp32 = dstExpBuf_.Get<float32_t>();
    LocalTensor<float32_t> tempTensorFp32 = subExpBuf_.Get<float32_t>();
    LocalTensor<int32_t> dstServerIdInt32 = dstExpBuf_.Get<int32_t>();
    LocalTensor<int16_t> dstServerIdInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = subExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> compareInt16 = gatherMaskTBuf_.Get<int16_t>();
    LocalTensor<uint8_t> sharedTensorInt8 = sharedTmpBuf.Get<uint8_t>();
    LocalTensor<int32_t> indexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint8_t> expertMaskTensorU8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = gatherMaskTBuf_.Get<uint32_t>();
    LocalTensor<int32_t> gatherIndex = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> dstOffset = dstExpBuf_.Get<uint32_t>();
    Cast(dstServerIdFp32, expertIdsTensor_, RoundMode::CAST_NONE, calCnt);
    Duplicate<float32_t>(tempTensorFp32, static_cast<float32_t>(moeExpertNumPerRank_), calCnt);
    Div(dstServerIdFp32, dstServerIdFp32, tempTensorFp32, calCnt);
    Adds(dstServerIdFp32, dstServerIdFp32, static_cast<float32_t>(sharedExpertRankNum_), calCnt);
    Duplicate<float32_t>(tempTensorFp32, static_cast<float32_t>(rankSizePerServer_), calCnt);
    Div(dstServerIdFp32, dstServerIdFp32, tempTensorFp32, calCnt); // 计算每个专家所在的serverID
    Cast(dstServerIdInt32, dstServerIdFp32, RoundMode::CAST_TRUNC, calCnt);
    for (int32_t serverIndex = 0; serverIndex < serverNum_; serverIndex++) {
        Adds(dstServerIdInt32, dstServerIdInt32, -serverIndex, calCnt);
        Abs(dstServerIdFp32, dstServerIdFp32, calCnt);
        Mins(dstServerIdInt32, dstServerIdInt32, 1, calCnt);
        Cast(dstServerIdInt16, dstServerIdInt32, RoundMode::CAST_NONE, calCnt);
        Duplicate<int16_t>(tempTensorInt16, 1, calCnt);
        Xor(compareInt16, dstServerIdInt16, tempTensorInt16, sharedTensorInt8, calCnt);
        Cast(dstServerIdFp32, compareInt16, RoundMode::CAST_NONE, calCnt); // 这里是 0 1 结果
        uint32_t srcShape[2] = {activeMaskBsCnt, axisK_};
        ReduceSum<float32_t, Pattern::Reduce::AR, true>(tempTensorFp32, dstServerIdFp32, sharedTensorInt8, srcShape,
                                                        false);
        Mins(dstServerIdFp32, tempTensorFp32, float32_t(1.0), activeMaskBsCnt); // 得到需要发送到这个server的TokenID
        Cast(dstServerIdInt32, dstServerIdFp32, RoundMode::CAST_RINT, activeMaskBsCnt);
        CreateVecIndex(indexTensor, 0, activeMaskBsCnt);
        Duplicate<uint32_t>(expertMaskTensorU32, 0, maskSize);
        CompareScalar(expertMaskTensorU8, dstServerIdInt32, 1, CMPMODE::EQ, compareCount);
        GatherMask(gatherIndex, indexTensor, expertMaskTensorU32, true, activeMaskBsCnt, {1, 1, 0, 0}, sendTokenCnt);
        if (sendTokenCnt == 0) {
            continue;
        }
        serverCountTensor_.SetValue(serverIndex, static_cast<uint32_t>(sendTokenCnt));
        if constexpr (isStoreOffset) {
            Muls(gatherIndex, gatherIndex, static_cast<int32_t>(sizeof(int32_t)), static_cast<uint32_t>(sendTokenCnt));
            Scatter(tokenSendMap_[serverIndex * perServerMapCnt_], indexTensor, dstOffset, 0,
                    static_cast<uint32_t>(sendTokenCnt)); // 32字节对齐
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalExpertTotalSendCnt()
{
    uint64_t maskCnt = 0;
    uint32_t compareCount = 0;
    uint32_t maskNumPerExpert = 0;
    uint32_t mask = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;

    compareCount = Ceil(mask * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t); // 一共多少元素
    maskNumPerExpert = compareCount / BITS_PER_BYTE / sizeof(int32_t); // 为了匹配CompareScalar接口1bit一共8位
    LocalTensor<uint8_t> expertMaskTensorU8 = subExpBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = subExpBuf_.Get<uint32_t>();
    LocalTensor<int32_t> gatherTempTensor = gatherMaskTBuf_.Get<int32_t>();
    tokenNumToExpertTensor_ = dstExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(tokenNumToExpertTensor_, 0, moeExpertNum_);
    for (int32_t expertIndex = 0; expertIndex < moeExpertNum_; expertIndex++) { // 处理moe专家
        int32_t dstExpertId = expertIndex;
        Duplicate<uint32_t>(expertMaskTensorU32, 0, maskNumPerExpert);
        CompareScalar(expertMaskTensorU8, expertIdsTensor_, dstExpertId, CMPMODE::EQ, compareCount);
        GatherMask(gatherTempTensor, expertIdsTensor_, expertMaskTensorU32, true, mask, {1, 1, 0, 0}, maskCnt);
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt));
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendExpertCnt()
{
    // 发送每个专家的cnt到目标server
    uint32_t startServerId, endServerId, sendServerNum;
    SplitToCore(serverNum_, 0, aivUsedCumSum_, startServerId, endServerId, sendServerNum);
    if (startServerId >= endServerId || sendServerNum == 0) {
        return;
    }
    LocalTensor<int32_t> sendTokenNumToexpert = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(sendTokenNumToexpert, 0, moeExpertNumPerRank_ * rankSizePerServer_);
    for (uint32_t serverIndex = startServerId; serverIndex < endServerId; serverIndex++) {
        uint32_t serverstartRankId = serverIndex * rankSizePerServer_;
        uint32_t moeOffset = 0;
        uint32_t moeIdstart = (serverstartRankId - sharedExpertRankNum_) * moeExpertNumPerRank_;
        uint32_t moeNumInServer = moeExpertNumPerRank_ * rankSizePerServer_;
        uint32_t moeIdend = moeIdstart + moeNumInServer;
        if (serverstartRankId < sharedExpertRankNum_) { // 目标server有共享专家
            moeIdstart = 0;
            moeNumInServer = (sharedExpertRankNum_ - serverstartRankId) * moeExpertNumPerRank_;
            if (sharedExpertRankNum_ - serverstartRankId >= rankSizePerServer_) { // 全部都是共享专家
                moeNumInServer = 0;
            }
            moeIdend = moeIdstart + moeNumInServer;
            uint32_t groupId = epRankId_ % rankNumPerSharedExpert_;
            moeOffset += sharedExpertRankNum_ - serverstartRankId; // 当前server有几个共享专家卡
            for (uint32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) {
                uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + groupId;
                uint32_t dstServerId = dstRankId / rankSizePerServer_;
                if (dstServerId != serverIndex) {
                    continue;
                }
                uint32_t dstOffsetIndex = dstRankId - serverstartRankId;
                sendTokenNumToexpert.SetValue(dstOffsetIndex, serverCountTensor_.GetValue(dstServerId));
            }
        }

        for (uint32_t moeIndex = moeIdstart; moeIndex < moeIdend; moeIndex++) {
            uint32_t dst = moeOffset + moeIndex - moeIdstart;
            sendTokenNumToexpert.SetValue(dst, tokenNumToExpertTensor_.GetValue(moeIndex));
        }

        uint32_t dstOffset = serverIndex * blockCntExpertPerServer_ * SERVER_STATE_ALIGN;
        GM_ADDR rankGM = (GetSendStatusAddrBetweenServer(COMM_EP_IDX, serverIndex) + dstOffset);
        WriteDataToWinOut(sendTokenNumToexpert, rankGM, blockCntExpertPerServer_);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitStatusTensor()
{
    uint32_t rcvExpertDataSize = SPLIT_BLOCK_DATA_SIZE * blockCntExpertPerServer_;
    uint32_t flagMaxRecvNum = blockCntExpertPerServer_;
    uint32_t flagBufferSize = flagMaxRecvNum * UB_ALIGN;
    uint32_t statusBufBufferSize = flagBufferSize > rcvExpertDataSize ? flagBufferSize : rcvExpertDataSize;
    uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    TBuf flagRecvBuf;
    tpipe_->InitBuffer(flagRecvBuf, statusBufBufferSize);
    tpipe_->InitBuffer(statusBuf_, statusBufCntAlign_ * UB_ALIGN);
    flagRecvTensor_ = flagRecvBuf.Get<float32_t>();
    statusBufCntAlign_ = Ceil(expertNumInServer_ * serverNum_, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign_ * 8);
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign_ / 8, 1, 8); // 0x3F800000是float的1
}

template <TemplateDispatchKFCTypeClass>
template <typename T>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ReadDataFromWindows(
    GM_ADDR dataAddr, uint32_t blockCnt, LocalTensor<T> &xInTensor)
{
    GlobalTensor<T> dataFlagGlobal;
    dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dataAddr));

    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCnt), static_cast<uint32_t>(SPLIT_BLOCK_DATA_SIZE),
                                      static_cast<uint32_t>(SERVER_STATE_ALIGN - SPLIT_BLOCK_DATA_SIZE), 0};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xInTensor, dataFlagGlobal, srcTokenCopyParams, srcTokenPadParams);
}

template <TemplateDispatchKFCTypeClass>
template <bool isContinueWait>
__aicore__ inline bool MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitFlag(
    GM_ADDR flagAddr, LocalTensor<float32_t> flagInLTTensor, uint32_t stride, uint32_t waitNum)
{
    uint32_t copyNum = 1; // 如果stride为0，表示所有数据全部在一起，直接一起拷贝进来
    uint32_t copyBlockLength = waitNum * sizeof(float32_t);
    uint32_t calCnt = waitNum;

    if (stride != 0) {
        copyNum = waitNum;
        copyBlockLength = sizeof(float32_t);
        calCnt = waitNum * UB_ALIGN / sizeof(float32_t); // 每次拷贝会自动填充满足32B对齐
    }

    GlobalTensor<float32_t> flagInGMTensor;
    flagInGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(flagAddr));
    DataCopyExtParams flagCopyParams{static_cast<uint16_t>(copyNum), copyBlockLength, static_cast<uint32_t>(stride), 0,
                                     0};
    DataCopyPadExtParams<float32_t> flagPadParams{true, 0U, 0U, 0U}; // 全部填充为0
    uint32_t rcvCnt = 0;
    do {
        DataCopyPad(flagInLTTensor, flagInGMTensor, flagCopyParams, flagPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(flagInLTTensor, flagInLTTensor, flagInLTTensor, calCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        rcvCnt = static_cast<uint32_t>(flagInLTTensor.GetValue(0));
        if constexpr (isContinueWait) {
            if (rcvCnt == waitNum) { // 循环等待直到全部到达
                break;
            }
        }

    } while (isContinueWait);
    return rcvCnt == waitNum;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void
MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetStatus() // 框间状态区接收 + 框内状态区发送
{
    uint32_t startServerId;
    uint32_t endServerId;
    uint32_t resvServerNum;
    SplitToCore(serverNum_, 0, aivUsedCumSum_, startServerId, endServerId, resvServerNum);
    if (startServerId >= endServerId || resvServerNum == 0) {
        return;
    }

    InitStatusTensor();

    for (int serverIdx = startServerId; serverIdx < endServerId; serverIdx++) {
        GM_ADDR serverRcvAddr = reinterpret_cast<GM_ADDR>(GetReceiveStatusAddrBetweenServer(COMM_EP_IDX, serverId_) +
                                                          blockCntExpertPerServer_ * SERVER_STATE_ALIGN * serverIdx);
        GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(serverRcvAddr + SPLIT_BLOCK_DATA_SIZE);

        GlobalTensor<uint32_t> statusWinGMTensor;
        statusWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(serverRcvAddr));

        WaitFlag<true>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, blockCntExpertPerServer_);
        LocalTensor<uint32_t> rcvExpertCntTensor = flagRecvTensor_.template ReinterpretCast<uint32_t>(); // buf复用
        ReadDataFromWindows<uint32_t>(serverRcvAddr, blockCntExpertPerServer_, rcvExpertCntTensor);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t expertIndex = 0; expertIndex < expertNumInServer_; expertIndex++) {
            statusTensor_(expertIndex * 8 + 1) = rcvExpertCntTensor(expertIndex);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> rankGMTensor;
        for (uint32_t expertIndex = 0; expertIndex < expertNumInServer_; expertIndex++) {
            uint32_t srcRankId = serverIdx * rankSizePerServer_ + epRankId_ % rankSizePerServer_;
            uint32_t dstRankId = expertIndex; // 共享专家
            uint32_t offset = stateOffset_ * srcRankId;
            if (expertIndex >= shareRankNumInServer_) { // moe专家
                dstRankId = (expertIndex - shareRankNumInServer_) / moeExpertNumPerRank_ + shareRankNumInServer_;
                offset += ((expertIndex - shareRankNumInServer_) % moeExpertNumPerRank_ * epWorldSize_ * stateOffset_);
            }
            GM_ADDR rankGM =
                reinterpret_cast<GM_ADDR>(GetWindStateAddrByRankId(COMM_EP_IDX, dstRankId) + offset); // 计算地址偏移
            rankGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rankGM));
            DataCopy<int32_t>(rankGMTensor, statusTensor_[expertIndex * 8], 8UL);
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitCalCumSumBuffer()
{
    tpipe_->Reset();
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

    uint32_t statusBufSize = rscvStatusNum_ * UB_ALIGN;
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t workLocalBufSize = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitDispatchClearStatus()
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

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::GatherSumRecvCnt(
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

    // 对当前核对应的专家recv cnt求和
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) // 对inner要求32对齐
                                        * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // 把当前核的所有专家recv cnt之和写到状态区
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
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)],
             sumCoreFP32Tensor, sumIntriParams);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitCntFrRemote()
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
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    // 清状态
    WaitDispatchClearStatus();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::GetCumSum(
    LocalTensor<int32_t> &outLocal, uint32_t newAivId)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1),
                                  0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    uint32_t mask = 2, cumSumFlag = 0;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, aivUsedCumSum_};
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();
    while (true) { // 等待cnt
        DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)],
                 sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask,
                   {1, static_cast<uint16_t>(aivUsedCumSum_), 1, 0}, recvCnt);
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<uint32_t>(recvCntSumOutTensor.GetValue(0));
        if (cumSumFlag == aivUsedCumSum_) {
            break;
        }
    }
    if (newAivId == 0) {
        outLocal.SetValue(0, 0);
    } else {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask,
                   {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }
    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8); // 一次处理256字节，8个datablock
    // 64 = 256 / sizeof(float) 一次操作字节数，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor,
             cleanParams);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalRecvAndSetFlag()
{
    // check flag 用于 aivUsedCumSum_ 软同步并计算 aivUsedCumSum_ 个核各自的recvCount
    LocalTensor<int32_t> outCountLocal;
    GetCumSum(outCountLocal, aivId_);
    // 计算epRecvCnt
    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = statusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + 1);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR wAddr = reinterpret_cast<GM_ADDR>(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(wAddr));
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
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_FLAG_OFFSET + aivId_ * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor,
             sumIntriParams); // 软同步
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetExpertTokenNums()
{
    // 接受并计算每个专家的cnt数量
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    DataCopyParams totalStatusCopyParams{static_cast<uint16_t>(localExpertNum * epWorldSize_), 1, 0, 0};
    LocalTensor<float> totalStatusTensorFp32 = statusBuf_.Get<float>();
    DataCopy(totalStatusTensorFp32, windowInstatusFp32Tensor_, totalStatusCopyParams);
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

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalCumSum()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    CalServerSendCnt<false>(); // 计算每个当前server发送给远端server的Token数量
    CalExpertTotalSendCnt();   // 计算每个专家的Token总数
    SendExpertCnt();           // 先把数据发送给远端server
    SetStatus();
    SplitToCore(rscvStatusNum_, 0, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_);
    InitCalCumSumBuffer();
    WaitCntFrRemote();   // 等待server间的cnt数据到达
    CalRecvAndSetFlag(); // 计算总的收到的cnt数，并进行server内通信
    // 使用newAivId为0的核进行计算
    if (aivId_ == 0) {
        SetExpertTokenNums();
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WriteExpertOffset()
{
    // 每个核负责计算一部分专家的偏移量，计算完成后写回全局内存，后续每个核只计算对应Token的cnt偏移量
    uint32_t newAivId = aivId_ - aivUsedCumSum_;
    uint32_t offsetPerMask = Ceil(expertIdsCnt_ * sizeof(uint16_t), UB_ALIGN) * UB_ALIGN;
    uint64_t dstOffset = newAivId * offsetPerMask; // gmExpertOffsetCnt
    uint64_t dstFlag = (aivNum_ - aivUsedCumSum_) * offsetPerMask + sizeof(uint32_t) * newAivId;
    GlobalTensor<int16_t> expertOffsetCntGMTensor;
    GlobalTensor<float32_t> expertOffsetFlagGMTensor;
    LocalTensor<float32_t> writeCntTensor = dstExpBuf_.Get<float32_t>();
    expertOffsetCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(gmExpertOffsetCnt + dstOffset));
    expertOffsetFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(gmExpertOffsetCnt + dstFlag));
    writeCntTensor.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyExtParams expertOffsetCopyOutParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int16_t)), 0U, 0U, 0U};
    DataCopyPad(expertOffsetCntGMTensor, expertOffsetCntTensor_, expertOffsetCopyOutParams); // 写对应的结果到全局内存
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams expertCntCopyOutParams{1U, static_cast<uint32_t>(sizeof(float32_t)), 0U, 0U, 0U};
    DataCopyPad(expertOffsetFlagGMTensor, writeCntTensor,
                expertCntCopyOutParams); // 写flag通知server对应专家偏移量已经准备好
    expertOffsetFlagGMTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ float32_t *>(gmExpertOffsetCnt + (aivNum_ - aivUsedCumSum_) * offsetPerMask));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalTokenSendExpertOffset()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    SplitToCore(moeExpertNum_, aivUsedCumSum_, aivNum_ - aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum);
    if (startExpertId >= endExpertId || sendExpertNum == 0) {
        return;
    }
    TBuf<> tempBuf, indexBuf;
    uint32_t activeBsCnt = activeMaskBsCnt_;
    uint32_t calCnt =
        isTokenMaskFlag_ ? activeBsCnt * axisK_ : expertIdsCnt_; // 二维mask时，为了对齐需要全部读取expertID
    uint32_t expertNumAlign256 = Ceil(calCnt * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint64_t sendNumperExpert = 0;
    tpipe_->InitBuffer(tempBuf, activeBsCnt * sizeof(uint32_t));
    tpipe_->InitBuffer(indexBuf, calCnt * sizeof(uint16_t));
    LocalTensor<int16_t> dstExpIdTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> expertIdsInt16 = subExpBuf_.Get<int16_t>();
    LocalTensor<half> dstExpIdTensorFP16 = dstExpBuf_.Get<half>();
    LocalTensor<int16_t> XorTensorInt16 = subExpBuf_.Get<int16_t>();
    LocalTensor<uint8_t> sharedTensorInt8 = gatherMaskTBuf_.Get<uint8_t>(); // 最小需要256B，最大是element * typesize
    LocalTensor<int16_t> expertsIndexTensor = indexBuf.Get<int16_t>();
    LocalTensor<uint16_t> gatherMaskTensor = gatherMaskTBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> tempOffset = gatherMaskTBuf_.Get<int32_t>();
    LocalTensor<int16_t> validExpertIndexTensor = tempBuf.Get<int16_t>();
    LocalTensor<uint32_t> dstoffset = gatherMaskTBuf_.Get<uint32_t>();
    CreateVecIndex(expertsIndexTensor, static_cast<int16_t>(0), calCnt);
    for (uint32_t Index = startExpertId; Index < endExpertId; Index++) {
        int16_t dstExpertId = static_cast<uint16_t>(Index);
        Cast(expertIdsInt16, expertIdsTensor_, RoundMode::CAST_NONE, calCnt);
        Adds(dstExpIdTensorInt16, expertIdsInt16, -dstExpertId, calCnt);
        Abs(dstExpIdTensorFP16, dstExpIdTensorFP16, calCnt);
        Mins(dstExpIdTensorInt16, dstExpIdTensorInt16, 1, calCnt);
        Duplicate<int16_t>(XorTensorInt16, 1, calCnt);
        Xor(dstExpIdTensorInt16, dstExpIdTensorInt16, XorTensorInt16, sharedTensorInt8, calCnt);
        Duplicate<uint16_t>(gatherMaskTensor, 0,
                            Ceil(calCnt, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE /
                                sizeof(uint16_t)); // 获取索引需要放在这里，因为需要256对齐
        Cast(dstExpIdTensorFP16, dstExpIdTensorInt16, RoundMode::CAST_RINT, calCnt);
        CompareScalar(gatherMaskTensorInt8, dstExpIdTensorFP16, static_cast<half>(1), AscendC::CMPMODE::EQ,
                      expertNumAlign256);
        GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0},
                   sendNumperExpert); // 当前需要发送的专家的索引以及需要发送的有效专家总数
        if (sendNumperExpert == 0) {
            continue;
        }
        Cast(tempOffset, validExpertIndexTensor, RoundMode::CAST_NONE, static_cast<uint32_t>(sendNumperExpert));
        Muls(tempOffset, tempOffset, static_cast<int32_t>(sizeof(half)), static_cast<uint32_t>(sendNumperExpert));
        Scatter(expertOffsetCntTensor_, expertsIndexTensor, dstoffset, 0, static_cast<uint32_t>(sendNumperExpert));
    }
    WriteExpertOffset(); // 计算完成将结果写入GM并写flag
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetServerFlag()
{
    uint32_t totalSeverCnt = serverNum_;
    uint32_t startServerNum, endServerNum, sendServerNum;
    SplitToCore(totalSeverCnt, aivUsedCumSum_, aivUsedRemotComm_, startServerNum, endServerNum, sendServerNum);
    if (startServerNum >= totalSeverCnt || sendServerNum == 0) {
        return;
    }
    GlobalTensor<uint32_t> dstStateGMTensor;
    TBuf<> tempbuf;
    tpipe_->InitBuffer(tempbuf, SERVER_STATE_ALIGN);
    LocalTensor<uint32_t> outTensor = tempbuf.Get<uint32_t>();
    uint32_t flagOffset = SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t); // 前面的状态区的最后32B的第一个元素是flag位为1
    outTensor(flagOffset) = 1;
    SyncFunc<AscendC::HardEvent::V_S>();
    for (uint32_t index = startServerNum; index < endServerNum; index++) {
        dstStateGMTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint32_t *>(GetSendAddrBetweenServer(COMM_EP_IDX, index)));
        outTensor(0) = serverCountTensor_(index);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(dstStateGMTensor, outTensor, SERVER_STATE_ALIGN / sizeof(uint32_t));
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalServerSplitCore(
    uint32_t serverNum, uint32_t startAivId, uint32_t useAivNum, uint32_t &startServerId, uint32_t &endServerId,
    uint32_t &useAivPerServer, uint32_t &firstAivId)
{
    useAivPerServer = useAivNum / serverNum; // 每个server分几个核处理
    uint32_t remainderAivNum = useAivNum % serverNum;
    uint32_t relativeAivId = aivId_ - startAivId;    // 相对Id
    startServerId = relativeAivId / useAivPerServer; // 当前核需要处理的
    endServerId = startServerId + 1;
    firstAivId = startAivId + startServerId * useAivPerServer + remainderAivNum;
    if (startServerId < remainderAivNum) { // 如果有剩余的核
        useAivPerServer += 1;              // 前面的server多分一个核
        firstAivId = startAivId + startServerId * useAivPerServer;
    }
}

template <TemplateDispatchKFCTypeClass>
template <CommMode mode>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitQuantBuffer()
{
    if constexpr (QuantMode > UNQUANT) {
        if constexpr (mode == CommMode::INTER_NODE) {
            tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize_); // max{28K, BS * K * 4B}
            tpipe_->InitBuffer(smoothScalesBuf_, maxSize_);         // max{28K, BS * K * 4B}
        } else {
            receiveDataCastFloatBuf_ = dstExpBuf_;
            smoothScalesBuf_ = subExpBuf_; // 内存复用
        }
        floatLocalTemp_ = receiveDataCastFloatBuf_.Get<float>();
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
    }
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf_, dynamicScalesOutGMTensor_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitScatterBuffer(uint32_t serverId)
{
    if (serverId == serverId_) {
        InitQuantBuffer<CommMode::INTRA_NODE>();                      // 初始化量化参数
        tpipe_->InitBuffer(xInQueue_, 1, hAlignSize_);                // 17.5K
        tpipe_->InitBuffer(xOutQueue_, 1, hOutAlignUbSize_);          // 16.03K
        tpipe_->InitBuffer(expertScalesBuf_, axisK_ * sizeof(float)); // 16 * 4
        tpipe_->InitBuffer(xTempQueue_, 1, hCommuSize_);              // 17.5KB
        tokenData_ = xInQueue_.AllocTensor<XType>();
        insertFlagTensor_ = xTempQueue_.AllocTensor<ExpandXOutType>();
        expertScalesTensor_ = expertScalesBuf_.Get<float>();
    } else {
        tpipe_->InitBuffer(xSendBuf_, sendTokenLength_); // 17K
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CalExpertOffset(
    uint32_t startTokenId, uint32_t endTokenId)
{
    uint32_t offsetPerMask = Ceil(expertIdsCnt_ * sizeof(uint16_t), UB_ALIGN) * UB_ALIGN;
    uint64_t dstFlag = (aivNum_ - aivUsedCumSum_) * offsetPerMask;
    uint64_t usedCoreNum = (aivNum_ - aivUsedCumSum_) > moeExpertNum_ ? moeExpertNum_ : (aivNum_ - aivUsedCumSum_);
    uint64_t dstOffset = startTokenId * axisK_ * sizeof(int16_t);
    uint32_t computeRow = endTokenId - startTokenId;
    uint32_t expertOffsetNum = computeRow * moeCntRowAlign_; // 32B对齐
    GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(gmExpertOffsetCnt + dstFlag);
    LocalTensor<float32_t> flagInLTTensor = dstExpBuf_.Get<float32_t>();
    LocalTensor<int16_t> expertCntTensor = dstExpBuf_.Get<int16_t>();
    GlobalTensor<int16_t> expertOffsetCntGMTensor;
    DataCopyParams expertCntCopyInParams{static_cast<uint16_t>(computeRow),
                                         static_cast<uint16_t>(sizeof(int16_t) * axisK_), 0, 0};
    DataCopyPadParams expertCntPadParams{false, 0, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    Duplicate<int16_t>(expertOffsetCntTensor_, static_cast<int16_t>(0), expertOffsetNum);
    WaitFlag<true>(flagAddr, flagInLTTensor, EXPERT_CNT_FLAG_SPACE, usedCoreNum);
    for (uint32_t index = 0; index < usedCoreNum; index++) {
        // 计算每个核心的专家偏移量
        uint64_t expertOffset = dstOffset + index * offsetPerMask;
        expertOffsetCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(gmExpertOffsetCnt + expertOffset));
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(expertCntTensor, expertOffsetCntGMTensor, expertCntCopyInParams, expertCntPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Add(expertOffsetCntTensor_, expertOffsetCntTensor_, expertCntTensor, expertOffsetNum);
    }
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::FillTriple(
    LocalTensor<ExpandXOutType> &xOutTensor, uint32_t srcRankIndex, uint32_t tokenIndex, uint32_t k, float expertScale)
{
    // 填充三元组 (srcRankIndex, tokenIndex, k) 和 expertScale
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    LocalTensor<float> xOutTfloat = xOutTensor.template ReinterpretCast<float>();
    xOutTint32(tokenQuantAlign_) = srcRankIndex;
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;
    xOutTint32(tokenQuantAlign_ + 2) = k;
    xOutTfloat(expertScaleAlign_) = expertScale;
    SyncFunc<AscendC::HardEvent::S_V>();
}

template <TemplateDispatchKFCTypeClass>
template <typename T>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WriteDataToWinOut(
    LocalTensor<T> &xOutTensor, GM_ADDR rankGM, uint32_t blockCnt)
{
    // 这里是先发数据，然后写flag，
    GlobalTensor<T> dataDstWinGMTensor;
    GlobalTensor<uint32_t> flagDstWinGMTensor;
    dataDstWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rankGM));
    flagDstWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(rankGM));
    DataCopyExtParams dataCopyOutParams = {static_cast<uint16_t>(blockCnt), SPLIT_BLOCK_DATA_SIZE, 0U, UB_ALIGN, 0U};
    DataCopyPad(dataDstWinGMTensor, xOutTensor, dataCopyOutParams);
    PipeBarrier<PIPE_MTE3>();
    DataCopyExtParams flagCopyOutParams = {static_cast<uint16_t>(blockCnt), UB_ALIGN, 0U, SPLIT_BLOCK_DATA_SIZE, 0U};
    DataCopyPad(flagDstWinGMTensor[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)], commFlagTensor_, flagCopyOutParams);
}

template <TemplateDispatchKFCTypeClass>
template <typename T>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WriteDataToWinOut(
    LocalTensor<T> &xOutTensor, LocalTensor<T> &xTempTensor, GM_ADDR rankGM, uint32_t blockCnt)
{
    // 这种方式是先把数据搬运到临时tensor进行组合，再从临时tensor搬运到输出tensor，能够提升性能
    DataCopyParams insertDataParams = {static_cast<uint16_t>(blockCnt), COMM_COPY_BLOCK, 0U, INSERT_DATA_STRIDE};
    DataCopyParams insertFlagParams = {static_cast<uint16_t>(blockCnt), FLAG_COPY_BLOCK, 0U, INSERT_FLAG_STRIDE};
    LocalTensor<uint32_t> xFlagTensor = xTempTensor.template ReinterpretCast<uint32_t>();
    DataCopy(xTempTensor, xOutTensor, insertDataParams);
    DataCopy(xFlagTensor[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)], commFlagTensor_, insertFlagParams);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    uint32_t copyCount = blockCnt * SERVER_STATE_ALIGN / sizeof(T);
    GlobalTensor<T> dataDstWinGMTensor;
    dataDstWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rankGM));
    DataCopy(dataDstWinGMTensor, xTempTensor, copyCount);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::QuantCleanTensor()
{
#if defined(__DAV_C310__)
    if constexpr (QuantMode > UNQUANT && ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT))) {
        // 由于MX以及PERGROUP量化在计算scales时每次搬入256字节数据，所以在token搬入前需要对空间填0，避免引入脏数据
        LocalTensor<uint8_t> singleByteTok = tokenData_.template ReinterpretCast<uint8_t>();
        Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XType));
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }
#endif
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::IsInSameServer(
    uint32_t targetRankId)
{
    return targetRankId / rankSizePerServer_ == serverId_;
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ProcessToken(
    GM_ADDR rankGM, uint32_t tokenIndex, uint32_t topKIndex, uint32_t expertIndex, uint32_t srcRankIndex,
    float expertScale)
{
    xInQueue_.EnQue(tokenData_);
    tokenData_ = xInQueue_.DeQue<XType>();
    xTempQueue_.EnQue(insertFlagTensor_);
    insertFlagTensor_ = xTempQueue_.DeQue<ExpandXOutType>();
    if constexpr (QuantMode > UNQUANT) { // 如果是量化模式
        xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
        quantInst_.QuantProcess(xOutTensor_, tokenData_, expertIndex, scalesCount_, scalesGMTensor_); // 量化
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
        FillTriple(xOutTensor_, srcRankIndex, tokenIndex, topKIndex, expertScale);
        WriteDataToWinOut(xOutTensor_, insertFlagTensor_, rankGM, blockCntPerTokenToExpert_);
        xOutQueue_.FreeTensor<ExpandXOutType>(xOutTensor_);
    } else { // 非量化模式直接搬运数据
        xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
        DataCopy(xOutTensor_, tokenData_, hAlignSize_ / sizeof(ExpandXOutType));
#if defined(__DAV_C310__)
        if constexpr (IsSmoothScaleExist) {
            DataCopyPadParams padParams = {true, 0, 0, 0};
            DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
            auto tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
            DataCopyPad(xOutTensor_[axisH_].template ReinterpretCast<uint8_t>(), tmp[tokenIndex * scaleInBytes_],
                        scaleInParams, padParams);
        }
#endif
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
        FillTriple(xOutTensor_, srcRankIndex, tokenIndex, topKIndex, expertScale);
        WriteDataToWinOut(xOutTensor_, insertFlagTensor_, rankGM, blockCntPerTokenToExpert_);
        xOutQueue_.FreeTensor<ExpandXOutType>(xOutTensor_);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendTokenToLocalServer(
    uint32_t firstAivId, uint32_t usedAivNum)
{
    uint32_t totalSendCnt = activeMaskBsCnt_;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(axisK_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0U};
    DataCopyPadParams copyPadExtParams{true, 0U, 0U, 0U};
    SplitToCore(totalSendCnt, firstAivId, usedAivNum, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= endTokenId || sendTokenNum == 0) {
        return;
    }
    CalExpertOffset(startTokenId, endTokenId); // 计算专家偏移量，即当前token是发送给当前专家的第几个token
    float expertScale = float(1.0);
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; tokenIndex++) {
        uint32_t offset = serverId_ * perServerMapCnt_ + tokenIndex;
        int32_t tokenSendOffset = tokenSendMap_.GetValue(offset);
        if (tokenSendOffset == -1) {
            continue;
        }
        QuantCleanTensor();
        xInQueue_.EnQue(tokenData_);
        tokenData_ = xInQueue_.DeQue<XType>();
        DataCopyPad(tokenData_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
        DataCopyPad(expertScalesTensor_, expertScalesGMTensor_[tokenIndex * axisK_], copyParams, expertScalePadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t topKIndex = 0; topKIndex < axisK_; topKIndex++) {
            uint32_t expertIndex = tokenIndex * axisK_ + topKIndex;
            uint32_t dstExpertId = expertIdsTensor_(expertIndex);                           // 目标专家ID
            uint32_t dstRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_; // 发往的卡号
            uint32_t expertCntIndex = (tokenIndex - startTokenId) * moeCntRowAlign_ + topKIndex;
            uint32_t curExpertCnt = expertOffsetCntTensor_(expertCntIndex); // 发送给当前专家的第几个token
            expertScale = expertScalesTensor_(topKIndex);
            GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(
                GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                (expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                hCommuSize_ * curExpertCnt); // 计算地址偏移
            ProcessToken(rankGM, tokenIndex, topKIndex, dstExpertId + sharedExpertNum_, epRankId_, expertScale);
        }
        for (int32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) { // 发共享专家
            uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_;
            uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + rankIDSharedGroup;
            uint32_t curExpertCnt = tokenIndex; // cnt即tokenid
            if (!IsInSameServer(dstRankId)) {   // 不同server
                continue;
            }
            GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                                       expertPerSizeOnWin_ * epRankId_ + hCommuSize_ * curExpertCnt);
            ProcessToken(rankGM, tokenIndex, axisK_ + shareIndex, shareIndex, epRankId_, expertScale);
        }
    }
    xInQueue_.FreeTensor<XType>(tokenData_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::FillQuadruple(
    LocalTensor<XType> &xOutTensor, uint32_t startTokenIndex, uint32_t tokenIndex)
{
    uint32_t realIndex = tokenIndex - startTokenIndex;
    uint64_t mask = axisK_;
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    LocalTensor<int16_t> xOutTint16 = xOutTensor.template ReinterpretCast<int16_t>();
    LocalTensor<float> xoutTfloat32 = xOutTensor.template ReinterpretCast<float>();
    xOutTint32(sendXTypeElemAlign_) = epRankId_;      // 源rankId
    xOutTint32(sendXTypeElemAlign_ + 1) = tokenIndex; // TokenID ,

    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(axisK_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> CopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xOutTint32[moeListAlign_], expertIdsGMTensor_[tokenIndex * axisK_], copyParams, CopyPadParams);
    DataCopy(xOutTint16[moeCntAlign_], expertOffsetCntTensor_[realIndex * moeCntRowAlign_], moeCntRowAlign_);
    DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0U};
    DataCopyPad(xoutTfloat32[moeExpertScalesAlign_], expertScalesGMTensor_[tokenIndex * axisK_], copyParams,
                expertScalePadParams); // 专家scale填充
    if (isExpertMaskFlag_) { // 如果有二维mask,需要把不用发送的专家的ID设置为一个无效值
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t index = 0; index < axisK_; index++) {
            uint32_t expertPos = moeListAlign_ + index;
            uint32_t offset = tokenIndex * axisK_ + index;
            if (!expertMaskInputTensor_(offset)) {
                xOutTint32(expertPos) = moeExpertNum_;
            }
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SingleTokenProcess(
    uint32_t startTokenIndex, uint32_t tokenIndex, uint32_t dstServerId, uint32_t cnt)
{
    LocalTensor<XType> sendTokenTensor = xSendBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyPadParams copyPadExtParams{true, 0U, 0U, 0U};
    DataCopyPad(sendTokenTensor, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
    FillQuadruple(sendTokenTensor, startTokenIndex, tokenIndex); // 填充对应的数据，包括专家ID和偏移cnt
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    GM_ADDR rankGM =
        (GetSendAddrBetweenServer(COMM_EP_IDX, dstServerId) + SERVER_STATE_ALIGN + cnt * sendTokenLengthAlign_);
    WriteDataToWinOut(sendTokenTensor, rankGM, blockCntPerToken_); // 将数据写入对应的发送区，等待接收server搬运
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ScatterToken(uint32_t startIndex,
                                                                                                 uint32_t endIndex,
                                                                                                 uint32_t serverId)
{
    for (uint32_t tokenIndex = startIndex; tokenIndex < endIndex; tokenIndex++) {
        uint32_t offset = serverId * perServerMapCnt_ + tokenIndex;
        int32_t tokenSendOffset = tokenSendMap_.GetValue(offset);
        if (tokenSendOffset == -1) {
            continue;
        }
        SingleTokenProcess(startIndex, tokenIndex, serverId, tokenSendOffset); // 将数据写入对应的发送区
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendTokenToRemoteServer(
    uint32_t serverIndex, uint32_t firstAivId, uint32_t usedAivNum)
{
    uint32_t totalSendCnt = activeMaskBsCnt_;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(totalSendCnt, firstAivId, usedAivNum, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= endTokenId || sendTokenNum == 0) {
        return;
    }
    CalExpertOffset(startTokenId, endTokenId); // 计算专家偏移量，即当前token是发送给当前专家的第几个token
    ScatterToken(startTokenId, endTokenId, serverIndex); // 将数据写入对应的发送区，等待接收server搬运
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::ScatterTokenToRemote()
{
    CalServerSendCnt<true>();
    if (aivId_ < aivUsedCumSum_ + aivUsedRemotComm_) {
        SetServerFlag(); // 需要先发送Token总数，每个核负责一个server，Token为0的时候也需要发送
        // 专门处理发送给其他server的Token
        return;
    }
    uint32_t startServerId, endServerId, useAivPerServer;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    uint32_t firstAivId;

    CalServerSplitCore(serverNum_, aivUsedCumSum_ + aivUsedRemotComm_, aivUsedScatter_, startServerId, endServerId,
                       useAivPerServer, firstAivId);
    InitScatterBuffer(startServerId);
    for (uint32_t serverIndex = startServerId; serverIndex < endServerId; serverIndex++) {
        if (serverCountTensor_.GetValue(serverIndex) == 0) { // 没有Token需要发送到这个server，
            continue;
        }
        if (serverIndex == serverId_) {
            SendTokenToLocalServer(firstAivId, useAivPerServer); // 处理发送给本server的Token
            continue;
        }
        SendTokenToRemoteServer(serverIndex, firstAivId, useAivPerServer); // 处理发送个其他server的Token
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitGatherTokenBuffer()
{
    tpipe_->Reset();
    InitQuantBuffer<CommMode::INTER_NODE>();
    uint32_t flagMaxRecvBuf = blockCntPerToken_ * UB_ALIGN;
    uint64_t mask[1] = {0x0101010101010101};
    uint8_t commRepeatTime = static_cast<uint8_t>(Ceil(maxBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256));
    uint8_t tokenRepeatTime = static_cast<uint8_t>(Ceil(blockCntPerToken_ * UB_ALIGN, SIZE_ALIGN_256));
    uint32_t commFlagSize = commRepeatTime * SIZE_ALIGN_256;
    uint32_t tokenFlagSize = tokenRepeatTime * SIZE_ALIGN_256;
    TBuf<> tokenRcvStatusBuf;
    TBuf<> tokenFlagBuf;
    TBuf<> cleanBuf;

    tpipe_->InitBuffer(tokenInQueue_, BUFFER_NUM, sendTokenLengthAlign_); // token输入队列
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hAlignSize_);               // 14K * 2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hOutAlignUbSize_);         // 7K * 2 + 32 + 6
    tpipe_->InitBuffer(flagBuf_, commFlagSize);
    tpipe_->InitBuffer(tokenRcvStatusBuf, TOKEN_RECV_CNT_MAX * sizeof(uint32_t)); // 一个server最多发512个token给
    tpipe_->InitBuffer(tokenFlagBuf, flagMaxRecvBuf);
    tpipe_->InitBuffer(cleanBuf, tokenFlagSize);              // 每个Token的flag个数为 blockCntPerToken_
    tpipe_->InitBuffer(xTempQueue_, BUFFER_NUM, hCommuSize_); // 17.5KB

    commFlagTensor_ = flagBuf_.Get<uint32_t>();
    tokenData_ = xInQueue_.AllocTensor<XType>();
    insertFlagTensor_ = xTempQueue_.AllocTensor<ExpandXOutType>();
    recvTmpTensor_ = tokenInQueue_.AllocTensor<XType>();
    tokenRcvStatus_ = tokenRcvStatusBuf.Get<uint32_t>();
    flagRecvTensor_ = tokenFlagBuf.Get<float32_t>();
    statusCleanFp32Tensor_ = cleanBuf.Get<float32_t>();

    Duplicate<uint32_t>(commFlagTensor_, 0x3F800000, mask, commRepeatTime, uint16_t(1),
                        uint8_t(8)); // 0x3F800000是float的1
    Duplicate<float32_t>(statusCleanFp32Tensor_, float32_t(0), mask, tokenRepeatTime, uint16_t(1), uint8_t(8));
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitStatusFlag(
    uint32_t serverIdx, TBuf<> &serverFlagBuf, uint32_t &rcvTokenCnt)
{
    LocalTensor<uint32_t> statusFlagTensor, tokenCntTensor;
    GlobalTensor<uint32_t> statusCntGlobal;
    GM_ADDR wAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx); // 对应server接收区地址
    statusCntGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(wAddr));
    statusFlagTensor = serverFlagBuf.Get<uint32_t>();
    tokenCntTensor = serverFlagBuf.Get<uint32_t>();
    while (true) {
        DataCopy(statusFlagTensor, statusCntGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t)],
                 (SERVER_STATE_ALIGN - SPLIT_BLOCK_DATA_SIZE) / sizeof(uint32_t)); // 搬运后续的8个数据
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t flag = statusFlagTensor.GetValue(0);
        if (flag == 1) { // 等待flag
            break;
        }
    }
    DataCopy(tokenCntTensor, statusCntGlobal, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    rcvTokenCnt = tokenCntTensor.GetValue(0); // 获取接收到的token数量
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SendToExpert(uint32_t rcvCnt)
{
    QuantCleanTensor();
    tokenInQueue_.EnQue(recvTmpTensor_);
    recvTmpTensor_ = tokenInQueue_.DeQue<XType>();
    xInQueue_.EnQue(tokenData_);
    tokenData_ = xInQueue_.DeQue<XType>();
    DataCopy(tokenData_, recvTmpTensor_, hAlignSize_ / sizeof(XType));
    LocalTensor<int16_t> xInTensorInt16 = recvTmpTensor_.template ReinterpretCast<int16_t>();
    LocalTensor<uint32_t> xInTensor = recvTmpTensor_.template ReinterpretCast<uint32_t>();
    LocalTensor<float> xInToFloatTensor = recvTmpTensor_.template ReinterpretCast<float>();
    uint32_t srcRankIndex = xInTensor(sendXTypeElemAlign_);
    uint32_t tokenIndex = xInTensor(sendXTypeElemAlign_ + 1);
    float expertScale = float(1.0);
    for (int32_t topKIndex = 0; topKIndex < axisK_; topKIndex++) {                      // 发moe专家
        uint32_t dstExpertId = xInTensor(moeListAlign_ + topKIndex);                    // 取出专家id
        uint32_t dstRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_; // 发往的卡号
        uint32_t curExpertCnt = xInTensorInt16(moeCntAlign_ + topKIndex);               // 取出cnt
        expertScale = xInToFloatTensor(moeExpertScalesAlign_ + topKIndex);
        if (dstExpertId >= moeExpertNum_ || !IsInSameServer(dstRankId)) { // 无效数据或者不在本server
            continue;
        }
        GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(
            GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
            (expertPerSizeOnWin_ * (srcRankIndex * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
            hCommuSize_ * curExpertCnt); // 计算地址偏移
        ProcessToken(rankGM, tokenIndex, topKIndex, dstExpertId + sharedExpertNum_, srcRankIndex, expertScale);
    }

    tokenInQueue_.EnQue(recvTmpTensor_);

    for (int32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) { // 发共享专家
        uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_;
        uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + rankIDSharedGroup;
        uint32_t curExpertCnt = rcvCnt;   // cnt即tokenid
        if (!IsInSameServer(dstRankId)) { // 不同server
            continue;
        }
        GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(GetWindAddrByRankId(COMM_EP_IDX, dstRankId) +
                                                   expertPerSizeOnWin_ * srcRankIndex + hCommuSize_ * curExpertCnt);
        ProcessToken(rankGM, tokenIndex, axisK_ + shareIndex, shareIndex, srcRankIndex, expertScale);
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitToken(uint32_t tokenCnt,
                                                                                              uint32_t serverIdx,
                                                                                              uint32_t startTokenIdx)
{
    uint32_t rcvStatusIndex = 0;
    uint32_t rcvTokenSum = 0;
    uint32_t processTokenNumPerBatch = 1;
    uint32_t count = 0;
    while (true) {
        if (tokenRcvStatus_(rcvStatusIndex) == 1) {
            rcvStatusIndex = (rcvStatusIndex + 1) % tokenCnt; // 轮询查询每个有效的index
            continue;
        }
        GM_ADDR flagAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx) + SERVER_STATE_ALIGN +
                           (rcvStatusIndex + startTokenIdx) * sendTokenLengthAlign_ +
                           SPLIT_BLOCK_DATA_SIZE; // 第一个flag起始地址
        bool isArrived = WaitFlag<false>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, blockCntPerToken_);
        if (isArrived) {                                           // 处理到达的情况
            uint32_t dstPosition = rcvStatusIndex + startTokenIdx; // 需要拷贝的Token的位置
            uint32_t processTokenBlock = blockCntPerToken_ * processTokenNumPerBatch;
            GM_ADDR dataAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverIdx) + SERVER_STATE_ALIGN +
                               dstPosition * sendTokenLengthAlign_;
            recvTmpTensor_ = tokenInQueue_.DeQue<XType>();
            ReadDataFromWindows<XType>(dataAddr, processTokenBlock, recvTmpTensor_); // 读取数据
            SendToExpert(dstPosition);                                               // 处理数据并发送给专家
            tokenRcvStatus_(rcvStatusIndex) = 1;
            rcvTokenSum++;
            if (rcvTokenSum == tokenCnt) {
                break;
            }
            continue;
        }
        rcvStatusIndex = (rcvStatusIndex + 1) % tokenCnt;
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CleanTokenFlag(
    uint32_t serverId, uint32_t startTokenId, uint32_t cleanTokenNum)
{
    uint32_t cleanUpNum = blockCntPerToken_ * cleanTokenNum;
    GM_ADDR dataCleanAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverId) + SERVER_STATE_ALIGN +
                            startTokenId * sendTokenLengthAlign_ + SPLIT_BLOCK_DATA_SIZE;
    GM_ADDR flagCleanAddr = GetReceiveAddrBetweenServer(COMM_EP_IDX, serverId);
    GlobalTensor<float32_t> cleanGlobal;
    cleanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float32_t *>(dataCleanAddr));
    DataCopyExtParams cleanUpParams = {uint16_t(cleanUpNum), sizeof(float32_t), 0U,
                                       SERVER_STATE_ALIGN - sizeof(int32_t), 0U}; // 清理512B的最后32B的最前面的4B
    DataCopyPad(cleanGlobal, statusCleanFp32Tensor_, cleanUpParams);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::GatherTokenFrRemote()
{
    InitGatherTokenBuffer();
    TBuf<> serverFlagBuf;
    uint32_t startServerId, endServerId, useAivPerServer;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    uint32_t firstAivId;
    CalServerSplitCore(serverNum_, aivUsedCumSum_ + aivUsedScatter_ + aivUsedRemotComm_, aivUsedGather_, startServerId,
                       endServerId, useAivPerServer, firstAivId);
    tpipe_->InitBuffer(serverFlagBuf, UB_ALIGN);
    for (uint32_t serverIndex = startServerId; serverIndex < endServerId; serverIndex++) {
        if (serverIndex == serverId_) { // 本server不用从其他server接收token，直接跳过
            continue;
        }
        uint32_t tokenCnt;
        WaitStatusFlag(serverIndex, serverFlagBuf, tokenCnt); // 等待其他server的token数量
        if (tokenCnt == 0) { // 如果没有token到达，直接跳过，在发送时Token为0时，也必须发送cnt总数
            continue;
        }
        SplitToCore(tokenCnt, firstAivId, useAivPerServer, startTokenId, endTokenId, sendTokenNum);
        if (startTokenId >= endTokenId || sendTokenNum == 0) {
            continue;
        }
        Duplicate<uint32_t>(tokenRcvStatus_, 0, TOKEN_RECV_CNT_MAX);
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenInQueue_.EnQue(recvTmpTensor_);
        WaitToken(sendTokenNum, serverIndex, startTokenId);      // 等待token到达,并进行server内通信
        CleanTokenFlag(serverIndex, startTokenId, sendTokenNum); // 清理flag，准备下一轮接收
    }
    xInQueue_.FreeTensor<XType>(tokenData_);
    tokenInQueue_.FreeTensor<XType>(recvTmpTensor_);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::InitLocalCopyBuffer(
    uint32_t processNum)
{
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf, expertMapBuf, expertFinishBuf, expertLeftBuf, flagMaskBuf, tBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t expInfoSize = Ceil(processNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tBufRealSize_ = MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) - (expInfoSize * 3) -
                    BUFFER_NUM * UB_ALIGN; // 3为expInfoSize大小buffer申请个数
    uint32_t maxCopyTokenCnt = tBufRealSize_ / hCommuSize_;
    uint32_t flagMaxRecvNum = (blockCntPerTokenToExpert_ * maxCopyTokenCnt * UB_ALIGN) / sizeof(uint32_t);

    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN); // max CompareScalar
    tpipe_->InitBuffer(tBuf, tBufRealSize_);

    xTmpTensor_ = tBuf.Get<ExpandXOutType>();
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();

    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();

    LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::S_V>();
    flagRecvTensor_ = tBuf.GetWithOffset<float32_t>(flagMaxRecvNum, 0); // buf复用
    cleanTensor_ = tBuf.GetWithOffset<int32_t>(flagMaxRecvNum * UB_ALIGN, 0);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitCumSumFlag()
{
    // Check cumsum is finished
    int32_t cumSumFlag = 0;
    int32_t targetFlag = aivUsedCumSum_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (CUMSUM_FLAG_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = aivUsedCumSum_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();
    while (true) { // 等待server内cumsum完成
        DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    // Clean flag for next round
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::SetValidExpertInfo(
    uint32_t startProcessId, uint32_t endProcessId, uint32_t processNum, uint32_t &validNum)
{
    // 获取cumSum
    uint32_t expInfoSize = Ceil(processNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    for (uint32_t index = startProcessId; index < endProcessId;
         index++) { // 从sendCnt中挑选当前有发送过来的卡的token数量
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

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::CopyOut(
    LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount)
{
    GlobalTensor<ExpandXOutType> dataFlagGlobal, expandXOutGlobal;
    dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXOutType *>(wAddr));
    expandXOutGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXOutType *>(expandXOutGM_) + (dstPosition)*axisH_);
    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerTokenToExpert_ * arriveCount),
                                      static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint16_t>(UB_ALIGN), 0};
    DataCopyExtParams scalesCopyParams{
        uint16_t(arriveCount), static_cast<uint32_t>(scaleOutBytes_),
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams tokenCopyParams{
        uint16_t(arriveCount), hOutSize_,
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{
        uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyExtParams dataCopyexpandScaleParams{
        uint16_t(arriveCount), sizeof(float),
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};

    DataCopyPad(xTmpTensor_, dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(ExpandXOutType)],
                srcTokenCopyParams, srcTokenPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    quantInst_.CopyScalesToOut(dstPosition, scaleOutBytes_, xTmpTensor_, scalesCopyParams);
    DataCopyPad(expandXOutGlobal, xTmpTensor_, tokenCopyParams); // 拷贝token数据
    DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO], xOutInt32Tensor[tokenQuantAlign_],
                expandIdxCopyParams); // 拷贝三元组信息
    if (!isShareExpertRankFlag_) {
        LocalTensor<float> xOutFloatTensor;
        xOutFloatTensor = xOutInt32Tensor.template ReinterpretCast<float>();
        DataCopyPad(expandScalesOutGMTensor_[dstPosition], xOutFloatTensor[expertScaleAlign_],
                    dataCopyexpandScaleParams); // 拷贝expert专家系数
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::WaitAndFormatOutput(
    uint32_t validNum)
{
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpRankId, dstPosition, arriveCount, copyCnt, srcDataBlockIdx;
    uint32_t maxCopyTokenCnt = tBufRealSize_ / hCommuSize_;
    uint32_t flagMaxRecvNum = (blockCntPerTokenToExpert_ * maxCopyTokenCnt * UB_ALIGN) / sizeof(uint32_t);
    uint32_t gatherOutSize =
        Ceil(blockCntPerTokenToExpert_ * maxCopyTokenCnt * sizeof(uint32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    GlobalTensor<int32_t> cleanGlobal;
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    while (true) {
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum;     // 轮询查询每个有效的index
            continue;
        }
        srcExpRankId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ? maxCopyTokenCnt : expertLeftNumTensor_(index);
        srcDataBlockIdx = srcExpRankId % epWorldSize_ * localExpertNum + srcExpRankId / epWorldSize_;
        int32_t beginIdx = expertFinishNumTensor_(index);
        GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(windowGM_ + srcDataBlockIdx * expertPerSizeOnWin_ +
                                                     beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE);
        uint32_t flagNum = blockCntPerTokenToExpert_ * uint32_t(copyCnt);
        bool isArrived = WaitFlag<false>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, flagNum); // 等待专家数据到达
        if (isArrived) {
            dstPosition = srcExpRankId != 0 ? sendCntTensor_(srcExpRankId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = reinterpret_cast<GM_ADDR>(windowGM_ + srcDataBlockIdx * expertPerSizeOnWin_);
            CopyOut(xOutInt32Tensor, wAddr, index, dstPosition, copyCnt);
            expertFinishNumTensor_(index) += copyCnt;
            expertLeftNumTensor_(index) -= copyCnt;
            if (expertLeftNumTensor_(index) == 0) {
                uint32_t cleanUpNum = expertFinishNumTensor_(index) * blockCntPerTokenToExpert_;
                DataCopyExtParams cleanUoParams = {uint16_t(cleanUpNum), sizeof(int32_t), 0U,
                                                   SERVER_STATE_ALIGN - sizeof(int32_t), 0U};
                cleanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(wAddr));
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                Duplicate<int32_t>(cleanTensor_, 0, UB_ALIGN_DATA_COUNT);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopyPad(cleanGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(int32_t)], cleanTensor_, cleanUoParams);
                finishNum++;
            }
            PipeBarrier<PIPE_ALL>();
        } else {
            index = (index + 1) % validNum;
        }
        if (validNum == finishNum) {
            break;
        }
    }
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::LocalWindowCopy()
{
    uint32_t startProcessId, endProcessId, processNum;
    uint32_t validNum = 0;
    SplitToCore(rscvStatusNum_, 0, aivNum_, startProcessId, endProcessId, processNum);
    if (processNum == 0) {
        return;
    }
    InitLocalCopyBuffer(processNum);
    WaitCumSumFlag(); // 软同步
    SetValidExpertInfo(startProcessId, endProcessId, processNum, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    WaitAndFormatOutput(validNum); // 等待并格式化输出，包含等待专家数据到达和搬运数据到输出区
}

template <TemplateDispatchKFCTypeClass>
__aicore__ inline void MoeDistributeDispatchV2HostKfc<TemplateDispatchKFCTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        CalTokenMask();
        if (aivId_ < aivUsedCumSum_) { // cnt计算 + cumsum
            CalCumSum();
        } else {
            CalTokenSendExpertOffset(); // 计算每个token对应的expert偏移量
            if (aivId_ < aivUsedCumSum_ + aivUsedRemotComm_ + aivUsedScatter_) {
                ScatterTokenToRemote(); // 发送token到其他server
            } else {
                GatherTokenFrRemote(); // 从其他server接收token
            }
        }
        // localWindowCopy中包含reset操作，需确保前面操作完成
        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy(); // 本卡上专家数据连续化，输出expandX/scales/expandIdx
    }
}

} // namespace MoeDistributeDispatchV2HostKfcImpl
#endif // MoeDistributeDispatchV2HostKfcImpl
