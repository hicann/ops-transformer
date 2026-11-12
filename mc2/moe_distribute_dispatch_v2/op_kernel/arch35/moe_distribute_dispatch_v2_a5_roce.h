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
 * \file moe_distribute_dispatch_v2_a5_roce.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_A5_ROCE_H
#define MOE_DISTRIBUTE_DISPATCH_V2_A5_ROCE_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
// This RoCE implementation requires the Ascend 950 Hcomm APIs.
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/reduce.h"
#include "adv_api/math/xor.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/hcomm/hcomm.h"
#include "kernel_tiling/kernel_tiling.h"
#include "utils/std/algorithm.h"
#include "../moe_distribute_dispatch_v2_tiling.h"
#include "../moe_distribute_v2_constant.h"
#include "../moe_distribute_dispatch_v2_quant.h"
#include "../moe_distribute_elastic.h"
#include "../moe_distribute_v2_base.h"
#include "../check_winsize.h"
#include "../hccl_context_holder.h"
#include "../mc2_moe_context_holder.h"
#if __has_include("../../common/mc2_moe_context.h")
#include "../../common/mc2_moe_context.h"
#include "../../common/moe_distribute_base.h"
#else
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#endif

namespace Mc2Kernel {
constexpr uint32_t MAX_UB_SIZE_ROCE = 248U * 1024U; // UB大小
constexpr uint64_t SERVER_STATE_ALIGN = 512UL;
constexpr uint64_t SERVER_TOKEN_AREA_OFFSET = SERVER_STATE_ALIGN;
constexpr uint32_t TOKEN_RECV_CNT_MAX = 512U;
constexpr uint32_t INSERT_DATA_STRIDE = 1U;
constexpr uint32_t COMM_COPY_BLOCK = 15U;
constexpr uint32_t FLAG_COPY_BLOCK = 1U;
constexpr uint32_t INSERT_FLAG_STRIDE = 15U;
constexpr uint32_t COMM_FLAG_COPY_STRIDE = 504;
constexpr uint32_t EXPERT_CNT_FLAG_SPACE = 0;     // 专家结果flag全部顺序存放
constexpr uint32_t BUFPOOL_MAX_BUFFER_ID = 16;    // Tbufpool 最大buffer id
constexpr uint32_t BUFPOLL_MAX_SIZE = 192 * 1024; // 为pool外常驻buffer预留至少56KB
constexpr uint32_t DOUBLE_BUFFER = 2;             // 双缓冲
constexpr uint32_t ROCE_DATA_STATE_NUM = 2;       // RoCE区按dataState_对半分区
constexpr uint32_t ROCE_HCOMM_INIT_SIZE = 512U;   // Hcomm RoCE普通接口的UB工作区(HCOMM_UB_BUF_SIZE)
// 每个对端server两条RoCE channel：0=token收发(count头块+token记录)，1=专家计数(cumsum)，避免两类核并发操作同一通道。
constexpr uint32_t ROCE_CHAN_TOKEN = 0U;
constexpr uint32_t ROCE_CHAN_EXPERT_CNT = 1U;
constexpr uint32_t ROCE_TOKEN_BATCH_SIZE = 5U;
constexpr uint32_t ROCE_WQEBB_BYTES = 64U;
constexpr uint32_t ROCE_BATCH_UB_BYTES = ROCE_TOKEN_BATCH_SIZE * ROCE_WQEBB_BYTES;
constexpr uint32_t FLAG_COMPARE_ALIGN = 256U;
#define FLOAT_OVERFLOW_MODE_CTRL 60

enum class CommMode {
    INTRA_NODE = 0, // server内
    INTER_NODE = 1  // server间
};

#define RoceDispatchTypeClass \
    typename ContextHolder, typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define RoceDispatchTypeFunc ContextHolder, XType, ExpandXOutType, QuantMode, IsSmoothScaleExist

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace Mc2Aclnn;

template <RoceDispatchTypeClass>
class MoeDistributeDispatchV2Roce {
public:
    using XInType = typename std::conditional<
        (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value), uint8_t, XType>::type;
    using XOutType = typename std::conditional<(Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) ||
                                                   (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
                                               uint8_t, ExpandXOutType>::type;
    using XCommInType = typename std::conditional<(QuantMode > UNQUANT) && IsSmoothScaleExist, XInType, XOutType>::type;
    using XCommOutType = XOutType;
    __aicore__ inline MoeDistributeDispatchV2Roce(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                GM_ADDR expertScales, GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
                                GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe* pipe,
                                const MoeDistributeDispatchV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitRecieveTilingContext(GM_ADDR expandXOut, TPipe* pipe,
                                                    const MoeDistributeDispatchV2TilingData* tilingData);
    __aicore__ inline void InitCommInfo(GM_ADDR mc2Context);
    __aicore__ inline void InitComputeInfo();
    __aicore__ inline void InitSetWindows(const MoeDistributeDispatchV2TilingData* tilingData);
    __aicore__ inline void InitExtraInfo();
    __aicore__ inline void InitMaskInfo();
    __aicore__ inline void InitDispatchBetweenServerInfo();
    __aicore__ inline void CalcSplitRange(uint32_t totalCnt, uint32_t coreNum, uint32_t relativeId, uint32_t& startIdx,
                                          uint32_t& endIdx, uint32_t& perCnt);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t startAivId, uint32_t curUseAivNum,
                                       uint32_t& startTokenId, uint32_t& endTokenId, uint32_t& sendTokenNum);
    __aicore__ inline void InitMaskReSource();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void CalTokenMask();
    __aicore__ inline void CalExpertTotalSendCnt();
    __aicore__ inline void SendExpertCnt();
    __aicore__ inline void FillServerExpertCounts(uint32_t serverIndex, LocalTensor<int32_t>& counts);
    __aicore__ inline void InitStatusTensor();
    __aicore__ inline void SetStatus();
    __aicore__ inline void InitCalCumSumBuffer();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float>& gatherMaskOutTensor,
                                            LocalTensor<uint32_t>& gatherTmpTensor,
                                            LocalTensor<float>& statusSumOutTensor);
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t>& outLocal, uint32_t totalCount);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WriteExpertOffset();
    __aicore__ inline void CalTokenSendExpertOffset();
    template <typename T, bool disableL2 = false>
    __aicore__ inline void ReadDataFromWindows(GM_ADDR dataAddr, uint32_t blockCnt, LocalTensor<T>& xInTensor);
    __aicore__ inline void FillQuadruple(LocalTensor<XCommInType>& xOutTensor, uint32_t startTokenIndex,
                                         uint32_t tokenIndex);
    template <typename T, bool disableL2 = false>
    __aicore__ inline void WriteDataToWinOut(LocalTensor<T>& xOutTensor, LocalTensor<T>& xTempTensor, GM_ADDR rankGM,
                                             uint32_t blockCnt);
    __aicore__ inline void ScatterToken(uint32_t startIndex, uint32_t endIndex, uint32_t serverId);
    __aicore__ inline void CalExpertOffset(uint32_t startTokenId, uint32_t endTokenId);
    __aicore__ inline void SendServerCnt(uint32_t startServerNum, uint32_t endServerNum);
    __aicore__ inline void HcclCommSend(uint32_t startServerNum, uint32_t endServerNum);
    __aicore__ inline void SendServerTokens(uint32_t serverIndex, uint32_t mapOffset, GM_ADDR localTokenBase,
                                            LocalTensor<uint8_t> batchUb, LocalTensor<uint64_t> flagTensor);
    template <bool isContinueWait>
    __aicore__ inline bool WaitFlag(GM_ADDR flagAddr, LocalTensor<uint64_t> flagInLTTensor, uint32_t srcStride,
                                    uint32_t waitNum);
    __aicore__ inline uint32_t GetFlagCheckBufferSize(uint32_t flagNum);
    __aicore__ inline bool CheckFlagBySequence(LocalTensor<uint64_t> flagTensor, uint32_t flagNum);
    __aicore__ inline void CalServerSendCnt(uint32_t startServerId, uint32_t endServerId);
    __aicore__ inline void CalServerSplitCore(uint32_t serverNum, uint32_t startAivId, uint32_t useAivNum,
                                              uint32_t& startServerId, uint32_t& endServerId, uint32_t& useAivPerServer,
                                              uint32_t& firstAivId);
    __aicore__ inline void InitScatterBuffer();
    __aicore__ inline void ScatterTokenToRemote();
    __aicore__ inline void InitGatherTokenBuffer();
    __aicore__ inline void WaitStatusFlag(uint32_t serverIdx, TBuf<>& serverFlagBuf, uint32_t& rcvTokenCnt);
    __aicore__ inline void FillTriple(LocalTensor<XOutType>& xOutTensor, uint32_t srcRankIndex, uint32_t tokenIndex,
                                      uint32_t k, float expertScale);
    __aicore__ inline void FillKAndScale(LocalTensor<XOutType>& xOutTensor, uint32_t k, float expertScale);
    __aicore__ inline void ProcessToken(LocalTensor<XCommInType>& xInTensor, GM_ADDR rankGM, uint32_t topKIndex,
                                        uint32_t expertIndex, uint32_t srcRankIndex, uint32_t tokenIndex,
                                        float expertScale);
    __aicore__ inline bool IsInSameServer(uint32_t targetRankId);
    __aicore__ inline void QuantCleanTensor(LocalTensor<XInType>& tokenData);
    template <CommMode mode>
    __aicore__ inline void InitQuantBuffer();
    __aicore__ inline void SendToExpert(LocalTensor<XCommInType>& recvTmpTensor, uint32_t rcvCnt);
    __aicore__ inline void WaitToken(uint32_t tokenCnt, uint32_t serverIdx, uint32_t startTokenIdx);
    __aicore__ inline void StoreSharedTokenCount(uint32_t serverIdx, uint32_t tokenCount, TBuf<>& serverFlagBuf);
    __aicore__ inline uint32_t GetCombineBitmapStart(uint32_t serverIdx, uint32_t startTokenIdx, TBuf<>& serverFlagBuf);
    __aicore__ inline void StoreCombineBitmap(uint32_t serverIdx, uint32_t startToken, uint32_t endToken);
    __aicore__ inline void GatherServerTokens(uint32_t serverIdx, uint32_t firstAivId, uint32_t useAivPerServer,
                                              TBuf<>& serverFlagBuf);
    __aicore__ inline void GatherTokenFrRemote();
    __aicore__ inline void StoreTokenMask();
    __aicore__ inline void HcclTokenDispatch();
    __aicore__ inline void ReadTokenData(uint32_t tokenIndex, LocalTensor<XInType>& tokenData);
    __aicore__ inline void QuantToken(uint32_t tokenIndex, LocalTensor<XInType>& xInTensor,
                                      LocalTensor<XCommInType>& xOutTensor, uint32_t expertIndex = 0);
    __aicore__ inline void WriteNotifyFlag(uint32_t tokenIndex);
    __aicore__ inline void WriteTokenWithNotify(uint32_t startTokenIndex, uint32_t tokenIndex,
                                                LocalTensor<XCommInType>& xOutTensor);
    __aicore__ inline void InitLocalCopyBuffer(uint32_t processNum);
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void SetValidExpertInfo(uint32_t startProcessId, uint32_t endProcessId, uint32_t processNum,
                                              uint32_t& validNum);
    __aicore__ inline void CopyOut(LocalTensor<XOutType> tokenTensor, uint32_t dstPosition, uint32_t tokenCount);
    __aicore__ inline void WaitAndFormatOutput(uint32_t validNum);
    __aicore__ inline void LocalWindowCopy();

    // RoCE注册区取址：本地取独立注册区，远端按RoCE注册tag从HcclChannelGetRemoteMems查找，
    // 与UB_MEM窗的epHcclBuffer_地址不可混用；半区容量取自context的roceBufferSize。
    __aicore__ inline GM_ADDR GetRoceDispatchAddr(GM_ADDR roceBufferBase) const
    {
        return roceBufferBase + static_cast<uint64_t>(dataState_) * roceStateCapacity_;
    }

    // 以下地址均以GetRoceDispatchAddr返回的当前半区起点为基准；serverId须小于serverNum_。
    __aicore__ inline GM_ADDR GetRoceServerCntSendAddr(GM_ADDR dispatchBase, uint32_t targetServerId) const
    {
        return dispatchBase + static_cast<uint64_t>(targetServerId) * SERVER_STATE_ALIGN;
    }

    __aicore__ inline GM_ADDR GetRoceExpertCntSendAddr(GM_ADDR dispatchBase, uint32_t targetServerId) const
    {
        return dispatchBase + static_cast<uint64_t>(serverNum_) * SERVER_STATE_ALIGN +
               static_cast<uint64_t>(targetServerId) * expertCntSlotSize_;
    }

    // 接收区按 source server 索引：框间只有与本地同 localIndex 的远端卡会发给本卡，
    // 一个 server 只需一个接收槽，容量按本卡单次最多发送的 token 数(axisMaxBS_)计算。
    __aicore__ inline GM_ADDR GetRoceReceiveAddr(GM_ADDR dispatchBase, uint32_t sourceServerId) const
    {
        return dispatchBase + roceSendAreaSize_ + static_cast<uint64_t>(sourceServerId) * roceReceiveSlotSize_;
    }

    __aicore__ inline GM_ADDR GetRoceExpertCntReceiveAddr(GM_ADDR dispatchBase, uint32_t sourceServerId) const
    {
        return GetRoceReceiveAddr(dispatchBase, sourceServerId) + SERVER_TOKEN_AREA_OFFSET +
               static_cast<uint64_t>(axisMaxBS_) * serverRecordAlignLen_;
    }

    // server的RoCE注册区首地址：本地项为本卡，远端项为同localIndex的对端卡
    __aicore__ inline GM_ADDR GetRoceBaseByServerId(uint32_t serverIndex) const
    {
        return reinterpret_cast<GM_ADDR>(ctx_.GetRoceBufferByServerId(serverIndex));
    }

    // 框间token/状态收发全部走RoCE注册区寻址（发送见HcclCommSend/SendServerCnt，接收见WaitToken/WaitStatusFlag），
    // 原workspace模拟用的GetXxxAddrBetweenServer已删除。

    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindAddrByRankId(rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindStateAddrByRankId(rankId, epRankIdOriginal_) + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe* tpipe_{nullptr};
    GlobalTensor<XInType> xGMTensor_;
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
    GlobalTensor<uint16_t> combineServerInfoGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<int32_t> offsetGMTensor_;

    LocalTensor<float> floatLocalTemp_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<uint32_t> serverCountTensor_;
    LocalTensor<int32_t> tokenSendMap_;
    LocalTensor<uint16_t> combineBitmapTensor_;
    LocalTensor<uint64_t> flagRecvTensor_;
    LocalTensor<uint32_t> tokenRcvStatus_;
    LocalTensor<int16_t> expertOffsetCntTensor_;
    LocalTensor<bool> expertMaskInputTensor_;
    LocalTensor<uint64_t> commFlagTensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    LocalTensor<float> expertScalesTensor_;
    LocalTensor<bool> validLocalTokenTensor_;
    LocalTensor<bool> validRemoteTokenTensor_;

    TBuf<> expertScalesBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_; // 辅助gather tensor定义
    TBuf<> rowMaxBuf_;
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
    TBuf<> combineBitmapBuf_;
    TBuf<> expertOffsetCntBuf_;
    TBuf<> xSendBuf_;
    TBuf<> flagBuf_;
    TBuf<> validLocalSrvTokenBuf_;
    TBuf<> validRemoteSrvTokenBuf_;

    TQue<QuePosition::VECIN, 1> tokenInQueue_;                     // token输入队列
    TQue<QuePosition::VECIN, 1> xInQueue_;                         // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                       // 量化使用，量化后的输出
    TQue<QuePosition::VECOUT, 1> xTempQueue_;                      // 用于填充flag
    TBufPool<TPosition::VECCALC, BUFPOOL_MAX_BUFFER_ID> tbufPool_; // 用于存储tbuf的pool

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
    GM_ADDR tokenIdOffsetGM_;

    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t combineBitmapTokenNum_{0};
    uint32_t combineBitmapEndToken_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t bufferId_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    int32_t epRankId_{0};
    int32_t serverId_{0};          // 当前卡所在server编号
    int32_t rankSizePerServer_{0}; // 一个server内的rank数 epWorldSize_ -> rankSizePerServer_
    // int32_t epRankIdInServer_{0}; // server内局部编号 epRankId_ -> epRankIdInServer_
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
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hAlignSize_{0};
    uint32_t hCommuSize_{0};
    uint32_t serverTripleOffset_{0};       // server窗记录：srcRank/tokenIndex起始，int32元素偏移
    uint32_t serverExpertIdsOffset_{0};    // server窗记录：expertIds列表起始，int32元素偏移
    uint32_t serverMoeCntOffset_{0};       // server窗记录：moeCnt列表起始，int16元素偏移
    uint32_t serverExpertScalesOffset_{0}; // server窗记录：expertScales起始，float元素偏移
    uint32_t serverRecordLen_{0};          // server窗单条记录字节长度（数据+三元组+元数据）
    uint32_t serverRecordBlockCnt_{0};     // server窗单条记录480B块数（写窗/等flag用）
    uint32_t serverRecordAlignLen_{0};     // server窗单条记录512B对齐后长度（记录槽位步长）
    uint64_t serverCommAreaSize_{0}; // 每server框间通信区大小=512B状态块+512token记录预留(TOKEN_RECV_CNT_MAX)
    uint64_t expertCntSlotSize_{0};   // 每server专家计数槽大小，按实际通信block数计算
    uint64_t roceSendAreaSize_{0};    // RoCE发送区大小：server计数区和专家计数区之和
    uint64_t roceReceiveSlotSize_{0}; // RoCE单个source server接收区大小，包含计数、token和专家计数
    uint64_t roceBufferSize_{0};      // RoCE注册区容量，取自context(与URMA窗口winSize同语义)
    uint64_t roceStateCapacity_{0};   // RoCE单个0/1半区容量=roceBufferSize_/ROCE_DATA_STATE_NUM
    uint32_t statusBufCntAlign_{0};
    uint32_t dataState_{0};
    uint64_t opSequence_{0}; // 本轮绝对轮次R(win状态块opCnt+1，InitWinState维护)，通信flag统一使用
    uint32_t axisBsAlignSize_{0};
    uint32_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_; // 接收Win区块数
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    uint64_t totalWinSizeEp_{0};
    uint32_t expertTokenNumsType_{1};
    uint32_t stateOffset_{0};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t maxSize_{0};
    uint32_t blockCntExpertPerServer_{0};
    uint32_t blockCntPerTokenToExpert_{0};
    uint32_t maxBlockCnt_{0};
    uint32_t serverNum_{0};
    uint32_t hScaleIdxSize_{0};
    uint32_t tokenAlign_{0};
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
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};

    AscendC::Hcomm<COMM_PROTOCOL_ROCE> hcomm_; // 框间RoCE通信对象
    ContextHolder ctx_;
    MoeDistributeDispatchV2Quant<XInType, ExpandXOutType, XOutType, QuantMode, IsSmoothScaleExist> quantInst_;
};

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitRecieveTilingContext(
    GM_ADDR expandXOut, TPipe* pipe, const MoeDistributeDispatchV2TilingData* tilingData)
{
    gmExpertOffsetCnt = workspaceGM_ + tilingData->moeDistributeDispatchV2Info.roceExpertOffset;
    tokenIdOffsetGM_ = workspaceGM_ + tilingData->moeDistributeDispatchV2Info.roceTokenMapOffset;

    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    tpipe_->InitBufPool(tbufPool_, BUFPOLL_MAX_SIZE);

    totalWinSizeEp_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
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
    copyInAxisH_ = axisH_;
    copyOutAxisH_ = axisH_;
    if constexpr (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value ||
                  Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value) {
        copyOutAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE); // 这里按照Int8进行搬运
    }
    if constexpr (Std::IsSame<XType, fp4x2_e2m1_t>::value || Std::IsSame<XType, fp4x2_e1m2_t>::value) {
        copyInAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE); // 这里按照Int8进行搬运
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitCommInfo(GM_ADDR mc2Context)
{
    ctx_.InitAndCheck(mc2Context, epWorldSize_, totalWinSizeEp_, tpipe_, expandXOutGM_);
    uint32_t epRankIdHccl = ctx_.GetEpRankId();
    statusDataSpaceGm_ = ctx_.GetStatusDataSpaceGm();
    selfRankWinInGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(statusDataSpaceGm_));
    selfDataStatusGMTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t*>(statusDataSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tbufPool_.InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, ctx_.GetEpWorldSize(), epRankIdOriginal_,
                              moeExpertNum_, epWorldSizeOriginal_, globalBS_, dataStateBuf);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    LocalTensor<uint64_t> opCntTensor = dataStateBuf.Get<uint64_t>();
    DataCopy(opCntTensor, selfDataStatusGMTensor_.template ReinterpretCast<uint64_t>(), UB_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    opSequence_ = opCntTensor.GetValue(OP_CNT_POSUL);
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    offsetGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenIdOffsetGM_));
    rankSizePerServer_ = ctx_.GetRankNumPerUbDomain();
    serverNum_ = epWorldSize_ / rankSizePerServer_;
    aivUsedCumSum_ = 2;
    aivUsedRemotComm_ = serverNum_; // 每个核负责一个核发送
    aivUsedScatter_ = (aivNum_ - aivUsedCumSum_ - aivUsedRemotComm_) / 2;
    aivUsedGather_ = aivNum_ - aivUsedScatter_ - aivUsedRemotComm_ - aivUsedCumSum_;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitComputeInfo()
{
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    combineBitmapTokenNum_ = Ceil(axisMaxBS_ * sizeof(uint16_t), UB_ALIGN) * UB_ALIGN / sizeof(uint16_t);
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
    if (isShareExpertRankFlag_) { // 当前卡是共享专家卡
        rscvStatusNum_ = epWorldSize_;
    } else { // 当前卡是moe专家卡
        rscvStatusNum_ = recvWinBlockNum_;
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitSetWindows(
    const MoeDistributeDispatchV2TilingData* tilingData)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<XOutType>(reinterpret_cast<__gm__ XOutType*>(GetWindAddrByRankId(tempepRankId)),
                                    totalWinSizeEp_);
        OOMCheckAddrRange<float>(reinterpret_cast<__gm__ float*>(GetWindStateAddrByRankId(tempepRankId)), STATE_SIZE);
    }
#endif
    winDataSizeOffsetEp_ =
        dataState_ * (totalWinSizeEp_ / 2) + tilingData->moeDistributeDispatchV2Info.roceDispatchOffset;
    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<XOutType> winDouble;
    winDouble.SetGlobalBuffer(reinterpret_cast<__gm__ XOutType*>(windowGM_));
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    OOMCheckAddrRange<XOutType>((__gm__ XOutType*)(winDouble.GetPhyAddr()), totalWinSizeEp_);
#endif
    windowInstatusFp32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(statusSpaceGm_));
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitExtraInfo()
{
    hOutSize_ = copyOutAxisH_ * sizeof(XOutType); // fp4输出时copyOutAxisH_=ceil(H/2)，按实际搬运元素数计算
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hScaleIdxSize_, scaleOutBytes_,
                         axisH_);

    hScaleIdxSize_ = Align32(hScaleIdxSize_); // 用于存放expertscale 16.06KB
    expertScaleAlign_ = hScaleIdxSize_ / sizeof(uint32_t);
    tokenAlign_ = hScaleIdxSize_ + UB_ALIGN;

    uint32_t serverDataLen = tokenQuantAlign_ * sizeof(int32_t);
    if constexpr ((QuantMode > UNQUANT) && IsSmoothScaleExist) {
        serverDataLen = hAlignSize_;
    }
    serverTripleOffset_ = serverDataLen / sizeof(int32_t);
    // 元数据基址必须32B对齐（DataCopy/DataCopyPad对UB地址有对齐要求），三元组+expertScale槽后整体Align32
    uint32_t serverExpertIdsByteOffset = Align32(serverDataLen + UB_ALIGN);
    serverExpertIdsOffset_ = serverExpertIdsByteOffset / sizeof(int32_t);

    uint32_t serverMoeCntByteOffset =
        serverExpertIdsByteOffset + Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // moeCntList偏移
    serverMoeCntOffset_ = serverMoeCntByteOffset / sizeof(int16_t); // 注意这里的offsetCnt是int16_t

    uint32_t serverExpertScalesByteOffset =
        serverMoeCntByteOffset + Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN; // moe expertscales
    serverExpertScalesOffset_ = serverExpertScalesByteOffset / sizeof(uint32_t);

    uint32_t moeNumPerServer = moeExpertNumPerRank_ * rankSizePerServer_;
    serverRecordLen_ = serverExpertScalesByteOffset + Ceil(axisK_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    serverRecordBlockCnt_ = Ceil(serverRecordLen_, SPLIT_BLOCK_DATA_SIZE);
    blockCntExpertPerServer_ = Ceil(moeNumPerServer * sizeof(int32_t), SPLIT_BLOCK_DATA_SIZE);
    expertCntSlotSize_ = static_cast<uint64_t>(blockCntExpertPerServer_) * SERVER_STATE_ALIGN;
    roceSendAreaSize_ = static_cast<uint64_t>(serverNum_) * (SERVER_STATE_ALIGN + expertCntSlotSize_);
    blockCntPerTokenToExpert_ = Ceil(tokenAlign_, SPLIT_BLOCK_DATA_SIZE); // 框内块数
    serverRecordAlignLen_ = serverRecordBlockCnt_ * SERVER_STATE_ALIGN;   // 17K
    roceReceiveSlotSize_ =
        SERVER_STATE_ALIGN + static_cast<uint64_t>(axisMaxBS_) * serverRecordAlignLen_ + expertCntSlotSize_;
    roceBufferSize_ = ctx_.GetRoceBufferSize();
    roceStateCapacity_ = roceBufferSize_ / ROCE_DATA_STATE_NUM;
    serverCommAreaSize_ = SERVER_TOKEN_AREA_OFFSET + TOKEN_RECV_CNT_MAX * serverRecordAlignLen_;
    hCommuSize_ = blockCntPerTokenToExpert_ * SERVER_STATE_ALIGN; // 拼接flag后的字节数

    uint32_t tokenAlignWinSize_ = Ceil(hCommuSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    expertPerSizeOnWin_ = axisMaxBS_ * tokenAlignWinSize_;
    maxBlockCnt_ = AscendC::Std::max(AscendC::Std::max(serverRecordBlockCnt_, blockCntExpertPerServer_),
                                     blockCntPerTokenToExpert_);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitMaskInfo()
{
    activeMaskBsCnt_ = axisBS_;
    expertIdsCnt_ = axisBS_ * axisK_;

    uint32_t hFp32Size = axisH_ * sizeof(float);
    if constexpr ((QuantMode == PERGROUP_DYNAMIC_QUANT) && IsSmoothScaleExist) {
        hFp32Size = Align128(axisH_) * sizeof(float);
    }
    uint32_t expertIdsSize = expertIdsCnt_ * sizeof(int32_t);
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    uint32_t expertCntAlign = Ceil(moeExpertNum_ * sizeof(int32_t), SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_DATA_SIZE;
    uint32_t bsKAlign256 = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t expertOffsetCntSize = axisBS_ * Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint32_t exprtflagCntSize = GetFlagCheckBufferSize(aivNum_ - aivUsedCumSum_);
    perServerMapLength_ = Ceil(axisMaxBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // max:2KB
    perServerMapCnt_ = perServerMapLength_ / sizeof(int32_t);                      // max:512
    moeCntRowAlign_ = Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN / sizeof(int16_t);
    expertIdsSize = Ceil(expertIdsSize, UB_ALIGN) * UB_ALIGN;
    maxSize_ = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSize_ = maxSize_ > xActivateMaskSize ? maxSize_ : xActivateMaskSize;
    maxSize_ = maxSize_ > bsKAlign256 ? maxSize_ : bsKAlign256;
    maxSize_ = maxSize_ > expertCntAlign ? maxSize_ : expertCntAlign;
    maxSize_ = maxSize_ > expertOffsetCntSize ? maxSize_ : expertOffsetCntSize;
    maxSize_ =
        maxSize_ > exprtflagCntSize ? maxSize_ : exprtflagCntSize; // 因为计算flag时复用了buffer因此需要保证32B对齐
    maxSize_ = Ceil(maxSize_, ALIGNED_LEN_256) * ALIGNED_LEN_256;

    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize);                            // BS * K * 4 = 32K
    tpipe_->InitBuffer(validLocalSrvTokenBuf_, perServerMapCnt_ * sizeof(bool)); // max: 0.5KB
    tbufPool_.InitBuffer(gatherMaskTBuf_, maxSize_);                             // max:32K
    tbufPool_.InitBuffer(dstExpBuf_, maxSize_);                                  // max:32K
    tbufPool_.InitBuffer(subExpBuf_, maxSize_);                                  // max:32K

    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    validLocalTokenTensor_ = validLocalSrvTokenBuf_.Get<bool>();
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    if (isExpertMaskFlag_) {
        tbufPool_.InitBuffer(expertMaskInputBuf_, expertIdsCnt_ * sizeof(bool));
        expertMaskInputTensor_ = expertMaskInputBuf_.Get<bool>();
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitDispatchBetweenServerInfo()
{
    uint32_t expertOffsetCntSize = axisBS_ * Ceil(axisK_ * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint8_t commRepeatTime = static_cast<uint8_t>(Ceil(maxBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256));
    uint32_t commFlagSize = commRepeatTime * SIZE_ALIGN_256;

    tbufPool_.InitBuffer(serverCountBuf_, serverNum_ * sizeof(uint32_t));
    tpipe_->InitBuffer(expertOffsetCntBuf_, expertOffsetCntSize); // 16K
    tpipe_->InitBuffer(flagBuf_, commFlagSize);

    serverCountTensor_ = serverCountBuf_.Get<uint32_t>();
    expertOffsetCntTensor_ = expertOffsetCntBuf_.Get<int16_t>();
    commFlagTensor_ = flagBuf_.Get<uint64_t>();

    Duplicate<uint64_t>(commFlagTensor_, opSequence_, commFlagSize / sizeof(uint64_t));
    Duplicate<int16_t>(expertOffsetCntTensor_, static_cast<int16_t>(0), expertIdsCnt_);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR sendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe* pipe,
    const MoeDistributeDispatchV2TilingData* tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    elasticInfoGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(elasticInfo));
    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ XInType*>(x));
    xGMTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    xActiveMaskGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(xActiveMask));
    expertScalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(expertScales));
    expandScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(expandScalesOut));
    expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
    dynamicScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dynamicScalesOut));
    expertTokenNumsOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(expertTokenNumsOut));
    expandIdxGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expandIdxOut));
    combineServerInfoGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(
        expandIdxOut +
        static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.a) * EXPAND_IDX_INFO * sizeof(int32_t)));
    workspaceGM_ = workspaceGM;
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut; // 无GlobalTensor
    sendCountGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceGM_));
    if constexpr (IsSmoothScaleExist) {
        scalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    }
    InitRecieveTilingContext(expandXOut, pipe, tilingData);
    InitCommInfo(mc2Context);
    InitComputeInfo();
    InitSetWindows(tilingData);
    InitExtraInfo();
    InitMaskInfo();
    InitDispatchBetweenServerInfo();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalcSplitRange(
    uint32_t totalCnt, uint32_t coreNum, uint32_t relativeId, uint32_t& startIdx, uint32_t& endIdx, uint32_t& perCnt)
{
    perCnt = totalCnt / coreNum;
    uint32_t remainderCnt = totalCnt % coreNum;
    if (relativeId < remainderCnt) {
        perCnt += 1;
        startIdx = relativeId * perCnt;
    } else {
        startIdx = perCnt * relativeId + remainderCnt;
    }
    endIdx = startIdx + perCnt;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t startAivId, uint32_t curUseAivNum, uint32_t& startTokenId, uint32_t& endTokenId,
    uint32_t& sendTokenNum)
{
    CalcSplitRange(curSendCnt, curUseAivNum, aivId_ - startAivId, startTokenId, endTokenId, sendTokenNum);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitMaskReSource()
{
    if (isExpertMaskFlag_) {
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskInputTensor_, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    }

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams); // copy expertid
    SyncFunc<AscendC::HardEvent::MTE2_V>(); // 后续全部都需要依赖此数据，在这里全部一起等待
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::TokenActiveMaskCal()
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
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ExpertActiveMaskCal()
{
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = expertMaskInputTensor_.ReinterpretCast<int8_t>();
    LocalTensor<half> maskStrideFp16Tensor = subExpBuf_.Get<half>();
    LocalTensor<int32_t> maskStrideInt32Tensor = gatherMaskTBuf_.Get<int32_t>();
    LocalTensor<uint8_t> sharedTensorInt8 = subExpBuf_.Get<uint8_t>();
    LocalTensor<int32_t> tempTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint8_t> tempTensorInt8 = validLocalTokenTensor_.ReinterpretCast<uint8_t>();

    Adds(expertIdsTensor_, expertIdsTensor_, moeExpertNum_, expertIdsCnt_);
    Cast(maskStrideFp16Tensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, expertIdsCnt_);
    Cast(maskStrideInt32Tensor, maskStrideFp16Tensor, RoundMode::CAST_RINT, expertIdsCnt_);
    Mul(expertIdsTensor_, expertIdsTensor_, maskStrideInt32Tensor, expertIdsCnt_);
    Adds(expertIdsTensor_, expertIdsTensor_, -moeExpertNum_, expertIdsCnt_);
    Abs(expertIdsTensor_, expertIdsTensor_, expertIdsCnt_);
    if (shareRankNumInServer_ <= 0) { // 本server 没有共享专家，需要重新计算
        uint32_t srcShape[2] = {activeMaskBsCnt_, axisK_};
        ReduceSum<int32_t, Pattern::Reduce::AR, false>(tempTensor, maskStrideInt32Tensor, sharedTensorInt8, srcShape,
                                                       false);
        Mins(tempTensor, tempTensor, static_cast<int32_t>(1), activeMaskBsCnt_);
        Cast(tempTensorInt8, tempTensor, RoundMode::CAST_NONE, activeMaskBsCnt_);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalTokenMask()
{
    activeMaskBsCnt_ = axisBS_;
    InitMaskReSource();
    if (isTokenMaskFlag_) { // 1维
        TokenActiveMaskCal();
    }

    Duplicate(validLocalTokenTensor_, true, perServerMapCnt_); // 共享专家受activeMaskBsCnt_影响

    if (isExpertMaskFlag_) { // 2维
        ExpertActiveMaskCal();
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::StoreTokenMask()
{
    LocalTensor<int32_t> tempTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int8_t> tempTensorInt8 = validLocalTokenTensor_.ReinterpretCast<int8_t>();
    Cast(tempTensor, tempTensorInt8, RoundMode::CAST_NONE, perServerMapCnt_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(offsetGMTensor_, tempTensor, perServerMapCnt_);
    PipeBarrier<PIPE_MTE3>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalServerSendCnt(
    uint32_t startServerId, uint32_t endServerId) // 如果有共享专家，应该所有的Token都需要发送
{
    TBuf<> sharedTmpBuf;
    uint32_t calCnt = isTokenMaskFlag_ ? activeMaskBsCnt_ * axisK_ : expertIdsCnt_;
    uint32_t sharedBufferXor = SIZE_ALIGN_256;
    uint32_t compareCount =
        Ceil(activeMaskBsCnt_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    uint32_t maskSize = Ceil(compareCount / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN; // 为了匹配CompareScalar接口1bit一共8位
    uint64_t sendTokenCnt = 0;
    // 每个目标server一格bool map，格宽按32B对齐（向量指令要求地址对齐），只分配本核负责的那几格
    uint32_t serverMapStride = Ceil(perServerMapCnt_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    uint32_t validSendTokenMaskLength = (endServerId - startServerId) * serverMapStride;
    tbufPool_.InitBuffer(sharedTmpBuf, sharedBufferXor);
    tbufPool_.InitBuffer(validRemoteSrvTokenBuf_, validSendTokenMaskLength);

    LocalTensor<float32_t> dstServerIdFp32 = dstExpBuf_.Get<float32_t>();
    LocalTensor<float32_t> tempTensorFp32 = subExpBuf_.Get<float32_t>();
    LocalTensor<half> tempTensorFp16 = subExpBuf_.Get<half>();
    LocalTensor<float32_t> gTempTensorFp32 = gatherMaskTBuf_.Get<float32_t>();
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
    validRemoteTokenTensor_ = validRemoteSrvTokenBuf_.Get<bool>();
    // expert维单独存一份serverId：循环内要做原地减法/Cast，和别的中间量共用一块UB会把原始值改掉（方案4.4第3条）
    TBuf<> serverIdBuf;
    tbufPool_.InitBuffer(serverIdBuf, Ceil(calCnt * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256);
    LocalTensor<int32_t> serverIdTensor = serverIdBuf.Get<int32_t>();

    Duplicate<bool>(validRemoteTokenTensor_, 0, validSendTokenMaskLength / sizeof(bool));
    Duplicate<uint32_t>(serverCountTensor_, uint32_t(activeMaskBsCnt_), serverNum_); // 假设server内有共享专家。
    Cast(dstServerIdFp32, expertIdsTensor_, RoundMode::CAST_NONE, calCnt);
    Duplicate<float32_t>(tempTensorFp32, static_cast<float32_t>(moeExpertNumPerRank_), calCnt);
    Div(dstServerIdFp32, dstServerIdFp32, tempTensorFp32, calCnt);
    Adds(dstServerIdFp32, dstServerIdFp32, static_cast<float32_t>(sharedExpertRankNum_), calCnt);
    Duplicate<float32_t>(tempTensorFp32, static_cast<float32_t>(rankSizePerServer_), calCnt);
    Div(dstServerIdFp32, dstServerIdFp32, tempTensorFp32, calCnt); // 计算每个专家所在的serverID
    Cast(serverIdTensor, dstServerIdFp32, RoundMode::CAST_TRUNC, calCnt);

    for (int32_t serverIndex = startServerId; serverIndex < endServerId; serverIndex++) {
        // 这里按“目标 server 上有共享专家卡”判断，适用于共享专家一卡一专家的场景。
        if (serverIndex * rankSizePerServer_ < sharedExpertRankNum_) { // 共享专家全部发送。
            // count保持预置的activeMaskBsCnt_，map也要给全1，否则popcount与count不一致，对端会一直等
            Duplicate<bool>(
                validRemoteTokenTensor_[(static_cast<uint32_t>(serverIndex) - startServerId) * serverMapStride], true,
                activeMaskBsCnt_);
            continue;
        }
        int32_t dstServerId = serverIndex;
        Adds(dstServerIdInt32, serverIdTensor, -dstServerId, calCnt);
        Abs(dstServerIdInt32, dstServerIdInt32, calCnt);
        Mins(dstServerIdInt32, dstServerIdInt32, 1, calCnt);
        Cast(dstServerIdInt16, dstServerIdInt32, RoundMode::CAST_NONE, calCnt);
        Duplicate<int16_t>(tempTensorInt16, 1, calCnt);
        Xor(compareInt16, dstServerIdInt16, tempTensorInt16, sharedTensorInt8, calCnt);
        Cast(dstServerIdFp32, compareInt16, RoundMode::CAST_NONE, calCnt); // 这里是 0 1 结果
        uint32_t srcShape[2] = {activeMaskBsCnt_, axisK_};
        ReduceSum<float32_t, Pattern::Reduce::AR, true>(tempTensorFp32, dstServerIdFp32, sharedTensorInt8, srcShape,
                                                        false);
        Mins(dstServerIdFp32, tempTensorFp32, float32_t(1.0), activeMaskBsCnt_); // 得到需要发送到这个server的TokenID
        Cast(dstServerIdInt32, dstServerIdFp32, RoundMode::CAST_RINT, activeMaskBsCnt_);
        CreateVecIndex(indexTensor, 0, activeMaskBsCnt_);
        // 使用 Cast 支持的 int32_t -> uint8_t 转换，0/1 字节供后续 bool 视图读取。
        Cast(validRemoteTokenTensor_[(static_cast<uint32_t>(serverIndex) - startServerId) * serverMapStride]
                 .ReinterpretCast<uint8_t>(),
             dstServerIdInt32, RoundMode::CAST_NONE, activeMaskBsCnt_);
        Duplicate<uint32_t>(expertMaskTensorU32, 0, maskSize / sizeof(uint32_t));
        CompareScalar(expertMaskTensorU8, dstServerIdInt32, 1, CMPMODE::EQ, compareCount);
        GatherMask(gatherIndex, indexTensor, expertMaskTensorU32, true, activeMaskBsCnt_, {1, 1, 0, 0},
                   sendTokenCnt); // 是否可以简化计算
        SyncFunc<AscendC::HardEvent::V_S>();
        serverCountTensor_.SetValue(serverIndex, static_cast<uint32_t>(sendTokenCnt));
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalExpertTotalSendCnt()
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
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt));
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::FillServerExpertCounts(
    uint32_t serverIndex, LocalTensor<int32_t>& counts)
{
    const uint32_t firstRank = serverIndex * rankSizePerServer_;
    const uint32_t sharedRanks =
        firstRank < sharedExpertRankNum_ ?
            AscendC::Std::min(sharedExpertRankNum_ - firstRank, static_cast<uint32_t>(rankSizePerServer_)) :
            0U;
    const uint32_t firstMoe =
        firstRank > sharedExpertRankNum_ ? (firstRank - sharedExpertRankNum_) * moeExpertNumPerRank_ : 0U;
    const uint32_t moeCount = (rankSizePerServer_ - sharedRanks) * moeExpertNumPerRank_;
    // 每个目标 server 都先清零，避免未选中的共享专家副本沿用上一轮计数。
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate<int32_t>(counts, 0, rankSizePerServer_ * moeExpertNumPerRank_);
    SyncFunc<AscendC::HardEvent::V_S>();
    if (sharedRanks != 0) {
        const uint32_t groupId = epRankId_ % rankNumPerSharedExpert_;
        for (uint32_t shareIndex = 0; shareIndex < sharedExpertNum_; ++shareIndex) {
            const uint32_t dstRank = shareIndex * rankNumPerSharedExpert_ + groupId;
            if (dstRank >= firstRank && dstRank < firstRank + sharedRanks) {
                counts.SetValue(dstRank - firstRank, static_cast<int32_t>(activeMaskBsCnt_));
            }
        }
    }
    for (uint32_t i = 0; i < moeCount; ++i) {
        counts.SetValue(sharedRanks + i, tokenNumToExpertTensor_.GetValue(firstMoe + i));
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SendExpertCnt()
{
    uint32_t startServerId, endServerId, sendServerNum;
    SplitToCore(serverNum_, 0, aivUsedCumSum_, startServerId, endServerId, sendServerNum);
    if (startServerId >= endServerId || sendServerNum == 0) {
        return;
    }
    LocalTensor<int32_t> counts = subExpBuf_.Get<int32_t>();
    TBuf<> seqFlagBuf;
    uint32_t seqFlagSize = Ceil(blockCntExpertPerServer_ * UB_ALIGN, SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tbufPool_.InitBuffer(seqFlagBuf, seqFlagSize);
    commFlagTensor_ = seqFlagBuf.Get<uint64_t>();
    Duplicate<uint64_t>(commFlagTensor_, opSequence_, seqFlagSize / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    TBuf<> hcommBuf;
    tbufPool_.InitBuffer(hcommBuf, ROCE_HCOMM_INIT_SIZE);
    hcomm_.Init(hcommBuf.Get<uint8_t>(), ROCE_HCOMM_INIT_SIZE);
    TBuf<> sendExpertCntTmpBuf;
    tbufPool_.InitBuffer(sendExpertCntTmpBuf, blockCntExpertPerServer_ * SERVER_STATE_ALIGN);
    LocalTensor<int32_t> sendExpertCntTmpTensor = sendExpertCntTmpBuf.Get<int32_t>();
    const uint32_t localServerId = static_cast<uint32_t>(serverId_);
    GM_ADDR localDispatchBase = GetRoceDispatchAddr(GetRoceBaseByServerId(localServerId));
    for (uint32_t serverIndex = startServerId; serverIndex < endServerId; ++serverIndex) {
        FillServerExpertCounts(serverIndex, counts);
        GM_ADDR serverDispatchBase = GetRoceDispatchAddr(GetRoceBaseByServerId(serverIndex));
        bool isLocalServer = serverIndex == localServerId;
        GM_ADDR rankGM = isLocalServer ? GetRoceExpertCntReceiveAddr(serverDispatchBase, localServerId) :
                                         GetRoceExpertCntSendAddr(localDispatchBase, serverIndex);
        SyncFunc<AscendC::HardEvent::S_V>();
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        WriteDataToWinOut<int32_t, true>(counts, sendExpertCntTmpTensor, rankGM, blockCntExpertPerServer_);
        if (!isLocalServer) {
            uint64_t commHandle = ctx_.GetRoceHandleByServerId(serverIndex, localServerId, ROCE_CHAN_EXPERT_CNT);
            PipeBarrier<PIPE_MTE3>();
            hcomm_.WriteNbi(commHandle, GetRoceExpertCntReceiveAddr(serverDispatchBase, localServerId), rankGM,
                            expertCntSlotSize_);
        }
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitStatusTensor()
{
    uint32_t rcvExpertDataSize = SPLIT_BLOCK_DATA_SIZE * blockCntExpertPerServer_;
    uint32_t flagMaxRecvNum = blockCntExpertPerServer_;
    uint32_t flagBufferSize = GetFlagCheckBufferSize(flagMaxRecvNum);
    uint32_t statusBufBufferSize = flagBufferSize > rcvExpertDataSize ? flagBufferSize : rcvExpertDataSize;
    TBuf flagRecvBuf;
    // tpipe被tbufPool_（192K）占掉大部分，Init阶段tpipe已接近上限；
    // 状态缓冲改从tbufPool_申请，避免额外挤占pool外常驻UB。
    tbufPool_.InitBuffer(flagRecvBuf, statusBufBufferSize);
    statusBufCntAlign_ = Ceil(expertNumInServer_ * serverNum_, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    tbufPool_.InitBuffer(statusBuf_, statusBufCntAlign_ * UB_ALIGN);
    flagRecvTensor_ = flagRecvBuf.Get<uint64_t>();
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    LocalTensor<uint64_t> statusU64Tensor = statusBuf_.Get<uint64_t>();
    // 每条32B记录都先由向量指令填入本轮序列号，随后只覆盖recvCnt字段。
    Duplicate<uint64_t>(statusU64Tensor, opSequence_, statusBufCntAlign_ * UB_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <RoceDispatchTypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GetFlagCheckBufferSize(uint32_t flagNum)
{
    uint32_t compareBufferSize = Ceil(flagNum * sizeof(uint64_t), FLAG_COMPARE_ALIGN) * FLAG_COMPARE_ALIGN;
    uint32_t gatherBufferSize = Ceil(flagNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t compareCount = compareBufferSize / sizeof(uint64_t);
    uint32_t compareMaskSize = Ceil(compareCount, BITS_PER_BYTE * UB_ALIGN) * UB_ALIGN;
    return compareBufferSize + gatherBufferSize + compareMaskSize;
}

template <RoceDispatchTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CheckFlagBySequence(
    LocalTensor<uint64_t> flagTensor, uint32_t flagNum)
{
    if (flagNum == 0) {
        return true;
    }

    uint32_t compareCount =
        Ceil(flagNum * sizeof(uint64_t), FLAG_COMPARE_ALIGN) * FLAG_COMPARE_ALIGN / sizeof(uint64_t);
    uint32_t compareBufferSize = compareCount * sizeof(uint64_t);
    uint32_t gatherBufferSize = Ceil(flagNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    LocalTensor<uint8_t> flagBuffer = flagTensor.ReinterpretCast<uint8_t>();
    LocalTensor<uint32_t> gatherOutTensor = flagBuffer[compareBufferSize].ReinterpretCast<uint32_t>();
    LocalTensor<uint8_t> compareMaskTensor = flagBuffer[compareBufferSize + gatherBufferSize];
    LocalTensor<uint32_t> compareMaskU32Tensor = compareMaskTensor.ReinterpretCast<uint32_t>();

    Compares(compareMaskTensor, flagTensor, opSequence_, AscendC::CMPMODE::EQ, compareCount);
    uint64_t arrivedNum = 0;
    GatherMask(gatherOutTensor, flagTensor.ReinterpretCast<uint32_t>(), compareMaskU32Tensor, true, flagNum,
               {1, 1, 0, 0}, arrivedNum);
    SyncFunc<AscendC::HardEvent::V_S>();
    return arrivedNum == flagNum;
}

template <RoceDispatchTypeClass>
template <bool isContinueWait>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitFlag(GM_ADDR flagAddr,
                                                                                   LocalTensor<uint64_t> flagInLTTensor,
                                                                                   uint32_t srcStride, uint32_t waitNum)
{
    uint32_t copyNum = 1; // srcStride为0时flag连续存放，可一次拷入
    uint32_t copyBlockLength = waitNum * sizeof(uint64_t);

    if (srcStride != 0) {
        copyNum = waitNum;
        copyBlockLength = sizeof(uint64_t);
    }

    GlobalTensor<uint64_t> flagInGMTensor;
    flagInGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(flagAddr));
    DataCopyExtParams flagCopyParams{static_cast<uint16_t>(copyNum), copyBlockLength, srcStride, 0, 0};
    DataCopyPadExtParams<uint64_t> flagPadParams{false, 0U, 0U, 0U};
    bool isArrived = false;
    do {
        DataCopyPad<uint64_t, PaddingMode::Compact>(flagInLTTensor, flagInGMTensor, flagCopyParams, flagPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        isArrived = CheckFlagBySequence(flagInLTTensor, waitNum);
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        if constexpr (isContinueWait) {
            if (isArrived) { // 循环等待直到全部到达
                break;
            }
        }

    } while (isContinueWait);

    return isArrived;
}

template <RoceDispatchTypeClass>
template <typename T, bool disableL2>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ReadDataFromWindows(GM_ADDR dataAddr,
                                                                                              uint32_t blockCnt,
                                                                                              LocalTensor<T>& xInTensor)
{
    GlobalTensor<T> dataFlagGlobal;
    dataFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dataAddr));
    if constexpr (disableL2) {
        dataFlagGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCnt), static_cast<uint32_t>(SPLIT_BLOCK_DATA_SIZE),
                                      static_cast<uint32_t>(SERVER_STATE_ALIGN - SPLIT_BLOCK_DATA_SIZE), 0};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyPad(xInTensor, dataFlagGlobal, srcTokenCopyParams, srcTokenPadParams);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SetStatus() // 框间状态区接收 + 框内状态区发送
{
    uint32_t startServerId;
    uint32_t endServerId;
    uint32_t resvServerNum;
    SplitToCore(serverNum_, 0, aivUsedCumSum_, startServerId, endServerId, resvServerNum);
    if (startServerId >= endServerId || resvServerNum == 0) {
        return;
    }

    InitStatusTensor();

    // 接收槽落在本卡RoCE注册区，按source server索引；对端写入用的也是本卡注册区地址
    GM_ADDR localDispatchBase = GetRoceDispatchAddr(GetRoceBaseByServerId(static_cast<uint32_t>(serverId_)));
    for (int serverIdx = startServerId; serverIdx < endServerId; serverIdx++) {
        GM_ADDR serverRcvAddr = GetRoceExpertCntReceiveAddr(localDispatchBase, static_cast<uint32_t>(serverIdx));
        GM_ADDR flagAddr = serverRcvAddr + SPLIT_BLOCK_DATA_SIZE;
        WaitFlag<true>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, blockCntExpertPerServer_);
        LocalTensor<uint32_t> rcvExpertCntTensor = flagRecvTensor_.template ReinterpretCast<uint32_t>(); // buf复用
        ReadDataFromWindows<uint32_t, true>(serverRcvAddr, blockCntExpertPerServer_, rcvExpertCntTensor);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t expertIndex = 0; expertIndex < expertNumInServer_; expertIndex++) {
            statusTensor_(expertIndex * 8 + 2) = rcvExpertCntTensor(expertIndex);
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
                reinterpret_cast<GM_ADDR>(GetWindStateAddrByRankId(startRankId_ + dstRankId) + offset); // 计算地址偏移
            rankGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(rankGM));
            DataCopy(rankGMTensor, statusTensor_[expertIndex * 8], 8UL);
        }
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitCalCumSumBuffer()
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    uint32_t statusDataSize = Ceil((recStatusNumPerCore_ * UB_ALIGN), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    uint32_t flagCheckSize = GetFlagCheckBufferSize(recStatusNumPerCore_);
    uint32_t waitStatusBufSize = statusDataSize > flagCheckSize ? statusDataSize : flagCheckSize;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * moeExpertNumPerRank_ * sizeof(float);
    uint64_t gatherMaskOutSize =
        (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    uint64_t sumContinueAlignSize = Ceil((aivNum_ * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize); // recStatusNumPerCore_32对齐后大小  * 32B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);      // 48 * 32B
    uint32_t sumLocalSize = aivNum_ * UB_ALIGN;
    uint32_t sumFlagCheckSize = GetFlagCheckBufferSize(aivUsedCumSum_);
    sumLocalSize = sumLocalSize > sumFlagCheckSize ? sumLocalSize : sumFlagCheckSize;
    tpipe_->InitBuffer(sumLocalBuf_, sumLocalSize);
    tpipe_->InitBuffer(sumContinueBuf_, sumContinueAlignSize); // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);              // 96 B

    uint32_t statusBufSize = rscvStatusNum_ * UB_ALIGN;
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t workLocalBufSize = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GatherSumRecvCnt(
    LocalTensor<float>& gatherMaskOutTensor, LocalTensor<uint32_t>& gatherTmpTensor,
    LocalTensor<float>& statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 4); // 源操作数每个datablock取下标为2的元素
    uint32_t mask = 4;              // 前8B为uint64_t flag，recvCnt位于第3个float槽位
    SyncFunc<AscendC::HardEvent::S_V>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();

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

    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    LocalTensor<uint64_t> sumCoreU64Tensor = sumCoreBuf_.Get<uint64_t>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8); // 8 = 256 / 32
    // 整块先写本轮uint64_t序列号，再覆盖每条32B记录的recvCnt槽位。
    Duplicate<uint64_t>(sumCoreU64Tensor, opSequence_, aivUsedCumSum_ * UB_ALIGN / sizeof(uint64_t));
    // 每次处理256字节，8个datablock，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8); // 重复写aivUsedCumSum_份
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)],
             sumCoreFP32Tensor, sumIntriParams);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitDispatch()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    LocalTensor<uint64_t> statusU64Tensor = waitStatusBuf_.Get<uint64_t>();
    gatherTmpTensor.SetValue(0, 1);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR flagAddr = statusSpaceGm_ + startStatusIndex_ * STATE_OFFSET;
    WaitFlag<true>(flagAddr, statusU64Tensor, STATE_OFFSET - sizeof(uint64_t), recStatusNumPerCore_);
    // flag全部到齐后再搬一次完整状态记录，供后续收集recvCnt。
    DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
             intriParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GetCumSum(LocalTensor<int32_t>& outLocal,
                                                                                    uint32_t newAivId)
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
    LocalTensor<uint64_t> sumLocalU64Tensor = sumLocalBuf_.Get<uint64_t>();
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

    // 获取状态区中每个核的recvCnt
    GM_ADDR flagAddr = statusDataSpaceGm_ + CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN + sizeof(uint64_t);
    WaitFlag<true>(flagAddr, sumLocalU64Tensor, aivUsedCumSum_ * UB_ALIGN - sizeof(uint64_t), aivUsedCumSum_);
    DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)],
             sumIntriParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

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
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalRecvAndSetFlag()
{
    LocalTensor<int32_t> outCountLocal;
    GetCumSum(outCountLocal, aivId_);
    // 计算epRecvCnt
    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = statusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + 2);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    GM_ADDR wAddr = reinterpret_cast<GM_ADDR>(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(wAddr));
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);
    // 复制aivNum_份
    for (uint32_t index = 0; index < aivNum_; index++) {
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
    }
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivNum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<uint64_t> syncOnCoreTensor = sumCoreBuf_.Get<uint64_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();
    Duplicate<uint64_t>(syncOnCoreTensor, opSequence_, aivNum_ * UB_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_FLAG_OFFSET + aivId_ * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor,
             sumIntriParams); // 软同步
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SetExpertTokenNums()
{
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
        uint64_t mask[2] = {4, 0}; // 前8B为uint64_t flag，recvCnt位于第3个float槽位
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

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalCumSum()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    // 本阶段不需要server count：共享专家槽的计数恒为本轮有效token数，见SendExpertCnt
    CalExpertTotalSendCnt();
    SendExpertCnt();
    SetStatus();

    SplitToCore(rscvStatusNum_, 0, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_);

    InitCalCumSumBuffer();
    WaitDispatch();
    CalRecvAndSetFlag();
    // 使用newAivId为0的核进行计算
    if (aivId_ == 0) {
        SetExpertTokenNums();
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WriteExpertOffset()
{
    uint32_t newAivId = aivId_ - aivUsedCumSum_;
    uint32_t offsetPerMask = Ceil(expertIdsCnt_ * sizeof(uint16_t), UB_ALIGN) * UB_ALIGN;
    uint64_t dstOffset = newAivId * offsetPerMask; // gmExpertOffsetCnt
    uint64_t dstFlag = (aivNum_ - aivUsedCumSum_) * offsetPerMask + sizeof(uint64_t) * newAivId;
    GlobalTensor<int16_t> expertOffsetCntGMTensor;
    GlobalTensor<uint64_t> expertOffsetFlagGMTensor;
    LocalTensor<uint64_t> writeCntTensor = subExpBuf_.Get<uint64_t>();
    expertOffsetCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(gmExpertOffsetCnt + dstOffset));
    expertOffsetFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(gmExpertOffsetCnt + dstFlag));
    Duplicate<uint64_t>(writeCntTensor, opSequence_, UB_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyExtParams expertOffsetCopyOutParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int16_t)), 0U, 0U, 0U};
    DataCopyPad(expertOffsetCntGMTensor, expertOffsetCntTensor_, expertOffsetCopyOutParams);
    PipeBarrier<PIPE_MTE3>();
    DataCopyExtParams expertCntCopyOutParams{1U, static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    DataCopyPad(expertOffsetFlagGMTensor, writeCntTensor, expertCntCopyOutParams);
    // CalServerSendCnt overwrites subExpBuf_ through V after MTE3 consumes the sequence.
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalTokenSendExpertOffset()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    SplitToCore(moeExpertNum_, aivUsedCumSum_, aivNum_ - aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum);
    if (startExpertId >= endExpertId || sendExpertNum == 0) {
        return;
    }
    TBuf<> indexBuf;
    uint32_t activeBsCnt = activeMaskBsCnt_;
    uint32_t calCnt =
        isTokenMaskFlag_ ? activeBsCnt * axisK_ : expertIdsCnt_; // 二维mask时，为了对齐需要全部读取expertID
    uint32_t expertNumAlign256 = Ceil(calCnt * sizeof(uint16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(uint16_t);
    uint64_t sendNumperExpert = 0;

    tbufPool_.InitBuffer(indexBuf, calCnt * sizeof(uint16_t));

    LocalTensor<int16_t> expertsIndexTensor = indexBuf.Get<int16_t>();
    LocalTensor<int32_t> dstExpIdTensorInt32 = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint16_t> gatherMaskTensor = gatherMaskTBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int16_t> validExpertIndexTensor = subExpBuf_.Get<int16_t>();
    LocalTensor<int32_t> temp = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> dstOffset = dstExpBuf_.Get<uint32_t>();

    CreateVecIndex(expertsIndexTensor, static_cast<int16_t>(0), calCnt);
    for (int32_t Index = startExpertId; Index < endExpertId; Index++) {
        int32_t dstExpertId = Index;
        Adds(dstExpIdTensorInt32, expertIdsTensor_, -dstExpertId, calCnt);
        Duplicate<uint16_t>(gatherMaskTensor, 0,
                            Ceil(calCnt, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE /
                                sizeof(uint16_t)); // 获取索引需要放在这里，因为需要256对齐
        Compares(gatherMaskTensorInt8, dstExpIdTensorInt32, static_cast<int32_t>(0), AscendC::CMPMODE::EQ,
                 expertNumAlign256);
        GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0},
                   sendNumperExpert); // 当前需要发送的专家的索引以及需要发送的有效专家总数
        SyncFunc<AscendC::HardEvent::V_S>();
        if (sendNumperExpert == 0) {
            continue;
        }
        Cast(dstOffset, validExpertIndexTensor, RoundMode::CAST_NONE, calCnt);
        Muls(temp, temp, static_cast<int32_t>(sizeof(int16_t)), static_cast<uint32_t>(sendNumperExpert));
        Scatter(expertOffsetCntTensor_, expertsIndexTensor, dstOffset, 0, static_cast<uint32_t>(sendNumperExpert));
    }
    WriteExpertOffset(); // 计算完成将结果写入GM并写flag
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::FillQuadruple(
    LocalTensor<XCommInType>& xOutTensor, uint32_t startTokenIndex, uint32_t tokenIndex)
{
    uint32_t realIndex = tokenIndex - startTokenIndex;
    xOutQueue_.EnQue(xOutTensor);
    xOutTensor = xOutQueue_.DeQue<XCommInType>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    LocalTensor<int16_t> xOutTint16 = xOutTensor.template ReinterpretCast<int16_t>();
    LocalTensor<float> xoutTfloat32 = xOutTensor.template ReinterpretCast<float>();
    // 复用输出槽前，等待上一轮 V 管线搬运读完，避免 Scalar 提前覆盖 tokenId。
    SyncFunc<AscendC::HardEvent::V_S>();
    xOutTint32(serverTripleOffset_) = epRankId_;      // 源rankId
    xOutTint32(serverTripleOffset_ + 1) = tokenIndex; // TokenID ,

    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(axisK_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> CopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xOutTint32[serverExpertIdsOffset_], expertIdsGMTensor_[tokenIndex * axisK_], copyParams, CopyPadParams);
    DataCopy(xOutTint16[serverMoeCntOffset_], expertOffsetCntTensor_[realIndex * moeCntRowAlign_], moeCntRowAlign_);
    DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0U};
    DataCopyPad(xoutTfloat32[serverExpertScalesOffset_], expertScalesGMTensor_[tokenIndex * axisK_], copyParams,
                expertScalePadParams);

    if (isExpertMaskFlag_) {
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t index = 0; index < axisK_; index++) {
            uint32_t expertPos = serverExpertIdsOffset_ + index;
            uint32_t offset = tokenIndex * axisK_ + index;
            xOutTint32(expertPos) = expertIdsTensor_(offset);
        }
    }
    // 普通分支的 srcRank/tokenIndex 也必须先对后续 Vector 搬运可见。
    SyncFunc<AscendC::HardEvent::S_V>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalExpertOffset(uint32_t startTokenId,
                                                                                          uint32_t endTokenId)
{
    uint32_t offsetPerMask = Ceil(expertIdsCnt_ * sizeof(uint16_t), UB_ALIGN) * UB_ALIGN;
    uint64_t dstFlag = (aivNum_ - aivUsedCumSum_) * offsetPerMask;
    uint64_t usedCoreNum = (aivNum_ - aivUsedCumSum_) > moeExpertNum_ ? moeExpertNum_ : (aivNum_ - aivUsedCumSum_);
    uint64_t dstOffset = startTokenId * axisK_ * sizeof(int16_t);
    uint32_t computeRow = endTokenId - startTokenId;
    uint32_t expertOffsetNum = computeRow * moeCntRowAlign_; // 32B对齐
    GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(gmExpertOffsetCnt + dstFlag);
    LocalTensor<uint64_t> flagInLTTensor = dstExpBuf_.Get<uint64_t>();
    LocalTensor<int16_t> expertCntTensor = dstExpBuf_.Get<int16_t>();
    GlobalTensor<int16_t> expertOffsetCntGMTensor;
    DataCopyParams expertCntCopyInParams{static_cast<uint16_t>(computeRow),
                                         static_cast<uint16_t>(sizeof(int16_t) * axisK_), 0, 0};
    DataCopyPadParams expertCntPadParams{false, 0, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate<int16_t>(expertOffsetCntTensor_, static_cast<int16_t>(0), expertOffsetNum);
    WaitFlag<true>(flagAddr, flagInLTTensor, EXPERT_CNT_FLAG_SPACE, usedCoreNum);
    for (uint32_t index = 0; index < usedCoreNum; index++) {
        uint64_t expertOffset = dstOffset + index * offsetPerMask;
        expertOffsetCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(gmExpertOffsetCnt + expertOffset));
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(expertCntTensor, expertOffsetCntGMTensor, expertCntCopyInParams, expertCntPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Add(expertOffsetCntTensor_, expertOffsetCntTensor_, expertCntTensor, expertOffsetNum);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SendServerCnt(uint32_t startServerNum,
                                                                                        uint32_t endServerNum)
{
    GlobalTensor<uint32_t> dstStateGMTensor;
    TBuf<> tempbuf;
    tbufPool_.InitBuffer(tempbuf, SERVER_STATE_ALIGN);
    LocalTensor<uint32_t> outTensor = tempbuf.Get<uint32_t>();
    LocalTensor<uint64_t> outTensorU64 = tempbuf.Get<uint64_t>();
    // 整个状态块向量填充本轮序列号，循环内只覆盖tokenCnt，flag位于最后32B首个uint64_t。
    Duplicate<uint64_t>(outTensorU64, opSequence_, SERVER_STATE_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::V_S>();
    // 每个目标server只发一个512B状态块，普通接口即可，不需要batch handle
    TBuf<> hcommBuf;
    tbufPool_.InitBuffer(hcommBuf, ROCE_HCOMM_INIT_SIZE);
    hcomm_.Init(hcommBuf.Get<uint8_t>(), ROCE_HCOMM_INIT_SIZE);
    const uint32_t localServerId = static_cast<uint32_t>(serverId_);
    GM_ADDR localDispatchBase = GetRoceDispatchAddr(GetRoceBaseByServerId(localServerId));
    for (uint32_t index = startServerNum; index < endServerNum; index++) {
        if (index == serverId_) {
            // 本server的map直接复用按目标server生成的那一份（依赖与CalServerSendCnt用同一次SplitToCore分片）
            validLocalTokenTensor_ =
                validRemoteTokenTensor_[(index - startServerNum) *
                                        (Ceil(perServerMapCnt_ * sizeof(bool), UB_ALIGN) * UB_ALIGN)];
            StoreTokenMask();
        }
        // 本server直接写本卡接收槽的头块；远端先写本卡发送区，再用一次普通WriteNbi写到对端接收槽头块
        GM_ADDR rankGM = (index == localServerId) ? GetRoceReceiveAddr(localDispatchBase, localServerId) :
                                                    GetRoceServerCntSendAddr(localDispatchBase, index);
        dstStateGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(rankGM));
        outTensor(0) = serverCountTensor_(index);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(dstStateGMTensor, outTensor, SERVER_STATE_ALIGN / sizeof(uint32_t));
        if (index != localServerId) {
            uint64_t commHandle = ctx_.GetRoceHandleByServerId(index, localServerId, ROCE_CHAN_TOKEN);
            GM_ADDR remoteHeader = GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(index)), localServerId);
            PipeBarrier<PIPE_MTE3>(); // 状态块落到本卡发送区后再提交RoCE写任务
            // 完成判据是对端头块里的opSequence_ flag，不等RDMA完成，因此不调Drain
            hcomm_.WriteNbi(commHandle, remoteHeader, rankGM, SERVER_STATE_ALIGN);
        }
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SendServerTokens(
    uint32_t serverIndex, uint32_t mapOffset, GM_ADDR localTokenBase, LocalTensor<uint8_t> batchUb,
    LocalTensor<uint64_t> flagTensor)
{
    const uint32_t localServerId = static_cast<uint32_t>(serverId_);
    GM_ADDR remoteTokenBase =
        GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(serverIndex)), localServerId) +
        SERVER_TOKEN_AREA_OFFSET;
    BatchHandle<ChannelHandle> batchHandle =
        hcomm_.MakeBatchHandle(ctx_.GetRoceHandleByServerId(serverIndex, localServerId, ROCE_CHAN_TOKEN), batchUb,
                               ROCE_BATCH_UB_BYTES, remoteTokenBase, localTokenBase);
    LocalTensor<bool> serverMap = validRemoteTokenTensor_[mapOffset];
    uint32_t pendingCount = 0;
    uint32_t compactIndex = 0;
    for (uint32_t tokenIndex = 0; tokenIndex < activeMaskBsCnt_; ++tokenIndex) {
        if (!serverMap.GetValue(tokenIndex)) {
            continue;
        }
        GM_ADDR tokenSrc = localTokenBase + static_cast<uint64_t>(tokenIndex) * serverRecordAlignLen_;
        WaitFlag<true>(tokenSrc + SPLIT_BLOCK_DATA_SIZE, flagTensor, COMM_FLAG_COPY_STRIDE, serverRecordBlockCnt_);
        // 一个完整 token 对应一个单 SGE WQE，多个 WQE 通过 batch 一起提交。
        hcomm_.WriteNbi(batchHandle, remoteTokenBase + static_cast<uint64_t>(compactIndex) * serverRecordAlignLen_,
                        tokenSrc, serverRecordAlignLen_);
        ++compactIndex;
        if (++pendingCount == ROCE_TOKEN_BATCH_SIZE) {
            hcomm_.BatchCommit(batchHandle);
            pendingCount = 0;
        }
    }
    if (pendingCount != 0) {
        hcomm_.BatchCommit(batchHandle);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::HcclCommSend(uint32_t startServerNum,
                                                                                       uint32_t endServerNum)
{
    TBuf<> batchBuf;
    // TBuf 基址不保证 64B 对齐，预留偏移空间后再选取 WQE staging 区。
    tbufPool_.InitBuffer(batchBuf, ROCE_BATCH_UB_BYTES + ROCE_WQEBB_BYTES);
    LocalTensor<uint8_t> batchStorage = batchBuf.Get<uint8_t>();
    uint32_t padding = (ROCE_WQEBB_BYTES - reinterpret_cast<uintptr_t>(batchStorage.GetPhyAddr()) % ROCE_WQEBB_BYTES) %
                       ROCE_WQEBB_BYTES;
    LocalTensor<uint8_t> batchUb = batchStorage[padding];
    TBuf<> sendFlagBuf;
    tbufPool_.InitBuffer(sendFlagBuf, GetFlagCheckBufferSize(serverRecordBlockCnt_));
    LocalTensor<uint64_t> sendFlagTensor = sendFlagBuf.Get<uint64_t>();
    const uint32_t localServerId = static_cast<uint32_t>(serverId_);
    GM_ADDR localTokenBase =
        GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(localServerId)), localServerId) +
        SERVER_TOKEN_AREA_OFFSET;
    uint32_t serverMapStride = Ceil(perServerMapCnt_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    for (uint32_t serverIndex = startServerNum; serverIndex < endServerNum; ++serverIndex) {
        if (serverIndex != localServerId && serverCountTensor_.GetValue(serverIndex) != 0) {
            SendServerTokens(serverIndex, (serverIndex - startServerNum) * serverMapStride, localTokenBase, batchUb,
                             sendFlagTensor);
        }
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::HcclTokenDispatch()
{
    uint32_t startServerId, endServerId, sendServerNum;
    SplitToCore(serverNum_, aivUsedCumSum_, aivUsedRemotComm_, startServerId, endServerId, sendServerNum);
    if (startServerId >= serverNum_ || sendServerNum == 0) {
        return;
    }
    CalServerSendCnt(startServerId, endServerId);
    SendServerCnt(startServerId, endServerId); // 需要先发送Token总数，每个核负责一个server
    HcclCommSend(startServerId, endServerId);  // 专门负责发送Token
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalServerSplitCore(
    uint32_t serverNum, uint32_t startAivId, uint32_t useAivNum, uint32_t& startServerId, uint32_t& endServerId,
    uint32_t& useAivPerServer, uint32_t& firstAivId)
{
    if (useAivNum == 0) { // 无核参与本阶段
        useAivPerServer = 0;
        startServerId = 0;
        endServerId = 0;
        firstAivId = startAivId;
        return;
    }
    uint32_t relativeAivId = aivId_ - startAivId; // 相对Id
    useAivPerServer = useAivNum / serverNum;      // 每个server分几个核处理
    if (useAivPerServer > 0) { // 核数>=server数：一个server由多个核协作，当前核只负责其中1个server
        uint32_t remainderAivNum = useAivNum % serverNum;
        if (relativeAivId < remainderAivNum * (useAivPerServer + 1)) { // 落在前remainder个多核桶
            startServerId = relativeAivId / (useAivPerServer + 1);
            useAivPerServer += 1; // 前面的server多分一个核
            firstAivId = startAivId + startServerId * useAivPerServer;
        } else {
            startServerId = (relativeAivId - remainderAivNum) / useAivPerServer;
            firstAivId = startAivId + startServerId * useAivPerServer + remainderAivNum;
        }
        endServerId = startServerId + 1;
    } else { // 核数<server数：一个核负责多个server，按SplitToCore同构对server分片
        CalcSplitRange(serverNum, useAivNum, relativeAivId, startServerId, endServerId, useAivPerServer);
        useAivPerServer = 1; // 组内仅当前核，整段token都由本核处理
        firstAivId = aivId_;
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitScatterBuffer()
{
    // CalExpertOffset has waited for MTE3/MTE2; its final Add must finish before MTE2 reuses the pool.
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    tbufPool_.Reset();
    InitQuantBuffer<CommMode::INTRA_NODE>(); // 初始化量化参数
    // server窗一条记录按serverRecordBlockCnt_整记录搬运（含expertIds/moeCnt/expertScales元数据），
    // 源缓冲和flag拼接缓冲都要按serverRecordAlignLen_申请；UB不足时退化为单缓冲
    uint32_t bufferNum = DOUBLE_BUFFER;
    uint32_t queueNeedSize = hAlignSize_ + 2 * serverRecordAlignLen_;
    if (bufferNum * queueNeedSize + 2 * maxSize_ > BUFPOLL_MAX_SIZE) {
        bufferNum = 1;
    }
    tbufPool_.InitBuffer(xInQueue_, bufferNum, hAlignSize_);
    tbufPool_.InitBuffer(xOutQueue_, bufferNum, serverRecordAlignLen_);
    tbufPool_.InitBuffer(xTempQueue_, bufferNum, serverRecordAlignLen_);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ReadTokenData(uint32_t tokenIndex,
                                                                                        LocalTensor<XInType>& tokenData)
{
    tokenData = xInQueue_.AllocTensor<XInType>();
    if constexpr (QuantMode > UNQUANT &&
                  ((QuantMode == MX_QUANT) || (QuantMode == MX_QUANT_CLIP) || (QuantMode == PERGROUP_DYNAMIC_QUANT))) {
        QuantCleanTensor(tokenData);
    }
    if constexpr (Std::IsSame<XInType, fp8_e5m2_t>::value || Std::IsSame<XInType, fp8_e4m3fn_t>::value ||
                  Std::IsSame<XInType, hifloat8_t>::value) {
        DataCopyPadParams copyPadExtParams{true, 0U, 0U, 0U};
        DataCopyParams xCopyParams{1U, static_cast<uint16_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U};
        DataCopyPad(tokenData, xGMTensor_[tokenIndex * copyInAxisH_], xCopyParams, copyPadExtParams);
    } else {
        DataCopyPadExtParams<XInType> copyPadExtParams{true, 0U, 0U, 0U};
        DataCopyExtParams xCopyParams{1U, static_cast<uint32_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U, 0U};
        DataCopyPad(tokenData, xGMTensor_[tokenIndex * copyInAxisH_], xCopyParams, copyPadExtParams);
    }
    if constexpr (QuantMode == UNQUANT && IsSmoothScaleExist) {
        GlobalTensor<uint8_t> tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
        DataCopyPadExtParams<uint8_t> scaleCopyPadExtParams{true, 0U, 0U, 0U};
        DataCopyExtParams scaleInParams = {1U, static_cast<uint32_t>(scaleInBytes_), 0U, 0U, 0U};
        DataCopyPad(tokenData[Align32(copyInAxisH_)].template ReinterpretCast<uint8_t>(),
                    tmp[tokenIndex * scaleInBytes_], scaleInParams, scaleCopyPadExtParams);
    }
    xInQueue_.EnQue(tokenData);
    tokenData = xInQueue_.DeQue<XInType>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::QuantToken(
    uint32_t tokenIndex, LocalTensor<XInType>& xInTensor, LocalTensor<XCommInType>& xOutTensor, uint32_t expertIndex)
{
    xOutTensor = xOutQueue_.AllocTensor<XCommInType>();
    if constexpr ((QuantMode > UNQUANT && !IsSmoothScaleExist) ||
                  (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
        if constexpr (QuantMode > UNQUANT) {
            quantInst_.QuantProcess(xOutTensor, xInTensor, expertIndex, scalesCount_, scalesGMTensor_);
        } else { // 非量化直转hifp8
            Cast(floatLocalTemp_, xInTensor, RoundMode::CAST_NONE, copyInAxisH_);
            Cast(xOutTensor, floatLocalTemp_, RoundMode::CAST_ROUND, copyInAxisH_);
        }
    } else {
        // Preserve the zero padding read by receiver-side group/MX quantization.
        const uint32_t dataByteEnd =
            QuantMode > UNQUANT && IsSmoothScaleExist ? hAlignSize_ : Align32(copyInAxisH_ * sizeof(XInType));
        DataCopy(xOutTensor, xInTensor, dataByteEnd / sizeof(XCommInType));
        if constexpr (QuantMode == UNQUANT && IsSmoothScaleExist) {
            DataCopyPadExtParams<uint8_t> copyPadParams{true, 0U, 0U, 0U};
            DataCopyExtParams scaleInParams{1U, static_cast<uint32_t>(scaleInBytes_), 0U, 0U, 0U};
            DataCopyPad(xOutTensor[dataByteEnd / sizeof(XInType)].template ReinterpretCast<uint8_t>(),
                        scalesGMTensor_.template ReinterpretCast<uint8_t>()[tokenIndex * scaleInBytes_], scaleInParams,
                        copyPadParams);
        }
    }
    xInQueue_.FreeTensor(xInTensor);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WriteNotifyFlag(uint32_t tokenIndex)
{
    (void)tokenIndex;
    PipeBarrier<PIPE_MTE3>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WriteTokenWithNotify(
    uint32_t startTokenIndex, uint32_t tokenIndex, LocalTensor<XCommInType>& xOutTensor)
{
    FillQuadruple(xOutTensor, startTokenIndex, tokenIndex);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    // token记录写到本卡RoCE接收槽的公共源区（按原始tokenIndex编址），远端发送核从这里取数
    GM_ADDR rankGM = GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(static_cast<uint32_t>(serverId_))),
                                        static_cast<uint32_t>(serverId_)) +
                     SERVER_TOKEN_AREA_OFFSET + static_cast<uint64_t>(tokenIndex) * serverRecordAlignLen_;
    LocalTensor<XCommInType> tempTensor = xTempQueue_.AllocTensor<XCommInType>();
    WriteDataToWinOut<XCommInType, true>(
        xOutTensor, tempTensor, rankGM, serverRecordBlockCnt_); // server窗整记录搬运（含expertIds/moeCnt/expertScales）
    WriteNotifyFlag(tokenIndex);
    xTempQueue_.FreeTensor(tempTensor);
    xOutQueue_.FreeTensor(xOutTensor);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ScatterTokenToRemote()
{
    if (aivId_ < aivUsedCumSum_ + aivUsedRemotComm_) {
        HcclTokenDispatch();
        return;
    }

    uint32_t totalSendCnt = activeMaskBsCnt_;
    uint32_t firstAivId = aivUsedCumSum_ + aivUsedRemotComm_; // 数据发送核从一级通信核之后开始
    uint32_t usedAivNum = aivUsedScatter_;                    // 数据发送核数
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(totalSendCnt, firstAivId, usedAivNum, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= endTokenId || sendTokenNum == 0) {
        return;
    }

    CalExpertOffset(startTokenId, endTokenId);
    InitScatterBuffer();
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; tokenIndex++) {
        if (sharedExpertNum_ <= 0 && isExpertMaskFlag_ && !validLocalTokenTensor_.GetValue(tokenIndex)) {
            continue;
        }
        LocalTensor<XInType> xInTensor;
        LocalTensor<XCommInType> xOutTensor;
        ReadTokenData(tokenIndex, xInTensor);
        QuantToken(tokenIndex, xInTensor, xOutTensor);
        WriteTokenWithNotify(startTokenId, tokenIndex, xOutTensor);
    }
}

template <RoceDispatchTypeClass>
template <CommMode mode>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitQuantBuffer()
{
    if constexpr ((QuantMode > UNQUANT) || ((QuantMode == UNQUANT) && !Std::IsSame<XInType, XOutType>::value)) {
        if constexpr (mode == CommMode::INTER_NODE) {
            tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize_); // max{32K, BS * K * 4B}
            tpipe_->InitBuffer(smoothScalesBuf_, maxSize_);         // max{32K, BS * K * 4B}
        } else {
            tbufPool_.InitBuffer(receiveDataCastFloatBuf_, maxSize_); // max{32K, BS * K * 4B}
            tbufPool_.InitBuffer(smoothScalesBuf_, maxSize_);         // max{32K, BS * K * 4B}
        }
        floatLocalTemp_ = receiveDataCastFloatBuf_.Get<float>();
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN); // 32B
            rowMaxTensor_ = rowMaxBuf_.Get<float>();
        }
    }
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf_, dynamicScalesOutGMTensor_);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitGatherTokenBuffer()
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    InitQuantBuffer<CommMode::INTER_NODE>();
    uint32_t flagDataSize = serverRecordBlockCnt_ * UB_ALIGN;
    uint32_t flagCheckSize = GetFlagCheckBufferSize(serverRecordBlockCnt_);
    uint32_t flagMaxRecvBuf = flagDataSize > flagCheckSize ? flagDataSize : flagCheckSize;
    uint8_t commRepeatTime = static_cast<uint8_t>(Ceil(maxBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256));
    uint32_t commFlagSize = commRepeatTime * SIZE_ALIGN_256;
    TBuf<> tokenRcvStatusBuf;
    TBuf<> tokenFlagBuf;

    uint32_t xInQueueSize = hAlignSize_ > hCommuSize_ ? hAlignSize_ : hCommuSize_;
    uint32_t bufferNum = DOUBLE_BUFFER;
    uint32_t queueNeedSize = serverRecordAlignLen_ + xInQueueSize + 2 * hCommuSize_;
    uint32_t fixedNeedSize = commFlagSize + TOKEN_RECV_CNT_MAX * sizeof(uint32_t) + flagMaxRecvBuf +
                             perServerMapLength_ + GetFlagCheckBufferSize(1) +
                             combineBitmapTokenNum_ * sizeof(uint16_t);
    if (bufferNum * queueNeedSize + 2 * maxSize_ + fixedNeedSize > MAX_UB_SIZE_ROCE) {
        bufferNum = 1;
    }

    tpipe_->InitBuffer(tokenInQueue_, bufferNum, serverRecordAlignLen_); // token输入队列
    tpipe_->InitBuffer(xInQueue_, bufferNum, xInQueueSize);
    tpipe_->InitBuffer(xOutQueue_, bufferNum, hCommuSize_);
    tpipe_->InitBuffer(flagBuf_, commFlagSize);
    tpipe_->InitBuffer(tokenRcvStatusBuf, TOKEN_RECV_CNT_MAX * sizeof(uint32_t)); // 一个server最多发512个token给
    tpipe_->InitBuffer(tokenFlagBuf, flagMaxRecvBuf);
    tpipe_->InitBuffer(xTempQueue_, bufferNum, hCommuSize_); // 17.5KB
    tpipe_->InitBuffer(serverMapBuf_, perServerMapLength_);
    tpipe_->InitBuffer(combineBitmapBuf_, combineBitmapTokenNum_ * sizeof(uint16_t));

    commFlagTensor_ = flagBuf_.Get<uint64_t>();
    tokenRcvStatus_ = tokenRcvStatusBuf.Get<uint32_t>();
    flagRecvTensor_ = tokenFlagBuf.Get<uint64_t>();
    tokenSendMap_ = serverMapBuf_.Get<int32_t>();
    combineBitmapTensor_ = combineBitmapBuf_.Get<uint16_t>();

    Duplicate<uint64_t>(commFlagTensor_, opSequence_, commFlagSize / sizeof(uint64_t));
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitStatusFlag(uint32_t serverIdx,
                                                                                         TBuf<>& serverFlagBuf,
                                                                                         uint32_t& rcvTokenCnt)
{
    LocalTensor<uint64_t> statusFlagTensor;
    LocalTensor<uint32_t> tokenCntTensor;
    GlobalTensor<uint32_t> statusCntGlobal;
    // 源server写入本卡RoCE接收槽的头块（本卡注册区，按source server索引）
    GM_ADDR wAddr =
        GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(static_cast<uint32_t>(serverId_))), serverIdx);
    statusCntGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(wAddr));
    statusFlagTensor = serverFlagBuf.Get<uint64_t>();
    tokenCntTensor = serverFlagBuf.Get<uint32_t>();
    WaitFlag<true>(wAddr + SPLIT_BLOCK_DATA_SIZE, statusFlagTensor, 0, 1);
    DataCopy(tokenCntTensor, statusCntGlobal, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    rcvTokenCnt = tokenCntTensor.GetValue(0);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::FillTriple(LocalTensor<XOutType>& xOutTensor,
                                                                                     uint32_t srcRankIndex,
                                                                                     uint32_t tokenIndex, uint32_t k,
                                                                                     float expertScale)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    LocalTensor<float> xOutTfloat = xOutTensor.template ReinterpretCast<float>();
    // 复用输出槽前，等待上一专家的 V 管线搬运读完。
    SyncFunc<AscendC::HardEvent::V_S>();
    xOutTint32(tokenQuantAlign_) = srcRankIndex;   // 0:srcRank index
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex; // 1:token index
    xOutTint32(tokenQuantAlign_ + 2) = k;          // 2:topK value index
    xOutTfloat(expertScaleAlign_) = expertScale;   // combine中需要，共享专家为1.0
    SyncFunc<AscendC::HardEvent::S_V>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::FillKAndScale(
    LocalTensor<XOutType>& xOutTensor, uint32_t k, float expertScale)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    LocalTensor<float> xOutTfloat = xOutTensor.template ReinterpretCast<float>();
    // 覆写 k/scale 前，等待首次 UB 搬运及上一专家的 Vector 读取完成。
    SyncFunc<AscendC::HardEvent::V_S>();
    xOutTint32(tokenQuantAlign_ + 2) = k;
    xOutTfloat(expertScaleAlign_) = expertScale;
    SyncFunc<AscendC::HardEvent::S_V>();
}

template <RoceDispatchTypeClass>
template <typename T, bool disableL2>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WriteDataToWinOut(LocalTensor<T>& xOutTensor,
                                                                                            LocalTensor<T>& xTempTensor,
                                                                                            GM_ADDR rankGM,
                                                                                            uint32_t blockCnt)
{
    DataCopyParams insertDataParams = {static_cast<uint16_t>(blockCnt), COMM_COPY_BLOCK, 0U, INSERT_DATA_STRIDE};
    DataCopyParams insertFlagParams = {static_cast<uint16_t>(blockCnt), FLAG_COPY_BLOCK, 0U, INSERT_FLAG_STRIDE};
    LocalTensor<uint64_t> xFlagTensor = xTempTensor.template ReinterpretCast<uint64_t>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    // A5 的 UB 到 UB 搬运走 V 管线。
    DataCopy(xTempTensor, xOutTensor, insertDataParams);
    DataCopy(xFlagTensor[SPLIT_BLOCK_DATA_SIZE / sizeof(uint64_t)], commFlagTensor_, insertFlagParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>(); // 等待数据与 flag 拼接完成后再写 GM。
    uint32_t copyCount = blockCnt * SERVER_STATE_ALIGN / sizeof(T);
    GlobalTensor<T> dataDstWinGMTensor;
    dataDstWinGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rankGM));
    if constexpr (disableL2) {
        dataDstWinGMTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    DataCopy(dataDstWinGMTensor, xTempTensor, copyCount);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ProcessToken(
    LocalTensor<XCommInType>& xInTensor, GM_ADDR rankGM, uint32_t topKIndex, uint32_t expertIndex,
    uint32_t srcRankIndex, uint32_t tokenIndex, float expertScale)
{
    LocalTensor<XCommOutType> insertFlagTensor = xTempQueue_.AllocTensor<XCommOutType>();
    xInQueue_.EnQue(xInTensor);
    xInTensor = xInQueue_.DeQue<XCommInType>();
    if constexpr (QuantMode > UNQUANT && IsSmoothScaleExist) {
        LocalTensor<XCommOutType> xOutTensor = xOutQueue_.AllocTensor<XCommOutType>();
        quantInst_.QuantProcess(xOutTensor, xInTensor, expertIndex, scalesCount_, scalesGMTensor_); // 量化
        xOutQueue_.EnQue(xOutTensor);
        xOutTensor = xOutQueue_.DeQue<XCommOutType>();
        FillTriple(xOutTensor, srcRankIndex, tokenIndex, topKIndex, expertScale);
        WriteDataToWinOut<XCommOutType, true>(xOutTensor, insertFlagTensor, rankGM, blockCntPerTokenToExpert_);
        xOutQueue_.FreeTensor(xOutTensor);
    } else {
        // 非平滑：接收记录已由发送侧FillQuadruple写入srcRank/tokenIndex，只需补k/scale
        FillKAndScale(xInTensor, topKIndex, expertScale);
        WriteDataToWinOut<XCommInType, true>(xInTensor, insertFlagTensor, rankGM, blockCntPerTokenToExpert_);
    }
    xTempQueue_.FreeTensor(insertFlagTensor);
}

template <RoceDispatchTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::IsInSameServer(uint32_t targetRankId)
{
    return targetRankId / rankSizePerServer_ == serverId_;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::QuantCleanTensor(
    LocalTensor<XInType>& tokenData)
{
    // 由于MX以及PERGROUP量化在计算scales时每次搬入256字节数据，所以在token搬入前需要对空间填0，避免引入脏数据
    LocalTensor<uint8_t> singleByteTok = tokenData.template ReinterpretCast<uint8_t>();
    Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XInType));
    SyncFunc<AscendC::HardEvent::V_MTE2>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SendToExpert(
    LocalTensor<XCommInType>& recvTmpTensor, uint32_t rcvCnt)
{
    LocalTensor<XCommInType> tokenData = xInQueue_.AllocTensor<XCommInType>();
    tokenInQueue_.EnQue(recvTmpTensor);
    recvTmpTensor = tokenInQueue_.DeQue<XCommInType>();
    if constexpr ((QuantMode > UNQUANT) && IsSmoothScaleExist) {
        // 平滑量化：量化输入只需要原始数据段（hAlignSize_），三元组不随输入带入，由ProcessToken在量化输出上重填
        DataCopy(tokenData, recvTmpTensor, hAlignSize_ / sizeof(XCommInType));
    } else {
        // 非平滑/非量化：记录本身就是要写入专家窗的数据（含发送侧已填的srcRank/tokenIndex）
        DataCopy(tokenData, recvTmpTensor, hScaleIdxSize_ / sizeof(XCommInType));
    }

    LocalTensor<int16_t> xInTensorInt16 = recvTmpTensor.template ReinterpretCast<int16_t>();
    LocalTensor<uint32_t> xInTensor = recvTmpTensor.template ReinterpretCast<uint32_t>();
    LocalTensor<float> xInToFloatTensor = recvTmpTensor.template ReinterpretCast<float>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t srcRankIndex = xInTensor(serverTripleOffset_);
    uint32_t tokenIndex = xInTensor(serverTripleOffset_ + 1);
    uint16_t moeBitmap = 0;
    float expertScale = float(1.0);
    for (int32_t topKIndex = 0; topKIndex < axisK_; topKIndex++) {                      // 发moe专家
        uint32_t dstExpertId = xInTensor(serverExpertIdsOffset_ + topKIndex);           // 取出专家id
        uint32_t dstRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_; // 发往的卡号
        uint32_t curExpertCnt = xInTensorInt16(serverMoeCntOffset_ + topKIndex);        // 取出cnt
        expertScale = xInToFloatTensor(serverExpertScalesOffset_ + topKIndex);

        if (dstExpertId >= moeExpertNum_ || !IsInSameServer(dstRankId)) { // 无效数据或者不在本server
            continue;
        }
        moeBitmap |= static_cast<uint16_t>(1U << topKIndex);
        GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(
            GetWindAddrByRankId(dstRankId) +
            (expertPerSizeOnWin_ * (srcRankIndex * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
            hCommuSize_ * curExpertCnt); // 计算地址偏移
        ProcessToken(tokenData, rankGM, topKIndex, dstExpertId + sharedExpertNum_, srcRankIndex, tokenIndex,
                     expertScale);
    }

    expertScale = float(1.0);
    for (int32_t shareIndex = 0; shareIndex < sharedExpertNum_; shareIndex++) { // 发共享专家
        uint32_t rankIDSharedGroup = epRankId_ % rankNumPerSharedExpert_;
        uint32_t dstRankId = shareIndex * rankNumPerSharedExpert_ + rankIDSharedGroup;
        uint32_t curExpertCnt = rcvCnt;   // cnt即tokenid
        if (!IsInSameServer(dstRankId)) { // 不同server
            continue;
        }
        GM_ADDR rankGM = reinterpret_cast<GM_ADDR>(GetWindAddrByRankId(dstRankId) + expertPerSizeOnWin_ * srcRankIndex +
                                                   hCommuSize_ * curExpertCnt);
        ProcessToken(tokenData, rankGM, axisK_ + shareIndex, shareIndex, srcRankIndex, tokenIndex, expertScale);
    }
    combineBitmapTensor_.SetValue(tokenIndex, moeBitmap);
    combineBitmapEndToken_ = AscendC::Std::max(combineBitmapEndToken_, tokenIndex + 1U);
    xInQueue_.FreeTensor(tokenData);
    tokenInQueue_.FreeTensor(recvTmpTensor);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitToken(uint32_t tokenCnt,
                                                                                    uint32_t serverIdx,
                                                                                    uint32_t startTokenIdx)
{
    uint32_t rcvStatusIndex = 0;
    uint32_t rcvTokenSum = 0;
    uint32_t processTokenNumPerBatch = 1;
    // 源server写入本卡RoCE接收槽token区的起点（本卡注册区，按source server索引）
    GM_ADDR serverRcvBase =
        GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(static_cast<uint32_t>(serverId_))), serverIdx) +
        SERVER_TOKEN_AREA_OFFSET;
    if (serverIdx == serverId_) {
        for (uint32_t i = 0; i < tokenCnt; i++) {
            if (tokenSendMap_.GetValue(i + startTokenIdx) == 0) {
                tokenRcvStatus_(i) = 1;
                rcvTokenSum++;
            }
        }
    }

    while (true) {
        if (rcvTokenSum == tokenCnt) {
            break;
        }
        if (tokenRcvStatus_(rcvStatusIndex) == 1) {
            rcvStatusIndex = (rcvStatusIndex + 1) % tokenCnt; // 轮询查询每个有效的index
            continue;
        }
        GM_ADDR flagAddr = serverRcvBase + (rcvStatusIndex + startTokenIdx) * serverRecordAlignLen_ +
                           SPLIT_BLOCK_DATA_SIZE; // 第一个flag起始地址
        bool isArrived = WaitFlag<false>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, serverRecordBlockCnt_);
        if (isArrived) { // 处理到达的情况
            uint32_t dstPosition = rcvStatusIndex + startTokenIdx;
            uint32_t processTokenBlock = serverRecordBlockCnt_ * processTokenNumPerBatch;
            GM_ADDR dataAddr = serverRcvBase + dstPosition * serverRecordAlignLen_;
            LocalTensor<XCommInType> recvTmpTensor = tokenInQueue_.AllocTensor<XCommInType>();
            ReadDataFromWindows<XCommInType, true>(dataAddr, processTokenBlock, recvTmpTensor);
            SendToExpert(recvTmpTensor, dstPosition);
            tokenRcvStatus_(rcvStatusIndex) = 1;
            rcvTokenSum++;
        }
        rcvStatusIndex = (rcvStatusIndex + 1) % tokenCnt;
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::StoreSharedTokenCount(uint32_t serverIdx,
                                                                                                uint32_t tokenCount,
                                                                                                TBuf<>& serverFlagBuf)
{
    LocalTensor<uint16_t> count = serverFlagBuf.Get<uint16_t>();
    count.SetValue(0, shareRankNumInServer_ > 0U ? static_cast<uint16_t>(tokenCount) : 0U);
    const uint64_t offset = static_cast<uint64_t>(serverIdx) * (combineBitmapTokenNum_ + 1U);
    DataCopyExtParams copyParams{1U, sizeof(uint16_t), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineServerInfoGMTensor_[offset], count, copyParams);
}

template <RoceDispatchTypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GetCombineBitmapStart(
    uint32_t serverIdx, uint32_t startTokenIdx, TBuf<>& serverFlagBuf)
{
    if (serverIdx == static_cast<uint32_t>(serverId_) || startTokenIdx == 0U) {
        return startTokenIdx;
    }
    GM_ADDR record = GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(serverId_)), serverIdx) +
                     SERVER_TOKEN_AREA_OFFSET + static_cast<uint64_t>(startTokenIdx - 1U) * serverRecordAlignLen_;
    WaitFlag<true>(record + SPLIT_BLOCK_DATA_SIZE, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, serverRecordBlockCnt_);
    const uint32_t tokenIdOffset = (serverTripleOffset_ + 1U) * sizeof(int32_t);
    GM_ADDR tokenIdAddr =
        record + tokenIdOffset / SPLIT_BLOCK_DATA_SIZE * SERVER_STATE_ALIGN + tokenIdOffset % SPLIT_BLOCK_DATA_SIZE;
    GlobalTensor<uint32_t> tokenIdGlobal;
    tokenIdGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tokenIdAddr));
    tokenIdGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    LocalTensor<uint32_t> tokenId = serverFlagBuf.Get<uint32_t>();
    DataCopyExtParams copyParams{1U, sizeof(uint32_t), 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> padParams{false, 0U, 0U, 0U};
    DataCopyPad(tokenId, tokenIdGlobal, copyParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    return tokenId.GetValue(0) + 1U;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::StoreCombineBitmap(uint32_t serverIdx,
                                                                                             uint32_t startToken,
                                                                                             uint32_t endToken)
{
    const uint32_t tokenCount = endToken - startToken;
    LocalTensor<uint16_t> output = serverMapBuf_.Get<uint16_t>();
    LocalTensor<int32_t> offsets = tokenRcvStatus_.ReinterpretCast<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    CreateVecIndex(offsets, 0, tokenCount);
    Muls(offsets, offsets, static_cast<int32_t>(sizeof(uint16_t)), tokenCount);
    Adds(offsets, offsets, static_cast<int32_t>(startToken * sizeof(uint16_t)), tokenCount);
    Gather(output, combineBitmapTensor_, tokenRcvStatus_, 0U, tokenCount);
    const uint64_t offset = static_cast<uint64_t>(serverIdx) * (combineBitmapTokenNum_ + 1U) + 1U + startToken;
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(tokenCount * sizeof(uint16_t)), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(combineServerInfoGMTensor_[offset], output, copyParams);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GatherServerTokens(uint32_t serverIdx,
                                                                                             uint32_t firstAivId,
                                                                                             uint32_t useAivPerServer,
                                                                                             TBuf<>& serverFlagBuf)
{
    uint32_t tokenCount;
    WaitStatusFlag(serverIdx, serverFlagBuf, tokenCount);
    if (aivId_ == firstAivId) {
        StoreSharedTokenCount(serverIdx, tokenCount, serverFlagBuf);
    }
    Duplicate(combineBitmapTensor_, static_cast<uint16_t>(0), combineBitmapTokenNum_);
    combineBitmapEndToken_ = 0;
    if (tokenCount == 0U) {
        if (aivId_ == firstAivId) {
            StoreCombineBitmap(serverIdx, 0U, combineBitmapTokenNum_);
        }
        return;
    }
    if (serverIdx == static_cast<uint32_t>(serverId_)) {
        tokenCount = activeMaskBsCnt_;
        DataCopy(tokenSendMap_, offsetGMTensor_, perServerMapCnt_);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
    uint32_t startToken, endToken, tokenNum;
    SplitToCore(tokenCount, firstAivId, useAivPerServer, startToken, endToken, tokenNum);
    if (tokenNum == 0U) {
        return;
    }
    Duplicate(tokenRcvStatus_, 0U, TOKEN_RECV_CNT_MAX);
    SyncFunc<AscendC::HardEvent::V_S>();
    WaitToken(tokenNum, serverIdx, startToken);
    // 远端记录按原 tokenId 递增压紧；本核连同前一记录之后的空洞一起写出，避免跨核覆盖。
    const uint32_t bitmapStart = GetCombineBitmapStart(serverIdx, startToken, serverFlagBuf);
    const uint32_t bitmapEnd = endToken == tokenCount ?
                                   combineBitmapTokenNum_ :
                                   (serverIdx == static_cast<uint32_t>(serverId_) ? endToken : combineBitmapEndToken_);
    StoreCombineBitmap(serverIdx, bitmapStart, bitmapEnd);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GatherTokenFrRemote()
{
    InitGatherTokenBuffer();
    TBuf<> serverFlagBuf;
    uint32_t startServerId, endServerId, useAivPerServer, firstAivId;
    CalServerSplitCore(serverNum_, aivUsedCumSum_ + aivUsedScatter_ + aivUsedRemotComm_, aivUsedGather_, startServerId,
                       endServerId, useAivPerServer, firstAivId);
    tpipe_->InitBuffer(serverFlagBuf, GetFlagCheckBufferSize(1));
    for (uint32_t serverIdx = startServerId; serverIdx < endServerId; ++serverIdx) {
        GatherServerTokens(serverIdx, firstAivId, useAivPerServer, serverFlagBuf);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitLocalCopyBuffer(uint32_t processNum)
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf_, dynamicScalesOutGMTensor_);
    TBuf<> cumSumBuf;
    TBuf<> statusWaitBuf;
    TBuf<> expertMapBuf;
    TBuf<> expertFinishBuf;
    TBuf<> expertLeftBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t expInfoSize = Ceil(processNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t statusWaitBufSize = GetFlagCheckBufferSize(aivUsedCumSum_);
    uint64_t fixedBufferSize =
        static_cast<uint64_t>(UB_ALIGN) + rscvNumAlign + statusWaitBufSize + static_cast<uint64_t>(expInfoSize) * 3U;
    tBufRealSize_ = static_cast<uint32_t>(MAX_UB_SIZE_ROCE - fixedBufferSize);

    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, statusWaitBufSize);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);

    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();

    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitCumSumFlag()
{
    LocalTensor<uint64_t> statusU64Tensor = statusFp32Tensor_.ReinterpretCast<uint64_t>();
    GM_ADDR flagAddr = statusDataSpaceGm_ + CUMSUM_FLAG_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN;
    WaitFlag<true>(flagAddr, statusU64Tensor, UB_ALIGN - sizeof(uint64_t), aivUsedCumSum_);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SetValidExpertInfo(uint32_t startProcessId,
                                                                                             uint32_t endProcessId,
                                                                                             uint32_t processNum,
                                                                                             uint32_t& validNum)
{
    // 获取cumSum
    uint32_t expInfoSize = Ceil(processNum * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    // 后续标量读取该计数计算 flag 地址，必须先完成 Vector 清零。
    SyncFunc<AscendC::HardEvent::V_S>();
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

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CopyOut(LocalTensor<XOutType> tokenTensor,
                                                                                  uint32_t dstPosition,
                                                                                  uint32_t tokenCount)
{
    GlobalTensor<XOutType> expandXOutGlobal;
    expandXOutGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ XOutType*>(expandXOutGM_) + (dstPosition)*copyOutAxisH_);
    expandXOutGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopyExtParams scalesCopyParams{
        uint16_t(tokenCount), static_cast<uint32_t>(scaleOutBytes_),
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams tokenCopyParams{
        uint16_t(tokenCount), hOutSize_,
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{
        uint16_t(tokenCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyExtParams dataCopyexpandScaleParams{
        uint16_t(tokenCount), sizeof(float),
        static_cast<uint32_t>((blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};

    LocalTensor<int32_t> tokenInt32Tensor = tokenTensor.template ReinterpretCast<int32_t>();
    quantInst_.CopyScalesToOut(dstPosition, scaleOutBytes_, tokenTensor, scalesCopyParams);
    DataCopyPad(expandXOutGlobal, tokenTensor, tokenCopyParams);
    DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO], tokenInt32Tensor[tokenQuantAlign_],
                expandIdxCopyParams);
    if (!isShareExpertRankFlag_) {
        LocalTensor<float> tokenFloatTensor = tokenTensor.template ReinterpretCast<float>();
        DataCopyPad(expandScalesOutGMTensor_[dstPosition], tokenFloatTensor[expertScaleAlign_],
                    dataCopyexpandScaleParams); // 拷贝expert专家系数
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitAndFormatOutput(uint32_t validNum)
{
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t tokenUbSize = blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE;
    uint32_t maxCopyTokenCnt = tBufRealSize_ / DOUBLE_BUFFER / tokenUbSize;
    uint32_t maxFlagNum = blockCntPerTokenToExpert_ * maxCopyTokenCnt;
    uint32_t flagBufferSize = GetFlagCheckBufferSize(maxFlagNum);
    // 判旗使用独立buffer；剩余UB均分为两块Token buffer，用队列隐藏MTE2/MTE3时延。
    uint32_t tokenBufferSize = (tBufRealSize_ - flagBufferSize) / DOUBLE_BUFFER / UB_ALIGN * UB_ALIGN;
    maxCopyTokenCnt = tokenBufferSize / tokenUbSize;
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpRankId, dstPosition, copyCnt, srcDataBlockIdx;
    TBuf<> flagCheckBuf;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tokenCopyQueue;
    tpipe_->InitBuffer(flagCheckBuf, flagBufferSize);
    tpipe_->InitBuffer(tokenCopyQueue, DOUBLE_BUFFER, tokenBufferSize);
    flagRecvTensor_ = flagCheckBuf.Get<uint64_t>();
    while (true) {
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum;     // 轮询查询每个有效的index
            continue;
        }
        srcExpRankId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ?
                      maxCopyTokenCnt :
                      expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx =
            srcExpRankId % epWorldSize_ * localExpertNum + srcExpRankId / epWorldSize_; // 转换成数据区的排布偏移
        int32_t beginIdx = expertFinishNumTensor_(index);
        GM_ADDR flagAddr = reinterpret_cast<GM_ADDR>(windowGM_ + srcDataBlockIdx * expertPerSizeOnWin_ +
                                                     beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE);
        uint32_t flagNum = blockCntPerTokenToExpert_ * uint32_t(copyCnt);
        bool isArrived = WaitFlag<false>(flagAddr, flagRecvTensor_, COMM_FLAG_COPY_STRIDE, flagNum);
        if (isArrived) {
            dstPosition = srcExpRankId != 0 ? sendCntTensor_(srcExpRankId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = reinterpret_cast<GM_ADDR>(windowGM_ + srcDataBlockIdx * expertPerSizeOnWin_);
            GlobalTensor<XOutType> tokenGlobal;
            tokenGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ XOutType*>(wAddr));
            tokenGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            DataCopyParams tokenCopyInParams{static_cast<uint16_t>(flagNum),
                                             static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE),
                                             static_cast<uint16_t>(UB_ALIGN), 0U};
            DataCopyPadParams tokenCopyInPadParams{false, 0U, 0U, 0U};
            LocalTensor<XOutType> tokenTensor = tokenCopyQueue.AllocTensor<XOutType>();
            DataCopyPad(tokenTensor, tokenGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(XOutType)],
                        tokenCopyInParams, tokenCopyInPadParams);
            tokenCopyQueue.EnQue(tokenTensor);
            tokenTensor = tokenCopyQueue.DeQue<XOutType>();
            CopyOut(tokenTensor, dstPosition, copyCnt);
            tokenCopyQueue.FreeTensor(tokenTensor);
            // finish更新
            expertFinishNumTensor_(index) += copyCnt;
            expertLeftNumTensor_(index) -= copyCnt;
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
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::LocalWindowCopy()
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
    WaitAndFormatOutput(validNum);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        CalTokenMask();
        if (aivId_ < aivUsedCumSum_) {
            // 共享专家场景要求所有相关数据参与发送。
            CalCumSum();
        } else {
            CalTokenSendExpertOffset();
            if (aivId_ < aivUsedCumSum_ + aivUsedRemotComm_ + aivUsedScatter_) {
                ScatterTokenToRemote();
            } else {
                GatherTokenFrRemote();
            }
        }
        LocalWindowCopy();
    }
}
} // namespace Mc2Kernel
#endif // defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#endif // MOE_DISTRIBUTE_DISPATCH_V2_A5_ROCE_H
