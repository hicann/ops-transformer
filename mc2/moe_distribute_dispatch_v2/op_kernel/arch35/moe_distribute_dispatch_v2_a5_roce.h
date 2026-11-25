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
constexpr uint32_t INSERT_DATA_STRIDE = 1U;
constexpr uint32_t COMM_COPY_BLOCK = 15U;
constexpr uint32_t FLAG_COPY_BLOCK = 1U;
constexpr uint32_t INSERT_FLAG_STRIDE = 15U;
constexpr uint32_t COMM_FLAG_COPY_STRIDE = 504;
constexpr uint32_t BUFPOOL_MAX_BUFFER_ID = 16;    // Tbufpool 最大buffer id
constexpr uint32_t BUFPOLL_MAX_SIZE = 192 * 1024; // 为pool外常驻buffer预留至少56KB
constexpr uint32_t MAX_LOCAL_SEND_BUFFERS = 8U;
constexpr uint32_t ROCE_DATA_STATE_NUM = 2; // RoCE区按dataState_对半分区
constexpr uint32_t ROCE_CHAN_TOKEN = 0U;
constexpr uint32_t ROCE_COUNT_CORES = 8U;
constexpr uint32_t MAX_ROCE_SENDER_CORES = 8U;
constexpr uint32_t MASK_TILE_BLOCKS = 32U;
constexpr uint32_t MASK_COUNT_WORD = 1U;
constexpr uint32_t MASK_HISTOGRAM_BINS = 256U;
constexpr uint32_t MASK_EXPERTS_PER_HISTOGRAM = 254U;
constexpr uint32_t ROCE_WQEBB_BYTES = 64U;
constexpr uint32_t FLAG_COMPARE_ALIGN = 256U;
#define FLOAT_OVERFLOW_MODE_CTRL 60

#define RoceDispatchTypeClass \
    typename ContextHolder, typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define RoceDispatchTypeFunc ContextHolder, XType, ExpandXOutType, QuantMode, IsSmoothScaleExist

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace Mc2Aclnn;

struct RoceMaskWireTile {
    GM_ADDR source;
    GM_ADDR target;
    uint32_t bytes, blocks;
};

struct RoceMaskCountTile {
    uint32_t firstRank, ranks, firstExpert, experts;
    bool fullCards;
};

struct RoceCopyCursor {
    uint32_t nextPair, pairsRead, pair, ordinal, count, destination;
};

template <RoceDispatchTypeClass>
class MoeDistributeDispatchV2Roce {
public:
    using XInType = typename std::conditional<
        (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value), uint8_t, XType>::type;
    using XOutType = typename std::conditional<(Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) ||
                                                   (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
                                               uint8_t, ExpandXOutType>::type;
    using XCommInType = XOutType;
    using XCommOutType = XOutType;
    __aicore__ inline MoeDistributeDispatchV2Roce(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                GM_ADDR expertScales, GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
                                GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe* pipe,
                                const MoeDistributeDispatchV2Info* dispatchV2Info);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitRecieveTilingContext(GM_ADDR expandXOut, TPipe* pipe,
                                                    const MoeDistributeDispatchV2Info* dispatchV2Info);
    __aicore__ inline void InitCommInfo(GM_ADDR mc2Context);
    __aicore__ inline void InitComputeInfo();
    __aicore__ inline void InitSetWindows(const MoeDistributeDispatchV2Info* dispatchV2Info);
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
    __aicore__ inline void CalExpertSendMask();
    __aicore__ inline void CountSourceExperts(uint32_t first, uint32_t count, LocalTensor<uint32_t>& rows);
    __aicore__ inline void WriteSourceMasks(uint32_t first, uint32_t count, LocalTensor<uint32_t>& rows);
    __aicore__ inline void PublishSourceMaskFlags();
    __aicore__ inline void InitCalCumSumBuffer();
    __aicore__ inline uint32_t GetCumSum();
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void PrepareMaskServer(uint32_t server);
    __aicore__ inline void ReadMaskRows(GM_ADDR wire, uint64_t offset, uint32_t bytes, LocalTensor<uint8_t>& rows);
    __aicore__ inline void ForwardMaskTile(const RoceMaskWireTile& tile);
    __aicore__ inline RoceMaskWireTile GetForwardMaskTile(uint32_t server, uint32_t firstRank, uint32_t item);
    __aicore__ inline void PrefetchMaskFlags(const RoceMaskWireTile& tile, uint32_t slot);
    __aicore__ inline RoceMaskCountTile GetCountTile(uint32_t item);
    __aicore__ inline RoceMaskWireTile GetCountWireTile(const RoceMaskCountTile& tile);
    __aicore__ inline bool IssueCountTile(uint32_t item, uint32_t total);
    __aicore__ inline void IssueSenderMask(uint32_t server, uint64_t offset);
    __aicore__ inline void ForwardMaskServer(uint32_t server, uint32_t firstRank, uint32_t endRank);
    __aicore__ inline bool IssueForwardMask(uint32_t server, uint32_t firstRank, uint32_t item, uint32_t total);
    __aicore__ inline void SplitRemoteMaskWork(uint32_t& firstServer, uint32_t& endServer, uint32_t& firstRank,
                                               uint32_t& endRank);
    __aicore__ inline void DistributeExpertMasks();
    __aicore__ inline void ReadReceiveMaskTile(const RoceMaskCountTile& tile, LocalTensor<uint32_t>& rows);
    __aicore__ inline uint32_t ReadReceiveCounts(const RoceMaskCountTile& tile, LocalTensor<uint32_t>& rows);
    __aicore__ inline void PublishCountReady(uint32_t total);
    __aicore__ inline void CollectReceiveCounts();
    __aicore__ inline void CalCumSum();
    template <typename T, bool disableL2 = false, bool queueOwnsBuffer = false>
    __aicore__ inline void ReadDataFromWindows(GM_ADDR dataAddr, uint32_t blockCnt, LocalTensor<T>& xInTensor);
    __aicore__ inline void FillQuadruple(LocalTensor<XCommInType>& xOutTensor, uint32_t tokenIndex);
    template <typename T, bool disableL2 = false, bool queueOutput = true>
    __aicore__ inline void WriteDataToWinOut(LocalTensor<T>& xOutTensor, LocalTensor<T>& xTempTensor, GM_ADDR rankGM,
                                             uint32_t blockCnt);
    template <bool isContinueWait>
    __aicore__ inline bool WaitFlag(GM_ADDR flagAddr, LocalTensor<uint64_t> flagInLTTensor, uint32_t srcStride,
                                    uint32_t waitNum);
    __aicore__ inline uint32_t GetFlagCheckBufferSize(uint32_t flagNum);
    __aicore__ inline bool CheckFlagBySequence(LocalTensor<uint64_t> flagTensor, uint32_t flagNum);
    __aicore__ inline void CalServerSplitCore(uint32_t serverNum, uint32_t startAivId, uint32_t useAivNum,
                                              uint32_t& startServerId, uint32_t& endServerId, uint32_t& useAivPerServer,
                                              uint32_t& firstAivId);
    __aicore__ inline void InitQuantBuffers();
    __aicore__ inline void StoreTokenRoutes(uint32_t startTokenId, uint32_t endTokenId);
    __aicore__ inline void QuantAllTokens();
    __aicore__ inline void InitGatherTokenBuffer();
    __aicore__ inline void PrefetchGatherHeader(uint32_t server);
    __aicore__ inline void WaitStatusFlag(uint32_t serverIdx, TBuf<>& serverFlagBuf, uint32_t& rcvTokenCnt);
    __aicore__ inline void FillKAndScale(LocalTensor<XOutType>& xOutTensor, uint32_t k, float expertScale);
    __aicore__ inline void ProcessToken(LocalTensor<XCommInType>& xInTensor, GM_ADDR rankGM, uint32_t topKIndex,
                                        float expertScale);
    __aicore__ inline bool IsInSameServer(uint32_t targetRankId);
    __aicore__ inline void QuantCleanTensor(LocalTensor<XInType>& tokenData);
    __aicore__ inline void InitQuantScratch(uint32_t bufferIndex);
    __aicore__ inline void StoreTokenBitmap(uint32_t serverIdx, uint32_t tokenId, uint32_t bitmap,
                                            uint32_t tokenStride);
    __aicore__ inline void SendToExpert(LocalTensor<XCommInType>& recvTmpTensor, uint32_t serverIdx,
                                        uint32_t tokenStride, uint32_t bufferIndex);
    __aicore__ inline void WaitToken(uint32_t tokenCnt, uint32_t serverIdx, uint32_t startTokenIdx,
                                     uint32_t tokenStride);
    __aicore__ inline void ClearOwnedBitmapTail(uint32_t serverIdx, uint32_t tokenCount, uint32_t firstToken,
                                                uint32_t tokenStride);
    __aicore__ inline void GatherServerTokens(uint32_t serverIdx, uint32_t firstAivId, uint32_t useAivPerServer,
                                              TBuf<>& serverFlagBuf);
    __aicore__ inline void GatherTokenFrRemote();
    __aicore__ inline void HcclTokenDispatch();
    __aicore__ inline void InitSenderBuffers(TBuf<>& batchBuf);
    __aicore__ inline void FillServerHeader(LocalTensor<uint64_t> header, uint32_t elements);
    __aicore__ inline void PublishLocalServerHeader(LocalTensor<uint64_t> header);
    __aicore__ inline void SubmitServerBatch(uint32_t server, GM_ADDR source, LocalTensor<uint8_t> storage);
    __aicore__ inline void ReadTokenData(uint32_t tokenIndex, LocalTensor<XInType>& tokenData);
    __aicore__ inline void QuantToken(uint32_t tokenIndex, LocalTensor<XInType>& xInTensor,
                                      LocalTensor<XCommInType>& xOutTensor, uint32_t bufferIndex);
    __aicore__ inline void WriteToken(uint32_t tokenIndex, LocalTensor<XCommInType>& xOutTensor);
    __aicore__ inline void InitLocalCopyBuffer(uint32_t firstPair, uint32_t endPair);
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void WaitAndFormatOutput(RoceCopyCursor& cursor, uint32_t end, uint32_t active,
                                               uint32_t tokenCount);
    __aicore__ inline void InitCopyTokenIndices();
    __aicore__ inline void BuildCopyTokenIds(uint32_t slot);
    __aicore__ inline bool IssueCopyPair(uint32_t& pair, uint32_t end, uint32_t slot);
    __aicore__ inline GM_ADDR GetCopyTokenAddr(uint32_t pair, uint32_t ordinal);
    __aicore__ inline void SubmitCopyFlag(uint32_t pair, uint32_t ordinal, uint32_t destination, uint32_t slot);
    __aicore__ inline bool IssueCopyToken(uint32_t slot, uint32_t queueSlot);
    __aicore__ inline void LoadNextCopyPair(RoceCopyCursor& cursor, uint32_t end);
    __aicore__ inline void PrefetchCopyTokens(RoceCopyCursor& cursor, uint32_t end, uint32_t active, uint32_t slots);
    __aicore__ inline void CopyExpertPairs(uint32_t first, uint32_t end, uint32_t active);
    __aicore__ inline void CopyOut(LocalTensor<XOutType> tokenTensor, uint32_t dstPosition, uint32_t tokenCount);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void InitMaskWorkspace(const MoeDistributeDispatchV2Info* dispatchV2Info);

    __aicore__ inline GM_ADDR GetRoceDispatchAddr(GM_ADDR roceBufferBase) const
    {
        return roceBufferBase + static_cast<uint64_t>(dataState_) * roceStateCapacity_;
    }

    // 以下地址均以GetRoceDispatchAddr返回的当前半区起点为基准；serverId须小于serverNum_。
    __aicore__ inline GM_ADDR GetRoceMaskSendAddr(GM_ADDR dispatchBase, uint32_t targetServerId) const
    {
        return dispatchBase + static_cast<uint64_t>(serverNum_) * SERVER_STATE_ALIGN +
               static_cast<uint64_t>(targetServerId) * maskSlotBytes_;
    }

    // 接收区按 source server 索引：框间只有与本地同 localIndex 的远端卡会发给本卡，
    // 一个 server 只需一个接收槽，容量按本卡单次最多发送的 token 数(axisMaxBS_)计算。
    __aicore__ inline GM_ADDR GetRoceReceiveAddr(GM_ADDR dispatchBase, uint32_t sourceServerId) const
    {
        return dispatchBase + roceSendAreaSize_ + static_cast<uint64_t>(sourceServerId) * roceReceiveSlotSize_;
    }

    __aicore__ inline GM_ADDR GetRoceMaskReceiveAddr(GM_ADDR dispatchBase, uint32_t sourceServerId) const
    {
        return GetRoceReceiveAddr(dispatchBase, sourceServerId) + SERVER_TOKEN_AREA_OFFSET +
               static_cast<uint64_t>(axisMaxBS_) * serverRecordAlignLen_;
    }

    // server的RoCE注册区首地址：本地项为本卡，远端项为同localIndex的对端卡
    __aicore__ inline GM_ADDR GetRoceBaseByServerId(uint32_t serverIndex) const
    {
        return reinterpret_cast<GM_ADDR>(ctx_.GetRoceBufferByServerId(serverIndex));
    }

    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindAddrByRankId(rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindStateAddrByRankId(rankId, epRankIdOriginal_) + dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline void InitPrefetchBuffers(uint32_t depth, uint32_t slotBytes);
    __aicore__ inline void ReleasePrefetchBuffers();
    __aicore__ inline void SubmitPrefetchedFlag(uint32_t slot, GM_ADDR address, uint32_t stride, uint32_t count);
    __aicore__ inline void ReloadPrefetchedFlag(uint32_t slot);
    __aicore__ inline bool TestPrefetchedFlag(uint32_t slot);
    __aicore__ inline void IssueQuantInput(uint32_t token, uint32_t slot);
    __aicore__ inline bool IssueGatherInput(GM_ADDR source, uint32_t index, uint32_t total, uint32_t first,
                                            uint32_t stride, bool local);
    TPipe* tpipe_{nullptr};
    GlobalTensor<XInType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<float> expertScalesGMTensor_;
    GlobalTensor<float> expandScalesOutGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<uint32_t> combineServerInfoGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<uint32_t> combineBitmapTensor_;
    LocalTensor<bool> expertMaskInputTensor_;
    LocalTensor<uint64_t> commFlagTensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<uint32_t> pendingTokenIds_;

    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> scalarBuf_; // 辅助gather tensor定义
    TBuf<> receiveDataCastFloatBuf_[MAX_LOCAL_SEND_BUFFERS];
    TBuf<> smoothScalesBuf_[MAX_LOCAL_SEND_BUFFERS];
    TBuf<> tokenDataBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> expertMaskInputBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> combineBitmapBuf_;
    TBuf<> gatherPacketBuf_;
    TBuf<> flagPrefetchBuf_;
    TEventID flagReadyEvents_[MAX_LOCAL_SEND_BUFFERS];
    TEventID flagFreeEvents_[MAX_LOCAL_SEND_BUFFERS];
    TEventID payloadReadyEvents_[MAX_LOCAL_SEND_BUFFERS];
    TEventID quantReadyEvents_[MAX_LOCAL_SEND_BUFFERS];
    TEventID quantFreeEvents_[MAX_LOCAL_SEND_BUFFERS];
    GM_ADDR flagAddresses_[MAX_LOCAL_SEND_BUFFERS];
    uint32_t flagStrides_[MAX_LOCAL_SEND_BUFFERS];
    uint32_t flagCounts_[MAX_LOCAL_SEND_BUFFERS];
    bool flagPending_[MAX_LOCAL_SEND_BUFFERS];
    uint32_t flagPrefetchDepth_{0};
    uint32_t flagSlotWords_{0};
    TEventID gatherPacketFree_[MAX_LOCAL_SEND_BUFFERS];
    uint32_t gatherPacketSlot_{0};
    uint32_t quantOutputBufferNum_{0};
    uint32_t quantInputBufferNum_{0};
    uint32_t gatherPacketBufferNum_{0};
    uint32_t gatherInputBufferNum_{0};
    uint32_t copyBufferNum_{0};
    uint32_t copyPairs_[MAX_LOCAL_SEND_BUFFERS];
    GM_ADDR copyTokenSources_[MAX_LOCAL_SEND_BUFFERS];
    uint32_t copyFlagPositions_[MAX_LOCAL_SEND_BUFFERS];
    uint32_t copyPayloadPositions_[MAX_LOCAL_SEND_BUFFERS];
    bool copyFlagActive_[MAX_LOCAL_SEND_BUFFERS];
    uint32_t countMaskBufferNum_{0};
    uint32_t senderMaskBufferNum_{0};
    TBuf<> flagBuf_;
    TBuf<> pendingTokenBuf_;
    TBuf<> localCopyPrefixBuf_;
    TBuf<> localCopyStatusBuf_;
    TBuf<> localCopyMaskBuf_;
    TBuf<> localCopyIndexBuf_;
    TBuf<> expertFreqBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, MAX_LOCAL_SEND_BUFFERS> localCopyQueue_;

    TQue<QuePosition::VECIN, MAX_LOCAL_SEND_BUFFERS> tokenInQueue_;
    TQue<QuePosition::VECIN, MAX_LOCAL_SEND_BUFFERS> xInQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, MAX_LOCAL_SEND_BUFFERS> countMaskQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, MAX_LOCAL_SEND_BUFFERS> senderMaskQueue_;
    TQue<QuePosition::VECOUT, 1> xTempQueue_;                      // 用于填充flag
    TBufPool<TPosition::VECCALC, BUFPOOL_MAX_BUFFER_ID> tbufPool_; // 用于存储tbuf的pool

    GM_ADDR expandXOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR windowGM_;
    GM_ADDR statusDataSpaceGm_;

    GM_ADDR workspaceGM_;
    GM_ADDR countReadyGM_;
    uint64_t maskAreaOffset_{0};
    GM_ADDR rawMaskGM_;
    GM_ADDR tokenRouteGM_;
    uint32_t tokenRouteBytes_{0};
    uint32_t maskBytes_{0};
    uint32_t maskCardBytes_{0};
    uint32_t copyPrefixStart_{0};
    uint64_t expertTokenStride_{0};

    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t combineBitmapTokenNum_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    int32_t epRankId_{0};
    int32_t serverId_{0};          // 当前卡所在server编号
    int32_t rankSizePerServer_{0}; // 一个server内的rank数 epWorldSize_ -> rankSizePerServer_
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t moeExpertNum_{0};
    uint32_t globalBS_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t startRankId_{0}; // 当前server的起始卡号 全局编号
    uint32_t expertNumInServer_{0}; // 当前server中共享专家副本数 + moe专家数 totalExpertNum_ -> expertNumInServer_
    uint32_t hOutSize_{0};
    uint32_t hAlignSize_{0};
    uint32_t hCommuSize_{0};
    uint32_t serverTripleOffset_{0};       // server窗记录：srcRank/tokenIndex起始，int32元素偏移
    uint32_t serverExpertIdsOffset_{0};    // server窗记录：expertIds列表起始，int32元素偏移
    uint32_t serverExpertScalesOffset_{0}; // server窗记录：expertScales起始，float元素偏移
    uint32_t serverRecordLen_{0};          // server窗单条记录字节长度（数据+三元组+元数据）
    uint32_t serverRecordBlockCnt_{0};     // server窗单条记录480B块数（写窗/等flag用）
    uint32_t serverRecordAlignLen_{0};     // server窗单条记录512B对齐后长度（记录槽位步长）
    uint64_t maskSlotBytes_{0};            // Registered mask staging bytes per destination server.
    uint64_t roceSendAreaSize_{0};         // Header and mask staging, separate from receive slots.
    uint64_t roceReceiveSlotSize_{0};      // One source server: header, fixed token slots and mask wire.
    uint64_t roceBufferSize_{0};           // RoCE注册区容量，取自context(与URMA窗口winSize同语义)
    uint64_t roceStateCapacity_{0};        // RoCE单个0/1半区容量=roceBufferSize_/ROCE_DATA_STATE_NUM
    uint32_t dataState_{0};
    uint64_t opSequence_{0}; // 本轮绝对轮次R(win状态块opCnt+1，InitWinState维护)，通信flag统一使用
    uint32_t axisBsAlignSize_{0};
    uint32_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffsetEp_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    uint64_t totalWinSizeEp_{0};
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t maxSize_{0};
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
    uint32_t aivUsedRemotComm_{0}; // 用于专门框间通信的aiv数
    uint32_t aivUsedGather_{0};
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};

    AscendC::Hcomm<COMM_PROTOCOL_ROCE> hcomm_; // 框间RoCE通信对象
    ContextHolder ctx_;
    MoeDistributeDispatchV2Quant<XInType, ExpandXOutType, XOutType, QuantMode, IsSmoothScaleExist> quantInst_;
};

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitRecieveTilingContext(
    GM_ADDR expandXOut, TPipe* pipe, const MoeDistributeDispatchV2Info* dispatchV2Info)
{
    InitMaskWorkspace(dispatchV2Info);

    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    tpipe_->InitBufPool(tbufPool_, BUFPOLL_MAX_SIZE);

    totalWinSizeEp_ = static_cast<uint64_t>(dispatchV2Info->totalWinSizeEp);
    epRankId_ = dispatchV2Info->epRankId;
    epRankIdOriginal_ = dispatchV2Info->epRankId;
    quantOutputBufferNum_ = dispatchV2Info->roceQuantBufferNum;
    quantInputBufferNum_ = quantOutputBufferNum_;
    gatherPacketBufferNum_ = dispatchV2Info->roceGatherBufferNum;
    gatherInputBufferNum_ = gatherPacketBufferNum_;
    copyBufferNum_ = dispatchV2Info->roceCopyBufferNum;
    countMaskBufferNum_ = dispatchV2Info->roceCountMaskBufferNum;
    senderMaskBufferNum_ = dispatchV2Info->roceSenderMaskBufferNum;
    axisBS_ = dispatchV2Info->bs;
    axisH_ = dispatchV2Info->h;
    epWorldSizeOriginal_ = dispatchV2Info->epWorldSize;
    epWorldSize_ = dispatchV2Info->epWorldSize;
    moeExpertNum_ = dispatchV2Info->moeExpertNum;
    globalBS_ = dispatchV2Info->globalBs;
    scaleInBytes_ = dispatchV2Info->scalesCol * dispatchV2Info->scalesTypeSize; // 量化使用
    scalesCount_ = dispatchV2Info->scalesCount;                                 // 量化使用
    isTokenMaskFlag_ = dispatchV2Info->isTokenMask;
    isExpertMaskFlag_ = dispatchV2Info->isExpertMask;
    expertTokenNumsType_ = dispatchV2Info->expertTokenNumsType;
    axisK_ = dispatchV2Info->k;
    aivNum_ = dispatchV2Info->aivNum;
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
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitMaskWorkspace(
    const MoeDistributeDispatchV2Info* dispatchV2Info)
{
    const auto& info = *dispatchV2Info;
    tokenRouteBytes_ = UB_ALIGN + Align32(info.k * sizeof(int32_t));
    maskBytes_ = Align32(Ceil(info.globalBs / info.epWorldSize * info.k, BITS_PER_BYTE));
    maskCardBytes_ = Ceil(info.moeExpertNum / info.epWorldSize * (UB_ALIGN + maskBytes_), SPLIT_BLOCK_DATA_SIZE) *
                     SERVER_STATE_ALIGN;
    rawMaskGM_ = workspaceGM_;
    countReadyGM_ = rawMaskGM_ + static_cast<uint64_t>(info.epWorldSize) * maskCardBytes_;
    tokenRouteGM_ = countReadyGM_ + ROCE_COUNT_CORES * UB_ALIGN;
    const uint64_t rawTokenBytes = Align32(static_cast<uint64_t>(info.h) * sizeof(half));
    expertTokenStride_ =
        (rawTokenBytes + 2U * UB_ALIGN + SPLIT_BLOCK_DATA_SIZE - 1U) / SPLIT_BLOCK_DATA_SIZE * SERVER_STATE_ALIGN;
    const uint64_t expertCapacity = AscendC::Std::max(1U, info.moeExpertNum / info.epWorldSize);
    maskAreaOffset_ = static_cast<uint64_t>(info.globalBs / info.epWorldSize) * info.epWorldSize * expertCapacity *
                      expertTokenStride_;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitCommInfo(GM_ADDR mc2Context)
{
    ctx_.InitAndCheck(mc2Context, epWorldSize_, totalWinSizeEp_, tpipe_, expandXOutGM_);
    uint32_t epRankIdHccl = ctx_.GetEpRankId();
    statusDataSpaceGm_ = ctx_.GetStatusDataSpaceGm();
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
    rankSizePerServer_ = ctx_.GetRankNumPerUbDomain();
    serverNum_ = epWorldSize_ / rankSizePerServer_;
    aivUsedCumSum_ = ROCE_COUNT_CORES;
    aivUsedRemotComm_ = AscendC::Std::min(serverNum_, MAX_ROCE_SENDER_CORES);
    aivUsedGather_ = aivNum_ - aivUsedRemotComm_ - aivUsedCumSum_;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitComputeInfo()
{
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    combineBitmapTokenNum_ = Align32(axisMaxBS_ * sizeof(uint32_t)) / sizeof(uint32_t);
    moeExpertNumPerRank_ = moeExpertNum_ / epWorldSize_;
    serverId_ = epRankId_ / rankSizePerServer_;
    startRankId_ = serverId_ * rankSizePerServer_;
    expertNumInServer_ = rankSizePerServer_ * moeExpertNumPerRank_;
    expertIdsCnt_ = axisBS_ * axisK_;
    rscvStatusNum_ = epWorldSize_ * moeExpertNumPerRank_;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitSetWindows(
    const MoeDistributeDispatchV2Info* dispatchV2Info)
{
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<XOutType>(reinterpret_cast<__gm__ XOutType*>(GetWindAddrByRankId(tempepRankId)),
                                    totalWinSizeEp_);
        OOMCheckAddrRange<float>(reinterpret_cast<__gm__ float*>(GetWindStateAddrByRankId(tempepRankId)), STATE_SIZE);
    }
#endif
    winDataSizeOffsetEp_ = dataState_ * (totalWinSizeEp_ / 2) + dispatchV2Info->roceDispatchOffset;
    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<XOutType> winDouble;
    winDouble.SetGlobalBuffer(reinterpret_cast<__gm__ XOutType*>(windowGM_));
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    OOMCheckAddrRange<XOutType>((__gm__ XOutType*)(winDouble.GetPhyAddr()), totalWinSizeEp_);
#endif
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitExtraInfo()
{
    hOutSize_ = copyOutAxisH_ * sizeof(XOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hScaleIdxSize_, scaleOutBytes_,
                         axisH_);
    if constexpr (QuantMode == MX_QUANT || QuantMode == MX_QUANT_CLIP) {
        // MX loads two vector registers (256 FP16/BF16 elements) even on the last iteration.
        hAlignSize_ = Align256(axisH_) * sizeof(XInType);
    }
    hScaleIdxSize_ = Align32(hScaleIdxSize_);
    expertScaleAlign_ = hScaleIdxSize_ / sizeof(uint32_t);
    tokenAlign_ = hScaleIdxSize_ + UB_ALIGN;
    serverTripleOffset_ = tokenQuantAlign_;
    serverExpertIdsOffset_ = serverTripleOffset_ + UB_ALIGN / sizeof(int32_t);
    serverExpertScalesOffset_ = serverExpertIdsOffset_ + Align32(axisK_ * sizeof(int32_t)) / sizeof(int32_t);
    serverRecordLen_ = serverExpertScalesOffset_ * sizeof(int32_t) + Align32(axisK_ * sizeof(float));
    serverRecordBlockCnt_ = Ceil(serverRecordLen_, SPLIT_BLOCK_DATA_SIZE);
    maskSlotBytes_ = static_cast<uint64_t>(rankSizePerServer_) * maskCardBytes_;
    roceSendAreaSize_ = static_cast<uint64_t>(serverNum_) * (SERVER_STATE_ALIGN + maskSlotBytes_);
    blockCntPerTokenToExpert_ = Ceil(tokenAlign_, SPLIT_BLOCK_DATA_SIZE);
    serverRecordAlignLen_ = serverRecordBlockCnt_ * SERVER_STATE_ALIGN;
    roceReceiveSlotSize_ =
        SERVER_STATE_ALIGN + static_cast<uint64_t>(axisMaxBS_) * serverRecordAlignLen_ + maskSlotBytes_;
    roceBufferSize_ = ctx_.GetRoceBufferSize();
    roceStateCapacity_ = roceBufferSize_ / ROCE_DATA_STATE_NUM;
    hCommuSize_ = blockCntPerTokenToExpert_ * SERVER_STATE_ALIGN;
    maxBlockCnt_ = AscendC::Std::max(serverRecordBlockCnt_, blockCntPerTokenToExpert_);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitMaskInfo()
{
    activeMaskBsCnt_ = axisBS_;
    const uint32_t idsBytes = Ceil(expertIdsCnt_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    maxSize_ = AscendC::Std::max(idsBytes, Align32(axisBS_ * sizeof(half)));
    maxSize_ = AscendC::Std::max(maxSize_, GetFlagCheckBufferSize(aivNum_));
    maxSize_ = Ceil(maxSize_, ALIGNED_LEN_256) * ALIGNED_LEN_256;
    tpipe_->InitBuffer(expertIdsBuf_, idsBytes);
    tbufPool_.InitBuffer(gatherMaskTBuf_, maxSize_);
    tbufPool_.InitBuffer(dstExpBuf_, maxSize_);
    tbufPool_.InitBuffer(expertFreqBuf_, MASK_HISTOGRAM_BINS * sizeof(uint16_t));
    const uint32_t maskBufferBytes = Ceil(moeExpertNum_, aivNum_) * (UB_ALIGN + maskBytes_);
    const uint32_t routeBufferBytes = Ceil(axisBS_, aivNum_) * tokenRouteBytes_;
    tbufPool_.InitBuffer(subExpBuf_, AscendC::Std::max(maxSize_, AscendC::Std::max(maskBufferBytes, routeBufferBytes)));
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    if (isExpertMaskFlag_) {
        tbufPool_.InitBuffer(expertMaskInputBuf_, expertIdsCnt_ * sizeof(bool));
        expertMaskInputTensor_ = expertMaskInputBuf_.Get<bool>();
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitDispatchBetweenServerInfo()
{
    const uint32_t commFlagSize = Ceil(maxBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(flagBuf_, commFlagSize);
    commFlagTensor_ = flagBuf_.Get<uint64_t>();
    Duplicate<uint64_t>(commFlagTensor_, opSequence_, commFlagSize / sizeof(uint64_t));
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR sendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, TPipe* pipe,
    const MoeDistributeDispatchV2Info* dispatchV2Info)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ XInType*>(x));
    xGMTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    xActiveMaskGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(xActiveMask));
    expertScalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(expertScales));
    expandScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(expandScalesOut));
    expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
    dynamicScalesOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dynamicScalesOut));
    expertTokenNumsOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(expertTokenNumsOut));
    expandIdxGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expandIdxOut));
    combineServerInfoGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
        expandIdxOut + static_cast<uint64_t>(dispatchV2Info->a) * EXPAND_IDX_INFO * sizeof(int32_t)));
    workspaceGM_ = workspaceGM;
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut; // 无GlobalTensor
    if constexpr (IsSmoothScaleExist) {
        scalesGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    }
    InitRecieveTilingContext(expandXOut, pipe, dispatchV2Info);
    InitCommInfo(mc2Context);
    InitComputeInfo();
    InitSetWindows(dispatchV2Info);
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
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ExpertActiveMaskCal()
{
    LocalTensor<int8_t> input = expertMaskInputTensor_.ReinterpretCast<int8_t>();
    LocalTensor<half> halfMask = subExpBuf_.Get<half>();
    LocalTensor<int32_t> mask = gatherMaskTBuf_.Get<int32_t>();
    Adds(expertIdsTensor_, expertIdsTensor_, moeExpertNum_, expertIdsCnt_);
    Cast(halfMask, input, RoundMode::CAST_NONE, expertIdsCnt_);
    Cast(mask, halfMask, RoundMode::CAST_RINT, expertIdsCnt_);
    Mul(expertIdsTensor_, expertIdsTensor_, mask, expertIdsCnt_);
    Adds(expertIdsTensor_, expertIdsTensor_, -moeExpertNum_, expertIdsCnt_);
    Abs(expertIdsTensor_, expertIdsTensor_, expertIdsCnt_);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalTokenMask()
{
    InitMaskReSource();
    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }
    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalExpertSendMask()
{
    uint32_t firstExpert, endExpert, expertCount;
    SplitToCore(moeExpertNum_, 0U, aivNum_, firstExpert, endExpert, expertCount);
    const uint32_t recordBytes = UB_ALIGN + maskBytes_;
    const uint32_t compareCount =
        Ceil(activeMaskBsCnt_ * axisK_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    LocalTensor<uint32_t> expertMaskTensorU32 = subExpBuf_.Get<uint32_t>();
    if (expertCount != 0U) {
        CountSourceExperts(firstExpert, expertCount, expertMaskTensorU32);
        if (compareCount != 0U) {
            for (uint32_t expertIndex = firstExpert; expertIndex < endExpert; ++expertIndex) {
                const uint32_t maskOffset = (expertIndex - firstExpert) * recordBytes + UB_ALIGN;
                LocalTensor<uint8_t> mask = expertMaskTensorU32.ReinterpretCast<uint8_t>()[maskOffset];
                CompareScalar(mask, expertIdsTensor_, static_cast<int32_t>(expertIndex), CMPMODE::EQ, compareCount);
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        WriteSourceMasks(firstExpert, expertCount, expertMaskTensorU32);
    }
    PublishSourceMaskFlags();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CountSourceExperts(
    uint32_t first, uint32_t count, LocalTensor<uint32_t>& rows)
{
    static_assert(AscendC::GetVecLen() == MASK_HISTOGRAM_BINS, "Mask histogram must match GetVecLen()");
    LocalTensor<int32_t> shifted = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint8_t> bins = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<uint16_t> freq = expertFreqBuf_.Get<uint16_t>();
    const uint32_t valid = activeMaskBsCnt_ * axisK_;
    const uint32_t rowWords = (UB_ALIGN + maskBytes_) / sizeof(uint32_t);
    for (uint32_t begin = 0U; begin < count; begin += MASK_EXPERTS_PER_HISTOGRAM) {
        const uint32_t size = AscendC::Std::min(MASK_EXPERTS_PER_HISTOGRAM, count - begin);
        if (valid != 0U) {
            Subs(shifted, expertIdsTensor_, static_cast<int32_t>(first + begin) - 1, valid);
            PipeBarrier<PIPE_V>();
            Mins(shifted, shifted, static_cast<int32_t>(size + 1U), valid);
            PipeBarrier<PIPE_V>();
            Maxs(shifted, shifted, int32_t(0), valid);
            PipeBarrier<PIPE_V>();
            Cast(bins, shifted, RoundMode::CAST_NONE, valid);
            PipeBarrier<PIPE_V>();
            GetExpertFreq(freq, bins, valid);
            SyncFunc<HardEvent::V_S>();
        }
        for (uint32_t i = 0U; i < size; ++i) {
            const uint32_t word = (begin + i) * rowWords;
            rows.SetValue(word, activeMaskBsCnt_);
            rows.SetValue(word + MASK_COUNT_WORD, valid == 0U ? 0U : freq.GetValue(i + 1U));
        }
        SyncFunc<HardEvent::S_V>();
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WriteSourceMasks(uint32_t first,
                                                                                           uint32_t count,
                                                                                           LocalTensor<uint32_t>& rows)
{
    const uint32_t rowBytes = UB_ALIGN + maskBytes_;
    const uint32_t cardPayload = moeExpertNumPerRank_ * rowBytes;
    uint32_t card = first / moeExpertNumPerRank_;
    uint32_t logical = first % moeExpertNumPerRank_ * rowBytes;
    uint32_t copied = 0U;
    while (copied < count * rowBytes) {
        const uint32_t within = logical % SPLIT_BLOCK_DATA_SIZE;
        const uint32_t bytes = AscendC::Std::min(
            count * rowBytes - copied, AscendC::Std::min(SPLIT_BLOCK_DATA_SIZE - within, cardPayload - logical));
        GlobalTensor<uint64_t> output;
        output.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t*>(rawMaskGM_ + static_cast<uint64_t>(card) * maskCardBytes_ +
                                               logical / SPLIT_BLOCK_DATA_SIZE * SERVER_STATE_ALIGN + within));
        DataCopy(output, rows.ReinterpretCast<uint64_t>()[copied / sizeof(uint64_t)], bytes / sizeof(uint64_t));
        copied += bytes;
        logical += bytes;
        if (logical == cardPayload) {
            ++card;
            logical = 0U;
        }
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::PublishSourceMaskFlags()
{
    const uint32_t cardBlocks = maskCardBytes_ / SERVER_STATE_ALIGN;
    const uint32_t tail = moeExpertNumPerRank_ * (UB_ALIGN + maskBytes_) % SPLIT_BLOCK_DATA_SIZE;
    uint32_t first, end, count;
    SplitToCore(epWorldSize_ * cardBlocks, 0U, aivNum_, first, end, count);
    LocalTensor<uint64_t> zeros = expertFreqBuf_.Get<uint64_t>();
    Duplicate(zeros, uint64_t(0), SPLIT_BLOCK_DATA_SIZE / sizeof(uint64_t));
    SyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(rawMaskGM_));
    for (uint32_t block = first; block < end; ++block) {
        const uint64_t offset = static_cast<uint64_t>(block) * SERVER_STATE_ALIGN;
        if (tail != 0U && (block + 1U) % cardBlocks == 0U) {
            DataCopy(output[(offset + tail) / sizeof(uint64_t)], zeros,
                     (SPLIT_BLOCK_DATA_SIZE - tail) / sizeof(uint64_t));
        }
        DataCopy(output[(offset + SPLIT_BLOCK_DATA_SIZE) / sizeof(uint64_t)], commFlagTensor_,
                 UB_ALIGN / sizeof(uint64_t));
    }
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
    flagInGMTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
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
template <typename T, bool disableL2, bool queueOwnsBuffer>
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
    if constexpr (!queueOwnsBuffer) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    DataCopyPad(xInTensor, dataFlagGlobal, srcTokenCopyParams, srcTokenPadParams);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::IssueSenderMask(uint32_t server,
                                                                                          uint64_t offset)
{
    const uint32_t bytes = AscendC::Std::min<uint64_t>(maskSlotBytes_ - offset, MASK_TILE_BLOCKS * SERVER_STATE_ALIGN);
    LocalTensor<uint8_t> wire = senderMaskQueue_.AllocTensor<uint8_t>();
    GlobalTensor<uint8_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(rawMaskGM_ + server * maskSlotBytes_ + offset));
    DataCopy(wire, source, bytes);
    senderMaskQueue_.EnQue(wire);
}

template <RoceDispatchTypeClass>
__aicore__ inline RoceMaskWireTile MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GetForwardMaskTile(
    uint32_t server, uint32_t firstRank, uint32_t item)
{
    const uint32_t tiles = Ceil(maskCardBytes_, MASK_TILE_BLOCKS * SERVER_STATE_ALIGN);
    const uint32_t rank = firstRank + item / tiles;
    const uint32_t offset = item % tiles * MASK_TILE_BLOCKS * SERVER_STATE_ALIGN;
    const uint32_t bytes = AscendC::Std::min(MASK_TILE_BLOCKS * SERVER_STATE_ALIGN, maskCardBytes_ - offset);
    GM_ADDR source = server == static_cast<uint32_t>(serverId_) ?
                         rawMaskGM_ + server * maskSlotBytes_ :
                         GetRoceMaskReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(serverId_)), server);
    const uint32_t sourceRank = server * rankSizePerServer_ + epRankId_ % rankSizePerServer_;
    GM_ADDR target = GetWindAddrByRankId(startRankId_ + rank) + maskAreaOffset_ +
                     static_cast<uint64_t>(sourceRank) * maskCardBytes_ + offset;
    return {source + static_cast<uint64_t>(rank) * maskCardBytes_ + offset, target, bytes,
            static_cast<uint32_t>(bytes / SERVER_STATE_ALIGN)};
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::PrefetchMaskFlags(
    const RoceMaskWireTile& tile, uint32_t slot)
{
    SubmitPrefetchedFlag(slot, tile.source + SPLIT_BLOCK_DATA_SIZE, COMM_FLAG_COPY_STRIDE, tile.blocks);
}

template <RoceDispatchTypeClass>
__aicore__ inline RoceMaskCountTile MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GetCountTile(uint32_t item)
{
    const uint32_t row = UB_ALIGN + maskBytes_;
    const uint32_t blocks = Ceil(moeExpertNumPerRank_ * row, SPLIT_BLOCK_DATA_SIZE);
    const bool full = blocks <= MASK_TILE_BLOCKS;
    const uint32_t first = startStatusIndex_ / epWorldSize_;
    const uint32_t end = Ceil(endStatusIndex_, epWorldSize_);
    const uint32_t experts =
        full ? end - first : (MASK_TILE_BLOCKS * SPLIT_BLOCK_DATA_SIZE - (SPLIT_BLOCK_DATA_SIZE - UB_ALIGN)) / row;
    const uint32_t ranks = full ? MASK_TILE_BLOCKS / blocks : 1U;
    const uint32_t rankTiles = Ceil(epWorldSize_, ranks);
    const uint32_t expert = first + item / rankTiles * experts;
    const uint32_t rank = item % rankTiles * ranks;
    return {rank, AscendC::Std::min(ranks, epWorldSize_ - rank), expert, AscendC::Std::min(experts, end - expert),
            full};
}

template <RoceDispatchTypeClass>
__aicore__ inline RoceMaskWireTile MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GetCountWireTile(
    const RoceMaskCountTile& tile)
{
    const uint32_t row = UB_ALIGN + maskBytes_;
    const uint32_t cardBlocks = Ceil(moeExpertNumPerRank_ * row, SPLIT_BLOCK_DATA_SIZE);
    GM_ADDR card = windowGM_ + maskAreaOffset_ + static_cast<uint64_t>(tile.firstRank) * maskCardBytes_;
    const uint32_t offset = tile.fullCards ? 0U : tile.firstExpert * row;
    const uint32_t bytes = tile.fullCards ? tile.ranks * cardBlocks * SPLIT_BLOCK_DATA_SIZE : tile.experts * row;
    return {card + offset / SPLIT_BLOCK_DATA_SIZE * SERVER_STATE_ALIGN, nullptr, bytes,
            Ceil(offset % SPLIT_BLOCK_DATA_SIZE + bytes, SPLIT_BLOCK_DATA_SIZE)};
}

template <RoceDispatchTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::IssueCountTile(uint32_t item, uint32_t total)
{
    const uint32_t slot = item % countMaskBufferNum_;
    const bool ready = TestPrefetchedFlag(slot);
    if (!ready) {
        return false;
    }
    LocalTensor<uint32_t> rows = countMaskQueue_.AllocTensor<uint32_t>();
    ReadReceiveMaskTile(GetCountTile(item), rows);
    AscendC::SetFlag<HardEvent::MTE2_S>(payloadReadyEvents_[slot]);
    countMaskQueue_.EnQue(rows);
    if (item + countMaskBufferNum_ < total) {
        PrefetchMaskFlags(GetCountWireTile(GetCountTile(item + countMaskBufferNum_)), slot);
    }
    return true;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitCalCumSumBuffer()
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    tpipe_->InitBuffer(countMaskQueue_, countMaskBufferNum_, MASK_TILE_BLOCKS * SERVER_STATE_ALIGN);
    tpipe_->InitBuffer(flagBuf_, 32U * UB_ALIGN);
    InitPrefetchBuffers(countMaskBufferNum_, GetFlagCheckBufferSize(32U));
    tpipe_->InitBuffer(statusBuf_, Align32(rscvStatusNum_ * sizeof(int32_t)));
    tpipe_->InitBuffer(tokenNumBuf_, Align32(moeExpertNumPerRank_ * sizeof(int64_t)));
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);
    tpipe_->InitBuffer(sumLocalBuf_,
                       AscendC::Std::max(aivUsedCumSum_ * UB_ALIGN, GetFlagCheckBufferSize(aivUsedCumSum_)));
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    commFlagTensor_ = flagBuf_.Get<uint64_t>();
    statusTensor_ = statusBuf_.Get<int32_t>();
    Duplicate(commFlagTensor_, opSequence_, 32U * UB_ALIGN / sizeof(uint64_t));
}

template <RoceDispatchTypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GetCumSum()
{
    LocalTensor<uint64_t> rows = sumLocalBuf_.Get<uint64_t>();
    if (aivId_ == 0U) {
        return 0U;
    }
    WaitFlag<true>(countReadyGM_, rows, UB_ALIGN - sizeof(uint64_t), aivId_);
    GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(countReadyGM_));
    global.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopy(rows, global, aivId_ * UB_ALIGN / sizeof(uint64_t));
    SyncFunc<HardEvent::MTE2_S>();
    uint32_t base = 0U;
    LocalTensor<uint32_t> counts = rows.ReinterpretCast<uint32_t>();
    for (uint32_t core = 0U; core < aivId_; ++core) {
        base += counts.GetValue(core * UB_ALIGN / sizeof(uint32_t) + 2U);
    }
    return base;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalRecvAndSetFlag()
{
    uint32_t prefix = GetCumSum();
    for (uint32_t index = 0U; index < recStatusNumPerCore_; ++index) {
        prefix += statusTensor_.GetValue(index);
        statusTensor_.SetValue(index, prefix);
    }
    GlobalTensor<int32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    SyncFunc<HardEvent::S_MTE3>();
    if (recStatusNumPerCore_ != 0U) {
        DataCopyExtParams copy{1U, recStatusNumPerCore_ * uint32_t(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(output[startStatusIndex_], statusTensor_, copy);
        PipeBarrier<PIPE_MTE3>(); // Complete this core's prefix write before publishing ready.
    }
    if (aivId_ == 0U) {
        if (aivUsedCumSum_ > 1U) {
            SyncFunc<HardEvent::MTE3_MTE2>();
            WaitFlag<true>(statusDataSpaceGm_ + CUMSUM_FLAG_OFFSET + UB_ALIGN, sumLocalBuf_.Get<uint64_t>(),
                           UB_ALIGN - sizeof(uint64_t), aivUsedCumSum_ - 1U);
        }
        SetExpertTokenNums();
    }
    LocalTensor<uint64_t> ready = sumCoreBuf_.Get<uint64_t>();
    Duplicate(ready, opSequence_, aivNum_ * UB_ALIGN / sizeof(uint64_t));
    GlobalTensor<uint64_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(statusDataSpaceGm_ + CUMSUM_FLAG_OFFSET +
                                                             static_cast<uint64_t>(aivId_) * UB_ALIGN));
    SyncFunc<HardEvent::V_MTE3>();
    DataCopy(flags, ready,
             DataCopyParams{static_cast<uint16_t>(aivNum_), 1U, 0U, static_cast<uint16_t>(aivUsedCumSum_ - 1U)});
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SetExpertTokenNums()
{
    GlobalTensor<int32_t> prefixes;
    prefixes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    prefixes.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    SyncFunc<HardEvent::MTE3_MTE2>();
    DataCopyPad(statusTensor_, prefixes, DataCopyExtParams{1U, rscvStatusNum_ * uint32_t(sizeof(int32_t)), 0U, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_S>();
    LocalTensor<int64_t> expertCounts = tokenNumBuf_.Get<int64_t>();
    int32_t previous = 0;
    for (uint32_t expert = 0U; expert < moeExpertNumPerRank_; ++expert) {
        const int32_t inclusive = statusTensor_.GetValue((expert + 1U) * epWorldSize_ - 1U);
        expertCounts.SetValue(expert, expertTokenNumsType_ == 0U ? inclusive : inclusive - previous);
        previous = inclusive;
    }
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyPad(expertTokenNumsOutGMTensor_, expertCounts,
                DataCopyExtParams{1U, moeExpertNumPerRank_ * uint32_t(sizeof(int64_t)), 0U, 0U, 0U});
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalCumSum()
{
    InitCalCumSumBuffer();
    uint32_t first, end, count;
    SplitToCore(rankSizePerServer_, 0U, aivUsedCumSum_, first, end, count);
    ForwardMaskServer(serverId_, first, end);
    DistributeExpertMasks();
    CollectReceiveCounts();
    PipeBarrier<PIPE_MTE3>();
    countMaskQueue_.FreeAllEvent();
    ReleasePrefetchBuffers();
    CalRecvAndSetFlag();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::PrepareMaskServer(uint32_t server)
{
    GM_ADDR staging = GetRoceMaskSendAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(serverId_)), server);
    const uint32_t tileBytes = MASK_TILE_BLOCKS * SERVER_STATE_ALIGN;
    const uint32_t tiles = Ceil(maskSlotBytes_, tileBytes);
    for (uint32_t item = 0U; item < AscendC::Std::min(tiles, senderMaskBufferNum_); ++item) {
        IssueSenderMask(server, static_cast<uint64_t>(item) * tileBytes);
    }
    for (uint32_t item = 0U; item < tiles; ++item) {
        const uint64_t offset = static_cast<uint64_t>(item) * tileBytes;
        const uint32_t bytes = AscendC::Std::min<uint64_t>(maskSlotBytes_ - offset, tileBytes);
        LocalTensor<uint8_t> wire = senderMaskQueue_.DeQue<uint8_t>();
        GlobalTensor<uint8_t> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(staging + offset));
        DataCopy(target, wire, bytes);
        senderMaskQueue_.FreeTensor(wire);
        if (item + senderMaskBufferNum_ < tiles) {
            IssueSenderMask(server, static_cast<uint64_t>(item + senderMaskBufferNum_) * tileBytes);
        }
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ReadMaskRows(GM_ADDR wire, uint64_t offset,
                                                                                       uint32_t bytes,
                                                                                       LocalTensor<uint8_t>& rows)
{
    uint32_t copied = 0U;
    while (copied < bytes) {
        const uint64_t position = offset + copied;
        const uint32_t within = position % SPLIT_BLOCK_DATA_SIZE;
        const uint32_t full = within == 0U ? (bytes - copied) / SPLIT_BLOCK_DATA_SIZE : 0U;
        const uint32_t part =
            full ? SPLIT_BLOCK_DATA_SIZE : AscendC::Std::min<uint32_t>(bytes - copied, SPLIT_BLOCK_DATA_SIZE - within);
        const uint32_t blocks = full ? full : 1U;
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(wire + position / SPLIT_BLOCK_DATA_SIZE * SERVER_STATE_ALIGN + within));
        source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyPad(rows[copied], source, DataCopyExtParams{static_cast<uint16_t>(blocks), part, UB_ALIGN, 0U, 0U},
                    DataCopyPadExtParams<uint8_t>{false, 0U, 0U, 0U});
        copied += blocks * part;
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ForwardMaskTile(const RoceMaskWireTile& tile)
{
    LocalTensor<uint8_t> wire = countMaskQueue_.DeQue<uint8_t>();
    GlobalTensor<uint8_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(tile.target));
    output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopy(output, wire, tile.bytes);
    countMaskQueue_.FreeTensor(wire);
}

template <RoceDispatchTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::IssueForwardMask(uint32_t server,
                                                                                           uint32_t firstRank,
                                                                                           uint32_t item,
                                                                                           uint32_t total)
{
    const bool local = server == static_cast<uint32_t>(serverId_);
    const uint32_t slot = item % countMaskBufferNum_;
    if (!local) {
        const bool ready = TestPrefetchedFlag(slot);
        if (!ready) {
            return false;
        }
    }
    const auto tile = GetForwardMaskTile(server, firstRank, item);
    LocalTensor<uint8_t> wire = countMaskQueue_.AllocTensor<uint8_t>();
    GlobalTensor<uint8_t> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(tile.source));
    input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopy(wire, input, tile.bytes);
    countMaskQueue_.EnQue(wire);
    if (!local && item + countMaskBufferNum_ < total) {
        PrefetchMaskFlags(GetForwardMaskTile(server, firstRank, item + countMaskBufferNum_), slot);
    }
    return true;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ForwardMaskServer(uint32_t server,
                                                                                            uint32_t firstRank,
                                                                                            uint32_t endRank)
{
    const uint32_t total = (endRank - firstRank) * Ceil(maskCardBytes_, MASK_TILE_BLOCKS * SERVER_STATE_ALIGN);
    const bool local = server == static_cast<uint32_t>(serverId_);
    for (uint32_t item = 0U; !local && item < AscendC::Std::min(total, countMaskBufferNum_); ++item) {
        PrefetchMaskFlags(GetForwardMaskTile(server, firstRank, item), item);
    }
    uint32_t issued = 0U;
    for (uint32_t done = 0U; done < total; ++done) {
        do {
            while (issued < total && issued < done + countMaskBufferNum_ &&
                   IssueForwardMask(server, firstRank, issued, total)) {
                ++issued;
            }
        } while (issued == done);
        ForwardMaskTile(GetForwardMaskTile(server, firstRank, done));
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SplitRemoteMaskWork(uint32_t& firstServer,
                                                                                              uint32_t& endServer,
                                                                                              uint32_t& firstRank,
                                                                                              uint32_t& endRank)
{
    const uint32_t remoteServers = serverNum_ - 1U; // Caller excludes the single-server case.
    uint32_t count;
    if (remoteServers > aivUsedCumSum_) {
        SplitToCore(remoteServers, 0U, aivUsedCumSum_, firstServer, endServer, count);
        firstRank = 0U;
        endRank = rankSizePerServer_;
        return;
    }
    const uint32_t coresPerServer = aivUsedCumSum_ / remoteServers;
    const uint32_t extra = aivUsedCumSum_ % remoteServers;
    const uint32_t wideCores = (coresPerServer + 1U) * extra;
    firstServer = aivId_ < wideCores ? aivId_ / (coresPerServer + 1U) : extra + (aivId_ - wideCores) / coresPerServer;
    endServer = firstServer + 1U;
    const uint32_t firstCore = firstServer * coresPerServer + AscendC::Std::min(firstServer, extra);
    const uint32_t groupCores = coresPerServer + (firstServer < extra ? 1U : 0U);
    SplitToCore(rankSizePerServer_, firstCore, groupCores, firstRank, endRank, count);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::DistributeExpertMasks()
{
    if (serverNum_ <= 1U) {
        return;
    }
    uint32_t first, end, firstRank, endRank;
    SplitRemoteMaskWork(first, end, firstRank, endRank);
    if (firstRank == endRank) {
        return;
    }
    const uint32_t local = static_cast<uint32_t>(serverId_);
    for (uint32_t index = first; index < end; ++index) {
        const uint32_t server = index < local ? index : index + 1U;
        ForwardMaskServer(server, firstRank, endRank);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ReadReceiveCounts(
    const RoceMaskCountTile& tile, LocalTensor<uint32_t>& rows)
{
    uint32_t total = 0U;
    for (uint32_t expert = 0U; expert < tile.experts; ++expert) {
        for (uint32_t rank = 0U; rank < tile.ranks; ++rank) {
            const uint32_t pair = (tile.firstExpert + expert) * epWorldSize_ + tile.firstRank + rank;
            if (pair < startStatusIndex_ || pair >= endStatusIndex_) {
                continue;
            }
            const uint32_t offset = (expert * tile.ranks + rank) * UB_ALIGN;
            const uint32_t selected = rows.GetValue(offset / sizeof(uint32_t) + MASK_COUNT_WORD);
            statusTensor_.SetValue(pair - startStatusIndex_, selected);
            total += selected;
        }
    }
    countMaskQueue_.FreeTensor(rows);
    return total;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ReadReceiveMaskTile(
    const RoceMaskCountTile& tile, LocalTensor<uint32_t>& rows)
{
    const uint32_t row = UB_ALIGN + maskBytes_;
    // Keep the existing full mask flag gate: Copy relies on CumSum ready for bitmap visibility.
    for (uint32_t expert = 0U; expert < tile.experts; ++expert) {
        const uint32_t base = (tile.firstExpert + expert) * epWorldSize_;
        const uint32_t first =
            AscendC::Std::max(tile.firstRank, startStatusIndex_ > base ? startStatusIndex_ - base : 0U);
        const uint32_t end = AscendC::Std::min(tile.firstRank + tile.ranks, endStatusIndex_ - base);
        if (first >= end) {
            continue;
        }
        const uint64_t offset = static_cast<uint64_t>(tile.firstExpert + expert) * row;
        GlobalTensor<uint32_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
            windowGM_ + maskAreaOffset_ + static_cast<uint64_t>(first) * maskCardBytes_ +
            offset / SPLIT_BLOCK_DATA_SIZE * SERVER_STATE_ALIGN + offset % SPLIT_BLOCK_DATA_SIZE));
        source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t target = (expert * tile.ranks + first - tile.firstRank) * UB_ALIGN / sizeof(uint32_t);
        DataCopyPad(rows[target], source,
                    DataCopyExtParams{static_cast<uint16_t>(end - first), UB_ALIGN,
                                      static_cast<uint32_t>(maskCardBytes_ - UB_ALIGN), 0U, 0U},
                    DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::PublishCountReady(uint32_t total)
{
    LocalTensor<uint64_t> ready = scalarBuf_.Get<uint64_t>();
    Duplicate(ready, opSequence_, UB_ALIGN / sizeof(uint64_t));
    SyncFunc<HardEvent::V_S>();
    ready.ReinterpretCast<uint32_t>().SetValue(2, total);
    GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint64_t*>(countReadyGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN));
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(global, ready, UB_ALIGN / sizeof(uint64_t));
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CollectReceiveCounts()
{
    SplitToCore(rscvStatusNum_, 0U, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_);
    if (recStatusNumPerCore_ == 0U) {
        PublishCountReady(0U);
        return;
    }
    const auto first = GetCountTile(0U);
    const uint32_t expertTiles = Ceil(Ceil(endStatusIndex_, epWorldSize_) - first.firstExpert, first.experts);
    const uint32_t total = expertTiles * Ceil(epWorldSize_, first.ranks);
    for (uint32_t item = 0U; item < AscendC::Std::min(total, countMaskBufferNum_); ++item) {
        PrefetchMaskFlags(GetCountWireTile(GetCountTile(item)), item);
    }
    uint32_t issued = 0U;
    uint32_t sum = 0U;
    for (uint32_t done = 0U; done < total; ++done) {
        do {
            while (issued < total && issued < done + countMaskBufferNum_ && IssueCountTile(issued, total)) {
                ++issued;
            }
        } while (issued == done);
        LocalTensor<uint32_t> rows = countMaskQueue_.DeQue<uint32_t>();
        AscendC::WaitFlag<HardEvent::MTE2_S>(payloadReadyEvents_[done % countMaskBufferNum_]);
        sum += ReadReceiveCounts(GetCountTile(done), rows);
    }
    PublishCountReady(sum);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::FillQuadruple(
    LocalTensor<XCommInType>& xOutTensor, uint32_t tokenIndex)
{
    GlobalTensor<int32_t> routes;
    routes.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(tokenRouteGM_ + static_cast<uint64_t>(tokenIndex) * tokenRouteBytes_));
    routes.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    LocalTensor<int32_t> metadata = xOutTensor.template ReinterpretCast<int32_t>();
    DataCopyExtParams routeCopy{1U, tokenRouteBytes_, 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> routePad{false, 0U, 0U, 0U};
    DataCopyPad(metadata[serverTripleOffset_], routes, routeCopy, routePad);
    DataCopyExtParams scaleCopy{1U, static_cast<uint32_t>(axisK_ * sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> scalePad{false, 0U, 0U, 0U};
    DataCopyPad(xOutTensor.template ReinterpretCast<float>()[serverExpertScalesOffset_],
                expertScalesGMTensor_[static_cast<uint64_t>(tokenIndex) * axisK_], scaleCopy, scalePad);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::FillServerHeader(LocalTensor<uint64_t> header,
                                                                                           uint32_t elements)
{
    Duplicate(header, opSequence_, elements);
    SyncFunc<HardEvent::V_S>();
    LocalTensor<uint32_t> fields = header.ReinterpretCast<uint32_t>();
    fields.SetValue(0, axisBS_);
    fields.SetValue(1, activeMaskBsCnt_);
    fields.SetValue(2, 1U);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::PublishLocalServerHeader(
    LocalTensor<uint64_t> header)
{
    FillServerHeader(header, SERVER_STATE_ALIGN / sizeof(uint64_t));
    GM_ADDR base = GetRoceDispatchAddr(GetRoceBaseByServerId(serverId_));
    GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(GetRoceReceiveAddr(base, serverId_)));
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(output, header, SERVER_STATE_ALIGN / sizeof(uint64_t));
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitSenderBuffers(TBuf<>& batchBuf)
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    tpipe_->InitBuffer(senderMaskQueue_, senderMaskBufferNum_, MASK_TILE_BLOCKS * SERVER_STATE_ALIGN);
    tpipe_->InitBuffer(batchBuf, 3U * ROCE_WQEBB_BYTES);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SubmitServerBatch(
    uint32_t server, GM_ADDR source, LocalTensor<uint8_t> storage)
{
    const uint32_t localServer = static_cast<uint32_t>(serverId_);
    GM_ADDR base = GetRoceDispatchAddr(GetRoceBaseByServerId(localServer));
    GM_ADDR remote = GetRoceDispatchAddr(GetRoceBaseByServerId(server));
    GM_ADDR target = GetRoceReceiveAddr(remote, localServer);
    BufDesc mask = {GetRoceMaskSendAddr(base, server), static_cast<uint32_t>(maskSlotBytes_)};
    BufDesc tokens = {source, static_cast<uint32_t>(SERVER_STATE_ALIGN +
                                                    static_cast<uint64_t>(activeMaskBsCnt_) * serverRecordAlignLen_)};
    auto batch = hcomm_.MakeBatchHandle(ctx_.GetRoceHandleByServerId(server, localServer, ROCE_CHAN_TOKEN), storage,
                                        2U * ROCE_WQEBB_BYTES, remote, base);
    hcomm_.WriteNbi(batch, GetRoceMaskReceiveAddr(remote, localServer), &mask, 1U);
    hcomm_.WriteNbi(batch, target, &tokens, 1U);
    hcomm_.BatchCommit(batch);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::HcclTokenDispatch()
{
    TBuf<> batchBuf;
    InitSenderBuffers(batchBuf);
    uint32_t first, end, count;
    SplitToCore(serverNum_, aivUsedCumSum_, aivUsedRemotComm_, first, end, count);
    const uint32_t localServer = static_cast<uint32_t>(serverId_);
    GM_ADDR base = GetRoceDispatchAddr(GetRoceBaseByServerId(localServer));
    GM_ADDR source = GetRoceReceiveAddr(base, localServer);
    LocalTensor<uint8_t> storage = batchBuf.Get<uint8_t>();
    const uint32_t padding =
        (ROCE_WQEBB_BYTES - reinterpret_cast<uintptr_t>(storage.GetPhyAddr()) % ROCE_WQEBB_BYTES) % ROCE_WQEBB_BYTES;
    for (uint32_t server = first; server < end; ++server) {
        if (server == localServer) {
            continue; // Local Gather already owns the metadata and prepared tokens after SyncAll.
        }
        PrepareMaskServer(server);
        PipeBarrier<PIPE_MTE3>();
        SubmitServerBatch(server, source, storage[padding]);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CalServerSplitCore(
    uint32_t serverNum, uint32_t startAivId, uint32_t useAivNum, uint32_t& startServerId, uint32_t& endServerId,
    uint32_t& useAivPerServer, uint32_t& firstAivId)
{
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
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitPrefetchBuffers(uint32_t depth,
                                                                                              uint32_t slotBytes)
{
    flagPrefetchDepth_ = depth;
    flagSlotWords_ = slotBytes / sizeof(uint64_t);
    tpipe_->InitBuffer(flagPrefetchBuf_, depth * slotBytes);
    for (uint32_t slot = 0U; slot < depth; ++slot) {
        flagReadyEvents_[slot] = tpipe_->AllocEventID<HardEvent::MTE2_V>();
        flagFreeEvents_[slot] = tpipe_->AllocEventID<HardEvent::V_MTE2>();
        AscendC::SetFlag<HardEvent::V_MTE2>(flagFreeEvents_[slot]);
    }
    for (uint32_t slot = 0U; slot < depth; ++slot) {
        flagPending_[slot] = false;
    }
    for (uint32_t slot = 0U; slot < depth; ++slot) {
        payloadReadyEvents_[slot] = tpipe_->AllocEventID<HardEvent::MTE2_S>();
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ReleasePrefetchBuffers()
{
    for (uint32_t slot = 0U; slot < flagPrefetchDepth_; ++slot) {
        if (flagPending_[slot]) {
            AscendC::WaitFlag<HardEvent::MTE2_V>(flagReadyEvents_[slot]);
            AscendC::SetFlag<HardEvent::V_MTE2>(flagFreeEvents_[slot]);
        }
    }
    for (uint32_t slot = 0U; slot < flagPrefetchDepth_; ++slot) {
        AscendC::WaitFlag<HardEvent::V_MTE2>(flagFreeEvents_[slot]);
        tpipe_->ReleaseEventID<HardEvent::MTE2_V>(flagReadyEvents_[slot]);
        tpipe_->ReleaseEventID<HardEvent::V_MTE2>(flagFreeEvents_[slot]);
    }
    for (uint32_t slot = 0U; slot < flagPrefetchDepth_; ++slot) {
        tpipe_->ReleaseEventID<HardEvent::MTE2_S>(payloadReadyEvents_[slot]);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SubmitPrefetchedFlag(uint32_t slot,
                                                                                               GM_ADDR address,
                                                                                               uint32_t stride,
                                                                                               uint32_t count)
{
    flagAddresses_[slot] = address;
    flagStrides_[slot] = stride;
    flagCounts_[slot] = count;
    ReloadPrefetchedFlag(slot);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ReloadPrefetchedFlag(uint32_t slot)
{
    AscendC::WaitFlag<HardEvent::V_MTE2>(flagFreeEvents_[slot]);
    GlobalTensor<uint64_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(flagAddresses_[slot]));
    source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    const uint32_t blocks = flagStrides_[slot] == 0U ? 1U : flagCounts_[slot];
    const uint32_t bytes = flagStrides_[slot] == 0U ? flagCounts_[slot] * sizeof(uint64_t) : sizeof(uint64_t);
    LocalTensor<uint64_t> destination = flagPrefetchBuf_.Get<uint64_t>()[slot * flagSlotWords_];
    DataCopyPadExtParams<uint64_t> pad{false, 0U, 0U, 0U};
    DataCopyPad<uint64_t, PaddingMode::Compact>(
        destination, source, DataCopyExtParams{static_cast<uint16_t>(blocks), bytes, flagStrides_[slot], 0U, 0U}, pad);
    AscendC::SetFlag<HardEvent::MTE2_V>(flagReadyEvents_[slot]);
    flagPending_[slot] = true;
}

template <RoceDispatchTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::TestPrefetchedFlag(uint32_t slot)
{
    AscendC::WaitFlag<HardEvent::MTE2_V>(flagReadyEvents_[slot]);
    LocalTensor<uint64_t> flags = flagPrefetchBuf_.Get<uint64_t>()[slot * flagSlotWords_];
    const bool ready = CheckFlagBySequence(flags, flagCounts_[slot]);
    AscendC::SetFlag<HardEvent::V_MTE2>(flagFreeEvents_[slot]);
    flagPending_[slot] = false;
    if (!ready) {
        ReloadPrefetchedFlag(slot);
    }
    return ready;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::IssueQuantInput(uint32_t token, uint32_t slot)
{
    AscendC::WaitFlag<HardEvent::V_MTE2>(quantFreeEvents_[slot]);
    LocalTensor<XInType> input;
    ReadTokenData(token, input);
    LocalTensor<XCommInType> plain =
        tokenDataBuf_.Get<XCommInType>()[slot * serverRecordAlignLen_ / sizeof(XCommInType)];
    FillQuadruple(plain, token);
    if constexpr (QuantMode == UNQUANT && IsSmoothScaleExist) {
        DataCopyPadExtParams<uint8_t> pad{true, 0U, 0U, 0U};
        DataCopyExtParams copy{1U, scaleInBytes_, 0U, 0U, 0U};
        LocalTensor<uint8_t> scale = plain.template ReinterpretCast<uint8_t>()[Align32(hOutSize_)];
        DataCopyPad(scale,
                    scalesGMTensor_.template ReinterpretCast<uint8_t>()[static_cast<uint64_t>(token) * scaleInBytes_],
                    copy, pad);
    }
    AscendC::SetFlag<HardEvent::MTE2_V>(quantReadyEvents_[slot]);
}

template <RoceDispatchTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::IssueGatherInput(
    GM_ADDR source, uint32_t index, uint32_t total, uint32_t first, uint32_t stride, bool local)
{
    const uint32_t slot = index % gatherInputBufferNum_;
    if (!local && !TestPrefetchedFlag(slot)) {
        return false;
    }
    GM_ADDR token = source + static_cast<uint64_t>(first + index * stride) * serverRecordAlignLen_;
    LocalTensor<XCommInType> input = tokenInQueue_.AllocTensor<XCommInType>();
    ReadDataFromWindows<XCommInType, true, true>(token, serverRecordBlockCnt_, input);
    tokenInQueue_.EnQue(input);
    AscendC::SetFlag<HardEvent::MTE2_S>(payloadReadyEvents_[slot]);
    const uint32_t next = index + gatherInputBufferNum_;
    if (!local && next < total) {
        GM_ADDR flags = source + static_cast<uint64_t>(first + next * stride) * serverRecordAlignLen_;
        SubmitPrefetchedFlag(slot, flags + SPLIT_BLOCK_DATA_SIZE, COMM_FLAG_COPY_STRIDE, serverRecordBlockCnt_);
    }
    return true;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitQuantBuffers()
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    // TPipe::Reset restores CTRL[60] to 1; quantization requires per-instruction saturation.
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_->InitBuffer(xInQueue_, quantInputBufferNum_, Align32(hAlignSize_));
    tpipe_->InitBuffer(tokenDataBuf_, quantInputBufferNum_ * serverRecordAlignLen_);
    tpipe_->InitBuffer(xTempQueue_, quantOutputBufferNum_, serverRecordAlignLen_);
    for (uint32_t slot = 0U; slot < quantInputBufferNum_; ++slot) {
        quantReadyEvents_[slot] = tpipe_->AllocEventID<HardEvent::MTE2_V>();
        quantFreeEvents_[slot] = tpipe_->AllocEventID<HardEvent::V_MTE2>();
        AscendC::SetFlag<HardEvent::V_MTE2>(quantFreeEvents_[slot]);
    }
    for (uint32_t bufferIndex = 0U; bufferIndex < quantInputBufferNum_; ++bufferIndex) {
        InitQuantScratch(bufferIndex);
    }
    const uint32_t flagBytes = Ceil(serverRecordBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(flagBuf_, flagBytes);
    commFlagTensor_ = flagBuf_.Get<uint64_t>();
    Duplicate(commFlagTensor_, opSequence_, flagBytes / sizeof(uint64_t));
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ReadTokenData(uint32_t tokenIndex,
                                                                                        LocalTensor<XInType>& tokenData)
{
    tokenData = xInQueue_.AllocTensor<XInType>();
    if constexpr (QuantMode == MX_QUANT || QuantMode == MX_QUANT_CLIP || QuantMode == PERGROUP_DYNAMIC_QUANT) {
        QuantCleanTensor(tokenData);
    }
    if constexpr (Std::IsSame<XInType, fp8_e5m2_t>::value || Std::IsSame<XInType, fp8_e4m3fn_t>::value ||
                  Std::IsSame<XInType, hifloat8_t>::value) {
        DataCopyPadParams pad{true, 0U, 0U, 0U};
        DataCopyParams copy{1U, static_cast<uint16_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U};
        DataCopyPad(tokenData, xGMTensor_[static_cast<uint64_t>(tokenIndex) * copyInAxisH_], copy, pad);
    } else {
        DataCopyPadExtParams<XInType> pad{true, 0U, 0U, 0U};
        DataCopyExtParams copy{1U, static_cast<uint32_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U, 0U};
        DataCopyPad(tokenData, xGMTensor_[static_cast<uint64_t>(tokenIndex) * copyInAxisH_], copy, pad);
    }
    xInQueue_.EnQue(tokenData);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::QuantToken(
    uint32_t tokenIndex, LocalTensor<XInType>& xInTensor, LocalTensor<XCommInType>& xOutTensor, uint32_t bufferIndex)
{
    LocalTensor<float> floatLocalTemp;
    if constexpr (QuantMode > UNQUANT || !Std::IsSame<ExpandXOutType, XType>::value) {
        floatLocalTemp = receiveDataCastFloatBuf_[bufferIndex].Get<float>();
        LocalTensor<float> smoothScalesTensor = smoothScalesBuf_[bufferIndex].Get<float>();
        quantInst_.SetQuantInitParams(floatLocalTemp, smoothScalesTensor, smoothScalesBuf_[bufferIndex],
                                      dynamicScalesOutGMTensor_);
    }
    xOutTensor = tokenDataBuf_.Get<XCommInType>()[bufferIndex * serverRecordAlignLen_ / sizeof(XCommInType)];
    if constexpr (QuantMode > UNQUANT) {
        quantInst_.QuantProcess(xOutTensor, xInTensor, 0U, scalesCount_, scalesGMTensor_);
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            SyncFunc<HardEvent::S_V>();
        }
    } else if constexpr (!Std::IsSame<ExpandXOutType, XType>::value) {
        Cast(floatLocalTemp, xInTensor, RoundMode::CAST_NONE, copyInAxisH_);
        Cast(xOutTensor, floatLocalTemp, RoundMode::CAST_ROUND, copyInAxisH_);
    } else {
        DataCopy(xOutTensor, xInTensor, Align32(copyInAxisH_ * sizeof(XInType)) / sizeof(XCommInType));
    }
    xInQueue_.FreeTensor(xInTensor);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WriteToken(
    uint32_t tokenIndex, LocalTensor<XCommInType>& xOutTensor)
{
    GM_ADDR sourceSlot =
        GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(static_cast<uint32_t>(serverId_))),
                           static_cast<uint32_t>(serverId_)) +
        SERVER_TOKEN_AREA_OFFSET + static_cast<uint64_t>(tokenIndex) * serverRecordAlignLen_;
    LocalTensor<XCommInType> flagged = xTempQueue_.AllocTensor<XCommInType>();
    WriteDataToWinOut<XCommInType, true>(xOutTensor, flagged, sourceSlot, serverRecordBlockCnt_);
    xTempQueue_.FreeTensor(flagged);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::StoreTokenRoutes(uint32_t startTokenId,
                                                                                           uint32_t endTokenId)
{
    LocalTensor<int32_t> route = subExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> offsets = dstExpBuf_.Get<int32_t>();
    GlobalTensor<int32_t> routes;
    routes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenRouteGM_));
    SyncFunc<HardEvent::V_S>();
    const uint32_t rowElements = tokenRouteBytes_ / sizeof(int32_t);
    for (uint32_t tokenId = startTokenId; tokenId < endTokenId; ++tokenId) {
        LocalTensor<int32_t> row = route[(tokenId - startTokenId) * rowElements];
        CreateVecIndex(offsets, static_cast<int32_t>(tokenId * axisK_), axisK_);
        Muls(offsets, offsets, static_cast<int32_t>(sizeof(int32_t)), axisK_);
        Gather(row[UB_ALIGN / sizeof(int32_t)], expertIdsTensor_, offsets.ReinterpretCast<uint32_t>(), 0U, axisK_);
        row.SetValue(0, epRankId_);
        row.SetValue(1, tokenId);
    }
    SyncFunc<HardEvent::V_MTE3>();
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(routes[static_cast<uint64_t>(startTokenId) * rowElements], route,
             (endTokenId - startTokenId) * rowElements);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::QuantAllTokens()
{
    uint32_t first, end, count;
    SplitToCore(activeMaskBsCnt_, 0U, aivNum_, first, end, count);
    if (count == 0U) {
        PipeBarrier<PIPE_ALL>();
        if (aivId_ == 0U) {
            PublishLocalServerHeader(subExpBuf_.Get<uint64_t>());
            PipeBarrier<PIPE_ALL>();
        }
        return;
    }
    StoreTokenRoutes(first, end);
    InitQuantBuffers();
    for (uint32_t item = 0U; item < AscendC::Std::min(count, quantInputBufferNum_); ++item) {
        IssueQuantInput(first + item, item);
    }
    for (uint32_t item = 0U; item < count; ++item) {
        const uint32_t slot = item % quantInputBufferNum_;
        LocalTensor<XInType> input = xInQueue_.DeQue<XInType>();
        AscendC::WaitFlag<HardEvent::MTE2_V>(quantReadyEvents_[slot]);
        LocalTensor<XCommInType> output;
        QuantToken(first + item, input, output, slot);
        WriteToken(first + item, output);
        AscendC::SetFlag<HardEvent::V_MTE2>(quantFreeEvents_[slot]);
        if (item + quantInputBufferNum_ < count) {
            IssueQuantInput(first + item + quantInputBufferNum_, slot);
        }
    }
    if (aivId_ == 0U) {
        PublishLocalServerHeader(tokenDataBuf_.Get<uint64_t>());
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t slot = 0U; slot < quantInputBufferNum_; ++slot) {
        AscendC::WaitFlag<HardEvent::V_MTE2>(quantFreeEvents_[slot]);
        tpipe_->ReleaseEventID<HardEvent::MTE2_V>(quantReadyEvents_[slot]);
        tpipe_->ReleaseEventID<HardEvent::V_MTE2>(quantFreeEvents_[slot]);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitQuantScratch(uint32_t bufferIndex)
{
    if constexpr (QuantMode > UNQUANT || !Std::IsSame<ExpandXOutType, XType>::value) {
        const uint32_t scratchBytes = Align32(axisH_ * sizeof(float));
        tpipe_->InitBuffer(receiveDataCastFloatBuf_[bufferIndex], scratchBytes);
        tpipe_->InitBuffer(smoothScalesBuf_[bufferIndex], scratchBytes);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitGatherTokenBuffer()
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    const uint32_t flagCheckSize = GetFlagCheckBufferSize(serverRecordBlockCnt_);
    const uint32_t flagBytes = Ceil(maxBlockCnt_ * UB_ALIGN, SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(tokenInQueue_, gatherInputBufferNum_, serverRecordAlignLen_);
    tpipe_->InitBuffer(tokenDataBuf_, gatherInputBufferNum_ * hCommuSize_);
    tpipe_->InitBuffer(gatherPacketBuf_, gatherPacketBufferNum_ * hCommuSize_);
    gatherPacketSlot_ = 0U;
    for (uint32_t slot = 0U; slot < gatherPacketBufferNum_; ++slot) {
        gatherPacketFree_[slot] = tpipe_->AllocEventID<HardEvent::MTE3_V>();
        SetFlag<HardEvent::MTE3_V>(gatherPacketFree_[slot]);
    }
    tpipe_->InitBuffer(flagBuf_, flagBytes);
    InitPrefetchBuffers(gatherInputBufferNum_, flagCheckSize);
    const uint32_t minGroupCores = AscendC::Std::max(1U, aivUsedGather_ / (serverNum_ - 1U));
    tpipe_->InitBuffer(combineBitmapBuf_, Ceil(combineBitmapTokenNum_, minGroupCores) * UB_ALIGN);
    commFlagTensor_ = flagBuf_.Get<uint64_t>();
    combineBitmapTensor_ = combineBitmapBuf_.Get<uint32_t>();
    Duplicate<uint64_t>(commFlagTensor_, opSequence_, flagBytes / sizeof(uint64_t));
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::PrefetchGatherHeader(uint32_t server)
{
    GM_ADDR header = GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(serverId_)), server);
    SubmitPrefetchedFlag(0U, header + SPLIT_BLOCK_DATA_SIZE, 0U, 1U);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitStatusFlag(uint32_t serverIdx,
                                                                                         TBuf<>& serverFlagBuf,
                                                                                         uint32_t& rcvTokenCnt)
{
    if (serverIdx == static_cast<uint32_t>(serverId_)) {
        FillServerHeader(serverFlagBuf.Get<uint64_t>(), UB_ALIGN / sizeof(uint64_t));
        rcvTokenCnt = activeMaskBsCnt_;
        return;
    }
    LocalTensor<uint32_t> tokenCntTensor;
    GlobalTensor<uint32_t> statusCntGlobal;
    // 源server写入本卡RoCE接收槽的头块（本卡注册区，按source server索引）
    GM_ADDR wAddr =
        GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(static_cast<uint32_t>(serverId_))), serverIdx);
    statusCntGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(wAddr));
    tokenCntTensor = serverFlagBuf.Get<uint32_t>();
    while (!TestPrefetchedFlag(0U)) {
    }
    DataCopy(tokenCntTensor, statusCntGlobal, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    rcvTokenCnt = tokenCntTensor.GetValue(1);
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
template <typename T, bool disableL2, bool queueOutput>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WriteDataToWinOut(LocalTensor<T>& xOutTensor,
                                                                                            LocalTensor<T>& xTempTensor,
                                                                                            GM_ADDR rankGM,
                                                                                            uint32_t blockCnt)
{
    DataCopyParams insertDataParams = {static_cast<uint16_t>(blockCnt), COMM_COPY_BLOCK, 0U, INSERT_DATA_STRIDE};
    DataCopyParams insertFlagParams = {static_cast<uint16_t>(blockCnt), FLAG_COPY_BLOCK, 0U, INSERT_FLAG_STRIDE};
    LocalTensor<uint64_t> xFlagTensor = xTempTensor.template ReinterpretCast<uint64_t>();
    // A5 的 UB 到 UB 搬运走 V 管线。
    DataCopy(xTempTensor, xOutTensor, insertDataParams);
    DataCopy(xFlagTensor[SPLIT_BLOCK_DATA_SIZE / sizeof(uint64_t)], commFlagTensor_, insertFlagParams);
    if constexpr (queueOutput) {
        xTempQueue_.EnQue(xTempTensor);
        xTempTensor = xTempQueue_.DeQue<T>();
    } else {
        SyncFunc<AscendC::HardEvent::V_MTE3>();
    }
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
    LocalTensor<XCommInType>& xInTensor, GM_ADDR rankGM, uint32_t topKIndex, float expertScale)
{
    AscendC::WaitFlag<HardEvent::MTE3_V>(gatherPacketFree_[gatherPacketSlot_]);
    LocalTensor<XCommInType> flagged =
        gatherPacketBuf_.Get<XCommInType>()[gatherPacketSlot_ * hCommuSize_ / sizeof(XCommInType)];
    FillKAndScale(xInTensor, topKIndex, expertScale);
    WriteDataToWinOut<XCommInType, true, false>(xInTensor, flagged, rankGM, blockCntPerTokenToExpert_);
    SetFlag<HardEvent::MTE3_V>(gatherPacketFree_[gatherPacketSlot_]);
    if (++gatherPacketSlot_ == gatherPacketBufferNum_) {
        gatherPacketSlot_ = 0U;
    }
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
    LocalTensor<uint8_t> singleByteTok = tokenData.template ReinterpretCast<uint8_t>();
    Duplicate(singleByteTok, QUANT_PADDING_VALUE, hAlignSize_);
    SyncFunc<AscendC::HardEvent::V_MTE2>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::StoreTokenBitmap(uint32_t serverIdx,
                                                                                           uint32_t tokenId,
                                                                                           uint32_t bitmap,
                                                                                           uint32_t tokenStride)
{
    const uint64_t offset = static_cast<uint64_t>(serverIdx) * (combineBitmapTokenNum_ + UB_ALIGN / sizeof(uint32_t)) +
                            UB_ALIGN / sizeof(uint32_t) + tokenId;
    const uint32_t source = tokenId / tokenStride * (UB_ALIGN / sizeof(uint32_t));
    combineBitmapTensor_.SetValue(source, bitmap);
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyPad(combineServerInfoGMTensor_[offset], combineBitmapTensor_[source],
                DataCopyExtParams{1U, uint32_t(sizeof(uint32_t)), 0U, 0U, 0U});
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SendToExpert(
    LocalTensor<XCommInType>& recvTmpTensor, uint32_t serverIdx, uint32_t tokenStride, uint32_t bufferIndex)
{
    LocalTensor<XCommInType> tokenData;
    LocalTensor<uint32_t> routing = recvTmpTensor.template ReinterpretCast<uint32_t>();
    LocalTensor<float> weights = recvTmpTensor.template ReinterpretCast<float>();
    const uint32_t sourceRank = routing.GetValue(serverTripleOffset_);
    const uint32_t tokenId = routing.GetValue(serverTripleOffset_ + 1U);
    uint32_t bitmap = 0U;
    for (uint32_t k = 0U; k < axisK_; ++k) {
        const uint32_t expertId = routing.GetValue(serverExpertIdsOffset_ + k);
        if (expertId >= moeExpertNum_) {
            continue;
        }
        const uint32_t targetRank = expertId / moeExpertNumPerRank_;
        if (!IsInSameServer(targetRank)) {
            continue;
        }
        if (bitmap == 0U) {
            tokenData = tokenDataBuf_.Get<XCommInType>()[bufferIndex * hCommuSize_ / sizeof(XCommInType)];
            DataCopy(tokenData, recvTmpTensor, hScaleIdxSize_ / sizeof(XCommInType));
        }
        bitmap |= 1U << k;
        const uint64_t pair =
            static_cast<uint64_t>(sourceRank) * moeExpertNumPerRank_ + expertId % moeExpertNumPerRank_;
        GM_ADDR target = GetWindAddrByRankId(targetRank) + (pair * axisMaxBS_ + tokenId) * expertTokenStride_;
        ProcessToken(tokenData, target, k, weights.GetValue(serverExpertScalesOffset_ + k));
    }
    StoreTokenBitmap(serverIdx, tokenId, bitmap, tokenStride);
    tokenInQueue_.FreeTensor(recvTmpTensor);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitToken(uint32_t tokenCnt,
                                                                                    uint32_t serverIdx,
                                                                                    uint32_t startTokenIdx,
                                                                                    uint32_t tokenStride)
{
    GM_ADDR source =
        GetRoceReceiveAddr(GetRoceDispatchAddr(GetRoceBaseByServerId(serverId_)), serverIdx) + SERVER_TOKEN_AREA_OFFSET;
    const bool local = serverIdx == static_cast<uint32_t>(serverId_);
    for (uint32_t item = 0U; !local && item < AscendC::Std::min(tokenCnt, gatherInputBufferNum_); ++item) {
        GM_ADDR token = source + static_cast<uint64_t>(startTokenIdx + item * tokenStride) * serverRecordAlignLen_;
        SubmitPrefetchedFlag(item, token + SPLIT_BLOCK_DATA_SIZE, COMM_FLAG_COPY_STRIDE, serverRecordBlockCnt_);
    }
    uint32_t issued = 0U;
    for (uint32_t done = 0U; done < tokenCnt; ++done) {
        do {
            while (issued < tokenCnt && issued < done + gatherInputBufferNum_ &&
                   IssueGatherInput(source, issued, tokenCnt, startTokenIdx, tokenStride, local)) {
                ++issued;
            }
        } while (issued == done);
        const uint32_t slot = done % gatherInputBufferNum_;
        LocalTensor<XCommInType> input = tokenInQueue_.DeQue<XCommInType>();
        AscendC::WaitFlag<HardEvent::MTE2_S>(payloadReadyEvents_[slot]);
        SendToExpert(input, serverIdx, tokenStride, slot);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::ClearOwnedBitmapTail(uint32_t serverIdx,
                                                                                               uint32_t tokenCount,
                                                                                               uint32_t firstToken,
                                                                                               uint32_t tokenStride)
{
    const uint32_t firstTail =
        firstToken < tokenCount ? firstToken + Ceil(tokenCount - firstToken, tokenStride) * tokenStride : firstToken;
    for (uint32_t tokenId = firstTail; tokenId < combineBitmapTokenNum_; tokenId += tokenStride) {
        StoreTokenBitmap(serverIdx, tokenId, 0U, tokenStride);
    }
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
        const uint64_t offset =
            static_cast<uint64_t>(serverIdx) * (combineBitmapTokenNum_ + UB_ALIGN / sizeof(uint32_t));
        if (serverIdx == static_cast<uint32_t>(serverId_)) {
            SyncFunc<HardEvent::S_MTE3>();
        } else {
            SyncFunc<HardEvent::MTE2_MTE3>();
        }
        DataCopyPad(combineServerInfoGMTensor_[offset], serverFlagBuf.Get<uint32_t>(),
                    DataCopyExtParams{1U, UB_ALIGN, 0U, 0U, 0U});
    }
    const uint32_t firstToken = aivId_ - firstAivId;
    const uint32_t count = firstToken < tokenCount ? Ceil(tokenCount - firstToken, useAivPerServer) : 0U;
    WaitToken(count, serverIdx, firstToken, useAivPerServer);
    ClearOwnedBitmapTail(serverIdx, tokenCount, firstToken, useAivPerServer);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GatherTokenFrRemote()
{
    InitGatherTokenBuffer();
    TBuf<> serverFlagBuf;
    tpipe_->InitBuffer(serverFlagBuf, GetFlagCheckBufferSize(1U));
    const uint32_t firstGather = aivUsedCumSum_ + aivUsedRemotComm_;
    uint32_t first, end, groupSize, groupBegin;
    CalServerSplitCore(serverNum_ - 1U, firstGather, aivUsedGather_, first, end, groupSize, groupBegin);
    if (first < end) {
        PrefetchGatherHeader(first < static_cast<uint32_t>(serverId_) ? first : first + 1U);
    }
    GatherServerTokens(serverId_, firstGather, aivUsedGather_, serverFlagBuf);
    SyncFunc<HardEvent::MTE3_S>();
    SyncFunc<HardEvent::MTE3_MTE2>();
    for (uint32_t index = first; index < end; ++index) {
        const uint32_t server = index < static_cast<uint32_t>(serverId_) ? index : index + 1U;
        GatherServerTokens(server, groupBegin, groupSize, serverFlagBuf);
        if (index + 1U < end) {
            PrefetchGatherHeader(index + 1U < static_cast<uint32_t>(serverId_) ? index + 1U : index + 2U);
        }
        SyncFunc<HardEvent::MTE3_S>();
        SyncFunc<HardEvent::MTE3_MTE2>();
    }
    ReleasePrefetchBuffers();
    for (uint32_t slot = 0U; slot < gatherPacketBufferNum_; ++slot) {
        AscendC::WaitFlag<HardEvent::MTE3_V>(gatherPacketFree_[slot]);
        tpipe_->ReleaseEventID<HardEvent::MTE3_V>(gatherPacketFree_[slot]);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::IssueCopyPair(uint32_t& pair, uint32_t end,
                                                                                        uint32_t slot)
{
    while (pair < end) {
        const uint32_t current = pair++;
        const uint32_t previous = current == 0U ? 0U : sendCntTensor_.GetValue(current - 1U - copyPrefixStart_);
        if (sendCntTensor_.GetValue(current - copyPrefixStart_) == previous) {
            continue;
        }
        const uint32_t rowBytes = UB_ALIGN + maskBytes_;
        GM_ADDR card = windowGM_ + maskAreaOffset_ + static_cast<uint64_t>(current % epWorldSize_) * maskCardBytes_;
        LocalTensor<uint8_t> mask = localCopyMaskBuf_.Get<uint8_t>()[slot * rowBytes];
        ReadMaskRows(card, current / epWorldSize_ * rowBytes, rowBytes, mask);
        AscendC::SetFlag<HardEvent::MTE2_S>(payloadReadyEvents_[slot]);
        copyPairs_[slot] = current;
        return true;
    }
    return false;
}

template <RoceDispatchTypeClass>
__aicore__ inline GM_ADDR MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::GetCopyTokenAddr(uint32_t pair,
                                                                                              uint32_t ordinal)
{
    const uint64_t expert = static_cast<uint64_t>(pair % epWorldSize_) * moeExpertNumPerRank_ + pair / epWorldSize_;
    return windowGM_ + (expert * axisMaxBS_ + pendingTokenIds_.GetValue(ordinal)) * expertTokenStride_;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::SubmitCopyFlag(uint32_t pair,
                                                                                         uint32_t ordinal,
                                                                                         uint32_t destination,
                                                                                         uint32_t slot)
{
    copyTokenSources_[slot] = GetCopyTokenAddr(pair, ordinal);
    copyFlagPositions_[slot] = destination + ordinal;
    copyFlagActive_[slot] = true;
    SubmitPrefetchedFlag(slot, copyTokenSources_[slot] + SPLIT_BLOCK_DATA_SIZE, COMM_FLAG_COPY_STRIDE,
                         blockCntPerTokenToExpert_);
}

template <RoceDispatchTypeClass>
__aicore__ inline bool MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::IssueCopyToken(uint32_t slot,
                                                                                         uint32_t queueSlot)
{
    if (!copyFlagActive_[slot] || !TestPrefetchedFlag(slot)) {
        return false;
    }
    LocalTensor<XOutType> token = localCopyQueue_.AllocTensor<XOutType>();
    ReadDataFromWindows<XOutType, true, true>(copyTokenSources_[slot], blockCntPerTokenToExpert_, token);
    localCopyQueue_.EnQue(token);
    copyPayloadPositions_[queueSlot] = copyFlagPositions_[slot];
    copyFlagActive_[slot] = false;
    return true;
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::LoadNextCopyPair(RoceCopyCursor& cursor,
                                                                                           uint32_t end)
{
    const uint32_t slot = cursor.pairsRead++ % copyBufferNum_;
    cursor.pair = copyPairs_[slot];
    cursor.destination = cursor.pair == 0U ? 0U : sendCntTensor_.GetValue(cursor.pair - 1U - copyPrefixStart_);
    cursor.count = sendCntTensor_.GetValue(cursor.pair - copyPrefixStart_) - cursor.destination;
    cursor.ordinal = 0U;
    BuildCopyTokenIds(slot);
    IssueCopyPair(cursor.nextPair, end, slot);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::PrefetchCopyTokens(RoceCopyCursor& cursor,
                                                                                             uint32_t end,
                                                                                             uint32_t active,
                                                                                             uint32_t slots)
{
    bool loadedPair = false;
    for (uint32_t slot = 0U; slot < slots; ++slot) {
        if (copyFlagActive_[slot]) {
            continue;
        }
        if (cursor.ordinal == cursor.count) {
            if (loadedPair || cursor.pairsRead == active) {
                break;
            }
            LoadNextCopyPair(cursor, end);
            loadedPair = true;
        }
        SubmitCopyFlag(cursor.pair, cursor.ordinal++, cursor.destination, slot);
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CopyExpertPairs(uint32_t first, uint32_t end,
                                                                                          uint32_t active)
{
    uint32_t next = first;
    for (uint32_t slot = 0U; slot < AscendC::Std::min(active, copyBufferNum_); ++slot) {
        IssueCopyPair(next, end, slot);
    }
    const uint32_t previous = first == 0U ? 0U : sendCntTensor_.GetValue(first - 1U - copyPrefixStart_);
    const uint32_t tokens = sendCntTensor_.GetValue(end - 1U - copyPrefixStart_) - previous;
    RoceCopyCursor cursor{next, 0U, 0U, 0U, 0U, 0U};
    WaitAndFormatOutput(cursor, end, active, tokens);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitLocalCopyBuffer(uint32_t firstPair,
                                                                                              uint32_t endPair)
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    quantInst_.dynamicScalesOutGMTensor_ = dynamicScalesOutGMTensor_;
    copyPrefixStart_ =
        (firstPair == 0U ? 0U : firstPair - 1U) / (UB_ALIGN / sizeof(int32_t)) * (UB_ALIGN / sizeof(int32_t));
    const uint32_t prefixBytes = Align32((endPair - copyPrefixStart_) * sizeof(int32_t));
    const uint32_t tokenBytes = blockCntPerTokenToExpert_ * SPLIT_BLOCK_DATA_SIZE;
    const uint32_t routeBytes = Align32(axisMaxBS_ * axisK_ * sizeof(uint32_t));
    tpipe_->InitBuffer(localCopyStatusBuf_, GetFlagCheckBufferSize(aivUsedCumSum_));
    tpipe_->InitBuffer(localCopyPrefixBuf_, prefixBytes);
    tpipe_->InitBuffer(pendingTokenBuf_, copyBufferNum_ * routeBytes);
    tpipe_->InitBuffer(localCopyIndexBuf_, routeBytes);
    tpipe_->InitBuffer(localCopyMaskBuf_, copyBufferNum_ * (UB_ALIGN + maskBytes_));
    tpipe_->InitBuffer(localCopyQueue_, copyBufferNum_, tokenBytes);
    statusFp32Tensor_ = localCopyStatusBuf_.Get<float>();
    sendCntTensor_ = localCopyPrefixBuf_.Get<int32_t>();
    pendingTokenIds_ = pendingTokenBuf_.Get<uint32_t>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitCumSumFlag()
{
    LocalTensor<uint64_t> statusU64Tensor = statusFp32Tensor_.ReinterpretCast<uint64_t>();
    GM_ADDR flagAddr =
        statusDataSpaceGm_ + CUMSUM_FLAG_OFFSET + static_cast<uint64_t>(aivId_) * aivUsedCumSum_ * UB_ALIGN;
    WaitFlag<true>(flagAddr, statusU64Tensor, UB_ALIGN - sizeof(uint64_t), aivUsedCumSum_);
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::CopyOut(LocalTensor<XOutType> tokenTensor,
                                                                                  uint32_t dstPosition,
                                                                                  uint32_t tokenCount)
{
    GlobalTensor<XOutType> expandXOutGlobal;
    expandXOutGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ XOutType*>(expandXOutGM_) +
                                     static_cast<uint64_t>(dstPosition) * copyOutAxisH_);
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
    DataCopyPad(expandIdxGMTensor_[static_cast<uint64_t>(dstPosition) * EXPAND_IDX_INFO],
                tokenInt32Tensor[tokenQuantAlign_], expandIdxCopyParams);
    LocalTensor<float> tokenFloatTensor = tokenTensor.template ReinterpretCast<float>();
    DataCopyPad(expandScalesOutGMTensor_[dstPosition], tokenFloatTensor[expertScaleAlign_],
                dataCopyexpandScaleParams); // 拷贝expert专家系数
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::InitCopyTokenIndices()
{
    LocalTensor<uint32_t> tokenIds = localCopyIndexBuf_.Get<uint32_t>();
    const uint32_t routes = axisMaxBS_ * axisK_;
    CreateVecIndex(tokenIds.ReinterpretCast<int32_t>(), int32_t(0), routes);
    Duplicate(pendingTokenIds_, axisK_, routes);
    Div(tokenIds, tokenIds, pendingTokenIds_, routes);
    PipeBarrier<PIPE_V>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::BuildCopyTokenIds(uint32_t slot)
{
    const uint32_t rowBytes = UB_ALIGN + maskBytes_;
    AscendC::WaitFlag<HardEvent::MTE2_S>(payloadReadyEvents_[slot]);
    LocalTensor<uint32_t> words = localCopyMaskBuf_.Get<uint8_t>()[slot * rowBytes].ReinterpretCast<uint32_t>();
    pendingTokenIds_ =
        pendingTokenBuf_.Get<uint32_t>()[slot * Align32(axisMaxBS_ * axisK_ * sizeof(uint32_t)) / sizeof(uint32_t)];
    LocalTensor<uint32_t> tokenIds = localCopyIndexBuf_.Get<uint32_t>();
    const uint32_t routes = words.GetValue(0U) * axisK_;
    uint64_t selected = 0U;
    SyncFunc<HardEvent::S_V>();
    GatherMask(pendingTokenIds_, tokenIds, words[UB_ALIGN / sizeof(uint32_t)], true, routes, {1, 1, 0, 0}, selected);
    SyncFunc<HardEvent::V_S>();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::WaitAndFormatOutput(RoceCopyCursor& cursor,
                                                                                              uint32_t end,
                                                                                              uint32_t active,
                                                                                              uint32_t tokenCount)
{
    const uint32_t slots = AscendC::Std::min(tokenCount, copyBufferNum_);
    for (uint32_t slot = 0U; slot < slots; ++slot) {
        copyFlagActive_[slot] = false;
    }
    PrefetchCopyTokens(cursor, end, active, slots);
    uint32_t issued = 0U;
    for (uint32_t done = 0U; done < tokenCount; ++done) {
        do {
            for (uint32_t slot = 0U; slot < slots && issued < done + copyBufferNum_; ++slot) {
                if (IssueCopyToken(slot, issued % copyBufferNum_)) {
                    ++issued;
                }
            }
            if (issued == done) {
                PrefetchCopyTokens(cursor, end, active, slots);
            }
        } while (issued == done);
        LocalTensor<XOutType> token = localCopyQueue_.DeQue<XOutType>();
        CopyOut(token, copyPayloadPositions_[done % copyBufferNum_], 1U);
        localCopyQueue_.FreeTensor(token);
        // Only wait for another pair's mask when no ready payload remains in the FIFO.
        if (issued == done + 1U) {
            PrefetchCopyTokens(cursor, end, active, slots);
        }
    }
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::LocalWindowCopy()
{
    uint32_t startProcessId, endProcessId, processNum;
    SplitToCore(rscvStatusNum_, 0, aivNum_, startProcessId, endProcessId, processNum);
    if (processNum == 0) {
        return;
    }
    InitLocalCopyBuffer(startProcessId, endProcessId);
    WaitCumSumFlag();
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopyExtParams copy{1U, static_cast<uint32_t>((endProcessId - copyPrefixStart_) * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[copyPrefixStart_], copy, pad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t active = 0U;
    for (uint32_t index = startProcessId; index < endProcessId; ++index) {
        const uint32_t previous = index == 0U ? 0U : sendCntTensor_.GetValue(index - 1U - copyPrefixStart_);
        active += static_cast<uint32_t>(sendCntTensor_.GetValue(index - copyPrefixStart_)) != previous;
    }
    if (active != 0U) {
        InitCopyTokenIndices();
    }
    InitPrefetchBuffers(copyBufferNum_, GetFlagCheckBufferSize(blockCntPerTokenToExpert_));
    CopyExpertPairs(startProcessId, endProcessId, active);
    PipeBarrier<PIPE_ALL>();
    ReleasePrefetchBuffers();
}

template <RoceDispatchTypeClass>
__aicore__ inline void MoeDistributeDispatchV2Roce<RoceDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        CalTokenMask();
        CalExpertSendMask();
        QuantAllTokens();
        SyncAll<true>();
        if (aivId_ < aivUsedCumSum_) {
            CalCumSum();
        } else if (aivId_ < aivUsedCumSum_ + aivUsedRemotComm_) {
            HcclTokenDispatch();
        } else {
            GatherTokenFrRemote();
        }
        LocalWindowCopy();
    }
}

} // namespace Mc2Kernel
#endif // defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#endif // MOE_DISTRIBUTE_DISPATCH_V2_A5_ROCE_H
