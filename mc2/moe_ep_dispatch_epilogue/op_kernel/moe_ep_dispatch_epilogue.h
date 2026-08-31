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
 * \file moe_ep_dispatch_epilogue.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_H
#define MOE_EP_DISPATCH_EPILOGUE_H

#include <cstddef>

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_epilogue_tiling_key.h"
#include "moe_ep_dispatch_epilogue_tiling.h"

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

namespace MoeEpDispatchEpilogueImpl {

#if defined(ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL)

using namespace AscendC;

static constexpr uint32_t UB_ALIGN = 32U;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 4;
static constexpr uint8_t BUFFER_NUM = 2;
static constexpr uint32_t ELEM_ALIGN = 8U;
static constexpr uint32_t META_TOPK_SECTION = 2U;
static constexpr uint32_t META_EXTRA_FIELDS = 2U;
static constexpr uint32_t META_SRC_RANK_OFFSET = 0U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_SLOT_IDX_OFFSET = 3U;
static constexpr uint32_t HIT_ROW_OFFSET = 0U;
static constexpr uint32_t HIT_TOPK_OFFSET = 1U;
static constexpr uint32_t HIT_ENTRY_SIZE = 2U;
static constexpr uint32_t CACHED_META_TILE = 8192U;
static constexpr uint32_t ALIGNED_LEN_256 = 256U;
static constexpr uint32_t SLOTS_TILE = 128U;

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
class MoeEpDispatchEpilogue {
public:
    __aicore__ inline MoeEpDispatchEpilogue(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR dstBufferSlotIdx, GM_ADDR numRecvPerRank,
                                GM_ADDR numRecvPerExpert, GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX,
                                GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe *pipe, const MoeEpDispatchEpilogueInfo *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputePrefixSums();
    __aicore__ inline void CountHits();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void CopyFromWindowByExpert();
    __aicore__ inline void CopyFromWindowByCachedMeta();

    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,
                                       uint32_t &sendNum);
    __aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *ctx, uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)ctx->epHcclBuffer[rankId] + offset;
    }
    __aicore__ inline uint32_t ReduceSumWorkNeedSize(int32_t count, int32_t typeSize)
    {
        int32_t elementsPerBlock = UB_ALIGN / typeSize;
        int32_t elementsPerRepeat = ALIGNED_LEN_256 / typeSize;
        int32_t iter1OutputCount = (count + elementsPerRepeat - 1) / elementsPerRepeat;
        uint32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    TPipe *tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    GlobalTensor<XType> recvXGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<ScalesType> recvScalesGm_;
    GlobalTensor<int32_t> cachedRecvSrcMetadataGm_; // cached 路径专用：来自上一轮 dispatch 的 recv_src_metadata

    GlobalTensor<int32_t> hitCountGm_;

    LocalTensor<int32_t> ubHitCount_;
    LocalTensor<int64_t> ubRowStart_;
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubTopkIds_;
    LocalTensor<int32_t> ubTargetExpertId_;
    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<int64_t> ubExpertPfx_;
    LocalTensor<int64_t> ubHitCountRowI64_;
    LocalTensor<float> ubStageWeights_;
    LocalTensor<int32_t> ubStageMeta_;
    LocalTensor<int32_t> ubLocalCursor_;
    LocalTensor<int64_t> ubHitList_;
    LocalTensor<int32_t> ubWaitStatus_;
    LocalTensor<int32_t> ubWaitSum_;

    TBuf<QuePosition::VECIN> ubHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRowStartBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<QuePosition::VECIN> ubTopkIdsBuf_;
    TBuf<QuePosition::VECIN> ubTargetExpertIdBuf_;
    TBuf<QuePosition::VECIN> ubRecvCntBuf_;
    TBuf<QuePosition::VECIN> ubExpertPfxBuf_;
    TBuf<QuePosition::VECIN> ubHitCountRowI64Buf_;
    TBuf<QuePosition::VECIN> ubStageWeightsBuf_;
    TBuf<QuePosition::VECIN> ubStageMetaBuf_;
    TBuf<QuePosition::VECIN> ubLocalCursorBuf_;
    TBuf<QuePosition::VECIN> ubHitListBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tokenQueue_;
    TBuf<> waitStatusBuf_;
    TBuf<> waitSumBuf_;
    TBuf<> sharedTmpBuf_;

    uint32_t expertSum_{0};
    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR localWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    uint32_t scalesOffset_{0};
    uint32_t scalesBytes_{0};
    uint32_t scalesElems_{0};
    uint32_t metaOffset_{0};
    uint32_t tokenQueueBufBytes_{0};
    uint32_t hitCountOffset_{0};
    uint32_t metaBytes_{0};
    uint32_t paddedMetaElems_{0};
    uint32_t axisKAlign_{0};
    uint32_t paddedTopkElems_{0};
    uint32_t numLocalExperts_{0};
    uint32_t hitCountStride_{0}; // 对齐到 32B 的 hitCount 存储步长 (int32 元素数)
    uint32_t aivNum_{0};
    uint32_t axisK_{0};
    uint32_t axisH_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t perSlotBytes_{0};
    uint32_t dispatchNotifyCount_{1};
    uint32_t totalNotifyCnt_{0};
    uint64_t winDataOffset_{0};
    uint64_t slotWinStateOffset_{0};
    int32_t ppEvtSToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToS_[2] = {0, 0};
};

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseAivNum;
    uint32_t remainderNum = curSendCnt % curUseAivNum;
    uint32_t newAivId = aivId_;
    startId = sendNum * newAivId;
    if (newAivId < remainderNum) {
        sendNum += 1;
        startId += newAivId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::Init(
    GM_ADDR context, GM_ADDR dstBufferSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX, GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales,
    GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe, const MoeEpDispatchEpilogueInfo *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspace;
    numLocalExperts_ = tilingData->cfg.numLocalExperts;
    aivNum_ = tilingData->aivNum;
    axisK_ = tilingData->cfg.topK;
    axisH_ = tilingData->cfg.hidden;
    epWorldSize_ = tilingData->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData->cfg.numMaxTokensPerRank;
    perSlotBytes_ = tilingData->cfg.perSlotBytes;
    dispatchNotifyCount_ = tilingData->dispatchNotifyCount;
    winDataOffset_ = tilingData->winDataOffset;
    slotWinStateOffset_ = tilingData->slotWinStateOffset;

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    epRankId_ = mc2Context_->epRankId;
    constexpr size_t metadataOffset = offsetof(MoeEpDispatchEpilogueTilingData, moeEpDispatchEpilogueInfo) +
                                      offsetof(MoeEpDispatchEpilogueInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_DISPATCH_EPILOGUE, tpipe_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_);
    metaOffset_ = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;

    numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    cachedRecvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)cachedRecvSrcMetadata);
    recvXGm_.SetGlobalBuffer((__gm__ XType *)recvX);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    if constexpr (HasTopkWeights) {
        recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)recvTopkWeights);
    }

    hitCountGm_.SetGlobalBuffer((__gm__ int32_t *)(workspace));

    axisKAlign_ = Ceil(axisK_, ELEM_ALIGN) * ELEM_ALIGN;
    metaBytes_ = (META_TOPK_SECTION * axisKAlign_) * (uint32_t)sizeof(int32_t) + UB_ALIGN;
    totalNotifyCnt_ = epWorldSize_ * dispatchNotifyCount_;
    uint32_t ubExpertPfxBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t expertReduceTmpBytes =
        ReduceSumWorkNeedSize(static_cast<int32_t>(numLocalExperts_), sizeof(int64_t)) * sizeof(int64_t);
    uint32_t statusReduceTmpBytes =
        ReduceSumWorkNeedSize(static_cast<int32_t>(totalNotifyCnt_), sizeof(float)) * sizeof(float);
    uint32_t sharedBytes = expertReduceTmpBytes > statusReduceTmpBytes ? expertReduceTmpBytes : statusReduceTmpBytes;
    tpipe_->InitBuffer(ubExpertPfxBuf_, ubExpertPfxBytes);
    tpipe_->InitBuffer(waitStatusBuf_, totalNotifyCnt_ * UB_ALIGN);
    tpipe_->InitBuffer(waitSumBuf_, UB_ALIGN);
    tpipe_->InitBuffer(sharedTmpBuf_, sharedBytes);
    ubExpertPfx_ = ubExpertPfxBuf_.Get<int64_t>();
    ubWaitStatus_ = waitStatusBuf_.Get<int32_t>();
    ubWaitSum_ = waitSumBuf_.Get<int32_t>();

    if constexpr (!IsCached) {
        hitCountStride_ = Ceil(numLocalExperts_, ELEM_ALIGN) * ELEM_ALIGN;
        paddedMetaElems_ = Ceil(META_TOPK_SECTION * axisKAlign_ + META_EXTRA_FIELDS, ELEM_ALIGN) * ELEM_ALIGN;
        paddedTopkElems_ = axisKAlign_;
        uint32_t ubHitCountBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        uint32_t ubRowStartBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
        uint32_t ubMetaBytes = Ceil((uint32_t)(SLOTS_TILE * paddedMetaElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        uint32_t ubTopkIdsBytes =
            Ceil((uint32_t)(SLOTS_TILE * paddedTopkElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        uint32_t ubRecvCntBytes = Ceil((uint32_t)(epWorldSize_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        uint32_t ubHitCountRowI64Bytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
        uint32_t ubStageMetaBytes = axisK_ * UB_ALIGN * 2;
        uint32_t ubStageWeightsBytes = axisK_ * UB_ALIGN * 2;
        uint32_t ubHitListBytes = Ceil((uint32_t)(axisK_ * HIT_ENTRY_SIZE * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
        uint32_t ubLocalCursorBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;

        tpipe_->InitBuffer(ubRecvCntBuf_, ubRecvCntBytes);
        tpipe_->InitBuffer(ubHitCountBuf_, ubHitCountBytes);
        tpipe_->InitBuffer(ubRowStartBuf_, ubRowStartBytes);
        tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
        tpipe_->InitBuffer(ubTopkIdsBuf_, ubTopkIdsBytes);
        tpipe_->InitBuffer(ubTargetExpertIdBuf_, ubTopkIdsBytes);
        tpipe_->InitBuffer(ubHitCountRowI64Buf_, ubHitCountRowI64Bytes);
        tpipe_->InitBuffer(ubStageWeightsBuf_, ubStageWeightsBytes);
        tpipe_->InitBuffer(ubStageMetaBuf_, ubStageMetaBytes);
        tpipe_->InitBuffer(ubHitListBuf_, ubHitListBytes);
        tpipe_->InitBuffer(ubLocalCursorBuf_, ubLocalCursorBytes);
        ubRecvCnt_ = ubRecvCntBuf_.Get<int32_t>();
        ubHitCount_ = ubHitCountBuf_.Get<int32_t>();
        ubRowStart_ = ubRowStartBuf_.Get<int64_t>();
        ubMeta_ = ubMetaBuf_.Get<int32_t>();
        ubTopkIds_ = ubTopkIdsBuf_.Get<int32_t>();
        ubTargetExpertId_ = ubTargetExpertIdBuf_.Get<int32_t>();
        ubHitCountRowI64_ = ubHitCountRowI64Buf_.Get<int64_t>();
        ubStageWeights_ = ubStageWeightsBuf_.Get<float>();
        ubStageMeta_ = ubStageMetaBuf_.Get<int32_t>();
        ubHitList_ = ubHitListBuf_.Get<int64_t>();
        ubLocalCursor_ = ubLocalCursorBuf_.Get<int32_t>();
        ppEvtSToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        ppEvtSToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        ppEvtMte3ToS_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        ppEvtMte3ToS_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    }

    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesOffset_ = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;
        scalesBytes_ = tilingData->cfg.scalesBytes;
        scalesElems_ = (scalesBytes_ == 0U) ? 0U : (scalesBytes_ / sizeof(ScalesType));
        uint32_t scalesBytesAlign = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
        metaOffset_ += scalesBytesAlign;
        recvScalesGm_.SetGlobalBuffer((__gm__ ScalesType *)recvScales);
    }

    tokenQueueBufBytes_ = metaOffset_;

    tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, tokenQueueBufBytes_);

    DataCopyExtParams expertPfxCopyParams{1U, static_cast<uint32_t>(numLocalExperts_ * sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int64_t> expertPfxPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubExpertPfx_, numRecvPerExpertGm_, expertPfxCopyParams, expertPfxPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<int64_t> expertReduceTmp = sharedTmpBuf_.Get<int64_t>();
    LocalTensor<int64_t> expertSumOut = waitSumBuf_.Get<int64_t>();
    ReduceSum<int64_t>(expertSumOut, ubExpertPfx_, expertReduceTmp, static_cast<int32_t>(numLocalExperts_));
    SyncFunc<AscendC::HardEvent::V_S>();
    expertSum_ = static_cast<uint32_t>(expertSumOut.GetValue(0));
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_INIT_DONE);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::Process()
{
    if constexpr (!IsCached) {
        ComputePrefixSums();
        WaitDispatch();
        SyncAll<true>();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_WAIT_DONE);
        CountHits();
        SyncAll<true>();

        uint32_t totalHits = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
            totalHits += static_cast<uint32_t>(ubHitCount_.GetValue(localExpertIdx));
        }

        if (totalHits != 0) {
            CopyFromWindowByExpert();
        }
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_OUTPUT_DONE);
    } else {
        WaitDispatch();
        SyncAll<true>();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_WAIT_DONE);
        CopyFromWindowByCachedMeta();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_OUTPUT_DONE);
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::WaitDispatch()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    uint32_t mask = 1;
    int32_t sumOfFlag = 0;
    int32_t commpareFlag = static_cast<int32_t>(totalNotifyCnt_);
    GlobalTensor<int32_t> statusGMTensor;
    LocalTensor<float> sharedTmp = sharedTmpBuf_.Get<float>();
    LocalTensor<float> ubWaitStatusFp32 = ubWaitStatus_.template ReinterpretCast<float>();
    LocalTensor<float> ubWaitSumFp32 = ubWaitSum_.template ReinterpretCast<float>();
    statusGMTensor.SetGlobalBuffer((__gm__ int32_t *)localSlotStateWinAddr_);
    DataCopyParams statusCopyParams = {static_cast<uint16_t>(totalNotifyCnt_), 1U,
                                       static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
    DataCopyParams clearStatusCopyParams = {static_cast<uint16_t>(totalNotifyCnt_), 1U, 0U,
                                            static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};

    SyncFunc<AscendC::HardEvent::S_V>(); // 确保expertSum_计算完成
    while (sumOfFlag != commpareFlag) {
        DataCopy(ubWaitStatus_, statusGMTensor, statusCopyParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(ubWaitSumFp32, ubWaitStatusFp32, sharedTmp, mask, totalNotifyCnt_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = ubWaitSum_.GetValue(0);
    }
    Duplicate<int32_t>(ubWaitStatus_, 0, totalNotifyCnt_ * UB_ALIGN / sizeof(int32_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusGMTensor, ubWaitStatus_, clearStatusCopyParams);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::ComputePrefixSums()
{
    int64_t cumulativeRowOffset = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        int64_t expertTokenCnt = ubExpertPfx_.GetValue(localExpertIdx);
        ubExpertPfx_.SetValue(localExpertIdx, cumulativeRowOffset);
        cumulativeRowOffset += expertTokenCnt;
    }
    SyncFunc<AscendC::HardEvent::S_V>();
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CountHits()
{
    DataCopyExtParams recvCntCopyParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> recvCntPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubRecvCnt_, numRecvPerRankGm_, recvCntCopyParams, recvCntPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    Duplicate(ubHitCount_, (int32_t)0, numLocalExperts_);

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t slotCnt = ubRecvCnt_.GetValue(rankId);
        if (slotCnt == 0) {
            continue;
        }

        uint32_t slotStart, slotEnd, slotCntPerAiv;
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        if (slotStart >= slotEnd) {
            continue;
        }

        GM_ADDR srcRankBase = localWinAddr_ + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_;
        GlobalTensor<int32_t> srcTopkIdsGm;
        srcTopkIdsGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(srcRankBase + (int64_t)slotStart * perSlotBytes_ + metaOffset_));

        uint32_t topkBytes = axisK_ * sizeof(int32_t);
        LocalTensor<uint8_t> sharedTmpInt8 = ubMetaBuf_.Get<uint8_t>();
        LocalTensor<uint32_t> sharedTmpInt32 = ubMetaBuf_.Get<uint32_t>();

        for (uint32_t tileStart = 0; tileStart < slotCntPerAiv; tileStart += SLOTS_TILE) {
            uint32_t tileCnt = (slotCntPerAiv - tileStart > SLOTS_TILE) ? SLOTS_TILE : (slotCntPerAiv - tileStart);

            DataCopyExtParams topkCopyParams{static_cast<uint16_t>(tileCnt), static_cast<uint32_t>(topkBytes),
                                             static_cast<int64_t>(perSlotBytes_ - topkBytes), 0, 0};
            DataCopyPadExtParams<int32_t> topkPadParams{true, 0, static_cast<uint8_t>(paddedTopkElems_ - axisK_), -1};
            DataCopyPad(ubTopkIds_, srcTopkIdsGm[(int64_t)tileStart * (perSlotBytes_ / sizeof(int32_t))],
                        topkCopyParams, topkPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();

            int32_t calCnt = static_cast<int32_t>(tileCnt * paddedTopkElems_);
            constexpr int32_t CMP_ALIGN = ALIGNED_LEN_256 / sizeof(int32_t);
            int32_t calCntAlign = Ceil(calCnt, CMP_ALIGN) * CMP_ALIGN;
            for (uint32_t localExpertId = 0; localExpertId < numLocalExperts_; ++localExpertId) {
                int32_t targetExpertId = static_cast<int32_t>(epRankId_ * numLocalExperts_ + localExpertId);
                uint64_t rsvdCnt = 0;
                CompareScalar(sharedTmpInt8, ubTopkIds_, static_cast<int32_t>(targetExpertId), AscendC::CMPMODE::EQ,
                              calCntAlign);
                GatherMask(ubTargetExpertId_, ubTopkIds_, sharedTmpInt32, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
                SyncFunc<AscendC::HardEvent::V_S>();
                int32_t curExpertCnt = rsvdCnt;
                int32_t currentHits = ubHitCount_.GetValue(localExpertId);
                ubHitCount_.SetValue(localExpertId, currentHits + curExpertCnt);
            }
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams hitCountCopyParams{1U, static_cast<uint32_t>(numLocalExperts_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(hitCountGm_[(int64_t)aivId_ * hitCountStride_], ubHitCount_, hitCountCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyFromWindowByExpert()
{
    DataCopyExtParams hitCountOneCopyParams{1U, static_cast<uint32_t>(numLocalExperts_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> hitCountOnePadParams{false, 0U, 0U, 0};
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    Adds(ubRowStart_, ubExpertPfx_, static_cast<int64_t>(0), numLocalExperts_);
    for (uint32_t aiv = 0; aiv < aivId_; ++aiv) {
        DataCopyPad(ubHitCount_, hitCountGm_[(int64_t)aiv * hitCountStride_], hitCountOneCopyParams,
                    hitCountOnePadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(ubHitCountRowI64_, ubHitCount_, RoundMode::CAST_NONE, numLocalExperts_);
        Add(ubRowStart_, ubRowStart_, ubHitCountRowI64_, numLocalExperts_);
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }
    Duplicate(ubLocalCursor_, (int32_t)0, numLocalExperts_);
    SyncFunc<AscendC::HardEvent::V_S>();
    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyPadParams tokenPadParams{false, 0, 0, 0};
    DataCopyExtParams metaOutParams{1U, static_cast<uint32_t>(RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyExtParams weightOutParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    int32_t rankExpertBase = static_cast<int32_t>(epRankId_ * numLocalExperts_);
    int32_t rankExpertEnd = rankExpertBase + static_cast<int32_t>(numLocalExperts_);

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t slotCnt = ubRecvCnt_.GetValue(rankId);
        if (slotCnt == 0) {
            continue;
        }

        uint32_t slotStart, slotEnd, slotCntPerAiv;
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        if (slotStart >= slotEnd) {
            continue;
        }

        GM_ADDR srcRankBase = localWinAddr_ + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_;
        GlobalTensor<int32_t> srcMetaGm;
        srcMetaGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(srcRankBase + (int64_t)slotStart * perSlotBytes_ + metaOffset_));

        for (uint32_t tileStart = 0; tileStart < slotCntPerAiv; tileStart += SLOTS_TILE) {
            uint32_t tileCnt = (slotCntPerAiv - tileStart > SLOTS_TILE) ? SLOTS_TILE : (slotCntPerAiv - tileStart);

            DataCopyExtParams metaCopyParams{static_cast<uint16_t>(tileCnt), metaBytes_, perSlotBytes_ - metaBytes_, 0,
                                             0};
            DataCopyPad(ubMeta_, srcMetaGm[(int64_t)tileStart * (perSlotBytes_ / sizeof(int32_t))], metaCopyParams,
                        metaPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();

            for (uint32_t localSlot = 0; localSlot < tileCnt; ++localSlot) {
                uint32_t metaBase = localSlot * paddedMetaElems_;
                uint32_t slotBufId = localSlot & 1U;
                uint32_t stageOff = slotBufId * axisK_ * ELEM_ALIGN;
                uint32_t hitCnt = 0;
                int32_t srcRankMeta = ubMeta_.GetValue(metaBase + META_TOPK_SECTION * axisKAlign_);
                int32_t tokenIdxMeta = ubMeta_.GetValue(metaBase + META_TOPK_SECTION * axisKAlign_ + 1);
                GM_ADDR slotAddr = srcRankBase + (int64_t)(slotStart + tileStart + localSlot) * perSlotBytes_;
                for (uint32_t topkIdx = 0; topkIdx < axisK_; ++topkIdx) {
                    int32_t expertId = ubMeta_.GetValue(metaBase + topkIdx);
                    if (expertId < rankExpertBase || expertId >= rankExpertEnd) {
                        continue;
                    }
                    uint32_t localExpertId = static_cast<uint32_t>(expertId - rankExpertBase);
                    int64_t expertRowStart = ubRowStart_.GetValue(localExpertId);
                    int32_t cursor = ubLocalCursor_.GetValue(localExpertId);
                    ubLocalCursor_.SetValue(localExpertId, cursor + 1);
                    int64_t globalRow = expertRowStart + cursor;
                    ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_ROW_OFFSET, globalRow);
                    ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET, static_cast<int64_t>(topkIdx));
                    hitCnt++;
                }
                if (hitCnt == 0) {
                    if (localSlot >= 2U) {
                        WaitFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[slotBufId]);
                    }
                    SetFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[slotBufId]);
                    continue;
                }

                if (localSlot >= 2U) {
                    WaitFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[slotBufId]);
                }

                GlobalTensor<XType> srcTokenTensor;
                srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(slotAddr), axisH_);

                LocalTensor<XType> tokenTensor = tokenQueue_.AllocTensor<XType>();
                DataCopyPad(tokenTensor, srcTokenTensor, tokenCopyParams, tokenPadParams);
                if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                    GlobalTensor<ScalesType> srcScalesTensor;
                    srcScalesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType *>(slotAddr + scalesOffset_),
                                                    scalesElems_);
                    DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U,
                                                    0U};
                    DataCopyPadParams scalesPadParams{false, 0, 0, 0};
                    DataCopyPad(tokenTensor[scalesOffset_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                                srcScalesTensor, scalesCopyParams, scalesPadParams);
                }
                tokenQueue_.EnQue(tokenTensor);
                LocalTensor<XType> tokenOut = tokenQueue_.DeQue<XType>();

                for (uint32_t i = 0; i < hitCnt; i++) {
                    int64_t globalRow = ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_ROW_OFFSET);
                    uint32_t topkIdx = static_cast<uint32_t>(ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET));

                    DataCopyPad(recvXGm_[globalRow * axisH_], tokenOut, tokenCopyParams);
                    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                        DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)),
                                                        0U, 0U};
                        DataCopyPad(recvScalesGm_[globalRow * scalesElems_],
                                    tokenOut[scalesOffset_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                                    scalesCopyParams);
                    }

                    if constexpr (HasTopkWeights) {
                        float weights = ubMeta_.ReinterpretCast<float>().GetValue(metaBase + axisKAlign_ + topkIdx);
                        ubStageWeights_.SetValue(stageOff + i * ELEM_ALIGN, weights);
                    }
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_SRC_RANK_OFFSET, srcRankMeta);
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_TOKEN_IDX_OFFSET, tokenIdxMeta);
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_TOPK_IDX_OFFSET,
                                          static_cast<int32_t>(topkIdx));
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_SLOT_IDX_OFFSET,
                                          static_cast<int32_t>(slotStart + tileStart + localSlot));
                }
                tokenQueue_.FreeTensor(tokenOut);

                SetFlag<AscendC::HardEvent::S_MTE3>(ppEvtSToMte3_[slotBufId]);
                WaitFlag<AscendC::HardEvent::S_MTE3>(ppEvtSToMte3_[slotBufId]);
                for (uint32_t i = 0; i < hitCnt; i++) {
                    int64_t globalRow = ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_ROW_OFFSET);
                    if constexpr (HasTopkWeights) {
                        DataCopyPad(recvTopkWeightsGm_[globalRow], ubStageWeights_[stageOff + i * ELEM_ALIGN],
                                    weightOutParams);
                    }
                    DataCopyPad(recvSrcMetadataGm_[globalRow * RECV_META_FIELDS],
                                ubStageMeta_[stageOff + i * ELEM_ALIGN], metaOutParams);
                }
                SetFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[slotBufId]);
            }
            if (tileCnt >= 1U) {
                WaitFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[(tileCnt - 1U) & 1U]);
            }
            if (tileCnt >= 2U) {
                WaitFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[(tileCnt - 2U) & 1U]);
            }
            SyncFunc<AscendC::HardEvent::S_MTE2>();
        }
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyFromWindowByCachedMeta()
{
    if (expertSum_ == 0) {
        return;
    }

    uint32_t startId, endId, cnt;
    SplitToCore(expertSum_, aivNum_, startId, endId, cnt);
    if (startId >= endId) {
        return;
    }

    uint32_t ubMetaBytes = Ceil(metaBytes_, UB_ALIGN) * UB_ALIGN;
    uint32_t ubStageWeightsBytes = Ceil((uint32_t)(CACHED_META_TILE * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubStageMetaBytes =
        Ceil((uint32_t)(CACHED_META_TILE * RECV_META_FIELDS * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
    tpipe_->InitBuffer(ubStageWeightsBuf_, ubStageWeightsBytes);
    tpipe_->InitBuffer(ubStageMetaBuf_, ubStageMetaBytes);
    ubMeta_ = ubMetaBuf_.Get<int32_t>();
    ubStageWeights_ = ubStageWeightsBuf_.Get<float>();
    ubStageMeta_ = ubStageMetaBuf_.Get<int32_t>();

    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyPadParams tokenPadParams{false, 0, 0, 0};

    uint32_t processed = 0;
    while (processed < cnt) {
        uint32_t tileCnt = (cnt - processed > CACHED_META_TILE) ? CACHED_META_TILE : (cnt - processed);
        uint32_t tileStartGlobal = startId + processed;

        DataCopyExtParams metaInParams{1U, static_cast<uint32_t>(tileCnt * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U,
                                       0U};
        DataCopyPad(ubStageMeta_, cachedRecvSrcMetadataGm_[(int64_t)tileStartGlobal * RECV_META_FIELDS], metaInParams,
                    metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < tileCnt; ++i) {
            uint32_t metaBase = i * RECV_META_FIELDS;
            int32_t srcRankId = ubStageMeta_.GetValue(metaBase + META_SRC_RANK_OFFSET);
            int32_t srcTopkIdx = ubStageMeta_.GetValue(metaBase + META_TOPK_IDX_OFFSET);
            int32_t slotIdx = ubStageMeta_.GetValue(metaBase + META_SLOT_IDX_OFFSET);

            GM_ADDR slotAddr = localWinAddr_ + (int64_t)srcRankId * numMaxTokensPerRank_ * perSlotBytes_ +
                               (int64_t)slotIdx * perSlotBytes_;

            GlobalTensor<XType> srcTokenTensor;
            srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(slotAddr), axisH_);
            LocalTensor<XType> tokenTensor = tokenQueue_.AllocTensor<XType>();
            DataCopyPad(tokenTensor, srcTokenTensor, tokenCopyParams, tokenPadParams);

            if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                GlobalTensor<ScalesType> srcScalesTensor;
                srcScalesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType *>(slotAddr + scalesOffset_),
                                                scalesElems_);
                DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
                DataCopyPadParams scalesPadParams{false, 0, 0, 0};
                DataCopyPad(tokenTensor[scalesOffset_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                            srcScalesTensor, scalesCopyParams, scalesPadParams);
            }

            if constexpr (HasTopkWeights) {
                GlobalTensor<int32_t> srcMetaGm;
                srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slotAddr + metaOffset_));
                DataCopyExtParams slotMetaParams{1U, metaBytes_, 0U, 0U, 0U};
                DataCopyPad(ubMeta_, srcMetaGm, slotMetaParams, metaPadParams);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                float weight = ubMeta_.ReinterpretCast<float>().GetValue(axisKAlign_ + srcTopkIdx);
                ubStageWeights_.SetValue(i, weight);
                SyncFunc<AscendC::HardEvent::S_MTE2>();
            }

            tokenQueue_.EnQue(tokenTensor);
            LocalTensor<XType> tokenOut = tokenQueue_.DeQue<XType>();
            uint32_t globalRow = tileStartGlobal + i;
            DataCopyPad(recvXGm_[(int64_t)globalRow * axisH_], tokenOut, tokenCopyParams);
            if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
                DataCopyPad(recvScalesGm_[(int64_t)globalRow * scalesElems_],
                            tokenOut[scalesOffset_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                            scalesCopyParams);
            }
            tokenQueue_.FreeTensor(tokenOut);
        }

        if constexpr (HasTopkWeights) {
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyExtParams weightOutParams{1U, static_cast<uint32_t>(tileCnt * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(recvTopkWeightsGm_[tileStartGlobal], ubStageWeights_, weightOutParams);
        }
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyExtParams metaOutParams{1U, static_cast<uint32_t>(tileCnt * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U,
                                        0U};
        DataCopyPad(recvSrcMetadataGm_[(int64_t)tileStartGlobal * RECV_META_FIELDS], ubStageMeta_, metaOutParams);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

        processed += tileCnt;
    }
}
#endif

} // namespace MoeEpDispatchEpilogueImpl

#endif // MOE_EP_DISPATCH_EPILOGUE_H
