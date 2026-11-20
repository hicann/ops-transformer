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

#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
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
#if defined(ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL)
#include "c_api/asc_simd.h"
#include "moe_ep_dispatch_epilogue_count.h"
#include "moe_ep_dispatch_epilogue_route.h"
#endif

#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_shared_jetty_context.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"
#include "../../common/op_kernel/moe_ep_send_completion.h"

namespace MoeEpDispatchEpilogueImpl {

#if defined(ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL)

using namespace AscendC;
using namespace MoeEpDispatchEpilogueRoute;

static constexpr uint32_t UB_ALIGN = MOE_EP_UB_BLOCK_BYTES;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t NETWORK_HYBRID = 1U; // 1 = hybrid dispatch, 0 = direct
static constexpr uint32_t MOE_EP_SLOT_META_EXTRA_FIELDS = 2U;
static constexpr uint32_t SHARED_PHASE_BYTES = 64U * 1024U;
static constexpr uint32_t WAIT_STATUS_RANKS_PER_BATCH = 16U;
static constexpr uint32_t MOE_EP_BUFFER_NUM = 2U;
static constexpr uint32_t MOE_EP_ROUTE_SCRATCH_TABLES = 2U;

// recv_src_metadata is compact in GM: [srcRank, srcToken, srcTopk, recvXIdx].
// The valid prefix is ordered by (srcRank, recvXIdx); each rank range follows increasing send-source rows.
// Each expert's UB metadata group starts on a 32-byte boundary and contains dense 16-byte records.
static constexpr uint32_t ELEM_ALIGN = MOE_EP_INT32_PER_BLOCK;
static constexpr uint32_t SLOTS_TILE = 128U;
static constexpr uint32_t PLAN_END_EXTRA = 1U; // UB额外读前一slot终点，GM不存起始0

// Three staging buffers let cached weights and non-cached local indices wait on each
// buffer's previous MTE3 copy before reuse.
static constexpr uint32_t STAGE_RING = 3U;
static constexpr uint32_t SHARED_HALF_BYTES = SHARED_PHASE_BYTES / MOE_EP_BUFFER_NUM;

// 预读扫描的起点哨兵：表示「还没发过任何一拍的 meta」，从第一个有活的 rank 的第一拍开始找。
static constexpr uint32_t TILE_NONE = 0xFFFFFFFFU;

// Event IDs are scoped to a source/destination pipe pair.
// MTE2_V/V_MTE2: 0..1 count/route frames; 2..3 rank/core-prefix input halves.
// V_S/V_MTE3/MTE3_V: 0..1 route plans, metadata and output reuse.
// MTE2_S: 0..1 metadata, 2..3 cached direct-self token IDs; S_MTE2: 0..1 input reuse.
// MTE2_MTE3/MTE3_MTE2: 0..1 token buffers; 2..3 independent cached-array copies.
// MTE3_S/S_MTE3: 2..4 index/weight stages; MTE3_S: 5 phase output completion.
// MTE2_S: 4 recv-count readiness; MTE2_MTE3/V_MTE3: 4 rank-offset/fill readiness.
// All used directions: 7 remaining temporary Set/Wait pairs in this kernel.
static constexpr uint32_t SYNC_EVENT_ID0 = 0U;
static constexpr uint32_t SYNC_EVENT_ID2 = 2U;
static constexpr uint32_t SYNC_EVENT_ID4 = 4U;
static constexpr uint32_t SYNC_EVENT_ID5 = 5U;
static constexpr uint32_t SYNC_EVENT_ID7 = 7U;

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
class MoeEpDispatchEpilogue {
public:
    __aicore__ inline MoeEpDispatchEpilogue(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR numRecvPerRank,
                                GM_ADDR numRecvPerExpert, GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX,
                                GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputePrefixSums();
    __aicore__ inline void CountHits();
    __aicore__ inline void BuildRankRowStarts();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void WaitDispatchBatch(GM_ADDR statusAddr, uint32_t notifyCount);
    __aicore__ inline void IssueRouteAndCopySlots();
    __aicore__ inline void CopySlotsCached();
    __aicore__ inline void IssueRouteInput(uint32_t rank, uint32_t slot, uint32_t count, uint32_t sequence);
    __aicore__ inline void BuildRouteInfo(uint32_t rank, uint32_t count, uint32_t planBase, uint32_t sequence);
    __aicore__ inline void ConsumeRoutePlan(uint32_t rank, uint32_t slot, uint32_t count, uint32_t planBase,
                                            uint32_t sequence, uint32_t& tokenSequence, uint32_t& tokenIssued,
                                            uint32_t& indexSequence, bool hasNext, uint32_t nextRank, uint32_t nextSlot,
                                            uint32_t nextCount);
    __aicore__ inline void IssueRouteTokens(uint32_t rank, uint32_t slot, uint32_t count, uint32_t first,
                                            uint32_t sequence, uint32_t tokenSequence);
    __aicore__ inline void WriteRouteMetadata(uint32_t rank, uint32_t slot, uint32_t count, uint32_t planBase,
                                              uint32_t bufferIdx);
    // 把 (rank, tile) 两层循环拍平成「取下一个 tile」。预读 meta 必须知道下一拍在哪，而下一拍会跨 rank，
    // 每个 rank 的槽位数又只有标量读得到；epWorldSize_ 很小，逐 rank 扫一遍的代价可以忽略。
    __aicore__ inline bool NextTile(uint32_t rankId, uint32_t tileStart, uint32_t& nextRank, uint32_t& nextTileStart,
                                    uint32_t& nextTileCnt, uint32_t& nextSlotStart, uint32_t slotsPerTile);
    __aicore__ inline void IssueTopkIds(uint32_t rankId, uint32_t slotIdx, uint32_t tileCnt, uint32_t buf);
    __aicore__ inline void IssueMeta(uint32_t rankId, uint32_t slotIdx, uint32_t tileCnt, uint32_t buf);
    __aicore__ inline void InitSlotBuffers(uint32_t tokenSlotBytes, const MoeEpDispatchEpilogueInfo* tilingData);
    __aicore__ inline uint32_t GlobalSlot(uint32_t rankId, uint32_t slotId);
    __aicore__ inline void CopyCachedArray(GlobalTensor<int32_t> src, GlobalTensor<int32_t> dst, uint32_t count);
    __aicore__ inline void CopyCachedMetadata();
    __aicore__ inline void IssueTokenBatch(GM_ADDR slotAddr, LocalTensor<int32_t> tokenIds, uint32_t count,
                                           uint32_t sequence, bool directSelf);
    __aicore__ inline void IssueCachedIds(uint32_t rank, uint32_t slot, uint32_t count, uint32_t bufferIdx);
    __aicore__ inline void IssueCachedTokens(uint32_t rank, uint32_t slot, uint32_t count, uint32_t first,
                                             uint32_t metaSequence, uint32_t tokenSequence);
    __aicore__ inline void CopyCachedRankOffsets();
    __aicore__ inline void InitLocalRecvIndex();

    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t& startId, uint32_t& endId,
                                       uint32_t& sendNum);
    __aicore__ inline GM_ADDR GetWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        if constexpr (ClosOnly) {
            auto ctx = reinterpret_cast<__gm__ Mc2Kernel::MoeSharedJettyContext*>(mc2Context_);
            return (GM_ADDR)ctx->epHcclBuffer[rankId] + offset;
        }
        auto ctx = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext*>(mc2Context_);
        return (GM_ADDR)ctx->epHcclBuffer[rankId] + offset;
    }
    TPipe* tpipe_{nullptr};
    GM_ADDR mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;
    uint32_t epRankId_{0};
    uint32_t networkMode_{0}; // 0 = direct(自段 x 直读输入), 1 = hybrid(自段 x 读win)
    uint32_t aivId_{0};
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    GlobalTensor<XType> xGm_; // 本端源 x（direct: 从输入直接获取）
    GlobalTensor<XType> recvXGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> recvRankOffsetsGm_;
    GlobalTensor<int32_t> localRecvIndexGm_;
    GlobalTensor<int32_t> cachedLocalRecvIndexGm_;
    GlobalTensor<int32_t> slotEndsGm_;
    GlobalTensor<int32_t> slotRecvXIdsGm_;
    GlobalTensor<int32_t> slotTopkIdsGm_;
    GlobalTensor<int32_t> cachedSlotEndsGm_;
    GlobalTensor<int32_t> cachedSlotRecvXIdsGm_;
    GlobalTensor<int32_t> cachedSlotTopkIdsGm_;
    GlobalTensor<ScalesType> recvScalesGm_;
    GlobalTensor<int32_t> cachedRecvSrcMetadataGm_; // cached 路径专用：来自上一轮 dispatch 的 recv_src_metadata
    GlobalTensor<int32_t> cachedRecvRankOffsetsGm_; // cached 路径专用：来自上一轮 dispatch 的 rank offsets

    GlobalTensor<int32_t> rankExpertHitCountGm_;

    LocalTensor<int32_t> ubHitCount_;
    LocalTensor<int32_t> ubRankExpertHitCount_;
    LocalTensor<int32_t> ubRankExpertRowStart_;
    LocalTensor<int32_t> ubRankOffsets_;
    LocalTensor<int64_t> ubRowStart_;
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<int64_t> ubExpertPfx_;
    LocalTensor<int64_t> ubHitCountRowI64_;
    LocalTensor<XType> tokenRing_;
    LocalTensor<float> ubStageWeightsRing_;
    LocalTensor<int32_t> ubStageLocalIndexRing_;
    LocalTensor<int32_t> ubLocalCursor_;
    uint32_t slotsPerTile_{SLOTS_TILE};
    uint32_t countSlotsPerTile_{1U};
    uint32_t planEndsElems_{0};
    uint32_t cachedTopkBase_{0};
    uint32_t cachedIdsBase_{0};
    uint32_t cachedWeightsBase_{0};
    uint32_t cachedIdsSequence_[MOE_EP_BUFFER_NUM] = {}; // 由 InitSlotBuffers 填 TILE_NONE，不依赖份数为 2
    uint32_t planRowBegin_[MOE_EP_BUFFER_NUM] = {0};
    LocalTensor<int32_t> ubWaitStatus_;
    LocalTensor<int32_t> ubWaitSum_;
    LocalTensor<uint8_t> ubSharedTmp_;

    TBuf<QuePosition::VECIN> ubHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRankExpertHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRankExpertRowStartBuf_;
    TBuf<QuePosition::VECIN> ubRankOffsetsBuf_;
    TBuf<QuePosition::VECIN> ubRowStartBuf_;
    TBuf<QuePosition::VECIN> sharedPhaseBuf_;
    TBuf<QuePosition::VECIN> ubRecvCntBuf_;
    TBuf<QuePosition::VECIN> ubExpertPfxBuf_;
    TBuf<QuePosition::VECIN> ubHitCountRowI64Buf_;
    TBuf<QuePosition::VECIN> ubLocalCursorBuf_;
    // 两条路径共用token环；cached weights及首次执行local-index保持独立stage。
    TBuf<QuePosition::VECIN> tokenRingBuf_;
    TBuf<QuePosition::VECIN> ubStageWeightsRingBuf_;
    TBuf<QuePosition::VECIN> ubStageLocalIndexRingBuf_;

    uint32_t expertSum_{0};
    GM_ADDR localWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    uint32_t scalesOffset_{0};
    uint32_t scalesStride_{0}; // token UB 槽内 scales 暂存偏移(32B 对齐, 与 GM slot 紧拼偏移解耦)
    uint32_t scalesElems_{0};
    uint32_t scalesBytes_{0};      // scales 的有效字节数，非 FP8 时为 0
    uint32_t scalesBytesAlign_{0}; // scales 按 UB block 对齐后的字节数，非 FP8 时为 0
    uint32_t metaOffset_{0};
    uint32_t metaBytes_{0};
    // 三处自管环的「一份」有多少个元素，用来做 份号 * 每份元素数 的偏移。
    uint32_t tokenRingElems_{0}; // Per-batch stride, including tokenBatchSlots_ token/scales rows.
    uint32_t tokenSlotElems_{0};
    uint32_t tokenBytes_{0};     // 单个 token 的有效 hidden 字节数，不含 UB 对齐与 scales
    uint32_t hAlignSize_{0};     // hidden 数据按 UB block 对齐后的字节数
    uint32_t tokenSlotBytes_{0}; // UB 中单个 token 的行宽，包含对齐与 scales 空间
    uint32_t tokenBatchSlots_{1U};
    uint32_t routeFrameSlots_{1U};
    uint32_t routeInputElems_{0};
    uint32_t routeInputBase_{0};
    uint32_t routeIdsBase_{0};
    uint32_t routeIdsElems_{0};
    uint32_t routeIdsSequence_[MOE_EP_BUFFER_NUM] = {}; // 由 InitSlotBuffers 填 TILE_NONE，不依赖份数为 2
    uint32_t routeOutputBase_{0};
    uint32_t stageSlotElems_{0};
    uint32_t metaRingElems_{0};
    uint32_t paddedMetaElems_{0};
    uint32_t axisKAlign_{0};
    uint32_t numLocalExperts_{0};
    // [rank][expert] 计数表里每跑一个 rank 的行步长。必须补到 ELEM_ALIGN 的整数倍：CopySlots 会按
    // prefixRank * rankExpertRowStride_ 切片交给 VEC，而 VEC 访问 UB 要求 32 字节对齐；numLocalExperts_ 本身没有
    // 对齐保证（host 侧就是 numExperts / epWorldSize，例：numExperts=4, epWorldSize=2 → 2）。
    uint32_t rankExpertRowStride_{0};
    uint32_t rankExpertCountStride_{0}; // One core row in int32 elements, padded per rank.
    uint32_t rankExpertCountBytes_{0};  // Byte size of one core row.
    uint32_t aivNum_{0};
    uint32_t axisK_{0};
    uint32_t axisH_{0};
    uint32_t numTokens_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t perSlotBytes_{0};
    uint32_t waitNotifyCount_{0};
    uint32_t waitNotifyBatchCount_{0};
    uint64_t winDataOffset_{0};
    uint64_t slotWinStateOffset_{0};
    MoeEpDispatchEpilogueRouteLayout routeLayout_{};
};

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t& startId, uint32_t& endId, uint32_t& sendNum)
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

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX, GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales,
    GM_ADDR workspace, GM_ADDR tilingGM, TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    numLocalExperts_ = tilingData->cfg.numLocalExperts;
    aivNum_ = tilingData->aivNum;
    networkMode_ = tilingData->networkMode;
    axisK_ = tilingData->cfg.topK;
    axisH_ = tilingData->cfg.hidden;
    tokenBytes_ = axisH_ * sizeof(XType);
    numTokens_ = tilingData->cfg.numTokens;
    epWorldSize_ = tilingData->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData->cfg.numMaxTokensPerRank;
    perSlotBytes_ = tilingData->cfg.perSlotBytes;
    winDataOffset_ = tilingData->winDataOffset;
    slotWinStateOffset_ = tilingData->slotWinStateOffset;

    mc2Context_ = context;
    epRankId_ = ClosOnly ? reinterpret_cast<__gm__ Mc2Kernel::MoeSharedJettyContext*>(context)->epRankId :
                           reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext*>(context)->epRankId;
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    // The local window covers states, payloads and dump records for either context layout.
    OOMCheckAddrRange<uint8_t>(tilingGM, sizeof(MoeEpDispatchEpilogueTilingData));
    OOMCheckAddrRange<uint8_t>(GetWinAddrByRankId(epRankId_, 0), tilingData->totalWinSizeEp);
#endif
    constexpr size_t metadataOffset = offsetof(MoeEpDispatchEpilogueTilingData, moeEpDispatchEpilogueInfo) +
                                      offsetof(MoeEpDispatchEpilogueInfo, dumpMetadata);
    if constexpr (ClosOnly) {
        MoeEpExceptionDump::WriteMetadata<Mc2Kernel::MoeSharedJettyContext>(context, tilingGM + metadataOffset);
        diagWriter_.Init<Mc2Kernel::MoeSharedJettyContext>(context, MOE_EP_CORE_DIAG_DISPATCH_EPILOGUE, tpipe_);
    } else {
        MoeEpExceptionDump::WriteMetadata<Mc2Aclnn::MoeCommContext>(context, tilingGM + metadataOffset);
        diagWriter_.Init<Mc2Aclnn::MoeCommContext>(context, MOE_EP_CORE_DIAG_DISPATCH_EPILOGUE, tpipe_);
    }
    uint32_t startRankId = 0U;
    uint32_t endRankId = epWorldSize_;
    if constexpr (ClosOnly) {
        uint32_t rankCount = 0U;
        SplitToCore(epWorldSize_, aivNum_, startRankId, endRankId, rankCount);
    }
    // Dispatch groups the flags by source rank, then by send core.
    uint32_t dispatchNotifyCount = tilingData->dispatchNotifyCount;
    waitNotifyCount_ = (endRankId - startRankId) * dispatchNotifyCount;
    waitNotifyBatchCount_ = waitNotifyCount_;
    if constexpr (ClosOnly) {
        const uint32_t maxBatchNotifyCount = WAIT_STATUS_RANKS_PER_BATCH * dispatchNotifyCount;
        waitNotifyBatchCount_ = waitNotifyCount_ < maxBatchNotifyCount ? waitNotifyCount_ : maxBatchNotifyCount;
    }
    uint64_t waitStatusOffset = static_cast<uint64_t>(startRankId) * dispatchNotifyCount * WIN_ADDR_ALIGN;
    localSlotStateWinAddr_ = GetWinAddrByRankId(epRankId_, slotWinStateOffset_ + waitStatusOffset);
    localWinAddr_ = GetWinAddrByRankId(epRankId_, winDataOffset_);
    // 槽内 scales/meta 基址: direct 紧拼 tokenSize; hybrid 为 ALIGN32(tokenSize)（hybrid dispatch 整槽布局）
    hAlignSize_ = Ceil(tokenBytes_, UB_ALIGN) * UB_ALIGN;
    uint32_t slotMetaBase = (tilingData->networkMode == NETWORK_HYBRID) ? hAlignSize_ : tokenBytes_;
    metaOffset_ = slotMetaBase;
    scalesStride_ = hAlignSize_ / sizeof(XType);

    numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert);
    cachedRecvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)cachedRecvSrcMetadata);
    if constexpr (IsCached) {
        cachedRecvRankOffsetsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cachedRecvSrcMetadata + tilingData->metadataRankOffsetsOffset));
        cachedLocalRecvIndexGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cachedRecvSrcMetadata + tilingData->localRecvIndexOffset));
    }
    slotEndsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvSrcMetadata + tilingData->slotEndsOffset));
    slotRecvXIdsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(recvSrcMetadata + tilingData->slotRecvXIdsOffset));
    slotTopkIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvSrcMetadata + tilingData->slotTopkIdsOffset));
    if constexpr (IsCached) {
        cachedSlotEndsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cachedRecvSrcMetadata + tilingData->slotEndsOffset));
        cachedSlotRecvXIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cachedRecvSrcMetadata + tilingData->slotRecvXIdsOffset));
        cachedSlotTopkIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cachedRecvSrcMetadata + tilingData->slotTopkIdsOffset));
    }
    xGm_.SetGlobalBuffer((__gm__ XType*)x);
    recvXGm_.SetGlobalBuffer((__gm__ XType*)recvX);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata);
    recvRankOffsetsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(recvSrcMetadata + tilingData->metadataRankOffsetsOffset));
    localRecvIndexGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(recvSrcMetadata + tilingData->localRecvIndexOffset));
    if constexpr (HasTopkWeights) {
        recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights);
    }

    // joint 计数矩阵占本 op 用户 workspace 的最前面一段，所以不需要偏移量，直接绑 workspace 首地址。
    rankExpertHitCountGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));

    axisKAlign_ = Ceil(axisK_, ELEM_ALIGN) * ELEM_ALIGN;
    metaBytes_ = (MOE_EP_SLOT_META_SECTIONS * axisKAlign_) * (uint32_t)sizeof(int32_t) + UB_ALIGN;
    uint32_t ubExpertPfxBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t waitStatusBytes = waitNotifyBatchCount_ * UB_ALIGN;
    tpipe_->InitBuffer(sharedPhaseBuf_, SHARED_PHASE_BYTES);
    tpipe_->InitBuffer(ubExpertPfxBuf_, ubExpertPfxBytes);
    ubExpertPfx_ = ubExpertPfxBuf_.Get<int64_t>();
    // 归约的输入、结果和scratch在本阶段互不重叠；结束后整块空间交给搬入流水。
    ubMeta_ = sharedPhaseBuf_.Get<int32_t>();
    ubWaitStatus_ = ubMeta_;
    ubWaitSum_ = ubMeta_[waitStatusBytes / sizeof(int32_t)];
    ubSharedTmp_ = sharedPhaseBuf_.Get<uint8_t>()[waitStatusBytes + UB_ALIGN];
    uint32_t ubRankOffsetsBytes = Ceil((epWorldSize_ + 1U) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubRankOffsetsBuf_, ubRankOffsetsBytes);
    ubRankOffsets_ = ubRankOffsetsBuf_.Get<int32_t>();

    scalesBytes_ = 0U;
    scalesBytesAlign_ = 0U;
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesOffset_ = slotMetaBase;
        scalesBytes_ = tilingData->cfg.scalesBytes;
        scalesElems_ = scalesBytes_ / sizeof(ScalesType);
        scalesBytesAlign_ = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
        metaOffset_ += scalesBytesAlign_;
        recvScalesGm_.SetGlobalBuffer((__gm__ ScalesType*)recvScales);
    }

    InitSlotBuffers(hAlignSize_ + scalesBytesAlign_, tilingData);
    event_t event = static_cast<event_t>(SYNC_EVENT_ID7);
    LocalTensor<int64_t> expertReduceTmp = ubSharedTmp_.ReinterpretCast<int64_t>();
    LocalTensor<int64_t> expertSumOut = ubWaitSum_.ReinterpretCast<int64_t>();
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubExpertPfx_.GetPhyAddr(),
                         (__gm__ uint8_t*)numRecvPerExpertGm_.GetPhyAddr(), numLocalExperts_ * sizeof(int64_t));
    asc_sync_notify(PIPE_MTE2, PIPE_V, event);
    asc_sync_wait(PIPE_MTE2, PIPE_V, event);
    ReduceSum<int64_t>(expertSumOut, ubExpertPfx_, expertReduceTmp, static_cast<int32_t>(numLocalExperts_));
    asc_sync_notify(PIPE_V, PIPE_S, event);
    asc_sync_wait(PIPE_V, PIPE_S, event);
    expertSum_ = static_cast<uint32_t>(expertSumOut.GetValue(0));
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_INIT_DONE);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::InitSlotBuffers(
    uint32_t tokenSlotBytes, const MoeEpDispatchEpilogueInfo* tilingData)
{
    tokenSlotBytes_ = tokenSlotBytes;
    rankExpertRowStride_ = Ceil(numLocalExperts_, ELEM_ALIGN) * ELEM_ALIGN;
    rankExpertCountStride_ = rankExpertRowStride_ * epWorldSize_;
    rankExpertCountBytes_ = rankExpertCountStride_ * sizeof(int32_t);
    paddedMetaElems_ =
        Ceil(MOE_EP_SLOT_META_SECTIONS * axisKAlign_ + MOE_EP_SLOT_META_EXTRA_FIELDS, ELEM_ALIGN) * ELEM_ALIGN;
    metaRingElems_ = SHARED_HALF_BYTES / sizeof(int32_t);
    // Compact IDs use K int32s per slot; each CountHits buffer owns a full 32 KiB half.
    countSlotsPerTile_ = metaRingElems_ / axisK_;
    stageSlotElems_ = axisK_ * ELEM_ALIGN;
    uint32_t expertBytes = Ceil(numLocalExperts_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t countBytes = Ceil(numLocalExperts_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t recvCountBytes = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t stageBytes = stageSlotElems_ * sizeof(int32_t) * STAGE_RING;
    if constexpr (IsCached) {
        // Each half holds one cached input frame: ends, recv-X IDs, optional top-k IDs and weights,
        // and token IDs for direct transfers from the same rank.
        // Aligning the slot count keeps ID columns aligned. Only the extra end entry needs padding,
        // so frameElems = slots * perSlotElems + ELEM_ALIGN.
        const uint32_t perSlotElems = 1U + (HasTopkWeights ? MOE_EP_SLOT_META_SECTIONS * axisK_ : axisK_) +
                                      (networkMode_ != NETWORK_HYBRID ? ELEM_ALIGN : 0U) +
                                      (HasTopkWeights ? axisKAlign_ : 0U);
        // The host bounds top-k so one aligned group of slots always fits in the half buffer.
        const uint32_t maxSlots = (metaRingElems_ - ELEM_ALIGN) / perSlotElems / ELEM_ALIGN * ELEM_ALIGN;
        slotsPerTile_ = maxSlots < SLOTS_TILE ? maxSlots : SLOTS_TILE;
        planEndsElems_ = slotsPerTile_ + ELEM_ALIGN;
        const uint32_t entries = slotsPerTile_ * axisK_;
        cachedTopkBase_ = planEndsElems_ + entries;
        cachedIdsBase_ = cachedTopkBase_ + (HasTopkWeights ? entries : 0U);
        cachedWeightsBase_ = cachedIdsBase_ + (networkMode_ != NETWORK_HYBRID ? slotsPerTile_ * ELEM_ALIGN : 0U);
    }
    tokenBatchSlots_ = tilingData->tokenBatch;
    tokenSlotElems_ = tokenSlotBytes_ / sizeof(XType);
    tokenRingElems_ = tokenSlotElems_ * tokenBatchSlots_;
    if constexpr (!IsCached) {
        routeFrameSlots_ = tilingData->routeFrameSlots;
        routeLayout_ = tilingData->routeLayout;
        routeIdsBase_ = Ceil(MOE_EP_ROUTE_SCRATCH_TABLES * numLocalExperts_, ELEM_ALIGN) * ELEM_ALIGN;
        routeIdsElems_ = Ceil(routeFrameSlots_, ELEM_ALIGN) * ELEM_ALIGN;
        routeInputBase_ = routeIdsBase_ + MOE_EP_BUFFER_NUM * routeIdsElems_;
        routeInputElems_ = routeFrameSlots_ * paddedMetaElems_;
        routeOutputBase_ = routeInputBase_ + MOE_EP_BUFFER_NUM * routeInputElems_;
    }
    tpipe_->InitBuffer(ubRecvCntBuf_, recvCountBytes);
    tpipe_->InitBuffer(tokenRingBuf_, tokenSlotBytes_ * tokenBatchSlots_ * MOE_EP_BUFFER_NUM);
    ubRecvCnt_ = ubRecvCntBuf_.Get<int32_t>();
    tokenRing_ = tokenRingBuf_.Get<XType>();
    if constexpr (IsCached && HasTopkWeights) {
        tpipe_->InitBuffer(ubStageWeightsRingBuf_, stageBytes);
        ubStageWeightsRing_ = ubStageWeightsRingBuf_.Get<float>();
    }
    if constexpr (!IsCached) {
        tpipe_->InitBuffer(ubHitCountBuf_, countBytes);
        tpipe_->InitBuffer(ubRankExpertHitCountBuf_, rankExpertCountBytes_);
        tpipe_->InitBuffer(ubRankExpertRowStartBuf_, rankExpertCountBytes_);
        tpipe_->InitBuffer(ubRowStartBuf_, expertBytes);
        tpipe_->InitBuffer(ubHitCountRowI64Buf_, expertBytes);
        tpipe_->InitBuffer(ubStageLocalIndexRingBuf_, stageBytes);
        tpipe_->InitBuffer(ubLocalCursorBuf_, countBytes);
        ubHitCount_ = ubHitCountBuf_.Get<int32_t>();
        ubRankExpertHitCount_ = ubRankExpertHitCountBuf_.Get<int32_t>();
        ubRankExpertRowStart_ = ubRankExpertRowStartBuf_.Get<int32_t>();
        ubRowStart_ = ubRowStartBuf_.Get<int64_t>();
        ubHitCountRowI64_ = ubHitCountRowI64Buf_.Get<int64_t>();
        ubStageLocalIndexRing_ = ubStageLocalIndexRingBuf_.Get<int32_t>();
        ubLocalCursor_ = ubLocalCursorBuf_.Get<int32_t>();
    }
    for (uint32_t i = 0; i < MOE_EP_BUFFER_NUM; ++i) {
        cachedIdsSequence_[i] = TILE_NONE;
        routeIdsSequence_[i] = TILE_NONE;
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void
MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::InitLocalRecvIndex()
{
    const uint32_t totalElements = numTokens_ * axisK_;
    if constexpr (IsCached) {
        // 路由未变，共用批量搬运函数；此时metadata缓冲尚未用于预取。
        CopyCachedArray(cachedLocalRecvIndexGm_, localRecvIndexGm_, totalElements);
    } else {
        uint32_t startBlock, endBlock, blockCount;
        SplitToCore(Ceil(totalElements, ELEM_ALIGN), aivNum_, startBlock, endBlock, blockCount);
        uint32_t start = startBlock * ELEM_ALIGN;
        uint32_t end = endBlock * ELEM_ALIGN > totalElements ? totalElements : endBlock * ELEM_ALIGN;
        if (start < end) {
            const uint32_t fillElements = end - start < metaRingElems_ ? end - start : metaRingElems_;
            Duplicate<int32_t>(ubMeta_, -1, fillElements);
            asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID4));
            asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID4));
            for (uint32_t offset = start; offset < end; offset += fillElements) {
                const uint32_t elements = end - offset < fillElements ? end - offset : fillElements;
                asc_copy_ub2gm_align((__gm__ uint8_t*)localRecvIndexGm_[offset].GetPhyAddr(),
                                     (__ubuf__ uint8_t*)ubMeta_.GetPhyAddr(), elements * sizeof(int32_t));
            }
        }
    }
    asc_sync_notify(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID5));
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::Process()
{
    InitLocalRecvIndex();
    if constexpr (!IsCached) {
        ComputePrefixSums();
    }
    // The fill and Scalar prefix construction use different UB regions. Wait before shared-UB reuse.
    asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID5));
    WaitDispatch();
    SyncAll<true>();
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_WAIT_DONE);

    if constexpr (!IsCached) {
        CountHits();
        SyncAll<true>();
        // 每核根据共享计数生成自己的metadata起点，结果留在UB，无须再次核间同步。
        BuildRankRowStarts();
        IssueRouteAndCopySlots();
    } else {
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubRecvCnt_.GetPhyAddr(),
                             (__gm__ uint8_t*)numRecvPerRankGm_.GetPhyAddr(), epWorldSize_ * sizeof(int32_t));
        asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID4));
        asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID4));
        // CopySlotsCached starts the handle clone before draining the payload/stage tails.
        CopySlotsCached();
    }

    // 所有输出在交给Combine前写完，包含metadata及slot索引。
    asc_sync_notify(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID7));
    asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID7));
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_OUTPUT_DONE);
    if constexpr (ClosOnly) {
        MoeEpCompletion::DrainChannels(reinterpret_cast<__gm__ Mc2Kernel::MoeSharedJettyContext*>(mc2Context_),
                                       epWorldSize_, tpipe_);
    } else {
        MoeEpCompletion::DrainChannels(reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext*>(mc2Context_), epWorldSize_,
                                       tpipe_);
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::WaitDispatch()
{
    if constexpr (ClosOnly) {
        uint32_t remaining = waitNotifyCount_;
        GM_ADDR slotStateAddr = localSlotStateWinAddr_;
        for (uint32_t offset = 0U; offset < waitNotifyCount_; offset += waitNotifyBatchCount_) {
            uint32_t notifyCount = remaining < waitNotifyBatchCount_ ? remaining : waitNotifyBatchCount_;
            WaitDispatchBatch(slotStateAddr, notifyCount);
            remaining -= notifyCount;
            slotStateAddr += notifyCount * WIN_ADDR_ALIGN;
        }
    } else {
        uint32_t notifyCount = aivId_ == aivNum_ - 1 ? waitNotifyCount_ : 0;
        WaitDispatchBatch(localSlotStateWinAddr_, notifyCount);
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::WaitDispatchBatch(
    GM_ADDR statusAddr, uint32_t notifyCount)
{
    if (notifyCount == 0) {
        return;
    }
    uint32_t mask = 1;
    int32_t sumOfFlag = 0;
    int32_t commpareFlag = static_cast<int32_t>(notifyCount);
    event_t waitEvent = static_cast<event_t>(SYNC_EVENT_ID7);
    GlobalTensor<int32_t> statusGMTensor;
    LocalTensor<float> sharedTmp = ubSharedTmp_.ReinterpretCast<float>();
    LocalTensor<float> ubWaitStatusFp32 = ubWaitStatus_.template ReinterpretCast<float>();
    LocalTensor<float> ubWaitSumFp32 = ubWaitSum_.template ReinterpretCast<float>();
    statusGMTensor.SetGlobalBuffer((__gm__ int32_t*)statusAddr);

    asc_sync_notify(PIPE_S, PIPE_V, waitEvent);
    asc_sync_wait(PIPE_S, PIPE_V, waitEvent); // 确保expertSum_计算完成
    while (sumOfFlag != commpareFlag) {
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubWaitStatus_.GetPhyAddr(),
                             (__gm__ uint8_t*)statusGMTensor.GetPhyAddr(), notifyCount, UB_ALIGN, 0U, 0U, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, WIN_ADDR_ALIGN, UB_ALIGN);
        asc_sync_notify(PIPE_MTE2, PIPE_V, waitEvent);
        asc_sync_wait(PIPE_MTE2, PIPE_V, waitEvent);
        ReduceSum(ubWaitSumFp32, ubWaitStatusFp32, sharedTmp, mask, notifyCount, 1);
        asc_sync_notify(PIPE_V, PIPE_S, waitEvent);
        asc_sync_wait(PIPE_V, PIPE_S, waitEvent);
        sumOfFlag = ubWaitSum_.GetValue(0);
    }
    Duplicate<int32_t>(ubWaitStatus_, 0, notifyCount * UB_ALIGN / sizeof(int32_t));
    asc_sync_notify(PIPE_V, PIPE_MTE3, waitEvent);
    asc_sync_wait(PIPE_V, PIPE_MTE3, waitEvent);
    asc_copy_ub2gm_align((__gm__ uint8_t*)statusGMTensor.GetPhyAddr(), (__ubuf__ uint8_t*)ubWaitStatus_.GetPhyAddr(),
                         notifyCount, UB_ALIGN, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, WIN_ADDR_ALIGN, UB_ALIGN);
    // Dispatch reuses these notification slots in the next continue round.  SyncAll only synchronizes AIVs; it does
    // not replace the MTE3 completion required before a later kernel can publish the next round's notifications.
    asc_sync_notify(PIPE_MTE3, PIPE_S, waitEvent);
    asc_sync_wait(PIPE_MTE3, PIPE_S, waitEvent);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::ComputePrefixSums()
{
    int64_t cumulativeRowOffset = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        int64_t expertTokenCnt = ubExpertPfx_.GetValue(localExpertIdx);
        ubExpertPfx_.SetValue(localExpertIdx, cumulativeRowOffset);
        cumulativeRowOffset += expertTokenCnt;
    }
    asc_sync_notify(PIPE_S, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID7));
    asc_sync_wait(PIPE_S, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID7));
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::IssueTopkIds(
    uint32_t rankId, uint32_t slotIdx, uint32_t tileCnt, uint32_t buf)
{
    GlobalTensor<int32_t> srcTopkIdsGm;
    srcTopkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        localWinAddr_ + (static_cast<int64_t>(rankId) * numMaxTokensPerRank_ + slotIdx) * perSlotBytes_ + metaOffset_));
    const uint32_t topkBytes = axisK_ * sizeof(int32_t);
    // 每slot仅搬K个ID，UB中紧密排列；VF按tileCnt*K屏蔽整批末尾的无效lane。
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_[buf * metaRingElems_].GetPhyAddr(),
                         (__gm__ uint8_t*)srcTopkIdsGm.GetPhyAddr(), tileCnt, topkBytes, 0U, 0U, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, perSlotBytes_, topkBytes);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::CountHits()
{
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubRecvCnt_.GetPhyAddr(), (__gm__ uint8_t*)numRecvPerRankGm_.GetPhyAddr(),
                         epWorldSize_ * sizeof(int32_t));
    Duplicate(ubRankExpertHitCount_, (int32_t)0, rankExpertCountStride_);
    asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID4));
    asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID4));

    uint32_t rankId = 0U;
    uint32_t tileStart = 0U;
    uint32_t tileCnt = 0U;
    uint32_t slotStart = 0U;
    uint32_t nextRank = 0U;
    uint32_t nextTileStart = 0U;
    uint32_t nextTileCnt = 0U;
    uint32_t nextSlotStart = 0U;
    uint32_t sequence = 0U;
    bool hasTile = NextTile(0U, TILE_NONE, rankId, tileStart, tileCnt, slotStart, countSlotsPerTile_);
    if (hasTile) {
        IssueTopkIds(rankId, slotStart + tileStart, tileCnt, 0U);
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID0));
    }
    while (hasTile) {
        const uint32_t buf = sequence % MOE_EP_BUFFER_NUM;
        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID0 + buf));
        hasTile = NextTile(rankId, tileStart, nextRank, nextTileStart, nextTileCnt, nextSlotStart, countSlotsPerTile_);
        if (hasTile) {
            const uint32_t nextBuf = (sequence + 1U) % MOE_EP_BUFFER_NUM;
            if (sequence + 1U >= MOE_EP_BUFFER_NUM) {
                // 只在覆盖旧半区前等Vector归还，另一半区的当前统计仍可并行。
                asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + nextBuf));
            }
            IssueTopkIds(nextRank, nextSlotStart + nextTileStart, nextTileCnt, nextBuf);
            asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID0 + nextBuf));
        }
        // 预取可跨rank，计数目标始终属于当前任务，不能使用nextRank。
        asc_vf_call<MoeEpDispatchEpilogueCount::CountTopkHits>(
            (__ubuf__ int32_t*)ubMeta_[buf * metaRingElems_].GetPhyAddr(),
            (__ubuf__ int32_t*)ubRankExpertHitCount_[rankId * rankExpertRowStride_].GetPhyAddr(), tileCnt * axisK_,
            numLocalExperts_, epRankId_ * numLocalExperts_);
        asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + buf));
        ++sequence;
        rankId = nextRank;
        tileStart = nextTileStart;
        tileCnt = nextTileCnt;
        slotStart = nextSlotStart;
    }
    // 最后每个使用过的半区各剩一次归还；空任务不发也不等，所有事件在本阶段配平。
    for (uint32_t buf = 0; buf < MOE_EP_BUFFER_NUM && buf < sequence; ++buf) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + buf));
    }

    event_t syncEvent = static_cast<event_t>(SYNC_EVENT_ID7);
    asc_sync_notify(PIPE_V, PIPE_MTE3, syncEvent);
    asc_sync_wait(PIPE_V, PIPE_MTE3, syncEvent);
    asc_sync_notify(PIPE_S, PIPE_MTE3, syncEvent);
    asc_sync_wait(PIPE_S, PIPE_MTE3, syncEvent);
    // 这里只输出联合计数；ubHitCount_在CopySlots准备阶段从此前核的前缀重新归约。
    asc_copy_ub2gm_align((__gm__ uint8_t*)rankExpertHitCountGm_[(int64_t)aivId_ * rankExpertCountStride_].GetPhyAddr(),
                         (__ubuf__ uint8_t*)ubRankExpertHitCount_.GetPhyAddr(), rankExpertCountBytes_);
    asc_sync_notify(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID5));
    asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID5));
    // Finish pending reuse waits before the route phase reuses IDs 0 and 1.
    asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID4));
    asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID4));
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void
MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::BuildRankRowStarts()
{
    // CountHits used ubRankExpertHitCount_ as an MTE3 source. Finish that read before reusing the buffer as the
    // per-rank prefix contributed by cores preceding this core.
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID7));
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID7));

    // H[c][r][e] counts expanded records. Reduce all cores and the cores before this AIV separately.
    // Flat rows are [rank][expert], padded only at the end of each core row.
    Duplicate(ubRankExpertRowStart_, (int32_t)0, rankExpertCountStride_);
    Duplicate(ubRankExpertHitCount_, (int32_t)0, rankExpertCountStride_);

    // Use both metadata halves: prefetch the next core-row tile while Vector reduces the current one.
    const uint32_t coreRowsPerTile = metaRingElems_ * sizeof(int32_t) / rankExpertCountBytes_;
    const uint32_t firstRows = aivNum_ < coreRowsPerTile ? aivNum_ : coreRowsPerTile;
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_.GetPhyAddr(), (__gm__ uint8_t*)rankExpertHitCountGm_.GetPhyAddr(),
                         firstRows * rankExpertCountBytes_);
    asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID2));

    uint32_t prefixSequence = 0U;
    for (uint32_t coreBase = 0; coreBase < aivNum_; coreBase += coreRowsPerTile) {
        const uint32_t bufferIdx = prefixSequence % MOE_EP_BUFFER_NUM;
        const uint32_t rowsThisTile = aivNum_ - coreBase < coreRowsPerTile ? aivNum_ - coreBase : coreRowsPerTile;
        const uint32_t nextCoreBase = coreBase + rowsThisTile;
        if (nextCoreBase < aivNum_) {
            const uint32_t nextBuffer = (prefixSequence + 1U) % MOE_EP_BUFFER_NUM;
            if (prefixSequence + 1U >= MOE_EP_BUFFER_NUM) {
                asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID2 + nextBuffer));
            }
            const uint32_t nextRows =
                aivNum_ - nextCoreBase < coreRowsPerTile ? aivNum_ - nextCoreBase : coreRowsPerTile;
            asc_copy_gm2ub_align(
                (__ubuf__ uint8_t*)ubMeta_[nextBuffer * metaRingElems_].GetPhyAddr(),
                (__gm__ uint8_t*)rankExpertHitCountGm_[static_cast<int64_t>(nextCoreBase) * rankExpertCountStride_]
                    .GetPhyAddr(),
                nextRows * rankExpertCountBytes_);
            asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID2 + nextBuffer));
        }
        // Only the current half must be ready; next-half DMA can overlap this reduction.
        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID2 + bufferIdx));
        for (uint32_t localCoreId = 0; localCoreId < rowsThisTile; ++localCoreId) {
            const uint32_t coreId = coreBase + localCoreId;
            LocalTensor<int32_t> jointCountRow =
                ubMeta_[bufferIdx * metaRingElems_ + localCoreId * rankExpertCountStride_];
            Add(ubRankExpertRowStart_, ubRankExpertRowStart_, jointCountRow, rankExpertCountStride_);
            if (coreId < aivId_) {
                Add(ubRankExpertHitCount_, ubRankExpertHitCount_, jointCountRow, rankExpertCountStride_);
            }
        }
        asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID2 + bufferIdx));
        ++prefixSequence;
    }
    // Route input/scratch reuse shared UB after every prefix reader has returned its half.
    const uint32_t prefixTail = prefixSequence < MOE_EP_BUFFER_NUM ? prefixSequence : MOE_EP_BUFFER_NUM;
    for (uint32_t tail = prefixSequence - prefixTail; tail < prefixSequence; ++tail) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID2 + tail % MOE_EP_BUFFER_NUM));
    }

    // P(c,r,e) = offsets[r] + sum_{e'<e,c'} H[c'][r][e'] + sum_{c'<c} H[c'][r][e].
    // recv_x stays expert/core/rank ordered, so (rank, expert, core, per-group cursor) is (rank, recv_x_idx) order.
    // 总计数做整数前缀和，再加此前core的计数；保持原metadata与recvXIdx顺序。
    asc_vf_call<MoeEpDispatchEpilogueCount::BuildMetadataStarts>(
        (__ubuf__ int32_t*)ubRankExpertRowStart_.GetPhyAddr(), (__ubuf__ int32_t*)ubRankExpertHitCount_.GetPhyAddr(),
        (__ubuf__ int32_t*)ubRankOffsets_.GetPhyAddr(), epWorldSize_, rankExpertRowStride_);
    if (aivId_ == 0U) {
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID7));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID7));
        asc_copy_ub2gm_align((__gm__ uint8_t*)recvRankOffsetsGm_.GetPhyAddr(),
                             (__ubuf__ uint8_t*)ubRankOffsets_.GetPhyAddr(), (epWorldSize_ + 1U) * sizeof(int32_t));
        // rankOffsets暂存不再复用，由Process末尾统一等待写出完成。
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void
MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::CopyCachedRankOffsets()
{
    if (aivId_ != 0U) {
        return;
    }

    uint32_t copyLen = (epWorldSize_ + 1U) * sizeof(int32_t);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubRankOffsets_.GetPhyAddr(),
                         (__gm__ uint8_t*)cachedRecvRankOffsetsGm_.GetPhyAddr(), copyLen);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID4));
    asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID4));
    asc_copy_ub2gm_align((__gm__ uint8_t*)recvRankOffsetsGm_.GetPhyAddr(),
                         (__ubuf__ uint8_t*)ubRankOffsets_.GetPhyAddr(), copyLen);
    // ubRankOffsets_ stays immutable; Process waits for all output DMA before returning.
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline bool MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::NextTile(
    uint32_t rankId, uint32_t tileStart, uint32_t& nextRank, uint32_t& nextTileStart, uint32_t& nextTileCnt,
    uint32_t& nextSlotStart, uint32_t slotsPerTile)
{
    for (uint32_t r = rankId; r < epWorldSize_; ++r) {
        int32_t slotCnt = ubRecvCnt_.GetValue(r);
        uint32_t slotStart, slotEnd, slotCntPerAiv;
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        // 同一个 rank 内接着上一拍往后走；跨到下一个 rank 就从它的第 0 拍开始。TILE_NONE 表示还没发过任何一拍。
        uint32_t start = (r == rankId && tileStart != TILE_NONE) ? (tileStart + slotsPerTile) : 0U;
        if (start >= slotCntPerAiv) {
            continue;
        }
        nextRank = r;
        nextTileStart = start;
        nextTileCnt = (slotCntPerAiv - start > slotsPerTile) ? slotsPerTile : (slotCntPerAiv - start);
        nextSlotStart = slotStart;
        return true;
    }
    return false;
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::IssueMeta(
    uint32_t rankId, uint32_t slotIdx, uint32_t tileCnt, uint32_t buf)
{
    const uint32_t base = buf * metaRingElems_;
    const uint32_t globalSlot = GlobalSlot(rankId, slotIdx);
    const uint32_t firstEnd = globalSlot == 0U ? 0U : globalSlot - 1U;
    const uint32_t prefix = globalSlot == 0U ? 0U : PLAN_END_EXTRA;
    const uint32_t begin = globalSlot == 0U ? 0U : cachedSlotEndsGm_.GetValue(firstEnd);
    const uint32_t end = cachedSlotEndsGm_.GetValue(globalSlot + tileCnt - 1U);
    planRowBegin_[buf] = begin;
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_[base].GetPhyAddr(),
                         (__gm__ uint8_t*)cachedSlotEndsGm_[firstEnd].GetPhyAddr(),
                         (tileCnt + prefix) * sizeof(int32_t));
    if (end > begin) {
        uint32_t copyLen = (end - begin) * sizeof(int32_t);
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_[base + planEndsElems_].GetPhyAddr(),
                             (__gm__ uint8_t*)cachedSlotRecvXIdsGm_[begin].GetPhyAddr(), copyLen);
        if constexpr (HasTopkWeights) {
            asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_[base + cachedTopkBase_].GetPhyAddr(),
                                 (__gm__ uint8_t*)cachedSlotTopkIdsGm_[begin].GetPhyAddr(), copyLen);
        }
    }
    if constexpr (HasTopkWeights) {
        GlobalTensor<int32_t> weights;
        weights.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            localWinAddr_ + (static_cast<int64_t>(rankId) * numMaxTokensPerRank_ + slotIdx) * perSlotBytes_ +
            metaOffset_ + axisKAlign_ * sizeof(int32_t)));
        const uint32_t bytes = axisK_ * sizeof(float);
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_[base + cachedWeightsBase_].GetPhyAddr(),
                             (__gm__ uint8_t*)weights.GetPhyAddr(), tileCnt, bytes, 0U, 0U, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, perSlotBytes_, axisKAlign_ * sizeof(float));
    }
    asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID0 + buf));
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::IssueRouteInput(
    uint32_t rank, uint32_t slot, uint32_t count, uint32_t sequence)
{
    uint32_t bufferIdx = sequence % MOE_EP_BUFFER_NUM;
    if (sequence >= MOE_EP_BUFFER_NUM) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    }
    GlobalTensor<int32_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        localWinAddr_ + (static_cast<int64_t>(rank) * numMaxTokensPerRank_ + slot) * perSlotBytes_ + metaOffset_));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_[routeInputBase_ + bufferIdx * routeInputElems_].GetPhyAddr(),
                         (__gm__ uint8_t*)source.GetPhyAddr(), count, metaBytes_, 0U, 0U, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, perSlotBytes_,
                         paddedMetaElems_ * sizeof(int32_t));
    asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    if (networkMode_ != NETWORK_HYBRID && rank == epRankId_) {
        asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::BuildRouteInfo(
    uint32_t rank, uint32_t count, uint32_t planBase, uint32_t sequence)
{
    uint32_t bufferIdx = sequence % MOE_EP_BUFFER_NUM;
    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    if (sequence >= MOE_EP_BUFFER_NUM) {
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    }
    asc_vf_call<MoeEpDispatchEpilogueRoute::BuildRoutePlan<HasTopkWeights>>(
        (__ubuf__ int32_t*)ubMeta_[routeInputBase_ + bufferIdx * routeInputElems_].GetPhyAddr(),
        (__ubuf__ int32_t*)ubMeta_[routeOutputBase_ + bufferIdx * routeLayout_.size].GetPhyAddr(),
        (__ubuf__ int32_t*)ubMeta_.GetPhyAddr(),
        (__ubuf__ int32_t*)ubRowStart_.template ReinterpretCast<int32_t>().GetPhyAddr(),
        (__ubuf__ int32_t*)ubLocalCursor_.GetPhyAddr(),
        (__ubuf__ int32_t*)ubRankExpertRowStart_[rank * rankExpertRowStride_].GetPhyAddr(), routeLayout_, count, axisK_,
        axisKAlign_, paddedMetaElems_, numLocalExperts_, epRankId_ * numLocalExperts_, planBase);
    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    asc_sync_notify(PIPE_V, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void
MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::IssueRouteAndCopySlots()
{
    // Keep the original deterministic expert/core/rank and rank/expert/core bases.
    Duplicate(ubHitCount_, (int32_t)0, numLocalExperts_);
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        Add(ubHitCount_, ubHitCount_, ubRankExpertHitCount_[r * rankExpertRowStride_], numLocalExperts_);
    }
    Cast(ubHitCountRowI64_, ubHitCount_, RoundMode::CAST_NONE, numLocalExperts_);
    Add(ubRowStart_, ubExpertPfx_, ubHitCountRowI64_, numLocalExperts_);
    Duplicate(ubLocalCursor_, (int32_t)0, numLocalExperts_);
    asc_vf_call<MoeEpDispatchEpilogueCount::ReduceRankCorePrefixes>(
        (__ubuf__ int32_t*)ubRankExpertHitCount_.GetPhyAddr(), epWorldSize_, rankExpertRowStride_);

    uint32_t rank = 0U;
    uint32_t start = 0U;
    uint32_t count = 0U;
    uint32_t coreStart = 0U;
    uint32_t nextRank = 0U;
    uint32_t nextTileStart = 0U;
    uint32_t nextTileCount = 0U;
    uint32_t nextCoreStart = 0U;
    uint32_t sequence = 0U;
    uint32_t tokenSequence = 0U;
    uint32_t tokenIssued = 0U;
    uint32_t indexSequence = 0U;
    bool hasTile = NextTile(0U, TILE_NONE, rank, start, count, coreStart, routeFrameSlots_);
    if (!hasTile) {
        return;
    }
    IssueRouteInput(rank, coreStart + start, count, 0U);
    // Prime payload before waiting for ANY route VF. Only source IDs are needed for direct-self.
    IssueRouteTokens(rank, coreStart + start, count, 0U, 0U, tokenIssued++);
    if (count > tokenBatchSlots_) {
        IssueRouteTokens(rank, coreStart + start, count, tokenBatchSlots_, 0U, tokenIssued++);
    }
    asc_sync_notify(PIPE_V, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID7));
    asc_sync_wait(PIPE_V, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID7));
    uint32_t planBase = ubRankOffsets_.GetValue(rank) + ubRankExpertHitCount_.GetValue(rank * rankExpertRowStride_);
    BuildRouteInfo(rank, count, planBase, 0U);
    while (hasTile) {
        bool hasNext = NextTile(rank, start, nextRank, nextTileStart, nextTileCount, nextCoreStart, routeFrameSlots_);
        if (hasNext) {
            // Route prefetch precedes payload buffer-reuse waits on the MTE2 FIFO.
            IssueRouteInput(nextRank, nextCoreStart + nextTileStart, nextTileCount, sequence + 1U);
            if (count <= tokenBatchSlots_ && tokenIssued == tokenSequence + 1U &&
                (networkMode_ == NETWORK_HYBRID || nextRank != epRankId_)) {
                IssueRouteTokens(nextRank, nextCoreStart + nextTileStart, nextTileCount, 0U, sequence + 1U,
                                 tokenIssued++);
            }
        }
        asc_sync_wait(PIPE_V, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID0 + sequence % MOE_EP_BUFFER_NUM));
        uint32_t hits = static_cast<uint32_t>(
            ubMeta_.GetValue(routeOutputBase_ + (sequence % MOE_EP_BUFFER_NUM) * routeLayout_.size));
        uint32_t nextPlanBase = 0U;
        if (hasNext) {
            nextPlanBase = nextRank == rank ? planBase + hits :
                                              ubRankOffsets_.GetValue(nextRank) +
                                                  ubRankExpertHitCount_.GetValue(nextRank * rankExpertRowStride_);
            // VF(n+1) overlaps consumption of n. Consumption of n-1 has returned before
            // this call, so its Scalar plan reads are already finished before slot reuse.
            BuildRouteInfo(nextRank, nextTileCount, nextPlanBase, sequence + 1U);
        }
        ConsumeRoutePlan(rank, coreStart + start, count, planBase, sequence, tokenSequence, tokenIssued, indexSequence,
                         hasNext, nextRank, nextCoreStart + nextTileStart, nextTileCount);
        ++sequence;
        hasTile = hasNext;
        rank = nextRank;
        start = nextTileStart;
        count = nextTileCount;
        coreStart = nextCoreStart;
        planBase = nextPlanBase;
    }
    const uint32_t routeTail = sequence < MOE_EP_BUFFER_NUM ? sequence : MOE_EP_BUFFER_NUM;
    for (uint32_t tail = sequence - routeTail; tail < sequence; ++tail) {
        event_t eventId = static_cast<event_t>(SYNC_EVENT_ID0 + tail % MOE_EP_BUFFER_NUM);
        asc_sync_wait(PIPE_V, PIPE_MTE2, eventId);
        asc_sync_wait(PIPE_MTE3, PIPE_V, eventId);
    }
    const uint32_t tokenTail = tokenSequence < MOE_EP_BUFFER_NUM ? tokenSequence : MOE_EP_BUFFER_NUM;
    for (uint32_t tail = tokenSequence - tokenTail; tail < tokenSequence; ++tail) {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + tail % MOE_EP_BUFFER_NUM));
    }
    const uint32_t stageTail = indexSequence < STAGE_RING ? indexSequence : STAGE_RING;
    for (uint32_t tail = indexSequence - stageTail; tail < indexSequence; ++tail) {
        asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID2 + tail % STAGE_RING));
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::IssueRouteTokens(
    uint32_t rank, uint32_t slot, uint32_t count, uint32_t first, uint32_t sequence, uint32_t tokenSequence)
{
    uint32_t bufferIdx = sequence % MOE_EP_BUFFER_NUM;
    bool directSelf = networkMode_ != NETWORK_HYBRID && rank == epRankId_;
    LocalTensor<int32_t> ids = ubMeta_[routeIdsBase_ + bufferIdx * routeIdsElems_];
    if (directSelf && routeIdsSequence_[bufferIdx] != sequence) {
        asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
        // Snapshot before this input slot can be reused. Later payload batches no longer read route input.
        const uint32_t input = routeInputBase_ + bufferIdx * routeInputElems_ +
                               MOE_EP_SLOT_META_SECTIONS * axisKAlign_ + MOE_EP_SLOT_TOKEN_OFFSET;
        for (uint32_t row = 0; row < count; ++row) {
            ids.SetValue(row, ubMeta_.GetValue(input + row * paddedMetaElems_));
        }
        routeIdsSequence_[bufferIdx] = sequence;
    }
    GM_ADDR window = localWinAddr_ + (static_cast<int64_t>(rank) * numMaxTokensPerRank_ + slot + first) * perSlotBytes_;
    uint32_t batch = (count - first) < tokenBatchSlots_ ? count - first : tokenBatchSlots_;
    IssueTokenBatch(window, ids[first], batch, tokenSequence, directSelf);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::WriteRouteMetadata(
    uint32_t rank, uint32_t slot, uint32_t count, uint32_t planBase, uint32_t bufferIdx)
{
    LocalTensor<int32_t> plan = ubMeta_[routeOutputBase_ + bufferIdx * routeLayout_.size];
    // These descriptors are immutable after VF completion; all GM writes finish before output-slot reuse.
    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    const uint32_t groups = static_cast<uint32_t>(plan.GetValue(MOE_EP_ROUTE_GROUP_COUNT_OFFSET));
    for (uint32_t group = 0; group < groups; ++group) {
        const uint32_t g = routeLayout_.groups + group * MOE_EP_ROUTE_GROUP_FIELDS;
        const uint32_t n = static_cast<uint32_t>(plan.GetValue(g + MOE_EP_ROUTE_GROUP_HITS_OFFSET));
        const uint32_t metadataOffset = static_cast<uint32_t>(plan.GetValue(g + MOE_EP_ROUTE_GROUP_META_OFFSET));
        const int64_t metadataRow = plan.GetValue(g + MOE_EP_ROUTE_GROUP_META_ROW_OFFSET);
        asc_copy_ub2gm_align((__gm__ uint8_t*)recvSrcMetadataGm_[metadataRow * MOE_EP_RECV_META_FIELDS].GetPhyAddr(),
                             (__ubuf__ uint8_t*)plan[routeLayout_.metadata + metadataOffset].GetPhyAddr(),
                             n * MOE_EP_RECV_META_FIELDS * sizeof(int32_t));
        if constexpr (HasTopkWeights) {
            const uint32_t weightOffset = static_cast<uint32_t>(plan.GetValue(g + MOE_EP_ROUTE_GROUP_WEIGHT_OFFSET));
            const int64_t destination = plan.template ReinterpretCast<int64_t>().GetValue(
                (g + MOE_EP_ROUTE_GROUP_DST_OFFSET) / MOE_EP_INT32_PER_INT64);
            asc_copy_ub2gm_align((__gm__ uint8_t*)recvTopkWeightsGm_[destination].GetPhyAddr(),
                                 (__ubuf__ uint8_t*)plan[routeLayout_.weights + weightOffset].GetPhyAddr(),
                                 n * sizeof(float));
        }
    }
    const uint32_t hits = static_cast<uint32_t>(plan.GetValue(MOE_EP_ROUTE_HIT_COUNT_OFFSET));
    asc_copy_ub2gm_align((__gm__ uint8_t*)slotEndsGm_[GlobalSlot(rank, slot)].GetPhyAddr(),
                         (__ubuf__ uint8_t*)plan[routeLayout_.ends].GetPhyAddr(), count * sizeof(int32_t));
    if (hits != 0U) {
        uint32_t copyLen = hits * sizeof(int32_t);
        asc_copy_ub2gm_align((__gm__ uint8_t*)slotRecvXIdsGm_[planBase].GetPhyAddr(),
                             (__ubuf__ uint8_t*)plan[routeLayout_.recvIdx].GetPhyAddr(), copyLen);
        asc_copy_ub2gm_align((__gm__ uint8_t*)slotTopkIdsGm_[planBase].GetPhyAddr(),
                             (__ubuf__ uint8_t*)plan[routeLayout_.topk].GetPhyAddr(), copyLen);
    }
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::ConsumeRoutePlan(
    uint32_t rank, uint32_t slot, uint32_t count, uint32_t planBase, uint32_t sequence, uint32_t& tokenSequence,
    uint32_t& tokenIssued, uint32_t& indexSequence, bool hasNext, uint32_t nextRank, uint32_t nextSlot,
    uint32_t nextCount)
{
    uint32_t bufferIdx = sequence % MOE_EP_BUFFER_NUM;
    LocalTensor<int32_t> plan = ubMeta_[routeOutputBase_ + bufferIdx * routeLayout_.size];
    auto destinations = plan[routeLayout_.dst].template ReinterpretCast<int64_t>();
    LocalTensor<int32_t> tokenIds = plan[routeLayout_.token];

    for (uint32_t first = 0; first < count; first += tokenBatchSlots_) {
        uint32_t end = (first + tokenBatchSlots_) < count ? first + tokenBatchSlots_ : count;
        uint32_t tokenBuffer = tokenSequence % MOE_EP_BUFFER_NUM;
        // A next-frame direct-self ID wait must not hold back this batch's output submission.
        bool deferNextToken = tokenIssued == tokenSequence + 1U && end == count && hasNext &&
                              networkMode_ != NETWORK_HYBRID && nextRank == epRankId_;
        // At most one batch ahead. Any reused token slot belongs to an already submitted output batch.
        if (tokenIssued == tokenSequence + 1U) {
            if (end < count) {
                IssueRouteTokens(rank, slot, count, end, sequence, tokenIssued++);
            } else if (hasNext && !deferNextToken) {
                IssueRouteTokens(nextRank, nextSlot, nextCount, 0U, sequence + 1U, tokenIssued++);
            }
        }
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID0 + tokenBuffer));
        for (uint32_t row = first; row < end; ++row) {
            uint32_t beginHit =
                row == 0U ? 0U : static_cast<uint32_t>(plan.GetValue(routeLayout_.ends + row - 1U)) - planBase;
            uint32_t endHit = static_cast<uint32_t>(plan.GetValue(routeLayout_.ends + row)) - planBase;
            uint32_t hits = endHit - beginHit;
            bool self = plan.GetValue(routeLayout_.rank + row) == static_cast<int32_t>(epRankId_);
            LocalTensor<XType> token = tokenRing_[tokenBuffer * tokenRingElems_ + (row - first) * tokenSlotElems_];
            for (uint32_t j = beginHit; j < endHit; ++j) {
                int64_t destination = destinations.GetValue(j);
                asc_copy_ub2gm_align((__gm__ uint8_t*)recvXGm_[destination * axisH_].GetPhyAddr(),
                                     (__ubuf__ uint8_t*)token.GetPhyAddr(), tokenBytes_);
                if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                    asc_copy_ub2gm_align((__gm__ uint8_t*)recvScalesGm_[destination * scalesElems_].GetPhyAddr(),
                                         (__ubuf__ uint8_t*)token[scalesStride_].GetPhyAddr(), scalesBytes_);
                }
            }
            if (row + 1U == end) {
                asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + tokenBuffer));
                if (deferNextToken) {
                    // Current payload can drain while Scalar waits for the next frame's IDs.
                    IssueRouteTokens(nextRank, nextSlot, nextCount, 0U, sequence + 1U, tokenIssued++);
                }
            }
            if (self && hits != 0U) {
                const uint32_t indexBuffer = indexSequence % STAGE_RING;
                if (indexSequence >= STAGE_RING) {
                    asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID2 + indexBuffer));
                }
                ++indexSequence;
                // Payload was submitted first; only index-stage writes need this reuse wait.
                for (uint32_t j = beginHit; j < endHit; ++j) {
                    ubStageLocalIndexRing_.SetValue(indexBuffer * stageSlotElems_ + (j - beginHit) * ELEM_ALIGN,
                                                    static_cast<int32_t>(destinations.GetValue(j)));
                }
                asc_sync_notify(PIPE_S, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID2 + indexBuffer));
                asc_sync_wait(PIPE_S, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID2 + indexBuffer));
                uint32_t originalToken = static_cast<uint32_t>(tokenIds.GetValue(row));
                for (uint32_t j = beginHit; j < endHit; ++j) {
                    uint32_t topk = static_cast<uint32_t>(plan.GetValue(routeLayout_.flatRouteIndex + j)) % axisK_;
                    asc_copy_ub2gm_align(
                        (__gm__ uint8_t*)localRecvIndexGm_[originalToken * axisK_ + topk].GetPhyAddr(),
                        (__ubuf__ uint8_t*)
                            ubStageLocalIndexRing_[indexBuffer * stageSlotElems_ + (j - beginHit) * ELEM_ALIGN]
                                .GetPhyAddr(),
                        sizeof(int32_t));
                }
                asc_sync_notify(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID2 + indexBuffer));
            }
            if (row == 0U) {
                // Preserve first-token priority; release metadata independently of the remaining payload writes.
                WriteRouteMetadata(rank, slot, count, planBase, bufferIdx);
            }
        }
        ++tokenSequence;
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::CopySlotsCached()
{
    uint32_t rank = 0U;
    uint32_t start = 0U;
    uint32_t count = 0U;
    uint32_t coreStart = 0U;
    uint32_t nextRank = 0U;
    uint32_t nextTileStart = 0U;
    uint32_t nextTileCount = 0U;
    uint32_t nextCoreStart = 0U;
    uint32_t sequence = 0U;
    uint32_t tokenSequence = 0U;
    uint32_t tokenIssued = 0U;
    uint32_t stageSequence = 0U;
    bool hasTile = NextTile(0U, TILE_NONE, rank, start, count, coreStart, slotsPerTile_);
    if (hasTile) {
        // Only direct-self needs token IDs. Prime payload BEFORE the descriptor-boundary GM reads.
        IssueCachedIds(rank, coreStart + start, count, 0U);
        IssueCachedTokens(rank, coreStart + start, count, 0U, 0U, tokenIssued++);
        IssueMeta(rank, coreStart + start, count, 0U);
        if (count > tokenBatchSlots_) {
            IssueCachedTokens(rank, coreStart + start, count, tokenBatchSlots_, 0U, tokenIssued++);
        }
    }
    while (hasTile) {
        uint32_t bufferIdx = sequence % MOE_EP_BUFFER_NUM;
        uint32_t base = bufferIdx * metaRingElems_;
        uint32_t globalSlot = GlobalSlot(rank, coreStart + start);
        uint32_t prefix = globalSlot == 0U ? 0U : PLAN_END_EXTRA;
        bool hasNext = NextTile(rank, start, nextRank, nextTileStart, nextTileCount, nextCoreStart, slotsPerTile_);
        if (hasNext) {
            const uint32_t nextBuffer = (sequence + 1U) % MOE_EP_BUFFER_NUM;
            if (sequence + 1U >= MOE_EP_BUFFER_NUM) {
                // Previous Scalar consumers (including token-ID reads) have returned this input frame.
                asc_sync_wait(PIPE_S, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + nextBuffer));
            }
            IssueCachedIds(nextRank, nextCoreStart + nextTileStart, nextTileCount, nextBuffer);
            IssueMeta(nextRank, nextCoreStart + nextTileStart, nextTileCount, nextBuffer);
        }
        asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
        uint32_t planBegin = planRowBegin_[bufferIdx];
        for (uint32_t first = 0U; first < count; first += tokenBatchSlots_) {
            uint32_t end = (first + tokenBatchSlots_) < count ? first + tokenBatchSlots_ : count;
            uint32_t tokenBuffer = tokenSequence % MOE_EP_BUFFER_NUM;
            // Wait for next-frame direct-self IDs only after this batch has been submitted.
            const bool deferNextToken = tokenIssued == tokenSequence + 1U && end == count && hasNext &&
                                        networkMode_ != NETWORK_HYBRID && nextRank == epRankId_;
            // Continue payload lookahead across both input-frame and source-rank boundaries.
            if (tokenIssued == tokenSequence + 1U) {
                if (end < count) {
                    IssueCachedTokens(rank, coreStart + start, count, end, sequence, tokenIssued++);
                } else if (hasNext && !deferNextToken) {
                    IssueCachedTokens(nextRank, nextCoreStart + nextTileStart, nextTileCount, 0U, sequence + 1U,
                                      tokenIssued++);
                }
            }
            asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID0 + tokenBuffer));
            for (uint32_t row = first; row < end; ++row) {
                uint32_t beginHit =
                    row == 0U ? 0U : static_cast<uint32_t>(ubMeta_.GetValue(base + row + prefix - 1U)) - planBegin;
                uint32_t endHit = static_cast<uint32_t>(ubMeta_.GetValue(base + row + prefix)) - planBegin;
                LocalTensor<XType> token = tokenRing_[tokenBuffer * tokenRingElems_ + (row - first) * tokenSlotElems_];
                for (uint32_t j = beginHit; j < endHit; ++j) {
                    int64_t dst = ubMeta_.GetValue(base + planEndsElems_ + j);
                    asc_copy_ub2gm_align((__gm__ uint8_t*)recvXGm_[dst * axisH_].GetPhyAddr(),
                                         (__ubuf__ uint8_t*)token.GetPhyAddr(), tokenBytes_);
                    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                        asc_copy_ub2gm_align((__gm__ uint8_t*)recvScalesGm_[dst * scalesElems_].GetPhyAddr(),
                                             (__ubuf__ uint8_t*)token[scalesStride_].GetPhyAddr(), scalesBytes_);
                    }
                }
                if (row + 1U == end) {
                    // Weights never reference payload UB: return it before any weight-stage wait.
                    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + tokenBuffer));
                    if (deferNextToken) {
                        IssueCachedTokens(nextRank, nextCoreStart + nextTileStart, nextTileCount, 0U, sequence + 1U,
                                          tokenIssued++);
                    }
                }
                if constexpr (HasTopkWeights) {
                    if (endHit > beginHit) {
                        uint32_t stage = stageSequence % STAGE_RING;
                        if (stageSequence >= STAGE_RING) {
                            asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID2 + stage));
                        }
                        ++stageSequence;
                        uint32_t stageBase = stage * stageSlotElems_;
                        for (uint32_t j = beginHit; j < endHit; ++j) {
                            uint32_t topk = static_cast<uint32_t>(ubMeta_.GetValue(base + cachedTopkBase_ + j));
                            float weight = ubMeta_.ReinterpretCast<float>().GetValue(base + cachedWeightsBase_ +
                                                                                     row * axisKAlign_ + topk);
                            ubStageWeightsRing_.SetValue(stageBase + (j - beginHit) * ELEM_ALIGN, weight);
                        }
                        asc_sync_notify(PIPE_S, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID2 + stage));
                        asc_sync_wait(PIPE_S, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID2 + stage));
                        for (uint32_t j = beginHit; j < endHit; ++j) {
                            int64_t dst = ubMeta_.GetValue(base + planEndsElems_ + j);
                            asc_copy_ub2gm_align(
                                (__gm__ uint8_t*)recvTopkWeightsGm_[dst].GetPhyAddr(),
                                (__ubuf__ uint8_t*)ubStageWeightsRing_[stageBase + (j - beginHit) * ELEM_ALIGN]
                                    .GetPhyAddr(),
                                sizeof(float));
                        }
                        asc_sync_notify(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID2 + stage));
                    }
                }
            }
            ++tokenSequence;
        }
        // DMA outputs use payload/weight stages, not the input frame. Return even the last
        // Scalar frame so the metadata clone has an explicit S -> MTE2 handoff before UB reuse.
        asc_sync_notify(PIPE_S, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
        ++sequence;
        hasTile = hasNext;
        rank = nextRank;
        start = nextTileStart;
        count = nextTileCount;
        coreStart = nextCoreStart;
    }
    // Finish the input-frame handoff without waiting for the independent payload/weight outputs.
    const uint32_t inputTail = sequence < MOE_EP_BUFFER_NUM ? sequence : MOE_EP_BUFFER_NUM;
    for (uint32_t tail = sequence - inputTail; tail < sequence; ++tail) {
        asc_sync_wait(PIPE_S, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + tail % MOE_EP_BUFFER_NUM));
    }
    // All Scalar input-frame reads are finished. Clone into the returned shared UB while MTE3
    // still drains the independent token/weight buffers; clone IDs 2/3 do not consume payload IDs 0/1.
    // This also runs on cores with no payload, since handle copying is split independently by core.
    CopyCachedRankOffsets();
    CopyCachedMetadata();

    const uint32_t tokenTail = tokenSequence < MOE_EP_BUFFER_NUM ? tokenSequence : MOE_EP_BUFFER_NUM;
    for (uint32_t tail = tokenSequence - tokenTail; tail < tokenSequence; ++tail) {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + tail % MOE_EP_BUFFER_NUM));
    }
    if constexpr (HasTopkWeights) {
        const uint32_t stageTail = stageSequence < STAGE_RING ? stageSequence : STAGE_RING;
        for (uint32_t tail = stageSequence - stageTail; tail < stageSequence; ++tail) {
            asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID2 + tail % STAGE_RING));
        }
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline uint32_t MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::GlobalSlot(
    uint32_t rankId, uint32_t slotId)
{
    for (uint32_t r = 0; r < rankId; ++r) {
        slotId += static_cast<uint32_t>(ubRecvCnt_.GetValue(r));
    }
    return slotId;
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::CopyCachedArray(
    GlobalTensor<int32_t> src, GlobalTensor<int32_t> dst, uint32_t count)
{
    uint32_t start, end, blocks;
    SplitToCore(Ceil(count, ELEM_ALIGN), aivNum_, start, end, blocks);
    start *= ELEM_ALIGN;
    end = end * ELEM_ALIGN > count ? count : end * ELEM_ALIGN;
    if (start >= end) {
        return;
    }
    // Both input-frame halves have no Scalar readers at the call sites. IDs 2/3 keep this
    // ring independent of payload IDs 0/1, including when output DMA is still in flight.
    // Drain this ring before another array copy reuses the same halves.
    uint32_t copies = Ceil(end - start, metaRingElems_);
    uint32_t elements = end - start < metaRingElems_ ? end - start : metaRingElems_;
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_.GetPhyAddr(), (__gm__ uint8_t*)src[start].GetPhyAddr(),
                         elements * sizeof(int32_t));
    asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID2));
    for (uint32_t sequence = 0U; sequence < copies; ++sequence) {
        uint32_t bufferIdx = sequence % MOE_EP_BUFFER_NUM;
        uint32_t offset = start + sequence * metaRingElems_;
        elements = end - offset < metaRingElems_ ? end - offset : metaRingElems_;
        if (sequence + 1U < copies) {
            uint32_t nextBuffer = (sequence + 1U) % MOE_EP_BUFFER_NUM;
            if (sequence + 1U >= MOE_EP_BUFFER_NUM) {
                asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID2 + nextBuffer));
            }
            uint32_t nextOffset = offset + metaRingElems_;
            uint32_t nextElements = end - nextOffset < metaRingElems_ ? end - nextOffset : metaRingElems_;
            asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_[nextBuffer * metaRingElems_].GetPhyAddr(),
                                 (__gm__ uint8_t*)src[nextOffset].GetPhyAddr(), nextElements * sizeof(int32_t));
            asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID2 + nextBuffer));
        }
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID2 + bufferIdx));
        asc_copy_ub2gm_align((__gm__ uint8_t*)dst[offset].GetPhyAddr(),
                             (__ubuf__ uint8_t*)ubMeta_[bufferIdx * metaRingElems_].GetPhyAddr(),
                             elements * sizeof(int32_t));
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID2 + bufferIdx));
    }
    uint32_t remaining = copies < MOE_EP_BUFFER_NUM ? copies : MOE_EP_BUFFER_NUM;
    for (uint32_t tail = copies - remaining; tail < copies; ++tail) {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID2 + tail % MOE_EP_BUFFER_NUM));
    }
    asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID7));
    asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID7));
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void
MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::CopyCachedMetadata()
{
    CopyCachedArray(cachedRecvSrcMetadataGm_, recvSrcMetadataGm_, expertSum_ * MOE_EP_RECV_META_FIELDS);
    CopyCachedArray(cachedSlotEndsGm_, slotEndsGm_, GlobalSlot(epWorldSize_, 0U));
    CopyCachedArray(cachedSlotRecvXIdsGm_, slotRecvXIdsGm_, expertSum_);
    // Preserve the complete reusable handle, even when this invocation does not output weights.
    CopyCachedArray(cachedSlotTopkIdsGm_, slotTopkIdsGm_, expertSum_);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::IssueCachedIds(
    uint32_t rank, uint32_t slot, uint32_t count, uint32_t bufferIdx)
{
    if (networkMode_ == NETWORK_HYBRID || rank != epRankId_) {
        return;
    }
    GlobalTensor<int32_t> ids;
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        localWinAddr_ + (static_cast<int64_t>(rank) * numMaxTokensPerRank_ + slot) * perSlotBytes_ + metaOffset_ +
        (MOE_EP_SLOT_META_SECTIONS * axisKAlign_ + MOE_EP_SLOT_TOKEN_OFFSET) * sizeof(int32_t)));
    // Each 4-byte GM block occupies one aligned 32-byte UB block; no VF compaction is needed.
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)ubMeta_[bufferIdx * metaRingElems_ + cachedIdsBase_].GetPhyAddr(),
                         (__gm__ uint8_t*)ids.GetPhyAddr(), count, sizeof(int32_t), 0U, 0U, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, perSlotBytes_, UB_ALIGN);
    asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID2 + bufferIdx));
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::IssueCachedTokens(
    uint32_t rank, uint32_t slot, uint32_t count, uint32_t first, uint32_t metaSequence, uint32_t tokenSequence)
{
    uint32_t bufferIdx = metaSequence % MOE_EP_BUFFER_NUM;
    bool directSelf = networkMode_ != NETWORK_HYBRID && rank == epRankId_;
    if (directSelf && cachedIdsSequence_[bufferIdx] != metaSequence) {
        asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(SYNC_EVENT_ID2 + bufferIdx));
        cachedIdsSequence_[bufferIdx] = metaSequence;
    }
    uint32_t batch = (count - first) < tokenBatchSlots_ ? count - first : tokenBatchSlots_;
    GM_ADDR source = localWinAddr_ + (static_cast<int64_t>(rank) * numMaxTokensPerRank_ + slot + first) * perSlotBytes_;
    IssueTokenBatch(source, ubMeta_[bufferIdx * metaRingElems_ + cachedIdsBase_ + first * ELEM_ALIGN], batch,
                    tokenSequence, directSelf);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights, bool ClosOnly>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights, ClosOnly>::IssueTokenBatch(
    GM_ADDR slotAddr, LocalTensor<int32_t> tokenIds, uint32_t count, uint32_t sequence, bool directSelf)
{
    uint32_t bufferIdx = sequence % MOE_EP_BUFFER_NUM;
    if (sequence >= MOE_EP_BUFFER_NUM) {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
    }
    // Copy raw bytes to preserve fp8 token/scales bit patterns.
    GlobalTensor<uint8_t> input;
    LocalTensor<uint8_t> token = tokenRing_[bufferIdx * tokenRingElems_].template ReinterpretCast<uint8_t>();
    if (directSelf) {
        uint32_t idStride = IsCached ? ELEM_ALIGN : 1U;
        // direct本rank的tokenId不保证连续，逐行入同一批，批末统一通知。
        for (uint32_t row = 0; row < count; ++row) {
            int32_t tokenId = tokenIds.GetValue(row * idStride);
            input = xGm_.template ReinterpretCast<uint8_t>()[static_cast<int64_t>(tokenId) * tokenBytes_];
            asc_copy_gm2ub_align((__ubuf__ uint8_t*)token[row * tokenSlotBytes_].GetPhyAddr(),
                                 (__gm__ uint8_t*)input.GetPhyAddr(), 1U, tokenBytes_, 0U, 0U, false,
                                 asc_load_l2_cache_mode::NOTALLOC_KEEP, 0U, 0U);
        }
    } else {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(slotAddr));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)token.GetPhyAddr(), (__gm__ uint8_t*)input.GetPhyAddr(), count,
                             tokenBytes_, 0U, 0U, false, asc_load_l2_cache_mode::NOTALLOC_KEEP, perSlotBytes_,
                             tokenSlotBytes_);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        GlobalTensor<uint8_t> scales;
        scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(slotAddr + scalesOffset_));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)token[hAlignSize_].GetPhyAddr(), (__gm__ uint8_t*)scales.GetPhyAddr(),
                             count, scalesBytes_, 0U, 0U, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             perSlotBytes_, tokenSlotBytes_);
    }
    asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(SYNC_EVENT_ID0 + bufferIdx));
}
#endif

} // namespace MoeEpDispatchEpilogueImpl

#endif // MOE_EP_DISPATCH_EPILOGUE_H
