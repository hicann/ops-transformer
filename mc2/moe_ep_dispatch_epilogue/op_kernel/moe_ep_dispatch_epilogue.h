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
#include "moe_ep_dispatch_epilogue_count.h"
#endif

#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"
#include "../../common/op_kernel/moe_ep_send_completion.h"

namespace MoeEpDispatchEpilogueImpl {

#if defined(ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL)

using namespace AscendC;

static constexpr uint32_t UB_ALIGN = 32U;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t NETWORK_HYBRID = 1U; // 1 = hybrid dispatch, 0 = direct
// recv_src_metadata is compact in GM: [srcRank, srcToken, srcTopk, srcSlot, recvXIdx].
// The valid prefix is ordered by (srcRank, recvXIdx); each rank range follows increasing send-source rows.
// UB staging still uses an aligned stride so that every non-aligned 20-byte copy starts from a 32-byte boundary.
static constexpr uint32_t RECV_META_FIELDS = 5;
static constexpr uint32_t ELEM_ALIGN = 8U;
static constexpr uint32_t INT32_PER_BLOCK = UB_ALIGN / sizeof(int32_t);
static constexpr uint32_t META_TOPK_SECTION = 2U;
static constexpr uint32_t META_EXTRA_FIELDS = 2U;
static constexpr uint32_t META_SRC_RANK_OFFSET = 0U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_SLOT_IDX_OFFSET = 3U;
static constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;
static constexpr uint32_t HIT_ROW_OFFSET = 0U;
static constexpr uint32_t HIT_TOPK_OFFSET = 1U;
static constexpr uint32_t HIT_ENTRY_SIZE = 2U;
static constexpr uint32_t ALIGNED_LEN_256 = 256U;
static constexpr uint32_t SLOTS_TILE = 128U;
static constexpr uint32_t UB_CONTROL_RESERVE = 4096U; // 诊断、通信结束及框架临时空间
static constexpr uint32_t PLAN_END_EXTRA = 1U;        // UB额外读前一slot终点，GM不存起始0

// slot搬运的三处流水缓冲份数。都是人工调的旋钮：调大 → 重叠更深、更抗上游抖动；调小 → 省 UB。
// 三处互相独立，可以单独调。都是「全局拍号 % 份数」选 buffer，所以份数改动不需要动任何下标表达式。
//   META_RING  : tile 元数据的 MTE2 → S 环。份数 >= 2 就能把下一个 tile 的 meta 读提前一拍发出去，
//                让读延迟盖在当前拍的 MTE3 写上，而不是让标量 pipe 干等。
//   TOKEN_RING : token/scales 的 MTE2 → MTE3 环，让这一拍的读盖在上一拍的写里。
//   STAGE_RING : stage meta/weights/local-index 的 S → MTE3 环。份数 >= 3 之后，同一份的下一次使用恒在 STAGE_RING
//                拍之后，稳态的 per-slot 等待就顶掉了原来 tile 尾那次「等整条 MTE3 排空」。
static constexpr uint32_t META_RING = 2U;
static constexpr uint32_t TOKEN_RING = 3U;
static constexpr uint32_t STAGE_RING = 3U;

// 预读扫描的起点哨兵：表示「还没发过任何一拍的 meta」，从第一个有活的 rank 的第一拍开始找。
static constexpr uint32_t TILE_NONE = 0xFFFFFFFFU;

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
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
    __aicore__ inline void CopySlots();
    __aicore__ inline uint32_t BuildSlotHits(uint32_t metaBuf, uint32_t localSlot, uint32_t globalTileSlot);
    // 把 (rank, tile) 两层循环拍平成「取下一个 tile」。预读 meta 必须知道下一拍在哪，而下一拍会跨 rank，
    // 每个 rank 的槽位数又只有标量读得到；epWorldSize_ 很小，逐 rank 扫一遍的代价可以忽略。
    __aicore__ inline bool NextTile(uint32_t rankId, uint32_t tileStart, uint32_t& nextRank, uint32_t& nextTileStart,
                                    uint32_t& nextTileCnt, uint32_t& nextSlotStart);
    __aicore__ inline void IssueMeta(uint32_t rankId, uint32_t slotIdx, uint32_t tileCnt, uint32_t buf);
    __aicore__ inline void InitSlotBuffers(uint32_t tokenBytes, uint32_t sharedBytes, uint64_t ubBytes);
    __aicore__ inline uint32_t GlobalSlot(uint32_t rankId, uint32_t slotId);
    __aicore__ inline void CopyCachedArray(GlobalTensor<int32_t> src, GlobalTensor<int32_t> dst, uint32_t count);
    __aicore__ inline void CopyCachedMetadata();
    __aicore__ inline LocalTensor<XType> IssueTokenRead(GM_ADDR slotAddr, int32_t tokenId, uint32_t sequence,
                                                        const DataCopyParams& tokenCopyParams,
                                                        const DataCopyParams& scalesCopyParams);
    __aicore__ inline void CopyCachedRankOffsets();
    __aicore__ inline void InitLocalRecvIndex();

    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t& startId, uint32_t& endId,
                                       uint32_t& sendNum);
    __aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext* ctx, uint32_t rankId, uint64_t offset)
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

    TPipe* tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext* mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;
    uint32_t epRankId_{0};
    uint32_t networkMode_{0};      // 0 = direct(自段 x 直读输入), 1 = hybrid(自段 x 读win)
    bool isDirectSelfRank_{false}; // 当前遍历的源 rank 是否为 direct 本端段
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
    GlobalTensor<int32_t> slotRowIdsGm_;
    GlobalTensor<int32_t> cachedSlotEndsGm_;
    GlobalTensor<int32_t> cachedSlotRowIdsGm_;
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
    LocalTensor<int32_t> ubTopkIds_;
    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<int64_t> ubExpertPfx_;
    LocalTensor<int64_t> ubHitCountRowI64_;
    LocalTensor<XType> tokenRing_;
    LocalTensor<float> ubStageWeightsRing_;
    LocalTensor<int32_t> ubStageMetaRing_;
    LocalTensor<int32_t> ubStageLocalIndexRing_;
    LocalTensor<int32_t> ubLocalCursor_;
    LocalTensor<int64_t> ubHitList_;
    LocalTensor<int32_t> ubSlotPlan_;
    TBuf<QuePosition::VECIN> ubSlotPlanBuf_;
    uint32_t slotsPerTile_{SLOTS_TILE};
    uint32_t planEndsElems_{0};
    uint32_t planRingElems_{0};
    uint32_t planRowBegin_[META_RING] = {0};
    int32_t planEvtMte3ToS_[META_RING] = {0};
    LocalTensor<int32_t> ubWaitStatus_;
    LocalTensor<int32_t> ubWaitSum_;

    TBuf<QuePosition::VECIN> ubHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRankExpertHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRankExpertRowStartBuf_;
    TBuf<QuePosition::VECIN> ubRankOffsetsBuf_;
    TBuf<QuePosition::VECIN> ubRowStartBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<QuePosition::VECIN> ubTopkIdsBuf_;
    TBuf<QuePosition::VECIN> ubRecvCntBuf_;
    TBuf<QuePosition::VECIN> ubExpertPfxBuf_;
    TBuf<QuePosition::VECIN> ubHitCountRowI64Buf_;
    TBuf<QuePosition::VECIN> ubLocalCursorBuf_;
    TBuf<QuePosition::VECIN> ubHitListBuf_;
    // 两条路径共用token环与路由预取；metadata生成的stage仅首次执行使用。
    TBuf<QuePosition::VECIN> tokenRingBuf_;
    TBuf<QuePosition::VECIN> ubStageWeightsRingBuf_;
    TBuf<QuePosition::VECIN> ubStageMetaRingBuf_;
    TBuf<QuePosition::VECIN> ubStageLocalIndexRingBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> waitSumBuf_;
    TBuf<> sharedTmpBuf_;

    uint32_t expertSum_{0};
    GM_ADDR localWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    uint32_t scalesOffset_{0};
    uint32_t scalesStride_{0}; // token UB 槽内 scales 暂存偏移(32B 对齐, 与 GM slot 紧拼偏移解耦)
    uint32_t scalesElems_{0};
    uint32_t metaOffset_{0};
    uint32_t metaBytes_{0};
    // 三处自管环的「一份」有多少个元素，用来做 份号 * 每份元素数 的偏移。
    uint32_t tokenRingElems_{0};
    uint32_t stageSlotElems_{0};
    uint32_t metaRingElems_{0};
    uint32_t paddedMetaElems_{0};
    uint32_t axisKAlign_{0};
    uint32_t numLocalExperts_{0};
    // [rank][expert] 计数表里每跑一个 rank 的行步长。必须补到 ELEM_ALIGN 的整数倍：CopySlots 会按
    // prefixRank * rankExpertRowStride_ 切片交给 VEC，而 VEC 访问 UB 要求 32 字节对齐；numLocalExperts_ 本身没有
    // 对齐保证（host 侧就是 numExperts / epWorldSize，例：numExperts=4, epWorldSize=2 → 2）。
    uint32_t rankExpertRowStride_{0};
    uint32_t rankExpertCountStride_{0}; // 一核一整行的字节数（= rankExpertRowStride_ * epWorldSize_），8 的倍数
    uint32_t aivNum_{0};
    uint32_t axisK_{0};
    uint32_t axisH_{0};
    uint32_t numTokens_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t perSlotBytes_{0};
    uint32_t dispatchNotifyCount_{1};
    uint32_t totalNotifyCnt_{0};
    uint64_t winDataOffset_{0};
    uint64_t slotWinStateOffset_{0};
    // 三处自管环的事件，份数各自等于对应 *_RING。全部走 AllocEventID 拿真份数：FetchEventID 不置占用位、
    // 恒返回同一个下标，一份以上的环会退化成一条 id，等待就分不出是哪一份。
    int32_t metaEvtMte2ToS_[META_RING] = {0};
    int32_t metaEvtSToMte2_[META_RING] = {0};
    int32_t tokenEvtFill_[TOKEN_RING] = {0}; // MTE2_MTE3
    int32_t tokenEvtFree_[TOKEN_RING] = {0}; // MTE3_MTE2
    int32_t stageEvtMte3ToS_[STAGE_RING] = {0};
    // stage 的 S → MTE3 握手是 Set/Wait 紧邻的，任何时刻只有一发在飞，一条 id 就够。
    int32_t stageEvtSToMte3_{0};
};

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::SplitToCore(
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

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::Init(
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
    numTokens_ = tilingData->cfg.numTokens;
    epWorldSize_ = tilingData->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData->cfg.numMaxTokensPerRank;
    perSlotBytes_ = tilingData->cfg.perSlotBytes;
    dispatchNotifyCount_ = tilingData->dispatchNotifyCount;
    winDataOffset_ = tilingData->winDataOffset;
    slotWinStateOffset_ = tilingData->slotWinStateOffset;

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext*>(context);
    epRankId_ = mc2Context_->epRankId;
    constexpr size_t metadataOffset = offsetof(MoeEpDispatchEpilogueTilingData, moeEpDispatchEpilogueInfo) +
                                      offsetof(MoeEpDispatchEpilogueInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_DISPATCH_EPILOGUE, tpipe_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_);
    // 槽内 scales/meta 基址: direct 紧拼 tokenSize; hybrid 为 ALIGN32(tokenSize)（hybrid dispatch 整槽布局）
    uint32_t hAlignSize = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;
    uint32_t slotMetaBase = (tilingData->networkMode == NETWORK_HYBRID) ? hAlignSize : axisH_ * sizeof(XType);
    metaOffset_ = slotMetaBase;
    scalesStride_ = hAlignSize / sizeof(XType);

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
    slotRowIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvSrcMetadata + tilingData->slotRowIdsOffset));
    if constexpr (IsCached) {
        cachedSlotEndsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cachedRecvSrcMetadata + tilingData->slotEndsOffset));
        cachedSlotRowIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cachedRecvSrcMetadata + tilingData->slotRowIdsOffset));
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
    uint32_t ubRankOffsetsBytes = Ceil((epWorldSize_ + 1U) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubRankOffsetsBuf_, ubRankOffsetsBytes);
    ubRankOffsets_ = ubRankOffsetsBuf_.Get<int32_t>();

    uint32_t scalesBytesAlign = 0U;
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesOffset_ = slotMetaBase;
        const uint32_t scalesBytes = tilingData->cfg.scalesBytes;
        scalesElems_ = scalesBytes / sizeof(ScalesType);
        scalesBytesAlign = Ceil(scalesBytes, UB_ALIGN) * UB_ALIGN;
        metaOffset_ += scalesBytesAlign;
        recvScalesGm_.SetGlobalBuffer((__gm__ ScalesType*)recvScales);
    }

    InitSlotBuffers(hAlignSize + scalesBytesAlign, sharedBytes, tilingData->totalUbSize);

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
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::InitSlotBuffers(
    uint32_t tokenBytes, uint32_t sharedBytes, uint64_t ubBytes)
{
    rankExpertRowStride_ = Ceil(numLocalExperts_, ELEM_ALIGN) * ELEM_ALIGN;
    rankExpertCountStride_ = rankExpertRowStride_ * epWorldSize_;
    paddedMetaElems_ = Ceil(META_TOPK_SECTION * axisKAlign_ + META_EXTRA_FIELDS, ELEM_ALIGN) * ELEM_ALIGN;
    stageSlotElems_ = axisK_ * ELEM_ALIGN;
    uint32_t expertBytes = Ceil(numLocalExperts_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t countBytes = Ceil(numLocalExperts_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t recvCountBytes = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t rankOffsetBytes = Ceil((epWorldSize_ + 1U) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t hitBytes = Ceil(axisK_ * HIT_ENTRY_SIZE * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t stageBytes = stageSlotElems_ * sizeof(int32_t) * STAGE_RING;
    uint64_t fixedBytes = expertBytes + totalNotifyCnt_ * UB_ALIGN + UB_ALIGN + sharedBytes + rankOffsetBytes +
                          recvCountBytes + hitBytes + TOKEN_RING * tokenBytes + UB_CONTROL_RESERVE;
    if constexpr (!IsCached) {
        fixedBytes +=
            2U * countBytes + 2U * rankExpertCountStride_ * sizeof(int32_t) + 2U * expertBytes + 2U * stageBytes;
    }
    if constexpr (HasTopkWeights) {
        fixedBytes += stageBytes;
    }
    // 按当前tiling全部buffer核算，最大K/专家数/卡数也不能挤出UB。
    while (true) {
        metaRingElems_ = slotsPerTile_ * paddedMetaElems_;
        if constexpr (!IsCached) {
            metaRingElems_ = metaRingElems_ < rankExpertCountStride_ ? rankExpertCountStride_ : metaRingElems_;
        }
        planEndsElems_ = Ceil(slotsPerTile_ + PLAN_END_EXTRA, ELEM_ALIGN) * ELEM_ALIGN;
        planRingElems_ = planEndsElems_ + Ceil(slotsPerTile_ * axisK_, ELEM_ALIGN) * ELEM_ALIGN;
        uint64_t routeBytes = META_RING * metaRingElems_ * sizeof(int32_t);
        uint64_t planBytes = META_RING * planRingElems_ * sizeof(int32_t);
        uint64_t countTileBytes = IsCached ? 0U : slotsPerTile_ * axisKAlign_ * sizeof(int32_t);
        if (fixedBytes + routeBytes + planBytes + countTileBytes <= ubBytes || slotsPerTile_ == 1U) {
            break;
        }
        slotsPerTile_ /= 2U;
    }
    tokenRingElems_ = tokenBytes / sizeof(XType);
    tpipe_->InitBuffer(ubRecvCntBuf_, recvCountBytes);
    tpipe_->InitBuffer(ubMetaBuf_, metaRingElems_ * sizeof(int32_t) * META_RING);
    tpipe_->InitBuffer(ubHitListBuf_, hitBytes);
    tpipe_->InitBuffer(ubSlotPlanBuf_, planRingElems_ * sizeof(int32_t) * META_RING);
    tpipe_->InitBuffer(tokenRingBuf_, tokenBytes * TOKEN_RING);
    ubRecvCnt_ = ubRecvCntBuf_.Get<int32_t>();
    ubMeta_ = ubMetaBuf_.Get<int32_t>();
    ubHitList_ = ubHitListBuf_.Get<int64_t>();
    ubSlotPlan_ = ubSlotPlanBuf_.Get<int32_t>();
    tokenRing_ = tokenRingBuf_.Get<XType>();
    if constexpr (HasTopkWeights) {
        tpipe_->InitBuffer(ubStageWeightsRingBuf_, stageBytes);
        ubStageWeightsRing_ = ubStageWeightsRingBuf_.Get<float>();
    }
    if constexpr (!IsCached) {
        tpipe_->InitBuffer(ubHitCountBuf_, countBytes);
        tpipe_->InitBuffer(ubRankExpertHitCountBuf_, rankExpertCountStride_ * sizeof(int32_t));
        tpipe_->InitBuffer(ubRankExpertRowStartBuf_, rankExpertCountStride_ * sizeof(int32_t));
        tpipe_->InitBuffer(ubRowStartBuf_, expertBytes);
        tpipe_->InitBuffer(ubTopkIdsBuf_, slotsPerTile_ * axisKAlign_ * sizeof(int32_t));
        tpipe_->InitBuffer(ubHitCountRowI64Buf_, expertBytes);
        tpipe_->InitBuffer(ubStageMetaRingBuf_, stageBytes);
        tpipe_->InitBuffer(ubStageLocalIndexRingBuf_, stageBytes);
        tpipe_->InitBuffer(ubLocalCursorBuf_, countBytes);
        ubHitCount_ = ubHitCountBuf_.Get<int32_t>();
        ubRankExpertHitCount_ = ubRankExpertHitCountBuf_.Get<int32_t>();
        ubRankExpertRowStart_ = ubRankExpertRowStartBuf_.Get<int32_t>();
        ubRowStart_ = ubRowStartBuf_.Get<int64_t>();
        ubTopkIds_ = ubTopkIdsBuf_.Get<int32_t>();
        ubHitCountRowI64_ = ubHitCountRowI64Buf_.Get<int64_t>();
        ubStageMetaRing_ = ubStageMetaRingBuf_.Get<int32_t>();
        ubStageLocalIndexRing_ = ubStageLocalIndexRingBuf_.Get<int32_t>();
        ubLocalCursor_ = ubLocalCursorBuf_.Get<int32_t>();
    }
    for (uint32_t i = 0; i < META_RING; ++i) {
        metaEvtMte2ToS_[i] = tpipe_->AllocEventID<AscendC::HardEvent::MTE2_S>();
        metaEvtSToMte2_[i] = tpipe_->AllocEventID<AscendC::HardEvent::S_MTE2>();
        if constexpr (!IsCached) {
            planEvtMte3ToS_[i] = tpipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
        }
    }
    for (uint32_t i = 0; i < TOKEN_RING; ++i) {
        tokenEvtFill_[i] = tpipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>();
        tokenEvtFree_[i] = tpipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    }
    if constexpr (!IsCached || HasTopkWeights) {
        for (uint32_t i = 0; i < STAGE_RING; ++i) {
            stageEvtMte3ToS_[i] = tpipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
        }
        stageEvtSToMte3_ = tpipe_->AllocEventID<AscendC::HardEvent::S_MTE3>();
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::InitLocalRecvIndex()
{
    const uint32_t totalElements = numTokens_ * axisK_;
    if constexpr (IsCached) {
        // 路由未变，共用批量搬运函数；此时metadata缓冲尚未用于预取。
        CopyCachedArray(cachedLocalRecvIndexGm_, localRecvIndexGm_, totalElements);
    } else {
        uint32_t startBlock, endBlock, blockCount;
        SplitToCore(Ceil(totalElements, INT32_PER_BLOCK), aivNum_, startBlock, endBlock, blockCount);
        const uint32_t start = startBlock * INT32_PER_BLOCK;
        const uint32_t end = endBlock * INT32_PER_BLOCK > totalElements ? totalElements : endBlock * INT32_PER_BLOCK;
        if (start < end) {
            InitOutput<int32_t>(localRecvIndexGm_[start], end - start, -1);
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::Process()
{
    InitLocalRecvIndex();
    if constexpr (!IsCached) {
        ComputePrefixSums();
    }
    WaitDispatch();
    SyncAll<true>();
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_WAIT_DONE);

    if constexpr (!IsCached) {
        CountHits();
        SyncAll<true>();
        // 每核根据共享计数生成自己的metadata起点，结果留在UB，无须再次核间同步。
        BuildRankRowStarts();
    } else {
        DataCopyPad(ubRecvCnt_, numRecvPerRankGm_,
                    {1U, epWorldSize_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        CopyCachedRankOffsets();
        CopyCachedMetadata();
    }

    // 两条路径共用slot分核与搬运流水，仅命中位置的准备方式不同。
    CopySlots();
    // 所有输出在交给Combine前写完，包含metadata及slot索引。
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_OUTPUT_DONE);
    MoeEpCompletion::DrainChannels(mc2Context_, epWorldSize_, tpipe_);
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
    statusGMTensor.SetGlobalBuffer((__gm__ int32_t*)localSlotStateWinAddr_);
    DataCopyParams statusCopyParams = {static_cast<uint16_t>(totalNotifyCnt_), 1U,
                                       static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
    DataCopyParams clearStatusCopyParams = {static_cast<uint16_t>(totalNotifyCnt_), 1U, 0U,
                                            static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};

    SyncFunc<AscendC::HardEvent::S_V>(); // 确保expertSum_计算完成
    // 2.3us
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
    // Dispatch reuses these notification slots in the next continue round.  SyncAll only synchronizes AIVs; it does
    // not replace the MTE3 completion required before a later kernel can publish the next round's notifications.
    SyncFunc<AscendC::HardEvent::MTE3_S>();
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
    Duplicate(ubRankExpertHitCount_, (int32_t)0, rankExpertCountStride_);

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

        uint32_t topkBytes = axisK_ * sizeof(int32_t);

        for (uint32_t tileStart = 0; tileStart < slotCntPerAiv; tileStart += slotsPerTile_) {
            uint32_t tileCnt =
                (slotCntPerAiv - tileStart > slotsPerTile_) ? slotsPerTile_ : (slotCntPerAiv - tileStart);

            srcTopkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                srcRankBase + (int64_t)(slotStart + tileStart) * perSlotBytes_ + metaOffset_));
            DataCopyExtParams topkCopyParams{static_cast<uint16_t>(tileCnt), static_cast<uint32_t>(topkBytes),
                                             static_cast<int64_t>(perSlotBytes_ - topkBytes), 0, 0};
            DataCopyPadExtParams<int32_t> topkPadParams{true, 0, static_cast<uint8_t>(axisKAlign_ - axisK_), -1};
            DataCopyPad(ubTopkIds_, srcTopkIdsGm, topkCopyParams, topkPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();

            // 一次直方图统计整批topk，避免每个专家都扫描和等待V_S。
            asc_vf_call<MoeEpDispatchEpilogueCount::CountTopkHits>(
                (__ubuf__ int32_t*)ubTopkIds_.GetPhyAddr(),
                (__ubuf__ int32_t*)ubRankExpertHitCount_[rankId * rankExpertRowStride_].GetPhyAddr(),
                tileCnt * axisKAlign_, numLocalExperts_, epRankId_ * numLocalExperts_);
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        Add(ubHitCount_, ubHitCount_, ubRankExpertHitCount_[rankId * rankExpertRowStride_], numLocalExperts_);
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    // 每核一份的 hitCount 矩阵不再落 GM：它是 CopySlots 里唯一的使用者，而那处已经改成
    // 对 ubRankExpertHitCount_ 的 rank 维求和，不需要再绕一趟 GM。ubHitCount_ 本身还有用（本核累计值）。
    DataCopyExtParams jointCountCopyParams{1U, rankExpertCountStride_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U,
                                           0U};
    DataCopyPad(rankExpertHitCountGm_[(int64_t)aivId_ * rankExpertCountStride_], ubRankExpertHitCount_,
                jointCountCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::BuildRankRowStarts()
{
    // CountHits used ubRankExpertHitCount_ as an MTE3 source. Finish that read before reusing the buffer as the
    // per-rank prefix contributed by cores preceding this core.
    SyncFunc<AscendC::HardEvent::MTE3_V>();

    // H[c][r][e] counts expanded records. Reduce all cores and the cores before this AIV separately.
    // Flat rows are [rank][expert], padded only at the end of each core row.
    Duplicate(ubRankExpertRowStart_, (int32_t)0, rankExpertCountStride_);
    Duplicate(ubRankExpertHitCount_, (int32_t)0, rankExpertCountStride_);

    // Reuse metadata staging for complete core-row tiles; even the maximum 2048-expert row fits.
    uint32_t jointRowBytes = rankExpertCountStride_ * sizeof(int32_t);
    uint32_t coreRowsPerTile = metaRingElems_ * sizeof(int32_t) / jointRowBytes;
    coreRowsPerTile = coreRowsPerTile > aivNum_ ? aivNum_ : coreRowsPerTile;

    DataCopyPadExtParams<int32_t> jointMatrixPadParams{false, 0U, 0U, 0};
    for (uint32_t coreBase = 0; coreBase < aivNum_; coreBase += coreRowsPerTile) {
        uint32_t rowsThisTile = (aivNum_ - coreBase > coreRowsPerTile) ? coreRowsPerTile : (aivNum_ - coreBase);
        DataCopyExtParams jointMatrixCopyParams{1U, rowsThisTile * jointRowBytes, 0U, 0U, 0U};
        DataCopyPad(ubMeta_, rankExpertHitCountGm_[(int64_t)coreBase * rankExpertCountStride_], jointMatrixCopyParams,
                    jointMatrixPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        for (uint32_t localCoreId = 0; localCoreId < rowsThisTile; ++localCoreId) {
            uint32_t coreId = coreBase + localCoreId;
            LocalTensor<int32_t> jointCountRow = ubMeta_[localCoreId * rankExpertCountStride_];
            Add(ubRankExpertRowStart_, ubRankExpertRowStart_, jointCountRow, rankExpertCountStride_);
            if (coreId < aivId_) {
                Add(ubRankExpertHitCount_, ubRankExpertHitCount_, jointCountRow, rankExpertCountStride_);
            }
        }
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }

    // P(c,r,e) = offsets[r] + sum_{e'<e,c'} H[c'][r][e'] + sum_{c'<c} H[c'][r][e].
    // recv_x stays expert/core/rank ordered, so (rank, expert, core, per-group cursor) is (rank, recv_x_idx) order.
    // 总计数做整数前缀和，再加此前core的计数；保持原metadata与recvXIdx顺序。
    asc_vf_call<MoeEpDispatchEpilogueCount::BuildMetadataStarts>(
        (__ubuf__ int32_t*)ubRankExpertRowStart_.GetPhyAddr(), (__ubuf__ int32_t*)ubRankExpertHitCount_.GetPhyAddr(),
        (__ubuf__ int32_t*)ubRankOffsets_.GetPhyAddr(), epWorldSize_, rankExpertRowStride_);
    if (aivId_ == 0U) {
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyExtParams offsetsCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U,
                                            0U};
        DataCopyPad(recvRankOffsetsGm_, ubRankOffsets_, offsetsCopyParams);
        // rankOffsets暂存不再复用，由Process末尾统一等待写出完成。
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyCachedRankOffsets()
{
    if (aivId_ != 0U) {
        return;
    }

    DataCopyExtParams offsetsCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> offsetsPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubRankOffsets_, cachedRecvRankOffsetsGm_, offsetsCopyParams, offsetsPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyPad(recvRankOffsetsGm_, ubRankOffsets_, offsetsCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline bool MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::NextTile(
    uint32_t rankId, uint32_t tileStart, uint32_t& nextRank, uint32_t& nextTileStart, uint32_t& nextTileCnt,
    uint32_t& nextSlotStart)
{
    for (uint32_t r = rankId; r < epWorldSize_; ++r) {
        int32_t slotCnt = ubRecvCnt_.GetValue(r);
        uint32_t slotStart, slotEnd, slotCntPerAiv;
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        // 同一个 rank 内接着上一拍往后走；跨到下一个 rank 就从它的第 0 拍开始。TILE_NONE 表示还没发过任何一拍。
        uint32_t start = (r == rankId && tileStart != TILE_NONE) ? (tileStart + slotsPerTile_) : 0U;
        if (start >= slotCntPerAiv) {
            continue;
        }
        nextRank = r;
        nextTileStart = start;
        nextTileCnt = (slotCntPerAiv - start > slotsPerTile_) ? slotsPerTile_ : (slotCntPerAiv - start);
        nextSlotStart = slotStart;
        return true;
    }
    return false;
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::IssueMeta(uint32_t rankId,
                                                                                                     uint32_t slotIdx,
                                                                                                     uint32_t tileCnt,
                                                                                                     uint32_t buf)
{
    DataCopyExtParams metaCopyParams{static_cast<uint16_t>(tileCnt), metaBytes_, perSlotBytes_ - metaBytes_, 0, 0};
    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    GlobalTensor<int32_t> srcMetaGm;
    srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localWinAddr_ +
                                                                (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_ +
                                                                (int64_t)slotIdx * perSlotBytes_ + metaOffset_));
    DataCopyPad(ubMeta_[buf * metaRingElems_], srcMetaGm, metaCopyParams, metaPadParams);
    if constexpr (IsCached) {
        const uint32_t globalSlot = GlobalSlot(rankId, slotIdx);
        const uint32_t firstEnd = globalSlot == 0U ? 0U : globalSlot - 1U;
        const uint32_t prefix = globalSlot == 0U ? 0U : PLAN_END_EXTRA;
        const uint32_t begin = globalSlot == 0U ? 0U : cachedSlotEndsGm_.GetValue(firstEnd);
        const uint32_t end = cachedSlotEndsGm_.GetValue(globalSlot + tileCnt - 1U);
        planRowBegin_[buf] = begin;
        // 终点与行号连续预取；与窗口meta共用一组就绪/归还事件。
        DataCopyPad(ubSlotPlan_[buf * planRingElems_], cachedSlotEndsGm_[firstEnd],
                    {1U, (tileCnt + prefix) * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, metaPadParams);
        if (end > begin) {
            DataCopyPad(ubSlotPlan_[buf * planRingElems_ + planEndsElems_], cachedSlotRowIdsGm_[begin],
                        {1U, (end - begin) * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, metaPadParams);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_S>(metaEvtMte2ToS_[buf]);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline uint32_t MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::BuildSlotHits(
    uint32_t metaBuf, uint32_t localSlot, uint32_t globalTileSlot)
{
    uint32_t hitCnt = 0U;
    if constexpr (IsCached) {
        const uint32_t planBase = metaBuf * planRingElems_;
        // 索引由首次dispatch生成；cached要求路由不变，无须重复校验每个索引。
        const uint32_t prefix = globalTileSlot == 0U ? 0U : PLAN_END_EXTRA;
        const uint32_t end = ubSlotPlan_.GetValue(planBase + localSlot + prefix);
        const uint32_t begin = localSlot == 0U ?
                                   planRowBegin_[metaBuf] :
                                   static_cast<uint32_t>(ubSlotPlan_.GetValue(planBase + localSlot + prefix - 1U));
        for (uint32_t row = begin; row < end; ++row) {
            const int32_t metadataRow = ubSlotPlan_.GetValue(planBase + planEndsElems_ + row - planRowBegin_[metaBuf]);
            // 索引只存行号，原metadata仍是唯一字段来源，不复制五字段大表。
            const int64_t offset = static_cast<int64_t>(metadataRow) * RECV_META_FIELDS;
            const int32_t recvRow = cachedRecvSrcMetadataGm_.GetValue(offset + META_RECV_X_IDX_OFFSET);
            const int32_t topk = cachedRecvSrcMetadataGm_.GetValue(offset + META_TOPK_IDX_OFFSET);
            ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_ROW_OFFSET, recvRow);
            ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET, topk);
            ++hitCnt;
        }
    } else {
        const uint32_t metaBase = metaBuf * metaRingElems_ + localSlot * paddedMetaElems_;
        const int32_t rankExpertBase = static_cast<int32_t>(epRankId_ * numLocalExperts_);
        const int32_t rankExpertEnd = rankExpertBase + static_cast<int32_t>(numLocalExperts_);
        for (uint32_t topkIdx = 0; topkIdx < axisK_; ++topkIdx) {
            int32_t expertId = ubMeta_.GetValue(metaBase + topkIdx);
            if (expertId < rankExpertBase || expertId >= rankExpertEnd) {
                continue;
            }
            uint32_t localExpertId = static_cast<uint32_t>(expertId - rankExpertBase);
            int64_t expertRowStart = ubRowStart_.GetValue(localExpertId);
            int32_t cursor = ubLocalCursor_.GetValue(localExpertId);
            ubLocalCursor_.SetValue(localExpertId, cursor + 1);
            int64_t recvXRow = expertRowStart + cursor;
            ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_ROW_OFFSET, recvXRow);
            ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET, static_cast<int64_t>(topkIdx));
            hitCnt++;
        }
    }
    return hitCnt;
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopySlots()
{
    if constexpr (!IsCached) {
        // 保持原专家输出和Combine metadata顺序，只改变cached的读入分组。
        Duplicate(ubHitCount_, (int32_t)0, numLocalExperts_);
        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            Add(ubHitCount_, ubHitCount_, ubRankExpertHitCount_[r * rankExpertRowStride_], numLocalExperts_);
        }
        Cast(ubHitCountRowI64_, ubHitCount_, RoundMode::CAST_NONE, numLocalExperts_);
        Add(ubRowStart_, ubExpertPfx_, ubHitCountRowI64_, numLocalExperts_);
        Duplicate(ubLocalCursor_, (int32_t)0, numLocalExperts_);
        // 专家输出起点已算完，原地归约此前核的计数，省去每rank逐专家的Scalar累加。
        asc_vf_call<MoeEpDispatchEpilogueCount::ReduceRankCorePrefixes>(
            (__ubuf__ int32_t*)ubRankExpertHitCount_.GetPhyAddr(), epWorldSize_, rankExpertRowStride_);
    }
    // 搬运参数在循环外准备，供每个token及其专家命中的读写复用。
    const DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    const DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
    DataCopyExtParams metaOutParams{1U, static_cast<uint32_t>(RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyExtParams localIndexOutParams{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyExtParams weightOutParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};

    // 拍平成一个「取下一个 tile」的循环。原来的 (rank, tile) 双层结构没法把 meta 读提前一拍发出去：
    // 预读必须知道下一拍在哪，而下一拍会跨 rank。
    uint32_t rankId = 0U;
    uint32_t tileStart = 0U;
    uint32_t tileCnt = 0U;
    uint32_t slotStart = 0U;
    uint32_t nextRank = 0U;
    uint32_t nextTileStart = 0U;
    uint32_t nextTileCnt = 0U;
    uint32_t nextSlotStart = 0U;
    uint32_t metaSeq = 0U;
    uint32_t tokenSeq = 0U;
    uint32_t stageSeq = 0U;
    uint32_t planRowCursor = 0U;
    bool hasTile = NextTile(0U, TILE_NONE, rankId, tileStart, tileCnt, slotStart);
    if (hasTile) { // 先发首拍路由，后续在处理当前拍时预取下一拍
        IssueMeta(rankId, slotStart + tileStart, tileCnt, 0U);
    }

    // 先发首批metadata读，再等V结果；真正读取行号和游标前完成同步。
    if constexpr (!IsCached) {
        SyncFunc<AscendC::HardEvent::V_S>();
    }
    while (hasTile) {
        // 当前拍的 meta 读是上一拍发出去的，这里只等它到。
        const uint32_t metaBuf = metaSeq % META_RING;
        WaitFlag<AscendC::HardEvent::MTE2_S>(metaEvtMte2ToS_[metaBuf]);
        ++metaSeq;
        // 先把下一拍定位并预读，再处理当前拍：读延迟盖在当前拍的 MTE3 写里，标量 pipe 不用等。
        hasTile = NextTile(rankId, tileStart, nextRank, nextTileStart, nextTileCnt, nextSlotStart);
        if (hasTile) {
            const uint32_t nextMetaBuf = metaSeq % META_RING;
            if (metaSeq >= META_RING) { // 这一份上一次用是 META_RING 拍之前，先确认标量已经读完
                WaitFlag<AscendC::HardEvent::S_MTE2>(metaEvtSToMte2_[nextMetaBuf]);
            }
            IssueMeta(nextRank, nextSlotStart + nextTileStart, nextTileCnt, nextMetaBuf);
        }

        // direct 本端x 直读输入；hybrid 保持读窗口
        isDirectSelfRank_ = (networkMode_ != NETWORK_HYBRID) && (rankId == static_cast<uint32_t>(epRankId_));
        GM_ADDR srcRankBase = localWinAddr_ + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_;
        const uint32_t metaRingBase = metaBuf * metaRingElems_;
        const uint32_t planBase = metaBuf * planRingElems_;
        const uint32_t globalTileSlot = GlobalSlot(rankId, slotStart + tileStart);
        uint32_t planRowCount = 0U;
        if constexpr (!IsCached) {
            if (metaSeq > META_RING) {
                WaitFlag<AscendC::HardEvent::MTE3_S>(planEvtMte3ToS_[metaBuf]);
            }
            if (tileStart == 0U) {
                // 每行首元素已由V归约为本rank此前核的命中总数。
                planRowCursor =
                    ubRankOffsets_.GetValue(rankId) + ubRankExpertHitCount_.GetValue(rankId * rankExpertRowStride_);
            }
        }

        for (uint32_t localSlot = 0; localSlot < tileCnt; ++localSlot) {
            uint32_t metaBase = metaRingBase + localSlot * paddedMetaElems_;
            int32_t srcRankMeta = ubMeta_.GetValue(metaBase + META_TOPK_SECTION * axisKAlign_);
            int32_t tokenIdxMeta = ubMeta_.GetValue(metaBase + META_TOPK_SECTION * axisKAlign_ + 1);
            GM_ADDR slotAddr = srcRankBase + (int64_t)(slotStart + tileStart + localSlot) * perSlotBytes_;

            const uint32_t tokenBuf = tokenSeq % TOKEN_RING;
            LocalTensor<XType> tokenOut =
                IssueTokenRead(slotAddr, tokenIdxMeta, tokenSeq++, tokenCopyParams, scalesCopyParams);

            const uint32_t hitCnt = BuildSlotHits(metaBuf, localSlot, globalTileSlot);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(tokenEvtFill_[tokenBuf]);
            if (hitCnt == 0U) {
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(tokenEvtFree_[tokenBuf]);
                if constexpr (!IsCached) {
                    ubSlotPlan_.SetValue(planBase + localSlot, planRowCursor + planRowCount);
                }
                continue;
            }
            uint32_t stageBuf = 0U;
            uint32_t stageOff = 0U;
            if constexpr (!IsCached || HasTopkWeights) {
                stageBuf = stageSeq % STAGE_RING;
                if (stageSeq >= STAGE_RING) {
                    WaitFlag<AscendC::HardEvent::MTE3_S>(stageEvtMte3ToS_[stageBuf]);
                }
                ++stageSeq;
                stageOff = stageBuf * stageSlotElems_;
            }

            // 同一遍命中循环下发hidden并准备metadata，避免为代码复用额外扫描命中表。
            for (uint32_t i = 0; i < hitCnt; ++i) {
                const int64_t recvXRow = ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_ROW_OFFSET);
                DataCopyPad(recvXGm_[recvXRow * axisH_], tokenOut, tokenCopyParams);
                if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                    DataCopyPad(recvScalesGm_[recvXRow * scalesElems_],
                                tokenOut[scalesStride_].template ReinterpretCast<ScalesType>(), scalesCopyParams);
                }
                if constexpr (!IsCached || HasTopkWeights) {
                    const uint32_t topkIdx =
                        static_cast<uint32_t>(ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET));
                    if constexpr (HasTopkWeights) {
                        float weights = ubMeta_.ReinterpretCast<float>().GetValue(metaBase + axisKAlign_ + topkIdx);
                        ubStageWeightsRing_.SetValue(stageOff + i * ELEM_ALIGN, weights);
                    }
                    if constexpr (!IsCached) {
                        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_SRC_RANK_OFFSET, srcRankMeta);
                        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_TOKEN_IDX_OFFSET, tokenIdxMeta);
                        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_TOPK_IDX_OFFSET,
                                                  static_cast<int32_t>(topkIdx));
                        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_SLOT_IDX_OFFSET,
                                                  static_cast<int32_t>(slotStart + tileStart + localSlot));
                        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_RECV_X_IDX_OFFSET,
                                                  static_cast<int32_t>(recvXRow));
                        if (srcRankMeta == static_cast<int32_t>(epRankId_)) {
                            ubStageLocalIndexRing_.SetValue(stageOff + i * ELEM_ALIGN, static_cast<int32_t>(recvXRow));
                        }
                    }
                }
            }
            // 全部hidden/scales请求入队后归还token，metadata的stage仍由独立事件保护。
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(tokenEvtFree_[tokenBuf]);
            if constexpr (!IsCached || HasTopkWeights) {
                // stage 的 S → MTE3 握手：Set/Wait 紧邻，任何时刻只有一发在飞，所以一条 id 就够。
                SetFlag<AscendC::HardEvent::S_MTE3>(stageEvtSToMte3_);
                WaitFlag<AscendC::HardEvent::S_MTE3>(stageEvtSToMte3_);
                // loop B：从 ubStage*Ring_ 读，所以归还要压在这个循环的 SetFlag<MTE3_S> 之前。
                for (uint32_t i = 0; i < hitCnt; i++) {
                    int64_t recvXRow = ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_ROW_OFFSET);
                    if constexpr (HasTopkWeights) {
                        DataCopyPad(recvTopkWeightsGm_[recvXRow], ubStageWeightsRing_[stageOff + i * ELEM_ALIGN],
                                    weightOutParams);
                    }
                    if constexpr (!IsCached) {
                        // Output metadata in source-row order without changing token/weight/scales placement.
                        uint32_t topkIdx =
                            static_cast<uint32_t>(ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET));
                        uint32_t localExpertId =
                            static_cast<uint32_t>(ubMeta_.GetValue(metaBase + topkIdx)) - epRankId_ * numLocalExperts_;
                        uint32_t rankExpertIndex = rankId * rankExpertRowStride_ + localExpertId;
                        int32_t metadataRow = ubRankExpertRowStart_.GetValue(rankExpertIndex);
                        DataCopyPad(recvSrcMetadataGm_[(int64_t)metadataRow * RECV_META_FIELDS],
                                    ubStageMetaRing_[stageOff + i * ELEM_ALIGN], metaOutParams);
                        if (srcRankMeta == static_cast<int32_t>(epRankId_)) {
                            uint32_t lookupIndex = static_cast<uint32_t>(tokenIdxMeta) * axisK_ + topkIdx;
                            DataCopyPad(localRecvIndexGm_[lookupIndex],
                                        ubStageLocalIndexRing_[stageOff + i * ELEM_ALIGN], localIndexOutParams);
                        }
                        ubRankExpertRowStart_.SetValue(rankExpertIndex, metadataRow + 1);
                        ubSlotPlan_.SetValue(planBase + planEndsElems_ + planRowCount++, metadataRow);
                    }
                }
                SetFlag<AscendC::HardEvent::MTE3_S>(stageEvtMte3ToS_[stageBuf]);
            }
            if constexpr (!IsCached) {
                ubSlotPlan_.SetValue(planBase + localSlot, planRowCursor + planRowCount);
            }
        }
        if constexpr (!IsCached) {
            // 每批连续写两个小数组；避免逐slot/命中增加4字节DMA和等待。
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(slotEndsGm_[globalTileSlot], ubSlotPlan_[planBase],
                        {1U, tileCnt * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
            if (planRowCount != 0U) {
                DataCopyPad(slotRowIdsGm_[planRowCursor], ubSlotPlan_[planBase + planEndsElems_],
                            {1U, planRowCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
            }
            SetFlag<AscendC::HardEvent::MTE3_S>(planEvtMte3ToS_[metaBuf]);
            planRowCursor += planRowCount;
        }

        // 这一拍的 meta 已经被标量全部读完（上面所有 ubMeta_ 的 GetValue），可以还给 MTE2 了。
        // 只有还有下一拍时才发：这样 S_MTE2 上「每次 Wait 恰好一次 Set」，不攒多余信用。
        if (hasTile) {
            SetFlag<AscendC::HardEvent::S_MTE2>(metaEvtSToMte2_[metaBuf]);
        }
        rankId = nextRank;
        tileStart = nextTileStart;
        tileCnt = nextTileCnt;
        slotStart = nextSlotStart;
    }
    // 这里不再做 tile 尾的 MTE3 排空：Caller(Process) 紧接着就有 SyncFunc<MTE3_S> 兜住 recv_x/metadata/weights
    // 的可见性，环的事件又是 AllocEventID 私有、不会串到别处，所以函数返回时的 pipe 状态不需要在这里再等一次。
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline uint32_t MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::GlobalSlot(
    uint32_t rankId, uint32_t slotId)
{
    for (uint32_t r = 0; r < rankId; ++r) {
        slotId += static_cast<uint32_t>(ubRecvCnt_.GetValue(r));
    }
    return slotId;
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyCachedArray(
    GlobalTensor<int32_t> src, GlobalTensor<int32_t> dst, uint32_t count)
{
    uint32_t start, end, blocks;
    SplitToCore(Ceil(count, INT32_PER_BLOCK), aivNum_, start, end, blocks);
    start *= INT32_PER_BLOCK;
    end = end * INT32_PER_BLOCK > count ? count : end * INT32_PER_BLOCK;
    // 借用搬运前尚未使用的meta缓冲，按32字节边界分核，避免输出块交叠。
    for (uint32_t offset = start; offset < end;) {
        uint32_t elements = end - offset > metaRingElems_ ? metaRingElems_ : end - offset;
        DataCopyExtParams params{1U, elements * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(ubMeta_, src[offset], params, {false, 0U, 0U, 0});
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(dst[offset], ubMeta_, params);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        offset += elements;
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyCachedMetadata()
{
    CopyCachedArray(cachedRecvSrcMetadataGm_, recvSrcMetadataGm_, expertSum_ * RECV_META_FIELDS);
    CopyCachedArray(cachedSlotEndsGm_, slotEndsGm_, GlobalSlot(epWorldSize_, 0U));
    CopyCachedArray(cachedSlotRowIdsGm_, slotRowIdsGm_, expertSum_);
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline LocalTensor<XType> MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::IssueTokenRead(
    GM_ADDR slotAddr, int32_t tokenId, uint32_t sequence, const DataCopyParams& tokenCopyParams,
    const DataCopyParams& scalesCopyParams)
{
    const uint32_t buffer = sequence % TOKEN_RING;
    if (sequence >= TOKEN_RING) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tokenEvtFree_[buffer]);
    }
    LocalTensor<XType> token = tokenRing_[buffer * tokenRingElems_];
    GlobalTensor<XType> input;
    if (isDirectSelfRank_) {
        input = xGm_[static_cast<int64_t>(tokenId) * axisH_];
    } else {
        input.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(slotAddr), axisH_);
    }
    // 输入不经过L2，节省搬运时间。
    input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopyPad(token, input, tokenCopyParams, {false, 0U, 0U, 0});
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        GlobalTensor<ScalesType> scales;
        scales.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType*>(slotAddr + scalesOffset_), scalesElems_);
        DataCopyPad(token[scalesStride_].template ReinterpretCast<ScalesType>(), scales, scalesCopyParams,
                    {false, 0U, 0U, 0});
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(tokenEvtFill_[buffer]);
    return token;
}

#endif

} // namespace MoeEpDispatchEpilogueImpl

#endif // MOE_EP_DISPATCH_EPILOGUE_H
