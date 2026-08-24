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
 * \file moe_ep_combine.h
 * \brief MoE Expert-Parallel Combine kernel implementation — send phase only.
 *        Sends expert output tokens to remote ranks' HCCL Window buffers.
 *        The recv+reduce phase is handled by moe_ep_combine_epilogue.
 */
#ifndef MOE_EP_COMBINE_H
#define MOE_EP_COMBINE_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_COMBINE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif

#include "moe_ep_combine_tiling_key.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "moe_ep_combine_base.h"
#include "moe_ep_combine_tiling.h"

namespace MoeEpCombineImpl {

#if defined(ENABLE_MOE_EP_COMBINE_KERNEL)

using namespace AscendC;

#define TemplateMoeEpCombineTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineTypeFunc XType, HasTopkWeight
#define HCOMM_INIT_SIZE 512UL

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 4;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint32_t SEND_DOUBLE_BUFFER_NUM = 2U;
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};

template <TemplateMoeEpCombineTypeClass>
class MoeEpCombine {
public:
    __aicore__ inline MoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr);
    __aicore__ inline void SendSlot(uint32_t tokenIndex, int32_t srcRank, int32_t srcTokenIdx, int32_t srcTopKIdx,
                                    uint32_t channelIndex, uint32_t weight);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup,
                                             uint32_t &groupSize);

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId, uint32_t channelIndex)
    {
        return hcommHandle_[rankId * channelsPerRank_ + channelIndex];
    }
    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }
    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
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
    const MoeEpCombineInfo *tilingData_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t channelsPerRank_{1};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t hWeightAlignSize_{0};
    uint32_t XTypeAlign32Size_{0};
    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint32_t aivNum_{0};
    uint32_t localmoeNum_{0};

    uint32_t aivId_{0};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;
    GlobalTensor<float> topkWeightsGm_;

    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<uint8_t> hcommTensor_;

    TBuf<> readStateBuf_;
    TBuf<> hcommBuf_;
    TBuf<> metadataBuf_;
    TBuf<> weightsBuf_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx,
                                                                        GM_ADDR recvSrcMetadata,
                                                                        GM_ADDR numRecvPerExpert, GM_ADDR topkWeights,
                                                                        GM_ADDR tilingGM, TPipe *pipe,
                                                                        const MoeEpCombineInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;
    localmoeNum_ = tilingData_->cfg.numLocalExperts;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    hWeightAlignSize_ = hAlignSize_ + UB_ALIGN;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    rankId_ = mc2Context_->epRankId;
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (channelsPerRank_ == 0 || (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_)) {
        channelsPerRank_ = 1;
    }
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    uint32_t handleCount = epWorldSize_ * channelsPerRank_;
    for (uint32_t i = 0; i < handleCount; ++i) {
        hcommHandle_[i] = mc2Context_->hcommHandle[i];
    }

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;

    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);

    // compute actualA_ from numRecvPerExpert
    uint32_t numRecvBytes = Ceil(localmoeNum_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t reduceTmpBytes = ReduceSumWorkNeedSize(localmoeNum_, sizeof(int64_t)) * sizeof(int64_t);
    TBuf<TPosition::VECCALC> numRecvBuf;
    tpipe_->InitBuffer(numRecvBuf, numRecvBytes + reduceTmpBytes + UB_ALIGN);
    LocalTensor<int64_t> numRecvLocal = numRecvBuf.Get<int64_t>();

    DataCopyExtParams numRecvCopyParams{1U, static_cast<uint32_t>(localmoeNum_ * sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int64_t> numRecvPadParams{false, 0U, 0U, 0U};
    DataCopyPad(numRecvLocal, numRecvPerExpertGm_, numRecvCopyParams, numRecvPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<int64_t> reduceTmp = numRecvLocal[numRecvBytes / sizeof(int64_t)];
    LocalTensor<int64_t> sumOut = reduceTmp[reduceTmpBytes / sizeof(int64_t)];
    ReduceSum<int64_t>(sumOut, numRecvLocal, reduceTmp, static_cast<int32_t>(localmoeNum_));
    SyncFunc<AscendC::HardEvent::V_S>();
    actualA_ = static_cast<uint64_t>(sumOut.GetValue(0));

    XTypeAlign32Size_ = hAlignSize_;
    if constexpr (HasTopkWeight == 1) {
        XTypeAlign32Size_ = hWeightAlignSize_;
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    tpipe_->InitBuffer(xQueue_, SEND_DOUBLE_BUFFER_NUM, XTypeAlign32Size_);
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);
    statusTensor_ = readStateBuf_.Get<uint32_t>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr)
{
    GlobalTensor<XType> outToken;
    outToken.SetGlobalBuffer((__gm__ XType *)dstAddr);

    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    LocalTensor<XType> tokenTensor = xQueue_.AllocTensor<XType>();
    DataCopyPad(tokenTensor, xGm_[tokenIndex * axisH_], copyParams, padParams);
    xQueue_.EnQue(tokenTensor);
    tokenTensor = xQueue_.DeQue<XType>();
    DataCopyPad(outToken, tokenTensor, copyParams);
    xQueue_.FreeTensor<XType>(tokenTensor);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendSlot(uint32_t tokenIndex, int32_t srcRank,
                                                                            int32_t srcTokenIdx, int32_t srcTopKIdx,
                                                                            uint32_t channelIndex, uint32_t weight)
{
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;

    GM_ADDR remoteRankWinAddr = GetUrmaWinAddrByRankId(srcRank, combineDataWinOffset_);
    GM_ADDR remoteRankStateAddr = GetUrmaStateAddrByRankId(srcRank, combineStateWinOffset_);

    if (srcRank != static_cast<int32_t>(rankId_)) {
        GM_ADDR tokenAddr = (GM_ADDR)xGm_.GetPhyAddr(tokenIndex * axisH_);
        uint64_t commHandle = GetCommHandle(srcRank, channelIndex);
        uint64_t notifyValue = 1ULL << 32;
        if constexpr (HasTopkWeight == 1) {
            notifyValue |= weight;
        }
        hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DEFAULT_WQE_CONFIG>(
            commHandle, remoteRankWinAddr + sendTokenOffset, tokenAddr, axisH_ * sizeof(XType),
            remoteRankStateAddr + recvStateOffset, notifyValue);
    } else {
        SendLocalToken(tokenIndex, remoteRankWinAddr + sendTokenOffset);
        if constexpr (HasTopkWeight == 1) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            statusTensor_(0) = weight;
            SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
        GlobalTensor<uint32_t> state;
        state.SetGlobalBuffer((__gm__ uint32_t *)(remoteRankStateAddr + recvStateOffset));
        DataCopy(state, statusTensor_, FLOAT_PER_UB_ALIGN);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(uint32_t totalBlocks,
                                                                                     uint32_t &targetRank,
                                                                                     uint32_t &coreIndexInGroup,
                                                                                     uint32_t &groupSize)
{
    uint32_t baseGroupSize = totalBlocks / epWorldSize_;
    uint32_t remainder = totalBlocks % epWorldSize_;
    uint32_t accumulated = 0;
    for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
        uint32_t currentGroupSize = baseGroupSize + ((rank < remainder) ? 1U : 0U);
        if (aivId_ < accumulated + currentGroupSize) {
            targetRank = rank;
            groupSize = currentGroupSize;
            coreIndexInGroup = aivId_ - accumulated;
            return;
        }
        accumulated += currentGroupSize;
    }
    targetRank = epWorldSize_;
    groupSize = 0;
    coreIndexInGroup = 0;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseExpertToToken()
{
    if (actualA_ == 0 || epWorldSize_ == 0 || aivNum_ == 0) {
        return;
    }
    uint32_t activeAivNum = aivNum_;
    uint32_t maxChannelAivNum = epWorldSize_ * channelsPerRank_;
    if (activeAivNum > maxChannelAivNum) {
        activeAivNum = maxChannelAivNum;
    }
    if (aivId_ >= activeAivNum) {
        return;
    }

    bool splitRankTokens = activeAivNum >= epWorldSize_;
    uint32_t targetRank = epWorldSize_;
    uint32_t coreIndexInGroup = 0;
    uint32_t groupSize = 1;
    if (splitRankTokens) {
        GetCoreAssignment(activeAivNum, targetRank, coreIndexInGroup, groupSize);
        if (targetRank >= epWorldSize_ || groupSize == 0) {
            return;
        }
    }
    uint32_t channelIndex = splitRankTokens ? coreIndexInGroup : 0;

    uint64_t scanStart = 0;
    uint64_t scanEnd = actualA_;
    if (splitRankTokens && groupSize > 1) {
        uint64_t tokensPerCore = actualA_ / groupSize;
        uint64_t remainder = actualA_ % groupSize;
        if (coreIndexInGroup < remainder) {
            scanStart = coreIndexInGroup * (tokensPerCore + 1);
            scanEnd = scanStart + tokensPerCore + 1;
        } else {
            scanStart = coreIndexInGroup * tokensPerCore + remainder;
            scanEnd = scanStart + tokensPerCore;
        }
    }

    bool statusInitialized = false;

    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    constexpr uint32_t metaChunkTokenMax = 8192U;

    uint32_t metaChunkTokens = (actualA_ < metaChunkTokenMax) ? actualA_ : metaChunkTokenMax;
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * metaBytesPerToken, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(metadataBuf_, metaChunkBytes);
    if constexpr (HasTopkWeight == 1) {
        uint32_t weightChunkBytes = Ceil(metaChunkTokens * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(weightsBuf_, weightChunkBytes);
    }
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};

    for (uint64_t chunkStart = scanStart; chunkStart < scanEnd; chunkStart += metaChunkTokens) {
        uint64_t chunkEnd = (chunkStart + metaChunkTokens > scanEnd) ? scanEnd : (chunkStart + metaChunkTokens);
        uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);

        LocalTensor<uint32_t> weightsForNotify;
        if constexpr (HasTopkWeight == 1) {
            weightsForNotify = weightsBuf_.Get<uint32_t>();
            DataCopyExtParams weightCopyParams{1U, static_cast<uint32_t>(curChunkTokens * sizeof(uint32_t)), 0U, 0U,
                                               0U};
            const DataCopyPadExtParams<uint32_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(weightsForNotify, topkWeightsGm_[chunkStart].template ReinterpretCast<uint32_t>(),
                        weightCopyParams, padParams);
        }

        DataCopyExtParams metaCopyParams{1U, curChunkTokens * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < curChunkTokens; ++i) {
            uint32_t tokenIndex = static_cast<uint32_t>(chunkStart) + i;
            int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS + 0);
            if (srcRank < 0 || srcRank >= static_cast<int32_t>(epWorldSize_)) {
                continue;
            }
            if (splitRankTokens) {
                if (srcRank != static_cast<int32_t>(targetRank)) {
                    continue;
                }
            } else if (static_cast<uint32_t>(srcRank) % activeAivNum != aivId_) {
                continue;
            }
            int32_t srcTokenIdx = metadataLocal.GetValue(i * RECV_META_FIELDS + 1);
            int32_t srcTopKIdx = metadataLocal.GetValue(i * RECV_META_FIELDS + 2);
            uint32_t weight = 0;
            if constexpr (HasTopkWeight == 1) {
                weight = weightsForNotify(i);
            }
            if (srcRank == static_cast<int32_t>(rankId_) && !statusInitialized) {
                Duplicate<uint32_t>(statusTensor_, (uint32_t)1, FLOAT_PER_UB_ALIGN);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                statusInitialized = true;
            }
            SendSlot(tokenIndex, srcRank, srcTokenIdx, srcTopKIdx, channelIndex, weight);
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }

    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Process()
{
    SendPhaseExpertToToken();
}

#endif

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
