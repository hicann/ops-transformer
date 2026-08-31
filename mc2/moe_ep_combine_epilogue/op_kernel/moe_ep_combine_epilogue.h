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
 * \file moe_ep_combine_epilogue.h
 * \brief MoE Expert-Parallel Combine Epilogue kernel — recv + reduce phase.
 *        Reads expert outputs from HCCL Window, accumulates per-token topK results,
 *        and writes combined_x / combined_topk_weights.
 */
#ifndef MOE_EP_COMBINE_EPILOGUE_H
#define MOE_EP_COMBINE_EPILOGUE_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_COMBINE_EPILOGUE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#include <cstddef>
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

#include "moe_ep_combine_epilogue_tiling_key.h"
#include "moe_ep_combine_epilogue_tiling.h"

namespace MoeEpCombineEpilogueImpl {

#if defined(ENABLE_MOE_EP_COMBINE_EPILOGUE_KERNEL)

using namespace AscendC;

#define TemplateMoeEpCombineEpilogueTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineEpilogueTypeFunc XType, HasTopkWeight

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;

template <TemplateMoeEpCombineEpilogueTypeClass>
class MoeEpCombineEpilogue {
public:
    __aicore__ inline MoeEpCombineEpilogue(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR topkIdx, GM_ADDR combinedX, GM_ADDR combinedTopkWeights,
                                GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpCombineEpilogueInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &tokenPerAivNum);
    __aicore__ inline void BuffInit();
    __aicore__ inline bool WaitDispatch(uint32_t completionChannelCount);
    __aicore__ inline void ProcessTopKToken(uint32_t tokenIndex);
    __aicore__ inline void RecvPhaseReduce();

    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    TPipe *tpipe_{nullptr};
    const MoeEpCombineEpilogueInfo *tilingData_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t aivNum_{0};
    uint32_t aivId_{0};

    uint32_t tStart_{0};
    uint32_t tEnd_{0};
    uint32_t tPerCore_{0};

    GlobalTensor<XType> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;

    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> xOutQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> weightQue_;
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<> stateBuf_;
    TBuf<> stateSumBuf_;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
};

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::Init(
    GM_ADDR context, GM_ADDR topkIdx, GM_ADDR combinedX, GM_ADDR combinedTopkWeights, GM_ADDR workspace,
    GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineEpilogueInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    numTokens_ = tilingData_->cfg.numTokens;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    aivNum_ = tilingData->aivNum;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;

    combinedXGm_.SetGlobalBuffer((__gm__ XType *)combinedX);

    if constexpr (HasTopkWeight == 1) {
        combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)combinedTopkWeights);
    }

    constexpr size_t metadataOffset = offsetof(MoeEpCombineEpilogueTilingData, moeEpCombineEpilogueInfo) +
                                      offsetof(MoeEpCombineEpilogueInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_COMBINE_EPILOGUE, tpipe_);
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_EPILOGUE_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
    uint32_t newAivId = aivId_;

    startTokenId = sendTokenNum * newAivId;
    if (newAivId < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::BuffInit()
{
    SplitToCore(numTokens_, aivNum_, tStart_, tEnd_, tPerCore_);
    if (tStart_ >= numTokens_) {
        return;
    }

    if constexpr (HasTopkWeight == 1) {
        tpipe_->InitBuffer(weightQue_, DOUBLE_BUFFER_NUM, UB_ALIGN);
    }
    tpipe_->InitBuffer(xInQue_, DOUBLE_BUFFER_NUM, hAlignSize_);
    tpipe_->InitBuffer(xOutQue_, DOUBLE_BUFFER_NUM, hAlignSize_);
    uint32_t ubFp32Bytes = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubAccFp32Buf_, ubFp32Bytes);
    tpipe_->InitBuffer(ubTmpFp32Buf_, ubFp32Bytes);

    ubAccFp32_ = ubAccFp32Buf_.Get<float>();
    ubTmpFp32_ = ubTmpFp32Buf_.Get<float>();
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline bool MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::WaitDispatch(
    uint32_t completionChannelCount)
{
    uint32_t completionFlagCount = epWorldSize_ * completionChannelCount;
    uint64_t flagOffset = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN;
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagOffset;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);

    LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopyExtParams params = {static_cast<uint16_t>(completionFlagCount), sizeof(uint32_t),
                                WIN_ADDR_ALIGN - sizeof(uint32_t), 0, 0};
    DataCopyPadExtParams<uint32_t> padParams = {true, 0, 0, 0};

    DataCopyPad<uint32_t>(stateTensor, stateGMTensor, params, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<uint32_t> stateSumTensor = stateSumBuf_.Get<uint32_t>();
    uint32_t shape[] = {completionFlagCount, UB_ALIGN / sizeof(uint32_t)};
    ReduceSum<uint32_t, AscendC::Pattern::Reduce::RA, false>(stateSumTensor, stateTensor, shape, true);
    SyncFunc<AscendC::HardEvent::V_S>();

    return stateSumTensor(0) == completionFlagCount;
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::ProcessTopKToken(uint32_t tokenIndex)
{
    Duplicate<float>(ubAccFp32_, (float)0, axisH_);
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};
    DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
    for (uint32_t topkId = 0U; topkId < topK_; topkId++) {
        uint64_t slotOffset = (static_cast<uint64_t>(tokenIndex) * topK_ + topkId) * tilingData_->cfg.perSlotBytes;
        GM_ADDR wAddr = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_) + slotOffset;
        GlobalTensor<XType> srcTokenTensor;
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(wAddr));
        LocalTensor<XType> xLocal = xInQue_.AllocTensor<XType>();
        DataCopyPad(xLocal, srcTokenTensor, xCopyParams, padParams);
        xInQue_.EnQue(xLocal);
        LocalTensor<XType> xIn = xInQue_.DeQue<XType>();
        Cast(ubTmpFp32_, xIn, AscendC::RoundMode::CAST_NONE, axisH_);
        Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, axisH_);
        xInQue_.FreeTensor(xIn);
        if constexpr (HasTopkWeight == 1) {
            GM_ADDR weightAddr = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) +
                                 (tokenIndex * topK_ + topkId) * WIN_ADDR_ALIGN;
            GlobalTensor<float> srcWeightTensor;
            srcWeightTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightAddr));
            LocalTensor<float> weightLocal = weightQue_.AllocTensor<float>();
            DataCopyPad(weightLocal, srcWeightTensor, weightCopyParams, padParams);
            weightQue_.EnQue(weightLocal);
            LocalTensor<float> weightOut = weightQue_.DeQue<float>();
            DataCopyPad(combinedTopkWeightsGm_[tokenIndex * topK_ + topkId], weightOut, weightCopyParams);
            weightQue_.FreeTensor(weightOut);
        }
    }
    LocalTensor<XType> ubResultBf16 = xOutQue_.AllocTensor<XType>();
    Cast(ubResultBf16, ubAccFp32_, RoundMode::CAST_RINT, axisH_);
    xOutQue_.EnQue(ubResultBf16);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::RecvPhaseReduce()
{
    constexpr uint32_t completionChannelCount = 1U;
    uint32_t completionFlagCount = epWorldSize_ * completionChannelCount;
    uint32_t flagBufferBytes = completionFlagCount * STATE_OFFSET;
    tpipe_->InitBuffer(stateBuf_, flagBufferBytes);
    tpipe_->InitBuffer(stateSumBuf_, UB_ALIGN);

    if (aivId_ == 0U) {
        while (!WaitDispatch(completionChannelCount)) {
        }
        LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
        SyncFunc<AscendC::HardEvent::S_V>();
        Duplicate<uint32_t>(stateTensor, static_cast<uint32_t>(0), flagBufferBytes / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        uint64_t flagOffset = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN;
        GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagOffset;
        GlobalTensor<uint32_t> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);
        DataCopyExtParams clearParams = {static_cast<uint16_t>(completionFlagCount), STATE_OFFSET, 0,
                                         WIN_ADDR_ALIGN - STATE_OFFSET, 0};
        DataCopyPad<uint32_t>(stateGMTensor, stateTensor, clearParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    SyncAll<true>();
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_EPILOGUE_RUN_POS_WAIT_DONE);

    if (tPerCore_ == 0) {
        return;
    }

    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    for (uint32_t tokenIdx = tStart_; tokenIdx < tEnd_; ++tokenIdx) {
        ProcessTopKToken(tokenIdx);
        LocalTensor<XType> ubResult = xOutQue_.DeQue<XType>();
        DataCopyPad(combinedXGm_[tokenIdx * axisH_], ubResult, xCopyParams);
        xOutQue_.FreeTensor(ubResult);
    }
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_EPILOGUE_RUN_POS_OUTPUT_DONE);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::Process()
{
    BuffInit();
    RecvPhaseReduce();
}

#endif

} // namespace MoeEpCombineEpilogueImpl

#endif // MOE_EP_COMBINE_EPILOGUE_H
