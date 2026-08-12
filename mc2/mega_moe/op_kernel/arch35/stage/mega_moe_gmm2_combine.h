/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_COMBINE_H
#define MEGA_MOE_GMM2_COMBINE_H

#include <type_traits>

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "adv_api/reduce/reduce.h"
#include "../common/mega_moe_gmm_common.h"
#include "../common/mega_moe_utils.h"
#include "../common/mega_moe_mxfp8_utils.h"
#if __has_include("../../../common/mc2_kernel_utils.h")
#include "../../../common/mc2_kernel_utils.h"
#include "../../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

namespace CombineImpl {

// 为 tile 内每个 token 行发送一段有效的 GMM2 tile 数据。
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(uint32_t nLoc, uint32_t n, LocalTensor<int32_t> &metaInfoTensor,
                                     LocalTensor<ElementMMadOut2> &l0cOutUbGMM2, BlockShape &actualBlockShape,
                                     uint32_t ubTileN, const Params &params)
{
    // 调用方在进入该数据操作前，保证批量加载的 metadata 对 Scalar 可见。
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = metaInfoTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = metaInfoTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = metaInfoTensor.GetValue(tileIdx * 8 + 2);
        gmRemoteD.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementMMadOut2 *>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[tileIdx * ubTileN], ub2GmParams);
    }
}

// 发送一行完整的 Combine 数据，BF16 与 FP8 记录复用相同的定长搬运流程。
template <typename Element>
__aicore__ inline void SendCombineTokenRow(uint32_t rowElements, uint64_t gmRemoteBaseOffset,
                                           LocalTensor<int32_t> &metaInfoTensor, LocalTensor<Element> &rowTensor,
                                           const Params &params)
{
    uint32_t toRankId = metaInfoTensor.GetValue(RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(TOPK_INDEX);

    AscendC::GlobalTensor<Element> gmRemoteD;
    gmRemoteD.SetGlobalBuffer(
        reinterpret_cast<__gm__ Element *>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
    uint64_t gmDstRowOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * rowElements;

    constexpr uint32_t TRANSFER_BYTES = 512U;
    constexpr uint32_t TILE_ELEMENTS = TRANSFER_BYTES / sizeof(Element);
    AscendC::DataCopyExtParams ub2GmParams{1U, 0U, 0U, 0U, 0U};
    for (uint32_t elementOffset = 0U; elementOffset < rowElements; elementOffset += TILE_ELEMENTS) {
        uint32_t remainingElements = rowElements - elementOffset;
        uint32_t currentElements = remainingElements < TILE_ELEMENTS ? remainingElements : TILE_ELEMENTS;
        ub2GmParams.blockLen = currentElements * sizeof(Element);
        AscendC::DataCopyPad(gmRemoteD[gmDstRowOffset + elementOffset], rowTensor[elementOffset], ub2GmParams);
    }
}

// 通过本地 MTE 路径或远端 URMA 路径发送一个 layered 模板 token。
template <typename DataType, bool IsQuantized = true>
__aicore__ inline void CombineSendTokenToRemote(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                                uint32_t groupIdx, uint32_t rankId,
                                                LocalTensor<int32_t> &metaInfoTensor, LocalTensor<DataType> &ubQuant,
                                                const Params &params, GM_ADDR localSrcPtr)
{
#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID3>();
    int64_t quantTokenSize = IsQuantized ? (n + nScale) : n;
    uint32_t toRankId = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOPK_INDEX);

    AscendC::GlobalTensor<DataType> gmLocalD;
    uint64_t gmRemoteOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    GM_ADDR srcAddr = localSrcPtr;

    if (toRankId == rankId) {
        srcAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
    }

    gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(srcAddr));
    uint64_t dstBaseOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantTokenSize;
    AscendC::DataCopyExtParams singleCopyParams{1, static_cast<uint32_t>(quantTokenSize * sizeof(DataType)), 0, 0, 0};

    if constexpr (!IsQuantized) {
        DataCopyPadExtParams<DataType> copyPadParams{false, 0U, 0U, 0U};
        if (toRankId == rankId) {
            AscendC::GlobalTensor<DataType> gmm2OutGm;
            gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(localSrcPtr));
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
            AscendC::DataCopyPad(ubQuant, gmm2OutGm, singleCopyParams, copyPadParams);
            SyncFuncStatic<AscendC::HardEvent::MTE2_MTE3, SYNC_EVENT_ID4>();
        }
    }

    if (IsQuantized || toRankId == rankId) {
        uint64_t dstOffset = toRankId == rankId ? dstBaseOffset : 0;
        AscendC::DataCopyPad(gmLocalD[dstOffset], ubQuant, singleCopyParams);
        SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID4>();
    }

    if (toRankId != rankId) {
        uint64_t channelHandle = GetUrmaCommHandle(params.combineCommParams.mc2Context, toRankId, rankId);
        GM_ADDR remoteAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset) + dstBaseOffset * sizeof(DataType);
        params.combineCommParams.hcomm->WriteNbi(channelHandle, remoteAddr, srcAddr,
                                                 quantTokenSize * sizeof(DataType));
    }
#endif
}

// 将一条量化 token 记录发送到 metadata 指定的 rank 和目标行。
template <typename QuantOutType>
__aicore__ inline void CombineQuantizedTokens(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                              uint32_t groupIdx, uint32_t rankId,
                                              LocalTensor<int32_t> &metaInfoTensor,
                                              LocalTensor<QuantOutType> &ubQuant, const Params &params,
                                              uint32_t quantTokenSizeBytes)
{
    uint32_t toRankId = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOPK_INDEX);

    AscendC::GlobalTensor<QuantOutType> gmRemoteD;
    uint64_t gmRemoteOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    __gm__ void *dstPeermemPtr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
    gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(dstPeermemPtr));

    uint64_t dstBaseOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantTokenSizeBytes;
    AscendC::DataCopyExtParams singleCopyParams{1, quantTokenSizeBytes, 0, 0, 0};
    AscendC::DataCopyPad(gmRemoteD[dstBaseOffset], ubQuant, singleCopyParams);
}

// 读取并按需量化一组普通/layered Combine token，然后发送到目标 rank。
template <uint8_t QuantMode, typename T, bool IsLayered = false, bool IsQuantized = true>
__aicore__ inline void CombineTokenGroup(uint32_t tokenStart, uint32_t tokenCount, uint32_t n, uint32_t groupIdx,
                                         uint32_t rankId, GM_ADDR gmm2OutAddr, const Params &params,
                                         LocalTensor<int32_t> &metaInfoTensor, int64_t ubTensorSize, int64_t offset,
                                         uint32_t quantTokenSizeBytes)
{
    LocalTensor<T> combineUbTensor(TPosition::VECIN, offset, ubTensorSize);
    offset += ubTensorSize * sizeof(T);

    uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t mxScaleNum = Align2(nScale);
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
    uint32_t floatTempSize = Align32(mxScaleNum) + mxScaleNum / 2;
    LocalTensor<float> floatTemp = LocalTensor<float>(TPosition::VECIN, offset, floatTempSize);

    GlobalTensor<T> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmm2OutAddr));
    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;

    uint32_t singleTokenElems = (nAlign32 * sizeof(T) + quantTokenSizeBytes) / sizeof(T);
    DataCopyPadExtParams<T> copyPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams gm2UbParams{static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};

    for (uint32_t i = 0; i < tokenCount; i++) {
        uint32_t pingPong = i % 2;
        LocalTensor<T> ubBf16 = combineUbTensor[pingPong * singleTokenElems];
        LocalTensor<T> ubQuantData = ubBf16[nAlign32];

        if constexpr (IsQuantized) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
            AscendC::DataCopyPad(ubBf16, gmm2OutGm[(tokenStart + i) * n], gm2UbParams, copyPadParams);
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID4>();
            Mxfp8::QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();
        }

        using SendType = typename std::conditional<IsQuantized, Fp8Type, T>::type;
        LocalTensor<SendType> ubQuantSend = ubQuantData.template ReinterpretCast<SendType>();
        if constexpr (IsLayered) {
            if constexpr (!IsQuantized) {
                ubQuantSend = ubBf16;
            }
            GM_ADDR localSrcPtr = gmm2OutAddr + (tokenStart + i) * n * sizeof(T);
            CombineSendTokenToRemote<SendType, IsQuantized>(i, 1, n, nScale, groupIdx, rankId,
                                                            metaInfoTensor, ubQuantSend, params, localSrcPtr);
        } else {
            CombineQuantizedTokens<SendType>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor, ubQuantSend,
                                             params, quantTokenSizeBytes);
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
}

} // namespace CombineImpl

constexpr uint32_t META_INFO_TENSOR_ADDR = 200U * 1024U;
constexpr int32_t MAX_AICORE_NUM = 36;

using Gmm2ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using Gmm2BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t>;

struct Gmm2ExpertLoopState {
    Gmm2ProblemShape problemShape;
    Gmm2BlockOffset baseOffset;
    uint32_t expertBeforeCnt = 0;
};

struct Gmm2Config {
    MoeStageCommonConfig common;
    BlockJobContext blockJob;
    BlockWorkspaceContext countWorkspace;
    int32_t activationFlagSlotsPerExpert;
    int32_t dispatchFlagSlotsPerExpert;
    uint64_t combineSyncSlotCountPerExpert;
    int32_t groupedMatmulMode;
    bool isPerExpertWeightTensor;
};

struct Gmm2Scratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
};

struct Gmm2RuntimeState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    int32_t &gmTileSequence;
    uint16_t &pingpongIdx;
};

struct QuantCombineBufferConfig {
    int64_t combineUbElementCount;
    uint32_t quantTokenSizeBytes;
};

struct QuantCombineConfig {
    MoeStageCommonConfig common;
    AivJobContext job;
    bool participates;
};

struct QuantCombineTokenRange {
    uint32_t tokenStart;
    uint32_t tokenCount;
    uint64_t metaInfoTokenOffset;
    uint32_t expertIdx;
};

// 加载当前任务负责的 metadata，并完成一段连续 token 的量化 Combine 发送。
template <uint8_t CombineMode>
__aicore__ inline void CombineQuantizedTokenRange(
    const QuantCombineConfig &context, const QuantCombineBufferConfig &bufferConfig,
    const Params &params, const GMMAddrInfo &gmmAddrInfo, const QuantCombineTokenRange &tokenRange)
{
    AscendC::SetCtrlSpr<60, 60>(0);
    int64_t ubOffset = 0;
    LocalTensor<int32_t> metaInfoTensor(TPosition::VECIN, ubOffset,
                                        tokenRange.tokenCount * META_INFO_SIZE);
    ubOffset += tokenRange.tokenCount * META_INFO_SIZE * sizeof(int32_t);
    GlobalTensor<int32_t> metaInfoGm;
    metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.metaInfoPtr + tokenRange.metaInfoTokenOffset * META_INFO_SIZE * sizeof(int32_t)));
    DataCopy(metaInfoTensor, metaInfoGm, tokenRange.tokenCount * META_INFO_SIZE);
    PipeBarrier<PIPE_MTE2>();
    CombineImpl::CombineTokenGroup<CombineMode, bfloat16_t>(
        tokenRange.tokenStart, tokenRange.tokenCount, context.common.tokenHiddenDim, tokenRange.expertIdx,
        context.common.rankId, gmmAddrInfo.gmm2OutGlobal, params, metaInfoTensor,
        bufferConfig.combineUbElementCount, ubOffset, bufferConfig.quantTokenSizeBytes);
}

struct WaveCombineConfig {
    MoeStageCommonConfig common;
    AivJobContext job;
};

struct WaveCombineBufferConfig {
    uint32_t rowBytes = 0;
    uint32_t rowStrideBytes = 0;
    uint32_t quantRowElements = 0;
    uint32_t quantRowStorageBytes = 0;
    uint32_t slotStrideBytes = 0;
    uint32_t quantTempElements = 0;
};

struct WaveCombineScratch {
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<bfloat16_t> rowBufferTensor;
    LocalTensor<float> quantTempTensor;
};

constexpr uint32_t WAVE_COMBINE_NO_QUANT_ROW_BUFFER_NUM = 6U;
constexpr uint32_t WAVE_COMBINE_QUANT_ROW_BUFFER_NUM = 2U;
constexpr uint32_t WAVE_COMBINE_UB_BASE = 64U * 1024U;
constexpr uint32_t WAVE_COMBINE_META_INFO_TOKEN_CAPACITY = 1536U;

// 构造 Combine 与 Unpermute 共用的量化 token 记录布局。
__aicore__ inline QuantCombineBufferConfig CreateQuantCombineBufferConfig(uint32_t tokenHiddenDim)
{
    uint32_t nAlign32 = Ops::Base::CeilAlign(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
    uint32_t nScale = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
    uint32_t tokenStorageBytes = Ops::Base::CeilAlign(tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
    uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
    uint32_t quantTokenSizeBytes =
        Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
    uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
    return {static_cast<int64_t>(singleTokenBytes * DOUBLE_BUFFER / sizeof(bfloat16_t)), quantTokenSizeBytes};
}

// 返回一个逻辑 wave Combine 任务负责的连续 token 范围。
__aicore__ inline WorkRange GetWaveCombineOwnedRange(const WaveCombineConfig &context, uint32_t tokenCount)
{
    if (GetSubBlockIdx() != 1U) {
        return {};
    }
    return GetBalancedTokenRange(tokenCount, context.job.jobIndex, context.job.totalJobs);
}

// 回收当前 wave 的 Combine 发送在行环形 buffer 上产生的事件。
template <uint8_t CombineMode>
__aicore__ inline void DrainWaveCombineRowRing(uint32_t issuedRowCount)
{
    constexpr uint32_t rowBufferCount =
        CombineMode == COMBINE_NO_QUANT ? WAVE_COMBINE_NO_QUANT_ROW_BUFFER_NUM :
                                          WAVE_COMBINE_QUANT_ROW_BUFFER_NUM;
    uint32_t activeSlotCount = issuedRowCount < rowBufferCount ? issuedRowCount : rowBufferCount;
    if (activeSlotCount > 0U) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if (activeSlotCount > 1U) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        if (activeSlotCount > 2U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        if (activeSlotCount > 3U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
        if (activeSlotCount > 4U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        }
        if (activeSlotCount > 5U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        }
    }
}

// 加载一段连续 wave Combine token 的 metadata。
__aicore__ inline void PreloadWaveCombineMetaInfo(const Params &params, WaveCombineScratch &scratch,
                                                  uint64_t gmTokenOffset, uint32_t tokenCount,
                                                  uint32_t ubTokenOffset)
{
    if (tokenCount == 0U) {
        return;
    }
    LocalTensor<int32_t> metaInfoUb = scratch.metaInfoTensor[ubTokenOffset * META_INFO_SIZE];
    GlobalTensor<int32_t> metaInfoGm;
    metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr));
    DataCopy(metaInfoUb, metaInfoGm[gmTokenOffset * META_INFO_SIZE], tokenCount * META_INFO_SIZE);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

// 搬运并按需量化一个 wave Combine token 后发送；函数内部不包含 GMM2-ready 依赖。
template <uint8_t CombineMode, bool IsBufferReuse>
__aicore__ inline void SendWaveCombineToken(const WaveCombineConfig &context,
                                            const WaveCombineBufferConfig &bufferConfig,
                                            WaveCombineScratch &scratch, const Params &params,
                                            GlobalTensor<bfloat16_t> &gmm2OutGm, uint64_t gmRemoteBaseOffset,
                                            uint32_t tokenLocal, LocalTensor<int32_t> &tokenMetaInfo, uint32_t slot)
{
    TEventID eventId =
        static_cast<TEventID>(static_cast<int32_t>(EVENT_ID0) + static_cast<int32_t>(slot));
    if constexpr (IsBufferReuse) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    uint32_t slotElementOffset = slot * bufferConfig.slotStrideBytes / sizeof(bfloat16_t);
    LocalTensor<bfloat16_t> rowUb = scratch.rowBufferTensor[slotElementOffset];
    DataCopyExtParams gm2UbParams{1U, bufferConfig.rowBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> gm2UbPad{false, 0U, 0U, 0U};
    DataCopyPad(rowUb, gmm2OutGm[static_cast<uint64_t>(tokenLocal) * context.common.tokenHiddenDim],
                gm2UbParams, gm2UbPad);
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CombineImpl::SendCombineTokenRow<bfloat16_t>(
            context.common.tokenHiddenDim, gmRemoteBaseOffset, tokenMetaInfo, rowUb, params);
    } else {
        LocalTensor<bfloat16_t> quantUb =
            scratch.rowBufferTensor[slotElementOffset + bufferConfig.rowStrideBytes / sizeof(bfloat16_t)];
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Mxfp8::QuantMxFp8<CombineMode, bfloat16_t>(
            quantUb, rowUb, scratch.quantTempTensor, context.common.tokenHiddenDim);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT,
                                                  fp8_e4m3fn_t, fp8_e5m2_t>::type;
        LocalTensor<Fp8Type> quantSendUb = quantUb.template ReinterpretCast<Fp8Type>();
        CombineImpl::SendCombineTokenRow<Fp8Type>(
            bufferConfig.quantRowElements, gmRemoteBaseOffset, tokenMetaInfo, quantSendUb, params);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

// 原型：MegaMoeWave::ProcessCombineGm。发送一个逻辑 AIV 任务负责的已就绪 token 范围。
template <uint8_t CombineMode>
__aicore__ inline void CombineWaveTokenRange(const WaveCombineConfig &context,
                                             const WaveCombineBufferConfig &bufferConfig,
                                             WaveCombineScratch &scratch, const Params &params,
                                             GM_ADDR gmm2OutGlobal, uint32_t tokenStart, uint32_t tokenCount,
                                             uint32_t metaInfoUbTokenOffset, uint32_t &rowSequence)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || tokenCount == 0U) {
        return;
    }
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        AscendC::SetCtrlSpr<60, 60>(0);
    }
    GlobalTensor<bfloat16_t> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gmm2OutGlobal));
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    constexpr uint32_t rowBufferCount =
        CombineMode == COMBINE_NO_QUANT ? WAVE_COMBINE_NO_QUANT_ROW_BUFFER_NUM :
                                          WAVE_COMBINE_QUANT_ROW_BUFFER_NUM;
    uint32_t tokenIdxInSlice = 0U;
    uint32_t firstUseCount = rowSequence < rowBufferCount ? rowBufferCount - rowSequence : 0U;
    firstUseCount = firstUseCount < tokenCount ? firstUseCount : tokenCount;
    for (; tokenIdxInSlice < firstUseCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        LocalTensor<int32_t> tokenMetaInfo =
            scratch.metaInfoTensor[(metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE];
        SendWaveCombineToken<CombineMode, false>(context, bufferConfig, scratch, params, gmm2OutGm,
                                                 gmRemoteBaseOffset, tokenStart + tokenIdxInSlice,
                                                 tokenMetaInfo, slot);
    }
    for (; tokenIdxInSlice < tokenCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        LocalTensor<int32_t> tokenMetaInfo =
            scratch.metaInfoTensor[(metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE];
        SendWaveCombineToken<CombineMode, true>(context, bufferConfig, scratch, params, gmm2OutGm,
                                                gmRemoteBaseOffset, tokenStart + tokenIdxInSlice,
                                                tokenMetaInfo, slot);
    }
}

// 等待当前 GMM2 tile 对应的 SwiGLU 输入就绪。
template <bool IsShared, typename Config>
__aicore__ inline void WaitForGmm2InputReady(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
        uint32_t waveIdx = mLoc / L1_TILE_M_256;
        uint32_t waveStart = waveIdx * L1_TILE_M_256;
        uint32_t waveM = waveStart + L1_TILE_M_256 > config.m ? config.m - waveStart : L1_TILE_M_256;
        constexpr uint32_t sourceNFactor = Config::SOURCE_GMM1_INTERLEAVED ? ACTIVATION_N_HALF : 1U;
        uint32_t targetLoops = Ops::Base::CeilDiv(waveM, config.activationTileM) *
                               Ops::Base::CeilDiv(config.k * sourceNFactor, L1_TILE_N);
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t *flagValueAddr = gmmAddrInfo.activationToGmm2Flag + flagOffset;
        while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    } else {
        GmmKernel::BlockScheduler gmmBlockScheduler(
            {config.m, config.k, config.n},
            GmmKernel::BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.activationTileM),
                                                            static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        while (targetLoops != AscendC::ReadGmByPassDCache(gmmAddrInfo.activationToGmm2Flag)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    }
}

// Notifies all AIV consumers assigned to the completed GMM2 token group.
__aicore__ inline void NotifyCombineConsumersOfTileCompletion(uint32_t rowTileOffset,
                                                              const GroupSyncSlotLayout &slotLayout,
                                                              __gm__ int32_t *expertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    uint32_t firstSyncSlot = 0;
    uint32_t syncSlotCount = 0;
    GetGroupSyncSlotRange(tokenGroupIndex, slotLayout, firstSyncSlot, syncSlotCount);
    for (uint32_t syncSlot = firstSyncSlot; syncSlot < firstSyncSlot + syncSlotCount; ++syncSlot) {
        AscendC::AtomicAdd(GetCombineSyncCounterAddress(expertCounterBase, syncSlot), int32_t(1));
    }
}

// Notifies the AIV consumer assigned to the completed shared-expert token group.
__aicore__ inline void NotifySharedExpertTileCompletion(uint32_t rowTileOffset,
                                                        __gm__ int32_t *sharedExpertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    AscendC::AtomicAdd(GetCombineSyncCounterAddress(sharedExpertCounterBase, tokenGroupIndex), int32_t(1));
}

namespace GmmKernel {

// 执行通用 GMM2 tile 循环。
template <uint8_t CombineQuantMode, typename BlockMmad, bool IsShared, bool IsLayered = false,
          typename WorkSet, typename Config>
__aicore__ inline void Gmm2AicMmadGeneric(BlockMmad &blockMmad, WorkSet &workSet,
                                          const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                          uint32_t startLoopIdx, uint32_t tileNum)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum;
        if constexpr (!std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
            logicalCoreCount *= 2U;
        }
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        // Slice only builds tensor views. Keep it ahead of the synchronization waits so the
        // current tile's address calculation is not inserted into the compute critical path.
        auto gmBlockA = workSet.gmA.Slice(Te::MakeCoord(mLoc, kLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA = workSet.gmScaleA.Slice(
            Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
            Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

        if constexpr (std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB = workSet.gmScaleB.Slice(
            Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
            Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
        auto gmBlockC = workSet.gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
        if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout, gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared && !IsLayered) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
}

// 执行 A8W4 GMM2 tile 循环。
template <uint8_t CombineQuantMode, bool IsShared, bool IsLayered = false, typename BlockMmad,
          typename Scheduler, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC,
          typename Config>
__aicore__ inline void Gmm2AicMmadA8W4(BlockMmad &blockMmad, Scheduler &scheduler, TensorA &gmA,
                                       TensorScaleA &gmScaleA, TensorScaleB &gmScaleB, TensorC &l0cOutGm,
                                       const GMMAddrInfo &gmmAddrInfo, int32_t &gmTileSequence,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum;
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
        }

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
        if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout, gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
        constexpr bool hasAiv1GmEpilogue = CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered;
        if constexpr (hasAiv1GmEpilogue) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++gmTileSequence);
        }
    }
}

// AIV1 消费 AIC 写入 GM 的 GMM2 tile，并执行后处理。
template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void Gmm2Aiv1EpilogueA8W4(
    Scheduler &scheduler, TensorC &l0cOutGm, const Params &params, const GMMAddrInfo &gmmAddrInfo,
    int32_t &gmTileSequence, const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        int32_t expectedReadySequence = gmTileSequence + 1;
        while (AscendC::ReadGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag) < expectedReadySequence) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < 100) {
            }
        }
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto layoutL0cUB = MakeLayoutC{}(config.tileM, L1_TILE_N);
        int64_t ubOffset = 0;
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutL0cUB);
        LocalTensor<ElementC> l0cOutUbGMM2 =
            LocalTensor<ElementC>(TPosition::VECIN, ubOffset, config.tileM * L1_TILE_N);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        AscendC::GlobalTensor<int32_t> metaInfoGm;
        int32_t lenTile = Get<M_VALUE>(actualShape);
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR, lenTile * META_INFO_SIZE);
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            gmmAddrInfo.metaInfoGlobal + static_cast<uint64_t>(mLoc) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, lenTile * META_INFO_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        CombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
            nLoc, config.n, metaInfoTensor, l0cOutUbGMM2, actualShape, L1_TILE_N, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        gmTileSequence = expectedReadySequence;
    }
}

// 为一个 GMM2 逻辑 block 的 AIV0 路径展开一段 A8W4 权重。
template <typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void Gmm2Aiv0PrologueA8W4(BlockPrologue &blockPrologue, Scheduler &scheduler, TensorB &gmB,
                                            const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);
        blockPrologue(gmB, mL1Size, config.k, nL1Size, nLoc, config.n, config.l1Params.kL1);
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM2 阶段。
template <uint8_t CombineQuantMode, typename BlockMmad, typename ElementC, bool IsLayered = false,
          bool IsShared = false, typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecGeneric(
    Scheduler &scheduler, const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t startLoopIdx,
    uint32_t tileNum, PersistentBlockMmadContext<BlockMmad> *persistentContext,
    bool allowWeightL2Bypass)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    if constexpr (Config::IS_WAVE_FLAG_GRAINED && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(
            config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                  reinterpret_cast<__gm__ ElementC *>(gmmAddrInfo.gmm2OutGlobal)),
                              layouts.c);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        if (persistentContext != nullptr) {
            InitBlockMmad(*persistentContext, config);
            Gmm2AicMmadGeneric<CombineQuantMode, BlockMmad, IsShared, IsLayered>(
                persistentContext->blockMmad, workSet, gmmAddrInfo, config, startLoopIdx, tileNum);
        } else {
            PersistentBlockMmadContext<BlockMmad> localContext;
            InitBlockMmad(localContext, config);
            Gmm2AicMmadGeneric<CombineQuantMode, BlockMmad, IsShared, IsLayered>(
                localContext.blockMmad, workSet, gmmAddrInfo, config, startLoopIdx, tileNum);
        }
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM2 阶段。
template <uint8_t CombineQuantMode, typename BlockMmad, typename BlockPrologue, typename ElementC,
          typename MakeLayoutC, bool IsShared, bool IsLayered = false, typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecA8W4(
    Scheduler &scheduler, const Params &params, const GMMAddrInfo &gmmAddrInfo, const Config &config,
    uint32_t startLoopIdx, uint32_t tileNum, int32_t &gmTileSequence)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                  reinterpret_cast<__gm__ ElementC *>(gmmAddrInfo.gmm2OutGlobal)),
                              layouts.c);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        Gmm2AicMmadA8W4<CombineQuantMode, IsShared, IsLayered, BlockMmad, decltype(workSet.scheduler),
                        decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                        std::remove_reference_t<decltype(workSet.gmC)>, Config>(
            blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC,
            gmmAddrInfo, gmTileSequence, config, startLoopIdx, tileNum);
    } else {
        if (GetSubBlockIdx() == 0) {
            BlockPrologue blockPrologue;
            Gmm2Aiv0PrologueA8W4(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        } else {
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
                Gmm2Aiv1EpilogueA8W4<ElementC, MakeLayoutC>(workSet.scheduler, workSet.gmC, params, gmmAddrInfo,
                                                            gmTileSequence, config, startLoopIdx, tileNum);
            }
        }
    }
}

} // namespace GmmKernel

// RunGmm2Generic：执行 Generic GMM2，支持量化和非量化 Combine 模式。
// =================================================================================================
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm2Generic(
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, const BlockJobContext &blockJob,
    void *persistentBlockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    using GmmConfig = GmmKernel::Config<false, CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA,
                                        ElementMxScaleB, IsWeightNZ, TopkWeightsPrefetch, IsShared,
                                        IsLayered, IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

    GmmKernel::BlockScheduler scheduler(
        {config.m, config.schedulerN, config.k},
        GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT && g_coreType == AscendC::AIV) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using PersistentContext = GmmKernel::PersistentBlockMmadContext<BlockMmad>;
    auto *persistentContext = reinterpret_cast<PersistentContext *>(persistentBlockMmadContext);
    GmmKernel::Gmm2ExecGeneric<CombineQuantMode, BlockMmad, ElementC, IsLayered, IsShared>(
        scheduler, gmmAddrInfo, config, startLoopIdx, tileNum, persistentContext,
        allowWeightL2Bypass);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm2Generic(
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, void *persistentBlockMmadContext = nullptr,
    bool allowWeightL2Bypass = false)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm2Generic<CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                   IsWeightNZ, IsLayered, Gmm1TileM, TopkWeightsPrefetch, IsShared,
                   IsGmm1Interleaved, IsWaveFlagGrained>(
        problemShape, gmmAddrInfo, startBlockIdx, blockJob, persistentBlockMmadContext,
        allowWeightL2Bypass);
}

// RunGmm2A8W4：执行 A8W4 prologue（W4→W8）、GMM2 和 Combine。
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false>
__aicore__ inline void RunGmm2A8W4(
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence,
    const BlockJobContext &blockJob)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = GmmKernel::Config<true, 0, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                                        false, TopkWeightsPrefetch, IsShared, IsLayered>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;

    GmmKernel::BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    GmmKernel::Gmm2ExecA8W4<CombineQuantMode, BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared, IsLayered,
                            GmmKernel::BlockScheduler, decltype(config)>(
        scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum, gmTileSequence);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false>
__aicore__ inline void RunGmm2A8W4(
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &gmTileSequence)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm2A8W4<CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM,
                TopkWeightsPrefetch, IsShared, IsLayered>(
        params, problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, blockJob);
}

__aicore__ inline Gmm2ExpertLoopState CreateGmm2ExpertLoopState(const Gmm2Config &context)
{
    Gmm2ProblemShape shape;
    Get<N_VALUE>(shape) = context.common.gmm1OutputDim;
    Get<K_VALUE>(shape) = context.common.tokenHiddenDim;
    Gmm2BlockOffset offset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return {shape, offset, 0};
}

/*
 * 动态 Wave 使用 rowOffset 索引紧凑 token buffer，使用 expertIdx 索引定长资源。Dispatch 与 GMM
 * 以不同 Wave 进度推进时，无需继续累加旧的 Gmm2BlockOffset。
 */
template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          bool ConfigureCombineCounter>
__aicore__ inline void UpdateA8W4WaveGmm2GlobalBuffer(
    const Gmm2Config &context, const WorkspaceInfo &workspace, const ExpertWeightTensorListAddrs &weights,
    GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx)
{
    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t activationOutputElementsPerByte =
        PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t gmm1OutputDim = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape));
    uint64_t tokenHiddenDim = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    uint64_t rowOffset = static_cast<uint64_t>(state.rowOffset);
    uint64_t activationOutputWidth = gmm1OutputDim / ACTIVATION_N_HALF;
    uint64_t activationScaleWidth =
        Ops::Base::CeilDiv(activationOutputWidth, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    gmmAddrInfo.gmm2OutGlobal = workspace.gmm2MmadResPtr + rowOffset * tokenHiddenDim * sizeof(bfloat16_t);
    gmmAddrInfo.metaInfoGlobal = workspace.metaInfoPtr + rowOffset * META_INFO_SIZE * sizeof(int32_t);
    gmmAddrInfo.aGlobal =
        workspace.activationQuantDataPtr +
        rowOffset * activationOutputWidth / activationOutputElementsPerByte * sizeof(ActivationOutType);
    gmmAddrInfo.aScaleGlobal =
        workspace.activationQuantScalePtr + rowOffset * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        weights.weight2, context.isPerExpertWeightTensor, expertIdx,
        static_cast<uint64_t>(expertIdx) * tokenHiddenDim * activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, context.isPerExpertWeightTensor, expertIdx,
        static_cast<uint64_t>(expertIdx) * tokenHiddenDim * activationScaleWidth);
    if constexpr (ConfigureCombineCounter) {
        uint64_t syncSlotOffset = static_cast<uint64_t>(expertIdx) * context.combineSyncSlotCountPerExpert;
        gmmAddrInfo.gmm2CombineSyncCounter =
            reinterpret_cast<__gm__ int32_t *>(workspace.gmm2CombineSyncCounterPtr) +
            syncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
    }
    gmmAddrInfo.activationToGmm2Flag =
        reinterpret_cast<__gm__ int32_t *>(workspace.flagActivationToGmm2Ptr) +
        static_cast<uint64_t>(expertIdx) * context.activationFlagSlotsPerExpert;
    gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagDispatchToGmm1Ptr) +
                                     static_cast<uint64_t>(expertIdx) * context.dispatchFlagSlotsPerExpert;
    gmmAddrInfo.gmmToEpilogueFlag = nullptr;
    if (workspace.flagGmmToEpiloguePtr != nullptr) {
        gmmAddrInfo.gmmToEpilogueFlag =
            reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
            static_cast<uint64_t>(context.blockJob.jobIndex) * INT_CACHELINE;
    }
}

template <uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t ActivationOutputElementsPerByte>
__aicore__ inline void AdvanceGmm2ExpertOffsets(Gmm2ExpertLoopState &state)
{
    uint64_t m = Get<M_VALUE>(state.problemShape);
    uint64_t n = Get<N_VALUE>(state.problemShape);
    uint64_t k = Get<K_VALUE>(state.problemShape);
    state.expertBeforeCnt += m;
    Get<IDX_A_OFFSET>(state.baseOffset) += m * k / ActivationElementsPerByte;
    Get<IDX_B_OFFSET>(state.baseOffset) += n * k / WeightElementsPerByte;
    uint64_t scaleK =
        Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
    Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
    Get<IDX_C_OFFSET>(state.baseOffset) += m * n / ACTIVATION_N_HALF / ActivationOutputElementsPerByte;
    Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
        m * Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
    Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / ACTIVATION_N_HALF / WeightElementsPerByte;
    Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
        k * Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
    Get<IDX_M_OFFSET>(state.baseOffset) += m;
    Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
    Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
}

// GMM2 沿用 GMM1 已确认就绪的专家 token 总数，并推进到当前 MoE 专家的完整状态。
template <uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t ActivationOutputElementsPerByte>
__aicore__ inline bool PrepareMoeExpertGmm2State(const Gmm2Config &context, Gmm2Scratch &scratch,
                                                 Gmm2ExpertLoopState &state, uint32_t expertIdx)
{
    if (expertIdx != 0U) {
        AdvanceGmm2ExpertOffsets<ActivationElementsPerByte, WeightElementsPerByte,
                                 ActivationOutputElementsPerByte>(state);
    }
    uint64_t countOffset = static_cast<uint64_t>(expertIdx) * INT32_PER_256B * context.countWorkspace.blockNum +
                           static_cast<uint64_t>(INT32_PER_256B) * context.countWorkspace.blockIdx;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        scratch.expertRevNumsGlobalTensor[countOffset]);
    Get<M_VALUE>(state.problemShape) = scratch.expertRevNumsGlobalTensor.GetValue(countOffset);
    return Get<M_VALUE>(state.problemShape) != 0;
}

// 使用固定 token 数推进到当前共享专家的完整 GMM2 状态。
template <uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t ActivationOutputElementsPerByte>
__aicore__ inline bool PrepareSharedExpertGmm2State(const Gmm2Config &context,
                                                    Gmm2ExpertLoopState &state, uint32_t expertIdx)
{
    if (expertIdx != 0U) {
        AdvanceGmm2ExpertOffsets<ActivationElementsPerByte, WeightElementsPerByte,
                                 ActivationOutputElementsPerByte>(state);
    }
    Get<M_VALUE>(state.problemShape) = context.common.tokenNum;
    return Get<M_VALUE>(state.problemShape) != 0;
}

template <typename ActivationType, typename ActivationOutType, typename QuantScaleType,
          bool ConfigureGmm2Output, bool ConfigureCombineCounter, bool ConfigureGmmToEpilogue>
__aicore__ inline void UpdateMoeExpertGmm2GlobalBuffer(
    const Gmm2Config &context, const WorkspaceInfo &workspace,
    const ExpertWeightTensorListAddrs &weights, GMMAddrInfo &gmmAddrInfo,
    const Gmm2ExpertLoopState &state, uint32_t expertIdx,
    uint32_t expertMGroupOffset = 0U)
{
    if constexpr (ConfigureGmm2Output) {
        gmmAddrInfo.gmm2OutGlobal =
            workspace.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
    }
    gmmAddrInfo.metaInfoGlobal =
        workspace.metaInfoPtr + static_cast<uint64_t>(state.expertBeforeCnt) * META_INFO_SIZE * sizeof(int32_t);
    gmmAddrInfo.aGlobal =
        workspace.activationQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationOutType);
    gmmAddrInfo.aScaleGlobal =
        workspace.activationQuantScalePtr + Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        weights.weight2, context.isPerExpertWeightTensor, expertIdx,
        Get<IDX_B2_OFFSET>(state.baseOffset));
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, context.isPerExpertWeightTensor, expertIdx,
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
    if constexpr (ConfigureCombineCounter) {
        uint64_t expertSyncSlotOffset =
            static_cast<uint64_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset)) * context.combineSyncSlotCountPerExpert;
        gmmAddrInfo.gmm2CombineSyncCounter =
            reinterpret_cast<__gm__ int32_t *>(workspace.gmm2CombineSyncCounterPtr) +
            expertSyncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
    }
    gmmAddrInfo.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagActivationToGmm2Ptr) +
                                       Get<IDX_FLAG_OFFSET>(state.baseOffset) * context.activationFlagSlotsPerExpert +
                                       static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagDispatchToGmm1Ptr) +
                                     Get<IDX_FLAG_OFFSET>(state.baseOffset) * context.dispatchFlagSlotsPerExpert +
                                     static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    if constexpr (ConfigureGmmToEpilogue) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag =
                reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
                static_cast<uint64_t>(context.blockJob.jobIndex) * INT_CACHELINE;
        }
    }
}

template <typename ActivationType, typename QuantScaleType, uint32_t Gmm1TileM,
          bool ConfigureGmmToEpilogue>
__aicore__ inline void UpdateSharedExpertGmm2GlobalBuffer(
    const Gmm2Config &context, const WorkspaceInfo &workspace,
    const ExpertWeightTensorListAddrs &weights, GMMAddrInfo &gmmAddrInfo,
    const Gmm2ExpertLoopState &state, uint32_t sharedExpertIdx)
{
    gmmAddrInfo.gmm2OutGlobal =
        workspace.sharedExpertResultPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
    gmmAddrInfo.aGlobal =
        workspace.sharedExpertActivationDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal =
        workspace.sharedExpertActivationScalePtr + Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        weights.weight2, context.isPerExpertWeightTensor, sharedExpertIdx,
        Get<IDX_B2_OFFSET>(state.baseOffset));
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, context.isPerExpertWeightTensor, sharedExpertIdx,
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(context.common.tokenNum, Gmm1TileM);
    gmmAddrInfo.sharedExpertGmm2TileCounter =
        reinterpret_cast<__gm__ int32_t *>(workspace.sharedExpertGmm2TileCounterPtr) +
        static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
    if constexpr (ConfigureGmmToEpilogue) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag =
                reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
                static_cast<uint64_t>(context.blockJob.jobIndex) * INT_CACHELINE;
        }
    }
}

// 供普通模板、Wave 模板和共享专家的 GMM2 阶段复用。
// 使用调用方传入的 block 任务执行 GMM2，并保持所选同步约定不变。
template <uint8_t CombineMode, typename GenericElementA, typename A8W4ElementA, typename WeightType,
          typename QuantScaleType, bool EnableA8W4, bool EnableA4W4, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch, bool IsShared, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm2ByMode(const Gmm2Config &context, const Params &params,
                                     const GMMAddrInfo &gmmAddrInfo, const Gmm2ExpertLoopState &state,
                                     Gmm2RuntimeState &runtimeState, void *persistentBlockMmadContext = nullptr,
                                     bool allowWeightL2Bypass = false)
{
    if constexpr (EnableA8W4 || EnableA4W4) {
        RunGmm2A8W4<CombineMode, A8W4ElementA, WeightType, bfloat16_t, QuantScaleType,
                    QuantScaleType, Gmm1TileM, TopkWeightsPrefetch, IsShared>(
            params, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.gmTileSequence,
            context.blockJob);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        RunGmm2Generic<CombineMode, GenericElementA, GenericElementA, bfloat16_t, QuantScaleType,
                       QuantScaleType, true, false, Gmm1TileM, TopkWeightsPrefetch, IsShared,
                       IsGmm1Interleaved, IsWaveFlagGrained>(
            state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx, context.blockJob,
            persistentBlockMmadContext, allowWeightL2Bypass);
    } else {
        RunGmm2Generic<CombineMode, GenericElementA, GenericElementA, bfloat16_t, QuantScaleType,
                       QuantScaleType, false, false, Gmm1TileM, TopkWeightsPrefetch, IsShared,
                       IsGmm1Interleaved, IsWaveFlagGrained>(
            state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx, context.blockJob,
            persistentBlockMmadContext, allowWeightL2Bypass);
    }
}

template <typename ActivationType, typename ActivationOutType, typename QuantScaleType,
          uint32_t ActivationElementsPerByte, uint32_t WeightElementsPerByte,
          uint32_t ActivationOutputElementsPerByte, bool ConfigureGmm2Output,
          bool ConfigureCombineCounter, bool ConfigureGmmToEpilogue>
__aicore__ inline bool PrepareMoeExpertGmm2Stage(
    const Gmm2Config &context, const WorkspaceInfo &workspace,
    const ExpertWeightTensorListAddrs &weights, Gmm2Scratch &scratch,
    Gmm2ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx)
{
    if (!PrepareMoeExpertGmm2State<ActivationElementsPerByte, WeightElementsPerByte,
                                   ActivationOutputElementsPerByte>(context, scratch, state, expertIdx)) {
        return false;
    }
    UpdateMoeExpertGmm2GlobalBuffer<ActivationType, ActivationOutType, QuantScaleType,
                                    ConfigureGmm2Output, ConfigureCombineCounter,
                                    ConfigureGmmToEpilogue>(
        context, workspace, weights, gmmAddrInfo, state, expertIdx);
    return true;
}

template <uint8_t CombineMode, typename GenericElementA, typename A8W4ElementA,
          typename WeightType, typename QuantScaleType, bool EnableA8W4,
          bool EnableA4W4, uint32_t Gmm1TileM, bool TopkWeightsPrefetch,
          bool IsGmm1Interleaved, bool IsWaveFlagGrained>
__aicore__ inline void RunMoeExpertGmm2Stage(
    const Gmm2Config &context, const Params &params, const GMMAddrInfo &gmmAddrInfo,
    const Gmm2ExpertLoopState &state, Gmm2RuntimeState &runtimeState,
    void *persistentBlockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    RunGmm2ByMode<CombineMode, GenericElementA, A8W4ElementA, WeightType,
                  QuantScaleType, EnableA8W4, EnableA4W4, Gmm1TileM,
                  TopkWeightsPrefetch, false, IsGmm1Interleaved,
                  IsWaveFlagGrained>(
        context, params, gmmAddrInfo, state, runtimeState, persistentBlockMmadContext,
        allowWeightL2Bypass);
}

template <typename GenericElementA, typename A8W4ElementA, typename WeightType,
          typename QuantScaleType, bool EnableA8W4, bool EnableA4W4,
          uint32_t Gmm1TileM, bool TopkWeightsPrefetch, bool IsGmm1Interleaved,
          bool IsWaveFlagGrained>
__aicore__ inline bool RunSharedExpertGmm2Stage(
    const Gmm2Config &context, const Params &params,
    const ExpertWeightTensorListAddrs &weights, Gmm2ExpertLoopState &state,
    GMMAddrInfo &gmmAddrInfo, Gmm2RuntimeState &runtimeState, uint32_t sharedExpertIdx,
    void *persistentBlockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    if (!PrepareSharedExpertGmm2State<PackedElementTraits<GenericElementA>::ELEMENTS_PER_BYTE,
                                      PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE,
                                      PackedElementTraits<A8W4ElementA>::ELEMENTS_PER_BYTE>(
            context, state, sharedExpertIdx)) {
        return false;
    }
    UpdateSharedExpertGmm2GlobalBuffer<A8W4ElementA, QuantScaleType, Gmm1TileM,
                                       EnableA8W4 || EnableA4W4>(
        context, params.workspaceInfo, weights, gmmAddrInfo, state, sharedExpertIdx);
    RunGmm2ByMode<COMBINE_NO_QUANT, GenericElementA, A8W4ElementA, WeightType,
                  QuantScaleType, EnableA8W4, EnableA4W4, Gmm1TileM,
                  TopkWeightsPrefetch, true, IsGmm1Interleaved,
                  IsWaveFlagGrained>(
        context, params, gmmAddrInfo, state, runtimeState, persistentBlockMmadContext,
        allowWeightL2Bypass);
    return true;
}

// 调度普通模板当前 MoE 专家的量化 Combine，并保持原有 GMM2-ready 等待顺序。
template <uint8_t CombineMode>
__aicore__ inline void ScheduleQuantizedMoeExpertCombine(
    const QuantCombineConfig &context, const QuantCombineBufferConfig &bufferConfig,
    const Params &params, const GMMAddrInfo &gmmAddrInfo, uint32_t expertTokenCount,
    uint32_t expertBeforeCnt, uint32_t expertIdx)
{
    if constexpr (g_coreType == AIC || CombineMode == COMBINE_NO_QUANT) {
        return;
    }
    if (!context.participates || context.job.totalJobs == 0U ||
        context.job.jobIndex >= context.job.totalJobs || expertTokenCount == 0U) {
        return;
    }
    uint32_t groupCount = Ops::Base::CeilDiv(expertTokenCount, COMBINE_TOKEN_GROUP_SIZE);
    uint32_t firstGroup = 0U;
    uint32_t groupStride = 0U;
    uint32_t jobIndexWithinGroup = 0U;
    uint32_t jobsAssignedToGroup = 0U;
    uint32_t gmm2NTilesPerGroup = Ops::Base::CeilDiv(context.common.tokenHiddenDim, L1_TILE_N);
    ComputeCombineGroupsForCore(context.job.jobIndex, groupCount, context.job.totalJobs, firstGroup,
                                groupStride, jobIndexWithinGroup, jobsAssignedToGroup);
    for (uint32_t groupIndex = firstGroup; groupIndex < groupCount; groupIndex += groupStride) {
        uint32_t syncSlotIndex = groupCount <= context.job.totalJobs ? context.job.jobIndex : groupIndex;
        __gm__ int32_t *syncCounterAddress =
            GetCombineSyncCounterAddress(gmmAddrInfo.gmm2CombineSyncCounter, syncSlotIndex);
        while (AscendC::ReadGmByPassDCache(syncCounterAddress) !=
               static_cast<int32_t>(gmm2NTilesPerGroup)) {
            int64_t waitStartCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - waitStartCycle < 100) {
            }
        }

        uint32_t groupTokenStart = groupIndex * COMBINE_TOKEN_GROUP_SIZE;
        uint32_t groupTokenCount =
            COMBINE_TOKEN_GROUP_SIZE < expertTokenCount - groupTokenStart ?
                COMBINE_TOKEN_GROUP_SIZE :
                expertTokenCount - groupTokenStart;
        uint32_t tokensPerJob = Ops::Base::CeilDiv(groupTokenCount, jobsAssignedToGroup);
        uint32_t tokenOffsetWithinGroup = jobIndexWithinGroup * tokensPerJob;
        if (tokenOffsetWithinGroup >= groupTokenCount) {
            continue;
        }
        uint32_t tokenCountForJob = groupTokenCount - tokenOffsetWithinGroup;
        tokenCountForJob = tokenCountForJob < tokensPerJob ? tokenCountForJob : tokensPerJob;

        QuantCombineTokenRange tokenRange{
            groupTokenStart + tokenOffsetWithinGroup,
            tokenCountForJob,
            static_cast<uint64_t>(expertBeforeCnt) + groupTokenStart + tokenOffsetWithinGroup,
            expertIdx};
        CombineQuantizedTokenRange<CombineMode>(context, bufferConfig, params, gmmAddrInfo, tokenRange);
    }
}

constexpr uint32_t WAVE_GMM2_READY_SCAN_UB_ADDR = 184U * 1024U;
constexpr uint32_t WAVE_GMM2_READY_MAX_SCAN_BYTES =
    MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) > ALIGN_512 ?
        MAX_AICORE_NUM *INT_CACHELINE * sizeof(int32_t) :
        ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR =
    WAVE_GMM2_READY_SCAN_UB_ADDR +
    ((WAVE_GMM2_READY_MAX_SCAN_BYTES + ALIGN_512 - 1U) / ALIGN_512) * ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_SUM_UB_ADDR = WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR + ALIGN_512;

__aicore__ inline uint64_t GetWaveGmm2ReadySlotStride(const WaveCombineConfig &context)
{
    return static_cast<uint64_t>(context.job.totalJobs) * INT_CACHELINE;
}

// 每个 AIC 在 GMM2 写入可见后发布一个独占 cache line 的完成标记。
__aicore__ inline void NotifyWaveGmm2Ready(const WaveCombineConfig &context,
                                           const Params &params, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm2ReadyPtr);
        uint64_t slotOffset =
            static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(context);
        AscendC::WriteGmByPassDCache(
            readyBase + slotOffset + static_cast<uint64_t>(context.job.jobIndex) * INT_CACHELINE,
            int32_t(1));
    }
}

// 一个 AIV Combine 任务等待全部 AIC 完成指定专家。
__aicore__ inline void WaitWaveGmm2Ready(const WaveCombineConfig &context,
                                         const Params &params, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || context.job.totalJobs == 0U) {
        return;
    }
    uint32_t readyElements = context.job.totalJobs * static_cast<uint32_t>(INT_CACHELINE);
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm2ReadyPtr);
    uint64_t readySlotOffset =
        static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(context);
    GlobalTensor<int32_t> readyGm;
    readyGm.SetGlobalBuffer(readyBase);
    LocalTensor<int32_t> scanUb(TPosition::VECCALC, WAVE_GMM2_READY_SCAN_UB_ADDR, readyElements);
    LocalTensor<uint8_t> reduceTmpUb(
        TPosition::VECCALC, WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR, ALIGN_512);
    LocalTensor<int32_t> sumUb(
        TPosition::VECCALC, WAVE_GMM2_READY_SUM_UB_ADDR, INT_CACHELINE);
    const uint32_t readyShape[] = {1U, readyElements};
    while (true) {
        DataCopy(scanUb, readyGm[readySlotOffset], readyElements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ReduceSum<int32_t, AscendC::Pattern::Reduce::AR, true>(
            sumUb, scanUb, reduceTmpUb, readyShape, true);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (sumUb.GetValue(0) >= static_cast<int32_t>(context.job.totalJobs)) {
            return;
        }
        int64_t waitStart = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - waitStart < 100) {
        }
    }
}

template <uint8_t CombineMode>
__aicore__ inline void RunWaveExpertCombineStage(
    const WaveCombineConfig &context, const WaveCombineBufferConfig &bufferConfig,
    WaveCombineScratch &scratch, const Params &params, const GMMAddrInfo &gmmAddrInfo,
    const Gmm2ExpertLoopState &state, uint32_t expertIdx, uint32_t &rowSequence)
{
    uint32_t currentExpertTokenNum = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    WorkRange currentCoreTokenRange = GetWaveCombineOwnedRange(context, currentExpertTokenNum);
    if (currentCoreTokenRange.count != 0U) {
        WaitWaveGmm2Ready(context, params, expertIdx);
    }
    for (uint32_t processedTokenNum = 0U; processedTokenNum < currentCoreTokenRange.count;) {
        uint32_t remainingTokenNum = currentCoreTokenRange.count - processedTokenNum;
        uint32_t chunkTokenNum =
            remainingTokenNum < WAVE_COMBINE_META_INFO_TOKEN_CAPACITY ?
                remainingTokenNum :
                WAVE_COMBINE_META_INFO_TOKEN_CAPACITY;
        PreloadWaveCombineMetaInfo(
            params, scratch,
            static_cast<uint64_t>(state.expertBeforeCnt) + currentCoreTokenRange.start + processedTokenNum,
            chunkTokenNum, 0U);
        CombineWaveTokenRange<CombineMode>(
            context, bufferConfig, scratch, params, gmmAddrInfo.gmm2OutGlobal,
            currentCoreTokenRange.start + processedTokenNum, chunkTokenNum, 0U, rowSequence);
        processedTokenNum += chunkTokenNum;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_H
