/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_COMBINE_SEND_H
#define MEGA_MOE_LAYERED_COMBINE_SEND_H

#include <type_traits>

#include "../common/mega_moe_utils.h"
#include "../common/mega_moe_copy.h"
#include "../common/mega_moe_sync.h"
#include "../common/mega_moe_mxfp8_utils.h"

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
namespace MegaMoeImpl::CombineImpl {

template <typename CommPolicy>
struct LayeredCombineBatchState {
    // 实际接收卡：普通模式为 token 原始输入卡，Server 去重模式为本 Server 中继卡。
    uint32_t dstRankId = 0U;
    typename CommPolicy::BatchHandle batchHandle{};
    uint32_t pendingTokenCount = 0U;
};

// 通过本地 MTE 路径或远端 URMA 路径发送一个 layered 模板 token。
template <typename DataType, typename CommPolicy>
__aicore__ inline void CombineSendTokenToRemote(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                                uint32_t groupIdx, uint32_t rankId, __ubuf__ int32_t* metaInfoTensor,
                                                __ubuf__ DataType* ubQuant, const Params& params, GM_ADDR localSrcPtr,
                                                CommPolicy& comm, typename CommPolicy::BatchHandle& batchHandle)
{
    constexpr bool IsQuantized = std::is_same_v<DataType, fp8_e4m3fn_t> || std::is_same_v<DataType, fp8_e5m2_t>;
    const auto& peermem = params.communication.layout;
    // metadata 的 MTE2_S 由 CombineTokensInRange 在每批搬入后统一完成。
    // FP8 记录的 scale 位于 Align256(n) 之后；发送长度同时也是接收 GM 的 record stride。
    int64_t quantTokenSize = CalcCombineTokenBytes(n, sizeof(DataType), IsQuantized) / sizeof(DataType);
    uint32_t toRankId = metaInfoTensor[batchStart * META_INFO_SIZE + RANK_ID];
    uint32_t tokenIdx = metaInfoTensor[batchStart * META_INFO_SIZE + TOKEN_ID];
    uint32_t topkIdx = metaInfoTensor[batchStart * META_INFO_SIZE + TOPK_INDEX];

    __gm__ DataType* gmLocalD = nullptr;
    uint64_t gmRemoteOffset = peermem.combineSend.offsetBytes;
    GM_ADDR srcAddr = localSrcPtr;

    if (toRankId == rankId) {
        srcAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
    }

    gmLocalD = reinterpret_cast<__gm__ DataType*>(srcAddr);
    uint64_t dstBaseOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->compute.topK + topkIdx) * quantTokenSize;

    if constexpr (!IsQuantized) {
        if (toRankId == rankId) {
            __gm__ DataType* gmm2OutGm = reinterpret_cast<__gm__ DataType*>(localSrcPtr);
            // ubQuant 槽的复用由本 token 尾部的 MTE3_S 与 Group 尾的 MTE3_MTE2 交接保护。
            CopyGmToUb(ubQuant, gmm2OutGm, static_cast<uint32_t>(quantTokenSize * sizeof(DataType)));
            PipeFence<PIPE_MTE2, PIPE_MTE3, SYNC_EVENT_ID4>();
        }
    }

    if (IsQuantized || toRankId == rankId) {
        uint64_t dstOffset = toRankId == rankId ? dstBaseOffset : 0;
        CopyUbToGm(gmLocalD + dstOffset, ubQuant, static_cast<uint32_t>(quantTokenSize * sizeof(DataType)));
        PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID4>();
    }

    if (toRankId != rankId) {
        // 调用方已经按目标 rank 和固定归属核完成分组，此处直接向已绑定的批量句柄追加写 WQE。
        GM_ADDR remoteAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset) + dstBaseOffset * sizeof(DataType);
        // Combine 使用 Hcomm 默认 WQE 配置，不能复用 Dispatch 一级的无 CQE 配置。
        comm.WriteNbi(batchHandle, remoteAddr, srcAddr, quantTokenSize * sizeof(DataType));
    }
}

/* 发送到本 Server 中继，metadata 中的 rank 仍表示 token 原始输入卡。 */
template <typename CommPolicy>
__aicore__ inline void CombineSendTokenToRelay(uint32_t batchStart, uint32_t n, uint32_t rankId,
                                               LocalTensor<int32_t>& metaInfoTensor,
                                               const LocalTensor<bfloat16_t>& tokenTensor, const Params& params,
                                               GM_ADDR localSrcPtr, CommPolicy& comm,
                                               LayeredCombineBatchState<CommPolicy>& batchState)
{
    const auto& peermem = params.communication.layout;
    uint32_t originRank = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOPK_INDEX);

    const uint32_t originServer = originRank / params.tilingData->rankNumPerServer;
    const uint64_t slot =
        (static_cast<uint64_t>(originServer) * params.tilingData->compute.numMaxTokensPerRank + tokenIdx) *
            params.tilingData->compute.topK +
        topkIdx;
    const uint64_t tokenBytes = static_cast<uint64_t>(n) * sizeof(bfloat16_t);
    const uint32_t targetRank = batchState.dstRankId;
    const uint64_t dataOffset = peermem.contribution.offsetBytes + slot * tokenBytes;
    const uint64_t readyOffset = peermem.contributionReady.offsetBytes + slot * sizeof(uint64_t);
    if (targetRank == rankId) {
        // 本地贡献整 token 复制，写回完成后发布 ready。
        __gm__ bfloat16_t* gmm2OutGlobalTensor = nullptr;
        __gm__ bfloat16_t* contributionGlobalTensor = nullptr;
        gmm2OutGlobalTensor = reinterpret_cast<__gm__ bfloat16_t*>(localSrcPtr);
        contributionGlobalTensor =
            reinterpret_cast<__gm__ bfloat16_t*>(GetRankWinAddrWithOffset(targetRank, dataOffset));
        const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(tokenBytes), 0U, 0U, 0U};
        CopyGmToUb(reinterpret_cast<__ubuf__ bfloat16_t*>((tokenTensor).GetPhyAddr()), gmm2OutGlobalTensor,
                   static_cast<uint32_t>(tokenBytes));
        PipeFence<PIPE_MTE2, PIPE_MTE3, SYNC_EVENT_ID2>();
        CopyUbToGm(contributionGlobalTensor, reinterpret_cast<__ubuf__ bfloat16_t*>((tokenTensor).GetPhyAddr()),
                   static_cast<uint32_t>(tokenBytes));
        PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();
        GmWrite(reinterpret_cast<__gm__ uint64_t*>(GetRankWinAddrWithOffset(targetRank, readyOffset)), uint64_t(1));
    } else {
        comm.WriteWithNotifyNbi(batchState.batchHandle, GetRankWinAddrWithOffset(targetRank, dataOffset), localSrcPtr,
                                static_cast<uint32_t>(tokenBytes), GetRankWinAddrWithOffset(targetRank, readyOffset),
                                uint64_t(1));
    }
}

// Layered 的逐 token 量化与发送；Group 自管本批 metadata 搬入与 UB 绑定，外层只给行范围和预算。
template <typename T, typename CommPolicy>
__aicore__ inline void CombineTokenGroup(uint8_t QuantMode, uint32_t tokenStart, uint32_t tokenCount, uint32_t n,
                                         uint32_t groupIdx, uint32_t rankId, GM_ADDR gmm2OutAddr, GM_ADDR metaInfoAddr,
                                         int64_t ubTensorSize, const Params& params, CommPolicy& comm,
                                         LayeredCombineBatchState<CommPolicy>& batchState)
{
    const bool IsQuantized = QuantMode != COMBINE_NO_QUANT;
    // metadata 从 UB 512B 起，整 token 搬运缓冲紧随其后；布局与 InitCombineBuffers 的预算一致。
    int64_t offset = ALIGN_512;
    __ubuf__ int32_t* metaInfoTensor = reinterpret_cast<__ubuf__ int32_t*>(offset);
    offset += tokenCount * META_INFO_SIZE * sizeof(int32_t);
    __gm__ int32_t* metaInfoGm = reinterpret_cast<__gm__ int32_t*>(metaInfoAddr);
    PipeFence<PIPE_S, PIPE_MTE2, SYNC_EVENT_ID1>();
    CopyGmToUb(metaInfoTensor, metaInfoGm, (tokenCount * META_INFO_SIZE) * sizeof(int32_t));
    // metadata 每批搬入后统一等待一次，随后逐 token 直接读取。
    PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID3>();
    __ubuf__ T* combineUbTensor = reinterpret_cast<__ubuf__ T*>(offset);
    offset += ubTensorSize * sizeof(T);

    uint32_t quantTokenSizeBytes = static_cast<uint32_t>(CalcCombineTokenBytes(n, sizeof(T), true));
    uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t mxScaleNum = Ops::Base::CeilAlign(nScale, 2U);
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
    uint32_t floatTempSize = Ops::Base::CeilAlign(mxScaleNum, static_cast<uint32_t>(ALIGN_32)) + mxScaleNum / 2;
    __ubuf__ float* floatTemp = reinterpret_cast<__ubuf__ float*>(offset);

    __gm__ T* gmm2OutGm = reinterpret_cast<__gm__ T*>(gmm2OutAddr);

    uint32_t singleTokenElems = (nAlign32 * sizeof(T) + quantTokenSizeBytes) / sizeof(T);

    for (uint32_t i = 0; i < tokenCount; i++) {
        uint32_t pingPong = i % 2;
        __ubuf__ T* ubBf16 = combineUbTensor + pingPong * singleTokenElems;
        __ubuf__ T* ubQuantData = ubBf16 + nAlign32;

        if (IsQuantized) {
            PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>();
            CopyGmToUb(ubBf16, gmm2OutGm + (tokenStart + i) * n, static_cast<uint32_t>(n * sizeof(T)));
            PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID4>();
            Mxfp8::QuantMxFp8<T>(QuantMode, ubQuantData, ubBf16, floatTemp, n);
            PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID5>();
        }

        GM_ADDR localSrcPtr = gmm2OutAddr + (tokenStart + i) * n * sizeof(T);
        if (!IsQuantized) {
            CombineSendTokenToRemote<T>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor, ubBf16, params, localSrcPtr,
                                        comm, batchState.batchHandle);
        } else if (QuantMode == MXFP8_E4M3_COMM_QUANT) {
            CombineSendTokenToRemote<fp8_e4m3fn_t>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor,
                                                   reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(ubQuantData), params,
                                                   localSrcPtr, comm, batchState.batchHandle);
        } else {
            CombineSendTokenToRemote<fp8_e5m2_t>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor,
                                                 reinterpret_cast<__ubuf__ fp8_e5m2_t*>(ubQuantData), params,
                                                 localSrcPtr, comm, batchState.batchHandle);
        }
        if (batchState.dstRankId != rankId) {
            ++batchState.pendingTokenCount;
            if (batchState.pendingTokenCount == L1_TILE_M_256) {
                comm.BatchCommit(batchState.batchHandle);
                batchState.pendingTokenCount = 0U;
            }
        }
    }
    PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID2>();
}

/*
 * Server 去重发送 BF16 专家结果；本地复制由发送函数负责。
 */
template <typename CommPolicy>
__aicore__ inline void CombineServerDedupTokenGroup(uint32_t tokenStart, uint32_t tokenCount, uint32_t n,
                                                    uint32_t groupIdx, uint32_t rankId, GM_ADDR gmm2OutAddr,
                                                    GM_ADDR metaInfoAddr, const Params& params, CommPolicy& comm,
                                                    LayeredCombineBatchState<CommPolicy>& batchState)
{
    constexpr uint32_t BATCH_TOKENS = 256U;
    int64_t offset = ALIGN_512;
    LocalTensor<int32_t> metaInfoTensor(TPosition::VECIN, offset, tokenCount * META_INFO_SIZE);
    offset += tokenCount * META_INFO_SIZE * sizeof(int32_t);
    // 紧接 metadata 准备整 token 搬运缓冲，当前阶段由本 Group 复用。
    LocalTensor<bfloat16_t> tokenTensor(TPosition::VECIN, offset, n);
    __gm__ int32_t* metaInfoGlobalTensor = nullptr;
    metaInfoGlobalTensor = reinterpret_cast<__gm__ int32_t*>(metaInfoAddr);
    PipeFence<PIPE_S, PIPE_MTE2, SYNC_EVENT_ID1>();
    CopyGmToUb(reinterpret_cast<__ubuf__ int32_t*>((metaInfoTensor).GetPhyAddr()), metaInfoGlobalTensor,
               (tokenCount * META_INFO_SIZE) * sizeof(int32_t));
    // 整批 metadata 搬入完成后，再逐 token 读取字段。
    PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID3>();
    for (uint32_t i = 0; i < tokenCount; i++) {
        GM_ADDR localSrcPtr = gmm2OutAddr + static_cast<uint64_t>(tokenStart + i) * n * sizeof(bfloat16_t);
        CombineSendTokenToRelay(i, n, rankId, metaInfoTensor, tokenTensor, params, localSrcPtr, comm, batchState);
        if (batchState.dstRankId != rankId) {
            ++batchState.pendingTokenCount;
            if (batchState.pendingTokenCount == BATCH_TOKENS) {
                comm.BatchCommit(batchState.batchHandle);
                batchState.pendingTokenCount = 0U;
            }
        }
    }
}

} // namespace MegaMoeImpl::CombineImpl
#endif
#endif
