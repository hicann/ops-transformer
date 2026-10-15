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

#include "../common/mega_moe_utils.h"
#include "../common/mega_moe_mxfp8_utils.h"

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
namespace MegaMoeImpl::CombineImpl {

template <typename CommPolicy>
struct LayeredCombineBatchState {
    uint32_t dstRankId = 0U;
    typename CommPolicy::BatchHandle batchHandle{};
    uint32_t pendingTokenCount = 0U;
};

// 通过本地 MTE 路径或远端 URMA 路径发送一个 layered 模板 token。
template <typename DataType, bool IsQuantized, typename CommPolicy>
__aicore__ inline void CombineSendTokenToRemote(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                                uint32_t groupIdx, uint32_t rankId,
                                                LocalTensor<int32_t>& metaInfoTensor, LocalTensor<DataType>& ubQuant,
                                                const Params& params, GM_ADDR localSrcPtr, CommPolicy& comm,
                                                typename CommPolicy::BatchHandle& batchHandle)
{
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

    gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(srcAddr));
    uint64_t dstBaseOffset = (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantTokenSize;
    AscendC::DataCopyExtParams singleCopyParams{1, static_cast<uint32_t>(quantTokenSize * sizeof(DataType)), 0, 0, 0};

    if constexpr (!IsQuantized) {
        DataCopyPadExtParams<DataType> copyPadParams{false, 0U, 0U, 0U};
        if (toRankId == rankId) {
            AscendC::GlobalTensor<DataType> gmm2OutGm;
            gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(localSrcPtr));
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
        // 调用方已经按目标 rank 和固定归属核完成分组，此处直接向已绑定的批量句柄追加写 WQE。
        GM_ADDR remoteAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset) + dstBaseOffset * sizeof(DataType);
        // Combine 使用 Hcomm 默认 WQE 配置，不能复用 Dispatch 一级的无 CQE 配置。
        comm.WriteNbi(batchHandle, remoteAddr, srcAddr, quantTokenSize * sizeof(DataType));
    }
}

// Layered 的逐 token 量化与发送；通信策略只接管请求构造和提交。
template <uint8_t QuantMode, typename T, bool IsQuantized, typename CommPolicy>
__aicore__ inline void CombineTokenGroup(uint32_t tokenStart, uint32_t tokenCount, uint32_t n, uint32_t groupIdx,
                                         uint32_t rankId, GM_ADDR gmm2OutAddr, const Params& params,
                                         LocalTensor<int32_t>& metaInfoTensor, int64_t ubTensorSize, int64_t offset,
                                         uint32_t quantTokenSizeBytes, CommPolicy& comm,
                                         LayeredCombineBatchState<CommPolicy>& batchState)
{
    LocalTensor<T> combineUbTensor(TPosition::VECIN, offset, ubTensorSize);
    offset += ubTensorSize * sizeof(T);

    uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t mxScaleNum = Ops::Base::CeilAlign(nScale, 2U);
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
    uint32_t floatTempSize = Ops::Base::CeilAlign(mxScaleNum, static_cast<uint32_t>(ALIGN_32)) + mxScaleNum / 2;
    LocalTensor<float> floatTemp = LocalTensor<float>(TPosition::VECIN, offset, floatTempSize);

    GlobalTensor<T> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gmm2OutAddr));
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
        if constexpr (!IsQuantized) {
            ubQuantSend = ubBf16;
        }
        GM_ADDR localSrcPtr = gmm2OutAddr + (tokenStart + i) * n * sizeof(T);
        CombineSendTokenToRemote<SendType, IsQuantized>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor, ubQuantSend,
                                                        params, localSrcPtr, comm, batchState.batchHandle);
        if (batchState.dstRankId != rankId) {
            ++batchState.pendingTokenCount;
            if (batchState.pendingTokenCount == L1_TILE_M_256) {
                comm.BatchCommit(batchState.batchHandle);
                batchState.pendingTokenCount = 0U;
            }
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
}

} // namespace MegaMoeImpl::CombineImpl
#endif
#endif
