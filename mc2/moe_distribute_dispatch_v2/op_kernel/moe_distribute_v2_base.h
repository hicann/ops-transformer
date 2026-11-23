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
 * \file moe_distribute_v2_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_V2_BASE_H
#define MOE_DISTRIBUTE_V2_BASE_H

#include "moe_distribute_v2_constant.h"

#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"

namespace MoeDistributeV2Base {

using namespace AscendC;
using namespace Mc2Kernel;

__aicore__ inline uint32_t GetRswCounterOffset(GM_ADDR address, uint32_t minPage, uint32_t trailer)
{
    return (2U * minPage - trailer - reinterpret_cast<uint64_t>(address) % minPage) % minPage;
}

__aicore__ inline void ClearWindowRecords(LocalTensor<uint8_t> zeros, uint32_t capacity, GM_ADDR begin, uint64_t span,
                                          uint32_t offset, uint32_t bytes, uint32_t stride)
{
    if (span == 0U || capacity == 0U) {
        return;
    }
    constexpr uint32_t maxBlockCount = 4095U;
    capacity = capacity < maxBlockCount ? capacity : maxBlockCount;
    GlobalTensor<uint8_t> dst;
    dst.SetGlobalBuffer(begin);
    // A marker can straddle a partition boundary. Only clear the bytes owned by this core.
    if (offset + bytes > stride) {
        const uint32_t overlap = offset + bytes - stride;
        const uint32_t prefix = span < overlap ? static_cast<uint32_t>(span) : overlap;
        DataCopyExtParams params{1U, prefix, 0U, 0U, 0U};
        DataCopyPad(dst, zeros, params);
    }
    uint64_t pos = offset;
    while (pos + bytes <= span) {
        const uint64_t count = (span - pos - bytes) / stride + 1U;
        const uint16_t batch = static_cast<uint16_t>(count < capacity ? count : capacity);
        DataCopyExtParams params{batch, bytes, 0U, stride - bytes, 0U};
        DataCopyPad(dst[pos], zeros, params);
        pos += static_cast<uint64_t>(batch) * stride;
    }
    if (pos < span) {
        DataCopyExtParams params{1U, static_cast<uint32_t>(span - pos), 0U, 0U, 0U};
        DataCopyPad(dst[pos], zeros, params);
    }
}

__aicore__ inline uint32_t InitWinState(GlobalTensor<uint32_t> selfDataStatusGMTensor, uint32_t epRankIdHccl,
                                        uint32_t epWorldSizeHccl, uint32_t epRankIdOriginal, uint32_t moeExpertNum,
                                        uint32_t epWorldSizeOriginal, uint32_t globalBS, TBuf<> dataStateBuf)
{
    LocalTensor<uint64_t> dataStateLocalTensor64 = dataStateBuf.Get<uint64_t>();
    LocalTensor<uint32_t> dataStateLocalTensor = dataStateBuf.Get<uint32_t>();
    DataCopy(dataStateLocalTensor, selfDataStatusGMTensor, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t dataState = dataStateLocalTensor.GetValue(ZERONE_STATE_POS);
    dataStateLocalTensor.SetValue(ZERONE_STATE_POS, dataState == 0 ? 1 : 0);
    dataStateLocalTensor.SetValue(OPOSITION_POS, 1);
    dataStateLocalTensor.SetValue(TILING_EPRANKID_POS, epRankIdOriginal);
    dataStateLocalTensor.SetValue(MOE_NUM_POS, moeExpertNum);
    dataStateLocalTensor.SetValue(TILING_WORLDSIZE_POS, epWorldSizeOriginal);
    dataStateLocalTensor.SetValue(GLOBALBS_POS, globalBS);
    uint64_t opCnt = dataStateLocalTensor64.GetValue(OP_CNT_POSUL);
    opCnt = opCnt + 1UL;
    dataStateLocalTensor64.SetValue(OP_CNT_POSUL, opCnt);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfDataStatusGMTensor, dataStateLocalTensor, UB_ALIGN / sizeof(uint32_t));

    if ((epRankIdOriginal != epRankIdHccl) || (epWorldSizeOriginal != epWorldSizeHccl)) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        DataCopyParams hcclDatacopyParams{1U, HCCL_DFX_NUM * sizeof(uint32_t), 0U, 0U};
        dataStateLocalTensor.SetValue(HCCL_EPRANKId_POS, epRankIdHccl);
        dataStateLocalTensor.SetValue(HCCL_WORLDSIZE_POS, epWorldSizeHccl);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor[HCCL_DFX_POS], dataStateLocalTensor, hcclDatacopyParams);
    }
    return dataState;
}

__aicore__ inline void AIVRDMAPostDoorBell(__gm__ HcclAiRMAWQ* qp_ctx_entry, uint64_t curHead,
                                           __gm__ uint32_t* curHardwareHead, LocalTensor<uint64_t>& ubLocalTensor,
                                           LocalTensor<uint32_t>& ubLocalHeadTensor)
{
    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn;                    // [0:23] DB_TAG (qp_num)
    doorBellInfo |= 0UL << 24UL;                          // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB (0)
    doorBellInfo |= (curHead % 65536UL) << 32UL;          // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL; // [48:50] DB_SL = qp.sl

    __gm__ uint64_t* doorBellAddr = (__gm__ uint64_t*)(qp_ctx_entry->dbAddr);
    PipeBarrier<PIPE_ALL>();

    ubLocalTensor.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocalTensor, copyParams);
    PipeBarrier<PIPE_ALL>();

    ubLocalHeadTensor.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer(curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHeadTensor, copyParamsHead);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AIVRDMAPostSendWriteWqe(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                               uint64_t messageLen, __gm__ HcclAiRMAInfo* QpInfo,
                                               __gm__ uint8_t* wqeAddr, uint64_t curHead)
{
    auto mem_info_table = ((__gm__ HcclAiRMAInfo*)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAMem;
    uint64_t ownBit = (curHead >> 15U) & 1U;
    uint32_t byte_4 = 3U;
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);
    byte_4 |= 1U << 8U;
    *(__gm__ uint32_t*)(wqeAddr) = byte_4;
    *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen;
    *(__gm__ uint32_t*)(wqeAddr + 8) = 0;
    *(__gm__ uint32_t*)(wqeAddr + 12) = 1U << 24U;
    *(__gm__ uint32_t*)(wqeAddr + 16) = 0;
    __gm__ HcclAiRMAMemInfo* memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(wqeAddr + 20) =
        ((__gm__ MemDetails*)(memDetail->memDetailPtr +
                              memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))
            ->key;
    *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)destDmaAddr;

    __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t*)(sgeAddr) = messageLen;
    *(__gm__ uint32_t*)(sgeAddr + sizeof(uint32_t)) =
        ((__gm__ MemDetails*)(memDetail->memDetailPtr +
                              memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))
            ->key;
    *(__gm__ uint64_t*)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)srcDmaAddr;
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
}

__aicore__ inline void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                       uint64_t messageLen, __gm__ HcclAiRMAInfo* QpInfo,
                                       LocalTensor<uint64_t>& ubLocalTensor, LocalTensor<uint32_t>& ubLocalHeadTensor)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo*)QpInfo)->qpNum;
    auto qp_ctx_entry =
        (__gm__ HcclAiRMAWQ*)(((__gm__ HcclAiRMAInfo*)QpInfo)->sqPtr +
                              destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAWQ));
    auto cur_rank_id = (((__gm__ HcclAiRMAInfo*)QpInfo)->curRankId);
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t*)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t*)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    auto QP_DEPTH = qp_ctx_entry->depth;

    PipeBarrier<PIPE_ALL>();

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while (1) {
        cacheWriteThrough((__gm__ uint8_t*)curHardwareTailAddr, CACHEWRITESIZE);
        if ((curHead - *(__gm__ uint32_t*)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        AscendC::GetSystemCycle(); // add this line to solve slow poll CQ issue
    }

    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    AIVRDMAPostSendWriteWqe(srcDmaAddr, destDmaAddr, destRankId, messageLen, QpInfo, wqeAddr, curHead);
    PipeBarrier<PIPE_ALL>();
    curHead++;
    AIVRDMAPostDoorBell(qp_ctx_entry, curHead, (__gm__ uint32_t*)curHardwareHead, ubLocalTensor, ubLocalHeadTensor);
}

} // namespace MoeDistributeV2Base
#endif // MOE_DISTRIBUTE_V2_BASE_H
