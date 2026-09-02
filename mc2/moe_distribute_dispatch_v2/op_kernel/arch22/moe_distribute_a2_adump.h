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
 * \file moe_distribute_a2_adump.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_A2_ADUMP_H
#define MOE_DISTRIBUTE_A2_ADUMP_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "moe_distribute_a2_constant.h"
namespace Mc2A2Kernel {

using namespace AscendC;
class MoeDistributeA2ADump {
public:
    __aicore__ inline void Init(GM_ADDR aDumpWinStartAddr, uint32_t aivId, uint32_t epRankIdHccl,
                                uint32_t epWorldSizeHccl, uint32_t epRankIdOriginal, uint32_t moeExpertNum,
                                uint32_t epWorldSizeOriginal, uint32_t globalBs, uint32_t bufferId, uint32_t isLayered,
                                uint32_t aivNum, LocalTensor<uint8_t> dataStateLocalTensor, uint32_t axisH,
                                uint32_t axisBS)
    {
        selfDataStatusGMTensor_.SetGlobalBuffer((__gm__ uint32_t *)(aDumpWinStartAddr + aivId * WIN_ADDR_ALIGN));
        selfTokenNumGMTensor_.SetGlobalBuffer((__gm__ uint32_t *)(aDumpWinStartAddr + (aivNum + 1) * WIN_ADDR_ALIGN));
        aivId_ = aivId;
        uint32_t dataSize = UB_ALIGN * 2U;
        dataStateLocalTensor_ = dataStateLocalTensor.template ReinterpretCast<uint32_t>();
        auto dataStateLocalTensor64 = dataStateLocalTensor.template ReinterpretCast<uint64_t>();
        worldSize_ = epWorldSizeHccl;
        CleanUp(isLayered);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(dataStateLocalTensor_, selfDataStatusGMTensor_, UB_ALIGN / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        dataStateLocalTensor_.SetValue(BUFFERID_POS, bufferId);
        dataStateLocalTensor_.SetValue(OPOSITION_POS, RUNPOS_INIT);
        dataStateLocalTensor_.SetValue(TILING_EPRANKID_POS, epRankIdOriginal);
        dataStateLocalTensor_.SetValue(MOE_NUM_POS, moeExpertNum);
        dataStateLocalTensor_.SetValue(TILING_WORLDSIZE_POS, epWorldSizeOriginal);
        dataStateLocalTensor_.SetValue(GLOBALBS_POS, globalBs);
        dataStateLocalTensor_.SetValue(H_POS, axisH);
        dataStateLocalTensor_.SetValue(BS_POS, axisBS);
        if ((epRankIdOriginal != epRankIdHccl) || (epWorldSizeOriginal != epWorldSizeHccl)) {
            dataStateLocalTensor_.SetValue(HCCL_DFX_POS + HCCL_EPRANKId_POS, epRankIdHccl);
            dataStateLocalTensor_.SetValue(HCCL_DFX_POS + HCCL_WORLDSIZE_POS, epWorldSizeHccl);
        }
        dataStateLocalTensor_.SetValue(ISLAYERED_POS, isLayered);
        dataStateLocalTensor_.SetValue(AIVNUM_POS, aivNum);
        uint64_t opCnt = dataStateLocalTensor64.GetValue(OP_CNT_POSUL);
        opCnt = opCnt + 1UL;
        dataStateLocalTensor64.SetValue(OP_CNT_POSUL, opCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        auto dataStateParams = DataCopyExtParams{1U, (H_POS + 1) * sizeof(uint32_t), 0U, 0U, 0U};
        DataCopyPad(selfDataStatusGMTensor_, dataStateLocalTensor_, dataStateParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline void RunPosRecord(const uint32_t runPos)
    {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        dataStateLocalTensor_.SetValue(0, runPos);
        auto dataStateParams = DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor_[OPOSITION_POS], dataStateLocalTensor_, dataStateParams);
    }

    __aicore__ inline void UpdateFullmeshRankTaskInfo(const uint32_t firstEpRankId, const uint32_t rankNum,
                                                      const bool isSend)
    {
        uint32_t offset = isSend ? FULLMESH_SEND_FIRST_EPRANKID_POS : FULLMESH_RECV_FIRST_EPRANKID_POS;
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        dataStateLocalTensor_.SetValue(0U, firstEpRankId);
        dataStateLocalTensor_.SetValue(1U, rankNum);
        uint32_t loc = isSend ? 0 : 1;
        firstEpRankId_[loc] = firstEpRankId;
        rankNum_[loc] = rankNum;
        auto dataStateParams = DataCopyExtParams{1U, 2U * sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor_[offset], dataStateLocalTensor_, dataStateParams);
    }

    __aicore__ inline void UpdateFullmeshEpRankSendOrRecv(const uint32_t epRankId, const bool isSend)
    {
        uint32_t loc = isSend ? 0U : 1U;
        uint32_t relativeLoc = (epRankId - firstEpRankId_[loc]) % (sizeof(uint32_t) * 8);
        uint32_t offset = FULLMESH_EPRANKID_POS + (epRankId - firstEpRankId_[loc]) / (sizeof(uint32_t) * 8);
        if (!isSend) {
            offset += ((rankNum_[0] + (sizeof(uint32_t) * 8U) - 1U) / (sizeof(uint32_t) * 8U));
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(dataStateLocalTensor_, selfDataStatusGMTensor_[offset], UB_ALIGN / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t status = dataStateLocalTensor_.GetValue(0);
        status |= (1U << relativeLoc);
        dataStateLocalTensor_.SetValue(0, status);
        auto dataStateParams = DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor_[offset], dataStateLocalTensor_, dataStateParams);
    }

    __aicore__ inline void UpdateFullmeshTokenSendOrRecv(const uint32_t epRankId, const uint32_t tokenNum,
                                                         const bool isSend)
    {
        uint32_t offset = epRankId;
        offset = isSend ? offset : offset + worldSize_;
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        dataStateLocalTensor_.SetValue(UB_ALIGN / sizeof(uint32_t), tokenNum);
        auto dataStateParams = DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfTokenNumGMTensor_[offset], dataStateLocalTensor_[UB_ALIGN / sizeof(uint32_t)], dataStateParams);
    }

    __aicore__ inline void UpdateHierarchyInterNodeSendOrRecv(uint32_t dstServerId, uint32_t tokenNum, bool isSend)
    {
        uint32_t offset = isSend ? HIERARCHY_INTER_NODE_SEND_POS : HIERARCHY_INTER_NODE_RECV_POS;
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        dataStateLocalTensor_.SetValue(0U, dstServerId + 1U);
        dataStateLocalTensor_.SetValue(1U, tokenNum);
        auto dataStateParams = DataCopyExtParams{1U, 2U * sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor_[offset], dataStateLocalTensor_, dataStateParams);
    }

    __aicore__ inline void UpdateHierarchyInnerSendOrRecv(uint32_t dstServerId, bool isSend)
    {
        uint32_t offset = isSend ? HIERARCHY_INNER_SEND_POS : HIERARCHY_INNER_RECV_POS;
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        dataStateLocalTensor_.SetValue(0U, dstServerId + 1U);
        auto dataStateParams = DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor_[offset], dataStateLocalTensor_, dataStateParams);
    }

    __aicore__ inline void CleanUp(bool isLayered)
    {
        AscendC::Duplicate<uint32_t>(dataStateLocalTensor_, 0U, dataStateLocalTensor_.GetSize());
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        uint32_t num = 0U;
        uint16_t blockNum = 1U;
        if (isLayered) {
            num = HIERARCHY_INNER_RECV_POS - HIERARCHY_INTER_NODE_SEND_POS + 1U;
        } else {
            num = FULLMESH_EPRANKID_POS - FULLMESH_SEND_FIRST_EPRANKID_POS + worldSize_ / (sizeof(uint32_t) * 8U);
            blockNum = (num + dataStateLocalTensor_.GetSize() - 1) / dataStateLocalTensor_.GetSize();
        }
        if (blockNum > 1U) {
            num = blockNum * dataStateLocalTensor_.GetSize();
        }
        auto dataStateParams = DataCopyExtParams{blockNum, static_cast<uint32_t>(num * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPad(selfDataStatusGMTensor_[AIVNUM_POS + 1U], dataStateLocalTensor_, dataStateParams);
    }

private:
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    GlobalTensor<uint32_t> selfTokenNumGMTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    uint32_t firstEpRankId_[2];
    uint32_t rankNum_[2];
    uint32_t aivId_;
    uint32_t worldSize_;
};
} // namespace Mc2A2Kernel

#endif
