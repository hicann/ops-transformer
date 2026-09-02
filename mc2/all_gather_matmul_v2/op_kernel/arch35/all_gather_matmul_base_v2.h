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
 * \file all_gather_matmul_base_v2.h
 * \brief
 */

#ifndef ALL_GATHER_MATMUL_V2_BASE_H
#define ALL_GATHER_MATMUL_V2_BASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../agm_v2_common.h"
#include "lib/hccl/hccl.h"
#include "../all_gather_matmul_tiling_data.h"

namespace AllGatherMatmulImpl {
using namespace AscendC;
constexpr uint64_t PERBLOCK_BLOCK_SIZE = 128;
constexpr uint32_t MAX_RANK_DIM = 64;

template <uint8_t commMode>
struct HcclTypeSelector {
    static_assert(commMode == HCCL_COMM_MODE_CCU || commMode == HCCL_COMM_MODE_AICPU,
                  "commMode must be 0 (CCU) or 1 (AICPU)");
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;
};

template <>
struct HcclTypeSelector<HCCL_COMM_MODE_AICPU> {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};

template <typename AType, typename CType, Mc2CoreType CoreType, uint8_t ServerMode = HCCL_COMM_MODE_CCU>
class AllGatherMatmulBase {
public:
    __aicore__ inline AllGatherMatmulBase(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs,
                                          Mc2Tiling::AllGatherMatmulTilingDataFp8 *tilingData, GM_ADDR contextGM,
                                          TPipe *tPipe)
        : addrs_(addrs),
          quantAddrs_(quantAddrs),
          tPipe_(tPipe)
    {
        tilingData_ = tilingData;
        paramInTiling_ = &tilingData->param;
        dataType_ = static_cast<AscendC::HcclDataType>(tilingData->dataType);
        debugMode_ = tilingData->debugMode;
        context_ = contextGM;
    }
    __aicore__ inline void Init(__gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling)
    {
        hccl_.InitV2(context_, &(tilingData_->mc2InitTiling));
        hccl_.SetCcTilingV2(offsetof(Mc2Tiling::AllGatherMatmulTilingDataFp8, mc2CcTiling));
        rankId_ = hccl_.GetRankId();
        // 计算scale1的数据个数及地址偏移
        nBlockSizeCnt_ = (tileInfo_.mmTiling->matmulTiling.Ka + PERBLOCK_BLOCK_SIZE - 1) / PERBLOCK_BLOCK_SIZE;
        oneCommBlockSizeOffset_ = static_cast<uint64_t>(tileInfo_.mmTiling->matmulTiling.M / PERBLOCK_BLOCK_SIZE) *
                                  nBlockSizeCnt_ * sizeof(float);
        oneRankBlockSizeCnt_ = static_cast<uint64_t>(paramInTiling_->rankM / PERBLOCK_BLOCK_SIZE) * nBlockSizeCnt_;
        oneRankBlockSizeOffset_ = oneRankBlockSizeCnt_ * sizeof(float);
        // 指定workspace空间来存放收集的各卡scale1
        gatherScale1Addr_ = addrs_->workspaceGM;
        // 若有指定gatherX1Addr_地址则用指定地址，若未指定gatherX1Addr_地址 则为workspace空间
        gatherX1Addr_ = addrs_->gatherOut;
        if ((paramInTiling_->gatherLen != 0) || (!addrs_->gatherOut)) {
            // 留出scale1的偏移空间
            uint64_t gatherScale1Offest =
                static_cast<uint64_t>(paramInTiling_->rankDim) * static_cast<uint64_t>(oneRankBlockSizeOffset_);
            gatherX1Addr_ = addrs_->workspaceGM + gatherScale1Offest;
            addrs_->workspaceGM += paramInTiling_->gatherLen;
        }
        gatherScaleAddr_ = addrs_->workspaceGM;

        UpdateNotifyFlag();
        UpdateMC2TileInfo();
        UpdateBatchWeight();
    }

protected:
    __aicore__ inline void HcclStart()
    {
        if (notifyFlag_) {
            uint8_t repeat = 1;

            // 先进行scale1通信的prepare工作
            GM_ADDR sendBuffer = quantAddrs_->scale2GM;
            GM_ADDR recvBuffer = gatherScale1Addr_;
            // 高阶API调用和tiling需要统一datatype;scale1数据类型为fp32.乘以SCALE1_MULTIPLE来适配通信
            uint64_t scaleCount = oneRankBlockSizeCnt_ * SCALE1_MULTIPLE;
            uint64_t scaleStride = oneRankBlockSizeCnt_ * SCALE1_MULTIPLE;
            hcclHandleIdList[SCALE1_HANDLE_IDX] =
                hccl_.template AllGather<true>(sendBuffer, recvBuffer, scaleCount, dataType_, scaleStride, repeat);

            // 再进行x1通信的prepare工作
            uint32_t tileCnt = paramInTiling_->tileCnt;
            uint32_t tailCnt = paramInTiling_->tailCnt;
            uint64_t stride = tileInfo_.aOffset * tileCnt + tailInfo_.aOffset * tailCnt;

            // 处理主块
            for (uint32_t i = 0; i < tileCnt; i++) {
                // 计算sendBuffer, recvBuffer偏移
                sendBuffer = addrs_->aGM + i * tileInfo_.aAddrOffset;
                recvBuffer = gatherX1Addr_ + i * tileInfo_.aAddrOffset;
                hcclHandleIdList[i] = hccl_.template AllGather<true>(sendBuffer, recvBuffer, tileInfo_.aOffset,
                                                                     dataType_, stride, repeat);
            }

            // 处理尾块
            for (uint32_t i = tileCnt; i < tileCnt + tailCnt; i++) {
                // 计算 sendBuff recvBuff 偏移
                sendBuffer = addrs_->aGM + tileCnt * tileInfo_.aAddrOffset + (i - tileCnt) * tailInfo_.aAddrOffset;
                recvBuffer = gatherX1Addr_ + tileCnt * tileInfo_.aAddrOffset + (i - tileCnt) * tailInfo_.aAddrOffset;
                hcclHandleIdList[i] = hccl_.template AllGather<true>(sendBuffer, recvBuffer, tailInfo_.aOffset,
                                                                     dataType_, stride, repeat);
            }
        }
    }

    __aicore__ inline void HcclFinalize()
    {
        if (notifyFlag_ && (AscendC::GetBlockIdx() == 0)) {
            hccl_.Finalize();
        }
    }

    __aicore__ inline void HcclWaitScale()
    {
        // 先完成scale1通信的wait工作
        if (notifyFlag_) {
            hccl_.Wait(hcclHandleIdList[SCALE1_HANDLE_IDX]);
        }
    }

    __aicore__ inline void HcclWait(uint32_t id, bool isTail = false)
    {
        // 开始计算前确认是否搬运完成，debug模式只做本卡内计算，则不需要判断搬运状态
        if (notifyFlag_) {
            uint32_t shift = isTail ? paramInTiling_->tileCnt : 0; // 若是尾快需要加上前tile块偏移
            hccl_.Wait(hcclHandleIdList[id + shift]);
        }
    }

    MC2GmAddrs *addrs_ = nullptr;
    QuantGmAddrs *quantAddrs_ = nullptr;
    Mc2Tiling::RCSTiling *paramInTiling_ = nullptr;
    Mc2Tiling::MC2TileInfo localInfo_;
    Mc2Tiling::MC2TileInfo tileInfo_;
    Mc2Tiling::MC2TileInfo tailInfo_;
    typename HcclTypeSelector<ServerMode>::type hccl_;
    AscendC::HcclDataType dataType_;
    GM_ADDR context_;
    bool notifyFlag_;
    uint32_t rankId_;
    uint64_t oneCommBlockSizeOffset_{0U};
    uint64_t nBlockSizeCnt_{0U};
    uint64_t oneRankBlockSizeCnt_{0U};
    uint64_t oneRankBlockSizeOffset_{0U};
    uint8_t debugMode_;
    TPipe *tPipe_;
    GM_ADDR gatherX1Addr_;
    GM_ADDR gatherScale1Addr_;
    GM_ADDR gatherScaleAddr_;
    AscendC::HcclHandle hcclHandleIdList[MAX_HANDLE_WITH_SCALE1]; /* hccl handle */
    uint32_t batchWeight_[MAX_RANK_DIM] = {0};
    Mc2Tiling::AllGatherMatmulTilingDataFp8 *tilingData_;

private:
    __aicore__ inline void UpdateNotifyFlag()
    {
        if constexpr (CoreType == Mc2CoreType::ON_CUBE) {
            notifyFlag_ = (g_coreType == AscendC::AIC);
        } else {
            notifyFlag_ = (g_coreType == AscendC::AIV) && (AscendC::GetBlockIdx() == 0);
        }
    }

    __aicore__ inline void UpdateMC2TileInfo()
    {
        tileInfo_.aOffset =
            (uint64_t)tileInfo_.mmTiling->matmulTiling.M * (uint64_t)tileInfo_.mmTiling->matmulTiling.Ka;
        tileInfo_.aAddrOffset = tileInfo_.aOffset * sizeof(AType);
        tileInfo_.cOffset = (uint64_t)tileInfo_.mmTiling->matmulTiling.M * (uint64_t)tileInfo_.mmTiling->matmulTiling.N;
        tileInfo_.cAddrOffset = tileInfo_.cOffset * sizeof(CType);
        uint32_t tailCnt = paramInTiling_->tailCnt;
        if (tailCnt) {
            tailInfo_.aOffset =
                (uint64_t)tailInfo_.mmTiling->matmulTiling.M * (uint64_t)tailInfo_.mmTiling->matmulTiling.Ka;
            tailInfo_.aAddrOffset = tailInfo_.aOffset * sizeof(AType);
            tailInfo_.cOffset =
                (uint64_t)tailInfo_.mmTiling->matmulTiling.M * (uint64_t)tailInfo_.mmTiling->matmulTiling.N;
            tailInfo_.cAddrOffset = tailInfo_.cOffset * sizeof(CType);
        }
        // 当N=0时，matmulTiling参数为0，重新计算通信参数
        if (paramInTiling_->rankN == 0) {
            uint32_t nTilingKa = paramInTiling_->rankK;
            uint32_t nTailM = paramInTiling_->tailM;
            uint32_t nTileM = (paramInTiling_->rankM - nTailM * tailCnt) / paramInTiling_->tileCnt;
            tileInfo_.aOffset = (uint64_t)nTileM * (uint64_t)nTilingKa;
            tileInfo_.aAddrOffset = tileInfo_.aOffset * sizeof(AType);
            tailInfo_.aOffset = (uint64_t)nTailM * (uint64_t)nTilingKa;
            tailInfo_.aAddrOffset = tailInfo_.aOffset * sizeof(AType);
            // 重新计算scale1通信相关参数
            nBlockSizeCnt_ = (nTilingKa + PERBLOCK_BLOCK_SIZE - 1) / PERBLOCK_BLOCK_SIZE;
            oneCommBlockSizeOffset_ =
                static_cast<uint64_t>(nTileM / PERBLOCK_BLOCK_SIZE) * nBlockSizeCnt_ * sizeof(float);
            oneRankBlockSizeCnt_ = static_cast<uint64_t>(paramInTiling_->rankM / PERBLOCK_BLOCK_SIZE) * nBlockSizeCnt_;
            oneRankBlockSizeOffset_ = oneRankBlockSizeCnt_ * sizeof(float);
        }
    }

    __aicore__ inline void UpdateBatchWeight()
    {
        // 计算batch权重值
        uint32_t k = 0;
        for (uint32_t j = 0; j < paramInTiling_->rankDim; j++) {
            if (j == rankId_) {
                continue;
            }
            batchWeight_[k] = j;
            k++;
        }
    }
};

} // namespace AllGatherMatmulImpl

#endif
