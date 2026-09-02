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
 * \file matmul_all_reduce_base.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_BASE_H
#define MATMUL_ALL_REDUCE_BASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"
#include "matmul_all_reduce_add_x3.h"
#include "../unquant_matmul_all_reduce_tiling_data.h"
#include "../weight_quant_matmul_all_reduce_tiling_data.h"
#include "../quant_matmul_all_reduce_tiling_data.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename xType, typename yType, Mc2CoreType coreType>
class MatmulAllReduceBase {
public:
    __aicore__ inline MatmulAllReduceBase(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs,
                                          MC2TilingHeader *tilingData, TPipe *tPipe)
        : addrs_(addrs),
          quantAddrs_(quantAddrs),
          arnAddrs_(arnAddrs),
          tPipe_(tPipe)
    {
        if constexpr (coreType == Mc2CoreType::ON_CUBE) {
            notifyFlag_ = (GetBlockIdx() == 0);
        } else {
            notifyFlag_ = (g_coreType == AscendC::AIV && GetBlockIdx() == 0);
        }
        msgInTiling_ = &tilingData->msg;
        paramInTiling_ = &tilingData->param;
    }

    __aicore__ inline void Init()
    {
        hccl_.Init(GetHcclContext<0>());

        __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
        OOMInit(context);

        if (msgInTiling_->useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN &&
            context->config.determinism != 1) {
            addrs_->cGM = hccl_.GetWindowsInAddr(hccl_.GetRankId());
        }

        addFlag_ = (paramInTiling_->isAdd != 0U);
        tailFlag_ = (paramInTiling_->tailCnt != 0U);

        tileInfo_.aOffset = (uint64_t)tileInfo_.mmTiling->M * (uint64_t)tileInfo_.mmTiling->Ka;
        tileInfo_.aAddrOffset = tileInfo_.aOffset * sizeof(xType);
        tileInfo_.cOffset = (uint64_t)tileInfo_.mmTiling->M * (uint64_t)tileInfo_.mmTiling->N;
        tileInfo_.cAddrOffset = tileInfo_.cOffset * sizeof(yType);
        if (tailFlag_) {
            tailInfo_.aOffset = (uint64_t)tailInfo_.mmTiling->M * (uint64_t)tailInfo_.mmTiling->Ka;
            tailInfo_.aAddrOffset = tailInfo_.aOffset * sizeof(xType);
            tailInfo_.cOffset = (uint64_t)tailInfo_.mmTiling->M * (uint64_t)tailInfo_.mmTiling->N;
            tailInfo_.cAddrOffset = tailInfo_.cOffset * sizeof(yType);
        }

        if (notifyFlag_) {
            tileInfo_.hcclHandleId = hccl_.AllReduce(addrs_->cGM, addrs_->outputGM, tileInfo_.cOffset, HCCL_DATA_TYPE,
                                                     AscendC::HCCL_REDUCE_SUM, paramInTiling_->tileCnt);
            if (tailFlag_) {
                const uint64_t offset = tileInfo_.cAddrOffset * paramInTiling_->tileCnt;
                tailInfo_.hcclHandleId =
                    hccl_.AllReduce(addrs_->cGM + offset, addrs_->outputGM + offset, tailInfo_.cOffset, HCCL_DATA_TYPE,
                                    AscendC::HCCL_REDUCE_SUM, paramInTiling_->tailCnt);
            }
        }
    }

protected:
#if (ORIG_DTYPE_X1 == DT_BF16)
    __aicore__ inline void PreProcForBiasOnVector()
    {
        if (paramInTiling_->biasLen == 0U) {
            return;
        }

        TBuf<TPosition::VECCALC> tmpBuf;
        tPipe_->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        CastBFtoFloatOnAiv0(addrs_->workspaceGM, addrs_->biasGM, paramInTiling_->rankN, tmpBuf);
        SyncAll<false>();
        addrs_->biasGM = addrs_->workspaceGM;
        addrs_->workspaceGM += paramInTiling_->biasLen;
    }
#endif

    __aicore__ inline void PostProcEachTurn(AscendC::HcclHandle handleId, uint64_t aOffset, uint64_t cOffset)
    {
        if (addFlag_ && addrs_->cGM != addrs_->addGM) {
            Mc2SyncAll<coreType>();
            Matmul_All_Reduce_Add_X3<yType>(addrs_->cGM, addrs_->addGM, cOffset / sizeof(yType),
                                            paramInTiling_->addX3UbCnt, tPipe_);
            addrs_->addGM += cOffset;
        }

        addrs_->aGM += aOffset;
        addrs_->cGM += cOffset;

        Mc2SyncAll<coreType>();
        if (notifyFlag_) {
            hccl_.Commit(handleId);
        }
    }

    __aicore__ inline void HcclFinalize()
    {
        if (notifyFlag_) {
            hccl_.Wait(tileInfo_.hcclHandleId);
            if (tailFlag_) {
                hccl_.Wait(tailInfo_.hcclHandleId);
            }
        }

        Mc2SyncAll<coreType>();
        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }

    MC2GmAddrs *addrs_;
    QuantGmAddrs *quantAddrs_;
    ArnGmAddrs *arnAddrs_;
    Mc2Tiling::Mc2Msg *msgInTiling_;
    Mc2Tiling::RCSTiling *paramInTiling_;
    MC2TileInfo tileInfo_, tailInfo_;
    TPipe *tPipe_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    bool notifyFlag_;
    bool addFlag_;
    bool tailFlag_;
};
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_BASE_H
