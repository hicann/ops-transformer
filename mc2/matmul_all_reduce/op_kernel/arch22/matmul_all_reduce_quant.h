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
 * \file matmul_all_reduce_quant.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_H
#define MATMUL_ALL_REDUCE_QUANT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"
#ifdef MC2_QUANT_BF16
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16.h"
#else
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.h"
#endif
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken.h"
#include "matmul_all_reduce_base.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename xType, typename wType, typename yType, class mmType, Mc2CoreType coreType, bool pertokenFlag>
class MatmulAllReduceQuantBF16 : public MatmulAllReduceBase<xType, yType, coreType> {
public:
    __aicore__ inline MatmulAllReduceQuantBF16(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs,
                                               MC2TilingHeader *tilingData, TPipe *tPipe)
        : MatmulAllReduceBase<xType, yType, coreType>(addrs, quantAddrs, arnAddrs, tilingData, tPipe)
    {
        mc2TilingData_ = (Mc2Tiling::QuantMatmulAllReduceTilingData *)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tilematmulTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailmatmulTiling.matmulTiling;
    }

    __aicore__ inline void Process(mmType &opTile, mmType &opTail)
    {
        InnerProcess(opTile, false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            InnerProcess(opTail, true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(mmType &mmOp, bool tailFlag, uint32_t tileCnt, const MC2TileInfo &tileInfo)
    {
        const Mc2QuantBatchMatmulV3TilingData *tiling =
            (tailFlag ? &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);
        const uint64_t pertokenOffset = sizeof(float) * tiling->matmulTiling.M;
        for (uint32_t i = 0U; i < tileCnt; ++i) {
            if (this->addFlag_ || i == 0U) {
                this->tPipe_->Reset();
                if constexpr (pertokenFlag) {
                    mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->dequantGM,
                              this->quantAddrs_->pertokenGM, this->addrs_->cGM, this->addrs_->workspaceGM, tiling,
                              this->tPipe_);
                } else {
                    mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->dequantGM,
                              this->addrs_->cGM, this->addrs_->workspaceGM, tiling, this->tPipe_);
                }
            } else {
                if constexpr (pertokenFlag) {
                    mmOp.UpdateGlobalAddr(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM,
                                          this->quantAddrs_->dequantGM, this->quantAddrs_->pertokenGM,
                                          this->addrs_->cGM, this->addrs_->workspaceGM);
                } else {
                    mmOp.UpdateGlobalAddr(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM,
                                          this->quantAddrs_->dequantGM, this->addrs_->cGM, this->addrs_->workspaceGM);
                }
            }
            mmOp.Process();
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset);
            this->quantAddrs_->pertokenGM += pertokenOffset;
        }
    }

private:
    Mc2Tiling::QuantMatmulAllReduceTilingData *mc2TilingData_;
};

#define REG_MM_OBJ(opTile, opTail) \
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm, &(tilingData.tilematmulTiling.matmulTiling), opTail.mm, \
                      &(tilingData.tailmatmulTiling.matmulTiling))

#define REG_NO_MM_OBJ(opTile, opTail)

#define INVOKE_MC2_QUANT_910_OP_IMPL(templateClass, coreType, regObjCb, pertokenFlag, ...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingData, tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, dequantGM, pertokenGM}; \
        using opType = templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, __VA_ARGS__>; \
        opType opTile, opTail; \
        regObjCb(opTile, opTail); \
        MatmulAllReduceQuantBF16<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType, coreType, pertokenFlag> op( \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(opTile, opTail); \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_H
