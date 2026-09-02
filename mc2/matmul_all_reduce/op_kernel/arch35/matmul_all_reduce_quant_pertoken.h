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
 * \file matmul_all_reduce_quant_pertoken.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_QUANT_PERTOKEN_H
#define MATMUL_ALL_REDUCE_QUANT_PERTOKEN_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "matmul_all_reduce_based_a2a_rs_ag.h"
#include "matmul_all_reduce_based_all_reduce.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h"
#include "matmul_all_reduce_add_x3_arch35.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, bool basedA2aRsAg,
          int commMode>
class MatmulAllReduceQuantPerToken : public MatmulAllReduceBase<XType, YType, CoreType, basedA2aRsAg, commMode> {
public:
    __aicore__ inline MatmulAllReduceQuantPerToken(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs,
                                                   MC2TilingHeader *tilingData, TPipe *tPipe)
        : MatmulAllReduceBase<XType, YType, CoreType, basedA2aRsAg, commMode>(addrs, quantAddrs, arnAddrs, tilingData,
                                                                              tPipe)
    {
        mc2TilingData_ = (Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tilematmulTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailmatmulTiling.matmulTiling;
    }

    __aicore__ inline void Process()
    {
        InnerProcess(false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            InnerProcess(true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }
        if constexpr (basedA2aRsAg) {
            this->ReduceSumAndAllGather();
        }
        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(bool tailFlag, uint32_t turnCnt, const MC2TileInfo &tileInfo)
    {
        MmType mmOp;
        const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tiling =
            (tailFlag ? &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);
        const uint64_t pertokenOffset = sizeof(float) * tiling->matmulTiling.M;
        for (uint32_t i = 0U; i < turnCnt; ++i) {
            if (this->addFlag_ || (i == 0U)) {
                this->tPipe_->Destroy();
                this->tPipe_->Init();
                mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->quantAddrs_->dequantGM,
                          this->quantAddrs_->offsetGM, this->addrs_->biasGM, this->quantAddrs_->pertokenGM,
                          this->addrs_->cGM, this->addrs_->workspaceGM, tiling, this->tPipe_);
            } else {
                mmOp.UpdateGlobalAddr(this->addrs_->aGM, this->addrs_->bGM, this->quantAddrs_->dequantGM,
                                      this->addrs_->biasGM, this->quantAddrs_->pertokenGM, this->addrs_->cGM,
                                      this->addrs_->workspaceGM);
            }

            mmOp.Process();
            const uint64_t index = tailFlag ? i + this->paramInTiling_->tileCnt : i;
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset, index);
            this->quantAddrs_->pertokenGM += pertokenOffset;
        }
        if constexpr (basedA2aRsAg) {
            this->WaitAlltoAllEachTurn(tailFlag, turnCnt);
        }
    }

private:
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *mc2TilingData_;
};

#define INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_IMPL(templateClass, coreType, basedA2aRsAg, commMode, scaleType, isATrans, \
                                                isBTrans, ...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM}; \
        using OpType = templateClass<DTYPE_X1, DTYPE_X2, scaleType, DTYPE_BIAS, float, DTYPE_Y, X1_FORMAT, X2_FORMAT, \
                                     Y_FORMAT, isATrans, isBTrans, DTYPE_LOC_LOCAL, \
                                     Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>; \
        MatmulAllReduceQuantPerToken<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType, basedA2aRsAg, commMode> op( \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(); \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_PERTOKEN_H
