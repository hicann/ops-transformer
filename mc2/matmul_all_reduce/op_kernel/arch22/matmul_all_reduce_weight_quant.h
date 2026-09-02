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
 * \file matmul_all_reduce_weight_quant.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_WEIGHT_QUANT_H
#define MATMUL_ALL_REDUCE_WEIGHT_QUANT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"
#if (FORMAT_X2 == FORMAT_FRACTAL_NZ)
#include "../../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom_weight_nz.h"
#define WEIGH_QUANT_MATMUL_CLASS_NAME Mc2WeightQuantBatchMatmulV2::Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel
#else
#include "../../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom.h"
#define WEIGH_QUANT_MATMUL_CLASS_NAME Mc2WeightQuantBatchMatmulV2::Mc2WeightQuantBatchMatmulV2CustomKernel
#endif
#include "matmul_all_reduce_base.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using Mc2WeightQuantBatchMatmulV2::Mc2QuantType;
template <typename xType, typename wType, typename yType, class mmType>
class MatmulAllReduceWeightQuant : public MatmulAllReduceBase<xType, yType, Mc2CoreType::ON_CUBE_AND_VECTOR> {
public:
    __aicore__ inline MatmulAllReduceWeightQuant(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs,
                                                 MC2TilingHeader *tilingData, TPipe *tPipe)
        : MatmulAllReduceBase<xType, yType, Mc2CoreType::ON_CUBE_AND_VECTOR>(addrs, quantAddrs, arnAddrs, tilingData,
                                                                             tPipe)
    {
        mc2TilingData_ = (Mc2Tiling::WeightQuantMatmulAllReduceTilingData *)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tilematmulTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailmatmulTiling.matmulTiling;
    }

    __aicore__ inline void Process()
    {
#if (ORIG_DTYPE_X1 == DT_BF16)
        this->PreProcForBiasOnVector();
#endif
        InnerProcess(false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            InnerProcess(true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(bool tailFlag, uint32_t tileCnt, const MC2TileInfo &tileInfo)
    {
        const Mc2WeightQuantBatchMatmulV2TilingData *tiling =
            (tailFlag ? &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);
        for (uint32_t i = 0U; i < tileCnt; ++i) {
            if (this->addFlag_ || i == 0U) {
                this->tPipe_->Reset();
                mmOp_.Init(this->addrs_->aGM, this->addrs_->bGM, this->quantAddrs_->antiquantScaleGM,
                           this->quantAddrs_->antiquantOffsetGM, nullptr, nullptr, this->addrs_->biasGM,
                           this->addrs_->cGM, this->addrs_->workspaceGM, tiling, this->tPipe_);
            } else {
                mmOp_.UpdateGlobalAddr(this->addrs_->aGM, this->addrs_->bGM, this->quantAddrs_->antiquantScaleGM,
                                       this->quantAddrs_->antiquantOffsetGM, nullptr, nullptr, this->addrs_->biasGM,
                                       this->addrs_->cGM, this->addrs_->workspaceGM);
            }
            mmOp_.Process();
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset);
        }
    }

private:
    Mc2Tiling::WeightQuantMatmulAllReduceTilingData *mc2TilingData_;
    mmType mmOp_;
};

#define INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(bTransFlag, quantType, offsetFlag) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::WeightQuantMatmulAllReduceTilingData, tilingData, tilingGM); \
        using opType = WEIGH_QUANT_MATMUL_CLASS_NAME<DTYPE_X1, DTYPE_X2, DTYPE_BIAS_FOR_MC2, DTYPE_Y, false, \
                                                     bTransFlag, quantType, offsetFlag, Mc2QuantType::NONE>; \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        QuantGmAddrs quantAddrs = {antiquantScaleGM, antiquantOffsetGM, nullptr, nullptr}; \
        MatmulAllReduceWeightQuant<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType> op(&addrs, &quantAddrs, nullptr, \
                                                                           (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(); \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_WEIGHT_QUANT_H
