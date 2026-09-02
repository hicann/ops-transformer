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
 * \file mm_allreduce_add_rms_norm_weight_quant.h
 * \brief
 */
#ifndef MM_ALLREDUCE_ADD_RMS_NORM_WEIGHT_QUANT_H
#define MM_ALLREDUCE_ADD_RMS_NORM_WEIGHT_QUANT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../../matmul_all_reduce/op_kernel/matmul_all_reduce_common.h"
#include "../../matmul_all_reduce/op_kernel/arch22/matmul_all_reduce_weight_quant.h"
#include "add_rms_norm_kernel.h"
#include "matmul_all_reduce_add_rms_norm_tiling_data.h"

namespace MatmulAllReduceAddRmsNormImpl {
using namespace AscendC;
using MatmulAllReduceImpl::MatmulAllReduceWeightQuant;
using Mc2WeightQuantBatchMatmulV2::Mc2QuantType;
template <typename xType, typename wType, typename yType, class mmType>
class MatmulAllReduceAddRmsNormWeightQuant : public MatmulAllReduceWeightQuant<xType, wType, yType, mmType> {
public:
    __aicore__ inline MatmulAllReduceAddRmsNormWeightQuant(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs,
                                                           ArnGmAddrs *arnAddrs, MC2TilingHeader *tilingData,
                                                           TPipe *tPipe)
        : MatmulAllReduceWeightQuant<xType, wType, yType, mmType>(addrs, quantAddrs, arnAddrs, tilingData, tPipe)
    {
        Mc2Tiling::WeightQuantMatmulAllReduceAddRmsNormTilingData *p =
            (Mc2Tiling::WeightQuantMatmulAllReduceAddRmsNormTilingData *)tilingData;
        arnTile_ = &p->addRMSNormTileTilingData;
        arnTail_ = &p->addRMSNormTailTilingData;
        arnTilineKey_ = &p->addRmsNormTilingeKeyData;
    }

    __aicore__ inline void Process()
    {
#if (ORIG_DTYPE_X1 == DT_BF16)
        this->PreProcForBiasOnVector();
#endif

        this->InnerProcess(false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            this->InnerProcess(true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        SyncAll<false>();
        if ASCEND_IS_AIV {
            AddRmsNormKernel op(this->arnAddrs_, this->tPipe_, sizeof(yType), this->paramInTiling_->tileCnt,
                                this->paramInTiling_->tailCnt, &this->hccl_, this->tileInfo_.hcclHandleId,
                                this->tailInfo_.hcclHandleId);
            op.ComputeAddRmsNorm(*arnTile_, *arnTail_, *arnTilineKey_, this->addrs_->workspaceGM);
        }

        SyncAll<false>();
        if (this->notifyFlag_) {
            this->hccl_.Finalize();
        }
    }

private:
    Mc2Tiling::AddRMSNormTilingeKeyData *arnTilineKey_;
    Mc2Tiling::AddRMSNormTilingData *arnTile_;
    Mc2Tiling::AddRMSNormTilingData *arnTail_;
};

#define INVOKE_MC2_ARN_WEIGHT_QUANT_910_OP_IMPL(bTransFlag, quantType, offsetFlag) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::WeightQuantMatmulAllReduceAddRmsNormTilingData, tilingData, tilingGM); \
        using opType = WEIGH_QUANT_MATMUL_CLASS_NAME<DTYPE_X1, DTYPE_X2, DTYPE_BIAS_FOR_MC2, DTYPE_Y, false, \
                                                     bTransFlag, quantType, offsetFlag, Mc2QuantType::NONE>; \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, nullptr, normOutGM, workspaceGM, normOutGM}; \
        QuantGmAddrs quantAddrs = {antiquantScaleGM, antiquantOffsetGM, nullptr, nullptr}; \
        ArnGmAddrs arnAddrs = {residualGM, gammaGM, yGM, normOutGM}; \
        MatmulAllReduceAddRmsNormWeightQuant<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType> op( \
            &addrs, &quantAddrs, &arnAddrs, (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(); \
    } while (0)
} // namespace MatmulAllReduceAddRmsNormImpl
#endif // MM_ALLREDUCE_ADD_RMS_NORM_WEIGHT_QUANT_H
