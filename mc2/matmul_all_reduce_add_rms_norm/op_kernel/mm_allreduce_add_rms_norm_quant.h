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
 * \file mm_allreduce_add_rms_norm_quant.h
 * \brief
 */
#ifndef MM_ALLREDUCE_ADD_RMS_NORM_QUANT_H
#define MM_ALLREDUCE_ADD_RMS_NORM_QUANT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../../matmul_all_reduce/op_kernel/matmul_all_reduce_common.h"
#include "../../matmul_all_reduce/op_kernel/arch22/matmul_all_reduce_quant.h"
#include "add_rms_norm_kernel.h"
#include "matmul_all_reduce_add_rms_norm_tiling_data.h"

namespace MatmulAllReduceAddRmsNormImpl {
using namespace AscendC;
using MatmulAllReduceImpl::MatmulAllReduceQuantBF16;
template <typename xType, typename wType, typename yType, class mmType, Mc2CoreType coreType>
class MatmulAllReduceAddRmsNormQuantBF16
    : public MatmulAllReduceQuantBF16<xType, wType, yType, mmType, coreType, false> {
public:
    __aicore__ inline MatmulAllReduceAddRmsNormQuantBF16(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs,
                                                         ArnGmAddrs *arnAddrs, MC2TilingHeader *tilingData,
                                                         TPipe *tPipe)
        : MatmulAllReduceQuantBF16<xType, wType, yType, mmType, coreType, false>(addrs, quantAddrs, arnAddrs,
                                                                                 tilingData, tPipe)
    {
        Mc2Tiling::QuantMatmulAllReduceAddRmsNormTilingData *p =
            (Mc2Tiling::QuantMatmulAllReduceAddRmsNormTilingData *)tilingData;
        arnTile_ = &p->addRMSNormTileTilingData;
        arnTail_ = &p->addRMSNormTailTilingData;
        arnTilineKey_ = &p->addRmsNormTilingeKeyData;
    }

    __aicore__ inline void Process(mmType &opTile, mmType &opTail)
    {
        this->InnerProcess(opTile, false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            this->InnerProcess(opTail, true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }

        Mc2SyncAll<coreType>();
        if ASCEND_IS_AIV {
            AddRmsNormKernel op(this->arnAddrs_, this->tPipe_, sizeof(yType), this->paramInTiling_->tileCnt,
                                this->paramInTiling_->tailCnt, &this->hccl_, this->tileInfo_.hcclHandleId,
                                this->tailInfo_.hcclHandleId);
            op.ComputeAddRmsNorm(*arnTile_, *arnTail_, *arnTilineKey_, this->addrs_->workspaceGM);
        }

        Mc2SyncAll<coreType>();
        if (this->notifyFlag_) {
            this->hccl_.Finalize();
        }
    }

private:
    Mc2Tiling::AddRMSNormTilingeKeyData *arnTilineKey_;
    Mc2Tiling::AddRMSNormTilingData *arnTile_;
    Mc2Tiling::AddRMSNormTilingData *arnTail_;
};

#define REG_MM_OBJ_FOR_ARN(opTile, opTail) \
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), opTile.mm, \
                      &(tilingData.quantMatmulAllReduceTilingData.tilematmulTiling.matmulTiling), opTail.mm, \
                      &(tilingData.quantMatmulAllReduceTilingData.tailmatmulTiling.matmulTiling))

#define INVOKE_MC2_ARN_QUANT_910_OP_IMPL(templateClass, coreType, regObjCb, ...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceAddRmsNormTilingData, tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, nullptr, normOutGM, workspaceGM, normOutGM}; \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, dequantGM, nullptr}; \
        ArnGmAddrs arnAddrs = {residualGM, gammaGM, yGM, normOutGM}; \
        using opType = templateClass<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, __VA_ARGS__>; \
        opType opTile, opTail; \
        regObjCb(opTile, opTail); \
        MatmulAllReduceAddRmsNormQuantBF16<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType, coreType> op( \
            &addrs, &quantAddrs, &arnAddrs, (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(opTile, opTail); \
    } while (0)
} // namespace MatmulAllReduceAddRmsNormImpl
#endif // MM_ALLREDUCE_ADD_RMS_NORM_QUANT_H
