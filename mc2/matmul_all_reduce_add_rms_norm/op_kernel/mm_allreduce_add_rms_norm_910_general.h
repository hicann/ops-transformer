/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mm_allreduce_add_rms_norm_910_general.h
 * \brief
 */
#ifndef MM_ALLREDUCE_ADD_RMS_NORM_910_GENERAL_H
#define MM_ALLREDUCE_ADD_RMS_NORM_910_GENERAL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "add_rms_norm_kernel.h"

#if __has_include("../../matmul_all_reduce/op_kernel/common.h")
#include "../../matmul_all_reduce/op_kernel/common.h"
#include "../../matmul_all_reduce/op_kernel/matmul_all_reduce_910_general.h"
#else
#include "../matmul_all_reduce/common.h"
#include "../matmul_all_reduce/matmul_all_reduce_910_general.h"
#endif

namespace MatmulAllReduceAddRmsNormImpl {
using namespace AscendC;
using MatmulAllReduceImpl::MatmulAllReduce910General;
template <typename xType, typename wType, typename yType, class mmType>
class MatmulAllReduceAddRmsNorm910General
    : public MatmulAllReduce910General<xType, wType, yType, mmType, Mc2CoreType::ON_CUBE_AND_VECTOR>
{
public:
    __aicore__ inline MatmulAllReduceAddRmsNorm910General(
        MC2GmAddrs* addrs, ArnGmAddrs* arnAddrs, MC2TilingHeader* tilingData, TPipe* tPipe)
        : MatmulAllReduce910General<xType, wType, yType, mmType, Mc2CoreType::ON_CUBE_AND_VECTOR>(
              addrs, arnAddrs, tilingData, tPipe)
    {
        MatmulAllReduceAddRmsNormTilingData* p = (MatmulAllReduceAddRmsNormTilingData*)tilingData;
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
            AddRmsNormKernel op(
                this->arnAddrs_, this->tPipe_, sizeof(yType), this->paramInTiling_->tileCnt,
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
    AddRMSNormTilingeKeyData* arnTilineKey_;
    AddRMSNormTilingData* arnTile_;
    AddRMSNormTilingData* arnTail_;
};

#define INVOKE_MC2_ARN_910_OP_IMPL_HELPER(opTemplateClass, bTransFlag)                                     \
    do {                                                                                                   \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;                 \
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, bTransFlag>;            \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS_FOR_MC2>;           \
        using opType = opTemplateClass<aType, bType, cType, biasType, MatmulBaseBlock, MM_CFG_NO_PRELOAD>; \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, nullptr, normOutGM, workspaceGM, normOutGM};                 \
        ArnGmAddrs arnAddrs = {residualGM, gammaGM, yGM, normOutGM};                                       \
        MatmulAllReduceAddRmsNorm910General<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType> op(                       \
            &addrs, &arnAddrs, (MC2TilingHeader*)&tilingData, &tPipe);                                     \
        op.Init();                                                                                         \
        op.Process();                                                                                      \
    } while (0)

#define INVOKE_MC2_ARN_910_OP_IMPL(opTemplateClass)                                             \
    do {                                                                                        \
        GET_TILING_DATA_WITH_STRUCT(MatmulAllReduceAddRmsNormTilingData, tilingData, tilingGM); \
        if (tilingData.matmulAllReduceTilingData.tilematmulTiling.matmulRunInfo.transB != 0U) { \
            INVOKE_MC2_ARN_910_OP_IMPL_HELPER(opTemplateClass, true);                           \
        } else {                                                                                \
            INVOKE_MC2_ARN_910_OP_IMPL_HELPER(opTemplateClass, false);                          \
        }                                                                                       \
    } while (0)
} // namespace MatmulAllReduceAddRmsNormImpl
#endif // MM_ALLREDUCE_ADD_RMS_NORM_910_GENERAL_H