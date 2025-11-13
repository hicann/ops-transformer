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
 * \file matmul_all_reduce_910_general.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_910_GENERAL_H
#define MATMUL_ALL_REDUCE_910_GENERAL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "common.h"
#include "matmul_all_reduce_base.h"

#if __has_include("../../3rd/mat_mul_v3/op_kernel/mat_mul_base_kernel.h")
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_base_kernel.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_unaligned_base_kernel.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#else
#include "../3rd/mat_mul_v3/op_kernel/mat_mul_base_kernel.h"
#include "../3rd/mat_mul_v3/op_kernel/mat_mul_unaligned_base_kernel.h"
#include "../3rd/mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#endif

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename xType, typename wType, typename yType, class mmType, Mc2CoreType coreType>
class MatmulAllReduce910General : public MatmulAllReduceBase<xType, yType, coreType>
{
public:
    __aicore__ inline MatmulAllReduce910General(
        MC2GmAddrs* addrs, ArnGmAddrs* arnAddrs, MC2TilingHeader* tilingData, TPipe* tPipe)
        : MatmulAllReduceBase<xType, yType, coreType>(addrs, nullptr, arnAddrs, tilingData, tPipe)
    {
        mc2TilingData_ = (MatmulAllReduce910TilingData*)tilingData;
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
    __aicore__ inline void InnerProcess(bool tailFlag, uint32_t tileCnt, const MC2TileInfo& tileInfo)
    {
        const MatmulTilingData* tiling =
            (tailFlag ? &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);

        mmType mmOp;
        for (uint32_t i = 0U; i < tileCnt; ++i) {
            if (block_idx < tiling->matmulTiling.usedCoreNum) {
                if (this->addFlag_ || i == 0U) {
                    this->tPipe_->Reset();
                    mmOp.Init(
                        this->addrs_->aGM, this->addrs_->bGM, this->addrs_->cGM, this->addrs_->biasGM, nullptr,
                        this->addrs_->workspaceGM, tiling, this->tPipe_);
                } else {
                    mmOp.UpdateGlobalTensor(
                        this->addrs_->aGM, this->addrs_->bGM, this->addrs_->cGM, this->addrs_->biasGM, nullptr,
                        this->addrs_->workspaceGM);
                }
                mmOp.Process();
            }
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset);
        }
    }

private:
    MatmulAllReduce910TilingData* mc2TilingData_;
};

#define INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, bTransFlag, coreType)                               \
    do {                                                                                                   \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;                 \
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, bTransFlag>;            \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS_FOR_MC2>;           \
        using opType = opTemplateClass<aType, bType, cType, biasType, MatmulBaseBlock, MM_CFG_NO_PRELOAD>; \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                               \
        MatmulAllReduce910General<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType, coreType> op(                       \
            &addrs, nullptr, (MC2TilingHeader*)&tilingData, &tPipe);                                       \
        op.Init();                                                                                         \
        op.Process();                                                                                      \
    } while (0)

#define INVOKE_MC2_910_OP_IMPL(opTemplateClass, coreType)                                \
    do {                                                                                 \
        GET_TILING_DATA_WITH_STRUCT(MatmulAllReduce910TilingData, tilingData, tilingGM); \
        if (tilingData.tilematmulTiling.matmulRunInfo.transB != 0U) {                    \
            INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, true, coreType);              \
        } else {                                                                         \
            INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, false, coreType);             \
        }                                                                                \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_910_GENERAL_H