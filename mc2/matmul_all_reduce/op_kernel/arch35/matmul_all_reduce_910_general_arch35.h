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
 * \file matmul_all_reduce_910_general_arch35.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_910_GENERAL_ARCH35_H
#define MATMUL_ALL_REDUCE_910_GENERAL_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "matmul_all_reduce_based_a2a_rs_ag.h"
#include "matmul_all_reduce_based_all_reduce.h"
#include "../../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, bool basedA2aRsAg,
          int commMode>
class MatmulAllReduce910General : public MatmulAllReduceBase<XType, YType, CoreType, basedA2aRsAg, commMode> {
public:
    __aicore__ inline MatmulAllReduce910General(MC2GmAddrs *addrs, ArnGmAddrs *arnAddrs, MC2TilingHeader *tilingData,
                                                TPipe *tPipe)
        : MatmulAllReduceBase<XType, YType, CoreType, basedA2aRsAg, commMode>(addrs, nullptr, arnAddrs, tilingData,
                                                                              tPipe)
    {
        mc2TilingData_ = (Mc2Tiling::MatmulAllReduce910TilingDataA5 *)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->mC2Mmv3TileTilingData.tCubeTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->mC2Mmv3TailTilingData.tCubeTiling;
        // 从 tiling struct 读取 bias 相关字段传入基类
        this->biasUbCnt_ = mc2TilingData_->extTiling.biasUbCnt;
        this->hasBias_ = (mc2TilingData_->extTiling.isVecBias != 0U);
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
        const Mc2MatMulV3TilingData *tiling =
            (tailFlag ? &mc2TilingData_->mC2Mmv3TailTilingData : &mc2TilingData_->mC2Mmv3TileTilingData);

        MmType mmOp;
        for (uint32_t i = 0U; i < turnCnt; ++i) {
            if (block_idx < tiling->tCubeTiling.usedCoreNum) {
                this->tPipe_->Reset();
                mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->cGM, this->addrs_->biasGM, nullptr,
                          this->addrs_->workspaceGM, tiling, this->tPipe_);
                mmOp.Process();
                mmOp.End();
            }
            const uint64_t index = tailFlag ? i + this->paramInTiling_->tileCnt : i;
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset, index,
                                   (uint64_t)tileInfo.mmTiling->M);
        }
        if constexpr (basedA2aRsAg) {
            this->WaitAlltoAllEachTurn(tailFlag, turnCnt);
        }
    }

private:
    Mc2Tiling::MatmulAllReduce910TilingDataA5 *mc2TilingData_;
};

#define INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, bTransFlag, coreType, basedA2aRsAg, commMode) \
    do { \
        using AType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, false>; \
        using BType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, bTransFlag>; \
        using CType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>; \
        using BiasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>; \
        using OpType = \
            opTemplateClass<AType, BType, CType, BiasType, Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD>; \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        MatmulAllReduce910General<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType, basedA2aRsAg, commMode> op( \
            &addrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(); \
    } while (0)

#define INVOKE_MC2_910_OP_IMPL(opTemplateClass, coreType, basedA2aRsAg, commMode) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::MatmulAllReduce910TilingDataA5, tilingData, tilingGM); \
        if (tilingData.param.isTransposeB != 0U) { \
            INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, true, coreType, basedA2aRsAg, commMode); \
        } else { \
            INVOKE_MC2_910_OP_IMPL_HELPER(opTemplateClass, false, coreType, basedA2aRsAg, commMode); \
        } \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_910_GENERAL_H
