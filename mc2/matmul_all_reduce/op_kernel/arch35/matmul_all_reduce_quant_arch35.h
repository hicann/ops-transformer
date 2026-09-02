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
 * \file matmul_all_reduce_quant_arch35.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_QUANT_ARCH35_H
#define MATMUL_ALL_REDUCE_QUANT_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "matmul_all_reduce_based_a2a_rs_ag.h"
#include "matmul_all_reduce_based_all_reduce.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, bool basedA2aRsAg,
          int commMode>
class MatmulAllReduceQuant : public MatmulAllReduceBase<XType, YType, CoreType, basedA2aRsAg, commMode> {
public:
    __aicore__ inline MatmulAllReduceQuant(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs,
                                           MC2TilingHeader *tilingData, TPipe *tPipe, bool isMX)
        : MatmulAllReduceBase<XType, YType, CoreType, basedA2aRsAg, commMode>(addrs, quantAddrs, arnAddrs, tilingData,
                                                                              tPipe)
    {
        mc2TilingData_ = (Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *)tilingData;
        this->tileInfo_.mmTiling = &mc2TilingData_->tilematmulTiling.matmulTiling;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailmatmulTiling.matmulTiling;
        isMXScene_ = isMX;
    }

    __aicore__ inline void Process()
    {
        MmType opTile;
        InnerProcess(opTile, false, this->paramInTiling_->tileCnt, this->tileInfo_);
        if (this->tailFlag_) {
            MmType opTail;
            InnerProcess(opTail, true, this->paramInTiling_->tailCnt, this->tailInfo_);
        }
        if constexpr (basedA2aRsAg) {
            this->ReduceSumAndAllGather();
        }
        this->HcclFinalize();
    }

protected:
    __aicore__ inline void InnerProcess(MmType &mmOp, bool tailFlag, uint32_t turnCnt, const MC2TileInfo &tileInfo)
    {
        const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tiling =
            (tailFlag ? &mc2TilingData_->tailmatmulTiling : &mc2TilingData_->tilematmulTiling);
        uint64_t pertokenOffset = 0UL;
        uint32_t MX_GROUP_SIZE = 64U;
        uint64_t NUM_TWO = 2;
        if (isMXScene_) {
            uint64_t matmulKa = static_cast<uint64_t>(tiling->matmulTiling.Ka);
            pertokenOffset =
                sizeof(AscendC::fp8_e8m0_t) * tiling->matmulTiling.M * CeilDiv(matmulKa, MX_GROUP_SIZE) * NUM_TWO;
        } else {
            pertokenOffset = sizeof(float) * tiling->matmulTiling.M;
        }
        for (uint32_t i = 0U; i < turnCnt; ++i) {
            this->tPipe_->Reset();
            // 当前MM的UpdateGlobalAddr接口存在问题，暂时每轮计算均使用Init接口更新地址
            mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->dequantGM,
                      this->quantAddrs_->pertokenGM, this->addrs_->cGM, this->addrs_->workspaceGM, tiling,
                      this->tPipe_);

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
    bool isMXScene_ = false;
};

#define INVOKE_MC2_QUANT_910_OP_IMPL(templateClass, coreType, basedA2aRsAg, commMode, scaleType, ...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM}; \
        using OpType = templateClass<DTYPE_X1, DTYPE_X2, scaleType, DTYPE_BIAS, DTYPE_Y, X1_FORMAT, X2_FORMAT, \
                                     Y_FORMAT, __VA_ARGS__>; \
        MatmulAllReduceQuant<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType, basedA2aRsAg, commMode> op( \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe, false); \
        op.Init(); \
        op.Process(); \
    } while (0)

#define INVOKE_MC2_QUANT_MXFP_910_OP_IMPL(templateClass, coreType, basedA2aRsAg, commMode, ...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM}; \
        using OpType = templateClass<DTYPE_X1, DTYPE_X2, AscendC::fp8_e8m0_t, DTYPE_BIAS, DTYPE_Y, X1_FORMAT, \
                                     X2_FORMAT, Y_FORMAT, __VA_ARGS__>; \
        MatmulAllReduceQuant<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType, basedA2aRsAg, commMode> op( \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe, true); \
        op.Init(); \
        op.Process(); \
    } while (0)

} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_H
