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
 * \file matmul_all_reduce_weight_quant_adaptive_split.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_WEIGHT_QUANT_ADAPTIVE_SPLIT_H
#define MATMUL_ALL_REDUCE_WEIGHT_QUANT_ADAPTIVE_SPLIT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"
#include "../../../3rd/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_constant.h"
#include "../../../3rd/weight_quant_batch_matmul_v2/op_kernel/arch35/n_first/weight_quant_batch_matmul_v2_basic_block_controller.h"
#include "matmul_all_reduce_based_a2a_rs_ag.h"
#include "matmul_all_reduce_based_all_reduce.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
using Mc2WeightQuantBatchMatmulV2::Mc2QuantType;
template <typename XType, typename WType, typename YType, class MmType, bool basedA2aRsAg, int commMode>
class MatmulAllReduceWeightQuantAdaptiveSplit
    : public MatmulAllReduceBase<XType, YType, Mc2CoreType::ON_CUBE_AND_VECTOR, basedA2aRsAg, commMode> {
public:
    __aicore__ inline MatmulAllReduceWeightQuantAdaptiveSplit(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs,
                                                              ArnGmAddrs *arnAddrs, MC2TilingHeader *tilingData,
                                                              TPipe *tPipe)
        : MatmulAllReduceBase<XType, YType, Mc2CoreType::ON_CUBE_AND_VECTOR, basedA2aRsAg, commMode>(
              addrs, quantAddrs, arnAddrs, tilingData, tPipe)
    {
        mc2TilingData_ = (Mc2Tiling::WeightQuantMatmulAllReduceA5Fp8TilingData *)tilingData;
        this->tailInfo_.mmTiling = &mc2TilingData_->tailMmASTiling.matmulTiling;
        this->tileInfo_.mmTiling = &mc2TilingData_->tileMmASTiling.matmulTiling;
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
    __aicore__ inline void InnerProcess(const bool tailFlag, const uint32_t turnCnt, const MC2TileInfo &tileInfo)
    {
        const Mc2WeightQuantBatchMatmulV2ASTilingData *tiling =
            (tailFlag) ? &mc2TilingData_->tailMmASTiling : &mc2TilingData_->tileMmASTiling;
        for (uint32_t i = 0; i < turnCnt; ++i) {
            MmType mmOp;
            this->tPipe_->Reset();
            mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->quantAddrs_->antiquantScaleGM,
                      this->quantAddrs_->antiquantOffsetGM, nullptr, nullptr, this->addrs_->biasGM, this->addrs_->cGM,
                      this->addrs_->workspaceGM, tiling, this->tPipe_);
            mmOp.Process();
            const uint64_t index = tailFlag ? i + this->paramInTiling_->tileCnt : i;
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset, index);
        }
        if constexpr (basedA2aRsAg) {
            this->WaitAlltoAllEachTurn(tailFlag, turnCnt);
        }
    }

private:
    Mc2Tiling::WeightQuantMatmulAllReduceA5Fp8TilingData *mc2TilingData_;
};

#define INVOKE_MC2_WEIGHT_QUANT_ADAPTIVE_SPLIT_KERNEL(bTransFlag, offsetFlag, quantType, biasType, vecAntiQuantConfig, \
                                                      basedA2aRsAg, commMode) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::WeightQuantMatmulAllReduceA5Fp8TilingData, tilingData, tilingGM); \
        static constexpr Mc2WeightQuantBatchMatmulV2::Arch35::WqmmConfig wqmmCfg = { \
            false, bTransFlag, quantType, offsetFlag, Mc2QuantType::NONE, CubeFormat::ND}; \
        using OpType = Mc2WeightQuantBatchMatmulV2::Arch35::WeightQuantBatchMatmulV2BasicBlockController< \
            DTYPE_X1, DTYPE_X2, DTYPE_X1, biasType, DTYPE_Y, wqmmCfg, vecAntiQuantConfig>; \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
\
        QuantGmAddrs quantAddrs = {antiquantScaleGM, antiquantOffsetGM, nullptr, nullptr}; \
        MatmulAllReduceWeightQuantAdaptiveSplit<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, basedA2aRsAg, commMode> op( \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(); \
    } while (0)

} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_WEIGHT_QUANT_ADAPTIVE_SPLIT_H
