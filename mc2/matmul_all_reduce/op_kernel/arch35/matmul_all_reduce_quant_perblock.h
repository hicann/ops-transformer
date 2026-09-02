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
 * \file matmul_all_reduce_quant_perblock.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_QUANT_PERBLOCK_H
#define MATMUL_ALL_REDUCE_QUANT_PERBLOCK_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "matmul_all_reduce_based_a2a_rs_ag.h"
#include "matmul_all_reduce_based_all_reduce.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_perblock.h"

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename XType, typename WType, typename YType, class MmType, Mc2CoreType CoreType, bool basedA2aRsAg,
          int commMode>
class MatmulAllReduceQuantPerBlock : public MatmulAllReduceBase<XType, YType, CoreType, basedA2aRsAg, commMode> {
public:
    __aicore__ inline MatmulAllReduceQuantPerBlock(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs, ArnGmAddrs *arnAddrs,
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
        // CeilDiv
        const uint64_t mOfscale = (tiling->matmulTiling.M + 128 - 1) / 128;
        const uint64_t kOfScale = (tiling->matmulTiling.Ka + 128 - 1) / 128;
        const uint64_t x1ScaleOffset = sizeof(float) * mOfscale * kOfScale;
        for (uint32_t i = 0U; i < turnCnt; ++i) {
            // mm自己实现的perblock模板，存在eventId互锁问题，需要彻底释放
            this->tPipe_->Destroy();
            this->tPipe_->Init();
            // 当前MM的UpdateGlobalAddr接口存在问题，暂时每轮计算均使用Init接口更新地址
            mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->dequantGM,
                      this->quantAddrs_->pertokenGM, this->addrs_->cGM, this->addrs_->workspaceGM, tiling,
                      this->tPipe_);

            mmOp.Process();
            const uint64_t index = tailFlag ? i + this->paramInTiling_->tileCnt : i;
            this->PostProcEachTurn(tileInfo.hcclHandleId, tileInfo.aAddrOffset, tileInfo.cAddrOffset, index);
            this->quantAddrs_->pertokenGM += x1ScaleOffset;
        }
        if constexpr (basedA2aRsAg) {
            this->WaitAlltoAllEachTurn(tailFlag, turnCnt);
        }
    }

private:
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *mc2TilingData_;
};

#define INVOKE_MC2_QUANT_PERBLOCK_910_OP_IMPL(templateClass, coreType, basedA2aRsAg, commMode, ...) \
    do { \
        REGISTER_TILING_DEFAULT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5); \
        GET_TILING_DATA(tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM}; \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM}; \
        using OpType = \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, X1_FORMAT, X2_FORMAT, Y_FORMAT, __VA_ARGS__>; \
        MatmulAllReduceQuantPerBlock<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType, basedA2aRsAg, commMode> op( \
            &addrs, &quantAddrs, nullptr, (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(); \
    } while (0)

} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_PERBLOCK_H
