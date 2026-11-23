/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BatchMatMulV3 AIC-only base kernel from ops-nn, for the full-K ND path.
#ifndef TGRG_ARCH22_BATCH_MAT_MUL_V3_COMMON_KERNEL_H
#define TGRG_ARCH22_BATCH_MAT_MUL_V3_COMMON_KERNEL_H

#include "tgrg_batch_mat_mul_v3_com_base_block.h"

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = BatchMatMulCommonBaseBlock,
          const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr> >
class BatchMatMulCommonKernel {
public:
    __aicore__ inline BatchMatMulCommonKernel() {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void* tilingData, TPipe* pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);

    __aicore__ inline void Process(bool accumulate = false);

    __aicore__ inline void InnerProcess(uint64_t mTileIndex, uint64_t nTileIndex, bool accumulate);

    __aicore__ inline void End()
    {
        mm_.End();
    }

protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe* pipe_;
    TBuf<> ubBuf_;

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
    __aicore__ inline void SetOrgShape();
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void* tilingData, TPipe* pipe)
{
    // cube core cases, ignore vector core
    if ASCEND_IS_AIV {
        return;
    }
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);

    if (GetBlockIdx() >= block_.batchMatmulTilingData_->multiBatchInfo.batchUsedCoreNum) {
        return;
    }
    SetAtomicNone();
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM);

    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.batchMatmulTilingData_->matmulTiling.matmulTiling, pipe_);
    mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    LocalTensor<uint8_t> buf = ubBuf_.template Get<uint8_t>();
    mm_.SetLocalWorkspace(buf);
#endif
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                               MM_CB>::InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM),
                             block_.params_.aBatchDimAll *
                                 static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM),
                             block_.params_.bBatchDimAll *
                                 static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Kb) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(cGM),
                             block_.params_.cBatchDimAll *
                                 static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);

    if (block_.batchMatmulTilingData_->matmulTiling.matmulTiling.isBias) {
        auto biasSize = block_.params_.biasWithBatch ?
                            block_.params_.cBatchDimAll *
                                static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N) :
                            block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N;
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGM), biasSize);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SetOrgShape()
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    // Set the aligned shape after ND-to-NZ conversion.
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.params_.alignedOriN, block_.params_.alignedKaSize,
                        block_.params_.alignedKbSize, block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N,
                        block_.params_.alignedKaSize, block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Kb,
                        block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M, block_.params_.alignedOriN,
                        block_.batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreK,
                        block_.params_.alignedKbSize, block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    }
#endif
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                               MM_CB>::UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                                          GM_ADDR biasGM, GM_ADDR offsetWGM,
                                                                          GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void
BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InnerProcess(uint64_t mTileIndex,
                                                                                                    uint64_t nTileIndex,
                                                                                                    bool accumulate)
{
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        if (block_.params_.rowOrder == 0) {
            block_.UpdateBasicIndex(j, mTileIndex, nTileIndex); // Update the index for staggered core assignment.
        }
        if (block_.params_.index < block_.params_.totalTileCnt) {
            block_.UpdateBlockParams(mTileIndex, nTileIndex);
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
            mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                               block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Ka);
            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
            if (block_.batchMatmulTilingData_->matmulTiling.matmulTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
            mm_.IterateAll(cGlobal_[block_.offset_.offsetC], accumulate);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
            auto eventMTE3MTE2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
#endif
        }
        // Advance the tile index in row- or column-major order.
        block_.UpdateBlockIndex();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB>
__aicore__ inline void BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Process(
    bool accumulate)
{
    // cube core cases, ignore vector core
    if ASCEND_IS_AIV {
        return;
    }
    if (GetBlockIdx() >= block_.batchMatmulTilingData_->multiBatchInfo.batchUsedCoreNum) {
        return;
    }

    // The ND base path has no MatMul L1 full-load callbacks.
    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }
    bool reverse = true;
    for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            block_.InitBlockIndex();
            InnerProcess(mTileIndex, nTileIndex, accumulate);
        }
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm_.SetHF32(false, 0);
    return;
}

#endif
