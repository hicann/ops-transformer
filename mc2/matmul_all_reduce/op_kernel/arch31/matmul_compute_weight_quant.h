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
 * \file matmul_compute_weight_quant.h
 * \brief
 */

#ifndef MC2_MATMUL_COMPUTE_WEIGHT_QUANT_H
#define MC2_MATMUL_COMPUTE_WEIGHT_QUANT_H

#include "matmul_compute.h"
namespace AscendC {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache = false, bool WeightQuant = false,
          AntiQuantType antiQuantType = AntiQuantType::NONE, bool hasAntiQuantOffset = false>
class MatmulComputeWeightQuant : public MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant,
                                                      antiQuantType, hasAntiQuantOffset> {
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline MatmulComputeWeightQuant() {}
    __aicore__ inline void Init(AscendC::tiling::TCubeTiling &tiling, Mc2Tiling::RCSTiling &cfg,
                                Mc2Tiling::Mc2L2cacheTilePara &tileL2cacheTiling,
                                const LocalTensor<uint8_t> &mmFormatUb);
    __aicore__ inline void InitGlobalBTensor(GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR antiquantScale,
                                             GM_ADDR antiquantOffset);
    __aicore__ inline void Compute(uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithL2Cache(uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithL2CacheOdd(int32_t mTileIndex, uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithL2CacheEven(int32_t mTileIndex, uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithNorm(uint32_t index = 0, uint8_t enAtomic = 0);

private:
    __aicore__ inline void CopyInAntiQuantParam();

private:
    GlobalTensor<A_T> addGlobal_;
    GlobalTensor<A_T> mulGlobal_;
    LocalTensor<A_T> antiQuantScaleLocal_;
    LocalTensor<A_T> antiQuantOffsetLocal_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                         hasAntiQuantOffset>::Init(AscendC::tiling::TCubeTiling &tiling, Mc2Tiling::RCSTiling &cfg,
                                                   Mc2Tiling::Mc2L2cacheTilePara &tileL2cacheTiling,
                                                   const LocalTensor<uint8_t> &mmFormatUb)
{
    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType, hasAntiQuantOffset>::Init(
        tiling, cfg, tileL2cacheTiling, mmFormatUb);
    antiQuantScaleLocal_ = mmFormatUb[this->block.tiling.transLength - this->block.tiling.baseN * 2 - 32 * 2]
                               .template ReinterpretCast<A_T>();
    antiQuantOffsetLocal_ = mmFormatUb[this->block.tiling.transLength * 2 - this->block.tiling.baseN * 2 - 32 * 2]
                                .template ReinterpretCast<A_T>();
    // bankconflict pad size is 32 point
    antiQuantScaleLocal_.SetSize(this->block.tiling.baseN + 32);
    antiQuantOffsetLocal_.SetSize(this->block.tiling.baseN + 32);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant,
                                                antiQuantType, hasAntiQuantOffset>::CopyInAntiQuantParam()
{
    DataCopyParams intriParamsAdd;
    intriParamsAdd.blockCount = 1;
    intriParamsAdd.srcStride = 0;
    intriParamsAdd.dstStride = 0;
    if constexpr (antiQuantType == AntiQuantType::PER_TENSOR) {
        intriParamsAdd.blockLen = 1; // 拷贝一个Block
    } else if constexpr (antiQuantType == AntiQuantType::PER_CHANNEL) {
        intriParamsAdd.blockLen = this->block.tiling.baseN * sizeof(A_T) / 32;
    }
    int32_t paramOffset = this->block.offset.offsetBias;
    if constexpr (antiQuantType == AntiQuantType::PER_TENSOR) {
        paramOffset = 0;
    }

    if constexpr (hasAntiQuantOffset) {
        // 搬运offset
        DataCopy(antiQuantOffsetLocal_, addGlobal_[paramOffset], intriParamsAdd);
    }
    // 搬运scale
    DataCopy(antiQuantScaleLocal_, mulGlobal_[paramOffset], intriParamsAdd);
    if constexpr (antiQuantType == AntiQuantType::PER_TENSOR) {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID7);
        this->mm.SetAntiQuantScalar(antiQuantOffsetLocal_.GetValue(0), antiQuantScaleLocal_.GetValue(0));
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
    } else {
        this->mm.SetAntiQuantVector(antiQuantOffsetLocal_, antiQuantScaleLocal_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                         hasAntiQuantOffset>::InitGlobalBTensor(GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR antiquantScale,
                                                                GM_ADDR antiquantOffset)
{
    // MC2的计算流中默认B矩阵不变，GM地址无需偏移
    this->bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), this->block.tiling.Kb * this->block.tiling.N);
    this->biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), this->block.tiling.N);
    this->mulGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(antiquantScale), this->block.tiling.N);
    this->addGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(antiquantOffset), this->block.tiling.N);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant,
                                                antiQuantType, hasAntiQuantOffset>::Compute(uint32_t index,
                                                                                            uint8_t enAtomic)
{
    if constexpr (Mc2L2Cache) {
        ComputeWithL2Cache(index, enAtomic);
    } else {
        ComputeWithNorm(index, enAtomic);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant,
                                                antiQuantType, hasAntiQuantOffset>::ComputeWithNorm(uint32_t index,
                                                                                                    uint8_t enAtomic)
{
    // 每次block循环开始前需要计算初始blockIndex
    this->block.UpdateBlockCnt(index);
    for (uint32_t i = 0; i < this->block.blockCnt; i++) {
        if (this->block.blockIndex < this->block.totalBlockCnt) {
            this->block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            this->SetSingleCoreShape();
            this->mm.SetTensorA(this->aGlobal[this->block.offset.offsetA], this->block.isTransposeA);
            CopyInAntiQuantParam();
            this->mm.SetTensorB(this->bGlobal[this->block.offset.offsetB], this->block.isTransposeB);
            if (this->block.tiling.isBias) {
                this->mm.SetBias(this->biasGlobal[this->block.offset.offsetBias]);
            }
            this->mm.Iterate();
            this->mm.GetTensorC(this->cGlobal[this->block.offset.offsetC], enAtomic);
#if __CCE_AICORE__ == 220
#else
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
#endif
        }
        this->block.UpdateBlockIndex();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant,
                                                antiQuantType, hasAntiQuantOffset>::ComputeWithL2Cache(uint32_t index,
                                                                                                       uint8_t enAtomic)
{
    for (int32_t mTileIndex = 0; mTileIndex < this->block.tilingL2.mTileCntL2; mTileIndex++) {
        if (mTileIndex % 2 == 0) {
            ComputeWithL2CacheEven(mTileIndex, index, enAtomic);
        } else {
            ComputeWithL2CacheOdd(mTileIndex, index, enAtomic);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                         hasAntiQuantOffset>::ComputeWithL2CacheOdd(int32_t mTileIndex, uint32_t index,
                                                                    uint8_t enAtomic)
{
    for (int32_t nTileIndex = this->block.tilingL2.nTileCntL2 - 1; nTileIndex >= 0; nTileIndex--) {
        this->block.UpdateBlockCnt(0, mTileIndex, nTileIndex);
        for (uint32_t j = 0; j < this->block.blockCnt; j++) {
            if (index < this->block.totalTileCnt) {
                this->block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
                this->mm.SetSingleShape(this->block.singleCoreM, this->block.singleCoreN,
                                        this->block.tiling.singleCoreK);
                this->mm.SetTensorA(this->aGlobal[this->block.offset.offsetA], this->block.isTransposeA);
                CopyInAntiQuantParam();
                this->mm.SetTensorB(this->bGlobal[this->block.offset.offsetB], this->block.isTransposeB);
                if (this->block.tiling.isBias) {
                    this->mm.SetBias(this->biasGlobal[this->block.offset.offsetBias]);
                }
                this->mm.Iterate();
                this->mm.GetTensorC(this->cGlobal[this->block.offset.offsetC], enAtomic);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            }
            this->block.UpdateBlockIndex();
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulComputeWeightQuant<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                         hasAntiQuantOffset>::ComputeWithL2CacheEven(int32_t mTileIndex, uint32_t index,
                                                                     uint8_t enAtomic)
{
    for (int32_t nTileIdx = 0; nTileIdx < this->block.tilingL2.nTileCntL2; nTileIdx++) {
        this->block.UpdateBlockCnt(0, mTileIndex, nTileIdx);
        for (uint32_t j = 0; j < this->block.blockCnt; j++) {
            if (index < this->block.totalTileCnt) {
                this->block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIdx);
                this->mm.SetSingleShape(this->block.singleCoreM, this->block.singleCoreN,
                                        this->block.tiling.singleCoreK);
                this->mm.SetTensorA(this->aGlobal[this->block.offset.offsetA], this->block.isTransposeA);
                CopyInAntiQuantParam();
                this->mm.SetTensorB(this->bGlobal[this->block.offset.offsetB], this->block.isTransposeB);
                if (this->block.tiling.isBias) {
                    this->mm.SetBias(this->biasGlobal[this->block.offset.offsetBias]);
                }
                this->mm.Iterate();
                this->mm.GetTensorC(this->cGlobal[this->block.offset.offsetC], enAtomic);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            }
            this->block.UpdateBlockIndex();
        }
    }
}
} // namespace AscendC
#endif // MC2_MATMUL_COMPUTE_WEIGHT_QUANT_H
