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
 * \file matmul_compute.h
 * \brief
 */
#ifndef MC2_MATMUL_COMPUTE_H
#define MC2_MATMUL_COMPUTE_H

#include "../matmul_all_reduce_common.h"
#include "matmul_block_l2cache.h"
namespace AscendC {
enum class IsPerTensor {
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool hasAntiQuantOffset,
          AntiQuantType antiQuantType>
struct AllReduceMMType {
    __aicore__ inline AllReduceMMType(){};
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
struct AllReduceMMType<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, false, AntiQuantType::PER_TENSOR> {
    __aicore__ inline AllReduceMMType(){};
    static constexpr MatmulConfig CFG_MDL = GetMDLConfig(false, false, false, true, true, false);
    using PARAMS = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
struct AllReduceMMType<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, false, AntiQuantType::PER_CHANNEL> {
    __aicore__ inline AllReduceMMType(){};
    static constexpr MatmulConfig CFG_MDL = GetMDLConfig(false, false, false, true, false, false);
    using PARAMS = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
struct AllReduceMMType<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, true, AntiQuantType::PER_TENSOR> {
    __aicore__ inline AllReduceMMType(){};
    static constexpr MatmulConfig CFG_MDL = GetMDLConfig(false, false, false, true, true, true);
    using PARAMS = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
struct AllReduceMMType<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, true, AntiQuantType::PER_CHANNEL> {
    __aicore__ inline AllReduceMMType(){};
    static constexpr MatmulConfig CFG_MDL = GetMDLConfig(false, false, false, true, false, true);
    using PARAMS = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
struct AllReduceMMType<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, false, AntiQuantType::NONE> {
    __aicore__ inline AllReduceMMType(){};
    static constexpr MatmulConfig CFG_MDL = GetMDLConfig(false, false, false, true);
    using PARAMS = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache = false, bool WeightQuant = false,
          AntiQuantType antiQuantType = AntiQuantType::NONE, bool hasAntiQuantOffset = false>
class MatmulCompute {
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline MatmulCompute() {}
    __aicore__ inline void Init(AscendC::tiling::TCubeTiling &tiling, Mc2Tiling::RCSTiling &cfg,
                                Mc2Tiling::Mc2L2cacheTilePara &tileL2cacheTiling,
                                const LocalTensor<uint8_t> &mmFormatUb);
    __aicore__ inline void InitGlobalBTensor(GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR antiquantScale,
                                             GM_ADDR antiquantOffset);
    __aicore__ inline void InitGlobalATensor(GM_ADDR aGM, uint32_t aSize, GM_ADDR cGM, uint32_t cSize);
    __aicore__ inline void InitDequantScale(GM_ADDR dequantGM);
    __aicore__ inline void InitDequantScale(uint64_t dequantGM);
    __aicore__ inline void Compute(uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithL2Cache(uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void MatmulComputeWithL2Cache(int32_t mTileIndex, int32_t nTileIndex, uint32_t index = 0,
                                                    uint8_t enAtomic = 0);
    __aicore__ inline void ComputeWithNorm(uint32_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void End();

    __aicore__ inline void SetOrgShapeAlign();
    __aicore__ inline void SetSingleCoreShape();

    typename BlockType<Mc2L2Cache>::PARAMS block;

    typename AllReduceMMType<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, hasAntiQuantOffset, antiQuantType>::PARAMS mm;
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<uint64_t> dequantGlobal;
    GlobalTensor<BiasT> biasGlobal;
    uint64_t dequantScalar;
    uint32_t isPerTensor = 0; // 为0不做dequant
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::InitGlobalBTensor(GM_ADDR bGM, GM_ADDR biasGM,
                                                                            GM_ADDR antiquantScale,
                                                                            GM_ADDR antiquantOffset)
{
    (void)antiquantScale;
    (void)antiquantOffset;
    // MC2的计算流中默认B矩阵不变，GM地址无需偏移
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), block.tiling.Kb * block.tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), block.tiling.N);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::InitGlobalATensor(GM_ADDR aGM, uint32_t aSize, GM_ADDR cGM,
                                                                            uint32_t cSize)
{
    // MC2的计算流中默认B矩阵不变，GM地址无需偏移
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), aSize);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), cSize);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::InitDequantScale(GM_ADDR dequantGM)
{
    // per-channel quant dequant与bias的shape类似，默认无需偏移
    dequantGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(dequantGM));
    isPerTensor = static_cast<uint32_t>(IsPerTensor::PER_CHANNEL); // per-channel 为2
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::InitDequantScale(uint64_t dequantGM)
{
    // per-tensor quant
    dequantScalar = dequantGM;
    isPerTensor = static_cast<uint32_t>(IsPerTensor::PER_TENSOR); // per-tensor 为1
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::Init(AscendC::tiling::TCubeTiling &tiling,
                                                               Mc2Tiling::RCSTiling &cfg,
                                                               Mc2Tiling::Mc2L2cacheTilePara &tileL2cacheTiling,
                                                               const LocalTensor<uint8_t> &mmFormatUb)
{
    // MatmulImpl初始化
    mm.SetSubBlockIdx(0);
#if __CCE_AICORE__ == 220
    PRELOAD(4);
#endif
    mm.Init(&tiling, GetTPipePtr());
    // MatmulBlock初始化
    block.Init(tiling, cfg, tileL2cacheTiling);
    SetOrgShapeAlign();
#if __CCE_AICORE__ == 220
#else
    mm.SetLocalWorkspace(mmFormatUb);
#endif
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::SetOrgShapeAlign()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        auto alignKa = AlignUp(block.tiling.Ka, SHAPE_ALIGNED_SIZE);
        auto alignKb = AlignUp(block.tiling.Kb, SHAPE_ALIGNED_SIZE);
        auto alignM = AlignUp(block.tiling.M, SHAPE_ALIGNED_SIZE);
        auto alignN = AlignUp(block.tiling.N, SHAPE_ALIGNED_SIZE);
        mm.SetOrgShape(alignM, alignN, alignKa, alignKb, block.cfg.rankN);
    } else if (A_TYPE::format == CubeFormat::NZ) {
        auto alignKa = AlignUp(block.tiling.Ka, SHAPE_ALIGNED_SIZE);
        auto alignM = AlignUp(block.tiling.M, SHAPE_ALIGNED_SIZE);
        mm.SetOrgShape(alignM, block.tiling.N, alignKa, block.tiling.Kb, block.cfg.rankN);
    } else if (B_TYPE::format == CubeFormat::NZ) {
        auto alignKb = AlignUp(block.tiling.Kb, SHAPE_ALIGNED_SIZE);
        auto alignN = AlignUp(block.tiling.N, SHAPE_ALIGNED_SIZE);
        mm.SetOrgShape(block.tiling.M, alignN, block.tiling.Ka, alignKb, block.cfg.rankN);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::SetSingleCoreShape()
{
    if (block.mBlockIndex == (block.mBlockCnt - 1) && block.nBlockIndex == (block.nBlockCnt - 1)) {
        // 最后一块是尾块
        mm.SetSingleShape(block.mBaseTail, block.nBaseTail, block.tiling.singleCoreK);
    } else if (block.mBlockIndex == (block.mBlockCnt - 1)) {
        // M方向的尾块
        mm.SetSingleShape(block.mBaseTail, block.tiling.baseN, block.tiling.singleCoreK);
    } else if (block.nBlockIndex == (block.nBlockCnt - 1)) {
        // N方向的尾块
        mm.SetSingleShape(block.tiling.baseM, block.nBaseTail, block.tiling.singleCoreK);
    } else {
        mm.SetSingleShape(block.tiling.baseM, block.tiling.baseN, block.tiling.singleCoreK);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::Compute(uint32_t index, uint8_t enAtomic)
{
    if constexpr (Mc2L2Cache) {
        ComputeWithL2Cache(index, enAtomic);
    } else {
        ComputeWithNorm(index, enAtomic);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::ComputeWithNorm(uint32_t index, uint8_t enAtomic)
{
    // 每次block循环开始前需要计算初始blockIndex
    block.UpdateBlockCnt(index);
    for (uint32_t i = 0; i < block.blockCnt; i++) {
        if (block.blockIndex < block.totalBlockCnt) {
            block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            SetSingleCoreShape();
            if (isPerTensor == static_cast<uint32_t>(IsPerTensor::PER_TENSOR)) {
                mm.SetQuantScalar(dequantScalar);
            } else if (isPerTensor == static_cast<uint32_t>(IsPerTensor::PER_CHANNEL)) {
                mm.SetQuantVector(dequantGlobal[block.offset.offsetBias]);
            }
            mm.SetTensorA(aGlobal[block.offset.offsetA], block.isTransposeA);
            mm.SetTensorB(bGlobal[block.offset.offsetB], block.isTransposeB);
            if (block.tiling.isBias) {
                mm.SetBias(biasGlobal[block.offset.offsetBias]);
            }
            mm.Iterate();
            mm.GetTensorC(cGlobal[block.offset.offsetC], enAtomic);
#if __CCE_AICORE__ == 220
#else
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
#endif
        }
        block.UpdateBlockIndex();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::MatmulComputeWithL2Cache(int32_t mTileIndex,
                                                                                   int32_t nTileIndex, uint32_t index,
                                                                                   uint8_t enAtomic)
{
    block.UpdateBlockCnt(0, mTileIndex, nTileIndex);
    for (uint32_t j = 0; j < block.blockCnt; j++) {
        if (index < block.totalTileCnt) {
            block.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
            mm.SetSingleShape(block.singleCoreM, block.singleCoreN, block.tiling.singleCoreK);
            if (isPerTensor == static_cast<uint32_t>(IsPerTensor::PER_TENSOR)) {
                mm.SetQuantScalar(dequantScalar);
            } else if (isPerTensor == static_cast<uint32_t>(IsPerTensor::PER_CHANNEL)) {
                mm.SetQuantVector(dequantGlobal[block.offset.offsetBias]);
            }
            mm.SetTensorA(aGlobal[block.offset.offsetA], block.isTransposeA);
            mm.SetTensorB(bGlobal[block.offset.offsetB], block.isTransposeB);
            if (block.tiling.isBias) {
                mm.SetBias(biasGlobal[block.offset.offsetBias]);
            }
            mm.Iterate();
            mm.GetTensorC(cGlobal[block.offset.offsetC], enAtomic);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
        }
        block.UpdateBlockIndex();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType,
                                     hasAntiQuantOffset>::ComputeWithL2Cache(uint32_t index, uint8_t enAtomic)
{
    for (int32_t mTileIndex = 0; mTileIndex < block.tilingL2.mTileCntL2; mTileIndex++) {
        if (mTileIndex % 2 == 0) {
            for (int32_t nTileIndex = 0; nTileIndex < block.tilingL2.nTileCntL2; nTileIndex++) {
                MatmulComputeWithL2Cache(mTileIndex, nTileIndex, index, enAtomic);
            }
        } else {
            for (int32_t nTileIndex = block.tilingL2.nTileCntL2 - 1; nTileIndex >= 0; nTileIndex--) {
                MatmulComputeWithL2Cache(mTileIndex, nTileIndex, index, enAtomic);
            }
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool Mc2L2Cache, bool WeightQuant,
          AntiQuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void
MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2L2Cache, WeightQuant, antiQuantType, hasAntiQuantOffset>::End()
{
    mm.End();
}

} // namespace AscendC
#endif // MC2_MATMUL_COMPUTE_H
