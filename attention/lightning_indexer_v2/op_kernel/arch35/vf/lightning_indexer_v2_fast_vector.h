/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_FAST_VECTOR_H
#define LIGHTNING_INDEXER_V2_FAST_VECTOR_H

#include "../lightning_indexer_v2_fast_common_arch35.h"
#include "../common/vf/lightning_indexer_v2_vector1_base.h"

namespace LIV2Fast::vf {
namespace Reg = AscendC::Reg;

__aicore__ inline void LoadTileWeights(__gm__ float* weights, uint32_t subBlock, const LocalTensor<float>& weightsLocal,
                                       uint32_t headCount = kHeads, uint32_t validQueryRows = kTileS1)
{
    // FP32 weights can be copied directly into the two WRS slot views.
    // A slot reserves 128 floats: two Q rows * 32 heads, duplicated for the WRS views.
    // AIVs take Q row pairs [0,1]/[2,3], then [4,5]/[6,7].
    for (uint32_t chunk = 0; chunk < kQueryPairRows; ++chunk) {
        const uint32_t firstRow = chunk * kQuerySlotRows + kQueryPairRows * subBlock;
        if (headCount == kHeads && validQueryRows == kTileS1) {
            CopyInUB(weightsLocal[chunk * kWeightSlotElements], weights + static_cast<uint64_t>(firstRow) * headCount,
                     kWeightPairElements);
            CopyInUB(weightsLocal[chunk * kWeightSlotElements + kWeightPairElements],
                     weights + static_cast<uint64_t>(firstRow) * headCount, kWeightPairElements);
        } else {
            for (uint32_t row = 0; row < kQueryPairRows; ++row) {
                const uint32_t queryRow = firstRow + row;
                auto* source = weights + static_cast<uint64_t>(queryRow < validQueryRows ? queryRow : 0U) * headCount;
                CopyInUB(weightsLocal[chunk * kWeightSlotElements + row * kHeads], source, headCount);
                CopyInUB(weightsLocal[chunk * kWeightSlotElements + kWeightPairElements + row * kHeads], source,
                         headCount);
            }
        }
    }
    SetFlag<HardEvent::MTE2_V>(FastEvent::WEIGHTS_READY);
    WaitFlag<HardEvent::MTE2_V>(FastEvent::WEIGHTS_READY);
}

__simd_vf__ inline void ConvertWrsScoresToKeys(__ubuf__ float* score, __ubuf__ uint32_t* keys)
{
    auto full = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<float> value;
    Reg::RegTensor<uint32_t> key;
    liV2Vector1::FloatSortConstCtx<float> context;
    liV2Vector1::InitFloatSortConstCtx(context, full);
    for (uint16_t fragment = 0; fragment < kWrsPairFragmentCount; ++fragment) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(value, score + fragment * kFloatRegLanes);
        liV2Vector1::FloatToSortableKey<float>(key, value, context, full);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(keys + fragment * kFloatRegLanes, key, full);
    }
}

// A row contains four K64 fragments in separate QK bank planes. Use fixed
// register indices, matching the mainline VF helpers' compiler constraints.
__simd_callee__ inline void LoadWrsRow(Reg::RegTensor<float> (&qkRow)[kWrsFragmentCount], __ubuf__ float* qk)
{
    // A K64 plane reserves 4 local Q rows * 32 heads * 64 keys = 8192 floats.
    // Plane bases 0/8192/16384/24576 are element offsets, not byte offsets.
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(qkRow[0], qk);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(qkRow[1], qk + kWrsPlaneStride);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(qkRow[kWrsFragment2], qk + kWrsPlane2Offset);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(qkRow[kWrsFragment3], qk + kWrsPlane3Offset);
}

__simd_callee__ inline void ReluWrsRow(Reg::RegTensor<float> (&qkRow)[kWrsFragmentCount], Reg::MaskReg& full)
{
    Reg::Relu(qkRow[0], qkRow[0], full);
    Reg::Relu(qkRow[1], qkRow[1], full);
    Reg::Relu(qkRow[kWrsFragment2], qkRow[kWrsFragment2], full);
    Reg::Relu(qkRow[kWrsFragment3], qkRow[kWrsFragment3], full);
}

__simd_callee__ inline void AccumulateWrsRow(Reg::RegTensor<float> (&scores)[kWrsFragmentCount],
                                             Reg::RegTensor<float> (&qkRow)[kWrsFragmentCount],
                                             Reg::RegTensor<float>& weight, Reg::MaskReg& full)
{
    Reg::MulAddDst(scores[0], qkRow[0], weight, full);
    Reg::MulAddDst(scores[1], qkRow[1], weight, full);
    Reg::MulAddDst(scores[kWrsFragment2], qkRow[kWrsFragment2], weight, full);
    Reg::MulAddDst(scores[kWrsFragment3], qkRow[kWrsFragment3], weight, full);
}

__simd_callee__ inline void MergeWrsOddHeads(Reg::RegTensor<float> (&scores)[kWrsFragmentCount],
                                             Reg::RegTensor<float> (&oddScores)[kWrsFragmentCount], Reg::MaskReg& full)
{
    Reg::Add(scores[0], scores[0], oddScores[0], full);
    Reg::Add(scores[1], scores[1], oddScores[1], full);
    Reg::Add(scores[kWrsFragment2], scores[kWrsFragment2], oddScores[kWrsFragment2], full);
    Reg::Add(scores[kWrsFragment3], scores[kWrsFragment3], oddScores[kWrsFragment3], full);
}

__simd_callee__ inline void StoreWrsRow(__ubuf__ float* destination, Reg::RegTensor<float> (&scores)[kWrsFragmentCount],
                                        Reg::MaskReg& full)
{
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(destination, scores[0], full);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(destination + kWrsFragment1Offset, scores[1], full);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(destination + kWrsFragment2Offset, scores[kWrsFragment2], full);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(destination + kWrsFragment3Offset, scores[kWrsFragment3], full);
}

// Keep both consumers on the same FP32 accumulation and tail-row rounding order.
// Each AIV processes two queries; their four fragments remain live in this VF.
template <bool SPLIT_FIRST_ROW, bool SPLIT_SECOND_ROW, uint32_t HEADS>
__simd_callee__ inline void AccumulateWrsPair(__ubuf__ float* qk, __ubuf__ float* weights,
                                              Reg::RegTensor<float> (&firstScores)[kWrsFragmentCount],
                                              Reg::RegTensor<float> (&secondScores)[kWrsFragmentCount],
                                              Reg::MaskReg& full, uint32_t runtimeHeads, uint32_t queryHeadStride)
{
    const uint32_t headCount = HEADS == 0U ? runtimeHeads : HEADS;
    Reg::RegTensor<float> firstQk[kWrsFragmentCount], secondQk[kWrsFragmentCount], firstOdd[kWrsFragmentCount],
        secondOdd[kWrsFragmentCount];
    Reg::RegTensor<float> firstWeight, secondWeight;
    if constexpr (SPLIT_SECOND_ROW) {
        liV2Vector1::DuplicateZero(secondOdd, full);
    }
    if constexpr (SPLIT_FIRST_ROW) {
        liV2Vector1::DuplicateZero(firstOdd, full);
    }
    liV2Vector1::DuplicateZero(firstScores, full);
    liV2Vector1::DuplicateZero(secondScores, full);
    for (uint16_t head = 0; head < static_cast<uint16_t>(headCount); ++head) {
        const uint16_t headParity = head % kHeadParityCount;
        const uint64_t base = static_cast<uint64_t>(head) * kWrsHeadStride;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(firstWeight, weights + head);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(secondWeight, weights + kHeads + head);
        LoadWrsRow(firstQk, qk + base);
        LoadWrsRow(secondQk, qk + base + queryHeadStride * kWrsHeadStride);
        ReluWrsRow(firstQk, full);
        ReluWrsRow(secondQk, full);
        // Tail queries retain separate even/odd-head sums to match mainline rounding.
        if (SPLIT_FIRST_ROW && headParity) {
            AccumulateWrsRow(firstOdd, firstQk, firstWeight, full);
        } else {
            AccumulateWrsRow(firstScores, firstQk, firstWeight, full);
        }
        if (SPLIT_SECOND_ROW && headParity) {
            AccumulateWrsRow(secondOdd, secondQk, secondWeight, full);
        } else {
            AccumulateWrsRow(secondScores, secondQk, secondWeight, full);
        }
    }
    if constexpr (SPLIT_FIRST_ROW) {
        MergeWrsOddHeads(firstScores, firstOdd, full);
    }
    if constexpr (SPLIT_SECOND_ROW) {
        MergeWrsOddHeads(secondScores, secondOdd, full);
    }
}

// Preserve the bank-plane addressing and the original sequential FP32
// fused accumulation over valid heads. All eight fragments stay live.
template <bool SPLIT_FIRST_ROW = false, bool SPLIT_SECOND_ROW = false, uint32_t HEADS = 0U>
__simd_vf__ inline void RunBankPlaneWrs(__ubuf__ float* qk, __ubuf__ float* weights, __ubuf__ float* score,
                                        uint32_t runtimeHeads = kHeads, uint32_t queryHeadStride = kHeads)
{
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<float> firstScores[kWrsFragmentCount], secondScores[kWrsFragmentCount];
    AccumulateWrsPair<SPLIT_FIRST_ROW, SPLIT_SECOND_ROW, HEADS>(qk, weights, firstScores, secondScores, full,
                                                                runtimeHeads, queryHeadStride);
    StoreWrsRow(score, firstScores, full);
    StoreWrsRow(score + kFixTile, secondScores, full);
}

} // namespace LIV2Fast::vf
#endif
