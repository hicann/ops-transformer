/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file get_kv_phy_addr_vf.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla / quant_sparse_flash_mla 三个算子共用的
 *        KV 物理地址计算 VF 实现（Pa / Tnd / Bsnd 三种布局）。
 */

#ifndef SPARSE_FLASH_MLA_GET_KV_PHY_ADDR_VF_H
#define SPARSE_FLASH_MLA_GET_KV_PHY_ADDR_VF_H

#include <stdint.h>
#include "static_buffer.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

namespace AttentionCommon {

template <typename T>
__simd_vf__ void GetKVPhyAddrVFPaImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                      __ubuf__ int32_t *blkTableUb, const uint16_t s2Loop, uint32_t s2Tail,
                                      const uint32_t blockSize, const int16_t shiftRightNum,
                                      const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    static const uint16_t paElemsPerLoop = 128;
    static const uint16_t paElemsPerReg = 64;
    static const uint16_t paAddrPerLoop = 256;
    static const uint16_t paAddrPerReg = 128;
    static const uint32_t paInvalidAddr = 0xFFFFFFFF;
    Reg::MaskReg paAllMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg paLowCarry0;
    Reg::MaskReg paHighCarry0;
    Reg::MaskReg paLowCarry1;
    Reg::MaskReg paHighCarry1;
    Reg::MaskReg paInvalidMask0;
    Reg::MaskReg paInvalidMask1;

    Reg::RegTensor<uint32_t> paKvStride;
    Reg::RegTensor<uint32_t> paSparseIndex0;
    Reg::RegTensor<uint32_t> paSparseIndex1;
    Reg::RegTensor<uint32_t> paBlockSize;
    Reg::RegTensor<uint32_t> paShiftRight;
    Reg::RegTensor<uint32_t> paBlockIndex0;
    Reg::RegTensor<uint32_t> paBlockIndex1;
    Reg::RegTensor<uint32_t> paBlockOffset0;
    Reg::RegTensor<uint32_t> paBlockOffset1;
    Reg::RegTensor<uint32_t> paTileOffset0;
    Reg::RegTensor<uint32_t> paTileOffset1;
    Reg::RegTensor<uint32_t> paPhysicalOffset0;
    Reg::RegTensor<uint32_t> paPhysicalOffset1;
    Reg::RegTensor<uint32_t> paPhysicalBlock0;
    Reg::RegTensor<uint32_t> paPhysicalBlock1;

    Reg::RegTensor<uint32_t> paBlockStrideHigh0;
    Reg::RegTensor<uint32_t> paBlockStrideTemp0;
    Reg::RegTensor<uint32_t> paBlockStrideLow0;
    Reg::RegTensor<uint32_t> paMultiplyCarry0;
    Reg::RegTensor<uint32_t> paAddressLow0;
    Reg::RegTensor<uint32_t> paAddressHigh0;

    Reg::RegTensor<uint32_t> paBlockStrideHigh1;
    Reg::RegTensor<uint32_t> paBlockStrideTemp1;
    Reg::RegTensor<uint32_t> paBlockStrideLow1;
    Reg::RegTensor<uint32_t> paMultiplyCarry1;
    Reg::RegTensor<uint32_t> paAddressLow1;
    Reg::RegTensor<uint32_t> paAddressHigh1;

    Reg::RegTensor<uint32_t> paZero;
    Reg::Duplicate(paZero, 0);
    Reg::Duplicate(paKvStride, kvStride);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)paSparseIndex0,
                                                              sparseIdxUb + i * paElemsPerLoop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)paSparseIndex1,
                                                              sparseIdxUb + paElemsPerReg + i * paElemsPerLoop);
            // * sparseBlockSize
            Reg::Muls(paSparseIndex0, paSparseIndex0, sparseBlockSize, paAllMask);
            Reg::Muls(paSparseIndex1, paSparseIndex1, sparseBlockSize, paAllMask);
            // 计算右移位数
            // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
            Reg::ShiftRights(paBlockIndex0, paSparseIndex0, shiftRightNum, paAllMask);
            Reg::ShiftRights(paBlockIndex1, paSparseIndex1, shiftRightNum, paAllMask);

            Reg::Muls(paBlockOffset0, paBlockIndex0, blockSize, paAllMask);
            Reg::Muls(paBlockOffset1, paBlockIndex1, blockSize, paAllMask);
            // offset
            Reg::Sub(paTileOffset0, paSparseIndex0, paBlockOffset0, paAllMask);
            Reg::Sub(paTileOffset1, paSparseIndex1, paBlockOffset1, paAllMask);
            // 物理页内offset
            Reg::Muls(paPhysicalOffset0, paTileOffset0, kvDim, paAllMask);
            Reg::Muls(paPhysicalOffset1, paTileOffset1, kvDim, paAllMask);

            // int32 paBlockId -> 物理id
            DataCopyGather(paPhysicalBlock0, blkTableUb, paBlockIndex0, paAllMask);
            DataCopyGather(paPhysicalBlock1, blkTableUb, paBlockIndex1, paAllMask);

            // 分高低32位计算int64物理地址 -- 乘 stride
            // 低位乘 带进位
            Reg::Mull(paBlockStrideLow0, paMultiplyCarry0, paPhysicalBlock0, paKvStride, paAllMask);
            Reg::Mull(paBlockStrideLow1, paMultiplyCarry1, paPhysicalBlock1, paKvStride, paAllMask);

            // 分高低32位计算int64物理地址 -- 加 offset
            Reg::Add(paLowCarry0, paAddressLow0, paBlockStrideLow0, paPhysicalOffset0, paAllMask);
            Reg::Add(paLowCarry1, paAddressLow1, paBlockStrideLow1, paPhysicalOffset1, paAllMask);

            Reg::AddC(paHighCarry0, paAddressHigh0, paMultiplyCarry0, paZero, paLowCarry0, paAllMask);
            Reg::AddC(paHighCarry1, paAddressHigh1, paMultiplyCarry1, paZero, paLowCarry1, paAllMask);

            // 搬出 由于拆分为了int32类型，元素个数翻倍
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(kvPhyAddrUb + i * paAddrPerLoop, paAddressLow0,
                                                                      paAddressHigh0, paAllMask);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(kvPhyAddrUb + paAddrPerReg + i * paAddrPerLoop,
                                                                      paAddressLow1, paAddressHigh1, paAllMask);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        Reg::MaskReg paTailMask0 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::MaskReg paTailMask1 = Reg::UpdateMask<int32_t>(s2Tail);
        Reg::Not(paInvalidMask0, paTailMask0, paAllMask);
        Reg::Not(paInvalidMask1, paTailMask1, paAllMask);

        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)paSparseIndex0,
                                                          sparseIdxUb + i * paElemsPerLoop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)paSparseIndex1,
                                                          sparseIdxUb + paElemsPerReg + i * paElemsPerLoop);
        // * sparseBlockSize
        Reg::Muls(paSparseIndex0, paSparseIndex0, sparseBlockSize, paTailMask0);
        Reg::Muls(paSparseIndex1, paSparseIndex1, sparseBlockSize, paTailMask1);
        // 计算右移位数
        // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
        Reg::ShiftRights(paBlockIndex0, paSparseIndex0, shiftRightNum, paTailMask0);
        Reg::ShiftRights(paBlockIndex1, paSparseIndex1, shiftRightNum, paTailMask1);

        Reg::Muls(paBlockOffset0, paBlockIndex0, blockSize, paTailMask0);
        Reg::Muls(paBlockOffset1, paBlockIndex1, blockSize, paTailMask1);
        // offset
        Reg::Sub(paTileOffset0, paSparseIndex0, paBlockOffset0, paTailMask0);
        Reg::Sub(paTileOffset1, paSparseIndex1, paBlockOffset1, paTailMask1);
        // 物理页内offset
        Reg::Muls(paPhysicalOffset0, paTileOffset0, kvDim, paTailMask0);
        Reg::Muls(paPhysicalOffset1, paTileOffset1, kvDim, paTailMask1);

        // int32 paBlockId -> 物理id
        DataCopyGather(paPhysicalBlock0, blkTableUb, paBlockIndex0, paTailMask0);
        DataCopyGather(paPhysicalBlock1, blkTableUb, paBlockIndex1, paTailMask1);

        // 分高低32位计算int64物理地址 -- 乘 stride
        // 低位乘 带进位
        Reg::Mull(paBlockStrideLow0, paMultiplyCarry0, paPhysicalBlock0, paKvStride, paTailMask0);
        Reg::Mull(paBlockStrideLow1, paMultiplyCarry1, paPhysicalBlock1, paKvStride, paTailMask1);

        // 分高低32位计算int64物理地址 -- 加 offset
        Reg::Add(paLowCarry0, paAddressLow0, paBlockStrideLow0, paPhysicalOffset0, paTailMask0);
        Reg::Add(paLowCarry1, paAddressLow1, paBlockStrideLow1, paPhysicalOffset1, paTailMask1);

        Reg::AddC(paHighCarry0, paAddressHigh0, paMultiplyCarry0, paZero, paLowCarry0, paTailMask0);
        Reg::AddC(paHighCarry1, paAddressHigh1, paMultiplyCarry1, paZero, paLowCarry1, paTailMask1);

        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(paAddressLow0, paInvalidAddr, paInvalidMask0);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(paAddressHigh0, paInvalidAddr, paInvalidMask0);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(paAddressLow1, paInvalidAddr, paInvalidMask1);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(paAddressHigh1, paInvalidAddr, paInvalidMask1);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(kvPhyAddrUb + i * paAddrPerLoop, paAddressLow0,
                                                                  paAddressHigh0, paAllMask);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(kvPhyAddrUb + paAddrPerReg + i * paAddrPerLoop,
                                                                  paAddressLow1, paAddressHigh1, paAllMask);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFPa(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                        LocalTensor<int32_t> blkTableTensor, const uint16_t s2Loop,
                                        const uint32_t s2Tail, const uint32_t blockSize, const int16_t shiftRightNum,
                                        const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    __ubuf__ int32_t *blk_table_ub = (__ubuf__ int32_t *)(blkTableTensor.GetPhyAddr());
    GetKVPhyAddrVFPaImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, blk_table_ub, s2Loop, s2Tail, blockSize,
                                   shiftRightNum, sparseBlockSize, kvDim, kvStride);
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFTndImpl(__ubuf__ uint32_t *tndAddrUb, __ubuf__ int32_t *tndSparseUb,
                                       const uint16_t tndLoopCount, uint32_t tndTailSize,
                                       const uint32_t tndSparseBlockSize, const uint32_t tndKvDim,
                                       const uint32_t tndKvPrefix)
{
    static const uint16_t tndElemsPerLoop = 128;
    static const uint16_t tndElemsPerReg = 64;
    static const uint16_t tndOutPerLoop = 256;
    static const uint16_t tndOutPerReg = 128;
    static const uint32_t tndInvalidAddr = 0xFFFFFFFF;
    Reg::MaskReg tndAllMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg tndInvalidMask0;
    Reg::MaskReg tndInvalidMask1;

    Reg::RegTensor<uint32_t> tndSparseIndex0;
    Reg::RegTensor<uint32_t> tndSparseIndex1;
    Reg::RegTensor<uint32_t> tndPrefixReg;
    Reg::RegTensor<uint32_t> tndDimReg;
    Reg::RegTensor<uint32_t> tndTokenOffset0;
    Reg::RegTensor<uint32_t> tndTokenOffset1;
    Reg::RegTensor<uint32_t> tndCarry0;
    Reg::RegTensor<uint32_t> tndCarry1;
    Reg::RegTensor<uint32_t> tndAddrLow0;
    Reg::RegTensor<uint32_t> tndAddrHigh0;
    Reg::RegTensor<uint32_t> tndAddrLow1;
    Reg::RegTensor<uint32_t> tndAddrHigh1;

    Reg::Duplicate(tndPrefixReg, tndKvPrefix);
    Reg::Duplicate(tndDimReg, tndKvDim);

    // * sparseBlockSize
    // (kvPrefix + sparseIdx) * kvDim -> int64 物理地址
    for (; tndLoopCount > 1;) {
        for (uint16_t i = 0; i < tndLoopCount - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)tndSparseIndex0,
                                                              tndSparseUb + i * tndElemsPerLoop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)tndSparseIndex1,
                                                              tndSparseUb + tndElemsPerReg + i * tndElemsPerLoop);
            Reg::Muls(tndSparseIndex0, tndSparseIndex0, tndSparseBlockSize, tndAllMask);
            Reg::Muls(tndSparseIndex1, tndSparseIndex1, tndSparseBlockSize, tndAllMask);
            Reg::Add(tndTokenOffset0, tndSparseIndex0, tndPrefixReg, tndAllMask);
            Reg::Add(tndTokenOffset1, tndSparseIndex1, tndPrefixReg, tndAllMask);
            // 带进位乘法
            Reg::Mull(tndAddrLow0, tndAddrHigh0, tndTokenOffset0, tndDimReg, tndAllMask);
            Reg::Mull(tndAddrLow1, tndAddrHigh1, tndTokenOffset1, tndDimReg, tndAllMask);
            // 搬出
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(tndAddrUb + i * tndOutPerLoop, tndAddrLow0,
                                                                      tndAddrHigh0, tndAllMask);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(tndAddrUb + tndOutPerReg + i * tndOutPerLoop,
                                                                      tndAddrLow1, tndAddrHigh1, tndAllMask);
        }
        break;
    }

    for (uint16_t i = tndLoopCount - 1; i < tndLoopCount; i++) {
        Reg::MaskReg tndTailMask0 = Reg::UpdateMask<int32_t>(tndTailSize);
        Reg::MaskReg tndTailMask1 = Reg::UpdateMask<int32_t>(tndTailSize);
        Reg::Not(tndInvalidMask0, tndTailMask0, tndAllMask);
        Reg::Not(tndInvalidMask1, tndTailMask1, tndAllMask);

        // * sparseBlockSize
        // (kvPrefix + sparseIdx) * kvDim -> int64 物理地址
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)tndSparseIndex0,
                                                          tndSparseUb + i * tndElemsPerLoop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)tndSparseIndex1,
                                                          tndSparseUb + tndElemsPerReg + i * tndElemsPerLoop);
        Reg::Muls(tndSparseIndex0, tndSparseIndex0, tndSparseBlockSize, tndTailMask0);
        Reg::Muls(tndSparseIndex1, tndSparseIndex1, tndSparseBlockSize, tndTailMask1);
        Reg::Add(tndTokenOffset0, tndSparseIndex0, tndPrefixReg, tndTailMask0);
        Reg::Add(tndTokenOffset1, tndSparseIndex1, tndPrefixReg, tndTailMask1);
        // 带进位乘法
        Reg::Mull(tndAddrLow0, tndAddrHigh0, tndTokenOffset0, tndDimReg, tndTailMask0);
        Reg::Mull(tndAddrLow1, tndAddrHigh1, tndTokenOffset1, tndDimReg, tndTailMask1);
        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(tndAddrLow0, tndInvalidAddr, tndInvalidMask0);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(tndAddrHigh0, tndInvalidAddr, tndInvalidMask0);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(tndAddrLow1, tndInvalidAddr, tndInvalidMask1);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(tndAddrHigh1, tndInvalidAddr, tndInvalidMask1);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(tndAddrUb + i * tndOutPerLoop, tndAddrLow0,
                                                                  tndAddrHigh0, tndAllMask);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(tndAddrUb + tndOutPerReg + i * tndOutPerLoop,
                                                                  tndAddrLow1, tndAddrHigh1, tndAllMask);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFTnd(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                         const uint16_t s2Loop, const uint32_t s2Tail, const uint32_t sparseBlockSize,
                                         const uint32_t kvDim, const uint32_t kvPrefix)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    GetKVPhyAddrVFTndImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, s2Loop, s2Tail, sparseBlockSize, kvDim, kvPrefix);
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFBsndImpl(__ubuf__ uint32_t *bsndAddrUb, __ubuf__ int32_t *bsndSparseUb,
                                        const uint16_t bsndLoopCount, uint32_t bsndTailSize,
                                        const uint32_t bsndSparseBlockSize, const uint32_t bsndKvDim,
                                        const uint32_t bsndBaseLow, const uint32_t bsndBaseHigh)
{
    static const uint16_t bsndElemsPerLoop = 128;
    static const uint16_t bsndElemsPerReg = 64;
    static const uint16_t bsndOutPerLoop = 256;
    static const uint16_t bsndOutPerReg = 128;
    static const uint32_t bsndInvalidAddr = 0xFFFFFFFF;
    Reg::MaskReg bsndAllMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg bsndLowCarry0;
    Reg::MaskReg bsndHighCarry0;
    Reg::MaskReg bsndLowCarry1;
    Reg::MaskReg bsndHighCarry1;
    Reg::MaskReg bsndInvalidMask0;
    Reg::MaskReg bsndInvalidMask1;

    Reg::RegTensor<uint32_t> bsndSparseIndex0;
    Reg::RegTensor<uint32_t> bsndSparseIndex1;
    Reg::RegTensor<uint32_t> bsndDimReg;
    Reg::RegTensor<uint32_t> bsndBaseLowReg;
    Reg::RegTensor<uint32_t> bsndBaseHighReg;
    Reg::RegTensor<uint32_t> bsndOffsetLow0;
    Reg::RegTensor<uint32_t> bsndOffsetLow1;
    Reg::RegTensor<uint32_t> bsndOffsetHigh0;
    Reg::RegTensor<uint32_t> bsndOffsetHigh1;
    Reg::RegTensor<uint32_t> bsndAddrLow0;
    Reg::RegTensor<uint32_t> bsndAddrHigh0;
    Reg::RegTensor<uint32_t> bsndAddrLow1;
    Reg::RegTensor<uint32_t> bsndAddrHigh1;
    Reg::RegTensor<uint32_t> bsndZeroReg;

    Reg::Duplicate(bsndZeroReg, 0);
    Reg::Duplicate(bsndDimReg, bsndKvDim);
    Reg::Duplicate(bsndBaseLowReg, bsndBaseLow);
    Reg::Duplicate(bsndBaseHighReg, bsndBaseHigh);

    // * sparseBlockSize
    // sparseIdx * kvDim (带进位乘法)
    for (; bsndLoopCount > 1;) {
        for (uint16_t i = 0; i < bsndLoopCount - 1; i++) {
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)bsndSparseIndex0,
                                                              bsndSparseUb + i * bsndElemsPerLoop);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)bsndSparseIndex1,
                                                              bsndSparseUb + bsndElemsPerReg + i * bsndElemsPerLoop);
            Reg::Muls(bsndSparseIndex0, bsndSparseIndex0, bsndSparseBlockSize, bsndAllMask);
            Reg::Muls(bsndSparseIndex1, bsndSparseIndex1, bsndSparseBlockSize, bsndAllMask);
            Reg::Mull(bsndOffsetLow0, bsndOffsetHigh0, bsndSparseIndex0, bsndDimReg, bsndAllMask);
            Reg::Mull(bsndOffsetLow1, bsndOffsetHigh1, bsndSparseIndex1, bsndDimReg, bsndAllMask);
            // s2_offset + bS2Base (int64 + int64)
            Reg::Add(bsndLowCarry0, bsndAddrLow0, bsndOffsetLow0, bsndBaseLowReg, bsndAllMask);
            Reg::Add(bsndLowCarry1, bsndAddrLow1, bsndOffsetLow1, bsndBaseLowReg, bsndAllMask);
            Reg::AddC(bsndHighCarry0, bsndAddrHigh0, bsndOffsetHigh0, bsndBaseHighReg, bsndLowCarry0, bsndAllMask);
            Reg::AddC(bsndHighCarry1, bsndAddrHigh1, bsndOffsetHigh1, bsndBaseHighReg, bsndLowCarry1, bsndAllMask);
            // 搬出
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(bsndAddrUb + i * bsndOutPerLoop, bsndAddrLow0,
                                                                      bsndAddrHigh0, bsndAllMask);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(bsndAddrUb + bsndOutPerReg + i * bsndOutPerLoop,
                                                                      bsndAddrLow1, bsndAddrHigh1, bsndAllMask);
        }
        break;
    }

    for (uint16_t i = bsndLoopCount - 1; i < bsndLoopCount; i++) {
        Reg::MaskReg bsndTailMask0 = Reg::UpdateMask<int32_t>(bsndTailSize);
        Reg::MaskReg bsndTailMask1 = Reg::UpdateMask<int32_t>(bsndTailSize);
        Reg::Not(bsndInvalidMask0, bsndTailMask0, bsndAllMask);
        Reg::Not(bsndInvalidMask1, bsndTailMask1, bsndAllMask);

        // * sparseBlockSize
        // sparseIdx * kvDim (带进位乘法)
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)bsndSparseIndex0,
                                                          bsndSparseUb + i * bsndElemsPerLoop);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<int32_t> &)bsndSparseIndex1,
                                                          bsndSparseUb + bsndElemsPerReg + i * bsndElemsPerLoop);
        Reg::Muls(bsndSparseIndex0, bsndSparseIndex0, bsndSparseBlockSize, bsndTailMask0);
        Reg::Muls(bsndSparseIndex1, bsndSparseIndex1, bsndSparseBlockSize, bsndTailMask1);
        Reg::Mull(bsndOffsetLow0, bsndOffsetHigh0, bsndSparseIndex0, bsndDimReg, bsndTailMask0);
        Reg::Mull(bsndOffsetLow1, bsndOffsetHigh1, bsndSparseIndex1, bsndDimReg, bsndTailMask1);
        // s2_offset + bS2Base (int64 + int64)
        Reg::Add(bsndLowCarry0, bsndAddrLow0, bsndOffsetLow0, bsndBaseLowReg, bsndTailMask0);
        Reg::Add(bsndLowCarry1, bsndAddrLow1, bsndOffsetLow1, bsndBaseLowReg, bsndTailMask1);
        Reg::AddC(bsndHighCarry0, bsndAddrHigh0, bsndOffsetHigh0, bsndBaseHighReg, bsndLowCarry0, bsndTailMask0);
        Reg::AddC(bsndHighCarry1, bsndAddrHigh1, bsndOffsetHigh1, bsndBaseHighReg, bsndLowCarry1, bsndTailMask1);
        // 无效值填充-1(0xFFFFFFFF)
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(bsndAddrLow0, bsndInvalidAddr, bsndInvalidMask0);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(bsndAddrHigh0, bsndInvalidAddr, bsndInvalidMask0);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(bsndAddrLow1, bsndInvalidAddr, bsndInvalidMask1);
        Reg::Duplicate<uint32_t, Reg::MaskMergeMode::MERGING>(bsndAddrHigh1, bsndInvalidAddr, bsndInvalidMask1);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(bsndAddrUb + i * bsndOutPerLoop, bsndAddrLow0,
                                                                  bsndAddrHigh0, bsndAllMask);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(bsndAddrUb + bsndOutPerReg + i * bsndOutPerLoop,
                                                                  bsndAddrLow1, bsndAddrHigh1, bsndAllMask);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVFBsnd(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                          const uint16_t s2Loop, const uint32_t s2Tail, const uint32_t sparseBlockSize,
                                          const uint32_t kvDim, const uint32_t bS2BaseLow, const uint32_t bS2BaseHigh)
{
    __ubuf__ uint32_t *kv_phy_addr_ub = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparse_idx_ub = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    GetKVPhyAddrVFBsndImpl<uint32_t>(kv_phy_addr_ub, sparse_idx_ub, s2Loop, s2Tail, sparseBlockSize, kvDim, bS2BaseLow,
                                     bS2BaseHigh);
}

} // namespace AttentionCommon

#endif // SPARSE_FLASH_MLA_GET_KV_PHY_ADDR_VF_H
