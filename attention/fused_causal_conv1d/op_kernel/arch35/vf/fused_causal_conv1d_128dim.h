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
 * \file compute.h
 * \brief
 */

#ifndef FUSED_CAUSAL_CONV1D_128DIM_H
#define FUSED_CAUSAL_CONV1D_128DIM_H

#include "kernel_operator.h"
#include "fused_causal_conv1d_state_single_tail.h"

template <typename T>
__simd_vf__ void Conv1dNeedStateNoConBHVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                          __ubuf__ T *yAddr, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32,
        mul13B32, y1B32;
    MicroAPI::RegTensor<float> x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32, y2B32;
    MicroAPI::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    MicroAPI::RegTensor<T> x21B16, x22B16, x23B16, y2B16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);

        MicroAPI::Duplicate(y1B32, 0, maskB32);
        MicroAPI::Duplicate(y2B32, 0, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, stateAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, stateAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset);

        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16, stateAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x22B16, xAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x23B16, xAddr + offset + dimLen);

        MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
        MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
        MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
        MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
        MicroAPI::Mul(mul21B32, x21B32, weight11B32, maskB32);
        MicroAPI::Mul(mul22B32, x22B32, weight12B32, maskB32);
        MicroAPI::Mul(mul23B32, x23B32, weight13B32, maskB32);

        MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);

        MicroAPI::Add(y2B32, y2B32, mul21B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul22B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul23B32, maskB32);

        MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset, y1B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + dimLen, y2B16, maskB32);
        offset += B32_REP_SIZE;
    }
}

template <typename T>
__simd_vf__ void Conv1dNeedStateConBHVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                        __ubuf__ T *yAddr, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32,
        mul13B32, y1B32;
    MicroAPI::RegTensor<float> x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32, y2B32;
    MicroAPI::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    MicroAPI::RegTensor<T> x21B16, x22B16, x23B16, y2B16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);

        MicroAPI::Duplicate(y1B32, 0, maskB32);
        MicroAPI::Duplicate(y2B32, 0, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, stateAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, stateAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset);

        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16, stateAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x22B16, xAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x23B16, xAddr + offset + dimLen);

        MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
        MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
        MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
        MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
        MicroAPI::Mul(mul21B32, x21B32, weight11B32, maskB32);
        MicroAPI::Mul(mul22B32, x22B32, weight12B32, maskB32);
        MicroAPI::Mul(mul23B32, x23B32, weight13B32, maskB32);

        MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, x13B32, maskB32);

        MicroAPI::Add(y2B32, y2B32, mul21B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul22B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul23B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, x23B32, maskB32);

        MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset, y1B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + dimLen, y2B16, maskB32);
        offset += B32_REP_SIZE;
    }
}

// 对stateAddr的数据进行原地读写出操作， stateAddr=yAddr
template <typename T>
__simd_vf__ void Conv1dNeedStateConVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                      __ubuf__ T *yAddr, uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> xB32, mulB32, weightB32, yB32;
    MicroAPI::RegTensor<T> xB16, weightB16, yB16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::Duplicate(yB32, 0, maskB32);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                                        weightAddr + offset + stateLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
            // MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(fAddr, weightB32, maskB32);
            MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
        }
        for (uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                weightB16, weightAddr + offset + (xLoop + stateSLen) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
            MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
        }
        MicroAPI::Add(yB32, yB32, xB32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset, yB16, maskB32);
        offset += B32_REP_SIZE;
    }
}

template <typename T>
__simd_vf__ void Conv1dNeedStateNoConVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                        __ubuf__ T *yAddr, uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> xB32, mulB32, weightB32, yB32;
    MicroAPI::RegTensor<T> xB16, weightB16, yB16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::Duplicate(yB32, 0, maskB32);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                                        weightAddr + offset + stateLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
            // MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(fAddr, weightB32, maskB32);
            MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
        }
        for (uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                weightB16, weightAddr + offset + (xLoop + stateSLen) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
            MicroAPI::Mul(mulB32, xB32, weightB32, maskB32);
            MicroAPI::Add(yB32, yB32, mulB32, maskB32);
        }
        MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset, yB16, maskB32);
        offset += B32_REP_SIZE;
    }
}

////残差连接模式，不含尾行
template <typename T>
__simd_vf__ void Conv1dNoNeedStateConNoResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                             uint32_t xLoopNum, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32,
        mul13B32, y1B32;
    MicroAPI::RegTensor<float> weight21B32, weight22B32, weight23B32, x21B32, x22B32, x23B32, mul21B32, mul22B32,
        mul23B32, y2B32;
    MicroAPI::RegTensor<float> x31B32, x32B32, x33B32, mul31B32, mul32B32, mul33B32, y3B32;
    MicroAPI::RegTensor<float> x41B32, x42B32, x43B32, mul41B32, mul42B32, mul43B32, y4B32;
    MicroAPI::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    MicroAPI::RegTensor<T> weight21B16, weight22B16, weight23B16, x21B16, x22B16, x23B16, y2B16;
    MicroAPI::RegTensor<T> x31B16, x32B16, x33B16, y3B16;
    MicroAPI::RegTensor<T> x41B16, x42B16, x43B16, y4B16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE / 2;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight21B16, weightAddr + offset + B32_REP_SIZE);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight22B16,
                                                                    weightAddr + offset + B32_REP_SIZE + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight23B16,
                                                                    weightAddr + offset + B32_REP_SIZE + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskB32);
        for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
            MicroAPI::Duplicate(y1B32, 0, maskB32);
            MicroAPI::Duplicate(y2B32, 0, maskB32);
            MicroAPI::Duplicate(y3B32, 0, maskB32);
            MicroAPI::Duplicate(y4B32, 0, maskB32);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16,
                                                                        xAddr + offset + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16,
                                                                        xAddr + offset + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x21B16, xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x31B16,
                                                                        xAddr + offset + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x32B16,
                                                                        xAddr + offset + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x33B16,
                                                                        xAddr + offset + (2 * xLoop + 3) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x41B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x42B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x43B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskB32);
            MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
            MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
            MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
            MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskB32);
            MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskB32);
            MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskB32);
            MicroAPI::Mul(mul31B32, x31B32, weight11B32, maskB32);
            MicroAPI::Mul(mul32B32, x32B32, weight12B32, maskB32);
            MicroAPI::Mul(mul33B32, x33B32, weight13B32, maskB32);
            MicroAPI::Mul(mul41B32, x41B32, weight21B32, maskB32);
            MicroAPI::Mul(mul42B32, x42B32, weight22B32, maskB32);
            MicroAPI::Mul(mul43B32, x43B32, weight23B32, maskB32);

            MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, x13B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul21B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul22B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul23B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, x23B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul31B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul32B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul33B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, x33B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul41B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul42B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul43B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, x43B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoop * dimLen, y1B16,
                                                                        maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                yAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + (2 * xLoop + 1) * dimLen,
                                                                        y3B16, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                yAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen, y4B16, maskB32);
        }
        offset += 2 * B32_REP_SIZE;
    }
}

// 残差连接模式，含尾行
template <typename T>
__simd_vf__ void Conv1dNoNeedStateConResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                           uint32_t xLoopNum, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32,
        mul13B32, y1B32;
    MicroAPI::RegTensor<float> weight21B32, weight22B32, weight23B32, x21B32, x22B32, x23B32, mul21B32, mul22B32,
        mul23B32, y2B32;
    MicroAPI::RegTensor<float> x31B32, x32B32, x33B32, mul31B32, mul32B32, mul33B32, y3B32;
    MicroAPI::RegTensor<float> x41B32, x42B32, x43B32, mul41B32, mul42B32, mul43B32, y4B32;
    MicroAPI::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    MicroAPI::RegTensor<T> weight21B16, weight22B16, weight23B16, x21B16, x22B16, x23B16, y2B16;
    MicroAPI::RegTensor<T> x31B16, x32B16, x33B16, y3B16;
    MicroAPI::RegTensor<T> x41B16, x42B16, x43B16, y4B16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE / 2;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight21B16, weightAddr + offset + B32_REP_SIZE);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight22B16,
                                                                    weightAddr + offset + B32_REP_SIZE + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight23B16,
                                                                    weightAddr + offset + B32_REP_SIZE + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskB32);
        for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
            MicroAPI::Duplicate(y1B32, 0, maskB32);
            MicroAPI::Duplicate(y2B32, 0, maskB32);
            MicroAPI::Duplicate(y3B32, 0, maskB32);
            MicroAPI::Duplicate(y4B32, 0, maskB32);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16,
                                                                        xAddr + offset + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16,
                                                                        xAddr + offset + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x21B16, xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x31B16,
                                                                        xAddr + offset + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x32B16,
                                                                        xAddr + offset + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x33B16,
                                                                        xAddr + offset + (2 * xLoop + 3) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x41B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x42B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x43B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskB32);
            MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
            MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
            MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
            MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskB32);
            MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskB32);
            MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskB32);
            MicroAPI::Mul(mul31B32, x31B32, weight11B32, maskB32);
            MicroAPI::Mul(mul32B32, x32B32, weight12B32, maskB32);
            MicroAPI::Mul(mul33B32, x33B32, weight13B32, maskB32);
            MicroAPI::Mul(mul41B32, x41B32, weight21B32, maskB32);
            MicroAPI::Mul(mul42B32, x42B32, weight22B32, maskB32);
            MicroAPI::Mul(mul43B32, x43B32, weight23B32, maskB32);

            MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, x13B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul21B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul22B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul23B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, x23B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul31B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul32B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul33B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, x33B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul41B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul42B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul43B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, x43B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoop * dimLen, y1B16,
                                                                        maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                yAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + (2 * xLoop + 1) * dimLen,
                                                                        y3B16, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                yAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen, y4B16, maskB32);
        }
        MicroAPI::Duplicate(y1B32, 0, maskB32);
        MicroAPI::Duplicate(y2B32, 0, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoopNum * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16,
                                                                    xAddr + offset + (2 * xLoopNum + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16,
                                                                    xAddr + offset + (2 * xLoopNum + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x21B16, xAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 2) * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
        MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
        MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
        MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
        MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskB32);
        MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskB32);
        MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, x13B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul21B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul22B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul23B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, x23B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoopNum * dimLen, y1B16,
                                                                    maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen, y2B16, maskB32);

        offset += 2 * B32_REP_SIZE;
    }
}

// 非残差连接模式，不含尾行
template <typename T>
__simd_vf__ void Conv1dNoNeedStateNoConNoResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                               uint32_t xLoopNum, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32,
        mul13B32, y1B32;
    MicroAPI::RegTensor<float> weight21B32, weight22B32, weight23B32, x21B32, x22B32, x23B32, mul21B32, mul22B32,
        mul23B32, y2B32;
    MicroAPI::RegTensor<float> x31B32, x32B32, x33B32, mul31B32, mul32B32, mul33B32, y3B32;
    MicroAPI::RegTensor<float> x41B32, x42B32, x43B32, mul41B32, mul42B32, mul43B32, y4B32;
    MicroAPI::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    MicroAPI::RegTensor<T> weight21B16, weight22B16, weight23B16, x21B16, x22B16, x23B16, y2B16;
    MicroAPI::RegTensor<T> x31B16, x32B16, x33B16, y3B16;
    MicroAPI::RegTensor<T> x41B16, x42B16, x43B16, y4B16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE / 2;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight21B16, weightAddr + offset + B32_REP_SIZE);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight22B16,
                                                                    weightAddr + offset + B32_REP_SIZE + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight23B16,
                                                                    weightAddr + offset + B32_REP_SIZE + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskB32);
        for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
            MicroAPI::Duplicate(y1B32, 0, maskB32);
            MicroAPI::Duplicate(y2B32, 0, maskB32);
            MicroAPI::Duplicate(y3B32, 0, maskB32);
            MicroAPI::Duplicate(y4B32, 0, maskB32);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16,
                                                                        xAddr + offset + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16,
                                                                        xAddr + offset + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x21B16, xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x31B16,
                                                                        xAddr + offset + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x32B16,
                                                                        xAddr + offset + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x33B16,
                                                                        xAddr + offset + (2 * xLoop + 3) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x41B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x42B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x43B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskB32);
            MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
            MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
            MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
            MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskB32);
            MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskB32);
            MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskB32);
            MicroAPI::Mul(mul31B32, x31B32, weight11B32, maskB32);
            MicroAPI::Mul(mul32B32, x32B32, weight12B32, maskB32);
            MicroAPI::Mul(mul33B32, x33B32, weight13B32, maskB32);
            MicroAPI::Mul(mul41B32, x41B32, weight21B32, maskB32);
            MicroAPI::Mul(mul42B32, x42B32, weight22B32, maskB32);
            MicroAPI::Mul(mul43B32, x43B32, weight23B32, maskB32);

            MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul21B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul22B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul23B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul31B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul32B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul33B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul41B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul42B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul43B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoop * dimLen, y1B16,
                                                                        maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                yAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + (2 * xLoop + 1) * dimLen,
                                                                        y3B16, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                yAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen, y4B16, maskB32);
        }
        offset += 2 * B32_REP_SIZE;
    }
}

// 非残差连接模式，含尾行
template <typename T>
__simd_vf__ void Conv1dNoNeedStateNoConResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                             uint32_t xLoopNum, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32,
        mul13B32, y1B32;
    MicroAPI::RegTensor<float> weight21B32, weight22B32, weight23B32, x21B32, x22B32, x23B32, mul21B32, mul22B32,
        mul23B32, y2B32;
    MicroAPI::RegTensor<float> x31B32, x32B32, x33B32, mul31B32, mul32B32, mul33B32, y3B32;
    MicroAPI::RegTensor<float> x41B32, x42B32, x43B32, mul41B32, mul42B32, mul43B32, y4B32;
    MicroAPI::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    MicroAPI::RegTensor<T> weight21B16, weight22B16, weight23B16, x21B16, x22B16, x23B16, y2B16;
    MicroAPI::RegTensor<T> x31B16, x32B16, x33B16, y3B16;
    MicroAPI::RegTensor<T> x41B16, x42B16, x43B16, y4B16;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE / 2;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight21B16, weightAddr + offset + B32_REP_SIZE);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight22B16,
                                                                    weightAddr + offset + B32_REP_SIZE + dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight23B16,
                                                                    weightAddr + offset + B32_REP_SIZE + 2 * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskB32);
        for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
            MicroAPI::Duplicate(y1B32, 0, maskB32);
            MicroAPI::Duplicate(y2B32, 0, maskB32);
            MicroAPI::Duplicate(y3B32, 0, maskB32);
            MicroAPI::Duplicate(y4B32, 0, maskB32);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16,
                                                                        xAddr + offset + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16,
                                                                        xAddr + offset + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x21B16, xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x31B16,
                                                                        xAddr + offset + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x32B16,
                                                                        xAddr + offset + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x33B16,
                                                                        xAddr + offset + (2 * xLoop + 3) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x41B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x42B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                x43B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskB32);
            MicroAPI::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskB32);
            MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
            MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
            MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
            MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskB32);
            MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskB32);
            MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskB32);
            MicroAPI::Mul(mul31B32, x31B32, weight11B32, maskB32);
            MicroAPI::Mul(mul32B32, x32B32, weight12B32, maskB32);
            MicroAPI::Mul(mul33B32, x33B32, weight13B32, maskB32);
            MicroAPI::Mul(mul41B32, x41B32, weight21B32, maskB32);
            MicroAPI::Mul(mul42B32, x42B32, weight22B32, maskB32);
            MicroAPI::Mul(mul43B32, x43B32, weight23B32, maskB32);

            MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
            MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul21B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul22B32, maskB32);
            MicroAPI::Add(y2B32, y2B32, mul23B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul31B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul32B32, maskB32);
            MicroAPI::Add(y3B32, y3B32, mul33B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul41B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul42B32, maskB32);
            MicroAPI::Add(y4B32, y4B32, mul43B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
            MicroAPI::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoop * dimLen, y1B16,
                                                                        maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                yAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + (2 * xLoop + 1) * dimLen,
                                                                        y3B16, maskB32);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                yAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen, y4B16, maskB32);
        }
        MicroAPI::Duplicate(y1B32, 0, maskB32);
        MicroAPI::Duplicate(y2B32, 0, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoopNum * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16,
                                                                    xAddr + offset + (2 * xLoopNum + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16,
                                                                    xAddr + offset + (2 * xLoopNum + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x21B16, xAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 2) * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
        MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
        MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
        MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
        MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskB32);
        MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskB32);
        MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul21B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul22B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul23B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoopNum * dimLen, y1B16,
                                                                    maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen, y2B16, maskB32);

        offset += 2 * B32_REP_SIZE;
    }
}

template <typename T>
__aicore__ inline void Conv1dNeedState(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &stateUb,
                                       LocalTensor<T> &yUb, uint8_t stateSLen, uint32_t xSLen, uint32_t dimLen,
                                       int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *stateAddr = (__ubuf__ T *)stateUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    if (isResidualConnection == 1) {
        Conv1dNeedStateConVF(xAddr, weightAddr, stateAddr, yAddr, stateSLen, xSLen, dimLen);
    } else {
        Conv1dNeedStateNoConVF(xAddr, weightAddr, stateAddr, yAddr, stateSLen, xSLen, dimLen);
    }
}

template <typename T>
__aicore__ inline void Conv1dNeedStateBH(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &stateUb,
                                         LocalTensor<T> &yUb, uint32_t convStateLen, uint32_t dimLen,
                                         int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *stateAddr = (__ubuf__ T *)stateUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    if (isResidualConnection == 1) {
        if (convStateLen == 2) {
            Conv1dNeedStateConBHVF(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        } else {
            uint32_t stateSLen = 2;
            uint32_t xSLen = 1;
            Conv1dNeedStateConVF(xAddr, weightAddr, stateAddr, yAddr, stateSLen, xSLen, dimLen);
        }
    } else {
        if (convStateLen == 2) {
            Conv1dNeedStateNoConBHVF(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        } else {
            uint32_t stateSLen = 2;
            uint32_t xSLen = 1;
            Conv1dNeedStateNoConVF(xAddr, weightAddr, stateAddr, yAddr, stateSLen, xSLen, dimLen);
        }
    }
}

template <typename T>
__aicore__ inline void Conv1dNoNeedState(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &yUb,
                                         uint32_t xSLen, uint32_t dimLen, int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    uint32_t xLoopNum = xSLen / 2;
    uint8_t xLoopRem = xSLen % 2;
    if (isResidualConnection == 1) {
        if (xLoopRem == 1) {
            Conv1dNoNeedStateConResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        } else if (xLoopRem == 0) {
            Conv1dNoNeedStateConNoResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        }
    } else {
        if (xLoopRem == 1) {
            Conv1dNoNeedStateNoConResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        } else if (xLoopRem == 0) {
            Conv1dNoNeedStateNoConNoResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        }
    }
}

#endif
