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

#ifndef FUSED_CAUSAL_CONV1D_NO_STATE_SINGLE_TAIL_H
#define FUSED_CAUSAL_CONV1D_NO_STATE_SINGLE_TAIL_H

#include "kernel_operator.h"
#include "fused_causal_conv1d_state_single_tail.h"

using namespace AscendC;
////残差连接模式，不含尾行
template <typename T>
__simd_vf__ void Conv1dNoStateSingleTailConNoResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                                   uint32_t xLoopNum, uint32_t dimLen)
{
    Reg::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32, mul13B32,
        y1B32;
    Reg::RegTensor<float> weight21B32, weight22B32, weight23B32, x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32,
        y2B32;
    Reg::RegTensor<float> x31B32, x32B32, x33B32, mul31B32, mul32B32, mul33B32, y3B32;
    Reg::RegTensor<float> x41B32, x42B32, x43B32, mul41B32, mul42B32, mul43B32, y4B32;
    Reg::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    Reg::RegTensor<T> weight21B16, weight22B16, weight23B16, x21B16, x22B16, x23B16, y2B16;
    Reg::RegTensor<T> x31B16, x32B16, x33B16, y3B16;
    Reg::RegTensor<T> x41B16, x42B16, x43B16, y4B16;
    uint8_t dimLoopNum = dimLen / (B32_REP_SIZE * 2);
    uint16_t dimRem = dimLen - dimLoopNum * (B32_REP_SIZE * 2);
    Reg::MaskReg maskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRem;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight21B16, weightAddr + offset + B32_REP_SIZE);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight22B16, weightAddr + offset + B32_REP_SIZE + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight23B16, weightAddr + offset + B32_REP_SIZE + 2 * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskB32);
        for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
            Reg::Duplicate(y1B32, 0, maskB32);
            Reg::Duplicate(y2B32, 0, maskB32);
            Reg::Duplicate(y3B32, 0, maskB32);
            Reg::Duplicate(y4B32, 0, maskB32);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                              xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x41B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x42B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x43B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
            Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskB32);
            Reg::Mul(mul11B32, x11B32, weight11B32, maskB32);
            Reg::Mul(mul12B32, x12B32, weight12B32, maskB32);
            Reg::Mul(mul13B32, x13B32, weight13B32, maskB32);
            Reg::Mul(mul21B32, x21B32, weight21B32, maskB32);
            Reg::Mul(mul22B32, x22B32, weight22B32, maskB32);
            Reg::Mul(mul23B32, x23B32, weight23B32, maskB32);
            Reg::Mul(mul31B32, x31B32, weight11B32, maskB32);
            Reg::Mul(mul32B32, x32B32, weight12B32, maskB32);
            Reg::Mul(mul33B32, x33B32, weight13B32, maskB32);
            Reg::Mul(mul41B32, x41B32, weight21B32, maskB32);
            Reg::Mul(mul42B32, x42B32, weight22B32, maskB32);
            Reg::Mul(mul43B32, x43B32, weight23B32, maskB32);

            Reg::Add(y1B32, y1B32, mul11B32, maskB32);
            Reg::Add(y1B32, y1B32, mul12B32, maskB32);
            Reg::Add(y1B32, y1B32, mul13B32, maskB32);
            Reg::Add(y1B32, y1B32, x13B32, maskB32);
            Reg::Add(y2B32, y2B32, mul21B32, maskB32);
            Reg::Add(y2B32, y2B32, mul22B32, maskB32);
            Reg::Add(y2B32, y2B32, mul23B32, maskB32);
            Reg::Add(y2B32, y2B32, x23B32, maskB32);
            Reg::Add(y3B32, y3B32, mul31B32, maskB32);
            Reg::Add(y3B32, y3B32, mul32B32, maskB32);
            Reg::Add(y3B32, y3B32, mul33B32, maskB32);
            Reg::Add(y3B32, y3B32, x33B32, maskB32);
            Reg::Add(y4B32, y4B32, mul41B32, maskB32);
            Reg::Add(y4B32, y4B32, mul42B32, maskB32);
            Reg::Add(y4B32, y4B32, mul43B32, maskB32);
            Reg::Add(y4B32, y4B32, x43B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoop * dimLen, y1B16, maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16,
                                                              maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + (2 * xLoop + 1) * dimLen, y3B16,
                                                              maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen,
                                                              y4B16, maskB32);
        }
        offset += 2 * B32_REP_SIZE;
    }
    // dim尾块单寄存器处理
    uint32_t sreg = dimRem;
    maskRem = Reg::UpdateMask<float>(sreg);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
    Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskRem);
    for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
        Reg::Duplicate(y1B32, 0, maskRem);
        Reg::Duplicate(y3B32, 0, maskRem);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);

        Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskRem);

        Reg::Mul(mul11B32, x11B32, weight11B32, maskRem);
        Reg::Mul(mul12B32, x12B32, weight12B32, maskRem);
        Reg::Mul(mul13B32, x13B32, weight13B32, maskRem);
        Reg::Mul(mul31B32, x31B32, weight11B32, maskRem);
        Reg::Mul(mul32B32, x32B32, weight12B32, maskRem);
        Reg::Mul(mul33B32, x33B32, weight13B32, maskRem);

        Reg::Add(y1B32, y1B32, mul11B32, maskRem);
        Reg::Add(y1B32, y1B32, mul12B32, maskRem);
        Reg::Add(y1B32, y1B32, mul13B32, maskRem);
        Reg::Add(y1B32, y1B32, x13B32, maskRem);
        Reg::Add(y3B32, y3B32, mul31B32, maskRem);
        Reg::Add(y3B32, y3B32, mul32B32, maskRem);
        Reg::Add(y3B32, y3B32, mul33B32, maskRem);
        Reg::Add(y3B32, y3B32, x33B32, maskRem);

        Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
        Reg::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskRem);

        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoop * dimLen,
                                                          y1B16, maskRem);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + (2 * xLoop + 1) * dimLen, y3B16, maskRem);
    }
}

// 残差连接模式，含尾行
template <typename T>
__simd_vf__ void Conv1dNoStateSingleTailConResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                                 uint32_t xLoopNum, uint32_t dimLen)
{
    Reg::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32, mul13B32,
        y1B32;
    Reg::RegTensor<float> weight21B32, weight22B32, weight23B32, x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32,
        y2B32;
    Reg::RegTensor<float> x31B32, x32B32, x33B32, mul31B32, mul32B32, mul33B32, y3B32;
    Reg::RegTensor<float> x41B32, x42B32, x43B32, mul41B32, mul42B32, mul43B32, y4B32;
    Reg::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    Reg::RegTensor<T> weight21B16, weight22B16, weight23B16, x21B16, x22B16, x23B16, y2B16;
    Reg::RegTensor<T> x31B16, x32B16, x33B16, y3B16;
    Reg::RegTensor<T> x41B16, x42B16, x43B16, y4B16;
    uint8_t dimLoopNum = dimLen / (B32_REP_SIZE * 2);
    uint16_t dimRem = dimLen - dimLoopNum * (B32_REP_SIZE * 2);
    Reg::MaskReg maskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRem;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight21B16, weightAddr + offset + B32_REP_SIZE);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight22B16, weightAddr + offset + B32_REP_SIZE + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight23B16, weightAddr + offset + B32_REP_SIZE + 2 * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskB32);
        for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
            Reg::Duplicate(y1B32, 0, maskB32);
            Reg::Duplicate(y2B32, 0, maskB32);
            Reg::Duplicate(y3B32, 0, maskB32);
            Reg::Duplicate(y4B32, 0, maskB32);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                              xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x41B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x42B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x43B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
            Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskB32);
            Reg::Mul(mul11B32, x11B32, weight11B32, maskB32);
            Reg::Mul(mul12B32, x12B32, weight12B32, maskB32);
            Reg::Mul(mul13B32, x13B32, weight13B32, maskB32);
            Reg::Mul(mul21B32, x21B32, weight21B32, maskB32);
            Reg::Mul(mul22B32, x22B32, weight22B32, maskB32);
            Reg::Mul(mul23B32, x23B32, weight23B32, maskB32);
            Reg::Mul(mul31B32, x31B32, weight11B32, maskB32);
            Reg::Mul(mul32B32, x32B32, weight12B32, maskB32);
            Reg::Mul(mul33B32, x33B32, weight13B32, maskB32);
            Reg::Mul(mul41B32, x41B32, weight21B32, maskB32);
            Reg::Mul(mul42B32, x42B32, weight22B32, maskB32);
            Reg::Mul(mul43B32, x43B32, weight23B32, maskB32);

            Reg::Add(y1B32, y1B32, mul11B32, maskB32);
            Reg::Add(y1B32, y1B32, mul12B32, maskB32);
            Reg::Add(y1B32, y1B32, mul13B32, maskB32);
            Reg::Add(y1B32, y1B32, x13B32, maskB32);
            Reg::Add(y2B32, y2B32, mul21B32, maskB32);
            Reg::Add(y2B32, y2B32, mul22B32, maskB32);
            Reg::Add(y2B32, y2B32, mul23B32, maskB32);
            Reg::Add(y2B32, y2B32, x23B32, maskB32);
            Reg::Add(y3B32, y3B32, mul31B32, maskB32);
            Reg::Add(y3B32, y3B32, mul32B32, maskB32);
            Reg::Add(y3B32, y3B32, mul33B32, maskB32);
            Reg::Add(y3B32, y3B32, x33B32, maskB32);
            Reg::Add(y4B32, y4B32, mul41B32, maskB32);
            Reg::Add(y4B32, y4B32, mul42B32, maskB32);
            Reg::Add(y4B32, y4B32, mul43B32, maskB32);
            Reg::Add(y4B32, y4B32, x43B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoop * dimLen, y1B16, maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16,
                                                              maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + (2 * xLoop + 1) * dimLen, y3B16,
                                                              maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen,
                                                              y4B16, maskB32);
        }
        Reg::Duplicate(y1B32, 0, maskB32);
        Reg::Duplicate(y2B32, 0, maskB32);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoopNum * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoopNum + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoopNum + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                          xAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16,
                                                          xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16,
                                                          xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 2) * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
        Reg::Mul(mul11B32, x11B32, weight11B32, maskB32);
        Reg::Mul(mul12B32, x12B32, weight12B32, maskB32);
        Reg::Mul(mul13B32, x13B32, weight13B32, maskB32);
        Reg::Mul(mul21B32, x21B32, weight21B32, maskB32);
        Reg::Mul(mul22B32, x22B32, weight22B32, maskB32);
        Reg::Mul(mul23B32, x23B32, weight23B32, maskB32);
        Reg::Add(y1B32, y1B32, mul11B32, maskB32);
        Reg::Add(y1B32, y1B32, mul12B32, maskB32);
        Reg::Add(y1B32, y1B32, mul13B32, maskB32);
        Reg::Add(y1B32, y1B32, x13B32, maskB32);
        Reg::Add(y2B32, y2B32, mul21B32, maskB32);
        Reg::Add(y2B32, y2B32, mul22B32, maskB32);
        Reg::Add(y2B32, y2B32, mul23B32, maskB32);
        Reg::Add(y2B32, y2B32, x23B32, maskB32);
        Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoopNum * dimLen, y1B16, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen, y2B16,
                                                          maskB32);

        offset += 2 * B32_REP_SIZE;
    }

    // dim尾块单寄存器处理
    uint32_t sreg = dimRem;
    maskRem = Reg::UpdateMask<float>(sreg);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
    Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskRem);
    for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
        Reg::Duplicate(y1B32, 0, maskRem);
        Reg::Duplicate(y3B32, 0, maskRem);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskRem);
        Reg::Mul(mul11B32, x11B32, weight11B32, maskRem);
        Reg::Mul(mul12B32, x12B32, weight12B32, maskRem);
        Reg::Mul(mul13B32, x13B32, weight13B32, maskRem);
        Reg::Mul(mul31B32, x31B32, weight11B32, maskRem);
        Reg::Mul(mul32B32, x32B32, weight12B32, maskRem);
        Reg::Mul(mul33B32, x33B32, weight13B32, maskRem);
        Reg::Add(y1B32, y1B32, mul11B32, maskRem);
        Reg::Add(y1B32, y1B32, mul12B32, maskRem);
        Reg::Add(y1B32, y1B32, mul13B32, maskRem);
        Reg::Add(y1B32, y1B32, x13B32, maskRem);
        Reg::Add(y3B32, y3B32, mul31B32, maskRem);
        Reg::Add(y3B32, y3B32, mul32B32, maskRem);
        Reg::Add(y3B32, y3B32, mul33B32, maskRem);
        Reg::Add(y3B32, y3B32, x33B32, maskRem);
        Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
        Reg::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskRem);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoop * dimLen,
                                                          y1B16, maskRem);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + (2 * xLoop + 1) * dimLen, y3B16, maskRem);
    }
    Reg::Duplicate(y1B32, 0, maskRem);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoopNum * dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoopNum + 1) * dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoopNum + 2) * dimLen);
    Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
    Reg::Mul(mul11B32, x11B32, weight11B32, maskRem);
    Reg::Mul(mul12B32, x12B32, weight12B32, maskRem);
    Reg::Mul(mul13B32, x13B32, weight13B32, maskRem);
    Reg::Add(y1B32, y1B32, mul11B32, maskRem);
    Reg::Add(y1B32, y1B32, mul12B32, maskRem);
    Reg::Add(y1B32, y1B32, mul13B32, maskRem);
    Reg::Add(y1B32, y1B32, x13B32, maskRem);
    Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoopNum * dimLen,
                                                      y1B16, maskRem);
}

// 非残差连接模式，不含尾行
template <typename T>
__simd_vf__ void Conv1dNoStateSingleTailNoConNoResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                                     uint32_t xLoopNum, uint32_t dimLen)
{
    Reg::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32, mul13B32,
        y1B32;
    Reg::RegTensor<float> weight21B32, weight22B32, weight23B32, x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32,
        y2B32;
    Reg::RegTensor<float> x31B32, x32B32, x33B32, mul31B32, mul32B32, mul33B32, y3B32;
    Reg::RegTensor<float> x41B32, x42B32, x43B32, mul41B32, mul42B32, mul43B32, y4B32;
    Reg::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    Reg::RegTensor<T> weight21B16, weight22B16, weight23B16, x21B16, x22B16, x23B16, y2B16;
    Reg::RegTensor<T> x31B16, x32B16, x33B16, y3B16;
    Reg::RegTensor<T> x41B16, x42B16, x43B16, y4B16;
    uint8_t dimLoopNum = dimLen / (B32_REP_SIZE * 2);
    uint16_t dimRem = dimLen - dimLoopNum * (B32_REP_SIZE * 2);
    Reg::MaskReg maskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRem;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight21B16, weightAddr + offset + B32_REP_SIZE);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight22B16, weightAddr + offset + B32_REP_SIZE + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight23B16, weightAddr + offset + B32_REP_SIZE + 2 * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskB32);
        for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
            Reg::Duplicate(y1B32, 0, maskB32);
            Reg::Duplicate(y2B32, 0, maskB32);
            Reg::Duplicate(y3B32, 0, maskB32);
            Reg::Duplicate(y4B32, 0, maskB32);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                              xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x41B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x42B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x43B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
            Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskB32);
            Reg::Mul(mul11B32, x11B32, weight11B32, maskB32);
            Reg::Mul(mul12B32, x12B32, weight12B32, maskB32);
            Reg::Mul(mul13B32, x13B32, weight13B32, maskB32);
            Reg::Mul(mul21B32, x21B32, weight21B32, maskB32);
            Reg::Mul(mul22B32, x22B32, weight22B32, maskB32);
            Reg::Mul(mul23B32, x23B32, weight23B32, maskB32);
            Reg::Mul(mul31B32, x31B32, weight11B32, maskB32);
            Reg::Mul(mul32B32, x32B32, weight12B32, maskB32);
            Reg::Mul(mul33B32, x33B32, weight13B32, maskB32);
            Reg::Mul(mul41B32, x41B32, weight21B32, maskB32);
            Reg::Mul(mul42B32, x42B32, weight22B32, maskB32);
            Reg::Mul(mul43B32, x43B32, weight23B32, maskB32);

            Reg::Add(y1B32, y1B32, mul11B32, maskB32);
            Reg::Add(y1B32, y1B32, mul12B32, maskB32);
            Reg::Add(y1B32, y1B32, mul13B32, maskB32);
            Reg::Add(y2B32, y2B32, mul21B32, maskB32);
            Reg::Add(y2B32, y2B32, mul22B32, maskB32);
            Reg::Add(y2B32, y2B32, mul23B32, maskB32);
            Reg::Add(y3B32, y3B32, mul31B32, maskB32);
            Reg::Add(y3B32, y3B32, mul32B32, maskB32);
            Reg::Add(y3B32, y3B32, mul33B32, maskB32);
            Reg::Add(y4B32, y4B32, mul41B32, maskB32);
            Reg::Add(y4B32, y4B32, mul42B32, maskB32);
            Reg::Add(y4B32, y4B32, mul43B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoop * dimLen, y1B16, maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16,
                                                              maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + (2 * xLoop + 1) * dimLen, y3B16,
                                                              maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen,
                                                              y4B16, maskB32);
        }
        offset += 2 * B32_REP_SIZE;
    }
    // dim尾块单寄存器处理
    uint32_t sreg = dimRem;
    maskRem = Reg::UpdateMask<float>(sreg);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
    Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskRem);
    for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
        Reg::Duplicate(y1B32, 0, maskRem);
        Reg::Duplicate(y3B32, 0, maskRem);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskRem);
        Reg::Mul(mul11B32, x11B32, weight11B32, maskRem);
        Reg::Mul(mul12B32, x12B32, weight12B32, maskRem);
        Reg::Mul(mul13B32, x13B32, weight13B32, maskRem);
        Reg::Mul(mul31B32, x31B32, weight11B32, maskRem);
        Reg::Mul(mul32B32, x32B32, weight12B32, maskRem);
        Reg::Mul(mul33B32, x33B32, weight13B32, maskRem);
        Reg::Add(y1B32, y1B32, mul11B32, maskRem);
        Reg::Add(y1B32, y1B32, mul12B32, maskRem);
        Reg::Add(y1B32, y1B32, mul13B32, maskRem);
        Reg::Add(y3B32, y3B32, mul31B32, maskRem);
        Reg::Add(y3B32, y3B32, mul32B32, maskRem);
        Reg::Add(y3B32, y3B32, mul33B32, maskRem);
        Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
        Reg::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskRem);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoop * dimLen,
                                                          y1B16, maskRem);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + (2 * xLoop + 1) * dimLen, y3B16, maskRem);
    }
}

// 非残差连接模式，含尾行
template <typename T>
__simd_vf__ void Conv1dNoStateSingleTailNoConResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
                                                   uint32_t xLoopNum, uint32_t dimLen)
{
    Reg::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32, mul13B32,
        y1B32;
    Reg::RegTensor<float> weight21B32, weight22B32, weight23B32, x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32,
        y2B32;
    Reg::RegTensor<float> x31B32, x32B32, x33B32, mul31B32, mul32B32, mul33B32, y3B32;
    Reg::RegTensor<float> x41B32, x42B32, x43B32, mul41B32, mul42B32, mul43B32, y4B32;
    Reg::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    Reg::RegTensor<T> weight21B16, weight22B16, weight23B16, x21B16, x22B16, x23B16, y2B16;
    Reg::RegTensor<T> x31B16, x32B16, x33B16, y3B16;
    Reg::RegTensor<T> x41B16, x42B16, x43B16, y4B16;
    uint8_t dimLoopNum = dimLen / (B32_REP_SIZE * 2);
    uint16_t dimRem = dimLen - dimLoopNum * (B32_REP_SIZE * 2);
    Reg::MaskReg maskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRem;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskB32);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight21B16, weightAddr + offset + B32_REP_SIZE);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight22B16, weightAddr + offset + B32_REP_SIZE + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight23B16, weightAddr + offset + B32_REP_SIZE + 2 * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskB32);
        for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
            Reg::Duplicate(y1B32, 0, maskB32);
            Reg::Duplicate(y2B32, 0, maskB32);
            Reg::Duplicate(y3B32, 0, maskB32);
            Reg::Duplicate(y4B32, 0, maskB32);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                              xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x41B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x42B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x43B16,
                                                              xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
            Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskB32);
            Reg::Mul(mul11B32, x11B32, weight11B32, maskB32);
            Reg::Mul(mul12B32, x12B32, weight12B32, maskB32);
            Reg::Mul(mul13B32, x13B32, weight13B32, maskB32);
            Reg::Mul(mul21B32, x21B32, weight21B32, maskB32);
            Reg::Mul(mul22B32, x22B32, weight22B32, maskB32);
            Reg::Mul(mul23B32, x23B32, weight23B32, maskB32);
            Reg::Mul(mul31B32, x31B32, weight11B32, maskB32);
            Reg::Mul(mul32B32, x32B32, weight12B32, maskB32);
            Reg::Mul(mul33B32, x33B32, weight13B32, maskB32);
            Reg::Mul(mul41B32, x41B32, weight21B32, maskB32);
            Reg::Mul(mul42B32, x42B32, weight22B32, maskB32);
            Reg::Mul(mul43B32, x43B32, weight23B32, maskB32);

            Reg::Add(y1B32, y1B32, mul11B32, maskB32);
            Reg::Add(y1B32, y1B32, mul12B32, maskB32);
            Reg::Add(y1B32, y1B32, mul13B32, maskB32);
            Reg::Add(y2B32, y2B32, mul21B32, maskB32);
            Reg::Add(y2B32, y2B32, mul22B32, maskB32);
            Reg::Add(y2B32, y2B32, mul23B32, maskB32);
            Reg::Add(y3B32, y3B32, mul31B32, maskB32);
            Reg::Add(y3B32, y3B32, mul32B32, maskB32);
            Reg::Add(y3B32, y3B32, mul33B32, maskB32);
            Reg::Add(y4B32, y4B32, mul41B32, maskB32);
            Reg::Add(y4B32, y4B32, mul42B32, maskB32);
            Reg::Add(y4B32, y4B32, mul43B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
            Reg::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoop * dimLen, y1B16, maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16,
                                                              maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + (2 * xLoop + 1) * dimLen, y3B16,
                                                              maskB32);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen,
                                                              y4B16, maskB32);
        }
        Reg::Duplicate(y1B32, 0, maskB32);
        Reg::Duplicate(y2B32, 0, maskB32);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoopNum * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoopNum + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoopNum + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                          xAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16,
                                                          xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16,
                                                          xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 2) * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
        Reg::Mul(mul11B32, x11B32, weight11B32, maskB32);
        Reg::Mul(mul12B32, x12B32, weight12B32, maskB32);
        Reg::Mul(mul13B32, x13B32, weight13B32, maskB32);
        Reg::Mul(mul21B32, x21B32, weight21B32, maskB32);
        Reg::Mul(mul22B32, x22B32, weight22B32, maskB32);
        Reg::Mul(mul23B32, x23B32, weight23B32, maskB32);
        Reg::Add(y1B32, y1B32, mul11B32, maskB32);
        Reg::Add(y1B32, y1B32, mul12B32, maskB32);
        Reg::Add(y1B32, y1B32, mul13B32, maskB32);
        Reg::Add(y2B32, y2B32, mul21B32, maskB32);
        Reg::Add(y2B32, y2B32, mul22B32, maskB32);
        Reg::Add(y2B32, y2B32, mul23B32, maskB32);
        Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + 2 * xLoopNum * dimLen, y1B16, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen, y2B16,
                                                          maskB32);

        offset += 2 * B32_REP_SIZE;
    }
    // dim尾块单寄存器处理
    uint32_t sreg = dimRem;
    maskRem = Reg::UpdateMask<float>(sreg);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + offset);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + offset + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + offset + 2 * dimLen);
    Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskRem);
    for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
        Reg::Duplicate(y1B32, 0, maskRem);
        Reg::Duplicate(y3B32, 0, maskRem);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskRem);
        Reg::Mul(mul11B32, x11B32, weight11B32, maskRem);
        Reg::Mul(mul12B32, x12B32, weight12B32, maskRem);
        Reg::Mul(mul13B32, x13B32, weight13B32, maskRem);
        Reg::Mul(mul31B32, x31B32, weight11B32, maskRem);
        Reg::Mul(mul32B32, x32B32, weight12B32, maskRem);
        Reg::Mul(mul33B32, x33B32, weight13B32, maskRem);
        Reg::Add(y1B32, y1B32, mul11B32, maskRem);
        Reg::Add(y1B32, y1B32, mul12B32, maskRem);
        Reg::Add(y1B32, y1B32, mul13B32, maskRem);
        Reg::Add(y3B32, y3B32, mul31B32, maskRem);
        Reg::Add(y3B32, y3B32, mul32B32, maskRem);
        Reg::Add(y3B32, y3B32, mul33B32, maskRem);
        Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
        Reg::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskRem);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoop * dimLen,
                                                          y1B16, maskRem);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + (2 * xLoop + 1) * dimLen, y3B16, maskRem);
    }
    Reg::Duplicate(y1B32, 0, maskRem);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoopNum * dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoopNum + 1) * dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoopNum + 2) * dimLen);
    Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
    Reg::Mul(mul11B32, x11B32, weight11B32, maskRem);
    Reg::Mul(mul12B32, x12B32, weight12B32, maskRem);
    Reg::Mul(mul13B32, x13B32, weight13B32, maskRem);
    Reg::Add(y1B32, y1B32, mul11B32, maskRem);
    Reg::Add(y1B32, y1B32, mul12B32, maskRem);
    Reg::Add(y1B32, y1B32, mul13B32, maskRem);
    Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoopNum * dimLen,
                                                      y1B16, maskRem);
}

template <typename T>
__aicore__ inline void Conv1dNoStateSingleTail(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &yUb,
                                               uint32_t xSLen, uint32_t dimLen, int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    uint32_t xLoopNum = xSLen / 2;
    uint8_t xLoopRem = xSLen % 2;
    if (isResidualConnection == 1) {
        if (xLoopRem == 1) {
            Conv1dNoStateSingleTailConResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        } else if (xLoopRem == 0) {
            Conv1dNoStateSingleTailConNoResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        }
    } else {
        if (xLoopRem == 1) {
            Conv1dNoStateSingleTailNoConResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        } else if (xLoopRem == 0) {
            Conv1dNoStateSingleTailNoConNoResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        }
    }
}

#endif
