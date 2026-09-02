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

#ifndef FUSED_CAUSAL_CONV1D_NO_STATE_DOUBLE_TAIL_H
#define FUSED_CAUSAL_CONV1D_NO_STATE_DOUBLE_TAIL_H

#include "kernel_operator.h"
#include "fused_causal_conv1d_state_single_tail.h"

using namespace AscendC;

////残差连接模式，不含尾行
template <typename T>
__simd_vf__ void Conv1dNoStateDoubleTailConNoResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
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
    uint8_t dimLoopNum = dimLen / (B32_REP_SIZE * 2);
    uint8_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE * 2;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRem;
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
    // 尾块非128对齐
    uint32_t sreg = dimRem - B32_REP_SIZE;
    maskRem = MicroAPI::UpdateMask<float>(sreg);
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
    MicroAPI::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskRem);
    for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
        MicroAPI::Duplicate(y1B32, 0, maskB32);
        MicroAPI::Duplicate(y2B32, 0, maskRem);
        MicroAPI::Duplicate(y3B32, 0, maskB32);
        MicroAPI::Duplicate(y4B32, 0, maskRem);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                                    xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x41B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x42B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x43B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskRem);
        MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
        MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
        MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
        MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskRem);
        MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskRem);
        MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskRem);
        MicroAPI::Mul(mul31B32, x31B32, weight11B32, maskB32);
        MicroAPI::Mul(mul32B32, x32B32, weight12B32, maskB32);
        MicroAPI::Mul(mul33B32, x33B32, weight13B32, maskB32);
        MicroAPI::Mul(mul41B32, x41B32, weight21B32, maskRem);
        MicroAPI::Mul(mul42B32, x42B32, weight22B32, maskRem);
        MicroAPI::Mul(mul43B32, x43B32, weight23B32, maskRem);
        MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, x13B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul21B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, mul22B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, mul23B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, x23B32, maskRem);
        MicroAPI::Add(y3B32, y3B32, mul31B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, mul32B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, mul33B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, x33B32, maskB32);
        MicroAPI::Add(y4B32, y4B32, mul41B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, mul42B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, mul43B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, x43B32, maskRem);
        MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
        MicroAPI::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskRem);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoop * dimLen, y1B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16, maskRem);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + (2 * xLoop + 1) * dimLen, y3B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + (2 * xLoop + 1) * dimLen, y4B16, maskRem);
    }
}

// 残差连接模式，含尾行
template <typename T>
__simd_vf__ void Conv1dNoStateDoubleTailConResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
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
    uint8_t dimLoopNum = dimLen / (B32_REP_SIZE * 2);
    uint8_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE * 2;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRem;
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
    // 尾块双寄存器处理
    uint32_t sreg = dimRem - B32_REP_SIZE;
    maskRem = MicroAPI::UpdateMask<float>(sreg);
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
    MicroAPI::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskRem);
    for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
        MicroAPI::Duplicate(y1B32, 0, maskB32);
        MicroAPI::Duplicate(y2B32, 0, maskRem);
        MicroAPI::Duplicate(y3B32, 0, maskB32);
        MicroAPI::Duplicate(y4B32, 0, maskRem);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                                    xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x41B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x42B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x43B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskRem);
        MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
        MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
        MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
        MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskRem);
        MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskRem);
        MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskRem);
        MicroAPI::Mul(mul31B32, x31B32, weight11B32, maskB32);
        MicroAPI::Mul(mul32B32, x32B32, weight12B32, maskB32);
        MicroAPI::Mul(mul33B32, x33B32, weight13B32, maskB32);
        MicroAPI::Mul(mul41B32, x41B32, weight21B32, maskRem);
        MicroAPI::Mul(mul42B32, x42B32, weight22B32, maskRem);
        MicroAPI::Mul(mul43B32, x43B32, weight23B32, maskRem);
        MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, x13B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul21B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, mul22B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, mul23B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, x23B32, maskRem);
        MicroAPI::Add(y3B32, y3B32, mul31B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, mul32B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, mul33B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, x33B32, maskB32);
        MicroAPI::Add(y4B32, y4B32, mul41B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, mul42B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, mul43B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, x43B32, maskRem);
        MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
        MicroAPI::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskRem);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoop * dimLen, y1B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16, maskRem);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + (2 * xLoop + 1) * dimLen, y3B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + (2 * xLoop + 1) * dimLen, y4B16, maskRem);
    }
    MicroAPI::Duplicate(y1B32, 0, maskB32);
    MicroAPI::Duplicate(y2B32, 0, maskRem);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoopNum * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoopNum + 1) * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoopNum + 2) * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                                xAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 1) * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 2) * dimLen);
    MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
    MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
    MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
    MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
    MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
    MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
    MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
    MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskRem);
    MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskRem);
    MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskRem);
    MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
    MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
    MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
    MicroAPI::Add(y1B32, y1B32, x13B32, maskB32);
    MicroAPI::Add(y2B32, y2B32, mul21B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, mul22B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, mul23B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, x23B32, maskRem);
    MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
    MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
        yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoopNum * dimLen, y1B16, maskB32);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
        yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + 2 * xLoopNum * dimLen, y2B16, maskRem);
}

// 非残差连接模式，不含尾行
template <typename T>
__simd_vf__ void Conv1dNoStateDoubleTailNoConNoResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
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
    uint8_t dimLoopNum = dimLen / (B32_REP_SIZE * 2);
    uint8_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE * 2;
    MicroAPI::MaskReg maskRem;
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
    // dim尾块双寄存器处理
    uint32_t sreg = dimRem - B32_REP_SIZE;
    maskRem = MicroAPI::UpdateMask<float>(sreg);
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
    MicroAPI::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskRem);
    for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
        MicroAPI::Duplicate(y1B32, 0, maskB32);
        MicroAPI::Duplicate(y2B32, 0, maskRem);
        MicroAPI::Duplicate(y3B32, 0, maskB32);
        MicroAPI::Duplicate(y4B32, 0, maskRem);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                                    xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x41B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x42B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x43B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskRem);
        MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
        MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
        MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
        MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskRem);
        MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskRem);
        MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskRem);
        MicroAPI::Mul(mul31B32, x31B32, weight11B32, maskB32);
        MicroAPI::Mul(mul32B32, x32B32, weight12B32, maskB32);
        MicroAPI::Mul(mul33B32, x33B32, weight13B32, maskB32);
        MicroAPI::Mul(mul41B32, x41B32, weight21B32, maskRem);
        MicroAPI::Mul(mul42B32, x42B32, weight22B32, maskRem);
        MicroAPI::Mul(mul43B32, x43B32, weight23B32, maskRem);
        MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul21B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, mul22B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, mul23B32, maskRem);
        MicroAPI::Add(y3B32, y3B32, mul31B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, mul32B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, mul33B32, maskB32);
        MicroAPI::Add(y4B32, y4B32, mul41B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, mul42B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, mul43B32, maskRem);
        MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
        MicroAPI::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskRem);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoop * dimLen, y1B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16, maskRem);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + (2 * xLoop + 1) * dimLen, y3B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + (2 * xLoop + 1) * dimLen, y4B16, maskRem);
    }
}

// 非残差连接模式，含尾行
template <typename T>
__simd_vf__ void Conv1dNoStateDoubleTailNoConResVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *yAddr,
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
    uint8_t dimLoopNum = dimLen / (B32_REP_SIZE * 2);
    uint8_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE * 2;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRem;
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
            MicroAPI::Duplicate(y2B32, 0, maskRem);
            MicroAPI::Duplicate(y3B32, 0, maskB32);
            MicroAPI::Duplicate(y4B32, 0, maskRem);
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
        MicroAPI::Duplicate(y2B32, 0, maskRem);
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
    // 尾块双寄存器处理
    uint32_t sreg = dimRem - B32_REP_SIZE;
    maskRem = MicroAPI::UpdateMask<float>(sreg);
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
    MicroAPI::Cast<float, T, castTraitB162B32>(weight21B32, weight21B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight22B32, weight22B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight23B32, weight23B16, maskRem);
    for (uint32_t xLoop = 0; xLoop < xLoopNum; xLoop++) {
        MicroAPI::Duplicate(y1B32, 0, maskB32);
        MicroAPI::Duplicate(y2B32, 0, maskB32);
        MicroAPI::Duplicate(y3B32, 0, maskB32);
        MicroAPI::Duplicate(y4B32, 0, maskB32);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                                    xAddr + offset + B32_REP_SIZE + 2 * xLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x31B16, xAddr + offset + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x32B16, xAddr + offset + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x33B16, xAddr + offset + (2 * xLoop + 3) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x41B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 1) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x42B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 2) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            x43B16, xAddr + offset + B32_REP_SIZE + (2 * xLoop + 3) * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x31B32, x31B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x32B32, x32B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x33B32, x33B16, maskB32);
        MicroAPI::Cast<float, T, castTraitB162B32>(x41B32, x41B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x42B32, x42B16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(x43B32, x43B16, maskRem);
        MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
        MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
        MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
        MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskRem);
        MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskRem);
        MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskRem);
        MicroAPI::Mul(mul31B32, x31B32, weight11B32, maskB32);
        MicroAPI::Mul(mul32B32, x32B32, weight12B32, maskB32);
        MicroAPI::Mul(mul33B32, x33B32, weight13B32, maskB32);
        MicroAPI::Mul(mul41B32, x41B32, weight21B32, maskRem);
        MicroAPI::Mul(mul42B32, x42B32, weight22B32, maskRem);
        MicroAPI::Mul(mul43B32, x43B32, weight23B32, maskRem);
        MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
        MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
        MicroAPI::Add(y2B32, y2B32, mul21B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, mul22B32, maskRem);
        MicroAPI::Add(y2B32, y2B32, mul23B32, maskRem);
        MicroAPI::Add(y3B32, y3B32, mul31B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, mul32B32, maskB32);
        MicroAPI::Add(y3B32, y3B32, mul33B32, maskB32);
        MicroAPI::Add(y4B32, y4B32, mul41B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, mul42B32, maskRem);
        MicroAPI::Add(y4B32, y4B32, mul43B32, maskRem);
        MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
        MicroAPI::Cast<T, float, castTraitB322B16>(y3B16, y3B32, maskB32);
        MicroAPI::Cast<T, float, castTraitB322B16>(y4B16, y4B32, maskRem);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoop * dimLen, y1B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + 2 * xLoop * dimLen, y2B16, maskRem);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + (2 * xLoop + 1) * dimLen, y3B16, maskB32);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
            yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + (2 * xLoop + 1) * dimLen, y4B16, maskRem);
    }
    MicroAPI::Duplicate(y1B32, 0, maskB32);
    MicroAPI::Duplicate(y2B32, 0, maskB32);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, xAddr + offset + 2 * xLoopNum * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, xAddr + offset + (2 * xLoopNum + 1) * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset + (2 * xLoopNum + 2) * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16,
                                                                xAddr + offset + B32_REP_SIZE + 2 * xLoopNum * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        x22B16, xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 1) * dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        x23B16, xAddr + offset + B32_REP_SIZE + (2 * xLoopNum + 2) * dimLen);
    MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
    MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
    MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
    MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
    MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskB32);
    MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskB32);
    MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskB32);
    MicroAPI::Mul(mul21B32, x21B32, weight21B32, maskRem);
    MicroAPI::Mul(mul22B32, x22B32, weight22B32, maskRem);
    MicroAPI::Mul(mul23B32, x23B32, weight23B32, maskRem);
    MicroAPI::Add(y1B32, y1B32, mul11B32, maskB32);
    MicroAPI::Add(y1B32, y1B32, mul12B32, maskB32);
    MicroAPI::Add(y1B32, y1B32, mul13B32, maskB32);
    MicroAPI::Add(y2B32, y2B32, mul21B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, mul22B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, mul23B32, maskRem);
    MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
    MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
        yAddr + dimLoopNum * B32_REP_SIZE * 2 + 2 * xLoopNum * dimLen, y1B16, maskB32);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(
        yAddr + dimLoopNum * B32_REP_SIZE * 2 + B32_REP_SIZE + 2 * xLoopNum * dimLen, y2B16, maskRem);
}

template <typename T>
__aicore__ inline void Conv1dNoStateDoubleTail(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &yUb,
                                               uint32_t xSLen, uint32_t dimLen, int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    uint32_t xLoopNum = xSLen / 2;
    uint8_t xLoopRem = xSLen % 2;
    if (isResidualConnection == 1) {
        if (xLoopRem == 1) {
            Conv1dNoStateDoubleTailConResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        } else if (xLoopRem == 0) {
            Conv1dNoStateDoubleTailConNoResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        }
    } else {
        if (xLoopRem == 1) {
            Conv1dNoStateDoubleTailNoConResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        } else if (xLoopRem == 0) {
            Conv1dNoStateDoubleTailNoConNoResVF(xAddr, weightAddr, yAddr, xLoopNum, dimLen);
        }
    }
}

#endif
