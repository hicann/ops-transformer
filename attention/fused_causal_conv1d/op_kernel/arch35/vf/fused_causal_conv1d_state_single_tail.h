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

#ifndef FUSED_CAUSAL_CONV1D_STATE_SINGLE_TAIL_H
#define FUSED_CAUSAL_CONV1D_STATE_SINGLE_TAIL_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr uint32_t REGSIZE = 256;
constexpr uint32_t B32_REP_SIZE = REGSIZE / sizeof(float);

// 对stateAddr的数据进行原地读写出操作， stateAddr=yAddr
template <typename T>
__simd_vf__ void Conv1dNeedStateSingleTailConVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                                __ubuf__ T *yAddr, uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> xB32, mulB32, weightB32, yB32;
    MicroAPI::RegTensor<T> xB16, weightB16, yB16;
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint16_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRem;
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
    // 尾块非128对齐
    uint32_t sreg = dimRem;
    maskRem = MicroAPI::UpdateMask<float>(sreg);
    MicroAPI::Duplicate(yB32, 0, maskRem);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    for (uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                                    weightAddr + offset + stateLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskRem);
        MicroAPI::Mul(mulB32, xB32, weightB32, maskRem);
        MicroAPI::Add(yB32, yB32, mulB32, maskRem);
    }
    for (uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                                    weightAddr + offset + (xLoop + stateSLen) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskRem);
        MicroAPI::Mul(mulB32, xB32, weightB32, maskRem);
        MicroAPI::Add(yB32, yB32, mulB32, maskRem);
    }
    MicroAPI::Add(yB32, yB32, xB32, maskRem);
    MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskRem);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE, yB16, maskRem);
}

template <typename T>
__simd_vf__ void Conv1dNeedStateSingleTailNoConVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                                  __ubuf__ T *yAddr, uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> xB32, mulB32, weightB32, yB32;
    MicroAPI::RegTensor<T> xB16, weightB16, yB16;
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint16_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE;
    MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRem;
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
    // 尾块非128对齐
    uint32_t sreg = dimRem;
    maskRem = MicroAPI::UpdateMask<float>(sreg);
    MicroAPI::Duplicate(yB32, 0, maskRem);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    for (uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                                    weightAddr + offset + stateLoop * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskRem);
        MicroAPI::Mul(mulB32, xB32, weightB32, maskRem);
        MicroAPI::Add(yB32, yB32, mulB32, maskRem);
    }
    for (uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                                    weightAddr + offset + (xLoop + stateSLen) * dimLen);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
        MicroAPI::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskRem);
        MicroAPI::Cast<float, T, castTraitB162B32>(xB32, xB16, maskRem);
        MicroAPI::Mul(mulB32, xB32, weightB32, maskRem);
        MicroAPI::Add(yB32, yB32, mulB32, maskRem);
    }
    MicroAPI::Cast<T, float, castTraitB322B16>(yB16, yB32, maskRem);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE, yB16, maskRem);
}

template <typename T>
__simd_vf__ void Conv1dNeedStateSingleTailConBHVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                                  __ubuf__ T *yAddr, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32,
        mul13B32, y1B32;
    MicroAPI::RegTensor<float> x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32, y2B32;
    MicroAPI::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    MicroAPI::RegTensor<T> x21B16, x22B16, x23B16, y2B16;
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint16_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE;
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
    // 尾块非128对齐
    uint32_t sreg = dimRem;
    maskRem = MicroAPI::UpdateMask<float>(sreg);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + dimLoopNum * B32_REP_SIZE);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight12B16,
                                                                weightAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight13B16,
                                                                weightAddr + dimLoopNum * B32_REP_SIZE + 2 * dimLen);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskRem);

    MicroAPI::Duplicate(y1B32, 0, maskRem);
    MicroAPI::Duplicate(y2B32, 0, maskRem);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, stateAddr + dimLoopNum * B32_REP_SIZE);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, stateAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + dimLoopNum * B32_REP_SIZE);

    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16, stateAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x22B16, xAddr + dimLoopNum * B32_REP_SIZE);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x23B16, xAddr + dimLoopNum * B32_REP_SIZE + dimLen);

    MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
    MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskRem);
    MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskRem);
    MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskRem);
    MicroAPI::Mul(mul21B32, x21B32, weight11B32, maskRem);
    MicroAPI::Mul(mul22B32, x22B32, weight12B32, maskRem);
    MicroAPI::Mul(mul23B32, x23B32, weight13B32, maskRem);

    MicroAPI::Add(y1B32, y1B32, mul11B32, maskRem);
    MicroAPI::Add(y1B32, y1B32, mul12B32, maskRem);
    MicroAPI::Add(y1B32, y1B32, mul13B32, maskRem);
    MicroAPI::Add(y1B32, y1B32, x13B32, maskRem);

    MicroAPI::Add(y2B32, y2B32, mul21B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, mul22B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, mul23B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, x23B32, maskRem);

    MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
    MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE, y1B16, maskRem);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE + dimLen, y2B16,
                                                                maskRem);
}

template <typename T>
__simd_vf__ void Conv1dNeedStateSingleTailNoConBHVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                                    __ubuf__ T *yAddr, uint32_t dimLen)
{
    MicroAPI::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32,
        mul13B32, y1B32;
    MicroAPI::RegTensor<float> x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32, y2B32;
    MicroAPI::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    MicroAPI::RegTensor<T> x21B16, x22B16, x23B16, y2B16;
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint16_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE;
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
    // 尾块非128对齐
    uint32_t sreg = dimRem;
    maskRem = MicroAPI::UpdateMask<float>(sreg);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + dimLoopNum * B32_REP_SIZE);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight12B16,
                                                                weightAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(weight13B16,
                                                                weightAddr + dimLoopNum * B32_REP_SIZE + 2 * dimLen);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskRem);

    MicroAPI::Duplicate(y1B32, 0, maskRem);
    MicroAPI::Duplicate(y2B32, 0, maskRem);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x11B16, stateAddr + dimLoopNum * B32_REP_SIZE);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x12B16, stateAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + dimLoopNum * B32_REP_SIZE);

    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x21B16, stateAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x22B16, xAddr + dimLoopNum * B32_REP_SIZE);
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(x23B16, xAddr + dimLoopNum * B32_REP_SIZE + dimLen);

    MicroAPI::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
    MicroAPI::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
    MicroAPI::Mul(mul11B32, x11B32, weight11B32, maskRem);
    MicroAPI::Mul(mul12B32, x12B32, weight12B32, maskRem);
    MicroAPI::Mul(mul13B32, x13B32, weight13B32, maskRem);
    MicroAPI::Mul(mul21B32, x21B32, weight11B32, maskRem);
    MicroAPI::Mul(mul22B32, x22B32, weight12B32, maskRem);
    MicroAPI::Mul(mul23B32, x23B32, weight13B32, maskRem);

    MicroAPI::Add(y1B32, y1B32, mul11B32, maskRem);
    MicroAPI::Add(y1B32, y1B32, mul12B32, maskRem);
    MicroAPI::Add(y1B32, y1B32, mul13B32, maskRem);

    MicroAPI::Add(y2B32, y2B32, mul21B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, mul22B32, maskRem);
    MicroAPI::Add(y2B32, y2B32, mul23B32, maskRem);

    MicroAPI::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
    MicroAPI::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE, y1B16, maskRem);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE + dimLen, y2B16,
                                                                maskRem);
}

template <typename T>
__aicore__ inline void Conv1dNeedStateSingleTail(LocalTensor<T> &xUb, LocalTensor<T> &weightUb, LocalTensor<T> &stateUb,
                                                 LocalTensor<T> &yUb, uint8_t stateSLen, uint32_t xSLen,
                                                 uint32_t dimLen, int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *stateAddr = (__ubuf__ T *)stateUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();
    if (isResidualConnection == 1) {
        Conv1dNeedStateSingleTailConVF(xAddr, weightAddr, stateAddr, yAddr, stateSLen, xSLen, dimLen);
    } else {
        Conv1dNeedStateSingleTailNoConVF(xAddr, weightAddr, stateAddr, yAddr, stateSLen, xSLen, dimLen);
    }
}

template <typename T>
__aicore__ inline void Conv1dNeedStateSingleTailBH(LocalTensor<T> &xUb, LocalTensor<T> &weightUb,
                                                   LocalTensor<T> &stateUb, LocalTensor<T> &yUb, uint32_t convStateLen,
                                                   uint32_t dimLen, int32_t isResidualConnection)
{
    __ubuf__ T *xAddr = (__ubuf__ T *)xUb.GetPhyAddr();
    __ubuf__ T *weightAddr = (__ubuf__ T *)weightUb.GetPhyAddr();
    __ubuf__ T *stateAddr = (__ubuf__ T *)stateUb.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yUb.GetPhyAddr();

    if (isResidualConnection == 1) {
        if (convStateLen == 2) {
            Conv1dNeedStateSingleTailConBHVF(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        } else {
            uint32_t stateSLen = 2;
            uint32_t xSLen = 1;
            Conv1dNeedStateSingleTailConVF(xAddr, weightAddr, stateAddr, yAddr, stateSLen, xSLen, dimLen);
        }
    } else {
        if (convStateLen == 2) {
            Conv1dNeedStateSingleTailNoConBHVF(xAddr, weightAddr, stateAddr, yAddr, dimLen);
        } else {
            uint32_t stateSLen = 2;
            uint32_t xSLen = 1;
            Conv1dNeedStateSingleTailNoConVF(xAddr, weightAddr, stateAddr, yAddr, stateSLen, xSLen, dimLen);
        }
    }
}

#endif
