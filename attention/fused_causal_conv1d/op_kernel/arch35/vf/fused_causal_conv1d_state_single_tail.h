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

constexpr Reg::CastTrait castTraitB162B32 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr Reg::CastTrait castTraitB322B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr uint32_t REGSIZE = 256;
constexpr uint32_t B32_REP_SIZE = REGSIZE / sizeof(float);

// 对stateAddr的数据进行原地读写出操作， stateAddr=yAddr
template <typename T>
__simd_vf__ void Conv1dNeedStateSingleTailConVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                                __ubuf__ T *yAddr, uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen)
{
    Reg::RegTensor<float> xB32, mulB32, weightB32, yB32;
    Reg::RegTensor<T> xB16, weightB16, yB16;
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint16_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE;
    Reg::MaskReg maskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRem;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        Reg::Duplicate(yB32, 0, maskB32);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + offset + stateLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
            Reg::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
            Reg::Mul(mulB32, xB32, weightB32, maskB32);
            Reg::Add(yB32, yB32, mulB32, maskB32);
        }
        for (uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                              weightAddr + offset + (xLoop + stateSLen) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
            Reg::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
            Reg::Mul(mulB32, xB32, weightB32, maskB32);
            Reg::Add(yB32, yB32, mulB32, maskB32);
        }
        Reg::Add(yB32, yB32, xB32, maskB32);
        Reg::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, yB16, maskB32);
        offset += B32_REP_SIZE;
    }
    // 尾块非128对齐
    uint32_t sreg = dimRem;
    maskRem = Reg::UpdateMask<float>(sreg);
    Reg::Duplicate(yB32, 0, maskRem);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + offset + stateLoop * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(xB32, xB16, maskRem);
        Reg::Mul(mulB32, xB32, weightB32, maskRem);
        Reg::Add(yB32, yB32, mulB32, maskRem);
    }
    for (uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                          weightAddr + offset + (xLoop + stateSLen) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(xB32, xB16, maskRem);
        Reg::Mul(mulB32, xB32, weightB32, maskRem);
        Reg::Add(yB32, yB32, mulB32, maskRem);
    }
    Reg::Add(yB32, yB32, xB32, maskRem);
    Reg::Cast<T, float, castTraitB322B16>(yB16, yB32, maskRem);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE, yB16, maskRem);
}

template <typename T>
__simd_vf__ void Conv1dNeedStateSingleTailNoConVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                                  __ubuf__ T *yAddr, uint8_t stateSLen, uint8_t xSLen, uint32_t dimLen)
{
    Reg::RegTensor<float> xB32, mulB32, weightB32, yB32;
    Reg::RegTensor<T> xB16, weightB16, yB16;
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint16_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE;
    Reg::MaskReg maskB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRem;
    int32_t offset = 0;
    for (uint8_t dimLoop = 0; dimLoop < dimLoopNum; dimLoop++) {
        Reg::Duplicate(yB32, 0, maskB32);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + offset + stateLoop * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
            Reg::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
            Reg::Mul(mulB32, xB32, weightB32, maskB32);
            Reg::Add(yB32, yB32, mulB32, maskB32);
        }
        for (uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                              weightAddr + offset + (xLoop + stateSLen) * dimLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
            Reg::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskB32);
            Reg::Cast<float, T, castTraitB162B32>(xB32, xB16, maskB32);
            Reg::Mul(mulB32, xB32, weightB32, maskB32);
            Reg::Add(yB32, yB32, mulB32, maskB32);
        }
        Reg::Cast<T, float, castTraitB322B16>(yB16, yB32, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, yB16, maskB32);
        offset += B32_REP_SIZE;
    }
    // 尾块非128对齐
    uint32_t sreg = dimRem;
    maskRem = Reg::UpdateMask<float>(sreg);
    Reg::Duplicate(yB32, 0, maskRem);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint8_t stateLoop = 0; stateLoop < stateSLen; stateLoop++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weightB16, weightAddr + offset + stateLoop * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, stateAddr + offset + stateLoop * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(xB32, xB16, maskRem);
        Reg::Mul(mulB32, xB32, weightB32, maskRem);
        Reg::Add(yB32, yB32, mulB32, maskRem);
    }
    for (uint8_t xLoop = 0; xLoop < xSLen; xLoop++) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weightB16,
                                                          weightAddr + offset + (xLoop + stateSLen) * dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, xAddr + offset + xLoop * dimLen);
        Reg::Cast<float, T, castTraitB162B32>(weightB32, weightB16, maskRem);
        Reg::Cast<float, T, castTraitB162B32>(xB32, xB16, maskRem);
        Reg::Mul(mulB32, xB32, weightB32, maskRem);
        Reg::Add(yB32, yB32, mulB32, maskRem);
    }
    Reg::Cast<T, float, castTraitB322B16>(yB16, yB32, maskRem);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE, yB16, maskRem);
}

template <typename T>
__simd_vf__ void Conv1dNeedStateSingleTailConBHVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                                  __ubuf__ T *yAddr, uint32_t dimLen)
{
    Reg::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32, mul13B32,
        y1B32;
    Reg::RegTensor<float> x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32, y2B32;
    Reg::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    Reg::RegTensor<T> x21B16, x22B16, x23B16, y2B16;
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint16_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE;
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

        Reg::Duplicate(y1B32, 0, maskB32);
        Reg::Duplicate(y2B32, 0, maskB32);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, stateAddr + offset);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, stateAddr + offset + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset);

        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16, stateAddr + offset + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16, xAddr + offset);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16, xAddr + offset + dimLen);

        Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
        Reg::Mul(mul11B32, x11B32, weight11B32, maskB32);
        Reg::Mul(mul12B32, x12B32, weight12B32, maskB32);
        Reg::Mul(mul13B32, x13B32, weight13B32, maskB32);
        Reg::Mul(mul21B32, x21B32, weight11B32, maskB32);
        Reg::Mul(mul22B32, x22B32, weight12B32, maskB32);
        Reg::Mul(mul23B32, x23B32, weight13B32, maskB32);

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
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, y1B16, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + dimLen, y2B16, maskB32);
        offset += B32_REP_SIZE;
    }
    // 尾块非128对齐
    uint32_t sreg = dimRem;
    maskRem = Reg::UpdateMask<float>(sreg);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + dimLoopNum * B32_REP_SIZE);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + dimLoopNum * B32_REP_SIZE + 2 * dimLen);
    Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskRem);

    Reg::Duplicate(y1B32, 0, maskRem);
    Reg::Duplicate(y2B32, 0, maskRem);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, stateAddr + dimLoopNum * B32_REP_SIZE);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, stateAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + dimLoopNum * B32_REP_SIZE);

    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16, stateAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16, xAddr + dimLoopNum * B32_REP_SIZE);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16, xAddr + dimLoopNum * B32_REP_SIZE + dimLen);

    Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
    Reg::Mul(mul11B32, x11B32, weight11B32, maskRem);
    Reg::Mul(mul12B32, x12B32, weight12B32, maskRem);
    Reg::Mul(mul13B32, x13B32, weight13B32, maskRem);
    Reg::Mul(mul21B32, x21B32, weight11B32, maskRem);
    Reg::Mul(mul22B32, x22B32, weight12B32, maskRem);
    Reg::Mul(mul23B32, x23B32, weight13B32, maskRem);

    Reg::Add(y1B32, y1B32, mul11B32, maskRem);
    Reg::Add(y1B32, y1B32, mul12B32, maskRem);
    Reg::Add(y1B32, y1B32, mul13B32, maskRem);
    Reg::Add(y1B32, y1B32, x13B32, maskRem);

    Reg::Add(y2B32, y2B32, mul21B32, maskRem);
    Reg::Add(y2B32, y2B32, mul22B32, maskRem);
    Reg::Add(y2B32, y2B32, mul23B32, maskRem);
    Reg::Add(y2B32, y2B32, x23B32, maskRem);

    Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
    Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE, y1B16, maskRem);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE + dimLen, y2B16, maskRem);
}

template <typename T>
__simd_vf__ void Conv1dNeedStateSingleTailNoConBHVF(__ubuf__ T *xAddr, __ubuf__ T *weightAddr, __ubuf__ T *stateAddr,
                                                    __ubuf__ T *yAddr, uint32_t dimLen)
{
    Reg::RegTensor<float> weight11B32, weight12B32, weight13B32, x11B32, x12B32, x13B32, mul11B32, mul12B32, mul13B32,
        y1B32;
    Reg::RegTensor<float> x21B32, x22B32, x23B32, mul21B32, mul22B32, mul23B32, y2B32;
    Reg::RegTensor<T> weight11B16, weight12B16, weight13B16, x11B16, x12B16, x13B16, y1B16;
    Reg::RegTensor<T> x21B16, x22B16, x23B16, y2B16;
    uint8_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint16_t dimRem = dimLen - dimLoopNum * B32_REP_SIZE;
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

        Reg::Duplicate(y1B32, 0, maskB32);
        Reg::Duplicate(y2B32, 0, maskB32);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, stateAddr + offset);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, stateAddr + offset + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + offset);

        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16, stateAddr + offset + dimLen);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16, xAddr + offset);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16, xAddr + offset + dimLen);

        Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskB32);
        Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskB32);
        Reg::Mul(mul11B32, x11B32, weight11B32, maskB32);
        Reg::Mul(mul12B32, x12B32, weight12B32, maskB32);
        Reg::Mul(mul13B32, x13B32, weight13B32, maskB32);
        Reg::Mul(mul21B32, x21B32, weight11B32, maskB32);
        Reg::Mul(mul22B32, x22B32, weight12B32, maskB32);
        Reg::Mul(mul23B32, x23B32, weight13B32, maskB32);

        Reg::Add(y1B32, y1B32, mul11B32, maskB32);
        Reg::Add(y1B32, y1B32, mul12B32, maskB32);
        Reg::Add(y1B32, y1B32, mul13B32, maskB32);

        Reg::Add(y2B32, y2B32, mul21B32, maskB32);
        Reg::Add(y2B32, y2B32, mul22B32, maskB32);
        Reg::Add(y2B32, y2B32, mul23B32, maskB32);

        Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskB32);
        Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset, y1B16, maskB32);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + offset + dimLen, y2B16, maskB32);
        offset += B32_REP_SIZE;
    }
    // 尾块非128对齐
    uint32_t sreg = dimRem;
    maskRem = Reg::UpdateMask<float>(sreg);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight11B16, weightAddr + dimLoopNum * B32_REP_SIZE);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight12B16, weightAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(weight13B16, weightAddr + dimLoopNum * B32_REP_SIZE + 2 * dimLen);
    Reg::Cast<float, T, castTraitB162B32>(weight11B32, weight11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight12B32, weight12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(weight13B32, weight13B16, maskRem);

    Reg::Duplicate(y1B32, 0, maskRem);
    Reg::Duplicate(y2B32, 0, maskRem);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x11B16, stateAddr + dimLoopNum * B32_REP_SIZE);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x12B16, stateAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x13B16, xAddr + dimLoopNum * B32_REP_SIZE);

    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x21B16, stateAddr + dimLoopNum * B32_REP_SIZE + dimLen);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x22B16, xAddr + dimLoopNum * B32_REP_SIZE);
    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x23B16, xAddr + dimLoopNum * B32_REP_SIZE + dimLen);

    Reg::Cast<float, T, castTraitB162B32>(x11B32, x11B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x12B32, x12B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x13B32, x13B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x21B32, x21B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x22B32, x22B16, maskRem);
    Reg::Cast<float, T, castTraitB162B32>(x23B32, x23B16, maskRem);
    Reg::Mul(mul11B32, x11B32, weight11B32, maskRem);
    Reg::Mul(mul12B32, x12B32, weight12B32, maskRem);
    Reg::Mul(mul13B32, x13B32, weight13B32, maskRem);
    Reg::Mul(mul21B32, x21B32, weight11B32, maskRem);
    Reg::Mul(mul22B32, x22B32, weight12B32, maskRem);
    Reg::Mul(mul23B32, x23B32, weight13B32, maskRem);

    Reg::Add(y1B32, y1B32, mul11B32, maskRem);
    Reg::Add(y1B32, y1B32, mul12B32, maskRem);
    Reg::Add(y1B32, y1B32, mul13B32, maskRem);

    Reg::Add(y2B32, y2B32, mul21B32, maskRem);
    Reg::Add(y2B32, y2B32, mul22B32, maskRem);
    Reg::Add(y2B32, y2B32, mul23B32, maskRem);

    Reg::Cast<T, float, castTraitB322B16>(y1B16, y1B32, maskRem);
    Reg::Cast<T, float, castTraitB322B16>(y2B16, y2B32, maskRem);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE, y1B16, maskRem);
    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + dimLoopNum * B32_REP_SIZE + dimLen, y2B16, maskRem);
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
