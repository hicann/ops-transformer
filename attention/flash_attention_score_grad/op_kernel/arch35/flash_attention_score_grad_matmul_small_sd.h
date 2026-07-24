/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

/*!
 * \file flash_attention_score_grad_matmul_small_sd.h
 * \brief Fixed single-Mmad wrappers for the SmallSD Cube path.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_MATMUL_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_MATMUL_SMALL_SD_H

#include "cube_api/matmul.h"

namespace FagBaseApi {

template <typename INPUT_TYPE, typename CALC_TYPE, uint32_t BASE_M, uint32_t BASE_N, uint32_t BASE_K,
          bool LEFT_TRANSPOSE, bool RIGHT_TRANSPOSE, ABLayout A_LAYOUT, ABLayout B_LAYOUT>
__aicore__ inline void SmallSDMmadOnce(const LocalTensor<INPUT_TYPE> &aTensor,
                                       const LocalTensor<INPUT_TYPE> &bTensor,
                                       MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC> &l0aBuf,
                                       MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC> &l0bBuf,
                                       const LocalTensor<CALC_TYPE> &cTensor, uint32_t m, uint32_t n, uint32_t k)
{
    MMParam param = {
        m,
        n,
        k,
        LEFT_TRANSPOSE,
        RIGHT_TRANSPOSE,
        true,
        true,
        UNITFLAG_EN_OUTER_LAST
    };
    MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, BASE_M, BASE_N, BASE_K, A_LAYOUT, B_LAYOUT>(
        aTensor, bTensor, l0aBuf, l0bBuf, cTensor, param);
}

template <typename INPUT_TYPE, typename CALC_TYPE, uint32_t BASE_M, uint32_t BASE_N, uint32_t HEAD_DIM>
__aicore__ inline void SmallSDMmadDyV(const LocalTensor<INPUT_TYPE> &dyTensor,
                                      const LocalTensor<INPUT_TYPE> &vTensor,
                                      MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC> &l0aBuf,
                                      MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC> &l0bBuf,
                                      const LocalTensor<CALC_TYPE> &cTensor, uint32_t s1, uint32_t s2, uint32_t dv)
{
    SmallSDMmadOnce<INPUT_TYPE, CALC_TYPE, BASE_M, BASE_N, HEAD_DIM, false, true, ABLayout::MK, ABLayout::KN>(
        dyTensor, vTensor, l0aBuf, l0bBuf, cTensor, s1, s2, dv);
}

template <typename INPUT_TYPE, typename CALC_TYPE, uint32_t BASE_M, uint32_t BASE_N, uint32_t HEAD_DIM>
__aicore__ inline void SmallSDMmadQK(const LocalTensor<INPUT_TYPE> &qTensor,
                                     const LocalTensor<INPUT_TYPE> &kTensor,
                                     MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC> &l0aBuf,
                                     MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC> &l0bBuf,
                                     const LocalTensor<CALC_TYPE> &cTensor, uint32_t s1, uint32_t s2, uint32_t d)
{
    SmallSDMmadOnce<INPUT_TYPE, CALC_TYPE, BASE_M, BASE_N, HEAD_DIM, false, true, ABLayout::MK, ABLayout::KN>(
        qTensor, kTensor, l0aBuf, l0bBuf, cTensor, s1, s2, d);
}

template <typename INPUT_TYPE, typename CALC_TYPE, uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void SmallSDMmadDsK(const LocalTensor<INPUT_TYPE> &dsTensor,
                                      const LocalTensor<INPUT_TYPE> &kTensor,
                                      MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC> &l0aBuf,
                                      MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC> &l0bBuf,
                                      const LocalTensor<CALC_TYPE> &cTensor, uint32_t s1, uint32_t d, uint32_t s2)
{
    SmallSDMmadOnce<INPUT_TYPE, CALC_TYPE, BASE_M, BASE_N, BASE_N, false, false, ABLayout::MK, ABLayout::KN>(
        dsTensor, kTensor, l0aBuf, l0bBuf, cTensor, s1, d, s2);
}

template <typename INPUT_TYPE, typename CALC_TYPE, uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void SmallSDMmadDsTQ(const LocalTensor<INPUT_TYPE> &dsTensor,
                                       const LocalTensor<INPUT_TYPE> &qTensor,
                                       MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC> &l0aBuf,
                                       MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC> &l0bBuf,
                                       const LocalTensor<CALC_TYPE> &cTensor, uint32_t s2, uint32_t d, uint32_t s1)
{
    SmallSDMmadOnce<INPUT_TYPE, CALC_TYPE, BASE_M, BASE_N, BASE_N, true, false, ABLayout::MK, ABLayout::KN>(
        dsTensor, qTensor, l0aBuf, l0bBuf, cTensor, s2, d, s1);
}

template <typename INPUT_TYPE, typename CALC_TYPE, uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void SmallSDMmadPTDy(const LocalTensor<INPUT_TYPE> &pTensor,
                                       const LocalTensor<INPUT_TYPE> &dyTensor,
                                       MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC> &l0aBuf,
                                       MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC> &l0bBuf,
                                       const LocalTensor<CALC_TYPE> &cTensor, uint32_t s2, uint32_t dv, uint32_t s1)
{
    SmallSDMmadOnce<INPUT_TYPE, CALC_TYPE, BASE_M, BASE_N, BASE_N, true, false, ABLayout::MK, ABLayout::KN>(
        pTensor, dyTensor, l0aBuf, l0bBuf, cTensor, s2, dv, s1);
}

} // namespace FagBaseApi

#endif
