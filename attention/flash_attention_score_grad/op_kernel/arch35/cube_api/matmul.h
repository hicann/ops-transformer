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
 * \file matmul.h
 * \brief Local mutex matmul overrides for FAG arch35 cube path.
 */
#ifndef MATMUL_MUTEX_H
#define MATMUL_MUTEX_H

#include "../../../../common/op_kernel/matmul.h"
#include "mutex_buffers_policy.h"

namespace fa_base_matmul {

// Mutex override of MatmulFull for L0A/L0B ping-pong buffers.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t, typename L0ADType = A,
          typename L0BDType = B>
__aicore__ inline void MatmulFullMutex(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                       MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC>& aL0BuffsDb,
                                       MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC>& bL0BuffsDb,
                                       const LocalTensor<C>& cL0Tensor, struct MMParam& param,
                                       const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                                       const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<AScaleType>())
{
    auto& l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.LockProd();
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK, param.singleM);
    } else
#endif
    {
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    }
    l0aBuffer.UnlockProd();

    auto& l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.LockProd();
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, 0, param.singleK, param.singleN);
    } else
#endif
    {
        LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }
    l0bBuffer.UnlockProd();

    l0aBuffer.LockCons();
    l0bBuffer.LockCons();

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }

    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    l0aBuffer.UnlockCons();
    l0bBuffer.UnlockCons();
}

// Mutex override of MatmulK for L0A/L0B ping-pong buffers.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t, typename L0ADType = A,
          typename L0BDType = B>
__aicore__ inline void MatmulKMutex(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                    MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC>& aL0BuffsDb,
                                    MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC>& bL0BuffsDb,
                                    const LocalTensor<C>& cL0Tensor, const MMParam& param,
                                    const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                                    const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
        L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
    }
    if constexpr (IsSameType<A, float>::value) {
        L1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        auto& l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.LockProd();
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, k * L1Aoffset, tileK,
                                         param.singleM);
        } else
#endif
        {
            LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        }
        l0aBuffer.UnlockProd();

        auto& l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.LockProd();
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, k * L1Boffset, tileK,
                                         param.singleN, loopNum);
        } else
#endif
        {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        }
        l0bBuffer.UnlockProd();

        l0aBuffer.LockCons();
        l0bBuffer.LockCons();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                      UNITFLAG_EN_OUTER_LAST :
                                      UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        l0aBuffer.UnlockCons();
        l0bBuffer.UnlockCons();
    }
}

// Mutex override of MatmulN for L0A/L0B ping-pong buffers.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t, typename L0ADType = A,
          typename L0BDType = B>
__aicore__ inline void MatmulNMutex(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                    MutexBuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC>& aL0BuffsDb,
                                    MutexBuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC>& bL0BuffsDb,
                                    const LocalTensor<C>& cL0Tensor, const MMParam& param,
                                    const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                                    const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN;
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t L1Boffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        L1Boffset = ((param.singleK + 31) >> 5 << 5) * baseN;
    }
#endif
    uint64_t L0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        L0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    auto& l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.LockProd();
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK, param.singleM);
    } else
#endif
    {
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    }
    l0aBuffer.UnlockProd();

    // L0A is loaded once and consumed by every n-iteration, so acquire the
    // consumer lock a single time around the whole loop (matching the original
    // event-based semantics where M_MTE1 is only set once after the loop).
    l0aBuffer.LockCons();

    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;

        auto& l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.LockProd();
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, n * L1Boffset, param.singleK,
                                         tileN, loopNum);
        } else
#endif
        {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN, loopNum);
        }
        l0bBuffer.UnlockProd();

        l0bBuffer.LockCons();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = tileN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = param.unitFlag;
        Mmad(cL0Tensor[n * L0Coffset], L0ATensor, L0BTensor, mmadParams);

        l0bBuffer.UnlockCons();
    }

    l0aBuffer.UnlockCons();
}

} // namespace fa_base_matmul

#endif
