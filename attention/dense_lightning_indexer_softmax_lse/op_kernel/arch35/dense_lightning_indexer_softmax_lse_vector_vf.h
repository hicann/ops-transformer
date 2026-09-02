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
 * \file dense_lightning_indexer_softmax_lse_vector_vf.h
 * \brief A5架构适配的SIMD_VF版本Vector辅助头文件
 */

#ifndef __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_VF_H__
#define __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_VF_H__

#include "kernel_operator.h"

namespace DenseLISoftmaxLseVF {
using namespace AscendC;
using namespace MicroAPI;

constexpr static CastTrait castTraitB162B32Even = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

template <typename T, bool outerGidxZero>
__simd_vf__ inline void DoScaleVF(__ubuf__ float *reduceCacheBuf, __ubuf__ float *mmOutUb, __ubuf__ float *weightsUb,
                                  __ubuf__ T *weightsTUb, int64_t groupInner, int64_t s2Inner, uint64_t repeatTimes,
                                  uint64_t countPerRepeat)
{
    RegTensor<float> mmReg;
    RegTensor<float> weightReg;
    RegTensor<float> brcReg;
    RegTensor<float> tmpReg;
    RegTensor<float> reduceReg;
    RegTensor<half> weightHalfReg;
    RegTensor<bfloat16_t> weightBfReg;

    MaskReg pregAllB32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregWeightB16 = CreateMask<T, MaskPattern::ALL>();

    for (int32_t i = 0; i < groupInner; i++) {
        if constexpr (IsSameType<T, half>::value) {
            AscendC::MicroAPI::LoadAlign<T, LoadDist::DIST_BRC_B16>(weightHalfReg, weightsTUb + i);
            AscendC::MicroAPI::Cast<float, T, castTraitB162B32Even>(weightReg, weightHalfReg, pregWeightB16);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            AscendC::MicroAPI::LoadAlign<T, LoadDist::DIST_BRC_B16>(weightBfReg, weightsTUb + i);
            AscendC::MicroAPI::Cast<float, T, castTraitB162B32Even>(weightReg, weightBfReg, pregWeightB16);
        } else {
            AscendC::MicroAPI::LoadAlign<T, LoadDist::DIST_BRC_B32>(weightReg, weightsUb + i);
        }

        for (int32_t j = 0; j < repeatTimes; j++) {
            uint32_t offset = i * s2Inner + j * countPerRepeat;
            LoadAlign(mmReg, mmOutUb + offset);
            Mul(tmpReg, mmReg, weightReg, pregAllB32);

            if constexpr (outerGidxZero) {
                AscendC::MicroAPI::Duplicate(reduceReg, 0.0f);
            } else {
                AscendC::MicroAPI::LoadAlign(reduceReg, reduceCacheBuf + offset);
            }
            Add(reduceReg, reduceReg, tmpReg, pregAllB32);
            StoreAlign(reduceCacheBuf + offset, reduceReg, pregAllB32);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

__simd_vf__ inline void DoReduceSumBlockVF(__ubuf__ float *dstTensor, __ubuf__ float *srcTensor,
                                           __ubuf__ float *maxValueTensor, __ubuf__ float *prevSumTensor,
                                           uint32_t count)
{
    RegTensor<float> srcReg;
    RegTensor<float> dstReg;
    RegTensor<float> maxReg;
    RegTensor<float> expReg;
    RegTensor<float> prevReg;

    MaskReg pregTail = UpdateMask<float>(count);

    LoadAlign(maxReg, maxValueTensor);
    LoadAlign(prevReg, prevSumTensor);
    LoadAlign(srcReg, srcTensor);
    ExpSub(expReg, srcReg, maxReg, pregTail);
    Add(dstReg, expReg, prevReg, pregTail);
    StoreAlign(dstTensor, dstReg, pregTail);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

} // namespace DenseLISoftmaxLseVF

#endif // __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_VF_H__
