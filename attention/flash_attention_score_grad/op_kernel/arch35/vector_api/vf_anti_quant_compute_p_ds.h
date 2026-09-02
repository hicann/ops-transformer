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
 * \file vf_anti_quant_compute_p_ds.h
 */
#ifndef MY_ANTI_QUANT_COMPUTE_P_DS_INTERFACE_H_
#define MY_ANTI_QUANT_COMPUTE_P_DS_INTERFACE_H_
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
constexpr static AscendC::MicroAPI::CastTrait castTraitB322B8Three = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitB322B8Two = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitB322B8One = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitB322B8Zero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

template <bool doubleLine>
__aicore__ inline void ComputePDS_VF_unalign(uint64_t sp, uint64_t dpds, uint64_t perm_ub, uint64_t max, uint64_t sum,
                                             uint64_t d, uint16_t srcM, uint16_t srcN, float ss, float dps,
                                             float pscale, float dscale)
{
    float dscale_neg = (float)(-1.0) * dscale;
    uint32_t tailN32 = static_cast<uint32_t>(srcN);
    uint32_t tailN8 = static_cast<uint32_t>(srcN * 4);
    uint32_t tailN16 = static_cast<uint32_t>(srcN * 2);
    __VEC_SCOPE__
    {
        MaskReg preg_all = UpdateMask<float>(tailN32);
        MaskReg preg_all8 = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();

        RegTensor<float> vreg_sp1, vreg_sp2, vreg_sp3, vreg_sp4;
        RegTensor<float> vreg_dps1, vreg_dps2, vreg_dps3, vreg_dps4;
        RegTensor<hifloat8_t> vreg_p1, vreg_p2, vreg_p3, vreg_p4;
        RegTensor<hifloat8_t> vreg_ds1, vreg_ds2, vreg_ds3, vreg_ds4;
        RegTensor<float> vreg_d, vreg_max, vreg_sum, vreg_dps, vreg_ps;
        RegTensor<uint16_t> vreg_perm;
        DataCopy(vreg_d, (__ubuf__ float *)d);
        DataCopy(vreg_max, (__ubuf__ float *)max);
        DataCopy(vreg_sum, (__ubuf__ float *)sum);
        DataCopy(vreg_perm, (__ubuf__ uint16_t *)perm_ub);

        // (max + log(sum) + log(ps)
        Duplicate(vreg_ps, pscale, preg_all);
        Log(vreg_ps, vreg_ps, preg_all);

        Log(vreg_sum, vreg_sum, preg_all);
        Add(vreg_max, vreg_max, vreg_sum, preg_all);
        Add(vreg_max, vreg_max, vreg_ps, preg_all);

        Muls(vreg_d, vreg_d, dscale_neg, preg_all);
        Duplicate(vreg_dps, dps, preg_all);
        Muls(vreg_dps, vreg_dps, dscale, preg_all);
        for (uint16_t i = 0; i < srcM; i += 8) {
            DataCopy(vreg_sp1, (__ubuf__ float *)sp + (i + 0) * 128);
            DataCopy(vreg_sp2, (__ubuf__ float *)sp + (i + 1) * 128);
            DataCopy(vreg_sp3, (__ubuf__ float *)sp + (i + 2) * 128);
            DataCopy(vreg_sp4, (__ubuf__ float *)sp + (i + 3) * 128);

            Muls(vreg_sp1, vreg_sp1, ss, preg_all);
            Muls(vreg_sp2, vreg_sp2, ss, preg_all);
            Muls(vreg_sp3, vreg_sp3, ss, preg_all);
            Muls(vreg_sp4, vreg_sp4, ss, preg_all);

            ExpSub(vreg_sp1, vreg_sp1, vreg_max, preg_all);
            ExpSub(vreg_sp2, vreg_sp2, vreg_max, preg_all);
            ExpSub(vreg_sp3, vreg_sp3, vreg_max, preg_all);
            ExpSub(vreg_sp4, vreg_sp4, vreg_max, preg_all);

            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_p1, vreg_sp1, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_p2, vreg_sp2, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_p3, vreg_sp3, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_p4, vreg_sp4, preg_all);

            Or((RegTensor<uint8_t> &)vreg_p1, (RegTensor<uint8_t> &)vreg_p1, (RegTensor<uint8_t> &)vreg_p3, preg_all8);
            Or((RegTensor<uint8_t> &)vreg_p2, (RegTensor<uint8_t> &)vreg_p2, (RegTensor<uint8_t> &)vreg_p4, preg_all8);

            Interleave((RegTensor<half> &)vreg_p1, (RegTensor<half> &)vreg_p2, (RegTensor<half> &)vreg_p1,
                       (RegTensor<half> &)vreg_p2);

            DataCopyScatter((__ubuf__ uint8_t *)sp + (i + 0) * 512, (RegTensor<uint8_t> &)vreg_p1, vreg_perm,
                            preg_all8);
            DataCopyScatter((__ubuf__ uint8_t *)sp + (i + 1) * 512, (RegTensor<uint8_t> &)vreg_p2, vreg_perm,
                            preg_all8);

            DataCopy(vreg_dps1, (__ubuf__ float *)dpds + (i + 0) * 128);
            DataCopy(vreg_dps2, (__ubuf__ float *)dpds + (i + 1) * 128);
            DataCopy(vreg_dps3, (__ubuf__ float *)dpds + (i + 2) * 128);
            DataCopy(vreg_dps4, (__ubuf__ float *)dpds + (i + 3) * 128);

            FusedMulDstAdd(vreg_dps1, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps2, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps3, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps4, vreg_dps, vreg_d, preg_all);

            Mul(vreg_dps1, vreg_dps1, vreg_sp1, preg_all);
            Mul(vreg_dps2, vreg_dps2, vreg_sp2, preg_all);
            Mul(vreg_dps3, vreg_dps3, vreg_sp3, preg_all);
            Mul(vreg_dps4, vreg_dps4, vreg_sp4, preg_all);

            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_ds1, vreg_dps1, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_ds2, vreg_dps2, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_ds3, vreg_dps3, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_ds4, vreg_dps4, preg_all);

            Or((RegTensor<uint8_t> &)vreg_ds1, (RegTensor<uint8_t> &)vreg_ds1, (RegTensor<uint8_t> &)vreg_ds3,
               preg_all8);
            Or((RegTensor<uint8_t> &)vreg_ds2, (RegTensor<uint8_t> &)vreg_ds2, (RegTensor<uint8_t> &)vreg_ds4,
               preg_all8);
            Interleave((RegTensor<half> &)vreg_ds1, (RegTensor<half> &)vreg_ds2, (RegTensor<half> &)vreg_ds1,
                       (RegTensor<half> &)vreg_ds2);

            DataCopyScatter((__ubuf__ uint8_t *)dpds + (i + 0) * 512, (RegTensor<uint8_t> &)vreg_ds1, vreg_perm,
                            preg_all8);
            DataCopyScatter((__ubuf__ uint8_t *)dpds + (i + 1) * 512, (RegTensor<uint8_t> &)vreg_ds2, vreg_perm,
                            preg_all8);
        }

        LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

        for (uint16_t i = 0; i < srcM; i += 8) {
            DataCopy(vreg_sp1, (__ubuf__ float *)sp + (i + 4) * 128);
            DataCopy(vreg_sp2, (__ubuf__ float *)sp + (i + 5) * 128);
            DataCopy(vreg_sp3, (__ubuf__ float *)sp + (i + 6) * 128);
            DataCopy(vreg_sp4, (__ubuf__ float *)sp + (i + 7) * 128);

            Muls(vreg_sp1, vreg_sp1, ss, preg_all);
            Muls(vreg_sp2, vreg_sp2, ss, preg_all);
            Muls(vreg_sp3, vreg_sp3, ss, preg_all);
            Muls(vreg_sp4, vreg_sp4, ss, preg_all);

            FusedExpSub(vreg_sp1, vreg_sp1, vreg_max, preg_all);
            FusedExpSub(vreg_sp2, vreg_sp2, vreg_max, preg_all);
            FusedExpSub(vreg_sp3, vreg_sp3, vreg_max, preg_all);
            FusedExpSub(vreg_sp4, vreg_sp4, vreg_max, preg_all);

            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_p1, vreg_sp1, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_p2, vreg_sp2, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_p3, vreg_sp3, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_p4, vreg_sp4, preg_all);

            Or((RegTensor<uint8_t> &)vreg_p1, (RegTensor<uint8_t> &)vreg_p1, (RegTensor<uint8_t> &)vreg_p3, preg_all8);
            Or((RegTensor<uint8_t> &)vreg_p2, (RegTensor<uint8_t> &)vreg_p2, (RegTensor<uint8_t> &)vreg_p4, preg_all8);

            Interleave((RegTensor<half> &)vreg_p1, (RegTensor<half> &)vreg_p2, (RegTensor<half> &)vreg_p1,
                       (RegTensor<half> &)vreg_p2);

            DataCopyScatter((__ubuf__ uint8_t *)sp + (i + 0) * 512 + 128, (RegTensor<uint8_t> &)vreg_p1, vreg_perm,
                            preg_all8);
            DataCopyScatter((__ubuf__ uint8_t *)sp + (i + 1) * 512 + 128, (RegTensor<uint8_t> &)vreg_p2, vreg_perm,
                            preg_all8);

            DataCopy(vreg_dps1, (__ubuf__ float *)dpds + (i + 4) * 128);
            DataCopy(vreg_dps2, (__ubuf__ float *)dpds + (i + 5) * 128);
            DataCopy(vreg_dps3, (__ubuf__ float *)dpds + (i + 6) * 128);
            DataCopy(vreg_dps4, (__ubuf__ float *)dpds + (i + 7) * 128);

            FusedMulDstAdd(vreg_dps1, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps2, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps3, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps4, vreg_dps, vreg_d, preg_all);

            Mul(vreg_dps1, vreg_dps1, vreg_sp1, preg_all);
            Mul(vreg_dps2, vreg_dps2, vreg_sp2, preg_all);
            Mul(vreg_dps3, vreg_dps3, vreg_sp3, preg_all);
            Mul(vreg_dps4, vreg_dps4, vreg_sp4, preg_all);

            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_ds1, vreg_dps1, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_ds2, vreg_dps2, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_ds3, vreg_dps3, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_ds4, vreg_dps4, preg_all);

            Or((RegTensor<uint8_t> &)vreg_ds1, (RegTensor<uint8_t> &)vreg_ds1, (RegTensor<uint8_t> &)vreg_ds3,
               preg_all8);
            Or((RegTensor<uint8_t> &)vreg_ds2, (RegTensor<uint8_t> &)vreg_ds2, (RegTensor<uint8_t> &)vreg_ds4,
               preg_all8);
            Interleave((RegTensor<half> &)vreg_ds1, (RegTensor<half> &)vreg_ds2, (RegTensor<half> &)vreg_ds1,
                       (RegTensor<half> &)vreg_ds2);

            DataCopyScatter((__ubuf__ uint8_t *)dpds + (i + 0) * 512 + 128, (RegTensor<uint8_t> &)vreg_ds1, vreg_perm,
                            preg_all8);
            DataCopyScatter((__ubuf__ uint8_t *)dpds + (i + 1) * 512 + 128, (RegTensor<uint8_t> &)vreg_ds2, vreg_perm,
                            preg_all8);
        }
    }
}

template <typename T>
__aicore__ inline void ComputePDS(const LocalTensor<T> &spTensor, const LocalTensor<T> &dpdsTensor,
                                  const LocalTensor<uint16_t> &permTensor, const LocalTensor<T> &maxTensor,
                                  const LocalTensor<T> &sumTensor, const LocalTensor<T> &dTensor, uint16_t srcM,
                                  uint16_t srcN, float ss, float dps, float pscale, float dscale)
{
    uint64_t sp = spTensor.GetPhyAddr();
    uint64_t dpds = dpdsTensor.GetPhyAddr();
    uint64_t perm_ub = permTensor.GetPhyAddr();
    uint64_t max = maxTensor.GetPhyAddr();
    uint64_t sum = sumTensor.GetPhyAddr();
    uint64_t d = dTensor.GetPhyAddr();
    ComputePDS_VF_unalign<true>(sp, dpds, perm_ub, max, sum, d, srcM, srcN, ss, dps, pscale, dscale);
}

#else
template <typename T>
__aicore__ inline void ComputePDS(const LocalTensor<T> &spTensor, const LocalTensor<T> &dpdsTensor,
                                  const LocalTensor<uint16_t> &permTensor, const LocalTensor<T> &maxTensor,
                                  const LocalTensor<T> &sumTensor, const LocalTensor<T> &dTensor, uint16_t srcM,
                                  uint16_t srcN, float ss, float dps, float pscale, float dscale)
{}
#endif
} // namespace AscendC

#endif // MY_ANTI_QUANT_COMPUTE_P_DS_INTERFACE_H_
