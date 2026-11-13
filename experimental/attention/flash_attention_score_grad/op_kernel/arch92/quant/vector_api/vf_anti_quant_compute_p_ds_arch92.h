/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef VF_ANTI_QUANT_COMPUTE_P_DS_ARCH92_H_
#define VF_ANTI_QUANT_COMPUTE_P_DS_ARCH92_H_
#include "../flash_attention_score_grad_quant_config_arch92.h"
#include "kernel_tensor.h"
namespace AscendC {
#ifndef __CCE_KT_TEST__
// Keep independent S/Exp and dP/D chains interleaved before P/dS quantization.
using namespace Reg;
constexpr static AscendC::Reg::CastTrait castTraitB322B8Three = {
    AscendC::Reg::RegLayout::THREE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
constexpr static AscendC::Reg::CastTrait castTraitB322B8Two = {
    AscendC::Reg::RegLayout::TWO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
constexpr static AscendC::Reg::CastTrait castTraitB322B8One = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
constexpr static AscendC::Reg::CastTrait castTraitB322B8Zero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

// Full input pitch, independent compact NZ outputs. Preload dP and prepare
// (dP * scale + delta) before the exponentials; keep the original arithmetic
// and cast traits for each row. Four-row registers are reused for rows 4..7.
// Unlike the in-place VF, no output can overwrite a future input load.
__aicore__ inline void ComputePDSScheduledVF(uint64_t sp, uint64_t dpds, uint64_t pOut, uint64_t dsOut,
                                             uint64_t perm_ub, uint64_t max, uint64_t sum, uint64_t d, uint16_t srcM,
                                             uint16_t srcN, float ss, float dps, float pscale, float dscale)
{
    constexpr uint32_t inputStride = 128;
    float dscale_neg = (float)(-1.0) * dscale;
    uint32_t tailN32 = static_cast<uint32_t>(srcN);
    __VEC_SCOPE__
    {
        MaskReg preg_all = UpdateMask<float>(tailN32);
        MaskReg preg_all8 = CreateMask<uint8_t, MaskPattern::ALL>();

        RegTensor<float> vreg_sp1, vreg_sp2, vreg_sp3, vreg_sp4;
        RegTensor<float> vreg_dps1, vreg_dps2, vreg_dps3, vreg_dps4;
        RegTensor<hifloat8_t> vreg_p1, vreg_p2, vreg_p3, vreg_p4;
        RegTensor<hifloat8_t> vreg_ds1, vreg_ds2, vreg_ds3, vreg_ds4;
        RegTensor<float> vreg_d, vreg_max, vreg_sum, vreg_dps, vreg_ps;
        RegTensor<uint16_t> vreg_perm;
        DataCopy(vreg_d, (__ubuf__ float*)d);
        DataCopy(vreg_max, (__ubuf__ float*)max);
        DataCopy(vreg_sum, (__ubuf__ float*)sum);
        DataCopy(vreg_perm, (__ubuf__ uint16_t*)perm_ub);

        // Cached max retains exactly (max + log(sum)) + log(pscale).

        Muls(vreg_d, vreg_d, dscale_neg, preg_all);
        Duplicate(vreg_dps, dps, preg_all);
        Muls(vreg_dps, vreg_dps, dscale, preg_all);
        for (uint16_t i = 0; i < srcM; i += 8) {
            DataCopy(vreg_sp1, (__ubuf__ float*)sp + (i + 0) * inputStride);
            DataCopy(vreg_sp2, (__ubuf__ float*)sp + (i + 1) * inputStride);
            DataCopy(vreg_sp3, (__ubuf__ float*)sp + (i + 2) * inputStride);
            DataCopy(vreg_sp4, (__ubuf__ float*)sp + (i + 3) * inputStride);

            DataCopy(vreg_dps1, (__ubuf__ float*)dpds + (i + 0) * inputStride);
            DataCopy(vreg_dps2, (__ubuf__ float*)dpds + (i + 1) * inputStride);
            DataCopy(vreg_dps3, (__ubuf__ float*)dpds + (i + 2) * inputStride);
            DataCopy(vreg_dps4, (__ubuf__ float*)dpds + (i + 3) * inputStride);

            Muls(vreg_sp1, vreg_sp1, ss, preg_all);
            Muls(vreg_sp2, vreg_sp2, ss, preg_all);
            Muls(vreg_sp3, vreg_sp3, ss, preg_all);
            Muls(vreg_sp4, vreg_sp4, ss, preg_all);

            FusedMulDstAdd(vreg_dps1, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps2, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps3, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps4, vreg_dps, vreg_d, preg_all);

            ExpSub(vreg_sp1, vreg_sp1, vreg_max, preg_all);
            ExpSub(vreg_sp2, vreg_sp2, vreg_max, preg_all);
            ExpSub(vreg_sp3, vreg_sp3, vreg_max, preg_all);
            ExpSub(vreg_sp4, vreg_sp4, vreg_max, preg_all);

            Mul(vreg_dps1, vreg_dps1, vreg_sp1, preg_all);
            Mul(vreg_dps2, vreg_dps2, vreg_sp2, preg_all);
            Mul(vreg_dps3, vreg_dps3, vreg_sp3, preg_all);
            Mul(vreg_dps4, vreg_dps4, vreg_sp4, preg_all);

            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_p1, vreg_sp1, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_p2, vreg_sp2, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_p3, vreg_sp3, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_p4, vreg_sp4, preg_all);

            Or((RegTensor<uint8_t>&)vreg_p1, (RegTensor<uint8_t>&)vreg_p1, (RegTensor<uint8_t>&)vreg_p3, preg_all8);
            Or((RegTensor<uint8_t>&)vreg_p2, (RegTensor<uint8_t>&)vreg_p2, (RegTensor<uint8_t>&)vreg_p4, preg_all8);

            Interleave((RegTensor<half>&)vreg_p1, (RegTensor<half>&)vreg_p2, (RegTensor<half>&)vreg_p1,
                       (RegTensor<half>&)vreg_p2);

            DataCopyScatter((__ubuf__ uint8_t*)pOut + ((i / 8) * 256), (RegTensor<uint8_t>&)vreg_p1, vreg_perm,
                            preg_all8);
            DataCopyScatter((__ubuf__ uint8_t*)pOut + (4096 + (i / 8) * 256), (RegTensor<uint8_t>&)vreg_p2, vreg_perm,
                            preg_all8);

            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_ds1, vreg_dps1, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_ds2, vreg_dps2, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_ds3, vreg_dps3, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_ds4, vreg_dps4, preg_all);

            Or((RegTensor<uint8_t>&)vreg_ds1, (RegTensor<uint8_t>&)vreg_ds1, (RegTensor<uint8_t>&)vreg_ds3, preg_all8);
            Or((RegTensor<uint8_t>&)vreg_ds2, (RegTensor<uint8_t>&)vreg_ds2, (RegTensor<uint8_t>&)vreg_ds4, preg_all8);
            Interleave((RegTensor<half>&)vreg_ds1, (RegTensor<half>&)vreg_ds2, (RegTensor<half>&)vreg_ds1,
                       (RegTensor<half>&)vreg_ds2);

            DataCopyScatter((__ubuf__ uint8_t*)dsOut + ((i / 8) * 256), (RegTensor<uint8_t>&)vreg_ds1, vreg_perm,
                            preg_all8);
            DataCopyScatter((__ubuf__ uint8_t*)dsOut + (4096 + (i / 8) * 256), (RegTensor<uint8_t>&)vreg_ds2, vreg_perm,
                            preg_all8);

            // Finish this eight-row group before advancing to the next one.
            DataCopy(vreg_sp1, (__ubuf__ float*)sp + (i + 4) * inputStride);
            DataCopy(vreg_sp2, (__ubuf__ float*)sp + (i + 5) * inputStride);
            DataCopy(vreg_sp3, (__ubuf__ float*)sp + (i + 6) * inputStride);
            DataCopy(vreg_sp4, (__ubuf__ float*)sp + (i + 7) * inputStride);

            DataCopy(vreg_dps1, (__ubuf__ float*)dpds + (i + 4) * inputStride);
            DataCopy(vreg_dps2, (__ubuf__ float*)dpds + (i + 5) * inputStride);
            DataCopy(vreg_dps3, (__ubuf__ float*)dpds + (i + 6) * inputStride);
            DataCopy(vreg_dps4, (__ubuf__ float*)dpds + (i + 7) * inputStride);

            Muls(vreg_sp1, vreg_sp1, ss, preg_all);
            Muls(vreg_sp2, vreg_sp2, ss, preg_all);
            Muls(vreg_sp3, vreg_sp3, ss, preg_all);
            Muls(vreg_sp4, vreg_sp4, ss, preg_all);

            FusedMulDstAdd(vreg_dps1, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps2, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps3, vreg_dps, vreg_d, preg_all);
            FusedMulDstAdd(vreg_dps4, vreg_dps, vreg_d, preg_all);

            FusedExpSub(vreg_sp1, vreg_sp1, vreg_max, preg_all);
            FusedExpSub(vreg_sp2, vreg_sp2, vreg_max, preg_all);
            FusedExpSub(vreg_sp3, vreg_sp3, vreg_max, preg_all);
            FusedExpSub(vreg_sp4, vreg_sp4, vreg_max, preg_all);

            Mul(vreg_dps1, vreg_dps1, vreg_sp1, preg_all);
            Mul(vreg_dps2, vreg_dps2, vreg_sp2, preg_all);
            Mul(vreg_dps3, vreg_dps3, vreg_sp3, preg_all);
            Mul(vreg_dps4, vreg_dps4, vreg_sp4, preg_all);

            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_p1, vreg_sp1, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_p2, vreg_sp2, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_p3, vreg_sp3, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_p4, vreg_sp4, preg_all);

            Or((RegTensor<uint8_t>&)vreg_p1, (RegTensor<uint8_t>&)vreg_p1, (RegTensor<uint8_t>&)vreg_p3, preg_all8);
            Or((RegTensor<uint8_t>&)vreg_p2, (RegTensor<uint8_t>&)vreg_p2, (RegTensor<uint8_t>&)vreg_p4, preg_all8);

            Interleave((RegTensor<half>&)vreg_p1, (RegTensor<half>&)vreg_p2, (RegTensor<half>&)vreg_p1,
                       (RegTensor<half>&)vreg_p2);

            DataCopyScatter((__ubuf__ uint8_t*)pOut + ((i / 8) * 256) + 128, (RegTensor<uint8_t>&)vreg_p1, vreg_perm,
                            preg_all8);
            DataCopyScatter((__ubuf__ uint8_t*)pOut + (4096 + (i / 8) * 256) + 128, (RegTensor<uint8_t>&)vreg_p2,
                            vreg_perm, preg_all8);

            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_ds1, vreg_dps1, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Zero>(vreg_ds2, vreg_dps2, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_ds3, vreg_dps3, preg_all);
            Cast<hifloat8_t, float, castTraitB322B8Two>(vreg_ds4, vreg_dps4, preg_all);

            Or((RegTensor<uint8_t>&)vreg_ds1, (RegTensor<uint8_t>&)vreg_ds1, (RegTensor<uint8_t>&)vreg_ds3, preg_all8);
            Or((RegTensor<uint8_t>&)vreg_ds2, (RegTensor<uint8_t>&)vreg_ds2, (RegTensor<uint8_t>&)vreg_ds4, preg_all8);
            Interleave((RegTensor<half>&)vreg_ds1, (RegTensor<half>&)vreg_ds2, (RegTensor<half>&)vreg_ds1,
                       (RegTensor<half>&)vreg_ds2);

            DataCopyScatter((__ubuf__ uint8_t*)dsOut + ((i / 8) * 256) + 128, (RegTensor<uint8_t>&)vreg_ds1, vreg_perm,
                            preg_all8);
            DataCopyScatter((__ubuf__ uint8_t*)dsOut + (4096 + (i / 8) * 256) + 128, (RegTensor<uint8_t>&)vreg_ds2,
                            vreg_perm, preg_all8);
        }
    }
}

template <typename T, typename QT>
__aicore__ inline void ComputePDSQuant(const LocalTensor<T>& spTensor, const LocalTensor<T>& dpdsTensor,
                                       const LocalTensor<QT>& pTensor, const LocalTensor<QT>& dsTensor,
                                       const LocalTensor<uint16_t>& permTensor, const LocalTensor<T>& maxTensor,
                                       const LocalTensor<T>& sumTensor, const LocalTensor<T>& dTensor, uint16_t srcM,
                                       uint16_t srcN, float ss, float dps, float pscale, float dscale)
{
    constexpr uint32_t inputStride = 128;
    for (uint16_t i = srcM; i < Ceil<uint16_t>(srcM, 8) * 8; ++i) {
        AscendC::Duplicate(spTensor[i * inputStride], static_cast<T>(0), 64);
        AscendC::Duplicate(dpdsTensor[i * inputStride], static_cast<T>(0), 64);
    }
    ComputePDSScheduledVF(spTensor.GetPhyAddr(), dpdsTensor.GetPhyAddr(), pTensor.GetPhyAddr(), dsTensor.GetPhyAddr(),
                          permTensor.GetPhyAddr(), maxTensor.GetPhyAddr(), sumTensor.GetPhyAddr(), dTensor.GetPhyAddr(),
                          srcM, srcN, ss, dps, pscale, dscale);
}

__aicore__ inline void NormalizeQuantStats(const LocalTensor<float>& maxTensor, const LocalTensor<float>& sumTensor,
                                           uint16_t rows, float pscale)
{
    uint64_t maxAddr = maxTensor.GetPhyAddr();
    uint64_t sumAddr = sumTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> maxValue, sumValue, pValue;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        Duplicate(pValue, pscale, full);
        Log(pValue, pValue, full);
        for (uint16_t base = 0; base < rows; base += 64) {
            uint32_t remaining = rows - base;
            MaskReg mask = UpdateMask<float>(remaining);
            DataCopy(maxValue, (__ubuf__ float*)maxAddr + base);
            DataCopy(sumValue, (__ubuf__ float*)sumAddr + base);
            Log(sumValue, sumValue, mask);
            Add(maxValue, maxValue, sumValue, mask);
            Add(maxValue, maxValue, pValue, mask);
            DataCopy((__ubuf__ float*)maxAddr + base, maxValue, mask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}
#else
template <typename T, typename QT>
__aicore__ inline void ComputePDSQuant(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<QT>&,
                                       const LocalTensor<QT>&, const LocalTensor<uint16_t>&, const LocalTensor<T>&,
                                       const LocalTensor<T>&, const LocalTensor<T>&, uint16_t, uint16_t, float, float,
                                       float, float)
{}
__aicore__ inline void NormalizeQuantStats(const LocalTensor<float>&, const LocalTensor<float>&, uint16_t, float) {}
#endif
} // namespace AscendC
#endif
