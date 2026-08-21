/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_grad_regbase_not_cut_h.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_H
#define MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"

namespace MoeFinalizeRoutingV2Grad {
using namespace AscendC;
using namespace AscendC::MicroAPI;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BINARY_ADD_COF = 2;
constexpr uint32_t BLOCK_BYTE_SIZE = Ops::Base::GetUbBlockSize();
constexpr uint32_t VL_FLOAT32_SIZE = static_cast<uint32_t>(Ops::Base::GetVRegSize()) / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

struct BinAddParams {
    int64_t binaryAddQuotient;
    int64_t binaryAddk;
    int64_t binaryAddLastNum;
};

__aicore__ inline void BinaryAddVF(__ubuf__ float *ubbinaryAddCache, __ubuf__ float *binaryAddTmpAddr,
                                   const uint16_t binaryAddKLoop, const uint16_t binaryAddLoop,
                                   const int64_t binaryAddLastNum, const int64_t pos)
{
    MicroAPI::RegTensor<float> vregXQ;
    MicroAPI::RegTensor<float> vregXR;
    MicroAPI::RegTensor<float> vregXSum;
    MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint16_t curBinaryAddLoop = binaryAddLoop;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddLoop = curBinaryAddLoop / BINARY_ADD_COF;
        for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
            LoadAlign(vregXQ, ((__ubuf__ float *)binaryAddTmpAddr + j * VL_FLOAT32_SIZE));
            LoadAlign(vregXR, ((__ubuf__ float *)binaryAddTmpAddr + (j + curBinaryAddLoop) * VL_FLOAT32_SIZE));
            Add(vregXQ, vregXQ, vregXR, pregMain);
            StoreAlign(binaryAddTmpAddr + j * VL_FLOAT32_SIZE, vregXQ, pregMain);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
    {
        uint32_t sreg2 = binaryAddLastNum;
        MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg2);
        LoadAlign(vregXSum, ((__ubuf__ float *)binaryAddTmpAddr));
        Reg::Reduce<Reg::ReduceType::SUM>(vregXSum, vregXSum, pregLoop);
        StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubbinaryAddCache + pos, vregXSum, pregMerge);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist = true>
class MoeFinalizeRoutingV2GradRegbase {
public:
    __aicore__ inline MoeFinalizeRoutingV2GradRegbase(){};
    __aicore__ inline ~MoeFinalizeRoutingV2GradRegbase(){};

protected:
    __aicore__ inline void InitCommon(TPipe *pipe);
    __aicore__ inline void CopyOutGradScales(int64_t batchIdx);
    __aicore__ inline void VfCalGradScale(const LocalTensor<T1> &biasUb, const LocalTensor<T1> &gradYUb,
                                          const BinAddParams &binAddParams, const int64_t reduceNum,
                                          const LocalTensor<float> &binaryAddCache, const int64_t pos);

protected:
    GlobalTensor<T1> gradYGm_;
    GlobalTensor<T2> expandedRowIdxGm_;
    GlobalTensor<T1> expandedXGm_;
    GlobalTensor<T3> scalesGm_;
    GlobalTensor<T2> expertIdxGm_;
    GlobalTensor<T1> biasGm_;
    GlobalTensor<T1> gradExpandedXGm_;
    GlobalTensor<T1> gradExpandedXInitGm_;
    GlobalTensor<T3> gradScalesGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gradYInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> biasInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedXInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedRowIdxInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> scalesInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expertIdxInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradExpandedXOutQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradScalesOutQueue_;
    TBuf<TPosition::VECCALC> binaryAddSumBuf_;

    TPipe *pipe_;
    int64_t blockIdx_ = 0;
    int64_t startBatchIdx_ = 0;
    int64_t endBatchIdx_ = 0;
};

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbase<T1, T2, T3, IsBiasExist>::InitCommon(TPipe *pipe)
{
    this->pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbase<T1, T2, T3, IsBiasExist>::CopyOutGradScales(int64_t batchIdx)
{
    LocalTensor<T3> gradScalesUb = this->gradScalesOutQueue_.template DeQue<T3>();
    DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
    copyExtParams.blockLen = sizeof(T3);
    DataCopyPad(this->gradScalesGm_[batchIdx], gradScalesUb, copyExtParams);
    this->gradScalesOutQueue_.FreeTensor(gradScalesUb);
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbase<T1, T2, T3, IsBiasExist>::VfCalGradScale(
    const LocalTensor<T1> &biasUb, const LocalTensor<T1> &gradYUb, const BinAddParams &binAddParams,
    const int64_t reduceNum, const LocalTensor<float> &binaryAddCache, const int64_t pos)
{
    __ubuf__ T1 *ubBias = (__ubuf__ T1 *)biasUb.GetPhyAddr();
    __ubuf__ T1 *ubGradY = (__ubuf__ T1 *)gradYUb.GetPhyAddr();
    __ubuf__ float *ubbinaryAddCache = (__ubuf__ float *)binaryAddCache.GetPhyAddr();
    LocalTensor<float> binaryAddSumUb = binaryAddSumBuf_.Get<float>();
    __ubuf__ float *ubAddSum = (__ubuf__ float *)binaryAddSumUb.GetPhyAddr();

    int64_t binaryAddQuotient = binAddParams.binaryAddQuotient;
    int64_t binaryAddKLoop = binAddParams.binaryAddk;
    int64_t binaryAddLastNum = binAddParams.binaryAddLastNum;
    uint32_t binaryAddRemainder = reduceNum % binaryAddQuotient;
    uint16_t remainderLoop = Ops::Base::CeilDiv(binaryAddRemainder, VL_FLOAT32_SIZE);
    uint16_t remainderGeneral = remainderLoop == 0 ? 0 : remainderLoop - 1;
    uint16_t quotientLoop = Ops::Base::CeilDiv(binaryAddQuotient, static_cast<int64_t>(VL_FLOAT32_SIZE));
    uint16_t binaryAddLoop = ((binaryAddQuotient / VL_FLOAT32_SIZE) / VL_FLOAT32_SIZE);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregBias;
        MicroAPI::RegTensor<float> tempBias;
        MicroAPI::RegTensor<float> vregBiasQ;
        MicroAPI::RegTensor<float> vregBiasR;
        MicroAPI::RegTensor<float> vregGradY;
        MicroAPI::RegTensor<float> vregGradYQ;
        MicroAPI::RegTensor<float> vregGradYR;
        MicroAPI::RegTensor<float> vregSumBias;
        MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg pregLoop;
        uint32_t sreg0 = binaryAddRemainder;
        // step1: the first helf reduce to 64, this part always 64 align
        for (uint16_t i = 0; i < remainderGeneral; i++) {
            pregLoop = MicroAPI::UpdateMask<float>(sreg0);
            ops::LoadTwoTensorForDtypeT<T1>(ubBias, ubBias, vregBiasQ, vregBiasR, pregMain, pregLoop,
                                            i * VL_FLOAT32_SIZE, i * VL_FLOAT32_SIZE + binaryAddQuotient);
            ops::LoadTwoTensorForDtypeT<T1>(ubGradY, ubGradY, vregGradYQ, vregGradYR, pregMain, pregLoop,
                                            i * VL_FLOAT32_SIZE, i * VL_FLOAT32_SIZE + binaryAddQuotient);
            Mul(vregBiasQ, vregBiasQ, vregGradYQ, pregMain);
            Mul(vregBiasR, vregBiasR, vregGradYR, pregLoop);
            Add(vregBiasQ, vregBiasQ, vregBiasR, pregLoop);
            Reg::Reduce<Reg::ReduceType::SUM>(vregSumBias, vregBiasQ, pregLoop);
            StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubAddSum + i, vregSumBias, pregMerge);
        }
        // step2: the tail (last 64 or less than 64) blocks reduce to 1.
        {
            pregLoop = MicroAPI::UpdateMask<float>(sreg0);
            ops::LoadTwoTensorForDtypeT<T1>(ubBias, ubBias, vregBiasQ, vregBiasR, pregMain, pregLoop,
                                            remainderGeneral * VL_FLOAT32_SIZE,
                                            remainderGeneral * VL_FLOAT32_SIZE + binaryAddQuotient);
            ops::LoadTwoTensorForDtypeT<T1>(ubGradY, ubGradY, vregGradYQ, vregGradYR, pregMain, pregLoop,
                                            remainderGeneral * VL_FLOAT32_SIZE,
                                            remainderGeneral * VL_FLOAT32_SIZE + binaryAddQuotient);
            Mul(vregBiasQ, vregBiasQ, vregGradYQ, pregMain);
            Mul(vregBiasR, vregBiasR, vregGradYR, pregLoop);
            Add(tempBias, vregBiasQ, vregBiasR, pregLoop);
            Copy<float, MicroAPI::MaskMergeMode::MERGING>(vregBiasQ, tempBias, pregLoop);
            Reg::Reduce<Reg::ReduceType::SUM>(vregSumBias, vregBiasQ, pregMain);
            StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubAddSum + remainderGeneral, vregSumBias,
                                                                           pregMerge);
        }
        // step3: non-overlapping portions of the first half reduce by 64, this part always 64 align
        for (uint16_t i = 0; i < static_cast<uint16_t>(quotientLoop - remainderLoop); i++) {
            ops::LoadOneTensorForDtypeT<T1>(ubBias, vregBias, pregMain, (i + remainderLoop) * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(ubGradY, vregGradY, pregMain, (i + remainderLoop) * VL_FLOAT32_SIZE);
            Mul(vregBias, vregBias, vregGradY, pregMain);
            Reg::Reduce<Reg::ReduceType::SUM>(vregSumBias, vregBias, pregMain);
            StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubAddSum + remainderLoop + i, vregSumBias,
                                                                           pregMerge);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        BinaryAddVF(ubbinaryAddCache, ubAddSum, binaryAddKLoop, binaryAddLoop, binaryAddLastNum, pos);
    } // end VF
}

} // namespace MoeFinalizeRoutingV2Grad

#endif // MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_H
