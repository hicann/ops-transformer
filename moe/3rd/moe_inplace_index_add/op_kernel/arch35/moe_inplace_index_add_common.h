/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_inplace_index_add_common.h
 * \brief moe_inplace_index_add
 */
#ifndef ASCENDC_MOE_INPLACE_INDEX_ADD_COMMON_H_
#define ASCENDC_MOE_INPLACE_INDEX_ADD_COMMON_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "op_kernel/load_store_utils.h"
#include "indices_sort_utils.h"

namespace MoeInplaceIndexAdd {
using namespace AscendC;

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint32_t VECTOR_LENGTH = Ops::Base::GetVRegSize();
constexpr uint32_t VL_B32 = VECTOR_LENGTH / sizeof(uint32_t);

constexpr uint64_t SORT_PAD_NUM = 2;
constexpr uint32_t HASH_SCORE_BUF_SIZE = 128;
constexpr float SORT_HIST_THRESHOLD = 0.01f;
static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

constexpr uint32_t TWO = 2;
constexpr uint32_t THREE = 3;
constexpr uint32_t FOUR = 4;
constexpr uint32_t CAST_0 = 0;
constexpr uint32_t CAST_1 = 1;
constexpr uint32_t CAST_2 = 2;
constexpr uint32_t CAST_3 = 3;
constexpr uint32_t CAST_4 = 4;
constexpr uint32_t CAST_5 = 5;
constexpr uint32_t MASK_UINT8 = 255;
constexpr int64_t VFLEN_INT64 = Ops::Base::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = Ops::Base::GetVRegSize() / sizeof(int32_t);
constexpr int64_t VFLEN_INT16 = Ops::Base::GetVRegSize() / sizeof(int16_t);
constexpr int64_t VFLEN_INT16HALF = Ops::Base::GetVRegSize() / sizeof(int16_t) / TWO;
constexpr int64_t VFLEN_UINT8 = Ops::Base::GetVRegSize() / sizeof(uint8_t);
constexpr int64_t VFLEN_UINT8HALFHALF = Ops::Base::GetVRegSize() / sizeof(uint8_t) / FOUR;

static constexpr MicroAPI::CastTrait castTraitB8B162B32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
template <typename T>
struct MulsSelType {
    using type = typename std::conditional<
        IsSameType<T, int8_t>::value, half,
        typename std::conditional<IsSameType<T, uint8_t>::value, half,
                                  typename std::conditional<IsSameType<T, bool>::value, int32_t, T>::type>::type>::type;
};

template <typename T, uint32_t CAST_MODE>
struct CastType {
    using type = typename std::conditional<
        CAST_MODE == CAST_1, int16_t,
        typename std::conditional<
            CAST_MODE == CAST_2, int32_t,
            typename std::conditional<
                CAST_MODE == CAST_3, int16_t,
                typename std::conditional<CAST_MODE == CAST_4, uint8_t,
                                          typename std::conditional<CAST_MODE == CAST_5, uint8_t, T>::type>::type>::
                type>::type>::type;
};

template <typename T>
struct AtomicSelType {
    using type =
        typename std::conditional<IsSameType<T, bool>::value, int8_t,
                                  typename std::conditional<IsSameType<T, uint8_t>::value, uint32_t, T>::type>::type;
};

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor, int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyInNoContiguous(const LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                          int64_t dataCount, int64_t dataLen, int64_t srcStride = 0,
                                          int64_t dstStride = 0)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(dataCount), static_cast<uint32_t>(dataLen * sizeof(T)),
                                    static_cast<uint32_t>(srcStride * sizeof(T)), static_cast<uint32_t>(dstStride),
                                    static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad<T, PaddingMode::Compact>(dstTensor, srcTensor, copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams);
}

template <typename T, typename CAST_T>
__aicore__ inline void CastToInt32(LocalTensor<CAST_T> &dstLocal, LocalTensor<T> &srcLocal, uint32_t dataLen)
{
    __local_mem__ T *srcAddr = (__local_mem__ T *)srcLocal.GetPhyAddr();
    __local_mem__ CAST_T *dstAddr = (__local_mem__ CAST_T *)dstLocal.GetPhyAddr();

    uint16_t loopTimes = Ops::Base::CeilDiv(dataLen, VL_B32);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcValue;
        MicroAPI::RegTensor<CAST_T> dstValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = dataLen;
        for (uint16_t j = 0; j < loopTimes; j++) {
            auto srcReg = MicroAPI::CreateAddrReg<T>(j, static_cast<uint16_t>(VL_B32));
            auto dstReg = MicroAPI::CreateAddrReg<CAST_T>(j, static_cast<uint16_t>(VL_B32));
            preg = MicroAPI::UpdateMask<uint32_t>(sregMask);
            if constexpr (IsSameType<T, int16_t>::value) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcValue, srcAddr, srcReg);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcValue, srcAddr, srcReg);
            }
            MicroAPI::Cast<CAST_T, T, castTraitB8B162B32>(dstValue, srcValue, preg);
            MicroAPI::DataCopy<CAST_T, MicroAPI::StoreDist::DIST_NORM>(dstAddr, dstValue, dstReg, preg);
        }
    }
}

template <typename T, typename CAST_T>
__aicore__ inline void CastToOrigin(LocalTensor<T> &dstLocal, LocalTensor<CAST_T> &srcLocal, uint32_t dataLen)
{
    __local_mem__ CAST_T *srcAddr = (__local_mem__ CAST_T *)srcLocal.GetPhyAddr();
    __local_mem__ T *dstAddr = (__local_mem__ T *)dstLocal.GetPhyAddr();

    uint16_t loopTimes = Ops::Base::CeilDiv(dataLen, VL_B32);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<CAST_T> srcValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = dataLen;
        for (uint16_t j = 0; j < loopTimes; j++) {
            auto srcReg = MicroAPI::CreateAddrReg<CAST_T>(j, static_cast<uint16_t>(VL_B32));
            auto dstReg = MicroAPI::CreateAddrReg<T>(j, static_cast<uint16_t>(VL_B32));
            preg = MicroAPI::UpdateMask<uint32_t>(sregMask);
            MicroAPI::DataCopy<CAST_T, MicroAPI::LoadDist::DIST_NORM>(srcValue, srcAddr, srcReg);
            if constexpr (IsSameType<T, int16_t>::value) {
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr, (MicroAPI::RegTensor<T> &)srcValue,
                                                                          dstReg, preg);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK4_B32>(dstAddr, (MicroAPI::RegTensor<T> &)srcValue,
                                                                           dstReg, preg);
            }
        }
    }
}
template <typename T>
__aicore__ inline void LoadOneTensorForDtypeInt(__local_mem__ T *input, MicroAPI::RegTensor<int16_t> &dst,
                                                MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, int8_t>::value) {
        DataCopy<int8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>((MicroAPI::RegTensor<int8_t> &)(dst),
                                                             ((__local_mem__ int8_t *)(input) + (offset)));
    } else if constexpr (IsSameType<T, bool>::value) {
        DataCopy<bool, MicroAPI::LoadDist::DIST_UNPACK_B8>((MicroAPI::RegTensor<bool> &)(dst),
                                                           ((__local_mem__ bool *)(input) + (offset)));
    } else {
        DataCopy(dst, ((__local_mem__ int16_t *)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void StoreOneTensorForDtypeInt(__local_mem__ T *output, MicroAPI::RegTensor<int16_t> &src,
                                                 MicroAPI::MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, int8_t>::value) {
        DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK_B16>(((__local_mem__ int8_t *)output + offset),
                                                             (MicroAPI::RegTensor<int8_t> &)src, preg);
    } else if constexpr (IsSameType<T, bool>::value) {
        DataCopy<bool, MicroAPI::StoreDist::DIST_PACK_B16>(((__local_mem__ bool *)output + offset),
                                                           (MicroAPI::RegTensor<bool> &)src, preg);
    } else {
        DataCopy(((__local_mem__ int16_t *)output + offset), src, preg);
    }
}

template <typename VAR_T>
__aicore__ inline void ComputeMulWithIntCast(LocalTensor<VAR_T> updateMulLocal, LocalTensor<VAR_T> updatesLocal,
                                             VAR_T alphaValue, int64_t colLen)
{
    __local_mem__ VAR_T *updatesAddr = (__local_mem__ VAR_T *)updatesLocal.GetPhyAddr();
    __local_mem__ VAR_T *updateMulAddr = (__local_mem__ VAR_T *)updateMulLocal.GetPhyAddr();

    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int16_t);
    int32_t loopSize = Ops::Base::CeilDiv(static_cast<uint32_t>(colLen), vfLen);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int16_t> sumReg;
        AscendC::MicroAPI::RegTensor<int16_t> castReg;
        AscendC::MicroAPI::RegTensor<int16_t> alphaReg;
        AscendC::MicroAPI::MaskReg maskReg;
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
            maskReg = AscendC::MicroAPI::UpdateMask<int16_t>(maskLen);
            AscendC::MicroAPI::Duplicate(alphaReg, static_cast<int16_t>(alphaValue), maskReg);
            auto updatesOffet = j * vfLen;
            LoadOneTensorForDtypeInt<VAR_T>(updatesAddr, castReg, maskReg, updatesOffet);
            AscendC::MicroAPI::Mul(sumReg, castReg, alphaReg, maskReg);
            StoreOneTensorForDtypeInt<VAR_T>(updateMulAddr, sumReg, maskReg, updatesOffet);
        }
    }
}

template <typename T>
__aicore__ inline T Aligned(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

template <typename VAR_T, typename IDX_T, uint32_t CAST_MODE>
class MoeInplaceIndexAddBase {
public:
    int64_t indicesFactor_ = 0;
    int64_t afterAxisFactor_ = 0;
    int64_t afterAxis_ = 0;
    int64_t varInAxis_ = 0;
    int64_t updatesInAxis_ = 0;
    int64_t eachCoreAfterAxisCount_ = 0;
    int64_t eachCoreIndexCount_ = 0;
    uint32_t uniqueIdNum_ = 0;
    float maxScore_ = static_cast<float>(0);
    using atomicSelType = typename AtomicSelType<VAR_T>::type;
    using CAST_T = typename CastType<IDX_T, CAST_MODE>::type;
    int64_t shiftOffset_ = UB_AGLIN_VALUE / sizeof(CAST_T);

    AscendC::GlobalTensor<IDX_T> indicesGm_;
    AscendC::GlobalTensor<VAR_T> updatesGm_;
    AscendC::GlobalTensor<atomicSelType> varGm_;

    TBuf<QuePosition::VECCALC> maxScoreBuf_;
    TBuf<QuePosition::VECCALC> sortIndicesBuf_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesCastQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesOriginIdexQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> uniqueIdCountQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> updateSumIdxQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> updateSumQue_;

    using IndexRegType =
        typename std::conditional<IsSameType<IDX_T, int64_t>::value,
                                  typename AscendC::MicroAPI::RegTensor<uint64_t, AscendC::MicroAPI::RegTraitNumTwo>,
                                  typename AscendC::MicroAPI::RegTensor<uint32_t>>::type;
    using InnerRegType =
        typename std::conditional<IsSameType<IDX_T, int64_t>::value,
                                  typename AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo>,
                                  typename AscendC::MicroAPI::RegTensor<int32_t>>::type;
    using selRegType = typename std::conditional<IsSameType<VAR_T, bool>::value, int8_t, VAR_T>::type;

    __aicore__ inline void InitBaseBuffer(TPipe &pipe, GM_ADDR var, GM_ADDR indices, GM_ADDR updates)
    {
        varGm_.SetGlobalBuffer((__gm__ atomicSelType *)(var));
        indicesGm_.SetGlobalBuffer((__gm__ IDX_T *)(indices));
        updatesGm_.SetGlobalBuffer((__gm__ VAR_T *)(updates));

        pipe.InitBuffer(maxScoreBuf_, HASH_SCORE_BUF_SIZE * sizeof(float));
        pipe.InitBuffer(
            sortIndicesBuf_,
            Ops::Base::CeilAlign(indicesFactor_ * sizeof(CAST_T) + SORT_PAD_NUM * UB_AGLIN_VALUE, UB_AGLIN_VALUE));
        pipe.InitBuffer(updatesOriginIdexQue_, DOUBLE_BUFFER, indicesFactor_ * sizeof(uint32_t));
        pipe.InitBuffer(uniqueIdCountQue_, DOUBLE_BUFFER,
                        Ops::Base::CeilAlign((indicesFactor_ + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
        pipe.InitBuffer(updateSumIdxQue_, DOUBLE_BUFFER,
                        Ops::Base::CeilAlign((indicesFactor_ + 1) * sizeof(IDX_T), UB_AGLIN_VALUE));
        pipe.InitBuffer(indicesCastQue_, DOUBLE_BUFFER, (indicesFactor_) * sizeof(CAST_T));
    }

    __aicore__ void inline ComputeUniqueIdNumInt64(__local_mem__ CAST_T *indicesAddr,
                                                   __local_mem__ int32_t *uniqueIdCountsAddr, uint16_t loopCnt,
                                                   int64_t dataLen)
    {
        uint32_t counter = dataLen + 1;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
        AscendC::MicroAPI::RegTensor<CAST_T> sortedIdxReg, sortedIdxShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskHalf;
        AscendC::MicroAPI::UnalignReg u0, uOut;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT64);
            maskReg = AscendC::MicroAPI::UpdateMask<CAST_T>(counter);
            auto startAddr = indicesAddr + i * VFLEN_INT64;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<CAST_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<CAST_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                                 maskHalf);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __aicore__ void inline ComputeUniqueIdNumInt32(__local_mem__ CAST_T *indicesAddr,
                                                   __local_mem__ int32_t *uniqueIdCountsAddr, uint16_t loopCnt,
                                                   int64_t dataLen)
    {
        uint32_t counter = dataLen + 1;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
        AscendC::MicroAPI::RegTensor<CAST_T> sortedIdxReg, sortedIdxShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask, maskReg;
        AscendC::MicroAPI::UnalignReg u0, uOut;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT32);
            maskReg = AscendC::MicroAPI::UpdateMask<CAST_T>(counter);
            auto startAddr = indicesAddr + i * VFLEN_INT32;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<CAST_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<CAST_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                                 cmpMask);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __aicore__ void inline ComputeUniqueIdNumInt16(__local_mem__ CAST_T *indicesAddr,
                                                   __local_mem__ int32_t *uniqueIdCountsAddr, uint16_t loopCnt,
                                                   int64_t dataLen)
    {
        uint32_t counter = dataLen + 1;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg, orderReg2, selReg, selReg2;
        AscendC::MicroAPI::RegTensor<CAST_T> sortedIdxReg, sortedIdxShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskDouble1, maskDouble2;
        AscendC::MicroAPI::UnalignReg u0, uOut;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT16);
            AscendC::MicroAPI::Arange(orderReg2, i * VFLEN_INT16 + VFLEN_INT16HALF);
            maskReg = AscendC::MicroAPI::UpdateMask<CAST_T>(counter);
            auto startAddr = indicesAddr + i * VFLEN_INT16;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<CAST_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<CAST_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskDouble1, cmpMask);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskDouble2, cmpMask);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                                 maskDouble1);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2,
                                                                                                 maskDouble2);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg, uOut);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg2, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __aicore__ void inline ComputeUniqueIdNumUint8(__local_mem__ CAST_T *indicesAddr,
                                                   __local_mem__ int32_t *uniqueIdCountsAddr, uint16_t loopCnt,
                                                   int64_t dataLen)
    {
        uint32_t counter = dataLen + 1;
        AscendC::MicroAPI::RegTensor<int32_t> orderReg, orderReg2, orderReg3, orderReg4;
        AscendC::MicroAPI::RegTensor<int32_t> selReg, selReg2, selReg3, selReg4;
        AscendC::MicroAPI::RegTensor<CAST_T> sortedIdxReg, sortedIdxShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskFour1, maskFour2, maskFour3, maskFour4;
        AscendC::MicroAPI::UnalignReg u0, uOut;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * VFLEN_UINT8);
            AscendC::MicroAPI::Arange(orderReg2, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF);
            AscendC::MicroAPI::Arange(orderReg3, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * TWO);
            AscendC::MicroAPI::Arange(orderReg4, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * THREE);
            maskReg = AscendC::MicroAPI::UpdateMask<CAST_T>(counter);
            auto startAddr = indicesAddr + i * VFLEN_UINT8;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<CAST_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
            AscendC::MicroAPI::Compare<CAST_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour3, cmpMask);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour4, cmpMask);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour1, maskFour3);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour2, maskFour3);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour3, maskFour4);
            AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour4, maskFour4);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                                 maskFour1);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2,
                                                                                                 maskFour2);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg3, orderReg3,
                                                                                                 maskFour3);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg4, orderReg4,
                                                                                                 maskFour4);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg, uOut);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg2, uOut);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg3, uOut);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                uniqueIdCountsAddr, selReg4, uOut);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
    }

    __aicore__ uint32_t inline ComputeUniqueIdNum(LocalTensor<CAST_T> indicesLocal,
                                                  LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
    {
        __local_mem__ CAST_T *indicesAddr =
            (__local_mem__ CAST_T *)indicesLocal[(UB_AGLIN_VALUE / sizeof(CAST_T))].GetPhyAddr();
        __local_mem__ int32_t *uniqueIdCountsAddr = (__local_mem__ int32_t *)uniqueIdCountLocal.GetPhyAddr();

        constexpr int64_t vfLen = Ops::Base::GetVRegSize() / sizeof(CAST_T);
        uint16_t loopCnt = Ops::Base::CeilDiv(dataLen + 1, vfLen);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

            if constexpr (std::is_same<int64_t, CAST_T>::value) {
                ComputeUniqueIdNumInt64(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
            } else if constexpr (std::is_same<int32_t, CAST_T>::value) {
                ComputeUniqueIdNumInt32(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
            } else if constexpr (std::is_same<int16_t, CAST_T>::value) {
                ComputeUniqueIdNumInt16(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
            } else { // uint8
                ComputeUniqueIdNumUint8(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
            }
        }
        uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
        return uniqueIdNum;
    }

    __aicore__ void inline ComputeUinqueIdTimes(LocalTensor<int32_t> uniqueIdCountLocal, uint32_t uniqueIdNum)
    {
        __local_mem__ int32_t *uniqueIdCountsAddr = (__local_mem__ int32_t *)uniqueIdCountLocal.GetPhyAddr();
        uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
        uint16_t loopSize = Ops::Base::CeilDiv(uniqueIdNum, vfLen);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int32_t> preReg;
            AscendC::MicroAPI::RegTensor<int32_t> postReg;
            AscendC::MicroAPI::RegTensor<int32_t> subReg;
            AscendC::MicroAPI::UnalignReg uIn;
            AscendC::MicroAPI::MaskReg maskReg;
            for (uint16_t i = 0; i < loopSize; ++i) {
                maskReg = AscendC::MicroAPI::UpdateMask<int32_t>(uniqueIdNum);
                auto startAddr = uniqueIdCountsAddr + i * vfLen;
                auto startAddrOfstOne = startAddr + 1;
                DataCopy(preReg, startAddr);
                AscendC::MicroAPI::DataCopyUnAlignPre(uIn, startAddrOfstOne);
                AscendC::MicroAPI::DataCopyUnAlign<int32_t>(postReg, uIn, startAddrOfstOne, vfLen);
                AscendC::MicroAPI::Sub(subReg, postReg, preReg, maskReg);
                DataCopy(startAddr, subReg, maskReg);
            }
        }
    }

    __aicore__ uint32_t inline SortAndComputeUniqueIdx(int64_t rowLen, LocalTensor<CAST_T> indicesSrcLocal,
                                                       LocalTensor<CAST_T> sortIndicesLocal,
                                                       LocalTensor<int32_t> uniqueIdxLocal,
                                                       LocalTensor<uint32_t> updatesOriginIdexLocal,
                                                       LocalTensor<uint8_t> sharedTmpBuffer)
    {
        int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(CAST_T);
        LocalTensor<CAST_T> shiftSortLocal = sortIndicesLocal[shiftOffset];
        AscendC::Sort<CAST_T, false, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, indicesSrcLocal,
                                                 sharedTmpBuffer, static_cast<uint32_t>(rowLen));
        Duplicate(sortIndicesLocal, (CAST_T)-1, shiftOffset);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        shiftSortLocal(rowLen) = -1;

        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        return ComputeUniqueIdNum(sortIndicesLocal, uniqueIdxLocal, rowLen);
    }

    __aicore__ void inline ComputeIdxCount(LocalTensor<IDX_T> indicesLocal, int64_t rowLen,
                                           int64_t &uniqueIdNumDuplicateIdx_)
    {
        LocalTensor<IDX_T> sortIndicesLocal = sortIndicesBuf_.Get<IDX_T>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
        LocalTensor<IDX_T> shiftSortLocal = sortIndicesLocal[shiftOffset_];
        AscendC::Sort<IDX_T, false, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, indicesLocal,
                                                static_cast<uint32_t>(rowLen));
        Duplicate(sortIndicesLocal, (IDX_T)-1, shiftOffset_);
        shiftSortLocal(rowLen) = -1;
        PipeBarrier<PIPE_V>();

        uniqueIdNumDuplicateIdx_ = ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);
        uniqueIdCountQue_.FreeTensor(uniqueIdCountLocal);
        updatesOriginIdexQue_.FreeTensor(updatesOriginIdexLocal);
        sortIndicesBuf_.FreeTensor(sortIndicesLocal);
    }

    __aicore__ void inline SortIndices(LocalTensor<CAST_T> indicesLocal, int64_t rowLen)
    {
        LocalTensor<CAST_T> sortIndicesLocal = sortIndicesBuf_.Get<CAST_T>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
        LocalTensor<IDX_T> updateSumIdxLocal = updateSumIdxQue_.AllocTensor<IDX_T>();
        LocalTensor<CAST_T> shiftSortLocal = sortIndicesLocal[shiftOffset_];
        AscendC::Sort<CAST_T, false, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, indicesLocal,
                                                 static_cast<uint32_t>(rowLen));
        Duplicate(sortIndicesLocal, (CAST_T)-1, shiftOffset_);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        shiftSortLocal(rowLen) = -1;
        PipeBarrier<PIPE_V>();

        uniqueIdNum_ = ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        for (uint32_t idx = 0; idx < uniqueIdNum_; idx++) {
            auto offset = uniqueIdCountLocal(idx);
            updateSumIdxLocal(idx) = shiftSortLocal(offset);
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        ComputeUinqueIdTimes(uniqueIdCountLocal, uniqueIdNum_);
        PipeBarrier<PIPE_V>();

        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
        updateSumIdxQue_.EnQue(updateSumIdxLocal);
    }

    __aicore__ void inline ComputeSumWithCast(LocalTensor<VAR_T> updatesLocal, LocalTensor<VAR_T> updateSumLocal,
                                              uint32_t uniqueIdNum, int64_t colLen)
    {
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.DeQue<uint32_t>();

        __local_mem__ VAR_T *updatesAddr = (__local_mem__ VAR_T *)updatesLocal.GetPhyAddr();
        __local_mem__ VAR_T *updateSumAddr = (__local_mem__ VAR_T *)updateSumLocal.GetPhyAddr();

        uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(float);
        int32_t loopSize = Ops::Base::CeilDiv(static_cast<uint32_t>(colLen), vfLen);
        int32_t idLocation = 0;
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);

        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
                AscendC::MicroAPI::RegTensor<float> sumReg;
                AscendC::MicroAPI::RegTensor<float> castReg;
                AscendC::MicroAPI::MaskReg maskReg;
                AscendC::MicroAPI::MaskReg zeroMask = AscendC::MicroAPI::CreateMask<float>();
                uint32_t maskLen = static_cast<uint32_t>(colLen);
                uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountLocal(i));
                for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
                    maskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                    AscendC::MicroAPI::Duplicate(sumReg, (float)0, zeroMask);
                    for (uint16_t k = 0; k < idRepeatTimes; k++) {
                        auto updatesOffet = updatesOriginIdexLocal(idLocation + k) * colLenAlignSize + j * vfLen;
                        ops::LoadOneTensorForDtypeT<VAR_T>(updatesAddr, castReg, maskReg, updatesOffet);
                        AscendC::MicroAPI::Add(sumReg, sumReg, castReg, maskReg);
                    }
                    auto updateSumAddrOfst = i * colLenAlignSize + j * vfLen;
                    ops::StoreOneTensorForDtypeT<VAR_T>(updateSumAddr, sumReg, maskReg, updateSumAddrOfst);
                }
                idLocation += idRepeatTimes;
            }
        }
        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
    }

    __aicore__ void inline ComputeSumWithOutCast(LocalTensor<VAR_T> updatesLocal, LocalTensor<VAR_T> updateSumLocal,
                                                 uint32_t uniqueIdNum, int64_t colLen)
    {
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.DeQue<uint32_t>();

        __local_mem__ selRegType *updatesAddr = (__local_mem__ selRegType *)updatesLocal.GetPhyAddr();
        __local_mem__ selRegType *updateSumAddr = (__local_mem__ selRegType *)updateSumLocal.GetPhyAddr();

        uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(selRegType);
        int32_t loopSize = (colLen + vfLen - 1) / vfLen;
        int32_t idLocation = 0;
        int64_t colLenAlignSize =
            Ops::Base::CeilAlign(colLen * sizeof(selRegType), UB_AGLIN_VALUE) / sizeof(selRegType);
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
                AscendC::MicroAPI::RegTensor<selRegType> sumReg;
                AscendC::MicroAPI::RegTensor<selRegType> updateReg;
                AscendC::MicroAPI::MaskReg maskReg;
                AscendC::MicroAPI::MaskReg zeroMask = AscendC::MicroAPI::CreateMask<selRegType>();
                uint32_t maskLen = static_cast<uint32_t>(colLen);
                uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountLocal(i));
                for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
                    maskReg = AscendC::MicroAPI::UpdateMask<selRegType>(maskLen);
                    AscendC::MicroAPI::Duplicate(sumReg, (selRegType)0, zeroMask);
                    for (uint16_t k = 0; k < idRepeatTimes; k++) {
                        auto updatesOffet = updatesOriginIdexLocal(idLocation + k) * colLenAlignSize + j * vfLen;
                        auto startAddr = updatesAddr + updatesOffet;
                        AscendC::MicroAPI::DataCopy(updateReg, startAddr);
                        AscendC::MicroAPI::Add(sumReg, sumReg, updateReg, maskReg);
                    }
                    auto updateSumAddrOfst = updateSumAddr + i * colLenAlignSize + j * vfLen;
                    AscendC::MicroAPI::DataCopy(updateSumAddrOfst, sumReg, maskReg);
                }
                idLocation += idRepeatTimes;
            }
        }
        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
    }

    __aicore__ void inline ExecConvertAlign(LocalTensor<VAR_T> updatesLocal, LocalTensor<float> updateSumLocal,
                                            int64_t rowLen, int64_t colLen)
    {
        __local_mem__ VAR_T *updatesAddr = (__local_mem__ VAR_T *)updatesLocal.GetPhyAddr();
        __local_mem__ float *updateSumAddr = (__local_mem__ float *)updateSumLocal.GetPhyAddr();

        uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(float);
        uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint32_t>(colLen), vfLen);
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
        int64_t colLenAlignFp32 = Ops::Base::CeilAlign(colLen * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);

        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < static_cast<uint16_t>(rowLen); i++) {
                AscendC::MicroAPI::RegTensor<float> castReg;
                AscendC::MicroAPI::MaskReg maskReg;
                uint32_t maskLen = static_cast<uint32_t>(colLen);
                for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
                    maskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                    auto updateSumAddrOfst = i * colLenAlignFp32 + j * vfLen;
                    auto updateAddrOfet = i * colLenAlignSize + j * vfLen;
                    ops::LoadOneTensorForDtypeT<float>(updateSumAddr, castReg, maskReg, updateSumAddrOfst);
                    ops::StoreOneTensorForDtypeT<VAR_T>(updatesAddr, castReg, maskReg, updateAddrOfet);
                }
            }
        }
    }

    __aicore__ inline void CopyOutSplitPre(LocalTensor<IDX_T> indicesLocal, LocalTensor<VAR_T> updateOutLocal,
                                           int64_t startPreAxis, int64_t preLen, int64_t colLen, int64_t colIdx)
    {
        LocalTensor<atomicSelType> updateOutLocalCast = updateOutLocal.template ReinterpretCast<atomicSelType>();
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
        if constexpr (IsSameType<VAR_T, bool>::value) {
            SetAtomicMax<int8_t>();
            for (int64_t preAxisIdx = 0; preAxisIdx < preLen; preAxisIdx++) {
                int64_t rowLocalOfset = preAxisIdx * updatesInAxis_;
                for (int64_t i = 0; i < updatesInAxis_; i++) {
                    int64_t rowOutOfset = ((startPreAxis + preAxisIdx) * varInAxis_ + indicesLocal(i)) * afterAxis_;
                    int64_t outOfset = rowOutOfset + colIdx * afterAxisFactor_;
                    int64_t localOfset = (rowLocalOfset + i) * colLenAlignSize;
                    CopyOut<int8_t>(varGm_[outOfset], updateOutLocalCast[localOfset], colLen);
                }
            }
            SetAtomicNone();
        } else {
            SetAtomicAdd<atomicSelType>();
            for (int64_t preAxisIdx = 0; preAxisIdx < preLen; preAxisIdx++) {
                int64_t rowLocalOfset = preAxisIdx * updatesInAxis_;
                for (int64_t i = 0; i < updatesInAxis_; i++) {
                    int64_t rowOutOfset = ((startPreAxis + preAxisIdx) * varInAxis_ + indicesLocal(i)) * afterAxis_;
                    int64_t outOfset = rowOutOfset + colIdx * afterAxisFactor_;
                    int64_t localOfset = (rowLocalOfset + i) * colLenAlignSize;
                    CopyOut<atomicSelType>(varGm_[outOfset], updateOutLocalCast[localOfset], colLen);
                }
            }
            SetAtomicNone();
        }
    }

    __aicore__ inline void CopyOutSplitIndices(LocalTensor<IDX_T> indicesLocal, LocalTensor<VAR_T> updateOutLocal,
                                               int64_t preAxisIdx, int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        LocalTensor<atomicSelType> updateOutLocalCast = updateOutLocal.template ReinterpretCast<atomicSelType>();
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
        if constexpr (IsSameType<VAR_T, bool>::value) {
            SetAtomicMax<int8_t>();
            for (int64_t i = 0; i < rowLen; i++) {
                int64_t rowOfset = (preAxisIdx * varInAxis_ + indicesLocal(i)) * afterAxis_;
                int64_t outOfset = rowOfset + colIdx * afterAxisFactor_;
                CopyOut<int8_t>(varGm_[outOfset], updateOutLocalCast[i * colLenAlignSize], colLen);
            }
            SetAtomicNone();
        } else {
            SetAtomicAdd<atomicSelType>();
            for (int64_t i = 0; i < rowLen; i++) {
                int64_t rowOfset = (preAxisIdx * varInAxis_ + indicesLocal(i)) * afterAxis_;
                int64_t outOfset = rowOfset + colIdx * afterAxisFactor_;
                CopyOut<atomicSelType>(varGm_[outOfset], updateOutLocalCast[i * colLenAlignSize], colLen);
            }
            SetAtomicNone();
        }
    }

    __aicore__ inline void CopyOutSplitAfter(LocalTensor<IDX_T> indicesLocal, LocalTensor<VAR_T> updateOutLocal,
                                             int64_t preAxisIdx, int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        LocalTensor<atomicSelType> updateOutLocalCast = updateOutLocal.template ReinterpretCast<atomicSelType>();
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
        if constexpr (IsSameType<VAR_T, bool>::value) {
            SetAtomicMax<int8_t>();
            for (int64_t i = 0; i < rowLen; i++) {
                int64_t rowOfset = (preAxisIdx * varInAxis_ + indicesLocal(i)) * afterAxis_;
                int64_t outOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
                CopyOut<int8_t>(varGm_[outOfset], updateOutLocalCast[i * colLenAlignSize], colLen);
            }
            SetAtomicNone();
        } else {
            SetAtomicAdd<atomicSelType>();
            for (int64_t i = 0; i < rowLen; i++) {
                int64_t rowOfset = (preAxisIdx * varInAxis_ + indicesLocal(i)) * afterAxis_;
                int64_t outOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
                CopyOut<atomicSelType>(varGm_[outOfset], updateOutLocalCast[i * colLenAlignSize], colLen);
            }
            SetAtomicNone();
        }
    }

    __aicore__ inline void IndicesSortCast(LocalTensor<IDX_T> indicesLocal, LocalTensor<CAST_T> indicesCastLocal,
                                           uint32_t indicesCount)
    {
        LocalTensor<int32_t> indicesCastTmpLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        if constexpr (CAST_MODE == CAST_4) { // int32 Cast uint8
            CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
            Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(MASK_UINT8),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
            Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        } else if constexpr (CAST_MODE == CAST_3) { // int64 Cast int16
            Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
            Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
        } else if constexpr (CAST_MODE == CAST_5) { // int64 Cast uint8
            CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
            Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(MASK_UINT8),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
            Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
            Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
        } else { // CAST_1 + CAST_2, int32 Cast int16 + int64 Cast int32
            Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        }
        uniqueIdCountQue_.FreeTensor(indicesCastTmpLocal);
    }
};
} // namespace MoeInplaceIndexAdd
#endif
