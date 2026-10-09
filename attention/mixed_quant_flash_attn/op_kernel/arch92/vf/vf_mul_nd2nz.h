/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file vf_mul_nd2nz.h
 * \brief
 */
#ifndef VF_MUL_ND2NZ_H
#define VF_MUL_ND2NZ_H

#include "kernel_tensor.h"

namespace AscendC {
/* **************************************************************************************************
 * mul + ND_2_NZ                                             *
 * ************************************************************************************************* */
/*
 * @ingroup MulND2NZ
 * @brief compute :res = MUL_ND_2_NZ(b16)
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input src LocalTensor
 */
template <typename T, uint8_t mode = 0>
__aicore__ inline void MulND2NZ(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                const LocalTensor<T>& antiQuantUb, uint32_t srcM, uint32_t realM, uint32_t srcN)
{
    constexpr uint32_t blockSize = 32;
    constexpr uint32_t blockN = blockSize / sizeof(T);
    const uint32_t fullExeSize = 128;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    __ubuf__ T* srcAddr = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T* antiAddr = (__ubuf__ T*)antiQuantUb.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    __ubuf__ T* dstAddr = (__ubuf__ T*)dstTensor.GetPhyAddr();
    uint16_t blockStride = ((srcM + 1) * blockN) * sizeof(T) / blockSize;
    constexpr uint16_t repeatStride = 1;

    if constexpr (mode == 0) {
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<T> vregRes;
                MicroAPI::RegTensor<T> vregAnti;
                MicroAPI::DataCopy<T>(vregAnti, antiAddr);
                MicroAPI::MaskReg pregRealExe = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
                // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]
                for (uint16_t m = 0; m < static_cast<uint16_t>(realM); m++) {
                    MicroAPI::DataCopy<T>(vregRes, (srcAddr + m * srcN));
                    MicroAPI::Mul<T>(vregRes, vregRes, vregAnti, pregRealExe);
                    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, vregRes, blockStride,
                                                                                repeatStride, pregRealExe);
                }
            }
        }
    } else if constexpr (mode == 1) {
        __ubuf__ T* srcAddr1 = (__ubuf__ T*)(srcTensor.GetPhyAddr() + fullExeSize * sizeof(T));
        __ubuf__ T* dstAddr1 = (__ubuf__ T*)(dstTensor.GetPhyAddr() + (srcM + 1) * fullExeSize * sizeof(T));
        __ubuf__ T* antiAddr1 = (__ubuf__ T*)(antiQuantUb.GetPhyAddr() + fullExeSize * sizeof(T));
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<T> vregRes;
                MicroAPI::RegTensor<T> vregAnti;
                MicroAPI::DataCopy<T>(vregAnti, antiAddr);
                MicroAPI::RegTensor<T> vregRes1;
                MicroAPI::RegTensor<T> vregAnti1;
                MicroAPI::DataCopy<T>(vregAnti1, antiAddr1);
                MicroAPI::MaskReg pregFullExe = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
                for (uint16_t m = 0; m < static_cast<uint16_t>(realM); m++) {
                    MicroAPI::DataCopy<T>(vregRes, (srcAddr + m * srcN));
                    MicroAPI::DataCopy<T>(vregRes1, (srcAddr1 + m * srcN));
                    MicroAPI::Mul<T>(vregRes, vregRes, vregAnti, pregFullExe);
                    MicroAPI::Mul<T>(vregRes1, vregRes1, vregAnti1, pregFullExe);
                    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, vregRes, blockStride,
                                                                                repeatStride, pregFullExe);
                    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr1, vregRes1, blockStride,
                                                                                repeatStride, pregFullExe);
                }
            }
        }
    } else if constexpr (mode == 2) {
        __ubuf__ T* srcAddr1 = (__ubuf__ T*)(srcTensor.GetPhyAddr() + fullExeSize * sizeof(T));
        __ubuf__ T* srcAddr2 = (__ubuf__ T*)(srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T));
        __ubuf__ T* srcAddr3 = (__ubuf__ T*)(srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T));
        __ubuf__ T* dstAddr1 = (__ubuf__ T*)(dstTensor.GetPhyAddr() + (srcM + 1) * fullExeSize * sizeof(T));
        __ubuf__ T* dstAddr2 = (__ubuf__ T*)(dstTensor.GetPhyAddr() + (srcM + 1) * fullExeSize * 2 * sizeof(T));
        __ubuf__ T* dstAddr3 = (__ubuf__ T*)(dstTensor.GetPhyAddr() + (srcM + 1) * fullExeSize * 3 * sizeof(T));
        __ubuf__ T* antiAddr1 = (__ubuf__ T*)(antiQuantUb.GetPhyAddr() + fullExeSize * sizeof(T));
        __ubuf__ T* antiAddr2 = (__ubuf__ T*)(antiQuantUb.GetPhyAddr() + fullExeSize * 2 * sizeof(T));
        __ubuf__ T* antiAddr3 = (__ubuf__ T*)(antiQuantUb.GetPhyAddr() + fullExeSize * 3 * sizeof(T));
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<T> vregRes;
                MicroAPI::RegTensor<T> vregAnti;
                MicroAPI::DataCopy<T>(vregAnti, antiAddr);
                MicroAPI::RegTensor<T> vregRes1;
                MicroAPI::RegTensor<T> vregAnti1;
                MicroAPI::DataCopy<T>(vregAnti1, antiAddr1);
                MicroAPI::RegTensor<T> vregRes2;
                MicroAPI::RegTensor<T> vregAnti2;
                MicroAPI::DataCopy<T>(vregAnti2, antiAddr2);
                MicroAPI::RegTensor<T> vregRes3;
                MicroAPI::RegTensor<T> vregAnti3;
                MicroAPI::DataCopy<T>(vregAnti3, antiAddr3);
                MicroAPI::MaskReg pregFullExe = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
                for (uint16_t m = 0; m < static_cast<uint16_t>(realM); m++) {
                    MicroAPI::DataCopy<T>(vregRes, (srcAddr + m * srcN));
                    MicroAPI::DataCopy<T>(vregRes1, (srcAddr1 + m * srcN));
                    MicroAPI::DataCopy<T>(vregRes2, (srcAddr2 + m * srcN));
                    MicroAPI::DataCopy<T>(vregRes3, (srcAddr3 + m * srcN));
                    MicroAPI::Mul<T>(vregRes, vregRes, vregAnti, pregFullExe);
                    MicroAPI::Mul<T>(vregRes1, vregRes1, vregAnti1, pregFullExe);
                    MicroAPI::Mul<T>(vregRes2, vregRes2, vregAnti2, pregFullExe);
                    MicroAPI::Mul<T>(vregRes3, vregRes3, vregAnti3, pregFullExe);
                    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, vregRes, blockStride,
                                                                                repeatStride, pregFullExe);
                    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr1, vregRes1, blockStride,
                                                                                repeatStride, pregFullExe);
                    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr2, vregRes2, blockStride,
                                                                                repeatStride, pregFullExe);
                    MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr3, vregRes3, blockStride,
                                                                                repeatStride, pregFullExe);
                }
            }
        }
    }
}

} // namespace AscendC

#endif // VF_MUL_ND2NZ_H
