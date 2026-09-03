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
 * \file vf_mul_quant_compressor.h
 * \brief
 */

#ifndef VF_MUL_QUANT_COMPRESSOR_H
#define VF_MUL_QUANT_COMPRESSOR_H

#include "kernel_operator.h"
#include <cstdint>
#include "../quant_compressor_comm.h"
using namespace AscendC;
using namespace QuantCompressor;

template <typename T>
__simd_callee__ inline T SimdCeilDivT(T num1, T num2)
{
    if (num2 == 0) {
        return static_cast<T>(0);
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
struct ReduceMulRegList {
    Reg::RegTensor<T> vreg0;
    Reg::RegTensor<T> vreg1;
    Reg::RegTensor<T> vregMul;
    Reg::RegTensor<T> vregSum;
};

template <typename T>
__simd_callee__ void LoadMulAddVFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, ReduceMulRegList<T> &regList,
                                      uint64_t offset, uint32_t maskValue)
{
    Reg::MaskReg mask = Reg::UpdateMask<T>(maskValue);
    Reg::LoadAlign(regList.vreg0, kvAddr + offset);
    Reg::LoadAlign(regList.vreg1, scoreAddr + offset);
    Reg::Mul(regList.vregMul, regList.vreg0, regList.vreg1, mask);
    Reg::Add(regList.vregSum, regList.vregSum, regList.vregMul, mask);
}

template <typename T>
__simd_vf__ void MulReduceSumbase8VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                         const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                         const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    Reg::RegTensor<T> vregSum0;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskL32 = Reg::CreateMask<T, Reg::MaskPattern::VL32>();
    Reg::MaskReg maskL16 = Reg::CreateMask<T, Reg::MaskPattern::VL16>();
    Reg::MaskReg maskL8 = Reg::CreateMask<T, Reg::MaskPattern::VL8>();
    Reg::MaskReg maskH32;
    Reg::MaskReg maskH48;
    Reg::MaskReg maskH56;
    Reg::Not(maskH48, maskL16, mask);
    Reg::Not(maskH32, maskL32, mask);
    Reg::Not(maskH56, maskL8, mask);
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        Reg::Duplicate(regList.vregSum, 0, mask);
        // 当前仅支持coff * cmpRatio为2的幂的情况
        for (uint32_t rLoop = 0; rLoop < SimdCeilDivT(rCnt, 8U); rLoop++) {
            uint32_t dealLen = min((rCnt - rLoop * 8) * baseD, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, dealLen);
            offset += dealLen;
        }
        // 64 -> 32
        Reg::Squeeze<T, AscendC::Reg::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH32);
        Reg::Add(regList.vregSum, regList.vregSum, vregSum0, maskL32);

        // 32 -> 16
        Reg::Squeeze<T, AscendC::Reg::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH48);
        Reg::Add(regList.vregSum, regList.vregSum, vregSum0, maskL16);

        // 16 -> 8
        Reg::Squeeze<T, AscendC::Reg::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH56);
        Reg::Add(regList.vregSum, regList.vregSum, vregSum0, maskL8);

        Reg::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL8);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase16VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    Reg::RegTensor<T> vregSum0;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskL32 = Reg::CreateMask<T, Reg::MaskPattern::VL32>();
    Reg::MaskReg maskL16 = Reg::CreateMask<T, Reg::MaskPattern::VL16>();
    Reg::MaskReg maskH32;
    Reg::MaskReg maskH48;
    Reg::Not(maskH48, maskL16, mask);
    Reg::Not(maskH32, maskL32, mask);
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        Reg::Duplicate(regList.vregSum, 0, mask);
        // 当前仅支持coff * cmpRatio为2的幂的情况
        for (uint32_t rLoop = 0; rLoop < SimdCeilDivT(rCnt, 4U); rLoop++) {
            uint32_t dealLen = min((rCnt - rLoop * 4) * baseD, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, dealLen);
            offset += dealLen;
        }
        // 64 -> 32
        Reg::Squeeze<T, AscendC::Reg::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH32);
        Reg::Add(regList.vregSum, regList.vregSum, vregSum0, maskL32);

        // 32 -> 16
        Reg::Squeeze<T, AscendC::Reg::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH48);
        Reg::Add(regList.vregSum, regList.vregSum, vregSum0, maskL16);

        Reg::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL16);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase32VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    Reg::RegTensor<T> vregSum0;
    Reg::RegTensor<T> vregSum1;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskL32 = Reg::CreateMask<T, Reg::MaskPattern::VL32>();
    Reg::MaskReg maskH32;
    Reg::Not(maskH32, maskL32, mask);
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        Reg::Duplicate(regList.vregSum, 0, mask);
        // 当前仅支持coff * cmpRatio为2的幂的情况
        for (uint32_t rLoop = 0; rLoop < SimdCeilDivT(rCnt, 2U); rLoop++) {
            uint32_t dealLen = min((rCnt - rLoop * 2) * baseD, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, dealLen);
            offset += dealLen;
        }
        // 64 -> 32
        Reg::Squeeze<T, AscendC::Reg::GatherMaskMode::NO_STORE_REG>(vregSum0, regList.vregSum, maskH32);
        Reg::Add(regList.vregSum, regList.vregSum, vregSum0, maskL32);

        Reg::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL32);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase64VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        Reg::Duplicate(regList.vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, VF_D_SIZE_64);
            offset += baseD;
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase128VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[2];
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        Reg::Duplicate(regList[0].vregSum, 0, mask);
        Reg::Duplicate(regList[1].vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + VF_D_SIZE_64, VF_D_SIZE_64);
            offset += baseD;
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + VF_D_SIZE_64, regList[1].vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase256VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[4];
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        Reg::Duplicate(regList[0].vregSum, 0, mask);
        Reg::Duplicate(regList[1].vregSum, 0, mask);
        Reg::Duplicate(regList[2].vregSum, 0, mask);
        Reg::Duplicate(regList[3].vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + VF_D_SIZE_64, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[2], offset + 2 * VF_D_SIZE_64, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[3], offset + 3 * VF_D_SIZE_64, VF_D_SIZE_64);
            offset += baseD;
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + VF_D_SIZE_64, regList[1].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 2 * VF_D_SIZE_64, regList[2].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 3 * VF_D_SIZE_64, regList[3].vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase512VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[8];
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        Reg::Duplicate(regList[0].vregSum, 0, mask);
        Reg::Duplicate(regList[1].vregSum, 0, mask);
        Reg::Duplicate(regList[2].vregSum, 0, mask);
        Reg::Duplicate(regList[3].vregSum, 0, mask);
        Reg::Duplicate(regList[4].vregSum, 0, mask);
        Reg::Duplicate(regList[5].vregSum, 0, mask);
        Reg::Duplicate(regList[6].vregSum, 0, mask);
        Reg::Duplicate(regList[7].vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + VF_D_SIZE_64, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[2], offset + 2 * VF_D_SIZE_64, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[3], offset + 3 * VF_D_SIZE_64, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[4], offset + 4 * VF_D_SIZE_64, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[5], offset + 5 * VF_D_SIZE_64, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[6], offset + 6 * VF_D_SIZE_64, VF_D_SIZE_64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[7], offset + 7 * VF_D_SIZE_64, VF_D_SIZE_64);
            offset += baseD;
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + VF_D_SIZE_64, regList[1].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 2 * VF_D_SIZE_64, regList[2].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 3 * VF_D_SIZE_64, regList[3].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 4 * VF_D_SIZE_64, regList[4].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 5 * VF_D_SIZE_64, regList[5].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 6 * VF_D_SIZE_64, regList[6].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 7 * VF_D_SIZE_64, regList[7].vregSum, mask);
    }
}

/**
 * @brief MulReduceSumbaseVF 包含mul和reducesum
 * @param outputLocal 输出tensor []
 * @param coff
 * @param cmpRatio 压缩块大小
 * @param baseD  核内d轴切分大小
 * @param scLoopCnt  sc数,
 */

// 当前仅支持coff * cmpRatio为2的幂的情况
template <typename T>
__aicore__ inline void MulReduceSumbaseVF(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                          const LocalTensor<T> &outputLocal, const uint32_t coff,
                                          const uint32_t cmpRatio, const uint32_t baseD, const uint32_t scLoopCnt)
{
    __ubuf__ T *kvAddr = (__ubuf__ T *)kvLocal.GetPhyAddr();
    __ubuf__ T *scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)outputLocal.GetPhyAddr();
    if (baseD == VF_D_SIZE_8) {
        MulReduceSumbase8VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == VF_D_SIZE_16) {
        MulReduceSumbase16VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == VF_D_SIZE_32) {
        MulReduceSumbase32VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == VF_D_SIZE_64) {
        MulReduceSumbase64VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == VF_D_SIZE_128) {
        MulReduceSumbase128VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == VF_D_SIZE_256) {
        MulReduceSumbase256VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == VF_D_SIZE_512) {
        MulReduceSumbase512VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    }
}

template <typename T>
__simd_vf__ void MulsVFImpl(__ubuf__ T *dstAddr, __ubuf__ T *srcAddr, float scalarValue, uint32_t cnt,
                            uint16_t repeatTimes)
{
    Reg::RegTensor<T> srcReg;
    Reg::RegTensor<T> dstReg;
    Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = Reg::UpdateMask<T>(cnt);
        Reg::LoadAlign(srcReg, srcAddr + i * VF_D_SIZE_64);
        Reg::Muls(dstReg, srcReg, scalarValue, mask);
        Reg::StoreAlign(dstAddr + i * VF_D_SIZE_64, dstReg, mask);
    }
}

/**
 * @brief MulsVF 标量乘法调度入口
 * @param srcLocal  输入LocalTensor
 * @param dstLocal  输出LocalTensor
 * @param scalar    标量乘数
 * @param baseD     d轴元素个数
 */
template <typename T>
__aicore__ inline void MulsVF(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, float scalar,
                              uint32_t row, uint32_t col)
{
    uint32_t cnt = row * col;
    uint16_t repeatTimes = static_cast<uint16_t>(SimdCeilDivT(cnt, VF_D_SIZE_64));
    __ubuf__ T *srcAddr = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T *dstAddr = (__ubuf__ T *)dstLocal.GetPhyAddr();
    MulsVFImpl<T>(dstAddr, srcAddr, scalar, cnt, repeatTimes);
}

#endif
