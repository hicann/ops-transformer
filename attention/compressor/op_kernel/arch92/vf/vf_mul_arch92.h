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
 * \file vf_mul.h
 * \brief
 */

#ifndef VF_MUL_H
#define VF_MUL_H

#include "kernel_operator.h"
#include <cstdint>
using namespace AscendC;

constexpr uint32_t FLOATBYTE = 4;
constexpr uint32_t baseD8 = 8;
constexpr uint32_t baseD16 = 16;
constexpr uint32_t baseD32 = 32;
constexpr uint32_t baseD64 = 64;
constexpr uint32_t baseD128 = 128;
constexpr uint32_t baseD256 = 256;
constexpr uint32_t baseD512 = 512;

template <typename T>
struct ReduceMulRegList {
    Reg::RegTensor<T> vreg0;
    Reg::RegTensor<T> vreg1;
    Reg::RegTensor<T> vregMul;
    Reg::RegTensor<T> vregSum;
};

/* 8 行窗口（coff*cmpRatio==8）在寄存器内一步完成 softmax 与加权归约，替掉
 * SoftmaxDN(SoftmaxDnVF) + KvMulReduceScore(MulReduceSumbaseVF) 两步，省掉概率写回 UB 再读出的往返。
 * 与旧两步路径的累加顺序对应关系（T=float，dDealSize 为本函数 baseD64 的整数倍）：
 *   1) max：旧路径偶数行链式 max、奇数行链式 max，再取两者 max
 *      （见 SoftmaxDndBase64 的 max00/max01 与 Max(max00, max00, max01)）；
 *      maxEven/maxOdd 同树。max 精确且可结合，与逐行 max 等价。
 *   2) 分母：旧路径 vregSum 按行奇偶分成 0/1 两组各自链式累加，最后 Add(sum0, sum0, sum1)；
 *      各列独立、SIMD lane 不参与跳列合并，所以这里 sumEven/sumOdd 只是把 4 列的列分块
 *      换成一次 64 列，逐列加法树完全一致（dDealSize==64 已对 SoftmaxDndBase64 核对）。
 *      128/256/512 走 SoftmaxDndBase128 / DndBaseChunked，树形需用例复核。
 *   3) 加权归约：旧路径 MulReduceSumbase64VFImpl 从 rLoop=0 顺序 Mul+Add 到 coff*cmpRatio-1；
 *      这里 Div/Mul/Add 也按行 0 -> 7 顺序推进、共用同一个累加器，顺序一致。
 * 仍需 batch 一致性用例覆盖（BATCH_CONSISTENCY 且 coff*cmpRatio==8）。*/
template <typename T>
__simd_vf__ void SoftmaxMulReduce8VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
                                         uint32_t baseD, uint32_t scLoopCnt, T minValue)
{
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    for (uint16_t sc = 0; sc < scLoopCnt; ++sc) {
        for (uint16_t chunk = 0; chunk < baseD / baseD64; ++chunk) {
            const uint32_t offset = sc * 8 * baseD + chunk * baseD64;
            Reg::RegTensor<T> score[8];
            Reg::RegTensor<T> maxEven, maxOdd, sumEven, sumOdd, sum, result, kv, product;
            Reg::Duplicate(maxEven, minValue);
            Reg::Duplicate(maxOdd, minValue);
            Reg::LoadAlign(score[0], scoreAddr + offset + 0 * baseD);
            Reg::LoadAlign(score[1], scoreAddr + offset + 1 * baseD);
            Reg::Max(maxEven, maxEven, score[0], mask);
            Reg::Max(maxOdd, maxOdd, score[1], mask);
            Reg::LoadAlign(score[2], scoreAddr + offset + 2 * baseD);
            Reg::LoadAlign(score[3], scoreAddr + offset + 3 * baseD);
            Reg::Max(maxEven, maxEven, score[2], mask);
            Reg::Max(maxOdd, maxOdd, score[3], mask);
            Reg::LoadAlign(score[4], scoreAddr + offset + 4 * baseD);
            Reg::LoadAlign(score[5], scoreAddr + offset + 5 * baseD);
            Reg::Max(maxEven, maxEven, score[4], mask);
            Reg::Max(maxOdd, maxOdd, score[5], mask);
            Reg::LoadAlign(score[6], scoreAddr + offset + 6 * baseD);
            Reg::LoadAlign(score[7], scoreAddr + offset + 7 * baseD);
            Reg::Max(maxEven, maxEven, score[6], mask);
            Reg::Max(maxOdd, maxOdd, score[7], mask);
            Reg::Max(maxEven, maxEven, maxOdd, mask);
            Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(sumEven, 0, mask);
            Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, T>(sumOdd, 0, mask);
            Reg::FusedExpSub(score[0], score[0], maxEven, mask);
            Reg::FusedExpSub(score[1], score[1], maxEven, mask);
            Reg::Add(sumEven, score[0], sumEven, mask);
            Reg::Add(sumOdd, score[1], sumOdd, mask);
            Reg::FusedExpSub(score[2], score[2], maxEven, mask);
            Reg::FusedExpSub(score[3], score[3], maxEven, mask);
            Reg::Add(sumEven, score[2], sumEven, mask);
            Reg::Add(sumOdd, score[3], sumOdd, mask);
            Reg::FusedExpSub(score[4], score[4], maxEven, mask);
            Reg::FusedExpSub(score[5], score[5], maxEven, mask);
            Reg::Add(sumEven, score[4], sumEven, mask);
            Reg::Add(sumOdd, score[5], sumOdd, mask);
            Reg::FusedExpSub(score[6], score[6], maxEven, mask);
            Reg::FusedExpSub(score[7], score[7], maxEven, mask);
            Reg::Add(sumEven, score[6], sumEven, mask);
            Reg::Add(sumOdd, score[7], sumOdd, mask);
            Reg::Add(sum, sumEven, sumOdd, mask);
            Reg::Duplicate(result, 0, mask);
            Reg::Div(score[0], score[0], sum, mask);
            Reg::LoadAlign(kv, kvAddr + offset + 0 * baseD);
            Reg::Mul(product, kv, score[0], mask);
            Reg::Add(result, result, product, mask);
            Reg::Div(score[1], score[1], sum, mask);
            Reg::LoadAlign(kv, kvAddr + offset + 1 * baseD);
            Reg::Mul(product, kv, score[1], mask);
            Reg::Add(result, result, product, mask);
            Reg::Div(score[2], score[2], sum, mask);
            Reg::LoadAlign(kv, kvAddr + offset + 2 * baseD);
            Reg::Mul(product, kv, score[2], mask);
            Reg::Add(result, result, product, mask);
            Reg::Div(score[3], score[3], sum, mask);
            Reg::LoadAlign(kv, kvAddr + offset + 3 * baseD);
            Reg::Mul(product, kv, score[3], mask);
            Reg::Add(result, result, product, mask);
            Reg::Div(score[4], score[4], sum, mask);
            Reg::LoadAlign(kv, kvAddr + offset + 4 * baseD);
            Reg::Mul(product, kv, score[4], mask);
            Reg::Add(result, result, product, mask);
            Reg::Div(score[5], score[5], sum, mask);
            Reg::LoadAlign(kv, kvAddr + offset + 5 * baseD);
            Reg::Mul(product, kv, score[5], mask);
            Reg::Add(result, result, product, mask);
            Reg::Div(score[6], score[6], sum, mask);
            Reg::LoadAlign(kv, kvAddr + offset + 6 * baseD);
            Reg::Mul(product, kv, score[6], mask);
            Reg::Add(result, result, product, mask);
            Reg::Div(score[7], score[7], sum, mask);
            Reg::LoadAlign(kv, kvAddr + offset + 7 * baseD);
            Reg::Mul(product, kv, score[7], mask);
            Reg::Add(result, result, product, mask);
            Reg::StoreAlign(outputAddr + sc * baseD + chunk * baseD64, result, mask);
        }
    }
}

template <typename T>
__aicore__ inline void SoftmaxMulReduce8VF(const LocalTensor<T>& kvLocal, const LocalTensor<T>& scoreLocal,
                                           const LocalTensor<T>& outputLocal, uint32_t baseD, uint32_t scLoopCnt,
                                           T minValue)
{
    SoftmaxMulReduce8VFImpl((__ubuf__ T*)kvLocal.GetPhyAddr(), (__ubuf__ T*)scoreLocal.GetPhyAddr(),
                            (__ubuf__ T*)outputLocal.GetPhyAddr(), baseD, scLoopCnt, minValue);
}

template <typename T>
__simd_callee__ void LoadMulAddVFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, ReduceMulRegList<T>& regList,
                                      uint64_t offset, uint32_t maskValue)
{
    Reg::MaskReg mask = Reg::UpdateMask<T>(maskValue);
    Reg::LoadAlign(regList.vreg0, kvAddr + offset);
    Reg::LoadAlign(regList.vreg1, scoreAddr + offset);
    Reg::Mul(regList.vregMul, regList.vreg0, regList.vreg1, mask);
    Reg::Add(regList.vregSum, regList.vregSum, regList.vregMul, mask);
}

template <typename T>
__simd_vf__ void MulReduceSumbase8VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
                                         const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                         const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskL8 = Reg::CreateMask<T, Reg::MaskPattern::VL8>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint16_t scLoop = 0; scLoop < static_cast<uint16_t>(scLoopCnt); scLoop++) {
        Reg::Duplicate(regList.vregSum, 0, mask);
        for (uint16_t rLoop = 0; rLoop < static_cast<uint16_t>(rCnt); rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, baseD);
            offset += baseD;
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL8);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase16VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskL16 = Reg::CreateMask<T, Reg::MaskPattern::VL16>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint16_t scLoop = 0; scLoop < static_cast<uint16_t>(scLoopCnt); scLoop++) {
        Reg::Duplicate(regList.vregSum, 0, mask);
        for (uint16_t rLoop = 0; rLoop < static_cast<uint16_t>(rCnt); rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, baseD);
            offset += baseD;
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL16);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase32VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskL32 = Reg::CreateMask<T, Reg::MaskPattern::VL32>();
    uint32_t offset = 0;
    uint32_t rCnt = coff * cmpRatio;
    for (uint16_t scLoop = 0; scLoop < static_cast<uint16_t>(scLoopCnt); scLoop++) {
        Reg::Duplicate(regList.vregSum, 0, mask);
        for (uint16_t rLoop = 0; rLoop < static_cast<uint16_t>(rCnt); rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset, baseD);
            offset += baseD;
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, maskL32);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase64VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    ReduceMulRegList<T> regList;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t rCnt = coff * cmpRatio;
    for (uint16_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        uint32_t offset = scLoop * rCnt * baseD;
        Reg::Duplicate(regList.vregSum, 0, mask);
        for (uint16_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList, offset + rLoop * baseD, baseD64);
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList.vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase128VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[2];
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t rCnt = coff * cmpRatio;
    for (uint16_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        uint32_t offset = scLoop * rCnt * baseD;
        Reg::Duplicate(regList[0].vregSum, 0, mask);
        Reg::Duplicate(regList[1].vregSum, 0, mask);
        for (uint16_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset + rLoop * baseD, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + rLoop * baseD + baseD64, baseD64);
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + baseD64, regList[1].vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase256VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[4];
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t rCnt = coff * cmpRatio;
    for (uint16_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        uint32_t offset = scLoop * rCnt * baseD;
        Reg::Duplicate(regList[0].vregSum, 0, mask);
        Reg::Duplicate(regList[1].vregSum, 0, mask);
        Reg::Duplicate(regList[2].vregSum, 0, mask);
        Reg::Duplicate(regList[3].vregSum, 0, mask);
        for (uint16_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset + rLoop * baseD, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + rLoop * baseD + baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[2], offset + rLoop * baseD + 2 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[3], offset + rLoop * baseD + 3 * baseD64, baseD64);
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + baseD64, regList[1].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 2 * baseD64, regList[2].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 3 * baseD64, regList[3].vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase512VFImpl(__ubuf__ T* kvAddr, __ubuf__ T* scoreAddr, __ubuf__ T* outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    ReduceMulRegList<T> regList[8];
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t rCnt = coff * cmpRatio;
    for (uint16_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        uint32_t offset = scLoop * rCnt * baseD;
        Reg::Duplicate(regList[0].vregSum, 0, mask);
        Reg::Duplicate(regList[1].vregSum, 0, mask);
        Reg::Duplicate(regList[2].vregSum, 0, mask);
        Reg::Duplicate(regList[3].vregSum, 0, mask);
        Reg::Duplicate(regList[4].vregSum, 0, mask);
        Reg::Duplicate(regList[5].vregSum, 0, mask);
        Reg::Duplicate(regList[6].vregSum, 0, mask);
        Reg::Duplicate(regList[7].vregSum, 0, mask);
        for (uint16_t rLoop = 0; rLoop < rCnt; rLoop++) {
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[0], offset + rLoop * baseD, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[1], offset + rLoop * baseD + baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[2], offset + rLoop * baseD + 2 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[3], offset + rLoop * baseD + 3 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[4], offset + rLoop * baseD + 4 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[5], offset + rLoop * baseD + 5 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[6], offset + rLoop * baseD + 6 * baseD64, baseD64);
            LoadMulAddVFImpl(kvAddr, scoreAddr, regList[7], offset + rLoop * baseD + 7 * baseD64, baseD64);
        }
        Reg::StoreAlign(outputAddr + scLoop * baseD, regList[0].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + baseD64, regList[1].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 2 * baseD64, regList[2].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 3 * baseD64, regList[3].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 4 * baseD64, regList[4].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 5 * baseD64, regList[5].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 6 * baseD64, regList[6].vregSum, mask);
        Reg::StoreAlign(outputAddr + scLoop * baseD + 7 * baseD64, regList[7].vregSum, mask);
    }
}

/**
 * @brief MulReduceSumbaseVF 逐元素乘法后按压缩块求和归约
 * @param kvLocal kv输入tensor [scLoopCnt * coff * cmpRatio * baseD]
 * @param scoreLocal score输入tensor [scLoopCnt * coff * cmpRatio * baseD]
 * @param outputLocal 输出tensor [scLoopCnt * baseD]
 * @param coff 重叠系数
 * @param cmpRatio 压缩块大小
 * @param baseD 核内d轴切分大小
 * @param scLoopCnt sc循环次数
 */

// 当前仅支持coff * cmpRatio为2的幂的情况
template <typename T>
__aicore__ inline void MulReduceSumbaseVF(const LocalTensor<T>& kvLocal, const LocalTensor<T>& scoreLocal,
                                          const LocalTensor<T>& outputLocal, const uint32_t coff,
                                          const uint32_t cmpRatio, const uint32_t baseD, const uint32_t scLoopCnt)
{
    __ubuf__ T* kvAddr = (__ubuf__ T*)kvLocal.GetPhyAddr();
    __ubuf__ T* scoreAddr = (__ubuf__ T*)scoreLocal.GetPhyAddr();
    __ubuf__ T* outputAddr = (__ubuf__ T*)outputLocal.GetPhyAddr();
    if (baseD == baseD8) {
        MulReduceSumbase8VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD16) {
        MulReduceSumbase16VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD32) {
        MulReduceSumbase32VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD64) {
        MulReduceSumbase64VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD128) {
        MulReduceSumbase128VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD256) {
        MulReduceSumbase256VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD512) {
        MulReduceSumbase512VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    }
}

#endif
