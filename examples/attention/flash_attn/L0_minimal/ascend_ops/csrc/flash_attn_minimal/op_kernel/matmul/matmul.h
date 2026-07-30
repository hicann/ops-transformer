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
 * \file matmul.h
 * \brief
 */
#ifndef MATMUL_H
#define MATMUL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;

namespace fa_base_matmul {
constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose;
    bool isRightTranspose;
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true; // 默认值为true，true：在L1切K轴的场景中，表示首轮K
    uint32_t unitFlag = 0;  // 0：disable: 不配置unitFlag
                            // 2：enable: 行为在切K接口中（MatmulK），会将mmadParams.unitFlag设置为 0b10
                            // 3：enable: 行为在切K接口中（MatmulK），在k的最后一轮循环，会将mmadParams.unitFlag设置为 0b11
                            // 外部使用时，在外层k循环的最后一轮将该参数配置为3
    uint32_t realM = 0; // bmm2以s1realsize为M轴，不赋值时不影响现有代码逻辑
};

__aicore__ inline MMParam MakeMMParam(uint32_t singleM, uint32_t singleN, uint32_t singleK, bool isLeftTranspose,
                                      bool isRightTranspose, bool cmatrixInitVal = true, bool isOutKFisrt = true,
                                      uint32_t unitFlag = 0, uint32_t realM = 0)
{
    return {.singleM = singleM,
            .singleN = singleN,
            .singleK = singleK,
            .isLeftTranspose = isLeftTranspose,
            .isRightTranspose = isRightTranspose,
            .cmatrixInitVal = cmatrixInitVal,
            .isOutKFisrt = isOutKFisrt,
            .unitFlag = unitFlag,
            .realM = realM
            };
}

enum class ABLayout {
    MK = 0,
    KM = 1,
    KN = 2,
    NK = 3,
};

template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t size) {
    return ((size + 15) >> 4 << 4) >> 4;
}
// L1->L0A + 切k/切M/全载
template <typename T>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor, 
                                    const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsA.kStep = GetBlockNum<T>(mSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsA.kStep = GetBlockNum<T>(kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) >> 4 : loadData2DParamsA.mStep;
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose ? (mSplitSize + 15) >> 4 : loadData2DParamsA.mStep;
    LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
}

// L1->L0B + 切k/切M/全载
template <typename T>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                    const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB; // 基础API LoadData的参数结构体
    loadData2DParamsB.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsB.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsB.ifTranspose = !mmParam.isRightTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsB.ifTranspose) {
        loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsB.kStep = GetBlockNum<T>(nSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsB.kStep = GetBlockNum<T>(kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    loadData2DParamsB.srcStride = loadData2DParamsB.ifTranspose ? (((mmParam.singleK + 15) >> 4 << 4) >> 4) : (((mmParam.singleN + 15 ) >> 4 << 4) >> 4); // 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    loadData2DParamsB.dstStride = loadData2DParamsB.ifTranspose ? (nSplitSize + 15) >> 4 : loadData2DParamsB.mStep; // 以M*K矩阵为例，目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
}

// 全载
// 外部L1切入K时，需要传入cmatrixInitVal的标记
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulFull(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  LocalTensor<L0ADType> &aL0Tensor,
                                  LocalTensor<L0BDType> &bL0Tensor,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param,
                                  uint32_t l0abIdx,
                                  const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                                  const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    Mutex::Lock<PIPE_MTE1>(l0abIdx);
    LoadDataToL0A(aL0Tensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    LoadDataToL0B(bL0Tensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    Mutex::Unlock<PIPE_MTE1>(l0abIdx);

    Mutex::Lock<PIPE_M>(l0abIdx);

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
 
    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);

    Mutex::Unlock<PIPE_M>(l0abIdx);
}
}
#endif
