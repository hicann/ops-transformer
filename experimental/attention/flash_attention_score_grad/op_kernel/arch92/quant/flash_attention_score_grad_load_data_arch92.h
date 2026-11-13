/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Local copies of arch35's HIFLOAT8-aware L1->L0 helpers. L0/L1 layouts are unchanged.
#ifndef FLASH_ATTENTION_SCORE_GRAD_LOAD_DATA_ARCH92_QUANT_H_
#define FLASH_ATTENTION_SCORE_GRAD_LOAD_DATA_ARCH92_QUANT_H_
#include "../../../../common/op_kernel/matmul.h"
namespace FagBaseApi {
namespace quant {
using namespace AscendC;
using namespace fa_base_matmul;
template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t size)
{
    if constexpr (IsSameType<T, float>::value) {
        return ((size + 7) >> 3 << 3) >> 3;
    } else if constexpr ((IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                          IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value)) {
        return ((size + 31) >> 5 << 5) >> 5;
    } else {
        return ((size + 15) >> 4 << 4) >> 4;
    }
}

template <typename T>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor, const MMParam& mmParam,
                                     uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep =
            ((kSplitSize + 15) >> 4 << 4) >>
            4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16
               // element,取值范围：mStep属于[0,255]
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                      IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
            loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsA.kStep = GetBlockNum<T>(
            mSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsA.mStep =
            ((mSplitSize + 15) >> 4 << 4) >>
            4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16
               // element,取值范围：mStep属于[0,255]
        loadData2DParamsA.kStep = GetBlockNum<T>(
            kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsA.ifTranspose) {
            loadData2DParamsA.kStep = CeilAlign(loadData2DParamsA.kStep, K_STEP_ALIGN_BASE);
        }
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        // 配合ub->L1使用 256 * 32 / 256
        // 64搬运对齐
        loadData2DParamsA.srcStride =
            loadData2DParamsA.ifTranspose ?
                ((kSplitSize + 63) >> 6 << 6) >> 4 :
                ((mSplitSize + 31) >> 5 << 5) >>
                    4; // 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    } else {
        loadData2DParamsA.srcStride =
            loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) >> 4 : loadData2DParamsA.mStep;
    }
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose ? (mSplitSize + 15) >> 4 : loadData2DParamsA.mStep;
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsA.ifTranspose && (loadData2DParamsA.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsA.mStep + 1) >> 1;
            loadData2DParamsA.mStep = M_STEP_ALIGN_BASE;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (mSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsA.mStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsA.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(aL0Tensor[dstOffset], aL1Tensor[L1Aoffset], loadData2DParamsA);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
        }
    } else {
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
    }
}

template <typename T>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor, const MMParam& mmParam,
                                     uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB; // 基础API LoadData的参数结构体
    loadData2DParamsB.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsB.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsB.ifTranspose = !mmParam.isRightTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsB.ifTranspose) {
        loadData2DParamsB.mStep =
            ((kSplitSize + 15) >> 4 << 4) >>
            4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16
               // element,取值范围：mStep属于[0,255]
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                      IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
            loadData2DParamsB.mStep = (loadData2DParamsB.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsB.kStep = GetBlockNum<T>(
            nSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsB.mStep =
            ((nSplitSize + 15) >> 4 << 4) >>
            4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16
               // element,取值范围：mStep属于[0,255]
        loadData2DParamsB.kStep = GetBlockNum<T>(
            kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.kStep = CeilAlign(loadData2DParamsB.kStep, K_STEP_ALIGN_BASE);
        }
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.srcStride = ((kSplitSize + 31) >> 5 << 5) >> 4;
        } else {
            loadData2DParamsB.srcStride = ((nSplitSize + 31) >> 5 << 5) >> 4;
        }
    } else {
        loadData2DParamsB.srcStride =
            loadData2DParamsB.ifTranspose ?
                (((mmParam.singleK + 15) >> 4 << 4) >> 4) :
                (((mmParam.singleN + 15) >> 4 << 4) >>
                 4); // 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    }
    loadData2DParamsB.dstStride =
        loadData2DParamsB.ifTranspose ?
            (nSplitSize + 15) >> 4 :
            loadData2DParamsB
                .mStep; // 以M*K矩阵为例，目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsB.ifTranspose && (loadData2DParamsB.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsB.mStep + 1) >> 1;
            loadData2DParamsB.mStep = M_STEP_ALIGN_BASE;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (nSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsB.mStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsB.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(bL0Tensor[dstOffset], bL1Tensor[L1Boffset], loadData2DParamsB);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
        }
    } else {
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
    }
}
} // namespace quant
} // namespace FagBaseApi
#endif
