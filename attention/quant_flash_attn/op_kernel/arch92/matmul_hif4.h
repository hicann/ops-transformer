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
 * \file matmul_hif4.h
 * \brief HiF4 matmul functions migrated from matmul.h
 */
#ifndef MATMUL_HIF4_H_
#define MATMUL_HIF4_H_
#include "../../../common/op_kernel/matmul.h"
using namespace AscendC;

namespace fa_hif4_matmul {

static constexpr uint32_t HI_FP4_PTG_PCG_SCALE_PARAM = 64;
static constexpr uint32_t DB = 2;

// 搬入Rowsum所需全1补充矩阵到L0A
template <typename INPUT_T, typename T> // isLoadFull表示进行全载
__aicore__ inline void LoadDataRowsumToL0A(LocalTensor<INPUT_T>& aL0Tensor, const LocalTensor<INPUT_T>& aL1Tensor,
                                           const LocalTensor<T>& aL1ScaleTensor, int32_t kSplitSize, int32_t mSize)
{
    // GQA hif4 全量化
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.ifTranspose = false;
    loadData2DParamsA.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element // 0
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B // L1Aoffset / 64
    loadData2DParamsA.mStep = 1; // 以M*K矩阵为例, 源矩阵M轴方向搬运长度(S2向上对齐分形(512B),16*16个fp16→向上对齐16),
                                 // 单位为16 element, 取值范围：mStep∈[0, 255]
    loadData2DParamsA.kStep =
        kSplitSize / 2 /
        32; // 以M*K矩阵为例, 源矩阵K轴方向搬运长度(qkD个fp16), 单位为32B, 取值范围：nStep∈[0, 255] // 8 一个hif4占0.5B
    loadData2DParamsA.srcStride = 1;
    loadData2DParamsA.dstStride =
        (mSize + 16) /
        16; // 以M*K矩阵为例，在目标矩阵的K方向上，源矩阵中前一个分形起始地址与后一个分形起始地址的间隔, 单位：512B // 8
    LoadData2DMxParams scaleParamsA; // 代表scale矩阵k轴偏移方向，其中32是相对key的shape倍数
    scaleParamsA.xStartPosition = 0; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
    scaleParamsA.yStartPosition =
        0; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。 / HI_FP4_PTG_PCG_SCALE_PARAM / 32
    scaleParamsA.xStep =
        1; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
    scaleParamsA.yStep =
        1; // 1个fp32伪装为2个bf16，故最小分型数量翻倍。源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0,
           // 255]。
    scaleParamsA.srcStride = 1;
    scaleParamsA.dstStride = 1; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
    LoadData(aL0Tensor, aL1Tensor, aL1ScaleTensor, loadData2DParamsA, scaleParamsA);
}

// 搬入Rowsum所需全1补充矩阵到L0A
template <typename INPUT_T, typename T> // isLoadFull表示进行全载
__aicore__ inline void LoadDataRowsumMxToL0A(LocalTensor<INPUT_T>& aL0Tensor, const LocalTensor<INPUT_T>& aL1Tensor,
                                             const LocalTensor<T>& aL1ScaleTensor, int32_t kSplitSize)
{
    // GQA hif4 全量化
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.ifTranspose = false;
    loadData2DParamsA.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element // 0
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B // L1Aoffset / 64
    loadData2DParamsA.mStep = 1; // 以M*K矩阵为例, 源矩阵M轴方向搬运长度(S2向上对齐分形(512B),16*16个fp16→向上对齐16),
                                 // 单位为16 element, 取值范围：mStep∈[0, 255]
    loadData2DParamsA.kStep =
        1; // 以M*K矩阵为例, 源矩阵K轴方向搬运长度(qkD个fp16), 单位为32B, 取值范围：nStep∈[0, 255] // 8
    loadData2DParamsA.srcStride = 1;
    loadData2DParamsA.dstStride =
        (128 + 16) /
        16; // 以M*K矩阵为例，在目标矩阵的K方向上，源矩阵中前一个分形起始地址与后一个分形起始地址的间隔, 单位：512B // 8
    LoadData2DMxParams scaleParamsA; // 代表scale矩阵k轴偏移方向，其中32是相对key的shape倍数
    scaleParamsA.xStartPosition = 0; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
    scaleParamsA.yStartPosition =
        0; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。 / HI_FP4_PTG_PCG_SCALE_PARAM / 32
    scaleParamsA.xStep =
        1; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
    scaleParamsA.yStep =
        kSplitSize / HI_FP4_PTG_PCG_SCALE_PARAM *
        2; // 1个fp32伪装为2个bf16，故最小分型数量翻倍。源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0,
           // 255]。
    scaleParamsA.srcStride = 1;
    scaleParamsA.dstStride = 1; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
    LoadData(aL0Tensor, aL1Tensor, aL1ScaleTensor, loadData2DParamsA, scaleParamsA);
}

template <typename INPUT_T, typename T>
__aicore__ inline void LoadDataMxToL0AHiF4Bmm1(LocalTensor<INPUT_T>& aL0Tensor, const LocalTensor<INPUT_T>& aL1Tensor,
                                               const LocalTensor<T>& aL1ScaleTensor, uint64_t L1Aoffset,
                                               uint64_t xStartPosition, uint32_t kSplitSize, uint32_t mSplitSize)
{
    // GQA hif4 全量化
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.mStartPosition = L1Aoffset / 16; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element // 0
    loadData2DParamsA.mStep =
        (mSplitSize + 15) / 16; // 以M*K矩阵为例, 源矩阵M轴方向搬运长度(S2向上对齐分形(512B),16*16个fp16→向上对齐16),
                                // 单位为16 element, 取值范围：mStep∈[0, 255]
    loadData2DParamsA.srcStride = loadData2DParamsA.mStep;
    loadData2DParamsA.dstStride =
        loadData2DParamsA
            .mStep; // 以M*K矩阵为例，在目标矩阵的K方向上，源矩阵中前一个分形起始地址与后一个分形起始地址的间隔,
                    // 单位：512B // 8
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B // L1Aoffset / 64
    loadData2DParamsA.kStep =
        kSplitSize / 64; // 以M*K矩阵为例, 源矩阵K轴方向搬运长度(qkD个fp16), 单位为32B, 取值范围：nStep∈[0, 255] // 8
    LoadData2DMxParams scaleParamsA; // 代表scale矩阵k轴偏移方向，其中32是相对key的shape倍数
    scaleParamsA.xStartPosition =
        xStartPosition; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
    scaleParamsA.yStartPosition =
        L1Aoffset * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。
                                                    // / HI_FP4_PTG_PCG_SCALE_PARAM / 32
    scaleParamsA.xStep =
        (mSplitSize + 15) /
        16; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
    scaleParamsA.yStep =
        (kSplitSize + 31) * 2 /
        HI_FP4_PTG_PCG_SCALE_PARAM; // 1个fp32伪装为2个bf16，故最小分型数量翻倍。源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0,
                                    // 255]。
    scaleParamsA.srcStride = scaleParamsA.yStep;
    scaleParamsA.dstStride =
        scaleParamsA.yStep; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
    LoadData(aL0Tensor, aL1Tensor, aL1ScaleTensor, loadData2DParamsA, scaleParamsA);
}

template <typename INPUT_T, typename T>
__aicore__ inline void LoadDataMxToL0BHiF4Bmm1(LocalTensor<INPUT_T>& bL0Tensor, const LocalTensor<INPUT_T>& bL1Tensor,
                                               const LocalTensor<T>& bL1ScaleTensor, uint64_t L1Boffset,
                                               uint64_t xStartPosition, uint32_t kSplitSize, uint32_t nSplitSize)
{
    // GQA hif4 全量化
    LoadData2DParamsV2 loadData2DParamsB; // 基础API LoadData的参数结构体
    loadData2DParamsB.mStartPosition = L1Boffset / 16; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsB.mStep =
        (nSplitSize + 15) / 16; // 以M*K矩阵为例, 源矩阵M轴方向搬运长度(S2向上对齐分形(512B),16*16个fp16→向上对齐16),
                                // 单位为16 element, 取值范围：mStep∈[0, 255]
    loadData2DParamsB.srcStride = loadData2DParamsB.mStep; // 都用s2BaseSize，mm1时为singleN，mm2时为singleK
    loadData2DParamsB.dstStride =
        loadData2DParamsB
            .mStep; // 以M*K矩阵为例，在目标矩阵的K方向上，源矩阵中前一个分形起始地址与后一个分形起始地址的间隔,
                    // 单位：512B // 8
    loadData2DParamsB.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B L1Boffset / 64
    loadData2DParamsB.kStep =
        kSplitSize / 64; // 以M*K矩阵为例, 源矩阵K轴方向搬运长度(qkD个fp16), 单位为32B, 取值范围：nStep∈[0, 255] // 8
    LoadData2DMxParams scaleParamsB; // 代表scale矩阵k轴偏移方向，其中32是相对key的shape倍数
    scaleParamsB.xStartPosition =
        xStartPosition; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
    scaleParamsB.yStartPosition =
        L1Boffset * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。
    scaleParamsB.xStep =
        (nSplitSize + 15) /
        16; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
    scaleParamsB.yStep =
        (kSplitSize + 31) * 2 /
        HI_FP4_PTG_PCG_SCALE_PARAM; // 1个fp32伪装为2个bf16，故最小分型数量翻倍。源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0,
                                    // 255]。
    scaleParamsB.srcStride = scaleParamsB.yStep;
    scaleParamsB.dstStride =
        scaleParamsB.yStep; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
    LoadData(bL0Tensor, bL1Tensor, bL1ScaleTensor, loadData2DParamsB, scaleParamsB);
}

template <typename A, typename B, typename C, typename D, typename E, uint32_t baseM, uint32_t baseN, uint32_t baseK>
__aicore__ inline void MatmulMxHiF4Bmm1First(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                             const LocalTensor<D>& aL1ScaleTensor, const LocalTensor<E>& bL1ScaleTensor,
                                             LocalTensor<A>& aL0BuffsDb, LocalTensor<B>& bL0BuffsDb,
                                             const LocalTensor<C>& cL0Tensor, const uint8_t subLoop,
                                             const uint32_t s1CalcSize, const uint32_t s2CalcSize)
{
    static constexpr int32_t EVENT_ID_C1_M_MTE1_PING = 3;
    static constexpr int32_t EVENT_ID_C1_MTE1_M_PING = 3;
    static constexpr int32_t EVENT_ID_C1_FIX_M_PING = 0;
    static constexpr int32_t EVENT_ID_C1_M_FIX_PING = 0;

    WaitFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_PING);
    LoadDataMxToL0AHiF4Bmm1<A, D>(aL0BuffsDb, aL1Tensor, aL1ScaleTensor, 0, 0, baseK, s2CalcSize);
    LoadDataMxToL0BHiF4Bmm1<B, E>(bL0BuffsDb, bL1Tensor, bL1ScaleTensor, 0, 0, baseK, s1CalcSize);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID_C1_MTE1_M_PING);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID_C1_MTE1_M_PING);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID_C1_FIX_M_PING);
    MmadParams mmadParams;
    mmadParams.m = s2CalcSize;
    mmadParams.n = s1CalcSize;
    mmadParams.k = baseK;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    mmadParams.disableGemv = true;
    Mmad(cL0Tensor, aL0BuffsDb, bL0BuffsDb, mmadParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_PING);
    SetFlag<HardEvent::M_FIX>(EVENT_ID_C1_M_FIX_PING);
}

template <typename A, typename B, typename C, typename D, typename E, uint32_t baseM, uint32_t baseN, uint32_t baseK>
__aicore__ inline void MatmulMxHiF4Bmm1(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                        const LocalTensor<D>& aL1ScaleTensor, const LocalTensor<E>& bL1ScaleTensor,
                                        LocalTensor<A>& aL0BuffsDb, LocalTensor<B>& bL0BuffsDb,
                                        const LocalTensor<C>& cL0Tensor, const uint8_t subLoop,
                                        const uint32_t s1CalcSize, const uint32_t s2CalcSize)
{
    static constexpr int32_t EVENT_ID_C1_M_MTE1_PING = 3;
    static constexpr int32_t EVENT_ID_C1_MTE1_M_PING = 3;
    static constexpr int32_t EVENT_ID_C1_FIX_M_PING = 0;
    static constexpr int32_t EVENT_ID_C1_M_FIX_PING = 0;

    WaitFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_PING + (subLoop % DB));
    LoadDataMxToL0AHiF4Bmm1<A, D>(aL0BuffsDb, aL1Tensor, aL1ScaleTensor, 0, 0, baseK, s2CalcSize);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID_C1_MTE1_M_PING + (subLoop % DB));
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID_C1_MTE1_M_PING + (subLoop % DB));
    WaitFlag<HardEvent::FIX_M>(EVENT_ID_C1_FIX_M_PING + subLoop % DB);
    MmadParams mmadParams;
    mmadParams.m = s2CalcSize;
    mmadParams.n = s1CalcSize;
    mmadParams.k = baseK;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    mmadParams.disableGemv = true;
    Mmad(cL0Tensor, aL0BuffsDb, bL0BuffsDb, mmadParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_C1_M_MTE1_PING + (subLoop % DB));
    SetFlag<HardEvent::M_FIX>(EVENT_ID_C1_M_FIX_PING + subLoop % DB);
}

template <typename INPUT_T, typename T>
__aicore__ inline void LoadDataMxfp8ToL0ABmm2(LocalTensor<mx_fp8_e4m3_t>& aL0Tensor,
                                              const LocalTensor<INPUT_T>& aL1Tensor,
                                              const LocalTensor<T>& aL1ScaleTensor, uint64_t L1Aoffset,
                                              uint64_t xStartPosition, uint32_t kSplitSize, uint32_t mSplitSize)
{
    // GQA hif4 全量化
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.ifTranspose = true;
    loadData2DParamsA.mStartPosition = L1Aoffset / 16; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element // 0
    loadData2DParamsA.mStep =
        (kSplitSize + 15) / 16; // 以M*K矩阵为例, 源矩阵M轴方向搬运长度(S2向上对齐分形(512B),16*16个fp16→向上对齐16),
                                // 单位为16 element, 取值范围：mStep∈[0, 255]
    loadData2DParamsA.srcStride = ((kSplitSize + 255) / 256 * 256) / 16; // loadData2DParamsA.mStep;
    loadData2DParamsA.dstStride =
        (mSplitSize + 15) /
        16; // 以M*K矩阵为例，在目标矩阵的K方向上，源矩阵中前一个分形起始地址与后一个分形起始地址的间隔, 单位：512B // 8
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B // L1Aoffset / 64
    loadData2DParamsA.kStep =
        mSplitSize / 32; // 以M*K矩阵为例, 源矩阵K轴方向搬运长度(qkD个fp16), 单位为32B, 取值范围：nStep∈[0, 255] // 8

    static constexpr uint32_t HIF4_GROUP_SIZE = 32;
    LoadData2DMxParams scaleParamsA; // 代表scale矩阵k轴偏移方向，其中32是相对key的shape倍数
    scaleParamsA.xStartPosition =
        xStartPosition; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
    scaleParamsA.yStartPosition =
        L1Aoffset / HIF4_GROUP_SIZE /
        2; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。 / HI_FP4_PTG_PCG_SCALE_PARAM / 32
    scaleParamsA.xStep =
        (mSplitSize + 15) /
        16; // 源矩阵X轴方向搬运长度，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。取值范围：xStep∈[0, 255]。
    scaleParamsA.yStep =
        (kSplitSize + 63) / HIF4_GROUP_SIZE /
        2; // 1个fp32伪装为2个bf16，故最小分型数量翻倍。源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0,
           // 255]。
    scaleParamsA.srcStride = ((kSplitSize + 255) / 256 * 256) / HIF4_GROUP_SIZE / 2; // scaleParamsA.yStep;
    scaleParamsA.dstStride =
        scaleParamsA.yStep; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
    LoadData(aL0Tensor, aL1Tensor, aL1ScaleTensor, loadData2DParamsA, scaleParamsA);
}

template <typename INPUT_T, typename T>
__aicore__ inline void LoadDataFif4ToL0BBmm2(LocalTensor<INPUT_T>& bL0Tensor, const LocalTensor<INPUT_T>& bL1Tensor,
                                             const LocalTensor<T>& bL1ScaleTensor, uint64_t L1Boffset,
                                             uint64_t xStartPosition, uint32_t kSplitSize, uint32_t mSplitSize)
{
    // GQA hif4 全量化
    LoadData2DParamsV2 loadData2DParamsB; // 基础API LoadData的参数结构体
    loadData2DParamsB.ifTranspose = true;
    loadData2DParamsB.mStartPosition = L1Boffset / 16; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsB.mStep =
        (kSplitSize + 15) / 16; // 以M*K矩阵为例, 源矩阵M轴方向搬运长度(S2向上对齐分形(512B),16*16个fp16→向上对齐16),
                                // 单位为16 element, 取值范围：mStep∈[0, 255]
    loadData2DParamsB.srcStride = loadData2DParamsB.mStep;    // 都用s2BaseSize，mm1时为singleN，mm2时为singleK
    loadData2DParamsB.dstStride = (mSplitSize + 15) / 16 + 1; // rowsum
    loadData2DParamsB.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B L1Boffset / 64
    loadData2DParamsB.kStep = mSplitSize / 64; // rowsum

    LoadData2DMxParams scaleParamsB; // 代表scale矩阵k轴偏移方向，其中32是相对key的shape倍数
    scaleParamsB.xStartPosition =
        xStartPosition; // 矩阵X轴方向的起始位置，即M维度方向，单位为1个分形（1个单位代表一个32B的分形）。
    scaleParamsB.yStartPosition =
        L1Boffset * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // 源矩阵Y轴方向的起始位置，即K维度方向，单位为32B。
    scaleParamsB.xStep = (mSplitSize + 15) / 16;
    scaleParamsB.yStep =
        (kSplitSize + 31) * 2 /
        HI_FP4_PTG_PCG_SCALE_PARAM; // 1个fp32伪装为2个bf16，故最小分型数量翻倍。源矩阵Y轴方向搬运长度，即K维度方向，单位为32B。取值范围：yStep∈[0,
                                    // 255]。
    scaleParamsB.srcStride = scaleParamsB.yStep;
    scaleParamsB.dstStride =
        scaleParamsB.yStep; // 目标矩阵X方向前一个分形起始地址与后一个分形起始地址的间隔，单位为32B。
    LoadData(bL0Tensor, bL1Tensor, bL1ScaleTensor, loadData2DParamsB, scaleParamsB);
}

template <typename A, typename B, typename C, typename D, typename E, uint32_t baseM, uint32_t baseN, uint32_t baseK>
__aicore__ inline void MatmulMxfp8HiF4Bmm2(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                           const LocalTensor<D>& aL1ScaleTensor, const LocalTensor<E>& bL1ScaleTensor,
                                           LocalTensor<mx_fp8_e4m3_t>& aL0BuffsDb, LocalTensor<B>& bL0BuffsDb,
                                           const LocalTensor<C>& cL0Tensor, const uint32_t kIdx)
{
    static constexpr int32_t EVENT_ID_M_MTE1_PING = 3;
    static constexpr int32_t EVENT_ID_M_MTE1_PONG = 4;
    static constexpr int32_t EVENT_ID_C2_MTE1_M_PING = 5;
    static constexpr int32_t EVENT_ID_C2_FIX_M = 2;
    static constexpr int32_t EVENT_ID_C2_M_FIX = 2;
    static constexpr int32_t TWO = 2;

    // P&Pscale Ping时使用id3/4，Pong时使用id5/6。P的Ping与V的Ping Pong地址复用，防止踩踏
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_PING + ((kIdx + 1) % DB) * TWO);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_PONG + ((kIdx + 1) % DB) * TWO);
    LoadDataMxfp8ToL0ABmm2<A, D>(aL0BuffsDb, aL1Tensor, aL1ScaleTensor, 0, 0, baseK, baseM);
    LoadDataFif4ToL0BBmm2<B, E>(bL0BuffsDb, bL1Tensor, bL1ScaleTensor, 0, 0, baseK, baseN);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID_C2_MTE1_M_PING + (kIdx % DB));
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID_C2_MTE1_M_PING + (kIdx % DB));
    MmadParams mmadParams;
    mmadParams.m = baseM;
    mmadParams.n = baseN + 16; // rowsum，N方向多一个分形
    mmadParams.k = baseK;
    mmadParams.cmatrixInitVal = (kIdx == 0); // true: C矩阵初始值为0。false：C矩阵初始值通过cmatrixSource参数进行配置。
    mmadParams.cmatrixSource = false; // C矩阵初始值来源于C01
    mmadParams.unitFlag = 0;
    mmadParams.disableGemv = true;
    Mmad(cL0Tensor, aL0BuffsDb, bL0BuffsDb, mmadParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_PING + ((kIdx + 1) % DB) * TWO);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_PONG + ((kIdx + 1) % DB) * TWO);
}

template <typename A, typename B, typename C, typename D, typename E, uint32_t baseM, uint32_t baseN, uint32_t baseK>
__aicore__ inline void MatmulMxfp8HiF4Bmm2Tail(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                               const LocalTensor<D>& aL1ScaleTensor,
                                               const LocalTensor<E>& bL1ScaleTensor,
                                               LocalTensor<mx_fp8_e4m3_t>& aL0BuffsDb, LocalTensor<B>& bL0BuffsDb,
                                               const LocalTensor<C>& cL0Tensor,
                                               const LocalTensor<uint16_t>& RowSumFullOneVandScale, const uint32_t kIdx,
                                               const uint32_t tailS2Align64)
{
    static constexpr int32_t EVENT_ID_M_MTE1_PING = 3;
    static constexpr int32_t EVENT_ID_M_MTE1_PONG = 4;
    static constexpr int32_t EVENT_ID_C2_MTE1_M_PING = 5;
    static constexpr int32_t EVENT_ID_C2_FIX_M = 2;
    static constexpr int32_t EVENT_ID_C2_M_FIX = 2;
    static constexpr int32_t TWO = 2;

    static constexpr uint32_t constOffset =
        (16 * baseK / 2) /
        2; // 16行为单位添加, splitS2 = 128, sizeof(hif4)=0.5B, 2因为AMLAFullOneVandScale为uint16，每次偏移2B
    LocalTensor<B> VTensorTmp = RowSumFullOneVandScale.template ReinterpretCast<B>();
    LocalTensor<uint16_t> VscaleTensorTmp = RowSumFullOneVandScale[constOffset];
    LocalTensor<B> tmpVL0BTensor = bL0BuffsDb[baseN * 64]; // 64: 32B C0_size
    LocalTensor<B> tmpVL0BScaleTensor = bL0BuffsDb[baseN * tailS2Align64];
    // P&Pscale Ping时使用id3/4，Pong时使用id5/6。P的Ping与V的Ping Pong地址复用，防止踩踏
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_PING +
                                ((kIdx + 1) % DB) * TWO); // P&Pscale Ping时使用id3/4，Pong时使用id5/6
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_PONG + ((kIdx + 1) % DB) * TWO);
    LoadDataMxfp8ToL0ABmm2<A, D>(aL0BuffsDb, aL1Tensor, aL1ScaleTensor, 0, 0, tailS2Align64, baseM); // baseK -> tileK

    LoadDataRowsumToL0A<B, uint16_t>(tmpVL0BTensor, VTensorTmp, VscaleTensorTmp, tailS2Align64,
                                     baseN); // C2在S2方向切128
    LoadDataRowsumMxToL0A<B, uint16_t>(tmpVL0BScaleTensor, VTensorTmp, VscaleTensorTmp, tailS2Align64);
    AscendC::PipeBarrier<PIPE_MTE1>();
    LoadDataFif4ToL0BBmm2<B, E>(bL0BuffsDb, bL1Tensor, bL1ScaleTensor, 0, 0, tailS2Align64, baseN);

    SetFlag<HardEvent::MTE1_M>(EVENT_ID_C2_MTE1_M_PING + (kIdx % DB));
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID_C2_MTE1_M_PING + (kIdx % DB));
    MmadParams mmadParams;
    mmadParams.m = baseM;
    mmadParams.n = baseN + 16; // rowsum，N方向多一个分形
    mmadParams.k = tailS2Align64;
    mmadParams.cmatrixInitVal = (kIdx == 0); // true: C矩阵初始值为0。false：C矩阵初始值通过cmatrixSource参数进行配置。
    mmadParams.cmatrixSource = false; // C矩阵初始值来源于C01
    mmadParams.unitFlag = 0;
    mmadParams.disableGemv = true;
    Mmad(cL0Tensor, aL0BuffsDb, bL0BuffsDb, mmadParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_PING + ((kIdx + 1) % DB) * TWO);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_PONG + ((kIdx + 1) % DB) * TWO);
}

} // namespace fa_hif4_matmul
#endif
