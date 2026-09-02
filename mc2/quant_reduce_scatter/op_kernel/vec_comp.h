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
 * \file vec_comp.h
 * \brief quant_all_reduce 与quant_reduce_scatter 反量化与ReduceSum计算相关 kernel代码公共部分
 */

#ifndef VEC_COMP_H
#define VEC_COMP_H

#include "adv_api/pad/broadcast.h"

namespace VectorComputeImpl {

using namespace AscendC;
// 后缀_BYTES 表示单位为字节大小B, _NUM 表示单位为个
constexpr static uint32_t KG_SCALE_TRANS_NUM = 8U; // KG量化时， scale 为 fp32, 32B对齐时单次转换32 / 4 = 8个
constexpr static uint32_t MX_SCALE_TRANS_NUM = 32U; // MX量化时，scale 为 fp8_e8m0, 32B对齐时单次转换32 / 1 = 32个
constexpr static uint32_t PER_GROUP_SIZE = 128U; // KG量化时，128个x数据共有一个scale
constexpr static uint32_t MX_SIZE = 32U;         // MX量化时，32个x数据共有一个scale
constexpr static uint32_t TWO_DIMS = 2U;         // 用于scale反量化时boardcast的数组维度

#define TemplateTypeClass typename XType, typename ScalesType, typename OutputType
#define TemplateType XType, ScalesType, OutputType

template <TemplateTypeClass>
class VectorCompute {
public:
    __aicore__ inline VectorCompute(){};
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void CastToFloat(LocalTensor<XType> &xTensor, LocalTensor<ScalesType> &scaleTensor);
    __aicore__ inline void DequantReduceSum(LocalTensor<XType> &xTensor, LocalTensor<ScalesType> &scaleTensor,
                                            LocalTensor<float> &sumTensor);
    __aicore__ inline void SetBlockSize(uint32_t elementsPerBlock);
    __aicore__ inline void SetScaleNums(uint32_t n)
    {
        scaleNumsPerBlock_ = n;
    }

private:
    uint32_t xNumPerBlock_{0};
    uint32_t scaleNumsPerBlock_{0};

    LocalTensor<float> scaleCalTensor_;
    LocalTensor<bfloat16_t> tempBf16Scale_;
    LocalTensor<float> xCastTemp_;

    TBuf<> xCastBuf_;
    TBuf<> xCastBf16Buf_;
    TBuf<> xCastfp16Buf_;
    TBuf<> tempBf16ScaleBuf_;
    TBuf<> castScaleBuf_;
    TBuf<> brcbBuf_;
};

template <TemplateTypeClass>
__aicore__ inline void VectorCompute<TemplateType>::InitBuffer(TPipe *tPipe)
{
    tPipe->InitBuffer(brcbBuf_, xNumPerBlock_ * sizeof(float));           // 用于scale BroadCast，动态大小
    tPipe->InitBuffer(xCastBuf_, xNumPerBlock_ * sizeof(float));          // 用于x Cast 成 fp32, 动态大小
    tPipe->InitBuffer(xCastBf16Buf_, xNumPerBlock_ * sizeof(bfloat16_t)); // 用于 x Cast 成 bf16, 动态大小
    tPipe->InitBuffer(xCastfp16Buf_, xNumPerBlock_ * sizeof(float16_t));  // 用于 x Cast 成 fp16, 动态大小

    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        // MX: fp8_e8m0 → bf16 → float
        tPipe->InitBuffer(tempBf16ScaleBuf_, scaleNumsPerBlock_ * sizeof(bfloat16_t));
        tPipe->InitBuffer(castScaleBuf_, scaleNumsPerBlock_ * sizeof(float));
    } else {
        // KG: scale 已经是 float，直接 ReinterpretCast
        tPipe->InitBuffer(castScaleBuf_, scaleNumsPerBlock_ * sizeof(float));
    }
}

/**
 * @brief 设置数据块大小（每个DataBlock包含的数据x元素数量）
 *
 * @param elementsPerBlock 每个数据块包含的x元素数量
 */
template <TemplateTypeClass>
__aicore__ inline void VectorCompute<TemplateType>::SetBlockSize(uint32_t elementsPerBlock)
{
    xNumPerBlock_ = elementsPerBlock;
}

/**
 * @brief 将量化数据x和缩放因子scale都转换为浮点数float
 *
 * 该函数根据输入数据类型执行不同的转换策略：
 * 1. 对于fp浮点4类型（fp4x2_e2m1_t/fp4x2_e1m2_t）：fp4 → bfloat16 → float32
 * 2. 对于int8类型：int8 → float16 → float32
 * 3. 对于其他fp8浮点类型（fp8_e5m2/fp8_e4m2）：直接转换为float32
 *
 * 同时处理缩放因子的转换和广播（一次处理）：
 * - 对于fp8_e8m0缩放因子：fp8_e8m0 → bfloat16 → float32 → MX量化将scale广播成32
 * - 对于其他缩放因子：直接转换为float32 → KG量化将scale广播成128
 *
 * @tparam TemplateType 模板类型，用于支持不同的数据类型
 *
 * @param xTensor 量化输入数据张量
 * @param scaleTensor 量化缩放系数张量
 *
 * @note 针对fp8_e8m0缩放因子使用微指令（VF_CALL<CastVf>）进行转换
 */
template <TemplateTypeClass>
__aicore__ inline void VectorCompute<TemplateType>::CastToFloat(LocalTensor<XType> &xTensor,
                                                                LocalTensor<ScalesType> &scaleTensor)
{
    xCastTemp_ = xCastBuf_.Get<float>();
    scaleCalTensor_ = brcbBuf_.Get<float>();
    LocalTensor<bfloat16_t> xTempBf16Tensor = xCastBf16Buf_.Get<bfloat16_t>();
    LocalTensor<float16_t> xTempfp16Tensor = xCastfp16Buf_.Get<float16_t>();
    LocalTensor<float> castLocalScale = castScaleBuf_.Get<float>();

    if constexpr (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value || AscendC::IsSameType<XType, fp4x2_e1m2_t>::value) {
        // fp4 -> bf16 -> fp32
        Duplicate<bfloat16_t>(xTempBf16Tensor, (bfloat16_t)0.0, xNumPerBlock_);
        Cast(xTempBf16Tensor, xTensor, RoundMode::CAST_NONE, xNumPerBlock_);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTempBf16Tensor, RoundMode::CAST_NONE, xNumPerBlock_);
        PipeBarrier<PIPE_V>();
    } else if constexpr (AscendC::IsSameType<XType, int8_t>::value) {
        // int8 -> fp16 -> fp32
        Duplicate<float16_t>(xTempfp16Tensor, (float16_t)0.0, xNumPerBlock_);
        Cast(xTempfp16Tensor, xTensor, RoundMode::CAST_NONE, xNumPerBlock_);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTempfp16Tensor, RoundMode::CAST_NONE, xNumPerBlock_);
        PipeBarrier<PIPE_V>();
    } else {
        // fp8 -> fp32
        Duplicate<float>(xCastTemp_, (float)0.0, xNumPerBlock_);
        PipeBarrier<PIPE_V>();
        Cast(xCastTemp_, xTensor, RoundMode::CAST_NONE, xNumPerBlock_);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        // MX量化：fp8_e8m0 -> bf16 -> fp32，每个scale覆盖MX_SIZE(32)个x元素
        // VF_CALL<CastVf>：fp8_e8m0->bf16的微指令，每次只能处理32B即MX_SCALE_TRANS_NUM(32)个元素
        //   （硬件VF向量单元每次处理一个32B数据块，fp8_e8m0为1B/元素，故上限为32）
        //   需按批次循环，不可一次传入totalScaleNum（实测精度失败验证）
        // Cast 和 BroadCast 支持任意长度，在循环外一次处理全部totalScaleNum个scale
        tempBf16Scale_ = tempBf16ScaleBuf_.Get<bfloat16_t>();
        uint32_t totalScaleNum = xNumPerBlock_ / MX_SIZE;
        uint32_t scaleLoopCnt = totalScaleNum / MX_SCALE_TRANS_NUM;
        uint32_t scaleTailNum = totalScaleNum % MX_SCALE_TRANS_NUM;
        __local_mem__ fp8_e8m0_t *srcPtr = (__local_mem__ fp8_e8m0_t *)scaleTensor.GetPhyAddr();
        __local_mem__ bfloat16_t *tempBf16ScalePtr = (__local_mem__ bfloat16_t *)tempBf16Scale_.GetPhyAddr();
        for (uint32_t loopIdx = 0; loopIdx < scaleLoopCnt; ++loopIdx) {
            Duplicate<bfloat16_t>(tempBf16Scale_[loopIdx * MX_SCALE_TRANS_NUM], (bfloat16_t)0.0, MX_SCALE_TRANS_NUM);
            PipeBarrier<PIPE_V>();
            VF_CALL<CastVf>(tempBf16ScalePtr + loopIdx * MX_SCALE_TRANS_NUM, srcPtr + loopIdx * MX_SCALE_TRANS_NUM,
                            MX_SCALE_TRANS_NUM);
            PipeBarrier<PIPE_V>();
        }
        if (scaleTailNum > 0) {
            Duplicate<bfloat16_t>(tempBf16Scale_[scaleLoopCnt * MX_SCALE_TRANS_NUM], (bfloat16_t)0.0, scaleTailNum);
            PipeBarrier<PIPE_V>();
            VF_CALL<CastVf>(tempBf16ScalePtr + scaleLoopCnt * MX_SCALE_TRANS_NUM,
                            srcPtr + scaleLoopCnt * MX_SCALE_TRANS_NUM, scaleTailNum);
            PipeBarrier<PIPE_V>();
        }
        // 一次性将全部totalScaleNum个bf16 scale转为float，再广播到xNumPerBlock_个元素
        Cast(castLocalScale, tempBf16Scale_, RoundMode::CAST_NONE, totalScaleNum);
        PipeBarrier<PIPE_V>();
        const uint32_t broadcastDst[TWO_DIMS]{totalScaleNum, MX_SIZE};
        const uint32_t broadcastSrc[TWO_DIMS]{totalScaleNum, 1};
        BroadCast<float, TWO_DIMS, 1>(scaleCalTensor_, castLocalScale, broadcastDst, broadcastSrc);
        PipeBarrier<PIPE_V>();
    } else {
        // KG量化：scale已为float，直接ReinterpretCast，一次BroadCast广播全部totalScaleNum个scale
        castLocalScale = scaleTensor.template ReinterpretCast<float>();
        uint32_t totalScaleNum = xNumPerBlock_ / PER_GROUP_SIZE;
        const uint32_t broadcastDst[TWO_DIMS]{totalScaleNum, PER_GROUP_SIZE};
        const uint32_t broadcastSrc[TWO_DIMS]{totalScaleNum, 1};
        BroadCast<float, TWO_DIMS, 1>(scaleCalTensor_, castLocalScale, broadcastDst, broadcastSrc);
        PipeBarrier<PIPE_V>();
    }
}

/**
 * @brief 向量反量化与归约求和计算
 *
 * 该函数执行以下操作序列：
 * 1. 将量化数据x和缩放因子scale都转换为浮点数float
 * 2. 反量化（乘以缩放因子）
 * 3. 将反量化结果累加到求和张量中
 *
 * @param xTensor 量化后的输入张量
 * @param scaleTensor 缩放因子张量
 * @param sumTensor 归约求和张量，结果累加到此张量
 *
 * @note 函数内部包含多次流水线同步（PipeBarrier<PIPE_V>）确保计算顺序
 */
template <TemplateTypeClass>
__aicore__ inline void VectorCompute<TemplateType>::DequantReduceSum(LocalTensor<XType> &xTensor,
                                                                     LocalTensor<ScalesType> &scaleTensor,
                                                                     LocalTensor<float> &sumTensor)
{
    // Cast成float计算
    CastToFloat(xTensor, scaleTensor);
    PipeBarrier<PIPE_V>();
    // 反量化
    Mul(xCastTemp_, xCastTemp_, scaleCalTensor_, xNumPerBlock_);
    PipeBarrier<PIPE_V>();
    // 累加
    Add(sumTensor, sumTensor, xCastTemp_, xNumPerBlock_);
    PipeBarrier<PIPE_V>();
}
} // namespace VectorComputeImpl
#endif // VEC_COMP_H
