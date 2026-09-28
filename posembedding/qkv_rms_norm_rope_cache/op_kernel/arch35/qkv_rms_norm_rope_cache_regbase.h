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
 * \file qkv_rms_norm_rope_cache_regbase.h
 * \brief QkvRmsNormRopeCache arch35(Ascend950 / DAV_3510)regbase 内核
 *
 * 算法语义与 A2 完全一致(见 README):qkv 按 head_nums 拆 Q/K/V;
 *   Q: RmsNorm -> RoPE -> 写 q_out(ND)
 *   K: RmsNorm -> RoPE -> [Quant] -> Scatter 写 k_cache(PA_NZ)
 *   V: [Quant] -> Scatter 写 v_cache(PA_NZ)
 * 实现范式为 arch35 regbase(VF),不复用 A2 的 arch22 TIK 内核。
 *
 * 关键不变量:D == 128 == 2 * VL_FP32(VL_FP32 = 256B / sizeof(float) = 64),
 * 一行恰好驻留在 2 个向量寄存器里,因此没有尾块掩码。
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_REGBASE_H
#define QKV_RMS_NORM_ROPE_CACHE_REGBASE_H

#include "kernel_operator.h"
#include "qkv_rms_norm_rope_cache_regbase_platform.h"

namespace QkvRmsNormRopeCache {
using namespace AscendC;

namespace Reg = AscendC::Reg;

constexpr int64_t REGBASE_VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);
constexpr int64_t REGBASE_UB_BLOCK = 32;
// head dim 固定 128 = 2 * REGBASE_VL_FP32,host tiling 已硬校验
constexpr int64_t REGBASE_HEAD_DIM = 128;
constexpr int64_t REGBASE_HALF_DIM = REGBASE_HEAD_DIM / 2;
static_assert(REGBASE_HEAD_DIM == 2 * REGBASE_VL_FP32, "head dim must be exactly two fp32 vector registers");

// b16 -> fp32 的随路 cast trait
constexpr static Reg::CastTrait CAST_B16_TO_B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32 -> b16 的随路 cast trait(与 A2 的 CAST_RINT / CAST_NONE 语义对齐)
constexpr static Reg::CastTrait CAST_FP32_TO_B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// 量化三步链的 trait,与 A2 RoundFloat2Int8 等价:round-half-even 后 clamp 到 int8
constexpr static Reg::CastTrait CAST_FP32_TO_INT16 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr static Reg::CastTrait CAST_INT16_TO_FP16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
constexpr static Reg::CastTrait CAST_FP16_TO_INT8 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

// ---------------------------------------------------------------------------
// 通用访存工具(与 kv_rms_norm_rope_cache/arch35 的 regbase 工具同构)
// ---------------------------------------------------------------------------

// UB(原生 dtype) -> 寄存器(fp32);16bit 类型随路 unpack + cast
template <typename T>
__aicore__ inline void LoadTensorForDtypeT(__ubuf__ T *input, Reg::RegTensor<float> &dst, Reg::MaskReg &preg,
                                           uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ half *)(input) + (offset)));
        Cast<float, half, CAST_B16_TO_B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, ((__ubuf__ bfloat16_t *)(input) + (offset)));
        Cast<float, bfloat16_t, CAST_B16_TO_B32>(dst, xBf16, preg);
    } else {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dst, ((__ubuf__ float *)(input) + (offset)));
    }
}

// 寄存器(fp32) -> UB(原生 dtype)
template <typename T>
__aicore__ inline void StoreTensorForDtypeTOut(__ubuf__ T *dst, Reg::RegTensor<float> &src, Reg::MaskReg &preg,
                                               uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        Reg::RegTensor<T> xB16;
        Cast<T, float, CAST_FP32_TO_B16>(xB16, src, preg);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst + offset, xB16, preg);
    }
}

// 寄存器(fp32,一个 VL = 64 lane) -> UB(int8):round-half-even + clamp,落 64 个 int8。
// 与 A2 RoundFloat2Int8(fp32 -> int32 RINT -> half -> int8 TRUNC)语义等价。
__aicore__ inline void StoreInt8OneVl(__ubuf__ int8_t *&dst, Reg::RegTensor<float> &src, Reg::UnalignRegForStore &ureg,
                                      Reg::MaskReg &preg)
{
    Reg::RegTensor<int16_t> tmpInt16;
    Reg::RegTensor<half> tmpHalf;
    Reg::RegTensor<int8_t> quantInt8;
    Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
    Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
    Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);
    Pack((Reg::RegTensor<uint16_t> &)tmpInt16, (Reg::RegTensor<uint32_t> &)quantInt8);
    Pack((Reg::RegTensor<uint8_t> &)quantInt8, (Reg::RegTensor<uint16_t> &)tmpInt16);
    Reg::StoreUnAlign(dst, quantInt8, ureg, static_cast<uint32_t>(REGBASE_VL_FP32));
}

// 量化:(out / scale)[+ offset],scale/offset 按 head 索引,shape [N_head, D]
__aicore__ inline void QuantTwoVl(Reg::RegTensor<float> &low, Reg::RegTensor<float> &high, __ubuf__ float *scaleRow,
                                  __ubuf__ float *offsetRow, int64_t headIdx, bool hasOffset, Reg::MaskReg &preg)
{
    const uint32_t headOffset = static_cast<uint32_t>(headIdx * REGBASE_HEAD_DIM);
    Reg::RegTensor<float> scaleLow, scaleHigh;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(scaleLow, scaleRow + headOffset);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(scaleHigh, scaleRow + headOffset + REGBASE_VL_FP32);
    Reg::Div(low, low, scaleLow, preg);
    Reg::Div(high, high, scaleHigh, preg);
    if (hasOffset) {
        Reg::RegTensor<float> offsetLow, offsetHigh;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(offsetLow, offsetRow + headOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(offsetHigh, offsetRow + headOffset + REGBASE_VL_FP32);
        Reg::Add(low, low, offsetLow, preg);
        Reg::Add(high, high, offsetHigh, preg);
    }
}

// ---------------------------------------------------------------------------
// VF:一行(head_dim = 128)的 RmsNorm + RoPE,结果落到 ND 与 cache 两个目的地
// WRITE_CACHE=false 时只写 ND(Q 分支:q_out 与 q_out_before_quant 同值)
// ---------------------------------------------------------------------------
template <typename T_QKV, typename T_CACHE, bool WRITE_CACHE>
__aicore__ inline void VfNormRopeRow(__ubuf__ T_QKV *xRow, __ubuf__ T_QKV *cosRow, __ubuf__ T_QKV *sinRow,
                                     __ubuf__ T_QKV *gammaRow, float reciprocal, float epsilon, __ubuf__ T_QKV *ndRow,
                                     __ubuf__ T_CACHE *cacheRow, __ubuf__ float *scaleRow, __ubuf__ float *offsetRow,
                                     bool hasOffset, int64_t headIdx)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg pFull = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> xLow, xHigh;
        LoadTensorForDtypeT<T_QKV>(xRow, xLow, pFull, 0);
        LoadTensorForDtypeT<T_QKV>(xRow, xHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));

        // ---- y = x / sqrt(mean(x^2) + eps) * gamma ----
        Reg::RegTensor<float> sumSq;
        Reg::Mul(sumSq, xLow, xLow, pFull);
        Reg::MulAddDst(sumSq, xHigh, xHigh, pFull); // sumSq += xHigh^2
        Reg::ReduceSum(sumSq, sumSq, pFull);
        Reg::Muls(sumSq, sumSq, reciprocal, pFull);
        Reg::Adds(sumSq, sumSq, epsilon, pFull);
        Reg::Sqrt(sumSq, sumSq, pFull);
        Reg::Duplicate(sumSq, sumSq, pFull); // 广播 lane0
        Reg::Div(xLow, xLow, sumSq, pFull);
        Reg::Div(xHigh, xHigh, sumSq, pFull);
        Reg::RegTensor<float> gammaLow, gammaHigh;
        LoadTensorForDtypeT<T_QKV>(gammaRow, gammaLow, pFull, 0);
        LoadTensorForDtypeT<T_QKV>(gammaRow, gammaHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));
        Reg::Mul(xLow, xLow, gammaLow, pFull);
        Reg::Mul(xHigh, xHigh, gammaHigh, pFull);

        // ---- RoPE(half-and-half):out_lo = x_lo*cos_lo - x_hi*sin_lo;out_hi = x_hi*cos_hi + x_lo*sin_hi ----
        Reg::RegTensor<float> cosLow, cosHigh, sinLow, sinHigh;
        LoadTensorForDtypeT<T_QKV>(cosRow, cosLow, pFull, 0);
        LoadTensorForDtypeT<T_QKV>(cosRow, cosHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));
        LoadTensorForDtypeT<T_QKV>(sinRow, sinLow, pFull, 0);
        LoadTensorForDtypeT<T_QKV>(sinRow, sinHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));
        Reg::RegTensor<float> outLow, outHigh, tmp;
        Reg::Mul(outLow, xLow, cosLow, pFull);
        Reg::Mul(tmp, xHigh, sinLow, pFull);
        Reg::Sub(outLow, outLow, tmp, pFull);
        Reg::Mul(outHigh, xHigh, cosHigh, pFull);
        Reg::MulAddDst(outHigh, xLow, sinHigh, pFull);

        // ---- 量化前的值(ND 输出:q_out / k_out_before_quant) ----
        StoreTensorForDtypeTOut<T_QKV>(ndRow, outLow, pFull, 0);
        StoreTensorForDtypeTOut<T_QKV>(ndRow, outHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));

        // ---- cache 输出 ----
        if constexpr (WRITE_CACHE) {
            if constexpr (IsSameType<T_CACHE, int8_t>::value) {
                QuantTwoVl(outLow, outHigh, scaleRow, offsetRow, headIdx, hasOffset, pFull);
                __ubuf__ int8_t *dst = (__ubuf__ int8_t *)cacheRow;
                Reg::UnalignRegForStore ureg;
                StoreInt8OneVl(dst, outLow, ureg, pFull);
                StoreInt8OneVl(dst, outHigh, ureg, pFull);
                Reg::StoreUnAlignPost(dst, ureg, 0);
            } else {
                StoreTensorForDtypeTOut<T_CACHE>(cacheRow, outLow, pFull, 0);
                StoreTensorForDtypeTOut<T_CACHE>(cacheRow, outHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VF:V 分支(不做 RmsNorm / RoPE,可选量化)
// ---------------------------------------------------------------------------
template <typename T_QKV, typename T_CACHE>
__aicore__ inline void VfQuantRow(__ubuf__ T_QKV *xRow, __ubuf__ T_QKV *ndRow, __ubuf__ T_CACHE *cacheRow,
                                  __ubuf__ float *scaleRow, __ubuf__ float *offsetRow, bool hasOffset, int64_t headIdx)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg pFull = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> xLow, xHigh;
        LoadTensorForDtypeT<T_QKV>(xRow, xLow, pFull, 0);
        LoadTensorForDtypeT<T_QKV>(xRow, xHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));

        // 量化前的值(ND 输出:v_out_before_quant)
        StoreTensorForDtypeTOut<T_QKV>(ndRow, xLow, pFull, 0);
        StoreTensorForDtypeTOut<T_QKV>(ndRow, xHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));

        if constexpr (IsSameType<T_CACHE, int8_t>::value) {
            QuantTwoVl(xLow, xHigh, scaleRow, offsetRow, headIdx, hasOffset, pFull);
            __ubuf__ int8_t *dst = (__ubuf__ int8_t *)cacheRow;
            Reg::UnalignRegForStore ureg;
            StoreInt8OneVl(dst, xLow, ureg, pFull);
            StoreInt8OneVl(dst, xHigh, ureg, pFull);
            Reg::StoreUnAlignPost(dst, ureg, 0);
        } else {
            StoreTensorForDtypeTOut<T_CACHE>(cacheRow, xLow, pFull, 0);
            StoreTensorForDtypeTOut<T_CACHE>(cacheRow, xHigh, pFull, static_cast<uint32_t>(REGBASE_VL_FP32));
        }
    }
}

// ---------------------------------------------------------------------------
// 内核主体
// ---------------------------------------------------------------------------
template <typename T_QKV, typename T_K_CACHE, typename T_V_CACHE>
class QkvRmsNormRopeCacheRegbase {
public:
    __aicore__ inline QkvRmsNormRopeCacheRegbase(TPipe *pipe, const QkvRmsNormRopeCacheRegbaseTilingData *tiling)
        : pipe_(pipe),
          tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR qGamma, GM_ADDR kGamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index,
                                GM_ADDR qOut, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kScale, GM_ADDR vScale,
                                GM_ADDR kOffset, GM_ADDR vOffset, GM_ADDR qOutOut, GM_ADDR kCacheOut, GM_ADDR vCacheOut,
                                GM_ADDR qOutProto, GM_ADDR kOutProto, GM_ADDR vOutProto)
    {
        ParseTiling();
        BindGlobalTensors(qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, kScale, vScale, kOffset, vOffset,
                          qOutOut, kCacheOut, vCacheOut, qOutProto, kOutProto, vOutProto);
        CalCoreRange();
        InitUbBuffers();
    }

    __aicore__ inline void Process()
    {
        if (coreStart_ >= coreEnd_) {
            return; // 该核无 token 可算
        }
        LoadGamma();
        for (int64_t tokenStart = coreStart_; tokenStart < coreEnd_; tokenStart += ubFactor_) {
            const int64_t numTokens = (tokenStart + ubFactor_ <= coreEnd_) ? ubFactor_ : (coreEnd_ - tokenStart);
            CopyIn(tokenStart, numTokens);
            Compute(numTokens);
            CopyOut(tokenStart, numTokens);
        }
    }

private:
    __aicore__ inline void ParseTiling()
    {
        numHead_ = tiling_->numHead;
        qkvDim_ = tiling_->qkvDim;
        numHeadQ_ = tiling_->numHeadQ;
        numHeadK_ = tiling_->numHeadK;
        numHeadV_ = tiling_->numHeadV;
        blockSize_ = tiling_->blockSize;
        blockNum_ = tiling_->blockNum;
        epsilon_ = tiling_->epsilon;
        reciprocal_ = tiling_->reciprocal;
        isOutputQkv_ = tiling_->isOutputQkv;
        blockFactor_ = tiling_->blockFactor;
        ubFactor_ = tiling_->ubFactor;
        tokenSum_ = tiling_->batchSize * tiling_->seqLength;
        tokenStride_ = numHead_ * qkvDim_;
        inUbBytes_ = tiling_->inUbBytes;
        cosSinUbBytes_ = tiling_->cosSinUbBytes;
        qOutUbBytes_ = tiling_->qOutUbBytes;
        kProtoUbBytes_ = tiling_->kProtoUbBytes;
        vProtoUbBytes_ = tiling_->vProtoUbBytes;
        kCacheUbBytes_ = tiling_->kCacheUbBytes;
        vCacheUbBytes_ = tiling_->vCacheUbBytes;
        gammaUbBytes_ = tiling_->gammaUbBytes;
        quantUbBytes_ = tiling_->quantUbBytes;
    }

    __aicore__ inline void BindGlobalTensors(GM_ADDR qkv, GM_ADDR qGamma, GM_ADDR kGamma, GM_ADDR cos, GM_ADDR sin,
                                             GM_ADDR index, GM_ADDR qOut, GM_ADDR kCache, GM_ADDR vCache,
                                             GM_ADDR kScale, GM_ADDR vScale, GM_ADDR kOffset, GM_ADDR vOffset,
                                             GM_ADDR qOutOut, GM_ADDR kCacheOut, GM_ADDR vCacheOut, GM_ADDR qOutProto,
                                             GM_ADDR kOutProto, GM_ADDR vOutProto)
    {
        qkvGm_.SetGlobalBuffer((__gm__ T_QKV *)qkv);
        gammaQGm_.SetGlobalBuffer((__gm__ T_QKV *)qGamma);
        gammaKGm_.SetGlobalBuffer((__gm__ T_QKV *)kGamma);
        cosGm_.SetGlobalBuffer((__gm__ T_QKV *)cos);
        sinGm_.SetGlobalBuffer((__gm__ T_QKV *)sin);
        indexGm_.SetGlobalBuffer((__gm__ int64_t *)index);
        // 原地输出:graph / aclnn 通路下 in-place 输入(6/7/8)与输出(0/1/2)同地址;
        // kernel 单算子通路下两者是不同 buffer,故对输出再写一份(地址相同时跳过,零开销)。
        qOutGm_.SetGlobalBuffer((__gm__ T_QKV *)qOut);
        kCacheGm_.SetGlobalBuffer((__gm__ T_K_CACHE *)kCache);
        vCacheGm_.SetGlobalBuffer((__gm__ T_V_CACHE *)vCache);
        qOutIsAliased_ = (qOut == qOutOut);
        kCacheIsAliased_ = (kCache == kCacheOut);
        vCacheIsAliased_ = (vCache == vCacheOut);
        if (!qOutIsAliased_) {
            qOutOutGm_.SetGlobalBuffer((__gm__ T_QKV *)qOutOut);
        }
        if (!kCacheIsAliased_) {
            kCacheOutGm_.SetGlobalBuffer((__gm__ T_K_CACHE *)kCacheOut);
        }
        if (!vCacheIsAliased_) {
            vCacheOutGm_.SetGlobalBuffer((__gm__ T_V_CACHE *)vCacheOut);
        }
        // k_scale/v_scale 只在对应 cache 为 int8 时存在;非量化档传 nullptr,不绑定也不读
        if constexpr (IsSameType<T_K_CACHE, int8_t>::value) {
            kScaleGm_.SetGlobalBuffer((__gm__ float *)kScale);
        }
        if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
            vScaleGm_.SetGlobalBuffer((__gm__ float *)vScale);
        }
        if (kOffset != nullptr) {
            kOffsetGm_.SetGlobalBuffer((__gm__ float *)kOffset);
            hasOffsetK_ = true;
        }
        if (vOffset != nullptr) {
            vOffsetGm_.SetGlobalBuffer((__gm__ float *)vOffset);
            hasOffsetV_ = true;
        }
        if (qOutProto != nullptr) {
            qOutProtoGm_.SetGlobalBuffer((__gm__ T_QKV *)qOutProto);
        }
        if (kOutProto != nullptr) {
            kOutProtoGm_.SetGlobalBuffer((__gm__ T_QKV *)kOutProto);
        }
        if (vOutProto != nullptr) {
            vOutProtoGm_.SetGlobalBuffer((__gm__ T_QKV *)vOutProto);
        }
    }

    __aicore__ inline void CalCoreRange()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        coreStart_ = blockIdx * blockFactor_;
        if (coreStart_ > tokenSum_) {
            coreStart_ = tokenSum_;
        }
        const int64_t remain = tokenSum_ - coreStart_;
        coreEnd_ = coreStart_ + ((remain < blockFactor_) ? remain : blockFactor_);
    }

    // 所有 UB 字节数一律由 host tiling 下发,内核只透传(不重算)
    __aicore__ inline void InitUbBuffers()
    {
        pipe_->InitBuffer(inQueueX_, 1, inUbBytes_);
        pipe_->InitBuffer(inQueueCos_, 1, cosSinUbBytes_);
        pipe_->InitBuffer(inQueueSin_, 1, cosSinUbBytes_);
        pipe_->InitBuffer(outQueueQ_, 1, qOutUbBytes_);
        pipe_->InitBuffer(outQueueKProto_, 1, kProtoUbBytes_);
        pipe_->InitBuffer(outQueueVProto_, 1, vProtoUbBytes_);
        pipe_->InitBuffer(outQueueK_, 1, kCacheUbBytes_);
        pipe_->InitBuffer(outQueueV_, 1, vCacheUbBytes_);
        pipe_->InitBuffer(gammaBuf_, gammaUbBytes_);
        pipe_->InitBuffer(quantBuf_, quantUbBytes_);
    }

    // q_gamma / k_gamma 每核只搬一次,常驻 UB
    __aicore__ inline void LoadGamma()
    {
        LocalTensor<T_QKV> gamma = gammaBuf_.Get<T_QKV>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(qkvDim_ * sizeof(T_QKV));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<T_QKV> padParams{false, 0, 0, 0};
        DataCopyPad(gamma, gammaQGm_, copyParams, padParams);
        DataCopyPad(gamma[qkvDim_], gammaKGm_, copyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>();
    }

    // 每个 tile 搬一次量化因子:k_scale / k_offset / v_scale / v_offset,全 head 连续
    __aicore__ inline void LoadQuantFactors()
    {
        LocalTensor<float> quant = quantBuf_.Get<float>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        // 偏移必须与 Compute 里 kOffsetIdx/vScaleIdx/vOffsetIdx 的固定布局一致:
        // 即使某一档不量化(该块不搬),后面的块也不得前移。
        const int64_t kOffsetIdx = numHeadK_ * qkvDim_;
        const int64_t vScaleIdx = 2 * numHeadK_ * qkvDim_;
        const int64_t vOffsetIdx = (2 * numHeadK_ + numHeadV_) * qkvDim_;
        if constexpr (IsSameType<T_K_CACHE, int8_t>::value) {
            copyParams.blockLen = static_cast<uint32_t>(numHeadK_ * qkvDim_ * sizeof(float));
            DataCopyPad(quant[0], kScaleGm_, copyParams, padParams);
            if (hasOffsetK_) {
                DataCopyPad(quant[kOffsetIdx], kOffsetGm_, copyParams, padParams);
            }
        }
        if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
            copyParams.blockLen = static_cast<uint32_t>(numHeadV_ * qkvDim_ * sizeof(float));
            DataCopyPad(quant[vScaleIdx], vScaleGm_, copyParams, padParams);
            if (hasOffsetV_) {
                DataCopyPad(quant[vOffsetIdx], vOffsetGm_, copyParams, padParams);
            }
        }
        SetWaitFlag<HardEvent::MTE2_V>();
    }

    __aicore__ inline void CopyIn(int64_t tokenStart, int64_t numTokens)
    {
        DataCopyPadExtParams<T_QKV> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(numTokens);
        copyParams.blockLen = static_cast<uint32_t>(tokenStride_ * sizeof(T_QKV));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        LocalTensor<T_QKV> x = inQueueX_.AllocTensor<T_QKV>();
        DataCopyPad(x, qkvGm_[tokenStart * tokenStride_], copyParams, padParams);
        inQueueX_.EnQue(x);

        copyParams.blockLen = static_cast<uint32_t>(qkvDim_ * sizeof(T_QKV));
        LocalTensor<T_QKV> cos = inQueueCos_.AllocTensor<T_QKV>();
        DataCopyPad(cos, cosGm_[tokenStart * qkvDim_], copyParams, padParams);
        inQueueCos_.EnQue(cos);

        LocalTensor<T_QKV> sin = inQueueSin_.AllocTensor<T_QKV>();
        DataCopyPad(sin, sinGm_[tokenStart * qkvDim_], copyParams, padParams);
        inQueueSin_.EnQue(sin);

        LoadQuantFactors();
    }

    __aicore__ inline void Compute(int64_t numTokens)
    {
        LocalTensor<T_QKV> x = inQueueX_.DeQue<T_QKV>();
        LocalTensor<T_QKV> cos = inQueueCos_.DeQue<T_QKV>();
        LocalTensor<T_QKV> sin = inQueueSin_.DeQue<T_QKV>();

        LocalTensor<T_QKV> qOut = outQueueQ_.AllocTensor<T_QKV>();
        LocalTensor<T_QKV> kProto = outQueueKProto_.AllocTensor<T_QKV>();
        LocalTensor<T_QKV> vProto = outQueueVProto_.AllocTensor<T_QKV>();
        LocalTensor<T_K_CACHE> kOut = outQueueK_.AllocTensor<T_K_CACHE>();
        LocalTensor<T_V_CACHE> vOut = outQueueV_.AllocTensor<T_V_CACHE>();

        __ubuf__ T_QKV *xBuf = (__ubuf__ T_QKV *)x.GetPhyAddr();
        __ubuf__ T_QKV *cosBuf = (__ubuf__ T_QKV *)cos.GetPhyAddr();
        __ubuf__ T_QKV *sinBuf = (__ubuf__ T_QKV *)sin.GetPhyAddr();
        __ubuf__ T_QKV *gammaBuf = (__ubuf__ T_QKV *)gammaBuf_.Get<T_QKV>().GetPhyAddr();
        __ubuf__ T_QKV *qOutBuf = (__ubuf__ T_QKV *)qOut.GetPhyAddr();
        __ubuf__ T_QKV *kProtoBuf = (__ubuf__ T_QKV *)kProto.GetPhyAddr();
        __ubuf__ T_QKV *vProtoBuf = (__ubuf__ T_QKV *)vProto.GetPhyAddr();
        __ubuf__ T_K_CACHE *kOutBuf = (__ubuf__ T_K_CACHE *)kOut.GetPhyAddr();
        __ubuf__ T_V_CACHE *vOutBuf = (__ubuf__ T_V_CACHE *)vOut.GetPhyAddr();
        __ubuf__ float *quantBuf = (__ubuf__ float *)quantBuf_.Get<float>().GetPhyAddr();

        // quantBuf 布局: [k_scale | k_offset | v_scale | v_offset]
        const int64_t kOffsetIdx = numHeadK_ * qkvDim_;
        const int64_t vScaleIdx = 2 * numHeadK_ * qkvDim_;
        const int64_t vOffsetIdx = (2 * numHeadK_ + numHeadV_) * qkvDim_;

        // ---- Q 分支: RmsNorm -> RoPE -> q_out(ND),q 不量化 ----
        for (int64_t h = 0; h < numHeadQ_; ++h) {
            for (int64_t t = 0; t < numTokens; ++t) {
                const int64_t row = (t * numHeadQ_ + h) * qkvDim_;
                // gammaBuf 布局: [q_gamma | k_gamma]
                VfNormRopeRow<T_QKV, T_QKV, false>(xBuf + t * tokenStride_ + h * qkvDim_, cosBuf + t * qkvDim_,
                                                   sinBuf + t * qkvDim_, gammaBuf, reciprocal_, epsilon_, qOutBuf + row,
                                                   (__ubuf__ T_QKV *)nullptr, nullptr, nullptr, false, 0);
            }
        }

        // ---- K 分支: RmsNorm -> RoPE -> [Quant] -> Scatter ----
        for (int64_t h = 0; h < numHeadK_; ++h) {
            for (int64_t t = 0; t < numTokens; ++t) {
                const int64_t row = (t * numHeadK_ + h) * qkvDim_;
                VfNormRopeRow<T_QKV, T_K_CACHE, true>(xBuf + t * tokenStride_ + (numHeadQ_ + h) * qkvDim_,
                                                      cosBuf + t * qkvDim_, sinBuf + t * qkvDim_, gammaBuf + qkvDim_,
                                                      reciprocal_, epsilon_, kProtoBuf + row, kOutBuf + row, quantBuf,
                                                      quantBuf + kOffsetIdx, hasOffsetK_, h);
            }
        }

        // ---- V 分支: [Quant] -> Scatter,无 RmsNorm / RoPE ----
        for (int64_t h = 0; h < numHeadV_; ++h) {
            for (int64_t t = 0; t < numTokens; ++t) {
                const int64_t row = (t * numHeadV_ + h) * qkvDim_;
                VfQuantRow<T_QKV, T_V_CACHE>(xBuf + t * tokenStride_ + (numHeadQ_ + numHeadK_ + h) * qkvDim_,
                                             vProtoBuf + row, vOutBuf + row, quantBuf + vScaleIdx,
                                             quantBuf + vOffsetIdx, hasOffsetV_, h);
            }
        }

        inQueueX_.FreeTensor(x);
        inQueueCos_.FreeTensor(cos);
        inQueueSin_.FreeTensor(sin);
        outQueueQ_.EnQue(qOut);
        outQueueKProto_.EnQue(kProto);
        outQueueVProto_.EnQue(vProto);
        outQueueK_.EnQue(kOut);
        outQueueV_.EnQue(vOut);
    }

    __aicore__ inline void CopyOut(int64_t tokenStart, int64_t numTokens)
    {
        LocalTensor<T_QKV> qOut = outQueueQ_.DeQue<T_QKV>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(numTokens);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        copyParams.blockLen = static_cast<uint32_t>(numHeadQ_ * qkvDim_ * sizeof(T_QKV));
        DataCopyPad(qOutGm_[tokenStart * numHeadQ_ * qkvDim_], qOut, copyParams);
        if (!qOutIsAliased_) {
            DataCopyPad(qOutOutGm_[tokenStart * numHeadQ_ * qkvDim_], qOut, copyParams);
        }
        // q 不量化,q_out_before_quant 与 q_out 同值,同一份 UB 再写一路
        if (isOutputQkv_ != 0) {
            DataCopyPad(qOutProtoGm_[tokenStart * numHeadQ_ * qkvDim_], qOut, copyParams);
        }
        outQueueQ_.FreeTensor(qOut);

        LocalTensor<T_QKV> kProto = outQueueKProto_.DeQue<T_QKV>();
        if (isOutputQkv_ != 0) {
            copyParams.blockLen = static_cast<uint32_t>(numHeadK_ * qkvDim_ * sizeof(T_QKV));
            DataCopyPad(kOutProtoGm_[tokenStart * numHeadK_ * qkvDim_], kProto, copyParams);
        }
        outQueueKProto_.FreeTensor(kProto);

        LocalTensor<T_QKV> vProto = outQueueVProto_.DeQue<T_QKV>();
        if (isOutputQkv_ != 0) {
            copyParams.blockLen = static_cast<uint32_t>(numHeadV_ * qkvDim_ * sizeof(T_QKV));
            DataCopyPad(vOutProtoGm_[tokenStart * numHeadV_ * qkvDim_], vProto, copyParams);
        }
        outQueueVProto_.FreeTensor(vProto);

        // k_cache / v_cache: 逐 (token, head) 按 index 散射
        LocalTensor<T_K_CACHE> kOut = outQueueK_.DeQue<T_K_CACHE>();
        LocalTensor<T_V_CACHE> vOut = outQueueV_.DeQue<T_V_CACHE>();
        for (int64_t t = 0; t < numTokens; ++t) {
            const int64_t pageOffset = indexGm_.GetValue(tokenStart + t);
            // index 的合法值域是 [-1, blockNum_*blockSize_);越界即散射写越过 cache 张量。
            // index 是设备张量,取值 host 读不到、无法在 tiling 阶段拦截,只能在这里判上界。
            // 实测越界量小时是静默踩内存(写进设备内存池里别的缓冲),大到出池则直接 VEC_ERROR。
            if (pageOffset < 0 || pageOffset >= blockNum_ * blockSize_) {
                continue; // index = -1(跳过)或越界(拒绝写)
            }
            const int64_t pageId = pageOffset / blockSize_;
            const int64_t tokenInPage = pageOffset - pageId * blockSize_;
            for (int64_t h = 0; h < numHeadK_; ++h) {
                const LocalTensor<T_K_CACHE> row = kOut[(t * numHeadK_ + h) * qkvDim_];
                ScatterNz<T_K_CACHE>(kCacheGm_, row, pageId, tokenInPage, h, numHeadK_);
                if (!kCacheIsAliased_) {
                    ScatterNz<T_K_CACHE>(kCacheOutGm_, row, pageId, tokenInPage, h, numHeadK_);
                }
            }
            for (int64_t h = 0; h < numHeadV_; ++h) {
                const LocalTensor<T_V_CACHE> row = vOut[(t * numHeadV_ + h) * qkvDim_];
                ScatterNz<T_V_CACHE>(vCacheGm_, row, pageId, tokenInPage, h, numHeadV_);
                if (!vCacheIsAliased_) {
                    ScatterNz<T_V_CACHE>(vCacheOutGm_, row, pageId, tokenInPage, h, numHeadV_);
                }
            }
        }
        outQueueK_.FreeTensor(kOut);
        outQueueV_.FreeTensor(vOut);
    }

    // PA_NZ 散射:cache 排布 [BlockNum, N*D1, BlockSize, D0],D0 = 32B/sizeof(T),D1 = D/D0
    template <typename T_CACHE>
    __aicore__ inline void ScatterNz(const GlobalTensor<T_CACHE> &cacheGm, const LocalTensor<T_CACHE> &srcRow,
                                     int64_t pageId, int64_t tokenInPage, int64_t headIdx, int64_t headNum)
    {
        constexpr int64_t D0 = REGBASE_UB_BLOCK / static_cast<int64_t>(sizeof(T_CACHE));
        const int64_t d1 = qkvDim_ / D0;
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(d1);
        copyParams.blockLen = static_cast<uint32_t>(D0 * sizeof(T_CACHE));
        copyParams.srcStride = 0;
        copyParams.dstStride = static_cast<uint32_t>((blockSize_ - 1) * D0 * sizeof(T_CACHE));
        copyParams.rsv = 0;
        const int64_t gmOffset =
            pageId * headNum * d1 * blockSize_ * D0 + headIdx * d1 * blockSize_ * D0 + tokenInPage * D0;
        // index 是 GM 标量读,结果参与 MTE3 目的地址 -> 需要 S 等 MTE3 的同步
        SetWaitFlag<HardEvent::S_MTE3>();
        DataCopyPad(cacheGm[gmOffset], srcRow, copyParams);
    }

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

private:
    TPipe *pipe_ = nullptr;
    const QkvRmsNormRopeCacheRegbaseTilingData *tiling_ = nullptr;

    GlobalTensor<T_QKV> qkvGm_, gammaQGm_, gammaKGm_, cosGm_, sinGm_, qOutGm_;
    GlobalTensor<T_QKV> qOutProtoGm_, kOutProtoGm_, vOutProtoGm_;
    GlobalTensor<T_QKV> qOutOutGm_;
    GlobalTensor<T_K_CACHE> kCacheGm_, kCacheOutGm_;
    GlobalTensor<T_V_CACHE> vCacheGm_, vCacheOutGm_;
    bool qOutIsAliased_{true}, kCacheIsAliased_{true}, vCacheIsAliased_{true};
    GlobalTensor<int64_t> indexGm_;
    GlobalTensor<float> kScaleGm_, vScaleGm_, kOffsetGm_, vOffsetGm_;

    TQue<QuePosition::VECIN, 1> inQueueX_, inQueueCos_, inQueueSin_;
    TQue<QuePosition::VECOUT, 1> outQueueQ_, outQueueKProto_, outQueueVProto_, outQueueK_, outQueueV_;
    TBuf<TPosition::VECCALC> gammaBuf_, quantBuf_;

    int64_t numHead_{0}, qkvDim_{0}, numHeadQ_{0}, numHeadK_{0}, numHeadV_{0};
    int64_t blockSize_{0}, blockNum_{0}, tokenStride_{0};
    int64_t blockFactor_{1}, ubFactor_{1};
    int64_t inUbBytes_{0}, cosSinUbBytes_{0}, qOutUbBytes_{0}, kProtoUbBytes_{0}, vProtoUbBytes_{0};
    int64_t kCacheUbBytes_{0}, vCacheUbBytes_{0}, gammaUbBytes_{0}, quantUbBytes_{0};
    int64_t tokenSum_{0}, coreStart_{0}, coreEnd_{0};
    float epsilon_{1e-6f}, reciprocal_{1.0f};
    int64_t isOutputQkv_{0};
    bool hasOffsetK_{false}, hasOffsetV_{false};
};

} // namespace QkvRmsNormRopeCache

#endif // QKV_RMS_NORM_ROPE_CACHE_REGBASE_H
