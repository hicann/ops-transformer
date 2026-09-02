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
 * \file kv_rms_norm_rope_cache_regbase_recompute.h
 * \brief
 */

#ifndef KV_RMS_NORM_ROPE_CACHE_REGBASE_RECOMPUTE_H_
#define KV_RMS_NORM_ROPE_CACHE_REGBASE_RECOMPUTE_H_
#include "kv_rms_norm_rope_cache_regbase_base.h"

namespace KvRmsNormRopeCache {
static constexpr int64_t RECOMPUTE_BINARY_CACHE_BTYES = 2048;
static constexpr int64_t RECOMPUTE_REDUCE_SUM_BUFFER_BTYES = 32;
static constexpr int64_t A_IN_IN = 1;
static constexpr float FLOAT_ZERO = 0.0;

using namespace AscendC;
template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
class KvRmsNormRopeCacheRegbaseRecompute : public KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE> {
public:
    __aicore__ inline KvRmsNormRopeCacheRegbaseRecompute(TPipe *pipe,
                                                         const KvRmsNormRopeCacheRegbaseRecomputeTilingData *tiling)
    {
        tilingData_ = tiling;
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache,
                                GM_ADDR v_cache, GM_ADDR k_scale, GM_ADDR v_scale, GM_ADDR k_offset, GM_ADDR v_offset,
                                GM_ADDR k_out, GM_ADDR v_out)
    {
        bs = tilingData_->bs;
        blockFactor = tilingData_->blockFactor; // 每个核需要处理的行数
        if (this->blockIdx == (this->useCoreNum - 1)) {
            blockFactor = bs - this->blockIdx * tilingData_->blockFactor; // 尾核需要处理的行数
        }
        this->reciprocal = tilingData_->reciprocal;
        this->epsilon = tilingData_->epsilon;
        this->dv = tilingData_->dv;
        this->dk = tilingData_->dk;

        this->ubFactor = tilingData_->ubFactor; // num of cols which needs to compute: split the R-axis for recompute
        ubFactorDvLoopCountCeil = tilingData_->ubFactorDvLoopCountCeil;
        ubFactorDvTail = tilingData_->ubFactorDvTail;
        ubFactorDkLoopCountCeil = tilingData_->ubFactorDkLoopCountCeil;
        ubFactorDkTail = tilingData_->ubFactorDkTail;

        // 数据搬运的参数
        xDataCopyParams.blockCount = 1;
        xDataCopyParams.blockLen = this->ubFactor * sizeof(T_KV);
        xDataCopyParams.srcStride = 0;
        xDataCopyParams.dstStride = 0;

        // 二分累加循环次数
        basicBlockLoop = tilingData_->basicBlockLoop;
        mainFoldCount = tilingData_->mainFoldCount;

        if (basicBlockLoop == 0) {
            resultCacheID_ = 0;
        } else {
            resultCacheID_ = GetCacheId(basicBlockLoop - 1);
        }

        // NZ参数
        dk0 = BLOCK_SIZE / sizeof(T_K_CACHE);
        dk1 = this->dk / dk0;
        dv0 = BLOCK_SIZE / sizeof(T_V_CACHE);
        dv1 = this->dv / dv0;

        // init global memory
        this->kvGm.SetGlobalBuffer((__gm__ T_KV *)kv);
        this->gammaGm.SetGlobalBuffer((__gm__ T_KV *)gamma);
        this->cosGm.SetGlobalBuffer((__gm__ T_KV *)cos);
        this->sinGm.SetGlobalBuffer((__gm__ T_KV *)sin);
        this->indexGm.SetGlobalBuffer((__gm__ int64_t *)index);
        this->kCacheGm.SetGlobalBuffer((__gm__ T_K_CACHE *)k_cache);
        this->vCacheGm.SetGlobalBuffer((__gm__ T_V_CACHE *)v_cache);
        this->kScaleGm.SetGlobalBuffer((__gm__ float *)k_scale);
        this->vScaleGm.SetGlobalBuffer((__gm__ float *)v_scale);
        this->kOffsetGm.SetGlobalBuffer((__gm__ float *)k_offset);
        this->vOffsetGm.SetGlobalBuffer((__gm__ float *)v_offset);
        this->kOutGm.SetGlobalBuffer((__gm__ T_KV *)k_out + this->blockIdx * tilingData_->blockFactor * this->dk);
        this->vOutGm.SetGlobalBuffer((__gm__ T_KV *)v_out + this->blockIdx * tilingData_->blockFactor * this->dv);

        // init pipe
        pipe_->InitBuffer(cacheBuffer, RECOMPUTE_BINARY_CACHE_BTYES);
        pipe_->InitBuffer(tmpReduceBuffer, RECOMPUTE_REDUCE_SUM_BUFFER_BTYES);
        pipe_->InitBuffer(inQueueGamma, BUFFER_COUNT_DOUBLE, this->ubFactor * sizeof(T_KV));
        pipe_->InitBuffer(inQueueCosSin, BUFFER_COUNT_DOUBLE, BUFFER_COUNT_DOUBLE * this->ubFactor * sizeof(T_KV));
        pipe_->InitBuffer(inQueueX, BUFFER_COUNT_DOUBLE, this->ubFactor * sizeof(T_KV));
        pipe_->InitBuffer(outQueue, BUFFER_COUNT_DOUBLE, BUFFER_COUNT_DOUBLE * this->ubFactor * sizeof(T_KV));
        // xPowBuffer 按 float 存放二分折叠的两段中间结果（x1、x2），每段 ubFactor 个 float
        pipe_->InitBuffer(xPowBuffer, BUFFER_COUNT_DOUBLE * this->ubFactor * sizeof(float));

        if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                      IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
            // scaleoffset需要放scale和offset，所以需要2 * 1 * ubFactor * sizeof(float)
            pipe_->InitBuffer(kScaleOffsetQueue, BUFFER_COUNT_SINGLE, 2 * this->ubFactor * sizeof(float));
        }
        if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                      IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
            pipe_->InitBuffer(vScaleOffsetQueue, BUFFER_COUNT_SINGLE, 2 * this->ubFactor * sizeof(float));
        }
    }

    __aicore__ inline void CastPowVF(__ubuf__ float *&xFp32Ptr, __ubuf__ T_KV *&xPtr, uint32_t ubFactor)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vreg1, vreg2;
            AscendC::MicroAPI::MaskReg mask;

            uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, VL_FP32);
            uint16_t regTailWidth = ubFactor - repeatTimesFloor * VL_FP32;

            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto xAddr = xPtr + j * VL_FP32;
                auto xFp32Addr = xFp32Ptr + j * VL_FP32;
                LoadDataAndCast2Fp32(vreg1, xAddr, mask);
                AscendC::MicroAPI::Mul(vreg2, vreg1, vreg1, mask);
                AscendC::MicroAPI::StoreAlign(xFp32Addr, vreg2, mask);
            }

            // 尾VL计算
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
            auto xFp32Addr = xFp32Ptr + repeatTimesFloor * VL_FP32;
            LoadDataAndCast2Fp32(vreg1, xAddr, mask);
            AscendC::MicroAPI::Mul(vreg2, vreg1, vreg1, mask);
            AscendC::MicroAPI::StoreAlign(xFp32Addr, vreg2, mask);
        }
    }

    __aicore__ inline void FoldBlockVF(__ubuf__ float *&x1Ptr, __ubuf__ float *&x2Ptr, uint32_t ubFactor)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vreg1, vreg2, vreg3;
            AscendC::MicroAPI::MaskReg mask;

            uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, VL_FP32);
            uint16_t regTailWidth = ubFactor - repeatTimesFloor * VL_FP32;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto x1Addr = x1Ptr + j * VL_FP32;
                auto x2Addr = x2Ptr + j * VL_FP32;
                LoadDataAndCast2Fp32<float>(vreg1, x1Addr, mask);
                LoadDataAndCast2Fp32<float>(vreg2, x2Addr, mask);
                AscendC::MicroAPI::Add(vreg3, vreg1, vreg2, mask);
                AscendC::MicroAPI::StoreAlign(x1Addr, vreg3, mask);
            }

            // 尾VL计算
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            auto x1Addr = x1Ptr + repeatTimesFloor * VL_FP32;
            auto x2Addr = x2Ptr + repeatTimesFloor * VL_FP32;
            LoadDataAndCast2Fp32<float>(vreg1, x1Addr, mask);
            LoadDataAndCast2Fp32<float>(vreg2, x2Addr, mask);
            AscendC::MicroAPI::Add(vreg3, vreg1, vreg2, mask);
            AscendC::MicroAPI::StoreAlign(x1Addr, vreg3, mask);
        }
    }

    __aicore__ inline int64_t GetCacheId(const uint64_t idx)
    {
        return AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + CONST_ONE)) - CONST_ONE;
    }

    __aicore__ inline void UpdateCache(const LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor,
                                       const int64_t cacheId, const int64_t stride, const int64_t count)
    {
        uint16_t outerLoopTimes =
            ops::CeilDiv(static_cast<int64_t>(count * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
        uint16_t innerLoopTimes = cacheId;
        uint32_t outerLoopStride = VL_FP32;
        uint32_t innerLoopStride = stride;

        __ubuf__ float *dst = (__ubuf__ float *)dstTensor.GetPhyAddr();
        __ubuf__ float *cache = (__ubuf__ float *)dstTensor.GetPhyAddr() + cacheId * stride;
        __ubuf__ float *src = (__ubuf__ float *)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                AscendC::MicroAPI::LoadAlign(aReg, (__ubuf__ float *)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    AscendC::MicroAPI::LoadAlign(bReg,
                                                 (__ubuf__ float *)dst + i * outerLoopStride + j * innerLoopStride);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                AscendC::MicroAPI::StoreAlign((__ubuf__ float *)cache + i * outerLoopStride, aReg, pMask);
            }
        }
    }

    __aicore__ inline void CalculateVOutVF(__ubuf__ T_KV *&yPtr, __ubuf__ T_KV *&xPtr, __ubuf__ T_KV *&gammaPtr,
                                           __ubuf__ float *&xSumPtr, uint32_t ubFactor)
    {
        float reciprocal = this->reciprocal;
        float epsilon = this->epsilon;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
            AscendC::MicroAPI::RegTensor<T_KV> vregY;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;

            uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, VL_FP32);
            uint16_t regTailWidth = ubFactor - repeatTimesFloor * VL_FP32;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
            AscendC::MicroAPI::Muls(sumReg, sumReg, reciprocal, mask);
            AscendC::MicroAPI::Adds(sumReg, sumReg, epsilon, mask);
            AscendC::MicroAPI::Sqrt(sumReg, sumReg, mask);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto xAddr = xPtr + j * VL_FP32;
                auto yAddr = yPtr + j * VL_FP32;
                auto gammaAddr = gammaPtr + j * VL_FP32;
                LoadDataAndCast2Fp32(vregX, xAddr, mask);
                LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
                AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);
                StoreDataAndCastFromFp32(yAddr, vregTmp, uReg, mask, width);
            }

            // 尾VL
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
            auto yAddr = yPtr + repeatTimesFloor * VL_FP32;
            auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
            LoadDataAndCast2Fp32(vregX, xAddr, mask);
            LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
            AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);
            StoreDataAndCastFromFp32(yAddr, vregTmp, uReg, mask, width);
        }
    }

    template <typename T = T_KV>
    __aicore__ inline void LoadDataAndCast2Fp32(AscendC::MicroAPI::RegTensor<float> &dst, __ubuf__ T *xAddr,
                                                AscendC::MicroAPI::MaskReg &mask)
    {
        if constexpr (!IsSameType<T, float>::value) {
            AscendC::MicroAPI::RegTensor<T> vregB16;
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregB16, xAddr);
            AscendC::MicroAPI::Cast<float, T, CAST_B16_TO_B32>(dst, vregB16, mask);
        } else {
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(dst, xAddr);
        }
    }

    template <typename T>
    __aicore__ inline void StoreQuantAndCastFromFp32(__ubuf__ T *&dst, AscendC::MicroAPI::RegTensor<float> &src,
                                                     AscendC::MicroAPI::MaskReg &mask)
    {
        AscendC::MicroAPI::RegTensor<T> vregQuant;
        if constexpr (IsSameType<T, int8_t>::value) {
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(vregInt16, src, mask);
            AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(vregHalf, vregInt16, mask);
            AscendC::MicroAPI::Cast<T, half, CAST_FP16_TO_INT8>(vregQuant, vregHalf, mask);
        } else if constexpr (IsSameType<T, hifloat8_t>::value) {
            AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_HIFLOAT8>(vregQuant, src, mask);
        } else if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value) {
            AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FLOAT8>(vregQuant, src, mask);
        }
        AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst, vregQuant, mask);
    }

    template <typename T = T_KV>
    __aicore__ inline void StoreDataAndCastFromFp32(__ubuf__ T *&dst, AscendC::MicroAPI::RegTensor<float> &src,
                                                    AscendC::MicroAPI::UnalignRegForStore &uReg,
                                                    AscendC::MicroAPI::MaskReg &mask, uint32_t postUpdateStride)
    {
        if constexpr (!IsSameType<T, float>::value) {
            AscendC::MicroAPI::RegTensor<T> vregB16, vregB16Pack;
            AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16>(vregB16, src, mask);
            AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst, vregB16, mask);
        } else {
            AscendC::MicroAPI::StoreAlign(dst, src, mask);
        }
    }

    __aicore__ inline void ReduceSumBasicComputeVF(int64_t xDimOffset)
    {
        // RmsNorm: step 1. Σ x^2
        LocalTensor<float> xPowLocal1 = xPowBuffer.Get<float>();
        LocalTensor<float> xPowLocal2 = xPowLocal1[this->ubFactor];
        tmpReduceLocal = tmpReduceBuffer.Get<float>();
        LocalTensor<float> cacheLocal = cacheBuffer.Get<float>();
        __ubuf__ float *x1Fp32Ptr = (__ubuf__ float *)xPowLocal1.GetPhyAddr();
        __ubuf__ float *x2Fp32Ptr = (__ubuf__ float *)xPowLocal2.GetPhyAddr();

        // dv 不超过一个 ubFactor 时无需二分折叠，basicBlockLoop 为 0，单块直接归约
        if (basicBlockLoop == 0) {
            uint32_t singleBlockFactor =
                ubFactorDvTail > 0 ? static_cast<uint32_t>(ubFactorDvTail) : static_cast<uint32_t>(this->ubFactor);
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            __ubuf__ T_KV *xPtr = (__ubuf__ T_KV *)xLocal.GetPhyAddr();

            xDataCopyParams.blockLen = singleBlockFactor * sizeof(T_KV);
            AscendC::DataCopyPad(xLocal, this->kvGm[xDimOffset], xDataCopyParams, padParams);
            inQueueX.EnQue<T_KV>(xLocal);
            xLocal = inQueueX.DeQue<T_KV>();

            CastPowVF(x1Fp32Ptr, xPtr, singleBlockFactor);
            inQueueX.FreeTensor(xLocal);

            AscendC::Duplicate(tmpReduceLocal, FLOAT_ZERO, BLOCK_SIZE / sizeof(float));
            uint32_t srcShape[2] = {A_IN_IN, singleBlockFactor};
            AscendC::ReduceSum<float, Pattern::Reduce::AR, true>(tmpReduceLocal, xPowLocal1, srcShape, false);
            UpdateCache(cacheLocal, tmpReduceLocal, 0, A_IN_IN * BLOCK_SIZE / sizeof(float), A_IN_IN);
            totalSumLocal = cacheLocal[resultCacheID_ * BLOCK_SIZE / sizeof(float)];
            return;
        }

        for (uint64_t basicBlockIdx = 0; basicBlockIdx < basicBlockLoop; basicBlockIdx++) {
            LocalTensor<T_KV> x1Local = inQueueX.AllocTensor<T_KV>();
            __ubuf__ T_KV *x1Ptr = (__ubuf__ T_KV *)x1Local.GetPhyAddr();
            int64_t xUbOffset1 = xDimOffset + this->ubFactor * basicBlockIdx;                    // 主块
            int64_t xUbOffset2 = xDimOffset + this->ubFactor * (basicBlockLoop + basicBlockIdx); // 被折叠块

            // Get the fold block x1
            xDataCopyParams.blockLen = this->ubFactor * sizeof(T_KV);
            AscendC::DataCopyPad(x1Local, this->kvGm[xUbOffset1], xDataCopyParams, padParams);
            inQueueX.EnQue<T_KV>(x1Local);
            x1Local = inQueueX.DeQue<T_KV>();

            CastPowVF(x1Fp32Ptr, x1Ptr, this->ubFactor);
            inQueueX.FreeTensor(x1Local);

            // Get the fold block x2
            LocalTensor<T_KV> x2Local = inQueueX.AllocTensor<T_KV>();
            __ubuf__ T_KV *x2Ptr = (__ubuf__ T_KV *)x2Local.GetPhyAddr();
            if (basicBlockIdx < this->mainFoldCount) {
                xDataCopyParams.blockLen = this->ubFactor * sizeof(T_KV);
                AscendC::DataCopyPad(x2Local, this->kvGm[xUbOffset2], xDataCopyParams, padParams);
                inQueueX.EnQue<T_KV>(x2Local);
                x2Local = inQueueX.DeQue<T_KV>();
                CastPowVF(x2Fp32Ptr, x2Ptr, this->ubFactor);
                FoldBlockVF(x1Fp32Ptr, x2Fp32Ptr, this->ubFactor);
            } else if ((basicBlockIdx == mainFoldCount) && (ubFactorDvTail > 0)) {
                xDataCopyParams.blockLen = ubFactorDvTail * sizeof(T_KV); // x2 is the tail of ub block here.
                AscendC::DataCopyPad(x2Local, this->kvGm[xUbOffset2], xDataCopyParams, padParams);
                inQueueX.EnQue<T_KV>(x2Local);
                x2Local = inQueueX.DeQue<T_KV>();
                CastPowVF(x2Fp32Ptr, x2Ptr, ubFactorDvTail);
                FoldBlockVF(x1Fp32Ptr, x2Fp32Ptr, ubFactorDvTail);
            }
            inQueueX.FreeTensor(x2Local);
            // add the fold x2 to the fold x1, then do reducesum for each ub by UpdateCache
            AscendC::Duplicate(tmpReduceLocal, FLOAT_ZERO, BLOCK_SIZE / sizeof(float));
            int64_t cacheId = GetCacheId(basicBlockIdx);
            uint32_t srcShape[2] = {A_IN_IN, static_cast<uint32_t>(this->ubFactor)};
            AscendC::ReduceSum<float, Pattern::Reduce::AR, true>(tmpReduceLocal, xPowLocal1, srcShape, false);
            UpdateCache(cacheLocal, tmpReduceLocal, cacheId, A_IN_IN * BLOCK_SIZE / sizeof(float), A_IN_IN);
            totalSumLocal = cacheLocal[resultCacheID_ * BLOCK_SIZE / sizeof(float)];
        }
    }

    // 不需要量化的情况
    __aicore__ inline void RmsNormWithoutQuant(int64_t xDimOffset, int64_t rowIdx, int64_t vCacheRowOffset)
    {
        // RmsNorm: step 1. Σ x^2
        ReduceSumBasicComputeVF(xDimOffset);
        // RmsNorm: step 2. x / (sqrt(1 / dv_ * (Σ x^2) + epsilon)) * gamma
        __ubuf__ float *xSumPtr = (__ubuf__ float *)totalSumLocal.GetPhyAddr();

        for (uint16_t ubIdx = 0; ubIdx < ubFactorDvLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> gammaLocal = inQueueGamma.AllocTensor<T_KV>();
            vOutLocal = outQueue.AllocTensor<T_KV>();
            __ubuf__ T_KV *xPtr = (__ubuf__ T_KV *)xLocal.GetPhyAddr();
            __ubuf__ T_KV *gammaPtr = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
            __ubuf__ T_KV *vPtr = (__ubuf__ T_KV *)vOutLocal.GetPhyAddr();

            int64_t xUbOffset = xDimOffset + this->ubFactor * ubIdx;
            int64_t gammaUbOffset = this->ubFactor * ubIdx;
            int64_t vOutOffset = rowIdx * this->dv + this->ubFactor * ubIdx;

            uint32_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDvLoopCountCeil - 1 && ubFactorDvTail > 0) {
                tmpFactor = ubFactorDvTail;
            }
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);

            AscendC::DataCopyPad(xLocal, this->kvGm[xUbOffset], xDataCopyParams, padParams);
            AscendC::DataCopyPad(gammaLocal, this->gammaGm[gammaUbOffset], xDataCopyParams, padParams);

            inQueueX.EnQue<T_KV>(xLocal);
            inQueueGamma.EnQue<T_KV>(gammaLocal);
            xLocal = inQueueX.DeQue<T_KV>();
            gammaLocal = inQueueGamma.DeQue<T_KV>();

            CalculateVOutVF(vPtr, xPtr, gammaPtr, xSumPtr, tmpFactor);

            outQueue.EnQue<T_KV>(vOutLocal);
            vOutLocal = outQueue.DeQue<T_KV>();

            if (tilingData_->isOutputKv) {
                AscendC::DataCopyPad(this->vOutGm[vOutOffset], vOutLocal, xDataCopyParams);
            }

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            } else {
                ScatterBlkUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            }

            inQueueX.FreeTensor(xLocal);
            inQueueGamma.FreeTensor(gammaLocal);
            outQueue.FreeTensor(vOutLocal);
        }
    }

    // 需要中间结果+量化+偏移
    __aicore__ inline void CalculateVOutAsymQuantWithKvVF(__ubuf__ T_KV *&outPtr, __ubuf__ T_KV *&xPtr,
                                                          __ubuf__ T_KV *&gammaPtr, __ubuf__ float *&xSumPtr,
                                                          __ubuf__ float *&vScalePtr, __ubuf__ float *&vOffsetPtr,
                                                          __ubuf__ T_V_CACHE *&vQuantPtr, uint32_t ubFactor)
    {
        float reciprocal = this->reciprocal;
        float epsilon = this->epsilon;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
            AscendC::MicroAPI::RegTensor<float> vregScale, vregOffset;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> vregQuant;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;

            uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, VL_FP32);
            uint16_t regTailWidth = ubFactor - repeatTimesFloor * VL_FP32;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
            AscendC::MicroAPI::Muls(sumReg, sumReg, reciprocal, mask);
            AscendC::MicroAPI::Adds(sumReg, sumReg, epsilon, mask);
            AscendC::MicroAPI::Sqrt(sumReg, sumReg, mask);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto xAddr = xPtr + j * VL_FP32;
                auto gammaAddr = gammaPtr + j * VL_FP32;
                auto vScaleAddr = vScalePtr + j * VL_FP32;
                auto vOffsetAddr = vOffsetPtr + j * VL_FP32;
                auto outAddr = outPtr + j * VL_FP32;
                auto vQuantAddr = vQuantPtr + j * VL_FP32;
                LoadDataAndCast2Fp32(vregX, xAddr, mask);
                LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
                AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);
                StoreDataAndCastFromFp32(outAddr, vregTmp, uReg, mask, width);

                LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
                LoadDataAndCast2Fp32<float>(vregOffset, vOffsetAddr, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregScale, mask);
                AscendC::MicroAPI::Add(vregTmp, vregTmp, vregOffset, mask);
                StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
            }

            // 尾VL
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
            auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
            auto vScaleAddr = vScalePtr + repeatTimesFloor * VL_FP32;
            auto vOffsetAddr = vOffsetPtr + repeatTimesFloor * VL_FP32;
            auto outAddr = outPtr + repeatTimesFloor * VL_FP32;
            auto vQuantAddr = vQuantPtr + repeatTimesFloor * VL_FP32;
            LoadDataAndCast2Fp32(vregX, xAddr, mask);
            LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
            AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);
            StoreDataAndCastFromFp32(outAddr, vregTmp, uReg, mask, width);

            LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
            LoadDataAndCast2Fp32<float>(vregOffset, vOffsetAddr, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregScale, mask);
            AscendC::MicroAPI::Add(vregTmp, vregTmp, vregOffset, mask);
            StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
        }
    }

    __aicore__ inline void RmsNormAsymQuantWithVCache(int64_t xDimOffset, int64_t rowIdx, int64_t vCacheRowOffset)
    {
        // RmsNorm: step 1. Σ x^2
        ReduceSumBasicComputeVF(xDimOffset);
        // RmsNorm: step 2. x / (sqrt(1 / dv_ * (Σ x^2) + epsilon)) * gamma
        __ubuf__ float *xSumPtr = (__ubuf__ float *)totalSumLocal.GetPhyAddr();

        for (uint16_t ubIdx = 0; ubIdx < ubFactorDvLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> gammaLocal = inQueueGamma.AllocTensor<T_KV>();
            LocalTensor<float> vScaleLocal = vScaleOffsetQueue.AllocTensor<float>();
            LocalTensor<float> vOffsetLocal = vScaleLocal[this->ubFactor];
            // 量化场景
            vOutLocal = outQueue.AllocTensor<T_KV>();
            vQuantLocal =
                vOutLocal.template ReinterpretCast<T_V_CACHE>()[this->ubFactor * sizeof(T_KV) / sizeof(T_V_CACHE)];

            __ubuf__ T_KV *xPtr = (__ubuf__ T_KV *)xLocal.GetPhyAddr();
            __ubuf__ T_KV *gammaPtr = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
            __ubuf__ T_KV *vPtr = (__ubuf__ T_KV *)vOutLocal.GetPhyAddr();
            __ubuf__ T_V_CACHE *vQuantPtr = (__ubuf__ T_V_CACHE *)vQuantLocal.GetPhyAddr();
            __ubuf__ float *vScalePtr = (__ubuf__ float *)vScaleLocal.GetPhyAddr();
            __ubuf__ float *vOffsetPtr = (__ubuf__ float *)vOffsetLocal.GetPhyAddr();

            int64_t xUbOffset = xDimOffset + this->ubFactor * ubIdx;
            int64_t gammaUbOffset = this->ubFactor * ubIdx;
            int64_t vScaleUbOffset = this->ubFactor * ubIdx;
            int64_t vOffsetUbOffset = this->ubFactor * ubIdx;
            int64_t vOutOffset = rowIdx * this->dv + this->ubFactor * ubIdx;

            uint32_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDvLoopCountCeil - 1 && ubFactorDvTail > 0) {
                tmpFactor = ubFactorDvTail;
            }
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);

            AscendC::DataCopyPad(xLocal, this->kvGm[xUbOffset], xDataCopyParams, padParams);
            AscendC::DataCopyPad(gammaLocal, this->gammaGm[gammaUbOffset], xDataCopyParams, padParams);

            if (tilingData_->vScaleType > 1) {
                xDataCopyParams.blockLen = tmpFactor * sizeof(float);
                AscendC::DataCopyPad(vScaleLocal, this->vScaleGm[vScaleUbOffset], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(vScaleLocal, this->vScaleGm, xDataCopyParams, padParams);
            }

            if (tilingData_->vOffsetType > 1) {
                xDataCopyParams.blockLen = tmpFactor * sizeof(float);
                AscendC::DataCopyPad(vOffsetLocal, this->vOffsetGm[vOffsetUbOffset], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(vOffsetLocal, this->vOffsetGm, xDataCopyParams, padParams);
            }

            inQueueX.EnQue<T_KV>(xLocal);
            inQueueGamma.EnQue<T_KV>(gammaLocal);
            xLocal = inQueueX.DeQue<T_KV>();
            gammaLocal = inQueueGamma.DeQue<T_KV>();
            vScaleOffsetQueue.EnQue<float>(vScaleLocal);
            vScaleLocal = vScaleOffsetQueue.DeQue<float>();

            // 将scale和offset Brc方便VF处理
            if (tilingData_->vScaleType == 1) {
                AscendC::Duplicate<float>(vScaleLocal, vScaleLocal, this->ubFactor);
            }

            if (tilingData_->vOffsetType == 1) {
                AscendC::Duplicate<float>(vOffsetLocal, vOffsetLocal, this->ubFactor);
            }

            CalculateVOutAsymQuantWithKvVF(vPtr, xPtr, gammaPtr, xSumPtr, vScalePtr, vOffsetPtr, vQuantPtr, tmpFactor);

            outQueue.EnQue<T_KV>(vOutLocal);
            vOutLocal = outQueue.DeQue<T_KV>();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
            AscendC::DataCopyPad(this->vOutGm[vOutOffset], vOutLocal, xDataCopyParams);

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            } else {
                ScatterBlkUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            }

            inQueueX.FreeTensor(xLocal);
            inQueueGamma.FreeTensor(gammaLocal);
            outQueue.FreeTensor(vOutLocal);
            vScaleOffsetQueue.FreeTensor(vScaleLocal);
        }
    }

    // 需要中间结果 + 量化
    __aicore__ inline void CalculateVOutSymQuantWithKvVF(__ubuf__ T_KV *&outPtr, __ubuf__ T_KV *&xPtr,
                                                         __ubuf__ T_KV *&gammaPtr, __ubuf__ float *&xSumPtr,
                                                         __ubuf__ float *&vScalePtr, __ubuf__ T_V_CACHE *&vQuantPtr,
                                                         uint32_t ubFactor)
    {
        float reciprocal = this->reciprocal;
        float epsilon = this->epsilon;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
            AscendC::MicroAPI::RegTensor<float> vregScale;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> vregQuant;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;

            uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, VL_FP32);
            uint16_t regTailWidth = ubFactor - repeatTimesFloor * VL_FP32;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
            AscendC::MicroAPI::Muls(sumReg, sumReg, reciprocal, mask);
            AscendC::MicroAPI::Adds(sumReg, sumReg, epsilon, mask);
            AscendC::MicroAPI::Sqrt(sumReg, sumReg, mask);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto xAddr = xPtr + j * VL_FP32;
                auto gammaAddr = gammaPtr + j * VL_FP32;
                auto vScaleAddr = vScalePtr + j * VL_FP32;
                auto outAddr = outPtr + j * VL_FP32;
                auto vQuantAddr = vQuantPtr + j * VL_FP32;
                LoadDataAndCast2Fp32(vregX, xAddr, mask);
                LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
                AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);
                StoreDataAndCastFromFp32(outAddr, vregTmp, uReg, mask, width);

                LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregScale, mask);
                StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
            }

            // 尾VL
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
            auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
            auto vScaleAddr = vScalePtr + repeatTimesFloor * VL_FP32;
            auto outAddr = outPtr + repeatTimesFloor * VL_FP32;
            auto vQuantAddr = vQuantPtr + repeatTimesFloor * VL_FP32;
            LoadDataAndCast2Fp32(vregX, xAddr, mask);
            LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
            AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);
            StoreDataAndCastFromFp32(outAddr, vregTmp, uReg, mask, width);

            LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregScale, mask);
            StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
        }
    }

    __aicore__ inline void RmsNormSymQuantWithVCache(int64_t xDimOffset, int64_t rowIdx, int64_t vCacheRowOffset)
    {
        // RmsNorm: step 1. Σ x^2
        ReduceSumBasicComputeVF(xDimOffset);
        // RmsNorm: step 2. x / (sqrt(1 / dv_ * (Σ x^2) + epsilon)) * gamma
        __ubuf__ float *xSumPtr = (__ubuf__ float *)totalSumLocal.GetPhyAddr();

        for (uint16_t ubIdx = 0; ubIdx < ubFactorDvLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> gammaLocal = inQueueGamma.AllocTensor<T_KV>();
            LocalTensor<float> vScaleLocal = vScaleOffsetQueue.AllocTensor<float>();
            // 量化场景
            vOutLocal = outQueue.AllocTensor<T_KV>();
            vQuantLocal =
                vOutLocal.template ReinterpretCast<T_V_CACHE>()[this->ubFactor * sizeof(T_KV) / sizeof(T_V_CACHE)];

            __ubuf__ T_KV *xPtr = (__ubuf__ T_KV *)xLocal.GetPhyAddr();
            __ubuf__ T_KV *gammaPtr = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
            __ubuf__ T_KV *vPtr = (__ubuf__ T_KV *)vOutLocal.GetPhyAddr();
            __ubuf__ float *vScalePtr = (__ubuf__ float *)vScaleLocal.GetPhyAddr();
            __ubuf__ T_V_CACHE *vQuantPtr = (__ubuf__ T_V_CACHE *)vQuantLocal.GetPhyAddr();

            int64_t xUbOffset = xDimOffset + this->ubFactor * ubIdx;
            int64_t gammaUbOffset = this->ubFactor * ubIdx;
            int64_t vScaleUbOffset = this->ubFactor * ubIdx;
            int64_t vOutOffset = rowIdx * this->dv + this->ubFactor * ubIdx;

            uint32_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDvLoopCountCeil - 1 && ubFactorDvTail > 0) {
                tmpFactor = ubFactorDvTail;
            }
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);

            AscendC::DataCopyPad(xLocal, this->kvGm[xUbOffset], xDataCopyParams, padParams);
            AscendC::DataCopyPad(gammaLocal, this->gammaGm[gammaUbOffset], xDataCopyParams, padParams);

            if (tilingData_->vScaleType > 1) {
                xDataCopyParams.blockLen = tmpFactor * sizeof(float);
                AscendC::DataCopyPad(vScaleLocal, this->vScaleGm[vScaleUbOffset], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(vScaleLocal, this->vScaleGm, xDataCopyParams, padParams);
            }

            inQueueX.EnQue<T_KV>(xLocal);
            inQueueGamma.EnQue<T_KV>(gammaLocal);
            xLocal = inQueueX.DeQue<T_KV>();
            gammaLocal = inQueueGamma.DeQue<T_KV>();
            vScaleOffsetQueue.EnQue<float>(vScaleLocal);
            vScaleLocal = vScaleOffsetQueue.DeQue<float>();

            // 将scale Brc方便VF处理
            if (tilingData_->vScaleType == 1) {
                AscendC::Duplicate<float>(vScaleLocal, vScaleLocal, this->ubFactor);
            }

            CalculateVOutSymQuantWithKvVF(vPtr, xPtr, gammaPtr, xSumPtr, vScalePtr, vQuantPtr, tmpFactor);

            outQueue.EnQue<T_KV>(vOutLocal);
            vOutLocal = outQueue.DeQue<T_KV>();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
            AscendC::DataCopyPad(this->vOutGm[vOutOffset], vOutLocal, xDataCopyParams);

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            } else {
                ScatterBlkUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            }

            inQueueX.FreeTensor(xLocal);
            inQueueGamma.FreeTensor(gammaLocal);
            outQueue.FreeTensor(vOutLocal);
            vScaleOffsetQueue.FreeTensor(vScaleLocal);
        }
    }

    // 不需要中间结果+量化+偏移
    __aicore__ inline void CalculateVOutAsymQuantVF(__ubuf__ T_KV *&xPtr, __ubuf__ T_KV *&gammaPtr,
                                                    __ubuf__ float *&xSumPtr, __ubuf__ float *&vScalePtr,
                                                    __ubuf__ float *&vOffsetPtr, __ubuf__ T_V_CACHE *&vQuantPtr,
                                                    uint32_t ubFactor)
    {
        float reciprocal = this->reciprocal;
        float epsilon = this->epsilon;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
            AscendC::MicroAPI::RegTensor<float> vregScale, vregOffset;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> vregQuant;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;

            uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, VL_FP32);
            uint16_t regTailWidth = ubFactor - repeatTimesFloor * VL_FP32;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
            AscendC::MicroAPI::Muls(sumReg, sumReg, reciprocal, mask);
            AscendC::MicroAPI::Adds(sumReg, sumReg, epsilon, mask);
            AscendC::MicroAPI::Sqrt(sumReg, sumReg, mask);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto xAddr = xPtr + j * VL_FP32;
                auto gammaAddr = gammaPtr + j * VL_FP32;
                auto vScaleAddr = vScalePtr + j * VL_FP32;
                auto vOffsetAddr = vOffsetPtr + j * VL_FP32;
                auto vQuantAddr = vQuantPtr + j * VL_FP32;
                LoadDataAndCast2Fp32(vregX, xAddr, mask);
                LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
                AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);

                LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
                LoadDataAndCast2Fp32<float>(vregOffset, vOffsetAddr, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregScale, mask);
                AscendC::MicroAPI::Add(vregTmp, vregTmp, vregOffset, mask);
                StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
            }

            // 尾VL
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
            auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
            auto vScaleAddr = vScalePtr + repeatTimesFloor * VL_FP32;
            auto vOffsetAddr = vOffsetPtr + repeatTimesFloor * VL_FP32;
            auto vQuantAddr = vQuantPtr + repeatTimesFloor * VL_FP32;
            LoadDataAndCast2Fp32(vregX, xAddr, mask);
            LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
            AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);

            LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
            LoadDataAndCast2Fp32<float>(vregOffset, vOffsetAddr, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregScale, mask);
            AscendC::MicroAPI::Add(vregTmp, vregTmp, vregOffset, mask);
            StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
        }
    }

    __aicore__ inline void RmsNormAsymQuant(int64_t xDimOffset, int64_t vCacheRowOffset)
    {
        // RmsNorm: step 1. Σ x^2
        ReduceSumBasicComputeVF(xDimOffset);
        // RmsNorm: step 2. x / (sqrt(1 / dv_ * (Σ x^2) + epsilon)) * gamma
        __ubuf__ float *xSumPtr = (__ubuf__ float *)totalSumLocal.GetPhyAddr();

        for (uint16_t ubIdx = 0; ubIdx < ubFactorDvLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> gammaLocal = inQueueGamma.AllocTensor<T_KV>();
            LocalTensor<float> vScaleLocal = vScaleOffsetQueue.AllocTensor<float>();
            LocalTensor<float> vOffsetLocal = vScaleLocal[this->ubFactor];
            // 量化场景
            vOutLocal = outQueue.AllocTensor<T_KV>();
            vQuantLocal = vOutLocal.template ReinterpretCast<T_V_CACHE>();

            __ubuf__ T_KV *xPtr = (__ubuf__ T_KV *)xLocal.GetPhyAddr();
            __ubuf__ T_KV *gammaPtr = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
            __ubuf__ float *vScalePtr = (__ubuf__ float *)vScaleLocal.GetPhyAddr();
            __ubuf__ float *vOffsetPtr = (__ubuf__ float *)vOffsetLocal.GetPhyAddr();
            __ubuf__ T_V_CACHE *vQuantPtr = (__ubuf__ T_V_CACHE *)vQuantLocal.GetPhyAddr();

            int64_t xUbOffset = xDimOffset + this->ubFactor * ubIdx;
            int64_t gammaUbOffset = this->ubFactor * ubIdx;
            int64_t vScaleUbOffset = this->ubFactor * ubIdx;
            int64_t vOffsetUbOffset = this->ubFactor * ubIdx;

            uint32_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDvLoopCountCeil - 1 && ubFactorDvTail > 0) {
                tmpFactor = ubFactorDvTail;
            }
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);

            AscendC::DataCopyPad(xLocal, this->kvGm[xUbOffset], xDataCopyParams, padParams);
            AscendC::DataCopyPad(gammaLocal, this->gammaGm[gammaUbOffset], xDataCopyParams, padParams);

            if (tilingData_->vScaleType > 1) {
                xDataCopyParams.blockLen = tmpFactor * sizeof(float);
                AscendC::DataCopyPad(vScaleLocal, this->vScaleGm[vScaleUbOffset], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(vScaleLocal, this->vScaleGm, xDataCopyParams, padParams);
            }

            if (tilingData_->vOffsetType > 1) {
                xDataCopyParams.blockLen = tmpFactor * sizeof(float);
                AscendC::DataCopyPad(vOffsetLocal, this->vOffsetGm[vOffsetUbOffset], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(vOffsetLocal, this->vOffsetGm, xDataCopyParams, padParams);
            }

            inQueueX.EnQue<T_KV>(xLocal);
            xLocal = inQueueX.DeQue<T_KV>();
            inQueueGamma.EnQue<T_KV>(gammaLocal);
            gammaLocal = inQueueGamma.DeQue<T_KV>();
            vScaleOffsetQueue.EnQue<float>(vScaleLocal);
            vScaleLocal = vScaleOffsetQueue.DeQue<float>();

            // 将scale和offset Brc方便VF处理
            if (tilingData_->vScaleType == 1) {
                AscendC::Duplicate<float>(vScaleLocal, vScaleLocal, this->ubFactor);
            }

            if (tilingData_->vOffsetType == 1) {
                AscendC::Duplicate<float>(vOffsetLocal, vOffsetLocal, this->ubFactor);
            }

            CalculateVOutAsymQuantVF(xPtr, gammaPtr, xSumPtr, vScalePtr, vOffsetPtr, vQuantPtr, tmpFactor);

            outQueue.EnQue<T_KV>(vOutLocal);
            vOutLocal = outQueue.DeQue<T_KV>();

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            } else {
                ScatterBlkUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            }

            inQueueX.FreeTensor(xLocal);
            inQueueGamma.FreeTensor(gammaLocal);
            outQueue.FreeTensor(vOutLocal);
            vScaleOffsetQueue.FreeTensor(vScaleLocal);
        }
    }

    // 不需要中间结果 + 量化
    __aicore__ inline void CalculateVOutSymQuantVF(__ubuf__ T_V_CACHE *&vQuantPtr, __ubuf__ T_KV *&xPtr,
                                                   __ubuf__ T_KV *&gammaPtr, __ubuf__ float *&xSumPtr,
                                                   __ubuf__ float *&vScalePtr, uint32_t ubFactor)
    {
        float reciprocal = this->reciprocal;
        float epsilon = this->epsilon;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
            AscendC::MicroAPI::RegTensor<float> vregScale;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<T_V_CACHE> vregQuant;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;

            uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, VL_FP32);
            uint16_t regTailWidth = ubFactor - repeatTimesFloor * VL_FP32;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
            AscendC::MicroAPI::Muls(sumReg, sumReg, reciprocal, mask);
            AscendC::MicroAPI::Adds(sumReg, sumReg, epsilon, mask);
            AscendC::MicroAPI::Sqrt(sumReg, sumReg, mask);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto xAddr = xPtr + j * VL_FP32;
                auto gammaAddr = gammaPtr + j * VL_FP32;
                auto vScaleAddr = vScalePtr + j * VL_FP32;
                auto vQuantAddr = vQuantPtr + j * VL_FP32;
                LoadDataAndCast2Fp32(vregX, xAddr, mask);
                LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
                AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);

                LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
                AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregScale, mask);
                StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
            }

            // 尾VL
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
            auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
            auto vScaleAddr = vScalePtr + repeatTimesFloor * VL_FP32;
            auto vQuantAddr = vQuantPtr + repeatTimesFloor * VL_FP32;
            LoadDataAndCast2Fp32(vregX, xAddr, mask);
            LoadDataAndCast2Fp32(vregGamma, gammaAddr, mask);
            AscendC::MicroAPI::Div(vregTmp, vregX, sumReg, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregGamma, mask);

            LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
            AscendC::MicroAPI::Mul(vregTmp, vregTmp, vregScale, mask);
            StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
        }
    }

    __aicore__ inline void ScatterUpdateV(int64_t rowIdx, int64_t ubIdx, int64_t tmpFactor)
    {
        xDataCopyParams.blockLen = tmpFactor * sizeof(T_V_CACHE);
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t batchIdx = rowIdx / tilingData_->seqLength;
        int64_t seqIdx = rowIdx % tilingData_->seqLength;
        int64_t vCacheIdx = this->indexGm(rowIdx);
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

        int64_t gmOffset;

        // index 的取值是 cache 的行号/槽位号，越界即写到 cache 之外，故必须同时校验上界
        if (vCacheIdx >= 0 && vCacheIdx < tilingData_->cacheRowLimit) {
            if (tilingData_->cacheMode == NORM_CACHE_MODE) {
                gmOffset = (batchIdx * tilingData_->cacheLength + vCacheIdx) * tilingData_->dv + this->ubFactor * ubIdx;
                if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                              IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                    AscendC::DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal, xDataCopyParams);
                } else {
                    xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
                    AscendC::DataCopyPad(this->vCacheGm[gmOffset], vOutLocal, xDataCopyParams);
                }
            } else if (tilingData_->cacheMode == PA_CACHE_MODE) {
                gmOffset = vCacheIdx * tilingData_->dv + this->ubFactor * ubIdx;
                if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                              IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                    AscendC::DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal, xDataCopyParams);
                } else {
                    xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
                    AscendC::DataCopyPad(this->vCacheGm[gmOffset], vOutLocal, xDataCopyParams);
                }
            } else if (tilingData_->cacheMode == PA_NZ_CACHE_MODE) {
                batchIdx = vCacheIdx / tilingData_->blockSize;
                int64_t blockOffset = vCacheIdx % tilingData_->blockSize;
                if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                              IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                    xDataCopyParams.blockLen = dv0 * sizeof(T_V_CACHE);
                } else {
                    xDataCopyParams.blockLen = dv0 * sizeof(T_KV);
                }
                for (int64_t i = 0; i < tmpFactor / dv0; i++) {
                    gmOffset = batchIdx * tilingData_->dv * tilingData_->blockSize +
                               (blockOffset + i * tilingData_->blockSize) * dv0 +
                               this->ubFactor * ubIdx * tilingData_->blockSize;
                    int64_t ubOffset = i * dv0;
                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                                  IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                  IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal[ubOffset], xDataCopyParams);
                    } else {
                        AscendC::DataCopyPad(this->vCacheGm[gmOffset], vOutLocal[ubOffset], xDataCopyParams);
                    }
                }
            } else {
                return;
            }
        }
    }

    __aicore__ inline void ScatterBlkUpdateV(int64_t rowIdx, int64_t ubIdx, int64_t tmpFactor)
    {
        xDataCopyParams.blockLen = tmpFactor * sizeof(T_V_CACHE);
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t ceilValue = ops::CeilDiv(tilingData_->seqLength, tilingData_->blockSize);
        int64_t batchIdx = rowIdx / tilingData_->seqLength;
        int64_t seqIdx = rowIdx % tilingData_->seqLength;
        int64_t blkIdx = seqIdx / tilingData_->blockSize;
        int64_t idx = batchIdx * ceilValue + blkIdx;
        int64_t vCacheIdx = this->indexGm(idx);
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

        int64_t gmOffset;

        // index 的取值是 cache 的行号/槽位号，越界即写到 cache 之外，故必须同时校验上界
        if (vCacheIdx >= 0 && vCacheIdx < tilingData_->cacheRowLimit) {
            if (tilingData_->cacheMode == PA_BLK_BNSD_CACHE_MODE) {
                int64_t blockOffset = vCacheIdx / tilingData_->blockSize;
                int64_t rowOffset = seqIdx % tilingData_->blockSize;
                gmOffset =
                    (blockOffset * tilingData_->blockSize + rowOffset) * tilingData_->dv + ubIdx * this->ubFactor;
                if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                              IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                    AscendC::DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal, xDataCopyParams);
                } else {
                    xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
                    AscendC::DataCopyPad(this->vCacheGm[gmOffset], vOutLocal, xDataCopyParams);
                }
            } else {
                int64_t blockOffset = vCacheIdx / tilingData_->blockSize;
                int64_t rowOffset = seqIdx % tilingData_->blockSize;
                if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                              IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                    xDataCopyParams.blockLen = dv0 * sizeof(T_V_CACHE);
                } else {
                    xDataCopyParams.blockLen = dv0 * sizeof(T_KV);
                }
                for (int i = 0; i < tmpFactor / dv0; i++) {
                    gmOffset = blockOffset * tilingData_->dv * tilingData_->blockSize +
                               (rowOffset + i * tilingData_->blockSize) * dv0 +
                               this->ubFactor * ubIdx * tilingData_->blockSize;
                    int64_t ubOffset = i * dv0;
                    if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                                  IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                  IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal[ubOffset], xDataCopyParams);
                    } else {
                        AscendC::DataCopyPad(this->vCacheGm[gmOffset], vOutLocal[ubOffset], xDataCopyParams);
                    }
                }
            }
        }
    }

    __aicore__ inline void RmsNormSymQuant(int64_t xDimOffset, int64_t vCacheRowOffset)
    {
        // RmsNorm: step 1. Σ x^2
        ReduceSumBasicComputeVF(xDimOffset);
        // RmsNorm: step 2. x / (sqrt(1 / dv_ * (Σ x^2) + epsilon)) * gamma
        __ubuf__ float *xSumPtr = (__ubuf__ float *)totalSumLocal.GetPhyAddr();

        for (uint16_t ubIdx = 0; ubIdx < ubFactorDvLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> gammaLocal = inQueueGamma.AllocTensor<T_KV>();
            LocalTensor<float> vScaleLocal = vScaleOffsetQueue.AllocTensor<float>();
            // 量化场景
            vOutLocal = outQueue.AllocTensor<T_KV>();
            vQuantLocal = vOutLocal.template ReinterpretCast<T_V_CACHE>();

            __ubuf__ T_KV *xPtr = (__ubuf__ T_KV *)xLocal.GetPhyAddr();
            __ubuf__ T_KV *gammaPtr = (__ubuf__ T_KV *)gammaLocal.GetPhyAddr();
            __ubuf__ float *vScalePtr = (__ubuf__ float *)vScaleLocal.GetPhyAddr();
            __ubuf__ T_KV *vPtr = (__ubuf__ T_KV *)vOutLocal.GetPhyAddr();
            __ubuf__ T_V_CACHE *vQuantPtr = (__ubuf__ T_V_CACHE *)vQuantLocal.GetPhyAddr();

            int64_t xUbOffset = xDimOffset + this->ubFactor * ubIdx;
            int64_t gammaUbOffset = this->ubFactor * ubIdx;
            int64_t vScaleUbOffset = this->ubFactor * ubIdx;

            uint32_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDvLoopCountCeil - 1 && ubFactorDvTail > 0) {
                tmpFactor = ubFactorDvTail;
            }
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);

            AscendC::DataCopyPad(xLocal, this->kvGm[xUbOffset], xDataCopyParams, padParams);
            AscendC::DataCopyPad(gammaLocal, this->gammaGm[gammaUbOffset], xDataCopyParams, padParams);

            if (tilingData_->vScaleType > 1) {
                xDataCopyParams.blockLen = tmpFactor * sizeof(float);
                AscendC::DataCopyPad(vScaleLocal, this->vScaleGm[vScaleUbOffset], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(vScaleLocal, this->vScaleGm, xDataCopyParams, padParams);
            }

            inQueueX.EnQue<T_KV>(xLocal);
            xLocal = inQueueX.DeQue<T_KV>();
            inQueueGamma.EnQue<T_KV>(gammaLocal);
            gammaLocal = inQueueGamma.DeQue<T_KV>();
            vScaleOffsetQueue.EnQue<float>(vScaleLocal);
            vScaleLocal = vScaleOffsetQueue.DeQue<float>();

            // 将scale Brc方便VF处理
            if (tilingData_->vScaleType == 1) {
                AscendC::Duplicate<float>(vScaleLocal, vScaleLocal, this->ubFactor);
            }

            CalculateVOutSymQuantVF(vQuantPtr, xPtr, gammaPtr, xSumPtr, vScalePtr, tmpFactor);

            outQueue.EnQue<T_KV>(vOutLocal);
            vOutLocal = outQueue.DeQue<T_KV>();

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            } else {
                ScatterBlkUpdateV(vCacheRowOffset, ubIdx, tmpFactor);
            }

            inQueueX.FreeTensor(xLocal);
            inQueueGamma.FreeTensor(gammaLocal);
            outQueue.FreeTensor(vOutLocal);
            vScaleOffsetQueue.FreeTensor(vScaleLocal);
        }
    }

    __aicore__ inline void RmsNorm(int64_t xDimOffset, int64_t rowIdx, int64_t vCacheRowOffset)
    {
        // 需要量化
        if constexpr (IsSameType<T_V_CACHE, int8_t>::value || IsSameType<T_V_CACHE, hifloat8_t>::value ||
                      IsSameType<T_V_CACHE, fp8_e5m2_t>::value || IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
            // 需要输出中间结果
            if (tilingData_->isOutputKv > 0) {
                // 量化+偏移
                if (tilingData_->vOffsetType > 0) {
                    RmsNormAsymQuantWithVCache(xDimOffset, rowIdx, vCacheRowOffset);
                } else { // 量化
                    RmsNormSymQuantWithVCache(xDimOffset, rowIdx, vCacheRowOffset);
                }
            } else {
                // 不需要输出中间结果
                if (tilingData_->vOffsetType > 0) {
                    // 量化+偏移
                    RmsNormAsymQuant(xDimOffset, vCacheRowOffset);
                } else {
                    // 量化
                    RmsNormSymQuant(xDimOffset, vCacheRowOffset);
                }
            }
        } else {
            // 普通情况， outLocal的类型需要指定
            RmsNormWithoutQuant(xDimOffset, rowIdx, vCacheRowOffset);
        }
    }

    __aicore__ inline void RopeVF(__ubuf__ T_KV *&outPtr1, __ubuf__ T_KV *&outPtr2, __ubuf__ T_KV *&ropePtr,
                                  __ubuf__ T_KV *&cosPtr1, __ubuf__ T_KV *&cosPtr2, __ubuf__ T_KV *&sinPtr1,
                                  __ubuf__ T_KV *&sinPtr2, __ubuf__ float *&tmpBufferPtr, uint32_t ubFactor)
    {
        uint32_t vlStride = VL_FP32 * CONST_TWO;
        uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, vlStride);
        uint16_t regTailWidth = (ubFactor - repeatTimesFloor * vlStride) / CONST_TWO;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32,
                vregSin2Fp32;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
                auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
                auto outAddr1 = outPtr1 + j * VL_FP32;
                auto outAddr2 = outPtr2 + j * VL_FP32;
                auto cosAddr1 = cosPtr1 + j * VL_FP32;
                auto cosAddr2 = cosPtr2 + j * VL_FP32;
                auto sinAddr1 = sinPtr1 + j * VL_FP32;
                auto sinAddr2 = sinPtr2 + j * VL_FP32;

                LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
                LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
                AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

                LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
                LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
                LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
                LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
                AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

                StoreDataAndCastFromFp32(outAddr1, vregCos1Fp32, uReg, mask, width);
                StoreDataAndCastFromFp32(outAddr2, vregCos2Fp32, uReg, mask, width);
            }

            // 尾VL计算范式
            auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
            auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
            auto outAddr1 = outPtr1 + repeatTimesFloor * VL_FP32;
            auto outAddr2 = outPtr2 + repeatTimesFloor * VL_FP32;
            auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
            auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
            auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
            auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;

            LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
            LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

            LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
            LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
            LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
            LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
            AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            StoreDataAndCastFromFp32(outAddr1, vregCos1Fp32, uReg, mask, width);
            StoreDataAndCastFromFp32(outAddr2, vregCos2Fp32, uReg, mask, width);
        }
    }

    __aicore__ inline void RopeSymQuantVF(__ubuf__ T_K_CACHE *&quantPtr1, __ubuf__ T_K_CACHE *&quantPtr2,
                                          __ubuf__ T_KV *&ropePtr, __ubuf__ T_KV *&cosPtr1, __ubuf__ T_KV *&cosPtr2,
                                          __ubuf__ T_KV *&sinPtr1, __ubuf__ T_KV *&sinPtr2, __ubuf__ float *&scalePtr1,
                                          __ubuf__ float *&scalePtr2, __ubuf__ float *&tmpBufferPtr, uint32_t ubFactor)
    {
        uint32_t vlStride = VL_FP32 * CONST_TWO;
        uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, vlStride);
        uint16_t regTailWidth = (ubFactor - repeatTimesFloor * vlStride) / CONST_TWO;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32,
                vregSin2Fp32;
            AscendC::MicroAPI::RegTensor<float> vregScale1Fp32, vregScale2Fp32;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<T_K_CACHE> vregQuant;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
                auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
                auto cosAddr1 = cosPtr1 + j * VL_FP32;
                auto cosAddr2 = cosPtr2 + j * VL_FP32;
                auto sinAddr1 = sinPtr1 + j * VL_FP32;
                auto sinAddr2 = sinPtr2 + j * VL_FP32;
                auto scaleAddr1 = scalePtr1 + j * VL_FP32;
                auto scaleAddr2 = scalePtr2 + j * VL_FP32;
                auto kQuantAddr1 = quantPtr1 + j * VL_FP32;
                auto kQuantAddr2 = quantPtr2 + j * VL_FP32;

                LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
                LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
                AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

                LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
                LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
                LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
                LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
                AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

                // 量化部分
                LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
                LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);

                // 将结果cast成对应的量化数据类型
                StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
                StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
            }

            // 尾VL计算范式
            auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
            auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
            auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
            auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
            auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
            auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;
            auto scaleAddr1 = scalePtr1 + repeatTimesFloor * VL_FP32;
            auto scaleAddr2 = scalePtr2 + repeatTimesFloor * VL_FP32;
            auto kQuantAddr1 = quantPtr1 + repeatTimesFloor * VL_FP32;
            auto kQuantAddr2 = quantPtr2 + repeatTimesFloor * VL_FP32;

            LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
            LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

            LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
            LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
            LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
            LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
            AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

            // 量化部分
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
            LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);

            // 将结果cast成对应的量化类型
            StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
            StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
        }
    }

    __aicore__ inline void RopeSymQuantWithVF(__ubuf__ T_KV *&outPtr1, __ubuf__ T_KV *&outPtr2,
                                              __ubuf__ T_K_CACHE *&quantPtr1, __ubuf__ T_K_CACHE *&quantPtr2,
                                              __ubuf__ T_KV *&ropePtr, __ubuf__ T_KV *&cosPtr1, __ubuf__ T_KV *&cosPtr2,
                                              __ubuf__ T_KV *&sinPtr1, __ubuf__ T_KV *&sinPtr2,
                                              __ubuf__ float *&scalePtr1, __ubuf__ float *&scalePtr2,
                                              __ubuf__ float *&tmpBufferPtr, uint32_t ubFactor)
    {
        uint32_t vlStride = VL_FP32 * CONST_TWO;
        uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, vlStride);
        uint16_t regTailWidth = (ubFactor - repeatTimesFloor * vlStride) / CONST_TWO;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32,
                vregSin2Fp32;
            AscendC::MicroAPI::RegTensor<float> vregScale1Fp32, vregScale2Fp32;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<T_K_CACHE> vregQuant;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
                auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
                auto cosAddr1 = cosPtr1 + j * VL_FP32;
                auto cosAddr2 = cosPtr2 + j * VL_FP32;
                auto sinAddr1 = sinPtr1 + j * VL_FP32;
                auto sinAddr2 = sinPtr2 + j * VL_FP32;
                auto scaleAddr1 = scalePtr1 + j * VL_FP32;
                auto scaleAddr2 = scalePtr2 + j * VL_FP32;
                auto kQuantAddr1 = quantPtr1 + j * VL_FP32;
                auto kQuantAddr2 = quantPtr2 + j * VL_FP32;
                auto kOutAddr1 = outPtr1 + j * VL_FP32;
                auto kOutAddr2 = outPtr2 + j * VL_FP32;

                LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
                LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
                AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

                LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
                LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
                LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
                LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
                AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

                // 中间结果
                StoreDataAndCastFromFp32(kOutAddr1, vregCos1Fp32, uReg, mask, width);
                StoreDataAndCastFromFp32(kOutAddr2, vregCos2Fp32, uReg, mask, width);

                // 量化部分
                LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
                LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);

                // 将结果cast成对应的量化类型
                StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
                StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
            }

            // 尾VL计算范式
            auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
            auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
            auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
            auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
            auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
            auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;
            auto scaleAddr1 = scalePtr1 + repeatTimesFloor * VL_FP32;
            auto scaleAddr2 = scalePtr2 + repeatTimesFloor * VL_FP32;
            auto kQuantAddr1 = quantPtr1 + repeatTimesFloor * VL_FP32;
            auto kQuantAddr2 = quantPtr2 + repeatTimesFloor * VL_FP32;
            auto kOutAddr1 = outPtr1 + repeatTimesFloor * VL_FP32;
            auto kOutAddr2 = outPtr2 + repeatTimesFloor * VL_FP32;

            LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
            LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

            LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
            LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
            LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
            LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
            AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

            // 中间结果
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            StoreDataAndCastFromFp32(kOutAddr1, vregCos1Fp32, uReg, mask, width);
            StoreDataAndCastFromFp32(kOutAddr2, vregCos2Fp32, uReg, mask, width);

            // 量化部分
            LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
            LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);

            // 将结果cast成对应的量化类型
            StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
            StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
        }
    }

    __aicore__ inline void RopeSymQuantWithKCache(int64_t xDimOffset, int64_t rowIdx, int64_t cosSinDimOffset,
                                                  int64_t kCacheRowOffset)
    {
        for (int64_t ubIdx = 0; ubIdx < ubFactorDkLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> ropeLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> cosLocalPart1 = inQueueCosSin.AllocTensor<T_KV>();
            kScaleLocalPart1 = kScaleOffsetQueue.AllocTensor<float>();

            kOutLocal = outQueue.AllocTensor<T_KV>();
            kQuantLocal =
                kOutLocal.template ReinterpretCast<T_K_CACHE>()[this->ubFactor * sizeof(T_KV) / sizeof(T_K_CACHE)];

            int64_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDkLoopCountCeil - 1 && ubFactorDkTail > 0) { // 尾块
                tmpFactor = ubFactorDkTail;
            }

            int64_t kInOffset = xDimOffset + this->dv + this->ubFactor * ubIdx;
            int64_t cosSinOffset1 = cosSinDimOffset + this->ubFactor * ubIdx / CONST_TWO;
            int64_t cosSinOffset2 = this->dk / CONST_TWO + cosSinOffset1;
            int64_t kScaleOffset1 = this->ubFactor * ubIdx / CONST_TWO;
            int64_t kScaleOffset2 = this->dk / CONST_TWO + kScaleOffset1;
            // 中间结果
            int64_t kOutOffset1 = rowIdx * this->dk + this->ubFactor * ubIdx / CONST_TWO;
            int64_t kOutOffset2 = this->dk / CONST_TWO + kOutOffset1;

            uint32_t cosSinLen = tmpFactor / CONST_TWO;
            uint32_t alignOffset = this->ubFactor / CONST_TWO;
            LocalTensor<T_KV> cosLocalPart2 = cosLocalPart1[alignOffset];
            LocalTensor<T_KV> sinLocalPart1 = cosLocalPart2[alignOffset];
            LocalTensor<T_KV> sinLocalPart2 = sinLocalPart1[alignOffset];
            kQuantLocal1 = kQuantLocal[alignOffset];
            kOutLocal1 = kOutLocal[alignOffset];
            kScaleLocalPart2 = kScaleLocalPart1[alignOffset];

            __ubuf__ T_KV *ropePtr = (__ubuf__ T_KV *)ropeLocal.GetPhyAddr();
            __ubuf__ T_KV *cosPtr1 = (__ubuf__ T_KV *)cosLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *cosPtr2 = (__ubuf__ T_KV *)cosLocalPart2.GetPhyAddr();
            __ubuf__ T_KV *sinPtr1 = (__ubuf__ T_KV *)sinLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *sinPtr2 = (__ubuf__ T_KV *)sinLocalPart2.GetPhyAddr();
            __ubuf__ float *scalePtr1 = (__ubuf__ float *)kScaleLocalPart1.GetPhyAddr();
            __ubuf__ float *scalePtr2 = (__ubuf__ float *)kScaleLocalPart2.GetPhyAddr();
            __ubuf__ T_K_CACHE *kQuantPtr1 = (__ubuf__ T_K_CACHE *)kQuantLocal.GetPhyAddr();
            __ubuf__ T_K_CACHE *kQuantPtr2 = (__ubuf__ T_K_CACHE *)kQuantLocal1.GetPhyAddr();
            __ubuf__ T_KV *kOutPtr1 = (__ubuf__ T_KV *)kOutLocal.GetPhyAddr();
            __ubuf__ T_KV *kOutPtr2 = (__ubuf__ T_KV *)kOutLocal1.GetPhyAddr();
            LocalTensor<float> tmpLocal = xPowBuffer.Get<float>();
            __ubuf__ float *tmpBufferPtr = (__ubuf__ float *)tmpLocal.GetPhyAddr();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
            AscendC::DataCopyPad(ropeLocal, this->kvGm[kInOffset], xDataCopyParams, padParams);
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV) / CONST_TWO;
            AscendC::DataCopyPad(cosLocalPart1, this->cosGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(cosLocalPart2, this->cosGm[cosSinOffset2], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart1, this->sinGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart2, this->sinGm[cosSinOffset2], xDataCopyParams, padParams);

            if (tilingData_->kScaleType > 1) {
                xDataCopyParams.blockLen = cosSinLen * sizeof(float);
                AscendC::DataCopyPad(kScaleLocalPart1, this->kScaleGm[kScaleOffset1], xDataCopyParams, padParams);
                AscendC::DataCopyPad(kScaleLocalPart2, this->kScaleGm[kScaleOffset2], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(kScaleLocalPart1, this->kScaleGm, xDataCopyParams, padParams);
            }

            inQueueX.EnQue<T_KV>(ropeLocal);
            ropeLocal = inQueueX.DeQue<T_KV>();
            inQueueCosSin.EnQue<T_KV>(cosLocalPart1);
            cosLocalPart1 = inQueueCosSin.DeQue<T_KV>();
            kScaleOffsetQueue.EnQue<float>(kScaleLocalPart1);
            kScaleLocalPart1 = kScaleOffsetQueue.DeQue<float>();

            // 将scale brc
            if (tilingData_->kScaleType == 1) {
                AscendC::Duplicate<float>(kScaleLocalPart1, kScaleLocalPart1, this->ubFactor);
            }

            RopeSymQuantWithVF(kOutPtr1, kOutPtr2, kQuantPtr1, kQuantPtr2, ropePtr, cosPtr1, cosPtr2, sinPtr1, sinPtr2,
                               scalePtr1, scalePtr2, tmpBufferPtr, tmpFactor);

            outQueue.EnQue<T_KV>(kOutLocal);
            kOutLocal = outQueue.DeQue<T_KV>();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV) / CONST_TWO;
            AscendC::DataCopyPad(this->kOutGm[kOutOffset1], kOutLocal, xDataCopyParams);
            AscendC::DataCopyPad(this->kOutGm[kOutOffset2], kOutLocal1, xDataCopyParams);

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            } else {
                ScatterBlkUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            }

            outQueue.FreeTensor(kOutLocal);
            inQueueCosSin.FreeTensor(cosLocalPart1);
            kScaleOffsetQueue.FreeTensor(kScaleLocalPart1);
            inQueueX.FreeTensor(ropeLocal);
        }
    }

    __aicore__ inline void RopeAsymQuantVF(__ubuf__ T_K_CACHE *&quantPtr1, __ubuf__ T_K_CACHE *&quantPtr2,
                                           __ubuf__ T_KV *&ropePtr, __ubuf__ T_KV *&cosPtr1, __ubuf__ T_KV *&cosPtr2,
                                           __ubuf__ T_KV *&sinPtr1, __ubuf__ T_KV *&sinPtr2, __ubuf__ float *&scalePtr1,
                                           __ubuf__ float *&scalePtr2, __ubuf__ float *&offsetPtr1,
                                           __ubuf__ float *&offsetPtr2, __ubuf__ float *&tmpBufferPtr,
                                           uint32_t ubFactor)
    {
        uint32_t vlStride = VL_FP32 * CONST_TWO;
        uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, vlStride);
        uint16_t regTailWidth = (ubFactor - repeatTimesFloor * vlStride) / CONST_TWO;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32,
                vregSin2Fp32;
            AscendC::MicroAPI::RegTensor<float> vregScale1Fp32, vregScale2Fp32, vregOffset1Fp32, vregOffset2Fp32;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<T_K_CACHE> vregQuant;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
                auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
                auto cosAddr1 = cosPtr1 + j * VL_FP32;
                auto cosAddr2 = cosPtr2 + j * VL_FP32;
                auto sinAddr1 = sinPtr1 + j * VL_FP32;
                auto sinAddr2 = sinPtr2 + j * VL_FP32;
                auto scaleAddr1 = scalePtr1 + j * VL_FP32;
                auto scaleAddr2 = scalePtr2 + j * VL_FP32;
                auto offsetAdd1 = offsetPtr1 + j * VL_FP32;
                auto offsetAdd2 = offsetPtr2 + j * VL_FP32;
                auto kQuantAddr1 = quantPtr1 + j * VL_FP32;
                auto kQuantAddr2 = quantPtr2 + j * VL_FP32;

                LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
                LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
                AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

                LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
                LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
                LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
                LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
                AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

                // 量化部分
                LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
                LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);
                LoadDataAndCast2Fp32<float>(vregOffset1Fp32, offsetAdd1, mask);
                LoadDataAndCast2Fp32<float>(vregOffset2Fp32, offsetAdd2, mask);
                AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregOffset1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregOffset2Fp32, mask);

                // 将结果cast成对应的量化类型
                StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
                StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
            }

            // 尾VL计算范式
            auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
            auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
            auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
            auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
            auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
            auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;
            auto scaleAddr1 = scalePtr1 + repeatTimesFloor * VL_FP32;
            auto scaleAddr2 = scalePtr2 + repeatTimesFloor * VL_FP32;
            auto offsetAdd1 = offsetPtr1 + repeatTimesFloor * VL_FP32;
            auto offsetAdd2 = offsetPtr2 + repeatTimesFloor * VL_FP32;
            auto kQuantAddr1 = quantPtr1 + repeatTimesFloor * VL_FP32;
            auto kQuantAddr2 = quantPtr2 + repeatTimesFloor * VL_FP32;

            LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
            LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

            LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
            LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
            LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
            LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
            AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

            // 量化部分
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
            LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);
            LoadDataAndCast2Fp32<float>(vregOffset1Fp32, offsetAdd1, mask);
            LoadDataAndCast2Fp32<float>(vregOffset2Fp32, offsetAdd2, mask);
            AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregOffset1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregOffset2Fp32, mask);

            // 将结果cast成对应的量化类型
            StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
            StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
        }
    }

    __aicore__ inline void RopeAsymQuant(int64_t xDimOffset, int64_t rowIdx, int64_t cosSinDimOffset,
                                         int64_t kCacheRowOffset)
    {
        for (int64_t ubIdx = 0; ubIdx < ubFactorDkLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> ropeLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> cosLocalPart1 = inQueueCosSin.AllocTensor<T_KV>();
            kScaleLocalPart1 = kScaleOffsetQueue.AllocTensor<float>();
            kOffsetLocalPart1 = kScaleLocalPart1[this->ubFactor];
            kQuantLocal = outQueue.AllocTensor<T_K_CACHE>();

            int64_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDkLoopCountCeil - 1 && ubFactorDkTail > 0) { // 尾块
                tmpFactor = ubFactorDkTail;
            }

            int64_t kInOffset = xDimOffset + this->dv + this->ubFactor * ubIdx;
            int64_t cosSinOffset1 = cosSinDimOffset + this->ubFactor * ubIdx / CONST_TWO;
            int64_t cosSinOffset2 = this->dk / CONST_TWO + cosSinOffset1;
            int64_t kScaleOffset1 = this->ubFactor * ubIdx / CONST_TWO;
            int64_t kScaleOffset2 = this->dk / CONST_TWO + kScaleOffset1;
            int64_t kOffset1 = this->ubFactor * ubIdx / CONST_TWO;
            int64_t kOffset2 = this->dk / CONST_TWO + kOffset1;

            uint32_t cosSinLen = tmpFactor / CONST_TWO;
            uint32_t alignOffset = this->ubFactor / CONST_TWO;
            LocalTensor<T_KV> cosLocalPart2 = cosLocalPart1[alignOffset];
            LocalTensor<T_KV> sinLocalPart1 = cosLocalPart2[alignOffset];
            LocalTensor<T_KV> sinLocalPart2 = sinLocalPart1[alignOffset];
            kQuantLocal1 = kQuantLocal[alignOffset];
            kScaleLocalPart2 = kScaleLocalPart1[alignOffset];
            kOffsetLocalPart2 = kOffsetLocalPart1[alignOffset];

            __ubuf__ T_KV *ropePtr = (__ubuf__ T_KV *)ropeLocal.GetPhyAddr();
            __ubuf__ T_KV *cosPtr1 = (__ubuf__ T_KV *)cosLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *cosPtr2 = (__ubuf__ T_KV *)cosLocalPart2.GetPhyAddr();
            __ubuf__ T_KV *sinPtr1 = (__ubuf__ T_KV *)sinLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *sinPtr2 = (__ubuf__ T_KV *)sinLocalPart2.GetPhyAddr();
            __ubuf__ float *scalePtr1 = (__ubuf__ float *)kScaleLocalPart1.GetPhyAddr();
            __ubuf__ float *scalePtr2 = (__ubuf__ float *)kScaleLocalPart2.GetPhyAddr();
            __ubuf__ float *offsetPtr1 = (__ubuf__ float *)kOffsetLocalPart1.GetPhyAddr();
            __ubuf__ float *offsetPtr2 = (__ubuf__ float *)kOffsetLocalPart2.GetPhyAddr();
            __ubuf__ T_K_CACHE *kQuantPtr1 = (__ubuf__ T_K_CACHE *)kQuantLocal.GetPhyAddr();
            __ubuf__ T_K_CACHE *kQuantPtr2 = (__ubuf__ T_K_CACHE *)kQuantLocal1.GetPhyAddr();
            LocalTensor<float> tmpLocal = xPowBuffer.Get<float>();
            __ubuf__ float *tmpBufferPtr = (__ubuf__ float *)tmpLocal.GetPhyAddr();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
            AscendC::DataCopyPad(ropeLocal, this->kvGm[kInOffset], xDataCopyParams, padParams);
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV) / CONST_TWO;
            AscendC::DataCopyPad(cosLocalPart1, this->cosGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(cosLocalPart2, this->cosGm[cosSinOffset2], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart1, this->sinGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart2, this->sinGm[cosSinOffset2], xDataCopyParams, padParams);

            if (tilingData_->kScaleType > 1) {
                xDataCopyParams.blockLen = cosSinLen * sizeof(float);
                AscendC::DataCopyPad(kScaleLocalPart1, this->kScaleGm[kScaleOffset1], xDataCopyParams, padParams);
                AscendC::DataCopyPad(kScaleLocalPart2, this->kScaleGm[kScaleOffset2], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(kScaleLocalPart1, this->kScaleGm, xDataCopyParams, padParams);
            }

            if (tilingData_->kOffsetType > 1) {
                xDataCopyParams.blockLen = cosSinLen * sizeof(float);
                AscendC::DataCopyPad(kOffsetLocalPart1, this->kOffsetGm[kOffset1], xDataCopyParams, padParams);
                AscendC::DataCopyPad(kOffsetLocalPart2, this->kOffsetGm[kOffset2], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(kOffsetLocalPart1, this->kOffsetGm, xDataCopyParams, padParams);
            }

            inQueueX.EnQue<T_KV>(ropeLocal);
            ropeLocal = inQueueX.DeQue<T_KV>();
            inQueueCosSin.EnQue<T_KV>(cosLocalPart1);
            cosLocalPart1 = inQueueCosSin.DeQue<T_KV>();
            kScaleOffsetQueue.EnQue<float>(kScaleLocalPart1);
            kScaleLocalPart1 = kScaleOffsetQueue.DeQue<float>();

            // 将scale brc方便处理
            if (tilingData_->kScaleType == 1) {
                AscendC::Duplicate<float>(kScaleLocalPart1, kScaleLocalPart1, this->ubFactor);
            }

            if (tilingData_->kOffsetType == 1) {
                AscendC::Duplicate<float>(kOffsetLocalPart1, kOffsetLocalPart1, this->ubFactor);
            }

            RopeAsymQuantVF(kQuantPtr1, kQuantPtr2, ropePtr, cosPtr1, cosPtr2, sinPtr1, sinPtr2, scalePtr1, scalePtr2,
                            offsetPtr1, offsetPtr2, tmpBufferPtr, tmpFactor);

            outQueue.EnQue<T_K_CACHE>(kQuantLocal);
            kQuantLocal = outQueue.DeQue<T_K_CACHE>();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_K_CACHE) / CONST_TWO;

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            } else {
                ScatterBlkUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            }

            outQueue.FreeTensor(kQuantLocal);
            inQueueCosSin.FreeTensor(cosLocalPart1);
            kScaleOffsetQueue.FreeTensor(kScaleLocalPart1);
            inQueueX.FreeTensor(ropeLocal);
        }
    }

    // NZ dk1 为奇数(dk/2 非 dk0 对齐)的 scatter，泛化到任意大 dk / 多 tile（去掉整行拼接的单 tile 约束）。
    // 关键：cache 一行由 dk1 个「独立的 dk0 分形子块」组成（子块 i 存 dims[i*dk0,(i+1)*dk0)，按 blockSize*dk0
    // 跨步），并不需要一整行连续搬出——逐子块写即可。奇 dk1 时 dk/2 = m*dk0 + r（r=dk0/2, m=halfDk/dk0），
    //   · 前半区 dims[0,dk/2)：0..m-1 是满 dk0 子块（源天然 32B 对齐，直接写）；末尾 r 个是「跨界子块 m」的前半。
    //   · 后半区 dims[dk/2,dk)：整体相对子块网格错位 r。每个 back 网格块 g 拆两半 r：
    //       first-half（源 backLocal[g*dk0]，32B 对齐）→ 写 cache 子块(m+g) 的 [r,dk0)；
    //       second-half（源 backLocal[g*dk0+r]，错位 r 非对齐）→ 需重排：用 LoadAlign→StoreUnAlign 把整个 back
    //       切片右移 r 存进空闲 scratch（scratch[r+i]=backLocal[i]），则 second-half 源变 scratch[(g+1)*dk0]（对齐）。
    //     两半分别落 GM 的两个 half（互不相交），无需在 UB 里拼块，因此天然无跨 tile 携带。
    //   · 跨界子块 m：前半 r 来自前半区末尾（本 tile 若含），后半 r 来自 back 网格块 0 的 first-half（tile0）。
    //   · 末子块 2m(=dk1-1)：[0,r) 由 back 末网格块的 second-half 写，[r,dk0) 由 back 末尾 r（dims[dk-r,dk)）写。
    // 每 tile 只处理本 tile 覆盖的 [base,base+width)（per-half）：base=ubFactor*ubIdx/2（dk0 对齐），
    // width=per-half tile 宽。跨界前半 / back 末尾 r 仅在最后一个 tile（base+width==halfDk）出现。
    // scratch 复用 frontLocal[ubFactor] 之后的空闲尾部（三种 out/quant buffer 布局下都有 >= ubFactor 个空闲元素）。
    // T=T_KV(非量化 kOutLocal) 或 T_K_CACHE(量化 kQuantLocal)；rowBase 为该 token 行首(列0)的 cache GM 偏移。
    template <typename T>
    __aicore__ inline void ScatterKNzOddDk1(const LocalTensor<T> &frontLocal, const LocalTensor<T> &backLocal,
                                            int64_t rowBase, int64_t base, int64_t width)
    {
        int64_t halfDk = tilingData_->dk / CONST_TWO;
        int64_t r = dk0 / CONST_TWO;
        int64_t m = halfDk / dk0;     // 跨界子块下标；halfDk = m*dk0 + r
        int64_t baseBlk = base / dk0; // 本 tile 前半区首个子块下标
        int64_t fullEndDim = (base + width < m * dk0) ? (base + width) : (m * dk0);
        int64_t numFull = (fullEndDim - base) / dk0; // 本 tile 满 dk0 子块数（前/后半一致）
        bool isLastTile = (base + width == halfDk);
        int64_t blkStride = tilingData_->blockSize * dk0; // 子块间元素跨步
        int64_t elemSize = static_cast<int64_t>(sizeof(T));

        // ① 把 back 切片右移 r 重排进 scratch：scratch[r+i]=backLocal[i]（LoadAlign 对齐读 + StoreUnAlign 非对齐写，
        //    这是非对齐 UB 存的正确指令；DIST_PACK_B32 会 VEC_ERROR）。主 VL 循环 + 无条件尾 VL(VF 惯例，不用 if)：
        //    width 恒 >= r > 0，取 repeatFloor=(width-1)/VL_FP32 使尾宽 tailW 恒落 [1,VL_FP32]（width 为 VL_FP32
        //    整数倍时尾块吃满一整 VL），故尾块无需判空。
        LocalTensor<T> scratch = frontLocal[this->ubFactor];
        __ubuf__ T *srcPtr = (__ubuf__ T *)backLocal.GetPhyAddr();
        __ubuf__ T *dstPtr = (__ubuf__ T *)scratch.GetPhyAddr() + r; // 非对齐 dst，StoreUnAlign 允许
        uint16_t repeatFloor = static_cast<uint16_t>((width - 1) / VL_FP32);
        uint32_t tailW = static_cast<uint32_t>(width - static_cast<int64_t>(repeatFloor) * VL_FP32);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T> reg;
            AscendC::MicroAPI::UnalignRegForStore ureg;
            for (uint16_t j = 0; j < repeatFloor; j++) {
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    reg, srcPtr, static_cast<int32_t>(VL_FP32));
                AscendC::MicroAPI::StoreUnAlign(dstPtr, reg, ureg, static_cast<uint32_t>(VL_FP32));
            }
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                reg, srcPtr, static_cast<int32_t>(tailW));
            AscendC::MicroAPI::StoreUnAlign(dstPtr, reg, ureg, tailW);
            AscendC::MicroAPI::StoreUnAlignPost(dstPtr, ureg, static_cast<int32_t>(0));
        }
        event_t evtVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evtVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(evtVToMTE3);

        // 逐子块 blockCount=1 写（与对齐路径同款，避免非对齐/strided 歧义）
        AscendC::DataCopyExtParams p;
        p.blockCount = 1;
        p.srcStride = 0;
        p.dstStride = 0;
        // ② 前半区满子块：frontLocal[i*dk0] → cache 子块(baseBlk+i)
        p.blockLen = static_cast<uint32_t>(dk0 * elemSize);
        for (int64_t i = 0; i < numFull; i++) {
            AscendC::DataCopyPad(this->kCacheGm[rowBase + (baseBlk + i) * blkStride], frontLocal[i * dk0], p);
        }
        // ③④ back 两半 r：first-half（backLocal[g*dk0] 对齐）→ 子块(m+baseBlk+g) 的 [r,dk0)（g=0/tile0 即跨界子块 m
        // 后半）；
        //    second-half（scratch[(g+1)*dk0] 右移重排后对齐）→ 子块(m+baseBlk+g+1) 的 [0,r)。两半落 GM 不同 half。
        p.blockLen = static_cast<uint32_t>(r * elemSize);
        for (int64_t g = 0; g < numFull; g++) {
            AscendC::DataCopyPad(this->kCacheGm[rowBase + (m + baseBlk + g) * blkStride + r], backLocal[g * dk0], p);
            AscendC::DataCopyPad(this->kCacheGm[rowBase + (m + baseBlk + g + 1) * blkStride], scratch[(g + 1) * dk0],
                                 p);
        }
        // ⑤ 最后一个 tile：跨界子块 m 前半 r（frontLocal 前半区末尾，对齐）+ 末子块 2m 尾 r（backLocal
        // 后半区末尾，对齐）
        if (isLastTile) {
            AscendC::DataCopyPad(this->kCacheGm[rowBase + m * blkStride], frontLocal[m * dk0 - base], p);
            AscendC::DataCopyPad(this->kCacheGm[rowBase + (CONST_TWO * m) * blkStride + r], backLocal[m * dk0 - base],
                                 p);
        }
    }

    // odd-dk1 每个 tile 的分派公共实现：rowBase 由各 cache_mode 算好传入，其余(base 计算 + 量化/非量化 constexpr
    // 分派)相同
    __aicore__ inline void DispatchScatterKNzOddDk1(int64_t rowBase, int64_t ubIdx, int64_t tmpFactor)
    {
        int64_t base = this->ubFactor * ubIdx / CONST_TWO;
        if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                      IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
            ScatterKNzOddDk1<T_K_CACHE>(kQuantLocal, kQuantLocal1, rowBase, base, tmpFactor);
        } else {
            ScatterKNzOddDk1<T_KV>(kOutLocal, kOutLocal1, rowBase, base, tmpFactor);
        }
    }

    __aicore__ inline void ScatterUpdateK(int64_t rowIdx, int64_t ubIdx, int64_t tmpFactor)
    {
        xDataCopyParams.blockLen = tmpFactor * sizeof(T_K_CACHE);
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t batchIdx = rowIdx / tilingData_->seqLength;
        int64_t seqIdx = rowIdx % tilingData_->seqLength;
        int64_t kCacheIdx = this->indexGm(rowIdx);
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        int64_t gmOffset1, gmOffset2;

        // index 的取值是 cache 的行号/槽位号，越界即写到 cache 之外，故必须同时校验上界
        if (kCacheIdx >= 0 && kCacheIdx < tilingData_->cacheRowLimit) {
            if (tilingData_->cacheMode == NORM_CACHE_MODE) {
                gmOffset1 =
                    (batchIdx * tilingData_->cacheLength + kCacheIdx) * tilingData_->dk + this->ubFactor * ubIdx / 2;
                gmOffset2 = gmOffset1 + tilingData_->dk / 2;
                if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                              IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kQuantLocal, xDataCopyParams);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kQuantLocal1, xDataCopyParams);
                } else {
                    xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kOutLocal, xDataCopyParams);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kOutLocal1, xDataCopyParams);
                }
            } else if (tilingData_->cacheMode == PA_CACHE_MODE) {
                gmOffset1 = kCacheIdx * tilingData_->dk + this->ubFactor * ubIdx / 2;
                gmOffset2 = gmOffset1 + tilingData_->dk / 2;
                if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                              IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kQuantLocal, xDataCopyParams);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kQuantLocal1, xDataCopyParams);
                } else {
                    xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kOutLocal, xDataCopyParams);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kOutLocal1, xDataCopyParams);
                }
            } else if (tilingData_->cacheMode == PA_NZ_CACHE_MODE) {
                batchIdx = kCacheIdx / tilingData_->blockSize;
                int64_t blockOffset = kCacheIdx % tilingData_->blockSize;
                // dk1 奇数(dk/2 非 dk0 对齐)：逐 dk0 子块写 + back 半区右移 r 重排；泛化到任意 dk / 多 tile。
                // 仅 rowBase(行首偏移)按 cache_mode 各算，其余分派见 DispatchScatterKNzOddDk1
                if ((tilingData_->dk / CONST_TWO) % dk0 != 0) {
                    int64_t rowBase = batchIdx * tilingData_->dk * tilingData_->blockSize + blockOffset * dk0;
                    DispatchScatterKNzOddDk1(rowBase, ubIdx, tmpFactor);
                    return;
                }
                if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                              IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                    xDataCopyParams.blockLen = dk0 * sizeof(T_K_CACHE);
                } else {
                    xDataCopyParams.blockLen = dk0 * sizeof(T_KV);
                }
                for (int64_t i = 0; i < tmpFactor / dk0; i++) {
                    gmOffset1 = batchIdx * tilingData_->dk * tilingData_->blockSize +
                                (blockOffset + i * tilingData_->blockSize) * dk0 +
                                this->ubFactor * ubIdx * tilingData_->blockSize / 2;
                    gmOffset2 = gmOffset1 + tilingData_->dk / 2 * tilingData_->blockSize;
                    int64_t ubOffset = i * dk0;
                    if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                                  IsSameType<T_K_CACHE, fp8_e5m2_t>::value ||
                                  IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kQuantLocal[ubOffset], xDataCopyParams);
                        AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kQuantLocal1[ubOffset], xDataCopyParams);
                    } else {
                        AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kOutLocal[ubOffset], xDataCopyParams);
                        AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kOutLocal1[ubOffset], xDataCopyParams);
                    }
                }
            } else {
                return;
            }
        }
    }

    __aicore__ inline void ScatterBlkUpdateK(int64_t rowIdx, int64_t ubIdx, int64_t tmpFactor)
    {
        xDataCopyParams.blockLen = tmpFactor * sizeof(T_K_CACHE);
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t ceilValue = ops::CeilDiv(tilingData_->seqLength, tilingData_->blockSize);
        int64_t batchIdx = rowIdx / tilingData_->seqLength;
        int64_t seqIdx = rowIdx % tilingData_->seqLength;
        int64_t blkIdx = seqIdx / tilingData_->blockSize;
        int64_t idx = batchIdx * ceilValue + blkIdx;
        int64_t kCacheIdx = this->indexGm(idx);
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

        int64_t gmOffset1, gmOffset2;

        // index 的取值是 cache 的行号/槽位号，越界即写到 cache 之外，故必须同时校验上界
        if (kCacheIdx >= 0 && kCacheIdx < tilingData_->cacheRowLimit) {
            if (tilingData_->cacheMode == PA_BLK_BNSD_CACHE_MODE) {
                int64_t blockOffset = kCacheIdx / tilingData_->blockSize;
                int64_t rowOffset = seqIdx % tilingData_->blockSize;
                gmOffset1 =
                    (blockOffset * tilingData_->blockSize + rowOffset) * tilingData_->dk + ubIdx * this->ubFactor / 2;
                gmOffset2 = gmOffset1 + tilingData_->dk / 2;
                if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                              IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kQuantLocal, xDataCopyParams);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kQuantLocal1, xDataCopyParams);
                } else {
                    xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kOutLocal, xDataCopyParams);
                    AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kOutLocal1, xDataCopyParams);
                }
            } else {
                int64_t blockOffset = kCacheIdx / tilingData_->blockSize;
                int64_t rowOffset = seqIdx % tilingData_->blockSize;
                // dk1 奇数：逐 dk0 子块写 + back 半区右移 r 重排(PA_BLK_NZ 行首 = blockOffset*dk*blockSize +
                // rowOffset*dk0)
                if ((tilingData_->dk / CONST_TWO) % dk0 != 0) {
                    int64_t rowBase = blockOffset * tilingData_->dk * tilingData_->blockSize + rowOffset * dk0;
                    DispatchScatterKNzOddDk1(rowBase, ubIdx, tmpFactor);
                    return;
                }
                if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                              IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                    xDataCopyParams.blockLen = dk0 * sizeof(T_K_CACHE);
                } else {
                    xDataCopyParams.blockLen = dk0 * sizeof(T_KV);
                }
                for (int i = 0; i < tmpFactor / dk0; i++) {
                    gmOffset1 = blockOffset * tilingData_->dk * tilingData_->blockSize +
                                (rowOffset + i * tilingData_->blockSize) * dk0 +
                                this->ubFactor * ubIdx * tilingData_->blockSize / 2;
                    gmOffset2 = gmOffset1 + tilingData_->dk * tilingData_->blockSize / 2;
                    int64_t ubOffset = i * dk0;
                    if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                                  IsSameType<T_K_CACHE, fp8_e5m2_t>::value ||
                                  IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
                        AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kQuantLocal[ubOffset], xDataCopyParams);
                        AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kQuantLocal1[ubOffset], xDataCopyParams);
                    } else {
                        AscendC::DataCopyPad(this->kCacheGm[gmOffset1], kOutLocal[ubOffset], xDataCopyParams);
                        AscendC::DataCopyPad(this->kCacheGm[gmOffset2], kOutLocal1[ubOffset], xDataCopyParams);
                    }
                }
            }
        }
    }

    __aicore__ inline void RopeSymQuant(int64_t xDimOffset, int64_t rowIdx, int64_t cosSinDimOffset,
                                        int64_t kCacheRowOffset)
    {
        for (int64_t ubIdx = 0; ubIdx < ubFactorDkLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> ropeLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> cosLocalPart1 = inQueueCosSin.AllocTensor<T_KV>();
            kScaleLocalPart1 = kScaleOffsetQueue.AllocTensor<float>();
            kQuantLocal = outQueue.AllocTensor<T_K_CACHE>();

            int64_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDkLoopCountCeil - 1 && ubFactorDkTail > 0) { // 尾块
                tmpFactor = ubFactorDkTail;
            }

            int64_t kInOffset = xDimOffset + this->dv + this->ubFactor * ubIdx;
            int64_t cosSinOffset1 = cosSinDimOffset + this->ubFactor * ubIdx / CONST_TWO;
            int64_t cosSinOffset2 = this->dk / CONST_TWO + cosSinOffset1;
            int64_t kScaleOffset1 = this->ubFactor * ubIdx / CONST_TWO;
            int64_t kScaleOffset2 = this->dk / CONST_TWO + kScaleOffset1;

            uint32_t cosSinLen = tmpFactor / CONST_TWO;
            uint32_t alignOffset = this->ubFactor / CONST_TWO;
            LocalTensor<T_KV> cosLocalPart2 = cosLocalPart1[alignOffset];
            LocalTensor<T_KV> sinLocalPart1 = cosLocalPart2[alignOffset];
            LocalTensor<T_KV> sinLocalPart2 = sinLocalPart1[alignOffset];
            kQuantLocal1 = kQuantLocal[alignOffset];
            kScaleLocalPart2 = kScaleLocalPart1[alignOffset];

            __ubuf__ T_KV *ropePtr = (__ubuf__ T_KV *)ropeLocal.GetPhyAddr();
            __ubuf__ T_KV *cosPtr1 = (__ubuf__ T_KV *)cosLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *cosPtr2 = (__ubuf__ T_KV *)cosLocalPart2.GetPhyAddr();
            __ubuf__ T_KV *sinPtr1 = (__ubuf__ T_KV *)sinLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *sinPtr2 = (__ubuf__ T_KV *)sinLocalPart2.GetPhyAddr();
            __ubuf__ float *scalePtr1 = (__ubuf__ float *)kScaleLocalPart1.GetPhyAddr();
            __ubuf__ float *scalePtr2 = (__ubuf__ float *)kScaleLocalPart2.GetPhyAddr();
            __ubuf__ T_K_CACHE *kQuantPtr1 = (__ubuf__ T_K_CACHE *)kQuantLocal.GetPhyAddr();
            __ubuf__ T_K_CACHE *kQuantPtr2 = (__ubuf__ T_K_CACHE *)kQuantLocal1.GetPhyAddr();
            LocalTensor<float> tmpLocal = xPowBuffer.Get<float>();
            __ubuf__ float *tmpBufferPtr = (__ubuf__ float *)tmpLocal.GetPhyAddr();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
            AscendC::DataCopyPad(ropeLocal, this->kvGm[kInOffset], xDataCopyParams, padParams);
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV) / CONST_TWO;
            AscendC::DataCopyPad(cosLocalPart1, this->cosGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(cosLocalPart2, this->cosGm[cosSinOffset2], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart1, this->sinGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart2, this->sinGm[cosSinOffset2], xDataCopyParams, padParams);

            if (tilingData_->kScaleType > 1) {
                xDataCopyParams.blockLen = cosSinLen * sizeof(float);
                AscendC::DataCopyPad(kScaleLocalPart1, this->kScaleGm[kScaleOffset1], xDataCopyParams, padParams);
                AscendC::DataCopyPad(kScaleLocalPart2, this->kScaleGm[kScaleOffset2], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(kScaleLocalPart1, this->kScaleGm, xDataCopyParams, padParams);
            }

            inQueueX.EnQue<T_KV>(ropeLocal);
            ropeLocal = inQueueX.DeQue<T_KV>();
            inQueueCosSin.EnQue<T_KV>(cosLocalPart1);
            cosLocalPart1 = inQueueCosSin.DeQue<T_KV>();
            kScaleOffsetQueue.EnQue<float>(kScaleLocalPart1);
            kScaleLocalPart1 = kScaleOffsetQueue.DeQue<float>();

            // 将scale brc
            if (tilingData_->kScaleType == 1) {
                AscendC::Duplicate<float>(kScaleLocalPart1, kScaleLocalPart1, this->ubFactor);
            }

            RopeSymQuantVF(kQuantPtr1, kQuantPtr2, ropePtr, cosPtr1, cosPtr2, sinPtr1, sinPtr2, scalePtr1, scalePtr2,
                           tmpBufferPtr, tmpFactor);

            outQueue.EnQue<T_K_CACHE>(kQuantLocal);
            kQuantLocal = outQueue.DeQue<T_K_CACHE>();

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            } else {
                ScatterBlkUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            }

            outQueue.FreeTensor(kQuantLocal);
            inQueueCosSin.FreeTensor(cosLocalPart1);
            kScaleOffsetQueue.FreeTensor(kScaleLocalPart1);
            inQueueX.FreeTensor(ropeLocal);
        }
    }

    __aicore__ inline void RopeWithoutQuant(int64_t xDimOffset, int64_t rowIdx, int64_t cosSinOffset,
                                            int64_t kCacheRowOffset)
    {
        for (int64_t ubIdx = 0; ubIdx < ubFactorDkLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> ropeLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> cosLocalPart1 = inQueueCosSin.AllocTensor<T_KV>();
            kOutLocal = outQueue.AllocTensor<T_KV>();

            int64_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDkLoopCountCeil - 1 && ubFactorDkTail > 0) {
                tmpFactor = ubFactorDkTail;
            }

            int64_t kInOffset = xDimOffset + this->dv + this->ubFactor * ubIdx;
            int64_t cosSinOffset1, cosSinOffset2;

            cosSinOffset1 = cosSinOffset + this->ubFactor * ubIdx / CONST_TWO;
            cosSinOffset2 = this->dk / CONST_TWO + cosSinOffset1;

            int64_t kOutOffset1 = rowIdx * this->dk + this->ubFactor * ubIdx / CONST_TWO;
            int64_t kOutOffset2 = this->dk / CONST_TWO + kOutOffset1;

            int64_t alignOffset = this->ubFactor / CONST_TWO;
            LocalTensor<T_KV> cosLocalPart2 = cosLocalPart1[alignOffset];
            LocalTensor<T_KV> sinLocalPart1 = cosLocalPart2[alignOffset];
            LocalTensor<T_KV> sinLocalPart2 = sinLocalPart1[alignOffset];
            kOutLocal1 = kOutLocal[alignOffset];

            __ubuf__ T_KV *ropePtr = (__ubuf__ T_KV *)ropeLocal.GetPhyAddr();
            __ubuf__ T_KV *cosPtr1 = (__ubuf__ T_KV *)cosLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *cosPtr2 = (__ubuf__ T_KV *)cosLocalPart2.GetPhyAddr();
            __ubuf__ T_KV *sinPtr1 = (__ubuf__ T_KV *)sinLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *sinPtr2 = (__ubuf__ T_KV *)sinLocalPart2.GetPhyAddr();
            __ubuf__ T_KV *outPtr1 = (__ubuf__ T_KV *)kOutLocal.GetPhyAddr();
            __ubuf__ T_KV *outPtr2 = (__ubuf__ T_KV *)kOutLocal1.GetPhyAddr();
            LocalTensor<float> tmpLocal = xPowBuffer.Get<float>();
            __ubuf__ float *tmpBufferPtr = (__ubuf__ float *)tmpLocal.GetPhyAddr();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
            AscendC::DataCopyPad(ropeLocal, this->kvGm[kInOffset], xDataCopyParams, padParams);
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV) / CONST_TWO;
            AscendC::DataCopyPad(cosLocalPart1, this->cosGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(cosLocalPart2, this->cosGm[cosSinOffset2], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart1, this->sinGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart2, this->sinGm[cosSinOffset2], xDataCopyParams, padParams);

            inQueueX.EnQue<T_KV>(ropeLocal);
            ropeLocal = inQueueX.DeQue<T_KV>();
            inQueueCosSin.EnQue<T_KV>(cosLocalPart1);
            cosLocalPart1 = inQueueCosSin.DeQue<T_KV>();

            RopeVF(outPtr1, outPtr2, ropePtr, cosPtr1, cosPtr2, sinPtr1, sinPtr2, tmpBufferPtr, tmpFactor);

            outQueue.EnQue<T_KV>(kOutLocal);
            kOutLocal = outQueue.DeQue<T_KV>();

            if (tilingData_->isOutputKv) {
                AscendC::DataCopyPad(this->kOutGm[kOutOffset1], kOutLocal, xDataCopyParams);
                AscendC::DataCopyPad(this->kOutGm[kOutOffset2], kOutLocal1, xDataCopyParams);
            }

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            } else {
                ScatterBlkUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            }

            outQueue.FreeTensor(kOutLocal);
            inQueueCosSin.FreeTensor(cosLocalPart1);
            inQueueX.FreeTensor(ropeLocal);
        }
    }

    __aicore__ inline void RopeAsymQuantWithKvVF(
        __ubuf__ T_KV *&outPtr1, __ubuf__ T_KV *&outPtr2, __ubuf__ T_K_CACHE *&quantPtr1,
        __ubuf__ T_K_CACHE *&quantPtr2, __ubuf__ T_KV *&ropePtr, __ubuf__ T_KV *&cosPtr1, __ubuf__ T_KV *&cosPtr2,
        __ubuf__ T_KV *&sinPtr1, __ubuf__ T_KV *&sinPtr2, __ubuf__ float *&scalePtr1, __ubuf__ float *&scalePtr2,
        __ubuf__ float *&offsetPtr1, __ubuf__ float *&offsetPtr2, __ubuf__ float *&tmpBufferPtr, uint32_t ubFactor)
    {
        uint32_t vlStride = VL_FP32 * CONST_TWO;
        uint16_t repeatTimesFloor = ops::FloorDiv(ubFactor, vlStride);
        uint16_t regTailWidth = (ubFactor - repeatTimesFloor * vlStride) / CONST_TWO;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32,
                vregSin2Fp32;
            AscendC::MicroAPI::RegTensor<float> vregScale1Fp32, vregScale2Fp32, vregOffset1Fp32, vregOffset2Fp32;
            AscendC::MicroAPI::RegTensor<half> vregHalf;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16;
            AscendC::MicroAPI::RegTensor<T_K_CACHE> vregQuant;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::UnalignRegForStore uReg;
            uint32_t width = VL_FP32;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);

            // 标准VL计算范式
            for (uint16_t j = 0; j < repeatTimesFloor; j++) {
                auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
                auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
                auto cosAddr1 = cosPtr1 + j * VL_FP32;
                auto cosAddr2 = cosPtr2 + j * VL_FP32;
                auto sinAddr1 = sinPtr1 + j * VL_FP32;
                auto sinAddr2 = sinPtr2 + j * VL_FP32;
                auto scaleAddr1 = scalePtr1 + j * VL_FP32;
                auto scaleAddr2 = scalePtr2 + j * VL_FP32;
                auto offsetAdd1 = offsetPtr1 + j * VL_FP32;
                auto offsetAdd2 = offsetPtr2 + j * VL_FP32;
                auto kQuantAddr1 = quantPtr1 + j * VL_FP32;
                auto kQuantAddr2 = quantPtr2 + j * VL_FP32;
                auto kOutAddr1 = outPtr1 + j * VL_FP32;
                auto kOutAddr2 = outPtr2 + j * VL_FP32;

                LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
                LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
                AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
                AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                               AscendC::MicroAPI::MemType::VEC_LOAD>();
                AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

                LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
                LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
                LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
                LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
                AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
                AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);
                // 中间结果
                StoreDataAndCastFromFp32(kOutAddr1, vregCos1Fp32, uReg, mask, width);
                StoreDataAndCastFromFp32(kOutAddr2, vregCos2Fp32, uReg, mask, width);

                // 量化部分
                LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
                LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
                AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
                AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);
                LoadDataAndCast2Fp32<float>(vregOffset1Fp32, offsetAdd1, mask);
                LoadDataAndCast2Fp32<float>(vregOffset2Fp32, offsetAdd2, mask);
                AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregOffset1Fp32, mask);
                AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregOffset2Fp32, mask);

                // 将结果cast成对应的量化类型
                StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
                StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
            }

            // 尾VL计算范式
            auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
            auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
            auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
            auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
            auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
            auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;
            auto scaleAddr1 = scalePtr1 + repeatTimesFloor * VL_FP32;
            auto scaleAddr2 = scalePtr2 + repeatTimesFloor * VL_FP32;
            auto offsetAdd1 = offsetPtr1 + repeatTimesFloor * VL_FP32;
            auto offsetAdd2 = offsetPtr2 + repeatTimesFloor * VL_FP32;
            auto kQuantAddr1 = quantPtr1 + repeatTimesFloor * VL_FP32;
            auto kQuantAddr2 = quantPtr2 + repeatTimesFloor * VL_FP32;
            auto kOutAddr1 = outPtr1 + repeatTimesFloor * VL_FP32;
            auto kOutAddr2 = outPtr2 + repeatTimesFloor * VL_FP32;

            LoadDataAndCast2Fp32(vregRope1Fp32, ropeAddr1, mask);
            LoadDataAndCast2Fp32(vregRope2Fp32, ropeAddr2, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
            AscendC::MicroAPI::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                vregRope1Fp32, vregRope2Fp32, ((__ubuf__ float *)(tmpBufferPtr)));

            LoadDataAndCast2Fp32(vregCos1Fp32, cosAddr1, mask);
            LoadDataAndCast2Fp32(vregCos2Fp32, cosAddr2, mask);
            LoadDataAndCast2Fp32(vregSin1Fp32, sinAddr1, mask);
            LoadDataAndCast2Fp32(vregSin2Fp32, sinAddr2, mask);

            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
            AscendC::MicroAPI::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
            AscendC::MicroAPI::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);
            // 中间结果
            width = regTailWidth;
            mask = AscendC::MicroAPI::UpdateMask<float>(width);
            StoreDataAndCastFromFp32(kOutAddr1, vregCos1Fp32, uReg, mask, width);
            StoreDataAndCastFromFp32(kOutAddr2, vregCos2Fp32, uReg, mask, width);

            // 量化部分
            LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
            LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
            AscendC::MicroAPI::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
            AscendC::MicroAPI::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);
            LoadDataAndCast2Fp32<float>(vregOffset1Fp32, offsetAdd1, mask);
            LoadDataAndCast2Fp32<float>(vregOffset2Fp32, offsetAdd2, mask);
            AscendC::MicroAPI::Add(vregCos1Fp32, vregCos1Fp32, vregOffset1Fp32, mask);
            AscendC::MicroAPI::Add(vregCos2Fp32, vregCos2Fp32, vregOffset2Fp32, mask);

            // 将结果cast成对应的量化类型
            StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
            StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
        }
    }

    __aicore__ inline void RopeAsymQuantWithKCache(int64_t xDimOffset, int64_t rowIdx, int64_t cosSinDimOffset,
                                                   int64_t kCacheRowOffset)
    {
        for (int64_t ubIdx = 0; ubIdx < ubFactorDkLoopCountCeil; ubIdx++) {
            LocalTensor<T_KV> ropeLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> cosLocalPart1 = inQueueCosSin.AllocTensor<T_KV>();
            kScaleLocalPart1 = kScaleOffsetQueue.AllocTensor<float>();
            kOffsetLocalPart1 = kScaleLocalPart1[this->ubFactor];

            kOutLocal = outQueue.AllocTensor<T_KV>();
            kQuantLocal =
                kOutLocal.template ReinterpretCast<T_K_CACHE>()[this->ubFactor * sizeof(T_KV) / sizeof(T_K_CACHE)];

            int64_t tmpFactor = this->ubFactor;
            if (ubIdx == ubFactorDkLoopCountCeil - 1 && ubFactorDkTail > 0) { // 尾块
                tmpFactor = ubFactorDkTail;
            }

            int64_t kInOffset = xDimOffset + this->dv + this->ubFactor * ubIdx;
            int64_t cosSinOffset1 = cosSinDimOffset + this->ubFactor * ubIdx / CONST_TWO;
            int64_t cosSinOffset2 = this->dk / CONST_TWO + cosSinOffset1;
            int64_t kScaleOffset1 = this->ubFactor * ubIdx / CONST_TWO;
            int64_t kScaleOffset2 = this->dk / CONST_TWO + kScaleOffset1;
            int64_t kOffset1 = this->ubFactor * ubIdx / CONST_TWO;
            int64_t kOffset2 = this->dk / CONST_TWO + kOffset1;
            // 中间结果
            int64_t kOutOffset1 = rowIdx * this->dk + this->ubFactor * ubIdx / CONST_TWO;
            int64_t kOutOffset2 = this->dk / CONST_TWO + kOutOffset1;

            uint32_t cosSinLen = tmpFactor / CONST_TWO;
            uint32_t alignOffset = this->ubFactor / CONST_TWO;
            LocalTensor<T_KV> cosLocalPart2 = cosLocalPart1[alignOffset];
            LocalTensor<T_KV> sinLocalPart1 = cosLocalPart2[alignOffset];
            LocalTensor<T_KV> sinLocalPart2 = sinLocalPart1[alignOffset];
            kQuantLocal1 = kQuantLocal[alignOffset];
            kOutLocal1 = kOutLocal[alignOffset];
            kScaleLocalPart2 = kScaleLocalPart1[alignOffset];
            kOffsetLocalPart2 = kOffsetLocalPart1[alignOffset];

            __ubuf__ T_KV *ropePtr = (__ubuf__ T_KV *)ropeLocal.GetPhyAddr();
            __ubuf__ T_KV *cosPtr1 = (__ubuf__ T_KV *)cosLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *cosPtr2 = (__ubuf__ T_KV *)cosLocalPart2.GetPhyAddr();
            __ubuf__ T_KV *sinPtr1 = (__ubuf__ T_KV *)sinLocalPart1.GetPhyAddr();
            __ubuf__ T_KV *sinPtr2 = (__ubuf__ T_KV *)sinLocalPart2.GetPhyAddr();
            __ubuf__ float *scalePtr1 = (__ubuf__ float *)kScaleLocalPart1.GetPhyAddr();
            __ubuf__ float *scalePtr2 = (__ubuf__ float *)kScaleLocalPart2.GetPhyAddr();
            __ubuf__ float *offsetPtr1 = (__ubuf__ float *)kOffsetLocalPart1.GetPhyAddr();
            __ubuf__ float *offsetPtr2 = (__ubuf__ float *)kOffsetLocalPart2.GetPhyAddr();
            __ubuf__ T_K_CACHE *kQuantPtr1 = (__ubuf__ T_K_CACHE *)kQuantLocal.GetPhyAddr();
            __ubuf__ T_K_CACHE *kQuantPtr2 = (__ubuf__ T_K_CACHE *)kQuantLocal1.GetPhyAddr();
            __ubuf__ T_KV *kOutPtr1 = (__ubuf__ T_KV *)kOutLocal.GetPhyAddr();
            __ubuf__ T_KV *kOutPtr2 = (__ubuf__ T_KV *)kOutLocal1.GetPhyAddr();
            LocalTensor<float> tmpLocal = xPowBuffer.Get<float>();
            __ubuf__ float *tmpBufferPtr = (__ubuf__ float *)tmpLocal.GetPhyAddr();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV);
            AscendC::DataCopyPad(ropeLocal, this->kvGm[kInOffset], xDataCopyParams, padParams);
            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV) / CONST_TWO;
            AscendC::DataCopyPad(cosLocalPart1, this->cosGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(cosLocalPart2, this->cosGm[cosSinOffset2], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart1, this->sinGm[cosSinOffset1], xDataCopyParams, padParams);
            AscendC::DataCopyPad(sinLocalPart2, this->sinGm[cosSinOffset2], xDataCopyParams, padParams);

            if (tilingData_->kScaleType > 1) {
                xDataCopyParams.blockLen = cosSinLen * sizeof(float);
                AscendC::DataCopyPad(kScaleLocalPart1, this->kScaleGm[kScaleOffset1], xDataCopyParams, padParams);
                AscendC::DataCopyPad(kScaleLocalPart2, this->kScaleGm[kScaleOffset2], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(kScaleLocalPart1, this->kScaleGm, xDataCopyParams, padParams);
            }

            if (tilingData_->kOffsetType > 1) {
                xDataCopyParams.blockLen = cosSinLen * sizeof(float);
                AscendC::DataCopyPad(kOffsetLocalPart1, this->kOffsetGm[kOffset1], xDataCopyParams, padParams);
                AscendC::DataCopyPad(kOffsetLocalPart2, this->kOffsetGm[kOffset2], xDataCopyParams, padParams);
            } else {
                xDataCopyParams.blockLen = sizeof(float);
                AscendC::DataCopyPad(kOffsetLocalPart1, this->kOffsetGm, xDataCopyParams, padParams);
            }

            inQueueX.EnQue<T_KV>(ropeLocal);
            ropeLocal = inQueueX.DeQue<T_KV>();
            inQueueCosSin.EnQue<T_KV>(cosLocalPart1);
            cosLocalPart1 = inQueueCosSin.DeQue<T_KV>();
            kScaleOffsetQueue.EnQue<float>(kScaleLocalPart1);
            kScaleLocalPart1 = kScaleOffsetQueue.DeQue<float>();

            // 将scale brc方便处理
            if (tilingData_->kScaleType == 1) {
                AscendC::Duplicate<float>(kScaleLocalPart1, kScaleLocalPart1, this->ubFactor);
            }

            if (tilingData_->kOffsetType == 1) {
                AscendC::Duplicate<float>(kOffsetLocalPart1, kOffsetLocalPart1, this->ubFactor);
            }

            RopeAsymQuantWithKvVF(kOutPtr1, kOutPtr2, kQuantPtr1, kQuantPtr2, ropePtr, cosPtr1, cosPtr2, sinPtr1,
                                  sinPtr2, scalePtr1, scalePtr2, offsetPtr1, offsetPtr2, tmpBufferPtr, tmpFactor);

            outQueue.EnQue<T_KV>(kOutLocal);
            kOutLocal = outQueue.DeQue<T_KV>();

            xDataCopyParams.blockLen = tmpFactor * sizeof(T_KV) / CONST_TWO;
            AscendC::DataCopyPad(this->kOutGm[kOutOffset1], kOutLocal, xDataCopyParams);
            AscendC::DataCopyPad(this->kOutGm[kOutOffset2], kOutLocal1, xDataCopyParams);

            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            } else {
                ScatterBlkUpdateK(kCacheRowOffset, ubIdx, tmpFactor / 2);
            }

            outQueue.FreeTensor(kOutLocal);
            inQueueCosSin.FreeTensor(cosLocalPart1);
            kScaleOffsetQueue.FreeTensor(kScaleLocalPart1);
            inQueueX.FreeTensor(ropeLocal);
        }
    }

    __aicore__ inline void Rope(int64_t xDimOffset, int64_t rowIdx, int64_t cosSinOffset, int64_t kCacheRowOffset)
    {
        if constexpr (IsSameType<T_K_CACHE, int8_t>::value || IsSameType<T_K_CACHE, hifloat8_t>::value ||
                      IsSameType<T_K_CACHE, fp8_e5m2_t>::value || IsSameType<T_K_CACHE, fp8_e4m3fn_t>::value) {
            // 量化部分需要增加量化输出 quantOutPtr
            if (tilingData_->isOutputKv > 0) {
                // 需要输出中间结果
                if (tilingData_->kOffsetType > 0) {
                    // 量化 + 偏移
                    RopeAsymQuantWithKCache(xDimOffset, rowIdx, cosSinOffset, kCacheRowOffset);
                } else {
                    // 量化
                    RopeSymQuantWithKCache(xDimOffset, rowIdx, cosSinOffset, kCacheRowOffset);
                }
            } else {
                // 不需要输出中间结果
                if (tilingData_->kOffsetType > 0) {
                    // 量化 + 偏移
                    RopeAsymQuant(xDimOffset, rowIdx, cosSinOffset, kCacheRowOffset);
                } else {
                    // 量化
                    RopeSymQuant(xDimOffset, rowIdx, cosSinOffset, kCacheRowOffset);
                }
            }
        } else {
            RopeWithoutQuant(xDimOffset, rowIdx, cosSinOffset, kCacheRowOffset);
        }
    }

    __aicore__ inline void Process()
    {
        int64_t kvGMOffsetPerCore = tilingData_->blockFactor * this->blockIdx; // rows offset for each core
        int64_t cosSinGmOffsetPerCore = tilingData_->blockFactor * this->blockIdx;
        int64_t cacheGmOffsetPerCore = tilingData_->blockFactor * this->blockIdx;

        // RmsNorm
        for (uint64_t rowIdx = 0; rowIdx < blockFactor; rowIdx++) {
            int64_t xDimOffset = (kvGMOffsetPerCore + rowIdx) * (this->dv + this->dk);
            int64_t cosSinDimOffset = (cosSinGmOffsetPerCore + rowIdx) * this->dk;
            int64_t vCacheRowOffset = cacheGmOffsetPerCore + rowIdx;
            int64_t kCacheRowOffset = cacheGmOffsetPerCore + rowIdx;

            // for RmsNorm
            RmsNorm(xDimOffset, rowIdx, vCacheRowOffset);

            int64_t cosSinOffset;
            if (tilingData_->cosSinNeedBrc == 1) {
                int64_t curRowIdx = cosSinDimOffset / this->dk;
                int64_t curBatchIdx = ops::FloorDiv(curRowIdx, tilingData_->seqLength);
                cosSinOffset = curBatchIdx * this->dk;
            } else {
                cosSinOffset = cosSinDimOffset;
            }

            // for Rope
            Rope(xDimOffset, rowIdx, cosSinOffset, kCacheRowOffset);
        }
    }

private:
    const KvRmsNormRopeCacheRegbaseRecomputeTilingData *tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueGamma, inQueueCosSin;
    TQue<QuePosition::VECIN, 1> kScaleOffsetQueue, vScaleOffsetQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> xPowBuffer;
    TBuf<TPosition::VECCALC> cacheBuffer;
    TBuf<TPosition::VECCALC> tmpReduceBuffer;
    TPipe *pipe_ = nullptr;

    LocalTensor<float> tmpReduceLocal;

    LocalTensor<T_KV> kOutLocal;
    LocalTensor<T_KV> kOutLocal1;
    LocalTensor<T_K_CACHE> kQuantLocal;
    LocalTensor<T_K_CACHE> kQuantLocal1;
    LocalTensor<T_KV> vOutLocal;
    LocalTensor<T_V_CACHE> vQuantLocal;

    LocalTensor<float> kScaleLocalPart1;
    LocalTensor<float> kScaleLocalPart2;
    LocalTensor<float> kOffsetLocalPart1;
    LocalTensor<float> kOffsetLocalPart2;

    event_t eventIDSToMTE3;

    uint32_t resultCacheID_ = 0;
    LocalTensor<float> totalSumLocal;

    int64_t bs = 0;
    int64_t ubFactorDvTail = 0;
    int64_t ubFactorDvLoopCountCeil = 0;
    int64_t ubFactorDkTail = 0;
    int64_t ubFactorDkLoopCountCeil = 0;

    int64_t blockFactor = 0; // rows need to calculate for each core
    int64_t basicBlockLoop = 0;
    int64_t mainFoldCount = 0;

    // NZ参数
    int64_t dk0 = 0;
    int64_t dk1 = 0;
    int64_t dv0 = 0;
    int64_t dv1 = 0;

    DataCopyParams xDataCopyParams;
    DataCopyPadParams padParams{false, 0, 0, 0};

    static constexpr bool xToFp32_ = !IsSameType<T_KV, float>::value;
    static constexpr bool yToFp32_ = IsSameType<T_KV, float>::value;
};
} // namespace KvRmsNormRopeCache

#endif // KV_RMS_NORM_ROPE_CACHE_REGBASE_RECOMPUTE_H_
