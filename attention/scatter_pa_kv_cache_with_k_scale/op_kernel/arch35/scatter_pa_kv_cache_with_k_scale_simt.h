/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_PA_KV_CACHE_WITH_K_SCALE_SIMT_H
#define SCATTER_PA_KV_CACHE_WITH_K_SCALE_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"
#include "scatter_pa_kv_cache_with_k_scale_tiling_data.h"
#include "scatter_pa_kv_cache_with_k_scale_tiling_key.h"

namespace NsScatterPaKvCacheWithKScale {

using namespace AscendC;
// ==================== 线程数配置常量 ====================
// 32位索引(uint32_t): 使用2048线程，优化寄存器利用率
// 64位索引(uint64_t): 使用1024线程，64位索引占用更多寄存器，需减少线程数保持相同寄存器预算

constexpr size_t SIZEOF_32BIT_IDX = 4;
constexpr size_t SIZEOF_64BIT_IDX = 8;
constexpr uint32_t THREADS_FOR_32BIT_IDX = 2048;
constexpr uint32_t THREADS_FOR_64BIT_IDX = 1024;

// ==================== stride数组索引常量 ====================

constexpr size_t STRIDE_IDX_TOKEN = 0;
constexpr size_t STRIDE_IDX_HEAD = 1;
constexpr size_t STRIDE_IDX_HEAD_SIZE = 2;
constexpr size_t STRIDE_IDX_ELEM = 3;

constexpr size_t STRIDE_IDX_BLOCK = 0;
constexpr size_t STRIDE_IDX_CACHE_HEAD = 1;
constexpr size_t STRIDE_IDX_BLOCK_OFFSET = 2;
constexpr size_t STRIDE_IDX_CACHE_ELEM = 3;

constexpr size_t STRIDE_IDX_SCALE_TOKEN = 0;
constexpr size_t STRIDE_IDX_SCALE_HEAD = 1;

constexpr size_t STRIDE_IDX_SCALE_CACHE_BLOCK = 0;
constexpr size_t STRIDE_IDX_SCALE_CACHE_HEAD = 1;
constexpr size_t STRIDE_IDX_SCALE_CACHE_OFFSET = 2;

template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == SIZEOF_32BIT_IDX) ? THREADS_FOR_32BIT_IDX : THREADS_FOR_64BIT_IDX;

// ==================== VF-A: 特化 Key+Value (全部参数直接传入) ====================

template <typename T_FP8, typename T_SLOT, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void ScatterKvKVVf(
    IDX_T count, __gm__ T_FP8* key, __gm__ T_FP8* value, __gm__ T_FP8* keyCache, __gm__ T_FP8* valueCache,
    __gm__ T_SLOT* slotMapping, IDX_T numHead, IDX_T kHeadSize, IDX_T blockSize, IDX_T maxSlot, IDX_T kStride0,
    IDX_T kStride1, IDX_T kStride2, IDX_T kcStride0, IDX_T kcStride1, IDX_T kcStride2, IDX_T kcStride3, IDX_T vStride0,
    IDX_T vStride1, IDX_T vStride2, IDX_T vcStride0, IDX_T vcStride1, IDX_T vcStride2, IDX_T vcStride3, IDX_T smStride0,
    IDX_T mNHKS, IDX_T sNHKS, IDX_T mHKS, IDX_T sHKS, IDX_T mBlk, IDX_T sBlk)
{
    IDX_T nhks = numHead * kHeadSize;

    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x) * THREADS<IDX_T> + static_cast<IDX_T>(threadIdx.x); idx < count;
         idx += THREADS<IDX_T> * static_cast<IDX_T>(gridDim.x)) {
        IDX_T token = Simt::UintDiv(idx, mNHKS, sNHKS);
        IDX_T rem = idx - token * nhks;
        IDX_T head = Simt::UintDiv(rem, mHKS, sHKS);
        IDX_T elem = rem - head * kHeadSize;

        T_SLOT slot = slotMapping[token * smStride0];
        if (slot < 0 || static_cast<IDX_T>(slot) >= maxSlot)
            continue;

        IDX_T blk = Simt::UintDiv(static_cast<IDX_T>(slot), mBlk, sBlk);
        IDX_T off = slot - blk * blockSize;

        keyCache[blk * kcStride0 + head * kcStride1 + off * kcStride2 + elem * kcStride3] =
            key[token * kStride0 + head * kStride1 + elem * kStride2];
        valueCache[blk * vcStride0 + head * vcStride1 + off * vcStride2 + elem * vcStride3] =
            value[token * vStride0 + head * vStride1 + elem * vStride2];
    }
}

// ==================== VF-B: 泛化 Key ====================

template <typename T_FP8, typename T_SLOT, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void ScatterKvKeyVf(
    IDX_T count, __gm__ T_FP8* key, __gm__ T_FP8* keyCache, __gm__ T_SLOT* slotMapping, IDX_T numHead, IDX_T kHeadSize,
    IDX_T blockSize, IDX_T maxSlot, IDX_T kStride0, IDX_T kStride1, IDX_T kStride2, IDX_T kcStride0, IDX_T kcStride1,
    IDX_T kcStride2, IDX_T kcStride3, IDX_T smStride0, IDX_T mNHKS, IDX_T sNHKS, IDX_T mHKS, IDX_T sHKS, IDX_T mBlk,
    IDX_T sBlk)
{
    IDX_T nhks = numHead * kHeadSize;

    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x) * THREADS<IDX_T> + static_cast<IDX_T>(threadIdx.x); idx < count;
         idx += THREADS<IDX_T> * static_cast<IDX_T>(gridDim.x)) {
        IDX_T token = Simt::UintDiv(idx, mNHKS, sNHKS);
        IDX_T rem = idx - token * nhks;
        IDX_T head = Simt::UintDiv(rem, mHKS, sHKS);
        IDX_T elem = rem - head * kHeadSize;

        T_SLOT slot = slotMapping[token * smStride0];
        if (slot < 0 || static_cast<IDX_T>(slot) >= maxSlot)
            continue;

        IDX_T blk = Simt::UintDiv(static_cast<IDX_T>(slot), mBlk, sBlk);
        IDX_T off = slot - blk * blockSize;

        keyCache[blk * kcStride0 + head * kcStride1 + off * kcStride2 + elem * kcStride3] =
            key[token * kStride0 + head * kStride1 + elem * kStride2];
    }
}

// ==================== VF-C: 泛化 Value ====================

template <typename T_FP8, typename T_SLOT, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void ScatterKvValueVf(
    IDX_T count, __gm__ T_FP8* value, __gm__ T_FP8* valueCache, __gm__ T_SLOT* slotMapping, IDX_T numHead,
    IDX_T vHeadSize, IDX_T blockSize, IDX_T maxSlot, IDX_T vStride0, IDX_T vStride1, IDX_T vStride2, IDX_T vcStride0,
    IDX_T vcStride1, IDX_T vcStride2, IDX_T vcStride3, IDX_T smStride0, IDX_T mNHVS, IDX_T sNHVS, IDX_T mHVS,
    IDX_T sHVS, IDX_T mBlk, IDX_T sBlk)
{
    IDX_T nhvs = numHead * vHeadSize;

    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x) * THREADS<IDX_T> + static_cast<IDX_T>(threadIdx.x); idx < count;
         idx += THREADS<IDX_T> * static_cast<IDX_T>(gridDim.x)) {
        IDX_T token = Simt::UintDiv(idx, mNHVS, sNHVS);
        IDX_T rem = idx - token * nhvs;
        IDX_T head = Simt::UintDiv(rem, mHVS, sHVS);
        IDX_T elem = rem - head * vHeadSize;

        T_SLOT slot = slotMapping[token * smStride0];
        if (slot < 0 || static_cast<IDX_T>(slot) >= maxSlot)
            continue;

        IDX_T blk = Simt::UintDiv(static_cast<IDX_T>(slot), mBlk, sBlk);
        IDX_T off = slot - blk * blockSize;

        valueCache[blk * vcStride0 + head * vcStride1 + off * vcStride2 + elem * vcStride3] =
            value[token * vStride0 + head * vStride1 + elem * vStride2];
    }
}

// ==================== VF-D: KeyScale 通用 ====================

template <typename T_SLOT, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void ScatterKvKeyScaleVf(
    IDX_T count, __gm__ float* keyScale, __gm__ float* keyScaleCache, __gm__ T_SLOT* slotMapping, IDX_T numHead,
    IDX_T blockSize, IDX_T maxSlot, IDX_T ksStride0, IDX_T ksStride1, IDX_T kscStride0, IDX_T kscStride1,
    IDX_T kscStride2, IDX_T smStride0, IDX_T mNH, IDX_T sNH, IDX_T mBlk, IDX_T sBlk)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x) * THREADS<IDX_T> + static_cast<IDX_T>(threadIdx.x); idx < count;
         idx += THREADS<IDX_T> * static_cast<IDX_T>(gridDim.x)) {
        IDX_T token = Simt::UintDiv(idx, mNH, sNH);
        IDX_T head = idx - token * numHead;

        T_SLOT slot = slotMapping[token * smStride0];
        if (slot < 0 || static_cast<IDX_T>(slot) >= maxSlot)
            continue;

        IDX_T blk = Simt::UintDiv(static_cast<IDX_T>(slot), mBlk, sBlk);
        IDX_T off = slot - blk * blockSize;

        keyScaleCache[blk * kscStride0 + head * kscStride1 + off * kscStride2] =
            keyScale[token * ksStride0 + head * ksStride1];
    }
}

// ==================== Launch helpers (magic/shift + stride cast + VF call) ====================

template <typename T_FP8, typename T_SLOT, typename IDX_T>
__aicore__ inline void LaunchKVVf(const ScatterPaKvCacheWithKScaleTilingData* t, IDX_T count, IDX_T nH, IDX_T kH,
                                  IDX_T bS, IDX_T mS, __gm__ T_FP8* kg, __gm__ T_FP8* vg, __gm__ T_FP8* kcg,
                                  __gm__ T_FP8* vcg, __gm__ T_SLOT* sg)
{
    IDX_T m1, s1, m2, s2, m3, s3;
    GetUintDivMagicAndShift(m1, s1, nH * kH);
    GetUintDivMagicAndShift(m2, s2, kH);
    GetUintDivMagicAndShift(m3, s3, bS);
    asc_vf_call<ScatterKvKVVf<T_FP8, T_SLOT, IDX_T>>(
        dim3(THREADS<IDX_T>), count, kg, vg, kcg, vcg, sg, nH, kH, bS, mS,
        static_cast<IDX_T>(t->keyStride[STRIDE_IDX_TOKEN]), static_cast<IDX_T>(t->keyStride[STRIDE_IDX_HEAD]),
        static_cast<IDX_T>(t->keyStride[STRIDE_IDX_HEAD_SIZE]), static_cast<IDX_T>(t->keyCacheStride[STRIDE_IDX_BLOCK]),
        static_cast<IDX_T>(t->keyCacheStride[STRIDE_IDX_CACHE_HEAD]),
        static_cast<IDX_T>(t->keyCacheStride[STRIDE_IDX_BLOCK_OFFSET]),
        static_cast<IDX_T>(t->keyCacheStride[STRIDE_IDX_CACHE_ELEM]),
        static_cast<IDX_T>(t->valueStride[STRIDE_IDX_TOKEN]), static_cast<IDX_T>(t->valueStride[STRIDE_IDX_HEAD]),
        static_cast<IDX_T>(t->valueStride[STRIDE_IDX_HEAD_SIZE]),
        static_cast<IDX_T>(t->valueCacheStride[STRIDE_IDX_BLOCK]),
        static_cast<IDX_T>(t->valueCacheStride[STRIDE_IDX_CACHE_HEAD]),
        static_cast<IDX_T>(t->valueCacheStride[STRIDE_IDX_BLOCK_OFFSET]),
        static_cast<IDX_T>(t->valueCacheStride[STRIDE_IDX_CACHE_ELEM]),
        static_cast<IDX_T>(t->slotMappingStride[STRIDE_IDX_TOKEN]), m1, s1, m2, s2, m3, s3);
}

template <typename T_FP8, typename T_SLOT, typename IDX_T>
__aicore__ inline void LaunchKeyVf(const ScatterPaKvCacheWithKScaleTilingData* t, IDX_T count, IDX_T nH, IDX_T kH,
                                   IDX_T bS, IDX_T mS, __gm__ T_FP8* kg, __gm__ T_FP8* kcg, __gm__ T_SLOT* sg)
{
    IDX_T m1, s1, m2, s2, m3, s3;
    GetUintDivMagicAndShift(m1, s1, nH * kH);
    GetUintDivMagicAndShift(m2, s2, kH);
    GetUintDivMagicAndShift(m3, s3, bS);
    asc_vf_call<ScatterKvKeyVf<T_FP8, T_SLOT, IDX_T>>(
        dim3(THREADS<IDX_T>), count, kg, kcg, sg, nH, kH, bS, mS, static_cast<IDX_T>(t->keyStride[STRIDE_IDX_TOKEN]),
        static_cast<IDX_T>(t->keyStride[STRIDE_IDX_HEAD]), static_cast<IDX_T>(t->keyStride[STRIDE_IDX_HEAD_SIZE]),
        static_cast<IDX_T>(t->keyCacheStride[STRIDE_IDX_BLOCK]),
        static_cast<IDX_T>(t->keyCacheStride[STRIDE_IDX_CACHE_HEAD]),
        static_cast<IDX_T>(t->keyCacheStride[STRIDE_IDX_BLOCK_OFFSET]),
        static_cast<IDX_T>(t->keyCacheStride[STRIDE_IDX_CACHE_ELEM]),
        static_cast<IDX_T>(t->slotMappingStride[STRIDE_IDX_TOKEN]), m1, s1, m2, s2, m3, s3);
}

template <typename T_FP8, typename T_SLOT, typename IDX_T>
__aicore__ inline void LaunchValueVf(const ScatterPaKvCacheWithKScaleTilingData* t, IDX_T count, IDX_T nH, IDX_T vH,
                                     IDX_T bS, IDX_T mS, __gm__ T_FP8* vg, __gm__ T_FP8* vcg, __gm__ T_SLOT* sg)
{
    IDX_T m1, s1, m2, s2, m3, s3;
    GetUintDivMagicAndShift(m1, s1, nH * vH);
    GetUintDivMagicAndShift(m2, s2, vH);
    GetUintDivMagicAndShift(m3, s3, bS);
    asc_vf_call<ScatterKvValueVf<T_FP8, T_SLOT, IDX_T>>(
        dim3(THREADS<IDX_T>), count, vg, vcg, sg, nH, vH, bS, mS, static_cast<IDX_T>(t->valueStride[STRIDE_IDX_TOKEN]),
        static_cast<IDX_T>(t->valueStride[STRIDE_IDX_HEAD]), static_cast<IDX_T>(t->valueStride[STRIDE_IDX_HEAD_SIZE]),
        static_cast<IDX_T>(t->valueCacheStride[STRIDE_IDX_BLOCK]),
        static_cast<IDX_T>(t->valueCacheStride[STRIDE_IDX_CACHE_HEAD]),
        static_cast<IDX_T>(t->valueCacheStride[STRIDE_IDX_BLOCK_OFFSET]),
        static_cast<IDX_T>(t->valueCacheStride[STRIDE_IDX_CACHE_ELEM]),
        static_cast<IDX_T>(t->slotMappingStride[STRIDE_IDX_TOKEN]), m1, s1, m2, s2, m3, s3);
}

template <typename T_SLOT, typename IDX_T>
__aicore__ inline void LaunchKeyScaleVf(const ScatterPaKvCacheWithKScaleTilingData* t, IDX_T count, IDX_T nH, IDX_T bS,
                                        IDX_T mS, __gm__ float* ksg, __gm__ float* kscg, __gm__ T_SLOT* sg)
{
    IDX_T m1, s1, m2, s2;
    GetUintDivMagicAndShift(m1, s1, nH);
    GetUintDivMagicAndShift(m2, s2, bS);
    asc_vf_call<ScatterKvKeyScaleVf<T_SLOT, IDX_T>>(
        dim3(THREADS<IDX_T>), count, ksg, kscg, sg, nH, bS, mS,
        static_cast<IDX_T>(t->keyScaleStride[STRIDE_IDX_SCALE_TOKEN]),
        static_cast<IDX_T>(t->keyScaleStride[STRIDE_IDX_SCALE_HEAD]),
        static_cast<IDX_T>(t->keyScaleCacheStride[STRIDE_IDX_SCALE_CACHE_BLOCK]),
        static_cast<IDX_T>(t->keyScaleCacheStride[STRIDE_IDX_SCALE_CACHE_HEAD]),
        static_cast<IDX_T>(t->keyScaleCacheStride[STRIDE_IDX_SCALE_CACHE_OFFSET]),
        static_cast<IDX_T>(t->slotMappingStride[STRIDE_IDX_TOKEN]), m1, s1, m2, s2);
}

template <typename T_FP8, typename T_SLOT, uint32_t SCH_MODE>
__aicore__ inline void Process(GM_ADDR key, GM_ADDR value, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR slotMapping,
                               GM_ADDR keyScale, GM_ADDR keyScaleCache, const ScatterPaKvCacheWithKScaleTilingData* t)
{
    uint64_t nT = static_cast<uint64_t>(t->numTokens);
    uint64_t nH = static_cast<uint64_t>(t->numHead);
    uint64_t kH = static_cast<uint64_t>(t->kHeadSize);
    uint64_t vH = static_cast<uint64_t>(t->vHeadSize);
    uint64_t bS = static_cast<uint64_t>(t->blockSize);
    uint64_t total = nT * nH * ((kH > vH) ? kH : vH);
    bool use32 = (total <= static_cast<uint64_t>(INT32_MAX));

    __gm__ T_FP8* kg = (__gm__ T_FP8*)key;
    __gm__ T_FP8* vg = (__gm__ T_FP8*)value;
    __gm__ T_FP8* kcg = (__gm__ T_FP8*)keyCache;
    __gm__ T_FP8* vcg = (__gm__ T_FP8*)valueCache;
    __gm__ T_SLOT* sg = (__gm__ T_SLOT*)slotMapping;
    __gm__ float* ksg = (__gm__ float*)keyScale;
    __gm__ float* kscg = (__gm__ float*)keyScaleCache;

    if constexpr (SCH_MODE == SCATTER_KV_CACHE_SCENE_SPECIALIZED) {
        if (use32) {
            uint32_t c = static_cast<uint32_t>(nT * nH * kH);
            uint32_t cKS = static_cast<uint32_t>(nT * nH);
            uint32_t _nH = static_cast<uint32_t>(nH);
            uint32_t _kH = static_cast<uint32_t>(kH);
            uint32_t _bS = static_cast<uint32_t>(bS);
            uint32_t _mS = static_cast<uint32_t>(t->maxSlot);
            LaunchKVVf<T_FP8, T_SLOT, uint32_t>(t, c, _nH, _kH, _bS, _mS, kg, vg, kcg, vcg, sg);
            LaunchKeyScaleVf<T_SLOT, uint32_t>(t, cKS, _nH, _bS, _mS, ksg, kscg, sg);
        } else {
            uint64_t c = nT * nH * kH;
            uint64_t cKS = nT * nH;
            uint64_t mS = static_cast<uint64_t>(t->maxSlot);
            LaunchKVVf<T_FP8, T_SLOT, uint64_t>(t, c, nH, kH, bS, mS, kg, vg, kcg, vcg, sg);
            LaunchKeyScaleVf<T_SLOT, uint64_t>(t, cKS, nH, bS, mS, ksg, kscg, sg);
        }
    } else {
        if (use32) {
            uint32_t cK = static_cast<uint32_t>(nT * nH * kH);
            uint32_t cV = static_cast<uint32_t>(nT * nH * vH);
            uint32_t cKS = static_cast<uint32_t>(nT * nH);
            uint32_t _nH = static_cast<uint32_t>(nH);
            uint32_t _kH = static_cast<uint32_t>(kH);
            uint32_t _vH = static_cast<uint32_t>(vH);
            uint32_t _bS = static_cast<uint32_t>(bS);
            uint32_t _mS = static_cast<uint32_t>(t->maxSlot);
            LaunchKeyVf<T_FP8, T_SLOT, uint32_t>(t, cK, _nH, _kH, _bS, _mS, kg, kcg, sg);
            LaunchValueVf<T_FP8, T_SLOT, uint32_t>(t, cV, _nH, _vH, _bS, _mS, vg, vcg, sg);
            LaunchKeyScaleVf<T_SLOT, uint32_t>(t, cKS, _nH, _bS, _mS, ksg, kscg, sg);
        } else {
            uint64_t cK = nT * nH * kH;
            uint64_t cV = nT * nH * vH;
            uint64_t cKS = nT * nH;
            uint64_t mS = static_cast<uint64_t>(t->maxSlot);
            LaunchKeyVf<T_FP8, T_SLOT, uint64_t>(t, cK, nH, kH, bS, mS, kg, kcg, sg);
            LaunchValueVf<T_FP8, T_SLOT, uint64_t>(t, cV, nH, vH, bS, mS, vg, vcg, sg);
            LaunchKeyScaleVf<T_SLOT, uint64_t>(t, cKS, nH, bS, mS, ksg, kscg, sg);
        }
    }

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

} // namespace NsScatterPaKvCacheWithKScale
#endif // SCATTER_PA_KV_CACHE_WITH_K_SCALE_SIMT_H
