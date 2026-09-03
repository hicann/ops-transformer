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
 * \file stem_oam_prep_paged_kv_simd.h
 * \brief
 */

#ifndef STEM_OAM_PREP_PAGED_KV_SIMD_H
#define STEM_OAM_PREP_PAGED_KV_SIMD_H

#include <algorithm>
#include <cstdint>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "stem_oam_prep_paged_kv_tiling_data.h"
#include "stem_oam_prep_paged_kv_base.h"

using namespace AscendC;

constexpr static float EPSILON = 1e-6;
constexpr static uint32_t BLOCKSIZE = 32;

class StemOamPrepPagedKvSimd {
public:
    __aicore__ inline StemOamPrepPagedKvSimd() = default;
    __aicore__ inline void Process(void);
    __aicore__ inline void Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices, GM_ADDR kvSeqLens,
                                GM_ADDR kScaleCache, GM_ADDR vScale, GM_ADDR kFlat, GM_ADDR vBias, GM_ADDR workspace,
                                const StemOamPrepPagedKvTilingData *tilingData, TPipe *pipe);

private:
    __aicore__ inline void CopyInKVCache(int32_t batchIdx, int32_t headIDx, int32_t stemBlockIdx, int64_t actualRows);
    __aicore__ inline void CopyInVFlatOut(int32_t batchIdx, int32_t headIdx, int64_t numStemBlocks);
    __aicore__ inline void CopyInKvSeqLens(void);

    __aicore__ inline void ComputeKFlat(void);
    __aicore__ inline void ComputeVFlat(int64_t headIdx, int64_t stemBlockIdx, int32_t kvLen);
    __aicore__ inline void ComputeLogVals(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                          __local_mem__ float *vFLatOutAddr);
    template <bool KDownLenFlag>
    __aicore__ inline void ComputeVStd(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                       __local_mem__ float *logValsAddr, float vMean);
    __aicore__ inline void ComputeVBias(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                        __local_mem__ float *logValsAddr, float vMean, float vSTd);
    __aicore__ inline void ComputeVBiasOut(int32_t batchIdx, int32_t headIdx, int64_t kDownLen, int64_t numStemBlocks);
    __aicore__ inline void ProcessKVFlat(int32_t batchIdx, int32_t headIdx, int32_t stemBlockIdx);
    __aicore__ inline void ProcessVBias(void);
    __aicore__ inline void ComputTotalStemBlockAndProcessKVFlat(void);

    __aicore__ inline void CopyOutKFlat(int32_t batchIdx, int32_t headIDx, int32_t stemBlocksIdx);
    __aicore__ inline void CopyOutVFlat(int32_t batchIdx, int32_t headIDx, int32_t stemBlocksIdx);
    __aicore__ inline void CopyOutVBias(int32_t batchIdx, int32_t headIdx, int64_t numStemBlocks);

private:
    TPipe *pipe_;
    StemOamPrepPagedKvTilingData tilingData_;

    GlobalTensor<fp8_e4m3fn_t> kCacheGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheGm_;
    GlobalTensor<int32_t> kvIndicesGm_;
    GlobalTensor<int32_t> kvSeqLensGm_;
    GlobalTensor<float> kScaleCacheGm_;
    GlobalTensor<float> vScaleGm_;

    GlobalTensor<bfloat16_t> kFlatGm_;
    GlobalTensor<float> vBiasGm_;

    GlobalTensor<float> vFlatOutWs_;

    TQue<QuePosition::VECIN, 1> kCacheQue_;
    TQue<QuePosition::VECIN, 1> vCacheQue_;
    TQue<QuePosition::VECIN, 1> kvSeqLensQue_;
    TQue<QuePosition::VECIN, 1> vFLatOutQue_;

    TQue<QuePosition::VECOUT, 1> groupSumQue_;
    TQue<QuePosition::VECOUT, 1> vNormDownQue_;
    TQue<QuePosition::VECOUT, 1> vBiasOutQue_;

    TBuf<TPosition::VECCALC> kScalarBuf_;
    TBuf<TPosition::VECCALC> totalStemBlockBuf_;
    TBuf<TPosition::VECCALC> meanTempBuf_;

    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 0;
    int64_t kvLayout_ = 0;
    int64_t kvBlockSize_ = 0;
    int64_t hKv_ = 0;
    int64_t maxKvBlocks_ = 0;
    int64_t dimQk_ = 0;
    int64_t dimV_ = 0;
    int64_t maxKb_ = 0;
    int64_t kflatDim_ = 0;
    int64_t batchSize_ = 0;
    int64_t stemBlockSize_ = 0;
    int64_t stemStride_ = 0;
    int64_t totalBh_ = 0;
    float lambdaMag_ = 0;
    int64_t r_ = 0;
    int64_t kCacheStride_[4] = {0};
    int64_t vCacheStride_[4] = {0};
    int64_t kScaleCacheStride_[4] = {0};
    int64_t vCacheOffset_ = 0;
    int64_t totalStemBlockNum_ = 0;
    int64_t curCoreProcessStemBlock_ = 0;
    int64_t tailCoreProcessStemBlock_ = 0;
    int64_t kvUsedCoreNum_ = 0;
    int64_t meanSize_ = 0;

    constexpr static int64_t V_REG_SIZE = Ops::Base::GetVRegSize();
};

__aicore__ inline void StemOamPrepPagedKvSimd::Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices,
                                                    GM_ADDR kvSeqLens, GM_ADDR kScaleCache, GM_ADDR vScale,
                                                    GM_ADDR kFlat, GM_ADDR vBias, GM_ADDR workspace,
                                                    const StemOamPrepPagedKvTilingData *tilingData, TPipe *pipe)
{
    kCacheGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)kCache);
    vCacheGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)vCache);
    kvIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)kvIndices);
    kvSeqLensGm_.SetGlobalBuffer((__gm__ int32_t *)kvSeqLens);
    kScaleCacheGm_.SetGlobalBuffer((__gm__ float *)kScaleCache);
    vScaleGm_.SetGlobalBuffer((__gm__ float *)vScale);
    kFlatGm_.SetGlobalBuffer((__gm__ bfloat16_t *)kFlat);
    vBiasGm_.SetGlobalBuffer((__gm__ float *)vBias);

    vFlatOutWs_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    kvLayout_ = tilingData->kvLayout;
    kvBlockSize_ = tilingData->kvBlockSize;
    hKv_ = tilingData->numKvHeads;
    maxKvBlocks_ = tilingData->maxKvBlocks;
    dimQk_ = tilingData->dimQk;
    dimV_ = tilingData->dimV;
    kflatDim_ = tilingData->kflatDim;
    batchSize_ = tilingData->batchSize;
    stemBlockSize_ = tilingData->stemBlockSize;
    stemStride_ = tilingData->stemStride;
    lambdaMag_ = tilingData->lambdaMag;
    maxKb_ = tilingData->maxKb;
    meanSize_ = tilingData->meanSize;
    r_ = tilingData->rVal;
    for (uint16_t i = 0; i < 4; i++) {
        kCacheStride_[i] = tilingData->kCacheStride[i];
        vCacheStride_[i] = tilingData->vCacheStride[i];
        kScaleCacheStride_[i] = tilingData->kScaleCacheStride[i];
    }
    totalBh_ = batchSize_ * hKv_;
    vCacheOffset_ = stemBlockSize_ * dimQk_;
    // inti UB
    pipe_->InitBuffer(kCacheQue_, 1, stemBlockSize_ * dimQk_ * sizeof(fp8_e4m3fn_t));
    pipe_->InitBuffer(vCacheQue_, 1, stemBlockSize_ * dimQk_ * sizeof(fp8_e4m3fn_t));
    pipe_->InitBuffer(kvSeqLensQue_, 1, batchSize_ * sizeof(int32_t));

    pipe_->InitBuffer(groupSumQue_, 1, stemStride_ * dimQk_ * sizeof(bfloat16_t));
    pipe_->InitBuffer(vNormDownQue_, 1, stemBlockSize_ * sizeof(float));

    pipe_->InitBuffer(kScalarBuf_, stemBlockSize_ * sizeof(float));
    pipe_->InitBuffer(totalStemBlockBuf_, sizeof(int32_t));

    if (blockIdx_ == 0) {
        InitOutput<bfloat16_t>(kFlatGm_, batchSize_ * hKv_ * maxKb_ * kflatDim_, bfloat16_t(0));
        InitOutput<float>(vBiasGm_, batchSize_ * hKv_ * maxKb_, float(0));
    }
    SyncAll();
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputTotalStemBlockAndProcessKVFlat(void)
{
    CopyInKvSeqLens();
    LocalTensor<int32_t> kvSeqLensLocal = kvSeqLensQue_.DeQue<int32_t>();
    LocalTensor<int32_t> totalStemBlockLocal = totalStemBlockBuf_.Get<int32_t>();

    __local_mem__ int32_t *kvSeqLenAddr = (__ubuf__ int32_t *)kvSeqLensLocal.GetPhyAddr();
    __local_mem__ int32_t *totalStemBlockAddr = (__ubuf__ int32_t *)totalStemBlockLocal.GetPhyAddr();
    __local_mem__ int32_t *kvStemBlockAddr = kvSeqLenAddr;
    uint32_t vfLen = V_REG_SIZE / sizeof(int32_t);
    uint16_t loopCnt = (batchSize_ + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> kvSeqLensReg;
        AscendC::Reg::RegTensor<int32_t> divNumReg;
        AscendC::Reg::RegTensor<int32_t> kvstemBlockReg;
        AscendC::Reg::RegTensor<int32_t> kvstemBlockAllReg;
        AscendC::Reg::RegTensor<int32_t> stemBlockNumReg;
        AscendC::Reg::RegTensor<int32_t> stemBlockSumNumReg;
        AscendC::Reg::MaskReg valueMaskReg;
        uint32_t maskLen = static_cast<uint32_t>(batchSize_);
        auto stemBlockSumNumAddr = totalStemBlockAddr;
        AscendC::Reg::MaskReg allMaskReg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg oneMaskReg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();
        Reg::Duplicate(divNumReg, int32_t(stemBlockSize_), allMaskReg);
        Reg::Duplicate(stemBlockSumNumReg, int32_t(0), oneMaskReg);
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::Reg::UpdateMask<int32_t>(maskLen);
            AscendC::Reg::AddrReg kvSeqLensAddrOfst = AscendC::Reg::CreateAddrReg<int32_t>(i, vfLen);
            AscendC::Reg::LoadAlign(kvSeqLensReg, kvSeqLenAddr, kvSeqLensAddrOfst);
            AscendC::Reg::Adds(kvstemBlockReg, kvSeqLensReg, int32_t(stemBlockSize_ - 1), valueMaskReg);
            AscendC::Reg::Div(kvstemBlockReg, kvstemBlockReg, divNumReg, valueMaskReg);
            AscendC::Reg::Muls(kvstemBlockAllReg, kvstemBlockReg, int32_t(hKv_), valueMaskReg);
            AscendC::Reg::Reduce<Reg::ReduceType::SUM, int32_t, int32_t>(stemBlockNumReg, kvstemBlockAllReg,
                                                                         valueMaskReg);
            AscendC::Reg::Add(stemBlockSumNumReg, stemBlockSumNumReg, stemBlockNumReg, oneMaskReg);
            AscendC::Reg::StoreAlign(kvStemBlockAddr + i * vfLen, kvstemBlockReg, valueMaskReg);
        }
        AscendC::Reg::Store(stemBlockSumNumAddr, stemBlockSumNumReg, 1);
    }
    EventMsg<HardEvent::V_S>();
    int32_t batchIdx = 0;
    int32_t hKvIdx = 0;
    int32_t stemBlockIdx = 0;
    int32_t baseStemBlockNum = 0;
    totalStemBlockNum_ = totalStemBlockLocal.GetValue(0);
    int32_t blockFlag = 0;
    int32_t setN = 0;
    curCoreProcessStemBlock_ = Ops::Base::CeilDiv(totalStemBlockNum_, blockNum_);
    kvUsedCoreNum_ = Ops::Base::CeilDiv(totalStemBlockNum_, curCoreProcessStemBlock_);
    tailCoreProcessStemBlock_ = totalStemBlockNum_ - (kvUsedCoreNum_ - 1) * curCoreProcessStemBlock_;
    if (blockIdx_ < kvUsedCoreNum_) {
        auto startStemBlock = blockIdx_ * curCoreProcessStemBlock_;
        auto endStemBlock = (blockIdx_ * curCoreProcessStemBlock_ >= totalStemBlockNum_) ?
                                totalStemBlockNum_ - 1 :
                                (startStemBlock + curCoreProcessStemBlock_);
        for (int32_t i = 0; i < totalStemBlockNum_; i++) {
            while (batchIdx < batchSize_ && kvSeqLensLocal.GetValue(batchIdx) == 0) {
                batchIdx++;
                hKvIdx = 0;
            }
            stemBlockIdx = i - baseStemBlockNum;
            if (kvSeqLensLocal.GetValue(batchIdx) == 0) {
                batchIdx++;
            } else {
                if (stemBlockIdx >= kvSeqLensLocal.GetValue(batchIdx)) {
                    baseStemBlockNum += kvSeqLensLocal.GetValue(batchIdx);
                    hKvIdx++;
                    stemBlockIdx = 0;
                }
                if (hKvIdx >= hKv_) {
                    batchIdx++;
                    hKvIdx = 0;
                    while (batchIdx < batchSize_ && kvSeqLensLocal.GetValue(batchIdx) == 0) {
                        batchIdx++;
                        hKvIdx = 0;
                    }
                }
            }
            if (i >= startStemBlock && i < endStemBlock) {
                ProcessKVFlat(batchIdx, hKvIdx, stemBlockIdx);
            }
            if (i == endStemBlock) {
                break;
            }
        }
    }
    kvSeqLensQue_.FreeTensor(kvSeqLensLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyInKvSeqLens(void)
{
    LocalTensor<int32_t> kvSeqLensLocal = kvSeqLensQue_.AllocTensor<int32_t>();

    CopyIn(kvSeqLensLocal, kvSeqLensGm_, 1, batchSize_, 0, 0);

    kvSeqLensQue_.EnQue(kvSeqLensLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyInVFlatOut(int32_t batchIdx, int32_t headIdx, int64_t numStemBlocks)
{
    LocalTensor<float> vFLatOutLocal = vFLatOutQue_.AllocTensor<float>();
    auto vFlatOutWsOffset = batchIdx * hKv_ * maxKb_ * r_ + headIdx * maxKb_ * r_;

    CopyIn(vFLatOutLocal, vFlatOutWs_[vFlatOutWsOffset], 1, numStemBlocks * r_, 0, 0);

    vFLatOutQue_.EnQue(vFLatOutLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyInKVCache(int32_t batchIdx, int32_t headIdx, int32_t stemBlockIdx,
                                                             int64_t actualRows)
{
    LocalTensor<int8_t> kCacheLocal = kCacheQue_.AllocTensor<int8_t>();
    LocalTensor<int8_t> vCacheLocal = vCacheQue_.AllocTensor<int8_t>();
    LocalTensor<float> kScalarLocal = kScalarBuf_.Get<float>();
    GlobalTensor<int8_t> kCacheGm = kCacheGm_.ReinterpretCast<int8_t>();
    GlobalTensor<int8_t> vCacheGm = vCacheGm_.ReinterpretCast<int8_t>();
    EventMsg<HardEvent::V_S>();
    for (int64_t i = 0; i < stemBlockSize_; i++) {
        int64_t kvIndicesIdx = (stemBlockIdx * stemBlockSize_ + i) / kvBlockSize_;
        int64_t kvBlockSizeIdx = (stemBlockIdx * stemBlockSize_ + i) % kvBlockSize_;
        int64_t logicIdx = kvIndicesGm_.GetValue(batchIdx * maxKvBlocks_ + kvIndicesIdx);
        int64_t kCacheOffset =
            logicIdx * kCacheStride_[0] + headIdx * kCacheStride_[1] + kvBlockSizeIdx * kCacheStride_[2];
        int64_t vCacheOffset =
            logicIdx * vCacheStride_[0] + headIdx * vCacheStride_[1] + kvBlockSizeIdx * vCacheStride_[2];
        int64_t kScaleCacheOffset =
            logicIdx * kScaleCacheStride_[0] + headIdx * kScaleCacheStride_[1] + kvBlockSizeIdx * kScaleCacheStride_[2];
        if ((stemBlockIdx * stemBlockSize_ + i) < actualRows) {
            CopyIn(kCacheLocal[i * dimQk_], kCacheGm[kCacheOffset], 1, dimQk_, 0, 0);
            CopyIn(vCacheLocal[i * dimV_], vCacheGm[vCacheOffset], 1, dimV_, 0, 0);
            kScalarLocal.SetValue(i, kScaleCacheGm_.GetValue(kScaleCacheOffset));
        } else {
            Duplicate(kCacheLocal[i * dimQk_], int8_t(0), dimQk_);
            Duplicate(vCacheLocal[i * dimV_], int8_t(0), dimV_);
            kScalarLocal.SetValue(i, 0);
        }
    }
    EventMsg<HardEvent::S_V>();
    kCacheQue_.EnQue(kCacheLocal);
    vCacheQue_.EnQue(vCacheLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyOutKFlat(int32_t batchIdx, int32_t headIdx, int32_t stemBlocksIdx)
{
    LocalTensor<bfloat16_t> groupSumLocal = groupSumQue_.DeQue<bfloat16_t>();
    int64_t kFlatGmOffset =
        batchIdx * hKv_ * maxKb_ * kflatDim_ + headIdx * maxKb_ * kflatDim_ + stemBlocksIdx * kflatDim_;
    CopyOut(kFlatGm_[kFlatGmOffset], groupSumLocal, 1, stemStride_ * dimQk_, 0, 0);
    groupSumQue_.FreeTensor(groupSumLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyOutVFlat(int32_t batchIdx, int32_t headIdx, int32_t stemBlocksIdx)
{
    LocalTensor<float> vNormDownLocal = vNormDownQue_.DeQue<float>();
    int64_t vFlatOutOffset = batchIdx * hKv_ * maxKb_ * r_ + headIdx * maxKb_ * r_ + stemBlocksIdx * r_;

    CopyOut(vFlatOutWs_[vFlatOutOffset], vNormDownLocal, 1, r_, 0, 0);

    vNormDownQue_.FreeTensor(vNormDownLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyOutVBias(int32_t batchIdx, int32_t headIdx, int64_t numStemBlocks)
{
    LocalTensor<float> vBiasOutLocal = vBiasOutQue_.DeQue<float>();

    auto vBiasOffset = batchIdx * hKv_ * maxKb_ + headIdx * maxKb_;
    CopyOut(vBiasGm_[vBiasOffset], vBiasOutLocal, 1, numStemBlocks, 0, 0);

    vBiasOutQue_.FreeTensor(vBiasOutLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeKFlat(void)
{
    LocalTensor<bfloat16_t> kGropSumLocal = groupSumQue_.AllocTensor<bfloat16_t>();
    LocalTensor<fp8_e4m3fn_t> kCacheLocal = kCacheQue_.DeQue<fp8_e4m3fn_t>();
    LocalTensor<float> kScalarLocal = kScalarBuf_.Get<float>();
    __local_mem__ fp8_e4m3fn_t *kCacheAddr = (__ubuf__ fp8_e4m3fn_t *)kCacheLocal.GetPhyAddr();
    __local_mem__ bfloat16_t *kGropSumAddr = (__ubuf__ bfloat16_t *)kGropSumLocal.GetPhyAddr();
    __local_mem__ float *kScalarAddr = (__ubuf__ float *)kScalarLocal.GetPhyAddr();

    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (dimQk_ + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<fp8_e4m3fn_t> kCacheReg;
        AscendC::Reg::RegTensor<float> kCacheFp32Reg;
        AscendC::Reg::RegTensor<float> kHReg;
        AscendC::Reg::RegTensor<float> kGropSumReg;
        AscendC::Reg::RegTensor<bfloat16_t> kFlatOutReg;
        AscendC::Reg::MaskReg valueMaskReg;
        for (uint16_t i = 0; i < static_cast<uint16_t>(stemStride_); i++) {
            uint32_t maskLen = static_cast<uint32_t>(dimQk_);
            for (uint16_t j = 0; j < loopCnt; j++) {
                auto kGropSumAddrStart = kGropSumAddr + (stemStride_ - i - 1) * dimQk_ + j * vfLen;
                valueMaskReg = AscendC::Reg::UpdateMask<float>(maskLen);
                Reg::Duplicate(kGropSumReg, float(0), valueMaskReg);
                for (uint16_t k = 0; k < r_; k++) {
                    auto kCacheAddrStart = kCacheAddr + k * stemStride_ * dimQk_ + i * dimQk_ + j * vfLen;
                    float kScalarValue = *(kScalarAddr + k * stemStride_ + i);
                    AscendC::Reg::LoadAlign<fp8_e4m3fn_t, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(kCacheReg,
                                                                                                   kCacheAddrStart);
                    AscendC::Reg::Cast<float, fp8_e4m3fn_t, castTraitFp8ToFloat>(kCacheFp32Reg, kCacheReg,
                                                                                 valueMaskReg);
                    AscendC::Reg::Muls(kHReg, kCacheFp32Reg, kScalarValue, valueMaskReg);
                    AscendC::Reg::Add(kGropSumReg, kGropSumReg, kHReg, valueMaskReg);
                }
                AscendC::Reg::Cast<bfloat16_t, float, castTraitFloatTo16>(kFlatOutReg, kGropSumReg, valueMaskReg);
                AscendC::Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(kGropSumAddrStart, kFlatOutReg,
                                                                                    valueMaskReg);
            }
        }
    }
    groupSumQue_.EnQue(kGropSumLocal);
    kCacheQue_.FreeTensor(kCacheLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVFlat(int64_t headIdx, int64_t stemBlockIdx, int32_t kvLen)
{
    LocalTensor<fp8_e4m3fn_t> vCacheLocal = vCacheQue_.DeQue<fp8_e4m3fn_t>();
    LocalTensor<float> vNormDownLocal = vNormDownQue_.AllocTensor<float>();
    float vScaleH = vScaleGm_.GetValue(headIdx);
    EventMsg<HardEvent::S_V>();
    int64_t rowIdsStart = stemBlockIdx * stemBlockSize_;

    __local_mem__ fp8_e4m3fn_t *vCacheAddr = (__ubuf__ fp8_e4m3fn_t *)vCacheLocal.GetPhyAddr();
    __local_mem__ float *normsAddr = (__ubuf__ float *)vNormDownLocal.GetPhyAddr();

    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (dimV_ + vfLen - 1) / vfLen;
    uint16_t loopCntWhere = (r_ * stemStride_ + vfLen - 1) / vfLen;
    uint16_t loopCntMax = (stemStride_ + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<fp8_e4m3fn_t> vCacheReg;
        AscendC::Reg::RegTensor<float> vCacheFp32Reg;
        AscendC::Reg::RegTensor<float> vHReg;
        AscendC::Reg::RegTensor<float> vH2Reg;
        AscendC::Reg::RegTensor<float> vRowsSqrtReg;
        AscendC::Reg::RegTensor<float> vHSumReg;
        AscendC::Reg::RegTensor<float> normsReg;
        AscendC::Reg::RegTensor<float> normsWhereReg;
        AscendC::Reg::RegTensor<float> normsMaxReg;
        AscendC::Reg::RegTensor<int32_t> rowIdxsReg;
        AscendC::Reg::RegTensor<float> zerosReg;
        AscendC::Reg::MaskReg valueMaskReg;
        AscendC::Reg::MaskReg whereMaskReg;
        AscendC::Reg::MaskReg maxMaskReg;
        AscendC::Reg::MaskReg oneMaskReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::MaskReg allMaskReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        Reg::Duplicate(zerosReg, float(0), allMaskReg);
        auto normsAddrStart = normsAddr;
        for (uint16_t i = 0; i < static_cast<uint16_t>(r_ * stemStride_); i++) {
            uint32_t maskLen = static_cast<uint32_t>(dimV_);
            Reg::Duplicate(normsReg, float(0), oneMaskReg);
            for (uint16_t j = 0; j < loopCnt; j++) {
                valueMaskReg = AscendC::Reg::UpdateMask<float>(maskLen);
                auto vCacheAddrStart = vCacheAddr + i * dimV_ + j * vfLen;
                AscendC::Reg::LoadAlign<fp8_e4m3fn_t, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(vCacheReg,
                                                                                               vCacheAddrStart);
                AscendC::Reg::Cast<float, fp8_e4m3fn_t, castTraitFp8ToFloat>(vCacheFp32Reg, vCacheReg, valueMaskReg);
                AscendC::Reg::Muls(vHReg, vCacheFp32Reg, vScaleH, valueMaskReg);
                AscendC::Reg::Mul(vH2Reg, vHReg, vHReg, valueMaskReg);
                AscendC::Reg::Reduce<Reg::ReduceType::SUM, float, float>(vHSumReg, vH2Reg, valueMaskReg);
                AscendC::Reg::Add(normsReg, normsReg, vHSumReg, oneMaskReg);
            }
            AscendC::Reg::Sqrt(vRowsSqrtReg, normsReg, valueMaskReg);
            AscendC::Reg::Store(normsAddrStart + i, vRowsSqrtReg, 1);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        normsAddrStart = normsAddr;
        uint32_t maskLen = static_cast<uint32_t>(r_ * stemStride_);
        for (uint16_t i = 0; i < loopCntWhere; i++) {
            valueMaskReg = AscendC::Reg::UpdateMask<float>(maskLen);
            AscendC::Reg::LoadAlign(normsReg, normsAddrStart + i * vfLen);
            AscendC::Reg::Arange(rowIdxsReg, rowIdsStart + i * vfLen);
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::LT>(whereMaskReg, rowIdxsReg, kvLen, valueMaskReg);
            AscendC::Reg::Select(normsWhereReg, normsReg, zerosReg, whereMaskReg);
            AscendC::Reg::StoreAlign(normsAddrStart + i * vfLen, normsWhereReg, valueMaskReg);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        auto vNormDownAddr = normsAddr;
        for (uint16_t i = 0; i < static_cast<uint16_t>(r_); i++) {
            uint32_t maskLen = static_cast<uint32_t>(stemStride_);
            Reg::Duplicate(normsMaxReg, float(FLT_MIN), allMaskReg);
            for (uint16_t j = 0; j < loopCntMax; j++) {
                valueMaskReg = AscendC::Reg::UpdateMask<float>(maskLen);
                auto normsWhereAddrStart = normsAddr + i * stemStride_ + j * vfLen;
                AscendC::Reg::LoadAlign(normsReg, normsWhereAddrStart);
                AscendC::Reg::Reduce<Reg::ReduceType::MAX, float, float>(normsReg, normsReg, valueMaskReg);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(maxMaskReg, normsMaxReg, normsReg, oneMaskReg);
                AscendC::Reg::Select(normsMaxReg, normsMaxReg, normsReg, maxMaskReg);
            }
            AscendC::Reg::Store(vNormDownAddr + i, normsMaxReg, 1);
        }
    }
    vNormDownQue_.EnQue(vNormDownLocal);
    vCacheQue_.FreeTensor(vCacheLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeLogVals(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                                              __local_mem__ float *vFLatOutAddr)
{
    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (numStemBlocks * r_ + vfLen - 1) / vfLen;
    float divNum = 1 / float(numStemBlocks * r_);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vFLatOutReg;
        AscendC::Reg::RegTensor<float> logValsReg;
        AscendC::Reg::RegTensor<float> logValsSumReg;
        AscendC::Reg::RegTensor<float> vMeanReg;
        AscendC::Reg::MaskReg valueMaskReg;
        AscendC::Reg::MaskReg oneMaskReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        Reg::Duplicate(vMeanReg, float(0), oneMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);
        auto logValsAddrStart = vFLatOutAddr;
        auto vMeanAddrStart = vBiasOutAddr;
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::Reg::UpdateMask<float>(maskLen);
            AscendC::Reg::AddrReg vFLatOutAddrOfst = AscendC::Reg::CreateAddrReg<float>(i, vfLen);
            AscendC::Reg::LoadAlign(vFLatOutReg, vFLatOutAddr, vFLatOutAddrOfst);
            AscendC::Reg::Adds(vFLatOutReg, vFLatOutReg, EPSILON, valueMaskReg);
            AscendC::Reg::Log(logValsReg, vFLatOutReg, valueMaskReg);
            AscendC::Reg::StoreAlign(logValsAddrStart + i * vfLen, logValsReg, valueMaskReg);
        }
    }
}

template <bool KDownLenFlag>
__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVStd(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                                           __local_mem__ float *logValsAddr, float vMean)
{
    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (numStemBlocks * r_ + vfLen - 1) / vfLen;
    float divNum = 1 / float(numStemBlocks * r_ - 1);
    float vMeanNum = vMean * (-1);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> sqrtReg;
        auto vStdAddrStart = vBiasOutAddr + 1;
        AscendC::Reg::MaskReg oneMaskReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        if constexpr (KDownLenFlag) {
            AscendC::Reg::RegTensor<float> logValsReg;
            AscendC::Reg::RegTensor<float> subReg;
            AscendC::Reg::RegTensor<float> squareReg;
            AscendC::Reg::RegTensor<float> sumReg;
            AscendC::Reg::MaskReg valueMaskReg;
            Reg::Duplicate(sumReg, float(0), oneMaskReg);
            uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);
            auto logValsAddrStart = logValsAddr;
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::Reg::UpdateMask<float>(maskLen);
                AscendC::Reg::AddrReg logValsAddrOfst = AscendC::Reg::CreateAddrReg<float>(i, vfLen);
                AscendC::Reg::LoadAlign(logValsReg, logValsAddrStart, logValsAddrOfst);
                AscendC::Reg::Adds(subReg, logValsReg, vMeanNum, valueMaskReg);
                AscendC::Reg::Mul(squareReg, subReg, subReg, valueMaskReg);
                AscendC::Reg::Reduce<Reg::ReduceType::SUM, float, float>(squareReg, squareReg, valueMaskReg);
                AscendC::Reg::Add(sumReg, sumReg, squareReg, oneMaskReg);
            }
            AscendC::Reg::Muls(sumReg, sumReg, divNum, oneMaskReg);
            AscendC::Reg::Sqrt(sqrtReg, sumReg, oneMaskReg);
        } else {
            Reg::Duplicate(sqrtReg, float(0), oneMaskReg);
        }
        AscendC::Reg::Store(vStdAddrStart, sqrtReg, 1);
    }
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVBias(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                                            __local_mem__ float *logValsAddr, float vMean, float vSTd)
{
    float vMeanNum = vMean * (-1);
    float invStd = float(1.0) / (vSTd + EPSILON);
    float divNum = 1 / float(r_);

    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (numStemBlocks * r_ + vfLen - 1) / vfLen;
    uint16_t loopRCnt = (r_ + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> logValsReg;
        AscendC::Reg::RegTensor<float> normalizedReg;
        AscendC::Reg::RegTensor<float> reluReg;
        AscendC::Reg::RegTensor<float> vFinalReg;
        AscendC::Reg::RegTensor<float> vFinalBlockReg;
        AscendC::Reg::RegTensor<float> vFinalBlockSumReg;
        AscendC::Reg::RegTensor<float> vFinalBlockMeanReg;
        AscendC::Reg::UnalignReg u0;
        AscendC::Reg::MaskReg valueMaskReg;
        AscendC::Reg::MaskReg oneMaskReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);

        auto logValsAddrStart = logValsAddr;
        auto vFinalAddrStart = logValsAddr;
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::Reg::UpdateMask<float>(maskLen);
            AscendC::Reg::AddrReg vFLatOutAddrOfst = AscendC::Reg::CreateAddrReg<float>(i, vfLen);
            AscendC::Reg::LoadAlign(logValsReg, logValsAddrStart, vFLatOutAddrOfst);
            AscendC::Reg::Adds(normalizedReg, logValsReg, vMeanNum, valueMaskReg);
            AscendC::Reg::Muls(normalizedReg, normalizedReg, invStd, valueMaskReg);
            AscendC::Reg::Relu(reluReg, normalizedReg, valueMaskReg);
            AscendC::Reg::Muls(vFinalReg, reluReg, lambdaMag_, valueMaskReg);
            AscendC::Reg::StoreAlign(vFinalAddrStart + i * vfLen, vFinalReg, valueMaskReg);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        auto vBiasOutAddrStart = vBiasOutAddr;
        for (uint16_t i = 0; i < static_cast<uint16_t>(numStemBlocks); i++) {
            uint32_t maskLen = static_cast<uint32_t>(r_);
            Reg::Duplicate(vFinalBlockSumReg, float(0), oneMaskReg);
            for (uint16_t j = 0; j < loopRCnt; j++) {
                valueMaskReg = AscendC::Reg::UpdateMask<float>(maskLen);
                auto vFinalBlockAddrStart = logValsAddr + i * r_ + j * vfLen;
                AscendC::Reg::LoadUnAlignPre(u0, vFinalBlockAddrStart);
                AscendC::Reg::LoadUnAlign(vFinalBlockReg, u0, vFinalBlockAddrStart);
                AscendC::Reg::Reduce<Reg::ReduceType::SUM, float, float>(vFinalBlockReg, vFinalBlockReg, valueMaskReg);
                AscendC::Reg::Add(vFinalBlockSumReg, vFinalBlockSumReg, vFinalBlockReg, oneMaskReg);
            }
            AscendC::Reg::Muls(vFinalBlockMeanReg, vFinalBlockSumReg, divNum, oneMaskReg);
            AscendC::Reg::Store(vBiasOutAddrStart + i, vFinalBlockMeanReg, 1);
        }
    }
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVBiasOut(int32_t batchIdx, int32_t headIdx, int64_t kDownLen,
                                                               int64_t numStemBlocks)
{
    LocalTensor<float> vBiasOutLocal = vBiasOutQue_.AllocTensor<float>();
    LocalTensor<float> vFLatOutLocal = vFLatOutQue_.DeQue<float>();
    LocalTensor<uint8_t> meanTempLocal = meanTempBuf_.Get<uint8_t>();

    __local_mem__ float *vBiasOutAddr = (__ubuf__ float *)vBiasOutLocal.GetPhyAddr();
    __local_mem__ float *vFLatOutAddr = (__ubuf__ float *)vFLatOutLocal.GetPhyAddr();

    ComputeLogVals(numStemBlocks, vBiasOutAddr, vFLatOutAddr);
    uint32_t n = static_cast<uint32_t>(numStemBlocks * r_);
    AscendC::MeanParams meanPara;
    meanPara.outter = 1;
    meanPara.n = n;
    meanPara.inner =
        static_cast<uint32_t>(Ops::Base::CeilAlign(uint32_t(n * sizeof(float)), BLOCKSIZE) / sizeof(float));
    AscendC::Mean<float, float>(vBiasOutLocal, vFLatOutLocal, meanTempLocal, meanPara);
    EventMsg<HardEvent::V_S>();
    float vMean = vBiasOutLocal.GetValue(0);
    EventMsg<HardEvent::S_V>();
    if (kDownLen > 1) {
        ComputeVStd<true>(numStemBlocks, vBiasOutAddr, vFLatOutAddr, vMean);
    } else {
        ComputeVStd<false>(numStemBlocks, vBiasOutAddr, vFLatOutAddr, vMean);
    }
    EventMsg<HardEvent::V_S>();
    float vSTd = vBiasOutLocal.GetValue(1);
    EventMsg<HardEvent::S_V>();
    ComputeVBias(numStemBlocks, vBiasOutAddr, vFLatOutAddr, vMean, vSTd);
    vBiasOutQue_.EnQue(vBiasOutLocal);
    vFLatOutQue_.FreeTensor(vFLatOutLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::ProcessVBias(void)
{
    pipe_->Reset();
    pipe_->InitBuffer(vFLatOutQue_, 1, maxKb_ * r_ * sizeof(float));
    pipe_->InitBuffer(vBiasOutQue_, 1, maxKb_ * sizeof(float));
    pipe_->InitBuffer(meanTempBuf_, meanSize_ * sizeof(float));
    for (int32_t bhIdx_ = blockIdx_; bhIdx_ < totalBh_; bhIdx_ += blockNum_) {
        int32_t batchIdx = bhIdx_ / hKv_;
        int32_t headIdx = bhIdx_ % hKv_;
        int32_t kvLen = kvSeqLensGm_.GetValue(batchIdx);
        int64_t kPadded = (static_cast<int64_t>(kvLen) + stemBlockSize_ - 1) / stemBlockSize_ * stemBlockSize_;
        int64_t numStemBlocks = kPadded / stemBlockSize_;
        int64_t kDownLen = kPadded / stemStride_;
        if (kDownLen <= 0) {
            continue;
        }
        CopyInVFlatOut(batchIdx, headIdx, numStemBlocks);
        ComputeVBiasOut(batchIdx, headIdx, kDownLen, numStemBlocks);
        CopyOutVBias(batchIdx, headIdx, numStemBlocks);
    }
}

__aicore__ inline void StemOamPrepPagedKvSimd::ProcessKVFlat(int32_t batchIdx, int32_t headIdx, int32_t stemBlockIdx)
{
    if (blockIdx_ >= kvUsedCoreNum_) {
        return;
    }
    int32_t kvLen = kvSeqLensGm_.GetValue(batchIdx);
    int64_t kPadded = (static_cast<int64_t>(kvLen) + stemBlockSize_ - 1) / stemBlockSize_ * stemBlockSize_;
    int64_t numKVBlocks = (static_cast<int64_t>(kvLen) + kvBlockSize_ - 1) / kvBlockSize_;
    int64_t actualRows = min((int64_t)kvLen, (int64_t)numKVBlocks * kvBlockSize_);
    CopyInKVCache(batchIdx, headIdx, stemBlockIdx, actualRows);
    ComputeKFlat();
    ComputeVFlat(headIdx, stemBlockIdx, kvLen);
    CopyOutKFlat(batchIdx, headIdx, stemBlockIdx);
    CopyOutVFlat(batchIdx, headIdx, stemBlockIdx);
}

__aicore__ inline void StemOamPrepPagedKvSimd::Process(void)
{
    ComputTotalStemBlockAndProcessKVFlat();
    SyncAll();
    ProcessVBias();
}

#endif //  STEM_OAM_PREP_PAGED_KV_SIMD_H
