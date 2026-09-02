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
 * \file scatter_pa_kv_cache_nz_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_NZ_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_NZ_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"
namespace ScatterPaKvCache {

using namespace AscendC;

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheNzFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheNzFullyLoad(TPipe *pipe, const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out,
                                GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    int64_t blockIdx_{0};
    uint32_t curBlockFactor_{0};
    uint64_t tokenSizeK_{0};
    uint64_t tokenSizeV_{0};
    int32_t lastDimK_{0};
    int32_t lastDimV_{0};

    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputKeyQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputValueQueue_;

    GlobalTensor<T1> inputKeyGm_;
    GlobalTensor<T2> inputValueGm_;
    GlobalTensor<T1> outputKeyCacheGm_;
    GlobalTensor<T2> outputValueCacheGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    const ScatterPaKvCacheTilingData *tilingData_;
};

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNzFullyLoad<T1, T2, IndexDtype, InOutMode>::Init(
    GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out, GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    curBlockFactor_ =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
    tokenSizeK_ = tilingData_->numHead * tilingData_->kHeadSize;
    tokenSizeV_ = tilingData_->numHead * tilingData_->vHeadSize;
    lastDimK_ = BLOCK_SIZE / sizeof(T1);
    lastDimV_ = BLOCK_SIZE / sizeof(T2);
    inputKeyGm_.SetGlobalBuffer((__gm__ T1 *)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping));
    outputKeyCacheGm_.SetGlobalBuffer((__gm__ T1 *)(key_cache_out));
    pipe_->InitBuffer(inputKeyQueue_, 1, tokenSizeK_ * sizeof(T1));
    if constexpr (InOutMode == DUAL_IN_OUT) {
        inputValueGm_.SetGlobalBuffer((__gm__ T2 *)(value));
        outputValueCacheGm_.SetGlobalBuffer((__gm__ T2 *)(value_cache_out));
        pipe_->InitBuffer(inputValueQueue_, 1, tokenSizeV_ * sizeof(T2));
    }
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNzFullyLoad<T1, T2, IndexDtype, InOutMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    DataCopyExtParams copyParamsInK{1, static_cast<uint32_t>(tokenSizeK_ * sizeof(T1)), 0, 0, 0};
    DataCopyExtParams copyParamsOutK{static_cast<uint16_t>(tokenSizeK_ / lastDimK_), BLOCK_SIZE, 0,
                                     static_cast<int64_t>((tilingData_->blockSize - 1) * BLOCK_SIZE), 0};
    DataCopyExtParams copyParamsInV{1, static_cast<uint32_t>(tokenSizeV_ * sizeof(T2)), 0, 0, 0};
    DataCopyExtParams copyParamsOutV{static_cast<uint16_t>(tokenSizeV_ / lastDimV_), BLOCK_SIZE, 0,
                                     static_cast<int64_t>((tilingData_->blockSize - 1) * BLOCK_SIZE), 0};
    DataCopyPadExtParams<T1> kPadParams;
    kPadParams.isPad = 0;
    kPadParams.leftPadding = 0;
    kPadParams.rightPadding = 0;
    kPadParams.paddingValue = 0;
    uint32_t startTaskId = blockIdx_ * tilingData_->blockFactor;
    for (uint32_t i = 0; i < curBlockFactor_; ++i) {
        int64_t cacheIndex = slotMappingGm_.GetValue(i + startTaskId);
        if (cacheIndex < 0) {
            continue;
        }
        uint64_t startK = (i + startTaskId) * tokenSizeK_;
        uint64_t blocksIdx = static_cast<uint64_t>(cacheIndex) / tilingData_->blockSize;
        uint64_t blocksOffset = static_cast<uint64_t>(cacheIndex) - tilingData_->blockSize * blocksIdx;
        uint64_t cacheStartK = blocksIdx * tilingData_->keyCacheStride0 + blocksOffset * lastDimK_;
        LocalTensor<T1> inputKey = inputKeyQueue_.AllocTensor<T1>();
        DataCopyPad(inputKey, inputKeyGm_[startK], copyParamsInK, kPadParams);
        inputKeyQueue_.EnQue(inputKey);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            DataCopyPadExtParams<T2> vPadParams;
            vPadParams.isPad = 0;
            vPadParams.leftPadding = 0;
            vPadParams.rightPadding = 0;
            vPadParams.paddingValue = 0;
            uint64_t startV = (i + startTaskId) * tokenSizeV_;
            LocalTensor<T2> inputValue = inputValueQueue_.AllocTensor<T2>();
            DataCopyPad(inputValue, inputValueGm_[startV], copyParamsInV, vPadParams);
            inputValueQueue_.EnQue(inputValue);
        }
        inputKey = inputKeyQueue_.DeQue<T1>();
        DataCopyPad(outputKeyCacheGm_[cacheStartK], inputKey, copyParamsOutK);
        inputKeyQueue_.FreeTensor(inputKey);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            uint64_t cacheStartV = blocksIdx * tilingData_->valueCacheStride0 + blocksOffset * lastDimV_;
            LocalTensor<T2> inputValue = inputValueQueue_.DeQue<T2>();
            DataCopyPad(outputValueCacheGm_[cacheStartV], inputValue, copyParamsOutV);
            inputValueQueue_.FreeTensor(inputValue);
        }
    }
}

} // namespace ScatterPaKvCache

#endif // SCATTER_PA_KV_CACHE_NZ_FULLY_LOAD_H_
