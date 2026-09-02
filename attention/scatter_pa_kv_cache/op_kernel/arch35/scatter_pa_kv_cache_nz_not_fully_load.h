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
 * \file scatter_pa_kv_cache_nz_not_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_NZ_NOT_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_NZ_NOT_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_arch35_common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheNzNotFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheNzNotFullyLoad(TPipe *pipe, const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out,
                                GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyToCacheNzK(uint64_t start, uint64_t cacheStart);
    __aicore__ inline void CopyToCacheNzV(uint64_t start, uint64_t cacheStart);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> inputQueue_;
    const ScatterPaKvCacheTilingData *tilingData_;

    GlobalTensor<T1> keyGm_;
    GlobalTensor<T2> valueGm_;
    GlobalTensor<T1> keyCacheOutGm_;
    GlobalTensor<T2> valueCacheOutGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    int64_t blockIdx_{0};
    uint64_t tokenSizeK_{0};
    uint64_t tokenSizeV_{0};
    int32_t lastDimK_{0};
    int32_t lastDimV_{0};
    uint64_t ubSizeK_{0};
    uint64_t ubSizeV_{0};
};

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNzNotFullyLoad<T1, T2, IndexDtype, InOutMode>::Init(
    GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out, GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    tokenSizeK_ = tilingData_->numHead * tilingData_->kHeadSize;
    tokenSizeV_ = tilingData_->numHead * tilingData_->vHeadSize;

    lastDimK_ = BLOCK_SIZE / sizeof(T1);
    lastDimV_ = BLOCK_SIZE / sizeof(T2);

    keyGm_.SetGlobalBuffer((__gm__ T1 *)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping));
    keyCacheOutGm_.SetGlobalBuffer((__gm__ T1 *)(key_cache_out));
    if constexpr (InOutMode == DUAL_IN_OUT) {
        valueGm_.SetGlobalBuffer((__gm__ T2 *)(value));
        valueCacheOutGm_.SetGlobalBuffer((__gm__ T2 *)(value_cache_out));
    }

    ubSizeK_ = tilingData_->kHandleNumPerLoop * sizeof(T1);
    ubSizeV_ = tilingData_->vHandleNumPerLoop * sizeof(T2);
    int64_t maxHandleSize = ubSizeK_ > ubSizeV_ ? ubSizeK_ : ubSizeV_;
    pipe_->InitBuffer(inputQueue_, DOUBLE_BUFFER, maxHandleSize);
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNzNotFullyLoad<T1, T2, IndexDtype, InOutMode>::CopyToCacheNzK(
    uint64_t start, uint64_t cacheStart)
{
    DataCopyExtParams copyParamsIn{1, static_cast<uint32_t>(ubSizeK_), 0, 0, 0};
    DataCopyExtParams copyParamsOut{static_cast<uint16_t>(ubSizeK_ / BLOCK_SIZE), BLOCK_SIZE, 0,
                                    static_cast<int64_t>((tilingData_->blockSize - 1) * BLOCK_SIZE), 0};
    DataCopyPadExtParams<T1> PadParams;
    PadParams.isPad = 0;
    PadParams.leftPadding = 0;
    PadParams.rightPadding = 0;
    PadParams.paddingValue = 0;
    int64_t keyStart = start;
    int64_t keyCacheStart = cacheStart;
    for (uint64_t i = 0; i < tilingData_->kLoopNum; i++) {
        // CopyIn
        LocalTensor<T1> input = inputQueue_.AllocTensor<T1>();
        DataCopyPad(input, keyGm_[keyStart], copyParamsIn, PadParams);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T1>();
        DataCopyPad(keyCacheOutGm_[keyCacheStart], input, copyParamsOut);
        inputQueue_.FreeTensor(input);
        keyStart += (ubSizeK_ / sizeof(T1));
        keyCacheStart += static_cast<uint64_t>((ubSizeK_ / sizeof(T1)) * tilingData_->blockSize);
    }
    if (tilingData_->kTailHandleNum > 0) {
        // CopyIn
        copyParamsIn.blockLen = tilingData_->kTailHandleNum * sizeof(T1);
        copyParamsOut.blockCount = tilingData_->kTailHandleNum / lastDimK_;
        LocalTensor<T1> input = inputQueue_.AllocTensor<T1>();
        DataCopyPad(input, keyGm_[keyStart], copyParamsIn, PadParams);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T1>();
        DataCopyPad(keyCacheOutGm_[keyCacheStart], input, copyParamsOut);
        inputQueue_.FreeTensor(input);
    }
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNzNotFullyLoad<T1, T2, IndexDtype, InOutMode>::CopyToCacheNzV(
    uint64_t start, uint64_t cacheStart)
{
    DataCopyExtParams copyParamsIn{1, static_cast<uint32_t>(ubSizeV_), 0, 0, 0};
    DataCopyExtParams copyParamsOut{static_cast<uint16_t>(ubSizeV_ / BLOCK_SIZE), BLOCK_SIZE, 0,
                                    static_cast<int64_t>((tilingData_->blockSize - 1) * BLOCK_SIZE), 0};
    DataCopyPadExtParams<T2> PadParams;
    PadParams.isPad = 0;
    PadParams.leftPadding = 0;
    PadParams.rightPadding = 0;
    PadParams.paddingValue = 0;
    uint64_t valueStart = start;
    uint64_t valueCacheStart = cacheStart;
    for (uint64_t i = 0; i < tilingData_->vLoopNum; i++) {
        // CopyIn
        LocalTensor<T2> input = inputQueue_.AllocTensor<T2>();
        DataCopyPad(input, valueGm_[valueStart], copyParamsIn, PadParams);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T2>();
        DataCopyPad(valueCacheOutGm_[valueCacheStart], input, copyParamsOut);
        inputQueue_.FreeTensor(input);
        valueStart += (ubSizeV_ / sizeof(T2));
        valueCacheStart += static_cast<uint64_t>((ubSizeV_ / sizeof(T2)) * tilingData_->blockSize);
    }
    if (tilingData_->vTailHandleNum > 0) {
        // CopyIn
        copyParamsIn.blockLen = tilingData_->vTailHandleNum * sizeof(T2);
        copyParamsOut.blockCount = tilingData_->vTailHandleNum / lastDimV_;
        LocalTensor<T2> input = inputQueue_.AllocTensor<T2>();
        DataCopyPad(input, valueGm_[valueStart], copyParamsIn, PadParams);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T2>();
        DataCopyPad(valueCacheOutGm_[valueCacheStart], input, copyParamsOut);
        inputQueue_.FreeTensor(input);
    }
}

template <typename T1, typename T2, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNzNotFullyLoad<T1, T2, IndexDtype, InOutMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    uint32_t startTaskId = blockIdx_ * tilingData_->blockFactor;
    int32_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;

    for (int32_t loopToken = 0; loopToken < curBlockFactor; loopToken++) {
        // The offset of the token's initial address, with a unit of 32B
        int64_t slotValue = slotMappingGm_.GetValue(loopToken + startTaskId);
        if (slotValue < 0) {
            continue;
        }
        uint64_t startK = (loopToken + startTaskId) * tokenSizeK_;
        uint64_t blocksIdx = static_cast<uint64_t>(slotValue) / tilingData_->blockSize;
        uint64_t blocksOffset = static_cast<uint64_t>(slotValue) - tilingData_->blockSize * blocksIdx;
        uint64_t cacheStartK = blocksIdx * tilingData_->keyCacheStride0 + blocksOffset * lastDimK_;

        CopyToCacheNzK(startK, cacheStartK);
        if constexpr (InOutMode == DUAL_IN_OUT) {
            uint64_t startV = (loopToken + startTaskId) * tokenSizeV_;
            uint64_t cacheStartV = blocksIdx * tilingData_->valueCacheStride0 + blocksOffset * lastDimV_;
            CopyToCacheNzV(startV, cacheStartV);
        }
    }
}
} // namespace ScatterPaKvCache

#endif // SCATTER_PA_KV_CACHE_NZ_NOT_FULLY_LOAD_H_
