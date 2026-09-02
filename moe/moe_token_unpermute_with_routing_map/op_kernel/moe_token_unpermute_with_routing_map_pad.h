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
 * \file moe_token_unpermute_with_routing_map_pad.h
 * \brief
 */

#ifndef MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_PAD_H
#define MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_PAD_H

#include "kernel_operator.h"
using namespace AscendC;

namespace MoeTokenUnpermuteWithRoutingMap {

template <typename T>
class KernelMoeTokenUnpermuteWithRoutingMapPad {
public:
    __aicore__ inline KernelMoeTokenUnpermuteWithRoutingMapPad(){};
    __aicore__ inline void Init(GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens,
                                const MoeTokenUnpermuteWithRoutingMapTilingData &tiling_data);
    __aicore__ inline void InitData(const MoeTokenUnpermuteWithRoutingMapTilingData &tiling_data);
    __aicore__ inline void Process();
    __aicore__ inline void Compute(uint64_t index, uint64_t loopN);

protected:
    static constexpr uint64_t BLOCK_SIZE = 32;
    static constexpr uint64_t BUFFER_NUM = 1;

    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> unpermutedTokensOutQue;
    GlobalTensor<int32_t> sortedIndicesGM;
    GlobalTensor<T> probsGM, unpermutedTokensGM;

    DataCopyPadExtParams<int32_t> extParamsIn{false, 0, 0, 0};
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};

    uint64_t blockIdx_ = 0;
    uint64_t core_num_use = 0;
    uint64_t num_tokens = 0;
    uint64_t num_experts = 0;
    uint64_t capacity = 0;
    uint64_t front_core = 0;
    uint64_t loop_time_each_front_core = 0;
    uint64_t num_tokens_each_front_core = 0;
    uint64_t num_tokens_front_core_each_loop = 0;
    uint64_t num_tokens_front_core_last_loop = 0;
    uint64_t tail_core = 0;
    uint64_t loop_time_each_tail_core = 0;
    uint64_t num_tokens_each_tail_core = 0;
    uint64_t num_tokens_tail_core_each_loop = 0;
    uint64_t num_tokens_tail_core_last_loop = 0;

    uint64_t loop_time_current_core = 0;
    uint64_t num_tokens_each_loop_current_core = 0;
    uint64_t num_tokens_last_loop_current_core = 0;
    uint64_t blockOffset = 0;
};

template <typename T>
__aicore__ inline void KernelMoeTokenUnpermuteWithRoutingMapPad<T>::Init(
    GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens,
    const MoeTokenUnpermuteWithRoutingMapTilingData &tilingData)
{
    InitData(tilingData);
    if (this->blockIdx_ < this->front_core) {
        blockOffset = this->num_tokens_each_front_core * this->blockIdx_;
    } else {
        blockOffset = this->num_tokens_each_front_core * (this->front_core) +
                      (this->blockIdx_ - this->front_core) * this->num_tokens_each_tail_core;
    }

    sortedIndicesGM.SetGlobalBuffer((__gm__ int32_t *)sorted_indices);
    probsGM.SetGlobalBuffer((__gm__ T *)probs);
    unpermutedTokensGM.SetGlobalBuffer((__gm__ T *)unpermuted_tokens + blockOffset);

    this->pipe.InitBuffer(unpermutedTokensOutQue, BUFFER_NUM, num_tokens_each_loop_current_core * sizeof(T));
}

template <typename T>
__aicore__ inline void KernelMoeTokenUnpermuteWithRoutingMapPad<T>::InitData(
    const MoeTokenUnpermuteWithRoutingMapTilingData &tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    blockIdx_ = AscendC::GetBlockIdx();
    const MoeTokenUnpermuteWithRoutingMapPadTilingData &padTiling =
        tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData;
    core_num_use = padTiling.core_num_use;
    num_tokens = padTiling.num_tokens;
    num_experts = padTiling.num_experts;
    capacity = padTiling.capacity;
    front_core = padTiling.front_core;
    loop_time_each_front_core = padTiling.loop_time_each_front_core;
    num_tokens_each_front_core = padTiling.num_tokens_each_front_core;
    num_tokens_front_core_each_loop = padTiling.num_tokens_front_core_each_loop;
    num_tokens_front_core_last_loop = padTiling.num_tokens_front_core_last_loop;
    tail_core = padTiling.tail_core;
    loop_time_each_tail_core = padTiling.loop_time_each_tail_core;
    num_tokens_each_tail_core = padTiling.num_tokens_each_tail_core;
    num_tokens_tail_core_each_loop = padTiling.num_tokens_tail_core_each_loop;
    num_tokens_tail_core_last_loop = padTiling.num_tokens_tail_core_last_loop;
    // 分核设置
    loop_time_current_core =
        (blockIdx_ < front_core) ? padTiling.loop_time_each_front_core : padTiling.loop_time_each_tail_core;
    num_tokens_each_loop_current_core =
        (blockIdx_ < front_core) ? padTiling.num_tokens_front_core_each_loop : padTiling.num_tokens_tail_core_each_loop;
    num_tokens_last_loop_current_core =
        (blockIdx_ < front_core) ? padTiling.num_tokens_front_core_last_loop : padTiling.num_tokens_tail_core_last_loop;
}

template <typename T>
__aicore__ inline void KernelMoeTokenUnpermuteWithRoutingMapPad<T>::Compute(uint64_t index, uint64_t loopN)
{
    int posIdx = 0;
    LocalTensor<T> unpermutedTokensLocal = this->unpermutedTokensOutQue.template AllocTensor<T>();
    for (int offset = 0; offset < loopN; offset++) {
        posIdx = blockOffset + offset;
        unpermutedTokensLocal.SetValue(
            offset, probsGM.GetValue(sortedIndicesGM.GetValue(posIdx) * num_experts + posIdx / capacity));
    }
    this->copyParams.blockLen = loopN * sizeof(T);
    DataCopyPad(unpermutedTokensGM, unpermutedTokensLocal, this->copyParams);
    this->unpermutedTokensOutQue.FreeTensor(unpermutedTokensLocal);
}

template <typename T>
__aicore__ inline void KernelMoeTokenUnpermuteWithRoutingMapPad<T>::Process()
{
    for (uint64_t n = 0; n < this->loop_time_current_core - 1; n++) {
        Compute(n, this->num_tokens_each_loop_current_core);
    }
    if (this->num_tokens_last_loop_current_core == 0) {
        Compute(this->loop_time_current_core - 1, this->num_tokens_each_loop_current_core);
    } else {
        Compute(this->loop_time_current_core - 1, this->num_tokens_last_loop_current_core);
    }
}
} // namespace MoeTokenUnpermuteWithRoutingMap
#endif // namespace MoeTokenUnpermuteWithRoutingMap
