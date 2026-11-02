/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_ARCH22_LAYOUT_UTILS_HPP
#define GBSA_ARCH22_LAYOUT_UTILS_HPP
#include "kernel_operator.h"
namespace GbsaKernelArch22 {
// 0=TND, 1=BSND, 2=BNSD. Storage capacity and effective length are distinct.
// This adapter lets the existing per-core task cursor handle fixed-capacity batches
// without materializing a prefix tensor or rescanning batches for every task.
struct StoragePrefix {
    AscendC::GlobalTensor<int64_t> tensor;
    uint32_t capacity = 0;
    __aicore__ inline void Init(GM_ADDR address, uint32_t seqCapacity)
    {
        capacity = seqCapacity;
        if (address != nullptr) {
            tensor.SetGlobalBuffer((__gm__ int64_t*)address);
        }
    }
    __aicore__ inline int64_t GetValue(uint32_t batch) const
    {
        return capacity != 0U ? static_cast<int64_t>(batch) * capacity : tensor.GetValue(batch);
    }
};
__aicore__ inline uint64_t TensorOffset(uint32_t layout, uint32_t batch, uint32_t token, uint64_t storageToken,
                                        uint32_t head, uint32_t heads, uint32_t dim, uint32_t capacity)
{
    if (layout == 2U) {
        return ((static_cast<uint64_t>(batch) * heads + head) * capacity + token) * dim;
    }
    return (static_cast<uint64_t>(storageToken) * heads + head) * dim;
}
__aicore__ inline uint64_t LseOffset(uint32_t layout, uint32_t batch, uint32_t token, uint32_t storageToken,
                                     uint32_t head, uint32_t heads, uint32_t capacity)
{
    return TensorOffset(layout == 0U ? 0U : 2U, batch, token, storageToken, head, heads, 1U, capacity);
}
} // namespace GbsaKernelArch22
#endif
