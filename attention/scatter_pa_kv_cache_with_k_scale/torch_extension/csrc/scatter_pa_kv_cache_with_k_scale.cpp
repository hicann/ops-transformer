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
 * \file scatter_pa_kv_cache_with_k_scale.cpp
 * \brief torch_npu适配：scatter_pa_kv_cache_with_k_scale算子的C++绑定
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

/**
 * scatter_pa_kv_cache_with_k_scale算子的C++实现
 *
 * 功能：将FP8格式的key/value和对应的key_scale更新到KV cache中（原地更新）
 *
 * @param key [num_tokens, num_head, k_head_size] FP8格式的key输入
 * @param value [num_tokens, num_head, v_head_size] FP8格式的value输入
 * @param keyCache [num_blocks, num_head, block_size, k_head_size] FP8格式的key cache（原地更新）
 * @param valueCache [num_blocks, num_head, block_size, v_head_size] FP8格式的value cache（原地更新）
 * @param slotMapping [num_tokens] int32/int64，每个token在cache中的偏移
 * @param keyScale [num_tokens, num_head] float32格式的key scale输入
 * @param keyScaleCache [num_blocks, num_head, block_size, 1] float32格式的key scale cache（原地更新）
 * @param cacheLayout cache布局格式字符串（默认"BNBD"）
 */
void ScatterPaKvCacheWithKScale(const at::Tensor &key, const at::Tensor &value, at::Tensor &keyCache,
                                at::Tensor &valueCache, const at::Tensor &slotMapping, const at::Tensor &keyScale,
                                at::Tensor &keyScaleCache, std::string cacheLayout)
{
    // 检查输入tensor维度
    TORCH_CHECK(key.dim() == 3, "key must be 3D tensor [num_tokens, num_head, k_head_size]");
    TORCH_CHECK(value.dim() == 3, "value must be 3D tensor [num_tokens, num_head, v_head_size]");
    TORCH_CHECK(keyCache.dim() == 4, "key_cache must be 4D tensor [num_blocks, num_head, block_size, k_head_size]");
    TORCH_CHECK(valueCache.dim() == 4, "value_cache must be 4D tensor [num_blocks, num_head, block_size, v_head_size]");
    TORCH_CHECK(slotMapping.dim() == 1, "slot_mapping must be 1D tensor [num_tokens]");
    TORCH_CHECK(keyScale.dim() == 2, "key_scale must be 2D tensor [num_tokens, num_head]");
    TORCH_CHECK(keyScaleCache.dim() == 4, "key_scale_cache must be 4D tensor [num_blocks, num_head, block_size, 1]");

    // 检查dtype一致性
    TORCH_CHECK(key.dtype() == value.dtype(), "key and value must have same dtype");
    TORCH_CHECK(key.dtype() == keyCache.dtype(), "key and key_cache must have same dtype");
    TORCH_CHECK(key.dtype() == valueCache.dtype(), "key and value_cache must have same dtype");
    TORCH_CHECK(key.dtype() == at::kFloat8_e5m2 || key.dtype() == at::kFloat8_e4m3fn,
                "key dtype must be float8_e5m2 or float8_e4m3fn");
    TORCH_CHECK(keyScale.dtype() == at::kFloat, "key_scale must be float32");
    TORCH_CHECK(keyScaleCache.dtype() == at::kFloat, "key_scale_cache must be float32");
    TORCH_CHECK(slotMapping.dtype() == at::kInt || slotMapping.dtype() == at::kLong,
                "slot_mapping dtype must be int32 or int64");

    // 转换cache_layout为char指针
    char *cacheLayoutPtr = const_cast<char *>(cacheLayout.c_str());

    // 调用ACLNN算子（inplace更新输入cache）
    ACLNN_CMD(aclnnScatterPaKvCacheWithKScale, key, value, keyCache, valueCache, slotMapping, keyScale, keyScaleCache,
              cacheLayoutPtr);
}

// 绑定C++函数到Python模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("scatter_pa_kv_cache_with_k_scale", &ScatterPaKvCacheWithKScale,
          "scatter_pa_kv_cache_with_k_scale operator for torch_npu");
}

} // namespace op_api
