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
 * \file kv_quant_sparse_attn_sharedkv_scfa_ub_config.h
 * \brief SCFA UB buffer configuration shared by host tiling and kernel.
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_UB_CONFIG_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_UB_CONFIG_H

#include <cstdint>

namespace KvQuantSparseAttnSharedkv {
namespace ScfaUbConfig {

constexpr uint32_t S1_BASE_SIZE = 64U;
constexpr uint32_t D_TEMPLATE_SIZE = 512U;
constexpr uint32_t D_INPUT_SIZE = 640U;
constexpr uint32_t V0_PROCESS_ROW_NUM = 16U;
constexpr uint32_t CV_RATIO = 2U;
constexpr uint32_t UB_ALIGN_SIZE = 64U;

constexpr uint32_t D_TEMPLATE_ALIGN64 = (D_TEMPLATE_SIZE + UB_ALIGN_SIZE - 1U) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;

template <typename T>
struct TypedBufferSize {
    static constexpr uint32_t STAGE0_OUT = D_INPUT_SIZE * (V0_PROCESS_ROW_NUM + 1U) * static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t STAGE2_OUT =
        (S1_BASE_SIZE / CV_RATIO) * D_TEMPLATE_ALIGN64 * static_cast<uint32_t>(sizeof(T));
};

} // namespace ScfaUbConfig
} // namespace KvQuantSparseAttnSharedkv

#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_UB_CONFIG_H
