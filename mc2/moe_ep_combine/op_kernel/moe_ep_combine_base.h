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
 * \file moe_ep_combine_base.h
 * \brief
 */

#ifndef MOE_EP_COMBINE_BASE_H
#define MOE_EP_COMBINE_BASE_H
#include <cstdint>

#include "../../common/op_kernel/mc2_moe_context.h"

namespace MoeEpCombineLayout {

// Each received token has five int32 metadata fields.
constexpr uint32_t RECV_META_FIELDS = 5U;
constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;

// Metadata batches are measured in tokens, independently of Hcomm's WQEBB capacity.
constexpr uint32_t METADATA_BATCH_TOKENS = 256U;
constexpr uint32_t METADATA_BATCH_ELEMS = METADATA_BATCH_TOKENS * RECV_META_FIELDS;

// One address buffer contains SoA regions of uint64 byte offsets, each padded to 256 tokens.
// The weight region is allocated and used only when HasTopkWeight == 1.
constexpr uint32_t X_OFFSET_REGION = METADATA_BATCH_TOKENS;
constexpr uint32_t WEIGHT_OFFSET_REGION = METADATA_BATCH_TOKENS * 2U;

} // namespace MoeEpCombineLayout

#endif // MOE_EP_COMBINE_BASE_H
