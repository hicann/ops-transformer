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
 * \file ffn_to_attention_tiling_base.h
 * \brief Common tiling entry and index mapping for FFNToAttention.
 */

#ifndef FFN_TO_ATTENTION_TILING_BASE_H
#define FFN_TO_ATTENTION_TILING_BASE_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace MC2Tiling {
struct FFNToAttentionTilingConfig {
    uint32_t contextIndex = 0U;
    uint32_t xIndex = 0U;
    uint32_t sessionIdsIndex = 1U;
    uint32_t microBatchIdsIndex = 2U;
    uint32_t tokenIdsIndex = 3U;
    uint32_t expertOffsetsIndex = 4U;
    uint32_t actualTokenNumIndex = 5U;
    uint32_t attnRankTableIndex = 6U;

    uint32_t attrGroupIndex = 0U;
    uint32_t attrWorldSizeIndex = 1U;
    uint32_t attrTokenInfoTableShapeIndex = 2U;
    uint32_t attrTokenDataShapeIndex = 3U;
    uint32_t attrCclBufferSizeIndex = UINT32_MAX;

    bool isMc2Context = false;
};

ge::graphStatus FFNToAttentionTilingFuncBase(gert::TilingContext *context, const FFNToAttentionTilingConfig &config);
ge::graphStatus FFNToAttentionTilingFunc(gert::TilingContext *context);
} // namespace MC2Tiling

#endif // FFN_TO_ATTENTION_TILING_BASE_H
