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
 * \file attention_to_ffn_tiling_base.h
 * \brief Common tiling entry and index mapping for AttentionToFFN.
 */

#ifndef ATTENTION_TO_FFN_TILING_BASE_H
#define ATTENTION_TO_FFN_TILING_BASE_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace MC2Tiling {
struct AttentionToFFNTilingConfig {
    uint32_t contextIndex = 0U;
    uint32_t xIndex = 0U;
    uint32_t sessionIdIndex = 1U;
    uint32_t microBatchIdIndex = 2U;
    uint32_t layerIdIndex = 3U;
    uint32_t expertIdsIndex = 4U;
    uint32_t expertRankTableIndex = 5U;
    uint32_t scalesIndex = 6U;
    uint32_t activeMaskIndex = 7U;

    uint32_t attrGroupIndex = 0U;
    uint32_t attrWorldSizeIndex = 1U;
    uint32_t attrFfnTokenInfoTableShapeIndex = 2U;
    uint32_t attrFfnTokenDataShapeIndex = 3U;
    uint32_t attrAttnTokenInfoTableShapeIndex = 4U;
    uint32_t attrMoeExpertNumIndex = 5U;
    uint32_t attrQuantModeIndex = 6U;
    uint32_t attrSyncFlagIndex = 7U;
    uint32_t attrFfnStartRankIdIndex = 8U;
    uint32_t attrCclBufferSizeIndex = UINT32_MAX;

    bool isMc2Context = false;
    bool allowMxQuantMode = false;
};

ge::graphStatus AttentionToFFNTilingFuncBase(gert::TilingContext *context, const AttentionToFFNTilingConfig &config);
ge::graphStatus AttentionToFFNTilingFunc(gert::TilingContext *context);
} // namespace MC2Tiling

#endif // ATTENTION_TO_FFN_TILING_BASE_H
