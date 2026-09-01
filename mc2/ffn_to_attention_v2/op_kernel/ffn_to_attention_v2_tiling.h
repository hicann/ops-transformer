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
 * \file ffn_to_attention_V2_tiling.h
 * \brief
 */
#ifndef FFN_TO_ATTENTION_V2_TILING_H
#define FFN_TO_ATTENTION_V2_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct FFNToAttentionV2Info {
    uint32_t H;
    uint32_t A;
    uint32_t microBatchNum;
    uint32_t BS;
    uint32_t expertNumPerToken; // 专家总数
    uint32_t HS;
    uint32_t aivNum;
    uint32_t worldSize;
    bool isInputRankTable;
    uint8_t windowType;
    uint64_t maxTokenNum;
    uint64_t urmaWorkspaceOffset;
};

struct FFNToAttentionV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    FFNToAttentionV2Info ffnToAttentionV2Info;
};

#endif // FFN_TO_ATTENTION_V2_TILING_H
