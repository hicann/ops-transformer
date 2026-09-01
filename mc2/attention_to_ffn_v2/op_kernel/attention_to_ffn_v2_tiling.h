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
 * \file attention_to_ffn_v2_tiling.h
 * \brief
 */
#ifndef ATTENTION_TO_FFN_V2_TILING_H
#define ATTENTION_TO_FFN_V2_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct AttentionToFfnV2Info {
    uint32_t X;
    uint32_t BS;
    uint32_t H;
    uint32_t K;
    uint32_t L;
    uint32_t expertNum;       // 所有专家数
    uint32_t moeExpertNum;    // 路由专家数
    uint32_t sharedExpertNum; // 共享专家
    uint32_t HS;
    uint32_t expRankTableM; // expert_rank_table 中最大支持冗余度
    uint32_t microBatchNum;
    uint32_t attentionWorkerNum;
    uint32_t infoTableLastDimNum;
    uint32_t aivNum;
    uint32_t worldSize;
    uint32_t syncFlag;
    uint32_t quantMode;
    uint32_t ffnStartRankId;
    bool isScales;
    bool isActiveMask;
    uint8_t windowType;
    // A5 V2 URMA reuses the V1-only fields without changing the shared tiling layout.
    uint64_t totalUbSize;
    uint64_t urmaWorkspaceOffset;
};

struct AttentionToFfnV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    AttentionToFfnV2Info attentionToFfnV2Info;
};

#endif // ATTENTION_TO_FFN_V2_TILING_H
