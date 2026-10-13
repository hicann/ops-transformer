/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

// Match the host serializer: scalar fields and nested tiling structs use
// alignment up to 8 bytes. In particular, workspaceOffset starts at byte 96.
#pragma pack(push, 8)
struct alignas(8) ChunkGatedDeltaRuleComputeWyTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t numChunks;
    int64_t groupSize;
    int64_t totalTasks;
    uint32_t localWorkspaceSize;
    uint32_t perCoreWorkspaceBytes;
    uint32_t usedCoreNum;
    uint64_t workspaceOffset;
    TCubeTiling mmAttn;
    TCubeTiling mmSquare;
    TCubeTiling mmApplyU;
    TCubeTiling mmApplyW;
};
#pragma pack(pop)

static_assert(__builtin_offsetof(ChunkGatedDeltaRuleComputeWyTilingData, workspaceOffset) == 96,
              "workspaceOffset must match the host tiling layout");

static_assert(__builtin_offsetof(ChunkGatedDeltaRuleComputeWyTilingData, mmAttn) == 104,
              "mmAttn must match the host tiling layout");

static_assert(__builtin_offsetof(ChunkGatedDeltaRuleComputeWyTilingData, mmSquare) ==
                  ((104 + sizeof(TCubeTiling) + 7) / 8) * 8,
              "mmSquare must match the host tiling layout");

static_assert(__builtin_offsetof(ChunkGatedDeltaRuleComputeWyTilingData, mmApplyU) ==
                  ((104 + sizeof(TCubeTiling) + 7) / 8) * 8 + ((sizeof(TCubeTiling) + 7) / 8) * 8,
              "mmApplyU must match the host tiling layout");

static_assert(__builtin_offsetof(ChunkGatedDeltaRuleComputeWyTilingData, mmApplyW) ==
                  ((104 + sizeof(TCubeTiling) + 7) / 8) * 8 + 2 * ((sizeof(TCubeTiling) + 7) / 8) * 8,
              "mmApplyW must match the host tiling layout");

#endif // CHUNK_GATED_DELTA_RULE_COMPUTE_WY_TILING_DATA_H
