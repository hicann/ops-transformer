/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_EXCEPTION_DUMP_DEFS_H
#define MOE_EP_EXCEPTION_DUMP_DEFS_H

#include <cstddef>
#include <cstdint>

constexpr uint32_t MOE_EP_DUMP_LAYOUT_VERSION = 1U;
constexpr uint32_t MOE_EP_DUMP_REGION_COUNT = 6U;
constexpr uint64_t MOE_EP_DUMP_METADATA_BYTES = 64UL * 1024UL;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_SLOT_BYTES = 512UL;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_SLOT_COUNT = 100UL;
constexpr uint64_t MOE_EP_PER_CORE_DIAG_BYTES = MOE_EP_PER_CORE_DIAG_SLOT_COUNT * MOE_EP_PER_CORE_DIAG_SLOT_BYTES;
constexpr uint64_t MOE_EP_FIXED_PREFIX_BYTES = MOE_EP_DUMP_METADATA_BYTES + MOE_EP_PER_CORE_DIAG_BYTES;
constexpr uint64_t MOE_EP_CORE_DIAG_RECORD_BYTES = 64UL;

enum MoeEpCoreDiagOpIndex : uint32_t {
    MOE_EP_CORE_DIAG_DISPATCH = 0U,
    MOE_EP_CORE_DIAG_DISPATCH_EPILOGUE,
    MOE_EP_CORE_DIAG_COMBINE,
    MOE_EP_CORE_DIAG_COMBINE_EPILOGUE, // Reserved until MoeEpCombineEpilogue exists.
};

enum MoeEpDispatchRunPosition : uint32_t {
    MOE_EP_DISPATCH_RUN_POS_INIT_DONE = 1U,
    MOE_EP_DISPATCH_RUN_POS_COUNT_READY,
    MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE,
};

enum MoeEpDispatchEpilogueRunPosition : uint32_t {
    MOE_EP_DISPATCH_EPILOGUE_RUN_POS_INIT_DONE = 1U,
    MOE_EP_DISPATCH_EPILOGUE_RUN_POS_WAIT_DONE,
    MOE_EP_DISPATCH_EPILOGUE_RUN_POS_OUTPUT_DONE,
};

enum MoeEpCombineRunPosition : uint32_t {
    MOE_EP_COMBINE_RUN_POS_INIT_DONE = 1U,
    MOE_EP_COMBINE_RUN_POS_URMA_REQUESTS_ISSUE_DONE,
};

enum MoeEpCombineEpilogueRunPosition : uint32_t {
    MOE_EP_COMBINE_EPILOGUE_RUN_POS_INIT_DONE = 1U,
    MOE_EP_COMBINE_EPILOGUE_RUN_POS_WAIT_DONE,
    MOE_EP_COMBINE_EPILOGUE_RUN_POS_OUTPUT_DONE,
};

struct MoeEpCoreDiagRecord {
    uint64_t opCnt;
    uint32_t runPosition;
    int32_t epRankId;
    uint32_t aivId;
    uint8_t reserved[44];
};

static_assert(sizeof(MoeEpCoreDiagRecord) == MOE_EP_CORE_DIAG_RECORD_BYTES, "MoeEpCoreDiagRecord must occupy 64 bytes");
static_assert(MOE_EP_CORE_DIAG_RECORD_BYTES * 4UL <= MOE_EP_PER_CORE_DIAG_SLOT_BYTES,
              "Moe EP diagnostic records must fit in one per-core slot");
static_assert(offsetof(MoeEpCoreDiagRecord, opCnt) == 0U, "Unexpected opCnt offset");
static_assert(offsetof(MoeEpCoreDiagRecord, runPosition) == 8U, "Unexpected runPosition offset");
static_assert(offsetof(MoeEpCoreDiagRecord, epRankId) == 12U, "Unexpected epRankId offset");
static_assert(offsetof(MoeEpCoreDiagRecord, aivId) == 16U, "Unexpected aivId offset");
static_assert(offsetof(MoeEpCoreDiagRecord, reserved) == 20U, "Unexpected reserved offset");

enum MoeEpDumpRegionIndex : uint32_t {
    MOE_EP_DUMP_REGION_PER_CORE_DIAG = 0U,
    MOE_EP_DUMP_REGION_COUNT_NOTIFY,
    MOE_EP_DUMP_REGION_EXPERT_COUNT,
    MOE_EP_DUMP_REGION_DISPATCH_SLOT_STATE,
    MOE_EP_DUMP_REGION_COMBINE_TOKEN_STATE,
    MOE_EP_DUMP_REGION_HYBRID_SCALEOUT_STATUS,
};

struct MoeEpDumpRegion {
    uint64_t offset;
    uint64_t size;
};

struct MoeEpDumpMetadata {
    uint32_t layoutVersion;
    uint32_t nmt;
    uint32_t topK;
    uint32_t hidden;
    uint32_t epWorldSize;
    uint32_t localExpertNum;
    uint32_t aivNum;
    uint32_t networkMode;
    uint32_t serverNum;
    uint32_t rankNumPerServer;
    MoeEpDumpRegion regions[MOE_EP_DUMP_REGION_COUNT];
};

#endif // MOE_EP_EXCEPTION_DUMP_DEFS_H
