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
 * \file flash_attention_score_grad_event_small_sd.h
 * \brief Small-S/Small-D dedicated event ids for arch35 regbase.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_EVENT_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_EVENT_SMALL_SD_H

#include <cstdint>

namespace FagBaseApi {

// SmallSD fixed DAG sync table.
//
// For AIV-owned events, primary ids are consumed/produced by subblock 0 and
// subblock 1 uses the same semantic event plus SMALL_SD_EVENT_MIRROR_OFFSET, so
// AIC can wait for both vector consumers before reusing shared slot resources.
// AIC-only events, such as SMALL_SD_CUBE_OUTPUT_COMMIT_FLAG, use only the primary
// id.  The numeric primary ids intentionally mirror the generic regbase allocation
// to keep cross-core resources unchanged, but the SmallSD path uses these semantic
// aliases only.
constexpr uint8_t SMALL_SD_EVENT_MIRROR_OFFSET = 16;
constexpr uint8_t SMALL_SD_CUBE_DYV_READY_FLAG[2] = {0, 1};
constexpr uint8_t SMALL_SD_CUBE_QK_READY_FLAG[2] = {2, 3};
constexpr uint8_t SMALL_SD_DS_L1_READY_FLAG = 4;
constexpr uint8_t SMALL_SD_P_L1_READY_FLAG = 5;
constexpr uint8_t SMALL_SD_DQ_UB_READY_FLAG = 6;
constexpr uint8_t SMALL_SD_DK_UB_READY_FLAG = 7;
constexpr uint8_t SMALL_SD_DS_L1_REUSABLE_FLAG = 8;
constexpr uint8_t SMALL_SD_P_L1_REUSABLE_FLAG = 9;
constexpr uint8_t SMALL_SD_SLOT_REUSE_READY_FLAG = 10;
constexpr uint8_t SMALL_SD_CUBE_OUTPUT_COMMIT_FLAG = 11;

struct SmallSDEventTable {
    static constexpr uint8_t cubeDyVReady0 = SMALL_SD_CUBE_DYV_READY_FLAG[0];
    static constexpr uint8_t cubeDyVReady1 = SMALL_SD_CUBE_DYV_READY_FLAG[1];
    static constexpr uint8_t cubeQKReady0 = SMALL_SD_CUBE_QK_READY_FLAG[0];
    static constexpr uint8_t cubeQKReady1 = SMALL_SD_CUBE_QK_READY_FLAG[1];
    static constexpr uint8_t dsL1Ready = SMALL_SD_DS_L1_READY_FLAG;
    static constexpr uint8_t pL1Ready = SMALL_SD_P_L1_READY_FLAG;
    static constexpr uint8_t dqUbReady = SMALL_SD_DQ_UB_READY_FLAG;
    static constexpr uint8_t dkUbReady = SMALL_SD_DK_UB_READY_FLAG;
    static constexpr uint8_t dsL1Reusable = SMALL_SD_DS_L1_REUSABLE_FLAG;
    static constexpr uint8_t pL1Reusable = SMALL_SD_P_L1_REUSABLE_FLAG;
    static constexpr uint8_t slotReuseReady = SMALL_SD_SLOT_REUSE_READY_FLAG;
    static constexpr uint8_t cubeOutputCommit = SMALL_SD_CUBE_OUTPUT_COMMIT_FLAG;
};

static_assert(SMALL_SD_CUBE_DYV_READY_FLAG[0] != SMALL_SD_CUBE_DYV_READY_FLAG[1],
              "SmallSD per-slot DyV ready events must be distinct.");
static_assert(SMALL_SD_CUBE_QK_READY_FLAG[0] != SMALL_SD_CUBE_QK_READY_FLAG[1],
              "SmallSD per-slot QK ready events must be distinct.");
static_assert(SMALL_SD_CUBE_DYV_READY_FLAG[1] < SMALL_SD_CUBE_QK_READY_FLAG[0],
              "SmallSD cube producer event ranges must not overlap.");
static_assert(SMALL_SD_CUBE_QK_READY_FLAG[1] < SMALL_SD_DS_L1_READY_FLAG,
              "SmallSD cube/vector event ranges must not overlap.");
static_assert(SMALL_SD_SLOT_REUSE_READY_FLAG < SMALL_SD_EVENT_MIRROR_OFFSET,
              "SmallSD primary event ids must stay below the mirror event offset.");
static_assert(SMALL_SD_CUBE_OUTPUT_COMMIT_FLAG < SMALL_SD_EVENT_MIRROR_OFFSET,
              "SmallSD cube output commit event must stay below the mirror event offset.");
static_assert(SMALL_SD_EVENT_MIRROR_OFFSET + SMALL_SD_CUBE_OUTPUT_COMMIT_FLAG < 32,
              "SmallSD mirrored event ids must stay inside the event id range.");

} // namespace FagBaseApi

#endif
