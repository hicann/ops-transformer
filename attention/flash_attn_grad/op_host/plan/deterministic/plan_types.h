/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FAG_DETERMINISTIC_PLAN_TYPES_H_
#define FAG_DETERMINISTIC_PLAN_TYPES_H_

#include <cstdint>
#include "family.h"

namespace optiling {

// Deterministic result shared by Host and the AICPU normalizer. Candidate builders never mutate FagParsedInfo.
// The device ABI is written by the existing serializer; this POD is not serialized by memcpy.
struct FagDeterPhase {
    int64_t kind = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t rowOffset = 0;
    int64_t colOffset = 0;
    int64_t fullRounds = 0;
    int64_t tailRounds = 0;
};

struct FagDeterBandPlan {
    FagDeterPhase phases[5] = {};
    FagDeterPhase tailPhases[5] = {};
    int64_t phaseCount = 0;
    int64_t tailPhaseCount = 0;
    int64_t k = 0;
    int64_t b = 0;
    int64_t g = 0;
    int64_t groupsPerChunk = 0;
    int64_t fullRounds = 0;
    int64_t tailRounds = 0;
    int64_t totalRounds = 0;
};

struct FagDeterPlan {
    int64_t family = DeterFamily::LEGACY;
    int64_t sparseKind = 0;
    int64_t maxRound = 0;
    int64_t denseRound = 0;
    int64_t causalRoundsPerGroup = 0;
    int64_t causalGroups = 0;
    int64_t bandMode = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
    int64_t rowOffset = 0;
    int64_t colOffset = 0;
    FagDeterBandPlan band;
};

} // namespace optiling

#endif // FAG_DETERMINISTIC_PLAN_TYPES_H_
