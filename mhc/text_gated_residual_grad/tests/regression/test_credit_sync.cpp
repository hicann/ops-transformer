/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cassert>
#include <iostream>
#include "../../op_kernel/arch35/text_gated_residual_grad_credit_sync.h"

using namespace TextGatedResidualGrad;

namespace {
constexpr uint32_t TEST_AIC_COUNT = 1;
constexpr uint32_t TAIL_TASK_COUNT = 4;
constexpr uint32_t TAIL_SLOT_COUNT = 3;
constexpr uint32_t REUSED_SLOT_TASK = 3;
constexpr uint32_t FINAL_REDUCTION_PAIR = 2;
constexpr uint32_t EXHAUSTIVE_TASK_LIMIT = 128;
constexpr uint32_t STRESS_TASK_COUNT = 1024;
} // namespace

void CheckDrained()
{
    for (const auto& core : testCredits) {
        for (int count : core) {
            assert(count == 0);
        }
    }
}

void CheckSchedule(uint32_t tasks, bool gamma)
{
    HcGradCreditSync credit;
    const auto slots = std::min(tasks, HC_GRAD_RING_SLOTS);
    credit.Init(nullptr, 0, tasks, slots, TEST_AIC_COUNT, TEST_AIV_PER_AIC);
    HcGradScheduler<false, false> scheduler(tasks, slots, gamma);
    unsigned beats = 0;
    while (!scheduler.Done()) {
        assert(beats++ < scheduler.MaxBeats());
        auto beat = scheduler.Next();
        testCore = TEST_AIC;
        credit.WaitCubeDeps(beat.cubeOp, beat.cubeTask);
        credit.WaitCubeDeps(beat.cubeOp2, beat.cubeTask2);
        for (testCore = TEST_AIV0; testCore <= TEST_AIV1; ++testCore) {
            for (unsigned i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                credit.WaitVectorDeps(beat.vectorOp[i], beat.vectorTask[i]);
            }
        }
        // Mark only after both sides' waits: a beat cannot consume its own output.
        testCore = TEST_AIC;
        credit.MarkCube(beat.cubeOp, beat.cubeTask);
        credit.MarkCube(beat.cubeOp2, beat.cubeTask2);
        for (testCore = TEST_AIV0; testCore <= TEST_AIV1; ++testCore) {
            for (unsigned i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                credit.MarkVector(beat.vectorOp[i], beat.vectorTask[i]);
            }
        }
    }
    CheckDrained();
}

void CheckPairedReductionTail()
{
    // Four tasks, three slots: only task 0 has a subsequent slot user.
    HcGradCreditSync credit;
    credit.Init(nullptr, 0, TAIL_TASK_COUNT, TAIL_SLOT_COUNT, TEST_AIC_COUNT, TEST_AIV_PER_AIC);
    for (testCore = TEST_AIV0; testCore <= TEST_AIV1; ++testCore) {
        credit.MarkVector(HcGradVectorOp::REDUCE_WD2, 0);
        credit.MarkVector(HcGradVectorOp::REDUCE_DACT, 0);
        credit.MarkVector(HcGradVectorOp::V1, REUSED_SLOT_TASK);
    }
    testCore = TEST_AIC;
    credit.WaitCubeDeps(HcGradCubeOp::C1, REUSED_SLOT_TASK);
    CheckDrained();
    for (testCore = TEST_AIV0; testCore <= TEST_AIV1; ++testCore) {
        credit.MarkVector(HcGradVectorOp::REDUCE_WD2, FINAL_REDUCTION_PAIR);
    }
    CheckDrained();
}

int main()
{
    for (unsigned tasks = 0; tasks <= EXHAUSTIVE_TASK_LIMIT; ++tasks) {
        CheckSchedule(tasks, false);
        CheckSchedule(tasks, true);
    }
    CheckSchedule(STRESS_TASK_COUNT, false);
    CheckSchedule(STRESS_TASK_COUNT, true);
    CheckPairedReductionTail();
    std::cout << "PASS: 260 schedules, both AIV subblocks, reserved IDs and paired reduction tail\n";
}
