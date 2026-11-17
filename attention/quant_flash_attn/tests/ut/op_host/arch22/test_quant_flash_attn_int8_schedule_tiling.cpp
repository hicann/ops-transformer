/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <random>
#include "../../../../op_host/arch22/quant_flash_attn_schedule_int8.h"

namespace QuantFlashAttnUT {
using namespace BaseApi::Int8GqaTnd;
namespace {
QfaInt8ScheduleInput Model(int layout = 2, int dim = 128, int mask = 3)
{
    return {2,
            24,
            4,
            dim,
            dim,
            10240,
            10240,
            layout == 2 ? 20480 : 0,
            layout == 2 ? 20480 : 0,
            layout,
            layout,
            layout,
            mask,
            20,
            192ULL << 20,
            16ULL << 20,
            sizeof(QuantFlashAttnTilingQBlockPayloadInt8),
            0.125f};
}
uint64_t ReferenceWork(uint32_t tasks, uint32_t sq, uint32_t sk, uint32_t group, uint32_t m, uint32_t n, bool causal)
{
    const uint32_t tiles = (sq * group + m - 1) / m;
    uint64_t cost = 0;
    for (uint32_t i = 0; i < tasks; ++i) {
        const int64_t lastToken = std::min(sq * group, ((i % tiles) + 1) * m) - 1;
        const int64_t visible = causal ? std::max<int64_t>(0, lastToken / group + 1 + sk - sq) : sk;
        cost += (visible + n - 1) / n + 2;
    }
    return cost;
}
} // namespace

TEST(QuantFlashAttnInt8Workload, PrefixMatchesIndependentEnumeration)
{
    std::mt19937 random(20261115);
    for (uint32_t i = 0; i < 2000; ++i) {
        const uint32_t sq = 1 + random() % 2048, sk = 1 + random() % 2048, group = 1 + random() % 32;
        const uint32_t m = 32U << (random() % 5), n = 128U << (random() % 3);
        const uint32_t tasks = random() % (3 * ((sq * group + m - 1) / m) + 1);
        for (bool causal : {false, true}) {
            ASSERT_EQ(QfaSequenceWork(tasks, sq, sk, group, m, n, causal),
                      ReferenceWork(tasks, sq, sk, group, m, n, causal))
                << i;
        }
    }
}

TEST(QuantFlashAttnInt8Workload, SlidingWindowUsesBottomRightAlignment)
{
    for (uint32_t sq : {1U, 7U, 257U}) {
        for (uint32_t sk : {1U, 9U, 512U}) {
            for (int32_t left : {-1, 0, 5, INT32_MAX}) {
                for (int32_t right : {-1, 0, 17, INT32_MAX}) {
                    for (uint32_t q = 0; q < sq; ++q) {
                        const auto range = QfaWindowRow(q, sq, sk, left, right);
                        for (uint32_t k = 0; k < sk; ++k) {
                            const int64_t center = static_cast<int64_t>(q) + sk - sq;
                            const bool visible = (left < 0 || k >= center - left) && (right < 0 || k <= center + right);
                            ASSERT_EQ(k >= range.begin && k < range.end, visible);
                        }
                        const auto tile = QfaWindowTile(range, 3, 8);
                        ASSERT_LE(tile.begin, tile.end);
                        ASSERT_LE(tile.end, 8U);
                    }
                }
            }
        }
    }
}

TEST(QuantFlashAttnInt8Schedule, ModelMatrixCoversEveryTask)
{
    for (int layout : {0, 1, 2}) {
        for (int dim : {64, 96, 128, 256}) {
            for (int mask : {0, 3}) {
                for (int batch : {1, 2, 8, 32}) {
                    auto input = Model(layout, dim, mask);
                    input.batch = batch;
                    input.totalQ = input.totalK = layout == 2 ? batch * input.maxQ : 0;
                    QfaInt8Schedule schedule;
                    ASSERT_TRUE(BuildInt8Schedule(input, schedule));
                    const auto& payload = schedule.payload.base;
                    const auto& config = payload.config;
                    if (payload.useCoreRanges != QFA_CORE_RANGES_NONE) {
                        EXPECT_EQ(payload.coreTaskBegin[0], 0U);
                        EXPECT_EQ(payload.coreTaskBegin[config.usedCoreNum], config.totalTasks);
                        for (uint32_t core = 0; core < config.usedCoreNum; ++core) {
                            EXPECT_LE(payload.coreTaskBegin[core], payload.coreTaskBegin[core + 1]);
                        }
                    }
                    if (payload.useCoreRanges == QFA_CORE_RANGES_TASK_MAP) {
                        std::vector<uint32_t> ids(schedule.payload.taskIds,
                                                  schedule.payload.taskIds + config.totalTasks);
                        std::sort(ids.begin(), ids.end());
                        for (uint32_t task = 0; task < ids.size(); ++task)
                            ASSERT_EQ(ids[task], task);
                    }
                }
            }
        }
    }
}

TEST(QuantFlashAttnInt8Schedule, QBlockPayloadSelectionMatchesTaskMapConfig)
{
    auto bnsd = Model(QFA_LAYOUT_BNSD, QFA_HEAD_DIM_96);
    bnsd.batch = 1;
    bnsd.headsQ = 32;
    bnsd.maxQ = bnsd.maxK = 3072;
    bnsd.aicNum = 24;

    auto tnd = Model(QFA_LAYOUT_TND, QFA_HEAD_DIM_128);
    tnd.batch = 1;
    tnd.maxQ = tnd.maxK = 4608;
    tnd.totalQ = tnd.totalK = tnd.maxQ;
    tnd.aicNum = 24;

    QfaInt8Schedule schedule;
    ASSERT_TRUE(BuildInt8Schedule(bnsd, schedule));
    EXPECT_EQ(schedule.config, QFA_INT8_CFG_D96_M512_FIXPIPE);
    EXPECT_TRUE(schedule.hasTaskBuffer);
    EXPECT_EQ(schedule.payload.base.useCoreRanges, QFA_CORE_RANGES_TASK_MAP);

    // Fixpipe always sends the full payload, including when it uses contiguous ranges.
    bnsd.mask = QFA_MASK_NONE;
    ASSERT_TRUE(BuildInt8Schedule(bnsd, schedule));
    EXPECT_EQ(schedule.config, QFA_INT8_CFG_D96_M512_FIXPIPE);
    EXPECT_TRUE(schedule.hasTaskBuffer);
    EXPECT_NE(schedule.payload.base.useCoreRanges, QFA_CORE_RANGES_TASK_MAP);

    ASSERT_TRUE(BuildInt8Schedule(tnd, schedule));
    EXPECT_EQ(schedule.config, QFA_INT8_CFG_D128_M512_TASK_MAP);
    EXPECT_TRUE(schedule.hasTaskBuffer);
    EXPECT_EQ(schedule.payload.base.useCoreRanges, QFA_CORE_RANGES_TASK_MAP);

    tnd.tilingCapacity = sizeof(QuantFlashAttnTilingPayloadInt8);
    ASSERT_TRUE(BuildInt8Schedule(tnd, schedule));
    EXPECT_EQ(schedule.config, QFA_INT8_CFG_D128_M512);
    EXPECT_FALSE(schedule.hasTaskBuffer);
    EXPECT_NE(schedule.payload.base.useCoreRanges, QFA_CORE_RANGES_TASK_MAP);
}

TEST(QuantFlashAttnInt8Schedule, EverySupportedHeadDimensionAndWindow)
{
    for (int dim = 1; dim <= 256; ++dim) {
        auto input = Model(2, dim);
        input.winLeft = dim % 2 ? -1 : 0;
        input.winRight = dim % 3 ? 0 : 127;
        QfaInt8Schedule schedule;
        ASSERT_TRUE(BuildInt8Schedule(input, schedule)) << dim;
        EXPECT_EQ(schedule.payload.base.config.headDim, dim);
        EXPECT_EQ(schedule.payload.base.config.headDimAlign, (dim + 31) / 32 * 32);
        EXPECT_EQ(schedule.config & 256U, 256U);
    }
}

TEST(QuantFlashAttnInt8Schedule, WorkspaceAndCoreCapacityLimits)
{
    auto input = Model();
    input.aicNum = 7;
    QfaInt8Schedule schedule;
    ASSERT_TRUE(BuildInt8Schedule(input, schedule));
    EXPECT_EQ(schedule.payload.base.config.usedCoreNum, 7U);
    input.l2Bytes = input.reservedWorkspaceBytes + schedule.payload.base.config.coreWorkspaceBytes * 3ULL;
    ASSERT_TRUE(BuildInt8Schedule(input, schedule));
    EXPECT_EQ(schedule.payload.base.config.usedCoreNum, 3U);
    input.l2Bytes = input.reservedWorkspaceBytes;
    EXPECT_FALSE(BuildInt8Schedule(input, schedule));
    input = Model();
    input.tilingCapacity = sizeof(QuantFlashAttnTilingPayloadInt8) - 1;
    EXPECT_FALSE(BuildInt8Schedule(input, schedule));
    input = Model();
    input.maxQ = UINT32_MAX;
    EXPECT_FALSE(BuildInt8Schedule(input, schedule));
    input = Model();
    input.aicNum = 0;
    EXPECT_FALSE(BuildInt8Schedule(input, schedule));
}

TEST(QuantFlashAttnInt8Workload, VariableLengthCoreRangesCoverEveryTask)
{
    struct Reader {
        std::vector<QfaSequenceDescriptor> sequences;
        QfaSequenceDescriptor ReadSequence(uint32_t b)
        {
            return sequences[b];
        }
    } reader;
    for (uint32_t sq : {0U, 17U, 35U, 20480U, 3U, 40000U}) {
        QfaSequenceDescriptor s;
        s.sq = s.sk = sq;
        s.tiles = (sq * 6 + 511) / 512;
        s.tasks = s.tiles * 4;
        reader.sequences.push_back(s);
    }
    uint64_t previous = 0;
    for (uint32_t core = 0; core < 20; ++core) {
        QfaTaskRangePlanner<Reader> planner({6, 20, 6, 512, 512, 1}, reader);
        planner.Init(core);
        EXPECT_EQ(planner.Begin(), previous);
        EXPECT_LE(planner.Begin(), planner.End());
        previous = planner.End();
    }
    uint64_t total = 0;
    for (const auto& seq : reader.sequences)
        total += seq.tasks;
    EXPECT_EQ(previous, total);
}

TEST(QuantFlashAttnInt8Workload, KScaleWorkspaceBoundaries)
{
    EXPECT_EQ(QfaKScaleWorkspaceElements(false, true, 3, 512), 35U * 512);
    EXPECT_TRUE(QfaKScaleCachedTile(true, true, 31 * 512, 512));
    EXPECT_FALSE(QfaKScaleCachedTile(true, true, 32 * 512, 512));
    EXPECT_EQ(QfaKScaleGmOffset(true, true, 32 * 512, 2, 512), 34U * 512);
}
} // namespace QuantFlashAttnUT
