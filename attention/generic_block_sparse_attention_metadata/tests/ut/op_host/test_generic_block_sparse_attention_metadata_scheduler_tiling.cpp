/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "../../../op_kernel_aicpu/generic_block_sparse_attention_metadata_q_seq_utils.h"
#include "../../../op_kernel_aicpu/generic_block_sparse_attention_metadata_scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>

using namespace aicpu::generic_block_sparse_attention_metadata;
using namespace optiling::generic_block_sparse_attention_metadata;

namespace {

constexpr int64_t SUPPORTED_HEAD_DIM = 128;
constexpr int64_t SUPPORTED_BLOCK_SHAPE_X = 1;
constexpr int64_t SUPPORTED_BLOCK_SHAPE_Y = 128;
constexpr int64_t DEFAULT_BLOCK_INDEX_STRIDE = 16;
constexpr int64_t PACKED_GQA = 1;
constexpr int64_t UNPACKED_GQA = 0;

constexpr int64_t DEFAULT_BATCH_SIZE = 2;
constexpr int64_t DEFAULT_NUM_Q_HEADS = 8;
constexpr int64_t DEFAULT_NUM_KV_HEADS = 2;
constexpr int64_t DEFAULT_MAX_Q_SEQ_LEN = 3;
constexpr int64_t UNSPECIFIED_MAX_Q_SEQ_LEN = -1;
constexpr int64_t DEFAULT_Q_BLOCK_STORAGE_NUM = 6;
constexpr int64_t DEFAULT_AIC_CORE_NUM = 8;
constexpr int64_t DEFAULT_FIRST_Q_SEQ_LEN = 2;
constexpr int64_t DEFAULT_SECOND_Q_SEQ_LEN = 1;
constexpr int64_t DEFAULT_TOTAL_Q_TOKEN_NUM = 3;
constexpr int64_t DEFAULT_SA_TASK_NUM = 6;
constexpr int64_t DEFAULT_GROUP_SIZE = 4;
constexpr int64_t DEFAULT_DECODE_TASK_INDEX = 4;

constexpr int64_t DECODE_BATCH_SIZE = 1;
constexpr int64_t DECODE_NUM_Q_HEADS = 4;
constexpr int64_t DECODE_NUM_KV_HEADS = 1;
constexpr int64_t DECODE_Q_SEQ_LEN = 2;
constexpr int64_t DECODE_Q_BLOCK_STORAGE_NUM = 2;
constexpr int64_t DECODE_AIC_CORE_NUM = 32;
constexpr int64_t DECODE_FIRST_BLOCK_NUM = 12;
constexpr int64_t DECODE_SECOND_BLOCK_NUM = 3;
constexpr int64_t DECODE_TOTAL_VALID_BLOCK_NUM = DECODE_FIRST_BLOCK_NUM + DECODE_SECOND_BLOCK_NUM;
constexpr int64_t DECODE_ACTIVE_CORE_NUM = 5;
constexpr int64_t DECODE_PER_CORE_TASK_NUM = 3;
constexpr int64_t DECODE_FOURTH_CORE_INDEX = 3;
constexpr int64_t DECODE_FIFTH_CORE_INDEX = 4;
constexpr int64_t DECODE_FIRST_UNUSED_CORE_INDEX = 5;
constexpr int64_t DECODE_FOURTH_CORE_FIRST_BLOCK = 9;
constexpr int64_t DECODE_COMBINE_PARTIAL_COUNT = 4;

constexpr int64_t UNSUPPORTED_DIM = 513;
constexpr int64_t MAX_BLOCK_CAPACITY = 256;
constexpr int64_t ACTIVE_BASE_TASK_NUM = 8;
constexpr int64_t ACTIVE_AIC_CORE_NUM = 28;
constexpr int64_t ACTIVE_VALID_BLOCK_NUM = 16;
constexpr int64_t PARTIAL_AIC_CORE_NUM = 7;
constexpr int64_t PARTIAL_VALID_BLOCK_NUM = 3;
constexpr int64_t PARTIAL_ACTIVE_CORE_NUM = 3;
constexpr int64_t PARTIAL_COMBINE_TASK_NUM = 2;
constexpr int64_t PARTIAL_TASK_NUM = 4;
constexpr uint32_t HOST_CAPACITY_AIC_CORE_NUM = 28U;
constexpr uint32_t HOST_PARTIAL_CAPACITY = 35U;
constexpr uint32_t FD_BASE_TASK_RATIO_NUMERATOR = 3U;
constexpr uint32_t FD_BASE_TASK_RATIO_DENOMINATOR = 10U;

constexpr int64_t UNPACKED_TASK_INDEX = 5;
constexpr int64_t TND_Q_STORAGE_LEN_PER_BATCH = 4;
constexpr int64_t TND_TOTAL_Q_STORAGE_LEN = 8;
constexpr int64_t TND_FIRST_BATCH_USED_Q = 2;
constexpr int64_t TND_SECOND_BATCH_USED_Q = 3;
constexpr int64_t TND_INVALID_FIRST_BATCH_USED_Q = 5;
constexpr int64_t TND_INVALID_TOTAL_Q_STORAGE_LEN = 7;
constexpr int32_t TND_FIRST_BLOCK_COUNT = 1;

class GenericBlockSparseAttentionMetadataSchedulerTest : public testing::Test {};

ScheduleInput MakeInput()
{
    ScheduleInput input;
    input.batchSize = DEFAULT_BATCH_SIZE;
    input.numQHeads = DEFAULT_NUM_Q_HEADS;
    input.numKvHeads = DEFAULT_NUM_KV_HEADS;
    input.maxQSeqLen = DEFAULT_MAX_Q_SEQ_LEN;
    input.headDim = SUPPORTED_HEAD_DIM;
    input.blockShapeX = SUPPORTED_BLOCK_SHAPE_X;
    input.blockShapeY = SUPPORTED_BLOCK_SHAPE_Y;
    input.blockIndexStride = DEFAULT_BLOCK_INDEX_STRIDE;
    input.qBlockStorageNum = DEFAULT_Q_BLOCK_STORAGE_NUM;
    input.isPackedGQA = PACKED_GQA;
    input.aicCoreNum = DEFAULT_AIC_CORE_NUM;
    input.qSeqLens = {DEFAULT_FIRST_Q_SEQ_LEN, DEFAULT_SECOND_Q_SEQ_LEN};
    input.validBlockNums.assign(DEFAULT_Q_BLOCK_STORAGE_NUM, 0);
    return input;
}

ScheduleInput MakeDecodeInput()
{
    ScheduleInput input;
    input.batchSize = DECODE_BATCH_SIZE;
    input.numQHeads = DECODE_NUM_Q_HEADS;
    input.numKvHeads = DECODE_NUM_KV_HEADS;
    input.maxQSeqLen = DECODE_Q_SEQ_LEN;
    input.headDim = SUPPORTED_HEAD_DIM;
    input.blockShapeX = SUPPORTED_BLOCK_SHAPE_X;
    input.blockShapeY = SUPPORTED_BLOCK_SHAPE_Y;
    input.blockIndexStride = DEFAULT_BLOCK_INDEX_STRIDE;
    input.qBlockStorageNum = DECODE_Q_BLOCK_STORAGE_NUM;
    input.isPackedGQA = PACKED_GQA;
    input.aicCoreNum = DECODE_AIC_CORE_NUM;
    input.qSeqLens = {DECODE_Q_SEQ_LEN};
    input.validBlockNums = {DECODE_FIRST_BLOCK_NUM, DECODE_SECOND_BLOCK_NUM};
    return input;
}

uint32_t GetHostFdPartialCapacity(uint32_t aicCoreNum)
{
    const uint32_t maxNonEmptyBaseTaskNum =
        aicCoreNum == 0U ? 0U :
                           std::min<uint32_t>(MAX_COMBINE_TASK_NUM, (aicCoreNum * FD_BASE_TASK_RATIO_NUMERATOR - 1U) /
                                                                        FD_BASE_TASK_RATIO_DENOMINATOR);
    const uint32_t maxActiveCoreNum = std::min<uint32_t>(aicCoreNum, MAX_FD_ACTIVE_CORE_NUM);
    return maxNonEmptyBaseTaskNum == 0U || maxActiveCoreNum == 0U ? 0U : maxNonEmptyBaseTaskNum + maxActiveCoreNum - 1U;
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PrefillSchedule)
{
    ScheduleInput input = MakeInput();
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.totalQTokenNum, DEFAULT_TOTAL_Q_TOKEN_NUM);
    ASSERT_EQ(result.saTotalTaskNum, DEFAULT_SA_TASK_NUM);
    ASSERT_EQ(result.saUsedCoreNum, DEFAULT_SA_TASK_NUM);
    ASSERT_EQ(result.sparseHeadNum, DEFAULT_NUM_KV_HEADS);
    ASSERT_EQ(result.groupSize, DEFAULT_GROUP_SIZE);
    ASSERT_EQ(result.fdActiveCoreNum, 0);

    TaskInfo task;
    ASSERT_EQ(DecodeTask(input, result, DEFAULT_DECODE_TASK_INDEX, task), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(task.qUnit, DEFAULT_FIRST_Q_SEQ_LEN);
    ASSERT_EQ(task.batchIdx, 1);
    ASSERT_EQ(task.qTokenInBatch, 0);
    ASSERT_EQ(task.kvHeadIdx, 0);
    ASSERT_EQ(task.qHeadStart, 0);
    ASSERT_EQ(task.qHeadCount, DEFAULT_GROUP_SIZE);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ActualBlockPrefixDecode)
{
    ScheduleInput input = MakeDecodeInput();
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.blockPrefix[0], 0);
    ASSERT_EQ(result.blockPrefix[1], DECODE_FIRST_BLOCK_NUM);
    ASSERT_EQ(result.blockPrefix[DECODE_Q_BLOCK_STORAGE_NUM], DECODE_TOTAL_VALID_BLOCK_NUM);
    ASSERT_EQ(result.fdActiveCoreNum, DECODE_ACTIVE_CORE_NUM);
    ASSERT_EQ(result.decodePerCoreTaskNum, DECODE_PER_CORE_TASK_NUM);

    ASSERT_EQ(result.decodeSchedules[0].baseTaskStart, 0);
    ASSERT_EQ(result.decodeSchedules[0].baseTaskEnd, 1);
    ASSERT_EQ(result.decodeSchedules[0].firstBlockStart, 0);
    ASSERT_EQ(result.decodeSchedules[0].lastBlockEnd, DECODE_PER_CORE_TASK_NUM);
    ASSERT_EQ(result.decodeSchedules[DECODE_FOURTH_CORE_INDEX].firstBlockStart, DECODE_FOURTH_CORE_FIRST_BLOCK);
    ASSERT_EQ(result.decodeSchedules[DECODE_FOURTH_CORE_INDEX].lastBlockEnd, DECODE_FIRST_BLOCK_NUM);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIFTH_CORE_INDEX].baseTaskStart, 1);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIFTH_CORE_INDEX].baseTaskEnd, DECODE_Q_BLOCK_STORAGE_NUM);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIFTH_CORE_INDEX].firstBlockStart, 0);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIFTH_CORE_INDEX].lastBlockEnd, DECODE_SECOND_BLOCK_NUM);
    ASSERT_EQ(result.decodeSchedules[DECODE_FIRST_UNUSED_CORE_INDEX].baseTaskEnd, 0);

    ASSERT_EQ(result.combineTaskNum, 1);
    ASSERT_EQ(result.combineSchedules[0].baseTask, 0);
    ASSERT_EQ(result.combineSchedules[0].firstCore, 0);
    ASSERT_EQ(result.combineSchedules[0].partialStart, 0);
    ASSERT_EQ(result.combineSchedules[0].partialCount, DECODE_COMBINE_PARTIAL_COUNT);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, DecodeShapeConstraints)
{
    for (int64_t headDim : {1, 64, 127, 128, 129, 256, 257, 384, 511, 512}) {
        ScheduleInput input = MakeDecodeInput();
        ScheduleResult result;
        input.headDim = headDim;
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        ASSERT_GT(result.fdActiveCoreNum, 0);
    }
    ScheduleInput input = MakeDecodeInput();
    ScheduleResult result;
    input.headDim = UNSUPPORTED_DIM;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.fdActiveCoreNum, 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, GeneralizedYUsesFdSchedule)
{
    for (int64_t y = 1; y <= 128; ++y) {
        ScheduleInput input = MakeDecodeInput();
        input.blockShapeY = y;
        ScheduleResult result;
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        ASSERT_GT(result.fdActiveCoreNum, 0);
        ASSERT_NE(static_cast<uint32_t>(result.fdScheduleFlags) & FD_SCHEDULE_ENABLED, 0U);
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ZeroCountsKeepFdSchedulingSlots)
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = DEFAULT_MAX_Q_SEQ_LEN;
    input.numQHeads = 1;
    input.numKvHeads = 1;
    input.aicCoreNum = DECODE_AIC_CORE_NUM;
    input.qSeqLens = {DEFAULT_MAX_Q_SEQ_LEN};
    input.validBlockNums = {0, DECODE_FIRST_BLOCK_NUM, DECODE_SECOND_BLOCK_NUM};
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.blockPrefix[0], 0);
    ASSERT_EQ(result.blockPrefix[1], 1);
    ASSERT_EQ(result.blockPrefix[DECODE_Q_BLOCK_STORAGE_NUM], DECODE_FIRST_BLOCK_NUM + 1);
    ASSERT_EQ(result.blockPrefix[DEFAULT_MAX_Q_SEQ_LEN], DECODE_TOTAL_VALID_BLOCK_NUM + 1);
    ASSERT_GT(result.fdActiveCoreNum, 0);
    ASSERT_EQ(result.decodeSchedules[0].baseTaskStart, 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, FdCoversEmptyTasksAtEachPosition)
{
    for (const auto& counts :
         {std::vector<int64_t>{0, 12, 3}, std::vector<int64_t>{12, 0, 3}, std::vector<int64_t>{12, 3, 0}}) {
        ScheduleInput input = MakeDecodeInput();
        input.qSeqLens = {3};
        input.maxQSeqLen = 3;
        input.qBlockStorageNum = 3;
        input.validBlockNums = counts;
        ScheduleResult result;
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        ASSERT_GT(result.fdActiveCoreNum, 0);
        std::array<int, 3> visits{};
        for (int64_t core = 0; core < result.fdActiveCoreNum; ++core) {
            const auto& schedule = result.decodeSchedules[core];
            for (int64_t task = schedule.baseTaskStart; task < schedule.baseTaskEnd; ++task) {
                ASSERT_GE(task, 0);
                ASSERT_LT(task, 3);
                ++visits[static_cast<size_t>(task)];
            }
        }
        for (size_t task = 0; task < counts.size(); ++task) {
            ASSERT_GT(visits[task], 0);
            if (counts[task] == 0) {
                EXPECT_EQ(visits[task], 1);
            }
        }
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, AllEmptyTasksKeepNormalScheduling)
{
    ScheduleInput input = MakeDecodeInput();
    input.validBlockNums = {0, 0};
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    EXPECT_EQ(result.fdActiveCoreNum, 0);
    EXPECT_EQ(static_cast<uint32_t>(result.fdScheduleFlags) & FD_SCHEDULE_ENABLED, 0U);
    EXPECT_EQ(result.saTotalTaskNum, 2);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, SparseBlockCapacityUpTo256)
{
    ScheduleInput lhs = MakeDecodeInput();
    lhs.blockIndexStride = DECODE_FIRST_BLOCK_NUM;
    ScheduleInput rhs = lhs;
    rhs.blockIndexStride = MAX_BLOCK_CAPACITY;
    ScheduleResult lhsResult;
    ScheduleResult rhsResult;
    ASSERT_EQ(BuildSchedule(lhs, lhsResult), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(BuildSchedule(rhs, rhsResult), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_TRUE(lhsResult.fdActiveCoreNum > 0);
    ASSERT_EQ(rhsResult.fdActiveCoreNum, lhsResult.fdActiveCoreNum);
    ASSERT_EQ(rhsResult.decodePerCoreTaskNum, lhsResult.decodePerCoreTaskNum);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ActiveCoreCapacityAcrossBaseTasks)
{
    ScheduleInput input = MakeDecodeInput();
    input.maxQSeqLen = ACTIVE_BASE_TASK_NUM;
    input.qBlockStorageNum = ACTIVE_BASE_TASK_NUM;
    input.aicCoreNum = ACTIVE_AIC_CORE_NUM;
    input.qSeqLens = {ACTIVE_BASE_TASK_NUM};
    input.validBlockNums.assign(ACTIVE_BASE_TASK_NUM, ACTIVE_VALID_BLOCK_NUM);

    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_TRUE(result.fdActiveCoreNum > ACTIVE_VALID_BLOCK_NUM);
    ASSERT_TRUE(result.fdActiveCoreNum <= input.aicCoreNum);
    ASSERT_EQ(result.combineTaskNum, ACTIVE_BASE_TASK_NUM);
    int64_t partialTaskNum = 0;
    for (int64_t task = 0; task < result.combineTaskNum; ++task) {
        const CombineSchedule& combine = result.combineSchedules[static_cast<size_t>(task)];
        ASSERT_TRUE(combine.partialCount > 1);
        ASSERT_TRUE(combine.partialCount <= static_cast<int64_t>(MAX_FD_PARTIAL_PER_BASE_TASK));
        ASSERT_TRUE(combine.firstCore + combine.partialCount <= result.fdActiveCoreNum);
        partialTaskNum += combine.partialCount;
    }
    ASSERT_EQ(result.fdPartialTaskNum, partialTaskNum);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PartialTaskNumCanExceedActiveCoreNum)
{
    ScheduleInput input = MakeDecodeInput();
    input.aicCoreNum = PARTIAL_AIC_CORE_NUM;
    input.validBlockNums.assign(DECODE_Q_BLOCK_STORAGE_NUM, PARTIAL_VALID_BLOCK_NUM);

    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.fdActiveCoreNum, PARTIAL_ACTIVE_CORE_NUM);
    ASSERT_EQ(result.combineTaskNum, PARTIAL_COMBINE_TASK_NUM);
    ASSERT_EQ(result.fdPartialTaskNum, PARTIAL_TASK_NUM);
    ASSERT_TRUE(result.fdPartialTaskNum > result.fdActiveCoreNum);
    ASSERT_EQ(result.fdPartialTaskNum, static_cast<int64_t>(input.validBlockNums.size()) + result.fdActiveCoreNum - 1);
    ASSERT_TRUE(result.fdPartialTaskNum <= GetHostFdPartialCapacity(static_cast<uint32_t>(input.aicCoreNum)));
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, PartialWorkspaceUpperBound)
{
    ASSERT_EQ(GetHostFdPartialCapacity(0U), 0U);
    ASSERT_EQ(GetHostFdPartialCapacity(HOST_CAPACITY_AIC_CORE_NUM), HOST_PARTIAL_CAPACITY);
    for (uint32_t aicCoreNum = 1U; aicCoreNum <= MAX_FD_ACTIVE_CORE_NUM; ++aicCoreNum) {
        const uint32_t maxNonEmptyBaseTaskNum =
            (aicCoreNum * FD_BASE_TASK_RATIO_NUMERATOR - 1U) / FD_BASE_TASK_RATIO_DENOMINATOR;
        const uint32_t hostCapacity = GetHostFdPartialCapacity(aicCoreNum);
        for (uint32_t baseTaskNum = 1U; baseTaskNum <= maxNonEmptyBaseTaskNum; ++baseTaskNum) {
            for (int64_t validBlockNum = 1; validBlockNum <= ACTIVE_VALID_BLOCK_NUM; ++validBlockNum) {
                ScheduleInput input = MakeDecodeInput();
                input.maxQSeqLen = baseTaskNum;
                input.qBlockStorageNum = baseTaskNum;
                input.aicCoreNum = aicCoreNum;
                input.qSeqLens = {baseTaskNum};
                input.validBlockNums.assign(baseTaskNum, validBlockNum);

                ScheduleResult result;
                ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
                if ((static_cast<uint32_t>(result.fdScheduleFlags) & FD_SCHEDULE_ENABLED) == 0U) {
                    continue;
                }
                const int64_t actualIntersectionUpperBound =
                    static_cast<int64_t>(baseTaskNum) + result.fdActiveCoreNum - 1;
                ASSERT_TRUE(result.fdPartialTaskNum <= actualIntersectionUpperBound);
                ASSERT_TRUE(result.fdPartialTaskNum <= hostCapacity);
                for (int64_t combineIdx = 0; combineIdx < result.combineTaskNum; ++combineIdx) {
                    const CombineSchedule& combine = result.combineSchedules[static_cast<size_t>(combineIdx)];
                    ASSERT_TRUE(combine.partialStart >= 0);
                    ASSERT_TRUE(combine.partialStart + combine.partialCount <= result.fdPartialTaskNum);
                }
            }
        }
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, UnpackedInternalTaskModel)
{
    ScheduleInput input = MakeInput();
    input.batchSize = 1;
    input.maxQSeqLen = 1;
    input.qSeqLens = {1};
    input.isPackedGQA = UNPACKED_GQA;
    input.validBlockNums.assign(DEFAULT_NUM_Q_HEADS, 0);
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.sparseHeadNum, DEFAULT_NUM_Q_HEADS);
    ASSERT_EQ(result.saTotalTaskNum, DEFAULT_NUM_Q_HEADS);

    TaskInfo task;
    ASSERT_EQ(DecodeTask(input, result, UNPACKED_TASK_INDEX, task), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(task.sparseHeadIdx, UNPACKED_TASK_INDEX);
    ASSERT_EQ(task.qHeadStart, UNPACKED_TASK_INDEX);
    ASSERT_EQ(task.qHeadCount, 1);
    ASSERT_EQ(task.kvHeadIdx, 1);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, InvalidBlockCount)
{
    ScheduleInput input = MakeInput();
    input.validBlockNums[0] = input.blockIndexStride + 1;
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_INVALID_PARAM);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ResidualBlockExtendsVirtualCapacity)
{
    ScheduleInput input = MakeInput();
    input.residualBlockMode = 1;
    input.validBlockNums[0] = input.blockIndexStride + 1;
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.blockPrefix[1], input.blockIndexStride + 1);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, TndSeqUsedUsesStorageOffsets)
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    ASSERT_TRUE(BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN},
                                   {TND_FIRST_BATCH_USED_Q, TND_SECOND_BATCH_USED_Q}, true, SUPPORTED_BLOCK_SHAPE_X,
                                   TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens, qStorageBlockStarts));
    ASSERT_EQ(actualQSeqLens[0], TND_FIRST_BATCH_USED_Q);
    ASSERT_EQ(actualQSeqLens[1], TND_SECOND_BATCH_USED_Q);
    ASSERT_EQ(qStorageBlockStarts[0], 0);
    ASSERT_EQ(qStorageBlockStarts[1], TND_Q_STORAGE_LEN_PER_BATCH);

    std::array<int32_t, TND_TOTAL_Q_STORAGE_LEN> counts{};
    std::iota(counts.begin(), counts.end(), TND_FIRST_BLOCK_COUNT);
    std::vector<int64_t> validBlockNums;
    ASSERT_TRUE(GatherTndValidBlockNums(counts.data(), 1, TND_TOTAL_Q_STORAGE_LEN, SUPPORTED_BLOCK_SHAPE_X,
                                        TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens, qStorageBlockStarts, validBlockNums));
    ASSERT_EQ(validBlockNums.size(), static_cast<size_t>(TND_FIRST_BATCH_USED_Q + TND_SECOND_BATCH_USED_Q));
    ASSERT_EQ(validBlockNums[0], counts[0]);
    ASSERT_EQ(validBlockNums[1], counts[1]);
    ASSERT_EQ(validBlockNums[TND_FIRST_BATCH_USED_Q], counts[TND_Q_STORAGE_LEN_PER_BATCH]);
    ASSERT_EQ(validBlockNums[TND_FIRST_BATCH_USED_Q + 1], counts[TND_Q_STORAGE_LEN_PER_BATCH + 1]);
    ASSERT_EQ(validBlockNums[TND_FIRST_BATCH_USED_Q + TND_SECOND_BATCH_USED_Q - 1],
              counts[TND_Q_STORAGE_LEN_PER_BATCH + TND_SECOND_BATCH_USED_Q - 1]);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, SchedulerAllowsUnspecifiedMaxQSeqLen)
{
    ScheduleInput input = MakeInput();
    input.maxQSeqLen = UNSPECIFIED_MAX_Q_SEQ_LEN;
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(result.totalQTokenNum, DEFAULT_TOTAL_Q_TOKEN_NUM);
    ASSERT_EQ(result.saTotalTaskNum, DEFAULT_SA_TASK_NUM);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, TndQSeqWithoutSeqUsedUsesStorageLengths)
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    ASSERT_TRUE(BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN}, {}, false,
                                   SUPPORTED_BLOCK_SHAPE_X, TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens,
                                   qStorageBlockStarts));
    ASSERT_EQ(actualQSeqLens[0], TND_Q_STORAGE_LEN_PER_BATCH);
    ASSERT_EQ(actualQSeqLens[1], TND_Q_STORAGE_LEN_PER_BATCH);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, TndQSeqValidation)
{
    std::vector<int64_t> actualQSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    ASSERT_TRUE(!BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN},
                                    {TND_INVALID_FIRST_BATCH_USED_Q, TND_SECOND_BATCH_USED_Q}, true,
                                    SUPPORTED_BLOCK_SHAPE_X, TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens,
                                    qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_SECOND_BATCH_USED_Q}, {}, false,
                                    SUPPORTED_BLOCK_SHAPE_X, TND_Q_STORAGE_LEN_PER_BATCH, actualQSeqLens,
                                    qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({1, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN}, {}, false,
                                    SUPPORTED_BLOCK_SHAPE_X, TND_TOTAL_Q_STORAGE_LEN, actualQSeqLens,
                                    qStorageBlockStarts));
    ASSERT_TRUE(!BuildTndQSeqLayout({0, TND_Q_STORAGE_LEN_PER_BATCH, TND_TOTAL_Q_STORAGE_LEN}, {}, false,
                                    SUPPORTED_BLOCK_SHAPE_X, TND_INVALID_TOTAL_Q_STORAGE_LEN, actualQSeqLens,
                                    qStorageBlockStarts));
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, MetadataEncoding)
{
    ScheduleInput input = MakeDecodeInput();
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
    std::array<MetadataType, METADATA_TOTAL_SIZE> metadata{};
    ASSERT_EQ(EncodeMetadata(result, metadata.data(), metadata.size()), BSAScheduleStatus::BSA_SUCCESS);
    ASSERT_EQ(metadata[MAGIC_INDEX], METADATA_MAGIC);
    ASSERT_EQ(metadata[VERSION_INDEX], METADATA_VERSION);
    ASSERT_EQ(metadata[METADATA_USED_SIZE_INDEX], static_cast<int32_t>(METADATA_USED_SIZE));
    ASSERT_EQ(metadata[SA_USED_CORE_NUM_INDEX], DECODE_Q_BLOCK_STORAGE_NUM);
    ASSERT_EQ(metadata[SA_TOTAL_TASK_NUM_INDEX], DECODE_Q_BLOCK_STORAGE_NUM);
    ASSERT_EQ(metadata[FD_ACTIVE_CORE_NUM_INDEX], DECODE_ACTIVE_CORE_NUM);
    ASSERT_EQ(metadata[DECODE_PER_CORE_TASK_NUM_INDEX], DECODE_PER_CORE_TASK_NUM);
    ASSERT_EQ(metadata[COMBINE_TASK_NUM_INDEX], 1);
    ASSERT_EQ(metadata[FD_SCHEDULE_FLAGS_INDEX], static_cast<int32_t>(FD_SCHEDULE_ENABLED | FD_ACTUAL_BLOCK_PREFIX));
    ASSERT_EQ(metadata[FD_TOTAL_FLAT_TASK_NUM_INDEX], DECODE_TOTAL_VALID_BLOCK_NUM);
    ASSERT_EQ(metadata[FD_PARTIAL_TASK_NUM_INDEX], DECODE_COMBINE_PARTIAL_COUNT);
    ASSERT_EQ(metadata[CONFIG_SIGNATURE_INDEX], static_cast<int32_t>(result.configSignature));
    ASSERT_EQ(metadata[GetDecodeScheduleIndex(DECODE_FIFTH_CORE_INDEX, DECODE_BASE_TASK_START_INDEX)], 1);
    ASSERT_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_BASE_TASK_INDEX)], 0);
    ASSERT_EQ(metadata[GetCombineScheduleIndex(0, COMBINE_PARTIAL_COUNT_INDEX)], DECODE_COMBINE_PARTIAL_COUNT);
    ASSERT_EQ(metadata[METADATA_USED_SIZE], 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ScheduleInputDefaultsMatchInterface)
{
    ScheduleInput input;
    EXPECT_EQ(input.residualBlockMode, 0);
    EXPECT_EQ(input.isConsistentTopk, 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, OptionalMxfp8EffectiveCountRange)
{
    for (const int64_t capacity : {513, 514, 1024, 8193}) {
        auto input = MakeInput();
        input.quantType = 6;
        input.blockIndexStride = capacity;
        input.validBlockNums.assign(input.validBlockNums.size(), 512);
        ScheduleResult result;
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        std::array<MetadataType, METADATA_TOTAL_SIZE> metadata{};
        ASSERT_EQ(EncodeMetadata(result, metadata.data(), metadata.size()), BSAScheduleStatus::BSA_SUCCESS);
        EXPECT_EQ(metadata[PACKED_COUNT_RANGE_TAG_INDEX], PACKED_COUNT_RANGE_TAG);
        EXPECT_EQ(metadata[MIN_SPARSE_COUNT_INDEX], 512);
        EXPECT_EQ(metadata[MAX_SPARSE_COUNT_INDEX], 512);

        // An empty batch contributes no padded counts; every logical head/query
        // in the other batches contributes, including zero and large counts.
        input.batchSize = 3;
        input.qSeqLens = {2, 0, 1};
        input.validBlockNums = {0, 1, 512, 513, capacity, 64};
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        ASSERT_EQ(EncodeMetadata(result, metadata.data(), metadata.size()), BSAScheduleStatus::BSA_SUCCESS);
        EXPECT_EQ(metadata[MIN_SPARSE_COUNT_INDEX], 0);
        EXPECT_EQ(metadata[MAX_SPARSE_COUNT_INDEX], capacity);

        input.quantType = 0;
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        ASSERT_EQ(EncodeMetadata(result, metadata.data(), metadata.size()), BSAScheduleStatus::BSA_SUCCESS);
        EXPECT_EQ(metadata[PACKED_COUNT_RANGE_TAG_INDEX], 0);
        EXPECT_EQ(metadata[MIN_SPARSE_COUNT_INDEX], 0);
        EXPECT_EQ(metadata[MAX_SPARSE_COUNT_INDEX], 0);
    }
    auto empty = MakeInput();
    empty.quantType = 6;
    empty.qSeqLens = {0, 0};
    empty.validBlockNums.clear();
    ScheduleResult result;
    ASSERT_EQ(BuildSchedule(empty, result), BSAScheduleStatus::BSA_SUCCESS);
    EXPECT_TRUE(result.hasPackedCountRange);
    EXPECT_EQ(result.minSparseCount, 0);
    EXPECT_EQ(result.maxSparseCount, 0);
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, ConfigSignatureTracksResidualAndConsistentTopk)
{
    ScheduleInput base = MakeDecodeInput();
    ScheduleResult baseResult;
    ASSERT_EQ(BuildSchedule(base, baseResult), BSAScheduleStatus::BSA_SUCCESS);
    const uint32_t baseSignature = baseResult.configSignature;
    EXPECT_EQ(baseSignature, CalculateConfigSignature(base));

    ScheduleInput residualChanged = MakeDecodeInput();
    residualChanged.residualBlockMode = 1;
    EXPECT_NE(CalculateConfigSignature(residualChanged), baseSignature);

    ScheduleInput topkChanged = MakeDecodeInput();
    topkChanged.isConsistentTopk = 1;
    EXPECT_NE(CalculateConfigSignature(topkChanged), baseSignature);

    // The reserved topk flag affects the signature, but not scheduling.
    ScheduleResult topkResult;
    ASSERT_EQ(BuildSchedule(topkChanged, topkResult), BSAScheduleStatus::BSA_SUCCESS);
    EXPECT_EQ(topkResult.blockPrefix, baseResult.blockPrefix);
    EXPECT_EQ(topkResult.saUsedCoreNum, baseResult.saUsedCoreNum);
    EXPECT_EQ(topkResult.saTotalTaskNum, baseResult.saTotalTaskNum);
    EXPECT_EQ(topkResult.fdPartialTaskNum, baseResult.fdPartialTaskNum);
    EXPECT_EQ(topkResult.combineTaskNum, baseResult.combineTaskNum);
    EXPECT_NE(topkResult.configSignature, baseSignature);
}

} // namespace

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, Mxfp8FdPreservesBaseTasksAndSparseRangeSchedule)
{
    for (int64_t blockY : {4, 8}) {
        for (int64_t maskType : {0, 1}) {
            ScheduleInput input = MakeDecodeInput();
            input.headDim = blockY == 4 ? 256 : 192;
            input.blockShapeY = blockY;
            input.blockIndexStride = 513;
            input.validBlockNums.assign(DECODE_Q_BLOCK_STORAGE_NUM, 513);
            input.maskType = maskType;
            ScheduleResult baseline;
            ASSERT_EQ(BuildSchedule(input, baseline), BSAScheduleStatus::BSA_SUCCESS);
            ASSERT_NE(static_cast<uint32_t>(baseline.fdScheduleFlags) & FD_SCHEDULE_ENABLED, 0U);
            for (int64_t quantType : {5, 6}) {
                input.quantType = quantType;
                ScheduleResult result;
                ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
                EXPECT_EQ(result.configSignature, baseline.configSignature);
                EXPECT_EQ(result.saTotalTaskNum, baseline.saTotalTaskNum);
                EXPECT_EQ(result.saUsedCoreNum, baseline.saUsedCoreNum);
                EXPECT_EQ(result.blockPrefix, baseline.blockPrefix);
                EXPECT_EQ(result.fdScheduleFlags, baseline.fdScheduleFlags);
                EXPECT_EQ(result.fdActiveCoreNum, baseline.fdActiveCoreNum);
                EXPECT_EQ(result.fdPartialTaskNum, baseline.fdPartialTaskNum);
                EXPECT_LE(result.fdPartialTaskNum, GetHostFdPartialCapacity(input.aicCoreNum));
            }
        }
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, Mxfp8FdCountAndTaskBoundaries)
{
    for (int64_t dim : {192, 256}) {
        for (int64_t group : {1, 16, 17, 32, 33, 64, 65, 128}) {
            for (int64_t capacity : {11, 12, 512, 513, 8193}) {
                ScheduleInput input = MakeDecodeInput();
                input.quantType = 6;
                input.headDim = dim;
                input.blockShapeY = dim == 192 ? 8 : 4;
                input.numQHeads = group;
                input.blockIndexStride = capacity;
                input.validBlockNums = {capacity, 0};
                ScheduleResult result;
                ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
                EXPECT_EQ((result.fdScheduleFlags & FD_SCHEDULE_ENABLED) != 0, capacity >= 512);
                EXPECT_EQ(result.saTotalTaskNum, 2);
                EXPECT_EQ(result.blockPrefix.back(), capacity + 1);
                EXPECT_LE(result.fdPartialTaskNum, GetHostFdPartialCapacity(input.aicCoreNum));
                if (capacity < 512) {
                    continue;
                }
                // Each valid sparse block, and the zero-output task's virtual
                // block, must have exactly one owner in the flat partition.
                int64_t expectedStart = 0;
                for (int64_t core = 0; core < result.fdActiveCoreNum; ++core) {
                    const auto& s = result.decodeSchedules[core];
                    const int64_t start = result.blockPrefix[s.baseTaskStart] + s.firstBlockStart;
                    const int64_t end = result.blockPrefix[s.baseTaskEnd - 1] + s.lastBlockEnd;
                    EXPECT_EQ(start, expectedStart);
                    EXPECT_GT(end, start);
                    expectedStart = end;
                }
                EXPECT_EQ(expectedStart, result.blockPrefix.back());
            }
        }
    }
    for (int64_t tasks : {0, 9, 10}) {
        ScheduleInput input = MakeDecodeInput();
        input.quantType = 6;
        input.headDim = 256;
        input.blockShapeY = 4;
        input.blockIndexStride = 512;
        input.maxQSeqLen = 10;
        input.qBlockStorageNum = 10;
        input.qSeqLens = {tasks};
        input.validBlockNums.assign(tasks, 512);
        ScheduleResult result;
        ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
        // 32 physical AICs: T=9 satisfies 10*T<96; T=10 does not.
        EXPECT_EQ((result.fdScheduleFlags & FD_SCHEDULE_ENABLED) != 0, tasks == 9);
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, Mxfp8FdSkipsInsufficientPackedWork)
{
    ScheduleInput input = MakeDecodeInput();
    input.headDim = 256;
    input.blockShapeY = 4;
    input.numQHeads = 12;
    input.maxQSeqLen = 8;
    input.qBlockStorageNum = 8;
    input.qSeqLens = {8};
    input.validBlockNums = {0, 1, 1, 12, 1, 12, 12, 31};
    for (int64_t capacity : {512, 513, 8193}) {
        input.blockIndexStride = capacity;
        input.quantType = 0;
        ScheduleResult unquantized;
        ASSERT_EQ(BuildSchedule(input, unquantized), BSAScheduleStatus::BSA_SUCCESS);
        EXPECT_NE(unquantized.fdScheduleFlags & FD_SCHEDULE_ENABLED, 0U);
        input.quantType = 6;
        ScheduleResult quantized;
        ASSERT_EQ(BuildSchedule(input, quantized), BSAScheduleStatus::BSA_SUCCESS);
        EXPECT_EQ(quantized.fdScheduleFlags & FD_SCHEDULE_ENABLED, 0U);
        EXPECT_EQ(quantized.blockPrefix, unquantized.blockPrefix);
        EXPECT_EQ(quantized.fdPartialTaskNum, 0);
    }
}

TEST_F(GenericBlockSparseAttentionMetadataSchedulerTest, Mxfp8FdPackedWorkTierBoundary)
{
    for (int64_t group : {1, 17, 33, 65}) {
        for (int64_t blockY : {4, 8}) {
            const int64_t tile = group <= 32 ? 512 : (group <= 64 ? 256 : 128);
            const int64_t boundary = 2 * tile / blockY;
            for (int64_t count : {boundary, boundary + 1}) {
                ScheduleInput input = MakeDecodeInput();
                input.quantType = 6;
                input.headDim = blockY == 4 ? 256 : 192;
                input.blockShapeY = blockY;
                input.numQHeads = group;
                input.blockIndexStride = 8193;
                input.validBlockNums = {count, 0};
                ScheduleResult result;
                ASSERT_EQ(BuildSchedule(input, result), BSAScheduleStatus::BSA_SUCCESS);
                EXPECT_EQ((result.fdScheduleFlags & FD_SCHEDULE_ENABLED) != 0, count > boundary);
            }
        }
    }
}
