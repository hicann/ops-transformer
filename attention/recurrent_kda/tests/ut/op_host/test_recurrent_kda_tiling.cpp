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
 * \file test_recurrent_kda_tiling_processor.cpp
 * \brief RecurrentKda tiling processor unit tests.
 */

#include <gtest/gtest.h>

#include "../../../op_host/recurrent_kda_tiling_processor.h"

namespace {
using optiling::RecurrentKdaTilingContext;
using optiling::RecurrentKdaTilingProcessor;
using RecurrentKdaTilingData = RecurrentKda::RecurrentKdaTilingData;

// 910B (A2) platform parameters used by every case.
constexpr uint64_t UT_UB_SIZE = 196608UL;
constexpr uint64_t UT_AIV_NUM = 40UL;

constexpr int64_t UT_BATCH = 1;
constexpr int64_t UT_SEQ_LEN = 8;
constexpr int64_t UT_H = 2;
constexpr int64_t UT_HV = 4;
constexpr int64_t UT_K = 128;
constexpr int64_t UT_V = 128;

// Without ssm_state_indices the state pool holds one slot per sequence, i.e. one per batch entry.
RecurrentKdaTilingContext MakeBsndContext(int64_t b, int64_t t, int64_t h, int64_t hv, int64_t k, int64_t v,
                                          int64_t stateCapacity = 0)
{
    RecurrentKdaTilingContext ctx;
    ctx.queryShape = gert::Shape({b, t, h, k});
    ctx.keyShape = ctx.queryShape;
    ctx.valueShape = gert::Shape({b, t, hv, v});
    ctx.gateShape = gert::Shape({b, t, hv, k});
    ctx.betaShape = gert::Shape({b, t, hv});
    ctx.stateShape = gert::Shape({stateCapacity > 0 ? stateCapacity : b, hv, k, v});
    ctx.stateDtype = ge::DT_FLOAT;
    ctx.aivNum = UT_AIV_NUM;
    ctx.ubSize = UT_UB_SIZE;
    return ctx;
}

RecurrentKdaTilingContext MakeDefaultContext()
{
    return MakeBsndContext(UT_BATCH, UT_SEQ_LEN, UT_H, UT_HV, UT_K, UT_V);
}

ge::graphStatus RunTiling(const RecurrentKdaTilingContext& ctx, RecurrentKdaTilingData& tiling)
{
    RecurrentKdaTilingProcessor processor(ctx);
    const auto ret = processor.ProcessShapes(tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return processor.ProcessUb(tiling);
}

TEST(RecurrentKdaTilingProcessor, BsndShapeDataAndCores)
{
    RecurrentKdaTilingData tiling = {};
    const auto ctx = MakeDefaultContext();
    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);

    EXPECT_EQ(tiling.t, static_cast<uint32_t>(UT_BATCH * UT_SEQ_LEN));
    EXPECT_EQ(tiling.seqLen, static_cast<uint32_t>(UT_SEQ_LEN));
    EXPECT_EQ(tiling.nk, static_cast<uint32_t>(UT_H));
    EXPECT_EQ(tiling.dk, static_cast<uint32_t>(UT_K));
    EXPECT_EQ(tiling.nv, static_cast<uint32_t>(UT_HV));
    EXPECT_EQ(tiling.dv, static_cast<uint32_t>(UT_V));
    EXPECT_EQ(tiling.b, static_cast<uint32_t>(UT_BATCH));
    EXPECT_EQ(tiling.sBlockNum, static_cast<uint32_t>(UT_BATCH));
    EXPECT_EQ(tiling.ssmStateStride, 0U);
    EXPECT_EQ(tiling.layout, optiling::RKDA_LAYOUT_BSND);
    EXPECT_EQ(tiling.gateDtype, 0U); // fp32
    EXPECT_EQ(tiling.betaDtype, 0U); // fp32
    // Dense [slot, HV, K, V] state pool strides.
    EXPECT_EQ(tiling.stateInStride0, static_cast<uint64_t>(UT_HV * UT_K * UT_V));
    EXPECT_EQ(tiling.stateInStride1, static_cast<uint64_t>(UT_K * UT_V));
    EXPECT_EQ(tiling.stateInStride2, static_cast<uint64_t>(UT_V));
    EXPECT_EQ(tiling.stateInStride3, 1U);
    EXPECT_EQ(tiling.stateOutStride0, tiling.stateInStride0);
    // Four task units on 40 vector cores collapse onto four cores on A2/A3.
    EXPECT_EQ(tiling.vectorCoreNum, static_cast<uint32_t>(UT_BATCH * UT_HV));
}

TEST(RecurrentKdaTilingProcessor, A2BalancesVectorCoresByTaskUnits)
{
    RecurrentKdaTilingData tiling = {};
    const auto ctx = MakeBsndContext(3, 4, 8, 48, UT_K, UT_V);
    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);
    // 144 task units over 40 cores: four units per core, 36 fully occupied cores.
    EXPECT_EQ(tiling.vectorCoreNum, 36U);
}

TEST(RecurrentKdaTilingProcessor, A5KeepsSingleWaveCores)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeBsndContext(3, 4, 8, 48, UT_K, UT_V);
    ctx.isA5 = 1;
    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.vectorCoreNum, static_cast<uint32_t>(UT_AIV_NUM));
}

TEST(RecurrentKdaTilingProcessor, TndWithCuSeqlens)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeBsndContext(1, 16, UT_H, UT_HV, UT_K, UT_V);
    ctx.layout = optiling::RKDA_LAYOUT_TND;
    ctx.queryShape = gert::Shape({16, UT_H, UT_K});
    ctx.keyShape = ctx.queryShape;
    ctx.valueShape = gert::Shape({16, UT_HV, UT_V});
    ctx.gateShape = gert::Shape({16, UT_HV, UT_K});
    ctx.betaShape = gert::Shape({16, UT_HV});
    ctx.stateShape = gert::Shape({2, UT_HV, UT_K, UT_V});
    ctx.hasCuSeqlens = 1;
    ctx.cuSeqlensShape = gert::Shape({3});

    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.layout, optiling::RKDA_LAYOUT_TND);
    EXPECT_EQ(tiling.t, 16U);
    EXPECT_EQ(tiling.seqLen, 16U);
    EXPECT_EQ(tiling.b, 2U); // cu_seqlens holds three offsets.
    EXPECT_EQ(tiling.hasCuSeqlens, 1U);
}

TEST(RecurrentKdaTilingProcessor, StateVFirstWithWideV)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeBsndContext(UT_BATCH, UT_SEQ_LEN, UT_H, UT_HV, UT_K, 256);
    ctx.stateVFirst = 1;
    ctx.stateShape = gert::Shape({UT_BATCH, UT_HV, 256, UT_K});

    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.stateVFirst, 1U);
    EXPECT_EQ(tiling.dv, 256U);
    EXPECT_EQ(tiling.stateInStride1, static_cast<uint64_t>(256 * UT_K));
    EXPECT_EQ(tiling.stateInStride0, static_cast<uint64_t>(UT_HV * 256 * UT_K));
}

TEST(RecurrentKdaTilingProcessor, PaddedStatePoolStridesAreHonored)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    // Slot and head strides carry padding; the inner K/V matrix stays dense.
    ctx.hasStateInStrides = true;
    ctx.stateInStrides = {67584, 16896, UT_V, 1};

    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.stateInStride0, 67584U);
    EXPECT_EQ(tiling.stateInStride1, 16896U);
    EXPECT_EQ(tiling.stateInStride2, static_cast<uint64_t>(UT_V));
    EXPECT_EQ(tiling.stateInStride3, 1U);
}

TEST(RecurrentKdaTilingProcessor, RejectsInvalidStateStrides)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.hasStateOutStrides = true;
    ctx.stateOutStrides = {65536, 16384, UT_V, 2}; // stride[3] must be 1.

    RecurrentKdaTilingProcessor processor(ctx);
    EXPECT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_FAILED);
}

TEST(RecurrentKdaTilingProcessor, RejectsUnsupportedK)
{
    RecurrentKdaTilingData tiling = {};
    const auto ctx = MakeBsndContext(UT_BATCH, UT_SEQ_LEN, UT_H, UT_HV, 64, UT_V);
    RecurrentKdaTilingProcessor processor(ctx);
    EXPECT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_FAILED);
}

TEST(RecurrentKdaTilingProcessor, RejectsUnsupportedV)
{
    RecurrentKdaTilingData tiling = {};
    const auto ctx = MakeBsndContext(UT_BATCH, UT_SEQ_LEN, UT_H, UT_HV, UT_K, 192);
    RecurrentKdaTilingProcessor processor(ctx);
    EXPECT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_FAILED);
}

TEST(RecurrentKdaTilingProcessor, RejectsHvNotMultipleOfH)
{
    RecurrentKdaTilingData tiling = {};
    const auto ctx = MakeBsndContext(UT_BATCH, UT_SEQ_LEN, 4, 6, UT_K, UT_V);
    RecurrentKdaTilingProcessor processor(ctx);
    EXPECT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_FAILED);
}

TEST(RecurrentKdaTilingProcessor, RejectsStateCapacityMismatch)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.stateShape = gert::Shape({2 * UT_BATCH, UT_HV, UT_K, UT_V});
    RecurrentKdaTilingProcessor processor(ctx);
    EXPECT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_FAILED);
}

TEST(RecurrentKdaTilingProcessor, RejectsSafeGateWithNonNegativeLowerBound)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.safeGate = 1;
    ctx.lowerBound = 0.5F;
    RecurrentKdaTilingProcessor processor(ctx);
    EXPECT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_FAILED);
}

TEST(RecurrentKdaTilingProcessor, AcceptsPerTokenPoolWithPackedSsmIndices)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.hasSsmStateIndices = 1;
    ctx.ssmStateShape = gert::Shape({UT_BATCH * UT_SEQ_LEN});
    // Packed indices let the pool hold one slot per token instead of one per sequence.
    ctx.stateShape = gert::Shape({UT_BATCH * UT_SEQ_LEN, UT_HV, UT_K, UT_V});

    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.hasSsmStateIndices, 1U);
    EXPECT_EQ(tiling.sBlockNum, static_cast<uint32_t>(UT_BATCH * UT_SEQ_LEN));
    EXPECT_EQ(tiling.ssmStateStride, 0U);
}

TEST(RecurrentKdaTilingProcessor, SpeculativeSsmIndicesFillStateStride)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.hasSsmStateIndices = 1;
    ctx.ssmStateShape = gert::Shape({UT_BATCH, 3});

    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.ssmStateStride, 3U);
}

TEST(RecurrentKdaTilingProcessor, RejectsGateInKernelWithoutALog)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.useGateInKernel = 1;
    ctx.hasALog = 0;
    RecurrentKdaTilingProcessor processor(ctx);
    EXPECT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_FAILED);
}

TEST(RecurrentKdaTilingProcessor, AcceptsGateInKernelWithALog)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.useGateInKernel = 1;
    ctx.hasALog = 1;
    ctx.aLogShape = gert::Shape({UT_HV});
    RecurrentKdaTilingProcessor processor(ctx);
    ASSERT_EQ(processor.ProcessShapes(tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.useGateInKernel, 1U);
    EXPECT_EQ(tiling.hasALog, 1U);
}

TEST(RecurrentKdaTilingProcessor, A2UbModelSelectsVStep)
{
    RecurrentKdaTilingData tiling = {};
    const auto ctx = MakeDefaultContext();
    ASSERT_EQ(RunTiling(ctx, tiling), ge::GRAPH_SUCCESS);

    EXPECT_EQ(tiling.ubCalSize, static_cast<uint32_t>(UT_UB_SIZE));
    // K=128, V=128, fp32 K-first state: the double-buffered attn output wins the profile search.
    EXPECT_EQ(tiling.stateOutBufferNum, 1U);
    EXPECT_EQ(tiling.attnOutBufferNum, 2U);
    EXPECT_EQ(tiling.vStep, 48U);
    EXPECT_EQ(tiling.ubRestBytes, 66048U);
    EXPECT_LT(tiling.ubRestBytes, tiling.ubCalSize);
}

TEST(RecurrentKdaTilingProcessor, A5UbModelSelectsVStep)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.isA5 = 1;
    ASSERT_EQ(RunTiling(ctx, tiling), ge::GRAPH_SUCCESS);

    // One raw input buffer and the fused broadTmp reduce leave room for a wider vStep.
    EXPECT_EQ(tiling.stateOutBufferNum, 1U);
    EXPECT_EQ(tiling.attnOutBufferNum, 2U);
    EXPECT_EQ(tiling.vStep, 64U);
    EXPECT_EQ(tiling.ubRestBytes, 52416U);
}

TEST(RecurrentKdaTilingProcessor, RejectsUbTooSmall)
{
    RecurrentKdaTilingData tiling = {};
    auto ctx = MakeDefaultContext();
    ctx.ubSize = 32768;
    EXPECT_EQ(RunTiling(ctx, tiling), ge::GRAPH_FAILED);
}
} // namespace
