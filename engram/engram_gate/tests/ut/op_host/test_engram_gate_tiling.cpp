/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>
#include "../../../op_graph/engram_gate_proto.h"
#include <limits>
#include "tiling_case_executor.h"
#include "test_engram_gate_utils.h"
#include "../../../op_kernel/arch35/engram_gate_tiling_data.h"

static EngramGateTilingData GetTilingData(const gert::TilingContextPara& para)
{
    TilingInfo info;
    if (!ExecuteTiling(para, info)) {
        ADD_FAILURE() << "ExecuteTiling failed";
        return {};
    }
    EngramGateTilingData data{};
    if (info.tilingDataSize < sizeof(data)) {
        ADD_FAILURE() << "Tiling data is shorter than EngramGateTilingData";
        return {};
    }
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    return data;
}

static gert::TilingContextPara Make(int64_t t = 17, int64_t h = 4, int64_t d = 64, ge::DataType dtype = ge::DT_BF16,
                                    bool save = true)
{
    static int compileInfo = 0;
    auto stat = save ? MakeStorageShape({t, h}) : MakeStorageShape({0});
    return gert::TilingContextPara("EngramGate",
                                   {{MakeStorageShape({t, h, d}), dtype, ge::FORMAT_ND},
                                    {MakeStorageShape({t, (h + 1) * d}), dtype, ge::FORMAT_ND},
                                    {MakeStorageShape({h, d}), ge::DT_FLOAT, ge::FORMAT_ND}},
                                   {{MakeStorageShape({t, h, d}), dtype, ge::FORMAT_ND},
                                    {stat, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {stat, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {stat, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {stat, ge::DT_FLOAT, ge::FORMAT_ND}},
                                   {{"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
                                    {"clamp_value", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
                                    {"save_for_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(save)}},
                                   &compileInfo, "Ascend950");
}

TEST(EngramGateTiling, Valid_0_BF16_true)
{
    ExecuteTestCase(Make(0, 4, 64, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_0_BF16_false)
{
    ExecuteTestCase(Make(0, 4, 64, ge::DT_BF16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_0_FLOAT16_true)
{
    ExecuteTestCase(Make(0, 4, 64, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_0_FLOAT16_false)
{
    ExecuteTestCase(Make(0, 4, 64, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_1_BF16_true)
{
    ExecuteTestCase(Make(1, 4, 64, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_1_BF16_false)
{
    ExecuteTestCase(Make(1, 4, 64, ge::DT_BF16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_1_FLOAT16_true)
{
    ExecuteTestCase(Make(1, 4, 64, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_1_FLOAT16_false)
{
    ExecuteTestCase(Make(1, 4, 64, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_2_BF16_true)
{
    ExecuteTestCase(Make(2, 4, 5120, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_2_BF16_false)
{
    ExecuteTestCase(Make(2, 4, 5120, ge::DT_BF16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_2_FLOAT16_true)
{
    ExecuteTestCase(Make(2, 4, 5120, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_2_FLOAT16_false)
{
    ExecuteTestCase(Make(2, 4, 5120, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_3_BF16_true)
{
    ExecuteTestCase(Make(17, 4, 32768, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_3_BF16_false)
{
    ExecuteTestCase(Make(17, 4, 32768, ge::DT_BF16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_3_FLOAT16_true)
{
    ExecuteTestCase(Make(17, 4, 32768, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_3_FLOAT16_false)
{
    ExecuteTestCase(Make(17, 4, 32768, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_4_BF16_true)
{
    ExecuteTestCase(Make(33, 4, 4096, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_4_BF16_false)
{
    ExecuteTestCase(Make(33, 4, 4096, ge::DT_BF16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_4_FLOAT16_true)
{
    ExecuteTestCase(Make(33, 4, 4096, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_4_FLOAT16_false)
{
    ExecuteTestCase(Make(33, 4, 4096, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_5_BF16_true)
{
    ExecuteTestCase(Make(2, 4, 8192, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_5_BF16_false)
{
    ExecuteTestCase(Make(2, 4, 8192, ge::DT_BF16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_5_FLOAT16_true)
{
    ExecuteTestCase(Make(2, 4, 8192, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_5_FLOAT16_false)
{
    ExecuteTestCase(Make(2, 4, 8192, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_6_BF16_true)
{
    ExecuteTestCase(Make(4096, 4, 5120, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_6_BF16_false)
{
    ExecuteTestCase(Make(4096, 4, 5120, ge::DT_BF16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_6_FLOAT16_true)
{
    ExecuteTestCase(Make(4096, 4, 5120, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_6_FLOAT16_false)
{
    ExecuteTestCase(Make(4096, 4, 5120, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_7_BF16_true)
{
    ExecuteTestCase(Make(1, 4, 16384, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_7_BF16_false)
{
    ExecuteTestCase(Make(1, 4, 16384, ge::DT_BF16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_7_FLOAT16_true)
{
    ExecuteTestCase(Make(1, 4, 16384, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, Valid_7_FLOAT16_false)
{
    ExecuteTestCase(Make(1, 4, 16384, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}
TEST(EngramGateTiling, RejectRank)
{
    auto p = Make();
    p.inputTensorDesc_[0].shape_ = MakeStorageShape({17, 195});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectHZero)
{
    ExecuteTestCase(Make(17, 0, 64), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectHBelowSupported)
{
    ExecuteTestCase(Make(17, 3, 64), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectHAboveSupported)
{
    ExecuteTestCase(Make(17, 5, 64), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectDZero)
{
    ExecuteTestCase(Make(17, 4, 0), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectDBelowAlignment)
{
    ExecuteTestCase(Make(17, 4, 63), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectDNotMultipleOf64)
{
    ExecuteTestCase(Make(17, 4, 65), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectBothZero)
{
    ExecuteTestCase(Make(0, 0, 0), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectNegativeT)
{
    ExecuteTestCase(Make(-1, 4, 64), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectNegativeH)
{
    ExecuteTestCase(Make(17, -1, 64), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectNegativeD)
{
    ExecuteTestCase(Make(17, 4, -64), ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAddressOverflow)
{
    auto p = Make();
    p.inputTensorDesc_[0].shape_ = MakeStorageShape({std::numeric_limits<int64_t>::max(), 4, 64});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectKvShape)
{
    auto p = Make();
    p.inputTensorDesc_[1].shape_ = MakeStorageShape({17, 319});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectWeightShape)
{
    auto p = Make();
    p.inputTensorDesc_[2].shape_ = MakeStorageShape({4, 63});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectYShape)
{
    auto p = Make();
    p.outputTensorDesc_[0].shape_ = MakeStorageShape({17, 4, 128});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectXDtype)
{
    auto p = Make();
    p.inputTensorDesc_[0].dtype_ = ge::DT_FLOAT;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectKvDtype)
{
    auto p = Make();
    p.inputTensorDesc_[1].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectWDtype)
{
    auto p = Make();
    p.inputTensorDesc_[2].dtype_ = ge::DT_BF16;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectYDtype)
{
    auto p = Make();
    p.outputTensorDesc_[0].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectNoCores)
{
    auto p = Make();
    p.coreNum_ = 0;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectSmallUB)
{
    auto p = Make();
    p.ubSize_ = EngramGateTiling::CHUNK_FIXED_BYTES +
                EngramGateTiling::VECTOR_LANES * EngramGateTiling::CHUNK_STREAM_BYTES_PER_ELEMENT - 1;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectStats1Shape)
{
    auto p = Make();
    p.outputTensorDesc_[1].shape_ = MakeStorageShape({17, 3});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectStats1Dtype)
{
    auto p = Make();
    p.outputTensorDesc_[1].dtype_ = ge::DT_BF16;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectStats2Shape)
{
    auto p = Make();
    p.outputTensorDesc_[2].shape_ = MakeStorageShape({17, 3});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectStats2Dtype)
{
    auto p = Make();
    p.outputTensorDesc_[2].dtype_ = ge::DT_BF16;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectStats3Shape)
{
    auto p = Make();
    p.outputTensorDesc_[3].shape_ = MakeStorageShape({17, 3});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectStats3Dtype)
{
    auto p = Make();
    p.outputTensorDesc_[3].dtype_ = ge::DT_BF16;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectStats4Shape)
{
    auto p = Make();
    p.outputTensorDesc_[4].shape_ = MakeStorageShape({17, 3});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectStats4Dtype)
{
    auto p = Make();
    p.outputTensorDesc_[4].dtype_ = ge::DT_BF16;
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAttr0Zero)
{
    auto p = Make();
    p.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(0.0F);
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAttr0Negative)
{
    auto p = Make();
    p.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(-1.0F);
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAttr0Nan)
{
    auto p = Make();
    p.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::quiet_NaN());
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAttr0Inf)
{
    auto p = Make();
    p.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::infinity());
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAttr1Zero)
{
    auto p = Make();
    p.attrs_[1].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(0.0F);
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAttr1Negative)
{
    auto p = Make();
    p.attrs_[1].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(-1.0F);
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAttr1Nan)
{
    auto p = Make();
    p.attrs_[1].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::quiet_NaN());
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, RejectAttr1Inf)
{
    auto p = Make();
    p.attrs_[1].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(std::numeric_limits<float>::infinity());
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
TEST(EngramGateTiling, MinimalUB)
{
    auto p = Make();
    p.ubSize_ = EngramGateTiling::CHUNK_FIXED_BYTES +
                EngramGateTiling::VECTOR_LANES * EngramGateTiling::CHUNK_STREAM_BYTES_PER_ELEMENT;
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, 0UL);
}

TEST(EngramGateTiling, RejectTHZero)
{
    auto p = Make();
    p.inputTensorDesc_[0].shape_ = MakeStorageShape({0, 0, 64});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

TEST(EngramGateTiling, RejectTDZero)
{
    auto p = Make();
    p.inputTensorDesc_[0].shape_ = MakeStorageShape({0, 4, 0});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

TEST(EngramGateTiling, RejectHDZero)
{
    auto p = Make();
    p.inputTensorDesc_[0].shape_ = MakeStorageShape({17, 0, 0});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

TEST(EngramGateTiling, RejectRankOneEmpty)
{
    auto p = Make();
    p.inputTensorDesc_[0].shape_ = MakeStorageShape({0});
    ExecuteTestCase(p, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

TEST(EngramGateTiling, ChunkedAlignedHeads)
{
    ExecuteTestCase(Make(257, 4, 64, ge::DT_BF16, true), ge::GRAPH_SUCCESS, 0UL);
}

TEST(EngramGateTiling, ChunkedLongRow)
{
    ExecuteTestCase(Make(257, 4, 32768, ge::DT_FLOAT16, true), ge::GRAPH_SUCCESS, 0UL);
}

TEST(EngramGateTiling, ChunkedNoStats)
{
    ExecuteTestCase(Make(257, 4, 512, ge::DT_FLOAT16, false), ge::GRAPH_SUCCESS, 0UL);
}

TEST(EngramGateTiling, PathDispatchEmpty)
{
    auto data = GetTilingData(Make(0, 4, 5120, ge::DT_BF16, true));
    EXPECT_EQ(data.pureSlots, 0U);
    EXPECT_EQ(data.chunkedEnabled, 0U);
}

TEST(EngramGateTiling, PathDispatchPure)
{
    auto data = GetTilingData(Make(2, 4, 8192, ge::DT_BF16, true));
    EXPECT_NE(data.pureSlots, 0U);
    EXPECT_EQ(data.chunkedEnabled, 0U);
    EXPECT_EQ(data.pureBypass, 0U);
}

TEST(EngramGateTiling, PathDispatchChunked)
{
    auto data = GetTilingData(Make(17, 4, 32768, ge::DT_BF16, true));
    EXPECT_EQ(data.pureSlots, 0U);
    EXPECT_EQ(data.chunkedEnabled, 1U);
    EXPECT_EQ(data.weightResident, 0U);
    EXPECT_GT(data.chunk, 4096U);
    EXPECT_LE((32768U + data.chunk - 1U) / data.chunk, 6U);
    EXPECT_EQ(data.chunk % EngramGateTiling::VECTOR_LANES, 0U);
    EXPECT_LE(EngramGateTiling::CHUNK_FIXED_BYTES +
                  static_cast<uint64_t>(data.chunk) * EngramGateTiling::CHUNK_STREAM_BYTES_PER_ELEMENT,
              data.ubLayoutBytes);
}

TEST(EngramGateTiling, PathDispatchPureBypass)
{
    auto data = GetTilingData(Make(4096, 4, 5120, ge::DT_BF16, true));
    EXPECT_NE(data.pureSlots, 0U);
    EXPECT_EQ(data.chunkedEnabled, 0U);
    EXPECT_EQ(data.pureBypass, 1U);
}

TEST(EngramGateTiling, PathDispatchPureBypassNoStatsFloat16)
{
    auto data = GetTilingData(Make(4096, 4, 5120, ge::DT_FLOAT16, false));
    EXPECT_NE(data.pureSlots, 0U);
    EXPECT_EQ(data.chunkedEnabled, 0U);
    EXPECT_EQ(data.pureBypass, 1U);
}

TEST(EngramGateTiling, NetworkShapeUsesDoubleBuffer)
{
    auto data = GetTilingData(Make(512, 4, 5120, ge::DT_BF16, true));
    EXPECT_EQ(data.pureSlots, 2U);
    EXPECT_GE(data.pureRowsPerSlot, 2U);
}

TEST(EngramGateTiling, PairPathUsesEvenCoreCount)
{
    auto p = Make(17, 4, 5120, ge::DT_BF16, true);
    p.coreNum_ = 3;
    auto data = GetTilingData(p);
    EXPECT_EQ(data.pureRowsPerSlot, 2U);
    EXPECT_EQ(data.usedCores, 2U);
    EXPECT_EQ(data.usedCores % 2U, 0U);
}

TEST(EngramGateTiling, PureLaunchCoresFollowBatchCount)
{
    auto tiny = GetTilingData(Make(1, 4, 64, ge::DT_BF16, false));
    EXPECT_EQ(tiny.pureSlots, 2U);
    EXPECT_EQ(tiny.usedCores, 1U);
    EXPECT_LE(tiny.pureRowsPerSlot, 32U);

    auto large = GetTilingData(Make(512, 4, 64, ge::DT_BF16, false));
    EXPECT_EQ(large.pureSlots, 2U);
    EXPECT_EQ(large.usedCores, 64U);
    EXPECT_EQ(large.pureRowsPerSlot, 32U);
}

TEST(EngramGateTiling, ChunkedResidentWeightUsesAtMostOneThirdUb)
{
    auto data = GetTilingData(Make(17, 4, 16384, ge::DT_BF16, true));
    EXPECT_EQ(data.pureSlots, 0U);
    EXPECT_EQ(data.chunkedEnabled, 1U);
    EXPECT_EQ(data.weightResident, 1U);
    const uint64_t weightBytes = 16384U * sizeof(float);
    EXPECT_LE(weightBytes, data.ubLayoutBytes / EngramGateTiling::WEIGHT_RESIDENT_UB_DIVISOR);
    EXPECT_EQ(data.chunk % EngramGateTiling::VECTOR_LANES, 0U);
    EXPECT_LE(EngramGateTiling::CHUNK_FIXED_BYTES + weightBytes +
                  static_cast<uint64_t>(data.chunk) * EngramGateTiling::CHUNK_RESIDENT_BYTES_PER_ELEMENT,
              data.ubLayoutBytes);
}

TEST(EngramGateTiling, AdaptiveChunkLargeDim)
{
    auto data = GetTilingData(Make(17, 4, 65536, ge::DT_FLOAT16, true));
    EXPECT_EQ(data.pureSlots, 0U);
    EXPECT_EQ(data.chunkedEnabled, 1U);
    EXPECT_EQ(data.weightResident, 0U);
    EXPECT_GT(data.chunk, 4096U);
    EXPECT_EQ(data.chunk % EngramGateTiling::VECTOR_LANES, 0U);
    EXPECT_LE(EngramGateTiling::CHUNK_FIXED_BYTES +
                  static_cast<uint64_t>(data.chunk) * EngramGateTiling::CHUNK_STREAM_BYTES_PER_ELEMENT,
              data.ubLayoutBytes);
}
