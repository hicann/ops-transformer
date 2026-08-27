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

#include "../../../op_host/fused_gdn_decode_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace ge;
using namespace optiling;

namespace {

gert::TilingContextPara MakeContext(int64_t batch, int64_t qkHeads, int64_t valueHeads, int64_t keyDim,
                                    int64_t valueDim, ge::DataType inputDtype, ge::DataType stateDtype,
                                    FusedGdnDecodeCompileInfo *compileInfo)
{
    const int64_t mixedDim = 2 * qkHeads * keyDim + valueHeads * valueDim;
    constexpr int64_t stateSlots = 128;
    gert::StorageShape mixedShape = {{batch, mixedDim}, {batch, mixedDim}};
    gert::StorageShape gateShape = {{batch, valueHeads}, {batch, valueHeads}};
    gert::StorageShape paramShape = {{valueHeads}, {valueHeads}};
    gert::StorageShape stateShape = {{stateSlots, valueHeads, valueDim, keyDim},
                                     {stateSlots, valueHeads, valueDim, keyDim}};
    gert::StorageShape indexShape = {{batch}, {batch}};
    gert::StorageShape outShape = {{batch, 1, valueHeads, valueDim}, {batch, 1, valueHeads, valueDim}};
    return gert::TilingContextPara("FusedGdnDecode",
                                   {
                                       {mixedShape, inputDtype, ge::FORMAT_ND},
                                       {gateShape, inputDtype, ge::FORMAT_ND},
                                       {gateShape, inputDtype, ge::FORMAT_ND},
                                       {paramShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {paramShape, inputDtype, ge::FORMAT_ND},
                                       {stateShape, stateDtype, ge::FORMAT_ND},
                                       {indexShape, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       {outShape, inputDtype, ge::FORMAT_ND},
                                       {stateShape, stateDtype, ge::FORMAT_ND},
                                   },
                                   {
                                       {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838835f)},
                                       {"softplus_threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                   },
                                   compileInfo);
}

} // namespace

TEST(FusedGdnDecodeTilingTest, Bf16Fp32State)
{
    FusedGdnDecodeCompileInfo compileInfo = {48, 196608};
    auto context = MakeContext(8, 8, 16, 128, 128, ge::DT_BF16, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 1UL);
    constexpr uint32_t expectedPlatformAivNum = 64;
    EXPECT_EQ(tilingInfo.blockNum, expectedPlatformAivNum);
}

TEST(FusedGdnDecodeTilingTest, Fp16Fp32State)
{
    FusedGdnDecodeCompileInfo compileInfo = {48, 196608};
    auto context = MakeContext(8, 8, 16, 128, 128, ge::DT_FLOAT16, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 2UL);
    constexpr uint32_t expectedPlatformAivNum = 64;
    EXPECT_EQ(tilingInfo.blockNum, expectedPlatformAivNum);
}

TEST(FusedGdnDecodeTilingTest, Fp16Fp16State)
{
    FusedGdnDecodeCompileInfo compileInfo = {48, 196608};
    auto context = MakeContext(1, 16, 32, 128, 128, ge::DT_FLOAT16, ge::DT_FLOAT16, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 4UL);
    EXPECT_EQ(tilingInfo.blockNum, 32U);
}

TEST(FusedGdnDecodeTilingTest, RejectsSmallKeyDim)
{
    FusedGdnDecodeCompileInfo compileInfo = {48, 196608};
    auto context = MakeContext(1, 8, 16, 32, 128, ge::DT_BF16, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST(FusedGdnDecodeTilingTest, RejectsExcessiveKeyDim)
{
    FusedGdnDecodeCompileInfo compileInfo = {48, 196608};
    auto context = MakeContext(1, 8, 16, 2048, 128, ge::DT_BF16, ge::DT_BF16, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST(FusedGdnDecodeTilingTest, RejectsFp32StatePaddingOverLimit)
{
    FusedGdnDecodeCompileInfo compileInfo = {48, 196608};
    auto context = MakeContext(1, 8, 16, 129, 128, ge::DT_BF16, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST(FusedGdnDecodeTilingTest, AcceptsLowPrecisionStatePadding)
{
    FusedGdnDecodeCompileInfo compileInfo = {48, 196608};
    auto context = MakeContext(1, 8, 16, 129, 128, ge::DT_BF16, ge::DT_BF16, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 3UL);
}

TEST(FusedGdnDecodeTilingTest, RejectsUnsupportedStateDtype)
{
    FusedGdnDecodeCompileInfo compileInfo = {48, 196608};
    auto context = MakeContext(1, 8, 16, 128, 128, ge::DT_BF16, ge::DT_FLOAT16, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}
