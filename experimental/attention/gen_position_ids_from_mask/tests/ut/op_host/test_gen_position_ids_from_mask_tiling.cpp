/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include <string>

#include "../../../op_host/gen_position_ids_from_mask_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class GenPositionIdsFromMaskTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- GenPositionIdsFromMask Tiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- GenPositionIdsFromMask Tiling UT TearDown ---" << std::endl;
    }
};

// int32 mask -> tiling key 1
TEST_F(GenPositionIdsFromMaskTilingTest, tiling_int32)
{
    struct GenPositionIdsFromMaskCompileInfo {
    } compileInfo;

    int64_t b = 4, s = 128;

    gert::TilingContextPara tilingContextPara(
        "GenPositionIdsFromMask", {{{{b, s}, {b, s}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{b, s}, {b, s}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}}, &compileInfo, "Ascend910B", 64,
        262144, 16384);

    uint64_t expectTilingKey = 1UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// int64 mask -> tiling key 2
TEST_F(GenPositionIdsFromMaskTilingTest, tiling_int64)
{
    struct GenPositionIdsFromMaskCompileInfo {
    } compileInfo;

    int64_t b = 4, s = 128;

    gert::TilingContextPara tilingContextPara(
        "GenPositionIdsFromMask", {{{{b, s}, {b, s}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{b, s}, {b, s}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}}, &compileInfo, "Ascend910B", 64,
        262144, 16384);

    uint64_t expectTilingKey = 2UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// bool mask -> tiling key 3
TEST_F(GenPositionIdsFromMaskTilingTest, tiling_bool)
{
    struct GenPositionIdsFromMaskCompileInfo {
    } compileInfo;

    int64_t b = 4, s = 128;

    gert::TilingContextPara tilingContextPara(
        "GenPositionIdsFromMask", {{{{b, s}, {b, s}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{b, s}, {b, s}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}}, &compileInfo, "Ascend910B", 64,
        262144, 16384);

    uint64_t expectTilingKey = 3UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 非 2D 输入 -> tiling 失败
TEST_F(GenPositionIdsFromMaskTilingTest, tiling_invalid_3d_fail)
{
    struct GenPositionIdsFromMaskCompileInfo {
    } compileInfo;

    int64_t b = 4, s = 128, k = 2;

    gert::TilingContextPara tilingContextPara(
        "GenPositionIdsFromMask", {{{{b, s, k}, {b, s, k}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{b, s, k}, {b, s, k}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}}, &compileInfo, "Ascend910B", 64,
        262144, 16384);

    // shape 维度非 2 时 TilingGenPositionIdsFromMask 返回 GRAPH_FAILED
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0UL);
}

// 非默认 fillValue 也能正常出 tiling
TEST_F(GenPositionIdsFromMaskTilingTest, tiling_fill_value_neg)
{
    struct GenPositionIdsFromMaskCompileInfo {
    } compileInfo;

    int64_t b = 8, s = 64;

    gert::TilingContextPara tilingContextPara(
        "GenPositionIdsFromMask", {{{{b, s}, {b, s}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{b, s}, {b, s}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"padding_fill_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}}, &compileInfo, "Ascend910B", 64,
        262144, 16384);

    uint64_t expectTilingKey = 1UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
