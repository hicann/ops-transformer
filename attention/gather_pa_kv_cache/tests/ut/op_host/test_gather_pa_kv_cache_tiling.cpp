/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/gather_pa_kv_cache_tiling.h"

using namespace std;
using namespace ge;

class GatherPaKvCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GatherPaKvCacheTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "GatherPaKvCacheTiling TearDown" << std::endl;
    }
};

TEST_F(GatherPaKvCacheTiling, GatherPaKvCacheTiling_test_case0)
{
    struct AddExampleCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GatherPaKvCache",
        {
            {{{30, 101, 128, 32}, {30, 101, 128, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // keyCache
            {{{30, 127, 128, 32}, {30, 127, 128, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // valueCache
            {{{36, 8}, {36, 8}}, ge::DT_INT32, ge::FORMAT_ND},                              // blocktables
            {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},                                    // seqLens
            {{{18933, 3232}, {18933, 3232}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},           // key
            {{{18933, 4064}, {18933, 4064}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},           // value
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                                      // seqOffset

        },
        {
            {{{18933, 3232}, {18933, 3232}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // key
            {{{18933, 4064}, {18933, 4064}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 577;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GatherPaKvCacheTiling, GatherPaKvCacheTiling_test_case1)
{
    struct AddExampleCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GatherPaKvCache",
        {
            {{{128, 128, 16, 144}, {128, 128, 16, 144}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // keyCache
            {{{128, 128, 16, 128}, {128, 128, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // valueCache
            {{{16, 12}, {16, 12}}, ge::DT_INT32, ge::FORMAT_ND},                         // blocktables
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                                 // seqLens
            {{{8931, 16, 144}, {8931, 16, 144}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // key
            {{{8931, 16, 128}, {8931, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // value
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                                 // seqOffset

        },
        {
            {{{8931, 16, 144}, {8931, 16, 144}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
            {{{8931, 16, 128}, {8931, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 619;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(GatherPaKvCacheTiling, GatherPaKvCacheTiling_test_case2)
{
    struct AddExampleCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GatherPaKvCache",
        {
            {{{128, 128, 16, 144}, {128, 128, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND}, // keyCache
            {{{128, 128, 16, 128}, {128, 128, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // valueCache
            {{{16, 12}, {16, 12}}, ge::DT_INT32, ge::FORMAT_ND},                      // blocktables
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                              // seqLens
            {{{9547, 16, 144}, {9547, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND},         // key
            {{{9547, 16, 128}, {9547, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},         // value
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                              // seqOffset

        },
        {
            {{{9547, 16, 144}, {9547, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{9547, 16, 128}, {9547, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 618;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
