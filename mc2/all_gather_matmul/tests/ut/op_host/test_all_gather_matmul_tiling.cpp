/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_kernel/all_gather_matmul_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace AllGatherMatmulUT {

class AllGatherMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AllGatherMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AllGatherMatmulTiling TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float16_1) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{512, 12288}, {512, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288, 3904}, {12288, 3904}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{512, 3904}, {512, 3904}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 12288}, {512, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );

    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float16_2) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2048, 1536}, {2048, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );

    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float16_3) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{327680, 15360}, {327680, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15360, 10240}, {15360, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{327680, 10240}, {327680, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{327680, 15360}, {327680, 15360}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );

    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_bfloat16) {
    // tilingFunc simulate
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 1536}, {4096, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{2048, 1536}, {2048, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );

    uint64_t expectTilingKey = 111UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_float16_l2cache) {
    // tilingFunc simulate
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{8192, 5120}, {8192, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 12288}, {5120, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12288}, {12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{8192, 12288}, {8192, 12288}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5120}, {8192, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );

    uint64_t expectTilingKey = 111UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_tiling_n_0) {
    struct AllGatherMatmulCompileInfo {} compileInfo;

    gert::TilingContextPara tilingContextPara("AllGatherMatmul",
        {
            {{{1024, 256}, {1024, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 0}, {256, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_STRING, ge::FORMAT_ND},
        },
        {
            {{{1024, 0}, {1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 256}, {1024, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo
    );

    uint64_t expectTilingKey = 110UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

} // AllGatherMatmulUT