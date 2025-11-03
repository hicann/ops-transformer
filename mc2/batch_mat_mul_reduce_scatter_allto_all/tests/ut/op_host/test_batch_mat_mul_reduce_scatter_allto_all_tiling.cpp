/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <float.h>
#include <array>
#include <gtest/gtest.h>
#include <opdev/platform.h>
#include <gmock/gmock.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class BatchMatMulReduceScatterAlltoAllTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "BatchMatMulReduceScatterAlltoAllTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "BatchMatMulReduceScatterAlltoAllTiling TearDown" << std::endl;
    }
};

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_1) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 1000000000000001001;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_1_weight_trans) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 128, 64}, {2, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 1000000000000001011;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_M_0) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 0}, {2, 1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 0, 128}, {2, 0, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 1000000000000001001;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 1000000000000001101;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_0) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test1) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test2) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test3) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{1, 1024, 64}, {1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test4) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128, 1}, {2, 1, 128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test5) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test6) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{1024, 64}, {1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test7) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 0}, {2, 1024, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test8) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 1, 128}, {3, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard0_with_bias) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 1000000000000000100;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard0_nonlocalE_tail_front) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{17, 3868, 637}, {17, 3868, 637}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{17, 637, 2366}, {17, 637, 2366}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{17, 1, 1183}, {17, 1, 1183}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{68, 967, 1183}, {68, 967, 1183}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 1000000000000000100;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_bf16_shard0_with_bias) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_BF16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 1000000000000000100;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard0_with_bias_invalid_Xshape) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1020, 64}, {2, 1020, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard1_with_bias_invalid_Xshape) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1020, 64}, {2, 1020, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(BatchMatMulReduceScatterAlltoAllTiling, batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard0_with_bias_invalid_H) {
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("BatchMatMulReduceScatterAlltoAll",
        {
            {{{2, 1024, 64}, {2, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 128}, {2, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}