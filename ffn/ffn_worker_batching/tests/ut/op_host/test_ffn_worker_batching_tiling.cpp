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
#include "../../../op_host/ffn_worker_batching_tiling.h"

using namespace ge;
using namespace optiling;

class FfnWorkerBatchingTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FfnWorkerBatchingTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FfnWorkerBatchingTilingTest TearDown" << std::endl;
    }
};

TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::StorageShape y_shape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape group_list_shape = {{8, 2}, {8, 2}};
    gert::StorageShape session_ids_shape = {{1024}, {1024}};
    gert::StorageShape micro_batch_ids_shape = {{1024}, {1024}};
    gert::StorageShape token_ids_shape = {{1024}, {1024}};
    gert::StorageShape expert_offsets_shape = {{1024}, {1024}};
    gert::StorageShape dynamic_scale_shape = {{1024}, {1024}};
    gert::StorageShape actual_token_num_shape = {{1}, {1}};

    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {y_shape, ge::DT_INT8, ge::FORMAT_ND},                 // y
            {group_list_shape, ge::DT_INT64, ge::FORMAT_ND},       // group_list
            {session_ids_shape, ge::DT_INT32, ge::FORMAT_ND},      // session_ids
            {micro_batch_ids_shape, ge::DT_INT32, ge::FORMAT_ND},  // micro_batch_ids
            {token_ids_shape, ge::DT_INT32, ge::FORMAT_ND},        // token_ids
            {expert_offsets_shape, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {dynamic_scale_shape, ge::DT_FLOAT, ge::FORMAT_ND},    // dynamic_scale
            {actual_token_num_shape, ge::DT_INT64, ge::FORMAT_ND}, // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo);
    int64_t expectTilingKey = 101;
    std::string expectTilingData = "1152 4096 0 8 32 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
