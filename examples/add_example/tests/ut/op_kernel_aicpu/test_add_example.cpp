/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "utils/aicpu_read_file.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_AddExample_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas) \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "AddExampleAicpu", "AddExampleAicpu") \
        .Input({"x1", data_types[0], shapes[0], datas[0]}) \
        .Input({"x2", data_types[1], shapes[1], datas[1]}) \
        .Output({"y", data_types[2], shapes[2], datas[2]});

TEST_F(TEST_ADD_UT, INT32_VECTOR_ADD_SCALAR_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

    int32_t input1[2] = {2, 5};
    int32_t input2[1] = {3};
    int32_t output[2] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[2] = {5, 8};
    bool compare = CompareResult(output, output_exp, 2);
    EXPECT_EQ(compare, true);
}
