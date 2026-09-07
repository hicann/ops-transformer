/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace AlltoAllAllGatherBmmInfershapeUT {
class AlltoAllAllGatherBmmInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllAllGatherBmmInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllAllGatherBmmInfershape TearDown" << std::endl;
    }
};

// infer shape with bias, success
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape0)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 16, 512}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

TEST_F(AlltoAllAllGatherBmmInfershape, InferShape0Shard0)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 0;

    gert::StorageShape xShape = {{E, C, H / tp}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 16, 512}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

// infer shape without bias, success
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape1)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = true;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 16, 512}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

// infer shape with bias, tp failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape2)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 3; // tp failed
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, group ep failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape3)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};
    ;

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, x shard -1 failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape4)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = -1; // x shard -1 failed

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, dim num failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape5)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp}, {}}; // dim num failed
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, common check failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape6)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, 0}, {}}; // common check failed
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, x shard 1 check failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape7)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E / ep, C / tp, H}, {}}; // x shard 1 check failed
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, x shard 4 failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape8)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 4; // x shard 4 failed

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, act type failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape9)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 16, 512}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

// infer shape with bias, group tp failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape10)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expertOutputShape = {{4, 16, 512}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expertOutputShape);
}

// infer shape with bias, ep failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape11)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 3;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, bias dim num failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape12)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep}, {}}; // bias dim num failed

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, bias dim 1 value failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape13)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 12, M / tp}, {}}; // dim value failed

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, x[E] != -1, w[E] = -1 failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape14)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{-1, H, M / tp}, {}}; // x[E] != -1, w[E] = -1 failed
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape without bias, w[E] * ep != x[E] failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape15)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = true;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E, H, M / tp}, {}}; // w[E] * ep != x[E] failed
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape without bias, w[H] != x[H] failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape16)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = true;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H / ep}, {}}; // w[H] != x[H] failed
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape without bias, y3Flag = true but actType = 0 failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape17)
{
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = true;
    constexpr bool y3Flag = true;
    constexpr int xShard = 1;
    constexpr int actType = 0; // y3Flag = true but actType = 0 failed

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, E < 0 failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape18)
{
    constexpr int E = -16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// infer shape with bias, C < 0 failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferShape19)
{
    constexpr int E = 16;
    constexpr int C = -16;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr bool y2Flag = false;
    constexpr bool y3Flag = false;
    constexpr int xShard = 1;

    gert::StorageShape xShape = {{E, C / tp, H}, {}};
    gert::StorageShape weightShape = {{E / ep, H, M / tp}, {}};
    gert::StorageShape biasShape = {{E / ep, 1, M / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllAllGatherBatchMatMul",
        {
            {xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(xShard)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)},
            {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y2Flag)},
            {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(y3Flag)},
        });
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// fp16 infer dtype without bias
TEST_F(AlltoAllAllGatherBmmInfershape, InferDtype0)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(2, 3)
                             .InputDataTypes({&xType, &weightType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllAllGatherBatchMatMul")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2), ge::DT_FLOAT16);
}

// fp16 infer dtype with bias
TEST_F(AlltoAllAllGatherBmmInfershape, InferDtype1)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 3)
                             .InputDataTypes({&xType, &weightType, &biasType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllAllGatherBatchMatMul")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(2), ge::DT_FLOAT16);
}

// fp16 infer dtype with bias, with xType != weightType
TEST_F(AlltoAllAllGatherBmmInfershape, InferDtype2)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_BF16;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 3)
                             .InputDataTypes({&xType, &weightType, &biasType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllAllGatherBatchMatMul")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// fp16 infer dtype with bias, with xType != biasType
TEST_F(AlltoAllAllGatherBmmInfershape, InferDtype3)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 3)
                             .InputDataTypes({&xType, &weightType, &biasType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllAllGatherBatchMatMul")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// fp16 infer dtype with bias, with weightType invalid failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferDtype4)
{
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_INT8;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 3)
                             .InputDataTypes({&xType, &weightType, &biasType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllAllGatherBatchMatMul")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// fp16 infer dtype with bias, with xType invalid failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferDtype5)
{
    ge::DataType xType = ge::DT_INT8;
    ge::DataType weightType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 3)
                             .InputDataTypes({&xType, &weightType, &biasType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllAllGatherBatchMatMul")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// infer dtype with bias, with biasType invalid failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferDtype6)
{
    ge::DataType xType = ge::DT_BF16;
    ge::DataType weightType = ge::DT_BF16;
    ge::DataType biasType = ge::DT_INT8;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 3)
                             .InputDataTypes({&xType, &weightType, &biasType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllAllGatherBatchMatMul")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// infer dtype with bias, with biasType invalid failed
TEST_F(AlltoAllAllGatherBmmInfershape, InferDtype7)
{
    ge::DataType xType = ge::DT_BF16;
    ge::DataType weightType = ge::DT_BF16;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(3, 3)
                             .InputDataTypes({&xType, &weightType, &biasType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllAllGatherBatchMatMul")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}
} // namespace AlltoAllAllGatherBmmInfershapeUT
