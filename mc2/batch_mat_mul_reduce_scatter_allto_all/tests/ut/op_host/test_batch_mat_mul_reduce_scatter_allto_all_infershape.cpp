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
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"

class BatchMatMulReduceScatterAlltoAllInfershape : public testing::Test {
    protected:
        static void SetUpTestCase() {
            std::cout << "BatchMatMulReduceScatterAlltoAllInfershape SetUp" << std::endl;
        }

        static void TearDownTestCase() {
            std::cout << "BatchMatMulReduceScatterAlltoAllInfershape TearDown" << std::endl;
        }
};

// infer shape with bias, success
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_0) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{E, C / tp, H}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// infer shape without bias, success
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_1) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{E, C / tp, H}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// infer shape without bias, tp failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_2) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 3; // tp failed
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape without bias, empty tensor, c = 0 failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_3) {
    constexpr int E = 16;
    constexpr int C = 0; // empty tensor, currently failed
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, ep failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_4) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 3; // ep failed
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, group ep failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_5) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

        gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, group tp failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_6) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, dim num invalid
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_7) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C}, {}}; // dim num invalid
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, x[2] != w[1] failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_8) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M}, {}}; // x[2] != w[1]
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, x[2] = w[1] = 0 failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_9) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 0;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}}; // x[2] = w[1] = 0
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape without bias, y shard = 1 condition check failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_10) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E, M / tp, H}, {}}; // y shard = 1 check failed
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, yShard failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_11) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 4; // yShard failed

    gert::StorageShape xStorageShape = {{E / ep, ep * tp * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape without bias, yShard = -1 failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_12) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = -1;

    gert::StorageShape xStorageShape = {{E / ep, ep * tp * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, bias dim failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_13) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H, H}, {}}; // bias dim failed

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, bias[2] != w[M] failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_14) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H / tp}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, bias[E] != w[E] failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_15) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, E < 0 failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_16) {
    constexpr int E = -16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, C < 0 failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_17) {
    constexpr int E = 16;
    constexpr int C = -16;
    constexpr int H = 1024;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, H > 65535 failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_18) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 6553500;
    constexpr int M = 2048;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// infer shape with bias, M/tp > 65535 failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_shape_19) {
    constexpr int E = 16;
    constexpr int C = 4;
    constexpr int H = 1024;
    constexpr int M = 2048000;
    constexpr int ep = 4;
    constexpr int tp = 4;
    constexpr bool transW = false;
    constexpr int yShard = 1;

    gert::StorageShape xStorageShape = {{E / ep, ep * C, M / tp}, {}};
    gert::StorageShape weightStorageShape = {{E / ep, M / tp, H}, {}};
    gert::StorageShape biasStorageShape = {{E / ep, 1, H}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        {
            {xStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {weightStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {biasStorageShape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComEp")},
            {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclComTp")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
            {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tp)},
            {"y_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yShard)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transW)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

// fp16 infer dtype without bias, success
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_0) {
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(2, 1)
        .InputDataTypes({&xType, &weightType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

// fp16 infer dtype with bias, success
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_1) {
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 1)
        .InputDataTypes({&xType, &weightType, &biasType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

// fp16 infer dtype with bias, xType != weightType, failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_2) {
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_BF16;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 1)
        .InputDataTypes({&xType, &weightType, &biasType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// fp16 infer dtype with bias, xType != biasType, failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_3) {
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 1)
        .InputDataTypes({&xType, &weightType, &biasType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// bf16 infer dtype with bias, success
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_4) {
    ge::DataType xType = ge::DT_BF16;
    ge::DataType weightType = ge::DT_BF16;
    ge::DataType biasType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 1)
        .InputDataTypes({&xType, &weightType, &biasType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

// infer dtype with bias, xType invalid, failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_5) {
    ge::DataType xType = ge::DT_INT8;
    ge::DataType weightType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 1)
        .InputDataTypes({&xType, &weightType, &biasType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// infer dtype with bias, weightType invalid, failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_6) {
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_INT8;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 1)
        .InputDataTypes({&xType, &weightType, &biasType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// infer dtype with bias, biasType invalid, failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_7) {
    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType weightType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_INT8;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 1)
        .InputDataTypes({&xType, &weightType, &biasType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}

// infer dtype with bias, biasType invalid, failed
TEST_F(BatchMatMulReduceScatterAlltoAllInfershape, infer_dtype_8) {
    ge::DataType xType = ge::DT_BF16;
    ge::DataType weightType = ge::DT_BF16;
    ge::DataType biasType = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 1)
        .InputDataTypes({&xType, &weightType, &biasType})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("BatchMatMulReduceScatterAlltoAll")->infer_datatype;
    ASSERT_NE(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}