/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing.h"
#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_v2.h"
#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_v3.h"
#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_weight_nz.h"
#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {

using ops::ut::ParseAclnnStatus;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t CSV_COLUMN_COUNT = 4U;
constexpr uint64_t WORKSPACE_SENTINEL = 12345U;

struct FinalizeRoutingStorageCase {
    std::string caseName;
    std::string api;
    std::string badArgument;
    aclnnStatus expectedStatus = ACLNN_ERR_PARAM_INVALID;

    void Run() const
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        auto normal = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
        auto bad = TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 3}, 1, {2, 3}).ToAclType();
        const auto required = [&](const std::string& name) -> aclTensor* {
            return badArgument == name ? bad.get() : normal.get();
        };
        const auto optional = [&](const std::string& name) -> const aclTensor* {
            return badArgument == name ? bad.get() : nullptr;
        };

        uint64_t workspaceSize = WORKSPACE_SENTINEL;
        aclOpExecutor* executor = nullptr;
        aclnnStatus status = ACLNN_ERR_INNER;
        if (api == "legacy") {
            status = aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSize(
                required("x1"), required("x2"), optional("scale"), optional("bias"), optional("per_token_scale"),
                optional("group_list"), optional("shared_input"), optional("logit"), optional("row_index"), 0, 1.0F, 0,
                false, false, 1, required("out"), &workspaceSize, &executor);
        } else if (api == "v2") {
            status = aclnnGroupedMatmulFinalizeRoutingV2GetWorkspaceSize(
                required("x1"), required("x2"), optional("scale"), optional("bias"), optional("offset"),
                optional("antiquant_scale"), optional("antiquant_offset"), optional("per_token_scale"),
                optional("group_list"), optional("shared_input"), optional("logit"), optional("row_index"), 0, 1.0F, 0,
                false, false, 1, required("out"), &workspaceSize, &executor);
        } else if (api == "v3") {
            status = aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(
                required("x1"), required("x2"), optional("scale"), optional("bias"), optional("offset"),
                optional("antiquant_scale"), optional("antiquant_offset"), optional("per_token_scale"),
                optional("group_list"), optional("shared_input"), optional("logit"), optional("row_index"), 0, 1.0F, 0,
                false, false, 1, nullptr, required("out"), &workspaceSize, &executor);
        } else if (api == "weight_nz") {
            status = aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize(
                required("x1"), required("x2"), optional("scale"), optional("bias"), optional("per_token_scale"),
                optional("group_list"), optional("shared_input"), optional("logit"), optional("row_index"), 0, 1.0F, 0,
                false, false, 1, required("out"), &workspaceSize, &executor);
        } else if (api == "weight_nz_v2") {
            status = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
                required("x1"), required("x2"), optional("scale"), optional("bias"), optional("offset"),
                optional("antiquant_scale"), optional("antiquant_offset"), optional("per_token_scale"),
                optional("group_list"), optional("shared_input"), optional("logit"), optional("row_index"), 0, 1.0F, 0,
                false, false, 1, nullptr, required("out"), &workspaceSize, &executor);
        } else {
            ADD_FAILURE() << "Unsupported API in CSV: " << api;
        }

        EXPECT_EQ(status, expectedStatus);
        EXPECT_EQ(workspaceSize, WORKSPACE_SENTINEL);
        EXPECT_EQ(executor, nullptr);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }
};

std::vector<FinalizeRoutingStorageCase> LoadCases(const std::string& csvPath)
{
    std::ifstream input(csvPath);
    EXPECT_TRUE(input.is_open()) << "Failed to open CSV file: " << csvPath;
    std::vector<FinalizeRoutingStorageCase> cases;
    std::string line;
    bool headerSkipped = false;
    while (std::getline(input, line)) {
        if (Trim(line).empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        std::vector<std::string> columns;
        SplitStr2Vec(line, ",", columns);
        if (columns.size() != CSV_COLUMN_COUNT) {
            ADD_FAILURE() << "Invalid CSV column count in " << csvPath << ": " << line;
            continue;
        }
        cases.push_back({Trim(columns[0]), Trim(columns[1]), Trim(columns[2]), ParseAclnnStatus(columns[3])});
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

std::string MakeCaseName(const testing::TestParamInfo<FinalizeRoutingStorageCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.api + "_" + info.param.caseName);
}

class FinalizeRoutingStorageCsvTest : public testing::TestWithParam<FinalizeRoutingStorageCase> {};

TEST_P(FinalizeRoutingStorageCsvTest, RejectsOutOfBoundsTensorBeforeWorkspaceConstruction)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMM_FINALIZE_ROUTING_STORAGE_CSV, FinalizeRoutingStorageCsvTest,
                         testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath(
                             "test_aclnn_grouped_matmul_finalize_routing_storage_check.csv",
                             "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api", __FILE__))),
                         MakeCaseName);

} // namespace
