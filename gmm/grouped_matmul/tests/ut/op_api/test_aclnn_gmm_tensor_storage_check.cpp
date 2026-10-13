/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../common/op_api/gmm_tensor_storage_check.h"
#include "../../../op_api/aclnn_grouped_matmul_v5.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {

using ops::ut::ParseAclDtype;
using ops::ut::ParseAclFormat;
using ops::ut::ParseBool;
using ops::ut::ParseDims;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t CSV_COLUMN_COUNT = 10U;
constexpr uint64_t WORKSPACE_SENTINEL = 12345U;

struct TensorStorageCase {
    std::string caseName;
    std::string scope;
    std::string target;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t offset = 0;
    std::vector<int64_t> storageShape;
    aclDataType dtype = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
    bool expected = false;

    TensorDesc MakeDesc() const
    {
        return TensorDesc(shape, dtype, format, strides, offset, storageShape);
    }

    void RunHelperCase() const
    {
        if (target == "null_tensor") {
            EXPECT_EQ(gmm::CheckTensorStorageBounds(nullptr, "grouped_matmul", "optionalTensor"), expected);
            return;
        }
        if (target == "null_list") {
            EXPECT_EQ(gmm::CheckTensorListStorageBounds(nullptr, "grouped_matmul", "optionalList"), expected);
            return;
        }
        if (target == "null_member") {
            const aclTensor* nullTensor = nullptr;
            std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> tensors(aclCreateTensorList(&nullTensor, 1),
                                                                                    aclDestroyTensorList);
            ASSERT_NE(tensors.get(), nullptr);
            EXPECT_EQ(gmm::CheckTensorListStorageBounds(tensors.get(), "grouped_matmul", "nullMember"), expected);
            return;
        }
        if (target == "empty_list") {
            std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> tensors(aclCreateTensorList(nullptr, 0),
                                                                                    aclDestroyTensorList);
            ASSERT_NE(tensors.get(), nullptr);
            EXPECT_EQ(gmm::CheckTensorListStorageBounds(tensors.get(), "grouped_matmul", "emptyList"), expected);
            return;
        }
        if (target == "list") {
            auto tensors = TensorListDesc({TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND), MakeDesc()}).ToAclType();
            EXPECT_EQ(gmm::CheckTensorListStorageBounds(tensors.get(), "grouped_matmul", "tensorList"), expected);
            return;
        }
        if (target == "rank_mismatch") {
            auto tensor = TensorDesc(shape, dtype, format).ToAclType();
            if (strides.size() == 1U) {
                tensor->SetViewStrides(op::Strides{strides[0]});
            } else {
                ASSERT_EQ(strides.size(), 3U);
                tensor->SetViewStrides(op::Strides{strides[0], strides[1], strides[2]});
            }
            EXPECT_EQ(gmm::CheckTensorStorageBounds(tensor.get(), "grouped_matmul", "tensor"), expected);
            return;
        }
        if (target == "mx_scale_view") {
            auto tensor = MakeDesc().ToAclType();
            op::Shape effectiveShape = tensor->GetViewShape();
            const size_t rank = effectiveShape.GetDimNum();
            ASSERT_TRUE(rank == 3U || rank == 4U);
            const size_t firstAxis = rank - 3U;
            std::swap(effectiveShape[firstAxis], effectiveShape[firstAxis + 1U]);
            EXPECT_EQ(gmm::CheckTensorStorageBoundsWithViewShape(tensor.get(), effectiveShape, "grouped_matmul",
                                                                 "antiquantScaleOptional[0]"),
                      expected);
            return;
        }
        auto tensor = MakeDesc().ToAclType();
        EXPECT_EQ(gmm::CheckTensorStorageBounds(tensor.get(), "grouped_matmul", "tensor"), expected);
    }

    void RunGroupedMatmulCase() const
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        auto x = TensorListDesc(1, TensorDesc({16, 4}, ACL_INT32, ACL_FORMAT_ND)).ToAclType();
        auto weight = TensorListDesc(1, TensorDesc({32, 2}, ACL_INT32, ACL_FORMAT_ND)).ToAclType();
        auto out = TensorListDesc(1, TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND)).ToAclType();
        auto badList = TensorListDesc({TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND), MakeDesc()}).ToAclType();
        auto badTensor = MakeDesc().ToAclType();
        const auto selectList = [&](const std::string& name, const aclTensorList* normal = nullptr) {
            return target == name ? badList.get() : normal;
        };
        const auto selectRequired = [&](const std::string& name, const aclTensorList* normal) {
            return target == name + "_null" ? nullptr : selectList(name, normal);
        };

        uint64_t workspaceSize = WORKSPACE_SENTINEL;
        aclOpExecutor* executor = nullptr;
        const aclnnStatus status = aclnnGroupedMatmulV5GetWorkspaceSize(
            selectRequired("x", x.get()), selectRequired("weight", weight.get()), selectList("bias"),
            selectList("scale"), selectList("offset"), selectList("antiquant_scale"), selectList("antiquant_offset"),
            selectList("per_token_scale"), target == "group_list" ? badTensor.get() : nullptr,
            selectList("activation_input"), selectList("activation_quant_scale"), selectList("activation_quant_offset"),
            0, -1, 0, 0, nullptr, const_cast<aclTensorList*>(selectRequired("out", out.get())),
            const_cast<aclTensorList*>(selectList("activation_feature_out")),
            const_cast<aclTensorList*>(selectList("dyn_quant_scale_out")), &workspaceSize, &executor);

        const bool isNullCase = target == "x_null" || target == "weight_null" || target == "out_null";
        EXPECT_EQ(status, isNullCase ? ACLNN_ERR_PARAM_NULLPTR : ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(workspaceSize, WORKSPACE_SENTINEL);
        EXPECT_EQ(executor, nullptr);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }

    void Run() const
    {
        if (scope == "helper") {
            RunHelperCase();
            return;
        }
        RunGroupedMatmulCase();
    }
};

std::vector<TensorStorageCase> LoadCases(const std::string& csvPath)
{
    std::ifstream input(csvPath);
    EXPECT_TRUE(input.is_open()) << "Failed to open CSV file: " << csvPath;
    std::vector<TensorStorageCase> cases;
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
        TensorStorageCase testCase;
        size_t index = 0;
        testCase.caseName = Trim(columns[index++]);
        testCase.scope = Trim(columns[index++]);
        testCase.target = Trim(columns[index++]);
        testCase.shape = ParseDims(columns[index++]);
        testCase.strides = ParseDims(columns[index++]);
        testCase.offset = std::stoll(Trim(columns[index++]));
        testCase.storageShape = ParseDims(columns[index++]);
        testCase.dtype = ParseAclDtype(columns[index++]);
        testCase.format = ParseAclFormat(columns[index++]);
        testCase.expected = ParseBool(columns[index++]);
        cases.emplace_back(std::move(testCase));
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

std::string MakeCaseName(const testing::TestParamInfo<TensorStorageCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.scope + "_" + info.param.caseName);
}

class GmmTensorStorageCsvTest : public testing::TestWithParam<TensorStorageCase> {};

TEST_P(GmmTensorStorageCsvTest, CsvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMM_TENSOR_STORAGE_CSV, GmmTensorStorageCsvTest,
                         testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath("test_aclnn_gmm_tensor_storage_check.csv",
                                                                             "gmm/grouped_matmul/tests/ut/op_api",
                                                                             __FILE__))),
                         MakeCaseName);

} // namespace
