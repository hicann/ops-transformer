/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_api/grouped_matmul_weight_quant_950_checker.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_csv_case_loader.h"
#include "opdev/platform.h"

namespace {

using TensorListPtr = std::unique_ptr<aclTensorList, void (*)(aclTensorList*)>;

TensorListPtr MakeTensorList(const csv_map& values, const std::string& name)
{
    const auto shape = ReadMap(values, name + "Shape");
    const auto destroyList = [](aclTensorList* tensors) { aclDestroyTensorList(tensors); };
    // NULL omits an optional list; NULL_ELEMENT creates a list containing one null tensor.
    if (shape == "NULL") {
        return TensorListPtr(nullptr, destroyList);
    }
    if (shape == "NULL_ELEMENT") {
        const aclTensor* nullTensor = nullptr;
        TensorListPtr tensors(aclCreateTensorList(&nullTensor, 1), destroyList);
        EXPECT_NE(tensors.get(), nullptr);
        return tensors;
    }
    auto tensors =
        ops::ut::BuildAclTensorListDesc(shape, ReadMap(values, name + "Dtype"), ReadMap(values, name + "Format"), false)
            .ToAclType();
    EXPECT_NE(tensors.get(), nullptr);
    return tensors;
}

gmm::GMMApiVersion ParseApiVersion(const std::string& api)
{
    static const std::map<std::string, gmm::GMMApiVersion> apiVersions = {{"V2", gmm::GMMApiVersion::V2},
                                                                          {"V3", gmm::GMMApiVersion::V3},
                                                                          {"V4", gmm::GMMApiVersion::V4},
                                                                          {"V5", gmm::GMMApiVersion::V5}};
    return apiVersions.at(api);
}

struct A16W8OpApiCase {
    explicit A16W8OpApiCase(const csv_map& csvValues)
        : caseName(ReadMap(csvValues, "caseName")),
          values(csvValues)
    {}

    void Run() const
    {
        auto x = MakeTensorList(values, "x");
        auto weight = MakeTensorList(values, "weight");
        auto y = MakeTensorList(values, "y");
        auto bias = MakeTensorList(values, "bias");
        auto scale = MakeTensorList(values, "antiquantScale");
        auto offset = MakeTensorList(values, "antiquantOffset");
        std::unique_ptr<aclTensor, void (*)(aclTensor*)> groupList(nullptr,
                                                                   [](aclTensor* tensor) { aclDestroyTensor(tensor); });
        if (ReadMap(values, "groupListShape") != "NULL") {
            groupList = ops::ut::BuildAclTensorDescFromSpec(ReadMap(values, "groupListShape"),
                                                            ReadMap(values, "groupListDtype"),
                                                            ReadMap(values, "groupListFormat"))
                            .ToAclType();
            ASSERT_NE(groupList.get(), nullptr);
            ASSERT_EQ(groupList->GetPlacement(), op::TensorPlacement::kOnDeviceHbm);
        }
        ASSERT_NE(x.get(), nullptr);
        ASSERT_NE(weight.get(), nullptr);
        ASSERT_NE(y.get(), nullptr);
        ASSERT_NE(scale.get(), nullptr);

        gmm::GroupedMatmulParams params;
        params.x = x.get();
        params.weight = weight.get();
        params.y = y.get();
        params.biasOptional = bias.get();
        params.antiquantScaleOptional = scale.get();
        params.antiquantOffsetOptional = offset.get();
        params.groupTensorOptional = groupList.get();
        params.xDtype = static_cast<op::DataType>(ops::ut::ParseAclDtype(ReadMap(values, "xDtype")));
        // Match the public API's stride-based transpose detection.
        params.transposeWeight = gmm::IsTransposeLastTwoDims((*weight)[0]);
        params.transposeX = ops::ut::ParseBool(ReadMap(values, "transposeX"));
        params.apiVersion = ParseApiVersion(ReadMap(values, "api"));
        params.groupType = std::stoll(ReadMap(values, "groupType"));
        params.splitItem = std::stoll(ReadMap(values, "splitItem"));
        params.groupListType = std::stoll(ReadMap(values, "groupListType"));
        params.activeType = std::stoll(ReadMap(values, "activeType"));
        EXPECT_EQ(gmm::AclnnGroupedMatmulWeightQuantDAV3510Checker(params).CheckGroupedMatmulWeightQuantDAV3510(),
                  ops::ut::ParseAclnnStatus(ReadMap(values, "expectRet")));
    }

    std::string caseName;
    csv_map values;
};

std::vector<A16W8OpApiCase> LoadCases(const std::string& csvPath)
{
    auto cases = GetCasesFromCsv<A16W8OpApiCase>(csvPath);
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

std::string MakeCaseName(const testing::TestParamInfo<A16W8OpApiCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class GroupedMatmulA16W8OpApiTest : public testing::TestWithParam<A16W8OpApiCase> {
protected:
    void SetUp() override
    {
        previousSoc_ = op::GetCurrentPlatformInfo().GetSocVersion();
        previousArch_ = op::GetCurrentPlatformInfo().GetCurNpuArch();
        const auto& values = GetParam().values;
        op::SetPlatformSocVersion(GetCaseSocVersion(values, "socVersion"));
        const auto expectedArch = ReadMap(values, "socVersion") == "DAV_9201" ? NpuArch::DAV_9201 : NpuArch::DAV_3510;
        ASSERT_EQ(op::GetCurrentPlatformInfo().GetCurNpuArch(), expectedArch);
    }

    void TearDown() override
    {
        op::SetPlatformSocVersion(previousSoc_);
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() != previousArch_) {
            op::SetPlatformNpuArch(previousArch_);
        }
    }

private:
    op::SocVersion previousSoc_ = op::SocVersion::ASCEND910B;
    NpuArch previousArch_ = NpuArch::DAV_RESV;
};

TEST_P(GroupedMatmulA16W8OpApiTest, CsvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_MATMUL_A16W8_CSV, GroupedMatmulA16W8OpApiTest,
                         testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath(
                             "test_aclnn_grouped_matmul_a16w8.csv", "gmm/grouped_matmul/tests/ut/op_api", __FILE__))),
                         MakeCaseName);

} // namespace
