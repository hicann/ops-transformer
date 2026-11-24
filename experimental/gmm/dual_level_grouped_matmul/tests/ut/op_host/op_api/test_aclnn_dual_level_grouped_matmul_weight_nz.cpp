// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "acl/acl.h"
#include "opdev/op_executor.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_csv_case_loader.h"
#include "../../../../op_api/dual_level_grouped_matmul_checker.h"

namespace {

using ops::ut::BuildAclTensorDescFromSpec;
using ops::ut::ParseAclnnStatus;

using TensorListPtr = std::unique_ptr<aclTensorList, void (*)(aclTensorList*)>;
using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor*)>;
using IntArrayPtr = std::unique_ptr<aclIntArray, void (*)(aclIntArray*)>;

constexpr char kCsvName[] = "test_aclnn_dual_level_grouped_matmul_weight_nz.csv";
constexpr char kCsvRepoDir[] = "experimental/gmm/dual_level_grouped_matmul/tests/ut/op_host/op_api";

// Descriptor-only tests call the production checker; no allocation or device launch is required.
struct DualLevelGmmCheckerCase {
    explicit DualLevelGmmCheckerCase(const csv_map& csvValues)
        : caseName(ReadMap(csvValues, "caseName")),
          values(csvValues)
    {}

    TensorListPtr MakeTensorList(const std::string& name) const
    {
        const auto destroy = [](aclTensorList* list) { aclDestroyTensorList(list); };
        const auto mode = ReadMap(values, "nullInput");
        if (mode == "empty_" + name || mode == "element_" + name) {
            const aclTensor* element = nullptr;
            auto list = TensorListPtr(aclCreateTensorList(&element, mode == "empty_" + name ? 0 : 1), destroy);
            EXPECT_NE(list.get(), nullptr);
            return list;
        }
        const std::string spec = ReadMap(values, name + "Spec");
        if (spec.empty() || spec == "NONE") {
            return TensorListPtr(nullptr, destroy);
        }
        auto list = TensorListDesc({BuildAclTensorDescFromSpec(spec, ReadMap(values, name + "Dtype"),
                                                               ReadMap(values, name + "Format"))})
                        .ToAclType();
        EXPECT_NE(list.get(), nullptr);
        return list;
    }

    TensorPtr MakeTensor(const std::string& name) const
    {
        const auto destroy = [](aclTensor* tensor) { aclDestroyTensor(tensor); };
        const std::string spec = ReadMap(values, name + "Spec");
        if (spec.empty() || spec == "NONE") {
            return TensorPtr(nullptr, destroy);
        }
        auto tensor =
            BuildAclTensorDescFromSpec(spec, ReadMap(values, name + "Dtype"), ReadMap(values, name + "Format"))
                .ToAclType();
        EXPECT_NE(tensor.get(), nullptr);
        return tensor;
    }

    static void SetFirstElementOffset(const aclTensorList* list)
    {
        if (list == nullptr || (*list)[0] == nullptr) {
            return;
        }
        const_cast<aclTensor*>((*list)[0])->SetViewOffset(1);
    }

    void Run() const
    {
        auto x = MakeTensorList("x");
        auto weight = MakeTensorList("weight");
        auto bias = MakeTensorList("bias");
        auto scale = MakeTensorList("scale");
        auto perTokenScale = MakeTensorList("perToken");
        auto groupList = MakeTensor("groupList");
        auto xLevel0Scale = MakeTensorList("xLevel0");
        auto weightLevel0Scale = MakeTensorList("weightLevel0");
        auto out = MakeTensorList("out");

        const std::string nullInput = ReadMap(values, "nullInput");
        const std::string offsetInput = ReadMap(values, "offsetInput");
        if (offsetInput == "groupList") {
            if (groupList != nullptr) {
                groupList->SetViewOffset(1);
            }
        } else if (offsetInput != "NONE") {
            if (offsetInput == "x") {
                SetFirstElementOffset(x.get());
            } else if (offsetInput == "weight") {
                SetFirstElementOffset(weight.get());
            } else if (offsetInput == "bias") {
                SetFirstElementOffset(bias.get());
            } else if (offsetInput == "scale") {
                SetFirstElementOffset(scale.get());
            } else if (offsetInput == "perToken") {
                SetFirstElementOffset(perTokenScale.get());
            } else if (offsetInput == "xLevel0") {
                SetFirstElementOffset(xLevel0Scale.get());
            } else if (offsetInput == "weightLevel0") {
                SetFirstElementOffset(weightLevel0Scale.get());
            } else if (offsetInput == "out") {
                SetFirstElementOffset(out.get());
            }
        }

        IntArrayPtr tuningConfig(nullptr, [](aclIntArray* array) { aclDestroyIntArray(array); });
        const std::string tuningValue = ReadMap(values, "tuningConfig");
        if (!tuningValue.empty() && tuningValue != "NONE") {
            int64_t config = std::stoll(tuningValue);
            tuningConfig.reset(aclCreateIntArray(&config, 1));
            ASSERT_NE(tuningConfig.get(), nullptr);
        }

        dual_level_gmm::DualLevelGroupedMatmulParams params;
        params.x = nullInput == "x" ? nullptr : x.get();
        params.weight = nullInput == "weight" ? nullptr : weight.get();
        params.biasOptional = nullInput == "bias" ? nullptr : bias.get();
        params.scaleOptional = nullInput == "scale" ? nullptr : scale.get();
        params.perTokenScaleOptional = nullInput == "perToken" ? nullptr : perTokenScale.get();
        params.groupListOptional = nullInput == "groupList" ? nullptr : groupList.get();
        params.xLevel0Scale = nullInput == "xLevel0" ? nullptr : xLevel0Scale.get();
        params.weightLevel0Scale = nullInput == "weightLevel0" ? nullptr : weightLevel0Scale.get();
        params.level0GroupSize = std::stoll(ReadMap(values, "level0GroupSize"));
        params.level1GroupSize = std::stoll(ReadMap(values, "level1GroupSize"));
        params.splitItem = std::stoll(ReadMap(values, "splitItem"));
        params.groupType = std::stoll(ReadMap(values, "groupType"));
        params.groupListType = std::stoll(ReadMap(values, "groupListType"));
        params.tuningConfigOptional = tuningConfig.get();
        params.out = nullInput == "out" ? nullptr : out.get();

        bool transposeWeight = false;
        EXPECT_EQ(dual_level_gmm::CheckParams(params, transposeWeight), ParseAclnnStatus(ReadMap(values, "expectRet")));
    }

    std::string caseName;
    csv_map values;
};

std::vector<DualLevelGmmCheckerCase> LoadCases(const std::string& csvPath)
{
    auto cases = GetCasesFromCsv<DualLevelGmmCheckerCase>(csvPath);
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

std::string MakeCaseName(const testing::TestParamInfo<DualLevelGmmCheckerCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class DualLevelGmmCheckerCsvTest : public testing::TestWithParam<DualLevelGmmCheckerCase> {};

// CSV-driven batch checker cases; add rows to test_aclnn_dual_level_grouped_matmul_weight_nz.csv to extend coverage.
TEST_P(DualLevelGmmCheckerCsvTest, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(DUAL_LEVEL_GMM_CHECKER_CSV, DualLevelGmmCheckerCsvTest,
                         testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath(kCsvName, kCsvRepoDir, __FILE__))),
                         MakeCaseName);

TEST(DualLevelGmmWorkspace, RejectsNullWorkspaceOutputBeforeCreatingExecutor)
{
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                  nullptr, nullptr, 512, 32, 3, 0, 0, nullptr, nullptr,
                                                                  nullptr, &executor),
              ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

TEST(DualLevelGmmWorkspace, RejectsNullExecutorOutputBeforeValidatingInputs)
{
    uint64_t workspaceSize = 123;
    EXPECT_EQ(aclnnDualLevelGroupedMatmulWeightNzGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                  nullptr, nullptr, 512, 32, 3, 0, 0, nullptr, nullptr,
                                                                  &workspaceSize, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspaceSize, 123U);
}
} // namespace
