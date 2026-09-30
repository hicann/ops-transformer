/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.cpp
 * \brief CSV-driven opapi UT for GroupedMatmulSwigluQuantWeightNzV3.
 */

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"
#include "opdev/make_op_executor.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "gmm_csv_acl_parse_utils.h"

using namespace std;
using namespace op;

namespace {

using ops::ut::BuildAclTensorDescFromSpec;
using ops::ut::ParseAclnnStatus;
using ops::ut::ParseI64List;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;
using ops::ut::BuildAclTensorListDesc;
using ops::ut::MakeAclTensorDesc;

constexpr int64_t MX_QUANT_MODE = 2;
constexpr int64_t SPLIT_SWIGLU_MODE = 2;
constexpr int64_t GROUP_PAIR_STRIDE = 2;
constexpr int64_t GROUP_UNIT_STRIDE = 1;
constexpr int64_t PREFIX_STORAGE_OFFSET = 1;
constexpr int64_t GROUP_LIST_COUNTS = 1;
constexpr int64_t GROUP_HALF_M = 16;
constexpr int64_t GROUP_TOTAL_M = 32;
constexpr int64_t GROUP_PADDING_SENTINEL = 99;
constexpr int64_t NEGATIVE_GROUP_VALUE = -1;
constexpr int64_t DECREASING_GROUP_OFFSET = 8;
constexpr int64_t FIRST_GROUP_LENGTH = 12;
constexpr int64_t SECOND_GROUP_LENGTH = 20;
constexpr int64_t EXCESS_GROUP_LENGTH = 1;
constexpr int64_t TWO_GROUP_COUNT = 2;
constexpr size_t SINGLE_TENSOR_COUNT = 1;
constexpr int64_t OPTIONAL_TENSOR_ELEMENTS = 1;
constexpr uint64_t NONEMPTY_WORKSPACE_BYTES = 1;
constexpr double DEFAULT_CLAMP_LIMIT = 7.0;
constexpr double DEFAULT_GLU_ALPHA = 1.702;
constexpr double DEFAULT_GLU_BIAS = 1.0;
constexpr size_t kSwigluV3CsvColumnCount = 31;

struct CsvExpectation {
    string caseName;
    bool checkRet{true};
    aclnnStatus expected{ACLNN_SUCCESS};
};

template <typename Case>
class SwigluOpApiCsvTest : public testing::TestWithParam<Case> {
protected:
    void CheckWorkspaceSize()
    {
        const auto& item = this->GetParam();
        const aclnnStatus actual = item.Run();
        if (item.checkRet) {
            EXPECT_EQ(actual, item.expected);
        }
    }
};

template <typename Case, typename ParseRow>
vector<Case> LoadSwigluCases(const string& csvPath, size_t columnCount, ParseRow parseRow)
{
    ifstream in(csvPath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvPath;
    vector<Case> cases;
    string line;
    bool headerSkipped = false;
    size_t lineNo = 0;
    while (getline(in, line)) {
        ++lineNo;
        if (line.empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != columnCount) {
            ADD_FAILURE() << "Invalid CSV column count at line " << lineNo << ": " << cols.size();
            continue;
        }
        try {
            Case item;
            if (parseRow(cols, item)) {
                cases.emplace_back(std::move(item));
            }
        } catch (const exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, "", error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

template <typename Case>
string BuildCaseName(const testing::TestParamInfo<Case>& info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

void ReleaseCaseTensor(aclTensor* tensor)
{
    Release(tensor);
}
void ReleaseCaseTensorList(aclTensorList* tensors)
{
    Release(tensors);
}

// Retain the framework's TensorDesc/TensorListDesc ownership and null semantics without
// instantiating a different OP_API_UT tuple for every combination of optional inputs.
struct SwigluApiTensors {
    using Tensor = unique_ptr<aclTensor, void (*)(aclTensor*)>;
    using TensorList = unique_ptr<aclTensorList, void (*)(aclTensorList*)>;

    Tensor x{nullptr, ReleaseCaseTensor};
    TensorList weight{nullptr, ReleaseCaseTensorList};
    TensorList weightScale{nullptr, ReleaseCaseTensorList};
    TensorList weightAssist{nullptr, ReleaseCaseTensorList};
    Tensor bias{nullptr, ReleaseCaseTensor};
    Tensor xScale{nullptr, ReleaseCaseTensor};
    Tensor smoothScale{nullptr, ReleaseCaseTensor};
    Tensor groupList{nullptr, ReleaseCaseTensor};
    Tensor output{nullptr, ReleaseCaseTensor};
    Tensor outputScale{nullptr, ReleaseCaseTensor};
    unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> tuningConfig{nullptr, aclDestroyIntArray};

    void SetTuningConfig(const string& value)
    {
        const auto values = ParseI64List(value, "|");
        if (!values.empty()) {
            tuningConfig.reset(aclCreateIntArray(values.data(), values.size()));
        }
    }

    template <typename Query>
    aclnnStatus GetWorkspaceSize(Query query) const
    {
        struct QueryResult {
            uint64_t workspaceSize{0};
            aclOpExecutor* executor{nullptr};
            // Match OpApiUt::TestGetWorkspaceSize, including exceptional exits.
            ~QueryResult()
            {
                delete executor;
            }
        } result;
        // The executor must be destroyed before the input tensor resources.
        return query(&result.workspaceSize, &result.executor);
    }
};

void SetupPlatformForCase()
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
}

enum class Phase1NullPointerTarget {
    kWorkspaceSize,
    kExecutor,
    kWeightElement,
    kWeightScaleElement,
    kWeightElementWithInvalidCount,
};

aclnnStatus RunPhase1NullPointerCase(Phase1NullPointerTarget target)
{
    SetupPlatformForCase();
    auto x = BuildAclTensorDescFromSpec("32:128", "FLOAT8_E4M3FN", "ND").ToAclType();
    auto weightTensor =
        BuildAclTensorDescFromSpec("1:128:128#16384:128:1#1:4:8:16:32", "FLOAT8_E4M3FN", "FRACTAL_NZ").ToAclType();
    auto weightScaleTensor = BuildAclTensorDescFromSpec("1:2:128:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto xScale = BuildAclTensorDescFromSpec("32:2:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto groupList = BuildAclTensorDescFromSpec("1", "INT64", "ND").ToAclType();
    auto output = BuildAclTensorDescFromSpec("32:64", "FLOAT8_E4M3FN", "ND").ToAclType();
    auto outputScale = BuildAclTensorDescFromSpec("32:1:2", "FLOAT8_E8M0", "ND").ToAclType();

    vector<aclTensor*> weightElements = {weightTensor.get()};
    vector<aclTensor*> weightScaleElements = {weightScaleTensor.get()};
    if (target == Phase1NullPointerTarget::kWeightElement) {
        weightElements = {nullptr};
    } else if (target == Phase1NullPointerTarget::kWeightScaleElement) {
        weightScaleElements = {nullptr};
    } else if (target == Phase1NullPointerTarget::kWeightElementWithInvalidCount) {
        weightElements.emplace_back(nullptr);
    }
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weight(
        aclCreateTensorList(weightElements.data(), weightElements.size()), aclDestroyTensorList);
    if (weight == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (target != Phase1NullPointerTarget::kWeightElement) {
        (void)weightTensor.release();
    }

    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weightScale(
        aclCreateTensorList(weightScaleElements.data(), weightScaleElements.size()), aclDestroyTensorList);
    if (weightScale == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (target != Phase1NullPointerTarget::kWeightScaleElement) {
        (void)weightScaleTensor.release();
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    uint64_t* workspaceSizeArg = target == Phase1NullPointerTarget::kWorkspaceSize ? nullptr : &workspaceSize;
    aclOpExecutor** executorArg = target == Phase1NullPointerTarget::kExecutor ? nullptr : &executor;
    const aclnnStatus ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x.get(), weight.get(), weightScale.get(), nullptr, nullptr, xScale.get(), nullptr, groupList.get(),
        MX_QUANT_MODE, 0, MX_QUANT_MODE, 0, nullptr, SPLIT_SWIGLU_MODE, DEFAULT_CLAMP_LIMIT, DEFAULT_GLU_ALPHA,
        DEFAULT_GLU_BIAS, "rint", 0, 0.0, output.get(), outputScale.get(), workspaceSizeArg, executorArg);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
    return ret;
}

aclnnStatus RunHostGroupListCase(const vector<int64_t>& storageValues, int64_t stride, int64_t offset,
                                 int64_t groupListType)
{
    SetupPlatformForCase();
    auto inputExecutor = CREATE_EXECUTOR();
    if (inputExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    // The input executor owns the copied Host storage until the API executor has been destroyed.
    auto* groupList =
        inputExecutor->AllocHostTensor(storageValues.data(), storageValues.size(), op::DataType::DT_INT64);
    if (groupList == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    groupList->SetViewShape(op::Shape({TWO_GROUP_COUNT}));
    groupList->SetViewStrides(op::Strides({stride}));
    groupList->SetViewOffset(offset);

    auto x = BuildAclTensorDescFromSpec("32:128", "FLOAT8_E4M3FN", "ND").ToAclType();
    auto weightTensor =
        BuildAclTensorDescFromSpec("2:128:128#16384:128:1#2:4:8:16:32", "FLOAT8_E4M3FN", "FRACTAL_NZ").ToAclType();
    auto weightScaleTensor = BuildAclTensorDescFromSpec("2:2:128:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto xScale = BuildAclTensorDescFromSpec("32:2:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto output = BuildAclTensorDescFromSpec("32:64", "FLOAT8_E4M3FN", "ND").ToAclType();
    auto outputScale = BuildAclTensorDescFromSpec("32:1:2", "FLOAT8_E8M0", "ND").ToAclType();

    const aclTensor* weightElements[] = {weightTensor.get()};
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weight(
        aclCreateTensorList(weightElements, SINGLE_TENSOR_COUNT), aclDestroyTensorList);
    if (weight == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    (void)weightTensor.release();
    const aclTensor* weightScaleElements[] = {weightScaleTensor.get()};
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weightScale(
        aclCreateTensorList(weightScaleElements, SINGLE_TENSOR_COUNT), aclDestroyTensorList);
    if (weightScale == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    (void)weightScaleTensor.release();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x.get(), weight.get(), weightScale.get(), nullptr, nullptr, xScale.get(), nullptr, groupList, MX_QUANT_MODE, 0,
        MX_QUANT_MODE, groupListType, nullptr, SPLIT_SWIGLU_MODE, DEFAULT_CLAMP_LIMIT, DEFAULT_GLU_ALPHA,
        DEFAULT_GLU_BIAS, "rint", 0, 0.0, output.get(), outputScale.get(), &workspaceSize, &executor);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
    return ret;
}

aclOpExecutor* CreateTestExecutor()
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return nullptr;
    }
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);
    return executor;
}

struct V3Case : CsvExpectation {
    string xSpec;
    string xDtype;
    string xFormat;
    string weightSpecs;
    string weightDtype;
    string weightFormat;
    string weightScaleSpecs;
    string weightScaleDtype;
    string weightScaleFormat;
    string xScaleSpec;
    string xScaleDtype;
    string groupListSpec;
    string groupListDtype;
    string outputSpec;
    string outputScaleSpec;
    int64_t dequantMode;
    int64_t dequantDtype;
    int64_t quantMode;
    int64_t groupListType;
    int64_t swigluMode;
    double clampLimit;
    double gluAlpha;
    double gluBias;
    string roundMode;
    int64_t scaleAlg;
    double dstTypeMax;
    string optionalInput;
    string tuningConfig;
    string nullArg;

    aclnnStatus Run() const
    {
        SetupPlatformForCase();
        SwigluApiTensors tensors;
        if (nullArg != "x") {
            tensors.x = BuildAclTensorDescFromSpec(xSpec, xDtype, xFormat).ToAclType();
        }
        if (nullArg != "weight") {
            tensors.weight = BuildAclTensorListDesc(weightSpecs, weightDtype, weightFormat).ToAclType();
        }
        if (nullArg != "weight_scale") {
            tensors.weightScale =
                BuildAclTensorListDesc(weightScaleSpecs, weightScaleDtype, weightScaleFormat).ToAclType();
        }
        if (nullArg != "x_scale") {
            tensors.xScale = BuildAclTensorDescFromSpec(xScaleSpec, xScaleDtype, "ND").ToAclType();
        }
        if (nullArg != "group_list") {
            tensors.groupList = BuildAclTensorDescFromSpec(groupListSpec, groupListDtype, "ND").ToAclType();
        }
        if (nullArg != "output") {
            tensors.output = BuildAclTensorDescFromSpec(outputSpec, "FLOAT8_E4M3FN", "ND").ToAclType();
        }
        if (nullArg != "output_scale") {
            tensors.outputScale = BuildAclTensorDescFromSpec(outputScaleSpec, "FLOAT8_E8M0", "ND").ToAclType();
        }
        // NULL cases take precedence over the optional-input rejection cases, as in the original UT.
        if (tensors.x && tensors.weight && tensors.weightScale && tensors.xScale && tensors.groupList &&
            tensors.output && tensors.outputScale) {
            if (optionalInput == "assist") {
                tensors.weightAssist = TensorListDesc(vector<TensorDesc>{MakeAclTensorDesc({OPTIONAL_TENSOR_ELEMENTS},
                                                                                           ACL_FLOAT, ACL_FORMAT_ND)})
                                           .ToAclType();
            } else if (optionalInput == "bias") {
                tensors.bias = MakeAclTensorDesc({OPTIONAL_TENSOR_ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
            } else if (optionalInput == "smooth") {
                tensors.smoothScale =
                    MakeAclTensorDesc({OPTIONAL_TENSOR_ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
            }
        }
        tensors.SetTuningConfig(tuningConfig);
        const char* roundModePtr = roundMode == "NULL" ? nullptr : roundMode.c_str();
        return tensors.GetWorkspaceSize([&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
                tensors.x.get(), tensors.weight.get(), tensors.weightScale.get(), tensors.weightAssist.get(),
                tensors.bias.get(), tensors.xScale.get(), tensors.smoothScale.get(), tensors.groupList.get(),
                dequantMode, dequantDtype, quantMode, groupListType, tensors.tuningConfig.get(), swigluMode, clampLimit,
                gluAlpha, gluBias, roundModePtr, scaleAlg, dstTypeMax, tensors.output.get(), tensors.outputScale.get(),
                workspaceSize, executor);
        });
    }
};

bool ParseV3Case(const vector<string>& cols, V3Case& item)
{
    size_t i = 0;
    item.caseName = Trim(cols[i++]);
    item.expected = ParseAclnnStatus(cols[i++]);
    item.xSpec = Trim(cols[i++]);
    item.xDtype = Trim(cols[i++]);
    item.xFormat = Trim(cols[i++]);
    item.weightSpecs = Trim(cols[i++]);
    item.weightDtype = Trim(cols[i++]);
    item.weightFormat = Trim(cols[i++]);
    item.weightScaleSpecs = Trim(cols[i++]);
    item.weightScaleDtype = Trim(cols[i++]);
    item.weightScaleFormat = Trim(cols[i++]);
    item.xScaleSpec = Trim(cols[i++]);
    item.xScaleDtype = Trim(cols[i++]);
    item.groupListSpec = Trim(cols[i++]);
    item.groupListDtype = Trim(cols[i++]);
    item.outputSpec = Trim(cols[i++]);
    item.outputScaleSpec = Trim(cols[i++]);
    item.dequantMode = stoll(Trim(cols[i++]));
    item.dequantDtype = stoll(Trim(cols[i++]));
    item.quantMode = stoll(Trim(cols[i++]));
    item.groupListType = stoll(Trim(cols[i++]));
    item.swigluMode = stoll(Trim(cols[i++]));
    item.clampLimit = stod(Trim(cols[i++]));
    item.gluAlpha = stod(Trim(cols[i++]));
    item.gluBias = stod(Trim(cols[i++]));
    item.roundMode = Trim(cols[i++]);
    item.scaleAlg = stoll(Trim(cols[i++]));
    item.dstTypeMax = stod(Trim(cols[i++]));
    item.optionalInput = Trim(cols[i++]);
    item.tuningConfig = Trim(cols[i++]);
    item.nullArg = Trim(cols[i++]);
    return true;
}

class grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test : public SwigluOpApiCsvTest<V3Case> {};

TEST_P(grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test, get_workspace_size)
{
    CheckWorkspaceSize();
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, phase1_output_pointers_return_param_nullptr)
{
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWorkspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kExecutor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_strided_views_ignore_padding)
{
    EXPECT_EQ(RunHostGroupListCase({GROUP_HALF_M, GROUP_PADDING_SENTINEL, GROUP_TOTAL_M}, GROUP_PAIR_STRIDE, 0, 0),
              ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({GROUP_PADDING_SENTINEL, GROUP_HALF_M, GROUP_PADDING_SENTINEL, GROUP_TOTAL_M},
                                   GROUP_PAIR_STRIDE, PREFIX_STORAGE_OFFSET, 0),
              ACLNN_SUCCESS);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_strided_views_check_logical_values)
{
    EXPECT_EQ(RunHostGroupListCase({GROUP_HALF_M, SECOND_GROUP_LENGTH, NEGATIVE_GROUP_VALUE}, GROUP_PAIR_STRIDE, 0, 0),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(
        RunHostGroupListCase({GROUP_HALF_M, SECOND_GROUP_LENGTH, DECREASING_GROUP_OFFSET}, GROUP_PAIR_STRIDE, 0, 0),
        ACLNN_ERR_PARAM_INVALID);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_strided_views_check_logical_sum)
{
    EXPECT_EQ(RunHostGroupListCase({FIRST_GROUP_LENGTH, GROUP_PADDING_SENTINEL, SECOND_GROUP_LENGTH}, GROUP_PAIR_STRIDE,
                                   0, GROUP_LIST_COUNTS),
              ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({GROUP_HALF_M, EXCESS_GROUP_LENGTH, SECOND_GROUP_LENGTH}, GROUP_PAIR_STRIDE, 0,
                                   GROUP_LIST_COUNTS),
              ACLNN_ERR_PARAM_INVALID);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_contiguous_values_are_unchanged)
{
    EXPECT_EQ(RunHostGroupListCase({GROUP_HALF_M, GROUP_TOTAL_M}, GROUP_UNIT_STRIDE, 0, 0), ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({FIRST_GROUP_LENGTH, SECOND_GROUP_LENGTH}, GROUP_UNIT_STRIDE, 0, GROUP_LIST_COUNTS),
              ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({GROUP_HALF_M, NEGATIVE_GROUP_VALUE}, GROUP_UNIT_STRIDE, 0, 0),
              ACLNN_ERR_PARAM_INVALID);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, tensor_list_elements_return_param_nullptr)
{
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWeightElement), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWeightScaleElement), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWeightElementWithInvalidCount),
              ACLNN_ERR_PARAM_NULLPTR);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, phase2_executor_returns_param_nullptr)
{
    EXPECT_EQ(aclnnGroupedMatmulSwigluQuantWeightNzV3(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, phase2_required_workspace_returns_param_nullptr)
{
    unique_ptr<aclOpExecutor, decltype(&aclDestroyAclOpExecutor)> executor(CreateTestExecutor(),
                                                                           aclDestroyAclOpExecutor);
    ASSERT_NE(executor.get(), nullptr);
    EXPECT_EQ(aclnnGroupedMatmulSwigluQuantWeightNzV3(nullptr, NONEMPTY_WORKSPACE_BYTES, executor.get(), nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv,
                         grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test,
                         testing::ValuesIn(LoadSwigluCases<V3Case>(
                             ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.csv",
                                                     "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_api", __FILE__),
                             kSwigluV3CsvColumnCount, ParseV3Case)),
                         BuildCaseName<V3Case>);
} // namespace
