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
#include "opdev/tensor_view_utils.h"
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
constexpr int64_t FIRST_GROUP_LENGTH = 12;
constexpr int64_t SECOND_GROUP_LENGTH = 20;
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

aclnnStatus RunMxA8W4PlacementCase(bool hostGroupList, bool hostSmoothScale, bool strided, int64_t groupListType,
                                   bool transposeWeight, bool torchBaseFormats = false)
{
    SetupPlatformForCase();
    // Own genuine Host tensors until the queried API executor has been destroyed.
    auto inputExecutor = CREATE_EXECUTOR();
    if (inputExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    SwigluApiTensors tensors;
    tensors.x = BuildAclTensorDescFromSpec("32:128", "FLOAT8_E4M3FN", "ND").ToAclType();
    tensors.weight = BuildAclTensorListDesc(
                         transposeWeight ? "2:128:128#16384:1:128#2:2:8:16:64" : "2:128:128#16384:128:1#2:2:8:16:64",
                         "FLOAT4_E2M1", "FRACTAL_NZ")
                         .ToAclType();
    tensors.weightScale = BuildAclTensorListDesc(transposeWeight ? "2:2:128:2#512:2:4:1" : "2:2:128:2", "FLOAT8_E8M0",
                                                 torchBaseFormats ? "NCHW" : "ND")
                              .ToAclType();
    tensors.xScale = BuildAclTensorDescFromSpec("32:2:2", "FLOAT8_E8M0", torchBaseFormats ? "NCL" : "ND").ToAclType();
    tensors.groupList = BuildAclTensorDescFromSpec(strided ? "2#2#3" : "2", "INT64", "ND").ToAclType();
    tensors.smoothScale = BuildAclTensorDescFromSpec(strided ? "2:64#128:2#2:128" : "2:64", "FLOAT", "ND").ToAclType();
    tensors.output = BuildAclTensorDescFromSpec("32:64", "FLOAT8_E4M3FN", "ND").ToAclType();
    tensors.outputScale =
        BuildAclTensorDescFromSpec("32:1:2", "FLOAT8_E8M0", torchBaseFormats ? "NCL" : "ND").ToAclType();
    const aclTensor* groupList = tensors.groupList.get();
    const aclTensor* smoothScale = tensors.smoothScale.get();
    EXPECT_EQ(groupList->GetPlacement(), op::TensorPlacement::kOnDeviceHbm);
    EXPECT_EQ(smoothScale->GetPlacement(), op::TensorPlacement::kOnDeviceHbm);
    if (hostGroupList) {
        const int64_t lastValue = groupListType == 0 ? GROUP_TOTAL_M : GROUP_HALF_M;
        const vector<int64_t> values = strided ? vector<int64_t>{GROUP_HALF_M, GROUP_PADDING_SENTINEL, lastValue} :
                                                 vector<int64_t>{GROUP_HALF_M, lastValue};
        auto* hostTensor = inputExecutor->AllocHostTensor(values.data(), values.size(), op::DataType::DT_INT64);
        if (hostTensor == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
        hostTensor->SetViewShape(op::Shape({TWO_GROUP_COUNT}));
        hostTensor->SetViewStrides(op::Strides({strided ? GROUP_PAIR_STRIDE : GROUP_UNIT_STRIDE}));
        groupList = hostTensor;
        EXPECT_TRUE(gert::TensorPlacementUtils::IsOnHost(groupList->GetPlacement()));
    }
    if (hostSmoothScale) {
        const vector<float> values(strided ? 256 : 128, 1.0F);
        auto* hostTensor = inputExecutor->AllocHostTensor(values.data(), values.size(), op::DataType::DT_FLOAT);
        if (hostTensor == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
        hostTensor->SetViewShape(op::Shape({2, 64}));
        hostTensor->SetStorageShape(op::Shape({2, strided ? 128 : 64}));
        hostTensor->SetViewStrides(strided ? op::Strides({128, 2}) : op::Strides({64, 1}));
        smoothScale = hostTensor;
        EXPECT_TRUE(gert::TensorPlacementUtils::IsOnHost(smoothScale->GetPlacement()));
    }
    return tensors.GetWorkspaceSize([&](uint64_t* workspaceSize, aclOpExecutor** executor) {
        return aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
            tensors.x.get(), tensors.weight.get(), tensors.weightScale.get(), nullptr, nullptr, tensors.xScale.get(),
            smoothScale, groupList, MX_QUANT_MODE, 0, MX_QUANT_MODE, groupListType, nullptr, SPLIT_SWIGLU_MODE,
            DEFAULT_CLAMP_LIMIT, DEFAULT_GLU_ALPHA, DEFAULT_GLU_BIAS, "rint", 0, 0.0, tensors.output.get(),
            tensors.outputScale.get(), workspaceSize, executor);
    });
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
    string outputDtype{"FLOAT8_E4M3FN"};
    string outputScaleDtype{"FLOAT8_E8M0"};
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
            tensors.output = BuildAclTensorDescFromSpec(outputSpec, outputDtype, "ND").ToAclType();
        }
        if (nullArg != "output_scale") {
            tensors.outputScale = BuildAclTensorDescFromSpec(outputScaleSpec, outputScaleDtype, "ND").ToAclType();
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
            } else if (optionalInput.compare(0, 7, "smooth|") == 0) {
                vector<string> smoothSpec;
                SplitStr2Vec(optionalInput, "|", smoothSpec);
                if (smoothSpec.size() != 4) {
                    ADD_FAILURE() << "Invalid smooth descriptor: " << optionalInput;
                    return ACLNN_ERR_PARAM_INVALID;
                }
                tensors.smoothScale =
                    BuildAclTensorDescFromSpec(smoothSpec[1], smoothSpec[2], smoothSpec[3]).ToAclType();
            }
        }
        tensors.SetTuningConfig(tuningConfig);
        const char* roundModePtr = roundMode == "NULL" ? nullptr : roundMode.c_str();
        return tensors.GetWorkspaceSize([&](uint64_t* workspaceSize, aclOpExecutor** executor) {
            const auto status = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
                tensors.x.get(), tensors.weight.get(), tensors.weightScale.get(), tensors.weightAssist.get(),
                tensors.bias.get(), tensors.xScale.get(), tensors.smoothScale.get(), tensors.groupList.get(),
                dequantMode, dequantDtype, quantMode, groupListType, tensors.tuningConfig.get(), swigluMode, clampLimit,
                gluAlpha, gluBias, roundModePtr, scaleAlg, dstTypeMax, tensors.output.get(), tensors.outputScale.get(),
                workspaceSize, executor);
            if (status == ACLNN_SUCCESS && tensors.output->IsEmpty()) {
                EXPECT_EQ(*workspaceSize, 0UL);
                EXPECT_NE(*executor, nullptr);
            }
            return status;
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

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, mxa8w4_requires_fp8_output_and_e8m0_scale)
{
    vector<string> cols;
    SplitStr2Vec("mxa8w4_output,ACLNN_SUCCESS,32:128,FLOAT8_E4M3FN,ND,"
                 "2:128:256#32768:1:128#2:2:16:16:64,FLOAT4_E2M1,FRACTAL_NZ,"
                 "2:2:256:2#1024:2:4:1,FLOAT8_E8M0,ND,32:2:2,FLOAT8_E8M0,2,INT64,"
                 "32:128,32:2:2,2,0,2,0,2,7.0,1.702,1.0,rint,0,0.0,none,NONE,none",
                 ",", cols);
    V3Case item;
    ASSERT_TRUE(ParseV3Case(cols, item));
    EXPECT_EQ(item.Run(), ACLNN_SUCCESS);
    item.outputDtype = "BF16";
    EXPECT_EQ(item.Run(), ACLNN_ERR_PARAM_INVALID);
    item.outputDtype = "FLOAT8_E4M3FN";
    item.outputScaleDtype = "FLOAT";
    EXPECT_EQ(item.Run(), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(grouped_matmul_swiglu_quant_weight_nz_v3_opapi_csv_test, get_workspace_size)
{
    CheckWorkspaceSize();
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, phase1_output_pointers_return_param_nullptr)
{
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kWorkspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(RunPhase1NullPointerCase(Phase1NullPointerTarget::kExecutor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, mxa8w4_requires_device_group_list_and_smooth_scale)
{
    for (int64_t groupListType : {int64_t{0}, GROUP_LIST_COUNTS}) {
        for (bool strided : {false, true}) {
            for (bool transposeWeight : {false, true}) {
                SCOPED_TRACE(testing::Message() << "groupListType=" << groupListType << ", strided=" << strided
                                                << ", transposeWeight=" << transposeWeight);
                const aclnnStatus expected = transposeWeight ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
                EXPECT_EQ(RunMxA8W4PlacementCase(false, false, strided, groupListType, transposeWeight), expected);
                EXPECT_EQ(RunMxA8W4PlacementCase(true, false, strided, groupListType, transposeWeight),
                          ACLNN_ERR_PARAM_INVALID);
                EXPECT_EQ(RunMxA8W4PlacementCase(false, true, strided, groupListType, transposeWeight),
                          ACLNN_ERR_PARAM_INVALID);
                EXPECT_EQ(RunMxA8W4PlacementCase(true, true, strided, groupListType, transposeWeight),
                          ACLNN_ERR_PARAM_INVALID);
            }
        }
    }
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, mxa8w4_accepts_torch_dense_scale_formats)
{
    for (int64_t groupListType : {int64_t{0}, GROUP_LIST_COUNTS}) {
        for (bool strided : {false, true}) {
            for (bool transposeWeight : {false, true}) {
                const aclnnStatus expected = transposeWeight ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
                EXPECT_EQ(RunMxA8W4PlacementCase(false, false, strided, groupListType, transposeWeight, true),
                          expected);
            }
        }
    }
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_strided_views_ignore_padding)
{
    EXPECT_EQ(RunHostGroupListCase({GROUP_HALF_M, GROUP_PADDING_SENTINEL, GROUP_TOTAL_M}, GROUP_PAIR_STRIDE, 0, 0),
              ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({GROUP_PADDING_SENTINEL, GROUP_HALF_M, GROUP_PADDING_SENTINEL, GROUP_TOTAL_M},
                                   GROUP_PAIR_STRIDE, PREFIX_STORAGE_OFFSET, 0),
              ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({FIRST_GROUP_LENGTH, GROUP_PADDING_SENTINEL, SECOND_GROUP_LENGTH}, GROUP_PAIR_STRIDE,
                                   0, GROUP_LIST_COUNTS),
              ACLNN_SUCCESS);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, host_group_list_contiguous_values_are_unchanged)
{
    EXPECT_EQ(RunHostGroupListCase({GROUP_HALF_M, GROUP_TOTAL_M}, GROUP_UNIT_STRIDE, 0, 0), ACLNN_SUCCESS);
    EXPECT_EQ(RunHostGroupListCase({FIRST_GROUP_LENGTH, SECOND_GROUP_LENGTH}, GROUP_UNIT_STRIDE, 0, GROUP_LIST_COUNTS),
              ACLNN_SUCCESS);
}

TEST(grouped_matmul_swiglu_quant_weight_nz_v3_opapi, a4w4_bf16_output_does_not_require_output_scale)
{
    SetupPlatformForCase();
    auto x = BuildAclTensorDescFromSpec("32:128", "FLOAT4_E2M1", "ND").ToAclType();
    auto weightTensor =
        BuildAclTensorDescFromSpec("1:128:128#16384:128:1#1:2:8:16:64", "FLOAT4_E2M1", "FRACTAL_NZ").ToAclType();
    auto weightScaleTensor = BuildAclTensorDescFromSpec("1:2:128:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto xScale = BuildAclTensorDescFromSpec("32:2:2", "FLOAT8_E8M0", "ND").ToAclType();
    auto groupList = BuildAclTensorDescFromSpec("1", "INT64", "ND").ToAclType();
    auto output = BuildAclTensorDescFromSpec("32:64", "BF16", "ND").ToAclType();

    const aclTensor* weightElements[] = {weightTensor.get()};
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weight(
        aclCreateTensorList(weightElements, SINGLE_TENSOR_COUNT), aclDestroyTensorList);
    ASSERT_NE(weight.get(), nullptr);
    (void)weightTensor.release();
    const aclTensor* weightScaleElements[] = {weightScaleTensor.get()};
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weightScale(
        aclCreateTensorList(weightScaleElements, SINGLE_TENSOR_COUNT), aclDestroyTensorList);
    ASSERT_NE(weightScale.get(), nullptr);
    (void)weightScaleTensor.release();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x.get(), weight.get(), weightScale.get(), nullptr, nullptr, xScale.get(), nullptr, groupList.get(),
        MX_QUANT_MODE, 0, MX_QUANT_MODE, 0, nullptr, SPLIT_SWIGLU_MODE, DEFAULT_CLAMP_LIMIT, DEFAULT_GLU_ALPHA,
        DEFAULT_GLU_BIAS, "rint", 0, 0.0, output.get(), nullptr, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
    executor = nullptr;
    auto smooth = BuildAclTensorDescFromSpec("1:64", "FLOAT", "ND").ToAclType();
    EXPECT_EQ(
        aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
            x.get(), weight.get(), weightScale.get(), nullptr, nullptr, xScale.get(), smooth.get(), groupList.get(),
            MX_QUANT_MODE, 0, MX_QUANT_MODE, 0, nullptr, SPLIT_SWIGLU_MODE, DEFAULT_CLAMP_LIMIT, DEFAULT_GLU_ALPHA,
            DEFAULT_GLU_BIAS, "rint", 0, 0.0, output.get(), nullptr, &workspaceSize, &executor),
        ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        (void)aclDestroyAclOpExecutor(executor);
    }
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
