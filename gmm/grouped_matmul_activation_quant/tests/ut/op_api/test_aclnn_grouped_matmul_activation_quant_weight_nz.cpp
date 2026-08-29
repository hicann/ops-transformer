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
 * \file test_aclnn_grouped_matmul_activation_quant_weight_nz.cpp
 * \brief CSV-driven opapi UT for aclnnGroupedMatmulActivationQuantWeightNz.
 */

#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_matmul_activation_quant_weight_nz.h"
#include "gmm_csv_acl_parse_utils.h"
#include "gmm_soc_version_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

namespace {

using ops::ut::BuildAclTensorListDesc;
using ops::ut::ParseBool;
using ops::ut::ParseDims;
using ops::ut::ParseDimsList;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kCsvColumnCount = 47UL;
constexpr int64_t kDefaultGroupListType = 0L;
constexpr int64_t kDefaultScaleAlg = 0L;
constexpr float kDefaultDstTypeMax = 0.0F;
constexpr const char *kActivationType = "gelu_tanh";
constexpr const char *kQuantMode = "mx";
constexpr const char *kRoundMode = "rint";
constexpr const char *kRunModeGetWorkspace = "GET_WORKSPACE";
constexpr const char *kRunModeWorkspaceOutputPtrNull = "WORKSPACE_OUTPUT_PTR_NULL";
constexpr const char *kRunModeRequiredTensorListNull = "REQUIRED_TENSORLIST_NULL";
constexpr const char *kRunModeEmptyTensorList = "EMPTY_TENSORLIST";
constexpr const char *kRunModeTensorListElementNull = "TENSORLIST_ELEMENT_NULL";
constexpr const char *kRunModeBiasEmptyTensorList = "BIAS_EMPTY_TENSORLIST";
constexpr const char *kRunModeOutputYNull = "OUTPUT_Y_NULL";
constexpr const char *kRunModePhase2NullExecutor = "PHASE2_NULL_EXECUTOR";
constexpr const char *kNullPtrSpec = "NULLPTR";

TensorDesc MakeTensorDesc(const string &shape, const string &dtype, const string &format,
                          const string &storageShape = "NONE", const string &stride = "NONE")
{
    return TensorDesc(ParseDims(shape), ops::ut::ParseAclDtype(dtype), ops::ut::ParseAclFormat(format),
                      ParseDims(stride), 0, ParseDims(storageShape))
        .ValueRange(-10, 10);
}

TensorListDesc MakeTensorListDesc(const string &shapes, const string &dtype, const string &format,
                                  const string &storageShapes = "NONE", const string &strides = "NONE")
{
    return BuildAclTensorListDesc(ParseDimsList(shapes), ops::ut::ParseAclDtype(dtype), ops::ut::ParseAclFormat(format),
                                  ParseDimsList(storageShapes), ParseDimsList(strides));
}

struct GroupedMatmulActivationQuantWeightNzOpApiCase {
    const char *GetQuantModePtr() const
    {
        return quantMode == kNullPtrSpec ? nullptr : quantMode.c_str();
    }

    aclnnStatus RunGetWorkspaceWithNullY(const TensorDesc &xDesc, const TensorDesc &groupListDesc,
                                         const TensorListDesc &weightDesc, const TensorListDesc &weightScaleDesc,
                                         const TensorDesc &xScaleDesc, const TensorDesc &yScaleDesc,
                                         uint64_t *workspaceSize, aclOpExecutor **executor) const
    {
        auto x = xDesc.ToAclType();
        auto groupList = groupListDesc.ToAclType();
        auto weight = weightDesc.ToAclType();
        auto weightScale = weightScaleDesc.ToAclType();
        auto xScale = xScaleDesc.ToAclType();
        auto yScale = yScaleDesc.ToAclType();
        return aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
            x.get(), groupList.get(), weight.get(), weightScale.get(), nullptr, xScale.get(), activationType.c_str(),
            groupListType, nullptr, GetQuantModePtr(), roundMode.c_str(), scaleAlg, dstTypeMax, nullptr, yScale.get(),
            workspaceSize, executor);
    }

    aclnnStatus RunGetWorkspaceWithRequiredTensorListNull(const TensorDesc &xDesc, const TensorDesc &groupListDesc,
                                                          const TensorDesc &xScaleDesc, const TensorDesc &yDesc,
                                                          const TensorDesc &yScaleDesc, uint64_t *workspaceSize,
                                                          aclOpExecutor **executor) const
    {
        auto x = xDesc.ToAclType();
        auto groupList = groupListDesc.ToAclType();
        auto xScale = xScaleDesc.ToAclType();
        auto y = yDesc.ToAclType();
        auto yScale = yScaleDesc.ToAclType();
        return aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
            x.get(), groupList.get(), nullptr, nullptr, nullptr, xScale.get(), activationType.c_str(), groupListType,
            nullptr, GetQuantModePtr(), roundMode.c_str(), scaleAlg, dstTypeMax, y.get(), yScale.get(), workspaceSize,
            executor);
    }

    aclnnStatus RunGetWorkspaceWithEmptyTensorList(const TensorDesc &xDesc, const TensorDesc &groupListDesc,
                                                   const TensorDesc &xScaleDesc, const TensorDesc &yDesc,
                                                   const TensorDesc &yScaleDesc, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor) const
    {
        auto x = xDesc.ToAclType();
        auto groupList = groupListDesc.ToAclType();
        auto xScale = xScaleDesc.ToAclType();
        auto y = yDesc.ToAclType();
        auto yScale = yScaleDesc.ToAclType();
        unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> emptyList(aclCreateTensorList(nullptr, 0),
                                                                             aclDestroyTensorList);
        return aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
            x.get(), groupList.get(), emptyList.get(), emptyList.get(), nullptr, xScale.get(), activationType.c_str(),
            groupListType, nullptr, GetQuantModePtr(), roundMode.c_str(), scaleAlg, dstTypeMax, y.get(), yScale.get(),
            workspaceSize, executor);
    }

    aclnnStatus RunGetWorkspaceWithTensorListElementNull(const TensorDesc &xDesc, const TensorDesc &groupListDesc,
                                                         const TensorListDesc &weightScaleDesc,
                                                         const TensorDesc &xScaleDesc, const TensorDesc &yDesc,
                                                         const TensorDesc &yScaleDesc, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor) const
    {
        auto x = xDesc.ToAclType();
        auto groupList = groupListDesc.ToAclType();
        auto weightScale = weightScaleDesc.ToAclType();
        auto xScale = xScaleDesc.ToAclType();
        auto y = yDesc.ToAclType();
        auto yScale = yScaleDesc.ToAclType();
        aclTensor *nullTensor = nullptr;
        unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> weightList(aclCreateTensorList(&nullTensor, 1),
                                                                              aclDestroyTensorList);
        return aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
            x.get(), groupList.get(), weightList.get(), weightScale.get(), nullptr, xScale.get(),
            activationType.c_str(), groupListType, nullptr, GetQuantModePtr(), roundMode.c_str(), scaleAlg, dstTypeMax,
            y.get(), yScale.get(), workspaceSize, executor);
    }

    aclnnStatus RunGetWorkspaceWithEmptyBiasTensorList(const TensorDesc &xDesc, const TensorDesc &groupListDesc,
                                                       const TensorListDesc &weightDesc,
                                                       const TensorListDesc &weightScaleDesc,
                                                       const TensorDesc &xScaleDesc, const TensorDesc &yDesc,
                                                       const TensorDesc &yScaleDesc, uint64_t *workspaceSize,
                                                       aclOpExecutor **executor) const
    {
        auto x = xDesc.ToAclType();
        auto groupList = groupListDesc.ToAclType();
        auto weight = weightDesc.ToAclType();
        auto weightScale = weightScaleDesc.ToAclType();
        auto xScale = xScaleDesc.ToAclType();
        auto y = yDesc.ToAclType();
        auto yScale = yScaleDesc.ToAclType();
        unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> emptyBias(aclCreateTensorList(nullptr, 0),
                                                                             aclDestroyTensorList);
        return aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
            x.get(), groupList.get(), weight.get(), weightScale.get(), emptyBias.get(), xScale.get(),
            activationType.c_str(), groupListType, nullptr, GetQuantModePtr(), roundMode.c_str(), scaleAlg, dstTypeMax,
            y.get(), yScale.get(), workspaceSize, executor);
    }

    void Run() const
    {
        if (!enable) {
            GTEST_SKIP() << "Skip disabled case: " << caseName;
        }
        ops::ut::SetPlatformSocVersion(socVersion);

        if (runMode == kRunModeWorkspaceOutputPtrNull) {
            EXPECT_NE(aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, kActivationType, kDefaultGroupListType,
                          nullptr, kQuantMode, kRoundMode, kDefaultScaleAlg, kDefaultDstTypeMax, nullptr, nullptr,
                          nullptr, nullptr),
                      ACLNN_SUCCESS)
                << "case=" << caseName;
            return;
        }
        if (runMode == kRunModePhase2NullExecutor) {
            EXPECT_NE(aclnnGroupedMatmulActivationQuantWeightNz(nullptr, 0, nullptr, nullptr), ACLNN_SUCCESS)
                << "case=" << caseName;
            return;
        }

        TensorDesc xDesc = MakeTensorDesc(xShape, xDtype, xFormat, xStorageShape, xStride);
        TensorDesc groupListDesc = MakeTensorDesc(groupListShape, groupListDtype, groupListFormat);
        TensorListDesc weightDesc =
            MakeTensorListDesc(weightShapes, weightDtype, weightFormat, weightStorageShapes, weightStrides);
        TensorListDesc weightScaleDesc = MakeTensorListDesc(weightScaleShapes, weightScaleDtype, weightScaleFormat,
                                                            weightScaleStorageShapes, weightScaleStrides);
        TensorDesc yDesc = MakeTensorDesc(yShape, yDtype, yFormat);
        TensorDesc yScaleDesc = MakeTensorDesc(yScaleShape, yScaleDtype, yScaleFormat);

        const char *activationTypePtr = activationTypeNull ? nullptr : activationType.c_str();
        const char *roundModePtr = roundModeNull ? nullptr : roundMode.c_str();

        vector<int64_t> tuningValues = ops::ut::ParseI64List(tuningConfig);
        aclIntArray *tuningConfigArr =
            tuningValues.empty() ? nullptr : aclCreateIntArray(tuningValues.data(), tuningValues.size());

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = ACLNN_SUCCESS;
        if (runMode == kRunModeRequiredTensorListNull) {
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            ret = RunGetWorkspaceWithRequiredTensorListNull(xDesc, groupListDesc, xScaleDesc, yDesc, yScaleDesc,
                                                            &workspaceSize, &executor);
        } else if (runMode == kRunModeEmptyTensorList) {
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            ret = RunGetWorkspaceWithEmptyTensorList(xDesc, groupListDesc, xScaleDesc, yDesc, yScaleDesc,
                                                     &workspaceSize, &executor);
        } else if (runMode == kRunModeTensorListElementNull) {
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            ret = RunGetWorkspaceWithTensorListElementNull(xDesc, groupListDesc, weightScaleDesc, xScaleDesc, yDesc,
                                                           yScaleDesc, &workspaceSize, &executor);
        } else if (runMode == kRunModeBiasEmptyTensorList) {
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            ret = RunGetWorkspaceWithEmptyBiasTensorList(xDesc, groupListDesc, weightDesc, weightScaleDesc, xScaleDesc,
                                                         yDesc, yScaleDesc, &workspaceSize, &executor);
        } else if (runMode == kRunModeOutputYNull) {
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            ret = RunGetWorkspaceWithNullY(xDesc, groupListDesc, weightDesc, weightScaleDesc, xScaleDesc, yScaleDesc,
                                           &workspaceSize, &executor);
        } else if (runMode != kRunModeGetWorkspace) {
            if (tuningConfigArr != nullptr) {
                aclDestroyIntArray(tuningConfigArr);
            }
            ADD_FAILURE() << "Unsupported runMode: " << runMode << ", case=" << caseName;
            return;
        } else if (biasNull && xScaleNull) {
            auto ut =
                OP_API_UT(aclnnGroupedMatmulActivationQuantWeightNz,
                          INPUT(xDesc, groupListDesc, weightDesc, weightScaleDesc, nullptr, nullptr, activationTypePtr,
                                groupListType, tuningConfigArr, GetQuantModePtr(), roundModePtr, scaleAlg, dstTypeMax),
                          OUTPUT(yDesc, yScaleDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (biasNull) {
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            auto ut = OP_API_UT(
                aclnnGroupedMatmulActivationQuantWeightNz,
                INPUT(xDesc, groupListDesc, weightDesc, weightScaleDesc, nullptr, xScaleDesc, activationTypePtr,
                      groupListType, tuningConfigArr, GetQuantModePtr(), roundModePtr, scaleAlg, dstTypeMax),
                OUTPUT(yDesc, yScaleDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (xScaleNull) {
            TensorListDesc biasDesc = MakeTensorListDesc(biasShapes, biasDtype, biasFormat);
            auto ut =
                OP_API_UT(aclnnGroupedMatmulActivationQuantWeightNz,
                          INPUT(xDesc, groupListDesc, weightDesc, weightScaleDesc, biasDesc, nullptr, activationTypePtr,
                                groupListType, tuningConfigArr, GetQuantModePtr(), roundModePtr, scaleAlg, dstTypeMax),
                          OUTPUT(yDesc, yScaleDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            TensorListDesc biasDesc = MakeTensorListDesc(biasShapes, biasDtype, biasFormat);
            TensorDesc xScaleDesc = MakeTensorDesc(xScaleShape, xScaleDtype, xScaleFormat);
            auto ut = OP_API_UT(
                aclnnGroupedMatmulActivationQuantWeightNz,
                INPUT(xDesc, groupListDesc, weightDesc, weightScaleDesc, biasDesc, xScaleDesc, activationTypePtr,
                      groupListType, tuningConfigArr, GetQuantModePtr(), roundModePtr, scaleAlg, dstTypeMax),
                OUTPUT(yDesc, yScaleDesc));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }

        if (checkRet) {
            EXPECT_EQ(ret, ops::ut::ParseAclnnStatus(expectRet)) << "case=" << caseName;
        }
        if (expectWorkspaceZero) {
            EXPECT_EQ(workspaceSize, 0U) << "case=" << caseName;
        }
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }

    string socVersion;
    string caseName;
    bool enable = true;
    bool checkRet = true;
    string expectRet;
    bool expectWorkspaceZero = false;
    string xShape;
    string xDtype;
    string xFormat;
    string xStorageShape;
    string xStride;
    string groupListShape;
    string groupListDtype;
    string groupListFormat;
    string weightShapes;
    string weightDtype;
    string weightFormat;
    string weightStorageShapes;
    string weightStrides;
    string weightScaleShapes;
    string weightScaleDtype;
    string weightScaleFormat;
    string weightScaleStorageShapes;
    string weightScaleStrides;
    bool biasNull = true;
    string biasShapes;
    string biasDtype;
    string biasFormat;
    bool xScaleNull = false;
    string xScaleShape;
    string xScaleDtype;
    string xScaleFormat;
    bool activationTypeNull = false;
    string activationType;
    int64_t groupListType = kDefaultGroupListType;
    string tuningConfig;
    string quantMode;
    bool roundModeNull = false;
    string roundMode;
    int64_t scaleAlg = kDefaultScaleAlg;
    float dstTypeMax = kDefaultDstTypeMax;
    string yShape;
    string yDtype;
    string yFormat;
    string yScaleShape;
    string yScaleDtype;
    string yScaleFormat;
    string runMode = kRunModeGetWorkspace;
};

vector<GroupedMatmulActivationQuantWeightNzOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulActivationQuantWeightNzOpApiCase> cases;
    string line;
    bool headerSkipped = false;
    size_t lineNo = 0U;
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
        if (cols.size() < kCsvColumnCount) {
            continue;
        }

        const string caseName = cols.size() > 1U ? Trim(cols[1]) : "";
        try {
            GroupedMatmulActivationQuantWeightNzOpApiCase c;
            size_t i = 0U;
            c.socVersion = Trim(cols[i++]);
            c.caseName = Trim(cols[i++]);
            c.enable = ParseBool(Trim(cols[i++]));
            c.checkRet = ParseBool(Trim(cols[i++]));
            c.expectRet = Trim(cols[i++]);
            c.expectWorkspaceZero = ParseBool(Trim(cols[i++]));
            c.xShape = Trim(cols[i++]);
            c.xDtype = Trim(cols[i++]);
            c.xFormat = Trim(cols[i++]);
            c.xStorageShape = Trim(cols[i++]);
            c.xStride = Trim(cols[i++]);
            c.groupListShape = Trim(cols[i++]);
            c.groupListDtype = Trim(cols[i++]);
            c.groupListFormat = Trim(cols[i++]);
            c.weightShapes = Trim(cols[i++]);
            c.weightDtype = Trim(cols[i++]);
            c.weightFormat = Trim(cols[i++]);
            c.weightStorageShapes = Trim(cols[i++]);
            c.weightStrides = Trim(cols[i++]);
            c.weightScaleShapes = Trim(cols[i++]);
            c.weightScaleDtype = Trim(cols[i++]);
            c.weightScaleFormat = Trim(cols[i++]);
            c.weightScaleStorageShapes = Trim(cols[i++]);
            c.weightScaleStrides = Trim(cols[i++]);
            c.biasNull = ParseBool(Trim(cols[i++]));
            c.biasShapes = Trim(cols[i++]);
            c.biasDtype = Trim(cols[i++]);
            c.biasFormat = Trim(cols[i++]);
            c.xScaleNull = ParseBool(Trim(cols[i++]));
            c.xScaleShape = Trim(cols[i++]);
            c.xScaleDtype = Trim(cols[i++]);
            c.xScaleFormat = Trim(cols[i++]);
            c.activationTypeNull = ParseBool(Trim(cols[i++]));
            c.activationType = Trim(cols[i++]);
            c.groupListType = stoll(Trim(cols[i++]));
            c.tuningConfig = Trim(cols[i++]);
            c.quantMode = Trim(cols[i++]);
            c.roundModeNull = ParseBool(Trim(cols[i++]));
            c.roundMode = Trim(cols[i++]);
            c.scaleAlg = stoll(Trim(cols[i++]));
            c.dstTypeMax = stod(Trim(cols[i++]));
            c.yShape = Trim(cols[i++]);
            c.yDtype = Trim(cols[i++]);
            c.yFormat = Trim(cols[i++]);
            c.yScaleShape = Trim(cols[i++]);
            c.yScaleDtype = Trim(cols[i++]);
            c.yScaleFormat = Trim(cols[i++]);
            if (i < cols.size()) {
                c.runMode = Trim(cols[i++]);
            }
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<GroupedMatmulActivationQuantWeightNzOpApiCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class grouped_matmul_activation_quant_weight_nz_opapi_csv_test
    : public testing::TestWithParam<GroupedMatmulActivationQuantWeightNzOpApiCase> {};

TEST_P(grouped_matmul_activation_quant_weight_nz_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(grouped_matmul_activation_quant_weight_nz_opapi_csv,
                         grouped_matmul_activation_quant_weight_nz_opapi_csv_test,
                         testing::ValuesIn(LoadCases(
                             ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_activation_quant_weight_nz.csv",
                                                     "gmm/grouped_matmul_activation_quant/tests/ut/op_api", __FILE__))),
                         BuildCaseName);

} // namespace
