/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_grouped_matmul_swiglu_quant_v2.cpp
 * \brief CSV-driven opapi UT for grouped_matmul_swiglu_quant_v2 (aligned with gmm grouped_matmul opapi UT style).
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "gmm_csv_acl_parse_utils.h"

using namespace std;
using namespace op;

namespace {

using ops::ut::ParseBool;
using ops::ut::ParseDims;
using ops::ut::ParseDimsList;
using ops::ut::ParseI64List;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;
using ops::ut::BuildAclTensorListDesc;
using ops::ut::MakeAclTensorDesc;

constexpr size_t kSwigluCsvColumnCount = 45;

aclDataType ParseDtype(const string &dtype) { return ops::ut::ParseAclDtype(dtype); }

aclFormat ParseFormat(const string &format) { return ops::ut::ParseAclFormat(format); }

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, op::SocVersion> socMap = {
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    auto it = socMap.find(socVersion);
    op::SetPlatformSocVersion(it == socMap.end() ? op::SocVersion::ASCEND910B : it->second);
}

void FallbackSingleNzWeightToNdStorage(const string &api, aclFormat &weightFormat,
                                       vector<vector<int64_t>> &weightShapesVec,
                                       vector<vector<int64_t>> &weightStorageShapesVec)
{
    if (api != "WeightNzV2" || weightFormat != ACL_FORMAT_FRACTAL_NZ) {
        return;
    }
    if (weightShapesVec.size() != 1 || !weightStorageShapesVec.empty()) {
        return;
    }
    const auto &w = weightShapesVec[0];
    if (w.size() != 5 || w[0] <= 0 || w[3] != 16 || w[4] != 32) {
        return;
    }
    // Some environments crash while creating single 5D FRACTAL_NZ desc in TensorList.
    // Fallback to equivalent ND view + NZ storage, same as stable hand-written UT path.
    const int64_t e = w[0];
    const int64_t n = w[1] * w[4];
    const int64_t k = w[2] * w[3];
    weightShapesVec[0] = {e, k, n};
    weightStorageShapesVec = {w};
    weightFormat = ACL_FORMAT_ND;
}

struct SwigluOpApiCase {
    string socVersion;
    string caseName;
    string api;
    string checkRet;
    uint64_t expectRet;
    string xShape;
    string xDtype;
    string xFormat;
    string xStorageShape;
    string xStride;
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
    string weightAssistShapes;
    string weightAssistDtype;
    string weightAssistFormat;
    string weightAssistStorageShapes;
    string weightAssistStrides;
    string xScaleShape;
    string xScaleDtype;
    string xScaleFormat;
    string smoothScaleShape;
    string smoothScaleDtype;
    string smoothScaleFormat;
    string groupListShape;
    string groupListDtype;
    string groupListFormat;
    int64_t dequantMode;
    int64_t dequantDtype;
    int64_t quantMode;
    int64_t quantDtype;
    string tuningConfig;
    string out1Shape;
    string out1Dtype;
    string out1Format;
    string out2Shape;
    string out2Dtype;
    string out2Format;

    void Run() const
    {
        SetupPlatformForCase(socVersion);
        // Same tensor style as grouped_matmul_swiglu_quant_v2_opapi_csv_test hand-written UT:
        // TensorDesc(...).ValueRange(-10, 10) and TensorListDesc(vector<TensorDesc>).
        auto weightShapesVec = ParseDimsList(weightShapes);
        auto weightStorageShapesVec = ParseDimsList(weightStorageShapes);
        auto weightFormatVal = ParseFormat(weightFormat);
        auto weightScaleShapesVec = ParseDimsList(weightScaleShapes);
        auto weightStridesVec = ParseDimsList(weightStrides);
        auto weightScaleStorageShapesVec = ParseDimsList(weightScaleStorageShapes);
        auto weightScaleStridesVec = ParseDimsList(weightScaleStrides);
        FallbackSingleNzWeightToNdStorage(api, weightFormatVal, weightShapesVec, weightStorageShapesVec);

        TensorDesc x_desc = MakeAclTensorDesc(ParseDims(xShape), ParseDtype(xDtype), ParseFormat(xFormat),
                                              ParseDims(xStorageShape), ParseDims(xStride));
        TensorListDesc weight_desc = BuildAclTensorListDesc(weightShapesVec, ParseDtype(weightDtype), weightFormatVal,
                                                            weightStorageShapesVec, weightStridesVec);

        TensorDesc xScale_desc =
            TensorDesc(ParseDims(xScaleShape), ParseDtype(xScaleDtype), ParseFormat(xScaleFormat)).ValueRange(-10, 10);
        TensorDesc groupList_desc =
            TensorDesc(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat))
                .ValueRange(-10, 10);
        TensorDesc out1_desc =
            TensorDesc(ParseDims(out1Shape), ParseDtype(out1Dtype), ParseFormat(out1Format)).ValueRange(-10, 10);
        TensorDesc out2_desc =
            TensorDesc(ParseDims(out2Shape), ParseDtype(out2Dtype), ParseFormat(out2Format)).ValueRange(-10, 10);

        const auto weightAssistShapesVec = ParseDimsList(weightAssistShapes);
        const auto smoothScaleDims = ParseDims(smoothScaleShape);
        const bool hasWeightScale = !weightScaleShapesVec.empty();
        const bool hasWeightAssist = !weightAssistShapesVec.empty();
        const bool hasSmoothScale = !smoothScaleDims.empty();

        vector<int64_t> tuningValues = ParseI64List(tuningConfig, "|");
        aclIntArray *tuningConfigArr =
            tuningValues.empty() ? nullptr : aclCreateIntArray(tuningValues.data(), tuningValues.size());

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACL_SUCCESS;

        // Optional inputs must use nullptr; do not build empty TensorListDesc.
        if (!hasWeightScale) {
            if (hasWeightAssist && hasSmoothScale) {
                vector<TensorDesc> weight_assist_tensor_vec;
                weight_assist_tensor_vec.reserve(weightAssistShapesVec.size());
                for (const auto &dims : weightAssistShapesVec) {
                    weight_assist_tensor_vec.emplace_back(
                        TensorDesc(dims, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat))
                            .ValueRange(-10, 10));
                }
                TensorListDesc weight_assist_desc(weight_assist_tensor_vec);
                TensorDesc smoothScale_desc =
                    TensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat))
                        .ValueRange(-10, 10);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, nullptr, weight_assist_desc, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, nullptr, weight_assist_desc, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasWeightAssist) {
                vector<TensorDesc> weight_assist_tensor_vec_b;
                weight_assist_tensor_vec_b.reserve(weightAssistShapesVec.size());
                for (const auto &dims : weightAssistShapesVec) {
                    weight_assist_tensor_vec_b.emplace_back(
                        TensorDesc(dims, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat))
                            .ValueRange(-10, 10));
                }
                TensorListDesc weight_assist_desc(weight_assist_tensor_vec_b);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, nullptr, weight_assist_desc, nullptr, xScale_desc, nullptr,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, nullptr, weight_assist_desc, nullptr, xScale_desc, nullptr,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasSmoothScale) {
                TensorDesc smoothScale_desc =
                    TensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat))
                        .ValueRange(-10, 10);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else {
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc, nullptr, groupList_desc,
                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc, nullptr, groupList_desc,
                              dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            }
        } else {
            TensorListDesc weight_scale_desc = BuildAclTensorListDesc(
                weightScaleShapesVec, ParseDtype(weightScaleDtype), ParseFormat(weightScaleFormat),
                weightScaleStorageShapesVec, weightScaleStridesVec);
            if (hasWeightAssist && hasSmoothScale) {
                vector<TensorDesc> weight_assist_tensor_vec_c;
                weight_assist_tensor_vec_c.reserve(weightAssistShapesVec.size());
                for (const auto &dims : weightAssistShapesVec) {
                    weight_assist_tensor_vec_c.emplace_back(
                        TensorDesc(dims, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat))
                            .ValueRange(-10, 10));
                }
                TensorListDesc weight_assist_desc(weight_assist_tensor_vec_c);
                TensorDesc smoothScale_desc =
                    TensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat))
                        .ValueRange(-10, 10);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_desc, nullptr,
                                              xScale_desc, smoothScale_desc, groupList_desc, dequantMode, dequantDtype,
                                              quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_desc, nullptr,
                                              xScale_desc, smoothScale_desc, groupList_desc, dequantMode, dequantDtype,
                                              quantMode, quantDtype, tuningConfigArr),
                                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasWeightAssist) {
                vector<TensorDesc> weight_assist_tensor_vec_d;
                weight_assist_tensor_vec_d.reserve(weightAssistShapesVec.size());
                for (const auto &dims : weightAssistShapesVec) {
                    weight_assist_tensor_vec_d.emplace_back(
                        TensorDesc(dims, ParseDtype(weightAssistDtype), ParseFormat(weightAssistFormat))
                            .ValueRange(-10, 10));
                }
                TensorListDesc weight_assist_desc(weight_assist_tensor_vec_d);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_desc, nullptr, xScale_desc, nullptr,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_desc, nullptr, xScale_desc, nullptr,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else if (hasSmoothScale) {
                TensorDesc smoothScale_desc =
                    TensorDesc(smoothScaleDims, ParseDtype(smoothScaleDtype), ParseFormat(smoothScaleFormat))
                        .ValueRange(-10, 10);
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            } else {
                if (api == "WeightNzV2") {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                } else {
                    auto ut = OP_API_UT(
                        aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, dequantMode, dequantDtype, quantMode, quantDtype, tuningConfigArr),
                        OUTPUT(out1_desc, out2_desc));
                    ret = ut.TestGetWorkspaceSize(&workspaceSize);
                }
            }
        }

        if (tuningConfigArr != nullptr) {
            aclDestroyIntArray(tuningConfigArr);
        }
        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, static_cast<aclnnStatus>(expectRet));
        }
    }
};

vector<SwigluOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<SwigluOpApiCase> cases;
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
        if (cols.size() != kSwigluCsvColumnCount) {
            continue;
        }
        const string caseName = cols.size() > 1U ? Trim(cols[1]) : "";
        try {
            SwigluOpApiCase c;
            size_t i = 0;
            c.socVersion = Trim(cols[i++]);
            c.caseName = Trim(cols[i++]);
            c.api = Trim(cols[i++]);
            c.checkRet = Trim(cols[i++]);
            c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
            c.xShape = Trim(cols[i++]);
            c.xDtype = Trim(cols[i++]);
            c.xFormat = Trim(cols[i++]);
            c.xStorageShape = Trim(cols[i++]);
            c.xStride = Trim(cols[i++]);
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
            c.weightAssistShapes = Trim(cols[i++]);
            c.weightAssistDtype = Trim(cols[i++]);
            c.weightAssistFormat = Trim(cols[i++]);
            c.weightAssistStorageShapes = Trim(cols[i++]);
            c.weightAssistStrides = Trim(cols[i++]);
            c.xScaleShape = Trim(cols[i++]);
            c.xScaleDtype = Trim(cols[i++]);
            c.xScaleFormat = Trim(cols[i++]);
            c.smoothScaleShape = Trim(cols[i++]);
            c.smoothScaleDtype = Trim(cols[i++]);
            c.smoothScaleFormat = Trim(cols[i++]);
            c.groupListShape = Trim(cols[i++]);
            c.groupListDtype = Trim(cols[i++]);
            c.groupListFormat = Trim(cols[i++]);
            c.dequantMode = stoll(Trim(cols[i++]));
            c.dequantDtype = stoll(Trim(cols[i++]));
            c.quantMode = stoll(Trim(cols[i++]));
            c.quantDtype = stoll(Trim(cols[i++]));
            c.tuningConfig = Trim(cols[i++]);
            c.out1Shape = Trim(cols[i++]);
            c.out1Dtype = Trim(cols[i++]);
            c.out1Format = Trim(cols[i++]);
            c.out2Shape = Trim(cols[i++]);
            c.out2Dtype = Trim(cols[i++]);
            c.out2Format = Trim(cols[i++]);
            if (c.socVersion == "Ascend950" || c.socVersion == "Ascend910B") {
                cases.emplace_back(c);
            }
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<SwigluOpApiCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class grouped_matmul_swiglu_quant_v2_opapi_csv_test : public testing::TestWithParam<SwigluOpApiCase> {};

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, direct_launch_entry_error_case)
{
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantV2(nullptr, 0, nullptr, nullptr), ACL_SUCCESS);
    EXPECT_NE(aclnnGroupedMatmulSwigluQuantWeightNzV2(nullptr, 0, nullptr, nullptr), ACL_SUCCESS);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_normal_nz_workspace_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_wrong_nd_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_wrong_nd_no_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a8_without_weight_assist_matrix_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_redundant_weight_assist_matrix_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc weight_assist_matrix = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_assist_matrix_desc = TensorListDesc({weight_assist_matrix});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_matrix_desc, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_smoothscale_1d_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_smoothscale_2d_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e, n / 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_smoothscale_invalid_dim_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e, n / 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_smoothscale_wrong_shape_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, smoothScale_desc,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w4a4_wtrans_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {e, 1, k}, 0, {e, k / 64, n / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_nz_workspace_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(grouped_matmul_swiglu_quant_v2_opapi_csv_test, ascend910B2_test_opapi_w8a8_multi_weight_wrong_nd_no_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = {10};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc, nullptr,
                              groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_P(grouped_matmul_swiglu_quant_v2_opapi_csv_test, run_case) { GetParam().Run(); }

INSTANTIATE_TEST_SUITE_P(grouped_matmul_swiglu_quant_v2_opapi_csv, grouped_matmul_swiglu_quant_v2_opapi_csv_test,
                         testing::ValuesIn(LoadCases(
                             ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_swiglu_quant_v2.csv",
                                                     "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_api", __FILE__))),
                         BuildCaseName);

} // namespace
