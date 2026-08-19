/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kWeightNzV2CsvColumnCount = 53;

vector<int64_t> ParseDims(const string &value)
{
    return ops::ut::ParseDims(value);
}

vector<int64_t> ParseI64List(const string &value)
{
    return ops::ut::ParseI64List(value);
}

aclDataType ParseDtype(const string &dtype)
{
    return ops::ut::ParseAclDtype(dtype);
}

aclFormat ParseFormat(const string &format)
{
    static const map<string, aclFormat> extraFormatMap = {
        {"FRACTAL_NZ_C0_8", ACL_FORMAT_FRACTAL_NZ_C0_8},
        {"FRACTAL_NZ_C0_16", ACL_FORMAT_FRACTAL_NZ_C0_16},
        {"FRACTAL_NZ_C0_32", ACL_FORMAT_FRACTAL_NZ_C0_32},
    };
    auto it = extraFormatMap.find(Trim(format));
    if (it != extraFormatMap.end()) {
        return it->second;
    }
    return ops::ut::ParseAclFormat(format);
}

aclnnStatus ParseStatus(const string &status)
{
    static const map<string, aclnnStatus> statusMap = {
        {"SUCCESS", ACLNN_SUCCESS},
        {"ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
    };
    auto it = statusMap.find(Trim(status));
    return it == statusMap.end() ? ACLNN_SUCCESS : it->second;
}

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, SocVersion> socMap = {
        {"Ascend910B", SocVersion::ASCEND910B},
        {"Ascend950", SocVersion::ASCEND950},
    };
    auto it = socMap.find(Trim(socVersion));
    SetPlatformSocVersion(it == socMap.end() ? SocVersion::ASCEND910B : it->second);
}

TensorDesc MakeTensorDesc(const vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                          const vector<int64_t> &storageShape, bool fillRange = true)
{
    if (fillRange) {
        return ops::ut::MakeAclTensorDesc(shape, dtype, format, storageShape).ValueRange(-1, 1);
    }
    return storageShape.empty() ? TensorDesc(shape, dtype, format) : TensorDesc(shape, dtype, format, {}, 0, storageShape);
}

struct GroupedMatmulFinalizeRoutingWeightNzV2Case {
    void Run() const
    {
        SetupPlatformForCase(socVersion);

        TensorDesc x1 = MakeTensorDesc(ParseDims(x1Shape), ParseDtype(x1Dtype), ACL_FORMAT_ND, {}, ParseBool(x1UseRange));
        TensorDesc x2 = MakeTensorDesc(ParseDims(x2Shape), ParseDtype(x2Dtype), ParseFormat(x2Format),
                                       ParseDims(x2StorageShape), ParseBool(x2UseRange));
        TensorDesc scale =
            MakeTensorDesc(ParseDims(scaleShape), ParseDtype(scaleDtype), ACL_FORMAT_ND, {}, ParseBool(scaleUseRange));
        if (ParseBool(scaleUseRange)) {
            scale.ValueRange(0, 3);
        }
        TensorDesc bias = MakeTensorDesc(ParseDims(biasShape), ParseDtype(biasDtype), ACL_FORMAT_ND, {}, true);
        TensorDesc offset = MakeTensorDesc(ParseDims(offsetShape), ParseDtype(offsetDtype), ACL_FORMAT_ND, {}, true);
        TensorDesc perTokenScale =
            MakeTensorDesc(ParseDims(perTokenScaleShape), ParseDtype(perTokenScaleDtype), ACL_FORMAT_ND, {},
                           ParseBool(perTokenScaleUseRange));
        if (ParseBool(perTokenScaleUseRange)) {
            perTokenScale.ValueRange(0, 3);
        }
        TensorDesc groupList =
            MakeTensorDesc(ParseDims(groupListShape), ACL_INT64, ACL_FORMAT_ND, {}, ParseBool(groupListUseRange));
        if (ParseBool(groupListUseRange)) {
            groupList.ValueRange(groupListRangeVal, groupListRangeVal);
        }
        const vector<int64_t> groupListVals = ParseI64List(groupListValues);
        if (!groupListVals.empty()) {
            groupList.Value(groupListVals);
        }
        TensorDesc sharedInput =
            MakeTensorDesc(ParseDims(sharedInputShape), ParseDtype(sharedInputDtype), ACL_FORMAT_ND, {}, ParseBool(sharedInputUseRange));
        TensorDesc logits =
            MakeTensorDesc(ParseDims(logitsShape), ParseDtype(logitsDtype), ACL_FORMAT_ND, {}, ParseBool(logitsUseRange));
        TensorDesc rowIndex =
            MakeTensorDesc(ParseDims(rowIndexShape), ParseDtype(rowIndexDtype), ACL_FORMAT_ND, {}, ParseBool(rowIndexUseRange));
        TensorDesc out = MakeTensorDesc(ParseDims(outShape), ParseDtype(outDtype), ACL_FORMAT_ND, {}, ParseBool(outUseRange));

        const vector<int64_t> tuningVals = ParseI64List(tuningConfig);
        aclIntArray *tuningConfigPtr = nullptr;
        if (ParseBool(hasTuningConfig)) {
            tuningConfigPtr = aclCreateIntArray(const_cast<int64_t *>(tuningVals.data()), tuningVals.size());
        }

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACLNN_SUCCESS;
        if (ParseBool(hasAntiquantScale)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, nullptr, nullptr, offset, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasAntiquantOffset)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, nullptr, nullptr, nullptr, offset, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasScale)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (!ParseBool(hasLogits) && !ParseBool(hasRowIndex)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, nullptr, nullptr, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, nullptr, nullptr, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasBias) && ParseBool(hasOffset)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, bias, offset, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (ParseBool(hasBias)) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, bias, nullptr, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, nullptr, nullptr, nullptr, nullptr, perTokenScale, groupList,
                                      sharedInput, logits, rowIndex, dtype, sharedInputWeight, sharedInputOffset,
                                      transposeX, transposeW, groupListType, tuningConfigPtr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }

        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, ParseStatus(expectRet));
        }
    }

    string socVersion;
    string caseName;
    string x1Shape;
    string x1Dtype;
    string x1UseRange;
    string x2Shape;
    string x2Dtype;
    string x2Format;
    string x2StorageShape;
    string x2UseRange;
    string scaleShape;
    string scaleDtype;
    string scaleUseRange;
    string biasShape;
    string biasDtype;
    string offsetShape;
    string offsetDtype;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string perTokenScaleUseRange;
    string groupListShape;
    string groupListValues;
    string groupListUseRange;
    int64_t groupListRangeVal = 0;
    string sharedInputShape;
    string sharedInputDtype;
    string sharedInputUseRange;
    string logitsShape;
    string logitsDtype;
    string logitsUseRange;
    string rowIndexShape;
    string rowIndexDtype;
    string rowIndexUseRange;
    string outShape;
    string outDtype;
    string outUseRange;
    int64_t dtype = 0;
    float sharedInputWeight = 1.0F;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = false;
    int64_t groupListType = 1;
    string tuningConfig;
    string hasTuningConfig;
    string hasScale;
    string hasBias;
    string hasOffset;
    string hasLogits;
    string hasRowIndex;
    string expectRet;
    string checkRet;
    string hasAntiquantScale;
    string hasAntiquantOffset;
};

vector<GroupedMatmulFinalizeRoutingWeightNzV2Case> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulFinalizeRoutingWeightNzV2Case> cases;
    string line;
    size_t lineNo = 0U;
    while (getline(in, line)) {
        ++lineNo;
        const string trimmedLine = Trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') {
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(trimmedLine, ",", cols);
        if (cols.empty() || cols[0] == "socVersion") {
            continue;
        }
        if (cols.size() != kWeightNzV2CsvColumnCount) {
            ADD_FAILURE() << "Bad csv row column count in " << csvFilePath << ": " << trimmedLine;
            continue;
        }
        const string caseName = cols.size() > 1U ? Trim(cols[1]) : "";
        try {
            GroupedMatmulFinalizeRoutingWeightNzV2Case c;
            size_t i = 0;
            c.socVersion = Trim(cols[i++]);
            c.caseName = Trim(cols[i++]);
            c.x1Shape = Trim(cols[i++]);
            c.x1Dtype = Trim(cols[i++]);
            c.x1UseRange = Trim(cols[i++]);
            c.x2Shape = Trim(cols[i++]);
            c.x2Dtype = Trim(cols[i++]);
            c.x2Format = Trim(cols[i++]);
            c.x2StorageShape = Trim(cols[i++]);
            c.x2UseRange = Trim(cols[i++]);
            c.scaleShape = Trim(cols[i++]);
            c.scaleDtype = Trim(cols[i++]);
            c.scaleUseRange = Trim(cols[i++]);
            c.biasShape = Trim(cols[i++]);
            c.biasDtype = Trim(cols[i++]);
            c.offsetShape = Trim(cols[i++]);
            c.offsetDtype = Trim(cols[i++]);
            c.perTokenScaleShape = Trim(cols[i++]);
            c.perTokenScaleDtype = Trim(cols[i++]);
            c.perTokenScaleUseRange = Trim(cols[i++]);
            c.groupListShape = Trim(cols[i++]);
            c.groupListValues = Trim(cols[i++]);
            c.groupListUseRange = Trim(cols[i++]);
            c.groupListRangeVal = stoll(Trim(cols[i++]));
            c.sharedInputShape = Trim(cols[i++]);
            c.sharedInputDtype = Trim(cols[i++]);
            c.sharedInputUseRange = Trim(cols[i++]);
            c.logitsShape = Trim(cols[i++]);
            c.logitsDtype = Trim(cols[i++]);
            c.logitsUseRange = Trim(cols[i++]);
            c.rowIndexShape = Trim(cols[i++]);
            c.rowIndexDtype = Trim(cols[i++]);
            c.rowIndexUseRange = Trim(cols[i++]);
            c.outShape = Trim(cols[i++]);
            c.outDtype = Trim(cols[i++]);
            c.outUseRange = Trim(cols[i++]);
            c.dtype = stoll(Trim(cols[i++]));
            c.sharedInputWeight = stof(Trim(cols[i++]));
            c.sharedInputOffset = stoll(Trim(cols[i++]));
            c.transposeX = ParseBool(Trim(cols[i++]));
            c.transposeW = ParseBool(Trim(cols[i++]));
            c.groupListType = stoll(Trim(cols[i++]));
            c.tuningConfig = Trim(cols[i++]);
            c.hasTuningConfig = Trim(cols[i++]);
            c.hasScale = Trim(cols[i++]);
            c.hasBias = Trim(cols[i++]);
            c.hasOffset = Trim(cols[i++]);
            c.hasLogits = Trim(cols[i++]);
            c.hasRowIndex = Trim(cols[i++]);
            c.expectRet = Trim(cols[i++]);
            c.checkRet = Trim(cols[i++]);
            c.hasAntiquantScale = Trim(cols[i++]);
            c.hasAntiquantOffset = Trim(cols[i++]);
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

const vector<GroupedMatmulFinalizeRoutingWeightNzV2Case> &GetWeightNzV2Cases()
{
    static const auto cases =
        LoadCases(ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2.csv",
                                          "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api", __FILE__));
    return cases;
}

string MakeWeightNzV2ParamName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingWeightNzV2Case> &info)
{
    return ops::ut::MakeSafeParamName(info.param.socVersion + "_" + info.param.caseName);
}

class grouped_matmul_finalize_routing_weight_nz_v2_csv_test
    : public testing::TestWithParam<GroupedMatmulFinalizeRoutingWeightNzV2Case> {};

TEST_P(grouped_matmul_finalize_routing_weight_nz_v2_csv_test, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMFR_WEIGHT_NZ_V2_CSV, grouped_matmul_finalize_routing_weight_nz_v2_csv_test,
                         testing::ValuesIn(GetWeightNzV2Cases()), MakeWeightNzV2ParamName);

constexpr size_t kWeightNzV2Mxa8w4CsvColumnCount = 30;

TensorDesc MakeMxa8w4TensorDesc(const vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                                const vector<int64_t> &storageShape = {}, const vector<int64_t> &stride = {})
{
    if (!stride.empty()) {
        return TensorDesc(shape, dtype, format, stride, 0, storageShape);
    }
    if (!storageShape.empty()) {
        return TensorDesc(shape, dtype, format, {}, 0, storageShape);
    }
    return TensorDesc(shape, dtype, format);
}

struct GroupedMatmulFinalizeRoutingWeightNzV2Mxa8w4Case {
    void Run() const
    {
        SetupPlatformForCase(Trim(socVersion));
        op::SetPlatformNpuArch(NpuArch::DAV_3510);

        const vector<int64_t> x1Dims = ParseDims(x1Shape);
        const vector<int64_t> x2Dims = ParseDims(x2Shape);
        const vector<int64_t> x2StorageDims = ParseDims(x2StorageShape);
        const vector<int64_t> x2StrideDims = ParseDims(x2Stride);
        const vector<int64_t> scaleDims = ParseDims(scaleShape);
        const vector<int64_t> biasDims = ParseDims(biasShape);
        const vector<int64_t> perTokenScaleDims = ParseDims(perTokenScaleShape);
        const vector<int64_t> groupListDims = ParseDims(groupListShape);
        const vector<int64_t> sharedInputDims = ParseDims(sharedInputShape);
        const vector<int64_t> logitsDims = ParseDims(logitsShape);
        const vector<int64_t> rowIndexDims = ParseDims(rowIndexShape);
        const vector<int64_t> outDims = ParseDims(outShape);

        TensorDesc x1 = MakeMxa8w4TensorDesc(x1Dims, ParseDtype(x1Dtype), ParseFormat(x1Format));
        TensorDesc x2 = MakeMxa8w4TensorDesc(x2Dims, ParseDtype(x2Dtype), ParseFormat(x2Format), x2StorageDims,
                                             x2StrideDims);
        TensorDesc scale = MakeMxa8w4TensorDesc(scaleDims, ParseDtype(scaleDtype), ParseFormat(scaleFormat));
        TensorDesc groupList = MakeMxa8w4TensorDesc(groupListDims, ACL_INT64, ACL_FORMAT_ND);
        TensorDesc logits = MakeMxa8w4TensorDesc(logitsDims, ACL_FLOAT, ACL_FORMAT_ND);
        TensorDesc rowIndex = MakeMxa8w4TensorDesc(rowIndexDims, ACL_INT64, ACL_FORMAT_ND);
        TensorDesc out = MakeMxa8w4TensorDesc(outDims, ParseDtype(outDtype), ACL_FORMAT_ND);

        std::optional<TensorDesc> bias;
        if (!biasDims.empty()) {
            bias.emplace(MakeMxa8w4TensorDesc(biasDims, ParseDtype(biasDtype), ACL_FORMAT_ND));
        }

        std::optional<TensorDesc> perTokenScale;
        if (!perTokenScaleDims.empty()) {
            perTokenScale.emplace(
                MakeMxa8w4TensorDesc(perTokenScaleDims, ParseDtype(perTokenScaleDtype), ACL_FORMAT_ND));
        }

        std::optional<TensorDesc> sharedInput;
        if (!sharedInputDims.empty()) {
            sharedInput.emplace(
                MakeMxa8w4TensorDesc(sharedInputDims, ParseDtype(sharedInputDtype), ACL_FORMAT_ND));
        }

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACLNN_SUCCESS;
        if (bias.has_value() && perTokenScale.has_value() && sharedInput.has_value()) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, *bias, nullptr, nullptr, nullptr, *perTokenScale, groupList,
                                      *sharedInput, logits, rowIndex, 0, 1.0, sharedInputOffset, false, transposeW,
                                      groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (bias.has_value() && perTokenScale.has_value()) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, *bias, nullptr, nullptr, nullptr, *perTokenScale, groupList,
                                      nullptr, logits, rowIndex, 0, 1.0, sharedInputOffset, false, transposeW,
                                      groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (bias.has_value() && sharedInput.has_value()) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, *bias, nullptr, nullptr, nullptr, nullptr, groupList,
                                      *sharedInput, logits, rowIndex, 0, 1.0, sharedInputOffset, false, transposeW,
                                      groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (bias.has_value()) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, *bias, nullptr, nullptr, nullptr, nullptr, groupList, nullptr,
                                      logits, rowIndex, 0, 1.0, sharedInputOffset, false, transposeW, groupListType,
                                      nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (perTokenScale.has_value() && sharedInput.has_value()) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, nullptr, nullptr, nullptr, nullptr, *perTokenScale, groupList,
                                      *sharedInput, logits, rowIndex, 0, 1.0, sharedInputOffset, false, transposeW,
                                      groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (perTokenScale.has_value()) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, nullptr, nullptr, nullptr, nullptr, *perTokenScale, groupList,
                                      nullptr, logits, rowIndex, 0, 1.0, sharedInputOffset, false, transposeW,
                                      groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else if (sharedInput.has_value()) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, nullptr, nullptr, nullptr, nullptr, nullptr, groupList,
                                      *sharedInput, logits, rowIndex, 0, 1.0, sharedInputOffset, false, transposeW,
                                      groupListType, nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
                                INPUT(x1, x2, scale, nullptr, nullptr, nullptr, nullptr, nullptr, groupList, nullptr,
                                      logits, rowIndex, 0, 1.0, sharedInputOffset, false, transposeW, groupListType,
                                      nullptr),
                                OUTPUT(out));
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }
        EXPECT_EQ(ret, ParseStatus(expectRet));
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string x1Shape;
    string x1Dtype;
    string x1Format;
    string x2Shape;
    string x2Dtype;
    string x2Format;
    string x2StorageShape;
    string x2Stride;
    string scaleShape;
    string scaleDtype;
    string scaleFormat;
    string biasShape;
    string biasDtype;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string groupListShape;
    string sharedInputShape;
    string sharedInputDtype;
    string logitsShape;
    string rowIndexShape;
    string outShape;
    string outDtype;
    int64_t sharedInputOffset = 0;
    bool transposeW = false;
    int64_t groupListType = 1;
    string expectRet;
    string comment;
};

vector<GroupedMatmulFinalizeRoutingWeightNzV2Mxa8w4Case> LoadMxa8w4Cases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulFinalizeRoutingWeightNzV2Mxa8w4Case> cases;
    string line;
    size_t lineNo = 0U;
    while (getline(in, line)) {
        ++lineNo;
        const string trimmedLine = Trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') {
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(trimmedLine, ",", cols);
        if (cols.empty() || cols[0] == "socVersion") {
            continue;
        }
        if (cols.size() != kWeightNzV2Mxa8w4CsvColumnCount) {
            ADD_FAILURE() << "Bad csv row column count in " << csvFilePath << ": " << trimmedLine;
            continue;
        }
        const string caseName = cols.size() > 1U ? Trim(cols[1]) : "";
        try {
            GroupedMatmulFinalizeRoutingWeightNzV2Mxa8w4Case c;
            size_t i = 0;
            c.socVersion = Trim(cols[i++]);
            c.caseName = Trim(cols[i++]);
            c.enable = ParseBool(Trim(cols[i++]));
            if (!c.enable) {
                continue;
            }
            c.x1Shape = Trim(cols[i++]);
            c.x1Dtype = Trim(cols[i++]);
            c.x1Format = Trim(cols[i++]);
            c.x2Shape = Trim(cols[i++]);
            c.x2Dtype = Trim(cols[i++]);
            c.x2Format = Trim(cols[i++]);
            c.x2StorageShape = Trim(cols[i++]);
            c.x2Stride = Trim(cols[i++]);
            c.scaleShape = Trim(cols[i++]);
            c.scaleDtype = Trim(cols[i++]);
            c.scaleFormat = Trim(cols[i++]);
            c.biasShape = Trim(cols[i++]);
            c.biasDtype = Trim(cols[i++]);
            c.perTokenScaleShape = Trim(cols[i++]);
            c.perTokenScaleDtype = Trim(cols[i++]);
            c.groupListShape = Trim(cols[i++]);
            c.sharedInputShape = Trim(cols[i++]);
            c.sharedInputDtype = Trim(cols[i++]);
            c.logitsShape = Trim(cols[i++]);
            c.rowIndexShape = Trim(cols[i++]);
            c.outShape = Trim(cols[i++]);
            c.outDtype = Trim(cols[i++]);
            c.sharedInputOffset = stoll(Trim(cols[i++]));
            c.transposeW = ParseBool(Trim(cols[i++]));
            c.groupListType = stoll(Trim(cols[i++]));
            c.expectRet = Trim(cols[i++]);
            c.comment = Trim(cols[i++]);
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid MxA8W4 cases parsed from CSV: " << csvFilePath;
    return cases;
}

const vector<GroupedMatmulFinalizeRoutingWeightNzV2Mxa8w4Case> &GetMxa8w4Cases()
{
    static const auto cases = LoadMxa8w4Cases(
        ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4.csv",
                                "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api", __FILE__));
    return cases;
}

string MakeMxa8w4ParamName(
    const testing::TestParamInfo<GroupedMatmulFinalizeRoutingWeightNzV2Mxa8w4Case> &info)
{
    return ops::ut::MakeSafeParamName(info.param.socVersion + "_" + info.param.caseName);
}

class grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4_csv_test
    : public testing::TestWithParam<GroupedMatmulFinalizeRoutingWeightNzV2Mxa8w4Case> {};

TEST_P(grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4_csv_test, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMFR_WEIGHT_NZ_V2_MXA8W4_CSV,
                         grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4_csv_test,
                         testing::ValuesIn(GetMxa8w4Cases()), MakeMxa8w4ParamName);
} // namespace
