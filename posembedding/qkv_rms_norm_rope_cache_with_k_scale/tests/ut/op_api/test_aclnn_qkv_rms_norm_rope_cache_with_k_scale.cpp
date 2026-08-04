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
 * \file test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp
 * \brief CSV driven QkvRmsNormRopeCacheWithKScale aclnn op_api UT.
 */

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

namespace {
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kCsvColumnCount = 38;

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

aclnnStatus ParseStatus(const string &status)
{
    static const map<string, aclnnStatus> statusMap = {
        {"SUCCESS", ACLNN_SUCCESS},
        {"ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
    };
    const auto it = statusMap.find(Trim(status));
    return it == statusMap.end() ? ACLNN_SUCCESS : it->second;
}

bool IsNullArg(const string &value)
{
    const string trimmed = Trim(value);
    return trimmed.empty() || trimmed == "<null>" || trimmed == "NONE";
}

TensorDesc MakeTensorDesc(const vector<int64_t> &shape, aclDataType dtype, bool useRange = true)
{
    auto desc = TensorDesc(shape, dtype, ACL_FORMAT_ND);
    if (useRange) {
        desc.ValueRange(-1, 1);
    }
    return desc;
}

aclTensor *BuildTensor(const string &shape, const string &dtype, bool useRange = true)
{
    if (IsNullArg(shape)) {
        return nullptr;
    }
    return MakeTensorDesc(ParseDims(shape), ParseDtype(dtype), useRange).ToAclTypeRawPtr();
}

aclIntArray *BuildIntArray(const string &value)
{
    if (IsNullArg(value)) {
        return nullptr;
    }
    const string trimmed = Trim(value);
    const vector<int64_t> values = trimmed == "<empty>" ? vector<int64_t>{} : ParseI64List(trimmed);
    return IntArrayDesc(values).ToAclTypeRawPtr();
}

const char *BuildStringArg(const string &value)
{
    if (IsNullArg(value)) {
        return nullptr;
    }
    return value == "<empty>" ? "" : value.c_str();
}

struct QkvRmsNormRopeCacheWithKScaleCase {
    string caseName;
    string qkvShape;
    string qkvDtype;
    string qGammaShape;
    string qGammaDtype;
    string kGammaShape;
    string kGammaDtype;
    string cosSinShape;
    string cosSinDtype;
    string slotMappingShape;
    string slotMappingDtype;
    string kCacheShape;
    string kCacheDtype;
    string vCacheShape;
    string vCacheDtype;
    string kScaleCacheShape;
    string kScaleCacheDtype;
    string queryStartLocShape;
    string queryStartLocDtype;
    string seqLensShape;
    string seqLensDtype;
    string rotationOptionalShape;
    string rotationOptionalDtype;
    string vScaleOptionalShape;
    string vScaleOptionalDtype;
    string headNumsValue;
    string layoutQkv;
    string layoutQOut;
    float epsilon = 1e-6f;
    string qOutShape;
    string qOutDtype;
    string qScaleShape;
    string qScaleDtype;
    string mropePositionShape;
    string mropePositionDtype;
    string mropeSectionValue;
    string qQuantMode;
    string expectRet;
};

struct QkvRmsNormRopeCacheWithKScaleAclArgs {
    aclTensor *qkv = nullptr;
    aclTensor *qGamma = nullptr;
    aclTensor *kGamma = nullptr;
    aclTensor *cosSin = nullptr;
    aclTensor *slotMapping = nullptr;
    aclTensor *kCache = nullptr;
    aclTensor *vCache = nullptr;
    aclTensor *kScaleCache = nullptr;
    aclTensor *queryStartLoc = nullptr;
    aclTensor *seqLens = nullptr;
    aclTensor *rotationOptional = nullptr;
    aclTensor *vScaleOptional = nullptr;
    aclTensor *mropePosition = nullptr;
    aclIntArray *headNums = nullptr;
    const char *layoutQkv = nullptr;
    const char *layoutQOut = nullptr;
    aclIntArray *mropeSection = nullptr;
    const char *qQuantMode = nullptr;
    aclTensor *qOut = nullptr;
    aclTensor *qScale = nullptr;
};

QkvRmsNormRopeCacheWithKScaleAclArgs BuildAclArgs(const QkvRmsNormRopeCacheWithKScaleCase &testCase)
{
    QkvRmsNormRopeCacheWithKScaleAclArgs args;
    args.qkv = BuildTensor(testCase.qkvShape, testCase.qkvDtype);
    args.qGamma = BuildTensor(testCase.qGammaShape, testCase.qGammaDtype);
    args.kGamma = BuildTensor(testCase.kGammaShape, testCase.kGammaDtype);
    args.cosSin = BuildTensor(testCase.cosSinShape, testCase.cosSinDtype);
    args.slotMapping = BuildTensor(testCase.slotMappingShape, testCase.slotMappingDtype);
    args.kCache = BuildTensor(testCase.kCacheShape, testCase.kCacheDtype);
    args.vCache = BuildTensor(testCase.vCacheShape, testCase.vCacheDtype);
    args.kScaleCache = BuildTensor(testCase.kScaleCacheShape, testCase.kScaleCacheDtype);
    args.queryStartLoc = BuildTensor(testCase.queryStartLocShape, testCase.queryStartLocDtype);
    args.seqLens = BuildTensor(testCase.seqLensShape, testCase.seqLensDtype);
    args.rotationOptional = BuildTensor(testCase.rotationOptionalShape, testCase.rotationOptionalDtype);
    args.vScaleOptional = BuildTensor(testCase.vScaleOptionalShape, testCase.vScaleOptionalDtype);
    args.mropePosition = BuildTensor(testCase.mropePositionShape, testCase.mropePositionDtype);
    args.headNums = BuildIntArray(testCase.headNumsValue);
    args.layoutQkv = BuildStringArg(testCase.layoutQkv);
    args.layoutQOut = BuildStringArg(testCase.layoutQOut);
    args.mropeSection = BuildIntArray(testCase.mropeSectionValue);
    args.qQuantMode = BuildStringArg(testCase.qQuantMode);
    args.qOut = BuildTensor(testCase.qOutShape, testCase.qOutDtype, false);
    args.qScale = BuildTensor(testCase.qScaleShape, testCase.qScaleDtype, false);
    return args;
}

void RunCase(const QkvRmsNormRopeCacheWithKScaleCase &testCase)
{
    const auto args = BuildAclArgs(testCase);
    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCacheWithKScale,
        INPUT(args.qkv, args.qGamma, args.kGamma, args.cosSin, args.slotMapping, args.kCache, args.vCache,
              args.kScaleCache, args.queryStartLoc, args.seqLens, args.rotationOptional, args.vScaleOptional,
              args.mropePosition, args.headNums, args.layoutQkv, args.layoutQOut, testCase.epsilon, args.mropeSection,
              args.qQuantMode),
        OUTPUT(args.qOut, args.qScale));
    const aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ParseStatus(testCase.expectRet)) << "caseName=" << testCase.caseName;
}

vector<QkvRmsNormRopeCacheWithKScaleCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;

    vector<QkvRmsNormRopeCacheWithKScaleCase> cases;
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
        if (cols.empty() || cols[0] == "caseName") {
            continue;
        }
        if (cols.size() != kCsvColumnCount) {
            ADD_FAILURE() << "Bad csv row column count in " << csvFilePath << ": " << trimmedLine;
            continue;
        }

        const string caseName = Trim(cols[0]);
        try {
            QkvRmsNormRopeCacheWithKScaleCase c;
            size_t i = 0;
            c.caseName = Trim(cols[i++]);
            c.qkvShape = Trim(cols[i++]);
            c.qkvDtype = Trim(cols[i++]);
            c.qGammaShape = Trim(cols[i++]);
            c.qGammaDtype = Trim(cols[i++]);
            c.kGammaShape = Trim(cols[i++]);
            c.kGammaDtype = Trim(cols[i++]);
            c.cosSinShape = Trim(cols[i++]);
            c.cosSinDtype = Trim(cols[i++]);
            c.slotMappingShape = Trim(cols[i++]);
            c.slotMappingDtype = Trim(cols[i++]);
            c.kCacheShape = Trim(cols[i++]);
            c.kCacheDtype = Trim(cols[i++]);
            c.vCacheShape = Trim(cols[i++]);
            c.vCacheDtype = Trim(cols[i++]);
            c.kScaleCacheShape = Trim(cols[i++]);
            c.kScaleCacheDtype = Trim(cols[i++]);
            c.queryStartLocShape = Trim(cols[i++]);
            c.queryStartLocDtype = Trim(cols[i++]);
            c.seqLensShape = Trim(cols[i++]);
            c.seqLensDtype = Trim(cols[i++]);
            c.rotationOptionalShape = Trim(cols[i++]);
            c.rotationOptionalDtype = Trim(cols[i++]);
            c.vScaleOptionalShape = Trim(cols[i++]);
            c.vScaleOptionalDtype = Trim(cols[i++]);
            c.headNumsValue = Trim(cols[i++]);
            c.layoutQkv = Trim(cols[i++]);
            c.layoutQOut = Trim(cols[i++]);
            c.epsilon = stof(Trim(cols[i++]));
            c.qOutShape = Trim(cols[i++]);
            c.qOutDtype = Trim(cols[i++]);
            c.qScaleShape = Trim(cols[i++]);
            c.qScaleDtype = Trim(cols[i++]);
            c.mropePositionShape = Trim(cols[i++]);
            c.mropePositionDtype = Trim(cols[i++]);
            c.mropeSectionValue = Trim(cols[i++]);
            c.qQuantMode = Trim(cols[i++]);
            c.expectRet = Trim(cols[i++]);
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

const vector<QkvRmsNormRopeCacheWithKScaleCase> &GetCases()
{
    static const auto cases =
        LoadCases(ops::ut::ResolveCsvPath("test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.csv",
                                          "attention/qkv_rms_norm_rope_cache_with_k_scale/tests/ut/op_api", __FILE__));
    return cases;
}

string MakeParamName(const testing::TestParamInfo<QkvRmsNormRopeCacheWithKScaleCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class qkv_rms_norm_rope_cache_with_k_scale_csv_test : public testing::TestWithParam<QkvRmsNormRopeCacheWithKScaleCase> {
};

TEST_P(qkv_rms_norm_rope_cache_with_k_scale_csv_test, csvDrivenCase)
{
    RunCase(GetParam());
}

INSTANTIATE_TEST_SUITE_P(QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CSV, qkv_rms_norm_rope_cache_with_k_scale_csv_test,
                         testing::ValuesIn(GetCases()), MakeParamName);

} // namespace
