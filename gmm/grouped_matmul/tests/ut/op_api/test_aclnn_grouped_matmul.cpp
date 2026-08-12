/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_matmul.h"
#include "../../../op_api/aclnn_grouped_matmul_v2.h"
#include "../../../op_api/aclnn_grouped_matmul_v3.h"
#include "../../../op_api/aclnn_grouped_matmul_v4.h"
#include "../../../op_api/aclnn_grouped_matmul_v5.h"
#include "../../../op_api/aclnn_grouped_matmul_weight_nz.h"
#include "../../../op_api/grouped_matmul_weight_quant_950_checker.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "gmm_csv_acl_parse_utils.h"

using namespace std;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;
using ops::ut::BuildAclTensorListDesc;

constexpr size_t kGroupedMatmulCsvColumnCount = 31;
constexpr size_t kGroupedMatmulCsvExtendedColumnCount = 35;
constexpr size_t kGroupedMatmulCsvOptionalTensorColumnCount = 44;

vector<int64_t> ParseDims(const string &value)
{
    return ops::ut::ParseAclTensorViewDims(value);
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
    return ops::ut::ParseAclFormat(format);
}

void SetupPlatformForCase(const string &socVersion)
{
    static const map<string, op::SocVersion> socMap = {
        {"Ascend310P", op::SocVersion::ASCEND310P},
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    auto it = socMap.find(socVersion);
    op::SetPlatformSocVersion(it == socMap.end() ? op::SocVersion::ASCEND910B : it->second);
}

struct GroupedMatmulOpApiCase {
    void Run() const
    {
        SetupPlatformForCase(socVersion);
        TensorListDesc x = BuildAclTensorListDesc(xShape, xDtype, xFormat);
        TensorListDesc weight = BuildAclTensorListDesc(weightShape, weightDtype, weightFormat);
        TensorListDesc out = BuildAclTensorListDesc(outShape, outDtype, outFormat, false);
        TensorDesc groupList(ParseDims(groupListShape), ParseDtype(groupListDtype), ParseFormat(groupListFormat));
        vector<int64_t> groupListVal = ParseI64List(groupListValues);
        IntArrayDesc groupListArray(groupListVal);
        if (!groupListVal.empty()) {
            groupList.Value(groupListVal);
        }
        TensorListDesc scale = BuildAclTensorListDesc(scaleShape, scaleDtype, scaleFormat);
        TensorListDesc bias = BuildAclTensorListDesc(biasShape, biasDtype, biasFormat);
        TensorListDesc perTokenScale =
            BuildAclTensorListDesc(perTokenScaleShape, perTokenScaleDtype, perTokenScaleFormat);
        TensorListDesc offset = BuildAclTensorListDesc(offsetShape, offsetDtype, offsetFormat);
        TensorListDesc antiquantScale =
            BuildAclTensorListDesc(antiquantScaleShape, antiquantScaleDtype, antiquantScaleFormat);
        TensorListDesc antiquantOffset =
            BuildAclTensorListDesc(antiquantOffsetShape, antiquantOffsetDtype, antiquantOffsetFormat);
        auto activationInputOptional = nullptr;
        auto activationQuantScaleOptional = nullptr;
        auto activationQuantOffsetOptional = nullptr;
        auto tuningConfigOptional = nullptr;
        auto activationFeatureOutOptional = nullptr;
        auto dynQuantScaleOutOptional = nullptr;
        auto biasOptional = nullptr;
        auto scaleOptional = nullptr;
        auto perTokenScaleOptional = nullptr;

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACL_SUCCESS;
        const bool enableBias = ParseBool(hasBias);
        const bool enableScale = ParseBool(hasScale);
        const bool enablePerToken = ParseBool(hasPerTokenScale);

        if (api == "V1") {
            if (enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmul, INPUT(x, weight, bias, scale, offset, antiquantScale,
                                                               antiquantOffset, groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmul, INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                                               antiquantOffset, groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmul, INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                                               antiquantOffset, groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmul, INPUT(x, weight, biasOptional, scaleOptional, offset,
                                                               antiquantScale, antiquantOffset, groupListArray,
                                                               splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V2") {
            if (enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2, INPUT(x, weight, bias, scale, offset, antiquantScale,
                                                                 antiquantOffset, groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2, INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                                                 antiquantOffset, groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2, INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                                                 antiquantOffset, groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2, INPUT(x, weight, biasOptional, scaleOptional, offset,
                                                                 antiquantScale, antiquantOffset, groupListArray,
                                                                 splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V3") {
            if (enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3, INPUT(x, weight, bias, scale, offset,
                                                                antiquantScale, antiquantOffset, groupList,
                                                                splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3, INPUT(x, weight, bias, scaleOptional, offset,
                                                                antiquantScale, antiquantOffset, groupList,
                                                                splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3, INPUT(x, weight, biasOptional, scale, offset,
                                                                antiquantScale, antiquantOffset, groupList,
                                                                splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3, INPUT(x, weight, biasOptional, scaleOptional, offset,
                                                                antiquantScale, antiquantOffset, groupList,
                                                                splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "WeightNz") {
            if (enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V4") {
            if (enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV4,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else {
            if (enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem, groupType,
                                          groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        }
        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, static_cast<aclnnStatus>(expectRet));
        }
    }

    string socVersion;
    string caseName;
    string xShape;
    string xDtype;
    string xFormat;
    string weightShape;
    string weightDtype;
    string weightFormat;
    string scaleShape;
    string scaleDtype;
    string scaleFormat;
    string biasShape;
    string biasDtype;
    string biasFormat;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string perTokenScaleFormat;
    string groupListShape;
    string groupListValues;
    string groupListDtype;
    string groupListFormat;
    string outShape;
    string outDtype;
    string outFormat;
    int64_t splitItem;
    int64_t groupType;
    int64_t groupListType;
    int64_t actType;
    uint64_t expectRet;
    string hasBias;
    string hasPerTokenScale;
    string api = "V5";
    string checkRet = "true";
    string hasScale = "true";
    int64_t quantGroupSize = 0;
    string offsetShape = "0";
    string offsetDtype = "FLOAT";
    string offsetFormat = "ND";
    string antiquantScaleShape = "0";
    string antiquantScaleDtype = "FLOAT16";
    string antiquantScaleFormat = "ND";
    string antiquantOffsetShape = "0";
    string antiquantOffsetDtype = "FLOAT16";
    string antiquantOffsetFormat = "ND";
};

vector<GroupedMatmulOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulOpApiCase> cases;
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
        if (cols.size() != kGroupedMatmulCsvColumnCount &&
            cols.size() != kGroupedMatmulCsvExtendedColumnCount &&
            cols.size() != kGroupedMatmulCsvOptionalTensorColumnCount) {
            continue;
        }
        const string caseName = cols.size() > 1U ? Trim(cols[1]) : "";
        try {
            GroupedMatmulOpApiCase c;
            size_t i = 0;
            c.socVersion = Trim(cols[i++]);
            c.caseName = Trim(cols[i++]);
            c.xShape = Trim(cols[i++]);
            c.xDtype = Trim(cols[i++]);
            c.xFormat = Trim(cols[i++]);
            c.weightShape = Trim(cols[i++]);
            c.weightDtype = Trim(cols[i++]);
            c.weightFormat = Trim(cols[i++]);
            c.scaleShape = Trim(cols[i++]);
            c.scaleDtype = Trim(cols[i++]);
            c.scaleFormat = Trim(cols[i++]);
            c.biasShape = Trim(cols[i++]);
            c.biasDtype = Trim(cols[i++]);
            c.biasFormat = Trim(cols[i++]);
            c.perTokenScaleShape = Trim(cols[i++]);
            c.perTokenScaleDtype = Trim(cols[i++]);
            c.perTokenScaleFormat = Trim(cols[i++]);
            c.groupListShape = Trim(cols[i++]);
            c.groupListValues = Trim(cols[i++]);
            c.groupListDtype = Trim(cols[i++]);
            c.groupListFormat = Trim(cols[i++]);
            c.outShape = Trim(cols[i++]);
            c.outDtype = Trim(cols[i++]);
            c.outFormat = Trim(cols[i++]);
            c.splitItem = stoll(Trim(cols[i++]));
            c.groupType = stoll(Trim(cols[i++]));
            c.groupListType = stoll(Trim(cols[i++]));
            c.actType = stoll(Trim(cols[i++]));
            c.expectRet = static_cast<uint64_t>(stoull(Trim(cols[i++])));
            c.hasBias = Trim(cols[i++]);
            c.hasPerTokenScale = Trim(cols[i++]);
            if (i < cols.size()) {
                c.api = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.checkRet = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.hasScale = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.quantGroupSize = stoll(Trim(cols[i++]));
            }
            if (i < cols.size()) {
                c.offsetShape = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.offsetDtype = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.offsetFormat = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.antiquantScaleShape = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.antiquantScaleDtype = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.antiquantScaleFormat = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.antiquantOffsetShape = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.antiquantOffsetDtype = Trim(cols[i++]);
            }
            if (i < cols.size()) {
                c.antiquantOffsetFormat = Trim(cols[i++]);
            }
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<GroupedMatmulOpApiCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class grouped_matmul_opapi_csv_test : public testing::TestWithParam<GroupedMatmulOpApiCase> {};

aclnnStatus CheckS8S4Inputs(int64_t groupListType, bool hasOffset, bool nullBiasElement = false,
                            aclFormat weightFormat = ACL_FORMAT_ND, int64_t scaleGroupNum = 4)
{
    auto x = BuildAclTensorListDesc("64:1024", "INT8", "ND").ToAclType();
    auto weight = BuildAclTensorListDesc({{2, 1024, 256}}, ACL_INT4, weightFormat).ToAclType();
    auto bias = BuildAclTensorListDesc("2:256", "FLOAT", "ND").ToAclType();
    const std::string scaleShape = hasOffset ? "2:1:256" : "2:" + std::to_string(scaleGroupNum) + ":256";
    auto scale = BuildAclTensorListDesc(scaleShape, "UINT64", "ND").ToAclType();
    auto offset = BuildAclTensorListDesc("2:1:256", "FLOAT", "ND").ToAclType();
    auto perTokenScale = BuildAclTensorListDesc("64", "FLOAT", "ND").ToAclType();
    auto out = BuildAclTensorListDesc("64:256", hasOffset ? "FLOAT16" : "BF16", "ND").ToAclType();
    auto groupList = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    const aclTensor *nullTensor = nullptr;
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> nullBias(
        aclCreateTensorList(&nullTensor, 1), aclDestroyTensorList);

    gmm::GroupedMatmulParams params;
    params.x = x.get();
    params.weight = weight.get();
    params.biasOptional = nullBiasElement ? nullBias.get() : bias.get();
    params.groupTensorOptional = groupList.get();
    params.scaleOptional = scale.get();
    params.offsetOptional = hasOffset ? offset.get() : nullptr;
    params.perTokenScaleOptional = perTokenScale.get();
    params.splitItem = 3;
    params.groupListType = groupListType;
    params.activeType = 0;
    params.apiVersion = gmm::GMMApiVersion::V5;
    params.groupType = 0;
    params.y = out.get();
    params.xDtype = op::DataType::DT_INT8;
    return gmm::AclnnGroupedMatmulWeightQuantDAV3510Checker(params).CheckGroupedMatmulWeightQuantDAV3510();
}

TEST(GroupedMatmulS8S4OpApiChecker, OnlyAcceptsCountGroupListType)
{
    for (bool hasOffset : {false, true}) {
        EXPECT_EQ(CheckS8S4Inputs(1, hasOffset), ACLNN_SUCCESS);
        for (int64_t groupListType : {0L, 2L, 99L}) {
            EXPECT_EQ(CheckS8S4Inputs(groupListType, hasOffset), ACLNN_ERR_PARAM_INVALID)
                << "hasOffset=" << hasOffset << ", groupListType=" << groupListType;
        }
    }
}

TEST(GroupedMatmulS8S4OpApiChecker, RejectsNullBiasElementBeforeReadingDtype)
{
    EXPECT_EQ(CheckS8S4Inputs(1, false, true), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedMatmulS8S4OpApiChecker, OnlyAcceptsPerGroupSize256)
{
    EXPECT_EQ(CheckS8S4Inputs(1, false, false, ACL_FORMAT_ND, 4), ACLNN_SUCCESS);
    EXPECT_EQ(CheckS8S4Inputs(1, false, false, ACL_FORMAT_ND, 8), ACLNN_ERR_PARAM_INVALID);
}

TEST(GroupedMatmulS8S4OpApiChecker, UsesA3WeightFormatWhitelist)
{
    for (const aclFormat weightFormat : {ACL_FORMAT_ND, ACL_FORMAT_NCL, ACL_FORMAT_NCHW}) {
        EXPECT_EQ(CheckS8S4Inputs(1, false, false, weightFormat), ACLNN_SUCCESS)
            << "weightFormat=" << static_cast<int32_t>(weightFormat);
    }
    EXPECT_EQ(CheckS8S4Inputs(1, false, false, ACL_FORMAT_NHWC), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(grouped_matmul_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(
    grouped_matmul_opapi_csv,
    grouped_matmul_opapi_csv_test,
    testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul.csv",
                                                        "gmm/grouped_matmul/tests/ut/op_api", __FILE__))),
    BuildCaseName);

}  // namespace
