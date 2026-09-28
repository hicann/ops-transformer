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
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
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
                auto ut = OP_API_UT(
                    aclnnGroupedMatmul,
                    INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupListArray, splitItem),
                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmul,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                                          groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmul,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                                          groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmul,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, groupListArray, splitItem),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V2") {
            if (enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset,
                                          groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                                          groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                                          groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV2,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, groupListArray, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V3") {
            if (enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList,
                                          splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                                          groupList, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                                          groupList, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV3,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, groupList, splitItem, groupType),
                                    OUTPUT(out));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "WeightNz") {
            if (enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(
                    aclnnGroupedMatmulWeightNz,
                    INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupList,
                          activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                          splitItem, groupType, groupListType, actType, tuningConfigOptional, quantGroupSize),
                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset,
                                          perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                                          perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                                          perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                                          perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                                          perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulWeightNz,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional, quantGroupSize),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else if (api == "V4") {
            if (enableBias && enableScale && enablePerToken) {
                auto ut =
                    OP_API_UT(aclnnGroupedMatmulV4,
                              INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale,
                                    groupList, activationInputOptional, activationQuantScaleOptional,
                                    activationQuantOffsetOptional, splitItem, groupType, groupListType, actType),
                              OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(
                    aclnnGroupedMatmulV4,
                    INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScaleOptional,
                          groupList, activationInputOptional, activationQuantScaleOptional,
                          activationQuantOffsetOptional, splitItem, groupType, groupListType, actType),
                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut =
                    OP_API_UT(aclnnGroupedMatmulV4,
                              INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                                    perTokenScale, groupList, activationInputOptional, activationQuantScaleOptional,
                                    activationQuantOffsetOptional, splitItem, groupType, groupListType, actType),
                              OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(
                    aclnnGroupedMatmulV4,
                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                          perTokenScaleOptional, groupList, activationInputOptional, activationQuantScaleOptional,
                          activationQuantOffsetOptional, splitItem, groupType, groupListType, actType),
                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut =
                    OP_API_UT(aclnnGroupedMatmulV4,
                              INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                                    perTokenScale, groupList, activationInputOptional, activationQuantScaleOptional,
                                    activationQuantOffsetOptional, splitItem, groupType, groupListType, actType),
                              OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(
                    aclnnGroupedMatmulV4,
                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                          perTokenScaleOptional, groupList, activationInputOptional, activationQuantScaleOptional,
                          activationQuantOffsetOptional, splitItem, groupType, groupListType, actType),
                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut =
                    OP_API_UT(aclnnGroupedMatmulV4,
                              INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale, antiquantOffset,
                                    perTokenScale, groupList, activationInputOptional, activationQuantScaleOptional,
                                    activationQuantOffsetOptional, splitItem, groupType, groupListType, actType),
                              OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(
                    aclnnGroupedMatmulV4,
                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale, antiquantOffset,
                          perTokenScaleOptional, groupList, activationInputOptional, activationQuantScaleOptional,
                          activationQuantOffsetOptional, splitItem, groupType, groupListType, actType),
                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            }
        } else {
            if (enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(
                    aclnnGroupedMatmulV5,
                    INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupList,
                          activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                          splitItem, groupType, groupListType, actType, tuningConfigOptional),
                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scale, offset, antiquantScale, antiquantOffset,
                                          perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                                          perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (enableBias && !enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, bias, scaleOptional, offset, antiquantScale, antiquantOffset,
                                          perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                                          perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && enableScale && !enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scale, offset, antiquantScale, antiquantOffset,
                                          perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else if (!enableBias && !enableScale && enablePerToken) {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScale, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional),
                                    OUTPUT(out, activationFeatureOutOptional, dynQuantScaleOutOptional));
                ret = ut.TestGetWorkspaceSize(&workspaceSize);
            } else {
                auto ut = OP_API_UT(aclnnGroupedMatmulV5,
                                    INPUT(x, weight, biasOptional, scaleOptional, offset, antiquantScale,
                                          antiquantOffset, perTokenScaleOptional, groupList, activationInputOptional,
                                          activationQuantScaleOptional, activationQuantOffsetOptional, splitItem,
                                          groupType, groupListType, actType, tuningConfigOptional),
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
        if (cols.size() != kGroupedMatmulCsvColumnCount && cols.size() != kGroupedMatmulCsvExtendedColumnCount &&
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
    std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> nullBias(aclCreateTensorList(&nullTensor, 1),
                                                                             aclDestroyTensorList);

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

using OptionalTensorLists = std::array<const aclTensorList *, 6>;
using OwnedTensorList = std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)>;

const std::array<const char *, 6> kOptionalTensorNames = {"bias",           "scale",           "offset",
                                                          "antiquantScale", "antiquantOffset", "perTokenScale"};
const std::array<gmm::GMMApiVersion, 6> kGroupedMatmulPublicApis = {
    gmm::GMMApiVersion::V1, gmm::GMMApiVersion::V2, gmm::GMMApiVersion::V3,
    gmm::GMMApiVersion::V4, gmm::GMMApiVersion::V5, gmm::GMMApiVersion::WeightNz};

OwnedTensorList MakeTensorListWithNullElement(size_t nullIndex, bool emptyFirst = false)
{
    auto first = TensorDesc({emptyFirst ? 0 : 32}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    const aclTensor *tensors[] = {nullptr, nullptr};
    if (nullIndex != 0) {
        tensors[0] = first.get();
    }
    OwnedTensorList list(aclCreateTensorList(tensors, nullIndex + 1), aclDestroyTensorList);
    // aclDestroyTensorList owns the tensors in the list; do not destroy the first tensor twice.
    if (list != nullptr && nullIndex != 0) {
        (void)first.release();
    }
    return list;
}

class GroupedMatmulOptionalTensorListTest : public testing::Test {
protected:
    void SetUp() override
    {
        previousSoc_ = op::GetCurrentPlatformInfo().GetSocVersion();
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    void TearDown() override
    {
        op::SetPlatformSocVersion(previousSoc_);
    }

    aclnnStatus GetWorkspaceSize(gmm::GMMApiVersion api, const OptionalTensorLists &optional)
    {
        // A valid no-split FP16 case with zero M avoids kernel execution for compatibility checks.
        auto x = BuildAclTensorListDesc("0:32", "FLOAT16", "ND").ToAclType();
        auto weight = BuildAclTensorListDesc("32:32", "FLOAT16", "ND").ToAclType();
        auto out = BuildAclTensorListDesc("0:32", "FLOAT16", "ND").ToAclType();
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus status = ACLNN_ERR_PARAM_INVALID;
        switch (api) {
            case gmm::GMMApiVersion::V1:
                status = aclnnGroupedMatmulGetWorkspaceSize(x.get(), weight.get(), optional[0], optional[1],
                                                            optional[2], optional[3], optional[4], nullptr, 0,
                                                            out.get(), &workspaceSize, &executor);
                break;
            case gmm::GMMApiVersion::V2:
                status = aclnnGroupedMatmulV2GetWorkspaceSize(x.get(), weight.get(), optional[0], optional[1],
                                                              optional[2], optional[3], optional[4], nullptr, 0, -1,
                                                              out.get(), &workspaceSize, &executor);
                break;
            case gmm::GMMApiVersion::V3:
                status = aclnnGroupedMatmulV3GetWorkspaceSize(x.get(), weight.get(), optional[0], optional[1],
                                                              optional[2], optional[3], optional[4], nullptr, 0, -1,
                                                              out.get(), &workspaceSize, &executor);
                break;
            case gmm::GMMApiVersion::V4:
                status = aclnnGroupedMatmulV4GetWorkspaceSize(x.get(), weight.get(), optional[0], optional[1],
                                                              optional[2], optional[3], optional[4], optional[5],
                                                              nullptr, nullptr, nullptr, nullptr, 0, -1, 0, 0,
                                                              out.get(), nullptr, nullptr, &workspaceSize, &executor);
                break;
            case gmm::GMMApiVersion::V5:
                status = aclnnGroupedMatmulV5GetWorkspaceSize(x.get(), weight.get(), optional[0], optional[1],
                                                              optional[2], optional[3], optional[4], optional[5],
                                                              nullptr, nullptr, nullptr, nullptr, 0, -1, 0, 0, nullptr,
                                                              out.get(), nullptr, nullptr, &workspaceSize, &executor);
                break;
            case gmm::GMMApiVersion::WeightNz:
                status = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
                    x.get(), weight.get(), optional[0], optional[1], optional[2], optional[3], optional[4], optional[5],
                    nullptr, nullptr, nullptr, nullptr, 0, -1, 0, 0, nullptr, 0, out.get(), nullptr, nullptr,
                    &workspaceSize, &executor);
                break;
            default:
                ADD_FAILURE() << "Unexpected grouped matmul API";
                break;
        }
        if (status == ACLNN_SUCCESS) {
            EXPECT_EQ(workspaceSize, 0U);
        }
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
        return status;
    }

private:
    op::SocVersion previousSoc_ = op::SocVersion::ASCEND910B;
};

TEST_F(GroupedMatmulOptionalTensorListTest, A5RejectsNullElementsAcrossPublicApis)
{
    for (const auto api : kGroupedMatmulPublicApis) {
        const size_t optionalCount =
            (api == gmm::GMMApiVersion::V1 || api == gmm::GMMApiVersion::V2 || api == gmm::GMMApiVersion::V3) ? 5 : 6;
        for (size_t input = 0; input < optionalCount; ++input) {
            for (size_t nullIndex : {0U, 1U}) {
                SCOPED_TRACE(testing::Message() << "API=" << static_cast<int>(api) << ", "
                                                << kOptionalTensorNames[input] << "[" << nullIndex << "]");
                auto list = MakeTensorListWithNullElement(nullIndex);
                ASSERT_NE(list.get(), nullptr);
                OptionalTensorLists optional{};
                optional[input] = list.get();
                EXPECT_EQ(GetWorkspaceSize(api, optional), ACLNN_ERR_PARAM_NULLPTR);
            }
        }
    }
}

TEST_F(GroupedMatmulOptionalTensorListTest, A5RejectsNullAfterEmptyTensor)
{
    for (size_t input = 0; input < kOptionalTensorNames.size(); ++input) {
        SCOPED_TRACE(kOptionalTensorNames[input]);
        auto list = MakeTensorListWithNullElement(1, true);
        ASSERT_NE(list.get(), nullptr);
        OptionalTensorLists optional{};
        optional[input] = list.get();
        EXPECT_EQ(GetWorkspaceSize(gmm::GMMApiVersion::V5, optional), ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(GroupedMatmulOptionalTensorListTest, A5WeightNzRejectsNullOffsetBeforeQuantGroupSizeCheck)
{
    auto x = BuildAclTensorListDesc("64:1024", "INT8", "ND").ToAclType();
    auto weight = BuildAclTensorListDesc("2:1024:256", "INT4", "FRACTAL_NZ").ToAclType();
    auto perTokenScale = BuildAclTensorListDesc("64", "FLOAT", "ND").ToAclType();
    auto groupList = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    auto out = BuildAclTensorListDesc("64:256", "FLOAT16", "ND").ToAclType();
    for (const aclDataType scaleDtype : {ACL_UINT64, ACL_INT64}) {
        auto scale = BuildAclTensorListDesc({{2, 1, 256}}, scaleDtype, ACL_FORMAT_ND).ToAclType();
        for (size_t nullIndex : {0U, 1U}) {
            SCOPED_TRACE(testing::Message()
                         << "scale dtype=" << static_cast<int>(scaleDtype) << ", offset[" << nullIndex << "]");
            auto firstOffset = TensorDesc({2, 1, 256}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
            const aclTensor *offsetTensors[] = {nullIndex == 0 ? nullptr : firstOffset.get(), nullptr};
            OwnedTensorList offset(aclCreateTensorList(offsetTensors, nullIndex + 1), aclDestroyTensorList);
            ASSERT_NE(offset.get(), nullptr);
            if (nullIndex != 0) {
                (void)firstOffset.release();
            }
            uint64_t workspaceSize = 0;
            aclOpExecutor *executor = nullptr;
            // A null element must not turn offset mode into symmetric mode and report invalid quantGroupSize.
            const auto status = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
                x.get(), weight.get(), nullptr, scale.get(), offset.get(), nullptr, nullptr, perTokenScale.get(),
                groupList.get(), nullptr, nullptr, nullptr, 3, 0, 1, 0, nullptr, 0, out.get(), nullptr, nullptr,
                &workspaceSize, &executor);
            EXPECT_EQ(status, ACLNN_ERR_PARAM_NULLPTR);
            if (executor != nullptr) {
                aclDestroyAclOpExecutor(executor);
            }
        }
    }
}

TEST_F(GroupedMatmulOptionalTensorListTest, A5AcceptsOmittedAndEmptyLists)
{
    EXPECT_EQ(GetWorkspaceSize(gmm::GMMApiVersion::V5, {}), ACLNN_SUCCESS);
    const aclTensor *unused = nullptr;
    OwnedTensorList emptyList(aclCreateTensorList(&unused, 0), aclDestroyTensorList);
    auto emptyTensor = BuildAclTensorListDesc("0", "FLOAT16", "ND").ToAclType();
    ASSERT_NE(emptyList.get(), nullptr);
    for (size_t input = 0; input < kOptionalTensorNames.size(); ++input) {
        SCOPED_TRACE(kOptionalTensorNames[input]);
        for (const aclTensorList *list : {emptyList.get(), emptyTensor.get()}) {
            OptionalTensorLists optional{};
            optional[input] = list;
            EXPECT_EQ(GetWorkspaceSize(gmm::GMMApiVersion::V5, optional), ACLNN_SUCCESS);
        }
    }
}

TEST_F(GroupedMatmulOptionalTensorListTest, A5PreservesValidBiasAndInvalidParameterErrors)
{
    auto validBias = BuildAclTensorListDesc("32", "FLOAT16", "ND").ToAclType();
    EXPECT_EQ(GetWorkspaceSize(gmm::GMMApiVersion::V5, {validBias.get()}), ACLNN_SUCCESS);
    auto invalidShape = BuildAclTensorListDesc("31", "FLOAT16", "ND").ToAclType();
    auto invalidDtype = BuildAclTensorListDesc("32", "INT8", "ND").ToAclType();
    auto invalidCount = TensorListDesc(2, TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND)).ToAclType();
    for (const aclTensorList *bias : {invalidShape.get(), invalidDtype.get(), invalidCount.get()}) {
        EXPECT_EQ(GetWorkspaceSize(gmm::GMMApiVersion::V5, {bias}), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(GroupedMatmulOptionalTensorListTest, OtherSocPreservesNullBiasAsAbsent)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    EXPECT_EQ(GetWorkspaceSize(gmm::GMMApiVersion::V5, {}), ACLNN_SUCCESS);
    auto nullBias = MakeTensorListWithNullElement(0);
    ASSERT_NE(nullBias.get(), nullptr);
    EXPECT_EQ(GetWorkspaceSize(gmm::GMMApiVersion::V5, {nullBias.get()}), ACLNN_SUCCESS);
}

TEST_P(grouped_matmul_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(grouped_matmul_opapi_csv, grouped_matmul_opapi_csv_test,
                         testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath(
                             "test_aclnn_grouped_matmul.csv", "gmm/grouped_matmul/tests/ut/op_api", __FILE__))),
                         BuildCaseName);

} // namespace
