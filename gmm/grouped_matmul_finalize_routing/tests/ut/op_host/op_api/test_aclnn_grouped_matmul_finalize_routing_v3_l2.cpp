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

#include "../../../../op_api/aclnn_grouped_matmul_finalize_routing_v3.h"
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

constexpr size_t kV3CsvColumnCount = 42;

vector<int64_t> ParseDims(const string& value)
{
    return ops::ut::ParseDims(value);
}

vector<int64_t> ParseI64List(const string& value)
{
    return ops::ut::ParseI64List(value);
}

aclDataType ParseDtype(const string& dtype)
{
    return ops::ut::ParseAclDtype(dtype);
}

aclFormat ParseFormat(const string& format)
{
    return ops::ut::ParseAclFormat(format);
}

aclnnStatus ParseStatus(const string& status)
{
    static const map<string, aclnnStatus> statusMap = {
        {"SUCCESS", ACLNN_SUCCESS},
        {"ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
    };
    auto it = statusMap.find(Trim(status));
    return it == statusMap.end() ? ACLNN_SUCCESS : it->second;
}

void SetupPlatformForCase(const string& socVersion)
{
    static const map<string, SocVersion> socMap = {
        {"Ascend910B", SocVersion::ASCEND910B},
        {"Ascend950", SocVersion::ASCEND950},
    };
    auto it = socMap.find(Trim(socVersion));
    SetPlatformSocVersion(it == socMap.end() ? SocVersion::ASCEND910B : it->second);
}

TensorDesc MakeTensorDesc(const vector<int64_t>& shape, aclDataType dtype, aclFormat format,
                          const vector<int64_t>& storageShape)
{
    return ops::ut::MakeAclTensorDesc(shape, dtype, format, storageShape).ValueRange(-1, 1);
}

struct GroupedMatmulFinalizeRoutingV3RunParam {
    TensorDesc& x1;
    TensorDesc& x2;
    TensorDesc& scale;
    TensorDesc& offset;
    TensorDesc& antiQuantScale;
    TensorDesc& perTokenScale;
    TensorDesc& groupList;
    TensorDesc& sharedInput;
    TensorDesc& logits;
    TensorDesc& rowIndex;
    TensorDesc& out;
    aclIntArray* tuningConfig = nullptr;
    int64_t dtype = 0;
    float sharedInputWeight = 1.0F;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = false;
    int64_t groupListType = 1;
    bool hasOffset = false;
    bool hasAntiQuantScale = false;
    bool hasLogits = false;
    bool hasSharedInputTensor = false;
};

template <typename BiasInput>
aclnnStatus GetGroupedMatmulFinalizeRoutingV3Workspace(const GroupedMatmulFinalizeRoutingV3RunParam& param,
                                                       const BiasInput& biasInput, uint64_t* workspaceSize)
{
    if (!param.hasOffset) {
        if (param.hasSharedInputTensor) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(param.x1, param.x2, param.scale, biasInput, nullptr, nullptr, nullptr,
                                      param.perTokenScale, param.groupList, param.sharedInput, param.logits,
                                      param.rowIndex, param.dtype, param.sharedInputWeight, param.sharedInputOffset,
                                      param.transposeX, param.transposeW, param.groupListType, nullptr),
                                OUTPUT(param.out));
            return ut.TestGetWorkspaceSize(workspaceSize);
        }

        auto ut = OP_API_UT(
            aclnnGroupedMatmulFinalizeRoutingV3,
            INPUT(param.x1, param.x2, param.scale, biasInput, nullptr, nullptr, nullptr, param.perTokenScale,
                  param.groupList, nullptr, param.logits, param.rowIndex, param.dtype, param.sharedInputWeight,
                  param.sharedInputOffset, param.transposeX, param.transposeW, param.groupListType, nullptr),
            OUTPUT(param.out));
        return ut.TestGetWorkspaceSize(workspaceSize);
    }

    if (!param.hasLogits) {
        if (param.hasSharedInputTensor) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(param.x1, param.x2, param.scale, biasInput, param.offset, nullptr, nullptr,
                                      param.perTokenScale, param.groupList, param.sharedInput, nullptr, param.rowIndex,
                                      param.dtype, param.sharedInputWeight, param.sharedInputOffset, param.transposeX,
                                      param.transposeW, param.groupListType, param.tuningConfig),
                                OUTPUT(param.out));
            return ut.TestGetWorkspaceSize(workspaceSize);
        }

        auto ut = OP_API_UT(
            aclnnGroupedMatmulFinalizeRoutingV3,
            INPUT(param.x1, param.x2, param.scale, biasInput, param.offset, nullptr, nullptr, param.perTokenScale,
                  param.groupList, nullptr, nullptr, param.rowIndex, param.dtype, param.sharedInputWeight,
                  param.sharedInputOffset, param.transposeX, param.transposeW, param.groupListType, param.tuningConfig),
            OUTPUT(param.out));
        return ut.TestGetWorkspaceSize(workspaceSize);
    }

    if (param.hasAntiQuantScale) {
        if (param.hasSharedInputTensor) {
            auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                                INPUT(param.x1, param.x2, param.scale, biasInput, param.offset, param.antiQuantScale,
                                      nullptr, param.perTokenScale, param.groupList, param.sharedInput, param.logits,
                                      param.rowIndex, param.dtype, param.sharedInputWeight, param.sharedInputOffset,
                                      param.transposeX, param.transposeW, param.groupListType, param.tuningConfig),
                                OUTPUT(param.out));
            return ut.TestGetWorkspaceSize(workspaceSize);
        }

        auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                            INPUT(param.x1, param.x2, param.scale, biasInput, param.offset, param.antiQuantScale,
                                  nullptr, param.perTokenScale, param.groupList, nullptr, param.logits, param.rowIndex,
                                  param.dtype, param.sharedInputWeight, param.sharedInputOffset, param.transposeX,
                                  param.transposeW, param.groupListType, param.tuningConfig),
                            OUTPUT(param.out));
        return ut.TestGetWorkspaceSize(workspaceSize);
    }

    if (param.hasSharedInputTensor) {
        auto ut = OP_API_UT(aclnnGroupedMatmulFinalizeRoutingV3,
                            INPUT(param.x1, param.x2, param.scale, biasInput, param.offset, nullptr, nullptr,
                                  param.perTokenScale, param.groupList, param.sharedInput, param.logits, param.rowIndex,
                                  param.dtype, param.sharedInputWeight, param.sharedInputOffset, param.transposeX,
                                  param.transposeW, param.groupListType, param.tuningConfig),
                            OUTPUT(param.out));
        return ut.TestGetWorkspaceSize(workspaceSize);
    }

    auto ut = OP_API_UT(
        aclnnGroupedMatmulFinalizeRoutingV3,
        INPUT(param.x1, param.x2, param.scale, biasInput, param.offset, nullptr, nullptr, param.perTokenScale,
              param.groupList, nullptr, param.logits, param.rowIndex, param.dtype, param.sharedInputWeight,
              param.sharedInputOffset, param.transposeX, param.transposeW, param.groupListType, param.tuningConfig),
        OUTPUT(param.out));
    return ut.TestGetWorkspaceSize(workspaceSize);
}

struct GroupedMatmulFinalizeRoutingV3Case {
    void Run() const
    {
        SetupPlatformForCase(socVersion);

        const vector<int64_t> x1Dims = ParseDims(x1Shape);
        const vector<int64_t> x2Dims = ParseDims(x2Shape);
        const vector<int64_t> x2StorageDims = ParseDims(x2StorageShape);
        const vector<int64_t> scaleDims = ParseDims(scaleShape);
        const vector<int64_t> biasDims = ParseDims(biasShape);
        const vector<int64_t> offsetDims = ParseDims(offsetShape);
        const vector<int64_t> antiQuantScaleDims = ParseDims(antiQuantScaleShape);
        const vector<int64_t> perTokenScaleDims = ParseDims(perTokenScaleShape);
        const vector<int64_t> groupListDims = ParseDims(groupListShape);
        const vector<int64_t> sharedInputDims = ParseDims(sharedInputShape);
        const vector<int64_t> logitsDims = ParseDims(logitsShape);
        const vector<int64_t> rowIndexDims = ParseDims(rowIndexShape);
        const vector<int64_t> outDims = ParseDims(outShape);

        TensorDesc x1 = MakeTensorDesc(x1Dims, ParseDtype(x1Dtype), ACL_FORMAT_ND, {});
        TensorDesc x2 = MakeTensorDesc(x2Dims, ParseDtype(x2Dtype), ParseFormat(x2Format), x2StorageDims);
        TensorDesc scale = MakeTensorDesc(scaleDims, ParseDtype(scaleDtype), ACL_FORMAT_ND, {}).ValueRange(0, 3);
        TensorDesc offset = MakeTensorDesc(offsetDims, ParseDtype(offsetDtype), ParseFormat(offsetFormat), {});
        TensorDesc antiQuantScale =
            MakeTensorDesc(antiQuantScaleDims, ParseDtype(antiQuantScaleDtype), ACL_FORMAT_ND, {});
        TensorDesc perTokenScale =
            MakeTensorDesc(perTokenScaleDims, ParseDtype(perTokenScaleDtype), ACL_FORMAT_ND, {}).ValueRange(0, 3);
        TensorDesc groupList = MakeTensorDesc(groupListDims, ACL_INT64, ACL_FORMAT_ND, {}).ValueRange(-1, 1);
        const vector<int64_t> groupListVals = ParseI64List(groupListValues);
        if (!groupListVals.empty()) {
            groupList.Value(groupListVals);
        }
        TensorDesc sharedInput = MakeTensorDesc(sharedInputDims, ParseDtype(sharedInputDtype), ACL_FORMAT_ND, {});
        TensorDesc logits = MakeTensorDesc(logitsDims, ParseDtype(logitsDtype), ACL_FORMAT_ND, {});
        TensorDesc rowIndex = MakeTensorDesc(rowIndexDims, ParseDtype(rowIndexDtype), ACL_FORMAT_ND, {});
        TensorDesc out = MakeTensorDesc(outDims, ParseDtype(outDtype), ACL_FORMAT_ND, {});
        const bool hasSharedInputTensor = !sharedInputDims.empty();
        optional<TensorDesc> bias;
        if (!biasDims.empty()) {
            bias.emplace(MakeTensorDesc(biasDims, ParseDtype(biasDtype), ACL_FORMAT_ND, {}));
        }

        const vector<int64_t> tuningVals = ParseI64List(tuningConfig);
        aclIntArray* tuningConfigPtr = nullptr;
        if (ParseBool(hasTuningConfig)) {
            tuningConfigPtr = aclCreateIntArray(const_cast<int64_t*>(tuningVals.data()), tuningVals.size());
        }

        uint64_t workspaceSize = 0;
        const GroupedMatmulFinalizeRoutingV3RunParam runParam{
            x1,
            x2,
            scale,
            offset,
            antiQuantScale,
            perTokenScale,
            groupList,
            sharedInput,
            logits,
            rowIndex,
            out,
            tuningConfigPtr,
            dtype,
            sharedInputWeight,
            sharedInputOffset,
            transposeX,
            transposeW,
            groupListType,
            ParseBool(hasOffset),
            ParseBool(hasAntiQuantScale),
            ParseBool(hasLogits),
            hasSharedInputTensor,
        };

        const aclnnStatus ret = bias.has_value() ?
                                    GetGroupedMatmulFinalizeRoutingV3Workspace(runParam, *bias, &workspaceSize) :
                                    GetGroupedMatmulFinalizeRoutingV3Workspace(runParam, nullptr, &workspaceSize);

        if (ParseBool(checkRet)) {
            EXPECT_EQ(ret, ParseStatus(expectRet));
        }
    }

    string socVersion;
    string caseName;
    string x1Shape;
    string x1Dtype;
    string x2Shape;
    string x2Dtype;
    string x2Format;
    string x2StorageShape;
    string scaleShape;
    string scaleDtype;
    string biasShape;
    string biasDtype;
    string offsetShape;
    string offsetDtype;
    string offsetFormat;
    string antiQuantScaleShape;
    string antiQuantScaleDtype;
    string perTokenScaleShape;
    string perTokenScaleDtype;
    string groupListShape;
    string groupListValues;
    string sharedInputShape;
    string sharedInputDtype;
    string logitsShape;
    string logitsDtype;
    string rowIndexShape;
    string rowIndexDtype;
    string outShape;
    string outDtype;
    int64_t dtype = 0;
    float sharedInputWeight = 1.0F;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = false;
    int64_t groupListType = 1;
    string tuningConfig;
    string hasTuningConfig;
    string hasOffset;
    string hasAntiQuantScale;
    string hasLogits;
    string expectRet;
    string checkRet;
};

vector<GroupedMatmulFinalizeRoutingV3Case> LoadCases(const string& csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<GroupedMatmulFinalizeRoutingV3Case> cases;
    string line;
    bool skippedHeader = false;
    while (getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (!skippedHeader) {
            skippedHeader = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != kV3CsvColumnCount) {
            continue;
        }
        GroupedMatmulFinalizeRoutingV3Case c;
        size_t i = 0;
        c.socVersion = Trim(cols[i++]);
        c.caseName = Trim(cols[i++]);
        c.x1Shape = Trim(cols[i++]);
        c.x1Dtype = Trim(cols[i++]);
        c.x2Shape = Trim(cols[i++]);
        c.x2Dtype = Trim(cols[i++]);
        c.x2Format = Trim(cols[i++]);
        c.x2StorageShape = Trim(cols[i++]);
        c.scaleShape = Trim(cols[i++]);
        c.scaleDtype = Trim(cols[i++]);
        c.biasShape = Trim(cols[i++]);
        c.biasDtype = Trim(cols[i++]);
        c.offsetShape = Trim(cols[i++]);
        c.offsetDtype = Trim(cols[i++]);
        c.offsetFormat = Trim(cols[i++]);
        c.antiQuantScaleShape = Trim(cols[i++]);
        c.antiQuantScaleDtype = Trim(cols[i++]);
        c.perTokenScaleShape = Trim(cols[i++]);
        c.perTokenScaleDtype = Trim(cols[i++]);
        c.groupListShape = Trim(cols[i++]);
        c.groupListValues = Trim(cols[i++]);
        c.sharedInputShape = Trim(cols[i++]);
        c.sharedInputDtype = Trim(cols[i++]);
        c.logitsShape = Trim(cols[i++]);
        c.logitsDtype = Trim(cols[i++]);
        c.rowIndexShape = Trim(cols[i++]);
        c.rowIndexDtype = Trim(cols[i++]);
        c.outShape = Trim(cols[i++]);
        c.outDtype = Trim(cols[i++]);
        c.dtype = stoll(Trim(cols[i++]));
        c.sharedInputWeight = stof(Trim(cols[i++]));
        c.sharedInputOffset = stoll(Trim(cols[i++]));
        c.transposeX = ParseBool(Trim(cols[i++]));
        c.transposeW = ParseBool(Trim(cols[i++]));
        c.groupListType = stoll(Trim(cols[i++]));
        c.tuningConfig = Trim(cols[i++]);
        c.hasTuningConfig = Trim(cols[i++]);
        c.hasOffset = Trim(cols[i++]);
        c.hasAntiQuantScale = Trim(cols[i++]);
        c.hasLogits = Trim(cols[i++]);
        c.expectRet = Trim(cols[i++]);
        c.checkRet = Trim(cols[i++]);
        cases.emplace_back(c);
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

const vector<GroupedMatmulFinalizeRoutingV3Case>& GetV3Cases()
{
    static const auto cases =
        LoadCases(ops::ut::ResolveCsvPath("test_aclnn_grouped_matmul_finalize_routing_v3_l2.csv",
                                          "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api", __FILE__));
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingV3Case>& info)
{
    return ops::ut::MakeSafeParamName(info.param.socVersion + "_" + info.param.caseName);
}

class grouped_matmul_finalize_routing_v3_csv_test : public testing::TestWithParam<GroupedMatmulFinalizeRoutingV3Case> {
};

TEST_P(grouped_matmul_finalize_routing_v3_csv_test, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMFR_V3_L2_CSV, grouped_matmul_finalize_routing_v3_csv_test, testing::ValuesIn(GetV3Cases()),
                         MakeParamName);
} // namespace
