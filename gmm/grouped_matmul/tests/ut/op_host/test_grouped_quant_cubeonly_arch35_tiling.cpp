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
 * \file test_grouped_quant_cubeonly_arch35_tiling.cpp
 * \brief CSV-driven unit tests for GroupedMatmul Quant CUBE/PERTENSOR_CUBE arch35 cube-only tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include "acl/acl_rt.h"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {
int32_t gDevkitVersion = 0;
aclError gDevkitVersionStatus = ACL_SUCCESS;

class ScopedDevkitVersion {
public:
    ScopedDevkitVersion(int32_t version, aclError status)
        : previousVersion_(gDevkitVersion),
          previousStatus_(gDevkitVersionStatus)
    {
        gDevkitVersion = version;
        gDevkitVersionStatus = status;
    }
    ~ScopedDevkitVersion()
    {
        gDevkitVersion = previousVersion_;
        gDevkitVersionStatus = previousStatus_;
    }

private:
    int32_t previousVersion_;
    aclError previousStatus_;
};
} // namespace

// Host UT executable stub; production resolves this API from acl_rt.
extern "C" aclError aclsysGetVersionNum(char* name, int32_t* version)
{
    if (name == nullptr || version == nullptr || std::strcmp(name, "asc-devkit") != 0) {
        return static_cast<aclError>(1);
    }
    *version = gDevkitVersion;
    return gDevkitVersionStatus;
}

namespace {
// Routing may only inspect this shared prefix before deciding which full layout to read.
static_assert(offsetof(GroupedMatmulTilingData::GMMQuantTilingData, gmmQuantParams) == 0);
static_assert(offsetof(GroupedMatmulTilingData::GMMQuantCubeBasicApiTilingData, gmmQuantParams) == 0);
static_assert(sizeof(GroupedMatmulTilingData::GMMQuantBasicApiTilingData) == 72);
static_assert(sizeof(GroupedMatmulTilingData::GMMQuantCubeBasicApiTilingData) == 1608);
static_assert(offsetof(GroupedMatmulTilingData::GMMQuantCubeBasicApiTilingData, mmTilingData) == 24);
static_assert(offsetof(GroupedMatmulTilingData::GMMQuantCubeBasicApiTilingData, gmmArray) == 72);

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

DataType ParseDtype(const string& dtype)
{
    const auto parsed = ops::ut::ParseGeDtype(dtype);
    if (parsed == ge::DT_UNDEFINED) {
        cerr << "Unsupported dtype in csv: " << dtype << endl;
    }
    return parsed;
}

gert::StorageShape MakeEmptyShape()
{
    return gert::StorageShape();
}

string TilingData2Str(const void* tilingData, size_t tilingSize)
{
    ostringstream oss;
    const auto* data = reinterpret_cast<const int32_t*>(tilingData);
    const size_t len = tilingSize / sizeof(int32_t);
    for (size_t i = 0; i < len; ++i) {
        if (i != 0) {
            oss << " ";
        }
        oss << data[i];
    }
    return oss.str();
}

vector<gert::TilingContextPara::OpAttr> GetGroupedQmmAttrs(bool transposeX, bool transposeWeight, int64_t groupType)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    };
}

class GroupedQuantCubeonlyArch35TilingTestParam {
public:
    void Prepare(optiling::GMMCompileInfo& compileInfo) const
    {
        compileInfo = {
            static_cast<uint32_t>(coreNum > 0 ? coreNum : 32),     // aicNum
            static_cast<uint32_t>(coreNum > 0 ? coreNum * 2 : 64), // aivNum
            262144,                                                // ubSize
            l1Bytes,                                               // l1Size
            134217728,                                             // l2Size
            l0CBytes,                                              // l0CSize
            l0ABytes,                                              // l0ASize
            l0BBytes,                                              // l0BSize
            platform_ascendc::SocVersion::ASCEND950,               // socVersion
            NpuArch::DAV_3510,
        };
    }

    void InvokeTilingFunc(optiling::GMMCompileInfo& compileInfo) const
    {
        vector<int64_t> xOriginDims = transposeX ? vector<int64_t>{k, m} : vector<int64_t>{m, k};
        gert::StorageShape xShape = ops::ut::MakeGertStorageShape(xOriginDims, xOriginDims);
        gert::StorageShape biasShape =
            hasBias ? ops::ut::MakeGertStorageShape({groupNum, n}, {groupNum, n}) : MakeEmptyShape();
        gert::StorageShape groupListShape = ops::ut::MakeGertStorageShape({groupNum}, {groupNum});
        // Cube-only Quant CUBE / PERTENSOR_CUBE STC scenario:
        //   scale is per-tensor [E] / [E,1] or per-channel [E,N].
        //   perTokenScale is usually empty; PERTENSOR_DOUBLE uses [E].
        gert::StorageShape perTokenScaleShape = MakeEmptyShape();
        if (perTokenScaleMode == "PERTENSOR" || scaleMode == "PERTENSOR_DOUBLE") {
            perTokenScaleShape = ops::ut::MakeGertStorageShape({groupNum}, {groupNum});
        }

        vector<int64_t> weightOriginDims =
            transposeWeight ? vector<int64_t>{groupNum, n, k} : vector<int64_t>{groupNum, k, n};
        vector<int64_t> weightStorageDims;
        if (weightFormat == "NZ") {
            const bool weight4Bit =
                weightDtype == ge::DT_INT4 || weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT4_E1M2;
            const int64_t n0 = weight4Bit ? 64 : 32;
            const int64_t n1 = (n + n0 - 1) / n0;
            const int64_t k1 = (k + 15) / 16;
            weightStorageDims = transposeWeight ? vector<int64_t>{groupNum, (k + n0 - 1) / n0, (n + 15) / 16, 16, n0} :
                                                  vector<int64_t>{groupNum, n1, k1, 16, n0};
        } else {
            weightStorageDims = weightOriginDims;
        }
        gert::StorageShape weightShape = ops::ut::MakeGertStorageShape(weightOriginDims, weightStorageDims);

        gert::StorageShape scaleShape = MakeEmptyShape();
        if (scaleMode == "PERTENSOR" || scaleMode == "PERTENSOR_DOUBLE") {
            scaleShape = ops::ut::MakeGertStorageShape({groupNum}, {groupNum});
        } else if (scaleMode == "PERCHANNEL") {
            scaleShape = ops::ut::MakeGertStorageShape({groupNum, n}, {groupNum, n});
        } else if (scaleMode == "NONE") {
            scaleShape = MakeEmptyShape();
        } else {
            ADD_FAILURE() << "Unsupported scaleMode: " << scaleMode << ", case=" << caseName;
        }

        gert::TilingContextPara tilingContextPara(
            "GroupedMatmul",
            {
                {xShape, xDtype, ge::FORMAT_ND},                                                          // x
                {weightShape, weightDtype, weightFormat == "NZ" ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND}, // weight
                {biasShape, biasDtype, ge::FORMAT_ND},                                                    // bias
                {scaleShape, scaleDtype, ge::FORMAT_ND},                                                  // scale
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},                                          // offset
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},         // antiquantScale
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},         // antiquantOffset
                {groupListShape, ge::DT_INT64, ge::FORMAT_ND},           // groupList
                {perTokenScaleShape, perTokenScaleDtype, ge::FORMAT_ND}, // perTokenScale
            },
            {{ops::ut::MakeGertStorageShape({m}, {n}), yDtype, ge::FORMAT_ND}},
            GetGroupedQmmAttrs(transposeX, transposeWeight, groupType), &compileInfo, "3510", compileInfo.aicNum,
            compileInfo.ubSize);

        // Tiling prefers PlatformInfo over CompileInfo; keep the simulated capacities consistent.
        std::ostringstream socInfo;
        socInfo << R"({"hardware_info":{"BT_SIZE":0,"load3d_constraints":"1",)"
                << R"("Intrinsic_fix_pipe_l0c2out":false,"Intrinsic_data_move_l12ub":true,)"
                << R"("Intrinsic_data_move_l0c2ub":true,"Intrinsic_data_move_out2l1_nd2nz":false,)"
                << R"("UB_SIZE":)" << compileInfo.ubSize << R"(,"L2_SIZE":33554432,"L1_SIZE":)" << compileInfo.l1Size
                << R"(,"L0A_SIZE":)" << compileInfo.l0ASize << R"(,"L0B_SIZE":)" << compileInfo.l0BSize
                << R"(,"L0C_SIZE":)" << compileInfo.l0CSize << R"(,"CORE_NUM":)" << compileInfo.aicNum
                << R"(,"socVersion":"3510"}})";
        tilingContextPara.socInfoString_ = socInfo.str();

        TilingInfo tilingInfo;
        bool tilingResult = ExecuteTiling(tilingContextPara, tilingInfo);
        ASSERT_EQ(tilingResult, result) << "prefix=" << prefix << ", case=" << caseName;
        if (!result) {
            return;
        }
        ASSERT_EQ(tilingInfo.blockNum, expectBlockDim) << "prefix=" << prefix;
        if (expectCubeBasicApi) {
            // Fixpipe keeps kernel type 0; B transpose occupies the low two bits.
            ASSERT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(transposeWeight));
            ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMQuantCubeBasicApiTilingData));
            const auto& data = *reinterpret_cast<const GroupedMatmulTilingData::GMMQuantCubeBasicApiTilingData*>(
                tilingInfo.tilingData.get());
            const auto& mm = data.mmTilingData;
            EXPECT_EQ(mm.m, m);
            EXPECT_EQ(mm.n, n);
            EXPECT_EQ(mm.k, k);
            EXPECT_EQ(mm.isBias, hasBias);
            EXPECT_EQ(mm.l1BufferStage, 2);
            EXPECT_EQ(mm.scaleKAL1, 0);
            EXPECT_EQ(mm.scaleKBL1, 0);
            EXPECT_EQ(mm.reserved1, 0);
            EXPECT_EQ(mm.reserved2, 0);
            EXPECT_EQ(data.gmmQuantParams.groupNum, groupNum);
            EXPECT_EQ(data.gmmQuantParams.groupType, groupType);
            EXPECT_EQ(data.gmmQuantParams.hasBias, hasBias);
            EXPECT_EQ(data.gmmQuantParams.singleX, 1);
            EXPECT_EQ(data.gmmQuantParams.singleW, 1);
            EXPECT_EQ(data.gmmQuantParams.singleY, 1);
            EXPECT_EQ(data.gmmArray.mList[0], -1);
            EXPECT_EQ(data.gmmArray.kList[0], k);
            EXPECT_EQ(data.gmmArray.nList[0], n);
            for (size_t i = 1; i < 128; ++i) {
                EXPECT_EQ(data.gmmArray.mList[i], 0);
                EXPECT_EQ(data.gmmArray.kList[i], 0);
                EXPECT_EQ(data.gmmArray.nList[i], 0);
            }
            ASSERT_GT(mm.baseK, 0);
            ASSERT_GE(mm.kAL1, mm.baseK);
            ASSERT_GE(mm.kBL1, mm.baseK);
            EXPECT_EQ(mm.kAL1 % mm.baseK, 0);
            EXPECT_EQ(mm.kBL1 % mm.baseK, 0);
            if (expectedKAL1 != 0) {
                EXPECT_EQ(mm.kAL1, expectedKAL1);
                EXPECT_EQ(mm.kBL1, expectedKBL1);
            }
            const auto align = [](uint64_t value, uint64_t unit) { return (value + unit - 1) / unit * unit; };
            const bool fp4 = xDtype == ge::DT_FLOAT4_E2M1 || xDtype == ge::DT_FLOAT4_E1M2;
            const uint64_t c0 = fp4 ? 64 : 32;
            // All covered input storage types have sizeof == 1 in BlockMmad.
            const uint64_t aStage = align(mm.baseM, 16) * align(mm.kAL1, c0);
            const uint64_t bStage =
                transposeWeight ? align(mm.kBL1, c0) * align(mm.baseN, 16) : align(mm.kBL1, 16) * align(mm.baseN, c0);
            const uint64_t scaleStage = scaleMode == "PERCHANNEL" ? align(mm.baseN * 8UL, 32) : 0;
            const uint64_t biasStage = hasBias ? align(mm.baseN * 4UL, 32) : 0;
            EXPECT_LE(aStage + bStage + scaleStage + biasStage, compileInfo.l1Size / 2);
            EXPECT_LE(align(mm.baseM, 16) * align(mm.baseK, c0), compileInfo.l0ASize / 2);
            EXPECT_LE(align(mm.baseN, c0) * align(mm.baseK, c0), compileInfo.l0BSize / 2);
            const uint64_t cBytes = uint64_t{mm.baseM} * mm.baseN * 4;
            EXPECT_LE(cBytes * mm.dbL0C, compileInfo.l0CSize);
            EXPECT_EQ(mm.dbL0C, cBytes * 2 <= compileInfo.l0CSize ? 2 : 1);
            return;
        }
        ASSERT_EQ(tilingInfo.tilingKey, expectTilingKey) << "prefix=" << prefix;
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMQuantTilingData));
        if (!expectTilingData.empty()) {
            EXPECT_EQ(TilingData2Str(tilingInfo.tilingData.get(), tilingInfo.tilingDataSize), expectTilingData);
        }
    }

    void Test() const
    {
        ScopedDevkitVersion versionScope(devkitVersion, devkitVersionStatus);
        optiling::GMMCompileInfo compileInfo;
        Prepare(compileInfo);
        InvokeTilingFunc(compileInfo);
    }

    int32_t devkitVersion = 90100000;
    aclError devkitVersionStatus = ACL_SUCCESS;
    bool expectCubeBasicApi = true;
    uint64_t l1Bytes = 524288;
    uint64_t l0ABytes = 65536;
    uint64_t l0BBytes = 65536;
    uint64_t l0CBytes = 262144;
    uint32_t expectedKAL1 = 0;
    uint32_t expectedKBL1 = 0;
    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    int64_t coreNum = -1;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t groupNum = 0;
    bool transposeX = false;
    bool transposeWeight = false;
    int64_t groupType = 0;
    string weightFormat;
    ge::DataType xDtype = ge::DT_UNDEFINED;
    ge::DataType weightDtype = ge::DT_UNDEFINED;
    ge::DataType scaleDtype = ge::DT_UNDEFINED;
    ge::DataType perTokenScaleDtype = ge::DT_UNDEFINED;
    ge::DataType biasDtype = ge::DT_UNDEFINED;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    bool hasBias = false;
    string scaleMode = "PERTENSOR";    // PERTENSOR / PERCHANNEL / PERTENSOR_DOUBLE
    string perTokenScaleMode = "NONE"; // NONE / PERTENSOR
    bool result = false;
    uint64_t expectBlockDim = 0;
    uint64_t expectTilingKey = 0;
    string expectTilingData;
};

vector<GroupedQuantCubeonlyArch35TilingTestParam> GetParams(const string& socVersion)
{
    vector<GroupedQuantCubeonlyArch35TilingTestParam> params;
    std::string csvPath = ops::ut::ResolveCsvPath("test_grouped_quant_cubeonly_arch35_tiling.csv",
                                                  "gmm/grouped_matmul/tests/ut/op_host", __FILE__);
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << ", maybe not exist" << endl;
        return params;
    }

    string line;
    size_t lineNo = 0U;
    while (getline(csvData, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < 26U) {
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            GroupedQuantCubeonlyArch35TilingTestParam param;
            size_t idx = 0UL;
            param.socVersion = items[idx++];
            if (param.socVersion != socVersion) {
                continue;
            }

            param.caseName = items[idx++];
            param.enable = ParseBool(items[idx++]);
            if (!param.enable) {
                continue;
            }
            param.prefix = items[idx++];
            param.coreNum = items[idx].empty() ? -1 : stoll(items[idx]);
            idx++;
            param.m = stoll(items[idx++]);
            param.k = stoll(items[idx++]);
            param.n = stoll(items[idx++]);
            param.groupNum = stoll(items[idx++]);
            param.transposeX = ParseBool(items[idx++]);
            param.transposeWeight = ParseBool(items[idx++]);
            param.groupType = stoll(items[idx++]);
            param.weightFormat = items[idx++];
            param.xDtype = ParseDtype(items[idx++]);
            param.weightDtype = ParseDtype(items[idx++]);
            param.scaleDtype = ParseDtype(items[idx++]);
            param.perTokenScaleDtype = ParseDtype(items[idx++]);
            param.biasDtype = ParseDtype(items[idx++]);
            param.yDtype = ParseDtype(items[idx++]);
            param.hasBias = ParseBool(items[idx++]);
            param.scaleMode = items[idx++];
            param.perTokenScaleMode = items[idx++];
            param.result = ParseBool(items[idx++]);
            param.expectBlockDim = static_cast<uint64_t>(stoull(items[idx++]));
            param.expectTilingKey = static_cast<uint64_t>(stoull(items[idx++]));
            param.expectTilingData = items[idx++];
            params.push_back(param);
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return params;
}

string MakeParamName(const testing::TestParamInfo<GroupedQuantCubeonlyArch35TilingTestParam>& info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedQuantCubeonlyArch35TilingUT {

const vector<GroupedQuantCubeonlyArch35TilingTestParam>& GetAscend950Params()
{
    static const vector<GroupedQuantCubeonlyArch35TilingTestParam> params = GetParams("Ascend950");
    return params;
}

GroupedQuantCubeonlyArch35TilingTestParam MakeCubeCase()
{
    GroupedQuantCubeonlyArch35TilingTestParam p;
    p.expectBlockDim = 32;
    p.caseName = p.prefix = "cube_basic_contract";
    p.m = 64;
    p.n = 256;
    p.k = 4096;
    p.groupNum = 4;
    p.xDtype = p.weightDtype = ge::DT_INT8;
    p.yDtype = ge::DT_FLOAT16;
    p.scaleDtype = ge::DT_UINT64;
    p.biasDtype = ge::DT_INT32;
    p.hasBias = false;
    p.transposeX = p.transposeWeight = false;
    p.weightFormat = "ND";
    p.scaleMode = "PERTENSOR";
    p.perTokenScaleMode = "NONE";
    p.groupType = 0;
    p.expectTilingKey = 0;
    p.expectTilingData.clear();
    p.result = true;
    return p;
}

TEST(TestGroupedQuantCubeBasicApi, BandwidthTarget)
{
    auto p = MakeCubeCase();
    p.expectedKAL1 = 1024;
    p.expectedKBL1 = 256;
    p.Test();
}

TEST(TestGroupedQuantCubeBasicApi, Int32WithoutScale)
{
    auto p = MakeCubeCase();
    p.yDtype = ge::DT_INT32;
    p.scaleMode = "NONE";
    for (auto scaleDtype : {ge::DT_UINT64, ge::DT_FLOAT, ge::DT_UNDEFINED}) {
        p.scaleDtype = scaleDtype;
        p.Test();
    }
}

TEST(TestGroupedQuantCubeBasicApi, AbsentBiasIgnoresDescriptorDefault)
{
    for (auto biasDtype : {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_UNDEFINED}) {
        auto p = MakeCubeCase();
        p.biasDtype = biasDtype;
        p.Test();
        p.xDtype = p.weightDtype = ge::DT_FLOAT8_E4M3FN;
        p.scaleDtype = ge::DT_FLOAT;
        p.yDtype = ge::DT_BF16;
        p.Test();
    }
}

TEST(TestGroupedQuantCubeBasicApi, HiFloat8Routing)
{
    for (bool transposeWeight : {false, true}) {
        for (const auto& format : {"ND", "NZ"}) {
            for (bool hasBias : {false, true}) {
                for (auto biasDtype : {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_UNDEFINED}) {
                    auto p = MakeCubeCase();
                    p.xDtype = p.weightDtype = ge::DT_HIFLOAT8;
                    p.yDtype = ge::DT_FLOAT;
                    p.scaleDtype = ge::DT_FLOAT;
                    p.biasDtype = biasDtype;
                    p.hasBias = hasBias;
                    p.transposeWeight = transposeWeight;
                    p.weightFormat = format;
                    p.expectTilingKey = static_cast<uint64_t>(transposeWeight);
                    // Existing public validation rejects HIFLOAT8 NZ and any actual bias.
                    p.result = p.weightFormat == "ND" && !hasBias;
                    p.expectCubeBasicApi = p.result;
                    p.Test();
                }
            }
        }
    }
}

TEST(TestGroupedQuantCubeBasicApi, NativeBiasAndKTail)
{
    auto p = MakeCubeCase();
    p.k = 4097;
    p.hasBias = true;
    p.scaleMode = "PERCHANNEL";
    p.Test();
}

TEST(TestGroupedQuantCubeBasicApi, RejectTransposeAForSplitM)
{
    auto p = MakeCubeCase();
    p.xDtype = p.weightDtype = ge::DT_FLOAT8_E4M3FN;
    p.scaleDtype = ge::DT_FLOAT;
    p.yDtype = ge::DT_BF16;
    p.transposeX = true;
    p.expectCubeBasicApi = false;
    p.result = false;
    p.Test();
}

TEST(TestGroupedQuantCubeBasicApi, BiasEpilogueUsesOriginalTiling)
{
    auto p = MakeCubeCase();
    p.hasBias = true;
    p.biasDtype = ge::DT_FLOAT;
    p.scaleDtype = ge::DT_FLOAT;
    p.expectCubeBasicApi = false;
    p.expectTilingKey = 0x10;
    p.Test();
}

TEST(TestGroupedQuantCubeBasicApi, DevkitVersionRouting)
{
    struct VersionCase {
        int32_t version;
        bool basicApi;
    };
    const VersionCase cases[] = {{0, false},        {80500000, false}, {90000000, false}, {90099999, false},
                                 {90100000, true},  {90101999, true},  {90200000, true},  {100000000, false},
                                 {100100000, true}, {-1, false}};
    for (const auto& versionCase : cases) {
        for (bool transposeWeight : {false, true}) {
            for (auto format : {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ}) {
                auto p = MakeCubeCase();
                p.devkitVersion = versionCase.version;
                p.expectCubeBasicApi = versionCase.basicApi;
                p.transposeWeight = transposeWeight;
                p.expectTilingKey = static_cast<uint64_t>(transposeWeight);
                p.weightFormat = format == ge::FORMAT_ND ? "ND" : "NZ";
                p.Test();
            }
        }
    }
}

TEST(TestGroupedQuantCubeBasicApi, DevkitQueryFailureUsesHostOriginalLayout)
{
    // Host behavior only: a Blaze-built kernel cannot distinguish this fallback layout.
    auto p = MakeCubeCase();
    p.devkitVersionStatus = static_cast<aclError>(1);
    p.expectCubeBasicApi = false;
    p.Test();
}

TEST(TestGroupedQuantCubeBasicApi, ScaleAndBiasReservedInBothHalves)
{
    auto p = MakeCubeCase();
    p.hasBias = true;
    p.scaleMode = "PERCHANNEL";
    // A 64 KiB + B 64 KiB + scale 2 KiB + bias 1 KiB requires 134144 per half.
    p.l1Bytes = 268288;
    p.expectedKAL1 = 1024;
    p.expectedKBL1 = 256;
    p.Test();
    p.l1Bytes -= 32; // must shrink rather than place auxiliary buffers past the half boundary
    p.expectedKAL1 = 512;
    p.Test();
}

TEST(TestGroupedQuantCubeBasicApi, RejectInsufficientL1)
{
    auto p = MakeCubeCase();
    p.l1Bytes = 1024;
    p.result = false;
    p.Test();
}

TEST(TestGroupedQuantCubeBasicApi, RejectInsufficientL0)
{
    auto p = MakeCubeCase();
    p.l0ABytes = p.l0BBytes = 1024;
    p.result = false;
    p.Test();
    p.l0ABytes = p.l0BBytes = 65536;
    p.l0CBytes = 1024;
    p.Test();
}

class TestGroupedQuantCubeonlyArch35Tiling : public testing::TestWithParam<GroupedQuantCubeonlyArch35TilingTestParam> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedQuantCubeonlyArch35Tiling, generalTest)
{
    GetParam().Test();
    auto oldDevkit = GetParam();
    oldDevkit.devkitVersion = 90000000;
    oldDevkit.expectCubeBasicApi = false;
    // Preserve the 128 KiB L0C capacity used to generate the legacy CSV expectations.
    oldDevkit.l0CBytes = 128 * 1024;
    oldDevkit.Test();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_QMM_CUBEONLY_950, TestGroupedQuantCubeonlyArch35Tiling,
                         testing::ValuesIn(GetAscend950Params()), MakeParamName);

} // namespace GroupedQuantCubeonlyArch35TilingUT
