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
 * \file test_grouped_matmul_finalize_routing_weight_quant_tiling.cpp
 * \brief Unit tests for MX-A8W4 weight quantization tiling (CSV-based)
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include <gtest/gtest.h>

#include "../../../op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace optiling
{
extern void DisablePatternCache();
extern void EnablePatternCache();
}  // namespace optiling

// Standard compile info for Ascend 950
static optiling::GroupedMatmulFinalizeRoutingCompileInfo DEFAULT_COMPILE_INFO = {
    24,                                          // aicNum
    48,                                          // aivNum
    196608,                                      // ubSize
    524288,                                      // l1Size
    33554432,                                    // l0ASize
    131072,                                      // l0BSize
    65536,                                       // l0CSize
    65536,                                       // l1CSize
    131072,                                      // reserved
    0,                                           // mixedBf16Fp32
    platform_ascendc::SocVersion::ASCEND950,     // socVersion
    false,                                       // isMsdEnable
    true,                                        // isArch950
    NpuArch::DAV_3510                            // npuArch
};

// ============================================================================
// CSV-Based Test Infrastructure
// ============================================================================

/**
 * @brief Test case data loaded from CSV
 */
struct CsvWeightQuantTestCase {
    string testName;
    int m, k, n, e, bs;
    string xDtype, wDtype, scaleDtype, biasDtype, pertokenScaleDtype;
    bool transposeW;
    string outputBs;
    int groupListType;
    int expectTilingKey;
    string expectResult;
    bool verifyBlockNum;
    string description;
};

/**
 * @brief Parse CSV line into fields
 */
static vector<string> ParseCsvLine(const string& line)
{
    vector<string> fields;
    string current;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            fields.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    fields.push_back(current);
    return fields;
}

/**
 * @brief Convert string to ge::DataType
 */
static ge::DataType StringToDataType(const string& dtype)
{
    if (dtype == "DT_FLOAT8_E4M3FN") return ge::DT_FLOAT8_E4M3FN;
    if (dtype == "DT_FLOAT4_E2M1") return ge::DT_FLOAT4_E2M1;
    if (dtype == "DT_FLOAT8_E8M0") return ge::DT_FLOAT8_E8M0;
    if (dtype == "DT_BF16") return ge::DT_BF16;
    if (dtype == "DT_FLOAT16") return ge::DT_FLOAT16;
    if (dtype == "DT_INT8") return ge::DT_INT8;
    if (dtype == "DT_FLOAT") return ge::DT_FLOAT;
    if (dtype == "DT_INT64") return ge::DT_INT64;
    return ge::DT_UNDEFINED;
}

/**
 * @brief Find CSV file path
 */
static string FindCsvFile()
{
    // First try based on the source file location so it works regardless of working directory
    string sourceFile = __FILE__;
    size_t lastSlash = sourceFile.find_last_of("/\\");
    if (lastSlash != string::npos) {
        string csvPath = sourceFile.substr(0, lastSlash) + "/test_data/weight_quant_test_cases.csv";
        ifstream testFile(csvPath);
        if (testFile.good()) {
            testFile.close();
            return csvPath;
        }
    }

    vector<string> searchPaths = {
        "gmm/grouped_matmul_finalize_routing/tests/ut/op_host/test_data/weight_quant_test_cases.csv",
        "test_data/weight_quant_test_cases.csv",
        "../test_data/weight_quant_test_cases.csv",
        "../../test_data/weight_quant_test_cases.csv",
    };

    for (const auto& path : searchPaths) {
        ifstream testFile(path);
        if (testFile.good()) {
            testFile.close();
            return path;
        }
    }
    return "";
}

/**
 * @brief Load test cases from CSV file
 */
static map<string, CsvWeightQuantTestCase> LoadCsvTestCases()
{
    map<string, CsvWeightQuantTestCase> cases;
    string csvPath = FindCsvFile();
    if (csvPath.empty()) {
        cerr << "Warning: CSV file not found" << endl;
        return cases;
    }
    
    ifstream file(csvPath);
    if (!file.is_open()) {
        cerr << "Warning: Could not open CSV file: " << csvPath << endl;
        return cases;
    }
    
    string line;
    getline(file, line);  // Skip header
    
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        vector<string> fields = ParseCsvLine(line);
        if (fields.size() < 16) continue;
        
        CsvWeightQuantTestCase tc;
        tc.testName = fields[0];
        tc.m = stoi(fields[1]);
        tc.k = stoi(fields[2]);
        tc.n = stoi(fields[3]);
        tc.e = stoi(fields[4]);
        tc.bs = stoi(fields[5]);
        tc.xDtype = fields[6];
        tc.wDtype = fields[7];
        tc.scaleDtype = fields[8];
        tc.biasDtype = fields[9];
        tc.pertokenScaleDtype = fields[10];
        tc.transposeW = (fields[11] == "true");
        tc.outputBs = fields[12];
        tc.groupListType = stoi(fields[13]);
        tc.expectTilingKey = stoi(fields[14]);
        tc.expectResult = fields[15];
        tc.verifyBlockNum = (fields.size() > 16 && fields[16] == "true");
        tc.description = (fields.size() > 17) ? fields[17] : "";
        
        cases[tc.testName] = tc;
    }
    
    return cases;
}

// Global CSV test case cache
static map<string, CsvWeightQuantTestCase>& GetCsvTestCases()
{
    static map<string, CsvWeightQuantTestCase> cases = LoadCsvTestCases();
    return cases;
}

// ============================================================================
// Test Fixture with CSV Support
// ============================================================================

class GroupedMatmulFinalizeRoutingWeightQuantTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulFinalizeRoutingWeightQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulFinalizeRoutingWeightQuantTiling TearDown" << std::endl;
    }
    
    void ExecuteCsvTest(const string& testName)
    {
        auto& testCases = GetCsvTestCases();
        
        auto it = testCases.find(testName);
        if (it == testCases.end()) {
            cerr << "Test case not found in CSV: " << testName << endl;
            EXPECT_TRUE(false) << "Test case not found in CSV: " << testName;
            return;
        }
        
        const CsvWeightQuantTestCase& tc = it->second;
        
        // Reconstruct test case from CSV data
        int m = tc.m, k = tc.k, n = tc.n, e = tc.e, bs = tc.bs;
        
        // Build shapes
        gert::StorageShape xShape = {{m, k}, {m, k}};
        gert::StorageShape wShape = {{e, n, k}, {e, (k + 31) / 32, (n + 15) / 16, 16, 32}};
        gert::StorageShape scaleShape = {{e, n, (k + 63) / 64, 2}, {e, n, (k + 63) / 64, 2}};
        gert::StorageShape biasShape = {{e, n}, {e, n}};
        gert::StorageShape pertokenScaleShape = {{m, (k + 63) / 64, 2}, {m, (k + 63) / 64, 2}};
        gert::StorageShape groupListShape = {{e}, {e}};
        gert::StorageShape sharedInputShape = {{bs, n}, {bs, n}};
        gert::StorageShape logitShape = {{m}, {m}};
        gert::StorageShape rowindexShape = {{m}, {m}};
        gert::StorageShape yShape = {{m, n}, {m, n}};
        
        // Get dtypes from CSV
        ge::DataType xDtype = StringToDataType(tc.xDtype);
        ge::DataType wDtype = StringToDataType(tc.wDtype);
        ge::DataType scaleDtype = StringToDataType(tc.scaleDtype);
        ge::DataType biasDtype = StringToDataType(tc.biasDtype);
        ge::DataType pertokenScaleDtype = StringToDataType(tc.pertokenScaleDtype);
        
        // Handle special cases from CSV
        if (tc.testName.find("NullScale") != string::npos) {
            scaleShape = {{}, {}};
        }
        if (tc.testName.find("NullPertokenScale") != string::npos) {
            pertokenScaleShape = {{}, {}};
        }
        if (tc.testName.find("NullRowIndex") != string::npos) {
            rowindexShape = {{}, {}};
        }
        if (tc.testName.find("NullGroupList") != string::npos) {
            groupListShape = {{}, {}};
        }
        if (tc.testName.find("EmptyBias") != string::npos) {
            biasShape = {{}, {}};
        }
        if (tc.testName.find("NullSharedInput") != string::npos) {
            sharedInputShape = {{}, {}};
        }
        if (tc.testName.find("WrongXShape") != string::npos) {
            xShape = {{m, k, 1}, {m, k, 1}};
        }
        if (tc.testName.find("WrongWShape") != string::npos) {
            wShape = {{e, n}, {e, n}};
        }
        if (tc.testName.find("EMismatch") != string::npos) {
            scaleShape = {{e + 1, n, (k + 63) / 64, 2}, {e + 1, n, (k + 63) / 64, 2}};
        }
        if (tc.testName.find("KMismatch") != string::npos) {
            scaleShape = {{e, n, (k + 63) / 64 + 1, 2}, {e, n, (k + 63) / 64 + 1, 2}};
        }
        
        // Build attributes - MUST be in correct index order:
        // 0:dtype, 1:shared_input_weight, 2:shared_input_offset, 3:transpose_x
        // 4:transpose_w, 5:output_bs, 6:group_list_type, 7:tuning_config
        vector<gert::TilingContextPara::OpAttr> attrs;
        attrs.push_back({"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)});                        // index 0
        attrs.push_back({"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)});         // index 1
        attrs.push_back({"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)});         // index 2
        attrs.push_back({"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)});                // index 3
        attrs.push_back({"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(tc.transposeW)});        // index 4
        // output_bs must be index 5
        if (tc.outputBs != "NULL") {
            attrs.push_back({"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(stoi(tc.outputBs))}); // index 5
        } else {
            // Add default value to maintain index alignment
            attrs.push_back({"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)});
        }
        attrs.push_back({"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tc.groupListType)}); // index 6
        attrs.push_back({"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)});                  // index 7
        
        gert::TilingContextPara tilingContextPara(
            "GroupedMatmulFinalizeRouting",
            {
                {xShape, xDtype, ge::FORMAT_ND},
                {wShape, wDtype, ge::FORMAT_FRACTAL_NZ},
                {scaleShape, scaleDtype, ge::FORMAT_ND},
                {biasShape, biasDtype, ge::FORMAT_ND},
                {pertokenScaleShape, pertokenScaleDtype, ge::FORMAT_ND},
                {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
                {sharedInputShape, ge::DT_BF16, ge::FORMAT_ND},
                {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
                {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}
            },
            {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
            attrs,
            &DEFAULT_COMPILE_INFO,
            "Ascend950"
        );
        
        // Execute test
        if (tc.expectResult == "SUCCESS") {
            TilingInfo tilingInfo;
            bool result = ExecuteTiling(tilingContextPara, tilingInfo);
            EXPECT_TRUE(result) << tc.description;
            EXPECT_EQ(tilingInfo.tilingKey, tc.expectTilingKey) << tc.description;
            if (tc.verifyBlockNum) {
                EXPECT_EQ(tilingInfo.blockNum, DEFAULT_COMPILE_INFO.aicNum);
            }
            
            // Verify initSize calculation
            if (tilingInfo.tilingData != nullptr && tilingInfo.tilingDataSize > 0) {
                auto* weightQuantTiling = reinterpret_cast<const GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData*>(tilingInfo.tilingData.get());
                uint64_t actualInitSize = weightQuantTiling->initSize;
                
                // Calculate expected initSize: (outputBS - sharedInputLen) * nSize
                int64_t outputBS = (tc.outputBs != "NULL") ? stoi(tc.outputBs) : (tc.e > 0 ? tc.m / tc.e : 0);
                uint64_t expectedSharedInputLen = tc.testName.find("NullSharedInput") != string::npos ? 0 : tc.bs;
                uint64_t expectedInitSize = (outputBS - expectedSharedInputLen) * tc.n;
                
                EXPECT_EQ(actualInitSize, expectedInitSize) 
                    << tc.testName << ": initSize mismatch. Expected=" << expectedInitSize 
                    << " (outputBS=" << outputBS << " - bs=" << tc.bs << ") * n=" << tc.n
                    << ", Actual=" << actualInitSize;
                
                // Verify nSize
                uint64_t actualNSize = weightQuantTiling->nSize;
                uint64_t expectedNSize = tc.n;
                EXPECT_EQ(actualNSize, expectedNSize) 
                    << tc.testName << ": nSize mismatch. Expected=" << expectedNSize 
                    << ", Actual=" << actualNSize;
                
                // Verify sharedInputLen
                uint64_t actualSharedInputLen = weightQuantTiling->sharedInputLen;
                EXPECT_EQ(actualSharedInputLen, expectedSharedInputLen) 
                    << tc.testName << ": sharedInputLen mismatch. Expected=" << expectedSharedInputLen 
                    << ", Actual=" << actualSharedInputLen;
            }
        } else {
            ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
        }
    }
};

// ============================================================================
// CSV-Based Test Cases (All 20 tests from CSV file)
// ============================================================================

TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzNormalCase) { ExecuteCsvTest("TestMXA8W4WeightNzNormalCase"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzCoreNum) { ExecuteCsvTest("TestMXA8W4WeightNzCoreNum"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzNullSharedInput) { ExecuteCsvTest("TestMXA8W4WeightNzNullSharedInput"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzKNotAligned) { ExecuteCsvTest("TestMXA8W4WeightNzKNotAligned"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzNNotAligned) { ExecuteCsvTest("TestMXA8W4WeightNzNNotAligned"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzKTooSmall) { ExecuteCsvTest("TestMXA8W4WeightNzKTooSmall"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzNTooSmall) { ExecuteCsvTest("TestMXA8W4WeightNzNTooSmall"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzNullScale) { ExecuteCsvTest("TestMXA8W4WeightNzNullScale"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzNullPertokenScale) { ExecuteCsvTest("TestMXA8W4WeightNzNullPertokenScale"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzNullRowIndex) { ExecuteCsvTest("TestMXA8W4WeightNzNullRowIndex"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzNullGroupList) { ExecuteCsvTest("TestMXA8W4WeightNzNullGroupList"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzWrongXDtype) { ExecuteCsvTest("TestMXA8W4WeightNzWrongXDtype"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzWrongWDtype) { ExecuteCsvTest("TestMXA8W4WeightNzWrongWDtype"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzWrongScaleDtype) { ExecuteCsvTest("TestMXA8W4WeightNzWrongScaleDtype"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzTransposeWFalse) { ExecuteCsvTest("TestMXA8W4WeightNzTransposeWFalse"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzWrongXShape) { ExecuteCsvTest("TestMXA8W4WeightNzWrongXShape"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzWrongWShape) { ExecuteCsvTest("TestMXA8W4WeightNzWrongWShape"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzEMismatch) { ExecuteCsvTest("TestMXA8W4WeightNzEMismatch"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzKMismatch) { ExecuteCsvTest("TestMXA8W4WeightNzKMismatch"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzWithBias) { ExecuteCsvTest("TestMXA8W4WeightNzWithBias"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzDefaultOutputBs) { ExecuteCsvTest("TestMXA8W4WeightNzDefaultOutputBs"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzSmallDimensions) { ExecuteCsvTest("TestMXA8W4WeightNzSmallDimensions"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzEmptyBias) { ExecuteCsvTest("TestMXA8W4WeightNzEmptyBias"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzEZeroNullOutputBs) { ExecuteCsvTest("TestMXA8W4WeightNzEZeroNullOutputBs"); }
TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzEZeroWithOutputBs) { ExecuteCsvTest("TestMXA8W4WeightNzEZeroWithOutputBs"); }

TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestCsvFileLoaded)
{
    auto& cases = GetCsvTestCases();
    EXPECT_EQ(cases.size(), 30) << "Expected 30 test cases in CSV file";

    cout << "Successfully loaded " << cases.size() << " test cases from CSV" << endl;
    for (const auto& pair : cases) {
        cout << "  - " << pair.first << endl;
    }
}

TEST_F(GroupedMatmulFinalizeRoutingWeightQuantTiling, TestMXA8W4WeightNzInvalidAicAivRatio)
{
    // Use invalid compile info: aicNum=24, aivNum=24 (expected 1:2, i.e. 24:48)
    optiling::GroupedMatmulFinalizeRoutingCompileInfo invalidCompileInfo = DEFAULT_COMPILE_INFO;
    invalidCompileInfo.aicNum = 24;
    invalidCompileInfo.aivNum = 24;

    int m = 128, k = 256, n = 256, e = 2;

    gert::StorageShape xShape = {{m, k}, {m, k}};
    gert::StorageShape wShape = {{e, n, k}, {e, (k + 31) / 32, (n + 15) / 16, 16, 32}};
    gert::StorageShape scaleShape = {{e, n, (k + 63) / 64, 2}, {e, n, (k + 63) / 64, 2}};
    gert::StorageShape biasShape = {{}, {}};
    gert::StorageShape pertokenScaleShape = {{m, (k + 63) / 64, 2}, {m, (k + 63) / 64, 2}};
    gert::StorageShape groupListShape = {{e}, {e}};
    gert::StorageShape sharedInputShape = {{}, {}};
    gert::StorageShape logitShape = {{m}, {m}};
    gert::StorageShape rowindexShape = {{m}, {m}};
    gert::StorageShape yShape = {{m, n}, {m, n}};

    vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.push_back({"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)});
    attrs.push_back({"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)});
    attrs.push_back({"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)});
    attrs.push_back({"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)});
    attrs.push_back({"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(true)});
    attrs.push_back({"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)});
    attrs.push_back({"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)});
    attrs.push_back({"tuning_config", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)});

    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulFinalizeRouting",
        {
            {xShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {wShape, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ},
            {scaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {pertokenScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
            {sharedInputShape, ge::DT_BF16, ge::FORMAT_ND},
            {logitShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {rowindexShape, ge::DT_INT64, ge::FORMAT_ND}
        },
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        attrs,
        &invalidCompileInfo,
        "Ascend950"
    );

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
