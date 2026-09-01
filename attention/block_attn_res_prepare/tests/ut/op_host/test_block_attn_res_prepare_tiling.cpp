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
 * \file test_block_attn_res_prepare_tiling.cpp
 * \brief CSV-driven tiling unit tests for BlockAttnResPrepare.
 */

#include <gtest/gtest.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "../../../op_host/op_tiling/block_attn_res_prepare_base_tiling.h"
#include "../../../op_kernel/arch35/block_attn_res_prepare_tiling_data.h"
#include "gmm_csv_ge_parse_utils.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

constexpr uint64_t SYSTEM_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t ASCEND_950_UB_SIZE = 253952UL;
constexpr uint64_t ASCEND_950_L1_SIZE = 512UL * 1024UL;
constexpr uint64_t ASCEND_950_L0A_SIZE = 64UL * 1024UL;
constexpr uint64_t ASCEND_950_L0B_SIZE = 64UL * 1024UL;
constexpr uint64_t ASCEND_950_L0C_SIZE = 256UL * 1024UL;
constexpr uint64_t FAKE_L2_SIZE = 4096UL;
constexpr uint32_t VECTOR_TILING_DATA_SIZE = 48U;
constexpr uint32_t MIX_TILING_DATA_SIZE = 136U;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;
constexpr uint64_t MAX_RUNTIME_BASE_T = 16U;
constexpr size_t T_DIM_INDEX = 0U;
constexpr size_t D_DIM_INDEX = 2U;
constexpr size_t S_DIM_INDEX = 0U;
constexpr size_t CSV_COLUMN_COUNT = 27U;
constexpr const char *CSV_FILE_NAME = "test_block_attn_res_prepare_tiling.csv";
constexpr const char *CSV_REPO_DIR = "attention/block_attn_res_prepare/tests/ut/op_host";

struct VectorTilingDataView {
    uint32_t totalT;
    uint32_t totalS;
    uint32_t totalWorkUnits;
    uint32_t totalD;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;
    uint32_t baseD;
    uint32_t statUbElems;
    float eps;
    uint16_t usedCoreNum;
    uint16_t bigCoreNum;
    uint8_t totalN;
    uint8_t vCacheRows;
    uint8_t qBufferNum;
    uint8_t vBufferNum;
    uint8_t oBufferNum;
};

struct MixTilingDataView {
    uint64_t qL1Elems;
    uint64_t vL1Elems;
    uint64_t eL1Elems;
    uint64_t vUbElems;
    uint64_t dotUbElems;
    uint64_t reduceUbElems;
    uint64_t softmaxUbElems;
    uint64_t workspacePerCoreElems;
    uint32_t totalT;
    uint32_t totalS;
    uint32_t totalWorkUnits;
    uint32_t totalD;
    uint32_t baseT;
    uint32_t baseS;
    uint32_t baseD;
    uint32_t baseDAlign;
    uint32_t sTileNum;
    uint32_t dTileNum;
    uint32_t sAlign;
    uint32_t nAlign;
    uint32_t mm1NAlign;
    uint32_t dAlign;
    float eps;
    uint16_t usedCoreNum;
    uint16_t aicCoreNum;
    uint16_t aivCoreNum;
    uint8_t totalN;
    uint8_t qL1BufferNum;
    uint8_t vL1BufferNum;
    uint8_t vUbBufferNum;
};

static_assert(sizeof(VectorTilingDataView) == VECTOR_TILING_DATA_SIZE);
static_assert(sizeof(MixTilingDataView) == MIX_TILING_DATA_SIZE);

void ExpectValueIfSpecified(uint64_t actual, int64_t expected, const std::string &fieldName,
                            const std::string &caseName)
{
    if (expected >= 0) {
        EXPECT_EQ(actual, static_cast<uint64_t>(expected)) << "field=" << fieldName << ", case=" << caseName;
    }
}

struct BlockAttnResPrepareTilingCase {
    optiling::BlockAttnResPrepareCompileInfo MakeCompileInfo() const
    {
        return {static_cast<uint64_t>(aicCoreNum),
                static_cast<uint64_t>(aivCoreNum),
                ASCEND_950_UB_SIZE,
                ASCEND_950_L1_SIZE,
                ASCEND_950_L0A_SIZE,
                ASCEND_950_L0B_SIZE,
                ASCEND_950_L0C_SIZE,
                SYSTEM_WORKSPACE_SIZE};
    }

    gert::TilingContextPara MakeContext(optiling::BlockAttnResPrepareCompileInfo &compileInfo) const
    {
        const std::vector<int64_t> blockDims = ops::ut::ParseDims(blockResShape);
        const std::vector<int64_t> queryDims = ops::ut::ParseDims(pseudoQueryShape);
        const int64_t totalT = blockDims[T_DIM_INDEX];
        const int64_t totalD = blockDims[D_DIM_INDEX];
        const int64_t totalS = queryDims[S_DIM_INDEX];
        return gert::TilingContextPara(
            "BlockAttnResPrepare",
            {
                {ops::ut::MakeGertStorageShape(blockDims), ops::ut::ParseGeDtype(blockResDtype), ge::FORMAT_ND},
                {ops::ut::MakeGertStorageShape(std::vector<int64_t>{1}), ops::ut::ParseGeDtype(validBlocksDtype),
                 ge::FORMAT_ND},
                {ops::ut::MakeGertStorageShape(queryDims), ops::ut::ParseGeDtype(pseudoQueryDtype), ge::FORMAT_ND},
            },
            {
                {ops::ut::MakeGertStorageShape(std::vector<int64_t>{totalS, totalT, totalD}), ge::DT_FLOAT,
                 ge::FORMAT_ND},
                {ops::ut::MakeGertStorageShape(std::vector<int64_t>{totalS, totalT}), ge::DT_FLOAT, ge::FORMAT_ND},
                {ops::ut::MakeGertStorageShape(std::vector<int64_t>{totalS, totalT}), ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {{"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)}}, &compileInfo, socVersion,
            static_cast<uint64_t>(contextCoreNum), ASCEND_950_UB_SIZE, FAKE_L2_SIZE);
    }

    void CheckVectorTiling(const TilingInfo &tilingInfo) const
    {
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(VectorTilingDataView));
        const auto *tilingData = reinterpret_cast<const VectorTilingDataView *>(tilingInfo.tilingData.get());
        ASSERT_NE(tilingData, nullptr);
        ExpectValueIfSpecified(tilingData->baseD, expectBaseD, "baseD", caseName);
        ExpectValueIfSpecified(tilingData->qBufferNum, expectQBufferNum, "qBufferNum", caseName);
        ExpectValueIfSpecified(tilingData->vBufferNum, expectVBufferNum, "vBufferNum", caseName);
        ExpectValueIfSpecified(tilingData->oBufferNum, expectOBufferNum, "oBufferNum", caseName);
        ExpectValueIfSpecified(tilingData->vCacheRows, expectVCacheRows, "vCacheRows", caseName);
    }

    void CheckMixTiling(const TilingInfo &tilingInfo) const
    {
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MixTilingDataView));
        const auto *tilingData = reinterpret_cast<const MixTilingDataView *>(tilingInfo.tilingData.get());
        ASSERT_NE(tilingData, nullptr);
        ExpectValueIfSpecified(tilingData->baseT, expectBaseT, "baseT", caseName);
        ExpectValueIfSpecified(tilingData->baseS, expectBaseS, "baseS", caseName);
        ExpectValueIfSpecified(tilingData->totalWorkUnits, expectTotalWorkUnits, "totalWorkUnits", caseName);
        ExpectValueIfSpecified(tilingData->nAlign, expectNAlign, "nAlign", caseName);
        ExpectValueIfSpecified(tilingData->mm1NAlign, expectMm1NAlign, "mm1NAlign", caseName);
        if (checkWorkspaceFormula) {
            const uint64_t expectedWorkspaceElems = static_cast<uint64_t>(tilingData->baseS) * tilingData->mm1NAlign +
                                                    std::min<uint64_t>(totalT, MAX_RUNTIME_BASE_T) *
                                                        static_cast<uint64_t>(tilingData->sAlign) * tilingData->nAlign;
            EXPECT_EQ(tilingData->workspacePerCoreElems, expectedWorkspaceElems) << "case=" << caseName;
        }
    }

    void Run() const
    {
        auto compileInfo = MakeCompileInfo();
        auto context = MakeContext(compileInfo);
        TilingInfo tilingInfo;
        const bool actualSuccess = ExecuteTiling(context, tilingInfo);
        ASSERT_EQ(actualSuccess, expectSuccess) << "case=" << caseName;
        if (!expectSuccess) {
            return;
        }
        ASSERT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(expectTilingKey)) << "case=" << caseName;
        if (checkMode == "vector") {
            CheckVectorTiling(tilingInfo);
        } else if (checkMode == "mix") {
            CheckMixTiling(tilingInfo);
        }
    }

    std::string socVersion;
    std::string caseName;
    std::string prefix;
    std::string blockResShape;
    std::string pseudoQueryShape;
    std::string blockResDtype;
    std::string validBlocksDtype;
    std::string pseudoQueryDtype;
    float eps = 1.0e-6F;
    int64_t aicCoreNum = 32;
    int64_t aivCoreNum = 64;
    int64_t contextCoreNum = 64;
    bool expectSuccess = false;
    int64_t expectTilingKey = -1;
    std::string checkMode;
    int64_t expectBaseD = -1;
    int64_t expectQBufferNum = -1;
    int64_t expectVBufferNum = -1;
    int64_t expectOBufferNum = -1;
    int64_t expectVCacheRows = -1;
    int64_t expectBaseT = -1;
    int64_t expectBaseS = -1;
    int64_t expectTotalWorkUnits = -1;
    int64_t expectNAlign = -1;
    int64_t expectMm1NAlign = -1;
    bool checkWorkspaceFormula = false;
};

std::vector<BlockAttnResPrepareTilingCase> LoadCases()
{
    std::vector<BlockAttnResPrepareTilingCase> cases;
    const std::string csvPath = ops::ut::ResolveCsvPath(CSV_FILE_NAME, CSV_REPO_DIR, __FILE__);
    std::ifstream csvData(csvPath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << csvPath << std::endl;
        return cases;
    }

    std::string line;
    size_t lineNo = 0U;
    while (std::getline(csvData, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::vector<std::string> items;
        ops::ut::SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < CSV_COLUMN_COUNT) {
            continue;
        }

        const std::string caseName = ops::ut::Trim(items[1]);
        try {
            size_t index = 0U;
            BlockAttnResPrepareTilingCase testCase;
            testCase.socVersion = ops::ut::Trim(items[index++]);
            testCase.caseName = ops::ut::Trim(items[index++]);
            if (!ops::ut::ParseBool(items[index++]) || testCase.socVersion != "Ascend950") {
                continue;
            }
            testCase.prefix = ops::ut::Trim(items[index++]);
            testCase.blockResShape = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryShape = ops::ut::Trim(items[index++]);
            testCase.blockResDtype = ops::ut::Trim(items[index++]);
            testCase.validBlocksDtype = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryDtype = ops::ut::Trim(items[index++]);
            testCase.eps = std::stof(ops::ut::Trim(items[index++]));
            testCase.aicCoreNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.aivCoreNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.contextCoreNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectSuccess = ops::ut::ParseBool(items[index++]);
            testCase.expectTilingKey = std::stoll(ops::ut::Trim(items[index++]));
            testCase.checkMode = ops::ut::Trim(items[index++]);
            testCase.expectBaseD = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectQBufferNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectVBufferNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectOBufferNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectVCacheRows = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectBaseT = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectBaseS = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectTotalWorkUnits = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectNAlign = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectMm1NAlign = std::stoll(ops::ut::Trim(items[index++]));
            testCase.checkWorkspaceFormula = ops::ut::ParseBool(items[index++]);
            cases.emplace_back(std::move(testCase));
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return cases;
}

std::string MakeParamName(const testing::TestParamInfo<BlockAttnResPrepareTilingCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

const std::vector<BlockAttnResPrepareTilingCase> &GetCases()
{
    static const std::vector<BlockAttnResPrepareTilingCase> cases = LoadCases();
    return cases;
}

class BlockAttnResPrepareTilingTest : public testing::TestWithParam<BlockAttnResPrepareTilingCase> {};

TEST_P(BlockAttnResPrepareTilingTest, CsvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(BLOCK_ATTN_RES_PREPARE_TILING_CSV, BlockAttnResPrepareTilingTest,
                         testing::ValuesIn(GetCases()), MakeParamName);

} // namespace
