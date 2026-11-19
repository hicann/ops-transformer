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

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../../../op_host/op_tiling/arch35/block_attn_res_prepare_base_tiling.h"
#include "../../../op_kernel/arch35/block_attn_res_prepare_tiling_data.h"
#include "gmm_csv_ge_parse_utils.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

constexpr uint64_t ASCEND_950_UB_SIZE = 253952UL;
constexpr uint64_t ASCEND_950_L1_SIZE = 512UL * 1024UL;
constexpr uint64_t ASCEND_950_L0A_SIZE = 64UL * 1024UL;
constexpr uint64_t ASCEND_950_L0B_SIZE = 64UL * 1024UL;
constexpr uint64_t ASCEND_950_L0C_SIZE = 256UL * 1024UL;
constexpr uint64_t FAKE_L2_SIZE = 4096UL;
constexpr uint32_t VECTOR_TILING_DATA_SIZE = 48U;
constexpr uint32_t MIX_TILING_DATA_SIZE = 112U;
constexpr uint64_t MAX_RUNTIME_BASE_T = 16U;
constexpr size_t T_DIM_INDEX = 0U;
constexpr size_t D_DIM_INDEX = 2U;
constexpr size_t S_DIM_INDEX = 0U;
constexpr size_t CSV_COLUMN_COUNT = 25U;
constexpr const char* CSV_FILE_NAME = "test_block_attn_res_prepare_tiling.csv";
constexpr const char* CSV_REPO_DIR = "attention/block_attn_res_prepare/tests/ut/op_host";

float ParseEps(const std::string& value)
{
    const float minEps = std::numeric_limits<float>::min();
    if (value == "float_min") {
        return minEps;
    }
    if (value == "float_min_above") {
        return std::nextafter(minEps, std::numeric_limits<float>::infinity());
    }
    if (value == "float_denorm_min") {
        return std::numeric_limits<float>::denorm_min();
    }
    if (value == "float_max_subnormal") {
        return std::nextafter(minEps, 0.0F);
    }
    if (value == "float_max") {
        return std::numeric_limits<float>::max();
    }
    return std::stof(value);
}

#pragma pack(push, 8)
struct alignas(8) VectorTilingDataView {
    uint32_t totalT;
    uint32_t totalS;
    uint32_t sBlockSize;
    float eps;
    uint32_t sBlockCount;
    uint32_t workPerCore;
    float reciprocalD;

    uint32_t totalD;
    uint32_t dAlign;

    // Host validates core counts within uint16_t; remainder < usedCoreNum.
    uint16_t usedCoreNum;
    uint16_t remainder;
    uint16_t qCacheCoresPerGroup;

    // N <= 64, and the V-cache row count never exceeds N.
    uint8_t totalN;
    uint8_t vCacheRows;
    uint8_t qBufferNum; // Initial Q/O slots before grouping S rows for V reuse.
    uint8_t threeSMaxN;
};

struct alignas(8) MixTilingDataView {
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
    float eps;
    uint16_t usedCoreNum;
    uint8_t totalN;
    uint8_t qL1BufferNum;
    uint8_t vL1BufferNum;
    uint8_t vUbBufferNum;
};
#pragma pack(pop)

static_assert(sizeof(VectorTilingDataView) == VECTOR_TILING_DATA_SIZE);
static_assert(sizeof(MixTilingDataView) == MIX_TILING_DATA_SIZE);

void ExpectValueIfSpecified(uint64_t actual, int64_t expected, const std::string& fieldName,
                            const std::string& caseName)
{
    if (expected >= 0) {
        EXPECT_EQ(actual, static_cast<uint64_t>(expected)) << "field=" << fieldName << ", case=" << caseName;
    }
}

struct BlockAttnResPrepareTilingCase {
    std::string MakePlatformInfoString() const
    {
        return std::string("{\"hardware_info\":{") + "\"BT_SIZE\":0," + "\"load3d_constraints\":\"1\"," +
               "\"Intrinsic_fix_pipe_l0c2out\":false," + "\"Intrinsic_data_move_l12ub\":true," +
               "\"Intrinsic_data_move_l0c2ub\":true," + "\"Intrinsic_data_move_out2l1_nd2nz\":false," +
               "\"UB_SIZE\":" + std::to_string(ASCEND_950_UB_SIZE) + "," + "\"L2_SIZE\":33554432," +
               "\"L1_SIZE\":" + std::to_string(ASCEND_950_L1_SIZE) + "," +
               "\"L0A_SIZE\":" + std::to_string(ASCEND_950_L0A_SIZE) + "," +
               "\"L0B_SIZE\":" + std::to_string(ASCEND_950_L0B_SIZE) + "," +
               "\"L0C_SIZE\":" + std::to_string(ASCEND_950_L0C_SIZE) + "," +
               "\"CORE_NUM\":" + std::to_string(contextCoreNum) + "," +
               "\"cube_core_cnt\":" + std::to_string(aicCoreNum) + "," +
               "\"vector_core_cnt\":" + std::to_string(aivCoreNum) + "," + "\"socVersion\":\"" + platformName + "\"}}";
    }

    gert::TilingContextPara MakeContext(optiling::BlockAttnResPrepareCompileInfo& compileInfo) const
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
            {{"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)}}, &compileInfo, npuArch,
            static_cast<uint64_t>(contextCoreNum), ASCEND_950_UB_SIZE, FAKE_L2_SIZE, MakePlatformInfoString());
    }

    void CheckVectorTiling(const TilingInfo& tilingInfo) const
    {
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(VectorTilingDataView));
        const auto* tilingData = reinterpret_cast<const VectorTilingDataView*>(tilingInfo.tilingData.get());
        ASSERT_NE(tilingData, nullptr);
        ExpectValueIfSpecified(tilingData->sBlockSize, expectSSize, "sBlockSize", caseName);
        ASSERT_GE(tilingData->sBlockSize, 1U);
        ASSERT_LE(tilingData->sBlockSize, tilingData->totalS);
        const uint64_t sBlockCount =
            (static_cast<uint64_t>(tilingData->totalS) + tilingData->sBlockSize - 1U) / tilingData->sBlockSize;
        const uint64_t totalWorkUnits = static_cast<uint64_t>(tilingData->totalT) * sBlockCount;
        ExpectValueIfSpecified(totalWorkUnits, expectTotalWorkUnits, "totalWorkUnits", caseName);
        EXPECT_EQ(tilingData->usedCoreNum, std::min<uint64_t>(aivCoreNum, totalWorkUnits));
        ExpectValueIfSpecified(tilingData->vCacheRows, expectVCacheRows, "vCacheRows", caseName);
        ExpectValueIfSpecified(tilingData->threeSMaxN, expectThreeSMaxN, "threeSMaxN", caseName);
        ExpectValueIfSpecified(tilingData->qCacheCoresPerGroup, expectQCacheCoresPerGroup, "qCacheCoresPerGroup",
                               caseName);
        if (tilingData->qCacheCoresPerGroup != 0U) {
            const uint64_t coresPerGroup = tilingData->qCacheCoresPerGroup;
            ASSERT_EQ(ops::ut::ParseGeDtype(blockResDtype), ge::DT_BF16);
            ASSERT_EQ(tilingData->totalD % 64U, 0U);
            ASSERT_EQ(tilingData->totalS % 3U, 0U);
            EXPECT_EQ(coresPerGroup * (tilingData->totalS / 3U), tilingData->usedCoreNum);
            const uint64_t baseTokens = tilingData->totalT / coresPerGroup;
            const uint64_t extraTokens = tilingData->totalT % coresPerGroup;
            ASSERT_GT(baseTokens, 0U);
            uint64_t nextToken = 0U;
            for (uint64_t core = 0U; core < coresPerGroup; ++core) {
                const uint64_t firstToken = core * baseTokens + std::min(core, extraTokens);
                EXPECT_EQ(firstToken, nextToken);
                nextToken += baseTokens + (core < extraTokens ? 1U : 0U);
            }
            EXPECT_EQ(nextToken, tilingData->totalT);
            // Three Q rows, three FP32 outputs, two BF16 V rows and three stat slots.
            EXPECT_LE(28UL * tilingData->totalD + 576UL, ASCEND_950_UB_SIZE);
        }
        ASSERT_GT(tilingData->usedCoreNum, 0U);
        EXPECT_EQ(tilingData->sBlockCount, sBlockCount);
        EXPECT_EQ(tilingData->workPerCore, totalWorkUnits / tilingData->usedCoreNum);
        EXPECT_EQ(tilingData->remainder, totalWorkUnits % tilingData->usedCoreNum);
        EXPECT_FLOAT_EQ(tilingData->reciprocalD, 1.0F / static_cast<float>(tilingData->totalD));
        const uint32_t rowAlign = ops::ut::ParseGeDtype(blockResDtype) == ge::DT_BF16 ? 16U : 8U;
        const uint64_t dAlign = (static_cast<uint64_t>(tilingData->totalD) + rowAlign - 1U) / rowAlign * rowAlign;
        EXPECT_EQ(tilingData->dAlign, dAlign);
        EXPECT_EQ(tilingData->qBufferNum,
                  tilingData->sBlockSize > 1U || totalWorkUnits > tilingData->usedCoreNum ? 2U : 1U);
    }

    void CheckMixTiling(const TilingInfo& tilingInfo) const
    {
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MixTilingDataView));
        const auto* tilingData = reinterpret_cast<const MixTilingDataView*>(tilingInfo.tilingData.get());
        ASSERT_NE(tilingData, nullptr);
        ExpectValueIfSpecified(tilingData->baseT, expectBaseT, "baseT", caseName);
        ExpectValueIfSpecified(tilingData->baseS, expectSSize, "baseS", caseName);
        ExpectValueIfSpecified(tilingData->totalWorkUnits, expectTotalWorkUnits, "totalWorkUnits", caseName);
        ExpectValueIfSpecified(tilingData->nAlign, expectNAlign, "nAlign", caseName);
        ExpectValueIfSpecified(tilingData->mm1NAlign, expectMm1NAlign, "mm1NAlign", caseName);
        const uint64_t expectedUsedCoreNum = std::min<uint64_t>(aicCoreNum, tilingData->totalWorkUnits);
        EXPECT_EQ(tilingData->usedCoreNum, expectedUsedCoreNum) << "case=" << caseName;
        EXPECT_EQ(tilingInfo.blockNum, expectedUsedCoreNum) << "case=" << caseName;
        if (checkWorkspaceFormula) {
            const uint64_t expectedWorkspaceElems = static_cast<uint64_t>(tilingData->baseS) * tilingData->mm1NAlign +
                                                    std::min<uint64_t>(tilingData->totalT, MAX_RUNTIME_BASE_T) *
                                                        static_cast<uint64_t>(tilingData->sAlign) * tilingData->nAlign;
            EXPECT_EQ(tilingData->workspacePerCoreElems, expectedWorkspaceElems) << "case=" << caseName;
        }
    }

    void Run() const
    {
        optiling::BlockAttnResPrepareCompileInfo compileInfo{};
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

    std::string platformName;
    std::string npuArch;
    std::string caseName;
    std::string prefix;
    std::string blockResShape;
    std::string pseudoQueryShape;
    std::string blockResDtype;
    std::string validBlocksDtype;
    std::string pseudoQueryDtype;
    std::string checkMode;

    int64_t aicCoreNum = 32;
    int64_t aivCoreNum = 64;
    int64_t contextCoreNum = 64;
    int64_t expectTilingKey = -1;
    int64_t expectVCacheRows = -1;
    int64_t expectBaseT = -1;
    int64_t expectSSize = -1; // Vector: sBlockSize; Mix: baseS.
    int64_t expectTotalWorkUnits = -1;
    int64_t expectNAlign = -1;
    int64_t expectMm1NAlign = -1;
    int64_t expectThreeSMaxN = -1;
    int64_t expectQCacheCoresPerGroup = -1;

    float eps = 1.0e-6F;
    bool expectSuccess = false;
    bool checkWorkspaceFormula = false;
};

std::vector<BlockAttnResPrepareTilingCase> LoadCases()
{
    static const std::map<std::string, std::string> npuArchMap = {
        {"Ascend950", "3510"},
        {"Ascend960DT", "9201"},
    };
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
            testCase.platformName = ops::ut::Trim(items[index++]);
            testCase.caseName = ops::ut::Trim(items[index++]);
            const auto archIterator = npuArchMap.find(testCase.platformName);
            if (!ops::ut::ParseBool(items[index++]) || archIterator == npuArchMap.end()) {
                continue;
            }
            testCase.npuArch = archIterator->second;
            testCase.prefix = ops::ut::Trim(items[index++]);
            testCase.blockResShape = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryShape = ops::ut::Trim(items[index++]);
            testCase.blockResDtype = ops::ut::Trim(items[index++]);
            testCase.validBlocksDtype = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryDtype = ops::ut::Trim(items[index++]);
            testCase.eps = ParseEps(ops::ut::Trim(items[index++]));
            testCase.aicCoreNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.aivCoreNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.contextCoreNum = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectSuccess = ops::ut::ParseBool(items[index++]);
            testCase.expectTilingKey = std::stoll(ops::ut::Trim(items[index++]));
            testCase.checkMode = ops::ut::Trim(items[index++]);
            testCase.expectVCacheRows = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectBaseT = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectSSize = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectTotalWorkUnits = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectNAlign = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectMm1NAlign = std::stoll(ops::ut::Trim(items[index++]));
            testCase.checkWorkspaceFormula = ops::ut::ParseBool(items[index++]);
            testCase.expectThreeSMaxN = std::stoll(ops::ut::Trim(items[index++]));
            testCase.expectQCacheCoresPerGroup = std::stoll(ops::ut::Trim(items[index++]));
            cases.emplace_back(std::move(testCase));
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return cases;
}

std::string MakeParamName(const testing::TestParamInfo<BlockAttnResPrepareTilingCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.platformName + "_" + info.param.prefix);
}

const std::vector<BlockAttnResPrepareTilingCase>& GetCases()
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
