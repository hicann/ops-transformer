/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_BLOCK_ATTN_RES_PREPARE_UTILS_H
#define TEST_BLOCK_ATTN_RES_PREPARE_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gmm_csv_parse_utils.h"
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/arch35/block_attn_res_prepare_apt_tiling_key.h"
#include "../../../op_kernel/arch35/block_attn_res_prepare_tiling_data.h"

struct BlockAttnResPrepareKernelTestParam {
    std::string socVersion;
    std::string caseName;
    std::string kernelUtTarget;
    std::string prefix;
    uint32_t t = 0U;
    uint32_t n = 0U;
    uint32_t s = 0U;
    uint32_t d = 0U;
    uint64_t validBlocks = 0U;
    uint32_t seed = 0U;
    bool result = false;
    uint32_t numBlocks = 0U;
    uint64_t tilingKey = 0UL;
    std::vector<int32_t> tilingData;
};

struct BlockAttnResPrepareKernelCsvLoadResult {
    std::vector<BlockAttnResPrepareKernelTestParam> params;
    std::vector<std::string> errors;
};

class BlockAttnResPrepareKernelTestUtils {
public:
    static constexpr uint32_t MAX_NUM_BLOCKS = 4U;

    static BlockAttnResPrepareKernelCsvLoadResult GetParams(const std::string &socVersion, const std::string &testSuite)
    {
        BlockAttnResPrepareKernelCsvLoadResult result;
        const std::string csvPath = ops::ut::ResolveCsvPath(
            "test_block_attn_res_prepare.csv", "attention/block_attn_res_prepare/tests/ut/op_kernel", __FILE__);
        std::ifstream csvData(csvPath, std::ios::in);
        if (!csvData.is_open()) {
            result.errors.emplace_back("Cannot open kernel case file: " + csvPath);
            return result;
        }

        std::string line;
        size_t lineNo = 0UL;
        while (std::getline(csvData, line)) {
            ++lineNo;
            if (ops::ut::Trim(line).empty()) {
                continue;
            }

            std::vector<std::string> fields;
            ops::ut::SplitStr2Vec(line, ",", fields);
            if (fields.empty() || ops::ut::Trim(fields[0]) == "socVersion") {
                continue;
            }

            const std::string caseName = fields.size() > 1UL ? ops::ut::Trim(fields[1]) : "";
            try {
                constexpr size_t REQUIRED_COLUMN_COUNT = 15UL;
                if (fields.size() < REQUIRED_COLUMN_COUNT) {
                    throw std::runtime_error("invalid CSV column count");
                }

                BlockAttnResPrepareKernelTestParam param;
                size_t index = 0UL;
                param.socVersion = ops::ut::Trim(fields[index++]);
                param.caseName = ops::ut::Trim(fields[index++]);
                param.kernelUtTarget = ops::ut::Trim(fields[index++]);
                if (param.socVersion != socVersion || param.kernelUtTarget != testSuite) {
                    continue;
                }

                param.prefix = ops::ut::Trim(fields[index++]);
                ++index;
                param.t = ParseU32(fields[index++], "t");
                param.n = ParseU32(fields[index++], "n");
                param.s = ParseU32(fields[index++], "s");
                param.d = ParseU32(fields[index++], "d");
                param.validBlocks = ParseU64(fields[index++], "validBlocks");
                param.seed = ParseU32(fields[index++], "seed");
                param.result = ops::ut::ParseBool(fields[index++]);
                param.numBlocks = ParseU32(fields[index++], "numBlocks");
                param.tilingKey = std::stoull(ops::ut::Trim(fields[index++]));
                param.tilingData = ParseTilingData(fields[index++]);
                ValidateParam(param);
                result.params.emplace_back(std::move(param));
            } catch (const std::exception &error) {
                result.errors.emplace_back(ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error));
            }
        }
        if (result.params.empty()) {
            result.errors.emplace_back("No matching kernel cases were loaded from: " + csvPath);
        }
        return result;
    }

#ifdef __CCE_KT_TEST__
    static void TestOneParamCase950(const BlockAttnResPrepareKernelTestParam &param)
    {
        ASSERT_TRUE(param.result) << "op_kernel CSV currently supports successful launch cases only";

        const size_t blockResElements = static_cast<size_t>(param.t) * param.n * param.d;
        const size_t pseudoQueryElements = static_cast<size_t>(param.s) * param.d;
        const size_t numeratorElements = static_cast<size_t>(param.s) * param.t * param.d;
        const size_t statsElements = static_cast<size_t>(param.s) * param.t;
        std::vector<float> blockResInput(blockResElements);
        std::vector<float> pseudoQueryInput(pseudoQueryElements);
        InitInputData(param, blockResInput, pseudoQueryInput);

        GmBuffer blockResGm(blockResElements * sizeof(float));
        GmBuffer validBlocksGm(sizeof(uint64_t));
        GmBuffer pseudoQueryGm(pseudoQueryElements * sizeof(float));
        GmBuffer numeratorGm(numeratorElements * sizeof(float));
        GmBuffer logitMaxGm(statsElements * sizeof(float));
        GmBuffer expSumGm(statsElements * sizeof(float));
        GmBuffer tilingGm(param.tilingData.size() * sizeof(int32_t));

        ASSERT_TRUE(blockResGm.IsValid());
        ASSERT_TRUE(validBlocksGm.IsValid());
        ASSERT_TRUE(pseudoQueryGm.IsValid());
        ASSERT_TRUE(numeratorGm.IsValid());
        ASSERT_TRUE(logitMaxGm.IsValid());
        ASSERT_TRUE(expSumGm.IsValid());
        ASSERT_TRUE(tilingGm.IsValid());

        std::memcpy(blockResGm.Get(), blockResInput.data(), blockResElements * sizeof(float));
        std::memcpy(validBlocksGm.Get(), &param.validBlocks, sizeof(param.validBlocks));
        std::memcpy(pseudoQueryGm.Get(), pseudoQueryInput.data(), pseudoQueryElements * sizeof(float));
        std::memset(numeratorGm.Get(), 0xFF, numeratorElements * sizeof(float));
        std::memset(logitMaxGm.Get(), 0xFF, statsElements * sizeof(float));
        std::memset(expSumGm.Get(), 0xFF, statsElements * sizeof(float));
        std::memcpy(tilingGm.Get(), param.tilingData.data(), param.tilingData.size() * sizeof(int32_t));

        const uint32_t launchBlocks = std::min(MAX_NUM_BLOCKS, param.numBlocks);
        const auto kernelFunc = &block_attn_res_prepare<BLOCK_ATTN_RES_PREPARE_TPL_ONLY_VECTOR>;
        ICPU_SET_TILING_KEY(param.tilingKey);
        ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, launchBlocks, blockResGm.Get(), validBlocksGm.Get(), pseudoQueryGm.Get(),
                                  numeratorGm.Get(), logitMaxGm.Get(), expSumGm.Get(), nullptr, tilingGm.Get()))
            << "case=" << param.caseName << ", kernel CPU simulation failed";
    }
#endif

private:
    static uint32_t ParseU32(const std::string &value, const char *fieldName)
    {
        const uint64_t parsed = std::stoull(ops::ut::Trim(value));
        if (parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range(std::string(fieldName) + " exceeds uint32_t range");
        }
        return static_cast<uint32_t>(parsed);
    }

    static uint64_t ParseU64(const std::string &value, const char *fieldName)
    {
        try {
            const std::string trimmed = ops::ut::Trim(value);
            if (trimmed.empty() || trimmed.front() == '-') {
                throw std::invalid_argument("invalid unsigned value");
            }
            size_t parsedLength = 0U;
            const uint64_t result = std::stoull(trimmed, &parsedLength);
            if (parsedLength != trimmed.size()) {
                throw std::invalid_argument("invalid trailing characters");
            }
            return result;
        } catch (const std::exception &) {
            throw std::invalid_argument(std::string(fieldName) + " is not a valid uint64 value");
        }
    }

    static std::vector<int32_t> ParseTilingData(const std::string &value)
    {
        std::istringstream input(ops::ut::Trim(value));
        std::vector<int32_t> words;
        int64_t word = 0;
        while (input >> word) {
            if (word < std::numeric_limits<int32_t>::min() || word > std::numeric_limits<int32_t>::max()) {
                throw std::out_of_range("tilingData word exceeds int32_t range");
            }
            words.emplace_back(static_cast<int32_t>(word));
        }
        if (words.empty()) {
            throw std::runtime_error("tilingData is empty");
        }
        return words;
    }

    static void ValidateParam(const BlockAttnResPrepareKernelTestParam &param)
    {
        static_assert(sizeof(optiling::BlockAttnResPrepareTilingData) == 48UL,
                      "BlockAttnResPrepareTilingData ABI changed; update the CSV serialization");
        if (param.t == 0U || param.n == 0U || param.s == 0U || param.d == 0U || param.numBlocks == 0U) {
            throw std::runtime_error("t, n, s, d and numBlocks must be positive");
        }
        if (!param.result) {
            throw std::runtime_error("op_kernel CSV does not support expected-failure launch cases");
        }
        if (param.tilingData.size() * sizeof(int32_t) != sizeof(optiling::BlockAttnResPrepareTilingData)) {
            throw std::runtime_error("tilingData byte size does not match BlockAttnResPrepareTilingData");
        }

        optiling::BlockAttnResPrepareTilingData tiling{};
        std::memcpy(&tiling, param.tilingData.data(), sizeof(tiling));
        const uint64_t expectedWorkUnits = static_cast<uint64_t>(param.t) * param.s;
        if (tiling.totalT != param.t || tiling.totalN != param.n || tiling.totalS != param.s ||
            tiling.totalD != param.d || tiling.totalWorkUnits != expectedWorkUnits) {
            throw std::runtime_error("tilingData dimensions do not match the CSV case");
        }
        using BlockAttnResPrepareTilingKey::g_tilingDeclareParams;
        if (param.tilingKey != GET_TPL_TILING_KEY(BLOCK_ATTN_RES_PREPARE_TPL_ONLY_VECTOR)) {
            throw std::runtime_error("kernel UT currently supports the Vector tiling key only");
        }
        if (tiling.usedCoreNum == 0U || tiling.usedCoreNum > param.numBlocks || tiling.blockFactor == 0U ||
            tiling.tailBlockFactor == 0U || tiling.baseD == 0U || tiling.baseD > tiling.totalD ||
            tiling.statUbElems == 0U || tiling.qBufferNum == 0U || tiling.vBufferNum == 0U || tiling.oBufferNum == 0U ||
            !std::isfinite(tiling.eps) || tiling.eps <= 0.0F) {
            throw std::runtime_error("tilingData contains an invalid scheduling or buffer field");
        }
        const uint32_t expectedTail = tiling.blockFactor + (tiling.bigCoreNum > 0U ? 1U : 0U);
        const uint64_t coveredWork = static_cast<uint64_t>(tiling.usedCoreNum) * tiling.blockFactor + tiling.bigCoreNum;
        if (tiling.bigCoreNum >= tiling.usedCoreNum || tiling.tailBlockFactor != expectedTail ||
            coveredWork != expectedWorkUnits || tiling.vCacheRows > tiling.totalN) {
            throw std::runtime_error("tilingData work distribution does not cover the CSV case");
        }
        const uint32_t launchBlocks = std::min(MAX_NUM_BLOCKS, param.numBlocks);
        if (tiling.usedCoreNum > launchBlocks) {
            throw std::runtime_error("numBlocks does not launch every used core");
        }
    }

#ifdef __CCE_KT_TEST__
    class GmBuffer {
    public:
        explicit GmBuffer(size_t size)
            : data_(static_cast<uint8_t *>(AscendC::GmAlloc(size)))
        {}
        ~GmBuffer()
        {
            if (data_ != nullptr) {
                AscendC::GmFree(data_);
            }
        }

        GmBuffer(const GmBuffer &) = delete;
        GmBuffer &operator=(const GmBuffer &) = delete;

        uint8_t *Get() const
        {
            return data_;
        }
        bool IsValid() const
        {
            return data_ != nullptr;
        }

    private:
        uint8_t *data_ = nullptr;
    };

    static void InitInputData(const BlockAttnResPrepareKernelTestParam &param, std::vector<float> &blockRes,
                              std::vector<float> &pseudoQuery)
    {
        for (size_t index = 0UL; index < blockRes.size(); ++index) {
            const int32_t value = static_cast<int32_t>((index * 3UL + param.seed) % 11UL) - 5;
            blockRes[index] = static_cast<float>(value) * 0.0625F;
        }
        for (size_t index = 0UL; index < pseudoQuery.size(); ++index) {
            const int32_t value = static_cast<int32_t>((index * 5UL + param.seed) % 9UL) - 4;
            pseudoQuery[index] = static_cast<float>(value) * 0.125F;
        }
    }
#endif
};

#endif // TEST_BLOCK_ATTN_RES_PREPARE_UTILS_H
