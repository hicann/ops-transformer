/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_BLOCK_ATTN_RES_UPDATE_UTILS_H
#define TEST_BLOCK_ATTN_RES_UPDATE_UTILS_H

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

#include "../../../op_kernel/arch35/block_attn_res_update_tiling_data.h"
#include "../../../op_kernel/arch35/block_attn_res_update_tiling_key.h"

struct BlockAttnResUpdateKernelTestParam {
    std::string socVersion;
    std::string caseName;
    std::string kernelUtTarget;
    std::string prefix;
    uint32_t t = 0U;
    uint32_t d = 0U;
    uint32_t seed = 0U;
    bool result = false;
    uint32_t numBlocks = 0U;
    uint64_t tilingKey = 0UL;
    std::vector<int32_t> tilingData;
};

struct BlockAttnResUpdateKernelCsvLoadResult {
    std::vector<BlockAttnResUpdateKernelTestParam> params;
    std::vector<std::string> errors;
};

class BlockAttnResUpdateKernelTestUtils {
public:
    static constexpr uint32_t MAX_NUM_BLOCKS = 4U;

    static BlockAttnResUpdateKernelCsvLoadResult GetParams(const std::string &socVersion, const std::string &testSuite)
    {
        BlockAttnResUpdateKernelCsvLoadResult result;
        const std::string csvPath = ops::ut::ResolveCsvPath(
            "test_block_attn_res_update.csv", "attention/block_attn_res_update/tests/ut/op_kernel", __FILE__);
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
                constexpr size_t REQUIRED_COLUMN_COUNT = 12UL;
                if (fields.size() < REQUIRED_COLUMN_COUNT) {
                    throw std::runtime_error("invalid CSV column count");
                }

                BlockAttnResUpdateKernelTestParam param;
                size_t index = 0UL;
                param.socVersion = ops::ut::Trim(fields[index++]);
                param.caseName = ops::ut::Trim(fields[index++]);
                param.kernelUtTarget = ops::ut::Trim(fields[index++]);
                if (param.socVersion != socVersion || param.kernelUtTarget != testSuite) {
                    continue;
                }

                param.prefix = ops::ut::Trim(fields[index++]);
                ++index; // coreNum is part of the CSV schema; this launcher derives its block count from numBlocks.
                param.t = ParseU32(fields[index++], "t");
                param.d = ParseU32(fields[index++], "d");
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
    static void TestOneParamCase950(const BlockAttnResUpdateKernelTestParam &param)
    {
        ASSERT_TRUE(param.result) << "op_kernel CSV currently supports successful launch cases only";

        const size_t elementCount = static_cast<size_t>(param.t) * param.d;
        const size_t partialBytes = elementCount * sizeof(float);
        const size_t bf16MatrixBytes = elementCount * sizeof(uint16_t);
        const size_t queryBytes = static_cast<size_t>(param.d) * sizeof(float);
        const size_t statsBytes = static_cast<size_t>(param.t) * sizeof(float);

        std::vector<float> partialInput(elementCount);
        std::vector<uint16_t> deltaInput(elementCount);
        std::vector<float> pseudoQueryInput(param.d);
        std::vector<float> numeratorInput(elementCount);
        std::vector<float> logitMaxInput(param.t);
        std::vector<float> expSumInput(param.t);
        InitInputData(param, partialInput, deltaInput, pseudoQueryInput, numeratorInput, logitMaxInput, expSumInput);

        GmBuffer partialGm(partialBytes);
        GmBuffer deltaGm(bf16MatrixBytes);
        GmBuffer pseudoQueryGm(queryBytes);
        GmBuffer numeratorGm(partialBytes);
        GmBuffer logitMaxGm(statsBytes);
        GmBuffer expSumGm(statsBytes);
        GmBuffer hGm(bf16MatrixBytes);
        GmBuffer tilingGm(param.tilingData.size() * sizeof(int32_t));

        ASSERT_TRUE(partialGm.IsValid());
        ASSERT_TRUE(deltaGm.IsValid());
        ASSERT_TRUE(pseudoQueryGm.IsValid());
        ASSERT_TRUE(numeratorGm.IsValid());
        ASSERT_TRUE(logitMaxGm.IsValid());
        ASSERT_TRUE(expSumGm.IsValid());
        ASSERT_TRUE(hGm.IsValid());
        ASSERT_TRUE(tilingGm.IsValid());

        std::memcpy(partialGm.Get(), partialInput.data(), partialBytes);
        std::memcpy(deltaGm.Get(), deltaInput.data(), bf16MatrixBytes);
        std::memcpy(pseudoQueryGm.Get(), pseudoQueryInput.data(), queryBytes);
        std::memcpy(numeratorGm.Get(), numeratorInput.data(), partialBytes);
        std::memcpy(logitMaxGm.Get(), logitMaxInput.data(), statsBytes);
        std::memcpy(expSumGm.Get(), expSumInput.data(), statsBytes);
        std::memset(hGm.Get(), 0xFF, bf16MatrixBytes);
        std::memcpy(tilingGm.Get(), param.tilingData.data(), param.tilingData.size() * sizeof(int32_t));

        const uint32_t launchBlocks = std::min(MAX_NUM_BLOCKS, param.numBlocks);
        const auto kernelFunc =
            param.tilingKey == GET_TPL_TILING_KEY(true) ? &block_attn_res_update<true> : &block_attn_res_update<false>;
        ICPU_SET_TILING_KEY(param.tilingKey);
        // CANN 9.2 tikicpulib does not emulate Ascend950 Reg/VF arithmetic, so this UT validates CSV scheduling and a
        // clean kernel launch rather than numerical results.
        // partial_block_ref and workspace are ABI-only here; nullptr makes any accidental kernel access fail this UT.
        ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, launchBlocks, partialGm.Get(), deltaGm.Get(), pseudoQueryGm.Get(),
                                  numeratorGm.Get(), logitMaxGm.Get(), expSumGm.Get(), nullptr, hGm.Get(), nullptr,
                                  tilingGm.Get()))
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

    static void ValidateParam(const BlockAttnResUpdateKernelTestParam &param)
    {
        static_assert(sizeof(BlockAttnResUpdateTilingData) == 32UL,
                      "BlockAttnResUpdateTilingData ABI changed; update the CSV serialization");
        if (param.t == 0U || param.d == 0U || param.numBlocks == 0U) {
            throw std::runtime_error("t, d and numBlocks must be positive");
        }
        if (!param.result) {
            throw std::runtime_error("op_kernel CSV does not support expected-failure launch cases");
        }
        if (param.tilingData.size() * sizeof(int32_t) != sizeof(BlockAttnResUpdateTilingData)) {
            throw std::runtime_error("tilingData byte size does not match BlockAttnResUpdateTilingData");
        }

        BlockAttnResUpdateTilingData tiling{};
        std::memcpy(&tiling, param.tilingData.data(), sizeof(tiling));
        if (tiling.dSize != param.d) {
            throw std::runtime_error("tilingData dSize does not match CSV d");
        }
        if (tiling.usedCoreNum == 0U || tiling.tPerCore == 0U || tiling.lastTPerCore == 0U || tiling.tileT == 0U ||
            tiling.statsTStride == 0U) {
            throw std::runtime_error("tilingData contains a zero scheduling field");
        }
        const uint32_t expectStatsTStride =
            static_cast<uint32_t>((static_cast<uint64_t>(tiling.tileT) + 7UL) / 8UL * 8UL);
        if (tiling.lastTPerCore > tiling.tPerCore || tiling.tileT > tiling.tPerCore ||
            tiling.statsTStride != expectStatsTStride) {
            throw std::runtime_error("tilingData contains an invalid T scheduling field");
        }
        const uint64_t expectedTilingKey = GET_TPL_TILING_KEY(tiling.tPerCore <= tiling.tileT);
        if (param.tilingKey != expectedTilingKey) {
            throw std::runtime_error("tilingKey does not match the SINGLE_TILE scheduling condition");
        }
        if (!std::isfinite(tiling.eps) || tiling.eps <= 0.0F || tiling.invD != 1.0F / static_cast<float>(param.d)) {
            throw std::runtime_error("tilingData contains an invalid eps or invD value");
        }
        const uint64_t coveredT =
            static_cast<uint64_t>(tiling.usedCoreNum - 1U) * tiling.tPerCore + tiling.lastTPerCore;
        if (coveredT != param.t) {
            throw std::runtime_error("tilingData core partition does not cover CSV t");
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

    static uint16_t FloatToBf16(float value)
    {
        uint32_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t roundBias = 0x7FFFU + ((bits >> 16U) & 1U);
        return static_cast<uint16_t>((bits + roundBias) >> 16U);
    }

    static void InitInputData(const BlockAttnResUpdateKernelTestParam &param, std::vector<float> &partial,
                              std::vector<uint16_t> &delta, std::vector<float> &pseudoQuery,
                              std::vector<float> &numerator, std::vector<float> &logitMax, std::vector<float> &expSum)
    {
        for (uint32_t dIndex = 0U; dIndex < param.d; ++dIndex) {
            const int32_t queryCode = static_cast<int32_t>((dIndex * 5U + param.seed) % 9U) - 4;
            pseudoQuery[dIndex] = static_cast<float>(queryCode) * 0.0625F;
        }
        for (uint32_t tIndex = 0U; tIndex < param.t; ++tIndex) {
            const int32_t maxCode = static_cast<int32_t>((tIndex + param.seed) % 5U) - 2;
            logitMax[tIndex] = static_cast<float>(maxCode) * 0.125F;
            expSum[tIndex] = 0.75F + static_cast<float>((tIndex + param.seed) % 3U) * 0.25F;
            for (uint32_t dIndex = 0U; dIndex < param.d; ++dIndex) {
                const size_t index = static_cast<size_t>(tIndex) * param.d + dIndex;
                const int32_t partialCode = static_cast<int32_t>((index + param.seed) % 7U) - 3;
                const int32_t deltaCode = static_cast<int32_t>((index * 3U + param.seed) % 5U) - 2;
                const int32_t numeratorCode = static_cast<int32_t>((index * 5U + param.seed) % 9U) - 4;
                partial[index] = static_cast<float>(partialCode) * 0.25F;
                delta[index] = FloatToBf16(static_cast<float>(deltaCode) * 0.125F);
                numerator[index] = static_cast<float>(numeratorCode) * 0.1875F;
            }
        }
    }
#endif
};

#endif // TEST_BLOCK_ATTN_RES_UPDATE_UTILS_H
