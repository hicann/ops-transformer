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
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmm_csv_parse_utils.h"
#include "gtest/gtest.h"
#include "securec.h"

#ifdef __CCE_KT_TEST__
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/arch35/block_attn_res_prepare_apt_tiling_key.h"
#include "../../../op_kernel/arch35/block_attn_res_prepare_tiling_data.h"

struct BlockAttnResPrepareKernelTestParam {
    std::string npuArch;
    std::string caseName;
    std::string kernelUtTarget;
    std::string prefix;
    std::vector<int32_t> tilingData;
    uint64_t validBlocks = 0U;
    uint64_t tilingKey = 0UL;

    uint32_t t = 0U;
    uint32_t n = 0U;
    uint32_t s = 0U;
    uint32_t d = 0U;
    uint32_t seed = 0U;
    uint32_t numBlocks = 0U;

    bool result = false;
    bool bf16 = false;
};

struct BlockAttnResPrepareKernelCsvLoadResult {
    std::vector<BlockAttnResPrepareKernelTestParam> params;
    std::vector<std::string> errors;
};

class BlockAttnResPrepareKernelTestUtils {
public:
    static constexpr uint32_t MAX_NUM_BLOCKS = 4U;

    static BlockAttnResPrepareKernelCsvLoadResult GetParams(const std::string& npuArch, const std::string& testSuite)
    {
        static const std::map<std::string, std::string> npuArchMap = {
            {"Ascend950", "3510"},
            {"Ascend960DT", "9201"},
        };
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
                constexpr size_t requiredColumnCount = 15UL;
                constexpr char bf16Dtype[] = "BF16";
                if (fields.size() < requiredColumnCount) {
                    throw std::runtime_error("invalid CSV column count");
                }

                BlockAttnResPrepareKernelTestParam param;
                size_t index = 0UL;
                const std::string platformName = ops::ut::Trim(fields[index++]);
                param.caseName = ops::ut::Trim(fields[index++]);
                param.kernelUtTarget = ops::ut::Trim(fields[index++]);
                const auto archIterator = npuArchMap.find(platformName);
                if (archIterator == npuArchMap.end() || archIterator->second != npuArch ||
                    param.kernelUtTarget != testSuite) {
                    continue;
                }
                param.npuArch = archIterator->second;

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
                if (index < fields.size() && ops::ut::Trim(fields[index]) == bf16Dtype) {
                    param.bf16 = true;
                }
                ValidateParam(param);
                result.params.emplace_back(std::move(param));
            } catch (const std::exception& error) {
                result.errors.emplace_back(ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error));
            }
        }
        if (result.params.empty()) {
            result.errors.emplace_back("No matching kernel cases were loaded from: " + csvPath);
        }
        return result;
    }

    static uint16_t FloatToBf16(float value)
    {
        constexpr uint32_t fp32ToBf16Shift = 16U;
        constexpr uint32_t fp32ExponentMask = 0x7F800000U;
        constexpr uint32_t fp32MantissaMask = 0x007FFFFFU;
        constexpr uint16_t bf16QuietNanMask = 0x0040U;
        constexpr uint32_t bf16RneRoundBias = 0x7FFFU;
        uint32_t bits = 0U;
        if (memcpy_s(&bits, sizeof(bits), &value, sizeof(value)) != EOK) {
            throw std::runtime_error("failed to copy FP32 bits");
        }
        if ((bits & fp32ExponentMask) == fp32ExponentMask && (bits & fp32MantissaMask) != 0U) {
            // Keep NaNs out of integer rounding, including payloads confined to the discarded low bits.
            return static_cast<uint16_t>((bits >> fp32ToBf16Shift) | bf16QuietNanMask);
        }
        // Round finite values to nearest even; signed zeros and infinities remain unchanged.
        const uint32_t roundingBias = bf16RneRoundBias + ((bits >> fp32ToBf16Shift) & 1U);
        return static_cast<uint16_t>((bits + roundingBias) >> fp32ToBf16Shift);
    }

#ifdef __CCE_KT_TEST__
    template <typename InputStorageT, typename KernelFunc>
    static void TestOneParamCase950Impl(const BlockAttnResPrepareKernelTestParam& param, KernelFunc kernelFunc)
    {
        ASSERT_TRUE(param.result) << "op_kernel CSV currently supports successful launch cases only";

        const size_t blockResElements = static_cast<size_t>(param.t) * param.n * param.d;
        const size_t pseudoQueryElements = static_cast<size_t>(param.s) * param.d;
        const size_t numeratorElements = static_cast<size_t>(param.s) * param.t * param.d;
        const size_t statsElements = static_cast<size_t>(param.s) * param.t;
        std::vector<float> blockResInput(blockResElements);
        std::vector<float> pseudoQueryInput(pseudoQueryElements);
        InitInputData(param, blockResInput, pseudoQueryInput);

        GmBuffer blockResGm(blockResElements * sizeof(InputStorageT));
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

        if constexpr (std::is_same<InputStorageT, uint16_t>::value) {
            std::vector<uint16_t> blockResBf16(blockResElements);
            for (size_t index = 0UL; index < blockResElements; ++index) {
                blockResBf16[index] = FloatToBf16(blockResInput[index]);
            }
            ASSERT_EQ(memcpy_s(blockResGm.Get(), blockResElements * sizeof(InputStorageT), blockResBf16.data(),
                               blockResElements * sizeof(uint16_t)),
                      EOK);
        } else {
            ASSERT_EQ(memcpy_s(blockResGm.Get(), blockResElements * sizeof(InputStorageT), blockResInput.data(),
                               blockResElements * sizeof(float)),
                      EOK);
        }
        ASSERT_EQ(memcpy_s(pseudoQueryGm.Get(), pseudoQueryElements * sizeof(float), pseudoQueryInput.data(),
                           pseudoQueryElements * sizeof(float)),
                  EOK);
        ASSERT_EQ(
            memcpy_s(validBlocksGm.Get(), sizeof(param.validBlocks), &param.validBlocks, sizeof(param.validBlocks)),
            EOK);
        ASSERT_EQ(
            memset_s(numeratorGm.Get(), numeratorElements * sizeof(float), 0xFF, numeratorElements * sizeof(float)),
            EOK);
        ASSERT_EQ(memset_s(logitMaxGm.Get(), statsElements * sizeof(float), 0xFF, statsElements * sizeof(float)), EOK);
        ASSERT_EQ(memset_s(expSumGm.Get(), statsElements * sizeof(float), 0xFF, statsElements * sizeof(float)), EOK);
        ASSERT_EQ(memcpy_s(tilingGm.Get(), param.tilingData.size() * sizeof(int32_t), param.tilingData.data(),
                           param.tilingData.size() * sizeof(int32_t)),
                  EOK);

        const uint32_t launchBlocks = std::min(MAX_NUM_BLOCKS, param.numBlocks);
        ICPU_SET_TILING_KEY(param.tilingKey);
        ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, launchBlocks, blockResGm.Get(), validBlocksGm.Get(), pseudoQueryGm.Get(),
                                  numeratorGm.Get(), logitMaxGm.Get(), expSumGm.Get(), nullptr, tilingGm.Get()))
            << "case=" << param.caseName << ", kernel CPU simulation failed";
    }
#endif

private:
    static uint32_t ParseU32(const std::string& value, const char* fieldName)
    {
        const uint64_t parsed = std::stoull(ops::ut::Trim(value));
        if (parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::out_of_range(std::string(fieldName) + " exceeds uint32_t range");
        }
        return static_cast<uint32_t>(parsed);
    }

    static uint64_t ParseU64(const std::string& value, const char* fieldName)
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
        } catch (const std::exception&) {
            throw std::invalid_argument(std::string(fieldName) + " is not a valid uint64 value");
        }
    }

    static std::vector<int32_t> ParseTilingData(const std::string& value)
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

    static void ValidateParam(const BlockAttnResPrepareKernelTestParam& param)
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
        if (memcpy_s(&tiling, sizeof(tiling), param.tilingData.data(), sizeof(tiling)) != EOK) {
            throw std::runtime_error("failed to copy tilingData");
        }
        if (tiling.sBlockSize == 0U || tiling.sBlockSize > param.s) {
            throw std::runtime_error("sBlockSize must be within the S dimension");
        }
        const uint64_t expectedSBlockCount =
            (static_cast<uint64_t>(param.s) + tiling.sBlockSize - 1U) / tiling.sBlockSize;
        const uint64_t expectedWorkUnits = static_cast<uint64_t>(param.t) * expectedSBlockCount;
        if (tiling.totalT != param.t || tiling.totalN != param.n || tiling.totalS != param.s ||
            tiling.totalD != param.d) {
            throw std::runtime_error("tilingData dimensions do not match the CSV case");
        }
        if (expectedWorkUnits > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("derived work count exceeds uint32_t range");
        }
        using BlockAttnResPrepareTilingKey::g_tilingDeclareParams;
        const uint64_t vectorKey = GET_TPL_TILING_KEY(BLOCK_ATTN_RES_PREPARE_ONLY_VECTOR_TPL);
        if (param.tilingKey != vectorKey) {
            throw std::runtime_error("kernel UT currently supports the shared Vector tiling key only");
        }
        if (tiling.usedCoreNum == 0U || tiling.usedCoreNum > param.numBlocks ||
            tiling.usedCoreNum > expectedWorkUnits || tiling.vCacheRows > tiling.totalN || !std::isfinite(tiling.eps) ||
            tiling.eps <= 0.0F) {
            throw std::runtime_error("tilingData contains an invalid scheduling or buffer field");
        }
        const uint32_t rowAlign = param.bf16 ? 16U : 8U;
        const uint64_t expectedDAlign = (static_cast<uint64_t>(param.d) + rowAlign - 1U) / rowAlign * rowAlign;
        const uint32_t expectedQBufferNum = tiling.sBlockSize > 1U || expectedWorkUnits > tiling.usedCoreNum ? 2U : 1U;
        if (tiling.sBlockCount != expectedSBlockCount || tiling.workPerCore != expectedWorkUnits / tiling.usedCoreNum ||
            tiling.remainder != expectedWorkUnits % tiling.usedCoreNum || tiling.dAlign != expectedDAlign ||
            tiling.qBufferNum != expectedQBufferNum || tiling.reciprocalD != 1.0F / static_cast<float>(param.d)) {
            throw std::runtime_error("tilingData contains inconsistent precomputed fields");
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
            : data_(static_cast<uint8_t*>(AscendC::GmAlloc(size)))
        {}
        ~GmBuffer()
        {
            if (data_ != nullptr) {
                AscendC::GmFree(data_);
            }
        }

        GmBuffer(const GmBuffer&) = delete;
        GmBuffer& operator=(const GmBuffer&) = delete;

        uint8_t* Get() const
        {
            return data_;
        }
        bool IsValid() const
        {
            return data_ != nullptr;
        }

    private:
        uint8_t* data_ = nullptr;
    };

    static void InitInputData(const BlockAttnResPrepareKernelTestParam& param, std::vector<float>& blockRes,
                              std::vector<float>& pseudoQuery)
    {
        // Perturb binary-exact query values to exercise FP32 Q precision without BF16 pre-rounding.
        constexpr float pseudoQueryFp32Offset = 0.000123F;
        for (size_t index = 0UL; index < blockRes.size(); ++index) {
            const int32_t value = static_cast<int32_t>((index * 3UL + param.seed) % 11UL) - 5;
            blockRes[index] = static_cast<float>(value) * 0.0625F;
        }
        for (size_t index = 0UL; index < pseudoQuery.size(); ++index) {
            const int32_t value = static_cast<int32_t>((index * 5UL + param.seed) % 9UL) - 4;
            pseudoQuery[index] = static_cast<float>(value) * 0.125F + pseudoQueryFp32Offset;
        }
    }

#endif
};

#endif // TEST_BLOCK_ATTN_RES_PREPARE_UTILS_H
