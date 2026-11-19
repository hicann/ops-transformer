/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"
#include "../../../op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "../../../op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_tensor_api_tiling_common.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_data.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

namespace {
constexpr int64_t SPLIT_SWIGLU_MODE = 2;
constexpr int SCALE_ALG_CUBLAS = 1;
constexpr int64_t SINGLE_INPUT_COUNT = 1;
constexpr int64_t MX_GROUP_SIZE = 64;
constexpr int64_t MX_HALF_GROUP_SIZE = 32;
constexpr int64_t SCALE_PAIR_SIZE = 2;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2;
constexpr int64_t NZ_K0 = 16;
constexpr int64_t NZ_C0 = 32;
constexpr int64_t CEIL_DIV_ADJUSTMENT = 1;
constexpr uint32_t AIC_CORE_COUNT = 32;
constexpr uint32_t AIV_CORE_COUNT = 64;
constexpr uint64_t UB_BYTES = 262144;
constexpr uint64_t L1_BYTES = 524288;
constexpr uint64_t L0C_BYTES = 262144;
constexpr float DEFAULT_CLAMP_LIMIT = 7.0F;
constexpr float DEFAULT_GLU_ALPHA = 1.702F;
constexpr float DEFAULT_GLU_BIAS = 1.0F;
constexpr float CUSTOM_CLAMP_LIMIT = 3.0F;
constexpr float CUSTOM_GLU_ALPHA = 0.5F;
constexpr float CUSTOM_GLU_BIAS = -1.0F;
constexpr float FP8_MAX_FINITE = 448.0F;
constexpr uint8_t DIRTY_BYTE_PATTERN = 0xA5U;
constexpr uint16_t DIRTY_HALFWORD_PATTERN = 0xA5A5U;
constexpr uint32_t SCALE_FACTOR_B_SHIFT = 8U;
constexpr uint64_t SCALE_FACTOR_MASK = 0xFFUL;
constexpr uint64_t MIN_SCALE_WINDOW_COUNT = 1;
constexpr uint64_t CGMCT_REGRESSION_FACTOR_B = 7;
constexpr uint64_t CGMCT_REGRESSION_L1_BYTES = 522240;
constexpr uint32_t EXPECTED_TENSOR_BASE_M = 256;
constexpr uint32_t EXPECTED_TENSOR_BASE_N = 128;
constexpr uint32_t EXPECTED_TENSOR_BASE_K = 128;
constexpr uint32_t EXPECTED_TENSOR_L1_K = 256;
constexpr uint32_t EXPECTED_CGMCT_BASE_M = 128;
constexpr uint32_t EXPECTED_CGMCT_BASE_N = 256;
constexpr uint32_t EXPECTED_CGMCT_BASE_K = 128;
constexpr uint8_t SINGLE_L0C_BUFFER = 1;
constexpr int64_t TRANSPOSE_KEY_OFFSET = 1;
constexpr int64_t LARGE_ROUTE_M = 8192;
constexpr size_t WEIGHT_INPUT_INDEX = 3;
constexpr size_t ROUTE_CSV_COLUMN_COUNT = 13;
constexpr size_t EXPECTED_SWIGLU_PARAMS_BYTES = 32U;
constexpr size_t EXPECTED_CGMCT_PARAMS_BYTES = 16U;
constexpr size_t EXPECTED_TENSOR_PARAMS_BYTES = 24U;
constexpr size_t EXPECTED_TENSOR_MM_BYTES = 48U;
constexpr size_t EXPECTED_TENSOR_SWIGLU_OFFSET = 72U;
constexpr size_t EXPECTED_TENSOR_TILING_BYTES = 104U;
struct RouteCase {
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t e;
    bool trans;
    int64_t key;
};

struct RouteCsvCase {
    std::string caseName;
    std::string scenario;
    RouteCase route{};
    std::string transposeValues;
    std::string scaleAlgValues;
    int aiv = AIV_CORE_COUNT;
    int64_t swigluMode = SPLIT_SWIGLU_MODE;
    std::string mutation;
    bool expectSuccess = true;
};

constexpr int64_t V3_CGMCT_NZ_KEY = 0;
constexpr int64_t V3_CGMCT_ZN_KEY = 1;
constexpr int64_t V3_TENSOR_API_NZ_KEY = 16;
constexpr int64_t V3_TENSOR_API_ZN_KEY = 17;
constexpr int64_t V3_A8W4_TENSOR_API_ZN_KEY = 49;
constexpr int64_t A8W4_GROUP_NUM = 4;
constexpr int64_t NZ_FP4_C0 = NZ_C0 * SWIGLU_SPLIT_FACTOR;
constexpr int64_t A8W4_UNALIGNED_SIZE = EXPECTED_TENSOR_BASE_N + MX_GROUP_SIZE;
constexpr int64_t A8W4_FULL_N_WITH_TAIL = EXPECTED_TENSOR_BASE_N * SWIGLU_SPLIT_FACTOR + EXPECTED_TENSOR_BASE_N;

using CgmctTilingData = ::GMMSwigluQuantV2TilingDataParams;
using TensorApiTilingData = GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData;

gert::TilingContextPara MakeContext(const RouteCase& c, optiling::GMMSwigluV2CompileInfo* info, int scaleAlg,
                                    int aiv = AIV_CORE_COUNT, int64_t swigluMode = SPLIT_SWIGLU_MODE)
{
    using Desc = gert::TilingContextPara::TensorDescription;
    using Value = Ops::Transformer::AnyValue;
    const int64_t s = (c.k + MX_GROUP_SIZE - CEIL_DIV_ADJUSTMENT) / MX_GROUP_SIZE;
    const int64_t h = c.n / SWIGLU_SPLIT_FACTOR;
    const auto tensor = [](std::vector<int64_t> dims, ge::DataType dtype) {
        return Desc(ops::ut::MakeGertStorageShape(dims, dims), dtype, ge::FORMAT_ND);
    };
    const std::vector<int64_t> w = c.trans ? std::vector<int64_t>{c.e, c.n, c.k} : std::vector<int64_t>{c.e, c.k, c.n};
    const std::vector<int64_t> ws = c.trans ? std::vector<int64_t>{c.e, c.n, s, SCALE_PAIR_SIZE} :
                                              std::vector<int64_t>{c.e, s, c.n, SCALE_PAIR_SIZE};
    const std::vector<int64_t> storage =
        c.trans ? std::vector<int64_t>{c.e, (c.k + NZ_C0 - CEIL_DIV_ADJUSTMENT) / NZ_C0,
                                       (c.n + NZ_K0 - CEIL_DIV_ADJUSTMENT) / NZ_K0, NZ_K0, NZ_C0} :
                  std::vector<int64_t>{c.e, (c.n + NZ_C0 - CEIL_DIV_ADJUSTMENT) / NZ_C0,
                                       (c.k + NZ_K0 - CEIL_DIV_ADJUSTMENT) / NZ_K0, NZ_K0, NZ_C0};
    info->aicNum_ = AIC_CORE_COUNT;
    info->aivNum_ = static_cast<uint32_t>(aiv);
    info->ubSize_ = UB_BYTES;
    info->l1Size_ = L1_BYTES;
    info->l0CSize_ = L0C_BYTES;
    info->npuArch_ = static_cast<int32_t>(NpuArch::DAV_3510);
    info->supportL12BtBf16 = true;
    gert::TilingContextPara context(
        "GroupedMatmulSwigluQuantV2",
        {tensor({c.m, c.k}, ge::DT_FLOAT8_E4M3FN), tensor({c.m, s, SCALE_PAIR_SIZE}, ge::DT_FLOAT8_E8M0),
         tensor({c.e}, ge::DT_INT64),
         Desc(ops::ut::MakeGertStorageShape(w, storage), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ),
         tensor(ws, ge::DT_FLOAT8_E8M0)},
        {tensor({c.m, h}, ge::DT_FLOAT8_E4M3FN),
         tensor({c.m, (h + MX_GROUP_SIZE - CEIL_DIV_ADJUSTMENT) / MX_GROUP_SIZE, SCALE_PAIR_SIZE}, ge::DT_FLOAT8_E8M0)},
        {{"dequant_mode", Value::CreateFrom<int64_t>(SPLIT_SWIGLU_MODE)},
         {"dequant_dtype", Value::CreateFrom<int64_t>(0)},
         {"quant_mode", Value::CreateFrom<int64_t>(SPLIT_SWIGLU_MODE)},
         {"quant_dtype", Value::CreateFrom<int64_t>(ge::DT_FLOAT8_E4M3FN)},
         {"transpose_weight", Value::CreateFrom<bool>(c.trans)},
         {"group_list_type", Value::CreateFrom<int64_t>(0)},
         {"tuning_config", Value::CreateFrom<std::vector<int64_t>>({})},
         {"swiglu_mode", Value::CreateFrom<int64_t>(swigluMode)},
         {"clamp_limit", Value::CreateFrom<float>(DEFAULT_CLAMP_LIMIT)},
         {"glu_alpha", Value::CreateFrom<float>(DEFAULT_GLU_ALPHA)},
         {"glu_bias", Value::CreateFrom<float>(DEFAULT_GLU_BIAS)},
         {"round_mode", Value::CreateFrom<std::string>("rint")},
         {"scale_alg", Value::CreateFrom<int64_t>(scaleAlg)},
         {"dst_type_max", Value::CreateFrom<float>(0.0F)}},
        {SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT, 0, 0, 0},
        {SINGLE_INPUT_COUNT, SINGLE_INPUT_COUNT}, info, "Ascend950", AIC_CORE_COUNT, UB_BYTES);
    context.socInfoString_ =
        R"({"hardware_info":{"CORE_NUM":32,"cube_core_cnt":32,"vector_core_cnt":)" + std::to_string(aiv) +
        R"(,"UB_SIZE":262144,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144}})";
    return context;
}

std::vector<RouteCsvCase> LoadRouteCases()
{
    const std::string path = ops::ut::ResolveCsvPath("test_grouped_matmul_swiglu_quant_weight_nz_v3_route_tiling.csv",
                                                     "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_host", __FILE__);
    std::ifstream input(path);
    EXPECT_TRUE(input.is_open()) << "Failed to open CSV file: " << path;
    std::vector<RouteCsvCase> cases;
    std::string line;
    size_t lineNo = 0;
    while (std::getline(input, line)) {
        ++lineNo;
        if (line.empty() || lineNo == 1) {
            continue;
        }
        std::vector<std::string> fields;
        ops::ut::SplitStr2Vec(line, ",", fields);
        if (fields.size() != ROUTE_CSV_COLUMN_COUNT) {
            ADD_FAILURE() << path << ':' << lineNo << " expected " << ROUTE_CSV_COLUMN_COUNT << " fields, got "
                          << fields.size();
            continue;
        }
        try {
            for (auto& field : fields) {
                field = ops::ut::Trim(field);
            }
            RouteCsvCase item;
            item.caseName = fields[0];
            item.scenario = fields[1];
            item.route.m = std::stoll(fields[2]);
            item.route.k = std::stoll(fields[3]);
            item.route.n = std::stoll(fields[4]);
            item.route.e = std::stoll(fields[5]);
            item.transposeValues = fields[6];
            item.scaleAlgValues = fields[7];
            item.aiv = std::stoi(fields[8]);
            item.swigluMode = std::stoll(fields[9]);
            item.route.key = std::stoll(fields[10]);
            item.mutation = fields[11];
            item.expectSuccess = ops::ut::ParseBool(fields[12]);
            if (item.caseName.empty() ||
                (item.transposeValues != "0" && item.transposeValues != "1" && item.transposeValues != "both") ||
                (item.scaleAlgValues != "0" && item.scaleAlgValues != "1" && item.scaleAlgValues != "both")) {
                throw std::invalid_argument("invalid case name, transpose, or scaleAlg field");
            }
            cases.push_back(std::move(item));
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(path, lineNo, "", error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid route cases in " << path;
    return cases;
}

gert::TilingContextPara MakeA8W4Context(const RouteCase& route, optiling::GMMSwigluV2CompileInfo* info,
                                        bool smooth = false, int groupListType = 0, int scaleAlg = 0)
{
    auto context = MakeContext(route, info, scaleAlg);
    const int64_t c0 = NZ_FP4_C0;
    const std::vector<int64_t> view =
        route.trans ? std::vector<int64_t>{route.e, route.n, route.k} : std::vector<int64_t>{route.e, route.k, route.n};
    const std::vector<int64_t> storage =
        route.trans ? std::vector<int64_t>{route.e, (route.k + c0 - 1) / c0, (route.n + NZ_K0 - 1) / NZ_K0, NZ_K0, c0} :
                      std::vector<int64_t>{route.e, (route.n + c0 - 1) / c0, (route.k + NZ_K0 - 1) / NZ_K0, NZ_K0, c0};
    using Desc = gert::TilingContextPara::TensorDescription;
    context.inputTensorDesc_[WEIGHT_INPUT_INDEX] =
        Desc(ops::ut::MakeGertStorageShape(view, storage), ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ);
    context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_GROUPLIST_TYPE].attr_ =
        Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType);
    if (smooth) {
        const std::vector<int64_t> dims{route.e, route.n / SWIGLU_SPLIT_FACTOR};
        context.inputInstanceNum_[optiling::GroupedMatmulSwigluQuantV2Tiling::SMOOTH_SCALE_INDEX] = 1;
        context.inputTensorDesc_.emplace_back(ops::ut::MakeGertStorageShape(dims, dims), ge::DT_FLOAT, ge::FORMAT_ND);
    }
    return context;
}

TEST(GroupedMatmulSwigluQuantV3Route, A8W4UsesDedicatedTensorApiKeyForSmallMTails)
{
    for (bool transpose : {true}) {
        for (int64_t n : {static_cast<int64_t>(EXPECTED_TENSOR_BASE_N), EXPECTED_TENSOR_BASE_N * SWIGLU_SPLIT_FACTOR,
                          A8W4_FULL_N_WITH_TAIL}) {
            for (int64_t m :
                 {int64_t{1}, NZ_K0 + 1, static_cast<int64_t>(EXPECTED_TENSOR_BASE_M) - 1,
                  static_cast<int64_t>(EXPECTED_TENSOR_BASE_M), static_cast<int64_t>(EXPECTED_TENSOR_BASE_M) + 1}) {
                for (bool smooth : {false, true}) {
                    for (int groupListType : {0, 1}) {
                        for (int scaleAlg : {0, 1}) {
                            SCOPED_TRACE(::testing::Message()
                                         << m << ',' << n << ',' << smooth << ',' << groupListType << ',' << scaleAlg);
                            optiling::GMMSwigluV2CompileInfo info{};
                            const int64_t key = V3_A8W4_TENSOR_API_ZN_KEY;
                            const RouteCase route{m, EXPECTED_TENSOR_BASE_K, n, A8W4_GROUP_NUM, transpose, key};
                            auto context = MakeA8W4Context(route, &info, smooth, groupListType, scaleAlg);
                            TilingInfo result;
                            ASSERT_TRUE(ExecuteTiling(context, result));
                            EXPECT_EQ(result.tilingKey, key);
                            ASSERT_EQ(result.tilingDataSize, sizeof(TensorApiTilingData));
                            TensorApiTilingData data{};
                            std::memcpy(&data, result.tilingData.get(), sizeof(data));
                            EXPECT_EQ(data.gmmQuantParams.groupNum, route.e);
                            EXPECT_EQ(data.gmmQuantParams.groupListType, groupListType);
                            EXPECT_EQ(data.mmTilingData.m, m);
                            EXPECT_EQ(data.mmTilingData.n, route.n);
                            EXPECT_EQ(data.mmTilingData.k, route.k);
                            EXPECT_EQ(data.mmTilingData.baseM, EXPECTED_TENSOR_BASE_M);
                            EXPECT_EQ(data.mmTilingData.baseN, EXPECTED_TENSOR_BASE_N);
                            EXPECT_EQ(data.mmTilingData.baseK, EXPECTED_TENSOR_BASE_K);
                            EXPECT_EQ(data.swigluParams.swigluMode, SPLIT_SWIGLU_MODE);
                            EXPECT_EQ(data.swigluParams.scaleAlg, scaleAlg);
                        }
                    }
                }
            }
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, A8W4RejectsUnsupportedLayoutShapesAndPlatform)
{
    for (const std::string mutation :
         {"k_alignment", "n_alignment", "smooth_rank", "smooth_width", "smooth_dtype", "smooth_format", "core_ratio",
          "l1_size", "nz_c0", "nd_weight", "non_transposed"}) {
        SCOPED_TRACE(mutation);
        optiling::GMMSwigluV2CompileInfo info{};
        RouteCase route{
            NZ_K0 + 1, EXPECTED_TENSOR_BASE_K,   EXPECTED_TENSOR_BASE_N * SWIGLU_SPLIT_FACTOR, A8W4_GROUP_NUM,
            true,      V3_A8W4_TENSOR_API_ZN_KEY};
        if (mutation == "k_alignment") {
            route.k = A8W4_UNALIGNED_SIZE;
        }
        if (mutation == "n_alignment") {
            route.n = A8W4_UNALIGNED_SIZE;
        }
        if (mutation == "non_transposed") {
            route.trans = false;
        }
        auto context = MakeA8W4Context(route, &info, true);
        using Desc = gert::TilingContextPara::TensorDescription;
        const std::vector<int64_t> smoothDims{route.e, route.n / SWIGLU_SPLIT_FACTOR};
        if (mutation.rfind("smooth_", 0) == 0) {
            const std::vector<int64_t> dims = mutation == "smooth_rank" ?
                                                  std::vector<int64_t>{route.n / SWIGLU_SPLIT_FACTOR} :
                                              mutation == "smooth_width" ? std::vector<int64_t>{route.e, route.n} :
                                                                           smoothDims;
            context.inputTensorDesc_.back() =
                Desc(ops::ut::MakeGertStorageShape(dims, dims), mutation == "smooth_dtype" ? ge::DT_BF16 : ge::DT_FLOAT,
                     mutation == "smooth_format" ? ge::FORMAT_NCHW : ge::FORMAT_ND);
        }
        if (mutation == "core_ratio") {
            info.aivNum_ = AIC_CORE_COUNT;
            const std::string original = "\"vector_core_cnt\":" + std::to_string(AIV_CORE_COUNT);
            context.socInfoString_.replace(context.socInfoString_.find(original), original.size(),
                                           "\"vector_core_cnt\":" + std::to_string(AIC_CORE_COUNT));
        }
        if (mutation == "l1_size") {
            info.l1Size_ = 0;
            const std::string original = "\"L1_SIZE\":" + std::to_string(L1_BYTES);
            context.socInfoString_.replace(context.socInfoString_.find(original), original.size(), "\"L1_SIZE\":0");
        }
        if (mutation == "nz_c0") {
            const std::vector<int64_t> view{route.e, route.n, route.k};
            const std::vector<int64_t> wrongStorage{route.e, route.k / NZ_C0, route.n / NZ_K0, NZ_K0, NZ_C0};
            context.inputTensorDesc_[WEIGHT_INPUT_INDEX] =
                Desc(ops::ut::MakeGertStorageShape(view, wrongStorage), ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ);
        }
        if (mutation == "nd_weight") {
            const std::vector<int64_t> view{route.e, route.n, route.k};
            context.inputTensorDesc_[WEIGHT_INPUT_INDEX] =
                Desc(ops::ut::MakeGertStorageShape(view, view), ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
        }
        TilingInfo result;
        EXPECT_FALSE(ExecuteTiling(context, result));
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, A8W4RequiresPlatformMemoryInfo)
{
    for (const std::string insufficientResource : {"none", "L1_SIZE", "L0C_SIZE"}) {
        SCOPED_TRACE(insufficientResource);
        optiling::GMMSwigluV2CompileInfo info{};
        const RouteCase route{NZ_K0 + 1, EXPECTED_TENSOR_BASE_K,   A8W4_FULL_N_WITH_TAIL, A8W4_GROUP_NUM,
                              true,      V3_A8W4_TENSOR_API_ZN_KEY};
        auto context = MakeA8W4Context(route, &info, true);
        info.l1Size_ = insufficientResource == "L1_SIZE" ? 0 : L1_BYTES;
        info.l0CSize_ = insufficientResource == "L0C_SIZE" ? 0 : L0C_BYTES;
        const auto replaceMemorySize = [&context](const std::string& name, uint64_t previous, uint64_t current) {
            const std::string original = "\"" + name + "\":" + std::to_string(previous);
            context.socInfoString_.replace(context.socInfoString_.find(original), original.size(),
                                           "\"" + name + "\":" + std::to_string(current));
        };
        replaceMemorySize("L1_SIZE", L1_BYTES, info.l1Size_);
        replaceMemorySize("L0C_SIZE", L0C_BYTES, info.l0CSize_);
        TilingInfo result;
        if (insufficientResource == "none") {
            ASSERT_TRUE(ExecuteTiling(context, result));
            EXPECT_EQ(result.tilingKey, V3_A8W4_TENSOR_API_ZN_KEY);
        } else {
            EXPECT_FALSE(ExecuteTiling(context, result));
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, PerTokenFp8PreservesEmptySmoothPlaceholder)
{
    namespace P = optiling::GroupedMatmulSwigluQuantV2Tiling;
    using Value = Ops::Transformer::AnyValue;
    using Desc = gert::TilingContextPara::TensorDescription;
    const RouteCase route{EXPECTED_TENSOR_BASE_M,
                          EXPECTED_TENSOR_BASE_K,
                          EXPECTED_TENSOR_BASE_N * SWIGLU_SPLIT_FACTOR,
                          A8W4_GROUP_NUM,
                          false,
                          0L};
    const auto tensor = [](std::vector<int64_t> dims, ge::DataType dtype) {
        return Desc(ops::ut::MakeGertStorageShape(dims, dims), dtype, ge::FORMAT_ND);
    };
    for (auto xType : {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2}) {
        for (auto weightType : {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2}) {
            optiling::GMMSwigluV2CompileInfo info{};
            auto context = MakeContext(route, &info, 0, AIV_CORE_COUNT, 0L);
            context.inputTensorDesc_[P::X_INDEX].dtype_ = xType;
            context.inputTensorDesc_[P::X_SCALE_INDEX] = tensor({route.m}, ge::DT_FLOAT);
            context.inputTensorDesc_[P::WEIGHT_INDEX] = tensor({route.e, route.k, route.n}, weightType);
            context.inputTensorDesc_[P::WEIGHT_SCALE_INDEX] = tensor({route.e, route.n}, ge::DT_FLOAT);
            context.outputTensorDesc_[P::Y_SCALE_INDEX] = tensor({route.m}, ge::DT_FLOAT);
            context.attrs_[P::ATTR_INDEX_DEQUANT_MODE].attr_ = Value::CreateFrom<int64_t>(0L);
            context.attrs_[P::ATTR_INDEX_QUANT_MODE].attr_ = Value::CreateFrom<int64_t>(0L);
            TilingInfo absent;
            ASSERT_TRUE(ExecuteTiling(context, absent));
            context.inputInstanceNum_[P::SMOOTH_SCALE_INDEX] = SINGLE_INPUT_COUNT;
            context.inputTensorDesc_.push_back(tensor({0L}, ge::DT_FLOAT));
            TilingInfo empty;
            ASSERT_TRUE(ExecuteTiling(context, empty));
            EXPECT_EQ(empty.tilingKey, absent.tilingKey);
            ASSERT_EQ(empty.tilingDataSize, absent.tilingDataSize);
            EXPECT_EQ(std::memcmp(empty.tilingData.get(), absent.tilingData.get(), empty.tilingDataSize), 0);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, A8W8RejectsNonemptySmoothScaleAndAcceptsEmptyPlaceholder)
{
    for (bool transpose : {false, true}) {
        for (int64_t swigluMode : {0L, SPLIT_SWIGLU_MODE}) {
            SCOPED_TRACE(testing::Message() << "transpose=" << transpose << ", swigluMode=" << swigluMode);
            optiling::GMMSwigluV2CompileInfo info{};
            const RouteCase route{LARGE_ROUTE_M,
                                  EXPECTED_TENSOR_BASE_K,
                                  EXPECTED_TENSOR_BASE_N * SWIGLU_SPLIT_FACTOR,
                                  A8W4_GROUP_NUM,
                                  transpose,
                                  0L};
            auto noSmooth = MakeContext(route, &info, 0, AIV_CORE_COUNT, swigluMode);
            TilingInfo validResult;
            ASSERT_TRUE(ExecuteTiling(noSmooth, validResult));
            for (bool emptyTensor : {false, true}) {
                SCOPED_TRACE(testing::Message() << "emptyTensor=" << emptyTensor);
                auto context = MakeContext(route, &info, 0, AIV_CORE_COUNT, swigluMode);
                const std::vector<int64_t> dims = emptyTensor ?
                                                      std::vector<int64_t>{0L} :
                                                      std::vector<int64_t>{route.e, route.n / SWIGLU_SPLIT_FACTOR};
                context.inputInstanceNum_[optiling::GroupedMatmulSwigluQuantV2Tiling::SMOOTH_SCALE_INDEX] =
                    SINGLE_INPUT_COUNT;
                context.inputTensorDesc_.emplace_back(ops::ut::MakeGertStorageShape(dims, dims), ge::DT_FLOAT,
                                                      ge::FORMAT_ND);
                TilingInfo result;
                EXPECT_EQ(ExecuteTiling(context, result), emptyTensor);
            }
        }
    }
}

template <typename T>
const T* GetTilingData(const TilingInfo& result)
{
    EXPECT_EQ(result.tilingDataSize, sizeof(T));
    if (result.tilingDataSize != sizeof(T)) {
        return nullptr;
    }
    return reinterpret_cast<const T*>(result.tilingData.get());
}

const ::GMMSwigluQuantSwigluParams* GetSwigluParams(const TilingInfo& result)
{
    if (result.tilingKey == V3_CGMCT_NZ_KEY || result.tilingKey == V3_CGMCT_ZN_KEY) {
        const auto* data = GetTilingData<CgmctTilingData>(result);
        return data == nullptr ? nullptr : &data->swigluParams;
    }
    if (result.tilingKey == V3_TENSOR_API_NZ_KEY || result.tilingKey == V3_TENSOR_API_ZN_KEY) {
        const auto* data = GetTilingData<TensorApiTilingData>(result);
        return data == nullptr ? nullptr : &data->swigluParams;
    }
    ADD_FAILURE() << "Unexpected MXFP8 tiling key: " << result.tilingKey;
    return nullptr;
}

void ExpectSwigluParams(const ::GMMSwigluQuantSwigluParams& params, int64_t mode, uint8_t scaleAlg,
                        float clampLimit = DEFAULT_CLAMP_LIMIT, float gluAlpha = DEFAULT_GLU_ALPHA,
                        float gluBias = DEFAULT_GLU_BIAS)
{
    EXPECT_EQ(params.swigluMode, mode);
    EXPECT_EQ(params.scaleAlg, scaleAlg);
    EXPECT_FLOAT_EQ(params.clampLimit, clampLimit);
    EXPECT_FLOAT_EQ(params.gluAlpha, gluAlpha);
    EXPECT_FLOAT_EQ(params.gluBias, gluBias);
    EXPECT_FLOAT_EQ(params.dstTypeMax, 0.0F);
    EXPECT_EQ(params.roundMode, 0U);
    EXPECT_EQ(params.reserved0, 0U);
    EXPECT_EQ(params.reserved1, 0U);
}

template <typename DeviceTiling, typename HostTiling>
void ExpectHostTailReset(HostTiling& host)
{
    ASSERT_EQ(host.GetDataSize(), sizeof(DeviceTiling));
    std::array<uint8_t, sizeof(DeviceTiling)> buffer;
    buffer.fill(DIRTY_BYTE_PATTERN);
    ::GMMSwigluQuantSwigluParams previous;
    previous.swigluMode = SPLIT_SWIGLU_MODE;
    previous.clampLimit = CUSTOM_CLAMP_LIMIT;
    previous.gluAlpha = CUSTOM_GLU_ALPHA;
    previous.gluBias = CUSTOM_GLU_BIAS;
    previous.scaleAlg = SCALE_ALG_CUBLAS;
    previous.dstTypeMax = FP8_MAX_FINITE;
    previous.roundMode = DIRTY_BYTE_PATTERN;
    previous.reserved0 = DIRTY_HALFWORD_PATTERN;
    previous.reserved1 = 0xA5A5A5A5U;
    optiling::SetGMMSwigluQuantSwigluParams(host.swigluParams, previous);
    host.SaveToBuffer(buffer.data(), buffer.size());
    DeviceTiling decoded{};
    std::memcpy(&decoded, buffer.data(), sizeof(decoded));
    EXPECT_EQ(std::memcmp(&decoded.swigluParams, &previous, sizeof(previous)), 0);

    // Exercise a reused Host payload and a nonzero destination: resetting
    // legacy attributes must overwrite every field, including reserved bytes.
    buffer.fill(DIRTY_BYTE_PATTERN);
    optiling::SetGMMSwigluQuantSwigluParams(host.swigluParams);
    host.SaveToBuffer(buffer.data(), buffer.size());
    std::memcpy(&decoded, buffer.data(), sizeof(decoded));
    ExpectSwigluParams(decoded.swigluParams, 0, 0);
}

TEST(GroupedMatmulSwigluQuantV3Route, SharedPayloadsPreserveLegacyPrefixes)
{
    EXPECT_EQ(sizeof(::GMMSwigluQuantSwigluParams), EXPECTED_SWIGLU_PARAMS_BYTES);
    EXPECT_EQ(offsetof(::GMMSwigluQuantSwigluParams, dstTypeMax), 20U);
    EXPECT_EQ(offsetof(::GMMSwigluQuantSwigluParams, scaleAlg), 24U);
    EXPECT_EQ(offsetof(::GMMSwigluQuantSwigluParams, roundMode), 25U);
    EXPECT_EQ(offsetof(::GMMSwigluQuantSwigluParams, reserved0), 26U);
    EXPECT_EQ(offsetof(::GMMSwigluQuantSwigluParams, reserved1), 28U);
    EXPECT_EQ(sizeof(::GMMSwigluQuantV2Params), EXPECTED_CGMCT_PARAMS_BYTES);
    EXPECT_EQ(offsetof(CgmctTilingData, mmTilingData), EXPECTED_CGMCT_PARAMS_BYTES);
    EXPECT_EQ(offsetof(CgmctTilingData, swigluParams), EXPECTED_CGMCT_PARAMS_BYTES + sizeof(::TCubeTiling));
    EXPECT_EQ(sizeof(CgmctTilingData),
              EXPECTED_CGMCT_PARAMS_BYTES + sizeof(::TCubeTiling) + EXPECTED_SWIGLU_PARAMS_BYTES);
    EXPECT_EQ(sizeof(GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiQuantParams), EXPECTED_TENSOR_PARAMS_BYTES);
    EXPECT_EQ(sizeof(GroupedMatmulSwigluQuantV2TensorApi::GMMTensorApiMMTiling), EXPECTED_TENSOR_MM_BYTES);
    EXPECT_EQ(offsetof(TensorApiTilingData, mmTilingData), EXPECTED_TENSOR_PARAMS_BYTES);
    EXPECT_EQ(offsetof(TensorApiTilingData, swigluParams), EXPECTED_TENSOR_SWIGLU_OFFSET);
    EXPECT_EQ(sizeof(TensorApiTilingData), EXPECTED_TENSOR_TILING_BYTES);

    optiling::GMMSwigluQuantTilingDataParams cgmctHost;
    optiling::GMMSwigluQuantV2TensorApiTilingData tensorApiHost;
    EXPECT_EQ(cgmctHost.GetDataSize(), sizeof(CgmctTilingData));
    EXPECT_EQ(tensorApiHost.GetDataSize(), sizeof(TensorApiTilingData));
}

TEST(GroupedMatmulSwigluQuantV3Route, HostTailResetOverwritesPreviousModeAndReservedBytes)
{
    optiling::GMMSwigluQuantTilingDataParams cgmctHost;
    ExpectHostTailReset<CgmctTilingData>(cgmctHost);
    optiling::GMMSwigluQuantV2TensorApiTilingData tensorApiHost;
    ExpectHostTailReset<TensorApiTilingData>(tensorApiHost);
}

void CheckCsvRouteCase(const RouteCsvCase& item, bool trans, int scaleAlg)
{
    RouteCase route = item.route;
    route.trans = trans;
    // The CSV records the non-transposed family key; transposed keys add one.
    route.key += trans ? TRANSPOSE_KEY_OFFSET : 0;
    SCOPED_TRACE(::testing::Message() << item.caseName << ", trans=" << trans << ", scaleAlg=" << scaleAlg);

    optiling::GMMSwigluV2CompileInfo info{};
    auto context = MakeContext(route, &info, scaleAlg, item.aiv, item.swigluMode);
    const bool legacyTensor = item.scenario == "legacy_tensor_explicit" || item.scenario == "legacy_tensor_omitted";
    if (legacyTensor) {
        const std::vector<int64_t> weightShape = {route.e, route.k, route.n};
        context.inputTensorDesc_[WEIGHT_INPUT_INDEX] = gert::TilingContextPara::TensorDescription(
            ops::ut::MakeGertStorageShape(weightShape, weightShape), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    }
    if (item.scenario == "legacy_cgmct_omitted" || item.scenario == "legacy_tensor_omitted") {
        context.attrs_.erase(
            context.attrs_.begin() + optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_SWIGLU_MODE,
            context.attrs_.end());
    }
    if (item.mutation == "invalid_platform") {
        info.npuArch_ = static_cast<int32_t>(NpuArch::DAV_2201);
    } else if (item.mutation == "dequant_overflow_pos" || item.mutation == "dequant_overflow_neg" ||
               item.mutation == "dequant_one") {
        const int64_t overflow = 1LL << 32;
        const int64_t value = item.mutation == "dequant_one"          ? 1 :
                              item.mutation == "dequant_overflow_neg" ? -overflow :
                                                                        overflow;
        context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_DEQUANT_DTYPE].attr_ =
            Ops::Transformer::AnyValue::CreateFrom<int64_t>(value);
    } else if (item.mutation.rfind("missing_input_", 0) == 0) {
        const size_t index = std::stoul(item.mutation.substr(std::string("missing_input_").size()));
        ASSERT_LT(index, context.inputTensorDesc_.size());
        context.inputTensorDesc_.erase(context.inputTensorDesc_.begin() + index);
        context.inputInstanceNum_[index] = 0;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
        return;
    } else if (item.mutation.rfind("missing_output_", 0) == 0) {
        const size_t index = std::stoul(item.mutation.substr(std::string("missing_output_").size()));
        ASSERT_LT(index, context.outputTensorDesc_.size());
        context.outputTensorDesc_.erase(context.outputTensorDesc_.begin() + index);
        context.outputInstanceNum_[index] = 0;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
        return;
    }
    const bool customParams =
        item.scenario == "attributes" || (item.scenario == "reuse" && item.swigluMode == SPLIT_SWIGLU_MODE);
    if (customParams) {
        context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_CLAMP_LIMIT].attr_ =
            Ops::Transformer::AnyValue::CreateFrom<float>(CUSTOM_CLAMP_LIMIT);
        context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_GLU_ALPHA].attr_ =
            Ops::Transformer::AnyValue::CreateFrom<float>(CUSTOM_GLU_ALPHA);
        context.attrs_[optiling::GroupedMatmulSwigluQuantV2Tiling::ATTR_INDEX_GLU_BIAS].attr_ =
            Ops::Transformer::AnyValue::CreateFrom<float>(CUSTOM_GLU_BIAS);
    }

    TilingInfo result;
    if (!item.expectSuccess) {
        EXPECT_FALSE(ExecuteTiling(context, result));
        return;
    }
    ASSERT_TRUE(ExecuteTiling(context, result));
    ASSERT_EQ(result.tilingKey, route.key);
    if (item.scenario == "capability") {
        EXPECT_EQ(result.blockNum, AIC_CORE_COUNT);
    }
    const auto* params = GetSwigluParams(result);
    ASSERT_NE(params, nullptr);
    const int expectedAlg = item.swigluMode == SPLIT_SWIGLU_MODE ? scaleAlg : 0;
    if (customParams) {
        ExpectSwigluParams(*params, item.swigluMode, static_cast<uint8_t>(expectedAlg), CUSTOM_CLAMP_LIMIT,
                           CUSTOM_GLU_ALPHA, CUSTOM_GLU_BIAS);
    } else {
        ExpectSwigluParams(*params, item.swigluMode, static_cast<uint8_t>(expectedAlg));
    }

    if (item.scenario == "unaligned" || item.scenario == "unaligned_regression") {
        const auto* tilingData = GetTilingData<CgmctTilingData>(result);
        ASSERT_NE(tilingData, nullptr);
        const auto& mm = tilingData->mmTilingData;
        ASSERT_GT(mm.stepKa, 0);
        ASSERT_GT(mm.stepKb, 0);
        ASSERT_GT(mm.baseK, 0);
        const auto mxTypePara = static_cast<uint32_t>(mm.mxTypePara);
        const uint64_t scaleFactorA = mxTypePara & SCALE_FACTOR_MASK;
        const uint64_t scaleFactorB = (mxTypePara >> SCALE_FACTOR_B_SHIFT) & SCALE_FACTOR_MASK;
        const uint64_t fullScaleWindowsA = std::max<int64_t>(MIN_SCALE_WINDOW_COUNT, route.k / (mm.stepKa * mm.baseK));
        const uint64_t fullScaleWindowsB = std::max<int64_t>(MIN_SCALE_WINDOW_COUNT, route.k / (mm.stepKb * mm.baseK));
        EXPECT_LE(scaleFactorA, fullScaleWindowsA);
        EXPECT_LE(scaleFactorB, fullScaleWindowsB);
        const uint64_t scaleK =
            ((mm.baseK + MX_HALF_GROUP_SIZE - CEIL_DIV_ADJUSTMENT) / MX_HALF_GROUP_SIZE + CEIL_DIV_ADJUSTMENT) /
            SCALE_PAIR_SIZE * SCALE_PAIR_SIZE;
        const uint64_t baseASize = mm.baseM * mm.baseK;
        const uint64_t baseBSize = mm.baseN * mm.baseK;
        const uint64_t baseScaleASize = mm.baseM * scaleK;
        const uint64_t baseScaleBSize = mm.baseN * scaleK;
        const uint64_t l1Size = mm.depthA1 * (baseASize + scaleFactorA * baseScaleASize) +
                                mm.depthB1 * (baseBSize + scaleFactorB * baseScaleBSize);
        EXPECT_LE(l1Size, info.l1Size_);
        if (item.scenario == "unaligned_regression") {
            EXPECT_EQ(scaleFactorA, MIN_SCALE_WINDOW_COUNT);
            EXPECT_EQ(scaleFactorB, CGMCT_REGRESSION_FACTOR_B);
            EXPECT_EQ(l1Size, CGMCT_REGRESSION_L1_BYTES);
        }
    }

    if (item.scenario == "symmetric_l1" || item.scenario == "long_k" || item.scenario == "performance") {
        if (route.key == V3_TENSOR_API_NZ_KEY || route.key == V3_TENSOR_API_ZN_KEY) {
            const auto* tilingData = GetTilingData<TensorApiTilingData>(result);
            ASSERT_NE(tilingData, nullptr);
            const auto& mm = tilingData->mmTilingData;
            EXPECT_EQ(mm.baseM, EXPECTED_TENSOR_BASE_M);
            EXPECT_EQ(mm.baseN, EXPECTED_TENSOR_BASE_N);
            EXPECT_EQ(mm.baseK, EXPECTED_TENSOR_BASE_K);
            if (item.scenario != "performance") {
                EXPECT_EQ(mm.kAL1, EXPECTED_TENSOR_L1_K);
                EXPECT_EQ(mm.kBL1, EXPECTED_TENSOR_L1_K);
                EXPECT_EQ(mm.dbL0C, SINGLE_L0C_BUFFER);
                EXPECT_EQ(mm.scaleKAL1, item.scenario == "long_k" ? LARGE_ROUTE_M : route.k);
                EXPECT_EQ(mm.scaleKBL1, item.scenario == "long_k" ? LARGE_ROUTE_M : route.k);
            }
        } else {
            ASSERT_EQ(item.scenario, "performance");
            const auto* tilingData = GetTilingData<CgmctTilingData>(result);
            ASSERT_NE(tilingData, nullptr);
            EXPECT_EQ(tilingData->mmTilingData.baseM, EXPECTED_CGMCT_BASE_M);
            EXPECT_EQ(tilingData->mmTilingData.baseN, EXPECTED_CGMCT_BASE_N);
            EXPECT_EQ(tilingData->mmTilingData.baseK, EXPECTED_CGMCT_BASE_K);
        }
    }
}

TEST(GroupedMatmulSwigluQuantV3Route, CsvRouteCases)
{
    std::vector<RouteCsvCase> reuseCases;
    for (const auto& item : LoadRouteCases()) {
        if (item.scenario == "reuse") {
            reuseCases.push_back(item);
            continue;
        }
        const std::vector<bool> transValues = item.transposeValues == "both" ?
                                                  std::vector<bool>{false, true} :
                                                  std::vector<bool>{item.transposeValues == "1"};
        const std::vector<int> scaleAlgs = item.scaleAlgValues == "both" ?
                                               std::vector<int>{0, SCALE_ALG_CUBLAS} :
                                               std::vector<int>{std::stoi(item.scaleAlgValues)};
        for (const bool trans : transValues) {
            for (const int scaleAlg : scaleAlgs) {
                CheckCsvRouteCase(item, trans, scaleAlg);
            }
        }
    }
    // Preserve the original per-layout sequence of V2 -> V3 -> V2 calls.
    for (const bool trans : {false, true}) {
        for (const auto& item : reuseCases) {
            CheckCsvRouteCase(item, trans, std::stoi(item.scaleAlgValues));
        }
    }
}

} // namespace
