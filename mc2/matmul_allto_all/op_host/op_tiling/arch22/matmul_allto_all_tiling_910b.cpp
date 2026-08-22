/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_allto_all_tiling_910b.cpp
 * \brief
 */
#include "vector"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "common/utils/op_mc2.h"
#include "mc2_hcom_topo_info.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <map>
#include "matmul_allto_all_tiling_910b.h"

using namespace AscendC;
using namespace ge;
using namespace matmul_allto_all_910b_tiling_key;

namespace {
const char *K_INNER_DEBUG = "MatmulAlltoAll Tiling Debug";
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t WORKSPACE_NUM = 2;
constexpr uint32_t FORMAT_ND_DIM = 1;
constexpr uint32_t X1_QUANT_SIGN = 10;
constexpr uint32_t SUPPORT_QUANT_MODE = 32; // x1_quant_mode * X1_QUANT_SIGN + x2_quant_mode

constexpr int32_t RANKSIZE_TWO = 2;
constexpr int32_t RANKSIZE_FOUR = 4;
constexpr int32_t RANKSIZE_EIGHT = 8;
constexpr int32_t MATMULALLTOALL_TWO_RANK_FP16_UBMOVENUM_DEFAULT = 40;
constexpr int32_t MATMULALLTOALL_TWO_RANK_FP16_PVALUE_DEFAULT = 10;
constexpr int32_t MATMULALLTOALL_TWO_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t MATMULALLTOALL_FOUR_RANK_FP16_UBMOVENUM_DEFAULT = 40;
constexpr int32_t MATMULALLTOALL_FOUR_RANK_FP16_PVALUE_DEFAULT = 10;
constexpr int32_t MATMULALLTOALL_FOUR_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t MATMULALLTOALL_EIGHT_RANK_FP16_M0_DEFAULT = 128;
constexpr int32_t MATMULALLTOALL_EIGHT_RANK_FP16_PVALUE_DEFAULT = 10;
constexpr int32_t MATMULALLTOALL_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 80;
constexpr int32_t HALF_KBYTE = 512;
constexpr int32_t DEFAULT_ROW = 128;
constexpr int32_t DEFAULT_COL = 256;
constexpr int32_t MAX_BUFF_BYTES = 200 * 1024 * 1024;
constexpr int32_t FLAG_BUFF_BYTES = 20 * 1024 * 1024;
constexpr int32_t MAX_BLOCK_COUNT = 2;
constexpr int32_t MIN_P_VALUE = 1;
constexpr int32_t ELEMENT_SIZE = 2; // 通信时每个元素均占用2字节
constexpr int32_t MB_BYTES = 1024 * 1024;
constexpr int32_t CONDITION_M_ST = 0;
constexpr int32_t CONDITION_M_END = 1;
constexpr int32_t CONDITION_K_ST = 2;
constexpr int32_t CONDITION_K_END = 3;
constexpr int32_t CONDITION_N_ST = 4;
constexpr int32_t CONDITION_N_END = 5;
constexpr uint32_t COUNT_PARAMS_WITH_BIAS = 4;    // [x1, x2, bias, y]
constexpr uint32_t COUNT_PARAMS_WITHOUT_BIAS = 3; // [x1, x2, y]
const std::set<int> SUPPORT_RANK_SIZE_910B{2, 4, 8};
const std::vector<std::vector<uint32_t>> SUPPORTED_TYPES_WITH_BIAS = {
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT, ge::DT_BF16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16}};
const std::vector<std::vector<uint32_t>> SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}};
} // namespace

template <typename T>
T ClampValue(T value, T minVal, T maxVal)
{
    return (value < minVal) ? minVal : (value > maxVal) ? maxVal : value;
}

void TilingParamDeal(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info)
{
    int32_t dataTypeSize = ELEMENT_SIZE;
    int32_t peerMemSize = (MAX_BUFF_BYTES - FLAG_BUFF_BYTES) / dataTypeSize;
    if (cocTilingData.m0 * cocTilingData.pValue * info.N > peerMemSize / MAX_BLOCK_COUNT) {
        int32_t maxValue = peerMemSize / MAX_BLOCK_COUNT / info.N;
        cocTilingData.m0 = DEFAULT_ROW;
        cocTilingData.n0 = DEFAULT_COL;
        cocTilingData.pValue = ClampValue(maxValue / cocTilingData.m0, MIN_P_VALUE, cocTilingData.pValue);
    }
}

namespace MC2Tiling {

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallTwoRankFP16M0Map = {
    {128,
     {{-1, 11264, -1, 2147483647, -1, 5632},
      {-1, 11264, -1, 8704, 5632, 6656},
      {-1, 11264, -1, 9728, 6656, 7680},
      {11264, 13312, -1, 2147483647, -1, 1536},
      {13312, 2147483647, 3072, 2147483647, -1, 1536},
      {11264, 2147483647, -1, 2147483647, 1536, 7680},
      {-1, 2147483647, -1, 8704, 7680, 8704},
      {-1, 2147483647, -1, 7680, 8704, 9728},
      {-1, 4608, -1, 8704, 9728, 2147483647},
      {4608, 2147483647, -1, 6656, 9728, 2147483647},
      {-1, 2560, 8704, 9728, 7680, 9216},
      {-1, 1536, 9728, 2147483647, 9728, 2147483647}}},
    {256,
     {{-1, 11264, 8704, 2147483647, 5632, 6656},
      {-1, 11264, 9728, 2147483647, 6656, 7680},
      {13312, 2147483647, -1, 3072, -1, 1536},
      {-1, 2147483647, 7680, 8704, 8704, 9728},
      {4608, 2147483647, 6656, 8704, 9728, 2147483647},
      {-1, 2560, 8704, 9728, 9216, 2147483647},
      {2560, 2147483647, 8704, 9728, 7680, 2147483647},
      {-1, 2147483647, 9728, 2147483647, 7680, 9728},
      {1536, 2147483647, 9728, 2147483647, 9728, 2147483647}}}};

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallTwoRankFP16PvalueMap = {
    {8, {{-1, 2560, -1, 1536, -1, 1536}, {2560, 2147483647, -1, 1536, 2560, 3584}}},
    {4,
     {{-1, 2560, 1536, 6144, -1, 1536},
      {-1, 2560, -1, 2304, 1536, 2560},
      {1536, 3584, -1, 1536, 4608, 7680},
      {3584, 4608, -1, 1536, 4608, 8192},
      {4608, 11264, -1, 1536, 7680, 2147483647},
      {11264, 2147483647, -1, 4608, 4608, 7680}}},
    {5,
     {{-1, 2560, 6144, 2147483647, -1, 1536},
      {1536, 2560, 4608, 2147483647, 1536, 2560},
      {2560, 11264, 1536, 2304, -1, 4608},
      {2560, 9728, 2304, 2816, -1, 4608},
      {2560, 7680, 2816, 2147483647, -1, 4608},
      {7680, 2147483647, 2816, 9728, -1, 4608},
      {7680, 2147483647, 9728, 2147483647, -1, 2560},
      {4608, 11264, -1, 1536, 4608, 7680},
      {11264, 2147483647, 4608, 8704, 4608, 7680},
      {11264, 2147483647, -1, 1536, 7680, 2147483647}}},
    {2,
     {{-1, 2560, 2304, 4608, 1536, 2560},
      {-1, 1536, 4608, 2147483647, 1536, 2560},
      {-1, 2560, -1, 4608, 3584, 4608},
      {1536, 2560, 4608, 2147483647, 3584, 4608},
      {-1, 1536, -1, 1536, 4608, 2147483647},
      {-1, 1536, 1536, 7680, 4608, 7680},
      {-1, 1536, 7680, 2147483647, 5632, 2147483647},
      {1536, 4608, 1536, 2147483647, 4608, 5632},
      {1536, 4608, 1536, 2147483647, 6656, 2147483647},
      {4608, 8704, 3840, 2147483647, 4608, 2147483647},
      {8704, 11264, 8704, 2147483647, 4608, 2147483647},
      {11264, 2147483647, 8704, 2147483647, 4608, 7680}}},
    {3,
     {{-1, 2560, -1, 2147483647, 2560, 3584},
      {1536, 3584, -1, 1536, 7680, 2147483647},
      {3584, 4608, -1, 1536, 8192, 2147483647},
      {1536, 4608, 1536, 2147483647, 5632, 6656},
      {4608, 11264, 1536, 3840, 4608, 2147483647},
      {8704, 11264, 3840, 8704, 4608, 2147483647},
      {11264, 2147483647, 1536, 2147483647, 7680, 2147483647}}},
    {1,
     {{-1, 1536, 4608, 2147483647, 3584, 4608},
      {-1, 1536, 1536, 7680, 7680, 2147483647},
      {-1, 1536, 7680, 2147483647, 4608, 5632}}},
    {10, {{2560, 2147483647, -1, 1536, -1, 2560}}},
    {6,
     {{2560, 2147483647, -1, 1536, 3584, 4608},
      {11264, 2147483647, 1536, 2304, -1, 4608},
      {7680, 2147483647, 9728, 2147483647, 2560, 4608}}},
    {7, {{9728, 2147483647, 2304, 2816, -1, 4608}}}};

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallTwoRankFP16UbmovenumMap = {
    {40, {{-1, 3584, -1, 3328, -1, 1536}, {15360, 2147483647, 2304, 5632, -1, 1536}}},
    {30,
     {{-1, 3584, 3328, 6144, -1, 1536},
      {-1, 3584, -1, 2147483647, 1536, 2560},
      {-1, 3584, -1, 5632, 2560, 3584},
      {-1, 1536, -1, 6144, 3584, 4608},
      {1536, 3584, -1, 5632, 3584, 4608},
      {-1, 1536, -1, 6656, 4608, 7680},
      {1536, 3584, -1, 5632, 4608, 7680},
      {3584, 15360, -1, 5632, -1, 1536},
      {15360, 2147483647, -1, 2304, -1, 1536},
      {3584, 2147483647, -1, 5632, 1536, 7680},
      {3584, 2147483647, 5632, 6656, -1, 1536},
      {-1, 2147483647, -1, 5632, 7680, 2147483647}}},
    {20,
     {{-1, 3584, 6144, 9728, -1, 1536},
      {-1, 3584, 5632, 2147483647, 2560, 3584},
      {-1, 1536, 6144, 2147483647, 3584, 4608},
      {1536, 3584, 5632, 2147483647, 3584, 4608},
      {-1, 1536, 6656, 2147483647, 4608, 7680},
      {1536, 3584, 5632, 2147483647, 4608, 7680},
      {3584, 2147483647, 6656, 7680, -1, 1536},
      {3584, 2147483647, 5632, 7680, 1536, 7680},
      {3584, 6656, 7680, 8704, -1, 7680},
      {-1, 11264, 5632, 9728, 7680, 8704},
      {11264, 2147483647, 5632, 7680, 7680, 8704},
      {-1, 8704, 9728, 2147483647, 7680, 8704},
      {-1, 11264, 5632, 2147483647, 8704, 9728},
      {15360, 2147483647, 5632, 2147483647, 8704, 9728},
      {-1, 2147483647, 5632, 2147483647, 9728, 2147483647}}},
    {16,
     {{-1, 3584, 9728, 2147483647, -1, 1536},
      {6656, 2147483647, 7680, 8704, -1, 7680},
      {3584, 2147483647, 8704, 2147483647, -1, 7680},
      {11264, 2147483647, 7680, 9728, 7680, 8704},
      {8704, 2147483647, 9728, 2147483647, 7680, 8704},
      {11264, 15360, 5632, 2147483647, 8704, 9728}}}};

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallFourRankFP16M0Map = {
    {128,
     {{-1, 6656, -1, 2147483647, -1, 6656},
      {6656, 2147483647, -1, 8704, -1, 6656},
      {6656, 2147483647, 8704, 9728, -1, 3584},
      {6656, 2147483647, 9728, 2147483647, -1, 6656},
      {-1, 2147483647, -1, 4608, 6656, 2147483647},
      {-1, 9728, 4608, 5632, 6656, 2147483647},
      {-1, 7680, 5632, 6656, 6656, 2147483647},
      {-1, 2560, 6656, 7680, 6656, 2147483647},
      {-1, 2560, 7680, 2147483647, 6656, 7680}}},
    {256,
     {{6656, 2147483647, 8704, 9728, 3584, 6656},
      {9728, 2147483647, 4608, 5632, 6656, 2147483647},
      {7680, 2147483647, 5632, 6656, 6656, 2147483647},
      {2560, 2147483647, 6656, 7680, 6656, 2147483647},
      {-1, 2560, 7680, 2147483647, 7680, 2147483647},
      {2560, 2147483647, 7680, 2147483647, 6656, 2147483647}}}};

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallFourRankFP16PvalueMap = {
    {8, {{-1, 2560, -1, 1536, -1, 1536}}},
    {4,
     {{-1, 2560, 1536, 2816, -1, 1536},
      {-1, 2560, -1, 2816, 1536, 2560},
      {1536, 2560, -1, 2560, 3584, 4608},
      {13312, 2147483647, 6656, 8704, -1, 4608},
      {3584, 2147483647, 1536, 3840, 4608, 5632},
      {13312, 2147483647, 3840, 2147483647, 4608, 5632},
      {3584, 2147483647, -1, 5632, 5632, 7680},
      {3584, 15360, -1, 1536, 7680, 2147483647},
      {15360, 2147483647, -1, 1536, 9728, 2147483647}}},
    {5,
     {{-1, 2560, 2816, 2147483647, -1, 1536},
      {1536, 2560, 2816, 2147483647, 1536, 2560},
      {2560, 2147483647, -1, 2816, 2560, 4608},
      {2560, 2147483647, 2816, 6656, -1, 4608},
      {2560, 3584, 6656, 9728, -1, 4608},
      {3584, 9728, 6656, 2147483647, -1, 2560},
      {9728, 13312, 6656, 8704, -1, 4608},
      {15360, 2147483647, -1, 1536, 7680, 9728}}},
    {2,
     {{-1, 1536, 2816, 2147483647, 1536, 2560},
      {-1, 1536, -1, 2560, 3584, 4608},
      {2560, 3584, 9728, 2147483647, -1, 4608},
      {-1, 1536, -1, 1536, 4608, 2147483647},
      {-1, 1536, 1536, 4352, 4608, 8704},
      {1536, 3584, 1536, 7680, 4608, 2147483647},
      {2560, 3584, 9728, 2147483647, 4608, 2147483647},
      {3584, 13312, 3840, 2147483647, 4608, 5632},
      {3584, 2147483647, 8704, 2147483647, 5632, 7680}}},
    {3,
     {{-1, 2560, -1, 2560, 2560, 3584},
      {-1, 2560, 2560, 2147483647, 2560, 4608},
      {3584, 9728, 6656, 2147483647, 2560, 4608},
      {-1, 1536, 9728, 2147483647, 4608, 5632},
      {-1, 1536, 7680, 2147483647, 5632, 2147483647},
      {1536, 3584, -1, 1536, 4608, 2147483647},
      {1536, 3584, 7680, 9728, 4608, 2147483647},
      {1536, 2560, 9728, 2147483647, 4608, 2147483647},
      {3584, 2147483647, 5632, 8704, 5632, 7680},
      {3584, 2147483647, 1536, 2147483647, 7680, 2147483647}}},
    {10, {{2560, 2147483647, -1, 2816, -1, 2560}}},
    {9, {{9728, 2147483647, 8704, 2147483647, -1, 1536}}},
    {6, {{9728, 2147483647, 8704, 2147483647, 1536, 4608}, {3584, 2147483647, -1, 1536, 4608, 5632}}},
    {1,
     {{-1, 1536, 1536, 4352, 8704, 2147483647},
      {-1, 1536, 4352, 9728, 4608, 5632},
      {-1, 1536, 4352, 7680, 5632, 2147483647}}}};

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallFourRankFP16UbmovenumMap = {
    {30,
     {{-1, 2560, -1, 3840, -1, 1536},
      {2560, 13312, 1536, 3840, -1, 1536},
      {13312, 15360, -1, 3072, -1, 1536},
      {15360, 2147483647, -1, 4352, -1, 1536},
      {-1, 2147483647, 1536, 3328, 1536, 2560},
      {-1, 2147483647, -1, 2816, 2560, 2147483647}}},
    {40, {{2560, 13312, -1, 1536, -1, 1536}, {-1, 2147483647, -1, 1536, 1536, 2560}}},
    {20,
     {{-1, 1536, 3840, 2147483647, -1, 1536},
      {9728, 13312, 3840, 5632, -1, 1536},
      {13312, 15360, 3072, 6144, -1, 1536},
      {15360, 2147483647, 4352, 2147483647, -1, 1536},
      {-1, 2147483647, 3328, 4608, 1536, 2560},
      {-1, 2147483647, 2816, 4608, 2560, 2147483647}}},
    {16,
     {{1536, 9728, 3840, 2147483647, -1, 1536},
      {9728, 13312, 5632, 2147483647, -1, 1536},
      {13312, 15360, 6144, 2147483647, -1, 1536},
      {-1, 2147483647, 4608, 2147483647, 1536, 2147483647}}}};

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallEightRankFP16UbmovenumMap = {
    {70, {{-1, 1536, -1, 2304, -1, 1536}}},
    {30,
     {{-1, 1536, 2304, 2816, -1, 1536},
      {1536, 2147483647, -1, 1536, -1, 1536},
      {-1, 1536, -1, 1536, 4096, 2147483647},
      {1536, 2147483647, -1, 1536, 1536, 2147483647},
      {-1, 1536, 1536, 2304, 1536, 5120}}},
    {20,
     {{1536, 8704, 1536, 2816, -1, 1536},
      {-1, 2560, 3840, 2147483647, -1, 1536},
      {15360, 2147483647, 1536, 6656, -1, 1536},
      {8704, 15360, 6656, 2147483647, -1, 1536},
      {-1, 1536, 1536, 2304, 5120, 2147483647},
      {1536, 2147483647, 1536, 2304, 1536, 2147483647},
      {-1, 3584, 7680, 2147483647, 6656, 2147483647}}},
    {16,
     {{-1, 2560, 2816, 3840, -1, 1536},
      {2560, 8704, 2816, 2147483647, -1, 1536},
      {8704, 15360, 1536, 6656, -1, 1536},
      {15360, 2147483647, 6656, 2147483647, -1, 1536},
      {-1, 3584, 2304, 7680, 1536, 2147483647},
      {-1, 3584, 7680, 2147483647, 1536, 6656},
      {3584, 2147483647, 2304, 2147483647, 1536, 2147483647}}},
    {80, {{-1, 1536, -1, 1536, 1536, 4096}}}};

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallEightRankFP16PvalueMap = {
    {2,
     {{-1, 1536, -1, 2147483647, -1, 1536},
      {1536, 2560, -1, 5632, -1, 1536},
      {-1, 1536, 2816, 2147483647, 1536, 2560},
      {-1, 1536, -1, 5632, 2560, 3584},
      {1536, 2560, -1, 3840, 2560, 3584},
      {-1, 2560, -1, 1536, 3584, 2147483647},
      {2560, 3584, -1, 3840, 9728, 2147483647},
      {2560, 3584, 3840, 7680, 2560, 2147483647},
      {2560, 3584, 7680, 2147483647, 2560, 7680},
      {5632, 2147483647, 5632, 2147483647, 4608, 6656},
      {3584, 4608, -1, 3328, 6656, 2147483647},
      {8704, 9728, 3328, 2147483647, 6656, 2147483647}}},
    {7,
     {{1536, 2560, 5632, 2147483647, -1, 1536},
      {5632, 9728, 6656, 2147483647, -1, 1536},
      {9728, 11264, 3328, 2147483647, -1, 2560},
      {11264, 2147483647, 4608, 8704, -1, 1536}}},
    {5,
     {{2560, 5632, -1, 2147483647, -1, 1536},
      {5632, 9728, -1, 6656, -1, 1536},
      {-1, 1536, -1, 2816, 1536, 2560},
      {1536, 5632, -1, 2147483647, 1536, 2560},
      {5632, 8704, -1, 6656, 1536, 2560},
      {8704, 9728, 2560, 2147483647, 1536, 2560},
      {9728, 11264, -1, 3328, -1, 1536},
      {15360, 2147483647, -1, 4608, -1, 1536},
      {5632, 6656, -1, 2147483647, 2560, 4608}}},
    {4,
     {{5632, 8704, 6656, 2147483647, 1536, 2560},
      {11264, 2147483647, 8704, 2147483647, 1536, 2560},
      {3584, 5632, -1, 2816, 2560, 4608},
      {5632, 2147483647, -1, 5632, 4608, 6656},
      {9728, 2147483647, -1, 3328, 6656, 2147483647}}},
    {10,
     {{8704, 9728, -1, 2560, 1536, 2560},
      {9728, 11264, -1, 3328, 1536, 2560},
      {11264, 15360, -1, 4608, -1, 2560},
      {15360, 2147483647, -1, 4608, 1536, 2560}}},
    {9, {{11264, 2147483647, 4608, 8704, 1536, 2560}}},
    {6, {{11264, 2147483647, 8704, 2147483647, -1, 1536}}},
    {3,
     {{-1, 1536, 5632, 2147483647, 2560, 3584},
      {1536, 2560, 3840, 2147483647, 2560, 3584},
      {-1, 2560, 6656, 2147483647, 3584, 2147483647},
      {2560, 3584, -1, 3840, 2560, 8704},
      {2560, 3584, 7680, 2147483647, 7680, 2147483647},
      {3584, 5632, 2816, 2147483647, 2560, 4608},
      {3584, 5632, -1, 2147483647, 4608, 6656},
      {6656, 2147483647, -1, 2147483647, 2560, 4608},
      {4608, 9728, -1, 3328, 6656, 2147483647},
      {3584, 8704, 3328, 2147483647, 6656, 2147483647},
      {9728, 2147483647, 3328, 2147483647, 6656, 2147483647}}},
    {1, {{-1, 2560, 1536, 6656, 3584, 2147483647}, {2560, 3584, -1, 3840, 8704, 9728}}}};

static std::map<int, std::vector<std::vector<int>>> g_matmulalltoallEightRankFP16M0Map = {
    {256, {{-1, 2147483647, -1, 2147483647, -1, 1536}, {-1, 2147483647, -1, 2147483647, 2560, 2147483647}}},
    {128, {{-1, 2147483647, -1, 2147483647, 1536, 2560}}}};

bool MatmulAlltoAllTiling910B::IsCapable()
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OP_LOGE(opName_, "fail to get platform info"), return false);
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;
    std::string socVersionStr;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersionStr);
    OP_LOGD(opName_, "Current SocVersion is : %s", socVersionStr.c_str());
    QuantMode mode = MatmulAlltoAllTilingUtil::GetQuantMode(context_, opName_);
    if (socVersionStr == "Ascend910B") {
        OP_LOGI(opName_, "Start with MatmulAllToAll tiling.");
        return true;
    }
    OP_LOGD(opName_, "Skip MatmulAlltoAllTiling910b tiling when the SocVersion is unsupported.");
    return false;
}

/**
 * @brief 校验attrs信息
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTiling910B::CheckAndSetAttrsInfo(MatmulAlltoAllInfo &info)
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Failed to get attrs."), return ge::GRAPH_FAILED);

    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    const char *commMode = attrs->GetAttrPointer<char>(ATTR_COMM_MODE_INDEX);
    const int64_t *x1_quant_mode = attrs->GetAttrPointer<int64_t>(ATTR_X1_QUANTMODE_INDEX);
    const int64_t *x2_quant_mode = attrs->GetAttrPointer<int64_t>(ATTR_X2_QUANTMODE_INDEX);
    // 判断为空或者空字符串
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "group"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group[0] == '\0', OP_LOGE_WITH_INVALID_INPUT(opName_, "group"), return ge::GRAPH_FAILED);

    // A2的comm_mode只支持aiv
    OP_TILING_CHECK(commMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "comm_mode"), return ge::GRAPH_FAILED);
    const size_t maxLength = 4UL;
    OP_TILING_CHECK((strncmp(commMode, "aiv", maxLength) != 0),
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "comm_mode", commMode, "aiv"), return ge::GRAPH_FAILED);

    info.worldSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);
    OP_TILING_CHECK(SUPPORT_RANK_SIZE_910B.find(info.worldSize) == SUPPORT_RANK_SIZE_910B.end(),
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "world_size", std::to_string(info.worldSize).c_str(), "2/4/8"),
                    return ge::GRAPH_FAILED);

    const bool *isTransX1 = attrs->GetAttrPointer<bool>(ATTR_X1_TRANSPOSE_INDEX);
    bool x1TransposeFlag = (isTransX1 != nullptr) ? *isTransX1 : false;
    OP_TILING_CHECK(x1TransposeFlag, OP_LOGE_WITH_INVALID_ATTR(opName_, "transpose_x1", "true", "false"),
                    return ge::GRAPH_FAILED);

    const bool *isTransX2 = attrs->GetAttrPointer<bool>(ATTR_X2_TRANSPOSE_INDEX);
    needTransX2 = (isTransX2 != nullptr) ? *isTransX2 : false;
    quantType = static_cast<uint32_t>((*x1_quant_mode) * X1_QUANT_SIGN + (*x2_quant_mode));

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 非量化场景校验参数的DType
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTiling910B::CheckTensorDataType(MatmulAlltoAllInfo &info)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context_->GetInputDesc(INPUT_X1_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "x1"), return ge::GRAPH_FAILED);
    auto x2TensorDesc = context_->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "x2"), return ge::GRAPH_FAILED);
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "y"), return ge::GRAPH_FAILED);

    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    ge::DataType yDtype = yDesc->GetDataType();
    auto biasTensorDesc = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX);

    if (x1Dtype == ge::DT_INT8) {
        // 校验 scale 张量不为空，量化模式
        auto x1ScaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
        auto x2ScaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
        OP_TILING_CHECK((x1ScaleTensorDesc == nullptr || x2ScaleTensorDesc == nullptr || biasTensorDesc == nullptr),
                        OP_LOGE_WITH_INVALID_INPUT(opName_, "scale/bias"), return ge::GRAPH_FAILED);
        ge::DataType x1ScaleDtype = x1ScaleTensorDesc->GetDataType();
        ge::DataType x2ScaleDtype = x2ScaleTensorDesc->GetDataType();
        OP_TILING_CHECK(
            (x1ScaleDtype != ge::DT_FLOAT || x2ScaleDtype != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE(
                opName_, "x1Scale/x2Scale",
                (Ops::Base::ToString(x1ScaleDtype) + "/" + Ops::Base::ToString(x2ScaleDtype)).c_str(), "FLOAT32"),
            return ge::GRAPH_FAILED);

        OP_TILING_CHECK(
            (quantType != SUPPORT_QUANT_MODE),
            OP_LOGE_WITH_INVALID_ATTR(
                opName_, "quant_mode",
                (std::to_string(quantType / X1_QUANT_SIGN) + "/" + std::to_string(quantType % X1_QUANT_SIGN)).c_str(),
                "x1=PerToken(3)/x2=PerChannel(2)"),
            return ge::GRAPH_FAILED);
    }

    // 校验 bias 数据类型（如果存在）
    if (biasTensorDesc != nullptr) {
        hasBias = true;
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        if (biasDtype == ge::DT_FLOAT16) {
            quantBiasType = TILINGKEY_TPL_FP16;
        } else if (biasDtype == ge::DT_FLOAT) {
            quantBiasType = TILINGKEY_TPL_FP32;
        } else if (biasDtype == ge::DT_BF16) {
            quantBiasType = TILINGKEY_TPL_BF16;
        }
        vector<uint32_t> paramsType = {x1Dtype, x2Dtype, biasDtype, yDtype};

        for (uint32_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITH_BIAS.size(); kindsId++) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITH_BIAS[kindsId], COUNT_PARAMS_WITH_BIAS)) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "x1,x2,bias,y",
            (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype) + "," + Ops::Base::ToString(biasDtype) +
             "," + Ops::Base::ToString(yDtype))
                .c_str(),
            "The dtypes of x1, x2, bias and y must be one of the supported combinations.");
        return ge::GRAPH_FAILED;
    } else {
        vector<uint32_t> paramsType = {x1Dtype, x2Dtype, yDtype};

        for (uint32_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITHOUT_BIAS.size(); kindsId++) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITHOUT_BIAS[kindsId], COUNT_PARAMS_WITHOUT_BIAS)) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "x1,x2,y",
            (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype) + "," + Ops::Base::ToString(yDtype))
                .c_str(),
            "The dtypes of x1, x2 and y must be one of the supported combinations.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckMatrixMulShapes(const gert::TilingContext *context, const char *opName, uint64_t worldSize)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);

    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    uint64_t yDim0 = yShape->GetStorageShape().GetDim(0);
    uint64_t yDim1 = yShape->GetStorageShape().GetDim(1);

    bool x2TransFlag = false;
    const bool *isTransX2 = context->GetAttrs()->GetAttrPointer<bool>(ATTR_X2_TRANSPOSE_INDEX);
    if (isTransX2) {
        x2TransFlag = *isTransX2;
    }

    if (x2TransFlag) {
        OP_TILING_CHECK((x1Dim1 != x2Dim1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            opName, "x1Dim1,x2Dim1", (std::to_string(x1Dim1) + "," + std::to_string(x2Dim1)).c_str(),
                            "When x2 is transposed, the second dim of x1 must be equal to the second dim of x2"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            ((x1Dim0 * worldSize != yDim0) || (x2Dim0 != yDim1 * worldSize)),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "x1,x2,y",
                (std::string("x1Dim0=") + std::to_string(x1Dim0) + " yDim0=" + std::to_string(yDim0) + " x2Dim0=" +
                 std::to_string(x2Dim0) + " yDim1=" + std::to_string(yDim1) + " worldSize=" + std::to_string(worldSize))
                    .c_str(),
                "When x2 is transposed, x1Dim0*worldSize must equal yDim0 and x2Dim0 must equal yDim1*worldSize"),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((x1Dim1 != x2Dim0),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            opName, "x1Dim1,x2Dim0", (std::to_string(x1Dim1) + "," + std::to_string(x2Dim0)).c_str(),
                            "The second dim of x1 must be equal to the first dim of x2"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(((x1Dim0 * worldSize != yDim0) || (x2Dim1 != yDim1 * worldSize)),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            opName, "x1,x2,y",
                            (std::string("x1Dim0=") + std::to_string(x1Dim0) + " yDim0=" + std::to_string(yDim0) +
                             " x2Dim1=" + std::to_string(x2Dim1) + " yDim1=" + std::to_string(yDim1) +
                             " worldSize=" + std::to_string(worldSize))
                                .c_str(),
                            "The value of x1Dim0*worldSize must equal yDim0 and x2Dim1 must equal yDim1*worldSize"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入的shape信息
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTiling910B::CheckShapeInfo(MatmulAlltoAllInfo &info)
{
    ge::graphStatus status;

    // 校验输入Input Shape是否为空
    const gert::StorageShape *x1Shape = context_->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context_->GetInputShape(INPUT_X2_INDEX);
    OP_TILING_CHECK((x1Shape == nullptr) || (x2Shape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "x1/x2"),
                    return ge::GRAPH_FAILED);

    // 校验维度数目是否合法
    uint64_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    uint64_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        (x1DimNum != 2 || x2DimNum != 2),
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "x1/x2",
                                     (std::to_string(x1DimNum) + "D, " + std::to_string(x2DimNum) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    // 校验输出
    const gert::StorageShape *yShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "y"), return ge::GRAPH_FAILED);
    uint64_t yDimNum = yShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((yDimNum != 2),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "y", (std::to_string(yDimNum) + "D").c_str(), "2D"),
                    return ge::GRAPH_FAILED);

    // 校验shape的维度矩阵是否合法
    status = CheckMatrixMulShapes(context_, opName_, info.worldSize);
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验量化场景中scale的shape信息
    auto x1TensorDesc = context_->GetInputDesc(INPUT_X1_INDEX);
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    info.M = x1Shape->GetStorageShape().GetDim(0);
    info.K = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    info.N = (info.K == x2Dim0) ? x2Dim1 : x2Dim0;
    if (x1Dtype == ge::DT_INT8) {
        const gert::StorageShape *x1ScaleShape = context_->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
        const gert::StorageShape *x2ScaleShape = context_->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
        uint64_t x1ScaleShapeDimNum = x1ScaleShape->GetStorageShape().GetDimNum();
        uint64_t x2ScaleShapeDimNum = x2ScaleShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            (x1ScaleShapeDimNum != 1 || x2ScaleShapeDimNum != 1),
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                opName_, "x1Scale/x2Scale",
                (std::to_string(x1ScaleShapeDimNum) + "D/" + std::to_string(x2ScaleShapeDimNum) + "D").c_str(), "1D"),
            return ge::GRAPH_FAILED);
        uint64_t x1ScaleDim0 = x1ScaleShape->GetStorageShape().GetDim(0);
        uint64_t x2ScaleDim0 = x2ScaleShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK((x1ScaleDim0 != info.M),
                        OP_LOGE_FOR_INVALID_VALUE(opName_, "x1Scale", std::to_string(x1ScaleDim0).c_str(),
                                                  std::to_string(info.M).c_str()),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((x2ScaleDim0 != info.N),
                        OP_LOGE_FOR_INVALID_VALUE(opName_, "x2Scale", std::to_string(x2ScaleDim0).c_str(),
                                                  std::to_string(info.N).c_str()),
                        return ge::GRAPH_FAILED);
    }
    const gert::StorageShape *biasShape = context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (biasShape != nullptr) {
        uint64_t biasShapeDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(
            (biasShapeDimNum != 1),
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "bias", (std::to_string(biasShapeDimNum) + "D").c_str(), "1D"),
            return ge::GRAPH_FAILED);
        uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK((biasDim0 != info.N),
                        OP_LOGE_FOR_INVALID_VALUE(opName_, "bias", std::to_string(biasDim0).c_str(),
                                                  std::to_string(info.N).c_str()),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(info.M == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1", std::to_string(info.M).c_str(),
                                                          "The value of dim 0 (m) of x1 cannot be 0"),
                    return ge::GRAPH_FAILED);
    // 校验K,K的范围应该在[1, 65535]
    OP_TILING_CHECK(info.K > K_MAX_VALUE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1", std::to_string(info.K).c_str(),
                                                          "The value of k dim of x1 must not exceed 65535"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.N == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x2", std::to_string(info.N).c_str(),
                                                          "The value of N of x2 cannot be 0"),
                    return ge::GRAPH_FAILED);
    // CCL buffer数据区: m0 * pValue * N * sizeof(element) 必须小于 180MB / MAX_BLOCK_COUNT
    int32_t maxN = (MAX_BUFF_BYTES - FLAG_BUFF_BYTES) / MAX_BLOCK_COUNT / ELEMENT_SIZE / DEFAULT_ROW;
    OP_TILING_CHECK(info.N > maxN,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x2", std::to_string(info.N),
                                                          "The value of N must not exceed " + std::to_string(maxN)),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTiling910B::CheckOpInputInfo(MatmulAlltoAllInfo &info)
{
    OP_TILING_CHECK(CheckAndSetAttrsInfo(info) != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Tiling check Attrs failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDataType(info) != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Tiling check Dtype failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckShapeInfo(info) != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Tiling check shape failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int32_t MatmulAlltoAllTiling910B::GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                                              std::map<int, std::vector<std::vector<int>>> conditionMap)
{
    int32_t value = defaultValue;
    for (auto &item : conditionMap) {
        for (auto &condition : item.second) {
            bool inRange = m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                           k > condition[CONDITION_K_ST] && k <= condition[CONDITION_K_END] &&
                           n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
            if (inRange) {
                return item.first;
            }
        }
    }
    return value;
}

void MatmulAlltoAllTiling910B::CalTilingParam(CoCTiling &cocTilingData,
                                              const std::map<int *, MatmulAlltoAllTilingValue> &TilingParamMap,
                                              MatmulAlltoAllInfo &info)
{
    int32_t m = info.M;
    int32_t k = info.K;
    int32_t n = info.N;

    for (auto &item : TilingParamMap) {
        auto value = item.second.value;
        auto conditionMap = item.second.conditionMap;
        if (!conditionMap.empty()) {
            *item.first = GetValueFromMKNConditionMap(m, k, n, value, conditionMap);
        } else if (value != -1) {
            *item.first = value;
        }
    }
    cocTilingData.ubMoveNum = cocTilingData.ubMoveNum * HALF_KBYTE;
    if (cocTilingData.m0 >= DEFAULT_ROW) {
        cocTilingData.k0 = DEFAULT_COL;
        cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
    }
    tileM0 = cocTilingData.m0;
    tileN0 = cocTilingData.n0;
}

void MatmulAlltoAllTiling910B::DoTwoRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info)
{
    std::map<int *, MatmulAlltoAllTilingValue> TilingParamMap;
    TilingParamMap[&cocTilingData.m0] =
        MatmulAlltoAllTilingValue(MATMULALLTOALL_TWO_RANK_FP16_M0_DEFAULT, g_matmulalltoallTwoRankFP16M0Map);
    TilingParamMap[&cocTilingData.pValue] =
        MatmulAlltoAllTilingValue(MATMULALLTOALL_TWO_RANK_FP16_PVALUE_DEFAULT, g_matmulalltoallTwoRankFP16PvalueMap);
    TilingParamMap[&cocTilingData.ubMoveNum] = MatmulAlltoAllTilingValue(MATMULALLTOALL_TWO_RANK_FP16_UBMOVENUM_DEFAULT,
                                                                         g_matmulalltoallTwoRankFP16UbmovenumMap);
    CalTilingParam(cocTilingData, TilingParamMap, info);
    TilingParamDeal(cocTilingData, info);
}

void MatmulAlltoAllTiling910B::DoFourRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info)
{
    std::map<int *, MatmulAlltoAllTilingValue> TilingParamMap;
    TilingParamMap[&cocTilingData.m0] =
        MatmulAlltoAllTilingValue(MATMULALLTOALL_FOUR_RANK_FP16_M0_DEFAULT, g_matmulalltoallFourRankFP16M0Map);
    TilingParamMap[&cocTilingData.pValue] =
        MatmulAlltoAllTilingValue(MATMULALLTOALL_FOUR_RANK_FP16_PVALUE_DEFAULT, g_matmulalltoallFourRankFP16PvalueMap);
    TilingParamMap[&cocTilingData.ubMoveNum] = MatmulAlltoAllTilingValue(
        MATMULALLTOALL_FOUR_RANK_FP16_UBMOVENUM_DEFAULT, g_matmulalltoallFourRankFP16UbmovenumMap);
    CalTilingParam(cocTilingData, TilingParamMap, info);
    TilingParamDeal(cocTilingData, info);
}

void MatmulAlltoAllTiling910B::DoEightRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info)
{
    std::map<int *, MatmulAlltoAllTilingValue> TilingParamMap;
    TilingParamMap[&cocTilingData.m0] =
        MatmulAlltoAllTilingValue(MATMULALLTOALL_EIGHT_RANK_FP16_M0_DEFAULT, g_matmulalltoallEightRankFP16M0Map);
    TilingParamMap[&cocTilingData.pValue] = MatmulAlltoAllTilingValue(MATMULALLTOALL_EIGHT_RANK_FP16_PVALUE_DEFAULT,
                                                                      g_matmulalltoallEightRankFP16PvalueMap);
    TilingParamMap[&cocTilingData.ubMoveNum] = MatmulAlltoAllTilingValue(
        MATMULALLTOALL_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT, g_matmulalltoallEightRankFP16UbmovenumMap);
    CalTilingParam(cocTilingData, TilingParamMap, info);
    TilingParamDeal(cocTilingData, info);
}

ge::graphStatus MatmulAlltoAllTiling910B::DoMmCommTiling(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info)
{
    if (info.worldSize == RANKSIZE_TWO) {
        DoTwoRankTiling(cocTilingData, info);
        return ge::GRAPH_SUCCESS;
    } else if (info.worldSize == RANKSIZE_FOUR) {
        DoFourRankTiling(cocTilingData, info);
        return ge::GRAPH_SUCCESS;
    }
    DoEightRankTiling(cocTilingData, info);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulAlltoAllTiling910B::DoOpTiling()
{
    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context_->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(opName_, ret);
    // 1. tilingData
    MatmulAlltoAllTilingData *tilingData = context_->GetTilingData<MatmulAlltoAllTilingData>();
    OPS_CHECK(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "tilingData is nullptr."),
              return ge::GRAPH_FAILED);
    OPS_LOG_I(opName_, "MatmulAlltoAll get tilingData.");
    MatmulAlltoAllInfo &info = tilingData->matmulAlltoAllInfo;
    OPS_LOG_I(opName_, "MatmulAlltoAll get tilingData info.");

    MC2_CHECK_LOG_RET(opName_, CheckOpInputInfo(info));
    MC2_CHECK_LOG_RET(opName_, DoMmCommTiling(tilingData->cocTiling, info));
    MC2_CHECK_LOG_RET(opName_, SetHcclTiling(tilingData));
    // 校验HCCL BUFF空间大小
    auto attrs = context_->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    uint64_t hcclBuffSize = 0ULL;
    auto cclRet = mc2tiling::GetCclBufferSize(group, &hcclBuffSize, opName_);
    if (cclRet == ge::GRAPH_SUCCESS) {
        OP_TILING_CHECK(hcclBuffSize < MAX_BUFF_BYTES,
                        OP_LOGE(opName_, "HCCL_BUFFSIZE (%lu Bytes) too small, min required %lu Bytes (%dMB)",
                                hcclBuffSize, MAX_BUFF_BYTES, MAX_BUFF_BYTES / MB_BYTES),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGW(opName_, "Can't get HCCL_BUFFSIZE, skip CCL buffer size validation.");
    }

    // 2. tilingkey
    SetTilingKey();
    OP_LOGD(context_->GetNodeName(), "tilingKey is %u.", tilingKey_);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    OPS_LOG_I(opName_, "Leave MatmulAlltoAll tiling func.");
    return ge::GRAPH_SUCCESS;
}

void MatmulAlltoAllTiling910B::SetTilingKey()
{
    tilingKey_ = GET_TPL_TILING_KEY(needTransX2, hasBias, quantBiasType, SOC_ASCEND910B);
}

/**
 * @brief 获取对应的tilingKey
 *
 * @return uint64_t tilingKey结果
 */
uint64_t MatmulAlltoAllTiling910B::GetTilingKey() const { return tilingKey_; }

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTiling910B::SetHcclTiling(MatmulAlltoAllTilingData *tilingData)
{
    auto attrs = context_->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    uint32_t opType = 18; // batch write=18,
    std::string algConfig = "MultiPut=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取额外申请的空间
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTiling910B::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Get workspace failed"), return ge::GRAPH_FAILED);
    size_t wsSize = SYSTEM_NEED_WORKSPACE;
    if (quantType == SUPPORT_QUANT_MODE) {
        wsSize += tileM0 * numBlocks * WORKSPACE_NUM * tileN0 * 4; // 4 is sizeof uint32_t
    }
    workspaces[0] = wsSize;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName_
 * @param tilingInfo
 */
void MatmulAlltoAllTiling910B::PrintMatmulAlltoAllTilingData(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info)
{
    OP_LOGD(opName_, "info.M: %u", info.M);
    OP_LOGD(opName_, "info.K: %u", info.K);
    OP_LOGD(opName_, "info.N: %u", info.N);
    OP_LOGD(opName_, "info.worldSize: %u", info.worldSize);
    OP_LOGD(opName_, "cocTilingData.m0: %u", cocTilingData.m0);
    OP_LOGD(opName_, "cocTilingData.k0: %u", cocTilingData.k0);
    OP_LOGD(opName_, "cocTilingData.n0: %u", cocTilingData.n0);
    OP_LOGD(opName_, "cocTilingData.pValue: %u", cocTilingData.pValue);
    OP_LOGD(opName_, "cocTilingData.ubMoveNum: %u", cocTilingData.ubMoveNum);
}

/**
 * @brief 保存tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus MatmulAlltoAllTiling910B::PostTiling()
{
    MatmulAlltoAllTilingData *outTilingData = context_->GetTilingData<MatmulAlltoAllTilingData>();
    context_->SetBlockDim(numBlocks);

    PrintMatmulAlltoAllTilingData(outTilingData->cocTiling, outTilingData->matmulAlltoAllInfo);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 构造函数，创建一个MatmulAllToAllTiling910B对象
 *
 * @param context
 */
MatmulAlltoAllTiling910B::MatmulAlltoAllTiling910B(gert::TilingContext *context)
    : MatmulAllToAllTilingBase(context)
{}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAlltoAll, MatmulAlltoAllTiling910B,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B), 0);
} // namespace MC2Tiling
