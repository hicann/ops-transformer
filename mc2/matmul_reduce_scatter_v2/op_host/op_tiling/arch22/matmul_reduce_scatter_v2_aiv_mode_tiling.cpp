/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_v2_aiv_mode_tiling.cpp
 * \brief
 */
#include "platform/platform_infos_def.h"
#include "vector"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#include "../../../op_kernel/matmul_reduce_scatter_v2_tiling_key.h"
#include "matmul_reduce_scatter_v2_aiv_mode_smallm_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace matmulReduceScatterV2_aivmode_tiling;
using namespace Mc2Tiling;
namespace {
const char *K_INNER_DEBUG = "MatmulReduceScatterV2AivMode Tiling Debug";
constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_IS_TRANS_A = 2;
constexpr uint32_t ATTR_IS_TRANS_B = 3;
constexpr uint64_t INIT_TILINGKEY = 10000U;
constexpr uint32_t A_INDEX = 0;
constexpr uint32_t B_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t X1_SCALE_INDEX = 3;
constexpr uint32_t X2_SCALE_INDEX = 4;
constexpr uint32_t C_INDEX = 0;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t USER_WORKSPACE_A2 = 1 * 1024 * 1024;         // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
constexpr uint64_t CCL_BUFFER_MIN_BYTES = 200ULL * 1024 * 1024; // 校验HCCL BUFF空间大小
constexpr uint32_t MB_BYTES = 1024 * 1024;
constexpr uint64_t TILINGKEY_BIAS = 1U;
constexpr uint64_t TILINGKEY_TRANS_A = 100U;
constexpr uint64_t TILINGKEY_TRANS_B = 10U;
constexpr uint64_t TILINGKEY_SMALL_M = 1000U;
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U;
constexpr uint32_t COUNT_PARAMS_WITH_BIAS = 4;    // [x1, x2, bias, y]
constexpr uint32_t COUNT_PARAMS_WITHOUT_BIAS = 3; // [x1, x2, y]
const std::vector<std::vector<uint32_t>> SUPPORTED_TYPES_WITH_BIAS = {
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT, ge::DT_BF16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16}};
const std::vector<std::vector<uint32_t>> SUPPORTED_TYPES_WITHOUT_BIAS = {
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_BF16},
    {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16}};

std::map<int, std::vector<std::vector<int>>> g_matmulReduceScatterA2FourRankINT8CodeMap = {
    {59785264, {{-1, 1536, -1, 7168, -1, 1536}}},
    {26226736, {{-1, 1536, 7168, 2147483647, -1, 1536}}},
    {160442416,
     {{1536, 2560, -1, 8704, -1, 1536},
      {3584, 4608, -1, 6656, 1536, 2560},
      {2560, 4608, -1, 6656, 2560, 3584},
      {4608, 6656, 6656, 7680, 1536, 3584}}},
    {160438320,
     {{1536, 2560, 8704, 2147483647, -1, 1536},
      {4608, 5632, 7680, 2147483647, -1, 2560},
      {6656, 7680, 8704, 2147483647, -1, 3584},
      {8704, 2147483647, 8704, 2147483647, -1, 3584},
      {-1, 2147483647, 8704, 9728, 5632, 7680},
      {-1, 5632, 8704, 2147483647, 8704, 13312}}},
    {26224688,
     {{2560, 4608, -1, 7680, -1, 1536}, {-1, 2560, -1, 2147483647, 2560, 3584}, {-1, 1536, 3584, 8704, 3584, 8704}}},
    {26222640, {{2560, 4608, 7680, 2147483647, -1, 1536}, {-1, 1536, -1, 8704, 13312, 2147483647}}},
    {193996848,
     {{-1, 3584, -1, 2147483647, 1536, 2560},
      {2560, 4608, 6656, 2147483647, 2560, 3584},
      {4608, 6656, 2560, 7680, -1, 1536}}},
    {26220592,
     {{3584, 4608, 6656, 2147483647, 1536, 2560},
      {7680, 8704, 8704, 2147483647, -1, 3584},
      {7680, 2147483647, 9728, 2147483647, 3584, 5632},
      {1536, 5632, 8704, 2147483647, 13312, 2147483647}}},
    {60829744, {{4608, 6656, -1, 2560, -1, 1536}}},
    {161490992,
     {{4608, 6656, -1, 6656, 1536, 3584}, {6656, 7680, 1536, 8704, -1, 3584}, {1536, 4608, -1, 7680, 3584, 8704}}},
    {160440368, {{5632, 6656, 7680, 2147483647, -1, 2560}, {4608, 5632, 7680, 2147483647, 2560, 3584}}},
    {193992752,
     {{5632, 6656, 7680, 2147483647, 2560, 3584},
      {-1, 2147483647, 8704, 9728, 3584, 5632},
      {-1, 6656, 8704, 9728, 7680, 8704},
      {-1, 6656, 9728, 2147483647, 3584, 8704}}},
    {62926896, {{6656, 7680, -1, 1536, -1, 3584}}},
    {161488944, {{7680, 2147483647, -1, 7680, -1, 3584}, {1536, 5632, 6656, 8704, 8704, 2147483647}}},
    {193994800, {{7680, 2147483647, 7680, 8704, -1, 3584}, {1536, 4608, 7680, 8704, 3584, 8704}}},
    {27275312, {{-1, 1536, -1, 3584, 3584, 8704}, {-1, 1536, -1, 8704, 8704, 13312}}},
    {195043376, {{4608, 2147483647, -1, 8704, 3584, 5632}}},
    {162541616, {{4608, 5632, -1, 8704, 5632, 8704}}},
    {163588144, {{5632, 2147483647, -1, 8704, 5632, 8704}}},
    {27269168, {{6656, 2147483647, 8704, 9728, 7680, 8704}}},
    {195041328, {{6656, 7680, 9728, 2147483647, 3584, 8704}}},
    {59775024, {{7680, 2147483647, 9728, 2147483647, 5632, 8704}}},
    {161486896, {{-1, 1536, 8704, 2147483647, 13312, 2147483647}}},
    {162539568, {{1536, 5632, -1, 6656, 8704, 2147483647}}},
    {32516144, {{5632, 2147483647, -1, 1536, 8704, 2147483647}}},
    {31467568, {{5632, 2147483647, 1536, 2560, 8704, 2147483647}}},
    {29370416, {{5632, 7680, 2560, 5632, 8704, 2147483647}}},
    {30418992, {{7680, 2147483647, 2560, 5632, 8704, 2147483647}}},
    {28321840, {{5632, 2147483647, 5632, 7680, 8704, 2147483647}}},
    {61872176, {{5632, 2147483647, 7680, 8704, 8704, 2147483647}}},
    {60823600, {{5632, 2147483647, 8704, 2147483647, 8704, 2147483647}}}};

std::map<int, std::vector<std::vector<int>>> g_matmulReduceScatterA2FourRankFP16CodeMap = {
    {26230832, {{-1, 1536, -1, 7168, -1, 1536}}},
    {27275312, {{-1, 1536, 7168, 2147483647, -1, 1536}, {-1, 1536, -1, 1536, 4608, 2147483647}}},
    {26226736, {{1536, 2560, -1, 2147483647, -1, 1536}, {-1, 2147483647, 4608, 6656, 4608, 5632}}},
    {26224688,
     {{2560, 4608, -1, 2147483647, -1, 1536},
      {4608, 6656, 2560, 2147483647, -1, 1536},
      {6656, 7680, -1, 8192, -1, 1536},
      {-1, 2560, -1, 2147483647, 1536, 4608},
      {3584, 4608, -1, 6656, 1536, 4608},
      {6656, 7680, -1, 4608, 1536, 4608},
      {7680, 2147483647, 1536, 3584, -1, 1536},
      {7680, 2147483647, 3584, 4608, -1, 4608},
      {7680, 8704, 4608, 2147483647, -1, 2560},
      {-1, 1536, 1536, 3584, 4608, 2147483647},
      {-1, 2147483647, 3584, 4608, 4608, 5632},
      {-1, 7680, 3584, 4608, 5632, 6656},
      {2560, 2147483647, 3584, 4608, 6656, 8704},
      {-1, 7680, 3584, 4608, 8704, 2147483647}}},
    {59779120, {{4608, 6656, -1, 2560, -1, 1536}}},
    {160438320,
     {{6656, 7680, 8192, 2147483647, -1, 1536},
      {6656, 7680, 4608, 2147483647, 1536, 4608},
      {-1, 8704, 6656, 2147483647, 4608, 6656},
      {-1, 2147483647, 4608, 5632, 6656, 7680},
      {-1, 5632, 5632, 6656, 8704, 2147483647},
      {-1, 4608, 6656, 2147483647, 8704, 2147483647},
      {5632, 2147483647, 5632, 7680, 8704, 2147483647}}},
    {26222640,
     {{2560, 3584, -1, 2147483647, 1536, 4608},
      {4608, 6656, 4608, 2147483647, 1536, 4608},
      {-1, 2560, 3584, 4608, 6656, 8704}}},
    {26220592,
     {{3584, 4608, 6656, 2147483647, 1536, 4608},
      {8704, 2147483647, 4608, 2147483647, -1, 2560},
      {7680, 8704, 4608, 2147483647, 2560, 4608},
      {-1, 2147483647, 4608, 6656, 5632, 6656}}},
    {27273264,
     {{4608, 6656, -1, 4608, 1536, 4608},
      {7680, 2147483647, 1536, 3584, 1536, 3584},
      {1536, 3584, -1, 3584, 4608, 2147483647}}},
    {28319792, {{7680, 8704, -1, 1536, -1, 3584}}},
    {28321840,
     {{7680, 8704, -1, 1536, 3584, 4608},
      {8704, 2147483647, -1, 1536, -1, 4608},
      {7680, 2147483647, 1536, 3584, 3584, 4608},
      {3584, 2147483647, -1, 3584, 4608, 7680},
      {3584, 2147483647, -1, 3584, 8704, 2147483647}}},
    {193992752,
     {{8704, 2147483647, 4608, 2147483647, 2560, 4608},
      {-1, 2147483647, 5632, 2147483647, 6656, 7680},
      {3584, 2147483647, 4608, 2147483647, 7680, 8704},
      {-1, 5632, 4608, 5632, 8704, 2147483647},
      {5632, 2147483647, 7680, 8704, 8704, 2147483647}}},
    {28323888, {{3584, 2147483647, -1, 3584, 7680, 8704}}},
    {161486896, {{7680, 2147483647, 3584, 4608, 5632, 6656}, {5632, 2147483647, 4608, 5632, 8704, 2147483647}}},
    {161488944, {{7680, 2147483647, 3584, 4608, 8704, 2147483647}}},
    {193990704,
     {{8704, 2147483647, 6656, 2147483647, 4608, 6656}, {5632, 2147483647, 8704, 2147483647, 8704, 2147483647}}},
    {160440368, {{-1, 3584, 4608, 2147483647, 7680, 8704}}},
    {160436272, {{4608, 5632, 6656, 2147483647, 8704, 2147483647}}}};

std::map<int, std::vector<std::vector<int>>> g_matmulReduceScatterA2EightRankINT8CodeMap = {
    {199245872, {{-1, 1536, -1, 4096, -1, 1536}}},
    {200294448, {{-1, 1536, 4096, 7168, -1, 1536}}},
    {164642864, {{-1, 1536, 7168, 8704, -1, 1536}}},
    {196108336, {{-1, 1536, 8704, 2147483647, -1, 1536}}},
    {59789360, {{-1, 1536, -1, 3584, 1536, 5632}}},
    {160448560, {{-1, 1536, 3584, 2147483647, 1536, 5632}}},
    {194007088, {{1536, 2147483647, -1, 4608, -1, 1536}}},
    {195055664, {{1536, 2147483647, -1, 4608, 1536, 2560}}},
    {197152816, {{1536, 2147483647, -1, 3584, 2560, 5632}, {3584, 5632, -1, 3584, 5632, 11264}}},
    {195051568,
     {{1536, 2147483647, 3584, 4608, 2560, 5632},
      {-1, 4608, 3584, 4608, 5632, 11264},
      {4608, 2147483647, 3584, 2147483647, 5632, 11264},
      {-1, 5632, 3584, 5632, 11264, 2147483647},
      {5632, 2147483647, 4608, 2147483647, 11264, 2147483647}}},
    {194002992,
     {{1536, 2147483647, 4608, 8704, -1, 5632},
      {1536, 2147483647, 8704, 2147483647, -1, 1536},
      {-1, 4608, 4608, 2147483647, 5632, 11264},
      {-1, 5632, 5632, 2147483647, 11264, 2147483647}}},
    {193998896, {{1536, 2147483647, 8704, 2147483647, 1536, 5632}}},
    {196104240, {{-1, 3584, -1, 3584, 5632, 11264}}},
    {198201392, {{5632, 8704, -1, 3584, 5632, 11264}}},
    {199249968, {{8704, 2147483647, -1, 3584, 5632, 11264}, {-1, 6656, -1, 3584, 11264, 13312}}},
    {200298544, {{-1, 6656, -1, 3584, 13312, 2147483647}, {6656, 9728, -1, 3584, 11264, 2147483647}}},
    {468734000, {{9728, 2147483647, -1, 3584, 11264, 2147483647}}},
    {196100144, {{5632, 2147483647, 3584, 4608, 11264, 2147483647}}}};

std::map<int, std::vector<std::vector<int>>> g_matmulReduceScatterA2EightRankFP16CodeMap = {
    {61890608, {{-1, 1536, -1, 4096, -1, 1536}}},
    {59793456, {{-1, 1536, 4096, 7168, -1, 1536}}},
    {63987760, {{-1, 1536, 7168, 8704, -1, 1536}}},
    {199254064, {{-1, 1536, 8704, 2147483647, -1, 1536}}},
    {26234928, {{-1, 1536, -1, 1536, 1536, 4096}}},
    {161501232, {{-1, 1536, -1, 1536, 4096, 7680}, {2560, 3584, 1536, 2560, 9728, 2147483647}}},
    {194007088,
     {{-1, 1536, 1536, 4608, 1536, 5632},
      {1536, 2147483647, 2560, 2147483647, -1, 5632},
      {1536, 2147483647, 2560, 4608, 5632, 6656},
      {1536, 2560, 2560, 5632, 6656, 7680},
      {4608, 2147483647, 2560, 2147483647, 6656, 7680},
      {-1, 2560, 2560, 5632, 8704, 2147483647},
      {-1, 1536, 5632, 9728, 7680, 13312},
      {2560, 9728, 2560, 2147483647, 7680, 13312},
      {9728, 2147483647, 2560, 4608, 7680, 13312},
      {2560, 6656, 2560, 8704, 13312, 2147483647}}},
    {160452656,
     {{-1, 1536, 4608, 2147483647, 1536, 5632},
      {-1, 1536, 1536, 5632, 5632, 7680},
      {2560, 4608, 2560, 2147483647, 6656, 7680},
      {-1, 1536, 9728, 2147483647, 7680, 8704}}},
    {194002992, {{-1, 1536, 5632, 2147483647, 5632, 7680}, {1536, 2147483647, 4608, 2147483647, 5632, 6656}}},
    {197152816, {{1536, 2147483647, -1, 1536, -1, 5632}, {2560, 6656, -1, 1536, 7680, 8704}}},
    {195055664,
     {{1536, 2147483647, 1536, 2560, -1, 6656},
      {1536, 7680, 1536, 2560, 6656, 7680},
      {-1, 2560, -1, 2560, 8704, 11264},
      {2560, 3584, 1536, 2560, 7680, 9728}}},
    {198201392,
     {{1536, 2147483647, -1, 1536, 5632, 6656},
      {2560, 2147483647, -1, 1536, 6656, 7680},
      {2560, 6656, -1, 1536, 8704, 2147483647}}},
    {196104240,
     {{1536, 2560, -1, 1536, 6656, 7680},
      {7680, 2147483647, 1536, 2560, 6656, 7680},
      {3584, 2147483647, 1536, 2560, 8704, 2147483647}}},
    {462432304, {{1536, 2560, 5632, 2147483647, 6656, 7680}}},
    {160448560,
     {{-1, 1536, -1, 5632, 7680, 8704},
      {1536, 2560, 6656, 7680, 7680, 2147483647},
      {1536, 2560, 7680, 2147483647, 9216, 2147483647},
      {9728, 2147483647, 4608, 2147483647, 7680, 13312}}},
    {462438448, {{1536, 2560, -1, 4608, 7680, 8704}}},
    {463482928, {{1536, 2560, 4608, 5632, 7680, 8704}, {1536, 2560, 5632, 6656, 7680, 2147483647}}},
    {27283504, {{-1, 2560, -1, 2560, 11264, 2147483647}}},
    {193996848, {{-1, 1536, 5632, 9728, 13312, 2147483647}}},
    {26218544, {{-1, 1536, 9728, 2147483647, 8704, 2147483647}}},
    {195051568, {{1536, 2560, 7680, 2147483647, 7680, 9216}}},
    {199249968, {{6656, 2147483647, -1, 1536, 7680, 9728}}},
    {200298544, {{6656, 2147483647, -1, 1536, 9728, 2147483647}}},
    {463487024, {{3584, 2147483647, 1536, 2560, 7680, 8704}}},
    {59785264, {{6656, 2147483647, 2560, 8704, 13312, 2147483647}}},
    {162545712, {{2560, 2147483647, 8704, 2147483647, 13312, 2147483647}}}};

std::map<int, std::vector<std::vector<int>>> g_matmulReduceScatterA3EightRankINT8CodeMap = {
    {161542192, {{-1, 5632, -1, 2560, -1, 1536}}},
    {194027568,
     {{-1, 5632, 2560, 2147483647, -1, 1536},
      {1536, 5632, -1, 2147483647, 1536, 2560},
      {-1, 5632, 4608, 2147483647, 2560, 3584},
      {-1, 2560, -1, 2147483647, 3584, 4608},
      {1536, 5632, 4608, 2147483647, 4608, 5632},
      {5632, 7680, 4608, 5632, -1, 1536},
      {7680, 2147483647, 5632, 9728, -1, 1536},
      {-1, 3584, 5632, 6656, 5632, 9728},
      {-1, 4608, 5632, 6656, 9728, 11264}}},
    {26275888, {{-1, 1536, -1, 2147483647, 1536, 2560}}},
    {195076144,
     {{-1, 5632, -1, 4608, 2560, 3584},
      {4608, 5632, 3584, 2147483647, 3584, 4608},
      {5632, 7680, -1, 4608, -1, 1536},
      {5632, 7680, 3584, 5632, 1536, 5632},
      {7680, 8704, -1, 4608, -1, 5632},
      {-1, 5632, 3584, 5632, 6656, 7680},
      {5632, 6656, -1, 5632, 5632, 6656}}},
    {194019376,
     {{2560, 4608, -1, 2147483647, 3584, 4608},
      {5632, 6656, 5632, 2147483647, -1, 1536},
      {5632, 2147483647, 5632, 2147483647, 1536, 5632},
      {3584, 2147483647, 5632, 6656, 5632, 9728},
      {4608, 2147483647, 5632, 6656, 9728, 11264},
      {-1, 5120, 5632, 6656, 11264, 2147483647},
      {-1, 1536, 6656, 2147483647, 5632, 11264},
      {-1, 1536, 6656, 2147483647, 13312, 2147483647},
      {1536, 2147483647, 6656, 2147483647, 5632, 2147483647}}},
    {197173296,
     {{4608, 5632, -1, 3584, 3584, 4608},
      {2560, 5632, -1, 4608, 4608, 5632},
      {5632, 7680, -1, 3584, 1536, 5632},
      {8704, 2147483647, -1, 3584, -1, 5632},
      {3072, 5632, -1, 1536, 5632, 6656},
      {-1, 5632, -1, 3584, 6656, 7680},
      {-1, 5632, -1, 4608, 7680, 2147483647},
      {6656, 9728, -1, 4608, 5632, 2147483647}}},
    {161521712, {{-1, 2560, -1, 4608, 4608, 5632}}},
    {59809840, {{-1, 1536, 4608, 2147483647, 4608, 5632}}},
    {194011184, {{6656, 7680, 5632, 2147483647, -1, 1536}, {-1, 1536, 6656, 2147483647, 11264, 13312}}},
    {195067952,
     {{7680, 8704, 4608, 5632, -1, 5632},
      {8704, 2147483647, 3584, 5632, -1, 5632},
      {-1, 5632, 4608, 5632, 7680, 2147483647},
      {6656, 2147483647, 4608, 5632, 5632, 2147483647},
      {5120, 2147483647, 5632, 6656, 11264, 2147483647}}},
    {194007088, {{7680, 2147483647, 9728, 2147483647, -1, 1536}}},
    {196145200, {{-1, 3072, -1, 1536, 5632, 6656}}},
    {196124720, {{-1, 3584, 1536, 5632, 5632, 6656}, {5632, 6656, 3584, 5632, 7680, 2147483647}}},
    {163618864, {{3584, 5632, 1536, 5632, 5632, 6656}}},
    {467726384, {{5632, 6656, -1, 5632, 6656, 7680}}},
    {466657328, {{5632, 6656, -1, 3584, 7680, 2147483647}, {9728, 2147483647, -1, 4608, 5632, 2147483647}}}};

std::map<int, std::vector<std::vector<int>>> g_matmulReduceScatterA3EightRankFP16CodeMap = {
    {196116528, {{-1, 1536, -1, 4096, -1, 1536}, {5632, 2147483647, 1536, 2560, 8704, 2147483647}}},
    {200319024, {{-1, 1536, 4096, 2147483647, -1, 1536}, {6656, 9728, -1, 1536, 8704, 2147483647}}},
    {160493616, {{1536, 2560, -1, 1536, -1, 1536}}},
    {194027568,
     {{1536, 2560, 1536, 2560, -1, 1536},
      {-1, 1536, 1536, 3584, 1536, 7680},
      {1536, 2560, 6656, 2147483647, 1536, 7680},
      {5632, 2147483647, 1536, 2560, -1, 1536},
      {-1, 1536, 1536, 2560, 9728, 2147483647},
      {-1, 1536, 2560, 5120, 7680, 8704}}},
    {194019376,
     {{1536, 2560, 2560, 2147483647, -1, 1536},
      {-1, 1536, 3584, 2147483647, 1536, 7680},
      {1536, 2560, 1536, 6656, 1536, 7680},
      {2560, 2147483647, 2560, 7680, -1, 7680},
      {-1, 1536, 2560, 5120, 8704, 2147483647},
      {1536, 4608, 2560, 8704, 7680, 2147483647}}},
    {26255408, {{-1, 1536, -1, 1536, 1536, 4096}}},
    {195076144, {{-1, 1536, -1, 1536, 4096, 7680}, {2560, 3584, 7680, 2147483647, -1, 1536}}},
    {463511600, {{1536, 2560, -1, 1536, 1536, 6656}, {2560, 6656, -1, 1536, -1, 3584}}},
    {196145200,
     {{1536, 2560, -1, 1536, 6656, 7680},
      {6656, 7680, -1, 1536, -1, 7680},
      {-1, 1536, -1, 1536, 11264, 2147483647},
      {-1, 1536, 9728, 2147483647, 11264, 13312}}},
    {465608752, {{2560, 6656, -1, 1536, 3584, 7680}}},
    {466657328, {{7680, 2147483647, -1, 1536, -1, 7680}, {1536, 6656, -1, 1536, 8704, 9728}}},
    {160473136, {{2560, 5632, 1536, 2560, -1, 1536}}},
    {195067952,
     {{2560, 2147483647, 1536, 2560, 1536, 7680},
      {-1, 1536, 1536, 2560, 7680, 9728},
      {1536, 5632, 1536, 2560, 8704, 2147483647}}},
    {161513520, {{2560, 3584, 7680, 2147483647, 1536, 7680}}},
    {194011184, {{3584, 2147483647, 7680, 9728, -1, 7680}, {4608, 2147483647, 2560, 8704, 7680, 2147483647}}},
    {162562096, {{3584, 2147483647, 9728, 2147483647, -1, 7680}}},
    {161521712, {{-1, 1536, -1, 1536, 7680, 11264}}},
    {194048048,
     {{-1, 1536, 5120, 8704, 7680, 2147483647},
      {-1, 1536, 8704, 9728, 8704, 2147483647},
      {-1, 1536, 9728, 2147483647, 7680, 11264},
      {-1, 1536, 9728, 2147483647, 13312, 2147483647}}},
    {195096624, {{-1, 1536, 8704, 9728, 7680, 8704}}},
    {197173296, {{1536, 4096, -1, 1536, 7680, 8704}}},
    {198221872, {{4096, 6656, -1, 1536, 7680, 8704}}},
    {199290928, {{1536, 6656, -1, 1536, 9728, 2147483647}}},
    {468754480, {{6656, 9728, -1, 1536, 7680, 8704}, {9728, 2147483647, -1, 1536, 7680, 2147483647}}},
    {197165104, {{1536, 7168, 1536, 2560, 7680, 8704}}},
    {197156912, {{7168, 2147483647, 1536, 2560, 7680, 8704}}},
    {160464944, {{1536, 4608, 8704, 2147483647, 7680, 2147483647}}},
    {161505328, {{4608, 2147483647, 8704, 2147483647, 7680, 2147483647}}}};

std::map<int, std::vector<std::vector<int>>> g_matmulReduceScatterA3FourRankINT8CodeMap = {
    {199290928, {{-1, 1536, -1, 4096, -1, 1536}}},
    {198242352, {{-1, 1536, 4096, 7168, -1, 1536}, {-1, 1536, -1, 1536, 1536, 3584}}},
    {196116528,
     {{-1, 1536, 7168, 8704, -1, 1536},
      {-1, 2147483647, 2560, 4608, 7680, 8704},
      {3584, 2147483647, 1536, 4608, 8704, 2147483647}}},
    {200310832, {{-1, 1536, 8704, 2147483647, -1, 1536}}},
    {194048048, {{-1, 1536, 1536, 2560, 1536, 3584}}},
    {194027568, {{-1, 1536, 2560, 3584, 1536, 2560}, {-1, 1536, 3584, 9216, 1536, 3584}}},
    {26255408, {{-1, 1536, 2560, 3584, 2560, 3584}, {7680, 8704, 6656, 2147483647, 1536, 3584}}},
    {161513520, {{-1, 1536, 9216, 2147483647, 1536, 3584}}},
    {195096624, {{1536, 6656, -1, 2560, -1, 1536}}},
    {195076144,
     {{6656, 2147483647, -1, 2560, -1, 1536},
      {1536, 2147483647, 2560, 3584, 1536, 3584},
      {-1, 2147483647, 2560, 3584, 5632, 7680}}},
    {194019376,
     {{1536, 2147483647, 2560, 6656, -1, 1536},
      {1536, 2147483647, 3584, 6656, 1536, 3584},
      {1536, 8704, 8704, 2147483647, -1, 1536},
      {-1, 3072, 3584, 4608, 3584, 7680},
      {-1, 2560, 4608, 5632, 4608, 7680}}},
    {197193776, {{1536, 2147483647, -1, 2560, 1536, 3584}}},
    {194007088,
     {{1536, 8704, 6656, 8704, -1, 1536},
      {8704, 2147483647, 6656, 8704, -1, 3584},
      {8704, 2147483647, 8704, 2147483647, 2560, 3584}}},
    {194011184,
     {{1536, 7680, 6656, 2147483647, 1536, 3584},
      {-1, 2560, 4608, 9728, 3584, 4608},
      {-1, 2560, 5632, 2147483647, 4608, 7680},
      {2560, 9728, 4608, 2147483647, 3584, 7680},
      {9728, 2147483647, 4608, 2147483647, 4608, 7680},
      {-1, 2560, 4608, 5632, 7680, 8704},
      {2560, 4608, 4608, 2147483647, 7680, 2147483647},
      {4608, 2147483647, 5632, 2147483647, 7680, 2147483647}}},
    {26239024, {{8704, 2147483647, 8704, 2147483647, -1, 2560}, {9728, 2147483647, 4608, 2147483647, 3584, 4608}}},
    {162590768, {{-1, 3584, -1, 2560, 3584, 7680}}},
    {467726384, {{3584, 2147483647, -1, 2560, 3584, 7680}, {2560, 2147483647, -1, 1536, 7680, 2147483647}}},
    {195067952, {{-1, 2147483647, 2560, 3584, 3584, 5632}, {-1, 3584, 1536, 4608, 8704, 2147483647}}},
    {195059760, {{3072, 2147483647, 3584, 4608, 3584, 7680}, {4608, 2147483647, 4608, 5632, 7680, 2147483647}}},
    {160464944, {{-1, 2560, 9728, 2147483647, 3584, 4608}}},
    {164687920, {{-1, 1536, -1, 1536, 7680, 2147483647}}},
    {465608752, {{1536, 2560, -1, 1536, 7680, 2147483647}}},
    {198221872, {{-1, 2147483647, 1536, 2560, 7680, 8704}}},
    {59793456, {{-1, 2560, 4608, 5632, 8704, 2147483647}}},
    {160473136, {{-1, 1536, 5632, 2147483647, 7680, 2147483647}}},
    {160456752, {{1536, 2560, 5632, 2147483647, 7680, 2147483647}}}};

std::map<int, std::vector<std::vector<int>>> g_matmulReduceScatterA3FourRankFP16CodeMap = {
    {200319024, {{-1, 1536, -1, 4096, -1, 1536}}},
    {164687920, {{-1, 1536, 4096, 7168, -1, 1536}}},
    {200339504, {{-1, 1536, 7168, 8704, -1, 1536}}},
    {199290928, {{-1, 1536, 8704, 2147483647, -1, 1536}}},
    {194019376,
     {{1536, 8704, -1, 4608, -1, 1536},
      {-1, 1536, 2560, 5632, 1536, 2560},
      {1536, 2560, -1, 2147483647, 1536, 2560},
      {3584, 8704, 1536, 2147483647, 1536, 2560},
      {8704, 9728, -1, 2560, 1536, 2560},
      {1536, 2560, 1536, 2147483647, 3584, 2147483647},
      {2560, 7680, 1536, 3584, 2560, 5632},
      {9728, 2147483647, 1536, 2147483647, 2560, 3584}}},
    {59793456, {{1536, 8704, 4608, 5632, -1, 1536}, {2560, 7680, 1536, 3584, 5632, 2147483647}}},
    {160464944,
     {{1536, 8704, 5632, 9728, -1, 1536},
      {-1, 1536, -1, 2560, 1536, 2560},
      {-1, 1536, 5632, 2147483647, 1536, 2560},
      {8704, 9728, 9728, 2147483647, 1536, 2560},
      {-1, 1536, 1536, 8704, 2560, 2147483647},
      {7680, 8704, 9728, 2147483647, 2560, 2147483647}}},
    {28344368, {{1536, 8704, 9728, 2147483647, -1, 1536}}},
    {161513520, {{2560, 3584, -1, 2147483647, 1536, 2560}}},
    {161521712, {{3584, 8704, -1, 1536, 1536, 2560}}},
    {160456752,
     {{8704, 2147483647, -1, 3584, -1, 1536},
      {8704, 9728, 2560, 8704, 1536, 2560},
      {-1, 1536, 8704, 2147483647, 2560, 2147483647},
      {5632, 7680, 3584, 2147483647, 2560, 2147483647}}},
    {194011184,
     {{8704, 2147483647, 3584, 4608, -1, 1536},
      {9728, 2147483647, 1536, 8704, 1536, 2560},
      {2560, 5632, 3584, 2147483647, 2560, 2147483647},
      {7680, 8704, 1536, 5632, 2560, 11264},
      {8704, 9728, 1536, 2147483647, 2560, 13312},
      {9728, 2147483647, 1536, 2147483647, 3584, 2147483647}}},
    {59801648, {{8704, 2147483647, 4608, 5632, -1, 1536}}},
    {26247216, {{8704, 2147483647, 5632, 2147483647, -1, 1536}}},
    {27295792, {{8704, 9728, 8704, 9728, 1536, 2560}}},
    {162570288, {{9728, 2147483647, -1, 1536, 1536, 2560}, {1536, 2560, -1, 1536, 5120, 7680}}},
    {160473136, {{9728, 2147483647, 8704, 2147483647, 1536, 2560}}},
    {194048048, {{-1, 1536, -1, 1536, 2560, 4096}}},
    {60858416, {{-1, 1536, -1, 1536, 4096, 6656}}},
    {27303984, {{-1, 1536, -1, 1536, 6656, 8704}}},
    {195076144, {{-1, 1536, -1, 1536, 8704, 2147483647}, {1536, 2560, -1, 1536, 2560, 5120}}},
    {61906992, {{1536, 2560, -1, 1536, 7680, 2147483647}}},
    {62955568, {{2560, 4608, -1, 1536, 2560, 6656}, {4608, 5632, -1, 1536, 8704, 2147483647}}},
    {197173296,
     {{2560, 4608, -1, 1536, 6656, 2147483647},
      {5632, 8704, -1, 1536, 2560, 2147483647},
      {8704, 2147483647, -1, 1536, 2560, 5632}}},
    {196124720, {{4608, 5632, -1, 1536, 2560, 8704}}},
    {198221872, {{8704, 2147483647, -1, 1536, 5632, 2147483647}}},
    {195067952, {{1536, 2560, 1536, 2147483647, 2560, 3584}}},
    {195059760, {{7680, 8704, 1536, 5632, 11264, 2147483647}}},
    {194002992, {{7680, 8704, 5632, 9728, 2560, 2147483647}}},
    {60837936, {{8704, 9728, 1536, 2147483647, 13312, 2147483647}}}};
} // namespace

namespace optiling {
static ge::graphStatus MatmulReduceScatterV2CheckAttrAndSetTiling(gert::TilingContext *context,
                                                                  MatmulReduceScatterV2AivModeInfo &info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);

    // Attr相关tilingdata的设置、校验、打印
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto is_trans_a = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_A);
    auto is_trans_b = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B);
    OP_TILING_CHECK(groupPtr == nullptr || strlen(groupPtr) == 0,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "group"), return GRAPH_FAILED);
    OP_TILING_CHECK(is_trans_a == nullptr || is_trans_b == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "is_trans_a"), return GRAPH_FAILED);
    info.isTransposeA = false; // 当前默认a矩阵不转置
    info.isTransposeB = *is_trans_b ? *is_trans_b : false;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MatmulReduceScatterV2CheckShapeAndSetTiling(const gert::TilingContext *context,
                                                                   MatmulReduceScatterV2AivModeInfo &info)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI("MatmulReduceScatterV2AivMode MatmulReduceScatterV2CheckShapeAndSetTiling.");

    const gert::StorageShape *aStorageShape = context->GetInputShape(A_INDEX);
    const gert::StorageShape *bStorageShape = context->GetInputShape(B_INDEX);
    uint32_t M = aStorageShape->GetStorageShape().GetDim(0);
    uint32_t K = aStorageShape->GetStorageShape().GetDim(1);
    uint32_t N = bStorageShape->GetStorageShape().GetDim(1);

    if (aStorageShape->GetStorageShape().GetDim(1) != bStorageShape->GetStorageShape().GetDim(0)) {
        OP_LOGD(nodeName, "A.shape(1) %lu B.shape(0) %lu, istransB = %d", aStorageShape->GetStorageShape().GetDim(1),
                bStorageShape->GetStorageShape().GetDim(0), info.isTransposeB);
        N = bStorageShape->GetStorageShape().GetDim(0);
    }

    info.M = M;
    info.N = N;
    info.K = K;
    OP_LOGD(K_INNER_DEBUG, "M=%d", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%d", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%d", info.N);

    return ge::GRAPH_SUCCESS;
}

static inline bool IsArrayEqual(std::vector<uint32_t> &arr1, const std::vector<uint32_t> &arr2, uint32_t size)
{
    if (arr1.size() < size || arr2.size() < size) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus MatmulReduceScatterV2CheckDtypeAndSetTiling(const gert::TilingContext *context)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context->GetInputDesc(A_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x1"),
                    return ge::GRAPH_FAILED);
    auto x2TensorDesc = context->GetInputDesc(B_INDEX);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x2"),
                    return ge::GRAPH_FAILED);
    auto yDesc = context->GetOutputDesc(C_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y"),
                    return ge::GRAPH_FAILED);

    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    ge::DataType yDtype = yDesc->GetDataType();

    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        vector<uint32_t> paramsType = {x1Dtype, x2Dtype, biasDtype, yDtype};

        for (uint32_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITH_BIAS.size(); kindsId++) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITH_BIAS[kindsId], COUNT_PARAMS_WITH_BIAS)) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x1,x2,bias,y",
                                              (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype) + "," +
                                               Ops::Base::ToString(biasDtype) + "," + Ops::Base::ToString(yDtype))
                                                  .c_str(),
                                              "unsupported dtype combination");
        return ge::GRAPH_FAILED;
    } else {
        vector<uint32_t> paramsType = {x1Dtype, x2Dtype, yDtype};
        for (uint32_t kindsId = 0; kindsId < SUPPORTED_TYPES_WITHOUT_BIAS.size(); kindsId++) {
            if (IsArrayEqual(paramsType, SUPPORTED_TYPES_WITHOUT_BIAS[kindsId], COUNT_PARAMS_WITHOUT_BIAS)) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "x1,x2,y",
            (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype) + "," + Ops::Base::ToString(yDtype))
                .c_str(),
            "unsupported dtype combination");
        return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus MatmulReduceScatterV2GetPlatformInfoAndSetTiling(const gert::TilingContext *context,
                                                                        MatmulReduceScatterV2AivModeInfo &info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%d", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

const std::map<ge::DataType, int64_t> D_TYPE_SIZE_MAP = {
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
    {ge::DT_INT8, 1},
};

static uint32_t AlignUp(uint32_t len, uint32_t size)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(len) + size - 1) & ~(size - 1));
}

static bool IsMatrixAligned(const uint32_t &m, const uint32_t &n, const bool &transpose, const uint32_t &nElemAlign)
{
    return (transpose ? m : n) % nElemAlign == 0;
}

enum class AlgorithmStrategy : int {
    ALGORITHM_STRATEGY_UNDEFINED = -1,
    LARGE_M_OPTIMIZED = 0, // 针对大m优化的算法
    SMALL_M_OPTIMIZED = 1, // 针对小m优化的算法
    ALGORITHM_STRATEGY_MAX = 2,
};

inline AlgorithmStrategy GetAlgorithmPolicy(uint32_t M, uint32_t N, bool is910C, bool isQuant)
{
    bool isOptimizationScenario = (M <= 2048) && (N > 512); // 1. 优化场景
    bool isExcludeTuningScenario = (!is910C) && (!isQuant); // 2. 当前调优场景，后续增加
    const uint64_t PEERMEM_THRESHOLD = 180ULL * 1024 * 1024;
    uint64_t output_matrix_bytes = static_cast<uint64_t>(M) * N * 2;
    bool isMatrixSizeWithinPeermem = (output_matrix_bytes < PEERMEM_THRESHOLD); // 3. 结果矩阵内存小于peermem

    if (isOptimizationScenario && isExcludeTuningScenario && isMatrixSizeWithinPeermem) {
        return AlgorithmStrategy::SMALL_M_OPTIMIZED;
    }
    return AlgorithmStrategy::LARGE_M_OPTIMIZED;
}

static int ConvertGeTypeToBiasDTypeKey(ge::DataType type)
{
    if (type == ge::DT_FLOAT) {
        return TILINGKEY_TPL_FP32;
    } else if (type == ge::DT_BF16) {
        return TILINGKEY_TPL_BF16;
    }
    return TILINGKEY_TPL_FP16;
}

static void GetTilingKey(uint64_t &tilingKey, const MatmulReduceScatterV2AivModeInfo &info,
                         const gert::TilingContext *context)
{
    const gert::StorageShape *matrix_bias = context->GetOptionalInputShape(BIAS_INDEX);
    bool isBias = (matrix_bias == nullptr) ? false : true;
    bool isSmallM =
        GetAlgorithmPolicy(info.M, info.N, info.is910C, info.quantFlag) == AlgorithmStrategy::SMALL_M_OPTIMIZED;
    ge::DataType biasDatatype = context->GetOutputDesc(0)->GetDataType();
    if (isBias) {
        auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
        if (biasDesc != nullptr) {
            biasDatatype = biasDesc->GetDataType();
        }
    }
    uint32_t biasDType = ConvertGeTypeToBiasDTypeKey(biasDatatype);
    tilingKey = GET_TPL_TILING_KEY(isBias, info.isTransposeA, info.isTransposeB, isSmallM, biasDType);
    return;
}

int32_t GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
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

int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

void CalTilingParam(CoCTiling &cocTilingData,
                    const std::map<int *, MatmulReduceScatterV2AivModeTilingValue> &TilingParamMap,
                    const MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t m = static_cast<int32_t>(info.M);
    int32_t k = static_cast<int32_t>(info.K);
    int32_t n = static_cast<int32_t>(info.N);

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
}

void ReduceScatterV2DecodeTilingData(int32_t code, CoCTiling &tilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t m = static_cast<int32_t>(info.M);
    int32_t k = static_cast<int32_t>(info.K);
    int32_t n = static_cast<int32_t>(info.N);
    uint32_t bitCode = static_cast<uint32_t>(code);
    tilingData.commDataSplit = bitCode & COMMDATASPLIT_MASK;
    bitCode >>= COMMDATASPLIT_BNUM;
    tilingData.commNpuSplit = bitCode & COMMNPUSPLIT_MASK;
    bitCode >>= COMMNPUSPLIT_BNUM;
    tilingData.commDirect = bitCode & COMMDIRECT_MASK;
    bitCode >>= COMMDIRECT_BNUM;
    tilingData.ubMoveNum = (bitCode & UBMOVENUM_MASK) * HALF_KBYTE;
    bitCode >>= UBMOVENUM_BNUM;
    tilingData.pValue = bitCode & PVALUE_MASK;
    bitCode >>= PVALUE_BNUM;
    tilingData.swizzlCount = bitCode & SWIZZLCOUNT_MASK;
    bitCode >>= SWIZZLCOUNT_BNUM;
    tilingData.swizzlDirect = bitCode & SWIZZLDIRECT_MASK;
    bitCode >>= SWIZZLDIRECT_BNUM;
    tilingData.m0 = (bitCode & M0_MASK) * DEFAULT_ROW + DEFAULT_ROW;
    tilingData.k0 = DEFAULT_COL;
    tilingData.n0 = tilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
    tilingData.mLoop = CeilDev(m, tilingData.m0);
    tilingData.nLoop = CeilDev(n, tilingData.n0);
    tilingData.kLoop = CeilDev(k, tilingData.k0);
}

void MatmulReduceScatterA2FourRankINT8Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A2_FOUR_RANK_INT8_CODE_DEFAULT;
    std::map<int *, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(MATMUL_REDUCESCATTER_A2_FOUR_RANK_INT8_CODE_DEFAULT,
                                                                    g_matmulReduceScatterA2FourRankINT8CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA2FourRankFP16Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A2_FOUR_RANK_FP16_CODE_DEFAULT;
    std::map<int *, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(MATMUL_REDUCESCATTER_A2_FOUR_RANK_FP16_CODE_DEFAULT,
                                                                    g_matmulReduceScatterA2FourRankFP16CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA2EightRankINT8Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A2_EIGHT_RANK_INT8_CODE_DEFAULT;
    std::map<int *, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
        MATMUL_REDUCESCATTER_A2_EIGHT_RANK_INT8_CODE_DEFAULT, g_matmulReduceScatterA2EightRankINT8CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA2EightRankFP16Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A2_EIGHT_RANK_FP16_CODE_DEFAULT;
    std::map<int *, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
        MATMUL_REDUCESCATTER_A2_EIGHT_RANK_FP16_CODE_DEFAULT, g_matmulReduceScatterA2EightRankFP16CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA3EightRankINT8Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A3_EIGHT_RANK_INT8_CODE_DEFAULT;
    std::map<int *, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
        MATMUL_REDUCESCATTER_A3_EIGHT_RANK_INT8_CODE_DEFAULT, g_matmulReduceScatterA3EightRankINT8CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA3EightRankFP16Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A3_EIGHT_RANK_FP16_CODE_DEFAULT;
    std::map<int *, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(
        MATMUL_REDUCESCATTER_A3_EIGHT_RANK_FP16_CODE_DEFAULT, g_matmulReduceScatterA3EightRankFP16CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA3FourRankINT8Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A3_FOUR_RANK_INT8_CODE_DEFAULT;
    std::map<int *, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(MATMUL_REDUCESCATTER_A3_FOUR_RANK_INT8_CODE_DEFAULT,
                                                                    g_matmulReduceScatterA3FourRankINT8CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void MatmulReduceScatterA3FourRankFP16Tiling(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info)
{
    int32_t code = MATMUL_REDUCESCATTER_A3_FOUR_RANK_FP16_CODE_DEFAULT;
    std::map<int *, MatmulReduceScatterV2AivModeTilingValue> TilingParamMap;
    TilingParamMap[&code] = MatmulReduceScatterV2AivModeTilingValue(MATMUL_REDUCESCATTER_A3_FOUR_RANK_FP16_CODE_DEFAULT,
                                                                    g_matmulReduceScatterA3FourRankFP16CodeMap);

    CalTilingParam(cocTilingData, TilingParamMap, info);

    ReduceScatterV2DecodeTilingData(code, cocTilingData, info);
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;
}

void SetTilingData(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info, int64_t rankSize)
{
    if (info.is910C) {
        if (rankSize == RANKSIZE_FOUR && info.quantFlag) {
            MatmulReduceScatterA3FourRankINT8Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_FOUR && !info.quantFlag) {
            MatmulReduceScatterA3FourRankFP16Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_EIGHT && info.quantFlag) {
            MatmulReduceScatterA3EightRankINT8Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_EIGHT && !info.quantFlag) {
            MatmulReduceScatterA3EightRankFP16Tiling(cocTilingData, info);
            return;
        }
    } else {
        if (rankSize == RANKSIZE_FOUR && info.quantFlag) {
            MatmulReduceScatterA2FourRankINT8Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_FOUR && !info.quantFlag) {
            MatmulReduceScatterA2FourRankFP16Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_EIGHT && info.quantFlag) {
            MatmulReduceScatterA2EightRankINT8Tiling(cocTilingData, info);
            return;
        } else if (rankSize == RANKSIZE_EIGHT && !info.quantFlag) {
            MatmulReduceScatterA2EightRankFP16Tiling(cocTilingData, info);
            return;
        }
    }
    MatmulReduceScatterA2EightRankFP16Tiling(cocTilingData, info);
}

void SetTilingData_SmallM(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info, int64_t rankSize)
{
    if (rankSize == Tiling_Small_M::RANKSIZE_TWO && !info.quantFlag) {
        cocTilingData.m0 = Tiling_Small_M::Tiling_Rank2_A2::GetOptimalM0(info.M, info.K, info.N);
        cocTilingData.swizzlCount = Tiling_Small_M::Tiling_Rank2_A2::GetOptimalSwizzlCount(info.M, info.K, info.N);
        cocTilingData.swizzlDirect = Tiling_Small_M::Tiling_Rank2_A2::GetOptimalSwizzlDirect(info.M, info.K, info.N);
        cocTilingData.pValue = Tiling_Small_M::Tiling_Rank2_A2::GetOptimalPValue(info.M, info.K, info.N);
        cocTilingData.ubMoveNum = Tiling_Small_M::Tiling_Rank2_A2::GetOptimalUbmovenum(info.M, info.K, info.N);
    } else if (rankSize == Tiling_Small_M::RANKSIZE_FOUR && !info.quantFlag) {
        cocTilingData.m0 = Tiling_Small_M::Tiling_Rank4_A2::GetOptimalM0(info.M, info.K, info.N);
        cocTilingData.swizzlCount = Tiling_Small_M::Tiling_Rank4_A2::GetOptimalSwizzlCount(info.M, info.K, info.N);
        cocTilingData.swizzlDirect = Tiling_Small_M::Tiling_Rank4_A2::GetOptimalSwizzlDirect(info.M, info.K, info.N);
        cocTilingData.pValue = Tiling_Small_M::Tiling_Rank4_A2::GetOptimalPValue(info.M, info.K, info.N);
        cocTilingData.ubMoveNum = Tiling_Small_M::Tiling_Rank4_A2::GetOptimalUbmovenum(info.M, info.K, info.N);
    } else if (rankSize == Tiling_Small_M::RANKSIZE_EIGHT && !info.quantFlag) {
        cocTilingData.m0 = Tiling_Small_M::Tiling_Rank8_A2::GetOptimalM0(info.M, info.K, info.N);
        cocTilingData.swizzlCount = Tiling_Small_M::Tiling_Rank8_A2::GetOptimalSwizzlCount(info.M, info.K, info.N);
        cocTilingData.swizzlDirect = Tiling_Small_M::Tiling_Rank8_A2::GetOptimalSwizzlDirect(info.M, info.K, info.N);
        cocTilingData.pValue = Tiling_Small_M::Tiling_Rank8_A2::GetOptimalPValue(info.M, info.K, info.N);
        cocTilingData.ubMoveNum = Tiling_Small_M::Tiling_Rank8_A2::GetOptimalUbmovenum(info.M, info.K, info.N);
    } else {
        cocTilingData.m0 = Tiling_Small_M::DEFAULT_M0;
        cocTilingData.swizzlCount = Tiling_Small_M::DEFAULT_SWIZZLCOUNT;
        cocTilingData.swizzlDirect = Tiling_Small_M::DEFAULT_SWIZZLDIRECT;
        cocTilingData.pValue = Tiling_Small_M::DEFAULT_PVALUE;
        cocTilingData.ubMoveNum = Tiling_Small_M::DEFAULT_UBMOVENUM;
    }

    cocTilingData.commNpuSplit = Tiling_Small_M::DEFAULT_COMMNPUSPLIT;
    cocTilingData.commDataSplit = Tiling_Small_M::DEFAULT_COMMDATASPLIT;
}

inline ge::graphStatus checkAndResetTilingData_SmallM(CoCTiling &cocTilingData, MatmulReduceScatterV2AivModeInfo &info,
                                                      gert::TilingContext *context, int64_t rankSize)
{
    OP_TILING_CHECK(
        cocTilingData.m0 != 128 && cocTilingData.m0 != 256,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "m0", std::to_string(cocTilingData.m0).c_str(), "128 or 256"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(cocTilingData.swizzlCount < 1 || cocTilingData.swizzlCount > 16,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "swizzlCount",
                                              std::to_string(cocTilingData.swizzlCount).c_str(), "1 to 16"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(cocTilingData.swizzlDirect != 0 && cocTilingData.swizzlDirect != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "swizzlDirect",
                                              std::to_string(cocTilingData.swizzlDirect).c_str(), "0 or 1"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(cocTilingData.pValue < 1 || cocTilingData.pValue > 20,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "pValue",
                                              std::to_string(cocTilingData.pValue).c_str(), "1 to 20"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(cocTilingData.ubMoveNum < 4 || cocTilingData.ubMoveNum > 100,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ubMoveNum",
                                              std::to_string(cocTilingData.ubMoveNum).c_str(), "4 to 100"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(cocTilingData.commNpuSplit != 0 && cocTilingData.commNpuSplit != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "commNpuSplit",
                                              std::to_string(cocTilingData.commNpuSplit).c_str(), "0 or 1"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(cocTilingData.commDataSplit != 8 && cocTilingData.commDataSplit != 16,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "commDataSplit",
                                              std::to_string(cocTilingData.commDataSplit).c_str(), "8 or 16"),
                    return ge::GRAPH_FAILED);

    int32_t m_loop = (info.M + cocTilingData.m0 - 1) / (cocTilingData.m0);
    if ((cocTilingData.swizzlDirect == 0 && m_loop > cocTilingData.swizzlCount)) {
        cocTilingData.swizzlCount = m_loop;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = ascendcPlatform.GetCoreNumAic();
    OP_TILING_CHECK(coreNum == 0, OP_LOGE(context->GetNodeName(), "ascendcPlatform.GetCoreNumAic() return 0 cores."),
                    return ge::GRAPH_FAILED);
    int32_t count_m_tile = cocTilingData.swizzlDirect != 0 ?
                               ((coreNum * (cocTilingData.pValue)) / cocTilingData.swizzlCount) :
                               cocTilingData.swizzlCount;

    auto cType = context->GetOutputDesc(0)->GetDataType();
    uint32_t elementSize = D_TYPE_SIZE_MAP.at(cType);

    OP_TILING_CHECK((static_cast<uint64_t>(info.M) * info.N / rankSize * elementSize) >= (180 * 1024 * 1024),
                    OP_LOGE(context->GetNodeName(),
                            "The space required for the output result is larger than the peermem space, and "
                            "a single copy is not possible."),
                    return ge::GRAPH_FAILED);

    if (cocTilingData.swizzlDirect == 1 && m_loop > count_m_tile) {
        cocTilingData.pValue = CeilDev(m_loop * cocTilingData.swizzlCount, coreNum);
    } else if (cocTilingData.swizzlDirect == 0 && m_loop > (coreNum * (cocTilingData.pValue))) {
        cocTilingData.pValue = coreNum == 0 ? 0 : (m_loop + coreNum - 1) / coreNum;
    }

    cocTilingData.ubMoveNum = cocTilingData.ubMoveNum * HALF_KBYTE;
    if (cocTilingData.m0 >= DEFAULT_ROW) {
        cocTilingData.k0 = DEFAULT_COL;
        cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
    }
    cocTilingData.kLoop = CeilDev(info.K, cocTilingData.k0);
    cocTilingData.nLoop = CeilDev(info.N, cocTilingData.n0);
    cocTilingData.mLoop = m_loop;
    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum;

    return ge::GRAPH_SUCCESS;
}
void GetUsrWorkSpaceSize(uint32_t elementSize, uint32_t numBlocks, uint64_t &userWorkSpaceSize,
                         MatmulReduceScatterV2AivModeInfo &info, CoCTiling &cocTilingData, bool needCastBias)
{
    constexpr int32_t TWO = 2;
    constexpr uint32_t NUMSIZE_ONE = 1;
    uint32_t nElemAlign = HALF_KBYTE / elementSize;
    bool hasAAlign = (!IsMatrixAligned(info.M, info.K, info.isTransposeA, nElemAlign) && info.M != NUMSIZE_ONE);
    bool hasBAlign = !IsMatrixAligned(info.K, info.N, info.isTransposeB, nElemAlign);
    uint32_t mAlign = AlignUp(info.M, nElemAlign);
    uint32_t kAlign = AlignUp(info.K, nElemAlign);
    uint32_t nAlign = AlignUp(info.N, nElemAlign);

    info.aAlignSize = 0;
    info.bAlignSize = 0;
    info.dequantSize = 0;
    info.biasCastSize = 0;
    info.hasAAlign = hasAAlign;
    info.hasBAlign = hasBAlign;
    if (info.hasAAlign) {
        info.aAlignSize =
            (info.isTransposeA ? static_cast<uint64_t>(info.K) * mAlign : static_cast<uint64_t>(info.M) * kAlign) *
            elementSize;
        userWorkSpaceSize += info.aAlignSize;
    }
    if (info.hasBAlign) {
        info.bAlignSize =
            (info.isTransposeB ? static_cast<uint64_t>(info.N) * kAlign : static_cast<uint64_t>(info.K) * nAlign) *
            elementSize;
        userWorkSpaceSize += info.bAlignSize;
    }
    if (info.quantFlag) {
        if (GetAlgorithmPolicy(info.M, info.N, info.is910C, info.quantFlag) == AlgorithmStrategy::SMALL_M_OPTIMIZED) {
            // 针对小m场景优化算法，输出矩阵可以一次性放入peermem中，因此workspace匹配整个输出矩阵大小。
            info.dequantSize = static_cast<uint64_t>(mAlign) * nAlign * sizeof(int32_t);
        } else {
            // 大m场景算法，peermem可能转不下输出矩阵，需要double buffer搬运，因此workspace匹配double buffer空间。
            info.dequantSize = static_cast<uint64_t>(cocTilingData.pValue) * numBlocks * cocTilingData.m0 *
                               cocTilingData.n0 * TWO * sizeof(int32_t);
        }
    }
    userWorkSpaceSize += info.dequantSize;
    if (needCastBias) {
        const uint64_t biasBytes = static_cast<uint64_t>(info.N) * sizeof(float);
        info.biasCastSize =
            (biasBytes + HALF_KBYTE - 1) & ~(static_cast<uint64_t>(HALF_KBYTE) - 1);
        userWorkSpaceSize += info.biasCastSize;
    }
}

static bool CheckDtype_X1(const gert::TilingContext *context)
{
    const gert::Tensor *x1Scale = context->GetInputTensor(X1_SCALE_INDEX);
    if (x1Scale == nullptr) {
        return false;
    }
    auto x1Type = x1Scale->GetDataType();
    if (x1Type != ge::DT_FLOAT) {
        return false;
    }
    return true;
}

static bool CheckDtype_X2(const gert::TilingContext *context, MatmulReduceScatterV2AivModeInfo &info,
                          ge::DataType cType)
{
    const gert::Tensor *x2Scale = context->GetInputTensor(X2_SCALE_INDEX);
    if (x2Scale == nullptr) {
        return false;
    }
    auto x2ScaleType = x2Scale->GetDataType();
    info.isX2ScaleTypeInt64 = false;
    /* x2ScaleType支持float类型 */
    if (x2ScaleType == ge::DT_FLOAT) {
        return true;
    }
    /* 在输出类型为fp16时，x2ScaleType支持int64类型 */
    if (cType == ge::DT_FLOAT16 && x2ScaleType == ge::DT_INT64) {
        info.isX2ScaleTypeInt64 = true;
        return true;
    }
    return false;
}

static void PrintTilingDataInfo(MatmulReduceScatterV2AivModeInfo &info, CoCTiling &cocTilingInfo)
{
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.M %u", info.M);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.k %u", info.K);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.N %u", info.N);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.aivNum %u", info.aivNum);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.totalUbSize %u", info.totalUbSize);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.isTransposeA %d", info.isTransposeA);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.isTransposeB %d", info.isTransposeB);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.aAlignSize %lu", info.aAlignSize);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.bAlignSize %lu", info.bAlignSize);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.biasCastSize %lu", info.biasCastSize);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.quantFlag %d", info.quantFlag);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.is910C %d", info.is910C);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.isX2ScaleTypeInt64 %d", info.isX2ScaleTypeInt64);

    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.m0 %d", cocTilingInfo.m0);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.k0 %d", cocTilingInfo.k0);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.n0 %d", cocTilingInfo.n0);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.swizzlCount %d", cocTilingInfo.swizzlCount);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.swizzlDirect %d", cocTilingInfo.swizzlDirect);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.pValue %d", cocTilingInfo.pValue);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.ubMoveNum %d", cocTilingInfo.ubMoveNum);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.commNpuSplit %d", cocTilingInfo.commNpuSplit);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.commDataSplit %d", cocTilingInfo.commDataSplit);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tiling.lenPerLoop %d", cocTilingInfo.lenPerLoop);
}

ge::graphStatus MatmulReduceScatterTilingV2AivModeFunc(gert::TilingContext *context)
{
    OP_LOGI("Enter MatmulReduceScatterV2 aivMode tiling func.");

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(context->GetNodeName(), ret);

    // 1. tilingData
    MatmulReduceScatterV2AivModeTilingData *tilingData =
        context->GetTilingData<MatmulReduceScatterV2AivModeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);
    MatmulReduceScatterV2AivModeInfo &info = tilingData->matmulReduceScatterV2AivModeInfo;
    OP_TILING_CHECK(MatmulReduceScatterV2CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "MatmulReduceScatterV2 aivMode CheckAttrAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulReduceScatterV2CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "MatmulReduceScatterV2 aivMode CheckShapeAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulReduceScatterV2CheckDtypeAndSetTiling(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "MatmulReduceScatterV2 aivMode CheckDtypeAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulReduceScatterV2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "MatmulReduceScatterV2 aivMode GetPlatformInfoAndSetTiling Failed"),
                    return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    const char *opName = context->GetNodeName();
    // 校验HCCL BUFF空间大小
    uint64_t hcclBuffSize = 0ULL;
    auto cclRet = mc2tiling::GetCclBufferSize(group, &hcclBuffSize, opName);
    if (cclRet == ge::GRAPH_SUCCESS) {
        OP_TILING_CHECK(hcclBuffSize < CCL_BUFFER_MIN_BYTES,
                        OP_LOGE(opName, "HCCL_BUFFSIZE (%lu Bytes) too small, min required %lu Bytes (%dMB)",
                                hcclBuffSize, CCL_BUFFER_MIN_BYTES, CCL_BUFFER_MIN_BYTES / MB_BYTES),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGW(opName, "Can't get HCCL_BUFFSIZE, skip CCL buffer size validation.");
    }
    int64_t rankSize = 0;
    mc2tiling::GetRankSize(opName, group, rankSize);
    OP_TILING_CHECK(
        rankSize != 2 && rankSize != 4 && rankSize != 8,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "rankSize", std::to_string(rankSize).c_str(), "2, 4 or 8"),
        return ge::GRAPH_FAILED);
    // 2. set numBlocks
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(numBlocks);

    auto aType = context->GetInputTensor(A_INDEX)->GetDataType();
    auto bType = context->GetInputTensor(B_INDEX)->GetDataType();
    auto cType = context->GetOutputDesc(0)->GetDataType();
    info.quantFlag =
        (aType == ge::DT_INT8) && (bType == ge::DT_INT8) && (cType == ge::DT_BF16 || cType == ge::DT_FLOAT16);
    const auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    const bool needCastBias = !info.quantFlag && aType == ge::DT_BF16 && bType == ge::DT_BF16 &&
                              biasDesc != nullptr && biasDesc->GetDataType() == ge::DT_BF16;

    // 3. 配置平台信息，GetTilingKey
    info.is910C = false;
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    if (socVersion == "Ascend910_93") {
        info.is910C = true;
    }

    // 4. set tilingKey
    uint64_t tilingKey = INIT_TILINGKEY;
    GetTilingKey(tilingKey, info, context);
    context->SetTilingKey(tilingKey);
    OP_LOGD("MatmulReduceScatterV2AivModeTiling", "tilingkey is %lu", tilingKey);

    // 5. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "workSpaces"),
                    return ge::GRAPH_FAILED);
    if (info.quantFlag) {
        OP_TILING_CHECK(!CheckDtype_X2(context, info, cType),
                        OP_LOGE(context->GetNodeName(), "MatmulReduceScatterV2 aivMode Invalid x2Scale."),
                        return ge::GRAPH_FAILED);
        info.dequant_type = DequantType::PER_CHANNEL;
        if (CheckDtype_X1(context)) {
            info.dequant_type = DequantType::PER_TOKEN;
        }
    }

    if (GetAlgorithmPolicy(info.M, info.N, info.is910C, info.quantFlag) == AlgorithmStrategy::SMALL_M_OPTIMIZED) {
        SetTilingData_SmallM(tilingData->cocTiling, info, rankSize);
        OP_TILING_CHECK(
            checkAndResetTilingData_SmallM(tilingData->cocTiling, info, context, rankSize) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "MatmulReduceScatterV2 aivMode checkAndResetTilingData_SmallM Failed"),
            return ge::GRAPH_FAILED);
    } else {
        SetTilingData(tilingData->cocTiling, info, rankSize);
    }

    uint32_t elementSize = D_TYPE_SIZE_MAP.at(aType);
    uint64_t userWorkSpaceSize = 0;
    GetUsrWorkSpaceSize(elementSize, numBlocks, userWorkSpaceSize, info, tilingData->cocTiling, needCastBias);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + userWorkSpaceSize;

    PrintTilingDataInfo(info, tilingData->cocTiling);

    // 6. communication
    if (info.is910C) {
        uint32_t opType = OP_TYPE_REDUCE_SCATTER;
        std::string algConfig = "ReduceScatter=level0:fullmesh";
        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
                        OP_LOGE(context->GetNodeName(), "mc2CcTilingConfig mc2InitTiling GetTiling failed."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
                        OP_LOGE(context->GetNodeName(), "mc2CcTilingConfig mc2CcTiling GetTiling failed."),
                        return ge::GRAPH_FAILED);
    } else {
        uint32_t opType = 18;
        std::string algConfig = "MultiPut=level0:fullmesh";
        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
                        OP_LOGE(context->GetNodeName(), "mc2CcTilingConfig mc2InitTiling GetTiling failed."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
                        OP_LOGE(context->GetNodeName(), "mc2CcTilingConfig mc2CcTiling GetTiling failed."),
                        return ge::GRAPH_FAILED);
    }

    OP_LOGI("Leave MatmulReduceScatterV2AivMode tiling func.");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
