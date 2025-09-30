/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_tiling_utils.h
 * \brief
 */

#ifndef __MC2_TILING_UTILS_H__
#define __MC2_TILING_UTILS_H__

#include <cstdint>
#include <map>
#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "formulaic_tiling_datatype.h"
#include "graph/utils/type_utils.h"
#include "matmul_formulaic_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_type.h"

namespace mc2tiling {
constexpr uint32_t COMM_MESH = 0b1U;
constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
constexpr uint32_t COMM_UNDEFINED = 0xFFFFFFFFU;
constexpr uint8_t COMM_ALG_FULL_MESH_HOST = 6;
constexpr uint64_t CHECK_VALUE_ODD = 2;
constexpr uint32_t AIC_NUM_910D = 32;
constexpr uint64_t MC2_TILINGKEY_OFFSET =
    uint64_t(1000000000000000000UL);  // 10^18
constexpr size_t RES_LEN = 64;
constexpr size_t MAX_MSG_NUM = 16;
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4;  // 只通信不计算
constexpr char HCCL_DETERMINISTIC[] = "HCCL_DETERMINISTIC";
constexpr uint8_t Y_INDEX = 3;
constexpr uint8_t COMM_ALG_DEFAULT = 0;
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr uint8_t COMM_ALG_DOUBLE_RING = 2;
constexpr uint8_t COMM_ALG_SWITCH_WING = 3;
constexpr uint8_t COMM_VERSION3 = 3;
constexpr double COMM_GROW_RATIO = 1.15;

constexpr uint64_t LARGE_K = 8192;
constexpr uint64_t LARGE_N = 5120;
constexpr uint64_t SMALL_N_BOUNDARY = 2048;
constexpr uint64_t TINY_M = 512;
constexpr uint64_t SMALL_M = 2048;
constexpr uint64_t MEDIAN_M = 4096;
constexpr double GATHER_LARGERNK_COMM_GROW_RATIO1 = 3;
constexpr double GATHER_LARGERNK_COMM_GROW_RATIO2 = 1.5;

constexpr uint8_t TIME_LOWER_RATIO = 2;
constexpr double TIME_UPPER_RATIO = 3.5;
constexpr double SCATTER_LARGERNK_COMM_GROW_RATIO1 = 1.5;
constexpr double SCATTER_LARGERNK_COMM_GROW_RATIO2 = 1.2;
constexpr double CUBE_UTIL_THRESH = 0.85;
constexpr uint32_t AICPU_BLOCK_DIM_A2 = 6U;

constexpr auto DEFAULT_KEY_FOR_FITTING_MAP = "0_0";

enum class Mc2QuantMode {
  DEFAULT = 0,
  PERTENSOR_MODE,
  PERBLOCK_MODE,
  INVALID_MODE,
};

struct HcclAicpuOpParam {
  uint8_t res[RES_LEN];
};

struct KFCMsgBody {
  // Rank* aiv * MsgSize * sizeof(消息)
  HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
  HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};
struct KFCNotify {
  // 消息通信
  HcclAicpuOpParam msgSend[MAX_MSG_NUM];  // 填充16个
  HcclAicpuOpParam msgCnt[MAX_MSG_NUM];
};

constexpr std::initializer_list<ge::DataType> FP8DTYPE_SUPPORT_LIST = {
    ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2,
    ge::DataType::DT_HIFLOAT8};

matmul_tiling::DataType ConvertGeTypeToMmType(const std::string &opName,
                                              ge::DataType type);
ge::DataType ConvertMmTypeToGeType(const std::string &opName,
                                   matmul_tiling::DataType type);
uint64_t GetDataTypeSize(const std::string &opName, ge::DataType type);
HcclDataType ConvertGeTypeToHcclType(const std::string &opName,
                                     ge::DataType type);
bool CheckSuppportedFormat(ge::Format format);
bool IsDeterministic();
bool GetRankSize(const std::string &opName, const char *group,
                 int64_t &rankSize);
bool CheckRankSize(const platform_ascendc::SocVersion socVersion,
                   const uint32_t rankSize);
uint8_t Mc2GetCommAlgo(int64_t rankDim, uint64_t mValue, const char *group,
                       const gert::TilingContext *context);

bool CheckDataTypeVaild(ge::DataType type,
                        std::initializer_list<ge::DataType> supportDtypeList);

class Mc2TilingUtils {
 public:
  static uint8_t GetDebugMode();
  static uint8_t GetDebugCommAlg();
  static uint8_t GetDebugStepSize();
  static uint32_t GetCommSets(const char *group);
  static ge::graphStatus CommonParamCheck(const gert::TilingContext *context);
  static mc2tiling::HcclDataType GetDataType(ge::DataType type);
  static uint64_t GetMaxWindowSize();
  static bool CheckRankSize(platform_ascendc::SocVersion socVersion,
                            uint32_t rankSize);
  static HcclDataType ConvertGeTypeToHcclType(const std::string &opName,
                                              ge::DataType type);

  template <typename T>
  static uint64_t GetTilingKey(T &tilingData, bool isFullMeshHost = false) {
    uint8_t commAlg =
        isFullMeshHost ? COMM_ALG_FULL_MESH_HOST : tilingData.msg.get_commAlg();
    uint64_t castBias = tilingData.param.get_biasLen() == 0 ? 0 : 1;
    // tiling key: commAlg(switch/doublering/fullmesh) nd2nz bias2float
    uint64_t tilingKey =
        optiling::RecursiveSum(castBias, 1, static_cast<uint64_t>(commAlg));
    return tilingKey;
  };
};

const std::map<ge::DataType, matmul_tiling::DataType> D_TYPE_MAP = {
    {ge::DT_BF16, matmul_tiling::DataType::DT_BFLOAT16},
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
};

const std::map<matmul_tiling::DataType, ge::DataType> D_TYPE_MATMUL_MAP = {
    {matmul_tiling::DataType::DT_BFLOAT16, ge::DT_BF16},
    {matmul_tiling::DataType::DT_FLOAT16, ge::DT_FLOAT16},
    {matmul_tiling::DataType::DT_FLOAT, ge::DT_FLOAT},
};

const std::map<ge::DataType, int64_t> D_TYPE_SIZE_MAP = {
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
};

const std::map<ge::DataType, mc2tiling::HcclDataType> HCCL_DATA_TYPE = {
    {ge::DataType::DT_INT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT8},
    {ge::DataType::DT_UINT8, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT8},
    {ge::DataType::DT_INT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT16},
    {ge::DataType::DT_UINT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT16},
    {ge::DataType::DT_INT32, mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT32},
    {ge::DataType::DT_UINT32, mc2tiling::HcclDataType::HCCL_DATA_TYPE_UINT32},
    {ge::DataType::DT_FLOAT16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP16},
    {ge::DataType::DT_FLOAT, mc2tiling::HcclDataType::HCCL_DATA_TYPE_FP32},
    {ge::DataType::DT_BF16, mc2tiling::HcclDataType::HCCL_DATA_TYPE_BFP16},
};

const std::map<platform_ascendc::SocVersion, std::set<uint32_t>>
    supportedRankSizeSet = {
        {platform_ascendc::SocVersion::ASCEND310P, {1, 2, 4}},
        {platform_ascendc::SocVersion::ASCEND910B, {1, 2, 4, 8}},
};

const std::set<ge::Format> SUPPORTED_FORMAT = {
    ge::FORMAT_NCL,  ge::FORMAT_NCDHW, ge::FORMAT_DHWCN,
    ge::FORMAT_NHWC, ge::FORMAT_NCHW,  ge::FORMAT_ND};
}  // namespace mc2tiling

#endif