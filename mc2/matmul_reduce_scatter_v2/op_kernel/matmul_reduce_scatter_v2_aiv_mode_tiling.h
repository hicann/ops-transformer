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
 * \file matmul_reduce_scatter_v2_aiv_mode_tiling.h
 * \brief
 */
#ifndef __MATMUL_REDUCE_SCATTER_V2_AIV_MODE_TILING_H__
#define __MATMUL_REDUCE_SCATTER_V2_AIV_MODE_TILING_H__
#include <vector>
#include <map>
#include "kernel_tiling/kernel_tiling.h"
namespace matmulReduceScatterV2_aivmode_tiling {

enum class DequantType : int {
    DEQUANT_TYPE_UNDEFINED = -1,
    PER_CHANNEL = 0,
    PER_TOKEN = 1,
    DEQUANT_TYPE_MAX = 2,
};

struct MatmulReduceScatterV2AivModeInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t aivNum;
    uint32_t totalUbSize;
    bool isTransposeA;
    bool isTransposeB;
    uint64_t aAlignSize;
    uint64_t bAlignSize;
    uint64_t dequantSize;
    uint64_t biasCastSize;
    bool hasAAlign;
    bool hasBAlign;
    bool quantFlag;
    bool is910C;
    bool isX2ScaleTypeInt64; /* 当前x2Scale存在入参类型为int64的场景，需要tilingKey的方式传入kernel */
    DequantType dequant_type;
};

struct CoCTiling {
    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t mLoop = -1;
    int32_t kLoop = -1;
    int32_t nLoop = -1;
    int32_t swizzlCount = -1;
    int32_t swizzlDirect = -1;
    int32_t pValue = -1;
    int32_t ubMoveNum = -1;
    int32_t commNpuSplit = -1;
    int32_t commDataSplit = -1;
    int32_t lenPerLoop = -1;
    int32_t commDirect = -1;
};

struct MatmulReduceScatterV2AivModeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MatmulReduceScatterV2AivModeInfo matmulReduceScatterV2AivModeInfo;
    CoCTiling cocTiling;
};

struct MatmulReduceScatterV2AivModeTilingValue {
    int32_t value = -1;
    std::map<int, std::vector<std::vector<int>>> conditionMap = {};
    explicit MatmulReduceScatterV2AivModeTilingValue(int32_t v = -1,
                                                     std::map<int, std::vector<std::vector<int>>> m = {})
        : value(v),
          conditionMap(std::move(m))
    {
    }
};

constexpr int32_t REDUCESCATTER_M0_DEFAULT = 128;
constexpr int32_t REDUCESCATTER_UBMOVENUM_DEFAULT = 20;
constexpr int32_t REDUCESCATTER_COMMDATASPLIT_DEFAULT = 16;
constexpr int32_t REDUCESCATTER_PVALUE_DEFAULT = 12;
constexpr int32_t RANKSIZE_FOUR = 4;
constexpr int32_t RANKSIZE_EIGHT = 8;
constexpr int32_t SWIZZLE_DIRECT_ONE = 1;
constexpr int32_t SWIZZLE_COUNT_FOUR = 4;
constexpr int32_t COMM_DATA_DIRECT = 0;
constexpr int32_t COMMNPUSPLIT_ONE = 1;
constexpr int32_t HALF_KBYTE = 512;
constexpr int32_t DEFAULT_ROW = 128;
constexpr int32_t DEFAULT_COL = 256;
// Tiling Code Mask
constexpr int32_t COMMDATASPLIT_MASK = 0b11111;
constexpr int32_t COMMDATASPLIT_BNUM = 5;
constexpr int32_t COMMNPUSPLIT_MASK = 0b11111;
constexpr int32_t COMMNPUSPLIT_BNUM = 5;
constexpr int32_t COMMDIRECT_MASK = 0b1;
constexpr int32_t COMMDIRECT_BNUM = 1;
constexpr int32_t UBMOVENUM_MASK = 0b11111111;
constexpr int32_t UBMOVENUM_BNUM = 8;
constexpr int32_t PVALUE_MASK = 0b1111;
constexpr int32_t PVALUE_BNUM = 4;
constexpr int32_t SWIZZLCOUNT_MASK = 0b1111;
constexpr int32_t SWIZZLCOUNT_BNUM = 4;
constexpr int32_t SWIZZLDIRECT_MASK = 0b1;
constexpr int32_t SWIZZLDIRECT_BNUM = 1;
constexpr int32_t M0_MASK = 0b1;
constexpr int32_t M0_BNUM = 1;

constexpr int32_t CONDITION_M_ST = 0;
constexpr int32_t CONDITION_M_END = 1;
constexpr int32_t CONDITION_K_ST = 2;
constexpr int32_t CONDITION_K_END = 3;
constexpr int32_t CONDITION_N_ST = 4;
constexpr int32_t CONDITION_N_END = 5;

constexpr int32_t MATMUL_REDUCESCATTER_A2_FOUR_RANK_INT8_CODE_DEFAULT = 195043376;
constexpr int32_t MATMUL_REDUCESCATTER_A2_FOUR_RANK_FP16_CODE_DEFAULT = 193992752;
constexpr int32_t MATMUL_REDUCESCATTER_A2_EIGHT_RANK_INT8_CODE_DEFAULT = 468734000;
constexpr int32_t MATMUL_REDUCESCATTER_A2_EIGHT_RANK_FP16_CODE_DEFAULT = 463487024;
constexpr int32_t MATMUL_REDUCESCATTER_A3_EIGHT_RANK_INT8_CODE_DEFAULT = 467726384;
constexpr int32_t MATMUL_REDUCESCATTER_A3_EIGHT_RANK_FP16_CODE_DEFAULT = 468754480;
constexpr int32_t MATMUL_REDUCESCATTER_A3_FOUR_RANK_INT8_CODE_DEFAULT = 467726384;
constexpr int32_t MATMUL_REDUCESCATTER_A3_FOUR_RANK_FP16_CODE_DEFAULT = 200339504;
} // namespace matmulReduceScatterV2_aivmode_tiling
#endif
