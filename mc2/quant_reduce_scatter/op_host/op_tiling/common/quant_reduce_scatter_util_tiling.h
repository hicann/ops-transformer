/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_REDECE_SCATTER_UTIL_TILING_H
#define QUANT_REDECE_SCATTER_UTIL_TILING_H

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "util/math_util.h"

namespace MC2Tiling {

using namespace std;
using namespace ge;
using namespace gert;

// input index
constexpr size_t CONTEXT_IDDEX = 0;
constexpr size_t X_INDEX = 1;
constexpr size_t SCALES_INDEX = 2;
// output index
constexpr size_t OUTPUT_INDEX = 0;
// attr index
constexpr size_t HCCL_BUFFER_SIZE_INDEX = 0;
constexpr size_t REDUCE_OP_INDEX = 1;
constexpr size_t OUTPUT_DTYPE_INDEX = 2;
constexpr size_t WORLD_SIZE_INDEX = 3;
// 参数范围
const std::string REDUCE_OP_TYPE = "sum";
const std::vector<uint32_t> X_DTYPE_LIST = {ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::vector<uint32_t> SCALES_DTYPE_LIST = {ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};
const std::vector<uint32_t> OUTPUT_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};
const std::vector<int64_t> RANK_SIZE_LIST = {2, 4, 8};
constexpr int64_t RANK_SIZE_DEFAULT = -1;
constexpr uint64_t H_VALUE_5120 = 5120UL;
constexpr uint64_t H_VALUE_7168 = 7168UL;
constexpr uint64_t H_VALUE_LOWER_LIMIT = 1024UL;
constexpr uint64_t H_VALUE_UPPER_LIMIT = 8192UL;
// 维度范围
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t THREE_DIMS = 3;
constexpr uint32_t FOUR_DIMS = 4;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
// 量化模式
constexpr uint32_t TG_QUANT_MOD = 1;
constexpr uint32_t MX_QUANT_MOD = 2;
constexpr uint64_t TG_QUANT_NUMBER = 128UL;
constexpr uint64_t MX_QUANT_NUMBER = 64UL;
// MX量化模式下scales的最后一维一定是2，比如(bs, h/64, 2)
constexpr uint64_t MX_SCALE_LAST_DIM = 2UL;
// 通信域参数
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t X_DTYPE_SIZE_ONE = 1;
constexpr uint32_t SCALE_DTYPE_SIZE_ONE = 1;
constexpr uint32_t SCALE_DTYPE_SIZE_FOUR = 4;
constexpr uint32_t AIV_TYPE = 3;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t HCCL_BUFFSIZE_FACTOR = 2;

// op类型
enum OpType : size_t {
    OP_QUANT_REDUCE_SCATTER,
    OP_QUANT_ALL_REDUCE,
};

// 封装参数
struct TilingRunInfo {
    uint32_t quantMode; // 量化方式
    int64_t rankSize;   // rank大小
    int64_t hcclBufferSize;
};

class QuantReduceScatterUtilTiling {
public:
    static ge::graphStatus CheckNpuArch(const gert::TilingContext *context);
    static ge::graphStatus CheckTilingFunc(gert::TilingContext *context, TilingRunInfo &runInfo, const OpType opType);
};

}; // namespace MC2Tiling

#endif // QUANT_REDECE_SCATTER_UTIL_TILING_H
