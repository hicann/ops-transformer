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
 * \file quant_grouped_mat_mul_allto_allv_tiling_base.h
 * \brief
 */
#ifndef MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_H
#define MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_H

#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"

namespace Mc2Tiling {
enum GmmA2AvInputTensorIndex : uint32_t {
    GMM_X_INDEX = 0,
    GMM_WEIGHT_INDEX,
    GMM_X_SCALE_INDEX,
    GMM_WEIGHT_SCALE_INDEX,
    SEND_COUNTS_TENSOR_OPTIONAL_INDEX,
    RECV_COUNTS_TENSOR_OPTIONAL_INDEX = 5,
    MM_X_OPTIONAL_INDEX,
    MM_WEIGHT_OPTIONAL_INDEX,
    MM_X_SCALE_OPTIONAL_INDEX,
    MM_WEIGHT_SCALE_OPTIONAL_INDEX,
    COMM_QUANT_SCALE_OPTIONAL_INDEX,
    COMM_MODE_INDEX,
};

enum GmmA2AvOutputTensorIndex : uint32_t {
    OUTPUT_Y_INDEX = 0,
    OUTPUT_MM_Y_OPTIONAL_INDEX
};

enum GmmA2AvAttrIndex : uint32_t {
    ATTR_GROUP_INDEX = 0,
    ATTR_EP_WORLD_SIZE_INDEX,
    ATTR_SEND_COUNTS_INDEX,
    ATTR_RECV_COUNTS_INDEX,
    ATTR_GMM_X_QUANT_MODE_INDEX,
    ATTR_GMM_WEIGHT_QUANT_MODE_INDEX = 5,
    ATTR_TRANS_GMM_WEIGHT_INDEX,
    ATTR_TRANS_MM_WEIGHT_INDEX,
    ATTR_MM_X_QUANT_MODE_INDEX,
    ATTR_MM_WEIGHT_QUANT_MODE_INDEX,
    ATTR_COMM_QUANT_MODE_INDEX = 10,
    ATTR_GROUP_SIZE_OPTIONAL_INDEX,
    ATTR_Y_DTYPE,
    ATTR_MM_Y_DTYPE,
    ATTR_COMM_QUANT_DTYPE_INDEX,
    ATTR_COMM_MODE_INDEX
};

enum QuantizationMode {
    QUANT_NONE = 0,          // 不量化
    QUANT_PERTENSOR = 1,     // pertensor
    QUANT_PERCHANNEL = 2,    // perchannel
    QUANT_PERTOKEN = 3,      // pertoken
    QUANT_PERGROUP = 4,      // pergroup
    QUANT_PERBLOCK = 5,      // perblock
    QUANT_MX = 6,            // mx量化
    QUANT_PERTOKEN_DYNAMIC = 7  // pertoken动态量化
};

enum QuantModePair {
    QUANT_PAIR_NONE = 0,   // 不量化
    QUANT_PAIR_TT = 1,     // pertensor - pertensor
    QUANT_PAIR_KC = 2,     // pertoken - perchannel
    QUANT_PAIR_MX = 6,     // mxfp8
    QUANT_PAIR_ERROR = 255
};

constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t PERTENSOR_MODE = 1;
constexpr uint32_t SINGLE_GROUP_NUM = 1;
constexpr uint32_t GMM_ACT_TYPE_NONE = 0;
constexpr uint64_t DB_SIZE = 2UL;

constexpr uint32_t DIM_ZERO = 0;
constexpr uint32_t DIM_ONE = 1;
constexpr uint32_t DIM_TWO = 2;
constexpr uint32_t DIM_THREE = 3;
constexpr uint32_t DIM_FOUR = 4;


constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_EIGHT = 8;

constexpr uint32_t INDEX_TWO = 2U;

constexpr int64_t BEST_BASEN = 256;
constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;

constexpr uint64_t MX_SCALE_GROUP = 64;
constexpr uint64_t EVEN_ALIGN = 2;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t UB_DIVIDE_NUM = 2;
constexpr uint32_t UB_CALSIZE_PER_BLOCK = 16 * 1024;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t MAX_TURN_NUM = 24;
constexpr int32_t MAX_BASE_K = 128;
constexpr uint64_t MAX_EXPERT_NUM = 256;
constexpr uint64_t COMM_TILE = 8; // 每卡数据分配几次计算
constexpr int64_t MAX_EXPERT_NUM_PER_RANK = 32;
constexpr int64_t MAX_DIM_VALUE = 65536;
constexpr uint64_t MAX_H1_VALUE = 65536;
constexpr uint64_t MAX_N1_VALUE = 65536;
constexpr uint64_t MAX_N2_VALUE = 65536;
constexpr uint64_t MIN_K_VALUE = 2;
constexpr uint64_t MAX_K_VALUE = 8;
constexpr uint32_t MAX_SHARED_H_SHAPE_SIZE = 12288;
constexpr int64_t MAX_BSK_VALUE = 52428800;
constexpr int64_t RECV_SEND_MIN = static_cast<int64_t>((2 * 1024 * 1024) / 2);         // 2M / sizeof(gmmX)
constexpr uint32_t MAX_HCCL_HANDLE_NUM = 63;
inline constexpr int64_t RANK_DEFAULT_NUM = -1;
inline const std::set<int64_t> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64, 128};

class QuantGroupedMatmulAllToAllvTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit QuantGroupedMatmulAllToAllvTilingBase(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context){};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    QuantModePair GetQuantMode(const gert::TilingContext *context, const char *opName);
    const char *opName_{nullptr};

    NpuArch npuArch_;
};
} // namespace Mc2Tiling

#endif // MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_H
