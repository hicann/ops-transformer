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
 * \file unquant_matmul_all_reduce_tiling_data.h
 * \brief
 */

#ifndef UNQUANT_MATMUL_ALL_REDUCE_TILING_DATA_H
#define UNQUANT_MATMUL_ALL_REDUCE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_tiling_data.h"
#else
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_tiling_data.h"
#endif

namespace Mc2Tiling {

// L2 cache alignment requirement: 8 bytes for optimal memory access
static constexpr size_t L2_CACHE_ALIGNMENT = 8;

#pragma pack(push, 8)
struct alignas(L2_CACHE_ALIGNMENT) Mc2L2cacheTilePara {
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(L2_CACHE_ALIGNMENT) UnQuantMatmulAllReduceTilingData {
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    Mc2MatmulV3TilingData tilematmulTiling;
    Mc2MatmulV3TilingData tailmatmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(L2_CACHE_ALIGNMENT) MatmulAllReduce910TilingData {
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    Mc2MatmulV3TilingData tilematmulTiling;
    Mc2MatmulV3TilingData tailmatmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(L2_CACHE_ALIGNMENT) MatmulAllReduceTilingData {
    Mc2Tiling::Mc2Msg msg;
    Mc2Tiling::RCSTiling param;
    AscendC::tiling::TCubeTiling matmulTiling;
    AscendC::tiling::TCubeTiling tailTiling;
    AscendC::tiling::TCubeTiling matmulTiling2;
    Mc2Tiling::Mc2L2cacheTilePara tileL2cacheTiling;
    Mc2Tiling::Mc2L2cacheTilePara tailL2cacheTiling;
};
#pragma pack(pop)

} // namespace Mc2Tiling
#endif
