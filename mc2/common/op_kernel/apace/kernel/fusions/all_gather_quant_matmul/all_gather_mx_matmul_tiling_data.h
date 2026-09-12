/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_mx_matmul_tiling_data.h
 * \brief Tiling data for AllGather + QuantMatmul fusion kernel (URMA + Hcomm/CCU variants)
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "../../../../mc2_tiling_struct.h"

namespace Apace {
namespace AivComm {
// Convenience wrapper so the kernel receives one pointer instead of two.
struct CommContext {
    CommUdmaContext udmaCtx;
    CommUbmemContext ubmemCtx;
};
} // namespace AivComm
} // namespace Apace

#pragma pack(push, 8)
// 8 means 8 bytes aligned
// DFX 头部约定: dumpInfo 必须是第一个成员(offset 0)
struct alignas(8) AllGatherMxMatmulUrmaTilingData {
    Utils::DfxDumpInfo dumpInfo{};
    QuantMatmulTilingData mmTile;
    CommTilingData commTile;
    uint8_t isBias{0};
};
#pragma pack(pop)

// Hcomm/CCU 路径 tiling 结构体（apace 自有，不依赖外部算子 tiling）
namespace Apace {
struct hcommAllGatherMatmulTilingData {
    // DFX 头部约定: dumpInfo 必须是第一个成员(offset 0)
    Utils::DfxDumpInfo dumpInfo{};
    Mc2InitTiling mc2InitTiling;  // HCCL 初始化参数
    Mc2CcTiling mc2CcTiling;      // HCCL CC task 配置
    CommTilingData commTile;      // AllGather 切分参数（splitAxis=M_per_rank, nonSplitAxis=K）
    QuantMatmulTilingData mmTile; // Matmul tiling 参数
    uint64_t gatherLen;           // 0: gather_out 输出可用，通信写入 gatherOut; >0: 用 workspace
};
} // namespace Apace
