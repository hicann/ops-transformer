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
 * \file matmul_all_to_all_tiling_data.h
 * \brief Serialized tiling data for the MatmulAlltoAll Apace path (Udma and Hcomm variants).
 */

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../../tiling/matmul_tiling/quant_matmul_tiling_data.h"
#include "../../../tiling/matmul_tiling_data.h"
#include "../../../tiling/tiling_data_struct_define.h"
#include "../../../core/aiv_comm/collective_comm_context.h"

struct MatmulAllToAllTilingData {
    CommTilingData commTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
    QuantMatmulTilingData tailQbmmTilingData;
    QuantMatmulTilingData localQbmmTilingData;
};

struct CommContext {
    Apace::AivComm::CommUdmaContext udmaCtx;
    Apace::AivComm::CommUbmemContext ubmemCtx;
};

// MX 量化（MXFP4/MXFP8/HiF4）Apace 路径的 tiling data
struct hcommMatmulAllToAllTilingData {
    Mc2InitTiling mc2InitTiling;   // HcclCommunicator 初始化片
    Mc2CcTiling mc2CcTiling;       // HCCL AllToAll 命令片
    CommTilingData commTilingData; // 通算切分(splitAxisTileSize/Cnt/TailSize/Cnt)
    uint32_t rankDim{0};           // 通信 rank 数(发送缓冲 rank 区数)
    uint32_t rankM{0};             // 本 rank 的 M 行数(发送缓冲 rank 区布局基数)
    uint64_t hcclDataType{0};      // HCCL 通信数据类型枚举(通信交换 mm 结果, host 按 geCType 转换)
    QuantMatmulTilingData tileQbmmTilingData; // 主块批的 Blaze qbmm tiling
    QuantMatmulTilingData tailQbmmTilingData; // 尾块批的 Blaze qbmm tiling（tailCnt=0 时无效）
};

// 非量化（FP16/BF16）Apace 路径的 tiling data
struct hcommNonQuantMatmulAllToAllTilingData {
    Mc2InitTiling mc2InitTiling;       // HcclCommunicator 初始化片
    Mc2CcTiling mc2CcTiling;           // HCCL AllToAll 命令片
    CommTilingData commTilingData;     // 通算切分(splitAxisTileSize/Cnt/TailSize/Cnt)
    uint32_t rankDim{0};               // 通信 rank 数(发送缓冲 rank 区数)
    uint32_t rankM{0};                 // 本 rank 的 M 行数(发送缓冲 rank 区布局基数)
    uint64_t hcclDataType{0};          // HCCL 通信数据类型枚举(通信交换 mm 结果, host 按 geCType 转换)
    MatmulTilingData tileMmTilingData; // 主块批的 Blaze mm tiling
    MatmulTilingData tailMmTilingData; // 尾块批的 Blaze mm tiling（tailCnt=0 时无效）
};
