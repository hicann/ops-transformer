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
 * \file matmul_reduce_scatter_tiling.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_TILING_H
#define MATMUL_REDUCE_SCATTER_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#if __has_include("../../common/inc/kernel/mc2_tiling_struct.h")
#include "../../common/inc/kernel/mc2_tiling_struct.h"
#else
#include "../common/inc/kernel/mc2_tiling_struct.h"
#endif

struct ReduceScatterSoc {
    uint32_t commAlg;
	uint32_t isA3;
    uint32_t isStep;
    uint32_t isND2NZ;
    uint32_t baseBD;
    uint32_t baseBN;
};

struct L2cacheUseInfo {
    uint32_t l2CacheFlag;
};

class MatmulReduceScatterTilingData {
public:
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    RCSTiling param;
    TCubeTiling tileTiling;
    TCubeTiling tailTiling;
    TCubeTiling localTiling;
    TileL2Tiling tileL2Tiling;
    TileL2Tiling tailL2Tiling;
    TileL2Tiling localL2Tiling;
	ReduceScatterSoc socParam;
    L2cacheUseInfo l2cacheUseInfo;
};

#endif //__MATMUL_REDUCE_SCATTER_TILING_H__