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
 * \brief Serialized tiling data passed from the host launcher to the kernel.
 */

#pragma once

#include "../../../tiling/quant_matmul_tiling_data.h"
#include "../../../tiling/comm_tiling_data.h"
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
