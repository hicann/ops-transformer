/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file text_gated_residual_grad_tiling_data_arch35.h
 * \brief TextGatedResidualGrad tiling data shared between host and device.
 */

#ifndef TEXT_GATED_RESIDUAL_GRAD_TILING_DATA_ARCH35_H
#define TEXT_GATED_RESIDUAL_GRAD_TILING_DATA_ARCH35_H

#include <cstdint>

class TextGatedResidualGradTilingDataArch35 {
public:
    int64_t n{0};
    int64_t c{0};
    int64_t h{0};
    int64_t r{0};
    int64_t d{0};
    int64_t taskCount{0};
    int64_t workspaceSlots{0};
    int64_t splitkFactor{0};
    int64_t hasGamma{0};
    int64_t hasGate{0};
    int64_t aicNum{0};
    int64_t aivNum{0};
    int64_t scheduleBeatCount{0};
    int64_t workspaceSize{0};
    int64_t vectorUbBytes{0};  // Detected platform UB capacity in bytes.
    int64_t pipeStages{0};     // 0 = serial blocks; 2 = input/output ping-pong stages.
    int64_t vectorRowBlock{1}; // V1/V6 rows per block, computed by the host UB budget.
    int64_t scheduleMode{0};   // 0 = SyncAll each step; 1 = bounded producer-consumer ready/free credits.
};

#endif // TEXT_GATED_RESIDUAL_GRAD_TILING_DATA_ARCH35_H
