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
 * \file text_gated_residual_grad_chunk_state.h
 * \brief State shared by all stages of one full-batch chunk.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_CHUNK_STATE_H
#define TEXT_GATED_RESIDUAL_GRAD_CHUNK_STATE_H

#include <cstdint>

namespace TextGatedResidualGrad {

// A chunk is the unit of retained dW operands. Parity selects one of the two
// independent V1 operand buffers used for cross-chunk overlap.
struct ChunkState {
    uint32_t rowBegin{0};
    uint32_t rows{0};
    uint32_t taskCount{0};
    uint32_t parity{0};
};

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_CHUNK_STATE_H
