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
 * \file gbsag_epilogue_packet_ub_layout.hpp
 * \brief packet softmax ub layout
 */

#ifndef CATLASS_EPILOGUE_BLOCK_PACKET_UB_LAYOUT_HPP
#define CATLASS_EPILOGUE_BLOCK_PACKET_UB_LAYOUT_HPP

#include <cstdint>

namespace NpuArch::Epilogue::Block {

// Keep softmax and prepD buffers in disjoint UB regions.
//
// Softmax uses [0, 104 KiB); prepD uses [104 KiB, 192 KiB).
// CAST storage is sized for 32-row chunks because the full two-buffer
// representation does not fit in the 192 KiB user-addressable UB window.
//
// 注意：本算子 host 侧锁死 headDim=128（EZ1001 参数校验），输入/CAST 按 128
// 定容；若将来解除该限制，本布局需按新 headDim 重排。
struct PacketSoftmaxUbLayout {
    static constexpr uint64_t MAX_ROWS_PER_AIV = 64;
    static constexpr uint64_t INPUT_BUFFER_BYTES = MAX_ROWS_PER_AIV * 128 * 2;      // 16KB, fp16
    static constexpr uint64_t CAST_BUFFER_BYTES = (MAX_ROWS_PER_AIV / 2) * 128 * 4; // 16KB, fp32 半行块
    static constexpr uint64_t INPUT_COUNT = 2;
    static constexpr uint64_t D_WIDTH = 8;
    static constexpr uint64_t D_BYTES = MAX_ROWS_PER_AIV * D_WIDTH * sizeof(float); // 2KB
    static constexpr uint64_t PREPD_BASE = 104 * 1024;
    static constexpr uint64_t D_OFFSET = PREPD_BASE + (INPUT_BUFFER_BYTES + CAST_BUFFER_BYTES) * INPUT_COUNT; // 168KB
    static constexpr uint64_t TEMP_OFFSET = 172 * 1024;
    static constexpr uint64_t TEMP_BUFFER_BYTES = 20 * 1024; // front 按此自适应分裂
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_PACKET_UB_LAYOUT_HPP
