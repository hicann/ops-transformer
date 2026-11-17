/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_MMAD_ARCH35_MXFP4_UTILS_HPP
#define GBSA_BLOCK_MMAD_ARCH35_MXFP4_UTILS_HPP

#include "../../arch/gbsa_resource.hpp"
#include <cstdint>

namespace GbsaKernelArch35::Mxfp4 {

// Runtime layout data, calculated by the kernel before block construction.
struct L1TileHelper {
    uint32_t D, Q;
    uint32_t QUERY_TILE;
    uint32_t R;
    uint32_t QUERY_ROWS;
    uint32_t HEAD_CAPACITY;
    uint32_t KV_ROWS;
    uint32_t K_ROWS;
    static constexpr uint32_t Q_SLOTS = 1;
    static constexpr uint32_t K_SLOTS = 3;
    static constexpr uint32_t V_SLOTS = 3;
    static constexpr uint32_t P_SLOTS = 3;
    static constexpr uint32_t Q_OFFSET = 0;
    uint32_t Q_SLOT_BYTES;
    uint32_t K_OFFSET;
    uint32_t K_SLOT_BYTES;
    uint32_t P_OFFSET;
    uint32_t P_SLOT_BYTES;
    uint32_t V_OFFSET;
    uint32_t V_SLOT_BYTES;
    uint32_t QSCALE_OFFSET;
    uint32_t QSCALE_SLOT_BYTES;
    uint32_t KSCALE_OFFSET;
    uint32_t KSCALE_SLOT_BYTES;
    uint32_t PSCALE_OFFSET;
    uint32_t PSCALE_SLOT_BYTES;
    uint32_t VSCALE_OFFSET;
    uint32_t VSCALE_SLOT_BYTES;
    uint32_t TOTAL_BYTES;
    static constexpr uint32_t QK_A_OFFSET = 0;
    static constexpr uint32_t QK_A_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t QK_B_OFFSET = 0;
    static constexpr uint32_t QK_B_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t PV_A_OFFSET = 32 * 1024;
    static constexpr uint32_t PV_A_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t PV_B_OFFSET = 32 * 1024;
    static constexpr uint32_t PV_B_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t QK_COLUMNS = 128;
    static constexpr uint32_t PV_REDUCTION = 256;
    uint32_t PV_COLUMNS;
    static constexpr uint32_t QK_C_OFFSET = 0;
    static constexpr uint32_t QK_C_SLOT_BYTES = 64 * 1024;
    static constexpr uint32_t PV_C_OFFSET = 128 * 1024;
    static constexpr uint32_t PV_C_SLOT_BYTES = 64 * 1024;
};

template <bool Paged>
struct DescriptorTileHelper {
    uint32_t D, Q;
    uint32_t KV;
    uint32_t ROWS;
    uint32_t GROUPS;
    uint32_t GROUPS_PER_BANK;
    uint32_t INPUT_BYTES;
    uint32_t INPUT_SLOTS;
    uint32_t LOCAL_DESC_SLOTS;
    uint32_t LOCAL_DESC_SLOT_BYTES;
};

} // namespace GbsaKernelArch35::Mxfp4

namespace NpuArch::Gemm::Block {

__aicore__ inline void FillMxfp4L1(const AscendC::LocalTensor<uint8_t>& dst, uint32_t bytes, uint16_t value = 0)
{
    AscendC::InitConstValueParams<uint16_t> params(1, bytes / 32, 0, value);
    AscendC::Fill(dst.ReinterpretCast<uint16_t>(), params);
}

} // namespace NpuArch::Gemm::Block
#endif // GBSA_BLOCK_MMAD_ARCH35_MXFP4_UTILS_HPP
