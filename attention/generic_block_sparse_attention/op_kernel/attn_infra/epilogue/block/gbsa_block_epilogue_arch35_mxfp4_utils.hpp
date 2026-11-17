/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_ARCH35_MXFP4_UTILS_HPP
#define GBSA_BLOCK_EPILOGUE_ARCH35_MXFP4_UTILS_HPP

#include "../../arch/gbsa_resource.hpp"
#include <cstdint>

namespace GbsaKernelArch35::Mxfp4 {

// Runtime layout data, calculated by the kernel before block construction.
struct UBufTileHelper {
    uint32_t D, Q;
    uint32_t R;
    uint32_t KV;
    static constexpr uint32_t VECTOR_BYTES = 256;
    static constexpr uint32_t STATS_SLOTS = 4;
    static constexpr uint32_t S_OFFSET = 0;
    uint32_t S_SLOT_BYTES;
    uint32_t PV_OFFSET;
    uint32_t PV_SLOTS;
    uint32_t PV_SLOT_BYTES;
    uint32_t O_OFFSET;
    uint32_t P_OFFSET;
    uint32_t P_SLOT_BYTES;
    uint32_t PSCALE_SLOT_BYTES;
    uint32_t LOCAL_MAX_OFFSET;
    uint32_t LOCAL_MAX_SLOT_BYTES;
    uint32_t GROUP_SUM_OFFSET;
    uint32_t TILE_G_OFFSET;
    uint32_t OUT_G_OFFSET;
    uint32_t TILE_DM_OFFSET;
    uint32_t LOCAL_ROW_SUM_OFFSET;
    static constexpr uint32_t LOCAL_ROW_SUM_BYTES = STATS_SLOTS * VECTOR_BYTES;
    uint32_t OUT_L_OFFSET;
    uint32_t TILE_MAX_OFFSET;
    uint32_t DESC_OFFSET;
    uint32_t KSCALE_GROUPS;
    static constexpr uint32_t KSCALE_INPUT_BYTES = 4096;
    uint32_t KSCALE_INPUT_SLOTS;
    uint32_t KSCALE_OUTPUT_BYTES;
    uint32_t KSCALE_OUTPUT_OFFSET;
    uint32_t LOCAL_DESC_OFFSET;
    uint32_t LSE_PACKED_OFFSET;
    uint32_t DESC_BYTES;
    uint32_t OUTPUT_SCRATCH_OFFSET;
    uint32_t OUTPUT_SCRATCH_BYTES;
    uint32_t LSE_OFFSET;
    uint32_t PSCALE_OFFSET;
    uint32_t TOTAL_BYTES;
};

} // namespace GbsaKernelArch35::Mxfp4
#endif
