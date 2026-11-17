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
 * \file mhc_pre_sinkhorn_tiling_key.h
 * \brief compile-time tiling keys shared by the arch22 kernel entry and the host tiling
 */

#ifndef MHC_PRE_SINKHORN_TILING_KEY_H
#define MHC_PRE_SINKHORN_TILING_KEY_H

#include <cstdint>
#include "ascendc/host_api/tiling/template_argument.h"

namespace MhcPreSinkhorn {
// Compile-time tiling keys of the arch22 (ascend910b) kernel entry. The kernel is
// instantiated per key via template <uint32_t schMode> + if constexpr; the numeric
// values must stay identical to the runtime SetTilingKey values written by the host
// tiling (the legacy TILING_KEY_IS dispatch used exactly 0/1/2).
enum class MhcPreSinkhornTilingKey : uint32_t {
    // MPMD core split: Stage1/Stage2 run on separate vector cores (pre/post vs
    // Sinkhorn), chosen for large bs when needGrad is false.
    MPMD_KSPLIT = 0,
    // Original single-schedule k-split kernel: Part1 + full-flow Part2 on one core
    // set, forced for small bs (bs <= 32) or when needGrad is true.
    ORIGINAL_KSPLIT = 1,
    // m-split kernel: K (= hcMult * d) exceeds the nd2nz srcDValue limit (65535),
    // so B goes through MatmulImpl instead of the k-split copy path.
    MSPLIT_ND2NZ_EXCEED = 2,
};
} // namespace MhcPreSinkhorn

// Template parameter space for opc kernel instantiation: a single UINT template
// argument schMode whose value list [0, 1, 2] equals the enum values above. With a
// value list equal to its own indices, the encoded tiling key equals the schMode
// value, so the host keeps writing SetTilingKey(static_cast<uint64_t>(enum value))
// with zero behavior change (keys 0/1/2 as before).
ASCENDC_TPL_ARGS_DECL(MhcPreSinkhorn, ASCENDC_TPL_UINT_DECL(schMode, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2), );

// Instantiation list: one kernel per schMode value (0/1/2).
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 0, 1, 2), ), );

#endif // MHC_PRE_SINKHORN_TILING_KEY_H
