/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file all_gather_matmul_tiling_v2.h
 * \brief
 */

#ifndef __ALL_GATHER_MATMUL_TILING_V2__
#define __ALL_GATHER_MATMUL_TILING_V2__

#pragma once
#include "../all_gather_matmul_tiling_base_v2.h"
#include "register/tilingdata_base.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "mc2_matmul_tiling_cfg.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "../../../op_kernel/all_gather_matmul_tiling_data.h"

namespace optiling {

using namespace mc2_matmul_v3_advanced;

class AllGatherMatmulTilingV2 : public AllGatherMatmulTilingBase {
public:
    explicit AllGatherMatmulTilingV2(gert::TilingContext *context);
    ~AllGatherMatmulTilingV2() override = default;
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg, Mc2MMRegisterCfg &registerCfg,
                                     Mc2MatMulV3TilingData &tilingData);

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus PostTiling() override;

    CutResult GetTilingResult() override;

    Mc2Tiling::RCSTiling &MutableRCSTilingData()
    {
        return allGatherMatmulTilingDataV2_->param;
    }

    Mc2MatMulV3TilingData &MutableMC2MatmulV3LocalTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3LocalTilingData;
    }
    Mc2MatMulV3TilingData &MutableMC2MatmulV3TileTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3TileTilingData;
    }
    Mc2MatMulV3TilingData &MutableMC2MatmulV3TailTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3TailTilingData;
    }

    void PrintAllTilingData();
    ge::graphStatus DoVersion2Tiling();
    ge::graphStatus SetMc2Hcomm(Mc2Tiling::RCSTiling &rcsCfg);
    ge::graphStatus SetRawTilingData();
    ge::graphStatus CheckInput() override;

private:
    Mc2Tiling::AllGatherMatmulTilingDataV2 allGatherMatmulTilingDataV2Self_;
    Mc2Tiling::AllGatherMatmulTilingDataV2 *allGatherMatmulTilingDataV2_;
    uint64_t myWorkSpaceSize_{0U};
    Mc2MatMulV3Args mmV3Args_;
    Mc2MatmulV3CompileInfo compileInfo_;
};
ge::graphStatus AllGatherMatmulTilingAIVModeFunc(gert::TilingContext *context);
} // namespace optiling
#endif
