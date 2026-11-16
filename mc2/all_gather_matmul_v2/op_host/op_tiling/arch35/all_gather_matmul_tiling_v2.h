/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_tiling_data.h"

namespace optiling {

using namespace mc2_matmul_v3_advanced;

class AllGatherMatmulTilingV2 : public AllGatherMatmulTilingBase {
public:
    explicit AllGatherMatmulTilingV2(gert::TilingContext* context);
    ~AllGatherMatmulTilingV2() override = default;
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg& tilingCfg, Mc2MMRegisterCfg& registerCfg,
                                     Mc2MatMulV3TilingData& tilingData);

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus PostTiling() override;

    [[nodiscard]] uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    CutResult GetTilingResult() override;

    Mc2Tiling::RCSTiling& MutableRCSTilingData()
    {
        return allGatherMatmulTilingDataV2_->param;
    }

    Mc2MatMulV3TilingData& MutableMC2MatmulV3LocalTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3LocalTilingData;
    }
    Mc2MatMulV3TilingData& MutableMC2MatmulV3TileTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3TileTilingData;
    }
    Mc2MatMulV3TilingData& MutableMC2MatmulV3TailTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3TailTilingData;
    }

    void PrintAllTilingData();
    ge::graphStatus DoVersion2Tiling();
    ge::graphStatus SetMc2Hcomm(Mc2Tiling::RCSTiling& rcsCfg);
    ge::graphStatus SetRawTilingData();
    ge::graphStatus CheckInput() override;

private:
    ge::graphStatus DoApaceTiling();
    ge::graphStatus FillApaceMmTile(const Mc2MatMulV3TilingData& localTiling);
    // apace kernel L1 半区布局为 A|B|bias（bmm_block_mmad_fragment.h 分区公式，约束 A+B+bias ≤ L1/2），
    // 与 local tiling 自身的 A1/B1 双缓冲约束不等价，零余量档位叠加 bias 会导致 bias 槽越界/重叠。
    // 超限仅降 stepK（不降 baseM/baseN：非 2 幂折半破坏 16 行分形对齐）；stepK=1 时用量由 L0 tile
    // 硬约束（baseM·baseK·dtype ≤ L0A、baseN·baseK·dtype ≤ L0B）保证必然满足，仍超限即档位本身
    // 病态，报错拒绝（同 3rd gmm_qbmm_tiling "L1 space overflow" 处置）
    ge::graphStatus FitApaceMmTileToL1(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t& stepK) const;

    Mc2Tiling::AllGatherMatmulTilingDataV2 allGatherMatmulTilingDataV2Self_;
    Mc2Tiling::AllGatherMatmulTilingDataV2* allGatherMatmulTilingDataV2_;
    // A5(DAV_3510) 非量化 apace 模板性能劣化，仅 A6(DAV_9201) 切 apace，其余走原 V2 路径
    bool usingApaceImpl_{false};
    uint64_t myWorkSpaceSize_{0U};
    Mc2MatmulV3CompileInfo compileInfo_;
    Mc2MatMulV3Args mmV3Args_;
    Apace::hcommAllGatherMatmulTilingData hcommTilingDataSelf_{};
};
ge::graphStatus AllGatherMatmulTilingAIVModeFunc(gert::TilingContext* context);
} // namespace optiling
#endif
