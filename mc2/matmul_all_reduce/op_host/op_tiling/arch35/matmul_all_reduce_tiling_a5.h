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
 * \file matmul_all_reduce_tiling_a5.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_A5_H
#define MATMUL_ALL_REDUCE_TILING_A5_H

#include "../matmul_all_reduce_tiling_base.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mc2_matmul_tiling_cfg.h"
#include "../../../op_kernel/arch35/matmul_all_reduce_tiling_struct_ar35.h"

namespace optiling {
using namespace mc2_matmul_v3_advanced;

class MatmulAllReduceTilingA5 : public MatmulAllReduceTilingBase {
public:
    explicit MatmulAllReduceTilingA5(gert::TilingContext* context);
    MatmulAllReduceTilingA5(gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo,
                            Mc2Tiling::MatmulAllReduce910TilingDataA5* out);
    ~MatmulAllReduceTilingA5() override = default;

protected:
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg& tilingCfg, Mc2MMRegisterCfg& registerCfg,
                                     Mc2MatMulV3TilingData& tilingData);
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus GetWorkspaceSize() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;

    ge::graphStatus Do910Tiling();

    Mc2Tiling::RCSTiling& MutableRCSTilingData() override;

    ::TCubeTiling& MutableTCubeTileTilingData() override
    {
        return matmulAllReduce910TilingData_.mC2Mmv3TileTilingData.tCubeTiling;
    }

    ::TCubeTiling& MutableTCubeTailTilingData() override
    {
        return matmulAllReduce910TilingData_.mC2Mmv3TailTilingData.tCubeTiling;
    }

    inline Mc2MatMulV3TilingData& MutableMC2MmV3TileTilingData()
    {
        return matmulAllReduce910TilingData_.mC2Mmv3TileTilingData;
    }

    inline Mc2MatMulV3TilingData& MutableMC2MmV3TailTilingData()
    {
        return matmulAllReduce910TilingData_.mC2Mmv3TailTilingData;
    }

    void PrintExtendMatmulTiling(bool isTail) override;
    void DoEmptyTensorTiling() override;
    ge::graphStatus GetWorkspaceSizeForA2ARSAG();
    ge::graphStatus SetMc2HcommAllReduce(const char* groupName, const uint32_t reduceType);
    ge::graphStatus SetMc2HcommTwoShot(const char* groupName, const uint32_t reduceType);
    ge::graphStatus SetMc2Hcomm();
    ge::graphStatus CheckInput() override;
    CutResult GetTilingResult() override;

private:
    ge::graphStatus CheckAxisSize();
    ge::graphStatus CheckX1X2();
    Mc2Tiling::MatmulAllReduce910TilingDataA5 matmulAllReduce910TilingDataSelf_{};
    Mc2Tiling::MatmulAllReduce910TilingDataA5& matmulAllReduce910TilingData_;
    uint64_t myWorkSpaceSize_{0U};
    Mc2MatMulV3Args mmV3Args_;
    Mc2MatmulV3CompileInfo compileInfo_;
};

} // namespace optiling
#endif // MATMUL_ALL_REDUCE_TILING_A5_H
