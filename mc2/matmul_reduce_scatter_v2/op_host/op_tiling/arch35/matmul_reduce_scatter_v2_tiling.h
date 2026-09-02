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
 * \file matmul_reduce_scatter_v2_tiling.h
 * \brief
 */
#ifndef __MATMUL_REDUCE_SCATTER_V2_TILING_H__
#define __MATMUL_REDUCE_SCATTER_V2_TILING_H__

#include "../matmul_reduce_scatter_tiling_base_v2.h"
#include "mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "mc2_matmul_tiling_cfg.h"
#include "../../../op_kernel/matmul_reduce_scatter_v2_c_tiling.h"

namespace optiling {
using namespace mc2_matmul_v3_advanced;

class MatmulReduceScatterV2Tiling : public MatmulReduceScatterTilingBase {
public:
    explicit MatmulReduceScatterV2Tiling(gert::TilingContext *context)
        : MatmulReduceScatterTilingBase(context),
          matmulReduceScatterV2TilingData_(&matmulReduceScatterV2TilingDataSelf_)
    {
        matmulReduceScatterV2TilingData_ = context_->GetTilingData<Mc2Tiling::MatmulReduceScatterV2TilingData>();
    }

protected:
    bool IsCapable() override; // done
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus CheckInput() override;
    ge::graphStatus DoAllMatmulTiling();
    void PrintAllTilingData() const;
    CutResult GetTilingResult() override;
    ge::graphStatus SetMc2Hcomm();
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg, Mc2MMRegisterCfg &registerCfg,
                                     Mc2MatMulV3TilingData &tilingData);
    inline Mc2MatMulV3TilingData &MutableMC2MmV3TileTilingData() const
    {
        return matmulReduceScatterV2TilingData_->mC2Mmv3TileTilingData;
    }
    inline Mc2MatMulV3TilingData &MutableMC2MmV3TailTilingData() const
    {
        return matmulReduceScatterV2TilingData_->mC2Mmv3TailTilingData;
    }

private:
    Mc2Tiling::MatmulReduceScatterV2TilingData matmulReduceScatterV2TilingDataSelf_;
    Mc2Tiling::MatmulReduceScatterV2TilingData
        *matmulReduceScatterV2TilingData_; // 用于存储matmulReduceScatter算子的tiling结果
    Mc2MatMulV3Args mmV3Args_;
    Mc2MatmulV3CompileInfo compileInfo_;
};
} // namespace optiling

#endif // __MATMUL_REDUCE_SCATTER_V2_TILING_H__
