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
 * \file fp_matmul_allto_all_tiling_base.h
 * \brief
 */

#ifndef FP_MATMUL_ALLTO_ALL_TILING_BASE_H
#define FP_MATMUL_ALLTO_ALL_TILING_BASE_H

#include "securec.h"
#include <algorithm>
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "../matmul_allto_all_tiling_base.h"
#include "../common/matmul_allto_all_util_tiling.h"
#include "../../../op_kernel/arch35/matmul_allto_all_tiling_data.h"
#include "../../../op_kernel/arch35/matmul_allto_all_tiling_key.h"

namespace MC2Tiling {

using namespace optiling;
using namespace mc2_matmul_v3_advanced;

class FpMatmulAllToAllTilingBase : public MatmulAllToAllTilingBase {
public:
    explicit FpMatmulAllToAllTilingBase(gert::TilingContext *context);
    ~FpMatmulAllToAllTilingBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    CutResult GetTilingResult() override;

    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg, Mc2MMRegisterCfg &registerCfg,
                                     Mc2MatMulV3TilingData &tilingData);
    ge::graphStatus DoMMTiling();
    ge::graphStatus SetHcclTiling();
    void SetTilingInfo(MatmulAlltoAllTilingInfo &tilingInfo) const;
    uint8_t ChooseAddBiasSplitMode() const;
    void PrintMatmulAlltoAllTilingData(MatmulAlltoAllTilingData &outTilingData);

private:
    MatmulAlltoAllTilingData localTilingData_;

    Mc2MatMulV3Args mmV3Args_;
    Mc2MatmulV3CompileInfo compileInfo_;

    void PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling);
    void PrintMatmulAlltoAllTilingInfo(const std::string &opName, MatmulAlltoAllTilingInfo &tilingInfo);
};
} // namespace MC2Tiling
#endif
