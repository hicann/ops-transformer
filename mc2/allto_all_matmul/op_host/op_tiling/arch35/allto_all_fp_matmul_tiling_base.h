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
 * \file allto_all_fp_matmul_tiling_base.h
 * \brief
 */

#ifndef ALLTO_ALL_FP_MATMUL_TILING_BASE_H
#define ALLTO_ALL_FP_MATMUL_TILING_BASE_H

#include "../../../op_kernel/arch35/allto_all_matmul_tiling_data.h"
#include "../../../op_kernel/arch35/allto_all_matmul_tiling_key.h"
#include "../allto_all_matmul_tiling_base.h"
#include "./allto_all_matmul_fit_balance_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "securec.h"

namespace MC2Tiling {
using QuantType = MC2Tiling::AlltoAllMatmulFitBalanceTiling::QuantType;
using namespace optiling;
using namespace mc2_matmul_v3_advanced;

class AllToAllFpMatmulTilingBase : public AllToAllMatmulTilingBase {
public:
    explicit AllToAllFpMatmulTilingBase(gert::TilingContext *context);
    ~AllToAllFpMatmulTilingBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    CutResult GetCutResOfCommAndCompute() override;
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoMMTiling();
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg, Mc2MMRegisterCfg &registerCfg,
                                     Mc2MatMulV3TilingData &tilingData);
    ge::graphStatus SetHcclTiling();
    void SetTilingInfo(AlltoAllMatmulTilingInfo &tilingInfo) const;
    // 非量化+有bias时, 选择 add_bias 节点的分核策略
    uint8_t ChooseAddBiasSplitMode() const;

private:
    void PrintAlltoAllMatmulTilingData(AlltoAllMatmulTilingData &alltoAllMatmulTilingData);
    void PrintAlltoAllMatmulTilingInfo(const std::string &opName, AlltoAllMatmulTilingInfo &tilingInfo);
    void PrintMMV3TilingData(const std::string &opName, Mc2MatMulV3TilingData &tiling);

    AlltoAllMatmulTilingData localTilingData_;
    Mc2MatMulV3Args mmV3Args_;
    Mc2MatmulV3CompileInfo mmV3compileInfo_;
    QuantType matmulQuantType_ = QuantType::FP_QUANT;
};

} // namespace MC2Tiling
#endif
