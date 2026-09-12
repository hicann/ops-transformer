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
 * \file allto_all_kc_quant_matmul_tiling_base.h
 * \brief
 */
#ifndef ALLTO_ALL_KC_QUANT_MATMUL_TILING_BASE_H
#define ALLTO_ALL_KC_QUANT_MATMUL_TILING_BASE_H

#pragma once
#include "../../../op_kernel/arch35/allto_all_matmul_tiling_data.h"
#include "../../../op_kernel/arch35/allto_all_matmul_tiling_key.h"
#include "../allto_all_matmul_tiling_base.h"
#include "./allto_all_matmul_fit_balance_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"
#include "securec.h"

namespace MC2Tiling {
using QuantType = MC2Tiling::AlltoAllMatmulFitBalanceTiling::QuantType;
using namespace optiling;
using namespace mc2_matmul_v3_advanced;
constexpr size_t X1_QUANTMODE_VALUES = 7;
constexpr size_t X2_QUANTMODE_VALUES = 2;
constexpr size_t FP8_E5M2_VALUES = 35;
constexpr size_t FP8_E4M3_VALUES = 36;
class AllToAllKcQuantMatmulTilingBase : public AllToAllMatmulTilingBase {
    friend class AlltoAllKcQuantMatmulHelper;

public:
    explicit AllToAllKcQuantMatmulTilingBase(gert::TilingContext *context);
    ~AllToAllKcQuantMatmulTilingBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    CutResult GetCutResOfCommAndCompute() override;
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoKcQuantMMTiling();
    ge::graphStatus SetHcclTiling();
    void SetUserWorkSpace();
    ge::graphStatus CheckKcTensorFormat(const gert::TilingContext *context, const char *opName);
    ge::graphStatus SetKcDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                      TilingContextInfo &contextInfo);

    void SetTilingInfo(AlltoAllMatmulTilingInfo &tilingInfo) const;
    void PrintAlltoAllKcQuantMatmulTilingData(AlltoAllQuantMatmulTilingData &outTilingData);

private:
    AlltoAllQuantMatmulTilingData localTilingData_{};
    uint64_t mm_mvalue_len = 0;
    QuantType matmulQuantType_ = QuantType::KC_QUANT;
    void PrintAlltoAllKcQuantMatmulTilingInfo(const std::string &opName, AlltoAllMatmulTilingInfo &tilingInfo);
    void PrintKcQuantMMV3TilingData(const std::string &opName,
                                    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling);
    void PrintExtendMatmulTiling(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling);
};

class AlltoAllKcQuantMatmulHelper : public Mc2AdaptiveSlidingWindowTiling {
public:
    AlltoAllKcQuantMatmulHelper(AllToAllKcQuantMatmulTilingBase &AllToAllKcQuantMatmulTilingBase,
                                DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &out, uint64_t &mmMvalueLen);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape &GetScaleShape(const size_t index) override;
    const gert::StorageShape *GetPertokenShape(const size_t index) override;
    const gert::StorageShape *GetBiasShape(const size_t index) override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    uint64_t mm_len = 0;
    AllToAllKcQuantMatmulTilingBase &tilingProcesser_;
    QuantType matmulQuantType_ = QuantType::KC_QUANT;
};
} // namespace MC2Tiling
#endif
