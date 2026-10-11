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
 * \file kc_quant_matmul_allto_all_tiling_base.h
 * \brief
 */
#ifndef KC_QUANT_MATMUL_ALLTO_ALL_TILING_BASE_H
#define KC_QUANT_MATMUL_ALLTO_ALL_TILING_BASE_H

#pragma once
#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "../common/matmul_allto_all_util_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "../matmul_allto_all_tiling_base.h"
#include "../../../op_kernel/arch35/matmul_allto_all_tiling_data.h"
#include "../../../op_kernel/arch35/matmul_allto_all_tiling_key.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"

namespace MC2Tiling {
using namespace optiling;
using namespace mc2_matmul_v3_advanced;

class KcQuantMatmulAllToAllTilingBase : public MatmulAllToAllTilingBase {
    friend class KcQuantMatmulAlltoAllHelper;

public:
    explicit KcQuantMatmulAllToAllTilingBase(gert::TilingContext* context);
    ~KcQuantMatmulAllToAllTilingBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    CutResult GetTilingResult() override;
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoKcQuantMMTiling();
    ge::graphStatus SetHcclTiling();
    ge::graphStatus CheckKcTensorFormat(const gert::TilingContext* context, const char* opName);
    void SetTilingInfo(MatmulAlltoAllTilingInfo& tilingInfo) const;
    void PrintKcQuantMatmulAlltoAllTilingData(QuantMatmulAlltoAllTilingData& outTilingData);

private:
    QuantMatmulAlltoAllTilingData localTilingData_;
    uint64_t mmMvalueLen = 0;
    void PrintKcQuantMatmulAlltoAllTilingInfo(const std::string& opName, MatmulAlltoAllTilingInfo& tilingInfo);
    void PrintKcQuantMMV3TilingData(const std::string& opName,
                                    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& tiling);
    void PrintExtendMatmulTiling(const std::string& opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& tiling);
};

class KcQuantMatmulAlltoAllHelper : public Mc2AdaptiveSlidingWindowTiling {
public:
    KcQuantMatmulAlltoAllHelper(KcQuantMatmulAllToAllTilingBase& kcQuantMatmulAllToAllTilingBase,
                                DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& out, uint64_t& mmMvalueLen);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape& GetScaleShape(const size_t index) override;
    const gert::StorageShape* GetOffsetShape(const size_t index);
    const gert::StorageShape* GetPertokenShape(const size_t index) override;
    const gert::StorageShape* GetBiasShape(const size_t index) override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo& quantMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    uint64_t mmLen = 0;
    KcQuantMatmulAllToAllTilingBase& tilingProcesser_;
};
} // namespace MC2Tiling
#endif
