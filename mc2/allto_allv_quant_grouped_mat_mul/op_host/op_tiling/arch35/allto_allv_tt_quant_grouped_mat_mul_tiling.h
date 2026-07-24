/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_tt_quant_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef ALLTO_ALLV_TT_GROUPED_MATMUL_QUANT_TILING_H
#define ALLTO_ALLV_TT_GROUPED_MATMUL_QUANT_TILING_H

#include "allto_allv_quant_grouped_mat_mul_tiling_common.h"

namespace optiling {

class AlltoAllvTTQuantGmmTiling : public AlltoAllvQuantGmmTilingCommon {
public:
    explicit AlltoAllvTTQuantGmmTiling(gert::TilingContext *context) : AlltoAllvQuantGmmTilingCommon(context) {};

protected:
    // Tiling base
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    // quant base
    ge::graphStatus CheckScaleFormatAndDtype() const override;
    ge::graphStatus CheckInputDtype() const override;
    ge::graphStatus CheckScaleShape() const override;
    ge::graphStatus CheckQuantMode() const override;
    ge::graphStatus DoGmmTiling(uint64_t gmmMSize) override;
    void GetPermuteOutSize() override;

private:
    void SetGMMQuantParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData,
                           uint32_t groupNum = 1) const;
    void SetTilingArray(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData,
                        uint64_t M, uint64_t N, uint64_t K, uint32_t groupNum = 1) const;
    void SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData,
                         uint64_t M, uint64_t N, uint64_t K, bool transB) const;
};
} // namespace optiling
#endif // ALLTO_ALLV_TT_GROUPED_MATMUL_QUANT_TILING_H
