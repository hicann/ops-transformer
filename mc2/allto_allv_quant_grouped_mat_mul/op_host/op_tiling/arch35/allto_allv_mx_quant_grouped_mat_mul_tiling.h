/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_mx_quant_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H
#define ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H

#include "allto_allv_quant_grouped_mat_mul_tiling_common.h"
#include "../../../../3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.h"

namespace optiling {
class AlltoAllvMXQuantGmmTiling : public AlltoAllvQuantGmmTilingCommon {
    friend class AlltoAllvMXQuantGmmTilingHelper;

public:
    explicit AlltoAllvMXQuantGmmTiling(gert::TilingContext *context)
        : AlltoAllvQuantGmmTilingCommon(context) {};

protected:
    // tiling base
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
    ge::graphStatus CheckGmmInputDtype(ge::DataType gXDataType, ge::DataType gWtDataType,
                                       ge::DataType gYDataType) const;
    ge::graphStatus CheckMmInputDtype(ge::DataType gXDataType, ge::DataType gWtDataType, ge::DataType gYDataType) const;
    ge::graphStatus CheckFp4Input(ge::DataType gXDataType, ge::DataType gWtDataType) const;
    ge::graphStatus CheckGmmScale() const;
    ge::graphStatus CheckMmScale() const;
    ge::graphStatus CheckQuantGroupSize() const;
    ge::graphStatus CheckGmmScaleShape() const;
    ge::graphStatus CheckMmScaleShape() const;
};

class AlltoAllvMXQuantGmmTilingHelper : public Mc2GroupedMatmulTiling::Mc2GroupedQbmmTiling {
public:
    AlltoAllvMXQuantGmmTilingHelper(AlltoAllvMXQuantGmmTiling &AlltoAllvMXQuantGmmTiling)
        : Mc2GroupedQbmmTiling(AlltoAllvMXQuantGmmTiling.context_)
    {}
    const Mc2GroupedMatmulTilingData::GMMQuantTilingData &GetAlltoAllvQuantHelperData() const
    {
        return tilingData_;
    }
    bool AnalyzeAttrs() override
    {
        return true;
    }
    bool AnalyzeDtype() override
    {
        return true;
    }
    bool AnalyzeInputs() override
    {
        return true;
    }
    void Reset() override {}
    ge::graphStatus SetInputParams(uint64_t M, uint64_t N, uint64_t K, bool transB);
    ge::graphStatus Process();
    void SetGroupNum(uint32_t groupNum)
    {
        gmmGroupNum_ = groupNum;
    }

private:
    uint32_t gmmGroupNum_ = 1;
};

} // namespace optiling
#endif // ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H
