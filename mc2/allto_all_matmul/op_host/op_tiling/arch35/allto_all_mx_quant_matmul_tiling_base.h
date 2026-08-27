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
 * \file allto_all_mx_quant_matmul_tiling_base.h
 * \brief
 */
#ifndef ALLTO_ALL_MX_QUANT_MATMUL_TILING_BASE_H
#define ALLTO_ALL_MX_QUANT_MATMUL_TILING_BASE_H

#pragma once
#include "../../../op_kernel/arch35/allto_all_matmul_tiling_data.h"
#include "../../../op_kernel/arch35/allto_all_matmul_tiling_key.h"
#include "../allto_all_matmul_tiling_base.h"
#include "./allto_all_matmul_fit_balance_tiling.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
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
constexpr size_t X1_QUANTMODE_VALUES = 6;
constexpr size_t X2_QUANTMODE_VALUES = 6;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr uint64_t MX_SCALE_ALIGN = 64;
constexpr uint64_t MX_SCALE_BLOCK_M = 1;
constexpr uint64_t MX_SCALE_BLOCK_K = 32;
constexpr uint64_t MX_SCALE_BLOCK_N = 1;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t BIT_NUMBER = 1;
class AllToAllMxQuantMatmulTilingBase : public AllToAllMatmulTilingBase {
    friend class AlltoAllMxQuantMatmulHelper;

public:
    explicit AllToAllMxQuantMatmulTilingBase(gert::TilingContext *context);
    ~AllToAllMxQuantMatmulTilingBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    CutResult GetCutResOfCommAndCompute() override;
    void AlignCutResForKernel(CutResult &cutRes, uint64_t mValue) const;
    ge::graphStatus CheckMxQuantTensorDataType(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckX2Transpose(const gert::TilingContext *context, const char *opName,
                                     const OpAttrIndexSchema &indexSchema);
    ge::graphStatus CheckMxQuantShapeInfo(const gert::TilingContext *context, const char *opName,
                                          const OpAttrIndexSchema &indexSchema);
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoMxQuantMMTiling();
    ge::graphStatus SetHcclTiling();
    ge::graphStatus CheckGroupSize(const gert::TilingContext *context, const char *opName,
                                   const OpAttrIndexSchema &indexSchema);
    void SetUserWorkSpace();
    ge::graphStatus CheckMxTensorFormat(const gert::TilingContext *context, const char *opName);
    ge::graphStatus SetMxDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                      TilingContextInfo &contextInfo);

    void SetTilingInfo(AlltoAllMatmulTilingInfo &tilingInfo) const;
    void SetHcommTilingInfo(CommTilingData &tilingInfo) const;
    ge::graphStatus DoHcommMxQuantMMTiling();
    void PrintAlltoAllMxQuantMatmulTilingData(AlltoAllQuantMatmulTilingData &outTilingData);

private:
    AlltoAllQuantMatmulTilingData localTilingData_;
    Apace::hcommAllToAllMatmulTilingData hcommTilingData_;
    bool isMxFp4_ = false;
    bool usingApaceImpl_ = false;
    uint64_t mmMvalueLen_ = 0;
    QuantType matmulQuantType_ = QuantType::MXFP8_QUANT;
    void PrintAlltoAllMxQuantMatmulTilingInfo(const std::string &opName, AlltoAllMatmulTilingInfo &tilingInfo);
    void PrintMxQuantMMV3TilingData(const std::string &opName,
                                    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling);
    void PrintExtendMatmulTiling(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling);
};

class AlltoAllMxQuantMatmulHelper : public Mc2AdaptiveSlidingWindowTiling {
public:
    AlltoAllMxQuantMatmulHelper(AllToAllMxQuantMatmulTilingBase &allToAllMxQuantMatmulTilingBase,
                                DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &out, uint64_t &mmMvalueLen_);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::StorageShape *GetPertokenShape(const size_t index) override;
    const gert::Shape &GetScaleShape(const size_t index) override;
    const gert::StorageShape *GetBiasShape(const size_t index) override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantMatmulInfo);
    ge::graphStatus PostTiling() override;

private:
    uint64_t mmLen_ = 0;
    AllToAllMxQuantMatmulTilingBase &tilingProcesser_;
};
} // namespace MC2Tiling
#endif
