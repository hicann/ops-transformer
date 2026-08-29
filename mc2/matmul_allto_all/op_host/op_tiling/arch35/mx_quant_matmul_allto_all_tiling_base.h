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
 * \file mx_quant_matmul_allto_all_tiling_base.h
 * \brief
 */
#ifndef MX_QUANT_MATMUL_ALLTO_ALL_TILING_BASE_H
#define MX_QUANT_MATMUL_ALLTO_ALL_TILING_BASE_H

#pragma once
#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "../common/matmul_allto_all_util_tiling.h"
#include "../matmul_allto_all_tiling_base.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"
#include "quant_batch_matmul_v3/op_kernel/arch35/qbmv3_arch35_tiling_data.h"
#include "../../../op_kernel/arch35/matmul_allto_all_tiling_data.h"
#include "../../../op_kernel/arch35/matmul_allto_all_tiling_key.h"

namespace MC2Tiling {
using namespace optiling;
using namespace DequantBmm;
constexpr size_t X1_QUANTMODE_VALUES = 6;
constexpr size_t X2_QUANTMODE_VALUES = 6;
constexpr uint64_t MX_SCALE_OFFSET = 64;
constexpr uint64_t EVEN_ALIGN = 2;
constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr uint64_t MX_GROUP_SIZE_K = 32;
constexpr uint64_t MX_GROUP_SIZE_M = 1;
constexpr uint64_t MX_GROUP_SIZE_N = 1;

class MxQuantMatmulAllToAllTilingBase : public MatmulAllToAllTilingBase {
    friend class MxQuantMatmulAlltoAllHelper;

public:
    explicit MxQuantMatmulAllToAllTilingBase(gert::TilingContext *context);
    ~MxQuantMatmulAllToAllTilingBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    CutResult GetTilingResult() override;
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoMxQuantMMTiling();
    ge::graphStatus SetHcclTiling();
    ge::graphStatus CheckMxQuantTensorDataType(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckX2Transpose(const gert::TilingContext *context, const char *opName,
                                     const OpAttrIndexSchema &indexSchema);
    ge::graphStatus CheckMxTensorFormat(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckMxQuantShapeInfo(const gert::TilingContext *context, const char *opName,
                                          const OpAttrIndexSchema &indexSchema);
    ge::graphStatus CheckMxQuantMatrixMulShapes(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckMxQuantScaleShapes(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckMxQuantInputShapesValid(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckQuantGroupSize(const gert::TilingContext *context, const char *opName);
    ge::graphStatus SetMxDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                      TilingContextInfo &contextInfo);
    ge::graphStatus CheckMxQuantScaleDataType(const gert::TilingContext *context, const char *opName);
    void SetTilingInfo(MatmulAlltoAllTilingInfo &tilingInfo) const;
    void PrintMxQuantMatmulAlltoAllTilingData(QuantMatmulAlltoAllTilingData &outTilingData);

private:
    QuantMatmulAlltoAllTilingData localTilingData_;
    uint64_t mmMvalueLen_ = 0;
    bool isMxfp4_ = false;
    void PrintMxQuantMatmulAlltoAllTilingInfo(const std::string &opName, MatmulAlltoAllTilingInfo &tilingInfo);
    void PrintMxQuantMMV3TilingData(const std::string &opName,
                                    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling);
    void PrintExtendMatmulTiling(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling);
};

class MxQuantMatmulAlltoAllHelper : public Mc2AdaptiveSlidingWindowTiling {
public:
    MxQuantMatmulAlltoAllHelper(MxQuantMatmulAllToAllTilingBase &mxQuantMatmulAllToAllTilingBase,
                                DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &out, uint64_t &mmMvalueLen);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::StorageShape *GetPertokenShape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape &GetScaleShape(const size_t index) override;
    const gert::StorageShape *GetBiasShape(const size_t index) override;
    const gert::StorageShape *GetOffsetShape(const size_t index);
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantMatmulInfo);

private:
    uint64_t mmLen_ = 0;
    MxQuantMatmulAllToAllTilingBase &tilingProcesser_;
};
} // namespace MC2Tiling
#endif
