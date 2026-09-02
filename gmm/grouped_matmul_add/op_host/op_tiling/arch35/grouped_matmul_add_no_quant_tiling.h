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
 * \file grouped_matmul_add_no_quant_tiling.h
 * \brief
 */
#ifndef __GROUPED_MATMUL_ADD_NO_QUANT_TILING_H
#define __GROUPED_MATMUL_ADD_NO_QUANT_TILING_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "../../../op_kernel/arch35/grouped_matmul_add_tiling_data.h"
#include "grouped_matmul_add_utils_advanced.h"
#include "../../../../grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.h"

namespace optiling {
using namespace GroupedMatmulAdd;

class GroupedMatmulAddNoQuantTiling {
public:
    bool SetTiling(gert::TilingContext *context);

protected:
    bool Init(const gert::TilingContext *context);
    bool GetAttrs(const gert::TilingContext *context);
    bool CalMatMulTiling(const gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr);
    void SetGMMTiling();
    void SetMatMulTiling();
    void PrintTilingResult(const gert::TilingContext *context);
    void FormulateBasicBlock(const GMMCompileInfo *compileInfoPtr, uint32_t remainCoreNum);
    void CalAswtL1Tiling(const GMMCompileInfo *compileInfoPtr);
    void CalcTailBasicBlock(const GMMCompileInfo *compileInfoPtr);
    bool SetCustomParam(gert::TilingContext *context);
    void SetTilingKey(gert::TilingContext *context) const;
    bool SplitKSingleXSingleWeightSingleY(const gert::TilingContext *context, const gert::Shape xShape,
                                          const gert::Shape wShape);

private:
    bool transposeX_ = false;
    bool transposeWeight_ = false;

    uint64_t m_ = 0UL;
    uint64_t k_ = 0UL;
    uint64_t n_ = 0UL;
    int32_t groupType_ = 0;
    uint32_t groupNum_ = 0U;
    uint32_t groupListType_ = 0U;
    uint32_t xKDim_ = 0U;
    uint32_t weightNDim_ = 0U;
    uint32_t xDimNum_ = 0U;
    uint64_t baseM_ = BASE_M_DEFAULT;
    uint64_t baseN_ = BASE_N_DEFAULT;
    uint64_t baseK_ = BASE_K_DEFAULT;
    uint32_t usedCoreNum_ = 0U;
    uint64_t stepKa_ = STEP_K_DEFAULT;
    uint64_t stepKb_ = STEP_K_DEFAULT;
    uint64_t depthA1_ = DEPTH_DEFAULT;
    uint64_t depthB1_ = DEPTH_DEFAULT;
    uint64_t mTailCnt_ = 1UL;
    uint64_t nTailCnt_ = 1UL;

    ge::DataType xDType_ = ge::DT_UNDEFINED;
    ge::DataType weightDtype_ = ge::DT_UNDEFINED;

    GroupedMatmulAdd::GmmAddTilingDataParams tilingData_;
};
} // namespace optiling

#endif // GROUPED_MATMUL_ADD_NO_QUANT_TILING_H
