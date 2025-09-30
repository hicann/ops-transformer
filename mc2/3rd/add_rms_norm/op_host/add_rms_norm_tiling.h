/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_H_

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(AddRMSNormTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, num_row);
    TILING_DATA_FIELD_DEF(uint32_t, num_col);
    TILING_DATA_FIELD_DEF(uint32_t, block_factor);
    TILING_DATA_FIELD_DEF(uint32_t, row_factor);
    TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
    TILING_DATA_FIELD_DEF(float, epsilon);
    TILING_DATA_FIELD_DEF(float, avg_factor);
    END_TILING_DATA_DEF;

    BEGIN_TILING_DATA_DEF(AddRMSNormRegbaseTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numRow);
    TILING_DATA_FIELD_DEF(uint32_t, numCol);
    TILING_DATA_FIELD_DEF(uint32_t, numColAlign);
    TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
    TILING_DATA_FIELD_DEF(uint32_t, rowFactor);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
    TILING_DATA_FIELD_DEF(float, epsilon);
    TILING_DATA_FIELD_DEF(float, avgFactor);
    TILING_DATA_FIELD_DEF(uint32_t, ubLoop);
    TILING_DATA_FIELD_DEF(uint32_t, colBuferLength);
    TILING_DATA_FIELD_DEF(uint32_t, multiNNum);
    TILING_DATA_FIELD_DEF(uint32_t, isNddma);
    END_TILING_DATA_DEF;

    struct AddRmsNormCompileInfo
    {
        uint32_t totalCoreNum = 0;
        uint64_t totalUbSize = 0;
        platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910_95;
    };

    REGISTER_TILING_DATA_CLASS(AddRmsNorm, AddRMSNormTilingData)
    REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm, AddRMSNormTilingData)

    REGISTER_TILING_DATA_CLASS(AddRmsNorm_1, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(AddRmsNorm_2, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(AddRmsNorm_3, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(AddRmsNorm_1001, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(AddRmsNorm_1002, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(AddRmsNorm_1003, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_1, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_2, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_3, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_1001, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_1002, AddRMSNormRegbaseTilingData)
    REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_1003, AddRMSNormRegbaseTilingData)
    }  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_H_
