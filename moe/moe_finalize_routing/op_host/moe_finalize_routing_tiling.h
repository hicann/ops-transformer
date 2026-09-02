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
 * \file moe_finalize_routing_tiling.h
 * \brief
 */
#pragma once
#include <cstdint>
#include <limits>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {
const int64_t DTYPE_FLOAT_BIG_K = 20000;
const int64_t DTYPE_FLOAT16_BIG_K = 20001;
const int64_t DTYPE_BF16_BIG_K = 20002;
const int64_t DTYPE_BF16 = 20003;
const int64_t DTYPE_FLOAT_DB = 20004;
const int64_t DTYPE_FLOAT16_DB = 20005;
const int64_t DTYPE_FLOAT_CUTH_K2 = 20006;
const int64_t DTYPE_FLOAT16_CUTH_K2 = 20007;
const int64_t DTYPE_BF16_CUTH_K2 = 20008;
const int64_t DTYPE_FLOAT_CUTH_K4 = 20009;
const int64_t DTYPE_FLOAT16_CUTH_K4 = 20010;
const int64_t DTYPE_BF16_CUTH_K4 = 20011;
const int64_t DTYPE_FLOAT_CUTH = 20012;
const int64_t DTYPE_FLOAT16_CUTH = 20013;
const int64_t DTYPE_BF16_CUTH = 20014;
const int64_t DTYPE_FLOAT_DB_ALL_BIAS = 20015;
const int64_t DTYPE_FLOAT16_DB_ALL_BIAS = 20016;
const int64_t DTYPE_BF16_ALL_BIAS = 20017;
const int64_t DTYPE_FLOAT_CUTH_NETWORK = 20018;
BEGIN_TILING_DATA_DEF(MoeFinalizeRoutingTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, skip2IsNull);
TILING_DATA_FIELD_DEF(int64_t, biasRowNum);          // bias的行数
TILING_DATA_FIELD_DEF(int64_t, totalRowNum);         // skip1的行数
TILING_DATA_FIELD_DEF(int64_t, H);                   // skip1的列数
TILING_DATA_FIELD_DEF(int64_t, normalH);             // skip1的列数切分后满载的列数大小
TILING_DATA_FIELD_DEF(int64_t, unnormalH);           // skip1的列数切分后满载后剩余的列数大小
TILING_DATA_FIELD_DEF(int64_t, hSliceNum);           // skip1的列数切分的次数
TILING_DATA_FIELD_DEF(int64_t, normalK);             // scales的列数切分后满载的列数大小
TILING_DATA_FIELD_DEF(int64_t, unnormalK);           // scales的列数切分后满载后剩余的列数大小
TILING_DATA_FIELD_DEF(int64_t, kSliceNum);           // scales的列数切片的数量
TILING_DATA_FIELD_DEF(int64_t, K);                   // scales的列数
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNum); // 非尾核，每个核处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, normalCoreLoopNum);   // 非尾核，每个核需要的循环次数
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNumPerLoop); // 非尾核，每个核，非尾Loop，每次loop需要处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNumTailLoop); // 非尾核，每个核，尾Loop需要处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNum);           // 尾核处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoopNum);             // 尾核，每个核需要的循环次数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumPerLoop);  // 尾核，每个核，非尾Loop需要处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumTailLoop); // 尾核，每个核，尾Loop需要处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, tilingKey)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFinalizeRouting, MoeFinalizeRoutingTilingData)

struct MoeFinalizeRoutingCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

class MoeFinalizeRoutingTiling {
public:
    ge::graphStatus Init(gert::TilingContext *context);
    ge::graphStatus CalcTilingData();
    void GetTilingData(MoeFinalizeRoutingTilingData &tilingData) const;
    int64_t GetTilingKey() const;

protected:
    ge::graphStatus CheckParamsShape();
    ge::graphStatus SetPlatformInfo();
    ge::graphStatus SetParamInfo();
    ge::graphStatus LoadHKAndCalcTiling();
    ge::graphStatus LoadBiasAndCalcTiling();
    ge::graphStatus OptimizedCutH();
    void CutH();
    int64_t GetAllBiasTilingKey() const;

private:
    gert::TilingContext *context_;
    size_t offset_{0};
    int64_t totalCoreNum_{0};
    int64_t ubSize_{0};
    int64_t usedCoreNum_{0};
    int64_t skip2IsNull_{0};
    int64_t biasRowNum_{0};
    int64_t totalRowNum_{0};
    int64_t h_{0};
    int64_t normalH_{0};
    int64_t unnormalH_{0};
    int64_t hSliceNum_{0};
    int64_t normalK_{0};
    int64_t unnormalK_{0};
    int64_t kSliceNum_{0};
    int64_t k_{0};
    int64_t normalCoreHandleNum_{0};
    int64_t normalCoreLoopNum_{0};
    int64_t normalCoreHandleNumPerLoop_{0};
    int64_t normalCoreHandleNumTailLoop_{0};
    int64_t tailCoreHandleNum_{0};
    int64_t tailCoreLoopNum_{0};
    int64_t tailCoreHandleNumPerLoop_{0};
    int64_t tailCoreHandleNumTailLoop_{0};
    ge::DataType dataType_;
    int inputDataTypeSize_{0};
    bool isCanLoadH_{true};
    bool isCanLoadAllBias_{false};
    bool isOptimizedCutH_{false};
};
} // namespace optiling
