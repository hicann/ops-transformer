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
 * \file bidirection_lstm_tiling.h
 */
#ifndef GROUPED_MATMUL_QUANT_TILING_H
#define GROUPED_MATMUL_QUANT_TILING_H
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
namespace optiling {

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t NUM_PER_REPEAT_FLOAT16 = 128;
constexpr int32_t NUM_PER_BLOCK_FLOAT16 = 16;
constexpr int32_t SYS_WORKSPACE_910B = 16 * 1024 * 1024;
constexpr int32_t FRACTAL_FLOAT16 = 16;
enum class GroupedMatmulQuantTilingKey : uint64_t {
    NORMAL = 10000001,
    UNDFINED = 10000099
};

struct GroupedMatmulQuantCompileInfo {};
struct GroupedMatmulQuantParam {
    // platform
    uint64_t CoreNum;
    uint64_t UBSize;
    uint64_t L1Size;
    uint64_t L2Size;
    uint64_t L0ASize;
    uint64_t L0BSize;
    uint64_t L0CSize;

    ge::DataType dataType;

    uint64_t noGroup;
    uint64_t originE;
    uint64_t originM;
    uint64_t originN;
    uint64_t originK;
    uint64_t scaleK;
    uint64_t scaleGroupSize;
    uint64_t fracN;
    uint64_t fracK;
    uint64_t splitK;

    uint64_t clearBaseN;
    uint64_t clearOutLoop;
    uint64_t clearOutTailN;
    uint64_t clearOutTailCoreNum;
    uint64_t castBaseN;
    uint64_t castOutLoop;
    uint64_t castOutTailN;
    uint64_t castOutTailCoreNum;
};

BEGIN_TILING_DATA_DEF(GroupedMatmulQuantTilingData)

TILING_DATA_FIELD_DEF(uint32_t, CoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dataType);

TILING_DATA_FIELD_DEF(uint32_t, UBSize);
TILING_DATA_FIELD_DEF(uint32_t, L1Size);
TILING_DATA_FIELD_DEF(uint32_t, L0ASize);
TILING_DATA_FIELD_DEF(uint32_t, L0BSize);
TILING_DATA_FIELD_DEF(uint32_t, L0CSize);

TILING_DATA_FIELD_DEF(uint32_t, noGroup);
TILING_DATA_FIELD_DEF(uint32_t, originE);
TILING_DATA_FIELD_DEF(uint32_t, originM);
TILING_DATA_FIELD_DEF(uint32_t, originN);
TILING_DATA_FIELD_DEF(uint32_t, originK);
TILING_DATA_FIELD_DEF(uint32_t, scaleK);
TILING_DATA_FIELD_DEF(uint32_t, scaleGroupSize);
TILING_DATA_FIELD_DEF(uint32_t, fracN);
TILING_DATA_FIELD_DEF(uint32_t, fracK);
TILING_DATA_FIELD_DEF(uint32_t, splitK);

TILING_DATA_FIELD_DEF(uint32_t, clearBaseN);
TILING_DATA_FIELD_DEF(uint32_t, clearOutLoop);
TILING_DATA_FIELD_DEF(uint32_t, clearOutTailN);
TILING_DATA_FIELD_DEF(uint32_t, clearOutTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, castBaseN);
TILING_DATA_FIELD_DEF(uint32_t, castOutLoop);
TILING_DATA_FIELD_DEF(uint32_t, castOutTailN);
TILING_DATA_FIELD_DEF(uint32_t, castOutTailCoreNum);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulQuant, GroupedMatmulQuantTilingData)

class GroupedMatmulQuantTiling {
public:
    ge::graphStatus runTiling(gert::TilingContext* context);

protected:
    bool GetPlatformInfo(gert::TilingContext* context);

    bool GetCheckAttr(gert::TilingContext* context);

    bool CheckTensorShape(gert::TilingContext* context, gert::Shape& shape, uint64_t ndim, std::vector<uint64_t> dims);

    bool CheckInOutShapes(gert::TilingContext* context);

    bool GetTilingData(gert::TilingContext* context);

    bool SetTilingData(gert::TilingContext* context);

    bool SetLaunchInfo(gert::TilingContext* context);

private:
    GroupedMatmulQuantTilingKey tilingKey;
    GroupedMatmulQuantTilingData tilingData;
    GroupedMatmulQuantParam _Params;
};

} // namespace optiling
#endif
