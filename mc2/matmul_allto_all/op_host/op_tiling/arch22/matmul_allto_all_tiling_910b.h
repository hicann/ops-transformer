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
 * \file matmul_allto_all_tiling_910b.h
 * \brief
 */

#ifndef MATMUL_ALLTO_ALL_TILING_910B_H
#define MATMUL_ALLTO_ALL_TILING_910B_H

#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "../matmul_allto_all_tiling_base.h"
#include "../common/matmul_allto_all_util_tiling.h"
#include "../../../op_kernel/arch22/matmul_allto_all_tiling.h"
#include "../../../op_kernel/arch22/matmul_allto_all_tiling_key.h"

namespace MC2Tiling {
using namespace optiling;

struct MatmulAlltoAllTilingValue {
    int32_t value = -1;
    std::map<int, std::vector<std::vector<int>>> conditionMap = {};
    explicit MatmulAlltoAllTilingValue(int32_t v = -1, std::map<int, std::vector<std::vector<int>>> m = {})
        : value(v),
          conditionMap(std::move(m))
    {}
};

class MatmulAlltoAllTiling910B : public MatmulAllToAllTilingBase {
public:
    explicit MatmulAlltoAllTiling910B(gert::TilingContext *context);
    ~MatmulAlltoAllTiling910B() override = default;

protected:
    bool IsCapable() override; // 恒为true
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckOpInputInfo(MatmulAlltoAllInfo &info);
    ge::graphStatus CheckAndSetAttrsInfo(MatmulAlltoAllInfo &info);
    ge::graphStatus CheckTensorDataType(MatmulAlltoAllInfo &info);
    ge::graphStatus CheckShapeInfo(MatmulAlltoAllInfo &info);
    ge::graphStatus DoMmCommTiling(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info);
    void DoTwoRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info);
    void DoFourRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info);
    void DoEightRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info);
    void CalTilingParam(CoCTiling &cocTilingData, const std::map<int *, MatmulAlltoAllTilingValue> &TilingParamMap,
                        MatmulAlltoAllInfo &info);
    int32_t GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                        std::map<int, std::vector<std::vector<int>>> conditionMap);
    ge::graphStatus SetHcclTiling(MatmulAlltoAllTilingData *tilingData);
    void PrintMatmulAlltoAllTilingData(CoCTiling &cocTilingData, MatmulAlltoAllInfo &info);
    void SetTilingKey();

private:
    uint64_t tilingKey_;
    bool needTransX2 = false;
    bool hasBias = false;
    uint32_t quantBiasType = 0;
    uint32_t quantType = 0;
    uint32_t tileM0 = 0;
    uint32_t tileN0 = 0;
    uint32_t numBlocks = 1U;
};
} // namespace MC2Tiling
#endif
