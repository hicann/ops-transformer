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
 * \file allto_all_matmul_tiling_910b.h
 * \brief
 */

#ifndef ALLTO_ALL_MATMUL_TILING_910_H
#define ALLTO_ALL_MATMUL_TILING_910_H

#include "../../../op_kernel/arch22/allto_all_matmul_tiling.h"
#include "../../../op_kernel/arch22/allto_all_matmul_tiling_key.h"
#include "../allto_all_matmul_tiling_base.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "securec.h"

namespace MC2Tiling {
using namespace optiling;

struct AlltoAllMatmulTilingValue {
    int32_t value = -1;
    std::map<int, std::vector<std::vector<int>>> conditionMap = {};
    explicit AlltoAllMatmulTilingValue(int32_t v = -1, std::map<int, std::vector<std::vector<int>>> m = {})
        : value(v),
          conditionMap(std::move(m))
    {}
};

class AlltoAllMatmulTiling910b : public AllToAllMatmulTilingBase {
public:
    explicit AlltoAllMatmulTiling910b(gert::TilingContext *context);
    ~AlltoAllMatmulTiling910b() override = default;

protected:
    bool IsCapable() override; // 恒为true
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    void CalcQuantWorkspaceSize(const CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    void CalcQuantTokenNumPerUb(const CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    ge::graphStatus CheckOpInputInfo(AlltoAllMatmulInfo &info);
    ge::graphStatus CheckAndSetAttrsInfo(AlltoAllMatmulInfo &info);
    ge::graphStatus CheckTensorDataType(AlltoAllMatmulInfo &info);
    ge::graphStatus CheckShapeInfo(AlltoAllMatmulInfo &info);
    ge::graphStatus DoMmCommTiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    // basic和A4W4的tiling
    void DoTwoRankTiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    void DoFourRankTiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    void DoEightRankTiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    // A16W8的tiling
    void AlltoAllMatmulNPU910BTwoRankA16W8Tiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    void AlltoAllMatmulNPU910BFourRankA16W8Tiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    void AlltoAllMatmulNPU910BEightRankA16W8Tiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    // A16W4的tiling
    void AlltoAllMatmulNPU910BTwoRankA16W4Tiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    void AlltoAllMatmulNPU910BFourRankA16W4Tiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    void AlltoAllMatmulNPU910BEightRankA16W4Tiling(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);
    void CalTilingParam(CoCTiling &cocTilingData, const std::map<int *, AlltoAllMatmulTilingValue> &TilingParamMap,
                        AlltoAllMatmulInfo &info);
    void DecodeTilingData(int32_t code, CoCTiling &cocTilingData);
    int32_t GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                        std::map<int, std::vector<std::vector<int>>> conditionMap);
    ge::graphStatus SetHcclTiling(AlltoAllMatmulTilingData *tilingData);
    void PrintAlltoAllMatmulTilingData(CoCTiling &cocTilingData, AlltoAllMatmulInfo &info);

private:
    bool x2Transpose = false;
    bool hasBias = false;
    uint32_t quantType = TILINGKEY_TPL_NOQUANT;
    uint32_t biasDtype_ = 0;
    uint32_t rankSize = 0;
    uint32_t orgM = 0;
    uint32_t orgN = 0;
    uint32_t orgK = 0;
    uint32_t numBlocks = 1U;
    size_t quantWorkspaceSize = 0;
};
} // namespace MC2Tiling
#endif
