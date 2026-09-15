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
 * \file all_gather_matmul_tiling_a5.h
 * \brief
 */
#ifndef __ALL_GATHER_MATMUL_TILING_A5_H__
#define __ALL_GATHER_MATMUL_TILING_A5_H__
#include "../all_gather_matmul_tiling_base.h"

namespace optiling {
class AllGatherMatmulTilingA5 : public AllGatherMatmulTilingBase {
public:
    CutResult GetCutResult(Mc2Tiling::AllGatherMatmulTilingData &tilingData, mc2tiling::TilingArgs &args) override;
    ge::graphStatus CheckValidRank(Mc2Tiling::AllGatherMatmulTilingData *tilingData,
                                   const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK,
                                   gert::TilingContext *context, uint32_t rankSize) override;
    void SetSocParam(Mc2Tiling::AllGatherMatmulTilingData *tilingData, const char *group) override;
    std::string GetAlgConfig(Mc2Tiling::AllGatherMatmulTilingData *tilingData,
                             const mc2tiling::TilingArgs &args) override;
};
} // namespace optiling
#endif //__ALL_GATHER_MATMUL_TILING_A5_H__
