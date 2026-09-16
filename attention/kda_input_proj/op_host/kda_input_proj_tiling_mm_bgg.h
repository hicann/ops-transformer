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
 * \file kda_input_proj_tiling_mm_bgg.h
 * \brief Stage1 AIC: Matmul(beta / gate / g) tiling
 */

#ifndef KDA_INPUT_PROJ_TILING_MM_BGG_H
#define KDA_INPUT_PROJ_TILING_MM_BGG_H

#include "kda_input_proj_tiling.h"

namespace optiling {
class KdaInputProjMmBggTiling {
public:
    explicit KdaInputProjMmBggTiling(const KdaInputProjTilingInfo &tilingInfo)
        : tilingInfo_(tilingInfo)
    {}

    ge::graphStatus CalcTiling(KdaInputProjMmBggParams &params) const
    {
        (void)tilingInfo_;
        (void)params;
        // TODO: 用 tilingInfo_.baseParams，AIC 核数经 platform 运行时获取，填充 mmBggParams
        return ge::GRAPH_SUCCESS;
    }

private:
    const KdaInputProjTilingInfo &tilingInfo_;
};
} // namespace optiling

#endif // KDA_INPUT_PROJ_TILING_MM_BGG_H
