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
 * \file tt_quant_grouped_mat_mul_allto_allv_gmm_tiling.h
 * \brief
 */

#ifndef TT_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H
#define TT_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H

#pragma once
#include "../quant_grouped_mat_mul_allto_allv_tiling_common.h"

namespace Mc2Tiling {
namespace Mc2GroupedMatmul {
class TTQuantGroupedMatmulAllToAllvTiling : public QuantGroupedMatmulAllToAllvTilingCommon {
public:
    explicit TTQuantGroupedMatmulAllToAllvTiling(gert::TilingContext* context)
        : QuantGroupedMatmulAllToAllvTilingCommon(context) {};
    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~TTQuantGroupedMatmulAllToAllvTiling() override = default;

protected:
    void Reset();
    bool IsCapable() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
};

} // namespace Mc2GroupedMatmul
} // namespace Mc2Tiling
#endif
