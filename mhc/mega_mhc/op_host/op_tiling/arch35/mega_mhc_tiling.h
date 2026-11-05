/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_H

#include <cstdint>
#include <tiling/tiling_api.h>
#include "op_host/tiling_base.h"
#include "mega_mhc_cube_tiling.h"
#include "mega_mhc_tiling_data.h"

namespace optiling {
struct MegaMhcCompileInfo {};

class MegaMhcTiling final : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    /** Bind the tiling context. */
    explicit MegaMhcTiling(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {}
    ~MegaMhcTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus ParseInputsAndAttributes();
    ge::graphStatus CalcWorkspaceSize();
    void FillCubeTilingData();

    MegaMhcTilingData tilingData_;
    mega_mhc::CubeTilingInfo cubeTiling_;
    mega_mhc::ScheduleMode scheduleMode_ = mega_mhc::ScheduleMode::BS;
    mega_mhc::ScheduleMode dispatchMode_ = mega_mhc::ScheduleMode::BS;
    uint64_t aicCount_ = 0;
    uint64_t aivCount_ = 0;
    uint64_t ubBytes_ = 0;
    uint64_t l1Bytes_ = 0;
    uint64_t l0ABytes_ = 0;
    uint64_t l0BBytes_ = 0;
    uint64_t l0CBytes_ = 0;
    uint64_t totalRows_ = 0;
    uint32_t d_ = 0;
    uint32_t k_ = 0;
    bool hasRms0_ = false;
    bool hasMid_ = false;
    ge::DataType inputType_ = ge::DT_UNDEFINED;
    mega_mhc::ProjectionMode projectionMode_ = mega_mhc::ProjectionMode::FP32;
    float mhcPreNormEps_ = 1.0e-6f;
    float normEps_ = 1.0e-6f;
    float hcPostScale_ = 2.0f;
    float hcEps_ = 1.0e-6f;
    uint64_t workspaceBytes_ = 0;
};
} // namespace optiling

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_H
