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
 * \file moe_gating_top_k_backward_tiling_arch35.h
 * \brief MoeGatingTopKBackward tiling class declaration for Arch35 (ascend950)
 */

#ifndef MOE_GATING_TOP_K_BACKWARD_TILING_ARCH35_H
#define MOE_GATING_TOP_K_BACKWARD_TILING_ARCH35_H

#include "log/log.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "moe_gating_top_k_backward_tiling.h"
#include "../op_kernel/arch35/moe_gating_top_k_backward_struct.h"
#include "op_host/tiling_util.h"

namespace optiling {

class MoeGatingTopKBackwardTilingArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeGatingTopKBackwardTilingArch35(gert::TilingContext *context);
    ~MoeGatingTopKBackwardTilingArch35() override = default;

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
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckXNorm();
    ge::graphStatus CheckGradY();
    ge::graphStatus CheckExpertIdx();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    ge::graphStatus CalcMaxRows();
    ge::graphStatus SplitRows();
    void DumpTiling();

    const gert::Shape *xNormShape_ = nullptr;
    const gert::Shape *gradYShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outputGradXShape_ = nullptr;

    ge::DataType gradYDtype_;

    int64_t needCoreNum_ = 0;
    int64_t perCoreRows_ = 0;
    int64_t lastCoreRows_ = 0;
    int64_t baseRows_ = 0;
    int64_t perLoopTimes_ = 0;
    int64_t perTailRows_ = 0;
    int64_t lastLoopTimes_ = 0;
    int64_t lastTailRows_ = 0;
    int64_t tokenCount_ = 0;
    int64_t expertCount_ = 0;
    int64_t k_ = 0;
    int64_t gradYDtypeSize_ = 0;
    int64_t renorm_ = 1;
    int64_t normType_ = 0;
    float routedScalingFactor_ = 1.0f;
    float eps_ = 1e-20f;
};

} // namespace optiling
#endif // MOE_GATING_TOP_K_BACKWARD_TILING_ARCH35_H
