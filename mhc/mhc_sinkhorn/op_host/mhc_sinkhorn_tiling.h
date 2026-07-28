/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mhc_sinkhorn_tiling.h
 * \brief mhc_sinkhorn_tiling
 */

#ifndef MHC_SINKHORN_TILING_H_
#define MHC_SINKHORN_TILING_H_
#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../op_kernel/arch35/mhc_sinkhorn_struct.h"
#include "../op_kernel/arch35/mhc_sinkhorn_tiling_key.h"
#include "../../mhc_sinkhorn_common/op_host/arch35/mhc_sinkhorn_base_tiling.h"

namespace optiling {
using Ops::Transformer::OpTiling::TilingBaseClass;


struct MhcSinkhornCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

class MhcSinkhornTiling : public MhcSinkhornBaseTiling {
public:
    explicit MhcSinkhornTiling(gert::TilingContext *context) : MhcSinkhornBaseTiling(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckInputDtype();
    void SplitByCoreNum();
    void SetTilingData();

private:
    int64_t T_ = 0;
    int64_t n_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t tNormCoreLoop_ = 0;
    int64_t tUbFactor_ = 0;
    int64_t tUbFactorTail_ = 0;
    int64_t tTailCoreLoop_ = 0;
    int64_t tUbTailTail_ = 0;
    int64_t tNormCore_ = 0;

    int64_t tilingKey_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    float eps_ = 1E-6f;
    int64_t num_iters_ = 20;
    int64_t outFlag_ = 0;
    int64_t xDimNum_ = 0;
    int64_t yDimNum_ = 0;
    int64_t xDtypeSize_ = 0;

    gert::Shape outputShape_;
    ge::DataType xDtype_ = ge::DT_UNDEFINED;
    MhcSinkhornTilingData tilingData_;
    const char *opName_ = "MhcSinkhorn";
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MHC_SINKHORN_TILING_H_
