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
 * \file mhc_pre_sinkhorn_backward_arch35_tiling_base.h
 * \brief
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_BASE_H_
#define MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_BASE_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_data_arch35.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_key_arch35.h"
#include "err/ops_err.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

class MhcPreSinkhornBackwardArch35Tiling : public TilingBaseClass {
public:
    explicit MhcPreSinkhornBackwardArch35Tiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void DumpTilingInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;

protected:
    ge::graphStatus CheckShapeBase(int64_t batchSize, int64_t seqLength, int64_t n, int64_t c, bool is3D = false);

private:
    ge::graphStatus CheckShape(int64_t batchSize, int64_t seqLength, int64_t n, int64_t c);
    ge::graphStatus SetTilingData();
    void DoUbTiling();

protected:
    uint64_t coreNumAic_{0};
    uint64_t coreNumAiv_{0};
    uint64_t ubSize_{0};

    int64_t batchSize_{0};
    int64_t seqLength_{0};
    int64_t n_{0};
    int64_t c_{0};
    int64_t c0_{0};
    int64_t c1_{0};
    int64_t coreTaskCount_{0};
    int64_t tile_{0};
    int64_t tileUB_{0};
    int64_t skIterCount_{0};
    float hcEps_{0};

    int64_t mm1K_{0};
    int64_t mm1M_{0};
    int64_t mm1N_{0};
    int64_t mm2K_{0};
    int64_t mm2M_{0};
    int64_t mm2N_{0};

    const char *opName = "MhcPreSinkhornBackward";
};

} // namespace optiling
#endif // MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_BASE_H_
