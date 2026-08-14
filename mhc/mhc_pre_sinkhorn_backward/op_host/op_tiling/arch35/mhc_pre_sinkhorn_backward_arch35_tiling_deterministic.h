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
 * \file mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.h
 * \brief
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_DETERMINISTIC_H_
#define MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_DETERMINISTIC_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_data_arch35.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_key_arch35.h"
#include "mhc_pre_sinkhorn_backward_arch35_tiling_base.h"
#include "err/ops_err.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

class MhcPreSinkhornBackwardArch35DeterminiticTiling : public MhcPreSinkhornBackwardArch35Tiling {
public:
    explicit MhcPreSinkhornBackwardArch35DeterminiticTiling(gert::TilingContext *context)
        : MhcPreSinkhornBackwardArch35Tiling(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void DumpTilingInfo() override;
    ge::graphStatus DoLibApiTiling() override;

private:
    ge::graphStatus SetTilingData();
    void DoUbTiling();

private:
    int64_t bs_{0};
    int64_t bsTaskCount_{0};
    int64_t tailBsTaskCount_{0};
    int64_t bsLoopDataLen_{0};
    int64_t cLoopDataLen_{0};
    int64_t ubBlock_{0};
    int64_t usedAivNum_{0};
    int64_t cBlockLoops_{0};
    int64_t cBlockTailLoops_{0};
    int64_t cBlockCount_{0};
    int64_t cBlockTailCount_{0};
    int64_t cBlockTailTailCount_{0};
    int64_t finalUsedAivNum_{0};
};

} // namespace optiling
#endif // MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_DETERMINISTIC_H_
