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
 * \file moe_distribute_dispatch_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_BASE_H
#define MOE_DISTRIBUTE_DISPATCH_TILING_BASE_H

#include "op_host/op_tiling/moe_tiling_base.h"
#include "../../../moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_helper.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_tiling.h"
#include "../../op_kernel/moe_distribute_dispatch_tiling_key_decl.h"
#include "op_host/tiling_templates_registry.h"
#include "mc2_hcom_topo_info.h"

namespace optiling {
class MoeDistributeDispatchTilingBase : public MoeTilingBase {
public:
    explicit MoeDistributeDispatchTilingBase(gert::TilingContext *context)
        : MoeTilingBase(context) {};
    ge::graphStatus CheckAttrs(gert::TilingContext *context, const char *nodeName,
                               MoeDistributeDispatchTilingData &tilingData, uint32_t &localMoeExpertNum);
    ge::graphStatus MoeDistributeDispatchA3A5TilingCheckAttr(gert::TilingContext *context, uint32_t &quantMode,
                                                             bool &isScales);
    ge::graphStatus MoeDistributeDispatchA3A5TilingFuncImpl(gert::TilingContext *context);
    virtual ge::graphStatus MoeDistributeDispatchTilingFunc(gert::TilingContext *context) = 0;
    virtual ge::graphStatus CheckEpWorldSizeAttrs(const char *nodeName,
                                                  MoeDistributeDispatchTilingData &tilingData) = 0;

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};
} // namespace optiling

#endif // MOE_DISTRIBUTE_DISPATCH_TILING_BASE_H
