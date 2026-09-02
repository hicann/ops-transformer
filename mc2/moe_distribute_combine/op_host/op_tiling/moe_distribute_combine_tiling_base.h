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
 * \file moe_distribute_combine_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TILING_BASE_H
#define MOE_DISTRIBUTE_COMBINE_TILING_BASE_H

#include "op_host/op_tiling/moe_tiling_base.h"
#include "../../../moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.h"
#include "../../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_tiling.h"
#include "../../op_kernel/moe_distribute_combine_tiling_key_decl.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
class MoeDistributeCombineTilingBase : public MoeTilingBase {
public:
    explicit MoeDistributeCombineTilingBase(gert::TilingContext *context)
        : MoeTilingBase(context) {};
    ge::graphStatus MoeDistributeCombineTilingFunc(gert::TilingContext *context);
    bool CheckEpWorldSizeAttrs(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
                               const char *nodeName);
    bool CheckAttrs(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData, const char *nodeName,
                    uint32_t &localMoeExpertNum);
    bool CheckTensorShape(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
                          const char *nodeName, bool isShared, uint32_t localExpertNum);
    ge::graphStatus SetWorkspace(gert::TilingContext *context, const char *nodeName);
    ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
                                            const char *nodeName, std::string &groupEp, std::string &groupTp,
                                            uint32_t &commQuantMode);
    ge::graphStatus SetHCommCfg(const gert::TilingContext *context, MoeDistributeCombineTilingData *tiling,
                                const std::string groupEp, const std::string groupTp, const uint32_t tpWorldSize);
    ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeDistributeCombineTilingData *tilingData,
                                 const char *nodeName, uint32_t localMoeExpertNum);
    void PrintTilingDataInfo(const char *nodeName, MoeDistributeCombineTilingData &tilingData);
    virtual ge::graphStatus MoeDistributeCombineTilingFuncImpl(std::string &socVersion,
                                                               gert::TilingContext *context) = 0;
    virtual bool CheckEpWorldSize(const char *nodeName, uint32_t epWorldSize) = 0;

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};
} // namespace optiling

#endif // MOE_DISTRIBUTE_COMBINE_TILING_A2A3_H
