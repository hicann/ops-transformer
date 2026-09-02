/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_tiling_arch35.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_TILING_ARCH35_H
#define MOE_DISTRIBUTE_DISPATCH_V2_TILING_ARCH35_H

#include "op_host/op_tiling/moe_tiling_base.h"
#include "../moe_distribute_dispatch_tiling_helper.h"
#include "../moe_distribute_dispatch_tiling_v2.h"

namespace optiling {

ge::graphStatus MoeDistributeDispatchTilingImpl(gert::TilingContext *context, uint32_t opVersion);

class MoeDistributeDispatchTilingA5 : public MoeTilingBase {
public:
    explicit MoeDistributeDispatchTilingA5(gert::TilingContext *context)
        : MoeTilingBase(context) {};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};

class MoeDistributeDispatchV2TilingFuncA5 : public MoeDistributeDispatchV2TilingFuncBase {
public:
    ge::graphStatus MoeDistributeDispatchV2TilingFunc(gert::TilingContext *context) override;
    ge::graphStatus MoeDistributeDispatchTilingFuncImpl(gert::TilingContext *context) override;
    bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                             const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                             const bool isPerformance, DispatchV2Config &config) override;
    uint64_t CalTilingKey(const bool isScales, const uint32_t quantMode, const bool isSetFullMeshV2,
                          bool isLayered) override;
    ge::graphStatus CheckQuantModeMatchScales(gert::TilingContext *context, const char *nodeName, bool isScales,
                                              uint32_t quantMode, DispatchV2Config &config) override;
    ge::graphStatus CheckCommAlgPtr(const char *commAlgPtr, const char *nodeName) override;
    ge::graphStatus CheckQuantModePtr(const int64_t *quantModePtr, const char *nodeName) override;
};
} // namespace optiling

#endif // MOE_DISTRIBUTE_DISPATCH_TILING_ARCH35_H
