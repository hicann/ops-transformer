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
 * \file moe_distribute_dispatch_teardown_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_BASE_H_
#define MOE_DISTRIBUTE_DISPATCH_TEARDOWN_TILING_BASE_H_

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/moe_tiling_base.h"
#include "../../op_kernel/moe_distribute_dispatch_teardown_tiling.h"

namespace optiling {
class MoeDistributeDispatchTeardownTilingBase : public MoeTilingBase {
public:
    explicit MoeDistributeDispatchTeardownTilingBase(gert::TilingContext *context)
        : MoeTilingBase(context),
          nodeName_(context->GetNodeName()) {};

protected:
    const char *socTilingName_;
    const char *nodeName_;
    MoeDistributeDispatchTeardownTilingData *tilingData_ = nullptr;
    std::string groupEp_;

    uint64_t GetTilingKey() const;

    const ge::graphStatus CheckRequiredAttrValue();
    ge::graphStatus GetRequiredAttrAndSetTilingData();
    const ge::graphStatus CheckOptionalAttrValue();
    ge::graphStatus GetOptionalAttrAndSetTilingData();

    const ge::graphStatus CheckTensorShape();
    const ge::graphStatus CheckTensorDataType();
    const ge::graphStatus CheckHcclBuffSize();
    ge::graphStatus SetWorkSpace();
    void SetTilingKey();
    void SetHcommCfg();
    void SetPlatformInfo();
    const void PrintTilingDataInfo();

    const bool CheckInputTensorShapeDim();
    const bool CheckOutputTensorShapeDim();
    const bool CheckTensorShapeRelation();
    const bool CheckTensorShapeSize();
    const bool CheckInputTensorDataType();
    const bool CheckOutputTensorDataType();
    const bool CheckRelationTensorDataType();

    ge::graphStatus MoeDistributeDispatchTeardownTilingFuncImpl();
};
} // namespace optiling
#endif
