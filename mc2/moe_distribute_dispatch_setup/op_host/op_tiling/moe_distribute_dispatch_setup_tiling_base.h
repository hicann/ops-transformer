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
 * \file moe_distribute_dispatch_setup_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_BASE_H_
#define MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_BASE_H_

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/moe_tiling_base.h"
#include "../../op_kernel/moe_distribute_dispatch_setup_tiling.h"

namespace optiling {
class MoeDistributeDispatchSetupTilingBase : public MoeTilingBase {
public:
    explicit MoeDistributeDispatchSetupTilingBase(gert::TilingContext *context)
        : MoeTilingBase(context),
          nodeName_(context->GetNodeName()) {};
    enum class TensorType {
        INPUT = 0,
        OUTPUT = 1,
        OPTIONINPUT = 2
    };

protected:
    const char *socTilingName_;
    std::string nodeName_;
    MoeDistributeDispatchSetupTilingData *tilingData_ = nullptr;
    std::string groupEp_;

    uint64_t GetTilingKey() const;
    const void PrintTilingDataInfo();

    const ge::graphStatus CheckRequiredAttrValue();
    ge::graphStatus GetRequiredAttrAndSetTilingData();
    const ge::graphStatus CheckSharedExpertAttrValue(const uint32_t sharedExpertNum,
                                                     const uint32_t sharedExpertRankNum);
    const ge::graphStatus CheckOptionalAttrValue();
    ge::graphStatus GetOptionalAttrAndSetTilingData();
    ge::graphStatus GetComplexAttrAndSetTilingData();
    const ge::graphStatus CheckOneTensorDim(std::string name, TensorType tensortype, uint32_t index, uint32_t dims);
    const ge::graphStatus CheckInputTensorDim();
    const ge::graphStatus CheckOptionalInputTensorDim();
    const ge::graphStatus CheckOutputTensorDim();
    const ge::graphStatus CheckTensorDim();
    const ge::graphStatus CheckTensorShapeRelation();
    const ge::graphStatus CheckInputTensorDataType();
    const ge::graphStatus CheckOptionalInputTensorDataType();
    const ge::graphStatus CheckOutputTensorDataType();
    const ge::graphStatus CheckTensorDataType();
    ge::graphStatus CheckTensorShapeSizeAndSetTilingData();
    const ge::graphStatus CheckComplexTensorShapeSize();
    ge::graphStatus CheckCalcTensorShapeSizeAndSetTilingData();

    void SetTilingKey();
    const ge::graphStatus CheckHcclBuffSize();
    ge::graphStatus SetWorkspace();
    void SetPlatformInfo();

    virtual void SetHcommCfg();
    ge::graphStatus MoeDistributeDispatchSetupTilingFuncImpl();
};
} // namespace optiling
#endif
