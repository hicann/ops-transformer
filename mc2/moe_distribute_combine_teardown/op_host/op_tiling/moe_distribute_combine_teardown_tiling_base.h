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
 * \file moe_distribute_combine_teardown_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_BASE_H_
#define MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_BASE_H_

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/moe_tiling_base.h"
#include "../../op_kernel/moe_distribute_combine_teardown_tiling_key.h"
#include "../../op_kernel/moe_distribute_combine_teardown_tiling_data.h"

namespace MC2Tiling {

class MoeDistributeCombineTeardownTilingBase : public optiling::MoeTilingBase {
public:
    explicit MoeDistributeCombineTeardownTilingBase(gert::TilingContext *context)
        : optiling::MoeTilingBase(context),
          nodeName_(context->GetNodeName()) {};

protected:
    enum TensorType {
        INPUT = 0,
        OUTPUT = 1,
        OPTIONINPUT = 2
    };

    ge::graphStatus DoOpTiling() override;
    bool IsCapable() override;

    ge::graphStatus MoeDistributeCombineTeardownTilingFuncImpl();
    ge::graphStatus CheckAttrs();
    ge::graphStatus CheckAttrsNullptr();
    virtual ge::graphStatus CheckAttrsWithoutRelation() = 0;
    virtual ge::graphStatus CheckAttrsComplex() = 0;
    ge::graphStatus CheckOneTensorDim(std::string name, TensorType tensortype, uint32_t index, uint32_t dims);
    ge::graphStatus CheckInputTensorDim();
    ge::graphStatus CheckOptionalInputTensorDim();
    ge::graphStatus CheckOutputTensorDim();
    ge::graphStatus CheckTensorDim();
    ge::graphStatus CheckTensorShapeRelation();
    ge::graphStatus CheckTensorShapeRelationSecondPart();
    ge::graphStatus CheckTensorShapeRelationThirdPart();
    ge::graphStatus CheckTensorShapeSize();
    virtual ge::graphStatus CheckBsHKSize(int64_t bs, int64_t h, int64_t k);
    ge::graphStatus CheckTensorDataType();
    ge::graphStatus CheckTensorDataTypeSecondPart();
    ge::graphStatus CheckTensorFormat();
    ge::graphStatus SetWorkspace();
    ge::graphStatus CheckHcclBuffsize();
    virtual ge::graphStatus SetHcommCfg() = 0;

    void SetAttrToTilingData();
    void SetDimsToTilingData();
    void SetTilingKey();
    void SetPlatformInfo();
    void PrintTilingDataInfo();

    const char *socTilingName_{nullptr};
    const char *nodeName_{nullptr};
    MoeDistributeCombineTeardownTilingData *tilingData_{nullptr};
    std::string groupEp_;

    uint64_t GetTilingKey() const override;
};

} // namespace MC2Tiling

#endif // MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_BASE_H_
