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
 * \file moe_distribute_combine_setup_tiling_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_TILING_BASE_H_
#define MOE_DISTRIBUTE_COMBINE_SETUP_TILING_BASE_H_

#include "op_host/op_tiling/moe_tiling_base.h"
#include "../../op_kernel/moe_distribute_combine_setup_tiling_key.h"
#include "../../op_kernel/moe_distribute_combine_setup_tiling_data.h"

namespace MC2Tiling {
class MoeDistributeCombineSetupTilingBase : public optiling::MoeTilingBase {
public:
    explicit MoeDistributeCombineSetupTilingBase(gert::TilingContext *context)
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
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckRequiredAttrValue();
    ge::graphStatus GetRequiredAttrAndSetTilingData();
    ge::graphStatus CheckSharedExpertAttrValue();
    ge::graphStatus CheckOptionalAttrValue();
    ge::graphStatus GetOptionalAttrAndSetTilingData();
    ge::graphStatus GetComplexAttrAndSetTilingData();

    ge::graphStatus CheckTensorDataTypeAndFormat();
    bool CheckTensorDataType();
    bool CheckTensorDataFormat();
    ge::graphStatus CheckTensorDim();
    ge::graphStatus CheckTensorShapeRelation();
    ge::graphStatus CheckTensorShapeSizeAInMoeRank();
    ge::graphStatus CheckTensorShapeSizeAInSharedRank();
    ge::graphStatus CheckTensorShapeSizeAndSetTilingData();
    ge::graphStatus CheckCalcTensorShapeSizeAndSetTilingData();

    ge::graphStatus MoeDistributeCombineSetupTilingFuncImpl();
    ge::graphStatus CheckOneTensorDim(std::string name, TensorType tensortype, uint32_t index, uint32_t dims);
    ge::graphStatus CheckInputTensorDim();
    ge::graphStatus CheckOutputTensorDim();
    ge::graphStatus SetWorkspace();
    ge::graphStatus CheckHcclBuffSize();
    void SetTilingKey();
    void SetPlatformInfo();
    void PrintTilingDataInfo();
    virtual ge::graphStatus CheckEpWorldSize() = 0;
    virtual ge::graphStatus CheckMoeExpertNum() = 0;
    virtual ge::graphStatus CheckSharedExpertAttr() = 0;
    virtual ge::graphStatus CheckMoeExpertNumPerRank();
    virtual ge::graphStatus CheckTensorShapeSize(int64_t h, int64_t bs, int64_t k) = 0;
    virtual void SetHcommCfg() = 0;

    const char *socTilingName_ = nullptr;
    std::string nodeName_;
    MoeDistributeCombineSetupTilingData *tilingData_ = nullptr;
    std::string groupEp_;
};
} // namespace MC2Tiling
#endif // MOE_DISTRIBUTE_COMBINE_SETUP_TILING_BASE_H_
