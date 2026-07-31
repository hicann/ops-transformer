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
 * \file grouped_quant_basic_api_matmul_tiling.h
 * \brief
 */
#ifndef GROUPED_QUANT_BASIC_API_MATMUL_TILING_H
#define GROUPED_QUANT_BASIC_API_MATMUL_TILING_H

#include "grouped_quant_matmul_tiling.h"
namespace optiling {

class GroupedQmmBasicApiTiling : public GroupedQmmTiling {
public:
    explicit GroupedQmmBasicApiTiling(gert::TilingContext *context);
    ~GroupedQmmBasicApiTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    virtual void Reset();
    ge::graphStatus CalL1Tiling();

private:
    ge::graphStatus CalL1Depth(uint64_t leftL1Size);
    uint64_t GetDepthWithHighBW(uint64_t mnL1) const;
    void ModifyDepthForUnalign(uint64_t leftL1Size, uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleABSize);
    ge::graphStatus CalScaleFactors();
    bool CanEnableThreeL1Buffer() const;
    GroupedMatmulTilingData::GMMQuantBasicApiTilingData tilingData_;
};
} // namespace optiling

#endif // GROUPED_QUANT_BASIC_API_MATMUL_TILING_H
