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
 * \file grouped_weight_quant_matmul_asw_tiling.h
 * \brief
 */

#pragma once

#include <sstream>

#include "../../grouped_matmul_host_util.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "grouped_weight_quant_matmul_tiling_tool.h"
#include "grouped_weight_quant_matmul_tiling_checker.h"
#include "op_host/tiling_base.h"

namespace optiling {
class AswTilingKeyConfigure {
public:
    // 1 - 算法大类
    uint8_t algorithmCategory = 0;

    // 2 - 算法小类
    uint8_t algorithmSubCategory = 0;

    // 3 - 模板自定义场景
    uint8_t templateCustom = 0;

    // 4 - 输入A转置情况
    uint8_t transA = 0;

    // 5 - 输入B转置情况
    uint8_t transB = 0;

    // 6 - weight的量化类型 None/per_tensor/per_channel/per_group/mx
    uint8_t antiquantType = 0;

    // 7 - C矩阵的量化类型 None/per_tensor/per_channel/per_group/mx
    uint8_t quantType = 0;

    // 8 - offset\bias的可选输入是否存在
    uint8_t optionInputSituation = 0;

    // 9 - weight的数据格式 ND/NZ
    uint8_t weightFormat = 0;

public:
    void PrintTilingKeyLog() const
    {
        std::stringstream ss;
        ss << "algorithmCategory: " << static_cast<uint32_t>(this->algorithmCategory)
           << ", algorithmSubCategory: " << static_cast<uint32_t>(this->algorithmSubCategory)
           << ", templateCustom: " << static_cast<uint32_t>(this->templateCustom)
           << ", transA: " << static_cast<uint32_t>(this->transA) << ", transB: " << static_cast<uint32_t>(this->transB)
           << ", antiquantType: " << static_cast<uint32_t>(this->antiquantType)
           << ", quantType: " << static_cast<uint32_t>(this->quantType)
           << ", optionInputSituation: " << static_cast<uint32_t>(this->optionInputSituation)
           << ", weightFormat: " << static_cast<uint32_t>(this->weightFormat);
        OP_LOGI("GroupedMatmul", "tilingKeyConfigure: {%s}", ss.str().c_str());
        return;
    }

    uint64_t GenTilingKey() const;
};

class GroupedWeightQuantMatmulAswTiling {
public:
    bool SetTiling(gert::TilingContext* context);

private:
    bool GetPlatformInfo();
    bool AnalyzeInputs();
    bool SetMKN();
    void SetGroupSize();
    void DoLibApiTiling();
    void DoOpTiling();
    void CalBasicBlock();
    bool GetWorkspaceSize();
    bool PostTiling();
    bool SetTilingKey();
    void PrintTilingData();

    GMMWeightQuantTilingTool::GMMWeightQuantInputInfo inputParams_;
    GMMWeightQuantTilingTool::GMMWeightQuantInputNum inputNum_;
    GMMWeightQuantTilingTool::GMMWeightQuantBasicTiling basicTiling_;
    AswTilingKeyConfigure tilingKeyConfig_;
    GroupedMatmulTilingData::GMMWeightQuantAswTilingData tilingData_;

    gert::TilingContext* context_ = nullptr;
    Ops::Transformer::OpTiling::AiCoreParams aicoreParams_;

    bool isWeightNz_ = false;
};
} // namespace optiling
