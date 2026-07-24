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
 * \file quant_grouped_mat_mul_allto_allv_tiling_adapter.cpp
 * \brief
 */

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_adapter.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;
using namespace optiling::Mc2GroupedMatmulTiling;
using namespace optiling::Mc2GroupedMatmulTiling::GmmConstant;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace Mc2Tiling {

const std::vector<uint32_t> QUANT_MODE_MP = {0, 0, 1, 2, 4, 5, 3}; // 不量化 pertensor perchannel pertoken pergroup perblock mx; 分别对应gmm中各量化的移位数

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetCommonInputParams(const QuantGmmAlltoAllvParamsInfo& params)
{
    GetPlatformInfo();
    inputParams_.opName = params.opName;
    inputParams_.kernelType = 0UL;
    // 输出是否切分，0/1代表输出多tensor， 2/3代表输出单tensor
    inputParams_.splitItem = 2;
    inputParams_.actType = 0;
    // common
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = false;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetGroupExpertInputParameters(
    const QuantGmmAlltoAllvParamsInfo& params, uint64_t gmmX, uint64_t expertNum)
{
    inputParams_.mSize = gmmX;
    inputParams_.kSize = params.H1;
    inputParams_.nSize = params.N1;
    inputParams_.groupNum = expertNum;
    if (params.gmmXQuantMode > 0 && params.gmmXQuantMode < QUANT_MODE_MP.size()) {
        inputParams_.aQuantMode = static_cast<QuantMode>(1U << (QUANT_MODE_MP[params.gmmXQuantMode]));
        inputParams_.bQuantMode = static_cast<QuantMode>(1U << (QUANT_MODE_MP[params.gmmWeightQuantMode]));
    }
    // 是否做切分
    inputParams_.groupType = optiling::Mc2GroupedMatmul::SPLIT_M;
    inputParams_.groupListType = 1;
    inputParams_.aDtype = params.gmmXDtype;
    inputParams_.bDtype = params.gmmWeightDtype;
    inputParams_.cDtype = params.gmmYDtype;
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = params.gmmXScaleDtype;
    inputParams_.perTokenScaleDtype = params.gmmXScaleDtype;

    inputParams_.transB = params.isGmmWeightTrans;
    OP_TILING_CHECK(expertNum > optiling::Mc2GroupedMatmul::MAX_TENSOR_CONT,
        OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "expertNum",
            std::to_string(expertNum).c_str(),
            (std::string("<=") + std::to_string(optiling::Mc2GroupedMatmul::MAX_TENSOR_CONT)).c_str()),
        return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < expertNum; i++) {
        mList_[i] = static_cast<int32_t>(gmmX);
    }
    kList_[0] = static_cast<int32_t>(params.H1);
    nList_[0] = static_cast<int32_t>(params.N1);
    SetKernelType();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::SetSharedExpertInputParameters(const QuantGmmAlltoAllvParamsInfo& params)
{
    inputParams_.mSize = params.Bs;
    inputParams_.kSize = params.H2;
    inputParams_.nSize = params.N2;
    inputParams_.groupNum = 1;
    if (params.gmmXQuantMode > 0 && QUANT_MODE_MP.size()) {
        inputParams_.aQuantMode = static_cast<QuantMode>(1U << (QUANT_MODE_MP[params.mmXQuantMode]));
        inputParams_.bQuantMode = static_cast<QuantMode>(1U << (QUANT_MODE_MP[params.mmWeightQuantMode]));
    }
    // 是否做切分
    inputParams_.groupType = optiling::Mc2GroupedMatmul::SPLIT_M;
    // 非负递增为0，非负数列为1
    inputParams_.groupListType = 1;
    inputParams_.aDtype = params.mmXDtype;
    inputParams_.bDtype = params.mmWeightDtype;
    inputParams_.cDtype = params.mmYDtype;
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = params.mmXScaleDtype;
    inputParams_.perTokenScaleDtype = params.mmXScaleDtype;

    inputParams_.transB = params.isMmWeightTrans;
    mList_[0] = static_cast<int32_t>(params.Bs);
    kList_[0] = static_cast<int32_t>(params.H2);
    nList_[0] = static_cast<int32_t>(params.N2);
    SetKernelType();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvAdapter::Process()
{
    MC2_CHECK_LOG_RET(inputParams_.opName, DoOpTiling());
    MC2_CHECK_LOG_RET(inputParams_.opName, DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

}