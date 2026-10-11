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
 * \file moe_gating_top_k_softmax_tiling_310p.cpp
 * \brief
 */

#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "moe_gating_top_k_softmax_tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "moe_gating_top_k_softmax_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

#include <algorithm>
#include <climits>

using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace optiling {
static const int32_t FP16_SIZE = 2;
static const bool IS_SOFTMAX_REUSE_SOURCE = true;
static const int32_t MAX_COL_IN_UB = 98304; // ubSize/minTypeSize
const int32_t UB_SIZE = 245760;

static inline uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    if (factor == 0U) {
        return value;
    }
    return (value + factor - 1U) / factor;
}

static inline int64_t Align(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y * y;
}

static inline int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

class MoeGatingTopKSoftmax310PTiling : public MoeGatingTopKSoftmaxBaseTiling {
public:
    explicit MoeGatingTopKSoftmax310PTiling(gert::TilingContext* context)
        : MoeGatingTopKSoftmaxBaseTiling(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint32_t maxRow;
    uint32_t gatingAlignCol;
    bool doubleBufferFlag;
    MoeGatingTopKSoftmax310PTilingData tilingData;
};

bool MoeGatingTopKSoftmax310PTiling::IsCapable()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto is310P = (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P);
    return is310P;
}

ge::graphStatus MoeGatingTopKSoftmax310PTiling::DoOpTiling()
{
    const uint32_t perBlockEleNum = 32 / 2;
    bool flagAlignExperts = (col % perBlockEleNum == 0);
    if (!flagAlignExperts) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "expert count", std::to_string(col).c_str(),
                                              "Expert count must be 32-byte aligned.");
        return false;
    }
    const size_t x_dim_num = 2;
    const size_t expertIdx_dim_num = 2;
    auto x_shape = ge::Shape({row, col});
    auto y_shape = ge::Shape({row, k});
    auto expertIdx_shape = ge::Shape({row, k});
    // Set row
    int64_t rowTmp = 1;
    for (size_t i = 0; i < x_dim_num - 1; ++i) {
        rowTmp *= x_shape.GetDim(i);
    }
    tilingData.set_row(static_cast<int32_t>(rowTmp));

    // Set col
    const int64_t colTmp = x_shape.GetDim(x_dim_num - 1);
    tilingData.set_col(static_cast<int32_t>(colTmp));

    // Set k
    const int64_t kTmp = expertIdx_shape.GetDim(expertIdx_dim_num - 1);
    tilingData.set_k(static_cast<int32_t>(kTmp));

    const int64_t kAlign = Align(kTmp, 16);
    tilingData.set_kAlign(static_cast<int32_t>(kAlign));

    // Set blockNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto blockNum = ascendcPlatform.GetCoreNumAic();
    tilingData.set_blockNum(blockNum);
    int32_t hasFinishedTmp = 0;
    tilingData.set_hasFinished(hasFinishedTmp);

    // Set FormerRow
    const int64_t oneCoreRow = Align(std::min(rowTmp, UB_SIZE / (32 * (colTmp + 3 * kAlign))), 16 * blockNum);
    const int64_t activateCore = CeilDiv(rowTmp, oneCoreRow);
    const int64_t tailRow = rowTmp - (activateCore - 1) * oneCoreRow;

    tilingData.set_oneCoreRow(oneCoreRow);
    tilingData.set_activateCore(activateCore);
    tilingData.set_tailRow(tailRow);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmax310PTiling::DoLibApiTiling()
{
    int32_t dataTypeSize = FP16_SIZE;
    auto FormersoftmaxShape = ge::Shape({tilingData.get_oneCoreRow(), tilingData.get_col()});
    SoftMaxTilingFunc(FormersoftmaxShape, dataTypeSize,
                      GetSoftMaxMaxTmpSize(FormersoftmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
                      tilingData.FormerSoftmaxTilingData);

    auto TailsoftmaxShape = ge::Shape({tilingData.get_tailRow(), tilingData.get_col()});
    SoftMaxTilingFunc(TailsoftmaxShape, dataTypeSize,
                      GetSoftMaxMaxTmpSize(TailsoftmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
                      tilingData.TailSoftmaxTilingData);
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmax310PTiling::GetTilingKey() const
{
    switch (dtype) {
        case ge::DataType::DT_FLOAT16:
            return MOE_GATING_SOFTMAX_310P_FLOAT16_OPTIONAL_FINISHED;
        default:
            break;
    }
    return tilingKey_;
}

ge::graphStatus MoeGatingTopKSoftmax310PTiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmax310PTiling::PostTiling()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    tilingData.set_tilingKey(GetTilingKey());
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData.get_blockNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    int32_t dataTypeSize = FP16_SIZE;
    auto FormertopkShape = ge::Shape({tilingData.get_oneCoreRow(), tilingData.get_col()});
    bool FormerTopkTilingSuccess =
        TopKTilingFunc(ascendcPlatform,
                       tilingData.get_col(),        // inner
                       tilingData.get_oneCoreRow(), // outter
                       tilingData.get_kAlign(),     // k
                       dataTypeSize, true, TopKMode::TOPK_NORMAL, true, tilingData.FormerTopkTilingData);
    auto TailtopkShape = ge::Shape({tilingData.get_tailRow(), tilingData.get_col()});
    bool TailTopkTilingSuccess =
        TopKTilingFunc(ascendcPlatform, tilingData.get_col(), tilingData.get_tailRow(), tilingData.get_kAlign(),
                       dataTypeSize, true, TopKMode::TOPK_NORMAL, true, tilingData.TailTopkTilingData);
    if (!(FormerTopkTilingSuccess && TailTopkTilingSuccess))
        return ge::GRAPH_FAILED;
    uint32_t maxsize = 0;
    uint32_t minsize = 0;
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, tilingData.get_col(), tilingData.get_oneCoreRow(), false, true,
                                  AscendC::TopKMode::TOPK_NORMAL, true, dataTypeSize, maxsize, minsize);
    tilingData.set_FormerTmpMinsize(minsize);
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, tilingData.get_col(), tilingData.get_tailRow(), false, true,
                                  AscendC::TopKMode::TOPK_NORMAL, true, dataTypeSize, maxsize, minsize);
    tilingData.set_TailTmpMinsize(minsize);
    const int64_t kAlign = tilingData.get_kAlign();
    const int64_t activateCore = tilingData.get_activateCore();
    size_t userWorkspaceSize = 1024 + kAlign * row * activateCore * (sizeof(int16_t) + sizeof(int32_t));
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    tilingData.set_workspaceSize(currentWorkspace[0]);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKSoftmax, MoeGatingTopKSoftmax310PTiling, 200);
} // namespace optiling
