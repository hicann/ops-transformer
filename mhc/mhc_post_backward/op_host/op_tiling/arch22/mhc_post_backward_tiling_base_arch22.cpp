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
 * \file mhc_post_backward_tiling.cpp
 * \brief
 */

#include "../mhc_post_backward_tiling.h"
#include <vector>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../op_kernel/arch22/mhc_post_backward_tiling_data_arch22.h"
#include "../../../op_kernel/arch22/mhc_post_backward_tiling_key_arch22.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace {
constexpr uint8_t GRAD_Y_IDX = 0;
constexpr uint8_t X_IDX = 1;
constexpr uint8_t H_RES_IDX = 2;
constexpr uint8_t H_OUT_IDX = 3;
constexpr uint8_t H_POST_IDX = 4;

constexpr uint8_t DIM_INDEX_0 = 0;
constexpr uint8_t DIM_INDEX_1 = 1;
constexpr uint8_t DIM_INDEX_2 = 2;
constexpr uint8_t DIM_INDEX_3 = 3;

constexpr uint8_t X_MIX_GRAD_IDX = 0;
constexpr uint8_t H_MIX_GRAD_IDX = 1;

constexpr uint8_t SIZE_BFLOAT16 = 2;
constexpr uint8_t SIZE_FLOAT = 4;

static int32_t GetCeilInt(int32_t value1, int32_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return static_cast<int32_t>((value1 + value2 - 1) / value2);
}

} // namespace

namespace optiling {
const uint32_t BLOCK_C = 1024;

class MhcPostBackwardTilingBaseArch22 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPostBackwardTilingBaseArch22(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {}
    ~MhcPostBackwardTilingBaseArch22() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(0);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspaces[0] = workspaceSize_;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        context_->SetBlockDim(blockDim_);
        return ge::GRAPH_SUCCESS;
    }

private:
    uint64_t blockDim_ = 0;
    size_t workspaceSize_ = 0;
};

ge::graphStatus MhcPostBackwardTilingBaseArch22::DoOpTiling()
{
    if (context_ == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoptr = context_->GetPlatformInfo();
    if (platformInfoptr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ascendplatformInfo = platform_ascendc::PlatformAscendC(platformInfoptr);
    const auto coreNumber = ascendplatformInfo.GetCoreNumAiv();

    auto gradYTensor = context_->GetInputTensor(GRAD_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYTensor);
    auto xTensor = context_->GetInputTensor(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    auto hResTensor = context_->GetInputTensor(H_RES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hResTensor);
    auto hOutTensor = context_->GetInputTensor(H_OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hOutTensor);
    auto hPostTensor = context_->GetInputTensor(H_POST_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hPostTensor);

    auto gradYDesc = context_->GetInputDesc(GRAD_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    auto gradYDtype = gradYDesc->GetDataType();
    OP_CHECK_IF(gradYDtype != ge::DataType::DT_BF16 && gradYDtype != ge::DataType::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(), "grad_y dtype only supports bf16,half."), return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(xDesc->GetDataType() != gradYDtype,
                OP_LOGE(context_->GetNodeName(), "the dtype of x should be same with grad_y."),
                return ge::GRAPH_FAILED);

    auto hOutDesc = context_->GetInputDesc(H_OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hOutDesc);
    OP_CHECK_IF(hOutDesc->GetDataType() != gradYDtype,
                OP_LOGE(context_->GetNodeName(), "the dtype of h_out should be same with grad_y."),
                return ge::GRAPH_FAILED);

    auto hResDesc = context_->GetInputDesc(H_RES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hResDesc);
    OP_CHECK_IF(hResDesc->GetDataType() != ge::DataType::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "h_res dtype only supports float32."), return ge::GRAPH_FAILED);

    auto hPostDesc = context_->GetInputDesc(H_POST_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, hPostDesc);
    OP_CHECK_IF(hPostDesc->GetDataType() != ge::DataType::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "h_post dtype only supports float32."), return ge::GRAPH_FAILED);

    const auto dFPostResShape = gradYTensor->GetStorageShape();
    const uint32_t dimNum = dFPostResShape.GetDimNum();

    uint32_t totalTasks = 0;
    uint32_t n = 0;
    uint32_t channel = 0;

    if (dimNum == 3) {
        totalTasks = dFPostResShape.GetDim(DIM_INDEX_0);
        n = dFPostResShape.GetDim(DIM_INDEX_1);
        channel = dFPostResShape.GetDim(DIM_INDEX_2);
    } else if (dimNum == 4) {
        totalTasks = dFPostResShape.GetDim(DIM_INDEX_0) * dFPostResShape.GetDim(DIM_INDEX_1);
        n = dFPostResShape.GetDim(DIM_INDEX_2);
        channel = dFPostResShape.GetDim(DIM_INDEX_3);
    } else {
        OP_LOGE(context_->GetNodeName(), "grad_y dim num should be 3 (TND) or 4 (BSND), but got %u.", dimNum);
        return ge::GRAPH_FAILED;
    }

    uint64_t frontCore = totalTasks % coreNumber != 0 ? static_cast<uint64_t>(totalTasks % coreNumber) : coreNumber;
    uint64_t tailCore = totalTasks <= coreNumber ? 0 : coreNumber - frontCore;

    int32_t singleCoreBS = GetCeilInt(totalTasks, coreNumber);
    int32_t tailBS = totalTasks / coreNumber;

    blockDim_ = frontCore + tailCore;
    uint64_t ubSizePlatForm;
    ascendplatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    uint32_t dFPostResSize = gradYTensor->GetShapeSize();
    uint32_t xSize = xTensor->GetShapeSize();
    uint32_t hResSize = hResTensor->GetShapeSize();
    uint32_t hOutSize = hOutTensor->GetShapeSize();
    uint32_t hPostSize = hPostTensor->GetShapeSize();

    const uint32_t alignN = GetCeilInt(n * SIZE_FLOAT, 32) * 32 / SIZE_FLOAT;
    const uint32_t blockChannel = BLOCK_C > channel ? channel : BLOCK_C;
    const uint32_t loopC = channel / blockChannel;
    const uint32_t tailC = channel % blockChannel;

    MhcPostBackwardTilingDataArch22* tiling = context_->GetTilingData<MhcPostBackwardTilingDataArch22>();
    tiling->singleCoreBS = singleCoreBS;
    tiling->tailBS = tailBS;
    tiling->coreUsed = blockDim_;
    tiling->frontCore = frontCore;
    tiling->tailCore = tailCore;

    tiling->dFPostResSize = dFPostResSize;
    tiling->xSize = xSize;
    tiling->hResSize = hResSize;
    tiling->hOutSize = hOutSize;
    tiling->hPostSize = hPostSize;

    tiling->channel = channel;
    tiling->blockChannel = blockChannel;
    tiling->n = n;
    tiling->alignN = alignN;
    tiling->tailC = tailC;
    tiling->loopC = loopC;

    workspaceSize_ = ascendplatformInfo.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(
    MhcPostBackward, MhcPostBackwardTilingBaseArch22,
    std::vector<int32_t>({static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B),
                          static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910_93)}),
    1);
} // namespace optiling
