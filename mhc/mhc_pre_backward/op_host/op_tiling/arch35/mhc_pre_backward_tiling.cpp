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
 * \file mhc_pre_backward_tiling.cpp
 * \brief
 */
#include "mhc_pre_backward_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;
const constexpr int64_t BSD_DIM_NUM = 3;
const constexpr int64_t BSNN_DIM_NUM = 4;
const constexpr int64_t TD_DIM_NUM = 2;
const constexpr int64_t TN_DIM_NUM = 2;
const constexpr int64_t TNN_DIM_NUM = 3;
const constexpr uint32_t GRAD_H_IN_INDEX = 3;
const constexpr uint32_t GRAD_H_POST_INDEX = 4;
const constexpr uint32_t GRAD_H_RES_INDEX = 5;

const constexpr int64_t INDEX_B = 0;
const constexpr int64_t INDEX_S = 1;
const constexpr int64_t INDEX_D = 2;

const constexpr int64_t INDEX_T = 0;
const constexpr int64_t INDEX_N = 1;
const constexpr int64_t INDEX_D_TND = 1;

const constexpr int32_t C0_BASE_M = 256;
const constexpr int32_t C0_BASE_N = 128;
const constexpr int32_t C0_BASE_K = 32;
const constexpr uint32_t MIN_D_LENGTH = 1;
const constexpr uint32_t MAX_D_LENGTH = 16384;
const constexpr uint32_t D_ALIGN = 64;

const constexpr uint32_t DEFAULT_TILING_PRIORITY = 1000;
const constexpr uint32_t LEGAL_N_VALUES[] = {4, 6, 8};
const constexpr uint32_t LEGAL_N_COUNT = 3;
const constexpr float DEFAULT_HC_EPS = 1e-6f;

const constexpr uint32_t C0_SET_VALUE = 1;
const constexpr uint32_t ALPHA_GRAD_CORE_FACTOR = 24;
const constexpr uint32_t BUFFER_NUM = 2;
const constexpr uint32_t EXTRA_BUFFER_SIZE = 2 * 1024 * 1024;
const constexpr uint32_t WORKSPACE_ALIGN_SIZE = 32;
const constexpr uint64_t SYSTEM_WORKSPACE_SIZE = 40 * 1024 * 1024;
const constexpr int32_t SCHEDULE_MODE = 1;

const constexpr uint32_t C0_SET_SHAPE_M = 1024;
const constexpr uint32_t C0_SET_SHAPE_N = 128;
const constexpr uint32_t C1_SET_SHAPE_M_BASE = 128;
const constexpr uint32_t C1_SET_SHAPE_K = 1024;

REGISTER_OPS_TILING_TEMPLATE(MhcPreBackward, MhcPreBackwardTiling, DEFAULT_TILING_PRIORITY);

ge::graphStatus MhcPreBackwardTiling::GetInputTensors(const gert::Tensor *&gradHInTensor,
                                                      const gert::Tensor *&gradHPostTensor,
                                                      const gert::Tensor *&gradHResTensor)
{
    gradHInTensor = context_->GetDynamicInputTensor(GRAD_H_IN_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradHInTensor);
    gradHPostTensor = context_->GetDynamicInputTensor(GRAD_H_POST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradHPostTensor);
    gradHResTensor = context_->GetDynamicInputTensor(GRAD_H_RES_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradHResTensor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ValidateInputDims(int64_t gradHInDims, int64_t gradHPostDims,
                                                        int64_t gradHResDims)
{
    if ((gradHInDims != BSD_DIM_NUM && gradHInDims != TD_DIM_NUM) ||
        (gradHPostDims != BSD_DIM_NUM && gradHPostDims != TN_DIM_NUM) ||
        (gradHResDims != BSNN_DIM_NUM && gradHResDims != TNN_DIM_NUM)) {
        OP_LOGE(context_->GetNodeName(),
                "input dims invalid for MhcPreBackward, gradHInDims=%ld (expected %ld or %ld), gradHPostDims=%ld "
                "(expected %ld or %ld), gradHResDims=%ld (expected %ld or %ld)",
                gradHInDims, BSD_DIM_NUM, TD_DIM_NUM, gradHPostDims, BSD_DIM_NUM, TN_DIM_NUM, gradHResDims,
                BSNN_DIM_NUM, TNN_DIM_NUM);
        return ge::GRAPH_FAILED;
    }
    if (gradHInDims != gradHPostDims) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_in and grad_h_post must have the same dim num, gradHInDims=%ld, gradHPostDims=%ld", gradHInDims,
                gradHPostDims);
        return ge::GRAPH_FAILED;
    }
    if ((gradHInDims == BSD_DIM_NUM && gradHResDims != BSNN_DIM_NUM) ||
        (gradHInDims == TD_DIM_NUM && gradHResDims != TNN_DIM_NUM)) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_in and grad_h_res dims mismatch, gradHInDims=%ld requires gradHResDims=%ld, but got %ld",
                gradHInDims, gradHInDims == BSD_DIM_NUM ? BSNN_DIM_NUM : TNN_DIM_NUM, gradHResDims);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ParseBSDFormat(const gert::Tensor *gradHInTensor,
                                                     const gert::Tensor *gradHPostTensor,
                                                     const gert::Tensor *gradHResTensor)
{
    uint64_t batch = gradHInTensor->GetStorageShape().GetDim(INDEX_B);
    uint64_t sequence = gradHInTensor->GetStorageShape().GetDim(INDEX_S);
    D_ = gradHInTensor->GetStorageShape().GetDim(INDEX_D);
    N_ = gradHPostTensor->GetStorageShape().GetDim(INDEX_D);

    if (gradHPostTensor->GetStorageShape().GetDim(INDEX_B) != batch ||
        gradHPostTensor->GetStorageShape().GetDim(INDEX_S) != sequence) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_post shape must align with grad_h_in on B and S dims, grad_h_post[B,S]=[%ld,%ld], expected "
                "[%lu,%lu]",
                gradHPostTensor->GetStorageShape().GetDim(INDEX_B), gradHPostTensor->GetStorageShape().GetDim(INDEX_S),
                batch, sequence);
        return ge::GRAPH_FAILED;
    }
    if (gradHResTensor->GetStorageShape().GetDim(0) != batch ||
        gradHResTensor->GetStorageShape().GetDim(1) != sequence || gradHResTensor->GetStorageShape().GetDim(2) != N_ ||
        gradHResTensor->GetStorageShape().GetDim(3) != N_) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_res shape must be [B, S, N, N], actual=[%ld,%ld,%ld,%ld], expected=[%lu,%lu,%lu,%lu]",
                gradHResTensor->GetStorageShape().GetDim(0), gradHResTensor->GetStorageShape().GetDim(1),
                gradHResTensor->GetStorageShape().GetDim(2), gradHResTensor->GetStorageShape().GetDim(3), batch,
                sequence, N_, N_);
        return ge::GRAPH_FAILED;
    }
    totalLength_ = batch * sequence;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ParseTNDFormat(const gert::Tensor *gradHInTensor,
                                                     const gert::Tensor *gradHPostTensor,
                                                     const gert::Tensor *gradHResTensor)
{
    uint64_t t = gradHInTensor->GetStorageShape().GetDim(INDEX_T);
    D_ = gradHInTensor->GetStorageShape().GetDim(INDEX_D_TND);
    N_ = gradHPostTensor->GetStorageShape().GetDim(INDEX_N);

    if (gradHPostTensor->GetStorageShape().GetDim(INDEX_T) != t) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_post shape must align with grad_h_in on T dim, grad_h_post[T]=%ld, expected %lu",
                gradHPostTensor->GetStorageShape().GetDim(INDEX_T), t);
        return ge::GRAPH_FAILED;
    }
    if (gradHResTensor->GetStorageShape().GetDim(0) != t || gradHResTensor->GetStorageShape().GetDim(1) != N_ ||
        gradHResTensor->GetStorageShape().GetDim(2) != N_) {
        OP_LOGE(context_->GetNodeName(),
                "grad_h_res shape must be [T, N, N], actual=[%ld,%ld,%ld], expected=[%lu,%lu,%lu]",
                gradHResTensor->GetStorageShape().GetDim(0), gradHResTensor->GetStorageShape().GetDim(1),
                gradHResTensor->GetStorageShape().GetDim(2), t, N_, N_);
        return ge::GRAPH_FAILED;
    }
    totalLength_ = t;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ValidateShapeParams()
{
    static const std::vector<uint64_t> legalN(LEGAL_N_VALUES, LEGAL_N_VALUES + LEGAL_N_COUNT);
    if (std::find(legalN.begin(), legalN.end(), N_) == legalN.end()) {
        OP_LOGE(context_->GetNodeName(), "Invalid input shape N=%lu. Expected one of {4,6,8}", N_);
        return ge::GRAPH_FAILED;
    }
    if (D_ < MIN_D_LENGTH || D_ > MAX_D_LENGTH || D_ % D_ALIGN != 0) {
        OP_LOGE(context_->GetNodeName(),
                "Invalid input shape D=%lu. Expected to be in [%u, %u] and %u-element alignment", D_, MIN_D_LENGTH,
                MAX_D_LENGTH, D_ALIGN);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::GetInputShape()
{
    const gert::Tensor *gradHInTensor = nullptr;
    const gert::Tensor *gradHPostTensor = nullptr;
    const gert::Tensor *gradHResTensor = nullptr;

    auto ret = GetInputTensors(gradHInTensor, gradHPostTensor, gradHResTensor);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto gradHInDims = gradHInTensor->GetStorageShape().GetDimNum();
    auto gradHPostDims = gradHPostTensor->GetStorageShape().GetDimNum();
    auto gradHResDims = gradHResTensor->GetStorageShape().GetDimNum();

    ret = ValidateInputDims(gradHInDims, gradHPostDims, gradHResDims);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (gradHInDims == BSD_DIM_NUM) {
        ret = ParseBSDFormat(gradHInTensor, gradHPostTensor, gradHResTensor);
    } else if (gradHInDims == TD_DIM_NUM) {
        ret = ParseTNDFormat(gradHInTensor, gradHPostTensor, gradHResTensor);
    }
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ValidateShapeParams();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    fusionSize_ = (2 * N_) + (N_ * N_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::ParseInputAndAttr()
{
    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "get input shape failed");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    blockDim_ = ascendcPlatform.GetCoreNumAic();
    vecCoreNum_ = ascendcPlatform.GetCoreNumAiv();

    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get attrs failed");
        return ge::GRAPH_FAILED;
    }
    auto hcEpsPtr = attrs->GetAttrPointer<float>(0);
    hcEps_ = (hcEpsPtr != nullptr) ? *hcEpsPtr : DEFAULT_HC_EPS;

    return ge::GRAPH_SUCCESS;
}

void MhcPreBackwardTiling::SetC0TilingParams()
{
    tilingData_.matmulTilingC0.set_dbL0C(C0_SET_VALUE);
    tilingData_.matmulTilingC0.set_stepKa(C0_SET_VALUE);
    tilingData_.matmulTilingC0.set_stepKb(C0_SET_VALUE);
    tilingData_.matmulTilingC0.set_depthA1(C0_SET_VALUE);
    tilingData_.matmulTilingC0.set_depthB1(C0_SET_VALUE);
    tilingData_.matmulTilingC0.set_stepM(C0_SET_VALUE);
    tilingData_.matmulTilingC0.set_stepN(C0_SET_VALUE);
    tilingData_.matmulTilingC0.set_baseM(C0_BASE_M);
    tilingData_.matmulTilingC0.set_baseN(C0_BASE_N);
    tilingData_.matmulTilingC0.set_baseK(C0_BASE_K);
}

void MhcPreBackwardTiling::SetC1TilingParams()
{
    tilingData_.matmulTilingC1.set_dbL0C(C0_SET_VALUE);
    tilingData_.matmulTilingC1.set_stepKa(C0_SET_VALUE);
    tilingData_.matmulTilingC1.set_stepKb(C0_SET_VALUE);
    tilingData_.matmulTilingC1.set_depthA1(C0_SET_VALUE);
    tilingData_.matmulTilingC1.set_depthB1(C0_SET_VALUE);
    tilingData_.matmulTilingC1.set_stepM(C0_SET_VALUE);
    tilingData_.matmulTilingC1.set_stepN(C0_SET_VALUE);
}

void MhcPreBackwardTiling::SetCommonTilingParams()
{
    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_vecCoreNum(vecCoreNum_);
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_nD(N_ * D_);
    tilingData_.set_fusionSize(fusionSize_);
    tilingData_.set_N(N_);
    tilingData_.set_D(D_);
    tilingData_.set_hcEps(hcEps_);
}

void MhcPreBackwardTiling::FillTilingData()
{
    SetC0TilingParams();
    SetC1TilingParams();
    SetCommonTilingParams();
}

uint64_t MhcPreBackwardTiling::CalculateWorkspaceSize(uint64_t totalLength, uint64_t fusionSize, uint64_t cubeCoreNum,
                                                      uint64_t vecCoreNum, uint64_t elementSize)
{
    uint64_t v1Elements = totalLength * fusionSize +            // h_mix_grad
                          ALPHA_GRAD_CORE_FACTOR * vecCoreNum + // alpha_grad
                          vecCoreNum * fusionSize +             // bias_grad
                          totalLength;                          // inv_rms_grad

    uint64_t v2Elements = C0_SET_SHAPE_M * C0_SET_SHAPE_N * cubeCoreNum * BUFFER_NUM + // x_rs_grad_mm
                          C0_SET_SHAPE_M * C0_SET_SHAPE_N * cubeCoreNum * BUFFER_NUM + // x_rs
                          EXTRA_BUFFER_SIZE;
    uint64_t totalElements =
        ((v1Elements + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE * WORKSPACE_ALIGN_SIZE) + v2Elements;
    return totalElements * elementSize;
}

void MhcPreBackwardTiling::SetMatmulC0Tiling(matmul_tiling::MatmulApiTiling &mm)
{
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetShape(C0_SET_SHAPE_M, C0_SET_SHAPE_N, N_ * N_ + 2 * N_);
    mm.SetOrgShape(totalLength_, N_ * D_, N_ * N_ + 2 * N_);
    mm.SetBufferSpace(-1, -1, -1);
}

void MhcPreBackwardTiling::SetMatmulC1Tiling(matmul_tiling::MatmulApiTiling &mm)
{
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetShape(N_ * N_ + 2 * N_, C1_SET_SHAPE_M_BASE, C1_SET_SHAPE_K);
    mm.SetOrgShape(N_ * N_ + 2 * N_, N_ * D_, totalLength_);
    mm.SetBufferSpace(-1, -1, -1);
}

ge::graphStatus MhcPreBackwardTiling::GetMatmulTiling(matmul_tiling::MatmulApiTiling &mm, bool isC0)
{
    if (isC0) {
        SetMatmulC0Tiling(mm);
    } else {
        SetMatmulC1Tiling(mm);
    }
    if (mm.GetTiling(isC0 ? tilingData_.matmulTilingC0 : tilingData_.matmulTilingC1) == -1) {
        OP_LOGE(context_->GetNodeName(), "ProcessC%d Get Tiling Failed!, m = %lu, n = %lu, k = %lu", isC0 ? 0 : 1,
                isC0 ? totalLength_ : N_ * N_ + 2 * N_, N_ * D_, isC0 ? N_ * N_ + 2 * N_ : totalLength_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::TilingProcess()
{
    size_t userWorkspaceSize = CalculateWorkspaceSize(totalLength_, fusionSize_, blockDim_, vecCoreNum_, sizeof(float));
    size_t systemWorkspaceSize = SYSTEM_WORKSPACE_SIZE;

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);

    matmul_tiling::MatmulApiTiling mm0_(ascendcPlatform);
    matmul_tiling::MatmulApiTiling mm1_(ascendcPlatform);

    auto ret = GetMatmulTiling(mm0_, true);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetMatmulTiling(mm1_, false);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBackwardTiling::DoOpTiling()
{
    auto inputXDesc = context_->GetInputDesc(0);
    if (inputXDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer: x");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void MhcPreBackwardTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "blockDim: [%u]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "totalLength: [%lu]", tilingData_.get_totalLength());
    OP_LOGD(context_->GetNodeName(), "nD: [%lu]", tilingData_.get_nD());
    OP_LOGD(context_->GetNodeName(), "fusionSize: [%lu]", tilingData_.get_fusionSize());
    OP_LOGD(context_->GetNodeName(), "hcEps: [%f]", tilingData_.get_hcEps());
}

uint64_t MhcPreBackwardTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(MHC_PRE_BACKWARD_DEFAULT));
}

ge::graphStatus MhcPreBackwardTiling::PostTiling()
{
    OP_CHECK_IF(
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(SCHEDULE_MODE);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);

    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
