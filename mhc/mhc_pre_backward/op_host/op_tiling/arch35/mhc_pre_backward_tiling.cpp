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
#include "../../../op_kernel/arch35/mhc_pre_backward_tiling_key.h"
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

const constexpr uint32_t MIN_D_LENGTH = 1;
const constexpr uint32_t MAX_D_LENGTH = 16384;
const constexpr uint32_t D_ALIGN = 64;

const constexpr uint32_t DEFAULT_TILING_PRIORITY = 1000;
const constexpr uint32_t LEGAL_N_VALUES[] = {4, 6, 8};
const constexpr uint32_t LEGAL_N_COUNT = 3;
const constexpr float DEFAULT_HC_EPS = 1e-6f;
const constexpr int64_t IMPL_MODE_FP32 = 0;
const constexpr int64_t IMPL_MODE_HF32 = 1;
const constexpr uint32_t IMPL_MODE_ATTR_INDEX = 1;

const constexpr uint32_t ALPHA_GRAD_CORE_FACTOR = 24;
const constexpr uint32_t BUFFER_NUM = 2;
const constexpr uint32_t EXTRA_BUFFER_SIZE = 2 * 1024 * 1024;
const constexpr uint32_t WORKSPACE_ALIGN_SIZE = 32;
const constexpr uint64_t SYSTEM_WORKSPACE_SIZE = 40 * 1024 * 1024;
const constexpr int32_t SCHEDULE_MODE = 1;

const constexpr uint32_t C0_SET_SHAPE_M = 512U;
const constexpr uint32_t C0_SET_SHAPE_N = 128U;
const constexpr uint32_t C1_SET_SHAPE_K = 512U;
const constexpr uint32_t C0_TO_V2_ROWS = 128U;
const constexpr uint32_t V2_TO_C1_ROWS = 512U;

const constexpr uint32_t C0_BASE_M = 128U;
const constexpr uint32_t C0_BASE_N = 128U;
const constexpr uint32_t C0_BASE_K = 32U;
const constexpr uint32_t C0_L1_K = 64U;
const constexpr uint32_t C0_DEPTH_A1 = 2U;
const constexpr uint32_t C0_DEPTH_B1 = 2U;
const constexpr uint32_t C0_DB_L0A = 2U;
const constexpr uint32_t C0_DB_L0B = 2U;
const constexpr uint32_t C0_DB_L0C = 2U;

const constexpr uint32_t C1_BASE_M = 128U;
const constexpr uint32_t C1_BASE_N = 128U;
const constexpr uint32_t C1_BASE_K = 32U;
const constexpr uint32_t C1_L1_K = 64U;
const constexpr uint32_t C1_DEPTH_A1 = 2U;
const constexpr uint32_t C1_DEPTH_B1 = 2U;
const constexpr uint32_t C1_DB_L0A = 2U;
const constexpr uint32_t C1_DB_L0B = 2U;
const constexpr uint32_t C1_DB_L0C = 2U;

REGISTER_OPS_TILING_TEMPLATE(MhcPreBackward, MhcPreBackwardTiling, DEFAULT_TILING_PRIORITY);

ge::graphStatus MhcPreBackwardTiling::GetPlatformInfo()
{
    const auto *compileInfo = context_->GetCompileInfo<MhcPreBackwardCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "get compile info failed"),
                return ge::GRAPH_FAILED);

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "get platform info failed"),
                return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);

    uint64_t aicNum = compileInfo->aicNum != 0U ? compileInfo->aicNum : ascendcPlatform.GetCoreNumAic();
    uint64_t aivNum = compileInfo->aivNum != 0U ? compileInfo->aivNum : ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0U || aivNum == 0U || aicNum > UINT32_MAX || aivNum > UINT32_MAX,
                OP_LOGE(context_->GetNodeName(), "invalid core count, aic=%lu, aiv=%lu", aicNum, aivNum),
                return ge::GRAPH_FAILED);
    blockDim_ = static_cast<uint32_t>(aicNum);
    vecCoreNum_ = static_cast<uint32_t>(aivNum);

    auto getMemorySize = [&ascendcPlatform](uint64_t parsedSize, platform_ascendc::CoreMemType type) {
        if (parsedSize != 0U) {
            return parsedSize;
        }
        uint64_t platformSize = 0U;
        ascendcPlatform.GetCoreMemSize(type, platformSize);
        return platformSize;
    };
    ubSize_ = getMemorySize(compileInfo->ubSize, platform_ascendc::CoreMemType::UB);
    l1Size_ = getMemorySize(compileInfo->l1Size, platform_ascendc::CoreMemType::L1);
    l2Size_ = getMemorySize(compileInfo->l2Size, platform_ascendc::CoreMemType::L2);
    l0ASize_ = getMemorySize(compileInfo->l0ASize, platform_ascendc::CoreMemType::L0_A);
    l0BSize_ = getMemorySize(compileInfo->l0BSize, platform_ascendc::CoreMemType::L0_B);
    l0CSize_ = getMemorySize(compileInfo->l0CSize, platform_ascendc::CoreMemType::L0_C);
    OP_CHECK_IF(ubSize_ == 0U || l1Size_ == 0U || l0ASize_ == 0U || l0BSize_ == 0U || l0CSize_ == 0U,
                OP_LOGE(context_->GetNodeName(), "invalid platform memory, ub=%lu, l1=%lu, l0a=%lu, l0b=%lu, l0c=%lu",
                        ubSize_, l1Size_, l0ASize_, l0BSize_, l0CSize_),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "platform info: aic=%u, aiv=%u, ub=%lu, l1=%lu, l2=%lu, l0a=%lu, l0b=%lu, l0c=%lu",
            blockDim_, vecCoreNum_, ubSize_, l1Size_, l2Size_, l0ASize_, l0BSize_, l0CSize_);
    return ge::GRAPH_SUCCESS;
}

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

void MhcPreBackwardTiling::SetMmConfig()
{
    tilingData_.mmConfigC0.set_baseM(C0_BASE_M);
    tilingData_.mmConfigC0.set_baseN(C0_BASE_N);
    tilingData_.mmConfigC0.set_baseK(C0_BASE_K);
    tilingData_.mmConfigC0.set_l1K(C0_L1_K);
    tilingData_.mmConfigC0.set_depthA1(C0_DEPTH_A1);
    tilingData_.mmConfigC0.set_depthB1(C0_DEPTH_B1);
    tilingData_.mmConfigC0.set_dbL0A(C0_DB_L0A);
    tilingData_.mmConfigC0.set_dbL0B(C0_DB_L0B);
    tilingData_.mmConfigC0.set_dbL0C(C0_DB_L0C);

    tilingData_.mmConfigC1.set_baseM(C1_BASE_M);
    tilingData_.mmConfigC1.set_baseN(C1_BASE_N);
    tilingData_.mmConfigC1.set_baseK(C1_BASE_K);
    tilingData_.mmConfigC1.set_l1K(C1_L1_K);
    tilingData_.mmConfigC1.set_depthA1(C1_DEPTH_A1);
    tilingData_.mmConfigC1.set_depthB1(C1_DEPTH_B1);
    tilingData_.mmConfigC1.set_dbL0A(C1_DB_L0A);
    tilingData_.mmConfigC1.set_dbL0B(C1_DB_L0B);
    tilingData_.mmConfigC1.set_dbL0C(C1_DB_L0C);
}

ge::graphStatus MhcPreBackwardTiling::ValidateMmConfig()
{
    MMConfig &mmConfigC0_ = tilingData_.mmConfigC0;
    MMConfig &mmConfigC1_ = tilingData_.mmConfigC1;
    constexpr uint64_t elementSize = sizeof(float);

    uint64_t c0A1SingleBufferBytes =
        static_cast<uint64_t>(mmConfigC0_.get_baseM()) * mmConfigC0_.get_l1K() * elementSize;
    uint64_t c0B1SingleBufferBytes =
        static_cast<uint64_t>(mmConfigC0_.get_baseN()) * mmConfigC0_.get_l1K() * elementSize;
    uint64_t c1A1SingleBufferBytes =
        static_cast<uint64_t>(mmConfigC1_.get_baseM()) * mmConfigC1_.get_l1K() * elementSize;
    uint64_t c1B1SingleBufferBytes =
        static_cast<uint64_t>(mmConfigC1_.get_baseN()) * mmConfigC1_.get_l1K() * elementSize;
    uint64_t l1SingleBufferBytes = std::max(std::max(c0A1SingleBufferBytes, c0B1SingleBufferBytes),
                                            std::max(c1A1SingleBufferBytes, c1B1SingleBufferBytes));

    uint64_t c0A2SingleBufferBytes =
        static_cast<uint64_t>(mmConfigC0_.get_baseM()) * mmConfigC0_.get_baseK() * elementSize;
    uint64_t c0B2SingleBufferBytes =
        static_cast<uint64_t>(mmConfigC0_.get_baseN()) * mmConfigC0_.get_baseK() * elementSize;
    uint64_t c1A2SingleBufferBytes =
        static_cast<uint64_t>(mmConfigC1_.get_baseM()) * mmConfigC1_.get_baseK() * elementSize;
    uint64_t c1B2SingleBufferBytes =
        static_cast<uint64_t>(mmConfigC1_.get_baseN()) * mmConfigC1_.get_baseK() * elementSize;
    uint64_t l0ABSingleBufferBytes = std::max(std::max(c0A2SingleBufferBytes, c0B2SingleBufferBytes),
                                              std::max(c1A2SingleBufferBytes, c1B2SingleBufferBytes));

    uint64_t c0L0CSingleBufferBytes =
        static_cast<uint64_t>(mmConfigC0_.get_baseM()) * mmConfigC0_.get_baseN() * elementSize;
    uint64_t c1L0CSingleBufferBytes =
        static_cast<uint64_t>(mmConfigC1_.get_baseM()) * mmConfigC1_.get_baseN() * elementSize;
    uint64_t l0CSingleBufferBytes = std::max(c0L0CSingleBufferBytes, c1L0CSingleBufferBytes);
    uint64_t depthA1 = std::max(mmConfigC0_.get_depthA1(), mmConfigC1_.get_depthA1());
    uint64_t depthB1 = std::max(mmConfigC0_.get_depthB1(), mmConfigC1_.get_depthB1());
    uint64_t dbL0 = std::max(std::max(mmConfigC0_.get_dbL0A(), mmConfigC1_.get_dbL0A()),
                             std::max(mmConfigC0_.get_dbL0B(), mmConfigC1_.get_dbL0B()));
    uint64_t dbL0C = std::max(mmConfigC0_.get_dbL0C(), mmConfigC1_.get_dbL0C());
    uint64_t l1UsedBytes = (depthA1 + depthB1) * l1SingleBufferBytes;
    uint64_t l0ABUsedBytes = dbL0 * l0ABSingleBufferBytes;
    uint64_t l0CUsedBytes = dbL0C * l0CSingleBufferBytes;
    OP_CHECK_IF(l1UsedBytes > l1Size_ || l0ABUsedBytes > l0ASize_ || l0ABUsedBytes > l0BSize_ ||
                    l0CUsedBytes > l0CSize_ || l1UsedBytes > UINT32_MAX || l0ABUsedBytes > UINT32_MAX ||
                    l0CUsedBytes > UINT32_MAX,
                OP_LOGE(context_->GetNodeName(),
                        "MM config exceeds platform memory: L1=%lu/%lu, L0A/B=%lu/%lu/%lu, L0C=%lu/%lu", l1UsedBytes,
                        l1Size_, l0ABUsedBytes, l0ASize_, l0BSize_, l0CUsedBytes, l0CSize_),
                return ge::GRAPH_FAILED);
    maxBufferDepth_ = static_cast<uint32_t>(std::max(std::max(depthA1, depthB1), std::max(dbL0, dbL0C)));
    l1UsedBytes_ = static_cast<uint32_t>(l1UsedBytes);
    l1SingleBufferElems_ = static_cast<uint32_t>(l1SingleBufferBytes / elementSize);
    l1BOffsetElems_ = static_cast<uint32_t>(depthA1 * l1SingleBufferElems_);
    l0ABUsedBytes_ = static_cast<uint32_t>(l0ABUsedBytes);
    l0ABSingleBufferElems_ = static_cast<uint32_t>(l0ABSingleBufferBytes / elementSize);
    l0CUsedBytes_ = static_cast<uint32_t>(l0CUsedBytes);
    l0CSingleBufferElems_ = static_cast<uint32_t>(l0CSingleBufferBytes / elementSize);

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

    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get attrs failed");
        return ge::GRAPH_FAILED;
    }
    auto hcEpsPtr = attrs->GetAttrPointer<float>(0);
    hcEps_ = (hcEpsPtr != nullptr) ? *hcEpsPtr : DEFAULT_HC_EPS;

    auto implModePtr = attrs->GetAttrPointer<int64_t>(IMPL_MODE_ATTR_INDEX);
    int64_t implMode = (implModePtr != nullptr) ? *implModePtr : IMPL_MODE_FP32;
    if (implMode != IMPL_MODE_FP32 && implMode != IMPL_MODE_HF32) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "op_impl_mode", std::to_string(implMode).c_str(),
                                              "op_impl_mode must be 0 (FP32) or 1 (HF32)");
        return ge::GRAPH_FAILED;
    }
    implMode_ = static_cast<uint32_t>(implMode);

    return ge::GRAPH_SUCCESS;
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
    tilingData_.set_implMode(implMode_);
    tilingData_.set_maxBufferDepth(maxBufferDepth_);
    tilingData_.set_l1UsedBytes(l1UsedBytes_);
    tilingData_.set_l1BOffsetElems(l1BOffsetElems_);
    tilingData_.set_l1SingleBufferElems(l1SingleBufferElems_);
    tilingData_.set_l0ABUsedBytes(l0ABUsedBytes_);
    tilingData_.set_l0ABSingleBufferElems(l0ABSingleBufferElems_);
    tilingData_.set_l0CUsedBytes(l0CUsedBytes_);
    tilingData_.set_l0CSingleBufferElems(l0CSingleBufferElems_);
    tilingData_.set_c0MBlock(C0_SET_SHAPE_M);
    tilingData_.set_c0NBlock(C0_SET_SHAPE_N);
    tilingData_.set_c1KBlock(C1_SET_SHAPE_K);
    tilingData_.set_c0ToV2Rows(C0_TO_V2_ROWS);
    tilingData_.set_v2ToC1Rows(V2_TO_C1_ROWS);
}

void MhcPreBackwardTiling::FillTilingData()
{
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

ge::graphStatus MhcPreBackwardTiling::TilingProcess()
{
    size_t userWorkspaceSize = CalculateWorkspaceSize(totalLength_, fusionSize_, blockDim_, vecCoreNum_, sizeof(float));
    size_t systemWorkspaceSize = SYSTEM_WORKSPACE_SIZE;
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

    SetMmConfig();
    if (ValidateMmConfig() != ge::GRAPH_SUCCESS) {
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
    OP_LOGD(context_->GetNodeName(), "implMode: [%u]", tilingData_.get_implMode());
    OP_LOGD(context_->GetNodeName(),
            "mmConfig: C0=[%u,%u,%u,L1K=%u], C1=[%u,%u,%u,L1K=%u], depth=[%u,%u,%u,%u,%u], "
            "block=[%u,%u,%u], sync=[%u,%u]",
            tilingData_.mmConfigC0.get_baseM(), tilingData_.mmConfigC0.get_baseN(), tilingData_.mmConfigC0.get_baseK(),
            tilingData_.mmConfigC0.get_l1K(), tilingData_.mmConfigC1.get_baseM(), tilingData_.mmConfigC1.get_baseN(),
            tilingData_.mmConfigC1.get_baseK(), tilingData_.mmConfigC1.get_l1K(), tilingData_.mmConfigC0.get_depthA1(),
            tilingData_.mmConfigC0.get_depthB1(), tilingData_.mmConfigC0.get_dbL0A(),
            tilingData_.mmConfigC0.get_dbL0B(), tilingData_.mmConfigC0.get_dbL0C(), tilingData_.get_c0MBlock(),
            tilingData_.get_c0NBlock(), tilingData_.get_c1KBlock(), tilingData_.get_c0ToV2Rows(),
            tilingData_.get_v2ToC1Rows());
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
