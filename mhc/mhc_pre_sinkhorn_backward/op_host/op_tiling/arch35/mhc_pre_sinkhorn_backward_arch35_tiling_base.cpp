/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mhc_pre_sinkhorn_backward_arch35_tiling_base.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "mhc_pre_sinkhorn_backward_arch35_tiling_base.h"
#include "log/log.h"

using namespace AscendC;
namespace optiling {

constexpr uint8_t GRAD_HIN_IDX = 0;
constexpr uint8_t GRAD_H_POST_IDX = 1;
constexpr uint8_t GRAD_H_RES_IDX = 2;
constexpr uint8_t INPUT_X_IDX = 3;
constexpr uint8_t PHI_IDX = 4;
constexpr uint8_t ALPHA_IDX = 5;
constexpr uint8_t BIAS_IDX = 6;
constexpr uint8_t H_PRE_IDX = 7;
constexpr uint8_t HC_BEFORE_NORM_IDX = 8;
constexpr uint8_t INV_RMS_IDX = 9;
constexpr uint8_t SUM_OUT_IDX = 10;
constexpr uint8_t NORM_OUT_IDX = 11;
constexpr uint8_t GRAD_X_IDX = 0;
constexpr uint8_t GRAD_PHI_IDX = 1;
constexpr uint8_t GRAD_ALPHA_IDX = 2;
constexpr uint8_t GRAD_BIAS_IDX = 3;
constexpr uint8_t BATCH_SIZE_DIM_IDX = 0;
constexpr uint8_t SEQ_LENGTH_DIM_IDX = 1;
constexpr uint8_t N_DIM_IDX = 2;
constexpr uint8_t C_DIM_IDX = 3;
constexpr float DEFAULT_EPS = 1e-6f;
constexpr uint8_t ITER_COUNT_IDX = 0;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint8_t BF16_BYTE_SIZE = 2;
constexpr int64_t ALPHA_SIZE_3 = 3;
constexpr int64_t N_SIZE_4 = 4;
constexpr int64_t C0_SIZE = 64;
constexpr int64_t TILE_UPPER_BOUND = 64;
constexpr double UB_USAGE_RATIO = 0.95;
constexpr int64_t N_SQUARE_BUF_COUNT = 4;
constexpr int64_t N_LINEAR_BUF_COUNT = 9;
constexpr int64_t SCALAR_BUF_COUNT = 3;

using namespace ge;
using namespace std;
using namespace AscendC;

bool MhcPreSinkhornBackwardArch35Tiling::IsCapable() { return true; }

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetPlatformInfo()
{
    OP_LOGD(opName, "MhcPreSinkhornBackwardArch35Tiling GetPlatformInfo");
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNumAiv <= 0), OP_LOGE(opName, "ScatterNdUpdateTiling fail to get coreNumAiv."),
                return ge::GRAPH_FAILED);
    coreNumAiv_ = coreNumAiv;
    auto coreNumAic = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF((coreNumAic <= 0), OP_LOGE(opName, "ScatterNdUpdateTiling fail to get coreNumAic."),
                return ge::GRAPH_FAILED);
    coreNumAic_ = coreNumAic;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName, "MhcPreSinkhornBackwardArch35Tiling GetShapeAttrsInfo");
    const auto xShapePtr = context_->GetInputShape(INPUT_X_IDX);
    if (xShapePtr == nullptr) {
        OP_LOGE(context_, "input x shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const auto skSumPtr = context_->GetInputShape(SUM_OUT_IDX);
    if (skSumPtr == nullptr) {
        OP_LOGE(context_, "input sum_out shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto xShape = xShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 4,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "xShape verify failed, x must be 4D, but got %lu dims", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    auto skSumShape = skSumPtr->GetStorageShape();

    auto attrsPtr = context_->GetAttrs();
    if (attrsPtr == nullptr) {
        OP_LOGE(context_, "attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto epsPtr = attrsPtr->GetAttrPointer<float>(0);
    hcEps_ = (epsPtr != nullptr) ? static_cast<float>(*epsPtr) : DEFAULT_EPS;

    batchSize_ = xShape.GetDim(BATCH_SIZE_DIM_IDX);
    seqLength_ = xShape.GetDim(SEQ_LENGTH_DIM_IDX);
    n_ = xShape.GetDim(N_DIM_IDX);
    c_ = xShape.GetDim(C_DIM_IDX);
    skIterCount_ = skSumShape.GetDim(ITER_COUNT_IDX) / 2;
    OP_CHECK_IF(CheckShape(batchSize_, seqLength_, n_, c_) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "CheckShape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        skIterCount_ != 20,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "sk_iter_count must be 20, but got %ld", skIterCount_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(c_ <= 0 || c_ >= 100000 || c_ % 128 != 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "c must be > 0, < 100000 and divisible by 128, but got %ld", c_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::CheckShapeBase(int64_t batchSize, int64_t seqLength, int64_t n,
                                                                   int64_t c)
{
    auto opName = context_->GetNodeName();

    auto gradHinShapePtr = context_->GetInputShape(GRAD_HIN_IDX);
    OP_CHECK_IF(gradHinShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradHin shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradHinShape = gradHinShapePtr->GetStorageShape();
    OP_CHECK_IF(gradHinShape.GetDimNum() != 3,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradHin must be 3D, but got %lu dims",
                                            gradHinShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        gradHinShape.GetDim(0) != batchSize || gradHinShape.GetDim(1) != seqLength || gradHinShape.GetDim(2) != c,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName, "ShapeVerify gradHin failed, expected (B=%ld, S=%ld, C=%ld), but got (%ld, %ld, %ld)", batchSize,
            seqLength, c, gradHinShape.GetDim(0), gradHinShape.GetDim(1), gradHinShape.GetDim(2)),
        return ge::GRAPH_FAILED);

    auto gradHPostShapePtr = context_->GetInputShape(GRAD_H_POST_IDX);
    OP_CHECK_IF(gradHPostShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradHPost shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradHPostShape = gradHPostShapePtr->GetStorageShape();
    OP_CHECK_IF(gradHPostShape.GetDimNum() != 3,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradHPost must be 3D, but got %lu dims",
                                            gradHPostShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        gradHPostShape.GetDim(0) != batchSize || gradHPostShape.GetDim(1) != seqLength || gradHPostShape.GetDim(2) != n,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName, "ShapeVerify gradHPost failed, expected (B=%ld, S=%ld, N=%ld), but got (%ld, %ld, %ld)", batchSize,
            seqLength, n, gradHPostShape.GetDim(0), gradHPostShape.GetDim(1), gradHPostShape.GetDim(2)),
        return ge::GRAPH_FAILED);

    auto gradHResShapePtr = context_->GetInputShape(GRAD_H_RES_IDX);
    OP_CHECK_IF(gradHResShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradHRes shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradHResShape = gradHResShapePtr->GetStorageShape();
    auto gradHResDimNum = gradHResShape.GetDimNum();
    OP_CHECK_IF((gradHResDimNum != 3 && gradHResDimNum != 4),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradHRes must be 3D or 4D, but got %lu dims",
                                            gradHResDimNum),
                return ge::GRAPH_FAILED);
    if (gradHResDimNum == 3) {
        OP_CHECK_IF(
            gradHResShape.GetDim(0) != batchSize || gradHResShape.GetDim(1) != seqLength ||
                gradHResShape.GetDim(2) != n * n,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "ShapeVerify gradHRes failed, expected (B=%ld, S=%ld, N*N=%ld), but got (%ld, %ld, %ld)",
                batchSize, seqLength, n * n, gradHResShape.GetDim(0), gradHResShape.GetDim(1), gradHResShape.GetDim(2)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            gradHResShape.GetDim(0) != batchSize || gradHResShape.GetDim(1) != seqLength ||
                gradHResShape.GetDim(2) != n || gradHResShape.GetDim(3) != n,
            OPS_REPORT_VECTOR_INNER_ERR(opName,
                                        "ShapeVerify gradHRes failed, expected (B=%ld, S=%ld, N=%ld, N=%ld), "
                                        "but got (%ld, %ld, %ld, %ld)",
                                        batchSize, seqLength, n, n, gradHResShape.GetDim(0), gradHResShape.GetDim(1),
                                        gradHResShape.GetDim(2), gradHResShape.GetDim(3)),
            return ge::GRAPH_FAILED);
    }

    auto phiShapePtr = context_->GetInputShape(PHI_IDX);
    OP_CHECK_IF(phiShapePtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, phi shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto phiShape = phiShapePtr->GetStorageShape();
    OP_CHECK_IF(phiShape.GetDimNum() != 2,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, phi must be 2D, but got %lu dims",
                                            phiShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(phiShape.GetDim(0) != n * n + 2 * n || phiShape.GetDim(1) != n * c,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify phi failed, expected (2N+N^2=%ld, N*C=%ld), but got (%ld, %ld)", n * n + 2 * n,
                    n * c, phiShape.GetDim(0), phiShape.GetDim(1)),
                return ge::GRAPH_FAILED);

    auto alphaShapePtr = context_->GetInputShape(ALPHA_IDX);
    OP_CHECK_IF(alphaShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, alpha shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto alphaShape = alphaShapePtr->GetStorageShape();
    OP_CHECK_IF(alphaShape.GetDimNum() != 1 || alphaShape.GetDim(0) != ALPHA_SIZE_3,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify alpha failed, expected (3), but got shape with %lu dims, dim0=%ld",
                    alphaShape.GetDimNum(), alphaShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    auto biasShapePtr = context_->GetInputShape(BIAS_IDX);
    OP_CHECK_IF(biasShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, bias shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto biasShape = biasShapePtr->GetStorageShape();
    OP_CHECK_IF(biasShape.GetDimNum() != 1 || biasShape.GetDim(0) != n * n + 2 * n,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify bias failed, expected (2N+N^2=%ld), but got shape with %lu dims, dim0=%ld",
                    n * n + 2 * n, biasShape.GetDimNum(), biasShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    auto hPreShapePtr = context_->GetInputShape(H_PRE_IDX);
    OP_CHECK_IF(hPreShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, hPre shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto hPreShape = hPreShapePtr->GetStorageShape();
    OP_CHECK_IF(hPreShape.GetDimNum() != 3,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, hPre must be 3D, but got %lu dims",
                                            hPreShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hPreShape.GetDim(0) != batchSize || hPreShape.GetDim(1) != seqLength || hPreShape.GetDim(2) != n,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify hPre failed, expected (B=%ld, S=%ld, N=%ld), but got (%ld, %ld, %ld)",
                    batchSize, seqLength, n, hPreShape.GetDim(0), hPreShape.GetDim(1), hPreShape.GetDim(2)),
                return ge::GRAPH_FAILED);

    auto hcBeforeNormShapePtr = context_->GetInputShape(HC_BEFORE_NORM_IDX);
    OP_CHECK_IF(hcBeforeNormShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, hcBeforeNorm shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto hcBeforeNormShape = hcBeforeNormShapePtr->GetStorageShape();
    OP_CHECK_IF(hcBeforeNormShape.GetDimNum() != 3,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, hcBeforeNorm must be 3D, but got %lu dims",
                                            hcBeforeNormShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hcBeforeNormShape.GetDim(0) != batchSize || hcBeforeNormShape.GetDim(1) != seqLength ||
            hcBeforeNormShape.GetDim(2) != n * n + 2 * n,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName, "ShapeVerify hcBeforeNorm failed, expected (B=%ld, S=%ld, N^2+2N=%ld), but got (%ld, %ld, %ld)",
            batchSize, seqLength, n * n + 2 * n, hcBeforeNormShape.GetDim(0), hcBeforeNormShape.GetDim(1),
            hcBeforeNormShape.GetDim(2)),
        return ge::GRAPH_FAILED);

    auto invRmsShapePtr = context_->GetInputShape(INV_RMS_IDX);
    OP_CHECK_IF(invRmsShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, invRms shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto invRmsShape = invRmsShapePtr->GetStorageShape();
    OP_CHECK_IF(invRmsShape.GetDimNum() != 3,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, invRms must be 3D, but got %lu dims",
                                            invRmsShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(invRmsShape.GetDim(0) != batchSize || invRmsShape.GetDim(1) != seqLength || invRmsShape.GetDim(2) != 1,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify invRms failed, expected (B=%ld, S=%ld, 1), but got (%ld, %ld, %ld)", batchSize,
                    seqLength, invRmsShape.GetDim(0), invRmsShape.GetDim(1), invRmsShape.GetDim(2)),
                return ge::GRAPH_FAILED);

    auto sumOutShapePtr = context_->GetInputShape(SUM_OUT_IDX);
    OP_CHECK_IF(sumOutShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, sumOut shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto sumOutShape = sumOutShapePtr->GetStorageShape();
    OP_CHECK_IF(sumOutShape.GetDimNum() != 4,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, sumOut must be 4D, but got %lu dims",
                                            sumOutShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sumOutShape.GetDim(1) != batchSize || sumOutShape.GetDim(2) != seqLength || sumOutShape.GetDim(3) != n,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName, "ShapeVerify sumOut failed, expected (2*iter, B=%ld, S=%ld, N=%ld), but got (%ld, %ld, %ld, %ld)",
            batchSize, seqLength, n, sumOutShape.GetDim(0), sumOutShape.GetDim(1), sumOutShape.GetDim(2),
            sumOutShape.GetDim(3)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sumOutShape.GetDim(0) % 2 != 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify sumOut failed, dim0 must be even (2*iter_count), but got %ld",
                                    sumOutShape.GetDim(0)),
        return ge::GRAPH_FAILED);

    auto normOutShapePtr = context_->GetInputShape(NORM_OUT_IDX);
    OP_CHECK_IF(normOutShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, normOut shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto normOutShape = normOutShapePtr->GetStorageShape();
    OP_CHECK_IF(normOutShape.GetDimNum() != 5,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, normOut must be 5D, but got %lu dims",
                                            normOutShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(normOutShape.GetDim(1) != batchSize || normOutShape.GetDim(2) != seqLength ||
                    normOutShape.GetDim(3) != n || normOutShape.GetDim(4) != n,
                OPS_REPORT_VECTOR_INNER_ERR(opName,
                                            "ShapeVerify normOut failed, expected (2*iter, B=%ld, S=%ld, N=%ld, "
                                            "N=%ld), but got (%ld, %ld, %ld, %ld, %ld)",
                                            batchSize, seqLength, n, n, normOutShape.GetDim(0), normOutShape.GetDim(1),
                                            normOutShape.GetDim(2), normOutShape.GetDim(3), normOutShape.GetDim(4)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(normOutShape.GetDim(0) != sumOutShape.GetDim(0),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify normOut dim0(%ld) must equal sumOut dim0(%ld)",
                                            normOutShape.GetDim(0), sumOutShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    auto gradXShapePtr = context_->GetOutputShape(GRAD_X_IDX);
    OP_CHECK_IF(gradXShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradX shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradXShape = gradXShapePtr->GetStorageShape();
    OP_CHECK_IF(gradXShape.GetDimNum() != 4,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradX must be 4D, but got %lu dims",
                                            gradXShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        gradXShape.GetDim(0) != batchSize || gradXShape.GetDim(1) != seqLength || gradXShape.GetDim(2) != n ||
            gradXShape.GetDim(3) != c,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName, "ShapeVerify gradX failed, expected (B=%ld, S=%ld, N=%ld, C=%ld), but got (%ld, %ld, %ld, %ld)",
            batchSize, seqLength, n, c, gradXShape.GetDim(0), gradXShape.GetDim(1), gradXShape.GetDim(2),
            gradXShape.GetDim(3)),
        return ge::GRAPH_FAILED);

    auto gradPhiShapePtr = context_->GetOutputShape(GRAD_PHI_IDX);
    OP_CHECK_IF(gradPhiShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradPhi shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradPhiShape = gradPhiShapePtr->GetStorageShape();
    OP_CHECK_IF(gradPhiShape.GetDimNum() != 2,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradPhi must be 2D, but got %lu dims",
                                            gradPhiShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gradPhiShape.GetDim(0) != n * n + 2 * n || gradPhiShape.GetDim(1) != n * c,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify gradPhi failed, expected (2N+N^2=%ld, N*C=%ld), but got (%ld, %ld)",
                    n * n + 2 * n, n * c, gradPhiShape.GetDim(0), gradPhiShape.GetDim(1)),
                return ge::GRAPH_FAILED);

    auto gradAlphaShapePtr = context_->GetOutputShape(GRAD_ALPHA_IDX);
    OP_CHECK_IF(gradAlphaShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradAlpha shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradAlphaShape = gradAlphaShapePtr->GetStorageShape();
    OP_CHECK_IF(gradAlphaShape.GetDimNum() != 1 || gradAlphaShape.GetDim(0) != ALPHA_SIZE_3,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify gradAlpha failed, expected (3), but got shape with %lu dims, dim0=%ld",
                    gradAlphaShape.GetDimNum(), gradAlphaShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    auto gradBiasShapePtr = context_->GetOutputShape(GRAD_BIAS_IDX);
    OP_CHECK_IF(gradBiasShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradBias shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradBiasShape = gradBiasShapePtr->GetStorageShape();
    OP_CHECK_IF(gradBiasShape.GetDimNum() != 1 || gradBiasShape.GetDim(0) != n * n + 2 * n,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify gradBias failed, expected (2N+N^2=%ld), but got shape with %lu dims, dim0=%ld",
                    n * n + 2 * n, gradBiasShape.GetDimNum(), gradBiasShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::CheckShape(int64_t batchSize, int64_t seqLength, int64_t n,
                                                               int64_t c)
{
    auto opName = context_->GetNodeName();
    OP_CHECK_IF(n != N_SIZE_4, OPS_REPORT_VECTOR_INNER_ERR(opName, "n must be 4, but got %lu", n),
                return ge::GRAPH_FAILED);
    return CheckShapeBase(batchSize, seqLength, n, c);
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::SetTilingData()
{
    MhcPreSinkhornBackwardArch35TilingData *tilingData =
        context_->GetTilingData<MhcPreSinkhornBackwardArch35TilingData>();
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);
    ubSize_ = ubSize_ - 32 * 1024;
    auto floatDataType = matmul_tiling::DataType::DT_FLOAT;
    auto bf16DataType = matmul_tiling::DataType::DT_BF16;

    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling mm2Tiling(ascendPlatformInfo);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bf16DataType);
    mm1Tiling.SetOrgShape(mm1M_, mm1N_, mm1K_);
    mm1Tiling.SetShape(mm1M_, mm1N_, mm1K_);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    mm1Tiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN);
    if (mm1Tiling.GetTiling(tilingData->mm1TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    mm2Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType, true);
    mm2Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm2Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm2Tiling.SetOrgShape(mm2M_, mm2N_, mm2K_);
    mm2Tiling.SetShape(mm2M_, mm2N_, mm2K_);
    mm2Tiling.SetBias(false);
    mm2Tiling.SetBufferSpace(-1, -1, -1);
    mm2Tiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN);
    if (mm2Tiling.GetTiling(tilingData->mm2TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    tilingData->batchSize = batchSize_;
    tilingData->seqLength = seqLength_;
    tilingData->c = c_;
    tilingData->n = n_;
    tilingData->c0 = c0_;
    tilingData->c1 = c1_;
    tilingData->aivNum = coreNumAiv_;
    tilingData->skIterCount = skIterCount_;
    tilingData->tileSize = tile_;

    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35Tiling::DoUbTiling()
{
    c0_ = C0_SIZE;
    c1_ = c_ / c0_;
    coreTaskCount_ = (batchSize_ * seqLength_ + coreNumAiv_ - 1) / coreNumAiv_;
    tile_ = min(coreTaskCount_, TILE_UPPER_BOUND);
    tileUB_ = (ubSize_ * UB_USAGE_RATIO - (n_ * sizeof(float) - DOUBLE_BUFFER * 2 * BF16_BYTE_SIZE) * c_) /
              ((n_ * n_ * N_SQUARE_BUF_COUNT + n_ * N_LINEAR_BUF_COUNT + SCALAR_BUF_COUNT) * sizeof(float));
    tile_ = min(tile_, tileUB_);

    mm1K_ = n_ * n_ + 2 * n_;
    mm1M_ = tile_ * 2;
    mm1N_ = n_ * c_;

    mm2K_ = tile_ * 2;
    mm2M_ = n_ * n_ + 2 * n_;
    mm2N_ = n_ * c_;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::DoOpTiling()
{
    DoUbTiling();
    return SetTilingData();
}

uint64_t MhcPreSinkhornBackwardArch35Tiling::GetTilingKey() const
{
    bool isDeterministic = false;
    return GET_TPL_TILING_KEY(isDeterministic);
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::PostTiling()
{
    context_->SetDynUBufSize(ubSize_);
    context_->SetBlockDim(coreNumAic_);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);
    size_t gradHat2Workspace = batchSize_ * seqLength_ * (n_ * n_ + 2 * n_ + n_ * c_) * sizeof(float);
    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = gradHat2Workspace;
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(opName, "fail to GetWorkspaceSizes"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35Tiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "batchSize: " << batchSize_ << std::endl;
    info << "seqLength: " << seqLength_ << std::endl;
    info << "c: " << c_ << std::endl;
    info << "n: " << n_ << std::endl;
    info << "c0: " << c0_ << std::endl;
    info << "c1: " << c1_ << std::endl;
    info << "aivNum: " << coreNumAiv_ << std::endl;
    info << "skIterCount: " << skIterCount_ << std::endl;
    info << "tileSize: " << tile_ << std::endl;
    info << "mm1K: " << mm1K_ << std::endl;
    info << "mm1M: " << mm1M_ << std::endl;
    info << "mm1N: " << mm1N_ << std::endl;
    info << "mm2K: " << mm2K_ << std::endl;
    info << "mm2M: " << mm2M_ << std::endl;
    info << "mm2N: " << mm2N_ << std::endl;

    OP_LOGI(opName, "Tiling info is: %s", info.str().c_str());
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

REGISTER_OPS_TILING_TEMPLATE(MhcPreSinkhornBackward, MhcPreSinkhornBackwardArch35Tiling, 10);
} // namespace optiling
