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
 * \file mhc_pre_sinkhorn_backward_arch22_tiling.cpp
 * \brief MhcPreSinkhornBackward operator tiling implementation
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "mhc_pre_sinkhorn_backward_arch22_tiling.h"
#include "log/log.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch22/mhc_pre_sinkhorn_backward_data_arch22.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch22/mhc_pre_sinkhorn_backward_key_arch22.h"

#define CHECK_NULLPTR(ptr) \
    if (ptr == nullptr) { \
        return ge::GRAPH_FAILED; \
    }

namespace {
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
constexpr float DEFAULT_EPS = 1e-6f;
constexpr uint8_t BATCH_SIZE_DIM_IDX = 0;
constexpr uint8_t SEQ_LENGTH_DIM_IDX = 1;
constexpr uint8_t N_DIM_IDX = 2;
constexpr uint8_t C_DIM_IDX = 3;
constexpr uint8_t ITER_COUNT_IDX = 0;
constexpr int64_t N_SIZE_4 = 4;
constexpr int64_t ALPHA_SIZE_3 = 3;
constexpr int64_t C_V_RATIO = 2;
constexpr int64_t DEFAULT_KEY = 0;
constexpr int64_t DETERMINISTIC_KEY = 1;
constexpr int64_t ELEMENTS_SIZE_PER_BLOCK = 8;
} // namespace

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {

ge::graphStatus ShapeVerify(gert::TilingContext *context, int64_t batchSize, int64_t seqLength, int64_t n, int64_t c)
{
    auto opName = context->GetNodeName();

    auto gradHinShapePtr = context->GetInputShape(GRAD_HIN_IDX);
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

    auto gradHPostShapePtr = context->GetInputShape(GRAD_H_POST_IDX);
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

    auto gradHResShapePtr = context->GetInputShape(GRAD_H_RES_IDX);
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

    auto phiShapePtr = context->GetInputShape(PHI_IDX);
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

    auto alphaShapePtr = context->GetInputShape(ALPHA_IDX);
    OP_CHECK_IF(alphaShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, alpha shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto alphaShape = alphaShapePtr->GetStorageShape();
    OP_CHECK_IF(alphaShape.GetDimNum() != 1 || alphaShape.GetDim(0) != ALPHA_SIZE_3,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify alpha failed, expected (3), but got shape with %lu dims, dim0=%ld",
                    alphaShape.GetDimNum(), alphaShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    auto biasShapePtr = context->GetInputShape(BIAS_IDX);
    OP_CHECK_IF(biasShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, bias shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto biasShape = biasShapePtr->GetStorageShape();
    OP_CHECK_IF(biasShape.GetDimNum() != 1 || biasShape.GetDim(0) != n * n + 2 * n,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify bias failed, expected (2N+N^2=%ld), but got shape with %lu dims, dim0=%ld",
                    n * n + 2 * n, biasShape.GetDimNum(), biasShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    auto hPreShapePtr = context->GetInputShape(H_PRE_IDX);
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

    auto hcBeforeNormShapePtr = context->GetInputShape(HC_BEFORE_NORM_IDX);
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

    auto invRmsShapePtr = context->GetInputShape(INV_RMS_IDX);
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

    auto sumOutShapePtr = context->GetInputShape(SUM_OUT_IDX);
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

    auto normOutShapePtr = context->GetInputShape(NORM_OUT_IDX);
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

    auto gradXShapePtr = context->GetOutputShape(GRAD_X_IDX);
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

    auto gradPhiShapePtr = context->GetOutputShape(GRAD_PHI_IDX);
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

    auto gradAlphaShapePtr = context->GetOutputShape(GRAD_ALPHA_IDX);
    OP_CHECK_IF(gradAlphaShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradAlpha shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradAlphaShape = gradAlphaShapePtr->GetStorageShape();
    OP_CHECK_IF(gradAlphaShape.GetDimNum() != 1 || gradAlphaShape.GetDim(0) != ALPHA_SIZE_3,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify gradAlpha failed, expected (3), but got shape with %lu dims, dim0=%ld",
                    gradAlphaShape.GetDimNum(), gradAlphaShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    auto gradBiasShapePtr = context->GetOutputShape(GRAD_BIAS_IDX);
    OP_CHECK_IF(gradBiasShapePtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradBias shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto gradBiasShape = gradBiasShapePtr->GetStorageShape();
    OP_CHECK_IF(gradBiasShape.GetDimNum() != 1 || gradBiasShape.GetDim(0) != n * n + 2 * n,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName, "ShapeVerify gradBias failed, expected (2N+N^2=%ld), but got shape with %lu dims, dim0=%ld",
                    n * n + 2 * n, gradBiasShape.GetDimNum(), gradBiasShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n != N_SIZE_4, OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, N must be 4, but got %ld", n),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMhcPreSinkhornBackwardArch22(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t ubSize;
    ascendPlatformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto aicNum = ascendPlatformInfo.GetCoreNumAic();
    auto aivNum = ascendPlatformInfo.GetCoreNumAiv();
    if (aicNum == 0 || aivNum == 0) {
        OP_LOGE(context, "aicNum=%lu or aivNum=%lu is invalid", aicNum, aivNum);
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(aicNum);
    const auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    if (xShapePtr == nullptr) {
        OP_LOGE(context, "input x shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto xShape = xShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 4,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "xShape verify failed, x must be 4D, but got %lu dims", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        OP_LOGE(context, "attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto epsPtr = attrsPtr->GetAttrPointer<float>(0);
    float eps = (epsPtr != nullptr) ? static_cast<float>(*epsPtr) : DEFAULT_EPS;

    int64_t batchSize = xShape.GetDim(BATCH_SIZE_DIM_IDX);
    int64_t seqLength = xShape.GetDim(SEQ_LENGTH_DIM_IDX);
    int64_t n = xShape.GetDim(N_DIM_IDX);
    int64_t c = xShape.GetDim(C_DIM_IDX);

    OP_CHECK_IF(ShapeVerify(context, batchSize, seqLength, n, c) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ShapeVerify failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(n != N_SIZE_4,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "n must be %ld, but got %ld", N_SIZE_4, n),
                return ge::GRAPH_FAILED);

    int64_t c0 = 256; // must be VL
    int64_t c1 = c / c0;
    int64_t cTail = c % c0;
    int64_t cTailunaglin = cTail % 64 + 64 * cTail / (64 + 128);

    int64_t tile = 32;
    const auto sumOutPtr = context->GetInputShape(10);
    CHECK_NULLPTR(sumOutPtr);
    auto sumOutShape = sumOutPtr->GetStorageShape();
    int64_t skIterCount = sumOutShape.GetDim(0) / C_V_RATIO;

    OP_CHECK_IF(
        skIterCount != 20,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "sk_iter_count must be 20, but got %ld", skIterCount),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(c <= 0 || c >= 100000 || c % 128 != 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "c must be > 0, < 100000 and divisible by 128, but got %ld", c),
                return ge::GRAPH_FAILED);

    int64_t mm1K = n * n + 2 * n;
    int64_t mm1M = tile * 2;
    int64_t mm1N = n * c;

    int64_t mm2K = tile * 2;
    int64_t mm2M = n * n + 2 * n;
    int64_t mm2N = n * c;

    MhcPreSinkhornBackwardArch22TilingData *tilingData =
        context->GetTilingData<MhcPreSinkhornBackwardArch22TilingData>();
    auto featureDataType = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling mm2Tiling(ascendPlatformInfo);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm1Tiling.SetOrgShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetShape(mm1M, mm1N, mm1K);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    if (mm1Tiling.GetTiling(tilingData->mm1TilingData) == -1) {
        OP_LOGE(context, "mm1Tiling.GetTiling failed, M=%ld, N=%ld, K=%ld", mm1M, mm1N, mm1K);
        return ge::GRAPH_FAILED;
    }

    mm2Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType, true);
    mm2Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm2Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, featureDataType);
    mm2Tiling.SetOrgShape(mm2M, mm2N, mm2K);
    mm2Tiling.SetShape(mm2M, mm2N, mm2K);
    mm2Tiling.SetBias(false);
    mm2Tiling.SetBufferSpace(-1, -1, -1);
    if (mm2Tiling.GetTiling(tilingData->mm2TilingData) == -1) {
        OP_LOGE(context, "mm2Tiling.GetTiling failed, M=%ld, N=%ld, K=%ld", mm2M, mm2N, mm2K);
        return ge::GRAPH_FAILED;
    }

    tilingData->n = n;
    tilingData->batchSize = batchSize;
    tilingData->seqLength = seqLength;
    tilingData->c = c;
    tilingData->cTail = cTail;
    tilingData->c0 = c0;
    tilingData->c1 = c1;
    tilingData->aivNum = aivNum;
    tilingData->ubSize = ubSize;
    tilingData->skIterCount = skIterCount;
    tilingData->eps = eps;
    tilingData->tileSize = tile;

    size_t xCastWorkspace = batchSize * seqLength * n * c * sizeof(float) * 2;
    size_t gradHat2Workspace = batchSize * seqLength * (n * n + 2 * n) * sizeof(float);

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t usrWorkSpaceSize = xCastWorkspace + gradHat2Workspace;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE(context, "GetWorkspaceSizes() returned nullptr");
        return ge::GRAPH_FAILED;
    }

    auto isDeterministic = (context->GetDeterministic() == 1);
    context->SetScheduleMode(1); // 1: batchmode模式
    ASCENDC_TPL_SEL_PARAM(context, isDeterministic);
    if (isDeterministic) {
        currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize +
                              aicNum * (n * n + 2 * n) * (n * c) * sizeof(float) +
                              aivNum * (n * n + 2 * n + ELEMENTS_SIZE_PER_BLOCK) * sizeof(float);
    } else {
        currentWorkspace[0] = systemWorkspaceSize + usrWorkSpaceSize;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
