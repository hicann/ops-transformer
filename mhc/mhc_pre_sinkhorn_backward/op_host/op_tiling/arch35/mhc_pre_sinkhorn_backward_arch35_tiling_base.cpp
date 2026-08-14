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
#include <initializer_list>

using namespace AscendC;
namespace optiling {

namespace {
const gert::StorageShape *GetInputShapeChecked(gert::TilingContext *context, const char *opName,
                                               const char *tensorName, uint8_t idx)
{
    const gert::StorageShape *ptr = context->GetInputShape(idx);
    OP_CHECK_IF(ptr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, %s shape is nullptr", tensorName),
                return nullptr);
    return ptr;
}

const gert::StorageShape *GetOutputShapeChecked(gert::TilingContext *context, const char *opName,
                                                const char *tensorName, uint8_t idx)
{
    const gert::StorageShape *ptr = context->GetOutputShape(idx);
    OP_CHECK_IF(ptr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, %s shape is nullptr", tensorName),
                return nullptr);
    return ptr;
}

ge::graphStatus VerifyShape(const char *opName, const char *tensorName, const gert::StorageShape *shapePtr,
                            const int64_t *batchDims, size_t batchDimCount,
                            std::initializer_list<int64_t> suffixDims, size_t startDim = 0)
{
    const gert::Shape &shape = shapePtr->GetStorageShape();
    size_t expectedDimNum = startDim + batchDimCount + suffixDims.size();
    OP_CHECK_IF(shape.GetDimNum() != expectedDimNum,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, %s must be %luD, but got %lu dims",
                                            tensorName, expectedDimNum, shape.GetDimNum()),
                return ge::GRAPH_FAILED);

    size_t dimIdx = startDim;
    for (size_t i = 0; i < batchDimCount; i++) {
        OP_CHECK_IF(shape.GetDim(dimIdx) != batchDims[i],
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify %s dim[%lu] failed: expected %ld, got %ld",
                                                tensorName, dimIdx, batchDims[i], shape.GetDim(dimIdx)),
                    return ge::GRAPH_FAILED);
        dimIdx++;
    }

    for (auto expected : suffixDims) {
        OP_CHECK_IF(shape.GetDim(dimIdx) != expected,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify %s dim[%lu] failed: expected %ld, got %ld",
                                                tensorName, dimIdx, expected, shape.GetDim(dimIdx)),
                    return ge::GRAPH_FAILED);
        dimIdx++;
    }

    return ge::GRAPH_SUCCESS;
}
} // anonymous namespace

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
constexpr int64_t MAX_N = 8;
constexpr int64_t EXPECTED_SK_ITER_COUNT = 20;
constexpr int64_t MAX_C_VALUE = 100000;
constexpr int64_t C_ALIGNMENT = 128;
constexpr int64_t ITER_COUNT_DIVISOR = 2;
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
    skIterCount_ = skSumShape.GetDim(ITER_COUNT_IDX) / ITER_COUNT_DIVISOR;
    OP_CHECK_IF(CheckShape(batchSize_, seqLength_, n_, c_) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "CheckShape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        skIterCount_ != EXPECTED_SK_ITER_COUNT,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "sk_iter_count must be %ld, but got %ld",
                                    EXPECTED_SK_ITER_COUNT, skIterCount_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(c_ <= 0 || c_ >= MAX_C_VALUE || c_ % C_ALIGNMENT != 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "c must be > 0, < %ld and divisible by %ld, but got %ld", MAX_C_VALUE,
                                            C_ALIGNMENT, c_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::CheckShapeBase(int64_t batchSize, int64_t seqLength, int64_t n,
                                                                   int64_t c, bool is3D)
{
    auto opName = context_->GetNodeName();
    int64_t bs = batchSize * seqLength;
    int64_t hcMix = n * n + 2 * n;

    int64_t batchPrefix[2] = {0, 0};
    size_t batchDimCount;
    if (is3D) {
        batchPrefix[0] = bs;
        batchDimCount = 1;
    } else {
        batchPrefix[0] = batchSize;
        batchPrefix[1] = seqLength;
        batchDimCount = 2;
    }

    auto gradHin = GetInputShapeChecked(context_, opName, "gradHin", GRAD_HIN_IDX);
    if (!gradHin) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "gradHin", gradHin, batchPrefix, batchDimCount, {c}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto gradHPost = GetInputShapeChecked(context_, opName, "gradHPost", GRAD_H_POST_IDX);
    if (!gradHPost) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "gradHPost", gradHPost, batchPrefix, batchDimCount, {n}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto gradHRes = GetInputShapeChecked(context_, opName, "gradHRes", GRAD_H_RES_IDX);
    if (!gradHRes) {
        return ge::GRAPH_FAILED;
    }
    auto gradHResDimNum = gradHRes->GetStorageShape().GetDimNum();
    if (gradHResDimNum == batchDimCount + 1) {
        if (VerifyShape(opName, "gradHRes", gradHRes, batchPrefix, batchDimCount, {n * n}) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (gradHResDimNum == batchDimCount + 2) {
        if (VerifyShape(opName, "gradHRes", gradHRes, batchPrefix, batchDimCount, {n, n}) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify failed, gradHRes must be %luD or %luD, but got %lu dims",
                                    batchDimCount + 1, batchDimCount + 2, gradHResDimNum);
        return ge::GRAPH_FAILED;
    }

    auto phi = GetInputShapeChecked(context_, opName, "phi", PHI_IDX);
    if (!phi) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "phi", phi, nullptr, 0, {hcMix, n * c}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto alpha = GetInputShapeChecked(context_, opName, "alpha", ALPHA_IDX);
    if (!alpha) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "alpha", alpha, nullptr, 0, {(int64_t)ALPHA_SIZE_3}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto bias = GetInputShapeChecked(context_, opName, "bias", BIAS_IDX);
    if (!bias) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "bias", bias, nullptr, 0, {hcMix}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto hPre = GetInputShapeChecked(context_, opName, "hPre", H_PRE_IDX);
    if (!hPre) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "hPre", hPre, batchPrefix, batchDimCount, {n}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto hcBeforeNorm = GetInputShapeChecked(context_, opName, "hcBeforeNorm", HC_BEFORE_NORM_IDX);
    if (!hcBeforeNorm) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "hcBeforeNorm", hcBeforeNorm, batchPrefix, batchDimCount, {hcMix}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto invRms = GetInputShapeChecked(context_, opName, "invRms", INV_RMS_IDX);
    if (!invRms) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "invRms", invRms, batchPrefix, batchDimCount, {(int64_t)1}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto sumOut = GetInputShapeChecked(context_, opName, "sumOut", SUM_OUT_IDX);
    if (!sumOut) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "sumOut", sumOut, batchPrefix, batchDimCount, {n}, 1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(sumOut->GetStorageShape().GetDim(0) % ITER_COUNT_DIVISOR != 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify sumOut failed, dim0 must be even, but got %ld",
                                            sumOut->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);

    auto normOut = GetInputShapeChecked(context_, opName, "normOut", NORM_OUT_IDX);
    if (!normOut) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "normOut", normOut, batchPrefix, batchDimCount, {n, n}, 1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(normOut->GetStorageShape().GetDim(0) != sumOut->GetStorageShape().GetDim(0),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ShapeVerify normOut dim0(%ld) must equal sumOut dim0(%ld)",
                                            normOut->GetStorageShape().GetDim(0), sumOut->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);

    auto gradX = GetOutputShapeChecked(context_, opName, "gradX", GRAD_X_IDX);
    if (!gradX) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "gradX", gradX, batchPrefix, batchDimCount, {n, c}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto gradPhi = GetOutputShapeChecked(context_, opName, "gradPhi", GRAD_PHI_IDX);
    if (!gradPhi) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "gradPhi", gradPhi, nullptr, 0, {hcMix, n * c}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto gradAlpha = GetOutputShapeChecked(context_, opName, "gradAlpha", GRAD_ALPHA_IDX);
    if (!gradAlpha) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "gradAlpha", gradAlpha, nullptr, 0, {(int64_t)ALPHA_SIZE_3}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto gradBias = GetOutputShapeChecked(context_, opName, "gradBias", GRAD_BIAS_IDX);
    if (!gradBias) {
        return ge::GRAPH_FAILED;
    }
    if (VerifyShape(opName, "gradBias", gradBias, nullptr, 0, {hcMix}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

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
