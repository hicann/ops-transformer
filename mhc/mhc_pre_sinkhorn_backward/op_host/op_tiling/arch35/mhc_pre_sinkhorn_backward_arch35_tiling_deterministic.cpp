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
#include "mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"

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
constexpr int64_t MAX_N = 8;
constexpr int64_t EXPECTED_SK_ITER_COUNT = 20;
constexpr int64_t MAX_C_VALUE = 100000;
constexpr int64_t C_ALIGNMENT = 128;
constexpr int64_t ITER_COUNT_DIVISOR = 2;
constexpr int64_t ALIGN_32 = 32;
constexpr int64_t FP32_BYTE_SIZE = 4;
constexpr uint64_t UB_RESERVE_SIZE = 256;
constexpr int64_t WS_BUFFER_INTERVAL = 1024;
constexpr int64_t GRAD_X_FROM_BUF_COUNT = 3;
constexpr int64_t GRAD_ALPHA_BUF_COUNT = 3;
constexpr int64_t N_SQUARE_GRAD_BUF_COUNT = 5;
constexpr int64_t N_SQUARE_FINAL_BUF_COUNT = 4;
constexpr int64_t N_SQUARE_TAIL_BUF_COUNT = 4;

template <typename T>
constexpr T AlignUp(T value, T align)
{
    return (value + align - 1) / align * align;
}

using namespace ge;
using namespace std;
using namespace AscendC;

bool MhcPreSinkhornBackwardArch35DeterminiticTiling::IsCapable()
{
    return true;
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName, "MhcPreSinkhornBackwardArch35DeterministicTiling GetShapeAttrsInfo");
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
    auto xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(xDimNum != 3 && xDimNum != 4,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "xShape verify failed, x must be 3D or 4D, but got %lu dims", xDimNum),
                return ge::GRAPH_FAILED);
    auto skSumShape = skSumPtr->GetStorageShape();

    auto attrsPtr = context_->GetAttrs();
    if (attrsPtr == nullptr) {
        OP_LOGE(context_, "attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto epsPtr = attrsPtr->GetAttrPointer<float>(0);
    hcEps_ = (epsPtr != nullptr) ? static_cast<float>(*epsPtr) : DEFAULT_EPS;

    if (xDimNum == 3) {
        bs_ = xShape.GetDim(0);
        n_ = xShape.GetDim(1);
        c_ = xShape.GetDim(2);
        OP_CHECK_IF(CheckShapeBase(bs_, 1, n_, c_, true) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "CheckShape3D failed"),
                    return ge::GRAPH_FAILED);
    } else {
        int64_t batchSize = xShape.GetDim(BATCH_SIZE_DIM_IDX);
        int64_t seqLength = xShape.GetDim(SEQ_LENGTH_DIM_IDX);
        bs_ = batchSize * seqLength;
        n_ = xShape.GetDim(N_DIM_IDX);
        c_ = xShape.GetDim(C_DIM_IDX);
        OP_CHECK_IF(CheckShapeBase(batchSize, seqLength, n_, c_) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "CheckShape failed"), return ge::GRAPH_FAILED);
    }
    skIterCount_ = skSumShape.GetDim(ITER_COUNT_IDX) / ITER_COUNT_DIVISOR;

    OP_CHECK_IF(
        n_ <= 0 || n_ > MAX_N,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "n must be > 0 and <= %ld, but got %ld", MAX_N, n_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(skIterCount_ != EXPECTED_SK_ITER_COUNT,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "sk_iter_count must be %ld, but got %ld",
                                            EXPECTED_SK_ITER_COUNT, skIterCount_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        c_ <= 0 || c_ >= MAX_C_VALUE || c_ % C_ALIGNMENT != 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "c must be > 0, < %ld and divisible by %ld, but got %ld",
                                    MAX_C_VALUE, C_ALIGNMENT, c_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::SetTilingData()
{
    MhcPreSinkhornBackwardArch35DeterminiticTilingData *tilingData =
        context_->GetTilingData<MhcPreSinkhornBackwardArch35DeterminiticTilingData>();
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);

    auto floatDataType = matmul_tiling::DataType::DT_FLOAT;

    int64_t hcMix = n_ * n_ + 2 * n_;
    mm1K_ = hcMix;
    mm1M_ = bs_;
    mm1N_ = n_ * c_;

    mm2K_ = bs_;
    mm2M_ = hcMix;
    mm2N_ = n_ * c_;

    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling mm2Tiling(ascendPlatformInfo);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
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

    tilingData->bs = bs_;
    tilingData->c = c_;
    tilingData->n = n_;
    tilingData->usedAivNum = usedAivNum_;
    tilingData->skIterCount = skIterCount_;
    tilingData->cLoopDataLen = cLoopDataLen_;
    tilingData->bsLoopDataLen = bsLoopDataLen_;
    tilingData->bsTaskCount = bsTaskCount_;
    tilingData->tailBsTaskCount = tailBsTaskCount_;
    tilingData->cBlockLoops = cBlockLoops_;
    tilingData->cBlockTailLoops = cBlockTailLoops_;
    tilingData->cBlockCount = cBlockCount_;
    tilingData->cBlockTailCount = cBlockTailCount_;
    tilingData->cBlockTailTailCount = cBlockTailTailCount_;
    tilingData->finalUsedAivNum = finalUsedAivNum_;
    tilingData->eps = hcEps_;
    tilingData->aicNum = coreNumAic_;

    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35DeterminiticTiling::DoUbTiling()
{
    int64_t totalBs = bs_;
    bsTaskCount_ = Ops::Base::CeilDiv(totalBs, static_cast<int64_t>(coreNumAiv_));
    usedAivNum_ = Ops::Base::CeilDiv(totalBs, bsTaskCount_);
    tailBsTaskCount_ = totalBs - bsTaskCount_ * (usedAivNum_ - 1);
    cLoopDataLen_ = c_;
    bsLoopDataLen_ = bsTaskCount_;
    ubBlock_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    ubSize_ = ubSize_ - UB_RESERVE_SIZE;
    auto restSize = static_cast<int64_t>(-1);
    int64_t hcMix = n_ * n_ + 2 * n_;

    /* UB分布：x、h_pre And grad_h_pre、grad_h_in、hc_before_norm、inv_rms、
               grad_h_res、xFP32、grad_x_from_in、grad_alpha And grad_bias、
               grad_h_pre、grad_z、bias、norm_out_forward、z、gradCalcBuf1_、
               gradCalcBuf2_、tmpBuf、sumOutBuf、normOutBuf_、idxBuf~maxBuf
    */
    while (restSize <= 0) {
        int64_t occupy =
            DOUBLE_BUFFER * bsLoopDataLen_ * n_ * Ops::Base::CeilAlign(cLoopDataLen_ * BF16_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * Ops::Base::CeilAlign(bsLoopDataLen_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * bsLoopDataLen_ * Ops::Base::CeilAlign(cLoopDataLen_ * BF16_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * bsLoopDataLen_ * Ops::Base::CeilAlign(n_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * Ops::Base::CeilAlign(bsLoopDataLen_ * FP32_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * bsLoopDataLen_ * Ops::Base::CeilAlign(n_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * bsLoopDataLen_ * n_ * Ops::Base::CeilAlign(cLoopDataLen_ * FP32_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * bsLoopDataLen_ * n_ * Ops::Base::CeilAlign(cLoopDataLen_ * FP32_BYTE_SIZE, ubBlock_) +
            Ops::Base::CeilAlign(n_ * n_ * FP32_BYTE_SIZE, ubBlock_) + ubBlock_ +
            Ops::Base::CeilAlign(bsLoopDataLen_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            Ops::Base::CeilAlign(bsLoopDataLen_ * n_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            Ops::Base::CeilAlign(hcMix * FP32_BYTE_SIZE, ubBlock_) +
            Ops::Base::CeilAlign(bsLoopDataLen_ * n_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            Ops::Base::CeilAlign(bsLoopDataLen_ * n_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            Ops::Base::CeilAlign(hcMix * FP32_BYTE_SIZE, ubBlock_) +
            Ops::Base::CeilAlign(hcMix * FP32_BYTE_SIZE, ubBlock_) +
            N_SQUARE_GRAD_BUF_COUNT * Ops::Base::CeilAlign(bsLoopDataLen_ * n_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * ITER_COUNT_DIVISOR * skIterCount_ * bsLoopDataLen_ *
                Ops::Base::CeilAlign(n_ * FP32_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * skIterCount_ * bsLoopDataLen_ * Ops::Base::CeilAlign(n_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            N_SQUARE_TAIL_BUF_COUNT * bsLoopDataLen_ * Ops::Base::CeilAlign(n_ * n_ * FP32_BYTE_SIZE, ubBlock_);

        restSize = ubSize_ - occupy;
        if (restSize <= 0 && bsLoopDataLen_ > 1) {
            bsLoopDataLen_--;
        } else if (restSize <= 0 && bsLoopDataLen_ == 1) {
            cLoopDataLen_--;
        }
    }

    // final
    auto bsncTaskCount = Ops::Base::CeilDiv(totalBs * n_ * c_, static_cast<int64_t>(coreNumAiv_));
    finalUsedAivNum_ = Ops::Base::CeilDiv(totalBs * n_ * c_, bsncTaskCount);
    auto tailBsncTaskCount_ = totalBs * n_ * c_ - bsncTaskCount * (finalUsedAivNum_ - 1);
    cBlockCount_ = bsncTaskCount;
    cBlockTailCount_ = tailBsncTaskCount_;
    restSize = static_cast<int64_t>(-1);
    while (restSize <= 0) {
        int64_t occupy =
            N_SQUARE_FINAL_BUF_COUNT * DOUBLE_BUFFER * Ops::Base::CeilAlign(cBlockCount_ * FP32_BYTE_SIZE, ubBlock_) +
            DOUBLE_BUFFER * Ops::Base::CeilAlign(usedAivNum_ * FP32_BYTE_SIZE, ubBlock_) + ubBlock_ +
            usedAivNum_ * Ops::Base::CeilAlign(n_ * n_ * FP32_BYTE_SIZE, ubBlock_) +
            ITER_COUNT_DIVISOR * Ops::Base::CeilAlign(hcMix * FP32_BYTE_SIZE, ubBlock_) +
            ITER_COUNT_DIVISOR * DOUBLE_BUFFER * Ops::Base::CeilAlign(cLoopDataLen_ * FP32_BYTE_SIZE, ubBlock_);
        restSize = ubSize_ - occupy;
        if (restSize <= 0 && cBlockCount_ > 1) {
            cBlockCount_--;
        }
    }
    cBlockLoops_ = Ops::Base::CeilDiv(bsncTaskCount, cBlockCount_);
    cBlockTailLoops_ = Ops::Base::CeilDiv(tailBsncTaskCount_, cBlockCount_);
    cBlockTailCount_ = bsncTaskCount - cBlockCount_ * (cBlockLoops_ - 1);
    cBlockTailTailCount_ = tailBsncTaskCount_ - cBlockCount_ * (cBlockTailLoops_ - 1);
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::DoOpTiling()
{
    DoUbTiling();
    return SetTilingData();
}

uint64_t MhcPreSinkhornBackwardArch35DeterminiticTiling::GetTilingKey() const
{
    bool isDeterministic = true;
    return GET_TPL_TILING_KEY(isDeterministic);
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::PostTiling()
{
    context_->SetBlockDim(coreNumAic_);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t hcMix = n_ * n_ + 2 * n_;
    size_t userWorkspaceSize = bs_ * n_ * c_ * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL +                      // xFP32
                               GRAD_ALPHA_BUF_COUNT * usedAivNum_ * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL + // gradAlpha
                               usedAivNum_ * hcMix * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL +                // gradBias
                               bs_ * n_ * c_ * FP32_BYTE_SIZE * GRAD_X_FROM_BUF_COUNT +
                               GRAD_X_FROM_BUF_COUNT * WS_BUFFER_INTERVAL +        // gradXFrom
                               bs_ * hcMix * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL + // grad_hc_before_norm
                               bs_ * hcMix * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL;  // grad_norm_out

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(opName, "fail to GetWorkspaceSizes"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = systemWorkspaceSize + userWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35DeterminiticTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "bs: " << bs_ << std::endl;
    info << "c: " << c_ << std::endl;
    info << "n: " << n_ << std::endl;
    info << "aivNum: " << coreNumAiv_ << std::endl;
    info << "usedAivNum: " << usedAivNum_ << std::endl;
    info << "cLoopDataLen: " << cLoopDataLen_ << std::endl;
    info << "bsLoopDataLen: " << bsLoopDataLen_ << std::endl;
    info << "skIterCount: " << skIterCount_ << std::endl;
    info << "bsTaskCount: " << bsTaskCount_ << std::endl;
    info << "tailBsTaskCount: " << tailBsTaskCount_ << std::endl;
    info << "cBlockLoops: " << cBlockLoops_ << std::endl;
    info << "cBlockTailLoops: " << cBlockTailLoops_ << std::endl;
    info << "cBlockCount: " << cBlockCount_ << std::endl;
    info << "cBlockTailCount: " << cBlockTailCount_ << std::endl;
    info << "cBlockTailTailCount: " << cBlockTailTailCount_ << std::endl;
    info << "finalUsedAivNum: " << finalUsedAivNum_ << std::endl;
    info << "mm1K: " << mm1K_ << std::endl;
    info << "mm1M: " << mm1M_ << std::endl;
    info << "mm1N: " << mm1N_ << std::endl;
    info << "mm2K: " << mm2K_ << std::endl;
    info << "mm2M: " << mm2M_ << std::endl;
    info << "mm2N: " << mm2N_ << std::endl;

    OP_LOGI(opName, "Tiling info is: %s", info.str().c_str());
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MhcPreSinkhornBackward, MhcPreSinkhornBackwardArch35DeterminiticTiling, 0);
} // namespace optiling
