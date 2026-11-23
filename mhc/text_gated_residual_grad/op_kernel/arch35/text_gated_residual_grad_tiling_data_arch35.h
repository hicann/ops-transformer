/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file text_gated_residual_grad_tiling_data_arch35.h
 * \brief TextGatedResidualGrad tiling data shared between host and device.
 */

#ifndef TEXT_GATED_RESIDUAL_GRAD_TILING_DATA_ARCH35_H
#define TEXT_GATED_RESIDUAL_GRAD_TILING_DATA_ARCH35_H

#include <cstdint>
#include "../arch22/nn_base/tgrg_batch_mat_mul_v3_tiling_data.h"

constexpr uint32_t HC_GRAD_A5_TILING_BYTES = 100U;

// GEMM parameters shared by the local A5 host tiler and Cube kernels.
struct HcGradA5NnTiling {
    uint32_t usedCoreNum, m, n, k, mL1, nL1, kL1, baseM, baseN, baseK, skSingleCoreK;
    uint32_t mTailCnt, nTailCnt, mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain;
    uint8_t mmadParam;
    uint8_t l1BufferNum, l0cDB, ubDB;
    uint32_t l2CacheDisable, sliceM, srcNdStride, rowStride, innerBatch;
    uint32_t splitSingleCoreKBase;
    uint8_t splitM, aswWindowLen;
    uint8_t reserved[2];
};
static_assert(sizeof(HcGradA5NnTiling) == HC_GRAD_A5_TILING_BYTES, "A5 host/device GEMM tiling size");

class TextGatedResidualGradTilingDataArch35 {
public:
    int64_t taskRows{1024};     // Runtime token batch shared by host, Cube, Vector and workspace.
    int64_t fullBatchRows{0};   // A2 retained chunk size; zero in bounded token-ring modes.
    int64_t recomputeDirect{0}; // A2 full-batch V6 rebuilds direct from grad_mixed/up.
    int64_t lowpDx{0};          // Preserve the tiling-data ABI. The host requests FP32 intermediate DX.
    int64_t nnMatmul{0}; // A5 NN backend; disables direct-output shortcuts so every GEMM follows its planned shape.
    int64_t n{0};
    int64_t c{0};
    int64_t h{0};
    int64_t r{0};
    int64_t d{0};
    int64_t taskCount{0};
    int64_t workspaceSlots{0};
    int64_t splitkFactor{0};
    int64_t hasGamma{0};
    int64_t hasGate{0};
    int64_t aicNum{0};
    int64_t aivNum{0};
    int64_t scheduleBeatCount{0};
    int64_t workspaceSize{0};
    int64_t vectorUbBytes{0};  // Detected platform UB capacity in bytes.
    int64_t pipeStages{0};     // 0 = serial blocks; 2 = input/output ping-pong stages.
    int64_t vectorRowBlock{1}; // V1/V6 rows per block, computed by the host UB budget.
    // A2/A3 MatMulV3 base tiling: full/tail task x C1a,C1b,G56,DX,gate.
    MatmulTilingData nnBase[2][5]{};
    uint32_t nnBaseMask[2]{};
    uint32_t nnMixMask[2]{};
    // C1b and G56 accept the native batch=1 ND plan when the regular GEMM chooses Split-K.
    BatchMatmulTilingData nnWeightBatch[2]{};
    uint32_t nnWeightBatchMask{0};
    HcGradA5NnTiling nnA5[2][5]{};
    uint64_t nnA5Key[2][5]{};
    uint64_t nnScratchOffset{0}, nnScratchBytes{0}, nnUbBackupOffset{0};
    int64_t scheduleMode{0}; // 0 = SyncAll each step; 1 = bounded producer-consumer ready/free credits.
};

#endif // TEXT_GATED_RESIDUAL_GRAD_TILING_DATA_ARCH35_H
