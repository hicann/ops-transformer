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
 * \file masked_causal_conv1d_backward_arch35.h
 * \brief MaskedCausalConv1dBackward tiling implementation
 */

#ifndef MASKED_CAUSAL_CONV1D_BACKWARD_TILING_ARCH35_H
#define MASKED_CAUSAL_CONV1D_BACKWARD_TILING_ARCH35_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "../op_kernel/arch35/masked_causal_conv1d_backward_struct.h"

namespace optiling {

struct MaskedCausalConv1dBackwardArch35CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

constexpr uint64_t TILING_KEY_MASKED_CAUSAL_CONV1D_BACKWARD_BF16 = 10000;
constexpr uint64_t TILING_KEY_MASKED_CAUSAL_CONV1D_BACKWARD_FP16 = 10001;

// Input tensor indices
constexpr int32_t GRAD_Y_INDEX = 0;
constexpr int32_t X_INDEX = 1;
constexpr int32_t WEIGHT_INDEX = 2;
constexpr int32_t MASK_INDEX = 3; // optional

// Constants for validation and tiling
constexpr int64_t DIM_ALIGN_ELEMENT = 64;             // H split granularity (elements)
constexpr int64_t ALIGN_BYTES = 32;                   // Base alignment requirement
constexpr int64_t DTYPE_SIZE = 2;                     // bf16/fp16 size in bytes
constexpr int64_t BUFFER_NUM = 2;                     // Double buffering recommended
constexpr int64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024; // Reserve for system usage
constexpr int64_t B_MAX = 32;
constexpr int64_t BS_MAX = 512 * 1024;
constexpr int64_t H_MIN = 192 * 2;
constexpr int64_t H_MAX = 192 * 128;

// Shape dim indices
constexpr int64_t DIM_0 = 0; // S or W
constexpr int64_t DIM_1 = 1; // B or H
constexpr int64_t DIM_2 = 2; // H
constexpr int64_t NUM_2 = 2;
constexpr int64_t NUM_3 = 3;

class MaskedCausalConv1dBackwardTilingArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MaskedCausalConv1dBackwardTilingArch35(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    ge::graphStatus CheckInputParams();
    ge::graphStatus ValidateGradYShape();
    ge::graphStatus ValidateXShape();
    ge::graphStatus ValidateWeightShape();
    ge::graphStatus ValidateMaskShape();

    ge::graphStatus ValidateGradYType();
    ge::graphStatus ValidateXType();
    ge::graphStatus ValidateWeightType();
    ge::graphStatus ValidateMaskType();

private:
    ge::graphStatus ComputeInterCoreSplit();    // 核间切分
    ge::graphStatus ComputeIntraCoreUbTiling(); // 核内切分

    // Hardware information
    uint64_t ubSize_ = 0;       // UB size per core (bytes)
    uint64_t totalCoreNum_ = 0; // Total AIV cores

    // Input tensor shape information
    int64_t S_ = 0; // seq length
    int64_t B_ = 0; // batch size
    int64_t H_ = 0; // hidden size
    int64_t W_ = 0; // kernel width (must be 3)

    // Data type information
    ge::DataType dataType_{}; // f16/bf16
    size_t dtypeSize_ = DTYPE_SIZE;
    int64_t hasMask_ = 0; // 1 if mask provided

    // Inter-core tiling parameters (H split by 64)
    int64_t hMainCoreCnt_ = 0; // big cores count
    int64_t hTailCoreCnt_ = 0; // small cores count
    int64_t hMainSize_ = 0;    // elements per big core (multiple of 64)
    int64_t hTailSize_ = 0;    // elements per small core (multiple of 64)
    int64_t usedCoreNum_ = 0;  // total used core number

    // Intra-core tiling parameters (UB loop for H/B/S)
    int64_t hUB_ = DIM_ALIGN_ELEMENT; // start with 64
    int64_t sUB_ = 1;

    int64_t hLoopCnt_ = 0;
    int64_t sLoopCnt_ = 0;

    int64_t hUBTail_ = 0;
    int64_t sUBTail_ = 0;
    int64_t allUb_ = 0;

    // 主核UB切块参数（有效输出大小，不包含重叠）
    int64_t ubMainFactorH_ = 0;
    int64_t ubTailFactorH_ = 0;
    int64_t ubMainFactorS_ = 0; // 有效输出行数，不包含重叠部分
    int64_t ubTailFactorS_ = 0; // 有效输出行数，不包含重叠部分
    int64_t ubMainFactorSCount_ = 0;
    int64_t ubTailFactorSCount_ = 0;

    // 尾核参数（有效输出大小，不包含重叠）
    int64_t tailHLoopCnt_ = 0;
    int64_t tailSLoopCnt_ = 0;

    int64_t tailCoreUbMainFactorH_ = 0;
    int64_t tailCoreUbTailFactorH_ = 0;
    int64_t tailCoreUbMainFactorS_ = 0; // 有效输出行数，不包含重叠部分
    int64_t tailCoreUbTailFactorS_ = 0; // 有效输出行数，不包含重叠部分

    MaskedCausalConv1dBackwardArch35Tiling::MaskedCausalConv1dBackwardTilingDataV35 tilingData_{};
};

} // namespace optiling

#endif // MASKED_CAUSAL_CONV1D_BACKWARD_TILING_ARCH35_H
