/* *
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling_base.h
 * \brief
 */
#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/op_tiling/mc2_tiling_struct.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../op_kernel/allto_allv_quant_grouped_mat_mul_tiling.h"
#include "../../op_kernel/allto_allv_quant_grouped_mat_mul_tiling_key.h"

namespace optiling {
// input
constexpr uint32_t GMM_X_INDEX = 0U;
constexpr uint32_t GMM_WEIGHT_INDEX = 1U;
constexpr uint32_t GMM_X_SCALE_INDEX = 2U;
constexpr uint32_t GMM_WEIGHT_SCALE_INDEX = 3U;
constexpr uint32_t SEND_COUNTS_TENSOR_INDEX = 4U;
constexpr uint32_t RECV_COUNTS_TENSOR_INDEX = 5U;
constexpr uint32_t MM_X_INDEX = 6U;
constexpr uint32_t MM_WEIGHT_INDEX = 7U;
constexpr uint32_t MM_X_SCALE_INDEX = 8U;
constexpr uint32_t MM_WEIGHT_SCALE_INDEX = 9U;
// output
constexpr uint32_t OUTPUT_GMM_Y_INDEX = 0U;
constexpr uint32_t OUTPUT_MM_Y_INDEX = 1U;
constexpr uint32_t OUTPUT_PERMUTE_OUT_INDEX = 2U;
// attr
constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2;
constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3;
constexpr uint32_t ATTR_GMM_X_QUANT_MODE_INDEX = 4;
constexpr uint32_t ATTR_GMM_WEIGHT_QUANT_MODE_INDEX = 5;
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 6;
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 7;
constexpr uint32_t ATTR_PERMUTE_OUT_FLAG_INDEX = 8;
constexpr uint32_t ATTR_MM_X_QUANT_MODE_INDEX = 9;
constexpr uint32_t ATTR_MM_WEIGHT_QUANT_MODE_INDEX = 10;
constexpr uint32_t ATTR_GROUP_SIZE_INDEX = 11;
constexpr uint32_t ATTR_Y_DTYPE = 12;
constexpr uint32_t ATTR_MM_DTYPE = 13;
constexpr uint32_t ATTR_COMM_MODE = 14;
// dim index
constexpr uint32_t DIM_ZERO = 0;
constexpr uint32_t DIM_ONE = 1;
constexpr uint32_t DIM_TWO = 2;
constexpr uint32_t DIM_THREE = 3;
constexpr uint32_t DIM_FOUR = 4;
// input data range
constexpr uint64_t BSK_MIN_VALUE = 0;
constexpr uint64_t BSK_MAX_VALUE = 52428800;
constexpr uint64_t H1_MIN_VALUE = 0;
constexpr uint64_t H1_MAX_VALUE = 65536;
constexpr uint64_t N1_MIN_VALUE = 0;
constexpr uint64_t N1_MAX_VALUE = 65536;
constexpr uint64_t H2_MIN_VALUE = 0;
constexpr uint64_t H2_MAX_VALUE = 12288;
constexpr uint64_t N2_MIN_VALUE = 0;
constexpr uint64_t N2_MAX_VALUE = 65536;
constexpr uint64_t K_MIN_VALUE = 2;
constexpr uint64_t K_MAX_VALUE = 8;
constexpr uint64_t BS_MIN_VALUE = 0;
constexpr uint64_t SEND_COUNTS_MIN_VALUE = 0;
constexpr uint64_t RECV_COUNTS_MIN_VALUE = 0;
constexpr uint64_t E_MIN_VALUE = 0;
constexpr uint64_t E_MAX_VALUE = 32;
constexpr uint64_t EXPERT_MIN_VALUE = 0;
constexpr uint64_t EXPERT_MAX_VALUE = 256;
// 910_93 comm size range
constexpr uint64_t COMM_MIN_SIZE = 2 * 1024 * 1024;
constexpr uint32_t COMM_MAX_SIZE = 100 * 1024 * 1024;
// cube compute
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
// quant mode
constexpr uint64_t NO_QUANT_MODE = 0;
constexpr uint64_t PERTENSOR_QUANT_MODE = 1;
constexpr uint64_t MX_QUANT_MODE = 6;
constexpr uint32_t RANK_DIM_BOUNDARY = 8;

class AlltoAllvQuantGmmTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AlltoAllvQuantGmmTilingBase(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

protected:
    ge::graphStatus GetCommonPlatformInfo();
    ge::graphStatus CheckCommonPlatformInfo();
    ge::graphStatus GetCommonShapeAttrsInfo();
    ge::graphStatus CheckCommonShapeAttrsInfo();
    ge::graphStatus GetShapeInfo();

    // platform info
    platform_ascendc::SocVersion socVersion_;
    uint32_t aicCoreNum_;
    uint32_t aivCoreNum_;
    uint64_t ubSize_;
    uint64_t l1Size_;
    uint64_t l0aSize_;
    uint64_t l0bSize_;
    uint64_t l0cSize_;
    uint64_t libApiWorkSpaceSize_;
    // attr ptr
    const char *groupPtr_;
    const int64_t *epWorldSizePtr_;
    const gert::ContinuousVector *sendCountsPtr_;
    const gert::ContinuousVector *recvCountsPtr_;
    const bool *transGmmWeightPtr_;
    const bool *transMmWeightPtr_;
    const bool *permuteOutFlagPtr_;
    const int64_t *gmmXQuantModePtr_;
    const int64_t *gmmWeightQuantModePtr_;
    const int64_t *mmXQuantModePtr_;
    const int64_t *mmWeightQuantModePtr_;
    // shape
    uint64_t bsk_ = 0;
    uint64_t h1_ = 0;
    uint64_t e_ = 0;
    uint64_t n1_ = 0;
    uint64_t a_ = 0;
    uint64_t bs_ = 0;
    uint64_t h2_ = 0;
    uint64_t n2_ = 0;
    uint64_t k_ = 0;
    // attr
    const char *group_;
    uint64_t epWorldSize_ = 0;
    bool transGmmWeight_ = false;
    bool transMmWeight_ = false;
    bool permuteOutFlag_ = false;
    const int64_t *sendCounts = nullptr;
    const int64_t *recvCounts = nullptr;
    // data type
    ge::DataType gmmXDataType_;
    ge::DataType gmmWeightDataType_;
    ge::DataType mmXDataType_;
    ge::DataType mmWeightDataType_;
    // flag
    bool hasSharedExpertFlag_ = false;

private:
    // get
    ge::graphStatus GetAttrsInfo();
    ge::graphStatus GetGmmXShapeInfo();
    ge::graphStatus GetGmmWeightShapeInfo();
    ge::graphStatus GetCountsTensorShapeInfo() const;
    ge::graphStatus GetMmxShapeInfo();
    ge::graphStatus GetMmWeightShapeInfo();
    ge::graphStatus GetGmmYShapeInfo();
    ge::graphStatus GetMmYShapeInfo();
    ge::graphStatus GetPermuteOutShapeInfo() const;
    // check
    ge::graphStatus CheckAttrsInfo();
    ge::graphStatus CheckShapeInfo();
    ge::graphStatus CheckGmmXShapeInfo() const;
    ge::graphStatus CheckGmmWeightShapeInfo() const;
    ge::graphStatus CheckCountsTensorShapeInfo();
    ge::graphStatus CheckMmxShapeInfo();
    ge::graphStatus CheckMmWeightShapeInfo() const;
    ge::graphStatus CheckGmmYShapeInfo() const;
    ge::graphStatus CheckMmYShapeInfo();
    ge::graphStatus CheckPermuteOutShapeInfo() const;
    ge::graphStatus CheckCommSize();
    ge::graphStatus CheckEpWorldSizeValue();
    ge::graphStatus CheckCommCountsRange();
    ge::graphStatus CheckCommCountsValue();
    ge::graphStatus CheckFormat();
};
} // namespace optiling
