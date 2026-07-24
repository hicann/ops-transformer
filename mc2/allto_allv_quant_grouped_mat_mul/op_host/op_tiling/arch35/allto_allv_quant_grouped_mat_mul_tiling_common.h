/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling_common.h
 * \brief
 */
#ifndef ALLTO_ALLV_QUANT_GROUPED_MATMUL_TILING_COMMON_H
#define ALLTO_ALLV_QUANT_GROUPED_MATMUL_TILING_COMMON_H

#include <string>
#include <numeric>
#include <climits>
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "context_util.h"
#include "op_host/op_tiling/hccl_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/allto_allv_quant_grouped_mat_mul_tiling.h"
#include "../allto_allv_quant_grouped_mat_mul_tiling_base.h"

namespace optiling {
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t SINGLE_GROUP_NUM = 1;
constexpr uint32_t GMM_ACT_TYPE_NONE = 0;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t MTE2_MIN_LOAD_SIZE = 64 * 1024UL;
// pertensor
constexpr uint32_t PERTENSOR_MODE = 1;
// pergroup
constexpr uint32_t MX_MODE = 6;
constexpr uint64_t MX_BASIC_FACTOR = 64;
constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr uint64_t MX_GROUP_SIZE_K = 32;
constexpr uint64_t MX_GROUP_SIZE_M = 1;
constexpr uint64_t MX_GROUP_SIZE_N = 1;
constexpr uint64_t MXFP4_PACK_FACTOR = 2;
constexpr uint32_t SCALE_BATCH_THRESHOLD = 32;
// quant mode offset
const std::vector<uint32_t> QUANT_MODE_MAP = {0, 0, 1, 2, 4, 5, 3};

constexpr uint32_t GMM_ARRAY_MAX_NUM = 128U;
constexpr uint32_t MAX_HANDLE_ID_NUM = 64U;
constexpr uint32_t DEFAULT_MERGED_EXPERT_NUM = 4U;
constexpr uint64_t PER_RANK_TOTAL_MN_THRESHOLD = 20UL * 1024UL * 1024UL;
constexpr uint32_t SMALL_EXPERT_THRESHOLD = 4U;
constexpr uint32_t ORIGINAL_LOOP_THRESHOLD = 4U;
constexpr uint32_t MIN_LOOP_COUNT = 2U;

class AlltoAllvQuantGmmTilingCommon : public AlltoAllvQuantGmmTilingBase {
public:
    explicit AlltoAllvQuantGmmTilingCommon(gert::TilingContext *context) : AlltoAllvQuantGmmTilingBase(context)
    {
        tilingData = context_->GetTilingData<QuantAlltoAllvGroupedMatmulTilingData>();
    };
    QuantAlltoAllvGroupedMatmulTilingData *tilingData;

    static uint32_t CalcExpertNum(uint64_t e, uint64_t epWorldSize, uint64_t bsk, uint64_t n1,
                                  uint32_t packFactor = 1U)
    {
        if (e == 0 || e == 1) {
            return 1U;
        }
        if (e <= SMALL_EXPERT_THRESHOLD) {
            return 1U;
        }
        uint32_t originalLoopCount = static_cast<uint32_t>((e + DEFAULT_MERGED_EXPERT_NUM - 1U)
            / DEFAULT_MERGED_EXPERT_NUM);
        if (originalLoopCount <= ORIGINAL_LOOP_THRESHOLD) {
            return DEFAULT_MERGED_EXPERT_NUM;
        }

        uint64_t perRankTokens = (epWorldSize == 0) ? bsk : bsk / epWorldSize;
        uint64_t perRankTotalMN = (packFactor == 0) ? perRankTokens * n1
            : perRankTokens * n1 / packFactor;

        uint32_t expertNum = DEFAULT_MERGED_EXPERT_NUM;
        if (perRankTotalMN < PER_RANK_TOTAL_MN_THRESHOLD) {
            uint32_t upperByArray = (epWorldSize == 0) ? GMM_ARRAY_MAX_NUM
                : static_cast<uint32_t>(GMM_ARRAY_MAX_NUM / epWorldSize);
            uint32_t upperByLoop = static_cast<uint32_t>(e / MIN_LOOP_COUNT);
            uint32_t upperBound = std::min({static_cast<uint32_t>(e), upperByArray, upperByLoop});
            expertNum = std::max(upperBound, 1U);
        }
        return expertNum;
    }

protected:
    // Tiling base
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    // quant base
    virtual ge::graphStatus CheckInputDtype() const
    {
        return ge::GRAPH_SUCCESS;
    };
    virtual ge::graphStatus CheckScaleFormatAndDtype() const
    {
        return ge::GRAPH_SUCCESS;
    };
    virtual ge::graphStatus CheckQuantMode() const
    {
        return ge::GRAPH_SUCCESS;
    };
    virtual ge::graphStatus CheckScaleShape() const
    {
        return ge::GRAPH_SUCCESS;
    };
    virtual ge::graphStatus DoGmmTiling(uint64_t gmmMSize)
    {
        return ge::GRAPH_SUCCESS;
    };
    virtual void GetPermuteOutSize() {};
    ge::graphStatus CheckInputNotNull() const;
    ge::graphStatus SetHcclTiling() const;
    ge::graphStatus QuantGetAndConvertCommMode(gert::TilingContext *context, uint8_t &commMode) const;
    void PrintGMMQuantTilingData(const Mc2GroupedMatmulTilingData::GMMQuantTilingData &data) const;
    void PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo &taskTilingInfo) const;
    uint64_t permuteScaleOutSize_{0};
    uint64_t permuteOutSize_{0};
};
} // namespace optiling
#endif // ALLTO_ALLV_QUANT_GROUPED_MATMUL_TILING_COMMON_H
