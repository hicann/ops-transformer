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
 * \file grouped_mat_mul_allto_allv_mte_tiling.h
 */
#ifndef MC2_GROUPED_MATMUL_ALLTO_ALLV_MTE_TILING_STRUCT_H
#define MC2_GROUPED_MATMUL_ALLTO_ALLV_MTE_TILING_STRUCT_H

#include <array>

#include "../../../../quant_grouped_mat_mul_allto_allv/op_host/op_tiling/quant_grouped_mat_mul_allto_allv_tiling_common.h"

namespace optiling {
class GroupedMatmulAllToAllvMteTiling : public Mc2Tiling::Mc2GroupedMatmul::QuantGroupedMatmulAllToAllvTilingCommon {
public:
    explicit GroupedMatmulAllToAllvMteTiling(gert::TilingContext *context)
        : Mc2Tiling::Mc2GroupedMatmul::QuantGroupedMatmulAllToAllvTilingCommon(context) {};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus CheckOpInputSingleParamsTensorNotSupport() override;
    ge::graphStatus CheckOpInputSingleParamsTensorSupport() override;
    ge::graphStatus CheckOpInputSingleParamsTensorMM() override;
    ge::graphStatus CheckAndSetLocalParamsGmm() override;
    ge::graphStatus CheckAndSetLocalParamsMm() override;
    ge::graphStatus CheckAndSetLocalParamsAttr() override;
    ge::graphStatus CheckAndSetLocalParams() override;
    ge::graphStatus CheckFormat() override;
    ge::graphStatus CheckParamsRelationGmm() override;
    ge::graphStatus CheckParamsRelationMm() override;
    ge::graphStatus CheckParamsAttrEpAndSetLocalParams() override;
    ge::graphStatus CheckAndSetSendRecvCountsAttr() override;
    ge::graphStatus CheckLocalParams() override;
    ge::graphStatus CheckTopK(uint64_t topK) override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus SetMteWorkspaceInfo();
    ge::graphStatus CheckExpertPipeline() const;
    ge::graphStatus SetExpertChunkRows(uint32_t &expertChunkRows) const;
    uint64_t GetTilingKey() const override;
    uint32_t GetCommModeIndex() const override;
    ge::graphStatus GetAndConvertCommMode(gert::TilingContext *context, uint8_t &commMode) const;

    static constexpr uint32_t GMM_X_INDEX = 0;
    static constexpr uint32_t GMM_WEIGHT_INDEX = 1;
    static constexpr uint32_t SEND_COUNTS_TENSOR_OPTIONAL_INDEX = 2;
    static constexpr uint32_t RECV_COUNTS_TENSOR_OPTIONAL_INDEX = 3;
    static constexpr uint32_t MM_X_OPTIONAL_INDEX = 4;
    static constexpr uint32_t MM_WEIGHT_OPTIONAL_INDEX = 5;
    static constexpr uint32_t OUTPUT_Y_INDEX = 0;
    static constexpr uint32_t OUTPUT_MM_Y_OPTIONAL_INDEX = 1;
    static constexpr uint32_t ATTR_GROUP_INDEX = 0;
    static constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
    static constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2;
    static constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3;
    static constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
    static constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5;
    static constexpr uint32_t ATTR_COMM_MODE = 6;
    static constexpr int64_t RANK_DIM_BOUNDARY = 8;
    static constexpr uint64_t MAX_GLOBAL_EXPERT_NUM = 1024UL;
    static constexpr uint64_t MAX_TOKEN_NUM = 5000000UL;

    static const std::vector<uint32_t> GMM_X_DTYPE_LIST;
    static const std::vector<uint32_t> GMM_WEIGHT_DTYPE_LIST;
    static const std::vector<uint32_t> GMM_Y_DTYPE_LIST;
    static const std::set<int64_t> A5_SUPPORT_RANK_SIZE;
    static const std::set<int64_t> A2_SUPPORT_RANK_SIZE;
    static const std::set<int64_t> A3_SUPPORT_RANK_SIZE;

    uint64_t mteWorkspaceSize_{0UL};
    std::array<int32_t, MAX_GLOBAL_EXPERT_NUM> sendCounts_{};
    std::array<int32_t, MAX_GLOBAL_EXPERT_NUM> recvCounts_{};
    bool isA3_{false};
};
} // namespace optiling
#endif
