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
 * \file grouped_quant_basic_api_matmul_tiling.h
 * \brief
 */
#ifndef GROUPED_QUANT_BASIC_API_MATMUL_TILING_H
#define GROUPED_QUANT_BASIC_API_MATMUL_TILING_H

#include "grouped_quant_matmul_tiling.h"
#include "../grouped_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "op_host/tiling_base.h"

namespace optiling {

class GroupedQmmBasicApiTiling : public GroupedQmmTiling {
public:
    explicit GroupedQmmBasicApiTiling(gert::TilingContext *context);
    ~GroupedQmmBasicApiTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    virtual void Reset();
    ge::graphStatus CalL1Tiling();

private:
    struct L1SearchResult {
        uint64_t depthA1 = 0UL;
        uint64_t depthB1 = 0UL;
        uint64_t stepKa = 0UL;
        uint64_t stepKb = 0UL;
        uint32_t scaleFactorA = 0U;
        uint32_t scaleFactorB = 0U;
        uint64_t highBwPayload = 0UL;
        uint64_t totalPayload = 0UL;
        bool found = false;
    };

    ge::graphStatus CalL1Depth(uint64_t leftL1Size);
    bool SearchReducedThreeBufferTiling(uint64_t initialDepthA1, uint64_t initialDepthB1, uint64_t baseASize,
                                        uint64_t baseBSize, L1SearchResult &result);
    void SearchExpandedThreeBufferTiling(uint64_t initialDepthA1, uint64_t initialDepthB1, uint64_t maxDepthA1,
                                         uint64_t maxDepthB1, uint64_t baseASize, uint64_t baseBSize,
                                         L1SearchResult &result);
    void TryUpdateThreeBufferCandidate(uint64_t candidateDepthA1, uint64_t candidateDepthB1, uint64_t baseASize,
                                       uint64_t baseBSize, bool requireMinLoadSize, L1SearchResult &result);
    bool IsBetterThreeBufferCandidate(uint64_t highBwPayload, uint64_t totalPayload,
                                      const L1SearchResult &result) const;
    void ApplyThreeBufferSearchResult(const L1SearchResult &result);
    ge::graphStatus FinalizeStepAndScale();
    uint64_t GetDepthWithHighBW(uint64_t mnL1) const;
    void ModifyDepthForUnalign(uint64_t leftL1Size, uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleABSize);
    ge::graphStatus CalScaleFactors();
    bool CanEnableThreeL1Buffer() const;
    GroupedMatmulTilingData::GMMQuantBasicApiTilingData tilingData_;
};

constexpr uint32_t S4S4_QUANT_GROUP_SIZE = 2U;
constexpr uint32_t S4S4_BASE_M_950 = 128U;
constexpr uint32_t S4S4_BASE_N_950 = 256U;
constexpr uint32_t S4S4_VEC_BASE_M_950 = 16U;
constexpr uint32_t S4S4_BASE_K_950 = 128U;
constexpr uint32_t S4S4_EPILOGUE_UB_COEFF = 24U;
constexpr uint32_t PROLOGUE_MIN_BYTES_950 = 48U * 1024U;
constexpr uint32_t S4S4_MMOUT_PIPELINE = 4U;
constexpr uint32_t S4S4_N_ALIGN = 8U;
constexpr uint32_t S4S4_PER_CHANNEL_SCALE_DIM = 2U;
constexpr uint32_t S4S4_PER_GROUP_SCALE_DIM = 3U;
constexpr uint32_t S4S4_PER_TOKEN_QUEUE_BYTES_PER_M = 8U;
constexpr uint64_t S4S4_AIV_PER_AIC = 2UL;
constexpr uint64_t S4S4_KERNEL_TYPE_MIX = 3UL;
constexpr uint64_t S4S4_WEIGHT_NZ_FLAG = 1ULL;
constexpr uint64_t S4S4_TRANSPOSE_WEIGHT_FLAG = 2ULL;
constexpr uint64_t S4S4_WEIGHT_NZ_C0_32_FLAG = 4ULL;

class GroupedS4S4IntQuantTiling : public GroupedQmmBasicApiTiling {
public:
    explicit GroupedS4S4IntQuantTiling(gert::TilingContext *context);
    ~GroupedS4S4IntQuantTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQmmBasicApiTiling::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    virtual void Reset();

    // S4S4-specific (hide base's same-named; call base then sync s4s4Tiling_)
    bool SetMKN(const gert::Shape &xShape, const gert::Shape &wShape);
    bool SetGroupNum(uint32_t groupListIndex);
    bool SetMKNList();
    bool AnalyzeS4S4();
    bool CheckS4S4Params();
    void CalBasicBlockS4S4();
    ge::graphStatus CalUbDivideS4S4();
    ge::graphStatus CalWorkspaceS4S4();
    uint32_t FindBestSingleN() const;
    void InitCommonL1TilingFields();
    ge::graphStatus CalcLeftL1Size(uint64_t &leftL1Size) const;
    ge::graphStatus CalL1Tiling();
    ge::graphStatus CalL1Depth(uint64_t leftL1Size);

private:
    struct S4S4BasicTiling {
        uint64_t m = 0UL;
        uint64_t n = 0UL;
        uint64_t k = 0UL;
        uint32_t baseM = S4S4_BASE_M_950;
        uint32_t baseN = S4S4_BASE_N_950;
        uint32_t baseK = 0U;
        uint32_t ubCalSize = 0U;
        uint32_t ubRestBytes = 0U;
        uint32_t quantGroupNum = 1U;
        uint32_t singleN = 0U;
        uint32_t usedCoreNum = 1U;
        uint32_t isPerTokenQuant = 0U;
        uint32_t isS4S4Optimize = 0U;
        int8_t groupType = 0;
        uint8_t groupListType = 0;
        uint32_t groupNum = 0U;
        uint64_t singleCoreM = 1UL;
        uint64_t singleCoreN = 1UL;
        uint64_t singleCoreK = 1UL;
        uint64_t depthA1 = 1UL;
        uint64_t depthB1 = 1UL;
        uint64_t stepM = 1UL;
        uint64_t stepN = 1UL;
        uint64_t stepKa = 1UL;
        uint64_t stepKb = 1UL;
        uint32_t iterateOrder = 0U;
        uint32_t dbL0c = 1U;
    };
    S4S4BasicTiling s4s4Tiling_;
    // Shadows base's private tilingData_ (GMMQuantBasicApiTilingData); own type
    GroupedMatmulTilingData::GMMS4S4IntQuantTilingData tilingData_;
    uint64_t int8WeightWs_ = 0UL;
    uint64_t int8XWs_ = 0UL; // ★ x int4->int8 预转换 workspace (m·k·1B, m=totalM single x)
    uint64_t mmOutWs_ = 0UL;
    uint64_t perTokenScaleFillWs_ = 0UL; // F3: isPerTokenQuant=0 时全 1 perTokenScale 兜底段 (m·float)
    bool weightNzC032_ = false;
    // mList_/kList_/nList_ shadow base's private same-named (base private, derived inaccessible)
    int32_t mList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t kList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t nList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};
};

} // namespace optiling

#endif // GROUPED_QUANT_BASIC_API_MATMUL_TILING_H
