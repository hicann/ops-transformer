/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_STRUCT_H
#define MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_STRUCT_H

#include <string>
#include <numeric>
#include <climits>
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "op_host/op_tiling/hccl_formulaic_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_calc_num_blocks.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling.h"
#include "allto_allv_grouped_mat_mul_tiling_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "context_util.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling_key.h"
#include "../../../3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.h"

namespace optiling {
class AlltoAllvGmmTilingStruct : public AlltoAllvGmmTilingBase {
public:
    explicit AlltoAllvGmmTilingStruct(gert::TilingContext *context) : AlltoAllvGmmTilingBase(context) {};

protected:
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};

class AlltoAllvGmmTiling : public AlltoAllvGmmTilingBase {
    friend class AlltoAllvGmmTilingHelper;

public:
    explicit AlltoAllvGmmTiling(gert::TilingContext *context) : AlltoAllvGmmTilingBase(context)
    {
    }

    AlltoAllvGmmTilingData *tilingData;

    ge::graphStatus Init(gert::TilingContext *context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);
    virtual std::vector<int64_t> GetEpWorldSizeOptional() const = 0;
    virtual bool NeedToCheckCounts() const = 0;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetContextAttr(const gert::TilingContext *context);
    ge::graphStatus GetShapeAndFormat(const gert::TilingContext *context);
    ge::graphStatus CheckMKN(const gert::TilingContext *context);
    ge::graphStatus CheckShapeSize(const gert::TilingContext *context) const;
    ge::graphStatus CheckAttrsShapeSize(const gert::TilingContext *context) const;
    ge::graphStatus CheckAttrsShapeRelation(const gert::TilingContext *context) const;
    ge::graphStatus CheckSendRecvDataVolumn(const gert::TilingContext *context) const;
    ge::graphStatus CheckShapeRelation(const gert::TilingContext *context) const;
    ge::graphStatus CheckShapeDims(const gert::TilingContext *context);
    ge::graphStatus CheckDType(const gert::TilingContext *context) const;
    ge::graphStatus CheckMmShapeDims(const gert::TilingContext *context) const;
    ge::graphStatus GetAndConvertCommMode(gert::TilingContext *context, uint8_t &commMode) const;
    ge::graphStatus SetHcclTiling(const gert::TilingContext *context) const;

    ge::graphStatus DoAiCoreTiling(const gert::TilingContext *context);
    uint64_t GetTilingKey() const override;
    ge::graphStatus setNumBlocks(gert::TilingContext *context);

private:
    int32_t maxM_ = 0;
    int32_t maxN_ = 0;
    int32_t maxK_ = 0;
    int32_t baseM_ = 0;
    int32_t baseN_ = 0;
    int32_t baseK_ = 0;
    uint32_t mmDataTypeSize = 0;

    int32_t maxMForMM_ = 0;
    int32_t maxNForMM_ = 0;
    int32_t maxKForMM_ = 0;
    int32_t baseMForMM_ = 0;
    int32_t baseNForMM_ = 0;
    int32_t baseKForMM_ = 0;
};

class AlltoAllvGmmTilingHelper : public Mc2GroupedMatmulTiling::Mc2GroupedQbmmTiling {
public:
    AlltoAllvGmmTilingHelper(AlltoAllvGmmTiling &parent) : Mc2GroupedQbmmTiling(parent.context_), parent_(parent)
    {
    }
    const Mc2GroupedMatmulTilingData::GMMQuantTilingData &GetAlltoAllvQuantHelperData() const
    {
        return tilingData_;
    }
    bool AnalyzeAttrs() override
    {
        return true;
    }
    bool AnalyzeDtype() override
    {
        return true;
    }
    bool AnalyzeInputs() override
    {
        return true;
    }
    void Reset() override
    {
    }
    ge::graphStatus SetInputParams(uint64_t M, uint64_t N, uint64_t K, bool transB, ge::DataType aDtype,
                                   ge::DataType bDtype, ge::DataType cDtype, uint32_t groupNum);
    ge::graphStatus Process();

private:
    AlltoAllvGmmTiling &parent_;
};
} // namespace optiling
#endif
