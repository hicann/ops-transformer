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
 * \file block_attention_residuals_tiling.h
 * \brief BlockAttentionResiduals host tiling header
 */
#ifndef ATTN_RES_FWD_TILING_H
#define ATTN_RES_FWD_TILING_H

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "err/ops_err.h"
#include "../op_kernel/block_attention_residuals_tiling_data.h"
#include "../op_kernel/tiling_key_block_attention_residuals.h"

namespace optiling {

struct BlockAttentionResidualsCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct BlockAttentionResidualsInfo {
    const char *opName = "BlockAttentionResiduals";
};

class BlockAttentionResidualsTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockAttentionResidualsTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    }
    ~BlockAttentionResidualsTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();
    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus GetValidBlockNum();
    ge::graphStatus GetNormEps();
    ge::graphStatus GetNeedBackward();
    ge::graphStatus FillStagingFields();
    uint64_t EstimateUbComputeBytes(bool resident) const;
    bool CanResidentAllBlocks(int64_t blockCount, int64_t hiddenSize) const;
    uint32_t CalcMaxResidentRows(int64_t hiddenSize) const;
    int64_t CalcHSliceChunk() const;
    uint32_t GetMinStagingBytes() const;

    BlockAttentionResidualsCompileInfo compileInfo_;
    BlockAttentionResiduals::BlockAttentionResidualsTilingData tilingData_;
    BlockAttentionResidualsInfo inputParams_;
    ge::DataType inputDtype_{ge::DT_BF16};
    uint64_t tilingKey_{TILING_KEY_RELOAD};
    uint64_t workspaceSize_{0UL};

    uint32_t DtypeElemBytes() const;
    uint32_t DtypeElemsPerBlk() const;
};

} // namespace optiling
#endif // ATTN_RES_FWD_TILING_H
