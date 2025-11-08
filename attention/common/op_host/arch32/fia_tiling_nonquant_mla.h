/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_tiling_nonquant_mla.h
 * \brief
 */
#ifndef FIA_TILING_NONQUNAT_MLA_H
#define FIA_TILING_NONQUNAT_MLA_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../fia_tiling_base.h"
#include "../fia_tiling_info.h"
#include "../../../fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h"
#include "../../../fused_infer_attention_score/op_kernel/fused_infer_attention_score_tilingdata.h"

namespace optiling {

class FiaTilingNonQuantMla : public FiaTilingBase {
public:
    explicit FiaTilingNonQuantMla(gert::TilingContext *context) : FiaTilingBase(context) {}
    ~FiaTilingNonQuantMla() override = default;

protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus GetPlatformInfo();
    void GenTilingKey();
    bool DealSameSeqEachBatch() const;

    void ZeroTensorProcess() const;
    void InitParams();

    void Split();
    void CalcInnerSize(uint32_t seqSize);
    void CalcMBaseSize();

    bool IsFlashDecode();

    void CalcMmResSize();
    void CalcMaxMmResSize();

    void FillTilingBaseParams();
    void FillTilingPageAttenParams();
    void FillTilingMaskParams();
    void FillTilingWorkspaceParams();

    void FillTiling();

    uint32_t CalcFlashDecodeParamNums(const uint32_t coreNum) const;
    uint64_t CalcNormalWorkspaceSize(uint32_t coreNum, int64_t mm1ResSize, int64_t mm2ResSize, uint32_t mBaseSize) const;
    uint64_t CalcFlashDecodeWorkspace(const uint32_t coreNum) const;
    void CalcWorkspaceSize();
    void CalcMaxWorkspaceSize();
    void CalcBlockDim(uint32_t coreNum);

    bool splitKVFlag_ = false;

    uint32_t coreNum_ = 0;
    IfaPerfMode perfMode_ = IfaPerfMode::NORMAL;
    uint32_t kvSplitPart_ = 1;
    int64_t mm1ResSize_ = 0;
    int64_t mm2ResSize_ = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t kvSplit_ = 0;
    uint32_t usedCoreNum_ = 0;

    // platform info
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;

    // set info to context
    FusedInferAttentionScoreTilingData *tilingData_ = GetContext()->GetTilingData<FusedInferAttentionScoreTilingData>();
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 256;
    uint32_t mFdBaseSize_ = 8;

    // Tiling Info
    FiaTilingInfo *fiaInfo_ = nullptr;
};

} // namespace optiling
#endif // FIA_TILING_NONQUNAT_MLA_H
