/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file mixed_quant_flash_attn_tiling_arch92.h
 * \brief
 */
#ifndef MQFA_MIXED_QUANT_FLASH_ATTN_TILING_ARCH92_H
#define MQFA_MIXED_QUANT_FLASH_ATTN_TILING_ARCH92_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../mqfa_fa_tiling_info.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/mixed_quant_flash_attn_tiling_data.h"

namespace optiling {
namespace mixed_quant_flash_attn {

struct FaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t kvLayoutType = 0;
    bool hasAttenMask = false;
    uint64_t config = 0;
    uint64_t quantComputeMode = 0;
};

struct FaPlatFormInfo {
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

class MixedQuantFlashAttnTilingImpl : public FiaTilingBase {
public:
    explicit MixedQuantFlashAttnTilingImpl(gert::TilingContext* context)
        : FiaTilingBase(context)
    {}
    ~MixedQuantFlashAttnTilingImpl() override = default;

    void InitTilingInfo(TilingInfo* tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus SetPlatMemoryInfo();
    void SplitPolicy();
    void ComputeTilingData();
    void GenTilingKey();
    void CalcWorkspaceSize();
    void UpdateTilingKeyConfig();
    void UpdateTilingKeyLayout();
    void UpdateTilingKeyKvLayout();
    void UpdateTilingKeyInfo();
    void SetFATilingData();
    void InitImplParam();
    void PrintAllTilingData();
    void CalcMaxWorkspaceSize();
    void CalcScheduleMode();
    bool CheckNeedInitOutput() const;
    void AdjustSinnerAndSouter(uint32_t vHeadDim, int64_t maxSeqQ, int64_t maxSeqKv, int32_t maskMode, int64_t winLeft,
                               int64_t winRight, uint32_t qLayout, uint32_t& sOuterFactor, uint32_t& sInnerFactor);

    void CalcNumBlocks(uint32_t aicNum);
    void FillTiling();
    ge::graphStatus SetTilingData(MixedQuantFlashAttnTilingData& tilingData);

    MixedQuantFlashAttnTilingData tilingData_;
    FaTilingKeyInfo tilingKeyInfo_;
    FaPlatFormInfo platformInfo_;
    uint32_t sOuterFactor_;
    uint32_t sInnerFactor_;
    bool flashDecodeFlag_ = false;
    bool dnFlag_ = false;
    bool cuSeqLenQFlag_ = false;
    bool cuSeqLenKVFlag_ = false;
    bool seqUsedQFlag_ = false;
    bool seqUsedKvFlag_ = false;
    std::vector<int64_t> cuSeqLengthsQ_ = {};
    std::vector<int64_t> cuSeqLengthsKV_ = {};
    bool needInit_ = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    uint32_t numBlocks_ = 0;

    FaTilingInfo* faInfo_ = nullptr;
};

} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif // MQFA_MIXED_QUANT_FLASH_ATTN_TILING_ARCH92_H
