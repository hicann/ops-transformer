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
 * \file quant_flash_attn_tiling.h
 * \brief QuantFlashAttn arch35 tiling implementation
 */
#ifndef QUANT_FLASH_ATTN_TILING_IMPL_H_
#define QUANT_FLASH_ATTN_TILING_IMPL_H_

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../qfa_tiling_info.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/quant_flash_attn_tiling_data.h"
#include "../../op_kernel/arch35/quant_flash_attn_template_tiling_key.h"

namespace optiling {
namespace quant_flash_attn {

struct QfaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t config = 0;
    uint64_t quantMode = 0;
    bool hasAttenMask = false;
    uint64_t kvLayoutType = 0;
    bool isFd = false;
    bool isReconstructTemp = false;
};

struct QfaPlatFormInfo {
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

class QuantFlashAttnTilingImpl : public FiaTilingBase {
public:
    explicit QuantFlashAttnTilingImpl(gert::TilingContext *context) : FiaTilingBase(context)
    {
    }
    ~QuantFlashAttnTilingImpl() override = default;

    void InitTilingInfo(TilingInfo *tilingInfo) override;
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
    void UpdateTilingKeyQuantMode();
    void UpdateTilingKeyInfo();
    void SetQFATilingData();
    void InitImplParam();
    void PrintAllTilingData();
    void CalcScheduleMode();
void CalcNumBlocks(uint32_t aicNum);
    void FillTiling();
    ge::graphStatus SetTilingData(QuantFlashAttnTilingData &tilingData);
    bool CheckNeedInitOutput() const;

    QuantFlashAttnTilingData tilingData_;
    QfaTilingKeyInfo tilingKeyInfo_;
    QfaPlatFormInfo platformInfo_;
    uint32_t sOuterFactor_ = 0;
    uint32_t sInnerFactor_ = 0;
    bool flashDecodeFlag_ = false;
    bool cuSeqLenQFlag_ = false;
    bool cuSeqLenKVFlag_ = false;
    bool seqUsedQFlag_ = false;
    bool seqUsedKvFlag_ = false;
    bool decodeS1GMerge_ = false;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    uint32_t numBlocks_ = 0;

    QfaTilingInfo *qfaInfo_ = nullptr;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_TILING_IMPL_H_
