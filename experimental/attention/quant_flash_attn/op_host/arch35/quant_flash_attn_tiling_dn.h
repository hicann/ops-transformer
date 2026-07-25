/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_tiling_dn.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_TILING_H
#define QUANT_FLASH_ATTN_TILING_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../quant_flash_attn_tiling_info.h"
#include "tiling/tiling_api.h" //这个头文件顺序必须在手写的tiling data前
#include "../../op_kernel/arch35/quant_flash_attn_tiling_data.h"

namespace optiling {

struct FaTilingKeyInfo {
    uint64_t inputLayout = 0;
    uint64_t config = 0;
    uint64_t pseMode = 0;
    uint64_t quantMode = 31;
    bool hasAttenMask = false;
    bool hasRope = false;
    uint64_t kvLayoutType = 0;
    bool isFd = false;
    bool emptyTensor = false;
    uint64_t maskMode = 0;
    uint64_t matmulMode = 0;
    bool enableKvPrefix = false;
    bool enableS1OutSplit = false;
};

class QuantFlashAttnTilingDn : public FiaTilingBase {
public:
    explicit QuantFlashAttnTilingDn(gert::TilingContext *context) : FiaTilingBase(context)
    {
    }
    ~QuantFlashAttnTilingDn() override = default;

protected:
    void InitTilingInfo(TilingInfo *tilingInfo) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    ge::graphStatus SetPlatMemoryInfo();
    void SplitPolicy();
    void GenTilingKey();
    void CalcWorkspaceSize();
    void InitImplParam();
    void CalcScheduleMode();
    void CalcNumBlocks(uint32_t coreNum);
    void FillTiling();
    ge::graphStatus SetTilingData(QuantFlashAttnTilingData &tilingData);

    QuantFlashAttnTilingData tilingData_;
    QfaPlatFormInfo platformInfo_;
    FaTilingKeyInfo tilingKeyInfo_;

    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::BATCH_MODE;
    int32_t numBlocks_ = 0;

    // Tiling Info
    QuantFlashAttnTilingInfo *tilingInfo_ = nullptr;
};

} // namespace optiling
#endif
