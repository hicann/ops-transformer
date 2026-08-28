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
 * \file bsa_select_block_mask_tiling_base.h
 * \brief
 */
#ifndef __BSA_SELECT_BLOCK_MASK_TILING_BASE_H
#define __BSA_SELECT_BLOCK_MASK_TILING_BASE_H

#include <numeric>
#include <algorithm>
#include "../op_kernel/bsa_select_block_mask_tiling_key.h"
#include "../op_kernel/bsa_select_block_mask_tiling_data.h"
#include "tiling_base_class.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

struct BSASelectBlockMaskCompileInfo {
    int64_t core_num;
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;
};

class BSASelectBlockMaskTilingBase : public TilingBaseClass {
public:
    explicit BSASelectBlockMaskTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~BSASelectBlockMaskTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    void Reset()
    {
        batchSize = 0;
        numHeads = 0;
        maxQSeqlen = 0;
        maxKvSeqlen = 0;
        dSize = D_SIZE_128;
        blockShapeX = 128;
        blockShapeY = 128;
        xBlocks = 0;
        yBlocks = 0;
        scaleValue = 0.08838834764831845f;
        sparsity = 0.5f;
        topKValue = 0ULL;
        sparsityMode = 0;
        useActualBlockLenQ = 0;
        useActualBlockLenK = 0;
        useActualSeqLenQ = 0;
        useActualSeqLenK = 0;
        usePostBlockShape = 0;
        postBlockShapeX = 1;
        postBlockShapeY = 1;
        postXBlocks = 0;
        postYBlocks = 0;
        queryLayout = nullptr;
        kvLayout = nullptr;
        opName = nullptr;
        socVersion = platform_ascendc::SocVersion::ASCEND910B;
    }

    [[nodiscard]] gert::TilingContext *GetContext()
    {
        return context_;
    }
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus CheckContext();
    bool AnalyzeAttrs();
    bool AnalyzeDtype();
    bool AnalyzeLayout();
    bool CrossShapeVerify();
    void CalcMultiCoreParams();
    void CalcOutputParams();
    bool IsA5Platform() const;
    uint32_t GetAivRatio() const;
    uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const;
    uint32_t CalcLaunchAivNum(uint32_t activeCoreNum) const;

    uint32_t batchSize;
    uint32_t numHeads;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t dSize;
    uint64_t blockShapeX;
    uint64_t blockShapeY;
    uint32_t xBlocks;
    uint32_t yBlocks;
    float scaleValue;
    float sparsity;
    uint64_t topKValue;
    uint8_t sparsityMode;
    uint8_t useActualBlockLenQ;
    uint8_t useActualBlockLenK;
    uint8_t useActualSeqLenQ;
    uint8_t useActualSeqLenK;
    uint8_t usePostBlockShape;
    uint16_t postBlockShapeX;
    uint16_t postBlockShapeY;
    uint32_t postXBlocks;
    uint32_t postYBlocks;

    const char *opName;
    const char *queryLayout;
    const char *kvLayout;

    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;

    LayoutType tilingKeyLayoutQ;
    LayoutType tilingKeyLayoutKV;

    BSASelectBlockMaskTilingData *tilingData = nullptr;
    BSABaseParams *baseParams_ = nullptr;
    BSAMultiCoreParams *multiCoreParams_ = nullptr;
    BSAOutputParams *outputParams_ = nullptr;
};

} // namespace optiling
#endif
