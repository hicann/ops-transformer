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
 * \file dense_lightning_indexer_grad_kl_loss_tiling.h
 * \brief
 */

#ifndef __DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_H
#define __DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_H

#include <numeric>
#include <algorithm>
#include "../op_kernel/dense_lightning_indexer_grad_kl_loss_tiling_key.h"
#include "../op_kernel/dense_lightning_indexer_grad_kl_loss_tiling_data.h"
#include "lightning_indexer_grad_kl_loss_tiling_base.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

struct DenseLightningIndexerGradKLLossCompileInfo {
    int64_t core_num;

    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;
};

class DenseLightningIndexerGradKLLossTilingBase : public TilingBaseClass {
public:
    explicit DenseLightningIndexerGradKLLossTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~DenseLightningIndexerGradKLLossTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    void Reset()
    {
        bSize = 0;
        gSizeQuery = 0;
        gSizeQueryIndex = 0;
        dSizeQuery = 0;
        dSizeQueryIndex = 0;
        kSize = 2048;
        n2Size = 0;
        n2IndexSize = 0;
        s1Size = 0;
        s2Size = 0;
        accumS1 = 0;
        accumS2 = 0;
        maxS1Val = 0;
        maxS2Val = 0;
        sparseMode = 3;
        scaleValue = 1.0f;

        deterministic = false;

        opName = nullptr;
        inputLayout = nullptr;
    }

    [[nodiscard]] gert::TilingContext *GetContext()
    {
        return context_;
    }
    bool IsCapable()
    {
        return true;
    }

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 5、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus CheckContext();
    bool AnalyzeAttrs();
    bool AnalyzeDtype();
    bool AnalyzeLayout();
    bool CrossShapeVerify(const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape);
    bool CheckSeqLenLimit(int64_t querySeqLength, int64_t keySeqLength) const;
    bool Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                           const gert::Shape &queryIndexShape, const gert::Shape &keyIndexShape, size_t layoutLen,
                           const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape);
    void GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res, int64_t &actualLen) const;
    int64_t CalcTotalSize();
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);
    bool SetSparseParamsRegbase();
    void InitOutputSplit();

    // deter 相关
    void CalcMaxLoop();
    int32_t GetS2SparseLen(int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK, int32_t sparseMode);
    int32_t GetActualSeqLens(int32_t bIdx, int32_t defaultLens,
                             std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &actualSeqLenData, LayoutType layout);
    int64_t FindBIndex(int64_t bIndex, int64_t curBs1Index, int64_t &accumulateLen);
    void CalcMultiCoreOffset(int64_t &bStartIdx, int64_t &s1StartIdx, int64_t &bEndIdx, int64_t &s1EndIdx,
                             int64_t &aicIdx);
    int64_t GetEndS1Etx(int32_t bIdx, int32_t defaultLens, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &actualSeqLenData,
                        LayoutType layout);

    int64_t GetS2RealSize(int32_t sparseMode, int32_t s1Size, int32_t s2Size, int32_t s1Idx);
    inline bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize,
                              const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue);
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, std::vector<int64_t> &localValue,
                     std::vector<int64_t> &sparseStartIdx);
    bool Balance4DLoad(std::vector<int64_t> &tmpSparseValue, const std::vector<int64_t> sparseValidArray,
                       const int64_t balanceNum);
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray);
    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray);

    // 基础输入参数
    int32_t bSize;
    int32_t n2Size;
    int32_t n2IndexSize;
    int32_t gSizeQuery;
    int32_t gSizeQueryIndex;
    int32_t s1Size;
    int32_t s2Size;
    int32_t dSizeQuery = 0;
    int32_t dSizeQueryIndex;
    int32_t kSize;
    int32_t sparseMode;
    int32_t rsvd;
    float scaleValue;

    // 新增的一些参数
    const char *templateName = "dlikbase";

    LayoutType tilingKeyLayout;
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t l2CacheSize;

    int64_t realT1Size = 0;
    int64_t realT2Size = 0;
    int64_t padT1Size = 0;
    int64_t padT2Size = 0;
    int64_t dQueryRopeSize;
    int64_t dKeyRopeSize;
    int64_t accumS1;
    int64_t accumS2;
    int64_t maxS1Val;
    int64_t maxS2Val;

    platform_ascendc::SocVersion socVersion;
    bool deterministic;
    bool hasRope = false;

    const char *opName;
    const char *inputLayout;

    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKData;

    DenseLightningIndexerGradKLLossTilingData *tilingData =
        context_->GetTilingData<DenseLightningIndexerGradKLLossTilingData>();
    DLIGradKLLossBaseParams *dliGradkllossBaseParams_ = &tilingData->baseParams;
    DLIGradKLLossMultiCoreParams *dliGradkllossMultiCoreParams_ = &tilingData->multiCoreParams;
};

} // namespace optiling
#endif
