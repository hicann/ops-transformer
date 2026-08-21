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
 * \file sparse_lightning_indexer_grad_kl_loss_tiling_general_regbase.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_GENERAL_REGBASE_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_GENERAL_REGBASE_H

#include <numeric>
#include <algorithm>
#include "op_host/tiling_base.h"
#include "op_host/tiling_type.h"
#include "../sparse_lightning_indexer_grad_kl_loss_tiling_common.h"
#include "../../op_kernel/arch35/sparse_lightning_indexer_grad_kl_loss_template_tiling_key_regbase.h"
#include "../../op_kernel/arch35/sparse_lightning_indexer_grad_kl_loss_tiling_regbase.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

class SparseLightningIndexerGradKLLossTilingBaseRegbase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit SparseLightningIndexerGradKLLossTilingBaseRegbase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~SparseLightningIndexerGradKLLossTilingBaseRegbase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    void Reset() {
        bSize = 0;
        gSizeQuery = 0;
        gSizeQueryIndex = 0;
        dSizeQuery = 0;
        dSizeQueryIndex = 0;
        kSize = 2048;
        n2Size = 0;
        s1Size = 0;
        s2Size = 0;
        accumS1 = 0;
        accumS2 = 0;
        sparseMode = 3;
        scaleValue = 1.0f;
        dQueryRopeSize = 0;
        dKeyRopeSize = 0;

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
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    void GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const;
    ge::graphStatus CheckOutShape(gert::Shape &inputshape, const char *inputName, gert::Shape &outputshape);
    ge::graphStatus CheckOutPut();
    ge::graphStatus CheckContext();
    bool AnalyzeAttrs();
    bool CrossShapeVerify(const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShap);
    bool AnalyzeDimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &queryIndexShape,
                         const gert::Shape &topKShape, size_t layoutLen, const gert::Shape &queryRopeShape, const gert::Shape &keyRopeShape);
    bool AnalyzeDtype();
    bool AnalyzeLayout();
    int64_t GetS2RealSize(int32_t sparseMode, int32_t s1Size, int32_t s2Size, int32_t s1Idx);
    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray);
    inline bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize,
                            const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue);
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx);
    bool Balance4DLoad(std::vector<int64_t> &tmpSparseValue, const std::vector<int64_t> sparseValidArray, 
                    const int64_t balanceNum);
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                       int64_t maxCoreNum);
    void SetSparseParamsRegbase(int64_t maxCoreNum);
    int64_t CalcTotalSize();
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum, uint32_t syKTotalSize, uint32_t pkTotalSize);
    void InitOutputSplit();

    // 基础输入参数
    int32_t bSize;
    int32_t n2Size;
    int32_t gSizeQuery;
    int32_t gSizeQueryIndex;
    int32_t s1Size;
    int32_t s2Size;
    int32_t dSizeQuery;
    int32_t dSizeQueryIndex;
    int32_t kSize;
    int32_t sparseMode;
    int32_t rsvd;
    float scaleValue;

    const char *templateName = "slikbase";

    LayoutTypeRegbase tilingKeyLayout;
    TopKRangeRegbase topKRange;
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t l2CacheSize;

    int64_t realT1Size;
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

    std::vector<int64_t> actualSeqLenData;
    std::vector<int64_t> actualSeqLenKData;

    SparseLightningIndexerGradKLLossRegBaseTilingData *tilingData = context_->GetTilingData<SparseLightningIndexerGradKLLossRegBaseTilingData>();
    SLIGradKLLossBaseParamsRegbase *sliGradkllossBaseParams_ = &tilingData->baseParams;
    SLIGradKLLossMultiCoreParamsRegbase *sliGradkllossMultiCoreParams_ = &tilingData->multiCoreParams;
};

} // optiling
#endif