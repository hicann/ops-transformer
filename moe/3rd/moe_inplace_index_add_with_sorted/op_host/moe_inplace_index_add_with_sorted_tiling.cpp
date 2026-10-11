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
 * \file moe_inplace_index_add_with_sorted_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "moe_inplace_index_add_with_sorted_tiling.h"
#include "platform/platform_infos_def.h"

using namespace std;

namespace {
const int32_t FLOAT32_TILING_KEY = 1;
const int32_t FLOAT16_TILING_KEY = 2;
const int32_t BF16_TILING_KEY = 3;
const int32_t INT16_TILING_KEY = 4;
const int32_t INT32_TILING_KEY = 5;
const int32_t FLOAT32_FIX_TILING_KEY = 6;
const int32_t OTHER_DIM_TILING_KEY = 100;
const int32_t FLOAT32_OTHER_DIM_TILING_KEY = 101;
const int32_t FLOAT16_OTHER_DIM_TILING_KEY = 102;
const int32_t BF16_OTHER_DIM_TILING_KEY = 103;
const int32_t INT16_OTHER_DIM_TILING_KEY = 104;
const int32_t INT32_OTHER_DIM_TILING_KEY = 105;

const int32_t SIZE_OF_FP16 = 2;
const int32_t SIZE_OF_FP32 = 4;
const int32_t SIZE_OF_INT32 = 4;
const int32_t SIZE_OF_BF16 = 2;

const int32_t INPUT_0 = 0;
const int32_t INPUT_1 = 1;
const int32_t INPUT_2 = 2;
const int32_t INPUT_3 = 3;
const int32_t INPUT_4 = 4;
const int32_t BUF_CNT_2 = 2;
const int32_t BUF_CNT_3 = 3;
const int32_t BUF_CNT_6 = 6;
const int32_t BUF_CNT_7 = 7;
const int32_t BLOCK_SIZE = 32;
const int64_t UB_INDEX_NUM = 1536;
const int64_t INDEX_BUFFER_SIZE = UB_INDEX_NUM * 2 * SIZE_OF_INT32;
const int64_t RESERVED_BUFFER_SIZE = 1024;
static const std::map<int32_t, int32_t> DTYPE_BUF_CNT_MAP = {{FLOAT32_TILING_KEY, BUF_CNT_2},
                                                             {FLOAT16_TILING_KEY, BUF_CNT_7},
                                                             {BF16_TILING_KEY, BUF_CNT_7},
                                                             {INT16_TILING_KEY, BUF_CNT_2},
                                                             {INT32_TILING_KEY, BUF_CNT_2},
                                                             {FLOAT32_FIX_TILING_KEY, BUF_CNT_3},
                                                             {FLOAT32_OTHER_DIM_TILING_KEY, BUF_CNT_2},
                                                             {FLOAT16_OTHER_DIM_TILING_KEY, BUF_CNT_6},
                                                             {BF16_OTHER_DIM_TILING_KEY, BUF_CNT_6},
                                                             {INT16_OTHER_DIM_TILING_KEY, BUF_CNT_2},
                                                             {INT32_OTHER_DIM_TILING_KEY, BUF_CNT_2}};
} // namespace

namespace optiling {
class MoeInplaceIndexAddWithSortedTiling {
public:
    explicit MoeInplaceIndexAddWithSortedTiling(gert::TilingContext* context)
        : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    void TilingDataSet();
    void TilingDataPrint() const;
    void processFirstDimTilingData();
    bool CheckParam();
    MoeInplaceIndexAddWithSortedTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t tilingKey = 0;
    int64_t dimAttr = -1;
    int64_t ubSize = 1;
    int64_t inputCount = 1;
    int64_t updatesCount = 1;
    int64_t indicesCount = 1;
    int64_t updatesOneTime = 1;
    int64_t inputSize = 1;
    int32_t coreNum = 1;

    int32_t usedCoreNum = 1;
    int32_t enableAlpha = 0;
    int64_t eachIndexCount = 1;
    int64_t lastIndexCount = 1;
    int64_t maxSize = 0;
    int64_t eachNum = 1;
    int64_t eachLoop = 1;
    int64_t eachTail = 0;
    int64_t batchNum = 1;
    int64_t eachBatchNum = 1;
    int64_t lastBatchNum = 1;
    int64_t varDimNum = 1;
    int64_t eachUBIndexRound = 1;
    int64_t eachUBIndexCount = 1;
    int64_t eachUBIndexTail = 0;
    int64_t lastUBIndexRound = 1;
    int64_t lastUBIndexCount = 1;
    int64_t lastUBIndexTail = 0;
    uint64_t workspaceSize = 1024 * 1024 * 16;
};

bool MoeInplaceIndexAddWithSortedTiling::CheckParam()
{
    if (tilingContext->GetInputShape(INPUT_0) == nullptr || tilingContext->GetInputShape(INPUT_1) == nullptr ||
        tilingContext->GetInputShape(INPUT_2) == nullptr || tilingContext->GetInputShape(INPUT_3) == nullptr ||
        tilingContext->GetInputDesc(INPUT_0) == nullptr || tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE(tilingContext, "tilingContext inputShape or outputShape is nullptr.");
        return false;
    }
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    auto valueDtype = tilingContext->GetInputDesc(INPUT_1)->GetDataType();
    inputSize = ge::GetSizeByDataType(inputDtype);

    if (inputDtype != valueDtype) {
        OP_LOGE(tilingContext, "value dtype must be same as var.");
        return false;
    }

    if (ge::DT_INT32 != tilingContext->GetInputDesc(INPUT_2)->GetDataType() ||
        ge::DT_INT32 != tilingContext->GetInputDesc(INPUT_3)->GetDataType()) {
        OP_LOGE(tilingContext, "sorted_index and pos only support int32.");
        return false;
    }
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto alphaShape = tilingContext->GetOptionalInputShape(INPUT_4);
    enableAlpha = (alphaShape == nullptr) ? 0 : 1;

    auto inputDimNum = inputShape.GetDimNum();
    if (inputDimNum != updatesShape.GetDimNum()) {
        OP_LOGE(tilingContext, "the dimNum of input must equal the dimNum of updates.");
        return false;
    }
    const int64_t* ptrDim = tilingContext->GetAttrs()->GetAttrPointer<int64_t>(0);
    dimAttr = *ptrDim;
    dimAttr = dimAttr < 0 ? inputDimNum + dimAttr : dimAttr;
    // 当前版本仅支持dim = 0，此处做个拦截
    if (dimAttr != 0) {
        OP_LOGE(tilingContext, "Dim only support 0 on the version, but get %ld.", dimAttr);
        return false;
    }
    for (int64_t idx = 0; idx < static_cast<int64_t>(inputDimNum); ++idx) {
        if (dimAttr != idx) {
            if (inputShape.GetDim(idx) != updatesShape.GetDim(idx)) {
                OP_LOGE(tilingContext, "The size of self must match the size of source at dimension %ld.", idx);
                return false;
            }
        }
    }
    return true;
}

ge::graphStatus MoeInplaceIndexAddWithSortedTiling::Init()
{
    if (tilingContext == nullptr) {
        OP_LOGE(tilingContext, "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = static_cast<const MoeInplaceIndexAddWithSortedCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    coreNum = static_cast<int32_t>(compileInfo->totalCoreNum);
    if (coreNum == 0) {
        OP_LOGE(tilingContext, "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    // 固定预留13K的索引ub和posub空间用于遍历索引
    ubSize = static_cast<int64_t>(compileInfo->ubSizePlatForm) - INDEX_BUFFER_SIZE - RESERVED_BUFFER_SIZE;
    workspaceSize = compileInfo->workspaceSize;

    if (!CheckParam()) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddWithSortedTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    auto inputDimNum = inputShape.GetDimNum();
    for (int64_t i = 0; i < static_cast<int64_t>(inputDimNum); ++i) {
        auto dimInput = inputShape.GetDim(i);
        auto dimUpdates = updatesShape.GetDim(i);
        if (i < dimAttr) {
            batchNum *= dimUpdates;
        }
        if (i == dimAttr) {
            varDimNum = dimInput;
        }
        if (i >= dimAttr + 1) {
            updatesOneTime *= dimUpdates;
        }
        inputCount *= dimInput;
        updatesCount *= dimUpdates;
    }
    indicesCount = indicesShape.GetDim(INPUT_0);
    if (inputCount == 0 || updatesCount == 0 || indicesCount == 0) {
        OP_LOGE(tilingContext, "shape size cannot equal 0.");
        return ge::GRAPH_FAILED;
    }
    uint32_t fixedOutputFlag = tilingContext->GetDeterministic() == 1 ? 1 : 0;
    if (ge::DT_FLOAT == inputDtype && fixedOutputFlag == 1) {
        tilingKey = FLOAT32_FIX_TILING_KEY;
    } else if (ge::DT_FLOAT == inputDtype) {
        tilingKey = FLOAT32_TILING_KEY;
    } else if (ge::DT_FLOAT16 == inputDtype) {
        tilingKey = FLOAT16_TILING_KEY;
    } else if (ge::DT_BF16 == inputDtype) {
        tilingKey = BF16_TILING_KEY;
    } else if (ge::DT_INT16 == inputDtype) {
        tilingKey = INT16_TILING_KEY;
    } else if (ge::DT_INT32 == inputDtype) {
        tilingKey = INT32_TILING_KEY;
    }
    processFirstDimTilingData();
    TilingDataSet();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void MoeInplaceIndexAddWithSortedTiling::processFirstDimTilingData()
{
    const auto iter = DTYPE_BUF_CNT_MAP.find(tilingKey);
    maxSize = ((ubSize / iter->second) / BLOCK_SIZE * BLOCK_SIZE) / inputSize;
    usedCoreNum = indicesCount < coreNum ? indicesCount : coreNum;
    eachIndexCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (indicesCount + eachIndexCount - 1) / eachIndexCount;
    lastIndexCount = indicesCount - eachIndexCount * (usedCoreNum - 1);
    eachNum = updatesOneTime;
    eachTail = eachNum;
    if (eachNum > maxSize) {
        eachLoop = (eachNum + maxSize - 1) / maxSize;
        eachNum = maxSize;
        eachTail = updatesOneTime - (eachLoop - 1) * eachNum;
    }
    if (eachIndexCount > UB_INDEX_NUM) {
        eachUBIndexRound = (eachIndexCount + UB_INDEX_NUM - 1) / UB_INDEX_NUM;
        eachUBIndexCount = UB_INDEX_NUM;
        eachUBIndexTail = eachIndexCount - (eachUBIndexRound - 1) * UB_INDEX_NUM;
    } else {
        eachUBIndexRound = 1;
        eachUBIndexCount = eachIndexCount;
        eachUBIndexTail = eachIndexCount;
    }
    if (lastIndexCount > UB_INDEX_NUM) {
        lastUBIndexRound = (lastIndexCount + UB_INDEX_NUM - 1) / UB_INDEX_NUM;
        lastUBIndexCount = UB_INDEX_NUM;
        lastUBIndexTail = lastIndexCount - (lastUBIndexRound - 1) * UB_INDEX_NUM;
    } else {
        lastUBIndexRound = 1;
        lastUBIndexCount = lastIndexCount;
        lastUBIndexTail = lastIndexCount;
    }
}

void MoeInplaceIndexAddWithSortedTiling::TilingDataSet()
{
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_enableAlpha(enableAlpha);
    tilingData.set_eachIndexCount(eachIndexCount);
    tilingData.set_lastIndexCount(lastIndexCount);
    tilingData.set_batchCount(batchNum);
    tilingData.set_eachBatchNum(eachBatchNum);
    tilingData.set_lastBatchNum(lastBatchNum);
    tilingData.set_inputCount(inputCount);
    tilingData.set_indicesCount(indicesCount);
    tilingData.set_updatesCount(updatesCount);
    tilingData.set_updatesOneTime(updatesOneTime);
    tilingData.set_maxSize(maxSize);
    tilingData.set_eachNum(eachNum);
    tilingData.set_eachLoop(eachLoop);
    tilingData.set_eachTail(eachTail);
    tilingData.set_varDimNum(varDimNum);
    tilingData.set_eachUBIndexRound(eachUBIndexRound);
    tilingData.set_eachUBIndexCount(eachUBIndexCount);
    tilingData.set_eachUBIndexTail(eachUBIndexTail);
    tilingData.set_lastUBIndexRound(lastUBIndexRound);
    tilingData.set_lastUBIndexCount(lastUBIndexCount);
    tilingData.set_lastUBIndexTail(lastUBIndexTail);

    TilingDataPrint();

    tilingContext->SetBlockDim(usedCoreNum);
    tilingContext->SetTilingKey(static_cast<uint64_t>(tilingKey));
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize;
}

void MoeInplaceIndexAddWithSortedTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "usedCoreNum: %u.", usedCoreNum);
    OP_LOGD(tilingContext, "enableAlpha: %u.", enableAlpha);
    OP_LOGD(tilingContext, "eachIndexCount: %ld.", eachIndexCount);
    OP_LOGD(tilingContext, "lastIndexCount: %ld.", lastIndexCount);
    OP_LOGD(tilingContext, "batchNum: %ld.", batchNum);
    OP_LOGD(tilingContext, "eachBatchNum: %ld.", eachBatchNum);
    OP_LOGD(tilingContext, "lastBatchNum: %ld.", lastBatchNum);
    OP_LOGD(tilingContext, "inputCount: %ld.", inputCount);
    OP_LOGD(tilingContext, "indicesCount: %ld.", indicesCount);
    OP_LOGD(tilingContext, "updatesCount: %ld.", updatesCount);
    OP_LOGD(tilingContext, "updatesOneTime: %ld.", updatesOneTime);
    OP_LOGD(tilingContext, "maxSize: %ld.", maxSize);
    OP_LOGD(tilingContext, "eachNum: %ld.", eachNum);
    OP_LOGD(tilingContext, "eachLoop: %ld.", eachLoop);
    OP_LOGD(tilingContext, "eachTail: %ld.", eachTail);
    OP_LOGD(tilingContext, "varDimNum: %ld.", varDimNum);
    OP_LOGD(tilingContext, "eachUBIndexRound: %ld.", eachUBIndexRound);
    OP_LOGD(tilingContext, "eachUBIndexCount: %ld.", eachUBIndexCount);
    OP_LOGD(tilingContext, "eachUBIndexTail: %ld.", eachUBIndexTail);
    OP_LOGD(tilingContext, "lastUBIndexRound: %ld.", lastUBIndexRound);
    OP_LOGD(tilingContext, "lastUBIndexCount: %ld.", lastUBIndexCount);
    OP_LOGD(tilingContext, "lastUBIndexTail: %ld.", lastUBIndexTail);
    OP_LOGD(tilingContext, "tilingKey: %u.", tilingKey);
}

ge::graphStatus TilingMoeInplaceIndexAddWithSorted(gert::TilingContext* context)
{
    MoeInplaceIndexAddWithSortedTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForMoeInplaceIndexAddWithSorted(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForMoeInplaceIndexAddWithSorted start.");
    auto compileInfo = context->GetCompiledInfo<MoeInplaceIndexAddWithSortedCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForMoeInplaceIndexAddWithSorted end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInplaceIndexAddWithSorted)
    .Tiling(TilingMoeInplaceIndexAddWithSorted)
    .TilingParse<MoeInplaceIndexAddWithSortedCompileInfo>(TilingPrepareForMoeInplaceIndexAddWithSorted);
} // namespace optiling
