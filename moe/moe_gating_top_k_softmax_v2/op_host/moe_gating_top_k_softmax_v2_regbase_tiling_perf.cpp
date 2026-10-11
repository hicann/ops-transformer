/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_softmax_v2_regbase_tiling_perf.cpp
 * \brief
 */
#include <list>
#include <algorithm>
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "op_host/tiling_util.h"
#include "moe_gating_top_k_softmax_v2_tiling.h"

using namespace Ops::Transformer::OpTiling;
using namespace AscendC;

namespace optiling {
static const uint32_t SIZE_2 = 2;
static const uint32_t SIZE_3 = 3;
static const uint32_t TWO = 2;
static const uint32_t SIX = 6;
static const uint32_t FP32_SIZE = 4;
static const uint32_t FP16_SIZE = 2;
static const uint32_t BF16_SIZE = 2;
static const uint32_t INT32_SIZE = 4;
static const uint32_t BOOL_SIZE = 1;
static const int64_t BUFFER_SIZE = 96;
static const uint32_t CONSTANT_EIGHT = 8;
static const int64_t MAX_COL = 1024;
static const uint32_t MAX_ROW = 255;
static const uint32_t ONE_BLOCK_SIZE = 32;

class MoeGatingTopKSoftmaxV2PerfRegbaseTiling : public MoeGatingTopKSoftmaxV2BaseTiling {
public:
    explicit MoeGatingTopKSoftmaxV2PerfRegbaseTiling(gert::TilingContext* context)
        : MoeGatingTopKSoftmaxV2BaseTiling(context)
    {}

protected:
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint32_t maxRow;
    uint32_t gatingAlignCol;
    bool doubleBufferFlag;
    MoeGatingTopKSoftmaxV2PerfRegbaseTilingData tilingData;

    uint32_t calcMaxRowInUb(const int64_t ubSize, const uint32_t blockRow);

    bool isBufferSizeEnough(const uint32_t curRowInUb, const uint32_t tmpUbSize);

    bool getDoubleBufferFlag();
};

bool MoeGatingTopKSoftmaxV2PerfRegbaseTiling::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    if ((col <= MAX_COL && col % CONSTANT_EIGHT == 0 && k <= static_cast<int32_t>(CONSTANT_EIGHT)) ||
        (col < CONSTANT_EIGHT)) {
        return true;
    }
    return false;
}

ge::graphStatus MoeGatingTopKSoftmaxV2PerfRegbaseTiling::DoOpTiling()
{
    gatingAlignCol = calcGatingAlignCol(col, dtype);
    doubleBufferFlag = getDoubleBufferFlag();
    maxRow = calcMaxRowInUb(ubSize, CeilDiv(row, coreNum));
    maxRow = std::min(maxRow, static_cast<uint32_t>(MAX_ROW / (gatingAlignCol / ALIGN_NUM)));

    tilingData.set_row(row);
    tilingData.set_col(col);

    tilingData.set_k(k);
    uint32_t kAlign = CeilDiv(static_cast<uint32_t>(k), CONSTANT_EIGHT) * CONSTANT_EIGHT;
    tilingData.set_kAlign(kAlign);

    tilingData.set_blockFormer(CeilDiv(row, coreNum));
    tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
    tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());
    tilingData.set_colBytesAlign(CeilDiv(col, ONE_BLOCK_SIZE / FP32_SIZE) * (ONE_BLOCK_SIZE / FP32_SIZE));
    tilingData.set_colAlign(gatingAlignCol);
    tilingData.set_ubLoopOfFormerBlock(CeilDiv(tilingData.get_blockFormer(), maxRow));
    tilingData.set_ubFormer(maxRow);
    tilingData.set_ubTailOfFormerBlock(tilingData.get_blockFormer() -
                                       (tilingData.get_ubLoopOfFormerBlock() - 1) * tilingData.get_ubFormer());
    tilingData.set_ubLoopOfTailBlock(CeilDiv(tilingData.get_blockTail(), maxRow));
    tilingData.set_ubTailOfTailBlock(tilingData.get_blockTail() -
                                     (tilingData.get_ubLoopOfTailBlock() - 1) * tilingData.get_ubFormer());
    tilingData.set_bufferElemSize(std::max(maxRow * gatingAlignCol, 64u));
    tilingData.set_softmaxFlag(softmaxFlag);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2PerfRegbaseTiling::DoLibApiTiling()
{
    int32_t dataTypeSize;
    if (renorm == 0 || dtype == ge::DataType::DT_BF16) {
        dataTypeSize = FP32_SIZE;
    } else {
        dataTypeSize = ge::GetSizeByDataType(dtype);
    }

    auto softmaxShape = ge::Shape({tilingData.get_ubFormer(), gatingAlignCol});
    if (renorm == 1) {
        softmaxShape.SetDim(1, tilingData.get_k());
    }
    SoftMaxTilingFunc(softmaxShape, dataTypeSize, GetSoftMaxMaxTmpSize(softmaxShape, dataTypeSize, true),
                      tilingData.softmaxTilingData);
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxV2PerfRegbaseTiling::GetTilingKey() const
{
    int32_t sceneValue = 3;
    //
    return TOPK_SOFTMAX_TILING_KEY_BASE_ALL + sceneValue * TOPK_SOFTMAX_TILING_KEY_BASE_SCENE +
           renorm * TOPK_SOFTMAX_TILING_KEY_BASE_RENORM + dtypeKey(dtype) * TOPK_SOFTMAX_TILING_KEY_BASE_DTYPE;
}

ge::graphStatus MoeGatingTopKSoftmaxV2PerfRegbaseTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYSTEM_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2PerfRegbaseTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData.get_blockNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool MoeGatingTopKSoftmaxV2PerfRegbaseTiling::getDoubleBufferFlag()
{
    // 判断一行的数据是否能搬进一半大小的ub空间
    return isBufferSizeEnough(static_cast<uint32_t>(1), static_cast<uint32_t>(ubSize) / SIZE_2);
}

bool MoeGatingTopKSoftmaxV2PerfRegbaseTiling::isBufferSizeEnough(const uint32_t curRowInUb, const uint32_t tmpUbSize)
{
    // 1.搬入gating
    uint32_t typeSize = ge::GetSizeByDataType(dtype);
    uint32_t gatingAlignBufferSize = curRowInUb * gatingAlignCol * typeSize;
    if (gatingAlignBufferSize > tmpUbSize) {
        return false;
    }

    uint32_t topkSoftmaxUbBuffer = 0;
    if (renorm == 1) {
        // 2.计算topk
        uint32_t finishedUbBufferSize = CeilDiv(BOOL_SIZE * curRowInUb, ONE_BLOCK_SIZE) * ONE_BLOCK_SIZE;

        uint32_t kAlignBlock =
            CeilDiv(static_cast<uint32_t>(k) * FP32_SIZE, ONE_BLOCK_SIZE) * ONE_BLOCK_SIZE / FP32_SIZE;
        // 3.执行softmax
        auto shape = ge::Shape({curRowInUb, kAlignBlock});
        uint32_t softMaxMaxTmpSize = GetSoftMaxMaxTmpSize(shape, FP32_SIZE, true);
        topkSoftmaxUbBuffer = finishedUbBufferSize * TWO + softMaxMaxTmpSize;
    } else {
        // softmax
        uint32_t kAlignBlock = CeilDiv(col * FP32_SIZE, ONE_BLOCK_SIZE) * ONE_BLOCK_SIZE / FP32_SIZE;
        auto shape = ge::Shape({curRowInUb, kAlignBlock});
        uint32_t softMaxMaxTmpSize = GetSoftMaxMaxTmpSize(shape, FP32_SIZE, true);

        // topk
        uint32_t finishedUbBufferSize = CeilDiv(BOOL_SIZE * curRowInUb, ONE_BLOCK_SIZE) * ONE_BLOCK_SIZE;
        topkSoftmaxUbBuffer = finishedUbBufferSize * TWO + softMaxMaxTmpSize;
    }

    // sourceRowOut用来复用缓存softmax输出，按r*E分配大小
    uint32_t sourceRowOutAlignBufferSize = curRowInUb * gatingAlignCol * INT32_SIZE;

    uint32_t tempBufferSize = sourceRowOutAlignBufferSize * SIX;
    if (dtype == ge::DataType::DT_FLOAT) {
        tempBufferSize += sourceRowOutAlignBufferSize;
    }

    if (BUFFER_SIZE + topkSoftmaxUbBuffer + tempBufferSize > tmpUbSize) {
        return false;
    }
    return true;
}

uint32_t MoeGatingTopKSoftmaxV2PerfRegbaseTiling::calcMaxRowInUb(const int64_t ubSizeLocal, const uint32_t blockRow)
{
    uint32_t ubOuter = 1;
    int64_t tmpUbSize = ubSizeLocal;
    uint32_t curRowInUb;
    while (true) {
        curRowInUb = CeilDiv(blockRow, ubOuter);
        if (isBufferSizeEnough(curRowInUb, tmpUbSize)) {
            break;
        }
        ubOuter++;
        if (ubOuter > blockRow) {
            return 0;
        }
    }
    return curRowInUb;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2PerfRegbaseTiling, 999);
} // namespace optiling
