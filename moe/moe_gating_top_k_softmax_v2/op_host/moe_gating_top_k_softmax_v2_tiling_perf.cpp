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
 * \file moe_gating_top_k_softmax_v2_tiling_perf.cpp
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

static const int32_t SIZE_2 = 2;
static const int32_t SIZE_3 = 3;
static const int32_t TWO = 2;
static const int32_t SIX = 6;
static const int32_t FP32_SIZE = 4;
static const int32_t FP16_SIZE = 2;
static const int32_t BF16_SIZE = 2;
static const int32_t INT32_SIZE = 4;
static const int32_t BOOL_SIZE = 1;
static const int64_t BUFFER_SIZE = 96;
static const int64_t CONSTANT_EIGHT = 8;
static const int64_t MAX_COL = 1024;

static uint32_t GenIndicesMask(int64_t k)
{
    uint32_t mask = TWO; // 10
    for (int i = 1; i < k; ++i) {
        mask = (mask << TWO) | TWO;
    }
    return mask;
}

static uint32_t GenValuesMask(int64_t k)
{
    uint32_t mask = 1; // 01
    for (int i = 1; i < k; ++i) {
        mask = (mask << TWO) | 1;
    }
    return mask;
}

class MoeGatingTopKSoftmaxV2PerfTiling : public MoeGatingTopKSoftmaxV2BaseTiling {
public:
    explicit MoeGatingTopKSoftmaxV2PerfTiling(gert::TilingContext* context)
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
    MoeGatingTopKSoftmaxV2PerfTilingData tilingData;

    uint32_t calcMaxRowInUb(const int64_t ubSize, const ge::DataType dtype, const uint32_t k, const uint32_t blockRow,
                            const uint32_t col);

    bool isBufferSizeEnough(const uint32_t curRowInUb, const uint32_t gatingAlignCol, const int64_t tmpUbSize,
                            const ge::DataType dtype, const uint32_t k);

    bool getDoubleBufferFlag(const uint32_t gatingAlignCol, const int64_t ubSize, const ge::DataType dtype,
                             const uint32_t k);
};

bool MoeGatingTopKSoftmaxV2PerfTiling::IsCapable()
{
    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    if ((col <= MAX_COL && col % CONSTANT_EIGHT == 0 && k <= CONSTANT_EIGHT) || (col < CONSTANT_EIGHT)) {
        return true;
    }
    return false;
}

ge::graphStatus MoeGatingTopKSoftmaxV2PerfTiling::DoOpTiling()
{
    gatingAlignCol = calcGatingAlignCol(col, dtype);
    doubleBufferFlag = getDoubleBufferFlag(gatingAlignCol, ubSize, dtype, k);
    maxRow = calcMaxRowInUb(ubSize, dtype, k, CeilDiv(row, coreNum), gatingAlignCol);

    tilingData.set_row(row);
    tilingData.set_col(col);

    tilingData.set_k(k);
    uint32_t kAlign = CeilDiv(k, CONSTANT_EIGHT) * CONSTANT_EIGHT;
    tilingData.set_kAlign(kAlign);

    tilingData.set_blockFormer(CeilDiv(row, coreNum));
    tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
    tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());
    tilingData.set_colBytesAlign(CeilDiv(col, BLOCK_SIZE / FP32_SIZE) * (BLOCK_SIZE / FP32_SIZE));
    tilingData.set_colAlign(gatingAlignCol);
    tilingData.set_ubLoopOfFormerBlock(CeilDiv(tilingData.get_blockFormer(), maxRow));
    tilingData.set_ubFormer(maxRow);
    tilingData.set_ubTailOfFormerBlock(tilingData.get_blockFormer() -
                                       (tilingData.get_ubLoopOfFormerBlock() - 1) * tilingData.get_ubFormer());
    tilingData.set_ubLoopOfTailBlock(CeilDiv(tilingData.get_blockTail(), maxRow));
    tilingData.set_ubTailOfTailBlock(tilingData.get_blockTail() -
                                     (tilingData.get_ubLoopOfTailBlock() - 1) * tilingData.get_ubFormer());
    if (renorm == 1) {
        tilingData.set_topKValuesMask(GenValuesMask(kAlign));
    } else {
        tilingData.set_topKValuesMask(GenValuesMask(k));
    }

    tilingData.set_topKIndicesMask(GenIndicesMask(k));
    tilingData.set_bufferElemSize(std::max(maxRow * gatingAlignCol, 64u));
    tilingData.set_softmaxFlag(softmaxFlag);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2PerfTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxV2PerfTiling::GetTilingKey() const
{
    int sceneValue = 3;
    //
    return TOPK_SOFTMAX_TILING_KEY_BASE_ALL + sceneValue * TOPK_SOFTMAX_TILING_KEY_BASE_SCENE +
           renorm * TOPK_SOFTMAX_TILING_KEY_BASE_RENORM + dtypeKey(dtype) * TOPK_SOFTMAX_TILING_KEY_BASE_DTYPE +
           colNumKey(col);
}

ge::graphStatus MoeGatingTopKSoftmaxV2PerfTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYSTEM_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxV2PerfTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData.get_blockNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool MoeGatingTopKSoftmaxV2PerfTiling::getDoubleBufferFlag(const uint32_t gatingAlignColLocal,
                                                           const int64_t ubSizeLocal, const ge::DataType dtypeLocal,
                                                           const uint32_t kLocal)
{
    // 判断一行的数据是否能搬进一半大小的ub空间
    return isBufferSizeEnough(1, gatingAlignColLocal, ubSizeLocal / SIZE_2, dtypeLocal, kLocal);
}

bool MoeGatingTopKSoftmaxV2PerfTiling::isBufferSizeEnough(const uint32_t curRowInUb, const uint32_t gatingAlignColLocal,
                                                          const int64_t tmpUbSize, const ge::DataType dtypeLocal,
                                                          const uint32_t kLocal)
{
    (void)kLocal;
    // 1.搬入gating
    int typeSize = ge::GetSizeByDataType(dtypeLocal);
    int64_t gatingAlignBufferSize = curRowInUb * gatingAlignColLocal * typeSize;
    if (gatingAlignBufferSize > tmpUbSize) {
        return false;
    }

    // 2.计算topk
    int64_t finishedUbBufferSize = CeilDiv(BOOL_SIZE * curRowInUb, BLOCK_SIZE) * BLOCK_SIZE;

    // sourceRowOut用来复用缓存softmax输出，按r*E分配大小
    int64_t sourceRowOutAlignBufferSize = curRowInUb * gatingAlignColLocal * INT32_SIZE;

    int64_t tempBufferSize = sourceRowOutAlignBufferSize * SIX;
    if (dtypeLocal == ge::DataType::DT_FLOAT) {
        tempBufferSize += sourceRowOutAlignBufferSize;
    }

    if (BUFFER_SIZE + finishedUbBufferSize * TWO + tempBufferSize > tmpUbSize) {
        return false;
    }
    return true;
}

uint32_t MoeGatingTopKSoftmaxV2PerfTiling::calcMaxRowInUb(const int64_t ubSizeLocal, const ge::DataType dtypeLocal,
                                                          const uint32_t kLocal, const uint32_t blockRow,
                                                          const uint32_t colLocal)
{
    (void)colLocal;
    uint32_t ubOuter = 1;
    int64_t tmpUbSize = ubSizeLocal;
    uint32_t curRowInUb;
    while (true) {
        curRowInUb = CeilDiv(blockRow, ubOuter);
        if (isBufferSizeEnough(curRowInUb, gatingAlignCol, tmpUbSize, dtypeLocal, kLocal)) {
            break;
        }
        ubOuter++;
        if (ubOuter > blockRow) {
            return 0;
        }
    }
    return curRowInUb;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2PerfTiling, 1000);
} // namespace optiling
