/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file chunk_local_cumsum_tiling.cpp
 */
#include "chunk_local_cumsum_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNumAiv();

    const int32_t chunkSize = *(context->GetAttrs()->GetAttrPointer<int32_t>(0));
    const int64_t* cuSeqlenDictAttr = const_cast<int64_t*>(context->GetAttrs()->GetListInt(1)->GetData());

    auto shape_input = context->GetInputTensor(0)->GetOriginShape();
    auto shape_cu_seqlens = context->GetInputTensor(1)->GetOriginShape();

    int32_t headNum = shape_input.GetDim(2);
    int32_t tokenNum = shape_input.GetDim(1);
    int32_t seqNum = shape_cu_seqlens.GetDim(0) - 1;

    constexpr int32_t MAX_SEQ_NUM = 128; // curChunkNumDict 固定 128 个槽，对应最大 batch 数
    if (seqNum > MAX_SEQ_NUM) {
        OP_LOGE("ChunkLocalCumsum", "cu_seqlens length is %ld, seqNum %d exceeds max supported batch %d.",
                static_cast<int64_t>(shape_cu_seqlens.GetDim(0)), seqNum, MAX_SEQ_NUM);
        return ge::GRAPH_FAILED;
    }

    constexpr int32_t MAX_HEAD_NUM = 32;   // kernel UB 槽按 64*32*4B=8KB 分配，headNum>32 时单组超容量
    constexpr int32_t MAX_CHUNK_SIZE = 64; // kernel 侧 UB 槽与扫描逻辑均按 64 设计
    if (headNum > MAX_HEAD_NUM) {
        OP_LOGE("ChunkLocalCumsum", "headNum %d exceeds max supported %d.", headNum, MAX_HEAD_NUM);
        return ge::GRAPH_FAILED;
    }
    if (chunkSize > MAX_CHUNK_SIZE) {
        OP_LOGE("ChunkLocalCumsum", "chunkSize %d exceeds max supported %d.", chunkSize, MAX_CHUNK_SIZE);
        return ge::GRAPH_FAILED;
    }

    if (headNum > 16) {
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(2);
    }

    int32_t totalChunkNum = 0;
    int32_t curChunkNumArray[MAX_SEQ_NUM] = {0}; // 初始化为0

    for (int32_t seqIdx = 0; seqIdx < seqNum; seqIdx++) {
        int32_t curSeqlen = cuSeqlenDictAttr[seqIdx + 1] - cuSeqlenDictAttr[seqIdx];
        int32_t curChunkNum = ((curSeqlen + chunkSize - 1) / chunkSize);

        totalChunkNum += curChunkNum;
        curChunkNumArray[seqIdx] = totalChunkNum;
    }

    ChunkLocalCumsumTilingData tiling;
    int32_t usedCore = totalChunkNum > aivNum ? aivNum : totalChunkNum;

    context->SetBlockDim(usedCore); // 直接拉起所有的core
    tiling.set_coreNum(aivNum);
    tiling.set_tokenNum(tokenNum);
    tiling.set_headNum(headNum);
    tiling.set_chunkNum(totalChunkNum);
    tiling.set_chunkSize(chunkSize);
    tiling.set_curChunkNumDict(curChunkNumArray);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkLocalCumsum).Tiling(TilingFunc);

} // namespace optiling
