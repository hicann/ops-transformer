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
 * \file paged_attention_checker.cpp
 * \brief Checker for block_table (文档约束: Paged Attention参数组)
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../quant_flash_attn_tiling_info.h"
#include "paged_attention_checker.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;

bool PagedAttentionChecker::IsPageAttention(const QuantFlashAttnTilingInfo &qfaInfo) const
{
    // 仅通过 layout_kv 判断是否为 PA 场景 (BnBsH/BnNBsD/NZ)
    return (qfaInfo.layoutKV == FiaLayout::BnBsH || qfaInfo.layoutKV == FiaLayout::BnNBsD ||
            qfaInfo.layoutKV == FiaLayout::NZ);
}

ge::graphStatus PagedAttentionChecker::CheckSingleParaBlockTable(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // block_table 是否传入的校验属于特性交叉校验列, 此处为空时跳过 dtype/shape 校验
    const gert::Tensor *blockTableTensor = qfaInfo.opParamInfo.blockTable.tensor;
    if (blockTableTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *blockTableDesc = qfaInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, "TensorDesc of block_table"),
                return ge::GRAPH_FAILED);

    // dtype 校验
    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, BLOCK_TABLE_NAME.c_str(),
                                          DataTypeToSerialStr(blockTableDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    // format 校验
    if (ge::GRAPH_SUCCESS != CheckFormatSupport(blockTableDesc, BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    // shape dim 校验: 2D
    const gert::Shape &shape = blockTableTensor->GetStorageShape();
    size_t dimNum = shape.GetDimNum();
    OP_CHECK_IF(dimNum != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, BLOCK_TABLE_NAME.c_str(),
                                             (std::to_string(dimNum) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);

    // shape size 校验: (B, Bn)
    // B 由 qfaInfo.bSize 给出; Bn(每批次最大块数) 由 qfaInfo.maxBlockNumPerBatch 给出
    int64_t dim0 = shape.GetDim(0);
    int64_t dim1 = shape.GetDim(1);
    OP_CHECK_IF(dim0 != qfaInfo.bSize,
                OP_LOGE(qfaInfo.opName, "block_table shape dim0 should be %ld (B), but got %ld.", qfaInfo.bSize, dim0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dim1 != qfaInfo.maxBlockNumPerBatch,
                OP_LOGE(qfaInfo.opName, "block_table shape dim1 should be %ld (Bn), but got %ld.",
                        qfaInfo.maxBlockNumPerBatch, dim1),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    // block_table 的 dtype/shape
    // 仅 PA 场景下校验 block_table, 通过 layout_kv 判断
    if (!IsPageAttention(qfaInfo)) {
        return ge::GRAPH_SUCCESS;
    }

    if (CheckSingleParaBlockTable(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo)
{
    const gert::Tensor *blockTableTensor = qfaInfo.opParamInfo.blockTable.tensor;

    if (IsPageAttention(qfaInfo)) {
        // 约束1: PA 场景下 block_table 必须不为空
        OP_CHECK_IF(blockTableTensor == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, BLOCK_TABLE_NAME.c_str(), "empty",
                        "When layout_kv is PA (BnBsH/BnNBsD/NZ), block_table must not be empty"),
                    return ge::GRAPH_FAILED);

        // 约束2: PA 场景下必须传入 seqused_kv
        const gert::Tensor *sequsedKvTensor = qfaInfo.opParamInfo.sequsedKv.tensor;
        OP_CHECK_IF(
            sequsedKvTensor == nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, SEQUSED_KV_NAME.c_str(), "empty",
                                                  "When PagedAttention is enabled, seqused_kv must be provided"),
            return ge::GRAPH_FAILED);

        // 约束3: MxFP8 场景下 blockSize 仅支持 512 或 1024
        if (qfaInfo.opParamInfo.quantMode != nullptr && *qfaInfo.opParamInfo.quantMode == 1) {
            OP_CHECK_IF(qfaInfo.blockSize != 512 && qfaInfo.blockSize != 1024,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            qfaInfo.opName, "block_size", std::to_string(qfaInfo.blockSize).c_str(),
                            "When quant_mode is MxFP8, block_size must be 512 or 1024"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // 约束4: 非 PA 场景 (含 MxFP4) 下 block_table 不应传入
        OP_CHECK_IF(blockTableTensor != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        qfaInfo.opName, BLOCK_TABLE_NAME.c_str(), "provided",
                        "When layout_kv is not PA (BnBsH/BnNBsD/NZ), block_table must not be provided"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
