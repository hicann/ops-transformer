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
 * \file quant_block_sparse_attn_check.cpp
 * \brief QuantBlockSparseAttn parameter validation implementation.
 */

#include "quant_block_sparse_attn_check.h"
#include "quant_block_sparse_attn_tiling.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr size_t DIM_NUM_1 = 1U;
constexpr size_t DIM_NUM_2 = 2U;
constexpr size_t DIM_NUM_3 = 3U;
constexpr size_t DIM_NUM_4 = 4U;
constexpr size_t DIM_0 = 0U;
constexpr size_t DIM_1 = 1U;
constexpr size_t DIM_2 = 2U;
constexpr size_t DIM_3 = 3U;
constexpr int64_t BSA_ATTEN_MASK_DIM_VALUE = 2048;
} // namespace

QuantBlockSparseAttnCheck::QuantBlockSparseAttnCheck(const QuantBlockSparseAttnTilingInfo &tilingInfo)
    : tilingInfo_(tilingInfo)
{
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckDtype() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;

    // 必选输入 dtype
    if (tilingInfo_.qDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "query", std::to_string(static_cast<int>(tilingInfo_.qDtype)),
                                  "FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.kvDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "key/value", std::to_string(static_cast<int>(tilingInfo_.kvDtype)),
                                  "FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.value.desc != nullptr && opParamInfo.value.desc->GetDataType() != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE(
            kOpName, "value", std::to_string(static_cast<int>(opParamInfo.value.desc->GetDataType())), "FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.qDescale.desc != nullptr && opParamInfo.qDescale.desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "q_descale",
                                  std::to_string(static_cast<int>(opParamInfo.qDescale.desc->GetDataType())), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.kDescale.desc != nullptr && opParamInfo.kDescale.desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "k_descale",
                                  std::to_string(static_cast<int>(opParamInfo.kDescale.desc->GetDataType())), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.vDescale.desc != nullptr && opParamInfo.vDescale.desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "v_descale",
                                  std::to_string(static_cast<int>(opParamInfo.vDescale.desc->GetDataType())), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.pScale.desc != nullptr && opParamInfo.pScale.desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "p_scale",
                                  std::to_string(static_cast<int>(opParamInfo.pScale.desc->GetDataType())), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.sparseIndices.desc != nullptr && opParamInfo.sparseIndices.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "sparse_indices",
                                  std::to_string(static_cast<int>(opParamInfo.sparseIndices.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.sparseSeqLen.desc != nullptr && opParamInfo.sparseSeqLen.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "sparse_seq_len",
                                  std::to_string(static_cast<int>(opParamInfo.sparseSeqLen.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }

    // 可选输入 dtype（非 null 且有实际 shape 时校验）
    if (opParamInfo.attenMask.desc != nullptr && opParamInfo.attenMask.shape != nullptr &&
        opParamInfo.attenMask.shape->GetStorageShape().GetShapeSize() > 0 &&
        opParamInfo.attenMask.desc->GetDataType() != ge::DT_UINT8) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "atten_mask",
                                  std::to_string(static_cast<int>(opParamInfo.attenMask.desc->GetDataType())), "UINT8");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.metadata.desc != nullptr && opParamInfo.metadata.tensor != nullptr &&
        opParamInfo.metadata.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "metadata",
                                  std::to_string(static_cast<int>(opParamInfo.metadata.desc->GetDataType())), "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.seqUsedQ.desc != nullptr && opParamInfo.seqUsedQ.tensor != nullptr &&
        opParamInfo.seqUsedQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "seqused_q",
                                  std::to_string(static_cast<int>(opParamInfo.seqUsedQ.desc->GetDataType())), "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.seqUsedKV.desc != nullptr && opParamInfo.seqUsedKV.tensor != nullptr &&
        opParamInfo.seqUsedKV.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "seqused_kv",
                                  std::to_string(static_cast<int>(opParamInfo.seqUsedKV.desc->GetDataType())), "INT32");
        return ge::GRAPH_FAILED;
    }

    // 必传输入 dtype（block_table / cu_seqlens_q / cu_seqlens_kv）
    if (opParamInfo.blockTable.desc != nullptr && opParamInfo.blockTable.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "block_table",
                                  std::to_string(static_cast<int>(opParamInfo.blockTable.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.cuSeqlensQ.desc != nullptr && opParamInfo.cuSeqlensQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "cu_seqlens_q",
                                  std::to_string(static_cast<int>(opParamInfo.cuSeqlensQ.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.cuSeqlensKV.desc != nullptr && opParamInfo.cuSeqlensKV.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "cu_seqlens_kv",
                                  std::to_string(static_cast<int>(opParamInfo.cuSeqlensKV.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }

    // 输出 dtype
    if (opParamInfo.attnOut.desc != nullptr && opParamInfo.attnOut.desc->GetDataType() != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "attention_out",
                                  std::to_string(static_cast<int>(opParamInfo.attnOut.desc->GetDataType())),
                                  "BFLOAT16");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.lseOut.desc != nullptr && opParamInfo.lseOut.desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "softmax_lse",
                                  std::to_string(static_cast<int>(opParamInfo.lseOut.desc->GetDataType())), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckFormat() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    // 必选输入
    const gert::CompileTimeTensorDesc *requiredDescs[] = {
        opParamInfo.query.desc,    opParamInfo.key.desc,           opParamInfo.value.desc,
        opParamInfo.qDescale.desc, opParamInfo.kDescale.desc,      opParamInfo.vDescale.desc,
        opParamInfo.pScale.desc,   opParamInfo.sparseIndices.desc, opParamInfo.sparseSeqLen.desc,
        opParamInfo.attnOut.desc,  opParamInfo.lseOut.desc,
    };
    const char *requiredNames[] = {"query",          "key",           "value",      "q_descale",
                                   "k_descale",      "v_descale",     "p_scale",    "sparse_indices",
                                   "sparse_seq_len", "attention_out", "softmax_lse"};
    for (size_t i = 0; i < sizeof(requiredDescs) / sizeof(requiredDescs[0]); ++i) {
        if (requiredDescs[i] != nullptr && requiredDescs[i]->GetOriginFormat() != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(kOpName, requiredNames[i],
                                       Ops::Base::ToString(requiredDescs[i]->GetOriginFormat()).c_str(), "ND");
            return ge::GRAPH_FAILED;
        }
    }
    // 可选输入（非 null 时校验）
    const gert::CompileTimeTensorDesc *optionalDescs[] = {
        opParamInfo.attenMask.desc,  opParamInfo.blockTable.desc,  opParamInfo.metadata.desc,
        opParamInfo.cuSeqlensQ.desc, opParamInfo.cuSeqlensKV.desc, opParamInfo.seqUsedQ.desc,
        opParamInfo.seqUsedKV.desc,
    };
    const char *optionalNames[] = {"atten_mask",    "block_table", "metadata",  "cu_seqlens_q",
                                   "cu_seqlens_kv", "seqused_q",   "seqused_kv"};
    for (size_t i = 0; i < sizeof(optionalDescs) / sizeof(optionalDescs[0]); ++i) {
        if (optionalDescs[i] != nullptr && optionalDescs[i]->GetOriginFormat() != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(kOpName, optionalNames[i],
                                       Ops::Base::ToString(optionalDescs[i]->GetOriginFormat()).c_str(), "ND");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckBlockSize() const
{
    if (tilingInfo_.qBlockSizeVal != BSA_BLOCK_SIZE || tilingInfo_.kvBlockSizeVal != BSA_BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "q_block_size/kv_block_size",
                                  std::to_string(tilingInfo_.qBlockSizeVal) + "/" +
                                      std::to_string(tilingInfo_.kvBlockSizeVal),
                                  std::to_string(BSA_BLOCK_SIZE));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckExistence() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    // block_table 必传
    if (opParamInfo.blockTable.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "block_table", "nullptr",
                                              "block_table is required for PA execution path");
        return ge::GRAPH_FAILED;
    }
    // cu_seqlens_q 必传
    if (opParamInfo.cuSeqlensQ.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "cu_seqlens_q", "nullptr",
                                              "cu_seqlens_q is required for TND/NTD layout");
        return ge::GRAPH_FAILED;
    }
    // cu_seqlens_kv 必传
    if (opParamInfo.cuSeqlensKV.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "cu_seqlens_kv", "nullptr",
                                              "cu_seqlens_kv is required for TND/NTD layout");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckShapeConsistency() const
{
    if (tilingInfo_.bSize == 0U || tilingInfo_.n1Size == 0U || tilingInfo_.n2Size == 0U || tilingInfo_.gSize == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "bSize/n1Size/n2Size/gSize",
            std::to_string(tilingInfo_.bSize) + "/" + std::to_string(tilingInfo_.n1Size) + "/" +
                std::to_string(tilingInfo_.n2Size) + "/" + std::to_string(tilingInfo_.gSize),
            "all dimensions must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.n1Size % tilingInfo_.n2Size != 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "n1Size (query head num) ", std::to_string(tilingInfo_.n1Size),
                                              "must be divisible by n2Size (kv head num) " +
                                                  std::to_string(tilingInfo_.n2Size));
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.s1Size == 0U || tilingInfo_.s2Size == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "s1Size/s2Size", std::to_string(tilingInfo_.s1Size) + "/" + std::to_string(tilingInfo_.s2Size),
            "sequence lengths must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qbMax == 0U || tilingInfo_.kbMax == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "qbMax/kbMax", std::to_string(tilingInfo_.qbMax) + "/" + std::to_string(tilingInfo_.kbMax),
            "block counts must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.dSize == 0U || tilingInfo_.dSizeV == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "dSize/dSizeV", std::to_string(tilingInfo_.dSize) + "/" + std::to_string(tilingInfo_.dSizeV),
            "head dimensions must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.dSize != BSA_D_SIZE || tilingInfo_.dSizeV != BSA_D_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "dSize/dSizeV", std::to_string(tilingInfo_.dSize) + "/" + std::to_string(tilingInfo_.dSizeV),
            "only dSize=128 and dSizeV=128 are currently supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckKeyValueShape() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.key.shape == nullptr) {
        OP_LOGE(kOpName, "CheckKeyValueShape: key shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();

    if (opParamInfo.value.shape == nullptr) {
        OP_LOGE(kOpName, "CheckKeyValueShape: value shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &valueShape = opParamInfo.value.shape->GetStorageShape();

    if (keyShape.GetDimNum() == DIM_NUM_1) {
        // 1D combined KV storage 路径
        if (tilingInfo_.paBlockStrideVal == 0U) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "pa_block_stride", "0",
                                                  "must be greater than 0 for 1D combined KV storage");
            return ge::GRAPH_FAILED;
        }
        const int64_t keyStorageSize = keyShape.GetShapeSize();
        if (keyStorageSize <= 0 || static_cast<uint64_t>(keyStorageSize) % tilingInfo_.paBlockStrideVal != 0U) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                kOpName, "key", Ops::Base::ToString(keyShape),
                "1D combined KV storage size must be a positive multiple of pa_block_stride=" +
                    std::to_string(tilingInfo_.paBlockStrideVal));
            return ge::GRAPH_FAILED;
        }
        if (valueShape.GetDimNum() != DIM_NUM_1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "value", std::to_string(valueShape.GetDimNum()) + "D",
                                                     "1D (must match key when using 1D combined KV storage)");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 4D PA 路径
    if (keyShape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "key", std::to_string(keyShape.GetDimNum()) + "D",
                                                 "4D [blockNum, kvHeadNum, blockSize, headDim] or 1D combined KV");
        return ge::GRAPH_FAILED;
    }
    if (keyShape.GetDim(DIM_2) != static_cast<int64_t>(tilingInfo_.kvBlockSizeVal)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "key", Ops::Base::ToString(keyShape),
                                              "dim[2] (blockSize) must be " +
                                                  std::to_string(tilingInfo_.kvBlockSizeVal));
        return ge::GRAPH_FAILED;
    }
    if (keyShape.GetDim(DIM_3) != static_cast<int64_t>(BSA_D_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "key", Ops::Base::ToString(keyShape),
                                              "dim[3] (headDim) must be " + std::to_string(BSA_D_SIZE));
        return ge::GRAPH_FAILED;
    }

    if (valueShape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "value", std::to_string(valueShape.GetDimNum()) + "D",
                                                 "4D [blockNum, kvHeadNum, blockSize, valueHeadDim]");
        return ge::GRAPH_FAILED;
    }
    if (valueShape.GetDim(DIM_0) != keyShape.GetDim(DIM_0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "value", Ops::Base::ToString(valueShape),
                                              "dim[0] (blockNum) must match key dim[0]=" +
                                                  std::to_string(keyShape.GetDim(DIM_0)));
        return ge::GRAPH_FAILED;
    }
    if (valueShape.GetDim(DIM_1) != keyShape.GetDim(DIM_1)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "value", Ops::Base::ToString(valueShape),
                                              "dim[1] (kvHeadNum) must match key dim[1]=" +
                                                  std::to_string(keyShape.GetDim(DIM_1)));
        return ge::GRAPH_FAILED;
    }
    if (valueShape.GetDim(DIM_2) != keyShape.GetDim(DIM_2)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "value", Ops::Base::ToString(valueShape),
                                              "dim[2] (blockSize) must match key dim[2]=" +
                                                  std::to_string(keyShape.GetDim(DIM_2)));
        return ge::GRAPH_FAILED;
    }
    if (valueShape.GetDim(DIM_3) != static_cast<int64_t>(BSA_D_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "value", Ops::Base::ToString(valueShape),
                                              "dim[3] (valueHeadDim) must be " + std::to_string(BSA_D_SIZE));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckQuantShape() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;

    // q_descale: TND -> (T1, N1), NTD -> (N1, T1)
    if (opParamInfo.qDescale.shape != nullptr) {
        const gert::Shape &qDescaleShape = opParamInfo.qDescale.shape->GetStorageShape();
        if (qDescaleShape.GetDimNum() != DIM_NUM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "q_descale", std::to_string(qDescaleShape.GetDimNum()) + "D", "2D");
            return ge::GRAPH_FAILED;
        }
        if (tilingInfo_.layoutQValue == 2U) { // TND
            if (qDescaleShape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.qTokenNum) ||
                qDescaleShape.GetDim(DIM_1) != static_cast<int64_t>(tilingInfo_.n1Size)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "q_descale", Ops::Base::ToString(qDescaleShape),
                                                      "TND: dim[0] must be " + std::to_string(tilingInfo_.qTokenNum) +
                                                          ", dim[1] must be " + std::to_string(tilingInfo_.n1Size));
                return ge::GRAPH_FAILED;
            }
        } else { // NTD
            if (qDescaleShape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.n1Size) ||
                qDescaleShape.GetDim(DIM_1) != static_cast<int64_t>(tilingInfo_.qTokenNum)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "q_descale", Ops::Base::ToString(qDescaleShape),
                                                      "NTD: dim[0] must be " + std::to_string(tilingInfo_.n1Size) +
                                                          ", dim[1] must be " + std::to_string(tilingInfo_.qTokenNum));
                return ge::GRAPH_FAILED;
            }
        }
    }

    // k_descale: 3D (paBlockNumSum, N2, kvBlockSizeVal) 或 1D combined KV storage 切片
    if (opParamInfo.kDescale.shape != nullptr) {
        const gert::Shape &kDescaleShape = opParamInfo.kDescale.shape->GetStorageShape();
        const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();
        if (kDescaleShape.GetDimNum() == DIM_NUM_1) {
            // 1D combined KV storage 路径，仅当 key 也是 1D 时才允许
            if (keyShape.GetDimNum() != DIM_NUM_1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "k_descale",
                                                         std::to_string(kDescaleShape.GetDimNum()) + "D",
                                                         "3D (when key is 4D) or 1D (when key is 1D combined KV)");
                return ge::GRAPH_FAILED;
            }
        } else if (kDescaleShape.GetDimNum() == DIM_NUM_3) {
            if (kDescaleShape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.paBlockNumSum) ||
                kDescaleShape.GetDim(DIM_1) != static_cast<int64_t>(tilingInfo_.n2Size) ||
                kDescaleShape.GetDim(DIM_2) != static_cast<int64_t>(tilingInfo_.kvBlockSizeVal)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "k_descale", Ops::Base::ToString(kDescaleShape),
                                                      "must be (" + std::to_string(tilingInfo_.paBlockNumSum) + ", " +
                                                          std::to_string(tilingInfo_.n2Size) + ", " +
                                                          std::to_string(tilingInfo_.kvBlockSizeVal) + ")");
                return ge::GRAPH_FAILED;
            }
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                kOpName, "k_descale", std::to_string(kDescaleShape.GetDimNum()) + "D", "3D or 1D combined KV");
            return ge::GRAPH_FAILED;
        }
    }

    // v_descale: (N2,)
    if (opParamInfo.vDescale.shape != nullptr) {
        const gert::Shape &vDescaleShape = opParamInfo.vDescale.shape->GetStorageShape();
        if (vDescaleShape.GetDimNum() != DIM_NUM_1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "v_descale", std::to_string(vDescaleShape.GetDimNum()) + "D", "1D");
            return ge::GRAPH_FAILED;
        }
        if (vDescaleShape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.n2Size)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "v_descale", Ops::Base::ToString(vDescaleShape),
                                                  "dim[0] must be " + std::to_string(tilingInfo_.n2Size));
            return ge::GRAPH_FAILED;
        }
    }

    // p_scale: scalar (shapeSize == 1)
    if (opParamInfo.pScale.shape != nullptr) {
        const gert::Shape &pScaleShape = opParamInfo.pScale.shape->GetStorageShape();
        if (pScaleShape.GetShapeSize() != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "p_scale", Ops::Base::ToString(pScaleShape),
                                                  "p_scale must be a scalar (shape size == 1)");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckActualSeqLen() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;

    // cu_seqlens_q (必传): 1D, dim[0] == B+1
    if (opParamInfo.cuSeqlensQ.tensor != nullptr) {
        const auto *cuSeqlensQShape = reinterpret_cast<const gert::StorageShape *>(opParamInfo.cuSeqlensQ.tensor);
        const gert::Shape &shape = cuSeqlensQShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "cu_seqlens_q", std::to_string(shape.GetDimNum()) + "D", "1D");
            return ge::GRAPH_FAILED;
        }
        if (shape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.bSize) + 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "cu_seqlens_q", Ops::Base::ToString(shape),
                                                  "dim[0] must be B+1=" + std::to_string(tilingInfo_.bSize + 1U));
            return ge::GRAPH_FAILED;
        }
    }

    // cu_seqlens_kv (必传): 1D, dim[0] == B+1
    if (opParamInfo.cuSeqlensKV.tensor != nullptr) {
        const auto *cuSeqlensKVShape = reinterpret_cast<const gert::StorageShape *>(opParamInfo.cuSeqlensKV.tensor);
        const gert::Shape &shape = cuSeqlensKVShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "cu_seqlens_kv", std::to_string(shape.GetDimNum()) + "D", "1D");
            return ge::GRAPH_FAILED;
        }
        if (shape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.bSize) + 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "cu_seqlens_kv", Ops::Base::ToString(shape),
                                                  "dim[0] must be B+1=" + std::to_string(tilingInfo_.bSize + 1U));
            return ge::GRAPH_FAILED;
        }
    }

    // seqused_q (可选): 1D, dim[0] == B
    if (opParamInfo.seqUsedQ.tensor != nullptr) {
        const auto *seqUsedQShape = reinterpret_cast<const gert::StorageShape *>(opParamInfo.seqUsedQ.tensor);
        const gert::Shape &shape = seqUsedQShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "seqused_q", std::to_string(shape.GetDimNum()) + "D", "1D");
            return ge::GRAPH_FAILED;
        }
        if (shape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.bSize)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "seqused_q", Ops::Base::ToString(shape),
                                                  "dim[0] must be B=" + std::to_string(tilingInfo_.bSize));
            return ge::GRAPH_FAILED;
        }
    }

    // seqused_kv (可选): 1D, dim[0] == B
    if (opParamInfo.seqUsedKV.tensor != nullptr) {
        const auto *seqUsedKVShape = reinterpret_cast<const gert::StorageShape *>(opParamInfo.seqUsedKV.tensor);
        const gert::Shape &shape = seqUsedKVShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "seqused_kv", std::to_string(shape.GetDimNum()) + "D", "1D");
            return ge::GRAPH_FAILED;
        }
        if (shape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.bSize)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "seqused_kv", Ops::Base::ToString(shape),
                                                  "dim[0] must be B=" + std::to_string(tilingInfo_.bSize));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckAttenMask() const
{
    if (tilingInfo_.maskModeVal != 0U && tilingInfo_.maskModeVal != 3U) {
        OP_LOGE_WITH_INVALID_ATTR(kOpName, "mask_mode", std::to_string(tilingInfo_.maskModeVal), "0 or 3");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_.maskModeVal == 3U) {
        const auto &opParamInfo = tilingInfo_.opParamInfo;
        if (opParamInfo.attenMask.shape == nullptr ||
            opParamInfo.attenMask.shape->GetStorageShape().GetShapeSize() <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "atten_mask", "nullptr",
                                                  "atten_mask is required when mask_mode=3");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape &attenMaskShape = opParamInfo.attenMask.shape->GetStorageShape();
        if (attenMaskShape.GetDimNum() != DIM_NUM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "atten_mask", std::to_string(attenMaskShape.GetDimNum()) + "D", "2D");
            return ge::GRAPH_FAILED;
        }
        if (attenMaskShape.GetDim(DIM_0) != BSA_ATTEN_MASK_DIM_VALUE ||
            attenMaskShape.GetDim(DIM_1) != BSA_ATTEN_MASK_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "atten_mask", Ops::Base::ToString(attenMaskShape),
                                                  "must be (2048, 2048)");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::Process()
{
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckKeyValueShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckActualSeqLen() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttenMask() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
