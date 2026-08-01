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
 * \file quant_flash_attn_tiling_info.h
 * \brief
 */

#ifndef FA_TILING_INFO_H
#define FA_TILING_INFO_H

#include <vector>
#include "../common/op_host/fia_tiling_base.h"
#include "../common/op_host/fia_tiling_shape.h"

namespace optiling {

// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t QUERY_DESCALE_INDEX = 3;
constexpr uint32_t KEY_DESCALE_INDEX = 4;
constexpr uint32_t VALUE_DESCALE_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INDEX = 6;
constexpr uint32_t P_SCALE_INDEX = 7;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 8;
constexpr uint32_t CU_SEQLENS_KV_INDEX = 9;
constexpr uint32_t SEQUSED_Q_INDEX = 10;
constexpr uint32_t SEQUSED_KV_INDEX = 11;
constexpr uint32_t SINKS_INDEX = 12;
constexpr uint32_t ATTN_MASK_INDEX = 13;
constexpr uint32_t METADATA_INDEX = 14;

// Attributes Index
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 0;
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 1;  // scaleValue
constexpr uint32_t ATTR_MASK_MODE_INDEX = 2;      // mask_mode
constexpr uint32_t ATTR_WIN_LEFT_INDEX = 3;       // win_left (preToken)
constexpr uint32_t ATTR_WIN_RIGHT_INDEX = 4;      // win_right (nextToken)
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 5;   // max_seqlen_q
constexpr uint32_t ATTR_MAX_SEQLEN_KV_INDEX = 6;  // max_seqlen_kv
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 7;       // layout_q
constexpr uint32_t ATTR_LAYOUT_Q_DESCALE_INDEX = 8; // layout_q_descale
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 9;      // layout_kv
constexpr uint32_t ATTR_LAYOUT_OUT_INDEX = 10;    // layout_out
constexpr uint32_t ATTR_RETURN_LSE_INDEX = 11;    // return_softmax_lse

// Output Index
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

// Params Name
const std::string QUERY_NAME = "q";
const std::string KEY_NAME = "k";
const std::string VALUE_NAME = "v";
const std::string Q_DESCALE_NAME = "q_descale";
const std::string K_DESCALE_NAME = "k_descale";
const std::string V_DESCALE_NAME = "v_descale";
const std::string BLOCK_TABLE_NAME = "block_table";
const std::string P_SCALE_NAME = "p_scale";
const std::string CU_SEQLENS_Q_NAME = "cu_seqlens_q";
const std::string CU_SEQLENS_KV_NAME = "cu_seqlens_kv";
const std::string SEQUSED_Q_NAME = "seqused_q";
const std::string SEQUSED_KV_NAME = "seqused_kv";
const std::string SINKS_NAME = "sinks";
const std::string ATTEN_MASK_NAME = "attn_mask";
const std::string METADATA_NAME = "metadata";
const std::string QUANT_MODE_NAME = "quant_mode";
const std::string SOFTMAX_SCALE_NAME = "softmax_scale";
const std::string MASK_MODE_NAME = "mask_mode";
const std::string WIN_LEFT_NAME = "win_left";
const std::string WIN_RIGHT_NAME = "win_right";
const std::string MAX_SEQLEN_Q_NAME = "max_seqlen_q";
const std::string MAX_SEQLEN_KV_NAME = "max_seqlen_kv";
const std::string LAYOUT_Q_NAME = "layout_q";
const std::string LAYOUT_Q_DESCALE_NAME = "layout_q_descale";
const std::string LAYOUT_KV_NAME = "layout_kv";
const std::string LAYOUT_OUT_NAME = "layout_out";
const std::string RETURN_SOFTMAX_LSE_NAME = "return_softmax_lse";
const std::string ATTEN_OUT_NAME = "attn_out";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";

// 对外 quant_mode 取值，与 cann_ops_transformer/ops/quant_flash_attn.py 的 QuantMode IntEnum 对齐。
// 全部取值均通过单参数校验；其中仅 1(MXFP8)/5(MXFP4) 已实现，其余在一致性/特性交叉校验中报错。
constexpr int64_t QUANT_MODE_A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 1;
constexpr int64_t QUANT_MODE_A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP32 = 2;
constexpr int64_t QUANT_MODE_A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16 = 3;
constexpr int64_t QUANT_MODE_A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP16 = 4;
constexpr int64_t QUANT_MODE_A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16 = 5;
constexpr int64_t QUANT_MODE_A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 6;
constexpr int64_t QUANT_MODE_A8C8_QKV_HIF8_PER_TENSOR_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 7;
constexpr int64_t QUANT_MODE_A4C4_QKV_HIF4_P_HIF4_LEVEL1_SOFTMAX_FP16 = 8;
constexpr int64_t QUANT_MODE_A4C4_QKV_HIF4_P_HIF4_LEVEL2_SOFTMAX_FP16 = 9;
constexpr int64_t QUANT_MODE_A4C4_QKV_HIF4_P_HIF4_LEVEL3_SOFTMAX_FP16 = 10;
// 已实现 quant_mode 别名（语义化使用）
constexpr int64_t QUANT_MODE_MXFP8 = QUANT_MODE_A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32;
constexpr int64_t QUANT_MODE_MXFP4 = QUANT_MODE_A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16;

constexpr int64_t MASK_MODE_INT_MAX = 2147483647;

enum class MaskMode : int32_t {
    NO_MASK = 0,
    CAUSAL = 3,
    BAND = 4
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

enum class QuantMode : uint32_t {
    GROUP_SCALING = 3,
    PER_BLOCK = 4
};

enum class QFA_DTYPE : uint32_t {
    FP8_E4M3 = 1,
    FP8_E8M0 = 2,
    HI_FLOAT8 = 3,
    FP4_E2M1 = 11,
    HI_FLOAT4 = 12,
};

struct QfaPlatFormInfo {
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

struct FARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct FAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct FAParaInfo {
    FARequiredParaInfo query = {nullptr, nullptr};
    FARequiredParaInfo key = {nullptr, nullptr};
    FARequiredParaInfo value = {nullptr, nullptr};
    FARequiredParaInfo qDescale = {nullptr, nullptr};
    FARequiredParaInfo kDescale = {nullptr, nullptr};
    FARequiredParaInfo vDescale = {nullptr, nullptr};

    FAOptionalParaInfo blockTable = {nullptr, nullptr};
    FAOptionalParaInfo pScale = {nullptr, nullptr};
    FAOptionalParaInfo cuSeqlensQ = {nullptr, nullptr};
    FAOptionalParaInfo cuSeqlensKv = {nullptr, nullptr};
    FAOptionalParaInfo sequsedQ = {nullptr, nullptr};
    FAOptionalParaInfo sequsedKv = {nullptr, nullptr};
    FAOptionalParaInfo sinks = {nullptr, nullptr};
    FAOptionalParaInfo attnMask = {nullptr, nullptr};
    FAOptionalParaInfo metadata = {nullptr, nullptr};

    const int64_t *quantMode = nullptr;
    const float *softmaxScale = nullptr;
    const int64_t *maskMode = nullptr;
    const int64_t *winLeft = nullptr;
    const int64_t *winRight = nullptr;
    const int64_t *maxSeqlenQ = nullptr;
    const int64_t *maxSeqlenKV = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutQDescale = nullptr;
    const char *layoutKV = nullptr;
    const char *layoutOut = nullptr;
    const bool *returnSoftMaxLse = nullptr;

    FARequiredParaInfo attnOut = {nullptr, nullptr};
    FARequiredParaInfo lseOut = {nullptr, nullptr};
};

class QuantFlashAttnTilingInfo : public TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    FAParaInfo opParamInfo;

    // BaseParams
    int64_t bSize = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t gSize = 0;
    int64_t qkHeadDim = 0;
    int64_t vHeadDim = 0;
    int64_t queryTSize = 0;
    int64_t keyTSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    KvStorageMode kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;
    QuantMode qQuantMode = QuantMode::GROUP_SCALING;
    QuantMode kQuantMode = QuantMode::GROUP_SCALING;
    QuantMode vQuantMode = QuantMode::GROUP_SCALING;
    float softmaxScale = 0.0;

    // PageAttention
    int64_t blockSize = 0;
    int64_t maxBlockNumPerBatch = 0;

    // mask 信息
    bool attnMaskFlag = false;
    uint32_t maskMode = 0;
    int64_t winLeft = -1;
    int64_t winRight = -1;

    // layout信息
    FiaLayout layoutQ;
    FiaLayout layoutQDescale = FiaLayout::BSND;
    FiaLayout layoutKV;
    FiaLayout layoutOut;

    // seqLen信息
    int64_t maxSeqLenQ = 0;
    int64_t maxSeqLenKv = 0;
    uint32_t qSeqUsedSize = 0;
    uint32_t kvSeqUsedSize = 0;
    uint32_t qCuSeqLensSize = 0;
    uint32_t kvCuSeqLensSize = 0;

    // learnable sink 信息
    bool learnableSinkFlag = false;
    uint32_t returnSoftmaxLse = 0;
    uint32_t softmaxPresision = 1;
    uint32_t quantBlockSizeQs = 0;
    uint32_t quantBlockSizeKs = 0;
    uint32_t quantBlockSizeVs = 0;

    // DTYPE
    // ge::DT_FLOAT8_E8M0
    // ge::DT_FLOAT8_E4M3FN
    // ge::DT_FLOAT4_E2M1
    // ge::DT_HIFLOAT8
    ge::DataType inputQType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType inputKType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType inputVType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType outputType = ge::DT_BF16;
};
} // namespace optiling
#endif // FA_TILING_INFO_H