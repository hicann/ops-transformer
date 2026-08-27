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
 * \file qfa_tiling_info.h
 * \brief QuantFlashAttn tiling info definitions
 */

#ifndef QUANT_FLASH_ATTN_QFA_TILING_INFO_H
#define QUANT_FLASH_ATTN_QFA_TILING_INFO_H

#include <map>
#include <set>
#include <string>
#include <vector>
#include <cstdint>
#include "../../common/op_host/fia_tiling_base.h"

namespace optiling {
namespace quant_flash_attn {

// ============================================================
// String name constants
// ============================================================
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
const std::string METADATA_NAME = "metadata";
const std::string ATTN_MASK_NAME = "attn_mask";
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
const std::string QUANT_MODE_NAME = "quant_compute_mode";
const std::string RETURN_SOFTMAX_LSE_NAME = "return_softmax_lse";
const std::string ATTN_OUT_NAME = "attn_out";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";

// ============================================================
// Input / Attribute / Output Index Constants
// ============================================================

// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t Q_DESCALE_INDEX = 3;
constexpr uint32_t K_DESCALE_INDEX = 4;
constexpr uint32_t V_DESCALE_INDEX = 5;
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
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 1;
constexpr uint32_t ATTR_MASK_MODE_INDEX = 2;
constexpr uint32_t ATTR_WIN_LEFT_INDEX = 3;
constexpr uint32_t ATTR_WIN_RIGHT_INDEX = 4;
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 5;
constexpr uint32_t ATTR_MAX_SEQLEN_KV_INDEX = 6;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 7;
constexpr uint32_t ATTR_LAYOUT_Q_DESCALE_INDEX = 8;
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 9;
constexpr uint32_t ATTR_LAYOUT_OUT_INDEX = 10;
constexpr uint32_t ATTR_RETURN_LSE_INDEX = 11;

// Output Index
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

constexpr int64_t MASK_MODE_INT_MAX = 2147483647;

// ============================================================
// Enums
// ============================================================

enum class MaskMode : int32_t {
    NO_MASK = 0,
    CAUSAL = 3,
    SLIDING_WINDOW = 4
};

enum class QfaLayout : uint32_t {
    BSND = 0,
    BNSD = 1,
    TND = 2,
    PA_BBND = 3,
    PA_BNBD = 4,
    PA_NZ = 5,
    LSE_BNS = 6,
    LSE_NT = 7,
    N2TGD = 8,
    NTD = 9,
    NT = 10
};

const std::map<std::string, QfaLayout> qfaLayoutMap = {
    {"BSND", QfaLayout::BSND},       {"BNSD", QfaLayout::BNSD},       {"TND", QfaLayout::TND},
    {"PA_BBND", QfaLayout::PA_BBND}, {"PA_BNBD", QfaLayout::PA_BNBD}, {"PA_NZ", QfaLayout::PA_NZ},
    {"N2TGD", QfaLayout::N2TGD},     {"NTD", QfaLayout::NTD},         {"NT", QfaLayout::NT}};

enum class QfaAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    H = 4,
    T = 5,
    D1 = 6,
    D0 = 7,
    S1 = 8,
    S2 = 9,
    Bn = 10,
    Bs = 11,
    CONST = 12
};

enum class QfaQuantMode : uint32_t {
    A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 1,
    A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 6,
    A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32 = 0
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

// ============================================================
// Function declarations
// ============================================================

std::string QfaLayoutToSerialString(QfaLayout layout);
std::string QfaAxisToSerialString(QfaAxis axis);
std::string QfaQuantModeToSerialString(QfaQuantMode qfaQuantMode);

// ============================================================
// Structs
// ============================================================

struct QfaRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct QfaOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct QfaParaInfo {
    QfaRequiredParaInfo query = {nullptr, nullptr};
    QfaRequiredParaInfo key = {nullptr, nullptr};
    QfaRequiredParaInfo value = {nullptr, nullptr};
    QfaRequiredParaInfo qDescale = {nullptr, nullptr};
    QfaRequiredParaInfo kDescale = {nullptr, nullptr};
    QfaRequiredParaInfo vDescale = {nullptr, nullptr};

    QfaOptionalParaInfo blockTable = {nullptr, nullptr};
    QfaOptionalParaInfo pScale = {nullptr, nullptr};
    QfaOptionalParaInfo cuSeqlensQ = {nullptr, nullptr};
    QfaOptionalParaInfo cuSeqlensKv = {nullptr, nullptr};
    QfaOptionalParaInfo sequsedQ = {nullptr, nullptr};
    QfaOptionalParaInfo sequsedKv = {nullptr, nullptr};
    QfaOptionalParaInfo sinks = {nullptr, nullptr};
    QfaOptionalParaInfo metadata = {nullptr, nullptr};
    QfaOptionalParaInfo attnMask = {nullptr, nullptr};

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

    QfaRequiredParaInfo attnOut = {nullptr, nullptr};
    QfaRequiredParaInfo lseOut = {nullptr, nullptr};
};

// ============================================================
// QfaTilingInfo class
// ============================================================

class QfaTilingInfo : public TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    QfaParaInfo opParamInfo;

    // Base Param
    int64_t bSize = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t qkHeadDim = 0;
    int64_t vHeadDim = 0;
    int64_t gSize = 0;
    int64_t qTSize = 0;
    int64_t kTSize = 0;
    float softmaxScale = 0;

    uint64_t totalOutputSize = 0;
    uint64_t totalLseSize = 0;

    // Quant Param
    QfaQuantMode quantMode = QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32;

    // PageAttention
    bool pageAttentionFlag = false;
    int64_t blockSize = 0;
    int64_t blockTypeSize = 0;
    int64_t maxBlockNumPerBatch = 0;
    int64_t totalBlockNum = 0;

    // Q seq_lens
    int64_t seqUsedQDims = 0;
    int64_t cuSeqLenQDims = 0;
    int64_t maxSeqQ = 0;
    int64_t maxSeqKv = 0;
    int64_t maskMode = 0;
    int64_t winLeft = 0;
    int64_t winRight = 0;

    // Others Flag
    bool batchContinuousFlag = true;
    bool softmaxLseFlag = false;
    bool sinksFlag = false;
    bool emptyTensorFlag = false;

    // DType
    ge::DataType inputQType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType inputKvType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType outputType = ge::DT_BF16;
    ge::DataType qDescaleType = ge::DT_FLOAT8_E8M0;
    ge::DataType kDescaleType = ge::DT_FLOAT8_E8M0;
    ge::DataType vDescaleType = ge::DT_FLOAT8_E8M0;

    // Layout
    QfaLayout qLayout = QfaLayout::BSND;
    QfaLayout outLayout = QfaLayout::BSND;
    QfaLayout kvLayout = QfaLayout::BSND;
    QfaLayout layoutQDescale = QfaLayout::BSND;

    // Strides (for non-contiguous tensor check)
    const gert::Stride *keyStrides = nullptr;
    const gert::Stride *valueStrides = nullptr;
    const gert::Stride *kDescaleStrides = nullptr;
    const gert::Stride *vDescaleStrides = nullptr;
    bool hasStride = false;
};

// ============================================================
// arch35QFA namespace: shape limits, dtype maps, tiling constants
// ============================================================

namespace arch35QFA {
constexpr uint32_t B_LIMIT = 65536;
constexpr uint32_t N1_LIMIT = 256;
constexpr uint32_t N2_LIMIT = 256;
constexpr uint32_t G_LIMIT = 64;
constexpr uint32_t D_LIMIT = 512;
constexpr uint32_t T_LIMIT = 1048576;
constexpr uint32_t H_LIMIT = 65535;
constexpr uint32_t S_LIMIT = 20971520;

constexpr uint32_t INPUT_Q_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_Q_SHAPE_MAX_DIMS = 4;
constexpr uint32_t INPUT_KV_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_KV_SHAPE_MAX_DIMS = 5;

constexpr uint32_t DIM_NUM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_NUM_5 = 5;
constexpr uint32_t DIM_NUM_6 = 6;

constexpr uint32_t BLOCK_SIZE_MAX = 512;
constexpr uint32_t BLOCK_SIZE_ALIGN_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_ALIGN_SIZE_128 = 128;

constexpr uint32_t MASK_DIM_SS = 2;
constexpr uint32_t MASK_DIM_BSS = 3;
constexpr uint32_t MASK_DIM_B1SS = 4;

constexpr uint32_t PER_CHANNEL_MODE = 0;
constexpr uint32_t PER_TOKEN_MODE = 1;
constexpr uint32_t PER_TENSOR_HEAD_MODE = 2;
constexpr uint32_t PER_TOKEN_HEAD_MODE = 3;
constexpr uint32_t PER_TOKEN_PA_MODE = 4;
constexpr uint32_t PER_TOKEN_HEAD_PA_MODE = 5;
constexpr uint32_t PER_TOKEN_GROUP_MODE = 6;
constexpr uint32_t PER_BLOCK_MODE = 7;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr int64_t SHAPE_PARAMS_CONST = 1;
constexpr int64_t SHAPE_NUM_ONE = 1;

constexpr uint32_t FLOAT16SIZE = 2;
constexpr uint32_t BFLOAT16SIZE = 2;
constexpr uint32_t INT8SIZE = 1;
constexpr uint32_t FLOAT8SIZE = 1;
constexpr float INT4SIZE = 0.5f;
constexpr float FLOAT4SIZE = 0.5f;

constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {{ge::DT_UNDEFINED, "DT_UNDEFINED"},
                                                                    {ge::DT_FLOAT, "DT_FLOAT"},
                                                                    {ge::DT_FLOAT16, "DT_FLOAT16"},
                                                                    {ge::DT_INT8, "DT_INT8"},
                                                                    {ge::DT_INT16, "DT_INT16"},
                                                                    {ge::DT_UINT16, "DT_UINT16"},
                                                                    {ge::DT_UINT8, "DT_UINT8"},
                                                                    {ge::DT_INT32, "DT_INT32"},
                                                                    {ge::DT_INT64, "DT_INT64"},
                                                                    {ge::DT_UINT32, "DT_UINT32"},
                                                                    {ge::DT_UINT64, "DT_UINT64"},
                                                                    {ge::DT_BOOL, "DT_BOOL"},
                                                                    {ge::DT_DOUBLE, "DT_DOUBLE"},
                                                                    {ge::DT_BF16, "DT_BFLOAT16"},
                                                                    {ge::DT_INT4, "DT_INT4"},
                                                                    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
                                                                    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
                                                                    {ge::DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0FN"},
                                                                    {ge::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"}};

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME, {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8}},
    {KEY_NAME, {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8}},
    {VALUE_NAME, {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8}},
    {Q_DESCALE_NAME, {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT8_E8M0}},
    {K_DESCALE_NAME, {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT8_E8M0}},
    {V_DESCALE_NAME, {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT8_E8M0}},
    {BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {P_SCALE_NAME, {ge::DT_FLOAT}},
    {ATTN_MASK_NAME, {ge::DT_INT8}},
    {ATTN_OUT_NAME, {ge::DT_BF16}},
    {SOFTMAX_LSE_NAME, {ge::DT_FLOAT}},
};

const std::set<ge::Format> FORMAT_SUPPORT_SET = {ge::FORMAT_ND};
} // namespace arch35QFA

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_QFA_TILING_INFO_H
