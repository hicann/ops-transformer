/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file compressor_grad_tiling.h
 * \brief
 */

#ifndef COMPRESSOR_GRAD_TILING_H_
#define COMPRESSOR_GRAD_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;

constexpr uint32_t D_CMP_KV_INPUT_INDEX = 3;

constexpr uint32_t SOFTMAX_SCORE_INPUT_INDEX = 4;

// INPUT(OPTION)
constexpr uint32_t KV_INPUT_INDEX = 5;

constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 6;
constexpr uint32_t SEQ_USED_INPUT_INDEX = 7;
constexpr uint32_t START_POS_INPUT_INDEX = 8;

// ATTR
constexpr uint32_t CMP_RATIO_ATTR_INDEX = 0;
constexpr uint32_t COFF_ATTR_INDEX = 1;

// OUTPUT
constexpr uint32_t D_X_OUTPUT_INDEX = 0;
constexpr uint32_t D_WKV_OUTPUT_INDEX = 1;
constexpr uint32_t D_WGATE_OUTPUT_INDEX = 2;
constexpr uint32_t D_APE_OUTPUT_INDEX = 3;

// ATTR DEFAULT VALUE
constexpr uint32_t CMP_RATIO_VALUE = 4;
constexpr uint32_t COFF_VALUE = 1;

constexpr uint32_t COMPRESSOR_GRAD_DIM_NUM_1 = 1;
constexpr uint32_t COMPRESSOR_GRAD_DIM_NUM_2 = 2;
constexpr uint32_t COMPRESSOR_GRAD_DIM_NUM_3 = 3;
constexpr uint32_t COMPRESSOR_GRAD_DIM_NUM_4 = 4;
constexpr uint32_t COMPRESSOR_GRAD_DIM_INDEX_0 = 0;
constexpr uint32_t COMPRESSOR_GRAD_DIM_INDEX_1 = 1;
constexpr uint32_t COMPRESSOR_GRAD_DIM_INDEX_2 = 2;
constexpr uint32_t COMPRESSOR_GRAD_DIM_INDEX_3 = 3;

// CONSTRAINTS
constexpr uint32_t MAX_HIDDEN_SIZE = 10240;
constexpr uint32_t MIN_HIDDEN_SIZE = 1024;
constexpr uint32_t ALIGN_FACTOR_HIDDEN_SIZE = 512;
constexpr uint32_t MIN_CMP_RATIO = 2;
constexpr uint32_t MAX_CMP_RATIO = 128;

constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
const uint32_t CMP_MAX_AIC_CORE_NUM = 36;

static const std::string X_NAME = "query";
static const std::string WKV_NAME = "wkv";
static const std::string WGATE_NAME = "wgate";
static const std::string D_CMP_KV_NAME = "d_cmp_kv";
static const std::string SOFTMAX_SCORE_NAME = "softmax_score";
static const std::string KV_NAME = "kv";
static const std::string CU_SEQLENS_NAME = "cu_seqlens";
static const std::string SEQUSED_NAME = "seq_used";
static const std::string START_POS_NAME = "start_pos";
static const std::string CMP_RATIO_NAME = "cmp_ratio";
static const std::string COFF_NAME = "coff";
static const std::string D_X_NAME = "d_x";
static const std::string D_WKV_NAME = "d_wkv";
static const std::string D_WGATE_NAME = "d_wgate";
static const std::string D_APE_NAME = "d_ape";

static std::string DataTypeToSerialString(ge::DataType type);

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {X_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {WKV_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {WGATE_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {D_CMP_KV_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {SOFTMAX_SCORE_NAME, {ge::DT_FLOAT}},
    {KV_NAME, {ge::DT_FLOAT}},
    {CU_SEQLENS_NAME, {ge::DT_INT32}},
    {SEQUSED_NAME, {ge::DT_INT32}},
    {START_POS_NAME, {ge::DT_INT32}},
    {D_X_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {D_WKV_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {D_WGATE_NAME, {ge::DT_BF16, ge::DT_FLOAT16}},
    {D_APE_NAME, {ge::DT_FLOAT}}};

const std::map<std::string, std::vector<uint32_t>> DIM_NUM_MAP = {
    {X_NAME, {COMPRESSOR_GRAD_DIM_NUM_2, COMPRESSOR_GRAD_DIM_NUM_3}},
    {WKV_NAME, {COMPRESSOR_GRAD_DIM_NUM_2}},
    {WGATE_NAME, {COMPRESSOR_GRAD_DIM_NUM_2}},
    {D_CMP_KV_NAME, {COMPRESSOR_GRAD_DIM_NUM_2, COMPRESSOR_GRAD_DIM_NUM_3}},
    {SOFTMAX_SCORE_NAME, {COMPRESSOR_GRAD_DIM_NUM_3, COMPRESSOR_GRAD_DIM_NUM_4}},
    {KV_NAME, {COMPRESSOR_GRAD_DIM_NUM_3, COMPRESSOR_GRAD_DIM_NUM_4}},
    {CU_SEQLENS_NAME, {COMPRESSOR_GRAD_DIM_NUM_1}},
    {SEQUSED_NAME, {COMPRESSOR_GRAD_DIM_NUM_1}},
    {START_POS_NAME, {COMPRESSOR_GRAD_DIM_NUM_1}},
    {D_X_NAME, {COMPRESSOR_GRAD_DIM_NUM_2, COMPRESSOR_GRAD_DIM_NUM_3}},
    {D_WKV_NAME, {COMPRESSOR_GRAD_DIM_NUM_2}},
    {D_WGATE_NAME, {COMPRESSOR_GRAD_DIM_NUM_2}},
    {D_APE_NAME, {COMPRESSOR_GRAD_DIM_NUM_2}}};

static const std::map<std::string, uint32_t> LAYOUT_DIM_MAP = {
    {"BSH", COMPRESSOR_GRAD_DIM_NUM_3},
    {"TH", COMPRESSOR_GRAD_DIM_NUM_2},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    const gert::Tensor *tensor;
};

const std::vector<uint32_t> COFF{1, 2};
const std::vector<uint32_t> HEAD_DIM{128, 512};

enum class LayoutType {
    LAYOUT_BSH,
    LAYOUT_TH
};

enum class TemplateId : uint8_t {
    NORMAL = 0,
    EMPTY_X = 1,
    FULL_LOAD = 2
};

struct CompressorGradContext {
    const char *opName;
    const char *opType;
    fe::PlatFormInfos *platformInfo;

    RequiredParaInfo x;
    RequiredParaInfo wkv;
    RequiredParaInfo wgate;
    RequiredParaInfo dCmpKv;
    RequiredParaInfo softmaxScore;
    OptionalParaInfo kv;
    OptionalParaInfo cuSeqlens;
    OptionalParaInfo seqUsed;
    OptionalParaInfo startPos;
    RequiredParaInfo dX;
    RequiredParaInfo dWkv;
    RequiredParaInfo dWgate;
    RequiredParaInfo dApe;

    const uint32_t *coff;
    const uint32_t *cmpRatio;
    TemplateId templateId;

    ge::DataType dtype = ge::DT_BF16;
    LayoutType layout = LayoutType::LAYOUT_BSH;

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

struct CompressorGradSplitCoreParams {
    uint32_t mStart;
    uint32_t mEnd;
    uint32_t nStart;
    uint32_t nEnd;
    uint32_t kStart;
    uint32_t kEnd;
};

// 1. 基础参数结构体
struct CompressorGradBaseParams {
    uint32_t batchSize = 0;            // bastch size（批大小）
    uint32_t seqSize = 0;              // sequence size（kvs大小）
    uint32_t hiddenSize = 0;           // hidden size（隐藏层大小）
    uint32_t tokenSize = 0;            // token size = batchSize * seqSize(token总数：批大小x序列1长度)
    uint32_t headDim = 0;              // head size of kv
    uint32_t featureDim = 0;           // head size of kv
    uint32_t csSize = 0;               // Compress sequence len
    uint32_t cmpRatio = 4;             // Compress ratio
    uint32_t usedCoreNum = 0;          // 使用核数
    uint32_t nSize = 0;                // 预留字段（当前未参与 tiling 决策）
    uint64_t stateCacheStrideDim0 = 0; // stateCache第0维的stride
    uint32_t kBaseNum = 0;
    uint32_t kBaseSize = 0;
    uint32_t coreGroupNum = 0;
    uint32_t mLoopNum = 0;
    CompressorGradSplitCoreParams splitCoreParam[CMP_MAX_AIC_CORE_NUM];
};

struct CompressorGradInnerSplitParams {
    uint32_t mBaseSize;
    uint32_t dBaseSize;
};

struct CompressorGradWorkspaceParams {
    uint32_t mm1KvResSize;
    uint32_t mm1ScoreResSize;
    uint32_t vec1ResSize;
    uint32_t vec1TailCacheSize;
    uint32_t dbWorkspaceRatio = 1;
};

BEGIN_TILING_DATA_DEF(CompressorGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch_size)
TILING_DATA_FIELD_DEF(int64_t, token_size)
TILING_DATA_FIELD_DEF(int64_t, seq_size)
TILING_DATA_FIELD_DEF(int64_t, cmp_ratio)
TILING_DATA_FIELD_DEF(int64_t, hidden_size)
TILING_DATA_FIELD_DEF(int64_t, head_dim)
TILING_DATA_FIELD_DEF(int64_t, cube_core_num)
TILING_DATA_FIELD_DEF(int64_t, core_num)
TILING_DATA_FIELD_DEF(int64_t, total_head_dim)
TILING_DATA_FIELD_DEF(int64_t, cmp_row_cnt)
TILING_DATA_FIELD_DEF(int64_t, cmp_size)
TILING_DATA_FIELD_DEF(int64_t, cmp_kv_batch_stride)
TILING_DATA_FIELD_DEF(int64_t, cmp_kv_rows)
TILING_DATA_FIELD_DEF(int64_t, x_rows)
TILING_DATA_FIELD_DEF(int64_t, group_size)
TILING_DATA_FIELD_DEF(int64_t, group_num)
TILING_DATA_FIELD_DEF(int64_t, group_deal_sc_num)
TILING_DATA_FIELD_DEF(int64_t, deal_sc_num)
TILING_DATA_FIELD_DEF(int64_t, total_sc_num_per_round)
TILING_DATA_FIELD_DEF(int64_t, db_row_cnt)
TILING_DATA_FIELD_DEF(int64_t, group_row_stride)
TILING_DATA_FIELD_DEF(int64_t, coff_coef)
TILING_DATA_FIELD_DEF(int64_t, cube_m_base_size)
TILING_DATA_FIELD_DEF(int64_t, d_deal_size)
TILING_DATA_FIELD_DEF(int64_t, m_deal_size)
TILING_DATA_FIELD_DEF(int64_t, dape_ws_size)
TILING_DATA_FIELD_DEF(int64_t, dx_ws_size)
TILING_DATA_FIELD_DEF(int64_t, d_weight_ws_size)
TILING_DATA_FIELD_DEF(int64_t, x_ws_size)
TILING_DATA_FIELD_DEF(int64_t, dx_cache_ws_size)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(CompressorGrad, CompressorGradTilingData)

struct CompressorGradCompileInfo {
    int64_t core_num;
};

} // namespace optiling

#endif // COMPRESSOR_GRAD_TILING_H_
