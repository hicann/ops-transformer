/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_selected_attention_infer_tiling_base.h
 * \brief
 */
#ifndef NSA_SELECTED_ATTENTION_INFER_BASE_DEFINE_H
#define NSA_SELECTED_ATTENTION_INFER_BASE_DEFINE_H

#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
using namespace ge;
namespace optiling {

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

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t TOPK_INDICES_INPUT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INPUT_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 5;
constexpr uint32_t ACT_Q_SEQ_LEN_INPUT_INDEX = 6;
constexpr uint32_t ACT_KV_SEQ_LEN_INPUT_INDEX = 7;

constexpr uint32_t PER_TOKEN_Split_B = 0;
constexpr uint32_t PER_TOKEN_Split_S = 1;
constexpr uint32_t BNSD_B_IDX = 0;
constexpr uint32_t BNSD_N_IDX = 1;
constexpr uint32_t BNSD_S_IDX = 2;
constexpr uint32_t BNSD_D_IDX = 3;
constexpr uint32_t BSND_B_IDX = 0;
constexpr uint32_t BSND_S_IDX = 1;
constexpr uint32_t BSND_N_IDX = 2;
constexpr uint32_t BSND_D_IDX = 3;
constexpr uint32_t BSH_B_IDX = 0;
constexpr uint32_t BSH_S_IDX = 1;
constexpr uint32_t BSH_H_IDX = 2;
constexpr uint32_t DIM_BH = 2;
constexpr uint32_t BH_B_IDX = 0;
constexpr uint32_t BH_H_IDX = 1;
constexpr uint32_t BND_B_IDX = 0;
constexpr uint32_t BND_N_IDX = 1;
constexpr uint32_t BND_D_IDX = 2;
constexpr uint32_t OUTPUT_INDEX = 0;
constexpr uint32_t LAYOUT_ATTR_INDEX = 0;
constexpr uint32_t NUM_HEADS_ATTR_INDEX = 1;
constexpr uint32_t KV_NUM_HEADS_ATTR_INDEX = 2;
constexpr uint32_t SELECTED_BLOCK_SIZE_ATTR_INDEX = 3;
constexpr uint32_t SELECTED_BLOCK_COUNT_ATTR_INDEX = 4;
constexpr uint32_t BLOCK_SIZE_ATTR_INDEX = 5;
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 6;
constexpr uint32_t SPARSE_MODE_ATTR_INDEX = 7;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t MAX_BLOCK_SIZE = 512;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
}// namespace optiling


#endif // NSA_SELECTED_ATTENTION_INFER_BASE_DEFINE_H