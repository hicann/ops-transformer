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
 * \file qmla_tiling_info_parser.h
 * \brief QuantFlashMlaWithKvcache tiling info parser
 */

#pragma once

#include <map>
#include <string>
#include "tiling/tiling_api.h"
#include "qmla_tiling_info.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

// 输入索引
constexpr size_t QUERY_INDEX = 0;
constexpr size_t K_CACHE_INDEX = 1;
constexpr size_t Q_DESCALE_INDEX = 2;
constexpr size_t K_DESCALE_INDEX = 3;
constexpr size_t BLOCK_TABLE_INDEX = 4;
constexpr size_t CACHE_SEQLENS_INDEX = 5;
constexpr size_t CU_SEQLENS_Q_INDEX = 6;
constexpr size_t SEQUSED_Q_INDEX = 7;
constexpr size_t ATTN_MASK_INDEX = 8;
constexpr size_t METADATA_INDEX = 9;

// 输出索引
constexpr size_t ATTN_OUT_INDEX = 0;
constexpr size_t SOFTMAX_LSE_INDEX = 1;

// 属性索引
constexpr size_t ATTR_QUANT_MODE_INDEX = 0;
constexpr size_t ATTR_SOFTMAX_SCALE_INDEX = 1;
constexpr size_t ATTR_MASK_MODE_INDEX = 2;
constexpr size_t ATTR_MAX_SEQLEN_Q_INDEX = 3;
constexpr size_t ATTR_MAX_SEQLEN_KV_INDEX = 4;
constexpr size_t ATTR_HEAD_DIM_V_INDEX = 5;
constexpr size_t ATTR_LAYOUT_Q_INDEX = 6;
constexpr size_t ATTR_LAYOUT_KV_INDEX = 7;
constexpr size_t ATTR_LAYOUT_OUT_INDEX = 8;
constexpr size_t ATTR_RETURN_SOFTMAX_LSE_INDEX = 9;

// 参数名
constexpr const char* QUERY_NAME = "q";
constexpr const char* K_CACHE_NAME = "k_cache";
constexpr const char* Q_DESCALE_NAME = "q_descale";
constexpr const char* K_DESCALE_NAME = "k_descale";
constexpr const char* BLOCK_TABLE_NAME = "block_table";
constexpr const char* CACHE_SEQLENS_NAME = "cache_seqlens";
constexpr const char* CU_SEQLENS_Q_NAME = "cu_seqlens_q";
constexpr const char* SEQUSED_Q_NAME = "seqused_q";
constexpr const char* ATTN_MASK_NAME = "attn_mask";
constexpr const char* METADATA_NAME = "metadata";
constexpr const char* ATTN_OUT_NAME = "attn_out";
constexpr const char* SOFTMAX_LSE_NAME = "softmax_lse";

class QmlaInfoParser {
public:
    explicit QmlaInfoParser(const gert::TilingContext* context)
        : context_(context)
    {}
    ~QmlaInfoParser() = default;

    ge::graphStatus Parse(QmlaTilingInfo& qmlaInfo);

private:
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus GetEmptyTensorFlag();
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetInAndOutLayout();
    ge::graphStatus GetQuantMode();
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus ParseFeatureInfo();
    void GenerateInfo(QmlaTilingInfo& qmlaInfo);

    // 轴解析
    ge::graphStatus GetQShapeInfo();       // bSize/n1Size/s1Size/qTSize/headDimQk
    ge::graphStatus GetKvCacheShapeInfo(); // blockSize/blockNum/n2Size校验
    ge::graphStatus GetS2Size();

    const gert::TilingContext* context_ = nullptr;
    const char* opName_ = nullptr;
    QmlaParaInfo opParamInfo_;

    // 轴
    int64_t bSize_ = 0;
    int64_t n1Size_ = 0;
    int64_t n2Size_ = 1;
    int64_t gSize_ = 1;
    int64_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t headDimQk_ = 576;
    int64_t headDimV_ = 512;
    int64_t qTSize_ = 0;
    int64_t t2Size_ = 0;

    // PA
    int64_t blockSize_ = 0;
    int64_t blockNum_ = 0;
    int64_t maxBlockNumPerBatch_ = 0;

    // seq
    bool cuSeqLenQFlag_ = false;
    bool seqUsedQFlag_ = false;
    int64_t maxSeqQ_ = -1;
    int64_t maxSeqKv_ = -1;

    // mask
    int64_t maskMode_ = 0;
    bool attnMaskFlag_ = false;
    int64_t attenMaskS1Size_ = 0;
    int64_t attenMaskS2Size_ = 0;

    // quant
    QmlaQuantMode quantMode_ = QmlaQuantMode::MLA_FP8_E4M3_FULLQUANT;
    float softmaxScale_ = -1.0f;

    // layout
    QmlaLayout layoutQ_ = QmlaLayout::BSND;
    QmlaOutLayout layoutOut_ = QmlaOutLayout::BSND;
    QmlaKvLayout layoutKv_ = QmlaKvLayout::PA_BNBD;

    // strides (for non-contiguous k_cache check)
    const gert::Stride* keyStrides_ = nullptr;
    bool hasStride_ = false;

    // 特性
    bool returnSoftmaxLse_ = false;
    bool metadataFlag_ = false;
    bool emptyTensorFlag_ = false;

    // dtype
    ge::DataType qType_ = ge::DT_FLOAT8_E4M3FN;
    ge::DataType kvType_ = ge::DT_FLOAT8_E4M3FN;
    ge::DataType outType_ = ge::DT_BF16;
};

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
