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
 * \file qmla_tiling_info.h
 * \brief QuantFlashMlaWithKvcache tiling info 结构定义
 */

#ifndef QMLA_TILING_INFO_H
#define QMLA_TILING_INFO_H

#include <string>
#include "tiling/tiling_api.h"
#include "../../common/op_host/fia_tiling_base.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

// MLA 约束常量
constexpr int64_t QMLA_HEAD_DIM_QK = 576; // nope 512 + rope 64
constexpr int64_t QMLA_HEAD_DIM_V = 512;
constexpr int64_t QMLA_KV_N = 1;
constexpr int64_t QMLA_MASK_MODE_NO_MASK = 0;
constexpr int64_t QMLA_MASK_MODE_CAUSAL = 3;

// 输入layout（q）
enum class QmlaLayout : int64_t {
    BSND = 0,
    BNSD = 1,
    TND = 2,
};

// 输出layout（attn_out）
enum class QmlaOutLayout : int64_t {
    BSND = 0,
    BNSD = 1,
    TND = 2,
    NTD = 3,
};

// KV cache layout
enum class QmlaKvLayout : int64_t {
    PA_BBND = 0,
    PA_BNBD = 1,
    PA_NZ = 2,
};

// MLA量化模式
enum class QmlaQuantMode : int64_t {
    MLA_FP8_E4M3_FULLQUANT = 1,
};

struct QmlaTensorInfo {
    const gert::Tensor* tensor = nullptr;
    const gert::CompileTimeTensorDesc* desc = nullptr;
    const gert::StorageShape* shape = nullptr;
};

struct QmlaParaInfo {
    // 必选输入
    QmlaTensorInfo query;
    QmlaTensorInfo kCache;
    QmlaTensorInfo qDescale;
    QmlaTensorInfo kDescale;
    QmlaTensorInfo blockTable;
    QmlaTensorInfo cacheSeqlens;
    // 可选输入
    QmlaTensorInfo cuSeqlensQ; // cu_seq: Q累积序列长度(B+1), layout_q为TND时必传
    QmlaTensorInfo sequsedQ;   // seq_used: 每batch实际使用的Q长度(B)
    QmlaTensorInfo attnMask;
    QmlaTensorInfo metadata;
    // 输出
    QmlaTensorInfo attnOut;
    QmlaTensorInfo softmaxLse;
    // 属性
    const int64_t* quantMode = nullptr;
    const float* softmaxScale = nullptr;
    const int64_t* maskMode = nullptr;
    const int64_t* maxSeqlenQ = nullptr;
    const int64_t* maxSeqlenKv = nullptr;
    const int64_t* headDimV = nullptr;
    const char* layoutQ = nullptr;
    const char* layoutKv = nullptr;
    const char* layoutOut = nullptr;
    const bool* returnSoftmaxLse = nullptr;
};

struct QmlaTilingInfo : public TilingInfo {
    const char* opName = nullptr;
    QmlaParaInfo opParamInfo; // 输入/输出tensor与属性原始信息

    // 轴信息
    int64_t bSize = 0;
    int64_t n1Size = 0; // Q_N
    int64_t n2Size = 1; // KV_N, MLA固定为1
    int64_t gSize = 1;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t headDimQk = 576;
    int64_t headDimV = 512;
    int64_t qTSize = 0; // Q_T, layout_q为TND时有效
    int64_t t2Size = 0; // KV total

    // SeqLengths（act_seq分裂：cu_seq与seq_used）
    bool cuSeqLenQFlag = false; // cu_seqlens_q 已传入
    bool seqUsedQFlag = false;  // seqused_q 已传入
    int64_t maxSeqQ = -1;
    int64_t maxSeqKv = -1;

    // Paged Attention
    int64_t blockSize = 0;
    int64_t maxBlockNumPerBatch = 0;
    int64_t totalBlockNum = 0;

    // Mask
    int64_t maskMode = 0;
    bool attnMaskFlag = false;
    int64_t attenMaskS1Size = 0;
    int64_t attenMaskS2Size = 0;

    // Quant
    QmlaQuantMode quantMode = QmlaQuantMode::MLA_FP8_E4M3_FULLQUANT;
    float softmaxScale = -1.0f;

    // Layout
    QmlaLayout layoutQ = QmlaLayout::BSND;
    QmlaOutLayout layoutOut = QmlaOutLayout::BSND;
    QmlaKvLayout layoutKv = QmlaKvLayout::PA_BNBD;

    // Strides (for non-contiguous k_cache check)
    const gert::Stride* keyStrides = nullptr;
    bool hasStride = false;

    // SoftmaxLse
    bool returnSoftmaxLse = false;

    // Metadata
    bool metadataFlag = false;

    // 空tensor
    bool emptyTensorFlag = false;

    // dtype
    ge::DataType qType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType kvType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType outType = ge::DT_BF16;
};

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling

#endif // QMLA_TILING_INFO_H
