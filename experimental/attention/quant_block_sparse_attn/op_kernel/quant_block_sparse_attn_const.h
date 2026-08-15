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
 * \file quant_block_sparse_attn_const.h
 * \brief QuantBlockSparseAttn shared constants and layout definitions for host and kernel.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_CONST_H
#define QUANT_BLOCK_SPARSE_ATTN_CONST_H

#include <cstdint>
#include <limits>

// === Layout enum (shared by host and kernel) ===
enum class QBSALayout : uint32_t {
    TND = 2,
    PA_BNBD = 3,
    PA_BSND = 4,
    NTD = 5,
};

// === Q layout values (derived from QBSALayout enum, used in tiling data layoutQ field) ===
constexpr uint32_t QBSA_LAYOUT_Q_TND_VALUE = static_cast<uint32_t>(QBSALayout::TND);
constexpr uint32_t QBSA_LAYOUT_Q_NTD_VALUE = static_cast<uint32_t>(QBSALayout::NTD);

// === Semantic axis enum (layout-independent axis meaning) ===
enum class QBSAAxis : size_t {
    T,             // token / seq (query)
    N,             // head
    D,             // head dim
    B,             // batch
    BLOCK_NUM,     // PA block count (key/value dim0)
    BLOCK_SIZE,    // PA block size (key/value dim2)
    HEAD_DIM,      // PA head dim (key/value dim3)
    QB,            // sparse_indices dim2: max query block count
    KB,            // sparse_indices dim3: sparse count
    MAX_BLOCK_NUM, // block_table dim1: max blocks per batch
};

// === Layout-aware axis index mapping for Q/KV (QBSALayout::TND/NTD/PA_BNBD) ===
constexpr size_t QBSAGetAxisIdx(QBSALayout layout, QBSAAxis axis)
{
    switch (layout) {
        case QBSALayout::TND:
            switch (axis) {
                case QBSAAxis::T:
                    return 0U;
                case QBSAAxis::N:
                    return 1U;
                case QBSAAxis::D:
                    return 2U;
                default:
                    break;
            }
            break;
        case QBSALayout::NTD:
            switch (axis) {
                case QBSAAxis::N:
                    return 0U;
                case QBSAAxis::T:
                    return 1U;
                case QBSAAxis::D:
                    return 2U;
                default:
                    break;
            }
            break;
        case QBSALayout::PA_BNBD:
            switch (axis) {
                case QBSAAxis::BLOCK_NUM:
                    return 0U;
                case QBSAAxis::N:
                    return 1U;
                case QBSAAxis::BLOCK_SIZE:
                    return 2U;
                case QBSAAxis::HEAD_DIM:
                    return 3U;
                default:
                    break;
            }
            break;
        default:
            break;
    }
    return std::numeric_limits<size_t>::max();
}

// === sparse_indices layout: [B, N, Qb, Kb] ===
constexpr size_t QBSAGetSparseIndicesAxisIdx(QBSAAxis axis)
{
    switch (axis) {
        case QBSAAxis::B:
            return 0U;
        case QBSAAxis::N:
            return 1U;
        case QBSAAxis::QB:
            return 2U;
        case QBSAAxis::KB:
            return 3U;
        default:
            break;
    }
    return std::numeric_limits<size_t>::max();
}

// === block_table layout: [B, maxBlockNumPerBatch] ===
constexpr size_t QBSAGetBlockTableAxisIdx(QBSAAxis axis)
{
    switch (axis) {
        case QBSAAxis::B:
            return 0U;
        case QBSAAxis::MAX_BLOCK_NUM:
            return 1U;
        default:
            break;
    }
    return std::numeric_limits<size_t>::max();
}

// === Regbase layout encoding (consumed by regbase infrastructure, NOT by kernel QBSALayout enum) ===
constexpr uint32_t QBSA_REGBASE_KV_PA_BNSD = 1U;
constexpr uint32_t QBSA_REGBASE_SPARSE_B_N_QB_KB = 0U;
constexpr uint8_t QBSA_REGBASE_LAYOUT_TYPE_TND = static_cast<uint8_t>(QBSALayout::TND);
constexpr uint8_t QBSA_REGBASE_LAYOUT_TYPE_NTD = static_cast<uint8_t>(QBSALayout::NTD);

#endif // QUANT_BLOCK_SPARSE_ATTN_CONST_H
