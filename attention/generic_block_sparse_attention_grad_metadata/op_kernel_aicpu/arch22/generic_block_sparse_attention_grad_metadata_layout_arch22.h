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
 * \file generic_block_sparse_attention_grad_metadata_layout_arch22.h
 * \brief
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_LAYOUT_ARCH22_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_LAYOUT_ARCH22_H

#include <cstddef>
#include <cstdint>

namespace gbsag_meta {

constexpr uint32_t HEADER_WORDS = 6;
constexpr uint32_t KTASK_WORDS = 3;
constexpr uint32_t BASIC_BLOCK_SIZE = 128;
constexpr uint32_t MAX_AIC_CORE_NUM = 64;

constexpr uint32_t OFF_TOTAL_TASK_NUM = 0;
constexpr uint32_t OFF_TASK_NUM_PER_CORE = 1;
constexpr uint32_t OFF_TAIL_TASK_NUM = 2;
constexpr uint32_t OFF_TASK_TABLE_OFFSET = 3;
constexpr uint32_t OFF_CORE_NUM = 4;
constexpr uint32_t OFF_TASK_WORDS = 5;

constexpr uint32_t TASK_BEGIN_BATCH = 0;
constexpr uint32_t TASK_BEGIN_KV_HEAD = 1;
constexpr uint32_t TASK_BEGIN_KV_SEQ_OFFSET = 2;

constexpr uint32_t LAYOUT_TND = 0;
constexpr uint32_t LAYOUT_BNSD = 1;
constexpr uint32_t LAYOUT_BSND = 2;

struct Header {
    int32_t totalTaskNum;
    int32_t taskNumPerCore;
    int32_t tailTaskNum;
    int32_t taskTableOffset;
    int32_t coreNum;
    int32_t taskWords;
};

struct KTask {
    int32_t beginBatch;
    int32_t beginKvHead;
    int32_t beginKvSeqOffset;
};

static_assert(sizeof(Header) == HEADER_WORDS * sizeof(int32_t), "metadata header ABI drift");
static_assert(sizeof(KTask) == KTASK_WORDS * sizeof(int32_t), "metadata KTask ABI drift");
static_assert(offsetof(Header, totalTaskNum) / sizeof(int32_t) == OFF_TOTAL_TASK_NUM, "header offset drift");
static_assert(offsetof(Header, taskTableOffset) / sizeof(int32_t) == OFF_TASK_TABLE_OFFSET, "header offset drift");
static_assert(offsetof(KTask, beginKvSeqOffset) / sizeof(int32_t) == TASK_BEGIN_KV_SEQ_OFFSET, "task offset drift");

inline uint32_t MetadataWords(uint32_t coreNum)
{
    return HEADER_WORDS + coreNum * KTASK_WORDS;
}

} // namespace gbsag_meta

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_LAYOUT_ARCH22_H
