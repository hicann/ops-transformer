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
 * \file nsa_compre_with_cache_tiling.h
 * \brief
 */

#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NsaCompressWithCacheTilingData)
TILING_DATA_FIELD_DEF(uint64_t, kvCacheSize);         // 全量数据的kvCache大小，单位为元素个数
TILING_DATA_FIELD_DEF(uint64_t, weightSize);          // 全量数据的weight大小，单位为元素个数
TILING_DATA_FIELD_DEF(uint64_t, compressKvCacheSize); // 全量数据的compressKvCache大小，单位为元素个数
TILING_DATA_FIELD_DEF(uint32_t, kvCacheSizePerCore);  // 每个core一次处理的kvCache大小，单位为元素个数
TILING_DATA_FIELD_DEF(uint32_t, weightSizePerCore);   // 每个core一次处理的weight大小，单位为元素个数
TILING_DATA_FIELD_DEF(uint32_t, compressKvCacheSizePerCore); // 每个core一次处理的compressKvCache大小，单位为元素个数
TILING_DATA_FIELD_DEF(uint32_t, blockTableSize);      // 全量数据的blockTable大小，单位为元素个数
TILING_DATA_FIELD_DEF(uint32_t, headNum);             // 全量数据的headNum大小
TILING_DATA_FIELD_DEF(uint32_t, headDim);             // 全量数据的headDim大小
TILING_DATA_FIELD_DEF(uint32_t, batchSize);           // 全量数据的batchSize大小
TILING_DATA_FIELD_DEF(uint32_t, pageBlockSize);       // 一个page的token数量
TILING_DATA_FIELD_DEF(uint32_t, pageNumPerBatch);     // 每个Batch使用几个page表示
TILING_DATA_FIELD_DEF(uint32_t, compressBlockSize);   // 全量的compressBlockSize大小
TILING_DATA_FIELD_DEF(uint32_t, compressStride);      // 全量的compressStride大小
TILING_DATA_FIELD_DEF(uint32_t, coresNumPerCompress); // 一个token的压缩需要分多少个core完成
TILING_DATA_FIELD_DEF(uint32_t, headsNumPerCore);     // 每个core一次处理多少个head
TILING_DATA_FIELD_DEF(uint32_t, weightInitFlag);      // weight的init方法
TILING_DATA_FIELD_DEF(uint32_t, isEmpty);             // 输入是否为空
TILING_DATA_FIELD_DEF(uint32_t, alignReduceSize);     // compressBlockSize对齐到2的幂次值
TILING_DATA_FIELD_DEF(uint32_t, tokenNumPerTile);     // 核内循环一次压缩多少个token
TILING_DATA_FIELD_DEF(uint32_t, tokenRepeatCount);    // 需要循环多少次才能压缩完compressBlockSize个token
TILING_DATA_FIELD_DEF(uint32_t, bufferNum);           // buffer数量
TILING_DATA_FIELD_DEF(uint32_t, alignHolder);         // 输入是否为空
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreStartBatchIdxList); // 每个核从哪个batch开始遍历seq_len
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreStartHeadIdxList);  // 每个核心处理的起始head_idx
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NsaCompressWithCache, NsaCompressWithCacheTilingData)
} // namespace optiling
