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
 * \file stem_oam_prep_paged_kv_tiling_data.h
 * \brief StemOamPrepPagedKv TilingData struct definition (arch35)
 */
#ifndef STEM_OAM_PREP_PAGED_KV_TILING_DATA_H
#define STEM_OAM_PREP_PAGED_KV_TILING_DATA_H

#include <cstdint>

struct StemOamPrepPagedKvTilingData {
    int64_t kvLayout = 0;
    int64_t kvBlockSize = 0;
    int64_t numKvHeads = 0;
    int64_t maxKvBlocks = 0;
    int64_t dimQk = 0;
    int64_t dimV = 0;
    int64_t maxKb = 0;
    int64_t kflatDim = 0;
    int64_t batchSize = 0;
    int64_t stemBlockSize = 0;
    int64_t stemStride = 0;
    float lambdaMag = 0;
    int64_t meanSize = 0;
    int64_t rVal = 0;
    int64_t kCacheStride[4] = {0};
    int64_t vCacheStride[4] = {0};
    int64_t kScaleCacheStride[4] = {0};
};

#endif // STEM_OAM_PREP_PAGED_KV_TILING_DATA_H
