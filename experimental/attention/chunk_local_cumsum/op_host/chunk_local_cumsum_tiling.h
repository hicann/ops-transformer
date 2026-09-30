/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file chunk_local_cumsum_tiling.h
 */
#ifndef CHUNK_LOCAL_CUMSUM_TILING_H_
#define CHUNK_LOCAL_CUMSUM_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkLocalCumsumTilingData)

TILING_DATA_FIELD_DEF(int, coreNum);
TILING_DATA_FIELD_DEF(int, tokenNum);
TILING_DATA_FIELD_DEF(int, headNum);
TILING_DATA_FIELD_DEF(int, chunkNum);
TILING_DATA_FIELD_DEF(int, chunkSize);
TILING_DATA_FIELD_DEF_ARR(int, 128, curChunkNumDict); // 128 is max batch

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkLocalCumsum, ChunkLocalCumsumTilingData)
} // namespace optiling

#endif // CHUNK_LOCAL_CUMSUM_TILING_H_
