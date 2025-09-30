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
 * \file service_scatter_cache.h
 * \brief
 */

#ifndef SERVICE_SCATTER_CACHE_H
#define SERVICE_SCATTER_CACHE_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

/**
 * @brief PA场景，将inputLocal中的数据scatter到cacheGm，支持ND和Nz cache
 * @param cacheGm 输出tensor
 *     ND [blockNum, blockSize, col]
 *     Nz [blockNum, ceil(col/col0), blockSize, col0]
 * @param inputLocal 输入tensor，[row, col]，一行对应一个token，只支持单行数据处理，row为1
 * @param scatterCacheParams 所需相关参数，包括
          blockSize KV blocks的大小
          paTokenIndex 待处理token在cache中的全局index，取值[0, blockNum*blockSize)
          row 待处理的行数
          col 待处理的列数，需满足32 bytes对齐
 */

struct ScatterCacheParams{
    int64_t blockSize;
    int64_t paTokenIndex;
    int64_t row;
    int64_t col;
};

template <typename T, bool isNz>
__aicore__ inline void ScatterCache(GlobalTensor<T>& cacheGm, const LocalTensor<T>& inputLocal,
                                    const ScatterCacheParams& scatterCacheParams) {
    if (scatterCacheParams.paTokenIndex < 0) {
        return;
    }
    if constexpr (!isNz) {
        DataCopy(cacheGm[scatterCacheParams.paTokenIndex * scatterCacheParams.col], inputLocal, scatterCacheParams.col);
    } else {
        constexpr uint8_t col0 = ALIGN_BLOCK_SIZE / sizeof(T);
        int64_t cacheOffset = scatterCacheParams.paTokenIndex / scatterCacheParams.blockSize * scatterCacheParams.blockSize * scatterCacheParams.col 
                            + scatterCacheParams.paTokenIndex % scatterCacheParams.blockSize * col0;
        DataCopyParams copyParams {static_cast<uint16_t>(scatterCacheParams.col / col0), 1, 0, static_cast<uint16_t>(scatterCacheParams.blockSize - 1)};
        DataCopy(cacheGm[cacheOffset], inputLocal, copyParams);
    }
}

}

#endif