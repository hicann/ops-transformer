/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SCATTER_PA_KV_CACHE_WITH_K_SCALE_H_
#define ACLNN_SCATTER_PA_KV_CACHE_WITH_K_SCALE_H_
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnScatterPaKvCacheWithKScale的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param key                  [IN]  待更新的key值，当前step多个token的key。数据类型FLOAT8_E5M2或FLOAT8_E4M3FN，
 *                                  ND格式，shape=(tokenNum, numHead, keyHeadSize)，tokenNum=batch*seqLen。
 * @param value                [IN]  待更新的value值，当前step多个token的value。数据类型与key一致，
 *                                  ND格式，shape=(tokenNum, numHead, valueHeadSize)。
 * @param keyCacheRef          [IN/OUT] 需要更新的key cache，当前layer的key cache。数据类型与key一致，
 *                                  ND格式，shape=(numBlocks, numHead, blockSize, keyHeadSize)。
 * @param valueCacheRef        [IN/OUT] 需要更新的value cache，当前layer的value cache。数据类型与key一致，
 *                                  ND格式，shape=(numBlocks, numHead, blockSize, valueHeadSize)。
 * @param slotMapping          [IN]  每个token key或value在cache中的存储偏移。数据类型INT32或INT64，
 *                                  ND格式，shape=(tokenNum,)。取值范围[0, numBlocks*blockSize-1]，元素值不重复。
 * @param keyScale             [IN]  待更新的key scale值，当前step多个token的key scale。数据类型FLOAT，
 *                                  ND格式，shape=(tokenNum, numHead)，尾轴可非连续。
 * @param keyScaleCacheRef     [IN/OUT] 需要更新的key scale cache，当前layer的key scale cache。数据类型FLOAT，
 *                                  ND格式，shape=(numBlocks, numHead, blockSize, 1)，最后一维必须为1，尾轴必须连续。
 * @param cacheLayoutOptional  [IN]  可选。表示keyCacheRef和valueCacheRef的内存排布格式。
 *                                  传空指针或"BNBD"时，表示格式为[numBlocks, numHead, blockSize, headSize]。
 * @param workspaceSize        [OUT] workspace大小（字节数）。
 * @param executor             [OUT] op执行器句柄，供第二段接口使用。
 * @return aclnnStatus 执行状态。ACLNN_SUCCESS表示成功。
 */
aclnnStatus aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize(const aclTensor* key, const aclTensor* value,
                                                            aclTensor* keyCacheRef, aclTensor* valueCacheRef,
                                                            const aclTensor* slotMapping, aclTensor* keyScale,
                                                            aclTensor* keyScaleCacheRef, char* cacheLayoutOptional,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnScatterPaKvCacheWithKScale的第二段接口，用于执行计算。
 * @param workspace       [IN] 由第一段接口计算得到的workspace设备内存指针。
 * @param workspaceSize   [IN] workspace大小（字节数）。
 * @param executor        [IN] 第一段接口输出的op执行器句柄。
 * @param stream          [IN] 用于执行计算的acl stream。
 * @return aclnnStatus 执行状态。
 */
aclnnStatus aclnnScatterPaKvCacheWithKScale(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
