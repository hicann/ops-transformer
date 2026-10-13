/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_KV_COMPRESS_EPILOG_H_
#define OP_API_INC_LEVEL2_ACLNN_KV_COMPRESS_EPILOG_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnKvCompressEpilog的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 将 bf16 激活 x([bs, d], d%64==0 且 d>64) 量化压缩并按 slotMapping 散写到 uint8 cache。
 * x 尾轴后 64 列为 rope 段，前 d-64 列为 nope 段。quantMode 控制量化方式：
 *  - 0: nope per-group(64) FP8(E4M3) 动态量化 + bf16 scale，rope 保留 bf16；roundScale 生效。
 *  - 1: 同 0，但 scale 为 e8m0(uint8) 指数。
 *  - 2: rope 段做 hifloat8 静态量化(乘 xScale 后取 hifloat8)，nope 段做 per-group
 *       FLOAT4_E2M1 动态量化(scale=组内 amax/6 以 bf16 写出)；quantGroupSize 仅支持 16/32/64，
 *       要求 (d-64)%quantGroupSize==0；roundScale 在该模式下被忽略。
 *       行布局(字节)：[rope hifloat8 64B][nope fp4 (d-64)/2 B][nope bf16 scale nGroup*2 B][pad 对齐128]。
 * xScale: mode2 下作为 rope 段 hifloat8 静态量化的缩放系数；mode0/1 下保留未使用，默认 1.0。
 * 仅支持 Ascend 950PR/Ascend 950DT。
 */
aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(aclTensor* cacheRef, const aclTensor* x, const aclTensor* slotMapping,
                                                  int64_t quantGroupSize, int64_t quantMode, bool roundScale,
                                                  float xScale, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnKvCompressEpilog的第二段接口，用于执行计算。
 */
aclnnStatus aclnnKvCompressEpilog(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_KV_COMPRESS_EPILOG_H_
