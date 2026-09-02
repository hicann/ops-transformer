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
 * \file ring_attention_update_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RING_ATTENTION_UPDATE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RING_ATTENTION_UPDATE_H_

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RingAttentionUpdateTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, headNum);
TILING_DATA_FIELD_DEF(int64_t, seqNum);
TILING_DATA_FIELD_DEF(int64_t, headDim);
TILING_DATA_FIELD_DEF(int64_t, softmaxTailSize);

TILING_DATA_FIELD_DEF(int64_t, coreNum);

TILING_DATA_FIELD_DEF(int64_t, coreNumGroup);
TILING_DATA_FIELD_DEF(int64_t, bnNumGroup);
TILING_DATA_FIELD_DEF(int64_t, seqNumCoreEach);
TILING_DATA_FIELD_DEF(int64_t, seqNumCoreTail);
TILING_DATA_FIELD_DEF(int64_t, seqNumLoopEach);
TILING_DATA_FIELD_DEF(int64_t, headNumLoopEach);
TILING_DATA_FIELD_DEF(int64_t, headDimLoopEach);

TILING_DATA_FIELD_DEF(int64_t, batchSizeCoreEach);
TILING_DATA_FIELD_DEF(int64_t, batchSizeCoreTail);
TILING_DATA_FIELD_DEF(int64_t, dimTCoreEach);
TILING_DATA_FIELD_DEF(int64_t, dimTCoreTail);
TILING_DATA_FIELD_DEF(uint8_t, tndSoftmaxLayout);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(RingAttentionUpdateRegbaseTNDTilingData)
// shape
TILING_DATA_FIELD_DEF(int64_t, dimT);
TILING_DATA_FIELD_DEF(int64_t, dimN);
TILING_DATA_FIELD_DEF(int64_t, dimD);
TILING_DATA_FIELD_DEF(int64_t, softmaxTailSize);
TILING_DATA_FIELD_DEF(int64_t, batchSize);

// 核间分核
TILING_DATA_FIELD_DEF(int64_t, usedVectorCoreNum); // 使用的vector核数
TILING_DATA_FIELD_DEF(int64_t, dimTCoreEach);      // 每核处理的T方向长度
TILING_DATA_FIELD_DEF(int64_t, dimTCoreTail);      // 尾核处理的T方向长度

// 核内循环
TILING_DATA_FIELD_DEF(int64_t, seqNumLoopEach);  // 单次UB循环Seq方向搬入长度
TILING_DATA_FIELD_DEF(int64_t, headNumLoopEach); // 单次UB循环headNum长度
TILING_DATA_FIELD_DEF(int64_t, headDimLoopEach); // 单次UB循环headDim长度
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(RingAttentionUpdateRegbaseSoftmaxTNDTilingData)
// shape
TILING_DATA_FIELD_DEF(int64_t, dimT);
TILING_DATA_FIELD_DEF(int64_t, dimN);
TILING_DATA_FIELD_DEF(int64_t, dimD);
TILING_DATA_FIELD_DEF(int64_t, softmaxTailSize);
TILING_DATA_FIELD_DEF(int64_t, batchSize);

// 核间分核
TILING_DATA_FIELD_DEF(int64_t, usedVectorCoreNum); // 使用的vector核数
TILING_DATA_FIELD_DEF(int64_t, tnNumCoreEach);     // 每核处理的TN方向长度
TILING_DATA_FIELD_DEF(int64_t, tnNumCoreTail);     // 尾核处理的TN方向长度

// 核内循环
TILING_DATA_FIELD_DEF(int64_t, tnNumLoopEach);   // 单次UB循环TN方向搬入长度
TILING_DATA_FIELD_DEF(int64_t, headDimLoopEach); // 单次UB循环headDim长度
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(RingAttentionUpdateRegbaseSBHTilingData)
// shape
TILING_DATA_FIELD_DEF(int64_t, dimB);
TILING_DATA_FIELD_DEF(int64_t, dimS);
TILING_DATA_FIELD_DEF(int64_t, dimN);
TILING_DATA_FIELD_DEF(int64_t, dimD);
TILING_DATA_FIELD_DEF(int64_t, softmaxTailSize);

// 核间分核
TILING_DATA_FIELD_DEF(int64_t, usedVectorCoreNum); // 使用的vector核数
TILING_DATA_FIELD_DEF(int64_t, coreNumGroup);      // 每组coreNumGroup个核
TILING_DATA_FIELD_DEF(int64_t, bnNumGroup);        // 每组计算bnNumGroup个bn
TILING_DATA_FIELD_DEF(int64_t, seqNumCoreEach);    // 每核处理的Seq长度
TILING_DATA_FIELD_DEF(int64_t, seqNumCoreTail);    // 尾核处理的Seq长度

// 核内循环
TILING_DATA_FIELD_DEF(int64_t, seqNumLoopEach);  // 单次UB循环Seq长度
TILING_DATA_FIELD_DEF(int64_t, headDimLoopEach); // 单次UB循环headDim长度
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RingAttentionUpdate, RingAttentionUpdateTilingData)

REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_100, RingAttentionUpdateRegbaseSBHTilingData)
REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_101, RingAttentionUpdateRegbaseSBHTilingData)
REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_102, RingAttentionUpdateRegbaseSBHTilingData)

REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_110, RingAttentionUpdateRegbaseSoftmaxTNDTilingData)
REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_111, RingAttentionUpdateRegbaseSoftmaxTNDTilingData)
REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_112, RingAttentionUpdateRegbaseSoftmaxTNDTilingData)

REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_120, RingAttentionUpdateRegbaseTNDTilingData)
REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_121, RingAttentionUpdateRegbaseTNDTilingData)
REGISTER_TILING_DATA_CLASS(RingAttentionUpdate_122, RingAttentionUpdateRegbaseTNDTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_RING_ATTENTION_UPDATE_H_
