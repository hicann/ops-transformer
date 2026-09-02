/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_sort_base.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SORT_BASE_H
#define OP_KERNEL_FFN_WB_SORT_BASE_H

#include "kernel_operator.h"

namespace FfnWbBatching {
using namespace AscendC;

class SortMaskBase {
public:
    __aicore__ inline SortMaskBase(){};

protected:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
    TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;

    GlobalTensor<int32_t> expertIdsGm;
    GlobalTensor<int32_t> rsvdCntGm;
    GlobalTensor<int32_t> sortedexpertIdsGm;
    GlobalTensor<int32_t> sortedRowIdsGm;
    GlobalTensor<int32_t> groupListTmpGm;

    int64_t bufferNum = 1;
    int64_t totalLength = 0;

    int64_t expertStart_ = 1000000;
    int64_t n = 0;
    int64_t k = 0;
    int64_t rowIdxType_ = 0;

    static constexpr int64_t SYNC_GM_NUM = 2;
    static constexpr int64_t WORK_GM_NUM = 2;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int32_t INT_MAX = 2147483647;
    static constexpr uint32_t BLOCK_SIZE = 32;
};

} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_SORT_BASE_H
