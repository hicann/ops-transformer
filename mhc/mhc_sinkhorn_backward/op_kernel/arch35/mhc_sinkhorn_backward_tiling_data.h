/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file mhc_sinkhorn_backward_tiling_data.h
 * \brief mhc_sinkhorn_backward
 */

#ifndef MHC_SINKHORN_BACKWARD_TILING_DATA_H
#define MHC_SINKHORN_BACKWARD_TILING_DATA_H

class MhcSinkhornBackwardTilingData {
public:
    int64_t totalLength{0};
    int64_t tSize{0};
    int64_t bSize{0};
    int64_t sSize{0};
    int64_t nSize{0};
    int64_t nAlignSize{0};
    int64_t numIters{0};
    int64_t needCoreNum{0};
    int64_t coreTSize{0};
    int64_t normCoreTLoops{0};
    int64_t normCorePerLoopTSize{0};
    int64_t normCoreLastLoopTSize{0};
    int64_t tailCoreTSize{0};
    int64_t tailCoreLoops{0};
    int64_t tailCorePerLoopTSize{0};
    int64_t tailCoreLastLoopTSize{0};
};

#endif // MHC_SINKHORN_BACKWARD_TILING_DATA_H
