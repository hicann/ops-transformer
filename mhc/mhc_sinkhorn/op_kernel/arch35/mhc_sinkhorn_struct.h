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
 * \file mhc_sinkhorn_struct.h
 * \brief mhc_sinkhorn
 */

#ifndef MHC_SINKHORN_STRUCT_H
#define MHC_SINKHORN_STRUCT_H

class MhcSinkhornTilingData {
public:
    float eps{0};
    int64_t num_iters{0};
    int64_t out_flag{0};
    int64_t n{0};
    int64_t usedCoreNum{0};
    int64_t tNormCoreLoop{0};
    int64_t tUbFactor{0};     // 正常循环大小
    int64_t tUbFactorTail{0}; // 正常核的尾循环大小
    int64_t tTailCoreLoop{0};
    int64_t tUbTailTail{0}; // 尾核的尾循环大小
    int64_t tNormCore{0};   // 正常核计算的总大小 (n * n 的个数)
};

#endif // MHC_SINKHORN_STRUCT_H
