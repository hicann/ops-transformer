/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elastic_receivable_test_tiling.h
 * \brief
 */
#ifndef ELASTIC_RECEIVABLE_TEST_TILING_H
#define ELASTIC_RECEIVABLE_TEST_TILING_H
#include "kernel_tiling/kernel_tiling.h"

struct ElasticReceivableTestInfo {
    uint32_t worldSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint32_t aivNum;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct ElasticReceivableTestTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    ElasticReceivableTestInfo elasticReceivableTestInfo;
};

#endif // ELASTIC_RECEIVABLE_TEST_TILING_H