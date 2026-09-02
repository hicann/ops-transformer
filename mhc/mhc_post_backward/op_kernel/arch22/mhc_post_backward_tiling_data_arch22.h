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
 * \file mhc_post_backward_tiling_data.h
 * \brief mhc_post_backward
 */

#ifndef MHC_POST_BACKWARD_TILING_DATA_ARCH22_H
#define MHC_POST_BACKWARD_TILING_DATA_ARCH22_H

class MhcPostBackwardTilingDataArch22 {
public:
    uint64_t coreUsed{0};
    uint64_t frontCore{0};
    uint64_t tailCore{0};
    uint64_t singleCoreBS{0};
    uint64_t tailBS{0};
    uint64_t dFPostResSize{0};
    uint64_t xSize{0};
    uint64_t hResSize{0};
    uint64_t hOutSize{0};
    uint64_t hPostSize{0};
    uint64_t channel{0};
    uint64_t blockChannel{0};
    uint64_t n{0};
    uint64_t alignN{0};
    uint64_t tileC{0};
    uint64_t tailC{0};
    uint64_t loopC{0};
};

#endif // MHC_POST_BACKWARD_TILING_DATA_ARCH22_H
