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
 * \file ifa_public_define.h
 * \brief
 */
#ifndef FIA_PUBLIC_DEFINE_ARCH92_H
#define FIA_PUBLIC_DEFINE_ARCH92_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../mqfa_fia_public_define.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;

/* ===== 静态 cross-core 事件 ID ===== */
#define CC_BMM1_0 0U // UB bmm1 slot0
#define CC_BMM1_1 1U // UB bmm1 slot1
#define CC_BMM2_0 2U // UB bmm2 slot0
#define CC_BMM2_1 3U // UB bmm2 slot1
#define CC_L1P_0 4U  // L1 P slot0
#define CC_L1P_1 5U  // L1 P slot1
#define CC_L1P_2 6U  // L1 P slot2
#define CC_L1P_3 7U  // L1 P slot3
#define CC_L1Q_0 8U  // L1 Q slot1
#define CC_L1Q_1 9U  // L1 Q slot2
// id [11-14] SyncAll要用

#endif // FIA_PUBLIC_DEFINE_ARCH92_H
