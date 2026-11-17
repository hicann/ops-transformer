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
 * \file all_gather_fit_balance_tiling.cpp
 * \brief
 */
#include "all_gather_fit_balance_tiling.h"

// 模板主体已移入头文件; 显式实例化A5/A6两种arch组合, 保证符号唯一并随tiling库链接
template class AllGatherMMFitBalanceTilingBase<MatmulPerformanceArch35, HCCLPerformanceArch35>;
template class AllGatherMMFitBalanceTilingBase<MatmulPerformance960dt, HCCLPerformance960dt>;
