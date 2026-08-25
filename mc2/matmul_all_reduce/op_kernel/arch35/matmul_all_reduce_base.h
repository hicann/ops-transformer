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
 * \file matmul_all_reduce_base.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_BASE_H
#define MATMUL_ALL_REDUCE_BASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../matmul_all_reduce_common.h"
#include "matmul_all_reduce_add_x3_arch35.h"
#include "matmul_all_reduce_tiling_struct_ar35.h"
#include "../../../common/op_kernel/reduce_sum_cast_fp32.h"
#include "../../../common/op_kernel/gm_ub_gm_copy.h"
namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename XType, typename YType, Mc2CoreType CoreType, bool basedA2aRsAg, int commMode>
class MatmulAllReduceBase {};
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_BASE_H
