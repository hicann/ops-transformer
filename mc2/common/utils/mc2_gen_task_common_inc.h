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
 * \file mc2_gen_task_common_inc.h
 * \brief mc2 各算子 op_graph 下 *_gen_task.cpp 的公共头文件聚合, 用于消除重复 include
 */
#ifndef MC2_GEN_TASK_COMMON_INC_H
#define MC2_GEN_TASK_COMMON_INC_H

#include <vector>
#include <set>
#include <string>

#include "common/utils/op_mc2.h"
#include "platform/platform_info.h"

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "graph/arg_desc_info.h"
#include "graph/kernel_launch_info.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"

#endif // MC2_GEN_TASK_COMMON_INC_H
