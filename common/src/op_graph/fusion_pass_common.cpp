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
 * \file fusion_pass_common.cpp
 * \brief FE 兼容融合注册适配层（预埋）。当前生产注册名单为空：不满足条件时
 *        本编译单元经预处理不产生任何符号引用；构建入口由 OP_GRAPH_FE_REGISTRY
 *        选项门控（默认 ON）。
 */

// 仅当编译 soc 为 ascend950 时，当前 cpp 中的桩注册内容才编入
#if defined(OP_GRAPH_TRANSFORMER_ASCEND950)

// 包含顺序契约：桩头（内含 log/log.h → log_types.h 的 GE 枚举）必须先于
// version/ge-compiler_version.h —— 后者的守卫宏 "#define GE-COMPILER_VERSION_H"
// 实为定义名为 GE 的宏，一旦先包含会污染 log_types.h 的 GE 枚举导致编译错误。
#include "fe_fusion_pass_registry_stub.h"
#include "version/ge-compiler_version.h"

#define GE_COMPILER_VERSION_900 90000000
// 编译期保护：构建环境 ge_compiler >= 9.0.0 才编入（运行时检查在 STUB_REG_PASS 宏内）
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900

// 生产注册名单（当前为空）。登记时名字/类型/attr 照抄 canndev 原
// REG_PASS/REGISTER_PASS，且须与同名真实实现在相同构建条件下编译。

#endif // GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
#endif // OP_GRAPH_TRANSFORMER_ASCEND950
