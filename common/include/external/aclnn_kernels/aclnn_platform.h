/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_PLATFORM_H_
#define ACLNN_PLATFORM_H_

#include "opdev/platform.h"
#include "op_host/util/op_const_def.h"
#include <set>

namespace Ops {
namespace Transformer {
namespace AclnnUtil {

using namespace op;

/**
 * 检查当前芯片架构是否为RegBase
 */
inline static bool IsRegbase(NpuArch npuArch)
{
    const static std::set<NpuArch> regbaseNpuArchs = {Ops::Base::DAV_3510, Ops::Base::DAV_5102};
    return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

inline static bool IsRegbase()
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return IsRegbase(npuArch);
}

} // namespace AclnnUtil
} // namespace Transformer
} // namespace Ops

#endif
