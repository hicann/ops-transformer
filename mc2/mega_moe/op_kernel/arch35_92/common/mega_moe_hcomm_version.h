/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_HCOMM_VERSION_H
#define MEGA_MOE_HCOMM_VERSION_H

#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif

#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#endif

// Missing headers retain legacy defaults; visible component versions still apply.
#if !__has_include("version/asc_devkit_version.h") || ASC_DEVKIT_MAJOR> 9 || \
                   (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)
#define ENABLE_TENSOR_API
#endif

#if (!__has_include("version/asc_devkit_version.h") || (ASC_DEVKIT_VERSION_NUM >= 90200000)) && \
     (!__has_include("version/hcomm_version.h") || (HCOMM_VERSION_NUM >= 90200000))
#define ENABLE_MEGA_MOE_LAYERED_KERNEL
#endif

// RoCE batch/channel APIs used by UBX require both components from 9.3.0.
#if (!__has_include("version/asc_devkit_version.h") || (ASC_DEVKIT_VERSION_NUM >= 90300000)) && \
     (!__has_include("version/hcomm_version.h") || (HCOMM_VERSION_NUM >= 90300000))
#define ENABLE_MEGA_MOE_LAYERED_ROCE_KERNEL
#endif

#endif // MEGA_MOE_HCOMM_VERSION_H
