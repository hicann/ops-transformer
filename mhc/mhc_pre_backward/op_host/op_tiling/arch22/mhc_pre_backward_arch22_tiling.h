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
 * \file mhc_pre_backward_arch22_tiling.h
 * \brief MhcPreBackward arch22 tiling declaration
 */

#ifndef MHC_PRE_BACKWARD_ARCH22_TILING_H_
#define MHC_PRE_BACKWARD_ARCH22_TILING_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"

namespace optiling {
ge::graphStatus TilingMhcPreBackwardArch22(gert::TilingContext *context);
}

#endif // MHC_PRE_BACKWARD_ARCH22_TILING_H_
