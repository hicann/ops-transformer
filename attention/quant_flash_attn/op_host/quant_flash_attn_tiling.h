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
 * \file quant_flash_attn_tiling.h
 * \brief QuantFlashAttn Tiling
 */

#ifndef QUANT_FLASH_ATTN_TILING_H_
#define QUANT_FLASH_ATTN_TILING_H_

#include <cstdint>
#include <register/op_impl_registry.h>
#include "../op_kernel/arch35/quant_flash_attn_tiling_data.h"
#include "quant_flash_attn_tiling_common.h"

namespace optiling {

ASCENDC_EXTERN_C ge::graphStatus TilingQuantFlashAttn(gert::TilingContext *context);
ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForQuantFlashAttn(gert::TilingParseContext *context);

} // namespace optiling

#endif // QUANT_FLASH_ATTN_TILING_H_
