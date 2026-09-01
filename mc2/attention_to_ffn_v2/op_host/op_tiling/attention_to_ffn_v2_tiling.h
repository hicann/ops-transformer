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
 * \file attention_to_ffn_v2_tiling.h
 * \brief
 */

#ifndef ATTENTION_TO_FFN_V2_TILING_HOST_H
#define ATTENTION_TO_FFN_V2_TILING_HOST_H

#include "register/tilingdata_base.h"
#include "../../../attention_to_ffn/op_host/op_tiling/attention_to_ffn_tiling_base.h"

namespace MC2Tiling {

ge::graphStatus AttentionToFfnV2TilingFunc(gert::TilingContext *context);
} // namespace MC2Tiling

#endif // ATTENTION_TO_FFN_V2_TILING_HOST_H
