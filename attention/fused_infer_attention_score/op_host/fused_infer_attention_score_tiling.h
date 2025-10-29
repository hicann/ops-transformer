/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
#include "../../incre_flash_attention/op_host/incre_flash_attention_tiling_impl.h"
#include "register/tilingdata_base.h"
#include "fused_infer_attention_score_tiling_compile_info.h"
#include "fused_infer_attention_score_tiling_index.h"

#ifdef ASCENDC_OP_TEST
#define FIA_EXTERN_C extern "C"
#else
#define FIA_EXTERN_C
#endif

namespace optiling {
extern "C" {
ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context);
ge::graphStatus DeviceDoOpTilingFusedInferAttentionScore(gert::TilingContext *context);
}
ge::graphStatus TilingFusedInferAttentionScore(gert::TilingContext *context);
FIA_EXTERN_C ge::graphStatus DoOpTilingFusedInferAttentionScore(gert::TilingContext *context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
