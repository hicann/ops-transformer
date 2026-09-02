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
 * \file swin_attention_ffn_tiling.h
 * \brief
 */
#ifndef SWIN_ATTENTION_FFN_TILING_H
#define SWIN_ATTENTION_FFN_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SwinAttentionFFNTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, bmmFormerNum);
TILING_DATA_FIELD_DEF(uint32_t, bmmTailNum);
TILING_DATA_FIELD_DEF(uint32_t, bmmFormerBatchNum);
TILING_DATA_FIELD_DEF(uint32_t, bmmTailBatchNum);
TILING_DATA_FIELD_DEF(uint32_t, aivNum);
TILING_DATA_FIELD_DEF(uint32_t, shift1);
TILING_DATA_FIELD_DEF(uint32_t, shift2);

TILING_DATA_FIELD_DEF(uint32_t, tpBlockSize);
TILING_DATA_FIELD_DEF(uint32_t, tpSpaceCnt);
TILING_DATA_FIELD_DEF(uint32_t, tpSpaceH);
TILING_DATA_FIELD_DEF(uint32_t, tpSpaceW);
TILING_DATA_FIELD_DEF(uint32_t, blockInSpace);
TILING_DATA_FIELD_DEF(uint32_t, tpSpaceSize);
TILING_DATA_FIELD_DEF(uint32_t, tpSpaceWTransposed);
TILING_DATA_FIELD_DEF(uint32_t, tpSpaceHTransposed);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmmTilingData);
TILING_DATA_FIELD_DEF(uint32_t, dataNumPerBatchA);
TILING_DATA_FIELD_DEF(uint32_t, dataNumPerBatchD);
TILING_DATA_FIELD_DEF(uint32_t, dataNumPerLoop);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwinAttentionFFN, SwinAttentionFFNTilingData)

struct SwinAttentionFFNCompileInfo {
    uint64_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
};

} // namespace optiling
#endif // SWIN_ATTENTION_FFN_TILING_H
