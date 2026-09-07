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
 * \file tiling_key_block_attention_residuals.h
 * \brief BlockAttentionResiduals tiling key definitions
 *
 * 说明：dtype 不参与 tiling key，而是通过编译期宏 DTYPE_<输入名>（如
 * DTYPE_PARTIAL_BLOCK）区分——框架按 def 中的 DataType 组合为每个 dtype 生成独立
 * kernel 二进制，并以 -DDTYPE_<NAME>=<type> 注入宏（参考 moe_token_permute）。
 * 因此 key 只区分算法分支 RELOAD / RESIDENT / HSLICE。
 */
#ifndef TILING_KEY_ATTN_RES_FWD_H
#define TILING_KEY_ATTN_RES_FWD_H

#define TILING_KEY_RELOAD 40010UL
#define TILING_KEY_RESIDENT 40020UL
#define TILING_KEY_HSLICE 40030UL

#endif // TILING_KEY_ATTN_RES_FWD_H
