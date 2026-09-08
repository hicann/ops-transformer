/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_CHECK_H_
#define FLASH_ATTN_GRAD_CHECK_H_

#include "exe_graph/runtime/tiling_context.h"
#include "log/log.h"

namespace optiling {

constexpr size_t Q_INDEX = 0;
constexpr size_t K_INDEX = 1;
constexpr size_t V_INDEX = 2;
constexpr size_t DOUT_INDEX = 3;
constexpr size_t ATTN_OUT_INDEX = 4;
constexpr size_t SOFTMAX_LSE_INDEX = 5;
constexpr size_t CU_SEQLENS_Q_INDEX = 6;
constexpr size_t CU_SEQLENS_KV_INDEX = 7;
constexpr size_t SEQUSED_Q_INDEX = 8;
constexpr size_t SEQUSED_KV_INDEX = 9;
constexpr size_t SINKS_INDEX = 10;
constexpr size_t ATTN_MASK_INDEX = 11;
constexpr size_t METADATA_INDEX = 12;

constexpr int64_t MASK_MODE_NO_MASK = 0;
constexpr int64_t MASK_MODE_CAUSAL = 3;
constexpr int64_t MASK_MODE_WINDOW = 4;
constexpr int64_t ATTN_MASK_DIM = 2048;

class FlashAttnGradCheck {
public:
    static ge::graphStatus CheckParams(gert::TilingContext *context);

private:
    static bool IsTensorExist(gert::TilingContext *context, size_t index);
    static ge::graphStatus CheckInputExistence(gert::TilingContext *context);
    static ge::graphStatus CheckDtypeConsistency(gert::TilingContext *context);
    static ge::graphStatus CheckAttrs(gert::TilingContext *context, int64_t &maskMode, int64_t &winLeft,
                                      int64_t &winRight, std::string &layoutQStr, std::string &layoutKvStr,
                                      std::string &layoutOutStr);
    static ge::graphStatus CheckMaskAndAttnMask(gert::TilingContext *context, int64_t maskMode, int64_t winLeft,
                                                int64_t winRight);
    static ge::graphStatus CheckLayout(gert::TilingContext *context, const std::string &layoutQStr,
                                       const std::string &layoutKvStr, const std::string &layoutOutStr);
    static ge::graphStatus CheckShape(gert::TilingContext *context, const std::string &layoutQStr);
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECK_H_
