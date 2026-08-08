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
 * \file quant_block_sparse_attn_check.h
 * \brief QuantBlockSparseAttn parameter validation class.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_CHECK_H_
#define QUANT_BLOCK_SPARSE_ATTN_CHECK_H_

#include <exe_graph/runtime/tiling_context.h>
#include "quant_block_sparse_attn_info_parser.h"

namespace optiling {

class QuantBlockSparseAttnCheck {
public:
    explicit QuantBlockSparseAttnCheck(const QuantBlockSparseAttnTilingInfo &tilingInfo);
    ~QuantBlockSparseAttnCheck() = default;

    ge::graphStatus Process();

private:
    ge::graphStatus CheckAttrs() const;
    ge::graphStatus CheckSoftmaxScale() const;
    ge::graphStatus CheckDtype() const;
    ge::graphStatus CheckFormat() const;
    ge::graphStatus CheckScaleDtype() const;
    ge::graphStatus CheckBlockSize() const;
    ge::graphStatus CheckExistence() const;
    ge::graphStatus CheckShapeConsistency() const;
    ge::graphStatus CheckKeyValueShape() const;
    ge::graphStatus CheckQuantShape() const;
    ge::graphStatus CheckActualSeqLen() const;
    ge::graphStatus CheckAttenMask() const;
    ge::graphStatus CheckMXFP8FullQuantShape() const;

    const QuantBlockSparseAttnTilingInfo &tilingInfo_;
};

} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_CHECK_H_
