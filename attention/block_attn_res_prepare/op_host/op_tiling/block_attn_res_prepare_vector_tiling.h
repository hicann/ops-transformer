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
 * \file block_attn_res_prepare_vector_tiling.h
 * \brief Vector tiling template for BlockAttnResPrepare.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_VECTOR_TILING_H
#define BLOCK_ATTN_RES_PREPARE_VECTOR_TILING_H

#include "block_attn_res_prepare_base_tiling.h"
#include "../../op_kernel/arch35/block_attn_res_prepare_tiling_data.h"

namespace optiling {

class BlockAttnResPrepareVectorTiling final : public BlockAttnResPrepareBaseTiling {
public:
    explicit BlockAttnResPrepareVectorTiling(gert::TilingContext *context)
        : BlockAttnResPrepareBaseTiling(context)
    {}
    ~BlockAttnResPrepareVectorTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    TilingDataView GetTilingDataView() const override;

private:
    bool SelectVectorBaseD(bool hasMultipleWorkRounds);

    uint32_t baseD_ = 0;
    uint32_t qBufferNum_ = 1;
    uint32_t vBufferNum_ = 1;
    uint32_t oBufferNum_ = 1;
    uint32_t vCacheRows_ = 0;

    BlockAttnResPrepareTilingData tilingData_{};
};

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_VECTOR_TILING_H
