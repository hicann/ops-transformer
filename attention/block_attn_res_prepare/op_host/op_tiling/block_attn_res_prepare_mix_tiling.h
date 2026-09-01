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
 * \file block_attn_res_prepare_mix_tiling.h
 * \brief Mixed Cube/Vector tiling template for BlockAttnResPrepare.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_MIX_TILING_H
#define BLOCK_ATTN_RES_PREPARE_MIX_TILING_H

#include "block_attn_res_prepare_base_tiling.h"
#include "../../op_kernel/arch35/block_attn_res_prepare_tiling_data.h"

namespace optiling {

class BlockAttnResPrepareMixTiling final : public BlockAttnResPrepareBaseTiling {
public:
    explicit BlockAttnResPrepareMixTiling(gert::TilingContext *context)
        : BlockAttnResPrepareBaseTiling(context)
    {}
    ~BlockAttnResPrepareMixTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    TilingDataView GetTilingDataView() const override;

private:
    struct TileCandidate;
    struct ResourceUsage;

    bool SelectMixTileShape();
    bool TrySelectTileForS(uint64_t sCandidate, uint64_t mixedCoreNum, uint64_t usableL1Bytes, uint64_t usableUbBytes);
    bool TrySelectTileForT(uint64_t sCandidate, uint64_t candidateT, uint64_t usableL1Bytes, uint64_t usableUbBytes);
    TileCandidate BuildTileCandidate(uint64_t sCandidate, uint64_t candidateT, uint64_t candidateD) const;
    ResourceUsage CalculateResourceUsage(const TileCandidate &candidate) const;
    bool DoesCandidateFit(const ResourceUsage &usage, uint64_t usableL1Bytes, uint64_t usableUbBytes) const;
    void ApplyTileCandidate(const TileCandidate &candidate);
    ge::graphStatus CalculateMixWork(uint64_t &totalWorkUnits);
    ge::graphStatus CalculateWorkspaceSize();
    void FillTilingData(uint64_t totalWorkUnits);

    BlockAttnResPrepareMixTilingData tilingData_{};
};

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_MIX_TILING_H
