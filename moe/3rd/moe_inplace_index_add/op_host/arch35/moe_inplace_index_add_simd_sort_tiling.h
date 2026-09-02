/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_inplace_index_add_simd_sort_tiling.h
 * \brief
 */

#ifndef MOE_INPLACE_INDEX_ADD_SIMD_SORT_TILING_H_
#define MOE_INPLACE_INDEX_ADD_SIMD_SORT_TILING_H_

#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeInplaceIndexAddSimdSortTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumBefore);
TILING_DATA_FIELD_DEF(int64_t, ubIndexFactor);
TILING_DATA_FIELD_DEF(int64_t, afterAxisFactor);

TILING_DATA_FIELD_DEF(int64_t, eachCoreIndexCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndexCount);
TILING_DATA_FIELD_DEF(int64_t, mainCoreIndicesLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreIndicesLoop);
TILING_DATA_FIELD_DEF(int64_t, mainCoreTailIndices);
TILING_DATA_FIELD_DEF(int64_t, tailCoreTailIndices);

/* pre */
TILING_DATA_FIELD_DEF(int64_t, eachCorePreAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailCorePreAxisCount);
TILING_DATA_FIELD_DEF(int64_t, updateLoopSize);
TILING_DATA_FIELD_DEF(int64_t, updateTailNum);
TILING_DATA_FIELD_DEF(int64_t, indicesLoopSize);
TILING_DATA_FIELD_DEF(int64_t, indiceAxisTailNum);
/* after */
TILING_DATA_FIELD_DEF(int64_t, eachCoreAfterAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAfterAxisCount);
TILING_DATA_FIELD_DEF(int64_t, tailUpdateLoopSize);
TILING_DATA_FIELD_DEF(int64_t, tailUpdateAxisNum);
TILING_DATA_FIELD_DEF(int64_t, isSplitPreAxis);
TILING_DATA_FIELD_DEF(int64_t, isSplitAfterAxis);
TILING_DATA_FIELD_DEF(int64_t, isSplitIndicesAxis);
TILING_DATA_FIELD_DEF(int64_t, isWithAlpha);
TILING_DATA_FIELD_DEF(int64_t, indicesStride);
TILING_DATA_FIELD_DEF(int64_t, indicesCastMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_200000, MoeInplaceIndexAddSimdSortTilingData)

class MoeInplaceIndexAddSimdSortTiling : public MoeInplaceIndexAddTiling {
public:
    explicit MoeInplaceIndexAddSimdSortTiling(gert::TilingContext *context)
        : MoeInplaceIndexAddTiling(context)
    {}
    ~MoeInplaceIndexAddSimdSortTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void DoOpTilingSplitAfterSort();
    void DoOpTilingSplitPreSort();
    void DoOpTilingSplitIndicesSort();
    int64_t GetIndicesAlignBlockSize(int64_t indicesFactor);
    int64_t GetAfterAlignBlockSize(int64_t indicesFactor, int64_t afterFactor);
    void SetTilingData();
    MoeInplaceIndexAddSimdSortTilingData tilingData_;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INPLACE_INDEX_ADD_TILING_H_
