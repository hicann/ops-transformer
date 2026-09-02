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
 * \file moe_inplace_index_add_simt_sort_tiling.h
 * \brief
 */

#ifndef MOE_INPLACE_INDEX_ADD_SIMT_SORT_TILING_H_
#define MOE_INPLACE_INDEX_ADD_SIMT_SORT_TILING_H_

#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeInplaceIndexAddSimtSortTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, indicesUbFactor);          // 整循环搬运indices数量
TILING_DATA_FIELD_DEF(int64_t, indicesLoopSize);          // 整核搬运indices循环次数
TILING_DATA_FIELD_DEF(int64_t, indiceAxisTailNum);        // 尾核尾循环搬运indices数量
TILING_DATA_FIELD_DEF(int64_t, tailBlockIndicesLoopSize); // 尾核搬运indices循环次数
TILING_DATA_FIELD_DEF(int64_t, eachCoreIndexCount);       // 每个整核处理indices数量
TILING_DATA_FIELD_DEF(int64_t, sortShareBufSize);
TILING_DATA_FIELD_DEF(int64_t, normalUpdatesPreNum); // 整循环搬运update的pre份数
TILING_DATA_FIELD_DEF(int64_t, tailUpdatesPreNum);   // 尾循环搬运update的pre份数
TILING_DATA_FIELD_DEF(int64_t, updatesPreLoop);      // 搬1次indices时，要搬多份updates的循环次数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, indicesStride);
TILING_DATA_FIELD_DEF(int64_t, indicesCastMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_500000, MoeInplaceIndexAddSimtSortTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_500001, MoeInplaceIndexAddSimtSortTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_500010, MoeInplaceIndexAddSimtSortTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_500011, MoeInplaceIndexAddSimtSortTilingData)

class MoeInplaceIndexAddSimtSortTiling : public MoeInplaceIndexAddTiling {
public:
    explicit MoeInplaceIndexAddSimtSortTiling(gert::TilingContext *context)
        : MoeInplaceIndexAddTiling(context)
    {}
    ~MoeInplaceIndexAddSimtSortTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    uint64_t computeIndicesUbFactor();
    void TilingForSimtSort();
    void SetTilingData();
    MoeInplaceIndexAddSimtSortTilingData tilingData_;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INPLACE_INDEX_ADD_TILING_H_
