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
 * \file moe_inplace_index_add_simt_tiling.h
 * \brief
 */

#ifndef MOE_INPLACE_INDEX_ADD_SIMT_TILING_H_
#define MOE_INPLACE_INDEX_ADD_SIMT_TILING_H_

#include "moe_inplace_index_add_tiling_arch35.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeInplaceIndexAddSimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, colUbFactor);
TILING_DATA_FIELD_DEF(int64_t, indicesStride);
TILING_DATA_FIELD_DEF(int64_t, indicesUbFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100000, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100001, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100002, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100003, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100004, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100006, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100027, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100009, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100012, MoeInplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100100, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100101, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100102, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100103, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100104, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100106, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100127, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100109, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_100112, MoeInplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101000, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101001, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101002, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101003, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101004, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101006, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101027, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101009, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101012, MoeInplaceIndexAddSimtTilingData)

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101100, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101101, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101102, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101103, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101104, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101106, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101127, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101109, MoeInplaceIndexAddSimtTilingData)
REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd_101112, MoeInplaceIndexAddSimtTilingData)

class MoeInplaceIndexAddSimtTiling : public MoeInplaceIndexAddTiling {
public:
    explicit MoeInplaceIndexAddSimtTiling(gert::TilingContext *context)
        : MoeInplaceIndexAddTiling(context)
    {}
    ~MoeInplaceIndexAddSimtTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();
    MoeInplaceIndexAddSimtTilingData tilingData_;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INPLACE_INDEX_ADD_TILING_H_
