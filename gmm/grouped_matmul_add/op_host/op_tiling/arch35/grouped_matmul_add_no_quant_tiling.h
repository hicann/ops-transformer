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
 * \file grouped_matmul_add_no_quant_tiling.h
 * \brief
 */
#ifndef __GROUPED_MATMUL_ADD_NO_QUANT_TILING_H
#define __GROUPED_MATMUL_ADD_NO_QUANT_TILING_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "../../../op_kernel/arch35/grouped_matmul_add_tiling_data.h"
#include "grouped_matmul_add_utils_advanced.h"
#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_tiling_base.h"

namespace optiling {
using namespace GroupedMatmulAdd;

class GroupedMatmulAddNoQuantTiling : public GroupedNoQuantTilingBase {
public:
    GroupedMatmulAddNoQuantTiling()
        : GroupedNoQuantTilingBase(INDEX_GROUPLIST)
    {}
    bool SetTiling(gert::TilingContext* context);

protected:
    bool Init(const gert::TilingContext* context);
    bool GetAttrs(const gert::TilingContext* context);
    bool CalMatMulTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
    void SetGMMTiling();
    void SetMatMulTiling();
    void PrintTilingResult(const gert::TilingContext* context);
    void SetTilingKey(gert::TilingContext* context) const;

private:
    GroupedMatmulAdd::GmmAddTilingDataParams tilingData_;
};
} // namespace optiling

#endif // GROUPED_MATMUL_ADD_NO_QUANT_TILING_H
