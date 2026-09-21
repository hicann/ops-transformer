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
 * \file fp_matmul_allto_all_tiling_base.h
 * \brief
 */

#ifndef FP_MATMUL_ALLTO_ALL_TILING_910_93_H
#define FP_MATMUL_ALLTO_ALL_TILING_910_93_H

#include <string>
#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "mc2/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "../matmul_allto_all_tiling_base.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"
#include "mc2/matmul_allto_all//op_kernel/arch22/matmul_allto_all_tiling_data_910_93.h"
#include "mc2/matmul_allto_all//op_kernel/arch22/matmul_allto_all_tiling_key.h"

namespace MC2Tiling {

using namespace optiling;
using namespace mc2_matmul_v3_advanced;

class FpMatmulAllToAllTilingBaseA3 : public MatmulAllToAllTilingBase {
    friend class FpMatmulAllToAllHelper;

public:
    explicit FpMatmulAllToAllTilingBaseA3(gert::TilingContext *context);
    ~FpMatmulAllToAllTilingBaseA3() override = default;
    ge::graphStatus GetWorkspaceSize() override;

protected:
    void SetUserWorkSpace();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckA3NonQuantTensorDataType(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus CheckBsRankSizeRange();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoMMTiling();

    ge::graphStatus SetHcclTiling();
    void SetTilingInfo(MatmulAlltoAllTilingInfoA3 &tilingInfo) const;

private:
    MatmulAlltoAllTilingDataA3 localTilingData_;
    std::string socVersionStr_;

    void PrintMatmulAlltoAllTilingData(MatmulAlltoAllTilingDataA3 &matmulAlltoAllTilingDataA3);
    void PrintMatmulAlltoAllTilingInfo(const std::string &opName, MatmulAlltoAllTilingInfoA3 &tilingInfo);
};

class FpMatmulAllToAllHelper : public mc2_matmul_v3::Mc2MatmulV3BaseTiling {
public:
    FpMatmulAllToAllHelper(FpMatmulAllToAllTilingBaseA3 &matmulAlltoAllTilingA3, Mc2MatmulV3TilingData &data);
    ge::graphStatus GetShapeAttrsInfo() override;

private:
    FpMatmulAllToAllTilingBaseA3 &tilingProcesser_;
};

} // namespace MC2Tiling
#endif
