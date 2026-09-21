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
 * \file allto_all_fp_matmul_tiling_910_93.h
 * \brief
 */

#ifndef ALLTO_ALL_FP_MATMUL_TILING_910_93_H
#define ALLTO_ALL_FP_MATMUL_TILING_910_93_H

#include <string>

#include "../allto_all_matmul_tiling_base.h"
#include "mc2/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mc2/allto_all_matmul/op_kernel/arch22/allto_all_matmul_tiling_data_910_93.h"
#include "mc2/allto_all_matmul/op_kernel/arch22/allto_all_matmul_tiling_key.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "securec.h"

namespace MC2Tiling {

using namespace optiling;
using namespace mc2_matmul_v3_advanced;

class AllToAllFpMatmulTilingBaseA3 : public AllToAllMatmulTilingBase {
    friend class AllToAllFpMatmulHelper;

public:
    explicit AllToAllFpMatmulTilingBaseA3(gert::TilingContext *context);
    ~AllToAllFpMatmulTilingBaseA3() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckA3NonQuantTensorDataType(const gert::TilingContext *context, const char *opName);
    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoMMTiling();

    ge::graphStatus SetHcclTiling();
    void SetTilingInfo(AlltoAllMatmulTilingInfoA3 &tilingInfo) const;

private:
    void PrintAlltoAllMatmulTilingData(AlltoAllMatmulTilingDataA3 &alltoAllMatmulTilingDataA3);
    void PrintAlltoAllMatmulTilingInfo(const std::string &opName, AlltoAllMatmulTilingInfoA3 &tilingInfo);

    AlltoAllMatmulTilingDataA3 localTilingData_;
    std::string socVersionStr_;
};

class AllToAllFpMatmulHelper : public mc2_matmul_v3::Mc2MatmulV3BaseTiling {
public:
    AllToAllFpMatmulHelper(AllToAllFpMatmulTilingBaseA3 &alltoAllMatmulTilingA3, Mc2MatmulV3TilingData &data);

    ge::graphStatus GetShapeAttrsInfo() override;

private:
    AllToAllFpMatmulTilingBaseA3 &tilingProcesser_;
};

} // namespace MC2Tiling
#endif
