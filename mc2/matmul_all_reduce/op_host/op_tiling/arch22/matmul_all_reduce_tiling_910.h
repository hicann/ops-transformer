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
 * \file matmul_all_reduce_tiling_910.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_910_H
#define MATMUL_ALL_REDUCE_TILING_910_H

#include "../matmul_all_reduce_tiling_base.h"
#include "../../../op_kernel/matmul_all_reduce_tiling_key.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"

namespace optiling {

struct MatmulTPLParam {
    uint64_t disableMixNd2nz{65535};
};

class MatmulAllReduceTiling910 : public MatmulAllReduceTilingBase {
    friend class MMNTilingTransferHelper;
    friend class TilingTransferHelper;
    friend class MatmulAllReduceAddRmsNormTiling;

public:
    explicit MatmulAllReduceTiling910(gert::TilingContext* context);
    MatmulAllReduceTiling910(gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo,
                             Mc2Tiling::MatmulAllReduce910TilingData* out);
    ~MatmulAllReduceTiling910() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus GetWorkspaceSize() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;

    ge::graphStatus Do910Tiling();

    Mc2Tiling::Mc2Msg& MutableMc2MsgData() override;

    Mc2Tiling::RCSTiling& MutableRCSTilingData() override;

    AscendC::tiling::TCubeTiling& MutableTCubeTileTilingData() override;

    AscendC::tiling::TCubeTiling& MutableTCubeTailTilingData() override;

    void DoEmptyTensorTiling() override;

    ge::graphStatus CheckInput() override;

    CutResult GetTilingResult() override;

private:
    ge::graphStatus CheckAxisSize();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputFormat();
    ge::graphStatus CheckInputShape();
    Mc2Tiling::MatmulAllReduce910TilingData matmulAllReduce910TilingDataSelf_{};
    Mc2Tiling::MatmulAllReduce910TilingData& matmulAllReduce910TilingData_;
    uint64_t myWorkSpaceSize_{0U};
    MatmulTPLParam matmulTPLParam_;
};

class TilingTransferHelper : public mc2_matmul_v3::Mc2MatmulV3BaseTiling {
public:
    TilingTransferHelper(MatmulAllReduceTiling910& matmulAllReduceTiling910, Mc2MatmulV3TilingData& data);

    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
    MatmulTPLParam GetMatmulTPLParam();

private:
    MatmulAllReduceTiling910& tilingProcesser_;
};
} // namespace optiling
#endif // MATMUL_ALL_REDUCE_TILING_910_H
