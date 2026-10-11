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
 * \file matmul_all_reduce_add_rms_norm_tiling.h
 * \brief
 */

#ifndef _MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#define _MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#include "../../../../matmul_all_reduce/op_host/op_tiling/arch22/matmul_all_reduce_tiling_910.h"
#include "common_add_rms_norm_tiling.h"
#include "context_transfer.h"
#include "../../../op_kernel/matmul_all_reduce_add_rms_norm_tiling_data.h"

namespace optiling {
class MMNTilingTransferHelper;
class MatmulAllReduceAddRmsNormTiling : public TilingBaseClass {
    friend class MMNTilingTransferHelper;

public:
    explicit MatmulAllReduceAddRmsNormTiling(gert::TilingContext* context);
    ~MatmulAllReduceAddRmsNormTiling() override = default;

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus CheckMRNInput(const MRNCtxInfo& mrnCtxInfo);

private:
    bool HasTail() const;
    MRNCtxInfo mrnCtxInfo_;
    Mc2Tiling::MatmulAllReduceAddRmsNormTilingData tilingData_;
    bool hasTail_;
    TilingOut tilingOutAddRmsNormTile_;
    TilingOut tilingOutAddRmsNormTail_;
    std::unique_ptr<MMNTilingTransferHelper> helper_;
};

class MMNTilingTransferHelper : public MatmulAllReduceTiling910 {
public:
    MMNTilingTransferHelper(MatmulAllReduceAddRmsNormTiling& MatmulAllReduceAddRmsNormTiling,
                            Mc2Tiling::MatmulAllReduce910TilingData& data);
    ge::graphStatus GetShapeAttrsInfo() override;

private:
    MatmulAllReduceAddRmsNormTiling& tilingProcesser_;
};
} // namespace optiling

#endif // _MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
