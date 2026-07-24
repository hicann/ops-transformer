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
 * \file quant_matmul_all_reduce_tiling.h
 * \brief
 */
#ifndef QUANT_MATMUL_ALL_REDUCE_TILING_H
#define QUANT_MATMUL_ALL_REDUCE_TILING_H

#include "../matmul_all_reduce_tiling_base.h"
#include "../../../op_kernel/matmul_all_reduce_tiling_key.h"
#include "../../../op_kernel/quant_matmul_all_reduce_tiling_data.h"

namespace optiling {
struct QuantMatmulTPLParam {
    uint64_t trans{65535};
    uint64_t isPertoken{65535};
};
class QuantMatmulAllReduceTiling : public MatmulAllReduceTilingBase {
    friend class QuantTilingTransferHelper;
    friend class QuantMatmulAllReduceAddRmsNormTiling;

public:
    explicit QuantMatmulAllReduceTiling(gert::TilingContext *context);
    QuantMatmulAllReduceTiling(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                               Mc2Tiling::QuantMatmulAllReduceTilingData *out);
    ~QuantMatmulAllReduceTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Tiling::Mc2Msg &MutableMc2MsgData() override;

    Mc2Tiling::RCSTiling &MutableRCSTilingData() override;

    AscendC::tiling::TCubeTiling &MutableTCubeTileTilingData() override;

    AscendC::tiling::TCubeTiling &MutableTCubeTailTilingData() override;

    ge::graphStatus DoQuantTiling();

    ge::graphStatus CheckInput() override;

    ge::graphStatus CheckDequantScaleType();

    CutResult GetTilingResult() override;

private:
    ge::graphStatus CheckAxisSize();
    Mc2Tiling::QuantMatmulAllReduceTilingData quantMatmulAllReduceTilingDataSelf_{};
    Mc2Tiling::QuantMatmulAllReduceTilingData &quantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
    bool isCommInt8Enable_ = false;
    QuantMatmulTPLParam quantMatmulTPLParam_;
};

class QuantTilingTransferHelper : public Mc2QuantBatchMatmulV3Tiling {
public:
    QuantTilingTransferHelper(QuantMatmulAllReduceTiling &quantMatmulAllReduceTiling,
                              Mc2QuantBatchMatmulV3TilingData &data);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape &GetScaleShape(const size_t index) override;
    const gert::StorageShape *GetPertokenShape(const size_t index) override;
    const gert::StorageShape *GetBiasShape(const size_t index) override;
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo quantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;
    QuantMatmulTPLParam GetQuantMatmulTPLParam();

private:
    QuantMatmulAllReduceTiling &tilingProcesser_;
};
} // namespace optiling
#endif // QUANT_MATMUL_ALL_REDUCE_TILING_H
