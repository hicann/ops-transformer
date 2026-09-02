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
 * \file quant_matmul_all_reduce_tiling_950.h
 * \brief
 */
#ifndef QUANT_MATMUL_ALL_REDUCE_TILING_950_H
#define QUANT_MATMUL_ALL_REDUCE_TILING_950_H

#include "../matmul_all_reduce_tiling_base.h"
#include "../../../op_kernel/arch35/matmul_all_reduce_tiling_struct_ar35.h"

namespace optiling {

struct QuantMMAllReduceTPLParam {
    bool transB{false};
    uint64_t kernelType{0};
};

class QuantMatmulAllReduceTilingA5 : public MatmulAllReduceTilingBase {
    friend class QuantTilingTransferHelperA5;

public:
    explicit QuantMatmulAllReduceTilingA5(gert::TilingContext *context);
    QuantMatmulAllReduceTilingA5(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                 Mc2Tiling::QuantMatmulAllReduceTilingDataA5 *out);
    ~QuantMatmulAllReduceTilingA5() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Tiling::RCSTiling &MutableRCSTilingData() override;

    ::TCubeTiling &MutableTCubeTileTilingData() override;

    ::TCubeTiling &MutableTCubeTailTilingData() override;

    void PrintExtendMatmulTiling(bool isTail) override;

    CutResult GetTilingResult() override;

    ge::graphStatus CheckHCCLSize();
    // When FP8 comm tiling result exceeds HCCL memory limit, re-split M dimension
    ge::graphStatus AdjustHCCLLimit();

    ge::graphStatus DoQuantTiling();
    ge::graphStatus GetWorkspaceSizeForA2ARSAG(const uint64_t gmcFloat);
    ge::graphStatus GetWorkspaceSizeOfCommQuantScaleOrFP8(const uint64_t gmcFloat);
    ge::graphStatus SetMc2HcommAllReduce(const char *groupName, const uint32_t reduceType);
    ge::graphStatus SetMc2HcommTwoShot(const char *groupName, const uint32_t reduceType, const uint8_t dataType);
    ge::graphStatus SetMc2HcommRSAG(const char *groupName, const uint32_t reduceType);
    ge::graphStatus SetMc2Hcomm();

    ge::graphStatus CheckInput() override;

    bool IsPerTensorDequantScale(const gert::Shape &dequantShape) const override;

    ge::graphStatus CheckDequantScaleType();
    ge::graphStatus CheckCommQuantScale();
    ge::graphStatus CheckBias();
    ge::graphStatus CheckX1X2();
    ge::graphStatus CheckA8W8ScenarioScaleType();
    ge::graphStatus CheckMXFPScenarioScaleType();
    ge::graphStatus CheckQuantGroupSize();
    ge::graphStatus GetDynamicQuantTempBuffSize();

private:
    ge::graphStatus CheckAxisSize();
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 quantMatmulAllReduceTilingDataSelf_{};
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5 &quantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
    bool isCommInt8Enable_ = false;
    bool isCommFp8Enable_ = false;
    QuantMMAllReduceTPLParam quantTPlparam_;
};

class QuantTilingTransferHelperA5 : public Mc2AdaptiveSlidingWindowTiling {
public:
    QuantTilingTransferHelperA5(QuantMatmulAllReduceTilingA5 &quantMatmulAllReduceTiling,
                                DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &data)
        : Mc2AdaptiveSlidingWindowTiling(quantMatmulAllReduceTiling.context_, &data),
          tilingProcesser_(quantMatmulAllReduceTiling)
    {}

    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape &GetScaleShape(const size_t index) override;
    const gert::StorageShape *GetOffsetShape(const size_t index); // matmulV3还未回合
    const gert::StorageShape *GetPertokenShape(const size_t index) override;
    const gert::StorageShape *GetBiasShape(const size_t index) override;
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo quantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;
    QuantMMAllReduceTPLParam GetQuantMMAllReduceTPLParam(const uint64_t kernelType);

private:
    QuantMatmulAllReduceTilingA5 &tilingProcesser_;
};

} // namespace optiling
#endif // QUANT_MATMUL_ALL_REDUCE_TILING_950_H
