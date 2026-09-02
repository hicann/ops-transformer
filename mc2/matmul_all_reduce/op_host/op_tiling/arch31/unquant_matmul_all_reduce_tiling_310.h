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
 * \file unquant_matmul_all_reduce_tiling_310.h
 * \brief
 */
#ifndef UNQUANT_MATMUL_ALL_REDUCE_TILING_310_H
#define UNQUANT_MATMUL_ALL_REDUCE_TILING_310_H

#include "../matmul_all_reduce_tiling_base.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"

namespace optiling {

struct Matmul310TPLParam {
    uint64_t disableMixNd2nz{65535};
};
class UnQuantMatmulAllReduceTiling310 : public MatmulAllReduceTilingBase {
    class UnQuantTilingTransferHelper : public mc2_matmul_v3::Mc2MatmulV3BaseTiling {
    public:
        UnQuantTilingTransferHelper(UnQuantMatmulAllReduceTiling310 &unquantMatmulAllReduceTiling,
                                    Mc2MatmulV3TilingData &data)
            : Mc2MatmulV3BaseTiling(unquantMatmulAllReduceTiling.context_, &data),
              tilingProcesser_(unquantMatmulAllReduceTiling)
        {}

        ge::graphStatus CheckInputInfo()
        {
            return ge::GRAPH_SUCCESS;
        }

        ge::graphStatus GetShapeAttrsInfo() override
        {
            OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
            if (CheckInputInfo() == ge::GRAPH_FAILED) {
                return ge::GRAPH_FAILED;
            }

            auto &&tilingArgs = tilingProcesser_.args_;
            args_.opName = tilingProcesser_.opName_;
            args_.isATrans = tilingArgs.isATrans;
            args_.isBTrans = tilingArgs.isBTrans;
            args_.isHf32 = false;
            args_.hasBias = tilingArgs.isBias;
            args_.nd2nzA = false;
            args_.nd2nzB = false;
            args_.aType = tilingArgs.geAType;
            args_.bType = tilingArgs.geBType;
            args_.cType = tilingArgs.geCType;
            args_.biasType = tilingArgs.geBiasType;
            args_.aFormat = ge::FORMAT_ND;
            args_.bFormat = ge::FORMAT_FRACTAL_NZ;
            args_.outFormat = ge::FORMAT_ND;
            args_.mValue = tilingArgs.mValue;
            args_.kValue = tilingArgs.kValue;
            args_.nValue = tilingArgs.nValue;
            args_.l2Ratio = 0;
            return ge::GRAPH_SUCCESS;
        }

        ge::graphStatus PostTiling() override
        {
            tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
            OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
            return ge::GRAPH_SUCCESS;
        }

        Matmul310TPLParam GetMatmulTPLParam()
        {
            Matmul310TPLParam param;
            param.disableMixNd2nz = static_cast<uint64_t>(GetMixNd2nzType());
            // 1: disable mix nd2nz 0: enable mix nd2nz
            return param;
        }

    private:
        UnQuantMatmulAllReduceTiling310 &tilingProcesser_;
    };

public:
    explicit UnQuantMatmulAllReduceTiling310(gert::TilingContext *context)
        : MatmulAllReduceTilingBase(context)
    {}
    ~UnQuantMatmulAllReduceTiling310() override = default;

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

    CutResult GetTilingResult() override;

    ge::graphStatus DoUnQuantTiling();

private:
    Mc2Tiling::UnQuantMatmulAllReduceTilingData unquantMatmulAllReduceTilingData_{};
    uint64_t myWorkSpaceSize_{0U};
    Matmul310TPLParam matmulTPLParam_;
};
} // namespace optiling
#endif // UNQUANT_MATMUL_ALL_REDUCE_TILING_310_H
