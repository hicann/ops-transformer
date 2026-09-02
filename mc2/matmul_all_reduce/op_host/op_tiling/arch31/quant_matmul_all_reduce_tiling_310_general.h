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
 * \file quant_matmul_all_reduce_tiling_310_general.h
 * \brief
 */
#ifndef QUANT_MATMUL_ALL_REDUCE_TILING_310_GENERAL_H
#define QUANT_MATMUL_ALL_REDUCE_TILING_310_GENERAL_H

#include "../matmul_all_reduce_tiling_base.h"
#include "../arch22/quant_matmul_all_reduce_tiling.h"
namespace optiling {
class QuantMatmulAllReduceTiling310General : public MatmulAllReduceTilingBase {
    class QuantTilingTransferHelper : public Mc2QuantBatchMatmulV3Tiling {
    public:
        QuantTilingTransferHelper(QuantMatmulAllReduceTiling310General &quantMatmulAllReduceTiling,
                                  Mc2QuantBatchMatmulV3TilingData &data)
            : Mc2QuantBatchMatmulV3Tiling(quantMatmulAllReduceTiling.context_, &data),
              tilingProcesser_(quantMatmulAllReduceTiling)
        {}

        ge::graphStatus CheckInputInfo()
        {
            bool transposeMatch = (tilingProcesser_.args_.isATrans == 0 && tilingProcesser_.args_.isBTrans == 1);
            OP_TILING_CHECK(
                !transposeMatch,
                OP_LOGE_FOR_INVALID_VALUE(tilingProcesser_.opName_, "transA and transB",
                                          (std::string("transA=") + std::to_string(tilingProcesser_.args_.isATrans) +
                                           ", transB=" + std::to_string(tilingProcesser_.args_.isBTrans))
                                              .c_str(),
                                          "transA=false, transB=true"),
                return ge::GRAPH_FAILED);
            constexpr int64_t K_ALIGN_SIZE_A8W8_310 = 32;
            bool kAligned = (tilingProcesser_.args_.kValue % K_ALIGN_SIZE_A8W8_310 == 0);
            OP_TILING_CHECK(!kAligned,
                            OP_LOGE_FOR_INVALID_VALUE(tilingProcesser_.opName_, "x1",
                                                      std::to_string(tilingProcesser_.args_.kValue).c_str(),
                                                      "should be 32 bytes aligned"),
                            return ge::GRAPH_FAILED);
            constexpr int64_t N_ALIGN_SIZE_A8W8_310 = 16;
            bool nAligned = (tilingProcesser_.args_.nValue % N_ALIGN_SIZE_A8W8_310 == 0);
            OP_TILING_CHECK(!nAligned,
                            OP_LOGE_FOR_INVALID_VALUE(tilingProcesser_.opName_, "x2",
                                                      std::to_string(tilingProcesser_.args_.nValue).c_str(),
                                                      "should be 16 aligned"),
                            return ge::GRAPH_FAILED);
            auto weightTensor = context_->GetInputDesc(static_cast<size_t>(ParamValue::WEIGHT));
            OP_TILING_CHECK(weightTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "weight"),
                            return ge::GRAPH_FAILED);
            auto format = weightTensor->GetStorageFormat();
            bool isWeightNZ = GetPrimaryFormat(format) == ge::Format::FORMAT_FRACTAL_NZ;
            OP_TILING_CHECK(!isWeightNZ,
                            OP_LOGE_FOR_INVALID_FORMAT(
                                tilingProcesser_.opName_, "x2", std::to_string(static_cast<int32_t>(format)).c_str(),
                                std::to_string(static_cast<int32_t>(ge::Format::FORMAT_FRACTAL_NZ)).c_str()),
                            return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        }

        ge::graphStatus GetShapeAttrsInfo() override
        {
            OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
            if (CheckInputInfo() == ge::GRAPH_FAILED) {
                return ge::GRAPH_FAILED;
            }
            auto &&tilingArgs = tilingProcesser_.args_;
            inputParams_.opName = tilingProcesser_.opName_;
            inputParams_.transA = tilingArgs.isATrans;
            inputParams_.transB = tilingArgs.isBTrans;
            inputParams_.hasBias = tilingArgs.isBias;
            inputParams_.mSize = tilingArgs.mValue;
            inputParams_.kSize = tilingArgs.kValue;
            inputParams_.nSize = tilingArgs.nValue;
            inputParams_.batchA = 1U;
            inputParams_.batchB = 1U;
            inputParams_.batchC = 1U;
            inputParams_.batchBias = 1U;
            inputParams_.aDtype = tilingArgs.geAType;
            inputParams_.bDtype = tilingArgs.geBType;
            inputParams_.cDtype = tilingArgs.geCType;
            inputParams_.biasDtype = tilingArgs.geBiasType;
            inputParams_.isPerTensor = tilingProcesser_.isPerTensor_;
            inputParams_.outDtype = 1; // default fp16
            inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
            // Ascend310P does not support bf16;
            inputParams_.bf16ExtreWorkSpaceSize = 0;
            PrintTilingInputParam(inputParams_);
            return ge::GRAPH_SUCCESS;
        }

        void PrintTilingInputParam(Mc2QuantBatchMatmulInfo &info)
        {
            OP_LOGD(tilingProcesser_.opName_, " transA %d transB %d, hasBias %d, mSize %ld, kSize %ld, nSize %ld ",
                    info.transA, info.transB, info.hasBias, info.mSize, info.kSize, info.nSize);
            OP_LOGD(tilingProcesser_.opName_, " aDtype %d bDtype %d cDtype %d biasDtype %d isPerTensor %d ",
                    info.aDtype, info.bDtype, info.cDtype, info.biasDtype, info.isPerTensor);
            OP_LOGD(tilingProcesser_.opName_, " outDtype %ld libApiWorkSpaceSize %u bf16ExtreWorkSpaceSize %lu ",
                    info.outDtype, info.libApiWorkSpaceSize, info.bf16ExtreWorkSpaceSize);
        }

        ge::graphStatus PostTiling() override
        {
            PrintTilingData();
            tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
            OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
            return ge::GRAPH_SUCCESS;
        }

    private:
        QuantMatmulAllReduceTiling310General &tilingProcesser_;
    };

public:
    explicit QuantMatmulAllReduceTiling310General(gert::TilingContext *context)
        : MatmulAllReduceTilingBase(context)
    {}
    ~QuantMatmulAllReduceTiling310General() override = default;

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

    ge::graphStatus DoQuantTiling();

private:
    Mc2Tiling::QuantMatmulAllReduceTilingData quantMatmulAllReduceTilingData_{};
    uint64_t myWorkSpaceSize_{0U};
};
} // namespace optiling
#endif // QUANT_MATMUL_ALL_REDUCE_TILING_310_GENERAL_H
