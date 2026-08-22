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
 * \file weight_quant_matmul_all_reduce_tiling_310p.h
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_310P_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_310P_H
#include "../matmul_all_reduce_tiling_base.h"

namespace optiling {

struct WeightQuantMatmul310TPLParam {
    uint64_t hasAntiquantOffset{65535};
    uint64_t antiQuantType{65535};
    uint64_t transA{65535};
    uint64_t transB{65535};
};

class WeightQuantMatmulAllReduceTiling310P : public MatmulAllReduceTilingBase {
    class WeightQuantTilingTransferHelper : public Mc2WeightQuantBatchMatmulV2WeightNz {
    public:
        WeightQuantTilingTransferHelper(WeightQuantMatmulAllReduceTiling310P &weightQuantMatmulAllReduceTiling,
                                        Mc2WeightQuantBatchMatmulV2NzTilingData &data)
            : Mc2WeightQuantBatchMatmulV2WeightNz(weightQuantMatmulAllReduceTiling.context_, &data),
              tilingProcesser_(weightQuantMatmulAllReduceTiling)
        {}
        ge::graphStatus GetShapeAttrsInfo() override
        {
            OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
            auto &&tilingArgs = tilingProcesser_.args_;
            inputParams_.opName = tilingProcesser_.opName_;
            inputParams_.transA = tilingArgs.isATrans;
            inputParams_.transB = tilingArgs.isBTrans;
            inputParams_.hasBias = tilingArgs.isBias;
            inputParams_.hasAntiQuantOffset = tilingProcesser_.HasAntiQuantOffset();
            inputParams_.mSize = tilingArgs.mValue;
            inputParams_.kSize = tilingArgs.kValue;
            inputParams_.nSize = tilingArgs.nValue;
            inputParams_.aDtype = tilingArgs.geAType;
            inputParams_.bDtype = tilingArgs.geBType;
            inputParams_.cDtype = tilingArgs.geCType;
            inputParams_.biasDtype = tilingArgs.geBiasType;
            inputParams_.antiQuantType = tilingProcesser_.antiQuantType_;
            inputParams_.groupSize = tilingProcesser_.antiGroupSize_;
            inputParams_.quantType = tilingProcesser_.quantType_;
            PrintTilingInputParam(inputParams_);
            return ge::GRAPH_SUCCESS;
        }
        void PrintTilingInputParam(Mc2WeightQuantBatchMatmulInfo &weightQuantBatchMatmulInfo)
        {
            OP_LOGD(tilingProcesser_.opName_, " transA_ %d transB_ %d, hasBias_ %d, hasAntiQuantOffset_ %d, ",
                    weightQuantBatchMatmulInfo.transA, weightQuantBatchMatmulInfo.transB,
                    weightQuantBatchMatmulInfo.hasBias, weightQuantBatchMatmulInfo.hasAntiQuantOffset);
            OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld groupSize_ %ld",
                    weightQuantBatchMatmulInfo.mSize, weightQuantBatchMatmulInfo.kSize,
                    weightQuantBatchMatmulInfo.nSize, weightQuantBatchMatmulInfo.groupSize);
            OP_LOGD(tilingProcesser_.opName_, "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d",
                    static_cast<int32_t>(weightQuantBatchMatmulInfo.aDtype),
                    static_cast<int32_t>(weightQuantBatchMatmulInfo.bDtype),
                    static_cast<int32_t>(weightQuantBatchMatmulInfo.cDtype),
                    static_cast<int32_t>(weightQuantBatchMatmulInfo.biasDtype));
            OP_LOGD(tilingProcesser_.opName_, "antiQuantType_ %d quantType_ %d",
                    static_cast<int32_t>(weightQuantBatchMatmulInfo.antiQuantType),
                    static_cast<int32_t>(weightQuantBatchMatmulInfo.quantType));
        }

        void PrintTilingData(bool debugLevel)
        {
            std::stringstream ss;
            ss << " cubeNumBlocksN: " << static_cast<uint32_t>(tilingData_->cubeBlockDimN)
               << " cubeNumBlocksM: " << static_cast<uint32_t>(tilingData_->cubeBlockDimM)
               << " AL1Pingpong: " << static_cast<uint32_t>(tilingData_->AL1Pingpong)
               << " BL1Pingpong: " << static_cast<uint32_t>(tilingData_->BL1Pingpong)
               << " kAlign: " << tilingData_->kAlign << " nAlign: " << tilingData_->nAlign
               << " mSize: " << tilingData_->mSize << " kSize: " << tilingData_->kSize
               << " nSize: " << tilingData_->nSize << " groupSize: " << tilingData_->groupSize
               << " mAubSize: " << tilingData_->mAubSize << " kAubSize: " << tilingData_->kAubSize
               << " nBubSize: " << tilingData_->nBubSize << " kBubSize: " << tilingData_->kBubSize
               << " mCubSize: " << tilingData_->mCubSize << " nCubSize: " << tilingData_->nCubSize
               << " mAL1Size: " << tilingData_->mAL1Size << " kAL1Size: " << tilingData_->kAL1Size
               << " nBL1Size: " << tilingData_->nBL1Size << " kBL1Size: " << tilingData_->kBL1Size;

            int32_t logLevel = debugLevel ? DLOG_DEBUG : DLOG_ERROR;
            OPS_LOG_FULL(logLevel, inputParams_.opName, "tiling data: %s", ss.str().c_str());
            PrintMatMulTiling(logLevel);
        }

        void PrintMatMulTiling([[maybe_unused]] int32_t logLevel)
        {
            std::stringstream ss;
            auto &matmulTiling = tilingData_->matmulTiling;
            ss << "usedCoreNum " << matmulTiling.usedCoreNum << " M " << matmulTiling.M << " N " << matmulTiling.N
               << " Ka " << matmulTiling.Ka << " Kb " << matmulTiling.Kb << " singleCoreM " << matmulTiling.singleCoreM
               << " singleCoreN " << matmulTiling.singleCoreN << " singleCoreK " << matmulTiling.singleCoreK
               << " baseM " << matmulTiling.baseM << " baseN " << matmulTiling.baseN << " baseK " << matmulTiling.baseK
               << " depthA1 " << matmulTiling.depthA1 << " depthB1 " << matmulTiling.depthB1 << " stepM "
               << matmulTiling.stepM << " stepN " << matmulTiling.stepN << " isBias " << matmulTiling.isBias
               << " transLength " << matmulTiling.transLength << " iterateOrder " << matmulTiling.iterateOrder
               << " shareMode " << matmulTiling.shareMode << " shareL1Size " << matmulTiling.shareL1Size
               << " shareL0CSize " << matmulTiling.shareL0CSize << " shareUbSize " << matmulTiling.shareUbSize
               << " batchM " << matmulTiling.batchM << " batchN " << matmulTiling.batchN << " stepKa "
               << matmulTiling.stepKa << " stepKb " << matmulTiling.stepKb << " dbL0A " << matmulTiling.dbL0A
               << " dbL0B " << matmulTiling.dbL0B << " dbL0C " << matmulTiling.dbL0C;

            OPS_LOG_FULL(logLevel, inputParams_.opName, "matmul tiling: %s", ss.str().c_str());
        }

        ge::graphStatus PostTiling() override
        {
            PrintTilingData(true);
            tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
            OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
            return ge::GRAPH_SUCCESS;
        }

        WeightQuantMatmul310TPLParam GetWeightQuantMatmul310TPLParam()
        {
            WeightQuantMatmul310TPLParam param;
            param.hasAntiquantOffset = inputParams_.hasAntiQuantOffset;
            param.antiQuantType = static_cast<uint64_t>(inputParams_.antiQuantType);
            param.transA = inputParams_.transA;
            param.transB = inputParams_.transB;
            return param;
        }

    private:
        WeightQuantMatmulAllReduceTiling310P &tilingProcesser_;
    };

public:
    explicit WeightQuantMatmulAllReduceTiling310P(gert::TilingContext *context)
        : MatmulAllReduceTilingBase(context)
    {}

    ~WeightQuantMatmulAllReduceTiling310P() override = default;

protected:
    bool IsCapable() override;

    void UpdateCommOffset();

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Tiling::Mc2Msg &MutableMc2MsgData() override;

    Mc2Tiling::RCSTiling &MutableRCSTilingData() override;

    AscendC::tiling::TCubeTiling &MutableTCubeTileTilingData() override;

    AscendC::tiling::TCubeTiling &MutableTCubeTailTilingData() override;

    CutResult GetTilingResult() override;

    ge::graphStatus DoWeightQuantTiling();

private:
    Mc2Tiling::WeightQuantMatmulAllReduceNzTilingData weightQuantMatmulAllReduceTilingData_{};
    uint64_t myWorkSpaceSize_{0U};
    uint64_t tileTilingKey_{0U};
    WeightQuantMatmul310TPLParam weightQuantMatmul310TPLParam_;
};
} // namespace optiling
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_310P_H
