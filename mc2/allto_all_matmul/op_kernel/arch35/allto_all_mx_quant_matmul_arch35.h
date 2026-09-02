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
 * \file allto_all_mx_quant_matmul_arch35.h
 * \brief
 */

#ifndef ALLTO_ALL_MX_QUANT_MATMUL_ARCH35_H
#define ALLTO_ALL_MX_QUANT_MATMUL_ARCH35_H

namespace Mc2Kernel {
template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
class AlltoAllMxQuantMatmulArch35 {
public:
    __aicore__ inline AlltoAllMxQuantMatmulArch35(SchedulerType *pipeLine, uint64_t x1HcclDataType)
        : pipeLine_(pipeLine),
          x1HcclDataType_(x1HcclDataType){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale,
                                GM_ADDR x2_scale, GM_ADDR workspaceGM, AlltoAllMatmulTilingDataType *tilingData,
                                AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    SchedulerType *pipeLine_;
    SchedulerContextType pipeLineContext_;
    AlltoAllMatmulTilingDataType *tilingData_;
    AscendC::TPipe *tPipe_;
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR commX1ScaleGM1_;
    GM_ADDR commOutGM_;
    GM_ADDR transX1ScaleGM1_;
    GM_ADDR transOutGM_;
    uint64_t rankForComm_;
    uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
    uint64_t x1HcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E4M3;

private:
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t NUM_TWO = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
    __aicore__ inline void ProcessScale();
};

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType, IsMxFp4>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM, AlltoAllMatmulTilingDataType *tilingData, AscendC::TPipe *tPipe)
{
    // 获取tilingdata数据
    tilingData_ = tilingData;
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;
    // MX场景下，通信数据的存储顺序是commX1Scale、commX1、transX1Scale_、transOut
    commX1ScaleGM1_ = workspaceGM;
    uint64_t x1ScaleLen = CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO *
                          static_cast<uint64_t>(mc2Tiling_.rankM) * sizeof(AscendC::fp8_e8m0_t);
    x1ScaleLen = CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM; // 对齐
    commOutGM_ = workspaceGM + x1ScaleLen;
    transX1ScaleGM1_ = commOutGM_ + mc2Tiling_.commLen;
    transOutGM_ = all2all_out;
    if (all2all_out == nullptr) {
        transOutGM_ = transX1ScaleGM1_ + x1ScaleLen;
    }

    if constexpr (IsMxFp4) {
        constexpr uint32_t dataNumPerByte = 2;
        rankForComm_ = CeilDiv(tilingData_->alltoAllQuantMatmulTilingInfo.rankK, dataNumPerByte);
    } else {
        rankForComm_ = tilingData_->alltoAllQuantMatmulTilingInfo.rankK;
    }

    // 初始化流水线
    pipeLine_->Init();
    // 获取流水线上下文
    pipeLine_->GetContext(&pipeLineContext_);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType, IsMxFp4>::Process()
{
    if (pipeLine_ == nullptr || tilingData_ == nullptr) {
        return;
    }

    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;

    if (mc2Tiling_.rankDim == 0) {
        return;
    }

    // 启动scale通信和转置
    ProcessScale();

    // 启动主块流水
    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 启动尾块流水
    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    // 结束流水线
    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType, IsMxFp4>::ProcessScale()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;

    uint64_t rankKdivByGroupSize = CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO;
    // x1Scale通信相关地址
    pipeLineContext_.communicationContext->taskCnt = 1;
    pipeLineContext_.communicationContext->sendBuffer = x1Scale_;
    pipeLineContext_.communicationContext->recvBuffer = commX1ScaleGM1_;
    pipeLineContext_.communicationContext->sendCount =
        static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim * rankKdivByGroupSize;
    pipeLineContext_.communicationContext->strideCount = pipeLineContext_.communicationContext->sendCount;
    pipeLineContext_.communicationContext->hcclDataType = mxScaleHcclDataType_;

    // x1Scale转置相关地址和偏移
    pipeLineContext_.scaleTransposeContext->transposeSrcAddr = commX1ScaleGM1_;
    pipeLineContext_.scaleTransposeContext->transposeDstAddr = transX1ScaleGM1_;
    pipeLineContext_.scaleTransposeContext->nextSrcBlockOffset =
        static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim * rankKdivByGroupSize;
    pipeLineContext_.scaleTransposeContext->nextDstBlockOffset = rankKdivByGroupSize;
    pipeLineContext_.scaleTransposeContext->rankCnt = static_cast<uint64_t>(mc2Tiling_.rankDim);
    pipeLineContext_.scaleTransposeContext->innerAxis = rankKdivByGroupSize;
    pipeLineContext_.scaleTransposeContext->transM = static_cast<uint64_t>(mc2Tiling_.rankM) / mc2Tiling_.rankDim;
    pipeLineContext_.scaleTransposeContext->innerOffsetIn = rankKdivByGroupSize;
    pipeLineContext_.scaleTransposeContext->innerOffsetOut =
        rankKdivByGroupSize * static_cast<uint64_t>(mc2Tiling_.rankDim);

    // 启动x1Scale的通信和转置
    pipeLine_->ProcessScale();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType,
                                                   IsMxFp4>::ProcessTile(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    // 复用的变量
    uint64_t tileMMultiRankK = static_cast<uint64_t>(mc2Tiling_.tileM) * rankForComm_;

    // 通信相关地址和偏移
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_;
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_;
    pipeLineContext_.communicationContext->sendOffset = tileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tileMMultiRankK;
    pipeLineContext_.communicationContext->strideCount =
        static_cast<uint64_t>(mc2Tiling_.rankM) * rankForComm_ / mc2Tiling_.rankDim;
    pipeLineContext_.communicationContext->hcclDataType = x1HcclDataType_;

    // 转置相关地址和偏移
    pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_;
    pipeLineContext_.transposeContext->transposeDstAddr = transOutGM_;
    pipeLineContext_.transposeContext->transposeSrcOffset = tileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstOffset = tileMMultiRankK * mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->nextSrcBlockOffset =
        static_cast<uint64_t>(mc2Tiling_.rankM) * rankForComm_ / mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->nextDstBlockOffset = rankForComm_;
    pipeLineContext_.transposeContext->rankCnt = static_cast<uint64_t>(mc2Tiling_.rankDim);
    pipeLineContext_.transposeContext->innerAxis = rankForComm_;
    pipeLineContext_.transposeContext->transM = static_cast<uint64_t>(mc2Tiling_.tileM);
    pipeLineContext_.transposeContext->innerOffsetIn = rankForComm_;
    pipeLineContext_.transposeContext->innerOffsetOut = rankForComm_ * static_cast<uint64_t>(mc2Tiling_.rankDim);

    // quantMatmul相关的地址和偏移
    pipeLineContext_.computationContext->baseData.aGM = transOutGM_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = y_;
    pipeLineContext_.computationContext->baseData.biasGM = bias_;
    pipeLineContext_.computationContext->baseData.aOffset = tileMMultiRankK * mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bOffset = 0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        static_cast<uint64_t>(mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y));
    pipeLineContext_.computationContext->additionalData.x1Scale = transX1ScaleGM1_;
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset =
        static_cast<uint64_t>(mc2Tiling_.tileM) *
        CeilDiv(static_cast<uint64_t>(mc2Tiling_.rankK) * mc2Tiling_.rankDim, MXFP_GROUP_SIZE) * NUM_TWO *
        sizeof(AscendC::fp8_e8m0_t);
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2QuantMmTileTilingData);

    pipeLine_->Process(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType,
                                                   IsMxFp4>::ProcessTail(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    uint64_t tailMMultiRankK = static_cast<uint64_t>(mc2Tiling_.tailM) * rankForComm_;
    uint64_t tileCntMultitileMMultiRankK =
        static_cast<uint64_t>(mc2Tiling_.tileCnt) * static_cast<uint64_t>(mc2Tiling_.tileM) * rankForComm_;

    // 通信相关地址和偏移
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->sendOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tailMMultiRankK;

    // 转置相关地址和偏移
    pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstAddr =
        transOutGM_ + tileCntMultitileMMultiRankK * mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeSrcOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstOffset = tailMMultiRankK * mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transM = static_cast<uint64_t>(mc2Tiling_.tailM);

    // quantMatmul相关的地址和偏移
    uint64_t tempScaleKBytes = CeilDiv(static_cast<uint64_t>(mc2Tiling_.rankK) * mc2Tiling_.rankDim, MXFP_GROUP_SIZE) *
                               NUM_TWO * sizeof(AscendC::fp8_e8m0_t);
    pipeLineContext_.computationContext->baseData.aGM =
        transOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1) * mc2Tiling_.rankDim;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM =
        y_ + static_cast<uint64_t>(mc2Tiling_.tileCnt) * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->baseData.aOffset = tailMMultiRankK * mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bOffset = 0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        static_cast<uint64_t>(mc2Tiling_.tailM) * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->additionalData.x1Scale =
        transX1ScaleGM1_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * tempScaleKBytes;
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset =
        static_cast<uint64_t>(mc2Tiling_.tailM) * tempScaleKBytes;
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2QuantMmTailTilingData);

    pipeLine_->Process(taskCnt);
}

} // namespace Mc2Kernel

#endif
