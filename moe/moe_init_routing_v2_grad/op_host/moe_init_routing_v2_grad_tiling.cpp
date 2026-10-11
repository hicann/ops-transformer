/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_tiling.cpp
 * \brief
 */
#include "moe_init_routing_v2_grad_tiling.h"
#include "op_host/tiling_util.h"

namespace optiling {
const static int64_t MAX_BINARY_ADD_BUFFER_CNT = 64;
const static int64_t MAX_EXPONENT_OF_BINARY = 6;
const static int64_t BOUND_K_FOR_BINARY = 128;
const static int64_t ELEMENT_ALIGN_SIZE = 32;
const static int64_t MAX_DATACOPY_SIZE = 65536;

class MoeInitRoutingV2GradTilingBase : public MoeInitRoutingV2GradTilingBaseClass {
public:
    explicit MoeInitRoutingV2GradTilingBase(gert::TilingContext* context)
        : MoeInitRoutingV2GradTilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRoutingV2GradTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        MoeInitRoutingV2GradTilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
            return false;
        }
        return true;
    }

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    void TilingBaseInfo();
    void TilingGradCompute();
    void TilingSplitCore();
    void ShowTilingData();
    MoeInitRoutingV2GradTilingData moeInitRoutingV2GradTilingData;
};

void MoeInitRoutingV2GradTilingBase::Reset()
{
    opName = nullptr;
    return;
}

void MoeInitRoutingV2GradTilingBase::ShowTilingData()
{
    OP_LOGI(opName,
            "MoeInitRoutingV2GradTilingData is coreNum:%ld, n:%ld, e:%ld, c:%ld, cols:%ld, k:%ld, activeNum:%ld",
            moeInitRoutingV2GradTilingData.get_coreNum(), moeInitRoutingV2GradTilingData.get_n(),
            moeInitRoutingV2GradTilingData.get_e(), moeInitRoutingV2GradTilingData.get_c(),
            moeInitRoutingV2GradTilingData.get_cols(), moeInitRoutingV2GradTilingData.get_k(),
            moeInitRoutingV2GradTilingData.get_activeNum());
    OP_LOGI(opName,
            "MoeV2GradComputeTilingData is needCoreNum:%ld, perCoreElements:%ld, lastCoreElements:%ld, "
            "elementCopyLoops:%ld, elementPerCopyCols:%ld, elementLastCopyCols:%ld, binaryAddBufferNum:%ld, "
            "tmpBufferNum:%ld, exponentOfBinary:%ld, copyBufferSize:%ld, perCoreTokensLoop:%ld, "
            "perCoreTailTokensFormer:%ld, lastCoreTokensLoop:%ld, lastCoreTailTokensFormer:%ld",
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_needCoreNum(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_perCoreElements(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_lastCoreElements(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_elementCopyLoops(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_elementPerCopyCols(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_elementLastCopyCols(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_binaryAddBufferNum(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_tmpBufferNum(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_exponentOfBinary(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_copyBufferSize(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_perCoreTokensLoop(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_perCoreTailTokensFormer(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_lastCoreTokensLoop(),
            moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp.get_lastCoreTailTokensFormer());
}

ge::graphStatus MoeInitRoutingV2GradTilingBase::DoOpTiling()
{
    TilingBaseInfo();
    TilingSplitCore();
    TilingGradCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingV2GradTilingBase::GetTilingKey() const
{
    uint64_t tilingKey = tilingKey_;
    if (dropPadMode == 1) {
        tilingKey += (dropPadMode * 100); // 100: 01 00
    } else if (activeNum > 0) {
        tilingKey += 10; // 10: for activate
    }

    if (inDtype == ge::DT_FLOAT16) {
        tilingKey += 1; // 1: for FP16
    } else if (inDtype == ge::DT_BF16) {
        tilingKey += 2; // 2: for BF16
    }

    return tilingKey;
}

ge::graphStatus MoeInitRoutingV2GradTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto tilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    moeInitRoutingV2GradTilingData.SaveToBuffer(tilingData->GetData(), tilingData->GetCapacity());
    tilingData->SetDataSize(moeInitRoutingV2GradTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeInitRoutingV2GradTilingBase::TilingBaseInfo()
{
    moeInitRoutingV2GradTilingData.set_coreNum(aivNum);
    // 设置tilingdata参数
    moeInitRoutingV2GradTilingData.set_cols(hiddenSize);
    moeInitRoutingV2GradTilingData.set_n(N);
    moeInitRoutingV2GradTilingData.set_e(E);
    moeInitRoutingV2GradTilingData.set_c(C);
    moeInitRoutingV2GradTilingData.set_k(topK);
    moeInitRoutingV2GradTilingData.set_activeNum(activeNum);
}

void MoeInitRoutingV2GradTilingBase::TilingSplitCore()
{
    tilingKey_ = 1000UL;

    auto tilingData = &moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp;

    int64_t NLocal = moeInitRoutingV2GradTilingData.get_n();
    int64_t perCoreElements = Ops::Base::CeilDiv(NLocal, aivNum);           // 单核处理最大token数
    int64_t needCoreNum = Ops::Base::CeilDiv(NLocal, perCoreElements);      // 实际使用核数
    int64_t lastCoreElement = NLocal - (needCoreNum - 1) * perCoreElements; // 尾核处理token数

    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_lastCoreElements(lastCoreElement);
}

void MoeInitRoutingV2GradTilingBase::TilingGradCompute()
{
    auto tilingData = &moeInitRoutingV2GradTilingData.MoeV2GradComputeParamsOp;
    bool notFloat = (inDtype != ge::DT_FLOAT) ? true : false;
    size_t typeSize = notFloat ? 2U : 4U; // 2: bf16/fp16, 4: float

    // 根据K大小，选择对应的二分累加buffer数量，以一个数exp + 1做2的指数，无限逼近K的大小
    int64_t K = moeInitRoutingV2GradTilingData.get_k();
    int64_t exponentOfBinary =
        (K >= BOUND_K_FOR_BINARY) ? MAX_EXPONENT_OF_BINARY : static_cast<int64_t>(std::floor(std::log2(K)) - 1);
    exponentOfBinary = (exponentOfBinary < 0) ? 0 : exponentOfBinary;
    int64_t binaryAddBuffNum = (K >= BOUND_K_FOR_BINARY) ?
                                   MAX_BINARY_ADD_BUFFER_CNT :
                                   static_cast<int64_t>(std::pow(2, exponentOfBinary)); //  2^exp

    // 根据K大小，选择tmpBuffer数量
    int64_t tmpBuffNum = (K >= BOUND_K_FOR_BINARY) ? binaryAddBuffNum : 1;

    // 计算block块大小
    int64_t coe = static_cast<int64_t>(notFloat) ? 2 : 1; // 2: 如果是bf16/fp16, 需要做cast，所以需要两倍空间
    int64_t H = moeInitRoutingV2GradTilingData.get_cols();
    int64_t hSize = H * static_cast<int64_t>(typeSize);
    int64_t alignHSize = (hSize + ELEMENT_ALIGN_SIZE - 1) / ELEMENT_ALIGN_SIZE * ELEMENT_ALIGN_SIZE;
    int64_t alignHSizeForFloat = alignHSize * coe;
    int64_t maxCopySize = std::min(alignHSizeForFloat, MAX_DATACOPY_SIZE); // 单行一次最大拷贝Size
    int64_t maxBufferSize = aicoreParams_.ubSize / (binaryAddBuffNum + tmpBuffNum);
    maxBufferSize = maxBufferSize / coe / ELEMENT_ALIGN_SIZE * ELEMENT_ALIGN_SIZE * coe;
    maxBufferSize = std::min(maxCopySize, maxBufferSize);

    int64_t maxCopyElements = maxBufferSize / static_cast<int64_t>(sizeof(float));
    int64_t elementCopyLoops = Ops::Base::CeilDiv(H, maxCopyElements);
    int64_t elementLastCopyCols = (H % maxCopyElements == 0) ? maxCopyElements : H % maxCopyElements;

    // 计算单token需要的UB size
    int64_t tokenUbSize = maxBufferSize * (binaryAddBuffNum + tmpBuffNum);

    // 计算并行处理tokens数量
    int64_t tokensFormer = aicoreParams_.ubSize / tokenUbSize;
    tokensFormer = std::min(tokensFormer, tilingData->get_perCoreElements());
    tokensFormer = (tokensFormer <= 0) ? 1 : tokensFormer;
    int64_t perCoreElements = tilingData->get_perCoreElements();
    int64_t lastCoreElement = tilingData->get_lastCoreElements();
    int64_t perCoreTokensLoop = Ops::Base::CeilDiv(perCoreElements, tokensFormer);
    int64_t perCoreTailTokensFormer = perCoreElements % tokensFormer;
    perCoreTailTokensFormer = (perCoreTailTokensFormer == 0) ? tokensFormer : perCoreTailTokensFormer;
    int64_t lastCoreTokensLoop = Ops::Base::CeilDiv(lastCoreElement, tokensFormer);
    int64_t lastCoreTailTokensFormer = lastCoreElement % tokensFormer;
    lastCoreTailTokensFormer = (lastCoreTailTokensFormer == 0) ? tokensFormer : lastCoreTailTokensFormer;

    // 设置tilingData
    tilingData->set_elementCopyLoops(elementCopyLoops);
    tilingData->set_elementPerCopyCols(maxCopyElements);
    tilingData->set_elementLastCopyCols(elementLastCopyCols);
    tilingData->set_binaryAddBufferNum(binaryAddBuffNum);
    tilingData->set_tmpBufferNum(tmpBuffNum);
    tilingData->set_exponentOfBinary(exponentOfBinary);
    tilingData->set_copyBufferSize(maxBufferSize);
    tilingData->set_tokensFormer(tokensFormer);
    tilingData->set_perCoreTokensLoop(perCoreTokensLoop);
    tilingData->set_perCoreTailTokensFormer(perCoreTailTokensFormer);
    tilingData->set_lastCoreTokensLoop(lastCoreTokensLoop);
    tilingData->set_lastCoreTailTokensFormer(lastCoreTailTokensFormer);
}

REGISTER_OPS_TILING_TEMPLATE(MoeInitRoutingV2Grad, MoeInitRoutingV2GradTilingBase, 10000);
} // namespace optiling
