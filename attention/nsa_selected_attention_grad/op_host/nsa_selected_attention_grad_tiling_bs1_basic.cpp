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
 * \file nsa_selected_attention_grad_bs1_basic.cpp
 * \brief
 */

#include "nsa_selected_attention_grad_tiling_bs1_basic.h"
#include "op_host/tiling_templates_registry.h"
#include "err/ops_err.h"
using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace nsa {
constexpr uint32_t WORKSPACE_BASE_CAL = 32 * 1024 * 1024; // 100MB系统预留
constexpr uint32_t BLOCK = 32;                            // 32B
constexpr uint32_t B32 = 4;                               // 4B
constexpr uint32_t B16 = 2;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t PING_PONG_BUFFER = 2;

ge::graphStatus NsaSelectedAttentionGradBasicTiling::GetShapeAttrsInfo()
{
    /*
    Get all shape info and attr
    */
    OP_CHECK_IF(context_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "context is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetAttrs() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetAttrs is nullptr."), return ge::GRAPH_FAILED);

    auto status = GetBaseShapeInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    OP_LOGI(context_, "NsaSelectedAttentionGrad with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] d2[%ld]!",
            tilingData.opInfo.get_B(), tilingData.opInfo.get_N2(), tilingData.opInfo.get_G(),
            tilingData.opInfo.get_S1(), tilingData.opInfo.get_S2(), tilingData.opInfo.get_D(),
            tilingData.opInfo.get_D2());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    uint64_t l2CacheSize;
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<NsaSelectedAttentionGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "compile_info is null."),
                    return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->aivNum;
        aicoreParams_.aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0aSize = compileInfoPtr->l0aSize;
        aicoreParams_.l0bSize = compileInfoPtr->l0bSize;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize =
            compileInfoPtr->l2CacheSize; // AiCoreParams使用的是cann仓的结构体，l2CacheSize暂时定义成类成员变量
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_CHECK_IF(
        (aicoreParams_.numBlocks == 0) || (aicoreParams_.aicNum == 0),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "num of coreNum(aivNum) is %lu, num of aicNum is %lu.",
                                    aicoreParams_.numBlocks, aicoreParams_.aicNum),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(aicoreParams_.ubSize <= 0 || l2CacheSize <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "ubSize or l2CacheSize is invalid."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool NsaSelectedAttentionGradBasicTiling::IsCapable()
{
    if (tmpData.selected_block_count != 16 || tmpData.selected_block_size != 64) {
        return false;
    }

    if (tmpData.d > 192 || tmpData.d2 > 128) {
        return false;
    }

    if (tmpData.g > 16) {
        return false;
    }

    if (context_->GetDeterministic() == 1) {
        return false;
    }

    OP_LOGI(context_, "NsaSelectedAttentionGrad basic template hit.");
    return true;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::DoOpTiling()
{
    OP_LOGI(context_, "NsaSelectedAttentionGrad DoTiling start");

    // Init
    tmpData.singleM = tilingData.opInfo.get_G();
    tmpData.singleN = 512;

    // setTilingData
    tilingData.splitCoreParams.set_singleM(tmpData.singleM);
    tilingData.splitCoreParams.set_singleN(tmpData.singleN);

    auto status = DoSftTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = DoBlockTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = DoCastTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::DoLibApiTiling()
{
    // calc for simpleSoftMax which dstShape is as same as srcShape
    auto simpleSoftMaxShape = ge::Shape({tmpData.singleM, tmpData.singleN});
    auto helpLenA = tmpData.singleM * tmpData.singleN * tmpData.dataTypeSize; // UB内数据类型
    AscendC::SoftMaxTilingFunc(simpleSoftMaxShape, sizeof(float), helpLenA, tilingData.softmaxTilingData);

    // calc for softmaxGrad
    auto softmaxGradShape = ge::Shape({tmpData.singleM, BLOCK / tmpData.dataTypeSize});
    auto helpLenB = 2 * tmpData.singleM * tmpData.singleN * tmpData.dataTypeSize; // UB内数据类型 64KB
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, tmpData.dataTypeSize, helpLenB, tilingData.softmaxGradTilingData,
                                   true);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::GetWorkspaceSize()
{
    int64_t currentUseCoreNum = tilingData.opInfo.get_usedCoreNum();
    int64_t launchBlockDims = aicoreParams_.numBlocks;

    // Tiling传递的内存大小、起始地址，统一为字节数，单位为B
    auto blockdim = CalcTschBlockDim(launchBlockDims, aicoreParams_.aicNum, aicoreParams_.numBlocks);
    OP_CHECK_IF(blockdim == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "blockdim is 0, aicNum is %lu, aivNum is %lu.",
                                            aicoreParams_.aicNum, aicoreParams_.numBlocks),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);
    context_->SetScheduleMode(1);

    // 系统预留
    int64_t sysLen = WORKSPACE_BASE_CAL;
    int64_t mm12WorkspaceLen = tmpData.singleM * tmpData.singleN * B32;
    mm12WorkspaceLen = AlignData(mm12WorkspaceLen, GM_ALIGN) * PING_PONG_BUFFER;

    int64_t dqWorkspaceLen = tilingData.opInfo.get_dqWorkspaceLen();
    int64_t dkWorkspaceLen = tilingData.opInfo.get_dkWorkspaceLen();
    int64_t dvWorkspaceLen = tilingData.opInfo.get_dvWorkspaceLen();

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysLen;
    workspaces[0] += mm12WorkspaceLen * 2 * currentUseCoreNum;
    workspaces[0] += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    tilingData.opInfo.set_mm12WorkspaceLen(mm12WorkspaceLen);

    int64_t workspaceOffsets = mm12WorkspaceLen * 2 * currentUseCoreNum;
    tilingData.postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + tilingData.opInfo.get_dqWorkspaceLen();
    tilingData.postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);
    workspaceOffsets = workspaceOffsets + tilingData.opInfo.get_dkWorkspaceLen();
    tilingData.postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::PostTiling()
{
    // 判断如果GetDataSize > GetCapacity的异常情况,其中确定性计算下终止流程，非确定性计算流入下一个模板判断
    OP_CHECK_IF(
        (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) && (context_->GetDeterministic() == 1),
        OP_LOGE(context_, "The size of TilingDataSize[%zu] is larger than the size of MaxDataCapacity[%zu].",
                tilingData.GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) && (context_->GetDeterministic() != 1),
        OP_LOGE(context_, "The size of TilingDataSize[%zu] is larger than the size of MaxDataCapacity[%zu].",
                tilingData.GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t NsaSelectedAttentionGradBasicTiling::GetTilingKey() const
{
    uint64_t tilingKey = 10;
    if (tmpData.attenEnable) {
        tilingKey += 1;
    }
    OP_LOGI(context_,
            "NsaSelectedAttentionGrad DoTiling success, tilingkey is"
            " %lu.",
            tilingKey);
    return tilingKey;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::DoBlockTiling()
{
    /*
     * 分核策略
     */
    int32_t nNums = tmpData.t1;
    int32_t aicNum = aicoreParams_.aicNum;
    int32_t usedCoreNum = nNums < aicNum ? nNums : aicNum;
    int64_t formerCoreProcessNNums = CeilCommon(nNums, aicNum);
    int64_t remainCoreNum = formerCoreProcessNNums * aicNum - nNums;

    tilingData.opInfo.set_usedCoreNum(usedCoreNum);
    tilingData.opInfo.set_formerCoreNum(aicNum - remainCoreNum);
    tilingData.opInfo.set_formerCoreProcessNNum(formerCoreProcessNNums);
    tilingData.opInfo.set_remainCoreProcessNNum(static_cast<uint32_t>(formerCoreProcessNNums - 1));
    tilingData.opInfo.set_castUsedCoreNum(aicoreParams_.numBlocks);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::DoSftTiling()
{
    /*
     * softmax tiling切分策略
     */
    constexpr int32_t maxProcessDataSize = 8 * 1024;

    uint32_t sftBaseN = tmpData.singleN;
    uint32_t sftBaseM = maxProcessDataSize / sftBaseN;

    tilingData.splitCoreParams.set_sftBaseM(sftBaseM);
    tilingData.splitCoreParams.set_sftBaseN(sftBaseN);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::DoCastTiling()
{
    int64_t dAlign = (tilingData.opInfo.get_D() + 15) / 16 * 16;
    int64_t d2Align = (tilingData.opInfo.get_D2() + 15) / 16 * 16;
    // query
    int64_t allNumQuery = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_G() *
                          tilingData.opInfo.get_S1() * dAlign;
    // TND时候要按照真实的query的num数计算
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumQuery = tmpData.t1 * tilingData.opInfo.get_N2() * tilingData.opInfo.get_G() * tilingData.opInfo.get_D();
    }

    // key
    int64_t allNumKey = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_S2() * dAlign;
    // TND时候要按照真实的key的num数计算
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumKey = tmpData.t2 * tilingData.opInfo.get_N2() * 1 * tilingData.opInfo.get_D();
    }

    // Value
    int64_t allNumValue = tilingData.opInfo.get_B() * tilingData.opInfo.get_N2() * tilingData.opInfo.get_S2() * d2Align;
    // TND时候要按照真实的value的num数计算
    if (tilingData.opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumValue = tmpData.t2 * tilingData.opInfo.get_N2() * 1 * tilingData.opInfo.get_D2();
    }

    uint32_t typeSize = tmpData.queryType == ge::DT_FLOAT ? B32 : B16;
    uint32_t usedCoreNum = tilingData.opInfo.get_castUsedCoreNum();
    constexpr uint32_t postNzCoexNode = 10;
    constexpr uint32_t blockSize = 32;
    constexpr uint32_t postNzReservedN = 1;

    uint32_t postUbBaseSize = 0;
    uint32_t qPostBaseNum = 0;
    int64_t nzReservedSize = 0;
    int64_t curPostCoexNode = postNzCoexNode;
    nzReservedSize = dAlign / 16 * blockSize * postNzReservedN;                      // 16为一个单元长度
    postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode / // 开DB预留2份nzReservedSize
                     BASE_LEN_256 * BASE_LEN_256;
    qPostBaseNum = postUbBaseSize / typeSize / dAlign * tilingData.opInfo.get_D();

    OP_CHECK_IF(qPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "qPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "castUsedCoreNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = allNumQuery / dAlign * tilingData.opInfo.get_D();
    int64_t qSizeAlign = (qPostBlockTotal + BASE_LEN_256 - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t kPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(kPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "kPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t kPostBlockTotal = allNumKey / dAlign * tilingData.opInfo.get_D();
    int64_t kSizeAlign = (kPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    int64_t kPostTailNum = kPostTailNumTmp == 0 ? kPostBaseNum : kPostTailNumTmp;
    int64_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - 1) / kPostBaseNum;
    int64_t kPostBlockFactor = (kPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t vPostBaseNum = postUbBaseSize / typeSize / d2Align * tilingData.opInfo.get_D2();
    OP_CHECK_IF(vPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "vPostBaseNum is 0."),
                return ge::GRAPH_FAILED);
    int64_t vPostBlockTotal = allNumValue / d2Align * tilingData.opInfo.get_D2();
    int64_t vSizeAlign = (vPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t vPostTailNumTmp = vPostBlockTotal % vPostBaseNum;
    int64_t vPostTailNum = vPostTailNumTmp == 0 ? vPostBaseNum : vPostTailNumTmp;
    int64_t vPostBlockOuterTotal = (vPostBlockTotal + vPostBaseNum - 1) / vPostBaseNum;
    int64_t vPostBlockFactor = (vPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    tilingData.postTilingData.set_coreNum(usedCoreNum);
    tilingData.postTilingData.set_scaleValue(tilingData.opInfo.get_scaleValue());
    tilingData.postTilingData.set_postUbBaseSize(postUbBaseSize);
    tilingData.postTilingData.set_nzReservedSize(nzReservedSize);

    tilingData.postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    tilingData.postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    tilingData.postTilingData.set_qPostBaseNum(qPostBaseNum);
    tilingData.postTilingData.set_qPostTailNum(qPostTailNum);
    tilingData.postTilingData.set_qSizeAlign(qSizeAlign);

    tilingData.postTilingData.set_kPostBlockFactor(kPostBlockFactor);
    tilingData.postTilingData.set_kPostBlockTotal(kPostBlockTotal);
    tilingData.postTilingData.set_kPostBaseNum(kPostBaseNum);
    tilingData.postTilingData.set_kPostTailNum(kPostTailNum);
    tilingData.postTilingData.set_kSizeAlign(kSizeAlign);

    tilingData.postTilingData.set_vPostBlockFactor(vPostBlockFactor);
    tilingData.postTilingData.set_vPostBlockTotal(vPostBlockTotal);
    tilingData.postTilingData.set_vPostBaseNum(vPostBaseNum);
    tilingData.postTilingData.set_vPostTailNum(vPostTailNum);
    tilingData.postTilingData.set_vSizeAlign(vSizeAlign);

    tilingData.opInfo.set_dqWorkspaceLen((allNumQuery * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    tilingData.opInfo.set_dkWorkspaceLen((allNumKey * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    tilingData.opInfo.set_dvWorkspaceLen((allNumValue * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);

    tilingData.postTilingData.set_b(tilingData.opInfo.get_B());
    tilingData.postTilingData.set_n2(tilingData.opInfo.get_N2());
    tilingData.postTilingData.set_g(tilingData.opInfo.get_G());
    tilingData.postTilingData.set_s1(tilingData.opInfo.get_S1());
    tilingData.postTilingData.set_s2(tilingData.opInfo.get_S2());
    tilingData.postTilingData.set_d(tilingData.opInfo.get_D());
    tilingData.postTilingData.set_d2(tilingData.opInfo.get_D2());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionGradBasicTiling::GetBaseShapeInfo()
{
    OP_CHECK_IF(((context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::KEY)) == nullptr) ||
                 (context_->GetInputShape(static_cast<size_t>(InputIndex::VALUE)) == nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "InputShape of query, key or value is nullptr."),
                return ge::GRAPH_FAILED);
    // input
    // query [t1, n1, d]   k [t2, n2, d]  v [t2, n2, d2]   dy/attentionIn [t1, n1, d2]
    const gert::Shape& queryShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
    const gert::Shape& keyShape = context_->GetInputShape(static_cast<size_t>(InputIndex::KEY))->GetStorageShape();
    const gert::Shape& valueShape = context_->GetInputShape(static_cast<size_t>(InputIndex::VALUE))->GetStorageShape();
    auto actualSeqQlenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_Q_LEN));
    auto actualSeqKvlenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
    uint32_t dimSize = queryShape.GetDimNum();
    int64_t dimDq = queryShape.GetDim(DIM_2);
    int64_t dimDk = keyShape.GetDim(DIM_2);
    int64_t dimDv = valueShape.GetDim(DIM_2);

    // attrs
    int64_t dimN1 = *context_->GetAttrs()->GetAttrPointer<int>(
        static_cast<size_t>(AttrIndex::HEAD_NUM)); // headNum is as same as N1
    const char* inputLayout = context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    auto selected_block_count =
        *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_COUNT));
    auto selected_block_size =
        *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    auto sparse_mode = *context_->GetAttrs()->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SPARSE_MODE));
    if (sparse_mode == 0) {
        tmpData.attenEnable = false;
    } else if (sparse_mode == 2) {
        OP_LOGI(context_, "NsaSelectedAttentionGrad AttenMask enable.");
        tmpData.attenEnable = true;
    } else {
        OP_LOGE(context_, "NsaSelectedAttentionGrad only support sparse_mode=2, now sparse_mode=%d.", sparse_mode);
        return ge::GRAPH_FAILED;
    }

    if (dimDq != dimDk) {
        OP_LOGE(context_, "head_dim of Query[%ld] should be equal to head_dim of Key[%ld].", dimDq, dimDk);
        return ge::GRAPH_FAILED;
    }
    if (dimDq < dimDv) {
        OP_LOGE(context_, "head_dim of Query[%ld] can not less than head_dim of Value[%ld].", dimDq, dimDv);
        return ge::GRAPH_FAILED;
    }
    if (dimN1 == 0) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad get the headNum of query is 0, now is %ld.", dimN1);
        return ge::GRAPH_FAILED;
    }
    if (strcmp(inputLayout, TND_STR) != 0) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad only support TND layout, now layout is %s.", inputLayout);
        return ge::GRAPH_FAILED;
    }
    if (static_cast<size_t>(dimSize) != strlen(inputLayout)) {
        OP_LOGE(context_,
                "NsaSelectedAttentionGrad layout dims is not equal to the input's dim, now the query of dim is %u.",
                dimSize);
        return ge::GRAPH_FAILED;
    }
    if (actualSeqQlenTensor == nullptr || actualSeqKvlenTensor == nullptr) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad actualSeqQlenTensor or actualSeqKvlenTensor is nullptr.");
        return ge::GRAPH_FAILED;
    }

    const size_t seqQShapeSize = static_cast<size_t>(actualSeqQlenTensor->GetShapeSize());
    const size_t kvSeqShapeSize = static_cast<size_t>(actualSeqKvlenTensor->GetShapeSize());
    const int64_t* qValue = actualSeqQlenTensor->GetData<int64_t>();
    const int64_t* kvValue = actualSeqKvlenTensor->GetData<int64_t>();
    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;

    if (seqQShapeSize != kvSeqShapeSize) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad actualSeqQlenTensor shapeSize is not equal actualSeqKvlenTensor.");
        return ge::GRAPH_FAILED;
    }
    if (qValue == nullptr || kvValue == nullptr) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad qValue or kvValue is nullptr.");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < seqQShapeSize; i++) {
        int64_t qSeqLen = (i == 0 ? qValue[i] : (qValue[i] - qValue[i - 1]));
        int64_t kvSeqLen = (i == 0 ? kvValue[i] : (kvValue[i] - kvValue[i - 1]));
        tmpData.actualSeqQlen.push_back(qSeqLen);
        tmpData.actualSeqKvlen.push_back(kvSeqLen);
    }

    uint64_t tailZeroCount = 0;
    for (auto i = seqQShapeSize - 1; i >= 1; --i) {
        if (tmpData.actualSeqQlen[i] <= 0) {
            ++tailZeroCount;
        } else {
            break;
        }
    }
    auto realBSize = seqQShapeSize - tailZeroCount;
    tilingData.opInfo.set_B(realBSize);
    OP_CHECK_IF(keyShape.GetDim(DIM_1) == 0, OP_LOGE(context_, "dim 1 of key is 0."), return ge::GRAPH_FAILED);
    tilingData.opInfo.set_G(queryShape.GetDim(DIM_1) / keyShape.GetDim(DIM_1));
    OP_CHECK_IF(tilingData.opInfo.get_G() == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
    tilingData.opInfo.set_N2(keyShape.GetDim(DIM_1));
    tilingData.opInfo.set_S1(*std::max_element(tmpData.actualSeqQlen.begin(), tmpData.actualSeqQlen.end()));
    tilingData.opInfo.set_S2(*std::max_element(tmpData.actualSeqKvlen.begin(), tmpData.actualSeqKvlen.end()));
    tilingData.opInfo.set_D(dimDq);
    tilingData.opInfo.set_D2(dimDv);
    tilingData.opInfo.set_layout(static_cast<uint32_t>(InputLayout::TND));
    tilingData.opInfo.set_scaleValue(
        *context_->GetAttrs()->GetAttrPointer<float>(static_cast<size_t>(AttrIndex::SCALE_VALUE)));
    tilingData.opInfo.set_selectedBlockCount(selected_block_count);
    tilingData.opInfo.set_selectedBlockSize(selected_block_size);

    tmpData.t1 = queryShape.GetDim(DIM_0);
    tmpData.t2 = keyShape.GetDim(DIM_0);
    tmpData.b = tilingData.opInfo.get_B();
    tmpData.n2 = tilingData.opInfo.get_N2();
    tmpData.g = tilingData.opInfo.get_G();
    tmpData.d = tilingData.opInfo.get_D();
    tmpData.d2 = tilingData.opInfo.get_D2();
    tmpData.s1 = tilingData.opInfo.get_S1();
    tmpData.s2 = tilingData.opInfo.get_S2();
    tmpData.layout = tilingData.opInfo.get_layout();
    tmpData.dataTypeSize = B32;
    tmpData.queryType =
        static_cast<uint32_t>(context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType());
    tmpData.selected_block_count = selected_block_count;
    tmpData.selected_block_size = selected_block_size;

    auto ret = CheckDtypeValid(context_);

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad the dtype of input is invalid.");
        return ge::GRAPH_FAILED;
    }

    ret = CheckTndShapeValid(context_, tmpData.t1, dimN1, tmpData.d, tmpData.d2);

    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "NsaSelectedAttentionGrad the input shpae of TND Layout is invalid.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(NsaSelectedAttentionGrad, NsaSelectedAttentionGradBasicTiling, 1);

} // namespace nsa
} // namespace optiling
