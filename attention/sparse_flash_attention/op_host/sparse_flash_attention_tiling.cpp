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
 * \file sparse_flash_attention_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "../op_kernel/sparse_flash_attention_template_tiling_key.h"
#include "sparse_flash_attention_tiling.h"

using std::map;
using std::pair;
using std::string;

using namespace AscendC;
using namespace ge;
namespace optiling {

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string BLOCK_TABLE_NAME = "block_table";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string QUERY_ROPE_NAME = "query_rope";
static const std::string KEY_ROPE_NAME = "key_rope";
static const std::string ATTEN_OUT_NAME = "attention_out";
static const std::string SOFTMAX_MAX_NAME = "softmax_max";
static const std::string SOFTMAX_SUM_NAME = "softmax_sum";
static const std::string SINKS_NAME = "sinks";

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {VALUE_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {QUERY_ROPE_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_ROPE_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_OUT_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {SOFTMAX_MAX_NAME, {ge::DT_FLOAT}},
    {SOFTMAX_SUM_NAME, {ge::DT_FLOAT}},
    {SINKS_NAME, {ge::DT_FLOAT}},
    {SPARSE_INDICES_NAME, {ge::DT_INT32}},
    {BLOCK_TABLE_NAME, {ge::DT_INT32}},
};

const std::map<std::string, std::vector<SFALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME, {SFALayout::BSND, SFALayout::TND}},
    {KEY_NAME, {SFALayout::BSND, SFALayout::TND, SFALayout::PA_BSND}},
    {VALUE_NAME, {SFALayout::BSND, SFALayout::TND, SFALayout::PA_BSND}},
    {ATTEN_OUT_NAME, {SFALayout::BSND, SFALayout::TND}},
    {SOFTMAX_MAX_NAME, {SFALayout::BNSG, SFALayout::NTG}},
    {SOFTMAX_SUM_NAME, {SFALayout::BNSG, SFALayout::NTG}},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"},                   // dt_variant type
    {ge::DT_INT4, "DT_INT4"}                      // dt_variant type
};

struct SparseFlashAttentionCompileInfo {
    int64_t core_num;
};

static const std::map<SFALayout, std::vector<SFAAxis>> SFA_LAYOUT_AXIS_MAP = {
    {SFALayout::BSND, {SFAAxis::B, SFAAxis::S, SFAAxis::N, SFAAxis::D}},
    {SFALayout::TND, {SFAAxis::T, SFAAxis::N, SFAAxis::D}},
    {SFALayout::PA_BSND, {SFAAxis::Bn, SFAAxis::Bs, SFAAxis::N, SFAAxis::D}},
    {SFALayout::BNSG, {SFAAxis::B, SFAAxis::N, SFAAxis::S, SFAAxis::G}},
    {SFALayout::NTG, {SFAAxis::N, SFAAxis::T, SFAAxis::G}},
};

static const std::map<SFALayout, size_t> SFA_LAYOUT_DIM_MAP = {
    {SFALayout::BSND, DIM_NUM_FOUR}, {SFALayout::TND, DIM_NUM_THREE}, {SFALayout::PA_BSND, DIM_NUM_FOUR},
    {SFALayout::BNSG, DIM_NUM_FOUR}, {SFALayout::NTG, DIM_NUM_THREE},
};

static uint32_t GetTypeSize(ge::DataType dtype)
{
    uint32_t sfaTypeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT16:
            sfaTypeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            sfaTypeSize = NUM_BYTES_BF16;
            break;
        default:
            sfaTypeSize = NUM_BYTES_FLOAT16;
    }
    return sfaTypeSize;
}

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static std::string ToStringRaw(const gert::Shape &shape)
{
    std::ostringstream oss;
    auto v = ToVector(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

template <typename T>
static std::string GetShapeStr(const T &shape)
{
    std::ostringstream sfaOss;
    sfaOss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            sfaOss << shape.GetDim(i) << ", ";
        }
        sfaOss << shape.GetDim(shape.GetDimNum() - 1);
    }
    sfaOss << "]";
    return sfaOss.str();
}

static std::string SFADataTypeToSerialString(ge::DataType type)
{
    const auto sfaIt = DATATYPE_TO_STRING_MAP.find(type);
    if (sfaIt != DATATYPE_TO_STRING_MAP.end()) {
        return sfaIt->second;
    } else {
        OP_LOGE("SparseFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

string SFATensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream sfaOss;
    sfaOss << "(dtype: " << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType()).GetString() << "),";
    sfaOss << "(shape:" << GetShapeStr(shape->GetStorageShape()) << "),";
    sfaOss << "(ori_shape:" << GetShapeStr(shape->GetOriginShape()) << "),";
    sfaOss << "(format: "
           << ge::TypeUtils::FormatToAscendString(
                  static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
                  .GetString()
           << "),";
    sfaOss << "(ori_format: " << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat()).GetString() << ") ";

    return sfaOss.str();
}

string SFADebugTilingContext(const gert::TilingContext *context)
{
    std::ostringstream sfaOss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        sfaOss << "input" << i << ": ";
        sfaOss << SFATensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        sfaOss << "output" << i << ": ";
        sfaOss << SFATensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return sfaOss.str();
}

std::string SFALayoutToSerialString(SFALayout layout)
{
    switch (layout) {
        case SFALayout::BSND:
            return "BSND";
        case SFALayout::TND:
            return "TND";
        case SFALayout::PA_BSND:
            return "PA_BSND";
        case SFALayout::BNSG:
            return "BNSG";
        case SFALayout::NTG:
            return "NTG";
        default:
            return "UNKNOWN";
    }
}

ge::graphStatus SFAMlaTiling::SetBlockDim(uint32_t blockDim) const
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAMlaTiling::SetTilingKey(uint64_t tilingKey) const
{
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1); // 1: batchmode模式
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAMlaTiling::SetWorkspaceSize(uint64_t workspaceSize) const
{
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAMlaTiling::SetTilingData(TilingDef &tilingData) const
{
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
                return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAMlaTiling::GetPlatformInfo()
{
    OP_CHECK_IF(sfaInfo_->platformInfo == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto sfaAscendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    libapiSize_ = sfaAscendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = sfaAscendcPlatform.GetCoreNumAiv();
    aicNum_ = sfaAscendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void SFAMlaTiling::GenTilingKey()
{
    uint32_t inputQType = static_cast<uint32_t>(sfaInfo_->inputQType);
    uint32_t inputKvType = static_cast<uint32_t>(sfaInfo_->inputKvType);
    uint32_t outputType = static_cast<uint32_t>(sfaInfo_->outputType);
    uint32_t layoutQuery = static_cast<uint32_t>(sfaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaInfo_->kvLayout);
    uint32_t pageAttention = 0U;
    if (sfaInfo_->kvLayout == SFALayout::PA_BSND) {
        pageAttention = 1U;
    }

    tilingKey_ = GET_TPL_TILING_KEY(
        0U, pageAttention, layoutQuery, layoutKV, perfMode_ == SFAPerfMode::V_TEMPLATE_MODE,
        static_cast<uint32_t>(sfaInfo_->gSize > 64)); // N1 > 128时核间切G; 64: 触发核间并行切分G维度的性能阈值

    OP_LOGI(sfaInfo_->opName, "SFA tilingKey_: %lu.", tilingKey_);
}

void SFAMlaTiling::ZeroTensorProcess() const
{
    if (sfaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        sfaInfo_->s2Size = 1024;
    }
}

void SFAMlaTiling::InitParams()
{
    if (sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) { // 4:当前支持范围
        perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
    } else {
        perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
    }

    coreNum_ = aicNum_;

    headDimAlign_ = Align(sfaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void SFAMlaTiling::CalcUbBmm()
{
    uint32_t sfaCubeMSize = sfaInfo_->gSize * sfaInfo_->s1Size;
    uint32_t sfaMaxMSize = mBaseSize_;
    if (sfaCubeMSize > sfaMaxMSize) {
        sfaCubeMSize = sfaMaxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(sfaCubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(sfaCubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;
}

void SFAMlaTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void SFAMlaTiling::CalcInnerSize(uint32_t sfaS2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && sfaInfo_->qLayout != SFALayout::TND) {
        if (sfaS2Size == 256) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128;  // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
        } else if (sfaS2Size > 256 && sfaS2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2;                  // 2:减半
        }
    }

    sInnerLoopTimes_ = (sfaS2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = sfaS2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > sfaS2Size) {
        sInnerSize_ = sfaS2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CheckUbSpace();
}

void SFAMlaTiling::SplitBalanced()
{
    CalcInnerSize(sfaInfo_->s2Size);

    InnerSplitParams sfaInnerSplitParams;
    sfaInnerSplitParams.s1GBaseSize = sfaInfo_->gSize;
    sfaInnerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(sfaInnerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(sfaInnerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void SFAMlaTiling::Split()
{
    SplitBalanced();
}

void SFAMlaTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(sfaInfo_->bSize);
    tilingData_.baseParams.set_seqSize(sfaInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(sfaInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(sfaInfo_->blockSize);
    tilingData_.baseParams.set_maxBlockNumPerBatch(sfaInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(sfaInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(sfaInfo_->n1Size / sfaInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(sfaInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(sfaInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(sfaInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(sfaInfo_->sparseMode);
    tilingData_.baseParams.set_preTokens(sfaInfo_->preTokens);
    tilingData_.baseParams.set_nextTokens(sfaInfo_->nextTokens);
    tilingData_.baseParams.set_sparseBlockSize(sfaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(sfaInfo_->sparseBlockCount);
    tilingData_.baseParams.set_attentionMode(sfaInfo_->attentionMode);
    tilingData_.baseParams.set_returnSoftmaxLse(sfaInfo_->returnSoftmaxLse);
    tilingData_.baseParams.set_isActualLenDimsNull(sfaInfo_->actualQSeqLenFlag ? 0U : 1U);
    tilingData_.baseParams.set_isActualLenDimsKVNull(sfaInfo_->actualSeqLenFlag ? 0U : 1U);
    tilingData_.baseParams.set_keyStride0(sfaInfo_->keyStride0);
}

// for flash decode
void SFAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);

    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2U * sfaInfo_->n2Size * mBaseSize_ *
                                               headDimAlign_); // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaInfo_->n2Size *
                                                mBaseSize_ * // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void SFAMlaTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void SFAMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void SFAMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t SFAMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum) const
{
    return actCoreNum * 2 * sfaInfo_->n2Size * mBaseSize_; // 2:每个核可能有头规约和尾规约，一共两份规约信息
}

void SFAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(
            actCoreNum); // balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) : CalcUnbalanceFDParamNums();
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize =
            2 * FDParamNums * (BYTE_BLOCK / sfaInfo_->blockTypeSize); // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += (accumOutSize + logSumExpSize) * sfaInfo_->blockTypeSize;
    }
}

void SFAMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void SFAMlaTiling::GetWorkspaceSize()
{
    uint32_t actCoreNum = coreNum_;
    if (sfaInfo_->isA5) {
        workspaceSize_ = libapiSize_;
        constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
        constexpr uint32_t S2_BASE_SIZE = 128; // S2轴基本块大小
        constexpr uint32_t D_SIZE = 576;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
        uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
        if (sfaInfo_->gSize > 64) { // N1大于64时切G，相邻两个cube核处理同一个s2Base
            aicNum = aicNum >> 1;
        }
        workspaceSize_ += (S2_BASE_SIZE * D_SIZE * GetTypeSize(sfaInfo_->inputQType) * TRIPLE_BUFFER_NUM * aicNum);
    } else {
        uint32_t mmResElemSize = 4;       // 4:fp32
        uint32_t vec1ResElemSize = 2;     // 2:fp16/bf16
        uint32_t bmm2ResElemSize = 4;     // 4:fp32
        uint32_t qPreProcResElemSize = 0; // 普通场景不涉及Q预处理
        uint32_t nUpdateElemSize = 4;     // 4:int32
        uint32_t softmaxSumElemSize = 4;  // 4:int32
        float kvDtypeRatio = 1.0;

        workspaceSize_ = libapiSize_;
        uint32_t preLoadNum = 1;
        preLoadNum = PRE_LOAD_NUM;

        workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
        workspaceSize_ +=
            preLoadNum *
            static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * vec1ResElemSize) * kvDtypeRatio);
        workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
        workspaceSize_ +=
            preLoadNum *
            static_cast<size_t>(static_cast<float>(qPreSizeMla_ * actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * nUpdateElemSize;
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
        // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
        //              bufNum  s2Base   D   dRope  sizeOf(half)
        // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
        workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
        // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
        workspaceSize_ +=
            4 * 128 * 4 * (2 * actCoreNum); // 4:缓存有效mte2 size的长度 128:份数  4:512B对齐的长度  2:aiv核数
    }

    CalcFDWorkSpace(actCoreNum);
}

void SFAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(sfaInfo_->opName, "SFA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus SFAMlaTiling::DoOpTiling(SFATilingInfo *sfaInfo)
{
    sfaInfo_ = sfaInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSparseFlashAttention(gert::TilingContext *context)
{
    SFATilingInfo sfaInfo;
    SFAInfoParser sfaInfoParser(context);
    if (sfaInfoParser.Parse(sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SFATilingCheck tilingChecker(sfaInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SFAMlaTiling tiling(context);
    return tiling.DoOpTiling(&sfaInfo);
}

ge::graphStatus TilingPrepareForSparseFlashAttention(gert::TilingParseContext *const context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::GetExpectedShape(gert::Shape &shapeExpected, const SFATilingShapeCompareParam &param,
                                                 const SFALayout &layout) const
{
    if (layout == SFALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == SFALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == SFALayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else if (layout == SFALayout::BNSG) {
        shapeExpected = gert::Shape({param.B, param.N, param.S, param.G});
    } else if (layout == SFALayout::NTG) {
        shapeExpected = gert::Shape({param.N, param.T, param.G});
    } else {
        OP_LOGE(opName_, "layout %s is unsupported", SFALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CompareShape(SFATilingShapeCompareParam &param, const gert::Shape &shape,
                                             const SFALayout &layout, const std::string &name) const
{
    gert::Shape sfaShapeExpected;
    if (GetExpectedShape(sfaShapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != sfaShapeExpected.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            opName_, name.c_str(), std::to_string(shape.GetDimNum()).c_str(),
            "The shape dim of " + name + " must be " + std::to_string(sfaShapeExpected.GetDimNum()));
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != sfaShapeExpected.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPE(opName_, name.c_str(), ToStringRaw(shape).c_str(),
                                      ToStringRaw(sfaShapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void SFATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                          const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream sfaOss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        sfaOss << SFADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            sfaOss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, name.c_str(), SFADataTypeToSerialString(actualDtype).c_str(),
                                          "The dtype of " + name + " must be " + sfaOss.str());
}

ge::graphStatus SFATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *sfaDesc,
                                                  const std::string &name) const
{
    if (sfaDesc != nullptr) {
        const auto &sfaIt = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(sfaIt == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &sfaExpectDtypeList = sfaIt->second;
        OP_CHECK_IF(std::find(sfaExpectDtypeList.begin(), sfaExpectDtypeList.end(), sfaDesc->GetDataType()) ==
                        sfaExpectDtypeList.end(),
                    LogErrorDtypeSupport(sfaExpectDtypeList, sfaDesc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void SFATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                           const std::string &name, const std::string subName) const
{
    std::ostringstream sfaOss;
    size_t count = expectNumberList.size();
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) {
            if (i == count - 1) {
                sfaOss << " or ";
            } else {
                sfaOss << ", ";
            }
        }
        sfaOss << std::to_string(expectNumberList[i]);
    }

    OP_LOGE_FOR_INVALID_VALUE(opName_, (name + " " + subName).c_str(), std::to_string(actualValue).c_str(),
                              sfaOss.str());
}

template <typename T>
void SFATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                           const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus SFATilingCheck::CheckDimNumInLayoutSupport(const SFALayout &layout, const gert::StorageShape *shape,
                                                           const std::string &name) const
{
    const auto &sfaDimIt = SFA_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != sfaDimIt->second,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, name.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                    "When layout is " + SFALayoutToSerialString(layout) + ", the shape dim of " + name + " should be " +
                        std::to_string(sfaDimIt->second)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
                                                   const std::vector<size_t> &sfaExpectDimNumList,
                                                   const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(sfaExpectDimNumList.begin(), sfaExpectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
        sfaExpectDimNumList.end()) {
        LogErrorDimNumSupport(sfaExpectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void SFATilingCheck::LogErrorLayoutSupport(const std::vector<SFALayout> &expectLayoutList,
                                           const SFALayout &actualLayout, const std::string &name) const
{
    std::ostringstream sfaOssLayout;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        sfaOssLayout << SFALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            sfaOssLayout << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, name.c_str(), SFALayoutToSerialString(actualLayout).c_str(),
                                          "Tensor " + name + " only supports layout " + sfaOssLayout.str());
}

ge::graphStatus SFATilingCheck::CheckLayoutSupport(const SFALayout &actualLayout, const std::string &name) const
{
    const auto &sfaItLayout = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(sfaItLayout == LAYOUT_SUPPORT_MAP.end(),
                OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &sfaExpectLayoutList = sfaItLayout->second;
    OP_CHECK_IF(
        std::find(sfaExpectLayoutList.begin(), sfaExpectLayoutList.end(), actualLayout) == sfaExpectLayoutList.end(),
        LogErrorLayoutSupport(sfaExpectLayoutList, actualLayout, name), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> sfaQueryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, sfaQueryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.query.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> sfaKeyDimNumList = {DIM_NUM_FOUR, DIM_NUM_THREE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.key.shape, sfaKeyDimNumList, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, opParamInfo_.key.shape, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaSparseMode() const
{
    OP_CHECK_IF(
        (*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_mode", std::to_string(*opParamInfo_.sparseMode).c_str(),
                                              "Sparse_mode must be 0 or 3"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaSparseBlockSize() const
{
    OP_CHECK_IF((npuArch_ == NpuArch::DAV_3510 && *opParamInfo_.sparseBlockSize != 1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_block_size",
                                                      std::to_string(*opParamInfo_.sparseBlockSize).c_str(),
                                                      "Sparse_block_size only support 1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((*opParamInfo_.sparseBlockSize <= 0 || *opParamInfo_.sparseBlockSize > 128 ||
                 (static_cast<uint64_t>(*opParamInfo_.sparseBlockSize) &
                  static_cast<uint64_t>(*opParamInfo_.sparseBlockSize - 1L)) != 0UL),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName_, "sparse_block_size", std::to_string(*opParamInfo_.sparseBlockSize).c_str(),
                    "Sparse_block_size must be in range [1, 128] and be a power of 2"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaPreTokens() const
{
    OP_CHECK_IF(
        (*opParamInfo_.preTokens != INT64_MAX),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "pre_tokens", std::to_string(*opParamInfo_.preTokens).c_str(),
                                              "Pre_tokens must be 9223372036854775807"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSingleParaNextTokens() const
{
    OP_CHECK_IF(
        (*opParamInfo_.nextTokens != INT64_MAX),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "next_tokens", std::to_string(*opParamInfo_.nextTokens).c_str(),
                                              "Next_tokens must be 9223372036854775807"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() || ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() || ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() || ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize() || ge::GRAPH_SUCCESS != CheckSingleParaPreTokens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNextTokens()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckRopeExistence()
{
    OP_CHECK_IF((opParamInfo_.queryRope.tensor != nullptr || opParamInfo_.keyRope.tensor != nullptr) &&
                    *opParamInfo_.attentionMode == 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query_rope and key_rope",
                    Ops::Base::ToString(opParamInfo_.queryRope.tensor->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.keyRope.tensor->GetStorageShape()),
                    "In MHA/GQA situation(attentionMode=0), query_rope and key_rope should be empty tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.attentionMode != 2,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "attention_mode",
                                                      std::to_string(*opParamInfo_.attentionMode).c_str(),
                                                      "Attention_mode only supports 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.queryRope.tensor != nullptr && opParamInfo_.keyRope.tensor == nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "query_rope", "Key_rope is an empty tensor, query_rope must also be an empty tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor != nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "key_rope", "Query_rope is an empty tensor, key_rope must also be an empty tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.keyRope.desc == nullptr || opParamInfo_.queryRope.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "query_rope and key_rope", "In Mla situation, desc of query_rope and key_rope should not be null"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, name.c_str(), name + " should not be null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer != nullptr, OP_LOGE(opName_, "%s should be null", name.c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto &kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto &kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
                                                    std::map<std::string, const void *> &notExistMap) const
{
    if (CheckExistsByMap(existMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNotExistsByMap(notExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus SFATilingCheck::CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const
{
    for (auto const &kv : attrMap) {
        const std::string &sfaAttrName = kv.first;
        const std::pair<const T *, T> &sfaPointerValuePair = kv.second;
        if (sfaPointerValuePair.first == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, sfaAttrName.c_str(),
                                                     sfaAttrName + " should not be nullptr");
            return ge::GRAPH_FAILED;
        }

        if (*(sfaPointerValuePair.first) != sfaPointerValuePair.second) {
            std::ostringstream sfaOssExpect;
            sfaOssExpect << std::to_string(sfaPointerValuePair.second);
            std::ostringstream sfaOssActual;
            sfaOssActual << std::to_string(*(sfaPointerValuePair.first));
            OP_LOGE_FOR_INVALID_VALUE(opName_, sfaAttrName.c_str(), sfaOssActual.str().c_str(),
                                      sfaOssExpect.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckParaExistenceMlaNoquant() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    std::map<std::string, const void *> mlaNoquantParamExistMap = {
        {"actualSeqLengths", opParamInfo_.actualSeqLengths.tensor},
        {"blockTable", opParamInfo_.blockTable.tensor},
    };
    std::map<std::string, const void *> mlaNoquantParamNotExistMap = {};
    if (CheckExistenceByMap(mlaNoquantParamExistMap, mlaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckParaExistenceMla() const
{
    return CheckParaExistenceMlaNoquant();
}

ge::graphStatus SFATilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceMla();
}

static ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, const std::string &name,
                                           const char *opName)
{
    if (tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, name.c_str(), name + " must be provided");
        return ge::GRAPH_FAILED;
    }
    int64_t sfaShapeSize = tensor->GetShapeSize();
    if (sfaShapeSize <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, name.c_str(), std::to_string(sfaShapeSize).c_str(),
                                                  "The shape size of " + name + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(sfaShapeSize);
    return ge::GRAPH_SUCCESS;
}

void SFATilingCheck::SetSFAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
    queryRopeShapeCmp_ = opParamInfo_.queryRope.tensor->GetStorageShape();
    keyRopeShapeCmp_ = opParamInfo_.keyRope.tensor->GetStorageShape();
    softmaxMaxShapeCmp_ = opParamInfo_.softmaxMax.shape->GetStorageShape();
    softmaxSumShapeCmp_ = opParamInfo_.softmaxSum.shape->GetStorageShape();
}

ge::graphStatus SFATilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, BLOCK_TABLE_NAME.c_str(),
                        "When the layout_kv is " + SFALayoutToSerialString(kvLayout_) + ", block_table should be null"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.blockTable.desc, BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(
        blockTableBatch != bSize_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, BLOCK_TABLE_NAME.c_str(), ToStringRaw(opParamInfo_.blockTable.tensor->GetStorageShape()).c_str(),
            "The first dim of " + BLOCK_TABLE_NAME + " should be equal to batch size " + std::to_string(bSize_)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype, const ge::DataType &expectDtype,
                                                      const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, name.c_str(), SFADataTypeToSerialString(actualDtype).c_str(),
            "The dtype of " + name + " must be " + SFADataTypeToSerialString(expectDtype));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckQRopeShape()
{
    SFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = ropeHeadDim_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, queryRopeShapeCmp_, qLayout_, QUERY_ROPE_NAME);
}

ge::graphStatus SFATilingCheck::CheckTopkShape()
{
    SFATilingShapeCompareParam sfaShapeParams;
    sfaShapeParams.B = bSize_;
    sfaShapeParams.N = n2Size_;
    sfaShapeParams.S = s1Size_;
    sfaShapeParams.D = sparseBlockCount_;
    sfaShapeParams.T = qTSize_;
    return CompareShape(sfaShapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus SFATilingCheck::CheckAttenOutShape()
{
    SFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSoftmaxMaxShape()
{
    if (*opParamInfo_.returnSoftmaxLse) {
        SFATilingShapeCompareParam shapeParams;
        shapeParams.B = bSize_;
        shapeParams.N = n2Size_;
        shapeParams.S = s1Size_;
        shapeParams.T = qTSize_;
        shapeParams.G = n1Size_ / n2Size_;
        if (CompareShape(shapeParams, softmaxMaxShapeCmp_, softmaxMaxLayout_, SOFTMAX_MAX_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSoftmaxSumShape()
{
    if (*opParamInfo_.returnSoftmaxLse) {
        SFATilingShapeCompareParam shapeParams;
        shapeParams.B = bSize_;
        shapeParams.N = n2Size_;
        shapeParams.S = s1Size_;
        shapeParams.D = vHeadDim_;
        shapeParams.T = qTSize_;
        shapeParams.G = n1Size_ / n2Size_;
        if (CompareShape(shapeParams, softmaxSumShapeCmp_, softmaxSumLayout_, SOFTMAX_SUM_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSinks() const
{
    if (opParamInfo_.sinks.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sinks.desc, SINKS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &sinksShape = opParamInfo_.sinks.tensor->GetStorageShape();
    OP_CHECK_IF(sinksShape.GetDimNum() != DIM_NUM_ONE,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, SINKS_NAME.c_str(),
                                                         std::to_string(sinksShape.GetDimNum()).c_str(),
                                                         "The shape dim of sinks must be 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinksShape.GetShapeSize() != n1Size_,
                OP_LOGE_FOR_INVALID_SHAPE(opName_, SINKS_NAME.c_str(), GetShapeStr(sinksShape).c_str(),
                                          GetShapeStr(gert::Shape({n1Size_})).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS !=
            CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(), inputQType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape() || ge::GRAPH_SUCCESS != CheckSoftmaxMaxShape() ||
        ge::GRAPH_SUCCESS != CheckSoftmaxSumShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SFATilingCheck::CheckSoftmaxMax()
{
    if (*opParamInfo_.returnSoftmaxLse) {
        OP_CHECK_IF(
            opParamInfo_.softmaxMax.shape->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                opName_, "softmax_max", "When return_softmax_lse is true, softmax_max tensor cannot be empty tensor"),
            return ge::GRAPH_FAILED);
        // type类型校验
        OP_CHECK_IF(
            opParamInfo_.softmaxMax.desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "softmax_max", SFADataTypeToSerialString(opParamInfo_.softmaxMax.desc->GetDataType()).c_str(),
                "The dtype of softmax_max must be FLOAT"),
            return ge::GRAPH_FAILED);
        // shape和维度校验
        if (ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (opParamInfo_.softmaxMax.shape->GetStorageShape().GetShapeSize() != 0) {
            OP_LOGW(opName_, "When return_softmax_lse is false, SoftmaxMax tensor must be empty tensor.");
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckSoftmaxSum()
{
    if (*opParamInfo_.returnSoftmaxLse) {
        OP_CHECK_IF(
            opParamInfo_.softmaxSum.shape->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                opName_, "softmax_sum", "When return_softmax_lse is true, softmax_sum tensor cannot be empty tensor"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.softmaxSum.desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "softmax_sum", SFADataTypeToSerialString(opParamInfo_.softmaxSum.desc->GetDataType()).c_str(),
                "The dtype of softmax_sum must be FLOAT"),
            return ge::GRAPH_FAILED);
    } else {
        if (opParamInfo_.softmaxSum.shape->GetStorageShape().GetShapeSize() != 0) {
            OP_LOGW(opName_, "When return_softmax_lse is false, softmax_sum tensor must be empty tensor.");
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckQRope()
{
    if (ge::GRAPH_SUCCESS !=
            CheckDTypeConsistency(opParamInfo_.queryRope.desc->GetDataType(), inputQType_, QUERY_ROPE_NAME) ||
        ge::GRAPH_SUCCESS != CheckQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckVAndKRopeShapeForBatchContinuous()
{
    SFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.T = kvTSize_;
    shapeParams.D = qkHeadDim_;
    if (CompareShape(shapeParams, keyShapeCmp_, kvLayout_, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_ROPE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckVAndKRopeShapeForPageAttention()
{
    int64_t blockNum = keyShapeCmp_.GetDim(0);
    OP_CHECK_IF(blockNum <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "key", ToStringRaw(keyShapeCmp_).c_str(),
                                                      "The first dim of key should be greater than 0"),
                return ge::GRAPH_FAILED);
    SFATilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = blockSize_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_ROPE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckVAndKRopeShape()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckVAndKRopeShapeForBatchContinuous();
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        return CheckVAndKRopeShapeForPageAttention();
    }

    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key",
                                             "Storage mode of key and value is " +
                                                 std::to_string(static_cast<int32_t>(kvStorageMode_)) +
                                                 ", it is incorrect");
    return ge::GRAPH_FAILED;
}

ge::graphStatus SFATilingCheck::CheckVAndKRope()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(), inputKvType_, VALUE_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDTypeConsistency(opParamInfo_.keyRope.desc->GetDataType(), inputKvType_, KEY_ROPE_NAME) ||
        ge::GRAPH_SUCCESS != CheckVAndKRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() || ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "actual_seq_lengths_query",
            "Actual_seq_lengths_query is not empty, but the desc ofactual_seq_lengths_query is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "actual_seq_lengths_query",
            SFADataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str(),
            "The dtype of actual_seq_lengths_query must be DT_INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t sfaShapeSize = 0;
    if (GetActualSeqLenSize(sfaShapeSize, opParamInfo_.actualSeqLengthsQ.tensor, "actual_seq_lengths_query", opName_) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (sfaShapeSize != bSize_) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, "actual_seq_lengths_query", std::to_string(sfaShapeSize),
            "The shape size of actual_seq_lengths_query should be equal to batch size " + std::to_string(bSize_));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLens()
{
    if (std::string(opParamInfo_.layoutKV) == "TND" && opParamInfo_.actualSeqLengths.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "actual_seq_lengths_kv",
                                                 "When the layout_kv is TND, "
                                                 "the actual_seq_lengths_kv shoule not be empty");
        return ge::GRAPH_PARAM_INVALID;
    }
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() || ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "actual_seq_lengths_kv",
            "Actual_seq_lengths_kv is not empty, but the dtype of actual_seq_lengths_kv is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "actual_seq_lengths_kv",
            SFADataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str(),
            "The dtype of actual_seq_lengths_kv must be DT_INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t sfaShapeSizeKv = 0;
    if (GetActualSeqLenSize(sfaShapeSizeKv, opParamInfo_.actualSeqLengths.tensor, "actual_seq_lengths_kv", opName_) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (sfaShapeSizeKv != bSize_) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, "actual_seq_lengths_kv", std::to_string(sfaShapeSizeKv),
            "The shape size of actual_seq_lengths_kv should be equal to batch size " + std::to_string(bSize_));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckMultiParaConsistency()
{
    SetSFAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckVAndKRope() || ge::GRAPH_SUCCESS != CheckQRope() ||
        ge::GRAPH_SUCCESS != CheckTopK() || ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckSoftmaxMax() || ge::GRAPH_SUCCESS != CheckSoftmaxSum() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() || ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoQuantShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "query", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                    "Batch_size of query should be greater than 0, but got " + std::to_string(bSize_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == SFALayout::TND),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "query", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                    "T_size of query should be greater than 0, but got " + std::to_string(qTSize_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "query", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                    "The head num of query should be greater than 0, but got " + std::to_string(n1Size_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                    "The head num of key should be 1, but got " + std::to_string(n2Size_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and key",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                    "The head num of query(" + std::to_string(n1Size_) + ") must be divisible by the head num of key(" +
                        std::to_string(n2Size_) + ")"),
                return ge::GRAPH_FAILED);

    if (isA5_) {
        OP_CHECK_IF(gSize_ < 1 || gSize_ > 128,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query and key",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                        "Group num should be in range [1, 128], but got " + std::to_string(gSize_)),
                    return ge::GRAPH_FAILED);
    } else {
        std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
        OP_CHECK_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query and key",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                        "Group num should be in [1, 2, 4, 8, 16, 32, 64, 128], but got " + std::to_string(gSize_)),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(qkHeadDim_ != 512,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and key",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                    "The head num of query and key only support 512, but got " + std::to_string(qkHeadDim_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qkHeadDim_ != vHeadDim_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query, key and value",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                        Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.value.shape->GetStorageShape()),
                    "The head num of query and key[" + std::to_string(qkHeadDim_) +
                        "] should be equal to the head num of value[" + std::to_string(vHeadDim_) + "]"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ropeHeadDim_ != 64,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "query_rope", ToStringRaw(opParamInfo_.queryRope.tensor->GetStorageShape()).c_str(),
                    "The head num of query_rope should be 64, but got " + std::to_string(ropeHeadDim_)),
                return ge::GRAPH_FAILED);

    if (isA5_) {
        OP_CHECK_IF(s1Size_ <= 0 && (qLayout_ == SFALayout::BSND),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "query", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                        "BSND case query dim 1 should be greater than 0, but got " + std::to_string(s1Size_)),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(s2Size_ <= 0 && (kvLayout_ == SFALayout::BSND),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                        "BSND case key dim 1 should be greater than 0, but got " + std::to_string(s2Size_)),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(kvTSize_ <= 0 && (kvLayout_ == SFALayout::TND),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                        "T_size of key should be greater than 0, but got " + std::to_string(kvTSize_)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoQuantLayout() const
{
    const std::vector<std::string> sfaLayoutSupportList = {"BSND", "TND"};
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OP_CHECK_IF(
        std::find(sfaLayoutSupportList.begin(), sfaLayoutSupportList.end(), layoutQuery) == sfaLayoutSupportList.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_query", layoutQuery.c_str(),
                                              "Layout_query only supports BSND or TND"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoQuantDtype() const
{
    OP_CHECK_IF(
        inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "query", SFADataTypeToSerialString(inputQType_).c_str(),
                                              "The dtype of query must be " + SFADataTypeToSerialString(ge::DT_BF16) +
                                                  " and " + SFADataTypeToSerialString(ge::DT_FLOAT16)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoquantPa() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                    "When page attention is enabled, block_size(" + std::to_string(blockSize_) +
                        ") should be in range (0, " + std::to_string(MAX_BLOCK_SIZE) + "]"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        blockSize_ % 16 > 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
            "When page attention is enabled, block_size(" + std::to_string(blockSize_) + ") should be 16-aligned"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ % sparseBlockSize_ > 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                    "When page attention is enabled, block_size(" + std::to_string(blockSize_) +
                        ") must be divided by sparse_block_size(" + std::to_string(sparseBlockSize_) +
                        "), but now the remainder is " + std::to_string(blockSize_ % sparseBlockSize_)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMlaNoquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape() || ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantDtype() || ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantPa() ||
        ge::GRAPH_SUCCESS != CheckSinks()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFATilingCheck::CheckFeatureMla() const
{
    return CheckFeatureMlaNoquant();
}

ge::graphStatus SFATilingCheck::CheckFeature() const
{
    return CheckFeatureMla();
}

void SFATilingCheck::Init()
{
    opName_ = sfaInfo_.opName;
    platformInfo_ = sfaInfo_.platformInfo;
    opParamInfo_ = sfaInfo_.opParamInfo;
    npuArch_ = sfaInfo_.npuArch;
    isA5_ = sfaInfo_.isA5;

    bSize_ = sfaInfo_.bSize;
    n1Size_ = sfaInfo_.n1Size;
    n2Size_ = sfaInfo_.n2Size;
    s1Size_ = sfaInfo_.s1Size;
    s2Size_ = sfaInfo_.s2Size;
    gSize_ = sfaInfo_.gSize;
    qkHeadDim_ = sfaInfo_.qkHeadDim;
    vHeadDim_ = sfaInfo_.vHeadDim;
    ropeHeadDim_ = sfaInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = sfaInfo_.maxBlockNumPerBatch;
    qTSize_ = sfaInfo_.qTSize;
    kvTSize_ = sfaInfo_.kvTSize;
    blockSize_ = sfaInfo_.blockSize;
    sparseBlockCount_ = sfaInfo_.sparseBlockCount;
    sparseBlockSize_ = sfaInfo_.sparseBlockSize;

    inputQType_ = sfaInfo_.inputQType;
    inputKvType_ = sfaInfo_.inputKvType;
    inputQRopeType_ = sfaInfo_.inputQRopeType;
    inputKRopeType_ = sfaInfo_.inputKRopeType;
    outputType_ = sfaInfo_.outputType;

    qLayout_ = sfaInfo_.qLayout;
    topkLayout_ = sfaInfo_.topkLayout;
    kvLayout_ = sfaInfo_.kvLayout;
    outLayout_ = sfaInfo_.outLayout;
    softmaxMaxLayout_ = sfaInfo_.softmaxMaxLayout;
    softmaxSumLayout_ = sfaInfo_.softmaxSumLayout;

    kvStorageMode_ = sfaInfo_.kvStorageMode;
    l2CacheSize_ = sfaInfo_.l2CacheSize;
}

ge::graphStatus SFATilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS || CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS || CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static constexpr int64_t kInvalidDimValue = std::numeric_limits<int64_t>::min();

static bool HasAxis(const SFAAxis &axis, const SFALayout &layout, const gert::Shape &shape)
{
    const auto &sfaLayoutIt = SFA_LAYOUT_AXIS_MAP.find(layout);
    if (sfaLayoutIt == SFA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<SFAAxis> &sfaAxes = sfaLayoutIt->second;
    const auto &sfaAxisIt = std::find(sfaAxes.begin(), sfaAxes.end(), axis);
    if (sfaAxisIt == sfaAxes.end()) {
        return false;
    }
    const auto &sfaDimIt = SFA_LAYOUT_DIM_MAP.find(layout);
    if (sfaDimIt == SFA_LAYOUT_DIM_MAP.end() || sfaDimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

static size_t GetAxisIdx(const SFAAxis &axis, const SFALayout &layout)
{
    const std::vector<SFAAxis> &axes = SFA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto &axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

static uint32_t GetAxisNum(const gert::Shape &shape, const SFAAxis &axis, const SFALayout &layout)
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : kInvalidDimValue;
}

ge::graphStatus SFAInfoParser::CheckTensorShapes() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "The shape of query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key", "The shape of key is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "value", "The shape of value is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.sparseIndices.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The shape of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.attenOut.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "attention_out", "The shape of attention_out is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxMax.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "softmax_max", "The shape of softmax_max is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxSum.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "softmax_sum", "The shape of softmax_sum is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.queryRope.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_rope", "The shape of query_rope is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::CheckTensorDescriptions() const
{
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "The desc of query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key", "The desc of key is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "value", "The desc of value is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.sparseIndices.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The desc of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.attenOut.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "attention_out", "The desc of attention_out is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxMax.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "softmax_max", "The desc of softmax_max is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxSum.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "softmax_sum", "The desc of softmax_sum is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.queryRope.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_rope", "The desc of query_rope is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::CheckRequiredInOutExistence() const
{
    ge::graphStatus status = CheckTensorShapes();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = CheckTensorDescriptions();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQuery == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_query", "Layout_query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_kv", "Layout_kv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseBlockSize == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_block_size", "Sparse_block_size is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_mode", "Sparse_mode is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleValue == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "scale_value", "Scale_value is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, "actual_seq_lengths_query", opName_);
}

ge::graphStatus SFAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("SparseFlashAttention", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto sfaAscendcPlat = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t sfaAivNum = sfaAscendcPlat.GetCoreNumAiv();
    uint32_t sfaAicNum = sfaAscendcPlat.GetCoreNumAic();
    OP_CHECK_IF(sfaAicNum == 0 || sfaAivNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."),
                return GRAPH_FAILED);

    npuArch_ = sfaAscendcPlat.GetCurNpuArch();
    isA5_ = (npuArch_ == NpuArch::DAV_3510);
    if (npuArch_ != NpuArch::DAV_2201 && npuArch_ != NpuArch::DAV_3510) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "Npu Arch Version[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    sfaAscendcPlat.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void SFAInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.queryRope.tensor = context_->GetOptionalInputTensor(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.queryRope.desc = context_->GetOptionalInputDesc(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetOptionalInputTensor(KEY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.desc = context_->GetOptionalInputDesc(KEY_ROPE_INPUT_INDEX);
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INPUT_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INPUT_INDEX);
}

void SFAInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
    opParamInfo_.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
    GetOptionalInputParaInfo();
}

void SFAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
    opParamInfo_.softmaxMax.desc = context_->GetOutputDesc(SOFTMAXMAX_INDEX);
    opParamInfo_.softmaxMax.shape = context_->GetOutputShape(SOFTMAXMAX_INDEX);
    opParamInfo_.softmaxSum.desc = context_->GetOutputDesc(SOFTMAXSUM_INDEX);
    opParamInfo_.softmaxSum.shape = context_->GetOutputShape(SOFTMAXSUM_INDEX);
}

ge::graphStatus SFAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from GE is nullptr"),
                return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(PRE_TOKENS_ATTR_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_ATTR_INDEX);
    opParamInfo_.attentionMode = attrs->GetAttrPointer<int64_t>(ATTENTION_MODE_ATTR_INDEX);
    opParamInfo_.returnSoftmaxLse = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);

    auto keyStrides = context_->GetDynamicInputStride(KEY_INPUT_INDEX, 0);
    if (keyStrides != nullptr && keyStrides->GetDimNum() > 1) {
        keyStride0_ = keyStrides->GetStride(0);
        keyStride1_ = keyStrides->GetStride(1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    if (opParamInfo_.queryRope.desc != nullptr) {
        inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
    }
    if (opParamInfo_.keyRope.desc != nullptr) {
        inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == SFALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, SFAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus SFAInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == SFALayout::TND) ? GetAxisNum(queryShape_, SFAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetKVTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以key的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    kvTSize_ = (kvLayout_ == SFALayout::TND) ? GetAxisNum(keyShape_, SFAAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    qkHeadDim_ = GetAxisNum(queryShape_, SFAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == SFALayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, SFAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, SFAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetKvStorageMode()
{
    if (kvLayout_ == SFALayout::PA_BSND) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
    // kv存储模式基准值
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetKvLayout()
{
    const map<string, SFALayout> layoutKVMap = {
        {"BSND", SFALayout::BSND}, {"PA_BSND", SFALayout::PA_BSND}, {"TND", SFALayout::TND}};

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_kv", layout.c_str(), "BSND, PA_BSND or TND");
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != SFALayout::PA_BSND && qLayout_ != kvLayout_) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "layout_kv and layout_query",
            SFALayoutToSerialString(kvLayout_) + " and " + SFALayoutToSerialString(qLayout_),
            "When layoutKV is not PA_BSND, layout_kv and layout_query must be same");
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == SFALayout::PA_BSND && keyDimNum != 4U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "key", std::to_string(keyDimNum).c_str(),
                                                 "When layout_kv is PA_BSND, the shape dim of key must be 4");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        *opParamInfo_.returnSoftmaxLse && kvLayout_ == SFALayout::PA_BSND,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_key", SFALayoutToSerialString(kvLayout_).c_str(),
                                              "When return_softmax_lse is true, layout_kv does not support PA_BSND"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetS2SizeForBatchContinuous()
{
    if (kvLayout_ == SFALayout::BSND) { // BSND
        s2Size_ = GetAxisNum(keyShape_, SFAAxis::S, kvLayout_);
    } else if (kvLayout_ == SFALayout::TND) {
        s2Size_ = GetAxisNum(keyShape_, SFAAxis::T, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "block_table",
            "The layout_kv is " + SFALayoutToSerialString(kvLayout_) + ", block_table must be provided");
        return ge::GRAPH_FAILED;
    }
    uint32_t sfaDimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (sfaDimNum != DIM_NUM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "block_table", std::to_string(sfaDimNum),
                                                 "The shape dim of block_table must be " + std::to_string(DIM_NUM_TWO));
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(DIM_IDX_ONE) <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "block_table",
                                              ToStringRaw(opParamInfo_.blockTable.tensor->GetStorageShape()).c_str(),
                                              "The second dim of block_table should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(DIM_IDX_ONE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetBlockSize()
{
    blockSize_ = GetAxisNum(keyShape_, SFAAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, SFAAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus SFAInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    vHeadDim_ = GetAxisNum(valueShape_, SFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetRopeHeadDim()
{
    if (queryShape_.GetDimNum() != queryRopeShape_.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName_, "query and query_rope",
            std::to_string(queryShape_.GetDimNum()) + " and " + std::to_string(queryRopeShape_.GetDimNum()),
            "The shape dim of query and query_rope should be equal");
        return ge::GRAPH_PARAM_INVALID;
    }
    ropeHeadDim_ = GetAxisNum(queryRopeShape_, SFAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const std::map<std::string, std::pair<SFALayout, SFALayout>> sfaLayoutMap = {
        {"BSND", {SFALayout::BSND, SFALayout::BSND}},
        {"TND", {SFALayout::TND, SFALayout::TND}},
    };

    std::string sfaLayout(opParamInfo_.layoutQuery);
    auto sfaLayoutIt = sfaLayoutMap.find(sfaLayout);
    if (sfaLayoutIt != sfaLayoutMap.end()) {
        qLayout_ = sfaLayoutIt->second.first;
        outLayout_ = sfaLayoutIt->second.second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_query", sfaLayout.c_str(), "BSND or TND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetSoftmaxMaxAndSumLayout()
{
    if (qLayout_ == SFALayout::BSND) {
        softmaxMaxLayout_ = SFALayout::BNSG;
        softmaxSumLayout_ = SFALayout::BNSG;
    } else if (qLayout_ == SFALayout::TND) {
        softmaxMaxLayout_ = SFALayout::NTG;
        softmaxSumLayout_ = SFALayout::NTG;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, SFAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, SFAAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void SFAInfoParser::SetSFAShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();
    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    queryRopeShape_ = opParamInfo_.queryRope.tensor->GetStorageShape();
}

ge::graphStatus SFAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDimsKV_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
    }
    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    }
    return ge::GRAPH_SUCCESS;
}

// 非连续校验：通过shape计算expected stride进行校验
// PA_BSND时，只允许0轴非连续，其余轴必须连续
// 非PA_BSND时，所有轴都必须连续
ge::graphStatus SFAInfoParser::CheckContiguous() const
{
    if (!isA5_) {
        return ge::GRAPH_SUCCESS;
    }

    bool keyNonContiguous = false;
    if (keyStride0_ != 0 && keyShape_.GetDimNum() > 0) {
        uint64_t totalElements = 0;
        uint64_t shapeSize = 0;

        if (kvLayout_ == SFALayout::PA_BSND) {
            totalElements = keyStride1_ * static_cast<uint64_t>(keyShape_.GetDim(DIM_IDX_ONE));
            shapeSize = static_cast<uint64_t>(keyShape_.GetDim(DIM_IDX_ONE)) *
                        static_cast<uint64_t>(keyShape_.GetDim(DIM_IDX_TWO)) *
                        static_cast<uint64_t>(keyShape_.GetDim(DIM_IDX_THREE));
        } else {
            totalElements = keyStride0_ * static_cast<uint64_t>(keyShape_.GetDim(0));
            shapeSize = static_cast<uint64_t>(context_->GetOptionalInputTensor(KEY_INPUT_INDEX)->GetShapeSize());
        }
        keyNonContiguous = (shapeSize != totalElements);
    }

    OP_CHECK_IF(keyNonContiguous,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "key", "Key only supports non-contiguous tensor on the 0-axis in PA scenarios"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void SFAInfoParser::GenerateInfo(SFATilingInfo &sfaInfo)
{
    sfaInfo.opName = opName_;
    sfaInfo.platformInfo = platformInfo_;
    sfaInfo.opParamInfo = opParamInfo_;
    sfaInfo.npuArch = npuArch_;
    sfaInfo.isA5 = isA5_;

    sfaInfo.bSize = bSize_;
    sfaInfo.n1Size = n1Size_;
    sfaInfo.n2Size = n2Size_;
    sfaInfo.s1Size = s1Size_;
    sfaInfo.s2Size = s2Size_;
    sfaInfo.gSize = gSize_;
    sfaInfo.qkHeadDim = qkHeadDim_;
    sfaInfo.vHeadDim = vHeadDim_;
    sfaInfo.ropeHeadDim = ropeHeadDim_;
    sfaInfo.qTSize = qTSize_;
    sfaInfo.kvTSize = kvTSize_;
    sfaInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    sfaInfo.sparseBlockCount = sparseBlockCount_;

    sfaInfo.inputQType = inputQType_;
    sfaInfo.inputKvType = inputKvType_;
    sfaInfo.inputQRopeType = inputQRopeType_;
    sfaInfo.inputKRopeType = inputKRopeType_;
    sfaInfo.outputType = outputType_;

    sfaInfo.kvStorageMode = kvStorageMode_;
    sfaInfo.l2CacheSize = l2CacheSize_;

    sfaInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    sfaInfo.scaleValue = *opParamInfo_.scaleValue;
    sfaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    sfaInfo.blockSize = blockSize_;
    sfaInfo.blockTypeSize = sizeof(float);
    sfaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    if (npuArch_ != NpuArch::DAV_3510) {
        sfaInfo.keyStride0 = 0; // 910无需使用stride
    } else {
        if ((keyStride0_ != 0) && (n2Size_ != 0) && (qkHeadDim_ != 0)) {
            sfaInfo.keyStride0 = keyStride0_ / (n2Size_ * qkHeadDim_);
        } else if (kvLayout_ == SFALayout::PA_BSND) {
            sfaInfo.keyStride0 = blockSize_;
        } else {
            sfaInfo.keyStride0 = 0; // 非PA无需使用stride
        }
    }

    FillTilingInfoAttrsAndLayouts(sfaInfo);
}

void SFAInfoParser::FillTilingInfoAttrsAndLayouts(SFATilingInfo &sfaInfo)
{
    sfaInfo.actualLenDimsQ = actualLenDimsQ_;
    sfaInfo.actualLenDimsKV = actualLenDimsKV_;
    sfaInfo.maxActualseq = maxActualseq_;
    sfaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    sfaInfo.isSameActualseq = isSameActualseq_;

    sfaInfo.actualQSeqLenFlag = (opParamInfo_.actualSeqLengthsQ.tensor != nullptr);
    sfaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);

    sfaInfo.sparseMode = *opParamInfo_.sparseMode;
    sfaInfo.preTokens = *opParamInfo_.preTokens;
    sfaInfo.nextTokens = *opParamInfo_.nextTokens;
    sfaInfo.attentionMode = *opParamInfo_.attentionMode;
    sfaInfo.returnSoftmaxLse = *opParamInfo_.returnSoftmaxLse;

    sfaInfo.qLayout = qLayout_;
    sfaInfo.topkLayout = topkLayout_;
    sfaInfo.kvLayout = kvLayout_;
    sfaInfo.outLayout = outLayout_;
    sfaInfo.softmaxMaxLayout = softmaxMaxLayout_;
    sfaInfo.softmaxSumLayout = softmaxSumLayout_;
}

ge::graphStatus SFAInfoParser::Parse(SFATilingInfo &sfaInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("SparseFlashAttention", "SFATilingInfo is nullptr");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() || ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() || ge::GRAPH_SUCCESS != GetSoftmaxMaxAndSumLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() || ge::GRAPH_SUCCESS != GetKvStorageMode()) {
        return ge::GRAPH_FAILED;
    }

    SetSFAShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size() || ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetQTSize() || ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetQkHeadDim() || ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() || ge::GRAPH_SUCCESS != GetRopeHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckContiguous()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(sfaInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseFlashAttention)
    .Tiling(TilingSparseFlashAttention)
    .TilingParse<SparseFlashAttentionCompileInfo>(TilingPrepareForSparseFlashAttention);
} // namespace optiling
