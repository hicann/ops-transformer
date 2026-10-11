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
 * \file incre_flash_attention_tiling_check.cpp
 * \brief
 */

#include <numeric>
#include "incre_flash_attention_tiling_impl.h"
#include "incre_flash_attention_tiling_base.h"
#include "log/log.h"

namespace optiling {

custom::graphStatus IFATiling::CheckPABlockSize() const
{
    OP_CHECK_IF(blockSize_ == 0,
                OP_LOGE(ifaContext_->opName, "When Page Attention is enabled, input attribute blocksize can not be 0."),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(blockSize_ > MAX_BLOCK_SIZE,
                OP_LOGE(ifaContext_->opName,
                        "When Page Attention is enabled, input attribute blocksize %u can not be larger than %u.",
                        blockSize_, MAX_BLOCK_SIZE),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(((inputKvType_ == at::ScalarType::Char) && (blockSize_ % 32 != 0)),
                OP_LOGE(ifaContext_->opName,
                        "When Page Attention is enabled, if kv cache dtype is int8, input attr "
                        "blocksize[%u] should be 32 aligned.",
                        blockSize_),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(((inputKvType_ == at::ScalarType::Half) || (inputKvType_ == at::ScalarType::BFloat16)) &&
                    (blockSize_ % 16 != 0),
                OP_LOGE(ifaContext_->opName,
                        "When Page Attention is enabled, "
                        "if kv cache dtype is float16/bfloat16, input attr blocksize should be 16 aligned"),
                return custom::graphStatus::GRAPH_FAILED);
    return custom::graphStatus::GRAPH_SUCCESS;
}

bool IFATiling::IsZeroDimTensor(const at::IntArrayRef& shape) const
{
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            return true;
        }
    }
    return false;
}

std::string IFATiling::GetTensorDimString(const at::IntArrayRef& shape) const
{
    std::stringstream ss;
    ss << "[";
    bool isFirst = true;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (!isFirst) {
            ss << ", ";
        }
        ss << std::to_string(shape[i]);
        isFirst = false;
    }
    ss << "]";
    return ss.str();
}

bool IFATiling::CheckIfRollBack() const
{
    if (sMax_ == 0U) {
        return false; // 空tensor由新模板处理
    }

    if (socVersion_ != IfaSocVersion::SOC_ASCEND_310P) {
        // 1 page attention
        if (ifaContext_->blockTable.hasValue) {
            return false;
        }
    }

    // 4 D>=1024
    if (headDim_ >= 1024U) {
        return true;
    }

    if (CanChangeToNew()) {
        return false;
    }

    return true;
}

bool IFATiling::ShapeEqual(const at::IntArrayRef& aShape, const at::IntArrayRef& bShape) const
{
    if (aShape.size() != bShape.size()) {
        return false;
    }

    for (size_t idx = 0; idx < aShape.size(); idx++) {
        if (aShape[idx] != bShape[idx]) {
            return false;
        }
    }

    return true;
}

bool IFATiling::CanChangeToNew() const
{
    if (inOutMode_ == TilingInOutMode::BF16_BF16) {
        return true;
    }
    if (inOutMode_ == TilingInOutMode::BF16_INT8) {
        return true;
    }

    if (inOutMode_ == TilingInOutMode::FP16_FP16 || inOutMode_ == TilingInOutMode::FP16_INT8) {
        return true;
    }

    if (inOutMode_ == TilingInOutMode::INT8_BF16 || inOutMode_ == TilingInOutMode::INT8_FP16) {
        return true;
    }
    return false;
}

custom::graphStatus IFATiling::CheckSupportKVLeftPadding()
{
    OP_CHECK_IF(ropeFlag_, OP_LOGE(ifaContext_->opName, "kvPadding is not supported in MLA."),
                return custom::graphStatus::GRAPH_FAILED);
    if (!batchContinuousFlag_ || !actualSeqLenFlag_ || pageAttentionFlag_) {
        OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition:  \
      pagedAttention scene: %d, not isBatchContinues: %d, actualSeqLen not exist: %d.",
                pageAttentionFlag_, !batchContinuousFlag_, !actualSeqLenFlag_);
        return custom::graphStatus::GRAPH_SUCCESS;
    }
    kvPaddingSizeFlag_ = true;
    OP_LOGD(ifaContext_->opName, "KVLeftPadding starts to be used.");
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::SharedPrefixCheckBasic()
{
    OP_CHECK_IF(!ifaContext_->keySharedPrefix.hasValue,
                OP_LOGE(ifaContext_->opName, "tensor of key_shared_prefix is null."),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(!ifaContext_->keySharedPrefix.hasValue,
                OP_LOGE(ifaContext_->opName, "desc of key_shared_prefix is null."),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(!ifaContext_->valueSharedPrefix.hasValue,
                OP_LOGE(ifaContext_->opName, "tensor of value_shared_prefix is null."),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(!ifaContext_->valueSharedPrefix.hasValue,
                OP_LOGE(ifaContext_->opName, "desc of value_shared_prefix is null."),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.dType != inputKvType_,
                OP_LOGE(ifaContext_->opName, "type of key_shared_prefix not equal to type of KV"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.dType != inputKvType_,
                OP_LOGE(ifaContext_->opName, "type of value_shared_prefix not equal to type of KV"),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(pageAttentionFlag_, OP_LOGE(ifaContext_->opName, "shared prefix with page attention is not supported"),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(kvPaddingSizeFlag_, OP_LOGE(ifaContext_->opName, "shared prefix with kv padding is not supported"),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(socVersion_ == IfaSocVersion::SOC_ASCEND_310P,
                OP_LOGE(ifaContext_->opName, "shared prefix is not supported on 310p"),
                return custom::graphStatus::GRAPH_FAILED);

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::SharedPrefixCheckShapes(const at::IntArrayRef& keyShape,
                                                       const at::IntArrayRef& valueShape) const
{
    OP_CHECK_IF(!ShapeEqual(keyShape, valueShape),
                OP_LOGE(ifaContext_->opName, "tensor shape of key_shared_prefix and value_shared_prefix not equal."),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(keyShape.size() != ifaContext_->query.shape.size(),
                OP_LOGE(ifaContext_->opName, "tensor shape dim of key_shared_prefix[%lu] is not compatable with query",
                        keyShape.size()),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(keyShape[0] != 1,
                OP_LOGE(ifaContext_->opName, "batch of key_shared_prefix[%ld] must be 1", keyShape[0]),
                return custom::graphStatus::GRAPH_FAILED);

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        OP_CHECK_IF(
            keyShape.size() == 3 && keyShape[2] != numKvHeads_ * headDim_,
            OP_LOGE(ifaContext_->opName, "H of key_shared_prefix[%lu] is not equal to H of key", keyShape.size()),
            return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(
            keyShape.size() == 4 && keyShape[2] != numKvHeads_,
            OP_LOGE(ifaContext_->opName, "N of key_shared_prefix[%lu] is not equal to N of key", keyShape.size()),
            return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(
            keyShape.size() == 4 && keyShape[3] != headDim_,
            OP_LOGE(ifaContext_->opName, "D of key_shared_prefix[%lu] is not equal to D of key", keyShape.size()),
            return custom::graphStatus::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(keyShape[1] != numKvHeads_,
                    OP_LOGE(ifaContext_->opName, "N of key_shared_prefix[%ld] is not equal to N of key", keyShape[1]),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(keyShape[3] != headDim_,
                    OP_LOGE(ifaContext_->opName, "D of key_shared_prefix[%ld] is not equal to D of key", keyShape[3]),
                    return custom::graphStatus::GRAPH_FAILED);
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

bool IFATiling::CheckUbSpace()
{
    if (!CalcUbBmm()
        // || !CalcUbSoftMax()
        || !CalcUbAttenMask()) {
        return false;
    }
    return true;
}

bool IFATiling::IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode)
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode = IfaPerfMode::BMM_ALL_BY_VEC;
        if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
            OP_LOGD(ifaContext_->opName, "flash decode has not been implemented when 310P and GQA > 1.");
            return false;
        }
    }

    if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B &&
        perfMode != IfaPerfMode::CUBE_VIEW_MM_MLA && maxActualseq_ <= 1024U) { // 1024, 经验值
        return false;
    }
    constexpr uint32_t maxQNum = 128U;
    if (ropeFlag_ && (qSeqSize_ > 1U || nNumOfQInOneGroup_ > maxQNum)) {
        return false; // 投机推理 和 G > 64 切G待实现
    }

    if (gqaMtpFlag_) {
        return false; // 投机推理FD待实现
    }

    if (CheckCoreOkFlag(coreNum, perfMode)) {
        return true;
    }

    return false;
}

bool IFATiling::CheckCoreOkFlag(uint32_t coreNum, IfaPerfMode perfMode) const
{
    bool coreOkFlag = false;
    float flashDecodeBNRatio = static_cast<float>(0.4); // 0.4, 经验值
    if (perfMode == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD || perfMode == IfaPerfMode::CUBE_VIEW_MM ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode == IfaPerfMode::CUBE_VIEW_MM_DD) {
        flashDecodeBNRatio = 0.5; // 0.5, 经验值
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) <= flashDecodeBNRatio * coreNum);
    } else if (perfMode == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5; // 0.5, 全V模板可以按0.5切分
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    } else {
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    }

    if (coreOkFlag && (nNumOfQInOneGroup_ == 1U)) {
        OP_LOGD(ifaContext_->opName, "Flash Decode Split kv."); // 非gqa时这里只判断bn是否满足
        return true;
    }

    if (coreOkFlag && (maxActualseq_ >= 2048U)) { // 2048, 在flash decode + gqa时的经验值
        OP_LOGD(ifaContext_->opName, "Flash Decode And GQA Split kv.");
        return true;
    }
    return false;
}

bool IFATiling::EnableCubeViewMM() const
{
    if (!CheckConstraints()) {
        return false;
    }

    if (!CheckDataCopyNd2Nz()) {
        return false;
    }

    uint32_t s2Loop = (maxActualseq_ + 2048U - 1U) / 2048U;
    if (static_cast<uint64_t>(batchSize_) * numKvHeads_ * s2Loop <= 128U) { // 128: 性能收益临界值
        return false;
    }

    std::string layOutStr = ifaContext_->layOut;
    if (pageAttentionFlag_ && layOutStr == "BNSD") {
        bool quantFlag = antiQuantFlag_ && (antiquantMode_ == PER_CHANNEL_MODE) &&
                         (inputQType_ == at::ScalarType::Half || inputQType_ == at::ScalarType::BFloat16) &&
                         (inputKvType_ == at::ScalarType::Char);
        bool noQuantFlag = (inputQType_ == at::ScalarType::Half && inputKvType_ == at::ScalarType::Half) ||
                           (inputQType_ == at::ScalarType::BFloat16 && inputKvType_ == at::ScalarType::BFloat16);
        if (quantFlag || noQuantFlag) {
            return true;
        }
    }

    return false;
}

bool IFATiling::CheckDataCopyNd2Nz() const
{
    if (innerPrecise_ == IFA_HIGH_PRECISION) {
        return false;
    }
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH" || layOutStr == "BSND") {
        // 拷贝KV时, DataCopy ND2NZ指令不支持行Stride大于等于65536
        if (static_cast<uint64_t>(headDim_) * numKvHeads_ >= 65536U) {
            return false;
        }
    }
    return true;
}

bool IFATiling::CheckConstraints() const
{
    if (sysPrefixFlag_) {
        return false;
    }
    if (nNumOfQInOneGroup_ > 16U) { // 16: group size
        return false;
    }
    if (headDim_ != 128U) { // 128: D dim
        return false;
    }
    return true;
}

bool IFATiling::EnableCubeViewMMDD() const
{
    if (!CheckConstraints()) {
        return false;
    }
    if (innerPrecise_ == IFA_HIGH_PRECISION) {
        return false;
    }
    if (softmaxLseFlag_) {
        return false;
    }

    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH" || layOutStr == "BSND") {
        // 拷贝KV时, DataCopy ND2NZ指令不支持行Stride大于等于65536
        if (static_cast<uint64_t>(headDim_) * numKvHeads_ >= 65536U) {
            return false;
        }
    }
    if (layOutStr == "TND" || layOutStr == "TND_NTD") {
        return false;
    }
    if (ifaContext_->antiquantOffset.hasValue || ifaContext_->keyAntiquantOffset.hasValue ||
        ifaContext_->valueAntiquantOffset.hasValue) {
        return false;
    }
    if (gqaKvNZFlag_ && pageAttentionFlag_ && antiQuantFlag_ &&
        (antiquantMode_ == PER_CHANNEL_MODE || antiquantMode_ == PER_TOKEN_MODE) &&
        (inputQType_ == at::ScalarType::BFloat16) && (inputKvType_ == at::ScalarType::Char)) {
        return true;
    }
    return false;
}

bool IFATiling::DealSameSeqEachBatch() const
{
    if (!batchContinuousFlag_) {
        if (actualSeqLenFlag_) {
            return isSameActualseq_;
        } else {
            return isSameSeqAllKVTensor_;
        }
    } else {
        return isSameActualseq_;
    }
}

bool IFATiling::HasZeroSeqBatch() const
{
    if (!batchContinuousFlag_) {
        if (actualSeqLenFlag_) {
            return hasZeroActualseq_;
        } else {
            return hasZeroSeqKVTensor_;
        }
    } else {
        return hasZeroActualseq_;
    }
}

bool IFATiling::IsKvZeroBatchSplit(bool needUpdate, uint32_t lastValidBMoreIdx, uint32_t bSize,
                                   const std::vector<uint32_t>& s1OuterNum,
                                   const std::vector<uint32_t>& s2OuterNum) const
{
    if (needUpdate) {
        return false;
    }
    if ((lastValidBMoreIdx <= 0U) || (lastValidBMoreIdx >= bSize)) {
        return false;
    }
    if (s1OuterNum[lastValidBMoreIdx] == 0) {
        return false;
    }
    if (s2OuterNum[lastValidBMoreIdx] != 0) {
        return false;
    }
    return true;
}

bool IFATiling::EnableGQA310P()
{
    if (IsFlashDecode(aicNum_, IfaPerfMode::BMM_ALL_BY_VEC)) {
        return false;
    }
    if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
        return true;
    }
    if (pageAttentionFlag_ && nNumOfQInOneGroup_ * qSeqSize_ == 1U && (seqSize_ >= 256U) &&
        inputLayout_ == IfaLayout::BSH_BSND) {
        return true;
    }
    if (!pageAttentionFlag_ && nNumOfQInOneGroup_ * qSeqSize_ == 1U && (seqSize_ >= 256U) &&
        inputLayout_ == IfaLayout::BNSD) {
        return true;
    }
    return false;
}

bool IFATiling::EnableCubeViewMMFullLoad()
{
    if (!CheckConstraints()) {
        return false;
    }
    if (IsFlashDecode(aicNum_, IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD)) {
        return false;
    }
    if (softmaxLseFlag_) {
        return false;
    }
    if (!CheckDataCopyNd2Nz()) {
        return false;
    }
    if (!DealSameSeqEachBatch()) {
        return false;
    }
    if (HasZeroSeqBatch()) {
        return false;
    }
    if (maxActualseq_ > 2048U || maxActualseq_ == 0U) {
        return false;
    }

    if (static_cast<uint64_t>(batchSize_) * numKvHeads_ <= 128U) { // 128: 性能收益临界值
        return false;
    }

    if (antiQuantFlag_) {
        return false;
    } else {
        return false;
    }
}

bool IFATiling::EnableAllVec() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return true;
    }
    if (pageAttentionFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    if (nNumOfQInOneGroup_ > 1U) {
        return false;
    }
    if (headDim_ > 512U) { // 全VEC模板仅支持headDim_不大于512
        return false;
    }
    return (inputQType_ == at::ScalarType::Half) && (inputKvType_ == at::ScalarType::Half) &&
           (outputType_ == at::ScalarType::Half);
}

bool IFATiling::EnableC1V1() const
{
    if (splitKVFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    // 2:核数不超过vector总核数一半，可以按1:1启动cube和vector
    return (perfMode_ == IfaPerfMode::NORMAL) && (batchSize_ * numKvHeads_ * 2U <= aivNum_);
}

custom::graphStatus IFATiling::CheckPseShiftDataType() const
{
    auto pseShiftDataType = ifaContext_->pseShift.dType;
    if (pseShiftDataType != at::ScalarType::Half && pseShiftDataType != at::ScalarType::BFloat16) {
        OP_LOGE(ifaContext_->opName, "Data type of pse shift is %s, which is not supported.",
                DataTypeToSerialString(pseShiftDataType).c_str());
        return custom::graphStatus::GRAPH_FAILED;
    }

    switch (pseShiftDataType) {
        case at::ScalarType::Half:
        case at::ScalarType::BFloat16:
            OP_CHECK_IF(
                (inputQType_ != at::ScalarType::Char) && (inputQType_ != pseShiftDataType),
                OP_LOGE(ifaContext_->opName, "Data type of pse is %s, which does not match data type of query: %s.",
                        DataTypeToSerialString(pseShiftDataType).c_str(), DataTypeToSerialString(inputQType_).c_str()),
                return custom::graphStatus::GRAPH_FAILED);
            break;
        default:
            OP_LOGE(ifaContext_->opName, "Data type of pse %s is not currently supported.",
                    DataTypeToSerialString(pseShiftDataType).c_str());
            return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessPseShift()
{
    // get pse shift data
    if (!ifaContext_->pseShift.hasValue) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(gqaKvNZFlag_, OP_LOGE(ifaContext_->opName, "PseShift is not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);

    if (CheckPseShiftDataType() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    // check pse shift shape (B/1, N, 1, Si)
    at::IntArrayRef& pseShiftShape = ifaContext_->pseShift.shape;
    uint32_t pseShiftDimNum = pseShiftShape.size();
    OP_CHECK_IF(pseShiftDimNum != 4,
                OP_LOGE(ifaContext_->opName, "The input shape of pse shift must have 4 dims, current dim num is %u.",
                        pseShiftDimNum),
                return custom::graphStatus::GRAPH_FAILED);
    pseShiftBatch_ = pseShiftShape[PSE_SHIFT_B];
    uint32_t pseShiftN = pseShiftShape[PSE_SHIFT_N];
    uint32_t pseShiftS0 = pseShiftShape[PSE_SHIFT_S0];
    pseShiftS1_ = pseShiftShape[PSE_SHIFT_S1];

    OP_CHECK_IF(
        (pseShiftBatch_ != 1 && pseShiftBatch_ != batchSize_) || (pseShiftN != numHeads_) || (pseShiftS0 != 1),
        OP_LOGE(ifaContext_->opName,
                "The shape of pse shift is (%u, %u, %u, %u), which does not match (B, N, 1, S) or (1, N, 1, S).",
                pseShiftBatch_, pseShiftN, pseShiftS0, pseShiftS1_),
        return custom::graphStatus::GRAPH_FAILED);

    if (pseShiftS1_ < seqSize_) {
        OP_LOGE(ifaContext_->opName,
                "The shape of pse shift is (%u, %u, %u, %u), the 3rd dim S[%u] shouldn't be less than sMax[%u]."
                "When Page Attention is enabled, sMax is maxBlockNumPerBatch * blockSize.",
                pseShiftBatch_, pseShiftN, pseShiftS0, pseShiftS1_, pseShiftS1_, seqSize_);
        return custom::graphStatus::GRAPH_FAILED;
    }

    pseShiftFlag_ = true;
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckTndMaskShapeWithSparseMode()
{
    auto maskShape = ifaContext_->attenMask; // input shape = 4
    if (sparseMode_ == 0U) {                 // sparse 0 模式只支持无mask
        OP_CHECK_IF(maskShape.hasValue,
                    OP_LOGE(ifaContext_->opName, "TND/TND_NTD only support noMask when sparse = 0."),
                    return custom::graphStatus::GRAPH_FAILED);
        attenMaskFlag_ = false;
    } else if (sparseMode_ == 3U) { // sparse 3 模式需要attenMask
        OP_CHECK_IF(!maskShape.hasValue,
                    OP_LOGE(ifaContext_->opName, "TND/TND_NTD need input attenMask when sparse = 3."),
                    return custom::graphStatus::GRAPH_FAILED);

        auto shape = ifaContext_->attenMask.shape;
        OP_CHECK_IF(
            shape.size() != 2U || shape[0] != 2048 || shape[1] != 2048, // 2， 只支持2维shape
            OP_LOGE(ifaContext_->opName, "TND/TND_NTD when sparse = 3, atten_mask tensor shape must be [2048, 2048]."),
            return custom::graphStatus::GRAPH_FAILED);
        attenMaskFlag_ = true;
    } else {
        OP_LOGE(ifaContext_->opName, "TND/TND_NTD only support sparse(%u) = 0 or 3.", sparseMode_);
        return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckMaskShapeWithQSeq() const
{
    if (antiQuantFlag_ || (ifaContext_->dequantScaleQuery.hasValue && ropeFlag_)) {
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ > 1U && static_cast<int32_t>(sparseMode_) != 3),
                    OP_LOGE(ifaContext_->opName,
                            "when queryS > 1, sparseMode(%d) only support 3 "
                            "in MLA when antiquant or full quant situation.",
                            static_cast<int32_t>(sparseMode_)),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U && static_cast<int32_t>(sparseMode_) != 0),
                    OP_LOGE(ifaContext_->opName,
                            "when queryS = 1, sparseMode(%d) only support 0 "
                            "in MLA when antiquant or full quant situation.",
                            static_cast<int32_t>(sparseMode_)),
                    return custom::graphStatus::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ > 1U &&
                     (static_cast<int32_t>(sparseMode_) != 3 &&
                      static_cast<int32_t>(sparseMode_) != 4)), // 3 : rightDown, 4 : band
                    OP_LOGE(ifaContext_->opName,
                            "when queryS > 1, sparseMode(%d) support 3 or 4 "
                            "in MLA when no quant situation.",
                            static_cast<int32_t>(sparseMode_)),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U &&
                     (static_cast<int32_t>(sparseMode_) != 0 && static_cast<int32_t>(sparseMode_) != 4)),
                    OP_LOGE(ifaContext_->opName,
                            "when queryS = 1, sparseMode(%d) support 0 or 4 in MLA "
                            "when no quant situation.",
                            static_cast<int32_t>(sparseMode_)),
                    return custom::graphStatus::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (ropeFlag_ && (qSeqSize_ > 1U || static_cast<int32_t>(sparseMode_) == 4) && !ifaContext_->attenMask.hasValue),
        OP_LOGE(ifaContext_->opName, "when queryS > 1 and sparse = 3, or sparse = 4, "
                                     "atten_mask tensor shape must be [2048, 2048] in MLA."),
        return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(
        (ropeFlag_ && qSeqSize_ == 1U && ifaContext_->attenMask.hasValue && static_cast<int32_t>(sparseMode_) != 4),
        OP_LOGE(ifaContext_->opName, "when queryS = 1 and sparseMode is not 4, atten_mask tensor must be null in MLA."),
        return custom::graphStatus::GRAPH_FAILED);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckAttenMaskShape()
{
    auto maskShape = ifaContext_->attenMask; // input shape = 4
    if (!maskShape.hasValue) {
        attenMaskFlag_ = false;
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    if (maskShape.GetShapeSize() == 0) {
        attenMaskFlag_ = false;
        OP_LOGW(ifaContext_->opName, "atten_mask tensor exist, but atten_mask shape size is 0.");
        return custom::graphStatus::GRAPH_SUCCESS;
    }
    return custom::graphStatus::GRAPH_FAILED;
}

custom::graphStatus IFATiling::ProcessAttenMask()
{
    // 与pfa保持一致，先判断sparsemode
    sparseMode_ = ifaContext_->sparseMode;

    if (inputLayout_ == IfaLayout::TND) {
        return CheckTndMaskShapeWithSparseMode();
    }

    if (CheckMaskShapeWithQSeq() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (CheckAttenMaskShape() == custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    at::ScalarType attenMaskType = ifaContext_->attenMask.dType;
    if (attenMaskType != at::ScalarType::Bool && attenMaskType != at::ScalarType::Char &&
        attenMaskType != at::ScalarType::Byte) {
        OP_LOGE(ifaContext_->opName, "not support atten_mask type %d, only support bool, int8 and uint8.",
                attenMaskType);
        return custom::graphStatus::GRAPH_FAILED;
    }

    if ((sparseMode_ == 3 || sparseMode_ == 4 || gqaMtpFlag_) &&
        (socVersion_ != IfaSocVersion::SOC_ASCEND_310P)) { // 3 : rightDown, 4 : band
        auto shape = ifaContext_->attenMask.shape;
        OP_CHECK_IF(
            (sparseMode_ != 0 && sparseMode_ != 1) &&
                (shape.size() != 2U || shape[0] != 2048 || shape[1] != 2048), // 2， 只支持2维shape
            OP_LOGE(ifaContext_->opName, "when sparse = %d, atten_mask tensor shape must be [2048, 2048] in MLA.",
                    static_cast<int32_t>(sparseMode_)),
            return custom::graphStatus::GRAPH_FAILED);

        attenMaskFlag_ = true;
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    auto maskShape = ifaContext_->attenMask.shape; // input shape = 4
    uint32_t batchSizeOfMask = static_cast<uint32_t>(maskShape[0]);
    if (batchSizeOfMask != batchSize_) {
        OP_LOGE(ifaContext_->opName, "batchSize[%u] of atten_mask must be equal to batchSize[%u] of query.",
                batchSizeOfMask, batchSize_);
        return custom::graphStatus::GRAPH_FAILED;
    }

    auto dimNumOfMask = maskShape.size();
    attenMaskSize_ = static_cast<uint32_t>(maskShape[dimNumOfMask - 1]);
    uint32_t minAttenMaskSize = pageAttentionFlag_ ? sMax_ : maxActualseq_;
    if (attenMaskSize_ < minAttenMaskSize) {
        OP_LOGE(ifaContext_->opName, "s Size[%u] of atten_mask must be greater than or equal to sMax[%u].",
                attenMaskSize_, minAttenMaskSize);
        return custom::graphStatus::GRAPH_FAILED;
    }

    attenMaskFlag_ = true;
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckDefaultMisc(std::string scene) const
{
    OP_CHECK_IF(pseShiftFlag_, OP_LOGE(ifaContext_->opName, "PseShift is not supported in %s.", scene.c_str()),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(!batchContinuousFlag_, OP_LOGE(ifaContext_->opName, "Kvcache must be continues in %s.", scene.c_str()),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(sysPrefixFlag_, OP_LOGE(ifaContext_->opName, "SysPrefix is not supported in %s.", scene.c_str()),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(outputType_ == at::ScalarType::Char,
                OP_LOGE(ifaContext_->opName, "PostQuant is not supported in %s.", scene.c_str()),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(kvPaddingSizeFlag_,
                OP_LOGE(ifaContext_->opName, "kvPaddingSizeFlag_ is not supported in %s.", scene.c_str()),
                return custom::graphStatus::GRAPH_FAILED);

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckGqaTensor() const
{
    if (CheckGqaIOTensor() != custom::graphStatus::GRAPH_SUCCESS ||
        CheckGqaAntiquantTensor() != custom::graphStatus::GRAPH_SUCCESS ||
        CheckGqaBlockTable() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    // other tensors should be empty
    return CheckGqaTensorEmpty();
}

custom::graphStatus IFATiling::CheckGqaIOTensor() const
{
    if (!antiQuantFlag_) {
        OP_LOGE(ifaContext_->opName, "IFA GQA with KV NZ only support antiquant!");
        return custom::graphStatus::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        ifaContext_->query.dType != at::ScalarType::BFloat16,
        OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, query type should be BFLOAT16, but now it's %d!",
                ifaContext_->query.dType),
        return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->key.dType != at::ScalarType::Char,
        OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, key type should be INT8, but now it's %d!",
                ifaContext_->key.dType),
        return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->value.dType != at::ScalarType::Char,
        OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, value type should be INT8, but now it's %d!",
                ifaContext_->value.dType),
        return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->attenOut.dType != at::ScalarType::BFloat16,
                OP_LOGE(ifaContext_->opName,
                        "In IFA GQA with KV NZ antiquant, attenOut type should be BFLOAT16, but now it's %d!",
                        ifaContext_->attenOut.dType),
                return custom::graphStatus::GRAPH_FAILED);
    constexpr uint32_t kvNzDimNum = 5U;
    if (ifaContext_->key.shape.size() == kvNzDimNum) { // NZ
        std::string kShapeStr = GetTensorDimString(ifaContext_->key.shape);
        std::string vShapeStr = GetTensorDimString(ifaContext_->value.shape);
        OP_CHECK_IF(IsZeroDimTensor(ifaContext_->key.shape),
                    OP_LOGE(ifaContext_->opName, "The shape of K%s should not have 0 dim in GQA with KV NZ!",
                            kShapeStr.c_str()),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(IsZeroDimTensor(ifaContext_->value.shape),
                    OP_LOGE(ifaContext_->opName, "The shape of V%s should not have 0 dim in GQA with KV NZ!",
                            vShapeStr.c_str()),
                    return custom::graphStatus::GRAPH_FAILED);
        uint32_t expected3rdDim = headDim_ / 32U;
        uint32_t expected5thDim = 32U; // 32U / sizeof(int8)
        bool isK3rdDimValid = expected3rdDim == ifaContext_->key.shape[2];
        bool isK5thDimValid = expected5thDim == ifaContext_->key.shape[4];
        OP_CHECK_IF(
            !isK3rdDimValid || !isK5thDimValid,
            OP_LOGE(ifaContext_->opName, "The shape of K%s should be [%ld, %ld, %u, %ld, %u] in GQA with KV NZ!",
                    kShapeStr.c_str(), ifaContext_->key.shape[0], ifaContext_->key.shape[1], expected3rdDim,
                    ifaContext_->key.shape[3], expected5thDim),
            return custom::graphStatus::GRAPH_FAILED);
        bool isV3rdDimValid = expected3rdDim == ifaContext_->value.shape[2];
        bool isV5thDimValid = expected5thDim == ifaContext_->value.shape[4];
        OP_CHECK_IF(
            !isV3rdDimValid || !isV5thDimValid,
            OP_LOGE(ifaContext_->opName, "The shape of V%s should be [%ld, %ld, %u, %ld, %u] in GQA with KV NZ!",
                    kShapeStr.c_str(), ifaContext_->value.shape[0], ifaContext_->value.shape[1], expected3rdDim,
                    ifaContext_->value.shape[3], expected5thDim),
            return custom::graphStatus::GRAPH_FAILED);
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckGqaAntiquantTensor() const
{
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckGqaBlockTable() const
{
    OP_CHECK_IF(!pageAttentionFlag_,
                OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, blocktable should not be null!"),
                return custom::graphStatus::GRAPH_FAILED);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckGqaTensorEmpty() const
{
    OP_CHECK_IF(ifaContext_->pseShift.hasValue,
                OP_LOGE(ifaContext_->opName, "PseShift not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSeqLengthsQ.hasValue,
                OP_LOGE(ifaContext_->opName, "The query's actual sequence lengths not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale1.hasValue,
                OP_LOGE(ifaContext_->opName, "deqScale1 not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale1.hasValue,
                OP_LOGE(ifaContext_->opName, "quantScale1 not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale2.hasValue,
                OP_LOGE(ifaContext_->opName, "deqScale2 not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale2.hasValue,
                OP_LOGE(ifaContext_->opName, "the output's dequant scale not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantOffset2.hasValue,
                OP_LOGE(ifaContext_->opName, "the output's dequant offset not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantScale.hasValue,
                OP_LOGE(ifaContext_->opName, "antiquantScale not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantOffset.hasValue,
                OP_LOGE(ifaContext_->opName, "antiquantOffset not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryPaddingSize.hasValue,
                OP_LOGE(ifaContext_->opName, "queryPaddingSize not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvPaddingSize.hasValue,
                OP_LOGE(ifaContext_->opName, "kvPaddingSize not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyAntiquantOffset.hasValue,
                OP_LOGE(ifaContext_->opName, "the key's dequant offset not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueAntiquantOffset.hasValue,
                OP_LOGE(ifaContext_->opName, "the value's dequant offset not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.hasValue,
                OP_LOGE(ifaContext_->opName, "keySharedPrefix not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.hasValue,
                OP_LOGE(ifaContext_->opName, "valueSharedPrefix not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSharedPrefixLen.hasValue,
                OP_LOGE(ifaContext_->opName, "actualSharedPrefixLen not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryRope.hasValue,
                OP_LOGE(ifaContext_->opName, "queryRope not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRope.hasValue,
                OP_LOGE(ifaContext_->opName, "keyRope not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRopeAntiquantScale.hasValue,
                OP_LOGE(ifaContext_->opName, "the key_rope's dequant scale not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->dequantScaleQuery.hasValue,
                OP_LOGE(ifaContext_->opName, "The query's dequant scale not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->lseOut.GetShapeSize() != 0,
                OP_LOGE(ifaContext_->opName, "lseOut not support for IFA GQA with KV NZ!"),
                return custom::graphStatus::GRAPH_FAILED);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckGqaSeqSize() const
{
    std::string layout(ifaContext_->layOut);
    if (layout == "TND" && isWorkspace_) { // tiling下沉场景
        return custom::graphStatus::GRAPH_SUCCESS;
    }
    if (qSeqSize_ == 1U) {
        if (layout == "TND") { // TND MTP场景qSeqSize有可能为1，也需要支持sparseMode3
            OP_CHECK_IF(
                (sparseMode_ != 0 && sparseMode_ != 3),
                OP_LOGE(ifaContext_->opName,
                        "SparseMode[%u] only support 0 or 3 in IFA GQA with KV NZ when query is 1 and layout is TND",
                        sparseMode_),
                return custom::graphStatus::GRAPH_FAILED);
        } else {
            OP_CHECK_IF((sparseMode_ != 0),
                        OP_LOGE(ifaContext_->opName,
                                "SparseMode[%u] only support 0 in IFA GQA with KV NZ when query is 1", sparseMode_),
                        return custom::graphStatus::GRAPH_FAILED);
        }
    } else if (qSeqSize_ > 1U) {
        OP_CHECK_IF(
            (sparseMode_ != 3), // when qs bigger than 1, sparse mode only support 3
            OP_LOGE(ifaContext_->opName,
                    "SparseMode[%u] only support 3 in IFA GQA with KV NZ when query S is bigger than 1", sparseMode_),
            return custom::graphStatus::GRAPH_FAILED);
    } else {
        OP_LOGE(ifaContext_->opName, "Invalid query S %u", qSeqSize_);
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (sparseMode_ == 0) {
        OP_CHECK_IF(ifaContext_->attenMask.hasValue,
                    OP_LOGE(ifaContext_->opName, "attenMask should be null for IFA GQA with KV NZ when sparseMode 0!"),
                    return custom::graphStatus::GRAPH_FAILED);
    } else {
        auto& attenMaskShape = ifaContext_->attenMask.shape;
        auto& attenMaskTensor = ifaContext_->attenMask;
        OP_CHECK_IF(attenMaskShape.size() == 0 || !attenMaskTensor.hasValue, // mask shape: 2048*2048
                    OP_LOGE(ifaContext_->opName,
                            "When sparseMode 3, attenMask shape for IFA GQA with KV NZ should not be null!"),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(attenMaskTensor.shape.size() != 2U, // mask shape: 2048*2048
                    OP_LOGE(ifaContext_->opName,
                            "The dim of attenMask shape[%lu] is not expected. "
                            "Expect 2 when sparseMode 3 in GQA KV NZ.",
                            attenMaskShape.size()),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(attenMaskTensor.shape[0] != 2048U || attenMaskTensor.shape[1] != 2048U, // mask shape: 2048*2048
                    OP_LOGE(ifaContext_->opName,
                            "The shape of attenMask shape[%ld, %ld] is not expected. "
                            "Expect [2048, 2048] when sparseMode 3 in GQA KV NZ.",
                            attenMaskTensor.shape[0], attenMaskTensor.shape[1]),
                    return custom::graphStatus::GRAPH_FAILED);
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckGqaAttribute() const
{
    OP_CHECK_IF(headDim_ != 128U, // 128: D dim
                OP_LOGE(ifaContext_->opName, "headDim = %u, IFA GQA with KV NZ only support 128.", headDim_),
                return custom::graphStatus::GRAPH_FAILED);

    std::string layout(ifaContext_->layOut);
    OP_CHECK_IF(
        layout != "BSH" && layout != "BSND" && layout != "BNSD" && layout != "TND",
        OP_LOGE(ifaContext_->opName,
                "In IFA GQA with KV NZ antiquant, only BSH, BSND, BNSD and TND layout are supported, but now it's %s",
                layout.c_str()),
        return custom::graphStatus::GRAPH_FAILED);

    int64_t keyAntiquantMode = ifaContext_->keyAntiquantMode;
    int64_t valueAntiquantMode = ifaContext_->valueAntiquantMode;
    OP_CHECK_IF(
        antiquantMode_ == PER_CHANNEL_MODE && (keyAntiquantMode != 0 || valueAntiquantMode != 0),
        OP_LOGE(ifaContext_->opName,
                "the key's quant mode(%ld) and the value's quant mode(%ld) should be 0 in GQA perchannel antiquant.",
                keyAntiquantMode, valueAntiquantMode),
        return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(
        antiquantMode_ == PER_TOKEN_MODE && (keyAntiquantMode != 1 || valueAntiquantMode != 1),
        OP_LOGE(ifaContext_->opName,
                "the key's quant mode(%ld) and the value's quant mode(%ld) should be 1 in GQA pertoken antiquant.",
                keyAntiquantMode, valueAntiquantMode),
        return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ != 128U && blockSize_ != 512U, // block size only support 128 and 512 in GQA
                OP_LOGE(ifaContext_->opName, "blockSize(%u) only support 128 and 512 in GQA.", blockSize_),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(
        (innerPrecise_ != IFA_HIGH_PERFORMANCE),
        OP_LOGE(ifaContext_->opName, "precision mode[%u] only support 1(hign performance) in GQA antiquant with KV NZ",
                innerPrecise_),
        return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckGqaSeqSize() != custom::graphStatus::GRAPH_SUCCESS,
        OP_LOGE(ifaContext_->opName, "Invalid query S %u with sparseMode %u and attenMask", qSeqSize_, sparseMode_),
        return custom::graphStatus::GRAPH_FAILED);

    if (CheckDefaultMisc("IFA GQA with KV NZ") != custom::graphStatus::GRAPH_SUCCESS ||
        CheckGqaDefault() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckGqaDefault() const
{
    int64_t queryQuantMode = ifaContext_->queryQuantMode;
    OP_CHECK_IF(
        queryQuantMode != DEQUANT_PER_CHANNEL_MODE,
        OP_LOGE(ifaContext_->opName, "The query's quant mode(%ld) for IFA GQA with KV NZ should be default value %u!",
                queryQuantMode, DEQUANT_PER_CHANNEL_MODE),
        return custom::graphStatus::GRAPH_FAILED);
    return custom::graphStatus::GRAPH_SUCCESS;
}
} // namespace optiling
