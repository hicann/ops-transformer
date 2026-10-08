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
 * \file qmla_tiling_info_parser.cpp
 * \brief QuantFlashMlaWithKvcache tiling info parser 实现
 */

#include <map>
#include <string>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "qmla_tiling_info_parser.h"

using std::map;
using std::string;
using namespace ge;
using namespace Ops::Base;

namespace optiling {
namespace quant_flash_mla_with_kvcache {

static const std::map<std::string, QmlaLayout> QMLA_LAYOUT_Q_MAP = {
    {"BSND", QmlaLayout::BSND},
    {"BNSD", QmlaLayout::BNSD},
    {"TND", QmlaLayout::TND},
};

static const std::map<std::string, QmlaOutLayout> QMLA_LAYOUT_OUT_MAP = {
    {"BSND", QmlaOutLayout::BSND},
    {"BNSD", QmlaOutLayout::BNSD},
    {"TND", QmlaOutLayout::TND},
    {"NTD", QmlaOutLayout::NTD},
};

static const std::map<std::string, QmlaKvLayout> QMLA_LAYOUT_KV_MAP = {
    {"PA_BBND", QmlaKvLayout::PA_BBND},
    {"PA_BNBD", QmlaKvLayout::PA_BNBD},
    {"PA_NZ", QmlaKvLayout::PA_NZ},
};

ge::graphStatus QmlaInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("QuantFlashMlaWithKvcache", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::GetNpuInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::CheckRequiredParaExistence() const
{
    auto checkTensor = [this](const QmlaTensorInfo& t, const char* name) -> bool {
        if (t.shape == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName_, (std::string("Shape of ") + name).c_str());
            return false;
        }
        if (t.desc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName_, (std::string("Desc of ") + name).c_str());
            return false;
        }
        return true;
    };
    if (!checkTensor(opParamInfo_.query, QUERY_NAME) || !checkTensor(opParamInfo_.kCache, K_CACHE_NAME) ||
        !checkTensor(opParamInfo_.qDescale, Q_DESCALE_NAME) || !checkTensor(opParamInfo_.kDescale, K_DESCALE_NAME) ||
        !checkTensor(opParamInfo_.blockTable, BLOCK_TABLE_NAME) ||
        !checkTensor(opParamInfo_.cacheSeqlens, CACHE_SEQLENS_NAME) ||
        !checkTensor(opParamInfo_.attnOut, ATTN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.quantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "quant_mode"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::GetEmptyTensorFlag()
{
    auto checkEmptyTensor = [this](const gert::StorageShape* shape, const std::string& name) -> bool {
        if (shape == nullptr) {
            return false;
        }
        for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
            if (shape->GetStorageShape().GetDim(i) == 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, ToString(shape->GetStorageShape()).c_str(), name.c_str(),
                                                      ("Tensor " + name + " has empty dimension at axis " +
                                                       std::to_string(i) + ", size is 0, which is not supported")
                                                          .c_str());
                return true;
            }
        }
        return false;
    };
    if (checkEmptyTensor(opParamInfo_.query.shape, QUERY_NAME) ||
        checkEmptyTensor(opParamInfo_.kCache.shape, K_CACHE_NAME) ||
        checkEmptyTensor(opParamInfo_.qDescale.shape, Q_DESCALE_NAME) ||
        checkEmptyTensor(opParamInfo_.kDescale.shape, K_DESCALE_NAME) ||
        checkEmptyTensor(opParamInfo_.blockTable.shape, BLOCK_TABLE_NAME) ||
        checkEmptyTensor(opParamInfo_.cacheSeqlens.shape, CACHE_SEQLENS_NAME) ||
        checkEmptyTensor(opParamInfo_.attnOut.shape, ATTN_OUT_NAME)) {
        emptyTensorFlag_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::GetOpParaInfo()
{
    auto fillTensorInfo = [this](QmlaTensorInfo& t, size_t idx, bool optional) {
        if (optional) {
            t.tensor = context_->GetOptionalInputTensor(idx);
            t.desc = context_->GetOptionalInputDesc(idx);
            t.shape = context_->GetOptionalInputShape(idx);
        } else {
            t.tensor = context_->GetInputTensor(idx);
            t.desc = context_->GetInputDesc(idx);
            t.shape = context_->GetInputShape(idx);
        }
    };

    fillTensorInfo(opParamInfo_.query, QUERY_INDEX, false);
    fillTensorInfo(opParamInfo_.kCache, K_CACHE_INDEX, false);
    fillTensorInfo(opParamInfo_.qDescale, Q_DESCALE_INDEX, false);
    fillTensorInfo(opParamInfo_.kDescale, K_DESCALE_INDEX, false);
    fillTensorInfo(opParamInfo_.blockTable, BLOCK_TABLE_INDEX, false);
    fillTensorInfo(opParamInfo_.cacheSeqlens, CACHE_SEQLENS_INDEX, false);
    fillTensorInfo(opParamInfo_.cuSeqlensQ, CU_SEQLENS_Q_INDEX, true);
    fillTensorInfo(opParamInfo_.sequsedQ, SEQUSED_Q_INDEX, true);
    fillTensorInfo(opParamInfo_.attnMask, ATTN_MASK_INDEX, true);
    fillTensorInfo(opParamInfo_.metadata, METADATA_INDEX, true);

    // k_cache为view输入时提取stride, 用于非连续Tensor校验与kernel侧offset计算
    if (context_->InputIsView(K_CACHE_INDEX)) {
        keyStrides_ = context_->GetInputStride(K_CACHE_INDEX);
        hasStride_ = keyStrides_ != nullptr;
    }

    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.softmaxLse.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.softmaxLse.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(opName_, "attrs got from ge is nullptr"), return ge::GRAPH_FAILED);
    opParamInfo_.quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    opParamInfo_.maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);
    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    opParamInfo_.maxSeqlenKv = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);
    opParamInfo_.headDimV = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_DIM_V_INDEX);
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    opParamInfo_.layoutKv = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);
    opParamInfo_.layoutOut = attrs->GetStr(ATTR_LAYOUT_OUT_INDEX);
    opParamInfo_.returnSoftmaxLse = attrs->GetAttrPointer<bool>(ATTR_RETURN_SOFTMAX_LSE_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::GetInAndOutLayout()
{
    OP_CHECK_IF(opParamInfo_.layoutQ == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_q"),
                return ge::GRAPH_FAILED);
    auto itQ = QMLA_LAYOUT_Q_MAP.find(opParamInfo_.layoutQ);
    if (itQ == QMLA_LAYOUT_Q_MAP.end()) {
        string reason = "layout_q: " + string(opParamInfo_.layoutQ) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", opParamInfo_.layoutQ, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutQ_ = itQ->second;

    OP_CHECK_IF(opParamInfo_.layoutKv == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_kv"),
                return ge::GRAPH_FAILED);
    auto itKv = QMLA_LAYOUT_KV_MAP.find(opParamInfo_.layoutKv);
    if (itKv == QMLA_LAYOUT_KV_MAP.end()) {
        string reason = "layout_kv: " + string(opParamInfo_.layoutKv) +
                        " is not supported, only PA_BBND/PA_BNBD/PA_NZ is supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_kv", opParamInfo_.layoutKv, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutKv_ = itKv->second;

    OP_CHECK_IF(opParamInfo_.layoutOut == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_out"),
                return ge::GRAPH_FAILED);
    auto itOut = QMLA_LAYOUT_OUT_MAP.find(opParamInfo_.layoutOut);
    if (itOut == QMLA_LAYOUT_OUT_MAP.end()) {
        string reason = "layout_out: " + string(opParamInfo_.layoutOut) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_out", opParamInfo_.layoutOut, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutOut_ = itOut->second;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::GetQuantMode()
{
    OP_CHECK_IF(opParamInfo_.quantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "quant_mode"),
                return ge::GRAPH_FAILED);
    int64_t quantModeVal = *opParamInfo_.quantMode;
    if (quantModeVal != static_cast<int64_t>(QmlaQuantMode::MLA_FP8_E4M3_FULLQUANT)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "quant_mode", std::to_string(quantModeVal).c_str(),
            "quant_mode must be 1 (MLA_FP8_E4M3_FULLQUANT: Q/K/V FP8_E4M3, Q per-token-head, K per-tensor)");
        return ge::GRAPH_FAILED;
    }
    quantMode_ = static_cast<QmlaQuantMode>(quantModeVal);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::GetQShapeInfo()
{
    const auto& qShape = opParamInfo_.query.shape->GetStorageShape();
    if (layoutQ_ == QmlaLayout::TND) {
        if (qShape.GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, QUERY_NAME,
                                                     (std::to_string(qShape.GetDimNum()) + "D").c_str(),
                                                     "The shape dim of q must be 3D when layout_q is TND");
            return ge::GRAPH_FAILED;
        }
        qTSize_ = qShape.GetDim(0);
        n1Size_ = qShape.GetDim(1);
        headDimQk_ = qShape.GetDim(2);
        OP_CHECK_IF(opParamInfo_.cuSeqlensQ.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, CU_SEQLENS_Q_NAME, "provided",
                                                          "When layout_q is TND, cu_seqlens_q must be provided"),
                    return ge::GRAPH_FAILED);
        int64_t cuSeqLenQSize = opParamInfo_.cuSeqlensQ.tensor->GetShapeSize();
        bSize_ = cuSeqLenQSize - 1;
        s1Size_ = (opParamInfo_.maxSeqlenQ != nullptr && *opParamInfo_.maxSeqlenQ > 0) ? *opParamInfo_.maxSeqlenQ : -1;
    } else {
        if (qShape.GetDimNum() != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, QUERY_NAME,
                                                     (std::to_string(qShape.GetDimNum()) + "D").c_str(),
                                                     "The shape dim of q must be 4D when layout_q is BSND/BNSD");
            return ge::GRAPH_FAILED;
        }
        if (layoutQ_ == QmlaLayout::BSND) {
            bSize_ = qShape.GetDim(0);
            s1Size_ = qShape.GetDim(1);
            n1Size_ = qShape.GetDim(2);
            headDimQk_ = qShape.GetDim(3);
        } else { // BNSD
            bSize_ = qShape.GetDim(0);
            n1Size_ = qShape.GetDim(1);
            s1Size_ = qShape.GetDim(2);
            headDimQk_ = qShape.GetDim(3);
        }
        qTSize_ = bSize_ * s1Size_;
    }
    gSize_ = 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::GetKvCacheShapeInfo()
{
    const auto& kvShape = opParamInfo_.kCache.shape->GetStorageShape();
    size_t kvDimNum = kvShape.GetDimNum();
    // PA_BBND: (blockNum, blockSize, KV_N, D) / PA_BNBD: (blockNum, KV_N, blockSize, D)
    if (layoutKv_ == QmlaKvLayout::PA_BBND) {
        if (kvDimNum != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, K_CACHE_NAME, (std::to_string(kvDimNum) + "D").c_str(),
                                                     "The shape dim of k_cache must be 4D when layout_kv is PA_BBND");
            return ge::GRAPH_FAILED;
        }
        blockNum_ = kvShape.GetDim(0);
        blockSize_ = kvShape.GetDim(1);
        n2Size_ = kvShape.GetDim(2);
    } else if (layoutKv_ == QmlaKvLayout::PA_BNBD) {
        if (kvDimNum != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, K_CACHE_NAME, (std::to_string(kvDimNum) + "D").c_str(),
                                                     "The shape dim of k_cache must be 4D when layout_kv is PA_BNBD");
            return ge::GRAPH_FAILED;
        }
        blockNum_ = kvShape.GetDim(0);
        n2Size_ = kvShape.GetDim(1);
        blockSize_ = kvShape.GetDim(2);
    } else { // PA_NZ
        if (kvDimNum != 5) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, K_CACHE_NAME, (std::to_string(kvDimNum) + "D").c_str(),
                                                     "The shape dim of k_cache must be 5D when layout_kv is PA_NZ");
            return ge::GRAPH_FAILED;
        }
        blockNum_ = kvShape.GetDim(0);
        n2Size_ = kvShape.GetDim(1);
        blockSize_ = kvShape.GetDim(3);
    }
    t2Size_ = blockNum_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::GetS2Size()
{
    const gert::Tensor* blockTable = opParamInfo_.blockTable.tensor;
    OP_CHECK_IF(blockTable == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, BLOCK_TABLE_NAME, "provided",
                                                      "block_table must be provided in paged attention scenario"),
                return ge::GRAPH_FAILED);
    if (blockTable->GetStorageShape().GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            opName_, BLOCK_TABLE_NAME, (std::to_string(blockTable->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of block_table must be 2D (B, maxBlockNumPerBatch)");
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = blockTable->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    maxSeqKv_ = (opParamInfo_.maxSeqlenKv != nullptr) ? *opParamInfo_.maxSeqlenKv : -1;
    if (maxSeqKv_ > 0 && maxSeqKv_ < s2Size_) {
        s2Size_ = maxSeqKv_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::ParseAxisInfo()
{
    if (GetQShapeInfo() != ge::GRAPH_SUCCESS || GetKvCacheShapeInfo() != ge::GRAPH_SUCCESS ||
        GetS2Size() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    headDimV_ = (opParamInfo_.headDimV != nullptr && *opParamInfo_.headDimV > 0) ? *opParamInfo_.headDimV : 512;

    // q_descale shape: BSND->(B,Q_S,Q_N), BNSD->(B,Q_N,Q_S), TND->(Q_T,Q_N)
    size_t qDescaleDimNum = opParamInfo_.qDescale.shape->GetStorageShape().GetDimNum();
    size_t expectDim = (layoutQ_ == QmlaLayout::TND) ? 2 : 3;
    if (qDescaleDimNum != expectDim) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            opName_, Q_DESCALE_NAME, (std::to_string(qDescaleDimNum) + "D").c_str(),
            (std::string("In MLA_FP8_FULLQUANT scenario, the shape dim of q_descale must be ") +
             std::to_string(expectDim) + "D")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    // k_descale: per-tensor量化, 1D (1,)
    if (opParamInfo_.kDescale.shape->GetStorageShape().GetDimNum() != 1 ||
        opParamInfo_.kDescale.shape->GetStorageShape().GetShapeSize() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            opName_, K_DESCALE_NAME,
            (std::to_string(opParamInfo_.kDescale.shape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "k_descale is per-tensor quantization, shape must be (1,)");
        return ge::GRAPH_FAILED;
    }

    maxSeqQ_ = (opParamInfo_.maxSeqlenQ != nullptr) ? *opParamInfo_.maxSeqlenQ : -1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QmlaInfoParser::ParseFeatureInfo()
{
    maskMode_ = (opParamInfo_.maskMode == nullptr) ? 0 : *opParamInfo_.maskMode;
    attnMaskFlag_ = (opParamInfo_.attnMask.tensor != nullptr);
    if (attnMaskFlag_) {
        const auto& maskShape = opParamInfo_.attnMask.shape->GetStorageShape();
        if (maskShape.GetDimNum() == 2) {
            attenMaskS1Size_ = maskShape.GetDim(0);
            attenMaskS2Size_ = maskShape.GetDim(1);
        }
    }

    cuSeqLenQFlag_ = (opParamInfo_.cuSeqlensQ.tensor != nullptr);
    seqUsedQFlag_ = (opParamInfo_.sequsedQ.tensor != nullptr);

    returnSoftmaxLse_ = (opParamInfo_.returnSoftmaxLse == nullptr) ? false : *opParamInfo_.returnSoftmaxLse;
    softmaxScale_ = (opParamInfo_.softmaxScale == nullptr) ? -1.0f : *opParamInfo_.softmaxScale;
    metadataFlag_ = (opParamInfo_.metadata.tensor != nullptr);
    return ge::GRAPH_SUCCESS;
}

void QmlaInfoParser::GenerateInfo(QmlaTilingInfo& qmlaInfo)
{
    qmlaInfo.opName = opName_;
    qmlaInfo.opParamInfo = opParamInfo_;

    qmlaInfo.bSize = bSize_;
    qmlaInfo.n1Size = n1Size_;
    qmlaInfo.n2Size = n2Size_;
    qmlaInfo.gSize = gSize_;
    qmlaInfo.s1Size = s1Size_;
    qmlaInfo.s2Size = s2Size_;
    qmlaInfo.headDimQk = headDimQk_;
    qmlaInfo.headDimV = headDimV_;
    qmlaInfo.qTSize = qTSize_;
    qmlaInfo.t2Size = t2Size_;

    qmlaInfo.cuSeqLenQFlag = cuSeqLenQFlag_;
    qmlaInfo.seqUsedQFlag = seqUsedQFlag_;
    qmlaInfo.maxSeqQ = maxSeqQ_;
    qmlaInfo.maxSeqKv = maxSeqKv_;

    qmlaInfo.blockSize = blockSize_;
    qmlaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;
    qmlaInfo.totalBlockNum = blockNum_;

    qmlaInfo.maskMode = maskMode_;
    qmlaInfo.attnMaskFlag = attnMaskFlag_;
    qmlaInfo.attenMaskS1Size = attenMaskS1Size_;
    qmlaInfo.attenMaskS2Size = attenMaskS2Size_;

    qmlaInfo.quantMode = quantMode_;
    qmlaInfo.softmaxScale = softmaxScale_;

    qmlaInfo.layoutQ = layoutQ_;
    qmlaInfo.layoutOut = layoutOut_;
    qmlaInfo.layoutKv = layoutKv_;

    qmlaInfo.keyStrides = keyStrides_;
    qmlaInfo.hasStride = hasStride_;

    qmlaInfo.returnSoftmaxLse = returnSoftmaxLse_;
    qmlaInfo.metadataFlag = metadataFlag_;
    qmlaInfo.emptyTensorFlag = emptyTensorFlag_;

    qmlaInfo.qType = qType_;
    qmlaInfo.kvType = kvType_;
    qmlaInfo.outType = outType_;
}

ge::graphStatus QmlaInfoParser::Parse(QmlaTilingInfo& qmlaInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("QuantFlashMlaWithKvcache", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    if (GetOpName() != ge::GRAPH_SUCCESS || GetNpuInfo() != ge::GRAPH_SUCCESS || GetOpParaInfo() != ge::GRAPH_SUCCESS ||
        CheckRequiredParaExistence() != ge::GRAPH_SUCCESS || GetEmptyTensorFlag() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (emptyTensorFlag_) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "input tensor", "",
                                              "Empty tensor (containing a dimension of size 0) is not supported");
        return ge::GRAPH_FAILED;
    }
    if (GetInAndOutLayout() != ge::GRAPH_SUCCESS || GetQuantMode() != ge::GRAPH_SUCCESS ||
        ParseAxisInfo() != ge::GRAPH_SUCCESS || ParseFeatureInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    qType_ = opParamInfo_.query.desc->GetDataType();
    kvType_ = opParamInfo_.kCache.desc->GetDataType();
    outType_ = opParamInfo_.attnOut.desc->GetDataType();

    GenerateInfo(qmlaInfo);
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
