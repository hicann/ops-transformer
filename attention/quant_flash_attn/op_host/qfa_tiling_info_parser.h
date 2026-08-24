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
 * \file qfa_tiling_info_parser.h
 * \brief QuantFlashAttn tiling info parser
 */

#pragma once

#include "qfa_tiling_info.h"
#include "qfa_tiling_shape.h"

namespace optiling {
namespace quant_flash_attn {
class QfaInfoParser {
public:
    explicit QfaInfoParser(const gert::TilingContext *context)
        : context_(context)
    {}
    ~QfaInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetEmptyTensorFlag();
    ge::graphStatus GetCuSeqLenQSize(int64_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();

    void GetOptionalInputParaQuantInfo();
    void GetOptionalInputParaMaskInfo();
    void GetOptionalInputParaSeqLengthInfo();
    void GetOptionalInputParaSinksInfo();

    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetOpParaInfo();

    void GetInOutDataType();
    ge::graphStatus GetBatchSize();
    void GetQueryTSize();
    void GetKeyTSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetS1Size();
    void GetKvStorageMode();

    void SetQfaShape();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetBlockNum();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetInAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    void GetMaskParams();
    ge::graphStatus GetQuantMode();
    ge::graphStatus GetBlockSize();

    ge::graphStatus GetActualSeqInfo();

    void GenerateAxisInfo(QfaTilingInfo &qfaInfo);
    void GenerateDtypeInfo(QfaTilingInfo &qfaInfo);
    void GenerateFeatureInfo(QfaTilingInfo &qfaInfo);
    void GenerateLayoutInfo(QfaTilingInfo &qfaInfo);
    void GenerateQuantInfo(QfaTilingInfo &qfaInfo);
    void GenerateInfo(QfaTilingInfo &qfaInfo);
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus ParseFeatureInfo();
    ge::graphStatus Parse(QfaTilingInfo &qfaInfo);

private:
    const gert::TilingContext *context_ = nullptr;

    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    QfaParaInfo opParamInfo_;

    // BaseParams
    int64_t bSize_ = 0;
    int64_t n1Size_ = 0;
    int64_t n2Size_ = 0;
    int64_t gSize_ = 0;
    int64_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t qkHeadDim_ = 0;
    int64_t vHeadDim_ = 0;
    int64_t queryTSize_ = 0;
    int64_t keyTSize_ = 0;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;

    // Layout
    QfaLayout layoutQ_ = QfaLayout::BSND;
    QfaLayout layoutKV_ = QfaLayout::BSND;
    QfaLayout layoutOut_ = QfaLayout::BSND;
    QfaLayout layoutQDescale_ = QfaLayout::BSND;

    // Strides (for non-contiguous tensor check)
    bool hasStride_ = false;
    const gert::Stride *keyStrides_ = nullptr;
    const gert::Stride *valueStrides_ = nullptr;
    const gert::Stride *kDescaleStrides_ = nullptr;
    const gert::Stride *vDescaleStrides_ = nullptr;

    // PageAttention
    int64_t maxBlockNumPerBatch_ = 0;
    int32_t blockNum_ = 0;

    // NPU
    NpuArch npuArch_ = NpuArch::DAV_3510;

    // Dtype
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_BF16;
    ge::DataType qDescaleType_ = ge::DT_FLOAT8_E8M0;
    ge::DataType kDescaleType_ = ge::DT_FLOAT8_E8M0;
    ge::DataType vDescaleType_ = ge::DT_FLOAT8_E8M0;

    // Mask
    int64_t winLeft_ = 0;
    int64_t winRight_ = 0;
    int64_t maskMode_ = 0;

    // Quant
    QfaQuantMode quantMode_ = QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32;
    int64_t blockSize_ = 0;

    // Sinks
    bool sinksFlag_ = false;

    // SoftmaxLSE
    bool softmaxLseFlag_ = false;
    bool returnSoftmaxLse_ = false;

    // Other attrs
    float softmaxScale_ = 1.0;

    // SeqLen
    int64_t seqLenQDims_ = 0;
    int64_t seqLenKvDims_ = 0;
    int64_t cuseqLenQDims_ = 0;
    int64_t cuseqLenKvDims_ = 0;
    int64_t maxSeqQ_ = -1;
    int64_t maxSeqKv_ = -1;

    // Empty Tensor
    bool emptyTensorFlag_ = false;

    // Shape
    std::shared_ptr<QfaTilingShape> queryShape_ = nullptr;
    std::shared_ptr<QfaTilingShape> keyShape_ = nullptr;
    std::shared_ptr<QfaTilingShape> valueShape_ = nullptr;
};
} // namespace quant_flash_attn
} // namespace optiling
