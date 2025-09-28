/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_info_parser.h
 * \brief
 */

#pragma once

#include "../../common/op_host/fia_tiling_info.h"
#include "../../common/op_host/fia_tiling_shape.h"

namespace optiling {
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
class FiaInfoParser {
public:
    explicit FiaInfoParser(const gert::TilingContext *context) : context_(context) {}
    ~FiaInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        const FiaLayout &layout, const std::string &actualSeqLenName, const std::string &attrName);
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaPostQuantInfo();
    void GetOptionalInputParaPrefixInfo();
    void GetOptionalInputParaRopeInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetKvCache();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetMaxWorkspaceFlag();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetKvStorageMode();
    ge::graphStatus GetKvLayout();
    void SetFiaShape();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetS2SizeForTensorList();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetRopeMode();
    ge::graphStatus GetRopeHeadDim();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetPaddingSizeFlag();
    ge::graphStatus GetActualSeqInfo();
    ge::graphStatus GetPreNextToken();
    TilingKeyLayout MapStringToLayout(FiaLayout &layoutString) const;
    void GenerateAxisInfo(FiaTilingInfo &fiaInfo);
    void GenerateDtypeInfo(FiaTilingInfo &fiaInfo);
    void GenerateInfo(FiaTilingInfo &fiaInfo);
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus ParseFeatureInfo();
    ge::graphStatus Parse(FiaTilingInfo &fiaInfo);

public:
    const gert::TilingContext *context_ = nullptr;

    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    FIAParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t qTSize_ = 0; // 仅TND/NTD时生效
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    RopeMode ropeMode_ = RopeMode::NO_ROPE;

    // Layout
    FiaLayout qLayout_ = FiaLayout::BSND;
    FiaLayout outLayout_ = FiaLayout::BSND;
    FiaLayout kvLayout_ = FiaLayout::BSND;

    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;

    // 局部参数, 暂存
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType inputQRopeType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;

    uint64_t l2CacheSize_ = 0;
    std::vector<gert::StorageShape *> kCache_ = {};
    std::vector<gert::StorageShape *> vCache_ = {};

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    bool attenMaskFlag_ = false;
    bool needInit_ = false;
    int64_t preToken_ = 0;
    int64_t nextToken_ = 0;
    uint32_t attenMaskSize_ = 0;
    bool kvPaddingSizeFlag_ = false;
    int64_t maxActualseq_ = 0;
    bool isMaxWorkspace_ = false;

    uint32_t actualLenQDims_ = 0;
    uint32_t actualLenDims_ = 0;
    std::vector<int64_t> kvListSeqLens_ {};

    std::shared_ptr<FiaTilingShape> queryShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> valueShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> queryRopeShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyRopeShape_ = nullptr;
};
} // optiling