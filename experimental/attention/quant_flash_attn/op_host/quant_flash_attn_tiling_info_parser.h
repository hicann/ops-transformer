/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_tiling_info_parser.h
 * \brief
 */

#pragma once

#include "quant_flash_attn_tiling_info.h"
#include "../common/op_host/fia_tiling_shape.h"

namespace optiling {
class QuantFlashAttnTilingInfoParser {
public:
    explicit QuantFlashAttnTilingInfoParser(const gert::TilingContext *context, QuantFlashAttnTilingInfo &faInfo)
        : context_(context),
          tilingInfo_(faInfo)
    {}
    ~QuantFlashAttnTilingInfoParser() = default;
    ge::graphStatus Parse();

private:
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckOptionalInputExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetCuSeqLenQDims();
    ge::graphStatus GetCuSeqLenKvDims();
    ge::graphStatus GetSeqUsedQDims();
    ge::graphStatus GetSeqUsedKvDims();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetValueHeadDim();
    void GetQueryTSize();
    void GetKeyTSize();
    ge::graphStatus GetMaxSeqLenQ();
    ge::graphStatus GetMaxSeqLenKv();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetInAndOutLayout();
    void GetPreNextToken();
    ge::graphStatus GetQkvDataType();
    void SetFaShape();
    void GetKvStorageMode();
    void GetSoftmaxScale();
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus ParseFeatureInfo();
    ge::graphStatus GetEmptyTensorFlag();

private:
    const gert::TilingContext *context_ = nullptr;
    QuantFlashAttnTilingInfo &tilingInfo_;

    // NPU信息
    NpuArch npuArch_ = NpuArch::DAV_3510;

    bool emptyTensorFlag_ = false;

    // shape信息
    std::shared_ptr<FiaTilingShape> queryShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> valueShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> qDescaleShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> kDescaleShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> vDescaleShape_ = nullptr;
};
} // namespace optiling
