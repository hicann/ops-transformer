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
 * \file scatter_pa_cache.cpp
 * \brief scatter_pa_cache op host
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> DataType = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_INT8,        ge::DT_UINT8,         ge::DT_INT16,       ge::DT_UINT16,
    ge::DT_INT32, ge::DT_UINT32,  ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_INT8,        ge::DT_UINT8,         ge::DT_INT16,       ge::DT_UINT16,
    ge::DT_INT32, ge::DT_UINT32,  ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2,
};

static const std::vector<ge::DataType> indexDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
};

static const std::vector<ge::Format> DataFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
};

class ScatterPaCache : public OpDef
{
public:
    explicit ScatterPaCache(const char* name) : OpDef(name)
    {
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType(DataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .AutoContiguous();
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType(DataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .AutoContiguous();
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType(indexDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .AutoContiguous();
        this->Input("compress_lens")
            .ParamType(OPTIONAL)
            .DataType(indexDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .AutoContiguous();
        this->Input("compress_seq_offset")
            .ParamType(OPTIONAL)
            .DataType(indexDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .AutoContiguous();
        this->Input("seq_lens")
            .ParamType(OPTIONAL)
            .DataType(indexDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .AutoContiguous();
        this->Output("key_cache")
            .ParamType(REQUIRED)
            .DataType(DataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .AutoContiguous();
        OpAICoreConfig config_950;
        config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "scatter_pa_cache");
        this->Attr("cache_mode").AttrType(OPTIONAL).String("Norm");
        this->AICore().AddConfig("ascend950", config_950);
        this->AICore().AddConfig("ascend350", config_950);
    }
};
OP_ADD(ScatterPaCache);
} // namespace ops
