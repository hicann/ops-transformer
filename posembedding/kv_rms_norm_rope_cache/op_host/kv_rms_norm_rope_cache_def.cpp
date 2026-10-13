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
 * \file kv_rms_norm_rope_cache_def.cpp
 * \brief KvRmsNormRopeCache op host config
 */

#include "register/op_def_registry.h"
constexpr uint32_t KV_RMS_NORM_ROPE_CACHE_VERSION_TWO = 2;

namespace ops {

static const std::vector<ge::DataType> kvDataTypeRegbase = {
ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,    ge::DT_BF16,
ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,    ge::DT_BF16,
ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,    ge::DT_BF16
};

static const std::vector<ge::DataType> indexDataTypeRegbase = {
ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,ge::DT_INT64,
ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
};

static const std::vector<ge::DataType> kCacheDataTypeRegbase = {
ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16,
ge::DT_HIFLOAT8,      ge::DT_FLOAT16, ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_BF16, ge::DT_HIFLOAT8,
ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_BF16, ge::DT_FLOAT8_E5M2,
ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN
};

static const std::vector<ge::DataType> ckvCacheDataTypeRegbase = {
ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT16, ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_BF16,
ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_BF16,
ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16
};

static const std::vector<ge::DataType> scaleDataTypeRegbase = {
ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
};

static const std::vector<ge::Format> formatRegbase = {
ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};

class KvRmsNormRopeCache : public OpDef {
public:
    explicit KvRmsNormRopeCache(const char* name) : OpDef(name)
    {
        this->Input("kv")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cos")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sin")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_cache")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ckv_cache")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
                 ge::DT_INT8})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_rope_scale")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                 ge::DT_FLOAT})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("c_kv_scale")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                 ge::DT_FLOAT})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_rope_offset")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                 ge::DT_FLOAT})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("c_kv_offset")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                 ge::DT_FLOAT})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("v")
            .ParamType(OPTIONAL)
            .Version(KV_RMS_NORM_ROPE_CACHE_VERSION_TWO)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("k_cache")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("ckv_cache")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
                 ge::DT_INT8})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("k_rope")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("c_kv")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                 ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
        this->Attr("cache_mode").AttrType(OPTIONAL).String("Norm");
        this->Attr("is_output_kv").AttrType(OPTIONAL).Bool(false);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config_950;
        config_950.Input("kv")
            .ParamType(REQUIRED)
            .DataType(kvDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("gamma")
            .ParamType(REQUIRED)
            .DataType(kvDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("cos")
            .ParamType(REQUIRED)
            .DataType(kvDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("sin")
            .ParamType(REQUIRED)
            .DataType(kvDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("index")
            .ParamType(REQUIRED)
            .DataType(indexDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("k_cache")
            .ParamType(REQUIRED)
            .DataType(kCacheDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("ckv_cache")
            .ParamType(REQUIRED)
            .DataType(ckvCacheDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("k_rope_scale")
            .ParamType(OPTIONAL)
            .DataType(scaleDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("c_kv_scale")
            .ParamType(OPTIONAL)
            .DataType(scaleDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("k_rope_offset")
            .ParamType(OPTIONAL)
            .DataType(scaleDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("c_kv_offset")
            .ParamType(OPTIONAL)
            .DataType(scaleDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Input("v")
            .ParamType(OPTIONAL)
            .Version(KV_RMS_NORM_ROPE_CACHE_VERSION_TWO)
            .DataType(kvDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        config_950.Output("k_cache")
            .ParamType(REQUIRED)
            .DataType(kCacheDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        config_950.Output("ckv_cache")
            .ParamType(REQUIRED)
            .DataType(ckvCacheDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        config_950.Output("k_rope")
            .ParamType(REQUIRED)
            .DataType(kvDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        config_950.Output("c_kv")
            .ParamType(REQUIRED)
            .DataType(kvDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        config_950.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "kv_rms_norm_rope_cache_apt");
        this->AICore().AddConfig("ascend950", config_950);
        this->AICore().AddConfig("ascend350", config_950);

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
        this->AICore().AddConfig("kirin9030", config_kirin);
    }

private:
    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig config_kirin;
        config_kirin.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config_kirin.Input("kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("k_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("ckv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("k_rope_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("c_kv_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("k_rope_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("c_kv_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("v")
            .ParamType(OPTIONAL)
            .Version(KV_RMS_NORM_ROPE_CACHE_VERSION_TWO)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Output("k_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("ckv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("c_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        return config_kirin;
    }
};

OP_ADD(KvRmsNormRopeCache);
} // namespace ops
