/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

#include <vector>

namespace ops {
class QkvRmsNormRopeCacheWithKScale : public OpDef {
public:
    explicit QkvRmsNormRopeCacheWithKScale(const char *name)
        : OpDef(name)
    {
        // Type groups: RoPE, M-RoPE, M-RoPE MX.
        const std::vector<ge::DataType> bf16Types = {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
        const std::vector<ge::DataType> fp32Types = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::DataType> int32Types = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::DataType> fp8Types = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN};
        const std::vector<ge::DataType> kCacheTypes = {ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN};
        const std::vector<ge::DataType> kScaleTypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};
        const std::vector<ge::DataType> qOutTypes = {ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN};
        const std::vector<ge::DataType> qScaleTypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};
        const std::vector<ge::Format> ndFormats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("qkv").ParamType(REQUIRED).DataType(bf16Types).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("q_gamma").ParamType(REQUIRED).DataType(fp32Types).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("k_gamma").ParamType(REQUIRED).DataType(fp32Types).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("cos_sin").ParamType(REQUIRED).DataType(fp32Types).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType(int32Types)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Input("k_cache")
            .ParamType(REQUIRED)
            .DataType(kCacheTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Input("v_cache").ParamType(REQUIRED).DataType(fp8Types).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("k_scale_cache")
            .ParamType(REQUIRED)
            .DataType(kScaleTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Input("query_start_loc")
            .ParamType(OPTIONAL)
            .DataType(int32Types)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Input("seq_lens")
            .ParamType(OPTIONAL)
            .DataType(int32Types)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Input("rotation").ParamType(OPTIONAL).DataType(bf16Types).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("v_scale").ParamType(OPTIONAL).DataType(fp32Types).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Input("mrope_position")
            .ParamType(OPTIONAL)
            .DataType(int32Types)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);

        this->Output("q_out").ParamType(REQUIRED).DataType(qOutTypes).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Output("q_scale")
            .ParamType(OPTIONAL)
            .DataType(qScaleTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Output("k_cache")
            .ParamType(REQUIRED)
            .DataType(kCacheTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);
        this->Output("v_cache").ParamType(REQUIRED).DataType(fp8Types).Format(ndFormats).UnknownShapeFormat(ndFormats);
        this->Output("k_scale_cache")
            .ParamType(REQUIRED)
            .DataType(kScaleTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);

        this->Attr("head_nums").AttrType(REQUIRED).ListInt();
        this->Attr("layout_qkv").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_q_out").AttrType(OPTIONAL).String("NTD");
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("mrope_section").AttrType(OPTIONAL).ListInt();
        this->Attr("q_quant_mode").AttrType(OPTIONAL).String("PerTokenPerHead");
        this->Attr("q_out_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN));
        this->Attr("k_quant_mode").AttrType(OPTIONAL).String("PerTokenPerHead");

        OpAICoreConfig config950;
        config950.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "qkv_rms_norm_rope_cache_with_k_scale_apt");
        this->AICore().AddConfig("ascend950", config950);
    }
};

OP_ADD(QkvRmsNormRopeCacheWithKScale);
} // namespace ops
