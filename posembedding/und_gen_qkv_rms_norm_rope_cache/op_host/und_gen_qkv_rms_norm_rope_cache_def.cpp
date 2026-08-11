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
 * \file und_gen_qkv_rms_norm_rope_cache_def.cpp
 * \brief UndGenQkvRmsNormRopeCache op host config
 */
#include "register/op_def_registry.h"

namespace ops {
class UndGenQkvRmsNormRopeCache : public OpDef {
public:
    explicit UndGenQkvRmsNormRopeCache(const char* name) : OpDef(name)
    {
        // ---------------- 必选输入（8 个）----------------
        this->Input("und_qkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("und_weights_q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("und_weights_k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cos_sin_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ---------------- 可选输入（4 个）----------------
        // NOTE: 当前实现要求这 4 个输入必须全部提供，缺省场景由 tiling 的 CheckSupportRange 拦截；
        //       这里保留 OPTIONAL 声明以便后续放开退化路径。
        this->Input("gen_qkv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gen_weights_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gen_weights_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cat_indices")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ---------------- 输出（3 个，k_cache/v_cache 与输入同地址原地写入）----------------
        this->Output("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("k_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("v_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // ---------------- 属性（5 个）----------------
        this->Attr("num_heads_q").AttrType(REQUIRED).Int();
        this->Attr("num_heads_k").AttrType(REQUIRED).Int();
        this->Attr("num_heads_v").AttrType(REQUIRED).Int();
        this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("mrope_section").AttrType(OPTIONAL).ListInt({});

        // 仅支持 Ascend 950（DAV_3510, arch35）
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true).DynamicShapeSupportFlag(true).DynamicRankSupportFlag(true);
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};

OP_ADD(UndGenQkvRmsNormRopeCache);
} // namespace ops
