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
 * \file stem_oam_prep_paged_kv_def.cpp
 * \brief StemOamPrepPagedKV OpDef registration
 */
#include "register/op_def_registry.h"

namespace ops {

constexpr double DEFAULT_LAMBDA_MAG = 0.3;
constexpr int64_t DEFAULT_STEM_BLOCK_SIZE = 128;
constexpr int64_t DEFAULT_STEM_STRIDE = 16;
const char *DEFAULT_KV_LAYOUT = "BNBD";

class StemOamPrepPagedKv : public OpDef {
public:
    explicit StemOamPrepPagedKv(const char *name) : OpDef(name)
    {
        this->Input("kCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND});
        this->Input("vCache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND});
        this->Input("kvIndices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("kvSeqLens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL);
        this->Input("kScaleCache")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("vScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("kFlat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Output("vBias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("lambdaMag").AttrType(OPTIONAL).Float(DEFAULT_LAMBDA_MAG);
        this->Attr("kvLayout").AttrType(OPTIONAL).String(DEFAULT_KV_LAYOUT);
        this->Attr("stemBlockSize").AttrType(OPTIONAL).Int(DEFAULT_STEM_BLOCK_SIZE);
        this->Attr("stemStride").AttrType(OPTIONAL).Int(DEFAULT_STEM_STRIDE);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "stem_oam_prep_paged_kv_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(StemOamPrepPagedKv);

} // namespace ops
