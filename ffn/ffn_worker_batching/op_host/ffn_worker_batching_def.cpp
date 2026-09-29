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
 * \file ffn_worker_batching_def.cpp
 * \brief FfnWorkerBatching def
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class FfnWorkerBatching : public OpDef {
public:
  explicit FfnWorkerBatching(const char *name) : OpDef(name) {
    this->Input("schedule_context")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("group_list")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("session_ids")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("micro_batch_ids")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("token_ids")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("expert_offsets")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("dynamic_scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("actual_token_num")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("expert_num").AttrType(REQUIRED).Int();
    this->Attr("max_out_shape").AttrType(REQUIRED).ListInt();
    this->Attr("token_dtype").AttrType(OPTIONAL).Int(0); // 0: FP16 1: BF16 2: dynamic_quant_int8
    this->Attr("need_schedule").AttrType(OPTIONAL).Int(0); //0: only batching; 1: recv and batching
    this->Attr("layer_num").AttrType(OPTIONAL).Int(0);

    this->AICore()
        .AddConfig("ascend910_93")
        .AddConfig("ascend910b");
  }
};
OP_ADD(FfnWorkerBatching);
} // namespace ops
