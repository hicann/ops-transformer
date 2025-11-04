/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_buffer_reset_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MoeDistributeBufferReset : public OpDef {
 public:
  explicit MoeDistributeBufferReset(const char *name) : OpDef(name) {
    this->Input("elastic_info")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("group_ep").AttrType(REQUIRED).String();
    this->Attr("ep_world_size").AttrType(REQUIRED).Int();
    this->Attr("need_sync").AttrType(REQUIRED).Int();

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")  // 动态shape,复用二进制,后续图支持后修改
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
	  this->AICore().AddConfig("ascend910_93", aicore_config);
    this->MC2().HcclGroup("group_ep");
  }
};

OP_ADD(MoeDistributeBufferReset);
}  // namespace ops
