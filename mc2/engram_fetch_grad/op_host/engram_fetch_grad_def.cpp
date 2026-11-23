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
 * \file engram_fetch_grad_def.cpp
 * \brief 算子信息库定义 — EngramFetchGrad
 */

#include "register/op_def_registry.h"

namespace ops {
class EngramFetchGrad : public OpDef {
public:
    explicit EngramFetchGrad(const char *name) : OpDef(name)
    {
        this->Input("comm_context").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Input("grad_fetched")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Input("perm").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Input("send_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Input("recv_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Input("recv_local_entry").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Input("num_recv").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Output("grad_unique_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Output("unique_local_entry_out")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Output("num_unique_out").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Attr("num_entries_per_rank").AttrType(REQUIRED).Int();
        this->Attr("comm_buffer_size").AttrType(REQUIRED).Int();
        // 轨道模型（rail-aligned，204800 适配）——感知 attr：torch 侧经 HCCL topo 感知
        // （超节点直连层）自动填入（用户零参数）。ranksPerSupernode=每超节点卡数是唯一
        // 拓扑事实，超节点数=world÷ranksPerSupernode、轨道对端数=world÷ranksPerSupernode−1
        // 全推导。缺省 0 = 未感知（tiling 分发按 128 兼容边界处理，rail tiling 校验要求 >0）。
        // 与 ctx 尾部字段 ranksPerSupernode 同源。
        this->Attr("ranks_per_supernode").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config_a5;
        aicore_config_a5.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend950", aicore_config_a5);
    }
};
OP_ADD(EngramFetchGrad);
} // namespace ops
