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
 * \file sparse_flash_attention_grad_tiling_bs1_regbase.h
 * \brief
 */

#pragma once

#include "../sparse_flash_attention_grad_tiling_common.h"
#include "../../op_kernel/arch92/sparse_flash_attention_grad_tiling_data_regbase.h"
#include "../../op_kernel/arch92/sparse_flash_attention_grad_template_tiling_key.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_type.h"

namespace optiling {
namespace sfag {
// 嵌套 arch92 命名空间：TempParams/AiCoreParams 以及 tiling 类名都与 arch35 同名，
// 而两者的 .cpp 会编进同一个 tiling_obj。REGISTER_TILING_TEMPLATE_WITH_ARCH 展开出的
// op_impl_register_template_<op>_<class><priority> 是外部链接的全局变量，同名同优先级
// 会直接重复符号；命名空间隔开后既能保持命名一致，也不用给结构体加后缀。
namespace arch92 {
struct TempParams {
    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;
    uint32_t dataTypeSize;
    uint32_t queryType;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t d1;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t dSinkWorkSpaceLen; // dSink 跨核 reduce 中间 workspace 长度（host 侧，不入 TilingData；仅
                               // deter+HAS_SINKS 非零）
    uint32_t layout;
    uint32_t selected_block_count;
    uint32_t selected_block_size = 1;
    bool attenEnable = false;
    uint32_t singleM;
    uint32_t singleN;
    int64_t ropeDim;
    bool ropeEnable = false;
    bool deterministic = false;
    bool kvMerge = false;
    bool hasSinks = false; // sinks 输入是否存在（编入 tiling key bit35 HasSinks）
};

struct AiCoreParams {
    uint64_t ubSize;
    uint64_t blockDim;
    uint64_t aicNum;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};

class SparseFlashAttentionGradBs1Regbase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit SparseFlashAttentionGradBs1Regbase(gert::TilingContext* context)
        : TilingBaseClass(context) {};
    SparseFlashAttentionGradTilingDataRegbase92* tilingData =
        context_->GetTilingData<SparseFlashAttentionGradTilingDataRegbase92>();
    SparseFlashAttentionGradBaseParamsRegbase92* baseParams_ = &tilingData->baseParams;
    PreParamsRegbase92* preTilingData_ = &tilingData->preTilingData;
    PostParamsRegbase* postTilingData_ = &tilingData->postTilingData;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus CheckOutShapeInfo(const gert::Shape& inputshape, const char* inputName,
                                      const gert::Shape& outputshape, const char* inputLayout);
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus DoSftTiling();
    ge::graphStatus DoBlockTiling();
    ge::graphStatus DoCastTiling();
    TempParams tmpData;

    const char* opName;
};
} // namespace arch92
} // namespace sfag
} // namespace optiling
