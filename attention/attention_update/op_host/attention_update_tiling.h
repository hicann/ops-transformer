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
 * \file attention_update_tiling.h
 * \brief
 */
#ifndef ASCEND_OPS_ATTENTION_UPDATE_TILING_H
#define ASCEND_OPS_ATTENTION_UPDATE_TILING_H
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "decode_update_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AttentionUpdateTilingData)
TILING_DATA_FIELD_DEF(uint64_t, sp);          // lse和go的tensor个数
TILING_DATA_FIELD_DEF(uint64_t, d);           // go的尾轴d的大小
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum); // 使用核数

TILING_DATA_FIELD_DEF(uint64_t, perCoreCount);  // 整核处理的bsh数
TILING_DATA_FIELD_DEF(uint64_t, lastCoreCount); // 尾核处理的bsh数
TILING_DATA_FIELD_DEF(uint64_t, perCoreLoops);  // 整核的loop数
TILING_DATA_FIELD_DEF(uint64_t, lastCoreLoops); // 尾核的loop数

TILING_DATA_FIELD_DEF(uint64_t, perCorePerLoopCount);   // 核内一次最多可以处理的bsh数
TILING_DATA_FIELD_DEF(uint64_t, perCoreLastLoopCount);  // 整核内最后一次处理的bsh数
TILING_DATA_FIELD_DEF(uint64_t, lastCoreLastLoopCount); // 尾核内最后一次处理的bsh数
TILING_DATA_FIELD_DEF(uint64_t, bshInLoop);             // 计算阶段内层循环一次最多可以处理的bsh数

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionUpdate_20000, AttentionUpdateTilingData)
REGISTER_TILING_DATA_CLASS(AttentionUpdate_20001, AttentionUpdateTilingData)

class AttentionUpdateTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AttentionUpdateTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputDim();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckOutputParams();

private:
    // 硬件信息
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t ubBlockSize_ = 0;

    // 输入参数信息
    uint64_t sp_ = 0;
    uint64_t d_ = 0;
    uint64_t updateType_ = 0;
    uint64_t bshSize_ = 0;
    uint64_t goDtypeSize_ = 0;

    // 分核信息，核个数
    uint64_t usedCoreNum_ = 0;

    // 分核信息，核长度
    uint64_t perCoreCount_ = 0;
    uint64_t lastCoreCount_ = 0;

    // 内循环信息
    uint64_t perCoreLoops_ = 0;
    uint64_t lastCoreLoops_ = 0;
    uint64_t perCorePerLoopCount_ = 0;
    uint64_t perCoreLastLoopCount_ = 0;
    uint64_t lastCorePerLoopCount_ = 0;
    uint64_t lastCoreLastLoopCount_ = 0;
    uint64_t bshInLoop_ = 0;

    gert::Shape goShape_;
    gert::Shape lseShape_;
    ge::DataType goType_;
    ge::DataType lseType_;
    AttentionUpdateTilingData tilingData_;
};

} // namespace optiling
#endif // ASCEND_OPS_ATTENTION_UPDATE_TILING_H
