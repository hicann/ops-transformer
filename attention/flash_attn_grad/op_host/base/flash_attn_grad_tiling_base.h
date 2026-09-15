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
 * \file flash_attn_grad_tiling_base.h
 * \brief tiling 模板类的三步骨架与写回封装。
 *
 * D1：采用 attention/common/op_host/
 * fia_tiling_base.h 里 FiaTilingBase 的**形态**，但在本算子目录下**自持一份**，
 * 不 include 正向那份 —— 理由与 D3 相同，正向改骨架不应牵动反向。
 *
 * 与 FiaTilingBase 的差异，都是 FAGrad 的实际需要：
 *   - `DoTiling` 收 `const FagParsedInfo &` 而不是空基类 `TilingInfo *`。
 *     解析结果是本算子自己的 POD，没必要为了对齐一个空基类再做向下转型。
 *   - 没有 `SetTilingData(TilingDef&)`。FAGrad 写的是 pypto codegen 出来的
 *     结构体（`context->GetTilingData<FlashAttnGradTilingData>()`），不是
 *     `TILING_DATA_FIELD_DEF` 那套 `TilingDef`，所以写 tilingdata 留给子类，
 *     基类只封装 blockDim / tilingkey / workspace / scheduleMode 这四个。
 *   - 去掉 `GetShapeDebugStr` / `GetTilingContextDebugStr` 等调试串。FAGrad 的
 *     决策追溯走 FagRouteTrace + 一行 OP_LOGI，不需要再拼 shape 字符串。
 *
 * 平台与解析已经上移到入口的 ParsePlatform / ParseFlashAttnGradInfo
 * （parser 独占 context），模板类只剩「能不能接 + 怎么算」三步。
 */

#ifndef FLASH_ATTN_GRAD_TILING_BASE_H_
#define FLASH_ATTN_GRAD_TILING_BASE_H_

#include "exe_graph/runtime/tiling_context.h"

#include "../info/flash_attn_grad_tiling_info.h"
#include "log/log.h"

namespace optiling {

// 与 FiaTilingBase::ScheduleMode 同义，自持一份。取值由 runtime 定义，不可改。
enum class FagScheduleMode : uint32_t {
    NORMAL_MODE = 0,
    BATCH_MODE = 1,
    SYNC_MODE = 2,
};

class FagTilingBase {
public:
    explicit FagTilingBase(gert::TilingContext *context)
        : context_(context)
    {}
    virtual ~FagTilingBase() = default;

    FagTilingBase(const FagTilingBase &) = delete;
    FagTilingBase &operator=(const FagTilingBase &) = delete;

    // 返回值语义与 FiaTilingBase::DoTiling 一致，注册表就靠它表达 fallback：
    //   GRAPH_SUCCESS       成功，不必再往下试
    //   GRAPH_PARAM_INVALID 本类不支持这组输入，应继续试下一个模板类
    //   GRAPH_FAILED        真失败，中止整个 tiling
    ge::graphStatus DoTiling(const FagParsedInfo &info)
    {
        InitTilingInfo(info);
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        return DoOpTiling();
    }

protected:
    virtual void InitTilingInfo(const FagParsedInfo &info) = 0;
    virtual bool IsCapable() = 0;
    virtual ge::graphStatus DoOpTiling() = 0;

    // ---- 写回封装：context 的写操作集中在这里，子类不直接碰 context ----
    ge::graphStatus SetNumBlocks(uint32_t numBlocks) const
    {
        context_->SetBlockDim(numBlocks);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetTilingKey(uint64_t tilingKey) const
    {
        context_->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetWorkspaceSize(size_t workspaceSize) const
    {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspace sizes got from ge is nullptr."),
                    return ge::GRAPH_FAILED);
        workspaces[0] = workspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetScheduleMode(FagScheduleMode mode) const
    {
        return context_->SetScheduleMode(static_cast<uint32_t>(mode));
    }

    gert::TilingContext *context_ = nullptr;
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_BASE_H_
