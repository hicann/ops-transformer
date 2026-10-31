/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_H
#define MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_H

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_data_arch35.h"

namespace optiling {
class MhcPreSinkhornBackwardArch35Tiling final : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPreSinkhornBackwardArch35Tiling(gert::TilingContext* context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    uint64_t AlignWorkspace(uint64_t value) const;
    bool CheckTensorShape(uint32_t index, const char* name, const int64_t* dims, size_t rank,
                          bool logError = true) const;
    bool CheckOutputShape(uint32_t index, const char* name, const int64_t* dims, size_t rank,
                          bool logError = true) const;
    ge::graphStatus CheckDataTypes() const;
    ge::graphStatus SetMmConfig();
    struct MkSplitResult {
        int64_t mBlockCount = 0;
        int64_t kCoreCount = 0;
        int64_t mBlockRows = 0;
        int64_t kBlockSize = 0;
        int64_t kBlockCount = 0;
    };
    void ComputeMkSplit(int64_t bs, int64_t k, bool empty, bool deterministic, MkSplitResult& result) const;

    MhcPreSinkhornBackwardArch35TilingData* tilingData_ = nullptr;
    uint64_t aicNum_ = 0;
    uint64_t aivNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t l0ASize_ = 0;
    uint64_t l0BSize_ = 0;
    uint64_t l0CSize_ = 0;
    uint64_t systemWorkspaceBytes_ = 0;
    uint64_t workspaceBytes_ = 0;
    uint64_t tilingKey_ = 0;
    int64_t xRank_ = 0;
};
} // namespace optiling

#endif // MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_H
