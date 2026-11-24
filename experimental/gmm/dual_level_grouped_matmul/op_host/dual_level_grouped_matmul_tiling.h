// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef DUAL_LEVEL_GROUPED_MATMUL_TILING_H_
#define DUAL_LEVEL_GROUPED_MATMUL_TILING_H_
#include "register/op_impl_registry.h"
#include "../op_kernel/dual_level_grouped_matmul_tiling_data.h"
namespace optiling {
struct DualLevelInputIndex {
    enum : size_t {
        X,
        X_LEVEL0_SCALE,
        X_LEVEL1_SCALE,
        WEIGHT,
        WEIGHT_LEVEL0_SCALE,
        WEIGHT_LEVEL1_SCALE,
        GROUP_LIST,
        BIAS
    };
};
struct DualLevelAttrIndex {
    enum : size_t {
        TRANSPOSE_WEIGHT,
        GROUP_LIST_TYPE,
        TUNING_CONFIG,
        OUTPUT_DTYPE,
        LEVEL0_GROUP_SIZE,
        LEVEL1_GROUP_SIZE
    };
};
inline constexpr size_t MATRIX_RANK = 2;
inline constexpr size_t EXPERT_WEIGHT_RANK = 3;
struct DualLevelGroupedMatmulCompileInfo {};
class DualLevelGroupedMatmulTiling {
public:
    explicit DualLevelGroupedMatmulTiling(gert::TilingContext* context)
        : context_(context)
    {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckAttributes();
    ge::graphStatus GetShapeAttrs();
    ge::graphStatus CheckAddressBounds();
    ge::graphStatus CheckInputShapes();
    ge::graphStatus CheckOutputShape();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus CalBasicBlock();
    ge::graphStatus SaveTilingData();
    gert::TilingContext* context_;
    DualLevelGroupedMatmul::DualLevelGroupedMatmulTilingData tilingData_{};
    bool transposeWeight_ = false;
    int64_t groupListType_ = 0;
    int64_t outputDtype_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t libApiWorkspaceSize_ = 0;
};
} // namespace optiling
#endif
