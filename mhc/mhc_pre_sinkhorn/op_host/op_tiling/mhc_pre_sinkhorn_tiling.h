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
 * \file mhc_pre_sinkhorn_tiling.h
 * \brief MhcPreSinkhorn tiling
 */

#ifndef MHC_PRE_SINKHORN_TILING_H
#define MHC_PRE_SINKHORN_TILING_H

#include <vector>
#include <iostream>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"

namespace optiling {

struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

BEGIN_TILING_DATA_DEF(MhcPreSinkhornTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);                         // batch size
TILING_DATA_FIELD_DEF(int64_t, hcMix);                      // hcMult * hcMult + 2 * hcMult
TILING_DATA_FIELD_DEF(int64_t, hcMult);                     // n维度大小
TILING_DATA_FIELD_DEF(int64_t, d);                          // 特征维度大小
TILING_DATA_FIELD_DEF(int64_t, hcMultAlign);                // 对齐后的hcMult
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);           // 前核处理BS总数
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);             // 尾核处理BS总数
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);       // 前核处理行循环次数
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);         // 尾核处理行循环次数
TILING_DATA_FIELD_DEF(int64_t, rowFactor);                  // 每次处理的行数
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock); // 前核尾次处理的行数
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);   // 尾核尾次处理的行数
TILING_DATA_FIELD_DEF(int64_t, dLoop);                      // d维度循环次数
TILING_DATA_FIELD_DEF(int64_t, dFactor);                    // 每次处理的d维度大小
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);                // 尾次处理的d维度大小
TILING_DATA_FIELD_DEF(int64_t, iterTimes);                  // Sinkhorn迭代轮数
TILING_DATA_FIELD_DEF(float, hcEps);                        // Sinkhorn算法的epsilon参数
TILING_DATA_FIELD_DEF(float, normEps);                      // 归一化的epsilon参数
TILING_DATA_FIELD_DEF(int64_t, kBlockFactor);               // K维度块因子
TILING_DATA_FIELD_DEF(int64_t, kFactor);                    // K维度每次处理大小
TILING_DATA_FIELD_DEF(int64_t, tailKFactor);                // K维度尾次处理大小
TILING_DATA_FIELD_DEF(int64_t, kLoop);                      // K维度循环次数
TILING_DATA_FIELD_DEF(int64_t, stage2RowFactor);            // 阶段2每次处理的行数
// 预留参数，切K优化使用
TILING_DATA_FIELD_DEF(int64_t, k);                      // K维度总大小
TILING_DATA_FIELD_DEF(int64_t, kLoopOfFormerBlock);     // 前核K维度循环次数
TILING_DATA_FIELD_DEF(int64_t, kLoopOfTailBlock);       // 尾核K维度循环次数
TILING_DATA_FIELD_DEF(int64_t, stage1KFactor);          // 阶段1 K维度每次处理大小
TILING_DATA_FIELD_DEF(int64_t, kFactorOfFormerBlock);   // 前核K维度每次处理大小
TILING_DATA_FIELD_DEF(int64_t, kL1Size);                // K维度L1缓冲大小
TILING_DATA_FIELD_DEF(int64_t, mL1Size);                // M维度L1缓冲大小
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDimM);          // Cube核M维度切分数量
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDimK);          // Cube核K维度切分数量
TILING_DATA_FIELD_DEF(int64_t, kUbSize);                // K维度UB缓冲大小
TILING_DATA_FIELD_DEF(int64_t, cvLoopKSize);            // CV循环K维度大小
TILING_DATA_FIELD_DEF(int64_t, multCoreSplitKSize);     // 多核切分K维度大小
TILING_DATA_FIELD_DEF(int64_t, multCoreSplitMSize);     // 多核切分M维度大小
TILING_DATA_FIELD_DEF(int64_t, tailKSizeOfFormerBlock); // 前核K维度尾次大小
TILING_DATA_FIELD_DEF(int64_t, tailKSizeOfTailBlock);   // 尾核K维度尾次大小

TILING_DATA_FIELD_DEF(int64_t, mLoopOfFormerBlock);     // 前核M维度循环次数
TILING_DATA_FIELD_DEF(int64_t, mLoopOfTailBlock);       // 尾核M维度循环次数
TILING_DATA_FIELD_DEF(int64_t, formerMSize);            // 前核M维度大小
TILING_DATA_FIELD_DEF(int64_t, tailMSizeOfFormerBlock); // 前核M维度尾次大小
TILING_DATA_FIELD_DEF(int64_t, tailMSizeOfTailBlock);   // 尾核M维度尾次大小

TILING_DATA_FIELD_DEF(int64_t, firstUsedCoreNum);  // 阶段1使用的核心数
TILING_DATA_FIELD_DEF(int64_t, secondUsedCoreNum); // 阶段2使用的核心数

TILING_DATA_FIELD_DEF(int64_t, rowInnerFactor); // 行内部因子

TILING_DATA_FIELD_DEF(int64_t, cubeCoreNum);   // Cube核心数
TILING_DATA_FIELD_DEF(int64_t, stage1MFactor); // 阶段1 M维度每次处理大小

TILING_DATA_FIELD_DEF(int64_t, bufferPool0Size); // 缓冲池0大小
TILING_DATA_FIELD_DEF(int64_t, bufferPool1Size); // 缓冲池1大小
TILING_DATA_FIELD_DEF(int64_t, mUbSize);         // M维度UB缓冲大小

TILING_DATA_FIELD_DEF(int64_t, needGrad);         // 是否需要梯度
TILING_DATA_FIELD_DEF(int64_t, bsSplitThreshold); // BS切分阈值
TILING_DATA_FIELD_DEF(int64_t, bsLoop);           // BS循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBs);           // 尾批次BS数
TILING_DATA_FIELD_DEF(int64_t, curBsSplit);       // 当前BS切分大小

// 预留参数，切K优化情况 尾批次专用参数 (tailBs != BS_SPLIT_THRESHOLD 时使用)
TILING_DATA_FIELD_DEF(int64_t, tailBsRowOfFormerBlock);           // 尾批次前核处理BS总数
TILING_DATA_FIELD_DEF(int64_t, tailBsRowOfTailBlock);             // 尾批次尾核处理BS总数
TILING_DATA_FIELD_DEF(int64_t, tailBsRowLoopOfFormerBlock);       // 尾批次前核处理行循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBsRowLoopOfTailBlock);         // 尾批次尾核处理行循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBsUsedCoreNum);                // 尾批次使用的核心数
TILING_DATA_FIELD_DEF(int64_t, tailBsRowFactor);                  // 尾批次每次处理的行数
TILING_DATA_FIELD_DEF(int64_t, tailBsTailRowFactorOfFormerBlock); // 尾批次前核尾次处理的行数
TILING_DATA_FIELD_DEF(int64_t, tailBsTailRowFactorOfTailBlock);   // 尾批次尾核尾次处理的行数
TILING_DATA_FIELD_DEF(int64_t, tailBsML1Size);                    // 尾批次M维度L1缓冲大小
TILING_DATA_FIELD_DEF(int64_t, tailBsKL1Size);                    // 尾批次K维度L1缓冲大小
TILING_DATA_FIELD_DEF(int64_t, tailBsMultCoreSplitMSize);         // 尾批次多核切分M维度大小
TILING_DATA_FIELD_DEF(int64_t, tailBsCubeBlockDimM);              // 尾批次Cube核M维度切分数量

// 阶段1专用 Tiling 字段
TILING_DATA_FIELD_DEF(int64_t, stage1VecCoreNum);       // 阶段1使用的 Vector Core 数量
TILING_DATA_FIELD_DEF(int64_t, stage1CubeCoreNum);      // 阶段1使用的 Cube Core 数量
TILING_DATA_FIELD_DEF(int64_t, stage1BsPerVecCore);     // 每个 Vector Core 处理的 BS 数
TILING_DATA_FIELD_DEF(int64_t, stage1TailBsPerVecCore); // 尾核处理的 BS 数
TILING_DATA_FIELD_DEF(int64_t, stage1BsLoop);           // BS 循环轮数
TILING_DATA_FIELD_DEF(int64_t, stage1BsFactor);         // 每轮 BS 数
TILING_DATA_FIELD_DEF(int64_t, stage1NcLoop);           // n*c 切分段数
TILING_DATA_FIELD_DEF(int64_t, stage1NcFactor);         // 每段大小
TILING_DATA_FIELD_DEF(int64_t, stage1TailNcFactor);     // 尾段大小
TILING_DATA_FIELD_DEF(int64_t, stage1XCastWsSize);      // X_cast workspace 大小
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPreSinkhorn, MhcPreSinkhornTilingData)

// Regbase TilingData
BEGIN_TILING_DATA_DEF(MhcPreSinkhornRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, hcMix);
TILING_DATA_FIELD_DEF(int64_t, hcMult);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, hcMultAlign);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);
TILING_DATA_FIELD_DEF(int64_t, iterTimes);
TILING_DATA_FIELD_DEF(float, hcEps);
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(int64_t, kBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, stage2RowFactor);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, kL1Size);
TILING_DATA_FIELD_DEF(int64_t, mL1Size);
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(int64_t, kUbSize);
TILING_DATA_FIELD_DEF(int64_t, multCoreSplitKSize);
TILING_DATA_FIELD_DEF(int64_t, secondUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, rowInnerFactor);
TILING_DATA_FIELD_DEF(int64_t, bufferPool0Size);
TILING_DATA_FIELD_DEF(int64_t, bufferPool1Size);
TILING_DATA_FIELD_DEF(int64_t, mUbSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPreSinkhorn_1000, MhcPreSinkhornRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MhcPreSinkhorn_1001, MhcPreSinkhornRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MhcPreSinkhorn_2000, MhcPreSinkhornRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MhcPreSinkhorn_2001, MhcPreSinkhornRegbaseTilingData)

struct MhcPreSinkhornCompileInfo {
    uint64_t coreNum = 0;
    uint64_t aicCoreNum = 0;
    uint64_t ubSize = 0;
    uint64_t sysWorkspaceSize = 0;
};

class MhcPreSinkhornTiling {
public:
    explicit MhcPreSinkhornTiling(gert::TilingContext *tilingContext)
        : context_(tilingContext)
    {}
    ~MhcPreSinkhornTiling() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    ge::graphStatus CalcMembaseOpTiling();
    ge::graphStatus CalcMKSplitCoreMembasePart2Tiling();
    ge::graphStatus CalcBsSplit();
    ge::graphStatus CalcTailBsTiling();
    ge::graphStatus CalcStage1Tiling();

private:
    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    MhcPreSinkhornTilingData tilingData_;
    uint64_t aivCoreNum_ = 0;
    uint64_t aicCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedAivCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t hcMix_ = 0;
    int64_t hcMult_ = 0;
    int64_t d_ = 0;
    int64_t hcMultAlign_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_ = 0;
    int64_t dLoop_ = 0;
    int64_t dFactor_ = 0;
    int64_t tailDFactor_ = 0;
    int64_t iterTimes_ = 0;
    float hcEps_ = 0.0;
    float normEps_ = 0.0;
    bool needGrad_ = true;
    int64_t bsLoop_ = 1;
    int64_t tailBs_ = 0;
    int64_t curBsSplit_ = 0;
    // 尾批次专用参数
    int64_t tailBsRowOfFormerBlock_ = 0;
    int64_t tailBsRowOfTailBlock_ = 0;
    int64_t tailBsRowLoopOfFormerBlock_ = 0;
    int64_t tailBsRowLoopOfTailBlock_ = 0;
    int64_t tailBsUsedCoreNum_ = 0;
    int64_t tailBsRowFactor_ = 0;
    int64_t tailBsTailRowFactorOfFormerBlock_ = 0;
    int64_t tailBsTailRowFactorOfTailBlock_ = 0;
    int64_t tailBsML1Size_ = 0;
    int64_t tailBsKL1Size_ = 0;
    int64_t tailBsMultCoreSplitMSize_ = 0;
    int64_t tailBsCubeBlockDimM_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

class MhcPreSinkhornTilingRegbase {
public:
    explicit MhcPreSinkhornTilingRegbase(gert::TilingContext *tilingContext);
    ~MhcPreSinkhornTilingRegbase() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    ge::graphStatus CalcRegbaseOpTiling();
    ge::graphStatus CalcMKSplitCorePart2Tiling();
    ge::graphStatus CalcRegbaseOpGradoutTiling();
    ge::graphStatus CalcMKSplitCorePart2GradoutTiling();

private:
    enum class GradOutMode {
        DISABLED,
        ENABLED,
    };

    enum class SplitMode {
        M_SPLIT,
        K_SPLIT,
    };

    ge::graphStatus CalcRegbaseCommonTiling(GradOutMode gradOutMode, SplitMode splitMode);

    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    MhcPreSinkhornRegbaseTilingData tilingData_;
    uint64_t aivCoreNum_ = 0;
    uint64_t aicCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t sysWorkspaceSize_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t hcMix_ = 0;
    int64_t hcMult_ = 0;
    int64_t d_ = 0;
    int64_t iterTimes_ = 0;
    double hcEps_ = 0.0;
    double normEps_ = 0.0;
    bool needGrad_ = true;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

} // namespace optiling
#endif // MHC_PRE_SINKHORN_TILING_H
