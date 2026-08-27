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
 * \file fused_causal_conv1d_cut_bsh_tiling_arch35.h
 * \brief
 */
#ifndef FUSED_CAUSAL_CONV1D_CUT_BSH_TILING_H
#define FUSED_CAUSAL_CONV1D_CUT_BSH_TILING_H
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "attention/fused_causal_conv1d/op_kernel/arch35/fused_causal_conv1d_cut_bsh_struct.h"
#include "platform/platform_ascendc.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

class FusedCausalConv1dCutBSHTiling : public TilingBaseClass {
public:
    explicit FusedCausalConv1dCutBSHTiling(gert::TilingContext *context, bool isInplace = false)
        : TilingBaseClass(context),
          isInplace_(isInplace)
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
    ge::graphStatus CheckXDim();
    ge::graphStatus CheckWeightDim();
    ge::graphStatus CheckCacheStatesDim();
    ge::graphStatus CheckCacheIndicesDim();
    ge::graphStatus CheckAttrs();
    ge::graphStatus CheckIndexDims();
    ge::graphStatus Calculate2DTiling(); // 二维切分的tiling计算
    ge::graphStatus GetInputShapes();
    ge::graphStatus GetInputDtypes();
    ge::graphStatus GetInputStrides();

private:
    // 辅助结构体：切cu_seq_len时的核间切分信息
    struct CuSeqLenSplitInfo {
        uint64_t realCoreNum;
        uint64_t effectiveTotal;
        uint64_t baseLen;
        uint64_t remainder;
        uint64_t blockFactor;
        uint64_t blockTailFactor;
    };

    // 辅助函数：计算切cu_seq_len时实际需要的核数（考虑因果卷积重叠）
    // 同时返回中间计算结果，避免重复计算
    CuSeqLenSplitInfo CalculateCuSeqLenSplitInfo(uint64_t cuSeqLen, uint64_t bsOverlap, uint64_t coreNum) const;
    ge::graphStatus SearchBestCoreSplit(uint64_t N, uint64_t bsOverlap, uint64_t &bestDimCores,
                                        CuSeqLenSplitInfo &bestBSSplitInfo);
    ge::graphStatus ApplyDimSplit(uint64_t N, uint64_t bestDimCores);
    ge::graphStatus CalcCoreUbTiling(uint64_t coreDim, uint64_t coreBS, uint64_t bsBlockFactor, int64_t availableUbSize,
                                     uint64_t weightCacheCoeffPerDim, uint64_t bsOverlap, uint64_t &ubFactorBS,
                                     uint64_t &ubFactorDim, uint64_t &loopNumBS, uint64_t &ubTailFactorBS,
                                     uint64_t &loopNumDim, uint64_t &ubTailFactorDim);

    // 硬件信息
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t ubBlockSize_ = 0;

    // 输入参数信息
    uint64_t cuSeqLen_ = 0;
    uint64_t dim_ = 0;
    uint64_t kernelWidth_ = 0; // K
    uint64_t batch_ = 0;
    uint64_t xDtypeSize_ = 0;
    uint64_t xStride_ = 0;      // x 的 stride
    uint64_t cacheStride0_ = 0; // cacheStates 的 stride[0]
    uint64_t cacheStride1_ = 0; // cacheStates 的 stride[1]

    // padding相关信息
    int64_t padSlotId_ = -1;
    uint64_t residualConnection_ = 0;
    int64_t runMode_ = 0; // 0: prefill  1: decode

    // ===== dim方向核间切分信息 =====
    uint64_t dimCoreNum_ = 0;         // dim方向总核数
    uint64_t dimRemainderCores_ = 0;  // 前多少个核是大核（分配base+1个128-块）
    uint64_t dimBlockFactor_ = 0;     // 大核的dim大小（(base+1) * 128，前dimRemainderCores个核）
    uint64_t dimBlockTailFactor_ = 0; // 小核的dim大小（base * 128，后面的核）

    // ===== BS方向核间切分信息 =====
    uint64_t bsCoreNum_ = 0;         // BS方向切分核数
    uint64_t bsRemainderCores_ = 0;  // BS方向前多少个核是大核（均分策略）
    uint64_t bsBlockFactor_ = 0;     // BS方向大核处理的长度（含overlap，前bsRemainderCores个核）
    uint64_t bsBlockTailFactor_ = 0; // BS方向小核处理的长度（后面的核）

    // ===== 核数信息 =====
    uint64_t realCoreNum_ = 0; // 实际使用核数（dimCoreNum × bsCoreNum）

    // 核内切分信息 - 整核
    uint64_t loopNumBS_ = 0;       // BS方向循环次数
    uint64_t loopNumDim_ = 0;      // Dim方向循环次数
    uint64_t ubFactorBS_ = 0;      // BS方向单次循环载入大小
    uint64_t ubTailFactorBS_ = 0;  // BS方向尾次循环载入大小
    uint64_t ubFactorDim_ = 0;     // Dim方向单次循环载入大小
    uint64_t ubTailFactorDim_ = 0; // Dim方向尾次循环载入大小

    // 核内切分信息 - 尾核
    uint64_t tailBlockloopNumBS_ = 0;
    uint64_t tailBlockloopNumDim_ = 0;
    uint64_t tailBlockubFactorBS_ = 0;
    uint64_t tailBlockubTailFactorBS_ = 0;
    uint64_t tailBlockubFactorDim_ = 0;
    uint64_t tailBlockubTailFactorDim_ = 0;

    // 核内切分信息 - 最后一个 dim 核 (处理 dim % 128 != 0 场景的余数元素)
    // 当 dim % 128 == 0 时，这些字段等同普通尾核参数（兜底保证向后兼容）
    uint64_t dimRemainderElems_ = 0;       // dim % 128
    uint64_t lastCoreloopNumDim_ = 0;      // 最后一个 dim 核 Dim 方向循环次数
    uint64_t lastCoreubFactorDim_ = 0;     // 最后一个 dim 核 UB 主块 dim 大小
    uint64_t lastCoreubTailFactorDim_ = 0; // 最后一个 dim 核 UB 尾块 dim 大小

    gert::Shape xShape_;
    gert::Shape weightShape_;
    gert::Shape cacheStatesShape_;
    gert::Shape seqStartIndexShape_;
    ge::DataType xType_;
    ge::DataType weightType_;

    uint64_t apcEnabled_ = 0; // APC 开关：block_idx_first_scheduled_token 存在即初判开启，缺其余必要条件立即报错
    uint64_t blockSize_ = 0;    // APC block 大小（0 表示未启用）
    uint64_t maxNumBlocks_ = 0; // cache_indices 第二维大小（APC 模式下 > 0）
    uint64_t convMode_ = 0;     // 0=Qwen3-Next, 1=Pangu v2 输出零填充
    uint64_t maxDraftTokens_ = 0;
    uint64_t inplace_ = 0;              // 是否原地更新（y 复用 x 的 GM）（由 isInplace_ 决定）
    bool isInplace_ = false;            // 构造时传入，替代 attr 读取
    uint64_t hasAcceptTokenNum_ = 0;    // 是否提供 num_accepted_tokens（MTP）
    uint64_t hasNumComputedTokens_ = 0; // 是否提供 num_computed_tokens
    uint64_t hasCacheIndices_ = 0;      // 是否提供 cache_indices
    int64_t maxQueryLen_ = -1;          // 路由参考值

    FusedCausalConv1dCutBSHTilingData tilingData_;
};

} // namespace optiling
#endif // FUSED_CAUSAL_CONV1D_CUT_BSH_TILING_H
