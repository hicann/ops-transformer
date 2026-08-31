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
 * \file moe_init_routing_v3_arch35_tiling_def.h
 * \brief
 */

#ifndef __MOE_INIT_ROUTING_V3_ARCH35_TILING_DEF_H__
#define __MOE_INIT_ROUTING_V3_ARCH35_TILING_DEF_H__

// 单核排序阶段
struct MoeV3Arch35VBSComputeTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
    int64_t oneLoopMaxElements{0};
};

// 多核归并排序阶段
struct MoeV3Arch35VMSMiddleComputeTilingData {
    int64_t needCoreNum{0};
};

// 排序输出阶段
struct MoeV3Arch35SortOutComputeTilingData {
    int64_t oneLoopMaxElements{0};
};

// ExpertTokensCount阶段
struct MoeV3Arch35ExpertTokensCountTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t lastCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
};

// 拷出阶段
struct MoeV3Arch35GatherOutComputeTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreIndicesElements{0};
    int64_t lastCoreIndicesElements{0};
    int64_t perCoreIndicesLoops{0};
    int64_t perCorePerLoopIndicesElements{0};
    int64_t perCoreLastLoopIndicesElements{0};
    int64_t lastCoreIndicesLoops{0};
    int64_t lastCorePerLoopIndicesElements{0};
    int64_t lastCoreLastLoopIndicesElements{0};
    int64_t colsLoops{0};
    int64_t perLoopCols{0};
    int64_t lastLoopCols{0};
    int64_t activeNum{0};
    int64_t xCopyInQueueBufferNum{2};
    int64_t useCompactGatherOutDropPad{0};
};

// DropPad模式下的SrcToDst阶段TilingData
struct MoeV3Arch35SrcToDstCapacityComputeTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreRows{0};
    int64_t perCorePerLoopRows{0};
    int64_t perCoreLastLoopRows{0};
    int64_t lastCoreRows{0};
    int64_t lastCorePerLoopRows{0};
    int64_t lastCoreLastLoopRows{0};
    int64_t perCoreLoops{0};
    int64_t lastCoreLoops{0};
    int64_t perLoopCols{0};
    int64_t lastLoopCols{0};
    int64_t colLoops{0};
};

// CountingSort性能模板TilingData
// countingSortMode: 0=未启用, 1=FullLoad(核内全载), 2=CutOrigin(核内4096 chunk切T)
struct MoeV3Arch35CountingSortTilingData {
    int64_t countingSortMode{0};
    int64_t filterNeedCoreNum{0};   // 过滤阶段使用的核数
    int64_t filterPerCoreTokens{0}; // 每核处理的 token 数（按 n 维切）
    int64_t lastCoreTokens{0};      // 尾核 token 数
    int64_t coreEntries{0};         // 每核 flat entries = filterPerCoreTokens * k（上限）
    int64_t expertCountStride{0};   // align8(actualExpertNum)，专家计数 stride
    int64_t filterChunkSize{0};     // CutOrigin chunk 大小（4096），FullLoad 不用
    int64_t csPerLoopCols{0};       // Phase C gather 每轮列数
    int64_t csColsLoops{0};         // Phase C gather 列方向循环数
    int64_t csLastLoopCols{0};      // Phase C gather 尾轮列数
    int64_t maxPerLoopEntries{0};   // CutOrigin 单 chunk 处理上限
    int64_t pairsWsOffset{0}; // CutOrigin 拆分：pairs/expertCount 区在 workspace 中的起始偏移（int32 元素）
    int64_t csAggrEnable{0};      // 聚合搬出开关：0=逐行(现状), 1=按专家k行切批
    int64_t csAggrOutRows{0};     // 搬出聚合 UB 容纳行数 k
    int64_t csAggrOutBufBytes{0}; // 搬出聚合区字节数 = k * colsAligned * sizeof(T)
};

// Arch35用的TilingData
struct MoeInitRoutingV3Arch35TilingData {
    int64_t coreNum{0};
    int64_t n{0};
    int64_t cols{0};
    int64_t k{0};
    int64_t expertStart{0};
    int64_t expertEnd{0};
    int64_t actualExpertNum{0};
    int64_t quantMode{0};
    int64_t rowIdxType{0};
    int64_t useGatherCopy{0};
    int64_t isInputScale{0};
    int64_t isInputOffset{0};
    int64_t expertNum{0};
    int64_t expertTokensNumType{0};
    int64_t expertTokensNumFlag{0};
    int64_t gatherFirstFullload{0};
    int64_t epFullload{0}; // 未使用
    int64_t activeNum{0};
    int64_t dropPadMode{0};
    int64_t smoothType{0};                          // 未使用
    int64_t expertCapacity{0};                      // 新增：DropPad模式下的expert capacity
    int64_t isInputTopkWeight{0};                   // 新增：topk_weight输入是否提供
    int64_t topkWeightOutPerCorePerLoopElements{0}; // 新增：TopkWeightOut每轮处理的索引元素数
    MoeV3Arch35VBSComputeTilingData vbsComputeParamsOp;
    MoeV3Arch35VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    MoeV3Arch35SortOutComputeTilingData sortOutComputeParamsOp;
    MoeV3Arch35ExpertTokensCountTilingData expertTokensCountTilingDataOp;
    MoeV3Arch35GatherOutComputeTilingData gatherOutComputeParamsOp;
    MoeV3Arch35SrcToDstCapacityComputeTilingData srcToDstDropPadParamsOp; // 新增：DropPad SrcToDst TilingData
    MoeV3Arch35CountingSortTilingData countingSortParamsOp;               // 新增：CountingSort 性能模板
};

#endif // __MOE_INIT_ROUTING_V3_ARCH35_TILING_DEF_H__
