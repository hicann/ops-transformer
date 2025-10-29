
/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _MOE_COMPUTE_EXPERT_TOKENS_TILING_H_
#define _MOE_COMPUTE_EXPERT_TOKENS_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct MoeComputeExpertTokensTilingData {
    int64_t totalCoreNum;                      // 物理总核数
    int64_t usedCoreNumBefore;                 // synAll前实际使用核数
    int64_t usedCoreNumBefore3;                // synAll前模板3实际使用核数
    int64_t usedCoreNumAfter;                  // synAll后实际使用核数
    int64_t ubSize;                            // 总ubsize大小
    int64_t workLocalNeedSize;                 // 计算workLocal大小
    int64_t sortedExpertNum;                   // 输入元素列表长度
    int64_t normalCoreHandleNumBefore;         // 非尾核，每个核处理的元素个数
    int64_t normalCoreLoopNumBefore;           // 非尾核，每个核需要的Loop循环次数
    int64_t normalCoreHandleNumPerLoopBefore;  // 非尾核，每个核，非尾Loop，每次loop需要处理的元素个数
    int64_t normalCoreHandleNumTailLoopBefore; // 非尾核，每个核，尾Loop需要处理的元素个数
    int64_t tailCoreHandleNumBefore;           // 尾核，处理的个数
    int64_t tailCoreLoopNumBefore;             // 尾核，每个核需要的Loop循环次数
    int64_t tailCoreHandleNumPerLoopBefore;    // 尾核，每个核，非尾Loop需要处理的元素个数
    int64_t tailCoreHandleNumTailLoopBefore;   // 尾核，每个核，尾Loop需要处理的元素个数
    int64_t numOfExpert;                       // 输入的专家个数
    int64_t normalCoreHandleNumAfter;          // 非尾核，每个核处理的元素个数
    int64_t normalCoreLoopNumAfter;            // 非尾核，每个核需要的Loop循环次数
    int64_t normalCoreHandleNumPerLoopAfter;   // 非尾核，每个核，非尾Loop，每次loop需要处理的元素个数
    int64_t normalCoreHandleNumTailLoopAfter;  // 非尾核，每个核，尾Loop需要处理的元素个数
    int64_t tailCoreHandleNumAfter;            // 尾核，处理的个数
    int64_t tailCoreLoopNumAfter;              // 尾核，每个核需要的Loop循环次数
    int64_t tailCoreHandleNumPerLoopAfter;     // 尾核，每个核，非尾Loop需要处理的元素个数
    int64_t tailCoreHandleNumTailLoopAfter;    // 尾核，每个核，尾Loop需要处理的元素个数
    int64_t handleNumPerCoreBefore;            // syncall前，模板3，非尾核需要处理的元素个数
    int64_t handleNumTailCoreBefore;           // syncall前，模板3，尾核需要处理的元素个数
    int64_t loopCountBefore;                   // syncall前，模板3，非尾核处理sorted_expert的loop次数
    int64_t loopCountTailBefore;               // syncall前，模板3，尾核处理sorted_expert的loop次数
    int64_t handleNumPerLoopBefore;            // syncall前，模板3，非尾核每次loop处理的sorted_expert数量
    int64_t handleNumTailCorePerLoopBefore;    // syncall前，模板3，尾核每次loop处理的sorted_expert数量
    int64_t handleExpertNumLoopCount;          // syncall前，模板3，切E需要的loop次数
    int64_t handleExpertNumMainCorePerLoop;    // syncall前，模板3，非尾loop切分处理的E的个数
    int64_t handleExpertNumTailCorePerLoop;    // syncall前，模板3，尾loop切分处理的E的个数
    int64_t loopCountTailCoreMainLoop;         // syncall前，模板3，尾核非尾loop切分处理的E的个数
    int64_t handleNumTailCoreMainLoop;         // syncall前，模板3，尾核尾loop切分处理的E的个数
    int64_t loopCountTailCoreTailLoop;         // syncall前，模板3，尾核非尾loop切分处理的E的个数
    int64_t handleNumTailCoreTailLoop;         // syncall前，模板3，尾核尾loop切分处理的E的个数
    int64_t tilingKey;                         // 使用的字段tilingKey
    int64_t userWorkspaceSize;                 // 使用的workspace
};

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                                         \
    MoeComputeExpertTokensTilingData tilingData;                                                           \
    INIT_TILING_DATA(MoeComputeExpertTokensTilingData, tilingDataPointer, tilingPointer);                  \
    (tilingData).totalCoreNum = tilingDataPointer->totalCoreNum;                                           \
    (tilingData).usedCoreNumBefore = tilingDataPointer->usedCoreNumBefore;                                 \
    (tilingData).usedCoreNumBefore3 = tilingDataPointer->usedCoreNumBefore3;                               \
    (tilingData).usedCoreNumAfter = tilingDataPointer->usedCoreNumAfter;                                   \
    (tilingData).ubSize = tilingDataPointer->ubSize;                                                       \
    (tilingData).workLocalNeedSize = tilingDataPointer->workLocalNeedSize;                                 \
    (tilingData).sortedExpertNum = tilingDataPointer->sortedExpertNum;                                     \
    (tilingData).normalCoreHandleNumBefore = tilingDataPointer->normalCoreHandleNumBefore;                 \
    (tilingData).normalCoreLoopNumBefore = tilingDataPointer->normalCoreLoopNumBefore;                     \
    (tilingData).normalCoreHandleNumPerLoopBefore = tilingDataPointer->normalCoreHandleNumPerLoopBefore;   \
    (tilingData).normalCoreHandleNumTailLoopBefore = tilingDataPointer->normalCoreHandleNumTailLoopBefore; \
    (tilingData).tailCoreHandleNumBefore = tilingDataPointer->tailCoreHandleNumBefore;                     \
    (tilingData).tailCoreLoopNumBefore = tilingDataPointer->tailCoreLoopNumBefore;                         \
    (tilingData).tailCoreHandleNumPerLoopBefore = tilingDataPointer->tailCoreHandleNumPerLoopBefore;       \
    (tilingData).tailCoreHandleNumTailLoopBefore = tilingDataPointer->tailCoreHandleNumTailLoopBefore;     \
    (tilingData).numOfExpert = tilingDataPointer->numOfExpert;                                             \
    (tilingData).normalCoreHandleNumAfter = tilingDataPointer->normalCoreHandleNumAfter;                   \
    (tilingData).normalCoreLoopNumAfter = tilingDataPointer->normalCoreLoopNumAfter;                       \
    (tilingData).normalCoreHandleNumPerLoopAfter = tilingDataPointer->normalCoreHandleNumPerLoopAfter;     \
    (tilingData).normalCoreHandleNumTailLoopAfter = tilingDataPointer->normalCoreHandleNumTailLoopAfter;   \
    (tilingData).tailCoreHandleNumAfter = tilingDataPointer->tailCoreHandleNumAfter;                       \
    (tilingData).tailCoreLoopNumAfter = tilingDataPointer->tailCoreLoopNumAfter;                           \
    (tilingData).tailCoreHandleNumPerLoopAfter = tilingDataPointer->tailCoreHandleNumPerLoopAfter;         \
    (tilingData).tailCoreHandleNumTailLoopAfter = tilingDataPointer->tailCoreHandleNumTailLoopAfter;       \
    (tilingData).handleNumPerCoreBefore = tilingDataPointer->handleNumPerCoreBefore;                       \
    (tilingData).handleNumTailCoreBefore = tilingDataPointer->handleNumTailCoreBefore;                     \
    (tilingData).loopCountBefore = tilingDataPointer->loopCountBefore;                                     \
    (tilingData).loopCountTailBefore = tilingDataPointer->loopCountTailBefore;                             \
    (tilingData).handleNumPerLoopBefore = tilingDataPointer->handleNumPerLoopBefore;                       \
    (tilingData).handleNumTailCorePerLoopBefore = tilingDataPointer->handleNumTailCorePerLoopBefore;       \
    (tilingData).handleExpertNumLoopCount = tilingDataPointer->handleExpertNumLoopCount;                   \
    (tilingData).handleExpertNumMainCorePerLoop = tilingDataPointer->handleExpertNumMainCorePerLoop;       \
    (tilingData).handleExpertNumTailCorePerLoop = tilingDataPointer->handleExpertNumTailCorePerLoop;       \
    (tilingData).loopCountTailCoreMainLoop = tilingDataPointer->loopCountTailCoreMainLoop;                 \
    (tilingData).handleNumTailCoreMainLoop = tilingDataPointer->handleNumTailCoreMainLoop;                 \
    (tilingData).loopCountTailCoreTailLoop = tilingDataPointer->loopCountTailCoreTailLoop;                 \
    (tilingData).handleNumTailCoreTailLoop = tilingDataPointer->handleNumTailCoreTailLoop;                 \
    (tilingData).tilingKey = tilingDataPointer->tilingKey;                                                 \
    (tilingData).userWorkspaceSize = tilingDataPointer->userWorkspaceSize;

#endif // _MOE_COMPUTE_EXPERT_TOKENS_TILING_H_
