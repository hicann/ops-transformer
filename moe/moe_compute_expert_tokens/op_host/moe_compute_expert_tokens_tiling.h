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
 * \file moe_compute_expert_tokens_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_COMPUTE_EXPERT_TOKENS_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_COMPUTE_EXPERT_TOKENS_H_
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

const int64_t COM_SCENE_FLAG_1 = 1001;
const int64_t COM_SCENE_FLAG_2 = 1002;
const int64_t COM_SCENE_FLAG_3 = 1003;

BEGIN_TILING_DATA_DEF(MoeComputeExpertTokensTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);             // 物理总核数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumBefore);        // synAll前模板2实际使用核数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumBefore3);       // synAll前模板3实际使用核数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumAfter);         // synAll后实际使用核数
TILING_DATA_FIELD_DEF(int64_t, ubSize);                   // 总ubsize大小
TILING_DATA_FIELD_DEF(int64_t, workLocalNeedSize);        // 计算workLocal大小
TILING_DATA_FIELD_DEF(int64_t, sortedExpertNum);          // 输入元素列表长度
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNumAfter); // syncall后，非尾核，每个核处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, normalCoreLoopNumAfter);   // syncall后，非尾核，每个核需要的Loop循环次数
TILING_DATA_FIELD_DEF(
    int64_t, normalCoreHandleNumPerLoopAfter); // syncall后，非尾核，每个核，非尾Loop，每次loop需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNumTailLoopAfter); // syncall后，非尾核，每个核，尾Loop需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumAfter); // syncall后，尾核，处理的个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoopNumAfter);   // syncall后，尾核，每个核需要的Loop循环次数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumPerLoopAfter); // syncall后，尾核，每个核，非尾Loop需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumTailLoopAfter); // syncall后，尾核，每个核，尾Loop需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, numOfExpert);                    // 输入的专家个数E
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNumBefore); // syncall前，非尾核，每个核处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, normalCoreLoopNumBefore); // syncall前，非尾核，每个核需要的Loop循环次数
TILING_DATA_FIELD_DEF(
    int64_t, normalCoreHandleNumPerLoopBefore); // syncall前，非尾核，每个核，非尾Loop，每次loop需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t,
                      normalCoreHandleNumTailLoopBefore); // syncall前，非尾核，每个核，尾Loop需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumBefore); // syncall前，尾核，处理的个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoopNumBefore);   // syncall前，尾核，每个核需要的Loop循环次数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumPerLoopBefore); // syncall前，尾核，每个核，非尾Loop需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumTailLoopBefore); // syncall前，尾核，每个核，尾Loop需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, handleNumPerCoreBefore);  // syncall前，模板3，非尾核需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, handleNumTailCoreBefore); // syncall前，模板3，尾核需要处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, loopCountBefore);     // syncall前，模板3，非尾核处理sorted_expert的loop次数
TILING_DATA_FIELD_DEF(int64_t, loopCountTailBefore); // syncall前，模板3，尾核处理sorted_expert的loop次数
TILING_DATA_FIELD_DEF(int64_t, handleNumPerLoopBefore); // syncall前，模板3，非尾核每次loop处理的sorted_expert数量
TILING_DATA_FIELD_DEF(int64_t, handleNumTailCorePerLoopBefore); // syncall前，模板3，尾核每次loop处理的sorted_expert数量
TILING_DATA_FIELD_DEF(int64_t, handleExpertNumLoopCount);       // syncall前，模板3，切E需要的loop次数
TILING_DATA_FIELD_DEF(int64_t, handleExpertNumMainCorePerLoop); // syncall前，模板3，非尾loop切分处理的E的个数
TILING_DATA_FIELD_DEF(int64_t, handleExpertNumTailCorePerLoop); // syncall前，模板3，尾loop切分处理的E的个数
TILING_DATA_FIELD_DEF(int64_t, loopCountTailCoreMainLoop);      // syncall前，模板3，尾核主loop的次数
TILING_DATA_FIELD_DEF(int64_t, handleNumTailCoreMainLoop); // syncall前，模板3，尾核主loop每次处理个数
TILING_DATA_FIELD_DEF(int64_t, loopCountTailCoreTailLoop); // syncall前，模板3，尾核尾loop的次数
TILING_DATA_FIELD_DEF(int64_t, handleNumTailCoreTailLoop); // syncall前，模板3，尾核尾loop每次处理个数
TILING_DATA_FIELD_DEF(int64_t, userWorkspaceSize);         // 使用的workspace
TILING_DATA_FIELD_DEF(int64_t, tilingKey);                 // 使用的字段tilingKey
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeComputeExpertTokens, MoeComputeExpertTokensTilingData)

struct MoeComputeExpertTokensCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_COMPUTE_EXPERT_TOKENS_H_
