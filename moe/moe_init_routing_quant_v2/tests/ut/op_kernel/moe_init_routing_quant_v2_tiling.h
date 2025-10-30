/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MOE_INIT_ROUTING_QUANT_V2_TILING_H__
#define __MOE_INIT_ROUTING_QUANT_V2_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct InnerMoeV2VBSComputeTilingData {
  int64_t needCoreNum = 0;
  int64_t perCoreElements = 0;
  int64_t perCoreLoops = 0;
  int64_t perCorePerLoopElements = 0;
  int64_t perCoreLastLoopElements = 0;
  int64_t lastCoreElements = 0;
  int64_t lastCoreLoops = 0;
  int64_t lastCorePerLoopElements = 0;
  int64_t lastCoreLastLoopElements = 0;
  int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, InnerMoeV2VBSComputeTilingData* const_data) {
  const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
  uint32_t* dst = (uint32_t*)const_data;
  for (auto i = 0; i < sizeof(InnerMoeV2VBSComputeTilingData) / 4; i++)
    *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, InnerMoeV2VBSComputeTilingData* const_data) {
  memcpy(const_data, tiling, sizeof(InnerMoeV2VBSComputeTilingData));
}
#endif

#pragma pack(1)
struct InnerMoeV2VMSMiddleComputeTilingData {
  int64_t needCoreNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, InnerMoeV2VMSMiddleComputeTilingData* const_data) {
  const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
  uint32_t* dst = (uint32_t*)const_data;
  for (auto i = 0; i < sizeof(InnerMoeV2VMSMiddleComputeTilingData) / 4; i++)
    *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, InnerMoeV2VMSMiddleComputeTilingData* const_data) {
  memcpy(const_data, tiling, sizeof(InnerMoeV2VMSMiddleComputeTilingData));
}
#endif

#pragma pack(1)
struct InnerMoeV2SortOutComputeTilingData {
  int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, InnerMoeV2SortOutComputeTilingData* const_data) {
  const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
  uint32_t* dst = (uint32_t*)const_data;
  for (auto i = 0; i < sizeof(InnerMoeV2SortOutComputeTilingData) / 4; i++)
    *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, InnerMoeV2SortOutComputeTilingData* const_data) {
  memcpy(const_data, tiling, sizeof(InnerMoeV2SortOutComputeTilingData));
}
#endif

#pragma pack(1)
struct InnerMoeV2GatherOutComputeTilingData {
  int64_t needCoreNum = 0;
  int64_t activateRows = 0;
  int64_t perCoreRows = 0;
  int64_t perCorePerLoopRows = 0;
  int64_t perCoreLastLoopRows = 0;
  int64_t lastCoreRows = 0;
  int64_t lastCorePerLoopRows = 0;
  int64_t lastCoreLastLoopRows = 0;
  int64_t perCoreLoops = 0;
  int64_t lastCoreLoops = 0;
  int64_t perLoopCols = 0;
  int64_t lastLoopCols = 0;
  int64_t colLoops = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, InnerMoeV2GatherOutComputeTilingData* const_data) {
  const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
  uint32_t* dst = (uint32_t*)const_data;
  for (auto i = 0; i < sizeof(InnerMoeV2GatherOutComputeTilingData) / 4; i++)
    *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, InnerMoeV2GatherOutComputeTilingData* const_data) {
  memcpy(const_data, tiling, sizeof(InnerMoeV2GatherOutComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeInitRoutingQuantV2TilingData {
  int64_t coreNum = 0;
  int64_t n = 0;
  int64_t cols = 0;
  int64_t k = 0;
  int64_t expertCapacity = 0;
  int64_t expertNum = 0;
  int64_t dropPadMode = 0;
  int64_t expertTokensCountOrCumsumFlag = 0;
  int64_t expertTokensBeforeCapacityFlag = 0;
  int64_t smoothType = 0;
  int64_t histWithRegBase = 0;
  int64_t tmpData = 0;
  InnerMoeV2VBSComputeTilingData vbsComputeParamsOp;
  InnerMoeV2VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
  InnerMoeV2SortOutComputeTilingData sortOutComputeParamsOp;
  InnerMoeV2GatherOutComputeTilingData srcToDstComputeParamsOp;
  InnerMoeV2GatherOutComputeTilingData srcToDstCapacityComputeParamsOp;
  InnerMoeV2GatherOutComputeTilingData gatherOutComputeParamsOp;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeInitRoutingQuantV2TilingData* const_data) {
  const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
  uint32_t* dst = (uint32_t*)const_data;
  for (auto i = 0; i < sizeof(MoeInitRoutingQuantV2TilingData) / 4; i++)
    *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeInitRoutingQuantV2TilingData* const_data) {
  memcpy(const_data, tiling, sizeof(MoeInitRoutingQuantV2TilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
  tiling_struct tiling_data;                                                \
  InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
  MoeInitRoutingQuantV2TilingData tiling_data;   \
  InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float

#endif