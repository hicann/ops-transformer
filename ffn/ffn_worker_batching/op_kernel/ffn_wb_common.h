/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_common.h
 * \brief
 */

#ifndef OP_KERNEL_FFN_WB_COMMON_H
#define OP_KERNEL_FFN_WB_COMMON_H

#include "kernel_operator.h"

namespace FfnWbBatching {
using namespace AscendC;

// used for tiling
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t MRG_LIST_NUM = 4;
constexpr int64_t SORT32_ALIGN_ELEMENT = 32;
constexpr int64_t MGR_SORT_MAX_ELEMENT = 1024;
constexpr int64_t TILINGKEY_ONECORE_SORT = 0;
constexpr int64_t TILINGKEY_MULTICORE_SORT = 1;
constexpr int64_t NORMAL_BSK = 1; // Bs*K+pad小于等于UB

// used for sort
constexpr float MIN_FP32 = -3.4e38;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;
// ONE_REPEAT_COMPARE_NUM used for mask
constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;
constexpr int64_t BLOCK_BYTES = 32;

constexpr int64_t MAX_EXPERT_NUM = 5120;
constexpr int64_t EXPERT_ID_VALUE_NUM = 2;

constexpr int64_t GROUP_LISTING_MULTI_CORE_LENGTH = 512;
constexpr uint32_t GROUP_LISTING_MULTI_AIV_NUM = 4;
constexpr uint32_t GROUP_LISTING_AIV_NUM = 1;
constexpr uint32_t GROUP_LISTING_SCAN_AIV_NUM = 1;
constexpr int64_t ASSIST_NUM = 256;
constexpr int64_t OFFSET_SORTED_EXPERT_IDS = 128; // workspace偏移128个uint32，开始写排序的结果
constexpr int64_t SCAN_BATCHID_GM_OFFSET = 32;

struct BufferInfo {
    uint64_t tokenInfoBuf = 0;
    uint64_t tokenDataBuf = 0;
    uint64_t sessionIdsBuf = 0;
    uint64_t microBatchIdsBuf = 0;
    uint64_t expertIdsBuf = 0;
};

struct ScheduleContextInfo {
    uint32_t A = 0;  // A, attention session num
    uint32_t M = 0;  // M, micro batch num
    uint32_t BS = 0; // BS, batch size
    uint32_t K = 0;  // K = topK+1
    uint32_t HS = 0; // HS, 读自 attn_to_ffn_token_size， 单位字节
    uint32_t H = 0;  // 算子的输入attr，个数
    uint32_t Y = 0;
    uint64_t curMicroBatchID = 0; // recv功能增加的，标识当前哪个micro batchid的expert id已经ready
    uint32_t outNum = 0; // 取自 FfnArea, 标识多少个attention session num的数据有效。 非recv功能时使用
    uint32_t tokenDtype = 0; // 算子attr. 标识Y的数据类型以及 FfnArea.token_data_buf的存储方式
                             // 0: FP16; 1: BF16 2: 输入是int8动态量化产生的，并且int8的数据和dynamic scale连续排布
    uint32_t expertNum = 0;         // 这里设置的是attr 的 expert_num；
    int64_t coreNum = 0;            // tiling中取到的总的aiv core
    int64_t ubSize = 0;             // tiling中取到的可用ub大小
    int64_t sortLoopMaxElement = 0; // 每次ub最多可以排序多少个int32。取自tilingdata
    int64_t sortNumWorkSpace = 0; // 用于记录每次排序后的个数。所有核一起需要占用的总空间。单位：个数。取自tilingdata
    int64_t validGatherIdxLength = 0; // 文件排序后，去掉 100w(无效值的长度） 其值 ≤ A * BS * K 即gather_idx 有效长度;
    int64_t BsKPaddingCount = 0; // BS*K 按block对齐需要补的pad个数, recv功能使用

    // 放的是 切切实实的 gm 地址, 目前通过bin文件放的是偏移 ,所以需要加GM_ADDR schedule_context
    BufferInfo bufferPtr;
};

// TilingInfo for sort
struct SortCustomTilingDataKernel {
    int64_t totalLength = 0;        // A*BS*K
    int64_t totalLengthWithPad = 0; // recv场景下使用 A*(BS*K+pad)
    int64_t needCoreNum = 0;
    int64_t needCoreNumMrg = 0;
    int64_t perCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreElements = 0;
    int64_t lastCoreLoops = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
    int64_t oneLoopMaxElementsMrg = 0;
    int64_t tilingKey = 0;
    int64_t perCoreSessionNum = 0;
    int64_t lastCoreSessionNum = 0;
    int64_t sortNumWorkSpacePerCore = 0; // 用于记录每次排序后的个数。每个核需要占用的空间。单位：个数。
    int64_t isNormalBsK = 0; // new 判断Bs*K是否大于UB
};

struct FfnWBGroupListingTileInfo {
    int64_t needCoreNum = 0;
    int64_t perCoreRows = 0;
    int64_t perCorePerLoopRows = 0;
    int64_t perCoreLastLoopRows = 0;
    int64_t lastCoreRows = 0;
    int64_t lastCorePerLoopRows = 0;
    int64_t lastCoreLastLoopRows = 0;
    int64_t perCoreLoops = 0;
    int64_t lastCoreLoops = 0;
    int64_t perLoopCols = 0;
};

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T2 CeilDiv(T1 x, T2 y)
{
    if (y != 0 && x != 0) {
        const T2 quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }

    return x;
}

template <typename T>
__aicore__ inline T PowerOfFourCeil(T x)
{
    if (x < 0) {
        return 0;
    }
    T result = 1;
    while (result < x) {
        result *= NUM_FOUR;
    }
    return result;
}

__aicore__ inline void TilingOneCoreSort(SortCustomTilingDataKernel *tilingData, const int64_t totalLength)
{
    tilingData->needCoreNum = totalLength == 0 ? 0 : 1;
    tilingData->perCoreElements = totalLength;
    tilingData->perCoreLoops = 1;
    tilingData->perCorePerLoopElements = totalLength;
    tilingData->perCoreLastLoopElements = totalLength;
    tilingData->lastCoreElements = totalLength;
    tilingData->lastCoreLoops = 1;
    tilingData->lastCorePerLoopElements = totalLength;
    tilingData->lastCoreLastLoopElements = totalLength;
}

__aicore__ inline void TilingMultiCoreSort(SortCustomTilingDataKernel *tilingData, const int64_t totalLength,
                                           const ScheduleContextInfo *contextInfo)
{
    int64_t needCoreNum = CeilDiv(totalLength, contextInfo->sortLoopMaxElement); // 向上取整
    needCoreNum = PowerOfFourCeil(needCoreNum);           // 用到多核时，核数最多是4^x, 计算不小于
                                                          // needCoreNum 的最小的 4 的整数次幂
    needCoreNum = Min(needCoreNum, contextInfo->coreNum); // 不能超过物理核数
    if (needCoreNum == 0) {
        tilingData->needCoreNum = 0;
        return;
    }
    int64_t perCoreElements = totalLength / needCoreNum; // 每个核处理的元素数
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = CeilDiv(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->needCoreNum = needCoreNum;
    tilingData->sortNumWorkSpacePerCore = contextInfo->sortNumWorkSpace / needCoreNum;
    do {
        tilingData->perCoreElements = perCoreElements;
        tilingData->perCoreLoops = CeilDiv(tilingData->perCoreElements, contextInfo->sortLoopMaxElement);
        tilingData->perCorePerLoopElements = Min(tilingData->perCoreElements, contextInfo->sortLoopMaxElement);

        tilingData->perCoreLastLoopElements =
            tilingData->perCoreElements - (tilingData->perCoreLoops - 1) * tilingData->perCorePerLoopElements;

        tilingData->lastCoreElements = totalLength - (tilingData->needCoreNum - 1) * tilingData->perCoreElements;
        tilingData->lastCoreLoops = tilingData->perCoreLoops;
        int64_t lastCorePerLoopElements =
            CeilDiv(CeilDiv(tilingData->lastCoreElements, tilingData->lastCoreLoops), SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->lastCorePerLoopElements = lastCorePerLoopElements;
        tilingData->lastCoreLastLoopElements =
            (tilingData->lastCoreElements - (tilingData->lastCoreLoops - 1) * tilingData->lastCorePerLoopElements);
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->lastCoreLastLoopElements <= 0 && perCoreElements > 0);

    if (needCoreNum <= MRG_LIST_NUM) {
        tilingData->needCoreNumMrg = 0;
    } else {
        int64_t needCoreNumMrg = CeilDiv(needCoreNum, MRG_LIST_NUM);
        tilingData->needCoreNumMrg = needCoreNumMrg;
    }
    tilingData->oneLoopMaxElementsMrg = MGR_SORT_MAX_ELEMENT;
}

__aicore__ inline void TilingScanMultiCoreSort(SortCustomTilingDataKernel *tilingdataSort,
                                               const ScheduleContextInfo *contextInfo)
{
    // 按A分核
    // 1. 计算每次排序可以放下多少个 session_num
    int64_t BsKPad = contextInfo->BS * contextInfo->K + contextInfo->BsKPaddingCount;
    int64_t maxSessionPerSort = Max(1L, contextInfo->sortLoopMaxElement / BsKPad);
    int64_t leftSessionNum = CeilDiv(contextInfo->A, maxSessionPerSort);
    int64_t perCoreLeftSessionNum = CeilDiv(leftSessionNum, contextInfo->coreNum);
    int64_t needCoreNum = Min(CeilDiv(leftSessionNum, perCoreLeftSessionNum), contextInfo->coreNum);
    tilingdataSort->needCoreNum = needCoreNum;
    if (needCoreNum == 0) {
        return;
    }

    tilingdataSort->sortNumWorkSpacePerCore = contextInfo->sortNumWorkSpace / needCoreNum;
    tilingdataSort->perCoreSessionNum = CeilDiv(contextInfo->A, needCoreNum);
    tilingdataSort->lastCoreSessionNum = contextInfo->A - (needCoreNum - 1) * tilingdataSort->perCoreSessionNum;

    if (needCoreNum <= MRG_LIST_NUM) {
        tilingdataSort->needCoreNumMrg = 0;
    } else {
        int64_t needCoreNumMrg = CeilDiv(needCoreNum, MRG_LIST_NUM);
        tilingdataSort->needCoreNumMrg = needCoreNumMrg;
    }
    tilingdataSort->oneLoopMaxElementsMrg = MGR_SORT_MAX_ELEMENT;
}

__aicore__ inline void TilingSort(SortCustomTilingDataKernel *tilingdataSort, const ScheduleContextInfo *contextInfo)
{
    auto totalLength = contextInfo->outNum * contextInfo->BS * contextInfo->K;
    tilingdataSort->totalLength = totalLength;

    if (totalLength <= contextInfo->sortLoopMaxElement) {
        TilingOneCoreSort(tilingdataSort, totalLength);
        tilingdataSort->tilingKey = TILINGKEY_ONECORE_SORT;
    } else {
        TilingMultiCoreSort(tilingdataSort, totalLength, contextInfo);
        tilingdataSort->tilingKey = TILINGKEY_MULTICORE_SORT;
    }
}

__aicore__ inline void TilingScanSort(SortCustomTilingDataKernel *tilingdataSort,
                                      const ScheduleContextInfo *contextInfo)
{
    tilingdataSort->totalLengthWithPad =
        contextInfo->A * (contextInfo->BS * contextInfo->K + contextInfo->BsKPaddingCount);
    tilingdataSort->totalLength = contextInfo->A * contextInfo->BS * contextInfo->K;

    if (tilingdataSort->totalLengthWithPad <= contextInfo->sortLoopMaxElement) {
        TilingOneCoreSort(tilingdataSort, tilingdataSort->totalLengthWithPad);
        tilingdataSort->tilingKey = TILINGKEY_ONECORE_SORT;
    } else {
        TilingScanMultiCoreSort(tilingdataSort, contextInfo);
        tilingdataSort->tilingKey = TILINGKEY_MULTICORE_SORT;
        if (contextInfo->BS * contextInfo->K + contextInfo->BsKPaddingCount <= contextInfo->sortLoopMaxElement) {
            tilingdataSort->isNormalBsK = NORMAL_BSK;
        }
    }
}

__aicore__ inline void ValidGatherIdxLengthCompute(GM_ADDR work_space, ScheduleContextInfo &contextInfo,
                                                   GM_ADDR actual_token_num)
{
    GlobalTensor<int32_t> workSpace;
    GlobalTensor<int64_t> actualTokenNumGm;

    workSpace.SetGlobalBuffer((__gm__ int32_t *)work_space, OFFSET_SORTED_EXPERT_IDS);
    actualTokenNumGm.SetGlobalBuffer((__gm__ int64_t *)actual_token_num);

    DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
        workSpace);
    contextInfo.validGatherIdxLength = workSpace.GetValue(0);

    if (GetBlockIdx() == 0) {
        actualTokenNumGm.SetValue(0, contextInfo.validGatherIdxLength);
        DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
            actualTokenNumGm);
    }
}

__aicore__ inline void Tiling4SrcToDstCompute(const ScheduleContextInfo *tilingData,
                                              FfnWBGroupListingTileInfo &tilingInfo)
{
    int64_t ubSizePlatForm = tilingData->ubSize;
    int64_t validGatherIdxLength = tilingData->validGatherIdxLength;

    int64_t perLoopMaxRows =
        (ubSizePlatForm - ASSIST_NUM * sizeof(float) - GROUP_LISTING_AIV_NUM * SORT32_ALIGN_ELEMENT) /
        (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;

    int64_t perCoreRows = CeilDiv(validGatherIdxLength, GROUP_LISTING_AIV_NUM);
    if (perCoreRows <= 0) {
        tilingInfo.needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = CeilDiv(validGatherIdxLength, perCoreRows);
    tilingInfo.needCoreNum = needCoreNum;
    int64_t lastCoreNum = validGatherIdxLength - perCoreRows * (needCoreNum - 1);

    tilingInfo.perCoreRows = perCoreRows;

    if (perLoopMaxRows >= perCoreRows) { // 一个loop结束
        tilingInfo.perCorePerLoopRows = perCoreRows;
        tilingInfo.perCoreLastLoopRows = perCoreRows;
    } else {
        tilingInfo.perCorePerLoopRows = perLoopMaxRows;
        tilingInfo.perCoreLastLoopRows = perCoreRows - (CeilDiv(perCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
    }

    tilingInfo.lastCoreRows = lastCoreNum;
    if (perLoopMaxRows >= lastCoreNum) {
        tilingInfo.lastCorePerLoopRows = lastCoreNum;
        tilingInfo.lastCoreLastLoopRows = lastCoreNum;
    } else {
        tilingInfo.lastCorePerLoopRows = perLoopMaxRows;
        tilingInfo.lastCoreLastLoopRows = lastCoreNum - (CeilDiv(lastCoreNum, perLoopMaxRows) - 1) * perLoopMaxRows;
    }
}

} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_COMMON_H
