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
 * \file ffn_worker_batching.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "ffn_wb_get_schedule_context.h"
#include "ffn_wb_common.h"
#include "ffn_wb_gather_out_all.h"
#include "ffn_wb_group_listing.h"
#include "ffn_wb_sort_multi_core.h"
#include "ffn_wb_sort_one_core.h"
#include "ffn_wb_scan_token_info.h"
#include "ffn_wb_scan_sort_one_core.h"
#include "ffn_wb_scan_sort_multi_core.h"
#include "ffn_wb_scan_group_listing_one_core.h"
#include "ffn_wb_scan_sort_multi_core_bsk.h"
#include "ffn_wb_scan_get_valid_experts.h"

#define TILING_KEY_NORM 100
#define TILING_KEY_RECV 101
#define TILING_KEY_RECV_GATHER 102

using namespace AscendC;
using namespace FfnWbBatching;

extern "C" __aicore__ inline void FfnWorkerBatchingProcess(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list,
                                                           GM_ADDR session_ids, GM_ADDR micro_batch_ids,
                                                           GM_ADDR token_ids, GM_ADDR expert_offsets,
                                                           GM_ADDR dynamic_scale, GM_ADDR actual_token_num,
                                                           GM_ADDR userWS,
                                                           const FfnWorkerBatchingTilingData* tilingData)
{
    TPipe ffnPipe;
    ScheduleContextInfo contextInfo;
    ScheduleContextInfoCompute<false>(schedule_context, tilingData, contextInfo, &ffnPipe);
    ffnPipe.Reset();

    SortCustomTilingDataKernel tilingdataSort;
    TilingSort(&tilingdataSort, &contextInfo);
    if (tilingdataSort.tilingKey == TILINGKEY_ONECORE_SORT) {
        KernelSortMaskOneCore op;
        op.Init(reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.expertIdsBuf), userWS, &tilingdataSort, &contextInfo,
                &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    } else {
        KernelSortMaskMultiCore op;
        op.Init(reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.expertIdsBuf), userWS, &tilingdataSort, &contextInfo,
                &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    }

    // 这里获取有效长度，赋值给 contextInfo
    ValidGatherIdxLengthCompute(userWS, contextInfo, actual_token_num);

    KernelFfnWBGroupListing expertTokenOutMultiOp;

    int64_t groupListingDealFlag = 0;
    uint32_t usedCoreNum = GROUP_LISTING_AIV_NUM;
    if (contextInfo.validGatherIdxLength > GROUP_LISTING_MULTI_CORE_LENGTH) {
        groupListingDealFlag = 1;
        usedCoreNum = GROUP_LISTING_MULTI_AIV_NUM;
    }

    if (GetBlockIdx() < contextInfo.coreNum - usedCoreNum) {
        // 这里传递的是sort后的索引
        KernelFfnWBGatherOutAll<false> op;
        op.Init(userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace + tilingdataSort.totalLength) *
                             sizeof(int32_t),
                y, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, &contextInfo, &ffnPipe,
                usedCoreNum);
        op.Process();
        ffnPipe.Reset();
    } else {
        expertTokenOutMultiOp.Init(
            userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace) * sizeof(int32_t), group_list,
            userWS +
                (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace * (NUM_TWO * NUM_FOUR + 1)) * sizeof(int32_t),
            &contextInfo, &ffnPipe, groupListingDealFlag);
        expertTokenOutMultiOp.Process(groupListingDealFlag);
    }

    SyncAll();
    // 多处理逻辑且最后一核进行处理
    if (groupListingDealFlag && (GetBlockIdx() == contextInfo.coreNum - 1)) {
        expertTokenOutMultiOp.ProcessExpertCount();
    }
}

extern "C" __aicore__ inline void FfnWorkerBatchingRecvProcess(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list,
                                                               GM_ADDR session_ids, GM_ADDR micro_batch_ids,
                                                               GM_ADDR token_ids, GM_ADDR expert_offsets,
                                                               GM_ADDR dynamic_scale, GM_ADDR actual_token_num,
                                                               GM_ADDR userWS,
                                                               const FfnWorkerBatchingTilingData* tilingData)
{
    TPipe ffnPipe;
    ScheduleContextInfo contextInfo;
    ScheduleContextInfoCompute<true>(schedule_context, tilingData, contextInfo, &ffnPipe);
    ffnPipe.Reset();

    GM_ADDR tokenInfoGmAddr = reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.tokenInfoBuf);
    KernelScanTokenInfo scanTokenInfo;
    scanTokenInfo.Init(schedule_context, tokenInfoGmAddr, userWS, &contextInfo, &ffnPipe);
    scanTokenInfo.Process();
    SyncAll();
    ffnPipe.Reset();

    SortCustomTilingDataKernel tilingdataSort;
    TilingScanSort(&tilingdataSort, &contextInfo);
    if (tilingdataSort.tilingKey == TILINGKEY_ONECORE_SORT) {
        KernelScanSortMaskOneCore op;
        op.Init(tokenInfoGmAddr, userWS, &tilingdataSort, &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    } else if (tilingdataSort.isNormalBsK != NORMAL_BSK) {
        KernelScanSortMaskMultiCoreBsK op;
        op.Init(tokenInfoGmAddr, userWS, &tilingdataSort, &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    } else {
        KernelScanSortMaskMultiCore op;
        op.Init(tokenInfoGmAddr, userWS, &tilingdataSort, &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    }

    // 这里确定 有效长度，赋值给 contextInfo
    ValidGatherIdxLengthCompute(userWS, contextInfo, actual_token_num);

    int64_t blockIdx = GetBlockIdx();
    if (blockIdx < tilingData->coreNum - GROUP_LISTING_SCAN_AIV_NUM) {
        // 这里传递的是sort后的索引
        KernelFfnWBGatherOutAll<true> op;
        op.Init(userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace + tilingdataSort.totalLengthWithPad) *
                             sizeof(int32_t),
                y, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, &contextInfo, &ffnPipe,
                GROUP_LISTING_SCAN_AIV_NUM);
        op.Process();
        ffnPipe.Reset();
    } else {
        // group listing
        KernelFfnWBScanGroupListingOneCore expertTokenOutOp;
        // 这里workspace 记得加上偏移 排序后的专家
        expertTokenOutOp.Init(userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace) * sizeof(int32_t),
                              group_list, &contextInfo, &ffnPipe);
        expertTokenOutOp.Process();
        ffnPipe.Reset();
    }
}

extern "C" __aicore__ inline void FfnWorkerBatchingRecvSpecProcess(GM_ADDR schedule_context, GM_ADDR y,
                                                                   GM_ADDR group_list, GM_ADDR session_ids,
                                                                   GM_ADDR micro_batch_ids, GM_ADDR token_ids,
                                                                   GM_ADDR expert_offsets, GM_ADDR dynamic_scale,
                                                                   GM_ADDR actual_token_num, GM_ADDR userWS,
                                                                   const FfnWorkerBatchingTilingData* tilingData)
{
    TPipe ffnPipe;
    ScheduleContextInfo contextInfo;
    ScheduleContextInfoCompute<true>(schedule_context, tilingData, contextInfo, &ffnPipe);
    ffnPipe.Reset();

    GM_ADDR tokenInfoGmAddr = reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.tokenInfoBuf);
    KernelScanTokenInfo scanTokenInfo;
    scanTokenInfo.Init(schedule_context, tokenInfoGmAddr, userWS, &contextInfo, &ffnPipe);
    scanTokenInfo.Process();
    SyncAll();
    ffnPipe.Reset();

    KernelScanGetValidExperts op;
    op.Init(tokenInfoGmAddr, userWS, group_list, &contextInfo, &ffnPipe);
    op.Process();
    ffnPipe.Reset();

    ValidGatherIdxLengthCompute(userWS, contextInfo, actual_token_num);

    int64_t totalLengthWithPad = contextInfo.A * (contextInfo.BS * contextInfo.K + contextInfo.BsKPaddingCount);
    KernelFfnWBGatherOutAll<true> opGather;
    opGather.Init(
        userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace + totalLengthWithPad) * sizeof(int32_t), y,
        session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, &contextInfo, &ffnPipe, 0);
    opGather.Process();
}

extern "C" __global__ __aicore__ void ffn_worker_batching(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list,
                                                          GM_ADDR session_ids, GM_ADDR micro_batch_ids,
                                                          GM_ADDR token_ids, GM_ADDR expert_offsets,
                                                          GM_ADDR dynamic_scale, GM_ADDR actual_token_num,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_NORM)) {
        FfnWorkerBatchingProcess(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                 expert_offsets, dynamic_scale, actual_token_num, userWS, &tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_RECV)) {
        FfnWorkerBatchingRecvProcess(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                     expert_offsets, dynamic_scale, actual_token_num, userWS, &tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_RECV_GATHER)) {
        FfnWorkerBatchingRecvSpecProcess(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                         expert_offsets, dynamic_scale, actual_token_num, userWS, &tilingData);
    }
}
