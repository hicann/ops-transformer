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
 * \file ffn_wb_scan_token_info.h
 * \brief
 */
#ifndef OP_KERNEL_FFN_WB_SCAN_TOKEN_INFO_H
#define OP_KERNEL_FFN_WB_SCAN_TOKEN_INFO_H

namespace FfnWbBatching {
using namespace AscendC;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t INT32_SIZE = 4;
constexpr uint32_t FLAG_SIZE = 4;
constexpr uint32_t FLAG_LAYER_LEN = 2;
constexpr uint32_t BUFFER_NUM_ONE = 1;

class KernelScanTokenInfo {
public:
    __aicore__ inline KernelScanTokenInfo(){};
    __aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR token_info_gm_addr, GM_ADDR workspace,
                                ScheduleContextInfo *scheduleContextInfo, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInFlag();

private:
    uint64_t curMicroBatchID = 0;
    uint32_t A = 0;
    uint32_t M = 0;
    uint32_t BS = 0;
    uint32_t K = 0;
    uint32_t F = 0;

    TPipe *pipe = nullptr;
    GlobalTensor<int32_t> tokenInfoGm;
    GlobalTensor<uint64_t> pollIdxGm;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> tokenFlagInQueue;
    TBuf<QuePosition::VECCALC> workBuf;
};

__aicore__ inline void KernelScanTokenInfo::CopyInFlag()
{
    int32_t offset = curMicroBatchID * F;
    LocalTensor<int32_t> tokenInfoLocalIn = tokenFlagInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(this->A), static_cast<uint32_t>(sizeof(int32_t)),
                                     static_cast<uint32_t>((M * F - 1) * sizeof(int32_t)), 0, 0};
    DataCopyPadExtParams dataCopyPadParams{true, 0, (BLOCK_SIZE - sizeof(int32_t)) / sizeof(int32_t), 0};
    DataCopyPad(tokenInfoLocalIn, tokenInfoGm[offset], dataCopyParams, dataCopyPadParams);
    tokenFlagInQueue.EnQue(tokenInfoLocalIn);
}

__aicore__ inline void KernelScanTokenInfo::Init(GM_ADDR schedule_context, GM_ADDR token_info_gm_addr,
                                                 GM_ADDR workspace, ScheduleContextInfo *scheduleContextInfo,
                                                 TPipe *tPipe)
{
    pipe = tPipe;
    A = scheduleContextInfo->A;
    M = scheduleContextInfo->M;
    BS = scheduleContextInfo->BS;
    K = scheduleContextInfo->K;
    F = 1 + 1 + BS * K;
    curMicroBatchID = scheduleContextInfo->curMicroBatchID;

    GlobalTensor<int32_t> rsvdCntGm;
    rsvdCntGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), SCAN_BATCHID_GM_OFFSET);
    if (GetBlockIdx() == 0) {
        InitGlobalMemory(rsvdCntGm, SCAN_BATCHID_GM_OFFSET, 0);

        GM_ADDR groupListTmpAddr =
            workspace + (OFFSET_SORTED_EXPERT_IDS + scheduleContextInfo->sortNumWorkSpace * (NUM_TWO * NUM_FOUR + 1)) *
                            sizeof(int32_t);
        GlobalTensor<int32_t> groupListTmpGm;
        groupListTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(groupListTmpAddr),
                                       scheduleContextInfo->expertNum);
        InitGlobalMemory(groupListTmpGm, scheduleContextInfo->expertNum, 0);

        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        // core0 清零 rsvdCnt(workspace[0]) 后必须 flush 到 GM 全局可见，再经 SyncAll 放行其他核：
        // 否则清零可能滞留 core0 cache，其他核 atomic-add 读到未清零的复用脏值(workspace 长跑复用
        // 脏页)→ 累加到脏基 → actual_token_num 偶发变负/减半(高频非确定)。单核路径(ffn_wb_sort_one_core.h)
        // 写 rsvdCnt 后同样用 DataCacheCleanAndInvalid 刷，多核清零此前漏了这步。
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
            rsvdCntGm);
    }
    // 为sort清理的gm,后面天然有 SyncAll,此处不需要

    tokenInfoGm.SetGlobalBuffer((__gm__ int32_t *)token_info_gm_addr, A * M * F);
    pollIdxGm.SetGlobalBuffer((__gm__ uint64_t *)schedule_context);

    pipe->InitBuffer(tokenFlagInQueue, 1, A * BLOCK_SIZE);
    pipe->InitBuffer(workBuf, A * BLOCK_SIZE);
}

__aicore__ inline void KernelScanTokenInfo::Process()
{
    if (GetBlockIdx() > 0) {
        return;
    }

    while (true) {
        CopyInFlag();
        LocalTensor<int32_t> tokenFlagLocal = tokenFlagInQueue.DeQue<int32_t>();
        LocalTensor<float> workLocal = workBuf.Get<float>();
        Cast(workLocal, tokenFlagLocal, RoundMode::CAST_ROUND, A * BLOCK_SIZE / sizeof(int32_t));
        PipeBarrier<PIPE_V>();

        ReduceSum<float>(workLocal, workLocal, workLocal, A * BLOCK_SIZE / sizeof(int32_t));
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> resultLocal = workLocal.ReinterpretCast<int32_t>();
        Cast(resultLocal, workLocal, RoundMode::CAST_ROUND, 1);
        PipeBarrier<PIPE_V>();

        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        int32_t readyNum = resultLocal.GetValue(0);

        tokenFlagInQueue.FreeTensor(tokenFlagLocal);
        workBuf.FreeTensor(workLocal);
        if (readyNum == A) {
            break;
        }
    }

    curMicroBatchID = (curMicroBatchID + 1) % M;
    pollIdxGm.SetValue(52, curMicroBatchID); // 52: pollidx offset
    DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_ALL>(
        pollIdxGm);
}
} // namespace FfnWbBatching
#endif // OP_KERNEL_FFN_WB_SCAN_TOKEN_INFO_H
