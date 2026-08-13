/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
 * \file ffn_wb_a5_prepare.h
 * \brief phase0:等待就绪(RECV) + 把两条路径的 expert_id 归一到同一个扁平缓冲。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_PREPARE_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_PREPARE_H
#include "ffn_wb_a5_context.h"
#include "kernel_operator.h"
namespace FfnWbBatchingArch35 {
using namespace AscendC;

// ===================== RECV:等待本 micro batch 就绪 =====================
class FfnWbA5RecvWait {
public:
    __aicore__ inline FfnWbA5RecvWait(){};

    __aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR tokenInfoBuf, const ScheduleContextInfo *ctx,
                                TPipe *pipe)
    {
        ctx_ = ctx;
        pipe_ = pipe;
        // FfnDataDesc 每块的 int32 个数:flag + layer_id + expert_ids[BS*K],块数由契约结构给出。
        descWords_ = static_cast<int64_t>(sizeof(aicpu::FfnDataDesc)) / static_cast<int64_t>(sizeof(int32_t)) +
                     static_cast<int64_t>(ctx_->BS) * ctx_->K;
        tokenInfoGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenInfoBuf),
                                     static_cast<int64_t>(ctx_->A) * ctx_->M * descWords_);
        ctxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(schedule_context));

        const int64_t flagWords = static_cast<int64_t>(ctx_->A) * (ONE_BLK_SIZE / sizeof(int32_t));
        pipe_->InitBuffer(flagQue_, 1, flagWords * sizeof(int32_t));
        pipe_->InitBuffer(workBuf_, flagWords * sizeof(float));
        pipe_->InitBuffer(pollBuf_, ONE_BLK_SIZE);
    }

    // 由 0 号核忙等;其余核在调用方的 SyncAll 处等待。
    __aicore__ inline void Process()
    {
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        if (GetBlockIdx() * subNum + GetSubBlockIdx() != 0) {
            return;
        }
        const int64_t flagElems = static_cast<int64_t>(ctx_->A) * (ONE_BLK_SIZE / sizeof(int32_t));
        const int64_t offset = static_cast<int64_t>(ctx_->curMicroBatchID) * descWords_;

        while (true) {
            LocalTensor<int32_t> flagLocal = flagQue_.AllocTensor<int32_t>();
            // 每 session 取一个 flag:块数 A、块长 4B、块间跨度 (M*F-1) 个 int32
            DataCopyExtParams cp{
                static_cast<uint16_t>(ctx_->A), static_cast<uint32_t>(sizeof(int32_t)),
                static_cast<uint32_t>((static_cast<int64_t>(ctx_->M) * descWords_ - 1) * sizeof(int32_t)), 0, 0};
            DataCopyPadExtParams<int32_t> pad{
                true, 0, static_cast<uint8_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t)), 0};
            DataCopyPad(flagLocal, tokenInfoGm_[offset], cp, pad);
            flagQue_.EnQue(flagLocal);

            LocalTensor<int32_t> flags = flagQue_.DeQue<int32_t>();
            LocalTensor<float> work = workBuf_.Get<float>();
            Cast(work, flags, RoundMode::CAST_ROUND, flagElems);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(work, work, work, flagElems);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            const float readyNum = work.GetValue(0);
            flagQue_.FreeTensor(flags);
            if (static_cast<uint32_t>(readyNum) >= ctx_->A) {
                break;
            }
        }

        // 推进轮询下标:偏移取自公共契约结构,回写经 UB 整段搬运。
        LocalTensor<uint64_t> pollLocal = pollBuf_.Get<uint64_t>();
        pollLocal.SetValue(0, (ctx_->curMicroBatchID + 1) % ctx_->M);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams cpPoll{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
        DataCopyPad(ctxGm_[FFN_WB_CTX_OFFSET(ffn.polling_index) / static_cast<int32_t>(sizeof(uint64_t))], pollLocal,
                    cpPoll);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

private:
    const ScheduleContextInfo *ctx_ = nullptr;
    TPipe *pipe_ = nullptr;
    GlobalTensor<int32_t> tokenInfoGm_;
    GlobalTensor<uint64_t> ctxGm_;
    TQue<QuePosition::VECIN, 1> flagQue_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> pollBuf_;
    int64_t descWords_ = 0;
};

// ===================== expert_id 归一 + 握手回写 =====================
// 补位/失效标记:>= sort 的 expertStart_(1000000),排序后落到末尾并被 mask 判据剔除。
constexpr int32_t MASK_SENTINEL = 2147483647;

class FfnWbPrepareArch35 {
public:
    __aicore__ inline FfnWbPrepareArch35(){};

    // flatIdsWs:归一后的扁平 expert_id 缓冲(长度 totalLen)。
    // rowsPerLoop 由 host 按运行时 UB 容量反推(见 tiling 的 preparePerLoopRows),此处不设任何容量常数。
    __aicore__ inline void Init(GM_ADDR flatIdsWs, const ScheduleContextInfo *contextInfo, TPipe *pipe,
                                int64_t totalLen, int64_t rowsPerLoop)
    {
        contextInfo_ = contextInfo;
        pipe_ = pipe;
        totalLen_ = totalLen;
        bsk_ = contextInfo_->BS * contextInfo_->K;
        F_ = NUM_TWO + bsk_;
        // 每 session 行在扁平缓冲中的跨度(RECV 含补位;NORM 无补位时即为 bsk)
        rowSpan_ = (contextInfo_->A > 0) ? (totalLen_ / contextInfo_->A) : bsk_;
        rowsPerLoop_ = (rowsPerLoop > 0) ? rowsPerLoop : 1;
        if (rowsPerLoop_ > contextInfo_->A) {
            rowsPerLoop_ = contextInfo_->A;
        }
        // 按**运行时实际块数**把 session 行分段,各核只处理自己那段:
        // 各段写入 flatIds 的区间互不重叠,握手回写也按行分离,故无需跨核同步。
        const int64_t coreNum = GetBlockNum();
        const int64_t perCore = (contextInfo_->A + coreNum - 1) / coreNum;
        rowBegin_ = GetBlockIdx() * perCore;
        rowEnd_ = rowBegin_ + perCore;
        if (rowEnd_ > contextInfo_->A) {
            rowEnd_ = contextInfo_->A;
        }
        if (rowBegin_ > contextInfo_->A) {
            rowBegin_ = contextInfo_->A;
        }
        if (rowsPerLoop_ > (rowEnd_ - rowBegin_) && (rowEnd_ - rowBegin_) > 0) {
            rowsPerLoop_ = rowEnd_ - rowBegin_;
        }
        flatIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flatIdsWs), totalLen_);
        pipe_->InitBuffer(que_, 1, Align(rowsPerLoop_ * rowSpan_ * sizeof(int32_t), BLOCK_BYTES));
        pipe_->InitBuffer(clrBuf_, Align(rowsPerLoop_ * BLOCK_BYTES, BLOCK_BYTES));
    }

    // NORM:expert_ids_buf 已是扁平布局,整段搬到 flatIds(单核即可,量级为 Y 个 int32)。
    __aicore__ inline void ProcessNorm(GM_ADDR expertIdsBuf)
    {
        if (rowEnd_ <= rowBegin_) {
            return;
        }
        GlobalTensor<int32_t> srcGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertIdsBuf), totalLen_);
        const int64_t perLoopElems = rowsPerLoop_ * rowSpan_;
        const int64_t beginElem = rowBegin_ * rowSpan_;
        const int64_t endElem = (rowEnd_ * rowSpan_ > totalLen_) ? totalLen_ : rowEnd_ * rowSpan_;
        for (int64_t off = beginElem; off < endElem; off += perLoopElems) {
            const int64_t n = ((endElem - off) > perLoopElems) ? perLoopElems : (endElem - off);
            LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();
            DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(buf, srcGm[off], cp, pad);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(flatIdsGm_[off], buf, cp);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            que_.FreeTensor(buf);
        }
    }

    // RECV:逐 session 行跨步取 ids(尾部补 MASK_SENTINEL),并回写 MASK_SENTINEL/0 完成握手。
    // 按 rowsPerLoop_ 行一轮处理,轮内 UB 只驻留本轮的 id 区与清零区。
    __aicore__ inline void ProcessRecv(GM_ADDR tokenInfoBuf)
    {
        if (rowEnd_ <= rowBegin_) {
            return;
        }
        GlobalTensor<int32_t> tokenInfoGm;
        tokenInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenInfoBuf),
                                    contextInfo_->A * contextInfo_->M * F_);
        const int64_t base = contextInfo_->curMicroBatchID * F_;
        const int64_t rowStride = contextInfo_->M * F_; // 相邻 session 在 token_info 中的跨度
        LocalTensor<int32_t> clr = clrBuf_.Get<int32_t>();

        for (int64_t r0 = rowBegin_; r0 < rowEnd_; r0 += rowsPerLoop_) {
            const int64_t rows = ((rowEnd_ - r0) > rowsPerLoop_) ? rowsPerLoop_ : (rowEnd_ - r0);
            LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();

            // 取 ids:每块 bsk 个 int32,块间跳过 flag/layer 与其余 micro batch;
            // 右侧补位由 DataCopyPad 按 BsKPaddingCount 填 MASK_SENTINEL,使每行在 UB 中占 rowSpan_
            DataCopyExtParams inParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(bsk_ * sizeof(int32_t)),
                                       static_cast<uint32_t>((rowStride - bsk_) * sizeof(int32_t)), 0, 0};
            DataCopyPadExtParams<int32_t> inPad{true, 0, static_cast<uint8_t>(contextInfo_->BsKPaddingCount),
                                                MASK_SENTINEL};
            DataCopyPad(buf, tokenInfoGm[base + r0 * rowStride + NUM_TWO], inParams, inPad);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

            DataCopyExtParams outParams{static_cast<uint16_t>(1),
                                        static_cast<uint32_t>(rows * rowSpan_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(flatIdsGm_[r0 * rowSpan_], buf, outParams);

            // 握手回写前必须等上面的搬出真正读完 buf:下面要就地把 buf 覆盖成回写内容。
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            // 回写值处处相同(ids 全 MASK_SENTINEL、flag 全 0),故按连续块读出即可,与行内跨度无关
            Duplicate<int32_t>(buf, MASK_SENTINEL, rows * rowSpan_);
            Duplicate<int32_t>(clr, 0, rows * (BLOCK_BYTES / static_cast<int64_t>(sizeof(int32_t))));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams clrIds{static_cast<uint16_t>(rows), static_cast<uint32_t>(bsk_ * sizeof(int32_t)), 0,
                                     static_cast<uint32_t>((rowStride - bsk_) * sizeof(int32_t)), 0};
            DataCopyExtParams clrFlag{static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(int32_t)), 0,
                                      static_cast<uint32_t>((rowStride - 1) * sizeof(int32_t)), 0};
            DataCopyPad(tokenInfoGm[base + r0 * rowStride + NUM_TWO], buf, clrIds);
            DataCopyPad(tokenInfoGm[base + r0 * rowStride], clr, clrFlag);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            que_.FreeTensor(buf);
        }
    }

private:
    const ScheduleContextInfo *contextInfo_ = nullptr;
    TPipe *pipe_ = nullptr;
    GlobalTensor<int32_t> flatIdsGm_;
    TQue<QuePosition::VECCALC, 1> que_;
    TBuf<QuePosition::VECCALC> clrBuf_;
    int64_t totalLen_ = 0;
    int64_t bsk_ = 0;
    int64_t rowSpan_ = 0;
    int64_t rowsPerLoop_ = 0;
    int64_t rowBegin_ = 0;
    int64_t rowEnd_ = 0;
    int64_t F_ = 0;
};
} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_A5_PREPARE_H
