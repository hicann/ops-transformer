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

// ===================== RECV:选择本次消费的 micro batch 和 session =====================
// 同步：只轮询入口 polling_index 指向的 micro batch，等 A 个 flag 全为 1。
// 异步：循环扫描 micro batch，遇到至少一个 flag=1 就停止，将该轮观察到的 flag
// 保存到 readyWs。后续 prepare 的读入与清理只认这份快照，不能重新查询生产者的实时 flag。
// readyWs 布局：[一个 32B 头块，低 8B 为 selectedMb][A 个 32B 块，每块低 4B 为 flag]。
// 只有 0 号向量核写快照和 polling_index；调用方负责发布前后的跨核 SyncAll。
class FfnWbA5RecvWait {
public:
    __aicore__ inline FfnWbA5RecvWait(){};

    __aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR tokenInfoBuf, const ScheduleContextInfo* ctx,
                                TPipe* pipe, GM_ADDR readyWs)
    {
        ctx_ = ctx;
        readyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(readyWs));
        pipe_ = pipe;
        // FfnDataDesc 每块的 int32 个数:flag + layer_id + expert_ids[BS*K],块数由契约结构给出。
        descWords_ = static_cast<int64_t>(sizeof(aicpu::FfnDataDesc)) / static_cast<int64_t>(sizeof(int32_t)) +
                     static_cast<int64_t>(ctx_->BS) * ctx_->K;
        tokenInfoGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenInfoBuf),
                                     static_cast<int64_t>(ctx_->A) * ctx_->M * descWords_);
        ctxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(schedule_context));

        const int64_t flagWords = static_cast<int64_t>(ctx_->A) * (ONE_BLK_SIZE / sizeof(int32_t));
        pipe_->InitBuffer(flagQue_, 1, flagWords * sizeof(int32_t));
        pipe_->InitBuffer(workBuf_, flagWords * sizeof(float));
        pipe_->InitBuffer(pollBuf_, ONE_BLK_SIZE);
    }

    // 由 0 号核忙等;其余核在调用方的 SyncAll 处等待。
    __aicore__ inline void Process()
    {
        if (GetFlatAivIdx() != 0) {
            return;
        }
        const int64_t flagElems = static_cast<int64_t>(ctx_->A) * (ONE_BLK_SIZE / sizeof(int32_t));
        // selectedMb 是本核的扫描游标。空轮只更新局部值，尚不回写共享 polling_index；
        // 找到数据后才发布选中项，并为下一次调用推进共享游标。
        uint64_t selectedMb = ctx_->curMicroBatchID;

        while (true) {
            LocalTensor<int32_t> flagLocal = flagQue_.AllocTensor<int32_t>();
            // 每 session 取一个 flag:块数 A、块长 4B、块间跨度 (M*F-1) 个 int32
            DataCopyExtParams cp{
                static_cast<uint16_t>(ctx_->A), static_cast<uint32_t>(sizeof(int32_t)),
                static_cast<uint32_t>((static_cast<int64_t>(ctx_->M) * descWords_ - 1) * sizeof(int32_t)), 0, 0};
            DataCopyPadExtParams<int32_t> pad{
                true, 0, static_cast<uint8_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t)), 0};
            DataCopyPad(flagLocal, tokenInfoGm_[selectedMb * descWords_], cp, pad);
            flagQue_.EnQue(flagLocal);

            LocalTensor<int32_t> flags = flagQue_.DeQue<int32_t>();
            LocalTensor<float> work = workBuf_.Get<float>();
            Cast(work, flags, RoundMode::CAST_ROUND, flagElems);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(work, work, work, flagElems);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            const float readyNum = work.GetValue(0);
            // 协议要求有效 flag 只取 0/1；DataCopyPad 补出的 7 个 int32 都是 0，
            // 因而 padded 数组的 ReduceSum 等于本轮观察到的 ready session 数。
            // 此处判断描述符就绪，不判断有效 token 数；全 mask 描述符也需要被消费。
            const bool ready = ctx_->asyncRecv ? readyNum > 0 : static_cast<uint32_t>(readyNum) >= ctx_->A;
            if (ready && ctx_->asyncRecv) {
                // flags 仍保留本轮 MTE2 读取结果，Cast/ReduceSum 使用另一个 work 缓冲。
                // 将原始 padded flag 整段存入快照，不重新读取 tokenInfoGm_：否则判定后
                // 到达的 session 可能被不同核纳入不同的集合。快照不代表所有 flag 的
                // 原子瞬时读取，而是这次跨步 DMA 实际观察到的集合。
                // 搬出完成后才释放 flags，防止下一轮 AllocTensor 复用尚在读取的 UB。
                DataCopyExtParams snapshot{1, static_cast<uint32_t>(flagElems * sizeof(int32_t)), 0, 0, 0};
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyPad(readyGm_[ONE_BLK_SIZE / sizeof(int32_t)], flags, snapshot);
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            }
            flagQue_.FreeTensor(flags);
            if (ready) {
                break;
            }
            if (ctx_->asyncRecv) {
                // 异步上游可能永远不给当前空 micro batch 置 flag，因此必须跳过空项。
                // 扫完 M 项仍无数据会回到起点继续等；发现任意 ready 即结束扫描。
                selectedMb = (selectedMb + 1) % ctx_->M;
            }
        }

        // 推进轮询下标:偏移取自公共契约结构,回写经 UB 整段搬运。
        LocalTensor<uint64_t> pollLocal = pollBuf_.Get<uint64_t>();
        if (ctx_->asyncRecv) {
            // 头块记录“本次消费哪一项”，共享 polling_index 则记录“下次从哪项开始”。
            // 两者含义不同：其它核必须从 readyWs 头块取本次索引，不能回读推进后的游标。
            pollLocal.SetValue(0, selectedMb);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams selected{1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
            DataCopyPad(readyGm_, pollLocal.ReinterpretCast<int32_t>(), selected);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        pollLocal.SetValue(0, (selectedMb + 1) % ctx_->M);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams cpPoll{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
        DataCopyPad(ctxGm_[FFN_WB_CTX_OFFSET(ffn.polling_index) / static_cast<int32_t>(sizeof(uint64_t))], pollLocal,
                    cpPoll);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

private:
    const ScheduleContextInfo* ctx_ = nullptr;
    TPipe* pipe_ = nullptr;
    GlobalTensor<int32_t> tokenInfoGm_;
    GlobalTensor<uint64_t> ctxGm_;
    GlobalTensor<int32_t> readyGm_;
    TQue<QuePosition::VECIN, 1> flagQue_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> pollBuf_;
    int64_t descWords_ = 0;
};

// 从已发布的 ready 快照恢复本次索引，更新调用核自己的上下文。
// 前置条件：仅异步 RECV 调用；调用方已执行发布 SyncAll，且 pipe 中没有待复用的活跃缓冲。
// 必须由所有参与核调用：ctx 是核私有对象，0 号核赋值不能替代其它核的更新。
// 本函数独占这段 UB 使用阶段，读完后 Reset；跨核屏障仍由 RunPrepare 统一安排。
__aicore__ inline void LoadSelectedMicroBatch(GM_ADDR readyWs, ScheduleContextInfo& ctx, TPipe& pipe)
{
    // 扫描可能跳过若干空项，故入口 ctx.curMicroBatchID 不一定是最终选中项。
    // 所有核统一读取快照头块，后续 descriptor 寻址和 gather 输出的 micro_batch_ids
    // 都使用这个本次索引；不使用已经指向下一项的 schedule_context.polling_index。
    GlobalTensor<uint64_t> selection;
    selection.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(readyWs));
    TBuf<TPosition::VECIN> selectionBuf;
    pipe.InitBuffer(selectionBuf, ONE_BLK_SIZE);
    LocalTensor<uint64_t> mb = selectionBuf.Get<uint64_t>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
    DataCopyPadExtParams<uint64_t> pad{false, 0, 0, 0};
    DataCopyPad(mb, selection, cp, pad);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    ctx.curMicroBatchID = mb.GetValue(0);
    pipe.Reset();
}

// ===================== expert_id 归一 + 握手回写 =====================
// 补位/失效标记:>= sort 的 expertStart_(1000000),排序后落到末尾并被 mask 判据剔除。
constexpr int32_t MASK_SENTINEL = INT32_MAX;

class FfnWbPrepareArch35 {
public:
    __aicore__ inline FfnWbPrepareArch35(){};

    // flatIdsWs:归一后的扁平 expert_id 缓冲(长度 totalLen)。
    // rowsPerLoop 由 host 按运行时 UB 容量反推(见 tiling 的 preparePerLoopRows),此处不设任何容量常数。
    __aicore__ inline void Init(GM_ADDR flatIdsWs, const ScheduleContextInfo* contextInfo, TPipe* pipe,
                                int64_t totalLen, int64_t rowsPerLoop, int64_t perLoopElements, int64_t activeRows,
                                GM_ADDR readyWs = nullptr)
    {
        contextInfo_ = contextInfo;
        pipe_ = pipe;
        totalLen_ = totalLen;
        bsk_ = static_cast<int64_t>(contextInfo_->BS) * contextInfo_->K;
        F_ = NUM_TWO + bsk_;
        // 每 session 行在扁平缓冲中的跨度(RECV 含补位;NORM 无补位时即为 bsk)
        rowSpan_ = (activeRows > 0) ? (totalLen_ / activeRows) : bsk_;
        // 异步 ready session 可不连续，因此每轮处理一行，先查快照再决定是否读取并清理该行。
        // 这里的 1 是选择粒度，不是核数或硬件容量；同步/NORM 仍使用 host 给出的批量行数。
        rowsPerLoop_ = contextInfo_->asyncRecv ? 1 : ((rowsPerLoop > 0) ? rowsPerLoop : 1);
        if (contextInfo_->asyncRecv) {
            readyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(readyWs));
            pipe_->InitBuffer(readyBuf_, ONE_BLK_SIZE);
        }
        if (rowsPerLoop_ > activeRows) {
            rowsPerLoop_ = activeRows > 0 ? activeRows : 1;
        }
        // 按**运行时实际块数**把 session 行分段,各核只处理自己那段:
        // 各段写入 flatIds 的区间互不重叠,握手回写也按行分离,故无需跨核同步。
        const int64_t coreNum = GetFlatAivNum();
        const int64_t perCore = (activeRows + coreNum - 1) / coreNum;
        rowBegin_ = GetFlatAivIdx() * perCore;
        rowEnd_ = rowBegin_ + perCore;
        if (rowEnd_ > activeRows) {
            rowEnd_ = activeRows;
        }
        if (rowBegin_ > activeRows) {
            rowBegin_ = activeRows;
        }
        if (rowsPerLoop_ > (rowEnd_ - rowBegin_) && (rowEnd_ - rowBegin_) > 0) {
            rowsPerLoop_ = rowEnd_ - rowBegin_;
        }
        flatIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(flatIdsWs), totalLen_ > 0 ? totalLen_ : 1);
        perLoopElements_ = Min(perLoopElements, rowsPerLoop_ * rowSpan_);
        pipe_->InitBuffer(que_, 1, Align(perLoopElements_ * sizeof(int32_t), BLOCK_BYTES));
        pipe_->InitBuffer(clrBuf_, Align(rowsPerLoop_ * BLOCK_BYTES, BLOCK_BYTES));
    }

    // NORM: totalLen_ 已按 outNum*BS*K 下发；sort 也只消费这个有效前缀。
    // 下面保留前缀边界保护，避免此类被复用于更大长度时搬入尾部旧 ID。
    __aicore__ inline void ProcessNorm(GM_ADDR expertIdsBuf)
    {
        if (rowEnd_ <= rowBegin_) {
            return;
        }
        GlobalTensor<int32_t> srcGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIdsBuf), totalLen_);
        const int64_t perLoopElems = perLoopElements_;
        const int64_t beginElem = rowBegin_ * rowSpan_;
        const int64_t endElem = (rowEnd_ * rowSpan_ > totalLen_) ? totalLen_ : rowEnd_ * rowSpan_;
        const int64_t validEnd = static_cast<int64_t>(contextInfo_->outNum) * bsk_;
        for (int64_t off = beginElem; off < endElem;) {
            int64_t n = ((endElem - off) > perLoopElems) ? perLoopElems : (endElem - off);
            // 在有效前缀的边界拆开循环：每块要么全是输入，要么全是 MASK。
            // 这样即使 BS*K 不是 8 的倍数，也无需对未对齐的 UB 子地址做 Duplicate，
            // 且 outNum=0 时完全不读取 expert_ids_buf 中的旧内容。
            if (off < validEnd && n > validEnd - off) {
                n = validEnd - off;
            }
            LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();
            DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
            if (off < validEnd) {
                DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                DataCopyPad(buf, srcGm[off], cp, pad);
                SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            } else {
                Duplicate<int32_t>(buf, MASK_SENTINEL, n);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            }
            DataCopyPad(flatIdsGm_[off], buf, cp);
            // 下一块可能由 DMA 或向量写入，两个生产者都必须等当前搬出完成。
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            que_.FreeTensor(buf);
            off += n;
        }
    }

    // RECV:读入 ids 归一到 flatIds,随后与同步路径完全一致,就地清原描述符的 ids/flag。
    //   同步:读全部行,读入后立即清。
    //   异步:只处理快照选中的行(读入并清理);未选中行只在 flatIds 填无效值,不读也不清。
    // 清理时序:已确认异步生产者也不会用新数据覆盖本次正在消费的数据(槽位复用与
    // 本次处理窗口不重叠),因此清理无需延迟到 gather 之后,读入后立即执行即可。
    // rowsPerLoop_ 控制本轮驻留的 id/清零区，异步固定一行以支持不连续的选择集合。
    __aicore__ inline void ProcessRecv(GM_ADDR tokenInfoBuf)
    {
        if (rowEnd_ <= rowBegin_) {
            return;
        }
        if (rowSpan_ > perLoopElements_) {
            ProcessRecvLargeRow(tokenInfoBuf);
            return;
        }
        GlobalTensor<int32_t> tokenInfoGm;
        tokenInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenInfoBuf),
                                    contextInfo_->A * contextInfo_->M * F_);
        const int64_t base = contextInfo_->curMicroBatchID * F_;
        const int64_t rowStride = contextInfo_->M * F_; // 相邻 session 在 token_info 中的跨度
        LocalTensor<int32_t> clr = clrBuf_.Get<int32_t>();

        for (int64_t r0 = rowBegin_; r0 < rowEnd_; r0 += rowsPerLoop_) {
            const int64_t rows = ((rowEnd_ - r0) > rowsPerLoop_) ? rowsPerLoop_ : (rowEnd_ - r0);
            LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();

            // 未选中行即使含有看似合法的 expert_id，也属于未发布/迟到的数据，不能读取。
            // 仅在私有 workspace 中填 sentinel，保持原 session 下标与 gather 解码一致；
            // 该行不属于本次消费，禁止读取或清除其 ids、flag 或 layer_id。
            if (contextInfo_->asyncRecv && !IsSelected(r0)) {
                Duplicate<int32_t>(buf, MASK_SENTINEL, rowSpan_);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyExtParams masked{1, static_cast<uint32_t>(rowSpan_ * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(flatIdsGm_[r0 * rowSpan_], buf, masked);
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                que_.FreeTensor(buf);
                continue;
            }

            // 异步快照每次处理一个session；只读取已选中行的层号，未选中行已跳过。
            const int32_t layerOffset =
                contextInfo_->asyncRecv ? ReadLayerOffset(tokenInfoGm, base + r0 * rowStride) : 0;
            // 取 ids:每块 bsk 个 int32,块间跳过 flag/layer 与其余 micro batch;
            // 右侧补位由 DataCopyPad 按 BsKPaddingCount 填 MASK_SENTINEL,使每行在 UB 中占 rowSpan_
            DataCopyExtParams inParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(bsk_ * sizeof(int32_t)),
                                       static_cast<uint32_t>((rowStride - bsk_) * sizeof(int32_t)), 0, 0};
            DataCopyPadExtParams<int32_t> inPad{true, 0, static_cast<uint8_t>(contextInfo_->BsKPaddingCount),
                                                MASK_SENTINEL};
            DataCopyPad(buf, tokenInfoGm[base + r0 * rowStride + NUM_TWO], inParams, inPad);
            if (contextInfo_->asyncRecv) {
                EncodeLayerExperts(buf, rows * rowSpan_, layerOffset);
            } else {
                SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            }

            DataCopyExtParams outParams{static_cast<uint16_t>(1),
                                        static_cast<uint32_t>(rows * rowSpan_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(flatIdsGm_[r0 * rowSpan_], buf, outParams);

            // 握手回写前必须等上面的搬出真正读完 buf:下面要就地把 buf 覆盖成回写内容。
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            // flatIds 已搬出，排序使用私有 ids 副本；gather 仍从原 token_data 读取 token。
            // 原描述符 ids 不再被本次调用读取，按已确认的不覆盖契约可在此清理 ids/flag。
            // 同步与异步共用以下回写：先发 ids 失效，再发 flag 清零，不额外插入异步等待；
            // 末尾 MTE3_MTE2 仍需保留，保证搬出完成后才能复用本轮 UB。
            // 未选中行不属于本次消费，已在上面跳过，原始描述符保持不变。
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
    // layer_id is the second descriptor word. Reuse the snapshot scratch block:
    // IsSelected has already consumed its scalar before this DMA overwrites it.
    __aicore__ inline int32_t ReadLayerOffset(GlobalTensor<int32_t>& info, int64_t descriptorBase)
    {
        constexpr int64_t LAYER_WORD = offsetof(aicpu::FfnDataDesc, layer_id) / sizeof(int32_t);
        LocalTensor<int32_t> layer = readyBuf_.Get<int32_t>();
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(layer, info[descriptorBase + LAYER_WORD], cp, pad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        const int32_t id = layer.GetValue(0);
        // Invalid layer IDs cannot address an expert on this worker. Mask that
        // descriptor's tokens rather than overflowing or merging it into another layer.
        return static_cast<uint32_t>(id) < contextInfo_->layerNum ?
                   id * static_cast<int32_t>(contextInfo_->expertsPerLayer) :
                   -1;
    }

    __aicore__ inline void EncodeLayerExperts(const LocalTensor<int32_t>& buf, int64_t count, int32_t offset)
    {
        // Transform only the private sort key, never the original gather position.
        // Unsigned comparison rejects negative IDs, >=1000000 masks and padding
        // before addition. INT32_MAX must not be added to a layer offset.
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        constexpr uint32_t LANES = GetVecLen() / sizeof(uint32_t);
        const uint32_t limit = offset >= 0 ? contextInfo_->expertsPerLayer : 0;
        const uint32_t bias = offset >= 0 ? static_cast<uint32_t>(offset) : 0;
        uint32_t remain = static_cast<uint32_t>(count);
        const uint16_t repeats = static_cast<uint16_t>((count + LANES - 1) / LANES);
        __ubuf__ uint32_t* addr = reinterpret_cast<__ubuf__ uint32_t*>(buf.GetPhyAddr());
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint32_t> ids, encoded, masked, result;
            Reg::MaskReg active, valid;
            for (uint16_t i = 0; i < repeats; ++i) {
                active = Reg::UpdateMask<uint32_t>(remain);
                Reg::LoadAlign(ids, addr + i * LANES);
                Reg::Compares<uint32_t, CMPMODE::LT>(valid, ids, limit, active);
                Reg::Adds(encoded, ids, bias, valid);
                Reg::Duplicate(masked, static_cast<uint32_t>(MASK_SENTINEL), active);
                Reg::Select(result, encoded, masked, valid);
                Reg::StoreAlign(addr + i * LANES, result, active);
            }
        }
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    }

    // 大行不能整行驻留 UB。按 Host 给出的元素容量分块，始终保持 workspace 的
    // 行跨度和原始 token 位置不变；最后一块才填 padding，且不把 padding 写回描述符。
    __aicore__ inline void ProcessRecvLargeRow(GM_ADDR tokenInfoBuf)
    {
        GlobalTensor<int32_t> info;
        info.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenInfoBuf));
        const int64_t stride = static_cast<int64_t>(contextInfo_->M) * F_;
        LocalTensor<int32_t> flag = clrBuf_.Get<int32_t>();
        for (int64_t row = rowBegin_; row < rowEnd_; ++row) {
            const bool selected = !contextInfo_->asyncRecv || IsSelected(row);
            const int64_t base = row * stride + contextInfo_->curMicroBatchID * F_;
            // 大行的所有ID分块共用同一个层偏移，不按块重新读取生产者描述符。
            const int32_t layerOffset = selected && contextInfo_->asyncRecv ? ReadLayerOffset(info, base) : 0;
            for (int64_t off = 0; off < rowSpan_; off += perLoopElements_) {
                const int64_t n = Min(perLoopElements_, rowSpan_ - off);
                const int64_t take = off < bsk_ ? Min(n, bsk_ - off) : 0;
                LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();
                if (selected && take > 0) {
                    DataCopyExtParams cp{1, static_cast<uint32_t>(take * sizeof(int32_t)), 0, 0, 0};
                    DataCopyPadExtParams<int32_t> pad{true, 0, static_cast<uint8_t>(n - take), MASK_SENTINEL};
                    DataCopyPad(buf, info[base + NUM_TWO + off], cp, pad);
                    if (contextInfo_->asyncRecv) {
                        EncodeLayerExperts(buf, n, layerOffset);
                    } else {
                        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
                    }
                } else {
                    Duplicate<int32_t>(buf, MASK_SENTINEL, n);
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                }
                DataCopyExtParams out{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(flatIdsGm_[row * rowSpan_ + off], buf, out);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                if (selected && take > 0) {
                    Duplicate<int32_t>(buf, MASK_SENTINEL, n);
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                    DataCopyExtParams clear{1, static_cast<uint32_t>(take * sizeof(int32_t)), 0, 0, 0};
                    DataCopyPad(info[base + NUM_TWO + off], buf, clear);
                }
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                que_.FreeTensor(buf);
            }
            // 仅选中行可清 flag，且必须等所有 ID 分块复制/清理完成；layer 字段不动。
            if (selected) {
                Duplicate<int32_t>(flag, 0, BLOCK_BYTES / sizeof(int32_t));
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyExtParams clearFlag{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(info[base], flag, clearFlag);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        }
    }

    // 从 workspace 快照读取，而不是从 token_info 读取实时 flag。
    // (row + 1) 跳过头块；每个 session 的 flag 独占一个块，与 Waiter 的 DMA padding 一致。
    // MTE2_S 确保 DMA 已完成，随后标量 GetValue(0) 才能用于分支判断。
    __aicore__ inline bool IsSelected(int64_t row)
    {
        LocalTensor<int32_t> flag = readyBuf_.Get<int32_t>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(flag, readyGm_[(row + 1) * (ONE_BLK_SIZE / sizeof(int32_t))], cp, pad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        return flag.GetValue(0) == 1;
    }

    GlobalTensor<int32_t> readyGm_;
    TBuf<TPosition::VECIN> readyBuf_;
    const ScheduleContextInfo* contextInfo_ = nullptr;
    TPipe* pipe_ = nullptr;
    GlobalTensor<int32_t> flatIdsGm_;
    TQue<QuePosition::VECCALC, 1> que_;
    TBuf<QuePosition::VECCALC> clrBuf_;
    int64_t totalLen_ = 0;
    int64_t bsk_ = 0;
    int64_t rowSpan_ = 0;
    int64_t rowsPerLoop_ = 0;
    int64_t perLoopElements_ = 0;
    int64_t rowBegin_ = 0;
    int64_t rowEnd_ = 0;
    int64_t F_ = 0;
};
} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_A5_PREPARE_H
