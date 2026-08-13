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
 * \file ffn_wb_a5_group_list.h
 * \brief arch35(A5) group_list:从有序 expert_id 直接压出 [expert_id, tokenNum] 稠密行。
 *
 * 序列已按 expert_id 升序,故"与前一个不同"即某专家的首元素、"与后一个不同"即其末元素。
 * 两张掩码各做一次 GatherMask 压缩,得到**等长**的首下标列与末下标列,逐元素相减即每专家
 * token 数。全程向量:不按 expert_id 散射写,也不需要 GM 直方图与跨核原子累加。
 *
 * 为何不用 SIMT 按 expert_id 散射计数:SIMT 访问 __local_mem__ 被限制在 UB 的低 8KB
 * (固定值,与 asc_vf_call 的 dim3 线程数无关,实测把线程数减半阈值不动),按 expert_id
 * 写 slot[cur] 在专家数超过 2048 时越界;越界写静默生效,会踩坏同一 UB 上的其它数据
 * (曾表现为编译器溢出到栈上的 totalValid 被清零,后续 gather 整段被跳过而硬件不报异常)。
 *
 * 分块由运行时 UB 反推(见 Init 中的 chunkElements_),不写死块长;块间用一个部分游程
 * (pendingId/pendingCnt)承接,故任意长度的游程都能跨块正确合并。
 *
 * 写出严格走整段搬运:行在 UB 拼好后按 host 给的块长 DataCopyPad 一次搬出,未使用的专家行
 * 用 Duplicate 填 0 随块带出。**不对 GM 做标量 SetValue**——历史缺陷正是"逐行 SetValue +
 * 末尾一次 SINGLE_CACHE_LINE 刷回"导致只有首条 cache line(4 行)落盘。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_GROUP_LIST_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_GROUP_LIST_H
#include "ffn_wb_a5_context.h"

namespace FfnWbBatchingArch35 {
using namespace AscendC;

constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64; // CompareScalar 的 repeat 粒度(ISA)
constexpr int64_t ONE_REPEAT_BLOCKS = 8;       // 一个 repeat 覆盖 8 个 32B 块(ISA)
constexpr int64_t ONE_BYTE_BITS = 8;
constexpr int32_t SENTINEL_ID = -1; // 哨兵:有效 expert_id 恒 >= 0,故它与谁都不等

// 单元素在 UB 上的占用:cur/prev/next/idx/diff/diffF/outStart/outEnd/outId/cnt 十路 int32,
// 外加首尾两张比较掩码(各 1 bit/元素)。分块大小由 UB 除以它得到,不是拍出来的常数。
constexpr int64_t CHUNK_VEC_NUM = 10;
constexpr int64_t CHUNK_MASK_NUM = 2;
constexpr int64_t CHUNK_BUF_NUM = CHUNK_VEC_NUM + CHUNK_MASK_NUM + 1; // +1:行拼装区

struct A5GroupListParam {
    int64_t rowsPerLoop = 0; // 写出时每块拼多少行(每行 16B)
};

// 游程编码用到的 UB 分道集合:一次取好,三个子步骤共用,免得每个函数各取一遍。
struct A5RunLanes {
    LocalTensor<int32_t> cur;      // ids[begin, begin+len)
    LocalTensor<int32_t> prev;     // ids[begin-1, begin+len-1)
    LocalTensor<int32_t> next;     // ids[begin+1, begin+len]
    LocalTensor<int32_t> idx;      // 块内局部下标 0,1,2,...
    LocalTensor<int32_t> diff;     // 相邻差(整型)
    LocalTensor<float> diffF;      // 相邻差(浮点,供比较指令)
    LocalTensor<int32_t> outStart; // 压缩后的游程首下标
    LocalTensor<int32_t> outEnd;   // 压缩后的游程末下标
    LocalTensor<int32_t> outId;    // 压缩后的游程 expert_id
    LocalTensor<int32_t> cnt;      // 每游程 token 数
    LocalTensor<uint32_t> maskS;   // 游程起点掩码
    LocalTensor<uint32_t> maskE;   // 游程终点掩码
};

class FfnWbA5GroupList {
public:
    __aicore__ inline FfnWbA5GroupList(){};

    // sortedIdsWs:有序 expert_id(长度为排序后的有效数);groupList:输出 [expertNum, 2]
    __aicore__ inline void Init(GM_ADDR sortedIdsWs, GM_ADDR groupList, const A5GroupListParam &param,
                                const ScheduleContextInfo *ctx, TPipe *pipe)
    {
        param_ = param;
        pipe_ = pipe;
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        vecId_ = GetBlockIdx() * subNum + GetSubBlockIdx();
        expertNum_ = static_cast<int64_t>(ctx->expertNum);
        totalLen_ = ctx->validGatherIdxLength;
        // 游程压缩是串行语义(相邻元素比较跨越切片边界),由 0 号核一趟流式做完;
        // 其余核不参与,也就不占 UB——它们直接进入下一相位。
        if (vecId_ != 0) {
            return;
        }

        idsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sortedIdsWs), (totalLen_ > 0) ? totalLen_ : 1);
        groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList), expertNum_ * NUM_TWO);

        // 分块长度由运行时 UB 反推:先扣掉行拼装区与各缓冲的块对齐余量,再按每元素占用摊分,
        // 最后向下取整到比较指令的 repeat 粒度。
        const int64_t rowBytes = param_.rowsPerLoop * NUM_TWO * static_cast<int64_t>(sizeof(int64_t));
        const int64_t reserved = rowBytes + CHUNK_BUF_NUM * ONE_BLK_SIZE;
        const int64_t avail = (ctx->ubSize > reserved) ? (ctx->ubSize - reserved) : 0;
        const int64_t bytesPerElem = CHUNK_VEC_NUM * static_cast<int64_t>(sizeof(int32_t)) + CHUNK_MASK_NUM;
        int64_t chunk = avail / bytesPerElem / ONE_REPEAT_COMPARE_NUM * ONE_REPEAT_COMPARE_NUM;
        if (chunk < ONE_REPEAT_COMPARE_NUM) {
            chunk = ONE_REPEAT_COMPARE_NUM;
        }
        // 数据装得下就不必开满,按实际长度收窄(仍保持 repeat 粒度)。
        const int64_t needed = Ceil((totalLen_ > 0) ? totalLen_ : 1, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
        chunkElements_ = (chunk > needed) ? needed : chunk;

        const int64_t vecBytes = chunkElements_ * static_cast<int64_t>(sizeof(int32_t)) + ONE_BLK_SIZE;
        pipe_->InitBuffer(curBuf_, vecBytes);
        pipe_->InitBuffer(prevBuf_, vecBytes);
        pipe_->InitBuffer(nextBuf_, vecBytes);
        pipe_->InitBuffer(idxBuf_, vecBytes);
        pipe_->InitBuffer(diffBuf_, vecBytes);
        pipe_->InitBuffer(diffFBuf_, vecBytes);
        pipe_->InitBuffer(outStartBuf_, vecBytes);
        pipe_->InitBuffer(outEndBuf_, vecBytes);
        pipe_->InitBuffer(outIdBuf_, vecBytes);
        pipe_->InitBuffer(cntBuf_, vecBytes);
        const int64_t maskBytes = chunkElements_ / ONE_BYTE_BITS + ONE_BLK_SIZE;
        pipe_->InitBuffer(maskSBuf_, maskBytes);
        pipe_->InitBuffer(maskEBuf_, maskBytes);
        pipe_->InitBuffer(rowBuf_, rowBytes + ONE_BLK_SIZE);

        // 块内局部下标序列与块无关,只生成一次;计数用差值,局部下标与全局下标同解。
        ArithProgression<int32_t>(idxBuf_.Get<int32_t>(), 0, 1, static_cast<int32_t>(chunkElements_));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        if (vecId_ != 0) {
            return;
        }
        LocalTensor<int64_t> rows = rowBuf_.Get<int64_t>();
        LocalTensor<int32_t> rows32 = rows.template ReinterpretCast<int32_t>();
        ClearRows(rows32);

        int64_t outIdx = 0; // 已搬出的行数
        int64_t fill = 0;   // 当前块已拼的行数
        if (totalLen_ > 0) {
            EncodeRuns(rows, outIdx, fill);
        }
        FlushRows(rows, outIdx, fill);
        // 未用到的专家:整块 [0,0] 补齐到 expertNum 行(块在上一次搬出后已清零)。
        while (outIdx < expertNum_) {
            const int64_t n = Min(param_.rowsPerLoop, expertNum_ - outIdx);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams cpOut{static_cast<uint16_t>(1), static_cast<uint32_t>(n * NUM_TWO * sizeof(int64_t)), 0,
                                    0, 0};
            DataCopyPad(groupListGm_[outIdx * NUM_TWO], rows, cpOut);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            outIdx += n;
        }
    }

private:
    // 流式游程编码:逐块比较"与前一个/与后一个是否不同",压出完整游程的首尾下标。
    // 一次取好全部分道视图。
    __aicore__ inline A5RunLanes MakeLanes()
    {
        A5RunLanes ln;
        ln.cur = curBuf_.Get<int32_t>();
        ln.prev = prevBuf_.Get<int32_t>();
        ln.next = nextBuf_.Get<int32_t>();
        ln.idx = idxBuf_.Get<int32_t>();
        ln.diff = diffBuf_.Get<int32_t>();
        ln.diffF = diffFBuf_.Get<float>();
        ln.outStart = outStartBuf_.Get<int32_t>();
        ln.outEnd = outEndBuf_.Get<int32_t>();
        ln.outId = outIdBuf_.Get<int32_t>();
        ln.cnt = cntBuf_.Get<int32_t>();
        ln.maskS = maskSBuf_.Get<uint32_t>();
        ln.maskE = maskEBuf_.Get<uint32_t>();
        return ln;
    }

    // 载入一块的 cur/prev/next 三路,并按边界置哨兵:块首一律记为游程起点(缺的那段由
    // pendingCnt 承接),全局末元素一律记为游程终点。
    __aicore__ inline void LoadChunk(A5RunLanes &ln, const DataCopyPadExtParams<int32_t> &pad, int64_t begin,
                                     int64_t len)
    {
        DataCopyExtParams cpLen{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(ln.cur, idsGm_[begin], cpLen, pad);
        DataCopyPad(ln.prev, idsGm_[begin - 1], cpLen, pad);
        const int64_t nextLen = Min(len, totalLen_ - begin - 1);
        if (nextLen > 0) {
            DataCopyExtParams cpNext{static_cast<uint16_t>(1), static_cast<uint32_t>(nextLen * sizeof(int32_t)), 0, 0,
                                     0};
            DataCopyPad(ln.next, idsGm_[begin + 1], cpNext, pad);
        }
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        // 块首一律记为游程起点:它若处在游程中间,缺的那一段正由 pendingCnt 承接。
        ln.prev.SetValue(0, SENTINEL_ID);
        if (nextLen < len) {
            ln.next.SetValue(len - 1, SENTINEL_ID); // 全局末元素必是游程终点
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    }

    // 出首尾两张掩码并压缩成等长的三列,返回本块内完整游程数。
    __aicore__ inline int64_t BuildAndCompact(A5RunLanes &ln, int64_t len, uint64_t &startNum, uint64_t &endNum)
    {
        const int64_t cmpLen = Ceil(len, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
        // 比较指令只吃浮点,故差值转 float 再与 0 比;差值幅度不超过专家数上界,转换精确。
        Sub(ln.diff, ln.cur, ln.prev, static_cast<int32_t>(len));
        PipeBarrier<PIPE_V>();
        Cast(ln.diffF, ln.diff, RoundMode::CAST_ROUND, static_cast<int32_t>(len));
        PipeBarrier<PIPE_V>();
        CompareScalar(ln.maskS.template ReinterpretCast<uint8_t>(), ln.diffF, static_cast<float>(0), CMPMODE::NE,
                      static_cast<int32_t>(cmpLen));
        PipeBarrier<PIPE_V>();
        Sub(ln.diff, ln.next, ln.cur, static_cast<int32_t>(len));
        PipeBarrier<PIPE_V>();
        Cast(ln.diffF, ln.diff, RoundMode::CAST_ROUND, static_cast<int32_t>(len));
        PipeBarrier<PIPE_V>();
        CompareScalar(ln.maskE.template ReinterpretCast<uint8_t>(), ln.diffF, static_cast<float>(0), CMPMODE::NE,
                      static_cast<int32_t>(cmpLen));
        PipeBarrier<PIPE_V>();

        GatherMaskParams gmp;
        gmp.repeatTimes = 1;
        gmp.src0BlockStride = 1;
        gmp.src0RepeatStride = ONE_REPEAT_BLOCKS;
        gmp.src1RepeatStride = 0;
        uint64_t idNum = 0;
        GatherMask(ln.outStart, ln.idx, ln.maskS, true, static_cast<uint32_t>(len), gmp, startNum);
        GatherMask(ln.outEnd, ln.idx, ln.maskE, true, static_cast<uint32_t>(len), gmp, endNum);
        GatherMask(ln.outId, ln.cur, ln.maskE, true, static_cast<uint32_t>(len), gmp, idNum);
        PipeBarrier<PIPE_V>();

        // 在本块结束的游程即完整游程,其首尾下标在两列中一一对应(块首已强制成起点)。
        const int64_t runs = static_cast<int64_t>(endNum);
        if (runs > 0) {
            Sub(ln.cnt, ln.outEnd, ln.outStart, static_cast<int32_t>(runs));
            PipeBarrier<PIPE_V>();
            Adds(ln.cnt, ln.cnt, 1, static_cast<int32_t>(runs));
            PipeBarrier<PIPE_V>();
        }
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        return static_cast<int64_t>(endNum);
    }

    __aicore__ inline void EncodeRuns(const LocalTensor<int64_t> &rows, int64_t &outIdx, int64_t &fill)
    {
        A5RunLanes ln = MakeLanes();
        LocalTensor<int32_t> &cur = ln.cur;
        LocalTensor<int32_t> &outStart = ln.outStart;
        LocalTensor<int32_t> &outId = ln.outId;
        LocalTensor<int32_t> &cnt = ln.cnt;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};

        // 首元素单独起头:此后每块的起点都有前驱,ln.prev 可整块从 ids[begin-1] 读入(免去
        // 4 字节偏移的非对齐视图)。它本身必是某游程的首元素,故作为初始的部分游程。
        DataCopyExtParams cpOne{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(cur, idsGm_, cpOne, pad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        int64_t pendingId = static_cast<int64_t>(cur.GetValue(0));
        int64_t pendingCnt = 1;

        int64_t begin = 1;
        while (begin < totalLen_) {
            const int64_t len = Min(chunkElements_, totalLen_ - begin);
            LoadChunk(ln, pad, begin, len);

            uint64_t startNum = 0;
            uint64_t endNum = 0;
            const int64_t runs = BuildAndCompact(ln, len, startNum, endNum);

            // 承接上一块的部分游程:块首若换了专家,说明那段已经完整,先单独成行。
            // (块中途留下的 pending 必与本块首元素同专家,故此判定只会在起头那一个元素上生效。)
            if (pendingCnt > 0 && static_cast<int64_t>(cur.GetValue(0)) != pendingId) {
                AppendRow(rows, outIdx, fill, pendingId, pendingCnt);
                pendingCnt = 0;
            }
            if (pendingCnt == 0) {
                pendingId = static_cast<int64_t>(cur.GetValue(0)); // 本块首元素开启的游程
            }

            if (runs == 0) {
                pendingCnt += len; // 整块仍落在同一个游程内,继续往后并
            } else {
                for (int64_t j = 0; j < runs; j++) {
                    int64_t num = static_cast<int64_t>(cnt.GetValue(j));
                    if (j == 0) {
                        num += pendingCnt; // 首个游程接上前面块里的部分
                    }
                    AppendRow(rows, outIdx, fill, static_cast<int64_t>(outId.GetValue(j)), num);
                }
                // 块尾若停在游程中间,把这段残长转为下一块的部分游程。
                if (static_cast<int64_t>(startNum) > runs) {
                    pendingCnt = len - static_cast<int64_t>(outStart.GetValue(runs));
                    pendingId = static_cast<int64_t>(cur.GetValue(len - 1));
                } else {
                    pendingCnt = 0;
                }
            }
            begin += len;
        }
        // 全局末元素被哨兵判为游程终点,故循环出来时通常已无残留;
        // 唯独 totalLen_ == 1(循环体没进过)时,起头的那个游程要在这里写出。
        if (pendingCnt > 0) {
            AppendRow(rows, outIdx, fill, pendingId, pendingCnt);
        }
    }

    __aicore__ inline void AppendRow(const LocalTensor<int64_t> &rows, int64_t &outIdx, int64_t &fill, int64_t expertId,
                                     int64_t tokenNum)
    {
        if (outIdx + fill >= expertNum_) {
            return; // 游程数不会超过专家数;越界即异常输入,按契约不再写
        }
        rows.SetValue(fill * NUM_TWO, expertId);
        rows.SetValue(fill * NUM_TWO + 1, tokenNum);
        fill++;
        if (fill == param_.rowsPerLoop) {
            FlushRows(rows, outIdx, fill);
        }
    }

    __aicore__ inline void FlushRows(const LocalTensor<int64_t> &rows, int64_t &outIdx, int64_t &fill)
    {
        if (fill <= 0) {
            return;
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(fill * NUM_TWO * sizeof(int64_t)), 0, 0,
                             0};
        DataCopyPad(groupListGm_[outIdx * NUM_TWO], rows, cp);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        outIdx += fill;
        fill = 0;
        // 复位缓冲:不复位的话,下一块未填满的行会带出上一块的残留值。
        ClearRows(rows.template ReinterpretCast<int32_t>());
    }

    // 整块清零:起点为缓冲区首地址,满足向量指令的对齐要求。
    __aicore__ inline void ClearRows(const LocalTensor<int32_t> &rows32)
    {
        // 清零是向量写:既要等前面的标量填充,也要等上一块的搬出真正读完。
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        Duplicate<int32_t>(rows32, 0, static_cast<int32_t>(param_.rowsPerLoop * NUM_FOUR));
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }

    A5GroupListParam param_;
    TPipe *pipe_ = nullptr;

    GlobalTensor<int32_t> idsGm_;
    GlobalTensor<int64_t> groupListGm_;

    TBuf<TPosition::VECCALC> curBuf_;
    TBuf<TPosition::VECCALC> prevBuf_;
    TBuf<TPosition::VECCALC> nextBuf_;
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> diffBuf_;
    TBuf<TPosition::VECCALC> diffFBuf_;
    TBuf<TPosition::VECCALC> outStartBuf_;
    TBuf<TPosition::VECCALC> outEndBuf_;
    TBuf<TPosition::VECCALC> outIdBuf_;
    TBuf<TPosition::VECCALC> cntBuf_;
    TBuf<TPosition::VECCALC> maskSBuf_;
    TBuf<TPosition::VECCALC> maskEBuf_;
    TBuf<TPosition::VECCALC> rowBuf_;

    int64_t vecId_ = 0;
    int64_t expertNum_ = 0;
    int64_t totalLen_ = 0;
    int64_t chunkElements_ = 0;
};

} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_A5_GROUP_LIST_H
