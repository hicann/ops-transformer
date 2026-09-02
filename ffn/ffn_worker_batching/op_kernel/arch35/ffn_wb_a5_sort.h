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
 * \file ffn_wb_a5_sort.h
 * \brief phase1:段内排序 → 段间归并 → 拆包,产出有序 expert_id 与 gather_idx。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_SORT_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_SORT_H
#include "ffn_wb_a5_context.h"
namespace FfnWbBatchingArch35 {
using namespace AscendC;

// ===================== 段内排序 =====================
// 段内排序的切分参数,全部由 host tiling 计算后下发。
struct A5SortSegParam {
    int64_t segNum = 0;         // 序列被切成多少段
    int64_t perSegElements = 0; // 每段元素数(末段可不足)
    int64_t totalElements = 0;  // 序列总元素数
    int64_t expertStart = 0;    // 有效 expert_id 的上界:>= 该值者视为被 mask
    int64_t sortLenPerSeg = 0;  // 每段 proposal 对区的 float 个数(host 按对齐后段长算)
};

class FfnWbA5SegSort {
public:
    __aicore__ inline FfnWbA5SegSort(){};

    // flatIdsWs   : 归一后的扁平 expert_id(int32)
    // pairWs      : 段内排序结果的 proposal 对区(float),按段 slice
    // cntWs       : 每段有效元素数(int32),每段占一个数据块槽位
    __aicore__ inline void Init(GM_ADDR flatIdsWs, GM_ADDR pairWs, GM_ADDR cntWs, const A5SortSegParam &param,
                                const ScheduleContextInfo *ctx, TPipe *pipe)
    {
        param_ = param;
        ctx_ = ctx;
        pipe_ = pipe;
        // 认领段按**真实向量核编号**:一个 block 下挂多个 AIV 子核时 GetBlockIdx() 相同,
        // 只用它认领会让同一段算两遍、另一段没人算。
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        vecId_ = GetBlockIdx() * subNum + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * subNum;

        flatIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flatIdsWs), param_.totalElements);
        pairGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pairWs), param_.segNum * param_.sortLenPerSeg);
        cntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cntWs),
                               param_.segNum * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t))));

        // UB:输入段(int32 id + int32 原下标) / proposal 对区 / 排序临时区
        // 输入区按比较指令的 repeat 粒度对齐：CompareScalar 以 ONE_REPEAT_COMPARE_NUM 为单位处理，
        // 传入的元素数向上取整后可能越过段长，缓冲区需按同一粒度留足，避免读写越界。
        const int64_t alignSeg = Ceil(param_.perSegElements, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
        pipe_->InitBuffer(inQue_, 1, alignSeg * NUM_TWO * sizeof(int32_t) + ONE_BLK_SIZE);
        pipe_->InitBuffer(concatBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(tmpBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(sortedBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(maskBuf_, alignSeg * sizeof(uint32_t) + ONE_BLK_SIZE);
    }

    __aicore__ inline void Process()
    {
        for (int64_t seg = vecId_; seg < param_.segNum; seg += vecNum_) {
            ProcessOneSeg(seg);
        }
    }

private:
    // 取负 + 判定有效位。取负让降序排序等价于 expert_id 升序;判定阈值同样取负后比较,
    // 于是"有效"= 键 > -expertStart。这一段是纯规则的逐元素向量运算,用 MicroAPI 写。
    __aicore__ inline void PrepareKeys(const LocalTensor<float> &keys, const LocalTensor<uint32_t> &maskBits,
                                       int64_t count)
    {
        const uint16_t repeatTimes = static_cast<uint16_t>(Ceil(count, FLOAT_REG_ELEMENTS));
        uint32_t remain = static_cast<uint32_t>(count);
        __local_mem__ float *keyAddr = reinterpret_cast<__local_mem__ float *>(keys.GetPhyAddr());
        const float negOne = -1.0f;

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg loopMask;
            MicroAPI::RegTensor<float> keyReg;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                loopMask = MicroAPI::UpdateMask<float>(remain);
                MicroAPI::DataCopy(keyReg, keyAddr + i * FLOAT_REG_ELEMENTS);
                MicroAPI::Muls(keyReg, keyReg, negOne, loopMask);
                MicroAPI::DataCopy(keyAddr + i * FLOAT_REG_ELEMENTS, keyReg, loopMask);
            }
        }
        PipeBarrier<PIPE_V>();
        // 有效位:键 > -expertStart(即原 expert_id < expertStart)。比较按 64 元素粒度对齐。
        LocalTensor<uint8_t> maskU8 = maskBits.template ReinterpretCast<uint8_t>();
        CompareScalar(maskU8, keys, static_cast<float>(-param_.expertStart), CMPMODE::GT,
                      Ceil(count, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessOneSeg(int64_t seg)
    {
        const int64_t begin = seg * param_.perSegElements;
        const int64_t len =
            (begin >= param_.totalElements) ? 0 : Min(param_.perSegElements, param_.totalElements - begin);

        LocalTensor<int32_t> inLocal = inQue_.AllocTensor<int32_t>();
        int64_t validCnt = 0;
        validCnt = SortOneSegBody(inLocal, seg, begin, len);

        // 本段有效数写入计数区:跨核可见的交换一律经 UB + DataCopyPad,不用 GM 标量写。
        LocalTensor<int32_t> cntLocal = inLocal;
        cntLocal.SetValue(0, static_cast<int32_t>(validCnt));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams cpCnt{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(cntGm_[seg * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t)))], cntLocal, cpCnt);
        // 下一段会用 MTE2 重新载入、用向量指令重写同一批缓冲:必须等本段的搬出真正读完。
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        inQue_.FreeTensor(inLocal);
    }

    // 段内排序主体:压掉被 mask 的 token,补尾到 Sort32 粒度后 Concat+Sort,结果落 pairGm_。
    // 返回本段的有效元素数。
    __aicore__ inline int64_t SortOneSegBody(const LocalTensor<int32_t> &inLocal, int64_t seg, int64_t begin,
                                             int64_t len)
    {
        int64_t validCnt = 0;
        if (len > 0) {
            DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(inLocal, flatIdsGm_[begin], cp, pad);

            // 原下标:排序的 payload。段内以全局下标编号,归并后即 gather_idx。
            const int64_t alignSeg = Ceil(param_.perSegElements, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
            LocalTensor<int32_t> idxLocal = inLocal[alignSeg];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            ArithProgression<int32_t>(idxLocal, static_cast<int32_t>(begin), 1, len);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> keys = inLocal.template ReinterpretCast<float>();
            Cast(keys, inLocal, RoundMode::CAST_ROUND, len);
            PipeBarrier<PIPE_V>();

            LocalTensor<uint32_t> maskBits = maskBuf_.Get<uint32_t>();
            PrepareKeys(keys, maskBits, len);

            // 压掉被 mask 的 token:键与下标用同一套有效位压缩,压缩后个数即本段有效数。
            uint64_t rsvd = 0;
            GatherMaskParams gp;
            gp.repeatTimes = 1;
            gp.src0BlockStride = 1;
            gp.src0RepeatStride = BLOCKS_PER_REPEAT;
            gp.src1RepeatStride = 0;
            GatherMask(keys, keys, maskBits, true, static_cast<uint32_t>(len), gp, rsvd);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint32_t> idxU32 = idxLocal.template ReinterpretCast<uint32_t>();
            GatherMask(idxU32, idxU32, maskBits, true, static_cast<uint32_t>(len), gp, rsvd);
            PipeBarrier<PIPE_V>();
            validCnt = static_cast<int64_t>(rsvd);

            if (validCnt > 0) {
                // 尾部补最小值到 Sort32 粒度:补位在降序中落到末尾,不影响有效元素次序。
                const int64_t alignCnt = Ceil(validCnt, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
                const int64_t tailNum = validCnt % ONE_REPEAT_SORT_NUM;
                if (tailNum > 0) {
                    // 补尾必须从**对齐的起点**发起:向量指令要求 UB 地址按数据块对齐,
                    // 直接从 keys[validCnt] 开始会触发 "VEC access UB not aligned" 硬件异常。
                    // 故起点退到本 repeat 的开头,用位掩码只写尾部那几个元素。
                    uint64_t maskBits = UINT64_MAX << tailNum;
                    maskBits &= (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
                    uint64_t dupMask[NUM_TWO] = {maskBits, 0};
                    Duplicate(keys[validCnt - tailNum], SORT_FILL_VALUE, dupMask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
                    PipeBarrier<PIPE_V>();
                }
                // Concat/Sort 的各 tensor 不允许地址重叠:concat 结果、临时区、排序结果各占一块。
                LocalTensor<float> concatLocal = concatBuf_.Get<float>();
                LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
                LocalTensor<float> sortedLocal = sortedBuf_.Get<float>();
                Concat(concatLocal, keys, tmpLocal, alignCnt / ONE_REPEAT_SORT_NUM);
                PipeBarrier<PIPE_V>();
                Sort<float, true>(sortedLocal, concatLocal, idxU32, tmpLocal, alignCnt / ONE_REPEAT_SORT_NUM);
                PipeBarrier<PIPE_V>();

                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyExtParams cpOut{static_cast<uint16_t>(1),
                                        static_cast<uint32_t>(GetSortLen<float>(alignCnt) * sizeof(float)), 0, 0, 0};
                DataCopyPad(pairGm_[seg * param_.sortLenPerSeg], sortedLocal, cpOut);
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            }
        }
        return validCnt;
    }

private:
    // 一个向量寄存器容纳的 fp32 个数,以及一次 repeat 覆盖的数据块数/比较元素数。
    static constexpr int64_t FLOAT_REG_ELEMENTS = 64;
    static constexpr int64_t BLOCKS_PER_REPEAT = 8;
    static constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;
    // Duplicate 的目的地块内步长与 repeat 间步长(按数据块计),取同仓 arch35 算子同款配置。
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;

    A5SortSegParam param_;
    const ScheduleContextInfo *ctx_ = nullptr;
    TPipe *pipe_ = nullptr;

    GlobalTensor<int32_t> flatIdsGm_;
    GlobalTensor<float> pairGm_;
    GlobalTensor<int32_t> cntGm_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TBuf<TPosition::VECCALC> concatBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> sortedBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;

    int64_t vecId_ = 0;
    int64_t vecNum_ = 0;
};

// ===================== 段间归并 =====================
// 归并的切分参数,由 host tiling 计算下发。
struct A5MergeParam {
    int64_t segNum = 0;             // 段数(= 第 0 轮的路数)
    int64_t sortLenPerSeg = 0;      // 每段 slot 的 float 个数
    int64_t oneLoopMaxElements = 0; // 单路单次载入 UB 的最大元素数
    int64_t rounds = 0;             // 归并轮数 = ceil(log4(segNum))
};

class FfnWbA5MrgSort {
public:
    __aicore__ inline FfnWbA5MrgSort(){};

    // wsA/wsB:两块等大的 proposal 对工作区(乒乓);cntWs:各段有效元素数
    __aicore__ inline void Init(GM_ADDR wsA, GM_ADDR wsB, GM_ADDR cntWs, const A5MergeParam &param, TPipe *pipe)
    {
        param_ = param;
        pipe_ = pipe;
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        vecId_ = GetBlockIdx() * subNum + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * subNum;

        const int64_t wsFloats = param_.segNum * param_.sortLenPerSeg;
        wsGm_[0].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsA), wsFloats);
        wsGm_[1].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsB), wsFloats);
        cntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cntWs),
                               param_.segNum * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t))));

        // UB:MRG_LIST_NUM 路输入 + 1 路输出,均按 proposal 对宽度分配
        const int64_t loopFloats = GetSortLen<float>(param_.oneLoopMaxElements);
        pipe_->InitBuffer(inQue_, 1, loopFloats * MRG_LIST_NUM * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(outQue_, 1, loopFloats * MRG_LIST_NUM * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(
            cntBuf_,
            Align(param_.segNum * (ONE_BLK_SIZE / sizeof(int32_t)), sizeof(int32_t)) * sizeof(int32_t) + ONE_BLK_SIZE);
    }

    // 跑完全部归并轮次;返回最终结果所在的工作区序号(0=wsA,1=wsB)。
    // 每轮内各组由不同向量核认领,轮与轮之间由调用方 SyncAll。
    __aicore__ inline int64_t ProcessRound(int64_t round, int64_t srcIdx)
    {
        LoadSegCounts();
        const int64_t groupStride = Pow4(round + 1); // 本轮一组覆盖的段数
        const int64_t listStride = Pow4(round);      // 组内相邻两路相隔的段数
        const int64_t groupNum = Ceil(param_.segNum, groupStride);

        for (int64_t g = vecId_; g < groupNum; g += vecNum_) {
            MergeOneGroup(g, groupStride, listStride, srcIdx);
        }
        return 1 - srcIdx;
    }

    // 最终一路的元素总数 = 各段有效数之和
    __aicore__ inline int64_t TotalValid()
    {
        LoadSegCounts();
        int64_t total = 0;
        for (int64_t s = 0; s < param_.segNum; s++) {
            total += SegCount(s);
        }
        return total;
    }

private:
    __aicore__ inline int64_t Pow4(int64_t e)
    {
        int64_t v = 1;
        for (int64_t i = 0; i < e; i++) {
            v *= MRG_LIST_NUM;
        }
        return v;
    }

    __aicore__ inline void LoadSegCounts()
    {
        if (cntLoaded_) {
            return;
        }
        cntLocal_ = cntBuf_.Get<int32_t>();
        const int64_t words = param_.segNum * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t)));
        DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(words * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(cntLocal_, cntGm_, cp, pad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        cntLoaded_ = true;
    }

    __aicore__ inline int64_t SegCount(int64_t seg)
    {
        if (seg >= param_.segNum) {
            return 0;
        }
        return static_cast<int64_t>(cntLocal_.GetValue(seg * (ONE_BLK_SIZE / sizeof(int32_t))));
    }

    // 第 r 轮中,以 seg 为首段、跨度 listStride 的那一路的元素数 = 其覆盖段的有效数之和
    __aicore__ inline int64_t ListLength(int64_t firstSeg, int64_t listStride)
    {
        int64_t sum = 0;
        for (int64_t s = firstSeg; s < firstSeg + listStride && s < param_.segNum; s++) {
            sum += SegCount(s);
        }
        return sum;
    }

    __aicore__ inline void MergeOneGroup(int64_t group, int64_t groupStride, int64_t listStride, int64_t srcIdx)
    {
        const int64_t baseSeg = group * groupStride;
        int64_t offsets[MRG_LIST_NUM];
        int64_t remains[MRG_LIST_NUM];
        int64_t listNum = 0;
        for (int64_t i = 0; i < MRG_LIST_NUM; i++) {
            const int64_t firstSeg = baseSeg + i * listStride;
            if (firstSeg >= param_.segNum) {
                break;
            }
            const int64_t len = ListLength(firstSeg, listStride);
            offsets[listNum] = firstSeg * param_.sortLenPerSeg;
            remains[listNum] = len;
            listNum++;
        }
        if (listNum == 0) {
            return;
        }

        LocalTensor<float> inLocal = inQue_.AllocTensor<float>();
        LocalTensor<float> outLocal = outQue_.AllocTensor<float>();
        const int64_t loopFloats = GetSortLen<float>(param_.oneLoopMaxElements);
        int64_t outOffset = baseSeg * param_.sortLenPerSeg;

        int64_t allRemain = 0;
        for (int64_t i = 0; i < listNum; i++) {
            allRemain += remains[i];
        }

        MergeStream(inLocal, outLocal, listNum, loopFloats, outOffset, allRemain, offsets, remains, srcIdx);

        inQue_.FreeTensor(inLocal);
        outQue_.FreeTensor(outLocal);
    }

    // 流式归并主循环:每轮从各路取一段填满 UB,MrgSort 后整段写出,直到各路耗尽。
    // 从各路各取一块填进 UB 输入区(已耗尽的路跳过),记下每块元素数与其对应的原路号。
    // 返回本轮真正参与归并的路数。
    __aicore__ inline int64_t FillLists(const LocalTensor<float> &inLocal, int64_t listNum, int64_t loopFloats,
                                        int64_t srcIdx, const int64_t (&offsets)[MRG_LIST_NUM],
                                        const int64_t (&remains)[MRG_LIST_NUM],
                                        LocalTensor<float> (&lists)[MRG_LIST_NUM], uint16_t (&counts)[MRG_LIST_NUM],
                                        int64_t (&liveMap)[MRG_LIST_NUM])
    {
        int64_t liveNum = 0;
        for (int64_t i = 0; i < listNum; i++) {
            const int64_t take = Min(param_.oneLoopMaxElements, remains[i]);
            if (take <= 0) {
                continue;
            }
            DataCopyExtParams cp{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(GetSortLen<float>(take) * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(inLocal[liveNum * loopFloats], wsGm_[srcIdx][offsets[i]], cp, pad);
            lists[liveNum] = inLocal[liveNum * loopFloats];
            counts[liveNum] = static_cast<uint16_t>(take);
            liveMap[liveNum] = i;
            liveNum++;
        }
        return liveNum;
    }

    __aicore__ inline void MergeStream(const LocalTensor<float> &inLocal, const LocalTensor<float> &outLocal,
                                       int64_t listNum, int64_t loopFloats, int64_t &outOffset, int64_t &allRemain,
                                       int64_t (&offsets)[MRG_LIST_NUM], int64_t (&remains)[MRG_LIST_NUM],
                                       int64_t srcIdx)
    {
        while (allRemain > 0) {
            // 载入各路当前块
            uint16_t counts[MRG_LIST_NUM] = {0, 0, 0, 0};
            LocalTensor<float> lists[MRG_LIST_NUM];
            int64_t liveMap[MRG_LIST_NUM];
            const int64_t liveNum =
                FillLists(inLocal, listNum, loopFloats, srcIdx, offsets, remains, lists, counts, liveMap);
            if (liveNum == 0) {
                break;
            }
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            uint32_t sortedNums[MRG_LIST_NUM] = {0, 0, 0, 0};
            if (liveNum == 1) {
                // 只剩一路:直接搬,不必归并
                DataCopy(outLocal, lists[0], Align(GetSortLen<float>(counts[0]), sizeof(float)));
                sortedNums[0] = counts[0];
            } else {
                MrgSortSrcList srcList =
                    MrgSortSrcList(lists[0], lists[liveNum > 1 ? 1 : 0], lists[liveNum > NUM_TWO ? NUM_TWO : 0],
                                   lists[liveNum > NUM_THREE ? NUM_THREE : 0]);
                const uint16_t validBit = static_cast<uint16_t>((1U << liveNum) - 1U);
                MrgSort<float, true>(outLocal, srcList, counts, sortedNums, validBit, 1);
            }
            PipeBarrier<PIPE_V>();

            // 推进各路偏移与剩余
            int64_t produced = 0;
            for (int64_t j = 0; j < liveNum; j++) {
                const int64_t consumed = static_cast<int64_t>(sortedNums[j]);
                const int64_t i = liveMap[j];
                offsets[i] += GetSortLen<float>(consumed);
                remains[i] -= consumed;
                allRemain -= consumed;
                produced += consumed;
            }
            if (produced <= 0) {
                break; // 保护:不应发生,避免死循环
            }

            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams cpOut{static_cast<uint16_t>(1),
                                    static_cast<uint32_t>(GetSortLen<float>(produced) * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[1 - srcIdx][outOffset], outLocal, cpOut);
            // 下一轮要用 MTE2 载入输入、用 MrgSort(向量)重写输出缓冲,两者都要等本次搬出读完。
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            outOffset += GetSortLen<float>(produced);
        }
    }

private:
    A5MergeParam param_;
    TPipe *pipe_ = nullptr;

    GlobalTensor<float> wsGm_[NUM_TWO];
    GlobalTensor<int32_t> cntGm_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> cntBuf_;
    LocalTensor<int32_t> cntLocal_;
    bool cntLoaded_ = false;

    int64_t vecId_ = 0;
    int64_t vecNum_ = 0;
};

// ===================== 归并收尾:拆出 id 与 idx =====================
struct A5ExtractParam {
    int64_t totalValid = 0;      // 最终有序序列的元素数
    int64_t perLoopElements = 0; // 单次载入 UB 的元素数(host 保证为 Sort32 粒度的整数倍)
};

class FfnWbA5Extract {
public:
    __aicore__ inline FfnWbA5Extract(){};

    // pairWs   : 最终一路 proposal 对
    // idsWs    : 输出,有序 expert_id(int32)
    // idxWs    : 输出,gather_idx(int32,指向原始扁平序列的下标)
    __aicore__ inline void Init(GM_ADDR pairWs, GM_ADDR idsWs, GM_ADDR idxWs, const A5ExtractParam &param, TPipe *pipe)
    {
        param_ = param;
        pipe_ = pipe;
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        vecId_ = GetBlockIdx() * subNum + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * subNum;

        pairGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pairWs), GetSortLen<float>(param_.totalValid));
        idsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idsWs), param_.totalValid);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idxWs), param_.totalValid);

        // Extract 按 ONE_REPEAT_SORT_NUM 成批写出:落盘量是**对齐后的长度**,不是有效长度。
        // 按有效长度分配会溢出到相邻缓冲(实测把索引缓冲的开头冲成 0),故三块都按对齐长度给。
        const int64_t alignLoop = Ceil(param_.perLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        pipe_->InitBuffer(pairQue_, 1, GetSortLen<float>(alignLoop) * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(idsBuf_, alignLoop * sizeof(int32_t) + ONE_BLK_SIZE);
        pipe_->InitBuffer(idxBuf_, alignLoop * sizeof(int32_t) + ONE_BLK_SIZE);
    }

    __aicore__ inline void Process()
    {
        const int64_t loops = Ceil(param_.totalValid, param_.perLoopElements);
        for (int64_t l = vecId_; l < loops; l += vecNum_) {
            const int64_t begin = l * param_.perLoopElements;
            const int64_t len = Min(param_.perLoopElements, param_.totalValid - begin);
            if (len <= 0) {
                continue;
            }
            ProcessOneChunk(begin, len);
        }
    }

private:
    __aicore__ inline void ProcessOneChunk(int64_t begin, int64_t len)
    {
        // Extract 按 Sort32 粒度成批工作,故按对齐后的长度拆包,超出有效长度的部分不写出。
        const int64_t alignLen = Ceil(len, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

        LocalTensor<float> pairLocal = pairQue_.AllocTensor<float>();
        DataCopyExtParams cpIn{static_cast<uint16_t>(1), static_cast<uint32_t>(GetSortLen<float>(len) * sizeof(float)),
                               0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(pairLocal, pairGm_[GetSortLen<float>(begin)], cpIn, pad);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        LocalTensor<int32_t> idsLocal = idsBuf_.Get<int32_t>();
        LocalTensor<int32_t> idxLocal = idxBuf_.Get<int32_t>();
        LocalTensor<float> keysLocal = idsLocal.template ReinterpretCast<float>();
        LocalTensor<uint32_t> idxU32 = idxLocal.template ReinterpretCast<uint32_t>();

        Extract(keysLocal, idxU32, pairLocal, static_cast<int32_t>(alignLen / ONE_REPEAT_SORT_NUM));
        PipeBarrier<PIPE_V>();
        // 还原排序前取的负号,再转回 int32
        Muls(keysLocal, keysLocal, static_cast<float>(-1), len);
        PipeBarrier<PIPE_V>();
        Cast(idsLocal, keysLocal, RoundMode::CAST_ROUND, len);
        PipeBarrier<PIPE_V>();

        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyExtParams cpOut{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(idsGm_[begin], idsLocal, cpOut);
        DataCopyPad(idxGm_[begin], idxLocal, cpOut);
        // 下一块会用 Extract(向量)重写这两块缓冲:必须等本块搬出读完。
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        pairQue_.FreeTensor(pairLocal);
    }

private:
    A5ExtractParam param_;
    TPipe *pipe_ = nullptr;

    GlobalTensor<float> pairGm_;
    GlobalTensor<int32_t> idsGm_;
    GlobalTensor<int32_t> idxGm_;

    TQue<QuePosition::VECIN, 1> pairQue_;
    TBuf<TPosition::VECCALC> idsBuf_;
    TBuf<TPosition::VECCALC> idxBuf_;

    int64_t vecId_ = 0;
    int64_t vecNum_ = 0;
};
} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_A5_SORT_H
