/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file und_gen_qkv_rms_norm_rope_cache_regbase.h
 * \brief UndGenQkvRmsNormRopeCache regbase(DAV_3510 / arch35) 模板
 *
 * 切分（与 op_host/und_gen_qkv_rms_norm_rope_cache_tiling.h 一一对应）：
 *   多核：按输出 token 维 total 余数分配，前 formerCoreNum 个核处理 blockFactor 个
 *         token，其余核处理 tailBlockFactor 个；各核独立无同步。
 *   UB  ：只切 token 维，每次处理 ubFactor 个 token。CopyIn 按 tile 批量预取，
 *         计算与写出逐 token，计算中间量走 vreg 不落 UB。
 *         cat_indices / slot_mapping / positions 在 GM 上连续，按 IDX_WINDOW_TOKENS
 *         大小的滑窗跨 tile 批量搬进 idxBuf_，标量读全部落在 UB 上，不逐 token 打 GM。
 *   流水：三段式软件流水，见 Process。Compute 发完 VF 就走，CopyIn 预取下一 tile 与之重叠，
 *         CopyOut 写的是上一拍已算完的 tile，DeQue 不会阻塞在刚发出的 VF 上。
 *
 * 每个 out_t 的计算流程：
 *   1) src_t = cat_indices[out_t]；isUnd = (src_t < undLen)
 *   2) 从 und_qkv / gen_qkv 取 [N, D] 行，按 N 维拆 Q/K/V；只有 Q/K 逐 head
 *      Cast bf16->float32 参与后续计算，V 全程 bf16 不做任何转换
 *   3) Q/K 做 RMSNorm（und/gen 两套权重按 isUnd 选择）
 *   4) 读 positions[0..2, out_t]，三轴 cos_sin 原样搬进 UB 的 [3, D] 窗口，
 *      VF 里用 Gather + axisLut 在寄存器内合并，再对 Q/K 做标准 RoPE；V 不参与 norm/rope
 *   5) Q/K Cast float32->bf16，V 直接透传；Q 写 q[out_t]，K/V 按 slot = slot_mapping[out_t]
 *      写入 k_cache/v_cache 的第 slot 行
 *
 * RMSNorm + MRoPE 融合在一个 VF 段里完成，见 und_gen_qkv_rms_norm_rope_cache_mrope_vf.h。
 * 中间量（fp32 的 x、合并后的 cos/sin、RoPE 交叉项）全部留在 vreg，不占 UB。
 */

#ifndef UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_
#define UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "und_gen_qkv_rms_norm_rope_cache_mrope_vf.h"

namespace UndGenQkvRmsNormRopeCache {
using namespace AscendC;

constexpr static int64_t MROPE_AXIS_NUM = 3;
constexpr static int64_t DIGIT_TWO = 2;
constexpr static int64_t DIGIT_THREE = 3;
constexpr static int64_t WEIGHT_NUM = 4; // und_q / und_k / gen_q / gen_k
constexpr static int64_t BLOCK_ALIGN_BYTES = 32;
constexpr static int64_t BUFFER_NUM_DB = 2;
constexpr static int64_t BUFFER_NUM_SINGLE = 1;

// wInQue / wFp32Buf 内 4 个 gamma 的排布下标（bf16 与 float 两份用同一套下标）
constexpr static int64_t WEIGHT_IDX_UND_Q = 0;
constexpr static int64_t WEIGHT_IDX_UND_K = 1;
constexpr static int64_t WEIGHT_IDX_GEN_Q = 2;
constexpr static int64_t WEIGHT_IDX_GEN_K = 3;

// 标量写 UB 后被向量读，需要显式 S->V 同步；这里只用一个事件号
constexpr static event_t EVT_S_TO_V_INDEX_READY = EVENT_ID0;
// 索引批量搬入 UB 后被标量读，需要 MTE2->S 同步；与上面不是同一对流水，事件号也分开
constexpr static event_t EVT_MTE2_TO_S_INDEX_READY = EVENT_ID1;
// 换窗口时要覆盖旧索引，先等上一窗口的标量读做完，S->MTE2
constexpr static event_t EVT_S_TO_MTE2_WINDOW_REUSE = EVENT_ID2;

// idxBuf_ 是一个跨 tile 复用的索引滑窗：5 个区（cat / slot / positions 三轴），
// 每区放 IDX_WINDOW_TOKENS 个 token 的索引。下面三个常量与 op_host/..._tiling.h
// 的 IDX_REGION_NUM / IDX_WINDOW_TOKENS 必须一致，host 的 residentBytes 就是按它们算的。
//
// 为什么要滑窗而不是每 tile 搬一次：LoadIndexWindow 里的 MTE2->S 屏障等的是整条 MTE2
// 队列（不只是这 5 笔），而且它阻塞的是标量流水。放在每 tile 的话，预取下一 tile 的
// 搬运指令就发不出去，Process 的两段式流水直接失效。窗口把屏障频率降到约 1/13，
// 预取才立得住。窗口至少要能同时装下"在算的"和"在预取的"两个 tile。
constexpr static int64_t IDX_REGION_NUM = 5;
constexpr static int64_t IDX_WINDOW_TOKENS = 256;
constexpr static int64_t IDX_REGION_CAT = 0;  // cat_indices[winBegin .. +winLen)
constexpr static int64_t IDX_REGION_SLOT = 1; // slot_mapping[winBegin .. +winLen)
constexpr static int64_t IDX_REGION_POS = 2;  // positions[axis, winBegin .. +winLen)，占 3 个区
constexpr static int64_t UND_MASK_BITS = static_cast<int64_t>(sizeof(uint64_t) * 8);
static_assert(IDX_REGION_POS + MROPE_AXIS_NUM == IDX_REGION_NUM, "index regions must be contiguous and exhaustive");
static_assert(IDX_WINDOW_TOKENS * static_cast<int64_t>(sizeof(int64_t)) % BLOCK_ALIGN_BYTES == 0,
              "index region stride must be 32B-aligned for DataCopyPad");
// ubFactor 上限是 undMask_ 的位宽（host 的 MAX_UB_FACTOR）。窗口要同时容纳
// pend / 在算 / 预取三个 tile，否则 CopyOut 读 pend 的 slot 时可能已被换窗覆盖
static_assert(IDX_WINDOW_TOKENS >= DIGIT_THREE * UND_MASK_BITS,
              "index window must hold the pending, in-flight and prefetched tiles");

/**
 * @brief UndGenQkvRmsNormRopeCache regbase 模板
 * @tparam T_QKV   und_qkv/gen_qkv 与 q 输出的数据类型（bfloat16_t）
 * @tparam T_CACHE k_cache/v_cache 的数据类型（bfloat16_t）
 */
template <typename T_QKV, typename T_CACHE>
class UndGenQkvRmsNormRopeCacheRegbase {
public:
    __aicore__ inline UndGenQkvRmsNormRopeCacheRegbase(TPipe *pipe, const UndGenQkvRmsNormRopeCacheTilingData *tiling)
        : pipe_(pipe),
          tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR undQkv, GM_ADDR undWeightsQ, GM_ADDR undWeightsK, GM_ADDR cosSinCache,
                                GM_ADDR kCache, GM_ADDR vCache, GM_ADDR slotMapping, GM_ADDR positions, GM_ADDR genQkv,
                                GM_ADDR genWeightsQ, GM_ADDR genWeightsK, GM_ADDR catIndices, GM_ADDR q)
    {
        ParseTiling();
        BindGlobalTensors(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache, vCache, slotMapping, positions, genQkv,
                          genWeightsQ, genWeightsK, catIndices, q);
        CalcCoreRange();
        InitUbBuffers();
    }

    __aicore__ inline void Process()
    {
        // 没分到 token 的核直接退出
        if (coreStart_ >= coreEnd_) {
            return;
        }

        // 权重每核只搬一次，常驻到整个 tile 循环结束
        PrepareWeights();
        // axisLut 与 token 无关，每核只建一次
        BuildMropeGatherIndex();

        // 三段式软件流水，稳态下每轮同时经手三个 tile。行序即循环体内的执行顺序
        //
        //   轮次       序幕   i=0    i=1    i=2    收尾
        //   Compute    --     T0     T1     T2     --     发 VF，发完就走，VF 自己跑
        //   CopyIn     T0     T1     T2     --     --     预取下一个，被 MTE2 队列反压时与 VF 并行
        //   CopyOut    --     --     T0     T1     T2     写上一拍的，其 VF 早已完成，DeQue 不阻塞
        //
        Tile cur;
        cur.start = coreStart_;
        cur.num = TileTokenNum(coreStart_);
        Tile pend; // 已算完、待写出的那个 tile；pend.Valid() 为假表示还没有

        EnsureIndexWindow(cur.start, cur.start + cur.num);
        CopyIn(cur);

        while (cur.Valid()) {
            const Tile next = NextTile(cur);
            Compute(cur);
            if (next.Valid()) {
                // anchor 取最早一个还要读索引的 tile：CopyOut 还要读 pend 的 slot_mapping
                EnsureIndexWindow(pend.Valid() ? pend.start : cur.start, next.start + next.num);
                CopyIn(next);
            }
            if (pend.Valid()) {
                CopyOut(pend);
            }
            pend = cur;
            cur = next;
        }
        // 收尾：最后一个 tile 的结果还挂在 outQue_ 上
        if (pend.Valid()) {
            CopyOut(pend);
        }

        wInQue_.FreeTensor(wBf16Local_);
    }

private:
    /*
    把 tilingData 里的标量取到成员变量，并预算出 halfDim_ / reciprocal_ 等派生量，
    同时填好 vfShape_ —— 它与 tile 无关，Compute 每轮直接复用
    */
    __aicore__ inline void ParseTiling()
    {
        totalTokens_ = tiling_->totalTokens;
        undLen_ = tiling_->undLen;
        numHead_ = tiling_->numHead;
        numHeadQ_ = tiling_->numHeadQ;
        numHeadK_ = tiling_->numHeadK;
        numHeadV_ = tiling_->numHeadV;
        headDim_ = tiling_->headDim;
        maxPos_ = tiling_->maxPos;
        blockNum_ = tiling_->blockNum;
        blockSize_ = tiling_->blockSize;
        hasGen_ = tiling_->hasGen != 0;
        hasCatIndices_ = tiling_->hasCatIndices != 0;
        mropeSection_[0] = tiling_->mropeSectionT;
        mropeSection_[1] = tiling_->mropeSectionH;
        mropeSection_[2] = tiling_->mropeSectionW;
        epsilon_ = tiling_->epsilon;
        reciprocal_ = tiling_->reciprocal;

        usedCoreNum_ = tiling_->usedCoreNum;
        formerCoreNum_ = tiling_->formerCoreNum;
        blockFactor_ = tiling_->blockFactor;
        tailBlockFactor_ = tiling_->tailBlockFactor;
        ubFactor_ = tiling_->ubFactor;

        headNumQK_ = numHeadQ_ + numHeadK_; // V 不参与 RMSNorm/RoPE
        halfDim_ = headDim_ / DIGIT_TWO;
        rowElems_ = numHead_ * headDim_;   // 单 token 的 QKV 元素数
        maxSlot_ = blockNum_ * blockSize_; // slot_mapping 的合法上界（不含）

        // VF 的 shape 参数全部由 shape/attr 决定，与 tile 无关，这里填一次给 Compute 复用
        vfShape_.qHeadNum = static_cast<uint16_t>(numHeadQ_);
        vfShape_.kHeadNum = static_cast<uint16_t>(numHeadK_);
        vfShape_.vHeadNum = static_cast<uint16_t>(numHeadV_);
        vfShape_.inTokenStride = static_cast<uint32_t>(rowElems_);
        vfShape_.cosSinTokenStride = static_cast<uint32_t>(MROPE_AXIS_NUM * headDim_);
        vfShape_.qOutTokenStride = static_cast<uint32_t>(numHeadQ_ * headDim_);
        vfShape_.kOutTokenStride = static_cast<uint32_t>(numHeadK_ * headDim_);
        vfShape_.vOutTokenStride = static_cast<uint32_t>(numHeadV_ * headDim_);
        vfShape_.headStride = static_cast<uint32_t>(headDim_);
        vfShape_.halfDim = static_cast<uint32_t>(halfDim_);
        vfShape_.epsilon = epsilon_;
        vfShape_.reciprocal = reciprocal_;
    }

    /*
    把 13 个 GM 地址绑定到 GlobalTensor 成员，参数与 aclnn 接口同名同序
    k_cache / v_cache 是原地更新的入参，绑定后由 CopyOut 按 slot_mapping 散写
    */
    __aicore__ inline void BindGlobalTensors(GM_ADDR undQkv, GM_ADDR undWeightsQ, GM_ADDR undWeightsK,
                                             GM_ADDR cosSinCache, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR slotMapping,
                                             GM_ADDR positions, GM_ADDR genQkv, GM_ADDR genWeightsQ,
                                             GM_ADDR genWeightsK, GM_ADDR catIndices, GM_ADDR q)
    {
        undQkvGm_.SetGlobalBuffer((__gm__ T_QKV *)undQkv);
        undWeightsQGm_.SetGlobalBuffer((__gm__ T_QKV *)undWeightsQ);
        undWeightsKGm_.SetGlobalBuffer((__gm__ T_QKV *)undWeightsK);
        cosSinCacheGm_.SetGlobalBuffer((__gm__ float *)cosSinCache);
        kCacheGm_.SetGlobalBuffer((__gm__ T_CACHE *)kCache);
        vCacheGm_.SetGlobalBuffer((__gm__ T_CACHE *)vCache);
        slotMappingGm_.SetGlobalBuffer((__gm__ int64_t *)slotMapping);
        positionsGm_.SetGlobalBuffer((__gm__ int64_t *)positions);
        if (hasGen_) {
            genQkvGm_.SetGlobalBuffer((__gm__ T_QKV *)genQkv);
            genWeightsQGm_.SetGlobalBuffer((__gm__ T_QKV *)genWeightsQ);
            genWeightsKGm_.SetGlobalBuffer((__gm__ T_QKV *)genWeightsK);
        }
        if (hasCatIndices_) {
            catIndicesGm_.SetGlobalBuffer((__gm__ int64_t *)catIndices);
        }
        qGm_.SetGlobalBuffer((__gm__ T_QKV *)q);
    }

    // 多核切分：前 formerCoreNum 个核多处理 1 个 token，与 host CalBlockTiling 一致
    __aicore__ inline void CalcCoreRange()
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum_) { // SetBlockDim 已按 usedCoreNum 下发，这里只是兜底
            coreStart_ = 0;
            coreEnd_ = 0;
            return;
        }
        int64_t coreTokenNum;
        if (blockIdx < formerCoreNum_) {
            coreStart_ = blockIdx * blockFactor_;
            coreTokenNum = blockFactor_;
        } else {
            coreStart_ = formerCoreNum_ * blockFactor_ + (blockIdx - formerCoreNum_) * tailBlockFactor_;
            coreTokenNum = tailBlockFactor_;
        }
        coreEnd_ = coreStart_ + coreTokenNum;
    }

    // UB 划分严格对应 op_host/und_gen_qkv_rms_norm_rope_cache_tiling.h 的 "UB 划分" 注释，
    // 改这里必须同步改 host 的 CalUbTiling，否则 ubFactor 反推会与实际占用不符
    __aicore__ inline void InitUbBuffers()
    {
        pipe_->InitBuffer(qkvInQue_, BUFFER_NUM_DB, ubFactor_ * rowElems_ * sizeof(T_QKV));
        pipe_->InitBuffer(cosSinInQue_, BUFFER_NUM_DB, ubFactor_ * MROPE_AXIS_NUM * headDim_ * sizeof(float));
        pipe_->InitBuffer(outQue_, BUFFER_NUM_DB, ubFactor_ * rowElems_ * sizeof(T_QKV));
        pipe_->InitBuffer(wInQue_, BUFFER_NUM_SINGLE, WEIGHT_NUM * headDim_ * sizeof(T_QKV));

        // 计算区是两块与 token 无关的小 buffer：4 份 gamma 的 float 版本 + gather 索引。
        // VF 按 undMask 算基址，直接从 wFp32Buf 取本 token 该用的那一组 gamma
        pipe_->InitBuffer(wFp32Buf_, WEIGHT_NUM * headDim_ * sizeof(float));
        pipe_->InitBuffer(gatherIdxBuf_, AlignUp(halfDim_ * sizeof(uint32_t), BLOCK_ALIGN_BYTES));
        pipe_->InitBuffer(idxBuf_, IDX_REGION_NUM * IDX_WINDOW_TOKENS * sizeof(int64_t));

        wFp32Local_ = wFp32Buf_.Get<float>();
        gatherIdxLocal_ = gatherIdxBuf_.Get<uint32_t>();
        idxLocal_ = idxBuf_.Get<int64_t>();
    }

    // 4 个 gamma 一次性搬入并 Cast 成 float 常驻，按 [undQ|undK|genQ|genK] 排布。
    // VF 里每 token 只加载一次，该段所有 head 共用同一份。
    __aicore__ inline void PrepareWeights()
    {
        LocalTensor<T_QKV> wBf16 = wInQue_.AllocTensor<T_QKV>();
        DataCopy(wBf16[WEIGHT_IDX_UND_Q * headDim_], undWeightsQGm_, headDim_);
        DataCopy(wBf16[WEIGHT_IDX_UND_K * headDim_], undWeightsKGm_, headDim_);
        if (hasGen_) {
            DataCopy(wBf16[WEIGHT_IDX_GEN_Q * headDim_], genWeightsQGm_, headDim_);
            DataCopy(wBf16[WEIGHT_IDX_GEN_K * headDim_], genWeightsKGm_, headDim_);
        }
        wInQue_.EnQue(wBf16);
        wBf16Local_ = wInQue_.DeQue<T_QKV>();

        const int64_t weightElems = (hasGen_ ? WEIGHT_NUM : DIGIT_TWO) * headDim_;
        Cast(wFp32Local_, wBf16Local_, RoundMode::CAST_NONE, weightElems);
        PipeBarrier<PIPE_V>();
    }

    // axisLut 展开成 gather 索引：gatherIndex[lane] = axis(lane) * D + lane。
    // 规则与竞品 _mrope / golden.py:mrope_axis_map 一致，注意 mropeSection_[0] 不参与判断，
    // T 轴既是 i%3==0 的归属轴，也是超出 3*section 之后的兜底轴。
    // mrope_section 为空时 host 填 [D/2, 0, 0]，此处 sectionH/W 均为 0，全部 lane 落 T 轴，
    // 自动退化成标准 RoPE，不需要额外分支。
    __aicore__ inline void BuildMropeGatherIndex()
    {
        for (int64_t lane = 0; lane < halfDim_; ++lane) {
            const int64_t repeat = lane / MROPE_AXIS_NUM;
            const int64_t axisInGroup = lane % MROPE_AXIS_NUM;
            int64_t axis = 0;
            if (axisInGroup == 1 && repeat < mropeSection_[1]) {
                axis = 1;
            } else if (axisInGroup == DIGIT_TWO && repeat < mropeSection_[DIGIT_TWO]) {
                axis = DIGIT_TWO;
            }
            gatherIdxLocal_.SetValue(lane, static_cast<uint32_t>(axis * headDim_ + lane));
        }
        // 标量写入的 UB 要被 VF 读，必须显式同步，否则 Gather 可能读到旧值
        SetFlag<HardEvent::S_V>(EVT_S_TO_V_INDEX_READY);
        WaitFlag<HardEvent::S_V>(EVT_S_TO_V_INDEX_READY);
    }

    /*
    一个 tile 的身份，Process 的三段流水按它传递
    start：本 tile 第一个 token 在本核区间内的绝对下标（out_t 口径）
    num：本 tile 的 token 数，num == 0 表示该 tile 不存在
    maskSlot：本 tile 的 und/gen 位图存放在 undMask_ 的哪一份，取值 0/1
    */
    struct Tile {
        int64_t start = 0;
        int64_t num = 0;
        int64_t maskSlot = 0;
        __aicore__ inline bool Valid() const
        {
            return num > 0;
        }
    };

    /*
    算一个 tile 能放多少 token
    tileStart：该 tile 的起始 out_t
    返回 min(ubFactor_, coreEnd_ - tileStart)，即满 tile 取 ubFactor、核尾取剩余
    */
    __aicore__ inline int64_t TileTokenNum(int64_t tileStart)
    {
        const int64_t rest = coreEnd_ - tileStart;
        return rest > ubFactor_ ? ubFactor_ : rest;
    }

    /*
    推进一拍，maskSlot 一并翻转，调用方不必关心乒乓
    cur：当前 tile
    返回紧接其后的 tile；已越过 coreEnd_ 时返回 num = 0 的 Tile 表示没有下一个
    */
    __aicore__ inline Tile NextTile(const Tile &cur)
    {
        Tile t;
        t.start = cur.start + cur.num;
        t.num = (t.start < coreEnd_) ? TileTokenNum(t.start) : 0;
        t.maskSlot = cur.maskSlot ^ 1;
        return t;
    }

    /*
    算索引滑窗内某个元素的 UB 下标
    region：区号，取 IDX_REGION_CAT / IDX_REGION_SLOT / IDX_REGION_POS+axis
    outT：输出 token 的绝对下标，必须落在当前窗口 [winBegin_, winEnd_) 内
    返回 idxLocal_ 上的元素下标，调用方直接拿去 GetValue
    */
    __aicore__ inline int64_t IdxAt(int64_t region, int64_t outT)
    {
        return region * IDX_WINDOW_TOKENS + (outT - winBegin_);
    }

    /*
    gm->ub 拷入一段 int64 索引
    dst：目标 UB 位置，须落在 idxBuf_ 的某个区起址上（区跨度已保证 32B 对齐）
    src：源 GM 位置，调用方自行偏移到起始 token
    num：拷贝的元素个数
    走 DataCopyPad，不足 32B 的尾部由硬件补齐，不需要调用方对齐 num
    */
    __aicore__ inline void CopyIndexToUb(const LocalTensor<int64_t> &dst, const GlobalTensor<int64_t> &src, int64_t num)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(num * sizeof(int64_t)), 0, 0, 0};
        DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
        DataCopyPad(dst, src, params, padParams);
    }

    /*
    换窗：把 [begin, begin+len) 这段 token 的 5 组索引搬进 idxBuf_
    begin：新窗口的起始 out_t
    len：窗口内的 token 数，调用方须保证不超过 IDX_WINDOW_TOKENS
    返回时 winBegin_/winEnd_ 已更新，且索引已落地可被标量读
    */
    __aicore__ inline void LoadIndexWindow(int64_t begin, int64_t len)
    {
        // 旧窗口的内容还可能刚被标量读过，覆盖前先等标量走到这里
        if (winBegin_ >= 0) {
            SetFlag<HardEvent::S_MTE2>(EVT_S_TO_MTE2_WINDOW_REUSE);
            WaitFlag<HardEvent::S_MTE2>(EVT_S_TO_MTE2_WINDOW_REUSE);
        }
        winBegin_ = begin;
        winEnd_ = begin + len;

        if (hasCatIndices_) {
            CopyIndexToUb(idxLocal_[IDX_REGION_CAT * IDX_WINDOW_TOKENS], catIndicesGm_[begin], len);
        }
        CopyIndexToUb(idxLocal_[IDX_REGION_SLOT * IDX_WINDOW_TOKENS], slotMappingGm_[begin], len);
        // positions 是 [3, total]，三个轴在 GM 上隔着 totalTokens_，只能一轴一笔
        for (int64_t axis = 0; axis < MROPE_AXIS_NUM; ++axis) {
            CopyIndexToUb(idxLocal_[(IDX_REGION_POS + axis) * IDX_WINDOW_TOKENS],
                          positionsGm_[axis * totalTokens_ + begin], len);
        }
        SetFlag<HardEvent::MTE2_S>(EVT_MTE2_TO_S_INDEX_READY);
        WaitFlag<HardEvent::MTE2_S>(EVT_MTE2_TO_S_INDEX_READY);
    }

    /*
    保证 [anchor, needEnd) 落在当前窗口内，不满足则换窗
    anchor：最早一个还要读索引的 token，即 pend tile 的起点而非待预取 tile 的起点，
            否则正在算的与待写出的 tile 的 slot 会被换窗覆盖
    needEnd：最晚一个还要读索引的 token 的下一个位置
    命中现有窗口时直接返回，不发任何搬运
    */
    __aicore__ inline void EnsureIndexWindow(int64_t anchor, int64_t needEnd)
    {
        if (winBegin_ >= 0 && anchor >= winBegin_ && needEnd <= winEnd_) {
            return;
        }
        int64_t len = coreEnd_ - anchor;
        if (len > IDX_WINDOW_TOKENS) {
            len = IDX_WINDOW_TOKENS;
        }
        LoadIndexWindow(anchor, len);
    }

    /*
    取输出 token 对应的源 token 下标
    outT：输出侧 token 下标，须在当前索引窗口内
    返回 cat_indices[outT]；未传 cat_indices 时退化为 outT。
    取值越界一律回落到 0，避免非法 GM 访问（这是运行期数据，host 校验不到）
    */
    __aicore__ inline int64_t GetSrcIndex(int64_t outT)
    {
        int64_t srcT = hasCatIndices_ ? idxLocal_.GetValue(IdxAt(IDX_REGION_CAT, outT)) : outT;
        if (srcT < 0 || srcT >= totalTokens_) {
            srcT = 0;
        }
        return srcT;
    }

    /*
    gm->ub 搬入一个 tile 的 qkv 与三轴 cos_sin
    tile：待搬入的 tile，用 start/num 定位 token 区间，maskSlot 定位位图存放处
    cat_indices 乱序使各 token 的源行不连续，qkv 只能一 token 一笔；cos_sin 每 token
    三轴各一笔。所有 GetValue 读的是 EnsureIndexWindow 已搬进 UB 的索引
    只发 MTE2 不等落地，落地由 Compute 的 DeQue 保证；
    返回时 undMask_[tile.maskSlot] 已填好本 tile 的 und/gen 位图供 VF 选 gamma
    */
    __aicore__ inline void CopyIn(const Tile &tile)
    {
        const int64_t tileStart = tile.start;
        const int64_t tokenNum = tile.num;
        LocalTensor<T_QKV> qkvLocal = qkvInQue_.AllocTensor<T_QKV>();
        // 顺带产出本 tile 的 und/gen 位图给 VF 选 gamma 用：srcT 这里本来就要读，
        // 不必在别处再读一次 cat_indices。
        // 位图按流水槽位存两份：预取下一 tile 时当前 tile 还没被 Compute 消费
        uint64_t mask = 0;
        for (int64_t i = 0; i < tokenNum; ++i) {
            int64_t srcT = GetSrcIndex(tileStart + i);
            if (srcT < undLen_) {
                mask |= (static_cast<uint64_t>(1) << i);
                DataCopy(qkvLocal[i * rowElems_], undQkvGm_[srcT * rowElems_], rowElems_);
            } else {
                DataCopy(qkvLocal[i * rowElems_], genQkvGm_[(srcT - undLen_) * rowElems_], rowElems_);
            }
        }
        qkvInQue_.EnQue(qkvLocal);

        // 三轴 cos_sin：每 token 按 positions[axis, out_t] 各取一行 [D]
        LocalTensor<float> cosSinLocal = cosSinInQue_.AllocTensor<float>();
        for (int64_t i = 0; i < tokenNum; ++i) {
            for (int64_t axis = 0; axis < MROPE_AXIS_NUM; ++axis) {
                int64_t pos = idxLocal_.GetValue(IdxAt(IDX_REGION_POS + axis, tileStart + i));
                if (pos < 0 || pos >= maxPos_) { // 运行期数据，host 校验不到，这里兜底
                    pos = 0;
                }
                DataCopy(cosSinLocal[(i * MROPE_AXIS_NUM + axis) * headDim_], cosSinCacheGm_[pos * headDim_], headDim_);
            }
        }
        cosSinInQue_.EnQue(cosSinLocal);
        undMask_[tile.maskSlot] = mask;
    }

    /*
    一个 tile 的 RMSNorm + MRoPE + Cast，整个 tile 只发一次 VF
    tile：待计算的 tile，其输入须已由 CopyIn 搬入并 EnQue
    Q/K 在 UB 上是同一行 token 内相邻的 head，合在一个 VF 里 token 循环只走一趟、
    cos/sin 只 Gather 一次；und/gen 的差异在 VF 内按 undMask_[tile.maskSlot] 逐 token
    选 gamma，所以 cat_indices 无论怎么交错都只发一次 VF。V 不参与，直通
    发完 VF 即返回不等它算完，结果已 EnQue 到 outQue_ 等 CopyOut 取走
    */
    __aicore__ inline void Compute(const Tile &tile)
    {
        const int64_t tokenNum = tile.num;
        LocalTensor<T_QKV> qkvLocal = qkvInQue_.DeQue<T_QKV>();
        LocalTensor<float> cosSinLocal = cosSinInQue_.DeQue<float>();
        LocalTensor<T_QKV> outLocal = outQue_.AllocTensor<T_QKV>();

        const int64_t kSegBase = ubFactor_ * numHeadQ_ * headDim_;
        const int64_t vSegBase = ubFactor_ * headNumQK_ * headDim_;

        // Q/K 的 RMSNorm+MRoPE 与 V 的直通都在这次 VF 里完成。
        // shape 参数在 ParseTiling 就填好了，这里只组本 tile 的地址
        QkvMropeTileAddr addr;
        addr.qkvIn = (__ubuf__ bfloat16_t *)qkvLocal.GetPhyAddr();
        addr.gammaAll = (__ubuf__ float *)wFp32Local_.GetPhyAddr();
        addr.rawCosSin = (__ubuf__ float *)cosSinLocal.GetPhyAddr();
        addr.gatherIndex = (__ubuf__ uint32_t *)gatherIdxLocal_.GetPhyAddr();
        addr.qOut = (__ubuf__ bfloat16_t *)outLocal.GetPhyAddr();
        addr.kOut = (__ubuf__ bfloat16_t *)outLocal[kSegBase].GetPhyAddr();
        addr.vOut = (__ubuf__ bfloat16_t *)outLocal[vSegBase].GetPhyAddr();
        QkRmsNormMropeTileVF(addr, vfShape_, static_cast<uint16_t>(tokenNum), undMask_[tile.maskSlot]);

        outQue_.EnQue(outLocal);
        qkvInQue_.FreeTensor(qkvLocal);
        cosSinInQue_.FreeTensor(cosSinLocal);
    }

    /*
    ub->gm 写出一个 tile 的结果
    tile：待写出的 tile，其结果须已由 Compute EnQue（DeQue 会等它的 VF 算完）
    outLocal 按 [ubFactor 个 q | ubFactor 个 k | ubFactor 个 v] 分段：q 的输出下标
    out_t 连续可一笔写出，k/v 按 slot_mapping 散写只能逐 token
    slot 越界的 token 跳过写入而不是踩内存（运行期数据，host 校验不到）
    */
    __aicore__ inline void CopyOut(const Tile &tile)
    {
        const int64_t tileStart = tile.start;
        const int64_t tokenNum = tile.num;
        LocalTensor<T_QKV> outLocal = outQue_.DeQue<T_QKV>();

        const int64_t qSegElems = numHeadQ_ * headDim_;
        const int64_t kSegElems = numHeadK_ * headDim_;
        const int64_t vSegElems = numHeadV_ * headDim_;
        const int64_t kSegBase = ubFactor_ * qSegElems;
        const int64_t vSegBase = ubFactor_ * headNumQK_ * headDim_;

        DataCopy(qGm_[tileStart * qSegElems], outLocal[0], tokenNum * qSegElems);

        for (int64_t i = 0; i < tokenNum; ++i) {
            int64_t slot = idxLocal_.GetValue(IdxAt(IDX_REGION_SLOT, tileStart + i));
            if (slot < 0 || slot >= maxSlot_) { // 运行期数据，越界则跳过写入而不是踩内存
                continue;
            }
            DataCopy(kCacheGm_[slot * kSegElems], outLocal[kSegBase + i * kSegElems], kSegElems);
            DataCopy(vCacheGm_[slot * vSegElems], outLocal[vSegBase + i * vSegElems], vSegElems);
        }

        outQue_.FreeTensor(outLocal);
    }

    /*
    向上对齐
    value：待对齐的值
    align：对齐粒度，须为正
    返回不小于 value 的最小 align 整数倍
    */
    __aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
    {
        return (value + align - 1) / align * align;
    }

private:
    TPipe *pipe_ = nullptr;
    const UndGenQkvRmsNormRopeCacheTilingData *tiling_ = nullptr;

    // 模板参数是 queue depth（同时"已 EnQue 未 DeQue"的句柄数），不是 buffer 数；
    // 物理 buffer 数由 InitUbBuffers 的 BUFFER_NUM_DB 决定，UB 占用不受 depth 影响。
    // outQue_ 的 CopyOut 延后一拍，稳态 outstanding 为 2，depth 必须取 2；
    // 输入队列是 Compute 先 DeQue、CopyIn 再 EnQue，outstanding 只有 1，depth 2 是余量
    TQue<QuePosition::VECIN, DIGIT_TWO> qkvInQue_;
    TQue<QuePosition::VECIN, DIGIT_TWO> cosSinInQue_;
    TQue<QuePosition::VECOUT, DIGIT_TWO> outQue_;
    TQue<QuePosition::VECIN, 1> wInQue_;

    TBuf<TPosition::VECCALC> wFp32Buf_;
    TBuf<TPosition::VECCALC> gatherIdxBuf_;
    TBuf<TPosition::VECCALC> idxBuf_;

    // 常驻 LocalTensor：权重来自 wInQue_（Process 开头 DeQue、结束时 Free），
    // 其余是 TBuf 的视图，Init 时取一次，整个 Process 复用
    LocalTensor<T_QKV> wBf16Local_;
    LocalTensor<float> wFp32Local_;
    LocalTensor<uint32_t> gatherIdxLocal_;
    LocalTensor<int64_t> idxLocal_;

    GlobalTensor<T_QKV> undQkvGm_;
    GlobalTensor<T_QKV> genQkvGm_;
    GlobalTensor<T_QKV> undWeightsQGm_;
    GlobalTensor<T_QKV> undWeightsKGm_;
    GlobalTensor<T_QKV> genWeightsQGm_;
    GlobalTensor<T_QKV> genWeightsKGm_;
    GlobalTensor<float> cosSinCacheGm_;
    GlobalTensor<T_CACHE> kCacheGm_;
    GlobalTensor<T_CACHE> vCacheGm_;
    GlobalTensor<int64_t> slotMappingGm_;
    GlobalTensor<int64_t> positionsGm_;
    GlobalTensor<int64_t> catIndicesGm_;
    GlobalTensor<T_QKV> qGm_;

    // shape / 属性
    int64_t totalTokens_ = 0;
    int64_t undLen_ = 0;
    int64_t numHead_ = 0;
    int64_t numHeadQ_ = 0;
    int64_t numHeadK_ = 0;
    int64_t numHeadV_ = 0;
    int64_t headDim_ = 0;
    int64_t maxPos_ = 0;
    int64_t blockNum_ = 0;
    int64_t blockSize_ = 0;
    int64_t mropeSection_[MROPE_AXIS_NUM] = {0, 0, 0};
    bool hasGen_ = false;
    bool hasCatIndices_ = false;
    float epsilon_ = 0.0f;
    float reciprocal_ = 0.0f;

    // 切分
    int64_t usedCoreNum_ = 0;
    int64_t formerCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t ubFactor_ = 0;
    int64_t coreStart_ = 0;
    int64_t coreEnd_ = 0;

    // VF 的 shape 入参，ParseTiling 期填一次，整个 Process 复用
    QkvMropeTileShape vfShape_ = {};
    // 各 token 的 und/gen 标志位图，CopyIn 产出、Compute 消费。CopyIn 预取的是下一个 tile，
    // 必须按槽位存两份，否则下一 tile 会把还没用的位图冲掉；CopyOut 不读它，两份就够
    uint64_t undMask_[DIGIT_TWO] = {0, 0};

    // 索引滑窗当前覆盖的 token 区间 [winBegin_, winEnd_)；winBegin_ < 0 表示尚未装载
    int64_t winBegin_ = -1;
    int64_t winEnd_ = 0;

    // 派生量
    int64_t headNumQK_ = 0;
    int64_t halfDim_ = 0;
    int64_t rowElems_ = 0;
    int64_t maxSlot_ = 0;
};
} // namespace UndGenQkvRmsNormRopeCache

#endif // UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_
