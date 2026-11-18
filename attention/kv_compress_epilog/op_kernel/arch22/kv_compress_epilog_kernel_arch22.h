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
 * \file kv_compress_epilog_kernel_arch22.h
 * \brief KV compress epilog arch22 (DAV_2201) kernel: quant_mode=1 (int8 动态 per-group 量化), 标准矢量 API
 */

#ifndef KV_COMPRESS_EPILOG_KERNEL_ARCH22_H
#define KV_COMPRESS_EPILOG_KERNEL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace KvCompressEpilogOps {

constexpr int32_t BYTE_BLOCK_A22 = 32;   // UB 对齐粒度 (字节)
constexpr int64_t SCALE_SLOTS_A22 = 128; // WholeReduceMax 显式 mask 路径 dst 实际写入远超声明的 32B block
                                         // (实测溢出 >=288B 且不被 PipeBarrier<PIPE_V> 确定排空, rowFactor>=2 时
                                         // 末组写跨行污染 amax 槽), 槽位加宽到 512B 使写约束在本组槽内
constexpr int64_t SCALE_BLKS_A22 = 8;      // 组数上限槽位 (d=512/64=7 组, 留 1 组余量)
constexpr float INT8_ABS_MAX_A22 = 127.0f; // int8 量化幅值上限 (兼作 scale=amax/127 的除数)
// f32 工作行布局 (元素偏移, 每行 F32_ROW_ELEMS_A22 个, INT8 硬约束 d=512/group=64 -> scaleCol=7):
//   [0, 448)       pass1 abs 区 / pass2 量化商区 qf (pass2 Cast 覆盖)
//   [448, 480)     scale f32 紧凑区 (S-pipe SetValue: 先 amax 后 scale, 输出 bf16 Cast 与 Broadcast 源)
//   [512, 960)     除数区 (Broadcast {7,1}->{7,64}: 先 amax, 精确除后为 scale, 零值修正后重播)
//   [960, 1408)    scale 链 dup127 区 / pass2 x 副本区 (复用)
constexpr int64_t F32_ROW_ELEMS_A22 = 1408; // 工作区行宽 (与 tiling 每行预算一致)

__aicore__ inline int32_t CeilAlignA22(int32_t a, int32_t b)
{
    return (a + b - 1) / b * b;
}

// GM -> UB 批量搬入 (连续 2D x, srcStride 为元素数)
template <typename T>
__aicore__ inline void CopyInRowsA22(const GlobalTensor<T>& srcGm, const LocalTensor<T>& dstLocal, uint16_t rows,
                                     uint32_t rowElems, uint32_t srcStrideElems)
{
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyExtParams copyParams;
    copyParams.blockCount = rows;
    copyParams.blockLen = rowElems * sizeof(T);
    copyParams.srcStride = srcStrideElems * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPad(dstLocal, srcGm, copyParams, padParams);
}

// UB -> GM 单行散写 (copyLen 单位: 字节)
template <typename T>
__aicore__ inline void CopyOutRowA22(const GlobalTensor<T>& dstGm, const LocalTensor<T>& srcLocal, uint32_t copyLen)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = copyLen;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(dstGm, srcLocal, copyParams);
}

template <typename T0, typename U, typename T1>
class KvCompressEpilogArch22 {
public:
    __aicore__ inline KvCompressEpilogArch22(TPipe* pipe)
        : pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR slotMapping, GM_ADDR kvCache,
                                const KvCompressEpilogTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(const LocalTensor<T1>& kvLocal, const LocalTensor<T0>& xLocal, int64_t rf);
    __aicore__ inline void Scatter(const LocalTensor<T1>& kvLocal, int64_t slotBase, int64_t rf);

    TPipe* pipe_ = nullptr;

    GlobalTensor<T0> xGm;
    GlobalTensor<U> slotMappingGm;
    GlobalTensor<T1> kvCacheGm;

    TQue<QuePosition::VECIN, 2> xQue;
    TQue<QuePosition::VECOUT, 2> kvCacheQue;
    TBuf<QuePosition::VECCALC> f32Buf;   // 每行 F32_ROW_ELEMS_A22 个 f32 工作区 (Cast/Abs/Div 复用)
    TBuf<QuePosition::VECCALC> q16Buf;   // 每行 nopeElems 个 int16 (RNE 中间量, f32->s16->half->int8 链路)
    TBuf<QuePosition::VECCALC> scaleBuf; // amax 缓冲: 每组独占 512B 槽位 (SCALE_SLOTS_A22=128 f32)

    const KvCompressEpilogTilingData* tilingData = nullptr;

    int64_t d_ = 0;
    int64_t kvCacheCol_ = 0;
    int64_t scaleCol_ = 0;
    int64_t perGroupSize_ = 0;
    int64_t nopeElems_ = 0;        // = scaleCol * perGroupSize
    int64_t ropeElems_ = 0;        // = d - nopeElems
    int64_t nopeByteOff_ = 0;      // = ropeElems * sizeof(T0)
    int64_t scaleByteOff_ = 0;     // = nopeByteOff + nopeElems
    int64_t xRowStride_ = 0;       // x 行 GM 步长 (元素)
    int64_t xRowElemsAlign_ = 0;   // x 行 UB 步长 (元素, 32B 对齐)
    int64_t outRowBytesAlign_ = 0; // 输出行 UB 步长 (字节, 32B 对齐)
    int64_t q16RowElemsAlign_ = 0; // q16 行 UB 步长 (元素, 32B 对齐)
};

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilogArch22<T0, U, T1>::Init(GM_ADDR x, GM_ADDR slotMapping, GM_ADDR kvCache,
                                                               const KvCompressEpilogTilingData* tilingDataPtr)
{
    tilingData = tilingDataPtr;
    d_ = tilingData->d;
    kvCacheCol_ = tilingData->kvCacheCol;
    scaleCol_ = tilingData->scaleCol;
    perGroupSize_ = tilingData->perGroupSize;
    xRowStride_ = tilingData->xRowStride;
    nopeElems_ = scaleCol_ * perGroupSize_;
    ropeElems_ = d_ - nopeElems_;
    nopeByteOff_ = ropeElems_ * static_cast<int64_t>(sizeof(T0));
    scaleByteOff_ = nopeByteOff_ + nopeElems_;
    xRowElemsAlign_ = CeilAlignA22(static_cast<int32_t>(d_ * sizeof(T0)), BYTE_BLOCK_A22) / sizeof(T0);
    outRowBytesAlign_ = CeilAlignA22(static_cast<int32_t>(kvCacheCol_), BYTE_BLOCK_A22);
    q16RowElemsAlign_ =
        CeilAlignA22(static_cast<int32_t>(nopeElems_ * sizeof(int16_t)), BYTE_BLOCK_A22) / sizeof(int16_t);

    int64_t xGmBaseOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * xRowStride_;
    xGm.SetGlobalBuffer((__gm__ T0*)x + xGmBaseOffset);
    kvCacheGm.SetGlobalBuffer((__gm__ T1*)kvCache);
    slotMappingGm.SetGlobalBuffer((__gm__ U*)slotMapping);

    const int64_t rf = tilingData->rowFactor;
    pipe_->InitBuffer(xQue, 2, rf * xRowElemsAlign_ * sizeof(T0));
    pipe_->InitBuffer(kvCacheQue, 2, rf * outRowBytesAlign_);
    pipe_->InitBuffer(f32Buf, rf * F32_ROW_ELEMS_A22 * sizeof(float));
    pipe_->InitBuffer(q16Buf, rf * q16RowElemsAlign_ * sizeof(int16_t));
    pipe_->InitBuffer(scaleBuf, rf * SCALE_SLOTS_A22 * SCALE_BLKS_A22 * sizeof(float));
}

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilogArch22<T0, U, T1>::Compute(const LocalTensor<T1>& kvLocal,
                                                                  const LocalTensor<T0>& xLocal, int64_t rf)
{
    LocalTensor<uint32_t> rowU32 = kvLocal.template ReinterpretCast<uint32_t>();
    LocalTensor<T0> rowB16 = kvLocal.template ReinterpretCast<T0>();
    LocalTensor<int8_t> rowI8 = kvLocal.template ReinterpretCast<int8_t>();
    LocalTensor<float> f32All = f32Buf.Get<float>();
    LocalTensor<int16_t> q16All = q16Buf.Get<int16_t>(); // RNE 中间量 (f32->s16(CAST_RINT)->half->int8 链路)
    LocalTensor<float> scaleAll = scaleBuf.Get<float>();

    // 整批清零 (rope/nope/scale/pad 先全 0), 与后续段写入存在 UB 内 RAW, V_V 事件同步
    const event_t eventVToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_V));
    Duplicate(rowU32, static_cast<uint32_t>(0), static_cast<uint32_t>(rf * outRowBytesAlign_ / sizeof(uint32_t)));
    SetFlag<HardEvent::V_V>(eventVToV);
    WaitFlag<HardEvent::V_V>(eventVToV);

    const uint32_t groupU32 = static_cast<uint32_t>(perGroupSize_);
    const uint32_t scaleColU32 = static_cast<uint32_t>(scaleCol_);
    const uint32_t nopeU32 = static_cast<uint32_t>(nopeElems_);
    // f32 行内布局见 F32_ROW_ELEMS_A22 注释
    const int64_t scaleF32Off = d_ - perGroupSize_; // 448, 紧凑区实际用 [448,448+scaleCol)
    const int64_t divOff = d_;                      // 512, 除数区 [512,960)
    const int64_t dup127WideOff = 960;              // dup127 宽常数区 [960,1408) / pass2 x 副本区
    const event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    const event_t eventSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    for (int64_t r = 0; r < rf; r++) {
        LocalTensor<float> f32Row = f32All[r * F32_ROW_ELEMS_A22];
        LocalTensor<int16_t> q16Row = q16All[r * q16RowElemsAlign_];
        LocalTensor<float> scaleRow = scaleAll[r * SCALE_SLOTS_A22 * SCALE_BLKS_A22];
        int64_t xRowBase = r * xRowElemsAlign_;
        int64_t rowB16Base = r * outRowBytesAlign_ / sizeof(T0);

        // 工作区清零 (消除 UB 残留不确定性), 与后续 V 写串行
        Duplicate(f32Row, 0.0f, static_cast<uint32_t>(F32_ROW_ELEMS_A22));
        Duplicate(q16Row, static_cast<int16_t>(0), static_cast<uint32_t>(q16RowElemsAlign_));
        PipeBarrier<PIPE_V>();

        // rope: bf16 -> bf16 位型直通 (禁止经 f32 数值往返)
        DataCopy(rowB16[rowB16Base], xLocal[xRowBase + nopeElems_], static_cast<uint32_t>(ropeElems_));

        // x -> f32 提前转换 (WholeReduceMax 的 src 读不被硬件 scoreboard 跟踪, 会在
        // Cast 写完成前抢读得到部分窗口的 amax; 提前 8 个 V 操作 + 屏障拉开写读距离)
        for (int64_t g = 0; g < scaleCol_; g++) {
            Cast<float, T0>(f32Row[g * perGroupSize_], xLocal[xRowBase + g * perGroupSize_], RoundMode::CAST_NONE,
                            groupU32);
        }
        PipeBarrier<PIPE_V>();

        // pass1: 逐组 Abs + 两级归约 (apply_adam_w_quant fp32 生产模式, 与本机同架构验证):
        // BlockReduceMax 组内 64->8 部分最大值 (arg-mask 形式), PipeBarrier, 再 WholeReduceMax 8->1
        // level1 槽位 scaleRow[g*128] (512B 槽位约束宽写), level2 槽位 scaleRow[896+g*8]
        for (int64_t g = 0; g < scaleCol_; g++) {
            Abs(f32Row[g * perGroupSize_], f32Row[g * perGroupSize_], groupU32);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t g = 0; g < scaleCol_; g++) {
            BlockReduceMax<float>(scaleRow[g * SCALE_SLOTS_A22], f32Row[g * perGroupSize_], 1, groupU32, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t g = 0; g < scaleCol_; g++) {
            WholeReduceMax<float>(scaleRow[896 + g * 8], scaleRow[g * SCALE_SLOTS_A22], 8, 1, 1, 1, 1,
                                  ReduceOrder::ORDER_ONLY_VALUE);
        }
        PipeBarrier<PIPE_V>();
    }

    // 排空块: WholeReduceMax 的 dst 写不被 PipeBarrier/V_S 同步跟踪 (实测迟到异步写),
    // 末行归约后无后续 V 工作, 这里用常量 Duplicate 忙等一个行周期的 V 流水,
    // 确保所有行的归约写在下面的标量读回前落盘 (除数区内容随后被 Broadcast 整体覆盖)
    for (int64_t r = 0; r < rf; r++) {
        LocalTensor<float> f32Row = f32All[r * F32_ROW_ELEMS_A22];
        for (int64_t g = 0; g < scaleCol_; g++) {
            Duplicate(f32Row[divOff + g * perGroupSize_], 1.0f, groupU32);
        }
    }
    PipeBarrier<PIPE_V>();

    for (int64_t r = 0; r < rf; r++) {
        LocalTensor<float> f32Row = f32All[r * F32_ROW_ELEMS_A22];
        LocalTensor<int16_t> q16Row = q16All[r * q16RowElemsAlign_];
        LocalTensor<float> scaleRow = scaleAll[r * SCALE_SLOTS_A22 * SCALE_BLKS_A22];
        int64_t xRowBase = r * xRowElemsAlign_;
        int64_t rowB16Base = r * outRowBytesAlign_ / sizeof(T0);
        int64_t rowByteBase = r * outRowBytesAlign_;

        // amax 读回 (V_S 同步后 S-pipe 读, 无除法); 紧凑区先放 amax 供 Broadcast
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
        for (int64_t g = 0; g < scaleCol_; g++) {
            float amax = scaleRow.GetValue(896 + g * 8);
            f32Row.SetValue(scaleF32Off + g, amax);
        }
        // S_V: 紧凑 amax(S 写) -> Broadcast(V 读) 的 RAW 同步
        SetFlag<HardEvent::S_V>(eventSToV);
        WaitFlag<HardEvent::S_V>(eventSToV);

        // scale = amax/127 硬件向量 Div (商 ~1ulp; <=1 LSB 口径下 bf16 舍入翻转 <=1 字节,
        // 无需残差校正; 原 PreciseDivA22 链已移除, 历史实现见 arch22_kv_com_chuanci 分支)。
        // amax 副本放商区 [0,448) (scale 阶段商区空闲), scale 输出就取除数区)
        Duplicate(f32Row[dup127WideOff], INT8_ABS_MAX_A22, nopeU32);
        uint32_t srcShape[2] = {scaleColU32, 1};
        uint32_t dstShape[2] = {scaleColU32, groupU32};
        Broadcast<float, 2, 1>(f32Row[divOff], f32Row[scaleF32Off], dstShape, srcShape);
        Muls(f32Row, f32Row[divOff], 1.0f, nopeU32);
        PipeBarrier<PIPE_V>();
        Div(f32Row[divOff], f32Row, f32Row[dup127WideOff], nopeU32);
        PipeBarrier<PIPE_V>();

        // scale 读回: 除数区每组首元素 (宽 Div 结果, IEEE); ==0 (amax==0 组) 置 1.0,
        // 覆盖紧凑区 amax -> 紧凑 scale, 供输出 bf16 Cast
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
        for (int64_t g = 0; g < scaleCol_; g++) {
            float scaleDiv = f32Row.GetValue(divOff + g * perGroupSize_);
            float scaleVal = (scaleDiv == 0.0f) ? 1.0f : scaleDiv;
            f32Row.SetValue(scaleF32Off + g, scaleVal);
        }
        // S_V: 紧凑 scale(S 写) -> 行尾 scale Cast/重播 Broadcast(V 读) 的 RAW 同步
        SetFlag<HardEvent::S_V>(eventSToV);
        WaitFlag<HardEvent::S_V>(eventSToV);

        // scale f32 -> bf16 (CAST_RINT 就近舍入) 写入行 [576, 590)
        Cast<T0, float>(rowB16[rowB16Base + scaleByteOff_ / sizeof(T0)], f32Row[scaleF32Off], RoundMode::CAST_RINT,
                        scaleColU32);
        PipeBarrier<PIPE_V>();

        // 零值修正后的紧凑 scale 重播入除数区 (amax==0 组除数置 1.0, 防除零;
        // 与输出 scale 同源, 保持 golden 语义)
        Broadcast<float, 2, 1>(f32Row[divOff], f32Row[scaleF32Off], dstShape, srcShape);
        PipeBarrier<PIPE_V>();

        // pass2: 分组量化, q = clip(RNE(x/scale)), 与 golden 同序 (先 rint 后 clip)。
        // 验收口径: int8 输出 <= 1 LSB (ATK kvce_int8_lsb1)。DAV_2201 向量 Div 商误差 ~1ulp,
        // RINT 结果与数学真值 RNE 最多相差 1 (商恰跨 x.5 边界时翻转), 天然满足 <=1 LSB;
        // 逐字节精确机制 (残差校正链 + 危险窗口 + 标量精确重算) 已按口径变更移除,
        // 历史实现见 arch22_kv_com_chuanci 分支。
        // [960,1408) 复用为 x 副本
        for (int64_t g = 0; g < scaleCol_; g++) {
            Cast<float, T0>(f32Row[dup127WideOff + g * perGroupSize_], xLocal[xRowBase + g * perGroupSize_],
                            RoundMode::CAST_NONE, groupU32);
        }
        PipeBarrier<PIPE_V>();
        Div(f32Row, f32Row[dup127WideOff], f32Row[divOff], nopeU32); // qf (~1ulp, <=1 LSB 口径内)
        PipeBarrier<PIPE_V>();
        Cast<int16_t, float>(q16Row, f32Row, RoundMode::CAST_RINT, nopeU32); // q0 = RNE(qf) (s16)
        PipeBarrier<PIPE_V>();
        // clip 到 [-127,127] (int16 域, 与 golden clip(rint) 同序; 标量为 int16 无精度截断问题)
        Maxs(q16Row, q16Row, static_cast<int16_t>(-INT8_ABS_MAX_A22), nopeU32);
        Mins(q16Row, q16Row, static_cast<int16_t>(INT8_ABS_MAX_A22), nopeU32);
        PipeBarrier<PIPE_V>();
        // s16->half->int8: DAV_2201 无 int16->int8 直通, 经 half 桥接 (|q|<=127 两段转换均精确);
        // half 缓冲复用除数区 [divOff,divOff+448) (Div 后不再使用), 避免 s16/half 同址转换 hazard
        LocalTensor<half> h16Row2 = f32Row[divOff].template ReinterpretCast<half>();
        Cast<half, int16_t>(h16Row2, q16Row, RoundMode::CAST_NONE, nopeU32);
        PipeBarrier<PIPE_V>();
        Cast<int8_t, half>(rowI8[rowByteBase + nopeByteOff_], h16Row2, RoundMode::CAST_RINT, nopeU32);
    }
}

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilogArch22<T0, U, T1>::Scatter(const LocalTensor<T1>& kvLocal, int64_t slotBase,
                                                                  int64_t rf)
{
    // slot 直读 GM, slot==-1 丢弃; 行 GM 地址 = blockIdx * blkStride + posInBlock * rowStride
    const int64_t blkSize = tilingData->kvCacheBlockSize;
    const int64_t blkStride = tilingData->kvCacheBlockStride;
    const int64_t rowStride = tilingData->kvCacheRowStride;
    for (int64_t r = 0; r < rf; r++) {
        int64_t slot = static_cast<int64_t>(slotMappingGm.GetValue(slotBase + r));
        if (slot == -1) {
            continue;
        }
        int64_t block = slot / blkSize;
        int64_t posInBlock = slot % blkSize;
        int64_t gmOffset = block * blkStride + posInBlock * rowStride;
        CopyOutRowA22(kvCacheGm[gmOffset], kvLocal[r * outRowBytesAlign_], static_cast<uint32_t>(kvCacheCol_));
    }
}

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilogArch22<T0, U, T1>::Process()
{
    int64_t rowOuterLoop =
        (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
    int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                   tilingData->tailRowFactorOfFormerBlock;
    if (rowOuterLoop <= 0) {
        return;
    }
    const int64_t rowFactorC = tilingData->rowFactor;
    const int64_t slotBlkBase = GetBlockIdx() * tilingData->rowOfFormerBlock;
    const uint32_t dCol = static_cast<uint32_t>(d_);
    const uint32_t srcStride = static_cast<uint32_t>(xRowStride_ - d_);

    int64_t rf0 = (rowOuterLoop == 1) ? tailRowFactor : rowFactorC;
    LocalTensor<T0> x0 = xQue.template AllocTensor<T0>();
    CopyInRowsA22(xGm[0], x0, static_cast<uint16_t>(rf0), dCol, srcStride);
    xQue.template EnQue(x0);

    LocalTensor<T1> prevKv;
    int64_t prevSlotBase = 0;
    int64_t prevRF = 0;
    bool havePrev = false;

    for (int64_t i = 0; i < rowOuterLoop; i++) {
        int64_t curRowFactor = (i == rowOuterLoop - 1) ? tailRowFactor : rowFactorC;
        // 预取下一块搬入 (其 MTE2 与本块量化、上一块散写并行)
        if (i + 1 < rowOuterLoop) {
            int64_t nextRF = (i + 1 == rowOuterLoop - 1) ? tailRowFactor : rowFactorC;
            int64_t nextXBase = (i + 1) * rowFactorC * d_;
            LocalTensor<T0> xn = xQue.template AllocTensor<T0>();
            CopyInRowsA22(xGm[nextXBase], xn, static_cast<uint16_t>(nextRF), dCol, srcStride);
            xQue.template EnQue(xn);
        }
        // 量化本块 (VEC)
        LocalTensor<T0> xl = xQue.template DeQue<T0>();
        LocalTensor<T1> curKv = kvCacheQue.template AllocTensor<T1>();
        Compute(curKv, xl, curRowFactor);
        xQue.template FreeTensor(xl);
        kvCacheQue.template EnQue(curKv);

        if (havePrev) {
            Scatter(prevKv, prevSlotBase, prevRF);
            kvCacheQue.template FreeTensor(prevKv);
        }
        prevKv = kvCacheQue.template DeQue<T1>();
        prevSlotBase = slotBlkBase + i * rowFactorC;
        prevRF = curRowFactor;
        havePrev = true;
    }
    if (havePrev) {
        Scatter(prevKv, prevSlotBase, prevRF);
        kvCacheQue.template FreeTensor(prevKv);
    }
}

} // namespace KvCompressEpilogOps

#endif // KV_COMPRESS_EPILOG_KERNEL_ARCH22_H
