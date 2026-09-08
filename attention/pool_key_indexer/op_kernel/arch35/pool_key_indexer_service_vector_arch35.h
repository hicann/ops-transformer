/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_SERVICE_VECTOR_ARCH35_H
#define POOL_KEY_INDEXER_SERVICE_VECTOR_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"
#include "../arch35/vf/pool_key_indexer_vector1.h"
#include "../arch35/vf/pool_key_indexer_topk.h"

namespace PkiKernel {
using namespace PkiCommon;
constexpr uint32_t TRUNK_LEN_8K = 8192;
constexpr uint32_t TRUNK_LEN_4K = 4096;
constexpr uint32_t TRUNK_LEN_2K = 2048;
constexpr uint32_t TOPK_LEN_7K = 7168;
constexpr uint32_t TOPK_LEN_5K = 5120;

template <typename Q_T, typename W_T = void>
struct PkiTypeTraits {
    using weightsType = Q_T; // 默认：weightsType绑定Q_T
};

template <typename Q_T>
struct PkiTypeTraits<Q_T, float> {
    using weightsType = float; // W_T=float时，强制weightsType为float
};

// FP8模式：weights张量为半精度(FP16/BF16, 由 entry 按 ORIG_DTYPE_WEIGHTS 分发,
// 见 op_host def.cpp dtype 约束: quant_mode=0/1时 q/k=FP8_E4M3FN, weights=FP16/BF16)。
// W_T必须绑定为半精度类型：
// 1) 功能上weights以半精度解释才是正确数据
// 2) 若W_T=fp8_e4m3fn_t，vector侧weights搬运/加载路径（DataCopyPadExtParams<W_T>、
//    LoadAlign<W_T>等）会实例化fp8标量语义，bisheng后端报错：
//    "fp8...type only supports pointer operations, scalar float type semantics
//    are not supported"
// 量化场景的 weightsType 由 PkiType 显式携带(half/bfloat16_t), 此处泛型分发
template <typename W_HALF_T>
struct PkiQuantWeightsTraits {
    using weightsType = W_HALF_T;
};
template <>
struct PkiTypeTraits<fp8_e4m3fn_t> {
    using weightsType = half;
};
template <typename LIT>
class PoolKeyIndexerServiceVector {
public:
    // =================================类型定义区=================================
    static constexpr PkiLayout LAYOUT_T = LIT::layout;
    static constexpr PkiLayout K_LAYOUT_T = LIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using SCORE_T = uint32_t;
    // 量化场景: PkiType 显式携带 weightsType(half/bfloat16_t), 优先于推导;
    // 非量化(WEIGHTS_T=void)沿用 PkiTypeTraits 推导
    using W_DERIVED_T =
        typename PkiTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;
    using W_EXPLICIT_T = typename LIT::weightsType;
    using W_T = typename std::conditional<std::is_same<W_EXPLICIT_T, void>::value, W_DERIVED_T, W_EXPLICIT_T>::type;

    __aicore__ inline PoolKeyIndexerServiceVector(){};
    __aicore__ inline void ProcessVec1(const PkiCommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const PkiCommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct PkiCommon::ConstInfo &constInfo,
                                      const PoolKeyIndexerTilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<float> valueOutGm, GlobalTensor<int32_t> blockTableGm,
                                              GlobalTensor<float> qScaleGm = {}, GlobalTensor<float> kScaleGm = {});
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

private:
    __aicore__ inline void ExpandAndAppendIndices(LocalTensor<int32_t> poolIndices, LocalTensor<int32_t> &tokenIndices,
                                                  LocalTensor<int32_t> &workBuf, uint32_t sparseCount,
                                                  uint32_t poolSize, uint32_t validS2Len, int32_t poolTailK,
                                                  int32_t L_orig, uint32_t curS1Idx, uint32_t curS1Size);
    // mode=0: k_descale GM -> UB 搬运(非 PA 连续寻址 / PA 按 block_table
    // × keyDequantScaleStride0, 参考 QLIv2 GetKeyScale)。
    // dstOffset: 乒乓区内本 s2 块的段偏移((s2Idx-s2Start)%16 * s2BaseSize),
    // 写入与读取(MulWeightAndReduceSumWithScale)位置一致
    __aicore__ inline void GetKeyScale(LocalTensor<float> kScaleUB, uint64_t keyScaleGmOffset, int64_t batchId,
                                       int64_t startS2, int64_t getLen, uint32_t dstOffset = 0);

    // arch35(950) 标量(S pipe)与向量(V pipe)/MTE3 间必须显式硬同步,
    // PipeBarrier<PIPE_V> 不保证 S 侧读写顺序(参照 bsa_select_block_mask
    // arch35 的 VToSSync/SToVSync/SToMTE3Sync 模式)
    __aicore__ inline void VToSSync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }
    __aicore__ inline void SToVSync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }
    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventID);
        WaitFlag<HardEvent::S_MTE3>(eventID);
    }

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // mode=0 (FP8 per-token-head) scale GM(q/k_descale, float)
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;
    // mode=0 k_descale 搬运: block_table 标量读(S pipe)后 MTE2 搬运的同步
    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;

private:
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<float> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;
    // mode=0: weight(fp32 cast) × qScale 预乘结果的 fp32 UB(向量 SIMD 预乘,
    // 参考 QLIV2 weightTempUB_; 替代逐 g 标量预乘)
    TBuf<TPosition::VECCALC> weightScaleFp32Buf_;
    LocalTensor<float> weightScaleFp32UB_;

    // mode=0 (FP8 per-token-head) scale UB(乒乓, 参考 QLIv2 qScaleBuf_/kScaleBuf_)
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<float> qScaleUB_;
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<float> kScaleUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;

    // tmp buff for returnValue K_T
    TBuf<TPosition::VECCALC> valueOutBuf_;
    LocalTensor<float> valueOutLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> expandOutBuf_;
    LocalTensor<int32_t> expandOutLocal_;
    TBuf<TPosition::VECCALC> workBuf_;
    LocalTensor<int32_t> workLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;
    int32_t blockId_ = -1;
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0;
    uint32_t trunkLen_ = 0;
    uint32_t poolSize_ = 1;
    uint32_t outputLen_ = 0;
    uint32_t kScaleLoop_ = 0; // mode=0: k_descale 16 轮乒乓搬运计数
    bool returnValue = false;

    struct PkiCommon::ConstInfo constInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(float));
    resMm1UB_ = resMm1Buf_.Get<float>();

    // weightBuf_: 行距 Align(gSize,16)。
    // 注意: MulWeightQScaleSIMD(量化 mode=0)的 LoadAlign<W_T, DIST_UNPACK_B16>
    // 每行固定读 64 个 W_T 元素; gSizeAlign16 < 64 时, 最后一个 pingpong 区
    // 末行的读取越出缓冲末端 (64 - gSizeAlign16) 个元素, 触发 VEC UB 越界
    // (aicore 507015, 实测 gSize=24/行距 32 时量化+0池 batch 崩溃)。
    pipe->InitBuffer(weightBuf_,
                     2 * CeilDiv(s1BaseSize_, 2) * PkiCommon::Align((uint64_t)gSize_, (uint64_t)16) * sizeof(W_T));
    weightUB_ = weightBuf_.Get<W_T>();
    if constexpr (LIT::isFp8PerToken) {
        // mode=0: weight×qScale SIMD 预乘输出缓冲(fp32, 行距 128 元素 bank 对齐,
        // 与 qScaleUB 同布局)。向量预乘替代原逐 g 标量循环(gSize 次标量 ->
        // 每行 3~4 条向量指令, W_T=bfloat16_t 也不再实例化 bf16 标量语义)
        pipe->InitBuffer(weightScaleFp32Buf_, 2 * CeilDiv(s1BaseSize_, 2) * 128 * sizeof(float));
        weightScaleFp32UB_ = weightScaleFp32Buf_.Get<float>();
        // mode=0 scale UB: qScale 行距固定 128 元素(512B bank 对齐)——
        // MulWeightAndReduceSumWithScale 的 LoadAlign<float> 一次取 64 lanes
        // (256B), 要求行起点 256B 对齐; 紧凑 gSizeAlign16 布局在 n1 非 64
        // 倍数时行距非 256B 对齐, LoadAlign 错位(参考 QLIv2 的 bank 交错布局)。
        // kScale 为 s2 维度(16 轮乒乓 × s2BaseSize, 参考 QLIv2 kScaleBuf_)
        pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * 128 * sizeof(float));
        qScaleUB_ = qScaleBuf_.Get<float>();
        pipe->InitBuffer(kScaleBuf_, 2 * 16 * s2BaseSize_ * sizeof(float));
        kScaleUB_ = kScaleBuf_.Get<float>();
        // kScale 铺 0(参考 QLIv2): 尾部未搬运区(如 S2 尾块不足 s2BaseSize)是
        // 垃圾值, 乘入聚合和会产出随机大分数污染 TopK(分数恒>=0, 0 与 ReLU
        // 后被 mask 的无效池语义一致)
        Duplicate(kScaleUB_, 0.0f, 2 * 16 * s2BaseSize_);
        SetFlag<HardEvent::V_MTE2>(KSCALE_S_MTE2_EVENT);
        WaitFlag<HardEvent::V_MTE2>(KSCALE_S_MTE2_EVENT);
    }
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    vec1OutUB_ = outBuf_.Get<SCORE_T>(); // out

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    // returnvalue
    if (topkCount_ <= 2048) {
        pipe->InitBuffer(valueOutBuf_, topkCountAlign256_ * sizeof(float));
        valueOutLocal_ = valueOutBuf_.Get<float>();
    } else {                                        // sparseCount > 2k时，复用return value相关UB
        valueOutLocal_ = mrgValueBuf_.Get<float>(); // returnValue float
    }

    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    if (poolSize_ > 1) {
        uint32_t expandOutLen = PkiCommon::Align(static_cast<uint64_t>(outputLen_ + 64), (uint64_t)8);
        pipe->InitBuffer(expandOutBuf_, expandOutLen * sizeof(uint32_t));
        expandOutLocal_ = expandOutBuf_.Get<int32_t>();
        uint32_t workSize = PkiCommon::Align(static_cast<uint64_t>(poolSize_ + 64), (uint64_t)8) +
                            PkiCommon::Align(static_cast<uint64_t>(outputLen_ + 64), (uint64_t)8);
        pipe->InitBuffer(workBuf_, workSize * sizeof(uint32_t));
        workLocal_ = workBuf_.Get<int32_t>();
        // 展开模板 0..ps-1 一次性构建并常驻 workLocal_ 头部(kernel 生命周期
        // 内不变)。CreateVecIndex 为 V pipe 向量写, 与 ExpandAndAppendIndices
        // 中消费它的 Add 同 pipe(中间隔多次向量操作, 同 pipe 有序性足够,
        // 与既有 CreateVecIndex→Duplicate+PipeBarrier 用法一致), 替代原先
        // 每行 ps 次 SetValue 标量重建 + SToVSync 硬同步的逐行开销
        AscendC::CreateVecIndex(workLocal_, static_cast<int32_t>(0), poolSize_);
    }

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitParams(
    const struct PkiCommon::ConstInfo &constInfo, const PoolKeyIndexerTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    returnValue = constInfo.returnValue;
    blockId_ = GetBlockIdx();
    trunkLen_ = constInfo.sparseCount > TOPK_LEN_5K ?
                    (constInfo.sparseCount > TOPK_LEN_7K ? TRUNK_LEN_2K : TRUNK_LEN_4K) :
                    TRUNK_LEN_8K;
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, trunkLen_);
    topkCountAlign256_ = PkiCommon::Align(constInfo.sparseCount, (uint64_t)256);
    poolSize_ = static_cast<uint32_t>(constInfo.poolSize);
    outputLen_ = constInfo.sparseCount * poolSize_ + poolSize_ - 1;
    kScaleLoop_ = 0;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitVecInputTensor(
    GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm, GlobalTensor<float> valueOutGm,
    GlobalTensor<int32_t> blockTableGm, GlobalTensor<float> qScaleGm, GlobalTensor<float> kScaleGm)
{
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->blockTableGm = blockTableGm;
    // mode=0 (FP8 per-token-head): vector 侧 scale 融合
    if constexpr (LIT::isFp8PerToken) {
        this->qScaleGm = qScaleGm;
        this->kScaleGm = kScaleGm;
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

// mode=0: k_descale GM -> UB(每池 1 个 float scale)
// 非 PA: 连续寻址 tensorKeyScaleOffset + startS2 起 getLen 个;
// PA: 按 block_table 逐物理块搬运, 块内偏移 + 物理块基址 blockId * keyDequantScaleStride0
// (参考 QLIv2 QLIV2Vector::GetKeyScale)
template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::GetKeyScale(LocalTensor<float> kScaleUB,
                                                                     uint64_t keyScaleGmOffset, int64_t batchId,
                                                                     int64_t startS2, int64_t getLen,
                                                                     uint32_t dstOffset)
{
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    if constexpr (PAGE_ATTENTION) {
        // 按池号连续写入 UB 段(dstOffset 起 getLen 个), GM 源按物理块跳转。
        // 注意 kCacheBlockSize_(=pa_block_size) 可大于 s2BaseSize(段容量),
        // 段内偏移始终用相对本段起点的池号, 与块边界解耦。
        int64_t written = 0; // 段内已写池数
        while (written < getLen) {
            int64_t absPool = startS2 + written; // batch 内绝对池号
            int64_t s2BlkId = absPool / kCacheBlockSize_;
            int64_t s2BlkOffset = absPool % kCacheBlockSize_;
            int64_t blockId = blockTableGm.GetValue(batchId * maxBlockNumPerBatch_ + s2BlkId);
            int64_t partLen = Min((int64_t)kCacheBlockSize_ - s2BlkOffset, getLen - written);
            copyInParams.blockLen = partLen * sizeof(float);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[dstOffset + static_cast<uint32_t>(written)],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0 + s2BlkOffset], copyInParams,
                                 padParams);
            written += partLen;
        }
    } else {
        copyInParams.blockLen = getLen * sizeof(float);
        AscendC::DataCopyPad(kScaleUB[dstOffset], kScaleGm[keyScaleGmOffset], copyInParams, padParams);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    // 与 ProcessVec1 无效行路径(validS2Len<=0)同构: Duplicate 铺 UB → V_MTE3
    // 硬同步 → DataCopyPad 出 GM → MTE3_V 回执。三点缺一不可:
    // 1) DataCopyPad: indices 行距 outputLen(=sparseCount*ps+ps-1) 非 32B 对齐,
    //    普通 DataCopy 的 MTE3 源/目的对齐校验失败(aivec 81 地址越界);
    // 2) 入口 WaitFlag<MTE3_V>: 等上一个 MTE3(TopK/Vec1/上次 clean)读完 staging
    //    (expandOutLocal_/indicesOutLocal_/valueOutLocal_ 复用)再覆写;
    // 3) 行尾 SetFlag<MTE3_V>: 维持事件链, 否则后续 ProcessVec1/FreeEventID 的
    //    WaitFlag<MTE3_V> 挂死。InitGlobalMemory 在 arch35 AIV 上以内部向量
    //    staging 直写 GM, dealSize 大时越界(aivec 341), 故弃用。
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = (poolSize_ > 1 ? outputLen_ : topkCount_) * sizeof(uint32_t);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    if (poolSize_ > 1) {
        Duplicate(expandOutLocal_, constInfo_.INVALID_IDX, PkiCommon::Align(outputLen_, (uint32_t)8));
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], expandOutLocal_, copyOutParams);
    } else {
        Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), constInfo_.INVALID_IDX, topkCount_);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
    }
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    if (returnValue) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>(), constInfo_.INVALID_VAL, topkCount_);

        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = topkCount_ * sizeof(float);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        // invalidS1Offset 是 indices 行偏移(行宽 outputLen_/sparseCount);
        // value 行宽为 sparseCount, 需换算行号后重算偏移, 否则越界写且本行 value 漏写
        uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
        uint64_t valueOffset = (static_cast<uint64_t>(invalidS1Offset) / idxStride) * constInfo_.sparseCount;
        AscendC::DataCopyPad(valueOutGm[valueOffset], valueOutLocal_, copyOutValueParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ExpandAndAppendIndices(
    LocalTensor<int32_t> poolIndices, LocalTensor<int32_t> &tokenIndices, LocalTensor<int32_t> &workBuf,
    uint32_t sparseCount, uint32_t poolSize, uint32_t validS2Len, int32_t poolTailK, int32_t L_orig, uint32_t curS1Idx,
    uint32_t curS1Size)
{
    uint32_t topk = sparseCount * poolSize;
    uint32_t totalOut = topk + poolSize - 1;
    uint32_t alignedTotalOut = PkiCommon::Align(totalOut, (uint32_t)8);

    // 尾块可见 token 数(两条路径共用)
    int32_t visibleTailK = 0;
    if (poolTailK > 0) {
        if (constInfo_.maskMode == 0) {
            visibleTailK = poolTailK;
        } else {
            int32_t globalPosQ = L_orig - static_cast<int32_t>(curS1Size) + static_cast<int32_t>(curS1Idx);
            visibleTailK = PkiCommon::Max(0, PkiCommon::Min(poolTailK, globalPosQ - static_cast<int32_t>(topk) + 1));
        }
    }

    // poolSize 非 8 的倍数: 向量写偏移(outOff=k*poolSize / tailPos=topk /
    // validExpand=v*poolSize)不满足 32B 操作数地址对齐, 触发 aicore
    // exception(EE9999/507015; 判别实验: sc=1+tail=0 的唯一对齐形态不崩,
    // 尾块/第2轮起展开必崩)。退化标量精确写路径: 先向量铺 -1(offset 0
    // 对齐)再逐 token 标量写, 无对齐要求; 标量写精确到 pool 边界, 同时
    // 消除向量按 Align(ps,8) 对齐写越出 pool 边界残留垃圾索引的问题。
    if (poolSize % 8 != 0) {
        Duplicate<int32_t>(tokenIndices, -1, alignedTotalOut);
        // V→S: 向量 -1 填充必须先于标量写落地(迟到的 V 写会覆盖标量写);
        // 同时保证 poolIndices(topkOp_ 向量写)对标量 GetValue 可见
        VToSSync();
        if (validS2Len > 0) {
            uint32_t expandRounds = PkiCommon::Min(validS2Len, sparseCount);
            for (uint32_t k = 0; k < expandRounds; k++) {
                int32_t base = poolIndices.GetValue(k) * static_cast<int32_t>(poolSize);
                uint32_t off = k * poolSize;
                for (uint32_t p = 0; p < poolSize; p++) {
                    tokenIndices.SetValue(off + p, base + static_cast<int32_t>(p));
                }
            }
        }
        for (int32_t t = 0; t < visibleTailK; t++) {
            tokenIndices.SetValue(topk + static_cast<uint32_t>(t), L_orig - poolTailK + t);
        }
        // SCALAR(SetValue) 写 UB 后由 MTE3(DataCopyPad) 读出, 需硬同步
        SToMTE3Sync();
        return;
    }

    uint32_t alignedPoolSize = PkiCommon::Align(poolSize, (uint32_t)8);

    Duplicate<int32_t>(tokenIndices, -1, alignedTotalOut);
    PipeBarrier<PIPE_V>();

    if (validS2Len > 0) {
        // V→S: poolIndices(indicesOutLocal_) 的最近写入方为向量操作
        // (topkOp_ 结尾 DataCopy / CreateVecIndex / -1 Duplicate 填充),
        // 后续标量 GetValue 必须等其可见, 否则读到陈旧/垃圾 pool 序号
        VToSSync();

        // 展开模板 offsetTpl(0..ps-1) 已在 InitBuffers 经 CreateVecIndex
        // 一次性构建并常驻, 无需逐行标量重建(省 ps 次 SetValue + SToVSync)
        LocalTensor<int32_t> offsetTpl = workBuf;

        uint32_t outOff = 0;
        // 展开轮数受 sparseCount(TopK 选池数)截断: validS2Len 为 causal 可见池数,
        // 可见池多于选中池时, 多余池不应展开(否则越写 outputLen 之外污染尾区/相邻行)
        uint32_t expandRounds = PkiCommon::Min(validS2Len, sparseCount);
        for (uint32_t k = 0; k < expandRounds; k++) {
            int32_t base = poolIndices.GetValue(k) * static_cast<int32_t>(poolSize);
            Duplicate<int32_t>(tokenIndices[outOff], base, alignedPoolSize);
            PipeBarrier<PIPE_V>();
            Add<int32_t>(tokenIndices[outOff], tokenIndices[outOff], offsetTpl, alignedPoolSize);
            outOff += poolSize;
        }
    }

    uint32_t validExpand = validS2Len * poolSize;
    if (validExpand < topk) {
        PipeBarrier<PIPE_V>();
        Duplicate<int32_t>(tokenIndices[validExpand], -1, PkiCommon::Align(topk - validExpand, (uint32_t)8));
        PipeBarrier<PIPE_V>();
    }

    if (visibleTailK > 0) {
        uint32_t tailPos = topk;
        // 先按 8 对齐清空整个尾区, 再标量精确写 visibleTailK 个尾 token;
        // 不可按 Align(visibleTailK,8) 对齐写, 否则会越出 tail 容量(ps-1)
        // 污染行尾/相邻行(与 arch22 修复一致)
        Duplicate<int32_t>(tokenIndices[tailPos], -1, PkiCommon::Align(totalOut - tailPos, (uint32_t)8));
        // V→S: 上述 -1 向量填充必须先于标量 SetValue 落地,
        // 否则迟到的 V 写会覆盖标量写(实测尾区全 -1)
        VToSSync();
        for (int32_t t = 0; t < visibleTailK; t++) {
            tokenIndices.SetValue(tailPos + t, L_orig - poolTailK + t);
        }
        // SCALAR(SetValue) 写 UB 后由 MTE3(DataCopyPad) 读出, 需硬同步
        SToMTE3Sync();
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ProcessVec1(const PkiCommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    // CV同步, V核等C核计算完mm1，mm1Res已搬运到UB
    CrossCoreWaitFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_CV_EVENT + pingpong);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    if (curAivS1ProcNum == 0) {
        // V核处理完，通知C核可以把mm1Res搬运到UB
        CrossCoreSetFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_VC_EVENT + pingpong);
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    // weightsGm --> weightUB_
    uint64_t gSizeAlign16 = PkiCommon::Align((uint64_t)gSize_, (uint64_t)16);
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    DataCopyPadExtParams<W_T> padWeightsParams{true, 0, 0, 0};
    DataCopyExtParams qwDataCopyExtParams;
    qwDataCopyExtParams.blockCount = curAivS1ProcNum;
    qwDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
    qwDataCopyExtParams.srcStride = 0;
    qwDataCopyExtParams.dstStride = (gSizeAlign16 - gSize_) * sizeof(W_T) / 32;
    DataCopyPad(weightUB_[pingpong * CeilDiv(s1BaseSize_, 2) * gSizeAlign16], weightsGm[weightGmOffset],
                qwDataCopyExtParams, padWeightsParams);

    if constexpr (LIT::isFp8PerToken) {
        // mode=0: q_descale 与 weights 同 pattern 同 GM 偏移(shape 同构,
        // 参考 QLIv2 直接复用 weightGmOffset 的做法)。dst 行距 128 元素
        // (512B bank 对齐, 与 qScaleUB 布局一致; 块间 dstStride 单位 32B)
        DataCopyPadExtParams<float> padQScaleParams{true, 0, 0, 0};
        DataCopyExtParams qScaleDataCopyExtParams;
        qScaleDataCopyExtParams.blockCount = curAivS1ProcNum;
        qScaleDataCopyExtParams.blockLen = gSize_ * sizeof(float);
        qScaleDataCopyExtParams.srcStride = 0;
        qScaleDataCopyExtParams.dstStride = (128 - gSize_) * sizeof(float) / 32;
        DataCopyPad(qScaleUB_[pingpong * CeilDiv(s1BaseSize_, 2) * 128], qScaleGm[weightGmOffset],
                    qScaleDataCopyExtParams, padQScaleParams);
        // k_descale: 每 16 轮 s2BaseSize 乒乓搬运一次(参考 QLIv2 kScale 机制:
        // (s2Idx - s2Start) % 16 == 0 时刷新, 一次刷 16 块容量滚动覆盖)。
        // 写入起点 = 乒乓区头 + 本块段偏移((s2Idx-s2Start)%16 * s2BaseSize),
        // 与读取位置一致(读侧 MulWeightAndReduceSumWithScale 同偏移取段)
        if ((info.s2Idx - info.s2Start) % 16 == 0) {
            uint32_t kScalePingpong = (kScaleLoop_ % 2);
            uint32_t kScaleBlkOff = ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_;
            uint32_t getLen = 16 * s2BaseSize_ > (info.actS2Size - info.s2Idx * s2BaseSize_) ?
                                  (info.actS2Size - info.s2Idx * s2BaseSize_) :
                                  16 * s2BaseSize_;
            // 钳制到本乒乓区剩余容量(理论不触发: 刷新点在段头, 防御保留)
            uint32_t capacity = 16 * s2BaseSize_ - kScaleBlkOff;
            if (getLen > capacity) {
                getLen = capacity;
            }
            GetKeyScale(kScaleUB_[kScalePingpong * 16 * s2BaseSize_], info.tensorKeyScaleOffset, info.bIdx, curS2Idx,
                        getLen, kScaleBlkOff);
            kScaleLoop_++;
        }
    }

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    if constexpr (LIT::isFp8PerToken) {
        // mode=0: weight(半精度) -> fp32 cast × qScale 的 SIMD 向量预乘落 UB。
        // 参考 QLIV2 MulWeightAndReduceSum2F32VF 的 weightTemp 模式
        // (LoadAlign + Mul + StoreAlign, 3~4 条向量指令/行), 替代逐 g 标量
        // 预乘循环(gSize 次/行; W_T=bfloat16_t 时还触发 bf16 标量语义限制)。
        // 行内 [gSize, 64) 补 0(LoadAlign 一次取 64 lanes, 0 与无效 g 语义一致)
        for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
            uint64_t dstOff = pingpong * CeilDiv(s1BaseSize_, 2) * 128 + s1IdxTmp * 128;
            Duplicate<float>(weightScaleFp32UB_[dstOff], 0.0f, 128);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
            uint64_t srcOff = pingpong * CeilDiv(s1BaseSize_, 2) * gSizeAlign16 + s1IdxTmp * gSizeAlign16;
            uint64_t dstOff = pingpong * CeilDiv(s1BaseSize_, 2) * 128 + s1IdxTmp * 128;
            // 向量预乘: 半精度 weight -> fp32 cast, 乘 qScale, 落 fp32 UB。
            // 独立 __simd_vf__ helper(同 QLI v1 的 vector1 函数模式):
            // bf16 的 Cast<float,bfloat16_t> 在 __simd_vf__ 内有硬件展开,
            // 内联到普通 __aicore__ 函数会触发 bisheng
            // "Do not know how to split the result of this operator"
            vector1::MulWeightQScaleSIMD((__local_mem__ W_T *)weightUB_[srcOff].GetPhyAddr(),
                                         (__local_mem__ float *)qScaleUB_[dstOff].GetPhyAddr(),
                                         (__local_mem__ float *)weightScaleFp32UB_[dstOff].GetPhyAddr());
        }
        // V 写落地后, WithScale 的向量 LoadAlign 才能读(同 pipe 有序性足够,
        // 但跨 ForEachS1 循环保险起见显式 barrier)
        PipeBarrier<PIPE_V>();
    }

    for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
        if constexpr (LIT::isFp8PerToken) {
            // mode=0: kScale 取本 s2 基本块对应的 float scale(逐 s2 元素;
            // 乒乓区起点 + 块内偏移 (s2Idx-s2Start)%16 * s2BaseSize)
            uint32_t kScalePingpong = ((kScaleLoop_ - 1) % 2);
            uint32_t kScaleBlkOff = ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_;
            vector1::MulWeightAndReduceSumWithScale(
                vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ + s1IdxTmp * s2BaseSize_],
                resMm1UB_[pingpong * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ + s1IdxTmp * gSize_ * s2BaseSize_],
                weightScaleFp32UB_[pingpong * CeilDiv(s1BaseSize_, 2) * 128 + s1IdxTmp * 128],
                kScaleUB_[kScalePingpong * 16 * s2BaseSize_ + kScaleBlkOff], gSize_, constInfo_.qkScale);
        } else {
            vector1::MulWeightAndReduceSum(
                vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ + s1IdxTmp * s2BaseSize_],
                resMm1UB_[pingpong * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ + s1IdxTmp * gSize_ * s2BaseSize_],
                weightUB_[pingpong * CeilDiv(s1BaseSize_, 2) * gSizeAlign16 + s1IdxTmp * gSizeAlign16], gSize_,
                constInfo_.qkScale);
        }
    }
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    // outUB_ --->  scoreGm
    uint64_t kSeqSizeAlign = PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_);
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx : CeilDiv(s1BaseSize_, 2) * kSeqSizeAlign + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = (kSeqSizeAlign - s2BaseSize_) * sizeof(SCORE_T);

    DataCopyPad(scoreGm[vec1OutGmOffset], vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_], copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    // V核处理完，通知C核可以把mm1Res搬运到UB
    CrossCoreSetFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_VC_EVENT + pingpong);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ProcessTopK(const PkiCommon::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = (poolSize_ > 1 ? outputLen_ : topkCount_) * sizeof(uint32_t);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;

        SCORE_T zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(constInfo_.poolSize);
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (poolSize_ > 1) {
                Duplicate(expandOutLocal_, neg, PkiCommon::Align(outputLen_, (uint32_t)8));
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * outputLen_],
                                     expandOutLocal_, copyOutParams);
            } else {
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                     indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            }
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (returnValue) {
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>(), constInfo_.INVALID_VAL, topkCount_);

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(float);
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                     copyOutValueParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            bool useSingleLoop =
                (s2LoopNum == 1) || ((topkCount_ > trunkLen_) && (validS2Len <= (uint32_t)topkCountAlign256_));
            if (useSingleLoop) {
                uint32_t validS2LenAlign = PkiCommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(
                    mrgValueLocal_,
                    scoreGm[vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, returnValue);
            } else {
                uint32_t actS2LoopNum = 0;
                if (topkCount_ > trunkLen_) {
                    actS2LoopNum = 1 + (validS2Len - topkCountAlign256_ + trunkLen_ - 1) / trunkLen_;
                } else {
                    actS2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
                }
                for (uint32_t loopIdx = 0; loopIdx < actS2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        if (topkCount_ > trunkLen_) {
                            copyInParams.blockLen = topkCountAlign256_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(scoreOutLocal_,
                                                 scoreGm[vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize,
                                                                                      (uint64_t)s2BaseSize_)],
                                                 copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero,
                                                    topkCountAlign256_);
                            AscendC::CreateVecIndex(topkSharedTmpLocal_.ReinterpretCast<int32_t>(), (int32_t)zero,
                                                    topkCountAlign256_);
                        } else {
                            copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(mrgValueLocal_,
                                                 scoreGm[vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize,
                                                                                      (uint64_t)s2BaseSize_)],
                                                 copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, actS2LoopNum,
                                    returnValue);
                        }
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = 0;
                    uint32_t offset = 0;
                    if (topkCount_ > trunkLen_) {
                        validTrunkLen = (topkCountAlign256_ + (loopIdx - 1) * trunkLen_ + trunkLen_) > validS2Len ?
                                            (validS2Len - topkCountAlign256_) % trunkLen_ :
                                            trunkLen_;
                        offset = vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 topkCountAlign256_ + (loopIdx - 1) * trunkLen_;
                    } else {
                        validTrunkLen =
                            (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                        offset = vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 loopIdx * trunkLen_;
                    }
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    // 如果是tok > trunklen, 第一轮每调用topk，是直接拷贝的，所以不需要刷零
                    bool isZeroPadding = (topkCount_ > trunkLen_) ? (loopIdx > 1) : true;
                    if (topkCountAlign256_ != topkCount_ && isZeroPadding) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero,
                                  topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero,
                                  PkiCommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            PkiCommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, actS2LoopNum,
                            returnValue);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
            if (returnValue) {
                copyInParams.blockLen = PkiCommon::Align(validS2Len, (int32_t)32) * sizeof(SCORE_T);
                AscendC::DataCopyPad(
                    scoreOutLocal_,
                    scoreGm[vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }

        if (validS2Len < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
        }

        if (validS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg,
                      topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        if (poolSize_ > 1) {
            PipeBarrier<PIPE_V>();
            ExpandAndAppendIndices(indicesOutLocal_.ReinterpretCast<int32_t>(), expandOutLocal_, workLocal_, topkCount_,
                                   poolSize_, static_cast<uint32_t>(validS2Len), info.poolTailK,
                                   static_cast<int32_t>(info.actS2SizeOrig), static_cast<uint32_t>(curAivS1Idx + i),
                                   info.actS1Size);
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        if (poolSize_ > 1) {
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * outputLen_], expandOutLocal_,
                                 copyOutParams);
        } else {
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        }

        // // 是否返回Value值
        if (returnValue) {
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            // uint32_t -> float
            vector1::UIntToFloatReturnValue(valueOutLocal_, scoreOutLocal_, topkCountAlign256_);

            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 16);
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>()[validS2Len / 16 * 16],
                          constInfo_.INVALID_VAL, mask, 1, 1, 0);
            }
            if (validS2Len / 16 * 16 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>()[validS2Len / 16 * 16 + 64],
                          constInfo_.INVALID_VAL, topkCount_ - (validS2Len / 16 * 16 + 64));
            }
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyParams copyOutValueParams;
            copyOutValueParams.blockCount = 1;
            copyOutValueParams.blockLen = topkCount_ * sizeof(float); // bytes
            copyOutValueParams.srcStride = 0;
            copyOutValueParams.dstStride = 0;
            // 搬运到GM
            AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                 copyOutValueParams);
        }
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}
} // namespace PkiKernel
#endif
