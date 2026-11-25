/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_KERNEL_ARCH35_H
#define POOL_KEY_INDEXER_KERNEL_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"
#include "pool_key_indexer_service_vector_arch35.h"
#include "pool_key_indexer_service_cube_arch35.h"

namespace PkiKernel {
using namespace PkiCommon;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bIdx = 0U;
    uint32_t bN2Idx = 0;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U; // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;  // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0U;
    uint32_t actS2SizeOrig = 0U; // 压缩前s2
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
};

template <typename LIT>
class PoolKeyIndexerKernel {
public:
    __aicore__ inline PoolKeyIndexerKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* poolKey, __gm__ uint8_t* weights,
                                __gm__ uint8_t* poolTailK, __gm__ uint8_t* actualSeqLengthsQ,
                                __gm__ uint8_t* actualSeqLengthsK, __gm__ uint8_t* blockTable, __gm__ uint8_t* qDescale,
                                __gm__ uint8_t* kDescale, __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues,
                                __gm__ uint8_t* workspace, const PoolKeyIndexerTilingData* __restrict tiling,
                                TPipe* tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    using SCORE_T = typename LIT::scoreType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr PkiLayout LAYOUT_T = LIT::layout;
    static constexpr PkiLayout K_LAYOUT_T = LIT::keyLayout;
    using W_T = typename PkiTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;
    // 量化场景: PkiType 显式携带 weightsType(half/bfloat16_t, entry 按
    // ORIG_DTYPE_WEIGHTS 分发), 优先于 PkiTypeTraits 推导
    using W_QUANT_T = typename LIT::weightsType;
    using W_ACT_T = typename std::conditional<std::is_same<W_QUANT_T, void>::value, W_T, W_QUANT_T>::type;

    PoolKeyIndexerServiceCube<LIT> matmulService;
    PoolKeyIndexerServiceVector<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S1_BASE_SIZE_SMALL = 2;
    // fp8 量化 gSize<=32 时的组深度(mBaseSize=6*gSize, >128 经 splitM 双 pass 消化);
    // 不取 8: resMm1Buf 等缓冲随 mBaseSize 放大, 256 在 poolSize>1 形态下超出 AIV UB 256KB
    static constexpr uint32_t S1_BASE_SIZE_MERGED_M = 6;
    // merged-M 半区交错装载的半区数: dualDstCtl 将 mBase 对半拆分给双 AIV
    static constexpr uint32_t SPLITM_HALVES = 2;
    // merged-M 启用门上限: gSize 与 sparseCount 超限回退常规分块
    static constexpr uint32_t MERGED_M_GSIZE_MAX = 32;
    static constexpr uint32_t MERGED_M_SPARSE_MAX = 2048;
    // 块级并行度门限: 每核 >= 64 块(按静态 shape 上界估算)
    static constexpr uint32_t MERGED_M_MIN_BLOCKS_PER_CORE = 64;
    // L0A 单 pass 行容量(双 pass 分界与禁用钳制基准)
    static constexpr uint32_t L0A_ROW_CAPACITY = 128;
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;

protected:
    TPipe* pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t qScaleCoreOffset = 0ULL; // 量化: q_descale GM 偏移(mode=0 vector 融合)
    uint64_t kScaleCoreOffset = 0ULL; // 量化: k_descale GM 偏移(mode=0 vector 融合)
    uint64_t indiceOutCoreOffset = 0ULL;
    uint64_t valueOutCoreOffset = 0ULL;
    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_ACT_T> weightsGm;
    // 量化 descale: mode=0(float) 走 vector 融合; mode=1(E8M0 按 2 打包成
    // bf16 寻址) 走 Cube LoadData-Mx。两个模式均以 bf16 视角 SetGlobalBuffer。
    GlobalTensor<bfloat16_t> qScaleGmBf16;
    GlobalTensor<bfloat16_t> kScaleGmBf16;
    GlobalTensor<float> qScaleGmFp32;
    GlobalTensor<float> kScaleGmFp32;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<int64_t> actualSeqLengthsGmQ;
    GlobalTensor<int64_t> actualSeqLengthsGmKv;
    GlobalTensor<int64_t> poolTailKGm_;
    bool hasPoolTailK_ = false;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    PkiCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    PkiCommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const PoolKeyIndexerTilingData* __restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengthsK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t& coreNum, PkiCommon::SplitCoreInfo& info);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    __aicore__ inline bool HasBlockGradient();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, PkiCommon::RunInfo& runInfo);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<int64_t>& actualSeqLengthsGmKv, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size, uint32_t& actS2Size,
                                               uint32_t& actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, PkiCommon::RunInfo& runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start,
                                               bool appendTail = false);
    // S2 跨核切分启用判定((bN2,gS1) 并行单元静态上界 < 核对数)
    __aicore__ inline uint32_t CalcS1ParallelUnitUpperBound();

protected:
    // isLD 核(跨核段段首)暂存的段末块信息——ProcessTopK 消费的
    // 字段(gS1Idx/actS*/poolTailK/输出偏移)在同一 (bN2,gS1) 段内恒定, 段末块值即可
    bool ldPending_ = false;
    PkiCommon::RunInfo ldRunInfo_{};
    uint32_t ldBN2T_ = 0; // 目标段 (bN2, gS1)
    uint32_t ldGS1T_ = 0;
    uint32_t ldS2BlockNum_ = 0; // 目标段 S2 总块数
    uint32_t ldS2EndHead_ = 0;  // 段首核已覆盖的末块号
};

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitTilingData(const PoolKeyIndexerTilingData* __restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.keyStride0 = tilingData->keyStride0;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.attenMaskFlag = (tilingData->maskMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.outputLayout = LAYOUT_T;  // 输出和输入形状一致
    constInfo.INVALID_VAL = 0xFF800000; // float -inf
    if constexpr (LAYOUT_T == PkiLayout::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if constexpr (K_LAYOUT_T == PkiLayout::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    if (constInfo.sparseCount > 2048) {
        constInfo.mBaseSize = S1_BASE_SIZE_SMALL * constInfo.gSize;
        constInfo.s1BaseSize = S1_BASE_SIZE_SMALL;
    } else {
        constInfo.mBaseSize = S1_BASE_SIZE * constInfo.gSize;
        constInfo.s1BaseSize = S1_BASE_SIZE;
    }
    // fp8 量化 gSize<=32 时组深度提升为 6: gS1 组数减少, 每 (bN2,s2) 的 K 搬运次数
    // 随组数下降; mBaseSize>128 经 splitM 双 pass 消化, gSize<32 走既有单块路径。
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.returnValue = tilingData->returnValue;
    constInfo.splitMFlag = (constInfo.gSize == 64 && constInfo.sparseCount <= 2048);
    if constexpr (LIT::isQuant) {
        // 量化路径默认禁用 splitM: splitM 的 LoadData2D dst 布局按 fp16 M 切分推导,
        // fp8 的 L1 NZ 行深变化后布局错位, 量化场景走常规单块路径
        constInfo.splitMFlag = false;
        bool mergedM = false;
        // 仅 mode 0(per-token)启用: mode 1(mxFP8) 的 splitM 交错装载与 MX scale
        // 取址布局未闭环, mxFP8 维持 s1BaseSize=4 的 legacy 路径
        if constexpr (LIT::isFp8PerToken) {
            // 启用门: 组数减少会放大小并行度场景的核间负载不均衡, 块级并行度充足
            // (总块数 >= 64*核数, 按静态 shape 上界估算)才启用
            uint64_t estBlocks = CeilDiv((uint64_t)tilingData->s1Size, (uint64_t)S1_BASE_SIZE_MERGED_M) *
                                 constInfo.batchSize * CeilDiv((uint64_t)constInfo.kSeqSize, (uint64_t)S2_BASE_SIZE);
            uint64_t strip = S1_BASE_SIZE_MERGED_M * constInfo.gSize / SPLITM_HALVES;
            if (constInfo.gSize <= MERGED_M_GSIZE_MAX && constInfo.sparseCount <= MERGED_M_SPARSE_MAX &&
                estBlocks >= MERGED_M_MIN_BLOCKS_PER_CORE * usedCoreNum && strip % BLOCK_CUBE == 0) {
                constInfo.s1BaseSize = S1_BASE_SIZE_MERGED_M;
                constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
                mergedM = true;
            }
        }
        // fp8 的 splitM 装载布局已按 dtype 泛化(见 LoadQueryToL0a), 仅 mBaseSize>128
        // 时启用双 pass; <=128 保持单块路径不变
        if (mergedM && constInfo.mBaseSize > L0A_ROW_CAPACITY) {
            constInfo.splitMFlag = true;
        } else if (constInfo.mBaseSize > L0A_ROW_CAPACITY) {
            // 禁用后 mBaseSize 必须钳到 L0 容量(128 行), 且能被 gSize 整除,
            // 否则 L0 越界或 Fixpipe 半区行与 Vec 读取行距错位
            uint32_t s1PerL0 = 128 / constInfo.gSize; // L0 128 行可容纳的 s1 行数(整除截断)
            if (s1PerL0 == 0) {
                s1PerL0 = 1; // gSize>128 时兜底(N1<=64, 正常不触达)
            }
            // 半区(mBaseSize/2)必须含整数个 s1 行(dualDstCtl 按半区拆分给
            // 两个 AIV, Vec 按 gSize 行距读取), s1PerL0 取偶
            if (s1PerL0 % 2 == 1) {
                s1PerL0 -= 1;
                if (s1PerL0 == 0) {
                    s1PerL0 = 1; // gSize>64 时只能 1 行(半区整行归属, mBaseSize=gSize<=64 无此分支)
                }
            }
            constInfo.mBaseSize = s1PerL0 * constInfo.gSize;
            constInfo.s1BaseSize = s1PerL0;
        }
    }
    constInfo.maskMode = tilingData->maskMode;
    constInfo.quantMode = tilingData->quantMode;
    constInfo.topk = tilingData->topk;
    constInfo.poolSize = tilingData->poolSize;
    constInfo.trunkLen = tilingData->trunkLen;
    constInfo.mBaseSizeMax = tilingData->mBaseSizeMax;
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.wsOffScore = tilingData->wsOffScore;
    constInfo.wsOffLdScore = tilingData->wsOffLdScore;
    constInfo.wsOffLdIdx = tilingData->wsOffLdIdx;
    constInfo.qkScale = tilingData->qkScale;
    constInfo.sparseIndicesMode = tilingData->sparseIndicesMode;
    constInfo.isSparseCountOver2K = (constInfo.sparseCount > 2048);
    // S2 跨核切分启用(批量/S1 维并行不足时放开 S2 跨核, 激活
    // SplitCore 既有 isLDOpen 门控, 段搬运归并见 LD 归并流程)。
    // 并行单元为 (bN2∈B×kHeadNum, gS1), 判定补 kHeadNum 因子与启用条件
    // 公式一致(当前 K_HEAD_NUM=1 因子恒 1 不改变取值; 防 kHeadNum>1 时 gate
    // 系统性过度开启)。isSparseCountOver2K 时 s2Loop 恒 1 无切分对象, 排除
    // fp8 per-token(quant_mode=0) 模板: 长上下文 prefill(并行单元≥核对数)亦开 S2 跨核切分
    // (u16 键后半宽 ldTail 归并代价减半, 收益为正); 其余模式维持并行单元不足门控
    constexpr bool forceLd = LIT::isFp8PerToken;
    constInfo.isLDOpen = (forceLd || CalcS1ParallelUnitUpperBound() * constInfo.kHeadNum < usedCoreNum) &&
                         !constInfo.isSparseCountOver2K;
}

// (bN2, gS1) 并行单元数静态上界。BSND 的 actual_seq_q 被 host 校验为
// 必 null -> actS1 == s1Size, 静态精确; TND actS1 运行期未知, 取上界
// CeilDiv(T1, s1BaseSize) + (B-1)(逐 batch 进位, 不低估 -> gate 保守关闭)
template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::CalcS1ParallelUnitUpperBound()
{
    if constexpr (LAYOUT_T == PkiLayout::TND) {
        return CeilDiv(constInfo.qSeqSize, constInfo.s1BaseSize) + (constInfo.batchSize - 1);
    } else {
        return constInfo.batchSize * CeilDiv(constInfo.qSeqSize, constInfo.s1BaseSize);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ,
                                                                   __gm__ uint8_t* actualSeqLengthsK)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengthsK == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGmKv.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsK, constInfo.actualLenDims);
    }
}

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                                      bool isAccumSeq,
                                                                      GlobalTensor<int64_t>& actualSeqLengthsGmKv,
                                                                      uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return static_cast<uint32_t>(actualSeqLengthsGmKv.GetValue(bIdx) - actualSeqLengthsGmKv.GetValue(bIdx - 1));
    } else {
        return static_cast<uint32_t>(actualSeqLengthsGmKv.GetValue(bIdx));
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size,
                                                                      uint32_t& actS2Size, uint32_t& actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    uint32_t actS2Pool = GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGmKv,
                                         constInfo.kSeqSize);
    uint32_t tailK = hasPoolTailK_ ? static_cast<uint32_t>(poolTailKGm_.GetValue(bIdx)) : 0;
    actS2SizeOrig = actS2Pool * constInfo.poolSize + tailK;
    actS2Size = actS2SizeOrig / constInfo.poolSize;
}

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                              uint32_t actS2SizeOrig)
{
    if (actS2SizeOrig / constInfo.poolSize == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len =
        (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) /
        static_cast<int32_t>(constInfo.poolSize);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / static_cast<int32_t>(constInfo.poolSize));
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

// E7: 仅当 mask 场景下各组 valid S2 块数存在梯度(首组 != 末组)时, 加权分核才有
// 收益; 均匀场景(非 mask / 块数相同)回退原始按组计数, 避免纯记账开销
template <typename LIT>
__aicore__ inline bool PoolKeyIndexerKernel<LIT>::HasBlockGradient()
{
    if (!constInfo.attenMaskFlag || constInfo.isLDOpen) {
        return false;
    }
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        uint32_t s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (s1GBaseNum > 1 && actS2Size > 0 &&
            GetS2BaseBlockNumOnMask(0, actS1Size, actS2SizeOrig) !=
                GetS2BaseBlockNumOnMask(s1GBaseNum - 1, actS1Size, actS2SizeOrig)) {
            return true;
        }
    }
    return false;
}

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    const bool weighted = HasBlockGradient(); // E7: 梯度检测提升到循环外(仅一次)
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = constInfo.isLDOpen ? CeilDiv(actS2Size, constInfo.s2BaseSize) : (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            // E7: 计数单位从"1组"改为"组内 valid S2 128块数"(仅梯度场景启用; isLDOpen=false
            // 时组仍原子归一核), 消除 mask_mode=3 下晚 s1 组工作量递增造成的长尾
            s2BaseNum = (constInfo.isLDOpen || weighted) ?
                            (actS2Size > 0 ? GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig) : 0U) :
                            (actS2Size > 0 ? 1U : 0U);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
template <typename LIT>
__aicore__ void inline PoolKeyIndexerKernel<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t& coreNum,
                                                            PkiCommon::SplitCoreInfo& info)
{
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    const bool weighted = HasBlockGradient(); // E7: 与 GetTotalBaseBlockNum 同判, 仅算一次
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    if (totalBlockNum == 0U) {
        // 全批次 0 池(纯尾块, pool_tail_k>0): 无块可枚举, 仅核 0 使能以整批区间走
        // DealActSeqLenIsZero(appendTail) 收尾; 其余核 isCoreEnable=false 直接退出
        coreNum = 1U;
        deal1MoreBlockCoreNum = 1U;
    }
    // E7: 加权记账下边界可少量超配; 末核不设上限, 吃掉全部剩余组(防止尾部组
    // 落入幻影核导致输出缺失)
    uint32_t coreDealBlockCnt = (coreIdx >= coreNum - 1) ?
                                    totalBlockNum :
                                    (coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore);
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;
    if (curCoreIdx < coreNum) {
        splitCoreInfo.isCoreEnable = true;
    } else {
        splitCoreInfo.isCoreEnable = false;
        return;
    }

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    // E7: 记录最后一个非空组坐标, 供末核在边界永不触发时(吃满剩余)补全终点
    uint32_t lastBn2 = 0, lastGS1 = 0, lastS2End = 0;
    bool hasLastGroup = false;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == PkiLayout::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2SizeOrig);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            s2Loop = constInfo.isLDOpen ? s2BaseNum : (actS2Size > 0 ? 1 : 0);
            if (s2Loop > 0) {
                lastBn2 = bN2Idx;
                lastGS1 = gS1Idx;
                lastS2End = constInfo.isLDOpen ? s2BaseNum - 1 : s2BaseNum - 1;
                hasLastGroup = true;
            }
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                // E7: 组原子归一核(isLDOpen=false 时 s2Loop=1), 梯度场景分核记账按
                // 组内 valid S2 块数(s2BaseNum)计权, 与 GetTotalBaseBlockNum 一致
                uint32_t s2RemainBaseNum = (constInfo.isLDOpen || weighted) ?
                                               (constInfo.isLDOpen ? (s2Loop - s2Idx) : s2BaseNum) :
                                               (s2Loop - s2Idx);
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End =
                        constInfo.isLDOpen ? s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1 : s2BaseNum - 1;

                    if (coreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize - 1) {
                            info.bN2End = constInfo.batchSize - 1;
                            info.s2End = 0;
                            info.gS1End = 0;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = (coreIdx >= coreNum - 1) ?
                                           totalBlockNum :
                                           (coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore);
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    break;
                }
            }
        }
    }
    // E7: 末核吃满剩余但边界从未触发 -> 终点补全为最后一个非空组。
    // 加权记账缺陷修复: 组原子归核使边界触发可超配(crossing 组整权重计入当前核),
    // 尾部剩余组权重不足以触发中间核边界时, 该核 start 已记录但 end 保持结构体
    // 初始 0 -> 区间倒置(bN2Start > bN2End)零迭代, 尾部组(末 batch)无人处理输出
    // (缺陷场景复现: B=53/S1=8/513 池/28 核下 batch52 整批丢失; B=3/S1=223/696 池/
    // topk=8192 下末 batch 尾部 6 组丢失)。终点补全放宽到任意"已找到 start 且
    // 边界未触发"的核: 均匀/LD 记账无超配(块粒度累计恰触配额), 中间核边界必
    // 触发, 该分支不改变其行为; 仅加权场景由此核兜底全部剩余组(等价末核语义)。
    if (totalBlockNum == 0U && splitCoreInfo.isCoreEnable) {
        // 0 池兜底: 区间覆盖全部 (bN2) 组, 由 ProcessMain 的 curActSeqLenIsZero
        // 路径逐批清理并追加尾 token
        splitCoreInfo.bN2Start = 0U;
        splitCoreInfo.gS1Start = 0U;
        splitCoreInfo.s2Start = 0U;
        splitCoreInfo.bN2End = constInfo.batchSize * constInfo.kHeadNum - 1U;
        splitCoreInfo.gS1End = 0U;
        splitCoreInfo.s2End = 0U;
        splitCoreInfo.isLD = false;
    }
    if (hasLastGroup && !findLastCoreEnd && coreIdx == curCoreIdx) {
        info.bN2End = lastBn2;
        info.gS1End = lastGS1;
        info.s2End = lastS2End;
        // 与末核钳制同语义: 尾部空 batch(S2=0)不在扫描循环内, 需延伸区间至
        // 末 batch 使 ProcessMain 经 curActSeqLenIsZero 走 DealActSeqLenIsZero
        // 清理输出; 原 bN2End 批次降为中间批, gS1End/s2End 不再绑定其组界,
        // 中间批全量处理语义不变
        if (info.bN2End != constInfo.batchSize - 1) {
            info.bN2End = constInfo.batchSize - 1;
            info.gS1End = 0;
            info.s2End = 0;
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start,
                                                                      bool appendTail)
{
    // 0 池批次的 l_orig 即尾 token 数; 可见性与 ExpandAndAppendIndices 同一公式
    const int32_t poolTailK = (appendTail && hasPoolTailK_) ? static_cast<int32_t>(poolTailKGm_.GetValue(bIdx)) : 0;
    const int32_t lOrig = static_cast<int32_t>(tempLoopInfo.actS2SizeOrig);
    const uint32_t tailPos = (constInfo.sparseIndicesMode == 1U) ? 0U : constInfo.sparseCount * constInfo.poolSize;
    if ASCEND_IS_AIV {
        // poolSize>1 时 indices 输出行宽为 outputLen(见 CalcRunInfo 中 idxOutStride 说明)
        uint32_t idxOutStride = (constInfo.poolSize > 1) ?
                                    (constInfo.sparseCount * constInfo.poolSize + constInfo.poolSize - 1) :
                                    constInfo.sparseCount;
        if (constInfo.outputLayout == PkiLayout::TND) {
            uint32_t tSize = static_cast<uint32_t>(actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1));
            uint32_t tBase = bIdx == 0 ? 0 : static_cast<uint32_t>(actualSeqLengthsGmQ.GetValue(bIdx - 1));
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                // S1 规模 x 步长乘法提升 64 位: qSeqSize*idxOutStride 可超 2^32(uint32 回绕)
                uint64_t indiceOutOffset = static_cast<uint64_t>(tBase + s1Idx) * constInfo.kHeadNum * idxOutStride +
                                           n2Idx * idxOutStride; // T轴、s1轴偏移
                int32_t visible = 0;
                if (poolTailK > 0) {
                    if (constInfo.maskMode == 0) {
                        visible = poolTailK;
                    } else {
                        int32_t globalPosQ = lOrig - static_cast<int32_t>(s1Count) + static_cast<int32_t>(s1Idx);
                        visible = PkiCommon::Max(0, PkiCommon::Min(poolTailK, globalPosQ - (lOrig - poolTailK) + 1));
                    }
                }
                vectorService.CleanInvalidOutput(indiceOutOffset, tailPos, visible, lOrig - poolTailK);
            }
        } else if (constInfo.outputLayout == PkiLayout::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset =
                    static_cast<uint64_t>(bIdx) * constInfo.qSeqSize * constInfo.kHeadNum * idxOutStride +
                    static_cast<uint64_t>(s1Idx) * constInfo.kHeadNum * idxOutStride +
                    n2Idx * idxOutStride; // B轴、S1轴偏移
                int32_t visible = 0;
                if (poolTailK > 0) {
                    if (constInfo.maskMode == 0) {
                        visible = poolTailK;
                    } else {
                        int32_t globalPosQ =
                            lOrig - static_cast<int32_t>(constInfo.qSeqSize) + static_cast<int32_t>(s1Idx);
                        visible = PkiCommon::Max(0, PkiCommon::Min(poolTailK, globalPosQ - (lOrig - poolTailK) + 1));
                    }
                }
                vectorService.CleanInvalidOutput(indiceOutOffset, tailPos, visible, lOrig - poolTailK);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* poolKey, __gm__ uint8_t* weights, __gm__ uint8_t* poolTailK,
    __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengthsK, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* qDescale, __gm__ uint8_t* kDescale, __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues,
    __gm__ uint8_t* workspace, const PoolKeyIndexerTilingData* __restrict tiling, TPipe* tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);
    if (poolTailK != nullptr) {
        poolTailKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(poolTailK), constInfo.batchSize);
        hasPoolTailK_ = true;
    }

    // 获取分核信息
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;

    uint64_t offset = 0;
    // vec 把整个s2的score存储在GM，大小为s1BaseSize * 16K * 4
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    uint64_t singleCoreScoreSize = constInfo.s1BaseSize *
                                   PkiCommon::Align((uint64_t)constInfo.kSeqSize, (uint64_t)constInfo.s2BaseSize) *
                                   sizeof(SCORE_T);
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T*)(workspace + aiCoreIdx * singleCoreScoreSize));
    offset += GetBlockNum() * singleCoreScoreSize;
    // LD-fix: 跨核尾段暂存区紧随 score 区(host CalcWorkspaceSize 同步预留),
    // 布局与 score 区一致(每核 singleCoreScoreSize, AIV 半区按 halfBase)
    GlobalTensor<SCORE_T> ldTailGm;
    ldTailGm.SetGlobalBuffer((__gm__ SCORE_T*)(workspace + offset));

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        valueOutGm.SetGlobalBuffer((__gm__ float*)sparseValues);
        weightsGm.SetGlobalBuffer((__gm__ W_ACT_T*)weights);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        if constexpr (LIT::isFp8PerToken) {
            // mode=0: q/k_descale 为 float, vector 侧融合
            qScaleGmFp32.SetGlobalBuffer((__gm__ float*)qDescale);
            kScaleGmFp32.SetGlobalBuffer((__gm__ float*)kDescale);
            vectorService.InitVecInputTensor(weightsGm, indiceOutGm, valueOutGm, blockTableGm, qScaleGmFp32,
                                             kScaleGmFp32);
        } else if constexpr (LIT::isInt8) {
            // int8: q/k_descale 为 FP16(half 视图), vector 侧融合(half 重载)
            GlobalTensor<half> qScaleGmHalf;
            GlobalTensor<half> kScaleGmHalf;
            qScaleGmHalf.SetGlobalBuffer((__gm__ half*)qDescale);
            kScaleGmHalf.SetGlobalBuffer((__gm__ half*)kDescale);
            vectorService.InitVecInputTensor(weightsGm, indiceOutGm, valueOutGm, blockTableGm, qScaleGmHalf,
                                             kScaleGmHalf);
        } else {
            vectorService.InitVecInputTensor(weightsGm, indiceOutGm, valueOutGm, blockTableGm);
        }
        vectorService.InitVecWorkspaceTensor(scoreGm, ldTailGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T*)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T*)poolKey);
        if constexpr (LIT::isMxFp8) {
            // mode=1: E8M0 scale 按 2 个打包成 1 个 bf16 寻址, 块数 = e8m0 元素数 / 2
            qScaleGmBf16.SetGlobalBuffer((__gm__ bfloat16_t*)qDescale);
            kScaleGmBf16.SetGlobalBuffer((__gm__ bfloat16_t*)kDescale);
            matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, kScaleGmBf16, qScaleGmBf16);
        } else {
            matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
        }
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (LAYOUT_T == PkiLayout::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                              PkiCommon::RunInfo& runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.s2Start = splitCoreInfo.s2Start;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;

    if (!runInfo.isValid) {
        return;
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    runInfo.poolTailK = hasPoolTailK_ ? static_cast<int32_t>(poolTailKGm_.GetValue(tempLoopInfo.bIdx)) : 0;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        PkiCommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, PkiCommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (LAYOUT_T == PkiLayout::TND) {
            actualSeqQPrefixSum =
                (runInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1));
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // 量化 q_descale 偏移: mode=0 与 weights 同构直接复用其公式;
        // mode=2(int8) scale shape 与 mode=0 同构(去 D 轴逐维相等)公式同用;
        // mode=1 为 (B,S1,N1,D/64,2), 每 s1 行 D/32 个 e8m0
        if constexpr (LIT::isFp8PerToken || LIT::isInt8) {
            qScaleCoreOffset = weightsCoreOffset;
        } else if constexpr (LIT::isMxFp8) {
            qScaleCoreOffset =
                actualSeqQPrefixSum * constInfo.qHeadNum * (constInfo.headDim / PKI_MX_SCALE_GROUP_SIZE) +
                runInfo.gS1Idx * constInfo.mBaseSize * (constInfo.headDim / PKI_MX_SCALE_GROUP_SIZE) +
                runInfo.n2Idx * constInfo.gSize * (constInfo.headDim / PKI_MX_SCALE_GROUP_SIZE);
        }
        // B,S1,N2,k/T,N2,k; poolSize>1 时 indices 行宽为 outputLen(≠sparseCount),
        // 否则 batch>0 行基址错位, 写入覆盖相邻行
        uint32_t idxOutStride = (constInfo.poolSize > 1) ?
                                    (constInfo.sparseCount * constInfo.poolSize + constInfo.poolSize - 1) :
                                    constInfo.sparseCount;
        indiceOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * idxOutStride + runInfo.n2Idx * idxOutStride;
        // values 输出行宽恒为 sparseCount(与 indices 的 outputLen 不同),
        // 不可复用 indiceOutCoreOffset, 否则错位写到相邻 batch
        valueOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == PkiLayout::TND) { // T N2 D
        actualSeqKPrefixSum =
            (runInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(actualSeqLengthsGmKv.GetValue(runInfo.bIdx - 1));
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    // 量化 k_descale 偏移: mode=0 每池 1 个 float(偏移=池前缀+s2Idx*s2BaseSize);
    // mode=2(int8) 每池 1 个 half 同式; mode=1 每池 D/32 个 e8m0
    if constexpr (LIT::isFp8PerToken || LIT::isInt8) {
        kScaleCoreOffset =
            actualSeqKPrefixSum * constInfo.kHeadNum + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum;
    } else if constexpr (LIT::isMxFp8) {
        uint64_t kScaleS2Offset = actualSeqKPrefixSum + static_cast<uint64_t>(runInfo.s2Idx) * constInfo.s2BaseSize;
        kScaleCoreOffset = kScaleS2Offset * constInfo.kHeadNum * (constInfo.headDim / PKI_MX_SCALE_GROUP_SIZE);
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.tensorQScaleOffset = qScaleCoreOffset;
    runInfo.tensorKeyScaleOffset = kScaleCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.valueOutOffset = valueOutCoreOffset;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::Process()
{
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();
    ProcessDecode();
    // LD-fix: AIV 的 FreeEventID 必须在 ProcessDecode 之后——LD 段首核的延迟
    // ProcessTopK 入口 WaitFlag 依赖 TOPK 信号量(AllocEventID 预置或上次调用
    // 结尾 re-arm), 若在 ProcessMain 尾部先 Free 会把信号量消费掉, 延迟 TopK
    // 永久等待导致内核挂死(isLDOpen 且存在跨核段时必现)
    if ASCEND_IS_AIV {
        if (splitCoreInfo.isCoreEnable) {
            vectorService.FreeEventID();
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        // poolSize>1 时 indices 输出行宽为 outputLen(见 CalcRunInfo 中 idxOutStride 说明)
        uint32_t idxOutStride = (constInfo.poolSize > 1) ?
                                    (constInfo.sparseCount * constInfo.poolSize + constInfo.poolSize - 1) :
                                    constInfo.sparseCount;
        uint64_t totalOutputSize = constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * idxOutStride;
        uint64_t singleCoreSize =
            PkiCommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
        if (constInfo.returnValue) {
            // values 总大小与 indices 不同(行宽 sparseCount vs outputLen),
            // 需按自身总大小独立切分清理
            uint64_t totalValueSize =
                constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
            uint64_t singleCoreValueSize =
                PkiCommon::Align((totalValueSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(uint32_t));
            uint64_t valueBase = tmpBlockIdx * singleCoreValueSize;
            if (valueBase < totalValueSize) {
                uint64_t valueDealSize = (valueBase + singleCoreValueSize <= totalValueSize) ?
                                             singleCoreValueSize :
                                             totalValueSize - valueBase;
                event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

                GlobalTensor<uint32_t> valueOutGmTmp;
                valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t*)valueOutGm.GetPhyAddr());
                GlobalTensor<uint32_t> valueOut = valueOutGmTmp[valueBase];
                AscendC::InitGlobalMemory(valueOut, valueDealSize, constInfo.INVALID_VAL);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessMain()
{
    if (!splitCoreInfo.isCoreEnable) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    PkiCommon::RunInfo runInfo;
    uint32_t gloop = 0;
    // LD-fix: 记录区间起始块号(s2Start 在首组后被清零), 首组处理完立即转存
    // 跨核尾段至 ldTail 暂存区(后续组写入会按同列布局覆盖常规 score 区)
    const uint32_t rangeS2Start = splitCoreInfo.s2Start;
    bool firstGroupDone = false;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U, true);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            if (!firstGroupDone) {
                firstGroupDone = true;
                // LD-fix: runInfo 由 ProcessBaseBlock 按引用回填(传值会留未初始化垃圾,
                // SaveLdTail 依其 gS1Idx/actS1Size 计算行数)
                if (constInfo.isLDOpen && rangeS2Start > 0 && rangeS2Start <= tempLoopInfo.s2LoopEnd) {
                    vectorService.SaveLdTail(rangeS2Start, static_cast<uint32_t>(tempLoopInfo.s2LoopEnd), runInfo);
                }
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    // AIV 的 FreeEventID 延后到 ProcessDecode() 之后(见 Process 尾部注释);
    // AIC 无延迟阶段, 尾部跨核等待后立即释放
    if ASCEND_IS_AIC {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_FIX>(PkiCommon::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreWaitFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_FIX>(PkiCommon::ConstInfo::CROSS_VC_EVENT + 1);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                                   PkiCommon::RunInfo& runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
        vectorService.ProcessVec1(runInfo);
        if (runInfo.isLastS2InnerLoop) { // 本核s2last
            if (!constInfo.isLDOpen) {
                vectorService.ProcessTopK(runInfo);
                return;
            }
            // 本核该 (bN2,gS1) 段覆盖整段 S2([0, s2BlockNum-1])才
            // 内联 TopK; 段首且未完段 -> 暂存待 LD 阶段(段搬运后补算);
            // 跨核中段/段尾 -> 跳过(段首核代算)
            uint32_t s2BlockNum =
                constInfo.attenMaskFlag ?
                    GetS2BaseBlockNumOnMask(tempLoopInfo.gS1Idx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig) :
                    (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
            bool segCoverAllSpan = (runInfo.s2Start == 0) && (tempLoopInfo.s2LoopEnd + 1 == s2BlockNum);
            if (segCoverAllSpan) {
                vectorService.ProcessTopK(runInfo);
            } else if (runInfo.s2Start == 0) {
                ldPending_ = true;
                ldRunInfo_ = runInfo;
                ldBN2T_ = tempLoopInfo.bN2Idx;
                ldGS1T_ = tempLoopInfo.gS1Idx;
                ldS2BlockNum_ = s2BlockNum;
                ldS2EndHead_ = tempLoopInfo.s2LoopEnd;
            }
        }
    }
}

// LD 段搬运归并(AIV)——SyncAll 后 isLD 核枚举段内后续核对目标
// (bN2T, gS1T) 的覆盖块区间, 把各 follower 写在自己 scoreGm 区的 score 列段
// 搬回本核区同列, 再以暂存 RunInfo 复用既有 ProcessTopK(输入字节与整段单核
// 计算一致 -> 输出逐位一致)。AIC 仅参与全局栅障后退出(scoreGm 仅 AIV 写)。
template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessDecode()
{
    // 检视 P1-1: isLDOpen 由 tiling 输入在全核一致推导(AIC/AIV 同 tiling), 非 LD
    // 场景全体核一致跳过全设备栅障——既有非量化/FP8 路径零新增同步开销(第 8 节前
    // 言"非启用场景逐字不变"); isLDOpen 时全核仍统一参与, 不破坏 SyncAll 全参与
    // 语义。isLD 只在 isLDOpen 下可能置位(SplitCore s2Loop 门控), ldPending_ 仅
    // isLDOpen 路径设置, 外层 gate 与执行体判定一致
    if (!constInfo.isLDOpen) {
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    SyncAll();
    if ASCEND_IS_AIV {
        if (!splitCoreInfo.isLD || !ldPending_) {
            return;
        }
        uint32_t nextS2 = ldS2EndHead_ + 1;
        for (uint32_t coreIdx = aiCoreIdx + 1; coreIdx < usedCoreNum && nextS2 < ldS2BlockNum_; coreIdx++) {
            // 重放 SplitCore 取 follower 核分核区间(确定性: 同 tiling 输入同结果);
            // coreNum 传已收缩的 usedCoreNum 副本, minBlockPerCore>=1 不再收缩
            uint32_t dummyCoreNum = usedCoreNum;
            PkiCommon::SplitCoreInfo followerInfo{};
            SplitCore(coreIdx, dummyCoreNum, followerInfo);
            // follower 区间起点越过目标段 -> 段已结束(前序 follower 已覆盖到段尾)
            if (followerInfo.bN2Start != ldBN2T_ || followerInfo.gS1Start != ldGS1T_) {
                break;
            }
            // follower 对目标段的覆盖: 起点为其 s2Start(必为段内续算起点);
            // 终点 = 其区间末段即目标段 ? s2End : 整段(目标段为其非末段, 必被完整消费)
            uint32_t segEnd = (followerInfo.bN2End == ldBN2T_ && followerInfo.gS1End == ldGS1T_) ? followerInfo.s2End :
                                                                                                   (ldS2BlockNum_ - 1);
            vectorService.CopyScoreSegFromCore(coreIdx, followerInfo.s2Start, segEnd, ldRunInfo_);
            nextS2 = segEnd + 1;
        }
        vectorService.ProcessTopK(ldRunInfo_);
    }
}

} // namespace PkiKernel
#endif // POOL_KEY_INDEXER_KERNEL_ARCH35_H
