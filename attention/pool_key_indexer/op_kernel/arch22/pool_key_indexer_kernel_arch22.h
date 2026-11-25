/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_KERNEL_ARCH22_H
#define POOL_KEY_INDEXER_KERNEL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"
#include "pool_key_indexer_service_vector_arch22.h"
#include "pool_key_indexer_service_cube_arch22.h"

namespace PkiKernel {
using namespace PkiCommon;
using namespace PkiServiceVec;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// arch22 ProcessDecode 的 isLDOpen 调用门开关。1=!isLDOpen 时整个 ProcessDecode 直接 return,
// prefill 不再白付 InitLDBuffers(pipe->Reset()+LD 缓冲重分配) + ICachePreLoad + 一次
// 40 AIV 全设备 SyncAll(); 0=无条件执行(基线路径)。
// 数值零影响: !isLDOpen ⇒ splitCoreInfo.isLD 恒 false(SplitCore 的 S2 跨核规避把
// coreDealBlockCnt 抬到整组 ⇒ s2End=s2BaseNum-1; over2K 分支同样直接给 s2End=s2BaseNum-1)
// ⇒ ProcessLD 本就是死代码; InitLDBuffers 只重分配 LD 专属缓冲、不写任何输出张量;
// ICachePreLoad 是纯 icache 提示。isLDOpen 由 constInfo(全核同 tiling)推导 ⇒ 全核一致
// 跳过或一致执行, 不会半参与导致栅栏挂死。
// 注意: 只包 InitLDBuffers 的调用点, 其函数体不动 —— 内部 ExpandSlotDrainAll →
// pipe->Reset() → ExpandSlotAcquire → SetWaitFlag<MTE3_S> → VToSSync 的次序是精度约束,
// 改次序会引入 MTE3 死锁/aicore 超时(507014)。
#ifndef PKI_DECODE_LD_GATE
#define PKI_DECODE_LD_GATE 1
#endif
// Init 期参数常驻开关: pool_tail_k / actual_seq_q / actual_seq_k 的每块 GM 标量读改
// Init 期常驻。1=在 Init 期(SplitCore 之前)把三张量逐 batch 的原始 GM 值缓存到 kernel
// 成员定长数组, 热路径只读缓存; 0=热路径逐次 GM 标量读(基线路径)。
// 数值 bit-exact: 同一 GM 同一元素, 且这些张量在 kernel 执行期只读 ⇒ 缓存值与逐次读回
// 的值逐位相同; TND 的前缀和差分仍由原函数 GetActualSeqLen/原读点原式完成, 本开关只替换
// "值的来源", 不做任何预差分(混用"缓存差分值"与"缓存前缀和"会 off-by-one)。
// 载体 = device 标量侧 local memory(类成员数组), 不占 UB(int8 UB 余量仅 7584B);
// batchSize 超 PKI_PARAM_RESIDENT_MAX_B 时 paramResidentOn_=false, 回退逐次 GM 读。
// 全核一致性: 缓存建于 SplitCore 之前, 各核读同一 GM ⇒ 缓存逐值相同。
#ifndef PKI_PARAM_RESIDENT
#define PKI_PARAM_RESIDENT 1
#endif
#ifndef PKI_PARAM_RESIDENT_MAX_B
#define PKI_PARAM_RESIDENT_MAX_B 64
#endif
// ReadSeqRaw 的张量标识(只用于选中对应的常驻数组, 不参与任何算术)
constexpr uint32_t PKI_SEQ_Q = 0U;
constexpr uint32_t PKI_SEQ_K = 1U;

// pack-loss LD gate(非量化 + int8): !isLDOpen 且 !over2K 时, 组原子装箱损失
// lossInd = ⌈G/C⌉·Nmax/(T/C) − 1 ≥ τ(10%) 则放开 S2 跨核切分。
#ifndef PKI_PACK_LOSS_LD_GATE
#define PKI_PACK_LOSS_LD_GATE 1
#endif
// 覆盖域(存储 quantMode): 非量化(0)与 int8(3), int8 不限定 compact;
// FP8 家族(1/2) arch22 tiling 层拒绝不可达, 显式排除。
constexpr int32_t PKI_QUANT_MODE_NONE_STORED = 0;
// τ=10% 的定点比较系数: lossInd≥τ ⇔ LOSS_NUM·⌈G/C⌉·Nmax·C ≥ LOSS_DEN·T
#define PKI_LD_GATE_LOSS_NUM 10ULL
#define PKI_LD_GATE_LOSS_DEN 11ULL
// quantMode 的存储口径: host -1/0/1/2 → tiling 存值 +1(SetTilingData), int8(用户值 2)⇒ 3
#define PKI_QUANT_MODE_INT8_STORED 3

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;  // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;   // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL; // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actS2SizeOrig = 0ULL;
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
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* blockTable, __gm__ uint8_t* qDescale,
                                __gm__ uint8_t* kDescale, __gm__ uint8_t* sparse_indices, __gm__ uint8_t* sparse_values,
                                __gm__ uint8_t* workspace, const PoolKeyIndexerTilingData* __restrict tiling,
                                TPipe* tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr PkiLayout LAYOUT_T = LIT::layout;
    static constexpr PkiLayout K_LAYOUT_T = LIT::keyLayout;
    // 编译期条件选择模板第二个参数的类型，直接声明W_T
    // 第一个模板参数：固定为Q_T；第二个模板参数：编译期选float/void
    using W_DERIVED_T =
        typename PkiTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;
    // 量化场景: PkiType 显式携带 weightsType(half/bfloat16_t), 优先于推导
    using W_EXPLICIT_T = typename LIT::weightsType;
    using W_T = typename std::conditional<std::is_same<W_EXPLICIT_T, void>::value, W_DERIVED_T, W_EXPLICIT_T>::type;

    // MM(score)输出 dtype: int8 走 fixpipe DEQF16 直写 GM fp16, 其余 fp32
    using MM1_OUT_T = typename std::conditional<LIT::isInt8, half, float>::type;

    PoolKeyIndexerServiceCube<LIT> matmulService;
    PoolKeyIndexerServiceVector<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t SPARSE_COUNT_8K = 8192;
    static constexpr uint32_t BLOCK_CUBE_SIZE = 16;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

protected:
    TPipe* pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    // int8 量化 scale 基址偏移(q_descale 随 weights 行 / k_descale 池行前缀)
    uint64_t qScaleCoreOffset = 0ULL;
    uint64_t kScaleCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    // values 输出行宽恒为 sparseCount, 与 indices(poolSize>1 时为 outputLen)
    // 不同, 必须单独维护基址偏移, 不可复用 indiceOutCoreOffset
    uint64_t valueOutCoreOffset = 0ULL;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_T> weightsGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<int64_t> actualSeqLengthsGmQ;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> poolTailKGm_;
    bool hasPoolTailK_ = false;
#if PKI_PARAM_RESIDENT
    // Init 期常驻的三张量原始 GM 值(逐 batch, int64 与 GetValue 返回类型一致)。
    // 载体是 device 标量侧 local memory(类成员定长数组), 不进 UB 池、不调用
    // InitBuffer ⇒ UB 各分配门与余量零影响。
    // 只在 paramResidentOn_ 为真(batchSize<=上界 且 TND 前缀和读点可达)时被读,
    // 且 [0, batchSize) 全部由 InitParamResident 先填满 ⇒ 无未初始化读。
    int64_t seqQRaw_[PKI_PARAM_RESIDENT_MAX_B];
    int64_t seqKRaw_[PKI_PARAM_RESIDENT_MAX_B];
    int64_t tailKRaw_[PKI_PARAM_RESIDENT_MAX_B];
    bool paramResidentOn_ = false;
#endif
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S
    GlobalTensor<float> vec1ResGm;     // 存放TopK计算中间结果
    GlobalTensor<int64_t> vec1ParamGm; // 存放LD参数信息

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
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths);
#if PKI_PARAM_RESIDENT
    // Init 期把三张量逐 batch 的原始 GM 值缓存到成员数组(见成员声明处)
    __aicore__ inline void InitParamResident();
#endif
    // 值的来源二选一 —— 常驻数组(PKI_PARAM_RESIDENT 启用且 paramResidentOn_ 时)
    // / gm.GetValue(其余情况)。srcTag 只用于选中数组, 不参与算术;
    // 返回类型与 GetValue 一致(int64_t)
    __aicore__ inline int64_t ReadSeqRaw(uint32_t srcTag, uint32_t idx, GlobalTensor<int64_t>& actualSeqLengthsGm);
    __aicore__ inline int64_t ReadTailKRaw(uint32_t idx);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t& coreNum, PkiCommon::SplitCoreInfo& info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, PkiCommon::RunInfo& runInfo);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<int64_t>& actualSeqLengthsGm, uint32_t defaultSeqLen,
                                               uint32_t srcTag);
    // 逐 batch actS1Size 的最小值 —— sortOutBuf 的 main/cache 两路径运行期互斥判据
    // (判据必须取 min, 见 CalcMinActS1OverBatches 定义处注释)
    __aicore__ inline uint32_t CalcMinActS1OverBatches();
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size, uint32_t& actS2Size,
                                               uint32_t& actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, PkiCommon::RunInfo& runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start,
                                               bool appendTail = false);
    // S2 跨核切分启用判定((bN2,gS1) 并行单元静态上界 < 核对数)
    __aicore__ inline uint32_t CalcS1ParallelUnitUpperBound();
#if PKI_PACK_LOSS_LD_GATE
    // ============================pack-loss LD gate=================
    // unitUB 项关闭且装箱损失 ≥ τ 时放开 S2 跨核切分
    __aicore__ inline void RefineLDGateByPackLoss();
    __aicore__ inline bool CalcPackLossOpen();
#endif
};

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitTilingData(const PoolKeyIndexerTilingData* __restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->maskMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.returnValue = tilingData->returnValue;
    constInfo.maskMode = tilingData->maskMode;
    constInfo.quantMode = tilingData->quantMode;
    constInfo.topk = tilingData->topk;
    constInfo.poolSize = tilingData->poolSize;
    constInfo.trunkLen = tilingData->trunkLen;
    constInfo.mBaseSizeMax = tilingData->mBaseSizeMax;
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.keyStride0 = tilingData->keyStride0;
    constInfo.wsOffScore = tilingData->wsOffScore;
    constInfo.wsOffLdScore = tilingData->wsOffLdScore;
    constInfo.wsOffLdIdx = tilingData->wsOffLdIdx;
    constInfo.qkScale = tilingData->qkScale;
    constInfo.sparseIndicesMode = tilingData->sparseIndicesMode;

    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if (LAYOUT_T == PkiLayout::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == PkiLayout::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;
    // arch22 切分常量以 kernel 侧硬编码为准, 覆盖 host tiling 字段
    // (host 下发字段供 arch35 消费); workspace 公式按 kernel 口径镜像推导
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.isSparseCountOver2K = (constInfo.sparseCount <= BASE_TOPK) ? false : true;

    constInfo.s1BaseSize = constInfo.isSparseCountOver2K ? SPARSE_COUNT_8K / constInfo.sparseCount * 2 : 8;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
    constInfo.mBaseSizeAlign = PkiCommon::Align(constInfo.mBaseSize, BLOCK_CUBE_SIZE);

    // S2 跨核切分启用(批量/S1 维并行不足时放开 S2 跨核, 走 LD 归并)。
    // isSparseCountOver2K 时 s2Loop 恒 1 无切分对象, 排除。
    constInfo.isLDOpen = (CalcS1ParallelUnitUpperBound() < usedCoreNum) && !constInfo.isSparseCountOver2K;
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

#if PKI_PACK_LOSS_LD_GATE
// pack-loss LD gate: unitUB 项关闭时按装箱损失补开; 调用须在 InitParamResident 之后、
// SplitCore 之前(C 为 host 原值), 组统计与 GetTotalBaseBlockNum 同源保证全核一致。
template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::RefineLDGateByPackLoss()
{
    if (constInfo.isLDOpen) {
        return;
    }
    if (constInfo.isSparseCountOver2K) {
        return;
    }
    // dtype 域: 非量化(存储 0)与 int8(存储 3); FP8 家族(存储 1/2)arch22 不可达, 排除
    if (constInfo.quantMode != PKI_QUANT_MODE_NONE_STORED && constInfo.quantMode != PKI_QUANT_MODE_INT8_STORED) {
        return;
    }
    if (CalcPackLossOpen()) {
        constInfo.isLDOpen = true;
    }
}

// 组统计一趟推导: G(有块组数)/Nmax(最大组块数)/T(总块数)与 GetTotalBaseBlockNum
// 同源同式, 保证全核一致。
template <typename LIT>
__aicore__ inline bool PoolKeyIndexerKernel<LIT>::CalcPackLossOpen()
{
    const uint64_t C = static_cast<uint64_t>(usedCoreNum);
    if (C == 0ULL) {
        return false;
    }
    uint64_t G = 0ULL;
    uint64_t T = 0ULL;
    uint64_t Nmax = 0ULL;
    bool hasZeroPoolBatch = false;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        if (actS2Size == 0U) {
            // 存在零池 batch: LD 归并与空 batch 清理分支(DealActSeqLenIsZero)交互
            // 未验证, 条件化开启在此类退化形态下保守关闭
            hasZeroPoolBatch = true;
        }
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            if (s2BaseNum > 0U && s1GBaseNum > 0U) {
                G += static_cast<uint64_t>(s1GBaseNum) * constInfo.kHeadNum;
                T += static_cast<uint64_t>(s1GBaseNum) * static_cast<uint64_t>(s2BaseNum) * constInfo.kHeadNum;
                Nmax = (static_cast<uint64_t>(s2BaseNum) > Nmax) ? static_cast<uint64_t>(s2BaseNum) : Nmax;
            }
        } else {
            for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2SizeOrig);
                if (s2BaseNum > 0U) {
                    G += constInfo.kHeadNum;
                    T += static_cast<uint64_t>(s2BaseNum) * constInfo.kHeadNum;
                    Nmax = (static_cast<uint64_t>(s2BaseNum) > Nmax) ? static_cast<uint64_t>(s2BaseNum) : Nmax;
                }
            }
        }
    }
    if (G == 0ULL || T == 0ULL || hasZeroPoolBatch) {
        return false;
    }
    const uint64_t groupRounds = (G + C - 1ULL) / C; // ⌈G/C⌉
    const uint64_t num = groupRounds * Nmax * C;     // lossInd 的分子(分母 = T)
    return PKI_LD_GATE_LOSS_NUM * num >= PKI_LD_GATE_LOSS_DEN * T;
}
#endif

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        // sortOutBuf main/cache 互斥判据 = 逐 batch actS1 最小值。
        // InitActualSeqLen 已在 Init() 中先于本函数完成(actualSeqLengthsGmQ 可读),
        // 且此处只有标量 GM 读(S pipe)、尚无 MTE2/V 在途 ⇒ 无需额外同步
        // (SplitCore 同样在 Init 期读该张量)。
        vectorService.InitBuffers(pipe, CalcMinActS1OverBatches());
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ,
                                                                   __gm__ uint8_t* actualSeqLengths)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, constInfo.actualLenDims);
    }
}

template <typename LIT>
__aicore__ inline int64_t PoolKeyIndexerKernel<LIT>::ReadSeqRaw(uint32_t srcTag, uint32_t idx,
                                                                GlobalTensor<int64_t>& actualSeqLengthsGm)
{
#if PKI_PARAM_RESIDENT
    if (paramResidentOn_) {
        return (srcTag == PKI_SEQ_Q) ? seqQRaw_[idx] : seqKRaw_[idx];
    }
#endif
    return actualSeqLengthsGm.GetValue(idx);
}

template <typename LIT>
__aicore__ inline int64_t PoolKeyIndexerKernel<LIT>::ReadTailKRaw(uint32_t idx)
{
#if PKI_PARAM_RESIDENT
    if (paramResidentOn_) {
        return tailKRaw_[idx];
    }
#endif
    return poolTailKGm_.GetValue(idx);
}

#if PKI_PARAM_RESIDENT
// Init 期一次性缓存 pool_tail_k / actual_seq_q / actual_seq_k 的原始 GM 值(不做任何
// 差分/换算, 差分仍由 GetActualSeqLen 原式完成)。
// 读取模式与 CalcMinActS1OverBatches 相同: "逐 batch 纯标量 GM 读"循环, 无 MTE2/V 在途
// ⇒ 无需额外同步。
// 调用点必须早于 SplitCore(其经 GetTotalBaseBlockNum/GetS1S2ActualSeqLen 逐 batch 读同
// 一批张量, 且分核结果要求全核一致: 同 GM 同输入 ⇒ 各核缓存逐值相同)。
// 启用条件(三条, 任一不满足即 paramResidentOn_=false ⇒ 回退逐次 GM 读路径):
//   ① batchSize <= PKI_PARAM_RESIDENT_MAX_B(载体是定长成员数组, 不占 UB);
//   ② TND 实例化下两路 actual_seq 均可读 —— CalcRunInfo 的 isFirstS2InnerLoop
//      前缀和读点不带 dims 判空, 若该路 actual_seq 为 null 则原路径读的是无效张量,
//      不能把它替换成缓存的 0(语义差); BSND 实例化无该读点 ⇒ 不受此条限制。
//   ③ 三张量的读点各自的可达性由调用处原有的 hasPoolTailK_/actualLenDims 判据保持
//      不变 ⇒ 缓存里对应位置永不被读, 无需额外条件。
template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::InitParamResident()
{
    bool prefixReadable = true;
    if constexpr (LAYOUT_T == PkiLayout::TND) {
        prefixReadable = (constInfo.actualLenQDims != 0) && (constInfo.actualLenDims != 0);
    }
    paramResidentOn_ = (constInfo.batchSize <= PKI_PARAM_RESIDENT_MAX_B) && prefixReadable;
    if (!paramResidentOn_) {
        return;
    }
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        seqQRaw_[bIdx] = (constInfo.actualLenQDims != 0) ? actualSeqLengthsGmQ.GetValue(bIdx) : 0;
        seqKRaw_[bIdx] = (constInfo.actualLenDims != 0) ? actualSeqLengthsGm.GetValue(bIdx) : 0;
        if (hasPoolTailK_) {
            tailKRaw_[bIdx] = poolTailKGm_.GetValue(bIdx);
        }
    }
}
#endif

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                                      bool isAccumSeq,
                                                                      GlobalTensor<int64_t>& actualSeqLengthsGm,
                                                                      uint32_t defaultSeqLen, uint32_t srcTag)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return static_cast<uint32_t>(ReadSeqRaw(srcTag, bIdx, actualSeqLengthsGm) -
                                     ReadSeqRaw(srcTag, bIdx - 1, actualSeqLengthsGm));
    } else {
        return static_cast<uint32_t>(ReadSeqRaw(srcTag, bIdx, actualSeqLengthsGm));
    }
}

// 逐 batch 求 actS1Size 的最小值, 作为 sortOutBuf 的 main/cache 两路径
// 运行期互斥判据传给 vectorService.InitBuffers。
// 为什么必须是 min: cache 路径的进入条件是 (actS1Size<=4 && !over2K)(ProcessVec 的
// if/else 分岔), 而 actS1Size 逐 batch 取自 actual_seq_q ⇒ 同一次 launch 内不同 batch
// 可分别走 cache/main。取首批或取 max 都会在"混合 batch"下漏判 ⇒ SortedBasicBlock_
// 写越 sortOutBuf, 而 release 态 InitBuffer 越池不报错 ⇒ 静默踩相邻 UB buffer。
// 与 GetS1S2ActualSeqLen 的 actS1Size **同源同式**(同一个 GetActualSeqLen 调用与同样
// 的实参), 故与 ProcessVec 的分岔判据严格一致, 不存在口径漂移。
// 边界: actualLenQDims==0(actual_seq_q 为 null, BSND 恒如此)时 GetActualSeqLen 返回
// qSeqSize ⇒ min 退化为静态值; 任一 batch actS1==0(空 batch)把 min 压到 0 ⇒ 判据自动
// 保守(保留 cache 区), 与该 batch 走 curActSeqLenIsZero 早退路径不冲突; batchSize==0
// 时无任何 ProcessVec, 返回 qSeqSize 亦安全。
template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::CalcMinActS1OverBatches()
{
    uint32_t minActS1 = constInfo.qSeqSize;
    for (uint64_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        uint32_t actS1 = GetActualSeqLen(static_cast<uint32_t>(bIdx), constInfo.actualLenQDims, constInfo.isAccumSeqS1,
                                         actualSeqLengthsGmQ, constInfo.qSeqSize, PKI_SEQ_Q);
        minActS1 = actS1 < minActS1 ? actS1 : minActS1;
    }
    return minActS1;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size,
                                                                      uint32_t& actS2Size, uint32_t& actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize, PKI_SEQ_Q);
    uint32_t actS2Pool = GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm,
                                         constInfo.kSeqSize, PKI_SEQ_K);
    uint32_t tailK = hasPoolTailK_ ? static_cast<uint32_t>(ReadTailKRaw(bIdx)) : 0;
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

template <typename LIT>
__aicore__ inline uint32_t PoolKeyIndexerKernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum =
                constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) : CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) :
                                                        GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间
template <typename LIT>
__aicore__ void inline PoolKeyIndexerKernel<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t& coreNum,
                                                            PkiCommon::SplitCoreInfo& info)
{
    // 计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    if (totalBlockNum == 0U) {
        // 全批次 0 池(纯尾块, pool_tail_k>0): 无块可枚举, 由核 0 以整批区间走
        // DealActSeqLenIsZero(appendTail) 收尾; 其余核区间置为不可循环(start>end)
        coreNum = 1U;
        if (curCoreIdx == 0U) {
            info.bN2Start = 0U;
            info.bN2End = constInfo.batchSize * constInfo.kHeadNum - 1U;
            info.gS1Start = 0U;
            info.gS1End = 0U;
            info.s2Start = 0U;
            info.s2End = 0U;
        } else {
            info.bN2Start = constInfo.batchSize * constInfo.kHeadNum;
            info.bN2End = constInfo.batchSize * constInfo.kHeadNum - 1U;
        }
        return;
    }
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    // 尾部残余 fill 的高水位终点跟踪(见函数末尾注释)
    uint32_t tailBN2End = 0;
    uint32_t tailGS1End = 0;
    uint32_t tailS2End = 0;
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
            s2Loop = constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) : s2BaseNum;
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                // S2 跨核规避(仅 !isLDOpen): 保证每个 (batch, gS1) 的 S2 块
                // 完整落单核(单核串行, 代价是批量并行不足时核利用率下降)。
                // isLDOpen 时放开: S2 块可跨核连续切分, 由 LD 归并(ProcessLD)
                // 合成全局 TopK。
                if (!constInfo.isLDOpen && s2Idx == 0 && lastGS1RemainBlockCnt + s2RemainBaseNum > coreDealBlockCnt) {
                    coreDealBlockCnt = lastGS1RemainBlockCnt + s2RemainBaseNum;
                }
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = constInfo.isSparseCountOver2K ? s2BaseNum - 1 :
                                                                 s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

                    if (coreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize - 1) {
                            info.bN2End = constInfo.batchSize - 1;
                            info.gS1End = 0;
                            info.s2End = 0;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    // 记录未完成 fill 的高水位终点(尾部残余 fill 用)
                    tailBN2End = bN2Idx;
                    tailGS1End = gS1Idx;
                    tailS2End = s2Loop - 1;
                    break;
                }
            }
        }
    }
    // 尾部残余 fill: 最后一段块数可能永远凑不满配额, 由当前核吃满剩余块, 防尾部块漏执行。
    if (!findLastCoreEnd && coreIdx == curCoreIdx) {
        info.bN2End = tailBN2End;
        info.gS1End = tailGS1End;
        info.s2End = tailS2End;
        if (info.bN2End != constInfo.batchSize - 1) {
            info.bN2End = constInfo.batchSize - 1;
            info.gS1End = 0;
            info.s2End = 0;
        }
        return;
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start,
                                                                      bool appendTail)
{
    // 0 池批次的 l_orig 即尾 token 数; 可见性与 ExpandAndAppendIndices 同一公式
    const int32_t poolTailK = (appendTail && hasPoolTailK_) ? static_cast<int32_t>(ReadTailKRaw(bIdx)) : 0;
    const int32_t lOrig = static_cast<int32_t>(tempLoopInfo.actS2SizeOrig);
    if ASCEND_IS_AIV {
        // poolSize>1 时 indices 输出行宽为 outputLen(见 CalcRunInfo 中 idxOutStride 说明)
        uint32_t idxOutStride = (constInfo.poolSize > 1) ?
                                    (constInfo.sparseCount * constInfo.poolSize + constInfo.poolSize - 1) :
                                    constInfo.sparseCount;
        // 0 池收尾: 尾 token 在 UB 内预写随清理整行一次 MTE3 落地(对齐 arch35),
        // 落位与可见性和主路径 ExpandAndAppendIndices 同一公式
        uint32_t tailPos = (constInfo.sparseIndicesMode == 1U) ? 0U : constInfo.sparseCount * constInfo.poolSize;
        if (constInfo.outputLayout == PkiLayout::TND) {
            uint32_t tSize = static_cast<uint32_t>(actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1));
            uint32_t tBase = bIdx == 0 ? 0 : static_cast<uint32_t>(actualSeqLengthsGmQ.GetValue(bIdx - 1));
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                // S1 规模 x 步长乘法提升 64 位: qSeqSize*idxOutStride 可超 2^32(uint32 回绕)
                uint64_t indiceOutOffset = static_cast<uint64_t>(tBase + s1Idx) * constInfo.kHeadNum * idxOutStride +
                                           n2Idx * idxOutStride; // T轴、s1轴偏移
                int32_t tailVisible = 0;
                int32_t tailFirstTok = 0;
                if (poolTailK > 0 && constInfo.poolSize > 1) {
                    tailVisible = poolTailK;
                    if (constInfo.maskMode != 0) {
                        int32_t globalPosQ = lOrig - static_cast<int32_t>(s1Count) + static_cast<int32_t>(s1Idx);
                        tailVisible =
                            PkiCommon::Max(0, PkiCommon::Min(poolTailK, globalPosQ - (lOrig - poolTailK) + 1));
                    }
                    tailFirstTok = lOrig - poolTailK;
                }
                vectorService.CleanInvalidOutput(indiceOutOffset, tailPos, tailVisible, tailFirstTok);
            }
        } else if (constInfo.outputLayout == PkiLayout::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset =
                    static_cast<uint64_t>(bIdx) * constInfo.qSeqSize * constInfo.kHeadNum * idxOutStride +
                    static_cast<uint64_t>(s1Idx) * constInfo.kHeadNum * idxOutStride +
                    n2Idx * idxOutStride; // B轴、S1轴偏移
                int32_t tailVisible = 0;
                int32_t tailFirstTok = 0;
                if (poolTailK > 0 && constInfo.poolSize > 1) {
                    tailVisible = poolTailK;
                    if (constInfo.maskMode != 0) {
                        int32_t globalPosQ =
                            lOrig - static_cast<int32_t>(constInfo.qSeqSize) + static_cast<int32_t>(s1Idx);
                        tailVisible =
                            PkiCommon::Max(0, PkiCommon::Min(poolTailK, globalPosQ - (lOrig - poolTailK) + 1));
                    }
                    tailFirstTok = lOrig - poolTailK;
                }
                vectorService.CleanInvalidOutput(indiceOutOffset, tailPos, tailVisible, tailFirstTok);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* poolKey, __gm__ uint8_t* weights, __gm__ uint8_t* poolTailK,
    __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* qDescale, __gm__ uint8_t* kDescale, __gm__ uint8_t* sparse_indices, __gm__ uint8_t* sparse_values,
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
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);
    if (poolTailK != nullptr) {
        poolTailKGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(poolTailK), constInfo.batchSize);
        hasPoolTailK_ = true;
    }
#if PKI_PARAM_RESIDENT
    // 必须早于 SplitCore —— 后者经 GetTotalBaseBlockNum/GetS1S2ActualSeqLen 逐 batch
    // 读同一批张量, 且分核结果要求全核一致。
    // 三张量的 SetGlobalBuffer 均已完成(上面两行 + InitActualSeqLen)。
    InitParamResident();
#endif
#if PKI_PACK_LOSS_LD_GATE
    // pack-loss LD gate 追加判定(非量化与 int8; FP8 家族 arch22 不可达, 不实例化)
    if constexpr (!LIT::isQuant || LIT::isInt8) {
        RefineLDGateByPackLoss();
    }
#endif

    // 计算分核
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;
    // workspace 排布: |mm1ResGm(存S)|vec1ResGm(LD中间结果)|vec1ParamGm(LD参数)|
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.mBaseSizeAlign * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T*)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    // ld流程 ws: [aic, s1_cube, 头尾, idx/value, virTopK] float; 宽度与 service_vector
    // 的 virTopK 同源(CalcVirTopK), host 侧 ldScoreSize 按 topk 上限 8192 预留,
    // kernel 实写宽度恒在其内
    uint32_t ldTopkW = CalcVirTopK(static_cast<uint32_t>(constInfo.sparseCount), constInfo.isSparseCountOver2K);
    vec1ResGm.SetGlobalBuffer((__gm__ float*)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * ldTopkW * sizeof(float);

    // ld参数 ws: [aic, s1_cube, 头尾, 16ele] int64
    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t*)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t*)sparse_indices);
        valueOutGm.SetGlobalBuffer((__gm__ float*)sparse_values);
        weightsGm.SetGlobalBuffer((__gm__ W_T*)weights);
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, vec1ParamGm, weightsGm, indiceOutGm, valueOutGm);
        if constexpr (LIT::isInt8) {
            // int8 AIV: GetKeyScaleHalf 的 PA 寻址需要 block_table(现仅 AIC 分支设置);
            // descale 以 half 视图接线(FP16, D2)
            if constexpr (PAGE_ATTENTION) {
                blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
            }
            GlobalTensor<half> qScaleGmHalf;
            GlobalTensor<half> kScaleGmHalf;
            qScaleGmHalf.SetGlobalBuffer((__gm__ half*)qDescale);
            kScaleGmHalf.SetGlobalBuffer((__gm__ half*)kDescale);
            vectorService.InitVecQuantTensor(qScaleGmHalf, kScaleGmHalf, blockTableGm);
        }
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T*)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T*)poolKey);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm);
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
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 当前 batch 的尾部 token 数(ExpandAndAppendIndices 尾块追加与 LD 参数均依赖)
    runInfo.poolTailK = hasPoolTailK_ ? static_cast<int32_t>(ReadTailKRaw(tempLoopInfo.bIdx)) : 0;
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
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == PkiLayout::TND) {
            // 组首块的前缀和读点(每 (bN2,gS1) 组一次)同样走常驻; 该读点不带 dims 判空
            // ⇒ InitParamResident 的启用条件② 保证 TND 实例化下两路 actual_seq 均可读,
            // 不存在语义差; 表达式除值的来源外未变。
            actualSeqQPrefixSum =
                (runInfo.bIdx <= 0) ?
                    0 :
                    static_cast<uint64_t>(ReadSeqRaw(PKI_SEQ_Q, runInfo.bIdx - 1, actualSeqLengthsGmQ));
            actualSeqKPrefixSum =
                (runInfo.bIdx <= 0) ?
                    0 :
                    static_cast<uint64_t>(ReadSeqRaw(PKI_SEQ_K, runInfo.bIdx - 1, actualSeqLengthsGm));
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        uint64_t tndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        keyCoreOffset = tndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        if constexpr (LIT::isInt8) {
            // int8 scale 偏移与 arch35 mode0 同构: q_descale 随 weights 行(去 D 轴逐维相等);
            // k_descale 非 PA 为池行前缀基址, PA 由 GetKeyScaleHalf 按 block_table 绝对寻址(不用线性偏移)
            qScaleCoreOffset = weightsCoreOffset;
            kScaleCoreOffset = actualSeqKPrefixSum * constInfo.kHeadNum;
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
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset =
        keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    if constexpr (LIT::isInt8) {
        runInfo.tensorQScaleOffset = qScaleCoreOffset;
        runInfo.tensorKeyScaleOffset = kScaleCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum;
    }
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

                uint32_t negInf = constInfo.INVALID_VAL;
                AscendC::InitGlobalMemory(valueOut, valueDealSize, negInf);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // 无任务核直接返回
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<PkiCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<PkiCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    PkiCommon::RunInfo runInfo;
    uint32_t gloop = 0;
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
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                                   PkiCommon::RunInfo& runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<PkiCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<PkiCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerKernel<LIT>::ProcessDecode()
{
#if PKI_DECODE_LD_GATE
    // isLDOpen 调用门: !isLDOpen 时直接 return。门放在函数首行(在 PipeBarrier/SyncAll
    // 之前、ASCEND_IS_AIV 之外)—— 只门 ProcessLD 而保留 SyncAll 拿不到"移除全设备
    // 会合点"的收益。
    // 数值零影响(逐条): !isLDOpen ⇒ splitCoreInfo.isLD 恒 false ⇒ ProcessLD 是死代码;
    // InitLDBuffers 只重分配 LD 专属缓冲、不写输出张量; ICachePreLoad 是纯 icache 提示。
    // 同步安全(三条独立论据):
    //   ① isLDOpen 由 constInfo(全核同 tiling)推导 ⇒ 全核一致跳过或一致执行,
    //      SyncAll 不会半参与挂死;
    //   ② 跳过 InitLDBuffers 的 pipe->Reset() 不影响主相输出落地: ProcessInvalid 本就
    //      以 InitGlobalMemory(纯 MTE3 GM 写)直接 return 收尾(无排干) ⇒ "kernel 退出时
    //      MTE3 在途写由 runtime/硬件排空"是既有路径已依赖的契约; SyncAll 内的
    //      PipeBarrier<PIPE_ALL> 为跨核会合服务, 不为 kernel 收尾服务;
    //   ③ 无悬挂事件: ExpandSlotDrainAll 在每次槽协议分支退出前已调用, FreeEventID 已
    //      在 ProcessMain 收尾排干 pingpong 槽 ⇒ 进 ProcessDecode 时无未消费的槽 set。
    // 注意: InitLDBuffers 函数体不动 —— 内部 ExpandSlotDrainAll → pipe->Reset() →
    //   ExpandSlotAcquire → SetWaitFlag<MTE3_S> → VToSSync 的次序是精度约束,
    //   改次序会引入 MTE3 死锁/aicore 超时(507014)。本门只改是否调用。
    if (!constInfo.isLDOpen) {
        return;
    }
#endif
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (splitCoreInfo.isLD) {
            vectorService.ProcessLD();
        }
    }
}
} // namespace PkiKernel
#endif // POOL_KEY_INDEXER_KERNEL_ARCH22_H
