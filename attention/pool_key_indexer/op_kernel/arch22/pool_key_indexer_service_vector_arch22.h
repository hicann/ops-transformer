/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_SERVICE_VECTOR_ARCH22_H
#define POOL_KEY_INDEXER_SERVICE_VECTOR_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"
#include "pool_key_indexer_vector.h"

namespace PkiKernel {
using namespace PkiCommon;
using namespace PkiServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;
// 累加器容量下限: 行宽不小于单个 S2 基本块(s2BaseSize=512), 保证
// "新块长度 <= 累加器长度"不变式(与官方 TopK 的 clamp-to-k 模式一致), 并使
// MrgSort/Extract(repeats=virTopK/32) 各长度约束天然满足
constexpr uint32_t VIRTOPK_ALIGN_FLOOR = 512;

// TopK 归并累加器容量统一口径(kernel Init 的 vec1Res 区域宽度与本类的 virTopK
// 必须同源调用本函数, 禁止各自内联重写):
//   over2K(sparseCount>2048): sparseCount; 非 over2K: max(Align(sparseCount,32), 512)
__aicore__ inline uint32_t CalcVirTopK(uint32_t sparseCount, bool isSparseCountOver2K)
{
    if (!isSparseCountOver2K) {
        uint32_t aligned = PkiCommon::Align(sparseCount, static_cast<uint32_t>(32));
        return PkiCommon::Max(aligned, VIRTOPK_ALIGN_FLOOR);
    }
    return sparseCount;
}
// 门2/3/4(sortPP / expandPP / outQueueDepth2)判定链的**纯尺寸模拟**: 三道门的
// 需求量都只由 shape 决定、与门1 的取值无关, 故可在门1 决策之前先跑一遍判定链,
// 得到"若门1 占用 X 字节, 三道门各自开/关"的结果。
// 用途(保证门序不变): 分别在 remain 与 remain-g2Cost 上各跑一遍,
// 要求两次状态**逐项相同**才允许开门1 ⇒ int8 预取区只能占用下游三道门本来也用不
// 到的余量, 任何 shape 下都不会把 sortPPOn_/expandPPOn_/outQueueDepth_ 挤翻
// (含"挤掉 outQueueDepth2"的反向劣化)。
// needX 传 -1 表示该门在本形态下不参与判定(over2K 的门2 / poolSize<=1 的门3),
// 与 InitBuffers 中对应的 if 条件同构。返回 bit0=sortPP bit1=expandPP
// bit2=outQueueDepth2; remainOut 为三道门之后的余量。
__aicore__ inline uint32_t PkiSimGates234(int64_t remain, int64_t needSortPP, int64_t needExpandPP, int64_t needOutQ2,
                                          int64_t& remainOut)
{
    uint32_t st = 0;
    remainOut = remain;
    if (needSortPP >= 0 && needSortPP <= remainOut) {
        st |= 1u;
        remainOut -= needSortPP;
    }
    if (needExpandPP >= 0 && needExpandPP <= remainOut) {
        st |= 2u;
        remainOut -= needExpandPP;
    }
    if (needOutQ2 >= 0 && needOutQ2 <= remainOut) {
        st |= 4u;
        remainOut -= needOutQ2;
    }
    return st;
}
// 展开输出分段长度(int32 元素数): 展开/清理按此分批生成并写 GM, 使
// expandOutBuf_ 固定 4KB, 不随 outputLen 增长越 arch22 192KB UB 上限
constexpr uint32_t EXPAND_CHUNK = 1024;
// vgather 展开路径的段长(独立于标量/vecPath 的 EXPAND_CHUNK; Gather/Muls/Add
// 均为 Level-2 count 形式, 2201 下软件展开为 count-mask, repeatTime=count/64)。
constexpr uint32_t EXPAND_GATHER_CHUNK = 512;
// A2/A3 AIV TPipe UB 池硬上限(release 模式 InitBuffer 越池不报错), 超限回退标量展开。
constexpr uint32_t PKI_UB_POOL_LIMIT_BYTES = 196608;
// poolSize%8!=0 展开路径向量化开关: 1=vgather 向量展开(Gather+Muls+Add);
// 0=回退全标量展开路径(逐 token GetValue/SetValue)。
#ifndef PKI_EXPAND_VEC
#define PKI_EXPAND_VEC 1
#endif
// int8 k_descale 块级提升开关: 1=把 GetKeyScaleHalf +
// SetWaitFlag<MTE2_V> + Cast 从 token 行循环内提升到 base block 之前, 按末行
// cuS2LenMax 抓一次, 行内只保留 Mul(各行仍按自己的 cuS2Len 消费);
// 0=回退原逐行路径(每行各抓一次)。两路径数值 bit-exact: 同块各行的
// bIdx/cuBaseS2Idx 与行号无关, cuS2Len 随行单调不减 => 多抓的
// [cuS2Len_row, cuS2LenMax) 元素不被任何行读取。
// 恒不提升(保持逐行原路径)的运行期条件: isSparseCountOver2K(其 MergeSort
// tmp 落 tmpUb_ 头部, 与本提升区 int8Tail 零间距相邻, 跨行存活不安全)
// 或 cuS2LenMax<=0(本块无有效行)。
#ifndef PKI_KDESCALE_HOIST
#define PKI_KDESCALE_HOIST 1
#endif
// sortOutBuf 的 main/cache 两路径运行期互斥开关(T4): 1=按"逐 batch
// actS1Size 的最小值 > 4"判定缓存路径本次 launch 是否可达, 不可达时只按主路径
// 尺寸分配(净释放 8KB UB); 0=无条件 Max(main, cache) 分配(基线路径)。
// 判据必须是 min over batches(取首批/取 max 都会在混合 batch 下漏判 ⇒
// SortedBasicBlock_ 写越 sortOutBuf, release 态越池不报错, 静默踩相邻 buffer)。
#ifndef PKI_SORTOUT_MAIN_CACHE_EXCLUSIVE
#define PKI_SORTOUT_MAIN_CACHE_EXCLUSIVE 1
#endif
// int8 weights/q_descale 块级行预取开关(G2): 1=int8 复用 bf16 已有的
// wRowCopyOn_ 预取路径, 把 weights 与 q_descale(两者 GM 同偏移)一次整块搬入 UB
// 并把 wq(half 域预乘)落到**独立副本**, 行内每 outerG 只下发 score 一条搬运
// (关门时每行三条 score+weights+q_descale); 0=int8 强制关门(预取门
// false, 走 CopyInInt8 原路径)。
// 硬前置: 本门只在 T4 实际排除了 cache 区(或 over2K 本就不含 cache 区)时才可能
// 打开 —— 否则 int8 的 96B 余量会被 ~704B 预取区吃掉, 静默顶掉 sortPPOn_(归并
// 乒乓)⇒ 每 row-block 多一次 4KB 回拷 + 2 barrier 的性能悬崖, 且 accRowStride_
// 随之改变(MergeSortPP↔MergeSort 切换)。两开关可分别回退, 但"只关 T4 留 G2"
// 会由该前置自动退化为 G2 关闭, 不会出现门序翻转。
#ifndef PKI_INT8_WROW_PREFETCH
#define PKI_INT8_WROW_PREFETCH 1
#endif
// block_table 批量搬运 + PA 分段环排干收敛开关:
// 1=把 GetKeyScaleHalf 的 PA 分段环里"每段 1 次 GM 标量 block_table.GetValue() +
// 1 次阻塞 SetWaitFlag<S_MTE2>"(每块 16 次)收敛为"1 次批量搬 block_table 分片入 UB
// + 1 次 MTE2_S 排干 + 环前 1 次 S_MTE2 排干 + 16 次 UB 标量读";
// 0=逐段路径(基线; 两路径结果逐字节一致)。
// 数值 bit-exact: blockId 取值逐位相同 ⇒ k_descale 的 16 条 DataCopyPad 的源地址/
// 长度/落点一字不变(只改搬运方式与同步条数, 不改计算逻辑)。
// 强制约束: 本开关打开后 blockTableGm **同时**存在标量读(GetValue,
// arch22 走 oriAddress_)与 DMA 读(DataCopyPad, 走已打 tag 的 address_)两条路 ⇒
// 此后**绝不得**对 blockTableGm 施加任何 SetL2CacheHint(尤其 CACHE_MODE_DISABLE),
// 否则两条路访问不同地址语义 ⇒ 静默错值。
#ifndef PKI_BLKTABLE_BATCH_DRAIN
#define PKI_BLKTABLE_BATCH_DRAIN 1
#endif
// block_table 批量分片载体的对齐策略: GM 侧 DataCopyPad 的非对齐源按 32B 包围窗
// 取数, 故越界判定按"元素索引向上对齐到 8 个 int32(=32B)"的窗尾做(见 GetKeyScaleHalf)
constexpr uint32_t BT_ALIGN_BYTES = 32;
constexpr uint32_t BT_ALIGN_ELEMS = BT_ALIGN_BYTES / sizeof(int32_t);
// 批量载体容量(16 × int32 = 64B): 本工况 s2BaseSize_ ≤ 512 且 kCacheBlockSize ≥ 32
// ⇒ 环段数恒 ≤ 16; 段数越出容量(如极小 blockSize 的 P2-2b 形态)时回退逐段路径。
constexpr uint32_t BT_BATCH_ELEMS = 16;
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;

// 主模板：Q_T必选，W_T可选（默认void），无论W_T传什么，默认weightsType=Q_T
template <typename Q_T, typename W_T = void>
struct PkiTypeTraits {
    using weightsType = Q_T; // 默认：weightsType绑定Q_T
};

// 偏特化1：固定第二个参数W_T=float，Q_T保留泛型
template <typename Q_T>
struct PkiTypeTraits<Q_T, float> {
    using weightsType = float; // W_T=float时，强制weightsType为float
};

// FP8模式（仅arch35实际实例化，此处保持同构以防误用）：weights张量为FP16，
// W_T绑定half，避免vector侧weights路径实例化fp8标量语义
template <>
struct PkiTypeTraits<fp8_e4m3fn_t> {
    using weightsType = half;
};

// int8 量化: weights/descale 均为 FP16, W_T 绑定 half;
// entry 已显式传 WEIGHTS_T=half, 此偏特化为双保险(防未显式传入时误推导 int8_t)
template <>
struct PkiTypeTraits<int8_t> {
    using weightsType = half;
};

template <typename LIT>
class PoolKeyIndexerServiceVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    static constexpr PkiLayout LAYOUT_T = LIT::layout;
    using W_DERIVED_T =
        typename PkiTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;
    // 量化场景: PkiType 显式携带 weightsType(half/bfloat16_t), 优先于推导
    using W_EXPLICIT_T = typename LIT::weightsType;
    using W_T = typename std::conditional<std::is_same<W_EXPLICIT_T, void>::value, W_DERIVED_T, W_EXPLICIT_T>::type;

    // MM输出数据类型: int8 为 fixpipe DEQF16 写出的 fp16, 其余 float
    using MM1_OUT_T = typename std::conditional<LIT::isInt8, half, float>::type;

    __aicore__ inline PoolKeyIndexerServiceVector(){};
    __aicore__ inline void ProcessVec(const PkiCommon::RunInfo& info);
    __aicore__ inline void ProcessLD();
    // minActS1OverBatches: 本次 launch 内逐 batch actS1Size 的最小值(kernel 侧
    // CalcMinActS1OverBatches 求出, 与 ProcessVec 的分岔判据 info.actS1Size 同源
    // 同式), 是 sortOutBuf main/cache 两路径运行期互斥的必备判据
    __aicore__ inline void InitBuffers(TPipe* pipe, uint32_t minActS1OverBatches);
    __aicore__ inline void InitParams(const struct PkiCommon::ConstInfo& constInfo,
                                      const PoolKeyIndexerTilingData* __restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<float> valueOutGm);
    // int8 量化输入接线(不改 InitVec1GlobalTensor 签名)
    __aicore__ inline void InitVecQuantTensor(GlobalTensor<half> qScaleGm, GlobalTensor<half> kScaleGm,
                                              GlobalTensor<int32_t> blockTableGm);
    // 无效行清理; tailVisible>0 时为 0 池批次(pool_tail_k>0)收尾行: -1 铺底后在
    // UB 内预写尾 token(对齐 arch35), 单次 DataCopyPad 整行出 GM —— 不做标量 GM
    // 写, 避免与相邻行 MTE3 清理在非 64B 对齐行宽下共享 cache line 的交错污染
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset, uint32_t tailPos = 0U, int32_t tailVisible = 0,
                                              int32_t tailFirstTok = 0);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe* pipe);

private:
    // k_descale(FP16) 分段搬入(非 PA 线性偏移 / PA 按 block_table 绝对寻址), 仅 int8 实例调用
    __aicore__ inline void GetKeyScaleHalf(const PkiCommon::RunInfo& runInfo, const LocalTensor<half>& resUb,
                                           int64_t batchId, int64_t startS2, int64_t getLen);
    // expand 分段写出. expandPP 由调用方按形态决定: 乒乓槽事件机的标量自旋只有在
    // 展开之后本 AIV 仍排有后续向量工作(同组后续行/后续基本块)时才能被重叠吸收,
    // 行数少的组与收尾阶段的展开传 false, 走单槽 + 段间 MTE3ToVSync 原同步语义
    __aicore__ inline void ExpandAndAppendIndices(LocalTensor<int32_t> poolIndices, LocalTensor<int32_t>& tokenIndices,
                                                  LocalTensor<int32_t>& workBuf, GlobalTensor<int32_t>& idxOutGm,
                                                  int64_t idxOutBase, uint32_t sparseCount, uint32_t poolSize,
                                                  uint32_t validS2Len, int32_t poolTailK, int32_t L_orig,
                                                  uint32_t curS1Idx, uint32_t curS1Size, bool expandPP);
    // vgather 展开模板(qOff/pTpl)在 workLocal_ 中的起始偏移(poolIndices 区之后)
    __aicore__ inline uint32_t ExpandGatherTplOffset()
    {
        uint32_t alignedPoolSize = PkiCommon::Align(poolSize_, (uint32_t)8);
        return alignedPoolSize + PkiCommon::Align(static_cast<uint32_t>(constInfo_.sparseCount), (uint32_t)8);
    }
    // 一次性构建 qOff[i]=(i/ps)*4 / pTpl[i]=i%ps 展开模板, 常驻 workLocal_ 尾部
    __aicore__ inline void BuildExpandGatherTpl();

    // 标量(S pipe)与向量(V pipe)/MTE3 间必须显式硬同步,
    // PipeBarrier<PIPE_V> 不保证 S 侧读写顺序
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
    // 向量写(V pipe)后由 MTE3 读出前的真跨流水同步(SetFlag 排入 V 队尾,
    // WaitFlag 在 MTE3 队等待), 分段展开输出每段 DataCopyPad 前使用
    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
    }
    // 上段 DataCopyPad(MTE3 读 tokenIndices) 完成后再覆写 tokenIndices 的同步
    __aicore__ inline void MTE3ToVSync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventID);
        WaitFlag<HardEvent::MTE3_V>(eventID);
    }
    // expand 乒乓槽事件: 段尾 DataCopyPad 后武装(排入 MTE3 队, 该段搬完才置位),
    // 同槽复用前等待并消费。两个槽的事件 id 在 InitBuffers(及 InitLDBuffers 的
    // Reset 之后)一次性专属分配、整个阶段全程持有、绝不中途释放:
    //   1) ReleaseEventID 只清软件占用位且在 wait 下发时刻即生效, 硬件 wait 可能
    //      仍在 V 队列排队; 段级 Alloc/Release 会让"刚释放的 id"被 Fetch 即用即
    //      弃用户(VToMTE3Sync/SetWaitFlag/行前 evtMte3V 屏障, 不置占用位、恒取
    //      first-free)即时取走, 同号 set/wait 交叉配对会丢唤醒(见下方排干注释);
    //   2) 槽 id 永久占用后, Fetch 用户只能取到更高编号, 与槽事件天然隔离。
    // armed 标志区分"已武装未消费"与"id 已持有但 flag=0", 配对严格封闭在本对
    // set/wait 之内, 不依赖池内编号复用。
    // 槽事件 id 的专属分配: InitBuffers 主集与 InitLDBuffers(Reset 清空事件池
    // 占用位后)各调用一次, 之后全程持有。仅 expandPPOn_ 开启(poolSize>1 且 UB
    // 余量门通过)时分配, 持有 2 个 id 后 Fetch 即用即弃屏障自动取更高编号,
    // 与 outQueue 的动态 id 亦不冲突。
    __aicore__ inline void ExpandSlotAcquire()
    {
        expandSlotEvt_[0] = static_cast<int8_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        expandSlotEvt_[1] = static_cast<int8_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        expandSlotArmed_[0] = false;
        expandSlotArmed_[1] = false;
    }
    __aicore__ inline void ExpandSlotWait(uint32_t slot)
    {
        if (expandSlotArmed_[slot]) {
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(expandSlotEvt_[slot]));
            expandSlotArmed_[slot] = false;
        }
    }
    __aicore__ inline void ExpandSlotArm(uint32_t slot)
    {
        expandSlotArmed_[slot] = true;
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(expandSlotEvt_[slot]));
    }
    // 排干两个槽: 等待在途段搬运的 MTE3_V 标记落地并消费。每个使用槽协议的
    // 调用(ExpandAndAppendIndices 的 expandPP 分支 / CleanInvalidOutput 的
    // expandPPOn_ 分支)退出前必须排干:
    //   - 未消费的 set 会让硬件 flag 残留为 1(跨行/跨块/跨 kernel 调用残留),
    //     污染后续在同号 id 上配对的 set/wait —— mssanitizer synccheck 曾报出
    //     60 条未配对的 ExpandSlotArm set_flag, 且残留 flag 与 Fetch 用户共享
    //     first-free 编号时会造成 MTE3 侧 wait_flag 永久等不到置位(aicore
    //     timeout 507014 的直接根因, 见 docs/ 下问题归档);
    //   - 排干的 WaitFlag 排入 V 队列, 天然门控其后下发的所有 V 写(含下一次
    //     展开对槽缓冲的覆写), 与既有行前 evtMte3V 屏障语义等价, 无额外性能代价。
    // InitLDBuffers 的 Reset 前(池占用位被清空、id 失效)也先排干兜底。
    __aicore__ inline void ExpandSlotDrainAll()
    {
        ExpandSlotWait(0);
        ExpandSlotWait(1);
    }
    // 当前行累加器当前位置(仅主路径 actS1Size>4 且非 over2K 时 ping-pong;
    // accPPActive=false 时恒为行距 virTopK*2, 缓存/over2K 路径走此口径)。
    // 乒乓位按行分槽(行距同 accRowStride_), 防跨行覆写
    __aicore__ inline LocalTensor<float> GetAccRow(uint32_t row, bool accPPActive)
    {
        if (accPPActive) {
            return ((accPPBits_ >> row) & 1u) != 0 ? sortPPLocal_[row * accRowStride_] :
                                                     globalTopkUb_[row * accRowStride_];
        }
        return globalTopkUb_[row * static_cast<uint32_t>(virTopK) * 2];
    }

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    // int8 量化输入(FP16 descale + block_table, 仅 isInt8 实例消费)
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // queue
    // outQueue 模板上限深度 2, 实际 bufNum 由 InitBuffers 按 UB 余量运行期决定(1/2);
    // num=1 时 AllocBuffer 游标恒 0、EnQue/DeQue 严格交替, 行为与深度 1 模板一致
    TQue<QuePosition::VECOUT, 2> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    TBuf<TPosition::VECCALC> expandOutBuf_;
    LocalTensor<int32_t> expandOutLocal_;
    TBuf<TPosition::VECCALC> workBuf_;
    LocalTensor<int32_t> workLocal_;

    // 每块一次预取的本 AIV 全部行 weights([rows][gSize] W_T, 行基址 32B 对齐)
    // + 尾部 Cast 落盘区(W_T!=float 时, groupInner float)
    TBuf<TPosition::VECCALC> weightsRowBuf_;
    LocalTensor<W_T> weightsRowT_;
    LocalTensor<float> weightsCastDst_;
    bool wRowCopyOn_ = false;

    // int8 专属(G2: PKI_INT8_WROW_PREFETCH, 下同): q_descale 的块级行预取区, 布局/尺寸与 weightsRowBuf_
    // 同构(两者 GM 偏移同源, 见 CopyInInt8)。wqRowBuf_ 是 wq(half 域预乘)的
    // **独立落盘区**(单行宽, 按 outerG 分片写): DoScaleInt8 的预乘是 Mul(dst,src0,
    // src1), 若 dst 落回预取区即为原地写, 会破坏"预取区全程只读"的不变式。
    // 仅在 wRowCopyOn_ && LIT::isInt8 时分配, 与 weightsRowBuf_ 同一道 UB 门记账。
    TBuf<TPosition::VECCALC> qScaleRowBuf_;
    LocalTensor<W_T> qScaleRowT_;
    TBuf<TPosition::VECCALC> wqRowBuf_;
    LocalTensor<W_T> wqRowT_;

    // int8 专属(block_table 批量分片(≤16 × int32 =
    // 64B)的 UB 载体。**不挂 paramBuf_**(其 128B 已被 LD_PARAM_NUM×int64 用满),
    // 独立 TBuf 在 InitBuffers 的 5 道门**全部之后**记账分配 ⇒ 门序
    // 与 bf16 结构上不可能被影响; remain < 64B ⇒ btBatchOn_=false, GetKeyScaleHalf
    // 走逐段原路径(回退结果逐字节一致)。生命周期同 tmpBuf_/paramBuf_(InitBuffers 一次
    // 分配, InitLDBuffers 不触碰)。
    TBuf<TPosition::VECCALC> btTabBuf_;
    LocalTensor<int32_t> btTabUb_;
    bool btBatchOn_ = false;

    // Sort/Mrg 归并 ping-pong 伙伴缓冲(容量 (virTopK+s2BaseSize)*2 float)
    TBuf<TPosition::VECCALC> sortPPBuf_;
    LocalTensor<float> sortPPLocal_;
    bool sortPPOn_ = false;
    uint32_t accRowStride_ = 0; // float 数; 乒乓开=(virTopK+s2BaseSize)*2, 关=virTopK*2
    uint8_t accPPBits_ = 0;     // per-row 累加器当前位置: 0=globalTopkUb_ 行, 1=sortPPLocal_ 行

    // expand 输出乒乓(2 槽 × 活动路径 segChunk; UB 门开后再由各展开调用点按
    // "展开后是否有后续向量工作"做形态门控, 见 ExpandAndAppendIndices) + outQueue 运行期深度
    bool expandPPOn_ = false;
    uint32_t expandSegChunk_ = 0;        // 活动路径分段长(int32 元素)
    uint32_t expandSegSeq_ = 0;          // 全局段序号(跨行/块/函数连续), slot = seq & 1
    int8_t expandSlotEvt_[2] = {-1, -1}; // per-slot 专属 MTE3_V 事件 id(ExpandSlotAcquire 分配, 阶段内恒定)
    bool expandSlotArmed_[2] = {false, false}; // per-slot 武装态: true=有未消费的段尾 set
    uint8_t outQueueDepth_ = 1;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;

    int32_t blockId_ = -1;
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    // int8: GetKeyScaleHalf 的 PA 寻址参数
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t poolSize_ = 1;
    uint32_t outputLen_ = 0;
    // poolSize%8!=0 且 UB 预算允许时置 true(InitBuffers), 展开走 vgather 路径
    bool expandGatherOn_ = false;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct PkiCommon::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitBuffers(TPipe* pipe, uint32_t minActS1OverBatches)
{
    uint32_t outNeedBufSize =
        (BASE_TOPK * 2) * 2 * sizeof(float); // over2K(sc<=8192) 时 Extract 上界, 不随 virTopK 缩小
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = static_cast<int32_t>(
        CalcVirTopK(static_cast<uint32_t>(constInfo_.sparseCount), constInfo_.isSparseCountOver2K));

    // 尺寸落具名局部量, 供下方 UB 预算按字节核算(与 InitBuffer 调用同源)
    uint32_t tmpBufSize = (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float);
    if constexpr (LIT::isInt8) {
        // int8 追加 k_descale 暂存区(tmpBuf_ 尾部): [kScaleHalf s2BaseSize*2B][kScaleFp32
        // s2BaseSize*4B](Cast 目的在后无重叠); score 的 fp16 暂存复用 float 计算区
        // 行后半(见 CopyInInt8), UB 预算零追加
        tmpBufSize += s2BaseSize_ * sizeof(half) + s2BaseSize_ * sizeof(float);
    }
    // sortOutBuf 布局: [累加器行 x ceil(s1BaseSize/2), virTopK 对/行] +
    //   [SortedBasicBlock_ 缓存行 x 2, 4 槽 x s2BaseSize 对/行](仅 !over2K 的
    //   actS1Size<=4 缓存路径使用)。两路径互斥但共享本 buffer:
    //   - 主路径: 每行 virTopK*2 float(与 MergeSort/Extract/LD 写出同口径)
    //   - 缓存路径: 累加器行同主路径, 缓存行 4*s2BaseSize*2 float(4 槽归并节奏
    //     不变); actS1Size<=4 时每 AIV 至多 2 行
    // "互斥"由**运行期形态**判定(T4: PKI_SORTOUT_MAIN_CACHE_EXCLUSIVE, 见其开关注释), 不再无条件取 max。缓存路径的进入
    //   条件是 (actS1Size<=4 && !over2K)(见 ProcessVec 的 if/else 分岔), 而
    //   actS1Size 逐 batch 取自 actual_seq_q ⇒ 同一次 launch 内不同 batch 可分别
    //   走 cache/main。判据必须是 **所有 batch 的最小值 > 4**(取首批或取 max 都会
    //   在混合 batch 下漏判 ⇒ SortedBasicBlock_ 写越 sortOutBuf, 而 release 态
    //   InitBuffer 越池不报错 ⇒ 静默踩相邻 buffer)。over2K 时缓存路径本就被
    //   ||isSparseCountOver2K 排除, 判据恒成立(与不开 T4 同式)。
    //   min<=4(含 decode 与任一空 batch actS1==0)时保持 max 分配, 与基线分配逐字节一致。
    uint32_t mainSortFloats = CeilDiv(s1BaseSize_, 2) * virTopK * 2;
    uint32_t cacheSortFloats = 2 * (virTopK * 2 + 4 * s2BaseSize_ * 2);
    bool cachePathExcluded = constInfo_.isSparseCountOver2K;
#if PKI_SORTOUT_MAIN_CACHE_EXCLUSIVE
    cachePathExcluded = cachePathExcluded || (minActS1OverBatches > 4);
#endif
    uint32_t sortOutFloats = cachePathExcluded ? mainSortFloats : PkiCommon::Max(mainSortFloats, cacheSortFloats);
    uint32_t sortOutBufSize = sortOutFloats * sizeof(float);
    // 门0 冻结: "T4 前"的 sortOut 尺寸, 专供下方
    // expandGatherOn_ 的绝对余量判定使用 ⇒ 门0 的开/关**结构上不可能**因 T4 释放
    // 8KB 而翻转(该翻转至今未在任何实测中出现, 此处升级为代码结构性保证, 不再依赖外部事实)。
    // PKI_SORTOUT_MAIN_CACHE_EXCLUSIVE=0 时 cachePathExcluded 退化为 over2K,
    // 本式与 sortOutFloats 恒等 ⇒ 该行为恒等变换, 不引入新的编译期路径。
    uint32_t sortOutFloatsNoT4 =
        constInfo_.isSparseCountOver2K ? mainSortFloats : PkiCommon::Max(mainSortFloats, cacheSortFloats);
    uint32_t sortOutBufSizeNoT4 = sortOutFloatsNoT4 * sizeof(float);
    uint32_t indexBufSize = s2BaseSize_ * sizeof(int32_t);
    uint32_t reduceOutBufSize = s2BaseSize_ * 2 * sizeof(float);
    uint32_t brcBufSize = groupInner_ * 8 * sizeof(float);
    uint32_t paramBufSize = LD_PARAM_NUM * sizeof(int64_t);
    uint32_t expandSegChunkBase = (poolSize_ % 8 != 0) ? EXPAND_GATHER_CHUNK : EXPAND_CHUNK;
    uint32_t expandOutLenOld = PkiCommon::Align(static_cast<uint64_t>(expandSegChunkBase), (uint64_t)8);

    // ==== UB 余量门(静态核算, 门序: weights 预取 > 归并乒乓 > expand 乒乓 > queue 深度) ====
    // 先按 vgather 模板决策得到基础占用, 再在 UB 池余量内逐件启用, 任一件
    // 超限则该件不启用(走对应旧路径)
    uint32_t workSize = 0;
    if (poolSize_ > 1) {
        uint32_t idxCopyElems =
            (constInfo_.sparseCount > SPARSE_COUNT_4K) ?
                2 * PkiCommon::Align(static_cast<uint32_t>(constInfo_.sparseCount / 2), (uint32_t)8) :
                PkiCommon::Align(static_cast<uint32_t>(virTopK), (uint32_t)8);
        workSize = PkiCommon::Align(static_cast<uint64_t>(poolSize_), (uint64_t)8) + idxCopyElems;
#if PKI_EXPAND_VEC
        // ps%8!=0 时 workLocal_ 尾部常驻 qOff/pTpl 模板; 按 UB 池上限核算,
        // 超限(如 ps=2/topk=8192→sc=4096)回退标量路径。
        expandGatherOn_ = (poolSize_ % 8 != 0);
        if (expandGatherOn_) {
            uint32_t tplEnd = ExpandGatherTplOffset() + EXPAND_GATHER_CHUNK * 2;
            // fixedOld 的 sortOut 项取 **T4 前**尺寸(门0 冻结, 见上): 使本门的
            // 判定式与基线等价 ⇒ 决策由代码保证不翻转。代价 = 放弃"T4 释放后
            // 本可白拿 vgather 模板"的窗口(机会成本, 不另行记账)。
            uint32_t fixedOld = outNeedBufSize + tmpBufSize + sortOutBufSizeNoT4 + indexBufSize + reduceOutBufSize +
                                brcBufSize + paramBufSize + expandOutLenOld * sizeof(uint32_t);
            uint32_t gatherBytes = PkiCommon::Max(workSize, tplEnd) * sizeof(uint32_t);
            if (fixedOld + gatherBytes <= PKI_UB_POOL_LIMIT_BYTES) {
                workSize = PkiCommon::Max(workSize, tplEnd);
            } else {
                expandGatherOn_ = false; // UB 预算不足, 回退标量路径
            }
        }
#else
        expandGatherOn_ = false;
#endif
    }
    int64_t remain = static_cast<int64_t>(PKI_UB_POOL_LIMIT_BYTES) -
                     static_cast<int64_t>(outNeedBufSize + tmpBufSize + sortOutBufSize + indexBufSize +
                                          reduceOutBufSize + brcBufSize + paramBufSize + workSize * sizeof(uint32_t)) -
                     static_cast<int64_t>((poolSize_ > 1 ? expandOutLenOld : 0u) * sizeof(uint32_t));

    // weights 合并拷贝: [rows][gSize] W_T(行基址 32B 对齐)+ Cast 落盘区
    // int8(G2)额外记 q_descale 预取区(与 weights 同构同尺寸)与 wq 独立
    // 落盘区(单行宽), 三者同属本道门、一次性合并记账(同开同关)
    uint32_t wRowRows = static_cast<uint32_t>(s1BaseSize_ / 2);
    uint32_t wRowBytesAlign = PkiCommon::Align(static_cast<uint64_t>(gSize_) * sizeof(W_T), (uint64_t)32);
    uint32_t wCastBytes = IsSameType<W_T, float>::value ?
                              0 :
                              PkiCommon::Align(static_cast<uint64_t>(groupInner_) * sizeof(float), (uint64_t)32);
    uint32_t wRowNeed = wRowRows * wRowBytesAlign + wCastBytes;
    uint32_t weightsRowBytes = 0;
    uint32_t qScaleRowBytes = 0;
    uint32_t wqRowBytes = 0;

    // ==== 门序冻结(保证门2/3/4 在任何 shape 下逐项不变) ====
    // T4 释放的 UB **只供门1 使用**: 门2/3/4 的开/关决策按"未做 T4"的余量与需求
    // 量冻结(PkiSimGates234 复刻基线判定链), 而**实际分配尺寸**仍按 T4 后的
    // sortOutFloats/spillFloats 计算 ⇒ 四门状态与基线逐项相同(含 bf16), 释放量
    // 全部留给 int8 的预取门。
    // 冻结后恒可负担(解析可证): 门2 的需求增量 = spillFloats 增量
    // = min(mainNew, cacheSort) - mainSort 的 float 数 <= releasedByT4/4, 故基线
    // 装得下 ⇒ 之后仍装得下; 门3/门4 的需求量与 sortOut 无关, 逐字不变。
    // 门0 expandGatherOn_ 的冻结不在此处而在其**自己的判定点**(见上 sortOutBufSizeNoT4):
    // 门0 先于 remain 存在、判据是"绝对装得下"(fixedOld+gatherBytes <= UB)而非
    // "余量够", 结构上无法表达为本判定链的一个 bit, 故按同一冻结思想就地取
    // T4 前尺寸 ⇒ 5 道门全部由代码保证不翻转。
    // sortOutFloatsNoT4 已在上文(门0 冻结处)求出, 此处直接复用。
    int64_t releasedByT4 =
        static_cast<int64_t>(sortOutFloatsNoT4 - sortOutFloats) * static_cast<int64_t>(sizeof(float));
    // remain 是按 **T4 后**(更小的)sortOutBufSize 算出的, 基线余量 = remain - 释放量
    int64_t remainNoT4 = remain - releasedByT4;
    uint32_t mainNewFloats = CeilDiv(s1BaseSize_, 2) * (virTopK + s2BaseSize_) * 2;
    // 需求量: -1 表示该门在本形态下不参与判定(与下方各门的 if 条件同构)
    int64_t needSortPPNoT4 = -1;
    int64_t needSortPPNow = -1;
    if (!constInfo_.isSparseCountOver2K) {
        uint32_t spillNoT4 = mainNewFloats > sortOutFloatsNoT4 ? mainNewFloats - sortOutFloatsNoT4 : 0;
        needSortPPNoT4 = static_cast<int64_t>(spillNoT4 + mainNewFloats) * static_cast<int64_t>(sizeof(float));
        uint32_t spillNow = mainNewFloats > sortOutFloats ? mainNewFloats - sortOutFloats : 0;
        needSortPPNow = static_cast<int64_t>(spillNow + mainNewFloats) * static_cast<int64_t>(sizeof(float));
    }
    uint32_t expandSegChunk = (poolSize_ % 8 != 0) ? EXPAND_GATHER_CHUNK : EXPAND_CHUNK;
    int64_t needExpandPP = (poolSize_ > 1) ? static_cast<int64_t>(2 * expandSegChunk - expandOutLenOld) *
                                                 static_cast<int64_t>(sizeof(uint32_t)) :
                                             -1;
    // 基线门1 的成本(int8 基线被强制关门 ⇒ 恒 0; 其余按同一判据与同一余量口径)
    int64_t g1CostNoT4 = 0;
    if constexpr (!LIT::isInt8) {
        g1CostNoT4 = (static_cast<int64_t>(wRowNeed) <= remainNoT4) ? static_cast<int64_t>(wRowNeed) : 0;
    }
    int64_t frozenRemain = 0;
    uint32_t frozenGates = PkiSimGates234(remainNoT4 - g1CostNoT4, needSortPPNoT4, needExpandPP,
                                          static_cast<int64_t>(outNeedBufSize), frozenRemain);
    // 冻结为 ON 的门在 **T4 后口径**下的实际占用 ⇒ 门1 的可用余量 = remain - frozenCost
    int64_t frozenCost = 0;
    if ((frozenGates & 1u) != 0 && needSortPPNow >= 0) {
        frozenCost += needSortPPNow;
    }
    if ((frozenGates & 2u) != 0 && needExpandPP >= 0) {
        frozenCost += needExpandPP;
    }
    if ((frozenGates & 4u) != 0) {
        frozenCost += static_cast<int64_t>(outNeedBufSize);
    }

    if constexpr (LIT::isInt8) {
        // int8 旁路: 基线路径(CopyInInt8 搬 q_descale 与 weights 同 pattern 同偏移, DoScaleInt8
        // 不消费 weightsRowBuf_)在此保持关门并回滚记账。G2 开时该链路改为消费块级预取
        // 区(行内只读 UB), 每 outerG 的 MTE2 下发由 3 条(score+weights+q_descale)降到 1 条(score)。
        // 两道前置约束(缺一不可, 违反即"门序翻转"):
        //   ① cachePathExcluded —— T4 未排除 cache 区时 int8 余量仅 96B(decode /
        //      混合 batch 恒如此), ~704B 预取区会挤掉下游门 ⇒ 保持基线门序与 remain;
        //   ② wNeedAll <= remain - frozenCost —— 预取区只允许占用"冻结为 ON 的下游
        //      门实际占用之后"的净余量 ⇒ 开门恒不越池、恒不翻门。
        wRowCopyOn_ = false;
        weightsRowBytes = 0;
#if PKI_INT8_WROW_PREFETCH
        if (cachePathExcluded) {
            qScaleRowBytes = wRowNeed;   // q_descale 与 weights 同构(GM 同偏移)
            wqRowBytes = wRowBytesAlign; // wq 独立落盘区: 单行宽, 按 outerG 分片写
            int64_t wNeedAll = static_cast<int64_t>(wRowNeed) + static_cast<int64_t>(qScaleRowBytes) +
                               static_cast<int64_t>(wqRowBytes);
            if (wNeedAll <= remain - frozenCost) {
                weightsRowBytes = wRowNeed;
                wRowCopyOn_ = true;
                remain -= wNeedAll;
            } else {
                qScaleRowBytes = 0;
                wqRowBytes = 0;
            }
        }
#endif
    } else {
        // 非 int8: 判据与基线同式, 只是余量口径换成 remainNoT4(= 基线余量),
        // 使 bf16/fp16 的门1 决策**结构上不可能**因 T4 释放而翻转
        weightsRowBytes = wRowNeed;
        if (static_cast<int64_t>(weightsRowBytes) <= remainNoT4) {
            wRowCopyOn_ = true;
            remain -= static_cast<int64_t>(weightsRowBytes);
        } else {
            wRowCopyOn_ = false;
            weightsRowBytes = 0;
        }
    }

    // Sort/Mrg 回拷乒乓: 仅非 over2K(over2K 的 merge tmp 与 DoScale db 复用
    // tmpUb_, 无法辟出乒乓位); 行距扩为 (virTopK+s2BaseSize)*2, 行尾
    // s2BaseSize 对为 MrgSort 全量输出(len1+len2 对)的 spill 区。
    // 伙伴缓冲必须按行分槽: 每行的归并输出都要落乒乓位, 累加器需跨块存活,
    // 单一共享缓冲会被下一行覆写(多 s2 块场景读旧值 -> 系统性错值)
    uint32_t sortPPPartnerBytes = 0;
    uint32_t spillFloats = 0;
    if (!constInfo_.isSparseCountOver2K) {
        uint32_t mainNew = mainNewFloats;
        spillFloats = mainNew > sortOutFloats ? mainNew - sortOutFloats : 0;
        sortPPPartnerBytes = mainNew * sizeof(float); // rows * (vt+s2)*2 float, 行距同 accRowStride_
        // 门序冻结(见上): 开/关取自基线口径的判定结果, 尺寸与扣减按 T4 后口径
        if ((frozenGates & 1u) != 0) {
            sortPPOn_ = true;
            remain -= static_cast<int64_t>(spillFloats) * static_cast<int64_t>(sizeof(float)) +
                      static_cast<int64_t>(sortPPPartnerBytes);
        } else {
            sortPPOn_ = false;
            spillFloats = 0;
            sortPPPartnerBytes = 0;
        }
    }
    sortOutBufSize += spillFloats * sizeof(float);
    accRowStride_ = sortPPOn_ ? (static_cast<uint32_t>(virTopK) + s2BaseSize_) * 2 : static_cast<uint32_t>(virTopK) * 2;

    // expand 输出乒乓: 2 槽 × 活动路径 segChunk(gather 路径 2*512 与单段
    // 1024 等量, vec/标量路径 +4KB)
    uint32_t expandOutLen = expandOutLenOld;
    if (poolSize_ > 1) {
        expandSegChunk_ = expandSegChunk;
        uint32_t expandOutNew = 2 * expandSegChunk_;
        int64_t delta = static_cast<int64_t>(expandOutNew - expandOutLenOld) * sizeof(uint32_t);
        // 门序冻结(见上): 需求量与 sortOut 无关, 判定结果与基线逐项相同
        if ((frozenGates & 2u) != 0) {
            expandPPOn_ = true;
            expandOutLen = expandOutNew;
            remain -= delta;
        } else {
            expandPPOn_ = false;
        }
    }

    // outQueue 深度 2(EnQue/CopyOut 路径的 MTE3 与下一 Alloc 解锁; ps>1 且
    // returnValue=false 的主输出不走 EnQue, 收益主要在 ps==1 / returnValue 路径)
    // 门序冻结(见上): T4 释放的 UB 不用于打开本门(避免 outQueue 缓冲身份变化)
    if ((frozenGates & 4u) != 0) {
        outQueueDepth_ = 2;
        remain -= static_cast<int64_t>(outNeedBufSize);
    }

#if PKI_BLKTABLE_BATCH_DRAIN
    // block_table 批量载体(64B)记账: 本扣减**排在全部 5 道门之后** ⇒ 门
    // 决策(取自 frozenGates 与各门判定点的 remain 快照)结构上看不到它 ⇒ 门序冻结
    // 零翻转由构造保证。int8-only(if constexpr), bf16/fp16 的 remain 不变。
    // 运行期余量不足 64B 时退化为 btBatchOn_=false(逐段原路径), 不新增失败模式。
    // 期望值: int8 prefill remain 7584→7520B(安全垫 3.8%), decode 96→32B。
    if constexpr (LIT::isInt8) {
        if (remain >= static_cast<int64_t>(BT_BATCH_ELEMS * sizeof(int32_t))) {
            btBatchOn_ = true;
            remain -= static_cast<int64_t>(BT_BATCH_ELEMS * sizeof(int32_t));
        }
    }
#endif

    pipe->InitBuffer(outQueue_, outQueueDepth_, outNeedBufSize); // 32KB  extract
    // 68KB 在搬运cube核计算得到的结果和weight时，分成两块34KB，用于db；在mrgsort时，用作临时UB
    pipe->InitBuffer(tmpBuf_, tmpBufSize);
    pipe->InitBuffer(sortOutBuf_, sortOutBufSize);     // 64KB
    pipe->InitBuffer(indexBuf_, indexBufSize);         // 2KB
    pipe->InitBuffer(reduceOutBuf_, reduceOutBufSize); // 4KB
    pipe->InitBuffer(brcBuf_, brcBufSize);
    pipe->InitBuffer(paramBuf_, paramBufSize);
    if (wRowCopyOn_) {
        pipe->InitBuffer(weightsRowBuf_, weightsRowBytes);
        uint32_t rowBytes = PkiCommon::Align(static_cast<uint64_t>(gSize_) * sizeof(W_T), (uint64_t)32);
        weightsRowT_ = weightsRowBuf_.Get<W_T>();
        if (!IsSameType<W_T, float>::value) {
            // Cast 落盘区: 行区之后 32B 对齐偏移(与源区不重叠)
            weightsCastDst_ =
                weightsRowBuf_.Get<float>()[(static_cast<uint32_t>(s1BaseSize_ / 2) * rowBytes) / sizeof(float)];
        }
        if constexpr (LIT::isInt8) {
            // G2: q_descale 预取区(与 weightsRowBuf_ 同布局同行距 ⇒ 消费侧
            // 可用同一 preOff 寻址)与 wq 独立落盘区(单行宽, 按 outerG 分片)。
            // 二者仅在 wRowCopyOn_ 时被分配与消费, 与上面同一道 UB 门记账。
            pipe->InitBuffer(qScaleRowBuf_, qScaleRowBytes);
            qScaleRowT_ = qScaleRowBuf_.Get<W_T>();
            pipe->InitBuffer(wqRowBuf_, wqRowBytes);
            wqRowT_ = wqRowBuf_.Get<W_T>();
        }
    }
    if (sortPPOn_) {
        pipe->InitBuffer(sortPPBuf_, sortPPPartnerBytes);
        sortPPLocal_ = sortPPBuf_.Get<float>();
    }

    if (poolSize_ > 1) {
        // expandOutLocal_ 2 槽(每槽 segChunk, 未启用乒乓时为单段), 展开/清理按此
        // 分段写 GM, 避免整行 outputLen 分配越 arch22 192KB UB 上限
        pipe->InitBuffer(expandOutBuf_, expandOutLen * sizeof(uint32_t));
        expandOutLocal_ = expandOutBuf_.Get<int32_t>();
        pipe->InitBuffer(workBuf_, workSize * sizeof(uint32_t));
        workLocal_ = workBuf_.Get<int32_t>();
        // 向量路径的展开模板经 CreateVecIndex 一次性构建常驻 workLocal_ 头部,
        // 替代每行标量重建的逐行开销(与消费方同 pipe, 常驻安全)
        if (poolSize_ % 8 == 0) {
            AscendC::CreateVecIndex(workLocal_, static_cast<int32_t>(0), poolSize_);
        }
#if PKI_EXPAND_VEC
        else if (expandGatherOn_) {
            BuildExpandGatherTpl();
        }
#endif
        // 槽事件 id 专属分配(见 ExpandSlotAcquire 注释): 两个 id 全程持有,
        // Fetch 即用即弃屏障只能取到更高编号, 与槽事件彻底隔离
        if (expandPPOn_) {
            ExpandSlotAcquire();
        }
    }

#if PKI_BLKTABLE_BATCH_DRAIN
    // 载体分配排在**所有既有 InitBuffer 之后**(与上方记账同一意图: 既有 buffer 的
    // UB 相对次序/地址逐字不变, 新 64B 只吃池尾, 不扰动 bank/对齐相位)。
    // DataCopyPad(GM→UB) 要求 dst 32B 对齐: TPipe 的 VECCALC buffer 基址天然 32B 对齐。
    if (btBatchOn_) {
        pipe->InitBuffer(btTabBuf_, BT_BATCH_ELEMS * sizeof(int32_t));
        btTabUb_ = btTabBuf_.Get<int32_t>();
    }
#endif

    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    SortedBasicBlock_ = globalTopkUb_[virTopK * 2 * 2];
    globalTopkNum_ = 0;
    accPPBits_ = 0;

    // 累加器索引模板: [0, s2BaseSize_) 有序索引, 供排序输入的 idx 区加上 S2 块基址
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // 累加器铺底: globalTopkUb_ [CeilDiv(s1BaseSize_,2), virTopK, 2] 全部置 (-inf,-1) 交织对;
    // 乒乓时按 accRowStride_ 连续铺满(交织周期 2, 偶偏移保相位, spill 区一并初始化)
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * accRowStride_);

    // LD 参数区铺底: vec1ParamGm 布局 [aic, 2, s1BaseSize_, 16], needFd 置 -1
    // (LD 轮询到非 1 即跳过该行归约)
    LocalTensor<float> tmpBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    outQueue_.EnQue<float>(tmpBuff);
    tmpBuff = outQueue_.DeQue<float>();
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +      // 2个AIV共同地址偏移
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_; // 每个AIV的地址偏移，S1方向
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    outQueue_.FreeTensor(tmpBuff);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitLDBuffers(TPipe* pipe)
{
    // Reset 会清空事件池占用位, 先排干在途 expand 段搬运并消费槽事件标记,
    // 防止跨 Reset 的悬挂置位残留
    ExpandSlotDrainAll();
    pipe->Reset();
    // Reset 已清空事件池占用位, 槽事件 id 需重新专属分配(LD 阶段 ProcessLD 的
    // 展开/清理仍走同一槽协议, 见 ExpandSlotAcquire 注释)
    if (poolSize_ > 1 && expandPPOn_) {
        ExpandSlotAcquire();
    }
    // Reset 后按 LD 布局重分配缓冲(地址与主集不同): 先排空主集在途的
    // MTE3 读与 V 写, 再重建展开模板, 避免与模板 S 写竞态
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    VToSSync();
    // isLDOpen gate 已排除 over2K(活域恒 !over2K): ldTopkW=virTopK 精确覆盖
    // ProcessLD 的读写宽度。!isLDOpen 时 SplitCore 的 S2 跨核规避使 isLD 恒
    // 不可达(over2K 被 gate 排除 / 并行单元充足), LD 读侧(ProcessLD, 本组
    // 缓冲的唯一消费者)无消费者, LD 缓冲宽度按 BASE_TOPK 预算钳制, 避免
    // 72*sc 字节需求越 arch22 192KB UB 池(release 态 InitBuffer 越池不报错)。
    // 若未来放开 !isLDOpen 的 S2 跨核切分, 须先恢复按 virTopK 分配并重核 UB 预算。
    uint32_t ldTopkW = constInfo_.isLDOpen ? static_cast<uint32_t>(virTopK) : BASE_TOPK;
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * ldTopkW * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * ldTopkW * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, ldTopkW * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, ldTopkW * sizeof(int32_t));
    if (poolSize_ > 1 && constInfo_.isLDOpen) {
        // expand/work 重配与展开模板重建仅 LD 阶段消费, !isLDOpen 随 isLD 不可达
        // 一并跳过(同为死路径预算, 免除模板重建的标量写与同步开销)
        // 与 InitBuffers 同源的乒乓槽尺寸(expandPPOn_/expandSegChunk_ 已在
        // InitBuffers 决策), LD 阶段 ExpandAndAppendIndices 沿用同一槽协议
        uint32_t expandOutLen =
            expandPPOn_ ? 2 * expandSegChunk_ : PkiCommon::Align(static_cast<uint64_t>(EXPAND_CHUNK), (uint64_t)8);
        pipe->InitBuffer(expandOutBuf_, expandOutLen * sizeof(uint32_t));
        expandOutLocal_ = expandOutBuf_.Get<int32_t>();
        uint32_t workSize = PkiCommon::Align(static_cast<uint64_t>(poolSize_ + 64), (uint64_t)8) +
                            PkiCommon::Align(static_cast<uint64_t>(constInfo_.sparseCount + 64), (uint64_t)8);
#if PKI_EXPAND_VEC
        // 与 InitBuffers 同式重算模板尺寸(expandGatherOn_ 已在 InitBuffers 门禁置位)
        if (expandGatherOn_) {
            uint32_t tplEnd = ExpandGatherTplOffset() + EXPAND_GATHER_CHUNK * 2;
            workSize = PkiCommon::Max(workSize, tplEnd);
        }
#endif
        pipe->InitBuffer(workBuf_, workSize * sizeof(uint32_t));
        workLocal_ = workBuf_.Get<int32_t>();
        // Reset 后重配的 LD 阶段 workLocal_ 需重建展开模板(见 InitBuffers)
        if (poolSize_ % 8 == 0) {
            AscendC::CreateVecIndex(workLocal_, static_cast<int32_t>(0), poolSize_);
        }
#if PKI_EXPAND_VEC
        else if (expandGatherOn_) {
            BuildExpandGatherTpl();
        }
#endif
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitParams(
    const struct PkiCommon::ConstInfo& constInfo, const PoolKeyIndexerTilingData* __restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;

    groupInner_ = 16;

    blockId_ = GetBlockIdx();
    poolSize_ = static_cast<uint32_t>(constInfo.poolSize);
    outputLen_ = constInfo.sparseCount * poolSize_ + poolSize_ - 1;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm, GlobalTensor<int64_t> vec1ParamGm,
    GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm, GlobalTensor<float> valueOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitVecQuantTensor(GlobalTensor<half> qScaleGm,
                                                                            GlobalTensor<half> kScaleGm,
                                                                            GlobalTensor<int32_t> blockTableGm)
{
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->blockTableGm = blockTableGm;
}

// k_descale(FP16) 分段搬入, 移植自 v1 GetKeyScale(L117-163)并 half 化:
// PA 按 block_table×keyDequantScaleStride0 绝对寻址(块边界截断分段), 非PA 线性偏移;
// S_MTE2 硬同步保证 block_table 标量读(S pipe)与搬运(MTE2)有序
template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::GetKeyScaleHalf(const PkiCommon::RunInfo& runInfo,
                                                                         const LocalTensor<half>& resUb,
                                                                         int64_t batchId, int64_t startS2,
                                                                         int64_t getLen)
{
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.blockCount = 1;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            // 首段跨块截断: 从块内偏移搬至块尾
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(half);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb, kScaleGm[blockId * constInfo_.keyDequantScaleStride0 + startBlockTableOffset],
                                 copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(half);
#if PKI_BLKTABLE_BATCH_DRAIN
        // 环内"每段 1 次 GM 标量读 + 1 次 S_MTE2 全
        // 排干"收敛为"1 次批量搬 block_table 分片入 UB + 1 次 MTE2_S 落地排干 +
        // 1 次环前 S_MTE2 排干(硬约束: 16→1, 不得→0) + getLoopNum 次 UB 标量读"。
        // 收益锚点 = 消除 15 次/块的**阻塞排干**(带排干档 ~23ns/条), 不是搬运字节
        // (搬运字节守恒, 收益来自消除阻塞排干而非字节合并; aiv_mte2 预期 Δ≈0)。
        // bit-exact: blockId 逐位相同(同一批 GM 值, 只是先经 UB 中转)⇒ 16 条
        // k_descale DataCopyPad 的源址/长度/落点一字不变; 首段跨块截断分支(上方,
        // startBlockTableOffset>0)不在此列, 保持逐字原样(自带 1 次 GM 读+排干)。
        // clamp(硬约束①): 批量读按 32B 包围窗取数, 越过本行行尾(读下一行行首)
        //   合法 ⇒ 窗口在 tensor 内且垃圾表项从不被消费; 但窗尾 ceil8 越过
        //   block_table **元素总数**(末 batch + ≤15 元素的包围窗越出 GM 分配尾,
        //   静默/宿主被杀)⇒ 判据 ceil8(btStart+getLoopNum) ≤ B*maxBlockNumPerBatch,
        //   不满足即整次调用回退逐段原路径(结果逐字节一致)。
        // 段数 > BT_BATCH_ELEMS(极小 blockSize 形态)同样回退, 容量由构造封顶。
        bool btBatchDone = false;
        if (btBatchOn_ && getLoopNum > 0 && getLoopNum <= static_cast<int32_t>(BT_BATCH_ELEMS)) {
            const int64_t btStart =
                static_cast<int64_t>(blockTableBatchOffset) + static_cast<int64_t>(startBlockTableIdx);
            const int64_t btTotal =
                static_cast<int64_t>(constInfo_.batchSize) * static_cast<int64_t>(maxBlockNumPerBatch_);
            const int64_t btWinTail =
                (btStart + static_cast<int64_t>(getLoopNum) + BT_ALIGN_ELEMS - 1) / BT_ALIGN_ELEMS * BT_ALIGN_ELEMS;
            if (btWinTail <= btTotal) {
                AscendC::DataCopyExtParams btCopyParams;
                btCopyParams.blockCount = 1;
                btCopyParams.blockLen = static_cast<uint16_t>(getLoopNum * sizeof(int32_t));
                btCopyParams.srcStride = 0;
                btCopyParams.dstStride = 0;
                btCopyParams.rsv = 0;
                AscendC::DataCopyPadExtParams<int32_t> btPadParams{false, 0, 0, 0};
                // 重要约束: 本行起 blockTableGm 同时存在 DMA 读(已打 tag 的
                // address_)与标量读(oriAddress_, 截断分支/回退路径)两条路 ⇒ 任何
                // 对其施加的 SetL2CacheHint 都会让两条路访问不同地址语义(静默错值)。
                AscendC::DataCopyPad(btTabUb_, blockTableGm[btStart], btCopyParams, btPadParams);
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S); // 批量拷贝落地后才能 S 读 UB(MTE2→S)
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2); // 环前保留 1 次排干(16→1, 保守档)
                for (int32_t i = 0; i < getLoopNum; i++) {
                    if (i == getLoopNum - 1) {
                        copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(half);
                    }
                    int32_t blockId = btTabUb_.GetValue(static_cast<uint32_t>(i));
                    uint64_t kScaleOffset = blockId * constInfo_.keyDequantScaleStride0;
                    AscendC::DataCopyPad(resUb[resUbBaseOffset + i * kCacheBlockSize_], kScaleGm[kScaleOffset],
                                         copyInParams, padParams);
                }
                btBatchDone = true;
            }
        }
        if (!btBatchDone) {
#endif
            for (int32_t i = 0; i < getLoopNum; i++) {
                if (i == getLoopNum - 1) {
                    copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(half);
                }
                int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                uint64_t kScaleOffset = blockId * constInfo_.keyDequantScaleStride0;
                AscendC::DataCopyPad(resUb[resUbBaseOffset + i * kCacheBlockSize_], kScaleGm[kScaleOffset],
                                     copyInParams, padParams);
            }
#if PKI_BLKTABLE_BATCH_DRAIN
        }
#endif
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(half);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(resUb, kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

#if PKI_EXPAND_VEC
// 一次性构建 vgather 展开模板常驻 workLocal_ 尾部只读: qOff[i]=(i/ps)*4,
// pTpl[i]=i%ps; S pipe 标量写, SToVSync 保证后续 V pipe 读可见
template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::BuildExpandGatherTpl()
{
    uint32_t ps = poolSize_;
    uint32_t tplOff = ExpandGatherTplOffset();
    LocalTensor<int32_t> qOffI32 = workLocal_[tplOff];
    LocalTensor<int32_t> pTplI32 = workLocal_[tplOff + EXPAND_GATHER_CHUNK];
    for (uint32_t i = 0; i < EXPAND_GATHER_CHUNK; i++) {
        qOffI32.SetValue(i, static_cast<int32_t>((i / ps) * sizeof(int32_t)));
        pTplI32.SetValue(i, static_cast<int32_t>(i % ps));
    }
    SToVSync(); // 标量写(S pipe) -> 后续 V pipe Gather/Muls/Add 读可见
}
#endif

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ExpandAndAppendIndices(
    LocalTensor<int32_t> poolIndices, LocalTensor<int32_t>& tokenIndices, LocalTensor<int32_t>& workBuf,
    GlobalTensor<int32_t>& idxOutGm, int64_t idxOutBase, uint32_t sparseCount, uint32_t poolSize, uint32_t validS2Len,
    int32_t poolTailK, int32_t L_orig, uint32_t curS1Idx, uint32_t curS1Size, bool expandPP)
{
    uint32_t topk = sparseCount * poolSize;
    uint32_t totalOut = topk + poolSize - 1;

    // 尾块可见 token 数: 尾池在原始序列中的真实起点为 seqTailStart = L_orig -
    // poolTailK (= actualSeqK*poolSize), mask_mode!=0 时按该起点对 query 全局位置
    // 因果截断。topk 为选池预算, 仅当 actualSeqK==sparseCount 时才与 seqTailStart
    // 相等, 不可混作尾池起点(短序列会漏合法尾 token, 长序列多 query 会追加未来
    // token)。mask_mode=0 分支保持全尾可见不变
    int32_t visibleTailK = 0;
    if (poolTailK > 0) {
        if (constInfo_.maskMode == 0) {
            visibleTailK = poolTailK;
        } else {
            int32_t globalPosQ = L_orig - static_cast<int32_t>(curS1Size) + static_cast<int32_t>(curS1Idx);
            int32_t seqTailStart = L_orig - poolTailK;
            visibleTailK = PkiCommon::Max(0, PkiCommon::Min(poolTailK, globalPosQ - seqTailStart + 1));
        }
    }

    uint32_t expandRounds = (validS2Len > 0) ? PkiCommon::Min(validS2Len, sparseCount) : 0;
    // 索引输出布局模式(尾块追加模式): 1=compact(展开区有效长度与 tail 写入位置均为
    // 实际展开 token 数, 短序列时 tail 紧跟有效展开, -1 全部落在行尾, 消除缝隙);
    // 其他(含 0)=legacy(展开区铺底到 topk, 短序列 [expandTotal, topk) 为 -1 缝隙,
    // tail 固定偏移 topk, 与历史版本逐比特一致)。V1 接口恒 legacy, V2 接口按
    // sparse_indices_mode 属性选择。
    uint32_t expandTotal = expandRounds * poolSize;
    uint32_t expandLimit = (constInfo_.sparseIndicesMode == 1U) ? expandTotal : topk;
    uint32_t tailStart = expandLimit;
    bool vecPath = (poolSize % 8 == 0);
#if PKI_EXPAND_VEC
    bool gatherPath = !vecPath && expandGatherOn_; // ps%8!=0 时走 vgather 向量展开
#else
    bool gatherPath = false;
#endif
    uint32_t alignedPoolSize = PkiCommon::Align(poolSize, (uint32_t)8);

    // 展开模板已一次性构建常驻 workBuf 头部, 无需逐行标量重建
    LocalTensor<int32_t> offsetTpl = workBuf;
    // vgather 展开模板 qOff/pTpl 常驻 workLocal_ 尾部(见 BuildExpandGatherTpl);
    // 声明置于 #if 外保证宏关时仍可编译(gatherPath 恒 false 不进入)
    LocalTensor<uint32_t> qOffTpl;
    LocalTensor<int32_t> pTpl;
#if PKI_EXPAND_VEC
    if (gatherPath) {
        uint32_t tplOff = ExpandGatherTplOffset();
        qOffTpl = workBuf[tplOff].template ReinterpretCast<uint32_t>();
        pTpl = workBuf[tplOff + EXPAND_GATHER_CHUNK];
    }
#endif
    if (expandRounds > 0 && !gatherPath) {
        VToSSync(); // poolIndices 向量写 → 后续标量 GetValue 可见(gatherPath 无 S 侧读)
    }

    // 分段生成写出: expandOutBuf_ 2 槽乒乓(每槽 segChunk, 未启用乒乓时为单段),
    // 不随 outputLen 增长越 UB 上限; 段边界对齐到池粒度(向量路径保持 32B 对齐)
    // segChunk 必须等于 expandOutBuf_ 单槽容量(expandSegChunk_), 不可按展开路径另取:
    // 段写(Duplicate/SetValue/Gather)按段长铺满, 超限越界覆写相邻 workBuf_ 的池索引区
    uint32_t segChunk = expandSegChunk_;
    uint32_t poolsPerSeg = PkiCommon::Max((uint32_t)1, segChunk / poolSize);

    // ---- 展开区 [0, expandLimit): 按池对齐分段 ----
    // (compact: expandLimit=expandTotal, 只写有效展开; legacy: expandLimit=topk,
    //  短序列时 [expandTotal, topk) 段保持本段 -1 铺底即历史缝隙行为)
    uint32_t segDone = 0;
    bool outStarted = false;
    while (segDone < expandLimit) {
        uint32_t kLen = PkiCommon::Min(poolsPerSeg, (expandLimit - segDone) / poolSize);
        uint32_t segLen = kLen * poolSize; // 池对齐, <= EXPAND_CHUNK

        // 段写目标按乒乓槽交替, 只等"同槽上一次 DataCopyPad 读完"(段 i-2),
        // 段 i 的 V 写与段 i-1 的 MTE3 读重叠; 未启用乒乓时段间全排干
        uint32_t slot = expandSegSeq_ & 1u;
        if (expandPP) {
            ExpandSlotWait(slot);
        } else if (outStarted) {
            MTE3ToVSync(); // 上一段 DataCopyPad 读完 tokenIndices 方可覆写
        }
        outStarted = true;
        LocalTensor<int32_t> segTok = (expandPP) ? tokenIndices[slot * segChunk] : tokenIndices;
        Duplicate<int32_t>(segTok, -1, segLen);
        PipeBarrier<PIPE_V>();
        if (!vecPath && !gatherPath) {
            // V→S: 标量 SetValue 须等本段 -1 填充(V 写)落地, 否则会被迟到的 V 写覆盖
            VToSSync();
        }

        uint32_t kLo = segDone / poolSize;
        uint32_t kHi = PkiCommon::Min(kLo + kLen, expandRounds);
        if (gatherPath) {
            // vgather 向量展开(全 V pipe): segTok[i] = poolIndices[kLo+i/ps]*ps + i%ps;
            // count 模式下 [validLen,segLen) 不写, kHi<kLo 时 validLen 归零防下溢
            uint32_t validLen = (kHi > kLo) ? (kHi - kLo) * poolSize : 0;
            if (validLen > 0) {
                Gather(segTok, poolIndices, qOffTpl, static_cast<uint32_t>(kLo * sizeof(int32_t)), validLen);
                PipeBarrier<PIPE_V>(); // Gather 写 → Muls 原地读
                Muls(segTok, segTok, static_cast<int32_t>(poolSize), validLen);
                PipeBarrier<PIPE_V>(); // Muls 写 → Add 原地读
                Add<int32_t>(segTok, segTok, pTpl, validLen);
            }
        } else {
            for (uint32_t k = kLo; k < kHi; k++) {
                int32_t base = poolIndices.GetValue(k) * static_cast<int32_t>(poolSize);
                uint32_t off = (k - kLo) * poolSize;
                if (vecPath) {
                    // 向量展开: Duplicate(广播 base) + Add(加模板), 全 SIMD
                    Duplicate<int32_t>(segTok[off], base, alignedPoolSize);
                    PipeBarrier<PIPE_V>();
                    Add<int32_t>(segTok[off], segTok[off], offsetTpl, alignedPoolSize);
                } else {
                    // 非 8 倍数 poolSize: 无法用 32B 对齐向量写, 逐 token 标量写
                    for (uint32_t p = 0; p < poolSize; p++) {
                        segTok.SetValue(off + p, base + static_cast<int32_t>(p));
                    }
                }
            }
        }

        // V 写(Duplicate -1 / 向量展开)对 MTE3(DataCopyPad) 可见; 标量路径另需 S→MTE3
        VToMTE3Sync();
        if (!vecPath && !gatherPath) {
            SToMTE3Sync();
        }
        DataCopyPad(idxOutGm[idxOutBase + segDone], segTok, {1, static_cast<uint16_t>(segLen * sizeof(int32_t)), 0, 0});
        if (expandPP) {
            ExpandSlotArm(slot); // 本段 DataCopyPad 完成后置位同槽事件
            expandSegSeq_++;
        }
        segDone += segLen;
    }

    // ---- 尾部区 [tailStart, totalOut): -1 铺底 + 区首 tail token, 分段写出 ----
    // (compact: tailStart=expandTotal, tail 紧跟有效展开, 短序列时区长可达
    //  topk+ps-1(如 2048+15=2063 > EXPAND_CHUNK), 须按 segChunk 分段;
    //  legacy: tailStart=topk, 区长恒 ps-1 单段, 与历史实现等价。
    //  tail token 恒落首段(visibleTailK <= ps-1 <= 127 << segChunk), 段间同步与
    //  展开段同协议(乒乓槽事件 / 单槽 MTE3ToVSync)。)
    uint32_t tailRegion = totalOut - tailStart;
    uint32_t tDone = 0;
    while (tDone < tailRegion) {
        uint32_t segLen = PkiCommon::Min(segChunk, tailRegion - tDone);
        uint32_t tSlot = expandSegSeq_ & 1u;
        if (expandPP) {
            ExpandSlotWait(tSlot);
        } else if (outStarted) {
            MTE3ToVSync(); // 上一段 DataCopyPad 读完 tokenIndices 方可覆写
        }
        outStarted = true;
        LocalTensor<int32_t> tailTok = (expandPP) ? tokenIndices[tSlot * segChunk] : tokenIndices;
        Duplicate<int32_t>(tailTok, -1, segLen); // 精确 segLen, 不 Align(尾段防越槽容量)
        if (tDone == 0 && visibleTailK > 0) {
            // tail token 固定在区首(tailPos == tailStart)
            VToSSync(); // -1 填充先落地, 防迟到 V 写覆盖标量写
            for (int32_t t = 0; t < visibleTailK; t++) {
                tailTok.SetValue(t, L_orig - poolTailK + t);
            }
            SToMTE3Sync();
        }
        VToMTE3Sync();
        DataCopyPad(idxOutGm[idxOutBase + tailStart + tDone], tailTok,
                    {1, static_cast<uint16_t>(segLen * sizeof(int32_t)), 0, 0});
        if (expandPP) {
            ExpandSlotArm(tSlot);
            expandSegSeq_++;
        }
        tDone += segLen;
    }

    // 退出前排干两槽(见 ExpandSlotDrainAll 注释): 展开区+尾区最后两个在途段
    // 的 set 必须在本调用内消费, 硬件 flag 归零后再返回 —— 任何跨行/跨块/
    // 跨 kernel 调用的残留置位都会污染后续同号事件的 set/wait 配对。
    // 排干 WaitFlag 排入 V 队列, 下一次展开对本双槽缓冲的覆写天然排在其后,
    // 与既有行前 evtMte3V 屏障语义等价, 不损失段间乒乓重叠
    if (expandPP) {
        ExpandSlotDrainAll();
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset, uint32_t tailPos,
                                                                            int32_t tailVisible, int32_t tailFirstTok)
{
    uint32_t idxOutLen = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
    if (poolSize_ > 1) {
        // 分段写 -1: 按活动分段长(乒乓启用时 expandSegChunk_ 分槽, 否则
        // EXPAND_CHUNK 单段) Duplicate -1 + DataCopyPad 写 GM, 避免按整行
        // idxOutLen 铺满 expandOutLocal_ 越界
        // 0 池收尾行(tailVisible>0): 含尾区间的段在 Duplicate 铺底后在 UB 内
        // 预写尾 token(VToSSync→SetValue→SToMTE3Sync, 与 ExpandAndAppendIndices
        // 尾块同款次序), 单次 DataCopyPad 整段出 GM —— 标量只落 UB, 不直接写
        // GM, 消除与相邻行 MTE3 清理在非对齐行宽共享 cache line 的交错污染
        uint32_t segDone = 0;
        while (segDone < idxOutLen) {
            uint32_t slot = expandSegSeq_ & 1u;
            uint32_t segLen;
            if (expandPPOn_) {
                ExpandSlotWait(slot);
                segLen = PkiCommon::Min(expandSegChunk_, idxOutLen - segDone);
            } else {
                segLen = PkiCommon::Min(EXPAND_CHUNK, idxOutLen - segDone);
                if (segDone > 0) {
                    MTE3ToVSync();
                }
            }
            LocalTensor<int32_t> segBuf = (expandPPOn_) ? expandOutLocal_[slot * expandSegChunk_] : expandOutLocal_;
            Duplicate(segBuf, constInfo_.INVALID_IDX, PkiCommon::Align(segLen, (uint32_t)8));
            if (tailVisible > 0) {
                // 段 [segDone, segDone+segLen) 与尾区 [tailPos, tailPos+tailVisible) 的交集
                uint32_t lo = PkiCommon::Max(segDone, tailPos);
                uint32_t hi = PkiCommon::Min(segDone + segLen, tailPos + static_cast<uint32_t>(tailVisible));
                if (lo < hi) {
                    VToSSync(); // Duplicate 的 -1 铺底落地后, 标量写才可见
                    for (uint32_t p = lo; p < hi; p++) {
                        segBuf.SetValue(p - segDone, tailFirstTok + static_cast<int32_t>(p - tailPos));
                    }
                    SToMTE3Sync(); // UB 标量写落地后, MTE3 才能读出
                }
            }
            // V(Duplicate/SetValue) 写 segBuf 后由 MTE3(DataCopyPad) 搬出,
            // 必须真跨流水同步(仅 PipeBarrier<PIPE_V> 不够, 与主输出路径一致)
            VToMTE3Sync();
            DataCopyPad(indiceOutGm[invalidS1offset + segDone], segBuf,
                        {1, static_cast<uint16_t>(segLen * sizeof(int32_t)), 0, 0});
            if (expandPPOn_) {
                ExpandSlotArm(slot);
                expandSegSeq_++;
            }
            segDone += segLen;
        }
        // 退出前排干两槽(见 ExpandSlotDrainAll 注释): 消费尾段在途 set, 不向
        // 跨行/跨调用残留硬件置位; 后续 V 写经排干 WaitFlag 在 V 队列自然门控
        if (expandPPOn_) {
            ExpandSlotDrainAll();
        }
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    } else {
        LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
        LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
        Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
        outQueue_.EnQue<float>(valueULocal);
        valueULocal = outQueue_.DeQue<float>();
        PkiServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }

    if (constInfo_.returnValue) {
        uint32_t negInf = constInfo_.INVALID_VAL;
        LocalTensor<uint32_t> valueULocal = outQueue_.AllocTensor<uint32_t>();
        Duplicate(valueULocal, negInf, constInfo_.sparseCount);
        outQueue_.EnQue<uint32_t>(valueULocal);
        valueULocal = outQueue_.DeQue<uint32_t>();
        GlobalTensor<uint32_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t*)valueOutGm.GetPhyAddr());
        // invalidS1offset 是 indices 行偏移(行宽 outputLen_/sparseCount);
        // value 行宽为 sparseCount, 需换算行号后重算偏移, 否则越界写且本行 value 漏写
        uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
        uint64_t valueOffset = (invalidS1offset / idxStride) * constInfo_.sparseCount;
        PkiServiceVec::CopyOut(valueOutGmTmp[valueOffset], valueULocal, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ProcessVec(const PkiCommon::RunInfo& info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 512*512 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    // (B,S1,N1,1);(T,N1,1) -> (B,S1,N2,G,1) 当前只切分到S1轴
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // 每块一次预取本 AIV 全部行的 weights(G 维 GM 连续, kHeadNum_=1), 消除
    // CopyIn 每 outerG 迭代的 32B 微拷贝。块首 PipeBarrier<PIPE_V>(上方, V 全
    // 排干)保证覆写对上一块 DoScale 读的 WAR 安全; 到达性由迭代 0 的
    // SetFlag/WaitFlag<MTE2_V>(pingpong) 覆盖(MTE2 队列按序, 预取在 mm1 拷贝之前入队)
    // int8(G2): q_descale 与 weights **GM 同偏移同 pattern**(见 CopyInInt8
    // 的两条 DataCopyPad 共用 weights_gmoffset), 故复用同一 cpW 参数一次整块搬入
    // qScaleRowBuf_(布局/行距与 weightsRowBuf_ 同构 ⇒ 消费侧同一 preOff 寻址)。
    // 同步语义与 weights 完全一致: 同一 MTE2 队列按序入队, 到达性同由迭代 0 的
    // SetFlag/WaitFlag<MTE2_V>(pingpong) 覆盖, WAR 同由块首 PipeBarrier<PIPE_V> 覆盖
    // (预取区在本块内全程只读 —— wq 落 wqRowBuf_ 独立副本, 不回写预取区)。
    if (wRowCopyOn_) {
        uint32_t wRowBytes = PkiCommon::Align(static_cast<uint64_t>(gSize_) * sizeof(W_T), (uint64_t)32);
        uint32_t wRowNum = static_cast<uint32_t>(cuS1ProcNumPerAiv);
        if ((gSize_ * sizeof(W_T)) % 32 == 0) {
            // 行宽天然 32B 对齐: 行区连续, 单次拷贝全部行
            if (wRowNum > 0) {
                AscendC::DataCopyPadExtParams<W_T> padWParams{false, 0, 0, 0};
                AscendC::DataCopyExtParams cpW;
                cpW.blockCount = 1;
                cpW.blockLen = wRowNum * gSize_ * sizeof(W_T);
                cpW.srcStride = 0;
                cpW.dstStride = 0;
                cpW.rsv = 0;
                AscendC::DataCopyPad(weightsRowT_, weightsGm[weightGmOffset], cpW, padWParams);
                if constexpr (LIT::isInt8) {
                    AscendC::DataCopyPad(qScaleRowT_, qScaleGm[weightGmOffset], cpW, padWParams);
                }
            }
        } else {
            // 行宽非 32B 对齐(如 gSize=8/bf16): 逐行拷入 32B 对齐行槽, 保持
            // DoScale 读侧操作数起始地址 32B 对齐
            for (uint32_t r = 0; r < wRowNum; r++) {
                AscendC::DataCopyPadExtParams<W_T> padWParams{false, 0, 0, 0};
                AscendC::DataCopyExtParams cpW;
                cpW.blockCount = 1;
                cpW.blockLen = gSize_ * sizeof(W_T);
                cpW.srcStride = 0;
                cpW.dstStride = 0;
                cpW.rsv = 0;
                AscendC::DataCopyPad(weightsRowT_[r * (wRowBytes / sizeof(W_T))],
                                     weightsGm[weightGmOffset + static_cast<int64_t>(r) * gSize_], cpW, padWParams);
                if constexpr (LIT::isInt8) {
                    AscendC::DataCopyPad(qScaleRowT_[r * (wRowBytes / sizeof(W_T))],
                                         qScaleGm[weightGmOffset + static_cast<int64_t>(r) * gSize_], cpW, padWParams);
                }
            }
        }
    }

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 主路径(actS1Size>4, sortPPOn_ 已含 !over2K)走归并乒乓, 累加器位置
    // per-row parity 跟踪(读点经 GetAccRow)
    bool accPPRow = sortPPOn_ && (info.actS1Size > 4);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        // (乒乓开启时按 accRowStride_ 连续铺满, (v,i) 交织模式周期 2 保各行相位; 见 InitBuffers)
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * accRowStride_);
        blockS2StartIdx_ = 0;
        accPPBits_ = 0; // 新 gS1 组从累加器行起算
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    int32_t cuRealAcSeqCount = 0;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + cuS1BeginIdxPerAiv;
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    int32_t cuRealAcSeqIni = cuRealAcSeq;
#if PKI_KDESCALE_HOIST
    // int8 k_descale 块级提升: 同一 base block 内 GetKeyScaleHalf 的三个
    // 数据实参 bIdx / cuBaseS2Idx(=info.s2Idx*s2BaseSize_) 与行号无关, cuS2Len 随行
    // 单调不减(attenMaskFlag 下 cuRealAcSeq=(count+Ini)/poolSize, count 每行 +1;
    // 非 causal 下恒定) => 按**末行** cuS2LenMax 抓一次即可覆盖全部行, 行内只保留
    // Mul(仍按各行自己的 cuS2Len 消费, [cuS2Len_row, cuS2LenMax) 不被任何行读取)。
    // 提升的是搬运位置与执行顺序, 不改任何数值 ⇒ 与逐行路径 bit-exact。
    // 契约(违反即错值, 见 optimization-analysis/plans/plan-01.md):
    //   ① cuS2LenMax 取末行值(取首行会读不足 → 尾块长行读到上一 S2 块残留);
    //   ② 门控 !isSparseCountOver2K(over2K 的 MergeSort tmp 落 tmpUb_ 头部, 尾端
    //      与 int8Tail 零间距相邻, 提升后跨行存活会被踩);
    //   ③ 每块必刷, 不引入跨块缓存(kScaleHoisted 为函数内局部量, 每次 ProcessVec
    //      重新求值, 结构上不可能跨 base block 复用);
    //   ④ SetWaitFlag<MTE2_V> 保持无 id 全排干语义不变(本改动不削同步)。
    // 提升点位于块首 PipeBarrier<PIPE_V> 之后 ⇒ 上一块行内 Mul 对 kScaleFp32Ub 的
    // 读已完成(WAR 安全); int8Tail(tmpUb_ 尾部专属区)与行循环的 dbTmpUb 双缓冲区
    // [0, (groupInner*s2Base+s2Base)*2) 不重叠 ⇒ 跨行常驻安全, UB 零追加。
    bool kScaleHoisted = false;
    if constexpr (LIT::isInt8) {
        if (!constInfo_.isSparseCountOver2K && cuS1ProcNumPerAiv > 0) {
            // 末行 cuRealAcSeq: 行循环内 count 从 1 递增到 cuS1ProcNumPerAiv
            int32_t cuRealAcSeqLast = cuRealAcSeqIni;
            if (constInfo_.attenMaskFlag) {
                cuRealAcSeqLast = (cuS1ProcNumPerAiv + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.poolSize);
            }
            int32_t cuS2LenMax =
                cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeqLast ? cuRealAcSeqLast - cuBaseS2Idx : s2BaseSize_;
            if (cuS2LenMax > 0) {
                LocalTensor<float> int8TailH = tmpUb_[(groupInner_ * s2BaseSize_ + s2BaseSize_) * 2];
                LocalTensor<half> kScaleHalfUbH = int8TailH.template ReinterpretCast<half>();
                LocalTensor<float> kScaleFp32UbH = int8TailH[s2BaseSize_ / 2];
                GetKeyScaleHalf(info, kScaleHalfUbH, info.bIdx, cuBaseS2Idx, cuS2LenMax);
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                AscendC::Cast(kScaleFp32UbH, kScaleHalfUbH, RoundMode::CAST_NONE, cuS2LenMax);
                AscendC::PipeBarrier<PIPE_V>(); // Cast(V 写) -> 行内 Mul(V 读) 的 RAW 屏障(原逐行路径同点位)
                kScaleHoisted = true;
            }
        }
    }
#endif
    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeqCount += 1;
            cuRealAcSeq = (cuRealAcSeqCount + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.poolSize);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            // UB 行距补齐数(32B 单位): 非量化 float 行距 s2BaseSize*4B;
            // int8 score 为 fp16, 搬入 float 计算区行后半, half 行距 2*s2BaseSize*2B
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            if constexpr (LIT::isInt8) {
                mmUbStride = (2 * s2BaseSize_ - info.actualSingleProcessSInnerSizeAlign) / (B32_BLOCK_ALIGN_NUM * 2);
            }
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            // 行首无需防御性 drain: 上一行所有退出路径的阻塞式等待已排干 V
            // (主输出段尾 VToMTE3Sync+SetWaitFlag / sort 区尾 barrier /
            // CleanInvalidOutput 段尾 VToMTE3Sync), 跨行冲突由保留点或既有
            // flag 协议覆盖; int8 分支出口(DoScaleInt8/dk 后乘尾 barrier)同理
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
            if (constInfo_.isSparseCountOver2K) {
                WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;

                int32_t pingpong = outerGidx % 2;
                LocalTensor<float> dbTmpUb = tmpUb_[pingpong * (groupInner_ * s2BaseSize_ + s2BaseSize_)];
                LocalTensor<float> weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                WaitFlag<HardEvent::V_MTE2>(pingpong);
                LocalTensor<W_T> weightsInTUb = weightsInUb.template ReinterpretCast<W_T>();
                if constexpr (!IsSameType<W_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }
                int64_t mmGmAllOffet = mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                       outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign;
                int64_t weightGmAllOffset = weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_;

                if constexpr (LIT::isInt8) {
                    // int8: score(fp16)+weights+q_descale 搬入(q_descale 与 weights 同 pattern 同偏移)
                    LocalTensor<half> qScaleTUb = weightsInUb.template ReinterpretCast<half>();
                    // G2 的四个读/写落点选择(初值 = 关门时的槽内地址):
                    //   预取路径(wRowCopyOn_=true, 仅 T4 排除 cache 区后才可能开):
                    //     weights/q_descale 源取自块级预取区 —— 行 innerS1Idx 的 32B
                    //     对齐行槽 + outerG 分片偏移, preOff 与 bf16 侧 wCastSrcUb
                    //     同式; wq(half 域预乘)落 wqRowT_ 的**独立副本**(按 outerG
                    //     分片, 单行宽 ⇒ 同块内各分片互不重叠, 跨行覆写由 DoScaleInt8
                    //     尾部 PipeBarrier<PIPE_V> 与行尾既有 barrier 保护 WAR);
                    //     fp32 wq 落 weightsCastDst_(与 bf16 的 Cast 落盘区同址同尺寸)。
                    //     预取区在本块内**全程只读** ⇒ 后续行/后续 outerG 读到的恒为
                    //     GM 原值(原地写会污染预取区)。
                    //   回退路径(wRowCopyOn_=false: G2 关 / T4 未排除 cache 区 / UB
                    //     门未过): 四落点全为槽内地址且 wqTUb==wSrcTUb ⇒ DoScaleInt8
                    //     的 Mul 退化为原地写, CopyInInt8 三条搬运保留。
                    LocalTensor<W_T> wSrcTUb = weightsInTUb;
                    LocalTensor<W_T> qSrcTUb = qScaleTUb;
                    LocalTensor<W_T> wqTUb = weightsInTUb;
                    LocalTensor<float> wqFp32Ub = weightsInUb;
                    if (wRowCopyOn_) {
                        uint32_t wRowBytes =
                            PkiCommon::Align(static_cast<uint64_t>(gSize_) * sizeof(W_T), (uint64_t)32);
                        uint32_t preOff =
                            (innerS1Idx * wRowBytes + static_cast<uint64_t>(outerGidx) * groupInner_ * sizeof(W_T)) /
                            sizeof(W_T);
                        wSrcTUb = weightsRowT_[preOff];
                        qSrcTUb = qScaleRowT_[preOff];
                        wqTUb = wqRowT_[static_cast<uint32_t>(outerGidx) * static_cast<uint32_t>(groupInner_)];
                        wqFp32Ub = weightsCastDst_;
                        // weights/q_descale 已整块预取, 本 outerG 只下发 score 一条
                        // (搬运参数与 CopyInInt8 的 score 段逐字一致 ⇒ UB 位型不变)
                        PkiServiceVec::CopyInScoreInt8(dbTmpUb, mm1ResGm, mmGmAllOffet, procGnum,
                                                       info.actualSingleProcessSInnerSizeAlign, s2BaseSize_,
                                                       mmUbStride);
                    } else {
                        PkiServiceVec::CopyInInt8(dbTmpUb, weightsInTUb, qScaleTUb, mm1ResGm, weightsGm, qScaleGm,
                                                  mmGmAllOffet, weightGmAllOffset, procGnum,
                                                  info.actualSingleProcessSInnerSizeAlign, s2BaseSize_, mmUbStride);
                    }

                    SetFlag<HardEvent::MTE2_V>(pingpong);
                    WaitFlag<HardEvent::MTE2_V>(pingpong);
                    // score: half(行后半暂存) -> float 计算区原地扩展 Cast, 逐行无行间依赖;
                    // 落点与非量化 mmOutUb 同位置, DoScaleInt8/DoReduce 原样复用
                    LocalTensor<half> mmOutHalfUb = dbTmpUb.template ReinterpretCast<half>()[s2BaseSize_];
                    for (int32_t row = 0; row < procGnum; row++) {
                        AscendC::Cast(dbTmpUb[row * s2BaseSize_], mmOutHalfUb[row * 2 * s2BaseSize_],
                                      RoundMode::CAST_NONE, s2BaseSize_);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    PkiServiceVec::DoScaleInt8(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, wqFp32Ub, wqTUb,
                                               wSrcTUb, qSrcTUb, brcBuf, procGnum, s2BaseSize_, outerGidx);
                } else {
                    if (wRowCopyOn_) {
                        // weights 已整块预取, CopyIn 只搬 mm1 结果
                        PkiServiceVec::CopyInMm1(dbTmpUb, mm1ResGm, mmGmAllOffet, procGnum,
                                                 info.actualSingleProcessSInnerSizeAlign, mmUbStride);
                    } else {
                        PkiServiceVec::CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm, mmGmAllOffet,
                                              weightGmAllOffset, procGnum, info.actualSingleProcessSInnerSizeAlign,
                                              mmUbStride);
                    }

                    SetFlag<HardEvent::MTE2_V>(pingpong);
                    WaitFlag<HardEvent::MTE2_V>(pingpong);
                    // DoScale 的 weights 读侧: 预取区行基址 32B 对齐, iter 偏移
                    // groupInner*sizeof(W_T) 亦 32B 对齐; W_T!=float 时 Cast 落盘区与
                    // 源区不重叠
                    LocalTensor<float> wScaleUb;
                    LocalTensor<W_T> wCastSrcUb;
                    if (wRowCopyOn_) {
                        uint32_t wRowBytes =
                            PkiCommon::Align(static_cast<uint64_t>(gSize_) * sizeof(W_T), (uint64_t)32);
                        wCastSrcUb = weightsRowT_[(innerS1Idx * wRowBytes +
                                                   static_cast<uint64_t>(outerGidx) * groupInner_ * sizeof(W_T)) /
                                                  sizeof(W_T)];
                        if constexpr (IsSameType<W_T, float>::value) {
                            wScaleUb = wCastSrcUb; // float: 直读预取源
                        } else {
                            wScaleUb = weightsCastDst_;
                        }
                    } else {
                        wScaleUb = weightsInUb;
                        wCastSrcUb = weightsInTUb;
                    }
                    PkiServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, wScaleUb, wCastSrcUb,
                                           brcBuf, procGnum, s2BaseSize_, outerGidx);
                }
                // confused reduceOp in DoScale
                // neednot use PkiServiceVec::doReduce(mmInUb, reduceOutInner, procGnum, (s2BaseSize_+8));
                SetFlag<HardEvent::V_MTE2>(pingpong);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            PkiServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            if constexpr (LIT::isInt8) {
                // dk 后乘: score = (Σ_g wq·S̃) × dk, 之后既有 Muls 折入 1024/√128,
                // 两次乘的舍入顺序与 spec formula 逐项对应;
                // k_descale 暂存区为 tmpBuf_ 尾部 int8 追加区(见 InitBuffers)
                LocalTensor<float> int8Tail = tmpUb_[(groupInner_ * s2BaseSize_ + s2BaseSize_) * 2];
                LocalTensor<half> kScaleHalfUb = int8Tail.template ReinterpretCast<half>();
                LocalTensor<float> kScaleFp32Ub = int8Tail[s2BaseSize_ / 2];
#if PKI_KDESCALE_HOIST
                // 块级提升未生效时(isSparseCountOver2K 门控 / 本块无有效行 /
                // 编译期开关关闭)原样走逐行路径 ⇒ 与基线路径逐字节一致
                if (!kScaleHoisted) {
#endif
                    GetKeyScaleHalf(info, kScaleHalfUb, info.bIdx, cuBaseS2Idx, cuS2Len);
                    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                    AscendC::Cast(kScaleFp32Ub, kScaleHalfUb, RoundMode::CAST_NONE, cuS2Len);
                    AscendC::PipeBarrier<PIPE_V>();
#if PKI_KDESCALE_HOIST
                }
#endif
                AscendC::Mul(reduceOutInner, reduceOutInner, kScaleFp32Ub, cuS2Len);
                AscendC::PipeBarrier<PIPE_V>();
            }

            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // Muls 读 idx 区(reduceOutInner 与 sortIndice 同区)写 score 区, Adds
            // 随后覆写 idx 区, Muls→Adds 间 WAR 依赖 barrier 保护。
            // 整块(len==align): Muls 全覆写 score 区, 免 -inf 预填充, 仅保留
            // WAR barrier。尾块(len<align): -inf/-1 填充仅覆盖 [aL, align) 尾段
            // (aL=AlignDown(len,8) 保 Duplicate dst 32B 对齐), 填充与 Muls
            // (count=len)形状互异且存在滑片 WAW/WAR, 各 barrier 保留
            if (cuS2LenVecAlign == cuS2Len) {
                // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
                // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
                // [键宽契约] int8 排序域不含 ×qkScale(1024/√d), 物理缩放到 values 输出补乘
                if constexpr (LIT::isInt8) {
                    Muls(sortScoreUb, reduceOutInner, 1.0f, cuS2Len);
                } else {
                    Muls(sortScoreUb, reduceOutInner, constInfo_.qkScale, cuS2Len);
                }
                PipeBarrier<PIPE_V>(); // WAR: Muls 读 reduceOutInner(=idx 区) 先于 Adds 覆写
                Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
                PipeBarrier<PIPE_V>();
            } else {
                int32_t alignedLen = cuS2Len & ~(static_cast<int32_t>(B32_BLOCK_ALIGN_NUM) - 1);
                Duplicate(sortScoreUb[alignedLen].template ReinterpretCast<int32_t>(), PkiServiceVec::NEG_INF,
                          cuS2LenVecAlign - alignedLen);
                PipeBarrier<PIPE_V>();
                // 池级分数乘 1/sqrt(headDim)(文档公式 S = Q@K^T/sqrt(headDim));
                // 缩放为正数, 与 ReLU/加权求和可交换, 在聚合后统一应用
                // [键宽契约] 同上: int8 排序域不含 ×qkScale
                if constexpr (LIT::isInt8) {
                    Muls(sortScoreUb, reduceOutInner, 1.0f, cuS2Len);
                } else {
                    Muls(sortScoreUb, reduceOutInner, constInfo_.qkScale, cuS2Len);
                }
                PipeBarrier<PIPE_V>(); // WAR: Muls 读 idx 区滑片先于尾填覆写
                Duplicate(sortIndiceUbInt[alignedLen], -1, cuS2LenVecAlign - alignedLen);
                PipeBarrier<PIPE_V>(); // WAW: 尾填滑片先落地再被 Adds 覆写
                Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
                PipeBarrier<PIPE_V>();
            }

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                PkiServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                       cuS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
                PipeBarrier<PIPE_V>();
                if (accPPRow) {
                    // 归并乒乓无回拷 —— src1=当前累加器(accCur), 全量输出
                    // (virTopK+cuS2LenVecAlign 对)写对侧缓冲(accNxt: 本行累加器行
                    // spill 区 或 本行的乒乓位槽, 均按行分槽), 前 virTopK 对与回拷
                    // 版截断结果一致; parity 翻转, 读点(Extract/CopyOut ws/下一轮
                    // src1)经 GetAccRow 定位
                    LocalTensor<float> accRowBase = globalTopkUb_[innerS1Idx * accRowStride_];
                    LocalTensor<float> accTmpSlot = sortPPLocal_[innerS1Idx * accRowStride_];
                    bool accInTmp = ((accPPBits_ >> innerS1Idx) & 1u) != 0;
                    LocalTensor<float> accCur = accInTmp ? accTmpSlot : accRowBase;
                    LocalTensor<float> accNxt = accInTmp ? accRowBase : accTmpSlot;
                    PkiServiceVec::MergeSortPP(accCur, accNxt, virTopK, reduceOutBuff, cuS2LenVecAlign);
                    accPPBits_ ^= static_cast<uint8_t>(1u << innerS1Idx);
                } else {
                    LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                    PkiServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                             cuS2LenVecAlign, UbTmpSort);
                }
            } else {
                // 缓存路径: 累加器行步长 virTopK*2, 与主路径/Extract/LD 写出统一
                // 口径; 缓存行步长 = 4 槽 x s2BaseSize_ 对(4 块归并节奏不变)
                int64_t cacheRowFloats = static_cast<int64_t>(4) * s2BaseSize_ * 2;
                LocalTensor<float> accRow = globalTopkUb_[innerS1Idx * virTopK * 2];
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
                Sort<float, true>(
                    SortedBasicBlock_[innerS1Idx * cacheRowFloats + globalTopkUbCacheIdx * s2BaseSize_ * 2],
                    reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                    cuS2LenVecAlign / 32);
                AscendC::PipeBarrier<PIPE_V>();
                // 缓存4块512或者S2结束, 需要进行精排
                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * cacheRowFloats];
                    // 前4块: 全量归并经 tmpSortBuf 中转, 截断 top-virTopK 写入累加器行
                    // (与 else 分支 MrgBasicBlock(tmpSortBuf,...) 同模式; 流式 top-K
                    // 语义等价, 行尾不足部分保持 InitSortOutBuf 的 -inf/-1 初值)
                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                        PipeBarrier<PIPE_V>();
                        int64_t mergedPairs = (globalTopkUbCacheIdx + 1) * s2BaseSize_;
                        DataCopy(accRow, tmpSortBuf, PkiCommon::Min(static_cast<int64_t>(virTopK), mergedPairs) * 2);
                        PipeBarrier<PIPE_V>();
                    } else {
                        if (globalTopkUbCacheIdx > 0) {
                            MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                            PipeBarrier<PIPE_V>();
                            DataCopy(SortedBasicBlock_[innerS1Idx * cacheRowFloats], tmpSortBuf,
                                     (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                        }
                        PipeBarrier<PIPE_V>();
                        SparseTopK(accRow, SortedBasicBlock_[innerS1Idx * cacheRowFloats], tmpSortBuf, virTopK,
                                   s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                    }
                }
            }
            if (constInfo_.isSparseCountOver2K) {
                SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;

            // 中间结果保存
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;

            if (needCopyOutGm) {
                int64_t offset = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? virTopK : constInfo_.sparseCount / 2;
                int64_t copyLen =
                    (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? constInfo_.sparseCount : constInfo_.sparseCount / 2;
                int64_t copyNum = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? 1 : 2;
                if (poolSize_ > 1) {
                    uint32_t alignedPoolSize = PkiCommon::Align(poolSize_, (uint32_t)8);
                    for (int64_t i = 0; i < copyNum; i++) {
                        LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                        LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                        Extract(outValueUb, outIdxUb, GetAccRow(innerS1Idx, accPPRow)[2 * i * offset], (offset / 32));

                        // [键宽契约] int8 排序域不含 ×qkScale, values 在键反解(Extract)
                        // 后补乘物理缩放(仅 value 区 [0,copyLen), idx 区不动); -inf×正常数=-inf。
                        if constexpr (LIT::isInt8) {
                            PipeBarrier<PIPE_V>();
                            Muls(outValueUb, outValueUb, constInfo_.qkScale, static_cast<int32_t>(copyLen));
                            PipeBarrier<PIPE_V>();
                        }
                        PipeBarrier<PIPE_V>();
                        LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                        DataCopy(workLocal_[alignedPoolSize + i * copyLen], idxULocal1,
                                 PkiCommon::Align(copyLen, (int64_t)8));

                        if (constInfo_.returnValue) {
                            outQueue_.EnQue<float>(outValueUb);
                            outValueUb = outQueue_.DeQue<float>();
                            PkiServiceVec::CopyOut(
                                valueOutGm[info.valueOutOffset +
                                           static_cast<int64_t>(cuS1Idx) * constInfo_.sparseCount + i * offset],
                                outValueUb, copyLen);
                        }
                        outQueue_.FreeTensor(outValueUb);
                    }
                    PipeBarrier<PIPE_V>();
                    // MTE3→V: 等上一行 DataCopyPad 读完 expandOutLocal_ 再覆写
                    // (行尾 SetWaitFlag 仅登记不等待, 此处显式等待)
                    event_t evtMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                    SetFlag<HardEvent::MTE3_V>(evtMte3V);
                    WaitFlag<HardEvent::MTE3_V>(evtMte3V);
                    // cuRealAcSeq 为有效池数上界, 不可回退 sparseCount(实际池数
                    // 不足时 TopK 尾部 -1 标记会展开成负索引残留)
                    uint32_t validS2Len = static_cast<uint32_t>(cuRealAcSeq);
                    // 组行数 <=4(与上方主/缓存路径分界同界)时每 AIV 至多 2 行,
                    // 行展开后已无足够后续行/块的向量工作吸收槽事件机自旋(末端
                    // 展开, 如单 token decode); 行数多的组展开后仍有本组后续行与
                    // 后续基本块在途, 维持乒乓解耦
                    ExpandAndAppendIndices(workLocal_[alignedPoolSize], expandOutLocal_, workLocal_, indiceOutGm,
                                           info.indiceOutOffset + static_cast<int64_t>(cuS1Idx) * outputLen_,
                                           constInfo_.sparseCount, poolSize_, validS2Len, info.poolTailK,
                                           static_cast<int32_t>(info.actS2SizeOrig), static_cast<uint32_t>(cuS1Idx),
                                           info.actS1Size, expandPPOn_ && (info.actS1Size > 4));
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                } else {
                    for (int64_t i = 0; i < copyNum; i++) {
                        LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                        LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                        Extract(outValueUb, outIdxUb, GetAccRow(innerS1Idx, accPPRow)[2 * i * offset], (offset / 32));

                        // [键宽契约] int8 values 补乘 ×qkScale, 同 ps>1 路径(仅 value 区)
                        if constexpr (LIT::isInt8) {
                            PipeBarrier<PIPE_V>();
                            Muls(outValueUb, outValueUb, constInfo_.qkScale, static_cast<int32_t>(copyLen));
                            PipeBarrier<PIPE_V>();
                        }

                        if (constInfo_.returnValue) {
                            PipeBarrier<PIPE_V>();
                        }

                        LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                        outQueue_.EnQue<float>(outValueUb);
                        outValueUb = outQueue_.DeQue<float>();

                        PkiServiceVec::CopyOut(
                            indiceOutGm[info.indiceOutOffset + static_cast<int64_t>(cuS1Idx) * constInfo_.sparseCount +
                                        i * offset],
                            idxULocal1, copyLen);
                        if (constInfo_.returnValue) {
                            PkiServiceVec::CopyOut(
                                valueOutGm[info.valueOutOffset +
                                           static_cast<int64_t>(cuS1Idx) * constInfo_.sparseCount + i * offset],
                                outValueUb, copyLen);
                        }
                        outQueue_.FreeTensor(outValueUb);
                    }
                }
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, virTopK] float32
                // (传输宽度与累加器容量一致, ProcessLD 读侧同口径; host 侧 LD 区
                // 按 topk 上限 8192 预留)
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64(LD 参数)

                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * virTopK + // 2个AIV共同地址偏移
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * virTopK + // 每个AIV的地址偏移，S1方向
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * virTopK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ + // 2个AIV共同地址偏移
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ + // 每个AIV的地址偏移，S1方向
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * idxStride));
                tmpiBuff.SetValue(9, static_cast<int64_t>(info.poolTailK));
                tmpiBuff.SetValue(10, static_cast<int64_t>(info.actS2SizeOrig));
                tmpiBuff.SetValue(11, static_cast<int64_t>(info.actS1Size));
                // slot 12: values 输出偏移(行宽 sparseCount, 与 slot 8 的 indices
                // 偏移行宽不同, poolSize>1 时不可复用 slot 8)
                tmpiBuff.SetValue(12, static_cast<int64_t>(info.valueOutOffset +
                                                           static_cast<int64_t>(cuS1Idx) * constInfo_.sparseCount));
                // 写入头尾判断: head 与前面块规约, tail 与后面块规约
                bool isTailReduce = blockS2StartIdx_ == 0; // 一定是isLastTile
                // WS偏移规则: 与前面块规约写 0 偏移; 与后面块规约写 1 偏移
                if (isTailReduce) { // S2不是最后结束的数据就需要往后做规约，放入第二块ws
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * virTopK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                PkiServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                PkiServiceVec::CopyOut(vec1ResGm[wsOffset], GetAccRow(innerS1Idx, accPPRow), 2 * virTopK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * idxStride);
        }
    }

    // BNSD场景无效S1 输出-1
    if (LAYOUT_T == PkiLayout::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * idxStride);
            }
        }

        int32_t invalidS1Num2 =
            static_cast<int32_t>(info.actS1Size - info.actS2SizeOrig) / static_cast<int32_t>(constInfo_.poolSize);
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) * idxStride);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq = 0;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t bn2Idx;
    int64_t s1Idx;
    int64_t poolTailKLd = 0;
    int64_t L_origLd = 0;
    int64_t actS1SizeLd = 0;
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;
    int64_t valueOutOffsetLd = 0; // slot 12: values 输出偏移(行宽 sparseCount)

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息: 从尾规约开始, while 读取后续核头规约, 存满 4 个 list 或
    // S2 结尾则 merge; 每个核忽略自己的头规约(由前面核完成)
    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    // S1逐行计算
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 重置偏移
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        // 搬入数据
        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * virTopK + // 2个AIV共同地址偏移
                   innerS1Idx * 2 * 2 * virTopK + 2 * virTopK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset], {1, static_cast<uint16_t>(2 * virTopK * sizeof(int32_t)), 0, 0},
                    {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * virTopK;

        // 获取下一个核规约信息
        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);
        s2ActSeq = vec1ParamGm.GetValue(wsInfoOffset + 1);
        poolTailKLd = vec1ParamGm.GetValue(wsInfoOffset + 9);
        L_origLd = vec1ParamGm.GetValue(wsInfoOffset + 10);
        actS1SizeLd = vec1ParamGm.GetValue(wsInfoOffset + 11);
        valueOutOffsetLd = vec1ParamGm.GetValue(wsInfoOffset + 12);

        while (needFd == 1) {
            // 搬入头规约数据
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * virTopK + // 2个AIV共同地址偏移
                       innerS1Idx * 2 * 2 * virTopK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * virTopK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * virTopK;
            acc_list_num++;

            // 每满4个list，聚合  前 virTopK 为 mrg 结果
            if (acc_list_num == mrgListNum_) {
                // MrgSort 四条virTopK的队列，Mrg成一条
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = virTopK;
                params.elementLengths[1] = virTopK;
                params.elementLengths[2] = virTopK;
                params.elementLengths[3] = virTopK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * virTopK];
                srcList.src3 = curValueIdxUb[4 * virTopK];
                srcList.src4 = curValueIdxUb[6 * virTopK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * virTopK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * virTopK;
            }

            // reduce到S2末尾，则跳出
            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        }

        // mrg不足4个list的数据: 凑满 4 表, 空槽位填 -inf/-1 空表(InitSortOutBuf
        // 交错填充 NEG_INF/INVALID_INDEX), 恒 validBit=0b1111。原部分归并
        // (validBit=0b0011/0b0111)的 src3/src4 指向未初始化/残留 UB, 属明确
        // 潜在隐患(910B3/cann-9.3.0 实测未复现); 空表 -inf 沉底被归并截断,
        // 结果与部分归并语义等价。
        // UB 容量: ldToBeMrgBuf_ = 2*ldTopkW*mrgListNum_(活域 ldTopkW=virTopK,
        // 即 8*virTopK float), 槽位 [4*virTopK, 8*virTopK) 为本核链尾必空的
        // 1~2 槽, 填充不出界。
        if (acc_list_num != 1) {
            if (acc_list_num == 2) {
                InitSortOutBuf(curValueIdxUb[4 * virTopK], 2 * virTopK);
                InitSortOutBuf(curValueIdxUb[6 * virTopK], 2 * virTopK);
            } else {
                InitSortOutBuf(curValueIdxUb[6 * virTopK], 2 * virTopK);
            }
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = virTopK;
            params.elementLengths[1] = virTopK;
            params.elementLengths[2] = virTopK;
            params.elementLengths[3] = virTopK;
            params.ifExhaustedSuspension = true;
            params.validBit = 0b1111;
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * virTopK];
            srcList.src3 = curValueIdxUb[4 * virTopK];
            srcList.src4 = curValueIdxUb[6 * virTopK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * virTopK);
            PipeBarrier<PIPE_V>();
        }

        // 搬出
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        uint16_t idxOutBytes =
            static_cast<uint16_t>((poolSize_ > 1 ? outputLen_ : constInfo_.sparseCount) * sizeof(int32_t));
        if (!constInfo_.returnValue) {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (virTopK / 32));
            if (poolSize_ > 1) {
                PipeBarrier<PIPE_V>();
                // MTE3→V: 等上一行 DataCopyPad 读完 expandOutLocal_ 再覆写
                // (与 ProcessVec 主输出路径相同)
                event_t evtMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(evtMte3V);
                WaitFlag<HardEvent::MTE3_V>(evtMte3V);
                // s2ActSeq 为有效池数上界, 不可回退 sparseCount
                // (否则 -1 无效池标记展开成负索引残留)
                uint32_t validS2Len = static_cast<uint32_t>(s2ActSeq);
                // LD 归并是收尾阶段, 行数少(<=4)时展开后无后续工作吸收自旋,
                // 与 ProcessVec 直出路径同界回退单槽
                ExpandAndAppendIndices(outIdxUb.template ReinterpretCast<int32_t>(), expandOutLocal_, workLocal_,
                                       indiceOutGm, outOffset, constInfo_.sparseCount, poolSize_, validS2Len,
                                       static_cast<int32_t>(poolTailKLd), static_cast<int32_t>(L_origLd),
                                       static_cast<uint32_t>(s1Idx), static_cast<uint32_t>(actS1SizeLd),
                                       expandPPOn_ && (actS1SizeLd > 4));
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(indiceOutGm[outOffset], idxULocal1, {1, idxOutBytes, 0, 0});
            }
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        } else {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (virTopK / 32));
            PipeBarrier<PIPE_V>();
            if (poolSize_ > 1) {
                // MTE3→V: 等上一行 DataCopyPad 读完 expandOutLocal_ 再覆写
                // (与 ProcessVec 主输出路径相同)
                event_t evtMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(evtMte3V);
                WaitFlag<HardEvent::MTE3_V>(evtMte3V);
                uint32_t validS2Len = static_cast<uint32_t>(s2ActSeq);
                ExpandAndAppendIndices(outIdxUb.template ReinterpretCast<int32_t>(), expandOutLocal_, workLocal_,
                                       indiceOutGm, outOffset, constInfo_.sparseCount, poolSize_, validS2Len,
                                       static_cast<int32_t>(poolTailKLd), static_cast<int32_t>(L_origLd),
                                       static_cast<uint32_t>(s1Idx), static_cast<uint32_t>(actS1SizeLd),
                                       expandPPOn_ && (actS1SizeLd > 4));
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            if (poolSize_ > 1) {
                // ExpandAndAppendIndices 已内部分段写 GM, 此处无需再写 indices
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            } else {
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(indiceOutGm[outOffset], idxULocal1, {1, idxOutBytes, 0, 0});
            }
            // [键宽契约] int8 LD 路径 values 补乘 ×qkScale, 与主路径同口径(value/idx 分离缓冲)
            if constexpr (LIT::isInt8) {
                if (constInfo_.returnValue) {
                    PipeBarrier<PIPE_V>();
                    Muls(outValueUb, outValueUb, constInfo_.qkScale, static_cast<int32_t>(constInfo_.sparseCount));
                    PipeBarrier<PIPE_V>();
                }
            }
            DataCopyPad(valueOutGm[valueOutOffsetLd], outValueUb,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(float)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}
} // namespace PkiKernel
#endif
