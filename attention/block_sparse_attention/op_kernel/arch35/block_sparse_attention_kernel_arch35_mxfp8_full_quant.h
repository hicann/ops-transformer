/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../arch35/kernel_utils.hpp"
#include "../attn_infra/epilogue/block/block_epilogue_arch35_utils.hpp"
#include "../attn_infra/gemm/block/block_mmad_arch35_utils.hpp"

using namespace NpuArch;
using namespace tla;
using namespace MXFP8Kernel;

namespace EpMXFP8 = Epilogue::Block::MXFP8; // UB 常量别名
namespace GmMXFP8 = Gemm::Block::MXFP8;     // L1 常量别名

namespace BsaKernelArch35 {

template <class EpilogueMask2Idx, class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV,
          class EpilogueRescaleO, class EpilogueComputePScale, class EpilogueCopyGlobalMaxUbToL1,
          class BlockMmadCopyGlobalMaxL1ToUB, Format qFormat, Format kvFormat, bool transposedMm1 = false>
class BsaMXFP8FullQuantKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementDisguiseP = typename EpilogueOnlineSoftmax::ElementDisguiseP;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using ElementSparseMask = typename EpilogueMask2Idx::ElementSparseMask;
    using ElementSparseIdx = typename EpilogueMask2Idx::ElementSparseIdx;
    using ElementSparseCount = typename EpilogueMask2Idx::ElementSparseCount;
    using ElementGroupMax = typename EpilogueComputePScale::ElementGroupMax;
    using ElementPScale = typename EpilogueComputePScale::ElementPScale;
    using ElementLocalGlobalMax = typename EpilogueCopyGlobalMaxUbToL1::ElementLocalGlobalMax;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPL1 = typename EpilogueOnlineSoftmax::LayoutPL1;
    using LayoutV = layout::ColumnMajor;
    using LayoutO = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLocalGlobalMax = typename EpilogueCopyGlobalMaxUbToL1::LayoutLocalGlobalMax;
    using LayoutPScale = typename EpilogueComputePScale::LayoutPScale;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    // 核间同步
    static constexpr uint8_t SYNC_MODE_2 = 2;
    static constexpr uint8_t SYNC_MODE_4 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {9, 10};
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_UB_BACK = 8;
    // 热路径用 BUF0 + tileMaxIdx/2（或 pscaleNum），连续占用 BUF0 与 BUF0+1。
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 = 6;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_BUF0 = 2;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 1;
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 0;

    static constexpr uint32_t PV_RE_DELAY_N = 20; // G=16 → Dpv>=19，取 20
    static constexpr uint32_t DELAY_PSCALE_N = 3;

    // 32 盖住 PV_RE_DELAY_N+1，下标 & mask。TaskInfo/TileInfo 无 NSDMI，声明不带 ={}，入口不整表清零。
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 32;
    static_assert((PRELOAD_TASK_CACHE_SIZE & (PRELOAD_TASK_CACHE_SIZE - 1u)) == 0u, "power of two");
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PV_RE_DELAY_N + 1u, "must hold delay window");
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1u;

    // Methods
    __aicore__ inline BsaMXFP8FullQuantKernelArch35() {}

    __aicore__ inline void operator()(BsaFullQuantKernelParamsArch35 const& params)
    {
        __gm__ BlockSparseAttentionTilingData* bsaTilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionTilingData*>(params.tiling);
        FetchBaseShapeInfo(bsaTilingData);
        // KV(V 数据) 与 V-scale 的 stride 预算（PV 稀疏 gather 用）
        // V 数据 e4m3：sizeof=1，元素偏移即字节偏移，行 stride = embed_
        // V-scale E8M0：GM [B,N2,S2//64,D,2]，每 64-S2 行 group 占 D*2 字节
        // O / Lse stride & batch offsets（同 BF16，在 while 循环中创建 task 时传入）
        InitLayoutStride();
        // global buffers
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.query);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.key);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.value);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t*)params.actualSeqLengths);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t*)params.actualSeqLengthsKv);
        AscendC::GlobalTensor<uint8_t> gBlockSparseMask;
        gBlockSparseMask.SetGlobalBuffer((__gm__ uint8_t*)params.blockSparseMask);
        AscendC::GlobalTensor<uint8_t> gQDequantScale;
        gQDequantScale.SetGlobalBuffer((__gm__ uint8_t*)params.qDequantScale);
        AscendC::GlobalTensor<uint8_t> gKDequantScale;
        gKDequantScale.SetGlobalBuffer((__gm__ uint8_t*)params.kDequantScale);
        AscendC::GlobalTensor<uint8_t> gVDequantScale;
        gVDequantScale.SetGlobalBuffer((__gm__ uint8_t*)params.vDequantScale);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.attentionOut);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.lse);
        AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx;
        gSparseIdx.SetGlobalBuffer((__gm__ ElementSparseIdx*)params.workSpace);
        AscendC::GlobalTensor<ElementSparseCount> gSparseCount;
        gSparseCount.SetGlobalBuffer((__gm__ ElementSparseCount*)(params.workSpace + sparseIdxSize_));

        // cross core data move dst buffers
        AscendC::LocalTensor<ElementS> ubSTensor[EpMXFP8::UB_S_BUF_CNT];
        AscendC::LocalTensor<ElementDisguiseP> l1PTensor[GmMXFP8::L1_P_BUF_CNT];
        AscendC::LocalTensor<ElementPScale> l1PScaleTensor[GmMXFP8::L1_P_SCALE_BUF_CNT];
        AscendC::LocalTensor<ElementS> l1LocalGlobalMaxTensor[GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_CNT];
        AscendC::LocalTensor<ElementS> ubPeerGlobalMaxTensor[EpMXFP8::UB_PEER_GLOBAL_MAX_CNT];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor;

        InitCrossCoreDstBuf(l1PTensor, l1PScaleTensor, l1LocalGlobalMaxTensor, ubSTensor, ubPeerGlobalMaxTensor,
                            ubOTmpTensor);
        // core idx
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // set reverse sync flags
        InitSyncFlags();
        // mask preprocess
        if ASCEND_IS_AIV {
            EpilogueMask2Idx epilogueMask2Idx(resource);
            uint32_t totalRowNumBlockMask = batch_ * qHeads_ * xBlockNumAligned_;
            epilogueMask2Idx(gBlockSparseMask, gSparseIdx, gSparseCount, totalRowNumBlockMask, yBlockNumAligned_,
                             avgRowPerSubCore_, preActiveSubCoreNum_);
        }
        AscendC::SyncAll<false>();

        // tla define
        auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(EpMXFP8::KVS_BASE_SIZE, EpMXFP8::QS_BASE_SIZE * 2);
        auto ubSTensorTla = tla::MakeTensor(ubSTensor[0], ubSLayoutTla, Arch::PositionUB{});

        auto l1PScaleLayoutTla =
            tla::MakeLayout<ElementPScale, LayoutPScale>(GmMXFP8::L1_P_SCALE_BUF_CNT, GmMXFP8::L1_P_SCALE_BUF_SIZE);
        auto l1PScaleTensorTla = tla::MakeTensor(l1PScaleTensor[0], l1PScaleLayoutTla, Arch::PositionL1{});

#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        // [QK/PV 共享] kernel 层单一 kvBufId，传引用进两 block ctor（QK 先构造预置 4 路 KV event；
        //   PV 后构造、其 ctor 内 InitL0BufferForReduceSum 把 kvBufId 0→1）
        uint32_t kvBufId = 0;
        uint64_t softmaxScale = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scaleValue_));
        BlockMmadQK blockMmadQK(resource, kvBufId, softmaxScale);
        BlockMmadPV blockMmadPV(resource, kvBufId);
#endif
        // 两边都构造，AIV 的 if constexpr(AIC) 被丢弃时仍能解析到这个名字。拷贝只在 Cube 函数体内发出。
        BlockMmadCopyGlobalMaxL1ToUB blockGMaxL1ToUb(resource);
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource);
        EpilogueCopyGlobalMaxUbToL1 epilogueCopyGlobalMaxUbToL1(resource);
        EpilogueComputePScale epilogueComputePScale(resource);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif

        // runMainLoop
        uint64_t coreTileLoop = 0;

        uint64_t curTaskTileIdx = 0;
        uint64_t delay20TaskTileIdx = 0;
        uint64_t delay3TaskTileIdx = 0;

        uint32_t coreTaskId = 0;
        uint32_t delay20CoreTaskId = 0; // PV / O，晚 PV_RE_DELAY_N 个 tile
        uint32_t delay3CoreTaskId = 0;  // pscale，晚 DELAY_PSCALE_N 个 tile

        uint64_t preTaskTileSum = 0;
        uint64_t delay20preTaskTileSum = 0;
        uint64_t delay3preTaskTileSum = 0;

        uint64_t coreTaskNum = coreTaskNum_;
        coreTaskNum = (coreIdx < totalTaskNum_ % coreNum) ? (coreTaskNum + 1) : coreTaskNum;

        // 无 NSDMI、无 ={}：平凡类型默认初始化不写零。delay 窗先写后读。
        TaskInfo taskInfos[PRELOAD_TASK_CACHE_SIZE];
        TileInfo tileInfos[PRELOAD_TASK_CACHE_SIZE];

        // CreateTaskInfo 跨调用 batch 扫描状态：curTotalTaskNum 从首 batch task 数起步，
        // 后续调用在 while 循环内基于该状态续算，不再从 batch 0 重扫
        BatchOffsetInfo batchOffsetInfo;
        batchOffsetInfo.curTotalTaskNum = firstBatchTaskNum_;
        // BSND/TND 头内 token 间隔 N*D。gather layout 的行/列 extent 乘 kvHeads_/qHeads_，
        // 列步长、行步长、group 步长才对。BNSD 头内连续，因子为 1。
        // extent 大于真实长度不影响 GetTile（按真实 extent 取子块）。
        const uint32_t kvHeadMul = (kvFormat == Format::BNSD) ? 1u : kvHeads_;
        const uint32_t qHeadMul = (qFormat == Format::BNSD) ? 1u : qHeads_;

        while (coreTaskId < coreTaskNum || delay20CoreTaskId < coreTaskNum || delay3CoreTaskId < coreTaskNum) {
            curTaskTileIdx = coreTileLoop - preTaskTileSum;
            delay20TaskTileIdx = coreTileLoop - PV_RE_DELAY_N - delay20preTaskTileSum;
            delay3TaskTileIdx = coreTileLoop - DELAY_PSCALE_N - delay3preTaskTileSum;

            // 空 KV 的 taskTileNum 为 0，这两级靠最后一块 tile 推进 task 号会下溢卡住。
            // 只跳过已经 CreateTaskInfo 的空 task，未写入的槽不读。
            SkipEmptyCreatedTasks(delay20CoreTaskId, coreTaskId, taskInfos);
            SkipEmptyCreatedTasks(delay3CoreTaskId, coreTaskId, taskInfos);

            TaskInfo& curTaskInfo = taskInfos[coreTaskId & PRELOAD_TASK_CACHE_MASK];
            TaskInfo& delay20TaskInfo = taskInfos[delay20CoreTaskId & PRELOAD_TASK_CACHE_MASK];
            TaskInfo& delay3TaskInfo = taskInfos[delay3CoreTaskId & PRELOAD_TASK_CACHE_MASK];

            TileInfo& curTileInfo = tileInfos[coreTileLoop & PRELOAD_TASK_CACHE_MASK];
            TileInfo& delay20TileInfo = tileInfos[(coreTileLoop - PV_RE_DELAY_N) & PRELOAD_TASK_CACHE_MASK];
            TileInfo& delay3TileInfo = tileInfos[(coreTileLoop - DELAY_PSCALE_N) & PRELOAD_TASK_CACHE_MASK];

            // 创建qs任务
            if (curTaskTileIdx == 0 && coreTaskId < coreTaskNum) {
                CreateTaskInfo(curTaskInfo, coreTaskId, coreIdx, coreNum, gActualQseqlen, gActualKvseqlen, gSparseCount,
                               gSparseIdx, batchOffsetInfo);
                if (curTaskInfo.taskTileNum == 0) {
                    if ASCEND_IS_AIV {
                        WriteEmptyKvOutput(curTaskInfo, gO, gLse);
                    }
                    coreTaskId++;
                    continue;
                }
            }

            // main process
            if (coreTileLoop >= PV_RE_DELAY_N && delay20CoreTaskId < coreTaskNum) {
                if ASCEND_IS_AIC {
                    auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kvHeadMul * embed_, kvBaseTile_);
                    auto gmVTensorTla =
                        tla::MakeTensor(gV[delay20TaskInfo.gmOffsetV], gmVLayoutTla, Arch::PositionGM{});
                    auto ubOTmpLayoutTla =
                        tla::MakeLayout(tla::MakeShape(GmMXFP8::ROW_SUM_NUM + 1, delay20TaskInfo.qsActBaseTileAlign64),
                                        tla::MakeStride(static_cast<int64_t>(64), tla::Int<1>{}));
                    auto ubOTmpTensorTla = tla::MakeTensor(ubOTmpTensor, ubOTmpLayoutTla, Arch::PositionUB{});
                    GemmCoord actualBlockShapePV{delay20TaskInfo.qsActBaseTile, embed_, delay20TileInfo.kvsActBaseTile};

#ifdef __DAV_CUBE__
                    // even（含 last even 尾块）先发 V gather；Wait 必须在 if 外面。
                    const bool pvPairOdd = (delay20TileInfo.pvGatheredKvSTileIdx & 1u) != 0u;
                    if (!pvPairOdd) {
                        blockMmadPV.PrefetchV(gmVTensorTla, gVDequantScale, gSparseIdx, delay20TileInfo,
                                              delay20TaskInfo, kvBaseTile_, blockShapeY_, kvHeadMul,
                                              actualBlockShapePV);
                    }
#endif
                    if (delay20TileInfo.isTileGoupFirstTile) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_BUF0 +
                                                                           delay20TileInfo.pscaleNum);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_BUF0 +
                                                                           delay20TileInfo.pscaleNum + 16);
                    }
                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                    }

#ifdef __DAV_CUBE__
                    blockMmadPV.ComputePV(ubOTmpTensorTla, actualBlockShapePV, delay20TileInfo, delay20TaskInfo);
#endif

                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
                    }
                } else {
                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
                        // O GM：cols=embed_（每 token D 个元素），row-stride=oShapeCol（token 步长 strideOS_）。
                        // BNSD 两者都是 D。BSND/TND 的 strideOS_ 是 N*D，cols 不能再设成 oShapeCol。
                        auto gmOLayoutTla = tla::MakeLayout(tla::MakeShape(qBaseTile_, embed_),
                                                            tla::MakeStride(delay20TaskInfo.oShapeCol, tla::Int<1>{}));
                        auto gmOTensorTla =
                            tla::MakeTensor(gO[delay20TaskInfo.gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
                        GemmCoord actualBlockShapeVPT{embed_, delay20TaskInfo.qsActBaseTile,
                                                      delay20TileInfo.kvsActBaseTile};
                        if constexpr (EpilogueRescaleO::DispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY) {
                            auto gmLseLayoutTla = tla::MakeLayout<ElementLse, LayoutLse>(delay20TaskInfo.qsActBaseTile,
                                                                                         delay20TaskInfo.lseShapeCol);
                            auto gmLseTensorTla =
                                tla::MakeTensor(gLse[delay20TaskInfo.gmOffsetLse], gmLseLayoutTla, Arch::PositionGM{});
                            auto kgSnap = resource.ubBuf.template GetBufferByByte<half>(
                                EpMXFP8::UB_LSE_MAX_SNAP_OFFSET +
                                (delay20CoreTaskId & (EpMXFP8::UB_LSE_MAX_SNAP_CNT - 1u)) *
                                    EpMXFP8::UB_LSE_MAX_SNAP_SIZE);
                            epilogueRescaleO(gmOTensorTla, gmLseTensorTla, kgSnap, actualBlockShapeVPT, delay20TaskInfo,
                                             delay20TileInfo);
                        } else {
                            epilogueRescaleO(gmOTensorTla, actualBlockShapeVPT, delay20TaskInfo, delay20TileInfo);
                        }
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                    }
                }
            }
            if (coreTileLoop >= DELAY_PSCALE_N && delay3CoreTaskId < coreTaskNum) {
                // delay3TileInfo = tileInfos[(coreTileLoop - DELAY_PSCALE_N) & PRELOAD_TASK_CACHE_MASK]
                if (delay3TileInfo.isUpdatePScale) {
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 +
                                                                           delay3TileInfo.tileMaxIdx / 2);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 +
                                                                           delay3TileInfo.tileMaxIdx / 2 + 16);
                        blockGMaxL1ToUb(delay3TileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 +
                                                                          delay3TileInfo.tileMaxIdx / 2);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 +
                                                                          delay3TileInfo.tileMaxIdx / 2 + 16);
                        if (delay3TileInfo.tileMaxIdx == 3 ||
                            (delay3CoreTaskId == coreTaskNum - 1 &&
                             delay3TileInfo.isLastKvsTile)) { // tilemax最后一块 或者 单核上的最后一块
                            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
                            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK + 16);
                        }
                    } else {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 +
                                                                        delay3TileInfo.tileMaxIdx / 2);
                        epilogueComputePScale(l1PScaleTensorTla, delay3TileInfo, delay3TaskInfo);
                        if constexpr (EpilogueRescaleO::DispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY) {
                            if (delay3TileInfo.isLastKvsTile) {
                                AscendC::PipeBarrier<PIPE_V>();
                                auto softmaxMax =
                                    resource.ubBuf.template GetBufferByByte<half>(EpMXFP8::UB_SOFTMAX_MAX_BUF_OFFSET);
                                auto kgSnap = resource.ubBuf.template GetBufferByByte<half>(
                                    EpMXFP8::UB_LSE_MAX_SNAP_OFFSET +
                                    (delay3CoreTaskId & (EpMXFP8::UB_LSE_MAX_SNAP_CNT - 1u)) *
                                        EpMXFP8::UB_LSE_MAX_SNAP_SIZE);
                                AscendC::DataCopy(kgSnap, softmaxMax, 128);
                            }
                        }
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_PSCALE_C2_BUF0 +
                                                                          delay3TileInfo.pscaleNum);
                    }
                }
            }
            if (coreTaskId < coreTaskNum) {
                CreateTileInfo(curTileInfo, coreTileLoop, static_cast<uint32_t>(curTaskTileIdx),
                               curTaskInfo); // 创建任务

                uint32_t mm1ResBufId = (curTileInfo.loop >> 1) & 1u;
                uint32_t subBlockIdx = curTileInfo.loop & 1u;

                if ASCEND_IS_AIC {
                    auto gQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(
                        curTaskInfo.qsActBaseTile, // rows = S1 实际 tile
                        qHeadMul * embed_);        // cols → row-stride（BNSD 时 qHeadMul=1，即 embed_）
                    auto gQTensorTla = tla::MakeTensor(gQ[curTaskInfo.gmOffsetQ], gQLayoutTla, Arch::PositionGM{});
                    auto gKLayout = tla::MakeLayout<ElementK, LayoutK>(
                        curTaskInfo.kvSeqlen,
                        kvHeadMul * embed_); // [BSND stride 修复] row-stride=kvHeadMul*embed(BNSD=embed)
                    auto gKTensorTla = tla::MakeTensor(gK[curTaskInfo.gmOffsetK], gKLayout, Arch::PositionGM{});
                    auto curUBSTensorTla = GetTile(ubSTensorTla, MakeCoord(0, EpMXFP8::QS_BASE_SIZE * mm1ResBufId),
                                                   MakeShape(EpMXFP8::KVS_BASE_SIZE, EpMXFP8::QS_BASE_SIZE));

                    // odd 不再进 PrefetchK。even（含 isLast 尾块 128）才 gather；pairEven 再 mmad256。
                    // Wait 必须在 if 外面。
                    const bool qkPairOdd = (curTileInfo.pvGatheredKvSTileIdx & 1u) != 0u;
                    const bool qkPairEven = !qkPairOdd && !curTileInfo.isLastKvsTile;
#ifdef __DAV_CUBE__
                    if (!qkPairOdd) {
                        blockMmadQK.PrefetchK(gQTensorTla, gKTensorTla, gQDequantScale[curTaskInfo.gmOffsetQScale],
                                              gKDequantScale[curTaskInfo.gmOffsetKScale],
                                              gSparseIdx[curTaskInfo.gmOffsetSparseIdx], curTaskInfo, curTileInfo,
                                              embed_, kvBaseTile_, blockShapeY_, kvHeadMul, qHeadMul);
                        if (qkPairEven) {
                            blockMmadQK.ComputeQKMmad256Even(curTaskInfo, embed_);
                        }
                    }
#endif
                    AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] +
                                                                      subBlockIdx * 16);
#ifdef __DAV_CUBE__
                    if (qkPairEven) {
                        blockMmadQK.EmitFixpipeQK256Even(curUBSTensorTla, curTaskInfo, subBlockIdx);
                    } else {
                        blockMmadQK.ComputeQK(curUBSTensorTla, curTaskInfo, curTileInfo, embed_, subBlockIdx);
                    }
#endif
                    AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] +
                                                                     subBlockIdx * 16);
                } else {
#ifdef __DAV_VEC__
                    if (subBlockIdx == AscendC::GetSubBlockIdx()) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                        uint32_t l1PBufId = curTileInfo.pSlot;
                        auto l1PLayoutTla =
                            tla::MakeLayout<ElementDisguiseP, LayoutPL1>(EpMXFP8::KVS_BASE_SIZE, EpMXFP8::QS_BASE_SIZE);
                        auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
                        GemmCoord actualBlockShapeQK{curTileInfo.kvsActBaseTile, curTaskInfo.qsActBaseTile, embed_};
                        epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, curTileInfo, curTaskInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                    }
                    if (curTileInfo.isUpdatePScale) {
                        if (curTileInfo.tileMaxIdx == 0) {
                            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
                        }
                        auto l1GMaxLayoutTla =
                            tla::MakeLayout<ElementLocalGlobalMax, LayoutLocalGlobalMax>(2, EpMXFP8::QS_BASE_SIZE);
                        auto l1GMaxTensorTla = tla::MakeTensor(l1LocalGlobalMaxTensor[curTileInfo.tileMaxIdx],
                                                               l1GMaxLayoutTla, Arch::PositionL1{});
                        epilogueCopyGlobalMaxUbToL1(l1GMaxTensorTla, curTileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 +
                                                                          curTileInfo.tileMaxIdx / 2);
                    }
#endif
                }
            }

            // 是否要切换到下一个qs任务, 新的kvs
            if (curTaskTileIdx == curTaskInfo.taskTileNum - 1 && coreTaskId < coreTaskNum) {
                preTaskTileSum += curTaskInfo.taskTileNum;
                coreTaskId++;
            }
            if (coreTileLoop >= PV_RE_DELAY_N && delay20TaskTileIdx == delay20TaskInfo.taskTileNum - 1) {
                delay20preTaskTileSum += delay20TaskInfo.taskTileNum;
                delay20CoreTaskId++;
            }
            if (coreTileLoop >= DELAY_PSCALE_N && delay3TaskTileIdx == delay3TaskInfo.taskTileNum - 1 &&
                delay3CoreTaskId < coreTaskNum) {
                delay3preTaskTileSum += delay3TaskInfo.taskTileNum;
                delay3CoreTaskId++;
            }
            coreTileLoop++;
        }

        // release reverse sync flags
        ReleaseSyncFlags();
    }

    __aicore__ inline void FetchBaseShapeInfo(__gm__ BlockSparseAttentionTilingData* bsaTilingData)
    {
        batch_ = bsaTilingData->batch;
        qHeads_ = bsaTilingData->numHeads;
        kvHeads_ = bsaTilingData->kvHeads;
        embed_ = bsaTilingData->embeddingSize;
        firstBatchTaskNum_ = bsaTilingData->firstBatchTaskNum;
        totalTaskNum_ = bsaTilingData->totalTaskNum;
        coreTaskNum_ = bsaTilingData->coreTaskNum;
        blockShapeX_ = bsaTilingData->blockShapeX;
        blockShapeY_ = bsaTilingData->blockShapeY;
        scaleValue_ = bsaTilingData->scaleValue;
        // mask2idx tile info
        xBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.xBlockNumAligned;
        yBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.yBlockNumAligned;
        avgRowPerSubCore_ = bsaTilingData->BsaMask2IdxTileInfo.avgRowPerSubCore;
        preActiveSubCoreNum_ = bsaTilingData->BsaMask2IdxTileInfo.preActiveSubCoreNum;
        // base tile info
        qBaseTile_ = bsaTilingData->BsaBaseTileInfo.qBaseTile;
        kvBaseTile_ = bsaTilingData->BsaBaseTileInfo.kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = (!bsaTilingData->useUniformQSeqlen) && (!bsaTilingData->useUniformKvSeqlen);
        sparseIdxSize_ = bsaTilingData->selectIdxSize;
        // aligned seqlen q & kv
        qSeqlenAligned_ = bsaTilingData->maxQSeqlen;
        kvSeqlenAligned_ = bsaTilingData->maxKvSeqlen;
    }

    __aicore__ inline void InitLayoutStride()
    {
        // O stride
        if constexpr (qFormat == Format::TND) {
            strideO_ = static_cast<int64_t>(qHeads_) * embed_;
        } else if constexpr (qFormat == Format::BNSD) {
            strideOB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * embed_;
            strideON_ = qSeqlenAligned_ * embed_;
            strideOS_ = embed_;
        } else if constexpr (qFormat == Format::BSND) {
            strideOB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * embed_;
            strideOS_ = static_cast<int64_t>(qHeads_) * embed_;
            strideON_ = embed_;
        }
        // LSE stride（TND [T,N,1]，BNSD [B,N,S,1]，BSND [B,S,N,1]）
        if constexpr (qFormat == Format::TND) {
            strideLse_ = static_cast<int64_t>(qHeads_);
        } else if constexpr (qFormat == Format::BNSD) {
            strideLseB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_;
            strideLseN_ = qSeqlenAligned_;
            strideLseS_ = 1;
        } else if constexpr (qFormat == Format::BSND) {
            strideLseB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_);
            strideLseS_ = static_cast<int64_t>(qHeads_);
            strideLseN_ = 1;
        }
        // V / V-scale / K-scale stride
        const int64_t kScaleBytesPerToken = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (kvFormat == Format::TND) {
            strideKVData_ = static_cast<int64_t>(kvHeads_) * embed_;
            vScaleStrideN_ = static_cast<int64_t>(embed_) * 2;
            kScaleStrideN_ = kScaleBytesPerToken;
        } else if constexpr (kvFormat == Format::BNSD) {
            strideKVDataB_ = static_cast<int64_t>(kvHeads_) * kvSeqlenAligned_ * embed_;
            strideKVDataN_ = kvSeqlenAligned_ * embed_;
            vScaleStrideB_ =
                static_cast<int64_t>(kvHeads_) * CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * embed_ * 2;
            vScaleStrideN_ = CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * embed_ * 2;
            kScaleStrideB_ = static_cast<int64_t>(kvHeads_) * kvSeqlenAligned_ * kScaleBytesPerToken;
            kScaleStrideN_ = kvSeqlenAligned_ * kScaleBytesPerToken;
        } else if constexpr (kvFormat == Format::BSND) {
            strideKVDataB_ = kvSeqlenAligned_ * static_cast<int64_t>(kvHeads_) * embed_;
            strideKVDataN_ = embed_;
            vScaleStrideB_ =
                CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * static_cast<int64_t>(kvHeads_) * embed_ * 2;
            vScaleStrideN_ = static_cast<int64_t>(embed_) * 2;
            kScaleStrideB_ = kvSeqlenAligned_ * static_cast<int64_t>(kvHeads_) * kScaleBytesPerToken;
            kScaleStrideN_ = kScaleBytesPerToken;
        }
        // Q 数据 / Q-scale stride（Host 沿 D）
        const int64_t qScaleBytesPerToken = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (qFormat == Format::TND) {
            strideQData_ = static_cast<int64_t>(qHeads_) * embed_;
            qScaleStrideN_ = qScaleBytesPerToken;
        } else if constexpr (qFormat == Format::BNSD) {
            strideQDataB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * embed_;
            strideQDataN_ = qSeqlenAligned_ * embed_;
            qScaleStrideB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * qScaleBytesPerToken;
            qScaleStrideN_ = qSeqlenAligned_ * qScaleBytesPerToken;
        } else if constexpr (qFormat == Format::BSND) {
            strideQDataB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * embed_;
            strideQDataN_ = embed_;
            qScaleStrideB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * qScaleBytesPerToken;
            qScaleStrideN_ = qScaleBytesPerToken;
        }
    }

    __aicore__ inline void InitCrossCoreDstBuf(
        AscendC::LocalTensor<ElementDisguiseP> (&l1PTensor)[GmMXFP8::L1_P_BUF_CNT],
        AscendC::LocalTensor<ElementPScale> (&l1PScaleTensor)[GmMXFP8::L1_P_SCALE_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&l1LocalGlobalMax)[GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[EpMXFP8::UB_S_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&ubPeerGlobalMax)[EpMXFP8::UB_PEER_GLOBAL_MAX_CNT],
        AscendC::LocalTensor<ElementOTmp>& ubOTmpTensor)

    {
        // L1 侧 buffer 初始化：OFFSET + i * SIZE 递增
        for (uint32_t i = 0; i < GmMXFP8::L1_P_BUF_CNT; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementDisguiseP>(GmMXFP8::L1_P_BUF_OFFSET +
                                                                                     GmMXFP8::L1_P_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < GmMXFP8::L1_P_SCALE_BUF_CNT; i++) {
            l1PScaleTensor[i] = resource.l1Buf.template GetBufferByByte<ElementPScale>(
                GmMXFP8::L1_P_SCALE_BUF_OFFSET + GmMXFP8::L1_P_SCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_CNT; i++) {
            l1LocalGlobalMax[i] = resource.l1Buf.template GetBufferByByte<ElementS>(
                GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_OFFSET + GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        // UB 侧 buffer 初始化
        for (uint32_t i = 0; i < EpMXFP8::UB_S_BUF_CNT; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(EpMXFP8::UB_S_BUF_OFFSET +
                                                                             EpMXFP8::UB_S_INNER_BUF_OFFSET * i);
        }
        for (uint32_t i = 0; i < EpMXFP8::UB_PEER_GLOBAL_MAX_CNT; i++) {
            ubPeerGlobalMax[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                EpMXFP8::UB_PEER_GLOBAL_MAX_BUF_OFFSET + EpMXFP8::UB_PEER_GLOBAL_MAX_BUF_SIZE * i);
        }
        // OTmp 单 buffer
        ubOTmpTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(EpMXFP8::UB_OTMP_BUF_OFFSET);
    }

    // 空 KV 行：attentionOut 置 0，softmaxLse 置 -inf。标量写 GM，不占用正在流水的 UB。
    template <class TensorO, class TensorLse>
    __aicore__ inline void WriteEmptyKvOutput(const TaskInfo& info, TensorO& oGm, TensorLse& lseGm)
    {
        constexpr uint32_t kNegInfBits = 0xFF800000u;
        float negInf = 0.0f;
        *reinterpret_cast<uint32_t*>(&negInf) = kNegInfBits;

        const uint32_t splitM = (EpMXFP8::QS_BASE_SIZE + 1) / 2;
        uint32_t rowBegin = 0;
        uint32_t rowEnd = info.qsActBaseTile < splitM ? info.qsActBaseTile : splitM;
        if (AscendC::GetSubBlockIdx() != 0) {
            rowBegin = info.qsActBaseTile < splitM ? info.qsActBaseTile : splitM;
            rowEnd = info.qsActBaseTile;
        }
        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            uint64_t oBase = static_cast<uint64_t>(info.gmOffsetO) + static_cast<uint64_t>(row) * info.oShapeCol;
            for (uint32_t col = 0; col < embed_; ++col) {
                oGm.SetValue(oBase + col, static_cast<ElementO>(0));
            }
            if constexpr (EpilogueRescaleO::DispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY) {
                lseGm.SetValue(static_cast<uint64_t>(info.gmOffsetLse) + static_cast<uint64_t>(row) * info.lseShapeCol,
                               negInf);
            }
        }
    }

    __aicore__ inline void SkipEmptyCreatedTasks(uint32_t& taskId, uint32_t createdTaskId, TaskInfo* taskInfos)
    {
        while (taskId < createdTaskId && taskInfos[taskId & PRELOAD_TASK_CACHE_MASK].taskTileNum == 0) {
            taskId++;
        }
    }

    // 由 coreTaskId (本核上的 task 序号) 反推 globalTaskId 并填充 TaskInfo
    // 仅在 task 起始 tile (taskTileId == 0) 调用，避免每个 tile 重复计算
    // coreIdx / coreNum 由调用方 Process 传入, 避免重复调用 GetBlockIdx/GetBlockNum
    __aicore__ inline void CreateTaskInfo(TaskInfo& info, uint32_t coreTaskId, uint32_t coreIdx, uint32_t coreNum,
                                          AscendC::GlobalTensor<int64_t>& gActualQseqlen,
                                          AscendC::GlobalTensor<int64_t>& gActualKvseqlen,
                                          AscendC::GlobalTensor<ElementSparseCount>& gSparseCount,
                                          AscendC::GlobalTensor<ElementSparseIdx>& gSparseIdx,
                                          BatchOffsetInfo& batchInfo)
    {
        uint32_t taskIdx = coreTaskId * coreNum + coreIdx; // 反推全局 task 序号

        uint32_t groupSize = qHeads_ / kvHeads_;
        uint32_t qSTileNumPerFullXBlock = CeilDiv(blockShapeX_, qBaseTile_);

        // 反推 taskIdx 所在的 batch：同核 taskIdx 单调递增，
        // 从 batchInfo 保存的上次扫描位置续算，不再从 batch 0 重扫
        int64_t qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(batchInfo.curBatch) : qSeqlenAligned_;
        int64_t kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(batchInfo.curBatch) : kvSeqlenAligned_;
        uint32_t curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
        while (taskIdx >= batchInfo.curTotalTaskNum) {
            ++batchInfo.curBatch;
            batchInfo.preTotalTaskNum = batchInfo.curTotalTaskNum;
            // TND: 跨 batch 时累加上一 batch 实际长度（此时 kvSeqlen/qSeqlen 仍为上一 batch 值）
            if constexpr (qFormat == Format::TND) {
                batchInfo.oBOffset += qSeqlen * strideO_;
                batchInfo.lseBOffset += qSeqlen * strideLse_;
                batchInfo.qBOffset += qSeqlen * strideQData_;
                batchInfo.qScaleBOffset += qSeqlen * static_cast<int64_t>(qHeads_) * qScaleStrideN_;
            }
            if constexpr (kvFormat == Format::TND) {
                batchInfo.vBOffset += kvSeqlen * strideKVData_;
                batchInfo.vScaleBOffset +=
                    CeilDiv(kvSeqlen, static_cast<int64_t>(64)) * static_cast<int64_t>(kvHeads_) * embed_ * 2;
                batchInfo.kScaleBOffset += kvSeqlen * static_cast<int64_t>(kvHeads_) * kScaleStrideN_;
            }
            qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(batchInfo.curBatch) : qSeqlenAligned_;
            kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(batchInfo.curBatch) : kvSeqlenAligned_;
            curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
            batchInfo.curTotalTaskNum += curQSTileNum * qHeads_;
        }
        uint32_t curBatch = batchInfo.curBatch;

        // 拆解 taskIdx 得到 head / qSTile / xBlock 索引
        uint32_t taskIdxCurBatch = taskIdx - batchInfo.preTotalTaskNum;
        uint32_t qHeadIdx = taskIdxCurBatch / curQSTileNum;
        uint32_t kvHeadIdx = qHeadIdx / groupSize;
        uint32_t qSTileIdx = taskIdxCurBatch - qHeadIdx * curQSTileNum;
        uint32_t xBlockIdx = qSTileIdx / qSTileNumPerFullXBlock;
        uint32_t qSTileIdxCurXBlock = qSTileIdx - xBlockIdx * qSTileNumPerFullXBlock;

        // 实际 Q tile 大小 (最后一个 tile 可能不足 qBaseTile_)
        uint32_t xBlockNumAval = static_cast<uint32_t>(CeilDiv(qSeqlen, static_cast<int64_t>(blockShapeX_)));
        uint32_t xBlockSize =
            (xBlockIdx == xBlockNumAval - 1) ? static_cast<uint32_t>(qSeqlen - xBlockIdx * blockShapeX_) : blockShapeX_;
        uint32_t qSTileNumCurXBlock = CeilDiv(xBlockSize, qBaseTile_);
        uint32_t qsActBaseTile = (qSTileIdxCurXBlock == qSTileNumCurXBlock - 1) ?
                                     (xBlockSize - qSTileIdxCurXBlock * qBaseTile_) :
                                     qBaseTile_;

        // 从 sparse mask 计算 gatheredKvSeqlen
        uint32_t gmOffsetSparseCount =
            curBatch * qHeads_ * xBlockNumAligned_ + qHeadIdx * xBlockNumAligned_ + xBlockIdx;
        uint32_t yBlockNumRsvd = gSparseCount.GetValue(gmOffsetSparseCount);
        uint32_t gmOffsetSparseIdx = gmOffsetSparseCount * yBlockNumAligned_;
        uint32_t yBlockNumAval = static_cast<uint32_t>(CeilDiv(kvSeqlen, static_cast<int64_t>(blockShapeY_)));
        // 保留块数为 0 时没有最后一块，不能做 yBlockNumRsvd - 1（uint32 下溢）。
        // 空 KV：attentionOut 写 0，softmaxLse 写 -inf，taskTileNum 为 0，主循环跳过该 task。
        int64_t gatheredKvSeqlen = 0;
        if (yBlockNumRsvd != 0) {
            uint32_t lastIdxOffset = gmOffsetSparseIdx + yBlockNumRsvd - 1;
            uint32_t lastSparseIdx = gSparseIdx.GetValue(lastIdxOffset);
            uint32_t lastYBlockSize = (lastSparseIdx == yBlockNumAval - 1) ?
                                          static_cast<uint32_t>(kvSeqlen - lastSparseIdx * blockShapeY_) :
                                          blockShapeY_;
            gatheredKvSeqlen = static_cast<int64_t>(yBlockNumRsvd - 1) * blockShapeY_ + lastYBlockSize;
        }

        // taskTileNum = ceil(gatheredKvSeqlen / kvBaseTile_)；空 KV 不进入 tile 循环
        uint32_t taskTileNum = (gatheredKvSeqlen == 0) ?
                                   0 :
                                   static_cast<uint32_t>(CeilDiv(gatheredKvSeqlen, static_cast<int64_t>(kvBaseTile_)));

        // O GM offset
        int64_t qSOffset = static_cast<int64_t>(xBlockIdx) * blockShapeX_ + qSTileIdxCurXBlock * qBaseTile_;
        if constexpr (qFormat == Format::TND) {
            info.gmOffsetO = batchInfo.oBOffset + qSOffset * strideO_ + qHeadIdx * embed_;
        } else if constexpr (qFormat == Format::BNSD) {
            int64_t oBOffset = static_cast<int64_t>(curBatch) * strideOB_;
            info.gmOffsetO = oBOffset + qHeadIdx * strideON_ + qSOffset * strideOS_;
        } else if constexpr (qFormat == Format::BSND) {
            int64_t oBOffset = static_cast<int64_t>(curBatch) * strideOB_;
            info.gmOffsetO = oBOffset + qSOffset * strideOS_ + qHeadIdx * strideON_;
        }
        if constexpr (qFormat == Format::TND) {
            info.gmOffsetLse = batchInfo.lseBOffset + qSOffset * strideLse_ + qHeadIdx;
            info.lseShapeCol = static_cast<uint32_t>(strideLse_);
        } else if constexpr (qFormat == Format::BNSD) {
            int64_t lseBOffset = static_cast<int64_t>(curBatch) * strideLseB_;
            info.gmOffsetLse = lseBOffset + qHeadIdx * strideLseN_ + qSOffset * strideLseS_;
            info.lseShapeCol = static_cast<uint32_t>(strideLseS_);
        } else if constexpr (qFormat == Format::BSND) {
            int64_t lseBOffset = static_cast<int64_t>(curBatch) * strideLseB_;
            info.gmOffsetLse = lseBOffset + qSOffset * strideLseS_ + qHeadIdx * strideLseN_;
            info.lseShapeCol = static_cast<uint32_t>(strideLseS_);
        }
        // O shape col
        if constexpr (qFormat == Format::TND) {
            info.oShapeCol = static_cast<uint32_t>(strideO_);
        } else if constexpr (qFormat == Format::BNSD) {
            info.oShapeCol = static_cast<uint32_t>(strideOS_);
        } else if constexpr (qFormat == Format::BSND) {
            info.oShapeCol = static_cast<uint32_t>(strideOS_);
        }

        info.batchIdx = curBatch;
        info.qHeadIdx = qHeadIdx;
        info.kvHeadIdx = kvHeadIdx;
        info.qSTileIdx = qSTileIdx;
        info.xBlockIdx = xBlockIdx;
        info.gatheredKvSeqlen = gatheredKvSeqlen;
        info.qsActBaseTile = qsActBaseTile;

        info.qsActBaseTileAlign8 = CeilDiv(qsActBaseTile, 8u) * 8;
        info.qsActBaseTileAlign16 = CeilDiv(qsActBaseTile, 16u) * 16;
        info.qsActBaseTileAlign64 = CeilDiv(qsActBaseTile, 64u) * 64;
        info.qsActBaseTileAlign128 = CeilDiv(qsActBaseTile, 128u) * 128;

        info.qsBaseTile = qBaseTile_;
        info.kvsBaseTile = kvBaseTile_;
        info.taskTileNum = taskTileNum;
        // ===== PV 稀疏 gather 所需 =====
        info.gmOffsetSparseIdx = gmOffsetSparseIdx; // 本 task sparseIdx 起始
        info.kvSeqlen = kvSeqlen;
        info.yBlockNumAval = yBlockNumAval;
        info.yBlockNumRsvd = yBlockNumRsvd;
        // V 数据 batch+head 基址（seq 由 gather 的 oriStartOffset 处理，基址只含 batch+head）
        // V-scale batch+head 基址（字节单位，布局 [B,N2,S2//64,D,2]）
        if constexpr (kvFormat == Format::TND) {
            // TND V [T2,N2,D]：head 在 token 内 stride = embed_（≠ strideKVData_，后者是 batch-token 步长
            // kvHeads_*embed_）
            info.gmOffsetV = batchInfo.vBOffset + kvHeadIdx * embed_;
            info.gmOffsetVScale = batchInfo.vScaleBOffset + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = batchInfo.kScaleBOffset + kvHeadIdx * kScaleStrideN_;
        } else if constexpr (kvFormat == Format::BNSD) {
            info.gmOffsetV = static_cast<int64_t>(curBatch) * strideKVDataB_ + kvHeadIdx * strideKVDataN_;
            info.gmOffsetVScale = static_cast<int64_t>(curBatch) * vScaleStrideB_ + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = static_cast<int64_t>(curBatch) * kScaleStrideB_ + kvHeadIdx * kScaleStrideN_;
        } else { // BSND
            info.gmOffsetV = static_cast<int64_t>(curBatch) * strideKVDataB_ + kvHeadIdx * strideKVDataN_;
            info.gmOffsetVScale = static_cast<int64_t>(curBatch) * vScaleStrideB_ + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = static_cast<int64_t>(curBatch) * kScaleStrideB_ + kvHeadIdx * kScaleStrideN_;
        }
        // Q / Q-scale：稠密搬需要带上本 task 的 Q seq 起点（与上方 O 的 qSOffset 一致）
        const int64_t qScaleBytesPerTok = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (qFormat == Format::TND) {
            info.gmOffsetQ = batchInfo.qBOffset + qSOffset * strideQData_ + qHeadIdx * embed_;
            info.gmOffsetQScale = batchInfo.qScaleBOffset +
                                  qSOffset * static_cast<int64_t>(qHeads_) * qScaleBytesPerTok +
                                  qHeadIdx * qScaleStrideN_;
        } else if constexpr (qFormat == Format::BNSD) {
            info.gmOffsetQ =
                static_cast<int64_t>(curBatch) * strideQDataB_ + qHeadIdx * strideQDataN_ + qSOffset * embed_;
            info.gmOffsetQScale = static_cast<int64_t>(curBatch) * qScaleStrideB_ + qHeadIdx * qScaleStrideN_ +
                                  qSOffset * qScaleBytesPerTok;
        } else { // BSND
            info.gmOffsetQ = static_cast<int64_t>(curBatch) * strideQDataB_ +
                             qSOffset * static_cast<int64_t>(qHeads_) * embed_ + qHeadIdx * embed_;
            info.gmOffsetQScale = static_cast<int64_t>(curBatch) * qScaleStrideB_ +
                                  qSOffset * static_cast<int64_t>(qHeads_) * qScaleBytesPerTok +
                                  qHeadIdx * qScaleStrideN_;
        }
    }

    // 每个 tile 调用一次，根据 coreTileLoop / curTaskTileIdx / taskInfo 填充 TileInfo
    __aicore__ inline void CreateTileInfo(TileInfo& info, uint64_t coreTileLoop, uint32_t curTaskTileIdx,
                                          const TaskInfo& taskInfo)
    {
        info.loop = static_cast<uint32_t>(coreTileLoop);
        info.curKvsTileLoopIdx = curTaskTileIdx;
        info.isFirstKvsTile = (curTaskTileIdx == 0);
        info.isLastKvsTile = (curTaskTileIdx + 1 == taskInfo.taskTileNum);
        info.isLastSecondKvsTile =
            ((curTaskTileIdx + 2 == taskInfo.taskTileNum) || (taskInfo.taskTileNum == 1)); // pv rowsum 刷 1

        info.isUpdatePScale = (info.isLastKvsTile || ((info.curKvsTileLoopIdx + 1) % MXFP8Kernel::TILE_GROUP_N == 0));
        info.isTileGoupFirstTile = (info.curKvsTileLoopIdx % MXFP8Kernel::TILE_GROUP_N == 0);
        if (info.isFirstKvsTile) {
            kvsFirstTileStartVecCore_ = static_cast<uint32_t>(coreTileLoop) & 1u;
        }
        info.kvsFirstTileStartVecCore = kvsFirstTileStartVecCore_;

        if (info.isTileGoupFirstTile && !info.isFirstKvsTile) { // 非第一个 kvs tileGroup，更新 dm id
            updateScaleIdx_ = (updateScaleIdx_ + 1u) & 3u;
        }

        // task 第一个 tile 或 TileGroup 第一个 tile 时切换 buff (双缓冲 / 四缓冲)
        if (info.isFirstKvsTile || info.isTileGoupFirstTile) {
            tileMaxIdx_ = (tileMaxIdx_ + 1u) & 3u;
            pscaleNum_ = pscaleNum_ + 1u;
            if (pscaleNum_ == 20u) {
                pscaleNum_ = 0u;
            }
        }
        info.pscaleNum = static_cast<uint16_t>(pscaleNum_ >= 10u);
        info.pSlot = static_cast<uint16_t>(pSlot_);
        pSlot_ = pSlot_ + 1u;
        if (pSlot_ == GmMXFP8::L1_P_BUF_CNT) {
            pSlot_ = 0u;
        }
        info.tileMaxIdx = tileMaxIdx_;
        info.updateScaleIdx = updateScaleIdx_;
        info.isKvsFirstTilePerCore = (info.curKvsTileLoopIdx % MXFP8Kernel::TILE_GROUP_N / 2 == 0);

        info.kvsActBaseTile =
            info.isLastKvsTile ? (taskInfo.gatheredKvSeqlen - (taskInfo.taskTileNum - 1) * kvBaseTile_) : kvBaseTile_;
        // PV 稀疏 gather 用：当前 tile 在 task 内 gather 后的 KV base tile 序号
        info.pvGatheredKvSTileIdx = curTaskTileIdx;
        info.kvsActBaseTileAlign16 = (info.kvsActBaseTile + 15) / 16 * 16;
        info.kvsActBaseTileAlign32 = (info.kvsActBaseTile + 31) / 32 * 32;
        info.kvsActBaseTileAlign64 = (info.kvsActBaseTile + 63) / 64 * 64;
    }

    __aicore__ inline void InitSyncFlags()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK + 16);
        } else {
            // 核间
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[0]);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[1]);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
            // mask2index
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);

            // vec
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_VEC1_RES_BUF0_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_VEC1_RES_BUF1_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_ATTN_BUF_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF0_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF1_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF2_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF3_FLAG);
        }
    }

    __aicore__ inline void ReleaseSyncFlags()
    {
        if ASCEND_IS_AIC {
            // 核间
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0] + 16);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1] + 16);
        } else {
            // 核间
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);

            // mask2index
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);

            // vec
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_VEC1_RES_BUF0_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_VEC1_RES_BUF1_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_ATTN_BUF_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF0_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF1_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF2_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF3_FLAG);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    /*
    tiling info, which are const in each kernel launch
    */
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    uint32_t coreTaskNum_; // 向下取整: totalTaskNum_ / blockDim
    uint32_t blockShapeX_;
    uint32_t blockShapeY_;
    float scaleValue_;
    // mask2idx tile info
    uint32_t xBlockNumAligned_;
    uint32_t yBlockNumAligned_;
    uint32_t avgRowPerSubCore_;
    uint32_t preActiveSubCoreNum_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // whether actual seqlen is provided
    uint32_t actSeqAval_;
    // workspace size
    uint64_t sparseIdxSize_;
    // aligned seqlen q & kv
    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;
    // ===== stride 族（O / V 数据 / V-scale）=====
    int64_t strideO_;
    int64_t strideOB_;
    int64_t strideON_;
    int64_t strideOS_;
    int64_t strideLse_ = 0;
    int64_t strideLseB_ = 0;
    int64_t strideLseN_ = 0;
    int64_t strideLseS_ = 0;
    // ===== V 数据 / V-scale stride（PV 稀疏 gather 用，operator() 内预算一次）=====
    // V 数据为 e4m3，sizeof=1，元素偏移即字节偏移
    int64_t strideKVData_ = 0;  // TND: 每 token 跨 kvHead 的步长（元素）
    int64_t strideKVDataB_ = 0; // BNSD/BSND: V 数据 batch 步长
    int64_t strideKVDataN_ = 0; // BNSD/BSND: V 数据 head 步长
    // V-scale 为 E8M0(1 字节)，GM 布局 [B,N2,S2//64,D,2]；B16 视图下每 64-S2 行 group 占 D 个 B16 = 2D 字节
    int64_t vScaleStrideB_ = 0; // V-scale batch 步长（字节）
    int64_t vScaleStrideN_ = 0; // V-scale head 步长（字节）
    // ===== K-scale / Q / Q-scale stride（QK Mm1 用；K-scale Host 沿 D）=====
    int64_t kScaleStrideB_ = 0;
    int64_t kScaleStrideN_ = 0;
    int64_t strideQData_ = 0;
    int64_t strideQDataB_ = 0;
    int64_t strideQDataN_ = 0;
    int64_t qScaleStrideB_ = 0;
    int64_t qScaleStrideN_ = 0;

    // ---- CreateTileInfo 持久状态 (跨 tileGroup 保持 ----
    uint32_t kvsFirstTileStartVecCore_ = 0; // task 首 tile 时切换 (loop % 2)
    uint32_t updateScaleIdx_ = 3;           // C2 Sync 时切换 (循环 0~3)// dm
    uint32_t tileMaxIdx_ = 3;               // 四缓冲索引 (0~3)
    uint32_t pscaleNum_ = 19;               // 20 缓冲索引 (0~19)，Dpv=20
    uint32_t pSlot_ = 0;                    // L1 P 环，每 tile +1，热路径读 TileInfo.pSlot
};

} // namespace BsaKernelArch35
