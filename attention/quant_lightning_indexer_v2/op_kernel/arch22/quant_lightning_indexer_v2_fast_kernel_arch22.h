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
 * \file quant_lightning_indexer_v2_fast_kernel_arch22.h
 * \brief Fast template kernel for QuantLightningIndexerV2 on arch22 (int8, TND query).
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH22_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"
#include "quant_lightning_indexer_v2_fast_service_vector_arch22.h"
#include "../quant_lightning_indexer_v2_metadata.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;

// fast 模板使用的主算子 tiling 视图，字段与 QLIV2TilingData 的对应关系见 Init()
struct QLIV2FastTilingData {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t sparseCount;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t sparseMode;
    uint32_t kCacheStride;
    uint32_t kScaleStride;
    uint64_t workspaceGroupStride;
};

template <typename QLIV2T>
class QLIV2FastKernel {
public:
    using Q_T = typename QLIV2T::queryType;
    using K_T = typename QLIV2T::keyType;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;

    __aicore__ inline QLIV2FastKernel() = default;

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights,
                                __gm__ uint8_t* queryScale, __gm__ uint8_t* keyScale, __gm__ uint8_t* cuSeqlensQ,
                                __gm__ uint8_t* cuSeqlensK, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK,
                                __gm__ uint8_t* cmpResidualK, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* outputIdxOffset, __gm__ uint8_t* metadata,
                                __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace,
                                const QLIV2TilingData* __restrict tiling, TPipe* tPipe)
    {
        // fast 模板只承接 int8 + TND query 场景，以下入参在 fast 语义下不参与计算
        (void)sequsedQ;
        (void)cmpResidualK;
        (void)outputIdxOffset;
        (void)sparseValues;
        (void)tPipe;

        // 分核与 slow 模板一致：由 metadata AICPU 算子写入，本核按 groupIndex 读回。
        // fast 分核数据位于扩展区（参考 A5，不占用分核记录区）：从头部读实际核数算出
        // 扩展区基址，3 槽连续布局，每 group 依次为 enable、tile 起点、tile 终点（左闭右开）
        groupIndex = QLIV2Fast::GetMixGroupIndex();
        GlobalTensor<uint32_t> metadataGm;
        if (metadata != nullptr) {
            metadataGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
            const uint32_t metadataAicNum = metadataGm.GetValue(QLI_V2_METADATA_AIC_NUM_INDEX);
            const uint32_t metadataAivNum = metadataGm.GetValue(QLI_V2_METADATA_AIV_NUM_INDEX);
            // 扩展区偏移量级为头部8 + 核数×8（实际核数下远小于 2^32），收窄安全
            const uint32_t extensionBase =
                static_cast<uint32_t>(GetQliV2MetadataExtensionOffset(metadataAicNum, metadataAivNum));
            const uint32_t groupBase = extensionBase + groupIndex * QLI_V2_A3_FAST_GROUP_SLOT_NUM;
            splitCore.coreEnable = metadataGm.GetValue(groupBase + QLI_V2_A3_FAST_ENABLE_SLOT);
            if (splitCore.coreEnable != 0U) {
                splitCore.tileBegin = metadataGm.GetValue(groupBase + QLI_V2_A3_FAST_START_SLOT);
                splitCore.tileEnd = metadataGm.GetValue(groupBase + QLI_V2_A3_FAST_END_SLOT);
            }
        }

        queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(query));
        keyGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
        weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weights));
        queryScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(queryScale));
        keyScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(keyScale));
        // Q 侧恒为 TND，cu_seqlens 为累积长度语义
        actualSeqLengthsGmQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensQ));
        // K 侧：TND 使用 cu_seqlens（累积），其余布局使用 seqused_k（每 batch 长度）
        if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqLengthsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensK));
        } else {
            actualSeqLengthsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequsedK));
        }
        blockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
        indiceOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sparseIndices));
        workspaceFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        workspaceFp16Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace));

        tilingData.bSize = tiling->bSize;
        tilingData.n2Size = tiling->n2Size;
        tilingData.gSize = tiling->gSize;
        tilingData.s1Size = tiling->s1Size;
        tilingData.s2Size = tiling->s2Size;
        tilingData.sparseCount = tiling->sparseCount;
        tilingData.blockSize = tiling->blockSize;
        tilingData.maxBlockNumPerBatch = tiling->maxBlockNumPerBatch;
        tilingData.sparseMode = tiling->sparseMode;
        tilingData.workspaceGroupStride = tiling->fastWorkspaceGroupStride;
        // v2 使用 keyStride0 / keyDequantScaleStride0，需复用与慢路径一致的 0 值回退：
        // PA 的 key 按 [block, blockSize, n2, D] 连续存放
        tilingData.kCacheStride = tiling->keyStride0;
        if (tilingData.kCacheStride == 0) {
            tilingData.kCacheStride = tiling->blockSize * tiling->n2Size * QLIV2Fast::D_SIZE;
        }
        tilingData.kScaleStride = tiling->keyDequantScaleStride0;
        if (tilingData.kScaleStride == 0) {
            tilingData.kScaleStride = tiling->blockSize;
        }
    }

    __aicore__ inline void Process()
    {
        if (tilingData.gSize == QLIV2Fast::GSIZE_FAST_24) {
            ProcessImpl<QLIV2Fast::GSIZE_FAST_24>();
        } else {
            ProcessImpl<QLIV2Fast::GSIZE_FAST_32>();
        }
    }

private:
    // v2 的 TND 序列长度输入为 cu_seqlens（长度 B+1，首元素为 0），
    // 第 b 个 batch 的结束位置是 cu_seqlens[b + 1]
    __aicore__ inline uint32_t ReadAccumSeqEnd(const GlobalTensor<int32_t>& seqEnds, uint32_t batchIdx, uint32_t prefix,
                                               uint32_t physicalLimit) const
    {
        const int32_t rawEnd = seqEnds.GetValue(batchIdx + 1U);
        if (rawEnd <= static_cast<int32_t>(prefix)) {
            return prefix;
        }
        const uint32_t end = static_cast<uint32_t>(rawEnd);
        return end < physicalLimit ? end : physicalLimit;
    }

    __aicore__ inline uint32_t ReadClampedSequenceLength(const GlobalTensor<int32_t>& sequenceLengths,
                                                         uint32_t batchIdx, uint32_t physicalLimit) const
    {
        const int32_t rawLength = sequenceLengths.GetValue(batchIdx);
        if (rawLength <= 0) {
            return 0;
        }
        const uint32_t length = static_cast<uint32_t>(rawLength);
        return length < physicalLimit ? length : physicalLimit;
    }

    struct WeightPreprocessUbLayout {
        static constexpr int INPUT_CAPACITY_ELEMENTS = 512;
        static constexpr int INPUT_BUFFER_BYTES = INPUT_CAPACITY_ELEMENTS * sizeof(half);
        static constexpr int PRODUCT_REGION_BYTES = 2 * QLIV2Fast::HardwareConstants::KILOBYTE_BYTES;
        static constexpr int PACKED_REGION_BYTES = 2 * QLIV2Fast::HardwareConstants::KILOBYTE_BYTES;

        static constexpr int WEIGHT_BASE = 0;
        static constexpr int SCALE_BASE = WEIGHT_BASE + INPUT_BUFFER_BYTES;
        static constexpr int PRODUCT_BASE = SCALE_BASE + INPUT_BUFFER_BYTES;
        static constexpr int PACKED_BASE = PRODUCT_BASE + PRODUCT_REGION_BYTES;
        static constexpr int INDICES_BASE = PACKED_BASE + PACKED_REGION_BYTES;

        static constexpr int PRODUCT_CAPACITY_ELEMENTS = 512;
        static constexpr int PACKED_CAPACITY_ELEMENTS = 1024;
        // Gather 在每行源数据末尾追加的零元素个数（uint32 字）及其对应的 half 个数
        static constexpr int GATHER_PADDING_WORDS = 8;
        static constexpr int GATHER_PADDING_HALF_ELEMENTS = GATHER_PADDING_WORDS * sizeof(uint32_t) / sizeof(half);
        static constexpr int INDEX_COUNT = QLIV2Fast::KernelConfig<QLIV2Fast::GSIZE_FAST_24>::S1_TILE *
                                           QLIV2Fast::KernelConfig<QLIV2Fast::GSIZE_FAST_24>::WEIGHT_ROW_STRIDE /
                                           (sizeof(uint32_t) / sizeof(half));
    };

    /**
     * 每个索引代表`productUb`内的字节偏移。一组 N24 源行包含`2 × 12`个 uint32 元素。
     * Gather 操作会对其做扩展：在每一行源数据末尾追加 8 个元素，得到两行目标行：
     *   [row 0: 12 source + 8 zero, row 1: 12 source + 8 zero]
     */
    __aicore__ inline void InitWeightGatherIndices(const LocalTensor<int32_t>& indices)
    {
        using N24Config = QLIV2Fast::KernelConfig<QLIV2Fast::GSIZE_FAST_24>;
        constexpr int rowsPerPair = 2;
        constexpr int halfElementsPerWord = sizeof(uint32_t) / sizeof(half);
        constexpr int rowPairs = N24Config::S1_TILE / rowsPerPair;
        constexpr int sourceWordsPerRow = N24Config::QUERY_HEAD_NUM / halfElementsPerWord;
        constexpr int paddedWordsPerRow = sourceWordsPerRow + WeightPreprocessUbLayout::GATHER_PADDING_WORDS;
        static_assert(paddedWordsPerRow * halfElementsPerWord == N24Config::WEIGHT_ROW_STRIDE);
        constexpr int wordsPerPair = rowsPerPair * paddedWordsPerRow;
        constexpr int sourceBytesPerRow = N24Config::QUERY_HEAD_NUM * sizeof(half);
        constexpr int sourceBytesPerPair = rowsPerPair * sourceBytesPerRow;
        constexpr int zeroOffset = N24Config::S1_TILE * N24Config::QUERY_HEAD_NUM * sizeof(half);

        SetFlag<HardEvent::V_S>(EVENT_ID3);
        WaitFlag<HardEvent::V_S>(EVENT_ID3);
        for (int pair = 0; pair < rowPairs; ++pair) {
            const int pairBase = pair * wordsPerPair;
            const int sourceBase = pair * sourceBytesPerPair;
            for (int row = 0; row < rowsPerPair; ++row) {
                const int destinationBase = pairBase + row * paddedWordsPerRow;
                const int rowSourceBase = sourceBase + row * sourceWordsPerRow * sizeof(uint32_t);
                for (int word = 0; word < sourceWordsPerRow; ++word) {
                    indices.SetValue(destinationBase + word, rowSourceBase + word * sizeof(uint32_t));
                }
                for (int word = sourceWordsPerRow; word < paddedWordsPerRow; ++word) {
                    indices.SetValue(destinationBase + word, zeroOffset);
                }
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID3);
        WaitFlag<HardEvent::S_V>(EVENT_ID3);
    }

    template <int N>
    __aicore__ inline void PrepareScaledWeight(uint64_t tokenPrefix, int tile, int validRows,
                                               const GlobalTensor<half>& outputGm, bool initializePadding)
    {
        constexpr int s1Tile = QLIV2Fast::KernelConfig<N>::S1_TILE;
        constexpr int weightStride = QLIV2Fast::KernelConfig<N>::WEIGHT_ROW_STRIDE;
        const int qHeadNum = static_cast<int>(tilingData.n2Size) * N;
        const uint64_t sourceOffset = (tokenPrefix + static_cast<uint64_t>(tile) * s1Tile) * qHeadNum;

        LocalTensor<half> weightUb(TPosition::VECCALC, WeightPreprocessUbLayout::WEIGHT_BASE,
                                   WeightPreprocessUbLayout::INPUT_CAPACITY_ELEMENTS);
        LocalTensor<half> scaleUb(TPosition::VECCALC, WeightPreprocessUbLayout::SCALE_BASE,
                                  WeightPreprocessUbLayout::INPUT_CAPACITY_ELEMENTS);
        LocalTensor<half> productUb(TPosition::VECCALC, WeightPreprocessUbLayout::PRODUCT_BASE,
                                    WeightPreprocessUbLayout::PRODUCT_CAPACITY_ELEMENTS);
        LocalTensor<half> packedWeightUb(TPosition::VECCALC, WeightPreprocessUbLayout::PACKED_BASE,
                                         WeightPreprocessUbLayout::PACKED_CAPACITY_ELEMENTS);
        LocalTensor<int32_t> weightIndicesUb(TPosition::VECCALC, WeightPreprocessUbLayout::INDICES_BASE,
                                             WeightPreprocessUbLayout::INDEX_COUNT);
        const GlobalTensor<half>& weightGm = weightsGm;
        const GlobalTensor<half>& scaleGm = queryScaleGm;

        constexpr uint32_t rowBytes = static_cast<uint32_t>(N * sizeof(half));
        const uint32_t validBytes = static_cast<uint32_t>(validRows) * rowBytes;

        // 不启用补齐：左/右补零数量与填充值参数在未启用时无效，恒取 0
        DataCopyPadExtParams<half> pad{false, 0, 0, 0};
        if constexpr (N == QLIV2Fast::GSIZE_FAST_32) {
            // N32 连续区整体搬运：1 个块，块长 validBytes 字节（validRows 行 × 行字节），无块间间隔
            DataCopyExtParams contiguousCopy;
            contiguousCopy.blockCount = 1;
            contiguousCopy.blockLen = validBytes;
            contiguousCopy.srcStride = 0;
            contiguousCopy.dstStride = 0;
            contiguousCopy.rsv = 0;
            DataCopyPad(weightUb, weightGm[sourceOffset], contiguousCopy, pad);
            DataCopyPad(scaleUb, scaleGm[sourceOffset], contiguousCopy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Mul(productUb, weightUb, scaleUb, validRows * N);
            // 下面的 MTE3_V 保护的是 productUb，而非这两个 MTE2 输入 buffer
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            DataCopyPad(outputGm, productUb, contiguousCopy);
            // productUb 会被下一个 Weight tile 复用；搬往 GM 的 DataCopyPad 是异步的，
            // 必须在下一次 Mul 覆盖前显式归还给 Vector
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            return;
        }

        // blockLen/srcStride 单位为字节：n2Size==1 时权重区连续整块搬；否则逐行搬，
        // srcStride 为行间跳过的字节数（跨过其余 head 的权重）
        DataCopyExtParams inputCopy;
        inputCopy.blockCount = tilingData.n2Size == 1 ? 1 : static_cast<uint16_t>(validRows);
        inputCopy.blockLen = tilingData.n2Size == 1 ? validBytes : rowBytes;
        inputCopy.srcStride = tilingData.n2Size == 1 ? 0 : static_cast<uint32_t>((qHeadNum - N) * sizeof(half));
        inputCopy.dstStride = 0;
        inputCopy.rsv = 0;
        DataCopyPad(weightUb, weightGm[sourceOffset], inputCopy, pad);
        DataCopyPad(scaleUb, scaleGm[sourceOffset], inputCopy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if (initializePadding) {
            InitWeightGatherIndices(weightIndicesUb);
            // Gather 的补齐元素始终读取这段对齐零块，它位于最多 384 个 half 的 Mul 目标区之外
            Duplicate(productUb[s1Tile * N], static_cast<half>(0.0f),
                      WeightPreprocessUbLayout::GATHER_PADDING_HALF_ELEMENTS);
        }
        if (validRows < s1Tile) {
            Duplicate(productUb, static_cast<half>(0.0f), s1Tile * N);
            PipeBarrier<PIPE_V>();
        }
        Mul(productUb, weightUb, scaleUb, validRows * N);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        PipeBarrier<PIPE_V>();
        // 把 [16, 24] 扩展为 [16, 40] 时保留每个 half 的全部比特；
        // 整块连续写 GM 也避免零区与数据区的 DMA 相互交叠
        // Gather 参数：相对 productUb 基地址偏移 0，元素数 = s1Tile × weightStride 个 half
        // （以 uint32 计为一半）
        Gather(packedWeightUb.template ReinterpretCast<uint32_t>(), productUb.template ReinterpretCast<uint32_t>(),
               weightIndicesUb.template ReinterpretCast<uint32_t>(), 0,
               static_cast<uint32_t>(s1Tile * weightStride / 2));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm, packedWeightUb, s1Tile * weightStride);
        // 在下一次 Gather 复用该 UB 之前释放 packed tile
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    struct BatchRunInfo {
        uint32_t batchIdx;
        uint32_t actualS1;
        uint32_t actualS2;
        uint64_t queryTokenPrefix;
        uint64_t keyTokenPrefix;
    };

    // 本 Mix group 从 metadata 读回的分核区间（S1 tile 维度，左闭右开）
    struct FastSplitCore {
        uint32_t coreEnable;
        uint32_t tileBegin;
        uint32_t tileEnd;
    };

    template <int N>
    __aicore__ inline void ProcessBatch(const BatchRunInfo& batchInfo, uint32_t begin, uint32_t end,
                                        const GlobalTensor<float>& localWorkspace,
                                        const GlobalTensor<half>& weightWorkspace)
    {
        using namespace QLIV2Fast;
        constexpr int s1Tile = KernelConfig<N>::S1_TILE;
        constexpr int sequences = KernelConfig<N>::SEQUENCES_PER_AIV;
        constexpr uint64_t weightWorkspaceOffset = KernelConfig<N>::GROUP_WORKSPACE_BYTES;
        constexpr uint32_t weightTileElements = s1Tile * KernelConfig<N>::WEIGHT_ROW_STRIDE;

        GlobalTensor<int8_t> batchQuery = queryGm[batchInfo.queryTokenPrefix * tilingData.n2Size * N * D_SIZE];
        GlobalTensor<uint32_t> output =
            indiceOutGm[batchInfo.queryTokenPrefix * tilingData.n2Size * tilingData.sparseCount];
        GlobalTensor<float> scoreRing = localWorkspace;
        GlobalTensor<half> scaledWeight = weightWorkspace[weightWorkspaceOffset / sizeof(half)];

        // Score tile 以 SCORE_ROWS 为粒度扫描；TopK 队列至少覆盖 TOP_K 行
        const int paddedS2 = Max(CeilDiv(static_cast<int>(batchInfo.actualS2), SCORE_ROWS) * SCORE_ROWS, TOP_K);
        KeyLoadContext keyContext{keyGm,
                                  blockTableGm,
                                  batchInfo.keyTokenPrefix,
                                  static_cast<uint64_t>(tilingData.kCacheStride),
                                  batchInfo.batchIdx,
                                  HardwareConstants::KEY_N2_INDEX,
                                  static_cast<uint32_t>(tilingData.n2Size),
                                  static_cast<uint32_t>(tilingData.blockSize),
                                  static_cast<uint32_t>(tilingData.maxBlockNumPerBatch),
                                  batchInfo.actualS2,
                                  PAGE_ATTENTION};

        if ASCEND_IS_AIC {
            InitCubeEvents();
            // key 尾块按 0 补齐
            SetLoadDataPaddingValue<int8_t>(0);
            // 算子契约：INT32 的 QK 结果按固定 1/1024 反量化后转 FP16
            SetFixpipePreQuantFlag(HardwareConstants::QK_DEQUANT_SCALE_BITS);

            const int initialS1 = begin * s1Tile;
            int initialValid = Min(static_cast<int>(batchInfo.actualS1) - initialS1, s1Tile);
            // 逻辑 K tile 0/1/2/3 初始占用物理 L1 槽 0/2/1/3，与下方静态乒乓流水对应
            LoadKeyToL1<CubePipelineConstants::PHYSICAL_KEY_SLOT_0>(keyContext, CubePipelineConstants::KEY_TILE_0);
            LoadQueryToL1<N>(batchQuery, begin, initialValid);
            CrossCoreWaitFlag<CrossCoreMessageId::MODE, PIPE_S>(CrossCoreMessageId::SCORE_FREE_OR_WEIGHT_READY);
            LoadWeightToL1<N>(scaledWeight, CubePipelineConstants::KEY_TILE_0, initialValid, true);
            LoadKeyToL1<CubePipelineConstants::PHYSICAL_KEY_SLOT_2>(keyContext, CubePipelineConstants::KEY_TILE_1);
            LoadKeyToL1<CubePipelineConstants::PHYSICAL_KEY_SLOT_1>(keyContext, CubePipelineConstants::KEY_TILE_2);
            LoadKeyToL1<CubePipelineConstants::PHYSICAL_KEY_SLOT_3>(keyContext, CubePipelineConstants::KEY_TILE_3);
            ActivateQueryWeight();
            PrefetchKeyToL0A<CubePipelineConstants::PHYSICAL_KEY_SLOT_0, CubePipelineConstants::KEY_L0_PING>();

            int scoreHead = 0;
            int scoreFreeWaits = 0;
            for (uint32_t tile = begin; tile + 1 < end; ++tile) {
                const int nextS1 = static_cast<int>(tile + 1) * s1Tile;
                int nextValid = Min(static_cast<int>(batchInfo.actualS1) - nextS1, s1Tile);
                LoadQueryToL1<N>(batchQuery, tile + 1, nextValid);
                LoadWeightToL1<N>(scaledWeight, tile + 1 - begin, nextValid, false);
                ProcessS1Tile<true>(keyContext, paddedS2, scoreRing, scoreHead, scoreFreeWaits);
                ReleaseQueryWeight();
                ActivateQueryWeight();
            }
            ProcessS1Tile<false>(keyContext, paddedS2, scoreRing, scoreHead, scoreFreeWaits);
            ReleaseQueryWeight();
            DrainScoreSlots(scoreHead, scoreFreeWaits);
            CrossCoreWaitFlag<CrossCoreMessageId::MODE, PIPE_S>(CrossCoreMessageId::VECTOR_DONE);
            NotifyCubeRelease();
            FinalizeCubeEvents();
        }

        if ASCEND_IS_AIV {
            SetAtomicNone();
            SetMaskNorm();
            ResetMask();

            const int subblock = static_cast<int>(GetSubBlockIdx());
            GlobalTensor<float> topkWorkspace =
                localWorkspace[(SCORE_RING_BYTES + subblock * KernelConfig<N>::TOPK_WORKSPACE_BYTES_PER_AIV) /
                               sizeof(float)];
            int scoreTail = 0;
            if (subblock == 0) {
                for (uint32_t tile = begin; tile < end; ++tile) {
                    const int tileS1 = static_cast<int>(tile) * s1Tile;
                    int tileValidRows = Min(static_cast<int>(batchInfo.actualS1) - tileS1, s1Tile);
                    PrepareScaledWeight<N>(batchInfo.queryTokenPrefix, tile, tileValidRows,
                                           scaledWeight[static_cast<uint64_t>(tile - begin) * weightTileElements],
                                           tile == begin);
                }
            }
            // 两个 AIV 都参与 mode 2 通知：Vec0 在 Weight 预处理完成后到达，Vec1 立即到达
            NotifyWeightReady();
            for (uint32_t tile = begin; tile < end; ++tile) {
                const int tileS1 = static_cast<int>(tile) * s1Tile;
                int tileValidRows = Min(static_cast<int>(batchInfo.actualS1) - tileS1, s1Tile);
                int validRows =
                    Max(0, Min(static_cast<int>(batchInfo.actualS1) - tileS1 - subblock * sequences, sequences));
                VectorPostContext postContext{keyScaleGm,
                                              blockTableGm,
                                              batchInfo.keyTokenPrefix,
                                              static_cast<uint64_t>(tilingData.kScaleStride),
                                              batchInfo.batchIdx,
                                              HardwareConstants::KEY_N2_INDEX,
                                              static_cast<uint32_t>(tilingData.n2Size),
                                              static_cast<uint32_t>(tilingData.blockSize),
                                              static_cast<uint32_t>(tilingData.maxBlockNumPerBatch),
                                              batchInfo.actualS1,
                                              batchInfo.actualS2,
                                              static_cast<uint32_t>(tileS1),
                                              static_cast<uint32_t>(tileValidRows),
                                              static_cast<uint32_t>(subblock),
                                              PAGE_ATTENTION,
                                              tilingData.sparseMode == SPARSE_MODE_LOWER};
                RunVectorTopk<N>(scoreRing,
                                 output[static_cast<uint64_t>(tileS1) * tilingData.sparseCount +
                                        static_cast<uint64_t>(subblock * sequences) * tilingData.sparseCount],
                                 topkWorkspace, paddedS2, static_cast<int>(tilingData.sparseCount), validRows, subblock,
                                 scoreTail, postContext);
            }
            PipeBarrier<PIPE_MTE3>();
            if (scoreTail % SCORE_SYNC_PERIOD != 0) {
                NotifyScoreFree();
            }
            NotifyVectorDone();
            CrossCoreWaitFlag<CrossCoreMessageId::MODE, PIPE_S>(CrossCoreMessageId::CUBE_RELEASE);
        }
    }

    template <int N>
    __aicore__ inline void ProcessImpl()
    {
        using namespace QLIV2Fast;
        constexpr uint64_t s1Tile = KernelConfig<N>::S1_TILE;

        // 分核与 slow 模板一致：区间由 metadata AICPU 算子写入，本核只读回
        // [tileBegin, tileEnd)（S1 tile 维度，左闭右开）
        if ((splitCore.coreEnable == 0U) || (splitCore.tileBegin >= splitCore.tileEnd)) {
            return;
        }
        const GlobalTensor<int32_t>& actualSeqLengthsQGm = actualSeqLengthsGmQ;
        const GlobalTensor<int32_t>& actualSeqLengthsKGm = actualSeqLengthsGm;
        // TND 下 cu_seqlens_q 的最后一个元素即物理 T1；v2 的 TilingData.s1Size 在 TND 下为 0，
        // 不能直接作为序列读取的物理上限
        const uint32_t physicalT1 = static_cast<uint32_t>(actualSeqLengthsQGm.GetValue(tilingData.bSize));

        const uint64_t groupBegin = splitCore.tileBegin;
        const uint64_t groupEnd = splitCore.tileEnd;
        const uint64_t groupByteOffset = static_cast<uint64_t>(groupIndex) * tilingData.workspaceGroupStride;
        GlobalTensor<float> localWorkspace = workspaceFp32Gm[groupByteOffset / sizeof(float)];
        GlobalTensor<half> weightWorkspace = workspaceFp16Gm[groupByteOffset / sizeof(half)];

        // 定位本核 tile 区间覆盖的 batch 区间
        uint64_t tilePrefix = 0;
        uint32_t queryPrefix = 0;
        uint32_t keyPrefix = 0;
        for (uint32_t batchIdx = 0; (batchIdx < tilingData.bSize) && (tilePrefix < groupEnd); ++batchIdx) {
            const uint32_t queryEnd = ReadAccumSeqEnd(actualSeqLengthsQGm, batchIdx, queryPrefix, physicalT1);
            const uint32_t actualS1 = queryEnd - queryPrefix;
            const uint64_t batchTileCount = (static_cast<uint64_t>(actualS1) + s1Tile - 1) / s1Tile;
            const uint64_t batchTileEnd = tilePrefix + batchTileCount;

            uint32_t actualS2 = ReadClampedSequenceLength(actualSeqLengthsKGm, batchIdx, tilingData.s2Size);
            uint32_t nextKeyPrefix = keyPrefix;
            if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
                const uint32_t keyEnd = ReadAccumSeqEnd(actualSeqLengthsKGm, batchIdx, keyPrefix, tilingData.s2Size);
                actualS2 = keyEnd - keyPrefix;
                nextKeyPrefix = keyEnd;
            }

            const uint64_t intersectBegin = Max(groupBegin, tilePrefix);
            const uint64_t intersectEnd = Min(groupEnd, batchTileEnd);
            if (intersectBegin < intersectEnd) {
                BatchRunInfo batchInfo{batchIdx, actualS1, actualS2, queryPrefix, keyPrefix};
                ProcessBatch<N>(batchInfo, static_cast<uint32_t>(intersectBegin - tilePrefix),
                                static_cast<uint32_t>(intersectEnd - tilePrefix), localWorkspace, weightWorkspace);
            }
            tilePrefix = batchTileEnd;
            queryPrefix = queryEnd;
            keyPrefix = nextKeyPrefix;
        }
    }

    GlobalTensor<int8_t> queryGm;
    GlobalTensor<int8_t> keyGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> queryScaleGm;
    GlobalTensor<half> keyScaleGm;
    GlobalTensor<int32_t> actualSeqLengthsGmQ;
    GlobalTensor<int32_t> actualSeqLengthsGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<uint32_t> indiceOutGm;
    GlobalTensor<float> workspaceFp32Gm;
    GlobalTensor<half> workspaceFp16Gm;
    // 本核信息：Mix group 索引与从 metadata 读回的分核区间（S1 tile 维度，左闭右开）
    uint32_t groupIndex = 0U;
    FastSplitCore splitCore{0U, 0U, 0U};
    QLIV2FastTilingData tilingData{};
};

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH22_H
