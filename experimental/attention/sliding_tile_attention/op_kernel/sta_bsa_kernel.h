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
 * \file sta_bsa_kernel.h
 * \brief SlidingTileAttention block-sparse kernel for Atlas A2.
 *
 * Derived from the ops-transformer BlockSparseAttention A2 kernel (arch22, BNSD, fp32 softmax) and reusing its
 * attn_infra modules. Differences:
 *  - there is no block mask: the kernel takes each head's window (in tiles) and the tile grid, and the vector cores
 *    write the sorted KV block list of every (head, query block) once for all batches (BuildSelectLists), so the
 *    work and memory grow with the selected blocks instead of blocks^2;
 *  - query and key/value lengths are uniform and come from the operator's own tiling data; the KV length is the
 *    valid length (image tokens + textLength) so reserved text slots never act as keys;
 *  - head dimensions 32 and 48 are accepted, and the host zero-pads narrower heads;
 *  - a query task is a whole 384-token tile instead of 128 rows. The cube computes Q*K^T for it in 128-row
 *    chunks (one QK block per chunk), while the vector cores process the 384 rows of a KV stack at once, so the
 *    per-stack vector work (online softmax bookkeeping, output rescaling, cross-core handshakes) is shared by
 *    three times as many (q, k) pairs;
 *  - stage 2 of a KV stack (PV matmul, output rescaling) runs stage2Delay stacks after its stage 1 (QK matmul,
 *    online softmax) also across task boundaries, so the two stages keep overlapping when tasks have few stacks;
 *  - the QK/PV blocks copy L0C to GM without a fix-pipe barrier per tile, and the QK blocks load a query chunk as
 *    one ND matrix instead of one per token (StaTileCopy);
 *  - with a large L2 cache, the vector cores read the queries of upcoming tasks into L2 before the cube loads them
 *    (WarmQueries).
 */

#ifndef STA_BSA_KERNEL_H
#define STA_BSA_KERNEL_H

#include "../../block_sparse_attention/op_kernel/block_sparse_attention_kernel_common.hpp"
#include "sta_epilogue.h"
#include "sta_tiling_data.h"

using namespace NpuArch;
using namespace BsaKernelCommon;

namespace StaBsa {

// Query rows of one task; the host sets blockShapeX to the same value.
constexpr uint32_t Q_TASK_ROWS = 384;
constexpr uint32_t Q_CHUNK_NUM = Q_TASK_ROWS / Q_TILE_CEIL;
// Score elements of one KV stack. Must match WORKSPACE_BLOCK_ELEMS in sliding_tile_attention_tiling.cpp.
constexpr uint64_t STACK_SCORE_ELEMS = static_cast<uint64_t>(Q_TASK_ROWS) * MAX_KV_STACK_LEN;
// Largest number of KV stacks between a stack's stage 1 and its stage 2 (the tiling chooses the delay by L2 size);
// each in-flight stack owns one workspace/UB slot. STACK_SLOT_MAX must match WORKSPACE_STAGES in
// sliding_tile_attention_tiling.cpp.
constexpr uint32_t STAGE2_DELAY_MAX = 3;
constexpr uint32_t STACK_SLOT_MAX = STAGE2_DELAY_MAX + 1;
static_assert(STACK_SLOT_MAX == NpuArch::Epilogue::StaUbLayout::STACK_SLOT_MAX,
              "the epilogues keep per-stack row values for every KV-stack slot");
// MTE1_MTE2 event (set once by the kernel before the first task): MTE1 has copied the current task's queries out of
// the QK L1 buffers, so MTE2 may load the next task's queries there.
constexpr uint32_t QUERY_L1_FREE_EVENT = EVENT_ID5;
// Tasks (of the same core) between the task being opened and the task whose queries the vector cores read ahead.
constexpr uint32_t QUERY_WARM_AHEAD = 2;

// Row-major L0C -> GM copy of the QK scores and PV outputs. The upstream copy calls Fixpipe, which issues the same
// intrinsic but drains the fix pipe (PipeBarrier) before every tile, so the cube scalar cannot queue the next tiles
// while earlier scores are still being written. DataCopy has no barrier; its nz->nd register is set to the value
// Fixpipe uses (one ND block, zero ND strides) before every copy, so queued copies always see the same setting.
template <class ElementAccumulator_, class ElementDst_>
struct CopyL0CToGmNoBarrier {
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = layout::zN;
    using LayoutDst = layout::RowMajor;
    static constexpr auto quantPre = Gemm::Tile::CopyL0CToDstQuantMode<Arch::AtlasA2, ElementSrc, ElementDst,
                                                                       Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementDst> const& dst,
                                      AscendC::LocalTensor<ElementSrc> const& src, LayoutDst const& dstLayout,
                                      LayoutSrc const& srcLayout, uint8_t unitFlag = 0)
    {
        static constexpr uint16_t ND_NUM = 1;
        static constexpr uint16_t ND_STRIDE = 0;
        AscendC::SetFixpipeNz2ndFlag(ND_NUM, ND_STRIDE, ND_STRIDE);
        AscendC::DataCopyCO12DstParams copyParams(
            static_cast<uint16_t>(dstLayout.shape(1)), static_cast<uint16_t>(dstLayout.shape(0)),
            static_cast<uint32_t>(dstLayout.stride(0)),
            static_cast<uint16_t>(srcLayout.stride(3) / srcLayout.stride(0)), quantPre, 0, false, true);
        copyParams.unitFlag = unitFlag;
        AscendC::DataCopy(dst, src, copyParams);
    }
};

// Row-major GM -> L1 copy of the A operand (queries of the QK blocks, probabilities of the PV block). The upstream QK
// block loads a query chunk as rowNum ND matrices of one row each (one per token, so that grouped heads stay
// together) and the DMA converts every matrix to NZ separately. With one head per task the chunk is copied as a
// single ND matrix of rowNum rows into the same L1 layout, which costs a fraction of the time.
template <class ArchTag, class AType>
struct CopyGmToL1OneMatrix : Gemm::Tile::CopyGmToL1<ArchTag, AType> {
    using Base = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using Element = typename AType::Element;
    using LayoutDst = typename Base::LayoutDst;
    using LayoutSrc = typename Base::LayoutSrc;
    using Base::operator();

    __aicore__ inline void operator()(AscendC::LocalTensor<Element> const& dstTensor,
                                      AscendC::GlobalTensor<Element> const& srcTensor, LayoutDst const& layoutDst,
                                      LayoutSrc const& layoutSrc, uint32_t ndNum, uint32_t srcNdMatrixStride,
                                      uint32_t dstNzNStride, uint32_t dstNzMatrixStride, uint32_t dstNzC0Stride)
    {
        if (layoutSrc.shape(0) == 1) {
            Base::operator()(dstTensor, srcTensor, layoutDst, LayoutSrc(ndNum, layoutSrc.shape(1), srcNdMatrixStride));
            return;
        }
        Base::operator()(dstTensor, srcTensor, layoutDst, layoutSrc, ndNum, srcNdMatrixStride, dstNzNStride,
                         dstNzMatrixStride, dstNzC0Stride);
    }
};

// The upstream A2 tile copies with the A operand loaded by CopyGmToL1OneMatrix and the L0C -> GM copy replaced by
// CopyL0CToGmNoBarrier.
template <class ArchTag, class AType, class BType, class CType>
struct StaTileCopy : Gemm::Tile::TileCopy<ArchTag, AType, BType, CType> {
    using CopyGmToL1A = CopyGmToL1OneMatrix<ArchTag, AType>;
    using CopyL0CToGm =
        CopyL0CToGmNoBarrier<typename Gemm::Tile::TileCopy<ArchTag, AType, BType, CType>::ElementAccumulator,
                             typename CType::Element>;
};

// int32 KV block indices per C0 (the select lists are written in whole C0s).
constexpr uint32_t IDX_PER_C0 = 8;
// Window values per head in the windows input: tiles along T, H and W.
constexpr uint32_t WINDOW_DIMS = 3;
// Heads whose list lengths every core keeps in a local table; more heads read their window per task.
constexpr uint32_t HEAD_CACHE_NUM = 64;

struct StaBsaKernelParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR windows;
    GM_ADDR o;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO>
class StaBsaKernel {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;
    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;
    using ElementO = typename EpilogueRescaleO::ElementOutput;
    using LayoutO = typename EpilogueRescaleO::LayoutOutput;
    using ElementOTmp = typename EpilogueRescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;
    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using LayoutLse = typename EpilogueRescaleO::LayoutLse;
    using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

    __aicore__ inline StaBsaKernel() {}

    // Window of one head in tiles, limited to the grid.
    struct HeadWindow {
        uint32_t t{1};
        uint32_t h{1};
        uint32_t w{1};
    };

    __aicore__ inline static uint32_t ClampWindow(int32_t window, uint32_t grid)
    {
        if (window < 1) {
            return 1;
        }
        return (static_cast<uint32_t>(window) < grid) ? static_cast<uint32_t>(window) : grid;
    }

    // First tile of a window of `window` tiles centered on `coord` and shifted inward at the grid edges (FastVideo).
    __aicore__ inline static uint32_t WindowStart(uint32_t coord, uint32_t window, uint32_t grid)
    {
        const uint32_t half = window / 2;
        const uint32_t start = (coord > half) ? (coord - half) : 0;
        const uint32_t maxStart = grid - window;
        return (start < maxStart) ? start : maxStart;
    }

    __aicore__ inline HeadWindow ReadWindow(const AscendC::GlobalTensor<int8_t>& gWindows, uint32_t head) const
    {
        const uint64_t base = static_cast<uint64_t>(head) * WINDOW_DIMS;
        HeadWindow window;
        window.t = ClampWindow(gWindows.GetValue(base), tilesT);
        window.h = ClampWindow(gWindows.GetValue(base + 1), tilesH);
        window.w = ClampWindow(gWindows.GetValue(base + 2), tilesW);
        return window;
    }

    // KV blocks in the list of an image query block of the head.
    __aicore__ inline uint32_t ImageSelectNum(const AscendC::GlobalTensor<int8_t>& gWindows, uint32_t head) const
    {
        if (head < HEAD_CACHE_NUM) {
            return headSelectNum[head];
        }
        const HeadWindow window = ReadWindow(gWindows, head);
        return window.t * window.h * window.w + textSelect;
    }

    // Streams int32 values to consecutive select-list entries through two ping-pong UB buffers; every flush but the
    // last is LIST_CHUNK entries and the stream always ends on a whole C0.
    __aicore__ inline void ListWriterStart(uint64_t gmOffset)
    {
        listUb[0] = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        listUb[1] = resource.ubBuf.template GetBufferByByte<int32_t>(LIST_CHUNK * sizeof(int32_t));
        listFill = 0;
        listRowFill = 0;
        listBuf = 0;
        listGmOffset = gmOffset;
        // An MTE3_S event is set while its buffer is free; each flush consumes the event of the buffer it fills next.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    }

    __aicore__ inline void ListWriterFlush(const AscendC::GlobalTensor<int32_t>& gSelectIdx)
    {
        if (listFill == 0) {
            return;
        }
        AscendC::DataCopy(gSelectIdx[listGmOffset], listUb[listBuf],
                          AscendC::DataCopyParams(1, listFill / IDX_PER_C0, 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(listBuf);
        listGmOffset += listFill;
        listFill = 0;
        listBuf = 1 - listBuf;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(listBuf);
    }

    __aicore__ inline void ListWriterPush(const AscendC::GlobalTensor<int32_t>& gSelectIdx, int32_t value)
    {
        listUb[listBuf].SetValue(listFill, value);
        ++listFill;
        ++listRowFill;
        if (listFill == LIST_CHUNK) {
            ListWriterFlush(gSelectIdx);
        }
    }

    // Fills the current list with zeros up to a multiple of `multiple` entries (never read) and starts a new list.
    __aicore__ inline void ListWriterPad(const AscendC::GlobalTensor<int32_t>& gSelectIdx, uint32_t multiple)
    {
        while (listRowFill % multiple != 0) {
            ListWriterPush(gSelectIdx, 0);
        }
        listRowFill = 0;
    }

    // Writes the last entries and waits until every store has finished, so the lists can be read right away.
    __aicore__ inline void ListWriterFinish(const AscendC::GlobalTensor<int32_t>& gSelectIdx)
    {
        ListWriterFlush(gSelectIdx);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(1 - listBuf);
    }

    // Vector sub-cores, before the cube starts: write the KV block list of every image query block. The list of
    // (head, block) starts at entry (head * imageBlocks + block) * selectCap and holds, in ascending order, the tiles
    // of the head's window around the query tile, then the text block when text keys exist. The last active
    // sub-core also writes the list of the text block (every block) right after the image lists. The lists are the
    // ones a [N, blocks, blocks] block mask of the same windows selects.
    __aicore__ inline void BuildSelectLists(const AscendC::GlobalTensor<int8_t>& gWindows,
                                            const AscendC::GlobalTensor<int32_t>& gSelectIdx)
    {
        const uint32_t subCoreIdx = AscendC::GetBlockIdx();
        if (subCoreIdx >= listSubCoreNum) {
            return;
        }
        const uint64_t rowBegin = static_cast<uint64_t>(subCoreIdx) * listRowsPerSubCore;
        const uint64_t rowEnd =
            (listRowNum - rowBegin < listRowsPerSubCore) ? listRowNum : (rowBegin + listRowsPerSubCore);
        const uint32_t tilesHW = tilesH * tilesW;
        ListWriterStart(rowBegin * selectCap);
        uint32_t windowHead = static_cast<uint32_t>(rowBegin / imageBlocks);
        HeadWindow window = ReadWindow(gWindows, windowHead);
        for (uint64_t row = rowBegin; row < rowEnd; ++row) {
            const uint32_t head = static_cast<uint32_t>(row / imageBlocks);
            const uint32_t block = static_cast<uint32_t>(row - static_cast<uint64_t>(head) * imageBlocks);
            if (head != windowHead) {
                window = ReadWindow(gWindows, head);
                windowHead = head;
            }
            const uint32_t startT = WindowStart(block / tilesHW, window.t, tilesT);
            const uint32_t startH = WindowStart((block / tilesW) % tilesH, window.h, tilesH);
            const uint32_t startW = WindowStart(block % tilesW, window.w, tilesW);
            for (uint32_t tileT = startT; tileT < startT + window.t; ++tileT) {
                for (uint32_t tileH = startH; tileH < startH + window.h; ++tileH) {
                    const uint32_t lineBase = (tileT * tilesH + tileH) * tilesW;
                    for (uint32_t tileW = startW; tileW < startW + window.w; ++tileW) {
                        ListWriterPush(gSelectIdx, static_cast<int32_t>(lineBase + tileW));
                    }
                }
            }
            if (textSelect != 0) {
                ListWriterPush(gSelectIdx, static_cast<int32_t>(imageBlocks));
            }
            ListWriterPad(gSelectIdx, selectCap);
        }
        if (rowEnd == listRowNum && blockNum > imageBlocks) {
            for (uint32_t block = 0; block < blockNum; ++block) {
                ListWriterPush(gSelectIdx, static_cast<int32_t>(block));
            }
            ListWriterPad(gSelectIdx, IDX_PER_C0);
        }
        ListWriterFinish(gSelectIdx);
    }

    // Vector sub-cores, before the cube starts: for zero-padded heads, fill the padded value columns of the private
    // copy with ones (see the tiling for the reasons).
    __aicore__ inline void PreparePaddedInputs(AscendC::GlobalTensor<ElementV> gV, uint32_t embed, uint32_t sumColumn,
                                               uint64_t totalRows, uint64_t rowsPerSubCore)
    {
        static constexpr uint32_t CHUNK_ROWS = 256; // PREPASS_ROWS in the host tiling
        static constexpr uint32_t ONES_UB_OFFSET = 48 * 1024;
        static constexpr uint32_t ONES_FLOAT_UB_OFFSET = 64 * 1024;
        if (sumColumn == 0) {
            return;
        }
        const uint64_t rowBegin = AscendC::GetBlockIdx() * rowsPerSubCore;
        if (rowBegin >= totalRows) {
            return;
        }
        const uint64_t rowEnd = (totalRows - rowBegin < rowsPerSubCore) ? totalRows : rowBegin + rowsPerSubCore;
        const uint32_t rowC0Num = embed * sizeof(ElementQ) / BYTE_PER_C0;
        const uint32_t padC0Num = (embed - sumColumn) * sizeof(ElementV) / BYTE_PER_C0;
        const AscendC::RoundMode toHalfMode =
            std::is_same<ElementQ, bfloat16_t>::value ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
        AscendC::LocalTensor<ElementV> onesUb = resource.ubBuf.template GetBufferByByte<ElementV>(ONES_UB_OFFSET);
        AscendC::LocalTensor<float> onesFloatUb = resource.ubBuf.template GetBufferByByte<float>(ONES_FLOAT_UB_OFFSET);
        // Flush the vector pipeline before the UB scratch below is used.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();
        // The vector ops below apply the current vector mask; start from a full one.
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        const uint32_t onesLines = CHUNK_ROWS * padC0Num * BYTE_PER_C0 / sizeof(ElementV) / FLOAT_VECTOR_SIZE;
        AscendC::Duplicate(onesFloatUb, 1.0f, onesLines * FLOAT_VECTOR_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<ElementV, float, false>(onesUb, onesFloatUb, toHalfMode, (uint64_t)0, onesLines,
                                              AscendC::UnaryRepeatParams(1, 1, 4, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint64_t row = rowBegin; row < rowEnd; row += CHUNK_ROWS) {
            const uint32_t rows = (rowEnd - row < CHUNK_ROWS) ? static_cast<uint32_t>(rowEnd - row) : CHUNK_ROWS;
            AscendC::DataCopy(gV[row * embed + sumColumn], onesUb,
                              AscendC::DataCopyParams(rows, padC0Num, 0, rowC0Num - padC0Num));
        }
        // The cube reads the rewritten values right after the cross-core sync: finish every store first.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Vector sub-cores: reads one C0 of every L2 line of the queries of task taskIdx into a UB scratch area, so the
    // cube's later load of those queries reads warm lines (the queries are read once, so their lines are cold).
    __aicore__ inline void WarmQueries(AscendC::GlobalTensor<ElementQ> gQ, uint32_t taskIdx, uint32_t tasksPerBatch,
                                       uint32_t heads, uint64_t strideB, uint64_t strideN, uint32_t qBlockX,
                                       uint32_t xBlockNum, uint32_t xTailNum, uint32_t embed)
    {
        static constexpr uint32_t L2_LINE_BYTES = 256;
        const uint32_t batch = taskIdx / tasksPerBatch;
        const uint32_t taskInBatch = taskIdx - batch * tasksPerBatch;
        const uint32_t tile = taskInBatch / heads;
        const uint32_t head = taskInBatch - tile * heads;
        const uint64_t offsetQ = batch * strideB + head * strideN + static_cast<uint64_t>(tile) * qBlockX * embed;
        const uint32_t rows = (tile == xBlockNum) ? xTailNum : qBlockX;
        const uint32_t lines = AscendC::Std::min(static_cast<uint32_t>(rows * embed * sizeof(ElementQ) / L2_LINE_BYTES),
                                                 Epilogue::StaUbLayout::QW_C0_NUM);
        if (lines == 0) {
            return;
        }
        AscendC::LocalTensor<ElementQ> warmUb =
            resource.ubBuf.template GetBufferByByte<ElementQ>(Epilogue::StaUbLayout::QW);
        AscendC::DataCopy(warmUb, gQ[offsetQ], AscendC::DataCopyParams(lines, 1, L2_LINE_BYTES / BYTE_PER_C0 - 1, 0));
    }

    __aicore__ inline void operator()(StaBsaKernelParams const& params)
    {
        __gm__ optiling::SlidingTileAttentionTilingData* td =
            reinterpret_cast<__gm__ optiling::SlidingTileAttentionTilingData*>(params.tiling);
        const uint64_t mm1OutSize = td->mm1OutSize;
        const uint64_t smOnlineOutSize = td->smOnlineOutSize;
        const uint64_t mm2OutSize = td->mm2OutSize;
        const uint64_t updateSize = td->updateSize;
        const uint32_t heads = static_cast<uint32_t>(td->numHeads);
        const uint32_t embed = static_cast<uint32_t>(td->embeddingSize);
        uint32_t pagedBlockSize = static_cast<uint32_t>(td->kvBlockSize);
        const uint32_t firstBatchTaskNum = static_cast<uint32_t>(td->firstBatchTaskNum);
        const uint32_t totalTaskNum = static_cast<uint32_t>(td->totalTaskNum);
        // The host stores the IEEE-754 bits of the float scale in the low 32 bits.
        union {
            uint32_t bits;
            float value;
        } scaleCast;
        scaleCast.bits = static_cast<uint32_t>(td->scaleValueBits);
        const ElementS scaleValue = static_cast<ElementS>(scaleCast.value);
        const uint32_t sumColumn = static_cast<uint32_t>(td->sumColumn);
        // Out-of-range values fall back to the largest delay the workspace and UB layout provide for.
        const uint32_t stage2Delay = (td->stage2Delay >= 1 && td->stage2Delay <= STAGE2_DELAY_MAX) ?
                                         static_cast<uint32_t>(td->stage2Delay) :
                                         STAGE2_DELAY_MAX;
        const uint32_t slotNum = stage2Delay + 1;
        const bool warmQueries = td->warmQueries != 0;
        tilesT = static_cast<uint32_t>(td->tilesT);
        tilesH = static_cast<uint32_t>(td->tilesH);
        tilesW = static_cast<uint32_t>(td->tilesW);
        imageBlocks = static_cast<uint32_t>(td->imageBlocks);
        blockNum = static_cast<uint32_t>(td->blockNum);
        textSelect = static_cast<uint32_t>(td->textSelect);
        selectCap = static_cast<uint32_t>(td->selectCap);
        listRowNum = td->listRowNum;
        listRowsPerSubCore = td->listRowsPerSubCore;
        listSubCoreNum = static_cast<uint32_t>(td->listSubCoreNum);
        uint32_t qBlockX = static_cast<uint32_t>(td->blockShapeX);
        uint32_t qBlockY = static_cast<uint32_t>(td->blockShapeY);
        const uint32_t seqLen = static_cast<uint32_t>(td->seqLen);
        uint32_t kvSeqlen = static_cast<uint32_t>(td->validKvSeqLen);

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t*)params.workspace);
        AscendC::GlobalTensor<int32_t> gSelectIdx;
        gSelectIdx.SetGlobalBuffer(
            (__gm__ int32_t*)(params.workspace + mm1OutSize + smOnlineOutSize + mm2OutSize + updateSize));
        AscendC::GlobalTensor<int8_t> gWindows;
        gWindows.SetGlobalBuffer((__gm__ int8_t*)params.windows);
        const uint32_t cachedHeads = (heads < HEAD_CACHE_NUM) ? heads : HEAD_CACHE_NUM;
        for (uint32_t head = 0; head < cachedHeads; ++head) {
            const HeadWindow window = ReadWindow(gWindows, head);
            headSelectNum[head] = window.t * window.h * window.w + textSelect;
        }
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.o);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)params.workspace);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)(params.workspace + mm1OutSize));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workspace + mm1OutSize + smOnlineOutSize));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workspace + mm1OutSize + smOnlineOutSize + mm2OutSize));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
#ifdef __DAV_C220_VEC__
        BuildSelectLists(gWindows, gSelectIdx);
        PreparePaddedInputs(gV, embed, sumColumn, td->totalRows, td->prepassRowsPerSubCore);
#endif
        resource.pipe.Reset();
        AscendC::SyncAll<false>();

#ifdef __DAV_C220_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

        // One QK block per 128-row query chunk, each with its own L1 region, so a task loads Q only once.
        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) +
            BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;
        BlockMmadQK blockMmadQK0(resource, 0);
        BlockMmadQK blockMmadQK1(resource, L1_QK_SIZE);
        BlockMmadQK blockMmadQK2(resource, 2 * L1_QK_SIZE);
        BlockMmadQK* blockMmadQK[Q_CHUNK_NUM] = {&blockMmadQK0, &blockMmadQK1, &blockMmadQK2};
        BlockMmadPV blockMmadPV(resource, Q_CHUNK_NUM * L1_QK_SIZE);
#endif

#ifdef __DAV_C220_VEC__
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue, sumColumn != 0, slotNum);
        EpilogueRescaleO epilogueRescaleO(resource, sumColumn);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
        // BNSD layout: strides of one batch and one head; Q and KV share the tensor length seqLen.
        const uint64_t strideB = static_cast<uint64_t>(heads) * seqLen * embed;
        const uint64_t strideN = static_cast<uint64_t>(seqLen) * embed;
        const uint64_t strideS = embed;
        const uint32_t embedRound = AlignUp<uint32_t>(embed, BLOCK_SIZE);
        const uint32_t qNBlockSize = 1;
        const uint32_t qSeqlen = seqLen;
        const uint32_t xBlockNum = qSeqlen / qBlockX;
        const uint32_t xTailNum = qSeqlen - xBlockNum * qBlockX;
        const uint32_t blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize;
        const uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
        const uint32_t kvYBlockNum = (kvSeqlen + qBlockY - 1) / qBlockY;
        const uint64_t coreScoreOffset = coreIdx * STACK_SCORE_ELEMS * STACK_SLOT_MAX;
        const uint64_t gmOffsetUpdate = coreIdx * STACK_SCORE_ELEMS;
        uint32_t strideKV = static_cast<uint32_t>(strideS);
#ifdef __DAV_C220_CUBE__
        LayoutK layoutKTemp(strideS, stackSeqTilePad);
        LayoutV layoutVTemp(stackSeqTilePad, strideS);
        uint32_t singleGroupHeads = qNBlockSize;
#endif

        // Stage-1 stacks whose stage 2 has not been issued yet, oldest first.
        StackJob pending[STAGE2_DELAY_MAX];
        uint32_t pendingHead = 0;
        uint32_t pendingNum = 0;
        uint32_t stackSeq = 0;
        QueryTask task;
        uint32_t nextTaskIdx = coreIdx;
        uint32_t nextKvSIdx = 0;
        while (true) {
            // Open tasks until one has a KV stack left for stage 1.
            while (nextKvSIdx >= task.kvSLoopNumTotal && nextTaskIdx < totalTaskNum) {
                const uint32_t taskIdx = nextTaskIdx;
                nextTaskIdx += coreNum;
                nextKvSIdx = 0;
                task.kvSLoopNumTotal = 0;
#ifdef __DAV_C220_VEC__
                // The cube loads the queries of that task a few stacks from now; read them into L2 first.
                const uint32_t warmTaskIdx = taskIdx + QUERY_WARM_AHEAD * coreNum;
                if (warmQueries && AscendC::GetSubBlockIdx() == 0 && warmTaskIdx < totalTaskNum) {
                    WarmQueries(gQ, warmTaskIdx, firstBatchTaskNum, heads, strideB, strideN, qBlockX, xBlockNum,
                                xTailNum, embed);
                }
#endif
                const uint32_t curBatch = taskIdx / firstBatchTaskNum;
                // Tasks of one batch are ordered by [query tile, head].
                const uint32_t taskIdxCurBatch = taskIdx - curBatch * firstBatchTaskNum;
                const uint32_t qXIdx = taskIdxCurBatch / heads;
                const uint32_t qHeadIdx = taskIdxCurBatch - qXIdx * heads;
                // The lists are shared by all batches: an image block's list is indexed by (head, block).
                if (qXIdx < imageBlocks) {
                    task.curSelectNum = ImageSelectNum(gWindows, qHeadIdx);
                    task.listOffset = (static_cast<uint64_t>(qHeadIdx) * imageBlocks + qXIdx) * selectCap;
                } else {
                    task.curSelectNum = blockNum;
                    task.listOffset = listRowNum * selectCap;
                }
                // The text block is a key block only when text keys exist: with textLength == 0 the lists of the
                // text query block end one block beyond the valid KV length.
                if (task.curSelectNum > kvYBlockNum) {
                    task.curSelectNum = kvYBlockNum;
                }
                if (task.curSelectNum == 0) {
                    continue;
                }
                const uint32_t lastSelectIdx =
                    static_cast<uint32_t>(gSelectIdx.GetValue(task.listOffset + task.curSelectNum - 1));
                task.noSkipKvS = (lastSelectIdx == kvYBlockNum - 1 && kvSeqlen % qBlockY != 0) ?
                                     qBlockY * (task.curSelectNum - 1) + kvSeqlen % qBlockY :
                                     qBlockY * task.curSelectNum;
                task.kvSLoopNumTotal = (task.noSkipKvS + pagedBlockSize - 1) / pagedBlockSize;
                const uint64_t bOffset = curBatch * strideB;
                task.gmOffsetK = bOffset + qHeadIdx * strideN;
                task.gmOffsetQ = task.gmOffsetK + static_cast<uint64_t>(qXIdx) * qBlockX * strideS;
                task.qSBlockSize = (qXIdx == xBlockNum) ? xTailNum : qBlockX;
                task.rowNum = task.qSBlockSize * qNBlockSize;
#ifdef __DAV_C220_CUBE__
                // Stage 2 of the previous task does not block this load, so wait for its last QK copies explicitly.
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_L1_FREE_EVENT);
                const uint32_t qChunkNum = (task.rowNum + Q_TILE_CEIL - 1) / Q_TILE_CEIL;
                for (uint32_t chunkIdx = 0; chunkIdx < qChunkNum; ++chunkIdx) {
                    uint32_t chunkRows = ChunkRows(task.rowNum, chunkIdx);
                    LayoutQ layoutQChunk(chunkRows, embed);
                    blockMmadQK[chunkIdx]->loadQGM(gQ[task.gmOffsetQ + chunkIdx * Q_TILE_CEIL * strideS], layoutQChunk,
                                                   chunkRows, singleGroupHeads, strideS);
                }
#endif
            }
            const bool haveStack = nextKvSIdx < task.kvSLoopNumTotal;
            StackJob job;
            if (haveStack) {
                job.task = task;
                job.kvSIdx = nextKvSIdx;
                const uint32_t restKv = task.noSkipKvS - nextKvSIdx * pagedBlockSize;
                job.stackSeqTile = (restKv < stackSeqTilePad) ? restKv : stackSeqTilePad;
                job.slot = stackSeq % slotNum;
                job.isFirst = (nextKvSIdx == 0) ? 1 : 0;
                job.isLast = (nextKvSIdx + blockStackNum >= task.kvSLoopNumTotal) ? 1 : 0;
                // Stage 1: QK matmul (cube) and online softmax (vector).
                const uint64_t gmOffsetS = coreScoreOffset + job.slot * STACK_SCORE_ELEMS;
#ifdef __DAV_C220_CUBE__
                uint32_t kvSIdx = job.kvSIdx;
                uint32_t kvSLoopNumTotal = task.kvSLoopNumTotal;
                uint32_t curSelectNum = task.curSelectNum;
                uint32_t kvYBlockNumVar = kvYBlockNum;
                const uint32_t qChunkNum = (task.rowNum + Q_TILE_CEIL - 1) / Q_TILE_CEIL;
                for (uint32_t chunkIdx = 0; chunkIdx < qChunkNum; ++chunkIdx) {
                    uint32_t chunkRows = ChunkRows(task.rowNum, chunkIdx);
                    uint64_t chunkOffset = static_cast<uint64_t>(chunkIdx) * Q_TILE_CEIL;
                    LayoutQ layoutQChunk(chunkRows, embed);
                    LayoutS layoutSChunk(chunkRows, job.stackSeqTile, stackSeqTilePad);
                    GemmCoord chunkShapeQK{chunkRows, job.stackSeqTile, embed};
                    (*blockMmadQK[chunkIdx])(gQ[task.gmOffsetQ + chunkOffset * strideS], gK[task.gmOffsetK],
                                             gS[gmOffsetS + chunkOffset * stackSeqTilePad], gBlockTable,
                                             gSelectIdx[task.listOffset], layoutQChunk, layoutKTemp, layoutSChunk,
                                             chunkShapeQK, kvSIdx, kvSLoopNumTotal, pagedBlockSize, strideKV, qBlockY,
                                             curSelectNum, kvYBlockNumVar, kvSeqlen);
                }
                if (job.isLast) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_L1_FREE_EVENT);
                }
                NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                GemmCoord actualBlockShapeQK{task.rowNum, job.stackSeqTile, embed};
                LayoutS layOutS(task.rowNum, job.stackSeqTile, stackSeqTilePad);
                LayoutP layOutP(task.rowNum, job.stackSeqTile, stackSeqTilePad);
                NpuArch::Arch::CrossCoreWaitFlag(qkReady);
                epilogueOnlineSoftmax(gP[gmOffsetS], gS[gmOffsetS], layOutP, layOutS, actualBlockShapeQK, job.isFirst,
                                      0, task.qSBlockSize, qNBlockSize, job.slot, softmaxReady);
#endif
            }
            if (pendingNum == stage2Delay || (!haveStack && pendingNum != 0)) {
                // Stage 2: PV matmul (cube) and output rescaling (vector) of the oldest pending stack.
                const StackJob& old = pending[pendingHead];
                const uint64_t gmOffsetOTmp = coreScoreOffset + old.slot * STACK_SCORE_ELEMS;
                GemmCoord actualBlockShapePV{old.task.rowNum, embed, old.stackSeqTile};
                LayoutOTmp layoutOTmp(old.task.rowNum, embed, embedRound);
#ifdef __DAV_C220_CUBE__
                LayoutP layoutPTemp(old.task.rowNum, old.stackSeqTile, stackSeqTilePad);
                uint32_t nowKvSIdx = old.kvSIdx;
                uint32_t oldKvSLoopNumTotal = old.task.kvSLoopNumTotal;
                uint32_t oldSelectNum = old.task.curSelectNum;
                uint32_t oldKvYBlockNum = kvYBlockNum;
                uint32_t oldBlockStackNum = blockStackNum;
                blockMmadPV(gP[gmOffsetOTmp], gV[old.task.gmOffsetK], gOTmp[gmOffsetOTmp], gBlockTable,
                            gSelectIdx[old.task.listOffset], layoutPTemp, layoutVTemp, layoutOTmp, actualBlockShapePV,
                            nowKvSIdx, oldKvSLoopNumTotal, pagedBlockSize, kvSeqlen, strideKV, oldBlockStackNum,
                            softmaxReady, qBlockY, oldSelectNum, oldKvYBlockNum);
                NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                LayoutO layoutO(qSeqlen, embed);
                LayoutLse layoutLse(qSeqlen, 1);
                LayoutUpdate layoutUpdate(old.task.rowNum, embed, embedRound);
                NpuArch::Arch::CrossCoreWaitFlag(pvReady);
                epilogueRescaleO(gO[old.task.gmOffsetQ], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[0],
                                 layoutO, layoutOTmp, layoutUpdate, layoutLse, actualBlockShapePV, old.task.qSBlockSize,
                                 qNBlockSize, old.isFirst, old.isLast, old.slot);
#endif
                pendingHead = (pendingHead + 1) % stage2Delay;
                --pendingNum;
            }
            if (!haveStack) {
                if (pendingNum == 0) {
                    break;
                }
                continue;
            }
            pending[(pendingHead + pendingNum) % stage2Delay] = job;
            ++pendingNum;
            ++stackSeq;
            nextKvSIdx += blockStackNum;
        }
#ifdef __DAV_C220_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // Values of one query task that its KV stacks need in both stages.
    struct QueryTask {
        uint64_t gmOffsetQ{0};
        uint64_t gmOffsetK{0};
        uint64_t listOffset{0};
        uint32_t curSelectNum{0};
        uint32_t noSkipKvS{0};
        uint32_t kvSLoopNumTotal{0};
        uint32_t qSBlockSize{0};
        uint32_t rowNum{0};
    };

    // One KV stack of a task: slot selects its score/probability/output workspace and its per-row UB values.
    struct StackJob {
        QueryTask task;
        uint32_t kvSIdx{0};
        uint32_t stackSeqTile{0};
        uint32_t slot{0};
        uint32_t isFirst{0};
        uint32_t isLast{0};
    };

    __aicore__ inline static uint32_t ChunkRows(uint32_t rowNum, uint32_t chunkIdx)
    {
        uint32_t rest = rowNum - chunkIdx * Q_TILE_CEIL;
        return rest < Q_TILE_CEIL ? rest : Q_TILE_CEIL;
    }

    NpuArch::Arch::Resource<ArchTag> resource;
    NpuArch::Arch::CrossCoreFlag qkReady{QK_READY_ID};
    NpuArch::Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    NpuArch::Arch::CrossCoreFlag pvReady{PV_READY_ID};

    // UB entries of each list-writer buffer (the two buffers take the first 64 KiB of the UB).
    static constexpr uint32_t LIST_CHUNK = 8192;

    uint32_t tilesT{1};
    uint32_t tilesH{1};
    uint32_t tilesW{1};
    uint32_t imageBlocks{1};
    uint32_t blockNum{1};
    uint32_t textSelect{0};
    uint32_t selectCap{IDX_PER_C0};
    uint64_t listRowNum{0};
    uint64_t listRowsPerSubCore{1};
    uint32_t listSubCoreNum{0};
    uint32_t headSelectNum[HEAD_CACHE_NUM] = {0};

    AscendC::LocalTensor<int32_t> listUb[2];
    uint32_t listFill{0};
    uint32_t listRowFill{0};
    uint32_t listBuf{0};
    uint64_t listGmOffset{0};
};

template <typename InputDtype>
__aicore__ inline void Infer(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR windows, GM_ADDR o, GM_ADDR workspace,
                             GM_ADDR tiling)
{
    using SoftmaxDtype = float;
    using QType = Gemm::GemmType<InputDtype, layout::RowMajor>;
    using KType = Gemm::GemmType<InputDtype, layout::ColumnMajor>;
    using SType = Gemm::GemmType<SoftmaxDtype, layout::RowMajor>;
    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using BlockMmadQK =
        Gemm::Block::BlockMmad<Gemm::MmadAtlasA2SFAIQK<false, false>, L1TileShapeQK, L0TileShapeQK, QType, KType, SType,
                               void, StaTileCopy<Arch::AtlasA2, QType, KType, SType>>;

    using PType = Gemm::GemmType<InputDtype, layout::RowMajor>;
    using VType = Gemm::GemmType<InputDtype, layout::RowMajor>;
    using OTmpType = Gemm::GemmType<SoftmaxDtype, layout::RowMajor>;
    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using BlockMmadPV =
        Gemm::Block::BlockMmad<Gemm::MmadAtlasA2SFAIPV<false, false>, L1TileShapePV, L0TileShapePV, PType, VType,
                               OTmpType, void, StaTileCopy<Arch::AtlasA2, PType, VType, OTmpType>>;

    using MaskType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<Epilogue::StaOnlineSoftmax, PType, SType, MaskType>;
    using OType = Gemm::GemmType<InputDtype, layout::RowMajor>;
    using OUpdateType = Gemm::GemmType<SoftmaxDtype, layout::RowMajor>;
    using LseType = Gemm::GemmType<float, layout::RowMajor>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::StaRescaleO, OType, OTmpType, OUpdateType, LseType>;

    using KernelType = StaBsaKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO>;
    StaBsaKernelParams params{q, k, v, windows, o, workspace, tiling};
    KernelType kernel;
    kernel(params);
}

} // namespace StaBsa

#endif // STA_BSA_KERNEL_H
