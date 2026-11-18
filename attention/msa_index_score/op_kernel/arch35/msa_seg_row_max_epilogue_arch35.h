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
 * \file msa_seg_row_max_epilogue_arch35.h
 * \brief AIV 侧 Epilogue：可选反量化列乘 -> atten_mask -> 分段 RowMax -> local_mask -> 写回。
 *
 * 数值路径：
 *   score = Maxpool[ (scale·) Q@Kᵀ + atten_mask ] + local_mask
 * atten_mask（sparse_mode=3）按 rightDownCausal 解析，不加载 [2048,2048] 模板。
 * local_mask 由 start_loc（query 所在逻辑 block）+ init/local_blocks 生成强制 +∞。
 */

#ifndef MSA_SEG_ROW_MAX_EPILOGUE_ARCH35_H
#define MSA_SEG_ROW_MAX_EPILOGUE_ARCH35_H

#include "kernel_operator.h"

#include "../catlass/arch/msa_arch.hpp"
#include "../catlass/arch/msa_resource.hpp"

#include "../msa_index_score_common.h"
#include "../msa_index_score_task.h"

namespace MsaIndexScoreNs {

#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
constexpr AscendC::Reg::CastTrait kMxScoreHalfToFloat = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::UNKNOWN};

// One page reduction stays in registers until the final FP32 stage scatter.
// Scatter uses element indices (u32 for FP32); only lane zero is written.
template <bool FINITE_FILL, bool LEGACY_ADDS>
__simd_vf__ inline void MxReducePagesToStageVf(__ubuf__ half* src, __ubuf__ float* dst, uint32_t rows, uint32_t pages,
                                               uint32_t stageStride, uint32_t srcRowStride)
{
    AscendC::Reg::RegTensor<half> values;
    AscendC::Reg::RegTensor<half> maximum;
    AscendC::Reg::RegTensor<float> fp32;
    AscendC::Reg::RegTensor<uint32_t> zeroIndex;
    auto allHalf = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();
    uint32_t one = 1U;
    auto firstFloat = AscendC::Reg::UpdateMask<float>(one);
    AscendC::Reg::Duplicate(zeroIndex, static_cast<uint32_t>(0));
    const uint32_t rowStride = srcRowStride;
    for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
        for (uint16_t page = 0; page < static_cast<uint16_t>(pages); ++page) {
            AscendC::Reg::LoadAlign(values, src + row * rowStride + page * MSA_BLOCK_SIZE);
            AscendC::Reg::ReduceMax(maximum, values, allHalf);
            AscendC::Reg::Cast<float, half, kMxScoreHalfToFloat>(fp32, maximum, firstFloat);
            if constexpr (FINITE_FILL) {
                AscendC::Reg::Maxs(fp32, fp32, MSA_FILL_VALUE, firstFloat);
            }
            if constexpr (LEGACY_ADDS) {
                AscendC::Reg::Adds(fp32, fp32, 0.0F, firstFloat);
            }
            AscendC::Reg::Scatter(dst + row * stageStride + page, fp32, zeroIndex, firstFloat);
        }
    }
}
// Reads only the existing fallback WRM half maxima. No new half-UB round trip.
// Columns already written by the direct prefix keep exclusive stage ownership.
template <bool FINITE_FILL, bool LEGACY_ADDS>
__simd_vf__ inline void MxScatterFallbackColumnsVf(__ubuf__ half* src, __ubuf__ float* dst, uint32_t rows,
                                                   uint32_t endCol, uint32_t firstFallbackCol, uint32_t stageStride)
{
    AscendC::Reg::RegTensor<half> maximum;
    AscendC::Reg::RegTensor<float> fp32;
    AscendC::Reg::RegTensor<uint32_t> zeroIndex;
    uint32_t one = 1U;
    auto firstFloat = AscendC::Reg::UpdateMask<float>(one);
    AscendC::Reg::Duplicate(zeroIndex, static_cast<uint32_t>(0));
    for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
        for (uint16_t col = static_cast<uint16_t>(firstFallbackCol); col < static_cast<uint16_t>(endCol); ++col) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_BRC_B16>(maximum,
                                                                                src + row * MSA_BLOCKS_PER_STILE + col);
            AscendC::Reg::Cast<float, half, kMxScoreHalfToFloat>(fp32, maximum, firstFloat);
            if constexpr (FINITE_FILL) {
                AscendC::Reg::Maxs(fp32, fp32, MSA_FILL_VALUE, firstFloat);
            }
            if constexpr (LEGACY_ADDS) {
                AscendC::Reg::Adds(fp32, fp32, 0.0F, firstFloat);
            }
            AscendC::Reg::Scatter(dst + row * stageStride + col, fp32, zeroIndex, firstFloat);
        }
    }
}

// Full-stile reduction: half maxima remain in registers. RegLayout::ZERO
// consumes the LOW half of each32-bit lane, so page maxima occupy half0/2/4/6.
template <uint32_t PAGES>
__simd_vf__ inline void MxBatchPageMaxToStageVf(__ubuf__ half* src, __ubuf__ float* dst, uint32_t rows,
                                                uint32_t srcRowStride, uint32_t stageStride)
{
    static_assert(PAGES == 2U || PAGES == 4U);
    AscendC::Reg::RegTensor<half> values;
    AscendC::Reg::RegTensor<half> packed;
    AscendC::Reg::RegTensor<half> maximum;
    AscendC::Reg::RegTensor<float> fp32;
    AscendC::Reg::RegTensor<uint32_t> indices;
    auto allHalf = AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();
    uint32_t two = 2U;
    uint32_t three = 3U;
    uint32_t outCount = PAGES;
    auto low2 = AscendC::Reg::UpdateMask<half>(two);
    auto low3 = AscendC::Reg::UpdateMask<half>(three);
    AscendC::Reg::MaskReg lane2;
    AscendC::Reg::Xor(lane2, low3, low2, allHalf);
    auto outMask = AscendC::Reg::UpdateMask<float>(outCount);
    AscendC::Reg::Arange(reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(indices), static_cast<int32_t>(0));
    AscendC::Reg::MaskReg lane4;
    AscendC::Reg::MaskReg lane6;
    if constexpr (PAGES == 4U) {
        uint32_t four = 4U;
        uint32_t five = 5U;
        uint32_t six = 6U;
        uint32_t seven = 7U;
        auto low4 = AscendC::Reg::UpdateMask<half>(four);
        auto low5 = AscendC::Reg::UpdateMask<half>(five);
        auto low6 = AscendC::Reg::UpdateMask<half>(six);
        auto low7 = AscendC::Reg::UpdateMask<half>(seven);
        AscendC::Reg::Xor(lane4, low5, low4, allHalf);
        AscendC::Reg::Xor(lane6, low7, low6, allHalf);
    }
    for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
        __ubuf__ half* input = src + row * srcRowStride;
        AscendC::Reg::LoadAlign(values, input);
        AscendC::Reg::ReduceMax(packed, values, allHalf);
        AscendC::Reg::LoadAlign(values, input + MSA_BLOCK_SIZE);
        AscendC::Reg::ReduceMax(maximum, values, allHalf);
        AscendC::Reg::Duplicate<half, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::MERGING>(
            packed, maximum, lane2);
        if constexpr (PAGES == 4U) {
            AscendC::Reg::LoadAlign(values, input + 2U * MSA_BLOCK_SIZE);
            AscendC::Reg::ReduceMax(maximum, values, allHalf);
            AscendC::Reg::Duplicate<half, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::MERGING>(
                packed, maximum, lane4);
            AscendC::Reg::LoadAlign(values, input + 3U * MSA_BLOCK_SIZE);
            AscendC::Reg::ReduceMax(maximum, values, allHalf);
            AscendC::Reg::Duplicate<half, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::MERGING>(
                packed, maximum, lane6);
        }
        AscendC::Reg::Cast<float, half, kMxScoreHalfToFloat>(fp32, packed, outMask);
        AscendC::Reg::Scatter(dst + row * stageStride, fp32, indices, outMask);
    }
}

#endif

/// IS_BLOCK16：M3 Index Branch。逻辑 block 长 MSA_BLOCK16_SIZE，一个 128 列窗 = 8 个逻辑 block；
/// 归约在 block 内 maxpool 之前先做 head 间 maxpool，输出对一个 token 的全部 head 取同一个值。
template <bool IS_QUANT, bool IS_BLOCK16 = false, bool IS_MXFP4 = false, bool MX_FUSED_STAGE = false>
class MsaSegRowMaxEpilogue {
public:
    using ArchTag = Catlass::Arch::AtlasA5;
    // S workspace 元素类型：非量化 fp16，int8 fp32。
    using ElementS = std::conditional_t<IS_QUANT, float, half>;

    // score 暂存容量：1 AIC : 2 AIV 下每个 subcore 最多拿到半个 M-tile 的行。
    // block16 + nWin=4 时 S ping 两级已占 128KB，stage 收到 128 列（仍 ≥ 一组 32 block），
    // 给 T64/rowmax 留余量，避免 UB 顶满后踩栈外。
    static constexpr uint32_t MSA_STAGE_ROWS = MSA_ROW_TILE_M / MSA_AIV_PER_AIC;
    static constexpr uint32_t MSA_STAGE_BLOCKS = IS_BLOCK16 ? 128U : 256U;
    static_assert(MSA_STAGE_BLOCKS % MSA_BLOCKS_PER_STILE == 0, "stage window must hold whole S-tiles");
    static_assert(MSA_STAGE_BLOCKS >= MSA_BLOCKS_PER_STILE * MSA_BLOCK16_MAX_WIN_GROUP || !IS_BLOCK16,
                  "stage must hold one block16 win-group");

    static constexpr uint32_t MSA_HALF_PER_BLOCK = 16;
    // repeat 次数与 repeat stride 都是 8 bit 字段。
    static constexpr uint32_t MSA_MAX_REPEAT = 255;
    static constexpr uint32_t MSA_MAX_REP_STRIDE = 255;

    // UB 手工布局（字节偏移），catlass 不提供 UB allocator。
    // Only MXFP4 block16 groups up to four windows. Native block16 uses one
    // window (at most128 rows on one AIV); keep room for its Q/K gather buffers.
    static constexpr uint32_t S_PING_BYTES = (IS_BLOCK16 && IS_MXFP4) ? MSA_A5_S_PING_BYTES_B16 : MSA_A5_S_PING_BYTES;
    static constexpr uint32_t UB_OFF_S = 0;
    static constexpr uint32_t UB_SIZE_S_GM = MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t UB_SIZE_S = IS_BLOCK16 ? (S_PING_BYTES * MSA_A5_S_STAGES) : UB_SIZE_S_GM;
    static constexpr uint32_t UB_OFF_T64 = UB_OFF_S + UB_SIZE_S;
    // GM：SegRowMax 的 [128,64] fp32（32KB）。C2UB：拼 [mSub,128] fp16 的 pack 区（16KB）复用这块。
    static constexpr uint32_t UB_SIZE_T64 = MSA_REDUCE_ROWS * (MSA_BLOCK_SIZE / 2) * sizeof(float);
    static constexpr uint32_t UB_OFF_T8 = UB_OFF_T64 + UB_SIZE_T64;
    static constexpr uint32_t UB_SIZE_T8 = MSA_REDUCE_ROWS * MSA_FP32_PER_BLOCK * sizeof(float);
    static constexpr uint32_t UB_OFF_ROWMAX = UB_OFF_T8 + UB_SIZE_T8;
    static constexpr uint32_t UB_SIZE_ROWMAX_GM = MSA_REDUCE_ROWS * sizeof(float);
    // 一组最多 MSA_BLOCK16_MAX_WIN_GROUP 个窗，结果数按组算。
    static constexpr uint32_t UB_SIZE_ROWMAX_C2UB =
        MSA_A5_S_PING_ROWS * MSA_BLOCKS_PER_STILE * MSA_BLOCK16_MAX_WIN_GROUP * sizeof(float);
    static constexpr uint32_t UB_SIZE_ROWMAX =
        (UB_SIZE_ROWMAX_C2UB > UB_SIZE_ROWMAX_GM) ? UB_SIZE_ROWMAX_C2UB : UB_SIZE_ROWMAX_GM;
    // 反量化 scale 一页：block_size 个 fp32 = 512B。
    static constexpr uint32_t UB_OFF_DEQ = UB_OFF_ROWMAX + UB_SIZE_ROWMAX;
    static constexpr uint32_t UB_SIZE_DEQ = MSA_BLOCK_SIZE * sizeof(float);
    // 非量化：GM 为 128 个 half；C2UB stile 为 64×8 个 half。
    static constexpr uint32_t UB_OFF_RED16 = UB_OFF_DEQ + UB_SIZE_DEQ;
    static constexpr uint32_t UB_SIZE_RED16_GM = MSA_REDUCE_ROWS * sizeof(half);
    static constexpr uint32_t UB_SIZE_RED16_C2UB =
        MSA_A5_S_PING_ROWS * MSA_BLOCKS_PER_STILE * MSA_BLOCK16_MAX_WIN_GROUP * sizeof(half);
    static constexpr uint32_t UB_SIZE_RED16 =
        (UB_SIZE_RED16_C2UB > UB_SIZE_RED16_GM) ? UB_SIZE_RED16_C2UB : UB_SIZE_RED16_GM;
    // score 暂存：本 subcore 的行 × 一次 flush 覆盖的 block 数。逐 pass 只写 32B 到 GM
    // 会把 MTE3 压到 ~9GB/s；改为在 UB 内按行累积、整行（≥1KB）一次写回。
    static constexpr uint32_t UB_OFF_STAGE =
        ((UB_OFF_RED16 + UB_SIZE_RED16 + MSA_UB_ALIGN_BYTES - 1U) / MSA_UB_ALIGN_BYTES) * MSA_UB_ALIGN_BYTES;
    static constexpr uint32_t UB_SIZE_STAGE = MSA_STAGE_ROWS * MSA_STAGE_BLOCKS * sizeof(float);
    // 非量化：S16（fp16 载入缓冲）复用量化路径的 fp32 S 区（64KB = 两级 32KB 乒乓）。
    // 两级缓冲让下一 pass 的 MTE2 叠在当前 pass 的 V 上。
    static constexpr uint32_t UB_OFF_S16 = UB_OFF_S;
    static constexpr uint32_t UB_SIZE_S16 = MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE * sizeof(half);
    static constexpr uint32_t S16_STAGES = 2;
    static constexpr uint32_t UB_TOTAL = UB_OFF_STAGE + UB_SIZE_STAGE;
    static_assert(UB_TOTAL <= ArchTag::UB_SIZE, "MsaSegRowMaxEpilogue UB out of bounds");
    static_assert(UB_OFF_S16 + S16_STAGES * UB_SIZE_S16 <= UB_OFF_T64, "S16 ping-pong overlaps T64");
    // C2UB pack：[mSub,64]×2 half + [mSub,128] packed，最大 32KB，落在 T64。
    static_assert(MSA_A5_S_PING_ROWS * MSA_BLOCK_SIZE * sizeof(half) * 2 <= UB_SIZE_T64,
                  "C2UB fp16 128-wide pack exceeds T64");
    // v40：两路 64 宽 fp32 WRM 的 dst 落在 T8，需 32B 对齐且互不重叠。
    static constexpr uint32_t C2UB_WRM_DST_STRIDE = (MSA_A5_S_PING_ROWS + 7U) & ~7U;
    static_assert(2U * C2UB_WRM_DST_STRIDE * sizeof(float) <= UB_SIZE_T8, "C2UB dual fp32 WRM dst exceeds T8");

    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    // fp16：一个 32B datablock 16 个元素，一个 repeat 128 个元素（掩码 128 位）。
    static constexpr uint32_t MSA_HALF_PER_MASK_WORD = 64;

    // -3.4e38 越界，float→half 按 IEEE 就近舍入得 -inf；归约后由 Maxs 抬回 MSA_FILL_VALUE。
    static constexpr half FILL_VALUE_H = static_cast<half>(MSA_FILL_VALUE);

    __aicore__ inline MsaSegRowMaxEpilogue() {}

    __aicore__ inline void Init(Catlass::Arch::Resource<ArchTag>& resource, uint32_t numQHeads, uint32_t strideOutHead,
                                uint32_t strideOutToken, uint32_t maxBlocksPerBatch, uint32_t tableWidth,
                                uint32_t blockSize, uint32_t pageSize, uint32_t strideScalePage,
                                uint32_t strideScaleHead, bool isQuant, uint32_t keyLayout, uint32_t totalK,
                                const AscendC::GlobalTensor<float>& gScore, const AscendC::GlobalTensor<float>& gScale,
                                const AscendC::GlobalTensor<int32_t>& gBlockTable)
    {
        numQHeads_ = numQHeads;
        strideOutHead_ = strideOutHead;
        strideOutToken_ = strideOutToken;
        maxBlocksPerBatch_ = maxBlocksPerBatch;
        tableWidth_ = tableWidth;
        blockSize_ = blockSize;
        pageSize_ = pageSize;
        strideScalePage_ = strideScalePage;
        strideScaleHead_ = strideScaleHead;
        isQuant_ = isQuant;
        keyLayout_ = keyLayout;
        totalK_ = totalK;
        if (keyLayout_ == MSA_KEY_LAYOUT_TND) {
            // N2=1 packed [T2] / [T2,1]：token 步长为 1，避免误用 PA page 步长。
            strideScalePage_ = 1;
            strideScaleHead_ = 1;
        }
        gScore_ = gScore;
        gScale_ = gScale;
        gBlockTable_ = gBlockTable;

        ubS_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_S);
        ubSPing_[0] = resource.ubBuf.template GetBufferByByte<float>(0);
        ubSPing_[1] = resource.ubBuf.template GetBufferByByte<float>(S_PING_BYTES);
        ubS16Ping_[0] = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_S16);
        ubS16Ping_[1] = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_S16 + UB_SIZE_S16);
        ubS16_ = ubS16Ping_[0];
        ubT64_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_T64);
        ubStash_ = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_T64);
        ubT8_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_T8);
        ubRowMax_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_ROWMAX);
        ubDeqScale_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_DEQ);
        ubRed16_ = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_RED16);
        ubStage_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_STAGE);
    }

    /// 进入一个新任务（M-tile）：确定本 subcore 负责的行区间并复位 score 暂存窗口。
    __aicore__ inline void BeginTask(const MsaTask& task, uint32_t subIdx, uint32_t subBlockNum)
    {
        mxDirectColumnMask_ = 0U;
        mxDirectStileBase_ = ~0U;
        const uint32_t subM = MsaCeilDiv(task.mActual, subBlockNum);
        mOff_ = subIdx * subM;
        mSub_ = (mOff_ >= task.mActual) ? 0U : MsaMinU32(subM, task.mActual - mOff_);
        if constexpr (MSA_A5_USE_C2UB && !IS_QUANT) {
            uint32_t m0 = 0;
            uint32_t m1 = 0;
            // blocksize=16 按 token（numQHeads 行）整组切，head 间 maxpool 不跨核；
            // 其余路径 group=1，退化为原来的偶数对半。必须与 Fixpipe 的切分一致。
            MsaC2UbGroupSplit(task.mActual, IS_BLOCK16 ? numQHeads_ : 1U, m0, m1);
            if (subIdx == 0U) {
                mOff_ = 0;
                mSub_ = MsaMinU32(m0, task.mActual);
            } else {
                mOff_ = m0;
                mSub_ = (task.mActual > m0) ? (task.mActual - m0) : 0U;
            }
        } else if constexpr (IS_BLOCK16 && IS_QUANT) {
            uint32_t m0 = 0;
            uint32_t m1 = 0;
            MsaC2UbGroupSplit(task.mActual, numQHeads_, m0, m1);
            if (subIdx == 0U) {
                mOff_ = 0;
                mSub_ = MsaMinU32(m0, task.mActual);
            } else {
                mOff_ = m0;
                mSub_ = (task.mActual > m0) ? (task.mActual - m0) : 0U;
            }
        }
        // blocksize=16 的 stage 一行是一个 token，写回 score 的唯一 head；其余路径一行就是一行。
        tokensSub_ = IS_BLOCK16 ? (mSub_ / numQHeads_) : mSub_;
        stageRows_ = IS_BLOCK16 ? tokensSub_ : mSub_;
        // 容量按 stage 行数判断。blocksize=16 时 N1=128 会让某个 AIV 拿到 128 个 head 行，
        // 但 stage 里只有 1 个 token 行，仍放得下。按 mSub_ 判断会关掉 stage，分数无法写回。
        stageOn_ = (stageRows_ > 0U) && (stageRows_ <= MSA_STAGE_ROWS);
        stageBlkBase_ = 0;
        stageBlkEnd_ = 0;
        rowInReqBase_ = task.mStart + mOff_;
        tokenBase0_ = task.cuQStart;
        writeLo_ = task.sBlkBegin;
        writeHi_ = task.writeTailFill ? strideOutToken_ : task.sBlkEnd;
        if constexpr (MSA_A5_USE_C2UB) {
            // 禁止 4B DataCopyPad：MTE3 会按 32B 写出，把后面的 fill 槽写成 0。
            // UB 一次最多攒 MSA_STAGE_BLOCKS 列；block_table 宽 >256 时 score 末维
            // RoundUp(width,16) 会超过该容量，必须按窗口 flush，不能关掉 stage
            // （关掉后 C2UB 没有逐 block 直写，GM 保持 at::empty 垃圾值）。
            stageOn_ = (stageRows_ > 0U) && (stageRows_ <= MSA_STAGE_ROWS);
            if (stageOn_) {
                OpenStageWindow(0);
            }
        } else if constexpr (!IS_QUANT) {
            // 本任务两级 S16 的 V_MTE2 余额；EndTask 对称 Wait，避免跨任务/跨启动泄漏。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }
    }

    /// 任务结束：把暂存窗口里剩余的 score 写回 GM。
    __aicore__ inline void EndTask()
    {
        FlushStageToStrideEnd();
        if constexpr (!MSA_A5_USE_C2UB && !IS_QUANT) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }
    }

    /// 处理一个 S tile（MSA_BLOCKS_PER_STILE 个 sparse block）中属于本 subcore 的行。
    __aicore__ inline void ProcessSTile(const AscendC::GlobalTensor<ElementS>& gS, const MsaTask& task,
                                        uint32_t blkBase)
    {
        if (mSub_ == 0U) {
            return;
        }
        const uint32_t mOff = mOff_;
        const uint32_t mSub = mSub_;
        if (stageOn_ && (blkBase >= stageBlkBase_ + MSA_STAGE_BLOCKS)) {
            FlushStage();
            stageBlkBase_ = blkBase;
            stageBlkEnd_ = blkBase;
        }

        if constexpr (IS_QUANT) {
            for (uint32_t p0 = 0; p0 < mSub; p0 += MSA_ROWS_PER_PASS) {
                const uint32_t rows = MsaMinU32(MSA_ROWS_PER_PASS, mSub - p0);
                ProcessPass(gS, task, blkBase, mOff + p0, rows);
            }
        } else if (blkBase >= task.visibleEndBlk) {
            for (uint32_t p0 = 0; p0 < mSub; p0 += MSA_ROWS_PER_PASS) {
                const uint32_t rows = MsaMinU32(MSA_ROWS_PER_PASS, mSub - p0);
                ProcessPass(gS, task, blkBase, mOff + p0, rows);
            }
        } else {
            // 非量化可见 tile：pass i 的 MTE2 与 pass i-1 的 mask/reduce/stage 重叠。
            uint32_t stage = 0;
            bool hasPrev = false;
            uint32_t prevRowOff = 0;
            uint32_t prevRows = 0;
            uint32_t prevStage = 0;
            for (uint32_t p0 = 0; p0 < mSub; p0 += MSA_ROWS_PER_PASS) {
                const uint32_t rows = MsaMinU32(MSA_ROWS_PER_PASS, mSub - p0);
                const uint32_t rowOff = mOff + p0;
                IssueS16(gS, rowOff, rows, stage);
                if (hasPrev) {
                    WaitS16(prevStage);
                    ProcessPass(gS, task, blkBase, prevRowOff, prevRows, true);
                    ReleaseS16(prevStage);
                }
                hasPrev = true;
                prevRowOff = rowOff;
                prevRows = rows;
                prevStage = stage;
                stage ^= 1U;
            }
            if (hasPrev) {
                WaitS16(prevStage);
                ProcessPass(gS, task, blkBase, prevRowOff, prevRows, true);
                ReleaseS16(prevStage);
            }
        }
        stageBlkEnd_ = blkBase + MSA_BLOCKS_PER_STILE;
    }

    // Return the exact leading page count written directly. Source row stride
    // remains the ORIGINAL group width even when only its prefix is reduced.
    __aicore__ inline uint32_t TryMxDirectStage(uint32_t ping, const MsaTask& task, uint32_t blk, uint32_t nPages)
    {
#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
        if constexpr (IS_MXFP4 && MX_FUSED_STAGE && !IS_BLOCK16 && !IS_QUANT) {
            const uint32_t stileBase = blk - blk % MSA_BLOCKS_PER_STILE;
            const uint32_t stileEnd = stileBase + MSA_BLOCKS_PER_STILE;
            // Original R27 whole-stile route is isolated before all partial
            // ownership reads/writes. Every group, including singleton PA runs,
            // takes this route for the whole stile, so no fallback can overwrite it.
            if (stageOn_ && mSub_ >= 32U && stageRows_ == mSub_ && nPages != 0U &&
                nPages <= MSA_BLOCKS_PER_STILE - blk % MSA_BLOCKS_PER_STILE && task.initBlocks == 0U &&
                task.localBlocks == 0U && stileEnd <= task.fullEndBlk && stileEnd <= task.visibleEndBlk &&
                stileEnd <= task.sBlkEnd) {
                AdvanceStageWindow(blk);
                auto page = ubSPing_[ping].template ReinterpretCast<half>();
                auto* src = reinterpret_cast<__ubuf__ half*>(page.GetPhyAddr());
                auto* dst = reinterpret_cast<__ubuf__ float*>(ubStage_.GetPhyAddr()) + blk - stageBlkBase_;
                if (nPages == 4U) {
                    MxBatchPageMaxToStageVf<4U>(src, dst, mSub_, nPages * MSA_BLOCK_SIZE, MSA_STAGE_BLOCKS);
                } else if (nPages == 2U) {
                    MxBatchPageMaxToStageVf<2U>(src, dst, mSub_, nPages * MSA_BLOCK_SIZE, MSA_STAGE_BLOCKS);
                } else {
                    MxReducePagesToStageVf<false, false>(src, dst, mSub_, nPages, MSA_STAGE_BLOCKS,
                                                         nPages * MSA_BLOCK_SIZE);
                }
                AscendC::PipeBarrier<PIPE_V>();
                return nPages;
            }
            if (mxDirectStileBase_ != stileBase) {
                mxDirectStileBase_ = stileBase;
                mxDirectColumnMask_ = 0U;
            }
            if (!stageOn_ || mSub_ < 32U || stageRows_ != mSub_ || nPages == 0U ||
                nPages > MSA_BLOCKS_PER_STILE - blk % MSA_BLOCKS_PER_STILE || task.initBlocks != 0U ||
                task.localBlocks != 0U || stileBase < task.sBlkBegin) {
                return 0U;
            }
            uint32_t directEnd = MsaMinU32(stileEnd, task.fullEndBlk);
            directEnd = MsaMinU32(directEnd, task.visibleEndBlk);
            directEnd = MsaMinU32(directEnd, task.sBlkEnd);
            if (blk >= directEnd) {
                return 0U;
            }
            const uint32_t directPages = MsaMinU32(nPages, directEnd - blk);
            const bool needFill = stileEnd > task.visibleEndBlk || stileEnd > task.sBlkEnd;
            // A partial direct stile bypasses the old col0 setup. Only actual
            // fallback columns will be read later, so no ubRed16 prefill needed.
            if (directEnd < stileEnd) {
                stileNeedFill_ = needFill;
            }
            AdvanceStageWindow(blk);
            auto page = ubSPing_[ping].template ReinterpretCast<half>();
            auto* src = reinterpret_cast<__ubuf__ half*>(page.GetPhyAddr());
            auto* dst = reinterpret_cast<__ubuf__ float*>(ubStage_.GetPhyAddr()) + blk - stageBlkBase_;
            if (needFill) {
                MxReducePagesToStageVf<true, true>(src, dst, mSub_, directPages, MSA_STAGE_BLOCKS,
                                                   nPages * MSA_BLOCK_SIZE);
            } else if (directEnd < stileEnd && (mSub_ & 1U) != 0U) {
                MxReducePagesToStageVf<false, true>(src, dst, mSub_, directPages, MSA_STAGE_BLOCKS,
                                                    nPages * MSA_BLOCK_SIZE);
            } else {
                MxReducePagesToStageVf<false, false>(src, dst, mSub_, directPages, MSA_STAGE_BLOCKS,
                                                     nPages * MSA_BLOCK_SIZE);
            }
            AscendC::PipeBarrier<PIPE_V>();
            mxDirectColumnMask_ |= ((1U << directPages) - 1U) << (blk - stileBase);
            return directPages;
        }
#endif
        return 0U;
    }

    /// C_to_UB：非量化 ping 已是 packed [mSub,128] fp16（两次单目标 F322F16）；
    /// int8 仍为两块 [mSub,64] fp32 panel，64 宽密排 WRM 后 Max。
    __aicore__ inline void ProcessPageUb(uint32_t ping, const MsaTask& task, uint32_t blk)
    {
        if (mSub_ == 0U) {
            return;
        }
        if (TryMxDirectStage(ping, task, blk, 1U)) {
            return;
        }
        AdvanceStageWindow(blk);
        constexpr uint32_t N_PER_ND = 64;
        constexpr uint32_t N_PAGE = MSA_BLOCK_SIZE;
        constexpr uint8_t SRC_REP_128 = N_PAGE / MSA_HALF_PER_BLOCK;  // 8
        constexpr uint8_t REP_BLK_64 = N_PER_ND / MSA_FP32_PER_BLOCK; // 8
        const uint32_t col = blk % MSA_BLOCKS_PER_STILE;
        // ubRed16_ 直接按 [row][block]：dstRepStride=8。仅 stile 不满/有不可见列时才预填 -inf。
        if (col == 0U) {
            const uint32_t stileEnd = blk + MSA_BLOCKS_PER_STILE;
            const bool needFill = (stileEnd > task.visibleEndBlk) || (stileEnd > task.sBlkEnd);
            if (needFill) {
                AscendC::Duplicate(ubRed16_, FILL_VALUE_H, MSA_A5_S_PING_ROWS * MSA_BLOCKS_PER_STILE);
                AscendC::PipeBarrier<PIPE_V>();
            }
            stileNeedFill_ = needFill;
        }
        const bool invalid = (blk >= task.visibleEndBlk);
        if (!invalid) {
            if constexpr (IS_QUANT) {
                auto ubPage = ubSPing_[ping];
                ApplyDequantScalePage(task, blk, ubPage);
                ApplyMaskPage(task, blk, ubPage);
                auto ubP1 = ubPage[mSub_ * N_PER_ND];
                auto dst0 = ubT8_;
                auto dst1 = ubT8_[C2UB_WRM_DST_STRIDE];
                AscendC::WholeReduceMax<float>(dst0, ubPage, static_cast<int32_t>(N_PER_ND),
                                               static_cast<int32_t>(mSub_), 1, 1, REP_BLK_64,
                                               AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                AscendC::WholeReduceMax<float>(dst1, ubP1, static_cast<int32_t>(N_PER_ND), static_cast<int32_t>(mSub_),
                                               1, 1, REP_BLK_64, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Max(dst0, dst0, dst1, mSub_);
                AscendC::PipeBarrier<PIPE_V>();
                // 先 Cast 到临时密排，再按 [row][block] 展开到 ubRed16_[col]。
                auto ubTmpH = ubStash_.template ReinterpretCast<half>();
                AscendC::Cast(ubTmpH, dst0, AscendC::RoundMode::CAST_RINT, mSub_);
                AscendC::PipeBarrier<PIPE_V>();
                ScatterRedCol(ubTmpH, col, mSub_);
            } else {
                auto ubPageH = ubSPing_[ping].template ReinterpretCast<half>();
                ApplyMaskPageHalf(task, blk, ubPageH);
                // dstRepStride=8：第 r 行写到 ubRed16_[col + r*8]。
                AscendC::WholeReduceMax<half>(ubRed16_[col], ubPageH, static_cast<int32_t>(N_PAGE),
                                              static_cast<int32_t>(mSub_), static_cast<uint16_t>(MSA_BLOCKS_PER_STILE),
                                              1, SRC_REP_128, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        if (col == (MSA_BLOCKS_PER_STILE - 1U) || (blk + 1U) == task.sBlkEnd || (blk + 1U) == task.visibleEndBlk) {
            ReduceStileToStage(task, blk - col);
        }
    }

    /// nPages 页合成的 [mSub, nPages*128] fp16。按 128 列逐页归约。
    /// 每页 WRM 完成后同步，再将对应列归约到 stage。
    __aicore__ inline void ProcessPageGroupUb(uint32_t ping, const MsaTask& task, uint32_t blk, uint32_t nPages)
    {
        if (mSub_ == 0U || nPages == 0U) {
            return;
        }
        const uint32_t directPages = TryMxDirectStage(ping, task, blk, nPages);
        if (directPages == nPages) {
            return;
        }
        auto ubPageH = ubSPing_[ping].template ReinterpretCast<half>();
        const uint32_t rowStride = nPages * MSA_BLOCK_SIZE;
        for (uint32_t i = directPages; i < nPages; ++i) {
            const uint32_t cur = blk + i;
            AdvanceStageWindow(cur);
            const uint32_t col = cur % MSA_BLOCKS_PER_STILE;
            if (col == 0U) {
                const uint32_t stileEnd = cur + MSA_BLOCKS_PER_STILE;
                const bool needFill = (stileEnd > task.visibleEndBlk) || (stileEnd > task.sBlkEnd);
                if (needFill) {
                    AscendC::Duplicate(ubRed16_, FILL_VALUE_H, MSA_A5_S_PING_ROWS * MSA_BLOCKS_PER_STILE);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                stileNeedFill_ = needFill;
            }
            if (cur < task.visibleEndBlk) {
                ApplyMaskPageHalf(task, cur, ubPageH, rowStride, i * MSA_BLOCK_SIZE);
                const uint8_t srcRep = static_cast<uint8_t>(rowStride / MSA_HALF_PER_BLOCK);
                AscendC::WholeReduceMax<half>(ubRed16_[col], ubPageH[i * MSA_BLOCK_SIZE],
                                              static_cast<int32_t>(MSA_BLOCK_SIZE), static_cast<int32_t>(mSub_),
                                              static_cast<uint16_t>(MSA_BLOCKS_PER_STILE), 1, srcRep,
                                              AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                AscendC::PipeBarrier<PIPE_V>();
            }
            const bool stileDone =
                (col == (MSA_BLOCKS_PER_STILE - 1U) || (cur + 1U) == task.sBlkEnd || (cur + 1U) == task.visibleEndBlk);
            if (stileDone) {
                ReduceStileToStage(task, cur - col);
            }
        }
    }

    /// 一组 = nWin 个 128 列窗，由 Cube 并成一次宽 N 的 MMAD。
    /// UB 上是 packed [mSub, nWin*128] fp16，行 = (token, head)，head 为低位。
    /// 顺序：head 折叠 → 掩码 → 每 token 每窗一条 BlockReduceMax。
    /// max 可交换，先折 head 再折 block 与先折 block 再折 head 等价；
    /// 掩码按列且对同 token 的所有 head 相同，放在折叠之后结果不变、代价小 numQHeads 倍。
    /// 折叠、掩码和尾处理都按整组做一次，每窗固定开销随 nWin 摊薄。
    /// 读完 S ping 后立刻还 VC，让 Cube 与后续 mask/reduce/StageScore/Flush 重叠。
    /// 调用方（block16 循环）不得再 Set 同 ping，否则 MODE4 计数会多一次。
    __aicore__ inline void ProcessNativeWindowUb(uint32_t ping, const MsaTask& task, uint32_t win)
    {
        if (mSub_ == 0U) {
            return;
        }
        const uint32_t blkBase = win * MSA_BLOCKS_PER_STILE;
        AdvanceStageWindow(blkBase);
        auto ubPageH = ubSPing_[ping].template ReinterpretCast<half>();
        const uint32_t rowStride = numQHeads_ * MSA_BLOCK_SIZE;

        if (blkBase >= task.visibleEndBlk) {
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, stageRows_ * MSA_BLOCKS_PER_STILE);
            AscendC::PipeBarrier<PIPE_V>();
            ApplyLocalMaskUb(task, blkBase, stageRows_);
            StageScore(blkBase, 0, stageRows_);
            return;
        }

        NativeFoldHeads(ubPageH, rowStride);
        NativeApplyMaskWindowHalf(task, blkBase, ubPageH, rowStride);

        // 每 token 一条 WholeReduceMax：mask=16 → 每 repeat 归约一个逻辑 block，
        // 8 个 repeat 以 dstRepStride=1 密排写出，正好是 StageScore 要的 [token][block]。
        constexpr int32_t RED_MASK = static_cast<int32_t>(MSA_BLOCK16_SIZE);
        for (uint32_t t = 0; t < tokensSub_; ++t) {
            AscendC::WholeReduceMax<half>(ubRed16_[t * MSA_BLOCKS_PER_STILE], ubPageH[t * rowStride], RED_MASK,
                                          static_cast<int32_t>(MSA_BLOCKS_PER_STILE), 1, 1, 1,
                                          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ubRowMax_, ubRed16_, AscendC::RoundMode::CAST_NONE, stageRows_ * MSA_BLOCKS_PER_STILE);
        AscendC::PipeBarrier<PIPE_V>();
        // fp16 域的 -inf 填充抬回 fp32 的 MSA_FILL_VALUE，与其他路径一致。
        AscendC::Maxs<float>(ubRowMax_, ubRowMax_, MSA_FILL_VALUE,
                             static_cast<int32_t>(stageRows_ * MSA_BLOCKS_PER_STILE));
        AscendC::PipeBarrier<PIPE_V>();
        ApplyLocalMaskUb(task, blkBase, stageRows_);
        StageScore(blkBase, 0, stageRows_);
    }

    __aicore__ inline void NativeFoldHeads(const AscendC::LocalTensor<half>& ubPageH, uint32_t rowStride)
    {
        if (numQHeads_ <= 1U) {
            return;
        }
        for (uint32_t half = numQHeads_ >> 1; half > 0U; half >>= 1) {
            const uint32_t count = half * MSA_BLOCK_SIZE;
            for (uint32_t t = 0; t < tokensSub_; ++t) {
                const uint32_t base = t * rowStride;
                AscendC::Max(ubPageH[base], ubPageH[base], ubPageH[base + count], static_cast<int32_t>(count));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void NativeApplyMaskWindowHalf(const MsaTask& task, uint32_t blkBase,
                                                     const AscendC::LocalTensor<half>& ubPageH, uint32_t rowStride)
    {
        const int32_t winTok = static_cast<int32_t>(blkBase) * static_cast<int32_t>(MSA_BLOCK16_SIZE);
        bool touched = false;
        for (uint32_t t = 0; t < tokensSub_; ++t) {
            const uint32_t rowInReq = task.mStart + mOff_ + t * numQHeads_;
            const uint32_t tOff = rowInReq / numQHeads_;
            const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
            int32_t validCount = visibleKeyEnd - winTok;
            if (validCount < 0) {
                validCount = 0;
            }
            if (validCount >= static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                continue;
            }
            FillBlockTailFp16At(ubPageH, t * rowStride, static_cast<uint32_t>(validCount));
            touched = true;
        }
        if (touched) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ReleaseSPing(uint32_t ping)
    {
        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + ping);
    }

    /// 返回 true：已在内部 ReleaseSPing（block16 循环勿再 Set VC）。
    __aicore__ inline bool ProcessWindowUb(uint32_t ping, const MsaTask& task, uint32_t win, uint32_t nWin = 1U)
    {
        if (mSub_ == 0U) {
            ReleaseSPing(ping);
            return true;
        }
        const uint32_t blkBase = win * MSA_BLOCKS_PER_STILE;
        const uint32_t nBlocks = nWin * MSA_BLOCKS_PER_STILE;
        AdvanceStageWindow(blkBase);
        auto ubPageH = ubSPing_[ping].template ReinterpretCast<half>();
        const uint32_t winCols = nWin * MSA_BLOCK_SIZE;
        const uint32_t rowStride = numQHeads_ * winCols;

        if (blkBase >= task.visibleEndBlk) {
            ReleaseSPing(ping);
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, stageRows_ * nBlocks);
            AscendC::PipeBarrier<PIPE_V>();
            ApplyLocalMaskUb(task, blkBase, stageRows_, nWin);
            StageScore(blkBase, 0, stageRows_, nWin);
            return true;
        }

        // RA 折叠结果留在 T64 compact [token][winCols]，mask/reduce 直接吃它，省掉逐 token
        // DataCopy 回 ping（GBSA fd_combine 也是 reduce 落独立 buf，不写回 src）。
        // 树 Max 回退路径仍是 in-place 写 head0，workStride 保持 rowStride。
        auto ubFold = ubT64_.template ReinterpretCast<half>();
        const bool compact = FoldHeads(ubPageH, ubFold, rowStride, winCols);
        auto ubWork = compact ? ubFold : ubPageH;
        const uint32_t workStride = compact ? winCols : rowStride;
        // compact：fold 后不再读 ping，提前还 VC，叠 Stage 侧尾巴。
        if (compact) {
            ReleaseSPing(ping);
        }
        const bool masked = ApplyMaskWindowHalf(task, blkBase, ubWork, workStride, nWin);

        // 归约到 [token][block]：每个逻辑 block=16 half=1 datablock。
        // nWin>1：WholeReduceMax(mask=16)。compact 时 [T][winCols] 与 [T][nBlocks] 均连续，
        // 一条 repeat=T*nBlocks 吃掉全部 token，省掉 T 次标量发射。
        // nWin=1 仍走满宽 BlockReduceMax（vec 更满）。
        const uint32_t totalRed = tokensSub_ * nBlocks;
        if (nWin > 1U && compact && totalRed <= MSA_MAX_REPEAT) {
            AscendC::WholeReduceMax<half>(ubRed16_, ubWork, static_cast<int32_t>(MSA_BLOCK16_SIZE),
                                          static_cast<int32_t>(totalRed), 1, 1, 1,
                                          AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        } else if (nWin > 1U && nBlocks <= MSA_MAX_REPEAT) {
            for (uint32_t t = 0; t < tokensSub_; ++t) {
                AscendC::WholeReduceMax<half>(ubRed16_[t * nBlocks], ubWork[t * workStride],
                                              static_cast<int32_t>(MSA_BLOCK16_SIZE), static_cast<int32_t>(nBlocks), 1,
                                              1, 1, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            }
        } else {
            constexpr int32_t RED_MASK = static_cast<int32_t>(MSA_BLOCK_SIZE);
            constexpr int32_t RED_SRC_REP = static_cast<int32_t>(MSA_BLOCK_SIZE / MSA_HALF_PER_BLOCK);
            for (uint32_t t = 0; t < tokensSub_; ++t) {
                AscendC::BlockReduceMax<half>(ubRed16_[t * nBlocks], ubWork[t * workStride], 1, RED_MASK, 1, 1,
                                              RED_SRC_REP);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (!compact) {
            ReleaseSPing(ping);
        }
        // 与 block128 的 ReduceStileToStage 同手法：本组没填过 -inf、也没有 local mask 时，下界保护
        // 和经 ubRowMax_ 的中转搬运都是恒等变换，一条带 stride 的 Cast 直接落 stage，省两道 barrier。
        // ubRed16_ 每行 nBlocks 个 half，不足一个 datablock 时 srcRepStride 表达不出来，故要求整除。
        const bool noLocalMask = (task.initBlocks == 0U) && (task.localBlocks == 0U);
        if (!masked && noLocalMask && ((nBlocks % MSA_HALF_PER_BLOCK) == 0U)) {
            AscendC::Cast(ubStage_[blkBase - stageBlkBase_], ubRed16_, AscendC::RoundMode::CAST_NONE,
                          static_cast<int32_t>(nBlocks), static_cast<uint8_t>(stageRows_),
                          AscendC::UnaryRepeatParams(1, 1, static_cast<uint16_t>(MSA_STAGE_BLOCKS / MSA_FP32_PER_BLOCK),
                                                     static_cast<uint16_t>(nBlocks / MSA_HALF_PER_BLOCK)));
            AscendC::PipeBarrier<PIPE_V>();
            return true;
        }
        AscendC::Cast(ubRowMax_, ubRed16_, AscendC::RoundMode::CAST_NONE, stageRows_ * nBlocks);
        AscendC::PipeBarrier<PIPE_V>();
        // fp16 域的 -inf 填充抬回 fp32 的 MSA_FILL_VALUE，与其他路径一致。
        AscendC::Maxs<float>(ubRowMax_, ubRowMax_, MSA_FILL_VALUE, static_cast<int32_t>(stageRows_ * nBlocks));
        AscendC::PipeBarrier<PIPE_V>();
        ApplyLocalMaskUb(task, blkBase, stageRows_, nWin);
        StageScore(blkBase, 0, stageRows_, nWin);
        return true;
    }

private:
    /// 把每个 token 的 numQHeads 行逐元素 max 成 1 行。
    /// 返回 true：结果在 ubFold 的 compact [token][winCols]（RA 路径，调用方勿再拷回 ping）；
    /// 返回 false：结果在 ubPageH 各 token 首行（树 Max / 单 head），rowStride 不变。
    /// host 已校验 MSA_ROW_TILE_M % numQHeads == 0，故 numQHeads 必为 2 的幂。
    __aicore__ inline bool FoldHeads(const AscendC::LocalTensor<half>& ubPageH,
                                     const AscendC::LocalTensor<half>& ubFold, uint32_t rowStride, uint32_t winCols)
    {
        if (numQHeads_ <= 1U) {
            return false;
        }
        const uint32_t nWin = winCols / MSA_BLOCK_SIZE;
        const bool colsAligned = ((winCols * sizeof(half)) % MSA_DATABLOCK_BYTES) == 0U;
        // tokensSub_×winCols 在 decode 典型形状下 ≤ 4×512；后半留给 RA scratch。
        // 多 token + 多窗：RA 逐 token 标量重；rowStride/16>255 又不能跨 token BinaryRepeat。
        // 改为按窗 gather 成 [T][H][128]（stride 合法），树 Max 把 token 维收进 repeat，再散回 compact。
        const uint64_t foldElems = static_cast<uint64_t>(tokensSub_) * winCols;
        const bool tmpFits = (foldElems * sizeof(half) * 2U) <= UB_SIZE_T64;
        // RA+compact 仍是多 token×多窗最快路径（树 Max ±compact ~148µs，pack ~180µs）。
        if (colsAligned && tmpFits) {
            const uint32_t foldBytes = static_cast<uint32_t>(foldElems * sizeof(half));
            auto ubTmp = ubT64_.template ReinterpretCast<uint8_t>()[foldBytes];
            uint32_t shape[2] = {numQHeads_, winCols};
            for (uint32_t t = 0; t < tokensSub_; ++t) {
                AscendC::ReduceMax<half, AscendC::Pattern::Reduce::RA, false>(
                    ubFold[t * winCols], ubPageH[t * rowStride], ubTmp, shape, true);
            }
            AscendC::PipeBarrier<PIPE_V>();
            return true;
        }
        constexpr uint8_t HEAD_REP_BLK = static_cast<uint8_t>(MSA_BLOCK_SIZE / MSA_HALF_PER_BLOCK);
        const uint32_t tokRepBlk = rowStride / MSA_HALF_PER_BLOCK;
        const bool tokenRepOk = (tokRepBlk <= MSA_MAX_REP_STRIDE) && (tokensSub_ <= MSA_MAX_REPEAT);
        if ((numQHeads_ & (numQHeads_ - 1U)) == 0U) {
            for (uint32_t pair = numQHeads_ >> 1U; pair > 0U; pair >>= 1U) {
                if (tokenRepOk && ((pair * nWin) < tokensSub_)) {
                    for (uint32_t h = 0; h < pair; ++h) {
                        for (uint32_t w = 0; w < nWin; ++w) {
                            const uint32_t base = h * winCols + w * MSA_BLOCK_SIZE;
                            AscendC::Max<half>(ubPageH[base], ubPageH[base], ubPageH[base + pair * winCols],
                                               static_cast<int32_t>(MSA_BLOCK_SIZE), static_cast<uint8_t>(tokensSub_),
                                               AscendC::BinaryRepeatParams(1, 1, 1, static_cast<uint8_t>(tokRepBlk),
                                                                           static_cast<uint8_t>(tokRepBlk),
                                                                           static_cast<uint8_t>(tokRepBlk)));
                        }
                    }
                } else {
                    for (uint32_t t = 0; t < tokensSub_; ++t) {
                        const uint32_t base = t * rowStride;
                        AscendC::Max<half>(
                            ubPageH[base], ubPageH[base], ubPageH[base + pair * winCols],
                            static_cast<int32_t>(MSA_BLOCK_SIZE), static_cast<uint8_t>(pair * nWin),
                            AscendC::BinaryRepeatParams(1, 1, 1, HEAD_REP_BLK, HEAD_REP_BLK, HEAD_REP_BLK));
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            return false;
        }
        for (uint32_t h = 1U; h < numQHeads_; ++h) {
            for (uint32_t t = 0; t < tokensSub_; ++t) {
                const uint32_t base = t * rowStride;
                AscendC::Max(ubPageH[base], ubPageH[base], ubPageH[base + h * winCols], static_cast<int32_t>(winCols));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        return false;
    }

    /// 掩码：窗内第 c 列对应 key token winTok + c，超出 visibleKeyEnd 的列置 -inf。
    /// 只作用在折叠后的首行，每 token 一条 128 位掩码 Duplicate。
    /// 返回本组是否真的填过 -inf：没填过时后续的下界保护就是恒等变换，可以省掉。
    __aicore__ inline bool ApplyMaskWindowHalf(const MsaTask& task, uint32_t blkBase,
                                               const AscendC::LocalTensor<half>& ubPageH, uint32_t rowStride,
                                               uint32_t nWin)
    {
        const int32_t winTok = static_cast<int32_t>(blkBase) * static_cast<int32_t>(MSA_BLOCK16_SIZE);
        const int32_t need = winTok + static_cast<int32_t>(nWin * MSA_BLOCK_SIZE);
        // 本组最「短视」的 token（mOff_ 对应 tOff0）：rightDown 下 tOff 越大可见越长；
        // 非 causal 则大家相同。tOff0 已盖住整组 → 无需逐 token 标量探测。
        if (tokensSub_ > 0U) {
            const uint32_t tOff0 = (task.mStart + mOff_) / numQHeads_;
            if (VisibleKeyEnd(task, static_cast<int32_t>(tOff0)) >= need) {
                return false;
            }
        }
        bool touched = false;
        for (uint32_t t = 0; t < tokensSub_; ++t) {
            const uint32_t rowInReq = task.mStart + mOff_ + t * numQHeads_;
            const uint32_t tOff = rowInReq / numQHeads_;
            const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
            const int32_t validCount = visibleKeyEnd - winTok;
            if (validCount >= static_cast<int32_t>(nWin * MSA_BLOCK_SIZE)) {
                continue;
            }
            // 组内逐窗裁：整窗不可见的填满 -inf，边界窗按位掩码填尾。
            for (uint32_t w = 0; w < nWin; ++w) {
                int32_t vc = validCount - static_cast<int32_t>(w * MSA_BLOCK_SIZE);
                if (vc < 0) {
                    vc = 0;
                }
                if (vc >= static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                    continue;
                }
                FillBlockTailFp16At(ubPageH, t * rowStride + w * MSA_BLOCK_SIZE, static_cast<uint32_t>(vc));
                touched = true;
            }
        }
        if (touched) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        return touched;
    }

    /// 非量化乒乓：把本 pass 的 S 搬进 ubS16Ping_[stage]，不阻塞 Vector。
    __aicore__ inline void IssueS16(const AscendC::GlobalTensor<ElementS>& gS, uint32_t rowOff, uint32_t rows,
                                    uint32_t stage)
    {
        if constexpr (IS_QUANT) {
            return;
        } else {
            const int32_t evt = static_cast<int32_t>(stage);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
            auto dst = ubS16Ping_[stage];
            if (rows < MSA_ROWS_PER_PASS) {
                AscendC::Duplicate(dst, static_cast<half>(MSA_FILL_VALUE), MSA_ROWS_PER_PASS * STILE_WIDTH);
                AscendC::PipeBarrier<PIPE_V>();
                // Duplicate 走 Vector，DataCopy 走 MTE2；不排空 V 会被 -inf 盖掉已搬入的 S。
                // M=16 满 pass 不走 Duplicate，所以 prefill 正常、decode/bf16 短 pass 会出 -inf 空洞。
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evt);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
            }
            AscendC::DataCopy(dst, gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], rows * STILE_WIDTH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
        }
    }

    __aicore__ inline void WaitS16(uint32_t stage)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<int32_t>(stage));
        ubS16_ = ubS16Ping_[stage];
    }

    __aicore__ inline void ReleaseS16(uint32_t stage)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<int32_t>(stage));
    }

    /// 处理一个 pass（最多 MSA_ROWS_PER_PASS 行 × MSA_BLOCKS_PER_STILE 个 block）。
    /// sLoaded：非量化乒乓路径已把 S 搬进 ubS16_，跳过本函数内的 GM→UB。
    __aicore__ inline void ProcessPass(const AscendC::GlobalTensor<ElementS>& gS, const MsaTask& task, uint32_t blkBase,
                                       uint32_t rowOff, uint32_t rows, bool sLoaded = false)
    {
        const bool allInvalid = (blkBase >= task.visibleEndBlk);

        if (!allInvalid) {
            if (!sLoaded) {
                // 短 pass 只拷了部分行；SegRowMax 固定按 16 行归约，未写入区域必须是 -inf。
                // 满 16 行时 DataCopy 覆盖整块 ubS_，可跳过 Duplicate。
                if (rows < MSA_ROWS_PER_PASS) {
                    if constexpr (IS_QUANT) {
                        AscendC::Duplicate(ubS_, MSA_FILL_VALUE, MSA_ROWS_PER_PASS * STILE_WIDTH);
                    } else {
                        AscendC::Duplicate(ubS16_, static_cast<half>(MSA_FILL_VALUE), MSA_ROWS_PER_PASS * STILE_WIDTH);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                // 上一 pass 的归约还在读 ubS_/ubS16_（score 已改为暂存在 UB，不再有
                // 逐 pass 的 MTE3 顺带排空 V），MTE2 覆盖该缓冲前必须显式等 V。
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                if constexpr (IS_QUANT) {
                    AscendC::DataCopy(ubS_, gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], rows * STILE_WIDTH);
                } else {
                    AscendC::DataCopy(ubS16_, gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], rows * STILE_WIDTH);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }

            if (isQuant_) {
                ApplyDequantScale(task, blkBase, MSA_ROWS_PER_PASS);
            }

            ApplyMask(task, blkBase, rowOff, rows);

            if constexpr (IS_QUANT) {
                SegRowMax();
            } else {
                SegRowMaxWhole();
            }
        } else {
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
        }

        FillInvalidBlocks(task, blkBase);
        ApplyLocalMaskUb(task, blkBase, rows);

        if (stageOn_) {
            StageScore(blkBase, rowOff - mOff_, rows);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            StoreScore(task, blkBase, rowOff, rows);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    /// 把本 pass 的 [rows, MSA_BLOCKS_PER_STILE] 归约结果搬到 score 暂存区对应的
    /// 行 × block 窗口（行内 stride 为 MSA_STAGE_BLOCKS），供整行 flush。
    __aicore__ inline void StageScore(uint32_t blkBase, uint32_t stageRow, uint32_t rows, uint32_t nWin = 1U)
    {
        const uint32_t colOff = blkBase - stageBlkBase_;
        const uint32_t nBlocks = nWin * MSA_BLOCKS_PER_STILE;
        AscendC::Adds<float>(ubStage_[stageRow * MSA_STAGE_BLOCKS + colOff], ubRowMax_, 0.0F,
                             static_cast<int32_t>(nBlocks), static_cast<uint8_t>(rows),
                             AscendC::UnaryRepeatParams(1, 1, MSA_STAGE_BLOCKS / MSA_FP32_PER_BLOCK,
                                                        static_cast<uint16_t>(nBlocks / MSA_FP32_PER_BLOCK)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// 打开以 base 为起点的 score 暂存窗口，整窗先填 -inf。
    __aicore__ inline void OpenStageWindow(uint32_t base)
    {
        stageBlkBase_ = base;
        stageBlkEnd_ = MsaMinU32(base + MSA_STAGE_BLOCKS, strideOutToken_);
        AscendC::Duplicate(ubStage_, MSA_FILL_VALUE, stageRows_ * MSA_STAGE_BLOCKS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// blk 越过当前窗时先 flush，再开下一窗（窗宽 MSA_STAGE_BLOCKS）。
    __aicore__ inline void AdvanceStageWindow(uint32_t blk)
    {
        if (!stageOn_) {
            return;
        }
        while (blk >= stageBlkBase_ + MSA_STAGE_BLOCKS) {
            FlushStage();
            OpenStageWindow(stageBlkBase_ + MSA_STAGE_BLOCKS);
        }
    }

    /// 当前窗中本 chunk 负责的列；writeTailFill 时 writeHi_ 延到 score 末维。
    __aicore__ inline void FlushStageToStrideEnd()
    {
        FlushStage();
        if constexpr (MSA_A5_USE_C2UB) {
            while (stageOn_ && stageBlkEnd_ < writeHi_) {
                OpenStageWindow(stageBlkEnd_);
                FlushStage();
            }
        }
    }

    /// 把 score 暂存窗口按行写回 GM：每行一次 ≥32B 对齐的连续 DataCopy。
    __aicore__ inline void FlushStage()
    {
        if ((rowInReqBase_ - mOff_) % MSA_ROW_TILE_M == 0U) {
            FlushStageRows<false>();
        } else {
            FlushStageRows<true>();
        }
    }

    template <bool TRIM_BORROWED_ROWS>
    __aicore__ inline void FlushStageRows()
    {
        if (!stageOn_ || stageBlkEnd_ <= stageBlkBase_) {
            return;
        }
        const uint32_t lo = (writeLo_ > stageBlkBase_) ? writeLo_ : stageBlkBase_;
        const uint32_t hi = (writeHi_ < stageBlkEnd_) ? writeHi_ : stageBlkEnd_;
        if (hi <= lo) {
            return;
        }
        const uint32_t count = hi - lo;
        const uint32_t ubOff = lo - stageBlkBase_;
        // GetMTile borrows preceding rows for a tail smaller than MSA_M_ALIGN.
        // Keep those rows for Cube, but leave their score writes to the previous task.
        uint32_t firstRow = 0U;
        if constexpr (TRIM_BORROWED_ROWS) {
            const uint32_t writeStart = MsaCeilDiv(rowInReqBase_ - mOff_, MSA_ROW_TILE_M) * MSA_ROW_TILE_M;
            firstRow = writeStart > rowInReqBase_ ? MsaMinU32(mSub_, writeStart - rowInReqBase_) : 0U;
            if (firstRow >= mSub_) {
                return;
            }
        }
        // Flush 只需 V→MTE3 同步；PIPE_ALL 在消融中无收益且偏重，已去掉。
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        uint32_t tOff = (rowInReqBase_ + firstRow) / numQHeads_;
        uint64_t tokenBase = static_cast<uint64_t>(tokenBase0_ + tOff) * strideOutToken_ + lo;
        if constexpr (IS_BLOCK16) {
            // head 间 maxpool 后一个 token 只有一个分数。score 的 head 维是 1，只写这一份。
            for (uint32_t r = firstRow / numQHeads_; r < stageRows_; ++r) {
                AscendC::DataCopy(gScore_[tokenBase], ubStage_[r * MSA_STAGE_BLOCKS + ubOff], count);
                tokenBase += strideOutToken_;
            }
        } else {
            uint32_t h = rowInReqBase_ + firstRow - tOff * numQHeads_;
            bool grouped = false;
            if constexpr (IS_MXFP4 && MX_FUSED_STAGE) {
                if (firstRow == 0U && mSub_ >= 32U && mSub_ > numQHeads_) {
                    const uint64_t srcStep = static_cast<uint64_t>(numQHeads_) * MSA_STAGE_BLOCKS;
                    const uint64_t srcGap = (srcStep - count) / MSA_FP32_PER_BLOCK;
                    const uint64_t dstGap = (strideOutToken_ - count) / MSA_FP32_PER_BLOCK;
                    if (numQHeads_ > 0U && mSub_ > numQHeads_ && count % MSA_FP32_PER_BLOCK == 0U && srcGap <= 65535U &&
                        dstGap <= 65535U) {
                        for (uint32_t firstRow = 0U; firstRow < numQHeads_; ++firstRow) {
                            const uint32_t head = (h + firstRow) % numQHeads_;
                            const uint32_t advance = (h + firstRow) / numQHeads_;
                            const uint32_t rows = (mSub_ - firstRow + numQHeads_ - 1U) / numQHeads_;
                            AscendC::DataCopyParams params(
                                static_cast<uint16_t>(rows), static_cast<uint16_t>(count / MSA_FP32_PER_BLOCK),
                                static_cast<uint16_t>(srcGap), static_cast<uint16_t>(dstGap));
                            AscendC::DataCopy(gScore_[tokenBase + static_cast<uint64_t>(advance) * strideOutToken_ +
                                                      static_cast<uint64_t>(head) * strideOutHead_],
                                              ubStage_[firstRow * MSA_STAGE_BLOCKS + ubOff], params);
                        }
                        grouped = true;
                    }
                }
            }
            if (!grouped) {
                for (uint32_t r = firstRow; r < mSub_; ++r) {
                    AscendC::DataCopy(gScore_[tokenBase + static_cast<uint64_t>(h) * strideOutHead_],
                                      ubStage_[r * MSA_STAGE_BLOCKS + ubOff], count);
                    if (++h == numQHeads_) {
                        h = 0;
                        tokenBase += strideOutToken_;
                    }
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    /// sparse_mode=3：rightDownCausal；否则仅 kv_len 截断。
    __aicore__ inline int32_t VisibleKeyEnd(const MsaTask& task, int32_t tOff) const
    {
        if (task.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN) {
            return MsaClampI32(task.kvLen - task.qLen + tOff + 1, 0, task.kvLen < 0 ? 0 : task.kvLen);
        }
        return task.kvLen < 0 ? 0 : task.kvLen;
    }

    /// 只对 [fullEndBlk, visibleEndBlk) 的边界 block 逐行置 -inf。
    /// 全可见 block 走无 mask 快通道；全不可见 block 的结果在 FillInvalidBlocks 中整体覆盖。
    __aicore__ inline void ApplyMask(const MsaTask& task, uint32_t blkBase, uint32_t rowOff, uint32_t rows)
    {
        if (task.fullEndBlk >= task.visibleEndBlk) {
            return;
        }
        const uint32_t jLo = (task.fullEndBlk > blkBase) ? (task.fullEndBlk - blkBase) : 0;
        if (jLo >= MSA_BLOCKS_PER_STILE) {
            return;
        }
        const uint32_t jHi =
            MsaMinU32(MSA_BLOCKS_PER_STILE, (task.visibleEndBlk > blkBase) ? (task.visibleEndBlk - blkBase) : 0);

        bool touched = false;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t rowInReq = task.mStart + rowOff + r;
            const uint32_t tOff = rowInReq / numQHeads_;
            const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
            for (uint32_t j = jLo; j < jHi; ++j) {
                const uint32_t blk = blkBase + j;
                int32_t validCount = visibleKeyEnd - static_cast<int32_t>(blk * MSA_BLOCK_SIZE);
                if (validCount < 0) {
                    validCount = 0;
                }
                if (validCount >= static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                    continue;
                }
                if constexpr (IS_QUANT) {
                    FillBlockTail((r * MSA_BLOCKS_PER_STILE + j) * MSA_BLOCK_SIZE, static_cast<uint32_t>(validCount));
                } else {
                    // 非量化归约在 fp16 域起步，mask 必须先落在 ubS16_ 上。
                    FillBlockTailFp16((r * MSA_BLOCKS_PER_STILE + j) * MSA_BLOCK_SIZE,
                                      static_cast<uint32_t>(validCount));
                }
                touched = true;
            }
        }
        if (touched) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// Maxpool 之后、Store 之前施加 local_mask。
    /// 用与 FillInvalidBlocks 相同的位掩码 Duplicate，保证 32B 对齐访问。
    __aicore__ inline void ApplyLocalMaskUb(const MsaTask& task, uint32_t blkBase, uint32_t rows, uint32_t nWin = 1U)
    {
        if (task.initBlocks == 0 && task.localBlocks == 0) {
            return;
        }
        const uint32_t nBlocks = nWin * MSA_BLOCKS_PER_STILE;
        const int32_t qBlock = task.startLoc;
        const int32_t localStart = (qBlock + 1 > static_cast<int32_t>(task.localBlocks)) ?
                                       (qBlock + 1 - static_cast<int32_t>(task.localBlocks)) :
                                       0;

        uint64_t initBits = 0;
        uint64_t localBits = 0;
        // 高分上限是本 M-tile 最后一行的 visibleEndBlk。块长 16 与 128 相同。
        const int32_t blkLimit = static_cast<int32_t>(task.visibleEndBlk);
        for (uint32_t j = 0; j < nBlocks; ++j) {
            const int32_t blk = static_cast<int32_t>(blkBase + j);
            if (blk < 0 || blk >= blkLimit) {
                continue;
            }
            if (static_cast<uint32_t>(blk) < task.initBlocks) {
                initBits |= (1ULL << j);
            }
            if (blk >= localStart && blk <= qBlock) {
                localBits |= (1ULL << j);
            }
        }
        initBits &= ~localBits; // local 覆盖 init

        const uint8_t repeats = static_cast<uint8_t>((rows * nBlocks + MSA_FP32_PER_REPEAT - 1) / MSA_FP32_PER_REPEAT);
        if (initBits != 0) {
            uint64_t bits = 0;
            for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / nBlocks; ++g) {
                bits |= (initBits << (g * nBlocks));
            }
            uint64_t mask[MSA_VEC_MASK_WORDS] = {bits, 0};
            AscendC::Duplicate<float>(ubRowMax_, MSA_LOCAL_SCORE_INIT, mask, repeats, 1,
                                      MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (localBits != 0) {
            uint64_t bits = 0;
            for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / nBlocks; ++g) {
                bits |= (localBits << (g * nBlocks));
            }
            uint64_t mask[MSA_VEC_MASK_WORDS] = {bits, 0};
            AscendC::Duplicate<float>(ubRowMax_, MSA_LOCAL_SCORE_LOCAL, mask, repeats, 1,
                                      MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// 把 ubS_[base + validCount, base + MSA_BLOCK_SIZE) 置为 -inf。
    /// 拆成「整 64-fp32 repeat 尾段」+「单 repeat 内的位掩码」两步，保证每次 Duplicate 都是 256B 对齐。
    __aicore__ inline void FillBlockTail(uint32_t base, uint32_t validCount)
    {
        FillBlockTailAt(ubS_, base, validCount);
    }

    /// 把某一行在某个 block 内 [validCount, MSA_BLOCK_SIZE) 的列置为 -inf。
    /// fp16 下一个 repeat 恰好 128 lane = 一个 block 的 8 个 datablock，
    /// 故一条带 128 位掩码的 Duplicate 即可。
    __aicore__ inline void FillBlockTailFp16At(const AscendC::LocalTensor<half>& ub, uint32_t base, uint32_t validCount)
    {
        if (validCount >= MSA_BLOCK_SIZE) {
            return;
        }
        uint64_t lo = 0;
        uint64_t hi = 0;
        if (validCount < MSA_HALF_PER_MASK_WORD) {
            lo = ~0ULL << validCount;
            hi = ~0ULL;
        } else {
            hi = ~0ULL << (validCount - MSA_HALF_PER_MASK_WORD);
        }
        uint64_t mask[MSA_VEC_MASK_WORDS] = {lo, hi};
        AscendC::Duplicate<half>(ub[base], FILL_VALUE_H, mask, 1, 1, MSA_BLOCK_SIZE / MSA_HALF_PER_BLOCK);
    }

    __aicore__ inline void FillBlockTailFp16(uint32_t base, uint32_t validCount)
    {
        FillBlockTailFp16At(ubS16_, base, validCount);
    }

    /// int8 路径：S 已是 Q · cast(K_int8)，再按列乘 scale[page,0,n] 完成反量化。
    /// 数学上 Q·(K_int8*s) = (Q·cast(K_int8))*s（int8→fp 对整数精确）。
    __aicore__ inline void ApplyDequantScale(const MsaTask& task, uint32_t blkBase, uint32_t rows)
    {
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            uint64_t scaleOff = 0;
            uint32_t nValidScale = MSA_BLOCK_SIZE;
            if (keyLayout_ == MSA_KEY_LAYOUT_TND) {
                const uint32_t tokenStart = task.cuKStart + blk * MSA_BLOCK_SIZE;
                // TND scale 为 packed [T2, N2]，N2=1 时按 token 连续存放，不能用 PA 的 page 步长。
                scaleOff = static_cast<uint64_t>(tokenStart);
                const uint32_t seqTok = blk * MSA_BLOCK_SIZE;
                nValidScale = 0;
                if (task.kvLen > 0 && seqTok < static_cast<uint32_t>(task.kvLen)) {
                    nValidScale = static_cast<uint32_t>(task.kvLen) - seqTok;
                    if (nValidScale > MSA_BLOCK_SIZE) {
                        nValidScale = MSA_BLOCK_SIZE;
                    }
                }
            } else {
                const int32_t page = PaPageId(task, blk);
                scaleOff = static_cast<uint64_t>(static_cast<uint32_t>(page)) * strideScalePage_;
            }
            // 上一轮 Mul 占用 ubDeqScale_，必须等 V 完成后再 MTE2 覆盖。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            if (nValidScale < MSA_BLOCK_SIZE) {
                AscendC::Duplicate(ubDeqScale_, 0.0f, MSA_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
            if (nValidScale > 0) {
                const uint32_t bytes = nValidScale * static_cast<uint32_t>(sizeof(float));
                if ((bytes % MSA_DATABLOCK_BYTES) == 0U) {
                    AscendC::DataCopy(ubDeqScale_, gScale_[scaleOff], nValidScale);
                } else {
                    AscendC::DataCopyExtParams params;
                    params.blockCount = 1;
                    params.blockLen = bytes;
                    params.srcStride = 0;
                    params.dstStride = 0;
                    AscendC::DataCopyPadExtParams<float> pad;
                    pad.isPad = false;
                    pad.leftPadding = 0;
                    pad.rightPadding = 0;
                    pad.paddingValue = 0;
                    AscendC::DataCopyPad(ubDeqScale_, gScale_[scaleOff], params, pad);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t base = r * STILE_WIDTH + j * MSA_BLOCK_SIZE;
                AscendC::Mul(ubS_[base], ubS_[base], ubDeqScale_, static_cast<int32_t>(MSA_BLOCK_SIZE));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// 把 ubS_ 视作 [MSA_REDUCE_ROWS, MSA_BLOCK_SIZE] 做行归约，结果落在 ubRowMax_[MSA_REDUCE_ROWS]，
    /// 其排布恰为 [MSA_ROWS_PER_PASS][MSA_BLOCKS_PER_STILE] 行主序。
    __aicore__ inline void SegRowMax()
    {
        constexpr uint32_t HALF_BLOCK = MSA_BLOCK_SIZE / 2;
        constexpr uint8_t SRC_REP_BLK = MSA_BLOCK_SIZE / MSA_FP32_PER_BLOCK; // 16
        constexpr uint8_t DST_REP_BLK = HALF_BLOCK / MSA_FP32_PER_BLOCK;     // 8

        AscendC::Max<float>(ubT64_, ubS_, ubS_[HALF_BLOCK], static_cast<int32_t>(MSA_FP32_PER_REPEAT),
                            static_cast<uint8_t>(MSA_REDUCE_ROWS),
                            AscendC::BinaryRepeatParams(1, 1, 1, DST_REP_BLK, SRC_REP_BLK, SRC_REP_BLK));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceMax<float, true>(ubT8_, ubT64_, static_cast<uint8_t>(MSA_REDUCE_ROWS),
                                             static_cast<int32_t>(MSA_FP32_PER_REPEAT), 1, 1, DST_REP_BLK);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceMax<float, true>(ubRowMax_, ubT8_,
                                             static_cast<uint8_t>(MSA_REDUCE_ROWS / MSA_FP32_PER_BLOCK),
                                             static_cast<int32_t>(MSA_FP32_PER_REPEAT), 1, 1, DST_REP_BLK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// 非量化路径：一个 sparse block（128 个 fp16）恰好等于向量单元一个满 repeat，
    /// 故整个 pass 的分段 RowMax 可由单条 WholeReduceMax 完成——每 repeat 归约一个
    /// block，128 次迭代按 [row][block] 行主序密集写出，与 ubRowMax_ 的排布一致。
    /// 相比「配对 Max + Cast + 两级 BlockReduceMax」省掉 3 条全宽向量指令和 3 次 barrier。
    /// max 在 fp16 上无舍入、Cast fp16→fp32 精确且保序，结果与 fp32 归约逐 bit 一致。
    __aicore__ inline void SegRowMaxWhole()
    {
        constexpr uint8_t SRC_REP_BLK = MSA_BLOCK_SIZE / MSA_HALF_PER_BLOCK; // 8
        AscendC::WholeReduceMax<half>(ubRed16_, ubS16_, static_cast<int32_t>(MSA_BLOCK_SIZE),
                                      static_cast<int32_t>(MSA_REDUCE_ROWS), 1, 1, SRC_REP_BLK,
                                      AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(ubRowMax_, ubRed16_, AscendC::RoundMode::CAST_NONE, MSA_REDUCE_ROWS);
        AscendC::PipeBarrier<PIPE_V>();

        // mask/短 pass 填充在 fp16 域是 -inf，归约后抬回 fp32 的 MSA_FILL_VALUE，
        // 与 int8 路径及原实现的填充值保持一致（真实分值远大于 -3.4e38，不受影响）。
        AscendC::Maxs<float>(ubRowMax_, ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// blk >= visibleEndBlk 的位置整体置为 -inf。
    __aicore__ inline void FillInvalidBlocks(const MsaTask& task, uint32_t blkBase)
    {
        const uint32_t invStart = (task.visibleEndBlk > blkBase) ? (task.visibleEndBlk - blkBase) : 0;
        if (invStart >= MSA_BLOCKS_PER_STILE) {
            return;
        }
        if (invStart == 0) {
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        uint64_t groupBits = 0;
        for (uint32_t k = invStart; k < MSA_BLOCKS_PER_STILE; ++k) {
            groupBits |= (1ULL << k);
        }
        uint64_t bits = 0;
        for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / MSA_BLOCKS_PER_STILE; ++g) {
            bits |= (groupBits << (g * MSA_BLOCKS_PER_STILE));
        }
        uint64_t mask[MSA_VEC_MASK_WORDS] = {bits, 0};
        AscendC::Duplicate<float>(ubRowMax_, MSA_FILL_VALUE, mask,
                                  static_cast<uint8_t>(MSA_REDUCE_ROWS / MSA_FP32_PER_REPEAT), 1,
                                  MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreScore(const MsaTask& task, uint32_t blkBase, uint32_t rowOff, uint32_t rows)
    {
        // 行 = (token, head) 扁平索引、head 为低位，故 pass 内逐行递推即可，
        // 无需每行一次标量整除（整除在 AIV 上是数十 cycle 的长指令）。
        const uint32_t writeStart = MsaCeilDiv(task.mStart, MSA_ROW_TILE_M) * MSA_ROW_TILE_M;
        const uint32_t rowBegin = task.mStart + rowOff;
        const uint32_t firstRow = writeStart > rowBegin ? MsaMinU32(rows, writeStart - rowBegin) : 0U;
        uint32_t tOff = (rowBegin + firstRow) / numQHeads_;
        uint32_t h = rowBegin + firstRow - tOff * numQHeads_;
        uint64_t tokenBase = static_cast<uint64_t>(task.cuQStart + tOff) * strideOutToken_ + blkBase;
        for (uint32_t r = firstRow; r < rows; ++r) {
            if constexpr (IS_BLOCK16) {
                if (h == 0U) {
                    AscendC::DataCopy(gScore_[tokenBase], ubRowMax_[r * MSA_BLOCKS_PER_STILE], MSA_BLOCKS_PER_STILE);
                }
            } else {
                AscendC::DataCopy(gScore_[tokenBase + static_cast<uint64_t>(h) * strideOutHead_],
                                  ubRowMax_[r * MSA_BLOCKS_PER_STILE], MSA_BLOCKS_PER_STILE);
            }
            if (++h == numQHeads_) {
                h = 0;
                tokenBase += strideOutToken_;
            }
        }
    }

    __aicore__ inline void ApplyMaskPage(const MsaTask& task, uint32_t blk, const AscendC::LocalTensor<float>& ubPage)
    {
        if (blk < task.fullEndBlk || blk >= task.visibleEndBlk) {
            return;
        }
        bool touched = false;
        for (uint32_t r = 0; r < mSub_; ++r) {
            const uint32_t rowInReq = task.mStart + mOff_ + r;
            const uint32_t tOff = rowInReq / numQHeads_;
            const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
            int32_t validCount = visibleKeyEnd - static_cast<int32_t>(blk * MSA_BLOCK_SIZE);
            if (validCount < 0) {
                validCount = 0;
            }
            if (validCount >= static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                continue;
            }
            const uint32_t vc = static_cast<uint32_t>(validCount);
            const uint32_t v0 = (vc >= 64U) ? 64U : vc;
            const uint32_t v1 = (vc >= 64U) ? (vc - 64U) : 0U;
            FillBlockTailAt(ubPage, r * 64U, v0, 64U);
            FillBlockTailAt(ubPage[mSub_ * 64U], r * 64U, v1, 64U);
            touched = true;
        }
        if (touched) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// packed [mSub,128] fp16：每行一条 128-bit Duplicate。
    __aicore__ inline void ApplyMaskPageHalf(const MsaTask& task, uint32_t blk,
                                             const AscendC::LocalTensor<half>& ubPage,
                                             uint32_t rowStride = MSA_BLOCK_SIZE, uint32_t colOffset = 0U)
    {
        if (blk < task.fullEndBlk || blk >= task.visibleEndBlk) {
            return;
        }
        // Reuse one causal tail mask for the contiguous head rows of a query.
        if constexpr (IS_MXFP4 && MX_FUSED_STAGE && !IS_BLOCK16 && !IS_QUANT) {
            if (numQHeads_ > 1U && mSub_ <= MSA_MAX_REPEAT && rowStride % MSA_HALF_PER_BLOCK == 0U &&
                rowStride / MSA_HALF_PER_BLOCK <= MSA_MAX_REP_STRIDE) {
                // SDK mask-array Duplicate interprets repeatTime only in normal
                // mode. Preserve CTRL[56] across this explicit normal-mode scope.
                const bool restoreCounter = AscendC::Internal::IsCounterMode();
                if (restoreCounter) {
                    AscendC::SetMaskNorm();
                }
                bool touched = false;
                for (uint32_t r = 0; r < mSub_;) {
                    const uint32_t rowInReq = task.mStart + mOff_ + r;
                    const uint32_t rows = MsaMinU32(numQHeads_ - rowInReq % numQHeads_, mSub_ - r);
                    const uint32_t tOff = rowInReq / numQHeads_;
                    const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
                    int32_t validCount = visibleKeyEnd - static_cast<int32_t>(blk * MSA_BLOCK_SIZE);
                    if (validCount < 0) {
                        validCount = 0;
                    }
                    if (validCount < static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                        uint64_t lo = 0U;
                        uint64_t hi = 0U;
                        if (validCount < static_cast<int32_t>(MSA_HALF_PER_MASK_WORD)) {
                            lo = ~0ULL << static_cast<uint32_t>(validCount);
                            hi = ~0ULL;
                        } else {
                            hi = ~0ULL << (static_cast<uint32_t>(validCount) - MSA_HALF_PER_MASK_WORD);
                        }
                        uint64_t mask[MSA_VEC_MASK_WORDS] = {lo, hi};
                        AscendC::Duplicate<half>(ubPage[r * rowStride + colOffset], FILL_VALUE_H, mask,
                                                 static_cast<uint8_t>(rows), 1,
                                                 static_cast<uint8_t>(rowStride / MSA_HALF_PER_BLOCK));
                        touched = true;
                    }
                    r += rows;
                }
                if (touched) {
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if (restoreCounter) {
                    AscendC::SetMaskCount();
                }
                return;
            }
        }
        bool touched = false;
        for (uint32_t r = 0; r < mSub_; ++r) {
            const uint32_t rowInReq = task.mStart + mOff_ + r;
            const uint32_t tOff = rowInReq / numQHeads_;
            const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
            int32_t validCount = visibleKeyEnd - static_cast<int32_t>(blk * MSA_BLOCK_SIZE);
            if (validCount < 0) {
                validCount = 0;
            }
            if (validCount >= static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                continue;
            }
            FillBlockTailFp16At(ubPage, r * rowStride + colOffset, static_cast<uint32_t>(validCount));
            touched = true;
        }
        if (touched) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// local/init 整页同一值：归约后再盖 packed 列，避免 Duplicate 整页 S。
    __aicore__ inline void ApplyLocalMaskRowMax(const MsaTask& task, uint32_t blk,
                                                const AscendC::LocalTensor<float>& dst)
    {
        if (task.initBlocks == 0 && task.localBlocks == 0) {
            return;
        }
        const int32_t qBlock = task.startLoc;
        const int32_t localStart = (qBlock + 1 > static_cast<int32_t>(task.localBlocks)) ?
                                       (qBlock + 1 - static_cast<int32_t>(task.localBlocks)) :
                                       0;
        const int32_t iblk = static_cast<int32_t>(blk);
        bool isLocal = (iblk >= localStart && iblk <= qBlock && iblk < static_cast<int32_t>(task.visibleEndBlk));
        bool isInit = (!isLocal) && (blk < task.initBlocks) && (blk < task.visibleEndBlk);
        if (!isLocal && !isInit) {
            return;
        }
        const float fill = isLocal ? MSA_LOCAL_SCORE_LOCAL : MSA_LOCAL_SCORE_INIT;
        AscendC::Duplicate(dst, fill, mSub_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// ubRed16_ 已是 [row][block]（与 ProcessWindowUb / StageScore 同布局）。
    /// 只 Cast 实际 mSub*8，去掉 TransDataTo5HD + take8。
    /// 满可见 stile（无 -inf 填、无 local/init mask）且 mSub 为偶数：half→float Cast
    /// 直接按 stage 行距写入 ubStage_，省掉 ubRowMax_ 中转与 StageScore 的 Adds。
    /// half 行宽 8 非整 32B，故每次 Cast 两行（16 half = 1 datablock），用 dstBlkStride
    /// 把两行 float block 拉开到相邻 stage 行；P1 的 [row][block] 布局不变。
    /// 注：行距补到 16 half 对齐过（19.45 B1），RV_VSTAI/VSTUI 条数不变，已撤。
    __aicore__ inline void ReduceStileToStage(const MsaTask& task, uint32_t blkBase)
    {
#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
        if constexpr (IS_MXFP4 && MX_FUSED_STAGE && !IS_BLOCK16 && !IS_QUANT) {
            if (mxDirectStileBase_ == blkBase && mxDirectColumnMask_ != 0U) {
                const uint32_t endBlk =
                    MsaMinU32(blkBase + MSA_BLOCKS_PER_STILE, MsaMinU32(task.visibleEndBlk, task.sBlkEnd));
                const uint32_t endCol = (endBlk > blkBase) ? endBlk - blkBase : 0U;
                auto* src = reinterpret_cast<__ubuf__ half*>(ubRed16_.GetPhyAddr());
                auto* dst = reinterpret_cast<__ubuf__ float*>(ubStage_.GetPhyAddr()) + blkBase - stageBlkBase_;
                // Direct ownership is a contiguous low-column prefix: this
                // task visits groups in increasing order, all eligibility guards
                // are stile/task invariant, and directEnd is one prefix bound.
                uint32_t firstFallbackCol = 0U;
                while (firstFallbackCol < MSA_BLOCKS_PER_STILE &&
                       (mxDirectColumnMask_ & (1U << firstFallbackCol)) != 0U) {
                    ++firstFallbackCol;
                }
                if (stileNeedFill_) {
                    MxScatterFallbackColumnsVf<true, true>(src, dst, mSub_, endCol, firstFallbackCol, MSA_STAGE_BLOCKS);
                } else if ((mSub_ & 1U) != 0U) {
                    MxScatterFallbackColumnsVf<false, true>(src, dst, mSub_, endCol, firstFallbackCol,
                                                            MSA_STAGE_BLOCKS);
                } else {
                    MxScatterFallbackColumnsVf<false, false>(src, dst, mSub_, endCol, firstFallbackCol,
                                                             MSA_STAGE_BLOCKS);
                }
                AscendC::PipeBarrier<PIPE_V>();
                return;
            }
        }
#endif
        const uint32_t nElem = mSub_ * MSA_BLOCKS_PER_STILE;
        const bool noLocalMask = (task.initBlocks == 0U) && (task.localBlocks == 0U);
        // stage 行数在 block16 下是 token 数，与 mSub_ 不等时这条融合写的行距不成立。
        if (stageOn_ && !stileNeedFill_ && noLocalMask && (stageRows_ == mSub_) && (mSub_ >= 2U) &&
            ((mSub_ & 1U) == 0U)) {
            constexpr uint64_t kMaskTwoRows = static_cast<uint64_t>(2U * MSA_BLOCKS_PER_STILE);           // 16
            constexpr uint8_t kStageRowBlk = static_cast<uint8_t>(MSA_STAGE_BLOCKS / MSA_FP32_PER_BLOCK); // 32
            const uint8_t repeats = static_cast<uint8_t>(mSub_ >> 1U);
            // 与 StageScore 同口径：写的是当前 stage 窗内的列偏移，不是绝对 block 下标。
            AscendC::Cast(ubStage_[blkBase - stageBlkBase_], ubRed16_, AscendC::RoundMode::CAST_NONE, kMaskTwoRows,
                          repeats,
                          AscendC::UnaryRepeatParams(kStageRowBlk, 1, static_cast<uint8_t>(kStageRowBlk << 1U), 1));
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        AscendC::Cast(ubRowMax_, ubRed16_, AscendC::RoundMode::CAST_NONE, nElem);
        AscendC::PipeBarrier<PIPE_V>();
        if (stileNeedFill_) {
            AscendC::Maxs<float>(ubRowMax_, ubRowMax_, MSA_FILL_VALUE, static_cast<int32_t>(nElem));
            AscendC::PipeBarrier<PIPE_V>();
        }
        ApplyLocalMaskUb(task, blkBase, mSub_);
        StageScore(blkBase, 0, mSub_);
    }

    /// 把密排 mSub 个 half 散射到 ubRed16_ 的第 col 列（[row][block]）。
    __aicore__ inline void ScatterRedCol(const AscendC::LocalTensor<half>& src, uint32_t col, uint32_t rows)
    {
        for (uint32_t r = 0; r < rows; ++r) {
            ubRed16_.SetValue(col + r * MSA_BLOCKS_PER_STILE, src.GetValue(r));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyDequantScalePage(const MsaTask& task, uint32_t blk, AscendC::LocalTensor<float> ubPage)
    {
        uint64_t scaleOff = 0;
        uint32_t nValidScale = MSA_BLOCK_SIZE;
        if (keyLayout_ == MSA_KEY_LAYOUT_TND) {
            const uint32_t tokenStart = task.cuKStart + blk * MSA_BLOCK_SIZE;
            scaleOff = static_cast<uint64_t>(tokenStart);
            const uint32_t seqTok = blk * MSA_BLOCK_SIZE;
            nValidScale = 0;
            if (task.kvLen > 0 && seqTok < static_cast<uint32_t>(task.kvLen)) {
                nValidScale = static_cast<uint32_t>(task.kvLen) - seqTok;
                if (nValidScale > MSA_BLOCK_SIZE) {
                    nValidScale = MSA_BLOCK_SIZE;
                }
            }
        } else {
            const int32_t page = PaPageId(task, blk);
            scaleOff = static_cast<uint64_t>(static_cast<uint32_t>(page)) * strideScalePage_;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        if (nValidScale < MSA_BLOCK_SIZE) {
            AscendC::Duplicate(ubDeqScale_, 0.0f, MSA_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }
        if (nValidScale > 0) {
            const uint32_t bytes = nValidScale * static_cast<uint32_t>(sizeof(float));
            if ((bytes % MSA_DATABLOCK_BYTES) == 0U) {
                AscendC::DataCopy(ubDeqScale_, gScale_[scaleOff], nValidScale);
            } else {
                AscendC::DataCopyExtParams params;
                params.blockCount = 1;
                params.blockLen = bytes;
                params.srcStride = 0;
                params.dstStride = 0;
                AscendC::DataCopyPadExtParams<float> pad;
                pad.isPad = false;
                pad.leftPadding = 0;
                pad.rightPadding = 0;
                pad.paddingValue = 0;
                AscendC::DataCopyPad(ubDeqScale_, gScale_[scaleOff], params, pad);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        for (uint32_t r = 0; r < mSub_; ++r) {
            AscendC::Mul(ubPage[r * MSA_FP32_PER_REPEAT], ubPage[r * MSA_FP32_PER_REPEAT], ubDeqScale_,
                         MSA_FP32_PER_REPEAT);
            AscendC::Mul(ubPage[mSub_ * MSA_FP32_PER_REPEAT + r * MSA_FP32_PER_REPEAT],
                         ubPage[mSub_ * MSA_FP32_PER_REPEAT + r * MSA_FP32_PER_REPEAT],
                         ubDeqScale_[MSA_FP32_PER_REPEAT], MSA_FP32_PER_REPEAT);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FillBlockTailAt(AscendC::LocalTensor<float> s, uint32_t base, uint32_t validCount,
                                           uint32_t width = MSA_BLOCK_SIZE)
    {
        if (validCount >= width) {
            return;
        }
        const uint32_t alignedStart = MsaCeilDiv(validCount, MSA_FP32_PER_REPEAT) * MSA_FP32_PER_REPEAT;
        if (alignedStart < width) {
            AscendC::Duplicate(s[base + alignedStart], MSA_FILL_VALUE, width - alignedStart);
        }
        if (validCount < alignedStart) {
            const uint32_t repeatBase = base + (validCount / MSA_FP32_PER_REPEAT) * MSA_FP32_PER_REPEAT;
            const uint32_t lane = validCount % MSA_FP32_PER_REPEAT;
            uint64_t bits = (lane == 0) ? ~0ULL : (~0ULL << lane);
            uint64_t mask[MSA_VEC_MASK_WORDS] = {bits, 0};
            AscendC::Duplicate<float>(s[repeatBase], MSA_FILL_VALUE, mask, 1, 1,
                                      MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
        }
    }

    AscendC::GlobalTensor<float> gScore_;
    AscendC::GlobalTensor<float> gScale_;
    AscendC::GlobalTensor<int32_t> gBlockTable_;
    AscendC::LocalTensor<float> ubS_;
    AscendC::LocalTensor<float> ubSPing_[MSA_A5_S_STAGES];
    AscendC::LocalTensor<half> ubS16_;
    AscendC::LocalTensor<half> ubS16Ping_[S16_STAGES];
    AscendC::LocalTensor<float> ubT64_;
    AscendC::LocalTensor<half> ubStash_;
    AscendC::LocalTensor<float> ubT8_;
    AscendC::LocalTensor<float> ubRowMax_;
    AscendC::LocalTensor<float> ubDeqScale_;
    AscendC::LocalTensor<half> ubRed16_;
    AscendC::LocalTensor<float> ubStage_;

    // 当前任务内本 subcore 的行区间与 score 暂存窗口。
    uint32_t mOff_ = 0;
    uint32_t mSub_ = 0;
    uint32_t tokensSub_ = 0; // BSND：本 subcore 的 token 数 = mSub_ / numQHeads_
    uint32_t stageRows_ = 0; // stage 的有效行数（BSND 为 token 数，其余等于 mSub_）
    uint32_t rowInReqBase_ = 0;
    uint32_t tokenBase0_ = 0;
    uint32_t stageBlkBase_ = 0;
    uint32_t stageBlkEnd_ = 0;
    uint32_t writeLo_ = 0;
    uint32_t writeHi_ = 0;
    bool stageOn_ = false;
    uint32_t mxDirectColumnMask_ = 0U;
    uint32_t mxDirectStileBase_ = ~0U;
    bool stileNeedFill_ = true; // 当前 stile 是否预填了 -inf（不满/有不可见列）

    uint32_t numQHeads_ = 1;
    uint32_t strideOutHead_ = 0;
    uint32_t strideOutToken_ = 0;
    /// PA 页表按存储页宽排布，不是 score 末维（那个会 RoundUp 到 16）。
    /// 用 score 末维当行距时，batch>0 会读到页表分配块外面的脏页号，scale 的 MTE 地址非法。
    __aicore__ inline int32_t PaPageId(const MsaTask& task, uint32_t blk) const
    {
        uint32_t pageIdx = blk;
        if (pageSize_ != 0U) {
            pageIdx = static_cast<uint32_t>((static_cast<uint64_t>(blk) * blockSize_) / pageSize_);
        }
        const uint32_t width = tableWidth_ > 0U ? tableWidth_ : 1U;
        if (pageIdx >= width) {
            pageIdx = width - 1U;
        }
        return gBlockTable_.GetValue(static_cast<uint64_t>(task.batchIdx) * width + pageIdx);
    }

    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t tableWidth_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t pageSize_ = 0;
    uint32_t strideScalePage_ = 0;
    uint32_t strideScaleHead_ = 1;
    uint32_t keyLayout_ = MSA_KEY_LAYOUT_BBND;
    uint32_t totalK_ = 0;
    bool isQuant_ = false;
};

} // namespace MsaIndexScoreNs

#endif // MSA_SEG_ROW_MAX_EPILOGUE_ARCH35_H
