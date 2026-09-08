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
 * \file gbsag_gather_and_scatter.hpp
 * \brief Gbsag Gather and Scatterx Kernel Implementation
 */

#ifndef GBSAG_EPILOGUE_GBSAG_GATHER_AND_SCATTER_HPP
#define GBSAG_EPILOGUE_GBSAG_GATHER_AND_SCATTER_HPP

#include "../../../attn_infra/arch/gbsag_resource.hpp"
#include "kernel_operator.h"

namespace GBSAG {

// 单个 Packet 最多聚合的有效 Q 行数，与 Cube M 方向上限保持一致。
constexpr uint32_t Q_PACKET_AGGREGATE_M = 128;
// Packet 内最多记录的 tile 数量；tile 至少 1 行，因此不超过 128。
constexpr uint32_t Q_PACKET_MAX_SEGMENTS = 128;

/**
 * @brief tile 行数：G≤128 时一个 tile 覆盖整个 GQA group；G>128 时按 128 head 分片。
 */
__aicore__ inline uint32_t GetQPacketTileRows(uint32_t groupSize)
{
    return groupSize < Q_PACKET_AGGREGATE_M ? groupSize : Q_PACKET_AGGREGATE_M;
}

/**
 * @brief tileRows 的 log2 值；非 2 的幂（G=3/5 等）返回 0xFF 表示走除法回退。
 *
 * AIV 标量核无硬件除法，`/ tileRows` 会展开为软件例程（数十周期 + 调用开销）；
 * 逐拍求段号时 2 的幂 tileRows（G∈{1,2,4,8,...,128}，全部 perf 档）必须用移位。
 */
__aicore__ inline uint32_t GetQPacketTileShift(uint32_t tileRows)
{
    if ((tileRows & (tileRows - 1)) != 0) {
        return 0xFF;
    }
    uint32_t shift = 0;
    while ((1u << shift) < tileRows) {
        ++shift;
    }
    return shift;
}

/**
 * @brief 段号计算：shift 有效时移位，否则除法回退（与 v / tileRows 严格等价）。
 */
__aicore__ inline uint32_t SegIndexByTileRows(uint32_t v, uint32_t tileRows, uint32_t tileShift)
{
    return tileShift != 0xFF ? (v >> tileShift) : v / tileRows;
}

/**
 * @brief 每 Packet 的 tile 数（整 token 切分：floor(128 / tileRows)，恒 >= 1）。
 */
__aicore__ inline uint32_t GetQPacketTilesPerPacket(uint32_t groupSize)
{
    return Q_PACKET_AGGREGATE_M / GetQPacketTileRows(groupSize);
}

/**
 * @brief 组聚合 Packet 的精简元数据。
 *
 * 整 token 切分下 tile 布局恒均匀，消费端按闭式推导：
 *   packStart[s]   = s * tileRows
 *   segmentRows[s] = tileRows
 *   headStart[s]   = headStart0（G≤128 恒 0；G>128 为 head 分片偏移）
 * 仅 token 号需要数组存储（qStart，AIV 侧顺序填充）；AIC 只读标量字段。
 */
struct QPacket {
    uint32_t rows;                          // 当前 Packet 的实际有效行数（= tileCount * tileRows）。
    uint32_t tileCount;                     // 当前 Packet 包含的 tile 数量。
    uint32_t headStart0;                    // tile 首 head 偏移（G>128 分片时非 0）。
    uint32_t qStart[Q_PACKET_MAX_SEGMENTS]; // tile 的 token 号（Q 序列内行号）。
};

/**
 * @brief 当前 AIV 在一个 Packet 内负责的连续行范围。
 */
struct QPacketAivRange {
    uint32_t begin; // 本 AIV 负责的 Packet 起始行，包含。
    uint32_t end;   // 本 AIV 负责的 Packet 结束行，不包含。
    uint32_t rows;  // 本 AIV 实际负责的行数。
};

/**
 * @brief 将 Packet 行沿 M 方向分配给一对 AIV。
 * @param packetRows 当前 Packet 的实际有效行数。
 * @return 当前 AIV 对应的半开行区间。
 */
__aicore__ inline QPacketAivRange GetQPacketAivRange(uint32_t packetRows)
{
    // 奇数行时前一个 AIV 多处理一行，保证两个 AIV 的负载差不超过一行。
    const uint32_t splitRow = packetRows / 2 + packetRows % 2;
    const bool isSecondAiv = (AscendC::GetBlockIdx() % 2) != 0;
    const uint32_t begin = isSecondAiv ? splitRow : 0;
    const uint32_t end = isSecondAiv ? packetRows : splitRow;
    return {begin, end, end - begin};
}

/**
 * @brief 根据 TND/BNSD 布局计算指定 (token, head) 在 GM Tensor 中的元素偏移。
 * @param qBatchBaseOffset 当前 batch 的 Q 数据元素基址。
 * @param qHead 当前 Q head 索引（组聚合时为 qHeadBegin + head 偏移）。
 * @param qSeq 当前 batch 内的 Q 序列行号（token 号）。
 * @param numHeads Q head 总数。
 * @param qSeqlen 当前 batch 的 Q 序列长度。
 * @param headDim 单个 head 的特征维度。
 * @param inputLayout 0 表示 TND，2 表示 BSND，其他值表示 BNSD。
 * @return 指定 Q 行首元素相对 Tensor 起点的元素偏移。
 */
__aicore__ inline uint64_t GetQPacketRowOffset(uint64_t qBatchBaseOffset, uint32_t qHead, uint32_t qSeq,
                                               uint32_t numHeads, uint32_t qSeqlen, uint32_t headDim,
                                               uint32_t inputLayout)
{
    if (inputLayout == 0) {
        return qBatchBaseOffset + static_cast<uint64_t>(qSeq) * numHeads * headDim + qHead * headDim;
    }
    if (inputLayout == 2) {
        return qBatchBaseOffset + (static_cast<uint64_t>(qSeq) * numHeads + qHead) * headDim;
    }
    return qBatchBaseOffset + (static_cast<uint64_t>(qHead) * qSeqlen + qSeq) * headDim;
}

/**
 * @brief 整 token 切分的组聚合 Packet 生成器。
 *
 * 聚合器只负责索引、游标和 Packet 标量计算，不持有任何 UB Resource 或 LocalTensor。
 * AIC 调用 AggregateScalars（纯闭式标量，零走查）；AIV 调用 Aggregate
 * （标量 + qStart 顺序填充）。两条路径由同一套闭式常量驱动，标量结果恒一致。
 */
class GBSAGAggregator {
public:
    /**
     * @brief 一个稀疏索引行对应的 Packet 聚合参数。
     */
    struct Params {
        AscendC::GlobalTensor<int32_t> rsvdBlockIdx; // KV block 对应的稀疏 Q block 索引 Tensor。
        uint64_t idxBase;                            // 当前稀疏索引行在 rsvdBlockIdx 中的元素起点。
        int32_t sparseCount;                         // 当前索引行中的有效 Q block 数量。
        uint32_t qBlockNum;                          // 当前 Q 序列包含的 Q block 总数。
        uint32_t blockShapeX;                        // 单个 Q block 的理论行数。
        uint32_t qSeqlen;                            // 当前 batch 的实际 Q 序列长度。
        uint32_t groupSize;                          // GQA group 大小（numHeads / kvHeads）。

        __aicore__ inline Params() {}

        __aicore__ inline Params(AscendC::GlobalTensor<int32_t> rsvdBlockIdx_, uint64_t idxBase_, int32_t sparseCount_,
                                 uint32_t qBlockNum_, uint32_t blockShapeX_, uint32_t qSeqlen_, uint32_t groupSize_)
            : rsvdBlockIdx(rsvdBlockIdx_),
              idxBase(idxBase_),
              sparseCount(sparseCount_),
              qBlockNum(qBlockNum_),
              blockShapeX(blockShapeX_),
              qSeqlen(qSeqlen_),
              groupSize(groupSize_)
        {}
    };

    /**
     * @brief 构造 Packet 聚合器；Init 时一次算出闭式常量与总 tile 数。
     */
    __aicore__ inline GBSAGAggregator()
        : walkInit_(false),
          walkSlot_(0),
          walkRowInBlock_(0),
          walkChunk_(0),
          token_(0)
    {}

    /**
     * @brief 绑定当前稀疏索引行，计算闭式常量并重置计数器。
     * @param params 当前 KV head group、K block 对应的聚合参数。
     */
    __aicore__ inline void Init(Params const &params)
    {
        params_ = params;
        // groupSize 至少为 1；GQA 场景恒 >= 1。
        const uint32_t groupSize = params_.groupSize > 0 ? params_.groupSize : 1;
        tileRows_ = GetQPacketTileRows(groupSize);
        tilesPerPacket_ = GetQPacketTilesPerPacket(groupSize);
        tilesPerToken_ = (groupSize + Q_PACKET_AGGREGATE_M - 1) / Q_PACKET_AGGREGATE_M;
        totalTiles_ = EstimateTotalTiles();
        remainingTiles_ = totalTiles_;
        tilePos_ = 0;
        walkInit_ = false;
    }

    /**
     * @brief 计算当前稀疏索引行需要生成的 Packet 数量。
     * @return 按整 token 切分（floor(128/tileRows) 个 tile/包）后的包数。
     */
    __aicore__ inline uint32_t GetPacketCount() const
    {
        if (totalTiles_ == 0) {
            return 0;
        }
        return (totalTiles_ + tilesPerPacket_ - 1) / tilesPerPacket_;
    }

    /**
     * @brief AIC 路径：仅闭式标量，零走查、零数组写。
     *
     * rows/tileCount/headStart0 与 AIV 路径完全一致（同一计数器驱动）。
     */
    __aicore__ inline void AggregateScalars(QPacket &packet)
    {
        const uint32_t tiles = remainingTiles_ < tilesPerPacket_ ? remainingTiles_ : tilesPerPacket_;
        packet.tileCount = tiles;
        packet.rows = tiles * tileRows_;
        // G>128 时同一 token 按 128 head 分片，包首分片偏移由 tile 流位置决定。
        packet.headStart0 = tilesPerToken_ > 1 ? (tilePos_ % tilesPerToken_) * Q_PACKET_AGGREGATE_M : 0;
        remainingTiles_ -= tiles;
        tilePos_ += tiles;
    }

    /**
     * @brief AIV 路径：闭式标量 + qStart 顺序填充。
     *
     * qStart 只在 AIV 侧填充（Gather/LSE/Scatter 均在 VEC 核）；
     * 走查为顺序推进，无逐 head 簿记。输入契约：稀疏索引升序且有效，
     * 估算与实际 tile 数一致；若上游给出非法索引，包按实际走查截断。
     */
    __aicore__ inline void Aggregate(QPacket &packet)
    {
        AggregateScalars(packet);
        if (!walkInit_) {
            walkSlot_ = 0;
            walkRowInBlock_ = 0;
            walkChunk_ = 0;
            AdvanceToValidToken();
            walkInit_ = true;
        }
        uint32_t filled = 0;
        for (; filled < packet.tileCount; ++filled) {
            if (walkSlot_ >= static_cast<uint32_t>(params_.sparseCount)) {
                // 估算多于实际（非法索引兜底）：按实际 tile 截断，避免搬入越界 token。
                break;
            }
            packet.qStart[filled] = token_;
            AdvanceWalk();
        }
        if (filled < packet.tileCount) {
            packet.tileCount = filled;
            packet.rows = filled * tileRows_;
        }
    }

private:
    /**
     * @brief 估算当前索引行的有效 tile 总数（与整 token 切分一致）。
     *
     * 默认每个有效索引贡献一个完整 Q block 的 token 数，命中尾 block 时
     * 扣除越界行；G>128 时每个 token 贡献 ceil(G/128) 个 tile。
     */
    __aicore__ inline uint32_t EstimateTotalTiles() const
    {
        if (params_.sparseCount <= 0 || params_.qBlockNum == 0) {
            return 0;
        }
        uint64_t hitTokens = static_cast<uint64_t>(params_.sparseCount) * params_.blockShapeX;
        const uint32_t tailQBlock = params_.qBlockNum - 1;
        const uint32_t tailRows = GetQBlockSize(tailQBlock);
        const int32_t lastQBlock =
            params_.rsvdBlockIdx.GetValue(params_.idxBase + static_cast<uint64_t>(params_.sparseCount - 1));
        if (static_cast<uint32_t>(lastQBlock) == tailQBlock) {
            hitTokens -= params_.blockShapeX - tailRows;
        }
        return static_cast<uint32_t>(hitTokens * tilesPerToken_);
    }

    /**
     * @brief 推进 tile 流游标一个 tile（G>128 时先推进 head 分片）。
     */
    __aicore__ inline void AdvanceWalk()
    {
        if (walkChunk_ + 1 < tilesPerToken_) {
            // G>128：同一 token 的下一个 128-head 分片，token 不变。
            ++walkChunk_;
            return;
        }
        walkChunk_ = 0;
        ++walkRowInBlock_;
        AdvanceToValidToken();
    }

    /**
     * @brief 将走查游标定位到下一个有效 token 并刷新 token_。
     *
     * 越界 block、越界 Q 行（尾 block 钳制）均跳过；游标耗尽时 token_ 保持
     * 最后一次有效值，由 Aggregate 的 filled 截断逻辑兜底。
     */
    __aicore__ inline void AdvanceToValidToken()
    {
        while (walkSlot_ < static_cast<uint32_t>(params_.sparseCount)) {
            const int32_t qBlockValue =
                params_.rsvdBlockIdx.GetValue(params_.idxBase + static_cast<uint64_t>(walkSlot_));
            if (qBlockValue < 0 || static_cast<uint32_t>(qBlockValue) >= params_.qBlockNum) {
                ++walkSlot_;
                walkRowInBlock_ = 0;
                continue;
            }
            const uint32_t qBlockRows = GetQBlockSize(static_cast<uint32_t>(qBlockValue));
            if (qBlockRows == 0 || walkRowInBlock_ >= qBlockRows) {
                ++walkSlot_;
                walkRowInBlock_ = 0;
                continue;
            }
            token_ = static_cast<uint32_t>(qBlockValue) * params_.blockShapeX + walkRowInBlock_;
            return;
        }
    }

    /**
     * @brief 计算指定 Q block 在实际序列范围内的有效行数。
     * @param qBlock Q block 编号。
     * @return 当前 Q block 的实际有效行数，起点越界时返回 0。
     */
    __aicore__ inline uint32_t GetQBlockSize(uint32_t qBlock) const
    {
        const uint64_t qBegin = static_cast<uint64_t>(qBlock) * params_.blockShapeX;
        if (qBegin >= params_.qSeqlen) {
            return 0;
        }
        const uint32_t remain = params_.qSeqlen - static_cast<uint32_t>(qBegin);
        return remain < params_.blockShapeX ? remain : params_.blockShapeX;
    }

    Params params_;           // 当前稀疏索引行的聚合输入。
    uint32_t tileRows_;       // 单 tile 行数（min(G, 128)）。
    uint32_t tilesPerPacket_; // 每包 tile 数（floor(128 / tileRows)）。
    uint32_t tilesPerToken_;  // 每 token tile 数（G>128 时为 ceil(G/128)）。
    uint32_t totalTiles_;     // 当前索引行估算的 tile 总数。
    uint32_t remainingTiles_; // 尚未分配到包的 tile 数。
    uint32_t tilePos_;        // 已发射的 tile 流位置（驱动 headStart0）。
    bool walkInit_;           // qStart 走查游标是否已初始化。
    uint32_t walkSlot_;       // 走查当前稀疏槽位。
    uint32_t walkRowInBlock_; // 走查当前 Q block 内 token 行。
    uint32_t walkChunk_;      // G>128 时 token 内 head 分片号。
    uint32_t token_;          // 走查当前 token 号。
};

/**
 * @brief 将原始布局中的 (token, head) 组聚合行 Gather 为连续 Packet。
 * @tparam InputType Q/dOut 的数据类型，例如 half 或 bfloat16_t。
 */
template <typename InputType>
class GBSAGGater {
public:
    using ArchTag = NpuArch::Arch::AtlasA2;

    /**
     * @brief 单次 Gather 的 GM Tensor 和布局参数。
     */
    struct Params {
        AscendC::GlobalTensor<InputType> src; // 原始布局的 Q 或 dOut Tensor。
        AscendC::GlobalTensor<InputType> dst; // 连续 Packet 的 qPack 或 doutPack Tensor。
        uint64_t qBatchBaseOffset;            // 当前 batch 的 Q 数据元素基址。
        uint32_t qHeadBegin;                  // 当前 GQA group 的首个 Q head 索引。
        uint32_t numHeads;                    // Q head 总数。
        uint32_t qSeqlen;                     // 当前 batch 的 Q 序列长度。
        uint32_t headDim;                     // 单个 head 的特征维度。
        uint32_t inputLayout;                 // 0 表示 TND，2 表示 BSND，其他值表示 BNSD。
        uint32_t tileRows;                    // 单 tile 行数（闭式表推导用）。

        __aicore__ inline Params() {}

        __aicore__ inline Params(AscendC::GlobalTensor<InputType> src_, AscendC::GlobalTensor<InputType> dst_,
                                 uint64_t qBatchBaseOffset_, uint32_t qHeadBegin_, uint32_t numHeads_,
                                 uint32_t qSeqlen_, uint32_t headDim_, uint32_t inputLayout_, uint32_t tileRows_)
            : src(src_),
              dst(dst_),
              qBatchBaseOffset(qBatchBaseOffset_),
              qHeadBegin(qHeadBegin_),
              numHeads(numHeads_),
              qSeqlen(qSeqlen_),
              headDim(headDim_),
              inputLayout(inputLayout_),
              tileRows(tileRows_)
        {}
    };

    /**
     * @brief 初始化 Gather 使用的 UB 视图。
     *
     * Q 和 dOut 按顺序执行 Gather，因此复用同一个 LocalTensor。
     */
    __aicore__ inline explicit GBSAGGater(NpuArch::Arch::Resource<ArchTag> &resource_)
        : resource(resource_)
    {
        localTensor = resource.ubBuf.template GetBufferByByte<InputType>(0);
    }

    /**
     * @brief 将当前 AIV 负责的 tile 行搬运到连续 Packet GM。
     * @param params 源、目的 Tensor 及布局参数。
     * @param packet 聚合器生成的精简 Packet 元数据。
     *
     * 整 token 切分下 tile 恒均匀：packStart = s * tileRows、segmentRows = tileRows、
     * headStart = headStart0，全部寄存器闭式计算，仅 qStart 需要数组读。
     */
    __aicore__ inline void Gather(Params const &params, QPacket const &packet)
    {
        const QPacketAivRange range = GetQPacketAivRange(packet.rows);
        if (range.rows == 0) {
            return;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        auto eventID1 = EVENT_ID6;
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
        // 闭式定位相交段区间（tile 恒均匀），只遍历本 AIV 拥有的段，
        // 免去对另一半段的空跑与逐段求交；首段从 range.begin 起（可能段内
        // 偏移），末段止于 range.end，中间段整段在范围内且 headOffset 恒为
        // headStart0，段边界增量推进。段号用移位求（tileRows 为 2 的幂时），
        // 避免 AIV 标量核软件除法吃掉省下的指令。
        // 闭式定位相交段区间（tile 恒均匀），只遍历本 AIV 拥有的段，
        // 免去对另一半段的空跑与逐段求交；首段从 range.begin 起（可能段内
        // 偏移），末段止于 range.end，中间段整段在范围内且 headOffset 恒为
        // headStart0，段边界增量推进。段号用移位求（tileRows 为 2 的幂时），
        // 避免 AIV 标量核软件除法吃掉省下的指令。
        const uint32_t tileRows = params.tileRows;
        const uint32_t tileShift = GetQPacketTileShift(tileRows);
        const uint32_t firstSeg = SegIndexByTileRows(range.begin, tileRows, tileShift);
        const uint32_t lastSeg = SegIndexByTileRows(range.end - 1, tileRows, tileShift);
        uint32_t segBegin = firstSeg * tileRows;
        uint32_t segEnd = segBegin + tileRows;
        uint32_t copyBegin = range.begin;
        for (uint32_t segment = firstSeg; segment <= lastSeg; ++segment) {
            const uint32_t copyEnd = segEnd < range.end ? segEnd : range.end;

            // 本段首行对应的 head 偏移与 token 号。
            const uint32_t headOffset = packet.headStart0 + copyBegin - segBegin;
            const uint32_t token = packet.qStart[segment];
            // localRowStart 是本段首行在本 AIV UB 中的相对行号。
            const uint32_t localRowStart = copyBegin - range.begin;
            const uint32_t copyHeads = copyEnd - copyBegin;
            const uint64_t srcOffset =
                GetQPacketRowOffset(params.qBatchBaseOffset, params.qHeadBegin + headOffset, token, params.numHeads,
                                    params.qSeqlen, params.headDim, params.inputLayout);
            const uint32_t localOffset = localRowStart * params.headDim;
            if (params.inputLayout == 1) {
                // BNSD：同一 token 的 head 行按 qSeqlen 等距分布，2D 搬运一次完成。
                const uint32_t srcGap =
                    static_cast<uint32_t>((params.qSeqlen - 1) * params.headDim * sizeof(InputType));
                AscendC::DataCopyPad(localTensor[localOffset], params.src[srcOffset],
                                     {static_cast<uint16_t>(copyHeads),
                                      static_cast<uint32_t>(params.headDim * sizeof(InputType)), srcGap, 0, 0},
                                     {false, 0, 0, 0});
            } else {
                // TND/BSND：同一 token 的 head 行连续，一次 1D 搬运完成。
                AscendC::DataCopy(localTensor[localOffset], params.src[srcOffset], copyHeads * params.headDim);
            }

            copyBegin = segEnd;
            segBegin = segEnd;
            segEnd += tileRows;
        }

        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);

        AscendC::DataCopy(params.dst[range.begin * params.headDim], localTensor, range.rows * params.headDim);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    NpuArch::Arch::Resource<ArchTag> &resource;  // 复用 Kernel 的架构资源和 UB 访问入口。
    AscendC::LocalTensor<InputType> localTensor; // 当前 AIV 最多 64 行的 Gather 临时缓冲。
};

/**
 * @brief 将连续 dQ Packet Scatter 回原始 Q 布局。
 * @tparam InputType Scatter 数据类型，K_OUT 当前实例化为 float。
 */
template <typename InputType>
class GBSAGScateer {
public:
    using ArchTag = NpuArch::Arch::AtlasA2;

    /**
     * @brief 单次 Scatter 的 GM Tensor 和布局参数。
     */
    struct Params {
        AscendC::GlobalTensor<InputType> dst;     // 原始布局的 dQ workspace。
        AscendC::GlobalTensor<InputType> srcPack; // Cube2 生成的连续 dQ Packet。
        uint64_t qBatchBaseOffset;                // 当前 batch 的 Q 数据元素基址。
        uint32_t qHeadBegin;                      // 当前 GQA group 的首个 Q head 索引。
        uint32_t numHeads;                        // Q head 总数。
        uint32_t qSeqlen;                         // 当前 batch 的 Q 序列长度。
        uint32_t headDim;                         // 单个 head 的特征维度。
        uint32_t inputLayout;                     // 0 表示 TND，2 表示 BSND，其他值表示 BNSD。
        uint32_t tileRows;                        // 单 tile 行数（闭式表推导用）。

        __aicore__ inline Params() {}

        __aicore__ inline Params(AscendC::GlobalTensor<InputType> dst_, AscendC::GlobalTensor<InputType> srcPack_,
                                 uint64_t qBatchBaseOffset_, uint32_t qHeadBegin_, uint32_t numHeads_,
                                 uint32_t qSeqlen_, uint32_t headDim_, uint32_t inputLayout_, uint32_t tileRows_)
            : dst(dst_),
              srcPack(srcPack_),
              qBatchBaseOffset(qBatchBaseOffset_),
              qHeadBegin(qHeadBegin_),
              numHeads(numHeads_),
              qSeqlen(qSeqlen_),
              headDim(headDim_),
              inputLayout(inputLayout_),
              tileRows(tileRows_)
        {}
    };

    /**
     * @brief 初始化 Scatter 使用的 UB 视图。
     */
    __aicore__ inline explicit GBSAGScateer(NpuArch::Arch::Resource<ArchTag> &resource_)
        : resource(resource_)
    {
        localTensor = resource.ubBuf.template GetBufferByByte<InputType>(0);
    }

    /**
     * @brief 将当前 AIV 负责的连续 dQ 行恢复到原始布局。
     * @param params 目的、源 Tensor 及布局参数。
     * @param packet 聚合器生成的精简 Packet 元数据。
     * @param enableAtomicAdd 是否使用 AtomicAdd 写回；K_OUT 当前传 true。
     *
     * 整 token 切分下 tile 恒均匀，地址映射与 Gather 完全同构。
     */
    __aicore__ inline void Scatter(Params const &params, QPacket const &packet, bool enableAtomicAdd)
    {
        const QPacketAivRange range = GetQPacketAivRange(packet.rows);
        if (range.rows == 0) {
            return;
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        auto eventID1 = EVENT_ID6;
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);

        AscendC::DataCopy(localTensor, params.srcPack[range.begin * params.headDim], range.rows * params.headDim);

        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);

        if (enableAtomicAdd) {
            AscendC::SetAtomicAdd<InputType>();
        }
        // 闭式定位相交段区间（同 Gather），只遍历本 AIV 拥有的段。
        const uint32_t tileRows = params.tileRows;
        const uint32_t tileShift = GetQPacketTileShift(tileRows);
        const uint32_t firstSeg = SegIndexByTileRows(range.begin, tileRows, tileShift);
        const uint32_t lastSeg = SegIndexByTileRows(range.end - 1, tileRows, tileShift);
        uint32_t segBegin = firstSeg * tileRows;
        uint32_t segEnd = segBegin + tileRows;
        uint32_t copyBegin = range.begin;
        for (uint32_t segment = firstSeg; segment <= lastSeg; ++segment) {
            const uint32_t copyEnd = segEnd < range.end ? segEnd : range.end;

            // 本段首行对应的 head 偏移与 token 号。
            const uint32_t headOffset = packet.headStart0 + copyBegin - segBegin;
            const uint32_t token = packet.qStart[segment];
            // localRowStart 是本段首行在本 AIV UB 中的相对行号。
            const uint32_t localRowStart = copyBegin - range.begin;
            const uint32_t copyHeads = copyEnd - copyBegin;
            const uint64_t dstOffset =
                GetQPacketRowOffset(params.qBatchBaseOffset, params.qHeadBegin + headOffset, token, params.numHeads,
                                    params.qSeqlen, params.headDim, params.inputLayout);
            const uint32_t localOffset = localRowStart * params.headDim;
            if (params.inputLayout == 1) {
                // BNSD：同一 token 的 head 行按 qSeqlen 等距分布，2D 原子搬运一次完成。
                // src(UB) 行连续 → srcStride=0；dst(GM) 行按 qSeqlen 跳转 → dstStride=dstGap。
                const uint32_t dstGap =
                    static_cast<uint32_t>((params.qSeqlen - 1) * params.headDim * sizeof(InputType));
                AscendC::DataCopyPad(params.dst[dstOffset], localTensor[localOffset],
                                     {static_cast<uint16_t>(copyHeads),
                                      static_cast<uint32_t>(params.headDim * sizeof(InputType)), 0, dstGap, 0});
            } else {
                // TND/BSND：同一 token 的 head 行连续，一次 1D 原子搬运完成。
                AscendC::DataCopy(params.dst[dstOffset], localTensor[localOffset], copyHeads * params.headDim);
            }

            copyBegin = segEnd;
            segBegin = segEnd;
            segEnd += tileRows;
        }
        if (enableAtomicAdd) {
            // 避免 AtomicAdd 状态泄漏到后续普通 GM 写操作。
            AscendC::SetAtomicNone();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    NpuArch::Arch::Resource<ArchTag> &resource;  // 复用 Kernel 的架构资源和 UB 访问入口。
    AscendC::LocalTensor<InputType> localTensor; // 当前 AIV 最多 64 行的 Scatter 临时缓冲。
};

} // namespace GBSAG
#endif // GBSAG_EPILOGUE_GBSAG_GATHER_AND_SCATTER_HPP
